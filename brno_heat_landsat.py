#!/usr/bin/env python3
"""Brno clear-sky surface heat, 2001-2025. Run with --offline to reuse caches."""
import argparse, hashlib, json, time, warnings
from pathlib import Path
import ee
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from scipy import stats
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'brno_heat_outputs'
CACHE, TABLE, RASTER, FIG, PDF = [OUT/x for x in ['cache','tables','rasters','figures','report']]
BOUNDARY = OUT/'boundary/brno_boundary.geojson'
EE_PROJECT = 'brno-urban-heat'
YEARS = np.arange(2001,2026)
LS_YEARS = list(range(2001,2006))+list(range(2021,2026))
NODATA, CRS = -9999., 'EPSG:32633'
MODIS_BANDS = ['LST_Day_1km','QC_Day','Day_view_time','LST_Night_1km','QC_Night','Night_view_time']

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def boundary():
    return gpd.read_file(BOUNDARY)

def get_aoi():
    if not BOUNDARY.exists():
        import osmnx as ox
        BOUNDARY.parent.mkdir(parents=True,exist_ok=True)
        ox.geocode_to_gdf('R438171',by_osmid=True).to_file(BOUNDARY,driver='GeoJSON')
    return ee.FeatureCollection(json.loads(BOUNDARY.read_text(encoding='utf-8'))).geometry()

def download(image,name,region,projection=None,scale=None,metadata=None):
    """Atomic download with graph-hash cache and sequential bounded retries."""
    path,side = CACHE/(name+'.tif'),CACHE/(name+'.json')
    params={'format':'GEO_TIFF','filePerBand':False,'region':region}
    if projection:
        params.update(crs=projection['crs'],crs_transform=projection['transform'])
    else:
        params.update(crs=CRS,crs_transform=[scale,0,0,0,-scale,0])
    digest=hashlib.sha256((image.serialize()+str(params)).encode()).hexdigest()
    if path.exists() and side.exists() and json.loads(side.read_text())['graph_sha256']==digest:
        with rasterio.open(path) as ds: ds.read(1)
        print('  cached',name,flush=True)
        return
    for attempt in range(4):
        try:
            print('  downloading',name,flush=True)
            url=image.unmask(NODATA).toFloat().getDownloadURL(params)
            response=requests.get(url,timeout=(30,300))
            response.raise_for_status()
            temp=path.with_suffix('.part.tif')
            temp.write_bytes(response.content)
            with rasterio.open(temp) as ds:
                assert ds.count>0
                ds.read(1)
            temp.replace(path)
            write_json(side,{'graph_sha256':digest,**(metadata or {})})
            return
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt*3)

def download_modis(aoi):
    collection=ee.ImageCollection('MODIS/061/MOD11A1')
    proj=ee.Image(collection.first()).select(0).projection().getInfo()
    for year in YEARS:
        c=collection.filterDate(f'{year}-06-01',f'{year}-09-01').sort('system:time_start')
        dates=c.aggregate_array('system:time_start').getInfo()
        download(c.select(MODIS_BANDS).toBands(),f'modis_daily_{year}',aoi.bounds(),projection=proj,
                 metadata={'dates':dates,'bands_per_day':MODIS_BANDS,'source':'MODIS/061/MOD11A1'})

def prep_landsat(image,band):
    good=(image.select('QA_PIXEL').bitwiseAnd(63).eq(0)
          .And(image.select('QA_RADSAT').eq(0))
          .And(image.select('ST_QA').multiply(.01).lte(3))
          .And(image.select('ST_CDIST').multiply(.01).gte(.5)))
    t=image.select(band).multiply(.00341802).add(149).subtract(273.15)
    return t.rename('LST').updateMask(good.And(t.gt(-20)).And(t.lt(70))).copyProperties(image,['system:time_start'])

def download_landsat(aoi):
    col=ee.ImageCollection([])
    for sensor,band in [('LT05','ST_B6'),('LE07','ST_B6'),('LC08','ST_B10'),('LC09','ST_B10')]:
        c=ee.ImageCollection(f'LANDSAT/{sensor}/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.eq('PROCESSING_LEVEL','L2SP'))
        col=col.merge(c.map(lambda x:prep_landsat(x,band)))
    for year in LS_YEARS:
        summer=col.filterDate(f'{year}-06-01',f'{year}-09-01')
        dates=ee.List(summer.aggregate_array('system:time_start')).map(lambda d:ee.Date(d).format('YYYY-MM-dd')).distinct().sort()
        daily=ee.ImageCollection.fromImages(dates.map(lambda d:summer.filterDate(ee.Date(d),ee.Date(d).advance(1,'day')).median()))
        annual=daily.median().rename('LST').addBands(daily.count().rename('count'))
        download(annual,f'landsat_annual_{year}',aoi.bounds(),scale=30,
                 metadata={'source':'Landsat C2 T1 L2SP','year':year,'uncertainty_K':3,'cloud_distance_km':.5,'independent_unit':'acquisition date'})

def download_quality(aoi):
    district_url='https://gis.brno.cz/ags1/rest/services/OMI/OMI_mc_web_brno/FeatureServer/0/query'
    r=requests.get(district_url,params={'where':'1=1','outFields':'kod,nazev,nazev_kratky','outSR':4326,'f':'geojson'},timeout=90)
    r.raise_for_status();j=r.json();assert len(j['features'])==29
    write_json(OUT/'boundary/brno_districts.geojson',j)
    write_json(OUT/'boundary/districts_source.json',{'url':district_url,'source':'Statutory City of Brno GIS','features':29,'retrieved':'2026-10-01'})
    col=ee.ImageCollection([])
    for sensor,band in [('LT05','ST_B6'),('LE07','ST_B6'),('LC08','ST_B10'),('LC09','ST_B10')]:
     def prep(i):
      t=i.select(band).multiply(.00341802).add(149).subtract(273.15)
      clear=i.select('QA_PIXEL').bitwiseAnd(63).eq(0).And(i.select('QA_RADSAT').eq(0))
      distant=i.select('ST_CDIST').multiply(.01).gte(.5)
      plausible=t.gt(-20).And(t.lt(70))
      q=i.select('ST_QA').multiply(.01)
      return (t.rename('retrieval').updateMask(plausible)
        .addBands(t.rename('clear').updateMask(plausible.And(clear).And(distant)))
        .addBands(t.rename('qa5').updateMask(plausible.And(clear).And(distant).And(q.lte(5))))
        .copyProperties(i,['system:time_start']))
     c=ee.ImageCollection(f'LANDSAT/{sensor}/C02/T1_L2').filterBounds(aoi).filter(ee.Filter.eq('PROCESSING_LEVEL','L2SP'))
     col=col.merge(c.map(prep))
    for year in LS_YEARS:
     summer=col.filterDate(f'{year}-06-01',f'{year}-09-01')
     dates=ee.List(summer.aggregate_array('system:time_start')).map(lambda d:ee.Date(d).format('YYYY-MM-dd')).distinct().sort()
     daily=ee.ImageCollection.fromImages(dates.map(lambda d:summer.filterDate(ee.Date(d),ee.Date(d).advance(1,'day')).median()))
     im=(daily.select('qa5').median().rename('qa5_LST')
     .addBands(daily.select('qa5').count().rename('qa5_count'))
     .addBands(daily.select('retrieval').count().rename('retrieval_count'))
     .addBands(daily.select('clear').count().rename('clear_count')))
     download(im,f'landsat_quality_{year}',aoi.bounds(),scale=30,metadata={'year':year,'bands':['qa5_LST','qa5_count','retrieval_count','clear_count'],'ST_QA_K':5,'cloud_distance_km':.5})
    
    

def read(path):
    with rasterio.open(path) as ds:
        a=ds.read().astype(float)
        a[(a==NODATA)|(~np.isfinite(a))]=np.nan
        profile=ds.profile
    if Path(path).stem.startswith(('modis_daily_','aqua_daily_','mod21_')):
        # EE GeoTIFF loses the MODIS reference-sphere parameter.
        # Native grid coordinates are retained; repair CRS metadata, do not warp.
        profile['crs']=rasterio.crs.CRS.from_string('+proj=sinu +R=6371007.181 +units=m +no_defs')
    return a,profile

def inside_mask(profile):
    return geometry_mask(boundary().to_crs(profile['crs']).geometry,(profile['height'],profile['width']),profile['transform'],invert=True)

def area_weights(profile):
    """Exact city/pixel overlap areas in the native equal-area MODIS grid."""
    from shapely.geometry import box
    city=boundary().to_crs(profile['crs']).geometry.union_all()
    tr=profile['transform']
    w=np.zeros((profile['height'],profile['width']))
    for r in range(w.shape[0]):
        for c in range(w.shape[1]):
            x0,y0=tr*(c,r); x1,y1=tr*(c+1,r+1)
            w[r,c]=city.intersection(box(x0,y1,x1,y0)).area
    return w

def weighted_median(v,w):
    ok=np.isfinite(v)&(w>0)
    if not ok.any(): return np.nan
    v,w=v[ok],w[ok]
    order=np.argsort(v)
    return float(v[order][np.searchsorted(np.cumsum(w[order]),w.sum()/2)])

def trend(years,values):
    ok=np.isfinite(values)
    x,y=np.asarray(years)[ok].astype(float),np.asarray(values)[ok]
    if len(y)<10: return None
    ols=stats.linregress(x,y)
    sen=stats.theilslopes(y,x,.95)
    crit=stats.t.ppf(.975,len(y)-2)
    residual=y-(ols.intercept+ols.slope*x)
    design=np.column_stack([np.ones(len(x)),x-x.mean()])
    scores=design*residual[:,None]
    meat=scores.T@scores
    for lag in range(1,3):
        cov=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/3)*(cov+cov.T)
    bread=np.linalg.inv(design.T@design)
    hac=np.sqrt((bread@meat@bread)[1,1]*len(x)/(len(x)-2))
    return {'n_years':len(y),'ols_C_decade':float(ols.slope*10),
            'ols_95CI':[float((ols.slope-crit*ols.stderr)*10),float((ols.slope+crit*ols.stderr)*10)],
            'ols_p':float(ols.pvalue),'ols_r2':float(ols.rvalue**2),
            'hac_lag2_95CI':[float((ols.slope-crit*hac)*10),float((ols.slope+crit*hac)*10)],
            'sen_C_decade':float(sen.slope*10),'sen_95CI':[float(sen.low_slope*10),float(sen.high_slope*10)],
            'residual_lag1_correlation':float(np.corrcoef(residual[:-1],residual[1:])[0,1]),
            'ols_intercept':float(ols.intercept)}

def recover_qa(q, raw_lst):
    # See MOD11 User Guide v6.1, section 3.3 (QC fill-value convention).
    return np.where(~np.isfinite(q)&np.isfinite(raw_lst)&(raw_lst>=7500)&(raw_lst<=65535),0,q)

def qa_good(q,error_flag=1):
    q=np.nan_to_num(q,nan=255).astype(np.uint16)
    return ((q&3)<=1)&(((q>>2)&3)==0)&(((q>>4)&3)<=1)&(((q>>6)&3)<=error_flag)

def raster_write(name,arr,profile):
    p=profile.copy()
    p.update(driver='GTiff',count=1,dtype='float32',nodata=NODATA,compress='deflate')
    with rasterio.open(RASTER/name,'w',**p) as ds:
        ds.write(np.where(np.isfinite(arr),arr,NODATA).astype('float32'),1)

def analyze_modis():
    # Coverage thresholds were chosen before examining slopes; no gap filling.
    configs={'primary':(1,1,20),'more_days':(1,2,20),'strict_months':(1,5,20),
             'strict_1K':(0,1,20),'relaxed_3K':(2,1,20),'no_qa_zero_recovery':(1,1,20),'fewer_days':(1,1,15)}
    arrays={k:{p:[] for p in ['day','night']} for k in configs}
    counts={p:[] for p in ['day','night']}
    times={p:[] for p in ['day','night']}
    qc_hist={p:{} for p in ['day','night']}
    for year in YEARS:
        a,profile=read(CACHE/f'modis_daily_{year}.tif')
        dates=pd.to_datetime(json.loads((CACHE/f'modis_daily_{year}.json').read_text())['dates'],unit='ms')
        assert len(a)==len(dates)*6
        a=a.reshape(len(dates),6,*a.shape[1:])
        city=inside_mask(profile)
        for phase,offset in [('day',0),('night',3)]:
            t=a[:,offset]*.02-273.15
            q_raw=a[:,offset+1]; view=a[:,offset+2]*.1
            # MOD11 guide: QC fill 0 is valid quality 0 when LST is valid.
            # GEE masks those zeros in QC_Night. Restore only QC, never LST.
            q=recover_qa(q_raw, a[:,offset])
            codes,freqs=np.unique(q[:,city][np.isfinite(q[:,city])].astype(int),return_counts=True)
            for code,freq in zip(codes,freqs):
                qc_hist[phase][str(code)]=qc_hist[phase].get(str(code),0)+int(freq)
            for key,(flag,min_month,min_total) in configs.items():
                good=qa_good(q_raw if key=='no_qa_zero_recovery' else q,flag)&(t>-20)&(t<(70 if phase=='day' else 50))
                values=np.where(good,t,np.nan)
                annual=monthly_summer(values, dates.month, min_month, min_total)
                n=np.isfinite(values).sum(axis=0)
                arrays[key][phase].append(annual)
                if key=='primary':
                    counts[phase].append(n)
                    times[phase].append(np.nanmedian(np.where(good,view,np.nan),axis=0))
    weights=area_weights(profile)
    rows=[{'year':int(y)} for y in YEARS]
    results={}
    for key in configs:
        results[key]={}
        for phase in ['day','night']:
            a=np.array(arrays[key][phase])
            common=np.all(np.isfinite(a),axis=0)&(weights>0)
            coverage=float(weights[common].sum()/weights.sum()*100)
            w=weights*common
            series=[weighted_median(v,w) for v in a]
            results[key][phase]={'fixed_area_coverage_pct':coverage,'fixed_grid_cells':int(common.sum()),'trend':trend(YEARS,series)}
            if key=='primary':
                if coverage<50: raise ValueError(f'MODIS {phase} fixed support only {coverage:.1f}%; review before report.')
                for i,row in enumerate(rows):
                    row[f'{phase}_LST_C']=series[i]
                    row[f'{phase}_valid_area_pct']=float(weights[np.isfinite(a[i])].sum()/weights.sum()*100)
                    row[f'{phase}_median_valid_days']=weighted_median(counts[phase][i],w)
                    row[f'{phase}_view_local_hour']=weighted_median(times[phase][i],w)
                results[key][phase]['trend_2001_2020']=trend(YEARS[:20],series[:20])
                results[key][phase]['trend_2002_2025']=trend(YEARS[1:],series[1:])
                results[key][phase]['trend_2002_2020']=trend(YEARS[1:20],series[1:20])
                results[key][phase]['variable_support_trend']=trend(YEARS,[weighted_median(v,weights) for v in a])
                results[key][phase]['early_mean_C']=float(np.mean(series[:5]))
                results[key][phase]['recent_mean_C']=float(np.mean(series[-5:]))
                results[key][phase]['first_last_5yr_difference_C']=float(np.mean(series[-5:])-np.mean(series[:5]))
                centered=YEARS-YEARS.mean()
                slope=np.sum(centered[:,None,None]*a,axis=0)/np.sum(centered**2)*10
                slope[~common]=np.nan
                raster_write(f'0{4 if phase=="day" else 5}_modis_{phase}_trend_C_per_decade.tif',slope,profile)
                raster_write(f'modis_{phase}_valid_years.tif',np.where(weights>0,np.isfinite(a).sum(axis=0),np.nan),profile)
                raster_write(f'modis_{phase}_fixed_support.tif',np.where(weights>0,common.astype(float),np.nan),profile)
    for phase in ['day','night']:
        base=np.array(arrays['primary'][phase])
        for key in configs:
            variant=np.array(arrays[key][phase])
            common=np.all(np.isfinite(variant),axis=0)&np.all(np.isfinite(base),axis=0)&(weights>0)
            results[key][phase]['primary_on_same_support']=trend(YEARS,[weighted_median(v,weights*common) for v in base])
    df=pd.DataFrame(rows)
    df.to_csv(TABLE/'modis_annual_citywide_LST.csv',index=False)
    write_json(TABLE/'modis_qa_histogram.json',qc_hist)
    write_json(TABLE/'modis_sensitivity.json',results)
    return df,results

def analyze_modis_districts():
    from shapely.geometry import box
    rows=[]
    for phase in ['day','night']:
     a,p=read(RASTER/f'modis_{phase}_fixed_support.tif');support=a[0];tr=p['transform']
     d=gpd.read_file(OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
     city=boundary().to_crs(p['crs']).geometry.union_all()
     cells=[]
     for r,c in zip(*np.where(support==1)):
      x0,y0=tr*(c,r);x1,y1=tr*(c+1,r+1);cells.append(box(x0,y1,x1,y0))
     for _,row in d.iterrows():
      geom=row.geometry.intersection(city)
      area=sum(geom.intersection(cell).area for cell in cells)
      rows.append({'phase':phase,'district':row['nazev'],'fixed_area_coverage_pct':100*area/geom.area})
    df=pd.DataFrame(rows);df.to_csv(TABLE/'modis_district_coverage.csv',index=False,encoding='utf-8-sig')
    return df

def analyze_landsat():
    values,counts=[],[]
    for year in LS_YEARS:
        a,p=read(CACHE/f'landsat_annual_{year}.tif')
        values.append(a[0]); counts.append(np.nan_to_num(a[1]))
    values,counts=np.array(values),np.array(counts)
    inside=inside_mask(p); values[:,~inside]=np.nan
    # Recent heat map does not inherit historical data gaps.
    valid_recent=np.where(counts[5:]>=2,values[5:],np.nan)
    recent_years=np.isfinite(valid_recent).sum(axis=0)
    current=np.nanmedian(valid_recent,axis=0)
    current[(recent_years<3)|(~inside)]=np.nan
    current_ref=float(np.nanmedian(current))
    current_anomaly=current-current_ref
    summaries={'current':{'coverage_pct':float(np.isfinite(current).sum()/inside.sum()*100),
                          'city_median_C':current_ref,
                          'hotspot_p90_C':float(np.nanpercentile(current_anomaly,90))}}
    raster_write('01_landsat_recent_LST_C.tif',current,p)
    raster_write('02_landsat_recent_anomaly_C.tif',current_anomaly,p)
    primary_change=None
    for min_obs,min_years,min_total in [(1,3,5),(1,3,3),(2,3,5),(1,4,5)]:
        valid=np.where(counts>=min_obs,values,np.nan)
        n0=np.isfinite(valid[:5]).sum(axis=0); n1=np.isfinite(valid[5:]).sum(axis=0)
        common=(n0>=min_years)&(n1>=min_years)&inside
        common&=(counts[:5].sum(axis=0)>=min_total)&(counts[5:].sum(axis=0)>=min_total)
        early=np.nanmedian(valid[:5],axis=0); recent=np.nanmedian(valid[5:],axis=0)
        early[~common]=np.nan; recent[~common]=np.nan
        e0,e1=float(np.nanmedian(early)),float(np.nanmedian(recent))
        change=(recent-e1)-(early-e0)
        key=f'{min_obs}_dates_{min_years}_years_{min_total}_total'
        summaries[key]={'area_coverage_pct':float(common.sum()/inside.sum()*100),
            'early_city_median_C':e0,'recent_city_median_C':e1,
            'change_p10_C':float(np.nanpercentile(change,10)),'change_p90_C':float(np.nanpercentile(change,90))}
        if primary_change is None:
            primary_change=change.copy()
            raster_write('03_landsat_relative_heat_change_C.tif',change,p)
            raster_write('landsat_early_valid_years.tif',np.where(inside,n0,np.nan),p)
            raster_write('landsat_recent_valid_years.tif',np.where(inside,n1,np.nan),p)
            raster_write('landsat_early_total_dates.tif',np.where(inside,counts[:5].sum(axis=0),np.nan),p)
        else:
            shared=np.isfinite(change)&np.isfinite(primary_change)
            summaries[key]['change_correlation_with_primary']=float(np.corrcoef(change[shared],primary_change[shared])[0,1])
            strong=shared&(np.abs(primary_change)>=2)
            summaries[key]['sign_agreement_where_primary_abs_ge_2C_pct']=float(np.mean(np.sign(change[strong])==np.sign(primary_change[strong]))*100)
    annual=[]
    for i,year in enumerate(LS_YEARS):
        annual.append({'year':year,'median_clear_dates':float(np.median(counts[i,inside])),
                       'pct_area_2plus_dates':float(((counts[i]>=2)&inside).sum()/inside.sum()*100)})
    pd.DataFrame(annual).to_csv(TABLE/'landsat_annual_coverage.csv',index=False)
    write_json(TABLE/'landsat_sensitivity.json',summaries)
    if summaries['1_dates_3_years_5_total']['area_coverage_pct']<50:
        raise ValueError('Landsat common support <50%; review.')
    return summaries

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--analysis-only',action='store_true')
    args=parser.parse_args()
    for folder in [CACHE,TABLE,RASTER,FIG,PDF]: folder.mkdir(parents=True,exist_ok=True)
    if not args.offline:
        ee.Initialize(project=EE_PROJECT)
        aoi=get_aoi()
        download_modis(aoi)
        download_landsat(aoi)
        download_quality(aoi)
        from diagnostics.download_alternatives import download_aqua,download_mod21
        download_aqua(range(2003,2026))
        download_mod21([2001,2025])
    if args.download_only: return
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        df,modis=analyze_modis()
        analyze_modis_districts()
        landsat=analyze_landsat()
        landsat["quality_audit"]=analyze_quality()
        import brno_heat_trends as balanced
        advanced=balanced.analyze()
    summary={'period':'2001-2025','months':[6,7,8],'modis':modis,'landsat':landsat,
             'balanced_trends':advanced,
             'meaning':'Clear-sky land-surface temperature; not air temperature or causal climate attribution.'}
    write_json(TABLE/'results_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if not args.analysis_only:
        make_figures(df,summary)
        import brno_heat_report as report
        report.create(summary)
    write_manifest()

# Additional verification helpers used by the numerical regression tests.
def monthly_summer(values, months, min_month=5, min_total=20):
    """Equal-month mean of daily medians; do not substitute a missing month."""
    medians=[]
    for month in [6,7,8]:
        v=values[months==month]
        m=np.nanmedian(v,axis=0)
        m=np.where(np.isfinite(v).sum(axis=0)>=min_month,m,np.nan)
        medians.append(m)
    result=np.mean(medians,axis=0)
    return np.where(np.isfinite(values).sum(axis=0)>=min_total,result,np.nan)

def comparison_mask(counts,inside,min_years=3,min_total=5):
    return inside & ((counts[:5]>0).sum(0)>=min_years) & ((counts[5:]>0).sum(0)>=min_years) & (counts[:5].sum(0)>=min_total) & (counts[5:].sum(0)>=min_total)


def change_classes(strict,broad,inside,threshold=2):
    """Exhaustive evidence classes; no interpolation and no significance claim.
    0 insufficient; 1 broad-only; 2 weak/sensitive; 3 cooler; 4 warmer.
    """
    classes=np.full(inside.shape,np.nan)
    classes[inside]=0
    valid=inside & np.isfinite(broad)
    classes[valid]=1
    shared=valid & np.isfinite(strict)
    classes[shared]=2
    classes[shared & (strict<=-threshold) & (broad<=-threshold)]=3
    classes[shared & (strict>=threshold) & (broad>=threshold)]=4
    return classes


def analyze_quality():
    values=[]; counts=[]; retrieval=[]; clear=[]
    for year in LS_YEARS:
        a,p=read(CACHE/f'landsat_quality_{year}.tif')
        values.append(a[0]); counts.append(np.nan_to_num(a[1]))
        retrieval.append(np.nan_to_num(a[2])); clear.append(np.nan_to_num(a[3]))
    values,counts,retrieval,clear=map(np.array,[values,counts,retrieval,clear])
    inside=inside_mask(p)
    broad_mask=comparison_mask(counts,inside)
    strict=read(RASTER/'03_landsat_relative_heat_change_C.tif')[0][0]
    shared=broad_mask & np.isfinite(strict)
    # Use exactly the SAME reference pixels for both filters, not different city subsets.
    early=np.nanmedian(values[:5],axis=0); recent=np.nanmedian(values[5:],axis=0)
    ref0=float(np.nanmedian(early[shared])); ref1=float(np.nanmedian(recent[shared]))
    broad=(recent-ref1)-(early-ref0)
    broad[~broad_mask]=np.nan
    classes=change_classes(strict,broad,inside)
    names={0:'Insufficient observations',1:'Broader filter only',2:'Small or filter-sensitive',3:'Relative decrease >=2 C under both filters',4:'Relative increase >=2 C under both filters'}
    pct=lambda mask:float(np.count_nonzero(mask)/inside.sum()*100)
    # Sequential exclusion attribution: first unmet requirement, not a causal diagnosis.
    available=comparison_mask(retrieval,inside)
    cloud_ok=comparison_mask(clear,inside)
    reason=np.full(inside.shape,np.nan)
    reason[inside]=0
    reason[inside & ~available]=1
    reason[available & ~cloud_ok]=2
    reason[cloud_ok & ~broad_mask]=3
    reason[broad_mask & ~np.isfinite(strict)]=4
    result={'broad_QA_K':5,'strict_QA_K':3,'min_years':3,'min_total_dates':5,
            'broad_coverage_pct':pct(broad_mask),'shared_reference_coverage_pct':pct(shared),
            'broad_early_reference_C':ref0,'broad_recent_reference_C':ref1,
            'shared_filter_correlation':float(np.corrcoef(strict[shared],broad[shared])[0,1]),
            'median_abs_filter_difference_C':float(np.median(np.abs(strict[shared]-broad[shared]))),
            'classes':{str(k):{'label':v,'area_pct':pct(classes==k)} for k,v in names.items()},
            'sequential_exclusions':{str(k):{'label':v,'area_pct':pct(reason==k)} for k,v in {
                0:'Passes strict comparison',1:'Too few retrieval dates or years',2:'Cloud/shadow/saturation/distance screen',
                3:'Additional ST_QA <=5 K screen',4:'Additional ST_QA <=3 K screen'}.items()},
            'interpretation':'Filter agreement is a sensitivity check, not statistical significance or validation.'}
    raster_write('06_landsat_broad_relative_change_C.tif',broad,p)
    raster_write('07_landsat_change_evidence_class.tif',classes,p)
    raster_write('08_landsat_missing_reason_class.tif',reason,p)
    raster_write('landsat_broad_early_valid_years.tif',np.where(inside,(counts[:5]>0).sum(0),np.nan),p)
    raster_write('landsat_broad_early_total_dates.tif',np.where(inside,counts[:5].sum(0),np.nan),p)
    current=read(RASTER/'02_landsat_recent_anomaly_C.tif')[0][0]
    threshold=float(np.nanpercentile(current,90))
    districts=gpd.read_file(OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
    rows=[]
    for _,row in districts.iterrows():
        mask=geometry_mask([row.geometry],inside.shape,p['transform'],invert=True)&inside
        observed=mask & np.isfinite(current)
        if not observed.any(): continue
        denominator=observed.sum()
        rows.append({'district':row['nazev'],'city_grid_area_km2':float(mask.sum()*.0009),
                     'recent_coverage_pct':float(100*denominator/mask.sum()),
                     'recent_median_anomaly_C':float(np.nanmedian(current[observed])),
                     'hotspot_threshold_anomaly_C':threshold,
                     'hotspot_share_of_observed_pct':float(100*np.count_nonzero(observed & (current>=threshold))/denominator),
                     'strict_change_coverage_pct':float(100*np.count_nonzero(mask & np.isfinite(strict))/mask.sum()),
                     'broad_change_coverage_pct':float(100*np.count_nonzero(mask & broad_mask)/mask.sum()),
                     'increase_both_filters_pct':float(100*np.count_nonzero(mask & (classes==4))/mask.sum()),
                     'decrease_both_filters_pct':float(100*np.count_nonzero(mask & (classes==3))/mask.sum())})
    district_table=pd.DataFrame(rows).sort_values('hotspot_share_of_observed_pct',ascending=False)
    district_table.to_csv(TABLE/'district_heat_screening.csv',index=False,encoding='utf-8-sig')
    result['district_priority_screen']=district_table.head(5).to_dict(orient='records')
    write_json(TABLE/'landsat_quality_audit.json',result)
    return result


def district_context(ax,p,labels=True):
    import matplotlib.patheffects as pe
    districts=gpd.read_file(OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
    districts.boundary.plot(ax=ax,color='#374753',linewidth=.27,alpha=.65)
    if labels:
        selected={'Brno-Bystrc':'Bystrc','Brno-Královo Pole':'Královo\nPole',
                  'Brno-střed':'střed','Brno-Černovice':'Černovice','Brno-Líšeň':'Líšeň','Brno-Tuřany':'Tuřany','Brno-jih':'jih'}
        for _,row in districts.iterrows():
            if row['nazev'] not in selected: continue
            pt=row.geometry.representative_point()
            ax.text(pt.x,pt.y,selected[row['nazev']],fontsize=9,ha='center',va='center',color='#152f3b',
                    path_effects=[pe.withStroke(linewidth=2,foreground='white')])


def map_base(ax,p,title):
    b=boundary().to_crs(p['crs'])
    district_context(ax,p)
    b.boundary.plot(ax=ax,color='#263b45',linewidth=.75)
    ax.set_facecolor('white')
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=9)
    xmin,ymin,xmax,ymax=b.total_bounds
    ax.set_xlim(xmin-650,xmax+650);ax.set_ylim(ymin-650,ymax+650)
    sx,sy=xmin+650,ymin+600
    ax.plot([sx,sx+5000],[sy,sy],color='#172c39',lw=2)
    ax.text(sx+2500,sy+350,'5 km',ha='center',fontsize=9)
    ax.annotate('N',xy=(.94,.95),xytext=(.94,.84),xycoords='axes fraction',
                ha='center',fontsize=10,arrowprops={'arrowstyle':'-|>','color':'#172c39'})
    for spine in ax.spines.values(): spine.set_visible(False)


def plot_panel(ax,filename,title,vmin,vmax,cmap,label,colorbar=True):
    from matplotlib.colors import ListedColormap
    a,p=read(RASTER/filename);arr=a[0];inside=inside_mask(p)
    left,bottom,right,top=array_bounds(p['height'],p['width'],p['transform'])
    extent=[left,right,bottom,top]
    city=boundary().to_crs(p['crs'])
    city.plot(ax=ax,color='#515b66',edgecolor='none',zorder=-1)
    kwargs={'norm':TwoSlopeNorm(vcenter=0,vmin=vmin,vmax=vmax)} if vmin<0<vmax else {'vmin':vmin,'vmax':vmax}
    im=ax.imshow(arr,extent=extent,cmap=cmap,interpolation='nearest',**kwargs)
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    geom=city.geometry.union_all()
    polys=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
    paths=[PlotPath(np.asarray(r.coords)) for poly in polys for r in [poly.exterior,*poly.interiors]]
    im.set_clip_path(PathPatch(PlotPath.make_compound_path(*paths),transform=ax.transData))
    map_base(ax,p,title)
    if colorbar:
        cb=ax.figure.colorbar(im,ax=ax,orientation='horizontal',pad=.04,shrink=.94,aspect=28,extend='both')
        cb.set_label(label,fontsize=10);cb.ax.tick_params(labelsize=9)
    return im


def evidence_panel(ax,title):
    from matplotlib.colors import ListedColormap,BoundaryNorm
    a,p=read(RASTER/'07_landsat_change_evidence_class.tif')
    bounds=array_bounds(p['height'],p['width'],p['transform'])
    colors=['#515b66','#bab7d5','#ede7bc','#3276a1','#c9472e']
    ax.imshow(a[0],extent=[bounds[0],bounds[2],bounds[1],bounds[3]],
              cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,5.5),5),interpolation='nearest')
    map_base(ax,p,title)
    return colors


def make_spatial_figures(s):
    from matplotlib.patches import Patch
    audit=s['landsat']['quality_audit']
    fig,axes=plt.subplots(1,2,figsize=(9,5.7))
    im=plot_panel(axes[0],'02_landsat_recent_anomaly_C.tif','A  Recent surface heat',-8,8,'RdYlBu_r','',False)
    colors=evidence_panel(axes[1],'B  Historical change evidence')
    fig.subplots_adjust(left=.018,right=.986,top=.925,bottom=.245,wspace=.055)
    cbax=fig.add_axes([.05,.16,.41,.019])
    cb=fig.colorbar(im,cax=cbax,orientation='horizontal',extend='both',ticks=[-8,-4,0,4,8])
    cb.ax.tick_params(labelsize=10,pad=2)
    cb.set_label('2021-2025: difference from city median (°C)',fontsize=10,labelpad=3)
    axes[0].legend(handles=[Patch(facecolor='#515b66',label=f'Insufficient data: {100-s["landsat"]["current"]["coverage_pct"]:.1f}% of Brno')],
                   loc='upper center',bbox_to_anchor=(.5,-.25),fontsize=10,frameon=False)
    labels={4:'Increase ≥2°C under both filters',3:'Decrease ≥2°C under both filters',
            2:'Small or filter-sensitive',1:'Broader filter only',0:'Insufficient observations'}
    handles=[Patch(facecolor=colors[k],label=f'{labels[k]}  {audit["classes"][str(k)]["area_pct"]:.1f}%') for k in [4,3,2,1,0]]
    axes[1].legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.025),fontsize=9.8,
                   frameon=False,handlelength=1.2,handleheight=.8,labelspacing=.38,borderaxespad=0)
    fig.savefig(FIG/'report_main_maps.png',dpi=300,facecolor='white');plt.close(fig)
    specs=[('01_landsat_recent_LST_C','Recent clear-sky summer surface temperature',24,44,'inferno','Surface temperature (°C)'),
           ('02_landsat_recent_anomaly_C','Recent surface heat relative to Brno',-8,8,'RdYlBu_r','Anomaly (°C)'),
           ('03_landsat_relative_heat_change_C','Strict-filter relative surface heat change',-4,4,'RdYlBu_r','Relative change (°C)'),
           ('06_landsat_broad_relative_change_C','Broader-filter change: exploratory only',-4,4,'RdYlBu_r','Relative change (°C)')]
    aliases=['map_recent_LST.png','map_recent_anomaly.png','map_relative_heat_change.png',None]
    for (name,title,lo,hi,cm,label),alias in zip(specs,aliases):
        fig,ax=plt.subplots(figsize=(7,7.6))
        plot_panel(ax,name+'.tif',title,lo,hi,cm,label)
        fig.text(.5,.022,'Dark grey: insufficient observations | District boundaries: City of Brno GIS',ha='center',fontsize=9)
        fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight')
        if alias:fig.savefig(FIG/alias,dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,8))
    evidence_panel(ax,'Historical change: agreement across quality filters')
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.01),frameon=False,fontsize=10)
    fig.subplots_adjust(bottom=.21,top=.94)
    fig.savefig(FIG/'07_landsat_change_evidence.png',dpi=220);plt.close(fig)
    # A separate, exhaustive exclusion map explains every rejected pixel.
    from matplotlib.colors import ListedColormap,BoundaryNorm
    a,p=read(RASTER/'08_landsat_missing_reason_class.tif')
    bounds=array_bounds(p['height'],p['width'],p['transform'])
    rc=['#55a69a','#303e4d','#dda451','#904ba0','#9baad0']
    fig,ax=plt.subplots(figsize=(7,8))
    ax.imshow(a[0],extent=[bounds[0],bounds[2],bounds[1],bounds[3]],cmap=ListedColormap(rc),norm=BoundaryNorm(np.arange(-.5,5.5),5),interpolation='nearest')
    map_base(ax,p,'Why observations were excluded')
    ax.legend(handles=[Patch(facecolor=rc[k],label=f'{v["label"]}: {v["area_pct"]:.2f}%') for k,v in [(int(k),v) for k,v in audit['sequential_exclusions'].items()]],
              loc='upper center',bbox_to_anchor=(.5,-.01),frameon=False,fontsize=10)
    fig.subplots_adjust(bottom=.21,top=.94)
    fig.savefig(FIG/'08_landsat_missing_reason.png',dpi=220);plt.close(fig)


def make_figures(df,s):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#233c48'})
    make_spatial_figures(s)
    # Separate axes retain nighttime variability instead of compressing it into the day scale.
    fig,axes=plt.subplots(2,1,figsize=(10,4.4),sharex=True)
    for ax,phase,color,label in [(axes[0],'day','#bd5a36','Daytime'),(axes[1],'night','#35778b','Night-time')]:
        y=df[phase+'_LST_C'].to_numpy()
        tr=s['modis']['primary'][phase]['trend']
        ax.axvspan(2020.5,2025.5,color='#e9edf0',zorder=0)
        ax.plot(YEARS,y,'o-',color=color,lw=1.3,ms=3)
        ax.plot(YEARS,tr['ols_intercept']+YEARS*tr['ols_C_decade']/10,color=color,ls='--',lw=1.5)
        ax.text(.012,.92,label,transform=ax.transAxes,ha='left',va='top',weight='bold',fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
        ax.set_ylabel('LST (°C)',fontsize=8); ax.grid(axis='y',alpha=.16)
        ax.set_xlim(2000.5,2025.5)
    axes[0].text(.985,.92,'Orbit-drift era',transform=axes[0].transAxes,ha='right',va='top',fontsize=8,color='#526874')
    axes[1].set_xticks([2001,2005,2010,2015,2020,2025])
    fig.subplots_adjust(left=.07,right=.99,bottom=.11,top=.98,hspace=.12)
    fig.savefig(FIG/'modis_citywide_timeseries.png',dpi=260); plt.close(fig)
    # Coverage and local observation-time diagnostics are delivered separately.
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for col,phase in enumerate(['day','night']):
        axes[0,col].plot(YEARS,df[phase+'_median_valid_days'],'o-',ms=3)
        axes[0,col].set_title(phase.capitalize()); axes[0,col].set_ylabel('Valid days / summer')
        axes[1,col].plot(YEARS,df[phase+'_view_local_hour'],'o-',ms=3)
        axes[1,col].set_ylabel('Median local observation hour')
    fig.tight_layout()
    fig.savefig(FIG/'modis_sampling_diagnostics.png',dpi=200); plt.close(fig)
    # Diagnostic spatial trend maps retain native MODIS grid: no apparent 30 m detail.
    for phase in ['day','night']:
        a,p=read(RASTER/f'0{4 if phase=="day" else 5}_modis_{phase}_trend_C_per_decade.tif')
        fig,ax=plt.subplots(figsize=(6,6))
        bounds=array_bounds(p['height'],p['width'],p['transform'])
        boundary().to_crs(p['crs']).plot(ax=ax,color='#515b66',edgecolor='none',zorder=-1)
        im=ax.imshow(a[0],extent=[bounds[0],bounds[2],bounds[1],bounds[3]],
                     cmap='RdYlBu_r',vmin=-1.5,vmax=1.5,interpolation='nearest')
        city=boundary().to_crs(p['crs'])
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import PathPatch
        geom=city.geometry.union_all()
        polys=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
        paths=[PlotPath(np.asarray(r.coords)) for poly in polys for r in [poly.exterior,*poly.interiors]]
        im.set_clip_path(PathPatch(PlotPath.make_compound_path(*paths),transform=ax.transData))
        city.boundary.plot(ax=ax,color='black',lw=.7,zorder=3)
        xmin,ymin,xmax,ymax=city.total_bounds
        ax.set_xlim(xmin-700,xmax+700);ax.set_ylim(ymin-700,ymax+700)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(f'MODIS {phase}: descriptive OLS trend\n2001-2025; no pixel significance claim',fontsize=10)
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im,ax=ax,shrink=.8,label='°C / decade',extend='both')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='#515b66',label='Insufficient 25-year coverage')],
                  loc='upper center',bbox_to_anchor=(.5,-.02),frameon=False,fontsize=8)
        fig.subplots_adjust(bottom=.13)
        fig.savefig(FIG/f'diagnostic_modis_{phase}_complete_cases.png',dpi=180,bbox_inches='tight'); plt.close(fig)




def write_manifest():
    import importlib.metadata as metadata
    from datetime import datetime, timezone
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    inputs=sorted((OUT/'boundary').glob('*'))+sorted(CACHE.glob('*.tif'))+sorted(CACHE.glob('*.json'))
    products=sorted(RASTER.glob('*.tif'))+sorted(p for p in FIG.iterdir() if p.suffix.lower() in ['.png','.pdf','.svg'])
    products += sorted(p for p in TABLE.glob('*') if p.is_file() and p.name!='run_manifest.json')
    if (PDF/'brno_heat_report.pdf').exists(): products.append(PDF/'brno_heat_report.pdf')
    versions={name:metadata.version(name) for name in
              ['earthengine-api','geopandas','rasterio','numpy','pandas','matplotlib','scipy','requests','reportlab','pypdf']}
    write_json(TABLE/'run_manifest.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':digest(Path(__file__)),
        'code_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),ROOT/'brno_heat_trends.py',ROOT/'brno_heat_report.py',ROOT/'brno_heat_history.py',ROOT/'README.md',ROOT/'requirements.txt']+sorted((ROOT/'tests').glob('*.py'))+sorted((ROOT/'diagnostics').glob('*.py'))},
        'packages':versions,'period':[2001,2025],'months':[6,7,8],
        'sources':['MODIS/061/MOD11A1','MODIS/061/MYD11A1','MODIS/061/MOD21A1D','MODIS/061/MOD21A1N','LANDSAT/LT05/C02/T1_L2','LANDSAT/LE07/C02/T1_L2',
                   'LANDSAT/LC08/C02/T1_L2','LANDSAT/LC09/C02/T1_L2','OSM relation 438171','City of Brno GIS municipal districts'],
        'balanced_trends':{'min_fraction_years':.8,'min_first_last_5_years':3,'min_each_5_temporal_blocks':2,'gap_filling':False,'report_CI':'joint year-score HAC lag2; t df=n_years-2','bootstrap':'joint circular 3-year pairs; 2500 draws; seed419'},
        'primary_modis':{'LST_error_flag_max':1,'emissivity_error_flag_max':1,
                         'min_days':20,'min_days_each_month':1,'QA_zero_recovery_when_LST_valid':True},
        'primary_landsat_change':{'min_years_per_period':3,'min_dates_per_period':5,
                                  'max_ST_QA_K':3,'min_cloud_distance_km':.5},
        'quality_audit':{'broad_ST_QA_K':5,'min_years_per_period':3,'min_dates_per_period':5,'same_reference_footprint':True,'classification_threshold_C':2,'gap_filling':False},
        'input_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
        'output_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in products},
        'reference_sphere_radius_m':6371007.181,
        'interpretation':'Clear-sky LST on stated observed footprints; not climate-only attribution.'
    })


if __name__=='__main__':
    main()

