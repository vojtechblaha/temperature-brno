"""Spatially complete trend assessment using observed, temporally balanced years.
No temperature imputation. Calendar-year blocks are resampled jointly over pixels.
"""
import json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import rasterio
from shapely.geometry import box
import brno_heat_landsat as b

SOURCES={'terra':('modis',np.arange(2001,2026)),'aqua':('aqua',np.arange(2003,2026))}
CONFIGS={'primary':(1,20,1),'days15':(1,15,1),'months2':(1,20,2),'qa3K':(2,20,1)}
SPHERE='+proj=sinu +R=6371007.181 +units=m +no_defs'

def geometry_weights(profile,geometry):
    weights=np.zeros((profile['height'],profile['width']))
    tr=profile['transform']
    for r in range(weights.shape[0]):
        for c in range(weights.shape[1]):
            x0,y0=tr*(c,r);x1,y1=tr*(c+1,r+1)
            weights[r,c]=geometry.intersection(box(x0,y1,x1,y0)).area
    return weights

def eligible_years(values,fraction=.8):
    valid=np.isfinite(values);n=len(values)
    keep=valid.sum(0)>=math.ceil(fraction*n)
    keep&=(valid[:5].sum(0)>=3)&(valid[-5:].sum(0)>=3)
    for group in np.array_split(np.arange(n),5):keep&=valid[group].sum(0)>=2
    return keep

def local_slopes(years,values):
    x=np.asarray(years,dtype=float)-np.mean(years)
    shape=(len(x),)+(1,)*(values.ndim-1)
    x=x.reshape(shape);ok=np.isfinite(values);n=ok.sum(0)
    y=np.where(ok,values,0.)
    sx=np.sum(np.where(ok,x,0),axis=0);sy=y.sum(0)
    sxx=np.sum(np.where(ok,x*x,0),axis=0);sxy=(x*y).sum(0)
    denom=sxx-sx*sx/np.maximum(n,1)
    return np.divide(sxy-sx*sy/np.maximum(n,1),denom,out=np.full_like(sy,np.nan),where=(n>=3)&(denom>0))*10

def weighted_mean(values,weights):
    ok=np.isfinite(values)&(weights>0)
    return float(np.sum(values[ok]*weights[ok])/weights[ok].sum()) if ok.any() else None

def bootstrap_trends(years,values,weights,replicates=2500,block=3,seed=419):
    """Pairs bootstrap of circular calendar-year blocks; same draws for all pixels.
    Fixed eligible footprint. Pixels are never treated as independent replicates.
    """
    x=np.asarray(years,dtype=float);x=x-x.mean();n=len(x)
    arr=values.reshape(n,-1);w=weights.reshape(-1);take=w>0
    arr=arr[:,take];w=w[take]
    if not len(w):return None
    valid=np.isfinite(arr).astype(float);y=np.nan_to_num(arr)
    moments=[valid,valid*x[:,None],y,valid*(x*x)[:,None],y*x[:,None]]
    rng=np.random.default_rng(seed);results=[]
    for start in range(0,replicates,128):
        size=min(128,replicates-start)
        starts=rng.integers(0,n,size=(size,math.ceil(n/block)))
        draws=((starts[:,:,None]+np.arange(block))%n).reshape(size,-1)[:,:n]
        counts=np.zeros((size,n))
        np.add.at(counts,(np.arange(size)[:,None],draws),1)
        nn,sx,sy,sxx,sxy=[counts@m for m in moments]
        den=sxx-sx*sx/np.maximum(nn,1)
        slopes=np.divide(sxy-sx*sy/np.maximum(nn,1),den,out=np.full_like(sy,np.nan),where=(nn>=3)&(den>0))*10
        good=np.isfinite(slopes)
        coverage=good@w/w.sum()
        sample=np.sum(np.nan_to_num(slopes)*w,axis=1)/np.sum(good*w,axis=1)
        results.extend(sample[coverage>.999].tolist())
    assert len(results)>=replicates*.99,'Too many bootstrap draws lose spatial support'
    return {'CI95_C_decade':np.percentile(results,[2.5,97.5]).tolist(),'replicates':len(results),
            'block_years':block,'seed':seed,'method':'circular pairs bootstrap of years; joint across pixels'}

def joint_hac_interval(years,values,weights,lag=2):
    x=np.asarray(years,dtype=float)-np.mean(years)
    y=values.reshape(len(x),-1);w=weights.ravel();take=w>0
    y=y[:,take];w=w[take];w=w/w.sum()
    valid=np.isfinite(y);n=valid.sum(0)
    xm=np.sum(valid*x[:,None],axis=0)/n
    ym=np.nansum(y,axis=0)/n
    dx=x[:,None]-xm
    sxx=np.sum(np.where(valid,dx*dx,0),axis=0)
    slope=local_slopes(years,y)/10
    residual=np.where(valid,y-ym-slope*dx,0)
    score=np.sum(residual*dx/sxx*w,axis=1)
    variance=float(score@score)
    for k in range(1,lag+1):variance+=2*(1-k/(lag+1))*float(score[k:]@score[:-k])
    se=math.sqrt(max(variance,0)*len(x)/(len(x)-2))*10
    point=float(slope@w*10);crit=stats.t.ppf(.975,len(x)-2)
    return {'CI95_C_decade':[float(point-crit*se),float(point+crit*se)],'SE_C_decade':se,
            'lag_years':lag,'df':len(x)-2,'method':'joint year scores with Bartlett HAC; finite-year t interval'}

def summary_for(years,values,weights,mask,bootstrap=True):
    w=weights*mask;slopes=local_slopes(years,values)
    answer={'coverage_pct':float(w.sum()/weights.sum()*100),'grid_cells':int(np.count_nonzero(w)),
            'mean_local_OLS_C_decade':weighted_mean(slopes,w)}
    if w.sum()==0:return answer
    answer['median_local_OLS_C_decade']=b.weighted_median(slopes,w)
    answer['median_valid_years']=b.weighted_median(np.isfinite(values).sum(0),w)
    sen=np.full(weights.shape,np.nan)
    for r,c in zip(*np.where(w>0)):
        ok=np.isfinite(values[:,r,c]);sen[r,c]=stats.theilslopes(values[ok,r,c],years[ok]).slope*10
    answer['mean_local_TheilSen_C_decade']=weighted_mean(sen,w)
    if bootstrap:
        answer['bootstrap']=bootstrap_trends(years,values,w)
        answer['joint_HAC']=joint_hac_interval(years,values,w)
    return answer

def annual_cubes(source):
    prefix,years=SOURCES[source]
    cubes={key:{phase:[] for phase in ['day','night']} for key in CONFIGS}
    counts={phase:[] for phase in ['day','night']};times={phase:[] for phase in ['day','night']}
    for year in years:
        path=b.CACHE/f'{prefix}_daily_{year}.tif'
        a,p=b.read(path);p['crs']=rasterio.crs.CRS.from_string(SPHERE)
        dates=pd.to_datetime(json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['dates'],unit='ms')
        a=a.reshape(len(dates),6,*a.shape[1:])
        for phase,offset in [('day',0),('night',3)]:
            t=a[:,offset]*.02-273.15;q=b.recover_qa(a[:,offset+1],a[:,offset])
            for key,(flag,total,monthly) in CONFIGS.items():
                good=b.qa_good(q,flag)&(t>-20)&(t<(70 if phase=='day' else 50))
                v=np.where(good,t,np.nan)
                cubes[key][phase].append(b.monthly_summer(v,dates.month,monthly,total))
                if key=='primary':
                    counts[phase].append(good.sum(0))
                    # Nighttime Aqua crosses midnight at some locations: unwrap to [-6,18) hours.
                    view=a[:,offset+2]*.1
                    if phase=='night' and source=='aqua':view=np.where(view>18,view-24,view)
                    times[phase].append(np.nanmedian(np.where(good,view,np.nan),axis=0))
    for k in CONFIGS:
        for ph in ['day','night']:cubes[k][ph]=np.array(cubes[k][ph])
    arrays={f'{k}_{ph}':cubes[k][ph] for k in CONFIGS for ph in ['day','night']}
    arrays.update({f'counts_{ph}':np.array(counts[ph]) for ph in ['day','night']})
    arrays.update({f'times_{ph}':np.array(times[ph]) for ph in ['day','night']})
    np.savez_compressed(b.TABLE/f'{source}_annual_cubes.npz',years=years,**arrays)
    return cubes,counts,times,p

def sampling_stress_test(years,values,city_weights,centre_weights):
    complete=np.all(np.isfinite(values),axis=0)&(city_weights>0)
    partial=eligible_years(values)&(centre_weights>0)
    if not complete.any() or not partial.any():return None
    x=values[:,complete];w=city_weights[complete]
    baseline=weighted_mean(local_slopes(years,x),w)
    patterns=np.unique(np.isfinite(values[:,partial]).T,axis=0)
    shifts=[]
    for pattern in patterns:
        slope=local_slopes(years,np.where(pattern[:,None],x,np.nan))
        shifts.append(weighted_mean(slope,w)-baseline)
    return {'patterns_tested':len(patterns),'mean_shift_C_decade':float(np.mean(shifts)),
            'shift_min_max_C_decade':[float(np.min(shifts)),float(np.max(shifts))],
            'meaning':'Observed centre missing-year patterns imposed on fully observed city pixels; diagnostic, not correction.'}

def alternative_product_audit():
    rows=[]
    for year in [2001,2025]:
        for phase in ['day','night']:
            path=b.CACHE/f'mod21_{phase}_{year}.tif'
            a,p=b.read(path)
            d=b.gpd.read_file(b.OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
            w=geometry_weights(p,d.loc[d.nazev=='Brno-střed'].geometry.union_all())
            dates=pd.to_datetime(json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['dates'],unit='ms')
            a=a.reshape(len(dates),3,*a.shape[1:]);temp=a[:,0];q=np.nan_to_num(a[:,1],nan=65535).astype(np.uint16)
            plausible=(temp>253.15)&(temp<343.15)
            masks={'available':plausible,'mandatory':plausible&((q&3)<=1),
              'data_quality':plausible&((q&3)<=1)&(((q>>2)&3)==0)}
            masks['cloud_free']=masks['data_quality']&(((q>>4)&3)==0)
            masks['quality2K']=masks['cloud_free']&(((q>>12)&3)>=1)&(((q>>14)&3)>=1)
            stats_out={}
            for key,mask in masks.items():
                annual=b.monthly_summer(np.where(mask,temp-273.15,np.nan),dates.month,1,20)
                stats_out[key]={'median_days':b.weighted_median(mask.sum(0),w),
                   'qualified_centre_area_pct':float(w[np.isfinite(annual)].sum()/w.sum()*100)}
            rows.append({'year':year,'phase':phase,'source':'MOD21A1D' if phase=='day' else 'MOD21A1N',
              'criteria':'QA per MOD21 v6.1 guide table11; <=2K and <=0.02 emissivity error; clear, good L1B; >=20 days/all months',
              'stages':stats_out})
    b.write_json(b.TABLE/'mod21_alternative_audit.json',rows)
    return rows


def cross_source_diagnostics():
    terra=np.load(b.TABLE/'terra_annual_cubes.npz');aqua=np.load(b.TABLE/'aqua_annual_cubes.npz')
    _,p=b.read(b.CACHE/'modis_daily_2025.tif');weights=b.area_weights(p);years=np.arange(2003,2026)
    results={}
    for phase in ['day','night']:
        tv=terra['primary_'+phase][2:];av=aqua['primary_'+phase]
        common_dates=np.isfinite(tv)&np.isfinite(av)
        tv=np.where(common_dates,tv,np.nan);av=np.where(common_dates,av,np.nan)
        support=eligible_years(tv)&(weights>0)
        results[phase]={'city_coverage_pct':float(weights[support].sum()/weights.sum()*100),
                       'terra_mean_C_decade':weighted_mean(local_slopes(years,tv),weights*support),
                       'aqua_mean_C_decade':weighted_mean(local_slopes(years,av),weights*support),
                       'meaning':'Same pixels and observed summers, 2003-2025; distinct local observation times.'}
    b.write_json(b.TABLE/'cross_sensor_matched_support.json',results)
    return results

def analyze():
    result={'estimator':'area-weighted mean of cellwise OLS slopes on observed annual temperatures',
            'report_interval':'joint_HAC; block-bootstrap and block-size sensitivity are supplementary; intervals exclude systematic errors',
            'support_rule':'at least 80% of years, >=3 in first/last five, >=2 in each of five temporal blocks; no imputation',
            'uncertainty':'joint year-score HAC lag2 with t interval; checked against 2500 joint 3-year pairs-bootstrap draws and block lengths 1/5; no systematic error bound',
            'sources':{}}
    district_rows=[];annual_rows=[]
    for source,(prefix,years) in SOURCES.items():
        print('Advanced analysis:',source,flush=True)
        cubes,counts,times,p=annual_cubes(source)
        city=b.boundary().to_crs(p['crs']).geometry.union_all()
        districts=b.gpd.read_file(b.OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
        weights=b.area_weights(p)
        centre=districts.loc[districts.nazev=='Brno-střed'].geometry.union_all().intersection(city)
        cw=geometry_weights(p,centre)
        source_result={'years':years.tolist(),'phases':{}}
        for phase in ['day','night']:
            arr=cubes['primary'][phase];mask=eligible_years(arr)&(weights>0)
            phase_result={'city':summary_for(years,arr,weights,mask),
                          'centre':summary_for(years,arr,cw,mask),'sensitivity':{}}
            b.raster_write(f'{source}_{phase}_balanced_trend_C_decade.tif',np.where(mask,local_slopes(years,arr),np.nan),p)
            b.raster_write(f'{source}_{phase}_balanced_support.tif',np.where(weights>0,mask.astype(float),np.nan),p)
            b.raster_write(f'{source}_{phase}_balanced_valid_years.tif',np.where(weights>0,np.isfinite(arr).sum(0),np.nan),p)
            for config in CONFIGS:
                av=cubes[config][phase];mv=eligible_years(av)&(weights>0)
                common=mask&mv
                phase_result['sensitivity'][config]={'city':summary_for(years,av,weights,mv,False),
                  'centre':summary_for(years,av,cw,mv,False),
                  'primary_mean_on_shared_support':weighted_mean(local_slopes(years,arr),weights*common),
                  'variant_mean_on_shared_support':weighted_mean(local_slopes(years,av),weights*common)}
            broad=cubes['qa3K'][phase];broad_mask=eligible_years(broad)&(weights>0)
            phase_result['broader_quality']={'city':summary_for(years,broad,weights,broad_mask),
                                            'centre':summary_for(years,broad,cw,broad_mask)}
            b.raster_write(f'{source}_{phase}_qa3K_balanced_trend_C_decade.tif',np.where(broad_mask,local_slopes(years,broad),np.nan),p)
            b.raster_write(f'{source}_{phase}_qa3K_balanced_support.tif',np.where(weights>0,broad_mask.astype(float),np.nan),p)
            phase_result['bootstrap_block_sensitivity']={scope:{str(block):bootstrap_trends(years,arr,w*mask,replicates=1200,block=block) for block in [1,5]} for scope,w in [('city',weights),('centre',cw)]}
            # Pre-drift comparison uses the SAME retained cells, without selecting new coverage.
            end=2020 if source=='terra' else 2021
            before=years<=end
            prevalid=np.isfinite(arr[before]).sum(0)>=math.ceil(.8*before.sum())
            premask=mask&prevalid
            phase_result['pre_drift']={'end_year':end,'city':summary_for(years[before],arr[before],weights,premask),
                                     'centre':summary_for(years[before],arr[before],cw,premask),
                                     'full_period_on_same_city_support':summary_for(years,arr,weights,premask,False),
                                     'full_period_on_same_centre_support':summary_for(years,arr,cw,premask,False)}
            phase_result['all25_or23_complete_city']=summary_for(years,arr,weights,np.isfinite(arr).all(0)&(weights>0),False)
            phase_result['missing_year_stress']=sampling_stress_test(years,arr,weights,cw)
            phase_result['observation_hours']={scope:[b.weighted_median(np.array(times[phase])[i],w*mask) for i in [0,-1]] for scope,w in [('city',weights),('centre',cw)]}
            # Annual index removes each cell's mean; changing availability is shown, not hidden.
            climatology=np.nanmean(arr,axis=0)
            for i,year in enumerate(years):
                for scope,w in [('city',weights),('centre',cw)]:
                    annual_rows.append({'source':source,'phase':phase,'scope':scope,'year':year,
                     'observed_area_pct':float(w[mask&np.isfinite(arr[i])].sum()/w.sum()*100),
                     'mean_anomaly_C':weighted_mean(arr[i]-climatology,w*mask),
                     'mean_LST_C':weighted_mean(arr[i],w*mask),
                     'median_valid_days':b.weighted_median(np.array(counts[phase])[i],w*mask) if np.sum(w*mask)>0 else None})
            for _,row in districts.iterrows():
                dw=geometry_weights(p,row.geometry.intersection(city))
                sm=summary_for(years,arr,dw,mask,False)
                district_rows.append({'source':source,'phase':phase,'quality_limit_K':2,'district':row['nazev'],**sm})
                broad_sm=summary_for(years,broad,dw,broad_mask,False)
                district_rows.append({'source':source,'phase':phase,'quality_limit_K':3,'district':row['nazev'],**broad_sm})
            source_result['phases'][phase]=phase_result
        result['sources'][source]=source_result
    pd.DataFrame(district_rows).to_csv(b.TABLE/'balanced_district_trends.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(annual_rows).to_csv(b.TABLE/'balanced_annual_series.csv',index=False)
    result['mod21_pilot']=alternative_product_audit()
    result['cross_source_matched_support']=cross_source_diagnostics()
    b.write_json(b.TABLE/'balanced_trend_results.json',result)
    return result

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        warnings.filterwarnings('ignore',message='Mean of empty slice')
        result=analyze()
    print('Complete',flush=True)
