"""Matched-footprint Landsat period comparison: observed differences, not causal warming."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath
import matplotlib.patheffects as pe
from rasterio.transform import array_bounds
import brno_heat_landsat as b


def period_pair(values, counts, inside):
    """Use one common support for early LST, recent LST and their pixelwise difference."""
    values=np.asarray(values,dtype=float);counts=np.asarray(counts)
    assert values.shape==counts.shape and values.shape[0]==10
    values=np.where((counts>0)&np.isfinite(values),values,np.nan)
    valid_counts=np.where(np.isfinite(values),counts,0)
    common=b.comparison_mask(valid_counts,inside)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        early=np.nanmedian(values[:5],axis=0);recent=np.nanmedian(values[5:],axis=0)
    early=np.where(common,early,np.nan);recent=np.where(common,recent,np.nan)
    return early,recent,recent-early


def build():
    products={}
    for variant,key in [('annual','primary_3K'),('quality','broader_5K')]:
        vals=[];counts=[]
        for year in b.LS_YEARS:
            a,p=b.read(b.CACHE/f'landsat_{variant}_{year}.tif')
            vals.append(a[0]);counts.append(np.nan_to_num(a[1]))
        inside=b.inside_mask(p)
        products[key]=period_pair(vals,counts,inside)
    early,recent,delta=products['broader_5K'];strict=products['primary_3K'][2]
    good=np.isfinite(delta);shared=good&np.isfinite(strict)
    audit=b.json.loads((b.TABLE/'landsat_quality_audit.json').read_text('utf-8'))
    reference_shift=audit['broad_recent_reference_C']-audit['broad_early_reference_C']
    relative=b.read(b.RASTER/'06_landsat_broad_relative_change_C.tif')[0][0]
    np.testing.assert_allclose(delta-reference_shift,relative,atol=5e-6,equal_nan=True)
    for name,arr in [('09_landsat_2001_2005_LST_broad_C',early),('10_landsat_2021_2025_LST_broad_C',recent),
                     ('11_landsat_observed_period_difference_broad_C',delta),('12_landsat_observed_period_difference_primary_C',strict)]:
        b.raster_write(name+'.tif',arr,p)
    def stats(mask):
        ok=mask&good;shared_here=mask&shared
        return {'area_coverage_5K_pct':float(100*ok.sum()/mask.sum()),
          'area_coverage_3K_pct':float(100*np.count_nonzero(mask&np.isfinite(strict))/mask.sum()),
          'median_paired_difference_5K_C':float(np.nanmedian(delta[ok])),
          'early_surface_median_5K_C':float(np.nanmedian(early[ok])),
          'recent_surface_median_5K_C':float(np.nanmedian(recent[ok])),
          'shared_support_median_difference_3K_C':float(np.nanmedian(strict[shared_here])),
          'shared_support_median_difference_5K_C':float(np.nanmedian(delta[shared_here])),
          'median_abs_filter_difference_on_shared_support_C':float(np.median(np.abs(delta[shared_here]-strict[shared_here])))}
    districts=b.gpd.read_file(b.OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
    rows=[]
    for _,row in districts.iterrows():
        mask=b.geometry_mask([row.geometry],inside.shape,p['transform'],invert=True)&inside
        rows.append({'district':row.nazev,**stats(mask)})
    pd.DataFrame(rows).to_csv(b.TABLE/'historical_observed_differences_by_district.csv',index=False,encoding='utf-8-sig')
    result={'early_period':[2001,2005],'recent_period':[2021,2025],
       'definition':'Pixelwise difference of period medians of annual clear-sky LST medians: recent minus early; no baseline subtraction.',
       'quality':'Exploratory 5 K ST_QA; cloud/shadow/saturation screen; cloud distance >=0.5 km; >=3 valid years and >=5 distinct dates in each period.',
       'interpretation':'Observed sampled-surface differences; weather, sensor and clear-sky selection effects are not removed. Not a climate-only change or a significance map.',
       'city':stats(inside),'districts':rows,'reference_shift_used_only_in_older_relative_maps_C':reference_shift,
       'broader_only_area_pct':float(100*np.count_nonzero(good&~shared)/inside.sum()),
       'raster_identity_max_error_C':float(np.nanmax(np.abs(delta-(recent-early)))),
       'native_thermal_detail_m':[60,120],'output_grid_m':30}
    b.write_json(b.TABLE/'historical_observed_comparison.json',result)
    design=b.json.loads((b.TABLE/'cartographic_design.json').read_text('utf-8')) if (b.TABLE/'cartographic_design.json').exists() else {}
    design.update({'report_map':'Landsat 2001-2005, 2021-2025 and observed recent-minus-early difference',
      'historical_native_thermal_detail_m':[60,120],'raster_interpolation':'none','temperature_gap_filling':False,
      'difference_definition':'B minus A on identical observed pixels, without city-baseline subtraction',
      'historical_quality_K':5,'primary_quality_K':3,'broader_only_display':'stippled on difference map'})
    b.write_json(b.TABLE/'cartographic_design.json',design)
    return result,(early,recent,delta,good&~shared),p


def figures(result,arrays,p):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    early,recent,delta,broad_only=arrays
    city=b.boundary().to_crs(p['crs']);districts=b.gpd.read_file(b.OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
    bb=city.total_bounds;left,bottom,right,top=array_bounds(p['height'],p['width'],p['transform'])
    geom=city.geometry.union_all();polys=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
    paths=[PlotPath(np.asarray(r.coords)) for poly in polys for r in [poly.exterior,*poly.interiors]]
    compound=PlotPath.make_compound_path(*paths)
    fig=plt.figure(figsize=(519.276/72,474/72),facecolor='white')
    axes=[fig.add_axes([.005,.56,.48,.37]),fig.add_axes([.515,.56,.48,.37]),fig.add_axes([.005,.085,.57,.40])]
    colors=plt.get_cmap('inferno').copy();colors.set_bad('#849099')
    changes=plt.get_cmap('RdBu_r').copy();changes.set_bad('#849099')
    titles=['A   2001-2005','B   2021-2025','C   DIFFERENCE: B MINUS A']
    handles=[]
    for i,(ax,arr,title) in enumerate(zip(axes,[early,recent,delta],titles)):
        city.plot(ax=ax,color='#849099',edgecolor='none',zorder=-1)
        kw={'cmap':changes,'norm':TwoSlopeNorm(vmin=-6,vcenter=0,vmax=6)} if i==2 else {'cmap':colors,'vmin':22,'vmax':42}
        im=ax.imshow(arr,extent=[left,right,bottom,top],interpolation='none',**kw)
        im.set_clip_path(PathPatch(compound,transform=ax.transData));handles.append(im)
        districts.boundary.plot(ax=ax,color='#183642',linewidth=.20,alpha=.48)
        city.boundary.plot(ax=ax,color='#183642',linewidth=.6)
        ax.set_xlim(bb[0]-350,bb[2]+350);ax.set_ylim(bb[1]-350,bb[3]+350);ax.set_axis_off()
        fig.text(.055 if i!=1 else .565,.96 if i<2 else .48,title if i<2 else 'C   OBSERVED CHANGE',fontsize=8,fontweight='bold',color='#183642')
        if i==2:
            yy,xx=np.indices(broad_only.shape);rr,cc=np.where(broad_only&(yy%8==0)&(xx%8==0))
            xs,ys=b.rasterio.transform.xy(p['transform'],rr,cc)
            ax.scatter(xs,ys,s=.8,color='#263d48',alpha=.65,linewidths=0)
            names={'Brno-Bystrc':'Bystrc','Brno-střed':'střed','Brno-Černovice':'Černovice','Brno-jih':'jih'}
            for _,row in districts.iterrows():
                if row.nazev in names:
                    pt=row.geometry.representative_point()
                    ax.text(pt.x,pt.y,names[row.nazev],ha='center',va='center',fontsize=6.4,color='#183642',
                       path_effects=[pe.withStroke(linewidth=1.8,foreground='white')])
        sx,sy=bb[0]+450,bb[1]+450
        ax.plot([sx,sx+5000],[sy,sy],color='#183642',lw=1.2)
        ax.text(sx+2500,sy+380,'5 km',fontsize=6.3,ha='center',color='#183642')
        ax.annotate('N',xy=(.94,.95),xytext=(.94,.85),xycoords='axes fraction',ha='center',fontsize=6.5,
          arrowprops={'arrowstyle':'-|>','color':'#183642','lw':.6})
    cb=fig.colorbar(handles[0],cax=fig.add_axes([.14,.524,.72,.014]),orientation='horizontal',extend='both',ticks=[22,26,30,34,38,42])
    cb.ax.tick_params(labelsize=7,length=2,pad=2);cb.outline.set_linewidth(.35)
    cb.set_label('Both periods: typical summer surface temperature (°C)',fontsize=7.6,labelpad=2)
    cb.ax.xaxis.set_label_position('top')
    cb=fig.colorbar(handles[2],cax=fig.add_axes([.06,.038,.46,.014]),orientation='horizontal',extend='both',ticks=[-6,-3,0,3,6])
    cb.ax.tick_params(labelsize=7,length=2,pad=2);cb.outline.set_linewidth(.35)
    cb.set_label('Recent minus early temperature (°C)',fontsize=7.6,labelpad=2)
    cb.ax.xaxis.set_label_position('top')
    for suffix in ['pdf','svg','png']:fig.savefig(b.FIG/f'report_historical_comparison.{suffix}',dpi=600,facecolor='white')
    plt.close(fig)


def create():
    result,arrays,p=build();figures(result,arrays,p);return result
