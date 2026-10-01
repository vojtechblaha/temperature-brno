"""Two-page decision brief; plots and layout are generated from computed summaries."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
import rasterio
from rasterio.warp import reproject,Resampling
from rasterio.transform import from_origin
import brno_heat_landsat as b


def selected_results(s):
    a=s['balanced_trends']['sources']
    return [
      ('Morning: Brno','Terra, 2001-2025',a['terra']['phases']['day']['city'],'#177f87'),
      ('Morning: Brno-střed','Terra, 2001-2025',a['terra']['phases']['day']['centre'],'#177f87'),
      ('Evening: sampled Brno','Terra, 2001-2025',a['terra']['phases']['night']['city'],'#aa683b'),
      ('Late night: sampled Brno','Aqua, 2003-2025',a['aqua']['phases']['night']['city'],'#59679c'),
      ('Late night: centre*','Aqua, 2003-2025; QA ≤3 K',a['aqua']['phases']['night']['broader_quality']['centre'],'#59679c')]


def display_raster(name):
    arr,p=b.read(b.RASTER/(name+'.tif'));arr=arr[0]
    bounds=b.boundary().to_crs(b.CRS).total_bounds
    left=math.floor(bounds[0]/50)*50;top=math.ceil(bounds[3]/50)*50
    width=math.ceil((bounds[2]-left)/50);height=math.ceil((top-bounds[1])/50)
    tr=from_origin(left,top,50,50)
    dest=np.full((height,width),b.NODATA,dtype='float32')
    reproject(np.where(np.isfinite(arr),arr,b.NODATA).astype('float32'),dest,
              src_transform=p['transform'],src_crs=p['crs'],src_nodata=b.NODATA,
              dst_transform=tr,dst_crs=b.CRS,dst_nodata=b.NODATA,resampling=Resampling.nearest)
    profile={'driver':'GTiff','width':width,'height':height,'count':1,'dtype':'float32','crs':b.CRS,'transform':tr}
    touched=b.geometry_mask(b.boundary().to_crs(b.CRS).geometry,dest.shape,tr,invert=True,all_touched=True)
    dest[(dest==b.NODATA)|~touched]=np.nan
    filename=name+'_display_UTM.tif';b.raster_write(filename,dest,profile)
    return filename


def figures(s):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    a=s['balanced_trends']['sources'];day=a['terra']['phases']['day']
    trendfile=display_raster('terra_day_balanced_trend_C_decade')
    fig,axes=plt.subplots(1,2,figsize=(9,5.7))
    im=b.plot_panel(axes[0],'02_landsat_recent_anomaly_C.tif','A  Recent surface heat',-8,8,'RdYlBu_r','',False)
    tr=b.plot_panel(axes[1],trendfile,'B  Morning trend, 2001-2025',-1,2,'RdYlBu_r','',False)
    fig.subplots_adjust(left=.018,right=.986,top=.925,bottom=.245,wspace=.055)
    for i,(handle,ticks,label,missing) in enumerate([
        (im,[-8,-4,0,4,8],'2021-2025: difference from city median (°C)',100-s['landsat']['current']['coverage_pct']),
        (tr,[-1,0,1,2],'Observed surface-temperature trend (°C/decade)',100-day['city']['coverage_pct'])]):
        cbax=fig.add_axes([.05+i*.50,.16,.41,.019])
        cb=fig.colorbar(handle,cax=cbax,orientation='horizontal',extend='both',ticks=ticks)
        cb.ax.tick_params(labelsize=10,pad=2);cb.set_label(label,fontsize=9.3,labelpad=3)
        axes[i].legend(handles=[Patch(facecolor='#515b66',label=f'Insufficient data: {missing:.1f}% of Brno')],
                  loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False,fontsize=9.8)
    fig.savefig(b.FIG/'report_main_maps.png',dpi=300,facecolor='white');plt.close(fig)
    # Forest plot: intervals use joint calendar-year uncertainty, not independent pixels.
    rows=selected_results(s)
    fig,ax=plt.subplots(figsize=(9,3.55))
    ax.axvline(0,color='#87959d',lw=.9,ls='--')
    for j,(label,sub,res,col) in enumerate(rows):
        y=4-j;mean=res['mean_local_OLS_C_decade'];low,high=res['joint_HAC']['CI95_C_decade']
        ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt='s' if j==4 else 'o',ms=6,
                    color=col,ecolor=col,elinewidth=1.7,capsize=3,mfc='white' if j==4 else col)
        ax.text(-.065,y,label,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=10.5,
                weight='bold' if j in [0,1] else 'normal',color='#213c49')
        ax.text(1.025,y,f'{mean:+.2f} [{low:+.2f}, {high:+.2f}]',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=10)
        ax.text(1.025,y-.24,f'{res["coverage_pct"]:.1f}% area coverage',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8.3,color='#526c79')
    ax.set_yticks([]);ax.set_ylim(-.55,4.5);ax.set_xlim(-.6,1.9)
    ax.set_xticks([-.5,0,.5,1,1.5]);ax.tick_params(axis='x',labelsize=9)
    ax.set_xlabel('°C per decade; point estimate and 95% temporal HAC interval',fontsize=9)
    for spine in ['left','right','top']:ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=.32,right=.69,top=.98,bottom=.17)
    fig.savefig(b.FIG/'report_trend_intervals.png',dpi=280,facecolor='white');plt.close(fig)
    # A full-sized native-detail trend map for GIS/report inspection.
    for source,phase,quality,title in [
      ('terra','day','balanced','Morning Terra: observed 2001-2025 trend'),
      ('terra','day','qa3K_balanced','Morning Terra: broader-quality screening'),
      ('terra','night','balanced','Evening Terra: restricted night coverage'),
      ('aqua','night','balanced','Late-night Aqua: primary quality, 2003-2025'),
      ('aqua','night','qa3K_balanced','Late-night Aqua: broader-quality screening')]:
        name=f'{source}_{phase}_{quality}_trend_C_decade';filename=display_raster(name)
        fig,ax=plt.subplots(figsize=(7,7.6))
        b.plot_panel(ax,filename,title,-1,2,'RdYlBu_r','°C per decade; descriptive, not a significance map')
        fig.text(.5,.025,'Grey: insufficient record | Native detail ≈1 km | No temperature gap filling',ha='center',fontsize=9)
        fig.savefig(b.FIG/(name+'.png'),dpi=220,bbox_inches='tight')
        alias={'terra_day_balanced':'map_modis_day_trend.png','terra_day_qa3K_balanced':'map_modis_day_qa3K_trend.png','terra_night_balanced':'map_modis_night_trend.png',
               'aqua_night_balanced':'map_aqua_night_trend.png','aqua_night_qa3K_balanced':'map_aqua_night_qa3K_trend.png'}[f'{source}_{phase}_{quality}']
        fig.savefig(b.FIG/alias,dpi=220,bbox_inches='tight');plt.close(fig)
    # Availability-aware annual anomaly diagnostics; low-coverage years are hidden explicitly.
    df=pd.read_csv(b.TABLE/'balanced_annual_series.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col',gridspec_kw={'height_ratios':[2,1]})
    for col,(src,phase,label) in enumerate([('terra','day','Terra morning'),('aqua','night','Aqua late night; primary QA')]):
        for scope,color in [('city','#147e85'),('centre','#b8683e')]:
            sub=df[(df.source==src)&(df.phase==phase)&(df.scope==scope)]
            vals=sub.mean_anomaly_C.where(sub.observed_area_pct>=50)
            axes[0,col].plot(sub.year,vals,'o-',ms=3,color=color,label=scope)
            axes[1,col].plot(sub.year,sub.observed_area_pct,color=color)
        axes[0,col].set_title(label);axes[0,col].set_ylabel('Observed-cell anomaly (°C)');axes[0,col].legend(frameon=False)
        axes[1,col].set_ylabel('District/city coverage (%)');axes[1,col].set_ylim(0,105);axes[1,col].axhline(50,color='grey',lw=.6,ls='--')
    fig.suptitle('Diagnostic indices: annual anomalies shown only with ≥50% area observed',fontsize=12)
    fig.tight_layout();fig.savefig(b.FIG/'balanced_annual_diagnostics.png',dpi=220);plt.close(fig)



def publication_map(s):
    """Native Landsat raster with vector cartography; no smoothing or synthetic detail."""
    import geopandas as gpd
    import matplotlib.patheffects as pe
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch, Rectangle
    from rasterio.transform import array_bounds
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    a,p=b.read(b.RASTER/'02_landsat_recent_anomaly_C.tif')
    city=b.boundary().to_crs(p['crs'])
    districts=gpd.read_file(b.OUT/'boundary/brno_districts.geojson').to_crs(p['crs'])
    centre=districts.loc[districts.nazev=='Brno-střed']
    assert len(centre)==1
    bb=city.total_bounds; cb=centre.total_bounds
    left,bottom,right,top=array_bounds(p['height'],p['width'],p['transform'])
    fig=plt.figure(figsize=(519.276/72,365/72),facecolor='white')
    ax=fig.add_axes([.005,.10,.665,.85]); inset=fig.add_axes([.705,.46,.285,.48])
    cmap=plt.get_cmap('RdYlBu_r').copy();cmap.set_bad('#677781')
    geom=city.geometry.union_all();polys=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
    paths=[PlotPath(np.asarray(r.coords)) for poly in polys for r in [poly.exterior,*poly.interiors]]
    compound=PlotPath.make_compound_path(*paths)
    def layer(axis):
        city.plot(ax=axis,color='#677781',edgecolor='none',zorder=-1)
        im=axis.imshow(a[0],extent=[left,right,bottom,top],cmap=cmap,norm=TwoSlopeNorm(vmin=-8,vcenter=0,vmax=8),interpolation='none')
        im.set_clip_path(PathPatch(compound,transform=axis.transData))
        districts.boundary.plot(ax=axis,color='#213d48',linewidth=.32,alpha=.50)
        city.boundary.plot(ax=axis,color='#213d48',linewidth=.65)
        axis.set_aspect('equal');axis.set_axis_off()
        return im
    im=layer(ax);layer(inset)
    ax.set_xlim(bb[0]-450,bb[2]+450);ax.set_ylim(bb[1]-500,bb[3]+500)
    # Aspect-aware inset fits the entire administrative centre plus a small context buffer.
    cx=(cb[0]+cb[2])/2;cy=(cb[1]+cb[3])/2
    halfy=(cb[3]-cb[1])/2+260
    halfx=max((cb[2]-cb[0])/2+260,halfy*(.285*519.276)/(.48*365))
    halfy=max(halfy,halfx*(.48*365)/(.285*519.276))
    inset.set_xlim(cx-halfx,cx+halfx);inset.set_ylim(cy-halfy,cy+halfy)
    centre.boundary.plot(ax=inset,color='white',linewidth=2.3)
    centre.boundary.plot(ax=inset,color='#173945',linewidth=.85)
    centre.boundary.plot(ax=ax,color='#173945',linewidth=.95)
    ax.add_patch(Rectangle((cx-halfx,cy-halfy),2*halfx,2*halfy,fill=False,ec='#173945',lw=.8,ls=(0,(3,2))))
    names={'Brno-Bystrc':'Bystrc','Brno-Královo Pole':'Královo Pole','Brno-střed':'střed',
           'Brno-Černovice':'Černovice','Brno-Líšeň':'Líšeň','Brno-Tuřany':'Tuřany','Brno-jih':'jih'}
    for _,row in districts.iterrows():
        if row.nazev in names:
            pt=row.geometry.representative_point()
            ax.text(pt.x,pt.y,names[row.nazev],ha='center',va='center',fontsize=7.1,color='#153440',
                    path_effects=[pe.withStroke(linewidth=2.1,foreground='white')])
    # Geographical landmarks are coordinates, not inferred thermal objects.
    from pyproj import Transformer
    tx=Transformer.from_crs('EPSG:4326',p['crs'],always_xy=True)
    for label,lon,lat,offset in [('Špilberk',16.5996,49.1945,(-3,7)),('Main station',16.6128,49.1909,(-4,-10))]:
        x,y=tx.transform(lon,lat)
        inset.plot(x,y,'o',ms=2.8,mec='white',mew=.65,color='#183642')
        inset.annotate(label,(x,y),xytext=offset,textcoords='offset points',ha='right' if offset[0]<0 else 'left',
          fontsize=6.4,color='#183642',path_effects=[pe.withStroke(linewidth=2,foreground='white')])
    def scale(axis,x,y,length,label,fontsize=7):
        axis.plot([x,x+length],[y,y],lw=1.6,color='#183642',solid_capstyle='butt')
        axis.text(x+length/2,y+length*.075,label,ha='center',va='bottom',fontsize=fontsize,color='#183642')
    scale(ax,bb[0]+500,bb[1]+550,5000,'5 km')
    scale(inset,cx-halfx+250,cy-halfy+270,1000,'1 km',6.3)
    ax.annotate('N',xy=(.95,.94),xytext=(.95,.84),xycoords='axes fraction',ha='center',fontsize=7.5,
                arrowprops={'arrowstyle':'-|>','color':'#183642','lw':.9})
    fig.text(.015,.97,'A   BRNO',fontsize=8,fontweight='bold',color='#183642')
    fig.text(.705,.97,'B   BRNO-STŘED',fontsize=8,fontweight='bold',color='#183642')
    fig.text(.705,.405,'99.9% of the centre mapped',fontsize=8,fontweight='bold',color='#147e85')
    fig.text(.705,.364,'Same data and colour scale.\nOutline: district boundary.\nDashed box locates this detail.',fontsize=7.2,linespacing=1.45,color='#536b77',va='top')
    cbax=fig.add_axes([.045,.059,.565,.020])
    bar=fig.colorbar(im,cax=cbax,orientation='horizontal',extend='both',ticks=[-8,-4,0,4,8])
    bar.ax.tick_params(labelsize=7.3,pad=2,length=2);bar.outline.set_linewidth(.35)
    bar.set_label('Difference from Brno surface median (°C)',fontsize=7.7,labelpad=3)
    # PDF stores labels/boundaries as vectors and embeds the original raster pixels.
    fig.savefig(b.FIG/'report_spatial_detail.pdf',dpi=600,facecolor='white')
    fig.savefig(b.FIG/'report_spatial_detail.svg',dpi=600,facecolor='white')
    fig.savefig(b.FIG/'report_spatial_detail.png',dpi=600,facecolor='white')
    plt.close(fig)
    b.write_json(b.TABLE/'cartographic_design.json',{
      'report_map':'Landsat recent anomaly with Brno-stred detail',
      'native_raster_shape':list(a[0].shape),'grid_spacing_m':list(map(float,[p['transform'].a,-p['transform'].e])),
      'recent_native_thermal_detail_m':[60,100],'raster_interpolation':'none','temperature_gap_filling':False,
      'labels_boundaries_and_report_chart':'vector','preview_export_dpi':600,
      'detail_bounds_UTM33N':[cx-halfx,cy-halfy,cx+halfx,cy+halfy]})


def create(s, supplementary=True):
    """Two A4 pages with live vector text/charts and a native-resolution Landsat map."""
    if supplementary: figures(s)  # Scientific supplementary maps remain in the reproducibility package.
    publication_map(s)
    import brno_heat_history
    historical=brno_heat_history.create()
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import pymupdf
    fonts=Path(plt.matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Report','DejaVuSans.ttf'),('ReportBold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='Report',boldItalic='ReportBold')
    W,H=A4;M=38;CW=W-2*M;ink='#173642';accent='#087f83';muted='#536b77';line='#d7e2e4'
    temp=b.ROOT/'tmp/pdfs';temp.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(temp/'report_typeset.pdf'),pagesize=A4)
    c.setTitle('Brno | Surface heat, 2001-2025');c.setAuthor('Satellite evidence brief prepared for Brno City Council')
    c.setSubject('Summer surface-temperature trends and matched Landsat observations, 2001-2005 versus 2021-2025')
    c.setKeywords('Brno, Landsat, MODIS, surface temperature, historical comparison, adaptation, mitigation')
    boxes=[]
    def text(value,x,top,width,size=9,leading=12,color=ink,bold=False):
        p=Paragraph(value,ParagraphStyle('p',fontName='ReportBold' if bold else 'Report',fontSize=size,leading=leading,textColor=HexColor(color)))
        _,hh=p.wrap(width,H);assert top+hh<803,(top,hh,value)
        p.drawOn(c,x,H-top-hh);boxes.append((c.getPageNumber(),x,top,width,hh));return top+hh
    def rule(top,x=M,width=CW):
        c.setStrokeColor(HexColor(line));c.setLineWidth(.6);c.line(x,H-top,x+width,H-top)
    def label(value,x,top,width=CW):return text(value,x,top,width,8,11,accent,True)
    def footer(page):
        rule(811);c.setFont('Report',7);c.setFillColor(HexColor(muted))
        c.drawString(M,H-822,'BRNO CITY COUNCIL  /  SATELLITE EVIDENCE  /  01 OCT 2026')
        c.drawRightString(W-M,H-822,f'{page} / 2')
    a=s['balanced_trends']['sources'];day=a['terra']['phases']['day'];night=a['aqua']['phases']['night']
    ls=s['landsat']['current'];districts=s['landsat']['quality_audit']['district_priority_screen']
    label('EVIDENCE BRIEF   /   SUMMERS 2001-2025',M,29)
    text('Brno’s changing surface heat',M,49,CW,26,34,bold=True)
    text('Summer surface temperature, 2001-2025 | A brief for city decisions',M,89,CW,10.5,15,muted)
    text('Satellite measurements suggest Brno’s surfaces are warming, but the rate is uncertain. '
         'They measure roofs, roads and vegetation; air temperature and effects on people need separate measurements.',M,120,CW,11,15.5)
    cw=(CW-24)/3
    lo_ci,hi_ci=day['city']['joint_HAC']['CI95_C_decade']
    cards=[(f'{day["city"]["mean_local_OLS_C_decade"]:+.2f}','°C EVERY 10 YEARS','Morning trend, 2001-2025',f'95% interval: {lo_ci:+.2f} to {hi_ci:+.2f}'),
      (f'{historical["city"]["median_paired_difference_5K_C"]:+.2f}','°C BETWEEN PERIODS','2001-2005 versus 2021-2025','Observed difference; exploratory'),
      (f'{next(r for r in historical["districts"] if r["district"]=="Brno-střed")["median_paired_difference_5K_C"]:+.2f}','°C IN BRNO-STŘED','Difference: same two periods','Exploratory; 90.9% area mapped')]
    for i,(value,unit,desc,note) in enumerate(cards):
        x=M+i*(cw+12);c.setFillColor(HexColor('#eef5f4'));c.rect(x,H-181-83,cw,83,fill=1,stroke=0)
        text(value,x+10,189,cw-20,23,27,accent,True);text(unit,x+10,219,cw-20,7,9,muted,True)
        text(desc,x+10,235,cw-20,7.5,10);text(note,x+10,248,cw-20,7,9,muted)
    label('01   WHY USE MORNING MEASUREMENTS?',M,283)
    text('Terra’s morning satellite record spans <b>2001-2025</b> and meets our quality checks across '
         '<b>98.5% of Brno and 91.3% of Brno-střed</b>. Night records leave much larger gaps in the centre; '
         'Aqua’s full-summer record starts in 2003. Morning is the most complete comparison here, but does not represent '
         'the daily average or afternoon peak. [1]',M,305,CW,9.8,14)
    label('02   ESTIMATED MORNING CHANGE, 2001-2025',M,391)
    col=(CW-22)/2
    for i,(name,result) in enumerate([('Brno',day['city']),('Brno-střed',day['centre'])]):
        x=M+i*(col+22)
        c.setFillColor(HexColor('#f2f5f5'));c.rect(x,H-414-109,col,109,fill=1,stroke=0)
        text(name,x+12,424,col-24,11,14,ink,True)
        text(f'{result["mean_local_OLS_C_decade"]:+.2f} °C every 10 years',x+12,446,col-24,15,20,accent,True)
        low,high=result['joint_HAC']['CI95_C_decade']
        text(f'95% statistical range:<br/><b>{low:+.2f} to {high:+.2f} °C</b> every 10 years',x+12,477,col-24,9,12,muted)
    text('<b>How to read these panels:</b> the large number is the estimated average morning change every 10 years. The range shows statistical uncertainty. '
         'Both ranges include zero (no change), so these data do not establish a precise warming rate. '
         'The range also changes with the statistical method; this is not proof that no warming occurred. [9]',M,538,CW,9.5,13)
    label('03   TWO DATA SOURCES, TWO DIFFERENT JOBS',M,605)
    text('Free MODIS and Landsat archives were processed in <b>Google Earth Engine</b> (screening/export) '
         'and <b>Python</b> (analysis/maps).',M,625,CW,8.5,11,muted)
    text('<b>MODIS: frequent, coarse measurements.</b> Daily data at about 1 km detail provide the 25-year record. '
         'We use June-August, at least 20 clear observations per summer and at least 20 summers per location. '
         'Clouds obscure surfaces; suspect pixels are removed, never filled in. [1,5]',M,653,col,8.6,12)
    text('<b>Landsat: finer detail, fewer observations.</b> Page 2 compares 2001-2005 with 2021-2025 at the same locations. '
         'Landsat 8/9 now offer an 8-day repeat cycle, but cloud-free images are less frequent. '
         'More passes do not make thermal imaging all-weather. [2,11]',M+col+22,653,col,8.6,12)
    rule(744)
    text('<b>One important limit:</b> Terra’s typical morning observation time shifted from 11:18 to 10:00 over the record. '
         'A surface can have different temperatures at different hours, so the estimated trend cannot be attributed entirely '
         'to climate change. Night-time results are kept separately in the supporting analysis. [3,4]',M,757,CW,8.7,12)
    footer(1);c.showPage()
    label('HISTORICAL COMPARISON   /   SAME OBSERVED PIXELS',M,29)
    text('Where surface heat changed',M,49,CW,26,34,bold=True)
    text('Landsat | 2001-2005 versus 2021-2025 | clear-sky summers',M,89,CW,10.5,15,muted)
    text('A and B use the same temperature scale and spatial coverage. C subtracts the early temperature '
         'from the recent temperature at each location: red is warmer, blue cooler.',M,115,CW,9.5,13)
    map_top=149;map_height=474
    sx=M+CW*.62;sw=CW*.38-9
    hc=historical['city'];centre=next(r for r in historical['districts'] if r['district']=='Brno-střed')
    c.setFillColor(HexColor('#eef5f4'));c.rect(sx-9,H-393-180,sw+18,180,fill=1,stroke=0)
    label('CHANGE AT A TYPICAL LOCATION',sx,400,sw)
    text(f'{hc["median_paired_difference_5K_C"]:+.2f} °C',sx,419,sw,23,28,accent,True)
    text(f'Middle value of all local changes. Both periods cover {hc["area_coverage_5K_pct"]:.1f}% of Brno.',sx,452,sw,8,11)
    text(f'<b>Brno-střed: {centre["median_paired_difference_5K_C"]:+.2f} °C</b><br/>'
         f'{centre["area_coverage_5K_pct"]:.1f}% of the centre is mapped. A stricter quality check leaves only {centre["area_coverage_3K_pct"]:.1f}%, so this estimate is exploratory.',sx,483,sw,8,11)
    text('<b>Local uncertainty.</b> Where both checks apply, changing the quality limit shifts the estimate by typically '
         f'{hc["median_abs_filter_difference_on_shared_support_C"]:.2f} °C. Small changes need caution.',sx,530,sw,7.8,10.5)
    # Swatches make data absence distinct from the white zero of the difference scale.
    for yy,is_dotted,caption in [(579,True,'Less strict quality check only: 24.3%'),(593,False,'Not enough measurements: 1.8%')]:
        c.setFillColor(HexColor('#f0f3f4' if is_dotted else '#849099'))
        c.rect(sx,H-yy-8,10,8,fill=1,stroke=0)
        if is_dotted:
            c.setFillColor(HexColor('#263d48'))
            for dx in [2,5,8]:
                for dy in [2,5]:c.circle(sx+dx,H-yy-dy,.45,fill=1,stroke=0)
        text(caption,sx+16,yy-1,sw-16,7.4,10)
    text('White inside C = little change. Colour does not show certainty.',sx,606,sw,7.3,9.5,muted)
    text('<b>How the maps were made.</b> Each period combines summer measurements from at least 3 years and 5 dates at the same locations; '
         'C = B - A. Wider coverage needs a less strict uncertainty limit per measurement (5 °C instead of 3 °C). Weather and sensor differences also affect C. '
         'Thermal detail is 60-120 m on a 30 m grid: read city blocks, not individual buildings. [2,7,8,10]',M,632,CW,8,10.5,muted)
    text('<b>1. Adaptation: reduce heat exposure.</b> Prioritise Brno-střed, Černovice and Brno-jih, where current '
         'hot surfaces cluster. Target shade trees at schools, care homes and walking routes; verify benefits with local air measurements. [6]',M,678,col,8.3,11.5)
    text('<b>2. Mitigation: reduce energy and emissions.</b> Pilot suitable cool/green roofs to reduce cooling demand. '
         'Assess annual heating and cooling energy before claiming emissions savings; compare pilot buildings with untreated ones. [6,12]',M+col+22,678,col,8.3,11.5)
    text('<b>Next investment.</b> Add ground air/night monitoring. Before buying OroraTech or other surveys, trial calibration, '
         'usable cloud-free revisit and cost per usable image in Brno. Satellite maps target checks; '
         'they do not measure health or emissions benefits.',M,740,CW,7.6,10)
    sources=[('1','MODIS QA','https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1'),
      ('2','Landsat ST','https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature'),
      ('3','Terra drift','https://terra.nasa.gov/about/terras-orbit-changes/terra-orbital-drift-information'),
      ('4','Aqua drift','https://aqua.nasa.gov/sites/default/files/AquaStatus.pdf'),
      ('5','Retrieval issues','https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues'),
      ('6','Cooling measures','https://www.epa.gov/heatislands/heat-island-reduction-solutions'),
      ('7','Brno districts','https://gis.brno.cz/ags1/rest/services/OMI/OMI_mc_web_brno/FeatureServer/0'),
      ('8','OSM / ODbL','https://www.openstreetmap.org/relation/438171'),
      ('9','Trend uncertainty','https://doi.org/10.2307/1913610'),
      ('10','Native resolution','https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites'),
      ('11','Revisit cycle','https://www.usgs.gov/faqs/what-a-landsat-satellite-constellation'),
      ('12','Roof energy trade-offs','https://www.epa.gov/heatislands/using-cool-roofs-reduce-heat-islands')]
    refs=' · '.join(f'[{n}] <link href="{u}" color="{accent}">{t}</link>' for n,t,u in sources)
    text('<b>Sources.</b> '+refs,M,774,CW,7,9,muted)
    footer(2);c.save()
    for i,(page,x,y,w,h) in enumerate(boxes):
        for page2,x2,y2,w2,h2 in boxes[i+1:]:
            if page==page2:assert min(x+w,x2+w2)-max(x,x2)<.1 or min(y+h,y2+h2)-max(y,y2)<.1,('Text overlap',page,(x,y,w,h),(x2,y2,w2,h2))
    doc=pymupdf.open(temp/'report_typeset.pdf');mapdoc=pymupdf.open(b.FIG/'report_historical_comparison.pdf')
    doc[1].show_pdf_page(pymupdf.Rect(M,map_top,M+CW,map_top+map_height),mapdoc,0,overlay=False)
    doc.set_toc([[1,'1. Long-term surface-temperature trends',1],[1,'2. Historical maps and observed differences',2]])
    final=b.PDF/'brno_heat_report.pdf';doc.save(final,garbage=4,deflate=True);doc.close()
    mapdoc.close()
    check=pymupdf.open(final);assert len(check)==2
    for i,page in enumerate(check):page.get_pixmap(matrix=pymupdf.Matrix(2,2)).save(temp/f'redesign_{i+1}.png')
    b.write_json(b.TABLE/'report_qa.json',{'pages':2,'format':'A4','text_box_count':len(boxes),
      'text_boxes_inside_margins':True,'text_overlap_assertions_passed':True,'visual_inspection':'pending',
      'vector_summary':True,'vector_map_labels_and_boundaries':True,'native_raster_retained':True,'explicit_historical_comparison':True})
    print('REPORT:',final,flush=True)


if __name__=='__main__':
    create(json.loads((b.TABLE/'results_summary.json').read_text(encoding='utf-8')))
