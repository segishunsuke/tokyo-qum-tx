import pandas as pd, geopandas as gpd, numpy as np, glob, warnings
from shapely.geometry import box
warnings.filterwarnings('ignore')
def mesh_poly(codes):
    c=codes.astype(str)
    YY=c.str[0:2].astype(int);XX=c.str[2:4].astype(int);J=c.str[4].astype(int);I=c.str[5].astype(int);j=c.str[6].astype(int);i=c.str[7].astype(int)
    lat=YY/1.5+J/12+j/120; lon=100+XX+I/8+i/80; dlat=1/120; dlon=1/80
    n=c.str.len()
    if (n>=9).all():
        L=c.str[8].astype(int); lat=lat+(L>=3)/240; lon=lon+(L%2==0)/160; dlat=1/240; dlon=1/160
    if (n>=10).all():
        K=c.str[9].astype(int); lat=lat+(K>=3)/480; lon=lon+(K%2==0)/320; dlat=1/480; dlon=1/320
    return [box(x,y,x+dlon,y+dlat) for x,y in zip(lon,lat)]
def read_estat(pat,col):
    d=pd.concat([pd.read_csv(f,skiprows=[1],dtype=str,encoding='cp932',usecols=['KEY_CODE',col]) for f in glob.glob(pat)])
    d['v']=pd.to_numeric(d[col]); return d.drop_duplicates('KEY_CODE')
def read_csis(pat):
    d=pd.concat([pd.read_csv(f,dtype=str,encoding='cp932',usecols=['F1','F3']) for f in glob.glob(pat) ])
    d=d.rename(columns={'F1':'KEY_CODE'}); d['v']=pd.to_numeric(d['F3']); return d.drop_duplicates('KEY_CODE')
EPSG_PROJ=6677  # JGD2000 / 平面直角9系
def test(old,new,first='5339',label=''):
    o=old[old.KEY_CODE.str[:4]==first].copy(); n=new[new.KEY_CODE.str[:4]==first].copy()
    ng=gpd.GeoDataFrame(n,geometry=mesh_poly(n.KEY_CODE),crs='EPSG:4612').to_crs(EPSG_PROJ)
    ng['a0']=ng.area
    res={}
    for name,crs in [('世界測地系(4612)','EPSG:4612'),('日本測地系(4301)','EPSG:4301')]:
        og=gpd.GeoDataFrame(o,geometry=mesh_poly(o.KEY_CODE),crs=crs).to_crs(EPSG_PROJ)
        x=gpd.overlay(og[['KEY_CODE','v','geometry']].rename(columns={'v':'v_old'}),ng[['v','a0','geometry']],how='intersection')
        x['w']=x['v']*x.area/x['a0']
        agg=x.groupby('KEY_CODE')['w'].sum()
        m=o.set_index('KEY_CODE')['v'].to_frame('old').join(agg.rename('new')).fillna(0)
        r=np.corrcoef(np.log1p(m.old),np.log1p(m.new))[0,1]
        res[name]=r
    print(label,'| log相関（1次メッシュ',first,'）:',{k:round(v,4) for k,v in res.items()})
pop95=read_estat('residence/1995/tblT000752H*.txt','T000752001')
pop15=read_estat('residence/2015/tblT000876Q*.txt','T000876001')
emp96=read_csis('jobs/1995/[0-9][0-9][0-9][0-9].csv')
emp16=read_estat('jobs/2015/tblT000918H*.txt','T000918020')
for f in ['5339','5340','5439']:
    test(pop95,pop15,f,'人口1995(500m) vs 2015(250m)')
    test(emp96,emp16,f,'従業者1996(1km) vs 2016(500m)')
# 対照：2015年の250m人口と2016年の500m従業者（どちらも世界測地系と分かっているもの同士）は参考外
