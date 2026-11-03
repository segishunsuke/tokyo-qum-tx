# -*- coding: utf-8 -*-
"""
generate_dummy_1995.py ― 公開用の1995年従業者数（代替データ）の作成

1996年事業所・企業統計調査の地域メッシュ統計は公開されていないため，以下の公開データから
ゾーン別の従業者数を推計する．
  (1) 2001年事業所・企業統計調査 4次メッシュ（500m）統計「全産業事業所数及び従業者総数」（e-Stat，世界測地系）
  (2) 1996年事業所・企業統計調査 市区町村別従業者数（e-Stat，都道府県編 表003）
  (3) 1995年10月1日時点の行政区域（国土数値情報 N03-951001）

手順
  1. 500mメッシュと行政区域ポリゴンを重ね，各メッシュの従業者数を重なった部分の面積で按分する．
  2. 行政区域ごとに，按分した2001年の従業者数の合計が1996年の市区町村別従業者数に一致するように
     倍率（1996年の従業者数／2001年の按分合計）を掛ける．
     複数の行政区域にまたがるメッシュは，部分ごとに異なる倍率が掛かる．
  3. 補正した部分を計画基本ゾーン（H30_kzone）に面積按分して集計する．

使い方（1995_predict フォルダで実行）
  python generate_dummy_1995.py
出力
  employee_1995_dummy.csv（列：id, kzone, 従業者数_1995）
"""
import argparse
import glob
import io
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ap = argparse.ArgumentParser()
ap.add_argument("--mesh_dir", default="2001_Jigyosho")
ap.add_argument("--muni_csv", default="1996_Jigyosho.csv")
ap.add_argument("--n03_dir", default="1995_N03")
ap.add_argument("--zones", default="kzone/H30_kzone.shp")
ap.add_argument("--mesh_crs", default="EPSG:4612", help="2001年メッシュの測地系（e-Stat の提供データは世界測地系）")
ap.add_argument("--out", default="employee_1995_dummy.csv")
ap.add_argument("--compare", help="比較用：実データから作成した従業者数の CSV（id, kzone, 従業者数_1995）")
a = ap.parse_args()

zones = gpd.read_file(a.zones)
zones["id"] = range(len(zones))
kz = "kzone" if "kzone" in zones.columns else zones.columns[0]
CRS = zones.crs

# ---------------------------------------------------------------- (1) 2001年 4次メッシュ
frames = []
for f in sorted(glob.glob(os.path.join(a.mesh_dir, "*.zip"))):
    with zipfile.ZipFile(f) as z:
        for n in z.namelist():
            if n.lower().endswith(".txt"):
                raw = z.read(n)
                for enc in ("cp932", "utf-8-sig"):
                    try:
                        txt = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                d = pd.read_csv(io.StringIO(txt), dtype=str, skiprows=[1])
                frames.append(d[["KEY_CODE", "T000388002"]])
m = pd.concat(frames, ignore_index=True).drop_duplicates("KEY_CODE")
m["emp"] = pd.to_numeric(m["T000388002"].str.replace(",", ""), errors="coerce").fillna(0.0)
c = m["KEY_CODE"].str.zfill(9)
lat = c.str[0:2].astype(int) / 1.5 + c.str[4].astype(int) / 12 + c.str[6].astype(int) / 120
lon = 100 + c.str[2:4].astype(int) + c.str[5].astype(int) / 8 + c.str[7].astype(int) / 80
q = c.str[8].astype(int)                                  # 1:南西 2:南東 3:北西 4:北東
lat = lat + np.where(q >= 3, 1 / 240, 0)
lon = lon + np.where(q % 2 == 0, 1 / 160, 0)
mesh = gpd.GeoDataFrame({"mesh": c, "emp": m["emp"]},
                        geometry=[box(x, y, x + 1 / 160, y + 1 / 240) for x, y in zip(lon, lat)],
                        crs=a.mesh_crs).to_crs(CRS)
mesh["mesh_area"] = mesh.geometry.area
print(f"2001年メッシュ：{len(mesh):,} 件，従業者数計 {mesh.emp.sum():,.0f}")

# ---------------------------------------------------------------- (2) 1996年 市区町村別従業者数
rows = pd.read_csv(a.muni_csv, header=None, skiprows=16, dtype=str, encoding="utf-8-sig")
hdr = pd.read_csv(a.muni_csv, header=None, skiprows=16, nrows=1, dtype=str, encoding="utf-8-sig").iloc[0].tolist()
col = [i for i, h in enumerate(hdr) if isinstance(h, str) and h.startswith("従業者数") and "平成８年" in h and "実数" in h][0]
muni = pd.DataFrame({"code": rows[3].str.zfill(5), "name": rows[5],
                     "E1996": pd.to_numeric(rows[col].str.replace(",", ""), errors="coerce")}).dropna(subset=["E1996"])

# ---------------------------------------------------------------- (3) 1995年 行政区域
polys = []
for f in sorted(glob.glob(os.path.join(a.n03_dir, "*.zip"))):
    with zipfile.ZipFile(f) as z:
        shp = [n for n in z.namelist() if n.endswith(".shp")][0]
    g = gpd.read_file(f"zip://{os.path.abspath(f)}!{shp}", encoding="cp932")
    g = g.set_crs("EPSG:4612", allow_override=True)       # N03-951001 は JGD2000（緯度経度）
    polys.append(g)
adm = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True), crs="EPSG:4612")
adm["geometry"] = adm.geometry.make_valid()
adm = adm.dropna(subset=["N03_007"]).dissolve(by="N03_007", as_index=False)[["N03_007", "geometry"]]
adm = adm.rename(columns={"N03_007": "code"}).to_crs(CRS)
# 1995年10月から1996年10月の間の市制施行によるコードの変更（1996年4月1日）
RECODE = {"11466": "11243",   # 埼玉県北葛飾郡吉川町 → 吉川市
          "12327": "12231"}   # 千葉県印旛郡印西町 → 印西市
adm["code"] = adm["code"].replace(RECODE)
adm = adm.merge(muni, on="code", how="left")
miss = adm[adm.E1996.isna()]
if len(miss):
    print("1996年の従業者数が見つからない行政区域：", miss.code.tolist())

# 計算量削減：ゾーンの範囲に関わる行政区域・メッシュに限る
zb = zones.total_bounds
adm_in = adm[adm.intersects(box(*zb))]
mesh = mesh[mesh.intersects(box(*adm_in.total_bounds))]

# ---------------------------------------------------------------- 1. メッシュ × 行政区域
pc = gpd.overlay(mesh, adm_in[["code", "E1996", "geometry"]], how="intersection", keep_geom_type=True)
pc["emp01"] = pc.emp * pc.geometry.area / pc.mesh_area
# ---------------------------------------------------------------- 2. 市区町村ごとの倍率
s01 = pc.groupby("code").emp01.sum()
fac = (adm_in.set_index("code").E1996 / s01).replace([np.inf, -np.inf], np.nan)
bad = fac[fac.isna() & (adm_in.set_index("code").E1996 > 0)]
if len(bad):
    print("2001年メッシュの従業者数がゼロで倍率を計算できない行政区域：", bad.index.tolist())
pc["fac"] = pc.code.map(fac).fillna(0.0)
pc["emp96"] = pc.emp01 * pc.fac
pc["piece_area"] = pc.geometry.area
print("倍率（1996年／2001年）の分位点 5/50/95%：", np.round(fac.dropna().quantile([0.05, 0.5, 0.95]).values, 3))

# ---------------------------------------------------------------- 3. ゾーンへの按分
x = gpd.overlay(zones[["id", kz, "geometry"]], pc[["emp96", "piece_area", "geometry"]], how="intersection",
                keep_geom_type=True)
x["v"] = x.emp96 * x.geometry.area / x.piece_area
res = zones[["id", kz]].merge(x.groupby("id").v.sum().rename("従業者数_1995").reset_index(), on="id", how="left")
res["従業者数_1995"] = res["従業者数_1995"].fillna(0.0)
res.sort_values("id").to_csv(a.out, index=False, encoding="utf-8-sig")
print(f"出力：{a.out}（ゾーン計 {res['従業者数_1995'].sum():,.1f} 人）")

if a.compare:
    r = pd.read_csv(a.compare, encoding="utf-8-sig").merge(res, on="id", suffixes=("_real", "_dummy"))
    xr, xd = r["従業者数_1995_real"], r["従業者数_1995_dummy"]
    print(f"実データとの比較：合計 {xr.sum():,.0f} / {xd.sum():,.0f}，相関係数 {np.corrcoef(xr, xd)[0, 1]:.4f}，"
          f"対数の相関 {np.corrcoef(np.log1p(xr), np.log1p(xd))[0, 1]:.4f}，"
          f"平均絶対対数比 {np.mean(np.abs(np.log((xd + 1) / (xr + 1)))):.4f}")
