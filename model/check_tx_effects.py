# -*- coding: utf-8 -*-
"""
check_tx_effects.py ― 5. (3) の記述の確認用：TX の効果（ゾーン別）を市区町村名つきで一覧する

  python check_tx_effects.py --zones ..\\H30_gis\\H30_kzone.shp --admin ..\\H30_gis\\N03-180101_GML.zip
  市区町村名は国土数値情報（行政区域，N03）から，各ゾーンの代表点を含む市区町村として付与する
  （政令指定都市は行政区名まで．--admin には zip でも展開後の shp でも指定可）

出力：
  - 各変数（居住者数密度・従業者数密度・地価）の増加・減少の大きいゾーン上位10（市区町村名つき）
  - 市区町村別の集計：居住者数・従業者数の増減（人），地価の変化量（面積加重平均，円/m²）
"""
import argparse

import geopandas as gpd
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zones", required=True)
ap.add_argument("--effects", default="analysis/tx_effect_zones.csv")
ap.add_argument("--tx_zones", default="analysis/tx_zones.csv")
ap.add_argument("--zone_field")
ap.add_argument("--admin", required=True, help="国土数値情報 行政区域（N03）の zip または shp")
ap.add_argument("--n", type=int, default=10)
a = ap.parse_args()

t = pd.read_csv(a.effects, encoding="utf-8-sig")
tz = set(pd.read_csv(a.tx_zones, encoding="utf-8-sig").kzone)
t["TX沿線"] = t.kzone.isin(tz).map({True: "○", False: ""})

z = gpd.read_file(a.zones)
attrs = z.drop(columns="geometry")
print("shapefile の列：", list(attrs.columns))
if a.zone_field:
    attrs["kzone"] = attrs[a.zone_field].astype(int)
else:                                       # レコード順 = analyze.py のゾーン順
    if len(attrs) != len(t):
        raise SystemExit("レコード数が一致しません。--zone_field を指定してください。")
    attrs["kzone"] = t.kzone.to_numpy()
import os
src = a.admin
if src.lower().endswith(".zip"):
    import zipfile
    with zipfile.ZipFile(src) as zf:
        shp = [n for n in zf.namelist() if n.lower().endswith(".shp")][0]
    src = f"zip://{os.path.abspath(src)}!{shp}"
zz = z.to_crs(6668)
adm = None
for enc in ("cp932", "utf-8"):
    try:
        adm = gpd.read_file(src, bbox=tuple(zz.total_bounds), encoding=enc)
        if adm["N03_004"].astype(str).str.contains("市|区|町|村").any():
            break
    except Exception as e:
        print("読み込みに失敗：", enc, e)
adm = adm.to_crs(6668)
def col(c):
    return adm[c].fillna("").astype(str) if c in adm.columns else pd.Series("", index=adm.index)
# 2018年版：政令指定都市は N03_003 に市名，N03_004 に区名（N03_003 が郡名の場合は付けない）．
# 新しい版：N03_004 に市名，N03_005 に区名
city = col("N03_003").where(col("N03_003").str.endswith("市"), "")
adm["市区町村"] = col("N03_001") + " " + city + col("N03_004") + col("N03_005")
pt = gpd.GeoDataFrame({"kzone": attrs["kzone"].to_numpy()}, geometry=zz.geometry.representative_point(), crs=6668)
j_ = gpd.sjoin(pt, adm[["市区町村", "geometry"]], how="left", predicate="within").drop_duplicates("kzone")
attrs = attrs.merge(j_[["kzone", "市区町村"]], on="kzone", how="left")
print("市区町村名が付かなかったゾーン：", int(attrs["市区町村"].isna().sum()))
mc = "市区町村"
cols = ["kzone"] + ([mc] if mc else [])
t = t.merge(attrs[cols].drop_duplicates("kzone"), on="kzone", how="left")

pd.set_option("display.width", 200)
pd.set_option("display.unicode.east_asian_width", True)
for c, lab in [("D_L_R_per_G", "居住者数密度の変化（人/km²）"), ("D_L_M_per_G", "従業者数密度の変化（人/km²）"),
               ("D_land_price", "地価の変化（円/m²）")]:
    show = cols + ["TX沿線", c]
    print(f"\n==== {lab}：増加の大きいゾーン")
    print(t.nlargest(a.n, c)[show].to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print(f"==== {lab}：減少の大きいゾーン")
    print(t.nsmallest(a.n, c)[show].to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

if mc:
    t["dR"] = t.D_L_R_per_G * t.area_km2
    t["dM"] = t.D_L_M_per_G * t.area_km2
    t["lpA"] = t.D_land_price * t.area_km2
    g = t.groupby(mc).agg(居住者数の増減=("dR", "sum"), 従業者数の増減=("dM", "sum"), lpA=("lpA", "sum"),
                         面積=("area_km2", "sum"))
    g["地価の変化（面積加重平均）"] = g.lpA / g.面積
    g = g.drop(columns=["lpA", "面積"])
    ff = lambda v: f"{v:,.0f}"
    for c in ["居住者数の増減", "従業者数の増減", "地価の変化（面積加重平均）"]:
        print(f"\n==== 市区町村別：{c}（上位・下位 {a.n}）")
        print(pd.concat([g.nlargest(a.n, c), g.nsmallest(a.n, c)])[[c]].to_string(float_format=ff))
