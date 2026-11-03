# -*- coding: utf-8 -*-
"""
land_price_change_1999_2016.py ― 5. (4) 用：TX 沿線の住宅地の地価公示価格の変化（1999年 → 2016年）

2016年版の国土数値情報（地価公示，L01-16）には，各標準地の過去の公示価格が列 L01_048（1983年）～
L01_081（2016年）に入っている（選定対象でない年は 0）．そこで 2016 年版だけを使い，
1999年（L01_064）と 2016年（L01_081）の両方に価格がある標準地について価格の比を計算する．
（同じ標準地の価格の比なので，地点の入れ替わりの影響を受けない）

  python land_price_change_1999_2016.py            （land_price フォルダで実行．2015/L01-16_*.shp を読む）

用途：L01_001 = "000"（住宅地）に限る．市区町村：L01_017（行政区域コード，2016年時点）
"""
import glob

import geopandas as gpd
import numpy as np
import pandas as pd

TX = {"08220": "つくば市", "08235": "つくばみらい市", "08224": "守谷市", "12217": "柏市", "12220": "流山市",
      "11237": "三郷市", "11234": "八潮市", "13121": "足立区", "13118": "荒川区", "13106": "台東区",
      "13101": "千代田区"}
Y0, Y1 = 1999, 2016
c0, c1 = f"L01_{48 + Y0 - 1983:03d}", f"L01_{48 + Y1 - 1983:03d}"   # L01_064, L01_081

d = pd.concat([gpd.read_file(f, encoding="cp932").drop(columns="geometry")
               for f in sorted(glob.glob("2015/L01-16_*.shp"))], ignore_index=True)
assert (d[c1].astype(float) == d["L01_006"].astype(float)).mean() > 0.95, "L01_081 が 2016 年の価格と一致しません"
d["code"] = d["L01_017"].astype(str).str.zfill(5)
d["p0"] = d[c0].astype(float)
d["p1"] = d[c1].astype(float)
r = d[(d["L01_001"].astype(str) == "000") & (d.p0 > 0) & (d.p1 > 0)].copy()
r["ratio"] = r.p1 / r.p0
print(f"住宅地で {Y0} 年と {Y1} 年の両方に価格がある標準地：{len(r):,}（全 {len(d):,} 地点中）")


def summ(x):
    return pd.Series({"地点数": len(x), "比の中央値": x.ratio.median(),
                      "比の幾何平均": float(np.exp(np.log(x.ratio).mean())),
                      f"{Y0}年の中央値[円/m²]": x.p0.median(), f"{Y1}年の中央値[円/m²]": x.p1.median()})


rows = {name: summ(r[r.code == c]) for c, name in TX.items() if (r.code == c).any()}
rows["TX沿線の計"] = summ(r[r.code.isin(TX)])
rows["TX沿線の計（茨城・千葉・埼玉）"] = summ(r[r.code.isin([c for c in TX if not c.startswith("13")])])
rows["5都県の計"] = summ(r)
out = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
pd.set_option("display.unicode.east_asian_width", True)
print(out.to_string(float_format=lambda v: f"{v:,.3f}" if v < 10 else f"{v:,.0f}"))
out.to_csv("land_price_change_1999_2016.csv", encoding="utf-8-sig")
print("→ land_price_change_1999_2016.csv")
