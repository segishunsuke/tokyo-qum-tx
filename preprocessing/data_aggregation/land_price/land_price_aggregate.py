# -*- coding: utf-8 -*-
"""
計画基本ゾーン別の平均地価の集計（1996年・2016年．論文 4. (2) c)）

  - 地価公示（L01，1月1日時点）と都道府県地価調査（L02，7月1日時点）の地点の価格の，ゾーン内の単純平均．
    地点がないゾーンは，ゾーン重心の最寄り地点の価格で補完する．
  - 都道府県地価調査の林地（用途区分コード 020）は除外する（価格が 10a 当たりで単位が異なるため）．
  - 都道府県地価調査の価格は，各地点の前年7月と当年7月の価格の幾何平均により1月1日時点に補正する：
        P_Jan = sqrt(P_Jul(前年) * P_Jul(当年))
    - 2016年：2016年データに含まれる各地点の価格履歴（2015年・2016年）を用いる．
    - 1996年：1995年の都道府県地価調査データ（L02-95）と，前年の基準地番号により地点ごとに照合する．
      番号で照合できない地点は位置（5m以内）で照合する．
    - 照合できない地点は，同一市区町村・同一用途 → 同一都県・同一用途 → 同一都県 → 全体 の順に，
      照合できた地点の補正率の中央値を用いる．

使い方
  python land_price_aggregate.py 2016 --dir 2015 --out_dir output
  python land_price_aggregate.py 1996 --dir 1995 --prev_dir 1995/L02-95 --out_dir output
  （--dir には国土数値情報の L01・L02 の shapefile と，計画基本ゾーンの H30_kzone.shp を置く）
出力
  land_price_{year}_average_corrected.csv  … 列：id, kzone, 平均地価_{year}
"""
import argparse
import glob
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
FOREST = "020"


def load_points(folder, year):
    """地価公示（L01）と地価調査（L02）を読み込み，列名を X_xxx に統一する．"""
    yy = str(year)[2:]
    pts = []
    for kind in ["L01", "L02"]:
        for f in sorted(glob.glob(os.path.join(folder, f"{kind}-{yy}_*.shp"))):
            g = gpd.read_file(f, encoding="cp932")
            pre = "L02" if "L02_006" in g.columns else "L01"  # 1996年の地価調査は L01_ の列名
            g = g.rename(columns={c: "X" + c[3:] for c in g.columns if c.startswith(pre)})
            g["src"] = kind
            pts.append(g)
    if not pts:
        raise FileNotFoundError(f"{folder} に {year} 年の地価データが見つかりません．")
    p = gpd.GeoDataFrame(pd.concat(pts, ignore_index=True))
    p.set_crs("EPSG:4612", allow_override=True, inplace=True)
    p["price"] = pd.to_numeric(p["X_006"], errors="coerce")
    return p.dropna(subset=["price"])


def half_year_rate_2016(p):
    """2016年：各地点の価格履歴から log(P_Jan2016 / P_Jul2016) を計算．"""
    p15 = pd.to_numeric(p["X_078"], errors="coerce")  # 2015年7月
    p16 = pd.to_numeric(p["X_079"], errors="coerce")  # 2016年7月
    ok = (p["src"] == "L02") & (p15 > 0) & (p16 > 0) & (np.abs(p16 - p["price"]) < 1)
    return pd.Series(np.where(ok, 0.5 * np.log(p15 / p16), np.nan), index=p.index)


def half_year_rate_1996(p, prev_dir):
    """1996年：1995年の地価調査（L02-95）と照合して log(P_Jan1996 / P_Jul1996) を計算．
    照合は (1) 前年の基準地番号，(2) 位置（5m以内）の順に行う．"""
    files = sorted(glob.glob(os.path.join(prev_dir, "**", "L02-95_*.shp"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{prev_dir} に1995年の地価調査データ（L02-95_*.shp）が見つかりません．")
    q = gpd.GeoDataFrame(pd.concat([gpd.read_file(f, encoding="cp932") for f in files], ignore_index=True))
    q.set_crs("EPSG:4612", allow_override=True, inplace=True)
    q = q.to_crs(p.crs)
    q["P95"] = pd.to_numeric(q["L01_006"], errors="coerce")
    q["key"] = q["L01_017"].astype(str) + "-" + q["L01_001"].astype(str) + "-" + q["L01_002"].astype(str)

    l2 = p[p["src"] == "L02"].copy()
    l2["pkey"] = l2["X_017"].astype(str) + "-" + l2["X_003"].astype(str) + "-" + l2["X_004"].astype(str)
    p95 = l2["pkey"].map(q.drop_duplicates("key").set_index("key")["P95"])
    how = pd.Series(np.where(p95.notna(), "番号", ""), index=l2.index)

    rest = l2[p95.isna()]
    if len(rest):
        nn = gpd.sjoin_nearest(rest[["geometry"]], q[["P95", "geometry"]], how="left",
                               max_distance=5, distance_col="dist")
        nn = nn[~nn.index.duplicated()]["P95"]
        p95.loc[nn.dropna().index] = nn.dropna()
        how.loc[nn.dropna().index] = "位置"

    ok = (p95 > 0) & (l2["price"] > 0)
    print(f"  1995年データとの照合：番号 {(how == '番号').sum():,}，位置 {(how == '位置').sum():,}，"
          f"照合不可 {(~ok).sum():,}（地価調査 {len(l2):,} 地点中）")
    r = pd.Series(np.nan, index=p.index)
    r.loc[ok[ok].index] = 0.5 * np.log(p95[ok] / l2.loc[ok, "price"])
    return r


def fill_rates(p, r):
    """照合できない地点の補正率を，近い属性の地点の中央値で補う．地価公示の地点は補正なし（0）．"""
    p = p.copy()
    p["hr"] = r
    p["muni"] = p["X_017"].astype(str)
    p["pref"] = p["muni"].str[:2]
    p["use"] = p["X_001"]
    l2 = p["src"] == "L02"
    for keys in [["muni", "use"], ["pref", "use"], ["pref"]]:
        med = p[l2].groupby(keys)["hr"].transform("median")
        fill = l2 & p["hr"].isna()
        p.loc[fill, "hr"] = med[fill]
    p.loc[l2 & p["hr"].isna(), "hr"] = p.loc[l2, "hr"].median()
    p["hr"] = p["hr"].where(l2, 0.0)
    return p


def zone_mean(zones, kz, p, col):
    """ゾーン内の地点の単純平均．地点がないゾーンは重心の最寄り地点で補完．"""
    j = gpd.sjoin(p, zones[["id", kz, "geometry"]], how="inner", predicate="within")
    res = zones[["id", kz, "geometry"]].copy()
    res["v"] = res["id"].map(j.groupby("id")[col].mean())
    miss = res["v"].isna()
    if miss.any():
        c = res[miss].copy()
        c["geometry"] = c.geometry.centroid
        nn = gpd.sjoin_nearest(c[["id", "geometry"]], p[[col, "geometry"]], how="left")
        res.loc[miss, "v"] = res.loc[miss, "id"].map(nn.groupby("id")[col].mean())
    return res.set_index("id")["v"].fillna(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int, choices=[1996, 2016])
    ap.add_argument("--dir", required=True, help="その年次のゾーン・地価データのフォルダ")
    ap.add_argument("--prev_dir", help="1996年の時点補正に用いる1995年地価調査データ（L02-95）のフォルダ")
    ap.add_argument("--out_dir", default=".", help="出力先フォルダ")
    a = ap.parse_args()

    zones = gpd.read_file(os.path.join(a.dir, "H30_kzone.shp"))
    zones["id"] = range(len(zones))
    kz = "kzone" if "kzone" in zones.columns else zones.columns[0]

    p_all = load_points(a.dir, a.year).to_crs(zones.crs)
    print(f"地価ポイント数：{len(p_all):,}（うち林地 {(p_all['X_001'] == FOREST).sum()}）")
    p = p_all[p_all["X_001"] != FOREST].copy()

    if a.year == 2016:
        r = half_year_rate_2016(p)
    else:
        if not a.prev_dir:
            raise ValueError("1996年の時点補正には --prev_dir（1995年地価調査データのフォルダ）が必要です．")
        r = half_year_rate_1996(p, a.prev_dir)
    n_l2 = (p["src"] == "L02").sum()
    print(f"地価調査の地点 {n_l2:,} のうち，地点ごとの前年価格で補正：{r.notna().sum():,}")
    p = fill_rates(p, r)
    p["price_adj"] = p["price"] * np.exp(p["hr"])
    q = np.exp(p.loc[p["src"] == "L02", "hr"]) - 1
    print("地価調査地点の補正率（7月→1月）の分位点 5/50/95%：",
          np.round(q.quantile([0.05, 0.5, 0.95]).values * 100, 2), "%")

    v = zone_mean(zones, kz, p, "price_adj")
    out = pd.DataFrame({"id": zones["id"].values, kz: zones[kz].values, f"平均地価_{a.year}": v.values})
    os.makedirs(a.out_dir, exist_ok=True)
    f = os.path.join(a.out_dir, f"land_price_{a.year}_average_corrected.csv")
    out.to_csv(f, index=False, encoding="utf-8-sig")
    print("出力：", f)

if __name__ == "__main__":
    main()
