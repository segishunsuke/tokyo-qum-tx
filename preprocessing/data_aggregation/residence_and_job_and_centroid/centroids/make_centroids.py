# -*- coding: utf-8 -*-
"""
ゾーン代表点（セントロイド）の作成

各計画基本ゾーンについて，ポリゴン内に中心点が含まれる500mメッシュのうち，
「人口＋従業者数」が最大のメッシュの中心点を代表点とする．

  人口   ：2015年国勢調査 250mメッシュ（tblT000876Q*.txt, T000876001）を500mメッシュに集計
  従業者 ：2014年経済センサス‐基礎調査 500mメッシュ（tblT000842H*.txt, T000842002）
           （国・地方公共団体の事業所を含む全産業．論文初版では2016年経済センサス‐活動調査（民営のみ）を使用）
  いずれも世界測地系（JGD2000）の地域メッシュ．

出力：centroids.gpkg（レイヤ名 'centroids'，EPSG:4326）
  - 記録順（fid順）は H30_kzone.shp の記録順と同じ（0〜614）
  - node_id : H30_kzone.shp の記録順の番号（0始まり）．最短経路探索の o_id / d_id になる
  - kzone   : 計画基本ゾーン番号
  - mesh500 : 代表点としたメッシュのコード
  - pop, emp, activity : そのメッシュの人口・従業者数・合計
  - method  : 'max_mesh'（通常），'fallback_intersect'（中心点が含まれるメッシュがないゾーン）
              'fallback_point'（交差するメッシュもないゾーン：ポリゴン内部の代表点）

使い方：
  python make_centroids.py --zone ../residence/2015/H30_kzone.shp \
      --pop_dir ../residence/2015 --emp_dir ../jobs/2014 --out centroids.gpkg
"""
import argparse
import glob
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

warnings.filterwarnings("ignore")


def read_estat(pattern, col):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    d = pd.concat([pd.read_csv(f, skiprows=[1], dtype=str, encoding="cp932", usecols=["KEY_CODE", col])
                   for f in files], ignore_index=True)
    d = d.drop_duplicates("KEY_CODE")
    d["v"] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    return d[["KEY_CODE", "v"]]


def mesh500_sw(code):
    """500mメッシュコード（9桁）の南西端の経度・緯度（JGD2000）"""
    c = code.astype(str)
    YY = c.str[0:2].astype(int); XX = c.str[2:4].astype(int)
    J = c.str[4].astype(int); I = c.str[5].astype(int)
    j = c.str[6].astype(int); i = c.str[7].astype(int); L = c.str[8].astype(int)
    lat = YY / 1.5 + J / 12.0 + j / 120.0 + (L >= 3) / 240.0
    lon = 100 + XX + I / 8.0 + i / 80.0 + (L % 2 == 0) / 160.0
    return lon, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", required=True)
    ap.add_argument("--pop_dir", required=True)
    ap.add_argument("--emp_dir", required=True)
    ap.add_argument("--out", default="centroids.gpkg")
    a = ap.parse_args()

    zones = gpd.read_file(a.zone)
    zones["node_id"] = range(len(zones))
    kz = "kzone" if "kzone" in zones.columns else zones.columns[0]

    # 人口：250m → 500m（先頭9桁）に集計
    pop = read_estat(os.path.join(a.pop_dir, "tblT000876Q*.txt"), "T000876001")
    pop["KEY_CODE"] = pop["KEY_CODE"].str[:9]
    pop = pop.groupby("KEY_CODE")["v"].sum().rename("pop")
    emp = read_estat(os.path.join(a.emp_dir, "tblT000842H*.txt"), "T000842002").set_index("KEY_CODE")["v"].rename("emp")
    m = pd.concat([pop, emp], axis=1).fillna(0.0)
    m["activity"] = m["pop"] + m["emp"]
    m = m[m["activity"] > 0].reset_index().rename(columns={"index": "mesh500", "KEY_CODE": "mesh500"})
    print(f"500mメッシュ数：{len(m):,}（人口合計 {m['pop'].sum():,.0f}，従業者合計 {m['emp'].sum():,.0f}）")

    lon, lat = mesh500_sw(m["mesh500"])
    cells = gpd.GeoDataFrame(m, geometry=[box(x, y, x + 1 / 160.0, y + 1 / 240.0) for x, y in zip(lon, lat)],
                             crs="EPSG:4612")
    centers = gpd.GeoDataFrame(m, geometry=[Point(x + 1 / 320.0, y + 1 / 480.0) for x, y in zip(lon, lat)],
                               crs="EPSG:4612")

    zp = zones[["node_id", kz, "geometry"]].to_crs("EPSG:4612")
    # 1) 中心点がゾーン内にあるメッシュのうち活動量最大
    j = gpd.sjoin(centers, zp, how="inner", predicate="within")
    best = j.loc[j.groupby("node_id")["activity"].idxmax()].set_index("node_id")
    best["method"] = "max_mesh"
    # 2) 中心点が含まれるメッシュがないゾーン：ゾーンと交差するメッシュのうち活動量最大
    miss = sorted(set(zp["node_id"]) - set(best.index))
    fb = []
    if miss:
        ji = gpd.sjoin(cells, zp[zp["node_id"].isin(miss)], how="inner", predicate="intersects")
        if len(ji):
            b2 = ji.loc[ji.groupby("node_id")["activity"].idxmax()].set_index("node_id")
            b2["geometry"] = centers.set_index("mesh500").loc[b2["mesh500"], "geometry"].values
            b2["method"] = "fallback_intersect"
            fb.append(b2)
    rows = pd.concat([best] + fb) if fb else best
    # 3) それでもないゾーン：ポリゴン内部の代表点
    rest = sorted(set(zp["node_id"]) - set(rows.index))
    if rest:
        r3 = zp.set_index("node_id").loc[rest].copy()
        r3["geometry"] = r3.geometry.representative_point()
        for c in ["mesh500", "pop", "emp", "activity"]:
            r3[c] = np.nan
        r3["method"] = "fallback_point"
        rows = pd.concat([rows, r3])

    rows = rows.reindex(zp["node_id"].values)
    out = gpd.GeoDataFrame({
        "node_id": zp["node_id"].values,
        "kzone": zp[kz].values,
        "name": zp[kz].astype(str).values,
        "mesh500": rows["mesh500"].values,
        "pop": rows["pop"].values,
        "emp": rows["emp"].values,
        "activity": rows["activity"].values,
        "method": rows["method"].values,
    }, geometry=gpd.GeoSeries(rows.geometry.values, crs="EPSG:4612")).to_crs("EPSG:4326")

    # 検査：代表点がゾーンポリゴン内にあるか
    inside = gpd.sjoin(out[["node_id", "geometry"]], zones[["node_id", "geometry"]].rename(columns={"node_id": "zid"}).to_crs("EPSG:4326"),
                       how="left", predicate="within")
    inside = inside[~inside.index.duplicated()]
    out["inside_zone"] = (inside["zid"].values == out["node_id"].values)

    if os.path.exists(a.out):
        os.remove(a.out)
    out.to_file(a.out, layer="centroids", driver="GPKG")
    print(f"出力：{a.out}（{len(out)} 点）")
    print("代表点の決め方：", out["method"].value_counts().to_dict())
    print("ゾーン外の代表点：", int((~out["inside_zone"]).sum()))
    return out


if __name__ == "__main__":
    main()
