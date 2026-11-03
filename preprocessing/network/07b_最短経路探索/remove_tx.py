# -*- coding: utf-8 -*-
"""
remove_tx.py ― 鉄道ネットワークから TX（つくばエクスプレス）のノード・リンクを削除する（論文 5. (1)）

  python remove_tx.py ./input/rail_network_2015.gpkg ./input/rail_network_2015_noTX.gpkg
  python remove_tx.py ./input/rail_network_2015.gpkg ./input/rail_network_2015_noTX.gpkg --dry_run   （確認のみ）

【判定】
  1. nodes / links の文字列の列のいずれかにキーワード（既定：首都圏新都市鉄道，つくばエクスプレス，常磐新線）を
     含むものを TX のノード・リンクとする
  2. 削除したノードに接続するリンク（乗換リンク等）も削除する
【確認用の出力】（画面と <出力>_removed.txt）
  - キーワードが見つかった列と件数，削除したノードの駅名の一覧
  - キーワードには一致しないが，削除したノードに接続していたため削除したリンク（乗換リンク等）
  - TX のリンクの端点なのにキーワードに一致せず残ったノード（他社線と共用のノードの可能性）
    → 共用ノードの場合，そのノードは残すのが正しい（他社線の駅として必要）
"""
import argparse
import os

import geopandas as gpd
import pandas as pd
import pyogrio

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--keywords", nargs="*", default=["首都圏新都市鉄道", "つくばエクスプレス", "常磐新線", "Tsukuba Express"])
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()

layers = [l[0] for l in pyogrio.list_layers(a.src)]
if not {"nodes", "links"} <= set(layers):
    raise SystemExit(f"{a.src} に nodes / links レイヤーがありません：{layers}")
nodes = gpd.read_file(a.src, layer="nodes")
links = gpd.read_file(a.src, layer="links")
nid = "node_id" if "node_id" in nodes.columns else None
if nid is None:
    raise SystemExit(f"nodes に node_id 列がありません：{list(nodes.columns)}")

report = []


def log(s=""):
    print(s)
    report.append(s)


def match(df, name):
    m = pd.Series(False, index=df.index)
    for c in df.columns:
        if c == "geometry" or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        mc = df[c].astype(str).str.contains("|".join(a.keywords), regex=True, na=False)
        if mc.any():
            log(f"  {name}.{c}：{int(mc.sum())} 件")
        m |= mc
    return m


log(f"入力：{a.src}（nodes {len(nodes):,}，links {len(links):,}）")
log(f"キーワード：{a.keywords}")
log("キーワードが見つかった列：")
mn = match(nodes, "nodes")
ml = match(links, "links")
if not mn.any() and not ml.any():
    raise SystemExit("キーワードに一致するノード・リンクがありません。--keywords を確認してください。")

tx_nodes = set(nodes.loc[mn, nid])
ends = set(links.loc[ml, "from_node"]) | set(links.loc[ml, "to_node"])
incident = links["from_node"].isin(tx_nodes) | links["to_node"].isin(tx_nodes)
drop_l = ml | incident

namecol = next((c for c in ["name", "station_name", "N02_005"] if c in nodes.columns), None)
log(f"\n削除するノード：{len(tx_nodes)}（うち駅 {int(nodes.loc[mn, 'is_station'].astype(bool).sum()) if 'is_station' in nodes else '?'}）")
if namecol:
    log("  " + "，".join(sorted(set(nodes.loc[mn, namecol].astype(str)))))
log(f"削除するリンク：{int(drop_l.sum())}（キーワード一致 {int(ml.sum())}，削除ノードへの接続のみ {int((incident & ~ml).sum())}）")

extra = links[incident & ~ml]
if len(extra):
    log("\nキーワードには一致しないが削除ノードに接続していたため削除するリンク（乗換リンク等）：")
    nm = nodes.set_index(nid)[namecol] if namecol else None
    show = [c for c in ["link_id", "from_node", "to_node", "type", "name", "operator", "route_name", "time_min", "fare"]
            if c in extra.columns]
    ex = extra[show].copy()
    if nm is not None:
        ex["from_name"] = ex["from_node"].map(nm)
        ex["to_name"] = ex["to_node"].map(nm)
    log(ex.to_string(index=False))

shared = ends - tx_nodes
if shared:
    log("\n※ TX のリンクの端点だがキーワードに一致せず残すノード（他社線と共用のノードの可能性．確認してください）：")
    sub = nodes[nodes[nid].isin(shared)]
    log(sub.drop(columns="geometry").to_string(index=False))

nodes2 = nodes[~mn]
links2 = links[~drop_l]
# 残ったリンクの端点がすべて残ったノードにあるか
lost = (~links2["from_node"].isin(nodes2[nid])) | (~links2["to_node"].isin(nodes2[nid]))
if lost.any():
    raise SystemExit(f"端点のノードがないリンクが {int(lost.sum())} 本あります（内部エラー）。")
log(f"\n出力：nodes {len(nodes2):,}（−{len(nodes) - len(nodes2)}），links {len(links2):,}（−{len(links) - len(links2)}）")

if a.dry_run:
    log("（--dry_run のため書き出しません）")
else:
    if os.path.exists(a.dst):
        os.remove(a.dst)
    nodes2.to_file(a.dst, layer="nodes", driver="GPKG")
    links2.to_file(a.dst, layer="links", driver="GPKG")
    for l in layers:                     # その他のレイヤーはそのまま複製
        if l not in ("nodes", "links"):
            gpd.read_file(a.src, layer=l).to_file(a.dst, layer=l, driver="GPKG")
    log(f"→ {a.dst}")
with open(os.path.splitext(a.dst)[0] + "_removed.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
