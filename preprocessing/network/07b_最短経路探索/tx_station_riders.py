# -*- coding: utf-8 -*-
"""
tx_station_riders.py ― AON 配分の結果から TX の駅別乗降客数を集計し，観測値と比較する

  python tx_station_riders.py --stations ./output/result_stations_passenger_pt_assign_2015.csv
         --rail ./input/rail_network_2015.gpkg [--observed observed.csv] [--out tx_station_riders]

  --observed：観測値（列 name, observed．国土数値情報 S12 の2015年の乗降客数など）．あれば比較図も作る
  乗降客数 = 乗車（BOARDING）+ 降車（ALIGHTING）．乗換リンクは乗降に数えない（鉄道の他社線との乗換は含まれない）．
"""
import argparse
import geopandas as gpd
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--stations", required=True)
ap.add_argument("--rail", required=True)
ap.add_argument("--observed")
ap.add_argument("--rail_volumes", help="配分結果の鉄道リンク交通量 result_rail_….csv（他社線からの乗換を乗降に含める場合）")
ap.add_argument("--keywords", default="首都圏新都市鉄道|つくばエクスプレス|常磐新線")
ap.add_argument("--out", default="tx_station_riders")
a = ap.parse_args()

nodes = gpd.read_file(a.rail, layer="nodes")
col = next(c for c in nodes.columns if c != "geometry" and nodes[c].astype(str).str.contains(a.keywords).any())
tx = nodes[nodes[col].astype(str).str.contains(a.keywords)]
namecol = "name" if "name" in tx.columns else next(c for c in tx.columns if "name" in c.lower())
def norm_id(x):
    """ID の表記ゆれ（2585 / 2585.0 / '2585'）を吸収する"""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x).strip()


st = pd.read_csv(a.stations, encoding="utf-8-sig")
print(f"{a.stations}：{len(st):,} 行．モード別の乗降の合計：" + "，".join(f"{k} {v:,.0f}" for k, v in st.groupby("MODE")["TOTAL"].sum().items()))
st = st[st["MODE"].astype(str).str.strip() == "Rail"].copy()
st["ORIG_ID"] = st["ORIG_ID"].map(norm_id)
tx_ids = dict(zip(tx["node_id"].map(norm_id), tx[namecol].astype(str)))
r = st[st["ORIG_ID"].isin(tx_ids)].copy()
print(f"鉄道の駅の行数 {len(st):,}，乗降の合計 {st['TOTAL'].sum():,.0f}．TX のノード {len(tx_ids)} 個のうち一致 {len(r)} 個")
if len(r) == 0:
    print("  例：結果の ORIG_ID", st["ORIG_ID"].head(5).tolist(), "／ TX の node_id", list(tx_ids)[:5])
    if "NAME" in st.columns:
        m = st[st["NAME"].astype(str).isin(set(tx_ids.values()))]
        print(f"  駅名での一致：{len(m)} 行（同名の他社線の駅を含む可能性あり）")
r["name"] = r["ORIG_ID"].map(tx_ids)
order = tx.assign(x=tx.geometry.x).sort_values("x", ascending=False)[namecol].astype(str).tolist()  # つくば（東）→秋葉原（西）
res = (pd.DataFrame({"name": order}).merge(r[["name", "BOARDING", "ALIGHTING", "TOTAL"]], on="name", how="left")
       .rename(columns={"TOTAL": "access_total"}))
for c in ["BOARDING", "ALIGHTING", "access_total"]:
    res[c] = pd.to_numeric(res[c], errors="coerce").fillna(0.0)
res["predicted"] = res["access_total"]

# 他社線との乗換：TX のノードと他のノードを結ぶリンクの交通量（他社線 → TX は乗車，TX → 他社線は降車として数える）
if a.rail_volumes:
    links = gpd.read_file(a.rail, layer="links")
    links["from_id"], links["to_id"] = links["from_node"].map(norm_id), links["to_node"].map(norm_id)
    vol = pd.read_csv(a.rail_volumes, encoding="utf-8-sig")
    vol = dict(zip(vol["ORIG_ID"].map(norm_id), vol["VOLUME"].astype(float)))
    links["vol"] = links["link_id"].map(norm_id).map(vol).fillna(0.0)
    tin = links[~links["from_id"].isin(tx_ids) & links["to_id"].isin(tx_ids)]
    tout = links[links["from_id"].isin(tx_ids) & ~links["to_id"].isin(tx_ids)]
    t_in = tin.groupby(tin["to_id"].map(tx_ids))["vol"].sum()
    t_out = tout.groupby(tout["from_id"].map(tx_ids))["vol"].sum()
    res["transfer_in"] = res["name"].map(t_in).fillna(0.0)
    res["transfer_out"] = res["name"].map(t_out).fillna(0.0)
    res["predicted"] = res["access_total"] + res["transfer_in"] + res["transfer_out"]
    print(f"乗換リンク：他社線→TX {len(tin)} 本（{tin.vol.sum():,.0f}），TX→他社線 {len(tout)} 本（{tout.vol.sum():,.0f}）")
if a.observed:
    ob = pd.read_csv(a.observed, encoding="utf-8-sig")
    res = res.merge(ob[["name", "observed"]], on="name", how="left")
res.to_csv(a.out + ".csv", index=False, encoding="utf-8-sig")
print(res.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
if "transfer_in" in res:
    print(f"\n（参考）乗換を含まない場合の予測値の合計：{res.access_total.sum():,.0f} 人/日")
print(f"\n予測値の合計：{res.predicted.sum():,.0f} 人/日" +
      (f"，観測値の合計：{res.observed.sum():,.0f} 人/日" if "observed" in res else ""))

if "observed" in res:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    jp = [f for f in ["Yu Gothic", "MS Gothic", "Meiryo", "IPAexGothic", "Noto Sans CJK JP"]
          if any(f == x.name for x in font_manager.fontManager.ttflist)]
    if jp:
        plt.rcParams["font.family"] = jp[0]
    # 1段の幅（約 8.5 cm）にそのまま載る大きさで描く（縮小されないので文字がつぶれない）．
    # 駅は路線順（左が都心側の秋葉原）に並べる
    r = res if res.name.iloc[0] == "秋葉原" else res.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    x = range(len(r))
    ax.bar([i - 0.2 for i in x], r.observed / 1000, width=0.4, color="#1f5a91", label="Observed")
    ax.bar([i + 0.2 for i in x], r.predicted / 1000, width=0.4, color="#ef7d31", label="Predicted")
    tate = ["\n".join(nm.replace("ー", "｜")) for nm in r.name]      # 駅名は縦書き（1文字ずつ改行）
    ax.set_xticks(list(x)); ax.set_xticklabels(tate, rotation=0, fontsize=5.5, linespacing=0.95, va="top")
    ax.set_xlim(-0.6, len(r) - 0.4)
    ax.set_ylabel("Passengers/day\n(thousands)", fontsize=6.5); ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="y", lw=0.4, color="0.85"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.2)
    fig.tight_layout(pad=0.2)
    fig.savefig("station_riders-crop.pdf", bbox_inches="tight", pad_inches=0.02)
    print("→ station_riders-crop.pdf")
