# -*- coding: utf-8 -*-
"""
compare_decay.py ― 集積の外部性の距離減衰の関数形による，TX の「立地固定の集積効果」の違いの概算

立地（従業者分布）と賃金を基準ケースの現実の均衡に固定し，TX による生産性の変化
  Δln A_i = η^A Δln Ω^A_i ,  Ω^A_i = Σ_j L^M_j f(τ_ij)
を，次の減衰関数で比較する（τ_ii = 係数 × sqrt(G_i) は両シナリオ共通）．
  exp   : f = exp(−δ^A τ)          （本研究．δ^A は input_data.xlsx の値）
  power : f = τ^(−α)               （α は --alpha で与える．例：τ ベースの重力モデル 2.45，WebTAG 1.65）
生産性の上昇による所得の増加の一次近似 Σ_i w_i L^M_i Δln A_i を出力する（立地固定の集積効果 (b)−(a) の目安）．

  python compare_decay.py --case ..\\cases\\base --alpha 2.45 1.65
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True, help="cases/base のフォルダ")
ap.add_argument("--alpha", type=float, nargs="+", default=[2.45, 1.65])
ap.add_argument("--t", type=int, default=1)
a = ap.parse_args()
inp, out = os.path.join(a.case, "input"), os.path.join(a.case, "output")
par = pd.read_excel(os.path.join(inp, "input_data.xlsx"), sheet_name="parameters", header=None)
P = dict(zip(par[0].astype(str).str.strip(), par[1]))
S, etaA, dA, c_ii = int(P["S"]), float(P["eta_A"]), float(P["delta_A"]), float(P["cost_per_root_area"])
Lbar = float(P[f"Lbar[{a.t}]"])
G = pd.read_excel(os.path.join(inp, "input_data.xlsx"), sheet_name="area", header=None).iloc[1:1 + S, 1].astype(float).to_numpy()


def tbl(name):
    d = pd.read_csv(os.path.join(out, name))
    return d[[c for c in d.columns if c.startswith("Period")]].to_numpy(float)[:, a.t]


def tau(name):
    d = pd.read_csv(os.path.join(inp, name))
    M = np.zeros((S, S))
    M[d.iloc[:, 0].astype(int), d.iloc[:, 1].astype(int)] = d.iloc[:, 2].astype(float)
    np.fill_diagonal(M, c_ii * np.sqrt(G))
    return M


LM = tbl("lam_M_predict.csv") * Lbar
w = tbl("w_predicted.csv")
tR, tC = tau(f"Real_tau[{a.t}].csv"), tau(f"CF_tau[{a.t}].csv")
bill = w * LM                                   # 百万円


def report(label, f):
    dlnO = np.log((f(tR) @ LM) / (f(tC) @ LM))
    gain = etaA * np.sum(bill * dlnO)
    print(f"{label:28s} Δln Ω の従業者加重平均 {np.sum(LM * dlnO) / LM.sum():.3e}，"
          f"最大 {dlnO.max():.3e}，所得増の近似 {gain / 100:,.1f} 億円/年")
    return gain


print(f"η^A = {etaA}，δ^A = {dA:.4e}，従業者 {LM.sum():,.0f} 人")
g0 = report("exp（本研究）", lambda t: np.exp(-dA * t))
for al in a.alpha:
    g = report(f"power α = {al}", lambda t, al=al: t ** (-al))
    print(f"{'':28s} → exp との比 {g / g0:.2f}")
d = tC - tR
m = d > 1e-9
print(f"τ が減った OD の τ（TX あり）：中央値 {np.median(tR[m]):,.0f} 円，従業者の多い着地への加重平均 "
      f"{np.average(tR[m], weights=np.broadcast_to(LM[None, :], tR.shape)[m]):,.0f} 円．"
      f"exp と power の限界的な弾力性が等しくなる τ = α/δ^A：" + "，".join(f"{al / dA:,.0f} 円（α={al}）" for al in a.alpha))
