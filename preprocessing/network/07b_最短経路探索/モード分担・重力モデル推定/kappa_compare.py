# -*- coding: utf-8 -*-
"""
kappa_compare.py ― 通勤不効用係数 κ の設定方法の比較

モデル上の通勤費用の金銭換算値（年額）は w̄ [1 − exp(−κ τ)]（τ：往復の一般化費用 [円/日]），
実勢の年間通勤費用は τ × days / 10^6 [百万円]．指数関数のため両者を全ての τ で一致させることはできず，
どこで一致させるかにより κ が変わる．

  (A) 平均費用で一致：     w̄ [1 − exp(−κ τ̄)] = τ̄ d / 10^6            （初版・現行）
  (B) τ = 0 での限界で一致： w̄ κ = d / 10^6                            （(A) の線形近似と同じ）
  (C) トリップ平均で一致：   Σ q_ij w̄ [1 − exp(−κ τ_ij)] / Σ q_ij = τ̄ d / 10^6
  (D) 限界評価のトリップ平均で一致： Σ q_ij w̄ κ exp(−κ τ_ij) / Σ q_ij = d / 10^6
       （τ が 1 円/日 変化したときの年間の金銭換算値が，平均的に d 円となる）

  q_ij：2018年PT調査の「自宅－勤務」トリップ（同一ゾーン内を除く），τ_ij：tau_2018.csv

使い方：python kappa_compare.py --tau ./result/tau_2018.csv --theta_kappa 7.117487e-04
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def clean_kzone(x):
    try:
        return int(str(x).replace(":", "").strip())
    except ValueError:
        return -1


def clean_trip(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--trips", default="d-1.csv")
ap.add_argument("--kzone", default="kzone.csv")
ap.add_argument("--tau", default="./result/tau_2018.csv")
ap.add_argument("--theta_kappa", type=float, default=7.117487e-04)
ap.add_argument("--wage", type=float, default=4.5141)
ap.add_argument("--days", type=float, default=224.4)
a = ap.parse_args()

kz = pd.read_csv(a.kzone); n = len(kz)
k2i = dict(zip(kz["kzone"], kz["id"]))
td = pd.read_csv(a.tau, encoding="utf-8-sig")
tau = np.full((n, n), np.nan); tau[td["o_id"].astype(int), td["d_id"].astype(int)] = td["tau"].to_numpy()
d1 = pd.read_csv(a.trips, skiprows=3, dtype=str, encoding="cp932")
t = d1[d1["目的種類"].astype(str).str.strip() == "自宅－勤務"]
o = t["発地"].apply(clean_kzone).map(k2i); d = t["着地"].apply(clean_kzone).map(k2i)
q = t["計"].apply(clean_trip)
ok = o.notna() & d.notna() & (o != d)
o = o[ok].astype(int).to_numpy(); d = d[ok].astype(int).to_numpy(); q = q[ok].to_numpy()
tij = tau[o, d]; keep = np.isfinite(tij) & (q > 0); tij, q = tij[keep], q[keep]
W, dd = a.wage, a.days
tbar = np.sum(q * tij) / q.sum()
c = dd / 1e6                                   # 1円/日 の年額 [百万円]

kA = -np.log(1 - tbar * c / W) / tbar
kB = c / W
kC = brentq(lambda k: np.sum(q * W * (1 - np.exp(-k * tij))) / q.sum() - tbar * c, 1e-8, 1e-2)
fD = lambda k: np.sum(q * W * k * np.exp(-k * tij)) / q.sum() - c
# f_D は κ について単峰（κ が小さい側の解を採用する）．最大値が c に届かない場合は解なし
kk = np.geomspace(kB, 10.0 / tbar, 400); fv = np.array([fD(k) for k in kk])
kD = brentq(fD, kB, kk[np.argmax(fv)]) if fv.max() > 0 else np.nan
if np.isnan(kD):
    print("(D) は解なし：限界評価のトリップ平均の最大値が実勢値に届かない")

ix = np.argsort(tij); cw = np.cumsum(q[ix]) / q.sum()
pct = [tij[ix][np.searchsorted(cw, p)] for p in (0.1, 0.5, 0.9)]
print(f"トリップ {q.sum():,.0f}，τ̄ = {tbar:,.2f} 円，τ のトリップ重み付き 10/50/90% 点：{np.round(pct, 0)}")
print(f"{'方法':<28}{'κ':>14}{'θ=θκ/κ':>10}{'平均年額/実勢':>14}{'τ̄での限界評価/実勢':>20}")
for name, k in [("(A) 平均費用で一致（現行）", kA), ("(B) τ=0 の限界で一致", kB),
                ("(C) トリップ平均で一致", kC), ("(D) 限界評価のトリップ平均", kD)]:
    if np.isnan(k):
        continue
    mean_ratio = np.sum(q * W * (1 - np.exp(-k * tij))) / q.sum() / (tbar * c)
    marg_ratio = W * k * np.exp(-k * tbar) / c
    print(f"{name:<28} {k} {a.theta_kappa / k} {mean_ratio:>14.3f}{marg_ratio:>20.3f}")
