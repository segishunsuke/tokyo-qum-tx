# -*- coding: utf-8 -*-
"""
make_pt_od.py ― TX 駅別乗降客数の推計（論文 5. (1) a)）のための公共交通 OD 表の作成

現実の均衡（With TX，2015年）の通勤 OD 交通量 L λ_ij（片道，就業者ベース）に，
交通モード分担モデル（4. (6) a)）の公共交通の選択確率を乗じ，
  ・往復を考慮して 2 倍
  ・一日あたりに換算するため 年間平均出勤日数 224.4 / 365 を乗じる
したものを，search_passenger_separated.py の入力 ./input/od_passenger.csv（o_id, d_id, volume）として出力する．
ゾーン内の OD（i = j）はネットワークに配分されないので除く．

使い方：
  python make_pt_od.py --od "...\\estimation_patched\\cases\\base\\output\\OD_demand[1]_predict.csv"
         --cost_dir ./output --estimates "./モード分担・重力モデル推定/result/modal_share_estimates.csv"
         --out ./input/od_passenger.csv
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--od", required=True, help="QUM の simulation の出力 OD_demand[1]_predict.csv（列 i, j, demand）")
ap.add_argument("--cost_dir", default="./output")
ap.add_argument("--year", type=int, default=2015)
ap.add_argument("--estimates", required=True, help="modal_share_pmle.py の出力 modal_share_estimates.csv")
ap.add_argument("--work_days", type=float, default=224.4)
ap.add_argument("--out", default="./input/od_passenger.csv")
a = ap.parse_args()

od = pd.read_csv(a.od)
S = int(max(od["i"].max(), od["j"].max())) + 1


def load_cost(path):
    c = pd.read_csv(path, encoding="utf-8-sig")
    M = np.full((S, S), np.inf)
    M[c.iloc[:, 0].astype(int), c.iloc[:, 1].astype(int)] = pd.to_numeric(c.iloc[:, 2], errors="coerce")
    return M


car = load_cost(f"{a.cost_dir}/od_min_cost_passenger_car_{a.year}.csv")
pt = load_cost(f"{a.cost_dir}/od_min_cost_passenger_pt_{a.year}.csv")
rc, rp = car + car.T, pt + pt.T                      # 往復の一般化費用
est = pd.read_csv(a.estimates, encoding="utf-8-sig").set_index("param")["estimate"]
g0, g1 = float(est["gamma0"]), float(est["gamma1"])
P_pt = 1.0 / (1.0 + np.exp(-(g0 + g1 * (rp - rc))))
P_pt[~np.isfinite(rp)] = 0.0                         # 公共交通で到達不能
P_pt[np.isfinite(rp) & ~np.isfinite(rc)] = 1.0

Q = np.zeros((S, S))
Q[od["i"].to_numpy(), od["j"].to_numpy()] = od["demand"].to_numpy()
np.fill_diagonal(Q, 0.0)
V = Q * P_pt * 2.0 * a.work_days / 365.0
oo, dd = np.nonzero(V > 0)
pd.DataFrame({"o_id": oo, "d_id": dd, "volume": V[oo, dd]}).to_csv(a.out, index=False, encoding="utf-8")
print(f"γ0 = {g0:.6f}，γ1 = {g1:.6e}")
print(f"通勤 OD（ゾーン間，片道）：{Q.sum():,.0f} 人，公共交通の分担率 {(Q * P_pt).sum() / Q.sum():.3f}")
print(f"一日あたりの公共交通トリップ（往復）：{V.sum():,.0f} → {a.out}（{len(oo):,} OD ペア）")
