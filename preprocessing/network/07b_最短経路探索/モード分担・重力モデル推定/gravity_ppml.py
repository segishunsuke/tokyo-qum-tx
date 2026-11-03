# -*- coding: utf-8 -*-
"""
gravity_ppml.py ― 発地・着地固定効果付き重力モデルの PPML 推定（＝二重制約型重力モデル）

【モデル】
  E[Q_ij] = exp(a_i + b_j − β τ_ij)   (i ≠ j)
  Q_ij : 2018年東京都市圏PT調査の計画基本ゾーン間 OD 交通量（ゼロ交通量の OD も含む）
  τ_ij : 往復の合成一般化費用（modal_share_pmle.py の tau_2018.csv）

  発地・着地の固定効果を含むポアソン疑似最尤推定の一階条件は
    Σ_j Q_ij = Σ_j Q̂_ij（発生量の保存），Σ_i Q_ij = Σ_i Q̂_ij（集中量の保存），
    Σ_ij τ_ij Q_ij = Σ_ij τ_ij Q̂_ij（総費用の保存）
  であり，二重制約型重力モデルの最尤推定と一致する．
  β を与えれば固定効果は Furness 法（バランシング）で決まるので，β は総費用の保存条件の
  1次元の求根で求める．

【推定する式】
  commute  : 目的「自宅－勤務」，β = θκ        （本文 4. (7)）
  business : 目的「勤務・業務」，β = δ^A       （本文 4. (9)）
  private  : 目的「自宅－私事」，β = δ^B       （本文 4. (9)）
  いずれも同一ゾーン内（i = j）の交通は除外する．

【標準誤差】
  PPML の標準的なロバスト（サンドイッチ型）標準誤差．固定効果は重み μ の下での
  二元配置の射影（交互平均化）で除去して計算する（Frisch–Waugh–Lovell の定理の GLM 版）．

【あわせて計算する量】（commute のみ）
  τ̄      : 同一ゾーン内を除く「自宅－勤務」トリップの平均往復費用（トリップ数で重み付け）
  τ_ii の係数 : τ̄ / sqrt(可住地面積の合計)        （本文 4. (8)）
  κ       : w̄ [1 − exp(−κ τ̄)] = τ̄ / 10^6 × 出勤日数 を解く  （本文 4. (7) b)）
  θ       : θκ / κ

【減衰の関数形】（--form）
  exp   : E[Q_ij] = exp(a_i + b_j − β τ_ij)         （本文の定式化．β の単位は 1/円）
  power : E[Q_ij] = exp(a_i + b_j − α ln c_ij)      （べき乗型 c_ij^{−α}．WebTAG・Graham and Gibbons (2019) との比較用）
  power の費用 c_ij は --cost で選ぶ：
    tau      … 往復の合成一般化費用 τ_ij（WebTAG の適用方法に対応：一般化費用に α を適用）
    distance … セントロイド間の直線距離 [km]（Graham and Gibbons の推定方法に対応．--centroids が必要）

使い方：
  python gravity_ppml.py --tau ./result/tau_2018.csv --area 9861.09
  python gravity_ppml.py --tau ./result/tau_2018.csv --form power --cost tau
  python gravity_ppml.py --tau ./result/tau_2018.csv --form power --cost distance --centroids ../input/centroids.gpkg
出力（./result/）：gravity_estimates.csv, gravity_summary.txt
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

PURPOSES = {
    "commute": ("自宅－勤務", "θκ"),
    "business": ("勤務・業務", "δA"),
    "private": ("自宅－私事", "δB"),
}


def clean_kzone(x):
    if pd.isna(x):
        return -1
    try:
        return int(str(x).replace(":", "").strip())
    except ValueError:
        return -1


def clean_trip(x):
    if pd.isna(x):
        return 0.0
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def load_od(d1, purpose, kzone_to_id, n):
    t = d1[d1["目的種類"].astype(str).str.strip() == purpose]
    o = t["発地"].apply(clean_kzone).map(kzone_to_id)
    d = t["着地"].apply(clean_kzone).map(kzone_to_id)
    q = t["計"].apply(clean_trip)
    ok = o.notna() & d.notna()
    Q = np.zeros((n, n))
    np.add.at(Q, (o[ok].astype(int).to_numpy(), d[ok].astype(int).to_numpy()), q[ok].to_numpy())
    return Q


def furness(K, O, D, mask, tol=1e-10, max_iter=5000):
    """Σ_j A_i B_j K_ij = O_i, Σ_i A_i B_j K_ij = D_j を満たす A, B を求める"""
    B = np.ones(K.shape[1])
    for _ in range(max_iter):
        A = O / np.maximum(K @ B, 1e-300)
        B_new = D / np.maximum(K.T @ A, 1e-300)
        if np.max(np.abs(B_new - B) / np.maximum(B, 1e-300)) < tol:
            B = B_new
            break
        B = B_new
    A = O / np.maximum(K @ B, 1e-300)
    return A[:, None] * B[None, :] * K


SCALE = "exp"


def fit(Q, tau, mask):
    """二重制約型重力モデル（=二元固定効果付き PPML）の推定"""
    rows = mask.any(axis=1) & (np.where(mask, Q, 0).sum(axis=1) > 0)
    cols = mask.any(axis=0) & (np.where(mask, Q, 0).sum(axis=0) > 0)
    Q = Q[np.ix_(rows, cols)]; tau = tau[np.ix_(rows, cols)]; m = mask[np.ix_(rows, cols)]
    Q = np.where(m, Q, 0.0)
    O, D = Q.sum(axis=1), Q.sum(axis=0)
    tmin = np.min(np.where(m, tau, np.inf), axis=1, keepdims=True)   # 行ごとの基準化（数値安定化）
    obs_cost = np.sum(np.where(m, tau, 0) * Q)

    def mu_of(beta):  # tau には exp 型なら τ，power 型なら ln c が入る
        K = np.where(m, np.exp(-beta * (tau - tmin)), 0.0)
        return furness(K, O, D, m)

    def g(beta):  # 総費用の保存条件
        return obs_cost - np.sum(np.where(m, tau, 0) * mu_of(beta))

    lo, hi = (1e-7, 1e-2) if SCALE == "exp" else (1e-4, 0.5)
    while g(hi) < 0 and hi < (1 if SCALE == "exp" else 100):
        hi *= 2
    beta = brentq(g, lo, hi, xtol=1e-14, rtol=1e-12)
    mu = mu_of(beta)

    # ロバスト標準誤差：重み μ の下で τ から二元固定効果を除去 → Var(β) = Σ τ̃²(Q−μ)² / (Σ μ τ̃²)²
    tt = np.where(m, tau, 0.0)
    w = np.where(m, mu, 0.0)
    a = np.zeros(tt.shape[0]); b = np.zeros(tt.shape[1])
    for _ in range(2000):
        a_new = np.sum(w * (tt - b[None, :]), axis=1) / np.maximum(w.sum(axis=1), 1e-300)
        b_new = np.sum(w * (tt - a_new[:, None]), axis=0) / np.maximum(w.sum(axis=0), 1e-300)
        conv = max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b)))
        a, b = a_new, b_new
        if conv < 1e-9 * np.max(np.abs(tt)):
            break
    tr = np.where(m, tt - a[:, None] - b[None, :], 0.0)
    info = np.sum(w * tr ** 2)
    se_robust = np.sqrt(np.sum(tr ** 2 * (Q - mu) ** 2 * m)) / info
    se_model = 1.0 / np.sqrt(info)      # ポアソン分布を仮定した場合（参考）
    # 当てはまりの比較用：ポアソン疑似対数尤度（Q の階乗の項を除く）と逸脱度．
    # 同じ OD データ・同じパラメータ数（固定効果 + 減衰1つ）なので，関数形の間で値を直接比較できる．
    qm = np.where(m, Q, 0.0); mm = np.where(m, mu, 1.0)
    pll = float(np.sum(np.where(m, qm * np.log(np.maximum(mm, 1e-300)) - mm, 0.0)))
    dev = float(2.0 * np.sum(np.where(m & (qm > 0), qm * np.log(np.maximum(qm, 1e-300) / mm), 0.0)
                             - np.where(m, qm - mm, 0.0)))
    return {
        "beta": beta, "se_robust": se_robust, "se_model": se_model,
        "n_obs": int(m.sum()), "n_zero": int((m & (Q == 0)).sum()),
        "n_origins": int(rows.sum()), "n_dests": int(cols.sum()),
        "trips": float(Q.sum()), "mean_cost_obs": obs_cost / Q.sum(),
        "pseudo_ll": pll, "deviance": dev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="d-1.csv")
    ap.add_argument("--kzone", default="kzone.csv")
    ap.add_argument("--tau", default="./result/tau_2018.csv")
    ap.add_argument("--area", type=float, default=9861.09, help="東京都市圏の可住地面積の合計 [km2]")
    ap.add_argument("--wage", type=float, default=4.5141, help="平均年収 [百万円]")
    ap.add_argument("--days", type=float, default=224.4, help="年間出勤日数")
    ap.add_argument("--out", default="./result")
    ap.add_argument("--form", choices=["exp", "power"], default="exp")
    ap.add_argument("--cost", choices=["tau", "distance"], default="tau", help="power 型の費用変数")
    ap.add_argument("--centroids", help="--cost distance のときのセントロイド（node_id = kzone.csv の id）")
    a = ap.parse_args()
    global SCALE
    SCALE = a.form
    os.makedirs(a.out, exist_ok=True)

    kz = pd.read_csv(a.kzone)
    n = len(kz)
    kzone_to_id = dict(zip(kz["kzone"], kz["id"]))
    tdf = pd.read_csv(a.tau, encoding="utf-8-sig")
    tau = np.full((n, n), np.inf)
    tau[tdf["o_id"].astype(int), tdf["d_id"].astype(int)] = tdf["tau"].to_numpy()
    mask = np.isfinite(tau) & ~np.eye(n, dtype=bool)
    tag = ""
    if a.form == "power":
        if a.cost == "distance":
            import geopandas as gpd
            c = gpd.read_file(a.centroids).sort_values("node_id")
            if len(c) != n or not (c["node_id"].astype(int).to_numpy() == np.arange(n)).all():
                raise SystemExit("centroids の node_id が kzone.csv の id（0〜n−1）と一致しません。")
            lon, lat = np.radians(c.geometry.x.to_numpy()), np.radians(c.geometry.y.to_numpy())
            dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
            h = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
            cost = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
            mask = mask & (cost > 0)
        else:
            cost = tau
        tau = np.where(mask, np.log(np.where(mask, cost, 1.0)), np.inf)   # 以降は ln c を費用変数として扱う
        tag = f"_power_{a.cost}"
        for k in PURPOSES:
            PURPOSES[k] = (PURPOSES[k][0], "α")

    d1 = pd.read_csv(a.trips, skiprows=3, dtype=str, encoding="cp932")
    rows, lines = [], []
    for key, (purpose, label) in PURPOSES.items():
        Q = load_od(d1, purpose, kzone_to_id, n)
        r = fit(Q, tau, mask)
        r.update({"model": key, "purpose": purpose, "param": label, "form": a.form,
                  "cost": a.cost if a.form == "power" else "tau"})
        rows.append(r)
        lines.append(f"[{key}] {purpose}  {label} = {r['beta']:.6e}  "
                     f"(ロバストSE {r['se_robust']:.4e}, ポアソンSE {r['se_model']:.4e})  "
                     f"擬似対数尤度 {r['pseudo_ll']:,.1f}，逸脱度 {r['deviance']:,.1f}  "
                     f"観測 {r['n_obs']:,}（うちゼロ {r['n_zero']:,}），発地 {r['n_origins']}，着地 {r['n_dests']}，"
                     + (f"トリップ {r['trips']:,.0f}，平均往復費用 {r['mean_cost_obs']:,.2f} 円" if a.form == "exp"
                        else f"トリップ {r['trips']:,.0f}，ln(費用) の平均 {r['mean_cost_obs']:.4f}"))
        if key == "commute" and a.form == "exp":
            tbar = r["mean_cost_obs"]
            coef_ii = tbar / np.sqrt(a.area)
            rhs = tbar / 1e6 * a.days
            kappa = -np.log(1.0 - rhs / a.wage) / tbar
            theta = r["beta"] / kappa
            extra = (f"  τ̄ = {tbar:,.2f} 円，τ_ii = {coef_ii:.4f} × sqrt(G_i)，"
                     f"κ = {kappa:.6e}，θ = θκ/κ = {theta:.4f}")
            lines.append(extra)
            r.update({"tau_bar": tbar, "coef_tau_ii": coef_ii, "kappa": kappa, "theta": theta})
    for l in lines:
        print(l)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"gravity_estimates{tag}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(a.out, f"gravity_summary{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
