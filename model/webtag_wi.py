# -*- coding: utf-8 -*-
"""
webtag_wi.py ― WebTAG（TAG unit A2.4）流の集積の経済による wider impacts（static clustering）の簡易計算

  WI = Σ_i E_i × GDPW × [ (ED_i^A / ED_i^B)^ρ − 1 ]
  ED_i = Σ_j E_j f(τ_ij)，f(τ) = τ^(−α)（--tau0 を指定すると min(1, (τ/τ0)^(−α))，QUM の切り捨て付きべき乗型）
  A = TX あり（Real_tau[t].csv），B = TX なし（CF_tau[t].csv）．E は両方で同じ（立地固定）．
  ゾーン内の費用 τ_ii は tau_ii.csv（QUM と同じ値）で両方とも上書きする．
  f が純粋なべき乗なら τ の単位（円・分）は ED の比に影響しない．

  全産業集計・全ゾーン共通の GDPW の簡易計算．GDPW は次の候補をすべて計算する：
    (1) GDP / 就業者数        （日本全体，2015年）
    (2) GDP / 雇用者数        （同）
    (3) w̄ / 雇用者報酬の対GDP比 （モデルの平均賃金を GDP ベースに換算）
    (4) w̄ / α                （モデル上の就業者一人当たり生産額．α は生産関数の労働分配率）
  (4) はモデルと同じ「生産額」の定義．(1)〜(3) との差は主に資本所得（モデルでは資本は超過レント
  ゼロで調整され，便益に入らない）と固定資本減耗・間接税．

  --check_rho, --check_alpha を与えると，同じ式で QUM の推定値を使った値も計算する．
  これは QUM の (b) 推定値・立地固定 − (a) η=0・立地固定 の便益（GDPW = (4) のとき）に一次近似で一致するはず．

使い方：
  python webtag_wi.py --input cases/base/input --period 1 ^
      --check_rho 0.1474 --check_alpha 1.3377 --check_tau0 422.70041242004663
"""
import argparse
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="cases/base/input", help="input_data.xlsx，CF_tau[t].csv，Real_tau[t].csv があるフォルダ")
ap.add_argument("--tau_ii", default=None, help="既定：<input>/tau_ii.csv")
ap.add_argument("--period", type=int, default=1, help="時点（0 = 1995, 1 = 2015）")
ap.add_argument("--rho", type=float, default=0.043)
ap.add_argument("--alpha", type=float, default=1.655)
ap.add_argument("--tau0", type=float, default=None, help="指定すると切り捨て付きべき乗型")
ap.add_argument("--wage", type=float, default=4.514129, help="モデルの平均賃金 [百万円/年]")
# 日本全体 2015年（暦年）．内閣府「国民経済計算」（2015年基準），総務省「労働力調査」．値は要確認
ap.add_argument("--gdp", type=float, default=538032.3, help="名目GDP [10億円]")
ap.add_argument("--comp", type=float, default=260613.9, help="雇用者報酬 [10億円]")
ap.add_argument("--workers", type=float, default=6401.0, help="就業者数 [万人]")
ap.add_argument("--employees", type=float, default=5640.0, help="雇用者数 [万人]")
ap.add_argument("--labor_share", type=float, default=None, help="生産関数の労働分配率 α（既定：input_data.xlsx から）")
ap.add_argument("--check_rho", type=float, default=None)
ap.add_argument("--check_alpha", type=float, default=None)
ap.add_argument("--check_tau0", type=float, default=None)
ap.add_argument("--E_file", default=None,
                help="従業者数の分布を読む CSV（例：cases/base/output/lam_M_CF.csv．TX なしの均衡で評価する場合）")
ap.add_argument("--wage_file", default=None,
                help="ゾーン別の賃金の CSV（例：cases/base/output/w_predicted.csv）．指定すると GDPW (4) をゾーン別の w_i/α にする")
ap.add_argument("--out", default="webtag_wi_zones.csv")
a = ap.parse_args()

xlsx = os.path.join(a.input, "input_data.xlsx")
par = pd.read_excel(xlsx, sheet_name="parameters", header=None)
vals = par.iloc[1:, :2].reset_index(drop=True)
S = int(float(vals.iloc[0, 1])); T = int(float(vals.iloc[1, 1]))
names = vals.iloc[:, 0].astype(str).str.strip().tolist()
v = vals.iloc[:, 1].tolist()
# README の順：S, T, τ_ii の係数, α, β, μ, θκ, δA, δB, θ, ηA, ηB, Abar, Hbar, pC, Lbar_t
labor_share = a.labor_share if a.labor_share is not None else float(v[3])
Lbar = float(v[15 + a.period])
print(f"input_data.xlsx：S = {S}，T = {T}，{names[3]} = {labor_share!r}（生産関数の労働分配率として使用），"
      f"{names[15 + a.period]} = {Lbar!r}")

lm = pd.read_excel(xlsx, sheet_name="lam_M", header=None).iloc[1:1 + S, 1 + a.period].to_numpy(dtype=float)


def read_period_csv(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    cols = [c for c in d.columns if str(c).startswith("Period")]
    x = d[cols[a.period]] if cols else d.iloc[:, 1 + a.period] if d.shape[1] > 1 + a.period else d.iloc[:, -1]
    x = x.to_numpy(dtype=float)
    if len(x) != S:
        raise SystemExit(f"{path} の行数 {len(x)} がゾーン数 {S} と一致しません。")
    return x


if a.E_file:
    lm = read_period_csv(a.E_file)
    print(f"従業者数の分布：{a.E_file}")
E = lm / lm.sum() * Lbar
print(f"従業者数（時点 {a.period}）：合計 {E.sum():,.0f}（lam_M のシェア × Lbar）")


def read_tau(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    m = np.full((S, S), np.nan)
    m[d.iloc[:, 0].astype(int), d.iloc[:, 1].astype(int)] = d.iloc[:, 2].to_numpy(dtype=float)
    return m


tii = pd.read_csv(a.tau_ii or os.path.join(a.input, "tau_ii.csv"), encoding="utf-8-sig")
tii = tii.set_index("id")["tau_ii"].reindex(range(S)).to_numpy(dtype=float)
tauB = read_tau(os.path.join(a.input, f"CF_tau[{a.period}].csv"))
tauA = read_tau(os.path.join(a.input, f"Real_tau[{a.period}].csv"))
for m in (tauA, tauB):
    np.fill_diagonal(m, tii)
    if np.isnan(m).any():
        raise SystemExit(f"τ に欠損があります（{int(np.isnan(m).sum())} 個）。")
dt = tauB - tauA
print(f"τ の低下（TX なし − あり）：ペア数 {int((dt > 1e-9).sum()):,}，最大 {dt.max():.1f} 円，"
      f"上昇したペア {int((dt < -1e-9).sum())}")


def ed(tau, alpha, tau0):
    f = tau ** (-alpha) if tau0 is None else np.minimum(1.0, (tau / tau0) ** (-alpha))
    return f @ E


def wi(rho, alpha, tau0):
    r = ed(tauA, alpha, tau0) / ed(tauB, alpha, tau0)
    return r, E * (r ** rho - 1.0)       # GDPW を掛ける前（人 × 比）


gdpw = {
    "(1) GDP/就業者": a.gdp * 1e9 / (a.workers * 1e4) / 1e6,
    "(2) GDP/雇用者": a.gdp * 1e9 / (a.employees * 1e4) / 1e6,
    "(3) w̄/雇用者報酬比": a.wage / (a.comp / a.gdp),
    "(4) w̄/α（モデル）": a.wage / labor_share,
}
if a.wage_file:
    wz = read_period_csv(a.wage_file)
    print(f"ゾーン別の賃金：{a.wage_file}（従業者で加重した平均 {np.average(wz, weights=E):.4f}）")
    gdpw["(5) w_i/α（ゾーン別）"] = wz / labor_share
print(f"\n雇用者報酬の対GDP比 = {a.comp / a.gdp:.4f}，雇用者報酬/雇用者 = {a.comp * 1e9 / (a.employees * 1e4) / 1e6:.3f} 百万円")
print("GDPW [百万円/人・年]：" + "，".join(f"{k} {np.average(x, weights=E) if np.ndim(x) else x:.3f}" for k, x in gdpw.items()))

rows = [("WebTAG", a.rho, a.alpha, a.tau0)]
if a.check_rho is not None:
    rows.append(("QUM 推定値", a.check_rho, a.check_alpha if a.check_alpha else a.alpha, a.check_tau0))
out = pd.DataFrame({"id": np.arange(S), "E": E})
print("\nWI [億円/年]")
for lab, rho, al, t0 in rows:
    r, x = wi(rho, al, t0)
    out[f"ED_ratio_{lab}"] = r
    out[f"WI_{lab}_GDPW4_oku"] = x * gdpw["(4) w̄/α（モデル）"] / 100.0
    f = "τ^(-α)" if t0 is None else f"min(1,(τ/{t0:.1f})^(-α))"
    print(f"  {lab}：ρ = {rho}，α = {al}，f = {f}．ED の比：平均 {np.average(r - 1, weights=E):.3e}，最大 {r.max() - 1:.3e}")
    for k, g in gdpw.items():
        print(f"      GDPW {k:<16}：{np.sum(x * g) / 100.0:10.1f}")
out.to_csv(a.out, index=False, encoding="utf-8-sig")
print(f"\nゾーン別 → {a.out}")
print("（QUM との照合：--E_file に lam_M_CF.csv，--wage_file にゾーン別賃金を与えたときの GDPW (5) の『QUM 推定値』の値"
      " ≒ (b) 推定値・立地固定 − (a) η=0・立地固定）")
