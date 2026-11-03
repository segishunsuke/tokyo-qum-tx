# -*- coding: utf-8 -*-
"""
kappa_power.py ― ゾーン内の費用 τ_ii と，通勤不効用のパラメータ（κ, θ．exp 型・べき乗型），べき乗型の τ0 の設定

【ゾーン内の費用】
  τ_ii = （隣接するゾーンへの往復の一般化費用 τ_ij の平均）× 1/2（--factor）
  （四段階推定法の実務で一般的な設定）
  隣接関係はゾーンのポリゴン（計画基本ゾーンの shapefile）で境界を共有するもの．
  隣接ゾーンがない場合（島など）は，セントロイドが最も近いゾーンを隣接ゾーンとみなす．
  τ_ij には2015年の TX ありの値（--tau で指定）を使い，全時点・全シナリオで共通とする．
  → tau_ii.csv（列 id, kzone, tau_ii）に出力．QUM では cases.csv の tau_ii 列にこのファイルを指定する．

【κ の設定：限界的な金銭価値を実勢に合わせる】
  一般化費用 τ を 1 円/日 下げたときの等価変分（年額）が，実勢の d 円（年間出勤日数）に一致するように κ を決める．
  τ̃ はゾーン間の通勤トリップ（PT調査「自宅－勤務」，トリップ数で重み付け）の τ の中央値．
    exp 型   ： 効用 ln w − κτ   → 等価変分 ≈ w̄ κ Δτ        → κ = d / (w̄ 10^6)
    べき乗型 ： 効用 ln w − κ ln τ → 等価変分 ≈ w̄ κ Δτ / τ   → κ = τ̃ d / (w̄ 10^6)
  （exp 型の限界価値は τ によらず一定なので τ̃ は使わない．べき乗型の κ は τ0 に依存しない）
  θ = θκ / κ（θκ は gravity_ppml.py の「自宅－勤務」の推定値．exp 型は gravity_estimates.csv，
  べき乗型は gravity_estimates_power_tau.csv）

【べき乗型の τ0】通勤不効用 min(1, (τ/τ0)^(−κ))，アクセシビリティの減衰 min(1, (τ/τ0)^(−δ)) の切り捨ての境目．
  PT調査の「自宅－勤務」トリップ（ゾーン内を含む．ゾーン内は上の τ_ii）の τ の分位点（トリップ数で重み付け）

使い方：
  python kappa_power.py --zones ../../H30_gis/H30_kzone.shp
  （--zone_field を指定するとその列の kzone で結合．指定しなければレコード順 = kzone.csv の id 順とみなす）
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
ap.add_argument("--zones", required=True, help="計画基本ゾーンのポリゴン（shapefile 等）")
ap.add_argument("--zone_field", help="ポリゴンの kzone の列名（省略時はレコード順）")
ap.add_argument("--gravity", default="./result/gravity_estimates_power_tau.csv")
ap.add_argument("--gravity_exp", default="./result/gravity_estimates.csv")
ap.add_argument("--quantiles", type=float, nargs="+", default=[0.005, 0.01, 0.05])
ap.add_argument("--wage", type=float, default=4.5141)
ap.add_argument("--days", type=float, default=224.4)
ap.add_argument("--out_tau_ii", default="./result/tau_ii.csv")
ap.add_argument("--factor", type=float, default=0.5, help="τ_ii = factor × 隣接ゾーンへの τ の平均")
a = ap.parse_args()

import geopandas as gpd

kz = pd.read_csv(a.kzone); n = len(kz)
k2i = dict(zip(kz["kzone"], kz["id"]))
td = pd.read_csv(a.tau, encoding="utf-8-sig")
if not {"o_id", "d_id", "tau"} <= set(td.columns):          # QUM の入力 tau[t].csv（列 i, j, cost）の形式も読めるように
    td = td.iloc[:, :3].copy(); td.columns = ["o_id", "d_id", "tau"]
tau = np.full((n, n), np.nan); tau[td["o_id"].astype(int), td["d_id"].astype(int)] = td["tau"].to_numpy(dtype=float)
print(f"一般化費用：{a.tau}")

# ---- 隣接関係
z = gpd.read_file(a.zones)
if a.zone_field:
    z["id"] = z[a.zone_field].astype(int).map(k2i)
    if z["id"].isna().any():
        raise SystemExit(f"kzone.csv にない kzone があります：{z.loc[z['id'].isna(), a.zone_field].tolist()[:10]}")
else:
    if len(z) != n:
        raise SystemExit(f"ゾーンのレコード数 {len(z)} が kzone.csv の {n} と一致しません（--zone_field を指定してください）。")
    z["id"] = np.arange(n)
z = z.to_crs(6677)
z["geometry"] = z.geometry.buffer(1.0)                 # 1 m の誤差を許して境界の共有を判定
pairs = gpd.sjoin(z[["id", "geometry"]], z[["id", "geometry"]], predicate="intersects")
pairs = pairs[pairs["id_left"] != pairs["id_right"]]
nb = {i: set() for i in range(n)}
for i, j in zip(pairs["id_left"].astype(int), pairs["id_right"].astype(int)):
    nb[i].add(j)
cent = np.c_[z.geometry.centroid.x, z.geometry.centroid.y][np.argsort(z["id"].to_numpy())]
isolated = [i for i in range(n) if not nb[i]]
for i in isolated:
    dd = np.hypot(*(cent - cent[i]).T); dd[i] = np.inf
    nb[i].add(int(np.argmin(dd)))

tii = np.array([a.factor * np.nanmean([tau[i, j] for j in nb[i]]) for i in range(n)])
if np.isnan(tii).any():
    raise SystemExit(f"隣接ゾーンへの τ が得られないゾーンがあります：{np.where(np.isnan(tii))[0].tolist()[:10]}")
pd.DataFrame({"id": np.arange(n), "kzone": kz.sort_values("id")["kzone"].to_numpy(), "tau_ii": tii}).to_csv(
    a.out_tau_ii, index=False, encoding="utf-8")
nn = np.array([len(nb[i]) for i in range(n)])
tmin = np.nanmin(np.where(np.eye(n, dtype=bool), np.nan, tau), axis=1)
print(f"【ゾーン内の費用】隣接ゾーン数：平均 {nn.mean():.2f}，最小 {nn.min()}，最大 {nn.max()}"
      f"（隣接ゾーンがなく最も近いゾーンで代用：{len(isolated)} ゾーン）")
print(f"  τ_ii：最小 {float(tii.min())!r}，中央値 {float(np.median(tii))!r}，最大 {float(tii.max())!r} 円（往復）")
print(f"  τ_ii が，そのゾーンから最も安いゾーン間の τ を上回るゾーン：{int(np.sum(tii > tmin))} / {n}")
print(f"  → {a.out_tau_ii}")
np.fill_diagonal(tau, tii)

# ---- 通勤トリップ
d1 = pd.read_csv(a.trips, skiprows=3, dtype=str, encoding="cp932")
t = d1[d1["目的種類"].astype(str).str.strip() == "自宅－勤務"]
o = t["発地"].apply(clean_kzone).map(k2i); d = t["着地"].apply(clean_kzone).map(k2i)
q = t["計"].apply(clean_trip)
ok = o.notna() & d.notna()
o = o[ok].astype(int).to_numpy(); d = d[ok].astype(int).to_numpy(); q = q[ok].to_numpy()
tij = tau[o, d]
valid = np.isfinite(tij) & (q > 0)
inter = valid & (o != d)
print(f"\nゾーン内の通勤トリップの割合：{q[valid & (o == d)].sum() / q[valid].sum():.1%}")

ta, qa = tij[valid], q[valid]                         # τ0 はゾーン内を含む全通勤トリップの分位点
order = np.argsort(ta); cw = np.cumsum(qa[order]) / qa.sum()
ti, qi = tij[inter], q[inter]                         # κ はゾーン間の通勤トリップの中央値で設定
oi = np.argsort(ti); cwi = np.cumsum(qi[oi]) / qi.sum()
tmed = float(ti[oi][np.searchsorted(cwi, 0.5)])
tbar = float(np.sum(qi * ti) / qi.sum())
c = a.days / (a.wage * 1e6)
tk_pw = float(pd.read_csv(a.gravity, encoding="utf-8-sig").set_index("model")["beta"]["commute"])
tk_ex = float(pd.read_csv(a.gravity_exp, encoding="utf-8-sig").set_index("model")["beta"]["commute"])
k_ex, k_pw = c, tmed * c
print(f"ゾーン間の通勤トリップの τ：中央値 τ̃ = {tmed!r} 円，平均 τ̄ = {tbar!r} 円（往復）")
print(f"\n[exp 型]   κ = d/(w̄ 10^6) = {k_ex!r}，θκ = {tk_ex!r}，θ = {tk_ex / k_ex!r}")
print(f"  set 列：theta={tk_ex / k_ex!r};kappa_theta={tk_ex!r}")
print(f"\n[べき乗型] κ = τ̃ d/(w̄ 10^6) = {k_pw!r}，θκ = {tk_pw!r}，θ = {tk_pw / k_pw!r}")
print(f"  set 列：theta={tk_pw / k_pw!r};kappa_theta={tk_pw!r}")
for qt in a.quantiles:
    tau0 = float(ta[order][np.searchsorted(cw, qt)])
    print(f"  τ0 = 通勤トリップの τ の {qt:.1%} 分位点 = {tau0!r} 円（往復），τ_ii < τ0 のゾーン {int(np.sum(tii < tau0))} / {n}"
          f" → tau_transform 列：log:{tau0!r}")
print(f"\n  tau_ii 列（共通）：input/tau_ii.csv（{a.out_tau_ii} を QUM の input フォルダにコピー）")
