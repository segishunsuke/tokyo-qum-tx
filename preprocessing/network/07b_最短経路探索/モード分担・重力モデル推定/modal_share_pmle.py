# -*- coding: utf-8 -*-
"""
modal_share_pmle.py ― 交通モード分担モデル（自動車 vs 徒歩＋公共交通）の推定と合成一般化費用の算出

【モデル】
  V_car = γ1 Ĝ_car,   V_pt = γ0 + γ1 Ĝ_pt,   Ĝ = 往復の一般化費用（GC_ij + GC_ji）
  P_pt = 1 / (1 + exp(-(γ0 + γ1 (Ĝ_pt − Ĝ_car))))
  2018年東京都市圏パーソントリップ調査の計画基本ゾーン間 OD 交通量（自宅－勤務，自宅－私事）を用い，
  二項 GLM（集計ロジット，ポアソン疑似最尤法と同じく OD 交通量を度数とする疑似最尤推定）で推定する．

【初版からの変更点】
  1. 公共交通の交通量に「バス」を含める（鉄道＋バス＋徒歩）．ネットワークにバスを追加したため．
  2. 費用ファイルを search_passenger_separated.py の出力（od_min_cost_passenger_{pt|car}_{year}.csv，
     列 o_id, d_id, cost）から直接読む．o_id / d_id は centroids.gpkg の node_id（= kzone.csv の id）．
  3. 標準誤差として，OD ペアを単位としたロバスト標準誤差（サンドイッチ型）も出力する．
     拡大推計後のトリップ数を度数とする通常の標準誤差は，トリップを独立な観測とみなすため過小になる．
  4. 推定したパラメータで，各年次の合成一般化費用 τ_ij を算出して出力する（--years）．
  5. 反実仮想（TX なし）の τ も同じパラメータで算出する（--cf_label，--cf_year）．
     公共交通の費用は od_min_cost_passenger_{cf_label}_{cf_year}.csv，自動車の費用は現実と共通．
  6. --qum_dir を指定すると，QUM（estimation / simulation / simulation2）の入力ファイル名で書き出す：
       tau[0].csv = Real_tau[0].csv = CF_tau[0].csv = τ_1995（1995年は TX 開業前なので CF も同じ）
       tau[1].csv = Real_tau[1].csv = τ_2015，  CF_tau[1].csv = τ_2015（TX なし）

使い方：
  python modal_share_pmle.py --cost_dir ../output --years 1995 2015 2018
  python modal_share_pmle.py --cost_dir ../output --years 1995 2015 --cf_label pt_noTX --cf_year 2015 --qum_dir ../qum_input
出力（./result/）：
  modal_share_estimates.csv  … 推定結果
  modal_share_summary.txt    … GLM のサマリー
  tau_{year}.csv             … 往復の合成一般化費用 τ_ij（列 o_id, d_id, tau．対角要素は本文式(…)で別途設定）
"""
import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def load_cost(path, n):
    """od_min_cost の長形式を n×n の行列に変換する（到達不能は inf）"""
    c = pd.read_csv(path, encoding="utf-8-sig")
    c.columns = [x.strip() for x in c.columns]
    o = c.iloc[:, 0].astype(int).to_numpy(); d = c.iloc[:, 1].astype(int).to_numpy()
    v = pd.to_numeric(c.iloc[:, 2], errors="coerce").to_numpy()
    M = np.full((n, n), np.inf)
    M[o, d] = np.where(np.isfinite(v), v, np.inf)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="d-1.csv")
    ap.add_argument("--kzone", default="kzone.csv")
    ap.add_argument("--cost_dir", default="../output")
    ap.add_argument("--est_year", type=int, default=2018)
    ap.add_argument("--years", type=int, nargs="*", default=[1995, 2015, 2018])
    ap.add_argument("--out", default="./result")
    ap.add_argument("--cf_label", default="", help="反実仮想の公共交通の費用ファイルのラベル（例：pt_noTX）")
    ap.add_argument("--cf_year", type=int, default=2015)
    ap.add_argument("--qum_dir", default="", help="QUM の入力ファイル名（tau[t].csv 等）でも書き出すフォルダ")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    kz = pd.read_csv(a.kzone)
    n = len(kz)
    kzone_to_id = dict(zip(kz["kzone"], kz["id"]))
    assert sorted(kz["id"]) == list(range(n)), "kzone.csv の id は 0〜n-1 の連番である必要があります。"

    # ---- トリップデータ ----
    d1 = pd.read_csv(a.trips, skiprows=3, dtype=str, encoding="cp932")
    t = d1[["発地", "着地", "目的種類", "鉄道", "バス", "自動車", "徒歩"]].copy()
    purpose = t["目的種類"].astype(str)
    t = t[purpose.str.contains("自宅") & (purpose.str.contains("勤務") | purpose.str.contains("私事"))]
    t["o"] = t["発地"].apply(clean_kzone).map(kzone_to_id)
    t["d"] = t["着地"].apply(clean_kzone).map(kzone_to_id)
    t = t.dropna(subset=["o", "d"])
    t = t[t["o"] != t["d"]]
    t["pt_trips"] = t["鉄道"].apply(clean_trip) + t["バス"].apply(clean_trip) + t["徒歩"].apply(clean_trip)
    t["car_trips"] = t["自動車"].apply(clean_trip)
    # 自宅－勤務と自宅－私事を OD ペアごとに合算
    t = t.groupby(["o", "d"], as_index=False)[["pt_trips", "car_trips"]].sum()
    t = t[(t["pt_trips"] > 0) | (t["car_trips"] > 0)]
    o = t["o"].astype(int).to_numpy(); d = t["d"].astype(int).to_numpy()

    # ---- 費用（推定年次）----
    gc_car = load_cost(os.path.join(a.cost_dir, f"od_min_cost_passenger_car_{a.est_year}.csv"), n)
    gc_pt = load_cost(os.path.join(a.cost_dir, f"od_min_cost_passenger_pt_{a.est_year}.csv"), n)
    rt_car = gc_car + gc_car.T
    rt_pt = gc_pt + gc_pt.T
    t["car_cost"] = rt_car[o, d]
    t["pt_cost"] = rt_pt[o, d]
    ok = np.isfinite(t["car_cost"]) & np.isfinite(t["pt_cost"])
    print(f"費用が得られない OD ペアを除外：{(~ok).sum():,} 件")
    t = t[ok]

    Y = t[["pt_trips", "car_trips"]].to_numpy()
    X = sm.add_constant((t["pt_cost"] - t["car_cost"]).astype(float).to_numpy())
    model = sm.GLM(Y, X, family=sm.families.Binomial())
    res = model.fit()                        # 通常の標準誤差（トリップを独立とみなす）
    res_r = model.fit(cov_type="HC0")        # OD ペア単位のロバスト標準誤差

    est = pd.DataFrame({
        "param": ["gamma0", "gamma1"],
        "estimate": res.params,
        "se_model": res.bse,
        "se_robust_OD": res_r.bse,
        "z_robust": res_r.tvalues,
    })
    est.to_csv(os.path.join(a.out, "modal_share_estimates.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(a.out, "modal_share_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"OD ペア数: {len(t):,}\n公共交通トリップ: {t['pt_trips'].sum():,.0f}，自動車トリップ: {t['car_trips'].sum():,.0f}\n\n")
        f.write(str(res.summary()) + "\n\n【ロバスト標準誤差（OD ペア単位）】\n" + str(res_r.summary()))
    pd.options.display.float_format = "{:.6e}".format
    print(f"分析対象 OD ペア数: {len(t):,}")
    print(est.to_string(index=False))
    g0, g1 = res.params
    if g1 >= 0:
        raise ValueError("γ1 が負になっていません。費用データを確認してください。")

    # ---- 合成一般化費用 τ_ij（各年次）----
    oo, dd = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    off = ~np.eye(n, dtype=bool)

    def compose(pt_label, y, name):
        car = load_cost(os.path.join(a.cost_dir, f"od_min_cost_passenger_car_{y}.csv"), n)
        pt = load_cost(os.path.join(a.cost_dir, f"od_min_cost_passenger_{pt_label}_{y}.csv"), n)
        rc, rp = car + car.T, pt + pt.T
        # τ = (1/γ1) ln[exp(γ1 Ĝcar) + exp(γ0 + γ1 Ĝpt)] − (1/γ1) ln[1 + exp(γ0)]
        tau = (np.logaddexp(g1 * rc, g0 + g1 * rp) - np.logaddexp(0.0, g0)) / g1
        out = pd.DataFrame({"o_id": oo.ravel(), "d_id": dd.ravel(), "tau": tau.ravel()})
        out.to_csv(os.path.join(a.out, name), index=False, encoding="utf-8")  # BOM なし（Cython 側の読み込みに合わせる）
        print(f"  {name}：非対角要素の平均 {np.mean(tau[off]):,.1f} 円，到達不能 {np.isinf(tau[off]).sum():,} 件")
        return tau, rp

    taus = {}
    for y in a.years:
        taus[y] = compose("pt", y, f"tau_{y}.csv")
    if a.cf_label:
        tau_cf, rp_cf = compose(a.cf_label, a.cf_year, f"tau_{a.cf_year}_{a.cf_label}.csv")
        if a.cf_year in taus:
            tau_r, rp_r = taus[a.cf_year]
            d = tau_cf - tau_r
            dp = rp_cf - rp_r
            print(f"  反実仮想と現実の差（{a.cf_year}年）：τ が増えた OD {np.sum(d[off] > 1e-6):,} 件，"
                  f"減った OD {np.sum(d[off] < -1e-6):,} 件（減るのは誤り），最大増加 {np.nanmax(d[off]):,.1f} 円")
            print(f"    公共交通の往復費用が減った OD：{np.sum(dp[off] < -1e-6):,} 件（0 であるべき）")
    if a.qum_dir:
        if not (len(a.years) >= 2 and a.cf_label):
            raise SystemExit("--qum_dir には --years に 2 時点（例：1995 2015）と --cf_label が必要です。")
        y0, y1 = a.years[0], a.years[1]
        if a.cf_year != y1:
            raise SystemExit(f"--cf_year（{a.cf_year}）は --years の2番目（{y1}）と同じにしてください。")
        os.makedirs(a.qum_dir, exist_ok=True)
        files = {f"tau[0].csv": f"tau_{y0}.csv", f"tau[1].csv": f"tau_{y1}.csv",
                 f"Real_tau[0].csv": f"tau_{y0}.csv", f"Real_tau[1].csv": f"tau_{y1}.csv",
                 f"CF_tau[0].csv": f"tau_{y0}.csv", f"CF_tau[1].csv": f"tau_{y1}_{a.cf_label}.csv"}
        import shutil
        for dst, src in files.items():
            shutil.copy2(os.path.join(a.out, src), os.path.join(a.qum_dir, dst))
            print(f"  {a.qum_dir}/{dst} ← {src}")

if __name__ == "__main__":
    main()
