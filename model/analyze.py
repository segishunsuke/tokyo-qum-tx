# -*- coding: utf-8 -*-
"""
analyze.py ― run_cases.py の結果の集計（旧 Analysis.xlsx の作業の自動化）

  python analyze.py
  python analyze.py --zones 計画基本ゾーン.shp --tx_stations N02-15_Station.shp   （地図も作る場合）
  python analyze.py --tx_zones tx_zones.csv     （TX沿線ゾーンを自分で指定する場合）

出力（analysis/）：
  summary.xlsx         全ての集計表（値のみ．Excel で確認・グラフ作成用）
  tables.tex           論文の表（そのまま貼り付けられる形）
  numbers.md           本文中の数値（通勤費用，便益の内訳など）と，結果の確認事項
  L_R_fitness-crop.pdf, L_M_fitness-crop.pdf, p_G_fitness-crop.pdf    4. (11) b) の散布図
  fitness_2x2-crop.pdf                                                4. (11) b) の散布図（3変数を 2×2 にまとめたもの）
  eps_1995_2015-crop.pdf                                              5. (2) の散布図（2×2）
  D_maps_2x2-crop.pdf                                                 5. (3) の地図（3変数を 2×2 にまとめたもの）
  tx_effect_zones.csv  TX の効果のゾーン別の値（GIS で地図を作る場合に結合して使う）
  err_map_*-crop.pdf, D_*-crop.pdf    （--zones を指定した場合）付録B と 5. (3) の地図

どのケースを何に使うかは，下の ROLE で指定する（cases.csv のケース名）．
集計の前に，比較するケースどうしの条件（構造残差の有無，η の値，親ケース）が想定通りかを確認し，
想定と異なる場合は numbers.md の冒頭と画面に警告を出す．
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(ROOT, "cases")

# ---------------------------------------------------------------- ケースの役割
ROLE = {
    "base": "base",                 # 基準ケース（構造残差あり，simulation2 まで）
    "zero": "zero_base",            # 構造残差ゼロ・集積あり
    "zero_noagg": "zero_noagg",     # 構造残差ゼロ・集積なし
    "dec_a": "dec_a",               # 便益の分解 (a)
    "dec_d": "dec_d",               # 便益の分解 (d)
    "webtag": "webtag",             # WebTAG の値（ηA = 0.043，δA = 1.655）
}
# 感度分析の表の列：(見出し, 推定ケース, 構造残差ゼロのケース)
# 自ゾーン密度型の最尤推定値は η = 0（境界）で，η = 0 のモデルは mass と同一なので dec_a・zero_noagg を使う
SENS = [("基準（減衰：べき乗型）", "base", "zero_base"),
        ("減衰：exp型", "exp_ml", "zero_exp_ml"),
        (r"自ゾーン密度（最尤，$\eta^A = 0$）", "dec_a", "zero_noagg"),
        (r"自ゾーン密度（$\eta^A = 0.05$）", "own_d05", "zero_own_d05")]
TAU0_SENS = []   # 切り捨てなしのべき乗型に変更したため τ0 の感度分析はなし   # べき乗型の τ0 の感度分析
YEARS = [1995, 2015]
WORK_DAYS = 224.4
VARS = [("L_R", "居住者数密度"), ("L_M", "従業者数密度"), ("p_G", "相対土地賃料")]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "analysis"))
ap.add_argument("--zones", help="ゾーンのポリゴン（shapefile 等）．レコード順が input_data.xlsx のゾーン順と同じこと")
ap.add_argument("--zone_field", help="ゾーンの shapefile の kzone の列名（指定すればレコード順ではなくこの列で結合）")
ap.add_argument("--tx_zones", help="TX沿線ゾーンの一覧（kzone 列を持つ CSV）")
ap.add_argument("--tx_stations", help="国土数値情報 N02 の駅データ（TX沿線ゾーンを自動で決める場合）")
ap.add_argument("--tx_radius_km", type=float, default=1.0, help="TX駅からこの距離以内のゾーンを沿線とする")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

WARN = []


def warn(msg):
    WARN.append(msg)
    print("※ " + msg)


# ---------------------------------------------------------------- 読み込み
def info_of(name):
    p = os.path.join(CASES_DIR, name, "case_info.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        inf = json.load(f)
    if inf.get("status") != "done":
        warn(f"ケース {name} の状態が {inf.get('status')} です（完了していないか，親ケースの再実行後に未更新）。集計から除きます。")
        return None
    return inf


def out_file(name, f):
    return os.path.join(CASES_DIR, name, "output", f)


def read_tbl(path):
    d = pd.read_csv(path)
    return d.loc[:, [c for c in d.columns if str(c).startswith("Period")]].to_numpy(dtype=float)  # S × T


def read_eps(name):
    return {k: read_tbl(os.path.join(CASES_DIR, name, "input", f"epsilon_{k}.csv")) for k in "ABG"}


def read_welfare(path):
    return pd.read_csv(path).set_index("Period")


all_cases = sorted(d for d in os.listdir(CASES_DIR) if os.path.isdir(os.path.join(CASES_DIR, d))) \
    if os.path.isdir(CASES_DIR) else []
INFO = {k: info_of(k) for k in all_cases}
INFO = {k: v for k, v in INFO.items() if v}
if not INFO:
    raise SystemExit("cases/ に完了したケースがありません。先に run_cases.py を実行してください。")

# 観測値（input/input_data.xlsx）
X = os.path.join(ROOT, "input", "input_data.xlsx")
area = pd.read_excel(X, sheet_name="area", header=None)
kzone = area.iloc[1:, 0].astype(int).to_numpy()
G = area.iloc[1:, 1].astype(float).to_numpy()
S = len(G)
T = len(YEARS)
obs_raw = {sh: pd.read_excel(X, sheet_name=sh, header=None).iloc[1:, 1:1 + T].to_numpy(dtype=float)
           for sh in ["lam_R", "lam_M", "p_G"]}
pg_sheet = pd.read_excel(X, sheet_name="p_G", header=None)
try:
    rate = float(pg_sheet.iloc[1, list(pg_sheet.iloc[0]).index("金利")])
except Exception:
    rate = 0.02
    warn("p_G シートに「金利」の列が見つからないため，地価への換算に金利 2% を用いました。")


def to_vars(lamR, lamM, pG, Lbar):
    """シェアと土地賃料から比較用の変数（密度・相対土地賃料）を作る．各 S × T"""
    lamR = lamR / lamR.sum(axis=0, keepdims=True)
    lamM = lamM / lamM.sum(axis=0, keepdims=True)
    return {"L_R": lamR * Lbar[None, :] / G[:, None],
            "L_M": lamM * Lbar[None, :] / G[:, None],
            "p_G": pG / pG.mean(axis=0, keepdims=True)}


def lbar_of(inf):
    return np.array([inf["parameters"][f"Lbar[{t}]"] for t in range(T)])


def pred_vars(name, suffix="_predict"):
    inf = INFO[name]
    return to_vars(read_tbl(out_file(name, f"lam_R{suffix}.csv")), read_tbl(out_file(name, f"lam_M{suffix}.csv")),
                   read_tbl(out_file(name, f"p_G{suffix}.csv")), lbar_of(inf))


def obs_vars(inf):
    return to_vars(obs_raw["lam_R"], obs_raw["lam_M"], obs_raw["p_G"], lbar_of(inf))


def has(name, step=None):
    return name in INFO and (step is None or step in INFO[name]["steps"])


# ---------------------------------------------------------------- 条件の確認
def same_except(n1, n2, keys_allowed):
    p1, p2 = INFO[n1]["parameters"], INFO[n2]["parameters"]
    diff = [k for k in p1 if k in p2 and k not in keys_allowed and not k.endswith("_fixed")
            and not np.isclose(p1[k], p2[k], rtol=1e-10, atol=0)]
    if INFO[n1]["accessibility"] != INFO[n2]["accessibility"]:
        diff.append("accessibility")
    return diff


def expect(name, eps=None, eta_A=None, eta_B=None, parent=None, step=None):
    if not has(name):
        return False
    inf, p = INFO[name], INFO[name]["parameters"]
    ok = True
    if eps is not None and (inf["epsilon"] == "zero") != (eps == "zero"):
        warn(f"{name}：構造残差の扱いが {inf['epsilon']} です（想定：{eps}）。"); ok = False
    if eta_A is not None and not np.isclose(p["eta_A"], eta_A):
        warn(f"{name}：ηA = {p['eta_A']}（想定：{eta_A}）。"); ok = False
    if eta_B is not None and not np.isclose(p["eta_B"], eta_B):
        warn(f"{name}：ηB = {p['eta_B']}（想定：{eta_B}）。"); ok = False
    if parent is not None and inf["parent"] != parent:
        warn(f"{name}：親ケースが {inf['parent'] or '(なし)'} です（想定：{parent}）。"); ok = False
    if step is not None and step not in inf["steps"]:
        warn(f"{name}：{step} を実行していません。"); ok = False
    return ok


B = ROLE["base"]
if has(B):
    expect(B, eps="estimate", step="simulation2")
    pb = INFO[B]["parameters"]
    if has(ROLE["zero"]):
        expect(ROLE["zero"], eps="zero", eta_A=pb["eta_A"], eta_B=pb["eta_B"], parent=B)
        d = same_except(ROLE["zero"], B, [])
        if d:
            warn(f"{ROLE['zero']} と {B} でパラメータが異なります：{d}")
    if has(ROLE["zero_noagg"]):
        expect(ROLE["zero_noagg"], eps="zero", eta_A=0, eta_B=0, parent=B)
        if has(ROLE["zero"]):
            d = same_except(ROLE["zero_noagg"], ROLE["zero"], ["eta_A", "eta_B"])
            if d:
                warn(f"{ROLE['zero_noagg']} と {ROLE['zero']} で η 以外のパラメータが異なります：{d}")
    if has(ROLE["dec_a"]):
        expect(ROLE["dec_a"], eps="estimate", eta_A=0, eta_B=0, step="simulation2")
    if has(ROLE["dec_d"]):
        expect(ROLE["dec_d"], eps="estimate", eta_A=0.05, eta_B=pb["eta_B"], step="simulation2")
for _, est, zero in SENS:
    if has(est) and has(zero):
        expect(zero, eps="zero", eta_A=INFO[est]["parameters"]["eta_A"], eta_B=INFO[est]["parameters"]["eta_B"])
        if INFO[est]["accessibility"] != INFO[zero]["accessibility"] and INFO[est]["parameters"]["eta_A"] > 0:
            warn(f"{est} と {zero} で集積の外部性の定式化が異なります。")

sheets, tex, md = {}, [], []

# ---------------------------------------------------------------- 1. 推定値
rows = []
for k, inf in INFO.items():
    p = inf["parameters"]
    r = {"case": k, "note": inf.get("note", ""), "accessibility": inf["accessibility"],
         "delta_A": p["delta_A"], "delta_B": p["delta_B"], "theta": p["theta"], "eta_A": p["eta_A"],
         "eta_B": p["eta_B"], "Abar": p["Abar"], "epsilon": inf["epsilon"], "parent": inf["parent"]}
    for kk, v in (inf.get("estimation") or {}).items():
        r["est_" + kk] = v
    rows.append(r)
sheets["推定値・条件"] = pd.DataFrame(rows)

# ---------------------------------------------------------------- 2. 構造残差ゼロの適合度
def fitness(name):
    pv, ov = pred_vars(name), obs_vars(INFO[name])
    out = []
    for v, lab in VARS:
        x, xh = ov[v], pv[v]
        lr = np.log(xh / x)
        row = {"case": name, "var": v, "label": lab}
        for t, y in enumerate(YEARS):
            row[f"MALE_{y}"] = np.mean(np.abs(lr[:, t]))
            row[f"corr_log_{y}"] = np.corrcoef(np.log(x[:, t]), np.log(xh[:, t]))[0, 1]
        dx, dxh = np.log(x[:, 1] / x[:, 0]), np.log(xh[:, 1] / xh[:, 0])
        r = np.corrcoef(dx, dxh)[0, 1]
        row["corr_change"] = r
        row["t_change"] = r * np.sqrt((S - 2) / (1 - r ** 2))
        row["sd_change_obs"], row["sd_change_pred"] = dx.std(), dxh.std()
        out.append(row)
    return out, pv, ov


fit_rows, zone_tables = [], {}
for k, inf in INFO.items():
    if inf["epsilon"] == "zero" and "simulation" in inf["steps"]:
        r, pv, ov = fitness(k)
        fit_rows += r
        zone_tables[k] = (pv, ov)
if fit_rows:
    sheets["適合度（構造残差ゼロ）"] = pd.DataFrame(fit_rows)
    fdf = pd.DataFrame(fit_rows).set_index(["case", "var"])

Z, ZN = ROLE["zero"], ROLE["zero_noagg"]
if Z in zone_tables:
    pv, ov = zone_tables[Z]
    zt = pd.DataFrame({"kzone": kzone, "area_km2": G})
    for v, _ in VARS:
        for t, y in enumerate(YEARS):
            zt[f"{v}_obs_{y}"] = ov[v][:, t]
            zt[f"{v}_pred_{y}"] = pv[v][:, t]
            zt[f"{v}_logratio_{y}"] = np.log(pv[v][:, t] / ov[v][:, t])
    if ZN in zone_tables:
        for v, _ in VARS:
            for t, y in enumerate(YEARS):
                zt[f"{v}_pred_noagg_{y}"] = zone_tables[ZN][0][v][:, t]
    sheets["ゾーン別（構造残差ゼロ）"] = zt
    zt.to_csv(os.path.join(a.out, "fitness_zones.csv"), index=False, encoding="utf-8-sig")

    # 散布図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # (観測値の記号, 予測値の記号, 単位)．ハットは上付き添字ではなく本体の文字に付ける
    sym = {"L_R": (r"L^R_{t,i}/\bar{G}_i", r"\hat{L}^R_{t,i}/\bar{G}_i", "[persons/km$^2$]"),
           "L_M": (r"L^M_{t,i}/\bar{G}_i", r"\hat{L}^M_{t,i}/\bar{G}_i", "[persons/km$^2$]"),
           "p_G": (r"p^G_{t,i}/\bar{p}^G_t", r"\hat{p}^G_{t,i}/\bar{p}^G_t", "")}
    COLS_Y = ["#1f5a91", "#ef7d31"]

    def fit_panel(ax, v, compact=False, legend=True):
        for t, (y, col) in enumerate(zip(YEARS, COLS_Y)):
            ax.scatter(ov[v][:, t], pv[v][:, t], s=4 if compact else 6, alpha=0.8, color=col, lw=0, label=str(y))
        lo = min(ov[v].min(), pv[v].min()) * 0.7
        hi = max(ov[v].max(), pv[v].max()) * 1.4
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        sy, sp, un = sym[v]
        if compact:                                    # 2×2 の図：記号をパネルの題に，軸は「観測値／予測値」のみ
            ax.set_xlabel("Observed", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
            ax.set_title(f"{PANEL_TITLE[v]}\n${sy}$ {un}".strip(), fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.set_xlabel(f"Observed ${sy}$ {un}".strip(), fontsize=9)
            ax.set_ylabel(f"Predicted ${sp}$ {un}".strip(), fontsize=9)
            ax.tick_params(labelsize=8)
        if legend:
            ax.legend(frameon=False, loc="upper left", fontsize=8, markerscale=2)

    PANEL_TITLE = {"L_R": "(a) Residential density", "L_M": "(b) Employment density", "p_G": "(c) Relative land rent"}
    for v, lab in VARS:                                # 個別の図（従来通り）
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        fit_panel(ax, v)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{v}_fitness-crop.pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    # 2×2 にまとめた図（1段の幅に入れる想定．4枠目は凡例）
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 4.9))
    for ax, (v, lab) in zip(axs.flat, VARS):
        fit_panel(ax, v, compact=True, legend=False)
    ax4 = axs.flat[3]; ax4.set_axis_off()
    h = [plt.Line2D([], [], ls="", marker="o", ms=5, color=c, label=str(y)) for y, c in zip(YEARS, COLS_Y)]
    h.append(plt.Line2D([], [], color="k", ls="--", lw=0.8, label="Predicted = Observed"))
    ax4.legend(handles=h, loc="center", frameon=False, fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(os.path.join(a.out, "fitness_2x2-crop.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # 集積の有無の比較の表
    if ZN in zone_tables:
        for t, y in enumerate(YEARS):
            tex += [f"% ---- 表：集積力の有無による平均相対誤差の変化（{y}年）  [{Z} vs {ZN}]",
                    r"\begin{tabular}{lccc}", r"\hline", r"比較項目 & 集積力有 & 集積力無   \\ \hline"]
            for v, lab in VARS:
                tex.append(f"{lab} & {fdf.loc[(Z, v), f'MALE_{y}']:.4f} & {fdf.loc[(ZN, v), f'MALE_{y}']:.4f}   \\\\")
            tex += [r"\hline", r"\end{tabular}", ""]
        md.append("## 4. (11) 構造残差ゼロの適合度\n")
        md.append(f"ケース：集積あり = {Z}，集積なし = {ZN}\n")
        md.append("| 変数 | 年 | 平均相対誤差（有） | （無） | 改善率 | 対数の相関（有） |\n|---|---|---|---|---|---|")
        for v, lab in VARS:
            for y in YEARS:
                m1, m0 = fdf.loc[(Z, v), f"MALE_{y}"], fdf.loc[(ZN, v), f"MALE_{y}"]
                md.append(f"| {lab} | {y} | {m1:.4f} | {m0:.4f} | {1 - m1 / m0:+.1%} | {fdf.loc[(Z, v), f'corr_log_{y}']:.3f} |")
        md.append("\n変化率（1995→2015 の対数差）の観測値と予測値の相関（集積あり）：")
        for v, lab in VARS:
            r = fdf.loc[(Z, v)]
            md.append(f"- {lab}：r = {r['corr_change']:.3f}（t = {r['t_change']:.2f}，N = {S}）．"
                      f"変化率の標準偏差：観測 {r['sd_change_obs']:.4f}，予測 {r['sd_change_pred']:.4f}")
        md.append("\n過小予測が大きいゾーン（2015年，対数比の小さい順に5つ）：")
        for v, lab in VARS:
            w = zt.nsmallest(5, f"{v}_logratio_2015")
            md.append(f"- {lab}：" + "，".join(f"{int(z)}（{r:+.2f}）" for z, r in zip(w.kzone, w[f'{v}_logratio_2015'])))
        md.append("\n過大予測が大きいゾーン（2015年，大きい順に5つ）：")
        for v, lab in VARS:
            w = zt.nlargest(5, f"{v}_logratio_2015")
            md.append(f"- {lab}：" + "，".join(f"{int(z)}（{r:+.2f}）" for z, r in zip(w.kzone, w[f'{v}_logratio_2015'])))
        md.append("")

# ---------------------------------------------------------------- 3. 通勤費用（基準ケース）
if has(B):
    wf = read_welfare(out_file(B, "welfare_indicators.csv"))
    md.append("## 4. (11) a) 通勤費用（基準ケース，構造残差あり）\n")
    for t, y in enumerate(YEARS):
        c = wf.loc[t, "Average_Commuting_Cost"]
        md.append(f"- {y}年：年間 {c * 100:.2f} 万円，一日あたり {c * 1e6 / WORK_DAYS:,.2f} 円（{WORK_DAYS}日で換算）")
    md.append(f"- 平均賃金：" + "，".join(f"{y}年 {wf.loc[t, 'Average_Wage']:.6f}" for t, y in enumerate(YEARS)) + " 百万円\n")

# ---------------------------------------------------------------- 4. 便益
# 家計の便益の定義（表に使うもの）
#   gross_real：現実の均衡（With TX）の総所得 W* を基準（初稿の定義）  −W*_R [exp(V_CF − V_R) − 1]
#   gross_cf  ：TX なしの均衡の総所得を基準                             W_CF [exp(V_R − V_CF) − 1]
#   net_cf    ：TX なしの均衡の純所得（総所得 − 通勤費用）を基準         W^net_CF [exp(V_R − V_CF) − 1]
#     （参考．η = 0・立地固定ではモデル上の通勤費用 w(1 − exp(−κτ)) の削減額に一致する．
#       ただし本モデルの通勤費用は効用の費用で予算制約には入らない（住宅需要も w に比例）ため，
#       一括の所得移転は総所得に加わる．等価変分としては gross_cf が整合的）
BASIS = "gross_cf"
BASIS_LABEL = {"gross_real": "総所得・TXあり基準", "gross_cf": "総所得・TXなし基準", "net_cf": "純所得・TXなし基準"}


def benefits(name, t=1):
    cf = read_welfare(out_file(name, "welfare_indicators_CF.csv")).loc[t]
    re_ = read_welfare(out_file(name, "welfare_indicators_Real.csv")).loc[t]
    fx = pd.read_csv(out_file(name, "EV_and_Landlord_FixedPopulation.csv")).set_index("Period").loc[t]
    Lb = lbar_of(INFO[name])[t]
    dV = re_["Logsum_V"] - cf["Logsum_V"]
    W_cf = cf["Total_Household_Income"]
    Wn_cf = W_cf - Lb * cf["Average_Commuting_Cost"]
    hh = {"gross_real": -re_["Total_Household_Income"] * (np.exp(-dV) - 1),
          "gross_cf": W_cf * (np.exp(dV) - 1),
          "net_cf": Wn_cf * (np.exp(dV) - 1)}
    fx_hh = {"gross_real": fx["EV_Total"], "gross_cf": fx["EV_Total"],
             "net_cf": fx.get("EV_Net_Total", np.nan)}
    ll = re_["Landlord_Income"] - cf["Landlord_Income"]
    fl = fx["Landlord_Income_Change(Fixed - CF)"]
    out = {"case": name, "landlord_CF": cf["Landlord_Income"], "landlord_Real": re_["Landlord_Income"],
           "W_star": re_["Total_Household_Income"], "W_CF": W_cf, "W_net_CF": Wn_cf,
           "expVdiff_minus1": np.exp(-dV) - 1, "exp_dV_minus1": np.exp(dV) - 1,
           "reloc_landlord": ll, "fixed_landlord": fl,
           "cost_saving_model": fx.get("Commuting_Cost_Saving_Model", np.nan)}
    for k in hh:
        out[f"reloc_household_{k}"] = hh[k]
        out[f"reloc_total_{k}"] = hh[k] + ll
        out[f"fixed_household_{k}"] = fx_hh[k]
        out[f"fixed_total_{k}"] = fx_hh[k] + fl
    for kind in ("reloc", "fixed"):                    # 表に使う定義
        out[f"{kind}_household"] = out[f"{kind}_household_{BASIS}"]
        out[f"{kind}_total"] = out[f"{kind}_total_{BASIS}"]
    return out


def check_real(name):
    """現実の均衡（With TX）が観測値を再現しているか"""
    inf = INFO[name]
    if inf["epsilon"] == "zero":
        return
    pv, ov = pred_vars(name, "_Real"), obs_vars(inf)
    e = max(np.max(np.abs(pv[v][:, 1] / ov[v][:, 1] - 1)) for v, _ in VARS)
    if e > 1e-4:
        warn(f"{name}：simulation2 の現実の均衡（With TX，2015年）が観測値を再現していません（最大相対誤差 {e:.1e}）。"
             "Real_tau[1].csv が tau[1].csv と同じか確認してください。")


ben = {}
for k, inf in INFO.items():
    if "simulation2" in inf["steps"]:
        if not os.path.exists(out_file(k, "lam_R_Real.csv")):
            warn(f"{k}：lam_R_Real.csv がありません（simulation2 を新しい版でビルドし直してください）。")
        else:
            check_real(k)
        ben[k] = benefits(k)
if ben:
    bdf = pd.DataFrame(ben.values())
    sheets["便益（百万円・年）"] = bdf
    oku = lambda x: x / 100.0
    if B in ben:
        b = ben[B]
        md.append("## 5. (5) 便益（基準ケース，2015年，百万円/年）\n")
        md.append(f"- 総土地賃料：Without TX {b['landlord_CF']:,.1f}，With TX {b['landlord_Real']:,.1f}，差 {b['reloc_landlord']:,.1f}")
        md.append(f"- W*（TXあり・総所得）= {b['W_star']:,.1f}，W_CF（TXなし・総所得）= {b['W_CF']:,.1f}，"
                  f"W^net_CF（TXなし・純所得）= {b['W_net_CF']:,.1f}")
        md.append(f"- exp(V_CF − V_R) − 1 = {b['expVdiff_minus1']:.9f}，exp(V_R − V_CF) − 1 = {b['exp_dV_minus1']:.9f}")
        for k, lab in BASIS_LABEL.items():
            md.append(f"- 家計の便益（{lab}）{b['reloc_household_' + k]:,.1f}，合計 {b['reloc_total_' + k]:,.1f}"
                      f"（家計の割合 {b['reloc_household_' + k] / b['reloc_total_' + k]:.1%}）" + ("  ← 表に使用" if k == BASIS else ""))
        md.append("")
    md.append("### 便益の分解（億円/年，定義別）\n")
    md.append("| ケース | " + " | ".join(BASIS_LABEL.values()) + " | モデル上の通勤費用の削減額 |\n|---|---|---|---|---|")
    for lab, k, kind in [("(a) η=0・立地固定", ROLE["dec_a"], "fixed"), ("(b) 推定値・立地固定", B, "fixed"),
                         ("(c) 推定値・立地変化あり", B, "reloc"),
                         ("(d) WebTAG値・立地固定", ROLE["webtag"], "fixed"), ("(e) WebTAG値・立地変化あり", ROLE["webtag"], "reloc")]:
        if k in ben:
            b = ben[k]
            md.append(f"| {lab} | " + " | ".join(f"{b[f'{kind}_total_{q}'] / 100:,.1f}" for q in BASIS_LABEL)
                      + (f" | {b['cost_saving_model'] / 100:,.1f} |" if kind == "fixed" else " | — |"))
    md.append("\n（合計＝家計＋地主．モデル上の通勤費用の削減額は Σ L λ_CF w_CF [exp(−κτ_R)/exp(−κτ_CF) − 1]（削減率 × 賃金）．"
              "(a) の総所得・TXなし基準の値に一致するはず．純所得基準はべき乗型では費用の水準の基準化に依存するので参考）\n")
    dec = [("(a) $\\eta = 0$・立地固定", ROLE["dec_a"], "fixed"), ("(b) 推定値・立地固定", B, "fixed"),
           ("(c) 推定値・立地変化あり", B, "reloc"),
           ("(d) WebTAG値・立地固定", ROLE["webtag"], "fixed"), ("(e) WebTAG値・立地変化あり", ROLE["webtag"], "reloc")]
    tex += ["% ---- 表：TX整備による年間便益の分解（億円/年）", r"\begin{tabular}{lccc}", r"\hline",
            r"ケース & 家計 & 地主 & 合計 \\ \hline"]
    for lab, k, kind in dec:
        if k in ben:
            b = ben[k]
            tex.append(f"{lab} & {oku(b[kind + '_household']):,.1f} & {oku(b[kind + '_landlord']):,.1f} & "
                       f"{oku(b[kind + '_total']):,.1f} \\\\   % {k}")
        else:
            tex.append(f"{lab} & -- & -- & -- \\\\   % {k} 未実行")
    tex += [r"\hline", r"\multicolumn{4}{r}{（単位：億円/年）}", r"\end{tabular}", ""]
    # 従来の交通費用削減便益（OD 交通量 × 一般化費用の削減額）：(a) との比較用
    #   OD 交通量は現実の均衡（With TX）の通勤 OD（simulation の出力，片道・就業者ベース），
    #   一般化費用は往復の合成一般化費用 τ（円/往復）．年間 224.4 日で換算．ゾーン内（対角）は TX で変わらないので除く．
    def read_tau(path):
        d = pd.read_csv(path)
        M = np.zeros((S, S))
        M[d.iloc[:, 0].astype(int).to_numpy(), d.iloc[:, 1].astype(int).to_numpy()] = d.iloc[:, 2].astype(float).to_numpy()
        return M
    def read_od(path):
        od = pd.read_csv(path)
        Q = np.zeros((S, S))
        Q[od["i"].to_numpy(), od["j"].to_numpy()] = od["demand"].to_numpy()
        return Q
    for k in [ROLE["dec_a"], B]:
        cfp = os.path.join(CASES_DIR, k, "input", "CF_tau[1].csv")
        rlp = os.path.join(CASES_DIR, k, "input", "Real_tau[1].csv")
        odr = out_file(k, "OD_demand[1]_predict.csv")     # 現実の均衡（With TX）の OD
        odc = out_file(k, "OD_demand[1]_CF.csv")          # 反実仮想（Without TX）の OD（新しい simulation2）
        if k not in ben or not all(os.path.exists(p) for p in (cfp, rlp, odr)):
            continue
        dtau = read_tau(cfp) - read_tau(rlp)
        np.fill_diagonal(dtau, 0.0)
        conv = {"With TX の OD": float((read_od(odr) * dtau).sum() * WORK_DAYS / 1e6)}
        if os.path.exists(odc):
            conv["Without TX の OD"] = float((read_od(odc) * dtau).sum() * WORK_DAYS / 1e6)
            conv["台形公式（両者の平均）"] = 0.5 * (conv["With TX の OD"] + conv["Without TX の OD"])
        else:
            warn(f"{k}：OD_demand[1]_CF.csv がありません（新しい simulation2 でケース {k} を再実行すると，"
                 "(a) と直接比較できる Without TX の OD による値も計算します）。")
        md.append(f"- 従来の交通費用削減便益（{k}，通勤 OD × 往復の τ の削減額 × {WORK_DAYS} 日，億円/年）："
                  + "，".join(f"{n} {v / 100:,.1f}" for n, v in conv.items()))
        md.append("  （(a) の立地固定の便益は Without TX の OD に対応する．With TX の OD による値は TX に転換した通勤者にも"
                  "削減額の全額を計上するため上限，台形公式が通常の費用便益分析の値）")
        ben[k].update({"conventional_" + n: v for n, v in conv.items()})
        break
    md.append("")

# ---------------------------------------------------------------- 5. 感度分析
cols = [(h, e, z) for h, e, z in SENS if has(e)]
if cols:
    def cell(e, z, what):
        if what in ("eta_A", "eta_B", "delta_A"):
            p = INFO[e]["parameters"][what]
            if what == "delta_A" and (INFO[e]["accessibility"] == "own_density" or INFO[e]["parameters"]["eta_A"] == 0):
                return "--"
            se = (INFO[e].get("estimation") or {}).get(what + "_se")
            fmt = "{:.4g}" if what == "delta_A" else "{:.4f}"
            return fmt.format(p) + (f" ({fmt.format(se)})" if se else "")
        if what == "LL":
            v = (INFO[e].get("estimation") or {}).get("LL")
            return f"{v:,.2f}" if v is not None else "--"
        if what.startswith("MALE"):
            v = what.split(":")[1]
            return f"{fdf.loc[(z, v), 'MALE_2015']:.4f}" if fit_rows and (z, v) in fdf.index else "--"
        if what == "benefit":
            return f"{ben[e]['reloc_total'] / 100:,.0f}" if e in ben else "--"
    tex += ["% ---- 表：集積の外部性の定式化に関する感度分析", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\hline",
            " & " + " & ".join(h for h, _, _ in cols) + r" \\ \hline"]
    for lab, what in [(r"$\eta^A$", "eta_A"), (r"$\delta^A$", "delta_A"), (r"$\eta^B$", "eta_B"), ("対数尤度", "LL")]:
        tex.append(lab + " & " + " & ".join(cell(e, z, what) for _, e, z in cols) + r" \\")
    tex.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{平均相対誤差（2015年）} \\")
    for v, lab in VARS:
        tex.append(r"\quad " + lab + " & " + " & ".join(cell(e, z, "MALE:" + v) for _, e, z in cols) + r" \\")
    tex.append("年間便益 [億円] & " + " & ".join(cell(e, z, "benefit") for _, e, z in cols) + r" \\ \hline")
    tex += [r"\end{tabular}", ""]

t0 = [(h, k) for h, k in TAU0_SENS if has(k, "estimation")]
if len(t0) < 2:
    t0 = []
if t0:
    tex += ["% ---- 表：べき乗型の τ0 に関する感度分析", r"\begin{tabular}{l" + "c" * len(t0) + "}", r"\hline",
            r"$\tau_0$（通勤トリップの分位点） & " + " & ".join(h for h, _ in t0) + r" \\",
            r"$\tau_0$ [円/往復] & " + " & ".join((f"{float(INFO[k]['tau_transform'].split(':')[1]):,.1f}" if INFO[k].get('tau_transform') else "--") for _, k in t0) + r" \\ \hline"]
    for lab, what in [(r"$\eta^A$", "eta_A"), (r"$\delta^A$", "delta_A")]:
        cells = []
        for _, k in t0:
            e = INFO[k].get("estimation") or {}
            v, se = INFO[k]["parameters"][what], e.get(what + "_se")
            cells.append(f"{v:.4f}" + (f" ({se:.4f})" if se else ""))
        tex.append(lab + " & " + " & ".join(cells) + r" \\")
    tex.append("対数尤度 & " + " & ".join(f"{(INFO[k].get('estimation') or {}).get('LL', float('nan')):,.2f}" for _, k in t0) + r" \\ \hline")
    tex += [r"\end{tabular}", ""]

# ---------------------------------------------------------------- 6. 構造残差の時点間比較（5. (2)）
tx_mask = None
gdf = None
if a.zones:
    import geopandas as gpd
    gdf = gpd.read_file(a.zones)
    if a.zone_field:
        gdf = gdf.set_index(gdf[a.zone_field].astype(int)).loc[kzone].reset_index(drop=True)
    elif len(gdf) != S:
        raise SystemExit(f"ゾーンの shapefile のレコード数 {len(gdf)} がゾーン数 {S} と一致しません。")
    gdf["kzone"] = kzone
if a.tx_zones:
    txz = set(pd.read_csv(a.tx_zones, encoding="utf-8-sig")["kzone"].astype(int))
    tx_mask = np.isin(kzone, list(txz))
tx_st = None
if a.tx_stations:
    import geopandas as gpd
    import pyogrio
    KEY = "常磐新線|首都圏新都市鉄道|つくばエクスプレス"   # 国土数値情報の路線名は「常磐新線」

    def find_col(df):
        return next((c for c in df.columns if c != "geometry" and df[c].astype(str).str.contains(KEY, regex=True).any()), None)

    lyr = None
    if a.tx_stations.lower().endswith(".gpkg"):      # rail_network_{year}.gpkg なら nodes レイヤー
        lyrs = [l[0] for l in pyogrio.list_layers(a.tx_stations)]
        lyr = "nodes" if "nodes" in lyrs else None
    tx_st, opcol = None, None
    for enc in (None, "cp932", "utf-8"):             # shapefile は文字コードの指定がないと文字化けする
        try:
            df = gpd.read_file(a.tx_stations, layer=lyr, encoding=enc) if enc else gpd.read_file(a.tx_stations, layer=lyr)
        except Exception:
            continue
        opcol = find_col(df)
        if opcol is not None:
            tx_st = df
            break
    if tx_st is None:
        raise SystemExit(f"駅データに「{KEY.replace('|', '」「')}」のいずれも見つかりません（列：{list(df.columns)}）。"
                         "rail_network_2015.gpkg（nodes レイヤー）を --tx_stations に指定することもできます。")
    print(f"TX の駅：{a.tx_stations} の列 {opcol}")
    tx_st = tx_st[tx_st[opcol].astype(str).str.contains(KEY, regex=True)].copy()
    if tx_st.geom_type.iloc[0] != "Point":            # N05 の駅はプラットホームの線
        tx_st["geometry"] = tx_st.to_crs(6677).geometry.centroid.to_crs(tx_st.crs)
    namec = next((c for c in tx_st.columns if c in ("name", "N05_011", "N02_005", "station_name")), None)
    if namec:                                         # N05 は年次ごとに同じ駅が複数行ある
        tx_st = tx_st.drop_duplicates(subset=namec)
    print(f"  {len(tx_st)} 駅" + (f"：{'，'.join(tx_st[namec].astype(str))}" if namec else ""))
if not a.tx_zones and tx_st is not None and gdf is not None:
    st = tx_st
    g = gdf.to_crs(6677)          # 平面直角座標系 IX 系（m）
    s = st.to_crs(6677)
    near = g.geometry.apply(lambda poly: s.distance(poly).min() <= a.tx_radius_km * 1000)
    tx_mask = near.to_numpy()
    lst = pd.DataFrame({"kzone": kzone[tx_mask]})
    lst.to_csv(os.path.join(a.out, "tx_zones.csv"), index=False, encoding="utf-8-sig")
    md.append(f"TX沿線ゾーン：TX駅（{len(s)}駅）から {a.tx_radius_km} km 以内にかかるゾーン {tx_mask.sum()} 個"
              f"（analysis/tx_zones.csv．編集して --tx_zones で指定し直せます）\n")

if has(B):
    eps = read_eps(B)
    rows = []
    for k in "ABG":
        e0 = eps[k][:, 0] - eps[k][:, 0].mean()
        e1 = eps[k][:, 1] - eps[k][:, 1].mean()
        r = {"eps": k, "corr_all": np.corrcoef(e0, e1)[0, 1], "N_all": S}
        if tx_mask is not None and tx_mask.sum() > 2:
            from scipy import stats
            d = e1 - e0
            r.update({"corr_tx": np.corrcoef(e0[tx_mask], e1[tx_mask])[0, 1], "N_tx": int(tx_mask.sum()),
                      "mean_change_tx": d[tx_mask].mean(), "mean_change_other": d[~tx_mask].mean()})
            tt = stats.ttest_ind(d[tx_mask], d[~tx_mask], equal_var=False)
            r.update({"t_diff": tt.statistic, "p_diff": tt.pvalue})
        rows.append(r)
    edf = pd.DataFrame(rows)
    sheets["構造残差の時点間比較"] = edf
    zt_eps = pd.DataFrame({"kzone": kzone, **{f"eps_{k}_{y}": eps[k][:, t] - eps[k][:, t].mean()
                                            for k in "ABG" for t, y in enumerate(YEARS)}})
    if tx_mask is not None:
        zt_eps["tx"] = tx_mask.astype(int)
    sheets["構造残差（ゾーン別）"] = zt_eps
    if tx_mask is not None and "corr_tx" in edf:
        f = lambda c, fmt: " & ".join(format(x, fmt) for x in edf[c])
        tex += ["% ---- 表：構造残差の時点間比較", r"\begin{tabular}{lccc}", r"\hline",
                r" & $\epsilon^A$ & $\epsilon^B$ & $\epsilon^G$ \\ \hline",
                f"相関係数（全ゾーン） & {f('corr_all', '.3f')} \\\\", f"相関係数（TX沿線） & {f('corr_tx', '.3f')} \\\\",
                f"平均変化量（TX沿線） & {f('mean_change_tx', '+.4f')} \\\\",
                f"平均変化量（その他） & {f('mean_change_other', '+.4f')} \\\\",
                f"差の $t$ 値 & {f('t_diff', '.2f')} \\\\ \\hline", r"\end{tabular}", ""]
    else:
        md.append("※ TX沿線ゾーンが指定されていないため，構造残差の比較は全ゾーンの相関のみです（--tx_zones か --zones と --tx_stations を指定）。\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs2 = plt.subplots(2, 2, figsize=(4.8, 4.9))
    axs = axs2.flat
    for ax, k in zip(axs, "ABG"):
        x, y = zt_eps[f"eps_{k}_1995"], zt_eps[f"eps_{k}_2015"]
        m = tx_mask if tx_mask is not None else np.zeros(S, bool)
        ax.scatter(x[~m], y[~m], s=5, color="0.6", lw=0, label="Other zones")
        if m.any():
            ax.scatter(x[m], y[m], s=12, color="#d62728", lw=0, label="TX corridor")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        ax.set_title(f"({'abc'['ABG'.index(k)]}) $\\epsilon^{k}$", fontsize=9)
        ax.set_xlabel("1995", fontsize=8); ax.set_ylabel("2015", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
    ax4 = axs2.flat[3]; ax4.set_axis_off()
    h = [plt.Line2D([], [], ls="", marker="o", ms=4, color="0.6", label="Other zones")]
    if tx_mask is not None:
        h.append(plt.Line2D([], [], ls="", marker="o", ms=5, color="#d62728", label="TX corridor zones"))
    h.append(plt.Line2D([], [], color="k", ls="--", lw=0.7, label="2015 = 1995"))
    ax4.legend(handles=h, loc="center", frameon=False, fontsize=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    fig.savefig(os.path.join(a.out, "eps_1995_2015-crop.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------------------------------------------------------- 7. TX の効果（ゾーン別）
if has(B) and os.path.exists(out_file(B, "lam_R_CF.csv")):
    Lb = lbar_of(INFO[B])
    t = 1
    def raw(name, sfx):
        return {v: read_tbl(out_file(name, f"{v}_{sfx}.csv"))[:, t] for v in ["lam_R", "lam_M", "p_G"]}
    cf, rl = raw(B, "CF"), raw(B, "Real")
    tx = pd.DataFrame({"kzone": kzone, "area_km2": G})
    for v, nm in [("lam_R", "L_R_per_G"), ("lam_M", "L_M_per_G")]:
        tx[f"{nm}_without"] = Lb[t] * cf[v] / G
        tx[f"{nm}_with"] = Lb[t] * rl[v] / G
        tx[f"D_{nm}"] = tx[f"{nm}_with"] - tx[f"{nm}_without"]
    tx["land_price_without"] = cf["p_G"] / rate      # 百万円/km² = 円/m²
    tx["land_price_with"] = rl["p_G"] / rate
    tx["D_land_price"] = tx["land_price_with"] - tx["land_price_without"]
    tx.to_csv(os.path.join(a.out, "tx_effect_zones.csv"), index=False, encoding="utf-8-sig")
    sheets["TXの効果（ゾーン別）"] = tx
    md.append("## 5. (3) TX の効果（基準ケース，2015年）\n")
    md.append(f"- 地価の変化量（円/m²，金利 {rate}）：最大 {tx.D_land_price.max():,.0f}，"
              f"上位10ゾーンの範囲 {tx.D_land_price.nlargest(10).min():,.0f}～{tx.D_land_price.max():,.0f}")
    if tx_mask is not None:
        q = tx.D_land_price[tx_mask]
        md.append(f"- TX沿線ゾーンの地価の変化量：中央値 {q.median():,.0f}，四分位 {q.quantile(.25):,.0f}～{q.quantile(.75):,.0f}")
    md.append(f"- 居住者数密度の変化量：最大 {tx.D_L_R_per_G.max():,.0f}，最小 {tx.D_L_R_per_G.min():,.0f} 人/km²")
    md.append(f"- 従業者数密度の変化量：最大 {tx.D_L_M_per_G.max():,.0f}，最小 {tx.D_L_M_per_G.min():,.0f} 人/km²\n")
elif has(B):
    warn(f"{B}：lam_R_CF.csv がないため TX の効果のゾーン別の値を作れません（新しい simulation2 で実行し直してください）。")

# ---------------------------------------------------------------- 8. 地図
if gdf is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def nice(x):
        if x == 0:
            return 0.0
        e = 10 ** np.floor(np.log10(abs(x)))
        return float(np.round(x / e) * e)

    def auto_breaks(v):
        """0 を境に，正負それぞれ3階級（分位点を丸めた値）"""
        br = []
        neg, pos = -v[v < 0], v[v > 0]
        if len(neg):
            br += sorted({-nice(np.quantile(neg, q)) for q in (0.9, 0.5)})
        br.append(0.0)
        if len(pos):
            br += sorted({nice(np.quantile(pos, q)) for q in (0.5, 0.9)})
        return sorted(set(br))

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib.patches import Rectangle
    # 地図は平面直角座標系（第IX系，m）で描く．距離標を正しくするため
    gplot = gdf.to_crs(6677) if gdf.crs is not None else gdf.copy()
    txline = None
    if tx_st is not None:
        s_ = tx_st.to_crs(gplot.crs)
        xy = np.c_[s_.geometry.x, s_.geometry.y]
        d = xy - xy.mean(0)
        txline = xy[np.argsort(d @ np.linalg.svd(d, full_matrices=False)[2][0])]   # 主軸方向に並べて線で結ぶ
    # 都心部の拡大図の範囲（東京駅を中心に ±INSET_HALF m）
    INSET_HALF = 12000.0
    try:
        tokyo = gpd.GeoSeries([gpd.points_from_xy([139.7671], [35.6812])[0]], crs=4326).to_crs(gplot.crs).iloc[0]
        INSET_C = (tokyo.x, tokyo.y)
    except Exception:
        INSET_C = None

    def map_decor(ax, inset_box=None):
        """方位記号と距離標"""
        ax.annotate("N", xy=(0.95, 0.97), xytext=(0.95, 0.87), xycoords="axes fraction", ha="center", va="center",
                    fontsize=7, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))
        sb = AnchoredSizeBar(ax.transData, 20000, "20 km", loc="lower left", pad=0.3, borderpad=0.4, sep=3,
                             frameon=False, size_vertical=600, fontproperties={"size": 6})
        ax.add_artist(sb)

    def draw(values, fname, title, breaks=None, fmt="{:,.0f}", ax=None, legend_fs=6, show_tx=False, inset=False):
        v = np.asarray(values, float)
        if breaks is None:
            breaks = auto_breaks(v)
        edges = [min(v.min(), breaks[0])] + list(breaks) + [max(v.max(), breaks[-1])]
        cls = np.clip(np.searchsorted(breaks, v, side="right"), 0, len(breaks))
        nneg = sum(b <= 0 for b in breaks)            # 0 以下の境界の数 = 負側の階級数
        npos = len(breaks) + 1 - nneg
        cols = list(plt.cm.Blues_r(np.linspace(0.15, 0.75, max(nneg, 1))))[:nneg] + \
               list(plt.cm.Reds(np.linspace(0.2, 0.9, max(npos, 1))))[:npos]
        if npos and nneg:                              # 0 をまたぐ階級を白に近くする
            cols[nneg] = (0.97, 0.97, 0.97, 1.0)
        g = gplot.copy()
        g["c"] = [cols[c] for c in cls]
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(3.6, 3.6))     # 1段の幅にそのまま載る大きさ（縮小しない）
        # 境界線は細く薄く（都心の小ゾーンで線が色を覆い隠さないように）
        g.plot(color=g["c"], ax=ax, lw=0.05, edgecolor="0.55", rasterized=True)

        def tx_on(a_):
            if show_tx and txline is not None:
                a_.plot(txline[:, 0], txline[:, 1], color="0.15", lw=0.8, ls=(0, (4, 2)), zorder=3)

        tx_on(ax)
        if inset and INSET_C is not None:              # 都心部の拡大図（左上）
            x0, y0 = INSET_C[0] - INSET_HALF, INSET_C[1] - INSET_HALF
            x1, y1 = INSET_C[0] + INSET_HALF, INSET_C[1] + INSET_HALF
            axin = ax.inset_axes([0.0, 0.58, 0.40, 0.40])
            g.cx[x0:x1, y0:y1].plot(color=g.cx[x0:x1, y0:y1]["c"], ax=axin, lw=0.15, edgecolor="0.4", rasterized=True)
            tx_on(axin)
            axin.set_xlim(x0, x1); axin.set_ylim(y0, y1); axin.set_aspect("equal")
            axin.set_xticks([]); axin.set_yticks([]); axin.set_xlabel(""); axin.set_ylabel("")
            for sp_ in axin.spines.values():
                sp_.set_linewidth(0.6)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.6, ec="k", zorder=5))
            axin.add_artist(AnchoredSizeBar(axin.transData, 5000, "5 km", loc="lower left", pad=0.2, borderpad=0.3,
                                            sep=2, frameon=False, size_vertical=150, fontproperties={"size": 6}))
        handles = [Patch(facecolor=cols[k], edgecolor="0.35", lw=0.4,
                         label=f"{fmt.format(edges[k])} – {fmt.format(edges[k + 1])}") for k in range(len(cols))]
        ax.legend(handles=handles[::-1], title=title, loc="lower right", fontsize=legend_fs,
                  title_fontsize=legend_fs + 0.5, frameon=True)
        map_decor(ax)
        ax.set_axis_off()
        if own:
            fig.savefig(os.path.join(a.out, fname), bbox_inches="tight", pad_inches=0.02, dpi=400)
            plt.close(fig)

    names = {"L_R": "residential density", "L_M": "employment density", "p_G": "relative land rent"}
    if Z in zone_tables:
        for v, lab in VARS:
            draw(zt[f"{v}_logratio_2015"].to_numpy(), f"err_map_{v}-crop.pdf",
                 "ln(pred./obs.)", breaks=[-1, -0.5, -0.2, 0.2, 0.5, 1], fmt="{:+.1f}")
    if "TXの効果（ゾーン別）" in sheets:
        draw(tx.D_L_R_per_G.to_numpy(), "D_L_R_per_G-crop.pdf", "persons/km²", show_tx=True)
        draw(tx.D_L_M_per_G.to_numpy(), "D_L_M_per_G-crop.pdf", "persons/km²", show_tx=True)
        draw(tx.D_land_price.to_numpy(), "D_land_price-crop.pdf", "JPY/m²", show_tx=True)
        # 2×2 にまとめた図（4枠目は空き）．地図は細かいので，2段抜き（figure*）での掲載を想定
        fig, axs = plt.subplots(2, 2, figsize=(9, 9))
        for ax, col, ttl in zip(axs.flat, ["D_L_R_per_G", "D_L_M_per_G", "D_land_price"],
                                ["(a) Residential density\n(persons/km²)", "(b) Employment density\n(persons/km²)",
                                 "(c) Land price\n(JPY/m², 2015)"]):
            draw(tx[col].to_numpy(), None, ttl, ax=ax, legend_fs=8, show_tx=True, inset=True)
        axs.flat[3].set_axis_off()
        if tx_st is not None:
            axs.flat[3].legend(handles=[plt.Line2D([], [], color="0.15", lw=0.8, ls=(0, (4, 2)), label="Tsukuba Express"),
                                        Rectangle((0, 0), 1, 1, fill=False, lw=0.6, ec="k", label="Area of the inset map")],
                               loc="center", frameon=False, fontsize=9)
        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        fig.savefig(os.path.join(a.out, "D_maps_2x2-crop.pdf"), bbox_inches="tight", pad_inches=0.02, dpi=400)
        plt.close(fig)

# ---------------------------------------------------------------- 書き出し
with pd.ExcelWriter(os.path.join(a.out, "summary.xlsx")) as xw:
    for name, df in sheets.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)
with open(os.path.join(a.out, "tables.tex"), "w", encoding="utf-8") as f:
    f.write("% analyze.py が自動生成．数値は cases/ の結果から計算\n\n" + "\n".join(tex))
with open(os.path.join(a.out, "numbers.md"), "w", encoding="utf-8") as f:
    f.write("# 集計結果\n\n")
    if WARN:
        f.write("## ※ 警告\n\n" + "\n".join(f"- {w}" for w in WARN) + "\n\n")
    f.write("集計したケース：" + "，".join(INFO) + "\n\n")
    f.write("\n".join(md) + "\n")
print(f"\n{a.out} に出力しました（警告 {len(WARN)} 件）。")
