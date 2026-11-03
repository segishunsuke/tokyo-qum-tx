# -*- coding: utf-8 -*-
"""
calibrate_Abar.py ― 生産性のスケーリング定数 Ā の設定（論文 4. (10) c)）

通常は run_cases.py の abar ステップから自動で呼ばれる．単独で使う場合：
  1. estimation の出力 output/epsilon_*_calibrated.csv を input/epsilon_*.csv にコピーし，simulation を実行
  2. python calibrate_Abar.py --target_wage 4.51412878932616 --period 1
  3. もう一度 simulation を実行し，--check を付けて実行して確認

【理屈】
  Ā を c 倍し，全ゾーン・全時点の ε^G を一律に ln(w̄_目標 / w̄_モデル) だけ下げる（c = (w̄_目標 / w̄_モデル)^α）と，
  床面積・土地市場の均衡条件は地価を変えずに満たされ，賃金は全ゾーン一律に c^{1/α} 倍，
  効用は全ての (i, j) で一律にずれるため居住地・就業地の分布は変わらない．
  尤度は基準ゾーンとの差 ε^G_i − ε^G_S のみに依存するので，推定結果にも影響しない．

【変更するファイル】（変更前のものは *.bak に保存）
  <input>/input_data.xlsx の parameters シートの Abar，<input>/epsilon_G.csv
"""
import argparse
import csv
import os
import shutil

import numpy as np
import pandas as pd
from openpyxl import load_workbook


def read_pred(path):
    d = pd.read_csv(path)
    return d.loc[:, [c for c in d.columns if c.startswith("Period")]].to_numpy(dtype=float)  # S × T


def read_params(xlsx):
    par = pd.read_excel(xlsx, sheet_name="parameters", header=None)
    return dict(zip(par[0].astype(str).str.strip(), par[1]))


def set_param(xlsx, name, value, flag=None):
    """parameters シートの name の行の値（B列）と，必要なら固定フラグ（C列）を書き換える．

    openpyxl で保存すると数式セルの計算結果（キャッシュ）が失われ，pandas で読むと空になる
    （p_G シートの土地賃料は数式なので，estimation が NaN を読んでしまう）．
    そのため，数式セルは計算結果の値に置き換えてから保存する．
    """
    wb = load_workbook(xlsx)
    wv = load_workbook(xlsx, data_only=True)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str) and c.value.startswith("="):
                    v = wv[ws.title][c.coordinate].value
                    if v is None:
                        raise SystemExit(f"{xlsx} の {ws.title}!{c.coordinate} の数式に計算結果が保存されていません。"
                                         "Excel で開いて保存し直してから実行してください。")
                    c.value = v
    ws = wb["parameters"]
    for row in ws.iter_rows(min_row=1, max_col=3):
        if str(row[0].value).strip() == name:
            row[1].value = float(value)
            if flag is not None:
                row[2].value = int(flag)
            break
    else:
        raise SystemExit(f"parameters シートに {name} の行が見つかりません。")
    wb.save(xlsx)


def validate_input(xlsx):
    """観測値のシートが数値として読めるか（数式の計算結果が失われていないか）を確認する"""
    pv = read_params(xlsx)
    S, T = int(pv["S"]), int(pv["T"])
    for sh, ncol in [("lam_R", T), ("lam_M", T), ("p_G", T), ("area", 1)]:
        x = pd.read_excel(xlsx, sheet_name=sh, header=None).iloc[1:1 + S, 1:1 + ncol]
        x = x.apply(pd.to_numeric, errors="coerce")
        if x.shape != (S, ncol) or x.isna().to_numpy().any():
            raise SystemExit(f"{xlsx} の {sh} シートに数値として読めないセルがあります（数式の計算結果が失われている可能性）。"
                             "Excel で開いて保存し直すか，*.bak から戻してください。")


def check(input_dir, output_dir, target_wage, period, verbose=True):
    """観測値の再現と平均賃金を確認する．戻り値：(平均賃金の配列, 観測値の最大相対誤差)"""
    xlsx = os.path.join(input_dir, "input_data.xlsx")
    pv = read_params(xlsx)
    T = int(pv["T"])
    obs, pred = {}, {}
    for sh in ["lam_M", "lam_R", "p_G"]:
        x = pd.read_excel(xlsx, sheet_name=sh, header=None).iloc[1:, 1:1 + T].to_numpy(dtype=float)
        obs[sh] = x / x.sum(axis=0, keepdims=True) if sh != "p_G" else x
        pred[sh] = read_pred(os.path.join(output_dir, f"{sh}_predict.csv"))
    w = read_pred(os.path.join(output_dir, "w_predicted.csv"))
    wbar = (pred["lam_M"] * w).sum(axis=0)
    worst = 0.0
    if verbose:
        print(f"現在の Ā = {float(pv['Abar']):.6f}，α = {pv['alpha']}")
    for k in ["lam_M", "lam_R", "p_G"]:
        rel = np.abs(pred[k] / obs[k] - 1)
        bad = np.where(~np.isfinite(rel).all(axis=1))[0]
        if len(bad):
            print(f"  ※ {k} の予測値または観測値が不正（nan/inf/0）なゾーン：{bad.tolist()[:20]}（0始まりの番号）")
        m = np.nanmax(rel, axis=0)
        worst = max(worst, float(np.nanmax(m)))
        if verbose:
            print(f"  観測値の再現（{k}）：最大相対誤差 " + "，".join(f"t={t}: {m[t]:.2e}" for t in range(T)))
    if verbose:
        print("  平均賃金（モデル）：" + "，".join(f"t={t}: {wbar[t]:.6f}" for t in range(T))
              + f"（目標 t={period}: {target_wage}）")
    return wbar, worst


def calibrate(input_dir, output_dir, target_wage, period):
    xlsx = os.path.join(input_dir, "input_data.xlsx")
    eps_g = os.path.join(input_dir, "epsilon_G.csv")
    if not os.path.exists(eps_g):
        raise SystemExit(f"{eps_g} がありません。構造残差なしで Ā を調整することはできません。")
    validate_input(xlsx)
    pv = read_params(xlsx)
    alpha, Abar, T = float(pv["alpha"]), float(pv["Abar"]), int(pv["T"])
    wbar, worst = check(input_dir, output_dir, target_wage, period)
    if worst > 1e-4:
        raise SystemExit("観測値が再現されていません。epsilon_*.csv が estimation の出力と一致しているか確認してください。")
    ratio = target_wage / wbar[period]
    shift = float(np.log(ratio))
    Abar_new = Abar * ratio ** alpha
    print(f"Ā を {Abar:.6f} → {Abar_new:.6f} に変更し，ε^G を一律に {shift:+.6f} だけ下げます。")
    shutil.copy2(xlsx, xlsx + ".bak")
    set_param(xlsx, "Abar", Abar_new)
    shutil.copy2(eps_g, eps_g + ".bak")
    with open(eps_g, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        for t in range(T):
            r[1 + t] = repr(float(r[1 + t]) - shift)
    with open(eps_g, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return Abar_new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_wage", type=float, default=4.51412878932616, help="平均賃金の目標値 [百万円]")
    ap.add_argument("--period", type=int, default=1, help="目標を合わせる時点（0 = 1995年，1 = 2015年）")
    ap.add_argument("--check", action="store_true", help="調整後の simulation の結果を確認するだけ")
    ap.add_argument("--input", default=os.environ.get("QUM_INPUT_DIR", "input"))
    ap.add_argument("--output", default=os.environ.get("QUM_OUTPUT_DIR", "output"))
    a = ap.parse_args()
    if a.check:
        check(a.input, a.output, a.target_wage, a.period)
    else:
        calibrate(a.input, a.output, a.target_wage, a.period)
        print("完了しました。simulation を再実行し，--check で平均賃金が目標と一致することを確認してください。")
