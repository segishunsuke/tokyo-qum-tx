# -*- coding: utf-8 -*-
"""
run_cases.py ― 分析ケースの一括実行

cases.csv に書いたケースを，ケースごとに独立したフォルダ cases/<ケース名>/ で実行する．
input/ の元データ（input_data.xlsx と tau ファイル）は一切書き換えない．

  cases/<ケース名>/input/      このケース用の input_data.xlsx（パラメータを書き換えたもの），tau ファイル，構造残差
  cases/<ケース名>/output/     estimation / simulation / simulation2 の出力
  cases/<ケース名>/log.txt     実行時の画面出力
  cases/<ケース名>/case_info.json   実際に使ったパラメータ，構造残差の扱い，推定値・標準誤差など

使い方：
  python run_cases.py                    （cases.csv の全ケースを上から順に実行．完了済みのケースは飛ばす）
  python run_cases.py zero_base zero_noagg   （指定したケースのみ）
  python run_cases.py --force base       （完了済みでもやり直す）
  python run_cases.py --list             （ケースの一覧と状態を表示）

cases.csv の列：
  case           ケース名（フォルダ名になる）
  parent         パラメータを引き継ぐケース．空欄なら input/input_data.xlsx．
                 親ケースの「実行後の」input_data.xlsx（推定値・Ā が書き込まれたもの）を出発点にする
  accessibility  mass / own_density．空欄なら親と同じ
  delta_scale    δA, δB に掛ける倍率．空欄なら 1（親の値のまま）
  eta_A, eta_B   数値 → その値に固定．est → 推定する．空欄 → 親の値に固定
  epsilon        estimate → estimation で逆算した構造残差を使う（steps に estimation が必要）
                 zero     → 構造残差ゼロ
                 parent   → 親ケースの構造残差をそのまま使う
  steps          estimation, abar, simulation, simulation2 を + でつないだもの（この順に実行）
                 abar = 構造残差ありで simulation → Ā の調整 → simulation → 確認
  note           メモ（結果には影響しない）
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import calibrate_Abar as cab

ROOT = os.path.dirname(os.path.abspath(__file__))
MASTER_INPUT = os.path.join(ROOT, "input")
CASES_DIR = os.path.join(ROOT, "cases")
EPS_FILES = ["epsilon_A.csv", "epsilon_B.csv", "epsilon_G.csv"]
STEPS = ["estimation", "abar", "simulation", "simulation2"]

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="*")
ap.add_argument("--cases_file", default=os.path.join(ROOT, "cases.csv"))
ap.add_argument("--force", action="store_true")
ap.add_argument("--list", action="store_true")
ap.add_argument("--only", choices=["simulation", "simulation2"],
                help="完了済みのケースで，このステップだけを今の入力のまま実行し直す（推定・Ā の調整はやり直さない）")
ap.add_argument("--target_wage", type=float, default=4.51412878932616, help="Ā の調整の目標平均賃金 [百万円]")
ap.add_argument("--period", type=int, default=1)
a = ap.parse_args()


# ---------------------------------------------------------------- 補助関数
def load_cases():
    df = pd.read_csv(a.cases_file, dtype=str, comment="#", encoding="utf-8-sig").fillna("")
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    for col in ("set", "tau_transform", "delta_A", "delta_B", "tau_ii", "decay_transform"):          # 後から追加した列（古い cases.csv との互換のため）
        if col not in df.columns:
            df[col] = ""
    if df["case"].duplicated().any():
        raise SystemExit(f"cases.csv にケース名の重複があります：{df.loc[df['case'].duplicated(), 'case'].tolist()}")
    rows = {}
    for _, r in df.iterrows():
        steps = [s.strip() for s in r["steps"].split("+") if s.strip()]
        bad = [s for s in steps if s not in STEPS]
        if bad:
            raise SystemExit(f"[{r['case']}] steps に不明な項目があります：{bad}")
        if r["epsilon"] not in ("estimate", "zero", "parent"):
            raise SystemExit(f"[{r['case']}] epsilon は estimate / zero / parent のいずれかです：{r['epsilon']!r}")
        if r["epsilon"] == "estimate" and "estimation" not in steps:
            raise SystemExit(f"[{r['case']}] epsilon=estimate には steps に estimation が必要です。")
        if r["epsilon"] == "parent" and not r["parent"]:
            raise SystemExit(f"[{r['case']}] epsilon=parent には parent の指定が必要です。")
        if "abar" in steps and r["epsilon"] == "zero":
            raise SystemExit(f"[{r['case']}] 構造残差ゼロでは Ā を調整できません（abar を外し，Ā は親から引き継いでください）。")
        if r["parent"] and r["parent"] not in rows:
            raise SystemExit(f"[{r['case']}] 親ケース {r['parent']} は cases.csv でこのケースより上に書いてください。")
        rows[r["case"]] = dict(r, steps=steps)
    return rows


def case_dir(name):
    return os.path.join(CASES_DIR, name)


def read_info(name):
    p = os.path.join(case_dir(name), "case_info.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_info(name, info):
    with open(os.path.join(case_dir(name), "case_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)          # 同じドライブならハードリンク（容量を使わない）
    except OSError:
        shutil.copy2(src, dst)


EXTRA_ENV = {}


def tau_ii_env(v):
    """tau_ii 列（ゾーン内の費用の CSV．例：input/tau_ii.csv）を環境変数に変換する．空欄なら τ_ii = cost_per_root_area × sqrt(G)"""
    v = str(v).strip()
    if not v:
        return {"QUM_TAU_II_FILE": ""}
    p = v if os.path.isabs(v) else os.path.join(ROOT, v)
    if not os.path.exists(p):
        raise SystemExit(f"tau_ii 列のファイルがありません：{p}")
    return {"QUM_TAU_II_FILE": p}


def run_module(module, inp, out, accessibility, log):
    env = dict(os.environ, QUM_INPUT_DIR=inp, QUM_OUTPUT_DIR=out, QUM_ACCESSIBILITY=accessibility, **EXTRA_ENV,
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    log.write(f"\n===== {module}（{datetime.datetime.now():%Y-%m-%d %H:%M:%S}）=====\n")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", f"import {module}"], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for raw in p.stdout:
        line = raw.decode("utf-8", errors="replace")
        lines.append(line)
        log.write(line)
        log.flush()
        sys.stdout.write("    " + line)
    p.wait()
    log.write(f"===== {module} 終了：コード {p.returncode}，{time.time() - t0:.0f} 秒 =====\n")
    return p.returncode, "".join(lines)


def params_of(xlsx):
    par = pd.read_excel(xlsx, sheet_name="parameters", header=None)
    out = {}
    for _, r in par.iloc[1:].iterrows():
        k = str(r[0]).strip()
        try:
            out[k] = float(r[1])
        except (TypeError, ValueError):
            continue
        if k in ("theta", "eta_A", "eta_B", "delta_A", "delta_B"):
            out[k + "_fixed"] = int(float(r[2])) if str(r[2]) not in ("", "nan") else 1
    return out


def parse_estimation(out_dir, text, free):
    res = {}
    pe = os.path.join(out_dir, "parameters_estimated.csv")
    with open(pe, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                res[row[0].strip()] = float(row[1])
    se_path = os.path.join(out_dir, "standard error.csv")
    if free and os.path.exists(se_path):
        se = [float(x) for x in open(se_path, encoding="utf-8-sig").read().split()]
        for k, v in zip(free, se):
            res[k + "_se"] = v
    pats = {"within_corr": r"個体内相関係数:\s*([-\d.eE+]+)", "sigma_A": r"sigma\[0,0\]:\s*([-\d.eE+]+)",
            "sigma_B": r"sigma\[1,1\]:\s*([-\d.eE+]+)", "sigma_G": r"sigma\[2,2\]:\s*([-\d.eE+]+)",
            "rho_BA": r"rho\[1,0\]:\s*([-\d.eE+]+)", "rho_GA": r"rho\[2,0\]:\s*([-\d.eE+]+)",
            "rho_GB": r"rho\[2,1\]:\s*([-\d.eE+]+)"}
    for k, p in pats.items():
        m = re.findall(p, text)
        if m:
            res[k] = float(m[-1])
    return res


# ---------------------------------------------------------------- 1ケースの実行
def run_case(c, cases):
    name = c["case"]
    d = case_dir(name)
    inp, out = os.path.join(d, "input"), os.path.join(d, "output")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(inp)
    os.makedirs(out)

    # --- 出発点の input_data.xlsx と定式化
    if c["parent"]:
        pinfo = read_info(c["parent"])
        if not pinfo or pinfo.get("status") != "done":
            raise SystemExit(f"[{name}] 親ケース {c['parent']} が完了していません。")
        src_xlsx = os.path.join(case_dir(c["parent"]), "input", "input_data.xlsx")
        acc = c["accessibility"] or pinfo["accessibility"]
        ttf = c["tau_transform"] or pinfo.get("tau_transform", "")
        dtf = c["decay_transform"] or pinfo.get("decay_transform", pinfo.get("tau_transform", ""))
        tii = c["tau_ii"] or pinfo.get("tau_ii", "")
    else:
        src_xlsx = os.path.join(MASTER_INPUT, "input_data.xlsx")
        acc = c["accessibility"] or "mass"
        ttf = c["tau_transform"]
        dtf = c["decay_transform"] or ttf          # 空欄なら通勤不効用と同じ関数形
        tii = c["tau_ii"]
    if acc not in ("mass", "own_density"):
        raise SystemExit(f"[{name}] accessibility は mass / own_density です：{acc!r}")
    xlsx = os.path.join(inp, "input_data.xlsx")
    shutil.copy2(src_xlsx, xlsx)
    p0 = params_of(xlsx)

    # --- パラメータの書き換え
    if ttf in ("none", "exp"):
        ttf = ""
    if dtf in ("none", "exp"):
        dtf = ""
    EXTRA_ENV.clear()
    EXTRA_ENV["QUM_TAU_TRANSFORM"] = ttf
    EXTRA_ENV["QUM_DECAY_TRANSFORM"] = dtf
    EXTRA_ENV.update(tau_ii_env(tii))
    if EXTRA_ENV["QUM_TAU_II_FILE"]:                  # 使ったファイルをケースの入力に複製して記録する
        shutil.copy2(EXTRA_ENV["QUM_TAU_II_FILE"], os.path.join(inp, "tau_ii.csv"))
        EXTRA_ENV["QUM_TAU_II_FILE"] = os.path.join(inp, "tau_ii.csv")
    # set 列：パラメータの上書き（例：theta=13.1;kappa_theta=2.76;delta_A=2.45;delta_B=3.55）
    for item in [x for x in c["set"].split(";") if x.strip()]:
        k, v = item.split("=")
        cab.set_param(xlsx, k.strip(), float(v))
    p0 = params_of(xlsx)
    ds = float(c["delta_scale"]) if c["delta_scale"] else 1.0
    if ds != 1.0:
        cab.set_param(xlsx, "delta_A", p0["delta_A"] * ds)
        cab.set_param(xlsx, "delta_B", p0["delta_B"] * ds)
    free = []
    if "estimation" in c["steps"] and not p0.get("theta_fixed", 1):
        free.append("theta")
    # eta_A, eta_B, delta_A, delta_B：
    #   est → 推定（初期値は親／input の値），est:<値> → その値を初期値として推定，
    #   数値 → その値に固定，空欄 → 親の値に固定（delta は delta_scale を掛けた値）
    p0 = params_of(xlsx)
    for k in ("eta_A", "eta_B", "delta_A", "delta_B"):
        v = c[k]
        if v.startswith("est"):
            if "estimation" not in c["steps"]:
                raise SystemExit(f"[{name}] {k}=est には steps に estimation が必要です。")
            start = float(v.split(":", 1)[1]) if ":" in v else p0[k]
            if start <= 0:
                start = 0.05                                  # 0 以下だと推定できない
            cab.set_param(xlsx, k, start, flag=0)
            free.append(k)
        elif v == "":
            cab.set_param(xlsx, k, p0[k], flag=1)
        else:
            cab.set_param(xlsx, k, float(v), flag=1)
    free = [k for k in ("theta", "eta_A", "eta_B", "delta_A", "delta_B") if k in free]

    # --- tau ファイル
    tau_files = sorted(glob.glob(os.path.join(MASTER_INPUT, "*tau[[]*].csv")))
    if not any(os.path.basename(f).startswith("tau[") for f in tau_files):
        raise SystemExit(f"input/ に tau[t].csv がありません。")
    for f in tau_files:
        link_or_copy(f, os.path.join(inp, os.path.basename(f)))
    if "simulation2" in c["steps"]:
        need = [f"{p}tau[{t}].csv" for p in ("CF_", "Real_") for t in range(int(p0["T"]))]
        miss = [f for f in need if not os.path.exists(os.path.join(inp, f))]
        if miss:
            raise SystemExit(f"[{name}] simulation2 に必要な {miss} が input/ にありません。")

    # --- 構造残差
    eps_init = ""
    if c["epsilon"] == "parent":
        for f in EPS_FILES:
            shutil.copy2(os.path.join(case_dir(c["parent"]), "input", f), os.path.join(inp, f))
    elif "estimation" in c["steps"]:
        # estimation の逆算の初期値（解には影響せず，収束の速さだけが変わる）．
        # 親ケースの構造残差，なければ input/epsilon_*.csv（以前の推定結果），どちらもなければゼロから
        src = os.path.join(case_dir(c["parent"]), "input") if c["parent"] else MASTER_INPUT
        if all(os.path.exists(os.path.join(src, f)) for f in EPS_FILES):
            for f in EPS_FILES:
                shutil.copy2(os.path.join(src, f), os.path.join(inp, f))
            eps_init = src
        print(f"  [{name}] 逆算の初期値：{eps_init or 'ゼロ（初期値のファイルなし．時間がかかります）'}")

    info = {"case": name, "status": "running", "started": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "parent": c["parent"], "accessibility": acc, "tau_transform": ttf, "decay_transform": dtf, "tau_ii": tii, "set": c["set"],
            "delta_scale": ds, "epsilon": c["epsilon"],
            "steps": c["steps"], "note": c.get("note", ""), "free_parameters": free,
            "epsilon_initial_values": eps_init}
    write_info(name, info)

    with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as log:
        for step in c["steps"]:
            print(f"  [{name}] {step}")
            if step == "estimation":
                t_start = time.time()
                code, text = run_module("estimation", inp, out, acc, log)
                ok = all(os.path.exists(os.path.join(out, f.replace(".csv", "_calibrated.csv")))
                         and os.path.getmtime(os.path.join(out, f.replace(".csv", "_calibrated.csv"))) >= t_start
                         for f in EPS_FILES)
                if code != 0:
                    if not (ok and not free):
                        raise SystemExit(f"[{name}] estimation が失敗しました。{d}/log.txt を確認してください。")
                    print(f"  [{name}] ※ estimation は終了コード {code} で終わりましたが，推定するパラメータがなく"
                          f"構造残差は出力されているので続行します（ヘッセ行列の計算でのエラーと思われます）。")
                est = parse_estimation(out, text, free)
                info["estimation"] = est
                for k in free:          # 推定値を xlsx に書き込む（以降のステップ・子ケースで使う）
                    cab.set_param(xlsx, k, est[k], flag=1)
                if c["epsilon"] == "estimate":
                    for f in EPS_FILES:
                        shutil.copy2(os.path.join(out, f.replace(".csv", "_calibrated.csv")), os.path.join(inp, f))
                elif c["epsilon"] == "zero":     # 初期値として置いたものを消す
                    for f in EPS_FILES:
                        if os.path.exists(os.path.join(inp, f)):
                            os.remove(os.path.join(inp, f))
            elif step == "abar":
                if run_module("simulation", inp, out, acc, log)[0] != 0:
                    raise SystemExit(f"[{name}] simulation が失敗しました。")
                info["Abar_before"] = params_of(xlsx)["Abar"]
                cab.calibrate(inp, out, a.target_wage, a.period)
                if run_module("simulation", inp, out, acc, log)[0] != 0:
                    raise SystemExit(f"[{name}] simulation が失敗しました。")
                wbar, worst = cab.check(inp, out, a.target_wage, a.period)
                if abs(wbar[a.period] / a.target_wage - 1) > 1e-6 or worst > 1e-4:
                    raise SystemExit(f"[{name}] Ā の調整後の確認に失敗しました（平均賃金 {wbar[a.period]}，観測値の誤差 {worst:.1e}）。")
                info["mean_wage"] = wbar.tolist()
            else:
                if run_module(step, inp, out, acc, log)[0] != 0:
                    raise SystemExit(f"[{name}] {step} が失敗しました。{d}/log.txt を確認してください。")
            write_info(name, info)

    # --- 実際に使われた条件を記録
    eps_present = all(os.path.exists(os.path.join(inp, f)) for f in EPS_FILES)
    if c["epsilon"] == "zero" and any(os.path.exists(os.path.join(inp, f)) for f in EPS_FILES):
        raise SystemExit(f"[{name}] 構造残差ゼロのケースに epsilon ファイルがあります（内部エラー）。")
    if c["epsilon"] != "zero" and not eps_present:
        raise SystemExit(f"[{name}] 構造残差ありのケースに epsilon ファイルがありません（内部エラー）。")
    info["parameters"] = params_of(xlsx)
    info["status"] = "done"
    info["finished"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    write_info(name, info)
    print(f"  [{name}] 完了")


# ---------------------------------------------------------------- メイン
cases = load_cases()
cab.validate_input(os.path.join(MASTER_INPUT, "input_data.xlsx"))
if a.list:
    for k, c in cases.items():
        info = read_info(k)
        st = info["status"] if info else "未実行"
        print(f"{k:20s} {st:8s} parent={c['parent'] or '-':12s} eps={c['epsilon']:8s} steps={'+'.join(c['steps'])}")
    raise SystemExit

targets = a.cases or list(cases)
if a.only:
    for k in targets:
        info = read_info(k)
        if not info or info.get("status") != "done":
            raise SystemExit(f"[{k}] 完了していないケースには --only を使えません。")
        if a.only not in info["steps"]:
            raise SystemExit(f"[{k}] このケースの steps に {a.only} がありません。")
        d = case_dir(k)
        EXTRA_ENV.clear()
        EXTRA_ENV["QUM_TAU_TRANSFORM"] = info.get("tau_transform", "")
        EXTRA_ENV["QUM_DECAY_TRANSFORM"] = info.get("decay_transform", info.get("tau_transform", ""))
        EXTRA_ENV.update(tau_ii_env(info.get("tau_ii", "")))
        if EXTRA_ENV["QUM_TAU_II_FILE"] and os.path.exists(os.path.join(d, "input", "tau_ii.csv")):
            EXTRA_ENV["QUM_TAU_II_FILE"] = os.path.join(d, "input", "tau_ii.csv")
        with open(os.path.join(d, "log.txt"), "a", encoding="utf-8") as log:
            print(f"[{k}] {a.only} のみ再実行")
            if run_module(a.only, os.path.join(d, "input"), os.path.join(d, "output"), info["accessibility"], log)[0] != 0:
                raise SystemExit(f"[{k}] {a.only} が失敗しました。")
        info.setdefault("reruns", []).append(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {a.only}")
        write_info(k, info)
    print("完了しました。")
    raise SystemExit
for k in targets:
    if k not in cases:
        raise SystemExit(f"cases.csv にケース {k} がありません。")
for k in targets:
    info = read_info(k)
    if info and info.get("status") == "done" and not a.force:
        print(f"[{k}] 完了済みのため飛ばします（やり直す場合は --force）")
        continue
    # 親をやり直すと子の結果は古くなるので，子孫ケースの完了状態を取り消す
    stack = [k]
    while stack:
        p = stack.pop()
        for kk, cc in cases.items():
            if cc["parent"] == p:
                stack.append(kk)
                inf = read_info(kk)
                if inf and inf.get("status") == "done":
                    inf["status"] = "stale"
                    write_info(kk, inf)
    print(f"[{k}] 開始")
    run_case(cases[k], cases)
print("すべて完了しました。結果の集計は python analyze.py")
