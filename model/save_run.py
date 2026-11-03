# -*- coding: utf-8 -*-
"""
save_run.py ― 実行結果の退避

output/ の全ファイルと，その実行に使った input/input_data.xlsx・環境変数の設定を，
runs/<名前>_<日時>/ にコピーする．estimation / simulation / simulation2 は output/ を上書きするので，
各実行の直後にこれを実行する．

使い方：
  python save_run.py base_mass            （例：基準ケースの推定結果）
  python save_run.py delta_x2 --note "δA, δB を 2 倍"
  python save_run.py base_mass --restore   （退避した結果を output/ と input/ に戻す．最新のものを使う）
"""
import argparse
import datetime
import glob
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--note", default="")
ap.add_argument("--restore", action="store_true",
                help="runs/<name>_* の最新の結果を output/ と input/input_data.xlsx に戻す")
a = ap.parse_args()

if a.restore:
    cands = sorted(glob.glob(os.path.join("runs", f"{a.name}_*")))
    if not cands:
        raise SystemExit(f"runs/{a.name}_* が見つかりません。")
    src = cands[-1]
    os.makedirs("output", exist_ok=True)
    for f in glob.glob(os.path.join(src, "output", "*")):
        shutil.copy2(f, "output")
    shutil.copy2(os.path.join(src, "input_data.xlsx"), os.path.join("input", "input_data.xlsx"))
    print(f"{src} を output/ と input/input_data.xlsx に戻しました。")
    raise SystemExit

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dst = os.path.join("runs", f"{a.name}_{stamp}")
os.makedirs(os.path.join(dst, "output"))
for f in glob.glob(os.path.join("output", "*")):
    if os.path.isfile(f):
        shutil.copy2(f, os.path.join(dst, "output"))
shutil.copy2(os.path.join("input", "input_data.xlsx"), dst)
for f in glob.glob(os.path.join("input", "epsilon_*.csv")):
    os.makedirs(os.path.join(dst, "input_epsilon"), exist_ok=True)
    shutil.copy2(f, os.path.join(dst, "input_epsilon"))
with open(os.path.join(dst, "run_info.json"), "w", encoding="utf-8") as fo:
    json.dump({"name": a.name, "saved_at": stamp, "note": a.note,
               "QUM_ACCESSIBILITY": os.environ.get("QUM_ACCESSIBILITY", "mass")},
              fo, ensure_ascii=False, indent=2)
print(f"退避しました：{dst}（output のファイル {len(os.listdir(os.path.join(dst, 'output')))} 個）")
