import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import glob
import re
import os

# 集計する従業者数の列
#   F3  : 全産業 従業者総数（国・地方公共団体の事業所を含む）
#   F642: 民営事業所の従業者総数
# 2015年次に2014年経済センサス‐基礎調査（国・地方公共団体を含む全産業）を用いるため，こちらも全産業（F3）を用いる
EMP_COL = 'F3'

def apportion_employees_from_raw():
    print("ゾーンデータを読み込んでいます...")
    zones = gpd.read_file('H30_kzone.shp')
    # 順番を保持するためのid
    zones['id'] = range(len(zones))
    kzone_col = 'kzone' if 'kzone' in zones.columns else zones.columns[0]

    print("CSVデータ（4桁数字.csv）を読み込んでいます...")
    # フォルダ内の「4桁の数字.csv」に合致するファイルをすべて取得
    csv_files = [f for f in glob.glob('*.csv') if re.match(r'^\d{4}\.csv$', os.path.basename(f))]
    
    if not csv_files:
        raise FileNotFoundError("4桁の数字名を持つCSVファイルが見つかりません。")

    df_list = []
    for f in csv_files:
        try:
            # メッシュコード(F1)と従業者数の列(EMP_COL)のみをShift-JISで読み込む
            df_temp = pd.read_csv(f, usecols=['F1', EMP_COL], encoding='shift-jis')
            df_list.append(df_temp)
        except UnicodeDecodeError:
            # Shift-JISでエラーになった場合はUTF-8で再試行
            df_temp = pd.read_csv(f, usecols=['F1', EMP_COL], encoding='utf-8')
            df_list.append(df_temp)

    # 読み込んだすべてのデータを縦に結合
    df = pd.concat(df_list, ignore_index=True)
    
    # 念のため、F1（メッシュコード）の重複を削除（最初に読み込まれたものを残す）
    df = df.drop_duplicates(subset=['F1']).copy()
    
    # メッシュコードを8桁の文字列化（先頭のゼロ落ち防止など）
    df['mesh_code_str'] = df['F1'].astype(str).str.zfill(8)

    print(f"結合完了: ユニークなメッシュ数は {len(df):,} 件です。")
    print("メッシュコードからポリゴンを生成しています...")
    
    # コードの各桁を抽出
    YY = df['mesh_code_str'].str[0:2].astype(int)
    XX = df['mesh_code_str'].str[2:4].astype(int)
    J = df['mesh_code_str'].str[4:5].astype(int)
    I = df['mesh_code_str'].str[5:6].astype(int)
    j = df['mesh_code_str'].str[6:7].astype(int)
    i = df['mesh_code_str'].str[7:8].astype(int)

    # 南西端（左下）の緯度経度を計算
    lat_sw = YY / 1.5 + J / (1.5 * 8) + j / (1.5 * 8 * 10)
    lon_sw = 100 + XX + I / 8 + i / (8 * 10)
    
    # 北東端（右上）の緯度経度を計算
    lat_ne = lat_sw + 1 / 120.0
    lon_ne = lon_sw + 1 / 80.0

    # 4つの角から四角形のポリゴン(Bounding Box)を作成
    geometries = [box(minx, miny, maxx, maxy) for minx, miny, maxx, maxy in zip(lon_sw, lat_sw, lon_ne, lat_ne)]
    
    # ジオデータフレームの作成
    # 1996年事業所・企業統計調査（CSIS提供）の地域メッシュは日本測地系（旧東京測地系）で定義されているため EPSG:4301 とする
    # （2015年データとの空間パターンの照合により確認．世界測地系として扱うと東京付近で約400m ずれる）
    mesh_gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4301")

    print("座標変換と面積按分処理を実行しています...")
    # ゾーンデータの投影座標系に変換
    mesh_gdf = mesh_gdf.to_crs(zones.crs)
    
    # ★計算量削減の工夫：ゾーンの最大範囲（バウンディングボックス）に含まれるメッシュだけに絞り込む
    minx, miny, maxx, maxy = zones.total_bounds
    mesh_gdf = mesh_gdf.cx[minx:maxx, miny:maxy]

    # 按分計算用に、各メッシュの正確な面積（平面直角座標系上の面積）を計算
    mesh_gdf['original_area'] = mesh_gdf.geometry.area

    # 空間結合（ゾーンとメッシュの交差部分を切り出し）
    intersected = gpd.overlay(zones, mesh_gdf, how='intersection')
    
    # 交差部分の面積を計算
    intersected['intersect_area'] = intersected.geometry.area

    # 按分割合を計算し、従業者数を割り当てる
    intersected['ratio'] = intersected['intersect_area'] / intersected['original_area']
    intersected['apportioned_emp'] = pd.to_numeric(intersected[EMP_COL], errors='coerce').fillna(0) * intersected['ratio']

    print("結果を集計しています...")
    # idとkzoneでグループ化し、按分された従業者数を合計
    emp_sum = intersected.groupby(['id', kzone_col])['apportioned_emp'].sum().reset_index()

    # 全てのゾーンを網羅するためLeft Join (メッシュがないゾーンは0)
    result = zones[['id', kzone_col]].merge(emp_sum, on=['id', kzone_col], how='left')
    result['apportioned_emp'] = result['apportioned_emp'].fillna(0)
    
    # 結果の数値を四捨五入して整数にする
    #result['apportioned_emp'] = result['apportioned_emp'].round(0).astype(int)

    # 出力形式の調整
    result.rename(columns={'apportioned_emp': '従業者数_1995'}, inplace=True)
    result = result.sort_values('id')

    # CSV出力
    out_csv = 'employee_1995_apportioned_corrected.csv'
    result.to_csv(out_csv, index=False, encoding='utf-8-sig')
    
    # Excel形式で出力する場合 (openpyxlがインストールされている必要があります)
    try:
        excel_filename = 'employee_1995_apportioned_corrected.xlsx'
        result.to_excel(excel_filename, index=False)
        print(f"処理完了: {excel_filename} に出力しました。")
    except ImportError:
        print("openpyxlライブラリがないため、Excelファイルの出力はスキップされました。(CSVは出力済)")
    
    print(f"\n--- 処理完了 ---")
    print(f"結果を {out_csv} に出力しました。")
    print(f"東京都市圏ゾーン内の従業者数合計: {result['従業者数_1995'].sum():,} 人")

    return result

if __name__ == "__main__":
    df_result = apportion_employees_from_raw()
    print(df_result.head())