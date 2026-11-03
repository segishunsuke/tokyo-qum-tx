# -*- coding: utf-8 -*-
# 2015年次の従業者数：2014年経済センサス‐基礎調査（世界測地系 500mメッシュ）の全産業従業者数
# （2016年経済センサス‐活動調査は民営事業所のみのため，公務等を含むこちらを用いる）
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import glob
import os

def apportion_employees_2014_500m():
    print("ゾーンデータを読み込んでいます...")
    zones = gpd.read_file('H30_kzone.shp')
    # 順番を保持するためのid
    zones['id'] = range(len(zones))
    kzone_col = 'kzone' if 'kzone' in zones.columns else zones.columns[0]

    print("2014年経済センサス‐基礎調査の500mメッシュデータ（tblT000842H*.txt）を読み込んでいます...")
    # 指定の形式のテキストファイルを全て取得（csv拡張子の場合も考慮）
    txt_files = glob.glob('tblT000842H*.txt') + glob.glob('tblT000842H*.csv')
    
    if not txt_files:
        raise FileNotFoundError("対象のテキスト(CSV)ファイルが見つかりません。")

    df_list = []
    for f in txt_files:
        # skiprows=[1] で2行目の日本語ヘッダーをスキップし、KEY_CODE と T000842002（全産業従業者数，国・地方公共団体の事業所を含む）を抽出
        try:
            df_temp = pd.read_csv(f, header=0, skiprows=[1], usecols=['KEY_CODE', 'T000842002'], encoding='shift-jis')
            df_list.append(df_temp)
        except Exception:
            # Shift-JISでダメならUTF-8で試行
            df_temp = pd.read_csv(f, header=0, skiprows=[1], usecols=['KEY_CODE', 'T000842002'], encoding='utf-8')
            df_list.append(df_temp)

    # 読み込んだ全てのデータを縦に結合
    df = pd.concat(df_list, ignore_index=True)
    
    # メッシュコードの重複があれば削除
    df = df.drop_duplicates(subset=['KEY_CODE']).copy()
    
    # 9桁の文字列に揃える（先頭のゼロ落ち対策）
    df['mesh_code_str'] = df['KEY_CODE'].astype(str).str.zfill(9)

    print(f"結合完了: ユニークな500mメッシュ数は {len(df):,} 件です。")
    print("メッシュコードからポリゴンを生成しています...")
    
    # 各桁の抽出（8桁目までは前回と同様）
    YY = df['mesh_code_str'].str[0:2].astype(int)
    XX = df['mesh_code_str'].str[2:4].astype(int)
    J  = df['mesh_code_str'].str[4:5].astype(int)
    I  = df['mesh_code_str'].str[5:6].astype(int)
    j  = df['mesh_code_str'].str[6:7].astype(int)
    i  = df['mesh_code_str'].str[7:8].astype(int)
    L  = df['mesh_code_str'].str[8:9].astype(int) # 9桁目 (1〜4)

    # 1km(第3次)メッシュの南西端（左下）の緯度経度
    lat_sw_3 = YY / 1.5 + J / (1.5 * 8) + j / (1.5 * 8 * 10)
    lon_sw_3 = 100 + XX + I / 8 + i / (8 * 10)
    
    # 500m(第4次)メッシュのオフセット計算
    # 1kmメッシュの緯度幅は 1/120度、経度幅は 1/80度。500mメッシュはその半分。
    # L=3, 4のとき緯度方向に +1/240度
    lat_offset = (L >= 3).astype(int) * (1 / 240.0)
    # L=2, 4のとき経度方向に +1/160度
    lon_offset = (L % 2 == 0).astype(int) * (1 / 160.0)
    
    # 最終的な500mメッシュの南西端
    lat_sw = lat_sw_3 + lat_offset
    lon_sw = lon_sw_3 + lon_offset

    # 北東端（右上）の緯度経度
    lat_ne = lat_sw + (1 / 240.0)
    lon_ne = lon_sw + (1 / 160.0)

    # Bounding Boxポリゴンの作成
    geometries = [box(minx, miny, maxx, maxy) for minx, miny, maxx, maxy in zip(lon_sw, lat_sw, lon_ne, lat_ne)]
    
    # EPSG:4612 (JGD2000) でGeoDataFrame化
    mesh_gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4612")

    print("座標変換と面積按分処理を実行しています（500mメッシュのため少し時間がかかります）...")
    # ゾーンデータの投影座標系に変換
    mesh_gdf = mesh_gdf.to_crs(zones.crs)
    
    # 処理軽減のため、ゾーンのバウンディングボックスに重なるメッシュのみ抽出
    minx, miny, maxx, maxy = zones.total_bounds
    mesh_gdf = mesh_gdf.cx[minx:maxx, miny:maxy]

    # 按分計算用に元の面積を計算
    mesh_gdf['original_area'] = mesh_gdf.geometry.area

    # 空間結合（Intersection）
    intersected = gpd.overlay(zones, mesh_gdf, how='intersection')
    
    # 切り取られた部分の面積
    intersected['intersect_area'] = intersected.geometry.area

    # 面積の割合（按分割合）と従業者数の割り当て（小数点のまま維持）
    intersected['ratio'] = intersected['intersect_area'] / intersected['original_area']
    intersected['apportioned_emp'] = intersected['T000842002'] * intersected['ratio']

    print("結果を集計しています...")
    # ゾーンごとに按分従業者数を合計
    emp_sum = intersected.groupby(['id', kzone_col])['apportioned_emp'].sum().reset_index()

    # 全ゾーン網羅のためLeft Join
    result = zones[['id', kzone_col]].merge(emp_sum, on=['id', kzone_col], how='left')
    result['apportioned_emp'] = result['apportioned_emp'].fillna(0)
    
    # ※今回は整数の丸め処理を行わず、小数を含んだままにします
    result.rename(columns={'apportioned_emp': '従業者数_2015'}, inplace=True)
    result = result.sort_values('id')

    # 出力
    out_csv = 'employee_2014_apportioned.csv'
    result.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"CSV出力を完了しました: {out_csv}")
    
    try:
        out_excel = 'employee_2014_apportioned.xlsx'
        result.to_excel(out_excel, index=False)
        print(f"Excel出力を完了しました: {out_excel}")
    except ImportError:
        print("openpyxlライブラリがないため、Excel出力はスキップされました。")

    print(f"東京都市圏ゾーン内の従業者数合計(2014年経済センサス‐基礎調査): {result['従業者数_2015'].sum():,.2f} 人")
    return result

if __name__ == "__main__":
    df_result = apportion_employees_2014_500m()
    print(df_result.head())