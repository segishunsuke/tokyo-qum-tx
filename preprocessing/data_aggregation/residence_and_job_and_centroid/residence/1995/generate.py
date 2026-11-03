# -*- coding: utf-8 -*-
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import glob
import os

def apportion_population_1995_500m():
    print("ゾーンデータを読み込んでいます...")
    zones = gpd.read_file('H30_kzone.shp')
    # 順番を保持するためのid
    zones['id'] = range(len(zones))
    kzone_col = 'kzone' if 'kzone' in zones.columns else zones.columns[0]

    print("1995年の人口メッシュデータ（tblT000752H*.txt 等）を読み込んでいます...")
    # 指定の形式のテキストファイルおよびCSVを全て取得
    txt_files = glob.glob('tblT000752H*.txt') + glob.glob('tblT000752H*.csv')
    
    if not txt_files:
        raise FileNotFoundError("対象のテキスト(CSV)ファイルが見つかりません。")

    df_list = []
    for f in txt_files:
        # skiprows=[1] で2行目の日本語ヘッダーをスキップし、1列目(KEY_CODE)と2列目(T000752001:人口総数)を抽出
        try:
            df_temp = pd.read_csv(f, header=0, skiprows=[1], usecols=['KEY_CODE', 'T000752001'], encoding='shift-jis')
            df_list.append(df_temp)
        except Exception:
            # Shift-JISでダメならUTF-8で試行
            df_temp = pd.read_csv(f, header=0, skiprows=[1], usecols=['KEY_CODE', 'T000752001'], encoding='utf-8')
            df_list.append(df_temp)

    # 読み込んだ全てのデータを縦に結合
    df = pd.concat(df_list, ignore_index=True)
    
    # メッシュコードの重複があれば削除
    df = df.drop_duplicates(subset=['KEY_CODE']).copy()
    
    # 9桁の文字列に揃える
    df['mesh_code_str'] = df['KEY_CODE'].astype(str).str.zfill(9)

    print(f"結合完了: ユニークな500mメッシュ数は {len(df):,} 件です。")
    print("メッシュコードからポリゴンを生成しています...")
    
    # 各桁の抽出
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
    lat_offset = (L >= 3).astype(int) * (1 / 240.0)
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

    print("座標変換と面積按分処理を実行しています...")
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

    # 面積の割合（按分割合）と人口の割り当て（小数のまま維持）
    intersected['ratio'] = intersected['intersect_area'] / intersected['original_area']
    intersected['apportioned_pop'] = intersected['T000752001'] * intersected['ratio']

    print("結果を集計しています...")
    # ゾーンごとに按分人口を合計
    pop_sum = intersected.groupby(['id', kzone_col])['apportioned_pop'].sum().reset_index()

    # 全ゾーン網羅のためLeft Join
    result = zones[['id', kzone_col]].merge(pop_sum, on=['id', kzone_col], how='left')
    result['apportioned_pop'] = result['apportioned_pop'].fillna(0)
    
    # カラム名を変更
    result.rename(columns={'apportioned_pop': '人口_1995'}, inplace=True)
    result = result.sort_values('id')

    # 出力
    out_csv = 'population_1995_apportioned.csv'
    result.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"CSV出力を完了しました: {out_csv}")
    
    try:
        out_excel = 'population_1995_apportioned.xlsx'
        result.to_excel(out_excel, index=False)
        print(f"Excel出力を完了しました: {out_excel}")
    except ImportError:
        print("openpyxlライブラリがないため、Excel出力はスキップされました。")

    print(f"東京都市圏ゾーン内の人口合計(1995年): {result['人口_1995'].sum():,.2f} 人")
    return result

if __name__ == "__main__":
    df_result = apportion_population_1995_500m()
    print(df_result.head())