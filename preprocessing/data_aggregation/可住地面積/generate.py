import geopandas as gpd
import pandas as pd
import glob
import os

def calculate_habitable_area():
    # 1. ゾーンデータの読み込みと準備
    print("ゾーンデータを読み込んでいます...")
    zones = gpd.read_file('H30_kzone.shp')
    
    # 元の順番を保持する0番始まりのid列を追加
    zones['id'] = range(len(zones))
    
    # 計算結果に含めるため、元の列名を確認（エラー回避のため）
    kzone_col = 'kzone' if 'kzone' in zones.columns else zones.columns[0]

    # 2. メッシュデータの読み込みとフィルタリング
    print("メッシュデータを処理しています...")
    mesh_files = glob.glob('L03-b-16_*.shp')
    
    if not mesh_files:
        raise FileNotFoundError("メッシュファイル (L03-b-16_*.shp) が見つかりません。")

    habitable_codes = ['0100', '0200', '0700', '0901', '0902', '1000']
    habitable_meshes = []

    for f in mesh_files:
        # メッシュの読み込み
        mesh = gpd.read_file(f, encoding="shift-jis")
        
        # 可住地コードでフィルタリング
        habitable = mesh[mesh['土地利用種'].isin(habitable_codes)].copy()
        
        if not habitable.empty:
            # 面積計算のため、メッシュをゾーンデータと同じ投影座標系（平面直角座標系9系など）に変換
            habitable = habitable.to_crs(zones.crs)
            habitable_meshes.append(habitable)

    # 抽出したメッシュを一つのGeoDataFrameに結合
    if habitable_meshes:
        all_habitable_mesh = gpd.GeoDataFrame(pd.concat(habitable_meshes, ignore_index=True), crs=zones.crs)
    else:
        raise ValueError("指定された条件の可住地メッシュが存在しません。")

    # 3. 空間結合（Intersection）と面積計算
    print("ゾーンとメッシュの空間交差処理（オーバーレイ）を実行しています。少し時間がかかります...")
    # how='intersection'により、ゾーンの境界線でメッシュが厳密に切り取られます
    intersected = gpd.overlay(zones, all_habitable_mesh, how='intersection')

    # 切り取られたポリゴンの面積を計算（平面直角座標系のため単位は平方メートル）
    intersected['area_sqm'] = intersected.geometry.area
    
    # 平方メートルを平方キロメートルに変換
    intersected['area_km2'] = intersected['area_sqm'] / 1e6

    # 4. 集計と整形
    print("結果を集計しています...")
    # idとkzoneでグループ化し、面積を合計
    area_sum = intersected.groupby(['id', kzone_col])['area_km2'].sum().reset_index()

    # 可住地面積が0のゾーンも網羅するため、元のデータフレームと外部結合(Left Join)
    result = zones[['id', kzone_col]].merge(area_sum, on=['id', kzone_col], how='left')
    
    # 欠損値(可住地メッシュが一切重なっていなかったゾーン)を0で埋める
    result['area_km2'] = result['area_km2'].fillna(0)
    
    # 出力要件に合わせてカラム名を変更
    result.rename(columns={'area_km2': '可住地面積(km2)'}, inplace=True)
    
    # id順にソート（念のため）
    result = result.sort_values('id')

    # 5. 出力
    # CSV形式で出力 (BOM付きUTF-8でExcelでも文字化けせずに開けます)
    csv_filename = 'habitable_area_result.csv'
    result.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"処理完了: {csv_filename} に出力しました。")

    # Excel形式で出力する場合 (openpyxlがインストールされている必要があります)
    try:
        excel_filename = 'habitable_area_result.xlsx'
        result.to_excel(excel_filename, index=False)
        print(f"処理完了: {excel_filename} に出力しました。")
    except ImportError:
        print("openpyxlライブラリがないため、Excelファイルの出力はスキップされました。(CSVは出力済)")

    return result

# 実行
if __name__ == "__main__":
    df_result = calculate_habitable_area()
    print(df_result.head())