# ゾーン代表点（centroids.gpkg）

最短経路探索（search_passenger_separated.py）の `./input/centroids.gpkg` に置くファイル．

- 作成：`python make_centroids.py --zone ../residence/2015/H30_kzone.shp --pop_dir ../residence/2015 --emp_dir ../jobs/2014 --out centroids.gpkg`
- 方法：各ゾーンについて，ポリゴン内に中心点が含まれる500mメッシュのうち「人口＋従業者数」が最大のメッシュの中心点（初版と同じ定義）
  - 人口：2015年国勢調査 250mメッシュを500mメッシュに集計
  - 従業者：2014年経済センサス‐基礎調査 500mメッシュ（初版は2016年経済センサス‐活動調査）
- レイヤ `centroids`，EPSG:4326，615点．記録順（fid順）＝ H30_kzone.shp の記録順
- 属性：`node_id`（0〜614，H30_kzone.shp の記録順．経路探索の o_id/d_id），`kzone`，`name`，`mesh500`，`pop`，`emp`，`activity`，`method`，`inside_zone`
- 全615ゾーンで通常の方法（max_mesh）で決まり，全点がゾーン内にあることを確認済み

## 初版の方法（従業者に2016年活動調査を使用）との比較

`centroids_compare_with_2016based.csv`：99ゾーンで代表点が変わる（移動距離の中央値 924m，最大 13.7km）．
移動が大きいのは，国・地方公共団体の大規模な事業所（自衛隊基地，港湾施設等）が人口の少ないメッシュにあるゾーン．
例：kzone 5012 は航空自衛隊百里基地付近（人口0，従業者1,855人）のメッシュが代表点になる．

※ 初版の代表点ファイルそのものとの比較ではない（初版は2015年国勢調査の500mメッシュ人口を直接使用．
  本版は250mメッシュを500mに集計しており，秘匿処理の扱いの違いによるわずかな差があり得る）．
