# Quantitative Urban Model (QUM) for the Tokyo Metropolitan Area

瀬木俊輔・山口瑞貴「定量的都市モデルの現況再現性と交通インフラ評価への適用可能性の検証 －東京都市圏とつくばエクスプレスを対象として－」（土木学会論文集）で用いた，定量的都市モデル（QUM）の推定・シミュレーション用のコードと入力データです．

Code and input data for the structural estimation and counterfactual simulation of the quantitative urban model (QUM) in Segi and Yamaguchi (Journal of JSCE). The model follows Ahlfeldt et al. (2015, *Econometrica*); the agglomeration elasticity is estimated by full-information maximum likelihood, a nonlinear generalization of the ML estimator for spatial autoregressive models.

> 研究用に書かれたコードをそのまま公開しています．質問への個別の対応はできないことがあります．
>
> **1995年の従業者数は，元データの利用条件により，公開データから推計した代替データを収録しています**（[下記](#1995年の従業者数代替データ)）．そのため，`model/input/` のデータによる実行結果は論文の値と一致しません．元データを用いれば，同じコードで論文の結果を再現できます．

## フォルダ構成

| フォルダ | 内容 |
|---|---|
| `model/` | モデル本体（推定・均衡計算・便益）と集計スクリプト．`model/input/` にモデルへの入力データ一式（1995年の従業者数は代替データ） |
| `preprocessing/` | 入力データを作るための前処理のコード（参考として現状のまま公開．[`preprocessing/README.md`](preprocessing/README.md)） |

交通ネットワークのデータ（一般化費用の計算に使用）はサイズが大きいため，作成済みのデータを Zenodo で公開しています：DOI: [10.5281/zenodo.23879232](https://doi.org/10.5281/zenodo.23879232)

## 準備

Python 3.11 以降．

```
pip install numpy pandas scipy openpyxl cython matplotlib geopandas
cd model
python setup.py build_ext --inplace
```

Windows では [Microsoft C++ Build Tools](https://learn.microsoft.com/ja-jp/cpp/overview/acquire-msvc?view=msvc-170) が必要です．

## 実行方法

```
cd model
python run_cases.py          # cases.csv の全ケースを実行
python run_cases.py --list   # ケースの一覧と状態
python analyze.py            # 集計（analysis/ に表・図・数値）
```

`cases.csv` の各行が1つの分析ケースで，`note` 列に内容を記しています．各ケースは `cases/<ケース名>/` の中だけを編集し，`input/` の元データは書き換えません．実際に使ったパラメータ・推定値・標準誤差・誤差項の共分散は `cases/<ケース名>/case_info.json`，画面出力は `log.txt` に保存されます．

地図を描く場合は，東京都市圏パーソントリップ調査の計画基本ゾーン（2018年）の shapefile と，国土数値情報の駅データ（N02）を指定します．

```
python analyze.py --zones H30_kzone.shp --tx_stations N02_Station.shp
```

### 1995年の従業者数（代替データ）

論文では，1995年の従業者数に事業所・企業統計調査（1996年）の地域メッシュ統計（公益財団法人統計情報研究開発センター）を用いていますが，このデータは公開されていません．そこで `input_data.xlsx` には，以下の公開データから推計した代替データを収録しています（シート `lam_M` の1995年の列と，シート `parameters` の1995年の総従業者数 $\bar{L}$）．

- 事業所・企業統計調査（2001年）4次メッシュ（500m）統計（e-Stat）
- 事業所・企業統計調査（1996年）の市区町村別従業者数（e-Stat）
- 1995年10月1日時点の行政区域（国土数値情報 N03）

2001年のメッシュの従業者数を，市区町村ごとに1996年の従業者数に一致するように補正した上で，計画基本ゾーンに面積按分したものです．実データによる値との相関係数は0.988（対数では0.987）です．作成方法の詳細と，元データを入手した場合の置き換え方は [`preprocessing/README.md`](preprocessing/README.md) を参照してください．

### 論文との対応

| 論文 | スクリプト・ケース |
|---|---|
| 4. (9) a), b) 構造推定（表-4） | `run_cases.py`（基準ケース `base`），`estimation.pyx` |
| 4. (9) c) $\bar A$ の同定 | `calibrate_Abar.py`（`run_cases.py` の `abar` ステップ） |
| 4. (9) c) $\bar c$ の同定 | `estimation.pyx`（$\epsilon^G$ の逆算の中で実質的に決まる．論文の $\bar c$ は $\epsilon^G$ を $\epsilon^G_S = 0$ と基準化したときの定数に当たる） |
| 4. (10) a)〜c) 適合度（図・表） | 構造残差ゼロのケース，`simulation.pyx`，`analyze.py` |
| 4. (10) d) 感度分析 | `cases.csv` の指数関数型・自ゾーン密度型のケース，`compare_decay.py` |
| 5. (1) a) TX 駅別乗降客数 | `preprocessing/network/07b_最短経路探索/tx_station_riders.py` |
| 5. (2) 構造残差の時点間比較 | `analyze.py` |
| 5. (3) TX の効果の地図 | `simulation2.pyx`，`analyze.py`，`check_tx_effects.py`（市区町村別の集計） |
| 5. (4) 地価公示価格の変化 | `preprocessing/data_aggregation/land_price/land_price_change_1999_2016.py` |
| 5. (5) 便益の分解・費用便益比 | `simulation2.pyx`，`analyze.py`，`webtag_wi.py`（WebTAG の static clustering の式） |

## モデルのファイル（`model/`）

| ファイル | 内容 |
|---|---|
| `estimation.pyx` | 観測値を再現する構造残差の逆算と，集積の弾力性・減衰パラメータの最尤推定（標準誤差はヘッセ行列から） |
| `simulation.pyx` | 与えられた構造残差・交通費用の下で均衡を解く（構造残差のファイルがなければゼロとして解く） |
| `simulation2.pyx` | 便益の計算．TX なし（`CF_`），TX あり（`Real_`），TX ありで立地を TX なしの均衡に固定，の3つの均衡を解き，地主の所得・家計の効用（ログサム）・便益を出力 |
| `liberrortermsre.py` | 構造残差（2時点・3種類）のランダム効果モデルの対数尤度 |
| `myoptimize.py` | 非線形方程式・最適化の数値計算ルーチン |
| `run_cases.py`, `cases.csv` | 分析ケースの一括実行 |
| `calibrate_Abar.py` | 平均賃金を再現する $\bar A$ の設定 |
| `analyze.py` ほか | 結果の集計 |

### 入力（`model/input/`）

| ファイル | 内容 |
|---|---|
| `input_data.xlsx` | シート `parameters`（パラメータ），`area`（可住地面積 [km²]），`lam_R`・`lam_M`（各時点の居住者数・従業者数．1995年の従業者数は代替データ），`p_G`（土地賃料）．ゾーンの並びは全ファイル共通（2018年東京都市圏パーソントリップ調査の計画基本ゾーン，615ゾーン） |
| `tau[t].csv` | 時点 $t$（0 = 1995年，1 = 2015年）の往復の合成一般化費用 [円/日]（列：`i, j, cost`） |
| `CF_tau[t].csv`, `Real_tau[t].csv` | 便益計算用の TX なし／あり（2015年）の合成一般化費用 |
| `tau_ii.csv` | ゾーン内の費用 $\tau_{ii}$（隣接ゾーンへの費用の平均の1/2．全時点・全シナリオ共通） |

### 定式化の切り替え（環境変数．`run_cases.py` では `cases.csv` の列で指定）

| 環境変数 | 内容 |
|---|---|
| `QUM_TAU_TRANSFORM` | 通勤不効用の関数形．未指定で $\exp(-\kappa\tau)$（基準），`log:<τ0>` でべき乗型 |
| `QUM_DECAY_TRANSFORM` | 集積の減衰の関数形．`log:1.0` で $\tau^{-\delta}$（基準），`exp` で $\exp(-\delta\tau)$ |
| `QUM_ACCESSIBILITY` | `mass`（$\Omega_i = \sum_s L_s f(\tau_{is})$，基準）または `own_density`（$\Omega_i = L_i/\bar{G}_i$） |
| `QUM_TAU_II_FILE` | ゾーン内の費用のファイル |
| `QUM_INPUT_DIR`, `QUM_OUTPUT_DIR` | 入出力フォルダ（既定は `input`, `output`） |

単独で実行する場合は `python -c "import estimation"`（`simulation`, `simulation2` も同様）．`estimation` の出力 `output/epsilon_*_calibrated.csv` を `input/epsilon_*.csv` にコピーすると，`simulation` は観測値を再現する均衡を解きます．**構造残差のファイルがない場合はゼロとして扱われ，エラーにならない**ので注意してください．

## データの出典とライセンス

- コード：MIT License（[`LICENSE`](LICENSE)）
- `model/input/` のデータ：以下のデータを加工して作成したもの．CC BY 4.0．利用の際は本論文と元データの出典を記載してください．
  - 居住者数：国勢調査（1995年，2015年）地域メッシュ統計（総務省統計局，e-Stat）
  - 従業者数：2015年は経済センサス‐基礎調査（2014年）地域メッシュ統計．1995年は代替データ（事業所・企業統計調査（2001年）地域メッシュ統計，同（1996年）市区町村別従業者数，国土数値情報 行政区域（1995年）から推計）
  - 土地賃料：地価公示，都道府県地価調査（国土数値情報）
  - 平均賃金：賃金構造基本統計調査（2015年）
  - 可住地面積：国土数値情報 土地利用細分メッシュ
  - 通勤 OD・モード分担：東京都市圏パーソントリップ調査（東京都市圏交通計画協議会）の公開集計値
  - 一般化費用：Zenodo で公開している交通ネットワークデータ（国土数値情報，OpenStreetMap，道路交通センサス等から作成）による最短経路探索
- 交通ネットワークデータ（Zenodo）：OpenStreetMap に由来する道路ネットワークは ODbL（© OpenStreetMap contributors），それ以外は CC BY 4.0．

## 引用

瀬木俊輔，山口瑞貴：定量的都市モデルの現況再現性と交通インフラ評価への適用可能性の検証 －東京都市圏とつくばエクスプレスを対象として－，土木学会論文集，（巻号・頁），2027．
