# 計画基本ゾーン別の平均地価

| ファイル | 内容 |
|---|---|
| `land_price_aggregate.py` | ゾーン別平均地価の集計（論文 4. (2) c)）．処理の内容はスクリプト冒頭のコメントを参照 |
| `land_price_change_1999_2016.py` | TX 沿線の住宅地の地価公示価格の変化（1999年→2016年，論文 5. (4)） |

元データ：国土数値情報 地価公示（L01：1996年，2016年），都道府県地価調査（L02：1995年，1996年，2016年）．

```
python land_price_aggregate.py 2016 --dir 2015 --out_dir output
python land_price_aggregate.py 1996 --dir 1995 --prev_dir 1995/L02-95 --out_dir output
python land_price_change_1999_2016.py
```
