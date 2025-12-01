# Generalized Benders Decomposition (GBD) 実装

## 概要

配電網の分散型最適化のためのGeneralized Benders Decomposition (GBD) アルゴリズムの実装です。

### 問題構造

```
全体問題:
  min Σ f_{H,h}(x_H) + f_agg(y_DN)
  s.t. 需要家制約, 配電網制約, 境界等式制約

分解後:
  マスター問題（アグリゲータ）: 境界変数 y_DN を決定
  サブ問題（各需要家h）: 境界変数を固定して電力コスト最小化
```

## クラス構造

### 1. `SystemParameters`
システムパラメータとデータの管理
- 電圧制約、蓄電池・ヒートポンプパラメータ
- CSVからのデータ読み込み

### 2. `Subproblem`
需要家hのサブ問題
- 境界変数 `y_DN_hat` を固定として受け取り
- 電力コスト最小化
- 最適カット用の双対変数を返す

### 3. `FeasibilitySubproblem`
実行可能性サブ問題
- サブ問題が実行不可能な場合に使用
- バイナリ変数を緩和
- 実行可能カット用の双対変数を返す

### 4. `MasterProblem`
アグリゲータのマスター問題
- 配電網制約（電圧、送電線容量）
- カット平面制約の管理
- 境界変数 `y_DN` と補助変数 `LBD` の決定

### 5. `BendersSolver`
GBDアルゴリズムのメインソルバー
- 反復ループの制御
- 収束判定
- 結果のエクスポート

## アルゴリズムフロー

```
Step 1: 初期化
  - LB = -∞, UB = +∞
  - 境界変数 y_DN_hat を初期化（電力需要に基づく推定値）
  - マスター問題モデルを構築

Step 2: 各需要家がサブ問題を解く
  For h = 0 to H-1:
    サブ問題(h, y_DN_hat[h]) を解く
    
    If 実行可能:
      - 目的関数値 UB_h を記録
      - 境界等式制約の双対変数 μ を取得
      - 最適カットを生成: LBD_h >= L* - μᵀy_DN
    
    If 実行不可能:
      - 実行可能性サブ問題を解く
      - 緩和制約の双対変数 λ を取得
      - 実行可能カットを生成: L* - λᵀy_DN <= 0

Step 3: 上界の更新（全需要家が実行可能な場合）
  UB = Σ UB_h + f_agg(y_DN_hat)

Step 4: マスター問題を解く
  - カット平面制約を追加
  - 新しい境界変数 y_DN_hat を取得
  - 下界を更新: LB = マスター問題の目的関数値

Step 5: 収束判定
  If |UB - LB| < tolerance:
    収束 → 終了
  Else:
    Step 2に戻る
```

## 使用方法

```python
from gbd_optimizer import SystemParameters, BendersSolver

# パラメータ読み込み
params = SystemParameters(data_dir="../data")

# GBDソルバーの作成
solver = BendersSolver(
    params=params,
    max_iterations=100,
    tolerance=1e-3
)

# 最適化実行
results = solver.solve()

# 結果のエクスポート
solver.export_results(results, output_dir="../data/output/gbd")

# 収束プロット
solver.plot_convergence(output_dir="../data/output/gbd")
```

## 必要なデータファイル

```
data/
├── input/
│   ├── electric_demand_1y_2004/electric_demand_1y_30min_01.csv
│   ├── buy_energy30.csv
│   ├── pv_output.csv
│   ├── heat_demand1y.csv
│   ├── temperature1y.csv
│   └── water_temperature_30min.csv
└── output/
    └── random_test/2.1.回帰係数・切片.csv
```

## 出力ファイル

```
data/output/gbd/
├── gbd_summary.csv           # サマリー（目的関数値、計算時間等）
├── gbd_convergence.csv       # 収束履歴
├── gbd_convergence.png       # 収束プロット
└── gbd_household_{h}.csv     # 各需要家の詳細解
```

## 主な変数の意味

### 境界変数 (y_DN)
| 変数 | 説明 |
|------|------|
| V_DN | 配電網側の電圧 [V] |
| P_DN | 配電網側の有効電力 [kW] |
| Q_DN | 配電網側の無効電力 [kVAR] |

### 需要家変数 (x_H)
| 変数 | 説明 |
|------|------|
| P_buy | 買電電力 [kW] |
| P_sell | 売電電力 [kW] |
| P_ch | 蓄電池充電電力 [kW] |
| P_dch | 蓄電池放電電力 [kW] |
| E_bat | 蓄電池残量 [kWh] |
| P_hp | ヒートポンプ消費電力 [kW] |
| H_tank | 貯湯槽残量 [MJ] |

## 全体最適化との比較

全体最適化（`centralized_optimizer.py`）と比較するには：

1. 両方を同じデータで実行
2. 目的関数値と計算時間を比較
3. 境界変数の一致を確認

GBDの利点:
- プライバシー保護（需要家情報の秘匿）
- 分散計算が可能
- 大規模問題へのスケーラビリティ

GBDの課題:
- 収束に反復が必要
- 計算時間が問題サイズに依存
- 初期値の選択が重要

## トラブルシューティング

### サブ問題が常に実行不可能
- 境界変数の初期値が不適切な可能性
- 電圧制約または電力制約が厳しすぎる

### マスター問題がunbounded
- LBDの下限が設定されていない
- カット平面が不足

### 収束が遅い
- toleranceを調整
- 初期値の改善
- 反復上限の増加

### 双対変数が取得できない
- `QCPDual=1` パラメータを確認
- 二次制約がある場合は必須
