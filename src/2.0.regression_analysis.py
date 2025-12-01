# regression_analysis.py は、電力潮流計算の結果を用いて線形回帰分析を行い、結果を3Dプロットするスクリプトです。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

# 図の初期化
fig = plt.figure(figsize=(15, 5))

# 回帰係数と切片を保存するDataFrame
PV_cul = [
    'ap(0)', 'aq(0)', 'b(0)',
    'ap(1)', 'aq(1)', 'b(1)',
    'ap(2)', 'aq(2)', 'b(2)',
]
nNum = 1
tate = np.arange(1, nNum + 1)
cut = pd.DataFrame(data=0, index=tate, columns=PV_cul)

# データ読み込み
df = pd.read_csv("../data/output/data_random_walk_1day.csv", encoding="shift_jis")

# 回帰・プロット
for k in range(3):
    # 特徴量Xと目的変数yの準備
    X = pd.DataFrame(0, index=df.index, columns=['p', 'q'])
    X['p'] = df[f'p_from_mw({k})'].astype(float) * 1000
    X['q'] = df[f'q_from_mvar({k})'].astype(float) * 1000
    y = df[f'vm_delta_pu({k})'].astype(float) * 100

    # 線形回帰モデルの学習
    model = LinearRegression()
    model.fit(X, y)

    # 回帰係数と切片の保存
    print(f"{k}回目の係数:", model.coef_, "切片:", model.intercept_)
    cut.iat[0, k*3]   = model.coef_[0]
    cut.iat[0, k*3+1] = model.coef_[1]
    cut.iat[0, k*3+2] = model.intercept_

    # 3Dプロット
    ax = fig.add_subplot(1, 3, k+1, projection='3d')
    ax.scatter(X['p'], X['q'], y, c='darkblue', s=20, label='データ点')

    # 回帰面の描画
    x_range = np.linspace(X['p'].min(), X['p'].max(), 20)
    y_range = np.linspace(X['q'].min(), X['q'].max(), 20)
    x_surf, y_surf = np.meshgrid(x_range, y_range)
    z_surf = model.coef_[0] * x_surf + model.coef_[1] * y_surf + model.intercept_
    ax.plot_surface(
    x_surf, y_surf, z_surf,
    alpha=0.5,
    color=(0.5, 1.0, 0.5),        # ← 線（面）の色を緑に指定
    edgecolor='none'
                    )

    # ラベルとタイトル
    ax.set_xlabel("p (kW)")
    ax.set_ylabel("q (kvar)")
    ax.set_zlabel("vm_delta (pu.)")
    ax.set_title(f"Step {k} 回帰散布図")

# グラフのレイアウト調整・保存
plt.tight_layout()
plt.savefig("./output/regression_all_steps.png", dpi=300, bbox_inches='tight')

# 回帰係数・切片を保存
cut.to_csv("./output/回帰係数・切片random_walk2_robust.csv", encoding="shift_jis", index=False)