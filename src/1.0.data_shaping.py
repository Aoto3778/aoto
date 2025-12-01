# 1.data_shaping.py は、電力潮流計算の結果を整形してCSVファイルに出力するスクリプトです。
import pandas as pd
import numpy as np

# セル作成
load_result_cul = [
    'DATE',
    'p_from_mw(0)', 'p_from_mw(1)', 'p_from_mw(2)',#送電線の始点側（"from"）での有効電力 [MW]
    'q_from_mvar(0)', 'q_from_mvar(1)','q_from_mvar(2)',#送電線の始点側での無効電力 [MVar]
    'vm_delta_pu(0)','vm_delta_pu(1)','vm_delta_pu(2)',#電圧の変動量
]
nNum = 48
tate = np.arange(1, nNum + 1)
cut = pd.DataFrame(data=0, index=tate, columns=load_result_cul)
print(len(load_result_cul))


x=0
zz=0
#cut.iatに値を代入
for j in range(0, 48):
    load_result = pd.read_excel(f"../data/output/random_test/timeslot_{j}.xlsx", sheet_name='res_line')
    cut.iat[x, 0] = float(zz)  # zzをfloat型に変更
    zz = zz + 1
    # p_from_mw
    cut.iat[x, 1] = float(load_result.iat[0, 1])
    cut.iat[x, 2] = float(load_result.iat[1, 1])
    cut.iat[x, 3] = float(load_result.iat[2, 1])

    # q_from_mvar
    cut.iat[x, 4] = float(load_result.iat[0, 2])
    cut.iat[x, 5] = float(load_result.iat[1, 2])
    cut.iat[x, 6] = float(load_result.iat[2, 2])

    # vm_Δ_pu = vm_from_pu - vm_to_pu
    cut.iat[x, 7] = float(load_result.iat[0, 10] - load_result.iat[0, 12])
    cut.iat[x, 8] = float(load_result.iat[1, 10] - load_result.iat[1, 12])
    cut.iat[x, 9] = float(load_result.iat[2, 10] - load_result.iat[2, 12])
    x = x + 1
#出力
cut.to_csv("../data/output/data_random_walk_1day.csv", encoding="shift_jis", index=False)