import logging
import os
import gurobipy as gp
import pandas as pd

# ロギングの設定
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

line = 4 # 電力系統の分岐数
T = 48 # タイムスロット数（24時間 / 30粒度 = 48スロット）
H = 3 # 需要家の数

# 下げDR期間の定義(13時～16時を30分粒度で表記)
DOWN_DR_START_HOUR = 13*2 # 13時
DOWN_DR_END_HOUR = 16*2 # 16時

# 変換効率
ETA_PV_PCS = 0.95
ETA_B_PCS = 0.95

GRANULARITY_CONVERSION = 30 / 60 # 30分粒度を1時間粒度に変換するための係数
N_B_PCS = 1.5 # 蓄電池kW容量[kW]
N_B = 6.3 # 蓄電池kWh容量[kWh]
SELL_PRICE = 2.0 # 売電単価[JPY/kW]
BUY_MAX = 5.0 # 買電上限[kW]
SELL_MAX = 5.0 # 売電上限[kW]
IN_MAX = BUY_MAX * H # 需要家への供給電力上限[kW]
OUT_MAX = SELL_MAX * H # 需要家からの放出電力上限[kW]

r1 = 0.05 # 最低熱製造率
r2 = 0.1  # HP給湯器起動時ロス率
cw = 0.0042  # 水の比熱[MJ/L]
ce = 3.6  # 変換係数
vlt = 370  # 貯水槽の容量[L]
cph = 16.2  # HP給湯器加熱能力[MJ]
xhp = 0.013  # HP給湯器補器消費電力[kWh]
tank_half_capacity = 0.5 * cw * vlt * 70

# 電圧上下限値の設定
V_UL=101.0+6.0
V_LL=101.0-6.0

#----------データの読み込み----------
for i in range(1,H+1):
    dmd_data = pd.read_csv(f"../data/input/electric_demand_1y_2004/electric_demand_1y_30min_0{i}.csv", encoding="shift_jis") #家庭の電力需要
buy_price_data= pd.read_csv("../data/input/buy_energy30.csv", encoding="shift_jis") #電気料金[JPY/kW]
spot_price_data = pd.read_csv("../data/input/spot_market_price_30min.csv", encoding="shift_jis") #スポット市場の電力価格
pv_data = pd.read_csv("../data/input/pv_output.csv") # 太陽光出力4.8kW想定
heat_dmd_data = pd.read_csv("../data/input/heat_demand1y.csv", encoding="shift_jis") # 熱需要の読み込み
temperature_outside_data = pd.read_csv("../data/input/temperature1y.csv", encoding="shift_jis") # 外気温度の読み込み
temperature_water_data = pd.read_csv("../data/input/water_temperature_30min.csv", encoding="shift_jis") # 給水温度
voltage_estimate = pd.read_csv("../data/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis") # 電圧推定に必要なパラメータ
time = {t: dmd_data.iat[t-1, 0] for t in range(1, T+1)} # 時刻データの読み込み

# 太陽光出力[kW・30min]
p_pv = {}
for t in range(1, T+1):
    pv_value = float(pv_data.iat[t + 30*T, 2]) / 1000
    for i in range(H):
        p_pv[t, i] = pv_value
    logger.debug(f"p_pv[{t},1]: {p_pv[t, 1]}")

# 家庭の需要
p_dmd = {}
for t in range(T+1):
    for i in range(H):
        p_dmd[t, i] = float(dmd_data.iat[t-1, 10])
        logger.debug(f"p_dmd[{t}, {i}]: {p_dmd[t, i]}")

# 電気料金[JPY/kW]
buy_price = {}
for t in range(1, T+1):
    buy_price[t] = float(buy_price_data.iat[t-1, 2])

# スポット市場単価
spot_price = {}
for t in range(1, T+1):
    spot_price[t] = float(spot_price_data.iat[t, 2])
logger.debug(f"spot_price[{t}]: {spot_price[t]}")

# 熱需要の切り抜き
H_dmd = {}
for t in range(1, T+1):
    for i in range(H):
        H_dmd[t, i] = float(heat_dmd_data.iat[t-1+T, 2])

tma = {} # 外気温度
k = {} # 補正係数
cop = {} # 成績係数
tmf = {} # 給水温度
for t in range(1, T+1):
    tmf[t] = float(temperature_water_data.iat[int(t-1), 2])
for t in range(1, T+1):
        tma[t] = float(temperature_outside_data.iat[int(t), 2])

        if tma[t] >= 5:
            k[t] = 1
        elif tma[t] > 2:
            k[t] = tma[t]/30 + 0.8333
        else:
            k[t] = 0.9
        cop[t] = k[t]*(0.175*tma[t]-0.1322*tmf[t]+4.076) # 成績係数の決定

# 電圧推定に必要なパラメタ
linear_ap = {}
linear_aq = {}
linear_b = {}
for i in range(H):
    linear_ap[i] = voltage_estimate.iat[0, 3*i]
    linear_aq[i] = voltage_estimate.iat[0, 3*i+1]
    linear_b[i] = voltage_estimate.iat[0, 3*i+2]
    logger.debug(f"linear_ap[{i}]: {linear_ap[i]}")
    logger.debug(f"linear_aq[{i}]: {linear_aq[i]}")
    logger.debug(f"linear_b[{i}]: {linear_b[i]}")

AG_PM = gp.Model("aggregator_benefit")

#----------決定変数の定義----------
# 全需要家のコスト[JPY]
cost = AG_PM.addVars(range(1, T+1), vtype=gp.GRB.CONTINUOUS, name='cost', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)

# アグリゲータの系統への売電電力[kW]
P_SG = AG_PM.addVars(range(1,T+1), vtype=gp.GRB.CONTINUOUS, name='P_SG', lb=0.0, ub=gp.GRB.INFINITY)
# アグリゲータの系統への売電フラグ
Delta_sg = AG_PM.addVars(range(T+1), vtype=gp.GRB.BINARY, name='Delta_sg', lb=0.0, ub=1.0)
# アグリゲータの系統からの買電電力[kW]
P_PG = AG_PM.addVars(range(1,T+1), vtype=gp.GRB.CONTINUOUS, name='P_PG', lb=0.0, ub=gp.GRB.INFINITY)
# アグリゲータの系統からの買電フラグ
Delta_pg = AG_PM.addVars(range(T+1), vtype=gp.GRB.BINARY, name='Delta_pg', lb=0.0, ub=1.0)
# 一軒に供給される電力量[kWh]
P_in = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_in', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
# P_inのフラグ
Delta_in = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_in', lb=0.0, ub=1.0)
# 一軒から放出される電力量[kWh]
P_out = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_out', lb=0.0, ub=gp.GRB.INFINITY)
# P_outのフラグ
Delta_out = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_out', lb=0.0, ub=1.0)
# 一軒の買電電力量[kWh]
P_pg = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_pg', lb=0.0, ub=gp.GRB.INFINITY)
# 一軒の売電電力量[kWh]
P_sg = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_sg', lb=0.0, ub=gp.GRB.INFINITY)
# 需要家間の電力融通量[kWh]
P_fp = AG_PM.addVars(range(H), range(H), range(1, T+1), vtype=gp.GRB.CONTINUOUS, name='P_fp', lb=0.0, ub=gp.GRB.INFINITY)

# 蓄電池充電電力[kW]
P_Ch = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_Ch', lb=0.0, ub=gp.GRB.INFINITY)
# 蓄電池放電電力[kW]
P_DCh = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_DCh', lb=0.0, ub=gp.GRB.INFINITY)
# 蓄電池蓄電電力量[kWh]
E_B = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='E_B', lb=0.0, ub=gp.GRB.INFINITY)
# 蓄電池の充電フラグ（充電中１、それ以外０）
Delta_ch = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_ch', lb=0.0, ub=1.0)
# 蓄電池の放電フラグ（放電中１、それ以外０）
Delta_dch = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_dch', lb=0.0, ub=1.0)

# HP消費電力[kW]
P_HP = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='P_HP', lb=0.0, ub=gp.GRB.INFINITY)
# HP運転開始時エネルギーロス[MJ]
H_ini = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='H_ini', lb=0.0, ub=gp.GRB.INFINITY)
# 熱製造量[MJ]
H_prod = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='produce_heat', lb=0.0, ub=gp.GRB.INFINITY)
# 貯湯量[MJ]
H_tank = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='tank_heat', lb=0.0, ub=gp.GRB.INFINITY)
# HP給湯機運転指標（運転中 1，停止中 0）
Delta_prod = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_prod', lb=0.0, ub=1.0)
#HP給湯機運転開始変数
Delta_S = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_S', lb=0.0, ub=1.0)
# HP給湯機運転終了変数
Delta_F = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_F', lb=0.0, ub=1.0)

# 電圧
V_d = AG_PM.addVars(range(1, T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='電圧', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
# 電圧変化
V_dd = AG_PM.addVars(range(T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='電圧変化', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
# 有効電力[kW]
ap = AG_PM.addVars(range(1, T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='有効電力', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
# 無効電力[kVar]
aq = AG_PM.addVars(range(1, T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='無効電力', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
# 上限セーフ値
V_UN = AG_PM.addVars(range(1, T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='上限セーフ値', lb=0.0, ub=gp.GRB.INFINITY)
# 下限セーフ値
V_LN = AG_PM.addVars(range(1, T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='下限セーフ値', lb=0.0, ub=gp.GRB.INFINITY)
# 上限違反値
V_ULV = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='上限違反値', lb=0.0, ub=gp.GRB.INFINITY)
# 下限違反値
V_LLV = AG_PM.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='下限違反値', lb=0.0, ub=gp.GRB.INFINITY)

AG_PM.update()

#----------目的関数----------
# 電力料金コスト + アグリゲータの下げDR時における買電電力量
AG_PM.setObjective(
    # 電力料金コスト（買電コスト - 売電収入）
    gp.quicksum(buy_price[t] * P_PG[t] - SELL_PRICE * P_SG[t]
                for t in range(1, T+1)) +
    # アグリゲータの下げDR時における買電電力量
    gp.quicksum(P_PG[t]
                for t in range(DOWN_DR_START_HOUR+1, DOWN_DR_END_HOUR+1)),
    gp.GRB.MINIMIZE
)

#----------制約条件----------
# 電力コスト
for t in range(1, T+1):
    AG_PM.addConstr(cost[t] == buy_price[t] * P_PG[t] - SELL_PRICE * P_SG[t])

# 需給バランス制約
for t in range(1, T+1):
    AG_PM.addConstr(P_SG[t] == gp.quicksum(P_sg[t, i] for i in range(H)))
    AG_PM.addConstr(P_PG[t] == gp.quicksum(P_pg[t, i] for i in range(H)))
    # アグリゲータレベルの需給バランス
    AG_PM.addConstr(
        P_SG[t] + gp.quicksum(P_in[t, i] for i in range(H)) == 
        P_PG[t] + gp.quicksum(P_out[t, i] for i in range(H))
    )
    # 各需要家の需給バランス
    for i in range(H):
        AG_PM.addConstr(
            p_pv[t, i] + P_in[t, i] + P_DCh[t, i] ==
            p_dmd[t, i] + P_Ch[t, i] + P_out[t, i] + P_HP[t, i]
        )
        # 需要家への供給電力定義
        AG_PM.addConstr(P_in[t, i] == P_pg[t, i] + gp.quicksum(P_fp[j, i, t] for j in range(H) if j != i))
        AG_PM.addConstr(P_out[t, i] == P_sg[t, i] + gp.quicksum(P_fp[i, j, t] for j in range(H) if j != i))

# 買電売電同時禁止制約
for t in range(1, T+1):
    # アグリゲータレベルの買電売電同時禁止制約
    AG_PM.addConstr(P_PG[t] <= Delta_pg[t] * BUY_MAX)
    AG_PM.addConstr(P_SG[t] <= Delta_sg[t] * SELL_MAX)
    AG_PM.addConstr(Delta_pg[t] + Delta_sg[t] <= 1)
    # 需要家レベルの買電売電同時禁止制約
    for i in range(H):
        AG_PM.addConstr(P_pg[t, i] <= Delta_in[t, i] * IN_MAX)
        AG_PM.addConstr(P_sg[t, i] <= Delta_out[t, i] * OUT_MAX)
        AG_PM.addConstr(Delta_in[t, i] + Delta_out[t, i] <= 1)

for i in range(H):
    #HP制約
    for t in range(1, T+1):
        # HP給湯機の消費電力[kWh]と生成する熱量[MJ]に関する等式制約
        AG_PM.addConstr(P_HP[t,i] == xhp * Delta_prod[t,i] + (H_prod[t,i] + H_ini[t,i]) / ce / cop[t])
        # 生成する熱量の上下限制約
        AG_PM.addConstr(H_prod[t,i] >= GRANULARITY_CONVERSION * r1 * cph * Delta_prod[t,i])
        AG_PM.addConstr(H_prod[t,i] <= GRANULARITY_CONVERSION * cph * Delta_prod[t,i])
        # 運転開始時のエネルギーロス
        AG_PM.addConstr(H_ini[t,i] == r2 * cph * Delta_S[t,i])
        #運転状態の更新制約
        AG_PM.addConstr(Delta_prod[t,i] - Delta_prod[t-1,i] == Delta_S[t,i] - Delta_F[t,i])
        # 起動と停止の同時禁止制約 
        AG_PM.addConstr(Delta_S[t,i] + Delta_F[t,i] <= 1)
        # 運転中
        AG_PM.addConstr(Delta_S[t,i] <= Delta_prod[t,i])
        # ON/OFFを繰り返さない(運転開始後定格運転を行う制約)
        AG_PM.addConstr(H_prod[t,i] >= cph * (Delta_prod[t,i] - Delta_S[t,i]))
        # 貯湯槽内の蓄熱量更新に関する制約
        AG_PM.addConstr(H_tank[t,i] == H_tank[t-1,i] + H_prod[t,i] - H_dmd[t,i])
        # 貯湯槽の容量制約（沸き上げ温度65度）
        AG_PM.addConstr(H_tank[t,i] <= cw * vlt * 70)
        AG_PM.addConstr(H_tank[t,i] >= 0.2 *(cw * vlt * 70))

    # HP給湯機の初期状態設定
    AG_PM.addConstr(Delta_prod[0,i] == 0)
    AG_PM.addConstr(Delta_S[0,i] == 0)
    AG_PM.addConstr(Delta_F[0,i] == 0)

    # 貯湯槽蓄熱量始端・終端制約（満タン容量の50%で開始・終了）
    AG_PM.addConstr(H_tank[0, i] == tank_half_capacity)
    AG_PM.addConstr(H_tank[T, i] == tank_half_capacity)

    # 蓄電池制約
    for t in range(1,T+1):
        #蓄電池充放電容量（kW容量）制約
        AG_PM.addConstr(P_Ch[t,i] <= N_B_PCS * Delta_ch[t,i])
        AG_PM.addConstr(P_DCh[t,i] <= N_B_PCS * Delta_dch[t,i])
        AG_PM.addConstr(Delta_ch[t,i] + Delta_dch[t,i] <= 1.0)
        # 蓄電池蓄電容量（kWh容量）制約
        AG_PM.addConstr(E_B[t,i] >= N_B * 0.2)
        AG_PM.addConstr(E_B[t,i] <= N_B)
        # 蓄電池蓄電量更新制約
        AG_PM.addConstr(E_B[t,i] == E_B[t-1,i] + 
                       GRANULARITY_CONVERSION * ETA_B_PCS * P_Ch[t,i] - 
                       GRANULARITY_CONVERSION * (1/ETA_B_PCS) * P_DCh[t,i])
        AG_PM.addConstr(P_DCh[t,i] <= Delta_dch[t,i]*N_B_PCS)

    # 蓄電池蓄電量始端・終端制約（50%で開始・終了）
    AG_PM.addConstr(E_B[0, i] == 0.5 * N_B)
    AG_PM.addConstr(E_B[T, i] == 0.5 * N_B)
    # 蓄電池初期状態設定（充放電なし）
    AG_PM.addConstr(Delta_ch[0, i] == 0)
    AG_PM.addConstr(Delta_dch[0, i] == 0)

# 電圧制約
for t in range(1, T+1):
    for k in range(H):
        # 上限・下限違反制約
        AG_PM.addConstr(V_UL - V_d[t,k] == V_UN[t,k] - V_ULV[t,k])
        AG_PM.addConstr(V_d[t,k] - V_LL == V_LN[t,k] - V_LLV[t,k])
        
# 電圧推定
for t in range(1, T+1):
    for k in range(H):
        # 電圧降下推定
        AG_PM.addConstr(V_dd[t, k] == linear_ap[k] * ap[t, k] + linear_aq[k] * aq[t, k] + linear_b[k])

# 各バスの電圧計算
for t in range(1, T+1):
    for k in range(H):
        if k == 0:
            # バス1の電圧（基準電圧 + 電圧降下）
            AG_PM.addConstr(V_d[t, k] == V_dd[t, k] + 99.8)
        else:
            # バス2以降の電圧（前段バス電圧 + 電圧降下）
            AG_PM.addConstr(V_d[t, k] == V_dd[t, k] + V_d[t, k-1])

# 有効電力・無効電力の制約
for t in range(1, T+1):
    for k in range(H):
        # 有効電力の制約
        AG_PM.addConstr(ap[t, k] == gp.quicksum(P_in[t, i] for i in range(k, H)))
        # 無効電力の制約（有効電力の10%と仮定）
        AG_PM.addConstr(aq[t, k] == gp.quicksum(P_in[t, i] for i in range(k, H)) * 0.1)

AG_PM.update()

# 最適化の実行
try:
    AG_PM.optimize()

    status = AG_PM.Status

    if status == gp.GRB.OPTIMAL:
        logger.info("最適解が見つかりました")
        logger.info(f"目的関数値: {AG_PM.ObjVal}")
        logger.info(f"計算時間: {AG_PM.Runtime:.2f} 秒")
    elif status == gp.GRB.INFEASIBLE:
        logger.error("問題が実行不可能です")
    elif status == gp.GRB.UNBOUNDED:
        logger.error("問題が非有界です")
    else:
        logger.warning(f"最適化が完了しませんでした。ステータス: {status}")
        
except Exception as e:
    logger.error(f"最適化中にエラーが発生しました: {e}")
    raise

if AG_PM.status == gp.GRB.OPTIMAL:
    # 下げDR期間における買電電力量の計算
    down_dr_power = sum(P_PG[t].X 
                       for t in range(DOWN_DR_START_HOUR+1, DOWN_DR_END_HOUR+1))

    # 電力料金コスト（買電コスト - 売電収入）の計算
    electricity_cost = sum(buy_price[t] * P_PG[t].X - SELL_PRICE * P_SG[t].X
                          for t in range(1, T+1))

    logger.info(f"下げDR期間の買電電力量: {down_dr_power}")
    logger.info(f"電力料金コスト: {electricity_cost}")
    
# 結果出力用データフレームの作成
benefit_df = pd.DataFrame()

# 各需要家の結果を出力
for i in range(H):
    # 時系列データの構築
    benefit_df = pd.DataFrame({
        'コマ': [time[t] for t in range(1, T+1)],
        '電力コスト[JPY]': [cost[t].x for t in range(1, T+1)],
        'アグリゲータの買電電力[kW・30分]': [P_PG[t].x for t in range(1, T+1)],
        'アグリゲータの売電電力[kW・30分]': [-P_SG[t].x for t in range(1, T+1)],
        '買電電力[kW・30分]': [P_pg[t, i].x for t in range(1, T+1)],
        '売電電力[kW・30分]': [-P_sg[t, i].x for t in range(1, T+1)],
        '一軒に供給される電力量[kW・30分]': [P_in[t, i].x for t in range(1, T+1)],
        '一軒から放出される電力量[kW・30分]': [-P_out[t, i].x for t in range(1, T+1)],
        '電気料金[JPY・30min/kW]': [buy_price[t] for t in range(1, T+1)],
        '蓄電量[kWh]': [E_B[t, i].x for t in range(1, T+1)],
        '充電量[kW・30分]': [-P_Ch[t, i].X for t in range(1, T+1)],
        '放電量[kW・30分]': [P_DCh[t, i].X for t in range(1, T+1)],
        'HP給湯機消費電力[kW・30分]': [P_HP[t, i].X for t in range(1, T+1)],
        '充電中変数': [Delta_ch[t, i].x for t in range(1, T+1)],
        '放電中変数': [Delta_dch[t, i].X for t in range(1, T+1)],
        '売買変数': [Delta_in[t, i].x for t in range(1, T+1)],
        '電力需要量[kW・30min]': [p_dmd[t, i] for t in range(1, T+1)],
        '太陽光発電出力[kW・30min]': [p_pv[t, i] for t in range(1, T+1)]
    })
    # CSVファイルとして保存
    benefit_df.to_csv(f"../data/output/optimization/0.house_{i}.csv", encoding="shift_jis")
    
# 電圧関連の結果出力
voltage_data = {
    'Time': [time[t] for t in range(1, T+1)],
    '電圧上限値': [V_UL] * T,
    '電圧下限値': [V_LL] * T
}

# 各バスの電圧関連データを動的に追加
for i in range(H):
    voltage_data[f'bus{i+1}の電圧'] = [V_d[t, i].X for t in range(1, T+1)]
    voltage_data[f'bus{i+1}の電圧上限違反値'] = [V_ULV[t, i].X for t in range(1, T+1)]
    voltage_data[f'bus{i+1}の電圧下限違反値'] = [V_LLV[t, i].X for t in range(1, T+1)]
    voltage_data[f'bus{i+1}の有効電力'] = [ap[t, i].X for t in range(1, T+1)]
    voltage_data[f'bus{i+1}の電圧降下'] = [V_dd[t, i].X for t in range(1, T+1)]

voltage_df = pd.DataFrame(voltage_data)
voltage_df.to_csv("../data/output/optimization/1.voltage.csv", encoding="shift_jis", index=False)

# ヒートポンプ結果の出力
for i in range(H):
    heat_pump_df = pd.DataFrame({
        'HP給湯機消費電力[kW・30分]': [P_HP[t, i].X for t in range(1, T+1)],
        '熱需要[kW・30分]': [H_dmd[t, i] for t in range(1, T+1)],
        '熱製造量[MJ・30分]': [H_prod[t, i].X for t in range(1, T+1)],
        '貯湯量[MJ]': [H_tank[t, i].X for t in range(1, T+1)],
        '成績係数': [cop[t] for t in range(1, T+1)],
        '運転変数': [Delta_prod[t, i].x for t in range(1, T+1)],
        '運転開始変数': [Delta_S[t, i].x for t in range(1, T+1)],
        '運転終了変数': [Delta_F[t, i].x for t in range(1, T+1)],
        '電気料金[JPY・30min/kW]': [buy_price[t] for t in range(1, T+1)]
    })
    # CSVファイルとして保存
    heat_pump_df.to_csv(f"../data/output/optimization/2.HP_{i}.csv", encoding="shift_jis")

# 蓄電池結果の出力
for i in range(H):
    # 蓄電池状態データの作成と出力
    ess_df = pd.DataFrame({
        'Time': [time[t] for t in range(1, T+1)],
        '蓄電量[kWh]': [E_B[t, i].X for t in range(1, T+1)],
        '蓄電池容量上限[kWh]': [N_B] * T,
        '蓄電池容量下限[kWh]': [N_B * 0.2] * T
    })
    ess_df.to_csv(f"../data/output/optimization/3.ESS_{i}.csv", encoding="shift_jis", index=False)

