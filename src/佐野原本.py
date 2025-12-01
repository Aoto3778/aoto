## 世帯別にエネルギー機器運転計画を最適化
import gurobipy as gp
import pandas as pd
import numpy as np




# 計算結果を保存するための辞書を作成
eneflow = {}
###太陽光はMJ換算なので注意###

## データ読み込み
# # 単位容量当たりのPV発電出力 [pu]
# PV_output = pd.read_csv("/home/shiga/INPUT/日射データ/solardata_saitama.csv")

heat_demand = pd.read_csv("./INPUT/heat_demand1y.csv") # 給湯需要[MJ]の読み込み
# パラメータ
T = 48*2 # 1日
DN = 6 # 需要家数

#変換効率
eta_PVpcs = 0.95
eta_Bpcs = 0.95

buy = pd.read_csv("./INPUT/buy_energy30.csv", encoding="shift_jis") # JPY/kW
sell_priENERGY_CONVERSION_FACTOR = 2.0
buy_max = 48.0 #買電最大値kW
sell_max = 48.0 #売電最大値kW
in_max = 6.0 #買電最大値kW
out_max = 6.0 #売電最大値kW
PV = pd.read_csv("./INPUT/PV_charge_date30.csv", encoding="shift_jis") #(0.85:損失係数、20:接地面積[m^2])
#PV = pd.read_csv("./INPUT/PV_OUTPUT_480_30min.csv", encoding="shift_jis") #(0.85:損失係数、20:接地面積[m^2])


#####################################
#電圧上下限値の設定
V_UL=101.00000000000000+6.000000000000000
V_LL=101.00000000000000-6.000000000000000

line=8

linear=pd.read_csv("./output/回帰係数・切片random2.csv", encoding="shift_jis")

###########################
w1 = 1  # コストの重み       

w2 = 0 # 電圧違反量の重み
w3 = 0  # 下げDRの重み
c774="1,0,0_h"

#test3 1000, 0.4 , 1000
#test4 0.5 , 0.5 , 0.2


linear_ap={}
linear_aq={}
linear_b={}            

linear_ap[1]=-0.967459143
linear_aq[1]=-0.12585453
linear_b[1]=-1.72E-06


linear_ap[2]=-1.291237696
linear_aq[2]=-0.167958302
linear_b[2]=-2.98E-06


linear_ap[3]=-1.615623189
linear_aq[3]=-0.210076519
linear_b[3]=-4.49E-06

linear_ap[4]=-0.97917953
linear_aq[4]=-0.127695895
linear_b[4]=-4.85E-06

linear_ap[5]=-1.306894759
linear_aq[5]=-0.169976303
linear_b[5]=-7.06E-06

linear_ap[6]=-1.635293645
linear_aq[6]=-0.212600759
linear_b[6]=-9.71E-06

linear_ap[7]=-1.965000546
linear_aq[7]=-0.258882354
linear_b[7]=-7.83E-05

linear_ap[8]=0.103517454
linear_aq[8]=0.592842926
linear_b[8]=0.999965215

####################################


N_PV = 70 #太陽光パネルの定格出力[KW]##
N_PV_PCS = 10##
BATTERY_POWER_CAPACITY = 1.5 # 蓄電池kW容量[kW]
BATTERY_ENERGY_CAPACITY = 6.3 # 蓄電池kWh容量[kWh]

tma = pd.read_csv("./INPUT/temputure1y.csv", encoding="shift_jis")  # 気温(1月)
tmf = pd.read_csv("./INPUT/東京都水温1年分.csv", encoding="shift_jis", dtype={"water tempereture": float}) # 水道水温度(1年)

HP_HEATING_CAPACITY = 16.2 # HP給湯機加熱能力[MJ/h]
WATER_SPECIFIC_HEAT = 0.0042 # 水の比熱[MJ/m3/℃]
ENERGY_CONVERSION_FACTOR = 3.6 # 変換係数[MJ/kWh]
HP_AUXILIARY_POWER = 0.013 # HP給湯機補機消費電力[kWh]
TANK_CAPACITY = 370 # 貯水槽の容量[L]
for day in range(0,365):
    date={}
    for t in range(1,T+1):
        date[t] = buy.iat[int(t + T/2*day - 1), 0]
    buy_priENERGY_CONVERSION_FACTOR = {}
    for t in range(1, T +1):
        buy_priENERGY_CONVERSION_FACTOR[t] = float(buy.iat[int(t+T/2*day-1), 1])
        
    P_PV ={}
    for t in range(1, T+1):
        P_PV[t] = float(PV.iat[int(t+T/2*day-1),1])
    
        
    dmh = {}
    for t in range(1, T+1):
        dmh[t] = float(heat_demand.iat[int(t+T/2*day-1), 1])

    # 各時刻における成績係数(COP)の算出 []
    k = {}
    p = {}
    cop = {}
    for t in range(1,T+1):
        p[t] = float(tma.iat[int(t+T/2*day+3),1])
        if p[t] >= 5:
            k[t] = 1
        elif p[t] > 2:
            k[t] = int(p[t])/30+0.8333
        else:
            k[t] = 0.9
        cop[t] = k[t]*(0.175*p[t]-0.1322*tmf.iat[int(t+T/2*day-1),2]+4.076)
    ## 需要家側機器の個別最適化モデル
    HEMS_IND = gp.Model("HEMS_Individual")
    
    ###############################
    
    # linear_ap={}
    # linear_aq={}
    # linear_b={}
    # for t in range(1, T+1):
    #     for k in range(1, line+1):
    #         if k <=6:
    #             linear_ap[k,t] = float(linear.iat[int(t+T/2*day-1), k*3+1])
    #             linear_aq[k,t] = float(linear.iat[int(t+T/2*day-1), k*3+2])
    #             linear_b[k,t] = float(linear.iat[int(t+T/2*day-1), k*3+3])
                
    #         else:
    #             linear_ap[k,t] = float(linear.iat[int(t+T/2*day-1), 1])
    #             linear_aq[k,t] = float(linear.iat[int(t+T/2*day-1), 2])
    #             linear_b[k,t] = float(linear.iat[int(t+T/2*day-1), 3])
    
                
    #######################################
    
    
    # 電力需要[kW]の読み込み
    P_dmd = {}
    demand = {}
    
    for i in range(1, DN+1):
        demand_file_base = 'a_electric_demand_0{:}_01.csv'
        demand_file = demand_file_base.format(i)
        demand[i] = pd.read_csv("./INPUT/electric_demand_1y/" + demand_file, encoding="shift_jis")
        for t in range(1, T + 1):
            P_dmd[i, t] = float(demand[i].iat[int(t + T/2*day - 1), 10])

    # 決定変数の定義
    # t=0は前日からの引継ぎでのみ使う。基本的にt=1~24
    # 売電電力[kW]
    vt = gp.GRB.CONTINUOUS
    vn = 'Selling_Power_of_HEMS'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_SG = {}
    for t in range(1,T+1):
        P_SG[t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

    # 買電電力[kW]
    vt = gp.GRB.CONTINUOUS
    vn = 'Purchacing_Power_of_HEMS'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_PG = {}
    for t in range(1,T+1):
        P_PG[t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
    #需要家の買電フラグ（買電中１、それ以外０）
    vt = gp.GRB.BINARY
    vn = 'Binary_buy'
    UB = 1.0
    LB = 0.0
    Delta_buy = {}
    for t in range(1,T+1):
        Delta_buy[t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
    #需要家の売電フラグ（売電中１、それ以外０）
    vt = gp.GRB.BINARY
    vn = 'Binary_sell'
    UB = 1.0
    LB = 0.0
    Delta_sell = {}
    for t in range(1,T+1):
        Delta_sell[t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

    # 一軒に供給される電力量[kWh]
    vt = gp.GRB.CONTINUOUS
    vn = 'Purchacing_in'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_in = {}
    for t in range(1, T+1):
        for i in range(1, DN+1):  # 1から8の範囲
            P_in[i, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # 一軒から放出される電力量[kWh]
    vt = gp.GRB.CONTINUOUS
    vn = 'Purchacing_out'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_out = {}
    for t in range(1, T+1):
        for i in range(1, DN+1):  # 1から8の範囲
            P_out[i, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)
            
    #需要家の供給フラグ（供給中１、それ以外０）
    vt = gp.GRB.BINARY
    vn = 'Binary_discharge'
    UB = 1.0
    LB = 0.0
    Delta_in = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_in[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # 需要家の放出フラグ（放出中１、それ以外０）
    vt = gp.GRB.BINARY
    vn = 'Binary_charge'
    UB = 1.0
    LB = 0.0
    Delta_out = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_out[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # 蓄電池充電電力[kW]
    vt = gp.GRB.CONTINUOUS
    vn = 'Charging_Power_of_BESS'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_Ch = {}
    for t in range(1,T+1):
        for i in range(1,DN+1):
            P_Ch[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # 蓄電池放電電力[kW]
    vt = gp.GRB.CONTINUOUS
    vn = 'Discharging_Power_of_BESS'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_DCh = {}
    for t in range(1,T+1):
        for i in range(1,DN+1):
            P_DCh[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # 蓄電池蓄電電力量[kWh]
    vt = gp.GRB.CONTINUOUS
    vn = 'Storaged_Energy_iBATTERY_ENERGY_CAPACITYESS'
    UB = gp.GRB.INFINITY
    LB = 0.0
    E_B = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            E_B[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    #蓄電池の充電フラグ（充電中１、それ以外０）
    vt = gp.GRB.BINARY
    vn = 'Binary_discharge'
    UB = 1.0
    LB = 0.0
    Delta_ch = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_ch[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # 蓄電池の放電フラグ（放電中１、それ以外０）
    vt = gp.GRB.BINARY
    vn = 'Binary_charge'
    UB = 1.0
    LB = 0.0
    Delta_dch = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_dch[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)


    # 各需要家のコスト[JPY]
    vt = gp.GRB.CONTINUOUS
    vn = 'cost1'
    UB = gp.GRB.INFINITY
    LB = -gp.GRB.INFINITY
    cost1 = {}
    for t in range(1,T+1):
        cost1[t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:})'.format(t), lb=LB, ub=UB)
        
    # ヒートポンプ給湯機の消費電力 [kW]
    vt = gp.GRB.CONTINUOUS
    vn = 'Power_Consumption_of_Heat-Pump-Water-Heater_in_HEMS'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_HP = {}
    for t in range(1,T+1):
        for i in range(1,DN+1):
            P_HP[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # HP給湯機から貯湯槽へ送られた熱量 [MJ]
    vt = gp.GRB.CONTINUOUS
    vn = 'Heat_Flow_from_HPWH_to_Tank'
    UB = gp.GRB.INFINITY
    LB = 0.0
    H_HS = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            H_HS[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

        
    # 運転開始時のエネルギーロス [MJ]
    vt = gp.GRB.CONTINUOUS
    vn = 'Energyloss_at_startup'
    UB = gp.GRB.INFINITY
    LB = 0.0
    H_EL ={}
    for t in range(1,T+1):
        for i in range(1,DN+1):
            H_EL[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)
        
    # HP給湯機の貯湯槽内の蓄熱量 [MJ]
    vt = gp.GRB.CONTINUOUS
    vn = 'Heat_storage_capacity_in_storage_tanks_of_HPWH'
    UB = gp.GRB.INFINITY
    LB = 0.0
    H_SC = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            H_SC[i,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

    # HP給湯機運転指標（運転中 1，停止中 0）
    vt = gp.GRB.BINARY
    vn = 'Indicator_of_HPWH_running'
    Delta_r = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_r[i,t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:},{:})'.format(i,t))
        
    # HP給湯機運転開始指標（運転開始時 1，それ以外 0）
    vt = gp.GRB.BINARY
    vn = 'Indicator_of_HPWH_starting'
    Delta_st = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_st[i,t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:},{:})'.format(i,t))

    # HP給湯機運転停止指標（運転停止時 1，それ以外 0）
    vt = gp.GRB.BINARY
    vn = 'Indicator_of_HPWH_stopping'
    Delta_sp = {}
    for t in range(0,T+1):
        for i in range(1,DN+1):
            Delta_sp[i,t] = HEMS_IND.addVar(vtype=vt,name=vn+'({:},{:})'.format(i,t))


##################################



    vt = gp.GRB.CONTINUOUS
    vn = '電圧'
    UB = gp.GRB.INFINITY
    LB = -gp.GRB.INFINITY
    V_d={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            V_d[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)
            
    vt = gp.GRB.CONTINUOUS
    vn = '電圧変化'
    UB = gp.GRB.INFINITY
    LB = -gp.GRB.INFINITY
    V_dd={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            V_dd[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)
    
            
    vt = gp.GRB.CONTINUOUS
    vn = '有効電力'
    UB = gp.GRB.INFINITY
    LB = -gp.GRB.INFINITY
    ap={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            ap[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)
    
    vt = gp.GRB.CONTINUOUS
    vn = '無効電力'
    UB = gp.GRB.INFINITY
    LB = -gp.GRB.INFINITY
    aq={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            aq[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)        
    
    vt = gp.GRB.CONTINUOUS
    vn = '上限セーフ値'
    UB = gp.GRB.INFINITY
    LB = 0.0
    V_UN={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            V_UN[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)  
    
    vt = gp.GRB.CONTINUOUS
    vn = '下限セーフ値'
    UB = gp.GRB.INFINITY
    LB = 0.0
    V_LN={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            V_LN[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)  
            
    vt = gp.GRB.CONTINUOUS
    vn = '上限違反値'
    UB = gp.GRB.INFINITY
    LB = 0.0
    V_ULV={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            V_ULV[k, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)  
    
    vt = gp.GRB.CONTINUOUS
    vn = '下限違反値'
    UB = gp.GRB.INFINITY
    LB = 0.0
    V_LLV={}
    for t in range(1, T+1):
        for k in range(1, line+1): 
            V_LLV[k,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)  
    
    vt = gp.GRB.BINARY
    vn = '上限バイナリ(違反)'
    UB = 1.0
    LB = 0.0
    Delta_ulv = {}
    for t in range(1, T+1):
        for k in range(1, line+1):
            Delta_ulv[k,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)
    
    vt = gp.GRB.BINARY
    vn = '上限バイナリ(セーフ)'
    UB = 1.0
    LB = 0.0
    Delta_un = {}
    for t in range(1, T+1):
        for k in range(1, line+1):
            Delta_un[k,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)
            
    
    vt = gp.GRB.BINARY
    vn = '下限バイナリ(違反)'
    UB = 1.0
    LB = 0.0
    Delta_llv = {}
    for t in range(1, T+1):
        for k in range(1, line+1):
            Delta_llv[k,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)
            
    vt = gp.GRB.BINARY
    vn = '下限バイナリ(セーフ)'
    UB = 1.0
    LB = 0.0
    Delta_ln = {}
    for t in range(1, T+1):
        for k in range(1, line+1):
            Delta_ln[k,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(k, t), lb=LB, ub=UB)

# 一軒の買電電力量[kWh]
    vt = gp.GRB.CONTINUOUS
    vn = 'Purchacing_in'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_pg = {}
    for t in range(1, T+1):
        for i in range(1, DN+1):  # 1から8の範囲
            P_pg[i, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)

# 一軒の売電電力量[kWh]
    vt = gp.GRB.CONTINUOUS
    vn = 'Purchacing_in'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_sg = {}
    for t in range(1, T+1):
        for i in range(1, DN+1):  # 1から8の範囲
            P_sg[i, t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:})'.format(i, t), lb=LB, ub=UB)    
            
    # 一軒の売電電力量[kWh]
    vt = gp.GRB.CONTINUOUS
    vn = 'Purchacing_in'
    UB = gp.GRB.INFINITY
    LB = 0.0
    P_vv8 = {}
    for t in range(1, T+1):
        for i in range(1, DN+1):  # 1から8の範囲
            for j in range(1, DN+1):
                P_vv8[i,j,t] = HEMS_IND.addVar(vtype=vt, name=vn + '({:},{:},{:})'.format(i, j, t), lb=LB, ub=UB)   
                #iからjに供給
    
            
            
    for t in range(1,T+1):
         for k in range(1,line+1):  
            if k <=6:
                HEMS_IND.addConstr(ap[k,t]==P_in[k, t])
                HEMS_IND.addConstr(aq[k,t]==P_in[k, t]*0.1)
                
            if k ==7:
                HEMS_IND.addConstr(ap[k,t]==(P_in[4, t]+P_in[5, t]+P_in[6, t])*1.003)
                HEMS_IND.addConstr(aq[k,t]==((P_in[4, t]+P_in[5, t]+P_in[6, t]))*0.1*1.004)
            
            if k ==8:
                HEMS_IND.addConstr(ap[k,t]==((P_in[4, t]+P_in[5, t]+P_in[6, t])*1.003*1.014)+(P_in[1, t]+P_in[2, t]+P_in[3, t])*1.003)
                HEMS_IND.addConstr(aq[k,t]==(((P_in[4, t]+P_in[5, t]+P_in[6, t])*0.1*1.004*1.019)+(P_in[1, t]+P_in[2, t]+P_in[3, t])*1.007*0.1))
                
                
    # for t in range(1,T+1):
    #     for k in range(1,line+1):  
    #         if k <=6:
    #             HEMS_IND.addConstr(ap[k,t]==P_in[k, t])
    #             HEMS_IND.addConstr(aq[k,t]==P_in[k, t]*0.1)
                
            
    #         if k ==7:
    #             HEMS_IND.addConstr(ap[k,t]==(P_in[4, t]+P_in[5, t]+P_in[6, t]))
    #             HEMS_IND.addConstr(aq[k,t]==((P_in[4, t]+P_in[5, t]+P_in[6, t]))*0.1)
            
    #         if k ==8:
    #             HEMS_IND.addConstr(ap[k,t]==((P_in[4, t]+P_in[5, t]+P_in[6, t]))+(P_in[1, t]+P_in[2, t]+P_in[3, t]))
    #             HEMS_IND.addConstr(aq[k,t]==(((P_in[4, t]+P_in[5, t]+P_in[6, t])*0.1)+(P_in[1, t]+P_in[2, t]+P_in[3, t])*0.1))
                
    for t in range(1,T+1):
        for k in range(1,line+1):  
                HEMS_IND.addConstr(V_dd[k,t]==linear_ap[k]*ap[k,t]+linear_aq[k]*aq[k,t]+linear_b[k])
                
    for t in range(1,T+1):
        for k in range(1,line+1):
            if k ==8:
               HEMS_IND.addConstr(V_d[k,t]==99.8) 
            if k <=3:  
                HEMS_IND.addConstr(V_d[k,t]==V_dd[k,t]+V_d[8,t])
            if k ==7:
                HEMS_IND.addConstr(V_d[k,t]==V_dd[k,t]+V_d[8,t])
            if 4<= k <=6:
                HEMS_IND.addConstr(V_d[k,t]== V_dd[k,t]+V_d[7,t])

   
    #    #右辺の文字を0以上とする制約    
   
    for t in range(1,T+1):
        for k in range(1,line+1):
                HEMS_IND.addConstr( V_ULV[k,t]>= 0)
            
    for t in range(1,T+1):
        for k in range(1,line+1):   
                HEMS_IND.addConstr( V_UN[k,t] >= 0)
    
    
    
    for t in range(1,T+1):
        for k in range(1,line+1):
                HEMS_IND.addConstr( V_LN[k,t] >= 0)
    for t in range(1,T+1):
        for k in range(1,line+1):
                HEMS_IND.addConstr( V_LLV[k,t]>= 0)
    '''
    
    
    for t in range(1,T+1):
         for k in range(1,line+1):
                 HEMS_IND.addConstr( V_LLV[k,t] *Delta_llv[k,t]>= 0)
    for t in range(1,T+1):
         for k in range(1,line+1):   
                HEMS_IND.addConstr( V_UN[k,t]*Delta_un[k,t] >= 0)
    for t in range(1,T+1):
        for k in range(1,line+1):
                HEMS_IND.addConstr( V_LN[k,t]*Delta_un[k,t] >= 0)
    for t in range(1,T+1):
         for k in range(1,line+1):
                 HEMS_IND.addConstr( V_ULV[k,t]*Delta_ulv[k,t] >= 0)           
    for t in range(1,T+1):
         for k in range(1,line+1):   
                 HEMS_IND.addConstr(Delta_ln[k,t] + Delta_llv[k,t] == 1.0)
    for t in range(1,T+1):
         for k in range(1,line+1):   
                 HEMS_IND.addConstr(Delta_ulv[k,t] + Delta_un[k,t] == 1.0)             
 '''
            
     #各ノードにおけるの許容電圧の上下限の違反値を求める制約式        
    #上限違反
    for t in range(1,T+1):
        for k in range(1,line+1):
                HEMS_IND.addConstr(V_UL-V_d[k,t] == V_UN[k,t]-(V_ULV[k,t]))
                
#下限違反            
    for t in range(1,T+1):
        for k in range(1,line+1): 
                HEMS_IND.addConstr(V_d[k,t]-V_LL == V_LN[k,t]-(V_LLV[k,t]))
    ## 目的関数
    # （注）決定変数の時間軸（t=1~24）と定数の時間軸（t=0~23）が異なる点に注意。
 

# 最終目的関数: コスト + 電圧違反量の重み付け和
    final_objective = w1 * gp.quicksum(buy_priENERGY_CONVERSION_FACTOR[t]*P_PG[t]-sell_priENERGY_CONVERSION_FACTOR*P_SG[t] for t in range(1,T+1)) + w2 * gp.quicksum(V_ULV[k, t] + V_LLV[k, t] for k in range(1, line) for t in range(1, T+1)) + w3 *  gp.quicksum(P_PG[t] for t in range(13*2+1, 16*2+1) for t in range(48+13*2+1, 48+16*2+1))
    HEMS_IND.setObjective(final_objective, gp.GRB.MINIMIZE)
  

    # 前日のデータ引き継ぎ
    if day >= 1:
        
        for i in range(1,DN+1):
            previous_result = pd.read_csv("./OUTPUT/{:}/{:}_Several_{:}_{:}.csv".format(c774,c774,i,day - 1), encoding="shift_jis")
            (H_SC[i,0] == float(previous_result.iat[int(T/2-1),20]))
            (Delta_r[i,0] == previous_result.iat[int(T/2-1),15])
            (Delta_st[i,0] == previous_result.iat[int(T/2-1),16])
            (Delta_sp[i,0] == previous_result.iat[int(T/2-1),17])
            (E_B[i,0] == previous_result.iat[int(T/2-1),13])
    ## 制約条件
    # 需給バランス制約    
    # 複数需要家モデル
    # 全体需給バランス制約
    for t in range(1, T+1):
        HEMS_IND.addConstr(P_SG[t] + gp.quicksum(P_in[i, t] for i in range(1, DN+1)) == P_PG[t] + gp.quicksum(P_out[i, t] for i in range(1, DN+1)))

    # 一軒に供給される電力量[kWh]と一軒から放出される電力量[kWh]
    for t in range(1, T+1):
        for i in range(1, DN+1):
            HEMS_IND.addConstr(P_PV[t] + P_in[i, t] + P_DCh[i, t] == P_dmd[i,t] + P_out[i, t] + P_Ch[i, t] + P_HP[i, t], name='demand_balanENERGY_CONVERSION_FACTOR({:},{:})'.format(i, t))
        
        
#売電バランス制約
    for t in range(1, T+1):     
            HEMS_IND.addConstr(P_SG[t]== gp.quicksum(P_sg[i, t] for i in range(1, DN+1)))
#買電バランス制約
    for t in range(1, T+1):     
            HEMS_IND.addConstr(P_PG[t]== gp.quicksum(P_pg[i, t] for i in range(1, DN+1)))
    for t in range(1, T+1):     
        for i in range(1, DN+1):
            HEMS_IND.addConstr(P_out[i,t]== P_sg[i, t] + gp.quicksum(P_vv8[i,j,t] for j in range(1, DN+1)))
            HEMS_IND.addConstr(P_in[i,t]== P_pg[i, t] + gp.quicksum(P_vv8[j,i,t] for j in range(1, DN+1)))    
            HEMS_IND.addConstr(P_vv8[i,i,t]==0)
            
      
        
    # 買電と売電の容量制約
    for t in range(1,T+1):
        HEMS_IND.addConstr(P_PG[t] <= buy_max * Delta_buy[t])
    for t in range(1,T+1):
        HEMS_IND.addConstr(P_SG[t] <= sell_max * Delta_sell[t])
    # 買電と売電の同時禁止制約
    for t in range(1,T+1):
        HEMS_IND.addConstr(Delta_buy[t] + Delta_sell[t] <= 1.0)

    # 供給と放出の容量制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(P_in[i,t] <= in_max * Delta_in[i,t])
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(P_out[i,t] <= out_max * Delta_out[i,t])
    # 供給と放出の同時禁止制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(Delta_in[i,t] + Delta_out[i,t] <= 1.0)
        
    #蓄電池充放電容量（kW容量）制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(P_DCh[i,t] <= BATTERY_POWER_CAPACITY * Delta_dch[i,t])
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(P_Ch[i,t] <= BATTERY_POWER_CAPACITY * Delta_ch[i,t])
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(Delta_ch[i,t] + Delta_dch[i,t] <= 1.0)
                
    # 蓄電池蓄電容量（kWh容量）制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(BATTERY_ENERGY_CAPACITY*0.2 <= E_B[i,t])
            HEMS_IND.addConstr(E_B[i,t] <= BATTERY_ENERGY_CAPACITY)

    # 蓄電池蓄電量更新制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(E_B[i,t] == E_B[i,t-1] + (eta_Bpcs*P_Ch[i,t] - (1/eta_Bpcs)*P_DCh[i,t]))
        
    # 蓄電池蓄電量始端・終端制約##
    for i in range(1,DN+1):
         HEMS_IND.addConstr(E_B[i,0] == 0.5*BATTERY_ENERGY_CAPACITY)
         HEMS_IND.addConstr(E_B[i,T] == 0.5*BATTERY_ENERGY_CAPACITY)

    # HP給湯機に関する制約条件
    # HP給湯機の消費電力[kWh]と生成する熱量[MJ]に関する等式制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(ENERGY_CONVERSION_FACTOR*cop[t]*(P_HP[i,t] - HP_AUXILIARY_POWER*Delta_sp[i,t]) == H_HS[i,t] + H_EL[i,t])

    # 生成する熱量に関する上下限制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(0.05*HP_HEATING_CAPACITY*Delta_r[i,t] <= H_HS[i,t]) # 定格出力で3分以上運転する
        
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(H_HS[i,t] <= HP_HEATING_CAPACITY * Delta_r[i,t])
        
    # 貯湯槽内の蓄熱量更新に関する制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(H_SC[i,t] - H_SC[i,t-1] == H_HS[i,t] - dmh[t])
        
    # 貯湯槽の容量制約（沸き上げ温度65度）
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(H_SC[i,t] >= 0.2*(WATER_SPECIFIC_HEAT*70*TANK_CAPACITY))
            HEMS_IND.addConstr(H_SC[i,t] <= (WATER_SPECIFIC_HEAT*70*TANK_CAPACITY))
            
        
    # 起動時の運転継続制約（運転を開始した場合は最低でも 1 時間は運転を継続する制約）
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(H_HS[i,t] + H_HS[i,t-1] >= HP_HEATING_CAPACITY * Delta_r[i,t])
        
    # 運転開始時のエネルギーロス
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(H_EL[i,t] == 0.01*HP_HEATING_CAPACITY*Delta_st[i,t])
        
    # HP給湯機の運転に関する制約条件
    # 起動と停止の同時禁止制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(Delta_st[i,t] + Delta_sp[i,t] <= 1)
    # 運転状態の更新制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(Delta_r[i,t] - Delta_r[i,t-1] == Delta_st[i,t] - Delta_sp[i,t])
    # 運転中    
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(Delta_st[i,t] <= Delta_r[i,t])
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(Delta_sp[i,t] <= 1 - Delta_r[i,t])

    # 運転開始後定格運転を行う制約
    for t in range(1,T+1):
        for i in range(1,DN+1):
            HEMS_IND.addConstr(H_HS[i,t] >= HP_HEATING_CAPACITY*(Delta_st[i,t] - Delta_r[i,t]))
            


    # 前回の最適化を初期値にする制約条件##
    for i in range(1,DN+1):
         HEMS_IND.addConstr(H_SC[i,0] == H_SC[i,T])

    #時間当たりの電力コスト
    for t in range(1,T+1):
        HEMS_IND.addConstr(cost1[t] == buy_priENERGY_CONVERSION_FACTOR[t]*P_PG[t]-sell_priENERGY_CONVERSION_FACTOR*P_SG[t])

    HEMS_IND.update()
  
    
    print('制約条件設定終了')
            
            

    ## 最適化
    status = HEMS_IND.optimize()
   
    
    print(status)
    print(HEMS_IND.ObjVal)
    print('最適化終了')
    
    HEMS_FIX =HEMS_IND.fixed()
    HEMS_FIX.setParam("Presolve",0)
    HEMS_FIX.optimize()
    
    for t in range(1, T+1):
        for k in range(1,line+1):
            print('ボーマンダ')
            print(V_d[k,t])
    

# 電力価格（双対変数）
    electric_priENERGY_CONVERSION_FACTOR = pd.DataFrame(np.zeros((DN+1, T+1)), index=range(1, DN+2), columns=range(0,T+1))
    for t in range(1, T+1):
        for i in range(1,DN+1):
            electric_priENERGY_CONVERSION_FACTOR.iloc[i,t] = HEMS_FIX.getConstrByName('demand_balanENERGY_CONVERSION_FACTOR({:},{:})'.format(i, t)).Pi

    # 計算結果の保存
    for i in range(1,line+1):
        # 計算結果を保存するDataFrameの作成
        if i<=6:
            eneflow = {}
            eneflow['date'] = [date[t] for t in range(1, 48+ 1)]
            eneflow['cost_{:}'.format(i)] = [cost1[t].X for t in range(1,48+1)]
            eneflow['buy_priENERGY_CONVERSION_FACTOR'] = [buy_priENERGY_CONVERSION_FACTOR[t] for t in range(1,48+1)]
            eneflow['sell_priENERGY_CONVERSION_FACTOR'] = [sell_priENERGY_CONVERSION_FACTOR for t in range(1,48+1)]
            eneflow['P_PG_{:}'.format(i)] = [P_PG[t].X for t in range(1,48+1)]
            eneflow['P_SG_{:}'.format(i)] = [P_SG[t].X for t in range(1,48+1)]
            eneflow['P_in_{:}'.format(i)] = [P_in[i, t].X for t in range(1, 48+1)]
            eneflow['P_out_{:}'.format(i)] = [P_out[i, t].X for t in range(1, 48+1)]
            eneflow['P_dmd_{:}'.format(i)] = [P_dmd[i,t] for t in range(1,48+1)]
            eneflow['P_PV_{:}'.format(i)] = [P_PV[t] for t in range(1,48+1)]
            eneflow['P_Ch_{:}'.format(i)] = [P_Ch[i,t].X for t in range(1,48+1)]
            eneflow['P_DCh_{:}'.format(i)] = [P_DCh[i,t].X for t in range(1,48+1)]
            eneflow['E_B_{:}'.format(i)] = [E_B[i,t].X for t in range(1,48+1)]
            eneflow['P_HP_{:}'.format(i)] = [P_HP[i,t].X for t in range(1,48+1)]
        
            eneflow['Delta_r_{:}'.format(i)] = [Delta_r[i,t].X for t in range(1,48+1)]
            eneflow['Delta_st_{:}'.format(i)] = [Delta_st[i,t].X for t in range(1,48+1)]
            eneflow['Delta_sp_{:}'.format(i)] = [Delta_sp[i,t].X for t in range(1,48+1)]
            eneflow['H_HS_{:}'.format(i)] = [H_HS[i,t].X for t in range(1,48+1)]
            eneflow['H_EL_{:}'.format(i)] = [H_EL[i,t].X for t in range(1,48+1)]
            eneflow['H_SC_{:}'.format(i)] = [H_SC[i,t].X for t in range(1,48+1)]
            eneflow['Heat_demand_{:}'.format(i)] = [dmh[t] for t in range(1,48+1)]
            eneflow['cop_{:}'.format(i)] = [cop[t] for t in range(1,48+1)]
            eneflow['electric_priENERGY_CONVERSION_FACTOR_{:}'.format(i)] = [electric_priENERGY_CONVERSION_FACTOR.at[i, t] for t in range(1, 48+1)]
            
            eneflow['P_out_{:}_minus'.format(i)] = [-P_out[i, t].X for t in range(1, 48+1)]
            eneflow['P_Ch_{:}_minus'.format(i)] = [-P_Ch[i, t].X for t in range(1, 48+1)]
            eneflow['P_HP_{:}_minus'.format(i)] = [-P_HP[i, t].X for t in range(1, 48+1)]

            eneflow['P_pg_{:}'.format(i)] = [P_pg[i, t].X for t in range(1,48+1)] 
            eneflow['P_sg_{:}'.format(i)] = [P_sg[i, t].X for t in range(1,48+1)] 
            
            eneflow['P_vv8_1_{:}'.format(i)] = [P_vv8[1,i,t].X for t in range(1,48+1)]
            eneflow['P_vv8_2_{:}'.format(i)] = [P_vv8[2,i,t].X for t in range(1,48+1)] 
            eneflow['P_vv8_3_{:}'.format(i)] = [P_vv8[3,i,t].X for t in range(1,48+1)] 
            eneflow['P_vv8_4_{:}'.format(i)] = [P_vv8[4,i,t].X for t in range(1,48+1)] 
            eneflow['P_vv8_5_{:}'.format(i)] = [P_vv8[5,i,t].X for t in range(1,48+1)] 
            eneflow['P_vv8_6_{:}'.format(i)] = [P_vv8[1,i,t].X for t in range(1,48+1)] 
            eneflow['P_vv8_total_{:}'.format(i)] = [sum(P_vv8[j, i, t].X for j in range(1, 6+1)) for t in range(1, 48+1)]
                            
            eneflow['電圧_{:}'.format(i)]=[V_d[i, t].X for t in range(1,48+1)]
            eneflow['電圧変化_{:}'.format(i)]=[V_dd[i, t].X for t in range(1,48+1)]
            eneflow['有効電力_{:}'.format(i)]=[ap[i, t].X for t in range(1,48+1)]
            eneflow['無効電力_{:}'.format(i)]=[aq[i, t].X for t in range(1,48+1)]
            eneflow['上限違反値_{:}'.format(i)] = [V_ULV[i,t].X for t in range(1,48+1)]
            eneflow['下限違反値_{:}'.format(i)] = [V_LLV[i, t].X for t in range(1,48+1)]
            
            
            eneflow['上限セーフ値_{:}'.format(i)]=[ V_UN[i,t].X for t in range(1,48+1)]
            eneflow['下限セーフ値_{:}'.format(i)]=[V_LN[i, t].X for t in range(1,48+1)]
            
        else:
            eneflow['電圧_{:}'.format(i)]=[V_d[i, t].X for t in range(1,48+1)]
            eneflow['有効電力_{:}'.format(i)]=[ap[i, t].X for t in range(1,48+1)]
            eneflow['無効電力_{:}'.format(i)]=[aq[i, t].X for t in range(1,48+1)]
            eneflow['上限違反値_{:}'.format(i)] = [V_ULV[i,t].X for t in range(1,48+1)]
            eneflow['下限違反値_{:}'.format(i)] = [V_LLV[i, t].X for t in range(1,48+1)]
            
            
            eneflow['上限セーフ値_{:}'.format(i)]=[ V_UN[i,t].X for t in range(1,48+1)]
            eneflow['下限セーフ値_{:}'.format(i)]=[V_LN[i, t].X for t in range(1,48+1)]
            
       



        # numpy配列に変換
        for key, value in eneflow.items():
            eneflow[key] = np.array(value)

        # 計算結果をDataFrameに変換
        eneflow_df = pd.DataFrame(eneflow)

        # 各家庭ごとにデータを保存
        eneflow_df.to_csv("./OUTPUT/{:}/{:}_Several_{:}_{:}.csv".format(c774,c774,i,day), encoding="shift_jis")