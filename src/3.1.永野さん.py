import gurobipy as gp
import pandas as pd

T=48#時間のコマ数
I=3#需要家の数
line=4
#変換効率
eta_PVpcs = 0.95
eta_Bpcs = 0.95
t_ref=30/60
N_B_PCS = 1.5 # 蓄電池kW容量[kW]
N_B = 6.3 # 蓄電池kWh容量[kWh]
sell_price = 2.0#売電単価[JPY/kW]
buy_MAX=5.0
sell_MAX=5.0
r1 = 0.05  # 最低熱製造率
r2 = 0.1  # HP給湯器起動時ロス率
cw = 0.0042  # 水の比熱[MJ/L]
ce = 3.6  # 変換係数
vlt = 370  # 貯水槽の容量[L]
cph = 16.2  # HP給湯器加熱能力[MJ]
xhp = 0.013  # HP給湯器補器消費電力[kWh]
#電圧上下限値の設定
V_UL=101.00000000000000+6.000000000000000
V_LL=101.00000000000000-6.000000000000000
for i in range(1,I+1):
    filename = 'a_electric_demand_0{:}_01.csv'.format(i)
    dmd = pd.read_csv("./INPUT/electric_demand_1y_2004/"+filename, encoding="shift_jis") # 電力需要[kW]の読み込み
buy= pd.read_csv("./INPUT/buy_energy30.csv", encoding="shift_jis")#電気料金[JPY/kW]
spot_price_data = pd.read_csv("./INPUT/spot_market_price_30min.csv", encoding="shift_jis")#スポット市場の電力価格
reserve_price = pd.read_csv("./INPUT/jissekichi_30.csv", encoding="shift_jis")#入札価格の読み込み
PV = pd.read_csv("./INPUT/PV_output.csv")#太陽光出力4.8kW想定
H_dme = pd.read_csv("./INPUT/heat_demand1y.csv", encoding="shift_jis")  # 熱需要の読み込み
T_out = pd.read_csv("./INPUT/temperature1y.csv", encoding="shift_jis")  # 外気温度の読み込み
T_water = pd.read_csv("./INPUT/saitama_WT1y.csv", encoding="shift_jis")  # 給水温度
voltage_estimate = pd.read_csv("./output/回帰係数・切片random_walk2_robust.csv", encoding="shift_jis")  # 電圧推定に必要なパラメータ
#入札電力量

#調整力単価
price_balancing = {}
for t in range(1,T+1):
    price_balancing[t] = float(reserve_price.iat[t-1,5])
    
#スポット市場単価

time = {}
for t in range(1,T+1):
    time[t] = dmd.iat[t-1,0]
    
print(time)
#太陽光出力[kW・30min]
P_PV ={}
for t in range(1,T+1):
    for i in range(I):
        P_PV[t,i] = float(PV.iat[t+30*T,2])/1000
        
    print(P_PV[t,1])
    
#家庭の需要
P_dmd = {}
for t in range(T+1):
    for i in range(I):
        P_dmd[t,i] = float(dmd.iat[t-1,10])
        #print(P_dmd)
#1kW・30min
buy_price = {}
for t in range(1,+T+1):
        buy_price[t] = float(buy.iat[t-1,2])
#スポット市場単価
spot_price={}
for t in range(1,+T+1):
    for i in range(I):
        spot_price[t] = float(spot_price_data.iat[t,2])
#print(spot_price)
H_dmd={}
for t in range(1,T+1):
    for i in range(I):
        H_dmd[t,i]=float(H_dme.iat[t-1+T,2])#熱需要の切り抜き
        
tma={}
k={}
cop={}
        
        
tmf={}
    
for t in range(1,T+1):
    tmf[t] = float(T_water.iat[int(t-1),2])
for t in range(1,T+1):
        tma[t]=float(T_out.iat[int(t),2])
        
        if tma[t]>=5:
            k[t]=1
        elif tma[t]>2:
            k[t]=tma[t]/30+0.8333
        else:
            k[t]=0.9
        cop[t]=k[t]*(0.175*tma[t]-0.1322*tmf[t]+4.076)#成績係数の決定
        
linear_ap={}
linear_aq={}
linear_b={}  
for i in range(I):
    linear_ap[i]=voltage_estimate.iat[0,3*i]
    linear_aq[i]=voltage_estimate.iat[0,3*i+1] 
    linear_b[i]=voltage_estimate.iat[0,3*i+2]
    #print(linear_ap[i])
    # print(linear_aq[i])
    # print(linear_b[i])

benefit_df = pd.DataFrame()
        
AG_PM= gp.Model("aggregator_benefit")
# 蓄電池充電電力[kW]
vt = gp.GRB.CONTINUOUS
vn = 'Charging_Power_of_BESS'
UB = gp.GRB.INFINITY
LB = 0.0
P_Ch = {}
for i in range(I):
    for t in range(T+1):
        
            P_Ch[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

# 蓄電池放電電力[kW]
vt = gp.GRB.CONTINUOUS
vn = 'Discharging_Power_of_BESS'
UB = gp.GRB.INFINITY
LB = 0.0
P_DCh = {}
for i in range(I):
    for t in range(T+1):
        
            P_DCh[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

#運転開始コスト
vt = gp.GRB.CONTINUOUS
vn = 'open_Electricity'
UB = gp.GRB.INFINITY
LB = -gp.GRB.INFINITY
opening = {}
for t in range(T+1):
   for i in range(I):
        opening[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)


# 買電電力[kW]
vt = gp.GRB.CONTINUOUS
vn = 'Purchacing_Power_of_HEMS'
UB = gp.GRB.INFINITY
LB = 0.0
P_PG = {}
for t in range(1,T+1):
    for i in range(I):
        P_PG[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
#電力系統との取引量[kW] 
vt = gp.GRB.CONTINUOUS
vn = 'Purchacing_Power_of_HEMS'
UB = gp.GRB.INFINITY
LB = -gp.GRB.INFINITY
P_pg = {}
for t in range(1,T+1):
    for i in range(I):
        P_pg[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
# 売電電力[kW]
vt = gp.GRB.CONTINUOUS
vn = 'Selling_Power_of_HEMS'
UB = gp.GRB.INFINITY
LB = 0.0
P_SG = {}
for t in range(1,T+1):
    for i in range(I):
        P_SG[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
# HP消費電力[kW]
vt = gp.GRB.CONTINUOUS
vn = 'Consume_Power_of_HP'
UB = gp.GRB.INFINITY
LB = 0.0
P_HP = {}
for t in range(1,T+1):
    for i in range(I):
        P_HP[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
#HP運転開始時エネルギーロス
    vt = gp.GRB.CONTINUOUS
    vn = 'HHT_flow'
    UB = gp.GRB.INFINITY
    LB = 0.0
    H_ini = {}
    for t in range(0,T+1):
        for i in range(I):
            H_ini[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)


# 蓄電池蓄電電力量[kWh]
vt = gp.GRB.CONTINUOUS
vn = 'Storaged_Energy_in_BESS'
UB = gp.GRB.INFINITY
LB = 0.0
E_B = {}
for t in range(0,T+1):
    for i in range(I):
        E_B[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

#蓄電池の充電フラグ（充電中１、それ以外０）
vt = gp.GRB.BINARY
vn = 'Binary_discharge'
UB = 1.0
LB = 0.0
Delta_ch = {}
for t in range(0,T+1):
    for i in range(I):
        Delta_ch[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

# 蓄電池の放電フラグ（放電中１、それ以外０）
vt = gp.GRB.BINARY
vn = 'Binary_charge'
UB = 1.0
LB = 0.0
Delta_dch = {}
for t in range(0,T+1):
    for i in range(I):
        Delta_dch[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)

#売買変数
vt = gp.GRB.BINARY
vn = 'Binary_power_grid'
UB = 1.0
LB = 0.0
Delta_PG = {}
for t in range(0,T+1):
    for i in range(I):
        Delta_PG[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
#HP給湯機運転変数
vt = gp.GRB.BINARY
vn = 'Binary_produce'
UB = 1.0
LB = 0.0
Delta_prod = {}
for t in range(0,T+1):
    for i in range(I):
        Delta_prod[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
#HP給湯機運転開始変数
vt = gp.GRB.BINARY
vn = 'Binary_Start'
UB = 1.0
LB = 0.0
Delta_S = {}
for t in range(0,T+1):
    for i in range(I):
        Delta_S[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
#HP給湯機運転開始変数
vt = gp.GRB.BINARY
vn = 'Binary_Finish'
UB = 1.0
LB = 0.0
Delta_F = {}
for t in range(0,T+1):
    for i in range(I):
        Delta_F[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        
#   熱製造量
vt = gp.GRB.CONTINUOUS
vn = 'produce_heat'
UB = gp.GRB.INFINITY
LB = 0.0
H_prod = {}
for t in range(0,T+1):
    for i in range(I):
        H_prod[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
#貯湯量
vt = gp.GRB.CONTINUOUS
vn = 'tank_heat'
UB = gp.GRB.INFINITY
LB = 0.0
H_tank = {}
for t in range(0,T+1):
    for i in range(I):
        H_tank[t,i] = AG_PM.addVar(vtype=vt,name=vn+'({:})'.format(t),lb=LB,ub=UB)
        

#HP給湯機の初期の状態
for i in range(I):
    Delta_prod[0,i]=0
    Delta_S[0,i]=0
    Delta_F[0,i]=0
    
##電圧に関する決定変数
vt = gp.GRB.CONTINUOUS
vn = '電圧'
UB = gp.GRB.INFINITY
LB = -gp.GRB.INFINITY
V_d={}
for t in range(1, T+1):
    for k in range(I): 
        V_d[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)
        
vt = gp.GRB.CONTINUOUS
vn = '電圧変化'
UB = gp.GRB.INFINITY
LB = -gp.GRB.INFINITY
V_dd={}
for t in range(0, T+1):
    for k in range(I): 
        V_dd[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)
        
        
vt = gp.GRB.CONTINUOUS
vn = '有効電力'
UB = gp.GRB.INFINITY
LB = -gp.GRB.INFINITY
ap={}
for t in range(1, T+1):
    for k in range(I): 
        ap[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)

vt = gp.GRB.CONTINUOUS
vn = '無効電力'
UB = gp.GRB.INFINITY
LB = -gp.GRB.INFINITY
aq={}
for t in range(1, T+1):
    for k in range(0,I): 
        aq[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)        

vt = gp.GRB.CONTINUOUS
vn = '上限セーフ値'
UB = gp.GRB.INFINITY
LB = 0.0
V_UN={}
for t in range(1, T+1):
    for k in range(0,I): 
        V_UN[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)  
        

vt = gp.GRB.CONTINUOUS
vn = '下限セーフ値'
UB = gp.GRB.INFINITY
LB = 0.0
V_LN={}
for t in range(1, T+1):
    for k in range(0,I): 
        V_LN[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)  
        
vt = gp.GRB.CONTINUOUS
vn = '上限違反値'
UB = gp.GRB.INFINITY
LB = 0.0
V_ULV={}
for t in range(1,T+1):
    for k in range(I): 
        V_ULV[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)  

vt = gp.GRB.CONTINUOUS
vn = '下限違反値'
UB = gp.GRB.INFINITY
LB = 0.0
V_LLV={}
for t in range(1,T+1):
    for k in range(I): 
        V_LLV[t,k] = AG_PM.addVar(vtype=vt, name=vn + '({:},{:})'.format(t,k), lb=LB, ub=UB)  
        
        
AG_PM.update()

    # 目的関数
AG_PM.setObjectiveN(gp.quicksum(-(P_UP[t,i] + P_DW[t,i]) * price_balancing[t]+ (spot_price[t]-buy_price[t])*P_PG[t,i] for t in range(1, T+1) for i in range(I)), index=0, priority=0)
AG_PM.setObjectiveN(gp.quicksum(P_PG[t,i] * buy_price[t] - P_SG[t,i] * sell_price for t in range(1, T+1) for i in range(I)), index=1, priority=0)
AG_PM.setObjectiveN(gp.quicksum(V_ULV[t,k] + V_LLV[t,k]+V_ULV_UP[t,k]+V_ULV_DW[t,k] + V_LLV_UP[t,k]+ V_LLV_DW[t,k] for t in range(1, T+1) for i in range(I) ), index=2, priority=1)
    ## 制約条件
    

for i in range(I):    
    AG_PM.addConstr(Delta_ch[0,i]==0)
    AG_PM.addConstr(Delta_dch[0,i]==0)


        
    # 需給バランス制約
    for t in range(1,T+1):
        AG_PM.addConstr( P_PV[t,i]+P_PG[t,i] == P_dmd[t,i] + P_Ch[t,i]+ P_SG[t,i] - P_DCh[t,i] +P_HP[t,i])
        AG_PM.addConstr( P_pg[t,i] == P_PG[t,i]-P_SG[t,i])
        
    #買電売電同時禁止制約
    for t in range(1,T+1):
        AG_PM.addConstr(Delta_PG[t,i]<=1)
        
    for t in range(1,T+1):
        AG_PM.addConstr(P_PG[t,i]<= Delta_PG[t,i]*buy_MAX)
        AG_PM.addConstr( P_SG[t,i]<= (1-Delta_PG[t,i])*sell_MAX)
        
    #蓄電池充放電容量（kW容量）制約
    for t in range(1,T+1):
            AG_PM.addConstr(P_DCh[t,i] <= N_B_PCS * Delta_dch[t,i])
    for t in range(1,T+1):
            AG_PM.addConstr(P_Ch[t,i] <= N_B_PCS * Delta_ch[t,i])
    for t in range(1,T+1):
            AG_PM.addConstr(Delta_ch[t,i] + Delta_dch[t,i] <= 1.0)
                
    # 蓄電池蓄電容量（kWh容量）制約
    for t in range(1,T+1):
        AG_PM.addConstr(N_B*0.2 <= E_B[t,i])
        AG_PM.addConstr(E_B[t,i] <= N_B)
        
    #HP制約
    for t in range(1,T+1):
        
        AG_PM.addConstr(P_HP[t,i]==xhp * Delta_prod[t,i] +(H_prod[t,i]+H_ini[t,i])/ce/cop[t])
    for t in range(1,T+1):
        AG_PM.addConstr(t_ref*r1*cph*Delta_prod[t,i]<=H_prod[t,i]-H_prod_DW[t,i])
        #AG_PM.addConstr(t_ref*r1*cph*Delta_prod[t,i]<=H_prod[t,i])
        AG_PM.addConstr(H_prod[t,i]<=t_ref*cph*Delta_prod[t,i])
        AG_PM.addConstr(H_prod[t,i]+H_prod_UP[t,i]<=t_ref*cph)
        AG_PM.addConstr(H_ini[t,i]==r2*cph*Delta_S[t,i])
        #HPのバイナリ制約
    for t in range(1,T+1):
        AG_PM.addConstr(Delta_prod[t,i]-Delta_prod[t-1,i]==Delta_S[t,i]-Delta_F[t,i])
        AG_PM.addConstr(Delta_prod[t,i]<=1)
        AG_PM.addConstr(Delta_S[t,i]+Delta_F[t,i]<=1)
        #ON/OFFを繰り返さない
        AG_PM.addConstr(H_prod[t,i]>=cph*(Delta_prod[t,i]-Delta_S[t,i]))
        #貯湯量更新制約
        AG_PM.addConstr(H_tank[t,i]==H_tank[t-1,i]+H_prod[t,i]-H_dmd[t,i])
        #貯湯量容量制約
        AG_PM.addConstr(H_tank[t,i]+gp.quicksum(H_prod_UP[τ,i] for τ in range(0,t))<=cw*vlt*70)
        AG_PM.addConstr(0<=H_tank[t,i]-gp.quicksum(H_prod_DW[τ,i] for τ in range(0,t)))
        AG_PM.addConstr(H_tank_DW[t,i]==H_tank[t,i]+gp.quicksum(H_prod_UP[τ,i] for τ in range(0,t)))
        AG_PM.addConstr(H_tank_UP[t,i]==H_tank[t,i]-gp.quicksum(H_prod_DW[τ,i] for τ in range(0,t)))
        AG_PM.addConstr(P_HP_UP[t,i]==H_prod_UP[t,i]/ce/cop[t])
        AG_PM.addConstr(P_HP_DW[t,i]==H_prod_DW[t,i]/ce/cop[t])
        
    # 貯湯槽蓄熱量始端・終端制約
    AG_PM.addConstr(H_tank[0,i] == 0.5*cw*vlt*70)
    AG_PM.addConstr(H_tank[T,i] == 0.5*cw*vlt*70)
    
    # for t in range(1,T):
    #     AG_PM.addConstr(gp.quicksum(H_prod[τ,i] for τ in range(t,t+1))>=cph*Delta_S[t,i])
        
        
    for t in range(1,T+1):
        AG_PM.addConstr(E_B_DW[t,i]==t_ref*gp.quicksum(P_DW_Ch[k,i]+P_DW_DCh[k,i] for k in range(1,t+1)))
        AG_PM.addConstr(E_B_UP[t,i]==t_ref*gp.quicksum(P_UP_Ch[k,i]+P_UP_DCh[k,i] for k in range(1,t+1)))
        AG_PM.addConstr(P_UP[t,i]==P_UP_Ch[t,i]+P_UP_DCh[t,i]+P_HP_UP[t,i])
        AG_PM.addConstr(P_DW[t,i]==P_DW_Ch[t,i]+P_DW_DCh[t,i]+P_HP_DW[t,i])
        AG_PM.addConstr(P_DCh[t,i]+P_DW_DCh[t,i]<=Delta_dch[t,i]*N_B_PCS)
        AG_PM.addConstr(0<=P_DCh[t,i]-P_UP_DCh[t,i])
        AG_PM.addConstr(P_Ch[t,i]+P_UP_Ch[t,i]<=N_B_PCS*Delta_ch[t,i])
        AG_PM.addConstr(0<=P_Ch[t,i]-P_DW_Ch[t,i])
        
    # 蓄電池蓄電量始端・終端制約
    AG_PM.addConstr(E_B[0,i] == 0.5*N_B)
    AG_PM.addConstr(E_B[T,i] == 0.5*N_B)
    # 蓄電池蓄電量更新制約
    for t in range(1,T+1):
        AG_PM.addConstr(E_B[t,i] == E_B[t-1,i] + t_ref*eta_Bpcs*P_Ch[t,i] - t_ref*(1/eta_Bpcs)*P_DCh[t,i])
        
#電圧制約
    #上限違反
for t in range(1,T+1):
    for k in range(I): 
        AG_PM.addConstr(V_UL-V_d[t,k] == V_UN[t,k]-V_ULV[t,k])
#下限違反            
for t in range(1,T+1):
    for k in range(I):    
        AG_PM.addConstr(V_d[t,k]-V_LL == V_LN[t,k]-V_LLV[t,k])
#右辺の文字を0以上とする制約    
for t in range(1,T+1):
    for k in range(I):  
        AG_PM.addConstr( V_ULV[t,k]>= 0)
        AG_PM.addConstr( V_UN[t,k] >= 0)
        AG_PM.addConstr( V_LN[t,k] >= 0)
        AG_PM.addConstr( V_LLV[t,k]>= 0)
        
#電圧推定
for t in range(1,T+1): 
    for k in range(I):
        AG_PM.addConstr(V_dd[t,k]==linear_ap[k]*ap[t,k]+linear_aq[k]*aq[t,k]+linear_b[k])#電圧降下推定
            
for t in range(1,T+1):
    for k in range(I):
        if k==0:
            AG_PM.addConstr(V_d[t,k]==V_dd[t,k]+99.8)
            
        if k==1:
            AG_PM.addConstr(V_d[t,k]==V_dd[t,k]+V_d[t,0])
        if k==2:
            AG_PM.addConstr(V_d[t,k]==V_dd[t,k]+V_d[t,1])

for t in range(1,T+1):
         for k in range(I): 
                AG_PM.addConstr(ap[t,k]==gp.quicksum(P_pg[t,i] for i in range(k,I) ))
                AG_PM.addConstr(aq[t,k]==gp.quicksum(P_pg[t,i] for i in range(k,I) )*0.1 )
                
            # if k ==7:
            #     AG_PM.addConstr(ap[t,k]==(P_PG[4, t]+P_PG[5, t]+P_PG[6, t])*1.003)
            #     AG_PM.addConstr(aq[t,k]==((P_PG[4, t]+P_PG[5, t]+P_PG[6, t]))*0.1*1.004)
            
           
    
status = AG_PM.optimize()
print(status)

print('目的関数値=')
print(AG_PM.ObjVal)

print(f"計算時間: {AG_PM.Runtime:.2f} 秒")

if AG_PM.status == gp.GRB.OPTIMAL:
    # 各目的を再計算
    obj0_val = -sum(-(P_UP[t,i].X + P_DW[t,i].X) * price_balancing[t] 
                   + (spot_price[t] - buy_price[t]) * P_PG[t,i].X
                   for t in range(1, T+1) for i in range(I))

    obj1_val = sum(P_PG[t,i].X * buy_price[t] - P_SG[t,i].X * sell_price
                   for t in range(1, T+1) for i in range(I))

    # #obj2_val = sum(V_ULV[t,k].X + V_LLV[t,k].X
    #                + V_ULV_UP[t,k].X + V_ULV_DW[t,k].X
    #                + V_LLV_UP[t,k].X + V_LLV_DW[t,k].X
    #                for t in range(1, T+1) for k in I)  # ← i ではなく k ですよね？

    print("目的0:", obj0_val)
    print("目的1:", obj1_val)
    #print("目的2:", obj2_val)
benefit_df = pd.DataFrame()
for i in range(I):
    benefit_df['コマ']=[time[t] for t in range(1,T+1)]
    benefit_df['平均落札価格（TSO別）[円/kW・30分]']=[price_balancing[t] for t in range(1,T+1)]
    benefit_df['電気料金[JPY・30min/kW]']=[buy_price[t] for t in range(1,T+1)]
    benefit_df['蓄電量[kWh]']=[E_B[t,i].x for t in range(1,T+1)]
    benefit_df['充電量[kW・30分]']=[-P_Ch[t,i].X for t in range(1,T+1)]
    benefit_df['放電量[kW・30分]']=[P_DCh[t,i].X for t in range(1,T+1)]
    benefit_df['HP給湯機消費電力[kW・30分]']=[P_HP[t,i].X for t in range(1,T+1)]
    benefit_df['上げ調整力[kW]']=[P_UP[t,i].x for t in range(1,T+1)]
    benefit_df['下げ調整力[kW]']=[P_DW[t,i].X for t in range(1,T+1)]
    benefit_df['充電中上げ調整力[kW]']=[P_UP_Ch[t,i].x for t in range(1,T+1)]
    benefit_df['放電中上げ調整力[kW]']=[P_UP_DCh[t,i].x for t in range(1,T+1)]
    benefit_df['充電中下げ調整力[kW]']=[P_DW_Ch[t,i].x for t in range(1,T+1)]
    benefit_df['放電中下げ調整力[kW]']=[P_DW_DCh[t,i].x for t in range(1,T+1)]
    benefit_df['充電中変数']=[Delta_ch[t,i].x for t in range(1,T+1)]
    benefit_df['放電中変数'] = [Delta_dch[t,i].X for t in range(1,T+1)]
    benefit_df['電力系統からの買電電力[kW・30分]']=[P_PG[t,i].x for t in range(1,T+1)]
    benefit_df['電力系統への売電電力[kW・30分]']=[-P_SG[t,i].x for t in range(1,T+1)]
    benefit_df['売買変数']=[Delta_PG[t,i].x for t in range(1,T+1)]
    benefit_df['電力需要量[kW・30min]']=[P_dmd[t,i] for t in range(1,T+1)]
    benefit_df['太陽光発電出力[kW・30min]']=[P_PV[t,i] for t in range(1,T+1)]
    benefit_df['上げDRが発動した場合の蓄電量']=[E_B_UP_minus[t,i].x for t in range(1,T+1)]
    benefit_df['下げDRが発動した場合の蓄電量']=[E_B_DW_plus[t,i].x for t in range(1,T+1)]
    
    


    filename = 'kekka_{:}.csv'.format(i)
    benefit_df.to_csv("./OUTPUT/"+filename,encoding="shift_jis")
heat_pump_df = pd.DataFrame()
for i in range(I):
    heat_pump_df['HP給湯機消費電力[kW・30分]']=[P_HP[t,i].X for t in range(1,T+1)]
    heat_pump_df['熱需要[kW・30分]']=[H_dmd[t,i] for t in range(1,T+1)]
    heat_pump_df['熱製造量[MJ・30分]']=[H_prod[t,i].X for t in range(1,T+1)]
    heat_pump_df['上げDR熱製造量[MJ・30分]']=[H_prod_UP[t,i].X for t in range(1,T+1)]
    heat_pump_df['下げDR熱製造量[MJ・30分]']=[H_prod_DW[t,i].X for t in range(1,T+1)]
    heat_pump_df['貯湯量[MJ]']=[H_tank[t,i].X for t in range(1,T+1)]
    heat_pump_df['成績係数']=[cop[t] for t in range(1,T+1)]
    heat_pump_df['運転変数']=[Delta_prod[t,i].x for t in range(1,T+1)]
    heat_pump_df['運転開始変数']=[Delta_S[t,i].x for t in range(1,T+1)]
    heat_pump_df['運転終了変数']=[Delta_F[t,i].x for t in range(1,T+1)]
    heat_pump_df['電気料金[JPY・30min/kW]']=[buy_price[t] for t in range(1,T+1)]
    heat_pump_df['平均落札価格（TSO別）[円/kW・30分]']=[price_balancing[t] for t in range(1,T+1)]
    heat_pump_df['上げDRが働いた時の蓄熱量']=[H_tank_UP[t,i].x for t in range(1,T+1)]
    heat_pump_df['下げDRが働いた時の蓄熱量']=[H_tank_DW[t,i].x for t in range(1,T+1)]
    filename = 'heat_pump{:}.csv'.format(i)
    heat_pump_df.to_csv("./OUTPUT/"+filename,encoding="shift_jis")
    
    
voltage_df = pd.DataFrame()
voltage_df['bus1の電圧']=[V_d[t,0].X for t in range(1,T+1)]
voltage_df['bus2の電圧']=[V_d[t,1].X for t in range(1,T+1)]
voltage_df['bus3の電圧']=[V_d[t,2].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus1の電圧']=[V_d_DW[t,0].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus2の電圧']=[V_d_DW[t,1].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus3の電圧']=[V_d_DW[t,2].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus1の電圧']=[V_d_UP[t,0].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus2の電圧']=[V_d_UP[t,1].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus3の電圧']=[V_d_UP[t,2].X for t in range(1,T+1)]
voltage_df['電圧上限値']=[107 for t in range(1,T+1)]
voltage_df['電圧下限値']=[95 for t in range(1,T+1)]
# voltage_df['busAの電圧上限違反値']=[V_ULV[3,t].X for t in range(1,T+1)] 
voltage_df['bus1の電圧上限違反値']=[V_ULV[t,0].X for t in range(1,T+1)]
voltage_df['bus2の電圧上限違反値']=[V_ULV[t,1].X for t in range(1,T+1)]
voltage_df['bus3の電圧上限違反値']=[V_ULV[t,2].X for t in range(1,T+1)]
# voltage_df['busAの電圧下限違反値']=[V_LLV[3,t].X for t in range(1,T+1)]
voltage_df['bus1の電圧下限違反値']=[V_LLV[t,0].X for t in range(1,T+1)]
voltage_df['bus2の電圧下限違反値']=[V_LLV[t,1].X for t in range(1,T+1)]
voltage_df['bus3の電圧下限違反値']=[V_LLV[t,2].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus1の電圧上限違反値']=[V_ULV_DW[t,0].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus2の電圧上限違反値']=[V_ULV_DW[t,1].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus3の電圧上限違反値']=[V_ULV_DW[t,2].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus1の電圧下限違反値']=[V_LLV_DW[t,0].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus2の電圧下限違反値']=[V_LLV_DW[t,1].X for t in range(1,T+1)]
voltage_df['下げDRが発動した場合のbus3の電圧下限違反値']=[V_LLV_DW[t,2].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus1の電圧上限違反値']=[V_ULV_UP[t,0].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus2の電圧上限違反値']=[V_ULV_UP[t,1].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus3の電圧上限違反値']=[V_ULV_UP[t,2].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus1の電圧下限違反値']=[V_LLV_UP[t,0].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus2の電圧下限違反値']=[V_LLV_UP[t,1].X for t in range(1,T+1)]
voltage_df['上げDRが発動した場合のbus3の電圧下限違反値']=[V_LLV_UP[t,2].X for t in range(1,T+1)]
voltage_df['bus1の有効電力']=[ap[t,0].X for t in range(1,T+1)]
voltage_df['bus2の有効電力']=[ap[t,1].X for t in range(1,T+1)]
voltage_df['bus3の有効電力']=[ap[t,2].X for t in range(1,T+1)]
voltage_df['bus1の電圧降下']=[V_dd[t,0].X for t in range(1,T+1)]
voltage_df['bus2の電圧降下']=[V_dd[t,1].X for t in range(1,T+1)]
voltage_df['bus3の電圧降下']=[V_dd[t,2].X for t in range(1,T+1)]

filename = 'voltage.csv'
voltage_df.to_csv("./OUTPUT/"+filename,encoding="shift_jis")



#学会原稿の資料作りのために用意しましたP_UPは上げDR，P_DWは下げDRによって調整力を提供するという意味で付けています。上げ調整力とかとごっちゃにしないでね
for i in range(I):  
    Balancing_market_df = pd.DataFrame()
    Balancing_market_df['Time']=[time[t] for t in range(1,T+1)]
    Balancing_market_df['Balancing market price[yen/kW・30min]']=[price_balancing[t] for t in range(1,T+1)]
    Balancing_market_df['Upward balancing power']=[P_DW[t,i].x for t in range(1,T+1)]
    Balancing_market_df['Downward balancing power']=[P_UP[t,i].x for t in range(1,T+1)]
    filename = 'balancing_power{:}.xlsx'.format(i)
    Balancing_market_df.to_excel("./OUTPUT/"+filename)


    ESS_df = pd.DataFrame()
    ESS_df['reserve is not activated']=[E_B[t,i].x for t in range(1,T+1)]
    ESS_df['upward reserve is activated']=[E_B[t,i].x-E_B_DW[t,i].x for t in range(1,T+1)]
    ESS_df['Downward reserve is activated']=[E_B[t,i].x+E_B_UP[t,i].x for t in range(1,T+1)]
    ESS_df['Upper limit of battery capacity']=[N_B for t in range(1,T+1)]
    ESS_df['Lower limit of battery capacity']=[N_B*0.2 for t in range(1,T+1)]
    ESS_df['upward reserve']=[E_B_DW[t,i].x for t in range(1,T+1)]
    ESS_df['Downward reserve']=[E_B_UP[t,i].x for t in range(1,T+1)]
    ESS_df['蓄電池で提供した上げ調整力[kW]']=[P_DW_Ch[t,i].x+P_DW_DCh[t,i].x for t in range(1,T+1)]
    ESS_df['蓄電池で提供した下げ調整力[kW]']=[P_UP_Ch[t,i].x+P_UP_DCh[t,i].x for t in range(1,T+1)]
    
    filename = 'ESS{:}.xlsx'.format(i)
    ESS_df.to_excel("./OUTPUT/"+filename)
print("結果発表!!!!!!")