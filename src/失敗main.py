"""
ベンダーズ分解法による配電網制約を考慮したDR最適配分
単一引込線での実装（将来的に複数引込線への拡張を想定）
"""

import logging
import os
import time
import gurobipy as gp
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlibの日本語化設定
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP"]

# ロギングの詳細設定（デバッグ用）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class BendersDecomposition:
    """ベンダーズ分解法によるアグリゲータ最適化"""
    
    def __init__(self):
        """初期化：パラメータ設定"""
        logger.info("=" * 60)
        logger.info("ベンダーズ分解法の初期化開始")
        logger.info("=" * 60)
        
        # 基本パラメータ
        self.line = 4  # 電力系統の分岐数
        self.T = 48    # タイムスロット数
        self.H = 3     # 需要家の数
        
        # 下げDR期間
        self.DOWN_DR_START_HOUR = 13 * 2  # 13時
        self.DOWN_DR_END_HOUR = 16 * 2    # 16時
        
        # 変換効率
        self.ETA_PV_PCS = 0.95
        self.ETA_B_PCS = 0.95
        
        # 容量制約
        self.GRANULARITY_CONVERSION = 30 / 60
        self.N_B_PCS = 1.5  # 蓄電池kW容量[kW]
        self.N_B = 6.3      # 蓄電池kWh容量[kWh]
        self.SELL_PRICE = 2.0  # 売電単価[JPY/kW]
        self.BUY_MAX = 5.0     # 買電上限[kW]
        self.SELL_MAX = 5.0    # 売電上限[kW]
        self.IN_MAX = self.BUY_MAX * self.H    # 需要家への供給電力上限
        self.OUT_MAX = self.SELL_MAX * self.H  # 需要家からの放出電力上限
        
        # HP給湯機パラメータ
        self.r1 = 0.05
        self.r2 = 0.1
        self.cw = 0.0042
        self.ce = 3.6
        self.vlt = 370
        self.cph = 16.2
        self.xhp = 0.013
        self.tank_half_capacity = 0.5 * self.cw * self.vlt * 70
        
        # 電圧制約
        self.V_UL = 101.0 + 6.0  # 上限107V
        self.V_LL = 101.0 - 6.0  # 下限95V
        self.S_LINE = 101 * 20    # 送電線定格電力
        
        # ベンダーズ分解法パラメータ
        self.epsilon = 0.01     # 収束判定基準（1%）
        self.max_iter = 100     # 最大反復回数
        self.UB = float('inf')  # 上界
        self.LB = float('-inf') # 下界
        
        # 履歴記録用
        self.lb_history = []
        self.ub_history = []
        self.gap_history = []
        self.time_history = []
        self.iteration_count = 0
        
        # データ格納用辞書
        self.data = {}
        
        logger.info(f"需要家数: {self.H}, タイムスロット数: {self.T}")
        logger.info(f"下げDR期間: {self.DOWN_DR_START_HOUR//2}:00-{self.DOWN_DR_END_HOUR//2}:00")
        logger.info(f"収束判定基準: {self.epsilon*100}%")
        
    def load_all_data(self):
        """データの読み込み"""
        logger.info("\nデータ読み込み開始...")
        
        try:
            # 電力需要データ
            dmd_data = pd.read_csv("../data/input/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", encoding="shift_jis")
            
            # 価格データ
            self.buy_price = {}
            buy_price_data = pd.read_csv("../data/input/buy_energy30.csv", encoding="shift_jis")
            for t in range(1, self.T + 1):
                self.buy_price[t] = float(buy_price_data.iat[t-1, 2])
            
            # スポット市場価格
            self.spot_price = {}
            spot_price_data = pd.read_csv("../data/input/spot_market_price_30min.csv", encoding="shift_jis")
            for t in range(1, self.T + 1):
                self.spot_price[t] = float(spot_price_data.iat[t, 2])
            
            # 太陽光出力
            self.p_pv = {}
            pv_data = pd.read_csv("../data/input/pv_output.csv")
            for t in range(1, self.T + 1):
                pv_value = float(pv_data.iat[t + 30*self.T, 2]) / 1000
                for i in range(self.H):
                    self.p_pv[t, i] = pv_value
            
            # 電力需要
            self.p_dmd = {}
            for t in range(self.T + 1):
                for i in range(self.H):
                    self.p_dmd[t, i] = float(dmd_data.iat[t-1, 10])
            
            # 熱需要
            self.H_dmd = {}
            heat_dmd_data = pd.read_csv("../data/input/heat_demand1y.csv", encoding="shift_jis")
            for t in range(1, self.T + 1):
                for i in range(self.H):
                    self.H_dmd[t, i] = float(heat_dmd_data.iat[t-1+self.T, 2])
            
            # 温度データとCOP計算
            self.cop = {}
            self.tmf = {}
            temperature_water_data = pd.read_csv("../data/input/water_temperature_30min.csv", encoding="shift_jis")
            temperature_outside_data = pd.read_csv("../data/input/temperature1y.csv", encoding="shift_jis")
            
            for t in range(1, self.T + 1):
                self.tmf[t] = float(temperature_water_data.iat[t-1, 2])
                tma = float(temperature_outside_data.iat[t, 2])
                
                if tma >= 5:
                    k = 1
                elif tma > 2:
                    k = tma/30 + 0.8333
                else:
                    k = 0.9
                self.cop[t] = k * (0.175*tma - 0.1322*self.tmf[t] + 4.076)
            
            # 電圧推定パラメータ
            self.linear_ap = {}
            self.linear_aq = {}
            self.linear_b = {}
            voltage_estimate = pd.read_csv("../data/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis")
            for i in range(self.H):
                self.linear_ap[i] = voltage_estimate.iat[0, 3*i]
                self.linear_aq[i] = voltage_estimate.iat[0, 3*i+1]
                self.linear_b[i] = voltage_estimate.iat[0, 3*i+2]
            
            # 時刻データ
            self.time = {t: dmd_data.iat[t-1, 0] for t in range(1, self.T+1)}
            
            logger.info("データ読み込み完了")
            logger.info(f"  - 買電価格範囲: {min(self.buy_price.values()):.1f} - {max(self.buy_price.values()):.1f} JPY/kW")
            logger.info(f"  - 平均PV出力: {np.mean(list(self.p_pv.values())):.2f} kW")
            logger.info(f"  - 平均電力需要: {np.mean([self.p_dmd[t,0] for t in range(1,self.T+1)]):.2f} kW")
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def create_master_problem(self):
        """マスター問題の作成"""
        logger.info("\n【マスター問題の作成】")
        
        try:
            self.master = gp.Model("Benders_Master")
            self.master.setParam('OutputFlag', 0)  # Gurobi出力を抑制
            
            # 決定変数
            self.P_PG_master = self.master.addVars(
                range(1, self.T+1), 
                vtype=gp.GRB.CONTINUOUS, 
                name='P_PG_master',
                lb=0.0,
                ub=self.BUY_MAX * self.H
            )
            
            self.P_SG_master = self.master.addVars(
                range(1, self.T+1),
                vtype=gp.GRB.CONTINUOUS,
                name='P_SG_master',
                lb=0.0,
                ub=self.SELL_MAX * self.H
            )
            
            # 買電売電フラグ
            self.Delta_pg_master = self.master.addVars(
                range(self.T+1),
                vtype=gp.GRB.BINARY,
                name='Delta_pg_master'
            )
            
            self.Delta_sg_master = self.master.addVars(
                range(self.T+1),
                vtype=gp.GRB.BINARY,
                name='Delta_sg_master'
            )
            
            # サブ問題の近似値
            self.theta = self.master.addVar(
                vtype=gp.GRB.CONTINUOUS,
                name='theta',
                lb=-gp.GRB.INFINITY
            )
            
            self.master.update()
            
            # 制約：買電売電同時禁止
            for t in range(1, self.T+1):
                self.master.addConstr(
                    self.P_PG_master[t] <= self.Delta_pg_master[t] * self.BUY_MAX * self.H,
                    name=f'buy_limit_{t}'
                )
                self.master.addConstr(
                    self.P_SG_master[t] <= self.Delta_sg_master[t] * self.SELL_MAX * self.H,
                    name=f'sell_limit_{t}'
                )
                self.master.addConstr(
                    self.Delta_pg_master[t] + self.Delta_sg_master[t] <= 1,
                    name=f'simultaneous_prohibition_{t}'
                )
            
            # 目的関数：下げDR期間の買電最小化 + 電力コスト
            self.master.setObjective(
                gp.quicksum(self.P_PG_master[t] 
                          for t in range(self.DOWN_DR_START_HOUR+1, self.DOWN_DR_END_HOUR+1)) +
                self.theta,
                gp.GRB.MINIMIZE
            )
            
            # 初期値設定（上限と下限の中間）
            for t in range(1, self.T+1):
                self.P_PG_master[t].start = self.BUY_MAX * self.H / 2
                self.P_SG_master[t].start = 0  # 初期は売電なし
            self.theta.start = 0
            
            logger.info(f"  マスター問題変数数: {len(self.master.getVars())}")
            logger.info(f"  マスター問題制約数: {len(self.master.getConstrs())}")
            
        except Exception as e:
            logger.error(f"マスター問題作成エラー: {e}")
            raise
    
    def create_sub_problem(self, P_PG_fixed, P_SG_fixed):
        """サブ問題の作成（P_PG, P_SGを固定）"""
        logger.info("\n【サブ問題の作成】")
        logger.debug(f"  固定値 - 買電: {sum(P_PG_fixed.values()):.2f} kW, 売電: {sum(P_SG_fixed.values()):.2f} kW")
        
        try:
            sub = gp.Model("Benders_Sub")
            sub.setParam('OutputFlag', 0)
            
            # ========== 決定変数の定義 ==========
            # 需要家への供給/放出電力
            P_in = sub.addVars(range(1, self.T+1), range(self.H), 
                              vtype=gp.GRB.CONTINUOUS, name='P_in',
                              lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            P_out = sub.addVars(range(1, self.T+1), range(self.H),
                               vtype=gp.GRB.CONTINUOUS, name='P_out',
                               lb=0.0, ub=gp.GRB.INFINITY)
            
            # 需要家の買電/売電
            P_pg = sub.addVars(range(1, self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='P_pg',
                              lb=0.0, ub=gp.GRB.INFINITY)
            P_sg = sub.addVars(range(1, self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='P_sg',
                              lb=0.0, ub=gp.GRB.INFINITY)
            
            # 需要家間電力融通
            P_fp = sub.addVars(range(self.H), range(self.H), range(1, self.T+1),
                              vtype=gp.GRB.CONTINUOUS, name='P_fp',
                              lb=0.0, ub=gp.GRB.INFINITY)
            
            # 買電売電フラグ
            Delta_in = sub.addVars(range(self.T+1), range(self.H),
                                  vtype=gp.GRB.BINARY, name='Delta_in')
            Delta_out = sub.addVars(range(self.T+1), range(self.H),
                                   vtype=gp.GRB.BINARY, name='Delta_out')
            
            # 蓄電池変数
            P_Ch = sub.addVars(range(self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='P_Ch',
                              lb=0.0, ub=gp.GRB.INFINITY)
            P_DCh = sub.addVars(range(self.T+1), range(self.H),
                               vtype=gp.GRB.CONTINUOUS, name='P_DCh',
                               lb=0.0, ub=gp.GRB.INFINITY)
            E_B = sub.addVars(range(self.T+1), range(self.H),
                             vtype=gp.GRB.CONTINUOUS, name='E_B',
                             lb=0.0, ub=gp.GRB.INFINITY)
            Delta_ch = sub.addVars(range(self.T+1), range(self.H),
                                  vtype=gp.GRB.BINARY, name='Delta_ch')
            Delta_dch = sub.addVars(range(self.T+1), range(self.H),
                                   vtype=gp.GRB.BINARY, name='Delta_dch')
            
            # HP給湯機変数
            P_HP = sub.addVars(range(1, self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='P_HP',
                              lb=0.0, ub=gp.GRB.INFINITY)
            H_ini = sub.addVars(range(self.T+1), range(self.H),
                               vtype=gp.GRB.CONTINUOUS, name='H_ini',
                               lb=0.0, ub=gp.GRB.INFINITY)
            H_prod = sub.addVars(range(self.T+1), range(self.H),
                                vtype=gp.GRB.CONTINUOUS, name='H_prod',
                                lb=0.0, ub=gp.GRB.INFINITY)
            H_tank = sub.addVars(range(self.T+1), range(self.H),
                                vtype=gp.GRB.CONTINUOUS, name='H_tank',
                                lb=0.0, ub=gp.GRB.INFINITY)
            Delta_prod = sub.addVars(range(self.T+1), range(self.H),
                                    vtype=gp.GRB.BINARY, name='Delta_prod')
            Delta_S = sub.addVars(range(self.T+1), range(self.H),
                                 vtype=gp.GRB.BINARY, name='Delta_S')
            Delta_F = sub.addVars(range(self.T+1), range(self.H),
                                 vtype=gp.GRB.BINARY, name='Delta_F')
            
            # ========== 電圧関連変数（原問題と同じ） ==========
            V_d = sub.addVars(range(1, self.T+1), range(self.H),
                             vtype=gp.GRB.CONTINUOUS, name='電圧',
                             lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            V_dd = sub.addVars(range(self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='電圧変化',
                              lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            ap = sub.addVars(range(1, self.T+1), range(self.H),
                            vtype=gp.GRB.CONTINUOUS, name='有効電力',
                            lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            aq = sub.addVars(range(1, self.T+1), range(self.H),
                            vtype=gp.GRB.CONTINUOUS, name='無効電力',
                            lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            V_UN = sub.addVars(range(1, self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='上限セーフ値',
                              lb=0.0, ub=gp.GRB.INFINITY)
            V_LN = sub.addVars(range(1, self.T+1), range(self.H),
                              vtype=gp.GRB.CONTINUOUS, name='下限セーフ値',
                              lb=0.0, ub=gp.GRB.INFINITY)
            V_ULV = sub.addVars(range(1, self.T+1), range(self.H),
                               vtype=gp.GRB.CONTINUOUS, name='上限違反値',
                               lb=0.0, ub=gp.GRB.INFINITY)
            V_LLV = sub.addVars(range(1, self.T+1), range(self.H),
                               vtype=gp.GRB.CONTINUOUS, name='下限違反値',
                               lb=0.0, ub=gp.GRB.INFINITY)
            
            sub.update()
            
            # ========== 制約条件 ==========
            
            # アグリゲータレベル需給バランス（固定値使用）
            self.agg_balance_constrs = {}
            for t in range(1, self.T+1):
                # 買電制約
                self.agg_balance_constrs[f'buy_{t}'] = sub.addConstr(
                    gp.quicksum(P_pg[t, i] for i in range(self.H)) == P_PG_fixed[t],
                    name=f'agg_buy_balance_{t}'
                )
                # 売電制約
                self.agg_balance_constrs[f'sell_{t}'] = sub.addConstr(
                    gp.quicksum(P_sg[t, i] for i in range(self.H)) == P_SG_fixed[t],
                    name=f'agg_sell_balance_{t}'
                )
            
            # 需給バランス制約
            for t in range(1, self.T+1):
                # アグリゲータレベル
                sub.addConstr(
                    gp.quicksum(P_sg[t, i] for i in range(self.H)) + 
                    gp.quicksum(P_in[t, i] for i in range(self.H)) ==
                    gp.quicksum(P_pg[t, i] for i in range(self.H)) + 
                    gp.quicksum(P_out[t, i] for i in range(self.H)),
                    name=f'agg_balance_{t}'
                )
                
                # 各需要家
                for i in range(self.H):
                    sub.addConstr(
                        self.p_pv[t, i] + P_in[t, i] + P_DCh[t, i] ==
                        self.p_dmd[t, i] + P_Ch[t, i] + P_out[t, i] + P_HP[t, i],
                        name=f'house_balance_{t}_{i}'
                    )
                    
                    sub.addConstr(
                        P_in[t, i] == P_pg[t, i] + 
                        gp.quicksum(P_fp[j, i, t] for j in range(self.H) if j != i),
                        name=f'P_in_def_{t}_{i}'
                    )
                    
                    sub.addConstr(
                        P_out[t, i] == P_sg[t, i] + 
                        gp.quicksum(P_fp[i, j, t] for j in range(self.H) if j != i),
                        name=f'P_out_def_{t}_{i}'
                    )
            
            # 買電売電同時禁止
            for t in range(1, self.T+1):
                for i in range(self.H):
                    sub.addConstr(P_pg[t, i] <= Delta_in[t, i] * self.IN_MAX,
                                 name=f'pg_limit_{t}_{i}')
                    sub.addConstr(P_sg[t, i] <= Delta_out[t, i] * self.OUT_MAX,
                                 name=f'sg_limit_{t}_{i}')
                    sub.addConstr(Delta_in[t, i] + Delta_out[t, i] <= 1,
                                 name=f'sim_prohibit_{t}_{i}')
            
            # HP制約
            for i in range(self.H):
                for t in range(1, self.T+1):
                    sub.addConstr(
                        P_HP[t,i] == self.xhp * Delta_prod[t,i] + 
                        (H_prod[t,i] + H_ini[t,i]) / self.ce / self.cop[t],
                        name=f'HP_power_{t}_{i}'
                    )
                    sub.addConstr(
                        H_prod[t,i] >= self.GRANULARITY_CONVERSION * self.r1 * 
                        self.cph * Delta_prod[t,i],
                        name=f'HP_min_prod_{t}_{i}'
                    )
                    sub.addConstr(
                        H_prod[t,i] <= self.GRANULARITY_CONVERSION * 
                        self.cph * Delta_prod[t,i],
                        name=f'HP_max_prod_{t}_{i}'
                    )
                    sub.addConstr(
                        H_ini[t,i] == self.r2 * self.cph * Delta_S[t,i],
                        name=f'HP_ini_{t}_{i}'
                    )
                    sub.addConstr(
                        Delta_prod[t,i] - Delta_prod[t-1,i] == Delta_S[t,i] - Delta_F[t,i],
                        name=f'HP_state_{t}_{i}'
                    )
                    sub.addConstr(Delta_S[t,i] + Delta_F[t,i] <= 1,
                                 name=f'HP_sf_{t}_{i}')
                    sub.addConstr(Delta_S[t,i] <= Delta_prod[t,i],
                                 name=f'HP_start_{t}_{i}')
                    sub.addConstr(
                        H_prod[t,i] >= self.cph * (Delta_prod[t,i] - Delta_S[t,i]),
                        name=f'HP_cont_{t}_{i}'
                    )
                    sub.addConstr(
                        H_tank[t,i] == H_tank[t-1,i] + H_prod[t,i] - self.H_dmd[t,i],
                        name=f'tank_balance_{t}_{i}'
                    )
                    sub.addConstr(H_tank[t,i] <= self.cw * self.vlt * 70,
                                 name=f'tank_max_{t}_{i}')
                    sub.addConstr(H_tank[t,i] >= 0.2 * (self.cw * self.vlt * 70),
                                 name=f'tank_min_{t}_{i}')
                
                # HP初期状態
                sub.addConstr(Delta_prod[0,i] == 0, name=f'HP_init_prod_{i}')
                sub.addConstr(Delta_S[0,i] == 0, name=f'HP_init_S_{i}')
                sub.addConstr(Delta_F[0,i] == 0, name=f'HP_init_F_{i}')
                sub.addConstr(H_tank[0,i] == self.tank_half_capacity,
                             name=f'tank_init_{i}')
                sub.addConstr(H_tank[self.T,i] == self.tank_half_capacity,
                             name=f'tank_final_{i}')
            
            # 蓄電池制約
            for i in range(self.H):
                for t in range(1, self.T+1):
                    sub.addConstr(P_Ch[t,i] <= self.N_B_PCS * Delta_ch[t,i],
                                 name=f'ch_limit_{t}_{i}')
                    sub.addConstr(P_DCh[t,i] <= self.N_B_PCS * Delta_dch[t,i],
                                 name=f'dch_limit_{t}_{i}')
                    sub.addConstr(Delta_ch[t,i] + Delta_dch[t,i] <= 1.0,
                                 name=f'ch_dch_{t}_{i}')
                    sub.addConstr(E_B[t,i] >= self.N_B * 0.2,
                                 name=f'soc_min_{t}_{i}')
                    sub.addConstr(E_B[t,i] <= self.N_B,
                                 name=f'soc_max_{t}_{i}')
                    sub.addConstr(
                        E_B[t,i] == E_B[t-1,i] + 
                        self.GRANULARITY_CONVERSION * self.ETA_B_PCS * P_Ch[t,i] -
                        self.GRANULARITY_CONVERSION * (1/self.ETA_B_PCS) * P_DCh[t,i],
                        name=f'soc_update_{t}_{i}'
                    )
                
                # 蓄電池初期状態
                sub.addConstr(E_B[0,i] == 0.5 * self.N_B, name=f'soc_init_{i}')
                sub.addConstr(E_B[self.T,i] == 0.5 * self.N_B, name=f'soc_final_{i}')
                sub.addConstr(Delta_ch[0,i] == 0, name=f'ch_init_{i}')
                sub.addConstr(Delta_dch[0,i] == 0, name=f'dch_init_{i}')
            
            # 電圧制約
            for t in range(1, self.T+1):
                for k in range(self.H):
                    # 電圧上下限
                    sub.addConstr(self.V_UL - V_d[t,k] == V_UN[t,k],
                                 name=f'V_upper_{t}_{k}')
                    sub.addConstr(V_d[t,k] - self.V_LL == V_LN[t,k],
                                 name=f'V_lower_{t}_{k}')
                    
                    # 電圧降下推定
                    sub.addConstr(
                        V_dd[t,k] == self.linear_ap[k] * ap[t,k] + 
                        self.linear_aq[k] * aq[t,k] + self.linear_b[k],
                        name=f'V_drop_{t}_{k}'
                    )
                    
                    # 電圧計算
                    if k == 0:
                        sub.addConstr(V_d[t,k] == V_dd[t,k] + 99.8,
                                     name=f'V_calc_{t}_{k}')
                    else:
                        sub.addConstr(V_d[t,k] == V_dd[t,k] + V_d[t,k-1],
                                     name=f'V_calc_{t}_{k}')
                    
                    # 有効/無効電力
                    sub.addConstr(
                        ap[t,k] == gp.quicksum(P_in[t,i] for i in range(k, self.H)),
                        name=f'ap_{t}_{k}'
                    )
                    sub.addConstr(
                        aq[t,k] == gp.quicksum(P_in[t,i] for i in range(k, self.H)) * 0.1,
                        name=f'aq_{t}_{k}'
                    )
                    
                    # 送電線容量（二次錐制約）
                    sub.addQConstr(
                        ap[t,k]*ap[t,k] + aq[t,k]*aq[t,k] <= self.S_LINE * self.S_LINE,
                        name=f'line_capacity_{t}_{k}'
                    )
            
            # 目的関数：電力料金コスト
            sub.setObjective(
                gp.quicksum(self.buy_price[t] * P_pg[t,i] - 
                          self.SELL_PRICE * P_sg[t,i]
                          for t in range(1, self.T+1)
                          for i in range(self.H)),
                gp.GRB.MINIMIZE
            )
            
            logger.info(f"  サブ問題変数数: {len(sub.getVars())}")
            logger.info(f"  サブ問題制約数: {len(sub.getConstrs())}")
            
            return sub
            
        except Exception as e:
            logger.error(f"サブ問題作成エラー: {e}")
            logger.error(f"エラー詳細: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def solve_sub_problem(self, P_PG_fixed, P_SG_fixed):
        """サブ問題を解く"""
        try:
            sub = self.create_sub_problem(P_PG_fixed, P_SG_fixed)
            sub.optimize()
            
            if sub.Status == gp.GRB.OPTIMAL:
                # 双対変数の取得（アグリゲータ需給バランス制約）
                dual_PG = {}
                dual_SG = {}
                for t in range(1, self.T+1):
                    dual_PG[t] = self.agg_balance_constrs[f'buy_{t}'].Pi
                    dual_SG[t] = self.agg_balance_constrs[f'sell_{t}'].Pi
                
                return sub.ObjVal, dual_PG, dual_SG, True
            
            elif sub.Status == gp.GRB.INFEASIBLE:
                logger.warning("サブ問題が実行不可能")
                # 実行可能性カット用の極値レイを取得
                sub.computeIIS()
                return None, None, None, False
            
            else:
                logger.error(f"サブ問題の状態: {sub.Status}")
                return None, None, None, False
                
        except Exception as e:
            logger.error(f"サブ問題求解エラー: {e}")
            raise
    
    def add_optimality_cut(self, dual_PG, dual_SG, sub_obj, P_PG_current, P_SG_current):
        """最適性カットの追加"""
        cut_expr = sub_obj
        
        for t in range(1, self.T+1):
            cut_expr += dual_PG[t] * (self.P_PG_master[t] - P_PG_current[t])
            cut_expr += dual_SG[t] * (self.P_SG_master[t] - P_SG_current[t])
        
        self.master.addConstr(
            self.theta >= cut_expr,
            name=f'optimality_cut_{self.iteration_count}'
        )
        
        logger.debug(f"  最適性カット追加 (反復 {self.iteration_count})")
    
    def solve(self):
        """ベンダーズ分解法のメインループ"""
        logger.info("\n" + "="*60)
        logger.info("ベンダーズ分解法の実行開始")
        logger.info("="*60)
        
        start_time = time.time()
        
        # マスター問題の作成
        self.create_master_problem()
        
        for iteration in range(self.max_iter):
            self.iteration_count = iteration + 1
            iter_start_time = time.time()
            
            logger.info(f"\n--- 反復 {self.iteration_count} ---")
            
            # ステップ1: マスター問題を解く
            try:
                self.master.optimize()
                if self.master.Status != gp.GRB.OPTIMAL:
                    logger.error(f"マスター問題が最適でない: Status={self.master.Status}")
                    break
                    
                # マスター問題の解を取得
                P_PG_current = {t: self.P_PG_master[t].X for t in range(1, self.T+1)}
                P_SG_current = {t: self.P_SG_master[t].X for t in range(1, self.T+1)}
                master_obj = self.master.ObjVal
                
                # 下界の更新
                self.LB = master_obj
                
                logger.info(f"  マスター問題: 目的関数値 = {master_obj:.2f}")
                logger.debug(f"    買電合計: {sum(P_PG_current.values()):.2f} kW")
                logger.debug(f"    売電合計: {sum(P_SG_current.values()):.2f} kW")
                logger.debug(f"    θ = {self.theta.X:.2f}")
                
            except Exception as e:
                logger.error(f"マスター問題求解エラー: {e}")
                break
            
            # ステップ2: サブ問題を解く
            sub_obj, dual_PG, dual_SG, is_feasible = self.solve_sub_problem(
                P_PG_current, P_SG_current
            )
            
            if not is_feasible:
                logger.error("サブ問題が実行不可能 - 実行可能性カットが必要")
                # 実装簡略化のため、ここでは停止
                break
            
            # 上界の更新
            # 下げDR期間の買電量を計算
            dr_cost = sum(P_PG_current[t] 
                         for t in range(self.DOWN_DR_START_HOUR+1, self.DOWN_DR_END_HOUR+1))
            current_obj = dr_cost + sub_obj
            
            if current_obj < self.UB:
                self.UB = current_obj
                self.best_P_PG = P_PG_current.copy()
                self.best_P_SG = P_SG_current.copy()
            
            logger.info(f"  サブ問題: 目的関数値 = {sub_obj:.2f}")
            logger.info(f"  現在の上界 (UB) = {self.UB:.2f}")
            logger.info(f"  現在の下界 (LB) = {self.LB:.2f}")
            
            # ステップ3: 収束判定
            if self.UB != float('inf'):
                gap = abs(self.UB - self.LB) / abs(self.UB)
                logger.info(f"  収束ギャップ = {gap*100:.2f}%")
                
                # 履歴の記録
                self.lb_history.append(self.LB)
                self.ub_history.append(self.UB)
                self.gap_history.append(gap)
                self.time_history.append(time.time() - start_time)
                
                if gap < self.epsilon:
                    logger.info("\n★ 収束条件を満たしました！")
                    break
            else:
                logger.info("  収束ギャップ = N/A (上界が無限大)")
            
            # ステップ4: カットの追加
            self.add_optimality_cut(dual_PG, dual_SG, sub_obj, 
                                   P_PG_current, P_SG_current)
            
            iter_time = time.time() - iter_start_time
            logger.info(f"  反復時間: {iter_time:.2f} 秒")
        
        # 最終結果
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("最適化完了")
        logger.info(f"  総反復回数: {self.iteration_count}")
        logger.info(f"  総計算時間: {total_time:.2f} 秒")
        logger.info(f"  最終上界: {self.UB:.2f}")
        logger.info(f"  最終下界: {self.LB:.2f}")
        if self.UB != float('inf'):
            final_gap = abs(self.UB - self.LB) / abs(self.UB)
            logger.info(f"  最終ギャップ: {final_gap*100:.2f}%")
        logger.info("="*60)
        
        # 最適解でサブ問題を再度解いて詳細解を取得
        if hasattr(self, 'best_P_PG'):
            self.final_sub = self.create_sub_problem(self.best_P_PG, self.best_P_SG)
            self.final_sub.optimize()
        
        return self.UB
    
    def save_results(self):
        """結果の保存"""
        logger.info("\n結果の保存中...")
        
        output_dir = "../data/output/benders"
        os.makedirs(output_dir, exist_ok=True)
        
        # 収束履歴の保存
        convergence_df = pd.DataFrame({
            '反復': range(1, len(self.lb_history)+1),
            '下界': self.lb_history,
            '上界': self.ub_history,
            'ギャップ': self.gap_history,
            '計算時間': self.time_history
        })
        convergence_df.to_csv(
            os.path.join(output_dir, "convergence.csv"),
            encoding="shift_jis", 
            index=False
        )
        
        # 最終解が存在する場合のみ詳細結果を保存
        if hasattr(self, 'final_sub') and self.final_sub.Status == gp.GRB.OPTIMAL:
            # 各需要家の結果を保存（原問題と同形式）
            for i in range(self.H):
                house_data = []
                for t in range(1, self.T+1):
                    # 変数値の取得
                    P_pg_val = self.final_sub.getVarByName(f'P_pg[{t},{i}]').X
                    P_sg_val = self.final_sub.getVarByName(f'P_sg[{t},{i}]').X
                    P_in_val = self.final_sub.getVarByName(f'P_in[{t},{i}]').X
                    P_out_val = self.final_sub.getVarByName(f'P_out[{t},{i}]').X
                    E_B_val = self.final_sub.getVarByName(f'E_B[{t},{i}]').X
                    P_Ch_val = self.final_sub.getVarByName(f'P_Ch[{t},{i}]').X
                    P_DCh_val = self.final_sub.getVarByName(f'P_DCh[{t},{i}]').X
                    P_HP_val = self.final_sub.getVarByName(f'P_HP[{t},{i}]').X
                    
                    house_data.append({
                        'コマ': self.time[t],
                        '買電電力[kW・30分]': P_pg_val,
                        '売電電力[kW・30分]': -P_sg_val,
                        '供給電力[kW・30分]': P_in_val,
                        '放出電力[kW・30分]': -P_out_val,
                        '蓄電量[kWh]': E_B_val,
                        '充電量[kW・30分]': -P_Ch_val,
                        '放電量[kW・30分]': P_DCh_val,
                        'HP消費電力[kW・30分]': P_HP_val,
                        '電力需要[kW・30分]': self.p_dmd[t,i],
                        '太陽光発電[kW・30分]': self.p_pv[t,i]
                    })
                
                house_df = pd.DataFrame(house_data)
                house_df.to_csv(
                    os.path.join(output_dir, f"0.house_{i}.csv"),
                    encoding="shift_jis",
                    index=False
                )
        
        logger.info(f"結果を {output_dir} に保存しました")
    
    def plot_convergence(self):
        """収束過程のプロット"""
        if len(self.lb_history) == 0:
            logger.warning("プロット用のデータがありません")
            return
        
        output_dir = "../data/output/benders"
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 上界・下界の推移
        iterations = range(1, len(self.lb_history)+1)
        ax1.plot(iterations, self.ub_history, 'b-', label='上界 (UB)', linewidth=2)
        ax1.plot(iterations, self.lb_history, 'r-', label='下界 (LB)', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('目的関数値')
        ax1.set_title('ベンダーズ分解法の収束過程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ギャップの推移
        ax2.plot(iterations, [g*100 for g in self.gap_history], 'g-', linewidth=2)
        ax2.axhline(y=self.epsilon*100, color='r', linestyle='--', 
                   label=f'収束基準 ({self.epsilon*100}%)')
        ax2.set_xlabel('反復回数')
        ax2.set_ylabel('ギャップ (%)')
        ax2.set_title('最適性ギャップの推移')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence.png'), dpi=200)
        plt.close()
        
        logger.info(f"収束グラフを保存しました")


def main():
    """メイン実行関数"""
    try:
        # ベンダーズ分解法のインスタンス作成
        benders = BendersDecomposition()
        
        # データの読み込み
        benders.load_all_data()
        
        # 最適化の実行
        optimal_value = benders.solve()
        
        # 結果の保存
        benders.save_results()
        
        # 収束過程のプロット
        benders.plot_convergence()
        
        logger.info("\n処理が正常に完了しました")
        
    except Exception as e:
        logger.error(f"\n致命的エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()