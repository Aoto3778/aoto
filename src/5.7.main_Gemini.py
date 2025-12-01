"""
Distributed Optimization using Generalized Benders Decomposition (GBD)
配電網のアグリゲータ（Master）と需要家（Slave）の分散最適化

Ref: User provided research summary and centralized code.
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import logging
import time
import os
from typing import Dict, List, Tuple

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GBDConfig:
    """共通パラメータ設定"""
    def __init__(self):
        self.T = 48
        self.H = 3
        self.V_BASE = 99.8
        self.V_UL = 107.0
        self.V_LL = 95.0
        self.S_LINE = 101 * 20
        self.N_B = 6.3
        self.N_B_PCS = 1.5
        self.ETA_B = 0.95
        self.VLT = 370.0
        self.CW = 0.0042
        self.CE = 3.6
        self.CPH = 16.2
        self.XHP = 0.013
        self.R1 = 0.05
        self.R2 = 0.1
        self.BUY_MAX = 5.0
        self.SELL_MAX = 5.0
        self.SELL_PRICE = 2.0
        self.DR_START = 26
        self.DR_END = 32
        self.EPSILON = 1e-4 # 許容誤差

class DataLoader:
    """データ読み込みクラス（全体最適化コードから流用）"""
    def __init__(self, config):
        self.cfg = config
        self.p_pv = {}
        self.p_dmd = {}
        self.buy_price = {}
        self.heat_dmd = {}
        self.cop = {}
        self.linear_ap = {}
        self.linear_aq = {}
        self.linear_b = {}
        self.load_data()

    def load_data(self):
        logger.info("データ読み込み開始...")
        try:
            # パスは環境に合わせて調整してください
            base_path = "../data/input"
            dmd_data = pd.read_csv(f"{base_path}/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", encoding="shift_jis")
            buy_price_data = pd.read_csv(f"{base_path}/buy_energy30.csv", encoding="shift_jis")
            spot_price_data = pd.read_csv(f"{base_path}/spot_market_price_30min.csv", encoding="shift_jis")
            pv_data = pd.read_csv(f"{base_path}/pv_output.csv")
            heat_dmd_data = pd.read_csv(f"{base_path}/heat_demand1y.csv", encoding="shift_jis")
            temperature_outside_data = pd.read_csv(f"{base_path}/temperature1y.csv", encoding="shift_jis")
            temperature_water_data = pd.read_csv(f"{base_path}/water_temperature_30min.csv", encoding="shift_jis")
            voltage_estimate = pd.read_csv("../data/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis")

            for t in range(1, self.cfg.T+1):
                pv_value = float(pv_data.iat[t + 30*self.cfg.T, 2]) / 1000
                self.buy_price[t] = float(buy_price_data.iat[t-1, 2])
                
                # COP計算
                tmf = float(temperature_water_data.iat[t-1, 2])
                tma = float(temperature_outside_data.iat[t, 2])
                k = 1 if tma >= 5 else (tma/30 + 0.8333 if tma > 2 else 0.9)
                self.cop[t] = k * (0.175*tma - 0.1322*tmf + 4.076)

                for h in range(self.cfg.H):
                    self.p_pv[t, h] = pv_value
                    self.p_dmd[t, h] = float(dmd_data.iat[t-1, 10])
                    self.heat_dmd[t, h] = float(heat_dmd_data.iat[t-1+self.cfg.T, 2])

            for h in range(self.cfg.H):
                self.linear_ap[h] = voltage_estimate.iat[0, 3*h]
                self.linear_aq[h] = voltage_estimate.iat[0, 3*h+1]
                self.linear_b[h] = voltage_estimate.iat[0, 3*h+2]
                
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise

class MasterProblem:
    """アグリゲータ（配電網）の最適化モデル"""
    def __init__(self, config: GBDConfig, data: DataLoader):
        self.cfg = config
        self.data = data
        self.model = gp.Model("MasterProblem")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1) # QCPの双対取得を許可（必要に応じて）
        
        # 変数定義
        self._setup_variables()
        self._setup_constraints()
        self.cut_count = 0

    def _setup_variables(self):
        # 配電網変数 y_DN (Voltage, P_flow, Q_flow)
        # ※ ここでのP_flow, Q_flowは配電網側の値
        self.V_DN = self.model.addVars(range(1, self.cfg.T+1), range(self.cfg.H),
                                       lb=self.cfg.V_LL, ub=self.cfg.V_UL, name="V_DN")
        self.P_DN = self.model.addVars(range(1, self.cfg.T+1), range(self.cfg.H),
                                       lb=-GRB.INFINITY, name="P_DN")
        self.Q_DN = self.model.addVars(range(1, self.cfg.T+1), range(self.cfg.H),
                                       lb=-GRB.INFINITY, name="Q_DN")
        
        self.V_drop = self.model.addVars(range(1, self.cfg.T+1), range(self.cfg.H),
                                         lb=-GRB.INFINITY, name="V_drop")
        
        # 補助変数 LBD_h (各需要家のコストの下界)
        self.LBD = self.model.addVars(range(self.cfg.H), lb=-GRB.INFINITY, name="LBD")

        # アグリゲータ自身の変数
        self.P_AG_buy = self.model.addVars(range(1, self.cfg.T+1), lb=0, name="P_AG_buy")
        self.P_AG_sell = self.model.addVars(range(1, self.cfg.T+1), lb=0, name="P_AG_sell")

    def _setup_constraints(self):
        # 1. 配電網物理制約
        for t in range(1, self.cfg.T+1):
            # アグリゲータ電力バランス (P_AG = sum(P_DN_h0_in)) 
            # ※ 厳密にはP_DNは各需要家点での注入/吸収だが、ここでは簡略化のため
            # P_AG_buy - P_AG_sell = sum(P_DN) と定義
            # 論文定義: f_agg = sum(P_H) in DDR. 
            # ここではCentralizedに合わせて P_AG_buy/sell を計算
            
            total_P_DN = gp.quicksum(self.P_DN[t, h] for h in range(self.cfg.H))
            self.model.addConstr(self.P_AG_buy[t] - self.P_AG_sell[t] == total_P_DN, name=f"AG_balance_{t}")

            for h in range(self.cfg.H):
                # 累積潮流計算 (下流の和)
                P_cumulative = gp.quicksum(self.P_DN[t, j] for j in range(h, self.cfg.H))
                Q_cumulative = gp.quicksum(self.Q_DN[t, j] for j in range(h, self.cfg.H))

                # 電圧降下推定
                self.model.addConstr(
                    self.V_drop[t, h] == 
                    self.data.linear_ap[h] * P_cumulative + 
                    self.data.linear_aq[h] * Q_cumulative + 
                    self.data.linear_b[h],
                    name=f"V_drop_{t}_{h}"
                )

                # 電圧計算
                prev_V = self.cfg.V_BASE if h == 0 else self.V_DN[t, h-1]
                self.model.addConstr(self.V_DN[t, h] == prev_V + self.V_drop[t, h], name=f"V_calc_{t}_{h}")

                # 送電線容量 (二次錐)
                self.model.addQConstr(
                    P_cumulative * P_cumulative + Q_cumulative * Q_cumulative <= 
                    (self.cfg.S_LINE/100) * (self.cfg.S_LINE/100),
                    name=f"line_cap_{t}_{h}"
                )

        # 2. 目的関数
        # Master Obj = sum(LBD_h) + f_agg(y)
        # f_agg: Grid Cost (Buy/Sell) + DDR penalty
        grid_cost = gp.quicksum(
            self.data.buy_price[t] * self.P_AG_buy[t] - 
            self.cfg.SELL_PRICE * self.P_AG_sell[t]
            for t in range(1, self.cfg.T+1)
        )
        
        # DDR Cost (DDR期間の買電量)
        ddr_cost = 0
        if self.cfg.DR_END <= self.cfg.T:
            ddr_cost = gp.quicksum(self.P_AG_buy[t] for t in range(self.cfg.DR_START, self.cfg.DR_END+1))

        self.model.setObjective(
            gp.quicksum(self.LBD[h] for h in range(self.cfg.H)) + grid_cost + ddr_cost,
            GRB.MINIMIZE
        )

    def optimize(self):
        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL:
            # 次の反復のために境界変数yの値を抽出
            y_hat = {
                'V': {(t,h): self.V_DN[t,h].X for t in range(1, self.cfg.T+1) for h in range(self.cfg.H)},
                'P': {(t,h): self.P_DN[t,h].X for t in range(1, self.cfg.T+1) for h in range(self.cfg.H)},
                'Q': {(t,h): self.Q_DN[t,h].X for t in range(1, self.cfg.T+1) for h in range(self.cfg.H)},
            }
            return self.model.ObjVal, y_hat
        else:
            logger.error(f"Master Problem Infeasible or Failed. Status: {self.model.Status}")
            return None, None

    def add_optimality_cut(self, h, L_star, mu):
        """
        最適性カット: LBD_h >= L_star - mu^T * y_DN
        L_star = UB_sub + mu^T * y_hat (定数)
        """
        # mu^T * y_DN の項を作成
        mu_y_term = gp.quicksum(
            mu['V'][t] * self.V_DN[t, h] +
            mu['P'][t] * self.P_DN[t, h] +
            mu['Q'][t] * self.Q_DN[t, h]
            for t in range(1, self.cfg.T+1)
        )
        
        self.model.addConstr(
            self.LBD[h] >= L_star - mu_y_term,
            name=f"OptCut_{h}_{self.cut_count}"
        )
        self.cut_count += 1

    def add_feasibility_cut(self, h, L_star_feas, lam):
        """
        実行可能カット: L_star_feas - lam^T * y_DN <= 0
        L_star_feas = lam^T * x_sub (または定数項)
        """
        lam_y_term = gp.quicksum(
            lam['V'][t] * self.V_DN[t, h] +
            lam['P'][t] * self.P_DN[t, h] +
            lam['Q'][t] * self.Q_DN[t, h]
            for t in range(1, self.cfg.T+1)
        )
        
        self.model.addConstr(
            L_star_feas - lam_y_term <= 0,
            name=f"FeasCut_{h}_{self.cut_count}"
        )
        self.cut_count += 1


class SubProblem:
    """各需要家の最適化モデル（GBDサブ問題）"""
    def __init__(self, h: int, config: GBDConfig, data: DataLoader):
        self.h = h
        self.cfg = config
        self.data = data
        self.model = gp.Model(f"SubProblem_{h}")
        self.model.setParam('OutputFlag', 0)
        
        # Dualを取得するためにQCPDualを有効化
        self.model.setParam('QCPDual', 1) 
        
        self._build_model()

    def _build_model(self):
        """モデル構築（変数は初期化、制約は定義するが境界制約は後で更新）"""
        T = self.cfg.T
        
        # --- 変数定義 (緩和: Binary -> Continuous [0,1]) ---
        # GBDでDualを取得するため、MIPではなくLP/QCPとして解く必要がある
        # 本来はBinaryだが、連続緩和して双対変数を得る
        self.P_buy = self.model.addVars(range(1, T+1), lb=0, ub=self.cfg.BUY_MAX, name="P_buy")
        self.P_sell = self.model.addVars(range(1, T+1), lb=0, ub=self.cfg.SELL_MAX, name="P_sell")
        
        # 緩和変数 (0 <= delta <= 1)
        self.delta_buy = self.model.addVars(range(1, T+1), lb=0, ub=1, name="delta_buy")
        self.delta_sell = self.model.addVars(range(1, T+1), lb=0, ub=1, name="delta_sell")
        
        self.P_ch = self.model.addVars(range(1, T+1), lb=0, ub=self.cfg.N_B_PCS, name="P_ch")
        self.P_dch = self.model.addVars(range(1, T+1), lb=0, ub=self.cfg.N_B_PCS, name="P_dch")
        self.E_bat = self.model.addVars(range(1, T+1), lb=0.2*self.cfg.N_B, ub=self.cfg.N_B, name="E_bat")
        
        # 緩和変数
        self.delta_ch = self.model.addVars(range(1, T+1), lb=0, ub=1, name="delta_ch")
        self.delta_dch = self.model.addVars(range(1, T+1), lb=0, ub=1, name="delta_dch")
        
        # HP系
        self.P_hp = self.model.addVars(range(1, T+1), lb=0, ub=10.0, name="P_hp")
        self.H_prod = self.model.addVars(range(1, T+1), lb=0, name="H_prod")
        self.H_tank = self.model.addVars(range(T+1), lb=0, name="H_tank")
        
        # 緩和変数
        self.delta_hp = self.model.addVars(range(T+1), lb=0, ub=1, name="delta_hp")
        self.delta_start = self.model.addVars(range(T+1), lb=0, ub=1, name="delta_start")
        self.delta_stop = self.model.addVars(range(T+1), lb=0, ub=1, name="delta_stop")
        
        # 境界変数（H側）
        self.P_H = self.model.addVars(range(1, T+1), lb=-GRB.INFINITY, name="P_H")
        self.Q_H = self.model.addVars(range(1, T+1), lb=-GRB.INFINITY, name="Q_H")
        self.V_H = self.model.addVars(range(1, T+1), lb=self.cfg.V_LL, ub=self.cfg.V_UL, name="V_H")

        # --- 制約条件 ---
        # 電力バランス
        for t in range(1, T+1):
            self.model.addConstr(
                self.data.p_pv[t, self.h] + self.P_buy[t] + self.P_dch[t] ==
                self.data.p_dmd[t, self.h] + self.P_sell[t] + self.P_ch[t] + self.P_hp[t]
            )
            # 排他制御（緩和版）
            self.model.addConstr(self.P_buy[t] <= self.delta_buy[t] * self.cfg.BUY_MAX)
            self.model.addConstr(self.P_sell[t] <= self.delta_sell[t] * self.cfg.SELL_MAX)
            self.model.addConstr(self.delta_buy[t] + self.delta_sell[t] <= 1.0)
            
            # 蓄電池
            self.model.addConstr(self.P_ch[t] <= self.cfg.N_B_PCS * self.delta_ch[t])
            self.model.addConstr(self.P_dch[t] <= self.cfg.N_B_PCS * self.delta_dch[t])
            self.model.addConstr(self.delta_ch[t] + self.delta_dch[t] <= 1.0)
            
            prev_soc = 0.5 * self.cfg.N_B if t == 1 else self.E_bat[t-1]
            self.model.addConstr(
                self.E_bat[t] == prev_soc + 
                0.5 * self.cfg.ETA_B * self.P_ch[t] - 
                0.5 * self.P_dch[t] / self.cfg.ETA_B
            )
            
            # HP
            cop = self.data.cop[t]
            self.model.addConstr(
                self.P_hp[t] == self.cfg.XHP * self.delta_hp[t] +
                (self.H_prod[t] + self.cfg.R2 * self.cfg.CPH * self.delta_start[t]) / (self.cfg.CE * cop)
            )
            self.model.addConstr(self.H_prod[t] >= 0.5 * self.cfg.R1 * self.cfg.CPH * self.delta_hp[t])
            self.model.addConstr(self.H_prod[t] <= 0.5 * self.cfg.CPH * self.delta_hp[t])
            self.model.addConstr(self.delta_hp[t] - self.delta_hp[t-1] == self.delta_start[t] - self.delta_stop[t])
            self.model.addConstr(self.delta_start[t] + self.delta_stop[t] <= 1.0)
            
            self.model.addConstr(
                self.H_tank[t] == self.H_tank[t-1] + self.H_prod[t] - self.data.heat_dmd[t, self.h]
            )
            self.model.addConstr(self.H_tank[t] >= 0.2 * self.cfg.CW * self.cfg.VLT * 70)
            self.model.addConstr(self.H_tank[t] <= self.cfg.CW * self.cfg.VLT * 70)

            # 定義式
            self.model.addConstr(self.P_H[t] == self.P_buy[t] - self.P_sell[t])
            self.model.addConstr(self.Q_H[t] == 0.1 * (self.P_buy[t] - self.P_sell[t]))

        # 初期・終端条件
        self.model.addConstr(self.E_bat[T] == 0.5 * self.cfg.N_B)
        self.model.addConstr(self.H_tank[0] == 0.5 * self.cfg.CW * self.cfg.VLT * 70)
        self.model.addConstr(self.delta_hp[0] == 0)
        self.model.addConstr(self.delta_start[0] == 0)
        self.model.addConstr(self.delta_stop[0] == 0)
        self.model.addConstr(self.H_tank[T] >= 0.5 * self.cfg.CW * self.cfg.VLT * 70)

        # --- 境界等式制約 (初期値はダミー) ---
        # V_H - V_DN = 0 -> V_H = V_hat (Fix)
        # P_H - P_DN = 0 -> P_H = P_hat (Fix)
        # Q_H - Q_DN = 0 -> Q_H = Q_hat (Fix)
        # ※ これらは update_boundary でRHSを更新し、双対変数を取得するための制約
        self.boundary_constrs_V = {}
        self.boundary_constrs_P = {}
        self.boundary_constrs_Q = {}
        
        for t in range(1, T+1):
            self.boundary_constrs_V[t] = self.model.addConstr(self.V_H[t] == 0, name=f"bound_V_{t}")
            self.boundary_constrs_P[t] = self.model.addConstr(self.P_H[t] == 0, name=f"bound_P_{t}")
            self.boundary_constrs_Q[t] = self.model.addConstr(self.Q_H[t] == 0, name=f"bound_Q_{t}")

        # 目的関数 (需要家コスト)
        cost = gp.quicksum(
            self.data.buy_price[t] * self.P_buy[t] - self.cfg.SELL_PRICE * self.P_sell[t]
            for t in range(1, T+1)
        )
        self.model.setObjective(cost, GRB.MINIMIZE)
        self.model.update()

    def update_and_solve(self, y_hat: Dict) -> Dict:
        """
        マスターからの境界変数y_hatを受け取り、サブ問題を解く
        戻り値: {'status', 'obj', 'mu', 'y_hat'}
        """
        # 境界制約の更新 (RHSを変更)
        for t in range(1, self.cfg.T+1):
            self.boundary_constrs_V[t].RHS = y_hat['V'][(t, self.h)]
            self.boundary_constrs_P[t].RHS = y_hat['P'][(t, self.h)]
            self.boundary_constrs_Q[t].RHS = y_hat['Q'][(t, self.h)]
        
        self.model.update()
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            # 双対変数(mu)の取得
            # 制約式: V_H - y_hat = 0  => Pi corresponds to this equality
            mu = {'V': {}, 'P': {}, 'Q': {}}
            for t in range(1, self.cfg.T+1):
                mu['V'][t] = self.boundary_constrs_V[t].Pi
                mu['P'][t] = self.boundary_constrs_P[t].Pi
                mu['Q'][t] = self.boundary_constrs_Q[t].Pi
            
            return {
                'status': 'OPTIMAL',
                'obj': self.model.ObjVal,
                'mu': mu,
                'y_hat': { # 今の反復で使ったy
                    'V': {t: y_hat['V'][(t, self.h)] for t in range(1, self.cfg.T+1)},
                    'P': {t: y_hat['P'][(t, self.h)] for t in range(1, self.cfg.T+1)},
                    'Q': {t: y_hat['Q'][(t, self.h)] for t in range(1, self.cfg.T+1)}
                }
            }
        else:
            return {'status': 'INFEASIBLE'}

    def solve_feasibility(self, y_hat: Dict) -> Dict:
        """
        実行不可能時の緩和問題 (Step 2b)
        min sum(alpha)
        """
        # 実行可能化モデルを作成 (既存モデルのコピーまたは変更)
        # ここでは簡易化のため、スラック変数を追加して目的関数を変更する
        # ※ 実際の実装ではモデルの状態を戻す必要があるため、コピーするか制約を一時的に変更する
        
        # 今回は一時的に制約を変更するアプローチ
        # 境界制約を不等式+スラックに変更
        alphas = self.model.addVars(range(1, self.cfg.T+1), range(6), lb=0, name="alpha")
        
        # 一時的に元の境界等式制約を削除（または無効化）して、緩和制約を追加
        # 実装簡略化のため、元の制約のRHSを変更するのではなく、新しい緩和制約を追加する形をとる
        # 実際には、SubProblemクラス内でFeasibility用モデルを別途持つのがクリーンだが、
        # ここでは動的に追加・削除を行う
        
        tmp_constrs = []
        for t in range(1, self.cfg.T+1):
            # 制約削除
            self.model.remove(self.boundary_constrs_V[t])
            self.model.remove(self.boundary_constrs_P[t])
            self.model.remove(self.boundary_constrs_Q[t])
            
            # 緩和制約 (V_H - V_hat <= alpha1, -V_H + V_hat <= alpha2, etc.)
            v_hat = y_hat['V'][(t, self.h)]
            p_hat = y_hat['P'][(t, self.h)]
            q_hat = y_hat['Q'][(t, self.h)]
            
            c1 = self.model.addConstr(self.V_H[t] - v_hat <= alphas[t, 0])
            c2 = self.model.addConstr(-self.V_H[t] + v_hat <= alphas[t, 1])
            c3 = self.model.addConstr(self.P_H[t] - p_hat <= alphas[t, 2])
            c4 = self.model.addConstr(-self.P_H[t] + p_hat <= alphas[t, 3])
            c5 = self.model.addConstr(self.Q_H[t] - q_hat <= alphas[t, 4])
            c6 = self.model.addConstr(-self.Q_H[t] + q_hat <= alphas[t, 5])
            
            tmp_constrs.extend([c1, c2, c3, c4, c5, c6])

        # 目的関数変更 (Min sum alpha)
        self.model.setObjective(gp.quicksum(alphas), GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        
        lam = {'V': {}, 'P': {}, 'Q': {}}
        L_star_feas = 0.0 # lambda^T * x の部分だが、定式化上Dualから計算
        
        if self.model.Status == GRB.OPTIMAL:
            # 双対変数の整理 lambda = lambda_pos - lambda_neg
            # 制約 c1 (lambda1), c2 (lambda2) -> Vに関する双対は lambda1 - lambda2
            for t in range(1, self.cfg.T+1):
                # インデックス計算が面倒だが、順番に追加した通りに取得
                l1 = tmp_constrs[(t-1)*6 + 0].Pi
                l2 = tmp_constrs[(t-1)*6 + 1].Pi
                l3 = tmp_constrs[(t-1)*6 + 2].Pi
                l4 = tmp_constrs[(t-1)*6 + 3].Pi
                l5 = tmp_constrs[(t-1)*6 + 4].Pi
                l6 = tmp_constrs[(t-1)*6 + 5].Pi
                
                lam['V'][t] = l1 - l2
                lam['P'][t] = l3 - l4
                lam['Q'][t] = l5 - l6
            
            # 実行可能カットの定数項 (L_*^q = lambda^T * x_H)
            # 変数x_Hの最適値を使って計算
            val_term = 0
            for t in range(1, self.cfg.T+1):
                val_term += lam['V'][t] * self.V_H[t].X
                val_term += lam['P'][t] * self.P_H[t].X
                val_term += lam['Q'][t] * self.Q_H[t].X
            L_star_feas = val_term
            
            status = 'OPTIMAL'
        else:
            status = 'FAILED'

        # --- クリーンアップ (元のモデル構造に戻す) ---
        for c in tmp_constrs:
            self.model.remove(c)
        for t in range(1, self.cfg.T+1):
            for i in range(6):
                self.model.remove(alphas[t, i])
                
        # 境界制約の再追加
        self.boundary_constrs_V = {}
        self.boundary_constrs_P = {}
        self.boundary_constrs_Q = {}
        for t in range(1, self.cfg.T+1):
            self.boundary_constrs_V[t] = self.model.addConstr(self.V_H[t] == 0, name=f"bound_V_{t}")
            self.boundary_constrs_P[t] = self.model.addConstr(self.P_H[t] == 0, name=f"bound_P_{t}")
            self.boundary_constrs_Q[t] = self.model.addConstr(self.Q_H[t] == 0, name=f"bound_Q_{t}")
        
        # 目的関数を元に戻す
        cost = gp.quicksum(
            self.data.buy_price[t] * self.P_buy[t] - self.cfg.SELL_PRICE * self.P_sell[t]
            for t in range(1, self.cfg.T+1)
        )
        self.model.setObjective(cost, GRB.MINIMIZE)
        self.model.update()

        return {'status': status, 'lam': lam, 'L_star_feas': L_star_feas}


class GBDController:
    """GBDアルゴリズム全体制御"""
    def __init__(self):
        self.cfg = GBDConfig()
        self.data = DataLoader(self.cfg)
        self.master = MasterProblem(self.cfg, self.data)
        self.subs = [SubProblem(h, self.cfg, self.data) for h in range(self.cfg.H)]
        
        self.LB = -GRB.INFINITY
        self.UB = GRB.INFINITY
        self.history = []

    def run(self, max_iter=20):
        logger.info("GBD Optimization Start")
        start_time = time.time()
        
        # Step 1: 初期化 (y_hatの初期値)
        # 初期解としてV=Base, P=0, Q=0などを設定
        y_hat = {
            'V': {(t,h): self.cfg.V_BASE for t in range(1, self.cfg.T+1) for h in range(self.cfg.H)},
            'P': {(t,h): 0.0 for t in range(1, self.cfg.T+1) for h in range(self.cfg.H)},
            'Q': {(t,h): 0.0 for t in range(1, self.cfg.T+1) for h in range(self.cfg.H)},
        }

        for k in range(1, max_iter+1):
            logger.info(f"=== Iteration {k} ===")
            
            sub_obj_sum = 0
            feasible_all = True
            
            # Step 2: サブ問題を解く (各需要家並列可)
            for h in range(self.cfg.H):
                res = self.subs[h].update_and_solve(y_hat)
                
                if res['status'] == 'OPTIMAL':
                    # Step 2a: 実行可能 -> 最適性カット生成
                    sub_obj_sum += res['obj']
                    
                    # 最適性カットの定数項計算
                    # L* = UB_h + mu^T * y_hat
                    mu = res['mu']
                    y_h = res['y_hat'] # このサブ問題用のy
                    
                    mu_y_val = sum(
                        mu['V'][t] * y_h['V'][t] + 
                        mu['P'][t] * y_h['P'][t] + 
                        mu['Q'][t] * y_h['Q'][t]
                        for t in range(1, self.cfg.T+1)
                    )
                    L_star = res['obj'] + mu_y_val
                    
                    self.master.add_optimality_cut(h, L_star, mu)
                    
                else:
                    # Step 2b: 実行不可能 -> 実行可能カット生成
                    feasible_all = False
                    logger.warning(f"SubProblem {h} Infeasible. Generating Feasibility Cut...")
                    
                    res_feas = self.subs[h].solve_feasibility(y_hat)
                    if res_feas['status'] == 'OPTIMAL':
                        self.master.add_feasibility_cut(h, res_feas['L_star_feas'], res_feas['lam'])
                    else:
                        logger.error(f"Feasibility Problem Failed for household {h}")
                        return
            
            # Step 3: マスター問題を解く
            # 上界の更新 (全サブが実行可能な場合のみ)
            if feasible_all:
                # UB = sum(SubObj) + MasterObj(fixed y)
                # Note: f_agg(y) need to be calculated with y_hat
                # 簡易的にマスターの現在の目的関数からLBDを除いたもの(grid cost)を計算してもよいが
                # y_hatは前回のMaster解なので、それを使って計算する
                
                # y_hatに基づくGridコスト計算
                grid_cost_val = self._calc_master_obj_fixed(y_hat)
                current_ub = sub_obj_sum + grid_cost_val
                self.UB = min(self.UB, current_ub)
                logger.info(f"Updated UB: {self.UB:.4f}")

            # マスター最適化
            m_obj, new_y_hat = self.master.optimize()
            if new_y_hat is None:
                logger.error("Master Problem Infeasible")
                break
                
            self.LB = m_obj # 新しい下界
            logger.info(f"Updated LB: {self.LB:.4f}")
            
            y_hat = new_y_hat # 次の反復へ
            
            # 収束判定
            gap = abs(self.UB - self.LB)
            logger.info(f"Gap: {gap:.4f}")
            self.history.append({'iter': k, 'LB': self.LB, 'UB': self.UB, 'Gap': gap})
            
            if gap < self.cfg.EPSILON:
                logger.info("Converged!")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Optimization Finished in {total_time:.2f} sec")
        return self.history

    def _calc_master_obj_fixed(self, y_hat):
        """y_hat固定時のマスター側のコスト(Grid Cost)を計算"""
        # 実際にはy_hatからP_AG_buy/sellを再計算する必要がある
        # ここでは簡易的に、P_AG = sum(P_DN)として計算
        cost = 0
        for t in range(1, self.cfg.T+1):
            p_total = sum(y_hat['P'][(t, h)] for h in range(self.cfg.H))
            if p_total >= 0:
                cost += p_total * self.data.buy_price[t]
            else:
                cost += p_total * self.cfg.SELL_PRICE # p_total < 0 means sell (sign correct?)
                # Centralized logic: Buy*Price - Sell*Price. 
                # If P_total > 0 (Buy from Grid), cost > 0.
                # If P_total < 0 (Sell to Grid), cost < 0.
        return cost

if __name__ == "__main__":
    gbd = GBDController()
    gbd.run()