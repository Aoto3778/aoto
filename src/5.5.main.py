"""
Generalized Benders Decomposition for Distribution Network Optimization
配電網最適化のための汎用ベンダーズ分解法

アグリゲータと需要家の情報を完全に分離し、境界変数のみで協調制御を実現
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==================== データ管理クラス ====================
@dataclass
class SystemParameters:
    """システム全体のパラメータ管理"""
    # 時間・需要家設定
    T: int = 48  # タイムスロット数（30分粒度）
    H: int = 3   # 需要家数
    
    # 買電売電上限
    BUY_MAX: float = 5.0   # 買電上限
    SELL_MAX: float = 5.0  # 売電上限
    
    # 電圧制約
    V_BASE: float = 99.8  # 基準電圧[V]
    V_UL: float = 107.0   # 電圧上限[V]
    V_LL: float = 95.0    # 電圧下限[V]
    
    # 送電線容量
    S_LINE: float = 2020.0  # 定格電力[VA] (101V * 20A)
    
    # 蓄電池パラメータ
    N_B: float = 6.3        # 蓄電池容量[kWh]
    N_B_PCS: float = 1.5    # PCS容量[kW]
    ETA_B: float = 0.95     # 充放電効率
    SOC_MIN: float = 0.2    # 最小SOC
    SOC_INIT: float = 0.5   # 初期・終端SOC
    
    # ヒートポンプパラメータ
    VLT: float = 370.0      # 貯湯槽容量[L]
    CW: float = 0.0042      # 水の比熱[MJ/L/℃]
    CE: float = 3.6         # 変換係数
    CPH: float = 16.2       # HP加熱能力[MJ]
    XHP: float = 0.013      # 補器消費電力[kWh]
    R1: float = 0.05        # 最低熱製造率
    R2: float = 0.1         # 起動時ロス率
    TANK_MAX_TEMP: float = 70.0  # 最高貯湯温度[℃]
    
    # 料金設定
    SELL_PRICE: float = 2.0  # 売電単価[JPY/kWh]
    
    # 下げDR期間（13時～16時）
    DR_START: int = 26  # 13:00 (slot 26)
    DR_END: int = 32    # 16:00 (slot 32)
    
    # 収束判定
    EPSILON: float = 0.01   # 収束許容誤差
    MAX_ITER: int = 100     # 最大反復回数


class DataManager:
    """データ読み込みと管理"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.data = {}
        self.load_all_data()
        
        
    def load_all_data(self) -> Dict:
        """全データを読み込み"""
        T = self.params.T
        H = self.params.H
        
        # 電力需要データ
        dmd_data = pd.read_csv("../data/input/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", encoding="shift_jis")
        
        # 料金データ
        buy_price_data = pd.read_csv("../data/input/buy_energy30.csv", encoding="shift_jis")
        spot_price_data = pd.read_csv("../data/input/spot_market_price_30min.csv", encoding="shift_jis")
        
        # 太陽光・その他
        pv_data = pd.read_csv("../data/input/pv_output.csv")
        heat_dmd_data = pd.read_csv("../data/input/heat_demand1y.csv", encoding="shift_jis")
        temperature_outside_data = pd.read_csv("../data/input/temperature1y.csv", encoding="shift_jis")
        temperature_water_data = pd.read_csv("../data/input/water_temperature_30min.csv", encoding="shift_jis")
        voltage_estimate = pd.read_csv("../data/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis")
        
        # データ格納
        self.data['time'] = {t: dmd_data.iat[t-1, 0] for t in range(1, T+1)}
        
        # 太陽光出力
        self.data['p_pv'] = {}
        for t in range(1, T+1):
            pv_value = float(pv_data.iat[t + 30*T, 2]) / 1000
            for i in range(H):
                self.data['p_pv'][t, i] = pv_value
        
        # 電力需要
        self.data['p_dmd'] = {}
        for t in range(T+1):
            for i in range(H):
                self.data['p_dmd'][t, i] = float(dmd_data.iat[t-1, 10])
        
        # 電気料金
        self.data['buy_price'] = {}
        for t in range(1, T+1):
            self.data['buy_price'][t] = float(buy_price_data.iat[t-1, 2])
        
        # スポット価格
        self.data['spot_price'] = {}
        for t in range(1, T+1):
            self.data['spot_price'][t] = float(spot_price_data.iat[t, 2])
        
        # 熱需要
        self.data['H_dmd'] = {}
        for t in range(1, T+1):
            for i in range(H):
                self.data['H_dmd'][t, i] = float(heat_dmd_data.iat[t-1+T, 2])
        
        # 温度・COP計算
        self.data['cop'] = {}
        for t in range(1, T+1):
            tmf = float(temperature_water_data.iat[t-1, 2])
            tma = float(temperature_outside_data.iat[t, 2])
            
            if tma >= 5:
                k = 1
            elif tma > 2:
                k = tma/30 + 0.8333
            else:
                k = 0.9
            
            self.data['cop'][t] = k * (0.175*tma - 0.1322*tmf + 4.076)
        
        # 電圧推定パラメータ
        self.data['linear_ap'] = {}
        self.data['linear_aq'] = {}
        self.data['linear_b'] = {}
        for i in range(H):
            self.data['linear_ap'][i] = voltage_estimate.iat[0, 3*i]
            self.data['linear_aq'][i] = voltage_estimate.iat[0, 3*i+1]
            self.data['linear_b'][i] = voltage_estimate.iat[0, 3*i+2]
        
        return self.data

# ==================== サブ問題（需要家最適化） ====================
class HouseholdSubproblem:
    """需要家個別の最適化問題（プライバシー保護）"""
    
    def __init__(self, household_id: int, data: DataManager, params: SystemParameters):
        self.h = household_id
        self.data = data
        self.params = params
        self.model = None
        self.setup_model()
        
    def setup_model(self):
        """最適化モデルの構築"""
        self.model = gp.Model(f"Household_{self.h}")
        self.model.setParam('OutputFlag', 0)  # ログ出力を抑制
        self.model.setParam('QCPDual', 1)     # 二次制約の双対変数取得
        
        T = self.params.T
        
        # === 決定変数 ===
        # 買電・売電
        self.P_buy = self.model.addVars(T, lb=0, name="P_buy")
        self.P_sell = self.model.addVars(T, lb=0, name="P_sell")
        self.delta_P_buy = self.model.addVars(T, vtype=GRB.BINARY, name="delta_P_buy")
        self.delta_P_sell = self.model.addVars(T, vtype=GRB.BINARY, name="delta_P_sell")
        
        # 蓄電池
        self.P_ch = self.model.addVars(T, lb=0, name="P_charge")
        self.P_dch = self.model.addVars(T, lb=0, name="P_discharge")
        self.E_bat = self.model.addVars(T+1, lb=0, name="E_battery")
        self.delta_ch = self.model.addVars(T, vtype=GRB.BINARY, name="delta_charge")
        self.delta_dch = self.model.addVars(T, vtype=GRB.BINARY, name="delta_discharge")
        
        # ヒートポンプ
        self.P_hp = self.model.addVars(T, lb=0, name="P_hp")
        self.H_prod = self.model.addVars(T, lb=0, name="H_produce")
        self.H_tank = self.model.addVars(T+1, lb=0, name="H_tank")
        self.delta_hp = self.model.addVars(T, vtype=GRB.BINARY, name="delta_hp")
        self.delta_start = self.model.addVars(T, vtype=GRB.BINARY, name="delta_start")
        
        # 境界変数（アグリゲータとの接続点）
        self.V_H = self.model.addVars(T, lb=self.params.V_LL, ub=self.params.V_UL, name="V_H")
        self.P_H = self.model.addVars(T, lb=-GRB.INFINITY, name="P_H")
        self.Q_H = self.model.addVars(T, lb=-GRB.INFINITY, name="Q_H")
        
        self.model.update()
        
    def solve(self, boundary_fixed: Dict) -> Tuple[float, Dict, bool]:
        """
        境界変数を固定してサブ問題を解く
        
        Args:
            boundary_fixed: 固定された境界変数 {'V_DN': [], 'P_DN': [], 'Q_DN': []}
        
        Returns:
            (目的関数値, 双対変数, 実行可能性)
        """
        T = self.params.T
        
        # 境界制約を一時的に追加
        temp_constrs = []
        
        # 境界等式制約の追加
        for t in range(T):
            c1 = self.model.addConstr(
                self.V_H[t] == boundary_fixed['V_DN'][t],
                name=f"boundary_V_{t}"
            )
            c2 = self.model.addConstr(
                self.P_H[t] == boundary_fixed['P_DN'][t],
                name=f"boundary_P_{t}"
            )
            c3 = self.model.addConstr(
                self.Q_H[t] == boundary_fixed['Q_DN'][t],
                name=f"boundary_Q_{t}"
            )
            temp_constrs.extend([c1, c2, c3])
        
        # 需要家内制約の設定
        self._add_household_constraints()
        
        # 目的関数：電力コスト最小化
        obj = gp.quicksum(
            self.data.buy_price[t] * self.P_buy[t] - 
            self.params.SELL_PRICE * self.P_sell[t]
            for t in range(T)
        )
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # 最適化実行
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            # 双対変数の取得
            duals = {
                'mu_V': [c.Pi for c in temp_constrs[0::3]],
                'mu_P': [c.Pi for c in temp_constrs[1::3]],
                'mu_Q': [c.Pi for c in temp_constrs[2::3]]
            }
            obj_val = self.model.ObjVal
            feasible = True
            
        elif self.model.Status == GRB.INFEASIBLE:
            # 実行不可能な場合は実行可能性サブ問題を解く
            duals, obj_val = self._solve_feasibility_subproblem(boundary_fixed)
            feasible = False
            
        else:
            raise RuntimeError(f"Unexpected status: {self.model.Status}")
        
        # 一時制約の削除
        for c in temp_constrs:
            self.model.remove(c)
        self.model.update()
        
        return obj_val, duals, feasible
    
    def _add_household_constraints(self):
        """需要家内の制約条件を追加"""
        T = self.params.T
        
        for t in range(T):
            # 電力バランス
            self.model.addConstr(
                self.data['p_pv'][self.h][t] + self.P_buy[t] + self.P_dch[t] ==
                self.data.p_demand[self.h][t] + self.P_sell[t] + 
                self.P_ch[t] + self.P_hp[t],
                name=f"power_balance_{t}"
            )
            
            # 買電売電同時禁止
            self.model.addConstr(self.delta_P_buy[t] * self.params.BUY_MAX >= self.P_buy[t])
            self.model.addConstr(self.P_sell[t] <= self.delta_sell[t] * self.params.SELL_MAX )
            self.model.addConstr(self.delta_P_buy[t] + self.delta_P_sell[t] <= 1)
        
            # 境界での電力関係
            self.model.addConstr(
                self.P_H[t] == self.P_buy[t] - self.P_sell[t],
                name=f"boundary_power_{t}"
            )
            
            # 無効電力（有効電力の10%）
            self.model.addConstr(
                self.Q_H[t] == 0.1 * self.P_H[t],
                name=f"reactive_power_{t}"
            )
            
            # 蓄電池制約
            self._add_battery_constraints(t)
            
            # ヒートポンプ制約
            self._add_heatpump_constraints(t)
    
    def _add_battery_constraints(self, t: int):
        """蓄電池制約の追加"""
        # 充放電制約
        self.model.addConstr(
            self.P_ch[t] <= self.params.N_B_PCS * self.delta_ch[t]
        )
        self.model.addConstr(
            self.P_dch[t] <= self.params.N_B_PCS * self.delta_dch[t]
        )
        self.model.addConstr(
            self.delta_ch[t] + self.delta_dch[t] <= 1
        )
        
        # SOC制約
        self.model.addConstr(
            self.E_bat[t+1] >= self.params.SOC_MIN * self.params.N_B
        )
        self.model.addConstr(
            self.E_bat[t+1] <= self.params.N_B
        )
        
        # SOC更新
        self.model.addConstr(
            self.E_bat[t+1] == self.E_bat[t] + 
            0.5 * self.params.ETA_B * self.P_ch[t] -
            0.5 * self.P_dch[t] / self.params.ETA_B
        )
        
        # 初期・終端条件
        if t == 0:
            self.model.addConstr(
                self.E_bat[0] == self.params.SOC_INIT * self.params.N_B
            )
        if t == self.params.T - 1:
            self.model.addConstr(
                self.E_bat[self.params.T] == self.params.SOC_INIT * self.params.N_B
            )
    
    def _add_heatpump_constraints(self, t: int):
        """ヒートポンプ制約の追加"""
        cop = self.data.cop[t]
        
        # HP消費電力
        self.model.addConstr(
            self.P_hp[t] == self.params.XHP * self.delta_hp[t] +
            self.H_prod[t] / (self.params.CE * cop)
        )
        
        # 熱製造量制約
        self.model.addConstr(
            self.H_prod[t] >= 0.5 * self.params.R1 * self.params.CPH * self.delta_hp[t]
        )
        self.model.addConstr(
            self.H_prod[t] <= 0.5 * self.params.CPH * self.delta_hp[t]
        )
        
        # 貯湯槽更新
        if t > 0:
            self.model.addConstr(
                self.H_tank[t] == self.H_tank[t-1] + 
                self.H_prod[t] - self.data.heat_demand[self.h][t]
            )
        else:
            self.model.addConstr(
                self.H_tank[0] == 0.5 * self.params.CW * self.params.VLT * 
                self.params.TANK_MAX_TEMP
            )
        
        # 貯湯槽容量制約
        self.model.addConstr(
            self.H_tank[t] <= self.params.CW * self.params.VLT * 
            self.params.TANK_MAX_TEMP
        )
        self.model.addConstr(
            self.H_tank[t] >= 0.2 * self.params.CW * self.params.VLT * 
            self.params.TANK_MAX_TEMP
        )
        
        # 終端条件
        if t == self.params.T - 1:
            self.model.addConstr(
                self.H_tank[self.params.T] == 0.5 * self.params.CW * 
                self.params.VLT * self.params.TANK_MAX_TEMP
            )
    
    def _solve_feasibility_subproblem(self, boundary_fixed: Dict) -> Tuple[Dict, float]:
        """実行可能性サブ問題を解く"""
        # 緩和変数を追加した実行可能性問題を構築
        feas_model = gp.Model(f"Feasibility_H{self.h}")
        feas_model.setParam('OutputFlag', 0)
        
        T = self.params.T
        
        # 緩和変数
        alpha = feas_model.addVars(3, T, lb=0, name="alpha")
        
        # 元の変数をコピー（簡略化のため主要変数のみ）
        V_H = feas_model.addVars(T, lb=-GRB.INFINITY, name="V_H_feas")
        P_H = feas_model.addVars(T, lb=-GRB.INFINITY, name="P_H_feas")
        Q_H = feas_model.addVars(T, lb=-GRB.INFINITY, name="Q_H_feas")
        
        # 緩和された境界制約
        for t in range(T):
            feas_model.addConstr(
                V_H[t] - boundary_fixed['V_DN'][t] <= alpha[0, t]
            )
            feas_model.addConstr(
                boundary_fixed['V_DN'][t] - V_H[t] <= alpha[0, t]
            )
            feas_model.addConstr(
                P_H[t] - boundary_fixed['P_DN'][t] <= alpha[1, t]
            )
            feas_model.addConstr(
                boundary_fixed['P_DN'][t] - P_H[t] <= alpha[1, t]
            )
            feas_model.addConstr(
                Q_H[t] - boundary_fixed['Q_DN'][t] <= alpha[2, t]
            )
            feas_model.addConstr(
                boundary_fixed['Q_DN'][t] - Q_H[t] <= alpha[2, t]
            )
        
        # 目的関数：緩和変数の総和最小化
        feas_model.setObjective(
            gp.quicksum(alpha[i, t] for i in range(3) for t in range(T)),
            GRB.MINIMIZE
        )
        
        feas_model.optimize()
        
        if feas_model.Status == GRB.OPTIMAL:
            # 双対変数（ラグランジュ乗数）を取得
            lambda_vals = {
                'lambda_V': [1.0] * T,  # 簡略化
                'lambda_P': [1.0] * T,
                'lambda_Q': [1.0] * T
            }
            return lambda_vals, feas_model.ObjVal
        else:
            raise RuntimeError("Feasibility subproblem failed")


# ==================== マスター問題（アグリゲータ最適化） ====================
class AggregatorMasterProblem:
    """配電網運用の最適化（アグリゲータ）"""
    
    def __init__(self, data: DataManager, params: SystemParameters):
        self.data = data
        self.params = params
        self.model = gp.Model("Aggregator_Master")
        self.model.setParam('OutputFlag', 0)
        
        # カット平面の保存
        self.optimality_cuts = []
        self.feasibility_cuts = []
        
        self.setup_model()
        
    def setup_model(self):
        """マスター問題のモデル構築"""
        T, H = self.params.T, self.params.H
        
        # === 決定変数 ===
        # 境界変数（各需要家との接続点）
        self.V_DN = {}  # 電圧
        self.P_DN = {}  # 有効電力
        self.Q_DN = {}  # 無効電力
        
        for h in range(H):
            self.V_DN[h] = self.model.addVars(
                T, lb=self.params.V_LL, ub=self.params.V_UL, name=f"V_DN_{h}"
            )
            self.P_DN[h] = self.model.addVars(
                T, lb=-GRB.INFINITY, name=f"P_DN_{h}"
            )
            self.Q_DN[h] = self.model.addVars(
                T, lb=-GRB.INFINITY, name=f"Q_DN_{h}"
            )
        
        # アグリゲータの系統買電
        self.P_grid = self.model.addVars(T, lb=0, name="P_grid")
        
        # 下界補助変数（各需要家）
        self.LBD = self.model.addVars(H, lb=-GRB.INFINITY, name="LBD")
        
        self.model.update()
        
    def add_network_constraints(self):
        """配電網制約の追加"""
        T, H = self.params.T, self.params.H
        
        for t in range(T):
            # 電圧降下の推定（各需要家）
            for h in range(H):
                # 累積有効電力（上流の需要家含む）
                P_cumulative = gp.quicksum(self.P_DN[j][t] for j in range(h, H))
                Q_cumulative = gp.quicksum(self.Q_DN[j][t] for j in range(h, H))
                
                # 電圧降下
                v_params = self.data.voltage_params[h]
                delta_V = (v_params['alpha_p'] * P_cumulative + 
                          v_params['alpha_q'] * Q_cumulative + 
                          v_params['beta'])
                
                # 電圧計算
                if h == 0:
                    self.model.addConstr(
                        self.V_DN[h][t] == self.params.V_BASE + delta_V,
                        name=f"voltage_{h}_{t}"
                    )
                else:
                    self.model.addConstr(
                        self.V_DN[h][t] == self.V_DN[h-1][t] + delta_V,
                        name=f"voltage_{h}_{t}"
                    )
                
                # 送電線容量制約（二次錐制約）
                self.model.addQConstr(
                    P_cumulative * P_cumulative + Q_cumulative * Q_cumulative <= 
                    self.params.S_LINE * self.params.S_LINE,
                    name=f"line_capacity_{h}_{t}"
                )
            
            # 系統買電量
            self.model.addConstr(
                self.P_grid[t] == gp.quicksum(self.P_DN[h][t] for h in range(H)),
                name=f"grid_power_{t}"
            )
    
    def solve(self) -> Tuple[Dict, float]:
        """マスター問題を解く"""
        T, H = self.params.T, self.params.H
        
        # 配電網制約の追加
        self.add_network_constraints()
        
        # カット平面制約の追加
        for cut in self.optimality_cuts:
            h = cut['household']
            self.model.addConstr(
                self.LBD[h] >= cut['constant'] +
                gp.quicksum(
                    cut['mu_V'][t] * self.V_DN[h][t] +
                    cut['mu_P'][t] * self.P_DN[h][t] +
                    cut['mu_Q'][t] * self.Q_DN[h][t]
                    for t in range(T)
                ),
                name=f"opt_cut_{h}_{len(self.optimality_cuts)}"
            )
        
        for cut in self.feasibility_cuts:
            h = cut['household']
            self.model.addConstr(
                0 >= cut['constant'] +
                gp.quicksum(
                    cut['lambda_V'][t] * self.V_DN[h][t] +
                    cut['lambda_P'][t] * self.P_DN[h][t] +
                    cut['lambda_Q'][t] * self.Q_DN[h][t]
                    for t in range(T)
                ),
                name=f"feas_cut_{h}_{len(self.feasibility_cuts)}"
            )
        
        # 目的関数：需要家コスト + DR期間の系統買電最小化
        obj = gp.quicksum(self.LBD[h] for h in range(H))
        obj += gp.quicksum(
            self.P_grid[t] 
            for t in range(self.params.DR_START, self.params.DR_END + 1)
        )
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # 最適化実行
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            # 境界変数の値を取得
            boundary_vars = {}
            for h in range(H):
                boundary_vars[h] = {
                    'V_DN': [self.V_DN[h][t].X for t in range(T)],
                    'P_DN': [self.P_DN[h][t].X for t in range(T)],
                    'Q_DN': [self.Q_DN[h][t].X for t in range(T)]
                }
            
            lower_bound = self.model.ObjVal
            return boundary_vars, lower_bound
        else:
            raise RuntimeError(f"Master problem failed with status {self.model.Status}")
    
    def add_optimality_cut(self, household: int, obj_val: float, duals: Dict):
        """最適カット平面の追加"""
        cut = {
            'household': household,
            'constant': obj_val,
            'mu_V': duals['mu_V'],
            'mu_P': duals['mu_P'],
            'mu_Q': duals['mu_Q']
        }
        self.optimality_cuts.append(cut)
    
    def add_feasibility_cut(self, household: int, duals: Dict):
        """実行可能カット平面の追加"""
        cut = {
            'household': household,
            'constant': 0,
            'lambda_V': duals['lambda_V'],
            'lambda_P': duals['lambda_P'],
            'lambda_Q': duals['lambda_Q']
        }
        self.feasibility_cuts.append(cut)


# ==================== GBDソルバー ====================
class GBDSolver:
    """Generalized Benders Decomposition統合ソルバー"""
    
    def __init__(self):
        self.params = SystemParameters()
        self.data = DataManager(self.params)
        self.master = AggregatorMasterProblem(self.data, self.params)
        self.subproblems = [
            HouseholdSubproblem(h, self.data, self.params)
            for h in range(self.params.H)
        ]
        
        # 収束履歴
        self.iteration_history = []
        
    def solve(self) -> Dict:
        """GBDアルゴリズムの実行"""
        logger.info("="*60)
        logger.info("Generalized Benders Decomposition開始")
        logger.info(f"需要家数: {self.params.H}, タイムスロット: {self.params.T}")
        logger.info("="*60)
        
        # 初期化
        LB = -float('inf')
        UB = float('inf')
        k = 0
        
        # 初期境界変数（各需要家に同じ初期値）
        boundary_vars = {}
        for h in range(self.params.H):
            boundary_vars[h] = {
                'V_DN': [self.params.V_BASE] * self.params.T,
                'P_DN': [1.0] * self.params.T,
                'Q_DN': [0.1] * self.params.T
            }
        
        start_time = time.time()
        
        while k < self.params.MAX_ITER:
            k += 1
            logger.info(f"\n反復 {k}:")
            
            # Step 1: 各需要家のサブ問題を解く
            subproblem_results = []
            household_obj_sum = 0
            
            for h in range(self.params.H):
                obj_val, duals, feasible = self.subproblems[h].solve(boundary_vars[h])
                subproblem_results.append((h, obj_val, duals, feasible))
                
                if feasible:
                    household_obj_sum += obj_val
                    self.master.add_optimality_cut(h, obj_val, duals)
                    logger.info(f"  需要家{h}: 実行可能 (obj={obj_val:.2f})")
                else:
                    self.master.add_feasibility_cut(h, duals)
                    logger.info(f"  需要家{h}: 実行不可能")
            
            # 上界の更新
            if all(result[3] for result in subproblem_results):  # 全て実行可能
                # DR期間の系統買電を計算
                dr_cost = sum(
                    sum(boundary_vars[h]['P_DN'][t] 
                        for h in range(self.params.H))
                    for t in range(self.params.DR_START, self.params.DR_END + 1)
                )
                new_UB = household_obj_sum + dr_cost
                UB = min(UB, new_UB)
                logger.info(f"  上界更新: {UB:.2f}")
            
            # Step 2: マスター問題を解く
            try:
                boundary_vars, LB = self.master.solve()
                logger.info(f"  下界: {LB:.2f}")
            except Exception as e:
                logger.error(f"マスター問題失敗: {e}")
                break
            
            # 収束判定
            gap = abs(UB - LB)
            logger.info(f"  ギャップ: {gap:.4f}")
            
            self.iteration_history.append({
                'iteration': k,
                'lower_bound': LB,
                'upper_bound': UB,
                'gap': gap,
                'time': time.time() - start_time
            })
            
            if gap < self.params.EPSILON:
                logger.info(f"\n収束しました！（反復{k}回）")
                break
        
        else:
            logger.warning(f"\n最大反復回数({self.params.MAX_ITER})に到達")
        
        # 最終結果の整理
        total_time = time.time() - start_time
        logger.info(f"\n計算時間: {total_time:.2f}秒")
        logger.info(f"最終ギャップ: {gap:.4f}")
        
        return {
            'boundary_vars': boundary_vars,
            'lower_bound': LB,
            'upper_bound': UB,
            'gap': gap,
            'iterations': k,
            'time': total_time,
            'history': self.iteration_history
        }
    
    def plot_convergence(self):
        """収束履歴のプロット"""
        if not self.iteration_history:
            return
        
        df = pd.DataFrame(self.iteration_history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 上界・下界の推移
        ax1.plot(df['iteration'], df['lower_bound'], 'b-', label='下界', linewidth=2)
        ax1.plot(df['iteration'], df['upper_bound'], 'r-', label='上界', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('目的関数値')
        ax1.set_title('GBD収束過程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ギャップの推移
        ax2.semilogy(df['iteration'], df['gap'], 'g-', linewidth=2)
        ax2.axhline(y=self.params.EPSILON, color='r', linestyle='--', 
                   label=f'収束閾値 ({self.params.EPSILON})')
        ax2.set_xlabel('反復回数')
        ax2.set_ylabel('ギャップ（対数スケール）')
        ax2.set_title('最適性ギャップ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gbd_convergence.png', dpi=150)
        plt.show()
        
    def export_results(self, results: Dict, filename: str = 'gbd_results.csv'):
        """結果のエクスポート"""
        # 各需要家の境界変数を整理
        data_list = []
        for h in range(self.params.H):
            for t in range(self.params.T):
                data_list.append({
                    'Household': h,
                    'Time': t,
                    'Voltage': results['boundary_vars'][h]['V_DN'][t],
                    'Active_Power': results['boundary_vars'][h]['P_DN'][t],
                    'Reactive_Power': results['boundary_vars'][h]['Q_DN'][t]
                })
        
        df = pd.DataFrame(data_list)
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"結果を{filename}に保存しました")


# ==================== メイン実行 ====================
def main():
    """メイン実行関数"""
    # GBDソルバーの初期化と実行
    solver = GBDSolver()
    
    # 最適化実行
    results = solver.solve()
    
    # 結果の可視化
    solver.plot_convergence()
    
    # 結果のエクスポート
    solver.export_results(results)
    
    # サマリー表示
    print("\n" + "="*60)
    print("最適化完了サマリー")
    print("="*60)
    print(f"反復回数: {results['iterations']}")
    print(f"計算時間: {results['time']:.2f}秒")
    print(f"最終下界: {results['lower_bound']:.2f}")
    print(f"最終上界: {results['upper_bound']:.2f}")
    print(f"最適性ギャップ: {results['gap']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()