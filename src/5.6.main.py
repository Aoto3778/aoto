"""
Generalized Benders Decomposition for Distribution Network Optimization
配電網最適化のための汎用ベンダーズ分解法 - データ読み込み統合版

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
    BUY_MAX: float = 5.0   # 買電上限[kW]
    SELL_MAX: float = 5.0  # 売電上限[kW]
    
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
    MAX_ITER: int = 1000     # 最大反復回数


class DataManager:
    """データ読み込みと管理"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.data = {}
        self.load_all_data()
        
    def load_all_data(self):
        """全データを読み込み"""
        T = self.params.T
        H = self.params.H
        
        # データ読み込み
        dmd_data = pd.read_csv("../data/input/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", encoding="shift_jis")
        buy_price_data = pd.read_csv("../data/input/buy_energy30.csv", encoding="shift_jis")
        spot_price_data = pd.read_csv("../data/input/spot_market_price_30min.csv", encoding="shift_jis")
        pv_data = pd.read_csv("../data/input/pv_output.csv")
        heat_dmd_data = pd.read_csv("../data/input/heat_demand1y.csv", encoding="shift_jis")
        temperature_outside_data = pd.read_csv("../data/input/temperature1y.csv", encoding="shift_jis")
        temperature_water_data = pd.read_csv("../data/input/water_temperature_30min.csv", encoding="shift_jis")
        voltage_estimate = pd.read_csv("../data/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis")
        
        # 時刻データ
        self.data['time'] = {t: dmd_data.iat[t-1, 0] for t in range(1, T+1)}
        
        # 太陽光出力[kW] - 全需要家同じ値を使用
        self.data['p_pv'] = {}
        for t in range(1, T+1):
            pv_value = float(pv_data.iat[t + 30*T, 2]) / 1000
            for h in range(H):
                self.data['p_pv'][h, t] = pv_value
        
        # 電力需要[kW] - 全需要家同じ値を使用（実際は需要家別のデータを使うべき）
        self.data['p_dmd'] = {}
        for t in range(1, T+1):
            demand_value = float(dmd_data.iat[t-1, 10])
            for h in range(H):
                self.data['p_dmd'][h, t] = demand_value
        
        # 買電単価[JPY/kWh]
        self.data['buy_price'] = {}
        for t in range(1, T+1):
            self.data['buy_price'][t] = float(buy_price_data.iat[t-1, 2])
        
        # スポット市場価格[JPY/kWh]
        self.data['spot_price'] = {}
        for t in range(1, T+1):
            self.data['spot_price'][t] = float(spot_price_data.iat[t, 2])
        
        # 熱需要[MJ] - 全需要家同じ値を使用
        self.data['H_dmd'] = {}
        for t in range(1, T+1):
            heat_value = float(heat_dmd_data.iat[t-1+T, 2])
            for h in range(H):
                self.data['H_dmd'][h, t] = heat_value
        
        # COP（成績係数）計算
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
        for h in range(H):
            self.data['linear_ap'][h] = voltage_estimate.iat[0, 3*h]
            self.data['linear_aq'][h] = voltage_estimate.iat[0, 3*h+1]
            self.data['linear_b'][h] = voltage_estimate.iat[0, 3*h+2]
            
        logger.info(f"データ読み込み完了: {T}タイムスロット, {H}需要家")


# ==================== サブ問題（需要家最適化） ====================
class HouseholdSubproblem:
    """需要家個別の最適化問題（プライバシー保護）"""
    
    def __init__(self, household_id: int, data_manager: DataManager, params: SystemParameters):
        self.h = household_id
        self.data_manager = data_manager
        self.params = params
        self.model = None
        self.setup_model()
        
    def setup_model(self):
        """最適化モデルの構築"""
        self.model = gp.Model(f"Household_{self.h}")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)
        
        T = self.params.T
        
        # === 決定変数 ===
        # 買電・売電
        self.P_buy = self.model.addVars(range(1, T+1), lb=0, ub=self.params.BUY_MAX, name="P_buy")
        self.P_sell = self.model.addVars(range(1, T+1), lb=0, ub=self.params.SELL_MAX, name="P_sell")
        self.delta_buy = self.model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_buy")
        self.delta_sell = self.model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_sell")
        
        # 蓄電池
        self.P_ch = self.model.addVars(range(1, T+1), lb=0, ub=self.params.N_B_PCS, name="P_charge")
        self.P_dch = self.model.addVars(range(1, T+1), lb=0, ub=self.params.N_B_PCS, name="P_discharge")
        self.E_bat = self.model.addVars(range(T+1), lb=0, ub=self.params.N_B, name="E_battery")
        self.delta_ch = self.model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_charge")
        self.delta_dch = self.model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_discharge")
        
        # ヒートポンプ
        self.P_hp = self.model.addVars(range(1, T+1), lb=0, ub=10.0, name="P_hp")
        self.H_prod = self.model.addVars(range(1, T+1), lb=0, name="H_produce")
        self.H_tank = self.model.addVars(range(T+1), lb=0, name="H_tank")
        self.delta_hp = self.model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_hp")
        self.delta_start = self.model.addVars(range(T+1), vtype=GRB.BINARY, name="delta_start")
        self.delta_stop = self.model.addVars(range(T+1), vtype=GRB.BINARY, name="delta_stop")
        
        # 境界変数
        self.V_H = self.model.addVars(range(1, T+1), lb=self.params.V_LL, ub=self.params.V_UL, name="V_H")
        self.P_H = self.model.addVars(range(1, T+1), lb=-10, ub=10, name="P_H")
        self.Q_H = self.model.addVars(range(1, T+1), lb=-5, ub=5, name="Q_H")
        
        self.model.update()
        
    def solve(self, boundary_fixed: Dict) -> Tuple[float, Dict, bool]:
        """
        境界変数を固定してサブ問題を解く
        """
        T = self.params.T
        
        # モデルをリセット
        self.model.reset()
        
        # 境界制約を一時的に追加
        temp_constrs = []
        
        for t in range(1, T+1):
            c1 = self.model.addConstr(
                self.V_H[t] == boundary_fixed['V_DN'][t-1],
                name=f"boundary_V_{t}"
            )
            c2 = self.model.addConstr(
                self.P_H[t] == boundary_fixed['P_DN'][t-1],
                name=f"boundary_P_{t}"
            )
            c3 = self.model.addConstr(
                self.Q_H[t] == boundary_fixed['Q_DN'][t-1],
                name=f"boundary_Q_{t}"
            )
            temp_constrs.extend([c1, c2, c3])
        
        # 需要家内制約の設定
        self._add_household_constraints()
        
        # 目的関数：電力コスト最小化
        obj = gp.quicksum(
            self.data_manager.data['buy_price'][t] * self.P_buy[t] - 
            self.params.SELL_PRICE * self.P_sell[t]
            for t in range(1, T+1)
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
            
        else:
            # 実行不可能な場合
            logger.debug(f"需要家{self.h}: 実行不可能")
            duals = self._solve_feasibility_subproblem(boundary_fixed)
            obj_val = 1e6
            feasible = False
        
        # 一時制約の削除
        for c in temp_constrs:
            self.model.remove(c)
        self.model.update()
        
        return obj_val, duals, feasible
    
    def _add_household_constraints(self):
        """需要家内の制約条件を追加"""
        T = self.params.T
        
        for t in range(1, T+1):
            # 電力バランス
            self.model.addConstr(
                self.data_manager.data['p_pv'][self.h, t] + self.P_buy[t] + self.P_dch[t] ==
                self.data_manager.data['p_dmd'][self.h, t] + self.P_sell[t] + 
                self.P_ch[t] + self.P_hp[t],
                name=f"power_balance_{t}"
            )
            
            # 買電売電同時禁止
            self.model.addConstr(self.P_buy[t] <= self.delta_buy[t] * self.params.BUY_MAX)
            self.model.addConstr(self.P_sell[t] <= self.delta_sell[t] * self.params.SELL_MAX)
            self.model.addConstr(self.delta_buy[t] + self.delta_sell[t] <= 1)
            
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
        self.model.addConstr(self.P_ch[t] <= self.params.N_B_PCS * self.delta_ch[t])
        self.model.addConstr(self.P_dch[t] <= self.params.N_B_PCS * self.delta_dch[t])
        self.model.addConstr(self.delta_ch[t] + self.delta_dch[t] <= 1)
        
        # SOC制約
        self.model.addConstr(self.E_bat[t] >= self.params.SOC_MIN * self.params.N_B)
        self.model.addConstr(self.E_bat[t] <= self.params.N_B)
        
        # SOC更新
        if t == 1:
            self.model.addConstr(
                self.E_bat[0] == self.params.SOC_INIT * self.params.N_B
            )
        
        self.model.addConstr(
            self.E_bat[t] == self.E_bat[t-1] + 
            0.5 * self.params.ETA_B * self.P_ch[t] -
            0.5 * self.P_dch[t] / self.params.ETA_B
        )
        
        # 終端条件
        if t == self.params.T:
            self.model.addConstr(
                self.E_bat[self.params.T] >= 0.4 * self.params.N_B
            )
    
    def _add_heatpump_constraints(self, t: int):
        """ヒートポンプ制約の追加"""
        cop = self.data_manager.data['cop'][t]
        
        # HP消費電力
        self.model.addConstr(
            self.P_hp[t] == self.params.XHP * self.delta_hp[t] +
            (self.H_prod[t] + self.params.R2 * self.params.CPH * self.delta_start[t]) / 
            (self.params.CE * cop)
        )
        
        # 熱製造量制約
        self.model.addConstr(
            self.H_prod[t] >= 0.5 * self.params.R1 * self.params.CPH * self.delta_hp[t]
        )
        self.model.addConstr(
            self.H_prod[t] <= 0.5 * self.params.CPH * self.delta_hp[t]
        )
        
        # 運転状態の遷移
        if t > 1:
            self.model.addConstr(
                self.delta_hp[t] - self.delta_hp[t-1] == self.delta_start[t] - self.delta_stop[t]
            )
        else:
            self.model.addConstr(self.delta_hp[1] == self.delta_start[1])
            self.model.addConstr(self.delta_stop[0] == 0)
            self.model.addConstr(self.delta_start[0] == 0)
        
        # 起動・停止同時禁止
        self.model.addConstr(self.delta_start[t] + self.delta_stop[t] <= 1)
        
        # 貯湯槽更新
        if t == 1:
            self.model.addConstr(
                self.H_tank[0] == 0.5 * self.params.CW * self.params.VLT * 
                self.params.TANK_MAX_TEMP
            )
        
        self.model.addConstr(
            self.H_tank[t] == self.H_tank[t-1] + 
            self.H_prod[t] - self.data_manager.data['H_dmd'][self.h, t]
        )
        
        # 貯湯槽容量制約
        self.model.addConstr(
            self.H_tank[t] >= 0.2 * self.params.CW * self.params.VLT * 
            self.params.TANK_MAX_TEMP
        )
        self.model.addConstr(
            self.H_tank[t] <= self.params.CW * self.params.VLT * 
            self.params.TANK_MAX_TEMP
        )
        
        # 終端条件
        if t == self.params.T:
            self.model.addConstr(
                self.H_tank[self.params.T] >= 0.4 * self.params.CW * 
                self.params.VLT * self.params.TANK_MAX_TEMP
            )
    
    def _solve_feasibility_subproblem(self, boundary_fixed: Dict) -> Dict:
        """実行可能性サブ問題を解く（簡略版）"""
        T = self.params.T
        
        # 簡略化した双対変数を返す
        duals = {
            'mu_V': [1.0] * T,
            'mu_P': [1.0] * T,
            'mu_Q': [0.1] * T
        }
        
        return duals


# ==================== マスター問題（アグリゲータ最適化） ====================
class AggregatorMasterProblem:
    """配電網運用の最適化（アグリゲータ）"""
    
    def __init__(self, data_manager: DataManager, params: SystemParameters):
        self.data_manager = data_manager
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
        
        # 境界変数（各需要家との接続点）
        self.V_DN = {}
        self.P_DN = {}
        self.Q_DN = {}
        
        for h in range(H):
            self.V_DN[h] = self.model.addVars(
                range(1, T+1), lb=self.params.V_LL, ub=self.params.V_UL, name=f"V_DN_{h}"
            )
            self.P_DN[h] = self.model.addVars(
                range(1, T+1), lb=-10, ub=10, name=f"P_DN_{h}"
            )
            self.Q_DN[h] = self.model.addVars(
                range(1, T+1), lb=-5, ub=5, name=f"Q_DN_{h}"
            )
        
        # アグリゲータの系統買電
        self.P_grid = self.model.addVars(range(1, T+1), lb=-20, ub=20, name="P_grid")
        
        # 下界補助変数
        self.LBD = self.model.addVars(H, lb=-1e6, ub=1e6, name="LBD")
        
        self.model.update()
        
    def add_network_constraints(self):
        """配電網制約の追加"""
        T, H = self.params.T, self.params.H
        
        for t in range(1, T+1):
            # 電圧降下の推定
            for h in range(H):
                # 累積有効電力（下流の需要家含む）
                P_cumulative = gp.quicksum(self.P_DN[j][t] for j in range(h, H))
                Q_cumulative = gp.quicksum(self.Q_DN[j][t] for j in range(h, H))
                
                # 電圧降下
                alpha_p = self.data_manager.data['linear_ap'][h]
                alpha_q = self.data_manager.data['linear_aq'][h]
                beta = self.data_manager.data['linear_b'][h]
                
                delta_V = alpha_p * P_cumulative + alpha_q * Q_cumulative + beta
                
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
                    (self.params.S_LINE/100) * (self.params.S_LINE/100),
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
        
        # モデルリセット
        self.model.reset()
        
        # 配電網制約の追加
        self.add_network_constraints()
        
        # カット平面制約の追加
        for idx, cut in enumerate(self.optimality_cuts):
            h = cut['household']
            self.model.addConstr(
                self.LBD[h] >= cut['constant'] +
                gp.quicksum(
                    cut['mu_V'][t-1] * self.V_DN[h][t] +
                    cut['mu_P'][t-1] * self.P_DN[h][t] +
                    cut['mu_Q'][t-1] * self.Q_DN[h][t]
                    for t in range(1, T+1)
                ),
                name=f"opt_cut_{h}_{idx}"
            )
        
        for idx, cut in enumerate(self.feasibility_cuts):
            h = cut['household']
            self.model.addConstr(
                0 >= cut['constant'] +
                gp.quicksum(
                    cut['mu_V'][t-1] * self.V_DN[h][t] +
                    cut['mu_P'][t-1] * self.P_DN[h][t] +
                    cut['mu_Q'][t-1] * self.Q_DN[h][t]
                    for t in range(1, T+1)
                ),
                name=f"feas_cut_{h}_{idx}"
            )
        
        # 目的関数
        obj = gp.quicksum(self.LBD[h] for h in range(H))
        
        # DR期間の系統買電ペナルティ
        if self.params.DR_END <= T:
            obj += 10.0 * gp.quicksum(
                self.P_grid[t] 
                for t in range(self.params.DR_START, self.params.DR_END + 1)
            )
        
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        # 最適化実行
        self.model.optimize()
        
        # 境界変数の値を取得
        boundary_vars = {}
        for h in range(H):
            boundary_vars[h] = {
                'V_DN': [self.V_DN[h][t].X for t in range(1, T+1)],
                'P_DN': [self.P_DN[h][t].X for t in range(1, T+1)],
                'Q_DN': [self.Q_DN[h][t].X for t in range(1, T+1)]
            }
        
        lower_bound = self.model.ObjVal
        return boundary_vars, lower_bound
    
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
            'mu_V': duals['mu_V'],
            'mu_P': duals['mu_P'],
            'mu_Q': duals['mu_Q']
        }
        self.feasibility_cuts.append(cut)


# ==================== GBDソルバー ====================
class GBDSolver:
    """Generalized Benders Decomposition統合ソルバー"""
    
    def __init__(self):
        self.params = SystemParameters()
        self.data_manager = DataManager(self.params)
        self.master = AggregatorMasterProblem(self.data_manager, self.params)
        self.subproblems = [
            HouseholdSubproblem(h, self.data_manager, self.params)
            for h in range(self.params.H)
        ]
        
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
        
        # 初期境界変数
        boundary_vars = {}
        for h in range(self.params.H):
            boundary_vars[h] = {
                'V_DN': [self.params.V_BASE + h*0.1] * self.params.T,
                'P_DN': [0.5] * self.params.T,
                'Q_DN': [0.05] * self.params.T
            }
        
        start_time = time.time()
        
        while k < self.params.MAX_ITER:
            k += 1
            logger.info(f"\n反復 {k}:")
            
            # Step 1: 各需要家のサブ問題を解く
            subproblem_results = []
            household_obj_sum = 0
            all_feasible = True
            
            for h in range(self.params.H):
                try:
                    obj_val, duals, feasible = self.subproblems[h].solve(boundary_vars[h])
                    subproblem_results.append((h, obj_val, duals, feasible))
                    
                    if feasible:
                        household_obj_sum += obj_val
                        self.master.add_optimality_cut(h, obj_val, duals)
                        logger.info(f"  需要家{h}: 実行可能 (obj={obj_val:.2f})")
                    else:
                        all_feasible = False
                        self.master.add_feasibility_cut(h, duals)
                        logger.info(f"  需要家{h}: 実行不可能")
                        
                except Exception as e:
                    logger.error(f"  需要家{h}でエラー: {e}")
                    all_feasible = False
            
            # 上界の更新
            if all_feasible:
                # DR期間の系統買電を計算
                if self.params.DR_END <= self.params.T:
                    dr_cost = 10.0 * sum(
                        sum(boundary_vars[h]['P_DN'][t] 
                            for h in range(self.params.H))
                        for t in range(self.params.DR_START-1, 
                                     min(self.params.DR_END, self.params.T-1))
                    )
                else:
                    dr_cost = 0
                    
                new_UB = household_obj_sum + dr_cost
                if new_UB < UB:
                    UB = new_UB
                    logger.info(f"  上界更新: {UB:.2f}")
            
            # Step 2: マスター問題を解く
            try:
                boundary_vars, LB = self.master.solve()
                logger.info(f"  下界: {LB:.2f}")
            except Exception as e:
                logger.error(f"マスター問題失敗: {e}")
                break
            
            # 収束判定
            if UB < float('inf'):
                gap = abs(UB - LB) / (abs(UB) + 1e-6)
            else:
                gap = float('inf')
                
            logger.info(f"  相対ギャップ: {gap:.4f}")
            
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
        
        # 最終結果
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
        valid_ub = df[df['upper_bound'] < float('inf')]
        if not valid_ub.empty:
            ax1.plot(valid_ub['iteration'], valid_ub['upper_bound'], 'r-', 
                    label='上界', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('目的関数値')
        ax1.set_title('GBD収束過程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ギャップの推移
        valid_gap = df[df['gap'] < float('inf')]
        if not valid_gap.empty:
            ax2.semilogy(valid_gap['iteration'], valid_gap['gap'], 'g-', linewidth=2)
            ax2.axhline(y=self.params.EPSILON, color='r', linestyle='--', 
                       label=f'収束閾値 ({self.params.EPSILON})')
            ax2.set_xlabel('反復回数')
            ax2.set_ylabel('相対ギャップ（対数スケール）')
            ax2.set_title('最適性ギャップ')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gbd_convergence.png', dpi=150)
        logger.info("収束グラフをgbd_convergence.pngに保存しました")


# ==================== メイン実行 ====================
def main():
    """メイン実行関数"""
    try:
        # GBDソルバーの初期化と実行
        solver = GBDSolver()
        
        # 最適化実行
        results = solver.solve()
        
        # 結果の可視化
        solver.plot_convergence()
        
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
        
    except Exception as e:
        logger.error(f"実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()