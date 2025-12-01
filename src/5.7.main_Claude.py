"""
Generalized Benders Decomposition (GBD) for Distribution Network
配電網の分散型最適化

マスター問題：アグリゲータ（配電網運用 + 境界変数決定）
サブ問題：各需要家（電力コスト最小化）
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import logging
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimalityCut:
    """最適カット平面のデータ構造"""
    household: int
    iteration: int
    UB_h: float  # サブ問題の目的関数値
    mu_V: Dict[int, float] = field(default_factory=dict)  # 電圧の双対変数
    mu_P: Dict[int, float] = field(default_factory=dict)  # 有効電力の双対変数
    mu_Q: Dict[int, float] = field(default_factory=dict)  # 無効電力の双対変数
    y_hat_V: Dict[int, float] = field(default_factory=dict)  # 固定した電圧値
    y_hat_P: Dict[int, float] = field(default_factory=dict)  # 固定した有効電力値
    y_hat_Q: Dict[int, float] = field(default_factory=dict)  # 固定した無効電力値


@dataclass
class FeasibilityCut:
    """実行可能カット平面のデータ構造"""
    household: int
    iteration: int
    lambda_V: Dict[int, float] = field(default_factory=dict)
    lambda_P: Dict[int, float] = field(default_factory=dict)
    lambda_Q: Dict[int, float] = field(default_factory=dict)
    x_H_V: Dict[int, float] = field(default_factory=dict)  # サブ問題のV_H最適値
    x_H_P: Dict[int, float] = field(default_factory=dict)  # サブ問題のP_H最適値
    x_H_Q: Dict[int, float] = field(default_factory=dict)  # サブ問題のQ_H最適値


class SystemParameters:
    """システムパラメータの管理"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        
        # システム定数
        self.T = 48  # タイムスロット数（30分粒度）
        self.H = 3   # 需要家数
        
        # 電圧制約
        self.V_BASE = 99.8  # 基準電圧[V]
        self.V_UL = 107.0   # 電圧上限[V]
        self.V_LL = 95.0    # 電圧下限[V]
        
        # 送電線容量
        self.S_LINE = 101 * 20  # 定格電力[VA]
        
        # 蓄電池パラメータ
        self.N_B = 6.3        # 蓄電池容量[kWh]
        self.N_B_PCS = 1.5    # PCS容量[kW]
        self.ETA_B = 0.95     # 充放電効率
        
        # ヒートポンプパラメータ
        self.VLT = 370.0      # 貯湯槽容量[L]
        self.CW = 0.0042      # 水の比熱[MJ/L/℃]
        self.CE = 3.6         # 変換係数
        self.CPH = 16.2       # HP加熱能力[MJ]
        self.XHP = 0.013      # 補器消費電力[kWh]
        self.R1 = 0.05        # 最低熱製造率
        self.R2 = 0.1         # 起動時ロス率
        
        # 料金設定
        self.BUY_MAX = 5.0    # 買電上限[kW]
        self.SELL_MAX = 5.0   # 売電上限[kW]
        self.SELL_PRICE = 2.0 # 売電単価[JPY/kWh]
        
        # 下げDR期間（13時～16時）
        self.DR_START = 26  # 13:00 (slot 26)
        self.DR_END = 32    # 16:00 (slot 32)
        
        # データ読み込み
        self.load_data()
        
    def load_data(self):
        """データ読み込み"""
        logger.info("データ読み込み開始...")
        
        # CSVファイル読み込み
        dmd_data = pd.read_csv(f"{self.data_dir}/input/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", 
                               encoding="shift_jis")
        buy_price_data = pd.read_csv(f"{self.data_dir}/input/buy_energy30.csv", encoding="shift_jis")
        pv_data = pd.read_csv(f"{self.data_dir}/input/pv_output.csv")
        heat_dmd_data = pd.read_csv(f"{self.data_dir}/input/heat_demand1y.csv", encoding="shift_jis")
        temperature_outside_data = pd.read_csv(f"{self.data_dir}/input/temperature1y.csv", encoding="shift_jis")
        temperature_water_data = pd.read_csv(f"{self.data_dir}/input/water_temperature_30min.csv", encoding="shift_jis")
        voltage_estimate = pd.read_csv(f"{self.data_dir}/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis")
        
        # 太陽光出力[kW]
        self.p_pv = {}
        for t in range(1, self.T+1):
            pv_value = float(pv_data.iat[t + 30*self.T, 2]) / 1000
            for h in range(self.H):
                self.p_pv[t, h] = pv_value
        
        # 電力需要[kW]
        self.p_dmd = {}
        for t in range(1, self.T+1):
            for h in range(self.H):
                self.p_dmd[t, h] = float(dmd_data.iat[t-1, 10])
        
        # 買電単価[JPY/kWh]
        self.buy_price = {}
        for t in range(1, self.T+1):
            self.buy_price[t] = float(buy_price_data.iat[t-1, 2])
        
        # 熱需要[MJ]
        self.heat_dmd = {}
        for t in range(1, self.T+1):
            for h in range(self.H):
                self.heat_dmd[t, h] = float(heat_dmd_data.iat[t-1+self.T, 2])
        
        # COP（成績係数）
        self.cop = {}
        for t in range(1, self.T+1):
            tmf = float(temperature_water_data.iat[t-1, 2])
            tma = float(temperature_outside_data.iat[t, 2])
            
            if tma >= 5:
                k = 1
            elif tma > 2:
                k = tma/30 + 0.8333
            else:
                k = 0.9
            
            self.cop[t] = k * (0.175*tma - 0.1322*tmf + 4.076)
        
        # 電圧推定パラメータ
        self.linear_ap = {}
        self.linear_aq = {}
        self.linear_b = {}
        for h in range(self.H):
            self.linear_ap[h] = voltage_estimate.iat[0, 3*h]
            self.linear_aq[h] = voltage_estimate.iat[0, 3*h+1]
            self.linear_b[h] = voltage_estimate.iat[0, 3*h+2]
        
        logger.info(f"データ読み込み完了: {self.T}タイムスロット, {self.H}需要家")


class Subproblem:
    """
    需要家hのサブ問題
    
    境界変数 y_DN_hat を固定として、需要家の電力コストを最小化
    """
    
    def __init__(self, household_id: int, params: SystemParameters):
        self.h = household_id
        self.params = params
        self.T = params.T
        self.model = None
        
    def build_model(self, y_DN_hat: Dict[str, Dict[int, float]]):
        """
        サブ問題モデルの構築
        
        Args:
            y_DN_hat: 固定された境界変数
                {'V': {t: value}, 'P': {t: value}, 'Q': {t: value}}
        """
        self.model = gp.Model(f"Subproblem_H{self.h}")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)  # 二次制約の双対変数取得
        
        p = self.params
        h = self.h
        
        # ========== 決定変数 ==========
        # 買電・売電
        self.P_buy = self.model.addVars(range(1, self.T+1), lb=0, ub=p.BUY_MAX, name="P_buy")
        self.P_sell = self.model.addVars(range(1, self.T+1), lb=0, ub=p.SELL_MAX, name="P_sell")
        self.delta_buy = self.model.addVars(range(1, self.T+1), vtype=GRB.BINARY, name="delta_buy")
        self.delta_sell = self.model.addVars(range(1, self.T+1), vtype=GRB.BINARY, name="delta_sell")
        
        # 蓄電池
        self.P_ch = self.model.addVars(range(1, self.T+1), lb=0, ub=p.N_B_PCS, name="P_ch")
        self.P_dch = self.model.addVars(range(1, self.T+1), lb=0, ub=p.N_B_PCS, name="P_dch")
        self.E_bat = self.model.addVars(range(1, self.T+1), lb=0.2*p.N_B, ub=p.N_B, name="E_bat")
        self.delta_ch = self.model.addVars(range(1, self.T+1), vtype=GRB.BINARY, name="delta_ch")
        self.delta_dch = self.model.addVars(range(1, self.T+1), vtype=GRB.BINARY, name="delta_dch")
        
        # ヒートポンプ
        self.P_hp = self.model.addVars(range(1, self.T+1), lb=0, ub=10.0, name="P_hp")
        self.H_prod = self.model.addVars(range(1, self.T+1), lb=0, name="H_prod")
        self.H_tank = self.model.addVars(range(self.T+1), lb=0, name="H_tank")
        self.delta_hp = self.model.addVars(range(self.T+1), vtype=GRB.BINARY, name="delta_hp")
        self.delta_start = self.model.addVars(range(self.T+1), vtype=GRB.BINARY, name="delta_start")
        self.delta_stop = self.model.addVars(range(self.T+1), vtype=GRB.BINARY, name="delta_stop")
        
        # 境界変数（H側）
        self.P_H = self.model.addVars(range(1, self.T+1), lb=-GRB.INFINITY, name="P_H")
        self.Q_H = self.model.addVars(range(1, self.T+1), lb=-GRB.INFINITY, name="Q_H")
        self.V_H = self.model.addVars(range(1, self.T+1), lb=p.V_LL, ub=p.V_UL, name="V_H")
        
        self.model.update()
        
        # ========== 制約条件 ==========
        self._add_power_balance_constraints()
        self._add_battery_constraints()
        self._add_heatpump_constraints()
        self._add_boundary_constraints(y_DN_hat)
        
        # ========== 目的関数 ==========
        self._set_objective()
        
    def _add_power_balance_constraints(self):
        """電力バランス制約"""
        p = self.params
        h = self.h
        
        for t in range(1, self.T+1):
            # 電力バランス
            self.model.addConstr(
                p.p_pv[t, h] + self.P_buy[t] + self.P_dch[t] ==
                p.p_dmd[t, h] + self.P_sell[t] + self.P_ch[t] + self.P_hp[t],
                name=f"power_balance_{t}"
            )
            
            # 買電売電同時禁止
            self.model.addConstr(self.P_buy[t] <= self.delta_buy[t] * p.BUY_MAX)
            self.model.addConstr(self.P_sell[t] <= self.delta_sell[t] * p.SELL_MAX)
            self.model.addConstr(self.delta_buy[t] + self.delta_sell[t] <= 1)
            
            # P_H, Q_Hの定義
            self.model.addConstr(
                self.P_H[t] == self.P_buy[t] - self.P_sell[t],
                name=f"P_H_def_{t}"
            )
            self.model.addConstr(
                self.Q_H[t] == 0.1 * (self.P_buy[t] - self.P_sell[t]),
                name=f"Q_H_def_{t}"
            )
    
    def _add_battery_constraints(self):
        """蓄電池制約"""
        p = self.params
        
        for t in range(1, self.T+1):
            # 充放電制約
            self.model.addConstr(self.P_ch[t] <= p.N_B_PCS * self.delta_ch[t])
            self.model.addConstr(self.P_dch[t] <= p.N_B_PCS * self.delta_dch[t])
            self.model.addConstr(self.delta_ch[t] + self.delta_dch[t] <= 1)
            
            # SOC更新
            if t == 1:
                prev_soc = 0.5 * p.N_B
            else:
                prev_soc = self.E_bat[t-1]
            
            self.model.addConstr(
                self.E_bat[t] == prev_soc + 
                0.5 * p.ETA_B * self.P_ch[t] - 
                0.5 * self.P_dch[t] / p.ETA_B,
                name=f"battery_soc_{t}"
            )
        
        # 終端条件
        self.model.addConstr(self.E_bat[self.T] == 0.5 * p.N_B, name="battery_final")
    
    def _add_heatpump_constraints(self):
        """ヒートポンプ制約"""
        p = self.params
        h = self.h
        
        # 初期条件
        self.model.addConstr(self.delta_hp[0] == 0)
        self.model.addConstr(self.delta_start[0] == 0)
        self.model.addConstr(self.delta_stop[0] == 0)
        self.model.addConstr(
            self.H_tank[0] == 0.5 * p.CW * p.VLT * 70,
            name="hp_tank_init"
        )
        
        for t in range(1, self.T+1):
            cop = p.cop[t]
            
            # HP消費電力
            self.model.addConstr(
                self.P_hp[t] == p.XHP * self.delta_hp[t] +
                (self.H_prod[t] + p.R2 * p.CPH * self.delta_start[t]) / (p.CE * cop),
                name=f"hp_power_{t}"
            )
            
            # 熱製造量制約
            self.model.addConstr(self.H_prod[t] >= 0.5 * p.R1 * p.CPH * self.delta_hp[t])
            self.model.addConstr(self.H_prod[t] <= 0.5 * p.CPH * self.delta_hp[t])
            
            # 運転状態の遷移
            self.model.addConstr(
                self.delta_hp[t] - self.delta_hp[t-1] == 
                self.delta_start[t] - self.delta_stop[t],
                name=f"hp_state_{t}"
            )
            
            # 起動停止同時禁止
            self.model.addConstr(self.delta_start[t] + self.delta_stop[t] <= 1)
            
            # 貯湯槽更新
            self.model.addConstr(
                self.H_tank[t] == self.H_tank[t-1] + self.H_prod[t] - p.heat_dmd[t, h],
                name=f"hp_tank_{t}"
            )
            
            # 貯湯槽容量制約
            self.model.addConstr(self.H_tank[t] >= 0.2 * p.CW * p.VLT * 70)
            self.model.addConstr(self.H_tank[t] <= p.CW * p.VLT * 70)
        
        # 終端条件
        self.model.addConstr(
            self.H_tank[self.T] >= 0.5 * p.CW * p.VLT * 70,
            name="hp_tank_final"
        )
    
    def _add_boundary_constraints(self, y_DN_hat: Dict[str, Dict[int, float]]):
        """
        境界等式制約
        H側の変数 = DN側の固定値
        """
        self.boundary_constrs_V = {}
        self.boundary_constrs_P = {}
        self.boundary_constrs_Q = {}
        
        for t in range(1, self.T+1):
            # V_H = V_DN_hat
            self.boundary_constrs_V[t] = self.model.addConstr(
                self.V_H[t] == y_DN_hat['V'][t],
                name=f"boundary_V_{t}"
            )
            
            # P_H = P_DN_hat
            self.boundary_constrs_P[t] = self.model.addConstr(
                self.P_H[t] == y_DN_hat['P'][t],
                name=f"boundary_P_{t}"
            )
            
            # Q_H = Q_DN_hat
            self.boundary_constrs_Q[t] = self.model.addConstr(
                self.Q_H[t] == y_DN_hat['Q'][t],
                name=f"boundary_Q_{t}"
            )
    
    def _set_objective(self):
        """目的関数：電力コスト最小化"""
        p = self.params
        
        cost = gp.quicksum(
            p.buy_price[t] * self.P_buy[t] - p.SELL_PRICE * self.P_sell[t]
            for t in range(1, self.T+1)
        )
        
        self.model.setObjective(cost, GRB.MINIMIZE)
    
    def solve(self, y_DN_hat: Dict[str, Dict[int, float]]) -> Tuple[bool, Optional[float], Optional[Dict]]:
        """
        サブ問題を解く
        
        Returns:
            (feasible, objective_value, dual_variables)
        """
        self.build_model(y_DN_hat)
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            obj_val = self.model.ObjVal
            
            # 双対変数の取得
            duals = {
                'mu_V': {},
                'mu_P': {},
                'mu_Q': {},
                'V_H': {},
                'P_H': {},
                'Q_H': {}
            }
            
            for t in range(1, self.T+1):
                # 境界等式制約の双対変数
                duals['mu_V'][t] = self.boundary_constrs_V[t].Pi
                duals['mu_P'][t] = self.boundary_constrs_P[t].Pi
                duals['mu_Q'][t] = self.boundary_constrs_Q[t].Pi
                
                # H側変数の値
                duals['V_H'][t] = self.V_H[t].X
                duals['P_H'][t] = self.P_H[t].X
                duals['Q_H'][t] = self.Q_H[t].X
            
            return True, obj_val, duals
        
        elif self.model.Status == GRB.INFEASIBLE:
            logger.warning(f"Subproblem H{self.h} is infeasible")
            return False, None, None
        
        else:
            logger.error(f"Subproblem H{self.h} failed with status {self.model.Status}")
            return False, None, None
    
    def get_solution(self) -> Dict:
        """解の詳細を取得"""
        if self.model.Status != GRB.OPTIMAL:
            return None
        
        solution = {
            'P_buy': {t: self.P_buy[t].X for t in range(1, self.T+1)},
            'P_sell': {t: self.P_sell[t].X for t in range(1, self.T+1)},
            'P_ch': {t: self.P_ch[t].X for t in range(1, self.T+1)},
            'P_dch': {t: self.P_dch[t].X for t in range(1, self.T+1)},
            'E_bat': {t: self.E_bat[t].X for t in range(1, self.T+1)},
            'P_hp': {t: self.P_hp[t].X for t in range(1, self.T+1)},
            'H_tank': {t: self.H_tank[t].X for t in range(0, self.T+1)},
            'H_prod': {t: self.H_prod[t].X for t in range(1, self.T+1)},
            'delta_hp': {t: self.delta_hp[t].X for t in range(0, self.T+1)},
            'P_H': {t: self.P_H[t].X for t in range(1, self.T+1)},
            'Q_H': {t: self.Q_H[t].X for t in range(1, self.T+1)},
            'V_H': {t: self.V_H[t].X for t in range(1, self.T+1)}
        }
        
        return solution


class FeasibilitySubproblem:
    """
    実行可能性サブ問題
    
    サブ問題が実行不可能な場合に、緩和変数を導入して実行可能カットを生成
    """
    
    def __init__(self, household_id: int, params: SystemParameters):
        self.h = household_id
        self.params = params
        self.T = params.T
        self.model = None
    
    def build_model(self, y_DN_hat: Dict[str, Dict[int, float]]):
        """実行可能性問題の構築"""
        self.model = gp.Model(f"FeasibilitySubproblem_H{self.h}")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)
        
        p = self.params
        h = self.h
        
        # ========== 決定変数（サブ問題と同様 + 緩和変数） ==========
        # 買電・売電（バイナリを緩和）
        self.P_buy = self.model.addVars(range(1, self.T+1), lb=0, ub=p.BUY_MAX, name="P_buy")
        self.P_sell = self.model.addVars(range(1, self.T+1), lb=0, ub=p.SELL_MAX, name="P_sell")
        self.delta_buy = self.model.addVars(range(1, self.T+1), lb=0, ub=1, name="delta_buy")  # 緩和
        self.delta_sell = self.model.addVars(range(1, self.T+1), lb=0, ub=1, name="delta_sell")  # 緩和
        
        # 蓄電池（バイナリを緩和）
        self.P_ch = self.model.addVars(range(1, self.T+1), lb=0, ub=p.N_B_PCS, name="P_ch")
        self.P_dch = self.model.addVars(range(1, self.T+1), lb=0, ub=p.N_B_PCS, name="P_dch")
        self.E_bat = self.model.addVars(range(1, self.T+1), lb=0.2*p.N_B, ub=p.N_B, name="E_bat")
        self.delta_ch = self.model.addVars(range(1, self.T+1), lb=0, ub=1, name="delta_ch")  # 緩和
        self.delta_dch = self.model.addVars(range(1, self.T+1), lb=0, ub=1, name="delta_dch")  # 緩和
        
        # ヒートポンプ（バイナリを緩和）
        self.P_hp = self.model.addVars(range(1, self.T+1), lb=0, ub=10.0, name="P_hp")
        self.H_prod = self.model.addVars(range(1, self.T+1), lb=0, name="H_prod")
        self.H_tank = self.model.addVars(range(self.T+1), lb=0, name="H_tank")
        self.delta_hp = self.model.addVars(range(self.T+1), lb=0, ub=1, name="delta_hp")  # 緩和
        self.delta_start = self.model.addVars(range(self.T+1), lb=0, ub=1, name="delta_start")  # 緩和
        self.delta_stop = self.model.addVars(range(self.T+1), lb=0, ub=1, name="delta_stop")  # 緩和
        
        # 境界変数（H側）
        self.P_H = self.model.addVars(range(1, self.T+1), lb=-GRB.INFINITY, name="P_H")
        self.Q_H = self.model.addVars(range(1, self.T+1), lb=-GRB.INFINITY, name="Q_H")
        self.V_H = self.model.addVars(range(1, self.T+1), lb=p.V_LL, ub=p.V_UL, name="V_H")
        
        # 緩和変数（境界制約用）
        self.alpha = self.model.addVars(range(1, self.T+1), 6, lb=0, name="alpha")
        
        self.model.update()
        
        # ========== 制約条件 ==========
        self._add_power_balance_constraints()
        self._add_battery_constraints()
        self._add_heatpump_constraints()
        self._add_relaxed_boundary_constraints(y_DN_hat)
        
        # ========== 目的関数：緩和変数の最小化 ==========
        self.model.setObjective(
            gp.quicksum(self.alpha[t, i] for t in range(1, self.T+1) for i in range(6)),
            GRB.MINIMIZE
        )
    
    def _add_power_balance_constraints(self):
        """電力バランス制約"""
        p = self.params
        h = self.h
        
        for t in range(1, self.T+1):
            self.model.addConstr(
                p.p_pv[t, h] + self.P_buy[t] + self.P_dch[t] ==
                p.p_dmd[t, h] + self.P_sell[t] + self.P_ch[t] + self.P_hp[t],
                name=f"power_balance_{t}"
            )
            
            self.model.addConstr(self.P_buy[t] <= self.delta_buy[t] * p.BUY_MAX)
            self.model.addConstr(self.P_sell[t] <= self.delta_sell[t] * p.SELL_MAX)
            self.model.addConstr(self.delta_buy[t] + self.delta_sell[t] <= 1)
            
            self.model.addConstr(self.P_H[t] == self.P_buy[t] - self.P_sell[t], name=f"P_H_def_{t}")
            self.model.addConstr(self.Q_H[t] == 0.1 * (self.P_buy[t] - self.P_sell[t]), name=f"Q_H_def_{t}")
    
    def _add_battery_constraints(self):
        """蓄電池制約"""
        p = self.params
        
        for t in range(1, self.T+1):
            self.model.addConstr(self.P_ch[t] <= p.N_B_PCS * self.delta_ch[t])
            self.model.addConstr(self.P_dch[t] <= p.N_B_PCS * self.delta_dch[t])
            self.model.addConstr(self.delta_ch[t] + self.delta_dch[t] <= 1)
            
            if t == 1:
                prev_soc = 0.5 * p.N_B
            else:
                prev_soc = self.E_bat[t-1]
            
            self.model.addConstr(
                self.E_bat[t] == prev_soc + 
                0.5 * p.ETA_B * self.P_ch[t] - 
                0.5 * self.P_dch[t] / p.ETA_B,
                name=f"battery_soc_{t}"
            )
        
        self.model.addConstr(self.E_bat[self.T] == 0.5 * p.N_B, name="battery_final")
    
    def _add_heatpump_constraints(self):
        """ヒートポンプ制約"""
        p = self.params
        h = self.h
        
        self.model.addConstr(self.delta_hp[0] == 0)
        self.model.addConstr(self.delta_start[0] == 0)
        self.model.addConstr(self.delta_stop[0] == 0)
        self.model.addConstr(self.H_tank[0] == 0.5 * p.CW * p.VLT * 70, name="hp_tank_init")
        
        for t in range(1, self.T+1):
            cop = p.cop[t]
            
            self.model.addConstr(
                self.P_hp[t] == p.XHP * self.delta_hp[t] +
                (self.H_prod[t] + p.R2 * p.CPH * self.delta_start[t]) / (p.CE * cop),
                name=f"hp_power_{t}"
            )
            
            self.model.addConstr(self.H_prod[t] >= 0.5 * p.R1 * p.CPH * self.delta_hp[t])
            self.model.addConstr(self.H_prod[t] <= 0.5 * p.CPH * self.delta_hp[t])
            
            self.model.addConstr(
                self.delta_hp[t] - self.delta_hp[t-1] == self.delta_start[t] - self.delta_stop[t],
                name=f"hp_state_{t}"
            )
            
            self.model.addConstr(self.delta_start[t] + self.delta_stop[t] <= 1)
            
            self.model.addConstr(
                self.H_tank[t] == self.H_tank[t-1] + self.H_prod[t] - p.heat_dmd[t, h],
                name=f"hp_tank_{t}"
            )
            
            self.model.addConstr(self.H_tank[t] >= 0.2 * p.CW * p.VLT * 70)
            self.model.addConstr(self.H_tank[t] <= p.CW * p.VLT * 70)
        
        self.model.addConstr(self.H_tank[self.T] >= 0.5 * p.CW * p.VLT * 70, name="hp_tank_final")
    
    def _add_relaxed_boundary_constraints(self, y_DN_hat: Dict[str, Dict[int, float]]):
        """緩和された境界制約"""
        self.boundary_ineq_constrs = {}
        
        for t in range(1, self.T+1):
            V_hat = y_DN_hat['V'][t]
            P_hat = y_DN_hat['P'][t]
            Q_hat = y_DN_hat['Q'][t]
            
            # V_H - V_DN_hat - alpha_1 <= 0 (インデックス0)
            self.boundary_ineq_constrs[t, 0] = self.model.addConstr(
                self.V_H[t] - V_hat - self.alpha[t, 0] <= 0,
                name=f"boundary_V_upper_{t}"
            )
            # -V_H + V_DN_hat - alpha_2 <= 0 (インデックス1)
            self.boundary_ineq_constrs[t, 1] = self.model.addConstr(
                -self.V_H[t] + V_hat - self.alpha[t, 1] <= 0,
                name=f"boundary_V_lower_{t}"
            )
            
            # P_H - P_DN_hat - alpha_3 <= 0 (インデックス2)
            self.boundary_ineq_constrs[t, 2] = self.model.addConstr(
                self.P_H[t] - P_hat - self.alpha[t, 2] <= 0,
                name=f"boundary_P_upper_{t}"
            )
            # -P_H + P_DN_hat - alpha_4 <= 0 (インデックス3)
            self.boundary_ineq_constrs[t, 3] = self.model.addConstr(
                -self.P_H[t] + P_hat - self.alpha[t, 3] <= 0,
                name=f"boundary_P_lower_{t}"
            )
            
            # Q_H - Q_DN_hat - alpha_5 <= 0 (インデックス4)
            self.boundary_ineq_constrs[t, 4] = self.model.addConstr(
                self.Q_H[t] - Q_hat - self.alpha[t, 4] <= 0,
                name=f"boundary_Q_upper_{t}"
            )
            # -Q_H + Q_DN_hat - alpha_6 <= 0 (インデックス5)
            self.boundary_ineq_constrs[t, 5] = self.model.addConstr(
                -self.Q_H[t] + Q_hat - self.alpha[t, 5] <= 0,
                name=f"boundary_Q_lower_{t}"
            )
    
    def solve(self, y_DN_hat: Dict[str, Dict[int, float]]) -> Tuple[bool, Optional[Dict]]:
        """
        実行可能性問題を解く
        
        Returns:
            (success, dual_variables)
        """
        self.build_model(y_DN_hat)
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            duals = {
                'lambda_V': {},
                'lambda_P': {},
                'lambda_Q': {},
                'x_H_V': {},
                'x_H_P': {},
                'x_H_Q': {}
            }
            
            for t in range(1, self.T+1):
                # λ_V = λ_1 - λ_2
                lambda_1 = self.boundary_ineq_constrs[t, 0].Pi
                lambda_2 = self.boundary_ineq_constrs[t, 1].Pi
                duals['lambda_V'][t] = lambda_1 - lambda_2
                
                # λ_P = λ_3 - λ_4
                lambda_3 = self.boundary_ineq_constrs[t, 2].Pi
                lambda_4 = self.boundary_ineq_constrs[t, 3].Pi
                duals['lambda_P'][t] = lambda_3 - lambda_4
                
                # λ_Q = λ_5 - λ_6
                lambda_5 = self.boundary_ineq_constrs[t, 4].Pi
                lambda_6 = self.boundary_ineq_constrs[t, 5].Pi
                duals['lambda_Q'][t] = lambda_5 - lambda_6
                
                # x_H の値
                duals['x_H_V'][t] = self.V_H[t].X
                duals['x_H_P'][t] = self.P_H[t].X
                duals['x_H_Q'][t] = self.Q_H[t].X
            
            return True, duals
        else:
            logger.error(f"FeasibilitySubproblem H{self.h} failed with status {self.model.Status}")
            return False, None


class MasterProblem:
    """
    マスター問題（アグリゲータの配電網最適化）
    
    境界変数 y_DN と補助変数 LBD を決定
    カット平面制約を管理
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.T = params.T
        self.H = params.H
        self.model = None
        
        # カット平面の記録
        self.optimality_cuts: List[OptimalityCut] = []
        self.feasibility_cuts: List[FeasibilityCut] = []
        
    def build_model(self):
        """マスター問題モデルの構築"""
        self.model = gp.Model("MasterProblem")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)
        
        p = self.params
        
        # ========== 境界変数（DN側） ==========
        self.V_DN = self.model.addVars(
            range(1, self.T+1), range(self.H),
            lb=p.V_LL, ub=p.V_UL, name="V_DN"
        )
        self.P_DN = self.model.addVars(
            range(1, self.T+1), range(self.H),
            lb=-p.SELL_MAX, ub=p.BUY_MAX, name="P_DN"
        )
        self.Q_DN = self.model.addVars(
            range(1, self.T+1), range(self.H),
            lb=-GRB.INFINITY, name="Q_DN"
        )
        
        # 電圧降下
        self.V_drop = self.model.addVars(
            range(1, self.T+1), range(self.H),
            lb=-GRB.INFINITY, name="V_drop"
        )
        
        # 補助変数 LBD（各需要家の下界）- 合理的な下限を設定
        # 電力コストは通常負（売電収入）にはならないため、大きな負の値を下限に
        self.LBD = self.model.addVars(range(self.H), lb=-1e6, name="LBD")
        
        # アグリゲータの買電・売電（DR用）
        self.P_AG_buy = self.model.addVars(range(1, self.T+1), lb=0, name="P_AG_buy")
        self.P_AG_sell = self.model.addVars(range(1, self.T+1), lb=0, name="P_AG_sell")
        
        self.model.update()
        
        # ========== 制約条件 ==========
        self._add_voltage_constraints()
        self._add_line_capacity_constraints()
        self._add_aggregator_constraints()
        
        # ========== 目的関数 ==========
        self._set_objective()
        
    def _add_voltage_constraints(self):
        """電圧制約"""
        p = self.params
        
        for t in range(1, self.T+1):
            for h in range(self.H):
                # 累積潮流（下流の需要家を含む）
                P_cumulative = gp.quicksum(self.P_DN[t, j] for j in range(h, self.H))
                Q_cumulative = gp.quicksum(self.Q_DN[t, j] for j in range(h, self.H))
                
                # Q_DN = 0.1 * P_DN
                self.model.addConstr(
                    self.Q_DN[t, h] == 0.1 * self.P_DN[t, h],
                    name=f"Q_DN_def_{t}_{h}"
                )
                
                # 電圧降下推定
                self.model.addConstr(
                    self.V_drop[t, h] == 
                    p.linear_ap[h] * P_cumulative + 
                    p.linear_aq[h] * Q_cumulative + 
                    p.linear_b[h],
                    name=f"voltage_drop_{t}_{h}"
                )
                
                # 電圧計算
                if h == 0:
                    self.model.addConstr(
                        self.V_DN[t, h] == p.V_BASE + self.V_drop[t, h],
                        name=f"voltage_{t}_{h}"
                    )
                else:
                    self.model.addConstr(
                        self.V_DN[t, h] == self.V_DN[t, h-1] + self.V_drop[t, h],
                        name=f"voltage_{t}_{h}"
                    )
    
    def _add_line_capacity_constraints(self):
        """送電線容量制約"""
        p = self.params
        
        for t in range(1, self.T+1):
            for h in range(self.H):
                P_cumulative = gp.quicksum(self.P_DN[t, j] for j in range(h, self.H))
                Q_cumulative = gp.quicksum(self.Q_DN[t, j] for j in range(h, self.H))
                
                self.model.addQConstr(
                    P_cumulative * P_cumulative + Q_cumulative * Q_cumulative <= 
                    (p.S_LINE/100) * (p.S_LINE/100),
                    name=f"line_capacity_{t}_{h}"
                )
    
    def _add_aggregator_constraints(self):
        """アグリゲータレベルの制約"""
        p = self.params
        
        # 補助変数：各需要家の買電・売電（正部分）
        self.P_DN_plus = self.model.addVars(
            range(1, self.T+1), range(self.H), lb=0, ub=p.BUY_MAX, name="P_DN_plus"
        )
        self.P_DN_minus = self.model.addVars(
            range(1, self.T+1), range(self.H), lb=0, ub=p.SELL_MAX, name="P_DN_minus"
        )
        
        # 相補性のためのバイナリ変数
        self.delta_DN = self.model.addVars(
            range(1, self.T+1), range(self.H), vtype=GRB.BINARY, name="delta_DN"
        )
        
        M_buy = p.BUY_MAX
        M_sell = p.SELL_MAX
        
        for t in range(1, self.T+1):
            for h in range(self.H):
                # P_DN = P_DN_plus - P_DN_minus
                self.model.addConstr(
                    self.P_DN[t, h] == self.P_DN_plus[t, h] - self.P_DN_minus[t, h],
                    name=f"P_DN_split_{t}_{h}"
                )
                
                # 相補性制約（Big-M）
                # delta=1 → 買電（P_DN_plus>0可能, P_DN_minus=0）
                # delta=0 → 売電（P_DN_plus=0, P_DN_minus>0可能）
                self.model.addConstr(
                    self.P_DN_plus[t, h] <= M_buy * self.delta_DN[t, h],
                    name=f"compl_plus_{t}_{h}"
                )
                self.model.addConstr(
                    self.P_DN_minus[t, h] <= M_sell * (1 - self.delta_DN[t, h]),
                    name=f"compl_minus_{t}_{h}"
                )
            
            # アグリゲータの買電 = 全需要家の正味買電（正の部分の合計）
            self.model.addConstr(
                self.P_AG_buy[t] == gp.quicksum(
                    self.P_DN_plus[t, h] for h in range(self.H)
                ),
                name=f"AG_buy_{t}"
            )
            
            # アグリゲータの売電 = 全需要家の正味売電（負の部分の合計）
            self.model.addConstr(
                self.P_AG_sell[t] == gp.quicksum(
                    self.P_DN_minus[t, h] for h in range(self.H)
                ),
                name=f"AG_sell_{t}"
            )
    
    def _set_objective(self):
        """目的関数"""
        p = self.params
        
        # 需要家コストの下界合計
        household_lb = gp.quicksum(self.LBD[h] for h in range(self.H))
        
        # DR期間の買電電力
        if p.DR_END <= self.T:
            ddr_total = gp.quicksum(
                self.P_AG_buy[t] for t in range(p.DR_START, p.DR_END+1)
            )
        else:
            ddr_total = 0
        
        self.model.setObjective(household_lb + ddr_total, GRB.MINIMIZE)
    
    def add_optimality_cut(self, cut: OptimalityCut):
        """最適カット平面の追加"""
        self.optimality_cuts.append(cut)
        
        h = cut.household
        
        # L*_H,p - μ^T y_DN
        # L*_H,p = UB_h + Σ_t (μ_V * y_hat_V + μ_P * y_hat_P + μ_Q * y_hat_Q)
        L_star = cut.UB_h
        for t in range(1, self.T+1):
            L_star += (cut.mu_V[t] * cut.y_hat_V[t] + 
                       cut.mu_P[t] * cut.y_hat_P[t] + 
                       cut.mu_Q[t] * cut.y_hat_Q[t])
        
        # LBD_h >= L*_H,p - μ^T y_DN
        self.model.addConstr(
            self.LBD[h] >= L_star - gp.quicksum(
                cut.mu_V[t] * self.V_DN[t, h] + 
                cut.mu_P[t] * self.P_DN[t, h] + 
                cut.mu_Q[t] * self.Q_DN[t, h]
                for t in range(1, self.T+1)
            ),
            name=f"optimality_cut_{h}_{cut.iteration}"
        )
    
    def add_feasibility_cut(self, cut: FeasibilityCut):
        """実行可能カット平面の追加"""
        self.feasibility_cuts.append(cut)
        
        h = cut.household
        
        # L* - λ^T y_DN <= 0
        # L* = Σ_t (λ_V * x_H_V + λ_P * x_H_P + λ_Q * x_H_Q)
        L_star = 0
        for t in range(1, self.T+1):
            L_star += (cut.lambda_V[t] * cut.x_H_V[t] + 
                       cut.lambda_P[t] * cut.x_H_P[t] + 
                       cut.lambda_Q[t] * cut.x_H_Q[t])
        
        self.model.addConstr(
            L_star - gp.quicksum(
                cut.lambda_V[t] * self.V_DN[t, h] + 
                cut.lambda_P[t] * self.P_DN[t, h] + 
                cut.lambda_Q[t] * self.Q_DN[t, h]
                for t in range(1, self.T+1)
            ) <= 0,
            name=f"feasibility_cut_{h}_{cut.iteration}"
        )
    
    def solve(self) -> Tuple[bool, Optional[float], Optional[Dict]]:
        """
        マスター問題を解く
        
        Returns:
            (success, objective_value, y_DN_values)
        """
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            obj_val = self.model.ObjVal
            
            y_DN = {}
            for h in range(self.H):
                y_DN[h] = {
                    'V': {t: self.V_DN[t, h].X for t in range(1, self.T+1)},
                    'P': {t: self.P_DN[t, h].X for t in range(1, self.T+1)},
                    'Q': {t: self.Q_DN[t, h].X for t in range(1, self.T+1)}
                }
            
            return True, obj_val, y_DN
        
        elif self.model.Status == GRB.INFEASIBLE:
            logger.error("Master problem is infeasible")
            self.model.computeIIS()
            self.model.write("master_iis.ilp")
            return False, None, None
        
        elif self.model.Status == GRB.UNBOUNDED:
            logger.error("Master problem is unbounded")
            return False, None, None
        
        else:
            logger.error(f"Master problem failed with status {self.model.Status}")
            return False, None, None
    
    def get_solution(self) -> Dict:
        """マスター問題の解を取得"""
        if self.model.Status != GRB.OPTIMAL:
            return None
        
        solution = {
            'V_DN': {},
            'P_DN': {},
            'Q_DN': {},
            'LBD': {},
            'P_AG_buy': {},
            'P_AG_sell': {}
        }
        
        for h in range(self.H):
            solution['V_DN'][h] = {t: self.V_DN[t, h].X for t in range(1, self.T+1)}
            solution['P_DN'][h] = {t: self.P_DN[t, h].X for t in range(1, self.T+1)}
            solution['Q_DN'][h] = {t: self.Q_DN[t, h].X for t in range(1, self.T+1)}
            solution['LBD'][h] = self.LBD[h].X
        
        for t in range(1, self.T+1):
            solution['P_AG_buy'][t] = self.P_AG_buy[t].X
            solution['P_AG_sell'][t] = self.P_AG_sell[t].X
        
        return solution


class BendersSolver:
    """
    Generalized Benders Decomposition ソルバー
    """
    
    def __init__(self, params: SystemParameters, max_iterations: int = 100, 
                 tolerance: float = 1e-4):
        self.params = params
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # マスター問題
        self.master = MasterProblem(params)
        
        # サブ問題（各需要家）
        self.subproblems = [Subproblem(h, params) for h in range(params.H)]
        
        # 実行可能性サブ問題
        self.feasibility_subproblems = [FeasibilitySubproblem(h, params) for h in range(params.H)]
        
        # 収束履歴
        self.history = {
            'iteration': [],
            'LB': [],
            'UB': [],
            'gap': [],
            'time': []
        }
        
    def initialize_boundary_variables(self) -> Dict[int, Dict[str, Dict[int, float]]]:
        """境界変数の初期化（電力需要とPV出力に基づく現実的な値）"""
        p = self.params
        y_DN_hat = {}
        
        for h in range(p.H):
            y_DN_hat[h] = {
                'V': {},
                'P': {},
                'Q': {}
            }
            
            cumulative_V = p.V_BASE
            
            for t in range(1, p.T+1):
                # 電力バランスからの初期推定
                # P_H ≈ P_dmd - P_pv（蓄電池・HPを無視した簡易推定）
                net_demand = p.p_dmd[t, h] - p.p_pv[t, h]
                
                # 買電上限・売電上限でクリップ
                P_init = max(-p.SELL_MAX, min(p.BUY_MAX, net_demand))
                Q_init = 0.1 * P_init
                
                # 電圧推定（線形近似）
                # 累積潮流を考慮
                P_cumulative = 0
                Q_cumulative = 0
                for j in range(h, p.H):
                    net_j = p.p_dmd[t, j] - p.p_pv[t, j]
                    P_j = max(-p.SELL_MAX, min(p.BUY_MAX, net_j))
                    P_cumulative += P_j
                    Q_cumulative += 0.1 * P_j
                
                V_drop = (p.linear_ap[h] * P_cumulative + 
                          p.linear_aq[h] * Q_cumulative + 
                          p.linear_b[h])
                
                if h == 0:
                    V_init = p.V_BASE + V_drop
                else:
                    V_init = cumulative_V + V_drop
                
                # 電圧制約内にクリップ
                V_init = max(p.V_LL, min(p.V_UL, V_init))
                
                y_DN_hat[h]['V'][t] = V_init
                y_DN_hat[h]['P'][t] = P_init
                y_DN_hat[h]['Q'][t] = Q_init
                
                cumulative_V = V_init
        
        logger.info("Boundary variables initialized with demand-based estimates")
        return y_DN_hat
    
    def solve(self) -> Dict:
        """GBDアルゴリズムの実行"""
        logger.info("="*60)
        logger.info("GBD Algorithm Started")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Step 1: 初期化
        LB = -np.inf
        UB = np.inf
        k = 0
        
        # 境界変数の初期化
        y_DN_hat = self.initialize_boundary_variables()
        
        # マスター問題の構築
        self.master.build_model()
        
        # カウンタ
        p_counts = [0] * self.params.H  # 最適カット数
        q_counts = [0] * self.params.H  # 実行可能カット数
        
        # 最良解の記録
        best_solution = None
        
        while k < self.max_iterations:
            k += 1
            iter_start = time.time()
            
            logger.info(f"\n--- Iteration {k} ---")
            
            # Step 2: 各需要家がサブ問題を解く
            all_feasible = True
            sub_objectives = {}
            sub_solutions = {}
            
            for h in range(self.params.H):
                feasible, obj_val, duals = self.subproblems[h].solve(y_DN_hat[h])
                
                if feasible:
                    sub_objectives[h] = obj_val
                    sub_solutions[h] = self.subproblems[h].get_solution()
                    
                    # Step 2a: 最適カット生成
                    p_counts[h] += 1
                    
                    cut = OptimalityCut(
                        household=h,
                        iteration=k,
                        UB_h=obj_val,
                        mu_V=duals['mu_V'].copy(),
                        mu_P=duals['mu_P'].copy(),
                        mu_Q=duals['mu_Q'].copy(),
                        y_hat_V=y_DN_hat[h]['V'].copy(),
                        y_hat_P=y_DN_hat[h]['P'].copy(),
                        y_hat_Q=y_DN_hat[h]['Q'].copy()
                    )
                    
                    self.master.add_optimality_cut(cut)
                    logger.info(f"  H{h}: Feasible, cost={obj_val:.2f}, added optimality cut #{p_counts[h]}")
                    
                else:
                    all_feasible = False
                    
                    # Step 2b: 実行可能カット生成
                    q_counts[h] += 1
                    
                    success, feas_duals = self.feasibility_subproblems[h].solve(y_DN_hat[h])
                    
                    if success:
                        cut = FeasibilityCut(
                            household=h,
                            iteration=k,
                            lambda_V=feas_duals['lambda_V'].copy(),
                            lambda_P=feas_duals['lambda_P'].copy(),
                            lambda_Q=feas_duals['lambda_Q'].copy(),
                            x_H_V=feas_duals['x_H_V'].copy(),
                            x_H_P=feas_duals['x_H_P'].copy(),
                            x_H_Q=feas_duals['x_H_Q'].copy()
                        )
                        
                        self.master.add_feasibility_cut(cut)
                        logger.info(f"  H{h}: Infeasible, added feasibility cut #{q_counts[h]}")
                    else:
                        logger.error(f"  H{h}: Failed to generate feasibility cut")
            
            # 上界の更新（全て実行可能な場合のみ）
            if all_feasible:
                # アグリゲータ目的関数の計算
                agg_obj = 0
                for t in range(self.params.DR_START, min(self.params.DR_END+1, self.params.T+1)):
                    for h in range(self.params.H):
                        agg_obj += max(y_DN_hat[h]['P'][t], 0)
                
                current_UB = sum(sub_objectives.values()) + agg_obj
                
                if current_UB < UB:
                    UB = current_UB
                    best_solution = {
                        'y_DN': {h: y_DN_hat[h].copy() for h in range(self.params.H)},
                        'sub_solutions': sub_solutions.copy(),
                        'sub_objectives': sub_objectives.copy()
                    }
                    logger.info(f"  Updated UB = {UB:.2f}")
            
            # Step 3: マスター問題を解く
            success, master_obj, new_y_DN = self.master.solve()
            
            if not success:
                logger.error("Master problem failed")
                break
            
            LB = master_obj
            y_DN_hat = new_y_DN
            
            logger.info(f"  Master solved: LB = {LB:.2f}")
            
            # 収束履歴の記録
            gap = (UB - LB) / max(abs(UB), 1e-10) if UB < np.inf else np.inf
            iter_time = time.time() - iter_start
            
            self.history['iteration'].append(k)
            self.history['LB'].append(LB)
            self.history['UB'].append(UB if UB < np.inf else None)
            self.history['gap'].append(gap if gap < np.inf else None)
            self.history['time'].append(iter_time)
            
            UB_str = f"{UB:.2f}" if UB < np.inf else "inf"
            gap_str = f"{gap:.4f}" if gap < np.inf else "inf"
            logger.info(f"  LB={LB:.2f}, UB={UB_str}, Gap={gap_str}")
            
            # Step 4: 収束判定
            if UB < np.inf and abs(UB - LB) < self.tolerance:
                logger.info(f"\n*** Converged at iteration {k} ***")
                logger.info(f"Final gap: {gap:.6f}")
                break
            
            if UB < np.inf and gap < self.tolerance:
                logger.info(f"\n*** Converged (relative gap) at iteration {k} ***")
                break
        
        total_time = time.time() - start_time
        
        # 結果の整理
        results = {
            'status': 'OPTIMAL' if k < self.max_iterations else 'MAX_ITERATIONS',
            'iterations': k,
            'LB': LB,
            'UB': UB if UB < np.inf else None,
            'gap': gap if gap < np.inf else None,
            'solve_time': total_time,
            'optimality_cuts': sum(p_counts),
            'feasibility_cuts': sum(q_counts),
            'best_solution': best_solution,
            'history': self.history
        }
        
        logger.info("\n" + "="*60)
        logger.info("GBD Algorithm Completed")
        logger.info(f"Total iterations: {k}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Final LB: {LB:.2f}")
        logger.info(f"Final UB: {UB:.2f if UB < np.inf else 'inf'}")
        logger.info(f"Optimality cuts: {sum(p_counts)}")
        logger.info(f"Feasibility cuts: {sum(q_counts)}")
        logger.info("="*60)
        
        return results
    
    def export_results(self, results: Dict, output_dir: str = "../data/output/gbd"):
        """結果のエクスポート"""
        os.makedirs(output_dir, exist_ok=True)
        
        # サマリーCSV
        summary_df = pd.DataFrame({
            'Metric': ['Status', 'Iterations', 'LB', 'UB', 'Gap', 
                       'Solve Time', 'Optimality Cuts', 'Feasibility Cuts'],
            'Value': [
                results['status'], results['iterations'],
                results['LB'], results['UB'], results['gap'],
                results['solve_time'], results['optimality_cuts'],
                results['feasibility_cuts']
            ]
        })
        summary_df.to_csv(f"{output_dir}/gbd_summary.csv", index=False)
        
        # 収束履歴CSV
        history_df = pd.DataFrame(results['history'])
        history_df.to_csv(f"{output_dir}/gbd_convergence.csv", index=False)
        
        # 最良解の詳細
        if results['best_solution'] is not None:
            for h in range(self.params.H):
                if h in results['best_solution']['sub_solutions']:
                    sol = results['best_solution']['sub_solutions'][h]
                    y_DN = results['best_solution']['y_DN'][h]
                    
                    rows = []
                    for t in range(1, self.params.T+1):
                        rows.append({
                            'Time': t,
                            'P_buy': sol['P_buy'][t],
                            'P_sell': sol['P_sell'][t],
                            'P_ch': sol['P_ch'][t],
                            'P_dch': sol['P_dch'][t],
                            'E_bat': sol['E_bat'][t],
                            'P_hp': sol['P_hp'][t],
                            'H_tank': sol['H_tank'][t],
                            'P_H': sol['P_H'][t],
                            'Q_H': sol['Q_H'][t],
                            'V_H': sol['V_H'][t],
                            'P_DN': y_DN['P'][t],
                            'Q_DN': y_DN['Q'][t],
                            'V_DN': y_DN['V'][t]
                        })
                    
                    df = pd.DataFrame(rows)
                    df.to_csv(f"{output_dir}/gbd_household_{h}.csv", index=False)
        
        logger.info(f"Results exported to {output_dir}")
    
    def plot_convergence(self, output_dir: str = "../data/output/gbd"):
        """収束プロットの作成"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = self.history['iteration']
        LBs = self.history['LB']
        UBs = [ub if ub is not None else np.nan for ub in self.history['UB']]
        gaps = [g if g is not None else np.nan for g in self.history['gap']]
        
        # 上下界プロット
        ax = axes[0]
        ax.plot(iterations, LBs, 'b-o', label='Lower Bound', markersize=4)
        ax.plot(iterations, UBs, 'r-o', label='Upper Bound', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('GBD Convergence: Bounds')
        ax.legend()
        ax.grid(True)
        
        # ギャッププロット
        ax = axes[1]
        ax.semilogy(iterations, gaps, 'g-o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Relative Gap')
        ax.set_title('GBD Convergence: Gap')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gbd_convergence.png", dpi=100)
        plt.close()
        
        logger.info(f"Convergence plot saved to {output_dir}/gbd_convergence.png")


def main():
    """メイン実行関数"""
    try:
        # パラメータ読み込み
        params = SystemParameters(data_dir="../data")
        
        # GBDソルバーの作成と実行
        solver = BendersSolver(
            params=params,
            max_iterations=100,
            tolerance=1e-3
        )
        
        results = solver.solve()
        
        # 結果のエクスポート
        solver.export_results(results)
        
        # 収束プロット
        solver.plot_convergence()
        
        # サマリー表示
        print("\n" + "="*60)
        print("GBD Optimization Completed")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Lower Bound: {results['LB']:.2f}")
        print(f"Upper Bound: {results['UB']:.2f if results['UB'] else 'N/A'}")
        print(f"Gap: {results['gap']:.4f if results['gap'] else 'N/A'}")
        print(f"Solve Time: {results['solve_time']:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()