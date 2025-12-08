"""
Generalized Benders Decomposition (GBD) for Distribution Network Optimization
配電網最適化のための一般化ベンダーズ分解法

アグリゲータ（マスター問題）と需要家（サブ問題）の情報を分離した
分散型最適化アルゴリズム

3ケース対応版：
- Case1: DRなし
- Case2: 下げDR
- Case3: 上げDR
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import copy

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DRCase(Enum):
    """DRケースの定義"""
    NO_DR = 1      # Case1: DRなし
    DOWN_DR = 2    # Case2: 下げDR（買電-売電を最小化）
    UP_DR = 3      # Case3: 上げDR（買電-売電を最大化 = -(買電-売電)を最小化）

    def __str__(self):
        if self == DRCase.NO_DR:
            return "Case1_NoDR"
        elif self == DRCase.DOWN_DR:
            return "Case2_DownDR"
        else:
            return "Case3_UpDR"

    @property
    def description(self):
        if self == DRCase.NO_DR:
            return "DRなし：電力コスト + ペナルティ"
        elif self == DRCase.DOWN_DR:
            return "下げDR：電力コスト + (買電-売電) + ペナルティ"
        else:
            return "上げDR：電力コスト - (買電-売電) + ペナルティ"


@dataclass
class OptimalityCut:
    """最適カット（サブ問題が実行可能な場合に生成）"""
    household_id: int
    iteration: int
    L_star: float                          # 定数項
    mu_P: Dict[int, float] = field(default_factory=dict)  # P_DNに対する双対変数
    mu_Q: Dict[int, float] = field(default_factory=dict)  # Q_DNに対する双対変数


@dataclass
class FeasibilityCut:
    """実行可能カット（サブ問題が実行不可能な場合に生成）"""
    household_id: int
    iteration: int
    L_star: float                          # 定数項
    lambda_P: Dict[int, float] = field(default_factory=dict)  # P_DNに対する係数
    lambda_Q: Dict[int, float] = field(default_factory=dict)  # Q_DNに対する係数


class SystemParameters:
    """システムパラメータとデータ管理"""

    def __init__(self, dr_case: DRCase = DRCase.DOWN_DR):
        """
        初期化

        Args:
            dr_case: DRケース（NO_DR, DOWN_DR, UP_DR）
        """
        self.dr_case = dr_case
        logger.info(f"ケース設定: {dr_case} - {dr_case.description}")

        # システムパラメータ
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

        # DR期間（13時～16時）
        self.DR_START = 26  # 13:00 (slot 26)
        self.DR_END = 32    # 16:00 (slot 32)

        # GBDアルゴリズムパラメータ
        self.MAX_ITERATIONS = 100  # 最大反復回数
        self.TOLERANCE = 1e-4      # 収束許容誤差
        self.BIG_M = 1e6           # 大きな数（制約緩和用）

        # データ読み込み
        self.load_data()

    def load_data(self):
        """データ読み込み"""
        logger.info("データ読み込み開始...")

        # CSVファイル読み込み
        dmd_data = pd.read_csv("../data/input/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", encoding="shift_jis")
        buy_price_data = pd.read_csv("../data/input/buy_energy30.csv", encoding="shift_jis")
        pv_data = pd.read_csv("../data/input/pv_output.csv")
        heat_dmd_data = pd.read_csv("../data/input/heat_demand1y.csv", encoding="shift_jis")
        temperature_outside_data = pd.read_csv("../data/input/temperature1y.csv", encoding="shift_jis")
        temperature_water_data = pd.read_csv("../data/input/water_temperature_30min.csv", encoding="shift_jis")
        voltage_estimate = pd.read_csv("../data/output/random_test/2.1.回帰係数・切片.csv", encoding="shift_jis")

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


class HouseholdSubproblem:
    """
    需要家のサブ問題

    境界変数（P_DN_hat, Q_DN_hat）を固定パラメータとして受け取り、
    需要家内の電力コストを最小化する
    """

    def __init__(self, params: SystemParameters, household_id: int):
        """
        初期化

        Args:
            params: システムパラメータ
            household_id: 需要家ID（0からH-1）
        """
        self.params = params
        self.h = household_id
        self.model = None
        self.variables = {}
        self.constraints = {}

        # 境界変数の固定値（マスター問題から受け取る）
        self.P_DN_hat: Dict[int, float] = {}
        self.Q_DN_hat: Dict[int, float] = {}

    def build_model(self):
        """サブ問題モデルの構築"""
        self.model = gp.Model(f"Subproblem_H{self.h}")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)  # 二次制約の双対変数を取得

        T = self.params.T
        h = self.h

        # ========== 決定変数 ==========
        # 買電・売電
        self.variables['P_buy'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.BUY_MAX, name="P_buy"
        )
        self.variables['P_sell'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.SELL_MAX, name="P_sell"
        )
        self.variables['delta_buy'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_buy"
        )
        self.variables['delta_sell'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_sell"
        )

        # 蓄電池
        self.variables['P_ch'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.N_B_PCS, name="P_charge"
        )
        self.variables['P_dch'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.N_B_PCS, name="P_discharge"
        )
        self.variables['E_bat'] = self.model.addVars(
            range(1, T+1), lb=0.2*self.params.N_B, ub=self.params.N_B, name="E_battery"
        )
        self.variables['delta_ch'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_charge"
        )
        self.variables['delta_dch'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_discharge"
        )

        # ヒートポンプ
        self.variables['P_hp'] = self.model.addVars(
            range(1, T+1), lb=0, ub=10.0, name="P_hp"
        )
        self.variables['H_prod'] = self.model.addVars(
            range(1, T+1), lb=0, name="H_produce"
        )
        self.variables['H_tank'] = self.model.addVars(
            range(T+1), lb=0, name="H_tank"
        )
        self.variables['delta_hp'] = self.model.addVars(
            range(T+1), vtype=GRB.BINARY, name="delta_hp"
        )
        self.variables['delta_start'] = self.model.addVars(
            range(T+1), vtype=GRB.BINARY, name="delta_start"
        )
        self.variables['delta_stop'] = self.model.addVars(
            range(T+1), vtype=GRB.BINARY, name="delta_stop"
        )

        # 境界変数（需要家側）
        self.variables['P_H'] = self.model.addVars(
            range(1, T+1), lb=-GRB.INFINITY, name="P_H"
        )
        self.variables['Q_H'] = self.model.addVars(
            range(1, T+1), lb=-GRB.INFINITY, name="Q_H"
        )

        self.model.update()

        # ========== 制約条件 ==========
        self._add_power_balance_constraints()
        self._add_battery_constraints()
        self._add_heatpump_constraints()

        # 境界制約は後で追加（P_DN_hatが設定されてから）

        # ========== 目的関数 ==========
        self._set_objective()

    def _add_power_balance_constraints(self):
        """電力バランス制約"""
        T = self.params.T
        h = self.h
        P_buy = self.variables['P_buy']
        P_sell = self.variables['P_sell']
        delta_buy = self.variables['delta_buy']
        delta_sell = self.variables['delta_sell']
        P_ch = self.variables['P_ch']
        P_dch = self.variables['P_dch']
        P_hp = self.variables['P_hp']
        P_H = self.variables['P_H']
        Q_H = self.variables['Q_H']

        for t in range(1, T+1):
            # 電力バランス
            self.model.addConstr(
                self.params.p_pv[t, h] + P_buy[t] + P_dch[t] ==
                self.params.p_dmd[t, h] + P_sell[t] + P_ch[t] + P_hp[t],
                name=f"power_balance_{t}"
            )

            # 買電売電同時禁止
            self.model.addConstr(
                P_buy[t] <= delta_buy[t] * self.params.BUY_MAX,
                name=f"buy_limit_{t}"
            )
            self.model.addConstr(
                P_sell[t] <= delta_sell[t] * self.params.SELL_MAX,
                name=f"sell_limit_{t}"
            )
            self.model.addConstr(
                delta_buy[t] + delta_sell[t] <= 1,
                name=f"buy_sell_exclusive_{t}"
            )

            # 境界変数の定義（P_H = 買電 - 売電）
            self.model.addConstr(
                P_H[t] == P_buy[t] - P_sell[t],
                name=f"P_H_def_{t}"
            )
            # Q = 0.1 * P
            self.model.addConstr(
                Q_H[t] == 0.1 * P_H[t],
                name=f"Q_H_def_{t}"
            )

    def _add_battery_constraints(self):
        """蓄電池制約"""
        T = self.params.T
        P_ch = self.variables['P_ch']
        P_dch = self.variables['P_dch']
        E_bat = self.variables['E_bat']
        delta_ch = self.variables['delta_ch']
        delta_dch = self.variables['delta_dch']

        for t in range(1, T+1):
            # 充放電制約
            self.model.addConstr(
                P_ch[t] <= self.params.N_B_PCS * delta_ch[t],
                name=f"charge_limit_{t}"
            )
            self.model.addConstr(
                P_dch[t] <= self.params.N_B_PCS * delta_dch[t],
                name=f"discharge_limit_{t}"
            )
            self.model.addConstr(
                delta_ch[t] + delta_dch[t] <= 1,
                name=f"charge_discharge_exclusive_{t}"
            )

            # SOC更新
            if t == 1:
                prev_soc = 0.5 * self.params.N_B
            else:
                prev_soc = E_bat[t-1]

            self.model.addConstr(
                E_bat[t] == prev_soc +
                0.5 * self.params.ETA_B * P_ch[t] -
                0.5 * P_dch[t] / self.params.ETA_B,
                name=f"battery_soc_{t}"
            )

        # 終端条件
        self.model.addConstr(
            E_bat[T] == 0.5 * self.params.N_B,
            name="battery_final"
        )

    def _add_heatpump_constraints(self):
        """ヒートポンプ制約"""
        T = self.params.T
        h = self.h
        P_hp = self.variables['P_hp']
        H_prod = self.variables['H_prod']
        H_tank = self.variables['H_tank']
        delta_hp = self.variables['delta_hp']
        delta_start = self.variables['delta_start']
        delta_stop = self.variables['delta_stop']

        # 初期条件
        self.model.addConstr(delta_hp[0] == 0, name="hp_init_delta")
        self.model.addConstr(delta_start[0] == 0, name="hp_init_start")
        self.model.addConstr(delta_stop[0] == 0, name="hp_init_stop")
        self.model.addConstr(
            H_tank[0] == 0.5 * self.params.CW * self.params.VLT * 70,
            name="hp_tank_init"
        )

        tank_max = self.params.CW * self.params.VLT * 70
        tank_min = 0.2 * tank_max

        for t in range(1, T+1):
            cop = self.params.cop[t]

            # HP消費電力
            self.model.addConstr(
                P_hp[t] == self.params.XHP * delta_hp[t] +
                (H_prod[t] + self.params.R2 * self.params.CPH * delta_start[t]) /
                (self.params.CE * cop),
                name=f"hp_power_{t}"
            )

            # 熱製造量制約
            self.model.addConstr(
                H_prod[t] >= 0.5 * self.params.R1 * self.params.CPH * delta_hp[t],
                name=f"hp_prod_min_{t}"
            )
            self.model.addConstr(
                H_prod[t] <= 0.5 * self.params.CPH * delta_hp[t],
                name=f"hp_prod_max_{t}"
            )

            # 運転状態の遷移
            self.model.addConstr(
                delta_hp[t] - delta_hp[t-1] == delta_start[t] - delta_stop[t],
                name=f"hp_state_{t}"
            )

            # 起動停止同時禁止
            self.model.addConstr(
                delta_start[t] + delta_stop[t] <= 1,
                name=f"hp_start_stop_exclusive_{t}"
            )

            # 貯湯槽更新
            self.model.addConstr(
                H_tank[t] == H_tank[t-1] + H_prod[t] - self.params.heat_dmd[t, h],
                name=f"hp_tank_{t}"
            )

            # 貯湯槽容量制約
            self.model.addConstr(H_tank[t] >= tank_min, name=f"tank_min_{t}")
            self.model.addConstr(H_tank[t] <= tank_max, name=f"tank_max_{t}")

        # 終端条件
        self.model.addConstr(
            H_tank[T] >= 0.5 * tank_max,
            name="hp_tank_final"
        )

    def _set_objective(self):
        """目的関数（需要家の電力コスト）"""
        T = self.params.T
        P_buy = self.variables['P_buy']
        P_sell = self.variables['P_sell']

        # 電力コスト = 買電コスト - 売電収入
        cost = gp.quicksum(
            self.params.buy_price[t] * P_buy[t] -
            self.params.SELL_PRICE * P_sell[t]
            for t in range(1, T+1)
        )
        self.model.setObjective(cost, GRB.MINIMIZE)

    def set_boundary_values(self, P_DN_hat: Dict[int, float], Q_DN_hat: Dict[int, float]):
        """
        境界変数の固定値を設定

        Args:
            P_DN_hat: 有効電力の固定値 {t: value}
            Q_DN_hat: 無効電力の固定値 {t: value}
        """
        self.P_DN_hat = P_DN_hat.copy()
        self.Q_DN_hat = Q_DN_hat.copy()

    def add_boundary_constraints(self):
        """境界等式制約を追加（P_H = P_DN_hat, Q_H = Q_DN_hat）"""
        T = self.params.T
        P_H = self.variables['P_H']
        Q_H = self.variables['Q_H']

        # 既存の境界制約を削除
        constrs_to_remove = []
        for name, constr in self.constraints.items():
            if name.startswith('boundary_P_') or name.startswith('boundary_Q_'):
                constrs_to_remove.append((name, constr))

        for name, constr in constrs_to_remove:
            self.model.remove(constr)
            del self.constraints[name]

        # 新しい境界制約を追加
        for t in range(1, T+1):
            constr_P = self.model.addConstr(
                P_H[t] == self.P_DN_hat[t],
                name=f"boundary_P_{t}"
            )
            self.constraints[f'boundary_P_{t}'] = constr_P

            constr_Q = self.model.addConstr(
                Q_H[t] == self.Q_DN_hat[t],
                name=f"boundary_Q_{t}"
            )
            self.constraints[f'boundary_Q_{t}'] = constr_Q

        self.model.update()

    def solve(self) -> Tuple[int, Optional[float], Dict[str, Any]]:
        """
        サブ問題を解く

        Returns:
            (status, objective, solution_dict)
            - status: Gurobi最適化ステータス
            - objective: 目的関数値（実行不可能の場合はNone）
            - solution_dict: 解の辞書
        """
        self.model.optimize()

        status = self.model.Status
        objective = None
        solution = {}

        if status == GRB.OPTIMAL:
            objective = self.model.ObjVal
            solution = self._extract_solution()

        return status, objective, solution

    def _extract_solution(self) -> Dict[str, Any]:
        """解を抽出"""
        T = self.params.T
        solution = {
            'P_buy': {t: self.variables['P_buy'][t].X for t in range(1, T+1)},
            'P_sell': {t: self.variables['P_sell'][t].X for t in range(1, T+1)},
            'P_ch': {t: self.variables['P_ch'][t].X for t in range(1, T+1)},
            'P_dch': {t: self.variables['P_dch'][t].X for t in range(1, T+1)},
            'E_bat': {t: self.variables['E_bat'][t].X for t in range(1, T+1)},
            'P_hp': {t: self.variables['P_hp'][t].X for t in range(1, T+1)},
            'H_tank': {t: self.variables['H_tank'][t].X for t in range(T+1)},
            'delta_hp': {t: self.variables['delta_hp'][t].X for t in range(T+1)},
            'P_H': {t: self.variables['P_H'][t].X for t in range(1, T+1)},
            'Q_H': {t: self.variables['Q_H'][t].X for t in range(1, T+1)},
            # バイナリ変数の値も保存（双対変数取得時の固定用）
            'delta_buy': {t: self.variables['delta_buy'][t].X for t in range(1, T+1)},
            'delta_sell': {t: self.variables['delta_sell'][t].X for t in range(1, T+1)},
            'delta_ch': {t: self.variables['delta_ch'][t].X for t in range(1, T+1)},
            'delta_dch': {t: self.variables['delta_dch'][t].X for t in range(1, T+1)},
            'delta_start': {t: self.variables['delta_start'][t].X for t in range(T+1)},
            'delta_stop': {t: self.variables['delta_stop'][t].X for t in range(T+1)},
        }
        return solution

    def get_dual_variables(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        境界制約の双対変数を取得

        バイナリ変数を連続変数に緩和し、MIP解の値に固定してから
        LPとして解いて双対変数を取得する

        Returns:
            (mu_P, mu_Q): 有効電力と無効電力の双対変数
        """
        T = self.params.T
        mu_P = {}
        mu_Q = {}

        # MIPの解が存在しない場合はゼロを返す
        if self.model.Status != GRB.OPTIMAL:
            for t in range(1, T+1):
                mu_P[t] = 0.0
                mu_Q[t] = 0.0
            return mu_P, mu_Q

        # バイナリ変数の現在の値を保存
        binary_vars = {
            'delta_buy': {t: self.variables['delta_buy'][t].X for t in range(1, T+1)},
            'delta_sell': {t: self.variables['delta_sell'][t].X for t in range(1, T+1)},
            'delta_ch': {t: self.variables['delta_ch'][t].X for t in range(1, T+1)},
            'delta_dch': {t: self.variables['delta_dch'][t].X for t in range(1, T+1)},
            'delta_hp': {t: self.variables['delta_hp'][t].X for t in range(T+1)},
            'delta_start': {t: self.variables['delta_start'][t].X for t in range(T+1)},
            'delta_stop': {t: self.variables['delta_stop'][t].X for t in range(T+1)},
        }

        # バイナリ変数を連続変数に緩和し、値を固定
        for t in range(1, T+1):
            self.variables['delta_buy'][t].VType = GRB.CONTINUOUS
            self.variables['delta_buy'][t].LB = binary_vars['delta_buy'][t]
            self.variables['delta_buy'][t].UB = binary_vars['delta_buy'][t]

            self.variables['delta_sell'][t].VType = GRB.CONTINUOUS
            self.variables['delta_sell'][t].LB = binary_vars['delta_sell'][t]
            self.variables['delta_sell'][t].UB = binary_vars['delta_sell'][t]

            self.variables['delta_ch'][t].VType = GRB.CONTINUOUS
            self.variables['delta_ch'][t].LB = binary_vars['delta_ch'][t]
            self.variables['delta_ch'][t].UB = binary_vars['delta_ch'][t]

            self.variables['delta_dch'][t].VType = GRB.CONTINUOUS
            self.variables['delta_dch'][t].LB = binary_vars['delta_dch'][t]
            self.variables['delta_dch'][t].UB = binary_vars['delta_dch'][t]

        for t in range(T+1):
            self.variables['delta_hp'][t].VType = GRB.CONTINUOUS
            self.variables['delta_hp'][t].LB = binary_vars['delta_hp'][t]
            self.variables['delta_hp'][t].UB = binary_vars['delta_hp'][t]

            self.variables['delta_start'][t].VType = GRB.CONTINUOUS
            self.variables['delta_start'][t].LB = binary_vars['delta_start'][t]
            self.variables['delta_start'][t].UB = binary_vars['delta_start'][t]

            self.variables['delta_stop'][t].VType = GRB.CONTINUOUS
            self.variables['delta_stop'][t].LB = binary_vars['delta_stop'][t]
            self.variables['delta_stop'][t].UB = binary_vars['delta_stop'][t]

        self.model.update()

        # LPとして再度解く
        self.model.optimize()

        # 双対変数を取得
        if self.model.Status == GRB.OPTIMAL:
            for t in range(1, T+1):
                constr_P = self.constraints.get(f'boundary_P_{t}')
                constr_Q = self.constraints.get(f'boundary_Q_{t}')

                if constr_P is not None:
                    mu_P[t] = constr_P.Pi
                else:
                    mu_P[t] = 0.0

                if constr_Q is not None:
                    mu_Q[t] = constr_Q.Pi
                else:
                    mu_Q[t] = 0.0
        else:
            for t in range(1, T+1):
                mu_P[t] = 0.0
                mu_Q[t] = 0.0

        # バイナリ変数を元に戻す
        for t in range(1, T+1):
            self.variables['delta_buy'][t].VType = GRB.BINARY
            self.variables['delta_buy'][t].LB = 0
            self.variables['delta_buy'][t].UB = 1

            self.variables['delta_sell'][t].VType = GRB.BINARY
            self.variables['delta_sell'][t].LB = 0
            self.variables['delta_sell'][t].UB = 1

            self.variables['delta_ch'][t].VType = GRB.BINARY
            self.variables['delta_ch'][t].LB = 0
            self.variables['delta_ch'][t].UB = 1

            self.variables['delta_dch'][t].VType = GRB.BINARY
            self.variables['delta_dch'][t].LB = 0
            self.variables['delta_dch'][t].UB = 1

        for t in range(T+1):
            self.variables['delta_hp'][t].VType = GRB.BINARY
            self.variables['delta_hp'][t].LB = 0
            self.variables['delta_hp'][t].UB = 1

            self.variables['delta_start'][t].VType = GRB.BINARY
            self.variables['delta_start'][t].LB = 0
            self.variables['delta_start'][t].UB = 1

            self.variables['delta_stop'][t].VType = GRB.BINARY
            self.variables['delta_stop'][t].LB = 0
            self.variables['delta_stop'][t].UB = 1

        self.model.update()

        return mu_P, mu_Q


class FeasibilitySubproblem:
    """
    実行可能サブ問題

    サブ問題が実行不可能な場合に解く
    境界制約の違反量を最小化する
    """

    def __init__(self, params: SystemParameters, household_id: int):
        """
        初期化

        Args:
            params: システムパラメータ
            household_id: 需要家ID
        """
        self.params = params
        self.h = household_id
        self.model = None
        self.variables = {}
        self.constraints = {}

        self.P_DN_hat: Dict[int, float] = {}
        self.Q_DN_hat: Dict[int, float] = {}

    def build_model(self):
        """実行可能サブ問題モデルの構築"""
        self.model = gp.Model(f"FeasibilitySubproblem_H{self.h}")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)

        T = self.params.T
        h = self.h

        # ========== 決定変数（通常のサブ問題と同じ）==========
        self.variables['P_buy'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.BUY_MAX, name="P_buy"
        )
        self.variables['P_sell'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.SELL_MAX, name="P_sell"
        )
        self.variables['delta_buy'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_buy"
        )
        self.variables['delta_sell'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_sell"
        )

        # 蓄電池
        self.variables['P_ch'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.N_B_PCS, name="P_charge"
        )
        self.variables['P_dch'] = self.model.addVars(
            range(1, T+1), lb=0, ub=self.params.N_B_PCS, name="P_discharge"
        )
        self.variables['E_bat'] = self.model.addVars(
            range(1, T+1), lb=0.2*self.params.N_B, ub=self.params.N_B, name="E_battery"
        )
        self.variables['delta_ch'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_charge"
        )
        self.variables['delta_dch'] = self.model.addVars(
            range(1, T+1), vtype=GRB.BINARY, name="delta_discharge"
        )

        # ヒートポンプ
        self.variables['P_hp'] = self.model.addVars(
            range(1, T+1), lb=0, ub=10.0, name="P_hp"
        )
        self.variables['H_prod'] = self.model.addVars(
            range(1, T+1), lb=0, name="H_produce"
        )
        self.variables['H_tank'] = self.model.addVars(
            range(T+1), lb=0, name="H_tank"
        )
        self.variables['delta_hp'] = self.model.addVars(
            range(T+1), vtype=GRB.BINARY, name="delta_hp"
        )
        self.variables['delta_start'] = self.model.addVars(
            range(T+1), vtype=GRB.BINARY, name="delta_start"
        )
        self.variables['delta_stop'] = self.model.addVars(
            range(T+1), vtype=GRB.BINARY, name="delta_stop"
        )

        # 境界変数
        self.variables['P_H'] = self.model.addVars(
            range(1, T+1), lb=-GRB.INFINITY, name="P_H"
        )
        self.variables['Q_H'] = self.model.addVars(
            range(1, T+1), lb=-GRB.INFINITY, name="Q_H"
        )

        # 違反量変数（スラック変数）
        self.variables['alpha_P_plus'] = self.model.addVars(
            range(1, T+1), lb=0, name="alpha_P_plus"
        )
        self.variables['alpha_P_minus'] = self.model.addVars(
            range(1, T+1), lb=0, name="alpha_P_minus"
        )
        self.variables['alpha_Q_plus'] = self.model.addVars(
            range(1, T+1), lb=0, name="alpha_Q_plus"
        )
        self.variables['alpha_Q_minus'] = self.model.addVars(
            range(1, T+1), lb=0, name="alpha_Q_minus"
        )

        self.model.update()

        # 需要家内制約を追加
        self._add_power_balance_constraints()
        self._add_battery_constraints()
        self._add_heatpump_constraints()

    def _add_power_balance_constraints(self):
        """電力バランス制約"""
        T = self.params.T
        h = self.h
        P_buy = self.variables['P_buy']
        P_sell = self.variables['P_sell']
        delta_buy = self.variables['delta_buy']
        delta_sell = self.variables['delta_sell']
        P_ch = self.variables['P_ch']
        P_dch = self.variables['P_dch']
        P_hp = self.variables['P_hp']
        P_H = self.variables['P_H']
        Q_H = self.variables['Q_H']

        for t in range(1, T+1):
            self.model.addConstr(
                self.params.p_pv[t, h] + P_buy[t] + P_dch[t] ==
                self.params.p_dmd[t, h] + P_sell[t] + P_ch[t] + P_hp[t],
                name=f"power_balance_{t}"
            )

            self.model.addConstr(
                P_buy[t] <= delta_buy[t] * self.params.BUY_MAX,
                name=f"buy_limit_{t}"
            )
            self.model.addConstr(
                P_sell[t] <= delta_sell[t] * self.params.SELL_MAX,
                name=f"sell_limit_{t}"
            )
            self.model.addConstr(
                delta_buy[t] + delta_sell[t] <= 1,
                name=f"buy_sell_exclusive_{t}"
            )

            self.model.addConstr(
                P_H[t] == P_buy[t] - P_sell[t],
                name=f"P_H_def_{t}"
            )
            self.model.addConstr(
                Q_H[t] == 0.1 * P_H[t],
                name=f"Q_H_def_{t}"
            )

    def _add_battery_constraints(self):
        """蓄電池制約"""
        T = self.params.T
        P_ch = self.variables['P_ch']
        P_dch = self.variables['P_dch']
        E_bat = self.variables['E_bat']
        delta_ch = self.variables['delta_ch']
        delta_dch = self.variables['delta_dch']

        for t in range(1, T+1):
            self.model.addConstr(
                P_ch[t] <= self.params.N_B_PCS * delta_ch[t],
                name=f"charge_limit_{t}"
            )
            self.model.addConstr(
                P_dch[t] <= self.params.N_B_PCS * delta_dch[t],
                name=f"discharge_limit_{t}"
            )
            self.model.addConstr(
                delta_ch[t] + delta_dch[t] <= 1,
                name=f"charge_discharge_exclusive_{t}"
            )

            if t == 1:
                prev_soc = 0.5 * self.params.N_B
            else:
                prev_soc = E_bat[t-1]

            self.model.addConstr(
                E_bat[t] == prev_soc +
                0.5 * self.params.ETA_B * P_ch[t] -
                0.5 * P_dch[t] / self.params.ETA_B,
                name=f"battery_soc_{t}"
            )

        self.model.addConstr(
            E_bat[T] == 0.5 * self.params.N_B,
            name="battery_final"
        )

    def _add_heatpump_constraints(self):
        """ヒートポンプ制約"""
        T = self.params.T
        h = self.h
        P_hp = self.variables['P_hp']
        H_prod = self.variables['H_prod']
        H_tank = self.variables['H_tank']
        delta_hp = self.variables['delta_hp']
        delta_start = self.variables['delta_start']
        delta_stop = self.variables['delta_stop']

        self.model.addConstr(delta_hp[0] == 0)
        self.model.addConstr(delta_start[0] == 0)
        self.model.addConstr(delta_stop[0] == 0)
        self.model.addConstr(
            H_tank[0] == 0.5 * self.params.CW * self.params.VLT * 70
        )

        tank_max = self.params.CW * self.params.VLT * 70
        tank_min = 0.2 * tank_max

        for t in range(1, T+1):
            cop = self.params.cop[t]

            self.model.addConstr(
                P_hp[t] == self.params.XHP * delta_hp[t] +
                (H_prod[t] + self.params.R2 * self.params.CPH * delta_start[t]) /
                (self.params.CE * cop),
                name=f"hp_power_{t}"
            )

            self.model.addConstr(
                H_prod[t] >= 0.5 * self.params.R1 * self.params.CPH * delta_hp[t]
            )
            self.model.addConstr(
                H_prod[t] <= 0.5 * self.params.CPH * delta_hp[t]
            )

            self.model.addConstr(
                delta_hp[t] - delta_hp[t-1] == delta_start[t] - delta_stop[t]
            )
            self.model.addConstr(delta_start[t] + delta_stop[t] <= 1)

            self.model.addConstr(
                H_tank[t] == H_tank[t-1] + H_prod[t] - self.params.heat_dmd[t, h]
            )
            self.model.addConstr(H_tank[t] >= tank_min)
            self.model.addConstr(H_tank[t] <= tank_max)

        self.model.addConstr(H_tank[T] >= 0.5 * tank_max)

    def set_boundary_values(self, P_DN_hat: Dict[int, float], Q_DN_hat: Dict[int, float]):
        """境界変数の固定値を設定"""
        self.P_DN_hat = P_DN_hat.copy()
        self.Q_DN_hat = Q_DN_hat.copy()

    def add_boundary_constraints(self):
        """境界不等式制約を追加（緩和変数付き）"""
        T = self.params.T
        P_H = self.variables['P_H']
        Q_H = self.variables['Q_H']
        alpha_P_plus = self.variables['alpha_P_plus']
        alpha_P_minus = self.variables['alpha_P_minus']
        alpha_Q_plus = self.variables['alpha_Q_plus']
        alpha_Q_minus = self.variables['alpha_Q_minus']

        # 既存の境界制約を削除
        constrs_to_remove = []
        for name, constr in self.constraints.items():
            if name.startswith('feas_boundary_'):
                constrs_to_remove.append((name, constr))

        for name, constr in constrs_to_remove:
            self.model.remove(constr)
            del self.constraints[name]

        # 境界不等式制約（緩和変数付き）
        # P_H - P_DN_hat <= alpha_P_plus
        # P_DN_hat - P_H <= alpha_P_minus
        # Q_H - Q_DN_hat <= alpha_Q_plus
        # Q_DN_hat - Q_H <= alpha_Q_minus
        for t in range(1, T+1):
            c1 = self.model.addConstr(
                P_H[t] - self.P_DN_hat[t] <= alpha_P_plus[t],
                name=f"feas_boundary_P_plus_{t}"
            )
            self.constraints[f'feas_boundary_P_plus_{t}'] = c1

            c2 = self.model.addConstr(
                self.P_DN_hat[t] - P_H[t] <= alpha_P_minus[t],
                name=f"feas_boundary_P_minus_{t}"
            )
            self.constraints[f'feas_boundary_P_minus_{t}'] = c2

            c3 = self.model.addConstr(
                Q_H[t] - self.Q_DN_hat[t] <= alpha_Q_plus[t],
                name=f"feas_boundary_Q_plus_{t}"
            )
            self.constraints[f'feas_boundary_Q_plus_{t}'] = c3

            c4 = self.model.addConstr(
                self.Q_DN_hat[t] - Q_H[t] <= alpha_Q_minus[t],
                name=f"feas_boundary_Q_minus_{t}"
            )
            self.constraints[f'feas_boundary_Q_minus_{t}'] = c4

        self.model.update()

    def set_objective(self):
        """目的関数（違反量の最小化）"""
        T = self.params.T
        alpha_P_plus = self.variables['alpha_P_plus']
        alpha_P_minus = self.variables['alpha_P_minus']
        alpha_Q_plus = self.variables['alpha_Q_plus']
        alpha_Q_minus = self.variables['alpha_Q_minus']

        total_violation = gp.quicksum(
            alpha_P_plus[t] + alpha_P_minus[t] +
            alpha_Q_plus[t] + alpha_Q_minus[t]
            for t in range(1, T+1)
        )
        self.model.setObjective(total_violation, GRB.MINIMIZE)

    def solve(self) -> Tuple[int, Optional[float], Dict[str, float], Dict[str, float]]:
        """
        実行可能サブ問題を解く

        バイナリ変数を連続変数に緩和してから解き、双対変数を取得する

        Returns:
            (status, objective, lambda_P, lambda_Q)
        """
        T = self.params.T

        # バイナリ変数を連続変数に緩和（双対変数取得のため）
        for t in range(1, T+1):
            self.variables['delta_buy'][t].VType = GRB.CONTINUOUS
            self.variables['delta_buy'][t].LB = 0
            self.variables['delta_buy'][t].UB = 1

            self.variables['delta_sell'][t].VType = GRB.CONTINUOUS
            self.variables['delta_sell'][t].LB = 0
            self.variables['delta_sell'][t].UB = 1

            self.variables['delta_ch'][t].VType = GRB.CONTINUOUS
            self.variables['delta_ch'][t].LB = 0
            self.variables['delta_ch'][t].UB = 1

            self.variables['delta_dch'][t].VType = GRB.CONTINUOUS
            self.variables['delta_dch'][t].LB = 0
            self.variables['delta_dch'][t].UB = 1

        for t in range(T+1):
            self.variables['delta_hp'][t].VType = GRB.CONTINUOUS
            self.variables['delta_hp'][t].LB = 0
            self.variables['delta_hp'][t].UB = 1

            self.variables['delta_start'][t].VType = GRB.CONTINUOUS
            self.variables['delta_start'][t].LB = 0
            self.variables['delta_start'][t].UB = 1

            self.variables['delta_stop'][t].VType = GRB.CONTINUOUS
            self.variables['delta_stop'][t].LB = 0
            self.variables['delta_stop'][t].UB = 1

        self.model.update()
        self.model.optimize()

        status = self.model.Status
        objective = None
        lambda_P = {}
        lambda_Q = {}

        if status == GRB.OPTIMAL:
            objective = self.model.ObjVal

            # 双対変数を取得
            for t in range(1, T+1):
                c_plus = self.constraints.get(f'feas_boundary_P_plus_{t}')
                c_minus = self.constraints.get(f'feas_boundary_P_minus_{t}')

                lambda_1 = c_plus.Pi if c_plus else 0.0
                lambda_2 = c_minus.Pi if c_minus else 0.0
                lambda_P[t] = lambda_1 - lambda_2

                c_q_plus = self.constraints.get(f'feas_boundary_Q_plus_{t}')
                c_q_minus = self.constraints.get(f'feas_boundary_Q_minus_{t}')

                lambda_3 = c_q_plus.Pi if c_q_plus else 0.0
                lambda_4 = c_q_minus.Pi if c_q_minus else 0.0
                lambda_Q[t] = lambda_3 - lambda_4

        # バイナリ変数を元に戻す（次回の呼び出しに備えて）
        for t in range(1, T+1):
            self.variables['delta_buy'][t].VType = GRB.BINARY
            self.variables['delta_sell'][t].VType = GRB.BINARY
            self.variables['delta_ch'][t].VType = GRB.BINARY
            self.variables['delta_dch'][t].VType = GRB.BINARY

        for t in range(T+1):
            self.variables['delta_hp'][t].VType = GRB.BINARY
            self.variables['delta_start'][t].VType = GRB.BINARY
            self.variables['delta_stop'][t].VType = GRB.BINARY

        self.model.update()

        return status, objective, lambda_P, lambda_Q


class MasterProblem:
    """
    マスター問題（アグリゲータ）

    配電網制約と累積したカット平面を含む
    """

    def __init__(self, params: SystemParameters):
        """
        初期化

        Args:
            params: システムパラメータ
        """
        self.params = params
        self.model = None
        self.variables = {}
        self.constraints = {}

        # カット平面のリスト
        self.optimality_cuts: List[OptimalityCut] = []
        self.feasibility_cuts: List[FeasibilityCut] = []

    def build_model(self):
        """マスター問題モデルの構築"""
        self.model = gp.Model("MasterProblem")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('QCPDual', 1)
        self.model.setParam('DualReductions', 0)

        T = self.params.T
        H = self.params.H

        # ========== 決定変数 ==========
        # 境界変数（配電網側）
        # P_DN = P_buy - P_sell の範囲: [-SELL_MAX, BUY_MAX]
        self.variables['P_DN'] = self.model.addVars(
            range(1, T+1), range(H),
            lb=-self.params.SELL_MAX, ub=self.params.BUY_MAX,  # [-5.0, 5.0]
            name="P_DN"
        )
        # Q_DN = 0.1 * P_DN の範囲
        self.variables['Q_DN'] = self.model.addVars(
            range(1, T+1), range(H),
            lb=-0.1*self.params.SELL_MAX, ub=0.1*self.params.BUY_MAX,  # [-0.5, 0.5]
            name="Q_DN"
        )

        # 電圧変数
        self.variables['V'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="Voltage"
        )
        self.variables['V_dd'] = self.model.addVars(
            range(1, T+1), range(H), lb=-GRB.INFINITY, name="V_dd"
        )
        self.variables['V_UN'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="V_UN"
        )
        self.variables['V_LN'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="V_LN"
        )
        self.variables['V_ULV'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="V_ULV"
        )
        self.variables['V_LLV'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="V_LLV"
        )

        # 皮相電力
        self.variables['S_DN'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="S_DN"
        )
        self.variables['S_UN'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="S_UN"
        )
        self.variables['S_ULV'] = self.model.addVars(
            range(1, T+1), range(H), lb=0, name="S_ULV"
        )

        # 下界は需要家の電力コストの理論的最小値（売電のみの場合）
        lb_lbd = -self.params.SELL_PRICE * self.params.SELL_MAX * T * 0.5
        self.variables['LBD'] = self.model.addVars(range(H), lb=lb_lbd, name="LBD")

        self.model.update()

        # 制約条件
        self._add_voltage_constraints()
        self._add_line_capacity_constraints()
        self._add_qp_relationship()

    def _add_voltage_constraints(self):
        """電圧制約"""
        T = self.params.T
        H = self.params.H
        V = self.variables['V']
        V_dd = self.variables['V_dd']
        V_UN = self.variables['V_UN']
        V_LN = self.variables['V_LN']
        V_ULV = self.variables['V_ULV']
        V_LLV = self.variables['V_LLV']
        P_DN = self.variables['P_DN']
        Q_DN = self.variables['Q_DN']

        for t in range(1, T+1):
            for h in range(H):
                # 累積潮流
                P_cumulative = gp.quicksum(P_DN[t, j] for j in range(h, H))
                Q_cumulative = gp.quicksum(Q_DN[t, j] for j in range(h, H))

                # 電圧降下推定
                self.model.addConstr(
                    V_dd[t, h] ==
                    self.params.linear_ap[h] * P_cumulative +
                    self.params.linear_aq[h] * Q_cumulative +
                    self.params.linear_b[h],
                    name=f"voltage_drop_{t}_{h}"
                )

                # 電圧計算
                if h == 0:
                    self.model.addConstr(
                        V[t, h] == self.params.V_BASE - V_dd[t, h],
                        name=f"voltage_{t}_{h}"
                    )
                else:
                    self.model.addConstr(
                        V[t, h] == V[t, h-1] - V_dd[t, h],
                        name=f"voltage_{t}_{h}"
                    )

                # 電圧上限制約
                self.model.addConstr(
                    self.params.V_UL - V[t, h] == V_UN[t, h] - V_ULV[t, h],
                    name=f"voltage_upper_{t}_{h}"
                )
                # 電圧下限制約
                self.model.addConstr(
                    V[t, h] - self.params.V_LL == V_LN[t, h] - V_LLV[t, h],
                    name=f"voltage_lower_{t}_{h}"
                )

    def _add_line_capacity_constraints(self):
        """送電線容量制約"""
        T = self.params.T
        H = self.params.H
        S_DN = self.variables['S_DN']
        S_UN = self.variables['S_UN']
        S_ULV = self.variables['S_ULV']
        P_DN = self.variables['P_DN']
        Q_DN = self.variables['Q_DN']

        for t in range(1, T+1):
            for h in range(H):
                P_cumulative = gp.quicksum(P_DN[t, j] for j in range(h, H))
                Q_cumulative = gp.quicksum(Q_DN[t, j] for j in range(h, H))

                # 皮相電力の二次制約
                self.model.addQConstr(
                    S_DN[t, h] * S_DN[t, h] >=
                    P_cumulative * P_cumulative + Q_cumulative * Q_cumulative,
                    name=f"line_power_{t}_{h}"
                )

                # 送電線容量制約
                self.model.addConstr(
                    self.params.S_LINE - S_DN[t, h] == S_UN[t, h] - S_ULV[t, h],
                    name=f"line_capacity_{t}_{h}"
                )

    def _add_qp_relationship(self):
        """Q = 0.1 * P の関係を追加"""
        T = self.params.T
        H = self.params.H
        P_DN = self.variables['P_DN']
        Q_DN = self.variables['Q_DN']

        for t in range(1, T+1):
            for h in range(H):
                self.model.addConstr(
                    Q_DN[t, h] == 0.1 * P_DN[t, h],
                    name=f"QP_relation_{t}_{h}"
                )

    def add_optimality_cut(self, cut: OptimalityCut):
        """最適カットを追加"""
        self.optimality_cuts.append(cut)

        h = cut.household_id
        T = self.params.T
        LBD = self.variables['LBD']
        P_DN = self.variables['P_DN']
        Q_DN = self.variables['Q_DN']

        # LBD_h >= L_star - sum_t(mu_P[t] * P_DN[t,h] + mu_Q[t] * Q_DN[t,h])
        cut_expr = cut.L_star - gp.quicksum(
            cut.mu_P[t] * P_DN[t, h] + cut.mu_Q[t] * Q_DN[t, h]
            for t in range(1, T+1)
        )
        cut_id = len(self.optimality_cuts)
        self.model.addConstr(
            LBD[h] >= cut_expr,
            name=f"opt_cut_{h}_{cut_id}"
        )
        self.model.update()

    def add_feasibility_cut(self, cut: FeasibilityCut):
        """実行可能カットを追加"""
        self.feasibility_cuts.append(cut)

        h = cut.household_id
        T = self.params.T
        P_DN = self.variables['P_DN']
        Q_DN = self.variables['Q_DN']

        # L_star - sum_t(lambda_P[t] * P_DN[t,h] + lambda_Q[t] * Q_DN[t,h]) <= 0
        cut_expr = cut.L_star - gp.quicksum(
            cut.lambda_P[t] * P_DN[t, h] + cut.lambda_Q[t] * Q_DN[t, h]
            for t in range(1, T+1)
        )
        cut_id = len(self.feasibility_cuts)
        self.model.addConstr(
            cut_expr <= 0,
            name=f"feas_cut_{h}_{cut_id}"
        )
        self.model.update()

    def set_objective(self):
        """
        目的関数の設定

        min sum_h(LBD_h) + f_agg
        f_agg = DR項 + ペナルティ項
        """
        T = self.params.T
        H = self.params.H
        LBD = self.variables['LBD']
        P_DN = self.variables['P_DN']
        V_ULV = self.variables['V_ULV']
        V_LLV = self.variables['V_LLV']
        S_ULV = self.variables['S_ULV']

        # 需要家の下界の合計
        household_lb_sum = gp.quicksum(LBD[h] for h in range(H))

        # ペナルティ項
        penalty = gp.quicksum(
            V_ULV[t, h] + V_LLV[t, h] + S_ULV[t, h]
            for t in range(1, T+1) for h in range(H)
        )

        # DR項（P_DN = P_H = 買電 - 売電）
        dr_case = self.params.dr_case
        if dr_case == DRCase.NO_DR:
            dr_term = 0
        else:
            # DR期間の純買電電力
            dr_net_power = gp.quicksum(
                P_DN[t, h]
                for t in range(self.params.DR_START, self.params.DR_END+1)
                for h in range(H)
            )
            if dr_case == DRCase.DOWN_DR:
                dr_term = dr_net_power
            else:  # UP_DR
                dr_term = -dr_net_power

        objective = household_lb_sum + dr_term + penalty
        self.model.setObjective(objective, GRB.MINIMIZE)

    def solve(self) -> Tuple[int, Optional[float], Dict[str, Any]]:
        """
        マスター問題を解く

        Returns:
            (status, objective, solution)
        """
        self.model.optimize()

        status = self.model.Status
        objective = None
        solution = {}

        if status == GRB.OPTIMAL:
            objective = self.model.ObjVal
            solution = self._extract_solution()

        return status, objective, solution

    def _extract_solution(self) -> Dict[str, Any]:
        """解を抽出"""
        T = self.params.T
        H = self.params.H

        solution = {
            'P_DN': {h: {t: self.variables['P_DN'][t, h].X
                        for t in range(1, T+1)} for h in range(H)},
            'Q_DN': {h: {t: self.variables['Q_DN'][t, h].X
                        for t in range(1, T+1)} for h in range(H)},
            'V': {h: {t: self.variables['V'][t, h].X
                     for t in range(1, T+1)} for h in range(H)},
            'V_ULV': {h: {t: self.variables['V_ULV'][t, h].X
                        for t in range(1, T+1)} for h in range(H)},
            'V_LLV': {h: {t: self.variables['V_LLV'][t, h].X
                        for t in range(1, T+1)} for h in range(H)},
            'S_ULV': {h: {t: self.variables['S_ULV'][t, h].X
                        for t in range(1, T+1)} for h in range(H)},
            'LBD': {h: self.variables['LBD'][h].X for h in range(H)},
        }
        return solution


class BendersSolver:
    """
    Generalized Benders Decomposition ソルバー

    マスター問題とサブ問題を交互に解いて収束を判定
    """

    def __init__(self, params: SystemParameters):
        """
        初期化

        Args:
            params: システムパラメータ
        """
        self.params = params

        # マスター問題
        self.master = MasterProblem(params)
        self.master.build_model()

        # サブ問題（各需要家）
        self.subproblems: List[HouseholdSubproblem] = []
        self.feasibility_subproblems: List[FeasibilitySubproblem] = []
        for h in range(params.H):
            sub = HouseholdSubproblem(params, h)
            sub.build_model()
            self.subproblems.append(sub)

            feas_sub = FeasibilitySubproblem(params, h)
            feas_sub.build_model()
            self.feasibility_subproblems.append(feas_sub)

        # 収束履歴
        self.history = {
            'iteration': [],
            'lower_bound': [],
            'upper_bound': [],
            'gap': [],
            'subproblem_status': [],
            'time': []
        }

        # 最終解
        self.final_solution = None

    def initialize_boundary_variables(self) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        """
        境界変数の初期化

        上限と下限の平均値を初期境界変数とする
        P_H = P_buy - P_sell の範囲: [-SELL_MAX, BUY_MAX]
        Q_H = 0.1 * P_H

        Returns:
            (P_DN_hat, Q_DN_hat): 初期境界変数
        """
        logger.info("境界変数の初期化（上限・下限の平均値）...")

        P_DN_hat = {h: {} for h in range(self.params.H)}
        Q_DN_hat = {h: {} for h in range(self.params.H)}

        # P_H の範囲: [-SELL_MAX, BUY_MAX] = [-5.0, 5.0]
        # 平均値: (BUY_MAX + (-SELL_MAX)) / 2 = (5.0 - 5.0) / 2 = 0.0
        P_init = (self.params.BUY_MAX - self.params.SELL_MAX) / 2.0
        Q_init = 0.1 * P_init

        for h in range(self.params.H):
            for t in range(1, self.params.T+1):
                P_DN_hat[h][t] = P_init
                Q_DN_hat[h][t] = Q_init
            logger.info(f"  需要家{h}: P_DN_hat={P_init:.2f}, Q_DN_hat={Q_init:.2f}")

        return P_DN_hat, Q_DN_hat

    def solve(self) -> Dict[str, Any]:
        """
        GBDアルゴリズムの実行

        Returns:
            結果辞書
        """
        logger.info("="*60)
        logger.info(f"GBD最適化開始: {self.params.dr_case}")
        logger.info("="*60)

        start_time = time.time()
        start_cpu_time = time.process_time()

        # 境界変数の初期化
        P_DN_hat, Q_DN_hat = self.initialize_boundary_variables()

        # 上界・下界の初期化
        LB = -float('inf')
        UB = float('inf')
        gap = float('inf')

        # 反復
        for k in range(1, self.params.MAX_ITERATIONS + 1):
            iter_start = time.time()
            logger.info(f"\n--- 反復 {k} ---")

            # ステップ2: サブ問題を解く
            subproblem_objectives = {}
            all_feasible = True

            for h in range(self.params.H):
                sub = self.subproblems[h]

                # 境界変数を設定
                sub.set_boundary_values(P_DN_hat[h], Q_DN_hat[h])
                sub.add_boundary_constraints()

                # サブ問題を解く
                status, obj, solution = sub.solve()

                if status == GRB.OPTIMAL:
                    subproblem_objectives[h] = obj
                    logger.info(f"  需要家{h}: 最適 (obj={obj:.2f})")

                    # 双対変数を取得
                    mu_P, mu_Q = sub.get_dual_variables()

                    # 最適カットの定数項を計算
                    # L_star = obj + sum_t(mu_P[t] * P_DN_hat[t] + mu_Q[t] * Q_DN_hat[t])
                    L_star = obj + sum(
                        mu_P[t] * P_DN_hat[h][t] + mu_Q[t] * Q_DN_hat[h][t]
                        for t in range(1, self.params.T+1)
                    )

                    # 最適カットを追加
                    cut = OptimalityCut(
                        household_id=h,
                        iteration=k,
                        L_star=L_star,
                        mu_P=mu_P,
                        mu_Q=mu_Q
                    )
                    self.master.add_optimality_cut(cut)

                else:
                    all_feasible = False
                    logger.info(f"  需要家{h}: 実行不可能 -> 実行可能サブ問題を解く")

                    # 実行可能サブ問題を解く
                    feas_sub = self.feasibility_subproblems[h]
                    feas_sub.set_boundary_values(P_DN_hat[h], Q_DN_hat[h])
                    feas_sub.add_boundary_constraints()
                    feas_sub.set_objective()

                    feas_status, feas_obj, lambda_P, lambda_Q = feas_sub.solve()

                    if feas_status == GRB.OPTIMAL:
                        # 実行可能サブ問題の最適解を使用
                        feas_status, feas_obj, lambda_P, lambda_Q, P_H_opt, Q_H_opt = feas_sub.solve()

                        L_star = sum(
                            lambda_P[t] * P_H_opt[t] + lambda_Q[t] * Q_H_opt[t]
                            for t in range(1, self.params.T+1)
                        )

                        # 実行可能カットを追加
                        cut = FeasibilityCut(
                            household_id=h,
                            iteration=k,
                            L_star=L_star,
                            lambda_P=lambda_P,
                            lambda_Q=lambda_Q
                        )
                        self.master.add_feasibility_cut(cut)
                        logger.info(f"    実行可能カット追加 (violation={feas_obj:.6f})")
                    else:
                        logger.error(f"    実行可能サブ問題も失敗: status={feas_status}")

            # 上界の更新（すべてのサブ問題が実行可能な場合）
            if all_feasible:
                # ペナルティ項を計算（マスター問題の解から）
                penalty = sum(
                    self.master.variables['V_ULV'][t, h].X +
                    self.master.variables['V_LLV'][t, h].X +
                    self.master.variables['S_ULV'][t, h].X
                    for t in range(1, self.params.T+1)
                    for h in range(self.params.H)
                ) if self.master.model.Status == GRB.OPTIMAL else 0

                # DR項
                dr_term = 0
                if self.params.dr_case != DRCase.NO_DR and self.master.model.Status == GRB.OPTIMAL:
                    dr_net = sum(
                        self.master.variables['P_DN'][t, h].X
                        for t in range(self.params.DR_START, self.params.DR_END+1)
                        for h in range(self.params.H)
                    )
                    if self.params.dr_case == DRCase.DOWN_DR:
                        dr_term = dr_net
                    else:
                        dr_term = -dr_net

                new_UB = sum(subproblem_objectives.values()) + dr_term + penalty
                if new_UB < UB:
                    UB = new_UB
                    logger.info(f"  上界更新: UB = {UB:.4f}")

            # ステップ3: マスター問題を解く
            self.master.set_objective()
            master_status, master_obj, master_solution = self.master.solve()

            if master_status == GRB.OPTIMAL:
                LB = master_obj
                logger.info(f"  マスター問題: 最適 (LB={LB:.4f})")

                # 境界変数を更新
                for h in range(self.params.H):
                    for t in range(1, self.params.T+1):
                        P_DN_hat[h][t] = master_solution['P_DN'][h][t]
                        Q_DN_hat[h][t] = master_solution['Q_DN'][h][t]

            else:
                logger.error(f"  マスター問題: 失敗 (status={master_status})")
                # デバッグ情報出力
                if master_status == GRB.INFEASIBLE:
                    logger.error("  マスター問題が実行不可能です。IISを計算します...")
                    try:
                        self.master.model.computeIIS()
                        self.master.model.write("master_iis.ilp")
                        logger.error("  IISをmaster_iis.ilpに出力しました")
                    except Exception as e:
                        logger.error(f"  IIS計算エラー: {e}")
                elif master_status == GRB.UNBOUNDED or master_status == GRB.INF_OR_UNBD:
                    logger.error("  マスター問題が非有界です。")
                    # 非有界の場合は、境界の設定を確認
                    logger.error(f"  最適カット数: {len(self.master.optimality_cuts)}")
                    logger.error(f"  実行可能カット数: {len(self.master.feasibility_cuts)}")
                break

            # 収束判定
            if UB != float('inf') and LB != -float('inf'):
                gap = abs(UB - LB) / max(1.0, abs(UB))
            else:
                gap = float('inf')

            iter_time = time.time() - iter_start
            self.history['iteration'].append(k)
            self.history['lower_bound'].append(LB)
            self.history['upper_bound'].append(UB)
            self.history['gap'].append(gap)
            self.history['subproblem_status'].append(all_feasible)
            self.history['time'].append(iter_time)

            logger.info(f"  LB={LB:.4f}, UB={UB:.4f}, Gap={gap:.6f}, Time={iter_time:.2f}s")

            if gap < self.params.TOLERANCE:
                logger.info(f"\n収束達成 (gap={gap:.6f} < {self.params.TOLERANCE})")
                break

        # 最終結果
        total_real_time = time.time() - start_time
        total_cpu_time = time.process_time() - start_cpu_time

        results = {
            'case': str(self.params.dr_case),
            'case_description': self.params.dr_case.description,
            'status': GRB.OPTIMAL if gap < self.params.TOLERANCE else GRB.TIME_LIMIT,
            'objective': (UB + LB) / 2 if UB != float('inf') else LB,
            'lower_bound': LB,
            'upper_bound': UB,
            'gap': gap,
            'iterations': k,
            'real_time': total_real_time,
            'cpu_time': total_cpu_time,
            'optimality_cuts': len(self.master.optimality_cuts),
            'feasibility_cuts': len(self.master.feasibility_cuts),
            'history': self.history
        }

        # 詳細解の抽出
        if all_feasible and master_status == GRB.OPTIMAL:
            results['detail'] = self._extract_final_solution(master_solution)

        logger.info(f"\n最終結果:")
        logger.info(f"  目的関数値: {results['objective']:.4f}")
        logger.info(f"  反復回数: {results['iterations']}")
        logger.info(f"  Real Time: {total_real_time:.2f}s, CPU Time: {total_cpu_time:.2f}s")
        logger.info(f"  最適カット数: {results['optimality_cuts']}, 実行可能カット数: {results['feasibility_cuts']}")

        return results

    def _extract_final_solution(self, master_solution: Dict) -> Dict:
        """最終解を抽出"""
        T = self.params.T
        H = self.params.H

        detail = {
            'P_AG_buy': {},
            'P_AG_sell': {},
            'voltage': {},
            'battery_soc': {},
            'hp_operation': {},
            'household_cost': {},
            'P_H': {}, 'Q_H': {},
            'P_DN': {}, 'Q_DN': {},
            'P_buy': {}, 'P_sell': {},
            'p_pv': {}, 'p_dmd': {},
            'P_ch': {}, 'P_dch': {},
            'P_hp': {},
            'H_tank': {}, 'H_prod': {}
        }

        # マスター問題からの値
        for h in range(H):
            detail['P_DN'][h] = master_solution['P_DN'][h]
            detail['Q_DN'][h] = master_solution['Q_DN'][h]
            detail['voltage'][h] = master_solution['V'][h]

        # サブ問題からの値
        for h in range(H):
            sub = self.subproblems[h]
            if sub.model.Status == GRB.OPTIMAL:
                for key in ['P_buy', 'P_sell', 'P_ch', 'P_dch', 'P_hp', 'P_H', 'Q_H']:
                    detail[key][h] = {t: sub.variables[key][t].X for t in range(1, T+1)}
                detail['battery_soc'][h] = {t: sub.variables['E_bat'][t].X for t in range(1, T+1)}
                detail['hp_operation'][h] = {t: int(sub.variables['delta_hp'][t].X) for t in range(T+1)}
                detail['H_tank'][h] = {t: sub.variables['H_tank'][t].X for t in range(T+1)}
                detail['H_prod'][h] = {t: sub.variables['H_prod'][t].X for t in range(1, T+1)}

                # 需要家コスト
                cost = sum(
                    self.params.buy_price[t] * detail['P_buy'][h][t] -
                    self.params.SELL_PRICE * detail['P_sell'][h][t]
                    for t in range(1, T+1)
                )
                detail['household_cost'][h] = cost

        # パラメータ
        for h in range(H):
            detail['p_pv'][h] = {t: self.params.p_pv[t, h] for t in range(1, T+1)}
            detail['p_dmd'][h] = {t: self.params.p_dmd[t, h] for t in range(1, T+1)}

        # アグリゲータレベル
        for t in range(1, T+1):
            total_buy = sum(
                detail['P_buy'].get(h, {}).get(t, 0) for h in range(H)
            )
            total_sell = sum(
                detail['P_sell'].get(h, {}).get(t, 0) for h in range(H)
            )
            detail['P_AG_buy'][t] = total_buy
            detail['P_AG_sell'][t] = total_sell

        return detail

    def export_results(self, results: Dict, filename_prefix: str = "gbd"):
        """結果のエクスポート"""
        if 'detail' not in results:
            logger.warning("詳細解がないため、結果をエクスポートできません")
            return

        case_name = str(self.params.dr_case)
        output_dir = f"../data/output/gbd/{case_name}"
        os.makedirs(output_dir, exist_ok=True)

        # サマリーCSV
        summary_df = pd.DataFrame({
            'Metric': ['Case', 'Description', 'Objective', 'Lower Bound', 'Upper Bound',
                      'Gap', 'Iterations', 'Real Time', 'CPU Time',
                      'Optimality Cuts', 'Feasibility Cuts', 'Total Household Cost'],
            'Value': [
                results['case'],
                results['case_description'],
                results['objective'],
                results['lower_bound'],
                results['upper_bound'],
                results['gap'],
                results['iterations'],
                results['real_time'],
                results['cpu_time'],
                results['optimality_cuts'],
                results['feasibility_cuts'],
                sum(results['detail']['household_cost'].values())
            ]
        })
        summary_df.to_csv(f"{output_dir}/{filename_prefix}_summary.csv",
                         index=False, encoding='utf-8')

        # 収束履歴CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(f"{output_dir}/{filename_prefix}_convergence.csv",
                         index=False, encoding='utf-8')

        # 時系列データ
        detail = results['detail']
        time_series_data = []
        for t in range(1, self.params.T+1):
            row = {
                'Time': t,
                'AG_Buy': detail['P_AG_buy'].get(t, 0),
                'AG_Sell': detail['P_AG_sell'].get(t, 0)
            }
            for h in range(self.params.H):
                row[f'Voltage_H{h}'] = detail['voltage'].get(h, {}).get(t, 0)
                row[f'Battery_H{h}'] = detail['battery_soc'].get(h, {}).get(t, 0)
                row[f'HP_Status_H{h}'] = detail['hp_operation'].get(h, {}).get(t, 0)
                row[f'P_H_h{h}'] = detail['P_H'].get(h, {}).get(t, 0)
                row[f'P_DN_h{h}'] = detail['P_DN'].get(h, {}).get(t, 0)
            time_series_data.append(row)

        time_df = pd.DataFrame(time_series_data)
        time_df.to_csv(f"{output_dir}/{filename_prefix}_timeseries.csv",
                       index=False, encoding='utf-8')

        logger.info(f"結果を{output_dir}に保存しました")

    def plot_convergence(self, results: Dict):
        """収束履歴のプロット"""
        history = results['history']

        case_name = str(self.params.dr_case)
        output_dir = f"../data/output/gbd/{case_name}/plots"
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # 上界・下界の推移
        ax = axes[0]
        iterations = history['iteration']
        ax.plot(iterations, history['lower_bound'], 'b-o', label='Lower Bound')
        ax.plot(iterations, history['upper_bound'], 'r-o', label='Upper Bound')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title(f'{case_name} - Convergence History')
        ax.legend()
        ax.grid(True)

        # ギャップの推移
        ax = axes[1]
        ax.semilogy(iterations, history['gap'], 'g-o', label='Gap')
        ax.axhline(self.params.TOLERANCE, color='r', linestyle='--', label='Tolerance')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gap')
        ax.set_title('Optimality Gap')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/convergence.png", dpi=100)
        plt.close()

        logger.info(f"収束グラフを{output_dir}/convergence.pngに保存しました")

    def plot_results(self, results: Dict):
        """結果の可視化"""
        if 'detail' not in results:
            return

        detail = results['detail']
        times = list(range(1, self.params.T + 1))

        case_name = str(self.params.dr_case)
        output_dir = f"../data/output/gbd/{case_name}/plots"
        os.makedirs(output_dir, exist_ok=True)

        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True

        for h in range(self.params.H):
            fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

            # 1. Power Balance
            ax = axes[0]
            p_pv = np.array([detail['p_pv'][h][t] for t in times])
            p_buy = np.array([detail['P_buy'].get(h, {}).get(t, 0) for t in times])
            p_dch = np.array([detail['P_dch'].get(h, {}).get(t, 0) for t in times])
            p_dmd = np.array([detail['p_dmd'][h][t] for t in times])
            p_sell = np.array([detail['P_sell'].get(h, {}).get(t, 0) for t in times])
            p_ch = np.array([detail['P_ch'].get(h, {}).get(t, 0) for t in times])
            p_hp = np.array([detail['P_hp'].get(h, {}).get(t, 0) for t in times])

            ax.bar(times, p_pv, label='PV', color='gold', alpha=0.9)
            ax.bar(times, p_buy, bottom=p_pv, label='Buy', color='red', alpha=0.6)
            ax.bar(times, p_dch, bottom=p_pv + p_buy, label='Discharge', color='blue', alpha=0.6)
            ax.bar(times, -p_dmd, label='Demand', color='gray', alpha=0.5)
            ax.bar(times, -p_sell, bottom=-p_dmd, label='Sell', color='cyan', alpha=0.6)
            ax.bar(times, -p_ch, bottom=-p_dmd - p_sell, label='Charge', color='green', alpha=0.6)
            ax.bar(times, -p_hp, bottom=-p_dmd - p_sell - p_ch, label='HP', color='orange', alpha=0.8)

            if self.params.dr_case != DRCase.NO_DR:
                ax.axvspan(self.params.DR_START, self.params.DR_END, alpha=0.2, color='yellow', label='DR Period')

            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel('Power [kW]')
            ax.set_title(f'{case_name} - Household {h}: Power Balance')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize='small')

            # 2. Battery SOC
            ax = axes[1]
            soc = [detail['battery_soc'].get(h, {}).get(t, 0) for t in times]
            ax.plot(times, soc, label='SOC', color='tab:blue', linewidth=2)
            ax.axhline(self.params.N_B, color='gray', linestyle=':', label='Max')
            ax.axhline(0.2 * self.params.N_B, color='gray', linestyle=':', label='Min')
            ax.set_ylabel('Energy [kWh]')
            ax.set_title(f'{case_name} - Household {h}: Battery SOC')
            ax.legend()

            # 3. HP Tank
            ax = axes[2]
            tank_max = self.params.CW * self.params.VLT * 70
            h_tank = [detail['H_tank'].get(h, {}).get(t, 0) for t in times]
            ax.plot(times, h_tank, label='Tank Level', color='orange', linewidth=2)
            ax.axhline(tank_max, color='red', linestyle=':', label='Max')
            ax.fill_between(times, h_tank, 0, color='orange', alpha=0.1)
            ax.set_ylabel('Thermal [MJ]')
            ax.set_title(f'{case_name} - Household {h}: HP Tank Level')
            ax.legend()

            # 4. Voltage
            ax = axes[3]
            v_dn = [detail['voltage'].get(h, {}).get(t, 0) for t in times]
            ax.plot(times, v_dn, label='Voltage', color='magenta', linewidth=2)
            ax.axhline(self.params.V_UL, color='red', linestyle=':', label='Upper Limit')
            ax.axhline(self.params.V_LL, color='blue', linestyle=':', label='Lower Limit')
            ax.set_ylabel('Voltage [V]')
            ax.set_title(f'{case_name} - Household {h}: Voltage')
            ax.legend()

            # 5. P_H vs P_DN
            ax = axes[4]
            p_h = [detail['P_H'].get(h, {}).get(t, 0) for t in times]
            p_dn = [detail['P_DN'].get(h, {}).get(t, 0) for t in times]
            ax.plot(times, p_h, label='P_H', color='black', linewidth=2, alpha=0.7)
            ax.plot(times, p_dn, label='P_DN', color='orange', linestyle='--', linewidth=2)
            ax.set_ylabel('Active Power [kW]')
            ax.set_xlabel('Time Slot')
            ax.set_title(f'{case_name} - Household {h}: Boundary Power')
            ax.legend()

            plt.tight_layout()
            plt.savefig(f"{output_dir}/household_{h}_analysis.png", dpi=100)
            plt.close(fig)

        logger.info(f"グラフを{output_dir}に保存しました")


def create_comparison_summary(all_results: Dict[DRCase, Dict]):
    """3ケースの比較サマリーを作成"""
    output_dir = "../data/output/gbd"
    os.makedirs(output_dir, exist_ok=True)

    comparison_data = []
    for case, results in all_results.items():
        if results.get('status') in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            comparison_data.append({
                'Case': str(case),
                'Description': case.description,
                'Objective': results['objective'],
                'Lower_Bound': results['lower_bound'],
                'Upper_Bound': results['upper_bound'],
                'Gap': results['gap'],
                'Iterations': results['iterations'],
                'Real_Time_sec': results['real_time'],
                'CPU_Time_sec': results['cpu_time'],
                'Optimality_Cuts': results['optimality_cuts'],
                'Feasibility_Cuts': results['feasibility_cuts'],
                'Total_Household_Cost': sum(results.get('detail', {}).get('household_cost', {}).values())
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"{output_dir}/comparison_summary.csv", index=False, encoding='utf-8')

    logger.info("="*60)
    logger.info("3ケース比較サマリー")
    logger.info("="*60)
    print(comparison_df.to_string(index=False))
    logger.info(f"比較サマリーを{output_dir}/comparison_summary.csvに保存しました")

    return comparison_df


def main():
    """メイン実行関数：3ケースを順番に実行"""
    all_results = {}

    cases = [DRCase.NO_DR, DRCase.DOWN_DR, DRCase.UP_DR]

    for case in cases:
        print("\n" + "="*70)
        print(f"実行中: {case} - {case.description}")
        print("="*70 + "\n")

        try:
            # システムパラメータの初期化
            params = SystemParameters(dr_case=case)

            # GBDソルバーの作成と実行
            solver = BendersSolver(params)
            results = solver.solve()

            # 結果のエクスポート
            solver.export_results(results)
            solver.plot_convergence(results)
            solver.plot_results(results)

            all_results[case] = results

            print(f"\n{case} 完了:")
            print(f"  目的関数値: {results.get('objective', 'N/A'):.4f}")
            print(f"  反復回数: {results.get('iterations', 'N/A')}")
            print(f"  Real Time: {results['real_time']:.2f}秒, CPU Time: {results['cpu_time']:.2f}秒")

        except Exception as e:
            logger.error(f"{case}の実行中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            all_results[case] = {'status': -1, 'error': str(e)}

    # 比較サマリーの作成
    if all_results:
        create_comparison_summary(all_results)

    return all_results


if __name__ == "__main__":
    results = main()