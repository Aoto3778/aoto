"""
Generalized Benders Decomposition (GBD) for Distribution Network Optimization
配電網最適化のための一般化ベンダーズ分解法

構造：
- マスター問題：アグリゲータが配電網を運用し、境界変数 P_DN, Q_DN を決定
- サブ問題：各需要家が独立に電力コスト最小化（境界変数は固定）
- カット平面：最適カット（実行可能時）と実行可能カット（実行不可能時）

3ケース対応：
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
from enum import Enum
from dataclasses import dataclass, field
import os
import copy

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DRCase(Enum):
    """DRケースの定義"""
    NO_DR = 1      # Case1: DRなし
    DOWN_DR = 2    # Case2: 下げDR（買電-売電を最小化）
    UP_DR = 3      # Case3: 上げDR（買電-売電を最大化）

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
    """最適カット平面のデータ構造"""
    L_star: float                      # L_{H,p}^* = UB_{H,h} + μ^T * y_hat_DN
    mu_P: Dict[int, float]             # μ_P^{h,t}: 有効電力の双対変数 (t -> value)
    mu_Q: Dict[int, float]             # μ_Q^{h,t}: 無効電力の双対変数 (t -> value)
    household_idx: int                 # 需要家インデックス
    iteration: int                     # 反復回数


@dataclass
class FeasibilityCut:
    """実行可能カット平面のデータ構造"""
    L_star: float                      # L_*^{h,q} = λ^T * x_H (サブ問題での値)
    lambda_P: Dict[int, float]         # λ_P^{h}: 有効電力の双対変数 (t -> value)
    lambda_Q: Dict[int, float]         # λ_Q^{h}: 無効電力の双対変数 (t -> value)
    household_idx: int                 # 需要家インデックス
    iteration: int                     # 反復回数


@dataclass
class SubproblemResult:
    """サブ問題の結果"""
    is_feasible: bool
    objective_value: Optional[float] = None
    optimality_cut: Optional[OptimalityCut] = None
    feasibility_cut: Optional[FeasibilityCut] = None
    solution: Optional[Dict] = None


class SystemParameters:
    """システムパラメータとデータ管理"""

    def __init__(self):
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

        # 境界変数の初期値範囲
        self.P_DN_MIN = -self.SELL_MAX
        self.P_DN_MAX = self.BUY_MAX
        self.Q_DN_MIN = -0.5
        self.Q_DN_MAX = 0.5

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
    需要家サブ問題
    
    境界変数 P_DN_hat, Q_DN_hat を固定して需要家の電力コストを最小化
    実行可能時：双対変数μを取得して最適カット生成
    実行不可能時：FeasibilitySubproblemを呼び出す
    """

    def __init__(self, params: SystemParameters, household_idx: int, dr_case: DRCase):
        self.params = params
        self.h = household_idx
        self.dr_case = dr_case
        self.T = params.T

    def solve(self, P_DN_hat: Dict[int, float], Q_DN_hat: Dict[int, float], 
              iteration: int) -> SubproblemResult:
        """
        サブ問題を解く
        
        Args:
            P_DN_hat: 固定された有効電力境界変数 {t: value}
            Q_DN_hat: 固定された無効電力境界変数 {t: value}
            iteration: 現在の反復回数
            
        Returns:
            SubproblemResult: 結果（実行可能性、目的関数値、カット情報）
        """
        # ステップ1: MIPとして解く
        model = self._build_model(P_DN_hat, Q_DN_hat, is_mip=True)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            # 実行不可能 → FeasibilitySubproblemへ
            logger.debug(f"  需要家{self.h}: サブ問題が実行不可能 → 実行可能カット生成")
            return self._solve_feasibility_subproblem(P_DN_hat, Q_DN_hat, iteration)

        if model.Status != GRB.OPTIMAL:
            logger.warning(f"  需要家{self.h}: サブ問題のステータス異常 ({model.Status})")
            return SubproblemResult(is_feasible=False)

        # MIP最適解を取得
        objective_value = model.ObjVal
        binary_values = self._extract_binary_values(model)
        solution = self._extract_solution(model)

        # ステップ2: バイナリ変数を固定してLPとして再度解く（双対変数取得）
        model_lp = self._build_model(P_DN_hat, Q_DN_hat, is_mip=False, 
                                      fixed_binaries=binary_values)
        model_lp.optimize()

        if model_lp.Status != GRB.OPTIMAL:
            logger.warning(f"  需要家{self.h}: LP再最適化失敗")
            # MIPの結果を使用（双対変数なし）
            return SubproblemResult(
                is_feasible=True,
                objective_value=objective_value,
                solution=solution
            )

        # 双対変数を取得して最適カットを生成
        mu_P, mu_Q = self._extract_dual_variables(model_lp)

        # L_{H,p}^* = UB_{H,h} + μ^T * y_hat_DN
        L_star = objective_value
        for t in range(1, self.T + 1):
            L_star += mu_P.get(t, 0.0) * P_DN_hat[t]
            L_star += mu_Q.get(t, 0.0) * Q_DN_hat[t]

        optimality_cut = OptimalityCut(
            L_star=L_star,
            mu_P=mu_P,
            mu_Q=mu_Q,
            household_idx=self.h,
            iteration=iteration
        )

        logger.debug(f"  需要家{self.h}: 最適カット生成 (目的関数={objective_value:.2f})")

        return SubproblemResult(
            is_feasible=True,
            objective_value=objective_value,
            optimality_cut=optimality_cut,
            solution=solution
        )

    def _build_model(self, P_DN_hat: Dict[int, float], Q_DN_hat: Dict[int, float],
                     is_mip: bool, fixed_binaries: Optional[Dict] = None) -> gp.Model:
        """サブ問題モデルを構築"""
        model = gp.Model(f"Subproblem_H{self.h}")
        model.setParam('OutputFlag', 0)

        if not is_mip:
            # LP用設定
            model.setParam('Method', 2)  # Interior point
            model.setParam('QCPDual', 1)  # 双対変数取得

        p = self.params
        T = self.T
        h = self.h

        # ========== 決定変数 ==========
        # 買電・売電
        P_buy = model.addVars(range(1, T+1), lb=0, ub=p.BUY_MAX, name="P_buy")
        P_sell = model.addVars(range(1, T+1), lb=0, ub=p.SELL_MAX, name="P_sell")

        if is_mip and fixed_binaries is None:
            delta_buy = model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_buy")
            delta_sell = model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_sell")
        else:
            delta_buy = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_buy")
            delta_sell = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_sell")

        # 蓄電池
        P_ch = model.addVars(range(1, T+1), lb=0, ub=p.N_B_PCS, name="P_charge")
        P_dch = model.addVars(range(1, T+1), lb=0, ub=p.N_B_PCS, name="P_discharge")
        E_bat = model.addVars(range(1, T+1), lb=0.2*p.N_B, ub=p.N_B, name="E_battery")

        if is_mip and fixed_binaries is None:
            delta_ch = model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_charge")
            delta_dch = model.addVars(range(1, T+1), vtype=GRB.BINARY, name="delta_discharge")
        else:
            delta_ch = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_charge")
            delta_dch = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_discharge")

        # ヒートポンプ
        P_hp = model.addVars(range(1, T+1), lb=0, ub=10.0, name="P_hp")
        H_prod = model.addVars(range(1, T+1), lb=0, name="H_produce")
        H_tank = model.addVars(range(T+1), lb=0, name="H_tank")

        if is_mip and fixed_binaries is None:
            delta_hp = model.addVars(range(T+1), vtype=GRB.BINARY, name="delta_hp")
            delta_start = model.addVars(range(T+1), vtype=GRB.BINARY, name="delta_start")
            delta_stop = model.addVars(range(T+1), vtype=GRB.BINARY, name="delta_stop")
        else:
            delta_hp = model.addVars(range(T+1), lb=0, ub=1, name="delta_hp")
            delta_start = model.addVars(range(T+1), lb=0, ub=1, name="delta_start")
            delta_stop = model.addVars(range(T+1), lb=0, ub=1, name="delta_stop")

        # 境界変数（需要家側）
        P_H = model.addVars(range(1, T+1), lb=-GRB.INFINITY, name="P_H")
        Q_H = model.addVars(range(1, T+1), lb=-GRB.INFINITY, name="Q_H")

        model.update()

        # バイナリ変数の固定（LP用）
        if fixed_binaries is not None:
            for t in range(1, T+1):
                delta_buy[t].lb = fixed_binaries['delta_buy'][t]
                delta_buy[t].ub = fixed_binaries['delta_buy'][t]
                delta_sell[t].lb = fixed_binaries['delta_sell'][t]
                delta_sell[t].ub = fixed_binaries['delta_sell'][t]
                delta_ch[t].lb = fixed_binaries['delta_ch'][t]
                delta_ch[t].ub = fixed_binaries['delta_ch'][t]
                delta_dch[t].lb = fixed_binaries['delta_dch'][t]
                delta_dch[t].ub = fixed_binaries['delta_dch'][t]
            for t in range(T+1):
                delta_hp[t].lb = fixed_binaries['delta_hp'][t]
                delta_hp[t].ub = fixed_binaries['delta_hp'][t]
                delta_start[t].lb = fixed_binaries['delta_start'][t]
                delta_start[t].ub = fixed_binaries['delta_start'][t]
                delta_stop[t].lb = fixed_binaries['delta_stop'][t]
                delta_stop[t].ub = fixed_binaries['delta_stop'][t]

        # ========== 制約条件 ==========
        # 境界制約を保存（双対変数取得用）
        self._boundary_P_constrs = {}
        self._boundary_Q_constrs = {}

        for t in range(1, T+1):
            # 電力バランス
            model.addConstr(
                p.p_pv[t, h] + P_buy[t] + P_dch[t] ==
                p.p_dmd[t, h] + P_sell[t] + P_ch[t] + P_hp[t],
                name=f"power_balance_{t}"
            )

            # 買電売電同時禁止
            model.addConstr(P_buy[t] <= delta_buy[t] * p.BUY_MAX)
            model.addConstr(P_sell[t] <= delta_sell[t] * p.SELL_MAX)
            model.addConstr(delta_buy[t] + delta_sell[t] <= 1)

            # P_H, Q_H の定義
            model.addConstr(P_H[t] == P_buy[t] - P_sell[t], name=f"P_H_def_{t}")
            model.addConstr(Q_H[t] == 0.1 * P_H[t], name=f"Q_H_def_{t}")

            # 境界等式制約: P_H = P_DN_hat, Q_H = Q_DN_hat
            self._boundary_P_constrs[t] = model.addConstr(
                P_H[t] == P_DN_hat[t], name=f"boundary_P_{t}"
            )
            self._boundary_Q_constrs[t] = model.addConstr(
                Q_H[t] == Q_DN_hat[t], name=f"boundary_Q_{t}"
            )

            # 蓄電池制約
            model.addConstr(P_ch[t] <= p.N_B_PCS * delta_ch[t])
            model.addConstr(P_dch[t] <= p.N_B_PCS * delta_dch[t])
            model.addConstr(delta_ch[t] + delta_dch[t] <= 1)

            # SOC更新
            if t == 1:
                prev_soc = 0.5 * p.N_B
            else:
                prev_soc = E_bat[t-1]

            model.addConstr(
                E_bat[t] == prev_soc +
                0.5 * p.ETA_B * P_ch[t] -
                0.5 * P_dch[t] / p.ETA_B,
                name=f"battery_soc_{t}"
            )

        # 蓄電池終端条件
        model.addConstr(E_bat[T] == 0.5 * p.N_B, name="battery_final")

        # ヒートポンプ制約
        model.addConstr(delta_hp[0] == 0)
        model.addConstr(delta_start[0] == 0)
        model.addConstr(delta_stop[0] == 0)
        model.addConstr(H_tank[0] == 0.5 * p.CW * p.VLT * 70, name="hp_tank_init")

        for t in range(1, T+1):
            cop = p.cop[t]

            # HP消費電力
            model.addConstr(
                P_hp[t] == p.XHP * delta_hp[t] +
                (H_prod[t] + p.R2 * p.CPH * delta_start[t]) / (p.CE * cop),
                name=f"hp_power_{t}"
            )

            # 熱製造量制約
            model.addConstr(H_prod[t] >= 0.5 * p.R1 * p.CPH * delta_hp[t])
            model.addConstr(H_prod[t] <= 0.5 * p.CPH * delta_hp[t])

            # 運転状態の遷移
            model.addConstr(
                delta_hp[t] - delta_hp[t-1] == delta_start[t] - delta_stop[t],
                name=f"hp_state_{t}"
            )
            model.addConstr(delta_start[t] + delta_stop[t] <= 1)

            # 貯湯槽更新
            model.addConstr(
                H_tank[t] == H_tank[t-1] + H_prod[t] - p.heat_dmd[t, h],
                name=f"hp_tank_{t}"
            )

            # 貯湯槽容量制約
            model.addConstr(H_tank[t] >= 0.2 * p.CW * p.VLT * 70)
            model.addConstr(H_tank[t] <= p.CW * p.VLT * 70)

        # ヒートポンプ終端条件
        model.addConstr(H_tank[T] >= 0.5 * p.CW * p.VLT * 70, name="hp_tank_final")

        # ========== 目的関数 ==========
        # 電力コスト: Σ(買電単価 × 買電量 - 売電単価 × 売電量)
        electricity_cost = gp.quicksum(
            p.buy_price[t] * P_buy[t] - p.SELL_PRICE * P_sell[t]
            for t in range(1, T+1)
        )

        model.setObjective(electricity_cost, GRB.MINIMIZE)

        # 変数参照を保存
        self._model = model
        self._P_buy = P_buy
        self._P_sell = P_sell
        self._P_ch = P_ch
        self._P_dch = P_dch
        self._E_bat = E_bat
        self._P_hp = P_hp
        self._H_prod = H_prod
        self._H_tank = H_tank
        self._P_H = P_H
        self._Q_H = Q_H
        self._delta_buy = delta_buy
        self._delta_sell = delta_sell
        self._delta_ch = delta_ch
        self._delta_dch = delta_dch
        self._delta_hp = delta_hp
        self._delta_start = delta_start
        self._delta_stop = delta_stop

        return model

    def _extract_binary_values(self, model: gp.Model) -> Dict:
        """バイナリ変数の値を抽出"""
        T = self.T
        return {
            'delta_buy': {t: round(self._delta_buy[t].X) for t in range(1, T+1)},
            'delta_sell': {t: round(self._delta_sell[t].X) for t in range(1, T+1)},
            'delta_ch': {t: round(self._delta_ch[t].X) for t in range(1, T+1)},
            'delta_dch': {t: round(self._delta_dch[t].X) for t in range(1, T+1)},
            'delta_hp': {t: round(self._delta_hp[t].X) for t in range(T+1)},
            'delta_start': {t: round(self._delta_start[t].X) for t in range(T+1)},
            'delta_stop': {t: round(self._delta_stop[t].X) for t in range(T+1)},
        }

    def _extract_dual_variables(self, model: gp.Model) -> Tuple[Dict[int, float], Dict[int, float]]:
        """境界制約の双対変数を抽出"""
        mu_P = {}
        mu_Q = {}

        for t in range(1, self.T + 1):
            try:
                mu_P[t] = self._boundary_P_constrs[t].Pi
            except:
                mu_P[t] = 0.0

            try:
                mu_Q[t] = self._boundary_Q_constrs[t].Pi
            except:
                mu_Q[t] = 0.0

        return mu_P, mu_Q

    def _extract_solution(self, model: gp.Model) -> Dict:
        """解を抽出"""
        T = self.T
        return {
            'P_buy': {t: self._P_buy[t].X for t in range(1, T+1)},
            'P_sell': {t: self._P_sell[t].X for t in range(1, T+1)},
            'P_ch': {t: self._P_ch[t].X for t in range(1, T+1)},
            'P_dch': {t: self._P_dch[t].X for t in range(1, T+1)},
            'E_bat': {t: self._E_bat[t].X for t in range(1, T+1)},
            'P_hp': {t: self._P_hp[t].X for t in range(1, T+1)},
            'H_prod': {t: self._H_prod[t].X for t in range(1, T+1)},
            'H_tank': {t: self._H_tank[t].X for t in range(T+1)},
            'P_H': {t: self._P_H[t].X for t in range(1, T+1)},
            'Q_H': {t: self._Q_H[t].X for t in range(1, T+1)},
        }

    def _solve_feasibility_subproblem(self, P_DN_hat: Dict[int, float], 
                                       Q_DN_hat: Dict[int, float],
                                       iteration: int) -> SubproblemResult:
        """
        実行可能性サブ問題（ステップ2b）
        
        緩和変数α_iを導入して境界制約を不等式に変換
        """
        model = gp.Model(f"FeasibilitySubproblem_H{self.h}")
        model.setParam('OutputFlag', 0)
        model.setParam('Method', 2)
        model.setParam('QCPDual', 1)

        p = self.params
        T = self.T
        h = self.h

        # ========== 決定変数 ==========
        # 緩和変数（各時刻、P/Qの正負で4つ）
        alpha_P_pos = model.addVars(range(1, T+1), lb=0, name="alpha_P_pos")  # α_1
        alpha_P_neg = model.addVars(range(1, T+1), lb=0, name="alpha_P_neg")  # α_2
        alpha_Q_pos = model.addVars(range(1, T+1), lb=0, name="alpha_Q_pos")  # α_3
        alpha_Q_neg = model.addVars(range(1, T+1), lb=0, name="alpha_Q_neg")  # α_4

        # 需要家変数（バイナリは連続緩和）
        P_buy = model.addVars(range(1, T+1), lb=0, ub=p.BUY_MAX, name="P_buy")
        P_sell = model.addVars(range(1, T+1), lb=0, ub=p.SELL_MAX, name="P_sell")
        delta_buy = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_buy")
        delta_sell = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_sell")

        P_ch = model.addVars(range(1, T+1), lb=0, ub=p.N_B_PCS, name="P_charge")
        P_dch = model.addVars(range(1, T+1), lb=0, ub=p.N_B_PCS, name="P_discharge")
        E_bat = model.addVars(range(1, T+1), lb=0.2*p.N_B, ub=p.N_B, name="E_battery")
        delta_ch = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_charge")
        delta_dch = model.addVars(range(1, T+1), lb=0, ub=1, name="delta_discharge")

        P_hp = model.addVars(range(1, T+1), lb=0, ub=10.0, name="P_hp")
        H_prod = model.addVars(range(1, T+1), lb=0, name="H_produce")
        H_tank = model.addVars(range(T+1), lb=0, name="H_tank")
        delta_hp = model.addVars(range(T+1), lb=0, ub=1, name="delta_hp")
        delta_start = model.addVars(range(T+1), lb=0, ub=1, name="delta_start")
        delta_stop = model.addVars(range(T+1), lb=0, ub=1, name="delta_stop")

        P_H = model.addVars(range(1, T+1), lb=-GRB.INFINITY, name="P_H")
        Q_H = model.addVars(range(1, T+1), lb=-GRB.INFINITY, name="Q_H")

        model.update()

        # ========== 制約条件 ==========
        # 境界不等式制約（双対変数取得用）
        boundary_P_pos_constrs = {}
        boundary_P_neg_constrs = {}
        boundary_Q_pos_constrs = {}
        boundary_Q_neg_constrs = {}

        for t in range(1, T+1):
            # 電力バランス
            model.addConstr(
                p.p_pv[t, h] + P_buy[t] + P_dch[t] ==
                p.p_dmd[t, h] + P_sell[t] + P_ch[t] + P_hp[t],
                name=f"power_balance_{t}"
            )

            # 買電売電制約
            model.addConstr(P_buy[t] <= delta_buy[t] * p.BUY_MAX)
            model.addConstr(P_sell[t] <= delta_sell[t] * p.SELL_MAX)
            model.addConstr(delta_buy[t] + delta_sell[t] <= 1)

            # P_H, Q_H の定義
            model.addConstr(P_H[t] == P_buy[t] - P_sell[t], name=f"P_H_def_{t}")
            model.addConstr(Q_H[t] == 0.1 * P_H[t], name=f"Q_H_def_{t}")

            # 境界不等式制約（緩和変数付き）
            # P_H - P_DN_hat - α_1 <= 0
            boundary_P_pos_constrs[t] = model.addConstr(
                P_H[t] - P_DN_hat[t] - alpha_P_pos[t] <= 0,
                name=f"boundary_P_pos_{t}"
            )
            # -P_H + P_DN_hat - α_2 <= 0
            boundary_P_neg_constrs[t] = model.addConstr(
                -P_H[t] + P_DN_hat[t] - alpha_P_neg[t] <= 0,
                name=f"boundary_P_neg_{t}"
            )
            # Q_H - Q_DN_hat - α_3 <= 0
            boundary_Q_pos_constrs[t] = model.addConstr(
                Q_H[t] - Q_DN_hat[t] - alpha_Q_pos[t] <= 0,
                name=f"boundary_Q_pos_{t}"
            )
            # -Q_H + Q_DN_hat - α_4 <= 0
            boundary_Q_neg_constrs[t] = model.addConstr(
                -Q_H[t] + Q_DN_hat[t] - alpha_Q_neg[t] <= 0,
                name=f"boundary_Q_neg_{t}"
            )

            # 蓄電池制約
            model.addConstr(P_ch[t] <= p.N_B_PCS * delta_ch[t])
            model.addConstr(P_dch[t] <= p.N_B_PCS * delta_dch[t])
            model.addConstr(delta_ch[t] + delta_dch[t] <= 1)

            if t == 1:
                prev_soc = 0.5 * p.N_B
            else:
                prev_soc = E_bat[t-1]

            model.addConstr(
                E_bat[t] == prev_soc +
                0.5 * p.ETA_B * P_ch[t] -
                0.5 * P_dch[t] / p.ETA_B,
                name=f"battery_soc_{t}"
            )

        model.addConstr(E_bat[T] == 0.5 * p.N_B, name="battery_final")

        # ヒートポンプ制約
        model.addConstr(delta_hp[0] == 0)
        model.addConstr(delta_start[0] == 0)
        model.addConstr(delta_stop[0] == 0)
        model.addConstr(H_tank[0] == 0.5 * p.CW * p.VLT * 70, name="hp_tank_init")

        for t in range(1, T+1):
            cop = p.cop[t]

            model.addConstr(
                P_hp[t] == p.XHP * delta_hp[t] +
                (H_prod[t] + p.R2 * p.CPH * delta_start[t]) / (p.CE * cop),
                name=f"hp_power_{t}"
            )

            model.addConstr(H_prod[t] >= 0.5 * p.R1 * p.CPH * delta_hp[t])
            model.addConstr(H_prod[t] <= 0.5 * p.CPH * delta_hp[t])

            model.addConstr(
                delta_hp[t] - delta_hp[t-1] == delta_start[t] - delta_stop[t],
                name=f"hp_state_{t}"
            )
            model.addConstr(delta_start[t] + delta_stop[t] <= 1)

            model.addConstr(
                H_tank[t] == H_tank[t-1] + H_prod[t] - p.heat_dmd[t, h],
                name=f"hp_tank_{t}"
            )

            model.addConstr(H_tank[t] >= 0.2 * p.CW * p.VLT * 70)
            model.addConstr(H_tank[t] <= p.CW * p.VLT * 70)

        model.addConstr(H_tank[T] >= 0.5 * p.CW * p.VLT * 70, name="hp_tank_final")

        # ========== 目的関数 ==========
        # Σα_i の最小化
        objective = gp.quicksum(
            alpha_P_pos[t] + alpha_P_neg[t] + alpha_Q_pos[t] + alpha_Q_neg[t]
            for t in range(1, T+1)
        )
        model.setObjective(objective, GRB.MINIMIZE)

        # 最適化
        model.optimize()

        if model.Status != GRB.OPTIMAL:
            logger.error(f"  需要家{self.h}: 実行可能性サブ問題も解けませんでした")
            return SubproblemResult(is_feasible=False)

        # 双対変数を取得
        lambda_P = {}
        lambda_Q = {}

        for t in range(1, T+1):
            # λ_P = λ_1 - λ_2
            lambda_1 = boundary_P_pos_constrs[t].Pi if boundary_P_pos_constrs[t].Pi else 0.0
            lambda_2 = boundary_P_neg_constrs[t].Pi if boundary_P_neg_constrs[t].Pi else 0.0
            lambda_P[t] = lambda_1 - lambda_2

            # λ_Q = λ_3 - λ_4
            lambda_3 = boundary_Q_pos_constrs[t].Pi if boundary_Q_pos_constrs[t].Pi else 0.0
            lambda_4 = boundary_Q_neg_constrs[t].Pi if boundary_Q_neg_constrs[t].Pi else 0.0
            lambda_Q[t] = lambda_3 - lambda_4

        # L_*^{h,q} = λ^T * x_H
        L_star = 0.0
        for t in range(1, T+1):
            L_star += lambda_P[t] * P_H[t].X
            L_star += lambda_Q[t] * Q_H[t].X

        feasibility_cut = FeasibilityCut(
            L_star=L_star,
            lambda_P=lambda_P,
            lambda_Q=lambda_Q,
            household_idx=self.h,
            iteration=iteration
        )

        logger.debug(f"  需要家{self.h}: 実行可能カット生成 (緩和量={model.ObjVal:.4f})")

        return SubproblemResult(
            is_feasible=False,
            feasibility_cut=feasibility_cut
        )


class MasterProblem:
    """
    マスター問題（アグリゲータ）
    
    配電網制約を考慮しながら境界変数 P_DN, Q_DN を決定
    カット平面制約を追加しながら反復的に解く
    """

    def __init__(self, params: SystemParameters, dr_case: DRCase):
        self.params = params
        self.dr_case = dr_case
        self.T = params.T
        self.H = params.H

        # カット平面のリスト
        self.optimality_cuts: List[OptimalityCut] = []
        self.feasibility_cuts: List[FeasibilityCut] = []

        # モデル構築
        self._build_model()

    def _build_model(self):
        """マスター問題モデルを構築"""
        self.model = gp.Model("MasterProblem")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Method', 2)
        self.model.setParam('QCPDual', 1)

        p = self.params
        T = self.T
        H = self.H

        # ========== 決定変数 ==========
        # 境界変数
        self.P_DN = self.model.addVars(range(1, T+1), range(H),
                                        lb=p.P_DN_MIN, ub=p.P_DN_MAX, name="P_DN")
        self.Q_DN = self.model.addVars(range(1, T+1), range(H),
                                        lb=p.Q_DN_MIN, ub=p.Q_DN_MAX, name="Q_DN")

        # 電圧関連
        self.V = self.model.addVars(range(1, T+1), range(H), lb=0, name="Voltage")
        self.V_dd = self.model.addVars(range(1, T+1), range(H), lb=-GRB.INFINITY, name="V_dd")
        self.V_UN = self.model.addVars(range(1, T+1), range(H), lb=0, name="V_UN")
        self.V_LN = self.model.addVars(range(1, T+1), range(H), lb=0, name="V_LN")
        self.V_ULV = self.model.addVars(range(1, T+1), range(H), lb=0, name="V_ULV")
        self.V_LLV = self.model.addVars(range(1, T+1), range(H), lb=0, name="V_LLV")

        # 送電線容量関連
        self.S_DN = self.model.addVars(range(1, T+1), range(H), lb=0, name="S_DN")
        self.S_UN = self.model.addVars(range(1, T+1), range(H), lb=0, name="S_UN")
        self.S_ULV = self.model.addVars(range(1, T+1), range(H), lb=0, name="S_ULV")

        # 補助変数 LBD_h（各需要家の下界）
        self.LBD = self.model.addVars(range(H), lb=-1e10, name="LBD")
        self.feas_slacks = [] # 実行可能性カット用のスラック変数リスト

        self.model.update()

        # ========== 制約条件 ==========
        # Q_DN = 0.1 * P_DN の関係
        for t in range(1, T+1):
            for h in range(H):
                self.model.addConstr(
                    self.Q_DN[t, h] == 0.1 * self.P_DN[t, h],
                    name=f"Q_P_relation_{t}_{h}"
                )

        # 電圧制約
        for t in range(1, T+1):
            for h in range(H):
                # 累積潮流
                P_cumulative = gp.quicksum(self.P_DN[t, j] for j in range(h, H))
                Q_cumulative = gp.quicksum(self.Q_DN[t, j] for j in range(h, H))

                # 電圧降下推定
                self.model.addConstr(
                    self.V_dd[t, h] ==
                    p.linear_ap[h] * P_cumulative +
                    p.linear_aq[h] * Q_cumulative +
                    p.linear_b[h],
                    name=f"voltage_drop_{t}_{h}"
                )

                # 電圧計算
                if h == 0:
                    self.model.addConstr(
                        self.V[t, h] == p.V_BASE - self.V_dd[t, h],
                        name=f"voltage_{t}_{h}"
                    )
                else:
                    self.model.addConstr(
                        self.V[t, h] == self.V[t, h-1] - self.V_dd[t, h],
                        name=f"voltage_{t}_{h}"
                    )

                # 電圧ペナルティ制約
                self.model.addConstr(
                    p.V_UL - self.V[t, h] == self.V_UN[t, h] - self.V_ULV[t, h],
                    name=f"voltage_upper_{t}_{h}"
                )
                self.model.addConstr(
                    self.V[t, h] - p.V_LL == self.V_LN[t, h] - self.V_LLV[t, h],
                    name=f"voltage_lower_{t}_{h}"
                )

        # 送電線容量制約
        for t in range(1, T+1):
            for h in range(H):
                P_cumulative = gp.quicksum(self.P_DN[t, j] for j in range(h, H))
                Q_cumulative = gp.quicksum(self.Q_DN[t, j] for j in range(h, H))

                self.model.addQConstr(
                    self.S_DN[t, h] * self.S_DN[t, h] ==
                    P_cumulative * P_cumulative + Q_cumulative * Q_cumulative,
                    name=f"line_power_{t}_{h}"
                )
                self.model.addConstr(
                    p.S_LINE - self.S_DN[t, h] == self.S_UN[t, h] - self.S_ULV[t, h],
                    name=f"line_capacity_{t}_{h}"
                )

        # 目的関数は後で設定（カット平面追加後）
        self._update_objective()

    def _update_objective(self):
        """目的関数の設定"""
        p = self.params
        T = self.T
        H = self.H

        # ペナルティ項
        penalty = gp.quicksum(
            self.V_ULV[t, h] + self.V_LLV[t, h] + self.S_ULV[t, h]
            for t in range(1, T+1) for h in range(H)
        )
        
        # 実行可能カットのスラック変数へのペナルティ（非常に大きな値: Big-M）
        # これにより、可能な限りスラックを0にしようとするが、どうしても無理な場合は許容する
        BIG_M = 1e6
        cut_penalty = gp.quicksum(slack for slack in self.feas_slacks) * BIG_M

        # DR期間の純買電電力
        if self.dr_case != DRCase.NO_DR and p.DR_END <= T:
            dr_net_power = gp.quicksum(
                self.P_DN[t, h]
                for t in range(p.DR_START, p.DR_END+1)
                for h in range(H)
            )
        else:
            dr_net_power = 0

        # ベースとなる目的関数
        base_obj = gp.quicksum(self.LBD[h] for h in range(H)) + penalty + cut_penalty

        if self.dr_case == DRCase.NO_DR:
            objective = base_obj
        elif self.dr_case == DRCase.DOWN_DR:
            objective = base_obj + dr_net_power
        else:  # UP_DR
            objective = base_obj - dr_net_power

        self.model.setObjective(objective, GRB.MINIMIZE)

    def add_optimality_cut(self, cut: OptimalityCut):
        """最適カット平面を追加"""
        h = cut.household_idx
        T = self.T

        cut_expr = cut.L_star
        for t in range(1, T+1):
            cut_expr -= cut.mu_P.get(t, 0.0) * self.P_DN[t, h]
            cut_expr -= cut.mu_Q.get(t, 0.0) * self.Q_DN[t, h]

        self.model.addConstr(
            self.LBD[h] >= cut_expr,
            name=f"optimality_cut_h{h}_iter{cut.iteration}"
        )

        self.optimality_cuts.append(cut)

    def add_feasibility_cut(self, cut: FeasibilityCut):
        """
        実行可能カット平面を追加
        スラック変数 s を導入し、制約違反を許容（ただし高コスト）する
        """
        h = cut.household_idx
        T = self.T

        # スラック変数（非負）
        slack_var = self.model.addVar(lb=0, name=f"s_feas_cut_h{h}_iter{cut.iteration}")
        self.feas_slacks.append(slack_var)

        # L_*^{h,q} - λ^T * y_DN <= 0  -->  λ^T * y_DN >= L_*^{h,q}
        # スラック導入: λ^T * y_DN + slack >= L_*^{h,q}
        
        cut_expr = gp.LinExpr()
        for t in range(1, T+1):
            cut_expr += cut.lambda_P.get(t, 0.0) * self.P_DN[t, h]
            cut_expr += cut.lambda_Q.get(t, 0.0) * self.Q_DN[t, h]
        
        # スラック変数を加える
        cut_expr += slack_var

        self.model.addConstr(
            cut_expr >= cut.L_star,
            name=f"feasibility_cut_h{h}_iter{cut.iteration}"
        )

        self.feasibility_cuts.append(cut)
        
        # 目的関数を更新してスラック変数のペナルティを含める
        self._update_objective()

    def solve(self) -> Tuple[float, Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        """
        マスター問題を解く
        
        Returns:
            (目的関数値, P_DN_hat, Q_DN_hat)
        """
        self.model.update()
        self.model.optimize()

        # ステータスチェックの強化
        if self.model.Status == GRB.INFEASIBLE:
            logger.error("マスター問題がまだ実行不可能です。IISを計算します。")
            self.model.computeIIS()
            self.model.write("master_iis.ilp")
            return float('inf'), None, None  # Noneを返す

        if self.model.Status != GRB.OPTIMAL:
            logger.warning(f"マスター問題のステータス異常: {self.model.Status}")
            # 最適でなくても、解が取れるなら進む（TimeLimitなど）
            if self.model.SolCount == 0:
                 return float('inf'), None, None

        # 解を抽出
        P_DN_hat = {h: {t: self.P_DN[t, h].X for t in range(1, self.T+1)} for h in range(self.H)}
        Q_DN_hat = {h: {t: self.Q_DN[t, h].X for t in range(1, self.T+1)} for h in range(self.H)}

        return self.model.ObjVal, P_DN_hat, Q_DN_hat

    def get_solution_details(self) -> Dict:
        """マスター問題の詳細解を取得"""
        T = self.T
        H = self.H

        return {
            'P_DN': {h: {t: self.P_DN[t, h].X for t in range(1, T+1)} for h in range(H)},
            'Q_DN': {h: {t: self.Q_DN[t, h].X for t in range(1, T+1)} for h in range(H)},
            'V': {h: {t: self.V[t, h].X for t in range(1, T+1)} for h in range(H)},
            'V_ULV': {h: {t: self.V_ULV[t, h].X for t in range(1, T+1)} for h in range(H)},
            'V_LLV': {h: {t: self.V_LLV[t, h].X for t in range(1, T+1)} for h in range(H)},
            'S_DN': {h: {t: self.S_DN[t, h].X for t in range(1, T+1)} for h in range(H)},
            'S_ULV': {h: {t: self.S_ULV[t, h].X for t in range(1, T+1)} for h in range(H)},
            'LBD': {h: self.LBD[h].X for h in range(H)},
        }


class BendersSolver:
    """
    GBDアルゴリズム全体の制御
    """

    def __init__(self, dr_case: DRCase = DRCase.DOWN_DR, max_iterations: int = 100,
                 tolerance: float = 1e-4):
        self.dr_case = dr_case
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # システムパラメータ
        self.params = SystemParameters()

        # 収束履歴
        self.convergence_history = {
            'iteration': [],
            'lower_bound': [],
            'upper_bound': [],
            'gap': [],
            'optimality_cuts': [],
            'feasibility_cuts': [],
        }

        logger.info(f"GBDソルバー初期化: {dr_case} - {dr_case.description}")

    def solve(self) -> Dict:
        """GBDアルゴリズムの実行"""
        logger.info("="*60)
        logger.info(f"GBD最適化開始: {self.dr_case}")
        logger.info("="*60)

        start_real_time = time.time()
        start_cpu_time = time.process_time()

        # 初期化
        p = self.params
        T = p.T
        H = p.H

        # ステップ1: 境界変数の初期化
        P_DN_hat = {h: {t: 0.0 for t in range(1, T+1)} for h in range(H)}
        Q_DN_hat = {h: {t: 0.0 for t in range(1, T+1)} for h in range(H)}

        LB = float('-inf')
        UB = float('inf')

        # マスター問題とサブ問題の初期化
        master = MasterProblem(self.params, self.dr_case)
        subproblems = [HouseholdSubproblem(self.params, h, self.dr_case) for h in range(H)]

        # 最良解の保存
        best_solution = None
        best_upper_bound = float('inf')

        # 反復
        for k in range(1, self.max_iterations + 1):
            logger.info(f"\n--- 反復 {k} ---")

            # ステップ2: 各需要家がサブ問題を解く
            all_feasible = True
            subproblem_objectives = []
            household_solutions = {}

            for h in range(H):
                result = subproblems[h].solve(P_DN_hat[h], Q_DN_hat[h], k)

                if result.is_feasible:
                    # ステップ2a: 最適カットを追加
                    if result.optimality_cut is not None:
                        master.add_optimality_cut(result.optimality_cut)
                    subproblem_objectives.append(result.objective_value)
                    household_solutions[h] = result.solution
                    logger.info(f"  需要家{h}: 実行可能 (コスト={result.objective_value:.2f})")
                else:
                    # ステップ2b: 実行可能カットを追加
                    all_feasible = False
                    if result.feasibility_cut is not None:
                        master.add_feasibility_cut(result.feasibility_cut)
                    logger.info(f"  需要家{h}: 実行不可能 → 実行可能カット追加")

            # 上界の更新（全需要家が実行可能な場合のみ）
            if all_feasible:
                # マスター問題の目的関数のうちペナルティ部分を計算
                master_detail = master.get_solution_details()
                penalty = sum(
                    master_detail['V_ULV'][h][t] + master_detail['V_LLV'][h][t] + master_detail['S_ULV'][h][t]
                    for t in range(1, T+1) for h in range(H)
                )

                # DR項の計算
                if self.dr_case == DRCase.DOWN_DR:
                    dr_term = sum(
                        P_DN_hat[h][t]
                        for t in range(p.DR_START, p.DR_END+1)
                        for h in range(H)
                    )
                elif self.dr_case == DRCase.UP_DR:
                    dr_term = -sum(
                        P_DN_hat[h][t]
                        for t in range(p.DR_START, p.DR_END+1)
                        for h in range(H)
                    )
                else:
                    dr_term = 0

                current_UB = sum(subproblem_objectives) + dr_term + penalty

                if current_UB < UB:
                    UB = current_UB
                    logger.info(f"  上界更新: UB = {UB:.2f}")

                    # 最良解の保存
                    if current_UB < best_upper_bound:
                        best_upper_bound = current_UB
                        best_solution = {
                            'P_DN': copy.deepcopy(P_DN_hat),
                            'Q_DN': copy.deepcopy(Q_DN_hat),
                            'household_solutions': copy.deepcopy(household_solutions),
                            'master_detail': copy.deepcopy(master_detail),
                            'subproblem_objectives': subproblem_objectives.copy(),
                        }

            # ステップ3: マスター問題を解く
            LB_new, P_DN_hat, Q_DN_hat = master.solve()

            if LB_new > LB:
                LB = LB_new
                logger.info(f"  下界更新: LB = {LB:.2f}")

            # 収束履歴の記録
            gap = abs(UB - LB) / max(abs(UB), 1e-10) if UB != float('inf') else float('inf')
            self.convergence_history['iteration'].append(k)
            self.convergence_history['lower_bound'].append(LB)
            self.convergence_history['upper_bound'].append(UB)
            self.convergence_history['gap'].append(gap)
            self.convergence_history['optimality_cuts'].append(len(master.optimality_cuts))
            self.convergence_history['feasibility_cuts'].append(len(master.feasibility_cuts))

            logger.info(f"  LB={LB:.2f}, UB={UB:.2f}, Gap={gap:.4f}")
            logger.info(f"  最適カット数: {len(master.optimality_cuts)}, 実行可能カット数: {len(master.feasibility_cuts)}")

            # ステップ4: 収束判定
            if gap < self.tolerance:
                logger.info(f"\n収束達成 (Gap={gap:.6f} < {self.tolerance})")
                break

        # 計算時間
        real_time = time.time() - start_real_time
        cpu_time = time.process_time() - start_cpu_time

        # 結果の整理
        results = {
            'case': str(self.dr_case),
            'case_description': self.dr_case.description,
            'status': 'OPTIMAL' if gap < self.tolerance else 'MAX_ITER',
            'objective': UB if UB != float('inf') else None,
            'lower_bound': LB,
            'upper_bound': UB,
            'gap': gap,
            'iterations': k,
            'real_time': real_time,
            'cpu_time': cpu_time,
            'optimality_cuts': len(master.optimality_cuts),
            'feasibility_cuts': len(master.feasibility_cuts),
            'convergence_history': self.convergence_history,
        }

        if best_solution is not None:
            results['detail'] = self._extract_final_solution(best_solution)
            results['dr_period_net_power'] = self._calculate_dr_net_power(best_solution)

        logger.info("="*60)
        logger.info("GBD最適化完了")
        logger.info(f"目的関数値: {results['objective']}")
        logger.info(f"反復回数: {k}")
        logger.info(f"Real Time: {real_time:.2f}秒, CPU Time: {cpu_time:.2f}秒")
        logger.info("="*60)

        return results

    def _extract_final_solution(self, best_solution: Dict) -> Dict:
        """最終解の詳細を抽出"""
        p = self.params
        T = p.T
        H = p.H

        solution = {
            'P_AG_buy': {},
            'P_AG_sell': {},
            'voltage': {},
            'battery_soc': {},
            'hp_operation': {},
            'household_cost': {},
            'P_H': {},
            'Q_H': {},
            'P_DN': {},
            'Q_DN': {},
            'P_buy': {},
            'P_sell': {},
            'p_pv': {},
            'p_dmd': {},
            'P_ch': {},
            'P_dch': {},
            'P_hp': {},
            'H_tank': {},
            'H_prod': {},
        }

        # アグリゲータレベル
        for t in range(1, T+1):
            ag_buy = sum(max(0, best_solution['P_DN'][h][t]) for h in range(H))
            ag_sell = sum(max(0, -best_solution['P_DN'][h][t]) for h in range(H))
            solution['P_AG_buy'][t] = ag_buy
            solution['P_AG_sell'][t] = ag_sell

        # 需要家レベル
        for h in range(H):
            hs = best_solution['household_solutions'].get(h, {})
            md = best_solution['master_detail']

            solution['voltage'][h] = md['V'][h]
            solution['P_DN'][h] = md['P_DN'][h]
            solution['Q_DN'][h] = md['Q_DN'][h]

            solution['battery_soc'][h] = hs.get('E_bat', {t: 0 for t in range(1, T+1)})
            solution['P_H'][h] = hs.get('P_H', {t: 0 for t in range(1, T+1)})
            solution['Q_H'][h] = hs.get('Q_H', {t: 0 for t in range(1, T+1)})
            solution['P_buy'][h] = hs.get('P_buy', {t: 0 for t in range(1, T+1)})
            solution['P_sell'][h] = hs.get('P_sell', {t: 0 for t in range(1, T+1)})
            solution['P_ch'][h] = hs.get('P_ch', {t: 0 for t in range(1, T+1)})
            solution['P_dch'][h] = hs.get('P_dch', {t: 0 for t in range(1, T+1)})
            solution['P_hp'][h] = hs.get('P_hp', {t: 0 for t in range(1, T+1)})
            solution['H_tank'][h] = hs.get('H_tank', {t: 0 for t in range(T+1)})
            solution['H_prod'][h] = hs.get('H_prod', {t: 0 for t in range(1, T+1)})

            solution['p_pv'][h] = {t: p.p_pv[t, h] for t in range(1, T+1)}
            solution['p_dmd'][h] = {t: p.p_dmd[t, h] for t in range(1, T+1)}

            # HP運転状態は推定
            solution['hp_operation'][h] = {t: 1 if solution['P_hp'][h].get(t, 0) > 0.01 else 0 
                                           for t in range(1, T+1)}

            # 需要家コスト
            cost = sum(
                p.buy_price[t] * solution['P_buy'][h].get(t, 0) -
                p.SELL_PRICE * solution['P_sell'][h].get(t, 0)
                for t in range(1, T+1)
            )
            solution['household_cost'][h] = cost

        return solution

    def _calculate_dr_net_power(self, best_solution: Dict) -> float:
        """DR期間の純買電電力を計算"""
        p = self.params

        net_power = sum(
            best_solution['P_DN'][h][t]
            for t in range(p.DR_START, p.DR_END+1)
            for h in range(p.H)
        )
        return net_power

    def export_results(self, results: Dict, filename_prefix: str = "gbd"):
        """結果のエクスポート"""
        case_name = str(self.dr_case)
        output_dir = f"../data/output/gbd/{case_name}"
        os.makedirs(output_dir, exist_ok=True)

        # サマリーCSV
        summary_data = {
            'Metric': ['Case', 'Description', 'Status', 'Objective Value', 'Lower Bound',
                      'Upper Bound', 'Gap', 'Iterations', 'Real Time', 'CPU Time',
                      'Optimality Cuts', 'Feasibility Cuts'],
            'Value': [
                results['case'],
                results['case_description'],
                results['status'],
                results.get('objective', 'N/A'),
                results['lower_bound'],
                results['upper_bound'],
                results['gap'],
                results['iterations'],
                results['real_time'],
                results['cpu_time'],
                results['optimality_cuts'],
                results['feasibility_cuts'],
            ]
        }

        if 'detail' in results:
            summary_data['Metric'].append('Total Household Cost')
            summary_data['Value'].append(sum(results['detail']['household_cost'].values()))
            summary_data['Metric'].append('DR Period Net Power')
            summary_data['Value'].append(results.get('dr_period_net_power', 'N/A'))

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/{filename_prefix}_summary.csv",
                         index=False, encoding='utf-8')

        # 収束履歴
        conv_df = pd.DataFrame(results['convergence_history'])
        conv_df.to_csv(f"{output_dir}/{filename_prefix}_convergence.csv",
                       index=False, encoding='utf-8')

        # 時系列データ
        if 'detail' in results:
            detail = results['detail']
            T = self.params.T
            H = self.params.H

            time_series_data = []
            for t in range(1, T+1):
                row = {
                    'Time': t,
                    'AG_Buy': detail['P_AG_buy'].get(t, 0),
                    'AG_Sell': detail['P_AG_sell'].get(t, 0),
                }
                for h in range(H):
                    row[f'Voltage_H{h}'] = detail['voltage'][h].get(t, 0)
                    row[f'Battery_H{h}'] = detail['battery_soc'][h].get(t, 0)
                    row[f'P_H_h{h}'] = detail['P_H'][h].get(t, 0)
                    row[f'P_DN_h{h}'] = detail['P_DN'][h].get(t, 0)
                time_series_data.append(row)

            time_df = pd.DataFrame(time_series_data)
            time_df.to_csv(f"{output_dir}/{filename_prefix}_timeseries.csv",
                           index=False, encoding='utf-8')

            # 需要家ごとの詳細
            for h in range(H):
                rows = []
                for t in range(1, T+1):
                    rows.append({
                        'Time': t,
                        'PV': detail['p_pv'][h].get(t, 0),
                        'P_buy': detail['P_buy'][h].get(t, 0),
                        'P_dch': detail['P_dch'][h].get(t, 0),
                        'Demand': detail['p_dmd'][h].get(t, 0),
                        'P_sell': detail['P_sell'][h].get(t, 0),
                        'P_ch': detail['P_ch'][h].get(t, 0),
                        'P_hp': detail['P_hp'][h].get(t, 0),
                        'Battery_SOC': detail['battery_soc'][h].get(t, 0),
                        'P_H': detail['P_H'][h].get(t, 0),
                        'P_DN': detail['P_DN'][h].get(t, 0),
                        'V_DN': detail['voltage'][h].get(t, 0),
                    })
                df_h = pd.DataFrame(rows)
                df_h.to_csv(f"{output_dir}/{filename_prefix}_household_{h}.csv",
                            index=False, encoding='utf-8')

        logger.info(f"結果を{output_dir}に保存しました")

    def plot_convergence(self, results: Dict):
        """収束履歴のプロット"""
        case_name = str(self.dr_case)
        output_dir = f"../data/output/gbd/{case_name}/plots"
        os.makedirs(output_dir, exist_ok=True)

        history = results['convergence_history']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 上界・下界
        ax = axes[0, 0]
        ax.plot(history['iteration'], history['lower_bound'], 'b-o', label='Lower Bound')
        ax.plot(history['iteration'], history['upper_bound'], 'r-o', label='Upper Bound')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Convergence: Bounds')
        ax.legend()
        ax.grid(True)

        # 2. ギャップ
        ax = axes[0, 1]
        ax.semilogy(history['iteration'], history['gap'], 'g-o')
        ax.axhline(self.tolerance, color='r', linestyle='--', label=f'Tolerance ({self.tolerance})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gap')
        ax.set_title('Convergence: Gap')
        ax.legend()
        ax.grid(True)

        # 3. カット数
        ax = axes[1, 0]
        ax.plot(history['iteration'], history['optimality_cuts'], 'b-o', label='Optimality Cuts')
        ax.plot(history['iteration'], history['feasibility_cuts'], 'r-o', label='Feasibility Cuts')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Cuts')
        ax.set_title('Cut Planes Added')
        ax.legend()
        ax.grid(True)

        # 4. 総カット数
        ax = axes[1, 1]
        total_cuts = [o + f for o, f in zip(history['optimality_cuts'], history['feasibility_cuts'])]
        ax.plot(history['iteration'], total_cuts, 'm-o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Cuts')
        ax.set_title('Total Cut Planes')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/convergence.png", dpi=100)
        plt.close()

        logger.info(f"収束プロットを{output_dir}/convergence.pngに保存しました")

    def plot_results(self, results: Dict):
        """結果の可視化"""
        if 'detail' not in results:
            return

        case_name = str(self.dr_case)
        output_dir = f"../data/output/gbd/{case_name}/plots"
        os.makedirs(output_dir, exist_ok=True)

        detail = results['detail']
        p = self.params
        T = p.T
        H = p.H
        times = list(range(1, T + 1))

        for h in range(H):
            fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

            # 1. Power Balance
            ax = axes[0]
            p_pv = np.array([detail['p_pv'][h].get(t, 0) for t in times])
            p_buy = np.array([detail['P_buy'][h].get(t, 0) for t in times])
            p_dch = np.array([detail['P_dch'][h].get(t, 0) for t in times])
            p_dmd = np.array([detail['p_dmd'][h].get(t, 0) for t in times])
            p_sell = np.array([detail['P_sell'][h].get(t, 0) for t in times])
            p_ch = np.array([detail['P_ch'][h].get(t, 0) for t in times])
            p_hp = np.array([detail['P_hp'][h].get(t, 0) for t in times])

            ax.bar(times, p_pv, label='PV', color='gold', alpha=0.9)
            ax.bar(times, p_buy, bottom=p_pv, label='Buy', color='red', alpha=0.6)
            ax.bar(times, p_dch, bottom=p_pv + p_buy, label='Discharge', color='blue', alpha=0.6)

            ax.bar(times, -p_dmd, label='Demand', color='gray', alpha=0.5)
            ax.bar(times, -p_sell, bottom=-p_dmd, label='Sell', color='cyan', alpha=0.6)
            ax.bar(times, -p_ch, bottom=-p_dmd - p_sell, label='Charge', color='green', alpha=0.6)
            ax.bar(times, -p_hp, bottom=-p_dmd - p_sell - p_ch, label='HP', color='orange', alpha=0.8)

            if self.dr_case != DRCase.NO_DR:
                ax.axvspan(p.DR_START, p.DR_END, alpha=0.2, color='yellow', label='DR Period')

            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel('Power [kW]')
            ax.set_title(f'{case_name} - Household {h}: Power Balance')
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize='small')

            # 2. Battery SOC
            ax = axes[1]
            soc = [detail['battery_soc'][h].get(t, 0) for t in times]
            ax.plot(times, soc, label='SOC', color='tab:blue', linewidth=2)
            ax.axhline(p.N_B, color='gray', linestyle=':', label='Max')
            ax.axhline(0.2 * p.N_B, color='gray', linestyle=':', label='Min')
            ax.set_ylabel('Energy [kWh]')
            ax.set_title(f'{case_name} - Household {h}: Battery SOC')
            ax.legend()

            # 3. Voltage
            ax = axes[2]
            v_dn = [detail['voltage'][h].get(t, 0) for t in times]
            ax.plot(times, v_dn, label='V', color='magenta', linewidth=2)
            ax.axhline(p.V_UL, color='red', linestyle=':', label='Upper')
            ax.axhline(p.V_LL, color='blue', linestyle=':', label='Lower')
            ax.set_ylabel('Voltage [V]')
            ax.set_title(f'{case_name} - Household {h}: Voltage')
            ax.legend()

            # 4. Active Power
            ax = axes[3]
            p_h = [detail['P_H'][h].get(t, 0) for t in times]
            p_dn = [detail['P_DN'][h].get(t, 0) for t in times]
            ax.plot(times, p_h, label='P_H', color='black', linewidth=2)
            ax.plot(times, p_dn, label='P_DN', color='orange', linestyle='--', linewidth=2)
            ax.set_ylabel('Active Power [kW]')
            ax.set_title(f'{case_name} - Household {h}: Active Power')
            ax.legend()

            # 5. HP Operation
            ax = axes[4]
            h_tank = [detail['H_tank'][h].get(t, 0) for t in times]
            ax.plot(times, h_tank, label='Tank Level', color='orange', linewidth=2)
            ax.set_ylabel('Thermal [MJ]')
            ax.set_xlabel('Time Slot')
            ax.set_title(f'{case_name} - Household {h}: HP Tank')
            ax.legend()

            plt.tight_layout()
            plt.savefig(f"{output_dir}/household_{h}_analysis.png", dpi=100)
            plt.close()

        logger.info(f"グラフを{output_dir}に保存しました")


def create_comparison_summary(all_results: Dict[DRCase, Dict]):
    """3ケースの比較サマリーを作成"""
    output_dir = "../data/output/gbd"
    os.makedirs(output_dir, exist_ok=True)

    comparison_data = []
    for case, results in all_results.items():
        if results.get('objective') is not None:
            comparison_data.append({
                'Case': str(case),
                'Description': case.description,
                'Objective': results['objective'],
                'Iterations': results['iterations'],
                'Real_Time_sec': results['real_time'],
                'CPU_Time_sec': results['cpu_time'],
                'Gap': results['gap'],
                'Optimality_Cuts': results['optimality_cuts'],
                'Feasibility_Cuts': results['feasibility_cuts'],
                'DR_Period_Net_Power': results.get('dr_period_net_power', 'N/A'),
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"{output_dir}/comparison_summary.csv", index=False, encoding='utf-8')

    logger.info("="*60)
    logger.info("3ケース比較サマリー")
    logger.info("="*60)
    print(comparison_df.to_string(index=False))

    return comparison_df


def main():
    """メイン実行関数"""
    all_results = {}

    cases = [DRCase.NO_DR, DRCase.DOWN_DR, DRCase.UP_DR]

    for case in cases:
        print("\n" + "="*70)
        print(f"実行中: {case} - {case.description}")
        print("="*70 + "\n")

        try:
            solver = BendersSolver(dr_case=case, max_iterations=50, tolerance=1e-3)
            results = solver.solve()

            solver.export_results(results)
            solver.plot_convergence(results)
            solver.plot_results(results)

            all_results[case] = results

            print(f"\n{case} 完了:")
            print(f"  目的関数値: {results.get('objective', 'N/A')}")
            print(f"  反復回数: {results['iterations']}")
            print(f"  Real Time: {results['real_time']:.2f}秒")

        except Exception as e:
            logger.error(f"{case}の実行中にエラーが発生: {e}")
            import traceback
            traceback.print_exc()
            all_results[case] = {'objective': None, 'error': str(e)}

    if all_results:
        create_comparison_summary(all_results)

    return all_results


if __name__ == "__main__":
    results = main()