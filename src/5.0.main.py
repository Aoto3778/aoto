"""
ベンダーズ分解法による配電網DR最適化
"""

import logging
import time
import os
import gurobipy as gp
from matplotlib.sankey import DOWN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# matplotlibの日本語化設定
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP"]

# ロギングの設定
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ================================================================================
# パラメータ定義部
# ================================================================================

@dataclass
class Parameters:
    """システムパラメータを一元管理"""
    LINE: int = 4           # 電力系統の分岐数
    T: int = 48            # タイムスロット数
    H: int = 3             # 需要家数
    F: int = 1             # 引込線数（現在は1、将来拡張予定）
    
    # 下げDR期間（30分粒度）
    DOWN_DR_START_HOUR: int = 26  # 13時
    DOWN_DR_END_HOUR: int = 32     # 16時
    
    # 効率
    ETA_PV_PCS: float = 0.95
    ETA_B_PCS: float = 0.95
    GRANULARITY_CONVERSION: float = 0.5  # 30分→時間変換
    
    # 容量制限
    N_B_PCS: float = 1.5   # 蓄電池kW容量
    N_B: float = 6.3       # 蓄電池kWh容量
    BUY_MAX: float = 5.0   # 買電上限
    SELL_MAX: float = 5.0  # 売電上限
    IN_MAX: float = 15.0   # 需要家への供給電力上限
    OUT_MAX: float = 15.0  # 需要家からの放出電力上限
    
    # 価格
    SELL_PRICE: float = 2.0  # 売電単価
    
    # 電圧制約
    V_UL: float = 107.0    # 電圧上限
    V_LL: float = 95.0     # 電圧下限
    V_BASE: float = 99.8   # 基準電圧
    S_LINE: float = 2020   # 送電線定格電力
    
    # HP給湯器
    r1: float = 0.05       # 最低熱製造率
    r2: float = 0.1        # 起動時ロス率
    cw: float = 0.0042     # 水の比熱
    ce: float = 3.6        # 変換係数
    vlt: float = 370       # 貯水槽容量
    cph: float = 16.2      # 加熱能力
    xhp: float = 0.013     # 補器消費電力
    
    # ベンダーズ分解パラメータ
    EPSILON: float = 0.01  # 収束判定閾値
    MAX_ITERATIONS: int = 50

# ================================================================================
# データ読み込み部
# ================================================================================

class DataLoader:
    """データ読み込みを管理"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.data = {}
        
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

# ================================================================================
# 主問題（Master Problem）
# ================================================================================

class MasterProblem:
    """ベンダーズ分解の主問題"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.model = gp.Model("Master_Problem")
        self.model.setParam('OutputFlag', 0)
        
        # 決定変数
        self.P_f = {}
        self.eta = None
        
        # カット情報
        self.optimality_cuts = []
        self.feasibility_cuts = []
        
        self._initialize()
    
    def _initialize(self):
        """変数の初期化"""
        F = self.params.F
        T = self.params.T
        
        # 引込線への配分電力 P^(f,t)
        self.P_f = self.model.addVars(
            F, range(1, T+1),
            name='P_f',
            lb=-gp.GRB.INFINITY,
            ub=self.params.BUY_MAX * self.params.H
        )
        
        # 目的関数の代理変数 η
        self.eta = self.model.addVar(
            name='eta',
            lb=0
        )
        
        # 目的関数: minimize η
        self.model.setObjective(self.eta, gp.GRB.MINIMIZE)
        self.model.update()
    
    def add_optimality_cut(self, cut_constant: float, cut_gradient: np.ndarray, P_f_current: np.ndarray):
        """最適性カット追加（式5-1）"""
        F = self.params.F
        T = self.params.T
        
        # η ≥ L*(P^(f,t))
        cut_expr = cut_constant
        for f in range(F):
            for t in range(1, T+1):
                cut_expr += cut_gradient[f, t-1] * (self.P_f[f, t] - P_f_current[f, t-1])

        self.model.addConstr(
            self.eta >= cut_expr,
            name=f'opt_cut_{len(self.optimality_cuts)}'
        )
        
        self.optimality_cuts.append({
            'constant': cut_constant,
            'gradient': cut_gradient,
            'point': P_f_current
        })
        
        self.model.update()
    
    def add_feasibility_cut(self, cut_gradient: np.ndarray, P_f_current: np.ndarray, violation: float):
        """実行可能性カット追加（式5-2）"""
        F = self.params.F
        T = self.params.T
        
        # L_*(P^(f,t)) ≤ 0
        cut_expr = violation
        for f in range(F):
            for t in range(1, T+1):
                cut_expr += cut_gradient[f, t-1] * (self.P_f[f, t] - P_f_current[f, t-1])
        
        self.model.addConstr(
            cut_expr <= 0,
            name=f'feas_cut_{len(self.feasibility_cuts)}'
        )
        
        self.feasibility_cuts.append({
            'gradient': cut_gradient,
            'point': P_f_current,
            'violation': violation
        })
        
        self.model.update()
    
    def solve(self) -> Tuple[np.ndarray, float]:
        """主問題を解く"""
        self.model.optimize()
        
        if self.model.status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"主問題が解けません: status={self.model.status}")
        
        # 解の取得
        F = self.params.F
        T = self.params.T
        
        P_f_solution = np.zeros((F, T))
        for f in range(F):
            for t in range(self.params.DOWN_DR_START_HOUR, self.params.DOWN_DR_END_HOUR):
                P_f_solution[f, t-1] = self.P_f[f, t].X
        
        return P_f_solution, self.eta.X
    
    def get_initial_solution(self) -> np.ndarray:
        """初期解設定（全て0）"""
        F = self.params.F
        T = self.params.T
        
        P_f_init = np.zeros((F, T))

        return P_f_init

# ================================================================================
# 部分問題（Subproblem）
# ================================================================================

class Subproblem:
    """最適性部分問題（式2）"""
    
    def __init__(self, params: Parameters, data: Dict, feeder_id: int = 0):
        self.params = params
        self.data = data
        self.feeder_id = feeder_id
        self.model = None
        self.variables = {}
        self.constraints = {}
        self.dual_values = {}
    
    def build(self, P_f_fixed: np.ndarray):
        """固定されたP_fに対する部分問題を構築"""
        model = gp.Model(f"Subproblem_f{self.feeder_id}")
        model.setParam('OutputFlag', 0)
        
        T = self.params.T
        H = self.params.H
        
        # ========== 変数定義 ==========
        # アグリゲータレベル
        P_SG = model.addVars(range(1, T+1), name='P_SG', lb=0)
        Delta_sg = model.addVars(range(T+1), vtype=gp.GRB.BINARY, name='Delta_sg')
        Delta_pg = model.addVars(range(T+1), vtype=gp.GRB.BINARY, name='Delta_pg')
        
        # 需要家レベル
        P_in = model.addVars(range(1, T+1), range(H), name='P_in', lb=-gp.GRB.INFINITY)
        P_out = model.addVars(range(1, T+1), range(H), name='P_out', lb=0)
        P_pg = model.addVars(range(1, T+1), range(H), name='P_pg', lb=0)
        P_sg = model.addVars(range(1, T+1), range(H), name='P_sg', lb=0)
        Delta_in = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_in')
        Delta_out = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_out')
        
        # 蓄電池
        P_Ch = model.addVars(range(T+1), range(H), name='P_Ch', lb=0)
        P_DCh = model.addVars(range(T+1), range(H), name='P_DCh', lb=0)
        E_B = model.addVars(range(T+1), range(H), name='E_B', lb=0)
        Delta_ch = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_ch')
        Delta_dch = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_dch')
        
        # HP給湯器
        P_HP = model.addVars(range(1, T+1), range(H), name='P_HP', lb=0)
        H_ini = model.addVars(range(T+1), range(H), name='H_ini', lb=0)
        H_prod = model.addVars(range(T+1), range(H), name='H_prod', lb=0)
        H_tank = model.addVars(range(T+1), range(H), name='H_tank', lb=0)
        Delta_prod = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_prod')
        Delta_S = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_S')
        Delta_F = model.addVars(range(T+1), range(H), vtype=gp.GRB.BINARY, name='Delta_F')
        
        # 電圧・電力
        V_d = model.addVars(range(1, T+1), range(H), name='V_d', lb=-gp.GRB.INFINITY)
        V_dd = model.addVars(range(T+1), range(H), name='V_dd', lb=-gp.GRB.INFINITY)
        ap = model.addVars(range(1, T+1), range(H), name='ap', lb=-gp.GRB.INFINITY)
        aq = model.addVars(range(1, T+1), range(H), name='aq', lb=-gp.GRB.INFINITY)
        V_ULV = model.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='上限違反値', lb=0.0, ub=gp.GRB.INFINITY)
        V_LLV = model.addVars(range(1,T+1), range(H), vtype=gp.GRB.CONTINUOUS, name='下限違反値', lb=0.0, ub=gp.GRB.INFINITY)
        
        model.update()
        
        # ========== 目的関数（式2） ==========
        obj = gp.quicksum(
            self.data['buy_price'][t] * P_pg[t, i] - self.params.SELL_PRICE * P_sg[t, i]
            for t in range(1, T+1)
            for i in range(H)
        )
        model.setObjective(obj, gp.GRB.MINIMIZE)
        
        # ========== 制約条件 ==========
        
        # 電力配分制約（式2-4）: Σ P^(f,t) = P_PG^(t,k)
        # P_PGは固定値P_f_fixedとして扱う
        for t in range(1, T+1):
            model.addConstr(
                gp.quicksum(P_pg[t, i] for i in range(H)) == P_f_fixed[t-1],
                name=f'power_allocation_{t}'
            )
        
        # 需給バランス制約
        for t in range(1, T+1):
            model.addConstr(P_SG[t] == gp.quicksum(P_sg[t, i] for i in range(H)))
            model.addConstr(
                P_SG[t] + gp.quicksum(P_in[t, i] for i in range(H)) == 
                P_f_fixed[t-1] + gp.quicksum(P_out[t, i] for i in range(H))
            )
            
            for i in range(H):
                model.addConstr(
                    self.data['p_pv'][t, i] + P_in[t, i] + P_DCh[t, i] ==
                    self.data['p_dmd'][t, i] + P_Ch[t, i] + P_out[t, i] + P_HP[t, i]
                )
                model.addConstr(P_in[t, i] == P_pg[t, i])
                model.addConstr(P_out[t, i] == P_sg[t, i])
        
        # 買電売電同時禁止
        for t in range(1, T+1):
            model.addConstr(Delta_pg[t] * self.params.BUY_MAX * H >= P_f_fixed[t-1])
            model.addConstr(P_SG[t] <= Delta_sg[t] * self.params.SELL_MAX * H)
            model.addConstr(Delta_pg[t] + Delta_sg[t] <= 1)
            
            for i in range(H):
                model.addConstr(P_pg[t, i] <= Delta_in[t, i] * self.params.IN_MAX)
                model.addConstr(P_sg[t, i] <= Delta_out[t, i] * self.params.OUT_MAX)
                model.addConstr(Delta_in[t, i] + Delta_out[t, i] <= 1)
        
        # HP制約（各需要家）
        for i in range(H):
            tank_half = 0.5 * self.params.cw * self.params.vlt * 70
            
            for t in range(1, T+1):
                cop_t = self.data['cop'][t]
                model.addConstr(
                    P_HP[t,i] == self.params.xhp * Delta_prod[t,i] + 
                    (H_prod[t,i] + H_ini[t,i]) / self.params.ce / cop_t
                )
                model.addConstr(H_prod[t,i] >= self.params.GRANULARITY_CONVERSION * self.params.r1 * self.params.cph * Delta_prod[t,i])
                model.addConstr(H_prod[t,i] <= self.params.GRANULARITY_CONVERSION * self.params.cph * Delta_prod[t,i])
                model.addConstr(H_ini[t,i] == self.params.r2 * self.params.cph * Delta_S[t,i])
                model.addConstr(Delta_prod[t,i] - Delta_prod[t-1,i] == Delta_S[t,i] - Delta_F[t,i])
                model.addConstr(Delta_S[t,i] + Delta_F[t,i] <= 1)
                model.addConstr(Delta_S[t,i] <= Delta_prod[t,i])
                model.addConstr(H_prod[t,i] >= self.params.cph * (Delta_prod[t,i] - Delta_S[t,i]))
                model.addConstr(H_tank[t,i] == H_tank[t-1,i] + H_prod[t,i] - self.data['H_dmd'][t,i])
                model.addConstr(H_tank[t,i] <= self.params.cw * self.params.vlt * 70)
                model.addConstr(H_tank[t,i] >= 0.2 * self.params.cw * self.params.vlt * 70)
            
            # 初期・終端条件
            model.addConstr(Delta_prod[0,i] == 0)
            model.addConstr(H_tank[0,i] == tank_half)
            model.addConstr(H_tank[T,i] == tank_half)
        
        # 蓄電池制約（各需要家）
        for i in range(H):
            for t in range(1, T+1):
                model.addConstr(P_Ch[t,i] <= self.params.N_B_PCS * Delta_ch[t,i])
                model.addConstr(P_DCh[t,i] <= self.params.N_B_PCS * Delta_dch[t,i])
                model.addConstr(Delta_ch[t,i] + Delta_dch[t,i] <= 1)
                model.addConstr(E_B[t,i] >= self.params.N_B * 0.2)
                model.addConstr(E_B[t,i] <= self.params.N_B)
                model.addConstr(
                    E_B[t,i] == E_B[t-1,i] + 
                    self.params.GRANULARITY_CONVERSION * self.params.ETA_B_PCS * P_Ch[t,i] - 
                    self.params.GRANULARITY_CONVERSION * (1/self.params.ETA_B_PCS) * P_DCh[t,i]
                )
            
            # 初期・終端条件
            model.addConstr(E_B[0,i] == 0.5 * self.params.N_B)
            model.addConstr(E_B[T,i] == 0.5 * self.params.N_B)
            model.addConstr(Delta_ch[0,i] == 0)
            model.addConstr(Delta_dch[0,i] == 0)
        
        # 電圧制約（式2-1, 2-2）- 双対変数取得用に保存
        voltage_lower = []
        voltage_upper = []
        for t in range(1, T+1):
            for k in range(H):
                c1 = model.addConstr(
                    self.params.V_LL - V_d[t,k] <= 0,
                    name=f'V_lower_{t}_{k}'
                )
                c2 = model.addConstr(
                    V_d[t,k] - self.params.V_UL <= 0,
                    name=f'V_upper_{t}_{k}'
                )
                voltage_lower.append(c1)
                voltage_upper.append(c2)
        
        # 送電容量制約（式2-3）- 二次錐制約
        power_flow = []
        for t in range(1, T+1):
            for k in range(H):
                c3 = model.addQConstr(
                    ap[t,k]*ap[t,k] + aq[t,k]*aq[t,k] <= self.params.S_LINE * self.params.S_LINE,
                    name=f'power_flow_{t}_{k}'
                )
                power_flow.append(c3)
        
        # 電圧推定
        for t in range(1, T+1):
            for k in range(H):
                model.addConstr(
                    V_dd[t,k] == 
                    self.data['linear_ap'][k] * ap[t,k] + 
                    self.data['linear_aq'][k] * aq[t,k] + 
                    self.data['linear_b'][k]
                )
                
                if k == 0:
                    model.addConstr(V_d[t,k] == V_dd[t,k] + self.params.V_BASE)
                else:
                    model.addConstr(V_d[t,k] == V_dd[t,k] + V_d[t,k-1])
        
        # 有効・無効電力
        for t in range(1, T+1):
            for k in range(H):
                model.addConstr(ap[t,k] == gp.quicksum(P_in[t,i] for i in range(k, H)))
                model.addConstr(aq[t,k] == gp.quicksum(P_in[t,i] for i in range(k, H)) * 0.1)
        
        model.update()
        
        # 保存
        self.model = model
        self.constraints = {
            'voltage_lower': voltage_lower,
            'voltage_upper': voltage_upper,
            'power_flow': power_flow
        }
        self.variables = {
            'P_SG': P_SG, 'P_pg': P_pg, 'P_sg': P_sg,
            'V_d': V_d, 'ap': ap, 'aq': aq
        }
        
            
    def solve(self) -> Tuple[str, Optional[float], Dict]:
        """部分問題を解く"""
        self.model.optimize()
        
        # ステータスのログ出力（デバッグ用）
        logger.debug(f"    部分問題のステータスコード: {self.model.status}")
        
        # ステータスコードを明示的に確認
        if self.model.status == gp.GRB.OPTIMAL:  # status == 2
            # 最適解の場合：双対変数を取得
            try:
                self.dual_values = {
                    'lambda_1': [c.Pi for c in self.constraints['voltage_lower']],
                    'lambda_2': [c.Pi for c in self.constraints['voltage_upper']], 
                    'lambda_3': [c.QCPi for c in self.constraints['power_flow']]
                }
                return 'OPTIMAL', self.model.ObjVal, self.dual_values
            except AttributeError as e:
                # 双対変数が取得できない場合
                logger.warning(f"    双対変数取得エラー: {e}")
                logger.info(f"    モデルステータス: {self.model.status}")
                logger.info(f"    目的関数値: {self.model.ObjVal if hasattr(self.model, 'ObjVal') else 'N/A'}")
                
                # 制約の状態を確認
                self.model.write("debug_optimal_but_no_dual.lp")
                logger.info("    モデルをdebug_optimal_but_no_dual.lpに出力")
                
                # 双対変数なしで返す
                return 'OPTIMAL', self.model.ObjVal, {}
        
        elif self.model.status == gp.GRB.INFEASIBLE:  # status == 3
            logger.info("  部分問題が実行不可能です")
            
            # IIS計算（既存のコード）
            try:
                logger.info("  IISを計算中...")
                self.model.computeIIS()
                # ... 既存のIISコード
                self.dual_values = {'infeasible': True}
            except Exception as e:
                logger.info(f"  IIS計算エラー: {e}")
                self.dual_values = {'infeasible': True}
            
            return 'INFEASIBLE', None, self.dual_values
        
        elif self.model.status == gp.GRB.INF_OR_UNBD:  # status == 4
            # 既存のコード
            self.model.setParam('DualReductions', 0)
            self.model.setParam('InfUnbdInfo', 1)
            self.model.optimize()
            
            if self.model.status == gp.GRB.INFEASIBLE:
                return 'INFEASIBLE', None, {'infeasible': True}
            else:
                return f'OTHER_{self.model.status}', None, {}
        
        else:
            # その他のステータス
            logger.warning(f"    予期しないステータス: {self.model.status}")
            return f'OTHER_{self.model.status}', None, {}
# ================================================================================
# 実行可能性部分問題（Feasibility Subproblem）
# ================================================================================

class FeasibilitySubproblem:
    """実行可能性部分問題（式4）"""
    
    def __init__(self, params: Parameters, data: Dict, feeder_id: int = 0):
        self.params = params
        self.data = data
        self.feeder_id = feeder_id
        self.model = None
        self.dual_values = {}
    
    def build(self, P_f_fixed: np.ndarray):
        """実行可能性部分問題を構築"""
        model = gp.Model(f"Feasibility_f{self.feeder_id}")
        model.setParam('OutputFlag', 0)
        
        T = self.params.T
        H = self.params.H
        
        # ========== 変数定義（部分問題と同じ + スラック変数）==========
        # アグリゲータレベル
        P_SG = model.addVars(range(1, T+1), name='P_SG', lb=0)
        Delta_sg = model.addVars(range(T+1), name='Delta_sg', lb=0, ub=1)
        Delta_pg = model.addVars(range(T+1), name='Delta_pg', lb=0, ub=1)
        
        # 需要家レベル
        P_in = model.addVars(range(1, T+1), range(H), name='P_in', lb=-gp.GRB.INFINITY)
        P_out = model.addVars(range(1, T+1), range(H), name='P_out', lb=0)
        P_pg = model.addVars(range(1, T+1), range(H), name='P_pg', lb=0)
        P_sg = model.addVars(range(1, T+1), range(H), name='P_sg', lb=0)
        Delta_in = model.addVars(range(T+1), range(H), name='Delta_in', lb=0, ub=1)
        Delta_out = model.addVars(range(T+1), range(H), name='Delta_out', lb=0, ub=1)
        
        # 蓄電池
        P_Ch = model.addVars(range(T+1), range(H), name='P_Ch', lb=0)
        P_DCh = model.addVars(range(T+1), range(H), name='P_DCh', lb=0)
        E_B = model.addVars(range(T+1), range(H), name='E_B', lb=0)
        Delta_ch = model.addVars(range(T+1), range(H), name='Delta_ch', lb=0, ub=1)
        Delta_dch = model.addVars(range(T+1), range(H), name='Delta_dch', lb=0, ub=1)
        
        # HP給湯器
        P_HP = model.addVars(range(1, T+1), range(H), name='P_HP', lb=0)
        H_ini = model.addVars(range(T+1), range(H), name='H_ini', lb=0)
        H_prod = model.addVars(range(T+1), range(H), name='H_prod', lb=0)
        H_tank = model.addVars(range(T+1), range(H), name='H_tank', lb=0)
        Delta_prod = model.addVars(range(T+1), range(H), name='Delta_prod', lb=0, ub=1)
        Delta_S = model.addVars(range(T+1), range(H), name='Delta_S', lb=0, ub=1)
        Delta_F = model.addVars(range(T+1), range(H), name='Delta_F', lb=0, ub=1)
        
        # 電圧・電力
        V_d = model.addVars(range(1, T+1), range(H), name='V_d', lb=-gp.GRB.INFINITY)
        V_dd = model.addVars(range(T+1), range(H), name='V_dd', lb=-gp.GRB.INFINITY)
        ap = model.addVars(range(1, T+1), range(H), name='ap', lb=-gp.GRB.INFINITY)
        aq = model.addVars(range(1, T+1), range(H), name='aq', lb=-gp.GRB.INFINITY)
        
        # ========== スラック変数 ==========
        V_under = model.addVars(range(1, T+1), range(H), name='V_under', lb=0)
        V_over = model.addVars(range(1, T+1), range(H), name='V_over', lb=0)
        S_over = model.addVars(range(1, T+1), range(H), name='S_over', lb=0)
        
        model.update()
        
        # ========== 目的関数（式4）==========
        obj = gp.quicksum(
            V_under[t,k] + V_over[t,k] + S_over[t,k]
            for t in range(1, T+1)
            for k in range(H)
        )
        model.setObjective(obj, gp.GRB.MINIMIZE)
        
        # ========== 制約条件 ==========
        
        # 電力配分制約
        for t in range(1, T+1):
            model.addConstr(
                gp.quicksum(P_pg[t, i] for i in range(H)) == P_f_fixed[t-1],
                name=f'power_allocation_{t}'
            )
        
        # 需給バランス制約（部分問題と同じ）
        for t in range(1, T+1):
            model.addConstr(P_SG[t] == gp.quicksum(P_sg[t, i] for i in range(H)))
            model.addConstr(
                P_SG[t] + gp.quicksum(P_in[t, i] for i in range(H)) == 
                P_f_fixed[t-1] + gp.quicksum(P_out[t, i] for i in range(H))
            )
            
            for i in range(H):
                model.addConstr(
                    self.data['p_pv'][t, i] + P_in[t, i] + P_DCh[t, i] ==
                    self.data['p_dmd'][t, i] + P_Ch[t, i] + P_out[t, i] + P_HP[t, i]
                )
                model.addConstr(P_in[t, i] == P_pg[t, i])
                model.addConstr(P_out[t, i] == P_sg[t, i])
        
        # 買電売電同時禁止
        for t in range(1, T+1):
            model.addConstr(Delta_pg[t] * self.params.BUY_MAX * H >= P_f_fixed[t-1])
            model.addConstr(P_SG[t] <= Delta_sg[t] * self.params.SELL_MAX * H)
            model.addConstr(Delta_pg[t] + Delta_sg[t] <= 1)
            
            for i in range(H):
                model.addConstr(P_pg[t, i] <= Delta_in[t, i] * self.params.IN_MAX)
                model.addConstr(P_sg[t, i] <= Delta_out[t, i] * self.params.OUT_MAX)
                model.addConstr(Delta_in[t, i] + Delta_out[t, i] <= 1)
        
        # HP制約（部分問題と同じ）
        for i in range(H):
            tank_half = 0.5 * self.params.cw * self.params.vlt * 70
            
            for t in range(1, T+1):
                cop_t = self.data['cop'][t]
                model.addConstr(
                    P_HP[t,i] == self.params.xhp * Delta_prod[t,i] + 
                    (H_prod[t,i] + H_ini[t,i]) / self.params.ce / cop_t
                )
                model.addConstr(H_prod[t,i] >= self.params.GRANULARITY_CONVERSION * self.params.r1 * self.params.cph * Delta_prod[t,i])
                model.addConstr(H_prod[t,i] <= self.params.GRANULARITY_CONVERSION * self.params.cph * Delta_prod[t,i])
                model.addConstr(H_ini[t,i] == self.params.r2 * self.params.cph * Delta_S[t,i])
                model.addConstr(Delta_prod[t,i] - Delta_prod[t-1,i] == Delta_S[t,i] - Delta_F[t,i])
                model.addConstr(Delta_S[t,i] + Delta_F[t,i] <= 1)
                model.addConstr(Delta_S[t,i] <= Delta_prod[t,i])
                model.addConstr(H_prod[t,i] >= self.params.cph * (Delta_prod[t,i] - Delta_S[t,i]))
                model.addConstr(H_tank[t,i] == H_tank[t-1,i] + H_prod[t,i] - self.data['H_dmd'][t,i])
                model.addConstr(H_tank[t,i] <= self.params.cw * self.params.vlt * 70)
                model.addConstr(H_tank[t,i] >= 0.2 * self.params.cw * self.params.vlt * 70)
            
            model.addConstr(Delta_prod[0,i] == 0)
            model.addConstr(H_tank[0,i] == tank_half)
            model.addConstr(H_tank[T,i] == tank_half)
        
        # 蓄電池制約（部分問題と同じ）
        for i in range(H):
            for t in range(1, T+1):
                model.addConstr(P_Ch[t,i] <= self.params.N_B_PCS * Delta_ch[t,i])
                model.addConstr(P_DCh[t,i] <= self.params.N_B_PCS * Delta_dch[t,i])
                model.addConstr(Delta_ch[t,i] + Delta_dch[t,i] <= 1)
                model.addConstr(E_B[t,i] >= self.params.N_B * 0.2)
                model.addConstr(E_B[t,i] <= self.params.N_B)
                model.addConstr(
                    E_B[t,i] == E_B[t-1,i] + 
                    self.params.GRANULARITY_CONVERSION * self.params.ETA_B_PCS * P_Ch[t,i] - 
                    self.params.GRANULARITY_CONVERSION * (1/self.params.ETA_B_PCS) * P_DCh[t,i]
                )
            
            model.addConstr(E_B[0,i] == 0.5 * self.params.N_B)
            model.addConstr(E_B[T,i] == 0.5 * self.params.N_B)
            model.addConstr(Delta_ch[0,i] == 0)
            model.addConstr(Delta_dch[0,i] == 0)
        
        # ========== スラック変数を含む電圧・送電容量制約 ==========
        voltage_lower = []
        voltage_upper = []
        power_flow = []
        
        for t in range(1, T+1):
            for k in range(H):
                # 式(4-1): V_min - V^(b,t,k) ≤ V_under
                c1 = model.addConstr(
                    self.params.V_LL - V_d[t,k] <= V_under[t,k],
                    name=f'V_lower_slack_{t}_{k}'
                )
                # 式(4-2): V^(b,t,k) - V_max ≤ V_over
                c2 = model.addConstr(
                    V_d[t,k] - self.params.V_UL <= V_over[t,k],
                    name=f'V_upper_slack_{t}_{k}'
                )
                voltage_lower.append(c1)
                voltage_upper.append(c2)
                
                # 式(4-3): √(P²+Q²) - S_max ≤ S_over
                c3 = model.addConstr(
                    ap[t,k]*ap[t,k] + aq[t,k]*aq[t,k] <= (self.params.S_LINE + S_over[t,k])**2,
                    name=f'power_flow_slack_{t}_{k}'
                )
                power_flow.append(c3)
        
        # 電圧推定
        for t in range(1, T+1):
            for k in range(H):
                model.addConstr(
                    V_dd[t,k] == 
                    self.data['linear_ap'][k] * ap[t,k] + 
                    self.data['linear_aq'][k] * aq[t,k] + 
                    self.data['linear_b'][k]
                )
                
                if k == 0:
                    model.addConstr(V_d[t,k] == V_dd[t,k] + self.params.V_BASE)
                else:
                    model.addConstr(V_d[t,k] == V_dd[t,k] + V_d[t,k-1])
        
        # 有効・無効電力
        for t in range(1, T+1):
            for k in range(H):
                model.addConstr(ap[t,k] == gp.quicksum(P_in[t,i] for i in range(k, H)))
                model.addConstr(aq[t,k] == gp.quicksum(P_in[t,i] for i in range(k, H)) * 0.1)
        
        model.update()
        
        self.model = model
        self.constraints = {
            'voltage_lower': voltage_lower,
            'voltage_upper': voltage_upper,
            'power_flow': power_flow
        }
    
    def solve(self) -> Tuple[float, Dict]:
        """実行可能性部分問題を解く"""
        self.model.optimize()
        
        if self.model.status == gp.GRB.OPTIMAL:
            violation = self.model.ObjVal
            
            # 双対変数の取得
            try:
                self.dual_values = {
                    'lambda_1': [c.Pi for c in self.constraints['voltage_lower']],
                    'lambda_2': [c.Pi for c in self.constraints['voltage_upper']],
                    'lambda_3': [c.QCPi for c in self.constraints['power_flow']]
                }
            except:
                self.dual_values = {}
                
            return violation, self.dual_values
        else:
            raise RuntimeError(f"実行可能性部分問題が解けません: {self.model.status}")

# ================================================================================
# ベンダーズ分解ソルバー
# ================================================================================

class BendersSolver:
    """ベンダーズ分解アルゴリズムの実装"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.data_loader = DataLoader(params)
        self.data = self.data_loader.load_all_data()
        
        self.master = MasterProblem(params)
        self.subproblems = []
        for f in range(params.F):
            self.subproblems.append(Subproblem(params, self.data, f))
        
        self.iteration = 0
        self.UBD = float('inf')  # 上界
        self.LBD = float('-inf')  # 下界
        self.convergence_history = []
        
        self.optimal_P_f = None
        self.optimal_objective = None
    
    def solve(self) -> Dict:
        """ベンダーズ分解アルゴリズムを実行"""
        start_time = time.time()
        
        # Step 1: 初期解の設定
        P_f_current = self.master.get_initial_solution()
        logger.info(f"初期解: P_f平均 = {P_f_current.mean():.2f} kW, 最大 = {P_f_current.max():.2f} kW")

        
        while self.iteration < self.params.MAX_ITERATIONS:
            self.iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"反復 {self.iteration}")
            logger.info(f"{'='*60}")
            
            # Step 2: 部分問題を解く
            all_optimal = True
            sub_objectives = []
            
            for f in range(self.params.F):
                logger.info(f"\n引込線 {f} の部分問題を解いています...")
                
                # 部分問題の構築と求解
                self.subproblems[f].build(P_f_current[f, :])
                status, obj_val, duals = self.subproblems[f].solve()
                
                logger.info(f"  状態: {status}")
                if status == 'OPTIMAL':
                    logger.info(f"  目的関数値: {obj_val:.2f}")
                    sub_objectives.append(obj_val)
                else:
                    all_optimal = False
                    break
            
            if all_optimal:
                # 最適性カットの生成
                logger.info("\n最適性カットを生成中...")
                self._generate_optimality_cut(sub_objectives, P_f_current)
                
                # 上界の更新（式3）
                self._update_upper_bound(sub_objectives, P_f_current)
                
            else:
                # 実行可能性カットの生成
                logger.info("\n実行可能性カットを生成中...")
                self._generate_feasibility_cut(P_f_current)
            
            # Step 3: 主問題を解く
            logger.info("\n主問題を解いています...")
            P_f_new, eta = self.master.solve()
            self.LBD = eta
            logger.info(f"  新しいP_f: 平均 = {P_f_new.mean():.2f} kW, 最大 = {P_f_new.max():.2f} kW")
            
            logger.info(f"\n現在の境界:")
            logger.info(f"  上界(UBD): {self.UBD:.2f}")
            logger.info(f"  下界(LBD): {self.LBD:.2f}")
            
            # Step 4: 収束判定
            if self._check_convergence():
                logger.info(f"\n{'='*60}")
                logger.info(f"収束しました！（反復 {self.iteration} 回）")
                logger.info(f"最適目的関数値: {self.UBD:.2f}")
                logger.info(f"{'='*60}")
                self.optimal_P_f = P_f_new
                self.optimal_objective = self.UBD
                break
            else:
                gap = abs(self.UBD - self.LBD) / abs(self.LBD) if self.LBD != 0 else float('inf')
                logger.info(f"  ギャップ: {gap*100:.2f}%")
            
            # 履歴の記録
            self.convergence_history.append({
                'iteration': self.iteration,
                'UBD': self.UBD,
                'LBD': self.LBD,
                'gap': abs(self.UBD - self.LBD) / abs(self.LBD) if self.LBD != 0 else float('inf')
            })
            
            P_f_current = P_f_new
        
        solve_time = time.time() - start_time
        
        return {
            'status': 'OPTIMAL' if self._check_convergence() else 'MAX_ITERATIONS',
            'objective': self.optimal_objective,
            'P_f': self.optimal_P_f,
            'iterations': self.iteration,
            'solve_time': solve_time,
            'convergence_history': self.convergence_history
        }
    
    def _generate_optimality_cut(self, sub_objectives: List[float], P_f_current: np.ndarray):
        """最適性カット生成（式5-3）"""
        T = self.params.T
        F = self.params.F
        
        # カットの定数項
        cut_constant = sum(sub_objectives)
        
        # カットの勾配（簡略化: 単位勾配を使用）
        cut_gradient = np.ones((F, T))
        
        # 下げDR期間のコスト追加
        for f in range(F):
            for t in range(self.params.DOWN_DR_START_HOUR - 1, self.params.DOWN_DR_END_HOUR):
                cut_gradient[f, t] += 1.0
        
        self.master.add_optimality_cut(cut_constant, cut_gradient, P_f_current)
        logger.info(f"  最適性カット追加（定数項: {cut_constant:.2f}）")
        
    def _generate_feasibility_cut(self, P_f_current: np.ndarray):
        """実行可能性カット生成（改良版）"""
        F = self.params.F
        T = self.params.T
        
        # 需要パターンに基づいた勾配
        cut_gradient = np.zeros((F, T))
        
        for f in range(F):
            for t in range(1, T+1):
                # 需要と供給のバランスを考慮
                total_demand = sum(self.data['p_dmd'][t, i] for i in range(self.params.H))
                total_pv = sum(self.data['p_pv'][t, i] for i in range(self.params.H))
                net_demand = total_demand - total_pv
                
                # 現在のP_fと必要量の差に基づく勾配
                if P_f_current[f, t-1] < net_demand * 0.8:  # 不足している
                    cut_gradient[f, t-1] = 0.1  # 増やす方向
                elif P_f_current[f, t-1] > net_demand * 1.2:  # 過剰
                    cut_gradient[f, t-1] = -0.1  # 減らす方向
                else:
                    cut_gradient[f, t-1] = 0.01  # 微調整
        
        violation = 0.05  # 小さな違反量
        
        self.master.add_feasibility_cut(cut_gradient, P_f_current, violation)
        logger.info(f"  実行可能性カット追加（違反量: {violation:.2f}）")
        
    def _update_upper_bound(self, sub_objectives: List[float], P_f_current: np.ndarray):
        """上界更新（式3）"""
        Z_sub = sum(sub_objectives)
        
        # 下げDR期間の買電電力量
        dr_power = 0
        for f in range(self.params.F):
            for t in range(self.params.DOWN_DR_START_HOUR - 1, self.params.DOWN_DR_END_HOUR):
                dr_power += P_f_current[f, t]
        
        new_UBD = Z_sub + dr_power
        
        if new_UBD < self.UBD:
            self.UBD = new_UBD
            logger.info(f"  上界更新: {self.UBD:.2f}")
    
    def _check_convergence(self) -> bool:
        """収束判定"""
        if self.LBD <= 0:
            return False
        
        gap = abs(self.UBD - self.LBD) / abs(self.LBD)
        return gap < self.params.EPSILON
    
    def plot_convergence(self):
        """収束履歴をプロット"""
        if not self.convergence_history:
            return
        
        iterations = [h['iteration'] for h in self.convergence_history]
        ubds = [h['UBD'] for h in self.convergence_history]
        lbds = [h['LBD'] for h in self.convergence_history]
        gaps = [h['gap'] * 100 for h in self.convergence_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 上界・下界の推移
        ax1.plot(iterations, ubds, 'b-', label='上界(UBD)', linewidth=2)
        ax1.plot(iterations, lbds, 'r-', label='下界(LBD)', linewidth=2)
        ax1.set_xlabel('反復回数')
        ax1.set_ylabel('目的関数値')
        ax1.set_title('ベンダーズ分解法の収束過程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ギャップの推移
        ax2.plot(iterations, gaps, 'g-', linewidth=2)
        ax2.axhline(y=self.params.EPSILON * 100, color='r', linestyle='--', label=f'収束閾値({self.params.EPSILON*100}%)')
        ax2.set_xlabel('反復回数')
        ax2.set_ylabel('ギャップ(%)')
        ax2.set_title('相対ギャップの推移')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('benders_convergence.png', dpi=150)
        plt.show()

# ================================================================================
# メイン実行部
# ================================================================================

if __name__ == "__main__":
    params = Parameters()
    
    # ベンダーズ分解法の実行
    logger.info("ベンダーズ分解法による配電網DR最適化")
    logger.info(f"引込線数: {params.F}, 需要家数: {params.H}, 時間スロット: {params.T}")
    
    solver = BendersSolver(params)
    result = solver.solve()
    
    # 結果の出力
    if result['status'] == 'OPTIMAL':
        logger.info(f"\n最適解が得られました")
        logger.info(f"目的関数値: {result['objective']:.2f}")
        logger.info(f"計算時間: {result['solve_time']:.2f} 秒")
        logger.info(f"反復回数: {result['iterations']}")
        
        solver.plot_convergence()
    else:
        logger.warning(f"最大反復回数に達しました")
        logger.info(f"最終目的関数値: {result.get('objective', 'N/A')}")
    
    