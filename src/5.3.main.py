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
    """実行可能性部分問題（式4）- バイナリ変数を連続緩和"""
    
    def __init__(self, params: Parameters, data: Dict, feeder_id: int = 0):
        self.params = params
        self.data = data
        self.feeder_id = feeder_id
        self.model = None
        self.dual_values = {}
        self.constraints = {}
    
    def build(self, P_f_fixed: np.ndarray):
        """実行可能性部分問題を構築（バイナリ変数を連続緩和）"""
        model = gp.Model(f"Feasibility_f{self.feeder_id}")
        model.setParam('OutputFlag', 0)
        
        T = self.params.T
        H = self.params.H
        
        # ========== 変数定義（バイナリ変数を連続緩和 + スラック変数）==========
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
        
        # ========== スラック変数（電圧違反のみ）==========
        V_under = model.addVars(range(1, T+1), range(H), name='V_under', lb=0)
        V_over = model.addVars(range(1, T+1), range(H), name='V_over', lb=0)
        S_over = model.addVars(range(1, T+1), range(H), name='S_over', lb=0)
        V_under_safe = model.addVars(range(1, T+1), range(H), name='V_under_safe', lb=0)
        V_over_safe = model.addVars(range(1, T+1), range(H), name='V_over_safe', lb=0)
        S_over_safe = model.addVars(range(1, T+1), range(H), name='S_over_safe', lb=0)
        
        model.update()
        
        # ========== 目的関数（式4）==========
        # スラック変数の合計を最小化
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
        
        # HP制約
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
        
        # 蓄電池制約
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
                # 式(4-1): V_min - V^(b,t,k) = V_under - V_safe
                c1 = model.addConstr(
                    self.params.V_LL - V_d[t,k] == V_under[t,k] - V_under_safe[t,k],
                    name=f'V_lower_slack_{t}_{k}'
                )
                # 式(4-2): V^(b,t,k) - V_max = V_over - V_over_safe
                c2 = model.addConstr(
                    V_d[t,k] - self.params.V_UL == V_over[t,k] - V_over_safe[t,k],
                    name=f'V_upper_slack_{t}_{k}'
                )
                voltage_lower.append(c1)
                voltage_upper.append(c2)
        
                # 式(4-3): √(P²+Q²) - S_max ≤ S_over
                # 【重要】スラック変数を含む二次錐制約の線形化
                # 元の制約: ap² + aq² ≤ (S_LINE + S_over)²
                # 連続緩和版では線形化が望ましい場合がありますが、
                # ここではそのまま二次制約として保持
                c3 = model.addConstr(
                    ap[t,k]*ap[t,k] + aq[t,k]*aq[t,k] == (S_over[t,k] + self.params.S_LINE)**2,
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
            except AttributeError as e:
                logger.warning(f"  実行可能性SPで双対変数取得エラー: {e}")
                self.dual_values = {}
                
            return violation, self.dual_values
        
        elif self.model.status == gp.GRB.INFEASIBLE:
            logger.error("  実行可能性部分問題が実行不可能です")
            
            # ★ 問題を診断して出力
            self.diagnose_and_write_problem("エラー.lp")
            
            raise RuntimeError(f"実行可能性部分問題が解けません: status={self.model.status}")
        
        elif self.model.status == gp.GRB.UNBOUNDED:
            logger.error("  実行可能性部分問題が非有界です")
            
            # ★ 問題を診断して出力
            self.diagnose_and_write_problem("エラー.lp")
            
            raise RuntimeError(f"実行可能性部分問題が非有界: status={self.model.status}")
        
        elif self.model.status == gp.GRB.INF_OR_UNBD:
            logger.error("  実行可能性部分問題: INF_OR_UNBD")
            
            # ★ 問題を診断して出力（再最適化して詳細判定）
            self.diagnose_and_write_problem("エラー.lp")
            
            raise RuntimeError(f"実行可能性部分問題がINF_OR_UNBD: status={self.model.status}")
        
        else:
            logger.warning(f"  実行可能性部分問題: 予期しないステータス={self.model.status}")
            
            # ★ 問題を診断して出力
            self.diagnose_and_write_problem("エラー.lp")
            
            raise RuntimeError(f"実行可能性部分問題が予期しないステータス: status={self.model.status}")

    def diagnose_and_write_problem(self, filename: str = "エラー.lp"):
            """実行不可能または非有界の原因を診断して出力"""
            if self.model is None:
                logger.error("モデルが構築されていません")
                return
            
            status = self.model.status
            
            if status == gp.GRB.INFEASIBLE:
                logger.info("="*80)
                logger.info("実行不可能な問題を診断中...")
                logger.info("="*80)
                
                try:
                    # IIS（矛盾する最小制約集合）を計算
                    self.model.computeIIS()
                    
                    # IISをファイルに出力
                    self.model.write(filename)
                    logger.info(f"矛盾する制約を {filename} に出力しました")
                    
                    # テキストファイルにも詳細を出力
                    txt_filename = filename.replace('.lp', '_detail.txt')
                    with open(txt_filename, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write("実行不可能の原因となっている制約\n")
                        f.write("="*80 + "\n\n")
                        
                        # 矛盾制約をリストアップ
                        infeasible_constrs = []
                        for c in self.model.getConstrs():
                            if c.IISConstr:
                                infeasible_constrs.append(c)
                        
                        f.write(f"矛盾制約数: {len(infeasible_constrs)}\n\n")
                        
                        # 制約タイプ別に集計
                        constr_types = {}
                        for c in infeasible_constrs:
                            cname = c.ConstrName
                            # 制約タイプを判定
                            if 'power_allocation' in cname:
                                ctype = '電力配分制約'
                            elif 'V_lower_slack' in cname:
                                ctype = '電圧下限制約（スラック付き）'
                            elif 'V_upper_slack' in cname:
                                ctype = '電圧上限制約（スラック付き）'
                            elif 'P_HP' in cname or 'H_prod' in cname or 'H_tank' in cname:
                                ctype = 'HP給湯器制約'
                            elif 'P_Ch' in cname or 'P_DCh' in cname or 'E_B' in cname:
                                ctype = '蓄電池制約'
                            elif 'V_dd' in cname or 'V_d[' in cname:
                                ctype = '電圧推定制約'
                            elif 'ap[' in cname or 'aq[' in cname:
                                ctype = '電力フロー計算制約'
                            elif 'Delta' in cname:
                                ctype = 'バイナリ制約'
                            else:
                                ctype = 'その他'
                            
                            if ctype not in constr_types:
                                constr_types[ctype] = []
                            constr_types[ctype].append(c)
                        
                        # タイプ別に出力
                        f.write("【制約タイプ別の集計】\n")
                        for ctype, constrs in sorted(constr_types.items(), key=lambda x: len(x[1]), reverse=True):
                            f.write(f"  {ctype}: {len(constrs)}個\n")
                        
                        f.write("\n" + "="*80 + "\n")
                        f.write("【詳細な制約式】\n")
                        f.write("="*80 + "\n\n")
                        
                        # 各制約の詳細を出力
                        for ctype, constrs in sorted(constr_types.items()):
                            f.write(f"\n■ {ctype} ({len(constrs)}個)\n")
                            f.write("-"*80 + "\n")
                            
                            for i, c in enumerate(constrs):
                                f.write(f"\n制約 {i+1}: {c.ConstrName}\n")
                                
                                # 制約式を構築
                                row = self.model.getRow(c)
                                expr_parts = []
                                for j in range(row.size()):
                                    coeff = row.getCoeff(j)
                                    var = row.getVar(j)
                                    if abs(coeff) > 1e-10:  # 極小値は無視
                                        if coeff >= 0 and len(expr_parts) > 0:
                                            expr_parts.append(f"+ {coeff:.6f}*{var.VarName}")
                                        elif coeff < 0:
                                            expr_parts.append(f"- {abs(coeff):.6f}*{var.VarName}")
                                        else:
                                            expr_parts.append(f"{coeff:.6f}*{var.VarName}")
                                
                                sense = c.Sense
                                if sense == '=':
                                    sense_str = "=="
                                elif sense == '<':
                                    sense_str = "<="
                                else:
                                    sense_str = ">="
                                
                                expr_str = " ".join(expr_parts)
                                f.write(f"  {expr_str}\n")
                                f.write(f"  {sense_str} {c.RHS:.6f}\n")
                        
                        # 矛盾変数をリストアップ
                        f.write("\n" + "="*80 + "\n")
                        f.write("【矛盾する変数の境界】\n")
                        f.write("="*80 + "\n\n")
                        
                        infeasible_vars = []
                        for v in self.model.getVars():
                            if v.IISLB or v.IISUB:
                                infeasible_vars.append(v)
                        
                        if infeasible_vars:
                            f.write(f"矛盾変数数: {len(infeasible_vars)}\n\n")
                            
                            var_types = {}
                            for v in infeasible_vars:
                                vname = v.VarName
                                if 'V_under' in vname or 'V_over' in vname:
                                    vtype = 'スラック変数'
                                elif 'P_' in vname:
                                    vtype = '電力変数'
                                elif 'E_B' in vname:
                                    vtype = '蓄電池変数'
                                elif 'H_' in vname:
                                    vtype = 'HP変数'
                                elif 'Delta' in vname:
                                    vtype = 'バイナリ変数（緩和）'
                                elif 'V_' in vname:
                                    vtype = '電圧変数'
                                else:
                                    vtype = 'その他'
                                
                                if vtype not in var_types:
                                    var_types[vtype] = []
                                var_types[vtype].append(v)
                            
                            for vtype, vars_list in sorted(var_types.items()):
                                f.write(f"\n■ {vtype} ({len(vars_list)}個)\n")
                                for v in vars_list[:10]:  # 最初の10個
                                    lb_marker = "★LB矛盾 " if v.IISLB else ""
                                    ub_marker = "★UB矛盾 " if v.IISUB else ""
                                    f.write(f"  {lb_marker}{ub_marker}{v.VarName}: [{v.LB}, {v.UB}]\n")
                                if len(vars_list) > 10:
                                    f.write(f"  ... 他 {len(vars_list)-10} 個\n")
                        else:
                            f.write("矛盾する変数境界はありません\n")
                    
                    logger.info(f"詳細な診断結果を {txt_filename} に出力しました")
                    
                    # コンソールにサマリーを表示
                    logger.info("\n" + "="*80)
                    logger.info("【実行不可能の原因サマリー】")
                    logger.info("="*80)
                    for ctype, constrs in sorted(constr_types.items(), key=lambda x: len(x[1]), reverse=True):
                        logger.info(f"  {ctype}: {len(constrs)}個")
                    logger.info("="*80 + "\n")
                    
                except Exception as e:
                    logger.error(f"IIS計算中にエラー: {e}")
                    # フォールバック: モデル全体を出力
                    self.model.write(filename)
                    logger.info(f"エラーのためモデル全体を {filename} に出力しました")
            
            elif status == gp.GRB.UNBOUNDED:
                logger.info("="*80)
                logger.info("非有界な問題を診断中...")
                logger.info("="*80)
                
                try:
                    # 非有界レイ（unbounded ray）を計算
                    self.model.setParam('InfUnbdInfo', 1)
                    self.model.optimize()
                    
                    # 非有界方向の変数を特定
                    with open(filename.replace('.lp', '_unbounded.txt'), 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write("非有界の原因となっている変数\n")
                        f.write("="*80 + "\n\n")
                        
                        unbounded_vars = []
                        for v in self.model.getVars():
                            try:
                                if abs(v.UnbdRay) > 1e-6:
                                    unbounded_vars.append((v.VarName, v.UnbdRay, v.LB, v.UB))
                            except:
                                pass
                        
                        if unbounded_vars:
                            f.write(f"非有界方向の変数数: {len(unbounded_vars)}\n\n")
                            f.write("変数名                           | UnbdRay  | 下限      | 上限\n")
                            f.write("-"*80 + "\n")
                            
                            for vname, ray, lb, ub in sorted(unbounded_vars, key=lambda x: abs(x[1]), reverse=True):
                                f.write(f"{vname:30s} | {ray:8.4f} | {lb:9.4f} | {ub}\n")
                            
                            # コンソールにも表示
                            logger.info("\n非有界方向の主要変数:")
                            for vname, ray, lb, ub in unbounded_vars[:10]:
                                logger.info(f"  {vname}: UnbdRay={ray:.4f}, bounds=[{lb}, {ub}]")
                        else:
                            f.write("非有界レイ情報が取得できませんでした\n")
                            logger.warning("非有界レイ情報が取得できませんでした")
                    
                    # モデル全体も出力
                    self.model.write(filename)
                    logger.info(f"モデル全体を {filename} に出力しました")
                    
                except Exception as e:
                    logger.error(f"非有界診断中にエラー: {e}")
                    self.model.write(filename)
            
            elif status == gp.GRB.INF_OR_UNBD:
                logger.info("実行不可能または非有界（INF_OR_UNBD）")
                # パラメータ設定後に再最適化
                self.model.setParam('DualReductions', 0)
                self.model.setParam('InfUnbdInfo', 1)
                self.model.optimize()
                
                # 再帰的に診断
                self.diagnose_and_write_problem(filename)
            
            else:
                logger.warning(f"診断不可能なステータス: {status}")
                self.model.write(filename)
                logger.info(f"モデル全体を {filename} に出力しました")

# ================================================================================
# ベンダーズ分解ソルバー
# ================================================================================

class BendersSolver:
    """アルゴリズムに準拠したベンダーズ分解法の実装"""
    
    def __init__(self, params: Parameters):
        self.params = params
        self.data_loader = DataLoader(params)
        self.data = self.data_loader.load_all_data()
        
        self.master = MasterProblem(params)
        self.subproblems = []
        self.feasibility_subproblems = []  # 実行可能性SP用
        
        for f in range(params.F):
            self.subproblems.append(Subproblem(params, self.data, f))
            self.feasibility_subproblems.append(FeasibilitySubproblem(params, self.data, f))
        
        self.iteration = 0
        self.UBD = float('inf')  # 上界
        self.LBD = float('-inf')  # 下界
        self.convergence_history = []
        
        self.optimal_P_f = None
        self.optimal_objective = None
    
    def solve(self) -> Dict:
        """アルゴリズムに準拠したベンダーズ分解法"""
        start_time = time.time()
        
        # Step 1: 初期設定
        # P_PG^f(n)を初期推定値で固定
        P_f_current = self.master.get_initial_solution()
        logger.info(f"初期P_PG: 平均 = {P_f_current.mean():.2f} kW")
        
        while self.iteration < self.params.MAX_ITERATIONS:
            self.iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"反復 n = {self.iteration}")
            logger.info(f"{'='*60}")
            
            # ========== Step 1: SPの解法 ==========
            all_optimal = True
            sub_objectives = []
            dual_values_list = []
            
            for f in range(self.params.F):
                logger.info(f"\n引込線 f={f} の最適性SPを解いています...")
                
                # 最適性SPを解く
                self.subproblems[f].build(P_f_current[f, :])
                status, obj_val, duals = self.subproblems[f].solve()
                
                if status == 'OPTIMAL':
                    # 実行可能な場合
                    logger.info(f"  最適性SP: 実行可能")
                    logger.info(f"  目的関数値 Z_sub = {obj_val:.2f}")
                    sub_objectives.append(obj_val)
                    dual_values_list.append(duals)
                    
                else:
                    # 実行不可能な場合
                    logger.info(f"  最適性SP: 実行不可能")
                    all_optimal = False
                    
                    # 実行可能性SPを解く
                    logger.info(f"  実行可能性SPを解いています...")
                    self.feasibility_subproblems[f].build(P_f_current[f, :])
                    violation, feas_duals = self.feasibility_subproblems[f].solve()
                    logger.info(f"  違反量 = {violation:.4f}")
                    
                    # 実行可能性カットを生成
                    self._generate_feasibility_cut_from_sp(violation, feas_duals, P_f_current)
                    break
            
            if all_optimal:
                # アルゴリズムの式: UBD^(n) = Z_sub + Σ P_PG^f(n)
                Z_sub = sum(sub_objectives)
                
                # 下げDR期間の総買電量を計算
                P_PG_dr_sum = 0
                for f in range(self.params.F):
                    for t in range(self.params.DOWN_DR_START_HOUR, self.params.DOWN_DR_END_HOUR+1):
                        if t <= self.params.T:
                            P_PG_dr_sum += P_f_current[f, t-1]
                
                # 目的関数値の計算（下げDR期間買電 + 電力コスト）
                UBD_n = Z_sub + P_PG_dr_sum
                
                logger.info(f"\n【上界の計算】")
                logger.info(f"  Z_sub = {Z_sub:.2f}")
                logger.info(f"  P_PG(下げDR) = {P_PG_dr_sum:.2f}")
                logger.info(f"  UBD^({self.iteration}) = {UBD_n:.2f}")
                
                # 上界の更新判定
                if UBD_n < self.UBD:
                    self.UBD = UBD_n
                    self.optimal_P_f = P_f_current.copy()
                    logger.info(f"  → 上界更新: UBD = {self.UBD:.2f}")
                
                # 最適性カットを生成
                logger.info("\n最適性カットを生成中...")
                self._generate_optimality_cut_from_duals(
                    sub_objectives, P_f_current, dual_values_list
                )
            
            # ========== Step 2: MPの解法と更新 ==========
            logger.info("\n主問題(MP)を解いています...")
            P_f_new, eta = self.master.solve()
            
            # 下界の更新（MPの目的関数値）
            self.LBD = eta
            logger.info(f"  新しいP_PG: 平均 = {P_f_new.mean():.2f} kW")
            logger.info(f"  下界 LBD = {self.LBD:.2f}")
            
            # ========== 収束判定 ==========
            logger.info(f"\n【収束状況】")
            logger.info(f"  上界 UBD = {self.UBD:.2f}")
            logger.info(f"  下界 LBD = {self.LBD:.2f}")
            
            if self.UBD != float('inf') and self.LBD != float('-inf'):
                gap = abs(self.UBD - self.LBD)
                relative_gap = gap / abs(self.UBD) if self.UBD != 0 else float('inf')
                
                logger.info(f"  絶対ギャップ = {gap:.4f}")
                logger.info(f"  相対ギャップ = {relative_gap*100:.2f}%")
                
                # 収束判定: UBD - LBD < ε
                if gap < self.params.EPSILON or relative_gap < self.params.EPSILON:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"★ 収束しました！（反復 {self.iteration} 回）")
                    logger.info(f"最適目的関数値: {self.UBD:.2f}")
                    logger.info(f"{'='*60}")
                    self.optimal_objective = self.UBD
                    break
            
            # 履歴の記録
            self.convergence_history.append({
                'iteration': self.iteration,
                'UBD': self.UBD,
                'LBD': self.LBD,
                'gap': abs(self.UBD - self.LBD) if self.UBD != float('inf') else float('inf')
            })
            
            # 次の反復へ
            P_f_current = P_f_new.copy()
        
        solve_time = time.time() - start_time
        
        return {
            'status': 'OPTIMAL' if self.optimal_objective is not None else 'MAX_ITERATIONS',
            'objective': self.optimal_objective,
            'P_f': self.optimal_P_f,
            'iterations': self.iteration,
            'solve_time': solve_time,
            'convergence_history': self.convergence_history
        }
    
    def _generate_optimality_cut_from_duals(self, sub_objectives, P_f_current, dual_values_list):
        """最適性カット生成（双対変数使用）"""
        T = self.params.T
        F = self.params.F
        
        # カットの定数項（サブ問題の目的関数値の合計）
        cut_constant = sum(sub_objectives)
        
        # カットの勾配（双対変数から）
        cut_gradient = np.zeros((F, T))
        
        for f in range(F):
            if f < len(dual_values_list) and dual_values_list[f]:
                # 双対変数が取得できた場合
                if 'power_allocation' in dual_values_list[f]:
                    for t, dual in dual_values_list[f]['power_allocation'].items():
                        if 1 <= t <= T:
                            cut_gradient[f, t-1] = dual
                            
        # 双対変数が不完全な場合のデフォルト勾配
        for f in range(F):
            for t in range(T):
                if cut_gradient[f, t] == 0:
                    # 下げDR期間により大きな重み
                    if self.params.DOWN_DR_START_HOUR <= t+1 <= self.params.DOWN_DR_END_HOUR:
                        cut_gradient[f, t] = 1.5
                    else:
                        cut_gradient[f, t] = 1.0
        
        self.master.add_optimality_cut(cut_constant, cut_gradient, P_f_current)
        logger.info(f"  最適性カット追加（定数項: {cut_constant:.2f}）")
    
    def _generate_feasibility_cut_from_sp(self, violation, feas_duals, P_f_current):
        """実行可能性カット生成（実行可能性SPの結果から）"""
        T = self.params.T
        F = self.params.F
        
        # カットの勾配（実行可能性SPの双対変数から）
        cut_gradient = np.ones((F, T)) * 0.1  # デフォルト
        
        if feas_duals and 'lambda_1' in feas_duals:
            # 双対変数から勾配を計算
            for f in range(F):
                for t in range(T):
                    # 電圧制約の双対変数などから勾配を推定
                    idx = f * T + t
                    if idx < len(feas_duals['lambda_1']):
                        cut_gradient[f, t] = abs(feas_duals['lambda_1'][idx]) * 0.1
        
        self.master.add_feasibility_cut(cut_gradient, P_f_current, violation)
        logger.info(f"  実行可能性カット追加（違反量: {violation:.4f}）")
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

    def save_detailed_results(self):
        """最適解の詳細結果を保存"""
        if self.optimal_P_f is None:
            logger.warning("最適解が存在しないため、結果を保存できません")
            return
        
        # 結果出力用ディレクトリ
        output_dir = "../data/output/optimization"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("\n詳細な結果を出力中...")
        
        # 最適解でサブ問題を再度解く（詳細な変数値を取得するため）
        for f in range(self.params.F):
            self.subproblems[f].build(self.optimal_P_f[f, :])
            status, obj_val, _ = self.subproblems[f].solve()
            
            if status != 'OPTIMAL':
                logger.error(f"最適解での部分問題が解けません（引込線{f}）")
                return
        
        # 引込線0の結果を使用（現在は引込線1本のみ）
        sub_model = self.subproblems[0].model
        sub_vars = self.subproblems[0].variables
        
        T = self.params.T
        H = self.params.H
        
        # ========== 各需要家の結果出力 ==========
        for i in range(H):
            # コスト計算
            cost_data = []
            for t in range(1, T+1):
                P_PG_val = self.optimal_P_f[0, t-1]  # 引込線0の電力
                P_SG_val = sub_vars['P_SG'][t].X
                cost_val = self.data['buy_price'][t] * P_PG_val - self.params.SELL_PRICE * P_SG_val
                cost_data.append(cost_val)
            
            # データフレーム作成
            benefit_df = pd.DataFrame({
                'コマ': [self.data['time'][t] for t in range(1, T+1)],
                '電力コスト[JPY]': cost_data,
                'アグリゲータの買電電力[kW・30分]': [self.optimal_P_f[0, t-1] for t in range(1, T+1)],
                'アグリゲータの売電電力[kW・30分]': [-sub_vars['P_SG'][t].X for t in range(1, T+1)],
                '買電電力[kW・30分]': [sub_vars['P_pg'][t, i].X for t in range(1, T+1)],
                '売電電力[kW・30分]': [-sub_vars['P_sg'][t, i].X for t in range(1, T+1)],
                '一軒に供給される電力量[kW・30分]': [sub_model.getVarByName(f'P_in[{t},{i}]').X for t in range(1, T+1)],
                '一軒から放出される電力量[kW・30分]': [-sub_model.getVarByName(f'P_out[{t},{i}]').X for t in range(1, T+1)],
                '電気料金[JPY・30min/kW]': [self.data['buy_price'][t] for t in range(1, T+1)],
                '蓄電量[kWh]': [sub_model.getVarByName(f'E_B[{t},{i}]').X for t in range(1, T+1)],
                '充電量[kW・30分]': [-sub_model.getVarByName(f'P_Ch[{t},{i}]').X for t in range(1, T+1)],
                '放電量[kW・30分]': [sub_model.getVarByName(f'P_DCh[{t},{i}]').X for t in range(1, T+1)],
                'HP給湯機消費電力[kW・30分]': [sub_model.getVarByName(f'P_HP[{t},{i}]').X for t in range(1, T+1)],
                '充電中変数': [sub_model.getVarByName(f'Delta_ch[{t},{i}]').X for t in range(1, T+1)],
                '放電中変数': [sub_model.getVarByName(f'Delta_dch[{t},{i}]').X for t in range(1, T+1)],
                '売買変数': [sub_model.getVarByName(f'Delta_in[{t},{i}]').X for t in range(1, T+1)],
                '電力需要量[kW・30min]': [self.data['p_dmd'][t, i] for t in range(1, T+1)],
                '太陽光発電出力[kW・30min]': [self.data['p_pv'][t, i] for t in range(1, T+1)]
            })
            
            # CSV保存
            house_csv_path = os.path.join(output_dir, f"0.house_{i}.csv")
            benefit_df.to_csv(house_csv_path, encoding="shift_jis", index=False)
        
        # ========== 電圧関連の結果出力 ==========
        voltage_data = {
            'Time': [self.data['time'][t] for t in range(1, T+1)],
            '電圧上限値': [self.params.V_UL] * T,
            '電圧下限値': [self.params.V_LL] * T
        }
        
        # 各バスの電圧関連データ
        for i in range(H):
            voltage_data[f'bus{i+1}の電圧'] = [sub_vars['V_d'][t, i].X for t in range(1, T+1)]
            voltage_data[f'bus{i+1}の電圧上限違反値'] = [
                max(0, sub_vars['V_d'][t, i].X - self.params.V_UL) for t in range(1, T+1)
            ]
            voltage_data[f'bus{i+1}の電圧下限違反値'] = [
                max(0, self.params.V_LL - sub_vars['V_d'][t, i].X) for t in range(1, T+1)
            ]
            voltage_data[f'bus{i+1}の有効電力'] = [sub_vars['ap'][t, i].X for t in range(1, T+1)]
            voltage_data[f'bus{i+1}の電圧降下'] = [sub_model.getVarByName(f'V_dd[{t},{i}]').X for t in range(1, T+1)]
        
        voltage_df = pd.DataFrame(voltage_data)
        voltage_csv_path = os.path.join(output_dir, "1.voltage.csv")
        voltage_df.to_csv(voltage_csv_path, encoding="shift_jis", index=False)
        
        # ========== グラフ生成 ==========
        times = pd.to_datetime(voltage_df["Time"], errors="coerce")
        x = times.dt.strftime("%H:%M").fillna(voltage_df["Time"].astype(str)).tolist()
        x_label = "時刻"
        
        # 1) 各バスの電圧と上下限
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for i in range(H):
            ax.plot(x, voltage_df[f'bus{i+1}の電圧'], label=f'bus{i+1}の電圧', linewidth=1.8)
        ax.plot(x, voltage_df['電圧上限値'], '--', color='red', label='電圧上限値', linewidth=1.2)
        ax.plot(x, voltage_df['電圧下限値'], '--', color='blue', label='電圧下限値', linewidth=1.2)
        ax.set_title("バス別電圧")
        ax.set_xlabel(x_label)
        ax.set_ylabel("電圧[V]")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        plt.xticks(rotation=45)
        fig.savefig(os.path.join(output_dir, "1a.voltage_timeseries.png"), dpi=200)
        plt.close(fig)
        
        # 2) 電圧違反値
        fig, axes = plt.subplots(H, 1, figsize=(12, max(3, 2*H)), sharex=True, constrained_layout=True)
        if H == 1:
            axes = [axes]
        for i in range(H):
            axes[i].plot(x, voltage_df[f'bus{i+1}の電圧上限違反値'], label='上限違反値', color='red', linewidth=1.5)
            axes[i].plot(x, voltage_df[f'bus{i+1}の電圧下限違反値'], label='下限違反値', color='blue', linewidth=1.5)
            axes[i].set_title(f'bus{i+1} 電圧違反値')
            axes[i].set_ylabel("違反[V]")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=9)
        axes[-1].set_xlabel(x_label)
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        fig.savefig(os.path.join(output_dir, "1b.voltage_violations.png"), dpi=200)
        plt.close(fig)
        
        # 3) 各バスの有効電力
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for i in range(H):
            ax.plot(x, voltage_df[f'bus{i+1}の有効電力'], label=f'bus{i+1}の有効電力', linewidth=1.8)
        ax.set_title("バス別 有効電力")
        ax.set_xlabel(x_label)
        ax.set_ylabel("有効電力[kW]")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        plt.xticks(rotation=45)
        fig.savefig(os.path.join(output_dir, "1c.active_power_timeseries.png"), dpi=200)
        plt.close(fig)
        
        # 4) 各バスの電圧降下
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for i in range(H):
            ax.plot(x, voltage_df[f'bus{i+1}の電圧降下'], label=f'bus{i+1}の電圧降下', linewidth=1.8)
        ax.set_title("バス別 電圧降下")
        ax.set_xlabel(x_label)
        ax.set_ylabel("電圧降下[V]")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
        plt.xticks(rotation=45)
        fig.savefig(os.path.join(output_dir, "1d.voltage_drop_timeseries.png"), dpi=200)
        plt.close(fig)
        
        # ========== ヒートポンプ結果の出力 ==========
        for i in range(H):
            heat_pump_df = pd.DataFrame({
                'Time': [self.data['time'][t] for t in range(1, T+1)],
                'HP給湯機消費電力[kW・30分]': [sub_model.getVarByName(f'P_HP[{t},{i}]').X for t in range(1, T+1)],
                '熱需要[kW・30分]': [self.data['H_dmd'][t, i] for t in range(1, T+1)],
                '熱製造量[MJ・30分]': [sub_model.getVarByName(f'H_prod[{t},{i}]').X for t in range(1, T+1)],
                '貯湯量[MJ]': [sub_model.getVarByName(f'H_tank[{t},{i}]').X for t in range(1, T+1)],
                '成績係数': [self.data['cop'][t] for t in range(1, T+1)],
                '運転変数': [sub_model.getVarByName(f'Delta_prod[{t},{i}]').X for t in range(1, T+1)],
                '運転開始変数': [sub_model.getVarByName(f'Delta_S[{t},{i}]').X for t in range(1, T+1)],
                '運転終了変数': [sub_model.getVarByName(f'Delta_F[{t},{i}]').X for t in range(1, T+1)],
                '電気料金[JPY・30min/kW]': [self.data['buy_price'][t] for t in range(1, T+1)]
            })
            
            output_path = os.path.join(output_dir, f"2.HP_{i}.csv")
            heat_pump_df.to_csv(output_path, encoding="shift_jis", index=False)
            
            # グラフ作成
            times = pd.to_datetime(heat_pump_df["Time"], errors="coerce")
            x = times.dt.strftime("%H:%M").fillna(heat_pump_df["Time"].astype(str)).tolist()
            
            fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
            ax1.bar(x, heat_pump_df['HP給湯機消費電力[kW・30分]'], color='skyblue', label='HP給湯機消費電力', alpha=0.7)
            ax1.set_xlabel("時刻")
            ax1.set_ylabel("HP給湯機消費電力[kW・30分]")
            ax1.tick_params(axis='y')
            ax1.set_xticks(x[::4])
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            ax2.plot(x, heat_pump_df['電気料金[JPY・30min/kW]'], color='orange', label='電気料金', linewidth=2)
            ax2.set_ylabel("電気料金[JPY・30min/kW]")
            ax2.tick_params(axis='y')
            
            fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), fontsize=9)
            plot_path = os.path.join(output_dir, f"2.HP_{i}_timeseries.png")
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
        
        # ========== 蓄電池結果の出力 ==========
        for i in range(H):
            ess_df = pd.DataFrame({
                'Time': [self.data['time'][t] for t in range(1, T+1)],
                '蓄電量[kWh]': [sub_model.getVarByName(f'E_B[{t},{i}]').X for t in range(1, T+1)],
                '蓄電池容量上限[kWh]': [self.params.N_B] * T,
                '蓄電池容量下限[kWh]': [self.params.N_B * 0.2] * T
            })
            
            output_path = os.path.join(output_dir, f"3.ESS_{i}.csv")
            ess_df.to_csv(output_path, encoding="shift_jis", index=False)
            
            times = pd.to_datetime(ess_df["Time"], errors="coerce")
            x = times.dt.strftime("%H:%M").fillna(ess_df["Time"].astype(str)).tolist()
            
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
            ax.plot(x, ess_df['蓄電量[kWh]'], label='蓄電量', linewidth=1.8)
            ax.plot(x, ess_df['蓄電池容量上限[kWh]'], '--', color='red', label='容量上限', linewidth=1.2)
            ax.plot(x, ess_df['蓄電池容量下限[kWh]'], '--', color='blue', label='容量下限', linewidth=1.2)
            ax.set_xlabel("時刻")
            ax.set_ylabel("蓄電量[kWh]")
            ax.set_title(f"蓄電池 状態（House {i}）")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            plt.xticks(rotation=45)
            
            plot_path = os.path.join(output_dir, f"3.ESS_{i}_timeseries.png")
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
        
        logger.info("詳細結果の出力完了")
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
        
        # 詳細結果の保存（BendersSolverクラスにメソッドを追加）
        solver.save_detailed_results()
        
        # 収束グラフの表示
        solver.plot_convergence()
    else:
        logger.warning(f"最大反復回数に達しました")
        logger.info(f"最終目的関数値: {result.get('objective', 'N/A')}")
    