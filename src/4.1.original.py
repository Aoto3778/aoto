"""
Centralized Optimization for Distribution Network
配電網の全体最適化（ベンチマーク用）

すべての需要家と配電網を統合した最適化問題を一度に解く
GBDとの比較評価用の集中型アプローチ
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import logging
import time
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Tuple
import os

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CentralizedOptimizer:
    """配電網と全需要家の統合最適化"""
    
    def __init__(self):
        """初期化"""
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
        
        # 下げDR期間（13時～16時）
        self.DR_START = 26  # 13:00 (slot 26)
        self.DR_END = 32    # 16:00 (slot 32)
        
        # データ読み込み
        self.load_data()
        
        # モデル構築
        self.model = gp.Model("Centralized_Optimization")
        self.setup_model()
        
    def load_data(self):
        """データ読み込み"""
        logger.info("データ読み込み開始...")
        
        # CSVファイル読み込み
        dmd_data = pd.read_csv("../data/input/electric_demand_1y_2004/electric_demand_1y_30min_01.csv", encoding="shift_jis")
        buy_price_data = pd.read_csv("../data/input/buy_energy30.csv", encoding="shift_jis")
        spot_price_data = pd.read_csv("../data/input/spot_market_price_30min.csv", encoding="shift_jis")
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
            demand_value = float(dmd_data.iat[t-1, 10])
            for h in range(self.H):
                self.p_dmd[t, h] = demand_value * (0.8 + 0.4 * np.random.rand())  # 需要家ごとに変動
        
        # 買電単価[JPY/kWh]
        self.buy_price = {}
        for t in range(1, self.T+1):
            self.buy_price[t] = float(buy_price_data.iat[t-1, 2])
        
        # 熱需要[MJ]
        self.heat_dmd = {}
        for t in range(1, self.T+1):
            heat_value = float(heat_dmd_data.iat[t-1+self.T, 2])
            for h in range(self.H):
                self.heat_dmd[t, h] = heat_value * (0.7 + 0.6 * np.random.rand())
        
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
        
    def setup_model(self):
        """最適化モデルの構築"""
        logger.info("最適化モデル構築開始...")
        
        # モデル設定
        self.model.setParam('OutputFlag', 1)
        self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', 3600)  # 1時間制限
        
        # ========== 決定変数 ==========
        # アグリゲータレベル
        self.P_AG_buy = self.model.addVars(range(1, self.T+1), 
                                           lb=0, name="P_AG_buy")
        self.P_AG_sell = self.model.addVars(range(1, self.T+1), 
                                            lb=0, name="P_AG_sell")
        self.delta_AG_buy = self.model.addVars(range(1, self.T+1), 
                                               vtype=GRB.BINARY, name="delta_AG_buy")
        self.delta_AG_sell = self.model.addVars(range(1, self.T+1), 
                                                vtype=GRB.BINARY, name="delta_AG_sell")
        
        # 需要家レベル
        self.P_buy = self.model.addVars(range(1, self.T+1), range(self.H), 
                                        lb=0, ub=self.BUY_MAX, name="P_buy")
        self.P_sell = self.model.addVars(range(1, self.T+1), range(self.H), 
                                         lb=0, ub=self.SELL_MAX, name="P_sell")
        self.delta_buy = self.model.addVars(range(1, self.T+1), range(self.H), 
                                            vtype=GRB.BINARY, name="delta_buy")
        self.delta_sell = self.model.addVars(range(1, self.T+1), range(self.H), 
                                             vtype=GRB.BINARY, name="delta_sell")
        
        # 蓄電池
        self.P_ch = self.model.addVars(range(1, self.T+1), range(self.H), 
                                       lb=0, ub=self.N_B_PCS, name="P_charge")
        self.P_dch = self.model.addVars(range(1, self.T+1), range(self.H), 
                                        lb=0, ub=self.N_B_PCS, name="P_discharge")
        self.E_bat = self.model.addVars(range(self.T+1), range(self.H), 
                                        lb=0.2*self.N_B, ub=self.N_B, name="E_battery")
        self.delta_ch = self.model.addVars(range(1, self.T+1), range(self.H), 
                                           vtype=GRB.BINARY, name="delta_charge")
        self.delta_dch = self.model.addVars(range(1, self.T+1), range(self.H), 
                                            vtype=GRB.BINARY, name="delta_discharge")
        
        # ヒートポンプ
        self.P_hp = self.model.addVars(range(1, self.T+1), range(self.H), 
                                       lb=0, ub=10.0, name="P_hp")
        self.H_prod = self.model.addVars(range(1, self.T+1), range(self.H), 
                                         lb=0, name="H_produce")
        self.H_tank = self.model.addVars(range(self.T+1), range(self.H), 
                                         lb=0, name="H_tank")
        self.delta_hp = self.model.addVars(range(self.T+1), range(self.H), 
                                           vtype=GRB.BINARY, name="delta_hp")
        self.delta_start = self.model.addVars(range(self.T+1), range(self.H), 
                                              vtype=GRB.BINARY, name="delta_start")
        self.delta_stop = self.model.addVars(range(self.T+1), range(self.H), 
                                             vtype=GRB.BINARY, name="delta_stop")
        
        # 電圧
        self.V = self.model.addVars(range(1, self.T+1), range(self.H), 
                                    lb=self.V_LL, ub=self.V_UL, name="Voltage")
        self.V_drop = self.model.addVars(range(1, self.T+1), range(self.H), 
                                         lb=-GRB.INFINITY, name="V_drop")
        
        # 有効電力・無効電力
        self.P_flow = self.model.addVars(range(1, self.T+1), range(self.H), 
                                         lb=-GRB.INFINITY, name="P_flow")
        self.Q_flow = self.model.addVars(range(1, self.T+1), range(self.H), 
                                         lb=-GRB.INFINITY, name="Q_flow")
        
        self.model.update()
        
        # ========== 制約条件 ==========
        self._add_power_balance_constraints()
        self._add_battery_constraints()
        self._add_heatpump_constraints()
        self._add_voltage_constraints()
        self._add_line_capacity_constraints()
        
        # ========== 目的関数 ==========
        self._set_objective()
        
        logger.info("モデル構築完了")
        logger.info(f"変数数: {self.model.NumVars}")
        logger.info(f"制約数: {self.model.NumConstrs}")
        
    def _add_power_balance_constraints(self):
        """電力バランス制約"""
        for t in range(1, self.T+1):
            # アグリゲータレベルの需給バランス
            self.model.addConstr(
                self.P_AG_buy[t] == gp.quicksum(self.P_buy[t, h] for h in range(self.H)),
                name=f"AG_buy_balance_{t}"
            )
            self.model.addConstr(
                self.P_AG_sell[t] == gp.quicksum(self.P_sell[t, h] for h in range(self.H)),
                name=f"AG_sell_balance_{t}"
            )
            
            # アグリゲータの買電売電同時禁止
            self.model.addConstr(
                self.P_AG_buy[t] <= self.delta_AG_buy[t] * self.BUY_MAX * self.H
            )
            self.model.addConstr(
                self.P_AG_sell[t] <= self.delta_AG_sell[t] * self.SELL_MAX * self.H
            )
            self.model.addConstr(
                self.delta_AG_buy[t] + self.delta_AG_sell[t] <= 1
            )
            
            # 各需要家の電力バランス
            for h in range(self.H):
                self.model.addConstr(
                    self.p_pv[t, h] + self.P_buy[t, h] + self.P_dch[t, h] ==
                    self.p_dmd[t, h] + self.P_sell[t, h] + 
                    self.P_ch[t, h] + self.P_hp[t, h],
                    name=f"power_balance_{t}_{h}"
                )
                
                # 買電売電同時禁止
                self.model.addConstr(
                    self.P_buy[t, h] <= self.delta_buy[t, h] * self.BUY_MAX
                )
                self.model.addConstr(
                    self.P_sell[t, h] <= self.delta_sell[t, h] * self.SELL_MAX
                )
                self.model.addConstr(
                    self.delta_buy[t, h] + self.delta_sell[t, h] <= 1
                )
                
                # 潮流計算用
                self.model.addConstr(
                    self.P_flow[t, h] == self.P_buy[t, h] - self.P_sell[t, h],
                    name=f"P_flow_{t}_{h}"
                )
                self.model.addConstr(
                    self.Q_flow[t, h] == 0.1 * self.P_flow[t, h],
                    name=f"Q_flow_{t}_{h}"
                )
    
    def _add_battery_constraints(self):
        """蓄電池制約"""
        for h in range(self.H):
            # 初期条件
            self.model.addConstr(
                self.E_bat[0, h] == 0.5 * self.N_B,
                name=f"battery_init_{h}"
            )
            
            for t in range(1, self.T+1):
                # 充放電制約
                self.model.addConstr(
                    self.P_ch[t, h] <= self.N_B_PCS * self.delta_ch[t, h]
                )
                self.model.addConstr(
                    self.P_dch[t, h] <= self.N_B_PCS * self.delta_dch[t, h]
                )
                self.model.addConstr(
                    self.delta_ch[t, h] + self.delta_dch[t, h] <= 1
                )
                
                # SOC更新
                self.model.addConstr(
                    self.E_bat[t, h] == self.E_bat[t-1, h] + 
                    0.5 * self.ETA_B * self.P_ch[t, h] -
                    0.5 * self.P_dch[t, h] / self.ETA_B,
                    name=f"battery_soc_{t}_{h}"
                )
            
            # 終端条件
            self.model.addConstr(
                self.E_bat[self.T, h] >= 0.5 * self.N_B,
                name=f"battery_final_{h}"
            )
    
    def _add_heatpump_constraints(self):
        """ヒートポンプ制約"""
        for h in range(self.H):
            # 初期条件
            self.model.addConstr(self.delta_hp[0, h] == 0)
            self.model.addConstr(self.delta_start[0, h] == 0)
            self.model.addConstr(self.delta_stop[0, h] == 0)
            self.model.addConstr(
                self.H_tank[0, h] == 0.5 * self.CW * self.VLT * 70,
                name=f"hp_tank_init_{h}"
            )
            
            for t in range(1, self.T+1):
                cop = self.cop[t]
                
                # HP消費電力
                self.model.addConstr(
                    self.P_hp[t, h] == self.XHP * self.delta_hp[t, h] +
                    (self.H_prod[t, h] + self.R2 * self.CPH * self.delta_start[t, h]) / 
                    (self.CE * cop),
                    name=f"hp_power_{t}_{h}"
                )
                
                # 熱製造量制約
                self.model.addConstr(
                    self.H_prod[t, h] >= 0.5 * self.R1 * self.CPH * self.delta_hp[t, h]
                )
                self.model.addConstr(
                    self.H_prod[t, h] <= 0.5 * self.CPH * self.delta_hp[t, h]
                )
                
                # 運転状態の遷移
                self.model.addConstr(
                    self.delta_hp[t, h] - self.delta_hp[t-1, h] == 
                    self.delta_start[t, h] - self.delta_stop[t, h],
                    name=f"hp_state_{t}_{h}"
                )
                
                # 起動停止同時禁止
                self.model.addConstr(
                    self.delta_start[t, h] + self.delta_stop[t, h] <= 1
                )
                
                # 貯湯槽更新
                self.model.addConstr(
                    self.H_tank[t, h] == self.H_tank[t-1, h] + 
                    self.H_prod[t, h] - self.heat_dmd[t, h],
                    name=f"hp_tank_{t}_{h}"
                )
                
                # 貯湯槽容量制約
                self.model.addConstr(
                    self.H_tank[t, h] >= 0.2 * self.CW * self.VLT * 70
                )
                self.model.addConstr(
                    self.H_tank[t, h] <= self.CW * self.VLT * 70
                )
            
            # 終端条件
            self.model.addConstr(
                self.H_tank[self.T, h] >= 0.5 * self.CW * self.VLT * 70,
                name=f"hp_tank_final_{h}"
            )
    
    def _add_voltage_constraints(self):
        """電圧制約"""
        for t in range(1, self.T+1):
            for h in range(self.H):
                # 累積潮流（下流の需要家を含む）
                P_cumulative = gp.quicksum(self.P_flow[t, j] for j in range(h, self.H))
                Q_cumulative = gp.quicksum(self.Q_flow[t, j] for j in range(h, self.H))
                
                # 電圧降下推定
                self.model.addConstr(
                    self.V_drop[t, h] == 
                    self.linear_ap[h] * P_cumulative + 
                    self.linear_aq[h] * Q_cumulative + 
                    self.linear_b[h],
                    name=f"voltage_drop_{t}_{h}"
                )
                
                # 電圧計算
                if h == 0:
                    self.model.addConstr(
                        self.V[t, h] == self.V_BASE + self.V_drop[t, h],
                        name=f"voltage_{t}_{h}"
                    )
                else:
                    self.model.addConstr(
                        self.V[t, h] == self.V[t, h-1] + self.V_drop[t, h],
                        name=f"voltage_{t}_{h}"
                    )
    
    def _add_line_capacity_constraints(self):
        """送電線容量制約"""
        for t in range(1, self.T+1):
            for h in range(self.H):
                # 累積潮流
                P_cumulative = gp.quicksum(self.P_flow[t, j] for j in range(h, self.H))
                Q_cumulative = gp.quicksum(self.Q_flow[t, j] for j in range(h, self.H))
                
                # 二次錐制約（送電線容量）
                self.model.addQConstr(
                    P_cumulative * P_cumulative + Q_cumulative * Q_cumulative <= 
                    (self.S_LINE/100) * (self.S_LINE/100),
                    name=f"line_capacity_{t}_{h}"
                )
    
    def _set_objective(self):
        """目的関数の設定"""
        # 電力コスト
        electricity_cost = gp.quicksum(
            self.buy_price[t] * self.P_AG_buy[t] - 
            self.SELL_PRICE * self.P_AG_sell[t]
            for t in range(1, self.T+1)
        )
        
        # DR期間の買電電力
        if self.DR_END <= self.T:
            ddr_total = gp.quicksum(
                self.P_AG_buy[t]
                for t in range(self.DR_START, self.DR_END+1)
            )
        else:
            ddr_total = 0
        
        # 目的関数設定
        self.model.setObjective(
            electricity_cost + ddr_total,
            GRB.MINIMIZE
        )
    
    def optimize(self) -> Dict:
        """最適化の実行"""
        logger.info("="*60)
        logger.info("全体最適化開始")
        logger.info("="*60)
        
        start_time = time.time()
        
        # 最適化実行
        self.model.optimize()
        
        solve_time = time.time() - start_time
        
        # 結果の処理
        results = {
            'status': self.model.Status,
            'objective': None,
            'solve_time': solve_time,
            'gap': None,
            'num_vars': self.model.NumVars,
            'num_constrs': self.model.NumConstrs
        }
        
        if self.model.Status == GRB.OPTIMAL:
            results['objective'] = self.model.ObjVal
            results['gap'] = self.model.MIPGap
            logger.info("最適解が見つかりました")
            logger.info(f"目的関数値: {results['objective']:.2f}")
            logger.info(f"計算時間: {solve_time:.2f}秒")
            logger.info(f"MIPギャップ: {results['gap']:.4f}")
            
            # 詳細結果の保存
            results['detail'] = self._extract_solution()
            
        elif self.model.Status == GRB.TIME_LIMIT:
            results['objective'] = self.model.ObjVal
            results['gap'] = self.model.MIPGap
            logger.warning("時間制限に到達しました")
            logger.info(f"暫定解: {results['objective']:.2f}")
            
        elif self.model.Status == GRB.INFEASIBLE:
            logger.error("問題が実行不可能です")
            self.model.computeIIS()
            self.model.write("centralized_iis.ilp")
            logger.info("IISをcentralized_iis.ilpに出力しました")
            
        else:
            logger.error(f"最適化失敗: Status={self.model.Status}")
        
        return results
    
    def _extract_solution(self) -> Dict:
        """解の詳細を抽出"""
        solution = {
            'P_AG_buy': {},
            'P_AG_sell': {},
            'voltage': {},
            'battery_soc': {},
            'hp_operation': {},
            'household_cost': {}
        }
        
        # アグリゲータレベル
        for t in range(1, self.T+1):
            solution['P_AG_buy'][t] = self.P_AG_buy[t].X
            solution['P_AG_sell'][t] = self.P_AG_sell[t].X
        
        # 需要家レベル
        for h in range(self.H):
            solution['voltage'][h] = {}
            solution['battery_soc'][h] = {}
            solution['hp_operation'][h] = {}
            household_cost = 0
            
            for t in range(1, self.T+1):
                solution['voltage'][h][t] = self.V[t, h].X
                solution['battery_soc'][h][t] = self.E_bat[t, h].X
                solution['hp_operation'][h][t] = self.delta_hp[t, h].X
                
                # 需要家コスト計算
                household_cost += (
                    self.buy_price[t] * self.P_buy[t, h].X -
                    self.SELL_PRICE * self.P_sell[t, h].X
                )
            
            solution['household_cost'][h] = household_cost
            logger.info(f"需要家{h}のコスト: {household_cost:.2f} JPY")
        
        return solution
    
    def export_results(self, results: Dict, filename_prefix: str = "centralized"):
        """結果のエクスポート"""
        if results['status'] not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            logger.warning("最適解が見つからなかったため、結果をエクスポートできません")
            return
        
        os.makedirs("../data/output/centralized", exist_ok=True)
        
        # 全体サマリー
        summary_df = pd.DataFrame({
            'Metric': ['Objective Value', 'Solve Time', 'MIP Gap', 
                      'Variables', 'Constraints', 'Total Household Cost'],
            'Value': [
                results['objective'],
                results['solve_time'],
                results['gap'],
                results['num_vars'],
                results['num_constrs'],
                sum(results['detail']['household_cost'].values())
            ]
        })
        summary_df.to_csv(f"../data/output/centralized/{filename_prefix}_summary.csv", 
                         index=False, encoding='utf-8')
        
        # 時系列データ
        time_series_data = []
        for t in range(1, self.T+1):
            row = {
                'Time': t,
                'AG_Buy': results['detail']['P_AG_buy'][t],
                'AG_Sell': results['detail']['P_AG_sell'][t]
            }
            
            for h in range(self.H):
                row[f'Voltage_H{h}'] = results['detail']['voltage'][h][t]
                row[f'Battery_H{h}'] = results['detail']['battery_soc'][h][t]
                row[f'HP_H{h}'] = results['detail']['hp_operation'][h][t]
            
            time_series_data.append(row)
        
        time_df = pd.DataFrame(time_series_data)
        time_df.to_csv(f"../data/output/centralized/{filename_prefix}_timeseries.csv", 
                      index=False, encoding='utf-8')
        
        logger.info(f"結果を../data/output/centralized/に保存しました")
    
    def plot_results(self, results: Dict):
        """結果の可視化"""
        if results['status'] not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return
        
        detail = results['detail']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. アグリゲータの買電・売電
        ax = axes[0, 0]
        times = list(range(1, self.T+1))
        buy_power = [detail['P_AG_buy'][t] for t in times]
        sell_power = [-detail['P_AG_sell'][t] for t in times]
        
        ax.bar(times, buy_power, label='Buy', alpha=0.7)
        ax.bar(times, sell_power, label='Sell', alpha=0.7)
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Power[kW]')
        ax.set_title('Power of Aggregator')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各需要家の電圧
        ax = axes[0, 1]
        for h in range(self.H):
            voltages = [detail['voltage'][h][t] for t in times]
            ax.plot(times, voltages, label=f'House{h}', linewidth=2)
        
        ax.axhline(y=self.V_UL, color='r', linestyle='--', label='Upper')
        ax.axhline(y=self.V_LL, color='b', linestyle='--', label='Lower')
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Voltage[V]')
        ax.set_title('Household Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 蓄電池SOC
        ax = axes[1, 0]
        for h in range(self.H):
            soc = [detail['battery_soc'][h][t] for t in times]
            ax.plot(times, soc, label=f'House{h}', linewidth=2)
        
        ax.axhline(y=self.N_B, color='r', linestyle='--', label='Upper')
        ax.axhline(y=0.2*self.N_B, color='b', linestyle='--', label='Lower')
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Charge Amount[kWh]')
        ax.set_title('SOC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. HP運転状態
        ax = axes[1, 1]
        for h in range(self.H):
            hp_status = [detail['hp_operation'][h][t] for t in times]
            ax.step(times, [s + h*1.1 for s in hp_status], 
                   label=f'House{h}', where='mid', linewidth=2)
        
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Operation Status (with offset)')
        ax.set_title('Heat Pump Operation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 需要家コスト
        ax = axes[2, 0]
        households = list(range(self.H))
        costs = [detail['household_cost'][h] for h in households]
        
        ax.bar(households, costs, alpha=0.7)
        ax.set_xlabel('Household Number')
        ax.set_ylabel('Cost [JPY]')
        ax.set_title('Electricity Cost by Household')
        ax.grid(True, alpha=0.3)
        
        # 6. 買電単価の推移
        ax = axes[2, 1]
        prices = [self.buy_price[t] for t in times]
        ax.plot(times, prices, linewidth=2)
        
        # DR期間をハイライト
        if self.DR_END <= self.T:
            ax.axvspan(self.DR_START, self.DR_END, alpha=0.3, color='red', label='DR期間')
        
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Electricity Price [JPY/kWh]')
        ax.set_title('Electricity Price Trend')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../data/output/centralized/centralized_results.png', dpi=150)
        plt.show()
        
        logger.info("グラフを保存しました")


def compare_with_gbd(centralized_results: Dict, gbd_results: Dict):
    """GBDとの比較"""
    print("\n" + "="*60)
    print("GBDとの比較")
    print("="*60)
    
    # 比較表の作成
    comparison_data = {
        'Method': ['Centralized', 'GBD'],
        'Objective': [
            centralized_results.get('objective', 'N/A'),
            gbd_results.get('upper_bound', 'N/A')
        ],
        'Solve Time [s]': [
            centralized_results.get('solve_time', 'N/A'),
            gbd_results.get('time', 'N/A')
        ],
        'Iterations': [
            'N/A',
            gbd_results.get('iterations', 'N/A')
        ],
        'Variables': [
            centralized_results.get('num_vars', 'N/A'),
            'Distributed'
        ],
        'Gap': [
            centralized_results.get('gap', 'N/A'),
            gbd_results.get('gap', 'N/A')
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 計算時間の比較
    if centralized_results.get('solve_time') and gbd_results.get('time'):
        speedup = centralized_results['solve_time'] / gbd_results['time']
        print(f"\n速度向上率: {speedup:.2f}x")
    
    # 最適性ギャップ
    if centralized_results.get('objective') and gbd_results.get('upper_bound'):
        gap = abs(centralized_results['objective'] - gbd_results['upper_bound'])
        gap_percent = gap / abs(centralized_results['objective']) * 100
        print(f"目的関数値の差: {gap:.2f} ({gap_percent:.2f}%)")


# ==================== メイン実行 ====================
def main():
    """メイン実行関数"""
    try:
        # 全体最適化の実行
        optimizer = CentralizedOptimizer()
        results = optimizer.optimize()
        
        # 結果のエクスポート
        optimizer.export_results(results)
        
        # 結果の可視化
        optimizer.plot_results(results)
        
        # サマリー表示
        print("\n" + "="*60)
        print("全体最適化完了")
        print("="*60)
        print(f"ステータス: {results['status']}")
        print(f"目的関数値: {results.get('objective', 'N/A')}")
        print(f"計算時間: {results['solve_time']:.2f}秒")
        print(f"変数数: {results['num_vars']}")
        print(f"制約数: {results['num_constrs']}")
        
        return results
        
    except Exception as e:
        logger.error(f"実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()