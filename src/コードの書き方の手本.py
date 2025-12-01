# main.py
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.visualization import ResultVisualizer
from utils.wandb_logger import WandBLogger

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 自作モジュールのインポート
from models.aggregator import AggregatorModel
from models.consumer import ConsumerModel
from models.network import NetworkModel


class InitialData:
    def __init__(self):
        self.T = 48  # タイムスロット数（24時間 / 30粒度 = 48スロット）
        self.H = 3   # 需要家数

        # 時間インデックス（30分間隔）
        self.time_index = pd.date_range(
            start='2024-01-01 00:00',
            periods=self.T,
            freq='30min'
        )

        # PV発電量データ [kW] - 既知
        self.pv_generation = self.fetch_pv_profile()

        # 電力需要データ [kW] - 既知
        self.demand = self.fetch_demand_profile()

        # 温度データ [℃] - 既知
        self.temperature = self.fetch_temparature_profile()

        # 水温データ [℃] - 既知
        self.water_temperature = self.fetch_water_temp_profile()

        # 給湯需要データ [MJ] - 既知
        self.heat_demand = self.fetch_heat_demand_profile()


        # DDR期間の設定（例：13時～16時 30粒度）
        self.DDR = [26, 27, 28, 29, 30, 31]  # 0-indexedで13-16時 30粒度

        # 配電網パラメータ
        self.network_params = {
            'voltage_nominal': 101,  # [V]
            'voltage_min': 95,       # [V]
            'voltage_max': 107,      # [V]
            'branch_rating': 1,      #! [kVA]
        }

    def fetch_pv_profile(self):
        """PV発電プロファイルの取得（CSVファイルから読み込み）"""
        pv_profile = np.zeros((self.H, self.T))

        # CSVファイルから東京のPV出力データを読み込み
        csv_path = "data/input/pv_output_30min.csv"
        df = pd.read_csv(csv_path)

        # Tokyo列のデータを取得（最初の48データポイント = 1日分）
        tokyo_pv_data = df['Tokyo'].values[:self.T]

        for h in range(self.H):
            for t in range(self.T):
                # 東京のPV出力データをそのまま使用
                pv_profile[h, t] = tokyo_pv_data[t]

        return pv_profile

    def fetch_demand_profile(self):
        """需要プロファイルの取得（CSVファイルから読み込み）"""
        demand_profile = np.zeros((self.H, self.T))

        for h in range(self.H):
            # CSVファイルのパスを構築
            csv_path = f"data/input/electric_demand_1y_30min/electric_demand_1y_30min_{h+1:02d}.csv"

            # CSVファイルを読み込み
            df = pd.read_csv(csv_path, encoding='shift-jis')

            # '合計[kW*30min]'列から需要データを取得
            demand_data = df['合計[kW*30min]'].values

            # 1日分のデータを抽出（最初の24時間 = 48データポイント）
            for t in range(self.T):
                demand_profile[h, t] = demand_data[t]

        return demand_profile

    def fetch_temparature_profile(self):
        """温度プロファイルの取得（CSVファイルから読み込み）"""
        temperature_profile = np.zeros((self.H, self.T))

        for h in range(self.H):
            # CSVファイルのパスを構築
            csv_path = f"data/input/temperature1y.csv"

            # CSVファイルを読み込み
            df = pd.read_csv(csv_path, encoding='shift-jis')

            # '気温[℃]'列から温度データを取得
            temp_data = df['気温[℃]'].values

            # 1日分のデータを抽出（最初の24時間 = 48データポイント）
            for t in range(self.T):
                temperature_profile[h, t] = temp_data[t]

        return temperature_profile

    def fetch_water_temp_profile(self):
        """水温プロファイルの取得（CSVファイルから読み込み）"""
        water_temp_profile = np.zeros((self.H, self.T))

        for h in range(self.H):
            # CSVファイルのパスを構築
            csv_path = f"data/input/water_temperature_30min.csv"

            # CSVファイルを読み込み
            df = pd.read_csv(csv_path, encoding='shift-jis')

            # '水温[℃]'列から水温データを取得
            water_temp_data = df['水温[℃]'].values

            # 1日分のデータを抽出（最初の24時間 = 48データポイント）
            for t in range(self.T):
                water_temp_profile[h, t] = water_temp_data[t]

        return water_temp_profile

    def fetch_heat_demand_profile(self):
        """給湯需要プロファイルの取得（CSVファイルから読み込み）"""
        heat_demand_profile = np.zeros((self.H, self.T))

        for h in range(self.H):
            # CSVファイルのパスを構築
            csv_path = f"data/input/heat_demand1y.csv"

            # CSVファイルを読み込み
            df = pd.read_csv(csv_path, encoding='shift-jis')

            # '給湯需要[MJ]'列から熱需要データを取得
            heat_demand_data = df['給湯需要[MJ]'].values

            # 1日分のデータを抽出（最初の24時間 = 48データポイント）
            for t in range(self.T):
                heat_demand_profile[h, t] = heat_demand_data[t]

        return heat_demand_profile

def save_iteration_results(k, agg_model, consumer_results, violations, data):
    """反復結果のCSV保存"""

    # 出力ディレクトリの作成
    output_dir = f"data/output/iteration_{k}"
    os.makedirs(output_dir, exist_ok=True)

    # アグリゲータの結果
    agg_df = pd.DataFrame({
        't': list(range(data.T)),
        'time': data.time_index.strftime('%H:%M'),
        'P_PG': [agg_model.P_PG_values.get(t, 0) for t in range(data.T)]
    })
    agg_df.to_csv(f"{output_dir}/aggregator_results.csv", index=False)

    # 需要家の結果
    for h, result in enumerate(consumer_results):
        cons_df = pd.DataFrame({
            't': list(range(data.T)),
            'time': data.time_index.strftime('%H:%M'),
            'P_buy': result['P_buy'],
            'P_sell': result['P_sell'],
            'P_charge': result['P_charge'],
            'P_discharge': result['P_discharge'],
            'SOC': result['SOC'],
            'P_hp': result['P_hp'],
            'H_hs': result['H_hs'],
            'H_loss': result['H_loss'],
            'H_storage': result['H_storage'],
            'delta_hp': result['delta_hp'],
            'start_hp': result['start_hp'],
            'stop_hp': result['stop_hp'],
            'cop': result['cop'],
            'cost': result['cost'],
        })
        cons_df.to_csv(f"{output_dir}/consumer_{h}_results.csv", index=False)

    # 違反量の結果
    viol_df = pd.DataFrame(violations)
    viol_df['t'] = list(range(data.T))
    viol_df['time'] = data.time_index.strftime('%H:%M')
    viol_df = viol_df[['t', 'time'] + [col for col in viol_df.columns if col not in ['t', 'time']]]
    viol_df.to_csv(f"{output_dir}/violations.csv", index=False)

    print(f"Results saved to {output_dir}")


def calculate_price_sensitivity(data, h, t, current_prices, consumer_results, epsilon=0.5):
    """価格感度の計算"""
    # 現在の違反量
    network = NetworkModel(data)
    current_violations = []
    for time in range(data.T):
        pf_results = network.run_power_flow(consumer_results, time)
        violations = network.calculate_violations(pf_results)
        current_violations.append(violations['L'])
    L_current = sum(current_violations)

    # 価格を摂動させて再計算
    perturbed_prices = {
        'buy': current_prices['buy'].copy(),
        'sell': current_prices['sell'].copy()
    }
    perturbed_prices['buy'][(h,t)] += epsilon
    perturbed_prices['sell'][(h,t)] = perturbed_prices['buy'][(h,t)] * 0.7

    # 摂動後の需要家問題を解く
    h_prices_perturbed = {
        'buy': [perturbed_prices['buy'][(h,time)] for time in range(data.T)],
        'sell': [perturbed_prices['sell'][(h,time)] for time in range(data.T)]
    }

    cons_model = ConsumerModel(data, h, 0)
    cons_model.build_model(h_prices_perturbed)
    result_perturbed = cons_model.solve()

    if result_perturbed is None:
        # 摂動後の問題が解けない場合はデフォルト値を返す
        return 0.1

    # 摂動後の違反量計算
    consumer_results_perturbed = consumer_results.copy()
    consumer_results_perturbed[h] = result_perturbed

    perturbed_violations = []
    for time in range(data.T):
        pf_results = network.run_power_flow(consumer_results_perturbed, time)
        violations = network.calculate_violations(pf_results)
        perturbed_violations.append(violations['L'])
    L_perturbed = sum(perturbed_violations)

    # 感度 = (L_perturbed - L_current) / epsilon
    sensitivity = (L_perturbed - L_current) / epsilon

    return sensitivity


def main(use_wandb=True):
    data = InitialData()
    data.beta_1 = 0.5
    data.beta_2 = 0.5

    # W&Bの初期化（オプション）
    logger = None
    if use_wandb:
        try:
            config = {
                'beta_1': data.beta_1,
                'beta_2': data.beta_2,
                'max_iter': 100,
                'num_consumers': data.H,
                'time_slots': data.T,
                'time_resolution': '30min',
                'DDR_hours': len(data.DDR) * 0.5  # 30分単位なので0.5を掛ける
            }
            logger = WandBLogger(project_name="power_grid_optimization", config=config)
            print("W&B logging enabled")
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            print("Continuing without W&B logging")
            use_wandb = False

    # 反復パラメータ
    max_iter = 20
    convergence_criteria = {
        'obj_tol': 1e-4,
        'price_tol': 1e-3,
        'theta_tol': 1e-6
    }

    # 履歴の保存
    history = {
        'objective': [],
        'prices': [],
        'violations': [],
        'theta': []
    }

    # 初期化
    k = 0
    converged = False
    price_sensitivity = None
    violation_total = None

    while k < max_iter and not converged:
        print(f"\n=== Iteration {k} ===")

        # Step 1: アグリゲータのマスター問題
        agg_model = AggregatorModel(data, k)
        agg_model.build_model(price_sensitivity, violation_total)

        if not agg_model.solve():
            print("Aggregator problem infeasible!")
            break

        print(f"Aggregator objective: {agg_model.model.objVal:.4f}")

        # 価格の取得
        current_prices = {
            'buy': {},
            'sell': {}
        }
        for h in range(data.H):
            for t in range(data.T):
                #! 簡易的に売電価格は買電価格の70%
                current_prices['buy'][(h,t)] = agg_model.C_p_values[(h,t)]
                current_prices['sell'][(h,t)] = agg_model.C_p_values[(h,t)] * 0.7

        # Step 2: 各需要家のサブ問題
        consumer_results = []
        total_consumer_cost = 0
        for h in range(data.H):
            # 需要家hの価格を抽出
            h_prices = {
                'buy': [current_prices['buy'][(h,t)] for t in range(data.T)],
                'sell': [current_prices['sell'][(h,t)] for t in range(data.T)]
            }

            cons_model = ConsumerModel(data, h, k)
            cons_model.build_model(h_prices)
            result = cons_model.solve()

            if result is None:
                print(f"Consumer {h} problem infeasible!")
                break

            consumer_results.append(result)
            total_consumer_cost += result['cost']
            print(f"Consumer {h} cost: {result['cost']:.4f}")

        # Step 3: 潮流計算と違反評価
        network = NetworkModel(data)
        violations_by_time = []

        for t in range(data.T):
            pf_results = network.run_power_flow(consumer_results, t)
            violations = network.calculate_violations(pf_results)
            violations_by_time.append(violations)

            # 詳細結果の保存
            network.save_power_flow_results(pf_results, k, t, "data/output")

        # 違反量の集計
        violation_total = sum(v['L'] for v in violations_by_time)
        print(f"Total violation: {violation_total:.4f}")

        # Step 4: 感度計算（数値微分）
        if k > 0:
            print("\nCalculating price sensitivities...")
            new_sensitivity = np.zeros((data.H, data.T))

            for h in range(data.H):
                for t in range(data.T):
                    new_sensitivity[h,t] = calculate_price_sensitivity(
                        data, h, t, current_prices, consumer_results
                    )
                    print(f"  Sensitivity[{h},{t}] = {new_sensitivity[h,t]:.4f}")

            price_sensitivity = new_sensitivity
            print(f"Sensitivity norm (λ^k): {np.linalg.norm(price_sensitivity):.4f}")

        # 履歴の更新
        history['objective'].append(agg_model.model.objVal)
        history['prices'].append(current_prices.copy())
        history['violations'].append(violation_total)

        # thetaの値を保存（k > 0の場合のみ存在）
        if hasattr(agg_model, 'theta') and k > 0:
            theta_value = agg_model.theta.X
            history['theta'].append(theta_value)
            print(f"Theta: {theta_value:.6f}")
        else:
            history['theta'].append(0)

        # W&Bへのログ記録
        if use_wandb and logger:
            metrics = {
                'objective': agg_model.model.objVal,
                'total_violation': violation_total,
                'voltage_violation': sum(v['V'] for v in violations_by_time),
                'flow_violation': sum(v['F'] for v in violations_by_time),
                'theta': history['theta'][-1] if history['theta'] else 0,
                'avg_buy_price': np.mean([
                    current_prices['buy'][(h,t)]
                    for h in range(data.H)
                    for t in range(data.T)
                ]),
                'total_P_PG_DDR': sum(
                    agg_model.P_PG_values.get(t, 0)
                    for t in data.DDR
                )
            }
            logger.log_iteration(k, metrics)

        # # 収束判定
        # if k > 0:
        #     # 1. 目的関数の改善
        #     if abs(history['objective'][k-1]) > 1e-10:  # ゼロ除算を防ぐ
        #         obj_change = abs(
        #             history['objective'][k] - history['objective'][k-1]
        #         ) / abs(history['objective'][k-1])
        #     else:
        #         obj_change = abs(history['objective'][k] - history['objective'][k-1])

        #     print(f"Objective change: {obj_change:.6f}")

        #     if obj_change < convergence_criteria['obj_tol']:
        #         converged = True
        #         print(f"Converged: Objective improvement < {convergence_criteria['obj_tol']}")

            # # 2. 価格の変化
            # price_change = 0
            # for h in range(data.H):
            #     for t in range(data.T):
            #         price_change += abs(
            #             current_prices['buy'][(h,t)] -
            #             history['prices'][k-1]['buy'][(h,t)]
            #         )
            # price_change /= (data.H * data.T)

            # print(f"Average price change: {price_change:.6f}")

            # if price_change < convergence_criteria['price_tol']:
            #     converged = True
            #     print(f"Converged: Price change < {convergence_criteria['price_tol']}")

            # # 3. θの大きさ（k > 1から確認）
            # if k > 1 and history['theta'][k] < convergence_criteria['theta_tol']:
            #     converged = True
            #     print(f"Converged: Theta < {convergence_criteria['theta_tol']}")

        # 4. 最大反復回数
        if k >= max_iter - 1:
            converged = True
            print(f"Reached maximum iterations: {max_iter}")

        # 結果の出力（CSV）
        save_iteration_results(k, agg_model, consumer_results, violations_by_time, data)

        # 結果のサマリー表示
        print(f"\n--- Iteration {k} Summary ---")
        print(f"Total P_PG in DDR: {sum(agg_model.P_PG_values.get(t, 0) for t in data.DDR):.4f} kW")
        print(f"Average buy price: {np.mean([current_prices['buy'][(h,t)] for h in range(data.H) for t in range(data.T)]):.2f} JPY/kWh")
        print(f"Voltage violations: {sum(v['V'] for v in violations_by_time):.4f} V")
        print(f"Flow violations: {sum(v['F'] for v in violations_by_time):.4f} kVA")

        k += 1

    print(f"\n=== Optimization completed in {k} iterations ===")

    # W&Bに最終結果を記録
    if use_wandb and logger:
        logger.log_final_results(history)
        logger.finish()
        print("Results logged to W&B")

    # 最終結果の表示
    if len(history['objective']) > 0:
        print(f"\nFinal Results:")
        print(f"Final objective value: {history['objective'][-1]:.4f}")
        print(f"Final total violation: {history['violations'][-1]:.4f}")
        if len(history['theta']) > 1:
            print(f"Final theta: {history['theta'][-1]:.6f}")

    return history, data

if __name__ == "__main__":
    # コマンドライン引数の処理
    import argparse
    parser = argparse.ArgumentParser(description='Power Grid Optimization')
    parser.add_argument('--no-wandb', action='store_true',  # 逆にする
                       help='Disable W&B logging')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    args = parser.parse_args()

    # メイン実行（デフォルトでW&B有効）
    results, data = main(use_wandb=not args.no_wandb)  # 論理を反転
    visualizer = ResultVisualizer(data=data, history=results)

    # 配電網の可視化
    network = NetworkModel(InitialData())
    network.visualize_network(save_path="data/output/figures/network_topology.png")

    # 最終反復の結果を可視化
    final_iteration = len(results['objective']) - 1
    visualizer.plot_power_balance(final_iteration)
    visualizer.plot_voltage_profile(final_iteration)
    visualizer.plot_battery_operation(final_iteration)
    visualizer.plot_price_evolution()
    visualizer.create_summary_report(final_iteration)

    # 収束履歴のプロット（既存のコード）
    if len(results['objective']) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 目的関数の推移
        axes[0, 0].plot(results['objective'])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Aggregator Objective Function')
        axes[0, 0].grid(True)

        # 違反量の推移
        axes[0, 1].plot(results['violations'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Total Violation')
        axes[0, 1].set_title('Network Constraint Violations')
        axes[0, 1].grid(True)

        # Thetaの推移
        if len(results['theta']) > 1:
            axes[1, 0].plot(results['theta'][1:])  # 最初の0を除く
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Theta')
            axes[1, 0].set_title('Lower Bound (Theta)')
            axes[1, 0].grid(True)

        # 価格の推移（平均）
        avg_prices = []
        for prices in results['prices']:
            avg_price = np.mean([
                prices['buy'][(h,t)]
                for h in range(3)
                for t in range(48)
            ])
            avg_prices.append(avg_price)

        axes[1, 1].plot(avg_prices)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Average Buy Price [JPY/kWh]')
        axes[1, 1].set_title('Average Electricity Price')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('data/output/convergence_history.png')
        print("\nConvergence history saved to data/output/convergence_history.png")
