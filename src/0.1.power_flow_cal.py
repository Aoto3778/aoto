"""
電力潮流計算シミュレーションスクリプト
Power Flow Calculation Simulation Script

pandapowerを使用して配電系統の潮流計算を実行し、
48タイムスロット（30分間隔×24時間）のシミュレーション結果を出力します。
"""

import pandapower as pp
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """ネットワーク構成パラメータ"""
    # 電圧レベル [kV]
    hv_voltage: float = 6.6  # 高圧側
    lv_voltage: float = 0.1  # 低圧側
    
    # 変圧器パラメータ
    transformer_sn_mva: float = 0.1  # 定格容量 [MVA]
    transformer_vkr_percent: float = 1.0  # 短絡銅損 [%]
    transformer_vk_percent: float = 6.0  # 短絡インピーダンス [%]
    transformer_pfe_kw: float = 1.0  # 鉄損 [kW]
    transformer_i0_percent: float = 0.5  # 無負荷電流 [%]
    
    # 電線パラメータ (単位長さあたり)
    line_c_nf_per_km: float = 0  # 静電容量 [nF/km]
    line_r_ohm_per_km: float = 0.497  # 抵抗 [Ω/km]
    line_x_ohm_per_km: float = 0.00109  # リアクタンス [Ω/km]
    line_max_i_ka: float = 0.12  # 最大電流 [kA]
    
    # 電線長 [km]
    line_lengths: Dict[str, float] = None
    
    def __post_init__(self):
        if self.line_lengths is None:
            self.line_lengths = {
                "21": 0.011,  # Bus1 → Bus21
                "22": 0.015,  # Bus21 → Bus22
                "23": 0.016,  # Bus22 → Bus23
            }


@dataclass
class SimulationConfig:
    """シミュレーション設定パラメータ"""
    num_timeslots: int = 48  # タイムスロット数
    max_iteration: int = 50  # 潮流計算の最大反復回数
    tolerance_mva: float = 1e-5  # 収束判定閾値 [MVA]
    
    # 無効電力係数のランダム範囲
    reactive_power_min: float = 0.0
    reactive_power_max: float = 0.15
    
    # ファイルパス
    base_net_path: str = "../data/output/random_test/0.0.base_net.p"
    demand_data_dir: str = "../data/input/electric_demand_1y_30min"
    output_dir: str = "../data/output/random_test"


class PowerNetworkBuilder:
    """電力系統ネットワーク構築クラス"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.net = None
        self.buses = {}
        
    def build_base_network(self) -> pp.pandapowerNet:
        """基本ネットワークを構築"""
        logger.info("基本ネットワークの構築を開始")
        
        # 空のネットワークを作成
        self.net = pp.create_empty_network()
        
        # バスを作成
        self._create_buses()
        
        # 変圧器を作成
        self._create_transformer()
        
        # グリッド接続を作成
        self._create_external_grid()
        
        # 電線タイプを定義
        self._define_line_type()
        
        # 電線を作成
        self._create_lines()
        
        logger.info("基本ネットワークの構築完了")
        return self.net
    
    def _create_buses(self):
        """バス（母線）を作成"""
        # 高圧バス
        self.buses["hv"] = pp.create_bus(
            self.net, 
            vn_kv=self.config.hv_voltage, 
            name="HV Bus"
        )
        
        # 低圧バス群
        self.buses["lv1"] = pp.create_bus(
            self.net, 
            vn_kv=self.config.lv_voltage, 
            name="LV Bus 1"
        )
        
        for i in range(21, 24):
            self.buses[f"lv{i}"] = pp.create_bus(
                self.net, 
                vn_kv=self.config.lv_voltage, 
                name=f"LV Bus {i}"
            )
        
        logger.debug(f"作成されたバス数: {len(self.buses)}")
    
    def _create_transformer(self):
        """変圧器を作成"""
        pp.create_transformer_from_parameters(
            self.net,
            hv_bus=self.buses["hv"],
            lv_bus=self.buses["lv1"],
            sn_mva=self.config.transformer_sn_mva,
            vn_hv_kv=self.config.hv_voltage,
            vn_lv_kv=self.config.lv_voltage,
            vkr_percent=self.config.transformer_vkr_percent,
            vk_percent=self.config.transformer_vk_percent,
            pfe_kw=self.config.transformer_pfe_kw,
            i0_percent=self.config.transformer_i0_percent,
            name=f"{self.config.hv_voltage}/{self.config.lv_voltage}kV Transformer"
        )
        logger.debug("変圧器を作成")
    
    def _create_external_grid(self):
        """外部グリッド接続を作成"""
        pp.create_ext_grid(
            self.net, 
            bus=self.buses["hv"], 
            vm_pu=1.0, 
            name="Grid Connection"
        )
        logger.debug("外部グリッド接続を作成")
    
    def _define_line_type(self):
        """電線タイプを定義"""
        line_data = {
            "c_nf_per_km": self.config.line_c_nf_per_km,
            "r_ohm_per_km": self.config.line_r_ohm_per_km,
            "x_ohm_per_km": self.config.line_x_ohm_per_km,
            "max_i_ka": self.config.line_max_i_ka
        }
        pp.create_std_type(self.net, line_data, "line_type1")
        logger.debug("電線タイプを定義")
    
    def _create_lines(self):
        """低圧配電線を作成"""
        # Bus1 → Bus21
        pp.create_line(
            self.net, 
            from_bus=self.buses["lv1"], 
            to_bus=self.buses["lv21"], 
            length_km=self.config.line_lengths["21"], 
            std_type="line_type1", 
            name="LV Line 21"
        )
        
        # Bus21 → Bus22
        pp.create_line(
            self.net, 
            from_bus=self.buses["lv21"], 
            to_bus=self.buses["lv22"], 
            length_km=self.config.line_lengths["22"], 
            std_type="line_type1", 
            name="LV Line 22"
        )
        
        # Bus22 → Bus23
        pp.create_line(
            self.net, 
            from_bus=self.buses["lv22"], 
            to_bus=self.buses["lv23"], 
            length_km=self.config.line_lengths["23"], 
            std_type="line_type1", 
            name="LV Line 23"
        )
        
        logger.debug(f"作成された電線数: {len(self.net.line)}")


class DemandDataLoader:
    """需要データ読み込みクラス"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.demand_data = {}
    
    def load_demand_data(self, house_numbers: List[int]) -> Dict[int, pd.DataFrame]:
        """複数世帯の需要データを読み込み"""
        logger.info(f"{len(house_numbers)}世帯分の需要データを読み込み中")
        
        for house_num in house_numbers:
            file_name = f"electric_demand_1y_30min_{house_num:02d}.csv"
            file_path = self.data_dir / file_name
            
            try:
                self.demand_data[house_num] = pd.read_csv(
                    file_path, 
                    encoding="shift_jis"
                )
                logger.debug(f"世帯{house_num}のデータを読み込み完了")
            except FileNotFoundError:
                logger.error(f"ファイルが見つかりません: {file_path}")
                raise
            except Exception as e:
                logger.error(f"データ読み込みエラー: {e}")
                raise
        
        logger.info("全需要データの読み込み完了")
        return self.demand_data


class PowerFlowSimulator:
    """電力潮流計算シミュレータ"""
    
    def __init__(
        self, 
        network_config: NetworkConfig, 
        simulation_config: SimulationConfig
    ):
        self.net_config = network_config
        self.sim_config = simulation_config
        self.base_network = None
        self.demand_loader = DemandDataLoader(simulation_config.demand_data_dir)
        
    def initialize(self):
        """シミュレーション環境を初期化"""
        # 基本ネットワークを構築
        builder = PowerNetworkBuilder(self.net_config)
        self.base_network = builder.build_base_network()
        
        # 基本ネットワークを保存
        self._save_base_network()
        
        # 需要データを読み込み
        self.demand_data = self.demand_loader.load_demand_data([1, 2, 3])
        
        logger.info("シミュレーション環境の初期化完了")
    
    def _save_base_network(self):
        """基本ネットワークを保存"""
        output_path = Path(self.sim_config.base_net_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pp.to_pickle(self.base_network, str(output_path))
        logger.info(f"基本ネットワークを保存: {output_path}")
    
    def run_simulation(self):
        """全タイムスロットのシミュレーションを実行"""
        logger.info(f"{self.sim_config.num_timeslots}タイムスロットのシミュレーション開始")
        
        # 出力ディレクトリを作成
        output_dir = Path(self.sim_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_slots = 0
        failed_slots = []
        
        for timeslot in range(self.sim_config.num_timeslots):
            try:
                self._simulate_timeslot(timeslot)
                successful_slots += 1
                
                # 進捗表示（10スロットごと）
                if (timeslot + 1) % 10 == 0:
                    logger.info(f"進捗: {timeslot + 1}/{self.sim_config.num_timeslots} タイムスロット完了")
                    
            except Exception as e:
                logger.error(f"タイムスロット{timeslot}でエラー: {e}")
                failed_slots.append(timeslot)
        
        # 結果サマリー
        logger.info("=" * 50)
        logger.info(f"シミュレーション完了")
        logger.info(f"成功: {successful_slots}/{self.sim_config.num_timeslots} タイムスロット")
        if failed_slots:
            logger.warning(f"失敗したタイムスロット: {failed_slots}")
    
    def _simulate_timeslot(self, timeslot: int):
        """単一タイムスロットのシミュレーション"""
        # 基本ネットワークを読み込み（各タイムスロットで新規作成）
        simulation_net = pp.from_pickle(self.sim_config.base_net_path)
        
        # 無効電力係数をランダムに生成
        reactive_factors = self._generate_reactive_factors()
        
        # 各世帯の負荷を作成
        self._create_loads(simulation_net, timeslot, reactive_factors)
        
        # 潮流計算を実行
        self._run_power_flow(simulation_net)
        
        # 結果をExcelファイルに保存
        self._save_results(simulation_net, timeslot)
    
    def _generate_reactive_factors(self) -> Dict[int, float]:
        """無効電力係数をランダムに生成"""
        return {
            house: random.uniform(
                self.sim_config.reactive_power_min,
                self.sim_config.reactive_power_max
            )
            for house in [1, 2, 3]
        }
    
    def _create_loads(
        self, 
        net: pp.pandapowerNet, 
        timeslot: int, 
        reactive_factors: Dict[int, float]
    ):
        """負荷を作成"""
        # 各世帯の負荷を作成
        load_configs = [
            (21, 1, "Load 1"),  # Bus21に世帯1
            (22, 2, "Load 2"),  # Bus22に世帯2
            (23, 3, "Load 3"),  # Bus23に世帯3
        ]
        
        for bus_num, house_num, load_name in load_configs:
            # 有効電力 [MW] (W → MW変換: ×0.001)
            p_mw = self.demand_data[house_num].iat[timeslot, self.demand_data[house_num].columns.get_loc('合計[kW*30min]')] * 0.001
            
            # 無効電力 [MVar] (有効電力 × ランダム係数)
            q_mvar = p_mw * reactive_factors[house_num]
            
            # バスIDを取得
            bus_id = net.bus[net.bus.name == f"LV Bus {bus_num}"].index[0]
            
            # 負荷を作成
            pp.create_load(
                net, 
                bus=bus_id, 
                p_mw=p_mw, 
                q_mvar=q_mvar, 
                name=load_name, 
                in_service=True
            )
            
            logger.debug(
                f"{load_name}: P={p_mw:.4f}MW, Q={q_mvar:.4f}MVar, "
                f"係数={reactive_factors[house_num]:.3f}"
            )
    
    def _run_power_flow(self, net: pp.pandapowerNet):
        """潮流計算を実行"""
        try:
            pp.runpp(
                net, 
                max_iteration=self.sim_config.max_iteration, 
                tolerance_mva=self.sim_config.tolerance_mva
            )
            
            # 収束確認
            if not net.converged:
                logger.warning("潮流計算が収束しませんでした")
                
        except Exception as e:
            logger.error(f"潮流計算エラー: {e}")
            raise
    
    def _save_results(self, net: pp.pandapowerNet, timeslot: int):
        """結果をExcelファイルに保存"""
        output_path = Path(self.sim_config.output_dir) / f"0.1.timeslot_{timeslot}.xlsx"
        
        pp.to_excel(
            net, 
            str(output_path), 
            include_empty_tables=False, 
            include_results=True
        )
        
        logger.debug(f"結果を保存: {output_path}")


def main():
    """メイン実行関数"""
    # 設定を定義
    network_config = NetworkConfig()
    simulation_config = SimulationConfig()
    
    # シミュレータを作成
    simulator = PowerFlowSimulator(network_config, simulation_config)
    
    # 初期化とシミュレーション実行
    try:
        simulator.initialize()
        simulator.run_simulation()
        
        logger.info("全処理が正常に完了しました")
        
    except Exception as e:
        logger.error(f"シミュレーション中に重大なエラーが発生: {e}")
        raise


if __name__ == "__main__":
    main()