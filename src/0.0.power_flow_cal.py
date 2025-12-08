# 0.power_flow_cal.py は pandapowerを使用して電力潮流計算を行うスクリプトです。
import pandapower as pp
import pandas as pd
import random

# 空のネットワークを作成
empty_net = pp.create_empty_network()

# 高電圧バスと低電圧バスを作成
bus_hv = pp.create_bus(empty_net, vn_kv=6.6, name="HV Bus")
bus_1 = pp.create_bus(empty_net, vn_kv=0.1, name="LV Bus 1")
bus_lv21 = pp.create_bus(empty_net, vn_kv=0.1, name="LV Bus 21")
bus_lv22 = pp.create_bus(empty_net, vn_kv=0.1, name="LV Bus 22")
bus_lv23 = pp.create_bus(empty_net, vn_kv=0.1, name="LV Bus 23")

# 変圧器の作成
pp.create_transformer_from_parameters(
    empty_net,
    #以下変圧器の接続バス
    bus_hv,
    bus_1,
    #以下変圧器パラメータ
    sn_mva=0.1,
    vn_hv_kv=6.6,
    vn_lv_kv=0.1,
    vkr_percent=1.0,
    vk_percent=6.0,
    pfe_kw=1.0,
    i0_percent=0.5,
    name="6.6/0.1kV Transformer"
    )

# グリッド接続と電圧パラメータ
pp.create_ext_grid(empty_net, bus=bus_hv, vm_pu=1.0, name="Grid Connection")

# 電線の単位距離当たりのパラメータを作成
line_data1 = {"c_nf_per_km": 0, "r_ohm_per_km": 0.497, "x_ohm_per_km": 0.00109, "max_i_ka": 0.12}
pp.create_std_type(empty_net, line_data1, "line_type1")

# 低圧バス接続
pp.create_line(empty_net, from_bus=bus_1, to_bus=bus_lv21, length_km=0.011, std_type="line_type1", name="LV Line 21")
pp.create_line(empty_net, from_bus=bus_lv21, to_bus=bus_lv22, length_km=0.015, std_type="line_type1", name="LV Line 22")
pp.create_line(empty_net, from_bus=bus_lv22, to_bus=bus_lv23, length_km=0.016, std_type="line_type1", name="LV Line 23")

# ベースネットワーク保存
pp.to_pickle(empty_net, "../data/output/random_test/base_net.p")

# CSVデータ読み込み
demand_house1 = pd.read_csv("../data/input/electric_demand_1y_30min/electric_demand_1y_30min_01.csv", encoding="shift_jis")
demand_house2 = pd.read_csv("../data/input/electric_demand_1y_30min/electric_demand_1y_30min_02.csv", encoding="shift_jis")
demand_house3 = pd.read_csv("../data/input/electric_demand_1y_30min/electric_demand_1y_30min_03.csv", encoding="shift_jis")

# メインループ
for i in range(0, 48):
    simulation_net = pp.from_pickle("../data/output/random_test/base_net.p")

    # ランダム値
    vv1=random.uniform(0,0.15)
    vv2=random.uniform(0,0.15)
    vv3=random.uniform(0,0.15)

    pp.create_load(simulation_net, bus_lv21, p_mw=demand_house1.iat[i, demand_house1.columns.get_loc('合計[kW*30min]')]*0.001, q_mvar=demand_house1.iat[i, demand_house1.columns.get_loc('合計[kW*30min]')]*0.001*vv1, name="Load 1", in_service=True)
    pp.create_load(simulation_net, bus_lv22, p_mw=demand_house2.iat[i, demand_house2.columns.get_loc('合計[kW*30min]')]*0.001, q_mvar=demand_house2.iat[i, demand_house2.columns.get_loc('合計[kW*30min]')]*0.001*vv2, name="Load 2", in_service=True)
    pp.create_load(simulation_net, bus_lv23, p_mw=demand_house3.iat[i, demand_house3.columns.get_loc('合計[kW*30min]')]*0.001, q_mvar=demand_house3.iat[i, demand_house3.columns.get_loc('合計[kW*30min]')]*0.001*vv3, name="Load 3", in_service=True)

    # 潮流計算（収束条件調整）
    pp.runpp(simulation_net, max_iteration=50, tolerance_mva=1e-5)
    # Excel出力
    pp.to_excel(simulation_net, f"../data/output/random_test/timeslot_{i}.xlsx", include_empty_tables=False, include_results=True)