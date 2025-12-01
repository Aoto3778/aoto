"""
電力潮流計算結果整形スクリプト
Power Flow Calculation Results Formatter

このスクリプトは、48時間分の電力潮流計算結果（Excelファイル）を読み込み、
主要な電力パラメータを抽出して単一のCSVファイルに整形・出力します。
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def create_empty_dataframe(num_timeslots: int) -> pd.DataFrame:
    """
    結果格納用の空のDataFrameを作成
    
    Args:
        num_timeslots: タイムスロット数（時間数）
    
    Returns:
        初期化されたDataFrame
    """
    # カラム定義：各送電線（0,1,2）の電力パラメータ
    column_names = [
        'DATE',  # タイムスロット番号
        # 有効電力 [MW] - 送電線始点側
        'p_from_mw(0)', 'p_from_mw(1)', 'p_from_mw(2)',
        # 無効電力 [MVar] - 送電線始点側  
        'q_from_mvar(0)', 'q_from_mvar(1)', 'q_from_mvar(2)',
        # 電圧変動量 [p.u.] - 始点と終点の電圧差
        'vm_delta_pu(0)', 'vm_delta_pu(1)', 'vm_delta_pu(2)',
    ]
    
    # インデックスは1から開始（時間スロット番号）
    row_indices = np.arange(1, num_timeslots + 1)
    
    # ゼロで初期化されたDataFrameを作成
    df = pd.DataFrame(
        data=0, 
        index=row_indices, 
        columns=column_names
    )
    
    print(f"作成されたDataFrameのカラム数: {len(column_names)}")
    return df


def extract_power_flow_data(
    excel_path: str, 
    timeslot: int, 
    result_df: pd.DataFrame, 
    row_index: int
) -> None:
    """
    単一タイムスロットのExcelファイルから電力潮流データを抽出
    
    Args:
        excel_path: 読み込むExcelファイルのパス
        timeslot: タイムスロット番号（DATE用）
        result_df: 結果を格納するDataFrame
        row_index: データを挿入する行番号（0ベース）
    """
    # Excelファイルから送電線データを読み込み
    line_data = pd.read_excel(excel_path, sheet_name='res_line')
    
    # タイムスロット番号を記録
    result_df.iat[row_index, 0] = float(timeslot)
    
    # 3つの送電線（インデックス0, 1, 2）のデータを抽出
    for line_idx in range(3):
        # 有効電力 p_from_mw（カラムインデックス1）
        result_df.iat[row_index, 1 + line_idx] = float(
            line_data.iat[line_idx, 1]
        )
        
        # 無効電力 q_from_mvar（カラムインデックス2）
        result_df.iat[row_index, 4 + line_idx] = float(
            line_data.iat[line_idx, 2]
        )
        
        # 電圧変動量 vm_delta_pu = vm_from_pu - vm_to_pu
        # vm_from_pu（カラム10）とvm_to_pu（カラム12）の差分
        voltage_delta = float(
            line_data.iat[line_idx, 10] - line_data.iat[line_idx, 12]
        )
        result_df.iat[row_index, 7 + line_idx] = voltage_delta


def process_all_timeslots(
    num_timeslots: int, 
    input_dir: str, 
    output_path: str
) -> pd.DataFrame:
    """
    全タイムスロットのデータを処理し、CSVファイルに出力
    
    Args:
        num_timeslots: 処理するタイムスロット数
        input_dir: 入力Excelファイルのディレクトリパス
        output_path: 出力CSVファイルのパス
    
    Returns:
        処理済みのDataFrame
    """
    # 結果格納用DataFrameを初期化
    result_df = create_empty_dataframe(num_timeslots)
    
    # 各タイムスロットのデータを順次処理
    for timeslot in range(num_timeslots):
        # 入力ファイルパスを構築
        excel_path = f"{input_dir}/0.1.timeslot_{timeslot}.xlsx"
        
        # データ抽出と格納
        extract_power_flow_data(
            excel_path=excel_path,
            timeslot=timeslot,  # DATEカラムには0から47の値が入る
            result_df=result_df,
            row_index=timeslot  # 行インデックスも0ベース
        )
        
        # 進捗表示（10タイムスロットごと）
        if (timeslot + 1) % 10 == 0:
            print(f"処理済み: {timeslot + 1}/{num_timeslots} タイムスロット")
    
    # CSVファイルとして出力
    # Shift-JISエンコーディングを使用（日本語Windows環境対応）
    result_df.to_csv(
        output_path, 
        encoding="shift_jis", 
        index=False
    )
    print(f"\n処理完了: {output_path} に出力されました")
    
    return result_df


def main():
    """
    メイン処理関数
    """
    # パラメータ設定
    NUM_TIMESLOTS = 48  # 48スロット分のデータ（1日分）
    INPUT_DIR = "../data/output/random_test"
    OUTPUT_PATH = "../data/output/random_test/1.data_shaping_1day.csv"
    
    # データ処理実行
    processed_data = process_all_timeslots(
        num_timeslots=NUM_TIMESLOTS,
        input_dir=INPUT_DIR,
        output_path=OUTPUT_PATH
    )
    
    # 結果サマリー表示
    print(f"\n=== 処理結果サマリー ===")
    print(f"処理したタイムスロット数: {NUM_TIMESLOTS}")
    print(f"出力データ形状: {processed_data.shape}")
    print(f"カラム一覧: {list(processed_data.columns)}")
    
    # データの簡易統計を表示（オプション）
    print(f"\n=== 有効電力の統計情報 ===")
    p_columns = [col for col in processed_data.columns if 'p_from_mw' in col]
    print(processed_data[p_columns].describe().round(2))


if __name__ == "__main__":
    main()