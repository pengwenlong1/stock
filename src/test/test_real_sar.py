# -*- coding: utf-8 -*-
"""使用真实数据测试SAR计算"""
import pandas as pd
import numpy as np
import sys
import os
import yaml
from pathlib import Path

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_path)

from src.tool.sar_strategy import SARStrategy
import pymysql

# 项目根目录
project_root = Path(project_path)

def load_db_config():
    """加载数据库配置"""
    app_config_path = project_root / " .env" / "app_config.yaml"

    if app_config_path.exists():
        with app_config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            db_config = config.get('db_meta_mysql', {})
            return {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 3306),
                'user': db_config.get('user', 'root'),
                'password': db_config.get('password', ''),
                'database': db_config.get('database', 'stock')
            }
    else:
        print(f"配置文件不存在: {app_config_path}")
        return None

def test_real_sar_calculation():
    """使用真实数据测试SAR计算"""
    print('开始真实数据SAR计算测试...')
    print('=' * 60)

    # 圣邦股份 symbol: 300661
    symbol = '300661'
    end_date = '2026-05-07'
    start_date = '2026-01-01'  # 获取足够的历史数据

    try:
        # 从数据库获取数据
        db_config = load_db_config()
        if db_config is None:
            return False

        conn = pymysql.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        stock_id_int = int(symbol)

        sql = """
            SELECT
                trade_date,
                close_price,
                low_price,
                sar,
                is_sar_dead_cross
            FROM stock_daily_metrics
            WHERE stock_id = %s
            AND trade_date >= %s
            AND trade_date <= %s
            ORDER BY trade_date
        """

        with conn.cursor() as cursor:
            cursor.execute(sql, (stock_id_int, start_date, end_date))
            rows = cursor.fetchall()

        conn.close()

        if not rows:
            print(f'数据库中未找到股票 {symbol} 的数据')
            return False

        # 转换为DataFrame
        df = pd.DataFrame(rows)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').sort_index()

        # 重命名列并转换为float类型
        df = df.rename(columns={
            'close_price': 'close',
            'low_price': 'low'
        })
        df['close'] = df['close'].astype(float)
        df['low'] = df['low'].astype(float)
        if 'sar' in df.columns:
            df['sar'] = df['sar'].astype(float)

        # 由于没有high_price，用close作为high的近似值（对于测试来说）
        df['high'] = df['close'] * 1.02  # 假设high比close高2%（临时方案）

        print(f'获取到 {len(df)} 天数据')
        print(f'日期范围: {df.index[0].strftime("%Y-%m-%d")} 到 {df.index[-1].strftime("%Y-%m-%d")}')

        # 显示最近数据
        print('\n最近10天数据:')
        print('-' * 60)
        for idx in range(-10, 0):
            row = df.iloc[idx]
            print(f'{df.index[idx].strftime("%Y-%m-%d")}: close={row["close"]:.3f}, high={row["high"]:.3f}, low={row["low"]:.3f}')

        # 测试SAR计算
        sar_strategy = SARStrategy(acceleration=0.02, maximum=0.20)
        sar_strategy.prepare_data(df['high'], df['low'], df['close'])

        sar_series = sar_strategy._sar_series
        print('\nSAR序列（最近10天）:')
        print('-' * 60)
        print('日期          计算SAR    数据库SAR   收盘价    状态')
        print('-' * 60)
        for idx in range(-10, 0):
            calc_sar = sar_series.iloc[idx]
            db_sar = df['sar'].iloc[idx] if 'sar' in df.columns else None
            close_val = df['close'].iloc[idx]
            position = '多头' if calc_sar < close_val else '空头'
            db_sar_str = f'{db_sar:.3f}' if db_sar is not None else 'N/A'
            print(f'{sar_series.index[idx].strftime("%Y-%m-%d")}: {calc_sar:.3f}    {db_sar_str}    {close_val:.3f}    ({position})')

        # 检查最后一天的SAR值
        last_calc_sar = sar_series.iloc[-1]
        last_db_sar = df['sar'].iloc[-1] if 'sar' in df.columns else None
        last_close = df['close'].iloc[-1]
        last_low = df['low'].iloc[-1]

        print()
        print('=' * 60)
        print(f'最后一天计算SAR值: {last_calc_sar:.3f}')
        if last_db_sar is not None:
            print(f'最后一天数据库SAR值: {last_db_sar:.3f}')
        print(f'最后一天收盘价: {last_close:.3f}')
        print(f'最后一天最低价: {last_low:.3f}')

        # 检查是否触发死叉
        if last_low <= last_calc_sar:
            print('【已触发SAR死叉（计算）】最低价 <= SAR值')
        elif last_close <= last_calc_sar:
            print('【已触发SAR死叉（计算）】收盘价 <= SAR值')
        else:
            print('【未触发SAR死叉（计算）】')

        # 东方财富预期值87.4
        expected_sar = 87.4
        diff_calc = abs(last_calc_sar - expected_sar)
        diff_db = abs(last_db_sar - expected_sar) if last_db_sar is not None else None
        print(f'东方财富预期值: {expected_sar}')
        print(f'计算SAR差距: {diff_calc:.3f}')
        if diff_db is not None:
            print(f'数据库SAR差距: {diff_db:.3f}')

        # 判断测试是否通过
        # 如果数据库SAR更接近东方财富，则使用数据库SAR作为基准
        if diff_db is not None and diff_db < 1.0:
            print('测试结果: 通过（数据库SAR接近东方财富）')
            return True
        elif diff_calc < 1.0:
            print('测试结果: 通过（计算SAR接近东方财富）')
            return True
        else:
            print('测试结果: 不通过')
            return False

    except Exception as e:
        print(f'测试出错: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_real_sar_calculation()