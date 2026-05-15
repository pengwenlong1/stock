# -*- coding: utf-8 -*-
"""SAR计算自测脚本"""
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_path)

from src.tool.sar_strategy import SARStrategy

def test_sar_calculation():
    """测试SAR计算是否匹配东方财富"""
    print('开始SAR计算自测...')
    print('=' * 60)

    # 模拟圣邦股份的数据：从4月9日买入后开始上涨
    # 根据用户反馈：4/9买入价约79.563，之后涨到91，然后跌到89
    # 昨天(5/7前一个交易日)的SAR应该是87.4
    np.random.seed(42)
    dates = pd.date_range('2026-04-09', '2026-05-07', freq='B')
    n = len(dates)
    print(f'模拟数据天数: {n}')

    # 模拟价格走势：从79.563涨到91，然后回调到89
    prices = []
    lows = []
    highs = []

    start_price = 79.563
    for i, date in enumerate(dates):
        if i < n - 5:  # 大部分时间上涨
            close = start_price + (91 - start_price) * (i / (n - 5))
            high = close + np.random.uniform(0.3, 1.0)
            low = close - np.random.uniform(0.3, 0.8)
        else:  # 最后几天回调
            days_into_decline = i - (n - 5)
            close = 91 - (91 - 89) * (days_into_decline / 4)
            high = close + np.random.uniform(0.2, 0.6)
            low = close - np.random.uniform(0.5, 1.2)

        prices.append(close)
        highs.append(high)
        lows.append(low)

    df = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows
    }, index=dates)

    print('\n模拟价格数据（最后10天）:')
    print('-' * 60)
    for idx in range(-10, 0):
        row = df.iloc[idx]
        print(f'{df.index[idx].strftime("%Y-%m-%d")}: close={row["close"]:.3f}, high={row["high"]:.3f}, low={row["low"]:.3f}')

    # 测试SAR计算
    sar_strategy = SARStrategy(acceleration=0.02, maximum=0.20)
    sar_strategy.prepare_data(df['high'], df['low'], df['close'])

    sar_series = sar_strategy._sar_series
    print('\nSAR序列（最后10天）:')
    print('-' * 60)
    for idx in range(-10, 0):
        sar_val = sar_series.iloc[idx]
        close_val = df['close'].iloc[idx]
        position = '多头' if sar_val < close_val else '空头'
        print(f'{sar_series.index[idx].strftime("%Y-%m-%d")}: SAR={sar_val:.3f}, close={close_val:.3f} ({position})')

    # 检查最后一天的SAR值
    last_sar = sar_series.iloc[-1]
    expected_sar = 87.4
    diff = abs(last_sar - expected_sar)

    print()
    print('=' * 60)
    print(f'最后一天SAR值: {last_sar:.3f}')
    print(f'东方财富预期值: {expected_sar}')
    print(f'差距: {diff:.3f}')

    if diff < 1.0:
        print('测试结果: 通过 (差距 < 1.0)')
        return True
    else:
        print(f'测试结果: 不通过 (差距 >= 1.0)')
        return False

if __name__ == '__main__':
    test_sar_calculation()