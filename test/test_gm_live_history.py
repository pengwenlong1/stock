# -*- coding: utf-8 -*-
import sys
import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd

# 将父目录添加到系统路径以导入 strategy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gm.api import *
from strategy import strategy


# --- 日志配置 ---
def setup_logger():
    logger = logging.getLogger("Test_History")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def convert_to_gm_format(ticker):
    """将 ticker 转换为掘金量化格式"""
    if len(ticker) == 5:
        return f"HKEX.{ticker}"
    if ticker.startswith(('5', '6', '9')):
        return f"SHSE.{ticker}"
    elif ticker.startswith(('0', '1', '2', '3')):
        if ticker == '000001':
            return "SHSE.000001"
        return f"SZSE.{ticker}"
    elif ticker.startswith(('4', '8')):
        return f"BSE.{ticker}"
    if ticker == '399006':
        return "SZSE.399006"
    return ticker

def fetch_data_by_range(symbol, frequency, start_date, end_date):
    """根据时间范围获取原始数据，并包含预热数据"""
    # 增加 400 天的预热数据以保证周线 MACD 等指标完全稳定
    pre_heating_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
    
    gm_freq = '7200s' if frequency == '7200s' else '1d'
    try:
        # 使用 history 获取范围数据，并启用前复权处理
        data = history(symbol=symbol, frequency=gm_freq, start_time=f"{pre_heating_start} 09:00:00", 
                       end_time=f"{end_date} 15:30:00", fields='eob,open,high,low,close,volume', 
                       df=True, adjust=ADJUST_PREV)
        
        if data is None or data.empty:
            return None
            
        data.rename(columns={'eob': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except Exception as e:
        logger.warning(f"获取 {symbol} {frequency} 数据失败: {e}")
        return None

def resample_to_weekly(df_daily):
    """动态将日线重采样为周线（包含当前周的实时数据）"""
    if df_daily is None or df_daily.empty: return None
    resampled = df_daily.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return resampled

def resample_to_monthly(df_daily):
    if df_daily is None or df_daily.empty: return None
    resampled = df_daily.resample('ME').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return resampled

def run_test_history(config):
    target_tickers = config.get('target_tickers')
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    debug_dates = set(config.get('debug_dates', []))
    include_structure_signals = bool(config.get('include_structure_signals', True))

    # 1. 加载配置
    base_dir = os.path.dirname(os.path.dirname(__file__))
    settings_path = os.path.join(base_dir, 'config', 'settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    set_token(settings.get('gm_token'))

    stocks_path = os.path.join(base_dir, 'config', 'stocks.json')
    with open(stocks_path, 'r', encoding='utf-8') as f:
        all_stock_configs = json.load(f)

    if target_tickers:
        stock_configs = [s for s in all_stock_configs if s['ticker'] in target_tickers]
    else:
        stock_configs = all_stock_configs

    logger.info(f"开始测试范围: {start_date} 至 {end_date}")
    
    # 获取大盘原始数据
    index_raw_map = {}
    for idx_ticker in ['000001', '399006']:
        df = fetch_data_by_range(convert_to_gm_format(idx_ticker), '1d', start_date, end_date)
        if df is not None:
            index_raw_map[idx_ticker] = df

    results = []

    for stock in stock_configs:
        ticker = stock['ticker']
        name = stock['name']
        symbol = convert_to_gm_format(ticker)
        logger.info(f"正在分析: {name} ({ticker})...")

        # 获取原始数据
        df_daily_raw = fetch_data_by_range(symbol, '1d', start_date, end_date)
        df_120min_raw = fetch_data_by_range(symbol, '7200s', start_date, end_date)

        if df_daily_raw is None:
            logger.warning(f"  {name} 数据获取失败，跳过")
            continue

        # 遍历测试范围内的每一个日线时间点
        test_start_dt = pd.to_datetime(start_date).tz_localize(df_daily_raw.index.tz)
        test_dates = df_daily_raw.index[df_daily_raw.index >= test_start_dt]

        for current_time in test_dates:
            judge_buy_ids = config.get('judge_buy_ids', stock['judge_buy_ids'])
            judge_t_ids = config.get('judge_t_ids', stock.get('judge_t_ids', []))
            judge_sell_ids = config.get('judge_sell_ids', stock['judge_sell_ids'])
            judge_buy_ids = [i for i in judge_buy_ids if i not in judge_t_ids]

            # 1. 准备日线：取到当前时间点
            data_daily_slice = df_daily_raw.loc[:current_time]
            data_daily = strategy.calculate_indicators(data_daily_slice.copy())

            # 2. 准备周线：基于当前时间点之前的日线进行即时采样
            data_weekly_raw = resample_to_weekly(data_daily_slice)
            data_weekly = strategy.calculate_indicators(data_weekly_raw)

            data_monthly_raw = resample_to_monthly(data_daily_slice)
            data_monthly = strategy.calculate_indicators(data_monthly_raw)

            # 3. 准备 120min
            data_120min = None
            if df_120min_raw is not None:
                day_end = current_time.replace(hour=23, minute=59, second=59)
                data_120min_slice = df_120min_raw.loc[:day_end]
                data_120min = strategy.calculate_indicators(data_120min_slice.copy())

            all_data = {
                'daily': data_daily.tail(100),
                'weekly': data_weekly.tail(100),
                'monthly': data_monthly.tail(100) if data_monthly is not None else None,
                '120min': data_120min.tail(100) if data_120min is not None else None
            }

            current_day = current_time.strftime('%Y-%m-%d')
            if include_structure_signals:
                daily_top_struct = strategy.detect_divergence(all_data['daily'], 'top', grace_bars=3)
                weekly_top_struct = strategy.detect_divergence(all_data['weekly'], 'top')
                if daily_top_struct:
                    results.append(f"{current_day} | {name} | 【{name}】结构提示: 触发 [日线顶背离(3日有效)]")
                if weekly_top_struct:
                    results.append(f"{current_day} | {name} | 【{name}】结构提示: 触发 [周线顶背离]")

            if current_day in debug_dates:
                d = all_data['daily']
                w = all_data['weekly']
                print(
                    f"DEBUG {current_day} | {name} | "
                    f"daily_top={strategy.detect_divergence(d, 'top', grace_bars=3)} "
                    f"daily_hist_shrink={bool(d['MACD_hist'].iloc[-1] < d['MACD_hist'].iloc[-2])} "
                    f"daily_dif={d['MACD'].iloc[-1]:.4f} "
                    f"daily_dif_prev={d['MACD'].iloc[-2]:.4f} "
                    f"daily_hist={d['MACD_hist'].iloc[-1]:.4f} "
                    f"daily_hist_prev={d['MACD_hist'].iloc[-2]:.4f} "
                    f"daily_high={d['High'].iloc[-1]:.2f} "
                    f"daily_close={d['Close'].iloc[-1]:.2f} "
                    f"weekly_top={strategy.detect_divergence(w, 'top')}"
                )

            # 信号判断
            def get_index_data(idx_ticker):
                idx_raw = index_raw_map.get(idx_ticker)
                if idx_raw is None: return None
                idx_slice = idx_raw.loc[:current_time]
                return strategy.calculate_indicators(idx_slice.copy()).tail(100)

            sell_msgs, _, _ = strategy.judge_sell(name, judge_sell_ids, all_data)
            buy_msgs = strategy.judge_buy(name, judge_buy_ids, all_data, get_index_data)
            t_buy_msgs = strategy.judge_t_buy(name, judge_t_ids, all_data, get_index_data)

            for msg in (sell_msgs + buy_msgs + t_buy_msgs):
                results.append(f"{current_time.strftime('%Y-%m-%d')} | {name} | {msg}")

    # 输出结果
    logger.info("\n" + "="*60)
    logger.info(f"历史触发时间点汇总 ({start_date} ~ {end_date})")
    logger.info("="*60)
    if not results:
        logger.info("指定周期内未触发任何信号。")
    else:
        results.sort()
        for res in results:
            print(res)
    logger.info("="*60)

if __name__ == '__main__':
    # 测试配置
    test_config = {
        'target_tickers': ['002594'], # 比亚迪
        'start_date': '2026-03-01',
        'end_date': '2026-03-31',
        'debug_dates': ['2026-03-22', '2026-03-23', '2026-03-24', '2026-03-25', '2026-03-26'],
        'include_structure_signals': True,
        'judge_buy_ids': [1, 2],
        'judge_t_ids': [2],
        'judge_sell_ids': [1]
    }
    run_test_history(test_config)
