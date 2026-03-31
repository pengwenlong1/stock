# -*- coding: utf-8 -*-
from gm.api import *
import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta

# 导入策略逻辑
from strategy import strategy


# --- 日志配置 ---
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("GM_Backtest")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, 'backtest.log'), mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = setup_logger()

# --- 加载配置 ---
def load_gm_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)
    return settings

gm_config = load_gm_config()

# --- 工具函数 ---
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

def get_stock_listing_date(symbol):
    """获取股票的上市日期"""
    try:
        data = history_n(symbol=symbol, frequency='1d', count=1300,
                         end_time=None, fields='eob', df=True)
        if not data.empty:
            earliest_date = data['eob'].min()
            if isinstance(earliest_date, str):
                return datetime.strptime(earliest_date[:10], '%Y-%m-%d')
            if hasattr(earliest_date, 'date'):
                return earliest_date.date()
            return earliest_date
    except Exception as e:
        logger.debug(f"获取 {symbol} 最早交易日期失败：{e}")
    return None

def calculate_backtest_period(symbol):
    """计算回测时间段：回测过去 5 年，不足 5 年则从上市时间开始"""
    end_time = datetime.now()
    listing_date = get_stock_listing_date(symbol)

    if listing_date is None:
        start_time = end_time - timedelta(days=5*365)
        logger.info(f"{symbol}: 无法获取上市日期，默认回测过去 5 年")
    else:
        if isinstance(listing_date, datetime):
            listing_date = listing_date.date()
        five_years_ago = (end_time - timedelta(days=5*365)).date()
        if listing_date > five_years_ago:
            start_time = datetime.combine(listing_date, datetime.min.time())
            logger.info(f"{symbol}: 上市时间不足 5 年，从 {listing_date} 开始")
        else:
            start_time = end_time - timedelta(days=5*365)
            logger.info(f"{symbol}: 回测过去 5 年")

    return start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S')

def fetch_and_format_data(symbol, frequency, count, end_time=None):
    """获取并格式化数据"""
    if frequency == 'w':
        actual_count = count * 5
        data = history_n(symbol=symbol, frequency='1d', count=actual_count, end_time=end_time,
                         fields='eob,open,high,low,close,volume', df=True, adjust=ADJUST_PREV)
        if data is None or data.empty:
            return None
        data['eob'] = pd.to_datetime(data['eob'])
        data.set_index('eob', inplace=True)
        resampled = data.resample('W-FRI').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        resampled.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        return resampled.tail(count)

    if frequency == '7200s':
        gm_freq = '7200s'
    else:
        gm_freq = '1d'

    data = history_n(symbol=symbol, frequency=gm_freq, count=count, end_time=end_time,
                     fields='eob,open,high,low,close,volume', df=True, adjust=ADJUST_PREV)
    if data is None or data.empty:
        return None
    data.rename(columns={
        'eob': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def calculate_performance_metrics(symbol, df_price, trade_records):
    """计算回测绩效指标"""
    if df_price is None or len(df_price) < 2:
        logger.warning(f"{symbol}: 数据不足")
        return None

    df = df_price.sort_index().copy()
    initial_cash = 1000000
    cash = initial_cash
    position = 0
    portfolio_values = []
    hold_values = []

    first_price = df['Close'].iloc[0]
    hold_shares = initial_cash / first_price

    trade_signals = {r['date']: r['action'] for r in trade_records}
    holding = False

    for idx, row in df.iterrows():
        current_date = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        price = row['Close']

        if current_date in trade_signals:
            action = trade_signals[current_date]
            if action == 'buy' and not holding:
                position = cash / price
                cash = 0
                holding = True
            elif action == 'sell' and holding:
                cash = position * price
                position = 0
                holding = False

        portfolio_values.append(cash + position * price)
        hold_values.append(hold_shares * price)

    final_value = portfolio_values[-1] if portfolio_values else initial_cash
    final_hold = hold_values[-1] if hold_values else initial_cash

    strategy_return = (final_value - initial_cash) / initial_cash * 100
    hold_return = (final_hold - initial_cash) / initial_cash * 100

    def calc_max_dd(values):
        if len(values) < 2:
            return 0
        peak = values[0]
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    import numpy as np
    def calc_sharpe(values):
        if len(values) < 2:
            return 0
        rets = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        if len(rets) < 2:
            return 0
        avg = np.mean(rets)
        std = np.std(rets)
        if std == 0:
            return 0
        return (avg - 0.03/252) / std * np.sqrt(252)

    return {
        'symbol': symbol,
        '回测天数': len(df),
        '策略总收益率 (%)': round(strategy_return, 2),
        '买入持有收益率 (%)': round(hold_return, 2),
        '策略超额收益 (%)': round(strategy_return - hold_return, 2),
        '策略最大回撤 (%)': round(calc_max_dd(portfolio_values), 2),
        '买入持有最大回撤 (%)': round(calc_max_dd(hold_values), 2),
        '策略夏普比率': round(calc_sharpe(portfolio_values), 2),
        '买入持有夏普比率': round(calc_sharpe(hold_values), 2),
    }

# --- 全局变量 ---
GLOBAL_CONTEXT = None

# ==================== 回测配置 ====================
# 在这里修改要回测的股票/基金代码
TARGET_TICKERS = ['159928']
TARGET_NAMES = {'159928': '消费 ETF'}
JUDGE_BUY_IDS = [1, 2]   # 买入策略 ID
JUDGE_SELL_IDS = [1]     # 卖出策略 ID
# 回测结束时间（用于判断何时输出最终结果）
BACKTEST_END_DATE = '2026-03-20'
# ================================================

# --- 策略核心 ---
def init(context):
    """初始化函数"""
    global GLOBAL_CONTEXT

    target_tickers = TARGET_TICKERS
    judge_buy_ids = JUDGE_BUY_IDS
    judge_sell_ids = JUDGE_SELL_IDS

    context.stock_configs = []
    for ticker in target_tickers:
        name = TARGET_NAMES.get(ticker, ticker)
        gm_symbol = convert_to_gm_format(ticker)
        stock = {
            'ticker': ticker,
            'name': name,
            'gm_symbol': gm_symbol,
            'judge_buy_ids': judge_buy_ids,
            'judge_sell_ids': judge_sell_ids
        }
        start_time, end_time = calculate_backtest_period(gm_symbol)
        stock['backtest_start_time'] = start_time
        stock['backtest_end_time'] = end_time
        context.stock_configs.append(stock)

    context.trade_records = {ticker: [] for ticker in target_tickers}
    GLOBAL_CONTEXT = context

    # 每天 15:00 执行（会在 run_monitor 中检查是否为最后一天并输出结果）
    schedule(schedule_func=run_monitor, date_rule='1d', time_rule='15:00:00')

    logger.info(f"回测初始化完成，回测 {len(context.stock_configs)} 只股票：{target_tickers}")

def check_signals_for_stock(context, stock, check_time):
    """对单只股票进行信号检查"""
    symbol = stock['gm_symbol']
    name = stock['name']
    ticker = stock['ticker']
    judge_buy_ids = stock['judge_buy_ids']
    judge_t_ids = stock.get('judge_t_ids', [])
    judge_sell_ids = stock['judge_sell_ids']

    try:
        start_time = stock['backtest_start_time']
        end_time = stock['backtest_end_time']

        if check_time < start_time or check_time > end_time:
            return

        df_weekly = fetch_and_format_data(symbol, 'w', 100, end_time=check_time)
        df_daily = fetch_and_format_data(symbol, 'd', 100, end_time=check_time)

        if df_weekly is None or df_daily is None or len(df_weekly) < 10 or len(df_daily) < 30:
            return

        all_data = {
            'weekly': strategy.calculate_indicators(df_weekly.copy()),
            'daily': strategy.calculate_indicators(df_daily.copy())
        }

        # 卖出信号
        sell_msgs = strategy.judge_sell(name, judge_sell_ids, all_data)
        if sell_msgs:
            for msg in sell_msgs:
                logger.info(f"[{check_time[:10]}] {msg}")
            context.trade_records[ticker].append({'date': check_time[:10], 'action': 'sell'})

        # 买入信号
        def get_index_data(idx_ticker):
            idx_sym = convert_to_gm_format(idx_ticker)
            idx_df = fetch_and_format_data(idx_sym, 'd', 100, end_time=check_time)
            if idx_df is not None:
                return strategy.calculate_indicators(idx_df.copy())
            return None

        buy_msgs = strategy.judge_buy(name, judge_buy_ids, all_data, get_index_data)
        if buy_msgs:
            for msg in buy_msgs:
                logger.info(f"[{check_time[:10]}] {msg}")
            context.trade_records[ticker].append({'date': check_time[:10], 'action': 'buy'})

        # 做 T 买回信号
        t_buy_msgs = strategy.judge_t_buy(name, judge_t_ids, all_data, get_index_data)
        if t_buy_msgs:
            for msg in t_buy_msgs:
                logger.info(f"[{check_time[:10]}] {msg}")
            # 在回测中，做 T 买回也视为买入行为
            context.trade_records[ticker].append({'date': check_time[:10], 'action': 'buy'})

    except Exception as e:
        logger.error(f"处理 {name} 出错：{e}")

def run_monitor(context):
    """回测主函数 - 每个交易日调用"""
    check_time = context.now.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"====== 回测检查时间：{check_time} ======")

    for stock in context.stock_configs:
        check_signals_for_stock(context, stock, check_time)

    # 在回测最后一天输出结果（检查是否到达回测结束日期）
    if check_time[:10] >= BACKTEST_END_DATE:
        logger.info(f"已到回测结束日期 {BACKTEST_END_DATE}，准备输出最终结果...")
        output_final_result(context)

def output_final_result(context):
    """输出最终回测结果"""
    logger.info("\n" + "="*60)
    logger.info("交易记录汇总")
    logger.info("="*60)

    total_buy = 0
    total_sell = 0

    for ticker, records in context.trade_records.items():
        buy_count = sum(1 for r in records if r['action'] == 'buy')
        sell_count = sum(1 for r in records if r['action'] == 'sell')
        total_buy += buy_count
        total_sell += sell_count
        name = TARGET_NAMES.get(ticker, ticker)
        logger.info(f"{name} ({ticker}): 买入 {buy_count} 次，卖出 {sell_count} 次")

    logger.info(f"\n总计：买入 {total_buy} 次，卖出 {total_sell} 次")

    all_results = []
    for stock in context.stock_configs:
        symbol = stock['gm_symbol']
        name = stock['name']
        ticker = stock['ticker']
        end_time = stock['backtest_end_time']

        df_daily = fetch_and_format_data(symbol, 'd', 1300, end_time=end_time)
        if df_daily is not None:
            trade_records = context.trade_records.get(ticker, [])
            result = calculate_performance_metrics(symbol, df_daily, trade_records)
            if result:
                all_results.append(result)
                logger.info(f"\n{'='*50}")
                logger.info(f"标的：{name} ({ticker})")
                logger.info(f"{'='*50}")
                logger.info(f"策略总收益率：{result['策略总收益率 (%)']}%")
                logger.info(f"买入持有收益率：{result['买入持有收益率 (%)']}%")
                logger.info(f"策略超额收益：{result['策略超额收益 (%)']}%")
                logger.info(f"策略最大回撤：{result['策略最大回撤 (%)']}%")
                logger.info(f"买入持有最大回撤：{result['买入持有最大回撤 (%)']}%")
                logger.info(f"策略夏普比率：{result['策略夏普比率']}")
                logger.info(f"买入持有夏普比率：{result['买入持有夏普比率']}")

    if all_results:
        logger.info(f"\n{'='*60}")
        logger.info("回测结果汇总")
        logger.info(f"{'='*60}")
        df_summary = pd.DataFrame(all_results)
        df_summary = df_summary[['symbol', '策略总收益率 (%)', '买入持有收益率 (%)',
                                  '策略超额收益 (%)', '策略最大回撤 (%)', '买入持有最大回撤 (%)']]
        logger.info(f"\n{df_summary.to_string(index=False)}")

        avg_strategy = df_summary['策略总收益率 (%)'].mean()
        avg_hold = df_summary['买入持有收益率 (%)'].mean()
        avg_excess = df_summary['策略超额收益 (%)'].mean()
        logger.info(f"\n平均策略收益率：{avg_strategy:.2f}%")
        logger.info(f"平均买入持有收益率：{avg_hold:.2f}%")
        logger.info(f"平均超额收益：{avg_excess:.2f}%")

if __name__ == '__main__':
    from gm.api import MODE_BACKTEST

    backtest_config = {
        'strategy_id': gm_config.get('gm_strategy_id', 'strategy_test'),
        'filename': 'gm_backtest.py',
        'mode': MODE_BACKTEST,
        'token': gm_config.get('gm_token'),
        'backtest_start_time': '2021-03-21 09:00:00',
        'backtest_end_time': '2026-03-21 15:30:00',
        'backtest_initial_cash': 1000000,
        'backtest_commission_ratio': 0.0003,
        'backtest_slippage_ratio': 0.0001,
    }

    run(**backtest_config)
