# -*- coding: utf-8 -*-
from gm.api import *
import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta
import sys

# 导入策略逻辑
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.json')
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

def format_price(price):
    p = float(price)
    return f"{p:.4f}" if p < 10 else f"{p:.2f}"

def resample_to_weekly(df_daily):
    if df_daily is None or df_daily.empty:
        return None
    resampled = df_daily.resample('W-FRI').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return resampled

def resample_to_monthly(df_daily):
    if df_daily is None or df_daily.empty:
        return None
    resampled = df_daily.resample('ME').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return resampled

def parse_sell_fraction(messages):
    if not messages:
        return 0.0, None
    picked = None
    fraction = 0.0
    for msg in messages:
        if '建议清仓' in msg or '建议卖出全部' in msg:
            return 1.0, msg
        if '卖出剩余 1/2' in msg or '卖出 1/2' in msg:
            if fraction < 0.5:
                fraction = 0.5
                picked = msg
        if '卖出剩余 1/3' in msg or '卖出 1/3' in msg:
            if fraction < (1.0 / 3.0):
                fraction = 1.0 / 3.0
                picked = msg
    return fraction, picked

def run_backtest_history(
    start_date,
    end_date,
    initial_cash=1000000,
    lot_size=100,
    warmup_days=400,
    stock_configs=None,
    target_tickers=None,
    judge_buy_ids=None,
    judge_sell_ids=None,
    target_names=None,
    judge_t_ids=None,
):
    set_token(gm_config.get('gm_token'))
    judge_t_ids = judge_t_ids or []

    if stock_configs is None:
        target_tickers = target_tickers or []
        judge_buy_ids = judge_buy_ids or []
        judge_sell_ids = judge_sell_ids or []
        stock_configs = []
        for t in target_tickers:
            stock_configs.append(
                {
                    'ticker': t,
                    'name': target_names.get(t, t) if isinstance(target_names, dict) else t,
                    'judge_buy_ids': list(judge_buy_ids),
                    'judge_sell_ids': list(judge_sell_ids),
                    'judge_t_ids': list(judge_t_ids),
                }
            )

    summary_rows = []
    for stock in stock_configs:
        ticker = stock.get('ticker')
        if not ticker:
            continue
        name = stock.get('name') or (target_names.get(ticker, ticker) if isinstance(target_names, dict) else ticker)
        symbol = convert_to_gm_format(ticker)
        if symbol.startswith('HKEX.'):
            logger.warning(f"[{name}] 掘金不支持港股Bar数据，跳过（{symbol}）")
            continue
        stock_judge_buy_ids = stock.get('judge_buy_ids', judge_buy_ids) or []
        stock_judge_sell_ids = stock.get('judge_sell_ids', judge_sell_ids) or []
        stock_judge_t_ids = stock.get('judge_t_ids', judge_t_ids) or []

        warmup_start = (pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)).strftime('%Y-%m-%d')
        df_daily_raw = history(
            symbol=symbol,
            frequency='1d',
            start_time=f'{warmup_start} 09:00:00',
            end_time=f'{end_date} 15:30:00',
            fields='eob,open,high,low,close,volume',
            df=True,
            adjust=ADJUST_PREV,
        )
        if df_daily_raw is None or df_daily_raw.empty:
            logger.warning(f"[{name}] 无数据，跳过")
            continue

        df_daily_raw = df_daily_raw.rename(
            columns={'eob': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        )
        df_daily_raw['Date'] = pd.to_datetime(df_daily_raw['Date'])
        df_daily_raw = df_daily_raw.set_index('Date').sort_index()

        tz = df_daily_raw.index.tz
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        if tz is not None:
            start_ts = start_ts.tz_localize(tz)
            end_ts = end_ts.tz_localize(tz)

        trade_days = df_daily_raw.loc[start_ts:end_ts].index
        if len(trade_days) == 0:
            logger.warning(f"[{name}] 回测区间无交易日，跳过")
            continue

        cash = float(initial_cash)
        shares = 0
        has_sold_before = False
        trades = []
        equity_curve = []

        first_close = float(df_daily_raw.loc[trade_days[0]]['Close'])
        hold_shares = int((initial_cash / first_close) // lot_size) * lot_size
        hold_cash = initial_cash - hold_shares * first_close

        index_cache = {}

        def get_index_data(idx_ticker, ts):
            cache_key = (idx_ticker, ts)
            if cache_key in index_cache:
                return index_cache[cache_key]
            idx_sym = convert_to_gm_format(idx_ticker)
            idx_df = history(
                symbol=idx_sym,
                frequency='1d',
                start_time=f'{warmup_start} 09:00:00',
                end_time=f'{end_date} 15:30:00',
                fields='eob,open,high,low,close,volume',
                df=True,
                adjust=ADJUST_PREV,
            )
            if idx_df is None or idx_df.empty:
                index_cache[cache_key] = None
                return None
            idx_df = idx_df.rename(
                columns={'eob': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            )
            idx_df['Date'] = pd.to_datetime(idx_df['Date'])
            idx_df = idx_df.set_index('Date').sort_index()
            idx_slice = idx_df.loc[:ts].tail(100)
            idx_ind = strategy.calculate_indicators(idx_slice.copy())
            index_cache[cache_key] = idx_ind
            return idx_ind

        for ts in trade_days:
            daily_slice_raw = df_daily_raw.loc[:ts].tail(600)
            weekly_raw = resample_to_weekly(daily_slice_raw)
            monthly_raw = resample_to_monthly(daily_slice_raw)

            daily = strategy.calculate_indicators(daily_slice_raw.copy())
            weekly = strategy.calculate_indicators(weekly_raw.copy()) if weekly_raw is not None else None
            monthly = strategy.calculate_indicators(monthly_raw.copy()) if monthly_raw is not None else None

            if daily is None or weekly is None:
                continue

            daily = daily.tail(100)
            weekly = weekly.tail(100)
            monthly = monthly.tail(100) if monthly is not None else None

            all_data = {'daily': daily, 'weekly': weekly, 'monthly': monthly, '120min': None}
            close = float(daily['Close'].iloc[-1])
            day_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]

            equity_curve.append({'Date': ts, 'Equity': cash + shares * close})

            sell_msgs = strategy.judge_sell(name, stock_judge_sell_ids, all_data)
            sell_fraction, sell_reason = parse_sell_fraction(sell_msgs)
            if sell_fraction > 0 and shares > 0:
                if sell_fraction >= 0.999:
                    sell_qty = shares
                else:
                    sell_qty = int((shares * sell_fraction) // lot_size) * lot_size
                    sell_qty = min(sell_qty, shares)
                if sell_qty > 0:
                    cash += sell_qty * close
                    shares -= sell_qty
                    trades.append(
                        {
                            'date': day_str,
                            'action': 'SELL',
                            'shares': int(sell_qty),
                            'price': close,
                            'reason': sell_reason or 'SELL',
                        }
                    )
                    has_sold_before = True

            def idx_getter(idx_ticker):
                return get_index_data(idx_ticker, ts)

            judge_buy_ids_clean = [i for i in stock_judge_buy_ids if i not in stock_judge_t_ids]
            buy_msgs = strategy.judge_buy(name, judge_buy_ids_clean, all_data, idx_getter)
            t_buy_msgs = strategy.judge_t_buy(name, stock_judge_t_ids, all_data, idx_getter, has_sold_before)

            buy_reason = buy_msgs[0] if buy_msgs else None
            t_buy_reason = t_buy_msgs[0] if t_buy_msgs else None

            executed_buy = False
            
            if buy_reason and shares == 0 and cash > close * lot_size:
                buy_qty = int((cash / close) // lot_size) * lot_size
                if buy_qty > 0:
                    cash -= buy_qty * close
                    shares += buy_qty
                    trades.append(
                        {
                            'date': day_str,
                            'action': 'BUY',
                            'shares': int(buy_qty),
                            'price': close,
                            'reason': buy_reason,
                        }
                    )
                    executed_buy = True

            if not executed_buy and t_buy_reason and has_sold_before and cash > close * lot_size:
                t_buy_qty = int((cash / close) // lot_size) * lot_size
                if t_buy_qty > 0:
                    cash -= t_buy_qty * close
                    shares += t_buy_qty
                    trades.append(
                        {
                            'date': day_str,
                            'action': 'BUY',
                            'shares': int(t_buy_qty),
                            'price': close,
                            'reason': t_buy_reason,
                        }
                    )

        end_close = float(df_daily_raw.loc[trade_days[-1]]['Close'])
        final_value = cash + shares * end_close
        hold_value = hold_cash + hold_shares * end_close

        eq = pd.DataFrame(equity_curve).set_index('Date')['Equity']
        peak = eq.cummax()
        max_dd = (eq / peak - 1.0).min() * 100.0 if len(eq) else 0.0

        strategy_return = (final_value / initial_cash - 1.0) * 100.0
        hold_return = (hold_value / initial_cash - 1.0) * 100.0

        logger.info("=" * 60)
        logger.info(f"{name} ({symbol}) | {start_date} ~ {end_date}")
        logger.info(f"策略收益率: {strategy_return:.2f}%")
        logger.info(f"买入持有收益率: {hold_return:.2f}%")
        logger.info(f"策略超额收益: {strategy_return - hold_return:.2f}%")
        logger.info(f"最大回撤: {max_dd:.2f}%")
        logger.info("-" * 60)
        if not trades:
            logger.info("无交易记录")
        else:
            for t in trades:
                logger.info(f"{t['date']} | {t['action']} {t['shares']} @ {format_price(t['price'])} | {t['reason']}")

        summary_rows.append(
            {
                'ticker': ticker,
                'symbol': symbol,
                '策略收益率(%)': round(strategy_return, 2),
                '买入持有(%)': round(hold_return, 2),
                '超额收益(%)': round(strategy_return - hold_return, 2),
                '最大回撤(%)': round(max_dd, 2),
                '交易次数': len(trades),
            }
        )

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        logger.info("=" * 60)
        logger.info("汇总")
        logger.info(df_sum.to_string(index=False))

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

# --- 全局变量 ---
GLOBAL_CONTEXT = None

# ==================== 回测配置 ====================
# 在这里修改要回测的股票/基金代码
TARGET_TICKERS = ['159928', '512480']
TARGET_NAMES = {'512480': '半导体 ETF'}
JUDGE_BUY_IDS = [1, 2]   # 买入策略 ID
JUDGE_SELL_IDS = [1]     # 卖出策略 ID
# 回测结束时间（用于判断何时输出最终结果）
BACKTEST_END_DATE = '2026-03-20'
# ================================================

STOCK_CONFIGS = [
    {'ticker': '000895', 'name': '双汇', 'judge_buy_ids': [4], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    {'ticker': '600941', 'name': '中国移动', 'judge_buy_ids': [4], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    {'ticker': '512480', 'name': '半导体 ETF', 'judge_buy_ids': [3], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    {'ticker': '159622', 'name': '创新药', 'judge_buy_ids': [3], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
]

if __name__ == '__main__':
    run_backtest_history(
        start_date='2024-02-01',
        end_date='2026-03-31',
        initial_cash=1000000,
        lot_size=100,
        stock_configs=STOCK_CONFIGS,
    )
