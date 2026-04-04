# -*- coding: utf-8 -*-
from gm.api import *
import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入策略逻辑
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from strategy import strategy


# --- 日志配置 ---
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("GM_Backtest")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, 'backtest.log'), mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
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
        if '建议清仓' in msg or '建议卖出全部剩余仓位' in msg:
            return 1.0, msg
        if '建议卖出剩余 1/2' in msg:
            if fraction < 0.5:
                fraction = 0.5
                picked = msg
        if '建议卖出 1/3' in msg:
            if fraction < (1.0 / 3.0):
                fraction = 1.0 / 3.0
                picked = msg
        if '建议卖出 50%' in msg:
            if fraction < 0.5:
                fraction = 0.5
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
        last_sell_date = None  # 记录上次卖出的日期
        last_sar_breakdown_week = None  # 记录上次 RSI 阶梯减仓触发的周

        # RSI flag 状态跟踪
        # 初始化 rsi_flag=0，根据周线 RSI 峰值更新
        # 当触发 RSI 阶梯减仓后，重置为 0 并更新 last_sar_breakdown_week
        rsi_flag = 0  # 当前 RSI flag 状态
        rsi_peak_after_breakdown = 0  # 从上次 RSI 卖出后的周线 RSI 峰值

        # 日线顶背离状态
        div_pending_flag = 0  # 顶背离潜在标志：0=无潜在顶背离，1=检测到潜在顶背离（等待均线确认）
        div_flag = 0  # 顶背离有效标志：0=顶背离未确认，1=顶背离已确认（均线已跌破）
        div_date = None  # 顶背离发生的日期（潜在顶背离检测到的日期）
        div_high_price = 0  # 顶背离时的最高价（用于确认均线跌破后该高点是否仍然有效）

        # 周线顶背离状态（使用 rsi_state 统一管理）
        # rsi_state 包含：
        # - diverse_week_flag: 周线顶背离标志 (0=无背离，1=背离形成)
        # - div_weekly_date: 周线顶背离形成日期
        # - prev_high_price: 前一个局部高点的价格
        # - prev_high_macd: 前一个局部高点的 MACD 柱值
        # - prev_high_idx: 前一个局部高点的索引
        # - position_1_price: 位置 1 的价格（局部高点，RSI>90 时）
        # - position_4_price: 位置 4 的价格（RSI>90 后下跌 2%）
        # - position_5_price: 位置 5 的价格（从位置 4 上涨 10%）
        # - rsi_peak: RSI 峰值
        # - rsi_trough: RSI 谷值
        rsi_state = {
            'current_position': 0,
            'rsi_peak': 0,
            'rsi_trough': 0,
            'price_at_peak': 0,
            'position_1_price': 0,
            'position_2_price': 0,
            'position_4_price': 0,
            'position_5_price': 0,
            'prev_high_price': 0,
            'prev_high_macd': 0,
            'prev_high_idx': None,
            'diverse_week_flag': 0,
            'div_weekly_date': None
        }

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

            rsi_weekly = weekly['RSI'].iloc[-1] if 'RSI' in weekly.columns and len(weekly) > 0 else 0

            equity_curve.append({'Date': ts, 'Equity': cash + shares * close})

            # ===== RSI flag 状态更新 =====
            # 规则：
            # 1. 初始化 rsi_flag=0
            # 2. 周线 RSI>80, rsi_flag=1
            # 3. 周线 RSI>85, rsi_flag=2
            # 4. 周线 RSI>90, rsi_flag=3
            # 5. 当 5 日均线跌破 10 日均线并触发 RSI 卖出后，rsi_flag=0

            current_ma5_breakdown = False

            # 先计算 RSI flag（在检测均线跌破之前）
            if weekly is not None and 'RSI' in weekly.columns:
                if last_sar_breakdown_week is not None:
                    weekly_filtered = weekly[weekly.index > last_sar_breakdown_week]
                    if weekly_filtered is not None and not weekly_filtered.empty:
                        rsi_peak_after_breakdown = weekly_filtered['RSI'].max()
                    else:
                        pass
                else:
                    rsi_peak_after_breakdown = weekly['RSI'].max()

                rsi_flag = strategy.get_rsi_flag_level(rsi_peak_after_breakdown)

                # 如果 RSI 峰值超过 80，记录到日志文件（DEBUG级别）
                if rsi_flag > 0:
                    logger.debug(f"[{day_str}] [{name}] 周 RSI 峰值={rsi_peak_after_breakdown:.2f}, flag={rsi_flag}")
            else:
                rsi_flag = 0

            # 检查当前是否有均线跌破（5 日均线下穿 10 日均线）
            if 'MA5' in daily.columns and 'MA10' in daily.columns:
                ma5_curr = daily['MA5'].iloc[-1]
                ma10_curr = daily['MA10'].iloc[-1]
                ma5_prev = daily['MA5'].iloc[-2] if len(daily) >= 2 else ma5_curr
                ma10_prev = daily['MA10'].iloc[-2] if len(daily) >= 2 else ma10_curr
                # 检测下穿动作：昨天 MA5>=MA10，今天 MA5<MA10
                if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                    current_ma5_breakdown = True
                    logger.debug(f"[{day_str}] [{name}] 5日均线跌破10日均线，当日flag={rsi_flag}")

            # 保存当前的 rsi_flag 用于 judge_sell（在重置之前）
            rsi_flag_for_sell = rsi_flag

            # 调用 judge_sell，传入当前的 rsi_flag 值
            sell_msgs, rsi_flag_returned, sar_breakdown, rsi_state = strategy.judge_sell(
                name, stock_judge_sell_ids, all_data, rsi_flag=rsi_flag_for_sell, rsi_state=rsi_state
            )

            # 处理 judge_sell 返回的卖出信号（RSI 阶梯减仓）
            sell_fraction, sell_reason = parse_sell_fraction(sell_msgs)

            # ===== 重要：优先级检查 =====
            # 如果 RSI flag=3（周线RSI>90），应该清仓，优先级高于日线顶背离
            # 日线顶背离只卖出1/3，而RSI>90应该清仓
            # 所以当RSI flag=3触发时，不再检查日线顶背离
            if sell_fraction >= 1.0:  # 已触发清仓信号
                div_sell_triggered = False  # 忽略日线顶背离
                div_flag = 0
                div_pending_flag = 0

            # 检查顶背离状态
            data_daily_for_div = all_data.get('daily')
            current_high = data_daily_for_div['High'].iloc[-1] if data_daily_for_div is not None else 0

            # 1. 检测潜在顶背离（只在没有潜在顶背离时检测，且没有触发RSI清仓时）
            # 顶背离检测条件：股价创新高 + MACD 柱缩短 + 两个高点之间有回调
            # 注意：此时只记录潜在顶背离，不立即触发卖出
            if div_pending_flag == 0 and sell_fraction < 1.0:  # 只在没有触发清仓时检测顶背离
                div_details = strategy.get_divergence_details(data_daily_for_div, 'top')
                if div_details and div_details.get('detected'):
                    div_pending_flag = 1
                    div_date = day_str
                    # 记录两个高点中的较高者
                    div_high_price = max(div_details['recent_high'], div_details['prev_high'])
            else:
                # 2. 已有潜在顶背离，等待 5 日均线下穿 10 日均线确认
                # 注意：需要检测的是"下穿"动作，而不是均线状态
                if 'MA5' in data_daily_for_div.columns and 'MA10' in data_daily_for_div.columns:
                    ma5_curr = data_daily_for_div['MA5'].iloc[-1]
                    ma10_curr = data_daily_for_div['MA10'].iloc[-1]
                    ma5_prev = data_daily_for_div['MA5'].iloc[-2] if len(data_daily_for_div) >= 2 else ma5_curr
                    ma10_prev = data_daily_for_div['MA10'].iloc[-2] if len(data_daily_for_div) >= 2 else ma10_curr

                    # 检测下穿动作：昨天 MA5>=MA10，今天 MA5<MA10
                    if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                        # 5 日均线下穿 10 日均线，确认顶背离有效，触发卖出
                        div_flag = 1
                    else:
                        # 均线未下穿，检查是否需要重置（假背离）
                        # 逻辑：如果股价突破前高且 MACD 柱也突破前高的 MACD 值，说明是假背离
                        current_hist = data_daily_for_div['MACD_hist'].iloc[-1] if 'MACD_hist' in data_daily_for_div.columns else 0
                        high_arr = data_daily_for_div['High'].values
                        macd_hist_arr = data_daily_for_div['MACD_hist'].values
                        curr = len(data_daily_for_div) - 1

                        # 找最近 5 天内的最高价
                        recent_lookback = 5
                        recent_start = max(curr - recent_lookback + 1, 0)
                        recent_max_idx = recent_start + high_arr[recent_start:curr+1].argmax()
                        recent_high = high_arr[recent_max_idx]
                        recent_hist = macd_hist_arr[recent_max_idx]

                        # 找前一个高点（背离时的 prev_high）之前 30 天内的最高价作为对比基准
                        prev_lookback = 30
                        prev_end = recent_max_idx - 1
                        if prev_end > 10:
                            prev_start = max(prev_end - prev_lookback + 1, 0)
                            prev_idx = prev_start + high_arr[prev_start:prev_end+1].argmax()
                            prev_high = high_arr[prev_idx]
                            prev_hist = macd_hist_arr[prev_idx]

                            interval = recent_max_idx - prev_idx

                            # 假背离判断：
                            # 1. 价格创新高：recent_high > prev_high
                            # 2. MACD 柱也创新高：recent_hist > prev_hist
                            # 3. 间隔足够：interval >= 5
                            # 4. MACD 柱为正：recent_hist > 0
                            if interval >= 5 and recent_high > prev_high and recent_hist > prev_hist and recent_hist > 0:
                                div_pending_flag = 0
                                div_date = None
                                div_high_price = 0

            # 3. 如果顶背离已确认（5 日均线下穿 10 日均线），触发卖出
            div_sell_triggered = False
            if div_flag == 1 and sell_fraction < (1.0 / 3.0):  # 只在没有更大的卖出信号时触发顶背离卖出
                div_sell_triggered = True
                # 如果已经有卖出信号，取最大值
                if sell_fraction < (1.0 / 3.0):
                    sell_fraction = 1.0 / 3.0
                    sell_reason = f"【{name}】卖出信号 (策略 1-顶背离均线): 日线顶背离 ({div_date}) + 5 日均线下穿 10 日均线确认，前高={div_high_price:.3f}, 建议卖出 1/3"
                div_flag = 0  # 卖出后重置
                div_pending_flag = 0  # 重置潜在标志
                div_date = None
                div_high_price = 0
                # 注意：实际卖出日志会在下面的统一卖出逻辑中打印，包含日 RSI 和周 RSI

            # ===== 周线顶背离逻辑（使用新的 diverse_week_flag） =====
            # 注意：周线顶背离的判断现在在 strategy.judge_sell 中处理
            # 这里只处理卖出信号的触发
            data_weekly_for_div = all_data.get('weekly')
            if data_weekly_for_div is not None and not data_weekly_for_div.empty:
                # 从 rsi_state 中获取周线顶背离状态
                diverse_week_flag = rsi_state.get('diverse_week_flag', 0) if rsi_state else 0
                prev_high_price = rsi_state.get('prev_high_price', 0) if rsi_state else 0
                prev_high_macd = rsi_state.get('prev_high_macd', 0) if rsi_state else 0

                # 如果周线顶背离已确认（在 judge_sell 中已经检测），触发卖出
                if diverse_week_flag == 1 and sell_fraction < 0.5:  # 只在没有更大的卖出信号时触发
                    # 检查是否发生 5 周均线下穿 10 周均线
                    if 'MA5' in data_weekly_for_div.columns and 'MA10' in data_weekly_for_div.columns:
                        ma5_weekly_curr = data_weekly_for_div['MA5'].iloc[-1]
                        ma10_weekly_curr = data_weekly_for_div['MA10'].iloc[-1]
                        if len(data_weekly_for_div) >= 2:
                            ma5_weekly_prev = data_weekly_for_div['MA5'].iloc[-2]
                            ma10_weekly_prev = data_weekly_for_div['MA10'].iloc[-2]

                            if ma5_weekly_prev >= ma10_weekly_prev and ma5_weekly_curr < ma10_weekly_curr:
                                # 顶背离生效，触发卖出
                                div_sell_triggered = True
                                sell_fraction = 0.5  # 卖出 50%
                                # 获取周线顶背离形成日期
                                div_weekly_date = rsi_state.get('div_weekly_date', '未知')
                                sell_reason = f"【{name}】卖出信号 (策略 1-周线顶背离): 顶背离形成于 {div_weekly_date}, 前高={prev_high_price:.3f}, 5 周均线下穿 10 周均线确认，建议卖出 50%"
                                rsi_state['diverse_week_flag'] = 0  # 重置标志
                                # 注意：实际卖出日志会在下面的统一卖出逻辑中打印，包含日 RSI 和周 RSI

            # 检查是否在 5 个交易日内重复触发
            should_skip = False
            if last_sell_date is not None and shares > 0:
                # 统一转换为不带时区的日期进行比较
                current_date = ts.tz_localize(None) if hasattr(ts, 'tz') and ts.tz is not None else pd.to_datetime(ts)
                last_date = pd.to_datetime(last_sell_date).tz_localize(None)
                days_since_last_sell = (current_date - last_date).days
                if days_since_last_sell < 5:
                    should_skip = True

            if sell_fraction > 0 and shares > 0 and not should_skip:
                # 执行卖出
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
                            'rsi_daily': daily['RSI'].iloc[-1] if 'RSI' in daily.columns else None,
                            'rsi_weekly': rsi_weekly,
                        }
                    )
                    has_sold_before = True
                    last_sell_date = day_str
                    # 任何卖出后都重置 RSI 峰值
                    last_sar_breakdown_week = weekly.index[-1] if weekly is not None else None
                    rsi_peak_after_breakdown = 0
                    # 打印卖出日志，包含日线和周线 RSI 值
                    rsi_daily = daily['RSI'].iloc[-1] if 'RSI' in daily.columns else 0
                    logger.info(f"[{day_str}] 卖出：{sell_qty} 股 @ {close:.3f} | 日 RSI:{rsi_daily:.2f} | 周 RSI:{rsi_weekly:.2f} | {sell_reason}")
            elif sell_fraction > 0 and shares == 0:
                pass  # 有卖出信号但已空仓，跳过
            elif sell_fraction == 0 and shares > 0:
                pass  # 无卖出信号，继续持有

            def idx_getter(idx_ticker):
                return get_index_data(idx_ticker, ts)

            judge_buy_ids_clean = [i for i in stock_judge_buy_ids if i not in stock_judge_t_ids]
            buy_msgs = strategy.judge_buy(name, judge_buy_ids_clean, all_data, idx_getter)
            t_buy_msgs = strategy.judge_t_buy(name, stock_judge_t_ids, all_data, idx_getter, has_sold_before)

            buy_reason = buy_msgs[0] if buy_msgs else None
            t_buy_reason = t_buy_msgs[0] if t_buy_msgs else None

            executed_buy = False

            if buy_reason and cash > close * lot_size:
                # 新资金买入
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
                            'rsi_daily': daily['RSI'].iloc[-1] if 'RSI' in daily.columns else None,
                            'rsi_weekly': rsi_weekly,
                        }
                    )
                    executed_buy = True
                    has_sold_before = False
                    logger.info(f"[{day_str}] 买入：{buy_qty} 股 @ {close:.3f} | {buy_reason}")
            elif buy_reason:
                pass  # 有买入信号但没执行（现金不足）

            if not executed_buy and t_buy_reason and has_sold_before and cash > close * lot_size:
                # 做 T 买回
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
                            'rsi_daily': daily['RSI'].iloc[-1] if 'RSI' in daily.columns else None,
                            'rsi_weekly': rsi_weekly,
                        }
                    )
                    logger.info(f"[{day_str}] 做 T 买回：{t_buy_qty} 股 @ {close:.3f} | {t_buy_reason}")

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

        # 绘制K线图
        plot_kline_with_trades(df_daily_raw, trades, name)

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
        sell_msgs, _, _ = strategy.judge_sell(name, judge_sell_ids, all_data)
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
    # {'ticker': '000895', 'name': '双汇', 'judge_buy_ids': [4], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    # {'ticker': '600941', 'name': '中国移动', 'judge_buy_ids': [4], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    {'ticker': '512480', 'name': '半导体 ETF', 'judge_buy_ids': [5], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    # {'ticker': '159622', 'name': '创新药', 'judge_buy_ids': [3], 'judge_t_ids': [2], 'judge_sell_ids': [1]},
    # {
    #     "ticker": "159928",
    #     "name": "消费ETF",
    #     "judge_buy_ids": [1],
    #     "judge_t_ids": [2],
    #     "judge_sell_ids": [1]
    # },
    # {
    #     "ticker": "159647",
    #     "name": "中药ETF",
    #     "judge_buy_ids": [1],
    #     "judge_t_ids": [2],
    #     "judge_sell_ids": [1]
    # },
    # {
    #     "ticker": "159873",
    #     "name": "医疗设备ETF",
    #     "judge_buy_ids": [1],
    #     "judge_t_ids": [2],
    #     "judge_sell_ids": [1]
    # },
]


def simplify_reason(reason):
    """
    简化买卖原因显示

    Args:
        reason: 完整的买卖原因

    Returns:
        简化后的原因
    """
    if not reason:
        return ''

    # 提取关键信息
    if '阶梯' in reason:
        if 'flag=3' in reason:
            return 'RSI>90清仓'
        elif 'flag=2' in reason:
            return 'RSI>85减半'
        elif 'flag=1' in reason:
            return 'RSI>80减1/3'
        return 'RSI阶梯减仓'
    elif '顶背离均线' in reason:
        return '顶背离+均线'
    elif '周线顶背离' in reason:
        return '周线顶背离'
    elif '清仓' in reason:
        return '清仓信号'
    elif '保守' in reason:
        return '保守买入'
    elif '标准' in reason:
        return '标准买入'
    elif '指数保护' in reason:
        return '指数保护买入'
    elif '做 T 买回' in reason:
        return '做T买回'

    # 如果没有匹配，返回简化版本
    return reason[:15] + '...' if len(reason) > 15 else reason


def plot_kline_with_trades(df_daily, trades, name, output_dir=None):
    """
    绘制K线图并标记买卖点

    Args:
        df_daily: 日线数据（包含 Open, High, Low, Close, RSI）
        trades: 交易记录列表，每个元素包含 date, action, shares, price, reason, rsi_daily, rsi_weekly
        name: 股票名称
        output_dir: 输出目录，默认为 logs 目录
    """
    if df_daily is None or len(df_daily) == 0:
        logger.warning(f"[{name}] 无数据，跳过绘图")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 准备数据
    df = df_daily.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date') if not isinstance(df.index, pd.DatetimeIndex) else df

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{name} 回测K线图', fontsize=14, fontweight='bold')

    # ===== 绘制K线图 =====
    # 计算K线的宽度
    width = 0.6
    width2 = 0.1

    # 定义颜色
    up_color = '#ff4136'  # 上涨红色
    down_color = '#0074D9'  # 下跌蓝色

    # 绘制K线
    for i, (idx, row) in enumerate(df.iterrows()):
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']

        # 判断涨跌
        if close_price >= open_price:
            color = up_color
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = down_color
            body_bottom = close_price
            body_height = open_price - close_price

        # 绘制实体
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor=color, alpha=0.8)
        ax1.add_patch(rect)

        # 绘制上下影线
        ax1.plot([i, i], [low_price, min(open_price, close_price)], color=color, linewidth=1)
        ax1.plot([i, i], [max(open_price, close_price), high_price], color=color, linewidth=1)

    # 设置K线图的Y轴范围和标签
    ax1.set_ylabel('价格', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ===== 标记买卖点 =====
    if trades:
        # 创建日期到索引的映射
        date_to_idx = {}
        for i, idx in enumerate(df.index):
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
            date_to_idx[date_str] = i

        buy_points = []
        sell_points = []

        for trade in trades:
            date_str = trade['date']
            if date_str in date_to_idx:
                idx_pos = date_to_idx[date_str]
                price = trade['price']
                action = trade['action']
                reason = trade.get('reason', '')
                rsi_daily = trade.get('rsi_daily', None)
                rsi_weekly = trade.get('rsi_weekly', None)

                if action == 'BUY':
                    buy_points.append((idx_pos, price, reason, rsi_daily, rsi_weekly))
                else:
                    sell_points.append((idx_pos, price, reason, rsi_daily, rsi_weekly))

        # 标记买入点
        for idx_pos, price, reason, rsi_daily, rsi_weekly in buy_points:
            # 画买入箭头
            ax1.annotate('',
                        xy=(idx_pos, price * 0.98),
                        xytext=(idx_pos, price * 0.92),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))

            # 获取日期
            date_str = df.index[idx_pos].strftime('%Y-%m-%d') if hasattr(df.index[idx_pos], 'strftime') else str(df.index[idx_pos])[:10]

            # 准备标注文本
            label_text = f'买入 {date_str}\n{price:.2f}'
            if rsi_daily is not None:
                label_text += f'\n日RSI:{rsi_daily:.1f}'
            if rsi_weekly is not None:
                label_text += f'\n周RSI:{rsi_weekly:.1f}'

            # 简化原因显示
            short_reason = simplify_reason(reason)
            label_text += f'\n{short_reason}'

            ax1.annotate(label_text,
                        xy=(idx_pos, price * 0.92),
                        fontsize=7,
                        color='green',
                        ha='center',
                        va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        # 标记卖出点
        for idx_pos, price, reason, rsi_daily, rsi_weekly in sell_points:
            # 画卖出箭头
            ax1.annotate('',
                        xy=(idx_pos, price * 1.02),
                        xytext=(idx_pos, price * 1.08),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

            # 获取日期
            date_str = df.index[idx_pos].strftime('%Y-%m-%d') if hasattr(df.index[idx_pos], 'strftime') else str(df.index[idx_pos])[:10]

            # 准备标注文本
            label_text = f'卖出 {date_str}\n{price:.2f}'
            if rsi_daily is not None:
                label_text += f'\n日RSI:{rsi_daily:.1f}'
            if rsi_weekly is not None:
                label_text += f'\n周RSI:{rsi_weekly:.1f}'

            # 简化原因显示
            short_reason = simplify_reason(reason)
            label_text += f'\n{short_reason}'

            ax1.annotate(label_text,
                        xy=(idx_pos, price * 1.08),
                        fontsize=7,
                        color='red',
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

    # 设置X轴刻度（每20个交易日显示一个）
    tick_positions = list(range(0, len(df), max(1, len(df)//10)))
    tick_labels = [df.index[i].strftime('%m-%d') if hasattr(df.index[i], 'strftime') else str(df.index[i])[:10]
                   for i in tick_positions if i < len(df)]
    ax1.set_xticks(tick_positions[:len(tick_labels)])
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    # ===== 绘制RSI指标 =====
    if 'RSI' in df.columns:
        ax2.plot(range(len(df)), df['RSI'].values, color='purple', linewidth=1, label='日RSI')
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='超买线(80)')
        ax2.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='超卖线(20)')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 标记RSI超买超卖区域
        ax2.fill_between(range(len(df)), 80, 100, alpha=0.1, color='red')
        ax2.fill_between(range(len(df)), 0, 20, alpha=0.1, color='green')

    ax2.set_xlabel('日期', fontsize=10)
    ax2.set_xticks(tick_positions[:len(tick_labels)])
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, f'{name}_backtest_kline.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"[{name}] K线图已保存至: {output_path}")
    plt.close()


if __name__ == '__main__':
    run_backtest_history(
        start_date='2025-01-01',
        end_date='2026-04-03',
        initial_cash=1000000,
        lot_size=100,
        stock_configs=STOCK_CONFIGS,
    )
