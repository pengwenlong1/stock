# -*- coding: utf-8 -*-
from gm.api import *
import pandas as pd
import json
import os
import logging
import hmac
import hashlib
import base64
import urllib.parse
import requests
import time
import sys

# 导入原有的策略逻辑
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from strategy import strategy


# --- 日志配置 ---
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger("GM_LiveMonitor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

logger = setup_logger()

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

def send_dingtalk_alert(message, webhook, secret):
    """发送钉钉告警"""
    if not webhook:
        return
    try:
        lines = [line for line in str(message).splitlines() if line.strip()]
        preview = "\n".join(lines[:20])
        logger.info(f"钉钉告警原因（共 {len(lines)} 条）：\n{preview}")

        timestamp = str(round(time.time() * 1000))
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code).decode('utf-8'))

        webhook_url = f"{webhook}&timestamp={timestamp}&sign={sign}"
        data = {"msgtype": "text", "text": {"content": message}}
        headers = {'Content-Type': 'application/json'}
        resp = requests.post(webhook_url, json=data, headers=headers, timeout=10)
        if resp.ok:
            logger.info("钉钉告警发送成功")
        else:
            logger.error(f"钉钉告警发送失败：HTTP {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.error(f"发送钉钉告警出错：{e}")

# --- 策略核心 ---
def init(context):
    # 1. 加载配置文件
    settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    context.stocks_file = '../config/stocks.json'
    context.dingtalk_webhook = settings.get('dingtalk_webhook')
    context.dingtalk_secret = settings.get('dingtalk_secret')

    # 2. 加载股票列表
    config_path = os.path.join(os.path.dirname(__file__), context.stocks_file)
    with open(config_path, 'r', encoding='utf-8') as f:
        context.stock_configs = json.load(f)

    # 3. 转换代码格式
    for stock in context.stock_configs:
        stock['gm_symbol'] = convert_to_gm_format(stock['ticker'])
        stock['skip'] = False
        stock['skip_reason'] = None

    context._skip_logged_symbols = set()

    hk_stocks = [s for s in context.stock_configs if str(s.get('gm_symbol', '')).startswith('HKEX.')]
    if hk_stocks:
        probe_symbol = hk_stocks[0]['gm_symbol']
        try:
            probe = history_n(symbol=probe_symbol, frequency='1d', count=1, end_time=None,
                              fields='eob,open,high,low,close,volume', df=True)
        except Exception:
            probe = None

        if probe is None or probe.empty:
            for s in hk_stocks:
                s['skip'] = True
                s['skip_reason'] = '港股日线Bar数据未返回（history_n 返回空）'
            logger.warning(f"检测到港股日线Bar数据未返回，将跳过港股标的（示例 {probe_symbol} 无日线数据返回）。")

    # 4. 设置定时任务 (每天 14:50 执行，接近收盘)
    schedule(schedule_func=run_monitor, date_rule='1d', time_rule='14:50:00')
    
    # 5. 初始化后立即执行一次检查 (方便用户确认程序正常运行)
    logger.info(f"实盘监控初始化完成，监控 {len(context.stock_configs)} 只股票")
    logger.info("正在执行初始化后的首次检查...")
    run_monitor(context)
    logger.info("首次检查完成，程序将按计划在 14:50 执行后续检查。")

def fetch_and_format_data(symbol, frequency, count):
    """
    获取并格式化数据

    Args:
        symbol: 股票代码 (掘金格式)
        frequency: 频率 'w'(周线), 'd'(日线), '7200s'(120min)
        count: 获取的 K 线数量
    """
    if frequency == 'w':
        # 周线：通过日线数据 resample 得到
        # 获取足够的日线数据以生成所需的周线数量
        actual_count = count * 5
        try:
            data = history_n(symbol=symbol, frequency='1d', count=actual_count, end_time=None,
                             fields='eob,open,high,low,close,volume', df=True, adjust=ADJUST_PREV)
        except Exception as e:
            logger.warning(f"[{symbol}] 获取 w 数据失败：{e}")
            return None
        if data is None or data.empty:
            logger.warning(f"[{symbol}] 获取 w 数据为空（可能无权限/停牌/无行情）")
            return None
        data['eob'] = pd.to_datetime(data['eob'])
        data.set_index('eob', inplace=True)
        resampled = data.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        resampled.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        return resampled.tail(count)

    # 日线或120min
    if frequency == '7200s':
        gm_freq = '7200s'
    else:
        gm_freq = '1d'

    try:
        data = history_n(symbol=symbol, frequency=gm_freq, count=count, end_time=None,
                         fields='eob,open,high,low,close,volume', df=True, adjust=ADJUST_PREV)
        if data is None or data.empty:
            logger.warning(f"[{symbol}] 获取 {frequency} 数据为空（可能无权限/停牌/无行情）")
            return None
    except Exception as e:
        logger.warning(f"[{symbol}] 获取 {frequency} 数据失败：{e}")
        return None
    data.rename(columns={
        'eob': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def run_monitor(context):
    """每日监控主函数"""
    logger.info("====== 实盘监控：开始新一轮检查 ======")

    index_data_map = {}
    for idx_ticker in ['000001', '399006']:
        idx_symbol = convert_to_gm_format(idx_ticker)
        idx_daily_raw = fetch_and_format_data(idx_symbol, 'd', 260)
        if idx_daily_raw is not None and not idx_daily_raw.empty:
            idx_daily = strategy.calculate_indicators(idx_daily_raw.copy())
            if idx_daily is not None:
                index_data_map[idx_ticker] = idx_daily

    for stock in context.stock_configs:
        symbol = stock['gm_symbol']
        name = stock['name']
        judge_t_ids_stock = stock.get('judge_t_ids', [])
        if stock.get('skip'):
            if symbol not in context._skip_logged_symbols:
                logger.warning(f"[{name}] {stock.get('skip_reason')}，已跳过（{symbol}）")
                context._skip_logged_symbols.add(symbol)
            continue
        try:
            # 获取数据（日常监控使用日线）
            df_daily_raw = fetch_and_format_data(symbol, 'd', 220)
            if df_daily_raw is None or df_daily_raw.empty:
                logger.warning(f"[{name}] 数据获取失败，跳过检查")
                continue
            df_weekly = fetch_and_format_data(symbol, 'w', 120)
            df_120min = fetch_and_format_data(symbol, '7200s', 120)

            all_data_now = {
                'weekly': strategy.calculate_indicators(df_weekly) if df_weekly is not None else None,
                'daily': strategy.calculate_indicators(df_daily_raw.copy()),
                '120min': strategy.calculate_indicators(df_120min) if df_120min is not None else None
            }

            if all_data_now['daily'] is None or all_data_now['weekly'] is None:
                logger.warning(f"[{name}] 数据获取失败，跳过检查")
                continue

            # --- 新增：打印股票详细指标信息 ---
            rsi_daily = all_data_now['daily']['RSI'].iloc[-1]
            rsi_weekly = all_data_now['weekly']['RSI'].iloc[-1]
            logger.info(f"[{name}] 日RSI:{rsi_daily:.2f} | 周RSI:{rsi_weekly:.2f}")
            # --------------------------------

            def build_weekly_from_daily(daily_raw):
                resampled = daily_raw.resample('W-FRI').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                return resampled

            def build_monthly_from_daily(daily_raw):
                resampled = daily_raw.resample('ME').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                return resampled

            recent_days = [ts for ts in df_daily_raw.index[-7:]]
            recent_msgs = []
            for ts in recent_days:
                daily_slice_raw = df_daily_raw.loc[:ts]
                weekly_slice_raw = build_weekly_from_daily(daily_slice_raw)
                monthly_slice_raw = build_monthly_from_daily(daily_slice_raw)
                daily_slice = strategy.calculate_indicators(daily_slice_raw.copy())
                weekly_slice = strategy.calculate_indicators(weekly_slice_raw.copy()) if weekly_slice_raw is not None else None
                monthly_slice = strategy.calculate_indicators(monthly_slice_raw.copy()) if monthly_slice_raw is not None else None
                if daily_slice is None or weekly_slice is None:
                    continue

                all_data_day = {
                    'weekly': weekly_slice.tail(100),
                    'monthly': monthly_slice.tail(100) if monthly_slice is not None else None,
                    'daily': daily_slice.tail(100),
                    '120min': None
                }

                day_str = ts.strftime('%Y-%m-%d')

                if strategy.detect_divergence(all_data_day['daily'], 'top', grace_bars=3):
                    recent_msgs.append(f"{day_str} | 【{name}】告警触发: 结构-日线顶背离(3日有效，价格新高+MACD柱/或DIF背离)")
                if strategy.detect_divergence(all_data_day['weekly'], 'top'):
                    recent_msgs.append(f"{day_str} | 【{name}】告警触发: 结构-周线顶背离(价格新高+MACD柱/或DIF背离)")

                sell_msgs = strategy.judge_sell(name, stock['judge_sell_ids'], all_data_day)
                for m in sell_msgs:
                    recent_msgs.append(f"{day_str} | {m}")

                def get_index_data(idx_ticker):
                    idx_df = index_data_map.get(idx_ticker)
                    if idx_df is None:
                        return None
                    return idx_df.loc[:ts].tail(100)

                judge_buy_ids = [i for i in stock['judge_buy_ids'] if i not in judge_t_ids_stock]

                buy_msgs = strategy.judge_buy(name, judge_buy_ids, all_data_day, get_index_data)
                for m in buy_msgs:
                    recent_msgs.append(f"{day_str} | {m}")

                t_buy_msgs = strategy.judge_t_buy(name, judge_t_ids_stock, all_data_day, get_index_data)
                for m in t_buy_msgs:
                    m = m.replace("新资金买入信号", "做 T 买回信号")
                    m = m.replace("买入信号", "做 T 买回信号")
                    recent_msgs.append(f"{day_str} | {m}")

            if recent_msgs:
                normalized_msgs = []
                for line in recent_msgs:
                    updated = line
                    for tid in judge_t_ids_stock:
                        if f"(策略 {tid}-" in updated and "买入信号" in updated and "做 T 买回信号" not in updated:
                            updated = updated.replace("新资金买入信号", "做 T 买回信号")
                            updated = updated.replace("买入信号", "做 T 买回信号")
                            break
                    normalized_msgs.append(updated)
                send_dingtalk_alert("\n".join(normalized_msgs), context.dingtalk_webhook, context.dingtalk_secret)

        except Exception as e:
            logger.error(f"处理 {name} 出错：{e}")

if __name__ == '__main__':
    # 从配置文件加载 token
    settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config/settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    run(strategy_id='YOUR_LIVE_STRATEGY_ID',
        filename='gm_live.py',
        mode=MODE_LIVE,
        token=settings.get('gm_token'))
