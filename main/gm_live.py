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

# 导入原有的策略逻辑
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
        timestamp = str(round(time.time() * 1000))
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code).decode('utf-8'))

        webhook_url = f"{webhook}&timestamp={timestamp}&sign={sign}"
        data = {"msgtype": "text", "text": {"content": message}}
        headers = {'Content-Type': 'application/json'}
        requests.post(webhook_url, json=data, headers=headers)
        logger.info("钉钉告警发送成功")
    except Exception as e:
        logger.error(f"发送钉钉告警出错：{e}")

# --- 策略核心 ---
def init(context):
    # 1. 加载配置文件
    settings_path = os.path.join(os.path.dirname(__file__), 'config/settings.json')
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
        data = history_n(symbol=symbol, frequency='1d', count=actual_count, end_time=None,
                         fields='eob,open,high,low,close,volume', df=True, adjust=ADJUST_PREV)
        if data is None or data.empty:
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

    data = history_n(symbol=symbol, frequency=gm_freq, count=count, end_time=None,
                     fields='eob,open,high,low,close,volume', df=True, adjust=ADJUST_PREV)
    if data is None or data.empty:
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
    for stock in context.stock_configs:
        symbol = stock['gm_symbol']
        name = stock['name']
        try:
            # 获取数据（日常监控使用日线）
            df_weekly = fetch_and_format_data(symbol, 'w', 100)
            df_daily = fetch_and_format_data(symbol, 'd', 100)
            df_120min = fetch_and_format_data(symbol, '7200s', 100)

            all_data = {
                'weekly': strategy.calculate_indicators(df_weekly),
                'daily': strategy.calculate_indicators(df_daily),
                '120min': strategy.calculate_indicators(df_120min) if df_120min is not None else None
            }

            if all_data['daily'] is None or all_data['weekly'] is None:
                logger.warning(f"[{name}] 数据获取失败，跳过检查")
                continue

            # --- 新增：打印股票详细指标信息 ---
            rsi_daily = all_data['daily']['RSI'].iloc[-1]
            rsi_weekly = all_data['weekly']['RSI'].iloc[-1]
            price = all_data['daily']['Close'].iloc[-1]
            sar = all_data['daily']['SAR'].iloc[-1]
            macd_dif = all_data['daily']['MACD'].iloc[-1]
            macd_dea = all_data['daily']['MACD_signal'].iloc[-1]
            
            status_sar = "高于SAR" if price > sar else "跌破SAR"
            status_macd = "金叉" if macd_dif > macd_dea else "死叉"
            
            logger.info(f"[{name}] 现价:{price:.2f} | 日RSI:{rsi_daily:.2f} | 周RSI:{rsi_weekly:.2f} | SAR:{sar:.2f}({status_sar}) | MACD:{status_macd}")
            # --------------------------------

            # 信号判断（卖出）
            sell_msgs = strategy.judge_sell(name, stock['judge_sell_ids'], all_data)
            if sell_msgs:
                send_dingtalk_alert("\n".join(sell_msgs), context.dingtalk_webhook, context.dingtalk_secret)

            # 获取指数数据的辅助函数
            def get_index_data(ticker):
                return strategy.calculate_indicators(
                    fetch_and_format_data(convert_to_gm_format(ticker), 'd', 100)
                )

            # 信号判断（买入）
            buy_msgs = strategy.judge_buy(name, stock['judge_buy_ids'], all_data, get_index_data)
            if buy_msgs:
                send_dingtalk_alert("\n".join(buy_msgs), context.dingtalk_webhook, context.dingtalk_secret)

            # 信号判断（做 T 买回）
            t_buy_msgs = strategy.judge_t_buy(name, stock.get('judge_t_ids', []), all_data, get_index_data)
            if t_buy_msgs:
                send_dingtalk_alert("\n".join(t_buy_msgs), context.dingtalk_webhook, context.dingtalk_secret)

        except Exception as e:
            logger.error(f"处理 {name} 出错：{e}")

if __name__ == '__main__':
    # 从配置文件加载 token
    settings_path = os.path.join(os.path.dirname(__file__), 'config/settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    run(strategy_id='YOUR_LIVE_STRATEGY_ID',
        filename='gm_live.py',
        mode=MODE_LIVE,
        token=settings.get('gm_token'))
