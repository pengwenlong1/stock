import pandas as pd
import talib

def calculate_indicators(data):
    """计算技术指标 (对齐国内行情软件算法)"""
    if data is None or data.empty:
        return None

    # 1. 计算 RSI (使用国内通用的 SMA(x, N, 1) 逻辑)
    close = data['Close']
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # 国内 RSI 公式：RSI = SMA(MAX(Close-LC,0),N,1) / SMA(ABS(Close-LC),N,1) * 100
    ma_up = up.ewm(alpha=1/6, adjust=False).mean()
    ma_down = down.ewm(alpha=1/6, adjust=False).mean()
    data['RSI'] = ma_up / (ma_up + ma_down) * 100

    # 2. 计算 MACD (12, 26, 9)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
        data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data['MACD_hist'] = data['MACD_hist'] * 2

    # 3. 计算 SAR (10, 2, 20)
    data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

    return data

def detect_divergence(data, type='top', grace_bars=None, confirm_sar=False):
    """
    检测背离

    Args:
        data: 包含 MACD 指标的数据
        type: 'top' 顶背离，'bottom' 底背离
        grace_bars: 允许信号在最近 N 根 K 线内持续有效（含当根）。None 表示不限制
        confirm_sar: 是否需要 SAR 跌破来确认顶背离有效（仅用于顶背离）
    """
    if data is None or len(data) < 30:
        return False

    close = data['Close'].values
    high = data['High'].values if 'High' in data.columns else close
    low = data['Low'].values if 'Low' in data.columns else close
    dif = data['MACD'].values  # DIF
    macd_hist = data['MACD_hist'].values  # MACD 柱

    curr = len(close) - 1

    # 追踪最近 N 根 K 线中的最高价/最低价，用于捕捉趋势顶点
    # 有时候顶背离发生时，最高价不在最后一根，而是在前 1-2 根
    recent_lookback = 5
    prev_lookback = 30
    if type == 'top':
        price = high
        recent_start = max(curr - recent_lookback + 1, 0)
        recent_max_idx = recent_start + price[recent_start:curr+1].argmax()

        prev_end = recent_max_idx - 1
        if prev_end <= 10:
            return False
        prev_start = max(prev_end - prev_lookback + 1, 0)
        prev_idx = prev_start + price[prev_start:prev_end+1].argmax()
        if recent_max_idx - prev_idx < 5:
            return False

        # 顶背离条件：
        # 1. 股价创新高（High 创新高）
        # 2. MACD 红柱缩短（红柱仍然为正，但比前一次高点短）
        # 3. 两次高点间隔 >= 5 根 K 线
        if price[recent_max_idx] > price[prev_idx] and \
           macd_hist[prev_idx] > 0 and macd_hist[recent_max_idx] > 0 and \
           macd_hist[recent_max_idx] < macd_hist[prev_idx]:
            # 如果需要 SAR 跌破确认，则检查是否已经跌破
            if confirm_sar:
                sar = data['SAR'].values if 'SAR' in data.columns else None
                if sar is not None:
                    # 检查最近是否 SAR 跌破（当前价 < SAR）
                    if close[curr] < sar[curr]:
                        return True
                return False
            if grace_bars is None:
                return True
            return (curr - recent_max_idx) <= (grace_bars - 1)
    else:
        price = low
        recent_start = max(curr - recent_lookback + 1, 0)
        recent_min_idx = recent_start + price[recent_start:curr+1].argmin()

        prev_end = recent_min_idx - 1
        if prev_end <= 10:
            return False
        prev_start = max(prev_end - prev_lookback + 1, 0)
        prev_idx = prev_start + price[prev_start:prev_end+1].argmin()
        if recent_min_idx - prev_idx < 3:
            return False

        if price[recent_min_idx] < price[prev_idx] and (dif[recent_min_idx] > dif[prev_idx] or macd_hist[recent_min_idx] > macd_hist[prev_idx]):
            if grace_bars is None:
                return True
            return (curr - recent_min_idx) <= (grace_bars - 1)
    return False

def is_death_cross(data, window=3):
    """
    检测 MACD 死叉 (DIF 下穿 DEA)

    Args:
        data: 包含 MACD 和 MACD_signal 的数据
        window: 检测最近 N 天内是否有死叉
    """
    if data is None or len(data) < window + 1:
        return False
    for i in range(1, window + 1):
        if data['MACD'].iloc[-i-1] > data['MACD_signal'].iloc[-i-1] and \
           data['MACD'].iloc[-i] < data['MACD_signal'].iloc[-i]:
            return True
    return False

def is_golden_cross(data, window=3):
    """
    检测 MACD 金叉 (DIF 上穿 DEA)

    Args:
        data: 包含 MACD 和 MACD_signal 的数据
        window: 检测最近 N 天内是否有金叉
    """
    if data is None or len(data) < window + 1:
        return False
    for i in range(1, window + 1):
        if data['MACD'].iloc[-i-1] < data['MACD_signal'].iloc[-i-1] and \
           data['MACD'].iloc[-i] > data['MACD_signal'].iloc[-i]:
            return True
    return False

def get_rsi_flag_level(max_rsi):
    """
    根据 RSI 值获取 flag 等级
    flag=0: RSI <= 80
    flag=1: 80 < RSI <= 85
    flag=2: 85 < RSI <= 90
    flag=3: RSI > 90
    """
    if max_rsi > 90:
        return 3
    elif max_rsi > 85:
        return 2
    elif max_rsi > 80:
        return 1
    return 0

def get_sell_fraction_by_flag(rsi_flag):
    """
    根据 RSI flag 获取卖出比例
    flag=0: 无卖出信号
    flag=1: 卖出 1/3
    flag=2: 卖出 1/2
    flag=3: 清仓
    """
    if rsi_flag == 3:
        return 1.0, "建议卖出全部剩余仓位"
    elif rsi_flag == 2:
        return 0.5, "建议卖出剩余 1/2"
    elif rsi_flag == 1:
        return 1.0/3.0, "建议卖出 1/3"
    return 0.0, None

def judge_sell(stock_name, judge_sell_ids, all_data, rsi_flag=None):
    """
    判断卖出信号

    卖出策略 1 (sell_id=1):
    1. 减仓 1/3：日线顶背离 + 120 分钟死叉
    2. 清仓：周线顶背离 + 120 分钟死叉
    3. 阶梯减仓：日线 SAR 跌破，根据周线 RSI flag 决定卖出比例

    卖出策略 2 (sell_id=2):
    1. 减仓 1/3：日线顶背离 + 日线死叉
    2. 清仓：周线顶背离 + 日线死叉
    3. 阶梯减仓：日线 SAR 跌破，根据周线 RSI flag 决定卖出比例

    Args:
        stock_name: 股票名称
        judge_sell_ids: 卖出策略 ID 列表
        all_data: 包含 weekly, daily, 120min 数据的字典
        rsi_flag: 可选，外部传入的 RSI flag 状态 (0/1/2/3)，如果不传则根据历史峰值自动计算

    Returns:
        messages: 卖出信号列表
        rsi_flag: 当前的 RSI flag 状态（如果 SAR 跌破则返回 0，否则保持原值）
        sar_breakdown: 是否发生了 SAR 跌破
    """
    messages = []
    data_weekly = all_data.get('weekly')
    sar_breakdown = False

    # 如果没有传入 rsi_flag，则根据周线 RSI 历史峰值自动计算
    if rsi_flag is None:
        if data_weekly is not None and 'RSI' in data_weekly.columns:
            max_rsi = data_weekly['RSI'].max()
            rsi_flag = get_rsi_flag_level(max_rsi)
        else:
            rsi_flag = 0

    for sell_id in judge_sell_ids:
        data_monthly = all_data.get('monthly')
        data_daily = all_data.get('daily')
        data_120min = all_data.get('120min')

        if data_weekly is None or data_daily is None:
            continue

        if sell_id == 1:
            # 策略 1: 优先使用 120 分钟死叉 (用于实时监控)，如果没有则回退到日线死叉 (用于回测)
            data_cross = data_120min if data_120min is not None else data_daily

            # 1. 清仓信号：周线顶背离 + 120 分钟 (或日线) 死叉
            if detect_divergence(data_weekly, 'top') and is_death_cross(data_cross, window=3):
                cross_name = "120 分钟" if data_120min is not None else "日线"
                messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 触发 [周线顶背离 + {cross_name}死叉]，建议清仓")
            # 2. 清仓信号：月线顶背离 + 日线死叉
            elif data_monthly is not None and detect_divergence(data_monthly, 'top') and is_death_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 触发 [月线顶背离 + 日线死叉]，建议清仓")
            # 2. 减仓信号：日线顶背离 + 120 分钟 (或日线) 死叉
            elif detect_divergence(data_daily, 'top', grace_bars=3) and is_death_cross(data_cross, window=3):
                cross_name = "120 分钟" if data_120min is not None else "日线"
                messages.append(f"【{stock_name}】卖出信号 (策略 1-减仓): 触发 [日线顶背离 + {cross_name}死叉]，建议卖出 1/3")
            # 3. 阶梯式减仓信号：日线 SAR 跌破
            else:
                sar_breakdown_ts = find_sar_breakdown_date(data_daily, window=3)
                if sar_breakdown_ts is not None:
                    sar_breakdown = True
                    # SAR 跌破，根据 rsi_flag 决定卖出比例
                    sar_breakdown_day = sar_breakdown_ts.strftime('%Y-%m-%d') if hasattr(sar_breakdown_ts, 'strftime') else str(sar_breakdown_ts)
                    sell_fraction, sell_msg = get_sell_fraction_by_flag(rsi_flag)
                    if sell_fraction > 0:
                        trigger_reason = f"触发 [日线 SAR 跌破 ({sar_breakdown_day})]，周线 RSI flag={rsi_flag}"
                        messages.append(f"【{stock_name}】卖出信号 (策略 1-阶梯): {trigger_reason}，{sell_msg}")
                    # SAR 跌破后，flag 重置为 0
                    rsi_flag = 0

        elif sell_id == 2:
            # 1. 清仓信号：周线顶背离 + 日线死叉
            if detect_divergence(data_weekly, 'top') and is_death_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 触发 [周线顶背离 + 日线死叉]，建议清仓")
            # 2. 清仓信号：月线顶背离 + 日线死叉
            elif data_monthly is not None and detect_divergence(data_monthly, 'top') and is_death_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 触发 [月线顶背离 + 日线死叉]，建议清仓")
            # 2. 减仓信号：日线顶背离 + 日线死叉
            elif detect_divergence(data_daily, 'top', grace_bars=3) and is_death_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 2-减仓): 触发 [日线顶背离 + 日线死叉]，建议卖出 1/3")
            # 3. 阶梯式减仓信号：日线 SAR 跌破
            else:
                sar_breakdown_ts = find_sar_breakdown_date(data_daily, window=3)
                if sar_breakdown_ts is not None:
                    sar_breakdown = True
                    # SAR 跌破，根据 rsi_flag 决定卖出比例
                    sar_breakdown_day = sar_breakdown_ts.strftime('%Y-%m-%d') if hasattr(sar_breakdown_ts, 'strftime') else str(sar_breakdown_ts)
                    sell_fraction, sell_msg = get_sell_fraction_by_flag(rsi_flag)
                    if sell_fraction > 0:
                        trigger_reason = f"触发 [日线 SAR 跌破 ({sar_breakdown_day})]，周线 RSI flag={rsi_flag}"
                        messages.append(f"【{stock_name}】卖出信号 (策略 2-阶梯): {trigger_reason}，{sell_msg}")
                    # SAR 跌破后，flag 重置为 0
                    rsi_flag = 0

    return messages, rsi_flag, sar_breakdown

def judge_t_buy(stock_name, judge_t_ids, all_data, get_index_data_func=None, has_sold_before=False):
    """
    判断做 T 买回信号 (买回之前卖出的资金)

    Args:
        stock_name: 股票名称
        judge_t_ids: 做 T 策略 ID 列表
        all_data: 包含 weekly, daily, 120min 数据的字典
        get_index_data_func: 获取指数数据的函数
        has_sold_before: 是否之前有过卖出操作（关键参数）

    Returns:
        messages: 做 T 买回信号列表（如果没有卖出过则返回空列表）
    """
    messages = []
    if not judge_t_ids:
        return messages

    if not has_sold_before:
        return messages

    base_messages = judge_buy(stock_name, judge_t_ids, all_data, get_index_data_func)

    for msg in base_messages:
        t_msg = msg.replace("新资金买入信号", "做 T 买回信号")
        t_msg = t_msg.replace("买入信号", "做 T 买回信号")
        messages.append(t_msg)

    return messages

def is_sar_breakdown(data_daily, window=3):
    """检测日线 SAR 是否被跌破"""
    return find_sar_breakdown_date(data_daily, window=window) is not None

def find_sar_breakdown_date(data_daily, window=3):
    if data_daily is None or len(data_daily) < window + 1:
        return None
    for i in range(1, window + 1):
        if len(data_daily) >= i + 1:
            close_prev, sar_prev = data_daily['Close'].iloc[-i-1], data_daily['SAR'].iloc[-i-1]
            close_curr, sar_curr = data_daily['Close'].iloc[-i], data_daily['SAR'].iloc[-i]
            if close_prev > sar_prev and close_curr < sar_curr:
                try:
                    return data_daily.index[-i]
                except Exception:
                    return -i
    return None

def judge_buy(stock_name, judge_buy_ids, all_data, get_index_data_func=None):
    """
    判断买入信号

    买入条件：
    - buy_id=1: 日线 RSI < 20 且 周线 RSI < 25 (保守型)
    - buy_id=2: 日线 RSI < 25 且 周线 RSI < 30 (标准型)
    - buy_id=3: 上证指数或创业板指 日线 RSI < 25 (指数保护型)
    - buy_id=4: 日线 RSI < 20 且 周线 RSI < 20 (极度保守型)

    Args:
        stock_name: 股票名称
        judge_buy_ids: 买入策略 ID 列表
        all_data: 包含 weekly, daily, 120min 数据的字典
        get_index_data_func: 获取指数数据的函数
    """
    messages = []
    for buy_id in judge_buy_ids:
        data_weekly = all_data.get('weekly')
        data_daily = all_data.get('daily')

        if data_weekly is None or data_daily is None:
            continue

        min_daily_rsi = data_daily['RSI'].iloc[-1]
        min_weekly_rsi = data_weekly['RSI'].iloc[-1]

        if buy_id == 1:
            if min_daily_rsi < 20 and min_weekly_rsi < 25:
                messages.append(f"【{stock_name}】新资金买入信号 (策略 1-保守): 触发 [日线 RSI({min_daily_rsi:.2f}) < 20 且 周线 RSI({min_weekly_rsi:.2f}) < 25]")

        elif buy_id == 2:
            if min_daily_rsi < 25 and min_weekly_rsi < 30:
                messages.append(f"【{stock_name}】新资金买入信号 (策略 2-标准): 触发 [日线 RSI({min_daily_rsi:.2f}) < 25 且 周线 RSI({min_weekly_rsi:.2f}) < 30]")

        elif buy_id == 3:
            if get_index_data_func:
                idx_ticker, idx_name = '399006', '创业板指'
                idx_data = get_index_data_func(idx_ticker)
                if idx_data is not None and not idx_data.empty and 'RSI' in idx_data.columns:
                    idx_rsi = idx_data['RSI'].iloc[-1]
                    if idx_rsi < 25:
                        messages.append(f"【{stock_name}】新资金买入信号 (策略 3-指数保护): 触发 [{idx_name} 日线 RSI({idx_rsi:.2f}) < 25]")

        elif buy_id == 5:
            if get_index_data_func:
                # 只检查创业板指
                idx_ticker, idx_name = '399006', '创业板指'
                idx_data = get_index_data_func(idx_ticker)
                if idx_data is not None and not idx_data.empty and 'RSI' in idx_data.columns:
                    idx_rsi = idx_data['RSI'].iloc[-1]
                    if idx_rsi < 20:
                        messages.append(f"【{stock_name}】新资金买入信号 (策略 5-指数保护): 触发 [{idx_name} 日线 RSI({idx_rsi:.2f}) < 20]")

        elif buy_id == 4:
            if min_daily_rsi < 20 and min_weekly_rsi < 20:
                messages.append(f"【{stock_name}】新资金买入信号 (策略 4-极度超卖): 触发 [日线 RSI({min_daily_rsi:.2f}) < 20 且 周线 RSI({min_weekly_rsi:.2f}) < 20]")

    return messages
