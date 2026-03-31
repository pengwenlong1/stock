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

def detect_divergence(data, type='top'):
    """
    检测背离 (仅根据 MACD DIF)

    Args:
        data: 包含 MACD 指标的数据
        type: 'top' 顶背离，'bottom' 底背离
    """
    if data is None or len(data) < 30:
        return False

    close = data['Close'].values
    dif = data['MACD'].values  # DIF
    macd_hist = data['MACD_hist'].values  # MACD 柱

    def find_last_significant_extrema(values, is_max=True, window=5):
        """寻找最近的显著极值点"""
        for i in range(len(values) - window - 1, window, -1):
            if is_max:
                if all(values[i] > values[i-j] for j in range(1, window+1)) and \
                   all(values[i] > values[i+j] for j in range(1, window+1)):
                    return i
            else:
                if all(values[i] < values[i-j] for j in range(1, window+1)) and \
                   all(values[i] < values[i+j] for j in range(1, window+1)):
                    return i
        return None

    curr = len(close) - 1

    if type == 'top':
        # 顶背离：价格创新高但 DIF 未创新高
        p1 = find_last_significant_extrema(close, is_max=True, window=5)
        if p1 is not None and curr - p1 >= 5:
            if close[curr] > close[p1] and dif[curr] < dif[p1]:
                # 确保 MACD 柱没有走强且 DIF 没有上升
                if macd_hist[curr] > macd_hist[curr-1]:
                    return False
                if dif[curr] > dif[curr-1]:
                    return False
                if dif[curr] > 0 and dif[curr-1] <= 0:
                    return False
                return True
    else:
        # 底背离：价格创新低但 DIF 未创新低
        p1 = find_last_significant_extrema(close, is_max=False, window=5)
        if p1 is not None and curr - p1 >= 5:
            if close[curr] < close[p1] and dif[curr] > dif[p1]:
                # 确保 MACD 柱没有走弱且 DIF 没有下降
                if macd_hist[curr] < macd_hist[curr-1]:
                    return False
                if dif[curr] < dif[curr-1]:
                    return False
                return True
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

def judge_sell(stock_name, judge_sell_ids, all_data):
    """
    判断卖出信号 (sell_id=1)

    卖出条件按优先级顺序执行：
    1. 清仓：周线顶背离 + 日线死叉
    2. 减半：日线顶背离 + 日线死叉
    3. 阶梯减仓：日线 SAR 跌破，根据周线 RSI 决定卖出比例

    Args:
        stock_name: 股票名称
        judge_sell_ids: 卖出策略 ID 列表
        all_data: 包含 weekly 和 daily 数据的字典
    """
    messages = []
    for sell_id in judge_sell_ids:
        if sell_id == 1:
            data_weekly = all_data.get('weekly')
            data_daily = all_data.get('daily')

            if data_weekly is None or data_daily is None:
                continue

            # 获取最近 3 日周线 RSI 最大值
            max_weekly_rsi = data_weekly['RSI'].tail(3).max()

            # 1. 清仓信号：周线顶背离 + 日线死叉
            if detect_divergence(data_weekly, 'top') and is_death_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (条件 1-清仓): 触发 [周线顶背离 + 日线死叉]，建议清仓")
            # 2. 减半信号：日线顶背离 + 日线死叉
            elif detect_divergence(data_daily, 'top') and is_death_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (条件 1-减半): 触发 [日线顶背离 + 日线死叉]，建议出 1/2")
            # 3. 阶梯式减仓信号：日线 SAR 跌破
            else:
                is_sar_breakdown_recent = False
                for i in range(1, 4):
                    if len(data_daily) >= i+1:
                        if data_daily['Close'].iloc[-i-1] > data_daily['SAR'].iloc[-i-1] and \
                           data_daily['Close'].iloc[-i] < data_daily['SAR'].iloc[-i]:
                            is_sar_breakdown_recent = True
                            break
                if is_sar_breakdown_recent:
                    trigger_reason = f"触发 [日线 SAR 跌破]，当前周线 RSI({max_weekly_rsi:.2f})"
                    if max_weekly_rsi > 90:
                        messages.append(f"【{stock_name}】卖出信号 (条件 1-阶梯): {trigger_reason} > 90，建议卖出全部剩余仓位")
                    elif max_weekly_rsi > 85:
                        messages.append(f"【{stock_name}】卖出信号 (条件 1-阶梯): {trigger_reason} > 85，建议卖出剩余 1/2")
                    elif max_weekly_rsi > 80:
                        messages.append(f"【{stock_name}】卖出信号 (条件 1-阶梯): {trigger_reason} > 80，建议卖出 1/3")
    return messages

def judge_buy(stock_name, judge_buy_ids, all_data, get_index_data_func=None):
    """
    判断买入信号

    买入条件：
    - buy_id=1: 日线 RSI<20 且 周线 RSI<25 (保守型)
    - buy_id=2: (日线 RSI<25 且 周线 RSI<30) 或 (日线底背离 + 日线金叉) (标准型)
    - buy_id=3: (日线 RSI<25 且 周线 RSI<30) 或 (日线底背离 + 日线金叉) 或 (指数 RSI<25) (指数增强型)
    - buy_id=4: 日线 RSI<20 且 周线 RSI<20 (极度保守型)

    Args:
        stock_name: 股票名称
        judge_buy_ids: 买入策略 ID 列表
        all_data: 包含 weekly 和 daily 数据的字典
        get_index_data_func: 获取指数数据的函数
    """
    messages = []
    for buy_id in judge_buy_ids:
        data_weekly = all_data.get('weekly')
        data_daily = all_data.get('daily')

        if data_weekly is None or data_daily is None:
            continue

        min_daily_rsi = data_daily['RSI'].tail(3).min()
        min_weekly_rsi = data_weekly['RSI'].tail(3).min()

        if buy_id == 1:
            # 保守型：日线 RSI<20 且 周线 RSI<25
            if min_daily_rsi < 20 and min_weekly_rsi < 25:
                messages.append(f"【{stock_name}】买入信号 (条件 1-保守): 触发 [日线 RSI({min_daily_rsi:.2f})<20 且 周线 RSI({min_weekly_rsi:.2f})<25]")

        elif buy_id == 2:
            # 标准型：(日线 RSI<25 且 周线 RSI<30) 或 (日线底背离 + 日线金叉)
            if min_daily_rsi < 25 and min_weekly_rsi < 30:
                messages.append(f"【{stock_name}】买入信号 (条件 2-标准): 触发 [日线 RSI({min_daily_rsi:.2f})<25 且 周线 RSI({min_weekly_rsi:.2f})<30]")
            elif detect_divergence(data_daily, 'bottom') and is_golden_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】买入信号 (条件 2-标准): 触发 [日线底背离 + 日线金叉]")

        elif buy_id == 3:
            # 指数增强型：(日线 RSI<25 且 周线 RSI<30) 或 (日线底背离 + 日线金叉) 或 (指数 RSI<25)
            if min_daily_rsi < 25 and min_weekly_rsi < 30:
                messages.append(f"【{stock_name}】买入信号 (条件 3-增强): 触发 [日线 RSI({min_daily_rsi:.2f})<25 且 周线 RSI({min_weekly_rsi:.2f})<30]")
            elif detect_divergence(data_daily, 'bottom') and is_golden_cross(data_daily, window=3):
                messages.append(f"【{stock_name}】买入信号 (条件 3-增强): 触发 [日线底背离 + 日线金叉]")
            elif get_index_data_func:
                for idx_ticker, idx_name in [('000001', '上证指数'), ('399006', '创业板指')]:
                    idx_data = get_index_data_func(idx_ticker)
                    if idx_data is not None and not idx_data.empty and 'RSI' in idx_data.columns:
                        idx_min_rsi = idx_data['RSI'].tail(3).min()
                        if idx_min_rsi < 25:
                            messages.append(f"【{stock_name}】买入信号 (条件 3-增强): 触发 [{idx_name} 日线 RSI({idx_min_rsi:.2f})<25]")
                            break

        elif buy_id == 4:
            # 极度保守型：日线 RSI<20 且 周线 RSI<20
            if min_daily_rsi < 20 and min_weekly_rsi < 20:
                messages.append(f"【{stock_name}】买入信号 (条件 4-极度超卖): 触发 [日线 RSI({min_daily_rsi:.2f})<20 且 周线 RSI({min_weekly_rsi:.2f})<20]")

    return messages
