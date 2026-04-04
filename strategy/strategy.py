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

    # 4. 计算 5 日均线和 10 日均线 (用于替代 SAR 和死叉判断)
    data['MA5'] = talib.SMA(data['Close'], timeperiod=5)
    data['MA10'] = talib.SMA(data['Close'], timeperiod=10)

    return data

def detect_divergence(data, type='top', grace_bars=None, confirm_sar=False):
    """
    检测背离

    Args:
        data: 包含 MACD 指标的数据
        type: 'top' 顶背离，'bottom' 底背离
        grace_bars: 允许信号在最近 N 根 K 线内持续有效（含当根）。None 表示不限制
        confirm_sar: 是否需要 SAR 跌破来确认顶背离有效（仅用于顶背离）

    Returns:
        bool: 如果检测到背离返回 True，否则返回 False
    """
    result = _detect_divergence_impl(data, type, grace_bars, confirm_sar)
    return result if result is False else True

def get_divergence_details(data, type='top'):
    """
    获取背离的详细信息

    Args:
        data: 包含 MACD 指标的数据
        type: 'top' 顶背离，'bottom' 底背离

    Returns:
        dict: 包含背离详细信息的字典，包括：
              - detected: 是否检测到背离
              - recent_high/low: 最近高点/低点的价格
              - prev_high/low: 前一个高点/低点的价格
              - recent_hist: 最近高点的 MACD 柱值
              - prev_hist: 前一个高点的 MACD 柱值
              - interval: 两个高点之间的间隔
              - trough_low: 波谷的低点价格（用于确认前高是局部高点）
        如果没有检测到背离，返回 None
    """
    return _detect_divergence_impl(data, type, None, False)

def _detect_divergence_impl(data, type='top', grace_bars=None, confirm_sar=False):
    """
    检测背离的内部实现

    顶背离判断标准：
    1. 股价创新高（High 创新高，不是 Close）
    2. MACD 柱缩短：红柱缩短 或 绿柱变长（hist 下降）
    3. 两个高点之间有明顯回调（确保是两个独立波段）
    4. 5 日均线跌破 10 日均线：确认信号（在 gm_backtest.py 中检查）

    Returns:
        bool 或 dict: 如果检测到背离返回 True（或详细信息字典），否则返回 False
    """
    if data is None or len(data) < 30:
        return False

    close = data['Close'].values
    high = data['High'].values if 'High' in data.columns else close
    low = data['Low'].values if 'Low' in data.columns else close
    dif = data['MACD'].values  # DIF
    macd_hist = data['MACD_hist'].values  # MACD 柱

    curr = len(close) - 1

    if type == 'top':
        # ===== 顶背离检测 =====
        # 步骤 1：找到最近 5 天内的最高点（最近高点）
        recent_lookback = 5
        recent_start = max(curr - recent_lookback + 1, 0)
        recent_max_idx = recent_start + high[recent_start:curr+1].argmax()
        recent_high = high[recent_max_idx]

        # 步骤 2：在 recent_max_idx 之前找前一个高点
        # 要求：两个高点之间有明显回调（至少有一个波谷）
        prev_lookback = 30
        prev_end = recent_max_idx - 1

        if prev_end < 10:
            return False

        # 在 prev_end 之前 10-30 天范围内找最高价作为前一个高点
        prev_start = max(prev_end - prev_lookback + 1, 0)
        prev_idx = prev_start + high[prev_start:prev_end+1].argmax()
        prev_high = high[prev_idx]

        # 检查两个高点之间是否有回调（波谷）
        # 在 prev_idx 和 recent_max_idx 之间找最低点
        trough_search_start = prev_idx + 1
        trough_search_end = recent_max_idx - 1
        if trough_search_end <= trough_search_start:
            # 没有足够的空间找波谷
            return False

        trough_idx = trough_search_start + low[trough_search_start:trough_search_end+1].argmin()
        trough_low = low[trough_idx]

        # 确保波谷比两个高点都低（形成真正的回调）
        if trough_low >= prev_high or trough_low >= recent_high:
            return False

        # 确保 prev_idx 处的 MACD 柱不是 nan
        if pd.isna(macd_hist[prev_idx]) or pd.isna(macd_hist[recent_max_idx]):
            return False

        # 确保两个高点之间有足够的间隔（至少 5 根 K 线）
        interval = recent_max_idx - prev_idx
        if interval < 5:
            return False

        # 步骤 3：检查顶背离条件
        # 1. 股价创新高
        # 2. MACD 柱缩短（hist 下降）
        price_innovates = recent_high > prev_high
        macd_declines = macd_hist[recent_max_idx] < macd_hist[prev_idx]

        if price_innovates and macd_declines:
            # 检测到顶背离
            if confirm_sar:
                sar = data['SAR'].values if 'SAR' in data.columns else None
                if sar is not None:
                    if close[curr] < sar[curr]:
                        return {
                            'detected': True,
                            'recent_high': recent_high,
                            'prev_high': prev_high,
                            'recent_hist': macd_hist[recent_max_idx],
                            'prev_hist': macd_hist[prev_idx],
                            'interval': interval,
                            'trough_low': trough_low
                        }
                return False

            if grace_bars is None:
                return {
                    'detected': True,
                    'recent_high': recent_high,
                    'prev_high': prev_high,
                    'recent_hist': macd_hist[recent_max_idx],
                    'prev_hist': macd_hist[prev_idx],
                    'interval': interval,
                    'trough_low': trough_low
                }

            if (curr - recent_max_idx) <= (grace_bars - 1):
                return {
                    'detected': True,
                    'recent_high': recent_high,
                    'prev_high': prev_high,
                    'recent_hist': macd_hist[recent_max_idx],
                    'prev_hist': macd_hist[prev_idx],
                    'interval': interval,
                    'trough_low': trough_low
                }
            return False
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
                return {
                    'detected': True,
                    'recent_low': price[recent_min_idx],
                    'prev_low': price[prev_idx],
                    'recent_hist': macd_hist[recent_min_idx],
                    'prev_hist': macd_hist[prev_idx],
                    'interval': recent_min_idx - prev_idx
                }
            if (curr - recent_min_idx) <= (grace_bars - 1):
                return {
                    'detected': True,
                    'recent_low': price[recent_min_idx],
                    'prev_low': price[prev_idx],
                    'recent_hist': macd_hist[recent_min_idx],
                    'prev_hist': macd_hist[prev_idx],
                    'interval': recent_min_idx - prev_idx
                }
            return False
    return False

def is_ma5_breakdown(data, window=3):
    """
    检测 5 日均线跌破 10 日均线 (替代死叉和 SAR 跌破判断)

    Args:
        data: 包含 MA5 和 MA10 的数据
        window: 检测最近 N 天内是否有跌破
    """
    if data is None or len(data) < window + 1:
        return False
    if 'MA5' not in data.columns or 'MA10' not in data.columns:
        return False
    for i in range(1, window + 1):
        if data['MA5'].iloc[-i-1] > data['MA10'].iloc[-i-1] and \
           data['MA5'].iloc[-i] < data['MA10'].iloc[-i]:
            return True
    return False


def find_ma5_breakdown_date(data, window=3):
    """
    查找 5 日均线跌破 10 日均线的日期

    Args:
        data: 包含 MA5 和 MA10 的数据
        window: 检测最近 N 天内是否有跌破

    Returns:
        跌破发生的日期索引，如果没有则返回 None
    """
    if data is None or len(data) < window + 1:
        return None
    if 'MA5' not in data.columns or 'MA10' not in data.columns:
        return None
    for i in range(1, window + 1):
        if len(data) >= i + 1:
            ma5_prev, ma10_prev = data['MA5'].iloc[-i-1], data['MA10'].iloc[-i-1]
            ma5_curr, ma10_curr = data['MA5'].iloc[-i], data['MA10'].iloc[-i]
            if ma5_prev > ma10_prev and ma5_curr < ma10_curr:
                try:
                    return data.index[-i]
                except Exception:
                    return -i
    return None


def is_death_cross(data, window=3):
    """
    检测 MACD 死叉 (DIF 下穿 DEA)
    注：此函数已废弃，改用 is_ma5_breakdown 判断 5 日均线跌破 10 日均线

    Args:
        data: 包含 MACD 和 MACD_signal 的数据
        window: 检测最近 N 天内是否有死叉

    Returns:
        bool: 返回 is_ma5_breakdown 的结果（5 日均线跌破 10 日均线）
    """
    # 改用 5 日均线跌破 10 日均线判断
    return is_ma5_breakdown(data, window=window)


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

    RSI 等级规则：
    - flag=0: RSI <= 80 (正常状态)
    - flag=1: 80 < RSI <= 85 (周线 RSI 超买，一级警戒)
    - flag=2: 85 < RSI <= 90 (周线 RSI 严重超买，二级警戒)
    - flag=3: RSI > 90 (周线 RSI 极度超买，三级警戒/清仓信号)

    Args:
        max_rsi: 周线 RSI 峰值（从上次 5 日均线跌破 10 日均线后到当前的最大值）

    Returns:
        int: RSI flag 等级 (0/1/2/3)
    """
    if max_rsi > 90:
        return 3
    elif max_rsi > 85:
        return 2
    elif max_rsi > 80:
        return 1
    return 0


def update_rsi_flag_v2(current_rsi_flag, current_weekly_rsi, rsi_peak_after_breakdown,
                        ma5_breakdown=False, rsi_drop_2pct=False, rsi_rise_10pct=False):
    """
    更新 RSI flag 状态（增强版：支持位置标记逻辑）

    更新规则：
    1. 初始化：rsi_flag = 0
    2. 周线 RSI > 80: rsi_flag = 1
    3. 周线 RSI > 85: rsi_flag = 2
    4. 周线 RSI > 90 后下跌 2%: 标记为位置 4（局部低点）
    5. 从位置 4 再上涨 10%: 标记为位置 5（反弹高点）
    6. 当 5 日均线跌破 10 日均线时：rsi_flag = 0 (重置)

    位置说明：
    - 位置 1: 局部高点（RSI>90 时的股价高点）
    - 位置 2: 局部低点（从位置 1 下跌后的低点）
    - 位置 3: 不是局部高点（因为从位置 1 只下跌了 2%，不满足显著回调）
    - 位置 4: RSI>90 后下跌 2% 的位置（确认局部低点）
    - 位置 5: 从位置 4 上涨 10% 的位置（反弹高点，用于顶背离判断）

    Args:
        current_rsi_flag: 当前的 rsi_flag 值
        current_weekly_rsi: 当前周线 RSI 值
        rsi_peak_after_breakdown: 上次均线跌破后的周线 RSI 峰值
        ma5_breakdown: 是否发生了 5 日均线跌破 10 日均线
        rsi_drop_2pct: RSI 从峰值下跌了 2%（用于标记位置 4）
        rsi_rise_10pct: RSI 从位置 4 上涨了 10%（用于标记位置 5）

    Returns:
        int: 更新后的 rsi_flag
    """
    # 如果发生 5 日均线跌破 10 日均线，重置为 0
    if ma5_breakdown:
        return 0

    # 根据当前周线 RSI 值更新 flag
    return get_rsi_flag_level(current_weekly_rsi)


def detect_weekly_top_divergence(data_weekly, rsi_state):
    """
    检测周线顶背离形成

    顶背离判断逻辑：
    1. 当前最高点股价比之前局部最高点股价高
    2. 且 MACD 红柱子比之前局部最高点 MACD 柱子短
    3. 此时认为顶背离形成，diverse_week_flag = 1

    假背离判断（顶背离失效）：
    1. 后续股价继续新高
    2. 且 MACD 柱子比之前局部最高点的 MACD 高
    3. 此时认为之前顶背离失效（假背离），diverse_week_flag = 0

    顶背离生效：
    1. 5 日均线跌破 10 日均线
    2. 此时认为顶背离生效，diverse_week_flag = 0，同时卖出规定比例的金额

    Args:
        data_weekly: 周线数据（包含 High, MACD_hist 等）
        rsi_state: RSI 状态字典，包含：
                   - diverse_week_flag: 周线顶背离标志 (0=无背离，1=背离形成)
                   - prev_high_price: 前一个局部高点的价格
                   - prev_high_macd: 前一个局部高点的 MACD 柱值
                   - prev_high_idx: 前一个局部高点的索引
                   - position_1_price: 位置 1 的价格（局部高点）
                   - position_2_price: 位置 2 的价格（局部低点）
                   - position_4_price: 位置 4 的价格（RSI>90 后下跌 2%）
                   - position_5_price: 位置 5 的价格（从位置 4 上涨 10%）
                   - rsi_peak: RSI 峰值（用于判断下跌 2%）
                   - rsi_trough: RSI 谷值（用于判断上涨 10% 的起点）

    Returns:
        tuple: (diverse_week_flag, is_valid_divergence, sell_signal, sell_fraction, messages)
               - diverse_week_flag: 更新后的顶背离标志
               - is_valid_divergence: 是否为有效背离
               - sell_signal: 是否触发卖出
               - sell_fraction: 卖出比例
               - messages: 卖出信号消息
    """
    messages = []
    sell_signal = False
    sell_fraction = 0.0
    diverse_week_flag = rsi_state.get('diverse_week_flag', 0)
    prev_high_price = rsi_state.get('prev_high_price', 0)
    prev_high_macd = rsi_state.get('prev_high_macd', 0)
    prev_high_idx = rsi_state.get('prev_high_idx', None)
    position_1_price = rsi_state.get('position_1_price', 0)
    position_4_price = rsi_state.get('position_4_price', 0)
    rsi_peak = rsi_state.get('rsi_peak', 0)

    if data_weekly is None or len(data_weekly) < 30:
        return diverse_week_flag, False, sell_signal, sell_fraction, messages

    high = data_weekly['High'].values if 'High' in data_weekly.columns else data_weekly['Close'].values
    macd_hist = data_weekly['MACD_hist'].values if 'MACD_hist' in data_weekly.columns else None
    close = data_weekly['Close'].values

    curr = len(close) - 1

    # ===== 顶背离形成判断 =====
    if diverse_week_flag == 0:
        # 寻找最近 5 周内的最高点
        recent_lookback = 5
        recent_start = max(curr - recent_lookback + 1, 0)
        recent_max_idx = recent_start + high[recent_start:curr+1].argmax()
        current_high = high[recent_max_idx]

        # 如果已有前一个高点记录，检查是否形成顶背离
        if prev_high_price > 0 and prev_high_idx is not None:
            # 确保两个高点之间有足够的间隔（至少 5 周）
            interval = recent_max_idx - prev_high_idx

            if interval >= 5:
                # 顶背离条件：
                # 1. 当前最高点股价 > 之前局部最高点股价
                # 2. 当前 MACD 红柱 < 之前局部最高点 MACD 红柱
                price_higher = current_high > prev_high_price

                if macd_hist is not None:
                    current_macd = macd_hist[recent_max_idx]
                    macd_lower = current_macd < prev_high_macd

                    if price_higher and macd_lower:
                        # 顶背离形成
                        diverse_week_flag = 1
                        messages.append(f"周线顶背离形成：股价{current_high:.2f} > {prev_high_price:.2f}, MACD 柱{current_macd:.4f} < {prev_high_macd:.4f}")

    # ===== 假背离判断（顶背离失效） =====
    elif diverse_week_flag == 1:
        # 寻找最近 5 周内的最高点
        recent_lookback = 5
        recent_start = max(curr - recent_lookback + 1, 0)
        recent_max_idx = recent_start + high[recent_start:curr+1].argmax()
        current_high = high[recent_max_idx]
        current_macd = macd_hist[recent_max_idx] if macd_hist is not None else 0

        # 假背离条件：
        # 1. 股价继续创新高
        # 2. MACD 柱子比之前局部最高点高
        if macd_hist is not None and current_high > prev_high_price and current_macd > prev_high_macd:
            # 顶背离失效（假背离）
            diverse_week_flag = 0
            messages.append(f"周线顶背离失效（假背离）：股价{current_high:.2f} > {prev_high_price:.2f}, MACD 柱{current_macd:.4f} > {prev_high_macd:.4f}")

    # ===== 顶背离生效判断（5 日均线跌破 10 日均线） =====
    if diverse_week_flag == 1:
        # 检查是否发生 5 日均线跌破 10 日均线
        if 'MA5' in data_weekly.columns and 'MA10' in data_weekly.columns:
            ma5_curr = data_weekly['MA5'].iloc[-1]
            ma10_curr = data_weekly['MA10'].iloc[-1]
            if len(data_weekly) >= 2:
                ma5_prev = data_weekly['MA5'].iloc[-2]
                ma10_prev = data_weekly['MA10'].iloc[-2]

                # 检测下穿动作
                if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                    # 顶背离生效，触发卖出
                    sell_signal = True
                    sell_fraction = 0.5  # 卖出 50%（可根据配置调整）
                    diverse_week_flag = 0  # 重置标志
                    messages.append(f"周线顶背离生效：5 周均线下穿 10 周均线，建议卖出{sell_fraction*100:.0f}%")

    return diverse_week_flag, (diverse_week_flag == 1), sell_signal, sell_fraction, messages


def update_rsi_position_state(rsi_state, current_rsi, current_price, ma5_breakdown=False):
    """
    更新 RSI 位置状态（位置 1-5 的标记）

    位置标记规则：
    - 位置 1: RSI>90 时的股价高点（局部高点）
    - 位置 2: 从位置 1 下跌后的低点（局部低点）
    - 位置 3: 不是局部高点（只下跌了 2%，不满足显著回调）
    - 位置 4: RSI>90 后下跌 2% 的位置
    - 位置 5: 从位置 4 上涨 10% 的位置

    Args:
        rsi_state: RSI 状态字典
        current_rsi: 当前周线 RSI 值
        current_price: 当前股价
        ma5_breakdown: 是否发生 5 日均线跌破 10 日均线

    Returns:
        dict: 更新后的 rsi_state
    """
    position = rsi_state.get('current_position', 0)
    rsi_peak = rsi_state.get('rsi_peak', 0)
    rsi_trough = rsi_state.get('rsi_trough', 0)
    price_at_peak = rsi_state.get('price_at_peak', 0)

    # 如果发生均线跌破，重置所有状态
    if ma5_breakdown:
        return {
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
            'diverse_week_flag': 0
        }

    # 状态机转换
    if position == 0:
        # 等待 RSI>90
        if current_rsi > 90:
            position = 1
            rsi_peak = current_rsi
            price_at_peak = current_price
            rsi_state['position_1_price'] = current_price  # 位置 1：局部高点

    elif position == 1:
        # RSI 从峰值下跌，等待下跌 2%
        if current_rsi < rsi_peak:
            drop_pct = (rsi_peak - current_rsi) / rsi_peak * 100
            if drop_pct >= 2:
                position = 4  # 位置 4：下跌 2%
                rsi_state['position_4_price'] = current_price
                rsi_state['rsi_trough'] = current_rsi

    elif position == 4:
        # 从位置 4 上涨 10%
        rsi_trough = rsi_state.get('rsi_trough', current_rsi)
        if current_rsi > rsi_trough:
            rise_pct = (current_rsi - rsi_trough) / rsi_trough * 100
            if rise_pct >= 10:
                position = 5  # 位置 5：上涨 10%
                rsi_state['position_5_price'] = current_price
                # 位置 5 可以作为新的前高用于顶背离判断
                rsi_state['prev_high_price'] = current_price

    rsi_state['current_position'] = position
    rsi_state['rsi_peak'] = max(rsi_peak, current_rsi)

    # 更新 RSI 峰值时的价格
    if current_rsi >= rsi_peak:
        rsi_state['price_at_peak'] = current_price

    return rsi_state


def update_rsi_flag(current_rsi_flag, current_weekly_rsi, ma5_breakdown=False):
    """
    更新 RSI flag 状态

    更新规则：
    1. 初始化：rsi_flag = 0
    2. 周线 RSI > 80: rsi_flag = 1
    3. 周线 RSI > 85: rsi_flag = 2
    4. 周线 RSI > 90: rsi_flag = 3
    5. 当 5 日均线跌破 10 日均线时：rsi_flag = 0 (重置)

    Args:
        current_rsi_flag: 当前的 rsi_flag 值
        current_weekly_rsi: 当前周线 RSI 值
        ma5_breakdown: 是否发生了 5 日均线跌破 10 日均线

    Returns:
        int: 更新后的 rsi_flag
    """
    # 如果发生 5 日均线跌破 10 日均线，重置为 0
    if ma5_breakdown:
        return 0

    # 根据当前周线 RSI 值更新 flag
    return get_rsi_flag_level(current_weekly_rsi)

def get_sell_fraction_by_flag(rsi_flag):
    """
    根据 RSI flag 获取卖出比例

    RSI flag 与卖出比例对应关系：
    - flag=0: 无卖出信号 (RSI <= 80)
    - flag=1: 卖出 1/3 (80 < RSI <= 85)
    - flag=2: 卖出 1/2 (85 < RSI <= 90)
    - flag=3: 清仓 (RSI > 90)

    Args:
        rsi_flag: RSI flag 等级 (0/1/2/3)

    Returns:
        tuple: (卖出比例，卖出建议描述)
    """
    if rsi_flag == 3:
        return 1.0, "建议卖出全部剩余仓位"
    elif rsi_flag == 2:
        return 0.5, "建议卖出剩余 1/2"
    elif rsi_flag == 1:
        return 1.0/3.0, "建议卖出 1/3"
    return 0.0, None


def judge_sell(stock_name, judge_sell_ids, all_data, rsi_flag=None, rsi_peak_map=None,
               rsi_state=None, check_weekly_divergence=True):
    """
    判断卖出信号

    卖出策略 1 (sell_id=1):
    1. 减仓 1/3：日线顶背离 + 5 日均线下穿 10 日均线
    2. 清仓：周线顶背离 + 5 日均线下穿 10 日均线
    3. 阶梯减仓：5 日均线下穿 10 日均线，根据周线 RSI flag 决定卖出比例

    卖出策略 2 (sell_id=2):
    1. 减仓 1/3：日线顶背离 + 5 日均线下穿 10 日均线
    2. 清仓：周线顶背离 + 5 日均线下穿 10 日均线
    3. 阶梯减仓：5 日均线下穿 10 日均线，根据周线 RSI flag 决定卖出比例

    RSI flag 更新规则：
    - 初始化：rsi_flag = 0
    - 周线 RSI > 80: rsi_flag = 1
    - 周线 RSI > 85: rsi_flag = 2
    - 周线 RSI > 90: rsi_flag = 3
    - 当 5 日均线跌破 10 日均线时：rsi_flag = 0 (重置)

    周线顶背离判断逻辑：
    - 当前最高点股价比之前局部最高点股价高，且 MACD 红柱子比之前局部最高点 MACD 柱子短
      → 顶背离形成，diverse_week_flag = 1
    - 后续股价继续新高且 MACD 柱子比之前局部最高点高
      → 顶背离失效（假背离），diverse_week_flag = 0
    - 5 日均线跌破 10 日均线
      → 顶背离生效，diverse_week_flag = 0，同时卖出规定比例的金额

    Args:
        stock_name: 股票名称
        judge_sell_ids: 卖出策略 ID 列表
        all_data: 包含 weekly, daily, 120min 数据的字典
        rsi_flag: 可选，外部传入的 RSI flag 状态 (0/1/2/3)
        rsi_peak_map: 可选，外部传入的 RSI 峰值字典 {'max_rsi': float, 'ma5_breakdown': bool}
                      用于更精确地控制 RSI flag 状态
        rsi_state: 可选，RSI 状态字典，用于跟踪位置 1-5 和周线顶背离状态
        check_weekly_divergence: 是否检查周线顶背离（默认 True）

    Returns:
        messages: 卖出信号列表
        rsi_flag: 当前的 RSI flag 状态（如果均线跌破则返回 0，否则保持原值或根据 RSI 更新）
        sar_breakdown: 是否发生了 5 日均线下穿 10 日均线
        rsi_state: 更新后的 RSI 状态字典（仅当传入 rsi_state 时返回）
    """
    messages = []
    data_weekly = all_data.get('weekly')
    sar_breakdown = False  # 保持变量名兼容，实际表示 5 日均线下穿 10 日均线

    # 初始化 RSI 状态字典（如果传入了 rsi_state）
    return_rsi_state = None
    if rsi_state is not None:
        return_rsi_state = rsi_state.copy()
    else:
        return_rsi_state = {
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
            'diverse_week_flag': 0
        }

    # 检查是否发生了 5 日均线跌破 10 日均线
    data_daily = all_data.get('daily')
    ma5_breakdown_now = False
    if data_daily is not None and 'MA5' in data_daily.columns and 'MA10' in data_daily.columns:
        ma5_curr = data_daily['MA5'].iloc[-1]
        ma10_curr = data_daily['MA10'].iloc[-1]
        if len(data_daily) >= 2:
            ma5_prev = data_daily['MA5'].iloc[-2]
            ma10_prev = data_daily['MA10'].iloc[-2]
            if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                ma5_breakdown_now = True

    # 如果传入了 rsi_peak_map，使用峰值计算 flag
    if rsi_peak_map is not None:
        max_rsi = rsi_peak_map.get('max_rsi', 0)
        override_breakdown = rsi_peak_map.get('ma5_breakdown', False)
        if override_breakdown:
            rsi_flag = 0
            # 均线跌破，重置 RSI 状态
            return_rsi_state = update_rsi_position_state(return_rsi_state, 0, 0, ma5_breakdown=True)
        else:
            rsi_flag = get_rsi_flag_level(max_rsi)
    elif rsi_flag is None:
        # 如果没有传入 rsi_flag 和 rsi_peak_map，则根据当前周线 RSI 值计算
        if data_weekly is not None and 'RSI' in data_weekly.columns and len(data_weekly) > 0:
            current_rsi = data_weekly['RSI'].iloc[-1]
            rsi_flag = get_rsi_flag_level(current_rsi)
            # 更新 RSI 位置状态
            current_price = data_weekly['Close'].iloc[-1] if 'Close' in data_weekly.columns else 0
            return_rsi_state = update_rsi_position_state(return_rsi_state, current_rsi, current_price, ma5_breakdown_now)
        else:
            rsi_flag = 0

    for sell_id in judge_sell_ids:
        data_monthly = all_data.get('monthly')
        data_120min = all_data.get('120min')

        if data_weekly is None or data_daily is None:
            continue

        # ===== 周线顶背离检查（新增逻辑） =====
        if check_weekly_divergence and data_weekly is not None:
            # 更新前一个高点信息（用于顶背离判断）
            high_arr = data_weekly['High'].values if 'High' in data_weekly.columns else data_weekly['Close'].values
            macd_hist_arr = data_weekly['MACD_hist'].values if 'MACD_hist' in data_weekly.columns else None
            curr = len(high_arr) - 1

            # 寻找最近 5 周内的最高点
            recent_lookback = 5
            recent_start = max(curr - recent_lookback + 1, 0)
            recent_max_idx = recent_start + high_arr[recent_start:curr+1].argmax()
            current_high = high_arr[recent_max_idx]
            current_macd = macd_hist_arr[recent_max_idx] if macd_hist_arr is not None else 0

            # 如果还没有记录前一个高点，且当前 MACD 为正，记录下来
            if return_rsi_state.get('prev_high_price', 0) == 0 and current_macd > 0:
                return_rsi_state['prev_high_price'] = current_high
                return_rsi_state['prev_high_macd'] = current_macd
                return_rsi_state['prev_high_idx'] = recent_max_idx

            # 检查周线顶背离状态
            diverse_week_flag = return_rsi_state.get('diverse_week_flag', 0)
            prev_high_price = return_rsi_state.get('prev_high_price', 0)
            prev_high_macd = return_rsi_state.get('prev_high_macd', 0)
            prev_high_idx = return_rsi_state.get('prev_high_idx', None)

            if diverse_week_flag == 0 and prev_high_price > 0 and prev_high_idx is not None:
                # 检查是否形成顶背离
                interval = recent_max_idx - prev_high_idx
                if interval >= 5 and current_high > prev_high_price:
                    if macd_hist_arr is not None and current_macd < prev_high_macd:
                        # 顶背离形成
                        diverse_week_flag = 1
                        return_rsi_state['diverse_week_flag'] = diverse_week_flag
                        messages.append(f"【{stock_name}】周线顶背离形成：股价{current_high:.2f} > {prev_high_price:.2f}, MACD 柱{current_macd:.4f} < {prev_high_macd:.4f}")

            elif diverse_week_flag == 1:
                # 检查是否假背离（顶背离失效）
                if macd_hist_arr is not None and current_high > prev_high_price and current_macd > prev_high_macd:
                    # 顶背离失效（假背离）
                    diverse_week_flag = 0
                    return_rsi_state['diverse_week_flag'] = 0
                    return_rsi_state['prev_high_price'] = current_high
                    return_rsi_state['prev_high_macd'] = current_macd
                    return_rsi_state['prev_high_idx'] = recent_max_idx
                    messages.append(f"【{stock_name}】周线顶背离失效（假背离）：股价{current_high:.2f} > {prev_high_price:.2f}, MACD 柱{current_macd:.4f} > {prev_high_macd:.4f}")

                # 检查是否 5 日均线跌破 10 日均线（顶背离生效）
                if 'MA5' in data_weekly.columns and 'MA10' in data_weekly.columns:
                    ma5_weekly_curr = data_weekly['MA5'].iloc[-1]
                    ma10_weekly_curr = data_weekly['MA10'].iloc[-1]
                    if len(data_weekly) >= 2:
                        ma5_weekly_prev = data_weekly['MA5'].iloc[-2]
                        ma10_weekly_prev = data_weekly['MA10'].iloc[-2]

                        if ma5_weekly_prev >= ma10_weekly_prev and ma5_weekly_curr < ma10_weekly_curr:
                            # 顶背离生效，触发卖出
                            diverse_week_flag = 0
                            return_rsi_state['diverse_week_flag'] = 0
                            sell_fraction = 0.5  # 卖出 50%
                            messages.append(f"【{stock_name}】周线顶背离生效：5 周均线下穿 10 周均线，建议卖出{sell_fraction*100:.0f}%")
                            # 注意：这里不立即添加卖出信号，而是通过后续的均线跌破逻辑处理

        if sell_id == 1:
            # 策略 1: 优先使用 120 分钟均线跌破 (用于实时监控)，如果没有则回退到日线 (用于回测)
            data_cross = data_120min if data_120min is not None else data_daily

            # 1. 清仓信号：周线顶背离 + 120 分钟 (或日线) 5 日均线下穿 10 日均线
            if detect_divergence(data_weekly, 'top') and is_ma5_breakdown(data_cross, window=3):
                cross_name = "120 分钟" if data_120min is not None else "日线"
                messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 触发 [周线顶背离 + {cross_name}5 日均线下穿 10 日均线]，建议清仓")
            # 2. 清仓信号：月线顶背离 + 日线 5 日均线下穿 10 日均线
            elif data_monthly is not None and detect_divergence(data_monthly, 'top') and is_ma5_breakdown(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 触发 [月线顶背离 + 日线 5 日均线下穿 10 日均线]，建议清仓")
            # 2. 减仓信号：日线顶背离 + 120 分钟 (或日线) 5 日均线下穿 10 日均线
            elif detect_divergence(data_daily, 'top', grace_bars=3) and is_ma5_breakdown(data_cross, window=3):
                cross_name = "120 分钟" if data_120min is not None else "日线"
                messages.append(f"【{stock_name}】卖出信号 (策略 1-减仓): 触发 [日线顶背离 + {cross_name}5 日均线下穿 10 日均线]，建议卖出 1/3")
            # 3. 阶梯式减仓信号：5 日均线下穿 10 日均线
            else:
                ma5_breakdown_ts = find_ma5_breakdown_date(data_daily, window=3)
                if ma5_breakdown_ts is not None:
                    sar_breakdown = True
                    # 5 日均线下穿 10 日均线，根据 rsi_flag 决定卖出比例
                    ma5_breakdown_day = ma5_breakdown_ts.strftime('%Y-%m-%d') if hasattr(ma5_breakdown_ts, 'strftime') else str(ma5_breakdown_ts)
                    sell_fraction, sell_msg = get_sell_fraction_by_flag(rsi_flag)
                    if sell_fraction > 0:
                        trigger_reason = f"触发 [日线 5 日均线下穿 10 日均线 ({ma5_breakdown_day})]，周线 RSI flag={rsi_flag}"
                        messages.append(f"【{stock_name}】卖出信号 (策略 1-阶梯): {trigger_reason}，{sell_msg}")
                    # 均线跌破后，flag 重置为 0
                    rsi_flag = 0
                elif ma5_breakdown_now:
                    # 当前发生了均线跌破，同样重置 flag
                    sar_breakdown = True
                    rsi_flag = 0

        elif sell_id == 2:
            # 1. 清仓信号：周线顶背离 + 日线 5 日均线下穿 10 日均线
            if detect_divergence(data_weekly, 'top') and is_ma5_breakdown(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 触发 [周线顶背离 + 日线 5 日均线下穿 10 日均线]，建议清仓")
            # 2. 清仓信号：月线顶背离 + 日线 5 日均线下穿 10 日均线
            elif data_monthly is not None and detect_divergence(data_monthly, 'top') and is_ma5_breakdown(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 触发 [月线顶背离 + 日线 5 日均线下穿 10 日均线]，建议清仓")
            # 2. 减仓信号：日线顶背离 + 日线 5 日均线下穿 10 日均线
            elif detect_divergence(data_daily, 'top', grace_bars=3) and is_ma5_breakdown(data_daily, window=3):
                messages.append(f"【{stock_name}】卖出信号 (策略 2-减仓): 触发 [日线顶背离 + 日线 5 日均线下穿 10 日均线]，建议卖出 1/3")
            # 3. 阶梯式减仓信号：5 日均线下穿 10 日均线
            else:
                ma5_breakdown_ts = find_ma5_breakdown_date(data_daily, window=3)
                if ma5_breakdown_ts is not None:
                    sar_breakdown = True
                    # 5 日均线下穿 10 日均线，根据 rsi_flag 决定卖出比例
                    ma5_breakdown_day = ma5_breakdown_ts.strftime('%Y-%m-%d') if hasattr(ma5_breakdown_ts, 'strftime') else str(ma5_breakdown_ts)
                    sell_fraction, sell_msg = get_sell_fraction_by_flag(rsi_flag)
                    if sell_fraction > 0:
                        trigger_reason = f"触发 [日线 5 日均线下穿 10 日均线 ({ma5_breakdown_day})]，周线 RSI flag={rsi_flag}"
                        messages.append(f"【{stock_name}】卖出信号 (策略 2-阶梯): {trigger_reason}，{sell_msg}")
                    # 均线跌破后，flag 重置为 0
                    rsi_flag = 0
                elif ma5_breakdown_now:
                    # 当前发生了均线跌破，同样重置 flag
                    sar_breakdown = True
                    rsi_flag = 0

    return messages, rsi_flag, sar_breakdown, return_rsi_state

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
    """
    检测日线 SAR 是否被跌破
    注：此函数已废弃，改用 is_ma5_breakdown 判断 5 日均线跌破 10 日均线

    Args:
        data_daily: 日线数据
        window: 检测最近 N 天内是否有跌破

    Returns:
        bool: 返回 is_ma5_breakdown 的结果（5 日均线跌破 10 日均线）
    """
    return is_ma5_breakdown(data_daily, window=window)


def find_sar_breakdown_date(data_daily, window=3):
    """
    查找 SAR 跌破的日期
    注：此函数已废弃，改用 find_ma5_breakdown_date 判断 5 日均线跌破 10 日均线

    Args:
        data_daily: 日线数据
        window: 检测最近 N 天内是否有跌破

    Returns:
        跌破发生的日期索引，如果没有则返回 None
    """
    return find_ma5_breakdown_date(data_daily, window=window)

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
