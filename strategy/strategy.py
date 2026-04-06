import pandas as pd
import talib
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

# 使用 backtest 的 logger，确保日志输出到同一位置
logger = logging.getLogger('backtest')


# ===== 局部高点日志记录器 =====
class LocalHighLogger:
    """
    局部高点日志记录器

    用于收集和存储日线、周线的局部高点信息，
    并在回测结束时生成独立的日志文件。
    """

    def __init__(self, ticker: str, name: str, log_dir: str):
        """
        初始化局部高点日志记录器

        Args:
            ticker: 股票代码
            name: 股票名称
            log_dir: 日志目录路径
        """
        self.ticker = ticker
        self.name = name
        self.log_dir = log_dir

        # 存储局部高点信息
        self.daily_highs: List[Dict] = []      # 日线局部高点列表
        self.weekly_highs: List[Dict] = []     # 周线局部高点列表

        # 存储顶背离事件记录
        self.daily_divergence_events: List[Dict] = []  # 日线顶背离事件
        self.weekly_divergence_events: List[Dict] = [] # 周线顶背离事件

        # 回测时间范围
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def set_backtest_period(self, start_date: str, end_date: str):
        """设置回测时间范围"""
        self.start_date = start_date
        self.end_date = end_date

    def update_daily_highs(self, local_highs: List[Dict], current_date: str):
        """
        更新日线局部高点列表

        Args:
            local_highs: detect_local_highs 返回的局部高点列表
            current_date: 当前日期
        """
        # 通过日期去重，只记录新高点
        existing_dates = {h['date'] for h in self.daily_highs}
        for high in local_highs:
            high_date = str(high.get('date', 'N/A'))[:10]
            if high_date not in existing_dates:
                self.daily_highs.append({
                    'index': high.get('index', 0),
                    'price': high.get('price', 0),
                    'macd_hist': high.get('macd_hist', 0),
                    'date': high_date,
                    'confirm_date': current_date  # 确认日期
                })
                existing_dates.add(high_date)

    def update_weekly_highs(self, local_highs: List[Dict], current_date: str):
        """
        更新周线局部高点列表

        Args:
            local_highs: detect_local_highs 返回的局部高点列表
            current_date: 当前日期
        """
        # 通过日期去重，只记录新高点
        existing_dates = {h['date'] for h in self.weekly_highs}
        for high in local_highs:
            high_date = str(high.get('date', 'N/A'))[:10]
            if high_date not in existing_dates:
                self.weekly_highs.append({
                    'index': high.get('index', 0),
                    'price': high.get('price', 0),
                    'macd_hist': high.get('macd_hist', 0),
                    'date': high_date,
                    'confirm_date': current_date  # 确认日期
                })
                existing_dates.add(high_date)

    def record_daily_divergence(self, event_type: str, date: str, details: Dict):
        """
        记录日线顶背离事件

        Args:
            event_type: 事件类型 ('形成', '生效', '失效')
            date: 事件日期
            details: 事件详情
        """
        self.daily_divergence_events.append({
            'type': event_type,
            'date': date,
            'details': details
        })

    def record_weekly_divergence(self, event_type: str, date: str, details: Dict):
        """
        记录周线顶背离事件

        Args:
            event_type: 事件类型 ('形成', '生效', '失效')
            date: 事件日期
            details: 事件详情
        """
        self.weekly_divergence_events.append({
            'type': event_type,
            'date': date,
            'details': details
        })

    def save_log_file(self):
        """
        生成并保存日志文件

        日志文件名格式：股票编号.log
        """
        log_file_path = os.path.join(self.log_dir, f"{self.ticker}.log")

        with open(log_file_path, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write(f"=== {self.name}（{self.ticker}）局部高点记录 ===\n")
            f.write(f"回测时间：{self.start_date or 'N/A'} ~ {self.end_date or 'N/A'}\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            # 写入日线局部高点列表
            f.write("=" * 60 + "\n")
            f.write("=== 日线局部高点列表 ===\n")
            f.write("=" * 60 + "\n")
            if self.daily_highs:
                f.write(f"序号    日期        价格        MACD柱值      索引    确认日期\n")
                f.write("-" * 70 + "\n")
                for i, high in enumerate(self.daily_highs):
                    f.write(f"{i+1:4d}    {high['date']:12s}  {high['price']:10.3f}  {high['macd_hist']:12.4f}  {high['index']:5d}  {high['confirm_date']}\n")
            else:
                f.write("无日线局部高点记录\n")
            f.write("\n")

            # 写入周线局部高点列表
            f.write("=" * 60 + "\n")
            f.write("=== 周线局部高点列表 ===\n")
            f.write("=" * 60 + "\n")
            if self.weekly_highs:
                f.write(f"序号    日期        价格        MACD柱值      索引    确认日期\n")
                f.write("-" * 70 + "\n")
                for i, high in enumerate(self.weekly_highs):
                    f.write(f"{i+1:4d}    {high['date']:12s}  {high['price']:10.3f}  {high['macd_hist']:12.4f}  {high['index']:5d}  {high['confirm_date']}\n")
            else:
                f.write("无周线局部高点记录\n")
            f.write("\n")

            # 写入日线顶背离事件记录
            f.write("=" * 60 + "\n")
            f.write("=== 日线顶背离事件记录 ===\n")
            f.write("=" * 60 + "\n")
            if self.daily_divergence_events:
                f.write(f"日期          事件类型    详情\n")
                f.write("-" * 70 + "\n")
                for event in self.daily_divergence_events:
                    detail_str = self._format_divergence_details(event['details'])
                    f.write(f"{event['date']:12s}  {event['type']:8s}    {detail_str}\n")
            else:
                f.write("无日线顶背离事件记录\n")
            f.write("\n")

            # 写入周线顶背离事件记录
            f.write("=" * 60 + "\n")
            f.write("=== 周线顶背离事件记录 ===\n")
            f.write("=" * 60 + "\n")
            if self.weekly_divergence_events:
                f.write(f"日期          事件类型    详情\n")
                f.write("-" * 70 + "\n")
                for event in self.weekly_divergence_events:
                    detail_str = self._format_divergence_details(event['details'])
                    f.write(f"{event['date']:12s}  {event['type']:8s}    {detail_str}\n")
            else:
                f.write("无周线顶背离事件记录\n")
            f.write("\n")

            # 写入统计信息
            f.write("=" * 60 + "\n")
            f.write("=== 统计信息 ===\n")
            f.write("=" * 60 + "\n")
            f.write(f"日线局部高点总数：{len(self.daily_highs)}\n")
            f.write(f"周线局部高点总数：{len(self.weekly_highs)}\n")
            f.write(f"日线顶背离事件数：{len(self.daily_divergence_events)}\n")
            f.write(f"周线顶背离事件数：{len(self.weekly_divergence_events)}\n")

        logger.info(f"[{self.ticker}] 局部高点日志已保存至：{log_file_path}")

    def _format_divergence_details(self, details: Dict) -> str:
        """格式化顶背离详情信息"""
        if not details:
            return "无详情"

        parts = []
        if 'prev_high' in details:
            parts.append(f"A价格={details['prev_high']:.3f}")
        if 'recent_high' in details:
            parts.append(f"B价格={details['recent_high']:.3f}")
        if 'prev_macd' in details:
            parts.append(f"A_MACD={details['prev_macd']:.4f}")
        if 'recent_macd' in details:
            parts.append(f"B_MACD={details['recent_macd']:.4f}")
        if 'prev_high_date' in details:
            parts.append(f"A日期={str(details['prev_high_date'])[:10]}")
        if 'recent_high_date' in details:
            parts.append(f"B日期={str(details['recent_high_date'])[:10]}")
        if 'reason' in details:
            parts.append(f"原因={details['reason']}")

        return ", ".join(parts) if parts else "无详情"


# ===== 顶背离计算参数配置（根据 conditions.md） =====
DIVERGENCE_CONFIG = {
    'daily': {
        'drop_threshold': 0.05,      # 局部高点确认跌幅：5%
        'min_interval': 5,           # 最小间隔K线数：5根
        'invalidation_bars': 10,     # 背离失效时间：10天
        'trend_reversal_pct': 0.10,  # 反向趋势失效：价格下跌10%
    },
    'weekly': {
        'drop_threshold': 0.08,      # 局部高点确认跌幅：8%（且需要均线跌破确认）
        'min_interval': 5,           # 最小间隔K线数：5周
        'invalidation_bars': 10,     # 背离失效时间：10周
        'trend_reversal_pct': 0.10,  # 反向趋势失效：价格下跌10%
    }
}

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


def detect_local_highs(data, drop_threshold=0.05):
    """
    检测局部高点（无未来函数实现）

    核心原则：不提前预判，只在价格下跌5%后才确认之前的高点是局部高点。

    检测流程：
    1. 维护一个"候选高点"变量（价格、位置、MACD柱值）
    2. 当价格创新高时，更新候选高点
    3. 当价格从候选高点下跌达到5%时：
       - 确认该候选高点是有效的局部高点
       - 将高点存入局部高点列表
       - 将当前价格设为新的候选高点

    Args:
        data: 包含 High, MACD_hist 的数据
        drop_threshold: 下跌阈值，默认5%

    Returns:
        list: 局部高点列表，每个元素是字典：
              {
                  'index': 高点索引,
                  'price': 高点价格,
                  'macd_hist': 高点MACD柱值,
                  'date': 高点日期（如果有索引）
              }
    """
    if data is None or len(data) < 10:
        return []

    high = data['High'].values if 'High' in data.columns else data['Close'].values
    macd_hist = data['MACD_hist'].values if 'MACD_hist' in data.columns else None
    close = data['Close'].values

    local_highs = []

    # 候选高点初始化
    candidate_high_price = high[0]
    candidate_high_index = 0
    candidate_high_macd = macd_hist[0] if macd_hist is not None else 0

    for i in range(1, len(high)):
        current_high = high[i]

        # 检查是否从候选高点下跌了5%
        drop_pct = (candidate_high_price - close[i]) / candidate_high_price

        if drop_pct >= drop_threshold:
            # 确认候选高点是有效的局部高点
            local_high = {
                'index': candidate_high_index,
                'price': candidate_high_price,
                'macd_hist': candidate_high_macd,
            }
            # 添加日期（如果有）
            if hasattr(data, 'index'):
                try:
                    local_high['date'] = data.index[candidate_high_index]
                except:
                    pass

            local_highs.append(local_high)

            # 重置候选高点为当前K线
            candidate_high_price = high[i]
            candidate_high_index = i
            candidate_high_macd = macd_hist[i] if macd_hist is not None else 0

        elif current_high > candidate_high_price:
            # 价格创新高，更新候选高点
            candidate_high_price = current_high
            candidate_high_index = i
            candidate_high_macd = macd_hist[i] if macd_hist is not None else 0

    # 注意：最后一个候选高点没有被确认（因为还没有下跌5%），不加入列表

    return local_highs


def detect_weekly_local_highs(data_weekly, data_daily, drop_threshold=0.08):
    """
    检测周线局部高点（无未来函数实现）

    根据 conditions.md 文档最新规则：
    - 当价格从候选高点下跌达到 **8%** 且 **5日均线跌破10日均线** 时：
      - 确认该候选高点是有效的周线级别局部高点
      - 将高点存入局部高点列表
      - 将当前价格设为新的周线级别候选高点

    Args:
        data_weekly: 周线数据，包含 High, MACD_hist
        data_daily: 日线数据，包含 MA5, MA10（用于均线跌破判断）
        drop_threshold: 下跌阈值，默认8%

    Returns:
        list: 周线局部高点列表，每个元素是字典：
              {
                  'index': 高点在周线数据中的索引,
                  'price': 高点价格,
                  'macd_hist': 高点MACD柱值,
                  'date': 高点日期,
                  'confirm_date': 确认日期
              }
    """
    if data_weekly is None or len(data_weekly) < 10:
        return []

    if data_daily is None or 'MA5' not in data_daily.columns or 'MA10' not in data_daily.columns:
        # 如果没有日线均线数据，回退到普通的局部高点检测
        return detect_local_highs(data_weekly, drop_threshold)

    high = data_weekly['High'].values if 'High' in data_weekly.columns else data_weekly['Close'].values
    macd_hist = data_weekly['MACD_hist'].values if 'MACD_hist' in data_weekly.columns else None
    close = data_weekly['Close'].values

    local_highs = []

    # 候选高点初始化
    candidate_high_price = high[0]
    candidate_high_index = 0
    candidate_high_macd = macd_hist[0] if macd_hist is not None else 0

    for i in range(1, len(high)):
        current_high = high[i]

        # 检查是否从候选高点下跌了8%
        drop_pct = (candidate_high_price - close[i]) / candidate_high_price

        if drop_pct >= drop_threshold:
            # 价格已下跌8%，现在检查是否有均线跌破确认
            # 获取当前周线对应的日期
            current_weekly_date = None
            try:
                current_weekly_date = data_weekly.index[i]
            except:
                pass

            # 检查日线数据中是否有均线跌破
            ma_breakdown_found = False
            confirm_date = None

            if current_weekly_date is not None:
                # 在日线数据中查找对应时间段内的均线跌破
                # 遍历日线数据，找到该周线日期对应的日线
                for j in range(len(data_daily) - 1, -1, -1):
                    daily_date = data_daily.index[j]
                    # 检查日线日期是否在当前周线日期附近（同一天或之后一周内）
                    if daily_date >= current_weekly_date or j == len(data_daily) - 1:
                        # 检查是否有均线跌破
                        if j >= 1:
                            ma5_prev = data_daily['MA5'].iloc[j-1]
                            ma10_prev = data_daily['MA10'].iloc[j-1]
                            ma5_curr = data_daily['MA5'].iloc[j]
                            ma10_curr = data_daily['MA10'].iloc[j]

                            if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                                ma_breakdown_found = True
                                confirm_date = daily_date
                                break

                        # 向前搜索更多日线（最多5天）
                        for k in range(1, 6):
                            if j + k < len(data_daily):
                                ma5_prev = data_daily['MA5'].iloc[j+k-1]
                                ma10_prev = data_daily['MA10'].iloc[j+k-1]
                                ma5_curr = data_daily['MA5'].iloc[j+k]
                                ma10_curr = data_daily['MA10'].iloc[j+k]

                                if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                                    ma_breakdown_found = True
                                    confirm_date = data_daily.index[j+k]
                                    break

                        if ma_breakdown_found:
                            break

            if ma_breakdown_found:
                # 确认候选高点是有效的周线局部高点
                local_high = {
                    'index': candidate_high_index,
                    'price': candidate_high_price,
                    'macd_hist': candidate_high_macd,
                    'confirm_date': confirm_date
                }

                # 添加日期
                if hasattr(data_weekly, 'index'):
                    try:
                        local_high['date'] = data_weekly.index[candidate_high_index]
                    except:
                        pass

                local_highs.append(local_high)

                # 重置候选高点为当前K线
                candidate_high_price = high[i]
                candidate_high_index = i
                candidate_high_macd = macd_hist[i] if macd_hist is not None else 0

        elif current_high > candidate_high_price:
            # 价格创新高，更新候选高点
            candidate_high_price = current_high
            candidate_high_index = i
            candidate_high_macd = macd_hist[i] if macd_hist is not None else 0

    return local_highs


def detect_divergence_v2(data, timeframe='daily', min_interval=None, drop_threshold=None):
    """
    检测顶背离（新版本，基于局部高点列表）

    根据 conditions.md 文档的顶背离判断逻辑：

    一、局部高点检测：
    - 维护"候选高点"变量
    - 当价格创新高时，更新候选高点
    - 当价格从候选高点下跌≥5%时，确认候选高点是有效的局部高点

    二、顶背离形成条件（需同时满足）：
    1. 存在上一个已确认的局部高点A
    2. 高点B与高点A的K线间隔：日线≥5根，周线≥5周
    3. 高点B的最高价 > 高点A的最高价
    4. 高点B的MACD柱值 < 高点A的MACD柱值

    三、连续背离处理：
    - 如果已有背离标志为1，又形成新的背离C且C比B更背离：更新背离记录为C，覆盖旧背离

    Args:
        data: 包含 High, MACD_hist 的数据
        timeframe: 时间周期，'daily' 或 'weekly'
        min_interval: 两个高点之间的最小间隔，None则使用配置值
        drop_threshold: 局部高点检测的下跌阈值，None则使用配置值

    Returns:
        dict or False: 如果检测到顶背离返回详细信息字典，否则返回 False
    """
    # 使用配置参数
    config = DIVERGENCE_CONFIG.get(timeframe, DIVERGENCE_CONFIG['daily'])
    if min_interval is None:
        min_interval = config['min_interval']
    if drop_threshold is None:
        drop_threshold = config['drop_threshold']

    local_highs = detect_local_highs(data, drop_threshold)

    if len(local_highs) < 2:
        return False

    # 根据 conditions.md 文档要求：
    # 从已确认的局部高点列表中，从最近向更早遍历，寻找第一个满足以下条件的高点 A：
    # - A 的日期比 B 早
    # - B 与 A 的K线间隔：日线 ≥ 5根，周线 ≥ 5周
    # - B 的最高价 > A 的最高价
    # - B 的MACD柱值 < A 的MACD柱值

    # 取最新的局部高点 B
    recent_high = local_highs[-1]  # B点（较晚，最新确认）

    # 从最近向更早遍历，找到第一个满足背离条件的 A
    for i in range(len(local_highs) - 2, -1, -1):
        prev_high = local_highs[i]  # A点（更早）

        # 检查间隔：B的索引 - A的索引 >= min_interval
        interval = recent_high['index'] - prev_high['index']
        if interval < min_interval:
            # 间隔不足，继续向更早遍历
            continue

        # 检查价格创新高：B的价格 > A的价格
        if recent_high['price'] <= prev_high['price']:
            # 价格未创新高，继续向更早遍历
            continue

        # 检查MACD柱缩短：B的MACD < A的MACD
        if pd.isna(recent_high['macd_hist']) or pd.isna(prev_high['macd_hist']):
            continue

        if recent_high['macd_hist'] >= prev_high['macd_hist']:
            # MACD未缩短，继续向更早遍历
            continue

        # 找到满足条件的 A，顶背离形成
        result = {
            'detected': True,
            'prev_high': prev_high['price'],
            'recent_high': recent_high['price'],
            'prev_macd': prev_high['macd_hist'],
            'recent_macd': recent_high['macd_hist'],
            'interval': interval,
            'prev_high_idx': prev_high['index'],
            'recent_high_idx': recent_high['index'],
            'timeframe': timeframe,
            'local_highs': local_highs,  # 返回局部高点列表用于后续判断
        }

        # 添加日期信息
        if 'date' in prev_high:
            result['prev_high_date'] = prev_high['date']
        if 'date' in recent_high:
            result['recent_high_date'] = recent_high['date']

        return result

    # 遍历完所有历史高点，没有找到满足条件的 A
    return False


def check_all_invalidation_conditions(data, divergence_info, timeframe='daily'):
    """
    检查顶背离是否失效（三种失效条件）

    根据 conditions.md 文档，满足以下任一条件，背离标志清零：

    1. 强信号失效：出现新的局部高点C，且C的价格 > B的价格 且 C的MACD柱值 > B的MACD柱值
    2. 时间衰减失效：背离形成后，超过10个交易日（日线）或10周（周线）仍未触发均线跌破确认
    3. 反向趋势失效：价格从高点B下跌超过10%且未反弹（视为趋势逆转）

    Args:
        data: 数据
        divergence_info: 背离信息字典（包含 recent_high_idx, recent_high 等）
        timeframe: 时间周期，'daily' 或 'weekly'

    Returns:
        tuple: (is_invalidated, invalidation_reason, updated_info)
               - is_invalidated: 是否失效
               - invalidation_reason: 失效原因（字符串），如果未失效则为 None
               - updated_info: 更新后的背离信息（如果需要更新）
    """
    if divergence_info is None or not divergence_info.get('detected'):
        return False, None, divergence_info

    config = DIVERGENCE_CONFIG.get(timeframe, DIVERGENCE_CONFIG['daily'])
    local_highs = divergence_info.get('local_highs', [])
    recent_high_idx = divergence_info.get('recent_high_idx', 0)
    recent_high_price = divergence_info.get('recent_high', 0)

    curr_idx = len(data) - 1
    close = data['Close'].values

    # ===== 失效条件1：时间衰减失效 =====
    bars_since_divergence = curr_idx - recent_high_idx
    if bars_since_divergence > config['invalidation_bars']:
        return True, f"时间衰减失效：背离形成后超过{config['invalidation_bars']}根K线仍未确认", None

    # ===== 失效条件2：反向趋势失效 =====
    # 价格从高点B下跌超过10%且未反弹
    current_price = close[curr_idx]
    drop_from_high = (recent_high_price - current_price) / recent_high_price
    if drop_from_high > config['trend_reversal_pct']:
        return True, f"反向趋势失效：价格从高点{recent_high_price:.3f}下跌{drop_from_high*100:.1f}%超过{config['trend_reversal_pct']*100}%", None

    # ===== 失效条件3：强信号失效 =====
    # 检查是否有新的局部高点C使顶背离失效
    if len(local_highs) >= 3:
        high_b = local_highs[-2]  # 之前的最近高点B
        high_c = local_highs[-1]  # 新确认的高点C

        # 失效条件：C价格 > B价格 且 C的MACD > B的MACD
        if high_c['price'] > high_b['price'] and high_c['macd_hist'] > high_b['macd_hist']:
            return True, f"强信号失效：新高点价格{high_c['price']:.3f} > {high_b['price']:.3f}, MACD柱{high_c['macd_hist']:.4f} > {high_b['macd_hist']:.4f}", None

    # ===== 检查连续背离（更新背离记录） =====
    # 如果有新的局部高点，检查是否形成新的更强背离
    if len(local_highs) >= 3:
        high_a = local_highs[-3]  # A点（更早）
        high_c = local_highs[-1]  # C点（最新）

        # 检查是否形成新的背离（C相对于A）
        interval_ac = high_c['index'] - high_a['index']
        if interval_ac >= config['min_interval']:
            if high_c['price'] > high_a['price'] and high_c['macd_hist'] < high_a['macd_hist']:
                # 新的背离形成，检查是否比原背离更强
                # "更强"的定义：MACD差距更大
                original_divergence_strength = divergence_info.get('prev_macd', 0) - divergence_info.get('recent_macd', 0)
                new_divergence_strength = high_a['macd_hist'] - high_c['macd_hist']

                if new_divergence_strength > original_divergence_strength:
                    # 更新背离记录为C，覆盖旧背离
                    updated_info = {
                        'detected': True,
                        'prev_high': high_a['price'],
                        'recent_high': high_c['price'],
                        'prev_macd': high_a['macd_hist'],
                        'recent_macd': high_c['macd_hist'],
                        'interval': interval_ac,
                        'prev_high_idx': high_a['index'],
                        'recent_high_idx': high_c['index'],
                        'timeframe': timeframe,
                        'local_highs': local_highs,
                    }
                    if 'date' in high_a:
                        updated_info['prev_high_date'] = high_a['date']
                    if 'date' in high_c:
                        updated_info['recent_high_date'] = high_c['date']

                    return False, "连续背离更新：新的背离比原背离更强，更新背离记录", updated_info

    return False, None, divergence_info


def detect_divergence(data, type='top', grace_bars=None, confirm_sar=False, timeframe='daily'):
    """
    检测背离（兼容旧接口，内部调用新版本）

    Args:
        data: 包含 MACD 指标的数据
        type: 'top' 顶背离，'bottom' 底背离
        grace_bars: 允许信号在最近 N 根 K 线内持续有效（含当根）。None 表示不限制
        confirm_sar: 是否需要 SAR 跌破来确认顶背离有效（仅用于顶背离）
        timeframe: 时间周期，'daily' 或 'weekly'

    Returns:
        bool: 如果检测到背离返回 True，否则返回 False
    """
    if type == 'top':
        result = detect_divergence_v2(data, timeframe=timeframe)
        if result is False:
            return False

        # 检查 grace_bars
        if grace_bars is not None:
            curr_idx = len(data) - 1
            if curr_idx - result['recent_high_idx'] > grace_bars - 1:
                return False

        # 检查 SAR 确认
        if confirm_sar:
            sar = data['SAR'].values if 'SAR' in data.columns else None
            close = data['Close'].values
            if sar is not None:
                if close[-1] >= sar[-1]:
                    return False

        return True
    else:
        # 底背离暂不修改，保持原有逻辑
        return _detect_bottom_divergence_legacy(data, grace_bars)


def _detect_bottom_divergence_legacy(data, grace_bars=None):
    """底背离检测（保留原有逻辑）"""
    if data is None or len(data) < 30:
        return False

    low = data['Low'].values if 'Low' in data.columns else data['Close'].values
    dif = data['MACD'].values
    macd_hist = data['MACD_hist'].values
    curr = len(low) - 1

    recent_lookback = 5
    prev_lookback = 30
    recent_start = max(curr - recent_lookback + 1, 0)
    recent_min_idx = recent_start + low[recent_start:curr+1].argmin()

    prev_end = recent_min_idx - 1
    if prev_end <= 10:
        return False
    prev_start = max(prev_end - prev_lookback + 1, 0)
    prev_idx = prev_start + low[prev_start:prev_end+1].argmin()
    if recent_min_idx - prev_idx < 3:
        return False

    if low[recent_min_idx] < low[prev_idx] and (dif[recent_min_idx] > dif[prev_idx] or macd_hist[recent_min_idx] > macd_hist[prev_idx]):
        if grace_bars is None:
            return True
        if (curr - recent_min_idx) <= (grace_bars - 1):
            return True
    return False

def get_divergence_details(data, type='top', timeframe='daily'):
    """
    获取背离的详细信息（使用新的局部高点检测逻辑）

    Args:
        data: 包含 MACD 指标的数据
        type: 'top' 顶背离，'bottom' 底背离
        timeframe: 时间周期，'daily' 或 'weekly'

    Returns:
        dict: 包含背离详细信息的字典，包括：
              - detected: 是否检测到背离
              - recent_high: 最近高点价格
              - prev_high: 前一个高点价格
              - recent_hist: 最近高点的 MACD 柱值
              - prev_hist: 前一个高点的 MACD 柱值
              - interval: 两个高点之间的间隔
              - recent_high_idx: 最近高点的索引
              - prev_high_idx: 前一个高点的索引
              - timeframe: 时间周期
              - local_highs: 局部高点列表
        如果没有检测到背离，返回 None
    """
    if type == 'top':
        result = detect_divergence_v2(data, timeframe=timeframe)
        if result is False:
            return None
        return result
    else:
        # 底背离暂不修改
        return None

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


def detect_weekly_top_divergence(data_weekly, rsi_state, data_daily=None):
    """
    检测周线顶背离形成（使用新的局部高点检测逻辑）

    根据 conditions.md 文档最新规则：

    一、周线局部高点检测：
    - 维护"候选高点"变量
    - 当价格创新高时，更新候选高点
    - 当价格从候选高点下跌≥8%且5日均线跌破10日均线时，确认候选高点是有效的周线局部高点

    二、顶背离形成条件：
    1. 局部高点列表中至少有2个高点
    2. 取最近确认的两个高点A（较早）和B（较晚）
    3. 高点B与高点A的K线间隔≥5周
    4. 高点B的最高价 > 高点A的最高价
    5. 高点B的MACD柱值 < 高点A的MACD柱值

    三、背离失效条件（满足任一条件，背离标志清零）：
    1. 强信号失效：出现新的局部高点C，且C的价格 > B的价格 且 C的MACD柱值 > B的MACD柱值
    2. 时间衰减失效：背离形成后，超过10周仍未触发均线跌破确认
    3. 反向趋势失效：价格从高点B下跌超过10%且未反弹

    四、连续背离处理：
    - 如果已有背离标志为1，又形成新的背离C且C比B更背离：更新背离记录为C，覆盖旧背离

    五、顶背离生效：
    - diverse_week_flag = 1
    - 5日均线跌破10日均线
    - 清仓

    Args:
        data_weekly: 周线数据（包含 High, MACD_hist 等）
        rsi_state: RSI 状态字典
        data_daily: 日线数据（用于均线跌破判断，周线局部高点确认必需）

    Returns:
        tuple: (diverse_week_flag, is_valid_divergence, sell_signal, sell_fraction, messages, updated_rsi_state)
    """
    messages = []
    sell_signal = False
    sell_fraction = 0.0
    diverse_week_flag = rsi_state.get('diverse_week_flag', 0)
    div_weekly_date = rsi_state.get('div_weekly_date', None)

    # 复制 rsi_state 用于返回更新后的状态
    updated_rsi_state = rsi_state.copy()

    if data_weekly is None or len(data_weekly) < 10:
        return diverse_week_flag, False, sell_signal, sell_fraction, messages, updated_rsi_state

    # 使用配置参数
    config = DIVERGENCE_CONFIG['weekly']

    # 使用专门的周线局部高点检测函数（需要日线均线数据确认）
    local_highs = detect_weekly_local_highs(data_weekly, data_daily, drop_threshold=config['drop_threshold'])

    # 当前K线索引
    curr_idx = len(data_weekly) - 1
    close = data_weekly['Close'].values

    # ===== 如果已有顶背离，先检查失效条件 =====
    if diverse_week_flag == 1:
        # 构建背离信息用于失效检测
        divergence_info = {
            'detected': True,
            'recent_high_idx': updated_rsi_state.get('div_recent_idx', 0),
            'recent_high': updated_rsi_state.get('div_recent_high', 0),
            'prev_high': updated_rsi_state.get('div_prev_high', 0),
            'prev_macd': updated_rsi_state.get('div_prev_macd', 0),
            'recent_macd': updated_rsi_state.get('div_recent_macd', 0),
            'local_highs': local_highs,
        }

        # 检查所有失效条件
        is_invalidated, invalidation_reason, updated_info = check_all_invalidation_conditions(
            data_weekly, divergence_info, timeframe='weekly'
        )

        if is_invalidated:
            # 顶背离失效
            diverse_week_flag = 0
            # 清除顶背离相关状态
            updated_rsi_state['div_prev_high'] = None
            updated_rsi_state['div_prev_macd'] = None
            updated_rsi_state['div_recent_high'] = None
            updated_rsi_state['div_recent_macd'] = None
            updated_rsi_state['div_recent_idx'] = None
            updated_rsi_state['div_weekly_date'] = None
            messages.append(f"周线顶背离失效：{invalidation_reason}")

        elif updated_info is not None and updated_info != divergence_info:
            # 连续背离更新
            updated_rsi_state['div_prev_high'] = updated_info.get('prev_high')
            updated_rsi_state['div_prev_macd'] = updated_info.get('prev_macd')
            updated_rsi_state['div_recent_high'] = updated_info.get('recent_high')
            updated_rsi_state['div_recent_macd'] = updated_info.get('recent_macd')
            updated_rsi_state['div_recent_idx'] = updated_info.get('recent_high_idx')
            if 'recent_high_date' in updated_info:
                updated_rsi_state['div_weekly_date'] = updated_info['recent_high_date']
            messages.append(f"周线顶背离更新：{invalidation_reason}")

    # ===== 顶背离形成判断 =====
    # 根据 conditions.md 文档要求：
    # 从已确认的局部高点列表中，从最近向更早遍历，寻找第一个满足背离条件的高点 A
    # 调试日志：显示函数被调用
    logger.info(f"[周线顶背离检测] 函数被调用: diverse_week_flag={diverse_week_flag}, local_highs数量={len(local_highs)}")

    if diverse_week_flag == 0 and len(local_highs) >= 2:
        # 取最新的局部高点 B
        recent_high = local_highs[-1]  # B点（较晚，最新确认）

        # 从最近向更早遍历，找到第一个满足背离条件的 A
        for i in range(len(local_highs) - 2, -1, -1):
            prev_high = local_highs[i]  # A点（更早）

            # 检查间隔：B的索引 - A的索引 >= min_interval
            interval = recent_high['index'] - prev_high['index']
            if interval < config['min_interval']:
                # 间隔不足，继续向更早遍历
                continue

            # 检查价格创新高：B的价格 > A的价格
            if recent_high['price'] <= prev_high['price']:
                # 价格未创新高，继续向更早遍历
                continue

            # 检查MACD柱缩短：B的MACD < A的MACD
            if pd.isna(recent_high['macd_hist']) or pd.isna(prev_high['macd_hist']):
                continue

            if recent_high['macd_hist'] >= prev_high['macd_hist']:
                # MACD未缩短，继续向更早遍历
                continue

            # 找到满足条件的 A，顶背离形成
            diverse_week_flag = 1
            # 记录顶背离信息
            updated_rsi_state['div_prev_high'] = prev_high['price']
            updated_rsi_state['div_prev_macd'] = prev_high['macd_hist']
            updated_rsi_state['div_recent_high'] = recent_high['price']
            updated_rsi_state['div_recent_macd'] = recent_high['macd_hist']
            updated_rsi_state['div_recent_idx'] = recent_high['index']

            # 获取日期
            if 'date' in recent_high:
                div_weekly_date = recent_high['date']
            elif hasattr(data_weekly, 'index'):
                try:
                    div_weekly_date = data_weekly.index[recent_high['index']]
                except:
                    div_weekly_date = None
            updated_rsi_state['div_weekly_date'] = div_weekly_date

            # 添加日期信息到 prev_high
            prev_high_date = prev_high.get('date', None)

            div_date_str = str(div_weekly_date)[:10] if div_weekly_date else "未知"
            prev_date_str = str(prev_high_date)[:10] if prev_high_date else "未知"
            messages.append(f"周线顶背离形成 ({div_date_str}): 股价{recent_high['price']:.3f} > {prev_high['price']:.3f} ({prev_date_str}), MACD柱{recent_high['macd_hist']:.4f} < {prev_high['macd_hist']:.4f}")
            logger.info(f"[周线顶背离检测] 检测到背离: A({prev_date_str}, 价{prev_high['price']:.3f}, MACD{prev_high['macd_hist']:.4f}) vs B({div_date_str}, 价{recent_high['price']:.3f}, MACD{recent_high['macd_hist']:.4f})")

            # 找到第一个满足条件的 A 后退出循环
            break

        if diverse_week_flag == 0:
            # 没有找到满足条件的 A，记录调试日志
            logger.info(f"[周线顶背离检测] 遍历所有历史高点后未找到满足背离条件的配对")

    # ===== 顶背离生效判断（5日均线跌破10日均线） =====
    if diverse_week_flag == 1 and data_daily is not None:
        if 'MA5' in data_daily.columns and 'MA10' in data_daily.columns:
            ma5_curr = data_daily['MA5'].iloc[-1]
            ma10_curr = data_daily['MA10'].iloc[-1]
            if len(data_daily) >= 2:
                ma5_prev = data_daily['MA5'].iloc[-2]
                ma10_prev = data_daily['MA10'].iloc[-2]

                # 检测下穿动作
                if ma5_prev >= ma10_prev and ma5_curr < ma10_curr:
                    # 顶背离生效，清仓
                    sell_signal = True
                    sell_fraction = 1.0  # 清仓
                    diverse_week_flag = 0  # 重置标志

                    # 获取顶背离详细信息
                    div_prev_high = updated_rsi_state.get('div_prev_high', 0)
                    div_weekly_date = updated_rsi_state.get('div_weekly_date', None)
                    div_date_str = str(div_weekly_date)[:10] if div_weekly_date else "未知"

                    # 清除顶背离相关状态
                    updated_rsi_state['div_prev_high'] = None
                    updated_rsi_state['div_prev_macd'] = None
                    updated_rsi_state['div_recent_high'] = None
                    updated_rsi_state['div_recent_macd'] = None
                    updated_rsi_state['div_recent_idx'] = None
                    updated_rsi_state['div_weekly_date'] = None

                    messages.append(f"周线顶背离生效：顶背离形成于{div_date_str}, 前高={div_prev_high:.3f}, 5日均线下穿10日均线确认，建议清仓")

    updated_rsi_state['diverse_week_flag'] = diverse_week_flag
    return diverse_week_flag, (diverse_week_flag == 1), sell_signal, sell_fraction, messages, updated_rsi_state


def update_rsi_position_state(rsi_state, current_rsi, current_price, ma5_breakdown=False):
    """
    更新 RSI 位置状态（位置 1-5 的标记）

    位置标记规则：
    - 位置 1: RSI>90 时的股价高点（局部高点）
    - 位置 2: 从位置 1 下跌后的低点（局部低点）
    - 位置 3: 不是局部高点（只下跌了 2%，不满足显著回调）
    - 位置 4: RSI>90 后下跌 2% 的位置
    - 位置 5: 从位置 4 上涨 10% 的位置

    注意：周线顶背离状态（prev_high_price, diverse_week_flag等）独立管理，
          不会被均线跌破重置，只有在周线顶背离触发卖出后才重置。

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

    # 如果发生均线跌破，只重置 RSI 位置状态，保留周线顶背离状态
    if ma5_breakdown:
        # 保留周线顶背离状态，不重置
        prev_high_price = rsi_state.get('prev_high_price', 0)
        prev_high_macd = rsi_state.get('prev_high_macd', 0)
        prev_high_idx = rsi_state.get('prev_high_idx', None)
        diverse_week_flag = rsi_state.get('diverse_week_flag', 0)
        div_weekly_date = rsi_state.get('div_weekly_date', None)

        return {
            'current_position': 0,
            'rsi_peak': 0,
            'rsi_trough': 0,
            'price_at_peak': 0,
            'position_1_price': 0,
            'position_2_price': 0,
            'position_4_price': 0,
            'position_5_price': 0,
            # 保留周线顶背离状态
            'prev_high_price': prev_high_price,
            'prev_high_macd': prev_high_macd,
            'prev_high_idx': prev_high_idx,
            'diverse_week_flag': diverse_week_flag,
            'div_weekly_date': div_weekly_date
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
               rsi_state=None, check_weekly_divergence=True, rsi_peak_info=None):
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

        # ===== 周线顶背离检查（改进逻辑） =====
        if check_weekly_divergence and data_weekly is not None:
            # 更新前一个高点信息（用于顶背离判断）
            high_arr = data_weekly['High'].values if 'High' in data_weekly.columns else data_weekly['Close'].values
            macd_hist_arr = data_weekly['MACD_hist'].values if 'MACD_hist' in data_weekly.columns else None
            curr = len(high_arr) - 1

            # 寻找最近 5 周内的最高点（当前高点）
            recent_lookback = 5
            recent_start = max(curr - recent_lookback + 1, 0)
            recent_max_idx = recent_start + high_arr[recent_start:curr+1].argmax()
            current_high = high_arr[recent_max_idx]
            current_macd = macd_hist_arr[recent_max_idx] if macd_hist_arr is not None else 0

            # 获取当前状态
            diverse_week_flag = return_rsi_state.get('diverse_week_flag', 0)
            prev_high_price = return_rsi_state.get('prev_high_price', 0)
            prev_high_macd = return_rsi_state.get('prev_high_macd', 0)
            prev_high_idx = return_rsi_state.get('prev_high_idx', None)

            # ===== 前高点初始化/更新逻辑 =====
            # 改进：使用更长的窗口（30周）来找前高点
            if diverse_week_flag == 0:  # 只有在未形成顶背离时才更新前高点
                if prev_high_price == 0 or prev_high_idx is None:
                    # 还没有记录前高点，在更长的窗口内寻找
                    prev_lookback = 30
                    # 排除最近5周，在之前的数据中找前高点
                    prev_end = recent_max_idx - 5 if recent_max_idx > 5 else 0
                    prev_start = max(prev_end - prev_lookback + 1, 0)
                    if prev_end > prev_start and macd_hist_arr is not None:
                        # 找前高点（股价最高且MACD为正）
                        valid_indices = []
                        for i in range(prev_start, prev_end + 1):
                            if macd_hist_arr[i] > 0:  # MACD为正
                                valid_indices.append(i)
                        if valid_indices:
                            prev_max_idx = max(valid_indices, key=lambda i: high_arr[i])
                            return_rsi_state['prev_high_price'] = high_arr[prev_max_idx]
                            return_rsi_state['prev_high_macd'] = macd_hist_arr[prev_max_idx]
                            return_rsi_state['prev_high_idx'] = prev_max_idx
                else:
                    # 已有前高点，检查是否需要更新
                    # 如果当前高点超过了前高点，且MACD红柱也超过了（假背离），则更新前高点
                    if current_high > prev_high_price and macd_hist_arr is not None:
                        interval = recent_max_idx - prev_high_idx
                        if interval >= 5:
                            # 间隔足够，检查是假背离还是新高点
                            # 修复：确保两个高点都是红柱
                            both_red = prev_high_macd > 0 and current_macd > 0
                            red_bar_grows = current_macd > prev_high_macd

                            if both_red and red_bar_grows:
                                # MACD红柱也创新高，这是新的高点，更新前高点
                                return_rsi_state['prev_high_price'] = current_high
                                return_rsi_state['prev_high_macd'] = current_macd
                                return_rsi_state['prev_high_idx'] = recent_max_idx

            # 重新获取状态（可能已被更新）
            diverse_week_flag = return_rsi_state.get('diverse_week_flag', 0)
            prev_high_price = return_rsi_state.get('prev_high_price', 0)
            prev_high_macd = return_rsi_state.get('prev_high_macd', 0)
            prev_high_idx = return_rsi_state.get('prev_high_idx', None)

            if diverse_week_flag == 0 and prev_high_price > 0 and prev_high_idx is not None:
                # 检查是否形成顶背离（修复：必须确保两个高点都是红柱）
                interval = recent_max_idx - prev_high_idx
                if interval >= 5 and current_high > prev_high_price:
                    if macd_hist_arr is not None:
                        # 修复：确保两个高点的MACD柱都是红柱（正值）
                        both_red_bars = prev_high_macd > 0 and current_macd > 0
                        # 红柱缩短
                        red_bar_shortens = current_macd < prev_high_macd

                        if both_red_bars and red_bar_shortens:
                            # 顶背离形成
                            diverse_week_flag = 1
                            return_rsi_state['diverse_week_flag'] = diverse_week_flag
                            # 记录顶背离形成日期
                            curr_date = data_weekly.index[-1]
                            if hasattr(curr_date, 'strftime'):
                                div_date_str = curr_date.strftime('%Y-%m-%d')
                            else:
                                div_date_str = str(curr_date)[:10]
                            return_rsi_state['div_weekly_date'] = div_date_str
                            messages.append(f"【{stock_name}】周线顶背离形成 ({div_date_str}): 股价{current_high:.3f} > {prev_high_price:.3f}, MACD红柱{current_macd:.4f} < {prev_high_macd:.4f}")

            elif diverse_week_flag == 1:
                # 检查是否假背离（顶背离失效）（修复：必须确保两个高点都是红柱）
                if macd_hist_arr is not None:
                    # 两个高点都必须是红柱
                    both_red = prev_high_macd > 0 and current_macd > 0
                    # 红柱增强（假背离）
                    red_bar_grows = current_macd > prev_high_macd

                    if current_high > prev_high_price and both_red and red_bar_grows:
                        # 顶背离失效（假背离）
                        diverse_week_flag = 0
                        return_rsi_state['diverse_week_flag'] = 0
                        return_rsi_state['prev_high_price'] = current_high
                        return_rsi_state['prev_high_macd'] = current_macd
                        return_rsi_state['prev_high_idx'] = recent_max_idx
                        messages.append(f"【{stock_name}】周线顶背离失效（假背离）：股价{current_high:.3f} > {prev_high_price:.3f}, MACD红柱{current_macd:.4f} > {prev_high_macd:.4f}，更新前高点")

                # 检查是否 5 日均线跌破 10 日均线（顶背离生效，使用日线数据）
                if 'MA5' in data_daily.columns and 'MA10' in data_daily.columns:
                    ma5_daily_curr = data_daily['MA5'].iloc[-1]
                    ma10_daily_curr = data_daily['MA10'].iloc[-1]
                    if len(data_daily) >= 2:
                        ma5_daily_prev = data_daily['MA5'].iloc[-2]
                        ma10_daily_prev = data_daily['MA10'].iloc[-2]

                        if ma5_daily_prev >= ma10_daily_prev and ma5_daily_curr < ma10_daily_curr:
                            # 顶背离生效，触发卖出
                            # 获取周线顶背离的详细信息（形成日期和前高价格）
                            div_weekly_date = return_rsi_state.get('div_weekly_date', '未知')
                            prev_high_price = return_rsi_state.get('prev_high_price', 0)

                            # 完全重置周线顶背离状态
                            diverse_week_flag = 0
                            return_rsi_state['diverse_week_flag'] = 0
                            return_rsi_state['prev_high_price'] = 0
                            return_rsi_state['prev_high_macd'] = 0
                            return_rsi_state['prev_high_idx'] = None
                            return_rsi_state['div_weekly_date'] = None
                            sell_fraction = 0.5  # 卖出 50%
                            messages.append(f"【{stock_name}】周线顶背离生效 (策略 1-周线顶背离): 顶背离形成于{div_weekly_date}, 前高={prev_high_price:.3f}, 5 日均线下穿 10 日均线确认，建议卖出{sell_fraction*100:.0f}%")
                            # 注意：这里不立即添加卖出信号，而是通过后续的均线跌破逻辑处理

        if sell_id == 1:
            # 策略 1: 优先使用 120 分钟均线跌破 (用于实时监控)，如果没有则回退到日线 (用于回测)
            data_cross = data_120min if data_120min is not None else data_daily

            # ===== 优先级1: 清仓信号 - 周线顶背离 =====
            # 使用 grace_bars=5，允许周线顶背离在最近 5 周内持续有效
            if detect_divergence(data_weekly, 'top', grace_bars=5) and is_ma5_breakdown(data_cross, window=3):
                cross_name = "120 分钟" if data_120min is not None else "日线"
                weekly_div_details = get_divergence_details(data_weekly, 'top')
                if weekly_div_details and weekly_div_details.get('detected'):
                    weekly_high_idx = weekly_div_details.get('recent_high_idx', 0)
                    if weekly_high_idx > 0 and len(data_weekly.index) > weekly_high_idx:
                        div_date = data_weekly.index[weekly_high_idx]
                        div_date_str = div_date.strftime('%Y-%m-%d') if hasattr(div_date, 'strftime') else str(div_date)[:10]
                    else:
                        div_date_str = "未知"
                    prev_high = weekly_div_details.get('prev_high', 0)
                    messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 周线顶背离 ({div_date_str}) + {cross_name}5 日均线下穿 10 日均线确认，前高={prev_high:.3f}，建议清仓")
                else:
                    messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 触发 [周线顶背离 + {cross_name}5 日均线下穿 10 日均线]，建议清仓")

            # ===== 优先级2: 清仓信号 - 月线顶背离 =====
            elif data_monthly is not None and detect_divergence(data_monthly, 'top') and is_ma5_breakdown(data_daily, window=3):
                monthly_div_details = get_divergence_details(data_monthly, 'top')
                if monthly_div_details and monthly_div_details.get('detected'):
                    monthly_high_idx = monthly_div_details.get('recent_high_idx', 0)
                    if monthly_high_idx > 0 and len(data_monthly.index) > monthly_high_idx:
                        div_date = data_monthly.index[monthly_high_idx]
                        div_date_str = div_date.strftime('%Y-%m-%d') if hasattr(div_date, 'strftime') else str(div_date)[:10]
                    else:
                        div_date_str = "未知"
                    prev_high = monthly_div_details.get('prev_high', 0)
                    messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 月线顶背离 ({div_date_str}) + 日线 5 日均线下穿 10 日均线确认，前高={prev_high:.3f}，建议清仓")
                else:
                    messages.append(f"【{stock_name}】卖出信号 (策略 1-清仓): 触发 [月线顶背离 + 日线 5 日均线下穿 10 日均线]，建议清仓")

            # ===== 优先级3: 清仓信号 - RSI flag=3（周线RSI>90）=====
            elif rsi_flag == 3 and is_ma5_breakdown(data_daily, window=3):
                sar_breakdown = True
                ma5_breakdown_ts = find_ma5_breakdown_date(data_daily, window=3)
                ma5_breakdown_day = ma5_breakdown_ts.strftime('%Y-%m-%d') if ma5_breakdown_ts is not None and hasattr(ma5_breakdown_ts, 'strftime') else "未知"

                rsi_detail_msg = ""
                # 优先使用传入的峰值信息（已正确过滤重置后的周线数据）
                if rsi_peak_info is not None and rsi_peak_info.get('peak_date') is not None:
                    peak_date_str = rsi_peak_info['peak_date']
                    peak_rsi = rsi_peak_info.get('peak_rsi', 0)
                    rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"
                elif data_weekly is not None and 'RSI' in data_weekly.columns and len(data_weekly) > 0:
                    # 兜底逻辑：从最近20周找峰值
                    weekly_rsi_values = data_weekly['RSI'].values
                    weekly_dates = data_weekly.index
                    curr_idx = len(weekly_rsi_values) - 1
                    lookback = min(20, len(weekly_rsi_values))
                    recent_start = max(curr_idx - lookback + 1, 0)
                    if len(weekly_rsi_values[recent_start:curr_idx+1]) > 0:
                        peak_idx_in_slice = recent_start + weekly_rsi_values[recent_start:curr_idx+1].argmax()
                        peak_rsi = weekly_rsi_values[peak_idx_in_slice]
                        peak_date = weekly_dates[peak_idx_in_slice]
                        if hasattr(peak_date, 'strftime'):
                            peak_date_str = peak_date.strftime('%Y-%m-%d')
                        else:
                            peak_date_str = str(peak_date)[:10]
                        rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"

                messages.append(f"【{stock_name}】卖出信号 (策略 1-阶梯): 触发 [日线 5 日均线下穿 10 日均线 ({ma5_breakdown_day})]，周线 RSI flag=3{rsi_detail_msg}，建议清仓")
                rsi_flag = 0

            # ===== 优先级4: 减仓信号 - 日线顶背离 =====
            elif detect_divergence(data_daily, 'top', grace_bars=3) and is_ma5_breakdown(data_cross, window=3):
                cross_name = "120 分钟" if data_120min is not None else "日线"
                daily_div_details = get_divergence_details(data_daily, 'top')
                if daily_div_details and daily_div_details.get('detected'):
                    daily_high_idx = daily_div_details.get('recent_high_idx', 0)
                    if daily_high_idx > 0 and len(data_daily.index) > daily_high_idx:
                        div_date = data_daily.index[daily_high_idx]
                        div_date_str = div_date.strftime('%Y-%m-%d') if hasattr(div_date, 'strftime') else str(div_date)[:10]
                    else:
                        div_date_str = "未知"
                    prev_high = daily_div_details.get('prev_high', 0)
                    recent_high = daily_div_details.get('recent_high', 0)
                    messages.append(f"【{stock_name}】卖出信号 (策略 1-顶背离均线): 日线顶背离 ({div_date_str}) + {cross_name}5 日均线下穿 10 日均线确认，前高={prev_high:.3f}, 高={recent_high:.3f}, 建议卖出 1/3")
                else:
                    messages.append(f"【{stock_name}】卖出信号 (策略 1-减仓): 触发 [日线顶背离 + {cross_name}5 日均线下穿 10 日均线]，建议卖出 1/3")

            # ===== 优先级5: 阶梯式减仓 - RSI flag=1或2 =====
            elif rsi_flag > 0 and is_ma5_breakdown(data_daily, window=3):
                sar_breakdown = True
                ma5_breakdown_ts = find_ma5_breakdown_date(data_daily, window=3)
                ma5_breakdown_day = ma5_breakdown_ts.strftime('%Y-%m-%d') if ma5_breakdown_ts is not None and hasattr(ma5_breakdown_ts, 'strftime') else "未知"

                rsi_detail_msg = ""
                # 优先使用传入的峰值信息（已正确过滤重置后的周线数据）
                if rsi_peak_info is not None and rsi_peak_info.get('peak_date') is not None:
                    peak_date_str = rsi_peak_info['peak_date']
                    peak_rsi = rsi_peak_info.get('peak_rsi', 0)
                    rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"
                elif data_weekly is not None and 'RSI' in data_weekly.columns and len(data_weekly) > 0:
                    # 兜底逻辑：从最近20周找峰值
                    weekly_rsi_values = data_weekly['RSI'].values
                    weekly_dates = data_weekly.index
                    curr_idx = len(weekly_rsi_values) - 1
                    lookback = min(20, len(weekly_rsi_values))
                    recent_start = max(curr_idx - lookback + 1, 0)
                    if len(weekly_rsi_values[recent_start:curr_idx+1]) > 0:
                        peak_idx_in_slice = recent_start + weekly_rsi_values[recent_start:curr_idx+1].argmax()
                        peak_rsi = weekly_rsi_values[peak_idx_in_slice]
                        peak_date = weekly_dates[peak_idx_in_slice]
                        if hasattr(peak_date, 'strftime'):
                            peak_date_str = peak_date.strftime('%Y-%m-%d')
                        else:
                            peak_date_str = str(peak_date)[:10]
                        rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"

                sell_fraction, sell_msg = get_sell_fraction_by_flag(rsi_flag)
                trigger_reason = f"触发 [日线 5 日均线下穿 10 日均线 ({ma5_breakdown_day})]，周线 RSI flag={rsi_flag}{rsi_detail_msg}"
                messages.append(f"【{stock_name}】卖出信号 (策略 1-阶梯): {trigger_reason}，{sell_msg}")
                rsi_flag = 0

            # 均线跌破但没有任何信号时，重置 flag
            if ma5_breakdown_now:
                sar_breakdown = True
                if rsi_flag > 0 and not messages:
                    rsi_flag = 0

        elif sell_id == 2:
            # ===== 优先级1: 清仓信号 - 周线顶背离 =====
            # 使用 grace_bars=5，允许周线顶背离在最近 5 周内持续有效
            if detect_divergence(data_weekly, 'top', grace_bars=5) and is_ma5_breakdown(data_daily, window=3):
                weekly_div_details_2 = get_divergence_details(data_weekly, 'top')
                if weekly_div_details_2 and weekly_div_details_2.get('detected'):
                    weekly_high_idx_2 = weekly_div_details_2.get('recent_high_idx', 0)
                    if weekly_high_idx_2 > 0 and len(data_weekly.index) > weekly_high_idx_2:
                        div_date = data_weekly.index[weekly_high_idx_2]
                        div_date_str = div_date.strftime('%Y-%m-%d') if hasattr(div_date, 'strftime') else str(div_date)[:10]
                    else:
                        div_date_str = "未知"
                    prev_high = weekly_div_details_2.get('prev_high', 0)
                    messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 周线顶背离 ({div_date_str}) + 日线 5 日均线下穿 10 日均线确认，前高={prev_high:.3f}，建议清仓")
                else:
                    messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 触发 [周线顶背离 + 日线 5 日均线下穿 10 日均线]，建议清仓")

            # ===== 优先级2: 清仓信号 - 月线顶背离 =====
            elif data_monthly is not None and detect_divergence(data_monthly, 'top') and is_ma5_breakdown(data_daily, window=3):
                monthly_div_details_2 = get_divergence_details(data_monthly, 'top')
                if monthly_div_details_2 and monthly_div_details_2.get('detected'):
                    monthly_high_idx_2 = monthly_div_details_2.get('recent_high_idx', 0)
                    if monthly_high_idx_2 > 0 and len(data_monthly.index) > monthly_high_idx_2:
                        div_date = data_monthly.index[monthly_high_idx_2]
                        div_date_str = div_date.strftime('%Y-%m-%d') if hasattr(div_date, 'strftime') else str(div_date)[:10]
                    else:
                        div_date_str = "未知"
                    prev_high = monthly_div_details_2.get('prev_high', 0)
                    messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 月线顶背离 ({div_date_str}) + 日线 5 日均线下穿 10 日均线确认，前高={prev_high:.3f}，建议清仓")
                else:
                    messages.append(f"【{stock_name}】卖出信号 (策略 2-清仓): 触发 [月线顶背离 + 日线 5 日均线下穿 10 日均线]，建议清仓")

            # ===== 优先级3: 清仓信号 - RSI flag=3（周线RSI>90）=====
            elif rsi_flag == 3 and is_ma5_breakdown(data_daily, window=3):
                sar_breakdown = True
                ma5_breakdown_ts = find_ma5_breakdown_date(data_daily, window=3)
                ma5_breakdown_day = ma5_breakdown_ts.strftime('%Y-%m-%d') if ma5_breakdown_ts is not None and hasattr(ma5_breakdown_ts, 'strftime') else "未知"

                rsi_detail_msg = ""
                # 优先使用传入的峰值信息（已正确过滤重置后的周线数据）
                if rsi_peak_info is not None and rsi_peak_info.get('peak_date') is not None:
                    peak_date_str = rsi_peak_info['peak_date']
                    peak_rsi = rsi_peak_info.get('peak_rsi', 0)
                    rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"
                elif data_weekly is not None and 'RSI' in data_weekly.columns and len(data_weekly) > 0:
                    # 兜底逻辑：从最近20周找峰值
                    weekly_rsi_values = data_weekly['RSI'].values
                    weekly_dates = data_weekly.index
                    curr_idx = len(weekly_rsi_values) - 1
                    lookback = min(20, len(weekly_rsi_values))
                    recent_start = max(curr_idx - lookback + 1, 0)
                    if len(weekly_rsi_values[recent_start:curr_idx+1]) > 0:
                        peak_idx_in_slice = recent_start + weekly_rsi_values[recent_start:curr_idx+1].argmax()
                        peak_rsi = weekly_rsi_values[peak_idx_in_slice]
                        peak_date = weekly_dates[peak_idx_in_slice]
                        if hasattr(peak_date, 'strftime'):
                            peak_date_str = peak_date.strftime('%Y-%m-%d')
                        else:
                            peak_date_str = str(peak_date)[:10]
                        rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"

                messages.append(f"【{stock_name}】卖出信号 (策略 2-阶梯): 触发 [日线 5 日均线下穿 10 日均线 ({ma5_breakdown_day})]，周线 RSI flag=3{rsi_detail_msg}，建议清仓")
                rsi_flag = 0

            # ===== 优先级4: 减仓信号 - 日线顶背离 =====
            elif detect_divergence(data_daily, 'top', grace_bars=3) and is_ma5_breakdown(data_daily, window=3):
                daily_div_details_2 = get_divergence_details(data_daily, 'top')
                if daily_div_details_2 and daily_div_details_2.get('detected'):
                    daily_high_idx_2 = daily_div_details_2.get('recent_high_idx', 0)
                    if daily_high_idx_2 > 0 and len(data_daily.index) > daily_high_idx_2:
                        div_date = data_daily.index[daily_high_idx_2]
                        div_date_str = div_date.strftime('%Y-%m-%d') if hasattr(div_date, 'strftime') else str(div_date)[:10]
                    else:
                        div_date_str = "未知"
                    prev_high = daily_div_details_2.get('prev_high', 0)
                    recent_high = daily_div_details_2.get('recent_high', 0)
                    messages.append(f"【{stock_name}】卖出信号 (策略 2-顶背离均线): 日线顶背离 ({div_date_str}) + 5 日均线下穿 10 日均线确认，前高={prev_high:.3f}, 高={recent_high:.3f}, 建议卖出 1/3")
                else:
                    messages.append(f"【{stock_name}】卖出信号 (策略 2-减仓): 触发 [日线顶背离 + 日线 5 日均线下穿 10 日均线]，建议卖出 1/3")

            # ===== 优先级5: 阶梯式减仓 - RSI flag=1或2 =====
            elif rsi_flag > 0 and is_ma5_breakdown(data_daily, window=3):
                sar_breakdown = True
                ma5_breakdown_ts = find_ma5_breakdown_date(data_daily, window=3)
                ma5_breakdown_day = ma5_breakdown_ts.strftime('%Y-%m-%d') if ma5_breakdown_ts is not None and hasattr(ma5_breakdown_ts, 'strftime') else "未知"

                rsi_detail_msg = ""
                # 优先使用传入的峰值信息（已正确过滤重置后的周线数据）
                if rsi_peak_info is not None and rsi_peak_info.get('peak_date') is not None:
                    peak_date_str = rsi_peak_info['peak_date']
                    peak_rsi = rsi_peak_info.get('peak_rsi', 0)
                    rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"
                elif data_weekly is not None and 'RSI' in data_weekly.columns and len(data_weekly) > 0:
                    # 兜底逻辑：从最近20周找峰值
                    weekly_rsi_values = data_weekly['RSI'].values
                    weekly_dates = data_weekly.index
                    curr_idx = len(weekly_rsi_values) - 1
                    lookback = min(20, len(weekly_rsi_values))
                    recent_start = max(curr_idx - lookback + 1, 0)
                    if len(weekly_rsi_values[recent_start:curr_idx+1]) > 0:
                        peak_idx_in_slice = recent_start + weekly_rsi_values[recent_start:curr_idx+1].argmax()
                        peak_rsi = weekly_rsi_values[peak_idx_in_slice]
                        peak_date = weekly_dates[peak_idx_in_slice]
                        if hasattr(peak_date, 'strftime'):
                            peak_date_str = peak_date.strftime('%Y-%m-%d')
                        else:
                            peak_date_str = str(peak_date)[:10]
                        rsi_detail_msg = f" (RSI 峰值:{peak_date_str}={peak_rsi:.2f})"

                sell_fraction, sell_msg = get_sell_fraction_by_flag(rsi_flag)
                trigger_reason = f"触发 [日线 5 日均线下穿 10 日均线 ({ma5_breakdown_day})]，周线 RSI flag={rsi_flag}{rsi_detail_msg}"
                messages.append(f"【{stock_name}】卖出信号 (策略 2-阶梯): {trigger_reason}，{sell_msg}")
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
