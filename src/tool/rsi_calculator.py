# -*- coding: utf-8 -*-
"""
RSI 指标计算模块 (rsi_calculator.py)

【功能说明】
提供 RSI（Relative Strength Index，相对强弱指标）的计算功能。
支持日线和周线两种周期类型。

【国内算法说明】
国内行情软件（同花顺、东方财富）使用 SMA 方式计算 RSI：
    RSI = SMA(MAX(Close-LC, 0), N, 1) / SMA(ABS(Close-LC), N, 1) * 100

其中 SMA(x, N, 1) 等价于 EMA(x, alpha=1/N)，即 EWMA。

【使用方法】
    from src.tool.rsi_calculator import RSI

    # 日线 RSI(6)
    rsi_daily = RSI(period=6, freq='daily')
    result_daily = rsi_daily.calculate(close_series)

    # 周线 RSI(6)
    rsi_weekly = RSI(period=6, freq='weekly')
    result_weekly = rsi_weekly.calculate(close_series)

作者：量化交易团队
创建日期：2024
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Literal


class RSI:
    """
    RSI（相对强弱指标）计算器（东方财富标准算法）

    【指标原理】
    RSI 通过比较一段时间内价格上涨和下跌的幅度，来衡量市场强弱。
    使用累积EMA方式计算，确保与东方财富结果一致。

    【周期类型】
    - daily: 日线 RSI，直接使用日线收盘价计算
    - weekly: 周线 RSI，先将日线数据转换为周线再计算

    【重要提示 - 预热数据】
    为了与东方财富结果一致，需要提供足够长的历史预热数据：
    - 日线RSI: 建议提供测试日期前30天以上的数据
    - 周线RSI: 建议提供测试日期前1年以上的数据（约50周）

    【参数说明】
    - period: 计算周期，国内常用值为 6（短期）、12（中期）、24（长期）
    - freq: 周期类型，'daily'（日线）或 'weekly'（周线）
    - RSI > 70: 超买区域
    - RSI < 30: 超卖区域

    Attributes:
        period: RSI 计算周期
        freq: 周期类型（'daily' 或 'weekly'）
    """

    # 超买超卖阈值常量
    THRESHOLDS: Dict[str, Tuple[float, float]] = {
        'extreme_overbought': (90, 100),
        'severe_overbought': (80, 90),
        'overbought': (70, 80),
        'normal': (30, 70),
        'oversold': (20, 30),
        'severe_oversold': (10, 20),
        'extreme_oversold': (0, 10),
    }

    # 支持的周期类型
    VALID_FREQS = ('daily', 'weekly')

    def __init__(self,
                 period: int = 6,
                 freq: Literal['daily', 'weekly'] = 'daily') -> None:
        """
        初始化 RSI 计算器

        Args:
            period: RSI 计算周期，默认 6
                   日线常用: 6（短期）、12（中期）、24（长期）
                   周线常用: 6（约6周）
            freq: 周期类型，默认 'daily'
                  'daily': 日线 RSI
                  'weekly': 周线 RSI

        Raises:
            ValueError: 当参数无效时抛出
        """
        if period <= 0:
            raise ValueError(f"RSI 周期必须为正整数，当前值: {period}")

        if freq not in self.VALID_FREQS:
            raise ValueError(f"周期类型必须为 'daily' 或 'weekly'，当前值: {freq}")

        self.period = period
        self.freq = freq
        self._prev_au: Optional[float] = None
        self._prev_ad: Optional[float] = None
        self._weekly_close: Optional[pd.Series] = None

    def _convert_to_weekly(self, close: pd.Series) -> pd.Series:
        """
        将日线数据转换为周线数据（仅用于历史完整周）

        【转换逻辑】
        使用 pandas 的 resample 功能，按周五作为周结束日进行聚合。
        取每周最后一个交易日的收盘价作为周收盘价。

        Args:
            close: 日线收盘价序列

        Returns:
            pd.Series: 周线收盘价序列
        """
        weekly_close = close.resample('W-FRI').last()
        weekly_close = weekly_close.dropna()
        weekly_close.name = 'Weekly_Close'
        return weekly_close

    def _get_week_number(self, date: pd.Timestamp) -> int:
        """
        获取日期对应的周编号（用于周线RSI计算）

        Args:
            date: 日期时间戳

        Returns:
            int: 周编号（基于ISO周）
        """
        return date.isocalendar()[1]

    def _calculate_weekly_rsi_for_each_day(self, close: pd.Series) -> pd.Series:
        """
        计算每个交易日的周线RSI值（东方财富标准算法）

        【计算逻辑 - 累积EMA方式】
        东方财富的周线RSI采用累积EMA方式：
        1. 首先将日线转换为周线（每周最后交易日收盘价）
        2. 对所有周线数据累积计算EMA(AU和AD)
        3. 对于每个交易日T：
           - 使用T之前所有完整周的累积EMA状态
           - 用T的收盘价作为当前周的临时收盘价
           - 计算当前日的周线RSI值

        【重要提示 - 预热数据】
        周线RSI计算需要足够长的历史数据来让EMA状态稳定。
        建议至少提供1年的预热数据（从测试日期前1年开始），
        否则早期数据的RSI值可能与东方财富有偏差。

        Args:
            close: 日线收盘价序列（建议包含足够长的历史预热数据）

        Returns:
            pd.Series: 每个交易日的周线RSI值
        """
        rsi_values = pd.Series(index=close.index, dtype=float, name=f'RSI{self.period}_weekly')

        # 先获取所有完整周的周线数据（用于累积EMA）
        weekly_close = close.resample('W-FRI').last().dropna()

        if len(weekly_close) < self.period + 1:
            # 需要至少 period+1 个周收盘价才能产生 period 个差值用于EMA初始化
            return rsi_values.fillna(np.nan)

        # 计算周线的差值
        weekly_delta = weekly_close.diff().dropna()

        # 分离上涨和下跌
        weekly_up = weekly_delta.clip(lower=0)
        weekly_down = (-weekly_delta).clip(lower=0)

        # 累积计算周线EMA（国内算法 - 标准初始化方式）
        # 创建AU和AD序列
        weekly_au = pd.Series(index=weekly_delta.index, dtype=float)
        weekly_ad = pd.Series(index=weekly_delta.index, dtype=float)

        # 初始化：前N个差值用SMA初始化，之后用EMA平滑
        for i in range(len(weekly_delta)):
            if i < self.period - 1:
                # 前 N-1 个差值：还没有足够数据，设为NaN
                weekly_au.iloc[i] = np.nan
                weekly_ad.iloc[i] = np.nan
            elif i == self.period - 1:
                # 第 N 个差值：用前N个差值的SMA初始化
                weekly_au.iloc[i] = weekly_up.iloc[:self.period].mean()
                weekly_ad.iloc[i] = weekly_down.iloc[:self.period].mean()
            else:
                # 从第 N+1 个差值开始：使用EMA平滑
                if pd.notna(weekly_au.iloc[i - 1]) and pd.notna(weekly_ad.iloc[i - 1]):
                    weekly_au.iloc[i] = (weekly_up.iloc[i] + (self.period - 1) * weekly_au.iloc[i - 1]) / self.period
                    weekly_ad.iloc[i] = (weekly_down.iloc[i] + (self.period - 1) * weekly_ad.iloc[i - 1]) / self.period

        # 为每个交易日计算周线RSI
        for date in close.index:
            # 找到当前日期所在的周（周五为周结束日）
            date_weekday = date.weekday()
            days_to_friday = (4 - date_weekday) % 7
            current_week_friday = date + pd.Timedelta(days=days_to_friday)
            current_week_monday = current_week_friday - pd.Timedelta(days=4)

            # 找到当前日期之前最近的完整周（上周五结束）
            prev_complete_weeks = weekly_close[weekly_close.index < current_week_monday]

            if len(prev_complete_weeks) < 1:
                rsi_values.loc[date] = np.nan
                continue

            # 获取上周五的收盘价和对应的EMA状态
            last_complete_week_idx = prev_complete_weeks.index[-1]

            # 检查是否在weekly_au中有对应的EMA值
            if last_complete_week_idx not in weekly_au.index:
                # 上周五是第一个完整周，还没有差值
                # 需要检查是否有足够的完整周来计算初始RSI
                if len(prev_complete_weeks) >= self.period + 1:
                    # 有足够的数据，用简单平均初始化
                    # 取过去period个完整周的收盘价 + 当天收盘价
                    recent_weekly = prev_complete_weeks.iloc[-self.period:].copy()
                    recent_weekly.loc[date] = close.loc[date]
                    delta = recent_weekly.diff().dropna()
                    up_avg = delta.clip(lower=0).mean()
                    down_avg = (-delta).clip(lower=0).mean()
                    if up_avg + down_avg > 0:
                        rsi_values.loc[date] = up_avg / (up_avg + down_avg) * 100
                    else:
                        rsi_values.loc[date] = np.nan
                else:
                    rsi_values.loc[date] = np.nan
                continue

            prev_au = weekly_au.loc[last_complete_week_idx]
            prev_ad = weekly_ad.loc[last_complete_week_idx]

            if pd.isna(prev_au) or pd.isna(prev_ad):
                rsi_values.loc[date] = np.nan
                continue

            # 获取上一周的收盘价（用于计算当前周的变化）
            prev_week_close = prev_complete_weeks.iloc[-1]
            current_close = close.loc[date]

            # 计算当前周的变化（用当天收盘价与上周收盘价对比）
            current_delta = current_close - prev_week_close
            current_up = max(current_delta, 0)
            current_down = abs(min(current_delta, 0))

            # 使用EMA公式更新AU和AD
            new_au = (current_up + (self.period - 1) * prev_au) / self.period
            new_ad = (current_down + (self.period - 1) * prev_ad) / self.period

            # 计算RSI
            if new_au + new_ad == 0:
                rsi_values.loc[date] = np.nan
            else:
                rsi_values.loc[date] = new_au / (new_au + new_ad) * 100

        return rsi_values

    def _calculate_rsi_on_sequence(self, close_seq: pd.Series) -> float:
        """
        计算单个序列的RSI值（用于周线RSI的逐日计算）

        Args:
            close_seq: 收盘价序列（长度至少为N）

        Returns:
            float: RSI值
        """
        if len(close_seq) < self.period:
            return np.nan

        # 计算价格变动
        delta = close_seq.diff().dropna()

        if len(delta) < self.period - 1:
            return np.nan

        # 分离上涨和下跌幅度
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)

        # 使用SMA计算平均上涨和下跌
        # 如果有N-1个差值（对应N个价格），直接用SMA
        if len(delta) <= self.period:
            up_avg = up.mean()
            down_avg = down.mean()
        else:
            # 如果有更多差值，用EMA平滑
            up_avg = up.iloc[:self.period].mean()
            down_avg = down.iloc[:self.period].mean()

            for i in range(self.period, len(up)):
                up_avg = (up.iloc[i] + (self.period - 1) * up_avg) / self.period
                down_avg = (down.iloc[i] + (self.period - 1) * down_avg) / self.period

        # 计算RSI
        if up_avg + down_avg == 0:
            return np.nan
        return up_avg / (up_avg + down_avg) * 100

    def calculate(self, close: pd.Series) -> pd.Series:
        """
        计算 RSI 指标值（国内同花顺/东方财富标准算法）

        【国内算法】
        同花顺/东方财富的 RSI 计算方法：
        1. 前 N 天使用简单移动平均（SMA）计算 AU 和 AD
        2. 从第 N+1 天开始使用平滑移动平均（EMA）
        3. 平滑因子 alpha = 1/N

        【周线RSI特殊说明】
        周线RSI每天计算一个值：
        - 使用过去N-1个完整周的收盘价
        - 加上当天收盘价作为当前周的收盘价
        - 每天的周线RSI都不同

        Args:
            close: 收盘价序列（日线数据）

        Returns:
            pd.Series: RSI 值序列，范围 0-100

        Raises:
            ValueError: 当输入数据为空时抛出
            TypeError: 当输入非 pd.Series 类型时抛出
        """
        if not isinstance(close, pd.Series):
            raise TypeError(f"close 参数必须为 pd.Series，当前类型: {type(close)}")

        if close.empty:
            raise ValueError("收盘价序列不能为空")

        # 周线模式：每天计算一个周线RSI值
        if self.freq == 'weekly':
            return self._calculate_weekly_rsi_for_each_day(close)

        if len(close) < 2:
            return pd.Series([np.nan], index=close.index, name=f'RSI{self.period}_{self.freq}')

        # 计算价格变动
        delta = close.diff()

        # 分离上涨和下跌幅度
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)

        # 国内 RSI 算法：
        # 前 N 天用 SMA 初始化，之后用 EMA 平滑
        alpha = 1.0 / self.period

        # 初始化：前 N 天使用 SMA
        # SMA 计算 N 日平均上涨/下跌
        up_sma = up.rolling(window=self.period, min_periods=self.period).mean()
        down_sma = down.rolling(window=self.period, min_periods=self.period).mean()

        # 从第 N+1 天开始使用 EMA 平滑
        # EMA_new = alpha * 新值 + (1-alpha) * EMA_old
        # 即：EMA_new = (新值 + (N-1) * EMA_old) / N

        # 创建结果序列
        up_avg = up_sma.copy()
        down_avg = down_sma.copy()

        # 从第 N+1 天开始用 EMA 平滑
        for i in range(self.period, len(close)):
            if pd.notna(up_sma.iloc[i - 1]) and pd.notna(down_sma.iloc[i - 1]):
                # EMA 平滑
                up_avg.iloc[i] = (up.iloc[i] + (self.period - 1) * up_avg.iloc[i - 1]) / self.period
                down_avg.iloc[i] = (down.iloc[i] + (self.period - 1) * down_avg.iloc[i - 1]) / self.period

        # 计算 RSI
        rsi = up_avg / (up_avg + down_avg) * 100
        rsi = rsi.fillna(np.nan)

        # 存储最后的均值用于增量计算
        if len(rsi) > 0 and pd.notna(up_avg.iloc[-1]):
            self._prev_au = up_avg.iloc[-1]
            self._prev_ad = down_avg.iloc[-1]

        rsi.name = f'RSI{self.period}_{self.freq}'
        return rsi

    def calculate_incremental(self, new_close: float, prev_close: float) -> float:
        """
        增量计算单个 RSI 值（用于实时更新场景）

        注意：增量计算仅适用于日线模式，周线需要等待周结束。

        Args:
            new_close: 新收盘价
            prev_close: 前一收盘价

        Returns:
            float: 新的 RSI 值

        Raises:
            ValueError: 当未初始化或用于周线模式时抛出
        """
        if self.freq == 'weekly':
            raise ValueError("周线 RSI 不支持增量计算，请使用 calculate() 方法")

        if self._prev_au is None or self._prev_ad is None:
            raise ValueError("增量计算需先调用 calculate() 初始化历史均值")

        delta = new_close - prev_close
        up = max(delta, 0.0)
        down = abs(min(delta, 0.0))

        alpha = 1.0 / self.period
        new_au = alpha * up + (1 - alpha) * self._prev_au
        new_ad = alpha * down + (1 - alpha) * self._prev_ad

        self._prev_au = new_au
        self._prev_ad = new_ad

        if new_au + new_ad == 0:
            return np.nan

        return new_au / (new_au + new_ad) * 100

    def get_level(self, rsi_value: float) -> str:
        """
        获取 RSI 超买超卖等级

        Args:
            rsi_value: RSI 值，范围 0-100

        Returns:
            str: 警戒等级描述
        """
        if np.isnan(rsi_value):
            return "无效值"

        if rsi_value >= 90:
            return "极度超买"
        elif rsi_value >= 80:
            return "严重超买"
        elif rsi_value >= 70:
            return "超买"
        elif rsi_value <= 10:
            return "极度超卖"
        elif rsi_value <= 20:
            return "严重超卖"
        elif rsi_value <= 30:
            return "超卖"
        else:
            return "正常"

    def is_overbought(self, rsi_value: float, threshold: float = 70.0) -> bool:
        """判断是否处于超买状态"""
        if np.isnan(rsi_value):
            return False
        return rsi_value >= threshold

    def is_oversold(self, rsi_value: float, threshold: float = 30.0) -> bool:
        """判断是否处于超卖状态"""
        if np.isnan(rsi_value):
            return False
        return rsi_value <= threshold

    def detect_cross_up(self, rsi_series: pd.Series, threshold: float = 30.0) -> pd.Series:
        """检测 RSI 上穿阈值"""
        prev = rsi_series.shift(1)
        cross_up = (prev <= threshold) & (rsi_series > threshold)
        return cross_up.fillna(False)

    def detect_cross_down(self, rsi_series: pd.Series, threshold: float = 70.0) -> pd.Series:
        """检测 RSI 下穿阈值"""
        prev = rsi_series.shift(1)
        cross_down = (prev >= threshold) & (rsi_series < threshold)
        return cross_down.fillna(False)

    def get_weekly_close(self) -> Optional[pd.Series]:
        """
        获取转换后的周线收盘价（仅周线模式有效）

        Returns:
            pd.Series: 周线收盘价序列，日线模式返回 None
        """
        return self._weekly_close

    def __repr__(self) -> str:
        """类实例的字符串表示"""
        return f"RSI(period={self.period}, freq={self.freq})"


# ==================== 测试模块 ====================

if __name__ == "__main__":


    '''
    512480 日线RSI测试样例
    2024.01.02:42.73
    2024.02.19:64.60
    2024.05.16:38.68
    2024.05.29:45.85
    
    512480 周线RSI测试样例
    2024.04.30:51.2
    2024.06.14:64.5
    2024.11.22:64.49
    2024.07.12:57.16
    '''
    import logging
    import os
    import json
    from datetime import datetime

    # ==================== 测试参数配置 ====================
    # 【手动填写以下三个参数】

    # 测试时间段
    START_DATE = '2024-01-01'
    END_DATE = '2025-01-20'

    # 测试标的（掘金量化代码格式：SHSE.512480 或 SZSE.000001）
    SYMBOL = 'SHSE.510880'

    # ==================== 日志配置 ====================

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    LOG_DIR = os.path.join(project_root, 'logs')

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_FILE = os.path.join(LOG_DIR,
                            f'rsi_test_{SYMBOL.replace(".", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # ==================== 测试开始 ====================

    logger.info("=" * 60)
    logger.info("RSI 指标计算器测试（日线 + 周线）")
    logger.info("=" * 60)

    logger.info(f"测试参数配置:")
    logger.info(f"  股票代码: {SYMBOL}")
    logger.info(f"  开始时间: {START_DATE}")
    logger.info(f"  结束时间: {END_DATE}")
    logger.info(f"  日志文件: {LOG_FILE}")
    logger.info("=" * 60)

    # ==================== 使用掘金量化 API 获取真实数据 ====================
    try:
        from gm.api import history, set_token, ADJUST_PREV

        # 从配置文件读取 token
        config_path = os.path.join(project_root, 'config', 'settings.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                gm_token = config.get('gm_token')
                if gm_token:
                    set_token(gm_token)
                    logger.info("掘金量化 SDK 初始化成功")
                else:
                    logger.error("配置文件中未找到 gm_token")
                    raise Exception("缺少 gm_token 配置")
        else:
            logger.error(f"配置文件不存在: {config_path}")
            raise Exception("缺少配置文件")

        logger.info("使用掘金量化 API 获取真实历史数据（前复权）...")
        logger.info("注意：周线RSI需要较长预热数据以保证EMA状态稳定")

        # 获取日线数据（前复权）
        # 周线RSI需要从更早的数据开始计算EMA累积状态
        # 使用1年的预热数据来确保EMA状态稳定
        warmup_start = '2023-01-01'
        daily_data = history(
            symbol=SYMBOL,
            frequency='1d',
            start_time=warmup_start + ' 09:00:00',
            end_time=END_DATE + ' 15:30:00',
            fields='eob,close',
            df=True,
            adjust=ADJUST_PREV
        )

        if daily_data is None or daily_data.empty:
            logger.error(f"获取数据失败，请检查股票代码 {SYMBOL} 是否正确")
            raise Exception("数据获取失败")

        logger.info(f"获取到 {len(daily_data)} 条日线数据（包含预热数据）")

        # 创建 RSI 计算器实例
        rsi_daily = RSI(period=6, freq='daily')

        # 构建收盘价序列
        daily_data['eob'] = pd.to_datetime(daily_data['eob'])
        close_series_full = pd.Series(
            daily_data['close'].values,
            index=daily_data['eob'],
            name='Close'
        )

        # 计算完整数据的 RSI
        rsi_full = rsi_daily.calculate(close_series_full)

        # 获取交易日列表，确保只使用实际交易日
        from gm.api import get_trading_dates

        # 获取测试时间段内的交易日
        trading_dates = get_trading_dates(
            exchange='SHSE',
            start_date=START_DATE,
            end_date=END_DATE
        )

        # 将交易日转换为日期字符串集合（不含时间）
        trading_dates_set = set(trading_dates)

        # 通过日期字符串匹配来筛选数据（避免timezone问题）
        # 从数据索引中提取日期部分进行匹配
        close_series = close_series_full[
            close_series_full.index.strftime('%Y-%m-%d').isin(trading_dates_set)
        ]
        rsi_daily_result = rsi_full[
            rsi_full.index.strftime('%Y-%m-%d').isin(trading_dates_set)
        ]
        # 直接使用索引进行迭代，避免.values转换问题
        dates_index = close_series.index
        prices = close_series.values

    except ImportError:
        logger.error("未安装掘金量化 SDK (gm)，请先安装: pip install gm")
        raise Exception("缺少掘金量化 SDK")
    except Exception as e:
        logger.error(f"获取真实数据失败: {e}")
        raise

    # ---------- 测试 1: 日线 RSI ----------
    logger.info("")
    logger.info("【1. 日线 RSI(6) 计算测试】")
    logger.info("-" * 40)

    logger.info(f"创建实例: {rsi_daily}")

    logger.info(f"计算完成: {len(rsi_daily_result)} 个数据点（测试时间段内）")
    logger.info(f"有效数据: {rsi_daily_result.notna().sum()} 个")
    logger.info(f"NaN数据: {rsi_daily_result.isna().sum()} 个")

    logger.info("")
    logger.info("日线 RSI(6) 计算结果（全部）:")
    logger.info("-" * 40)
    logger.info(f"日期              | 收盘价    | RSI值     | 等级")
    logger.info("-" * 40)

    for date, close_val, rsi_val in zip(dates_index, prices, rsi_daily_result):
        # 直接从DatetimeIndex获取日期字符串
        date_str = date.strftime('%Y-%m-%d')
        if pd.notna(rsi_val):
            level = rsi_daily.get_level(rsi_val)
            logger.info(f"{date_str} | {close_val:8.2f} | {rsi_val:8.2f} | {level}")
        else:
            logger.info(f"{date_str} | {close_val:8.2f} |     NaN  | 无效值")

    # ---------- 测试 2: 周线 RSI ----------
    logger.info("")
    logger.info("【2. 周线 RSI(6) 计算测试】")
    logger.info("-" * 40)

    rsi_weekly = RSI(period=6, freq='weekly')
    logger.info(f"创建实例: {rsi_weekly}")

    # 周线RSI需要使用完整数据（包含预热数据）来累积EMA状态
    rsi_weekly_full = rsi_weekly.calculate(close_series_full)

    # 然后筛选测试时间段内的结果
    rsi_weekly_result = rsi_weekly_full[
        rsi_weekly_full.index.strftime('%Y-%m-%d').isin(trading_dates_set)
    ]

    logger.info(f"周线RSI数据点: {len(rsi_weekly_result)} 天")
    logger.info(f"有效数据: {rsi_weekly_result.notna().sum()} 个")
    logger.info(f"NaN数据: {rsi_weekly_result.isna().sum()} 个")

    logger.info("")
    logger.info("周线 RSI(6) 计算结果（每天一个值）:")
    logger.info("-" * 60)
    logger.info(f"日期              | 收盘价    | 周线RSI值 | 等级")
    logger.info("-" * 60)

    # 周线RSI现在每个交易日都有一个值
    for date, close_val, rsi_val in zip(dates_index, prices, rsi_weekly_result):
        date_str = date.strftime('%Y-%m-%d')
        if pd.notna(rsi_val):
            level = rsi_weekly.get_level(rsi_val)
            logger.info(f"{date_str} | {close_val:8.3f} | {rsi_val:8.2f} | {level}")
        else:
            logger.info(f"{date_str} | {close_val:8.3f} |     NaN  | 预热期")

    # ---------- 测试 3: 日线 vs 周线对比 ----------
    logger.info("")
    logger.info("【3. 日线 vs 周线 RSI 对比】")
    logger.info("-" * 40)

    logger.info(f"日线 RSI 统计:")
    valid_daily = rsi_daily_result.dropna()
    logger.info(f"  最大值: {valid_daily.max():.2f}")
    logger.info(f"  最小值: {valid_daily.min():.2f}")
    logger.info(f"  平均值: {valid_daily.mean():.2f}")
    logger.info(f"  超买天数(>=70): {(valid_daily >= 70).sum()} 天")
    logger.info(f"  超卖天数(<=30): {(valid_daily <= 30).sum()} 天")

    logger.info("")
    logger.info(f"周线 RSI 统计:")
    valid_weekly = rsi_weekly_result.dropna()
    logger.info(f"  最大值: {valid_weekly.max():.2f}")
    logger.info(f"  最小值: {valid_weekly.min():.2f}")
    logger.info(f"  平均值: {valid_weekly.mean():.2f}")
    logger.info(f"  超买天数(>=70): {(valid_weekly >= 70).sum()} 天")
    logger.info(f"  超卖天数(<=30): {(valid_weekly <= 30).sum()} 天")

    # ---------- 测试 4: 信号检测 ----------
    logger.info("")
    logger.info("【4. RSI 信号检测测试】")
    logger.info("-" * 40)

    # 日线上穿30信号
    daily_cross_up_30 = rsi_daily.detect_cross_up(rsi_daily_result, threshold=30)
    daily_cross_up_dates = rsi_daily_result.index[daily_cross_up_30].tolist()
    logger.info(f"日线上穿30信号: {len(daily_cross_up_dates)} 次")
    for d in daily_cross_up_dates[:10]:
        rsi_v = rsi_daily_result.loc[d]
        logger.info(f"  {d.strftime('%Y-%m-%d')} -> RSI={rsi_v:.2f}")

    # 日线下穿70信号
    daily_cross_down_70 = rsi_daily.detect_cross_down(rsi_daily_result, threshold=70)
    daily_cross_down_dates = rsi_daily_result.index[daily_cross_down_70].tolist()
    logger.info(f"日线下穿70信号: {len(daily_cross_down_dates)} 次")
    for d in daily_cross_down_dates[:10]:
        rsi_v = rsi_daily_result.loc[d]
        logger.info(f"  {d.strftime('%Y-%m-%d')} -> RSI={rsi_v:.2f}")

    # 周线上穿30信号
    weekly_cross_up_30 = rsi_weekly.detect_cross_up(rsi_weekly_result, threshold=30)
    weekly_cross_up_dates = rsi_weekly_result.index[weekly_cross_up_30].tolist()
    logger.info(f"周线上穿30信号: {len(weekly_cross_up_dates)} 次")
    for d in weekly_cross_up_dates:
        rsi_v = rsi_weekly_result.loc[d]
        logger.info(f"  {d.strftime('%Y-%m-%d')} -> RSI={rsi_v:.2f}")

    # ---------- 测试 5: 边界情况 ----------
    logger.info("")
    logger.info("【5. 边界情况测试】")
    logger.info("-" * 40)

    # 测试无效周期
    try:
        RSI(period=0, freq='daily')
    except ValueError as e:
        logger.info(f"无效周期测试: 捕获异常 - {e}")

    # 测试无效频率
    try:
        RSI(period=6, freq='monthly')
    except ValueError as e:
        logger.info(f"无效频率测试: 捕获异常 - {e}")

    # 测试周线增量计算
    try:
        rsi_weekly.calculate_incremental(100, 99)
    except ValueError as e:
        logger.info(f"周线增量计算测试: 捕获异常 - {e}")

    # 测试空数据
    try:
        rsi_daily.calculate(pd.Series([], dtype=float))
    except ValueError as e:
        logger.info(f"空数据测试: 捕获异常 - {e}")

    # ---------- 测试完成 ----------
    logger.info("")
    logger.info("=" * 60)
    logger.info("所有测试完成!")
    logger.info(f"日志已保存至: {LOG_FILE}")
    logger.info("=" * 60)