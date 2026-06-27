#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
背离检测器模块 (Divergence Detector)

根据背离判断.md和背离规则.md中的判断逻辑检测顶背离和底背离。

核心逻辑：
1. 顶背离形成：新高点B与上一个高点A比较，B价格>A价格且B的MACD柱<A的MACD柱
2. 同趋势约束：时间间隔≤60日，中间回调幅度≤30%，两者高于60日均线
3. 顶背离生效：背离标志=1 + 5日均线跌破10日均线

作者：量化交易团队
"""
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入peak_detector和macd_calculator
from src.tool.peak_detector import PeakDetector, PeakInfo
from src.tool.macd_calculator import MACDCalculator

try:
    from gm.api import history, set_token, ADJUST_PREV
except ImportError:
    ADJUST_PREV = None
    history = None

# 配置日志（使用命名logger，不调用basicConfig避免影响根logger）
logger = logging.getLogger('divergence_detector')
logger.setLevel(logging.INFO)

# 默认添加控制台handler（可被外部通过setup_log_file添加文件handler）
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(_console_handler)

# 不传播到根logger
logger.propagate = False


def setup_log_file(log_path: str):
    """设置日志文件输出（为当前logger添加文件handler）"""
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)
    return file_handler


def clear_log_file():
    """清除文件handler（用于切换到新的日志文件）"""
    # 移除所有FileHandler，保留ConsoleHandler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)


class DivergenceType(Enum):
    """背离类型枚举"""
    TOP = "顶背离"
    BOTTOM = "底背离"


class TimeFrame(Enum):
    """时间级别枚举"""
    DAILY = "日线"
    WEEKLY = "周线"


@dataclass
class DivergenceSignal:
    """背离信号数据结构"""
    date: pd.Timestamp              # 背离高点日期（高点B）
    divergence_type: DivergenceType # 背离类型
    timeframe: TimeFrame            # 时间级别
    peak_a_date: pd.Timestamp       # 高点A日期
    peak_a_price: float             # 高点A价格
    peak_b_price: float             # 高点B价格
    peak_a_macd: float              # 高点A的MACD柱值
    peak_b_macd: float              # 高点B的MACD柱值
    days_interval: int              # A与B之间的间隔天数
    retracement_pct: float          # 中间回调幅度百分比
    is_confirmed: bool = False      # 是否已确认生效（均线跌破）
    confirmation_date: Optional[pd.Timestamp] = None  # 均线跌破生效日期
    divergence_flag: int = 0        # 背离标志


class DivergenceDetector:
    """
    背离检测器类

    根据背离判断.md中的逻辑：
    - 只比较相邻的、同趋势内的两个高点
    - 时间间隔 ≤ 60日
    - 中间回调幅度 ≤ 30%
    - 两者都高于60日均线
    """

    # 背离判断参数（根据文档）
    MIN_KLINE_INTERVAL = 5         # B与A之间最小K线间隔（日线≥5根）
    MAX_DAYS_INTERVAL = 60         # 最大时间间隔（日线60天）
    MAX_RETRACEMENT = 0.30         # 最大回调幅度（30%）
    MA_PERIOD = 60                 # 趋势确认均线周期（60日均线）

    def __init__(self):
        """初始化背离检测器"""
        # 子模块（日线）
        self._peak_detector: Optional[PeakDetector] = None
        self._macd_calculator: Optional[MACDCalculator] = None

        # 子模块（周线）
        self._weekly_peak_detector: Optional[PeakDetector] = None

        # 数据缓存
        self._df: Optional[pd.DataFrame] = None
        self._symbol: Optional[str] = None

        # 均线数据
        self._ma5: Optional[pd.Series] = None
        self._ma10: Optional[pd.Series] = None
        self._ma60: Optional[pd.Series] = None

        # 周线数据
        self._weekly_df: Optional[pd.DataFrame] = None
        self._weekly_macd: Optional[pd.Series] = None
        self._weekly_peaks: List[PeakInfo] = []
        self._weekly_troughs: List[PeakInfo] = []

        # 背离状态
        self._daily_top_divergence_flag: int = 0   # 日线顶背离标志
        self._weekly_top_divergence_flag: int = 0  # 周线顶背离标志
        self._current_daily_divergence: Optional[DivergenceSignal] = None
        self._current_weekly_divergence: Optional[DivergenceSignal] = None

    def prepare_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = '1d'
    ) -> pd.DataFrame:
        """
        准备历史数据

        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame
        """
        # 延迟导入gm.api（确保在调用时才导入）
        try:
            from gm.api import history, set_token, ADJUST_PREV
        except ImportError:
            logger.error("未安装掘金量化SDK，请先安装: pip install gm")
            return pd.DataFrame()

        logger.info(f"获取历史数据: {symbol}, {start_date} ~ {end_date}")

        df = history(
            symbol=symbol,
            frequency=frequency,
            start_time=start_date,
            end_time=end_date,
            fields='open,high,low,close,volume,bob,eob',
            df=True,
            skip_suspended=True,
            fill_missing='Last',
            adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
        )

        if df is None or df.empty:
            logger.error("获取数据失败")
            return pd.DataFrame()

        # 重命名并设置索引
        df = df.rename(columns={'bob': 'datetime'})
        df = df.set_index('datetime')
        df = df.sort_index()

        logger.info(f"获取到 {len(df)} 条日线数据")

        self._df = df
        self._symbol = symbol

        # 初始化PeakDetector（日线参数）
        # 注意：min_distance必须小于MAX_DAYS_INTERVAL(60)，否则可能错过峰点导致无法形成背离判断
        # 【修复】min_distance=5 与 MIN_KLINE_INTERVAL=5 保持一致，允许检测间隔≥5天的峰点
        self._peak_detector = PeakDetector(
            atr_period=14,
            prominence_factor=1.0,
            min_distance=5  # 日线最小间隔5天（与MIN_KLINE_INTERVAL一致）
        )
        self._peak_detector.prepare_data(df['high'], df['low'], df['close'])

        # 初始化MACDCalculator
        self._macd_calculator = MACDCalculator()
        self._macd_calculator.prepare_data(df['close'])

        # 计算均线
        self._ma5 = df['close'].rolling(window=5).mean()
        self._ma10 = df['close'].rolling(window=10).mean()
        self._ma60 = df['close'].rolling(window=60).mean()

        # 准备周线数据
        self._prepare_weekly_data()

        return df

    def _prepare_weekly_data(self):
        """准备周线数据"""
        if self._df is None:
            return

        # 按周聚合
        weekly_df = self._df.groupby(self._df.index.to_period('W')).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # 获取每周最后一天日期
        weekly_dates = []
        for period in weekly_df.index:
            week_start = period.start_time
            week_end = period.end_time
            if self._df.index.tz is not None:
                week_start = week_start.tz_localize(self._df.index.tz)
                week_end = week_end.tz_localize(self._df.index.tz)
            week_data = self._df[(self._df.index >= week_start) & (self._df.index <= week_end)]
            if len(week_data) > 0:
                weekly_dates.append(week_data.index[-1])

        self._weekly_df = pd.DataFrame({
            'high': weekly_df['high'].values,
            'low': weekly_df['low'].values,
            'close': weekly_df['close'].values
        }, index=weekly_dates)

        # 计算周线MACD
        weekly_macd_calc = MACDCalculator()
        weekly_macd_calc.prepare_data(self._weekly_df['close'])
        self._weekly_macd = weekly_macd_calc._macd_series

        # 初始化周线PeakDetector（周线参数）
        # 周线数据较少，min_distance用周数而非天数
        self._weekly_peak_detector = PeakDetector(
            atr_period=14,
            prominence_factor=1.0,
            min_distance=3  # 周线最小间隔3周
        )
        self._weekly_peak_detector.prepare_data(
            self._weekly_df['high'],
            self._weekly_df['low'],
            self._weekly_df['close']
        )

        logger.info(f"周线数据聚合完成: {len(self._weekly_df)} 周")

    def check_ma_cross_down(self, date: pd.Timestamp) -> bool:
        """
        检查5日均线是否跌破10日均线

        Args:
            date: 检查日期

        Returns:
            True表示跌破
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            ma5_data = self._ma5[self._ma5.index.strftime('%Y-%m-%d') == date_str]
            ma10_data = self._ma10[self._ma10.index.strftime('%Y-%m-%d') == date_str]

            if len(ma5_data) > 0 and len(ma10_data) > 0:
                return ma5_data.iloc[0] < ma10_data.iloc[0]
            return False
        except Exception:
            return False

    def check_above_ma60(self, date: pd.Timestamp) -> bool:
        """
        检查价格是否高于60日均线

        Args:
            date: 检查日期

        Returns:
            True表示高于
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            close_data = self._df['close'][self._df.index.strftime('%Y-%m-%d') == date_str]
            ma60_data = self._ma60[self._ma60.index.strftime('%Y-%m-%d') == date_str]

            if len(close_data) > 0 and len(ma60_data) > 0:
                return close_data.iloc[0] > ma60_data.iloc[0]
            return False
        except Exception:
            return False

    def calculate_retracement(self, peak_a: PeakInfo, peak_b: PeakInfo) -> Tuple[float, pd.Timestamp]:
        """
        计算两个高点之间的最大回调幅度

        Args:
            peak_a: 高点A
            peak_b: 高点B

        Returns:
            (回调幅度百分比, 回调最低点日期)
        """
        try:
            # 获取两点之间的数据
            date_a = peak_a.date
            date_b = peak_b.date

            between_data = self._df['low'][
                (self._df.index > date_a) & (self._df.index < date_b)
            ]

            if len(between_data) == 0:
                return 0.0, date_a

            min_low = between_data.min()
            min_date = between_data.idxmin()

            # 回调幅度 = (高点A - 最低点) / 高点A
            retracement = (peak_a.price - min_low) / peak_a.price

            return retracement, min_date
        except Exception:
            return 0.0, peak_a.date

    def detect_daily_top_divergence(self, buy_signal_date: Optional[pd.Timestamp] = None) -> List[DivergenceSignal]:
        """
        检测日线顶背离

        根据背离规则.md的逻辑：
        1. 确认新局部高点B（使用最高价判断）
        2. 从最近向更早遍历，找第一个满足条件的A：
           - A日期比B早
           - B与A间隔≥5根K线
           - B最高价>A最高价
           - B的MACD柱<A的MACD柱
        3. 同趋势约束：
           - 时间间隔≤60日
           - 回调幅度≤30%
           - 两者高于60日均线

        Args:
            buy_signal_date: 最近一次买入信号（SAR金叉）日期，用于确定peak B的搜索范围
                            如果传入此参数，将从买入信号日期到当前日期之间找最高点作为peak B

        Returns:
            顶背离信号列表
        """
        if self._peak_detector is None or self._macd_calculator is None:
            logger.error("请先调用 prepare_data() 准备数据")
            return []

        # 获取日线峰点列表（使用最高价检测）
        peaks = self._peak_detector.detect_peaks(
            self._df.index[0].strftime('%Y-%m-%d'),
            self._df.index[-1].strftime('%Y-%m-%d')
        )

        if len(peaks) < 2:
            logger.info("[日线] 峰点数量不足，无法检测顶背离")
            return []

        logger.info(f"[日线] 检测到 {len(peaks)} 个峰点（使用最高价）")

        # 【新增】输出所有峰点的详细信息
        macd_series = self._macd_calculator._macd_series
        for i, peak in enumerate(peaks):
            macd_val = self._get_value_at_date(peak.date, macd_series)
            macd_str = f"{macd_val:.4f}" if not np.isnan(macd_val) else "N/A"
            logger.info(f"  [峰点{i+1}] {peak.date.strftime('%Y-%m-%d')} | 最高价={peak.price:.3f} | MACD={macd_str}")

        divergences = []

        # ==================== 监控模式：从买入信号后的最新最高点判断顶背离 ====================
        # 用户需求：监控模式下只判断上一次金叉后的最新高点的顶背离
        # 传入buy_signal_date时，从该日期到当前日期之间找最高点作为peak B
        if buy_signal_date is not None:
            # 确保买入信号日期在数据范围内
            if buy_signal_date not in self._df.index:
                # 找最近的交易日
                valid_dates = self._df.index[self._df.index >= buy_signal_date]
                if len(valid_dates) == 0:
                    logger.warning(f"  [监控模式] 买入信号日期{buy_signal_date.strftime('%Y-%m-%d')}超出数据范围")
                    return divergences
                buy_signal_date = valid_dates[0]

            # 从买入信号日期到当前日期之间找最高点作为peak B
            recent_df = self._df.loc[buy_signal_date:]
            recent_high_idx = recent_df['high'].idxmax()
            recent_high_price = recent_df['high'].max()
            recent_high_macd = self._get_value_at_date(recent_high_idx, macd_series)

            if np.isnan(recent_high_macd):
                logger.warning(f"  [监控模式] 最新高点{recent_high_idx.strftime('%Y-%m-%d')}的MACD值为空")
                return divergences

            recent_high_macd_str = f"{recent_high_macd:.4f}"
            logger.info(f"  [监控模式] 买入信号后的最新高点: {recent_high_idx.strftime('%Y-%m-%d')} | 最高价={recent_high_price:.3f} | MACD={recent_high_macd_str}")

            # 创建虚拟波峰B
            virtual_peak_b = PeakInfo(
                date=recent_high_idx,
                price=recent_high_price,
                prominence=0,
                atr_value=0,
                is_peak=True
            )

            # 【核心修正】监控模式下找买入信号之前最近的局部高点A
            # 顶背离应该比较最近的局部高点，如果不满足条件，不形成背离
            # 局部高点定义：最高价比前后各1天都高的点

            # 从买入信号之前的数据中找最近的局部高点
            nearest_local_high_date = None
            nearest_local_high_price = 0.0
            nearest_local_high_macd = 0.0

            # 遍历买入信号之前的数据，找最近的局部高点
            for j in range(len(self._df) - 1, -1, -1):
                date_j = self._df.index[j]
                # 只考虑买入信号之前的数据
                if date_j >= buy_signal_date:
                    continue
                # 只考虑距离当前高点至少5天的数据
                days_to_high = (recent_high_idx - date_j).days
                if days_to_high < self.MIN_KLINE_INTERVAL:
                    continue
                if days_to_high > self.MAX_DAYS_INTERVAL:
                    continue
                # 检查是否是局部高点（比前后各1天都高）
                high_j = self._df['high'].iloc[j]
                # 检查前1天
                if j > 0:
                    high_prev = self._df['high'].iloc[j - 1]
                    if high_j <= high_prev:
                        continue
                # 检查后1天
                if j < len(self._df) - 1:
                    high_next = self._df['high'].iloc[j + 1]
                    if high_j <= high_next:
                        continue
                # 找到了局部高点，记录并退出（只找最近的）
                nearest_local_high_date = date_j
                nearest_local_high_price = high_j
                nearest_local_high_macd = self._get_value_at_date(date_j, macd_series)
                break

            if nearest_local_high_date is None:
                logger.info(f"  [监控模式] 买入信号之前没有局部高点")
                return divergences

            if np.isnan(nearest_local_high_macd):
                logger.warning(f"  [监控模式] 最近局部高点的MACD值为空")
                return divergences

            # 检查背离条件（只比较这一个最近的局部高点）
            # 条件1：价格创新高（peak B价格 > 最近局部高点价格）
            price_condition = recent_high_price > nearest_local_high_price

            # 条件2：MACD减弱（peak B的MACD < 最近局部高点的MACD）
            macd_condition = recent_high_macd < nearest_local_high_macd

            # 条件3：间隔≤60天
            days_interval = (recent_high_idx - nearest_local_high_date).days
            interval_condition = days_interval <= self.MAX_DAYS_INTERVAL

            logger.info(f"  [监控模式-背离判断] 高点B({recent_high_idx.strftime('%Y-%m-%d')}) vs 最近局部高点A({nearest_local_high_date.strftime('%Y-%m-%d')})")
            logger.info(f"    价格条件: B={recent_high_price:.3f} > A={nearest_local_high_price:.3f} ? {price_condition}")
            logger.info(f"    MACD条件: B={recent_high_macd:.4f} < A={nearest_local_high_macd:.4f} ? {macd_condition}")
            logger.info(f"    间隔条件: {days_interval}天 ≤ {self.MAX_DAYS_INTERVAL}天 ? {interval_condition}")

            # 只有所有条件满足才形成背离
            if not (price_condition and macd_condition and interval_condition):
                logger.info(f"  [监控模式] 最近的局部高点不满足背离条件，不形成顶背离")
                return divergences
                days_interval = (recent_high_idx - found_peak_a['date']).days
                # 计算回调幅度
            retracement, min_date = self.calculate_retracement(
                PeakInfo(date=nearest_local_high_date, price=nearest_local_high_price, prominence=0, atr_value=0, is_peak=True),
                virtual_peak_b
            )

            if retracement > self.MAX_RETRACEMENT:
                logger.info(f"  [监控模式] 回调幅度过大: {retracement:.1%} > {self.MAX_RETRACEMENT:.1%}, 不形成顶背离")
                return divergences

            # 所有条件满足，形成顶背离
            logger.info(f"  [监控模式-日线顶背离形成] {recent_high_idx.strftime('%Y-%m-%d')} | 局部高点A: {nearest_local_high_date.strftime('%Y-%m-%d')} 最高价={nearest_local_high_price:.3f} MACD={nearest_local_high_macd:.4f} | 高点B: 最高价={recent_high_price:.3f} MACD={recent_high_macd:.4f} | 间隔={days_interval}天 | 回调={retracement:.1%}")

            divergence = DivergenceSignal(
                date=recent_high_idx,
                divergence_type=DivergenceType.TOP,
                timeframe=TimeFrame.DAILY,
                peak_a_date=nearest_local_high_date,
                peak_a_price=nearest_local_high_price,
                peak_b_price=recent_high_price,
                peak_a_macd=nearest_local_high_macd,
                peak_b_macd=recent_high_macd,
                days_interval=days_interval,
                retracement_pct=retracement,
                is_confirmed=True,
                divergence_flag=1
            )
            divergences.append(divergence)

            # 监控模式只需要这一个背离，直接返回
            return divergences

        # ==================== 标准模式（回测）：找峰点B之前最近的局部高点A ====================
        # 【核心修正】顶背离判断应该比较峰点B之前最近的局部高点A
        # 如果最近的局部高点A不满足背离条件，则不形成背离
        # 不应该跳过中间的局部高点去找更早的峰点
        # 记录监控模式已检测到的背离日期，避免重复
        monitored_dates = set(d.date for d in divergences)

        # 从最新的峰点开始，找其之前最近的局部高点进行比较
        for i in range(len(peaks) - 1, 0, -1):
            peak_b = peaks[i]  # 当前峰点B（最新）

            # 如果监控模式已检测到此峰点的背离，跳过
            if peak_b.date in monitored_dates:
                continue

            # 获取峰点B的MACD柱值和最高价（peak.price是最高价）
            macd_b = self._get_value_at_date(peak_b.date, macd_series)
            price_b = peak_b.price  # 最高价
            high_b = peak_b.price   # 最高价

            # 【核心修改】找峰点B之前最近的局部高点A（不限必须是PeakDetector检测的峰点）
            # 局部高点定义：最高价比前后各1天都高的点
            peak_a_date = None
            peak_a_price = 0.0
            peak_a_macd = 0.0

            # 从峰点B之前的数据中找最近的局部高点
            # 遍历峰点B之前的数据，找最高价比前后1天都高的点
            for j in range(len(self._df) - 1, -1, -1):
                date_j = self._df.index[j]
                # 只考虑峰点B之前的数据
                if date_j >= peak_b.date:
                    continue
                # 只考虑距离峰点B至少5天的数据（MIN_KLINE_INTERVAL）
                days_to_b = (peak_b.date - date_j).days
                if days_to_b < self.MIN_KLINE_INTERVAL:
                    continue
                if days_to_b > self.MAX_DAYS_INTERVAL:
                    continue
                # 检查是否是局部高点（比前后各1天都高）
                high_j = self._df['high'].iloc[j]
                # 检查前1天
                if j > 0:
                    high_prev = self._df['high'].iloc[j - 1]
                    if high_j <= high_prev:
                        continue
                # 检查后1天
                if j < len(self._df) - 1:
                    high_next = self._df['high'].iloc[j + 1]
                    if high_j <= high_next:
                        continue
                # 找到了局部高点，记录并退出（只找最近的）
                peak_a_date = date_j
                peak_a_price = high_j
                peak_a_macd = self._get_value_at_date(date_j, macd_series)
                break

            # 如果没找到局部高点A，跳过此峰点B
            if peak_a_date is None:
                continue

            if np.isnan(peak_a_macd) or np.isnan(macd_b):
                continue

            # 检查条件1：B最高价 > A最高价（价格创新高）
            if high_b <= peak_a_price:
                continue

            # 检查条件2：B的MACD柱 < A的MACD柱（动能减弱）
            # 【关键】如果最近的局部高点A不满足MACD减弱条件，不形成背离
            if macd_b >= peak_a_macd:
                # 最近局部高点不满足背离条件，跳过此峰点B
                logger.info(f"[日线] {peak_b.date.strftime('%Y-%m-%d')} vs 最近局部高点{peak_a_date.strftime('%Y-%m-%d')}: MACD条件不满足 ({macd_b:.4f} >= {peak_a_macd:.4f}), 不形成顶背离")
                continue

            # 计算间隔天数
            days_interval = (peak_b.date - peak_a_date).days

            # 计算回调幅度（使用最低价计算回调）
            retracement, min_date = self.calculate_retracement(
                PeakInfo(date=peak_a_date, price=peak_a_price, prominence=0, atr_value=0, is_peak=True),
                peak_b
            )

            # 检查条件5：回调幅度≤30%
            if retracement > self.MAX_RETRACEMENT:
                logger.debug(f"[日线] 回调幅度过大: {retracement:.1%} > {self.MAX_RETRACEMENT:.1%}")
                continue

            # 所有条件满足，形成顶背离
            divergence = DivergenceSignal(
                date=peak_b.date,
                divergence_type=DivergenceType.TOP,
                timeframe=TimeFrame.DAILY,
                peak_a_date=peak_a_date,
                peak_a_price=peak_a_price,  # 使用局部高点价格
                peak_b_price=high_b,   # 使用最高价
                peak_a_macd=peak_a_macd,
                peak_b_macd=macd_b,
                days_interval=days_interval,
                retracement_pct=retracement,
                is_confirmed=False,
                divergence_flag=1
            )

            divergences.append(divergence)
            logger.info(f"[日线顶背离形成] {peak_b.date.strftime('%Y-%m-%d')} | "
                       f"局部高点A: {peak_a_date.strftime('%Y-%m-%d')} 最高价={peak_a_price:.3f} MACD={peak_a_macd:.4f} | "
                       f"高点B: 最高价={high_b:.3f} MACD={macd_b:.4f} | "
                       f"间隔={days_interval}天 | 回调={retracement:.1%}")

        logger.info(f"[日线] 检测到 {len(divergences)} 个顶背离形成信号")
        return divergences

    def detect_weekly_top_divergence(self) -> List[DivergenceSignal]:
        """
        检测周线顶背离（使用最高价判断）

        Returns:
            周线顶背离信号列表
        """
        if self._weekly_df is None or self._weekly_macd is None:
            logger.error("请先调用 prepare_data() 准备周线数据")
            return []

        if self._weekly_peak_detector is None:
            logger.error("周线PeakDetector未初始化")
            return []

        # 使用周线专用的PeakDetector检测峰点
        weekly_peaks = self._weekly_peak_detector.detect_peaks(
            self._weekly_df.index[0].strftime('%Y-%m-%d'),
            self._weekly_df.index[-1].strftime('%Y-%m-%d')
        )

        if len(weekly_peaks) < 2:
            logger.info("[周线] 峰点数量不足，无法检测顶背离")
            return []

        logger.info(f"[周线] 检测到 {len(weekly_peaks)} 个峰点（使用最高价）")

        divergences = []

        # 周线参数调整（根据峰点稀疏程度调整）
        max_weeks_interval = 20  # 最大20周间隔（适配min_distance=3周）
        min_weeks_interval = 3   # 最小3周间隔（与PeakDetector的min_distance一致）

        # 【核心修正】周线顶背离只比较相邻的两个峰点，不应该跳过中间的峰点
        for i in range(len(weekly_peaks) - 1, 0, -1):
            peak_b = weekly_peaks[i]
            peak_a = weekly_peaks[i - 1]  # 只比较相邻的前一个峰点

            macd_b = self._get_weekly_value_at_date(peak_b.date, self._weekly_macd)
            high_b = peak_b.price  # 最高价

            weeks_interval = (peak_b.date - peak_a.date).days // 7

            if weeks_interval < min_weeks_interval:
                continue
            if weeks_interval > max_weeks_interval:
                continue

            macd_a = self._get_weekly_value_at_date(peak_a.date, self._weekly_macd)
            high_a = peak_a.price  # 最高价

            # 检查条件：B最高价 > A最高价
            if high_b <= high_a:
                continue
            # 检查条件：B的MACD柱 < A的MACD柱（跳过NaN值）
            if np.isnan(macd_a) or np.isnan(macd_b):
                continue
            if macd_b >= macd_a:
                # 【关键修正】相邻峰点不满足MACD减弱条件，不形成背离
                continue

            divergence = DivergenceSignal(
                date=peak_b.date,
                divergence_type=DivergenceType.TOP,
                timeframe=TimeFrame.WEEKLY,
                peak_a_date=peak_a.date,
                peak_a_price=high_a,  # 使用最高价
                peak_b_price=high_b,   # 使用最高价
                peak_a_macd=macd_a,
                peak_b_macd=macd_b,
                days_interval=weeks_interval * 7,
                retracement_pct=0.0,
                is_confirmed=False,
                divergence_flag=1
            )

            divergences.append(divergence)
            logger.info(f"[周线顶背离形成] {peak_b.date.strftime('%Y-%m-%d')} | "
                       f"高点A: {peak_a.date.strftime('%Y-%m-%d')} 最高价={high_a:.3f} MACD={macd_a:.4f} | "
                       f"高点B: 最高价={high_b:.3f} MACD={macd_b:.4f} | "
                       f"间隔={weeks_interval}周")

        logger.info(f"[周线] 检测到 {len(divergences)} 个顶背离形成信号")
        return divergences

    def detect_daily_bottom_divergence(self) -> List[DivergenceSignal]:
        """
        检测日线底背离

        底背离逻辑：
        1. 低点B价格 < 低点A价格（价格创新低）
        2. 低点B的MACD柱 > 低点A的MACD柱（动能增强）
        3. 时间间隔 ≤ 60日
        4. 回调幅度 ≤ 30%

        Returns:
            底背离信号列表
        """
        if self._peak_detector is None or self._macd_calculator is None:
            logger.error("请先调用 prepare_data() 准备数据")
            return []

        # 获取日线谷点列表（使用最低价检测）
        troughs = self._peak_detector.detect_troughs(
            self._df.index[0].strftime('%Y-%m-%d'),
            self._df.index[-1].strftime('%Y-%m-%d')
        )

        if len(troughs) < 2:
            logger.info("[日线] 谷点数量不足，无法检测底背离")
            return []

        logger.info(f"[日线] 检测到 {len(troughs)} 个谷点（用于底背离检测）")

        # 输出所有谷点的详细信息
        macd_series = self._macd_calculator._macd_series
        for i, trough in enumerate(troughs):
            macd_val = self._get_value_at_date(trough.date, macd_series)
            macd_str = f"{macd_val:.4f}" if not np.isnan(macd_val) else "N/A"
            logger.info(f"  [谷点{i+1}] {trough.date.strftime('%Y-%m-%d')} | 最低价={trough.price:.3f} | MACD={macd_str}")

        divergences = []

        # 【核心修正】底背离只比较相邻的两个谷点，不应该跳过中间的谷点
        # 从最新的谷点开始，只与相邻的前一个谷点比较
        for i in range(len(troughs) - 1, 0, -1):
            trough_b = troughs[i]  # 当前谷点B（最新）
            trough_a = troughs[i - 1]  # 相邻的前一个谷点A（只比较相邻）

            # 获取谷点B的MACD柱值和最低价
            macd_b = self._get_value_at_date(trough_b.date, macd_series)
            low_b = trough_b.price  # 最低价

            # 计算间隔天数
            days_interval = (trough_b.date - trough_a.date).days

            # 检查条件1：间隔≥5根K线
            if days_interval < self.MIN_KLINE_INTERVAL:
                continue

            # 检查条件2：间隔≤60天
            if days_interval > self.MAX_DAYS_INTERVAL:
                continue

            # 获取谷点A的MACD柱值和最低价
            macd_a = self._get_value_at_date(trough_a.date, macd_series)
            low_a = trough_a.price  # 最低价

            # 检查条件3：B最低价 < A最低价（价格创新低）
            if low_b >= low_a:
                continue

            # 检查条件4：B的MACD柱 > A的MACD柱（动能增强）
            # 底背离时MACD柱通常为负值，B > A 表示B的负值更小（动能改善）
            if np.isnan(macd_a) or np.isnan(macd_b):
                continue
            if macd_b <= macd_a:
                # 【关键修正】相邻谷点不满足MACD增强条件，不形成背离
                continue

            # 计算反弹幅度（使用最高价计算反弹）
            retracement, max_date = self._calculate_bottom_retracement(trough_a, trough_b)

            # 检查条件5：反弹幅度≤30%
            if retracement > self.MAX_RETRACEMENT:
                logger.debug(f"[日线] 反弹幅度过大: {retracement:.1%} > {self.MAX_RETRACEMENT:.1%}")
                continue

            # 所有条件满足，形成底背离
            divergence = DivergenceSignal(
                date=trough_b.date,
                divergence_type=DivergenceType.BOTTOM,
                timeframe=TimeFrame.DAILY,
                peak_a_date=trough_a.date,
                peak_a_price=low_a,  # 使用最低价
                peak_b_price=low_b,   # 使用最低价
                peak_a_macd=macd_a,
                peak_b_macd=macd_b,
                days_interval=days_interval,
                retracement_pct=retracement,
                is_confirmed=False,
                divergence_flag=1
            )

            divergences.append(divergence)
            logger.info(f"[日线底背离形成] {trough_b.date.strftime('%Y-%m-%d')} | "
                       f"低点A: {trough_a.date.strftime('%Y-%m-%d')} 最低价={low_a:.3f} MACD={macd_a:.4f} | "
                       f"低点B: 最低价={low_b:.3f} MACD={macd_b:.4f} | "
                       f"间隔={days_interval}天 | 反弹={retracement:.1%}")

        logger.info(f"[日线] 检测到 {len(divergences)} 个底背离形成信号")
        return divergences

    def _calculate_bottom_retracement(self, trough_a: PeakInfo, trough_b: PeakInfo) -> Tuple[float, pd.Timestamp]:
        """
        计算两个低点之间的最大反弹幅度（用于底背离验证）

        Args:
            trough_a: 低点A
            trough_b: 低点B

        Returns:
            (反弹幅度百分比, 反弹最高点日期)
        """
        try:
            # 获取两点之间的数据
            date_a = trough_a.date
            date_b = trough_b.date

            between_data = self._df['high'][
                (self._df.index > date_a) & (self._df.index < date_b)
            ]

            if len(between_data) == 0:
                return 0.0, date_a

            max_high = between_data.max()
            max_date = between_data.idxmax()

            # 反弹幅度 = (最高点 - 低点A) / 低点A
            retracement = (max_high - trough_a.price) / trough_a.price

            return retracement, max_date
        except Exception:
            return 0.0, trough_a.date

    def detect_weekly_bottom_divergence(self) -> List[DivergenceSignal]:
        """
        检测周线底背离（使用最低价判断）

        Returns:
            周线底背离信号列表
        """
        if self._weekly_df is None or self._weekly_macd is None:
            logger.error("请先调用 prepare_data() 准备周线数据")
            return []

        if self._weekly_peak_detector is None:
            logger.error("周线PeakDetector未初始化")
            return []

        # 使用周线专用的PeakDetector检测谷点
        weekly_troughs = self._weekly_peak_detector.detect_troughs(
            self._weekly_df.index[0].strftime('%Y-%m-%d'),
            self._weekly_df.index[-1].strftime('%Y-%m-%d')
        )

        if len(weekly_troughs) < 2:
            logger.info("[周线] 谷点数量不足，无法检测底背离")
            return []

        logger.info(f"[周线] 检测到 {len(weekly_troughs)} 个谷点（用于底背离检测）")

        divergences = []

        # 周线参数调整
        max_weeks_interval = 20
        min_weeks_interval = 3

        # 【核心修正】周线底背离只比较相邻的两个谷点，不应该跳过中间的谷点
        for i in range(len(weekly_troughs) - 1, 0, -1):
            trough_b = weekly_troughs[i]
            trough_a = weekly_troughs[i - 1]  # 只比较相邻的前一个谷点

            macd_b = self._get_weekly_value_at_date(trough_b.date, self._weekly_macd)
            low_b = trough_b.price  # 最低价

            weeks_interval = (trough_b.date - trough_a.date).days // 7

            if weeks_interval < min_weeks_interval:
                continue
            if weeks_interval > max_weeks_interval:
                continue

            macd_a = self._get_weekly_value_at_date(trough_a.date, self._weekly_macd)
            low_a = trough_a.price  # 最低价

            # 检查条件：B最低价 < A最低价（价格创新低）
            if low_b >= low_a:
                continue
            # 检查条件：B的MACD柱 > A的MACD柱（动能增强）
            if np.isnan(macd_a) or np.isnan(macd_b):
                continue
            if macd_b <= macd_a:
                # 【关键修正】相邻谷点不满足MACD增强条件，不形成背离
                continue

            divergence = DivergenceSignal(
                date=trough_b.date,
                divergence_type=DivergenceType.BOTTOM,
                timeframe=TimeFrame.WEEKLY,
                peak_a_date=trough_a.date,
                peak_a_price=low_a,
                peak_b_price=low_b,
                peak_a_macd=macd_a,
                peak_b_macd=macd_b,
                days_interval=weeks_interval * 7,
                retracement_pct=0.0,
                is_confirmed=False,
                divergence_flag=1
            )

            divergences.append(divergence)
            logger.info(f"[周线底背离形成] {trough_b.date.strftime('%Y-%m-%d')} | "
                       f"低点A: {trough_a.date.strftime('%Y-%m-%d')} 最低价={low_a:.3f} MACD={macd_a:.4f} | "
                       f"低点B: 最低价={low_b:.3f} MACD={macd_b:.4f} | "
                       f"间隔={weeks_interval}周")

        logger.info(f"[周线] 检测到 {len(divergences)} 个底背离形成信号")
        return divergences

    def check_divergence_confirmation(self, divergences: List[DivergenceSignal]) -> List[DivergenceSignal]:
        """
        检查背离是否已形成（生效）

        【重要说明】
        根据buy_sell.md规则，背离"形成"即视为"生效"，不需要MA5<MA10等额外确认条件。
        卖出触发完全由sell_id的条件决定（如SAR死叉、MACD死叉）。

        Args:
            divergences: 背离信号列表

        Returns:
            已形成的背离信号列表（直接返回传入的列表）
        """
        confirmed = []

        for div in divergences:
            # 背离形成即视为生效，不需要额外确认条件
            # 卖出触发由sell_id的逻辑决定
            div.is_confirmed = True
            # 设置confirmation_date为背离形成日期
            div.confirmation_date = div.date

            # 根据背离类型输出不同的日志
            if div.divergence_type == DivergenceType.TOP:
                logger.info(f"[{div.timeframe.value}顶背离形成] {div.date.strftime('%Y-%m-%d')} "
                           f"高点A({div.peak_a_date.strftime('%Y-%m-%d')}) 价格={div.peak_a_price:.3f} MACD={div.peak_a_macd:.4f} | "
                           f"高点B({div.date.strftime('%Y-%m-%d')}) 价格={div.peak_b_price:.3f} MACD={div.peak_b_macd:.4f} | "
                           f"待sell_id触发条件（SAR死叉/MACD死叉）触发卖出")
            else:  # BOTTOM
                logger.info(f"[{div.timeframe.value}底背离形成] {div.date.strftime('%Y-%m-%d')} "
                           f"低点A({div.peak_a_date.strftime('%Y-%m-%d')}) 价格={div.peak_a_price:.3f} MACD={div.peak_a_macd:.4f} | "
                           f"低点B({div.date.strftime('%Y-%m-%d')}) 价格={div.peak_b_price:.3f} MACD={div.peak_b_macd:.4f} | "
                           f"待buy_id触发条件（SAR金叉/MACD金叉）触发买入")

            confirmed.append(div)

        return confirmed

    def _get_value_at_date(self, date: pd.Timestamp, series: pd.Series) -> float:
        """获取日线指定日期的值"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = series[series.index.strftime('%Y-%m-%d') == date_str]
            if len(matching) > 0:
                return matching.iloc[0]
            return 0.0
        except Exception:
            return 0.0

    def _get_weekly_value_at_date(self, date: pd.Timestamp, series: pd.Series) -> float:
        """获取周线指定日期的值"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = series[series.index.strftime('%Y-%m-%d') == date_str]
            if len(matching) > 0:
                return matching.iloc[0]
            return 0.0
        except Exception:
            return 0.0

    def detect_all_divergences(self, buy_signal_date: Optional[pd.Timestamp] = None) -> Dict[str, List[DivergenceSignal]]:
        """
        检测所有背离信号

        Args:
            buy_signal_date: 最近一次买入信号（SAR金叉）日期，用于监控模式确定peak B范围
                            如果传入此参数，监控模式只判断买入信号后最新高点的顶背离

        Returns:
            包含各类背离信号的字典（顶背离和底背离）
        """
        logger.info("\n" + "=" * 60)
        logger.info("开始背离检测（按照背离判断.md逻辑）")
        if buy_signal_date:
            logger.info(f"监控模式：买入信号日期={buy_signal_date.strftime('%Y-%m-%d')}")
        logger.info("=" * 60)

        results = {
            'daily_top_formed': [],      # 日线顶背离形成（形成即生效）
            'daily_top_confirmed': [],   # 日线顶背离生效（等同于形成）
            'weekly_top_formed': [],     # 周线顶背离形成（形成即生效）
            'weekly_top_confirmed': [],  # 周线顶背离生效（等同于形成）
            'daily_bottom_formed': [],   # 日线底背离形成（形成即生效）
            'daily_bottom_confirmed': [], # 日线底背离生效（等同于形成）
            'weekly_bottom_formed': [],  # 周线底背离形成（形成即生效）
            'weekly_bottom_confirmed': [] # 周线底背离生效（等同于形成）
        }

        # 日线顶背离检测
        logger.info("\n--- 日线顶背离检测 ---")
        daily_top = self.detect_daily_top_divergence(buy_signal_date)
        results['daily_top_formed'] = daily_top

        # 背离形成即视为生效（不需要额外确认）
        logger.info("\n--- 日线顶背离生效（形成即生效）---")
        daily_confirmed = self.check_divergence_confirmation(daily_top)
        results['daily_top_confirmed'] = daily_confirmed

        # 日线底背离检测
        logger.info("\n--- 日线底背离检测 ---")
        daily_bottom = self.detect_daily_bottom_divergence()
        results['daily_bottom_formed'] = daily_bottom

        logger.info("\n--- 日线底背离生效（形成即生效）---")
        daily_bottom_confirmed = self.check_divergence_confirmation(daily_bottom)
        results['daily_bottom_confirmed'] = daily_bottom_confirmed

        # 周线顶背离检测
        logger.info("\n--- 周线顶背离检测 ---")
        weekly_top = self.detect_weekly_top_divergence()
        results['weekly_top_formed'] = weekly_top

        # 背离形成即视为生效
        logger.info("\n--- 周线顶背离生效（形成即生效）---")
        weekly_confirmed = self.check_divergence_confirmation(weekly_top)
        results['weekly_top_confirmed'] = weekly_confirmed

        # 周线底背离检测
        logger.info("\n--- 周线底背离检测 ---")
        weekly_bottom = self.detect_weekly_bottom_divergence()
        results['weekly_bottom_formed'] = weekly_bottom

        logger.info("\n--- 周线底背离生效（形成即生效）---")
        weekly_bottom_confirmed = self.check_divergence_confirmation(weekly_bottom)
        results['weekly_bottom_confirmed'] = weekly_bottom_confirmed

        # 统计汇总
        logger.info("\n" + "=" * 60)
        logger.info("背离检测完成 - 统计汇总")
        logger.info("=" * 60)
        logger.info(f"日线顶背离: {len(results['daily_top_formed'])} 个（形成即生效）")
        logger.info(f"日线底背离: {len(results['daily_bottom_formed'])} 个（形成即生效）")
        logger.info(f"周线顶背离: {len(results['weekly_top_formed'])} 个（形成即生效）")
        logger.info(f"周线底背离: {len(results['weekly_bottom_formed'])} 个（形成即生效）")
        logger.info("注意：顶背离形成后，卖出由sell_id的触发条件（SAR死叉/MACD死叉）决定")
        logger.info("注意：底背离形成后，买入由buy_id的触发条件（SAR金叉/MACD金叉）决定")

        return results

    def plot_divergences(
        self,
        divergences: Dict[str, List[DivergenceSignal]],
        output_path: Optional[str] = None
    ) -> str:
        """
        绘制背离信号图表

        Args:
            divergences: 背离信号字典
            output_path: 输出路径

        Returns:
            图表保存路径
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 获取当前日期和最新数据
        current_date = self._df.index[-1].strftime('%Y-%m-%d')
        current_price = self._df['close'].iloc[-1]
        current_atr = self._peak_detector._atr_series.iloc[-1] if self._peak_detector._atr_series is not None else 0

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # 日线价格走势 + 均线 + ATR
        ax1 = axes[0]
        ax1.plot(self._df.index, self._df['close'], 'b-', linewidth=1, label='收盘价')
        ax1.plot(self._df.index, self._ma5, 'y-', linewidth=1, label='MA5')
        ax1.plot(self._df.index, self._ma10, 'm-', linewidth=1, label='MA10')
        ax1.plot(self._df.index, self._ma60, 'c-', linewidth=1, label='MA60')

        # 创建第二个y轴显示ATR
        ax1_atr = ax1.twinx()
        atr_series = self._peak_detector._atr_series
        ax1_atr.plot(self._df.index, atr_series, 'g--', linewidth=1, alpha=0.7, label='ATR')
        ax1_atr.set_ylabel('ATR', color='green')
        ax1_atr.tick_params(axis='y', labelcolor='green')

        ax1.set_ylabel('价格')
        ax1.set_title(f'{self._symbol} 日线顶背离分析\n'
                     f'当前日期: {current_date} | 收盘价: {current_price:.3f} | ATR: {current_atr:.4f}')
        ax1.legend(loc='upper left')
        ax1_atr.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 标记日线顶背离形成（红色三角形）
        for div in divergences['daily_top_formed']:
            ax1.scatter(div.date, div.peak_b_price, color='red', s=100, marker='^', zorder=6)
            ax1.scatter(div.peak_a_date, div.peak_a_price, color='orange', s=80, marker='o', zorder=5)
            ax1.annotate(f'顶背离形成\nA:{div.peak_a_date.strftime("%Y-%m-%d")}\nB:{div.date.strftime("%Y-%m-%d")}',
                        (div.date, div.peak_b_price),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=7, color='red')

        # 标记日线顶背离生效（大红色三角形）
        for div in divergences['daily_top_confirmed']:
            ax1.scatter(div.date, div.peak_b_price, color='darkred', s=200, marker='^', zorder=7)
            ax1.annotate(f'生效\n卖出1/3',
                        (div.date, div.peak_b_price),
                        textcoords="offset points", xytext=(5, 25),
                        fontsize=9, color='darkred', fontweight='bold')

        # 日线MACD柱
        ax2 = axes[1]
        macd_series = self._macd_calculator._macd_series
        colors = ['red' if v > 0 else 'green' for v in macd_series]
        ax2.bar(self._df.index, macd_series, color=colors, width=0.8, label='日线MACD柱')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('MACD柱')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 标记MACD背离点
        for div in divergences['daily_top_formed']:
            ax2.scatter(div.date, div.peak_b_macd, color='red', s=80, marker='^')
            ax2.scatter(div.peak_a_date, div.peak_a_macd, color='orange', s=60, marker='o')

        # 周线价格走势
        ax3 = axes[2]
        ax3.plot(self._weekly_df.index, self._weekly_df['close'], 'b-', linewidth=2, label='周线收盘价')
        ax3.set_ylabel('价格')
        ax3.set_title('周线顶背离分析')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 标记周线顶背离形成
        for div in divergences['weekly_top_formed']:
            ax3.scatter(div.date, div.peak_b_price, color='purple', s=150, marker='^', zorder=6)
            ax3.scatter(div.peak_a_date, div.peak_a_price, color='violet', s=100, marker='o', zorder=5)
            ax3.annotate(f'周线顶背离\nA:{div.peak_a_date.strftime("%Y-%m-%d")}\nB:{div.date.strftime("%Y-%m-%d")}',
                        (div.date, div.peak_b_price),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=9, color='purple')

        # 标记周线顶背离生效（清仓信号）
        for div in divergences['weekly_top_confirmed']:
            ax3.scatter(div.date, div.peak_b_price, color='darkviolet', s=250, marker='^', zorder=7)
            ax3.annotate(f'生效\n清仓',
                        (div.date, div.peak_b_price),
                        textcoords="offset points", xytext=(5, 25),
                        fontsize=11, color='darkviolet', fontweight='bold')

        # 周线MACD柱
        ax4 = axes[3]
        weekly_colors = ['red' if v > 0 else 'green' for v in self._weekly_macd]
        ax4.bar(self._weekly_df.index, self._weekly_macd, color=weekly_colors, width=5, label='周线MACD柱')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('MACD柱')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

        # 设置x轴日期格式
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # 添加统计信息
        stats_text = (f"日线: 形成{len(divergences['daily_top_formed'])}个, 生效{len(divergences['daily_top_confirmed'])}个 | "
                     f"周线: 形成{len(divergences['weekly_top_formed'])}个, 生效{len(divergences['weekly_top_confirmed'])}个")
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10)

        plt.tight_layout()

        # 保存图表
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"logs/divergence_{self._symbol.replace('.', '')}_{timestamp}.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"图表已保存: {output_path}")
        return output_path


# ==============================================================================
# 测试模块
# ==============================================================================

def run_demo():
    """背离检测器演示测试"""
    from gm.api import set_token
    import json

    # 设置日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f"logs/divergence_{timestamp}.log"
    file_handler = setup_log_file(log_path)

    # 加载配置设置token
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config', 'settings.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            gm_token = config.get('gm_token')
            if gm_token:
                set_token(gm_token)
                logger.info("已设置GM Token")
            else:
                logger.warning("未找到GM Token，请检查配置文件")
    else:
        logger.error(f"配置文件不存在: {config_path}")
        logger.removeHandler(file_handler)
        return

    logger.info("=" * 60)
    logger.info("背离检测器测试（按照背离判断.md逻辑）")
    logger.info("=" * 60)

    # 测试参数
    symbol = 'SHSE.510880'
    # 注意：需要包含预热数据（提前1年），否则峰点检测结果不同
    start_date = '2023-01-01'  # 预热数据起始
    test_start_date = '2024-01-01'  # 实际检测起始
    end_date = '2026-04-10'

    logger.info(f"测试股票: 上证红利ETF (510880)")
    logger.info(f"数据时间: {start_date} ~ {end_date} (包含预热)")
    logger.info(f"检测时间: {test_start_date} ~ {end_date}")

    # 创建检测器
    detector = DivergenceDetector()

    # 准备数据（包含预热）
    df = detector.prepare_data(symbol, start_date, end_date)

    if df.empty:
        logger.error("数据获取失败，测试终止")
        logger.removeHandler(file_handler)
        return

    # 检测所有背离
    divergences = detector.detect_all_divergences()

    # 绘制图表
    logger.info("\n生成背离图表...")
    chart_path = detector.plot_divergences(divergences)

    logger.info("\n" + "=" * 60)
    logger.info("测试完成!")
    logger.info(f"日志已保存: {log_path}")
    logger.info(f"图表已保存: {chart_path}")
    logger.info("=" * 60)

    # 关闭日志文件
    logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == '__main__':
    run_demo()