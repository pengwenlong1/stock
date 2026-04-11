# -*- coding: utf-8 -*-
"""
波峰波谷检测模块 (peak_detector.py)

【功能说明】
检测标的在规定时间段内的所有波峰（局部高点）和波谷（局部低点）。
使用ATR作为波峰突出程度的衡量标准，确保检测到的波峰/波谷具有实际意义。

【核心逻辑】
1. 波峰检测：局部高点，突出程度 >= ATR * factor
2. 波谷检测：局部低点，下降程度 >= ATR * factor
3. ATR计算：使用talib.ATR(high, low, close, timeperiod=14)

【数学描述】
设 price(t) 为价格序列，波峰定义：
- price(t) > price(t-1) 且 price(t) > price(t+1)（局部高点）
- prominence >= ATR * factor（突出程度足够显著）

波谷定义：
- price(t) < price(t-1) 且 price(t) < price(t+1)（局部低点）
- prominence >= ATR * factor（下降程度足够显著）

作者：量化交易团队
创建日期：2024
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
from scipy.signal import find_peaks

try:
    import talib
except ImportError:
    talib = None
    logging.warning("talib未安装，将使用自定义ATR计算")

logger = logging.getLogger(__name__)


@dataclass
class PeakInfo:
    """波峰信息"""
    date: pd.Timestamp      # 波峰日期
    price: float            # 波峰价格
    prominence: float       # 突出程度
    atr_value: float        # 当时的ATR值
    is_peak: bool = True    # True表示波峰，False表示波谷


class PeakDetector:
    """
    波峰波谷检测器

    【功能说明】
    检测标的在规定时间段内的所有波峰和波谷。

    【参数配置】
    - atr_period: ATR计算周期，默认14
    - prominence_factor: 波峰突出程度因子，默认0.5（即>=0.5倍ATR）
    - min_distance: 波峰之间的最小距离（天数），默认5

    【使用方法】
    detector = PeakDetector(atr_period=14, prominence_factor=0.5)
    detector.prepare_data(high_series, low_series, close_series)
    peaks = detector.detect_peaks(start_date, end_date)
    troughs = detector.detect_troughs(start_date, end_date)
    all_extremes = detector.detect_all_extremes(start_date, end_date)
    """

    def __init__(self,
                 atr_period: int = 14,
                 prominence_factor: float = 0.5,
                 min_distance: int = 5) -> None:
        """
        初始化波峰检测器

        Args:
            atr_period: ATR计算周期
            prominence_factor: 波峰突出程度因子（相对于ATR）
            min_distance: 波峰之间的最小距离（天数）
        """
        self.atr_period = atr_period
        self.prominence_factor = prominence_factor
        self.min_distance = min_distance

        # 数据存储（日线）
        self._high_series: Optional[pd.Series] = None
        self._low_series: Optional[pd.Series] = None
        self._close_series: Optional[pd.Series] = None
        self._atr_series: Optional[pd.Series] = None

        # 数据存储（周线）
        self._weekly_high: Optional[pd.Series] = None
        self._weekly_low: Optional[pd.Series] = None
        self._weekly_close: Optional[pd.Series] = None
        self._weekly_atr: Optional[pd.Series] = None
        self._weekly_dates: Optional[List[pd.Timestamp]] = None  # 周线对应的结束日期

    def prepare_data(self,
                      high_series: pd.Series,
                      low_series: pd.Series,
                      close_series: pd.Series) -> None:
        """
        准备计算数据

        Args:
            high_series: 最高价序列
            low_series: 最低价序列
            close_series: 收盘价序列
        """
        if not isinstance(high_series, pd.Series):
            raise TypeError(f"high_series必须为pd.Series")
        if not isinstance(low_series, pd.Series):
            raise TypeError(f"low_series必须为pd.Series")
        if not isinstance(close_series, pd.Series):
            raise TypeError(f"close_series必须为pd.Series")

        self._high_series = high_series
        self._low_series = low_series
        self._close_series = close_series

        # 计算日线ATR
        self._atr_series = self._calculate_atr(high_series, low_series, close_series)

        # 聚合周线数据并计算周线ATR
        self._prepare_weekly_data()

        logger.info(f"波峰检测器数据准备完成: {len(close_series)} 天数据, 周线 {len(self._weekly_close) if self._weekly_close is not None else 0} 周")

    def _prepare_weekly_data(self) -> None:
        """
        将日线数据聚合为周线数据

        【聚合规则】
        - 周最高价 = 该周所有日线最高价的最大值
        - 周最低价 = 该周所有日线最低价的最小值
        - 周收盘价 = 该周最后一天的收盘价
        - 周线日期 = 该周最后一天的日期（周五或最后一个交易日）
        """
        if self._high_series is None or self._low_series is None or self._close_series is None:
            return

        # 获取周线数据索引（使用周一作为周标识）
        weekly_high_list = []
        weekly_low_list = []
        weekly_close_list = []
        weekly_dates_list = []

        # 创建DataFrame便于聚合
        df = pd.DataFrame({
            'high': self._high_series,
            'low': self._low_series,
            'close': self._close_series
        })

        # 使用 'W-FRI' 按周五结束的周聚合（中国股市每周最后一个交易日）
        weekly_df = df.groupby(df.index.to_period('W')).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })

        # 获取每周的最后一天日期
        for i, period in enumerate(weekly_df.index):
            # 找到该周内的所有交易日
            week_start = period.start_time
            week_end = period.end_time

            # 处理时区问题：如果df.index是tz-aware的，需要给week_start和week_end加上时区
            if df.index.tz is not None:
                week_start = week_start.tz_localize(df.index.tz)
                week_end = week_end.tz_localize(df.index.tz)

            # 筛选该周的日线数据
            week_data = df[(df.index >= week_start) & (df.index <= week_end)]

            if len(week_data) > 0:
                # 取该周最后一天的日期
                last_day = week_data.index[-1]
                weekly_dates_list.append(last_day)

        # 创建周线Series
        if len(weekly_df) > 0:
            self._weekly_high = pd.Series(
                weekly_df['high'].values,
                index=weekly_dates_list,
                name='Weekly_High'
            )
            self._weekly_low = pd.Series(
                weekly_df['low'].values,
                index=weekly_dates_list,
                name='Weekly_Low'
            )
            self._weekly_close = pd.Series(
                weekly_df['close'].values,
                index=weekly_dates_list,
                name='Weekly_Close'
            )
            self._weekly_dates = weekly_dates_list

            # 计算周线ATR（使用相同的周期参数）
            self._weekly_atr = self._calculate_atr(
                self._weekly_high,
                self._weekly_low,
                self._weekly_close
            )

            logger.info(f"周线数据聚合完成: {len(self._weekly_close)} 周")

    def _calculate_atr(self,
                        high: pd.Series,
                        low: pd.Series,
                        close: pd.Series) -> pd.Series:
        """
        计算ATR（Average True Range）

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            pd.Series: ATR序列
        """
        if talib is not None:
            # 使用talib计算ATR
            atr_values = talib.ATR(
                high.values,
                low.values,
                close.values,
                timeperiod=self.atr_period
            )
            atr_series = pd.Series(atr_values, index=close.index, name='ATR')
        else:
            # 自定义ATR计算
            # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR = EMA of TR
            atr_series = tr.ewm(span=self.atr_period, adjust=False).mean()
            atr_series.name = 'ATR'

        return atr_series

    def _get_atr_at_date(self, date: pd.Timestamp) -> float:
        """
        获取指定日期的ATR值

        Args:
            date: 日期

        Returns:
            float: ATR值
        """
        if self._atr_series is None:
            return 0.0

        matching = self._atr_series[
            self._atr_series.index.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d')
        ]
        if len(matching) > 0:
            return matching.iloc[0]
        return 0.0

    def detect_peaks(self,
                      start_date: str,
                      end_date: str) -> List[PeakInfo]:
        """
        检测波峰（局部高点）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[PeakInfo]: 波峰列表
        """
        if self._high_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        high_data = self._high_series[
            (self._high_series.index >= start_ts) &
            (self._high_series.index <= end_ts)
        ]
        atr_data = self._atr_series[
            (self._atr_series.index >= start_ts) &
            (self._atr_series.index <= end_ts)
        ]

        if len(high_data) == 0:
            return []

        # 使用动态ATR作为prominence阈值
        # 由于find_peaks要求固定阈值，我们使用平均ATR
        mean_atr = atr_data.mean()
        min_prominence = mean_atr * self.prominence_factor

        # 检测波峰
        peaks_indices, peaks_properties = find_peaks(
            high_data.values,
            prominence=min_prominence,
            distance=self.min_distance
        )

        # 构建波峰信息列表
        peaks_list = []
        for idx in peaks_indices:
            date = high_data.index[idx]
            price = high_data.iloc[idx]
            prominence = peaks_properties['prominences'][list(peaks_indices).index(idx)]
            atr_value = self._get_atr_at_date(date)

            peaks_list.append(PeakInfo(
                date=date,
                price=price,
                prominence=prominence,
                atr_value=atr_value,
                is_peak=True
            ))

        logger.info(f"检测到 {len(peaks_list)} 个波峰")
        return peaks_list

    def detect_troughs(self,
                        start_date: str,
                        end_date: str) -> List[PeakInfo]:
        """
        检测波谷（局部低点）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[PeakInfo]: 波谷列表
        """
        if self._low_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        low_data = self._low_series[
            (self._low_series.index >= start_ts) &
            (self._low_series.index <= end_ts)
        ]
        atr_data = self._atr_series[
            (self._atr_series.index >= start_ts) &
            (self._atr_series.index <= end_ts)
        ]

        if len(low_data) == 0:
            return []

        # 波谷检测：对负值使用find_peaks
        mean_atr = atr_data.mean()
        min_prominence = mean_atr * self.prominence_factor

        # 反转低点序列来检测波谷
        troughs_indices, troughs_properties = find_peaks(
            -low_data.values,
            prominence=min_prominence,
            distance=self.min_distance
        )

        # 构建波谷信息列表
        troughs_list = []
        for idx in troughs_indices:
            date = low_data.index[idx]
            price = low_data.iloc[idx]
            prominence = troughs_properties['prominences'][list(troughs_indices).index(idx)]
            atr_value = self._get_atr_at_date(date)

            troughs_list.append(PeakInfo(
                date=date,
                price=price,
                prominence=prominence,
                atr_value=atr_value,
                is_peak=False
            ))

        logger.info(f"检测到 {len(troughs_list)} 个波谷")
        return troughs_list

    def detect_all_extremes(self,
                              start_date: str,
                              end_date: str) -> Dict[str, List[PeakInfo]]:
        """
        检测所有波峰和波谷

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            Dict: {'peaks': [...], 'troughs': [...]}
        """
        peaks = self.detect_peaks(start_date, end_date)
        troughs = self.detect_troughs(start_date, end_date)

        return {
            'peaks': peaks,
            'troughs': troughs
        }

    def detect_peaks_dynamic_atr(self,
                                   start_date: str,
                                   end_date: str) -> List[PeakInfo]:
        """
        使用动态ATR阈值检测波峰（更精确的方法）

        每个波峰的突出程度阈值 = 当时的ATR * prominence_factor

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[PeakInfo]: 波峰列表
        """
        if self._high_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        high_data = self._high_series[
            (self._high_series.index >= start_ts) &
            (self._high_series.index <= end_ts)
        ]

        if len(high_data) == 0:
            return []

        # 先用较低阈值检测所有候选波峰
        min_atr = self._atr_series[
            (self._atr_series.index >= start_ts) &
            (self._atr_series.index <= end_ts)
        ].min()
        min_prominence = min_atr * self.prominence_factor * 0.5  # 使用较小阈值

        peaks_indices, peaks_properties = find_peaks(
            high_data.values,
            prominence=min_prominence,
            distance=self.min_distance
        )

        # 过滤：只保留突出程度 >= 当时ATR * factor 的波峰
        peaks_list = []
        prominences = peaks_properties['prominences']

        for i, idx in enumerate(peaks_indices):
            date = high_data.index[idx]
            price = high_data.iloc[idx]
            prominence = prominences[i]
            atr_value = self._get_atr_at_date(date)
            required_prominence = atr_value * self.prominence_factor

            if prominence >= required_prominence:
                peaks_list.append(PeakInfo(
                    date=date,
                    price=price,
                    prominence=prominence,
                    atr_value=atr_value,
                    is_peak=True
                ))

        logger.info(f"检测到 {len(peaks_list)} 个有效波峰（动态ATR过滤后）")
        return peaks_list

    def detect_troughs_dynamic_atr(self,
                                     start_date: str,
                                     end_date: str) -> List[PeakInfo]:
        """
        使用动态ATR阈值检测波谷（更精确的方法）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[PeakInfo]: 波谷列表
        """
        if self._low_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        low_data = self._low_series[
            (self._low_series.index >= start_ts) &
            (self._low_series.index <= end_ts)
        ]

        if len(low_data) == 0:
            return []

        # 先用较低阈值检测所有候选波谷
        min_atr = self._atr_series[
            (self._atr_series.index >= start_ts) &
            (self._atr_series.index <= end_ts)
        ].min()
        min_prominence = min_atr * self.prominence_factor * 0.5

        troughs_indices, troughs_properties = find_peaks(
            -low_data.values,
            prominence=min_prominence,
            distance=self.min_distance
        )

        # 过滤：只保留下降程度 >= 当时ATR * factor 的波谷
        troughs_list = []
        prominences = troughs_properties['prominences']

        for i, idx in enumerate(troughs_indices):
            date = low_data.index[idx]
            price = low_data.iloc[idx]
            prominence = prominences[i]
            atr_value = self._get_atr_at_date(date)
            required_prominence = atr_value * self.prominence_factor

            if prominence >= required_prominence:
                troughs_list.append(PeakInfo(
                    date=date,
                    price=price,
                    prominence=prominence,
                    atr_value=atr_value,
                    is_peak=False
                ))

        logger.info(f"检测到 {len(troughs_list)} 个有效波谷（动态ATR过滤后）")
        return troughs_list

    # ==================== 周线波峰波谷检测方法 ====================

    def detect_weekly_peaks(self,
                             start_date: str,
                             end_date: str,
                             use_dynamic_atr: bool = True) -> List[PeakInfo]:
        """
        检测周线波峰（局部高点）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            use_dynamic_atr: 是否使用动态ATR阈值

        Returns:
            List[PeakInfo]: 周线波峰列表
        """
        if self._weekly_high is None:
            raise ValueError("请先调用prepare_data()准备数据（周线数据会自动生成）")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        weekly_high_data = self._weekly_high[
            (self._weekly_high.index >= start_ts) &
            (self._weekly_high.index <= end_ts)
        ]
        weekly_atr_data = self._weekly_atr[
            (self._weekly_atr.index >= start_ts) &
            (self._weekly_atr.index <= end_ts)
        ]

        if len(weekly_high_data) == 0:
            return []

        # 周线最小距离设为1（相邻周）
        min_distance_weekly = 1

        if use_dynamic_atr:
            # 使用动态ATR阈值
            min_atr = weekly_atr_data.min()
            min_prominence = min_atr * self.prominence_factor * 0.5

            peaks_indices, peaks_properties = find_peaks(
                weekly_high_data.values,
                prominence=min_prominence,
                distance=min_distance_weekly
            )

            # 过滤：只保留突出程度 >= 当时ATR * factor 的波峰
            peaks_list = []
            prominences = peaks_properties['prominences']

            for i, idx in enumerate(peaks_indices):
                date = weekly_high_data.index[idx]
                price = weekly_high_data.iloc[idx]
                prominence = prominences[i]
                atr_value = weekly_atr_data.loc[date] if date in weekly_atr_data.index else 0.0
                required_prominence = atr_value * self.prominence_factor

                if prominence >= required_prominence:
                    peaks_list.append(PeakInfo(
                        date=date,
                        price=price,
                        prominence=prominence,
                        atr_value=atr_value,
                        is_peak=True
                    ))
        else:
            # 使用固定ATR阈值
            mean_atr = weekly_atr_data.mean()
            min_prominence = mean_atr * self.prominence_factor

            peaks_indices, peaks_properties = find_peaks(
                weekly_high_data.values,
                prominence=min_prominence,
                distance=min_distance_weekly
            )

            peaks_list = []
            for idx in peaks_indices:
                date = weekly_high_data.index[idx]
                price = weekly_high_data.iloc[idx]
                prominence = peaks_properties['prominences'][list(peaks_indices).index(idx)]
                atr_value = weekly_atr_data.loc[date] if date in weekly_atr_data.index else 0.0

                peaks_list.append(PeakInfo(
                    date=date,
                    price=price,
                    prominence=prominence,
                    atr_value=atr_value,
                    is_peak=True
                ))

        logger.info(f"检测到 {len(peaks_list)} 个周线波峰")
        return peaks_list

    def detect_weekly_troughs(self,
                               start_date: str,
                               end_date: str,
                               use_dynamic_atr: bool = True) -> List[PeakInfo]:
        """
        检测周线波谷（局部低点）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            use_dynamic_atr: 是否使用动态ATR阈值

        Returns:
            List[PeakInfo]: 周线波谷列表
        """
        if self._weekly_low is None:
            raise ValueError("请先调用prepare_data()准备数据（周线数据会自动生成）")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        weekly_low_data = self._weekly_low[
            (self._weekly_low.index >= start_ts) &
            (self._weekly_low.index <= end_ts)
        ]
        weekly_atr_data = self._weekly_atr[
            (self._weekly_atr.index >= start_ts) &
            (self._weekly_atr.index <= end_ts)
        ]

        if len(weekly_low_data) == 0:
            return []

        # 周线最小距离设为1（相邻周）
        min_distance_weekly = 1

        if use_dynamic_atr:
            # 使用动态ATR阈值
            min_atr = weekly_atr_data.min()
            min_prominence = min_atr * self.prominence_factor * 0.5

            troughs_indices, troughs_properties = find_peaks(
                -weekly_low_data.values,
                prominence=min_prominence,
                distance=min_distance_weekly
            )

            # 过滤：只保留下降程度 >= 当时ATR * factor 的波谷
            troughs_list = []
            prominences = troughs_properties['prominences']

            for i, idx in enumerate(troughs_indices):
                date = weekly_low_data.index[idx]
                price = weekly_low_data.iloc[idx]
                prominence = prominences[i]
                atr_value = weekly_atr_data.loc[date] if date in weekly_atr_data.index else 0.0
                required_prominence = atr_value * self.prominence_factor

                if prominence >= required_prominence:
                    troughs_list.append(PeakInfo(
                        date=date,
                        price=price,
                        prominence=prominence,
                        atr_value=atr_value,
                        is_peak=False
                    ))
        else:
            # 使用固定ATR阈值
            mean_atr = weekly_atr_data.mean()
            min_prominence = mean_atr * self.prominence_factor

            troughs_indices, troughs_properties = find_peaks(
                -weekly_low_data.values,
                prominence=min_prominence,
                distance=min_distance_weekly
            )

            troughs_list = []
            for idx in troughs_indices:
                date = weekly_low_data.index[idx]
                price = weekly_low_data.iloc[idx]
                prominence = troughs_properties['prominences'][list(troughs_indices).index(idx)]
                atr_value = weekly_atr_data.loc[date] if date in weekly_atr_data.index else 0.0

                troughs_list.append(PeakInfo(
                    date=date,
                    price=price,
                    prominence=prominence,
                    atr_value=atr_value,
                    is_peak=False
                ))

        logger.info(f"检测到 {len(troughs_list)} 个周线波谷")
        return troughs_list

    def detect_weekly_all_extremes(self,
                                    start_date: str,
                                    end_date: str,
                                    use_dynamic_atr: bool = True) -> Dict[str, List[PeakInfo]]:
        """
        检测所有周线波峰和波谷

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            use_dynamic_atr: 是否使用动态ATR阈值

        Returns:
            Dict: {'peaks': [...], 'troughs': [...]}
        """
        peaks = self.detect_weekly_peaks(start_date, end_date, use_dynamic_atr)
        troughs = self.detect_weekly_troughs(start_date, end_date, use_dynamic_atr)

        return {
            'peaks': peaks,
            'troughs': troughs
        }

    def plot_weekly_peaks_troughs(self,
                                   start_date: str,
                                   end_date: str,
                                   save_path: str,
                                   stock_name: str = '',
                                   use_dynamic_atr: bool = True) -> None:
        """
        绘制周线波峰波谷图表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            save_path: 图片保存路径
            stock_name: 股票名称
            use_dynamic_atr: 是否使用动态ATR阈值
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        if self._weekly_close is None:
            logger.warning("无周线数据，无法绘制图表")
            return

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        weekly_close_data = self._weekly_close[
            (self._weekly_close.index >= start_ts) &
            (self._weekly_close.index <= end_ts)
        ]
        weekly_high_data = self._weekly_high[
            (self._weekly_high.index >= start_ts) &
            (self._weekly_high.index <= end_ts)
        ]
        weekly_low_data = self._weekly_low[
            (self._weekly_low.index >= start_ts) &
            (self._weekly_low.index <= end_ts)
        ]
        weekly_atr_data = self._weekly_atr[
            (self._weekly_atr.index >= start_ts) &
            (self._weekly_atr.index <= end_ts)
        ]

        if len(weekly_close_data) == 0:
            return

        # 检测周线波峰波谷
        peaks = self.detect_weekly_peaks(start_date, end_date, use_dynamic_atr)
        troughs = self.detect_weekly_troughs(start_date, end_date, use_dynamic_atr)

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'{stock_name} 周线波峰波谷检测 ({start_date} ~ {end_date})',
                    fontsize=14, fontweight='bold')

        ax_price = axes[0]
        ax_atr = axes[1]

        # 绘制周线价格曲线
        dates = weekly_close_data.index
        ax_price.plot(dates, weekly_close_data.values, 'b-', linewidth=2, label='周收盘价', alpha=0.8)

        # 绘制周线高低点区间
        ax_price.fill_between(dates, weekly_low_data.values, weekly_high_data.values,
                              color='blue', alpha=0.3, label='周内波动区间')

        # 标记周线波峰
        if peaks:
            peak_dates = [p.date for p in peaks]
            peak_prices = [p.price for p in peaks]
            ax_price.scatter(peak_dates, peak_prices, color='red', marker='^', s=150,
                            label=f'周线波峰({len(peaks)}个)', zorder=5)
            for p in peaks:
                ax_price.annotate(f'{p.price:.2f}\n突出:{p.prominence:.3f}',
                                 (p.date, p.price),
                                 textcoords="offset points", xytext=(5, 10),
                                 fontsize=8, color='red')

        # 标记周线波谷
        if troughs:
            trough_dates = [t.date for t in troughs]
            trough_prices = [t.price for t in troughs]
            ax_price.scatter(trough_dates, trough_prices, color='green', marker='v', s=150,
                            label=f'周线波谷({len(troughs)}个)', zorder=5)
            for t in troughs:
                ax_price.annotate(f'{t.price:.2f}\n下降:{t.prominence:.3f}',
                                 (t.date, t.price),
                                 textcoords="offset points", xytext=(5, -15),
                                 fontsize=8, color='green')

        ax_price.set_ylabel('价格')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # 绘制周线ATR曲线
        ax_atr.plot(dates, weekly_atr_data.values, 'purple', linewidth=1.5, label='周线ATR(14)')
        ax_atr.axhline(y=weekly_atr_data.mean(), color='red', linestyle='--',
                      alpha=0.5, label=f'平均ATR={weekly_atr_data.mean():.3f}')
        ax_atr.set_ylabel('周线ATR')
        ax_atr.legend(loc='upper left')
        ax_atr.grid(True, alpha=0.3)
        ax_atr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"周线波峰波谷图已保存至: {save_path}")

    def plot_peaks_troughs(self,
                           start_date: str,
                           end_date: str,
                           save_path: str,
                           stock_name: str = '',
                           use_dynamic_atr: bool = True) -> None:
        """
        绘制波峰波谷图表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            save_path: 图片保存路径
            stock_name: 股票名称
            use_dynamic_atr: 是否使用动态ATR阈值
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        if self._close_series is None:
            logger.warning("无数据，无法绘制图表")
            return

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]
        high_data = self._high_series[
            (self._high_series.index >= start_ts) &
            (self._high_series.index <= end_ts)
        ]
        low_data = self._low_series[
            (self._low_series.index >= start_ts) &
            (self._low_series.index <= end_ts)
        ]

        if len(close_data) == 0:
            return

        # 检测波峰波谷
        if use_dynamic_atr:
            peaks = self.detect_peaks_dynamic_atr(start_date, end_date)
            troughs = self.detect_troughs_dynamic_atr(start_date, end_date)
        else:
            peaks = self.detect_peaks(start_date, end_date)
            troughs = self.detect_troughs(start_date, end_date)

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'{stock_name} 波峰波谷检测 ({start_date} ~ {end_date})',
                    fontsize=14, fontweight='bold')

        ax_price = axes[0]
        ax_atr = axes[1]

        # 绘制价格曲线
        dates = close_data.index
        ax_price.plot(dates, close_data.values, 'b-', linewidth=1, label='收盘价', alpha=0.7)

        # 绘制高低点区间
        ax_price.fill_between(dates, low_data.values, high_data.values,
                              color='blue', alpha=0.2, label='日内波动区间')

        # 标记波峰
        if peaks:
            peak_dates = [p.date for p in peaks]
            peak_prices = [p.price for p in peaks]
            ax_price.scatter(peak_dates, peak_prices, color='red', marker='^', s=100,
                            label=f'波峰({len(peaks)}个)', zorder=5)
            for p in peaks:
                ax_price.annotate(f'{p.price:.2f}\nATR:{p.atr_value:.3f}',
                                 (p.date, p.price),
                                 textcoords="offset points", xytext=(5, 10),
                                 fontsize=7, color='red')

        # 标记波谷
        if troughs:
            trough_dates = [t.date for t in troughs]
            trough_prices = [t.price for t in troughs]
            ax_price.scatter(trough_dates, trough_prices, color='green', marker='v', s=100,
                            label=f'波谷({len(troughs)}个)', zorder=5)
            for t in troughs:
                ax_price.annotate(f'{t.price:.2f}\nATR:{t.atr_value:.3f}',
                                 (t.date, t.price),
                                 textcoords="offset points", xytext=(5, -15),
                                 fontsize=7, color='green')

        ax_price.set_ylabel('价格')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # 绘制ATR曲线
        atr_data = self._atr_series[
            (self._atr_series.index >= start_ts) &
            (self._atr_series.index <= end_ts)
        ]
        ax_atr.plot(dates, atr_data.values, 'purple', linewidth=1, label='ATR(14)')
        ax_atr.axhline(y=atr_data.mean(), color='red', linestyle='--',
                      alpha=0.5, label=f'平均ATR={atr_data.mean():.3f}')
        ax_atr.set_ylabel('ATR')
        ax_atr.legend(loc='upper left')
        ax_atr.grid(True, alpha=0.3)
        ax_atr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"波峰波谷图已保存至: {save_path}")


# ==================== 测试模块 ====================

if __name__ == "__main__":
    import os
    import json
    from datetime import datetime

    # 配置日志
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    LOG_DIR = os.path.join(project_root, 'logs')

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_FILE = os.path.join(LOG_DIR, f'peak_detector_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 60)
    logger.info("波峰波谷检测模块测试")
    logger.info("=" * 60)

    # ==================== 测试参数配置 ====================
    TEST_TICKER = '512480'
    TEST_NAME = '半导体ETF'
    START_DATE = '2024-01-01'
    END_DATE = '2026-04-10'

    # 日线检测参数
    DAILY_ATR_PERIOD = 14
    DAILY_PROMINENCE_FACTOR = 1.0
    DAILY_MIN_DISTANCE = 20  # 日线最小间隔20天

    # 周线检测参数
    WEEKLY_ATR_PERIOD = 14
    WEEKLY_PROMINENCE_FACTOR = 1.0
    WEEKLY_MIN_DISTANCE = 3   # 周线最小间隔3周

    logger.info(f"测试股票: {TEST_NAME} ({TEST_TICKER})")
    logger.info(f"  检测时间段: {START_DATE} ~ {END_DATE}")
    logger.info(f"  【日线参数】ATR周期={DAILY_ATR_PERIOD}, 突出因子={DAILY_PROMINENCE_FACTOR}, 最小距离={DAILY_MIN_DISTANCE}天")
    logger.info(f"  【周线参数】ATR周期={WEEKLY_ATR_PERIOD}, 突出因子={WEEKLY_PROMINENCE_FACTOR}, 最小距离={WEEKLY_MIN_DISTANCE}周")

    try:
        from gm.api import history, set_token, ADJUST_PREV

        # 读取配置
        config_path = os.path.join(project_root, 'config', 'settings.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            set_token(config.get('gm_token'))

        # 获取股票交易所前缀
        if TEST_TICKER.startswith('6') or TEST_TICKER.startswith('51') or TEST_TICKER.startswith('58'):
            full_symbol = f'SHSE.{TEST_TICKER}'
        else:
            full_symbol = f'SZSE.{TEST_TICKER}'

        # 获取历史数据（需要high, low, close）
        warmup_start = '2023-01-01'
        logger.info("获取历史数据...")

        daily_data = history(
            symbol=full_symbol,
            frequency='1d',
            start_time=warmup_start + ' 09:00:00',
            end_time=END_DATE + ' 15:30:00',
            fields='eob,open,high,low,close',
            df=True,
            adjust=ADJUST_PREV
        )

        if daily_data is None or daily_data.empty:
            logger.error(f"获取数据失败: {full_symbol}")
            raise Exception("数据获取失败")

        logger.info(f"获取到 {len(daily_data)} 条数据")

        daily_data['eob'] = pd.to_datetime(daily_data['eob'])

        # 创建价格序列
        high_series = pd.Series(
            daily_data['high'].values,
            index=daily_data['eob'],
            name='High'
        )
        low_series = pd.Series(
            daily_data['low'].values,
            index=daily_data['eob'],
            name='Low'
        )
        close_series = pd.Series(
            daily_data['close'].values,
            index=daily_data['eob'],
            name='Close'
        )

        # 创建日线检测器
        daily_detector = PeakDetector(
            atr_period=DAILY_ATR_PERIOD,
            prominence_factor=DAILY_PROMINENCE_FACTOR,
            min_distance=DAILY_MIN_DISTANCE
        )
        daily_detector.prepare_data(high_series, low_series, close_series)

        # ==================== 日线波峰波谷检测 ====================
        logger.info("")
        logger.info("=" * 60)
        logger.info("【日线】波峰检测结果（动态ATR阈值）")
        logger.info("=" * 60)

        peaks = daily_detector.detect_peaks_dynamic_atr(START_DATE, END_DATE)
        logger.info(f"检测到 {len(peaks)} 个日线波峰")

        for p in peaks:
            logger.info(f"[{p.date.strftime('%Y-%m-%d')}] 波峰价格={p.price:.3f}, "
                       f"突出程度={p.prominence:.3f}, ATR={p.atr_value:.3f}, "
                       f"满足条件: {p.prominence:.3f} >= {p.atr_value * DAILY_PROMINENCE_FACTOR:.3f}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("【日线】波谷检测结果（动态ATR阈值）")
        logger.info("=" * 60)

        troughs = daily_detector.detect_troughs_dynamic_atr(START_DATE, END_DATE)
        logger.info(f"检测到 {len(troughs)} 个日线波谷")

        for t in troughs:
            logger.info(f"[{t.date.strftime('%Y-%m-%d')}] 波谷价格={t.price:.3f}, "
                       f"下降程度={t.prominence:.3f}, ATR={t.atr_value:.3f}, "
                       f"满足条件: {t.prominence:.3f} >= {t.atr_value * DAILY_PROMINENCE_FACTOR:.3f}")

        # 绘制日线图表
        logger.info("")
        logger.info("绘制日线波峰波谷图表...")
        chart_path_daily = os.path.join(LOG_DIR, f'peaks_troughs_daily_{TEST_TICKER}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        daily_detector.plot_peaks_troughs(
            start_date=START_DATE,
            end_date=END_DATE,
            save_path=chart_path_daily,
            stock_name=TEST_NAME,
            use_dynamic_atr=True
        )

        # ==================== 周线波峰波谷检测（独立检测器） ====================
        # 从日线数据聚合周线数据
        df = pd.DataFrame({
            'high': high_series,
            'low': low_series,
            'close': close_series
        })
        weekly_df = df.groupby(df.index.to_period('W')).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })

        # 获取每周最后一天日期
        weekly_dates = []
        for period in weekly_df.index:
            week_start = period.start_time
            week_end = period.end_time
            if df.index.tz is not None:
                week_start = week_start.tz_localize(df.index.tz)
                week_end = week_end.tz_localize(df.index.tz)
            week_data = df[(df.index >= week_start) & (df.index <= week_end)]
            if len(week_data) > 0:
                weekly_dates.append(week_data.index[-1])

        weekly_high = pd.Series(weekly_df['high'].values, index=weekly_dates, name='Weekly_High')
        weekly_low = pd.Series(weekly_df['low'].values, index=weekly_dates, name='Weekly_Low')
        weekly_close = pd.Series(weekly_df['close'].values, index=weekly_dates, name='Weekly_Close')

        # 创建周线检测器（独立参数）
        weekly_detector = PeakDetector(
            atr_period=WEEKLY_ATR_PERIOD,
            prominence_factor=WEEKLY_PROMINENCE_FACTOR,
            min_distance=WEEKLY_MIN_DISTANCE
        )
        weekly_detector.prepare_data(weekly_high, weekly_low, weekly_close)

        logger.info("")
        logger.info("=" * 60)
        logger.info("【周线】波峰检测结果（独立检测器，动态ATR阈值）")
        logger.info("=" * 60)

        weekly_peaks = weekly_detector.detect_peaks_dynamic_atr(
            weekly_close.index[0].strftime('%Y-%m-%d'),
            weekly_close.index[-1].strftime('%Y-%m-%d')
        )
        logger.info(f"检测到 {len(weekly_peaks)} 个周线波峰")

        for p in weekly_peaks:
            logger.info(f"[{p.date.strftime('%Y-%m-%d')}] 周线波峰价格={p.price:.3f}, "
                       f"突出程度={p.prominence:.3f}, ATR={p.atr_value:.3f}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("【周线】波谷检测结果（独立检测器，动态ATR阈值）")
        logger.info("=" * 60)

        weekly_troughs = weekly_detector.detect_troughs_dynamic_atr(
            weekly_close.index[0].strftime('%Y-%m-%d'),
            weekly_close.index[-1].strftime('%Y-%m-%d')
        )
        logger.info(f"检测到 {len(weekly_troughs)} 个周线波谷")

        for t in weekly_troughs:
            logger.info(f"[{t.date.strftime('%Y-%m-%d')}] 周线波谷价格={t.price:.3f}, "
                       f"下降程度={t.prominence:.3f}, ATR={t.atr_value:.3f}")

        # 绘制周线图表
        logger.info("")
        logger.info("绘制周线波峰波谷图表...")
        chart_path_weekly = os.path.join(LOG_DIR, f'peaks_troughs_weekly_{TEST_TICKER}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        weekly_detector.plot_peaks_troughs(
            start_date=weekly_close.index[0].strftime('%Y-%m-%d'),
            end_date=weekly_close.index[-1].strftime('%Y-%m-%d'),
            save_path=chart_path_weekly,
            stock_name=TEST_NAME + '(周线)',
            use_dynamic_atr=True
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("测试完成!")
        logger.info(f"日志已保存至: {LOG_FILE}")
        logger.info(f"日线图表已保存至: {chart_path_daily}")
        logger.info(f"周线图表已保存至: {chart_path_weekly}")
        logger.info("=" * 60)

    except ImportError:
        logger.error("未安装掘金量化SDK，请先安装: pip install gm")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise