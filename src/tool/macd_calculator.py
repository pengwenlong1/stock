# -*- coding: utf-8 -*-
"""
MACD指标模块 (macd_calculator.py)

【功能说明】
计算MACD（Moving Average Convergence Divergence）指标，包括：
- DIF（快线）：EMA(12) - EMA(26)
- DEA（慢线/信号线）：DIF的EMA(9)
- MACD柱：DIF - DEA

【核心逻辑】
1. MACD指标原理：
   - DIF > 0：多头市场（快线在慢线上方）
   - DIF < 0：空头市场（快线在慢线下方）
   - MACD柱 > 0：多头动能增强
   - MACD柱 < 0：空头动能增强

2. 金叉信号：DIF上穿DEA（买入信号）
3. 死叉信号：DIF下穿DEA（卖出信号）
4. 背离信号：
   - 顶背离：价格创新高，但MACD不创新高（卖出信号）
   - 底背离：价格创新低，但MACD不创新低（买入信号）

【数学描述】
DIF = EMA(close, 12) - EMA(close, 26)
DEA = EMA(DIF, 9)
MACD = (DIF - DEA) * 2  # 国内常用乘以2

EMA计算公式：
EMA(t) = α * close(t) + (1-α) * EMA(t-1)
α = 2 / (period + 1)

作者：量化交易团队
创建日期：2024
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

try:
    import talib
except ImportError:
    talib = None
    logging.warning("talib未安装，将使用自定义MACD计算")

logger = logging.getLogger(__name__)


@dataclass
class MACDValue:
    """MACD数值信息"""
    date: pd.Timestamp      # 日期
    dif: float              # DIF值（快线）
    dea: float              # DEA值（慢线/信号线）
    macd: float             # MACD柱值
    close: float            # 收盘价


@dataclass
class MACDSignal:
    """MACD信号信息"""
    date: pd.Timestamp          # 信号日期
    signal_type: str            # '金叉' 或 '死叉'
    dif: float                  # DIF值
    dea: float                  # DEA值
    macd: float                 # MACD柱值
    close: float                # 收盘价
    position: str = ''          # '多头' 或 '空头'


@dataclass
class MACDDivergence:
    """MACD背离信息"""
    date: pd.Timestamp          # 背离确认日期
    divergence_type: str        # '顶背离' 或 '底背离'
    price_peak_date: pd.Timestamp  # 价格极值点日期
    price_peak_value: float     # 价格极值值
    macd_peak_date: pd.Timestamp  # MACD极值点日期
    macd_peak_value: float      # MACD极值值
    confirmation: str = ''      # 确认说明


class MACDCalculator:
    """
    MACD指标计算器

    【功能说明】
    计算MACD(12,26,9)指标，检测金叉死叉和背离信号。

    【参数配置】
    - fast_period: 快线EMA周期，默认12
    - slow_period: 慢线EMA周期，默认26
    - signal_period: DEA信号线周期，默认9

    【使用方法】
    calculator = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
    calculator.prepare_data(close_series)
    macd_values = calculator.get_macd_values(start_date, end_date)
    signals = calculator.detect_cross_signals(start_date, end_date)
    divergences = calculator.detect_divergences(start_date, end_date)
    """

    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9) -> None:
        """
        初始化MACD计算器

        Args:
            fast_period: 快线EMA周期（默认12）
            slow_period: 慢线EMA周期（默认26）
            signal_period: DEA信号线周期（默认9）
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # 数据存储
        self._close_series: Optional[pd.Series] = None
        self._dif_series: Optional[pd.Series] = None
        self._dea_series: Optional[pd.Series] = None
        self._macd_series: Optional[pd.Series] = None

    def prepare_data(self, close_series: pd.Series) -> None:
        """
        准备计算数据

        Args:
            close_series: 收盘价序列
        """
        if not isinstance(close_series, pd.Series):
            raise TypeError(f"close_series必须为pd.Series")

        self._close_series = close_series

        # 计算MACD指标
        self._dif_series, self._dea_series, self._macd_series = self._calculate_macd(close_series)

        logger.info(f"MACD计算器数据准备完成: {len(close_series)} 天数据")

    def _calculate_macd(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算MACD指标

        Args:
            close: 收盘价序列

        Returns:
            Tuple: (DIF序列, DEA序列, MACD柱序列)
        """
        if talib is not None:
            # 使用talib计算MACD
            dif_values, dea_values, macd_values = talib.MACD(
                close.values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )

            dif_series = pd.Series(dif_values, index=close.index, name='DIF')
            dea_series = pd.Series(dea_values, index=close.index, name='DEA')
            # 国内习惯MACD柱乘以2
            macd_series = pd.Series(macd_values * 2, index=close.index, name='MACD')
        else:
            # 自定义MACD计算
            dif_series, dea_series, macd_series = self._calculate_macd_custom(close)

        return dif_series, dea_series, macd_series

    def _calculate_macd_custom(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        自定义MACD计算（当talib不可用时）

        【算法说明】
        1. EMA计算：EMA(t) = α * price(t) + (1-α) * EMA(t-1)
        2. DIF = EMA(12) - EMA(26)
        3. DEA = EMA(DIF, 9)
        4. MACD = (DIF - DEA) * 2

        Args:
            close: 收盘价序列

        Returns:
            Tuple: (DIF, DEA, MACD柱)
        """
        # 计算EMA
        def calculate_ema(series: pd.Series, period: int) -> pd.Series:
            alpha = 2.0 / (period + 1.0)
            ema = series.copy()
            # 初始值使用第一个有效值
            first_valid = series.first_valid_index()
            if first_valid is None:
                return pd.Series([0] * len(series), index=series.index)

            ema_values = [series.iloc[0]]
            for i in range(1, len(series)):
                if pd.isna(series.iloc[i]):
                    ema_values.append(ema_values[-1])
                else:
                    ema_values.append(alpha * series.iloc[i] + (1 - alpha) * ema_values[-1])

            return pd.Series(ema_values, index=series.index)

        # 计算快线EMA和慢线EMA
        ema_fast = calculate_ema(close, self.fast_period)
        ema_slow = calculate_ema(close, self.slow_period)

        # 计算DIF
        dif = ema_fast - ema_slow

        # 计算DEA（DIF的EMA）
        dea = calculate_ema(dif, self.signal_period)

        # 计算MACD柱（国内习惯乘以2）
        macd = (dif - dea) * 2

        dif.name = 'DIF'
        dea.name = 'DEA'
        macd.name = 'MACD'

        return dif, dea, macd

    def get_macd_values(self,
                         start_date: str,
                         end_date: str) -> List[MACDValue]:
        """
        获取指定时间段的MACD数值

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[MACDValue]: MACD数值列表
        """
        if self._dif_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        dif_data = self._dif_series[
            (self._dif_series.index >= start_ts) &
            (self._dif_series.index <= end_ts)
        ]
        dea_data = self._dea_series[
            (self._dea_series.index >= start_ts) &
            (self._dea_series.index <= end_ts)
        ]
        macd_data = self._macd_series[
            (self._macd_series.index >= start_ts) &
            (self._macd_series.index <= end_ts)
        ]
        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]

        if len(dif_data) == 0:
            return []

        macd_values = []
        for date in dif_data.index:
            macd_values.append(MACDValue(
                date=date,
                dif=dif_data.loc[date],
                dea=dea_data.loc[date],
                macd=macd_data.loc[date],
                close=close_data.loc[date]
            ))

        return macd_values

    def detect_cross_signals(self,
                              start_date: str,
                              end_date: str) -> List[MACDSignal]:
        """
        检测金叉和死叉信号

        【信号定义】
        - 金叉：DIF上穿DEA（从下方穿过上方），买入信号
        - 死叉：DIF下穿DEA（从上方穿过下方），卖出信号

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[MACDSignal]: 信号列表
        """
        if self._dif_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        dif_data = self._dif_series[
            (self._dif_series.index >= start_ts) &
            (self._dif_series.index <= end_ts)
        ]
        dea_data = self._dea_series[
            (self._dea_series.index >= start_ts) &
            (self._dea_series.index <= end_ts)
        ]
        macd_data = self._macd_series[
            (self._macd_series.index >= start_ts) &
            (self._macd_series.index <= end_ts)
        ]
        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]

        if len(dif_data) < 2:
            return []

        signals = []

        # 遍历检测交叉点
        dates = dif_data.index
        for i in range(1, len(dates)):
            curr_date = dates[i]
            prev_date = dates[i-1]

            curr_dif = dif_data.loc[curr_date]
            prev_dif = dif_data.loc[prev_date]
            curr_dea = dea_data.loc[curr_date]
            prev_dea = dea_data.loc[prev_date]
            curr_macd = macd_data.loc[curr_date]
            curr_close = close_data.loc[curr_date]

            # 检测金叉：DIF从下方穿过DEA上方
            if prev_dif <= prev_dea and curr_dif > curr_dea:
                signal = MACDSignal(
                    date=curr_date,
                    signal_type='金叉',
                    dif=curr_dif,
                    dea=curr_dea,
                    macd=curr_macd,
                    close=curr_close,
                    position='多头'
                )
                signals.append(signal)
                logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 金叉信号: DIF={curr_dif:.4f}, DEA={curr_dea:.4f}")

            # 检测死叉：DIF从上方穿过DEA下方
            elif prev_dif >= prev_dea and curr_dif < curr_dea:
                signal = MACDSignal(
                    date=curr_date,
                    signal_type='死叉',
                    dif=curr_dif,
                    dea=curr_dea,
                    macd=curr_macd,
                    close=curr_close,
                    position='空头'
                )
                signals.append(signal)
                logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 死叉信号: DIF={curr_dif:.4f}, DEA={curr_dea:.4f}")

        logger.info(f"检测到 {len(signals)} 个金叉/死叉信号")
        return signals

    def detect_divergences(self,
                            start_date: str,
                            end_date: str,
                            lookback_period: int = 20) -> List[MACDDivergence]:
        """
        检测MACD背离信号

        【背离定义】
        - 顶背离：价格创新高，但MACD（DIF或MACD柱）不创新高
          → 多头动能减弱，可能是卖出信号
        - 底背离：价格创新低，但MACD（DIF或MACD柱）不创新低
          → 空头动能减弱，可能是买入信号

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            lookback_period: 回溯周期（用于判断极值点）

        Returns:
            List[MACDDivergence]: 背离信号列表
        """
        if self._dif_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]
        dif_data = self._dif_series[
            (self._dif_series.index >= start_ts) &
            (self._dif_series.index <= end_ts)
        ]
        macd_data = self._macd_series[
            (self._macd_series.index >= start_ts) &
            (self._macd_series.index <= end_ts)
        ]

        if len(close_data) < lookback_period * 2:
            return []

        divergences = []

        # 使用滚动窗口检测背离
        dates = close_data.index
        for i in range(lookback_period, len(dates)):
            curr_date = dates[i]

            # 回溯窗口
            window_start = i - lookback_period
            window_dates = dates[window_start:i+1]

            # 检测顶背离
            # 找到窗口内价格最高点
            window_prices = close_data.loc[window_dates]
            price_max_idx = window_prices.idxmax()
            price_max_value = window_prices.max()

            # 当前价格是否接近最高点（是局部高点）
            curr_price = close_data.loc[curr_date]
            if curr_price >= price_max_value * 0.98:  # 接近最高点
                # 检查MACD是否创新高
                window_macd = macd_data.loc[window_dates]
                prev_macd_max = window_macd.iloc[:-1].max()  # 之前的最大值
                curr_macd = macd_data.loc[curr_date]

                # 顶背离：价格新高，MACD未新高
                if curr_macd < prev_macd_max and curr_macd > 0:  # MACD柱为正（多头区域）
                    divergence = MACDDivergence(
                        date=curr_date,
                        divergence_type='顶背离',
                        price_peak_date=price_max_idx,
                        price_peak_value=price_max_value,
                        macd_peak_date=window_macd.idxmax(),
                        macd_peak_value=prev_macd_max,
                        confirmation=f'价格高点{price_max_value:.3f}，但MACD高点{prev_macd_max:.4f}未被突破'
                    )
                    divergences.append(divergence)
                    logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 顶背离检测")

            # 检测底背离
            # 找到窗口内价格最低点
            price_min_idx = window_prices.idxmin()
            price_min_value = window_prices.min()

            # 当前价格是否接近最低点（是局部低点）
            if curr_price <= price_min_value * 1.02:  # 接近最低点
                # 检查MACD是否创新低
                window_macd = macd_data.loc[window_dates]
                prev_macd_min = window_macd.iloc[:-1].min()  # 之前的最小值
                curr_macd = macd_data.loc[curr_date]

                # 底背离：价格新低，MACD未新低
                if curr_macd > prev_macd_min and curr_macd < 0:  # MACD柱为负（空头区域）
                    divergence = MACDDivergence(
                        date=curr_date,
                        divergence_type='底背离',
                        price_peak_date=price_min_idx,
                        price_peak_value=price_min_value,
                        macd_peak_date=window_macd.idxmin(),
                        macd_peak_value=prev_macd_min,
                        confirmation=f'价格低点{price_min_value:.3f}，但MACD低点{prev_macd_min:.4f}未被突破'
                    )
                    divergences.append(divergence)
                    logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 底背离检测")

        logger.info(f"检测到 {len(divergences)} 个背离信号")
        return divergences

    def get_current_state(self,
                           start_date: str,
                           end_date: str) -> Dict:
        """
        获取当前MACD状态

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 包含当前状态信息
        """
        if self._dif_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        dif_data = self._dif_series[
            (self._dif_series.index >= start_ts) &
            (self._dif_series.index <= end_ts)
        ]
        dea_data = self._dea_series[
            (self._dea_series.index >= start_ts) &
            (self._dea_series.index <= end_ts)
        ]
        macd_data = self._macd_series[
            (self._macd_series.index >= start_ts) &
            (self._macd_series.index <= end_ts)
        ]
        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]

        if len(dif_data) == 0:
            return {'error': '无数据'}

        last_date = dif_data.index[-1]
        last_dif = dif_data.loc[last_date]
        last_dea = dea_data.loc[last_date]
        last_macd = macd_data.loc[last_date]
        last_close = close_data.loc[last_date]

        # 判断多空状态
        position = '多头' if last_dif > last_dea else '空头'
        trend = '上涨' if last_macd > 0 else '下跌'

        return {
            'date': last_date,
            'dif': last_dif,
            'dea': last_dea,
            'macd': last_macd,
            'close': last_close,
            'position': position,
            'trend': trend,
            'cross_status': '金叉区域' if last_dif > last_dea else '死叉区域'
        }

    def plot_macd(self,
                   start_date: str,
                   end_date: str,
                   save_path: str,
                   stock_name: str = '',
                   show_signals: bool = True) -> None:
        """
        绘制MACD图表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            save_path: 图片保存路径
            stock_name: 股票名称
            show_signals: 是否显示金叉死叉信号
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
        dif_data = self._dif_series[
            (self._dif_series.index >= start_ts) &
            (self._dif_series.index <= end_ts)
        ]
        dea_data = self._dea_series[
            (self._dea_series.index >= start_ts) &
            (self._dea_series.index <= end_ts)
        ]
        macd_data = self._macd_series[
            (self._macd_series.index >= start_ts) &
            (self._macd_series.index <= end_ts)
        ]

        if len(close_data) == 0:
            return

        # 检测金叉死叉信号
        signals = self.detect_cross_signals(start_date, end_date) if show_signals else []

        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        fig.suptitle(f'{stock_name} MACD指标 ({self.fast_period},{self.slow_period},{self.signal_period}) '
                    f'({start_date} ~ {end_date})',
                    fontsize=14, fontweight='bold')

        ax_price = axes[0]
        ax_macd_line = axes[1]
        ax_macd_bar = axes[2]

        dates = close_data.index

        # ===== 绘制价格曲线 =====
        ax_price.plot(dates, close_data.values, 'b-', linewidth=1.5, label='收盘价')

        # 标记金叉死叉点
        if signals:
            for sig in signals:
                if sig.signal_type == '金叉':
                    ax_price.scatter(sig.date, sig.close, color='red', marker='^', s=100, zorder=5)
                    ax_price.annotate(f'金叉\n{sig.close:.2f}', (sig.date, sig.close),
                                     textcoords="offset points", xytext=(5, 10),
                                     fontsize=8, color='red')
                elif sig.signal_type == '死叉':
                    ax_price.scatter(sig.date, sig.close, color='green', marker='v', s=100, zorder=5)
                    ax_price.annotate(f'死叉\n{sig.close:.2f}', (sig.date, sig.close),
                                     textcoords="offset points", xytext=(5, -15),
                                     fontsize=8, color='green')

        ax_price.set_ylabel('价格')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # ===== 绘制DIF和DEA曲线 =====
        ax_macd_line.plot(dates, dif_data.values, 'r-', linewidth=1, label='DIF(快线)')
        ax_macd_line.plot(dates, dea_data.values, 'g-', linewidth=1, label='DEA(慢线)')
        ax_macd_line.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 标记交叉点
        if signals:
            for sig in signals:
                color = 'red' if sig.signal_type == '金叉' else 'green'
                ax_macd_line.scatter(sig.date, sig.dif, color=color, s=50, zorder=5)

        ax_macd_line.set_ylabel('DIF/DEA')
        ax_macd_line.legend(loc='upper left')
        ax_macd_line.grid(True, alpha=0.3)
        ax_macd_line.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # ===== 绘制MACD柱 =====
        # 正值红色，负值绿色
        macd_values = macd_data.values
        colors = ['red' if v > 0 else 'green' for v in macd_values]
        ax_macd_bar.bar(dates, macd_values, color=colors, alpha=0.6, width=0.8)
        ax_macd_bar.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

        ax_macd_bar.set_ylabel('MACD柱')
        ax_macd_bar.grid(True, alpha=0.3)
        ax_macd_bar.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"MACD图表已保存至: {save_path}")


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

    LOG_FILE = os.path.join(LOG_DIR, f'macd_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
    logger.info("MACD指标模块测试")
    logger.info("=" * 60)

    # ==================== 测试参数配置 ====================
    TEST_TICKER = '512480'
    TEST_NAME = '半导体ETF'
    START_DATE = '2024-01-01'
    END_DATE = '2026-04-10'

    # MACD参数
    FAST_PERIOD = 12
    SLOW_PERIOD = 26
    SIGNAL_PERIOD = 9

    logger.info(f"测试股票: {TEST_NAME} ({TEST_TICKER})")
    logger.info(f"  检测时间段: {START_DATE} ~ {END_DATE}")
    logger.info(f"  MACD参数: ({FAST_PERIOD}, {SLOW_PERIOD}, {SIGNAL_PERIOD})")

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

        # 获取历史数据
        warmup_start = '2023-01-01'
        logger.info("获取历史数据...")

        daily_data = history(
            symbol=full_symbol,
            frequency='1d',
            start_time=warmup_start + ' 09:00:00',
            end_time=END_DATE + ' 15:30:00',
            fields='eob,close',
            df=True,
            adjust=ADJUST_PREV
        )

        if daily_data is None or daily_data.empty:
            logger.error(f"获取数据失败: {full_symbol}")
            raise Exception("数据获取失败")

        logger.info(f"获取到 {len(daily_data)} 条数据")

        daily_data['eob'] = pd.to_datetime(daily_data['eob'])

        # 创建收盘价序列
        close_series = pd.Series(
            daily_data['close'].values,
            index=daily_data['eob'],
            name='Close'
        )

        # 创建MACD计算器
        calculator = MACDCalculator(
            fast_period=FAST_PERIOD,
            slow_period=SLOW_PERIOD,
            signal_period=SIGNAL_PERIOD
        )
        calculator.prepare_data(close_series)

        # 输出MACD数值（最近10天）
        logger.info("")
        logger.info("=" * 60)
        logger.info("MACD数值（最近10天）")
        logger.info("=" * 60)

        # 获取最近的数据
        recent_start = '2026-04-01'
        macd_values = calculator.get_macd_values(recent_start, END_DATE)

        for mv in macd_values[-10:]:
            logger.info(f"[{mv.date.strftime('%Y-%m-%d')}] "
                       f"DIF={mv.dif:.4f}, DEA={mv.dea:.4f}, MACD={mv.macd:.4f}, Close={mv.close:.3f}")

        # 检测金叉死叉信号
        logger.info("")
        logger.info("=" * 60)
        logger.info("金叉死叉信号检测")
        logger.info("=" * 60)

        signals = calculator.detect_cross_signals(START_DATE, END_DATE)
        logger.info(f"检测到 {len(signals)} 个金叉/死叉信号")

        # 统计金叉和死叉数量
        golden_cross_count = sum(1 for s in signals if s.signal_type == '金叉')
        death_cross_count = sum(1 for s in signals if s.signal_type == '死叉')
        logger.info(f"  金叉信号: {golden_cross_count} 个")
        logger.info(f"  死叉信号: {death_cross_count} 个")

        # 输出最近5个信号
        logger.info("")
        logger.info("最近5个信号详情:")
        for sig in signals[-5:]:
            logger.info(f"[{sig.date.strftime('%Y-%m-%d')}] {sig.signal_type}: "
                       f"DIF={sig.dif:.4f}, DEA={sig.dea:.4f}, MACD={sig.macd:.4f}, Close={sig.close:.3f}")

        # 检测背离信号
        logger.info("")
        logger.info("=" * 60)
        logger.info("背离信号检测")
        logger.info("=" * 60)

        divergences = calculator.detect_divergences(START_DATE, END_DATE)
        logger.info(f"检测到 {len(divergences)} 个背离信号")

        # 统计顶背离和底背离数量
        top_div_count = sum(1 for d in divergences if d.divergence_type == '顶背离')
        bottom_div_count = sum(1 for d in divergences if d.divergence_type == '底背离')
        logger.info(f"  顶背离: {top_div_count} 个")
        logger.info(f"  底背离: {bottom_div_count} 个")

        for div in divergences[-5:]:
            logger.info(f"[{div.date.strftime('%Y-%m-%d')}] {div.divergence_type}: "
                       f"价格极值={div.price_peak_value:.3f}, MACD极值={div.macd_peak_value:.4f}")

        # 获取当前状态
        logger.info("")
        logger.info("=" * 60)
        logger.info("当前MACD状态")
        logger.info("=" * 60)

        state = calculator.get_current_state(START_DATE, END_DATE)
        logger.info(f"  日期: {state['date'].strftime('%Y-%m-%d')}")
        logger.info(f"  DIF: {state['dif']:.4f}")
        logger.info(f"  DEA: {state['dea']:.4f}")
        logger.info(f"  MACD柱: {state['macd']:.4f}")
        logger.info(f"  收盘价: {state['close']:.3f}")
        logger.info(f"  多空状态: {state['position']}")
        logger.info(f"  趋势: {state['trend']}")
        logger.info(f"  交叉状态: {state['cross_status']}")

        # 绘制图表
        logger.info("")
        logger.info("绘制MACD图表...")
        chart_path = os.path.join(LOG_DIR, f'macd_{TEST_TICKER}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        calculator.plot_macd(
            start_date=START_DATE,
            end_date=END_DATE,
            save_path=chart_path,
            stock_name=TEST_NAME,
            show_signals=True
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("测试完成!")
        logger.info(f"日志已保存至: {LOG_FILE}")
        logger.info(f"图表已保存至: {chart_path}")
        logger.info("=" * 60)

    except ImportError:
        logger.error("未安装掘金量化SDK，请先安装: pip install gm")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise