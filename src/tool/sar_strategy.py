# -*- coding: utf-8 -*-
"""
SAR策略模块 (sar_strategy.py)

【功能说明】
基于抛物线转向指标（Parabolic SAR）的红转绿和绿转红策略。
检测趋势转折点，生成买入和卖出信号。

【核心逻辑】
1. SAR指标原理：
   - SAR点位于价格上方 → 红色 → 空头市场（做空信号）
   - SAR点位于价格下方 → 绿色 → 多头市场（做多信号）

2. 红转绿（买入信号）：
   - SAR从价格上方转到下方，趋势从空头转为多头

3. 绿转红（卖出信号）：
   - SAR从价格下方转到上方，趋势从多头转为空头

【数学描述】
SAR计算公式：
- 加速因子(AF)初始为0.02，每次创新高/新低增加0.02，最高0.20
- SAR(t) = SAR(t-1) + AF * (EP - SAR(t-1))
- EP = 极值点（多头时为最高价，空头时为最低价）

信号判断：
- 红转绿：SAR(t) < Close(t) 且 SAR(t-1) >= Close(t-1)
- 绿转红：SAR(t) > Close(t) 且 SAR(t-1) <= Close(t-1)

【实盘监控功能】
- 实时数据获取：使用掘金current()获取当天实时行情
- 实盘告警：通过钉钉机器人发送SAR转折信号

作者：量化交易团队
创建日期：2024
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:
    talib = None
    logging.warning("talib未安装，将使用自定义SAR计算")

try:
    from gm.api import history, set_token, ADJUST_PREV, current, get_trading_dates, get_instruments
except ImportError:
    history = None
    set_token = None
    ADJUST_PREV = None
    current = None
    get_trading_dates = None
    get_instruments = None
    logging.warning("掘金SDK未安装，实时数据功能不可用")

logger = logging.getLogger(__name__)


@dataclass
class SARSignal:
    """SAR信号信息"""
    date: pd.Timestamp          # 信号日期
    signal_type: str            # '红转绿' 或 '绿转红'
    sar_value: float            # SAR值
    close_price: float          # 收盘价
    position: str               # 'above' 或 'below'（SAR相对价格位置）
    af_value: float = 0.0       # 加速因子值


@dataclass
class SARState:
    """SAR状态信息"""
    current_position: str = 'unknown'  # '多头' 或 '空头'
    last_signal_date: Optional[pd.Timestamp] = None
    last_signal_type: str = ''


@dataclass
class RealtimeQuote:
    """实时行情数据"""
    symbol: str
    stock_name: str
    current_time: datetime
    open: float
    high: float
    low: float
    price: float              # 当前价格
    volume: float
    sar_today: float          # 今日SAR值
    sar_yesterday: float      # 昨日SAR值
    prev_close: float         # 昨日收盘价
    trend: str                # '多头' 或 '空头'


@dataclass
class RealtimeAlert:
    """实盘告警信息"""
    symbol: str
    stock_name: str
    alert_time: datetime
    signal_type: str          # '红转绿' 或 '绿转红'
    current_price: float
    sar_value: float
    prev_sar: float
    prev_close: float
    realtime_high: float
    realtime_low: float
    trend_before: str         # 转折前趋势
    trend_after: str          # 转折后趋势


class SARStrategy:
    """
    SAR策略类

    【功能说明】
    基于抛物线转向指标（SAR）的趋势转折检测。

    【参数配置】
    SAR(10,2,20) 参数格式：
    - acceleration: 加速因子步长，默认0.02 (对应参数中的2，即2%)
    - maximum: 加速因子最大值，默认0.20 (对应参数中的20，即20%)

    【使用方法】
    strategy = SARStrategy(acceleration=0.02, maximum=0.20)
    strategy.prepare_data(high_series, low_series, close_series)
    signals = strategy.detect_signals(start_date, end_date)
    """

    def __init__(self,
                 acceleration: float = 0.02,
                 maximum: float = 0.20) -> None:
        """
        初始化SAR策略

        Args:
            acceleration: 加速因子步长（SAR参数中的"2"，即2%步长）
            maximum: 加速因子最大值（SAR参数中的"20"，即20%极限）

        【SAR(10,2,20)参数说明】
        - 10: 观察周期（talib.SAR不直接使用此参数）
        - 2: 加速因子步长 = 2% = 0.02
        - 20: 加速因子最大值 = 20% = 0.20
        """
        self.acceleration = acceleration
        self.maximum = maximum

        # 数据存储
        self._high_series: Optional[pd.Series] = None
        self._low_series: Optional[pd.Series] = None
        self._close_series: Optional[pd.Series] = None
        self._sar_series: Optional[pd.Series] = None

        # 状态跟踪
        self.state = SARState()

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

        # 计算SAR
        self._sar_series = self._calculate_sar(high_series, low_series, close_series)

        logger.info(f"SAR策略数据准备完成: {len(close_series)} 天数据")

    def _calculate_sar(self,
                        high: pd.Series,
                        low: pd.Series,
                        close: pd.Series) -> pd.Series:
        """
        使用 TA-Lib 计算 SAR（最接近东方财富）

        【说明】
        TA-Lib 的 SAR 算法是标准实现，与东方财富高度一致。
        参数：SAR(10,2,20) 对应：
        - acceleration: 0.02 (步长)
        - maximum: 0.20 (最大值)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列（TA-Lib SAR不需要，但保留参数一致性）

        Returns:
            pd.Series: SAR序列

        Raises:
            ImportError: 如果 TA-Lib 未安装
        """
        if talib is None:
            raise ImportError("必须安装 TA-Lib: pip install TA-Lib")

        # TA-Lib 要求输入为 numpy array
        high_np = high.values
        low_np = low.values

        # TA-Lib SAR 参数：acceleration 和 maximum
        sar_array = talib.SAR(high_np, low_np, acceleration=self.acceleration, maximum=self.maximum)

        return pd.Series(sar_array, index=high.index, name='SAR')

    def _calculate_sar_eastmoney_fixed(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        修正版：分离内部递推SAR和输出SAR

        【核心逻辑】
        1. internal_sar：用于递推计算，不做clamp限制
        2. output_sar：用于最终输出，应用clamp限制
        3. raw_sar使用internal_sar计算，确保递推逻辑不被clamp干扰

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            pd.Series: SAR序列
        """
        n = len(close)
        if n == 0:
            return pd.Series([], dtype=float)

        af_step = self.acceleration  # 0.02
        af_max = self.maximum        # 0.20

        # 内部递推用的 SAR（不 clamp）
        internal_sar = np.full(n, np.nan)
        # 最终输出的 SAR（clamp 后）
        output_sar = np.full(n, np.nan)

        # 初始化：第一天为多头，SAR=最低价
        internal_sar[0] = low.iloc[0]
        output_sar[0] = low.iloc[0]
        is_long = True
        ep = high.iloc[0]
        af = af_step

        for i in range(1, n):
            # Step 1: 用 internal_sar 计算 raw_sar（不受clamp影响）
            raw_sar = internal_sar[i - 1] + af * (ep - internal_sar[i - 1])

            reverse = False
            new_is_long = is_long
            new_ep = ep
            new_af = af

            if is_long:
                # 多头：检查是否跌破 SAR（用最低价）
                if low.iloc[i] <= raw_sar:
                    reverse = True
                    new_is_long = False
                    # 反转 SAR = 上一多头趋势的 EP（最高价）
                    internal_sar[i] = ep
                    output_sar[i] = ep
                    new_ep = low.iloc[i]      # 新 EP 为空头最低
                    new_af = af_step
                else:
                    internal_sar[i] = raw_sar
                    output_sar[i] = raw_sar
                    if high.iloc[i] > ep:
                        new_ep = high.iloc[i]
                        new_af = min(af + af_step, af_max)
            else:
                # 空头：检查是否突破 SAR（用最高价）
                if high.iloc[i] >= raw_sar:
                    reverse = True
                    new_is_long = True
                    # 反转 SAR = 上一空头趋势的 EP（最低价）
                    internal_sar[i] = ep
                    output_sar[i] = ep
                    new_ep = high.iloc[i]     # 新 EP 为多头最高
                    new_af = af_step
                else:
                    internal_sar[i] = raw_sar
                    output_sar[i] = raw_sar
                    if low.iloc[i] < ep:
                        new_ep = low.iloc[i]
                        new_af = min(af + af_step, af_max)

            # Step 2: 仅对 output_sar 做 clamp（不影响 internal_sar 递推）
            if not reverse:
                if new_is_long:
                    # 多头：SAR 不能高于前两日最低价
                    output_sar[i] = min(output_sar[i], low.iloc[i - 1], low.iloc[i])
                else:
                    # 空头：SAR 不能低于前两日最高价
                    output_sar[i] = max(output_sar[i], high.iloc[i - 1], high.iloc[i])

            is_long = new_is_long
            ep = new_ep
            af = new_af

        return pd.Series(output_sar, index=close.index, name='SAR')

    def _get_position(self, sar: float, close: float) -> str:
        """
        获取SAR相对价格的位置

        Args:
            sar: SAR值
            close: 收盘价

        Returns:
            str: 'above' 或 'below'
        """
        if sar > close:
            return 'above'  # SAR在价格上方（空头）
        else:
            return 'below'  # SAR在价格下方（多头）

    def detect_signals(self,
                        start_date: str,
                        end_date: str) -> List[SARSignal]:
        """
        检测红转绿和绿转红信号

        【标准反转判断逻辑】
        严格按照 SAR 点与收盘价的位置关系判断，与东方财富等主流平台保持一致：
        - 红转绿（买入）: SAR(t) < Close(t) 且 SAR(t-1) >= Close(t-1)
        - 绿转红（卖出）: SAR(t) > Close(t) 且 SAR(t-1) <= Close(t-1)

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[SARSignal]: 信号列表
        """
        if self._sar_series is None or self._close_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 使用字符串筛选时间段（避免时区问题）
        sar_data = self._sar_series[
            (self._sar_series.index.strftime('%Y-%m-%d') >= start_date) &
            (self._sar_series.index.strftime('%Y-%m-%d') <= end_date)
        ]
        close_data = self._close_series[
            (self._close_series.index.strftime('%Y-%m-%d') >= start_date) &
            (self._close_series.index.strftime('%Y-%m-%d') <= end_date)
        ]

        if len(sar_data) < 2:
            return []

        signals = []
        dates = sar_data.index

        for i in range(1, len(dates)):
            curr_date = dates[i]
            prev_date = dates[i-1]

            curr_sar = sar_data.loc[curr_date]
            prev_sar = sar_data.loc[prev_date]
            curr_close = close_data.loc[curr_date]
            prev_close = close_data.loc[prev_date]

            # 检测空头转多头（红转绿）
            # SAR(t) < Close(t) 且 SAR(t-1) >= Close(t-1)
            if (curr_sar < curr_close) and (prev_sar >= prev_close):
                signal = SARSignal(
                    date=curr_date,
                    signal_type='红转绿',
                    sar_value=curr_sar,
                    close_price=curr_close,
                    position='below'
                )
                signals.append(signal)
                self.state.current_position = '多头'
                self.state.last_signal_date = curr_date
                self.state.last_signal_type = '红转绿'
                logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 红转绿信号: SAR={curr_sar:.3f}, Close={curr_close:.3f}")

            # 检测多头转空头（绿转红）
            # SAR(t) > Close(t) 且 SAR(t-1) <= Close(t-1)
            elif (curr_sar > curr_close) and (prev_sar <= prev_close):
                signal = SARSignal(
                    date=curr_date,
                    signal_type='绿转红',
                    sar_value=curr_sar,
                    close_price=curr_close,
                    position='above'
                )
                signals.append(signal)
                self.state.current_position = '空头'
                self.state.last_signal_date = curr_date
                self.state.last_signal_type = '绿转红'
                logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 绿转红信号: SAR={curr_sar:.3f}, Close={curr_close:.3f}")

        logger.info(f"检测到 {len(signals)} 个SAR转折信号")
        return signals

    def get_current_position(self,
                               start_date: str,
                               end_date: str) -> Dict:
        """
        获取当前SAR位置状态

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 包含当前状态信息
        """
        if self._sar_series is None or self._close_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        sar_data = self._sar_series[
            (self._sar_series.index >= start_ts) &
            (self._sar_series.index <= end_ts)
        ]
        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]

        if len(sar_data) == 0:
            return {'error': '无数据'}

        last_date = sar_data.index[-1]
        last_sar = sar_data.loc[last_date]
        last_close = close_data.loc[last_date]
        position = self._get_position(last_sar, last_close)

        return {
            'date': last_date,
            'sar': last_sar,
            'close': last_close,
            'position': position,
            'trend': '多头' if position == 'below' else '空头',
            'color': 'green' if position == 'below' else 'red'
        }

    def run_backtest(self,
                      start_date: str,
                      end_date: str,
                      initial_cash: float = 100000.0) -> Dict:
        """
        运行SAR策略回测

        【回测规则】
        - 红转绿：买入（全仓）
        - 绿转红：卖出（清仓）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            initial_cash: 初始资金

        Returns:
            Dict: 回测结果
        """
        if self._close_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 筛选时间段
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        close_data = self._close_series[
            (self._close_series.index >= start_ts) &
            (self._close_series.index <= end_ts)
        ]

        if len(close_data) == 0:
            return {'error': '无数据'}

        # 检测信号
        signals = self.detect_signals(start_date, end_date)

        # 创建信号字典
        signal_dict = {s.date: s for s in signals}

        # 初始状态
        first_price = close_data.iloc[0]
        cash = initial_cash
        shares = 0.0

        # 交易记录
        trades = []

        # 遍历交易日
        for date in close_data.index:
            price = close_data.loc[date]

            if date in signal_dict:
                signal = signal_dict[date]

                if signal.signal_type == '红转绿':
                    # 买入信号
                    if cash > 0:
                        buy_shares = cash / price
                        shares = buy_shares
                        cash = 0.0
                        trades.append({
                            'date': date,
                            'type': '买入（红转绿）',
                            'price': price,
                            'shares': buy_shares,
                            'cash_after': cash,
                            'shares_after': shares,
                            'sar': signal.sar_value
                        })

                elif signal.signal_type == '绿转红':
                    # 卖出信号
                    if shares > 0:
                        sell_value = shares * price
                        cash = sell_value
                        trades.append({
                            'date': date,
                            'type': '卖出（绿转红）',
                            'price': price,
                            'shares': shares,
                            'cash_after': cash,
                            'shares_after': 0.0,
                            'sar': signal.sar_value
                        })
                        shares = 0.0

        # 计算最终市值
        final_price = close_data.iloc[-1]
        final_value = shares * final_price + cash
        strategy_return = (final_value - initial_cash) / initial_cash * 100

        # 计算基准收益率
        benchmark_shares = initial_cash / first_price
        benchmark_value = benchmark_shares * final_price
        benchmark_return = (benchmark_value - initial_cash) / initial_cash * 100

        return {
            'trades': trades,
            'signals': signals,
            'initial_cash': initial_cash,
            'first_price': first_price,
            'final_price': final_price,
            'final_shares': shares,
            'final_cash': cash,
            'final_value': final_value,
            'strategy_return': strategy_return,
            'benchmark_shares': benchmark_shares,
            'benchmark_value': benchmark_value,
            'benchmark_return': benchmark_return,
            'total_trades': len(trades),
            'total_signals': len(signals)
        }

    def plot_sar_strategy(self,
                           start_date: str,
                           end_date: str,
                           save_path: str,
                           stock_name: str = '',
                           result: Optional[Dict] = None) -> None:
        """
        绘制SAR策略图表

        Args:
            start_date: 开始日期
            end_date: 结束日期
            save_path: 图片保存路径
            stock_name: 股票名称
            result: 回测结果（可选）
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
        sar_data = self._sar_series[
            (self._sar_series.index >= start_ts) &
            (self._sar_series.index <= end_ts)
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

        # 检测信号
        signals = self.detect_signals(start_date, end_date)

        # 创建图表
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                             hspace=0.25, wspace=0.08)

        ax_price = fig.add_subplot(gs[0, 0])
        ax_sar = fig.add_subplot(gs[1, 0])
        ax_signals = fig.add_subplot(gs[0, 1])
        ax_summary = fig.add_subplot(gs[1, 1])

        fig.suptitle(f'{stock_name} SAR策略 ({start_date} ~ {end_date})',
                    fontsize=14, fontweight='bold')

        # ===== 绘制价格和SAR =====
        dates = close_data.index

        # 绘制收盘价
        ax_price.plot(dates, close_data.values, 'b-', linewidth=1.5,
                     label='收盘价', alpha=0.8)

        # 绘制日内波动区间
        ax_price.fill_between(dates, low_data.values, high_data.values,
                              color='blue', alpha=0.15, label='日内波动')

        # 绘制SAR点（红色=上方/空头，绿色=下方/多头）
        sar_above = []
        sar_above_dates = []
        sar_below = []
        sar_below_dates = []

        for i, date in enumerate(dates):
            sar = sar_data.loc[date]
            close = close_data.loc[date]
            if sar > close:
                sar_above.append(sar)
                sar_above_dates.append(date)
            else:
                sar_below.append(sar)
                sar_below_dates.append(date)

        # 绘制SAR点
        if sar_above_dates:
            ax_price.scatter(sar_above_dates, sar_above, color='red', marker='.',
                            s=30, label='SAR(空头)', alpha=0.7)
        if sar_below_dates:
            ax_price.scatter(sar_below_dates, sar_below, color='green', marker='.',
                            s=30, label='SAR(多头)', alpha=0.7)

        # 标记转折信号
        for signal in signals:
            if signal.signal_type == '红转绿':
                ax_price.scatter(signal.date, signal.close_price, color='green',
                                marker='^', s=150, zorder=5)
                ax_price.annotate(f'红转绿\n{signal.close_price:.3f}',
                                 (signal.date, signal.close_price),
                                 textcoords="offset points", xytext=(10, 15),
                                 fontsize=8, color='green',
                                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            elif signal.signal_type == '绿转红':
                ax_price.scatter(signal.date, signal.close_price, color='red',
                                marker='v', s=150, zorder=5)
                ax_price.annotate(f'绿转红\n{signal.close_price:.3f}',
                                 (signal.date, signal.close_price),
                                 textcoords="offset points", xytext=(10, -15),
                                 fontsize=8, color='red',
                                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax_price.set_ylabel('价格')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # ===== 绘制SAR曲线 =====
        ax_sar.plot(dates, sar_data.values, 'purple', linewidth=1, label='SAR')
        ax_sar.plot(dates, close_data.values, 'b-', linewidth=0.5, label='收盘价', alpha=0.5)

        # 标记转折点
        for signal in signals:
            color = 'green' if signal.signal_type == '红转绿' else 'red'
            ax_sar.scatter(signal.date, signal.sar_value, color=color, s=80, zorder=5)

        ax_sar.set_ylabel('SAR')
        ax_sar.legend(loc='upper left')
        ax_sar.grid(True, alpha=0.3)
        ax_sar.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # ===== 绘制信号表格 =====
        ax_signals.axis('off')

        if signals:
            table_data = []
            for s in signals:
                date_str = s.date.strftime('%Y-%m-%d')
                signal_type = s.signal_type
                sar = f"{s.sar_value:.3f}"
                close = f"{s.close_price:.3f}"
                trend = '多头' if s.position == 'below' else '空头'
                table_data.append([date_str, signal_type, sar, close, trend])

            col_labels = ['日期', '信号类型', 'SAR值', '收盘价', '趋势']
            table = ax_signals.table(cellText=table_data, colLabels=col_labels,
                                      loc='upper center', cellLoc='left',
                                      colWidths=[0.2, 0.2, 0.15, 0.15, 0.15])

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # 设置表头样式
            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white', fontweight='bold')

            # 设置行样式
            for i, s in enumerate(signals):
                row_idx = i + 1
                if s.signal_type == '红转绿':
                    for j in range(len(col_labels)):
                        table[(row_idx, j)].set_facecolor('#E6FFE6')
                else:
                    for j in range(len(col_labels)):
                        table[(row_idx, j)].set_facecolor('#FFE6E6')

        ax_signals.set_title('SAR转折信号', fontsize=10, fontweight='bold', pad=10)

        # ===== 绘制收益对比 =====
        ax_summary.axis('off')

        if result is not None:
            strategy_return = result['strategy_return']
            benchmark_return = result['benchmark_return']
            diff = strategy_return - benchmark_return

            summary_text = (
                f"━━━━━━ SAR策略收益对比 ━━━━━━\n\n"
                f"【策略收益】\n"
                f"  收益率: {strategy_return:.2f}%\n"
                f"  最终市值: {result['final_value']:,.2f} 元\n"
                f"  交易次数: {result['total_trades']} 次\n\n"
                f"【基准收益】（不做买卖）\n"
                f"  收益率: {benchmark_return:.2f}%\n"
                f"  最终市值: {result['benchmark_value']:,.2f} 元\n\n"
                f"【超额收益】\n"
            )

            if diff > 0:
                summary_text += f"  +{diff:.2f}% （优于基准）"
                summary_color = 'green'
            elif diff < 0:
                summary_text += f"  {diff:.2f}% （劣于基准）"
                summary_color = 'red'
            else:
                summary_text += f"  {diff:.2f}% （与基准持平）"
                summary_color = 'gray'

            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                             edgecolor='#333333', alpha=0.9)
            ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                           fontsize=9, verticalalignment='center', horizontalalignment='center',
                           bbox=bbox_props, family='monospace')

            ax_summary.text(0.5, 0.15, f"超额收益: {diff:.2f}%",
                           transform=ax_summary.transAxes, fontsize=11,
                           verticalalignment='center', horizontalalignment='center',
                           color=summary_color, fontweight='bold')

        ax_summary.set_title('收益对比', fontsize=10, fontweight='bold', pad=5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"SAR策略图已保存至: {save_path}")

    # ==================== 实时数据功能 ====================

    def get_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """
        获取实时行情数据

        【功能说明】
        获取股票当前实时行情，并计算今日SAR值。

        Args:
            symbol: 股票代码（需要交易所前缀，如 SHSE.512480）

        Returns:
            RealtimeQuote: 实时行情数据对象
        """
        if current is None:
            logger.warning("掘金SDK未安装，无法获取实时数据")
            return None

        try:
            # 获取实时行情
            realtime_data = current(symbols=symbol)
            if realtime_data is None or len(realtime_data) == 0:
                logger.warning(f"获取实时数据失败: {symbol}")
                return None

            rt = realtime_data[0]
            rt_time = pd.Timestamp(rt['created_at'])

            # 获取历史数据用于计算SAR
            today_str = rt_time.strftime('%Y-%m-%d')
            warmup_start = (rt_time - pd.Timedelta(days=60)).strftime('%Y-%m-%d')

            if history is None:
                logger.warning("掘金SDK未安装，无法获取历史数据")
                return None

            # 获取历史日线数据（不包含今天）
            daily_data = history(
                symbol=symbol,
                frequency='1d',
                start_time=warmup_start + ' 09:00:00',
                end_time=today_str + ' 09:00:00',  # 截止到今天开盘前
                fields='eob,open,high,low,close,volume',
                df=True,
                adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
            )

            if daily_data is None or daily_data.empty:
                logger.warning(f"获取历史数据失败: {symbol}")
                return None

            daily_data['eob'] = pd.to_datetime(daily_data['eob'])
            df_hist = daily_data.set_index('eob').sort_index()

            # 创建历史价格序列
            high_series = pd.Series(df_hist['high'].values, index=df_hist.index, name='High')
            low_series = pd.Series(df_hist['low'].values, index=df_hist.index, name='Low')
            close_series = pd.Series(df_hist['close'].values, index=df_hist.index, name='Close')

            # 计算历史SAR
            sar_hist = self._calculate_sar(high_series, low_series, close_series)

            # 获取昨日SAR值和昨日收盘价
            if len(sar_hist) < 2:
                logger.warning(f"历史数据不足: {symbol}")
                return None

            sar_yesterday = sar_hist.iloc[-1]
            prev_close = close_series.iloc[-1]

            # 获取股票名称
            stock_name = "未知"
            if get_instruments is not None:
                try:
                    instruments = get_instruments(symbol)
                    if instruments:
                        stock_name = getattr(instruments[0], 'sec_name', '未知')
                except Exception:
                    pass

            # 判断昨日趋势
            trend = '多头' if sar_yesterday < prev_close else '空头'

            # 计算今日预估SAR（基于昨日数据）
            # 使用实时数据构建今日SAR
            today_high_series = pd.concat([high_series, pd.Series([rt['high']], index=[rt_time])])
            today_low_series = pd.concat([low_series, pd.Series([rt['low']], index=[rt_time])])
            today_close_series = pd.concat([close_series, pd.Series([rt['price']], index=[rt_time])])

            # 重新计算包含今日数据的SAR
            sar_today_full = self._calculate_sar(today_high_series, today_low_series, today_close_series)
            sar_today = sar_today_full.iloc[-1]

            quote = RealtimeQuote(
                symbol=symbol,
                stock_name=stock_name,
                current_time=rt_time.to_pydatetime(),
                open=rt['open'],
                high=rt['high'],
                low=rt['low'],
                price=rt['price'],
                volume=rt['cum_volume'],
                sar_today=sar_today,
                sar_yesterday=sar_yesterday,
                prev_close=prev_close,
                trend=trend
            )

            logger.info(f"实时行情: {symbol} ({stock_name}) "
                       f"价格={rt['price']:.3f}, SAR昨日={sar_yesterday:.3f}, SAR今日={sar_today:.3f}, 趋势={trend}")

            return quote

        except Exception as e:
            logger.error(f"获取实时数据异常: {symbol} - {e}")
            return None

    def check_realtime_signal(self, quote: RealtimeQuote) -> Optional[RealtimeAlert]:
        """
        检测实盘SAR转折信号

        【标准反转判断逻辑】
        与 detect_signals 保持一致，基于 SAR 与价格的位置关系判断：
        - 红转绿（买入）：今日SAR < 当前价格 且 昨日SAR >= 昨日收盘价
        - 绿转红（卖出）：今日SAR > 当前价格 且 昨日SAR <= 昨日收盘价

        注意：由于实盘中收盘价未确定，使用当前实时价格作为收盘价的代理

        Args:
            quote: 实时行情数据

        Returns:
            RealtimeAlert: 告警信息（如果发生转折）
        """
        alert = None

        # 检测红转绿（空头转多头）
        # SAR_today < 当前价格 且 SAR_yesterday >= prev_close
        if (quote.sar_today < quote.price) and (quote.sar_yesterday >= quote.prev_close):
            alert = RealtimeAlert(
                symbol=quote.symbol,
                stock_name=quote.stock_name,
                alert_time=quote.current_time,
                signal_type='红转绿',
                current_price=quote.price,
                sar_value=quote.sar_today,
                prev_sar=quote.sar_yesterday,
                prev_close=quote.prev_close,
                realtime_high=quote.high,
                realtime_low=quote.low,
                trend_before='空头',
                trend_after='多头'
            )
            logger.info(f"[实盘红转绿] {quote.stock_name} ({quote.symbol}) "
                       f"SAR={quote.sar_today:.3f} < 价格={quote.price:.3f}")

        # 检测绿转红（多头转空头）
        # SAR_today > 当前价格 且 SAR_yesterday <= prev_close
        elif (quote.sar_today > quote.price) and (quote.sar_yesterday <= quote.prev_close):
            alert = RealtimeAlert(
                symbol=quote.symbol,
                stock_name=quote.stock_name,
                alert_time=quote.current_time,
                signal_type='绿转红',
                current_price=quote.price,
                sar_value=quote.sar_today,
                prev_sar=quote.sar_yesterday,
                prev_close=quote.prev_close,
                realtime_high=quote.high,
                realtime_low=quote.low,
                trend_before='多头',
                trend_after='空头'
            )
            logger.info(f"[实盘绿转红] {quote.stock_name} ({quote.symbol}) "
                       f"SAR={quote.sar_today:.3f} > 价格={quote.price:.3f}")

        return alert

    def get_all_sar_values(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        获取所有SAR值（历史数据 + 今日实时数据）

        【功能说明】
        获取指定股票的历史SAR序列，包含收盘价和趋势状态。

        Args:
            symbol: 股票代码（需要交易所前缀，如 SHSE.512480）
            days: 返回最近多少天的数据，默认30天

        Returns:
            DataFrame: 包含 date, close, sar, trend 列的数据表
        """
        if history is None:
            logger.warning("掘金SDK未安装，无法获取历史数据")
            return None

        try:
            # 获取实时时间作为截止日期
            today_str = datetime.now().strftime('%Y-%m-%d')
            warmup_start = (datetime.now() - pd.Timedelta(days=days + 60)).strftime('%Y-%m-%d')

            # 获取历史日线数据
            daily_data = history(
                symbol=symbol,
                frequency='1d',
                start_time=warmup_start + ' 09:00:00',
                end_time=today_str + ' 15:30:00',
                fields='eob,open,high,low,close,volume',
                df=True,
                adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
            )

            if daily_data is None or daily_data.empty:
                logger.warning(f"获取数据失败: {symbol}")
                return None

            daily_data['eob'] = pd.to_datetime(daily_data['eob'])
            df = daily_data.set_index('eob').sort_index()

            # 创建价格序列
            high_series = pd.Series(df['high'].values, index=df.index, name='High')
            low_series = pd.Series(df['low'].values, index=df.index, name='Low')
            close_series = pd.Series(df['close'].values, index=df.index, name='Close')

            # 计算SAR
            sar_series = self._calculate_sar(high_series, low_series, close_series)

            # 构建结果DataFrame
            result = pd.DataFrame({
                'date': sar_series.index,
                'close': close_series.values,
                'sar': sar_series.values
            })

            # 添加趋势列
            result['trend'] = result.apply(
                lambda row: '多头' if row['sar'] < row['close'] else '空头', axis=1
            )

            # 添加趋势颜色
            result['color'] = result.apply(
                lambda row: 'green' if row['sar'] < row['close'] else 'red', axis=1
            )

            # 只返回最近N天的数据
            result = result.tail(days)

            return result

        except Exception as e:
            logger.error(f"获取SAR值异常: {symbol} - {e}")
            return None


class DingTalkNotifier:
    """
    钉钉机器人告警类

    【功能说明】
    通过钉钉机器人发送告警消息，使用加签安全模式。

    【安全模式】
    使用加签（signature）方式验证消息来源的安全性。
    签名算法：HmacSHA256(timestamp + "\\n" + secret, secret)
    """

    def __init__(self, webhook: str, secret: str) -> None:
        """
        初始化钉钉通知器

        Args:
            webhook: 钉钉机器人webhook地址
            secret: 钉钉加签密钥
        """
        self.webhook = webhook
        self.secret = secret

    def _generate_sign(self, timestamp: int) -> str:
        """
        生成加签

        Args:
            timestamp: 当前时间戳（毫秒）

        Returns:
            str: 签名字符串
        """
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return sign

    def send_markdown(self, title: str, content: str, at_all: bool = True) -> bool:
        """
        发送Markdown消息

        Args:
            title: 消息标题
            content: Markdown格式内容
            at_all: 是否@所有人

        Returns:
            bool: 发送是否成功
        """
        timestamp = int(time.time() * 1000)
        sign = self._generate_sign(timestamp)

        url = f"{self.webhook}&timestamp={timestamp}&sign={sign}"

        headers = {'Content-Type': 'application/json;charset=utf-8'}
        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": content
            },
            "at": {
                "atMobiles": [],
                "isAtAll": at_all
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers=headers
            )
            response = urllib.request.urlopen(req, timeout=10)
            result = json.loads(response.read().decode('utf-8'))

            if result.get('errcode') == 0:
                return True
            else:
                logger.error(f"钉钉发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"钉钉发送异常: {e}")
            return False


class SARRealtimeMonitor:
    """
    SAR实盘监控类

    【功能说明】
    实时监控股票的SAR指标转折信号，发送钉钉告警。

    【监控逻辑】
    1. 加载股票配置（stocks_backtest.csv中active=1的股票）
    2. 获取实时行情数据
    3. 计算SAR指标
    4. 检测转折信号（红转绿/绿转红）
    5. 发送钉钉告警

    【使用方法】
    monitor = SARRealtimeMonitor(settings_path, config_path)
    monitor.run_realtime_monitor()
    """

    # 监控时间点（交易时间内每小时）
    MONITOR_TIMES = ['10:00', '11:00', '13:00', '14:00']

    # SAR参数
    SAR_ACCELERATION = 0.02
    SAR_MAXIMUM = 0.20

    def __init__(self,
                 settings_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None) -> None:
        """
        初始化实盘监控器

        Args:
            settings_path: 设置文件路径（包含gm_token和钉钉配置）
            config_path: 股票配置文件路径（stocks_backtest.csv）
            output_dir: 输出目录路径
        """
        # 确定项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        # 配置文件路径
        if settings_path is None:
            settings_path = os.path.join(project_root, 'config', 'settings.json')
        self.settings_path = settings_path

        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'stocks_backtest.csv')
        self.config_path = config_path

        # 输出目录
        if output_dir is None:
            output_dir = os.path.join(project_root, 'logs', 'sar_realtime')
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 配置日志
        self._setup_logging()

        # 初始化设置
        self._init_settings()

        # SAR策略实例
        self.sar_strategy = SARStrategy(
            acceleration=self.SAR_ACCELERATION,
            maximum=self.SAR_MAXIMUM
        )

    def _setup_logging(self) -> None:
        """配置日志"""
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(self.output_dir, f'sar_realtime_{timestamp}.log')

        # 清除已有handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file

    def _init_settings(self) -> None:
        """初始化设置（掘金token和钉钉配置）"""
        if not os.path.exists(self.settings_path):
            self.logger.error(f"设置文件不存在: {self.settings_path}")
            return

        with open(self.settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # 初始化掘金token
        gm_token = settings.get('gm_token')
        if gm_token and set_token is not None:
            set_token(gm_token)
            self.logger.info("掘金token初始化成功")

        # 初始化钉钉通知器
        dingtalk_webhook = settings.get('dingtalk_webhook')
        dingtalk_secret = settings.get('dingtalk_secret')
        if dingtalk_webhook and dingtalk_secret:
            self.dingtalk = DingTalkNotifier(dingtalk_webhook, dingtalk_secret)
            self.logger.info("钉钉通知器初始化成功")
        else:
            self.dingtalk = None
            self.logger.warning("钉钉配置未设置")

    def _add_exchange_prefix(self, code: str) -> str:
        """自动添加交易所前缀"""
        code = code.strip()
        if code.startswith('SHSE.') or code.startswith('SZSE.'):
            return code
        if code.startswith('6') or code.startswith('5'):
            return f'SHSE.{code}'
        elif code.startswith('0') or code.startswith('3') or code.startswith('1'):
            return f'SZSE.{code}'
        else:
            return f'SZSE.{code}'

    def load_stock_config(self) -> List[Dict]:
        """
        加载股票配置

        Returns:
            List[Dict]: 股票配置列表（只包含active=1的股票）
        """
        if not os.path.exists(self.config_path):
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return []

        df = pd.read_csv(self.config_path)
        stocks = []

        for _, row in df.iterrows():
            # 只加载active=1的股票
            active = int(row['active']) if not pd.isna(row['active']) else 0
            if active != 1:
                continue

            symbol = self._add_exchange_prefix(str(row['symbol']))
            stock_name = str(row.get('stock_name', '未知'))

            stocks.append({
                'symbol': symbol,
                'stock_name': stock_name
            })

        self.logger.info(f"加载股票配置: {len(stocks)} 只（active=1）")
        return stocks

    def get_all_sar_values(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        获取所有SAR值

        Args:
            symbol: 股票代码
            days: 返回最近多少天的数据

        Returns:
            DataFrame: SAR值数据表
        """
        return self.sar_strategy.get_all_sar_values(symbol, days)

    def print_all_sar_values(self, symbol: str, days: int = 30) -> None:
        """
        打印所有SAR值

        Args:
            symbol: 股票代码
            days: 打印最近多少天的数据
        """
        # 添加交易所前缀
        full_symbol = self._add_exchange_prefix(symbol)

        # 获取股票名称
        stock_name = "未知"
        if get_instruments is not None:
            try:
                instruments = get_instruments(full_symbol)
                if instruments:
                    stock_name = getattr(instruments[0], 'sec_name', '未知')
            except Exception:
                pass

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"所有SAR值 - {stock_name} ({full_symbol})")
        self.logger.info("=" * 80)
        self.logger.info(f"{'日期':<12} {'收盘价':<10} {'SAR值':<10} {'趋势':<8} {'颜色':<6}")
        self.logger.info("-" * 50)

        df = self.get_all_sar_values(full_symbol, days)
        if df is None or df.empty:
            self.logger.warning("无SAR数据")
            return

        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
            self.logger.info(f"{date_str:<12} {row['close']:<10.3f} {row['sar']:<10.3f} "
                           f"{row['trend']:<8} {row['color']:<6}")

        self.logger.info("-" * 50)
        self.logger.info(f"共 {len(df)} 条数据")

    def run_realtime_monitor(self, show_all_sar: bool = True, sar_days: int = 30) -> List[RealtimeAlert]:
        """
        执行实盘监控

        Args:
            show_all_sar: 是否显示所有SAR值
            sar_days: 显示最近多少天的SAR值

        Returns:
            List[RealtimeAlert]: 检测到的告警列表
        """
        self.logger.info("=" * 80)
        self.logger.info("SAR实盘监控启动")
        self.logger.info(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"SAR参数: acceleration={self.SAR_ACCELERATION}, maximum={self.SAR_MAXIMUM}")
        self.logger.info("=" * 80)

        # 加载股票配置
        stocks = self.load_stock_config()
        if len(stocks) == 0:
            self.logger.error("无活跃股票配置")
            return []

        # 告警列表
        all_alerts = []

        # 逐只股票监控
        for stock in stocks:
            symbol = stock['symbol']
            self.logger.info(f"监控股票: {symbol} ({stock['stock_name']})")

            # 输出所有SAR值
            if show_all_sar:
                self.print_all_sar_values(symbol, sar_days)

            # 获取实时行情（如果掘金SDK可用）
            if current is not None:
                quote = self.sar_strategy.get_realtime_quote(symbol)
                if quote is None:
                    continue

                # 检测信号
                alert = self.sar_strategy.check_realtime_signal(quote)
                if alert is not None:
                    all_alerts.append(alert)

        # 输出监控报告
        self._output_report(all_alerts)

        # 发送钉钉告警
        self._send_dingtalk_alert(all_alerts)

        return all_alerts

    def _output_report(self, alerts: List[RealtimeAlert]) -> None:
        """
        输出监控报告

        Args:
            alerts: 告警列表
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SAR实盘监控报告")
        self.logger.info("=" * 80)

        if len(alerts) == 0:
            self.logger.info("无SAR转折信号")
        else:
            self.logger.info(f"检测到 {len(alerts)} 个转折信号")
            self.logger.info("-" * 60)

            for alert in alerts:
                self.logger.info(f"[{alert.alert_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                               f"{alert.stock_name} ({alert.symbol})")
                self.logger.info(f"  信号类型: {alert.signal_type}")
                self.logger.info(f"  当前价格: {alert.current_price:.3f}")
                self.logger.info(f"  今日SAR: {alert.sar_value:.3f}")
                self.logger.info(f"  昨日SAR: {alert.prev_sar:.3f}")
                self.logger.info(f"  转折前趋势: {alert.trend_before} -> 转折后: {alert.trend_after}")
                if alert.signal_type == '红转绿':
                    self.logger.info(f"  实时最高价: {alert.realtime_high:.3f} (突破SAR)")
                else:
                    self.logger.info(f"  实时最低价: {alert.realtime_low:.3f} (跌破SAR)")
                self.logger.info("")

        self.logger.info("=" * 80)

    def _send_dingtalk_alert(self, alerts: List[RealtimeAlert]) -> None:
        """
        发送钉钉告警

        Args:
            alerts: 告警列表
        """
        if self.dingtalk is None:
            self.logger.warning("钉钉配置未设置，跳过告警发送")
            return

        # 如果没有告警，不发送消息
        if len(alerts) == 0:
            self.logger.info("无SAR转折信号，不发送钉钉告警")
            return

        # 构建Markdown消息
        title = f"SAR实盘告警 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        content_lines = []
        content_lines.append(f"## SAR实盘转折告警\n")
        content_lines.append(f"**监控时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content_lines.append(f"**SAR参数**: acceleration={self.SAR_ACCELERATION}, maximum={self.SAR_MAXIMUM}\n")
        content_lines.append(f"**转折股票数**: {len(alerts)} 只\n\n")

        # 分组显示信号
        red_to_green = [a for a in alerts if a.signal_type == '红转绿']
        green_to_red = [a for a in alerts if a.signal_type == '绿转红']

        if len(red_to_green) > 0:
            content_lines.append(f"### 🟢 红转绿信号（买入）({len(red_to_green)}只)\n")
            content_lines.append("---\n")
            for alert in red_to_green:
                content_lines.append(f"- **{alert.stock_name}** ({alert.symbol})\n")
                content_lines.append(f"  - 当前价格: {alert.current_price:.3f}\n")
                content_lines.append(f"  - 今日SAR: {alert.sar_value:.3f}\n")
                content_lines.append(f"  - 实时最高价: {alert.realtime_high:.3f} (突破上方阻力)\n")
                content_lines.append(f"  - 趋势变化: 空头 -> 多头\n\n")

        if len(green_to_red) > 0:
            content_lines.append(f"### 🔴 绿转红信号（卖出）({len(green_to_red)}只)\n")
            content_lines.append("---\n")
            for alert in green_to_red:
                content_lines.append(f"- **{alert.stock_name}** ({alert.symbol})\n")
                content_lines.append(f"  - 当前价格: {alert.current_price:.3f}\n")
                content_lines.append(f"  - 今日SAR: {alert.sar_value:.3f}\n")
                content_lines.append(f"  - 实时最低价: {alert.realtime_low:.3f} (跌破下方支撑)\n")
                content_lines.append(f"  - 趋势变化: 多头 -> 空头\n\n")

        content_lines.append(f"### 操作建议\n")
        content_lines.append("---\n")
        content_lines.append(f"- **红转绿**: 趋势从空头转为多头，可考虑买入或加仓\n")
        content_lines.append(f"- **绿转红**: 趋势从多头转为空头，建议关注风险，考虑减仓或止损\n")

        content = "".join(content_lines)

        # 发送钉钉消息
        success = self.dingtalk.send_markdown(title, content, at_all=True)

        if success:
            self.logger.info("钉钉告警发送成功")
        else:
            self.logger.error("钉钉告警发送失败")

    def is_trading_time(self) -> bool:
        """
        检查是否为交易时间

        Returns:
            bool: 是否为交易时间（9:30-15:00）
        """
        now = datetime.now()
        current_time = now.strftime('%H:%M')

        # 上午交易时间: 9:30-11:30
        if '09:30' <= current_time <= '11:30':
            return True

        # 下午交易时间: 13:00-15:00
        if '13:00' <= current_time <= '15:00':
            return True

        return False

    def is_trading_day(self) -> bool:
        """
        检查是否为交易日

        Returns:
            bool: 是否为交易日
        """
        try:
            if get_trading_dates is None:
                weekday = datetime.now().weekday()
                return weekday < 5

            today = datetime.now().strftime('%Y-%m-%d')
            trading_dates = get_trading_dates(
                exchange='SHSE',
                start_date=today,
                end_date=today
            )
            return len(trading_dates) > 0

        except Exception:
            weekday = datetime.now().weekday()
            return weekday < 5

    def run_scheduler(self) -> None:
        """
        运行定时调度器

        交易时间每小时执行一次监控：
        - 10:00
        - 11:00
        - 13:00
        - 14:00
        """
        print("=" * 60)
        print("SAR实盘监控调度器启动")
        print(f"监控时间点: {self.MONITOR_TIMES}")
        print("=" * 60)

        while True:
            now = datetime.now()
            current_time = now.strftime('%H:%M')

            if self.is_trading_day() and self.is_trading_time():
                if current_time in self.MONITOR_TIMES:
                    print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发SAR实盘监控...")
                    self.run_realtime_monitor()
                    time.sleep(60)  # 避免同一分钟内重复触发

            time.sleep(60)


# ==================== 测试模块 ====================

def run_realtime_test() -> None:
    """
    测试实盘监控功能
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("实盘监控测试")
    logger.info("=" * 60)

    # 创建监控器
    monitor = SARRealtimeMonitor()
    alerts = monitor.run_realtime_monitor()

    logger.info(f"检测到 {len(alerts)} 个实盘告警")


def run_realtime_single_stock(symbol: str) -> None:
    """
    测试单只股票的实时数据获取

    Args:
        symbol: 股票代码（如 '512480' 或 'SHSE.512480'）
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"单只股票实时数据测试: {symbol}")
    logger.info("=" * 60)

    # 添加交易所前缀
    if not symbol.startswith('SHSE.') and not symbol.startswith('SZSE.'):
        if symbol.startswith('6') or symbol.startswith('5'):
            symbol = f'SHSE.{symbol}'
        else:
            symbol = f'SZSE.{symbol}'

    # 创建策略实例
    strategy = SARStrategy()
    quote = strategy.get_realtime_quote(symbol)

    if quote is not None:
        logger.info(f"股票名称: {quote.stock_name}")
        logger.info(f"当前时间: {quote.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"当前价格: {quote.price:.3f}")
        logger.info(f"今日开盘: {quote.open:.3f}")
        logger.info(f"今日最高: {quote.high:.3f}")
        logger.info(f"今日最低: {quote.low:.3f}")
        logger.info(f"昨日收盘: {quote.prev_close:.3f}")
        logger.info(f"昨日SAR: {quote.sar_yesterday:.3f}")
        logger.info(f"今日SAR: {quote.sar_today:.3f}")
        logger.info(f"当前趋势: {quote.trend}")

        # 检测信号
        alert = strategy.check_realtime_signal(quote)
        if alert is not None:
            logger.info("")
            logger.info(f"检测到转折信号: {alert.signal_type}")
            logger.info(f"转折前趋势: {alert.trend_before}")
            logger.info(f"转折后趋势: {alert.trend_after}")
        else:
            logger.info("无转折信号")


if __name__ == "__main__":
    import argparse
    import os
    import json
    from datetime import datetime

    # 配置日志
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    LOG_DIR = os.path.join(project_root, 'logs')

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_FILE = os.path.join(LOG_DIR, f'sar_strategy_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="SAR策略模块")
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'realtime', 'single', 'schedule', 'sar'],
                       help="运行模式: backtest(回测), realtime(实盘监控), single(单只股票), schedule(定时调度), sar(获取所有SAR值)")
    parser.add_argument('--symbol', type=str, default='512480',
                       help="股票代码（用于single/sar模式）")
    parser.add_argument('--start', type=str, default='2026-01-01',
                       help="回测开始日期")
    parser.add_argument('--end', type=str, default='2026-04-16',
                       help="回测结束日期")
    parser.add_argument('--days', type=int, default=30,
                       help="获取最近多少天的SAR值（用于sar模式）")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SAR策略模块测试")
    logger.info(f"运行模式: {args.mode}")
    logger.info("=" * 60)

    if args.mode == 'realtime':
        # 实盘监控测试
        run_realtime_test()

    elif args.mode == 'single':
        # 单只股票实时测试
        run_realtime_single_stock(args.symbol)

    elif args.mode == 'schedule':
        # 定时调度模式
        monitor = SARRealtimeMonitor()
        monitor.run_scheduler()

    elif args.mode == 'sar':
        # 获取所有SAR值
        monitor = SARRealtimeMonitor()
        monitor.print_all_sar_values(args.symbol, args.days)

    else:
        # 回测模式（原有逻辑）
        TEST_TICKER = args.symbol
        TEST_NAME = '半导体ETF' if args.symbol == '512480' else args.symbol
        START_DATE = args.start
        END_DATE = args.end

        # SAR参数 SAR(10,2,20)
        SAR_ACCELERATION = 0.02
        SAR_MAXIMUM = 0.20

        # 初始资金
        INITIAL_CASH = 100000.0

        logger.info(f"测试股票: {TEST_NAME} ({TEST_TICKER})")
        logger.info(f"  检测时间段: {START_DATE} ~ {END_DATE}")
        logger.info(f"  SAR参数: SAR(10,2,20)")
        logger.info(f"  加速因子步长: {SAR_ACCELERATION} (对应参数中的2)")
        logger.info(f"  加速因子最大: {SAR_MAXIMUM} (对应参数中的20)")
        logger.info(f"  初始资金: {INITIAL_CASH:,.2f} 元")

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

            # 创建SAR策略实例
            strategy = SARStrategy(
                acceleration=SAR_ACCELERATION,
                maximum=SAR_MAXIMUM
            )
            strategy.prepare_data(high_series, low_series, close_series)

            # 打印检测时间段内的所有SAR值
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"每日SAR值明细 ({START_DATE} ~ {END_DATE})")
            logger.info("=" * 60)
            logger.info(f"{'日期':<12} {'收盘价':<10} {'SAR值':<10} {'趋势状态':<20}")
            logger.info("-" * 60)

            sar_series = strategy._sar_series[
                (strategy._sar_series.index.strftime('%Y-%m-%d') >= START_DATE) &
                (strategy._sar_series.index.strftime('%Y-%m-%d') <= END_DATE)
            ]
            close_series_data = strategy._close_series[
                (strategy._close_series.index.strftime('%Y-%m-%d') >= START_DATE) &
                (strategy._close_series.index.strftime('%Y-%m-%d') <= END_DATE)
            ]

            for i in range(len(sar_series)):
                date = sar_series.index[i]
                sar_val = sar_series.iloc[i]
                close_val = close_series_data.iloc[i]
                if sar_val > close_val:
                    trend = '空头(SAR > Close, red)'
                else:
                    trend = '多头(SAR < Close, green)'
                logger.info(f"{date.strftime('%Y-%m-%d'):<12} {close_val:<10.4f} {sar_val:<10.4f} {trend:<20}")

            logger.info("-" * 60)
            logger.info(f"共 {len(sar_series)} 条数据")

            # 检测信号
            logger.info("")
            logger.info("=" * 60)
            logger.info("SAR转折信号检测")
            logger.info("=" * 60)

            signals = strategy.detect_signals(START_DATE, END_DATE)

            logger.info(f"检测到 {len(signals)} 个转折信号")

            for s in signals:
                logger.info(f"[{s.date.strftime('%Y-%m-%d')}] {s.signal_type}: "
                           f"SAR={s.sar_value:.3f}, Close={s.close_price:.3f}")

            # 运行回测
            result = strategy.run_backtest(START_DATE, END_DATE, INITIAL_CASH)

            logger.info("")
            logger.info("-" * 40)
            logger.info(f"策略收益率: {result['strategy_return']:.2f}%")
            logger.info(f"基准收益率: {result['benchmark_return']:.2f}%")

            diff = result['strategy_return'] - result['benchmark_return']
            if diff > 0:
                logger.info(f"超额收益: +{diff:.2f}%")
            else:
                logger.info(f"超额收益: {diff:.2f}%")

            logger.info("")
            logger.info("=" * 60)
            logger.info("测试完成!")
            logger.info(f"日志已保存至: {LOG_FILE}")
            logger.info("=" * 60)

        except ImportError:
            logger.error("未安装掘金量化SDK，请先安装: pip install gm")
        except Exception as e:
            logger.error(f"测试失败: {e}")
            raise