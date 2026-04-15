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
    logging.warning("talib未安装，将使用自定义SAR计算")

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
        计算抛物线转向指标（SAR）

        优先使用 TA-Lib 计算，这是最接近东方财富的方式。
        TA-Lib 的 SAR 默认参数：acceleration=0.02, maximum=0.20

        如果 TA-Lib 未安装，回退到自定义计算方法。

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            pd.Series: SAR序列
        """
        if talib is None:
            logger.warning("TA-Lib 未安装，回退到自定义计算")
            return self._calculate_sar_eastmoney_fixed(high, low, close)

        # TA-Lib 要求输入为 numpy array
        high_np = high.values
        low_np = low.values

        # TA-Lib SAR 参数：acceleration 和 maximum
        sar_array = talib.SAR(high_np, low_np, acceleration=self.acceleration, maximum=self.maximum)

        return pd.Series(sar_array, index=high.index, name='SAR')

    def _calculate_sar_eastmoney_fixed(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        修正版：严格对齐东方财富 SAR(10,2,20) 显示值
        """
        n_total = len(close)
        if n_total == 0:
            return pd.Series([], dtype=float)

        af_step = self.acceleration  # 0.02
        af_max = self.maximum        # 0.20
        n_init = 10                  # 初始周期

        sar = np.full(n_total, np.nan)

        if n_total <= n_init:
            # 数据不足，用简单初始化
            sar[0] = low.iloc[0]
            is_long = True
            ep = high.iloc[0]
            af = af_step
            start_idx = 1
        else:
            # 初始化：用前n_init日判断趋势
            if close.iloc[n_init - 1] > close.iloc[0]:
                is_long = True
                ep = high.iloc[:n_init].max()
                sar[n_init - 1] = low.iloc[:n_init].min()
            else:
                is_long = False
                ep = low.iloc[:n_init].min()
                sar[n_init - 1] = high.iloc[:n_init].max()
            af = af_step
            sar[:n_init - 1] = sar[n_init - 1]
            start_idx = n_init

        # 主循环
        for i in range(start_idx, n_total):
            # 计算原始 SAR
            raw_sar = sar[i - 1] + af * (ep - sar[i - 1])

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
                    sar[i] = ep
                    new_ep = low.iloc[i]      # 新 EP 为空头最低
                    new_af = af_step
                else:
                    sar[i] = raw_sar
                    if high.iloc[i] > ep:
                        new_ep = high.iloc[i]
                        new_af = min(af + af_step, af_max)
            else:
                # 空头：检查是否突破 SAR（用最高价）
                if high.iloc[i] >= raw_sar:
                    reverse = True
                    new_is_long = True
                    # 反转 SAR = 上一空头趋势的 EP（最低价）
                    sar[i] = ep
                    new_ep = high.iloc[i]     # 新 EP 为多头最高
                    new_af = af_step
                else:
                    sar[i] = raw_sar
                    if low.iloc[i] < ep:
                        new_ep = low.iloc[i]
                        new_af = min(af + af_step, af_max)

            # 价格区间限制（东方财富风格）
            if not reverse:
                if new_is_long:
                    # 多头：SAR 不能高于前两日最低价
                    sar[i] = min(sar[i], low.iloc[i - 1], low.iloc[i])
                else:
                    # 空头：SAR 不能低于前两日最高价
                    sar[i] = max(sar[i], high.iloc[i - 1], high.iloc[i])

            is_long = new_is_long
            ep = new_ep
            af = new_af

        return pd.Series(sar, index=close.index, name='SAR')

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

        【反转判断逻辑】
        使用High/Low判断趋势和反转，比使用Close更准确：
        - 多头趋势：SAR < Low（SAR在价格下方，作为支撑）
        - 空头趋势：SAR > High（SAR在价格上方，作为阻力）
        - 空头转多头（红转绿）：前一日空头(SAR > High) + 当日High >= SAR（突破上方阻力）
        - 多头转空头（绿转红）：前一日多头(SAR < Low) + 当日Low <= SAR（跌破下方支撑）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）

        Returns:
            List[SARSignal]: 信号列表
        """
        if self._sar_series is None or self._close_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        if self._high_series is None or self._low_series is None:
            raise ValueError("请先调用prepare_data()准备数据")

        # 使用字符串筛选时间段（避免时区问题）
        sar_data = self._sar_series[
            (self._sar_series.index.strftime('%Y-%m-%d') >= start_date) &
            (self._sar_series.index.strftime('%Y-%m-%d') <= end_date)
        ]
        high_data = self._high_series[
            (self._high_series.index.strftime('%Y-%m-%d') >= start_date) &
            (self._high_series.index.strftime('%Y-%m-%d') <= end_date)
        ]
        low_data = self._low_series[
            (self._low_series.index.strftime('%Y-%m-%d') >= start_date) &
            (self._low_series.index.strftime('%Y-%m-%d') <= end_date)
        ]
        close_data = self._close_series[
            (self._close_series.index.strftime('%Y-%m-%d') >= start_date) &
            (self._close_series.index.strftime('%Y-%m-%d') <= end_date)
        ]

        if len(sar_data) < 2:
            return []

        signals = []

        # 遍历检测转折点
        dates = sar_data.index
        for i in range(1, len(dates)):
            curr_date = dates[i]
            prev_date = dates[i-1]

            curr_sar = sar_data.loc[curr_date]
            prev_sar = sar_data.loc[prev_date]
            curr_high = high_data.loc[curr_date]
            curr_low = low_data.loc[curr_date]
            prev_high = high_data.loc[prev_date]
            prev_low = low_data.loc[prev_date]
            curr_close = close_data.loc[curr_date]

            # 判断前一日趋势（基于 SAR 与价格关系）
            prev_trend_long = prev_sar < prev_low   # SAR < Low → 多头
            prev_trend_short = prev_sar > prev_high  # SAR > High → 空头

            # 判断当日趋势
            curr_trend_long = curr_sar < curr_low
            curr_trend_short = curr_sar > curr_high

            # 检测空头转多头（红转绿）：前一日空头 + 当日High >= SAR
            if prev_trend_short and (curr_high >= curr_sar):
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
                logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 红转绿信号: SAR={curr_sar:.3f}, "
                           f"High={curr_high:.3f}突破SAR, Close={curr_close:.3f}")

            # 检测多头转空头（绿转红）：前一日多头 + 当日Low <= SAR
            elif prev_trend_long and (curr_low <= curr_sar):
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
                logger.info(f"[{curr_date.strftime('%Y-%m-%d')}] 绿转红信号: SAR={curr_sar:.3f}, "
                           f"Low={curr_low:.3f}跌破SAR, Close={curr_close:.3f}")

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

    logger.info("=" * 60)
    logger.info("SAR策略模块测试")
    logger.info("=" * 60)

    # ==================== 测试参数配置 ====================
    TEST_TICKER = '512480'
    TEST_NAME = '半导体ETF'
    START_DATE = '2026-01-01'
    END_DATE = '2026-04-16'

    # SAR参数 SAR(10,2,20)
    # acceleration = 2% = 0.02
    # maximum = 20% = 0.20
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

        # 打印每一天的SAR值
        logger.info("")
        logger.info("=" * 60)
        logger.info("每日SAR值明细")
        logger.info("=" * 60)
        logger.info(f"日期          收盘价    SAR值     趋势状态")
        logger.info("-" * 60)

        sar_series = strategy._sar_series
        close_series_data = strategy._close_series

        for i in range(len(sar_series)):
            date = sar_series.index[i]
            sar_val = sar_series.iloc[i]
            close_val = close_series_data.iloc[i]
            if sar_val > close_val:
                trend = '空头(SAR>Close)'
            else:
                trend = '多头(SAR<Close)'
            logger.info(f"{date.strftime('%Y-%m-%d')}  {close_val:.4f}    {sar_val:.4f}    {trend}")

        # 检测信号
        logger.info("")
        logger.info("=" * 60)
        logger.info("SAR转折信号检测")
        logger.info("=" * 60)

        signals = strategy.detect_signals(START_DATE, END_DATE)

        logger.info("")
        logger.info(f"检测到 {len(signals)} 个转折信号")

        for s in signals:
            logger.info(f"[{s.date.strftime('%Y-%m-%d')}] {s.signal_type}: "
                       f"SAR={s.sar_value:.3f}, Close={s.close_price:.3f}, "
                       f"趋势={s.position}")

        # 获取当前状态
        logger.info("")
        logger.info("=" * 60)
        logger.info("当前SAR状态")
        logger.info("=" * 60)

        current_state = strategy.get_current_position(START_DATE, END_DATE)
        logger.info(f"  日期: {current_state['date'].strftime('%Y-%m-%d')}")
        logger.info(f"  SAR值: {current_state['sar']:.3f}")
        logger.info(f"  收盘价: {current_state['close']:.3f}")
        logger.info(f"  趋势: {current_state['trend']}")
        logger.info(f"  颜色: {current_state['color']}")

        # 运行回测
        logger.info("")
        logger.info("=" * 60)
        logger.info("SAR策略回测")
        logger.info("=" * 60)

        result = strategy.run_backtest(START_DATE, END_DATE, INITIAL_CASH)

        # 输出交易记录
        logger.info("")
        logger.info("交易记录:")
        for trade in result['trades']:
            logger.info(f"[{trade['date'].strftime('%Y-%m-%d')}] {trade['type']}: "
                       f"价格={trade['price']:.3f}, 股数={trade['shares']:,.2f}")

        # 输出收益对比
        logger.info("")
        logger.info("-" * 40)
        logger.info(f"策略收益率: {result['strategy_return']:.2f}%")
        logger.info(f"  最终市值: {result['final_value']:,.2f} 元")
        logger.info(f"基准收益率: {result['benchmark_return']:.2f}%")
        logger.info(f"  最终市值: {result['benchmark_value']:,.2f} 元")

        diff = result['strategy_return'] - result['benchmark_return']
        if diff > 0:
            logger.info(f"超额收益: +{diff:.2f}% （优于基准）")
        elif diff < 0:
            logger.info(f"超额收益: {diff:.2f}% （劣于基准）")
        else:
            logger.info(f"超额收益: {diff:.2f}% （与基准持平）")

        # 绘制图表
        logger.info("")
        logger.info("绘制SAR策略图表...")
        chart_path = os.path.join(LOG_DIR, f'sar_strategy_{TEST_TICKER}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        strategy.plot_sar_strategy(
            start_date=START_DATE,
            end_date=END_DATE,
            save_path=chart_path,
            stock_name=TEST_NAME,
            result=result
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