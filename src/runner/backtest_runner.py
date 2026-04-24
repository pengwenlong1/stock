# -*- coding: utf-8 -*-
"""
回测执行器模块 (backtest_runner.py)

【功能说明】
执行历史数据回测验证。调用 strategy.py 中的交易策略类进行信号检测。
不包含策略逻辑，仅负责：数据准备、指标计算、调用策略、执行交易、生成报告。

【SAR跌破替代均线死叉】
使用SAR绿转红信号替代原有的MA5跌破MA10（均线死叉）逻辑：
- 原逻辑：MA5跌破MA10 → 卖出信号触发
- 新逻辑：SAR绿转红（SAR从价格下方转到上方）→ 卖出信号触发

【使用方法】
修改 main 函数中的参数后运行：
    python src/runner/backtest_runner.py

参数说明：
- SYMBOL: 股票代码（SHSE/SZSE前缀）
- START_DATE/END_DATE: 回测时间段
- BUY_ID: 买入策略ID (1-5)
- SELL_ID: 卖出策略ID (默认1)

作者：量化交易团队
创建日期：2024
"""
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入策略类和工具类
from src.tool.strategy import TradingStrategy, StrategyState, SellSignal, BuySignal, SellFlag
from src.tool.rsi_calculator import RSI
from src.tool.sar_strategy import SARStrategy, SARSignal  # 导入SAR策略类
from src.tool.macd_calculator import MACDCalculator
from src.tool.peak_detector import PeakDetector
from src.tool.divergence_detector import DivergenceDetector, DivergenceSignal, DivergenceType, TimeFrame

try:
    from gm.api import history, set_token, ADJUST_PREV, get_trading_dates, get_instruments
except ImportError:
    history = None
    set_token = None
    ADJUST_PREV = None
    get_trading_dates = None
    get_instruments = None

# SAR参数：SAR(10,2,20)
SAR_ACCELERATION = 0.02
SAR_MAXIMUM = 0.20


# ==================== 数据结构 ====================

@dataclass
class TradeRecord:
    """交易记录"""
    date: pd.Timestamp
    action: str                         # 'buy' or 'sell'
    amount: float                       # 交易比例
    price: float                        # 交易价格
    reason: str                         # 交易原因
    daily_rsi: float = 0.0
    weekly_rsi: float = 0.0


# ==================== 回测执行器 ====================

class BacktestRunner:
    """
    回测执行器

    【功能说明】
    调用 TradingStrategy 进行信号检测，执行模拟交易，生成回测报告。

    【职责分离】
    - 策略逻辑：由 strategy.py 的 TradingStrategy 类处理
    - 回测执行：由本类处理（数据准备、指标计算、交易执行、报告生成）
    """

    def __init__(self,
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 judge_buy_ids: List[int] = [1],
                 judge_t_ids: List[int] = [2],
                 judge_sell_ids: List[int] = [1],
                 initial_capital: float = 100000.0,
                 output_dir: Optional[str] = None) -> None:
        """
        初始化回测执行器

        Args:
            symbol: 股票代码（掘金格式，如 SHSE.512480）
            start_date: 回测开始日期（YYYY-MM-DD）
            end_date: 回测结束日期（YYYY-MM-DD）
            judge_buy_ids: 新资金买入策略ID列表（触发时买入新资金+买回卖出资金）
            judge_t_ids: 做T买回策略ID列表（触发时只买回卖出资金）
            judge_sell_ids: 卖出策略ID列表
            initial_capital: 初始资金
            output_dir: 输出目录路径（None则自动创建时间戳目录）

        买入规则说明：
        - judge_buy_ids触发: 一次性买入新资金 + 买回之前卖出的资金
        - judge_t_ids触发: 只买回之前卖出的资金，不买入新资金
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.judge_buy_ids = judge_buy_ids
        self.judge_t_ids = judge_t_ids
        self.judge_sell_ids = judge_sell_ids
        self.initial_capital = initial_capital

        # 初始化掘金token并获取股票名称
        self._init_gm_token()
        self.stock_name = self._get_stock_name(symbol)

        # 创建策略实例
        self.strategy = TradingStrategy(
            buy_ids=judge_buy_ids,
            t_ids=judge_t_ids,
            sell_ids=judge_sell_ids
        )

        # 策略状态
        self.state = StrategyState()

        # 输出目录设置
        if output_dir is None:
            logs_base = os.path.join(project_root, 'logs')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(logs_base, f'backtest_{timestamp}')
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 数据存储
        self._df: Optional[pd.DataFrame] = None
        self._close_series: Optional[pd.Series] = None
        self._high_series: Optional[pd.Series] = None
        self._low_series: Optional[pd.Series] = None

        # 指标数据
        self._daily_rsi: Optional[pd.Series] = None
        self._weekly_rsi: Optional[pd.Series] = None

        # SAR指标数据（替代MA均线）
        self._sar_strategy: Optional[SARStrategy] = None
        self._sar_series: Optional[pd.Series] = None
        self._sar_signals: List[SARSignal] = []  # 预计算的SAR转折信号

        # 创业板指数数据（用于buy_id=3和buy_id=5）
        self._index_daily_rsi: Optional[pd.Series] = None
        self._index_symbol = 'SZSE.399006'  # 创业板指数

        # 上证指数数据（用于buy_id=6和buy_id=7）
        self._sh_index_daily_rsi: Optional[pd.Series] = None
        self._sh_index_symbol = 'SHSE.000001'  # 上证指数

        # 背离检测器
        self._divergence_detector: Optional[DivergenceDetector] = None
        self._daily_divergences: List[DivergenceSignal] = []
        self._weekly_divergences: List[DivergenceSignal] = []

        # 已处理的背离信号ID集合（避免重复触发）
        self._processed_daily_divergences: Set[str] = set()
        self._processed_weekly_divergences: Set[str] = set()

        # 持仓状态（使用金额追踪）
        self._shares: float = 0.0         # 持有股票数量
        self._cash: float = initial_capital  # 剩余新资金（金额）
        self._sold_cash: float = 0.0      # 卖出后待买回的资金（金额）

        # 交易记录
        self._trade_records: List[TradeRecord] = []

        # 日志配置
        self._setup_logging()

    def _init_gm_token(self) -> None:
        """初始化掘金token"""
        config_path = os.path.join(project_root, 'config', 'settings.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                gm_token = config.get('gm_token')
                if gm_token and set_token is not None:
                    set_token(gm_token)

    def _get_stock_name(self, symbol: str) -> str:
        """
        获取股票/ETF的中文名称

        Args:
            symbol: 股票代码（掘金格式，如 SHSE.512480）

        Returns:
            股票名称
        """
        try:
            if get_instruments is None:
                return "未知"

            instruments = get_instruments(symbol)
            if instruments:
                ins = instruments[0]
                return getattr(ins, 'sec_name', '未知')
            return "未知"
        except Exception:
            return "未知"

    def _setup_logging(self) -> None:
        """配置日志"""
        symbol_clean = self.symbol.replace('.', '_')
        log_file = os.path.join(self.output_dir, f'backtest_{symbol_clean}.log')

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

    def prepare_data(self) -> bool:
        """
        准备历史数据

        Returns:
            bool: 数据准备是否成功
        """
        if history is None:
            self.logger.error("未安装掘金量化SDK，请先安装: pip install gm")
            return False

        # 设置token（确保已初始化）
        if set_token is None:
            self.logger.error("掘金SDK未正确导入")
            return False

        self.logger.info("掘金量化 SDK 初始化成功")

        # 获取预热数据起始日期（提前1年）
        warmup_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        self.logger.info(f"获取历史数据: {self.symbol}")
        self.logger.info(f"  回测时间段: {self.start_date} ~ {self.end_date}")
        self.logger.info(f"  预热数据起始: {warmup_start}")

        # 获取日线数据
        daily_data = history(
            symbol=self.symbol,
            frequency='1d',
            start_time=warmup_start + ' 09:00:00',
            end_time=self.end_date + ' 15:30:00',
            fields='eob,open,high,low,close,volume',
            df=True,
            adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
        )

        if daily_data is None or daily_data.empty:
            self.logger.error(f"获取数据失败: {self.symbol}")
            return False

        self.logger.info(f"获取到 {len(daily_data)} 条日线数据")

        # 构建价格序列
        daily_data['eob'] = pd.to_datetime(daily_data['eob'])
        self._df = daily_data.set_index('eob').sort_index()

        self._close_series = pd.Series(
            self._df['close'].values,
            index=self._df.index,
            name='Close'
        )
        self._high_series = pd.Series(
            self._df['high'].values,
            index=self._df.index,
            name='High'
        )
        self._low_series = pd.Series(
            self._df['low'].values,
            index=self._df.index,
            name='Low'
        )

        # 计算指标
        self._calculate_indicators()

        # 初始化背离检测器
        self._init_divergence_detector()

        return True

    def _calculate_indicators(self) -> None:
        """计算所有技术指标"""
        self.logger.info("计算技术指标...")

        # 计算日线RSI(6)
        rsi_daily = RSI(period=6, freq='daily')
        self._daily_rsi = rsi_daily.calculate(self._close_series)
        self.logger.info(f"日线RSI计算完成: {self._daily_rsi.notna().sum()} 个有效值")

        # 计算周线RSI(6)
        rsi_weekly = RSI(period=6, freq='weekly')
        self._weekly_rsi = rsi_weekly.calculate(self._close_series)
        self.logger.info(f"周线RSI计算完成: {self._weekly_rsi.notna().sum()} 个有效值")

        # 计算SAR指标（替代MA均线）
        self._sar_strategy = SARStrategy(
            acceleration=SAR_ACCELERATION,
            maximum=SAR_MAXIMUM
        )
        self._sar_strategy.prepare_data(self._high_series, self._low_series, self._close_series)
        self._sar_series = self._sar_strategy._sar_series
        self.logger.info(f"SAR计算完成: SAR(10,2,20)")

        # 预计算回测时间段内的SAR转折信号
        sar_signals = self._sar_strategy.detect_signals(self.start_date, self.end_date)
        self._sar_signals = sar_signals
        self.logger.info(f"SAR转折信号预计算完成: {len(sar_signals)} 个信号")
        for s in sar_signals:
            self.logger.info(f"  [{s.date.strftime('%Y-%m-%d')}] {s.signal_type}: SAR={s.sar_value:.3f}")

        # 计算创业板指数RSI（用于buy_id=3和buy_id=5）
        self._calculate_index_rsi()

    def _calculate_index_rsi(self) -> None:
        """计算指数的日线RSI（创业板指数和上证指数）"""
        warmup_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        # 检查是否需要创业板指数（buy_id=3或5）
        need_cyb_index = any(id in [3, 5] for id in self.judge_buy_ids + self.judge_t_ids)
        if need_cyb_index:
            self._calculate_single_index_rsi(
                symbol=self._index_symbol,
                index_name='创业板指数',
                rsi_attr='_index_daily_rsi',
                warmup_start=warmup_start
            )
        else:
            self.logger.info("不需要创业板指数数据（无buy_id=3或5）")

        # 检查是否需要上证指数（buy_id=6或7）
        need_sh_index = any(id in [6, 7] for id in self.judge_buy_ids + self.judge_t_ids)
        if need_sh_index:
            self._calculate_single_index_rsi(
                symbol=self._sh_index_symbol,
                index_name='上证指数',
                rsi_attr='_sh_index_daily_rsi',
                warmup_start=warmup_start
            )
        else:
            self.logger.info("不需要上证指数数据（无buy_id=6或7）")

    def _calculate_single_index_rsi(self, symbol: str, index_name: str, rsi_attr: str, warmup_start: str) -> None:
        """
        计算单个指数的日线RSI

        Args:
            symbol: 指数代码
            index_name: 指数名称（用于日志）
            rsi_attr: RSI存储属性名
            warmup_start: 预热数据起始日期
        """
        self.logger.info(f"获取{index_name}数据: {symbol}")

        try:
            index_data = history(
                symbol=symbol,
                frequency='1d',
                start_time=warmup_start + ' 09:00:00',
                end_time=self.end_date + ' 15:30:00',
                fields='eob,close',
                df=True,
                adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
            )

            if index_data is None or index_data.empty:
                self.logger.warning(f"获取{index_name}数据失败")
                return

            self.logger.info(f"获取到 {len(index_data)} 条{index_name}数据")

            index_data['eob'] = pd.to_datetime(index_data['eob'])
            index_close = pd.Series(
                index_data['close'].values,
                index=index_data['eob'],
                name=f'{index_name}_Close'
            )

            # 计算日线RSI(6)
            rsi_calculator = RSI(period=6, freq='daily')
            rsi_series = rsi_calculator.calculate(index_close)
            setattr(self, rsi_attr, rsi_series)
            self.logger.info(f"{index_name}RSI计算完成: {rsi_series.notna().sum()} 个有效值")

        except Exception as e:
            self.logger.warning(f"计算{index_name}RSI失败: {e}")

    def _init_divergence_detector(self) -> None:
        """初始化背离检测器"""
        self.logger.info("初始化背离检测器...")

        warmup_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        self._divergence_detector = DivergenceDetector()
        self._divergence_detector.prepare_data(
            symbol=self.symbol,
            start_date=warmup_start,
            end_date=self.end_date
        )

        # 检测所有背离
        divergences = self._divergence_detector.detect_all_divergences()
        self._daily_divergences = divergences.get('daily_top_confirmed', [])
        self._weekly_divergences = divergences.get('weekly_top_confirmed', [])

        self.logger.info(f"日线顶背离生效: {len(self._daily_divergences)} 个")
        self.logger.info(f"周线顶背离生效: {len(self._weekly_divergences)} 个")

    def _get_trading_dates(self) -> List[pd.Timestamp]:
        """获取回测时间段内的交易日列表"""
        # 使用字符串匹配避免时区问题
        start_str = self.start_date
        end_str = self.end_date

        return self._close_series[
            (self._close_series.index.strftime('%Y-%m-%d') >= start_str) &
            (self._close_series.index.strftime('%Y-%m-%d') <= end_str)
        ].index.tolist()

    def _get_value_at_date(self, date: pd.Timestamp, series: pd.Series) -> float:
        """获取指定日期的指标值"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = series[series.index.strftime('%Y-%m-%d') == date_str]
            if len(matching) > 0:
                return matching.iloc[0]
            return np.nan
        except Exception:
            return np.nan

    def _get_index_rsi_at_date(self, date: pd.Timestamp) -> float:
        """获取指定日期的创业板指数RSI值"""
        if self._index_daily_rsi is None:
            return np.nan
        return self._get_value_at_date(date, self._index_daily_rsi)

    def _get_sh_index_rsi_at_date(self, date: pd.Timestamp) -> float:
        """获取指定日期的上证指数RSI值"""
        if self._sh_index_daily_rsi is None:
            return np.nan
        return self._get_value_at_date(date, self._sh_index_daily_rsi)

    def _detect_sar_cross_down(self, date: pd.Timestamp) -> bool:
        """
        检测SAR是否发生绿转红（跌破信号）

        【逻辑说明】
        SAR绿转红 = SAR从价格下方转到上方，趋势从多头转为空头
        检测条件：SAR(t) > Close(t) 且 SAR(t-1) <= Close(t-1)

        替代原有的MA5跌破MA10（均线死叉）逻辑

        Args:
            date: 当前日期

        Returns:
            bool: 是否发生SAR绿转红（跌破）
        """
        # 检查预计算的SAR信号中是否有该日期的绿转红信号
        for signal in self._sar_signals:
            if signal.signal_type == '绿转红' and signal.date.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'):
                return True
        return False

    def _get_divergence_info(self, date: pd.Timestamp, timeframe: str) -> Optional[Dict]:
        """获取指定日期的背离信息（跳过已处理的信号）"""
        divergences = self._daily_divergences if timeframe == 'daily' else self._weekly_divergences
        processed_set = self._processed_daily_divergences if timeframe == 'daily' else self._processed_weekly_divergences

        for div in divergences:
            if div.confirmation_date is not None:
                if div.confirmation_date.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'):
                    # 生成唯一ID（使用形成日期）
                    div_id = div.date.strftime('%Y-%m-%d')

                    # 跳过已处理的背离信号
                    if div_id in processed_set:
                        continue

                    # 标记为已处理
                    processed_set.add(div_id)

                    return {
                        'date': div.date,
                        'prev_high': div.peak_a_price,
                        'curr_high': div.peak_b_price,
                        'prev_macd': div.peak_a_macd,
                        'curr_macd': div.peak_b_macd
                    }

        return None

    def _execute_sell(self, date: pd.Timestamp, signal: SellSignal) -> None:
        """
        执行卖出操作

        使用金额追踪：卖出股票转为现金
        """
        price = self._get_value_at_date(date, self._close_series)
        if np.isnan(price) or self._shares <= 0:
            return

        # 计算卖出股票数量比例
        if signal.flag == SellFlag.CLEAR_ALL:
            sell_ratio = 1.0
        elif signal.flag == SellFlag.SELL_HALF:
            sell_ratio = 0.5
        else:
            sell_ratio = 1.0 / 3.0

        # 计算实际卖出股票数量
        sell_shares = self._shares * sell_ratio

        # 计算卖出金额
        sell_amount = sell_shares * price

        # 更新持仓
        self._shares -= sell_shares
        self._sold_cash += sell_amount  # 卖出金额进入做T资金池

        # 调用策略重置状态
        self.strategy.reset_after_sell(self.state, signal.flag, date)

        # 记录交易
        trade = TradeRecord(
            date=date,
            action='sell',
            amount=sell_shares,
            price=price,
            reason=signal.reason,
            daily_rsi=signal.daily_rsi,
            weekly_rsi=signal.weekly_rsi
        )
        self._trade_records.append(trade)

        # 输出日志
        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 卖出: {sell_shares:.0f}股 @ {price:.3f} | "
                        f"金额:{sell_amount:.2f} | 日RSI:{signal.daily_rsi:.2f} | 周RSI:{signal.weekly_rsi:.2f} | "
                        f"{signal.reason}")

    def _execute_fake_sell(self, date: pd.Timestamp, signal: SellSignal) -> None:
        """
        执行假卖出操作（无持仓时重置状态）

        当均线跌破触发卖出信号，但当前无持仓时，
        仍需重置信号状态，避免信号继续生效影响后续交易。

        Args:
            date: 交易日期
            signal: 卖出信号
        """
        # 调用策略重置状态（不实际交易）
        self.strategy.reset_after_sell(self.state, signal.flag, date)

        # 记录假卖出日志（不加入交易记录）
        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 假卖出（无持仓）: {signal.reason}")
        self.logger.info(f"  状态重置: rsi_flag=0, daily_divergence_flag=0, weekly_divergence_flag=0")

    def _execute_buy(self, date: pd.Timestamp, signal: BuySignal) -> None:
        """
        执行买入操作

        使用金额追踪：用现金买入股票
        买入规则：
        - is_new_cash=True (judge_buy_ids触发): 买入新资金 + 买回卖出资金
        - is_new_cash=False (judge_t_ids触发): 只买回卖出资金
        """
        price = self._get_value_at_date(date, self._close_series)
        if np.isnan(price):
            return

        # 计算买入金额
        if signal.is_new_cash:
            # judge_buy_ids触发：买入新资金 + 买回卖出资金
            buy_amount = self._cash + self._sold_cash
            self._cash = 0.0
            self._sold_cash = 0.0
        else:
            # judge_t_ids触发：只买回卖出资金
            buy_amount = self._sold_cash
            self._sold_cash = 0.0

        if buy_amount <= 0:
            return

        # 计算买入股票数量
        buy_shares = buy_amount / price

        # 更新持仓
        self._shares += buy_shares

        # 记录交易
        trade = TradeRecord(
            date=date,
            action='buy',
            amount=buy_shares,
            price=price,
            reason=signal.reason,
            daily_rsi=signal.daily_rsi,
            weekly_rsi=signal.weekly_rsi
        )
        self._trade_records.append(trade)

        # 日志输出
        buy_type = "新资金+买回" if signal.is_new_cash else "做T买回"
        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 买入({buy_type}): {int(buy_shares)}股 @ {price:.3f} | "
                        f"金额:{buy_amount:.2f} | 日RSI:{signal.daily_rsi:.2f} | 周RSI:{signal.weekly_rsi:.2f} | {signal.reason}")

    def run_backtest(self) -> Dict:
        """
        执行回测

        Returns:
            回测结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("开始回测")
        self.logger.info("=" * 60)
        self.logger.info(f"股票: {self.symbol} ({self.stock_name})")
        self.logger.info(f"时间段: {self.start_date} ~ {self.end_date}")
        self.logger.info(f"策略: {self.strategy}")
        self.logger.info("=" * 60)

        # 准备数据
        if not self.prepare_data():
            self.logger.error("数据准备失败，回测终止")
            return {'error': '数据准备失败'}

        # 获取交易日列表
        trading_dates = self._get_trading_dates()
        self.logger.info(f"回测交易日数: {len(trading_dates)}")

        # 逐日遍历
        for date in trading_dates:
            # 获取当日指标值
            daily_rsi = self._get_value_at_date(date, self._daily_rsi)
            weekly_rsi = self._get_value_at_date(date, self._weekly_rsi)
            sar_cross_down = self._detect_sar_cross_down(date)  # SAR绿转红替代MA死叉

            # 更新RSI警戒级别
            self.strategy.update_rsi_flag(weekly_rsi, self.state, date)

            # 检查并设置背离状态
            daily_div_info = self._get_divergence_info(date, 'daily')
            weekly_div_info = self._get_divergence_info(date, 'weekly')

            if daily_div_info is not None:
                self.strategy.set_daily_divergence(self.state, daily_div_info)
            if weekly_div_info is not None:
                self.strategy.set_weekly_divergence(self.state, weekly_div_info)

            # 调用策略检测卖出信号
            sell_signal = self.strategy.check_sell_signal(
                date=date,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                ma_cross_down=sar_cross_down,  # SAR跌破传入策略（参数名保持兼容）
                state=self.state,
                position=self._shares * self._get_value_at_date(date, self._close_series) / self.initial_capital if self._shares > 0 else 0
            )

            # 处理卖出信号
            if sell_signal is not None:
                if self._shares > 0:
                    # 有持仓：执行实际卖出
                    self._execute_sell(date, sell_signal)
                    continue  # 卖出后当天不买入
                else:
                    # 无持仓：执行假卖出（只重置状态，不实际交易）
                    self._execute_fake_sell(date, sell_signal)
                    continue  # 假卖出后当天不买入

            # 获取创业板指数RSI值
            index_daily_rsi = self._get_index_rsi_at_date(date)

            # 获取上证指数RSI值
            sh_index_daily_rsi = self._get_sh_index_rsi_at_date(date)

            # 调用策略检测买入信号
            buy_signal = self.strategy.check_buy_signal(
                date=date,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                state=self.state,
                has_new_cash=self._cash > 0,
                has_sold_cash=self._sold_cash > 0,
                index_daily_rsi=index_daily_rsi,
                sh_index_daily_rsi=sh_index_daily_rsi
            )

            if buy_signal is not None and buy_signal.triggered:
                self._execute_buy(date, buy_signal)

        # 计算回测结果
        results = self._calculate_results()

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("回测完成")
        self.logger.info("=" * 60)

        return results

    def _calculate_results(self) -> Dict:
        """计算回测结果"""
        if len(self._trade_records) == 0:
            return {
                'total_return': 0.0,
                'trades': 0,
                'buy_count': 0,
                'sell_count': 0,
                'max_drawdown': 0.0,
                'excess_return': 0.0
            }

        # 计算最终市值
        last_price = self._close_series.iloc[-1]
        final_value = self._shares * last_price + self._cash + self._sold_cash
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # 计算基准收益率（买入持有策略）
        first_price = self._get_first_backtest_price()
        benchmark_return = (last_price / first_price - 1.0) * 100 if first_price > 0 else 0.0

        # 计算超额收益率
        excess_return = total_return - benchmark_return

        # 计算最大回撤率
        max_drawdown = self._calculate_max_drawdown()

        # 统计交易次数
        buy_count = sum(1 for t in self._trade_records if t.action == 'buy')
        sell_count = sum(1 for t in self._trade_records if t.action == 'sell')

        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'max_drawdown': max_drawdown,
            'final_shares': self._shares,
            'final_cash': self._cash,
            'final_sold_cash': self._sold_cash,
            'trades': len(self._trade_records),
            'buy_count': buy_count,
            'sell_count': sell_count
        }

        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'max_drawdown': max_drawdown,
            'final_shares': self._shares,
            'final_cash': self._cash,
            'final_sold_cash': self._sold_cash,
            'trades': len(self._trade_records),
            'buy_count': buy_count,
            'sell_count': sell_count
        }

    def _get_first_backtest_price(self) -> float:
        """获取回测开始日的价格"""
        start_str = self.start_date
        matching = self._close_series[
            self._close_series.index.strftime('%Y-%m-%d') >= start_str
        ]
        if len(matching) > 0:
            return matching.iloc[0]
        return 0.0

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤率"""
        if len(self._trade_records) == 0:
            return 0.0

        # 构建每日净值序列
        trading_dates = self._get_trading_dates()
        if len(trading_dates) == 0:
            return 0.0

        # 计算每日市值
        daily_values = []
        for date in trading_dates:
            price = self._get_value_at_date(date, self._close_series)
            if np.isnan(price):
                continue

            # 根据交易记录计算当日持仓状态
            shares = 0.0
            cash = self.initial_capital
            sold_cash = 0.0
            for trade in self._trade_records:
                if trade.date <= date:
                    if trade.action == 'buy':
                        shares += trade.amount
                        cash -= trade.amount * trade.price if trade.amount * trade.price <= cash else 0
                        sold_cash -= trade.amount * trade.price if trade.amount * trade.price <= sold_cash else sold_cash
                    else:
                        shares -= trade.amount
                        sold_cash += trade.amount * trade.price

            value = shares * price + cash + sold_cash
            daily_values.append((date, value))

        if len(daily_values) < 2:
            return 0.0

        # 计算最大回撤
        max_value = daily_values[0][1]
        max_drawdown = 0.0

        for date, value in daily_values:
            if value > max_value:
                max_value = value
            drawdown = (max_value - value) / max_value * 100 if max_value > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def generate_report(self) -> Dict:
        """
        生成回测报告

        Returns:
            Dict: 包含所有报告数据的字典，按指定列顺序排列
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("回测报告")
        self.logger.info("=" * 60)

        # 交易明细
        self.logger.info("")
        self.logger.info("【交易明细】")
        self.logger.info("-" * 80)

        for trade in self._trade_records:
            action = "买入" if trade.action == 'buy' else "卖出"
            self.logger.info(f"[{trade.date.strftime('%Y-%m-%d')}] {action}: "
                           f"股数={int(trade.amount)} | 价格={trade.price:.3f} | "
                           f"日RSI={trade.daily_rsi:.2f} | 周RSI={trade.weekly_rsi:.2f}")
            self.logger.info(f"  原因: {trade.reason}")

        # 统计汇总
        results = self._calculate_results()

        self.logger.info("")
        self.logger.info("【统计汇总】")
        self.logger.info("-" * 40)
        self.logger.info(f"策略收益率: {results['total_return']:.2f}%")
        self.logger.info(f"基准收益率: {results['benchmark_return']:.2f}%")
        self.logger.info(f"超额收益率: {results['excess_return']:.2f}%")
        self.logger.info(f"最大回撤率: {results['max_drawdown']:.2f}%")
        self.logger.info(f"交易次数: {results['trades']} 次")
        self.logger.info(f"  买入: {results['buy_count']} 次")
        self.logger.info(f"  卖出: {results['sell_count']} 次")
        self.logger.info(f"最终持仓股数: {int(results['final_shares'])}股")
        self.logger.info(f"剩余新资金: {results['final_cash']:.2f}元")
        self.logger.info(f"待买回资金: {results['final_sold_cash']:.2f}元")

        self.logger.info("")
        self.logger.info(f"日志文件: {self.log_file}")

        # 绘制图表
        chart_path = self.plot_backtest_results()
        if chart_path:
            self.logger.info(f"图表文件: {chart_path}")

        self.logger.info("=" * 60)

        # 计算最终市值
        last_price = self._close_series.iloc[-1] if self._close_series is not None else 0.0
        final_value = results['final_shares'] * last_price + results['final_cash'] + results['final_sold_cash']

        # 按指定列顺序构建报告数据
        report_data = {
            '股票代码': self.symbol,
            '策略收益率(%)': f"{results['total_return']:.2f}",
            '基准收益率(%)': f"{results['benchmark_return']:.2f}",
            '超额收益率(%)': f"{results['excess_return']:.2f}",
            '最大回撤率(%)': f"{results['max_drawdown']:.2f}",
            '买入策略ID': str(self.judge_buy_ids),
            '做T策略ID': str(self.judge_t_ids),
            '卖出策略ID': str(self.judge_sell_ids),
            '股票名称': self.stock_name,
            '开始日期': self.start_date,
            '结束日期': self.end_date,
            '初始资金': f"{self.initial_capital:.2f}",
            '交易次数': results['trades'],
            '买入次数': results['buy_count'],
            '卖出次数': results['sell_count'],
            '最终市值': f"{final_value:.2f}",
            '是否成功': '是',
            '错误信息': '',
            '日志文件': self.log_file,
            '图表文件': chart_path or ''
        }

        # 输出CSV格式汇总（一行）
        self.logger.info("")
        self.logger.info("【CSV格式汇总】")
        header = "股票代码,策略收益率(%),基准收益率(%),超额收益率(%),最大回撤率(%),买入策略ID,做T策略ID,卖出策略ID,股票名称,开始日期,结束日期,初始资金,交易次数,买入次数,卖出次数,最终市值,是否成功,错误信息,日志文件,图表文件"
        self.logger.info(header)
        row = f"{self.symbol},{results['total_return']:.2f},{results['benchmark_return']:.2f},{results['excess_return']:.2f},{results['max_drawdown']:.2f},{self.judge_buy_ids},{self.judge_t_ids},{self.judge_sell_ids},{self.stock_name},{self.start_date},{self.end_date},{self.initial_capital:.2f},{results['trades']},{results['buy_count']},{results['sell_count']},{final_value:.2f},是,,{self.log_file},{chart_path or ''}"
        self.logger.info(row)

        return report_data

    def plot_backtest_results(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        绘制回测结果图表（买入卖出点标注）

        Args:
            output_path: 输出路径

        Returns:
            图表保存路径
        """
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，避免线程问题
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        if self._close_series is None or len(self._trade_records) == 0:
            self.logger.warning("无数据，无法绘制图表")
            return None

        # 获取回测时间段的数据
        start_str = self.start_date
        end_str = self.end_date

        close_data = self._close_series[
            (self._close_series.index.strftime('%Y-%m-%d') >= start_str) &
            (self._close_series.index.strftime('%Y-%m-%d') <= end_str)
        ]
        sar_data = self._sar_series[
            (self._sar_series.index.strftime('%Y-%m-%d') >= start_str) &
            (self._sar_series.index.strftime('%Y-%m-%d') <= end_str)
        ]
        daily_rsi_data = self._daily_rsi[
            (self._daily_rsi.index.strftime('%Y-%m-%d') >= start_str) &
            (self._daily_rsi.index.strftime('%Y-%m-%d') <= end_str)
        ]
        weekly_rsi_data = self._weekly_rsi[
            (self._weekly_rsi.index.strftime('%Y-%m-%d') >= start_str) &
            (self._weekly_rsi.index.strftime('%Y-%m-%d') <= end_str)
        ]

        if len(close_data) == 0:
            return None

        # 创建图表（3个子图）
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'{self.symbol} ({self.stock_name}) 回测结果 ({self.start_date} ~ {self.end_date})\n'
                    f'策略: {self.strategy} | SAR参数: SAR(10,2,20)',
                    fontsize=12, fontweight='bold')

        ax_price = axes[0]
        ax_rsi = axes[1]
        ax_position = axes[2]

        dates = close_data.index

        # ===== 绘制价格曲线 + SAR点 =====
        ax_price.plot(dates, close_data.values, 'b-', linewidth=1.5, label='收盘价', alpha=0.8)

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

        # 标记SAR转折信号
        for signal in self._sar_signals:
            if signal.signal_type == '绿转红':
                ax_price.scatter(signal.date, signal.close_price, color='red',
                                marker='x', s=100, zorder=5, label='SAR跌破' if signal == self._sar_signals[0] else '')
            elif signal.signal_type == '红转绿':
                ax_price.scatter(signal.date, signal.close_price, color='green',
                                marker='o', s=100, zorder=5, label='SAR突破' if signal == self._sar_signals[0] else '')

        # 标记买入点（绿色向上箭头）
        buy_trades = [t for t in self._trade_records if t.action == 'buy']
        for trade in buy_trades:
            ax_price.scatter(trade.date, trade.price, color='green', marker='^', s=150,
                            label='买入' if trade == buy_trades[0] else '', zorder=6)
            ax_price.annotate(f'买入\n{trade.price:.3f}',
                             (trade.date, trade.price),
                             textcoords="offset points", xytext=(5, 15),
                             fontsize=8, color='green', fontweight='bold')

        # 标记卖出点（红色向下箭头）
        sell_trades = [t for t in self._trade_records if t.action == 'sell']
        for trade in sell_trades:
            ax_price.scatter(trade.date, trade.price, color='red', marker='v', s=150,
                            label='卖出' if trade == sell_trades[0] else '', zorder=6)
            ax_price.annotate(f'卖出\n{trade.price:.3f}',
                             (trade.date, trade.price),
                             textcoords="offset points", xytext=(5, -20),
                             fontsize=8, color='red', fontweight='bold')

        ax_price.set_ylabel('价格')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # ===== 绘制RSI曲线 =====
        ax_rsi.plot(dates, daily_rsi_data.values, 'b-', linewidth=1, label='日线RSI(6)', alpha=0.7)
        ax_rsi.plot(dates, weekly_rsi_data.values, 'r-', linewidth=1.5, label='周线RSI(6)', alpha=0.7)

        # RSI阈值线
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='超卖线(30)')
        ax_rsi.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='超买线(70)')
        ax_rsi.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='警戒线(80)')
        ax_rsi.axhline(y=85, color='darkred', linestyle='--', alpha=0.5, label='二级警戒(85)')
        ax_rsi.axhline(y=90, color='purple', linestyle='--', alpha=0.5, label='清仓线(90)')

        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(loc='upper left')
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # ===== 绘制持仓比例曲线 =====
        # 计算每日持仓比例
        position_values = []
        current_position = 0.0
        trade_idx = 0

        for date in dates:
            # 更新持仓状态
            while trade_idx < len(self._trade_records) and self._trade_records[trade_idx].date <= date:
                trade = self._trade_records[trade_idx]
                if trade.action == 'buy':
                    current_position += trade.amount
                else:
                    current_position -= trade.amount
                trade_idx += 1
            position_values.append(current_position)

        ax_position.fill_between(dates, 0, position_values, color='blue', alpha=0.3, label='持仓比例')
        ax_position.plot(dates, position_values, 'b-', linewidth=1.5, alpha=0.8)

        ax_position.set_ylabel('持仓比例')
        ax_position.set_ylim(0, 1.1)
        ax_position.legend(loc='upper left')
        ax_position.grid(True, alpha=0.3)
        ax_position.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_position.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图表
        if output_path is None:
            symbol_clean = self.symbol.replace('.', '_')
            output_path = os.path.join(self.output_dir, f'backtest_chart_{symbol_clean}.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"回测图表已保存: {output_path}")
        return output_path


# ==================== 主函数 ====================

def _add_exchange_prefix(code: str) -> str:
    """
    自动添加交易所前缀

    Args:
        code: 股票代码（纯数字，如 512480）

    Returns:
        str: 完整股票代码（如 SHSE.512480）
    """
    code = code.strip()
    if code.startswith('SHSE.') or code.startswith('SZSE.'):
        return code

    # 6开头: 上海主板
    # 5开头: 上海基金/ETF
    # 0开头: 深圳主板
    # 3开头: 创业板
    # 1开头: 深圳基金
    if code.startswith('6') or code.startswith('5'):
        return f'SHSE.{code}'
    elif code.startswith('0') or code.startswith('3') or code.startswith('1'):
        return f'SZSE.{code}'
    else:
        return f'SZSE.{code}'


if __name__ == "__main__":
    # ===== 可配置参数 =====
    # 股票代码只需输入数字，脚本自动添加交易所前缀
    SYMBOL = '512480'                   # 半导体ETF -> SHSE.512480
    START_DATE = '2024-01-01'           # 回测开始日期
    END_DATE = '2026-04-10'             # 回测结束日期

    '''
    ### `buy_id` = 1：保守型买入
    - **条件**：日线 RSI < 20 **并且** 周线 RSI < 25。
    
    ### `buy_id` = 2：标准型买入 / 做 T 买回
    - **说明**：此 ID 常用于判断卖出后的资金买回时机。
    - **条件**：日线 RSI < 25 **并且** 周线 RSI < 30。
    
    ### `buy_id` = 3：指数保护型买入
    - **说明**：此 ID 用于监控市场大盘是否处于超卖状态。
    - **条件**：**创业板指数 (399006)** 的日线 RSI < 25。
    
    ### `buy_id` = 4：极度保守型买入
    - **条件**：日线 RSI < 20 **并且** 周线 RSI < 20。
    
    ### `buy_id` = 5：指数保护型买入
    - **说明**：此 ID 用于监控市场大盘是否处于超卖状态。
    - **条件**：**创业板指数 (399006)** 的日线 RSI < 20。
    '''
    # ===== 策略ID配置 =====
    JUDGE_BUY_IDS = [1,5]                 # 新资金买入策略ID列表 (1-5)
    JUDGE_T_IDS = []                   # 做T买回策略ID列表 (1-5)
    JUDGE_SELL_IDS = [1]                # 卖出策略ID列表 (1)

    INITIAL_CAPITAL = 100000            # 初始资金

    # ===== 执行回测 =====
    # 自动添加交易所前缀
    SYMBOL_FULL = _add_exchange_prefix(SYMBOL)

    runner = BacktestRunner(
        symbol=SYMBOL_FULL,
        start_date=START_DATE,
        end_date=END_DATE,
        judge_buy_ids=JUDGE_BUY_IDS,
        judge_t_ids=JUDGE_T_IDS,
        judge_sell_ids=JUDGE_SELL_IDS,
        initial_capital=INITIAL_CAPITAL
    )

    results = runner.run_backtest()
    runner.generate_report()