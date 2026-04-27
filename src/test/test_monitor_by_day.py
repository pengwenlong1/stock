# -*- coding: utf-8 -*-
"""
每日监控脚本测试脚本 (test_monitor_by_day.py)

【功能说明】
测试 monitor_by_day.py 的信号检测逻辑。
模拟历史时间段内的每日监控行为，确保不涉及未来值（Look-Ahead Bias）。

【测试逻辑】
对于每个交易日T：
1. 使用T日及之前的数据计算指标
2. 检测T日的买入/卖出信号
3. 记录信号触发情况

【使用方法】
修改脚本开头的测试参数后运行：
    python test/test_monitor_by_day.py

或通过命令行参数：
    python test/test_monitor_by_day.py --symbol 512480 --start 2024-01-01 --end 2024-12-31 --buy-ids 1,5 --t-ids --sell-ids 1

【重要说明】
- 严格杜绝未来函数：所有信号计算基于T日及之前的数据点
- 支持可选的钉钉告警测试（--send-dingtalk）

作者：量化交易团队
创建日期：2024
"""
import argparse
import json
import logging
import os
import sys
import time
import base64
import hashlib
import hmac
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
# 文件位于 src/test/ 目录，需要向上两层到达项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入策略类和工具类
from src.tool.strategy import TradingStrategy, StrategyState, SellSignal, BuySignal, SellFlag
from src.tool.rsi_calculator import RSI
from src.tool.ma_calculator import MACalculator
from src.tool.divergence_detector import DivergenceDetector

try:
    from gm.api import history, set_token, ADJUST_PREV, get_trading_dates, get_instruments
except ImportError:
    history = None
    set_token = None
    ADJUST_PREV = None
    get_trading_dates = None
    get_instruments = None


# ==================== 测试配置 ====================

# 创业板指数和上证指数代码
INDEX_CYB = 'SZSE.399006'
INDEX_SH = 'SHSE.000001'

# 预热天数
WARMUP_DAYS = 365


# ==================== 数据结构 ====================

@dataclass
class TestConfig:
    """测试配置"""
    symbol: str
    stock_name: str
    start_date: str
    end_date: str
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]
    send_dingtalk: bool


@dataclass
class DailyTestResult:
    """单日测试结果"""
    date: str                                           # 测试日期
    daily_rsi: float                                    # 日线RSI
    weekly_rsi: float                                   # 周线RSI
    ma5: float                                          # MA5
    ma10: float                                         # MA10
    ma_cross_down: bool                                 # 均线死叉
    index_rsi_cyb: float                                # 创业板指数RSI
    index_rsi_sh: float                                 # 上证指数RSI
    sell_signal: Optional[str]                          # 卖出信号
    buy_signal: Optional[str]                           # 买入信号
    signal_detail: Optional[str]                        # 信号详情


# ==================== 钉钉告警类 ====================

class DingTalkNotifier:
    """钉钉机器人告警类"""

    def __init__(self, webhook: str, secret: str) -> None:
        self.webhook = webhook
        self.secret = secret

    def _generate_sign(self, timestamp: int) -> str:
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return sign

    def send_markdown(self, title: str, content: str, at_all: bool = True) -> bool:
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
            return result.get('errcode') == 0
        except Exception as e:
            logging.error(f"钉钉发送异常: {e}")
            return False


# ==================== 测试器类 ====================

class MonitorTester:
    """
    每日监控测试器

    【功能说明】
    模拟历史时间段内的每日监控行为，逐日检测买卖信号。

    【关键原则 - 无未来函数】
    对于每个交易日T：
    - 指标计算：使用T日及之前的所有历史数据
    - 信号检测：基于T日的指标值判断
    - 不使用T+1或之后的数据

    【测试流程】
    1. 获取完整历史数据（包含预热期）
    2. 获取测试时间段内的交易日列表
    3. 对于每个交易日T：
       - 使用T日数据计算指标
       - 检测买入/卖出信号
       - 记录检测结果
    4. 输出测试报告
    """

    def __init__(self, config: TestConfig) -> None:
        """
        初始化测试器

        Args:
            config: 测试配置
        """
        self.config = config
        self.dingtalk = None

        # 先设置日志，再初始化设置（确保logger可用）
        self._setup_logging()
        self._init_settings()

        # 创建策略实例
        self.strategy = TradingStrategy(
            buy_ids=config.judge_buy_ids,
            t_ids=config.judge_t_ids,
            sell_ids=config.judge_sell_ids
        )

        # 数据存储
        self._df: Optional[pd.DataFrame] = None
        self._close_series: Optional[pd.Series] = None
        self._trading_dates: List[str] = []
        self._results: List[DailyTestResult] = []

        # 指标缓存（避免重复计算）
        self._daily_rsi_cache: Dict[str, float] = {}
        self._weekly_rsi_cache: Dict[str, float] = {}
        self._ma5_cache: Dict[str, float] = {}
        self._ma10_cache: Dict[str, float] = {}
        self._index_rsi_cyb_cache: Dict[str, float] = {}
        self._index_rsi_sh_cache: Dict[str, float] = {}

        # 背离信号缓存
        self._daily_divergences: List = []
        self._weekly_divergences: List = []
        self._processed_daily_divergences: set = set()
        self._processed_weekly_divergences: set = set()

    def _init_settings(self) -> None:
        """初始化设置"""
        settings_path = os.path.join(project_root, 'config', 'settings.json')
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # 初始化掘金token
            gm_token = settings.get('gm_token')
            if gm_token and set_token is not None:
                set_token(gm_token)
                self.logger.info("掘金量化 SDK 初始化成功")

            # 初始化钉钉（仅用于测试告警）
            if self.config.send_dingtalk:
                webhook = settings.get('dingtalk_webhook')
                secret = settings.get('dingtalk_secret')
                if webhook and secret:
                    self.dingtalk = DingTalkNotifier(webhook, secret)
                else:
                    self.dingtalk = None
                    self.logger.warning("钉钉配置未找到，告警测试将跳过")
            else:
                self.dingtalk = None
        else:
            self.logger.error(f"配置文件不存在: {settings_path}")

    def _setup_logging(self) -> None:
        """配置日志"""
        logs_dir = os.path.join(project_root, 'logs', 'test')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_clean = self.config.symbol.replace('.', '_')
        log_file = os.path.join(logs_dir, f'test_monitor_{symbol_clean}_{timestamp}.log')

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

    def _get_stock_name(self, symbol: str) -> str:
        """获取股票名称"""
        try:
            if get_instruments is None:
                return "未知"
            instruments = get_instruments(symbol)
            if instruments:
                return getattr(instruments[0], 'sec_name', '未知')
            return "未知"
        except Exception:
            return "未知"

    def prepare_data(self) -> bool:
        """
        准备历史数据

        【重要说明】
        获取包含预热期的完整历史数据，后续逐日模拟时会截取使用。

        Returns:
            bool: 数据准备是否成功
        """
        if history is None:
            self.logger.error("未安装掘金量化SDK")
            return False

        # 预热数据起始日期（提前1年）
        warmup_start = (pd.Timestamp(self.config.start_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

        self.logger.info(f"获取历史数据: {self.config.symbol}")
        self.logger.info(f"  测试时间段: {self.config.start_date} ~ {self.config.end_date}")
        self.logger.info(f"  预热数据起始: {warmup_start}")

        # 获取日线数据
        daily_data = history(
            symbol=self.config.symbol,
            frequency='1d',
            start_time=warmup_start + ' 09:00:00',
            end_time=self.config.end_date + ' 15:30:00',
            fields='eob,open,high,low,close,volume',
            df=True,
            adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
        )

        if daily_data is None or daily_data.empty:
            self.logger.error(f"获取数据失败: {self.config.symbol}")
            return False

        self.logger.info(f"获取到 {len(daily_data)} 条日线数据")

        # 构建价格序列
        daily_data['eob'] = pd.to_datetime(daily_data['eob'])
        self._df = daily_data.set_index('eob').sort_index()
        self._close_series = pd.Series(self._df['close'].values, index=self._df.index, name='Close')

        # 获取交易日列表
        self._get_trading_dates()

        # 预计算背离信号（用于后续检测）
        self._prepare_divergences()

        return True

    def _get_trading_dates(self) -> None:
        """获取测试时间段内的交易日列表"""
        try:
            if get_trading_dates is None:
                # 使用简单推断
                dates = []
                current = pd.Timestamp(self.config.start_date)
                end = pd.Timestamp(self.config.end_date)
                while current <= end:
                    if current.weekday() < 5:
                        dates.append(current.strftime('%Y-%m-%d'))
                    current = current + pd.Timedelta(days=1)
                self._trading_dates = dates
                return

            trading_dates = get_trading_dates(
                exchange='SHSE',
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            self._trading_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in trading_dates]

        except Exception as e:
            self.logger.warning(f"获取交易日失败: {e}")
            # 使用简单推断
            dates = []
            current = pd.Timestamp(self.config.start_date)
            end = pd.Timestamp(self.config.end_date)
            while current <= end:
                if current.weekday() < 5:
                    dates.append(current.strftime('%Y-%m-%d'))
                current = current + pd.Timedelta(days=1)
            self._trading_dates = dates

        self.logger.info(f"测试交易日数: {len(self._trading_dates)}")

    def _prepare_divergences(self) -> None:
        """预计算背离信号"""
        try:
            warmup_start = (pd.Timestamp(self.config.end_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
            detector = DivergenceDetector()
            detector.prepare_data(
                symbol=self.config.symbol,
                start_date=warmup_start,
                end_date=self.config.end_date
            )
            divergences = detector.detect_all_divergences()
            self._daily_divergences = divergences.get('daily_top_confirmed', [])
            self._weekly_divergences = divergences.get('weekly_top_confirmed', [])
            self.logger.info(f"预计算背离信号: 日线={len(self._daily_divergences)}, 周线={len(self._weekly_divergences)}")
        except Exception as e:
            self.logger.warning(f"预计算背离信号失败: {e}")

    def _calculate_indicators_for_date(self, target_date: str) -> Tuple[float, float, float, float]:
        """
        计算指定日期的指标值（使用该日期及之前的数据）

        【关键原则 - 无未来函数】
        只使用 target_date 及之前的数据计算指标。

        Args:
            target_date: 目标日期

        Returns:
            Tuple: (日线RSI, 周线RSI, MA5, MA10)
        """
        # 检查缓存
        if target_date in self._daily_rsi_cache:
            return (
                self._daily_rsi_cache[target_date],
                self._weekly_rsi_cache[target_date],
                self._ma5_cache[target_date],
                self._ma10_cache[target_date]
            )

        # 截取目标日期及之前的数据
        target_ts = pd.Timestamp(target_date)
        close_up_to_date = self._close_series[self._close_series.index <= target_ts]

        if len(close_up_to_date) < 10:
            return np.nan, np.nan, np.nan, np.nan

        # 计算日线RSI(6)
        rsi_daily = RSI(period=6, freq='daily')
        daily_rsi_series = rsi_daily.calculate(close_up_to_date)

        # 计算周线RSI(6)
        rsi_weekly = RSI(period=6, freq='weekly')
        weekly_rsi_series = rsi_weekly.calculate(close_up_to_date)

        # 计算均线
        ma_calculator = MACalculator(fast_period=5, slow_period=10, ma_type='SMA')
        ma5_series, ma10_series = ma_calculator.calculate(close_up_to_date)

        # 获取目标日期的值
        daily_rsi = self._get_value_for_date(target_date, daily_rsi_series)
        weekly_rsi = self._get_value_for_date(target_date, weekly_rsi_series)
        ma5 = self._get_value_for_date(target_date, ma5_series)
        ma10 = self._get_value_for_date(target_date, ma10_series)

        # 缓存结果
        self._daily_rsi_cache[target_date] = daily_rsi
        self._weekly_rsi_cache[target_date] = weekly_rsi
        self._ma5_cache[target_date] = ma5
        self._ma10_cache[target_date] = ma10

        return daily_rsi, weekly_rsi, ma5, ma10

    def _get_value_for_date(self, target_date: str, series: pd.Series) -> float:
        """获取指定日期的指标值"""
        try:
            matching = series[series.index.strftime('%Y-%m-%d') == target_date]
            if len(matching) > 0:
                return matching.iloc[-1]
            return np.nan
        except Exception:
            return np.nan

    def _calculate_index_rsi_for_date(self, index_symbol: str, target_date: str) -> float:
        """
        计算指数在指定日期的RSI值（使用该日期及之前的数据）

        Args:
            index_symbol: 指数代码
            target_date: 目标日期

        Returns:
            float: RSI值
        """
        cache_key = f"{index_symbol}_{target_date}"
        if cache_key in self._index_rsi_cyb_cache:
            return self._index_rsi_cyb_cache.get(cache_key, np.nan)

        if history is None:
            return np.nan

        warmup_start = (pd.Timestamp(target_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

        try:
            index_data = history(
                symbol=index_symbol,
                frequency='1d',
                start_time=warmup_start + ' 09:00:00',
                end_time=target_date + ' 15:30:00',
                fields='eob,close',
                df=True,
                adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
            )

            if index_data is None or index_data.empty:
                return np.nan

            index_data['eob'] = pd.to_datetime(index_data['eob'])
            close_series = pd.Series(index_data['close'].values, index=index_data['eob'])

            rsi_calculator = RSI(period=6, freq='daily')
            rsi_series = rsi_calculator.calculate(close_series)

            rsi_value = self._get_value_for_date(target_date, rsi_series)

            # 缓存结果
            if index_symbol == INDEX_CYB:
                self._index_rsi_cyb_cache[cache_key] = rsi_value
            else:
                self._index_rsi_sh_cache[cache_key] = rsi_value

            return rsi_value

        except Exception as e:
            self.logger.warning(f"计算指数RSI失败: {index_symbol} @ {target_date} - {e}")
            return np.nan

    def _detect_ma_cross_down_for_date(self, target_date: str) -> bool:
        """
        检测指定日期的均线死叉（使用该日期及之前的数据）

        Args:
            target_date: 目标日期

        Returns:
            bool: 是否发生死叉
        """
        try:
            ma5_today = self._ma5_cache.get(target_date, np.nan)
            ma10_today = self._ma10_cache.get(target_date, np.nan)

            if np.isnan(ma5_today) or np.isnan(ma10_today):
                return False

            # 找到前一天
            target_ts = pd.Timestamp(target_date)
            prev_dates = self._close_series.index[self._close_series.index < target_ts]
            if len(prev_dates) == 0:
                return False
            prev_date = prev_dates[-1].strftime('%Y-%m-%d')

            ma5_prev = self._ma5_cache.get(prev_date, np.nan)
            ma10_prev = self._ma10_cache.get(prev_date, np.nan)

            if np.isnan(ma5_prev) or np.isnan(ma10_prev):
                return False

            # 死叉条件：前一天MA5>=MA10，当日MA5<MA10
            return ma5_prev >= ma10_prev and ma5_today < ma10_today

        except Exception:
            return False

    def _get_divergence_info_for_date(self, target_date: str, timeframe: str) -> Optional[Dict]:
        """
        获取指定日期的背离信息（检查是否在该日期确认）

        Args:
            target_date: 目标日期
            timeframe: 'daily' 或 'weekly'

        Returns:
            Dict: 背离信息，或 None
        """
        divergences = self._daily_divergences if timeframe == 'daily' else self._weekly_divergences
        processed_set = self._processed_daily_divergences if timeframe == 'daily' else self._processed_weekly_divergences

        for div in divergences:
            if div.confirmation_date is not None:
                div_confirm_date = div.confirmation_date.strftime('%Y-%m-%d')

                # 检查是否在目标日期确认
                if div_confirm_date == target_date:
                    div_id = div.date.strftime('%Y-%m-%d')

                    if div_id in processed_set:
                        continue

                    processed_set.add(div_id)

                    return {
                        'date': div.date,
                        'prev_high': div.peak_a_price,
                        'curr_high': div.peak_b_price,
                        'prev_macd': div.peak_a_macd,
                        'curr_macd': div.peak_b_macd
                    }

        return None

    def test_single_day(self, target_date: str) -> DailyTestResult:
        """
        测试单日信号检测

        【关键原则 - 无未来函数】
        所有指标计算和信号检测都只使用 target_date 及之前的数据。

        Args:
            target_date: 测试日期

        Returns:
            DailyTestResult: 测试结果
        """
        self.logger.debug(f"测试日期: {target_date}")

        # 计算指标（使用该日期及之前的数据）
        daily_rsi, weekly_rsi, ma5, ma10 = self._calculate_indicators_for_date(target_date)

        # 检测均线死叉
        ma_cross_down = self._detect_ma_cross_down_for_date(target_date)

        # 计算指数RSI
        index_rsi_cyb = self._calculate_index_rsi_for_date(INDEX_CYB, target_date)
        index_rsi_sh = self._calculate_index_rsi_for_date(INDEX_SH, target_date)

        # 创建策略状态
        state = StrategyState()

        # 更新RSI警戒级别
        if not np.isnan(weekly_rsi):
            target_ts = pd.Timestamp(target_date)
            self.strategy.update_rsi_flag(weekly_rsi, state, target_ts)

        # 设置背离状态
        daily_div_info = self._get_divergence_info_for_date(target_date, 'daily')
        weekly_div_info = self._get_divergence_info_for_date(target_date, 'weekly')

        if daily_div_info is not None:
            self.strategy.set_daily_divergence(state, daily_div_info)
        if weekly_div_info is not None:
            self.strategy.set_weekly_divergence(state, weekly_div_info)

        # 检测卖出信号
        target_ts = pd.Timestamp(target_date)
        sell_signal_obj = self.strategy.check_sell_signal(
            date=target_ts,
            daily_rsi=daily_rsi,
            weekly_rsi=weekly_rsi,
            ma_cross_down=ma_cross_down,
            state=state,
            position=1.0
        )

        sell_signal = None
        buy_signal = None
        signal_detail = None

        if sell_signal_obj is not None:
            sell_signal = sell_signal_obj.flag.name
            signal_detail = sell_signal_obj.reason

            # 重置状态
            self.strategy.reset_after_sell(state, sell_signal_obj.flag, target_ts)
        else:
            # 检测买入信号
            buy_signal_obj = self.strategy.check_buy_signal(
                date=target_ts,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                state=state,
                has_new_cash=True,
                has_sold_cash=True,
                index_daily_rsi=index_rsi_cyb,
                sh_index_daily_rsi=index_rsi_sh
            )

            if buy_signal_obj is not None and buy_signal_obj.triggered:
                buy_signal = '新资金买入' if buy_signal_obj.is_new_cash else '做T买回'
                signal_detail = buy_signal_obj.reason

        # 获取当日收盘价
        price = self._get_value_for_date(target_date, self._close_series)

        return DailyTestResult(
            date=target_date,
            daily_rsi=daily_rsi,
            weekly_rsi=weekly_rsi,
            ma5=ma5,
            ma10=ma10,
            ma_cross_down=ma_cross_down,
            index_rsi_cyb=index_rsi_cyb,
            index_rsi_sh=index_rsi_sh,
            sell_signal=sell_signal,
            buy_signal=buy_signal,
            signal_detail=signal_detail
        )

    def run_test(self) -> List[DailyTestResult]:
        """
        执行测试

        Returns:
            List[DailyTestResult]: 测试结果列表
        """
        self.logger.info("=" * 80)
        self.logger.info("每日监控脚本测试")
        self.logger.info("=" * 80)
        self.logger.info(f"股票: {self.config.symbol} ({self.config.stock_name})")
        self.logger.info(f"时间段: {self.config.start_date} ~ {self.config.end_date}")
        self.logger.info(f"策略: buy_ids={self.config.judge_buy_ids}, t_ids={self.config.judge_t_ids}, sell_ids={self.config.judge_sell_ids}")
        self.logger.info(f"发送钉钉: {self.config.send_dingtalk}")
        self.logger.info("=" * 80)

        # 准备数据
        if not self.prepare_data():
            self.logger.error("数据准备失败")
            return []

        # 逐日测试
        self.logger.info("")
        self.logger.info("开始逐日测试...")
        self.logger.info("-" * 80)

        for target_date in self._trading_dates:
            result = self.test_single_day(target_date)
            self._results.append(result)

            # 输出当日结果
            self._log_daily_result(result)

        # 输出测试报告
        self._output_test_report()

        # 发送钉钉告警（可选）
        if self.config.send_dingtalk and self.dingtalk is not None:
            self._send_test_dingtalk_alert()

        return self._results

    def _log_daily_result(self, result: DailyTestResult) -> None:
        """输出单日测试结果"""
        daily_rsi_str = f"{result.daily_rsi:.2f}" if not np.isnan(result.daily_rsi) else "NaN"
        weekly_rsi_str = f"{result.weekly_rsi:.2f}" if not np.isnan(result.weekly_rsi) else "NaN"
        ma5_str = f"{result.ma5:.3f}" if not np.isnan(result.ma5) else "NaN"
        ma10_str = f"{result.ma10:.3f}" if not np.isnan(result.ma10) else "NaN"

        signal_str = ""
        if result.sell_signal:
            signal_str = f"🔴 卖出[{result.sell_signal}]"
        elif result.buy_signal:
            signal_str = f"🟢 买入[{result.buy_signal}]"

        self.logger.info(
            f"[{result.date}] 日RSI:{daily_rsi_str} 周RSI:{weekly_rsi_str} "
            f"MA5:{ma5_str} MA10:{ma10_str} 死叉:{result.ma_cross_down} | {signal_str}"
        )

        if result.signal_detail:
            self.logger.info(f"    详情: {result.signal_detail}")

    def _output_test_report(self) -> None:
        """输出测试报告"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("测试报告")
        self.logger.info("=" * 80)

        # 统计信号数量
        sell_count = sum(1 for r in self._results if r.sell_signal)
        buy_count = sum(1 for r in self._results if r.buy_signal)
        no_signal_count = sum(1 for r in self._results if not r.sell_signal and not r.buy_signal)

        self.logger.info(f"总交易日数: {len(self._results)}")
        self.logger.info(f"卖出信号: {sell_count} 次")
        self.logger.info(f"买入信号: {buy_count} 次")
        self.logger.info(f"无信号: {no_signal_count} 次")

        # 输出信号详情
        self.logger.info("")
        self.logger.info("【卖出信号详情】")
        self.logger.info("-" * 80)

        for r in self._results:
            if r.sell_signal:
                self.logger.info(f"  [{r.date}] {r.sell_signal}: {r.signal_detail}")
                daily_rsi_str = f"{r.daily_rsi:.2f}" if not np.isnan(r.daily_rsi) else "N/A"
                weekly_rsi_str = f"{r.weekly_rsi:.2f}" if not np.isnan(r.weekly_rsi) else "N/A"
                self.logger.info(f"    日RSI: {daily_rsi_str}, 周RSI: {weekly_rsi_str}, 均线死叉: {r.ma_cross_down}")

        self.logger.info("")
        self.logger.info("【买入信号详情】")
        self.logger.info("-" * 80)

        for r in self._results:
            if r.buy_signal:
                self.logger.info(f"  [{r.date}] {r.buy_signal}: {r.signal_detail}")
                daily_rsi_str = f"{r.daily_rsi:.2f}" if not np.isnan(r.daily_rsi) else "N/A"
                weekly_rsi_str = f"{r.weekly_rsi:.2f}" if not np.isnan(r.weekly_rsi) else "N/A"
                cyb_rsi_str = f"{r.index_rsi_cyb:.2f}" if not np.isnan(r.index_rsi_cyb) else "N/A"
                sh_rsi_str = f"{r.index_rsi_sh:.2f}" if not np.isnan(r.index_rsi_sh) else "N/A"
                self.logger.info(f"    日RSI: {daily_rsi_str}, 周RSI: {weekly_rsi_str}, 创业板RSI: {cyb_rsi_str}, 上证RSI: {sh_rsi_str}")

        # 保存CSV报告
        self._save_csv_report()

        self.logger.info("")
        self.logger.info(f"测试完成，日志文件: {self.log_file}")
        self.logger.info("=" * 80)

    def _save_csv_report(self) -> None:
        """保存CSV格式测试报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbol_clean = self.config.symbol.replace('.', '_')
        csv_file = os.path.join(project_root, 'logs', 'test', f'test_result_{symbol_clean}_{timestamp}.csv')

        data = []
        for r in self._results:
            data.append({
                '日期': r.date,
                '日RSI': r.daily_rsi if not np.isnan(r.daily_rsi) else '',
                '周RSI': r.weekly_rsi if not np.isnan(r.weekly_rsi) else '',
                'MA5': r.ma5 if not np.isnan(r.ma5) else '',
                'MA10': r.ma10 if not np.isnan(r.ma10) else '',
                '均线死叉': '是' if r.ma_cross_down else '否',
                '创业板指数RSI': r.index_rsi_cyb if not np.isnan(r.index_rsi_cyb) else '',
                '上证指数RSI': r.index_rsi_sh if not np.isnan(r.index_rsi_sh) else '',
                '卖出信号': r.sell_signal or '',
                '买入信号': r.buy_signal or '',
                '信号详情': r.signal_detail or ''
            })

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"CSV报告: {csv_file}")

    def _send_test_dingtalk_alert(self) -> None:
        """发送测试钉钉告警"""
        sell_signals = [r for r in self._results if r.sell_signal]
        buy_signals = [r for r in self._results if r.buy_signal]

        if len(sell_signals) == 0 and len(buy_signals) == 0:
            self.logger.info("无信号触发，不发送钉钉告警")
            return

        title = f"监控脚本测试报告 - {self.config.stock_name}"

        content_lines = []
        content_lines.append(f"## 监控脚本测试报告\n\n")
        content_lines.append(f"**测试标的**: {self.config.stock_name} ({self.config.symbol})\n")
        content_lines.append(f"**测试时间段**: {self.config.start_date} ~ {self.config.end_date}\n")
        content_lines.append(f"**策略配置**: buy={self.config.judge_buy_ids}, t={self.config.judge_t_ids}, sell={self.config.judge_sell_ids}\n")
        content_lines.append(f"**总交易日**: {len(self._results)} 天\n\n")

        if len(sell_signals) > 0:
            content_lines.append(f"### 🔴 卖出信号 ({len(sell_signals)}次)\n---\n")
            for r in sell_signals:
                daily_rsi_str = f"{r.daily_rsi:.2f}" if not np.isnan(r.daily_rsi) else "N/A"
                weekly_rsi_str = f"{r.weekly_rsi:.2f}" if not np.isnan(r.weekly_rsi) else "N/A"
                content_lines.append(f"- **{r.date}** [{r.sell_signal}]\n")
                content_lines.append(f"  - 日RSI: {daily_rsi_str} | 周RSI: {weekly_rsi_str}\n")
                content_lines.append(f"  - {r.signal_detail}\n\n")

        if len(buy_signals) > 0:
            content_lines.append(f"### 🟢 买入信号 ({len(buy_signals)}次)\n---\n")
            for r in buy_signals:
                daily_rsi_str = f"{r.daily_rsi:.2f}" if not np.isnan(r.daily_rsi) else "N/A"
                weekly_rsi_str = f"{r.weekly_rsi:.2f}" if not np.isnan(r.weekly_rsi) else "N/A"
                cyb_rsi_str = f"{r.index_rsi_cyb:.2f}" if not np.isnan(r.index_rsi_cyb) else "N/A"
                sh_rsi_str = f"{r.index_rsi_sh:.2f}" if not np.isnan(r.index_rsi_sh) else "N/A"
                content_lines.append(f"- **{r.date}** [{r.buy_signal}]\n")
                content_lines.append(f"  - 日RSI: {daily_rsi_str} | 周RSI: {weekly_rsi_str}\n")
                content_lines.append(f"  - 创业板RSI: {cyb_rsi_str} | 上证RSI: {sh_rsi_str}\n")
                content_lines.append(f"  - {r.signal_detail}\n\n")

        content = "".join(content_lines)

        success = self.dingtalk.send_markdown(title, content, at_all=False)
        if success:
            self.logger.info("钉钉测试告警发送成功")
        else:
            self.logger.error("钉钉测试告警发送失败")


# ==================== 辅助函数 ====================

def parse_ids(id_str: str) -> List[int]:
    """解析策略ID字符串"""
    if not id_str:
        return []
    return [int(x.strip()) for x in id_str.split(',') if x.strip()]


def add_exchange_prefix(code: str) -> str:
    """添加交易所前缀"""
    code = code.strip()
    if code.startswith('SHSE.') or code.startswith('SZSE.'):
        return code
    if code.startswith('6') or code.startswith('5'):
        return f'SHSE.{code}'
    elif code.startswith('0') or code.startswith('3') or code.startswith('1'):
        return f'SZSE.{code}'
    else:
        return f'SZSE.{code}'


def get_stock_name(symbol: str) -> str:
    """获取股票名称"""
    try:
        if get_instruments is None:
            return "未知"
        instruments = get_instruments(symbol)
        if instruments:
            return getattr(instruments[0], 'sec_name', '未知')
        return "未知"
    except Exception:
        return "未知"


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="每日监控脚本测试")

    parser.add_argument('--symbol', type=str, default='512480',
                       help='股票代码（如 512480）')
    parser.add_argument('--start', type=str, default='2025-01-01',
                       help='测试开始日期（YYYY-MM-DD）')
    parser.add_argument('--end', type=str, default='2025-12-31',
                       help='测试结束日期（YYYY-MM-DD）')
    parser.add_argument('--buy-ids', type=str, default='1,5',
                       help='买入策略ID（逗号分隔，如 1,5）')
    parser.add_argument('--t-ids', type=str, default='',
                       help='做T策略ID（逗号分隔）')
    parser.add_argument('--sell-ids', type=str, default='1',
                       help='卖出策略ID（逗号分隔）')
    parser.add_argument('--send-dingtalk', action='store_true',
                       help='是否发送钉钉告警测试')

    args = parser.parse_args()

    # 解析参数
    symbol = add_exchange_prefix(args.symbol)
    stock_name = get_stock_name(symbol)
    judge_buy_ids = parse_ids(args.buy_ids)
    judge_t_ids = parse_ids(args.t_ids)
    judge_sell_ids = parse_ids(args.sell_ids)

    # 创建测试配置
    config = TestConfig(
        symbol=symbol,
        stock_name=stock_name,
        start_date=args.start,
        end_date=args.end,
        judge_buy_ids=judge_buy_ids,
        judge_t_ids=judge_t_ids,
        judge_sell_ids=judge_sell_ids,
        send_dingtalk=args.send_dingtalk
    )

    # 执行测试
    tester = MonitorTester(config)
    tester.run_test()


if __name__ == "__main__":
    main()