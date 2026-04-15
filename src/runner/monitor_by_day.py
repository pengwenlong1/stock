# -*- coding: utf-8 -*-
"""
日内监控脚本 (monitor_by_day.py)

【功能说明】
定时监控股票是否达到买卖点指标要求，根据backtest_runner.py中的逻辑实现。
每天三个时间点执行监控：
- 12:00（中午）- 盘中监控
- 14:30（尾盘）- 收盘前监控
- 20:00（晚间）- 复盘监控
监控最近两个交易日的信号，及时发现买卖机会。

【使用方法】
直接运行脚本：
    python src/runner/monitor_by_day.py

【监控逻辑】
基于 stocks_backtest.csv 中每只股票的独立策略配置：
- judge_buy_ids: 新资金买入策略ID
- judge_t_ids: 做T买回策略ID
- judge_sell_ids: 卖出策略ID
- active: 是否监控（1=监控, 0=不监控）

【告警方式】
通过钉钉机器人发送告警消息，使用加签安全模式。
告警最近两个交易日的买卖信号。

作者：量化交易团队
创建日期：2024
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
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


# ==================== 监控配置 ====================

# 监控时间点（小时:分钟）
# 12:00 - 中午监控
# 14:30 - 尾盘监控
# 20:00 - 晚间复盘监控
MONITOR_TIMES = ['12:00', '14:30', '20:00']

# 创业板指数和上证指数代码
INDEX_CYB = 'SZSE.399006'   # 创业板指数
INDEX_SH = 'SHSE.000001'    # 上证指数

# 预热天数（用于指标计算）
WARMUP_DAYS = 365

# 监控最近几个交易日
RECENT_TRADING_DAYS = 2


# ==================== 数据结构 ====================

@dataclass
class StockConfig:
    """股票配置"""
    symbol: str
    stock_name: str
    start_date: str
    end_date: str
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]
    initial_capital: float
    active: bool


@dataclass
class SignalInfo:
    """信号信息（单日）"""
    signal_date: str                           # 信号触发日期
    signal_type: str                           # 信号类型：'buy' 或 'sell'
    signal_detail: str                         # 信号详情
    current_price: float                       # 当日收盘价
    daily_rsi: float                           # 当日RSI
    weekly_rsi: float                          # 当日周线RSI
    index_daily_rsi_cyb: float                 # 创业板指数RSI
    index_daily_rsi_sh: float                  # 上证指数RSI


@dataclass
class MonitorResult:
    """监控结果（多日）"""
    symbol: str
    stock_name: str
    latest_date: str                           # 最新数据日期
    latest_price: float                        # 最新收盘价
    latest_daily_rsi: float                    # 最新日线RSI
    latest_weekly_rsi: float                   # 最新周线RSI
    latest_ma5: float                          # 最新MA5
    latest_ma10: float                         # 最新MA10
    latest_ma_cross_down: bool                 # 最新均线死叉状态
    latest_index_daily_rsi_cyb: float          # 最新创业板指数RSI
    latest_index_daily_rsi_sh: float           # 最新上证指数RSI
    signals: List[SignalInfo]                  # 最近两日的信号列表
    notes: str                                 # 备注
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]


# ==================== 钉钉告警类 ====================

class DingTalkNotifier:
    """
    钉钉机器人告警类

    【功能说明】
    通过钉钉机器人发送告警消息，使用加签安全模式。

    【安全模式】
    使用加签（signature）方式验证消息来源的安全性。
    签名算法：HmacSHA256(timestamp + "\n" + secret, secret)
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

    def send_text(self, content: str, at_all: bool = True) -> bool:
        """
        发送文本消息

        Args:
            content: 消息内容
            at_all: 是否@所有人

        Returns:
            bool: 发送是否成功
        """
        timestamp = int(time.time() * 1000)
        sign = self._generate_sign(timestamp)

        url = f"{self.webhook}&timestamp={timestamp}&sign={sign}"

        headers = {'Content-Type': 'application/json;charset=utf-8'}
        data = {
            "msgtype": "text",
            "text": {
                "content": content
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
                logging.error(f"钉钉发送失败: {result}")
                return False

        except Exception as e:
            logging.error(f"钉钉发送异常: {e}")
            return False

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
                logging.error(f"钉钉发送失败: {result}")
                return False

        except Exception as e:
            logging.error(f"钉钉发送异常: {e}")
            return False


# ==================== 监控器类 ====================

class DailyMonitor:
    """
    日内监控器

    【功能说明】
    定时监控股票的买卖信号，基于 backtest_runner.py 的策略逻辑。
    监控最近两个交易日的信号。

    【监控指标】
    - 日线RSI(6)
    - 周线RSI(6)
    - MA5/MA10均线及死叉检测
    - 创业板指数RSI
    - 上证指数RSI
    - 顶背离信号（日线/周线）
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 settings_path: Optional[str] = None,
                 output_dir: Optional[str] = None) -> None:
        """
        初始化监控器

        Args:
            config_path: 股票配置文件路径（默认为 config/stocks_backtest.csv）
            settings_path: 设置文件路径（默认为 config/settings.json）
            output_dir: 输出目录路径
        """
        # 配置文件路径
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'stocks_backtest.csv')
        self.config_path = config_path

        if settings_path is None:
            settings_path = os.path.join(project_root, 'config', 'settings.json')
        self.settings_path = settings_path

        # 输出目录
        if output_dir is None:
            output_dir = os.path.join(project_root, 'logs', 'monitor')
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 初始化掘金token和钉钉配置
        self._init_settings()

        # 日志配置
        self._setup_logging()

    def _init_settings(self) -> None:
        """初始化设置（掘金token和钉钉配置）"""
        if not os.path.exists(self.settings_path):
            self.logger = logging.getLogger(__name__)
            self.logger.error(f"设置文件不存在: {self.settings_path}")
            return

        with open(self.settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # 初始化掘金token
        gm_token = settings.get('gm_token')
        if gm_token and set_token is not None:
            set_token(gm_token)

        # 初始化钉钉通知器
        dingtalk_webhook = settings.get('dingtalk_webhook')
        dingtalk_secret = settings.get('dingtalk_secret')
        if dingtalk_webhook and dingtalk_secret:
            self.dingtalk = DingTalkNotifier(dingtalk_webhook, dingtalk_secret)
        else:
            self.dingtalk = None

    def _setup_logging(self) -> None:
        """配置日志"""
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(self.output_dir, f'monitor_{timestamp}.log')

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

    def _parse_ids(self, id_str: str) -> List[int]:
        """
        解析策略ID字符串

        Args:
            id_str: 策略ID字符串（如 "4" 或 "2;6" 或空）

        Returns:
            List[int]: 策略ID列表
        """
        if not id_str or pd.isna(id_str):
            return []
        if isinstance(id_str, int):
            return [id_str]
        # 分号分隔的多个ID
        return [int(x.strip()) for x in str(id_str).replace(';', ',').split(',') if x.strip()]

    def get_recent_trading_dates(self, end_date: str, count: int = RECENT_TRADING_DAYS) -> List[str]:
        """
        获取最近N个交易日

        Args:
            end_date: 结束日期
            count: 需要的交易日数量

        Returns:
            List[str]: 交易日列表（按时间倒序，最近的在前）
        """
        try:
            if get_trading_dates is None:
                # 无法获取交易日，使用简单推断
                dates = []
                current = pd.Timestamp(end_date)
                while len(dates) < count + 5:  # 多取几天
                    if current.weekday() < 5:  # 周一到周五
                        dates.append(current.strftime('%Y-%m-%d'))
                    current = current - pd.Timedelta(days=1)
                return dates[:count]

            # 获取最近10天的交易日列表
            start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
            trading_dates = get_trading_dates(
                exchange='SHSE',
                start_date=start_date,
                end_date=end_date
            )

            # 转换为字符串列表并倒序
            trading_dates_str = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in trading_dates]
            trading_dates_str.reverse()  # 倒序，最近的在前

            return trading_dates_str[:count]

        except Exception as e:
            self.logger.warning(f"获取交易日失败: {e}")
            # 使用简单推断
            dates = []
            current = pd.Timestamp(end_date)
            while len(dates) < count:
                if current.weekday() < 5:
                    dates.append(current.strftime('%Y-%m-%d'))
                current = current - pd.Timedelta(days=1)
            return dates

    def load_stock_config(self) -> List[StockConfig]:
        """
        加载股票配置

        Returns:
            股票配置列表（只包含active=1的股票）
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
            stock_name = self._get_stock_name(symbol)

            # 解析策略ID
            judge_buy_ids = self._parse_ids(row.get('judge_buy_ids', ''))
            judge_t_ids = self._parse_ids(row.get('judge_t_ids', ''))
            judge_sell_ids = self._parse_ids(row.get('judge_sell_ids', ''))

            stock_config = StockConfig(
                symbol=symbol,
                stock_name=stock_name,
                start_date=str(row['start_date']),
                end_date=str(row['end_date']),
                judge_buy_ids=judge_buy_ids,
                judge_t_ids=judge_t_ids,
                judge_sell_ids=judge_sell_ids,
                initial_capital=float(row.get('initial_capital', 100000)),
                active=True
            )
            stocks.append(stock_config)

        self.logger.info(f"加载股票配置: {len(stocks)} 只（active=1）")
        return stocks

    def get_historical_data(self,
                            symbol: str,
                            end_date: str) -> Optional[pd.DataFrame]:
        """
        获取历史数据

        Args:
            symbol: 股票代码
            end_date: 结束日期

        Returns:
            DataFrame: 包含 open, high, low, close, volume 列
        """
        if history is None:
            self.logger.error("未安装掘金量化SDK")
            return None

        warmup_start = (pd.Timestamp(end_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

        try:
            daily_data = history(
                symbol=symbol,
                frequency='1d',
                start_time=warmup_start + ' 09:00:00',
                end_time=end_date + ' 15:30:00',
                fields='eob,open,high,low,close,volume',
                df=True,
                adjust=ADJUST_PREV if ADJUST_PREV is not None else 1
            )

            if daily_data is None or daily_data.empty:
                self.logger.warning(f"获取数据失败: {symbol}")
                return None

            daily_data['eob'] = pd.to_datetime(daily_data['eob'])
            df = daily_data.set_index('eob').sort_index()
            return df

        except Exception as e:
            self.logger.error(f"获取历史数据异常: {symbol} - {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        计算技术指标

        Args:
            df: 历史数据DataFrame

        Returns:
            Tuple: (日线RSI, 周线RSI, MA5, MA10)
        """
        close_series = pd.Series(df['close'].values, index=df.index, name='Close')

        # 计算日线RSI(6)
        rsi_daily = RSI(period=6, freq='daily')
        daily_rsi = rsi_daily.calculate(close_series)

        # 计算周线RSI(6)
        rsi_weekly = RSI(period=6, freq='weekly')
        weekly_rsi = rsi_weekly.calculate(close_series)

        # 计算均线
        ma_calculator = MACalculator(fast_period=5, slow_period=10, ma_type='SMA')
        ma5, ma10 = ma_calculator.calculate(close_series)

        return daily_rsi, weekly_rsi, ma5, ma10

    def get_index_rsi(self, index_symbol: str, end_date: str) -> float:
        """
        获取指数RSI值

        Args:
            index_symbol: 指数代码
            end_date: 结束日期

        Returns:
            float: 指数日线RSI值
        """
        if history is None:
            return np.nan

        warmup_start = (pd.Timestamp(end_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

        try:
            index_data = history(
                symbol=index_symbol,
                frequency='1d',
                start_time=warmup_start + ' 09:00:00',
                end_time=end_date + ' 15:30:00',
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

            if len(rsi_series) > 0:
                return rsi_series.iloc[-1]
            return np.nan

        except Exception as e:
            self.logger.warning(f"获取指数RSI失败: {index_symbol} - {e}")
            return np.nan

    def get_index_rsi_at_date(self, index_symbol: str, target_date: str) -> float:
        """
        获取指数在指定日期的RSI值

        Args:
            index_symbol: 指数代码
            target_date: 目标日期

        Returns:
            float: 指数日线RSI值
        """
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

            # 找到目标日期的RSI值
            target_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
            matching = rsi_series[rsi_series.index.strftime('%Y-%m-%d') == target_date_str]
            if len(matching) > 0:
                return matching.iloc[-1]
            return np.nan

        except Exception as e:
            self.logger.warning(f"获取指数RSI失败: {index_symbol} @ {target_date} - {e}")
            return np.nan

    def detect_ma_cross_down(self,
                              date: pd.Timestamp,
                              ma5: pd.Series,
                              ma10: pd.Series,
                              close_series: pd.Series) -> bool:
        """
        检测均线死叉

        Args:
            date: 当前日期
            ma5: MA5序列
            ma10: MA10序列
            close_series: 收盘价序列

        Returns:
            bool: 是否发生死叉
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            matching_ma5 = ma5[ma5.index.strftime('%Y-%m-%d') == date_str]
            matching_ma10 = ma10[ma10.index.strftime('%Y-%m-%d') == date_str]

            if len(matching_ma5) == 0 or len(matching_ma10) == 0:
                return False

            ma5_today = matching_ma5.iloc[0]
            ma10_today = matching_ma10.iloc[0]

            prev_dates = close_series.index[close_series.index < date]
            if len(prev_dates) == 0:
                return False
            prev_date = prev_dates[-1]

            prev_date_str = prev_date.strftime('%Y-%m-%d')
            matching_ma5_prev = ma5[ma5.index.strftime('%Y-%m-%d') == prev_date_str]
            matching_ma10_prev = ma10[ma10.index.strftime('%Y-%m-%d') == prev_date_str]

            if len(matching_ma5_prev) == 0 or len(matching_ma10_prev) == 0:
                return False

            ma5_prev = matching_ma5_prev.iloc[0]
            ma10_prev = matching_ma10_prev.iloc[0]

            if np.isnan(ma5_today) or np.isnan(ma10_today) or np.isnan(ma5_prev) or np.isnan(ma10_prev):
                return False

            return ma5_prev >= ma10_prev and ma5_today < ma10_today

        except Exception:
            return False

    def get_divergence_signals(self, symbol: str, end_date: str) -> Tuple[List, List]:
        """
        获取背离信号

        Args:
            symbol: 股票代码
            end_date: 结束日期

        Returns:
            Tuple: (日线顶背离列表, 周线顶背离列表)
        """
        try:
            warmup_start = (pd.Timestamp(end_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

            detector = DivergenceDetector()
            detector.prepare_data(symbol=symbol, start_date=warmup_start, end_date=end_date)
            divergences = detector.detect_all_divergences()

            daily_div = divergences.get('daily_top_confirmed', [])
            weekly_div = divergences.get('weekly_top_confirmed', [])

            return daily_div, weekly_div

        except Exception as e:
            self.logger.warning(f"获取背离信号失败: {symbol} - {e}")
            return [], []

    def _get_value_at_date(self, date: pd.Timestamp, series: pd.Series) -> float:
        """获取指定日期的指标值"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = series[series.index.strftime('%Y-%m-%d') == date_str]
            if len(matching) > 0:
                return matching.iloc[-1]
            return np.nan
        except Exception:
            return np.nan

    def monitor_single_stock(self,
                              stock_config: StockConfig,
                              end_date: str) -> Optional[MonitorResult]:
        """
        监控单只股票（检测最近两个交易日）

        Args:
            stock_config: 股票配置
            end_date: 监控日期

        Returns:
            MonitorResult: 监控结果
        """
        symbol = stock_config.symbol
        self.logger.info(f"监控股票: {symbol} ({stock_config.stock_name}) "
                        f"策略: buy={stock_config.judge_buy_ids}, t={stock_config.judge_t_ids}, sell={stock_config.judge_sell_ids}")

        # 获取历史数据
        df = self.get_historical_data(symbol, end_date)
        if df is None or df.empty:
            self.logger.warning(f"无法获取数据: {symbol}")
            return None

        # 计算指标
        close_series = pd.Series(df['close'].values, index=df.index, name='Close')
        daily_rsi, weekly_rsi, ma5, ma10 = self.calculate_indicators(df)

        # 获取背离信号
        daily_div, weekly_div = self.get_divergence_signals(symbol, end_date)

        # 创建策略实例（使用股票独立配置）
        strategy = TradingStrategy(
            buy_ids=stock_config.judge_buy_ids,
            t_ids=stock_config.judge_t_ids,
            sell_ids=stock_config.judge_sell_ids
        )

        # 获取最近两个交易日
        recent_dates = self.get_recent_trading_dates(end_date, RECENT_TRADING_DAYS)
        self.logger.info(f"  检测交易日: {recent_dates}")

        # 收集最近两日的信号
        signals = []

        for target_date_str in recent_dates:
            # 找到对应的数据日期
            matching_dates = df.index[df.index.strftime('%Y-%m-%d') == target_date_str]
            if len(matching_dates) == 0:
                continue

            target_date = matching_dates[0]

            # 获取当日指标值
            daily_rsi_value = self._get_value_at_date(target_date, daily_rsi)
            weekly_rsi_value = self._get_value_at_date(target_date, weekly_rsi)
            ma5_value = self._get_value_at_date(target_date, ma5)
            ma10_value = self._get_value_at_date(target_date, ma10)
            current_price = self._get_value_at_date(target_date, close_series)

            # 检测均线死叉
            ma_cross_down = self.detect_ma_cross_down(target_date, ma5, ma10, close_series)

            # 获取当日指数RSI
            index_rsi_cyb = self.get_index_rsi_at_date(INDEX_CYB, target_date_str)
            index_rsi_sh = self.get_index_rsi_at_date(INDEX_SH, target_date_str)

            # 创建策略状态
            state = StrategyState()

            # 更新RSI警戒级别
            if not np.isnan(weekly_rsi_value):
                strategy.update_rsi_flag(weekly_rsi_value, state, target_date)

            # 设置背离状态（检查是否有最近确认的背离）
            for div in daily_div:
                if div.confirmation_date is not None:
                    if div.confirmation_date.strftime('%Y-%m-%d') == target_date_str or \
                       (div.confirmation_date <= target_date and div.confirmation_date >= target_date - pd.Timedelta(days=5)):
                        div_info = {
                            'date': div.date,
                            'prev_high': div.peak_a_price,
                            'curr_high': div.peak_b_price,
                            'prev_macd': div.peak_a_macd,
                            'curr_macd': div.peak_b_macd
                        }
                        strategy.set_daily_divergence(state, div_info)
                        break

            for div in weekly_div:
                if div.confirmation_date is not None:
                    if div.confirmation_date.strftime('%Y-%m-%d') == target_date_str or \
                       (div.confirmation_date <= target_date and div.confirmation_date >= target_date - pd.Timedelta(days=5)):
                        div_info = {
                            'date': div.date,
                            'prev_high': div.peak_a_price,
                            'curr_high': div.peak_b_price,
                            'prev_macd': div.peak_a_macd,
                            'curr_macd': div.peak_b_macd
                        }
                        strategy.set_weekly_divergence(state, div_info)
                        break

            # 检测卖出信号
            sell_signal_obj = strategy.check_sell_signal(
                date=target_date,
                daily_rsi=daily_rsi_value,
                weekly_rsi=weekly_rsi_value,
                ma_cross_down=ma_cross_down,
                state=state,
                position=1.0
            )

            if sell_signal_obj is not None:
                signal_info = SignalInfo(
                    signal_date=target_date_str,
                    signal_type='sell',
                    signal_detail=f"[{sell_signal_obj.flag.name}] {sell_signal_obj.reason}",
                    current_price=current_price,
                    daily_rsi=daily_rsi_value,
                    weekly_rsi=weekly_rsi_value,
                    index_daily_rsi_cyb=index_rsi_cyb,
                    index_daily_rsi_sh=index_rsi_sh
                )
                signals.append(signal_info)
                continue  # 卖出信号触发后不检测买入信号

            # 检测买入信号
            buy_signal_obj = strategy.check_buy_signal(
                date=target_date,
                daily_rsi=daily_rsi_value,
                weekly_rsi=weekly_rsi_value,
                state=state,
                has_new_cash=True,
                has_sold_cash=True,
                index_daily_rsi=index_rsi_cyb,
                sh_index_daily_rsi=index_rsi_sh
            )

            if buy_signal_obj is not None and buy_signal_obj.triggered:
                signal_info = SignalInfo(
                    signal_date=target_date_str,
                    signal_type='buy',
                    signal_detail=f"[{'新资金买入' if buy_signal_obj.is_new_cash else '做T买回'}] {buy_signal_obj.reason}",
                    current_price=current_price,
                    daily_rsi=daily_rsi_value,
                    weekly_rsi=weekly_rsi_value,
                    index_daily_rsi_cyb=index_rsi_cyb,
                    index_daily_rsi_sh=index_rsi_sh
                )
                signals.append(signal_info)

        # 获取最新日期的数据（用于展示当前状态）
        last_date = df.index[-1]
        last_date_str = last_date.strftime('%Y-%m-%d')

        latest_daily_rsi = daily_rsi.iloc[-1] if len(daily_rsi) > 0 else np.nan
        latest_weekly_rsi = weekly_rsi.iloc[-1] if len(weekly_rsi) > 0 else np.nan
        latest_ma5 = ma5.iloc[-1] if len(ma5) > 0 else np.nan
        latest_ma10 = ma10.iloc[-1] if len(ma10) > 0 else np.nan
        latest_price = df['close'].iloc[-1]
        latest_ma_cross_down = self.detect_ma_cross_down(last_date, ma5, ma10, close_series)
        latest_index_rsi_cyb = self.get_index_rsi(INDEX_CYB, end_date)
        latest_index_rsi_sh = self.get_index_rsi(INDEX_SH, end_date)

        # 构建备注
        notes = ""
        if latest_ma_cross_down:
            notes += "MA5死叉MA10; "

        # 创建监控结果
        result = MonitorResult(
            symbol=symbol,
            stock_name=stock_config.stock_name,
            latest_date=last_date_str,
            latest_price=latest_price,
            latest_daily_rsi=latest_daily_rsi,
            latest_weekly_rsi=latest_weekly_rsi,
            latest_ma5=latest_ma5,
            latest_ma10=latest_ma10,
            latest_ma_cross_down=latest_ma_cross_down,
            latest_index_daily_rsi_cyb=latest_index_rsi_cyb,
            latest_index_daily_rsi_sh=latest_index_rsi_sh,
            signals=signals,
            notes=notes,
            judge_buy_ids=stock_config.judge_buy_ids,
            judge_t_ids=stock_config.judge_t_ids,
            judge_sell_ids=stock_config.judge_sell_ids
        )

        return result

    def run_monitor(self) -> List[MonitorResult]:
        """
        执行监控

        Returns:
            List[MonitorResult]: 监控结果列表
        """
        self.logger.info("=" * 80)
        self.logger.info("日内监控启动")
        self.logger.info(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"监控范围: 最近 {RECENT_TRADING_DAYS} 个交易日")
        self.logger.info("=" * 80)

        # 加载股票配置
        stocks = self.load_stock_config()
        if len(stocks) == 0:
            self.logger.error("无活跃股票配置")
            return []

        # 获取当前日期
        today = datetime.now().strftime('%Y-%m-%d')

        # 监控结果列表
        results = []

        # 逐只股票监控
        for stock_config in stocks:
            result = self.monitor_single_stock(stock_config, today)
            if result is not None:
                results.append(result)

        # 输出监控报告
        self._output_report(results)

        # 发送钉钉告警
        self._send_dingtalk_alert(results)

        return results

    def _output_report(self, results: List[MonitorResult]) -> None:
        """
        输出监控报告

        Args:
            results: 监控结果列表
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("监控报告")
        self.logger.info("=" * 80)

        # 按信号数量排序（有信号的排前面）
        sorted_results = sorted(results, key=lambda r: -len(r.signals))

        # 输出表格格式
        self.logger.info("")
        self.logger.info(f"{'股票代码':<12} {'股票名称':<10} {'最新日期':<12} {'收盘价':<8} "
                        f"{'日RSI':<8} {'周RSI':<8} {'信号数':<6}")
        self.logger.info("-" * 70)

        for r in sorted_results:
            daily_rsi_str = f"{r.latest_daily_rsi:.2f}" if not np.isnan(r.latest_daily_rsi) else "NaN"
            weekly_rsi_str = f"{r.latest_weekly_rsi:.2f}" if not np.isnan(r.latest_weekly_rsi) else "NaN"
            signal_count = len(r.signals)

            self.logger.info(f"{r.symbol:<12} {r.stock_name:<10} {r.latest_date:<12} {r.latest_price:<8.3f} "
                           f"{daily_rsi_str:<8} {weekly_rsi_str:<8} {signal_count:<6}")

        # 输出信号详情
        self.logger.info("")
        self.logger.info("【信号详情】")
        self.logger.info("-" * 80)

        # 按日期分组输出信号
        for r in sorted_results:
            if len(r.signals) > 0:
                self.logger.info("")
                self.logger.info(f">>> {r.stock_name} ({r.symbol}) <<<")
                for sig in r.signals:
                    signal_type_str = "🔴 卖出" if sig.signal_type == 'sell' else "🟢 买入"
                    daily_rsi_str = f"{sig.daily_rsi:.2f}" if not np.isnan(sig.daily_rsi) else "N/A"
                    weekly_rsi_str = f"{sig.weekly_rsi:.2f}" if not np.isnan(sig.weekly_rsi) else "N/A"
                    cyb_rsi_str = f"{sig.index_daily_rsi_cyb:.2f}" if not np.isnan(sig.index_daily_rsi_cyb) else "N/A"
                    sh_rsi_str = f"{sig.index_daily_rsi_sh:.2f}" if not np.isnan(sig.index_daily_rsi_sh) else "N/A"

                    self.logger.info(f"  [{sig.signal_date}] {signal_type_str}")
                    self.logger.info(f"    价格: {sig.current_price:.3f} | 日RSI: {daily_rsi_str} | 周RSI: {weekly_rsi_str}")
                    if sig.signal_type == 'buy':
                        self.logger.info(f"    创业板RSI: {cyb_rsi_str} | 上证RSI: {sh_rsi_str}")
                    self.logger.info(f"    详情: {sig.signal_detail}")

        # 无信号统计
        no_signals = [r for r in sorted_results if len(r.signals) == 0]
        if len(no_signals) > 0:
            self.logger.info("")
            self.logger.info(f">>> 无信号 ({len(no_signals)}只) <<<")

        # 保存CSV报告
        self._save_csv_report(results)

        self.logger.info("")
        self.logger.info("=" * 80)

    def _send_dingtalk_alert(self, results: List[MonitorResult]) -> None:
        """
        发送钉钉告警

        Args:
            results: 监控结果列表
        """
        if self.dingtalk is None:
            self.logger.warning("钉钉配置未设置，跳过告警发送")
            return

        # 收集有信号的股票
        results_with_signals = [r for r in results if len(r.signals) > 0]

        # 如果没有信号，不发送告警
        if len(results_with_signals) == 0:
            self.logger.info("无信号触发，不发送钉钉告警")
            return

        # 构建Markdown消息
        title = f"股票监控告警 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        content_lines = []
        content_lines.append(f"## 股票监控告警\n")
        content_lines.append(f"**监控时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content_lines.append(f"**监控范围**: 最近 {RECENT_TRADING_DAYS} 个交易日\n")
        content_lines.append(f"**监控股票数**: {len(results)} 只\n")
        content_lines.append(f"**有信号股票**: {len(results_with_signals)} 只\n\n")

        # 按信号类型分组
        sell_signals_by_date: Dict[str, List] = {}
        buy_signals_by_date: Dict[str, List] = {}

        for r in results_with_signals:
            for sig in r.signals:
                if sig.signal_type == 'sell':
                    if sig.signal_date not in sell_signals_by_date:
                        sell_signals_by_date[sig.signal_date] = []
                    sell_signals_by_date[sig.signal_date].append((r, sig))
                else:
                    if sig.signal_date not in buy_signals_by_date:
                        buy_signals_by_date[sig.signal_date] = []
                    buy_signals_by_date[sig.signal_date].append((r, sig))

        # 卖出信号（按日期分组显示）
        if len(sell_signals_by_date) > 0:
            total_sell = sum(len(v) for v in sell_signals_by_date.values())
            content_lines.append(f"### 🔴 卖出信号 ({total_sell}条)\n")
            content_lines.append("---\n")

            for date in sorted(sell_signals_by_date.keys(), reverse=True):
                signals_list = sell_signals_by_date[date]
                content_lines.append(f"**日期: {date}** ({len(signals_list)}只)\n")
                for r, sig in signals_list:
                    daily_rsi_str = f"{sig.daily_rsi:.2f}" if not np.isnan(sig.daily_rsi) else "N/A"
                    weekly_rsi_str = f"{sig.weekly_rsi:.2f}" if not np.isnan(sig.weekly_rsi) else "N/A"
                    content_lines.append(f"- **{r.stock_name}** ({r.symbol})\n")
                    content_lines.append(f"  - 价格: {sig.current_price:.3f}\n")
                    content_lines.append(f"  - 日RSI: {daily_rsi_str} | 周RSI: {weekly_rsi_str}\n")
                    content_lines.append(f"  - {sig.signal_detail}\n\n")

        # 买入信号（按日期分组显示）
        if len(buy_signals_by_date) > 0:
            total_buy = sum(len(v) for v in buy_signals_by_date.values())
            content_lines.append(f"### 🟢 买入信号 ({total_buy}条)\n")
            content_lines.append("---\n")

            for date in sorted(buy_signals_by_date.keys(), reverse=True):
                signals_list = buy_signals_by_date[date]
                content_lines.append(f"**日期: {date}** ({len(signals_list)}只)\n")
                for r, sig in signals_list:
                    daily_rsi_str = f"{sig.daily_rsi:.2f}" if not np.isnan(sig.daily_rsi) else "N/A"
                    weekly_rsi_str = f"{sig.weekly_rsi:.2f}" if not np.isnan(sig.weekly_rsi) else "N/A"
                    cyb_rsi_str = f"{sig.index_daily_rsi_cyb:.2f}" if not np.isnan(sig.index_daily_rsi_cyb) else "N/A"
                    sh_rsi_str = f"{sig.index_daily_rsi_sh:.2f}" if not np.isnan(sig.index_daily_rsi_sh) else "N/A"
                    content_lines.append(f"- **{r.stock_name}** ({r.symbol})\n")
                    content_lines.append(f"  - 价格: {sig.current_price:.3f}\n")
                    content_lines.append(f"  - 日RSI: {daily_rsi_str} | 周RSI: {weekly_rsi_str}\n")
                    content_lines.append(f"  - 创业板RSI: {cyb_rsi_str} | 上证RSI: {sh_rsi_str}\n")
                    content_lines.append(f"  - {sig.signal_detail}\n\n")

        # 市场概况
        content_lines.append(f"### 📊 市场概况\n")
        content_lines.append("---\n")
        if len(results) > 0:
            cyb_rsi = results[0].latest_index_daily_rsi_cyb
            sh_rsi = results[0].latest_index_daily_rsi_sh
            cyb_rsi_str = f"{cyb_rsi:.2f}" if not np.isnan(cyb_rsi) else "N/A"
            sh_rsi_str = f"{sh_rsi:.2f}" if not np.isnan(sh_rsi) else "N/A"
            content_lines.append(f"- 创业板指数(399006) RSI: {cyb_rsi_str}\n")
            content_lines.append(f"- 上证指数(000001) RSI: {sh_rsi_str}\n")

        content = "".join(content_lines)

        # 发送钉钉消息
        success = self.dingtalk.send_markdown(title, content, at_all=True)

        if success:
            self.logger.info("钉钉告警发送成功")
        else:
            self.logger.error("钉钉告警发送失败")

    def _save_csv_report(self, results: List[MonitorResult]) -> None:
        """
        保存CSV格式报告

        Args:
            results: 监控结果列表
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = os.path.join(self.output_dir, f'monitor_report_{timestamp}.csv')

        data = []
        for r in results:
            # 如果有信号，每个信号一行
            for sig in r.signals:
                data.append({
                    '股票代码': r.symbol,
                    '股票名称': r.stock_name,
                    '信号日期': sig.signal_date,
                    '信号类型': '卖出' if sig.signal_type == 'sell' else '买入',
                    '信号详情': sig.signal_detail,
                    '收盘价': sig.current_price,
                    '日RSI': sig.daily_rsi if not np.isnan(sig.daily_rsi) else '',
                    '周RSI': sig.weekly_rsi if not np.isnan(sig.weekly_rsi) else '',
                    '创业板指数RSI': sig.index_daily_rsi_cyb if not np.isnan(sig.index_daily_rsi_cyb) else '',
                    '上证指数RSI': sig.index_daily_rsi_sh if not np.isnan(sig.index_daily_rsi_sh) else '',
                    '买入策略ID': str(r.judge_buy_ids),
                    '做T策略ID': str(r.judge_t_ids),
                    '卖出策略ID': str(r.judge_sell_ids),
                })

            # 如果没有信号，也记录一行
            if len(r.signals) == 0:
                data.append({
                    '股票代码': r.symbol,
                    '股票名称': r.stock_name,
                    '信号日期': r.latest_date,
                    '信号类型': '无信号',
                    '信号详情': '',
                    '收盘价': r.latest_price,
                    '日RSI': r.latest_daily_rsi if not np.isnan(r.latest_daily_rsi) else '',
                    '周RSI': r.latest_weekly_rsi if not np.isnan(r.latest_weekly_rsi) else '',
                    '创业板指数RSI': r.latest_index_daily_rsi_cyb if not np.isnan(r.latest_index_daily_rsi_cyb) else '',
                    '上证指数RSI': r.latest_index_daily_rsi_sh if not np.isnan(r.latest_index_daily_rsi_sh) else '',
                    '买入策略ID': str(r.judge_buy_ids),
                    '做T策略ID': str(r.judge_t_ids),
                    '卖出策略ID': str(r.judge_sell_ids),
                })

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"CSV报告: {csv_file}")
        self.logger.info(f"日志文件: {self.log_file}")


# ==================== 定时调度 ====================

def is_trading_day() -> bool:
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


def should_monitor_now() -> bool:
    """
    检查当前时间是否应该执行监控

    Returns:
        bool: 是否应该执行监控
    """
    current_time = datetime.now().strftime('%H:%M')
    return current_time in MONITOR_TIMES


def run_scheduler() -> None:
    """
    运行定时调度器

    每天在指定时间点执行监控：
    - 12:00（中午）- 盘中监控
    - 14:30（尾盘）- 收盘前监控
    - 20:00（晚间）- 复盘监控
    """
    print("=" * 60)
    print("日内监控调度器启动")
    print(f"监控时间点: {MONITOR_TIMES}")
    print(f"监控范围: 最近 {RECENT_TRADING_DAYS} 个交易日")
    print("=" * 60)

    monitor = DailyMonitor()

    while True:
        now = datetime.now()
        current_time = now.strftime('%H:%M')

        if is_trading_day():
            if current_time in MONITOR_TIMES:
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发监控...")
                monitor.run_monitor()
                time.sleep(60)

        time.sleep(60)


def run_once() -> None:
    """
    单次执行监控（用于测试或手动触发）
    """
    monitor = DailyMonitor()
    monitor.run_monitor()


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="日内监控脚本")
    parser.add_argument('--mode', type=str, default='once',
                       choices=['once', 'schedule'],
                       help="运行模式: once(单次执行), schedule(定时调度)")
    parser.add_argument('--config', type=str, default=None,
                       help="股票配置文件路径")

    args = parser.parse_args()

    if args.mode == 'schedule':
        run_scheduler()
    else:
        run_once()