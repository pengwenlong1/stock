# -*- coding: utf-8 -*-
"""
日内监控脚本 (monitor_by_day.py)

【功能说明】
定时监控股票是否达到买卖点指标要求，使用SAR死叉跌破替代均线死叉。

【监控逻辑】
1. 监控当天最新交易日的数据
2. 当发生SAR死叉跌破（绿转红）时触发分析
3. 寻找这一次死叉跌破与上一次死叉跌破之间的最高点作为局部高点
4. 背离检测逻辑：
   - 将当前局部高点(B)与上一次局部高点(A)进行比较
   - 顶背离判定：B的最高价 > A的最高价 且 B的MACD柱 < A的MACD柱
5. 根据买入卖出策略来判断是否符合买卖点

【SAR指标原理】
- SAR绿转红：SAR从价格下方转到上方，趋势从多头转为空头（跌破信号）
- 替代原有的MA5跌破MA10（均线死叉）逻辑

【使用方法】
直接运行脚本：
    python src/runner/monitor_by_day.py

【告警方式】
通过钉钉机器人发送告警消息，使用加签安全模式。

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
from src.tool.sar_strategy import SARStrategy, SARSignal  # 导入SAR策略类
from src.tool.divergence_detector import DivergenceDetector, DivergenceSignal  # 导入背离检测器

try:
    from gm.api import history, set_token, ADJUST_PREV, get_trading_dates, get_instruments, current
except ImportError:
    history = None
    set_token = None
    ADJUST_PREV = None
    get_trading_dates = None
    get_instruments = None
    current = None


# ==================== 监控配置 ====================

# 监控时间点（小时:分钟）
MONITOR_TIMES = ['12:00', '14:30', '20:00']

# 创业板指数和上证指数代码
INDEX_CYB = 'SZSE.399006'   # 创业板指数
INDEX_SH = 'SHSE.000001'    # 上证指数

# 预热天数（用于指标计算）
WARMUP_DAYS = 365

# SAR参数：SAR(10,2,20)
SAR_ACCELERATION = 0.02
SAR_MAXIMUM = 0.20

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
class LocalHighInfo:
    """局部高点信息（用于背离比较）"""
    cross_down_date: str               # SAR死叉日期
    high_date: pd.Timestamp            # 局部高点日期
    high_price: float                  # 局部高点价格（最高价）
    high_macd: float                   # 局部高点MACD柱值
    weekly_rsi: float                  # 局部高点周线RSI


@dataclass
class SARCrossDownInfo:
    """SAR死叉跌破信息"""
    signal_date: str                       # 转折日期
    sar_value: float                       # SAR值
    close_price: float                     # 收盘价
    local_high_date: Optional[str] = None  # 局部高点日期
    local_high_price: float = 0.0          # 局部高点价格
    local_high_weekly_rsi: float = 0.0     # 局部高点周线RSI
    has_daily_divergence: bool = False     # 是否有日线顶背离
    has_weekly_divergence: bool = False    # 是否有周线顶背离
    daily_rsi_at_signal: float = 0.0       # 转折时日线RSI
    weekly_rsi_at_signal: float = 0.0      # 转折时周线RSI
    # 新增：背离比较详情
    prev_high_date: Optional[str] = None   # 上一次局部高点日期
    prev_high_price: float = 0.0           # 上一次局部高点价格
    prev_high_macd: float = 0.0            # 上一次局部高点MACD柱值
    curr_high_macd: float = 0.0            # 当前局部高点MACD柱值


@dataclass
class SignalInfo:
    """信号信息"""
    signal_date: str
    signal_type: str                       # 'buy' 或 'sell'
    signal_detail: str
    current_price: float
    daily_rsi: float
    weekly_rsi: float
    index_daily_rsi_cyb: float
    index_daily_rsi_sh: float
    sar_cross_down_info: Optional[SARCrossDownInfo] = None


@dataclass
class MonitorResult:
    """监控结果"""
    symbol: str
    stock_name: str
    latest_date: str
    latest_price: float
    latest_daily_rsi: float
    latest_weekly_rsi: float
    latest_sar: float                      # 最新SAR值
    latest_sar_trend: str                  # 最新SAR趋势（多头/空头）
    latest_index_daily_rsi_cyb: float
    latest_index_daily_rsi_sh: float
    signals: List[SignalInfo]
    notes: str
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]
    sar_cross_down_events: List[SARCrossDownInfo]  # SAR死叉跌破事件列表


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
        # 先 base64 编码，再 decode 成字符串，最后 URL 编码
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code).decode('utf-8'))
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
            config_path: 股票配置文件路径（默认为 config/stocks_monitor_by_day.csv）
            settings_path: 设置文件路径（默认为 config/settings.json）
            output_dir: 输出目录路径
        """
        # 配置文件路径
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'stocks_monitor_by_day.csv')
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
        加载股票配置（从 stocks_monitor_by_day.csv）

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
            # name列可能为空，尝试获取股票名称
            if 'name' in row and not pd.isna(row['name']) and row['name']:
                stock_name = str(row['name'])
            else:
                stock_name = self._get_stock_name(symbol)

            # 解析策略ID
            judge_buy_ids = self._parse_ids(row.get('judge_buy_ids', ''))
            judge_t_ids = self._parse_ids(row.get('judge_t_ids', ''))
            judge_sell_ids = self._parse_ids(row.get('judge_sell_ids', ''))

            # 监控模式不需要start_date/end_date/initial_capital，使用默认值
            stock_config = StockConfig(
                symbol=symbol,
                stock_name=stock_name,
                start_date='',  # 监控模式不需要
                end_date='',    # 监控模式不需要
                judge_buy_ids=judge_buy_ids,
                judge_t_ids=judge_t_ids,
                judge_sell_ids=judge_sell_ids,
                initial_capital=0,  # 监控模式不需要
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

    def get_realtime_quote(self, symbol: str) -> Optional[Dict]:
        """
        获取实时行情快照（盘中最新数据）

        使用掘金的current函数获取当日实时行情，包括：
        - 当前最新价（盘中close是最新成交价）
        - 当日开盘价、最高价、最低价
        - 当日成交量

        Args:
            symbol: 股票代码

        Returns:
            Dict: 包含 open, high, low, close, volume 的实时行情数据
        """
        if current is None:
            self.logger.warning("未安装掘金量化SDK，无法获取实时行情")
            return None

        try:
            # 获取实时行情快照
            realtime_data = current(symbols=symbol)

            if realtime_data is None or len(realtime_data) == 0:
                self.logger.warning(f"获取实时行情失败: {symbol}")
                return None

            # current返回的是列表，取第一个
            quote = realtime_data[0]

            # 提取需要的字段
            # 使用今天的日期字符串作为datetime，而不是当前时间戳
            # 这样与历史数据的索引格式保持一致
            today_date = datetime.now().strftime('%Y-%m-%d')
            result = {
                'symbol': symbol,
                'open': quote.get('open', 0),
                'high': quote.get('high', 0),
                'low': quote.get('low', 0),
                'close': quote.get('price', quote.get('close', 0)),  # 盘中price是最新价
                'volume': quote.get('volume', 0),
                'datetime': pd.Timestamp(today_date)
            }

            self.logger.info(f"获取实时行情成功: {symbol} | "
                           f"当前价={result['close']:.3f} | "
                           f"最高={result['high']:.3f} | "
                           f"最低={result['low']:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"获取实时行情异常: {symbol} - {e}")
            return None

    def is_trading_time(self) -> bool:
        """
        判断当前是否在交易时段

        交易时段：
        - 上午：09:30 - 11:30
        - 下午：13:00 - 15:00
        - 集合竞价：09:15 - 09:25（可接受订单但不成交）

        Returns:
            bool: 是否在交易时段
        """
        now = datetime.now()
        current_time = now.strftime('%H:%M')

        # 检查是否是工作日
        if now.weekday() >= 5:  # 周六、周日
            return False

        # 检查是否在交易时段
        morning_start = '09:15'  # 包含集合竞价时段
        morning_end = '11:30'
        afternoon_start = '13:00'
        afternoon_end = '15:05'  # 包含收盘后几分钟

        if morning_start <= current_time <= morning_end:
            return True
        if afternoon_start <= current_time <= afternoon_end:
            return True

        return False

    def get_historical_data_with_realtime(self,
                                           symbol: str,
                                           end_date: str) -> Optional[pd.DataFrame]:
        """
        获取历史数据并补充实时行情（如果当前是交易时段）

        Args:
            symbol: 股票代码
            end_date: 结束日期（通常是今天）

        Returns:
            DataFrame: 包含历史数据和当天实时数据
        """
        # 先获取历史数据
        df = self.get_historical_data(symbol, end_date)
        if df is None or df.empty:
            return None

        # 如果当前是交易时段，尝试获取实时行情补充当天数据
        if self.is_trading_time():
            realtime_quote = self.get_realtime_quote(symbol)
            if realtime_quote is not None:
                # 检查历史数据的最后一条是否是今天
                today_str = datetime.now().strftime('%Y-%m-%d')
                last_date_str = df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else ''

                # 如果最后一条不是今天，添加实时数据作为新的一行
                if last_date_str != today_str:
                    # 获取历史数据索引的时区信息
                    tz_info = df.index.tz if hasattr(df.index, 'tz') else None
                    new_datetime = realtime_quote['datetime']
                    # 如果历史数据有时区，给新数据也加上时区
                    if tz_info is not None and new_datetime.tz is None:
                        new_datetime = new_datetime.tz_localize(tz_info)

                    new_row = pd.DataFrame({
                        'open': [realtime_quote['open']],
                        'high': [realtime_quote['high']],
                        'low': [realtime_quote['low']],
                        'close': [realtime_quote['close']],
                        'volume': [realtime_quote['volume']]
                    }, index=[new_datetime])
                    df = pd.concat([df, new_row])
                    self.logger.info(f"添加实时行情数据: {symbol} | 日期={today_str}")
                else:
                    # 如果最后一条是今天，用实时数据更新当天的高低价和收盘价
                    # 注意：开盘价和成交量不需要更新
                    df.loc[df.index[-1], 'high'] = max(df.loc[df.index[-1], 'high'], realtime_quote['high'])
                    df.loc[df.index[-1], 'low'] = min(df.loc[df.index[-1], 'low'], realtime_quote['low'])
                    df.loc[df.index[-1], 'close'] = realtime_quote['close']
                    self.logger.info(f"更新当天实时行情: {symbol} | "
                                   f"最高={realtime_quote['high']:.3f} | "
                                   f"最低={realtime_quote['low']:.3f} | "
                                   f"当前价={realtime_quote['close']:.3f}")

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, SARStrategy, pd.Series, pd.Series]:
        """
        计算技术指标（使用SAR替代MA均线）

        Args:
            df: 历史数据DataFrame

        Returns:
            Tuple: (日线RSI, 周线RSI, SAR策略实例, SAR序列, MACD柱序列)
        """
        close_series = pd.Series(df['close'].values, index=df.index, name='Close')
        high_series = pd.Series(df['high'].values, index=df.index, name='High')
        low_series = pd.Series(df['low'].values, index=df.index, name='Low')

        # 计算日线RSI(6)
        rsi_daily = RSI(period=6, freq='daily')
        daily_rsi = rsi_daily.calculate(close_series)

        # 计算周线RSI(6)
        rsi_weekly = RSI(period=6, freq='weekly')
        weekly_rsi = rsi_weekly.calculate(close_series)

        # 计算SAR指标（替代MA均线）
        sar_strategy = SARStrategy(
            acceleration=SAR_ACCELERATION,
            maximum=SAR_MAXIMUM
        )
        sar_strategy.prepare_data(high_series, low_series, close_series)
        sar_series = sar_strategy._sar_series

        # 计算MACD柱（用于背离检测）
        from src.tool.macd_calculator import MACDCalculator
        macd_calc = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
        macd_calc.prepare_data(close_series)
        macd_series = macd_calc._macd_series

        return daily_rsi, weekly_rsi, sar_strategy, sar_series, macd_series

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

    def detect_sar_cross_down(self,
                               sar_strategy: SARStrategy,
                               sar_signals: List[SARSignal],
                               date: pd.Timestamp) -> bool:
        """
        检测SAR死叉跌破（绿转红）

        Args:
            sar_strategy: SAR策略实例
            sar_signals: SAR转折信号列表
            date: 当前日期

        Returns:
            bool: 是否发生SAR绿转红（跌破）
        """
        date_str = date.strftime('%Y-%m-%d')
        for signal in sar_signals:
            if signal.signal_type == '绿转红' and signal.date.strftime('%Y-%m-%d') == date_str:
                return True
        return False

    def find_local_high_between_sar_cross_downs(self,
                                                    df: pd.DataFrame,
                                                    sar_signals: List[SARSignal],
                                                    current_cross_down_date: pd.Timestamp,
                                                    prev_cross_down_date: Optional[pd.Timestamp]) -> Tuple[Optional[pd.Timestamp], float]:
        """
        寻找两次SAR死叉跌破之间的最高点（局部高点）

        Args:
            df: 历史数据DataFrame
            sar_signals: SAR转折信号列表
            current_cross_down_date: 当前死叉跌破日期
            prev_cross_down_date: 上一次死叉跌破日期（None则从数据开始）

        Returns:
            Tuple: (局部高点日期, 局部高点价格)
        """
        try:
            # 确定搜索范围
            if prev_cross_down_date is None:
                start_date = df.index[0]
            else:
                start_date = prev_cross_down_date

            end_date = current_cross_down_date

            # 筛选范围内的数据
            high_series = df['high']
            mask = (high_series.index >= start_date) & (high_series.index <= end_date)
            filtered_high = high_series[mask]

            if len(filtered_high) == 0:
                return None, 0.0

            # 找到最高点
            max_high_idx = filtered_high.idxmax()
            max_high_value = filtered_high.loc[max_high_idx]

            return max_high_idx, max_high_value

        except Exception as e:
            self.logger.warning(f"寻找局部高点失败: {e}")
            return None, 0.0

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
        监控单只股票（检测SAR死叉跌破信号）

        【监控逻辑】
        1. 当发生SAR死叉跌破（绿转红）时触发分析
        2. 寻找这一次死叉跌破与上一次死叉跌破之间的最高点作为局部高点
        3. 将当前局部高点(B)与上一次局部高点(A)进行比较，判断顶背离：
           - B的最高价 > A的最高价 且 B的MACD柱 < A的MACD柱 → 顶背离
        4. 根据买入卖出策略来判断是否符合买卖点

        Args:
            stock_config: 股票配置
            end_date: 监控日期

        Returns:
            MonitorResult: 监控结果
        """
        symbol = stock_config.symbol
        self.logger.info(f"监控股票: {symbol} ({stock_config.stock_name}) "
                        f"策略: buy={stock_config.judge_buy_ids}, t={stock_config.judge_t_ids}, sell={stock_config.judge_sell_ids}")

        # 获取历史数据（如果是交易时段，会补充实时行情）
        df = self.get_historical_data_with_realtime(symbol, end_date)
        if df is None or df.empty:
            self.logger.warning(f"无法获取数据: {symbol}")
            return None

        # 计算指标（使用SAR替代MA，增加MACD）
        close_series = pd.Series(df['close'].values, index=df.index, name='Close')
        high_series = pd.Series(df['high'].values, index=df.index, name='High')
        daily_rsi, weekly_rsi, sar_strategy, sar_series, macd_series = self.calculate_indicators(df)

        # 获取SAR转折信号（检测整个历史数据的信号）
        sar_signals = sar_strategy.detect_signals(
            df.index[0].strftime('%Y-%m-%d'),
            end_date
        )
        self.logger.info(f"  SAR转折信号总数: {len(sar_signals)}")

        # 只关注绿转红信号（死叉跌破）
        cross_down_signals = [s for s in sar_signals if s.signal_type == '绿转红']
        self.logger.info(f"  SAR死叉跌破信号: {len(cross_down_signals)} 个")

        # 创建策略实例
        strategy = TradingStrategy(
            buy_ids=stock_config.judge_buy_ids,
            t_ids=stock_config.judge_t_ids,
            sell_ids=stock_config.judge_sell_ids
        )

        # 【新增】初始化背离检测器（与回测逻辑一致）
        warmup_start = (pd.Timestamp(end_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
        divergence_detector = DivergenceDetector()
        divergence_detector.prepare_data(symbol, warmup_start, end_date)
        all_divergences = divergence_detector.detect_all_divergences()
        daily_divergences_confirmed = all_divergences.get('daily_top_confirmed', [])
        weekly_divergences_confirmed = all_divergences.get('weekly_top_confirmed', [])
        self.logger.info(f"  日线顶背离生效: {len(daily_divergences_confirmed)} 个")
        self.logger.info(f"  周线顶背离生效: {len(weekly_divergences_confirmed)} 个")

        # 收集信号和SAR死叉跌破事件
        signals = []
        sar_cross_down_events = []

        # 缓存每次SAR死叉对应的局部高点信息（用于背离比较）
        local_high_history: List[LocalHighInfo] = []

        # 分析每个SAR死叉跌破事件
        for i, signal in enumerate(cross_down_signals):
            current_cross_down_date = signal.date

            # 寻找上一次死叉跌破日期
            prev_cross_down_date = None
            if i > 0:
                prev_cross_down_date = cross_down_signals[i-1].date

            # 寻找局部高点（当前死叉与上一次死叉之间的最高点）- 用于显示信息
            local_high_date, local_high_price = self.find_local_high_between_sar_cross_downs(
                df, cross_down_signals, current_cross_down_date, prev_cross_down_date
            )

            # 获取局部高点时的MACD柱值和周线RSI - 用于显示信息
            local_high_macd = 0.0
            local_high_weekly_rsi = 0.0
            if local_high_date is not None:
                local_high_macd = self._get_value_at_date(local_high_date, macd_series)
                local_high_weekly_rsi = self._get_value_at_date(local_high_date, weekly_rsi)

            # 缓存当前局部高点信息（用于下一次比较）
            if local_high_date is not None:
                local_high_info = LocalHighInfo(
                    cross_down_date=current_cross_down_date.strftime('%Y-%m-%d'),
                    high_date=local_high_date,
                    high_price=local_high_price,
                    high_macd=local_high_macd,
                    weekly_rsi=local_high_weekly_rsi
                )
                local_high_history.append(local_high_info)

            # 获取转折时的RSI值
            daily_rsi_at_signal = self._get_value_at_date(current_cross_down_date, daily_rsi)
            weekly_rsi_at_signal = self._get_value_at_date(current_cross_down_date, weekly_rsi)

            # 【注】背离检测在最近两日信号处理中使用DivergenceDetector，这里先设置默认值
            has_daily_div = False
            has_weekly_div = False

            # 创建SAR死叉跌破信息（背离状态会在后面更新）
            cross_down_info = SARCrossDownInfo(
                signal_date=current_cross_down_date.strftime('%Y-%m-%d'),
                sar_value=signal.sar_value,
                close_price=signal.close_price,
                local_high_date=local_high_date.strftime('%Y-%m-%d') if local_high_date else None,
                local_high_price=local_high_price,
                local_high_weekly_rsi=local_high_weekly_rsi,
                has_daily_divergence=has_daily_div,  # 会在后面更新
                has_weekly_divergence=has_weekly_div,  # 会在后面更新
                daily_rsi_at_signal=daily_rsi_at_signal,
                weekly_rsi_at_signal=weekly_rsi_at_signal
            )
            sar_cross_down_events.append(cross_down_info)

            # 只分析最近两个交易日的信号
            recent_dates = self.get_recent_trading_dates(end_date, 2)
            if current_cross_down_date.strftime('%Y-%m-%d') in recent_dates:
                self.logger.info(f"  [SAR死叉跌破] {current_cross_down_date.strftime('%Y-%m-%d')}: "
                               f"SAR={signal.sar_value:.3f}, Close={signal.close_price:.3f}")
                if local_high_date:
                    self.logger.info(f"    局部高点: {local_high_date.strftime('%Y-%m-%d')} @ {local_high_price:.3f}, "
                                   f"周线RSI={local_high_weekly_rsi:.2f}, MACD={local_high_macd:.4f}")
                    # 显示上次局部高点（如果有）
                    if len(local_high_history) > 0:
                        prev_local = local_high_history[-1]
                        self.logger.info(f"    上次高点: {prev_local.high_date.strftime('%Y-%m-%d')} @ {prev_local.high_price:.3f}, MACD={prev_local.high_macd:.4f}")

                # 创建策略状态（关键：需要从上一次SAR死叉跌破后重新累积rsi_flag）
                state = StrategyState()

                # 【关键修改】从上一次SAR死叉跌破之后，遍历更新rsi_flag
                # 上一次SAR死叉跌破后，rsi_flag被重置为0，需要重新累积
                if prev_cross_down_date is not None:
                    # 从上一次SAR死叉跌破的第二天开始遍历（因为卖出后当天不买入，rsi_flag重置）
                    start_date = prev_cross_down_date + pd.Timedelta(days=1)
                    # 找到start_date在df中的索引
                    mask_start = df.index >= start_date
                    if mask_start.any():
                        start_idx = mask_start.argmax()
                    else:
                        start_idx = 0
                else:
                    # 如果是第一个SAR死叉跌破，从数据开始遍历
                    start_idx = 0

                # 遍历到当前SAR死叉跌破日期（包含当天）
                end_date_ts = current_cross_down_date
                mask_end = df.index <= end_date_ts
                if mask_end.any():
                    end_idx = len(df) - 1 - mask_end[::-1].argmax()
                else:
                    end_idx = len(df) - 1

                # 遍历区间，更新rsi_flag（模拟回测的逐日更新逻辑）
                rsi_peak_value = 0.0
                rsi_peak_date = None
                for idx in range(start_idx, end_idx + 1):
                    date = df.index[idx]
                    weekly_rsi_value = weekly_rsi.iloc[idx] if idx < len(weekly_rsi) else np.nan
                    if not np.isnan(weekly_rsi_value):
                        # 更新rsi_flag（与回测逻辑一致）
                        strategy.update_rsi_flag(weekly_rsi_value, state, date)
                        # 记录峰值（用于卖出信号中的峰值信息）
                        if weekly_rsi_value > rsi_peak_value:
                            rsi_peak_value = weekly_rsi_value
                            rsi_peak_date = date

                # 设置rsi峰值信息（如果有的话）
                if rsi_peak_date is not None and rsi_peak_value > 0:
                    state.rsi_peak_date = rsi_peak_date
                    state.rsi_peak_value = rsi_peak_value

                self.logger.info(f"    RSI状态累积: rsi_flag={state.rsi_flag}, "
                               f"峰值={state.rsi_peak_value:.2f} @ {state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else 'N/A'}")

                # 【关键修改】背离检测逻辑：检查在上一次SAR死叉之后是否有背离生效
                # 与回测逻辑一致：背离在confirmation_date时触发，持续保持直到卖出后重置
                has_daily_div = False
                has_weekly_div = False
                daily_div_info = None
                weekly_div_info = None

                # 确定搜索区间：上一次SAR死叉之后到当前SAR死叉
                if prev_cross_down_date is not None:
                    search_start = prev_cross_down_date + pd.Timedelta(days=1)
                else:
                    search_start = df.index[0]
                search_end = current_cross_down_date

                # 检查是否有日线顶背离在区间内生效
                for div in daily_divergences_confirmed:
                    if div.confirmation_date is not None:
                        if search_start <= div.confirmation_date <= search_end:
                            has_daily_div = True
                            daily_div_info = {
                                'date': div.date,
                                'prev_high': div.peak_a_price,
                                'curr_high': div.peak_b_price,
                                'prev_macd': div.peak_a_macd,
                                'curr_macd': div.peak_b_macd
                            }
                            self.logger.info(f"    日线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}, "
                                           f"生效于{div.confirmation_date.strftime('%Y-%m-%d')}")
                            break

                # 检查是否有周线顶背离在区间内生效
                for div in weekly_divergences_confirmed:
                    if div.confirmation_date is not None:
                        if search_start <= div.confirmation_date <= search_end:
                            has_weekly_div = True
                            weekly_div_info = {
                                'date': div.date,
                                'prev_high': div.peak_a_price,
                                'curr_high': div.peak_b_price,
                                'prev_macd': div.peak_a_macd,
                                'curr_macd': div.peak_b_macd
                            }
                            self.logger.info(f"    周线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}, "
                                           f"生效于{div.confirmation_date.strftime('%Y-%m-%d')}")
                            break

                self.logger.info(f"    背离状态: 日线顶背离={has_daily_div}, 周线顶背离={has_weekly_div}")

                # 更新cross_down_info的背离状态
                cross_down_info.has_daily_divergence = has_daily_div
                cross_down_info.has_weekly_divergence = has_weekly_div

                # 设置背离状态（与回测逻辑一致）
                if has_daily_div and daily_div_info is not None:
                    strategy.set_daily_divergence(state, daily_div_info)
                if has_weekly_div and weekly_div_info is not None:
                    strategy.set_weekly_divergence(state, weekly_div_info)

                # 获取指数RSI
                index_rsi_cyb = self.get_index_rsi_at_date(INDEX_CYB, current_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_sh = self.get_index_rsi_at_date(INDEX_SH, current_cross_down_date.strftime('%Y-%m-%d'))

                # 检测卖出信号（SAR死叉跌破作为卖出触发条件）
                sell_signal_obj = strategy.check_sell_signal(
                    date=current_cross_down_date,
                    daily_rsi=daily_rsi_at_signal,
                    weekly_rsi=weekly_rsi_at_signal,
                    ma_cross_down=True,  # SAR死叉跌破触发卖出
                    state=state,
                    position=1.0
                )

                if sell_signal_obj is not None:
                    signal_info = SignalInfo(
                        signal_date=current_cross_down_date.strftime('%Y-%m-%d'),
                        signal_type='sell',
                        signal_detail=f"[{sell_signal_obj.flag.name}] {sell_signal_obj.reason} (SAR死叉跌破触发)",
                        current_price=signal.close_price,
                        daily_rsi=daily_rsi_at_signal,
                        weekly_rsi=weekly_rsi_at_signal,
                        index_daily_rsi_cyb=index_rsi_cyb,
                        index_daily_rsi_sh=index_rsi_sh,
                        sar_cross_down_info=cross_down_info
                    )
                    signals.append(signal_info)
                    self.logger.info(f"    >>> 卖险信号: {sell_signal_obj.reason}")

                # 检测买入信号（寻找买入机会）
                buy_signal_obj = strategy.check_buy_signal(
                    date=current_cross_down_date,
                    daily_rsi=daily_rsi_at_signal,
                    weekly_rsi=weekly_rsi_at_signal,
                    state=state,
                    has_new_cash=True,
                    has_sold_cash=True,
                    index_daily_rsi=index_rsi_cyb,
                    sh_index_daily_rsi=index_rsi_sh
                )

                if buy_signal_obj is not None and buy_signal_obj.triggered:
                    signal_info = SignalInfo(
                        signal_date=current_cross_down_date.strftime('%Y-%m-%d'),
                        signal_type='buy',
                        signal_detail=f"[{'新资金买入' if buy_signal_obj.is_new_cash else '做T买回'}] {buy_signal_obj.reason}",
                        current_price=signal.close_price,
                        daily_rsi=daily_rsi_at_signal,
                        weekly_rsi=weekly_rsi_at_signal,
                        index_daily_rsi_cyb=index_rsi_cyb,
                        index_daily_rsi_sh=index_rsi_sh,
                        sar_cross_down_info=cross_down_info
                    )
                    signals.append(signal_info)
                    self.logger.info(f"    >>> 买入信号: {buy_signal_obj.reason}")

        # 获取最新日期的数据
        last_date = df.index[-1]
        last_date_str = last_date.strftime('%Y-%m-%d')

        latest_daily_rsi = daily_rsi.iloc[-1] if len(daily_rsi) > 0 else np.nan
        latest_weekly_rsi = weekly_rsi.iloc[-1] if len(weekly_rsi) > 0 else np.nan
        latest_sar = sar_series.iloc[-1] if len(sar_series) > 0 else np.nan
        latest_price = df['close'].iloc[-1]
        latest_index_rsi_cyb = self.get_index_rsi(INDEX_CYB, end_date)
        latest_index_rsi_sh = self.get_index_rsi(INDEX_SH, end_date)

        # 判断当前SAR趋势
        latest_sar_trend = '多头' if latest_sar < latest_price else '空头'

        # 构建备注
        notes = f"SAR趋势={latest_sar_trend}; "
        if latest_sar > latest_price:
            notes += "SAR在上方(空头); "

        # 创建监控结果
        result = MonitorResult(
            symbol=symbol,
            stock_name=stock_config.stock_name,
            latest_date=last_date_str,
            latest_price=latest_price,
            latest_daily_rsi=latest_daily_rsi,
            latest_weekly_rsi=latest_weekly_rsi,
            latest_sar=latest_sar,
            latest_sar_trend=latest_sar_trend,
            latest_index_daily_rsi_cyb=latest_index_rsi_cyb,
            latest_index_daily_rsi_sh=latest_index_rsi_sh,
            signals=signals,
            notes=notes,
            judge_buy_ids=stock_config.judge_buy_ids,
            judge_t_ids=stock_config.judge_t_ids,
            judge_sell_ids=stock_config.judge_sell_ids,
            sar_cross_down_events=sar_cross_down_events
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