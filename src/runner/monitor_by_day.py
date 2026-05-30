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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入策略类和工具类
from src.tool.strategy import TradingStrategy, StrategyState, SellSignal, BuySignal, SellFlag, RSI_THRESHOLDS
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

# 监控时间点（交易时间内每30分钟执行）
# 上午：9:30, 10:00, 10:30, 11:00, 11:30
# 下午：13:00, 13:30, 14:00, 14:30, 15:00
MORNING_MONITOR_TIMES = ['09:30', '10:00', '10:30', '11:00', '11:30']
AFTERNOON_MONITOR_TIMES = ['13:00', '13:30', '14:00', '14:30', '15:00']
MONITOR_TIMES = MORNING_MONITOR_TIMES + AFTERNOON_MONITOR_TIMES

# 监控间隔（秒）
MONITOR_INTERVAL = 30 * 60  # 30分钟

# 创业板指数和上证指数代码
INDEX_CYB = 'SZSE.399006'   # 创业板指数
INDEX_SH = 'SHSE.000001'    # 上证指数
INDEX_KC = 'SHSE.000680'    # 科创综指
INDEX_BJ = 'BJSE.899050'    # 北证板块
INDEX_HS_TECH = 'SHSE.513180'  # 恒生科技ETF（替代港股恒生科技）
INDEX_HS_INDEX = 'SZSE.159920'  # 恒生指数ETF（替代港股恒生指数）

# 预热天数（用于指标计算）
# 【重要】月线RSI计算需要足够长的历史数据让EMA状态稳定
# 至少需要24个月线数据（约2年），所以预热天数设置为730天
WARMUP_DAYS = 730

# SAR参数：SAR(10,2,20)
SAR_ACCELERATION = 0.02
SAR_MAXIMUM = 0.20

# 监控最近几个交易日
RECENT_TRADING_DAYS = 2

# RSI历史高点回溯天数（用于检测曾经触发RSI高点但当前RSI已降低的情况）
# 延长回溯天数以覆盖更多历史高点
RSI_HIGH_LOOKBACK_DAYS = 30


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
class RSIHighInfo:
    """历史RSI高点信息（用于持续告警）"""
    high_date: str                     # RSI高点日期
    high_rsi: float                    # RSI高点值
    rsi_type: str                      # RSI类型: '日线' 或 '周线'
    current_rsi: float                 # 当前RSI值
    days_elapsed: int                  # 距离高点的天数


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
    latest_index_daily_rsi_cyb: float      # 创业板指数RSI
    latest_index_daily_rsi_sh: float       # 上证指数RSI
    latest_index_daily_rsi_kc: float       # 科创综指RSI
    latest_index_daily_rsi_bj: float       # 北证板块RSI
    latest_index_daily_rsi_hsTech: float   # 恪生科技ETF RSI
    latest_index_daily_rsi_hsIndex: float  # 恪生指数ETF RSI
    signals: List[SignalInfo]
    notes: str
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]
    sar_cross_down_events: List[SARCrossDownInfo]  # SAR死叉跌破事件列表
    pending_warnings: List[Dict] = field(default_factory=list)  # 待触发卖出信号告警
    rsi_high_warnings: List[RSIHighInfo] = field(default_factory=list)  # 历史RSI高点告警
    index_divergence_warnings: List[Dict] = field(default_factory=list)  # 指数顶背离告警


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
                self.logger.warning(f"get_instruments API未加载，无法获取 {symbol} 名称")
                return "未知"
            instruments = get_instruments(symbol)
            if instruments:
                name = getattr(instruments[0], 'sec_name', None)
                if name:
                    return name
                self.logger.warning(f"{symbol} API返回数据缺少sec_name属性")
                return "未知"
            self.logger.warning(f"get_instruments({symbol}) 返回空列表")
            return "未知"
        except Exception as e:
            self.logger.warning(f"获取股票名称失败: {symbol} - {e}")
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

        如果CSV的name列为空，会自动通过API获取并更新CSV文件。

        Returns:
            股票配置列表（只包含active=1的股票）
        """
        if not os.path.exists(self.config_path):
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return []

        df = pd.read_csv(self.config_path, dtype={'symbol': str})

        # 确保 name 列存在
        if 'name' not in df.columns:
            df['name'] = ''

        # 检查并填充缺失的 name（对所有股票，包括active=0的）
        needs_update = False
        for idx in df.index:
            name_val = df.loc[idx, 'name']
            if pd.isna(name_val) or str(name_val).strip() == '':
                ticker = str(df.loc[idx, 'symbol'])
                symbol = self._add_exchange_prefix(ticker)
                stock_name = self._get_stock_name(symbol)
                if stock_name and stock_name != "未知":
                    df.loc[idx, 'name'] = stock_name
                    self.logger.info(f"自动填充股票名称: {ticker} -> {stock_name}")
                    needs_update = True

        # 如果有更新，保存回CSV文件
        if needs_update:
            df.to_csv(self.config_path, index=False, encoding='utf-8')
            self.logger.info(f"已更新股票名称到CSV文件: {self.config_path}")

        stocks = []

        for _, row in df.iterrows():
            # 只加载active=1的股票
            # active字段支持分号分隔（如"1;2"），只要包含1就监控
            active_str = str(row['active']) if not pd.isna(row['active']) else '0'
            active_values = [int(x.strip()) for x in active_str.replace(';', ',').split(',') if x.strip().isdigit()]
            is_active = 1 in active_values
            if not is_active:
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

    def is_market_day(self) -> bool:
        """
        判断当前是否是交易日（包括盘前盘后时间）

        用于判断是否应该获取实时行情，只要当天是交易日就可以获取。

        Returns:
            bool: 是否是交易日
        """
        now = datetime.now()

        # 检查是否是工作日
        if now.weekday() >= 5:  # 周六、周日
            return False

        current_time = now.strftime('%H:%M')

        # 在工作日的 08:00 - 16:00 之间认为是交易日相关时间
        if '08:00' <= current_time <= '16:00':
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

        # 如果当前是交易日相关时间（08:00-16:00），尝试获取实时行情补充当天数据
        if self.is_market_day():
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
            Tuple: (日线RSI, 周线RSI, 月线RSI, SAR策略实例, SAR序列, MACD柱序列)
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

        # 计算月线RSI(6)（用于buy_id=10和buy_id=11）
        rsi_monthly = RSI(period=6, freq='monthly')
        monthly_rsi = rsi_monthly.calculate(close_series)

        # 计算SAR指标（替代MA均线）
        sar_strategy = SARStrategy(
            acceleration=SAR_ACCELERATION,
            maximum=SAR_MAXIMUM
        )
        sar_strategy.prepare_data(high_series, low_series, close_series)
        sar_series = sar_strategy._sar_series

        # 计算MACD（用于背离检测和死叉检测）
        from src.tool.macd_calculator import MACDCalculator
        macd_calc = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
        macd_calc.prepare_data(close_series)
        macd_series = macd_calc._macd_series
        dif_series = macd_calc._dif_series
        dea_series = macd_calc._dea_series

        return daily_rsi, weekly_rsi, monthly_rsi, sar_strategy, sar_series, macd_series, dif_series, dea_series, macd_calc

    def get_index_rsi(self, index_symbol: str, end_date: str) -> float:
        """
        获取指数RSI值（包含当天实时数据）

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

            # 【关键修改】如果是交易时段，尝试获取实时行情补充当天数据
            if self.is_trading_time():
                realtime_quote = self.get_realtime_quote(index_symbol)
                if realtime_quote is not None:
                    today_str = end_date
                    if len(close_series) > 0:
                        # 安全获取最后一条数据的日期字符串
                        last_idx = close_series.index[-1]
                        last_date_str = last_idx.strftime('%Y-%m-%d') if hasattr(last_idx, 'strftime') else str(last_idx)[:10]
                        if last_date_str != today_str:
                            new_datetime = realtime_quote['datetime']
                            # 处理timezone问题：如果index有时区，给新datetime也加上
                            if hasattr(close_series.index, 'tz') and close_series.index.tz is not None:
                                if new_datetime.tz is None:
                                    new_datetime = new_datetime.tz_localize(close_series.index.tz)
                            close_series.loc[new_datetime] = realtime_quote['close']
                            self.logger.info(f"指数实时行情补充: {index_symbol} | 日期={today_str} | 价格={realtime_quote['close']:.3f}")
                        else:
                            # 如果最后一条是今天，用实时数据更新
                            close_series.iloc[-1] = realtime_quote['close']

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
        获取指数在指定日期的RSI值（包含当天实时数据）

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

            # 【关键修改】如果目标日期是今天且是交易时段，尝试获取实时行情
            target_date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)
            today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            if target_date_str == today_str and self.is_trading_time():
                realtime_quote = self.get_realtime_quote(index_symbol)
                if realtime_quote is not None:
                    # 如果最后一条数据不是今天，添加实时数据
                    if len(close_series) > 0:
                        # 安全获取最后一条数据的日期字符串
                        last_idx = close_series.index[-1]
                        last_date_str = last_idx.strftime('%Y-%m-%d') if hasattr(last_idx, 'strftime') else str(last_idx)[:10]
                        if last_date_str != today_str:
                            new_datetime = realtime_quote['datetime']
                            # 处理timezone问题：如果index有时区，给新datetime也加上
                            if hasattr(close_series.index, 'tz') and close_series.index.tz is not None:
                                if new_datetime.tz is None:
                                    new_datetime = new_datetime.tz_localize(close_series.index.tz)
                            close_series.loc[new_datetime] = realtime_quote['close']
                        else:
                            # 如果最后一条是今天，用实时数据更新
                            close_series.iloc[-1] = realtime_quote['close']

            rsi_calculator = RSI(period=6, freq='daily')
            rsi_series = rsi_calculator.calculate(close_series)

            # 找到目标日期的RSI值（安全处理日期格式）
            # 如果目标日期是非交易日，返回最近一个交易日的RSI
            matching = rsi_series[rsi_series.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)[:10]) == target_date_str]
            if len(matching) > 0:
                return matching.iloc[-1]

            # 如果目标日期没有数据（可能是非交易日），返回最后一个有效RSI值
            if len(rsi_series) > 0:
                last_valid_rsi = rsi_series.dropna()
                if len(last_valid_rsi) > 0:
                    # 返回最近的交易日的RSI
                    return last_valid_rsi.iloc[-1]

            return np.nan

        except Exception as e:
            self.logger.warning(f"获取指数RSI失败: {index_symbol} @ {target_date} - {e}")
            return np.nan

    def get_index_macd_at_date(self, index_symbol: str, target_date: str) -> Tuple[float, float]:
        """
        获取指数在指定日期的MACD值（DIF和DEA）

        Args:
            index_symbol: 指数代码
            target_date: 目标日期

        Returns:
            Tuple[float, float]: (DIF值, DEA值)
        """
        from src.tool.macd_calculator import MACDCalculator

        if history is None:
            return np.nan, np.nan

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
                return np.nan, np.nan

            index_data['eob'] = pd.to_datetime(index_data['eob'])
            close_series = pd.Series(index_data['close'].values, index=index_data['eob'])

            # 计算MACD
            macd_calc = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
            macd_calc.prepare_data(close_series)

            # 获取最后一天的DIF和DEA值
            dif_series = macd_calc._dif_series
            dea_series = macd_calc._dea_series

            if dif_series is not None and len(dif_series) > 0:
                return dif_series.iloc[-1], dea_series.iloc[-1]

            return np.nan, np.nan

        except Exception as e:
            self.logger.warning(f"获取指数MACD失败: {index_symbol} @ {target_date} - {e}")
            return np.nan, np.nan

    def detect_index_macd_cross_down(self, index_symbol: str, target_date: str) -> bool:
        """
        检测指数MACD是否发生死叉（用于 sell_id=7,8,9）

        Args:
            index_symbol: 指数代码
            target_date: 目标日期

        Returns:
            bool: 是否发生指数MACD死叉

        【逻辑说明】
        指数MACD死叉：DIF（快线）从上方穿过DEA（慢线）下方
        """
        from src.tool.macd_calculator import MACDCalculator

        if history is None:
            return False

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

            if index_data is None or len(index_data) < 2:
                return False

            index_data['eob'] = pd.to_datetime(index_data['eob'])
            close_series = pd.Series(index_data['close'].values, index=index_data['eob'])

            # 计算MACD
            macd_calc = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
            macd_calc.prepare_data(close_series)

            dif_series = macd_calc._dif_series
            dea_series = macd_calc._dea_series

            if dif_series is None or len(dif_series) < 2:
                return False

            # 获取当前和前一天的DIF/DEA
            current_dif = dif_series.iloc[-1]
            current_dea = dea_series.iloc[-1]
            prev_dif = dif_series.iloc[-2]
            prev_dea = dea_series.iloc[-2]

            if np.isnan(current_dif) or np.isnan(current_dea) or np.isnan(prev_dif) or np.isnan(prev_dea):
                return False

            # MACD死叉：DIF从上方穿过DEA下方
            if prev_dif >= prev_dea and current_dif < current_dea:
                self.logger.info(f"  [{index_symbol}] MACD死叉检测: DIF={current_dif:.4f}<DEA={current_dea:.4f} (前一天: DIF={prev_dif:.4f}>=DEA={prev_dea:.4f})")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"检测指数MACD死叉失败: {index_symbol} @ {target_date} - {e}")
            return False

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

        # 【关键】先获取历史数据（不包括今天），计算昨天的SAR值
        # 这是今天判断SAR死叉的基准值
        df_history = self.get_historical_data(symbol, end_date)
        if df_history is None or df_history.empty:
            self.logger.warning(f"无法获取历史数据: {symbol}")
            return None

        # 计算历史数据的SAR（不包括今天），获取昨天的SAR值作为今天的判断基准
        close_series_history = pd.Series(df_history['close'].values, index=df_history.index, name='Close')
        high_series_history = pd.Series(df_history['high'].values, index=df_history.index, name='High')
        low_series_history = pd.Series(df_history['low'].values, index=df_history.index, name='Low')

        sar_strategy_history = SARStrategy(
            acceleration=SAR_ACCELERATION,
            maximum=SAR_MAXIMUM
        )
        sar_strategy_history.prepare_data(high_series_history, low_series_history, close_series_history)
        sar_series_history = sar_strategy_history._sar_series

        # 获取昨天的SAR值（作为今天判断基准）
        yesterday_sar_baseline = sar_series_history.iloc[-1] if len(sar_series_history) > 0 else np.nan
        self.logger.info(f"  [SAR基准值] 昨天SAR={yesterday_sar_baseline:.3f}（基于历史数据计算，不包含今天）")

        # 获取历史数据（如果是交易时段，会补充实时行情）
        df = self.get_historical_data_with_realtime(symbol, end_date)
        if df is None or df.empty:
            self.logger.warning(f"无法获取数据: {symbol}")
            return None

        # 计算指标（使用SAR替代MA，增加MACD）
        close_series = pd.Series(df['close'].values, index=df.index, name='Close')
        high_series = pd.Series(df['high'].values, index=df.index, name='High')
        daily_rsi, weekly_rsi, monthly_rsi, sar_strategy, sar_series, macd_series, dif_series, dea_series, macd_calc = self.calculate_indicators(df)

        # 获取SAR转折信号（检测整个历史数据的信号）
        sar_signals = sar_strategy.detect_signals(
            df.index[0].strftime('%Y-%m-%d'),
            end_date
        )
        self.logger.info(f"  SAR转折信号总数: {len(sar_signals)}")

        # 只关注绿转红信号（死叉跌破）
        cross_down_signals = [s for s in sar_signals if s.signal_type == '绿转红']
        self.logger.info(f"  SAR死叉跌破信号: {len(cross_down_signals)} 个")

        # 【新增】找到最近一次红转绿信号（买入信号）的日期，用于监控模式的背离检测
        cross_up_signals = [s for s in sar_signals if s.signal_type == '红转绿']
        last_buy_signal_date = None
        if len(cross_up_signals) > 0:
            last_buy_signal_date = cross_up_signals[-1].date  # 最近一次买入信号日期
            self.logger.info(f"  最近一次买入信号（红转绿）: {last_buy_signal_date.strftime('%Y-%m-%d')}")

        # 【新增】检测MACD死叉信号（用于sell_id=2或3的触发条件）
        # MACD日线跌破死叉 = MACD死叉（DIF下穿DEA）+ 价格跌破SAR
        macd_cross_signals = macd_calc.detect_cross_signals(
            df.index[0].strftime('%Y-%m-%d'),
            end_date
        )
        macd_cross_down_signals = [s for s in macd_cross_signals if s.signal_type == '死叉']
        self.logger.info(f"  MACD死叉信号: {len(macd_cross_down_signals)} 个")

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
        # 【修改】监控模式传入买入信号日期，只判断买入信号后最新高点的顶背离
        all_divergences = divergence_detector.detect_all_divergences(last_buy_signal_date)
        daily_divergences_confirmed = all_divergences.get('daily_top_confirmed', [])
        weekly_divergences_confirmed = all_divergences.get('weekly_top_confirmed', [])
        self.logger.info(f"  日线顶背离生效: {len(daily_divergences_confirmed)} 个")
        self.logger.info(f"  周线顶背离生效: {len(weekly_divergences_confirmed)} 个")

        # 【新增】初始化指数背离检测器（用于 sell_id=4,5,6）
        sh_index_daily_divergences = []
        sh_index_weekly_divergences = []
        cyb_index_daily_divergences = []
        cyb_index_weekly_divergences = []
        kc_index_daily_divergences = []
        kc_index_weekly_divergences = []

        # 【修复】指数背离检测器应该始终初始化，用于独立的指数顶背离告警功能
        # 不依赖于个股的 sell_ids 配置
        self.logger.info("  初始化指数背离检测器...")

        # 上证指数背离检测器
        sh_div_detector = DivergenceDetector()
        sh_div_detector.prepare_data(INDEX_SH, warmup_start, end_date)
        sh_all_div = sh_div_detector.detect_all_divergences()
        sh_index_daily_divergences = sh_all_div.get('daily_top_confirmed', [])
        sh_index_weekly_divergences = sh_all_div.get('weekly_top_confirmed', [])
        self.logger.info(f"  上证指数日线顶背离生效: {len(sh_index_daily_divergences)} 个")
        self.logger.info(f"  上证指数周线顶背离生效: {len(sh_index_weekly_divergences)} 个")

        # 创业板指数背离检测器
        cyb_div_detector = DivergenceDetector()
        cyb_div_detector.prepare_data(INDEX_CYB, warmup_start, end_date)
        cyb_all_div = cyb_div_detector.detect_all_divergences()
        cyb_index_daily_divergences = cyb_all_div.get('daily_top_confirmed', [])
        cyb_index_weekly_divergences = cyb_all_div.get('weekly_top_confirmed', [])
        self.logger.info(f"  创业板指数日线顶背离生效: {len(cyb_index_daily_divergences)} 个")
        self.logger.info(f"  创业板指数周线顶背离生效: {len(cyb_index_weekly_divergences)} 个")

        # 科创板指数背离检测器
        kc_div_detector = DivergenceDetector()
        kc_div_detector.prepare_data(INDEX_KC, warmup_start, end_date)
        kc_all_div = kc_div_detector.detect_all_divergences()
        kc_index_daily_divergences = kc_all_div.get('daily_top_confirmed', [])
        kc_index_weekly_divergences = kc_all_div.get('weekly_top_confirmed', [])
        self.logger.info(f"  科创板指数日线顶背离生效: {len(kc_index_daily_divergences)} 个")
        self.logger.info(f"  科创板指数周线顶背离生效: {len(kc_index_weekly_divergences)} 个")

        # 恒生科技ETF背离检测器
        hs_tech_daily_divergences = []
        hs_tech_weekly_divergences = []
        try:
            hs_tech_div_detector = DivergenceDetector()
            hs_tech_div_detector.prepare_data(INDEX_HS_TECH, warmup_start, end_date)
            hs_tech_all_div = hs_tech_div_detector.detect_all_divergences()
            hs_tech_daily_divergences = hs_tech_all_div.get('daily_top_confirmed', [])
            hs_tech_weekly_divergences = hs_tech_all_div.get('weekly_top_confirmed', [])
            self.logger.info(f"  恒生科技ETF日线顶背离生效: {len(hs_tech_daily_divergences)} 个")
            self.logger.info(f"  恒生科技ETF周线顶背离生效: {len(hs_tech_weekly_divergences)} 个")
        except Exception as e:
            self.logger.warning(f"  恒生科技ETF背离检测器初始化失败: {e}")

        # 恒生指数ETF背离检测器
        hs_index_daily_divergences = []
        hs_index_weekly_divergences = []
        try:
            hs_index_div_detector = DivergenceDetector()
            hs_index_div_detector.prepare_data(INDEX_HS_INDEX, warmup_start, end_date)
            hs_index_all_div = hs_index_div_detector.detect_all_divergences()
            hs_index_daily_divergences = hs_index_all_div.get('daily_top_confirmed', [])
            hs_index_weekly_divergences = hs_index_all_div.get('weekly_top_confirmed', [])
            self.logger.info(f"  恒生指数ETF日线顶背离生效: {len(hs_index_daily_divergences)} 个")
            self.logger.info(f"  恒生指数ETF周线顶背离生效: {len(hs_index_weekly_divergences)} 个")
        except Exception as e:
            self.logger.warning(f"  恒生指数ETF背离检测器初始化失败: {e}")

        self.logger.info("  指数背离检测器初始化完成")

        # 收集信号和SAR死叉跌破事件
        signals = []
        sar_cross_down_events = []

        # 缓存每次SAR死叉对应的局部高点信息（用于背离比较）
        local_high_history: List[LocalHighInfo] = []

        self.logger.info(f"  开始历史买卖信号检测: SAR死叉信号数={len(cross_down_signals)}, MACD死叉信号数={len(macd_cross_down_signals)}")

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

                # 【关键修改】追踪中间是否发生过卖出，用于确定背离检测的起始点
                last_sell_date_in_range = None  # 上一次卖出日期（如果有）

                # 【关键新增】先检查上一次SAR死叉是否实际触发卖出
                # 如果上一次SAR死叉之前有背离确认，那次SAR死叉应该触发卖出
                if prev_cross_down_date is not None:
                    # 检查上一次SAR死叉之前是否有背离确认
                    div_search_start = df.index[0]  # 从数据开始
                    if i > 1:
                        # 更早的上一次SAR死叉
                        prev_prev_cross_down = cross_down_signals[i-2].date if i >= 2 else None
                        if prev_prev_cross_down is not None:
                            div_search_start = prev_prev_cross_down + pd.Timedelta(days=1)
                    div_search_end = prev_cross_down_date

                    temp_has_daily_div = False
                    for div in daily_divergences_confirmed:
                        if div.confirmation_date is not None:
                            if div_search_start <= div.confirmation_date <= div_search_end:
                                temp_has_daily_div = True
                                self.logger.info(f"    上一次SAR死叉({prev_cross_down_date.strftime('%Y-%m-%d')})前有背离确认({div.confirmation_date.strftime('%Y-%m-%d')})")
                                break

                    if temp_has_daily_div:
                        # 上一次SAR死叉应该触发卖出
                        last_sell_date_in_range = prev_cross_down_date
                        state.daily_divergence_flag = 0  # 已卖出，重置
                        self.logger.info(f"    上一次SAR死叉实际卖出: {prev_cross_down_date.strftime('%Y-%m-%d')}")

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
                # 【关键新增】同时追踪中间是否有卖出发生，卖出后重置state
                rsi_peak_value = 0.0
                rsi_peak_date = None
                for idx in range(start_idx, end_idx + 1):
                    date = df.index[idx]
                    weekly_rsi_value = weekly_rsi.iloc[idx] if idx < len(weekly_rsi) else np.nan
                    daily_rsi_value = daily_rsi.iloc[idx] if idx < len(daily_rsi) else np.nan

                    if not np.isnan(weekly_rsi_value):
                        # 更新rsi_flag（与回测逻辑一致）
                        strategy.update_rsi_flag(weekly_rsi_value, state, date)
                        # 记录峰值（用于卖出信号中的峰值信息）
                        if weekly_rsi_value > rsi_peak_value:
                            rsi_peak_value = weekly_rsi_value
                            rsi_peak_date = date

                    # 【关键新增】检查这一天是否有SAR死叉且触发了卖出
                    # 如果中间有卖出，要重置state，并记录卖出日期
                    if idx > start_idx:  # 不检查起始点（上一次SAR死叉当天）
                        # 检查这一天是否是某次SAR死叉
                        for prev_signal in cross_down_signals[:i]:  # 只检查当前之前的SAR死叉
                            if prev_signal.date.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'):
                                # 这一天有SAR死叉，检查是否触发了卖出
                                # 【关键】先检测当时的背离状态（如果有的话）
                                # 检查在prev_cross_down_date到当前日期之间是否有背离生效
                                if prev_cross_down_date is not None:
                                    div_search_start = prev_cross_down_date + pd.Timedelta(days=1)
                                else:
                                    div_search_start = df.index[0]
                                div_search_end = date

                                # 检查是否有背离在那时候生效
                                temp_has_daily_div = False
                                for div in daily_divergences_confirmed:
                                    if div.confirmation_date is not None:
                                        if div_search_start <= div.confirmation_date <= div_search_end:
                                            temp_has_daily_div = True
                                            break

                                # 如果有背离生效，设置背离标志
                                if temp_has_daily_div:
                                    state.daily_divergence_flag = 1

                                # 使用当时的state检查卖出信号
                                temp_sell = strategy.check_sell_signal(
                                    date=date,
                                    daily_rsi=daily_rsi_value,
                                    weekly_rsi=weekly_rsi_value,
                                    sar_cross_down=True,
                                    macd_cross_down=False,
                                    state=state,
                                    position=1.0,
                                    sell_id=stock_config.judge_sell_ids[0] if stock_config.judge_sell_ids else 1
                                )
                                if temp_sell is not None:
                                    # 有卖出，重置state并记录卖出日期
                                    strategy.reset_after_sell(state, temp_sell.flag, date)
                                    last_sell_date_in_range = date
                                    rsi_peak_value = 0.0
                                    rsi_peak_date = None
                                    self.logger.info(f"    中间卖出: {date.strftime('%Y-%m-%d')}, "
                                                   f"重置state, 背离={state.daily_divergence_flag}")
                                break

                # 设置rsi峰值信息（如果有的话）
                if rsi_peak_date is not None and rsi_peak_value > 0:
                    state.rsi_peak_date = rsi_peak_date
                    state.rsi_peak_value = rsi_peak_value

                self.logger.info(f"    RSI状态累积: rsi_flag={state.rsi_flag}, "
                               f"峰值={state.rsi_peak_value:.2f} @ {state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else 'N/A'}")

                # 【关键修改】背离检测逻辑：检查在上一次SAR死叉之后是否有背离生效
                # 与回测逻辑一致：背离在confirmation_date时触发，持续保持直到卖出后重置
                # 【关键新增】如果有中间卖出，背离检测从最后一次卖出后开始
                has_daily_div = False
                has_weekly_div = False
                daily_div_info = None
                weekly_div_info = None

                # 确定搜索区间：如果有中间卖出，从最后一次卖出后开始；否则从上一次SAR死叉后开始
                if last_sell_date_in_range is not None:
                    # 有中间卖出，背离检测从卖出后开始（因为卖出后背离已重置）
                    search_start = last_sell_date_in_range + pd.Timedelta(days=1)
                    self.logger.info(f"    背离检测起始点（中间卖出后）: {search_start.strftime('%Y-%m-%d')}")
                elif prev_cross_down_date is not None:
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
                                'prev_high_date': div.peak_a_date,
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
                                'prev_high_date': div.peak_a_date,
                                'prev_high': div.peak_a_price,
                                'curr_high': div.peak_b_price,
                                'prev_macd': div.peak_a_macd,
                                'curr_macd': div.peak_b_macd
                            }
                            self.logger.info(f"    周线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}, "
                                           f"生效于{div.confirmation_date.strftime('%Y-%m-%d')}")
                            break

                self.logger.info(f"    背离状态: 日线顶背离={has_daily_div}, 周线顶背离={has_weekly_div}")

                # 更新state的背离状态（关键：check_sell_signal需要读取state中的背离标志）
                if has_daily_div:
                    state.daily_divergence_flag = 1
                    state.daily_divergence_info = daily_div_info
                if has_weekly_div:
                    state.weekly_divergence_flag = 1
                    state.weekly_divergence_info = weekly_div_info

                # 更新cross_down_info的背离状态
                cross_down_info.has_daily_divergence = has_daily_div
                cross_down_info.has_weekly_divergence = has_weekly_div

                # 设置背离状态（与回测逻辑一致）
                if has_daily_div and daily_div_info is not None:
                    strategy.set_daily_divergence(state, daily_div_info)
                if has_weekly_div and weekly_div_info is not None:
                    strategy.set_weekly_divergence(state, weekly_div_info)

                # 【新增】设置指数背离状态（用于 sell_id=4,5,6）
                if any(sid in [4, 5, 6] for sid in stock_config.judge_sell_ids):
                    # 检查上证指数背离（用于 sell_id=4）
                    for div in sh_index_daily_divergences:
                        if div.confirmation_date is not None:
                            if search_start <= div.confirmation_date <= search_end:
                                sh_daily_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_sh_index_divergence(state, daily_info=sh_daily_div_info)
                                self.logger.info(f"    上证指数日线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break
                    for div in sh_index_weekly_divergences:
                        if div.confirmation_date is not None:
                            if search_start <= div.confirmation_date <= search_end:
                                sh_weekly_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_sh_index_divergence(state, weekly_info=sh_weekly_div_info)
                                self.logger.info(f"    上证指数周线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break

                    # 检查创业板指数背离（用于 sell_id=5）
                    for div in cyb_index_daily_divergences:
                        if div.confirmation_date is not None:
                            if search_start <= div.confirmation_date <= search_end:
                                cyb_daily_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_cyb_index_divergence(state, daily_info=cyb_daily_div_info)
                                self.logger.info(f"    创业板指数日线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break
                    for div in cyb_index_weekly_divergences:
                        if div.confirmation_date is not None:
                            if search_start <= div.confirmation_date <= search_end:
                                cyb_weekly_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_cyb_index_divergence(state, weekly_info=cyb_weekly_div_info)
                                self.logger.info(f"    创业板指数周线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break

                    # 检查科创板指数背离（用于 sell_id=6）
                    for div in kc_index_daily_divergences:
                        if div.confirmation_date is not None:
                            if search_start <= div.confirmation_date <= search_end:
                                kc_daily_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_kc_index_divergence(state, daily_info=kc_daily_div_info)
                                self.logger.info(f"    科创板指数日线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break
                    for div in kc_index_weekly_divergences:
                        if div.confirmation_date is not None:
                            if search_start <= div.confirmation_date <= search_end:
                                kc_weekly_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_kc_index_divergence(state, weekly_info=kc_weekly_div_info)
                                self.logger.info(f"    科创板指数周线顶背离生效: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break

                # 获取指数RSI
                index_rsi_cyb = self.get_index_rsi_at_date(INDEX_CYB, current_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_sh = self.get_index_rsi_at_date(INDEX_SH, current_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_kc = self.get_index_rsi_at_date(INDEX_KC, current_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_bj = self.get_index_rsi_at_date(INDEX_BJ, current_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_hsTech = self.get_index_rsi_at_date(INDEX_HS_TECH, current_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_hsIndex = self.get_index_rsi_at_date(INDEX_HS_INDEX, current_cross_down_date.strftime('%Y-%m-%d'))

                # 获取当前日期在DataFrame中的索引位置（用于获取月线RSI）
                current_idx = df.index.get_loc(current_cross_down_date) if current_cross_down_date in df.index else -1

                # 获取月线RSI
                monthly_rsi_at_signal = monthly_rsi.iloc[current_idx] if current_idx >= 0 and current_idx < len(monthly_rsi) else np.nan

                # 检测卖出信号
                # 根据配置的sell_ids和当前日期的触发条件检测
                # sell_id=1: SAR死叉触发
                # sell_id=2或3: MACD日线跌破死叉触发（MACD死叉 + 价格跌破SAR)
                sell_signal_obj = None
                for sell_id in stock_config.judge_sell_ids:
                    # 根据sell_id选择触发条件
                    if sell_id == 1:
                        # sell_id=1: SAR死叉跌破作为触发条件
                        trigger_sar = True
                        trigger_macd = False
                        trigger_name = "SAR死叉跌破"
                    elif sell_id in [2, 3]:
                        # sell_id=2或3: 需要检测MACD死叉 + 价格跌破SAR的组合
                        # 检查当前日期是否发生了MACD死叉（DIF下穿DEA）
                        current_idx = df.index.get_loc(current_cross_down_date) if current_cross_down_date in df.index else -1
                        if current_idx >= 1:
                            prev_dif = dif_series.iloc[current_idx - 1]
                            curr_dif = dif_series.iloc[current_idx]
                            prev_dea = dea_series.iloc[current_idx - 1]
                            curr_dea = dea_series.iloc[current_idx]
                            # MACD死叉判断: 前一天DIF>=DEA，当天DIF<DEA
                            is_macd_cross_down = (prev_dif >= prev_dea) and (curr_dif < curr_dea)
                            # 价格跌破SAR判断: 当天最低价低于昨天SAR
                            yesterday_sar = sar_series.iloc[current_idx - 1]
                            today_low = df['low'].iloc[current_idx]
                            is_price_break_sar = today_low < yesterday_sar
                            # 组合条件: MACD死叉 + 价格跌破SAR
                            trigger_macd = is_macd_cross_down and is_price_break_sar
                            trigger_sar = False
                            trigger_name = "MACD日线跌破死叉"
                            if trigger_macd:
                                self.logger.info(f"    MACD日线跌破死叉触发: DIF={curr_dif:.4f}<DEA={curr_dea:.4f}, "
                                               f"最低价={today_low:.3f}<SAR={yesterday_sar:.3f}")
                        else:
                            trigger_macd = False
                            trigger_sar = False
                            trigger_name = "MACD日线跌破死叉(条件不满足)"
                    elif sell_id in [4, 5, 6]:
                        # sell_id=4,5,6: 个股MACD日线跌破死叉触发 + 指数顶背离判断
                        current_idx = df.index.get_loc(current_cross_down_date) if current_cross_down_date in df.index else -1
                        if current_idx >= 1:
                            prev_dif = dif_series.iloc[current_idx - 1]
                            curr_dif = dif_series.iloc[current_idx]
                            prev_dea = dea_series.iloc[current_idx - 1]
                            curr_dea = dea_series.iloc[current_idx]
                            yesterday_sar = sar_series.iloc[current_idx - 1]
                            today_low = df['low'].iloc[current_idx]
                            is_macd_cross_down = (prev_dif >= prev_dea) and (curr_dif < curr_dea)
                            is_price_break_sar = today_low < yesterday_sar
                            trigger_macd = is_macd_cross_down and is_price_break_sar
                            trigger_sar = False
                            trigger_name = "MACD日线跌破死叉"
                        else:
                            trigger_macd = False
                            trigger_sar = False
                            trigger_name = "MACD日线跌破死叉(条件不满足)"
                    elif sell_id in [7, 8, 9]:
                        # sell_id=7,8,9: 指数MACD日线死叉触发 + 指数顶背离判断
                        # 检测对应指数的MACD死叉
                        if sell_id == 7:
                            trigger_index_macd = self.detect_index_macd_cross_down(INDEX_SH, current_cross_down_date.strftime('%Y-%m-%d'))
                            trigger_name = "上证指数MACD跌破死叉"
                        elif sell_id == 8:
                            trigger_index_macd = self.detect_index_macd_cross_down(INDEX_CYB, current_cross_down_date.strftime('%Y-%m-%d'))
                            trigger_name = "创业板指数MACD跌破死叉"
                        elif sell_id == 9:
                            trigger_index_macd = self.detect_index_macd_cross_down(INDEX_KC, current_cross_down_date.strftime('%Y-%m-%d'))
                            trigger_name = "科创板指数MACD跌破死叉"
                        else:
                            trigger_index_macd = False
                            trigger_name = "未知指数MACD"
                        trigger_sar = False
                        trigger_macd = trigger_index_macd  # 使用指数MACD作为触发条件
                        if trigger_index_macd:
                            self.logger.info(f"    {trigger_name}触发")
                    else:
                        # 未知的sell_id，跳过
                        continue

                    # 计算指数MACD触发状态（用于传递给check_sell_signal）
                    sh_index_macd_cross_down = self.detect_index_macd_cross_down(INDEX_SH, current_cross_down_date.strftime('%Y-%m-%d')) if sell_id in [7] else False
                    cyb_index_macd_cross_down = self.detect_index_macd_cross_down(INDEX_CYB, current_cross_down_date.strftime('%Y-%m-%d')) if sell_id in [8] else False
                    kc_index_macd_cross_down = self.detect_index_macd_cross_down(INDEX_KC, current_cross_down_date.strftime('%Y-%m-%d')) if sell_id in [9] else False

                    # 调用check_sell_signal检测卖出信号
                    sell_signal_obj = strategy.check_sell_signal(
                        date=current_cross_down_date,
                        daily_rsi=daily_rsi_at_signal,
                        weekly_rsi=weekly_rsi_at_signal,
                        sar_cross_down=trigger_sar,
                        macd_cross_down=trigger_macd,
                        state=state,
                        position=1.0,
                        sell_id=sell_id,
                        sh_index_macd_cross_down=sh_index_macd_cross_down,
                        cyb_index_macd_cross_down=cyb_index_macd_cross_down,
                        kc_index_macd_cross_down=kc_index_macd_cross_down
                    )
                    if sell_signal_obj is not None:
                        break  # 找到一个卖出信号就停止

                if sell_signal_obj is not None:
                    signal_info = SignalInfo(
                        signal_date=current_cross_down_date.strftime('%Y-%m-%d'),
                        signal_type='sell',
                        signal_detail=f"[{sell_signal_obj.flag.name}] {sell_signal_obj.reason} ({trigger_name}触发)",
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
                    sh_index_daily_rsi=index_rsi_sh,
                    kc_index_daily_rsi=index_rsi_kc,
                    monthly_rsi=monthly_rsi_at_signal,
                    bj_index_daily_rsi=index_rsi_bj,
                    hsTech_index_daily_rsi=index_rsi_hsTech,
                    hsIndex_index_daily_rsi=index_rsi_hsIndex
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

        # ==================== MACD死叉卖出信号检测（sell_id=2,3,4,5,6）====================
        # 对于配置了sell_id=2,3,4,5,6的情况，触发条件都是个股MACD死叉
        # sell_id=2,3: 检查个股顶背离
        # sell_id=4,5,6: 检查指数顶背离（上证/创业板/科创板）
        macd_trigger_ids = [sid for sid in stock_config.judge_sell_ids if sid in [2, 3, 4, 5, 6]]
        if len(macd_trigger_ids) > 0 and len(macd_cross_down_signals) > 0:
            self.logger.info(f"  开始MACD死叉卖出检测（sell_ids={macd_trigger_ids})")

            # 遍历每个MACD死叉信号
            for macd_sig in macd_cross_down_signals:
                macd_cross_down_date = macd_sig.date

                # 只分析最近两个交易日的信号
                recent_dates = self.get_recent_trading_dates(end_date, 2)
                if macd_cross_down_date.strftime('%Y-%m-%d') not in recent_dates:
                    continue

                self.logger.info(f"  [MACD死叉检测] {macd_cross_down_date.strftime('%Y-%m-%d')}: "
                               f"DIF={macd_sig.dif:.4f}, DEA={macd_sig.dea:.4f}")

                # 获取当天的数据索引
                try:
                    current_idx = df.index.get_loc(macd_cross_down_date)
                except KeyError:
                    continue

                # 检查价格跌破SAR条件
                yesterday_sar = sar_series.iloc[current_idx - 1] if current_idx >= 1 else np.nan
                today_low = df['low'].iloc[current_idx]
                today_close = df['close'].iloc[current_idx]
                is_price_break_sar = today_low < yesterday_sar if not np.isnan(yesterday_sar) else False

                # 创建策略状态
                state = StrategyState()

                # 从上一次SAR买入信号（红转绿）之后累积rsi_flag
                cross_up_signals = [s for s in sar_signals if s.signal_type == '红转绿']
                last_cross_up_date = None
                if len(cross_up_signals) > 0:
                    for sig in reversed(cross_up_signals):
                        if sig.date < macd_cross_down_date:
                            last_cross_up_date = sig.date
                            break

                # 累积rsi_flag
                if last_cross_up_date is not None:
                    start_idx = df.index.get_loc(last_cross_up_date) + 1
                else:
                    start_idx = 0

                rsi_peak_value = 0.0
                rsi_peak_date = None
                for idx in range(start_idx, current_idx + 1):
                    weekly_rsi_val = weekly_rsi.iloc[idx] if idx < len(weekly_rsi) else np.nan
                    if not np.isnan(weekly_rsi_val):
                        strategy.update_rsi_flag(weekly_rsi_val, state, df.index[idx])
                        if weekly_rsi_val > rsi_peak_value:
                            rsi_peak_value = weekly_rsi_val
                            rsi_peak_date = df.index[idx]

                if rsi_peak_date is not None:
                    state.rsi_peak_date = rsi_peak_date
                    state.rsi_peak_value = rsi_peak_value

                # 检查背离状态（监控模式：背离形成后，待MACD死叉触发卖点）
                # 用户需求：顶背离形成时告警背离信息，MACD死叉时触发卖点
                # 【关键修改】背离触发卖点不需要SAR跌破条件，直接触发

                # 计算搜索起始日期（从上次SAR买入信号开始）
                # 【修复】使用买入信号日期本身，而不是+1天，这样同一天形成的背离也能触发
                if last_cross_up_date is not None:
                    search_start = last_cross_up_date
                else:
                    search_start = df.index[0]

                # 【关键】先检查是否有确认的背离，背离触发卖点不需要SAR跌破条件
                has_confirmed_divergence = False
                for div in daily_divergences_confirmed:
                    # 监控模式的背离已设置is_confirmed=True，不需要confirmation_date
                    # 只需要检查背离是否在MACD死叉之前形成
                    if div.is_confirmed and div.date <= macd_cross_down_date:
                        # 【修复】使用 >= 确保背离在买入信号当天或之后形成
                        if div.date >= search_start:
                            state.daily_divergence_flag = 1
                            state.daily_divergence_info = {
                                'date': div.date,
                                'prev_high': div.peak_a_price,
                                'curr_high': div.peak_b_price,
                                'prev_macd': div.peak_a_macd,
                                'curr_macd': div.peak_b_macd
                            }
                            has_confirmed_divergence = True
                            self.logger.info(f"    日线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}, "
                                           f"MACD死叉于{macd_cross_down_date.strftime('%Y-%m-%d')}")
                            break

                for div in weekly_divergences_confirmed:
                    # 监控模式：背离已形成，待MACD死叉触发卖点
                    if div.is_confirmed and div.date <= macd_cross_down_date:
                        if div.date >= search_start:
                            state.weekly_divergence_flag = 1
                            state.weekly_divergence_info = {
                                'date': div.date,
                                'prev_high': div.peak_a_price,
                                'curr_high': div.peak_b_price,
                                'prev_macd': div.peak_a_macd,
                                'curr_macd': div.peak_b_macd
                            }
                            has_confirmed_divergence = True
                            self.logger.info(f"    周线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}, "
                                           f"MACD死叉于{macd_cross_down_date.strftime('%Y-%m-%d')}")
                            break

                # 【关键】如果没有确认的背离，则需要SAR跌破条件才能触发卖点
                if not has_confirmed_divergence:
                    if not is_price_break_sar:
                        self.logger.info(f"    跌破SAR条件不满足（无背离）: 最低价={today_low:.3f} >= SAR={yesterday_sar:.3f}")
                        continue
                    self.logger.info(f"    跌破SAR条件满足（无背离）: 最低价={today_low:.3f} < SAR={yesterday_sar:.3f}")
                else:
                    self.logger.info(f"    有确认背离，无需SAR跌破条件: 日线={state.daily_divergence_flag}, 周线={state.weekly_divergence_flag}")

                # 【新增】设置指数背离状态（用于 sell_id=4,5,6）
                if any(sid in [4, 5, 6] for sid in macd_trigger_ids):
                    # 检查上证指数背离（用于 sell_id=4）
                    for div in sh_index_daily_divergences:
                        # 监控模式：背离已形成，待MACD死叉触发卖点
                        if div.is_confirmed and div.date <= macd_cross_down_date:
                            if div.date >= search_start:
                                sh_daily_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_sh_index_divergence(state, daily_info=sh_daily_div_info)
                                self.logger.info(f"    上证指数日线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break
                    for div in sh_index_weekly_divergences:
                        if div.is_confirmed and div.date <= macd_cross_down_date:
                            if div.date >= search_start:
                                sh_weekly_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_sh_index_divergence(state, weekly_info=sh_weekly_div_info)
                                self.logger.info(f"    上证指数周线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break

                    # 检查创业板指数背离（用于 sell_id=5）
                    for div in cyb_index_daily_divergences:
                        if div.is_confirmed and div.date <= macd_cross_down_date:
                            if div.date >= search_start:
                                cyb_daily_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_cyb_index_divergence(state, daily_info=cyb_daily_div_info)
                                self.logger.info(f"    创业板指数日线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break
                    for div in cyb_index_weekly_divergences:
                        if div.is_confirmed and div.date <= macd_cross_down_date:
                            if div.date >= search_start:
                                cyb_weekly_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_cyb_index_divergence(state, weekly_info=cyb_weekly_div_info)
                                self.logger.info(f"    创业板指数周线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break

                    # 检查科创板指数背离（用于 sell_id=6）
                    for div in kc_index_daily_divergences:
                        # 监控模式：背离已形成，待MACD死叉触发卖点
                        if div.is_confirmed and div.date <= macd_cross_down_date:
                            if div.date >= search_start:
                                kc_daily_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_kc_index_divergence(state, daily_info=kc_daily_div_info)
                                self.logger.info(f"    科创板指数日线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break
                    for div in kc_index_weekly_divergences:
                        if div.is_confirmed and div.date <= macd_cross_down_date:
                            if div.date >= search_start:
                                kc_weekly_div_info = {
                                    'date': div.date,
                                    'prev_high_date': div.peak_a_date,
                                    'prev_high': div.peak_a_price,
                                    'curr_high': div.peak_b_price,
                                    'prev_macd': div.peak_a_macd,
                                    'curr_macd': div.peak_b_macd
                                }
                                strategy.set_kc_index_divergence(state, weekly_info=kc_weekly_div_info)
                                self.logger.info(f"    科创板指数周线顶背离卖点触发: 背离形成于{div.date.strftime('%Y-%m-%d')}")
                                break

                self.logger.info(f"    RSI状态累积: rsi_flag={state.rsi_flag}, "
                               f"峰值={state.rsi_peak_value:.2f}, "
                               f"日线顶背离={state.daily_divergence_flag}, "
                               f"周线顶背离={state.weekly_divergence_flag}")

                # 检测卖出信号
                daily_rsi_at_signal = daily_rsi.iloc[current_idx] if current_idx < len(daily_rsi) else np.nan
                weekly_rsi_at_signal = weekly_rsi.iloc[current_idx] if current_idx < len(weekly_rsi) else np.nan

                # 获取当天的指数RSI
                index_rsi_cyb = self.get_index_rsi_at_date(INDEX_CYB, macd_cross_down_date.strftime('%Y-%m-%d'))
                index_rsi_sh = self.get_index_rsi_at_date(INDEX_SH, macd_cross_down_date.strftime('%Y-%m-%d'))

                sell_signal_obj = None
                for sell_id in macd_trigger_ids:
                    sell_signal_obj = strategy.check_sell_signal(
                        date=macd_cross_down_date,
                        daily_rsi=daily_rsi_at_signal,
                        weekly_rsi=weekly_rsi_at_signal,
                        sar_cross_down=False,
                        macd_cross_down=True,  # MACD死叉触发
                        state=state,
                        position=1.0,
                        sell_id=sell_id
                    )
                    if sell_signal_obj is not None:
                        break

                if sell_signal_obj is not None:
                    # 创建SAR死叉跌破信息（用于记录，虽然触发条件是MACD死叉）
                    cross_down_info = SARCrossDownInfo(
                        signal_date=macd_cross_down_date.strftime('%Y-%m-%d'),
                        sar_value=yesterday_sar,
                        close_price=today_close,
                        local_high_date=state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else None,
                        local_high_price=0.0,
                        local_high_weekly_rsi=state.rsi_peak_value,
                        has_daily_divergence=state.daily_divergence_flag == 1,
                        has_weekly_divergence=state.weekly_divergence_flag == 1,
                        daily_rsi_at_signal=daily_rsi_at_signal,
                        weekly_rsi_at_signal=weekly_rsi_at_signal
                    )

                    signal_info = SignalInfo(
                        signal_date=macd_cross_down_date.strftime('%Y-%m-%d'),
                        signal_type='sell',
                        signal_detail=f"[{sell_signal_obj.flag.name}] {sell_signal_obj.reason} (MACD日线跌破死叉触发)",
                        current_price=today_close,
                        daily_rsi=daily_rsi_at_signal,
                        weekly_rsi=weekly_rsi_at_signal,
                        index_daily_rsi_cyb=index_rsi_cyb,
                        index_daily_rsi_sh=index_rsi_sh,
                        sar_cross_down_info=cross_down_info
                    )
                    signals.append(signal_info)
                    self.logger.info(f"    >>> 卖险信号: {sell_signal_obj.reason}")

        # ==================== 独立卖出信号检测 ====================
        # 获取最新日期信息
        last_date = df.index[-1]
        last_date_str = last_date.strftime('%Y-%m-%d')

        # 【关键】初始化卖出状态变量（用于后续历史RSI高点告警判断）
        # 该变量会被SAR死叉检测逻辑更新
        has_actual_sell_after_buy = False
        last_actual_sell_date = None

        # 【关键】独立检测部分只对sell_id=1进行SAR死叉实时检测
        # 对于sell_id=2或3，触发条件是MACD日线跌破死叉，盘中无法准确判断MACD死叉
        # （MACD是基于收盘价计算的DIF和DEA关系）
        sar_trigger_ids = [sid for sid in stock_config.judge_sell_ids if sid == 1]
        do_sar_cross_down_check = len(sar_trigger_ids) > 0

        # 检测当天是否发生SAR死叉（实时盘中检测）
        # SAR死叉条件：当天最低价跌破SAR价位
        # 【关键】使用历史数据计算的SAR值作为判断基准（不包含今天的实时数据）
        if do_sar_cross_down_check and not np.isnan(yesterday_sar_baseline) and len(df) >= 1:
            today_low = df['low'].iloc[-1]
            today_close = df['close'].iloc[-1]
            yesterday_close = df_history['close'].iloc[-1] if len(df_history) > 0 else 0

            # 输出SAR值用于调试（显示正确的SAR基准值）
            self.logger.info(f"  [SAR值检查] {last_date_str}: 昨天SAR={yesterday_sar_baseline:.3f}（历史数据计算，不含今天）, "
                           f"今天最低价={today_low:.3f}, 昨天收盘价={yesterday_close:.3f}, "
                           f"最低价<SAR? {today_low < yesterday_sar_baseline}")

            # 判断是否发生SAR死叉（最低价跌破昨天的SAR值）
            is_sar_cross_down_today = today_low < yesterday_sar_baseline

            if is_sar_cross_down_today:
                self.logger.info(f"  [盘中SAR死叉检测] {last_date_str}: "
                               f"今天最低价={today_low:.3f} < 昨天SAR={yesterday_sar_baseline:.3f}，发生死叉")

                # 获取当天的RSI值
                today_daily_rsi = daily_rsi.iloc[-1] if len(daily_rsi) > 0 else np.nan
                today_weekly_rsi = weekly_rsi.iloc[-1] if len(weekly_rsi) > 0 else np.nan

                if not np.isnan(today_weekly_rsi):
                    # 创建策略实例和状态
                    strategy = TradingStrategy(
                        buy_ids=stock_config.judge_buy_ids,
                        t_ids=stock_config.judge_t_ids,
                        sell_ids=stock_config.judge_sell_ids
                    )

                    # 需要从上一次SAR买入信号（红转绿）之后累积rsi_flag
                    # 同时检查是否已经发生过卖出信号（如果卖出信号在买入信号之后，则不再触发）
                    state = StrategyState()

                    # 找到最近一次SAR买入信号（红转绿）日期
                    cross_up_signals = [s for s in sar_signals if s.signal_type == '红转绿']
                    last_cross_up_date = None
                    if len(cross_up_signals) > 0:
                        # 找到最后一个在昨天之前的买入信号
                        for sig in reversed(cross_up_signals):
                            if sig.date < last_date:
                                last_cross_up_date = sig.date
                                break

                    # 找到最近一次SAR卖出信号（绿转红）日期（在买入信号之后）
                    last_cross_down_date_before_today = None
                    if last_cross_up_date is not None:
                        for sig in reversed(cross_down_signals):
                            # 只找买入信号之后、今天之前的卖出信号
                            if sig.date > last_cross_up_date and sig.date < last_date:
                                last_cross_down_date_before_today = sig.date
                                break

                    # 【关键修改】检查买入后是否实际触发过卖出（不只是SAR卖出信号）
                    # 需要同时满足：1) 有SAR卖出信号，2) 有卖出触发条件（背离或RSI阈值）
                    # 【新增】卖出后有2个交易日的冷却期，超过冷却期后不再告警
                    has_actual_sell_after_buy = False
                    last_actual_sell_date = None  # 记录最后一次实际卖出日期
                    SELL_COOLDOWN_DAYS = 2  # 卖出后冷却期（交易日）

                    if last_cross_up_date is not None:
                        # 检查买入后所有卖出信号，看是否有任何一个触发过实际卖出
                        for sig in cross_down_signals:
                            # 只检查买入后、今天之前的卖出信号
                            if sig.date > last_cross_up_date and sig.date < last_date:
                                sar_sell_date = sig.date

                                # 检查这个卖出信号时是否有背离生效
                                for div in daily_divergences_confirmed:
                                    if div.confirmation_date is not None:
                                        if last_cross_up_date < div.confirmation_date <= sar_sell_date:
                                            has_actual_sell_after_buy = True
                                            last_actual_sell_date = sar_sell_date
                                            break
                                if has_actual_sell_after_buy:
                                    break

                                # 也检查周线背离
                                for div in weekly_divergences_confirmed:
                                    if div.confirmation_date is not None:
                                        if last_cross_up_date < div.confirmation_date <= sar_sell_date:
                                            has_actual_sell_after_buy = True
                                            last_actual_sell_date = sar_sell_date
                                            break
                                if has_actual_sell_after_buy:
                                    break

                                # 【新增】也检查RSI阈值触发：周线RSI峰值>=85
                                # 需要从买入后到卖出信号日期间累积rsi_flag和峰值
                                if not has_actual_sell_after_buy:
                                    # 计算当时的周线RSI峰值
                                    temp_peak_value = 0.0
                                    temp_peak_date = None
                                    temp_rsi_flag = 0
                                    buy_idx = df.index.get_loc(last_cross_up_date)
                                    sell_idx = df.index.get_loc(sar_sell_date)
                                    for idx in range(buy_idx + 1, sell_idx + 1):
                                        weekly_rsi_val = weekly_rsi.iloc[idx] if idx < len(weekly_rsi) else np.nan
                                        if not np.isnan(weekly_rsi_val):
                                            if weekly_rsi_val > temp_peak_value:
                                                temp_peak_value = weekly_rsi_val
                                                temp_peak_date = df.index[idx]
                                            # 更新rsi_flag
                                            if weekly_rsi_val >= RSI_THRESHOLDS['weekly_sell_level1']:
                                                temp_rsi_flag = 1
                                            if weekly_rsi_val >= RSI_THRESHOLDS['weekly_sell_level2']:
                                                temp_rsi_flag = 2
                                            if weekly_rsi_val >= RSI_THRESHOLDS['weekly_sell_level3']:
                                                temp_rsi_flag = 3

                                    # 检查是否触发卖出（周线RSI峰值>=85 + SAR死叉）
                                    # 使用配置的阈值而不是硬编码值
                                    threshold_85 = RSI_THRESHOLDS['weekly_sell_level2']
                                    if temp_peak_value >= threshold_85:
                                        has_actual_sell_after_buy = True
                                        last_actual_sell_date = sar_sell_date
                                        self.logger.info(f"    买入后RSI阈值触发卖出: SAR卖出={sar_sell_date.strftime('%Y-%m-%d')}, "
                                                       f"周线RSI峰值={temp_peak_value:.2f}")
                                        break

                    # 如果买入后已经实际卖出，检查是否在冷却期内
                    should_check_sell = True
                    if has_actual_sell_after_buy and last_actual_sell_date is not None:
                        # 计算卖出日期距今的交易日数
                        trading_days_since_sell = 0
                        for idx in range(len(df.index)):
                            if df.index[idx] > last_actual_sell_date and df.index[idx] <= last_date:
                                trading_days_since_sell += 1

                        if trading_days_since_sell > SELL_COOLDOWN_DAYS:
                            should_check_sell = False
                            self.logger.info(f"    卖出冷却期已过: 卖出日期={last_actual_sell_date.strftime('%Y-%m-%d')}, "
                                           f"距今{trading_days_since_sell}个交易日 > 冷却期{SELL_COOLDOWN_DAYS}天，跳过告警")
                        else:
                            self.logger.info(f"    卖出冷却期内: 卖出日期={last_actual_sell_date.strftime('%Y-%m-%d')}, "
                                           f"距今{trading_days_since_sell}个交易日 <= 冷却期{SELL_COOLDOWN_DAYS}天，继续告警")

                    if should_check_sell:
                        # 从最近一次买入信号后累积rsi_flag
                        if last_cross_up_date is not None:
                            start_idx = df.index.get_loc(last_cross_up_date) + 1
                        else:
                            start_idx = 0

                        # 遍历更新rsi_flag和背离状态
                        rsi_peak_value = 0.0
                        rsi_peak_date = None
                        for idx in range(start_idx, len(df)):
                            date = df.index[idx]
                            weekly_rsi_value = weekly_rsi.iloc[idx] if idx < len(weekly_rsi) else np.nan
                            if not np.isnan(weekly_rsi_value):
                                strategy.update_rsi_flag(weekly_rsi_value, state, date)
                                if weekly_rsi_value > rsi_peak_value:
                                    rsi_peak_value = weekly_rsi_value
                                    rsi_peak_date = date

                        if rsi_peak_date is not None and rsi_peak_value > 0:
                            state.rsi_peak_date = rsi_peak_date
                            state.rsi_peak_value = rsi_peak_value

                        # 检查背离状态
                        if last_cross_up_date is not None:
                            search_start = last_cross_up_date
                        else:
                            search_start = df.index[0]

                        for div in daily_divergences_confirmed:
                            if div.confirmation_date is not None:
                                if div.confirmation_date >= search_start and div.confirmation_date <= last_date:
                                    state.daily_divergence_flag = 1
                                    state.daily_divergence_info = {
                                        'date': div.date,
                                        'prev_high': div.peak_a_price,
                                        'curr_high': div.peak_b_price,
                                        'prev_macd': div.peak_a_macd,
                                        'curr_macd': div.peak_b_macd
                                    }
                                    break

                        for div in weekly_divergences_confirmed:
                            if div.confirmation_date is not None:
                                if div.confirmation_date >= search_start and div.confirmation_date <= last_date:
                                    state.weekly_divergence_flag = 1
                                    state.weekly_divergence_info = {
                                        'date': div.date,
                                        'prev_high': div.peak_a_price,
                                        'curr_high': div.peak_b_price,
                                        'prev_macd': div.peak_a_macd,
                                        'curr_macd': div.peak_b_macd
                                    }
                                    break

                        self.logger.info(f"    RSI状态累积: rsi_flag={state.rsi_flag}, "
                                       f"峰值={state.rsi_peak_value:.2f}, "
                                       f"日线顶背离={state.daily_divergence_flag}, "
                                       f"周线顶背离={state.weekly_divergence_flag}")

                        # 【关键】根据历史RSI峰值和背离状态判断卖出信号
                        # 不使用check_sell_signal（它检查当前RSI值），而是直接根据历史峰值判断
                        # 根据 judge_sell_ids 决定是否使用RSI阈值：
                        # - sell_id=1 或 2：使用RSI阈值判断
                        # - sell_id=3：仅背离判断，不使用RSI阈值
                        index_rsi_cyb = self.get_index_rsi_at_date(INDEX_CYB, last_date_str)
                        index_rsi_sh = self.get_index_rsi_at_date(INDEX_SH, last_date_str)

                        # 判断是否使用RSI阈值
                        use_rsi_threshold = any(sid in [1, 2] for sid in stock_config.judge_sell_ids)

                        # 获取阈值
                        threshold_90 = RSI_THRESHOLDS['weekly_sell_level3']  # 90
                        threshold_85 = RSI_THRESHOLDS['weekly_sell_level2']  # 85
                        threshold_80 = RSI_THRESHOLDS['weekly_sell_level1']  # 80

                        sell_flag = SellFlag.NO_SIGNAL
                        sell_reason = ""

                        # 判断卖出级别（按优先级）
                        # 优先级1：周线顶背离 + SAR死叉 → 清仓（所有sell_id都支持）
                        if state.weekly_divergence_flag == 1:
                            div_info = state.weekly_divergence_info or {}
                            div_date = div_info.get('date')
                            sell_flag = SellFlag.CLEAR_ALL
                            sell_reason = f"清仓信号: 周线顶背离生效 + SAR死叉触发, 建议清仓"
                            self.logger.info(f"    >>> 周线顶背离+SAR死叉触发清仓")

                        # 优先级2：历史RSI峰值>=90 + SAR死叉 → 清仓（仅sell_id=1或2）
                        elif use_rsi_threshold and state.rsi_peak_value >= threshold_90:
                            sell_flag = SellFlag.CLEAR_ALL
                            peak_date_str = state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else 'N/A'
                            sell_reason = f"清仓信号: 周线RSI峰值{peak_date_str}={state.rsi_peak_value:.2f}>90 + SAR死叉触发, 建议清仓"
                            self.logger.info(f"    >>> RSI峰值{state.rsi_peak_value:.2f}>90+SAR死叉触发清仓")

                        # 优先级3：历史RSI峰值>=85 + SAR死叉 → 卖出1/2（仅sell_id=1或2）
                        elif use_rsi_threshold and state.rsi_peak_value >= threshold_85:
                            sell_flag = SellFlag.SELL_HALF
                            peak_date_str = state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else 'N/A'
                            sell_reason = f"卖出信号: 周线RSI峰值{peak_date_str}={state.rsi_peak_value:.2f}>85 + SAR死叉触发, 建议卖出1/2"
                            self.logger.info(f"    >>> RSI峰值{state.rsi_peak_value:.2f}>85+SAR死叉触发卖出1/2")

                        # 优先级4：日线顶背离 + SAR死叉 → 卖出1/3（所有sell_id都支持）
                        elif state.daily_divergence_flag == 1:
                            sell_flag = SellFlag.SELL_ONE_THIRD
                            sell_reason = f"卖出信号: 日线顶背离生效 + SAR死叉触发, 建议卖出1/3"
                            self.logger.info(f"    >>> 日线顶背离+SAR死叉触发卖出1/3")

                        # 优先级5：历史RSI峰值>=80 + SAR死叉 → 卖出1/3（仅sell_id=1或2）
                        elif use_rsi_threshold and state.rsi_peak_value >= threshold_80:
                            sell_flag = SellFlag.SELL_ONE_THIRD
                            peak_date_str = state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else 'N/A'
                            sell_reason = f"卖出信号: 周线RSI峰值{peak_date_str}={state.rsi_peak_value:.2f}>80 + SAR死叉触发, 建议卖出1/3"
                            self.logger.info(f"    >>> RSI峰值{state.rsi_peak_value:.2f}>80+SAR死叉触发卖出1/3")

                        # 如果有卖出信号，添加到signals列表
                        if sell_flag != SellFlag.NO_SIGNAL:
                            cross_down_info = SARCrossDownInfo(
                                signal_date=last_date_str,
                                sar_value=yesterday_sar_baseline,
                                close_price=today_close,
                                local_high_date=state.rsi_peak_date.strftime('%Y-%m-%d') if state.rsi_peak_date else None,
                                local_high_price=0.0,
                                local_high_weekly_rsi=state.rsi_peak_value,
                                has_daily_divergence=state.daily_divergence_flag == 1,
                                has_weekly_divergence=state.weekly_divergence_flag == 1,
                                daily_rsi_at_signal=today_daily_rsi,
                                weekly_rsi_at_signal=today_weekly_rsi
                            )

                            signal_info = SignalInfo(
                                signal_date=last_date_str,
                                signal_type='sell',
                                signal_detail=f"[{sell_flag.name}] {sell_reason} (盘中SAR死叉触发)",
                                current_price=today_close,
                                daily_rsi=today_daily_rsi,
                                weekly_rsi=today_weekly_rsi,
                                index_daily_rsi_cyb=index_rsi_cyb,
                                index_daily_rsi_sh=index_rsi_sh,
                                sar_cross_down_info=cross_down_info
                            )
                            signals.append(signal_info)
                            self.logger.info(f"    >>> 卖出信号: {sell_reason}")

        # ==================== 独立买入信号检测 ====================
        # 买入条件只依赖RSI，不应该依赖SAR死叉事件
        # 如果最近交易日没有通过SAR循环触发买入信号，则独立检测买入条件
        recent_dates = self.get_recent_trading_dates(end_date, RECENT_TRADING_DAYS)

        # 检查最新日期是否在最近交易日范围内，且当天没有买入信号
        if last_date_str in recent_dates:
            # 检查当天是否已经有买入信号（通过SAR循环触发）
            has_buy_signal_today = any(
                sig.signal_type == 'buy' and sig.signal_date == last_date_str
                for sig in signals
            )

            if not has_buy_signal_today:
                latest_daily_rsi = daily_rsi.iloc[-1] if len(daily_rsi) > 0 else np.nan
                latest_weekly_rsi = weekly_rsi.iloc[-1] if len(weekly_rsi) > 0 else np.nan
                latest_monthly_rsi = monthly_rsi.iloc[-1] if len(monthly_rsi) > 0 else np.nan
                latest_index_rsi_cyb = self.get_index_rsi_at_date(INDEX_CYB, last_date_str)
                latest_index_rsi_sh = self.get_index_rsi_at_date(INDEX_SH, last_date_str)
                latest_index_rsi_kc = self.get_index_rsi_at_date(INDEX_KC, last_date_str)
                latest_index_rsi_bj = self.get_index_rsi_at_date(INDEX_BJ, last_date_str)
                latest_index_rsi_hsTech = self.get_index_rsi_at_date(INDEX_HS_TECH, last_date_str)
                latest_index_rsi_hsIndex = self.get_index_rsi_at_date(INDEX_HS_INDEX, last_date_str)

                if not np.isnan(latest_daily_rsi) and not np.isnan(latest_weekly_rsi):
                    # 创建策略实例和状态（用于独立买入检测）
                    strategy = TradingStrategy(
                        buy_ids=stock_config.judge_buy_ids,
                        t_ids=stock_config.judge_t_ids,
                        sell_ids=stock_config.judge_sell_ids
                    )
                    state = StrategyState()

                    # 检测买入信号（独立于SAR事件）
                    buy_signal_obj = strategy.check_buy_signal(
                        date=last_date,
                        daily_rsi=latest_daily_rsi,
                        weekly_rsi=latest_weekly_rsi,
                        state=state,
                        has_new_cash=True,
                        has_sold_cash=True,
                        index_daily_rsi=latest_index_rsi_cyb,
                        sh_index_daily_rsi=latest_index_rsi_sh,
                        kc_index_daily_rsi=latest_index_rsi_kc,
                        monthly_rsi=latest_monthly_rsi,
                        bj_index_daily_rsi=latest_index_rsi_bj,
                        hsTech_index_daily_rsi=latest_index_rsi_hsTech,
                        hsIndex_index_daily_rsi=latest_index_rsi_hsIndex
                    )

                    if buy_signal_obj is not None and buy_signal_obj.triggered:
                        signal_info = SignalInfo(
                            signal_date=last_date_str,
                            signal_type='buy',
                            signal_detail=f"[{'新资金买入' if buy_signal_obj.is_new_cash else '做T买回'}] {buy_signal_obj.reason} (独立RSI触发)",
                            current_price=df['close'].iloc[-1],
                            daily_rsi=latest_daily_rsi,
                            weekly_rsi=latest_weekly_rsi,
                            index_daily_rsi_cyb=latest_index_rsi_cyb,
                            index_daily_rsi_sh=latest_index_rsi_sh,
                            sar_cross_down_info=None  # 无SAR事件
                        )
                        signals.append(signal_info)
                        self.logger.info(f"  [独立买入检测] {last_date_str}: "
                                       f"日RSI={latest_daily_rsi:.2f}, 周RSI={latest_weekly_rsi:.2f}")
                        self.logger.info(f"    >>> 买入信号: {buy_signal_obj.reason}")

        # 获取最新数据用于监控结果展示

        latest_daily_rsi = daily_rsi.iloc[-1] if len(daily_rsi) > 0 else np.nan
        latest_weekly_rsi = weekly_rsi.iloc[-1] if len(weekly_rsi) > 0 else np.nan
        latest_sar = sar_series.iloc[-1] if len(sar_series) > 0 else np.nan
        latest_price = df['close'].iloc[-1]
        # 【关键修改】使用今天的日期获取最新RSI，而不是配置的end_date
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        latest_index_rsi_cyb = self.get_index_rsi(INDEX_CYB, today_str)
        latest_index_rsi_sh = self.get_index_rsi(INDEX_SH, today_str)
        latest_index_rsi_kc = self.get_index_rsi(INDEX_KC, today_str)
        latest_index_rsi_bj = self.get_index_rsi(INDEX_BJ, today_str)
        latest_index_rsi_hsTech = self.get_index_rsi(INDEX_HS_TECH, today_str)
        latest_index_rsi_hsIndex = self.get_index_rsi(INDEX_HS_INDEX, today_str)

        # 判断当前SAR趋势
        latest_sar_trend = '多头' if latest_sar < latest_price else '空头'

        # ==================== 待触发卖出信号告警 ====================
        # 只显示最近一次买入机会（红转绿）之后的告警
        # 如果已经发生卖出（绿转红），之前的告警就消失
        pending_warnings = []

        # 获取最近一次SAR买入信号日期（红转绿）和卖出信号日期（绿转红）
        # SAR买入（红转绿）= SAR从价格上方转到下方：SAR(t) < Close(t) 且 SAR(t-1) >= Close(t-1)
        # SAR卖出（绿转红）= SAR从价格下方转到上方：SAR(t) > Close(t) 且 SAR(t-1) <= Close(t-1)
        last_sar_cross_up_date = None   # 最近一次买入信号（红转绿）
        last_sar_cross_down_date = None  # 最近一次卖出信号（绿转红）

        if sar_series is not None and len(sar_series) > 1 and df is not None:
            # 从最新日期往前查找最近一次买入和卖出信号
            for i in range(len(sar_series) - 1, 0, -1):
                curr_sar = sar_series.iloc[i]
                prev_sar = sar_series.iloc[i - 1]
                curr_close = df['close'].iloc[i]
                prev_close = df['close'].iloc[i - 1]
                curr_date = sar_series.index[i]

                # 判断是否发生SAR买入信号（红转绿）
                if curr_sar < curr_close and prev_sar >= prev_close:
                    if last_sar_cross_up_date is None:
                        last_sar_cross_up_date = pd.Timestamp(curr_date.strftime('%Y-%m-%d'))

                # 判断是否发生SAR卖出信号（绿转红）
                if curr_sar > curr_close and prev_sar <= prev_close:
                    if last_sar_cross_down_date is None:
                        last_sar_cross_down_date = pd.Timestamp(curr_date.strftime('%Y-%m-%d'))

                # 如果两个都找到了，停止查找
                if last_sar_cross_up_date is not None and last_sar_cross_down_date is not None:
                    break

        # ==================== 独立计算历史卖出状态 ====================
        # 用于历史RSI高点告警判断：检查最近一次买入后是否已经发生过卖出
        # 【关键修复】根据 sell_ids 配置决定检查哪种卖出信号：
        # - sell_id=1: SAR 死叉卖出（检查 cross_down_signals）
        # - sell_id=2 或 3: MACD 死叉卖出（检查 macd_cross_down_signals）
        self.logger.info(f"  [历史卖出检测] 开始检测: sell_ids={stock_config.judge_sell_ids}")

        # 根据 sell_ids 决定检查哪种卖出信号
        use_sar_sell = any(sid in [1] for sid in stock_config.judge_sell_ids)
        use_macd_sell = any(sid in [2, 3] for sid in stock_config.judge_sell_ids)

        # 获取最近一次买入信号日期（根据策略类型）
        # 对于 MACD 死叉策略，买入信号也是 MACD 金叉
        last_buy_date = None
        if use_macd_sell and len(macd_cross_signals) > 0:
            # MACD 金叉信号
            macd_cross_up_signals = [s for s in macd_cross_signals if s.signal_type == '金叉']
            if len(macd_cross_up_signals) > 0:
                for sig in reversed(macd_cross_up_signals):
                    if sig.date < last_date:
                        last_buy_date = sig.date
                        self.logger.info(f"  [历史卖出检测] MACD金叉买入信号: {last_buy_date.strftime('%Y-%m-%d')}")
                        break
        elif use_sar_sell and len(cross_up_signals) > 0:
            # SAR 红转绿信号（对于 sell_id=1，这可能是买入或卖出，需要确认）
            # 这里使用 last_sar_cross_up_date（之前已经计算好的）
            last_buy_date = last_sar_cross_up_date
            if last_buy_date is not None:
                self.logger.info(f"  [历史卖出检测] SAR买入信号日期: {last_buy_date.strftime('%Y-%m-%d')}")

        if last_buy_date is not None:
            # 统一时区处理
            buy_date_naive = last_buy_date.tz_localize(None) if last_buy_date.tz is not None else last_buy_date
            last_date_naive = last_date.tz_localize(None) if last_date.tz is not None else last_date

            # 检查买入后的卖出信号
            sell_signals_to_check = []
            if use_macd_sell:
                sell_signals_to_check = macd_cross_down_signals
                self.logger.info(f"  [历史卖出检测] 检查MACD死叉信号: {len(sell_signals_to_check)} 个")
            if use_sar_sell:
                sell_signals_to_check = cross_down_signals
                self.logger.info(f"  [历史卖出检测] 检查SAR死叉信号: {len(sell_signals_to_check)} 个")

            for sig in sell_signals_to_check:
                # 统一时区处理
                sig_date_naive = sig.date.tz_localize(None) if sig.date.tz is not None else sig.date
                self.logger.info(f"  [历史卖出检测] 检查卖出信号: {sig_date_naive.strftime('%Y-%m-%d')}")

                # 只检查买入后、今天之前的卖出信号
                if sig_date_naive > buy_date_naive and sig_date_naive < last_date_naive:
                    sell_date = sig.date
                    sell_date_naive = sig_date_naive

                    # 检查这个卖出信号时是否有背离生效
                    for div in daily_divergences_confirmed:
                        if div.confirmation_date is not None:
                            div_date = div.confirmation_date.tz_localize(None) if div.confirmation_date.tz is not None else div.confirmation_date
                            if buy_date_naive < div_date <= sell_date_naive:
                                has_actual_sell_after_buy = True
                                last_actual_sell_date = sell_date
                                self.logger.info(f"    [历史卖出检测] 日线背离触发卖出: {sell_date.strftime('%Y-%m-%d')}")
                                break
                    if has_actual_sell_after_buy:
                        break

                    # 也检查周线背离
                    for div in weekly_divergences_confirmed:
                        if div.confirmation_date is not None:
                            div_date = div.confirmation_date.tz_localize(None) if div.confirmation_date.tz is not None else div.confirmation_date
                            if buy_date_naive < div_date <= sell_date_naive:
                                has_actual_sell_after_buy = True
                                last_actual_sell_date = sell_date
                                self.logger.info(f"    [历史卖出检测] 周线背离触发卖出: {sell_date.strftime('%Y-%m-%d')}")
                                break
                    if has_actual_sell_after_buy:
                        break

                    # 也检查RSI阈值触发：周线RSI峰值>=85
                    if not has_actual_sell_after_buy:
                        # 计算买入到卖出之间的周线RSI峰值
                        temp_peak_value = 0.0
                        try:
                            # 【修复】统一时区处理：将 df 索引转换为 tz-naive
                            df_index_naive = df.index
                            if df_index_naive.tz is not None:
                                df_index_naive = df_index_naive.tz_localize(None)

                            # 尝试在 df 索引中找到日期
                            try:
                                buy_idx = df_index_naive.get_loc(buy_date_naive)
                            except KeyError:
                                # 如果找不到精确匹配，尝试找最近的日期
                                buy_idx = df_index_naive.get_indexer([buy_date_naive], method='nearest')[0]
                                self.logger.info(f"    [历史卖出检测] 买入日期 {buy_date_naive.strftime('%Y-%m-%d')} 在df中未找到，使用最近索引: {buy_idx}")

                            try:
                                sell_idx = df_index_naive.get_loc(sell_date_naive)
                            except KeyError:
                                sell_idx = df_index_naive.get_indexer([sell_date_naive], method='nearest')[0]
                                self.logger.info(f"    [历史卖出检测] 卖出日期 {sell_date_naive.strftime('%Y-%m-%d')} 在df中未找到，使用最近索引: {sell_idx}")

                            # 确保索引有效
                            if buy_idx >= 0 and sell_idx >= 0 and sell_idx > buy_idx:
                                for idx in range(buy_idx + 1, sell_idx + 1):
                                    if idx < len(weekly_rsi):
                                        weekly_rsi_val = weekly_rsi.iloc[idx]
                                        if not np.isnan(weekly_rsi_val):
                                            if weekly_rsi_val > temp_peak_value:
                                                temp_peak_value = weekly_rsi_val
                                self.logger.info(f"    [历史卖出检测] 买入到卖出期间周线RSI峰值: {temp_peak_value:.2f}")
                        except Exception as e:
                            self.logger.warning(f"    [历史卖出检测] RSI峰值计算异常: {e}")
                            # 即使计算失败，也尝试直接检查当前周线RSI是否>=85
                            # 如果周线RSI在买入时或卖出时已经>=85，也算触发
                            if len(weekly_rsi) > 0:
                                # 检查最近几天的周线RSI
                                for idx in range(max(0, len(weekly_rsi) - 5), len(weekly_rsi)):
                                    weekly_rsi_val = weekly_rsi.iloc[idx]
                                    if not np.isnan(weekly_rsi_val) and weekly_rsi_val >= 85:
                                        temp_peak_value = weekly_rsi_val
                                        self.logger.info(f"    [历史卖出检测] 找到周线RSI>=85: {temp_peak_value:.2f}")
                                        break

                        # 检查是否触发卖出（周线RSI峰值>=85）
                        threshold_85 = RSI_THRESHOLDS['weekly_sell_level2']
                        if temp_peak_value >= threshold_85:
                            has_actual_sell_after_buy = True
                            last_actual_sell_date = sell_date
                            self.logger.info(f"    [历史卖出检测] RSI阈值触发卖出: 卖出={sell_date.strftime('%Y-%m-%d')}, 周线RSI峰值={temp_peak_value:.2f}")
                            break

        if has_actual_sell_after_buy:
            self.logger.info(f"    [历史卖出状态] 买入后已触发卖出，不再发出历史RSI高点告警")

        # 检查周线顶背离已形成但还没有SAR死叉
        # 监控模式：背离形成即告警，待MACD死叉触发卖点
        # 只显示在最近一次买入信号（红转绿）之后形成的背离，且当前还没卖出（绿转红）
        if weekly_divergences_confirmed and len(weekly_divergences_confirmed) > 0:
            for div in weekly_divergences_confirmed:
                # 监控模式：检查背离是否已形成（is_confirmed=True）
                if div.is_confirmed:
                    # 过滤条件：
                    # 1. 背离形成日期 > 最近一次买入信号日期（在买入后形成）
                    # 2. 当前SAR趋势是多头（还没收到卖出信号）
                    # 统一时区处理：去掉时区信息进行比较
                    div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                    # 【修复】使用 >= 确保同一天形成的背离也能显示待触发告警
                    if (last_sar_cross_up_date is None or div_date >= last_sar_cross_up_date) and latest_sar_trend == '多头':
                        peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div, 'peak_a_date') and div.peak_a_date else 'N/A'
                        peak_a_price_str = f"{div.peak_a_price:.3f}" if hasattr(div, 'peak_a_price') and div.peak_a_price else 'N/A'
                        pending_warnings.append({
                            'type': '周线顶背离',
                            'detail': f"周线顶背离形成于{div.date.strftime('%Y-%m-%d')}, 前高点{peak_a_date_str}({peak_a_price_str}), 等待SAR死叉触发清仓",
                            'severity': 'HIGH',
                            'divergence_date': div.date
                        })
                        self.logger.warning(f"  [待触发告警] 周线顶背离已形成({div.date.strftime('%Y-%m-%d')})，前高点{peak_a_date_str}({peak_a_price_str})，等待SAR死叉触发清仓")

        # 检查周线RSI已达警戒级别但还没有SAR死叉
        # 只有 sell_id=1 或 2 才检查RSI阈值（sell_id=3不含RSI阶梯）
        check_weekly_rsi_threshold = any(sid in [1, 2] for sid in stock_config.judge_sell_ids)
        if check_weekly_rsi_threshold and not np.isnan(latest_weekly_rsi) and latest_sar_trend == '多头':
            threshold_90 = RSI_THRESHOLDS['weekly_sell_level3']  # 90
            threshold_85 = RSI_THRESHOLDS['weekly_sell_level2']  # 85
            threshold_80 = RSI_THRESHOLDS['weekly_sell_level1']  # 80

            if latest_weekly_rsi >= threshold_90:
                pending_warnings.append({
                    'type': 'RSI清仓警戒',
                    'detail': f"周线RSI={latest_weekly_rsi:.2f}>90，等待SAR死叉触发清仓",
                    'severity': 'HIGH',
                    'confirmation_date': pd.Timestamp(end_date)
                })
                self.logger.warning(f"  [待触发告警] 周线RSI>90，等待SAR死叉触发清仓")
            elif latest_weekly_rsi >= threshold_85:
                pending_warnings.append({
                    'type': 'RSI减仓警戒',
                    'detail': f"周线RSI={latest_weekly_rsi:.2f}>85，等待SAR死叉触发卖出1/2",
                    'severity': 'MEDIUM',
                    'confirmation_date': pd.Timestamp(end_date)
                })
                self.logger.warning(f"  [待触发告警] 周线RSI>85，等待SAR死叉触发卖出1/2")
            elif latest_weekly_rsi >= threshold_80:
                pending_warnings.append({
                    'type': 'RSI减仓警戒',
                    'detail': f"周线RSI={latest_weekly_rsi:.2f}>80，等待SAR死叉触发卖出1/3",
                    'severity': 'LOW',
                    'confirmation_date': pd.Timestamp(end_date)
                })
                self.logger.info(f"  [待触发提示] 周线RSI>80，等待SAR死叉触发卖出1/3")

        # 检查日线顶背离已形成但还没有SAR死叉
        # 监控模式：背离形成即告警，待MACD死叉触发卖点
        # 只显示在最近一次买入信号（红转绿）之后形成的背离，且当前还没卖出（绿转红）
        if daily_divergences_confirmed and len(daily_divergences_confirmed) > 0:
            for div in daily_divergences_confirmed:
                # 监控模式：检查背离是否已形成（is_confirmed=True）
                if div.is_confirmed:
                    # 过滤条件：
                    # 1. 背离形成日期 > 最近一次买入信号日期（在买入后形成）
                    # 2. 当前SAR趋势是多头（还没收到卖出信号）
                    # 统一时区处理：去掉时区信息进行比较
                    div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                    # 【修复】使用 >= 确保同一天形成的背离也能显示待触发告警
                    if (last_sar_cross_up_date is None or div_date >= last_sar_cross_up_date) and latest_sar_trend == '多头':
                        peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div, 'peak_a_date') and div.peak_a_date else 'N/A'
                        peak_a_price_str = f"{div.peak_a_price:.3f}" if hasattr(div, 'peak_a_price') and div.peak_a_price else 'N/A'
                        pending_warnings.append({
                            'type': '日线顶背离',
                            'detail': f"日线顶背离形成于{div.date.strftime('%Y-%m-%d')}, 前高点{peak_a_date_str}({peak_a_price_str}), 等待SAR死叉触发卖出1/3",
                            'severity': 'LOW',
                            'divergence_date': div.date
                        })
                        self.logger.info(f"  [待触发提示] 日线顶背离已形成({div.date.strftime('%Y-%m-%d')})，前高点{peak_a_date_str}({peak_a_price_str})，等待SAR死叉触发卖出1/3")

        # ==================== 指数顶背离告警 ====================
        # 显示最近一个月（30天）内形成的顶背离，日线和周线合并显示
        index_divergence_warnings = []
        recent_days = 30  # 最近30天（一个月）
        end_date_ts = pd.Timestamp(end_date)

        # 收集最近一个月内形成的指数顶背离
        for div in sh_index_daily_divergences + sh_index_weekly_divergences:
            if div.is_confirmed:
                div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                days_diff = (end_date_ts - div_date).days
                if days_diff <= recent_days:
                    timeframe = '日线' if div in sh_index_daily_divergences else '周线'
                    peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div.peak_a_date, 'strftime') else str(div.peak_a_date)[:10]
                    peak_b_date_str = div.date.strftime('%Y-%m-%d')
                    index_divergence_warnings.append({
                        'index_name': '上证指数',
                        'index_code': '000001',
                        'type': f'{timeframe}顶背离',
                        'detail': f"上证指数{timeframe}顶背离形成于{peak_b_date_str}",
                        'severity': 'HIGH',
                        'divergence_date': div.date,
                        'days_elapsed': days_diff,
                        'peak_a_date': peak_a_date_str,
                        'peak_b_date': peak_b_date_str,
                        'peak_a_price': div.peak_a_price,
                        'peak_b_price': div.peak_b_price,
                        'peak_a_macd': div.peak_a_macd,
                        'peak_b_macd': div.peak_b_macd
                    })
                    self.logger.warning(f"  [指数背离告警] 上证指数{timeframe}顶背离: 前高点{peak_a_date_str}={div.peak_a_price:.2f}MACD={div.peak_a_macd:.4f}, 当前高点{peak_b_date_str}={div.peak_b_price:.2f}MACD={div.peak_b_macd:.4f}")

        for div in cyb_index_daily_divergences + cyb_index_weekly_divergences:
            if div.is_confirmed:
                div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                days_diff = (end_date_ts - div_date).days
                if days_diff <= recent_days:
                    timeframe = '日线' if div in cyb_index_daily_divergences else '周线'
                    peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div.peak_a_date, 'strftime') else str(div.peak_a_date)[:10]
                    peak_b_date_str = div.date.strftime('%Y-%m-%d')
                    index_divergence_warnings.append({
                        'index_name': '创业板指数',
                        'index_code': '399006',
                        'type': f'{timeframe}顶背离',
                        'detail': f"创业板指数{timeframe}顶背离形成于{peak_b_date_str}",
                        'severity': 'HIGH',
                        'divergence_date': div.date,
                        'days_elapsed': days_diff,
                        'peak_a_date': peak_a_date_str,
                        'peak_b_date': peak_b_date_str,
                        'peak_a_price': div.peak_a_price,
                        'peak_b_price': div.peak_b_price,
                        'peak_a_macd': div.peak_a_macd,
                        'peak_b_macd': div.peak_b_macd
                    })
                    self.logger.warning(f"  [指数背离告警] 创业板指数{timeframe}顶背离: 前高点{peak_a_date_str}={div.peak_a_price:.2f}MACD={div.peak_a_macd:.4f}, 当前高点{peak_b_date_str}={div.peak_b_price:.2f}MACD={div.peak_b_macd:.4f}")

        for div in kc_index_daily_divergences + kc_index_weekly_divergences:
            if div.is_confirmed:
                div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                days_diff = (end_date_ts - div_date).days
                if days_diff <= recent_days:
                    timeframe = '日线' if div in kc_index_daily_divergences else '周线'
                    peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div.peak_a_date, 'strftime') else str(div.peak_a_date)[:10]
                    peak_b_date_str = div.date.strftime('%Y-%m-%d')
                    index_divergence_warnings.append({
                        'index_name': '科创综指',
                        'index_code': '000680',
                        'type': f'{timeframe}顶背离',
                        'detail': f"科创综指{timeframe}顶背离形成于{peak_b_date_str}",
                        'severity': 'HIGH',
                        'divergence_date': div.date,
                        'days_elapsed': days_diff,
                        'peak_a_date': peak_a_date_str,
                        'peak_b_date': peak_b_date_str,
                        'peak_a_price': div.peak_a_price,
                        'peak_b_price': div.peak_b_price,
                        'peak_a_macd': div.peak_a_macd,
                        'peak_b_macd': div.peak_b_macd
                    })
                    self.logger.warning(f"  [指数背离告警] 科创综指{timeframe}顶背离: 前高点{peak_a_date_str}={div.peak_a_price:.2f}MACD={div.peak_a_macd:.4f}, 当前高点{peak_b_date_str}={div.peak_b_price:.2f}MACD={div.peak_b_macd:.4f}")

        # 恒生科技ETF顶背离告警
        for div in hs_tech_daily_divergences + hs_tech_weekly_divergences:
            if div.is_confirmed:
                div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                days_diff = (end_date_ts - div_date).days
                if days_diff <= recent_days:
                    timeframe = '日线' if div in hs_tech_daily_divergences else '周线'
                    peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div.peak_a_date, 'strftime') else str(div.peak_a_date)[:10]
                    peak_b_date_str = div.date.strftime('%Y-%m-%d')
                    index_divergence_warnings.append({
                        'index_name': '恒生科技ETF',
                        'index_code': '513180',
                        'type': f'{timeframe}顶背离',
                        'detail': f"恒生科技ETF{timeframe}顶背离形成于{peak_b_date_str}",
                        'severity': 'HIGH',
                        'divergence_date': div.date,
                        'days_elapsed': days_diff,
                        'peak_a_date': peak_a_date_str,
                        'peak_b_date': peak_b_date_str,
                        'peak_a_price': div.peak_a_price,
                        'peak_b_price': div.peak_b_price,
                        'peak_a_macd': div.peak_a_macd,
                        'peak_b_macd': div.peak_b_macd
                    })
                    self.logger.warning(f"  [指数背离告警] 恒生科技ETF{timeframe}顶背离: 前高点{peak_a_date_str}={div.peak_a_price:.2f}MACD={div.peak_a_macd:.4f}, 当前高点{peak_b_date_str}={div.peak_b_price:.2f}MACD={div.peak_b_macd:.4f}")

        # 恒生指数ETF顶背离告警
        for div in hs_index_daily_divergences + hs_index_weekly_divergences:
            if div.is_confirmed:
                div_date = div.date.tz_localize(None) if div.date.tz is not None else div.date
                days_diff = (end_date_ts - div_date).days
                if days_diff <= recent_days:
                    timeframe = '日线' if div in hs_index_daily_divergences else '周线'
                    peak_a_date_str = div.peak_a_date.strftime('%Y-%m-%d') if hasattr(div.peak_a_date, 'strftime') else str(div.peak_a_date)[:10]
                    peak_b_date_str = div.date.strftime('%Y-%m-%d')
                    index_divergence_warnings.append({
                        'index_name': '恒生指数ETF',
                        'index_code': '159920',
                        'type': f'{timeframe}顶背离',
                        'detail': f"恒生指数ETF{timeframe}顶背离形成于{peak_b_date_str}",
                        'severity': 'HIGH',
                        'divergence_date': div.date,
                        'days_elapsed': days_diff,
                        'peak_a_date': peak_a_date_str,
                        'peak_b_date': peak_b_date_str,
                        'peak_a_price': div.peak_a_price,
                        'peak_b_price': div.peak_b_price,
                        'peak_a_macd': div.peak_a_macd,
                        'peak_b_macd': div.peak_b_macd
                    })
                    self.logger.warning(f"  [指数背离告警] 恒生指数ETF{timeframe}顶背离: 前高点{peak_a_date_str}={div.peak_a_price:.2f}MACD={div.peak_a_macd:.4f}, 当前高点{peak_b_date_str}={div.peak_b_price:.2f}MACD={div.peak_b_macd:.4f}")

        # ==================== 历史RSI高点告警 ====================
        # 检查最近一次买入信号（红转绿）之后的历史RSI高点，即使当前RSI已降低也要告警
        # 如果已经卖出（绿转红），告警就消失
        # 告警条件根据 judge_sell_ids 配置确定：
        # - sell_id=1 或 2：检查周线RSI阈值（80/85/90）
        # - sell_id=3：不检查RSI阈值（仅背离判断）
        rsi_high_warnings = []

        # 判断是否需要检查周线RSI阈值
        check_weekly_rsi_threshold = any(sid in [1, 2] for sid in stock_config.judge_sell_ids)

        # 【修复】使用 has_actual_sell_after_buy 判断是否已经卖出（而不是只看最近2天的signals列表）
        # 这样可以正确识别历史卖出，避免卖出后仍发出历史RSI高点告警
        # 注意：has_actual_sell_after_buy 是在卖出信号检测逻辑中已经计算好的

        if latest_sar_trend == '多头' and check_weekly_rsi_threshold and not has_actual_sell_after_buy:
            # 计算回溯起始日期（从最近一次买入信号日期开始，或从配置的回溯天数开始）
            if last_sar_cross_up_date is not None:
                # 从最近一次买入信号（红转绿）日期开始查找
                lookback_start_date = last_sar_cross_up_date
            else:
                # 从未发生过买入信号，使用配置的回溯天数
                lookback_start_date = pd.Timestamp(end_date) - pd.Timedelta(days=RSI_HIGH_LOOKBACK_DAYS)

            # 检查周线RSI历史高点
            if weekly_rsi is not None and len(weekly_rsi) > 0:
                # 处理时区问题：确保比较时时区一致
                rsi_index = weekly_rsi.index
                if rsi_index.tz is not None:
                    # 如果index是tz-aware，将lookback_start_date转为相同时区
                    lookback_start_date_tz = lookback_start_date.tz_localize(rsi_index.tz)
                else:
                    # 如果index是tz-naive，保持lookback_start_date也是tz-naive
                    lookback_start_date_tz = lookback_start_date

                # 筛选最近一次买入信号之后的数据
                recent_weekly_rsi = weekly_rsi[rsi_index >= lookback_start_date_tz]
                if len(recent_weekly_rsi) > 0:
                    # 找到最高点
                    max_weekly_rsi = recent_weekly_rsi.max()
                    max_weekly_rsi_idx = recent_weekly_rsi.idxmax()

                    # 使用策略类定义的阈值
                    threshold_90 = RSI_THRESHOLDS['weekly_sell_level3']  # 90
                    threshold_85 = RSI_THRESHOLDS['weekly_sell_level2']  # 85
                    threshold_80 = RSI_THRESHOLDS['weekly_sell_level1']  # 80

                    # 如果历史高点>=80，且当前RSI低于历史高点，说明RSI已经降下来了
                    if max_weekly_rsi >= threshold_80 and latest_weekly_rsi < max_weekly_rsi:
                        # 计算距离高点的天数（处理时区）
                        max_idx_naive = max_weekly_rsi_idx.tz_localize(None) if max_weekly_rsi_idx.tz is not None else max_weekly_rsi_idx
                        days_elapsed = (pd.Timestamp(end_date) - max_idx_naive).days

                        # 只显示最近30天内的高点（防止太久的历史数据）
                        if days_elapsed <= RSI_HIGH_LOOKBACK_DAYS and days_elapsed > 0:
                            rsi_high_warnings.append(RSIHighInfo(
                                high_date=max_weekly_rsi_idx.strftime('%Y-%m-%d'),
                                high_rsi=max_weekly_rsi,
                                rsi_type='周线',
                                current_rsi=latest_weekly_rsi,
                                days_elapsed=days_elapsed
                            ))

                            # 根据高点值确定告警级别描述
                            if max_weekly_rsi >= threshold_90:
                                level_desc = "清仓警戒(RSI>90)"
                            elif max_weekly_rsi >= threshold_85:
                                level_desc = "卖出1/2警戒(RSI>85)"
                            else:
                                level_desc = "卖出1/3警戒(RSI>80)"

                            self.logger.warning(f"  [历史RSI高点] {days_elapsed}天前周线RSI达{max_weekly_rsi:.2f}({level_desc})，当前已降至{latest_weekly_rsi:.2f}，等待SAR死叉触发")

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
            latest_index_daily_rsi_kc=latest_index_rsi_kc,
            latest_index_daily_rsi_bj=latest_index_rsi_bj,
            latest_index_daily_rsi_hsTech=latest_index_rsi_hsTech,
            latest_index_daily_rsi_hsIndex=latest_index_rsi_hsIndex,
            signals=signals,
            notes=notes,
            judge_buy_ids=stock_config.judge_buy_ids,
            judge_t_ids=stock_config.judge_t_ids,
            judge_sell_ids=stock_config.judge_sell_ids,
            sar_cross_down_events=sar_cross_down_events,
            pending_warnings=pending_warnings,
            rsi_high_warnings=rsi_high_warnings,
            index_divergence_warnings=index_divergence_warnings
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

                # 输出待触发告警
                if len(r.pending_warnings) > 0:
                    self.logger.info(f"  ⚠️ 待触发卖出信号 ({len(r.pending_warnings)} 个):")
                    for warning in r.pending_warnings:
                        severity_str = "🔴" if warning['severity'] == 'HIGH' else "🟡" if warning['severity'] == 'MEDIUM' else "🟢"
                        self.logger.info(f"    {severity_str} [{warning['type']}] {warning['detail']}")

        # 输出待触发告警汇总（包括无信号的股票）
        all_warnings = [(r, w) for r in sorted_results for w in r.pending_warnings]
        if len(all_warnings) > 0:
            self.logger.info("")
            self.logger.info("【待触发卖出信号汇总】")
            self.logger.info("-" * 80)
            for r, warning in all_warnings:
                severity_str = "🔴" if warning['severity'] == 'HIGH' else "🟡" if warning['severity'] == 'MEDIUM' else "🟢"
                self.logger.info(f"  {severity_str} {r.stock_name}({r.symbol}): [{warning['type']}] {warning['detail']}")

        # 输出历史RSI高点告警汇总
        all_rsi_high_warnings = [(r, w) for r in sorted_results for w in r.rsi_high_warnings]
        if len(all_rsi_high_warnings) > 0:
            self.logger.info("")
            self.logger.info("【历史RSI高点告警汇总】（RSI已降但未触发死叉）")
            self.logger.info("-" * 80)
            for r, warning in all_rsi_high_warnings:
                self.logger.info(f"  ⚠️ {r.stock_name}({r.symbol}): {warning.days_elapsed}天前{warning.rsi_type}RSI达{warning.high_rsi:.2f}，当前已降至{warning.current_rsi:.2f}")

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

        # 收集有待触发告警的股票
        results_with_warnings = [r for r in results if len(r.pending_warnings) > 0]

        # 收集有历史RSI高点告警的股票
        results_with_rsi_high = [r for r in results if len(r.rsi_high_warnings) > 0]

        # 收集有指数顶背离告警的股票（去重，所有股票共享同一组指数背离信息）
        results_with_index_div = [r for r in results if len(r.index_divergence_warnings) > 0]

        # 如果没有信号且没有待触发告警且没有历史RSI高点告警且没有指数背离告警，不发送告警
        if len(results_with_signals) == 0 and len(results_with_warnings) == 0 and len(results_with_rsi_high) == 0 and len(results_with_index_div) == 0:
            self.logger.info("无信号且无待触发告警且无历史RSI高点告警且无指数背离告警，不发送钉钉告警")
            return

        # 构建Markdown消息
        title = f"股票监控告警 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        content_lines = []
        content_lines.append(f"## 股票监控告警\n")
        content_lines.append(f"**监控时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content_lines.append(f"**监控范围**: 最近 {RECENT_TRADING_DAYS} 个交易日\n")
        content_lines.append(f"**监控股票数**: {len(results)} 只\n")
        content_lines.append(f"**有信号股票**: {len(results_with_signals)} 只\n")
        content_lines.append(f"**待触发告警**: {len(results_with_warnings)} 只\n")
        content_lines.append(f"**历史RSI高点**: {len(results_with_rsi_high)} 只\n\n")

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

        # 待触发告警（已形成卖点但还没触发死叉）
        if len(results_with_warnings) > 0:
            # 按严重程度分组
            high_warnings = [(r, w) for r in results_with_warnings for w in r.pending_warnings if w['severity'] == 'HIGH']
            medium_warnings = [(r, w) for r in results_with_warnings for w in r.pending_warnings if w['severity'] == 'MEDIUM']
            low_warnings = [(r, w) for r in results_with_warnings for w in r.pending_warnings if w['severity'] == 'LOW']

            total_warnings = len(high_warnings) + len(medium_warnings) + len(low_warnings)
            content_lines.append(f"### ⚠️ 待触发卖出信号 ({total_warnings}条)\n")
            content_lines.append("---\n")

            # 高严重度告警
            if len(high_warnings) > 0:
                content_lines.append(f"**🔴 高警戒 ({len(high_warnings)}条)**\n")
                for r, warning in high_warnings:
                    content_lines.append(f"- **{r.stock_name}** ({r.symbol}): [{warning['type']}] {warning['detail']}\n")
                content_lines.append("\n")

            # 中严重度告警
            if len(medium_warnings) > 0:
                content_lines.append(f"**🟡 中警戒 ({len(medium_warnings)}条)**\n")
                for r, warning in medium_warnings:
                    content_lines.append(f"- **{r.stock_name}** ({r.symbol}): [{warning['type']}] {warning['detail']}\n")
                content_lines.append("\n")

            # 低严重度告警
            if len(low_warnings) > 0:
                content_lines.append(f"**🟢 低警戒 ({len(low_warnings)}条)**\n")
                for r, warning in low_warnings:
                    content_lines.append(f"- **{r.stock_name}** ({r.symbol}): [{warning['type']}] {warning['detail']}\n")
                content_lines.append("\n")

        # 历史RSI高点告警（RSI已降但未触发死叉）
        if len(results_with_rsi_high) > 0:
            all_rsi_high_warnings = [(r, w) for r in results_with_rsi_high for w in r.rsi_high_warnings]
            content_lines.append(f"### 📉 历史RSI高点告警 ({len(all_rsi_high_warnings)}条)\n")
            content_lines.append("---\n")
            content_lines.append("*RSI已降但未触发SAR死叉，需持续关注*\n\n")
            for r, warning in all_rsi_high_warnings:
                content_lines.append(f"- **{r.stock_name}** ({r.symbol}): {warning.days_elapsed}天前{warning.rsi_type}RSI达**{warning.high_rsi:.2f}**，当前已降至{warning.current_rsi:.2f}\n")
            content_lines.append("\n")

        # 指数顶背离告警（独立于个股，所有股票共享同一组指数背离信息）
        if len(results_with_index_div) > 0:
            # 去重：取第一个股票的指数背离信息（所有股票的指数背离信息相同）
            all_index_warnings = results_with_index_div[0].index_divergence_warnings
            # 去重：同一指数同一类型的背离只显示一次
            unique_warnings = {}
            for w in all_index_warnings:
                key = f"{w['index_name']}-{w['type']}"
                if key not in unique_warnings:
                    unique_warnings[key] = w

            if len(unique_warnings) > 0:
                content_lines.append(f"### 🚨 指数顶背离告警 ({len(unique_warnings)}条)\n")
                content_lines.append("---\n")
                content_lines.append("*指数顶背离对市场整体有影响，需重点关注*\n\n")
                for key, warning in unique_warnings.items():
                    content_lines.append(f"- **{warning['index_name']}** ({warning['index_code']}): [{warning['type']}]\n")
                    content_lines.append(f"  - 前高点({warning['peak_a_date']}): 价格={warning['peak_a_price']:.2f} MACD={warning['peak_a_macd']:.4f}\n")
                    content_lines.append(f"  - 当前高点({warning['peak_b_date']}): 价格={warning['peak_b_price']:.2f} MACD={warning['peak_b_macd']:.4f}\n")
                content_lines.append("\n")

        # 市场概况
        content_lines.append(f"### 📊 市场概况\n")
        content_lines.append("---\n")
        if len(results) > 0:
            cyb_rsi = results[0].latest_index_daily_rsi_cyb
            sh_rsi = results[0].latest_index_daily_rsi_sh
            kc_rsi = results[0].latest_index_daily_rsi_kc
            bj_rsi = results[0].latest_index_daily_rsi_bj
            hsTech_rsi = results[0].latest_index_daily_rsi_hsTech
            hsIndex_rsi = results[0].latest_index_daily_rsi_hsIndex

            cyb_rsi_str = f"{cyb_rsi:.2f}" if not np.isnan(cyb_rsi) else "N/A"
            sh_rsi_str = f"{sh_rsi:.2f}" if not np.isnan(sh_rsi) else "N/A"
            kc_rsi_str = f"{kc_rsi:.2f}" if not np.isnan(kc_rsi) else "N/A"
            bj_rsi_str = f"{bj_rsi:.2f}" if not np.isnan(bj_rsi) else "N/A"
            hsTech_rsi_str = f"{hsTech_rsi:.2f}" if not np.isnan(hsTech_rsi) else "N/A"
            hsIndex_rsi_str = f"{hsIndex_rsi:.2f}" if not np.isnan(hsIndex_rsi) else "N/A"

            content_lines.append(f"- 创业板指数(399006) RSI: {cyb_rsi_str}\n")
            content_lines.append(f"- 上证指数(000001) RSI: {sh_rsi_str}\n")
            content_lines.append(f"- 科创综指(000680) RSI: {kc_rsi_str}\n")
            content_lines.append(f"- 北证板块(899050) RSI: {bj_rsi_str}\n")
            content_lines.append(f"- 恒生科技ETF(513180) RSI: {hsTech_rsi_str}\n")
            content_lines.append(f"- 恒生指数ETF(159920) RSI: {hsIndex_rsi_str}\n")

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
                # 构建历史RSI高点信息字符串
                rsi_high_str = ''
                if len(r.rsi_high_warnings) > 0:
                    rsi_high_str = '; '.join([f"{w.days_elapsed}天前{w.rsi_type}RSI达{w.high_rsi:.2f}" for w in r.rsi_high_warnings])

                # 构建待触发告警信息字符串
                pending_str = ''
                if len(r.pending_warnings) > 0:
                    pending_str = '; '.join([f"[{w['type']}]" for w in r.pending_warnings])

                # 构建指数顶背离告警信息字符串
                index_div_str = ''
                if len(r.index_divergence_warnings) > 0:
                    index_div_str = '; '.join([f"[{w['index_name']}{w['type']}]" for w in r.index_divergence_warnings])

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
                    '历史RSI高点': rsi_high_str,
                    '待触发告警': pending_str,
                    '指数顶背离': index_div_str,
                })

            # 如果没有信号，也记录一行
            if len(r.signals) == 0:
                # 构建历史RSI高点信息字符串
                rsi_high_str = ''
                if len(r.rsi_high_warnings) > 0:
                    rsi_high_str = '; '.join([f"{w.days_elapsed}天前{w.rsi_type}RSI达{w.high_rsi:.2f}" for w in r.rsi_high_warnings])

                # 构建待触发告警信息字符串
                pending_str = ''
                if len(r.pending_warnings) > 0:
                    pending_str = '; '.join([f"[{w['type']}]" for w in r.pending_warnings])

                # 构建指数顶背离告警信息字符串
                index_div_str = ''
                if len(r.index_divergence_warnings) > 0:
                    index_div_str = '; '.join([f"[{w['index_name']}{w['type']}]" for w in r.index_divergence_warnings])

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
                    '历史RSI高点': rsi_high_str,
                    '待触发告警': pending_str,
                    '指数顶背离': index_div_str,
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


def is_trading_time() -> bool:
    """
    检查当前是否在交易时间内

    A股交易时间：
    - 上午：9:30 - 11:30
    - 下午：13:00 - 15:00

    Returns:
        bool: 是否在交易时间内
    """
    current_time = datetime.now().strftime('%H:%M')
    return current_time in MONITOR_TIMES


def should_monitor_now() -> bool:
    """
    检查当前时间是否应该执行监控

    Returns:
        bool: 是否应该执行监控
    """
    return is_trading_time()


def run_scheduler() -> None:
    """
    运行定时调度器

    在交易日的交易时间内每30分钟执行监控：
    - 上午：9:30, 10:00, 10:30, 11:00, 11:30
    - 下午：13:00, 13:30, 14:00, 14:30, 15:00
    """
    print("=" * 60)
    print("日内监控调度器启动")
    print(f"监控时间点（交易时间内每30分钟）:")
    print(f"  上午: {MORNING_MONITOR_TIMES}")
    print(f"  下午: {AFTERNOON_MONITOR_TIMES}")
    print(f"监控范围: 最近 {RECENT_TRADING_DAYS} 个交易日")
    print("=" * 60)

    monitor = DailyMonitor()
    last_monitor_time: Optional[datetime] = None

    while True:
        now = datetime.now()
        current_time = now.strftime('%H:%M')

        if is_trading_day():
            # 检查是否在交易时间内
            if current_time in MONITOR_TIMES:
                # 检查是否已经执行过（避免同一时间点重复执行）
                should_run = False
                if last_monitor_time is None:
                    should_run = True
                else:
                    # 如果距离上次执行超过25分钟，则执行（留5分钟余量避免重复）
                    elapsed = (now - last_monitor_time).total_seconds()
                    if elapsed >= MONITOR_INTERVAL - 300:
                        should_run = True

                if should_run:
                    print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发监控...")
                    monitor.run_monitor()
                    last_monitor_time = now

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