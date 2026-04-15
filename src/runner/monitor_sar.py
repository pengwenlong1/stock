# -*- coding: utf-8 -*-
"""
SAR跌破监控脚本 (monitor_sar.py)

【功能说明】
监控股票是否发生SAR跌破（绿转红）信号。
SAR跌破意味着趋势从多头转为空头，是重要的卖出信号。

【SAR指标原理】
- SAR点位于价格上方 → 红色 → 穠头市场（做空信号）
- SAR点位于价格下方 → 绿色 → 多头市场（做多信号）
- 绿转红：SAR从价格下方转到上方，趋势从多头转为空头（跌破信号）

【监控时间】
交易时间每小时执行一次（10:00, 11:00, 13:00, 14:00）

【告警方式】
通过钉钉机器人发送告警消息。

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

# 导入SAR策略类
from src.tool.sar_strategy import SARStrategy, SARSignal

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

# 监控时间点（交易时间内每小时）
MONITOR_TIMES = ['10:00', '11:00', '13:00', '14:00']

# SAR参数：SAR(10,2,20)
# acceleration = 2% = 0.02
# maximum = 20% = 0.20
SAR_ACCELERATION = 0.02
SAR_MAXIMUM = 0.20

# 预热天数（用于SAR计算）
WARMUP_DAYS = 60

# 检测信号的时间范围（向前推算天数）
SIGNAL_LOOKBACK_DAYS = 2


# ==================== 数据结构 ====================

@dataclass
class StockConfig:
    """股票配置"""
    symbol: str
    stock_name: str
    active: bool


@dataclass
class SARAlertInfo:
    """SAR告警信息"""
    symbol: str
    stock_name: str
    signal_date: str
    signal_type: str               # '绿转红' 或 '红转绿'
    sar_value: float
    close_price: float
    current_position: str          # '多头' 或 '空头'


# ==================== 钉钉告警类 ====================

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
                logging.error(f"钉钉发送失败: {result}")
                return False

        except Exception as e:
            logging.error(f"钉钉发送异常: {e}")
            return False


# ==================== SAR监控器类 ====================

class SARMonitor:
    """
    SAR跌破监控器

    【功能说明】
    监控股票的SAR指标转折信号，重点关注"绿转红"（跌破信号）。

    【监控逻辑】
    1. 加载 stocks_backtest.csv 中 active=1 的股票
    2. 获取历史数据计算SAR
    3. 检测最近 SIGNAL_LOOKBACK_DAYS 天的转折信号
    4. 发现"绿转红"信号时发送钉钉告警
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
            output_dir = os.path.join(project_root, 'logs', 'sar_monitor')
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 先配置日志（确保logger可用）
        self._setup_logging()

        # 初始化掘金token和钉钉配置
        self._init_settings()

    def _setup_logging(self) -> None:
        """配置日志"""
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(self.output_dir, f'sar_monitor_{timestamp}.log')

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

            stock_config = StockConfig(
                symbol=symbol,
                stock_name=stock_name,
                active=True
            )
            stocks.append(stock_config)

        self.logger.info(f"加载股票配置: {len(stocks)} 只（active=1）")
        return stocks

    def get_recent_trading_dates(self, end_date: str, count: int = 5) -> List[str]:
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
                while len(dates) < count + 5:
                    if current.weekday() < 5:  # 周一到周五
                        dates.append(current.strftime('%Y-%m-%d'))
                    current = current - pd.Timedelta(days=1)
                return dates[:count]

            # 获取最近15天的交易日列表
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

    def get_historical_data(self,
                            symbol: str,
                            end_date: str) -> Optional[pd.DataFrame]:
        """
        获取历史数据（包含当天的实时数据）

        【数据获取逻辑】
        1. 使用history()获取历史日线数据
        2. 使用current()获取当天实时行情
        3. 将实时数据追加到历史数据，合成完整的日线数据

        Args:
            symbol: 股票代码
            end_date: 结束日期

        Returns:
            DataFrame: 包含 open, high, low, close 列
        """
        if history is None:
            self.logger.error("未安装掘金量化SDK")
            return None

        warmup_start = (pd.Timestamp(end_date) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')

        try:
            # 获取历史日线数据
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
            if daily_data['eob'].dt.tz is None:
                daily_data['eob'] = daily_data['eob'].dt.tz_localize('Asia/Shanghai')
            else:
                daily_data['eob'] = daily_data['eob'].dt.tz_convert('Asia/Shanghai')
            df = daily_data.set_index('eob').sort_index()

            # 尝试获取当天实时数据并追加
            if current is not None:
                try:
                    realtime = current(symbols=symbol)
                    if realtime and len(realtime) > 0:
                        rt = realtime[0]
                        # 检查实时数据的时间是否为当天
                        rt_time = pd.Timestamp(rt['created_at'])
                        rt_date = rt_time.strftime('%Y-%m-%d')

                        # 如果当天数据不存在于历史数据中，追加实时数据
                        if rt_date not in [d.strftime('%Y-%m-%d') for d in df.index]:
                            rt_row = pd.DataFrame({
                                'open': [rt['open']],
                                'high': [rt['high']],
                                'low': [rt['low']],
                                'close': [rt['price']],  # 当前价格作为收盘价
                                'volume': [rt['cum_volume']]
                            }, index=[rt_time])
                            df = pd.concat([df, rt_row])
                            self.logger.info(f"追加当天实时数据: {symbol} {rt_date} "
                                           f"open={rt['open']:.4f}, high={rt['high']:.4f}, "
                                           f"low={rt['low']:.4f}, price={rt['price']:.4f}")
                except Exception as e:
                    self.logger.warning(f"获取实时数据失败: {symbol} - {e}")

            return df.sort_index()

        except Exception as e:
            self.logger.error(f"获取历史数据异常: {symbol} - {e}")
            return None

    def detect_sar_signals(self,
                           stock_config: StockConfig,
                           end_date: str) -> List[SARAlertInfo]:
        """
        检测SAR转折信号

        Args:
            stock_config: 股票配置
            end_date: 监控日期

        Returns:
            List[SARAlertInfo]: 检测到的信号列表
        """
        symbol = stock_config.symbol

        # 获取历史数据
        df = self.get_historical_data(symbol, end_date)
        if df is None or df.empty:
            self.logger.warning(f"无法获取数据: {symbol}")
            return []

        # 创建价格序列
        high_series = pd.Series(df['high'].values, index=df.index, name='High')
        low_series = pd.Series(df['low'].values, index=df.index, name='Low')
        close_series = pd.Series(df['close'].values, index=df.index, name='Close')

        # 创建SAR策略实例
        strategy = SARStrategy(
            acceleration=SAR_ACCELERATION,
            maximum=SAR_MAXIMUM
        )
        strategy.prepare_data(high_series, low_series, close_series)

        # 计算检测时间段的开始日期
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=SIGNAL_LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        # 告警列表
        alerts = []

        try:
            signals = strategy.detect_signals(start_date, end_date)

            for signal in signals:
                # 只关注绿转红信号（跌破信号，类似均线死叉）
                if signal.signal_type == '绿转红':
                    signal_date_str = signal.date.strftime('%Y-%m-%d')
                    alert = SARAlertInfo(
                        symbol=symbol,
                        stock_name=stock_config.stock_name,
                        signal_date=signal_date_str,
                        signal_type=signal.signal_type,
                        sar_value=signal.sar_value,
                        close_price=signal.close_price,
                        current_position='空头'
                    )
                    alerts.append(alert)
                    self.logger.info(f"[{signal_date_str}] {stock_config.stock_name} ({symbol}) "
                                   f"绿转红信号（跌破）: SAR={signal.sar_value:.3f}, Close={signal.close_price:.3f}")

        except Exception as e:
            self.logger.error(f"检测SAR信号异常: {symbol} - {e}")

        return alerts

    def run_monitor(self) -> List[SARAlertInfo]:
        """
        执行监控

        Returns:
            List[SARAlertInfo]: 检测到的告警列表
        """
        self.logger.info("=" * 80)
        self.logger.info("SAR跌破监控启动")
        self.logger.info(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"SAR参数: acceleration={SAR_ACCELERATION}, maximum={SAR_MAXIMUM}")
        self.logger.info(f"检测范围: 最近 {SIGNAL_LOOKBACK_DAYS} 天")
        self.logger.info("=" * 80)

        # 加载股票配置
        stocks = self.load_stock_config()
        if len(stocks) == 0:
            self.logger.error("无活跃股票配置")
            return []

        # 获取当前日期
        today = datetime.now().strftime('%Y-%m-%d')

        # 告警列表
        all_alerts = []

        # 逐只股票监控
        for stock_config in stocks:
            self.logger.info(f"监控股票: {stock_config.symbol} ({stock_config.stock_name})")
            alerts = self.detect_sar_signals(stock_config, today)
            all_alerts.extend(alerts)

        # 输出监控报告
        self._output_report(all_alerts)

        # 发送钉钉告警
        self._send_dingtalk_alert(all_alerts)

        return all_alerts

    def _output_report(self, alerts: List[SARAlertInfo]) -> None:
        """
        输出监控报告

        Args:
            alerts: 告警列表
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("SAR监控报告")
        self.logger.info("=" * 80)

        if len(alerts) == 0:
            self.logger.info("无SAR绿转红信号（跌破信号）")
        else:
            self.logger.info(f"检测到 {len(alerts)} 个绿转红信号（跌破/死叉信号）")
            self.logger.info("-" * 60)

            for alert in alerts:
                self.logger.info(f"[{alert.signal_date}] {alert.stock_name} ({alert.symbol})")
                self.logger.info(f"  SAR值: {alert.sar_value:.3f}")
                self.logger.info(f"  收盘价: {alert.close_price:.3f}")
                self.logger.info(f"  转折类型: {alert.signal_type}（趋势转为空头，类似均线死叉）")
                self.logger.info("")

        self.logger.info("=" * 80)

    def _send_dingtalk_alert(self, alerts: List[SARAlertInfo]) -> None:
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
            self.logger.info("无SAR红转绿信号，不发送钉钉告警")
            return

        # 构建Markdown消息
        title = f"SAR跌破告警 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        content_lines = []
        content_lines.append(f"## SAR跌破告警\n")
        content_lines.append(f"**监控时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        content_lines.append(f"**检测范围**: 最近 {SIGNAL_LOOKBACK_DAYS} 天\n")
        content_lines.append(f"**SAR参数**: acceleration={SAR_ACCELERATION}, maximum={SAR_MAXIMUM}\n")
        content_lines.append(f"**跌破股票数**: {len(alerts)} 只\n\n")

        content_lines.append(f"### 绿转红信号详情（跌破）\n")
        content_lines.append("---\n")

        # 按日期分组
        alerts_by_date: Dict[str, List[SARAlertInfo]] = {}
        for alert in alerts:
            if alert.signal_date not in alerts_by_date:
                alerts_by_date[alert.signal_date] = []
            alerts_by_date[alert.signal_date].append(alert)

        for date in sorted(alerts_by_date.keys(), reverse=True):
            date_alerts = alerts_by_date[date]
            content_lines.append(f"**日期: {date}** ({len(date_alerts)}只)\n")
            for alert in date_alerts:
                content_lines.append(f"- **{alert.stock_name}** ({alert.symbol})\n")
                content_lines.append(f"  - SAR值: {alert.sar_value:.3f}\n")
                content_lines.append(f"  - 收盘价: {alert.close_price:.3f}\n")
                content_lines.append(f"  - 信号类型: {alert.signal_type}（趋势转为空头，类似均线死叉）\n\n")

        content_lines.append(f"### 操作建议\n")
        content_lines.append("---\n")
        content_lines.append(f"- SAR绿转红意味着价格跌破SAR支撑位，趋势从多头转为空头\n")
        content_lines.append(f"- 类似于均线死叉，建议关注持仓风险，考虑减仓或止损\n")

        content = "".join(content_lines)

        # 发送钉钉消息
        success = self.dingtalk.send_markdown(title, content, at_all=True)

        if success:
            self.logger.info("钉钉告警发送成功")
        else:
            self.logger.error("钉钉告警发送失败")


# ==================== 定时调度 ====================

def is_trading_time() -> bool:
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
    if not is_trading_day():
        return False

    current_time = datetime.now().strftime('%H:%M')
    return current_time in MONITOR_TIMES


def run_scheduler() -> None:
    """
    运行定时调度器

    交易时间每小时执行一次：
    - 10:00
    - 11:00
    - 13:00
    - 14:00
    """
    print("=" * 60)
    print("SAR跌破监控调度器启动")
    print(f"监控时间点: {MONITOR_TIMES}")
    print(f"检测范围: 最近 {SIGNAL_LOOKBACK_DAYS} 天")
    print("=" * 60)

    monitor = SARMonitor()

    while True:
        now = datetime.now()
        current_time = now.strftime('%H:%M')

        if is_trading_day() and is_trading_time():
            if current_time in MONITOR_TIMES:
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 触发SAR监控...")
                monitor.run_monitor()
                time.sleep(60)  # 避免同一分钟内重复触发

        time.sleep(60)


def run_once() -> None:
    """
    单次执行监控（用于测试或手动触发）
    """
    monitor = SARMonitor()
    monitor.run_monitor()


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAR跌破监控脚本")
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