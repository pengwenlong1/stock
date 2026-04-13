# -*- coding: utf-8 -*-
"""
数据预处理模块 (data_processor.py)

【功能说明】
连接MySQL数据库，将股票数据和计算的技术指标写入以下三张表：
1. stock_info: 股票基本信息表
2. stock_daily_metrics: 股票日线指标明细表
3. trading_calendar: 交易日历表

【数据库配置】
- 库名: stocks
- 用户名: root
- 密码: 123456

【使用方法】
    python src/process/data_processor.py

作者：量化交易团队
"""
import json
import logging
import os
import sys
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入工具类
from src.tool.rsi_calculator import RSI
from src.tool.ma_calculator import MACalculator
from src.tool.macd_calculator import MACDCalculator
from src.tool.peak_detector import PeakDetector
from src.tool.divergence_detector import DivergenceDetector
from src.tool.sar_strategy import SARStrategy

try:
    from gm.api import history, set_token, get_instruments, get_trading_dates, ADJUST_PREV
except ImportError:
    history = None
    set_token = None
    get_instruments = None
    get_trading_dates = None
    ADJUST_PREV = None

try:
    import pymysql
except ImportError:
    pymysql = None
    logging.warning("pymysql未安装，请先安装: pip install pymysql")


# ==================== 数据库配置 ====================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'stocks',
    'charset': 'utf8mb4'
}


# ==================== 数据结构 ====================

def convert_stock_code(symbol: str) -> str:
    """
    将掘金格式的股票代码转换为数据库格式

    Args:
        symbol: 掘金格式代码（如 SHSE.600000 或 SZSE.300661）

    Returns:
        str: 数据库格式代码（如 600000.SH 或 300661.SZ）

    Examples:
        >>> convert_stock_code('SHSE.600000')
        '600000.SH'
        >>> convert_stock_code('SZSE.300661')
        '300661.SZ'
    """
    if '.' in symbol:
        exchange, code = symbol.split('.')
        # 转换交易所代码: SHSE -> SH, SZSE -> SZ
        exchange_short = exchange.replace('SHSE', 'SH').replace('SZSE', 'SZ')
        return f'{code}.{exchange_short}'
    return symbol


@dataclass
class StockInfo:
    """股票基本信息"""
    stock_code: str           # 股票代码（如 SHSE.600000）
    stock_name: str           # 股票名称
    listing_date: str         # 上市日期


@dataclass
class DailyMetrics:
    """日线指标数据"""
    stock_id: int
    trade_date: str
    close_price: float
    ma5: float
    ma10: float
    ma5_ma10_dead_cross: int  # 0-否, 1-是
    rsi_daily: float
    rsi_weekly: float
    macd: float
    is_daily_top_divergence: int    # 0-否, 1-是
    is_daily_bottom_divergence: int  # 0-否, 1-是
    is_weekly_top_divergence: int    # 0-否, 1-是
    is_weekly_bottom_divergence: int # 0-否, 1-是
    is_sar_dead_cross: int           # 0-否, 1-是
    is_local_high: int               # 0-否, 1-是
    is_local_low: int               # 0-否, 1-是
    status: int                     # 0-完成, 1-有遗漏


# ==================== 数据库连接器 ====================

class MySQLConnector:
    """
    MySQL数据库连接器

    【功能说明】
    管理MySQL数据库连接，执行SQL操作。
    """

    def __init__(self, config: Dict = None) -> None:
        """
        初始化连接器

        Args:
            config: 数据库配置字典
        """
        if config is None:
            config = DB_CONFIG

        self.config = config
        self.connection = None

        if pymysql is None:
            raise ImportError("请先安装pymysql: pip install pymysql")

    def connect(self) -> bool:
        """
        连接数据库

        Returns:
            bool: 是否成功连接
        """
        try:
            self.connection = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset']
            )
            logging.info(f"MySQL连接成功: {self.config['host']}:{self.config['port']}/{self.config['database']}")
            return True
        except Exception as e:
            logging.error(f"MySQL连接失败: {e}")
            return False

    def disconnect(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logging.info("MySQL连接已关闭")

    def execute(self, sql: str, params: tuple = None) -> int:
        """
        执行SQL语句

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            int: 影响的行数
        """
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            self.connection.rollback()
            logging.error(f"SQL执行失败: {e}\nSQL: {sql}")
            return 0

    def execute_batch(self, sql: str, params_list: List[tuple]) -> int:
        """
        批量执行SQL语句

        Args:
            sql: SQL语句
            params_list: 参数列表

        Returns:
            int: 影响的行数
        """
        if not self.connection:
            self.connect()

        if not params_list or len(params_list) == 0:
            return 0

        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(sql, params_list)
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            self.connection.rollback()
            logging.error(f"批量执行失败: {e}\nSQL: {sql}")
            return 0

    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        """
        查询数据

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            List[tuple]: 查询结果列表
        """
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"查询失败: {e}\nSQL: {sql}")
            return []

    def query_one(self, sql: str, params: tuple = None) -> Optional[tuple]:
        """
        查询单条数据

        Args:
            sql: SQL语句
            params: 参数元组

        Returns:
            tuple: 单条查询结果
        """
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchone()
        except Exception as e:
            logging.error(f"查询失败: {e}\nSQL: {sql}")
            return None

    def get_stock_id(self, stock_code: str) -> Optional[int]:
        """
        获取股票ID

        Args:
            stock_code: 股票代码

        Returns:
            int: 股票ID，不存在返回None
        """
        result = self.query_one(
            "SELECT stock_id FROM stock_info WHERE stock_code = %s",
            (stock_code,)
        )
        if result:
            return result[0]
        return None

    def insert_stock_info(self, stock: StockInfo) -> int:
        """
        插入股票基本信息

        Args:
            stock: 股票信息对象

        Returns:
            int: 插入后的stock_id
        """
        sql = """
        INSERT INTO stock_info (stock_code, stock_name, listing_date)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE stock_name = %s, updated_at = NOW()
        """
        self.execute(sql, (stock.stock_code, stock.stock_name, stock.listing_date, stock.stock_name))
        return self.get_stock_id(stock.stock_code)

    def insert_daily_metrics(self, metrics: DailyMetrics) -> int:
        """
        插入日线指标数据

        Args:
            metrics: 日线指标对象

        Returns:
            int: 影响的行数
        """
        sql = """
        INSERT INTO stock_daily_metrics
        (stock_id, trade_date, close_price, ma5, ma10, ma5_ma10_dead_cross,
         rsi_daily, rsi_weekly, macd, is_daily_top_divergence, is_daily_bottom_divergence,
         is_weekly_top_divergence, is_weekly_bottom_divergence, is_sar_dead_cross,
         is_local_high, is_local_low, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        close_price = %s, ma5 = %s, ma10 = %s, ma5_ma10_dead_cross = %s,
        rsi_daily = %s, rsi_weekly = %s, macd = %s,
        is_daily_top_divergence = %s, is_daily_bottom_divergence = %s,
        is_weekly_top_divergence = %s, is_weekly_bottom_divergence = %s, is_sar_dead_cross = %s,
        is_local_high = %s, is_local_low = %s, status = %s, updated_at = NOW()
        """
        params = (
            metrics.stock_id, metrics.trade_date, metrics.close_price,
            metrics.ma5, metrics.ma10, metrics.ma5_ma10_dead_cross,
            metrics.rsi_daily, metrics.rsi_weekly, metrics.macd,
            metrics.is_daily_top_divergence, metrics.is_daily_bottom_divergence,
            metrics.is_weekly_top_divergence, metrics.is_weekly_bottom_divergence, metrics.is_sar_dead_cross,
            metrics.is_local_high, metrics.is_local_low, metrics.status,
            # UPDATE部分
            metrics.close_price, metrics.ma5, metrics.ma10, metrics.ma5_ma10_dead_cross,
            metrics.rsi_daily, metrics.rsi_weekly, metrics.macd,
            metrics.is_daily_top_divergence, metrics.is_daily_bottom_divergence,
            metrics.is_weekly_top_divergence, metrics.is_weekly_bottom_divergence, metrics.is_sar_dead_cross,
            metrics.is_local_high, metrics.is_local_low, metrics.status
        )
        return self.execute(sql, params)

    def insert_trading_dates(self, dates: List[str]) -> int:
        """
        批量插入交易日历

        Args:
            dates: 日期列表（YYYY-MM-DD格式）

        Returns:
            int: 影响的行数
        """
        if not dates or len(dates) == 0:
            return 0

        sql = "INSERT IGNORE INTO trading_calendar (trade_date) VALUES (%s)"
        params_list = [(d,) for d in dates]
        return self.execute_batch(sql, params_list)

    def create_tables(self) -> bool:
        """
        创建数据库表（如果不存在）

        Returns:
            bool: 是否成功
        """
        try:
            # stock_info表
            sql_stock_info = """
            CREATE TABLE IF NOT EXISTS stock_info (
                stock_id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT COMMENT '股票ID（自增主键）',
                stock_code VARCHAR(10) NOT NULL UNIQUE COMMENT '股票代码（如 600000.SH）',
                stock_name VARCHAR(50) NOT NULL COMMENT '股票名称',
                listing_date DATE NOT NULL COMMENT '上市日期',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_stock_code (stock_code),
                INDEX idx_listing_date (listing_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """

            # stock_daily_metrics表
            sql_daily_metrics = """
            CREATE TABLE IF NOT EXISTS stock_daily_metrics (
                id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
                stock_id BIGINT UNSIGNED NOT NULL COMMENT '关联 stock_info.stock_id',
                trade_date DATE NOT NULL COMMENT '交易日（工作日）',
                close_price DECIMAL(10,2) NOT NULL COMMENT '收盘价（代表当日股价）',
                ma5 DECIMAL(10,2) COMMENT '5日均线',
                ma10 DECIMAL(10,2) COMMENT '10日均线',
                ma5_ma10_dead_cross TINYINT(1) NOT NULL DEFAULT 0 COMMENT '5日与10日均线死叉：0-否, 1-是',
                rsi_daily DECIMAL(5,2) COMMENT '日线RSI（0~100）',
                rsi_weekly DECIMAL(5,2) COMMENT '周线RSI',
                macd DECIMAL(10,4) COMMENT 'MACD值',
                is_daily_top_divergence TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否日线顶背离：0-否, 1-是',
                is_daily_bottom_divergergence TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否日线底背离：0-否, 1-是',
                is_weekly_top_divergence TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否周线顶背离：0-否, 1-是',
                is_weekly_bottom_divergergence TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否周线底背离：0-否, 1-是',
                is_sar_dead_cross TINYINT(1) NOT NULL DEFAULT 0 COMMENT 'SAR死叉：0-否, 1-是',
                is_local_high TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否局部高点：0-否, 1-是',
                is_local_low TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否局部低点：0-否, 1-是',
                status TINYINT(1) NOT NULL DEFAULT 0 COMMENT '计算状态：0-完成, 1-有遗漏',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_stock_date (stock_id, trade_date),
                INDEX idx_trade_date (trade_date),
                INDEX idx_stock_status (stock_id, status),
                INDEX idx_date_status (trade_date, status),
                CONSTRAINT fk_stock_daily_stock_id
                    FOREIGN KEY (stock_id) REFERENCES stock_info(stock_id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """

            # trading_calendar表
            sql_trading_calendar = """
            CREATE TABLE IF NOT EXISTS trading_calendar (
                trade_date DATE PRIMARY KEY COMMENT '工作日（交易日）'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """

            self.execute(sql_stock_info)
            self.execute(sql_daily_metrics)
            self.execute(sql_trading_calendar)

            logging.info("数据库表创建成功")
            return True

        except Exception as e:
            logging.error(f"数据库表创建失败: {e}")
            return False


# ==================== 数据处理器 ====================

class DataProcessor:
    """
    数据处理器

    【功能说明】
    从掘金SDK获取股票数据，计算技术指标，写入MySQL数据库。
    """

    def __init__(self, db_connector: MySQLConnector = None) -> None:
        """
        初始化数据处理器

        Args:
            db_connector: 数据库连接器对象
        """
        if db_connector is None:
            db_connector = MySQLConnector()

        self.db = db_connector
        self._init_gm_token()

        # 配置日志
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
                    logging.info("掘金token已设置")

    def _setup_logging(self) -> None:
        """配置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler()
            ]
        )

    def process_stock(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        warmup_days: int = 365
    ) -> bool:
        """
        处理单只股票数据

        Args:
            symbol: 股票代码（掘金格式，如 SHSE.600000）
            start_date: 处理开始日期
            end_date: 处理结束日期
            warmup_days: 预热天数（用于计算指标）

        Returns:
            bool: 是否成功
        """
        if history is None:
            logging.error("掘金SDK未安装，请先安装: pip install gm")
            return False

        logging.info(f"开始处理股票: {symbol}")

        try:
            # 1. 获取股票基本信息
            stock_info = self._get_stock_info(symbol)
            if stock_info is None:
                logging.error(f"获取股票信息失败: {symbol}")
                return False

            # 2. 插入stock_info表并获取stock_id
            stock_id = self.db.insert_stock_info(stock_info)
            if stock_id is None:
                logging.error(f"插入股票信息失败: {symbol}")
                return False

            logging.info(f"股票ID: {stock_id}, 名称: {stock_info.stock_name}")

            # 3. 获取历史数据（包含预热数据）
            warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)).strftime('%Y-%m-%d')

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
                logging.error(f"获取历史数据失败: {symbol}")
                return False

            logging.info(f"获取到 {len(daily_data)} 条日线数据")

            # 4. 构建价格序列
            daily_data['eob'] = pd.to_datetime(daily_data['eob'])
            df = daily_data.set_index('eob').sort_index()

            close_series = pd.Series(df['close'].values, index=df.index, name='Close')
            high_series = pd.Series(df['high'].values, index=df.index, name='High')
            low_series = pd.Series(df['low'].values, index=df.index, name='Low')

            # 5. 计算技术指标
            metrics_list = self._calculate_metrics(
                symbol, close_series, high_series, low_series,
                start_date, end_date
            )

            if len(metrics_list) == 0:
                logging.warning(f"计算指标数据为空: {symbol}")
                return False

            logging.info(f"计算出 {len(metrics_list)} 条指标数据")

            # 6. 写入数据库
            success_count = 0
            for metrics in metrics_list:
                metrics.stock_id = stock_id
                if self.db.insert_daily_metrics(metrics) > 0:
                    success_count += 1

            logging.info(f"写入数据库成功: {success_count}/{len(metrics_list)} 条")

            return success_count > 0

        except Exception as e:
            logging.error(f"处理股票失败: {symbol}, 错误: {e}")
            return False

    def _get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """
        获取股票基本信息

        Args:
            symbol: 股票代码（掘金格式，如 SHSE.600000）

        Returns:
            StockInfo: 股票信息对象
        """
        if get_instruments is None:
            return None

        try:
            instruments = get_instruments(symbol)
            if instruments:
                ins = instruments[0]
                stock_name = getattr(ins, 'sec_name', '未知')
                listing_date = getattr(ins, 'listed_date', None)

                if listing_date:
                    if hasattr(listing_date, 'strftime'):
                        listing_date_str = listing_date.strftime('%Y-%m-%d')
                    else:
                        listing_date_str = str(listing_date)[:10]
                else:
                    listing_date_str = '2000-01-01'  # 默认日期

                # 转换股票代码格式：SHSE.600000 -> 600000.SH
                stock_code_db = convert_stock_code(symbol)

                return StockInfo(
                    stock_code=stock_code_db,
                    stock_name=stock_name,
                    listing_date=listing_date_str
                )
        except Exception as e:
            logging.error(f"获取股票信息异常: {e}")

        return None

    def _calculate_metrics(
        self,
        symbol: str,
        close_series: pd.Series,
        high_series: pd.Series,
        low_series: pd.Series,
        start_date: str,
        end_date: str
    ) -> List[DailyMetrics]:
        """
        计算技术指标

        Args:
            symbol: 股票代码（掘金格式，如 SHSE.600000）
            close_series: 收盘价序列
            high_series: 最高价序列
            low_series: 最低价序列
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[DailyMetrics]: 指标数据列表
        """
        metrics_list = []

        # 筛选目标时间段
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # 处理时区
        if close_series.index.tz is not None:
            start_ts = start_ts.tz_localize(close_series.index.tz)
            end_ts = end_ts.tz_localize(close_series.index.tz)

        target_close = close_series[
            (close_series.index >= start_ts) &
            (close_series.index <= end_ts)
        ]

        if len(target_close) == 0:
            return metrics_list

        # 计算日线RSI
        rsi_daily_calc = RSI(period=6, freq='daily')
        rsi_daily = rsi_daily_calc.calculate(close_series)

        # 计算周线RSI
        rsi_weekly_calc = RSI(period=6, freq='weekly')
        rsi_weekly = rsi_weekly_calc.calculate(close_series)

        # 计算MA5/MA10
        ma_calc = MACalculator(fast_period=5, slow_period=10, ma_type='SMA')
        ma5, ma10 = ma_calc.calculate(close_series)

        # 计算MACD
        macd_calc = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
        macd_calc.prepare_data(close_series)
        macd_series = macd_calc._macd_series

        # 计算波峰波谷（局部高/低点）
        peak_detector = PeakDetector(atr_period=14, prominence_factor=1.0, min_distance=20)
        peak_detector.prepare_data(high_series, low_series, close_series)

        # 计算背离
        divergence_detector = DivergenceDetector()
        divergence_detector.prepare_data(
            symbol=symbol,
            start_date=close_series.index[0].strftime('%Y-%m-%d'),
            end_date=close_series.index[-1].strftime('%Y-%m-%d')
        )
        divergences = divergence_detector.detect_all_divergences()

        # 收集背离日期
        daily_top_div_dates = set()
        daily_bottom_div_dates = set()
        weekly_top_div_dates = set()
        weekly_bottom_div_dates = set()

        # 日线顶背离
        for div in divergences.get('daily_top_formed', []):
            daily_top_div_dates.add(div.date.strftime('%Y-%m-%d'))
        for div in divergences.get('daily_top_confirmed', []):
            daily_top_div_dates.add(div.date.strftime('%Y-%m-%d'))

        # 日线底背离
        for div in divergences.get('daily_bottom_formed', []):
            daily_bottom_div_dates.add(div.date.strftime('%Y-%m-%d'))
        for div in divergences.get('daily_bottom_confirmed', []):
            daily_bottom_div_dates.add(div.date.strftime('%Y-%m-%d'))

        # 周线顶背离
        for div in divergences.get('weekly_top_formed', []):
            weekly_top_div_dates.add(div.date.strftime('%Y-%m-%d'))
        for div in divergences.get('weekly_top_confirmed', []):
            weekly_top_div_dates.add(div.date.strftime('%Y-%m-%d'))

        # 周线底背离
        for div in divergences.get('weekly_bottom_formed', []):
            weekly_bottom_div_dates.add(div.date.strftime('%Y-%m-%d'))
        for div in divergences.get('weekly_bottom_confirmed', []):
            weekly_bottom_div_dates.add(div.date.strftime('%Y-%m-%d'))

        # 检测波峰波谷
        try:
            peaks = peak_detector.detect_peaks_dynamic_atr(start_date, end_date)
            troughs = peak_detector.detect_troughs_dynamic_atr(start_date, end_date)
        except Exception:
            peaks = []
            troughs = []

        peak_dates = set(p.date.strftime('%Y-%m-%d') for p in peaks)
        trough_dates = set(t.date.strftime('%Y-%m-%d') for t in troughs)

        # 计算SAR并检测SAR死叉（绿转红）
        sar_dead_cross_dates = set()
        try:
            sar_strategy = SARStrategy(acceleration=0.02, maximum=0.20)
            sar_strategy.prepare_data(high_series, low_series, close_series)
            # 检测SAR信号
            sar_signals = sar_strategy.detect_signals(start_date, end_date)
            # 收集绿转红（SAR死叉）日期
            for signal in sar_signals:
                if signal.signal_type == '绿转红':
                    sar_dead_cross_dates.add(signal.date.strftime('%Y-%m-%d'))
            logging.debug(f"检测到 {len(sar_dead_cross_dates)} 个SAR死叉信号")
        except Exception as e:
            logging.warning(f"SAR计算失败: {e}")

        # 遍历每一天计算指标
        dates = target_close.index
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')

            # 获取各项指标
            close_price = self._get_value_at_date(date, close_series)
            ma5_val = self._get_value_at_date(date, ma5)
            ma10_val = self._get_value_at_date(date, ma10)
            rsi_daily_val = self._get_value_at_date(date, rsi_daily)
            rsi_weekly_val = self._get_value_at_date(date, rsi_weekly)
            macd_val = self._get_value_at_date(date, macd_series)

            # 检测死叉：前一天MA5>=MA10，当天MA5<MA10
            ma5_ma10_dead_cross = 0
            if i > 0:
                prev_date = dates[i-1]
                ma5_prev = self._get_value_at_date(prev_date, ma5)
                ma10_prev = self._get_value_at_date(prev_date, ma10)
                if not np.isnan(ma5_prev) and not np.isnan(ma10_prev) and not np.isnan(ma5_val) and not np.isnan(ma10_val):
                    if ma5_prev >= ma10_prev and ma5_val < ma10_val:
                        ma5_ma10_dead_cross = 1

            # 背离和局部高低点标志
            is_daily_top_div = 1 if date_str in daily_top_div_dates else 0
            is_daily_bottom_div = 1 if date_str in daily_bottom_div_dates else 0
            is_weekly_top_div = 1 if date_str in weekly_top_div_dates else 0
            is_weekly_bottom_div = 1 if date_str in weekly_bottom_div_dates else 0
            is_sar_dead_cross = 1 if date_str in sar_dead_cross_dates else 0
            is_local_high = 1 if date_str in peak_dates else 0
            is_local_low = 1 if date_str in trough_dates else 0

            # 构建指标对象
            metrics = DailyMetrics(
                stock_id=0,  # 后续设置
                trade_date=date_str,
                close_price=round(close_price, 2) if not np.isnan(close_price) else 0.0,
                ma5=round(ma5_val, 2) if not np.isnan(ma5_val) else None,
                ma10=round(ma10_val, 2) if not np.isnan(ma10_val) else None,
                ma5_ma10_dead_cross=ma5_ma10_dead_cross,
                rsi_daily=round(rsi_daily_val, 2) if not np.isnan(rsi_daily_val) else None,
                rsi_weekly=round(rsi_weekly_val, 2) if not np.isnan(rsi_weekly_val) else None,
                macd=round(macd_val, 4) if not np.isnan(macd_val) else None,
                is_daily_top_divergence=is_daily_top_div,
                is_daily_bottom_divergence=is_daily_bottom_div,
                is_weekly_top_divergence=is_weekly_top_div,
                is_weekly_bottom_divergence=is_weekly_bottom_div,
                is_sar_dead_cross=is_sar_dead_cross,
                is_local_high=is_local_high,
                is_local_low=is_local_low,
                status=0  # 0-完成
            )

            metrics_list.append(metrics)

        return metrics_list

    def _get_value_at_date(self, date: pd.Timestamp, series: pd.Series) -> float:
        """获取指定日期的值"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = series[series.index.strftime('%Y-%m-%d') == date_str]
            if len(matching) > 0:
                return matching.iloc[0]
            return np.nan
        except Exception:
            return np.nan

    def update_trading_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = 'SHSE'
    ) -> int:
        """
        更新交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所代码

        Returns:
            int: 插入的交易日数量
        """
        if get_trading_dates is None:
            logging.error("掘金SDK未安装，无法获取交易日")
            return 0

        try:
            trading_dates = get_trading_dates(exchange, start_date, end_date)
            if trading_dates is None or len(trading_dates) == 0:
                logging.warning("获取交易日数据为空")
                return 0

            # 转换为字符串格式
            date_list = []
            for d in trading_dates:
                if hasattr(d, 'strftime'):
                    date_list.append(d.strftime('%Y-%m-%d'))
                else:
                    date_list.append(str(d)[:10])

            # 写入数据库
            count = self.db.insert_trading_dates(date_list)
            logging.info(f"交易日历更新成功: {count} 条")
            return count

        except Exception as e:
            logging.error(f"更新交易日历失败: {e}")
            return 0

    def batch_process(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, bool]:
        """
        批量处理多只股票

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict: 处理结果字典 {symbol: success}
        """
        results = {}

        # 先更新交易日历
        self.update_trading_calendar(start_date, end_date)

        for symbol in symbols:
            success = self.process_stock(symbol, start_date, end_date)
            results[symbol] = success

        # 统计结果
        success_count = sum(1 for v in results.values() if v)
        logging.info(f"批量处理完成: 成功 {success_count}/{len(symbols)}")

        return results


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


def _load_symbols_from_csv(csv_path: str) -> List[Tuple[str, str, str]]:
    """
    从CSV文件加载股票配置

    Args:
        csv_path: CSV文件路径

    Returns:
        List[Tuple]: [(symbol, start_date, end_date), ...]
    """
    symbols = []

    if not os.path.exists(csv_path):
        logging.error(f"CSV配置文件不存在: {csv_path}")
        return symbols

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 检查 write_to_db 字段，只有值为1才写入数据库
                write_to_db = row.get('write_to_db', '1').strip()
                if write_to_db != '1':
                    continue

                symbol = row['symbol'].strip()
                # 自动添加交易所前缀
                symbol_full = _add_exchange_prefix(symbol)

                start_date = row['start_date'].strip()
                end_date = row['end_date'].strip()

                symbols.append((symbol_full, start_date, end_date))
                logging.info(f"加载配置: {symbol_full} | 时间: {start_date} ~ {end_date}")
            except Exception as e:
                logging.error(f"解析CSV行失败: {row}, 错误: {e}")

    return symbols


if __name__ == "__main__":
    # ===== 数据库连接测试 =====
    db = MySQLConnector()
    if db.connect():
        # 创建表（如果不存在）
        db.create_tables()

    # ===== 数据处理配置 =====
    processor = DataProcessor(db)

    # 从stocks_process.csv读取股票配置
    csv_path = os.path.join(project_root, 'config', 'stocks_process.csv')
    symbols_config = _load_symbols_from_csv(csv_path)

    if len(symbols_config) == 0:
        logging.error("未加载到任何股票配置，请检查CSV文件")
        db.disconnect()
        sys.exit(1)

    logging.info(f"共加载 {len(symbols_config)} 只股票配置")

    # ===== 执行数据处理 =====
    # 先更新交易日历（取最大的时间范围）
    all_start_dates = [s[1] for s in symbols_config]
    all_end_dates = [s[2] for s in symbols_config]
    min_start = min(all_start_dates)
    max_end = max(all_end_dates)

    processor.update_trading_calendar(min_start, max_end)

    # 处理每只股票
    results = {}
    for symbol, start_date, end_date in symbols_config:
        success = processor.process_stock(symbol, start_date, end_date)
        results[symbol] = success

    # ===== 输出结果 =====
    logging.info("")
    logging.info("=" * 60)
    logging.info("处理结果汇总")
    logging.info("=" * 60)
    for symbol, success in results.items():
        status = "成功" if success else "失败"
        logging.info(f"{symbol}: {status}")

    success_count = sum(1 for v in results.values() if v)
    logging.info(f"处理成功: {success_count}/{len(results)}")
    logging.info("=" * 60)

    # 关闭数据库连接
    db.disconnect()