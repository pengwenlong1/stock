# -*- coding: utf-8 -*-
"""
数据库数据填充脚本 (fill_db_tables.py)

【功能说明】
根据建表SQL的表结构，使用tool中的工具类计算技术指标，
将股票信息和每日指标数据填入MySQL数据库。

【表结构对应】
1. stock_info: 股票基本信息（stock_id, stock_name, listing_date）
2. stock_daily_metrics: 股票每日技术指标数据
3. trading_calendar: 交易日历

【使用方法】
python fill_db_tables.py

作者：量化交易团队
创建日期：2024
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymysql
import yaml

# 添加项目路径（脚本在src/runner目录，需要向上两级到项目根目录）
# 使用 pathlib（推荐，跨平台且不易出错）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入工具类
from src.tool.ma_calculator import MACalculator
from src.tool.rsi_calculator import RSI
from src.tool.macd_calculator import MACDCalculator
from src.tool.peak_detector import PeakDetector
from src.tool.divergence_detector import DivergenceDetector
from src.tool.sar_strategy import SARStrategy

try:
    from gm.api import history, set_token, ADJUST_PREV, get_trading_dates, get_instruments
except ImportError:
    history = None
    set_token = None
    ADJUST_PREV = None
    get_trading_dates = None
    get_instruments = None
    logging.warning("掘金SDK未安装，请先安装: pip install gm")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class DBConfig:
    """数据库配置"""
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class StockInfo:
    """股票基本信息"""
    stock_id: str
    stock_name: str
    listing_date: str


@dataclass
class DailyMetrics:
    """每日指标数据"""
    stock_id: int  # 关联stock_info表的stock_id字段（但这里用股票代码）
    trade_date: str
    close_price: float
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma5_ma10_dead_cross: int = 0
    rsi_daily: Optional[float] = None
    rsi_weekly: Optional[float] = None
    macd: Optional[float] = None
    sar: Optional[float] = None
    is_daily_top_divergence: int = 0
    is_daily_bottom_divergence: int = 0
    is_weekly_top_divergence: int = 0
    is_weekly_bottom_divergence: int = 0
    is_sar_dead_cross: int = 0
    is_local_high: int = 0
    is_local_low: int = 0
    status: int = 0


class DatabaseFiller:
    """
    数据库数据填充器

    【功能说明】
    1. 从掘金API获取股票历史数据
    2. 使用tool工具类计算各项技术指标
    3. 将数据插入MySQL数据库的对应表中

    【使用方法】
    filler = DatabaseFiller()
    filler.connect_db()
    filler.fill_stock_info()
    filler.fill_trading_calendar()
    filler.fill_daily_metrics(stock_list)
    filler.close_db()
    """

    def __init__(self) -> None:
        """初始化数据库填充器"""
        self._db_config: Optional[DBConfig] = None
        self._gm_token: Optional[str] = None
        self._connection: Optional[pymysql.Connection] = None
        self._dingtalk_webhook: Optional[str] = None
        self._dingtalk_secret: Optional[str] = None

        # 加载配置
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        # 加载app_config.yaml（数据库配置）
        # 使用 pathlib 处理路径（跨平台且不易出错）
        # 注意：.env 目录名称前面有空格（ .env）
        app_config_path = project_root / " .env" / "app_config.yaml"

        if app_config_path.exists():
            with app_config_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

                # MySQL配置（db_meta_mysql）
                db_config = config.get('db_meta_mysql', {})
                self._db_config = DBConfig(
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 3306),
                    user=db_config.get('user', 'root'),
                    password=db_config.get('password', ''),
                    database=db_config.get('database', 'stock')
                )
                logger.info(f"数据库配置加载成功: {self._db_config.host}:{self._db_config.port}/{self._db_config.database}")
        else:
            logger.error(f"配置文件不存在: {app_config_path}")
            raise FileNotFoundError("缺少app_config.yaml配置文件")

        # 加载settings.json（掘金Token）
        settings_path = project_root / "config" / "settings.json"
        if settings_path.exists():
            with settings_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
                self._gm_token = config.get('gm_token')
                self._dingtalk_webhook = config.get('dingtalk_webhook')
                self._dingtalk_secret = config.get('dingtalk_secret')

                if self._gm_token:
                    set_token(self._gm_token)
                    logger.info("掘金Token设置成功")
                else:
                    logger.warning("未找到掘金Token配置")
        else:
            logger.warning(f"掘金配置文件不存在: {settings_path}")

    def connect_db(self) -> bool:
        """
        连接数据库

        Returns:
            bool: 是否成功连接
        """
        try:
            self._connection = pymysql.connect(
                host=self._db_config.host,
                port=self._db_config.port,
                user=self._db_config.user,
                password=self._db_config.password,
                database=self._db_config.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info(f"数据库连接成功: {self._db_config.host}:{self._db_config.port}")
            return True
        except pymysql.Error as e:
            logger.error(f"数据库连接失败: {e}")
            return False

    def close_db(self) -> None:
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            logger.info("数据库连接已关闭")

    def _execute_sql(self, sql: str, params: tuple = None) -> int:
        """
        执行SQL语句

        Args:
            sql: SQL语句
            params: 参数

        Returns:
            int: 影响行数
        """
        if not self._connection:
            raise RuntimeError("数据库未连接")

        try:
            with self._connection.cursor() as cursor:
                cursor.execute(sql, params)
                self._connection.commit()
                return cursor.rowcount
        except pymysql.Error as e:
            self._connection.rollback()
            logger.error(f"SQL执行失败: {e}\nSQL: {sql}\nParams: {params}")
            raise

    def _execute_batch(self, sql: str, params_list: List[tuple]) -> int:
        """
        批量执行SQL语句

        Args:
            sql: SQL语句
            params_list: 参数列表

        Returns:
            int: 影响行数
        """
        if not self._connection:
            raise RuntimeError("数据库未连接")

        if not params_list:
            return 0

        try:
            with self._connection.cursor() as cursor:
                cursor.executemany(sql, params_list)
                self._connection.commit()
                return cursor.rowcount
        except pymysql.Error as e:
            self._connection.rollback()
            logger.error(f"批量SQL执行失败: {e}\nSQL: {sql}")
            raise

    def fill_stock_info(self, stock_list: List[Dict]) -> int:
        """
        填充stock_info表（股票基本信息）

        Args:
            stock_list: 股票列表，格式: [{'ticker': '159928', 'name': '消费ETF'}, ...]

        Returns:
            int: 插入行数
        """
        logger.info("=" * 60)
        logger.info("填充stock_info表（股票基本信息）")
        logger.info("=" * 60)

        if not stock_list:
            logger.warning("股票列表为空")
            return 0

        # 获取股票上市日期（从掘金API）
        sql = """
            INSERT INTO stock_info (stock_id, stock_name, listing_date)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                stock_name = VALUES(stock_name),
                listing_date = VALUES(listing_date)
        """

        params_list = []
        for stock in stock_list:
            ticker = stock.get('ticker', '')

            # 通过API获取股票信息（名称和上市日期）
            stock_info = self._get_stock_info_from_api(ticker)
            name = stock_info.get('name', '')
            listing_date = stock_info.get('listing_date', '2000-01-01')

            params_list.append((ticker, name, listing_date))
            logger.info(f"股票信息: {ticker} | {name} | 上市日期: {listing_date}")

        rowcount = self._execute_batch(sql, params_list)
        logger.info(f"stock_info表填充完成，插入/更新 {rowcount} 条记录")
        return rowcount

    def _get_stock_info_from_api(self, ticker: str) -> Dict:
        """
        通过掘金API获取股票信息（名称和上市日期）

        Args:
            ticker: 股票代码

        Returns:
            Dict: 包含 name 和 listing_date 的字典
        """
        result = {'name': '', 'listing_date': '2000-01-01'}

        if get_instruments is None:
            return result

        try:
            # 添加交易所前缀
            full_symbol = self._add_exchange_prefix(ticker)

            instruments = get_instruments(symbols=full_symbol)
            if instruments and len(instruments) > 0:
                instrument = instruments[0]
                # 获取名称（掘金API不同版本属性名不同）
                name = getattr(instrument, 'sec_name', None) or \
                       getattr(instrument, 'symbol_name', None) or \
                       getattr(instrument, 'display_name', None) or ''
                result['name'] = name
                # 获取上市日期
                listing_date = getattr(instrument, 'listed_date', None) or \
                               getattr(instrument, 'list_date', None)
                if listing_date:
                    result['listing_date'] = listing_date.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"获取股票信息失败: {ticker}, {e}")

        return result

    def _add_exchange_prefix(self, code: str) -> str:
        """
        添加交易所前缀

        Args:
            code: 股票代码

        Returns:
            str: 完整代码（如SHSE.510880）
        """
        code = code.strip()
        if code.startswith('SHSE.') or code.startswith('SZSE.'):
            return code

        # ETF和上证股票
        if code.startswith('51') or code.startswith('58') or code.startswith('6') or code.startswith('5'):
            return f'SHSE.{code}'
        # 深证股票
        elif code.startswith('0') or code.startswith('3') or code.startswith('1'):
            return f'SZSE.{code}'
        # 科创板
        elif code.startswith('688'):
            return f'SHSE.{code}'
        else:
            return f'SZSE.{code}'

    def fill_trading_calendar(self, start_date: str, end_date: str,
                               exchange: str = 'SHSE') -> int:
        """
        填充trading_calendar表（交易日历）

        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            exchange: 交易所代码

        Returns:
            int: 插入行数
        """
        logger.info("=" * 60)
        logger.info("填充trading_calendar表（交易日历）")
        logger.info("=" * 60)

        if get_trading_dates is None:
            logger.warning("掘金SDK未安装，无法获取交易日历")
            return 0

        try:
            trading_dates = get_trading_dates(
                exchange=exchange,
                start_date=start_date,
                end_date=end_date
            )

            if not trading_dates:
                logger.warning("未获取到交易日数据")
                return 0

            sql = """
                INSERT INTO trading_calendar (trade_date)
                VALUES (%s)
                ON DUPLICATE KEY UPDATE trade_date = VALUES(trade_date)
            """

            params_list = [(d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d),)
                           for d in trading_dates]

            rowcount = self._execute_batch(sql, params_list)
            logger.info(f"trading_calendar表填充完成，插入 {rowcount} 个交易日")
            return rowcount

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return 0

    def fill_daily_metrics(self, stock_list: List[Dict]) -> Dict[str, int]:
        """
        填充stock_daily_metrics表（每日技术指标数据）

        【计算指标】
        1. MA5/MA10和死叉检测
        2. RSI（日线和周线）
        3. MACD
        4. SAR和SAR死叉
        5. 局部高低点检测
        6. 背离检测（日线/周线顶背离、底背离）

        Args:
            stock_list: 股票列表，每只股票包含 ticker, name, start_date, end_date

        Returns:
            Dict[str, int]: 各股票插入行数
        """
        logger.info("=" * 60)
        logger.info("填充stock_daily_metrics表（每日技术指标数据）")
        logger.info("=" * 60)

        results = {}

        for stock in stock_list:
            ticker = stock.get('ticker', '')
            name = stock.get('name', '')
            start_date = stock.get('start_date', '2015-01-01')
            end_date = stock.get('end_date', '2026-04-10')

            logger.info(f"\n处理股票: {ticker} | {name} | 时间段: {start_date} ~ {end_date}")

            try:
                rowcount = self._process_single_stock(ticker, name, start_date, end_date)
                results[ticker] = rowcount
                logger.info(f"股票 {ticker} 处理完成，插入 {rowcount} 条记录")
            except Exception as e:
                logger.error(f"股票 {ticker} 处理失败: {e}")
                results[ticker] = 0

        total = sum(results.values())
        logger.info(f"\n所有股票处理完成，总计插入 {total} 条记录")
        return results

    def _get_stock_db_id(self, ticker: str) -> int:
        """
        获取股票的数据库ID（用于stock_daily_metrics表）

        由于股票代码都是数字，直接转换为整数作为ID

        Args:
            ticker: 股票代码

        Returns:
            int: 股票ID
        """
        # 股票代码转为整数（去掉可能的交易所前缀）
        clean_ticker = ticker.replace('SHSE.', '').replace('SZSE.', '')
        try:
            return int(clean_ticker)
        except ValueError:
            logger.warning(f"无法将股票代码转换为整数: {ticker}")
            # 使用默认值
            return 0

    def _process_single_stock(self, ticker: str, name: str,
                               start_date: str,
                               end_date: str) -> int:
        """
        处理单个股票的指标计算和数据插入

        Args:
            ticker: 股票代码
            name: 股票名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            int: 插入行数
        """
        if history is None:
            logger.warning("掘金SDK未安装，无法获取历史数据")
            return 0

        # 添加交易所前缀
        full_symbol = self._add_exchange_prefix(ticker)

        # 预热数据起始日期：使用传入的start_date，但往前推一年用于计算指标
        # 如果start_date早于股票上市日期，则从上市日期开始
        warmup_start = start_date  # 直接使用配置的开始日期

        logger.info(f"获取历史数据: {full_symbol}, {warmup_start} ~ {end_date}")

        # 获取日线数据（前复权）
        daily_data = history(
            symbol=full_symbol,
            frequency='1d',
            start_time=warmup_start + ' 09:00:00',
            end_time=end_date + ' 15:30:00',
            fields='eob,open,high,low,close,volume',
            df=True,
            adjust=ADJUST_PREV
        )

        if daily_data is None or daily_data.empty:
            logger.warning(f"获取数据失败或数据为空: {full_symbol}")
            return 0

        logger.info(f"获取到 {len(daily_data)} 条日线数据")

        # 处理数据
        daily_data['eob'] = pd.to_datetime(daily_data['eob'])

        # 创建价格序列
        close_series = pd.Series(
            daily_data['close'].values,
            index=daily_data['eob'],
            name='Close'
        )
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

        # ========== 1. 计算MA5/MA10和死叉检测 ==========
        logger.info("计算MA5/MA10...")
        ma_calculator = MACalculator(fast_period=5, slow_period=10, ma_type='SMA')
        ma5_series, ma10_series = ma_calculator.calculate(close_series)
        ma_dead_cross_series = ma_calculator.detect_all_cross_downs()

        # ========== 2. 计算RSI（日线和周线） ==========
        logger.info("计算RSI...")
        rsi_daily_calculator = RSI(period=6, freq='daily')
        rsi_daily_series = rsi_daily_calculator.calculate(close_series)

        rsi_weekly_calculator = RSI(period=6, freq='weekly')
        rsi_weekly_series = rsi_weekly_calculator.calculate(close_series)

        # ========== 3. 计算MACD ==========
        logger.info("计算MACD...")
        macd_calculator = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
        macd_calculator.prepare_data(close_series)
        dif_series = macd_calculator._dif_series  # DIF值（快线）
        dea_series = macd_calculator._dea_series  # DEA值（慢线）
        macd_series = macd_calculator._macd_series  # MACD柱值

        # ========== 4. 计算SAR和SAR死叉 ==========
        logger.info("计算SAR...")
        sar_strategy = SARStrategy(acceleration=0.02, maximum=0.20)
        try:
            sar_strategy.prepare_data(high_series, low_series, close_series)
            sar_series = sar_strategy._sar_series

            # 检测SAR死叉（绿转红）
            sar_signals = sar_strategy.detect_signals(warmup_start, end_date)
            # 使用日期字符串格式存储，避免时区匹配问题
            sar_dead_cross_dates = set()
            # 创建转折点字典：记录每个转折点的SAR值
            sar_turning_points = {}  # {date_str: sar_value}
            for sig in sar_signals:
                date_str = sig.date.strftime('%Y-%m-%d')
                if sig.signal_type == '绿转红':
                    sar_dead_cross_dates.add(date_str)
                # 记录所有转折点（金叉和死叉）的SAR值
                sar_turning_points[date_str] = sig.sar_value
        except ImportError:
            logger.warning("TA-Lib未安装，跳过SAR计算")
            sar_series = pd.Series([np.nan] * len(close_series), index=close_series.index)
            sar_dead_cross_dates = set()
            sar_turning_points = {}

        # ========== 5. 计算局部高低点 ==========
        logger.info("计算局部高低点...")
        peak_detector = PeakDetector(atr_period=14, prominence_factor=1.0, min_distance=20)
        peak_detector.prepare_data(high_series, low_series, close_series)

        try:
            peaks = peak_detector.detect_peaks(start_date, end_date)
            troughs = peak_detector.detect_troughs(start_date, end_date)

            # 使用日期字符串格式存储，避免时区匹配问题
            local_high_dates = set(p.date.strftime('%Y-%m-%d') for p in peaks)
            local_low_dates = set(t.date.strftime('%Y-%m-%d') for t in troughs)
        except Exception as e:
            logger.warning(f"波峰波谷检测失败: {e}")
            local_high_dates = set()
            local_low_dates = set()

        # ========== 6. 计算背离（日线/周线） ==========
        logger.info("计算背离...")
        divergence_detector = DivergenceDetector()

        try:
            divergence_detector.prepare_data(full_symbol, warmup_start, end_date)
            divergences = divergence_detector.detect_all_divergences()

            # 日线顶背离生效日期（使用confirmation_date，使用日期字符串格式）
            daily_top_div_dates = set(d.confirmation_date.strftime('%Y-%m-%d') for d in divergences.get('daily_top_confirmed', []) if d.confirmation_date is not None)
            # 周线顶背离生效日期
            weekly_top_div_dates = set(d.confirmation_date.strftime('%Y-%m-%d') for d in divergences.get('weekly_top_confirmed', []) if d.confirmation_date is not None)

            # 底背离（目前暂不计算，设为空）
            daily_bottom_div_dates = set()
            weekly_bottom_div_dates = set()
        except Exception as e:
            logger.warning(f"背离检测失败: {e}")
            daily_top_div_dates = set()
            weekly_top_div_dates = set()
            daily_bottom_div_dates = set()
            weekly_bottom_div_dates = set()

        # ========== 篩选目标时间段数据 ==========
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')

        # 篩选数据
        filtered_dates = close_series.index[
            (close_series.index >= start_ts) &
            (close_series.index <= end_ts)
        ]

        if len(filtered_dates) == 0:
            logger.warning(f"筛选后数据为空: {start_date} ~ {end_date}")
            return 0

        # ========== 构建插入数据 ==========
        sql = """
            INSERT INTO stock_daily_metrics
            (stock_id, trade_date, close_price, low_price, ma5, ma10, ma5_ma10_dead_cross,
             rsi_daily, rsi_weekly, macd, dif, dea, sar,
             is_daily_top_divergence, is_daily_bottom_divergergence,
             is_weekly_top_divergence, is_weekly_bottom_divergergence,
             is_sar_dead_cross, is_local_high, is_local_low, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                close_price = VALUES(close_price),
                low_price = VALUES(low_price),
                ma5 = VALUES(ma5),
                ma10 = VALUES(ma10),
                ma5_ma10_dead_cross = VALUES(ma5_ma10_dead_cross),
                rsi_daily = VALUES(rsi_daily),
                rsi_weekly = VALUES(rsi_weekly),
                macd = VALUES(macd),
                dif = VALUES(dif),
                dea = VALUES(dea),
                sar = VALUES(sar),
                is_daily_top_divergence = VALUES(is_daily_top_divergence),
                is_daily_bottom_divergergence = VALUES(is_daily_bottom_divergergence),
                is_weekly_top_divergence = VALUES(is_weekly_top_divergence),
                is_weekly_bottom_divergergence = VALUES(is_weekly_bottom_divergergence),
                is_sar_dead_cross = VALUES(is_sar_dead_cross),
                is_local_high = VALUES(is_local_high),
                is_local_low = VALUES(is_local_low),
                status = VALUES(status)
        """

        params_list = []
        for date in filtered_dates:
            date_str = date.strftime('%Y-%m-%d')

            # 获取各项指标值
            close_price = close_series.loc[date]
            low_price = low_series.loc[date] if date in low_series.index else None  # 最低价

            # MA值
            ma5_val = ma5_series.loc[date] if date in ma5_series.index else None
            ma10_val = ma10_series.loc[date] if date in ma10_series.index else None

            # MA死叉
            ma_dead_cross = 1 if date in ma_dead_cross_series.index and ma_dead_cross_series.loc[date] else 0

            # RSI值
            rsi_daily_val = rsi_daily_series.loc[date] if date in rsi_daily_series.index else None
            rsi_weekly_val = rsi_weekly_series.loc[date] if date in rsi_weekly_series.index else None

            # MACD值（MACD柱、DIF、DEA）
            macd_val = macd_series.loc[date] if date in macd_series.index else None
            dif_val = dif_series.loc[date] if date in dif_series.index else None
            dea_val = dea_series.loc[date] if date in dea_series.index else None

            # SAR值
            sar_val = sar_series.loc[date] if date in sar_series.index else None

            # SAR死叉
            sar_dead_cross = 1 if date_str in sar_dead_cross_dates else 0

            # 局部高低点
            is_local_high = 1 if date_str in local_high_dates else 0
            is_local_low = 1 if date_str in local_low_dates else 0

            # 背离
            is_daily_top_div = 1 if date_str in daily_top_div_dates else 0
            is_daily_bottom_div = 1 if date_str in daily_bottom_div_dates else 0
            is_weekly_top_div = 1 if date_str in weekly_top_div_dates else 0
            is_weekly_bottom_div = 1 if date_str in weekly_bottom_div_dates else 0

            # 处理NaN值
            low_price = None if pd.isna(low_price) else round(float(low_price), 2)
            ma5_val = None if pd.isna(ma5_val) else round(float(ma5_val), 2)
            ma10_val = None if pd.isna(ma10_val) else round(float(ma10_val), 2)
            rsi_daily_val = None if pd.isna(rsi_daily_val) else round(float(rsi_daily_val), 2)
            rsi_weekly_val = None if pd.isna(rsi_weekly_val) else round(float(rsi_weekly_val), 2)
            macd_val = None if pd.isna(macd_val) else round(float(macd_val), 4)
            dif_val = None if pd.isna(dif_val) else round(float(dif_val), 4)
            dea_val = None if pd.isna(dea_val) else round(float(dea_val), 4)
            sar_val = None if pd.isna(sar_val) else round(float(sar_val), 4)

            params = (
                self._get_stock_db_id(ticker),  # stock_id (BIGINT)
                date_str,  # trade_date
                round(float(close_price), 2),  # close_price
                low_price,  # low_price
                ma5_val,  # ma5
                ma10_val,  # ma10
                ma_dead_cross,  # ma5_ma10_dead_cross
                rsi_daily_val,  # rsi_daily
                rsi_weekly_val,  # rsi_weekly
                macd_val,  # macd
                dif_val,  # dif
                dea_val,  # dea
                sar_val,  # sar
                is_daily_top_div,  # is_daily_top_divergence
                is_daily_bottom_div,  # is_daily_bottom_divergergence
                is_weekly_top_div,  # is_weekly_top_divergence
                is_weekly_bottom_div,  # is_weekly_bottom_divergergence
                sar_dead_cross,  # is_sar_dead_cross
                is_local_high,  # is_local_high
                is_local_low,  # is_local_low
                0  # status: 0-完成
            )

            params_list.append(params)

        # 执行批量插入
        rowcount = self._execute_batch(sql, params_list)
        logger.info(f"股票 {ticker} 指标计算完成，准备插入 {len(params_list)} 条记录，实际插入 {rowcount} 条")

        return rowcount

    def fill_all(self, stock_list: List[Dict]) -> Dict:
        """
        填充所有表数据

        Args:
            stock_list: 股票列表，每只股票包含 ticker, name, start_date, end_date

        Returns:
            Dict: 各表填充结果
        """
        logger.info("\n" + "=" * 60)
        logger.info("开始填充所有数据表")
        logger.info("=" * 60)

        # 计算所有股票的时间范围（用于填充交易日历）
        all_start_dates = [stock.get('start_date', '2015-01-01') for stock in stock_list]
        all_end_dates = [stock.get('end_date', '2026-04-10') for stock in stock_list]
        min_start_date = min(all_start_dates)
        max_end_date = max(all_end_dates)
        logger.info(f"交易日历范围: {min_start_date} ~ {max_end_date}")

        results = {
            'stock_info': 0,
            'trading_calendar': 0,
            'daily_metrics': {}
        }

        # 1. 填充stock_info表
        results['stock_info'] = self.fill_stock_info(stock_list)

        # 2. 填充trading_calendar表
        results['trading_calendar'] = self.fill_trading_calendar(min_start_date, max_end_date)

        # 3. 填充stock_daily_metrics表
        results['daily_metrics'] = self.fill_daily_metrics(stock_list)

        logger.info("\n" + "=" * 60)
        logger.info("所有数据表填充完成")
        logger.info(f"  stock_info: {results['stock_info']} 条")
        logger.info(f"  trading_calendar: {results['trading_calendar']} 条")
        logger.info(f"  daily_metrics总计: {sum(results['daily_metrics'].values())} 条")
        logger.info("=" * 60)

        return results


def load_stock_list() -> List[Dict]:
    """
    加载股票列表

    从config/stocks_backtest.csv加载股票列表（仅加载active字段为1的股票）
    包含每只股票的时间参数，名称通过API获取。
    如果CSV的name列为空，会自动通过API获取并更新CSV文件。

    Returns:
        List[Dict]: 股票列表，包含 ticker, name, start_date, end_date
    """
    stock_csv_path = project_root / "config" / "stocks_backtest.csv"

    if not stock_csv_path.exists():
        logger.warning(f"股票列表文件不存在: {stock_csv_path}")
        return []

    # 使用 dtype 参数确保股票代码作为字符串读取，不丢失前导0
    df = pd.read_csv(str(stock_csv_path), dtype={'symbol': str})

    # 确保 name 列存在
    if 'name' not in df.columns:
        df['name'] = ''

    # 仅筛选 active 字段为 1 的股票
    if 'active' in df.columns:
        df = df[df['active'] == 1]
        logger.info(f"筛选 active=1 的股票，共 {len(df)} 只")
    else:
        logger.warning("CSV文件中未找到 active 列，将加载所有股票")

    # 检查是否有缺失的 name，如果有则先设置token再获取
    needs_update = False
    has_missing_name = any(pd.isna(df.loc[idx, 'name']) or str(df.loc[idx, 'name']).strip() == '' for idx in df.index)

    if has_missing_name and get_instruments is not None:
        # 先设置掘金token（在调用API之前）
        _init_gm_token()

        # 检查并填充缺失的 name
        for idx in df.index:
            name_val = df.loc[idx, 'name']
            if pd.isna(name_val) or str(name_val).strip() == '':
                ticker = str(df.loc[idx, 'symbol'])
                # 通过API获取股票名称
                stock_name = _get_stock_name_from_api(ticker)
                if stock_name:
                    df.loc[idx, 'name'] = stock_name
                    logger.info(f"自动填充股票名称: {ticker} -> {stock_name}")
                    needs_update = True

    # 如果有更新，保存回CSV文件（保存完整文件，包括active=0的股票）
    if needs_update:
        # 重新读取完整文件以保留所有股票
        full_df = pd.read_csv(str(stock_csv_path), dtype={'symbol': str})
        if 'name' not in full_df.columns:
            full_df['name'] = ''
        # 更新缺失的name
        for idx in full_df.index:
            name_val = full_df.loc[idx, 'name']
            if pd.isna(name_val) or str(name_val).strip() == '':
                ticker = str(full_df.loc[idx, 'symbol'])
                stock_name = _get_stock_name_from_api(ticker)
                if stock_name:
                    full_df.loc[idx, 'name'] = stock_name
                    logger.info(f"更新CSV: {ticker} -> {stock_name}")
        # 保存完整文件
        full_df.to_csv(str(stock_csv_path), index=False, encoding='utf-8')
        logger.info(f"已更新股票名称到CSV文件: {stock_csv_path}")

    stock_list = []
    for row in df.to_dict('records'):
        ticker = str(row['symbol'])
        name = str(row['name']) if not pd.isna(row['name']) else ''
        stock_list.append({
            'ticker': ticker,
            'name': name,
            'start_date': str(row['start_date']),
            'end_date': str(row['end_date'])
        })

    return stock_list


def _init_gm_token() -> bool:
    """
    初始化掘金Token（用于调用API前设置token）

    Returns:
        bool: 是否成功设置token
    """
    if set_token is None:
        logger.warning("掘金SDK未安装")
        return False

    settings_path = project_root / "config" / "settings.json"
    if settings_path.exists():
        with settings_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
            gm_token = config.get('gm_token')
            if gm_token:
                set_token(gm_token)
                logger.info("掘金Token设置成功（load_stock_list）")
                return True
            else:
                logger.warning("settings.json中未找到gm_token")
    else:
        logger.warning(f"配置文件不存在: {settings_path}")

    return False


def _get_stock_name_from_api(ticker: str) -> Optional[str]:
    """
    通过掘金API获取股票名称

    Args:
        ticker: 股票代码（不带交易所前缀）

    Returns:
        Optional[str]: 股票名称，如果获取失败返回 None
    """
    if get_instruments is None:
        logger.warning(f"掘金SDK未安装，无法获取 {ticker} 名称")
        return None

    try:
        # 添加交易所前缀
        if ticker.startswith('SHSE.') or ticker.startswith('SZSE.'):
            full_symbol = ticker
        elif ticker.startswith('51') or ticker.startswith('58') or ticker.startswith('6') or ticker.startswith('5'):
            full_symbol = f'SHSE.{ticker}'
        elif ticker.startswith('0') or ticker.startswith('3') or ticker.startswith('1'):
            full_symbol = f'SZSE.{ticker}'
        elif ticker.startswith('688'):
            full_symbol = f'SHSE.{ticker}'
        else:
            full_symbol = f'SZSE.{ticker}'

        instruments = get_instruments(symbols=full_symbol)
        if instruments and len(instruments) > 0:
            instrument = instruments[0]
            # 获取名称（掘金API不同版本属性名不同）
            name = getattr(instrument, 'sec_name', None) or \
                   getattr(instrument, 'symbol_name', None) or \
                   getattr(instrument, 'display_name', None) or None
            return name
    except Exception as e:
        logger.warning(f"获取股票名称失败: {ticker} - {e}")

    return None


def run():
    """主运行函数"""
    # 配置日志文件
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"fill_db_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("数据库数据填充脚本启动")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)

    # 加载股票列表
    stock_list = load_stock_list()

    if not stock_list:
        logger.error("股票列表为空，退出程序")
        return

    logger.info(f"加载到 {len(stock_list)} 只股票")

    # 创建数据库填充器
    filler = DatabaseFiller()

    # 连接数据库
    if not filler.connect_db():
        logger.error("数据库连接失败，退出程序")
        return

    try:
        # 填充所有数据
        results = filler.fill_all(stock_list)

        # 输出统计结果
        logger.info("\n" + "=" * 60)
        logger.info("数据填充统计")
        logger.info("=" * 60)
        for ticker, count in results['daily_metrics'].items():
            logger.info(f"  {ticker}: {count} 条")

    except Exception as e:
        logger.error(f"数据填充过程出错: {e}")
        raise

    finally:
        # 关闭数据库连接
        filler.close_db()

    logger.info("=" * 60)
    logger.info("脚本执行完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()