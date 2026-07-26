# -*- coding: utf-8 -*-
"""
数据库回测执行器模块 (db_backtest_runner.py)

【功能说明】
从MySQL数据库读取已计算的技术指标数据，执行历史数据回测验证。
调用 strategy.py 中的交易策略类进行信号检测。

与 backtest_runner.py 的区别：
- backtest_runner.py: 从掘金API获取数据，实时计算指标
- db_backtest_runner.py: 从MySQL数据库读取预计算的指标数据

【表结构】
从 stock_daily_metrics 表读取：
- close_price: 收盘价
- rsi_daily: 日线RSI
- rsi_weekly: 周线RSI
- macd: MACD值
- sar: SAR值
- is_sar_dead_cross: SAR死叉标志
- is_daily_top_divergence: 日线顶背离标志
- is_weekly_top_divergence: 周线顶背离标志
- is_local_high/is_local_low: 局部高低点标志

【使用方法】
python src/runner/db_backtest_runner.py

作者：量化交易团队
创建日期：2024
"""
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pymysql
import yaml

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入策略类
from src.tool.strategy import TradingStrategy, StrategyState, SellSignal, BuySignal, SellFlag, COOLDOWN_DAYS
from src.tool.divergence_detector import DivergenceDetector, DivergenceSignal  # 导入背离检测器（用于指数）
from src.tool.rsi_calculator import RSI  # 导入RSI计算类（用于月线RSI）

# ==================== 数据结构 ====================

@dataclass
class DBConfig:
    """数据库配置"""
    host: str
    port: int
    user: str
    password: str
    database: str


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
    monthly_rsi: float = float('nan')
    dividend_yield_rate: float = float('nan')


# ==================== 数据库回测执行器 ====================

class DBBacktestRunner:
    """
    数据库回测执行器

    【功能说明】
    从MySQL数据库读取预计算的指标数据，调用 TradingStrategy 进行信号检测，
    执行模拟交易，生成回测报告。

    【职责分离】
    - 数据获取：从 stock_daily_metrics 表读取指标数据
    - 策略逻辑：由 strategy.py 的 TradingStrategy 类处理
    - 回测执行：由本类处理（交易执行、报告生成）
    """

    def __init__(self,
                 stock_id: str,
                 start_date: str,
                 end_date: str,
                 stock_name: str = '',
                 judge_buy_ids: List[int] = [1],
                 judge_t_ids: List[int] = [2],
                 judge_sell_ids: List[int] = [1],
                 initial_capital: float = 100000.0,
                 output_dir: Optional[Path] = None,
                 fully_invested: int = 1) -> None:
        """
        初始化数据库回测执行器

        Args:
            stock_id: 股票代码（如 512480）
            start_date: 回测开始日期（YYYY-MM-DD）
            end_date: 回测结束日期（YYYY-MM-DD）
            stock_name: 股票名称
            judge_buy_ids: 新资金买入策略ID列表
            judge_t_ids: 做T买回策略ID列表
            judge_sell_ids: 卖出策略ID列表
            initial_capital: 初始资金
            output_dir: 输出目录路径
            fully_invested: 是否默认全仓持有（1=全仓持有，0=空仓等待买入信号）
        """
        self.stock_id = stock_id
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date
        self.judge_buy_ids = judge_buy_ids
        self.judge_t_ids = judge_t_ids
        self.judge_sell_ids = judge_sell_ids
        self.initial_capital = initial_capital
        self.fully_invested = fully_invested

        # 加载数据库配置
        self._db_config = self._load_db_config()

        # 数据库连接
        self._connection: Optional[pymysql.Connection] = None

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
            logs_base = project_root / "logs"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = logs_base / f'db_backtest_{timestamp}'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self._df: Optional[pd.DataFrame] = None
        self._metrics_df: Optional[pd.DataFrame] = None
        self._monthly_rsi_series: Optional[pd.Series] = None  # 月线RSI序列（用于buy_id=15-26）

        # 指数数据（用于大盘参照）
        self._index_metrics_df: Optional[pd.DataFrame] = None  # 创业板指 (399006)
        self._sh_index_metrics_df: Optional[pd.DataFrame] = None  # 上证指数 (000001)
        self._kc_index_metrics_df: Optional[pd.DataFrame] = None  # 科创综指 (000680)

        # 指数背离检测器（用于 sell_id=4,5,6）
        self._sh_index_divergence_detector: Optional[DivergenceDetector] = None
        self._sh_index_daily_divergences: List[DivergenceSignal] = []
        self._sh_index_weekly_divergences: List[DivergenceSignal] = []
        self._cyb_index_divergence_detector: Optional[DivergenceDetector] = None
        self._cyb_index_daily_divergences: List[DivergenceSignal] = []
        self._cyb_index_weekly_divergences: List[DivergenceSignal] = []
        self._kc_index_divergence_detector: Optional[DivergenceDetector] = None
        self._kc_index_daily_divergences: List[DivergenceSignal] = []
        self._kc_index_weekly_divergences: List[DivergenceSignal] = []

        # 新指数背离检测器（用于 sell_id=13,14,15）
        self._sh50_index_divergence_detector: Optional[DivergenceDetector] = None
        self._sh50_index_daily_divergences: List[DivergenceSignal] = []
        self._sh50_index_weekly_divergences: List[DivergenceSignal] = []
        self._cyb50_index_divergence_detector: Optional[DivergenceDetector] = None
        self._cyb50_index_daily_divergences: List[DivergenceSignal] = []
        self._cyb50_index_weekly_divergences: List[DivergenceSignal] = []
        self._kc50_index_divergence_detector: Optional[DivergenceDetector] = None
        self._kc50_index_daily_divergences: List[DivergenceSignal] = []
        self._kc50_index_weekly_divergences: List[DivergenceSignal] = []

        # 已处理的指数背离信号ID集合（避免重复触发）
        self._processed_sh_index_divergences: Set[str] = set()
        self._processed_cyb_index_divergences: Set[str] = set()
        self._processed_kc_index_divergences: Set[str] = set()
        self._processed_sh50_index_divergences: Set[str] = set()
        self._processed_cyb50_index_divergences: Set[str] = set()
        self._processed_kc50_index_divergences: Set[str] = set()

        # 持仓状态（使用金额追踪）
        self._shares: float = 0.0         # 持有股票数量
        self._cash: float = initial_capital  # 剩余新资金（金额）
        self._sold_cash: float = 0.0      # 卖出后待买回的资金（金额）

        # 交易记录
        self._trade_records: List[TradeRecord] = []

        # 日志配置
        self._setup_logging()

    def _load_db_config(self) -> DBConfig:
        """加载数据库配置"""
        app_config_path = project_root / " .env" / "app_config.yaml"

        if app_config_path.exists():
            with app_config_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

                db_config = config.get('db_meta_mysql', {})
                return DBConfig(
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 3306),
                    user=db_config.get('user', 'root'),
                    password=db_config.get('password', ''),
                    database=db_config.get('database', 'stock')
                )
        else:
            raise FileNotFoundError("缺少数据库配置文件")

    def _connect_db(self) -> bool:
        """连接数据库"""
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
            self.logger.info(f"数据库连接成功: {self._db_config.host}:{self._db_config.port}")
            return True
        except pymysql.Error as e:
            self.logger.error(f"数据库连接失败: {e}")
            return False

    def _close_db(self) -> None:
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            self.logger.info("数据库连接已关闭")

    def _setup_logging(self) -> None:
        """配置日志（使用命名logger避免混淆）"""
        log_file = self.output_dir / f'db_backtest_{self.stock_id}.log'

        # 创建该股票专用的logger（不使用根logger）
        self.logger = logging.getLogger(f'db_backtest_{self.stock_id}')

        # 清除该logger已有的handlers
        self.logger.handlers.clear()

        # 设置日志级别
        self.logger.setLevel(logging.INFO)

        # 添加文件handler
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(file_handler)

        # 添加控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(console_handler)

        # 不传播到根logger
        self.logger.propagate = False

        self.log_file = str(log_file)

        # 为divergence_detector设置专用日志文件
        from src.tool.divergence_detector import setup_log_file, clear_log_file
        clear_log_file()  # 先清除旧的文件handler
        setup_log_file(str(log_file))

    def fetch_metrics_from_db(self) -> bool:
        """
        从数据库获取指标数据

        Returns:
            bool: 数据获取是否成功
        """
        if not self._connect_db():
            return False

        try:
            # 查询 stock_daily_metrics 表
            # 注意：stock_id 在数据库中是 BIGINT 类型，存储的是股票代码的数字值
            stock_id_int = int(self.stock_id)

            sql = """
                SELECT
                    trade_date,
                    close_price,
                    low_price,
                    ma5,
                    ma10,
                    ma5_ma10_dead_cross,
                    rsi_daily,
                    rsi_weekly,
                    macd,
                    dif,
                    dea,
                    sar,
                    is_sar_dead_cross,
                    is_daily_top_divergence,
                    is_daily_bottom_divergergence,
                    is_weekly_top_divergence,
                    is_weekly_bottom_divergergence,
                    is_local_high,
                    is_local_low,
                    status
                FROM stock_daily_metrics
                WHERE stock_id = %s
                  AND trade_date >= %s
                  AND trade_date <= %s
                ORDER BY trade_date ASC
            """

            with self._connection.cursor() as cursor:
                cursor.execute(sql, (stock_id_int, self.start_date, self.end_date))
                rows = cursor.fetchall()

            if not rows:
                self.logger.warning(f"数据库中未找到股票 {self.stock_id} 的数据")
                return False

            self.logger.info(f"从数据库获取到 {len(rows)} 条指标数据")

            # 转换为 DataFrame
            df = pd.DataFrame(rows)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date').sort_index()

            self._metrics_df = df

            # 输出数据概况
            self.logger.info(f"数据时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
            self.logger.info(f"有效RSI数据: 日线={df['rsi_daily'].notna().sum()}, 周线={df['rsi_weekly'].notna().sum()}")
            self.logger.info(f"SAR死叉信号数: {df['is_sar_dead_cross'].sum()}")
            self.logger.info(f"日线顶背离数: {df['is_daily_top_divergence'].sum()}")
            self.logger.info(f"周线顶背离数: {df['is_weekly_top_divergence'].sum()}")

            # 计算月线RSI(6)（用于buy_id=15-26）
            close_series = pd.Series(df['close_price'].values, index=df.index, name='Close')
            rsi_monthly = RSI(period=6, freq='monthly')
            monthly_rsi_series = rsi_monthly.calculate(close_series)
            df['rsi_monthly'] = monthly_rsi_series
            self._monthly_rsi_series = monthly_rsi_series
            self.logger.info(f"月线RSI数据: 有效={monthly_rsi_series.notna().sum()}")

            return True

        except Exception as e:
            self.logger.error(f"数据库查询失败: {e}")
            return False
        finally:
            self._close_db()

    def fetch_stock_info_from_db(self) -> str:
        """
        从数据库获取股票名称

        Returns:
            str: 股票名称
        """
        if not self._connect_db():
            return "未知"

        try:
            sql = "SELECT stock_name FROM stock_info WHERE stock_id = %s"

            with self._connection.cursor() as cursor:
                cursor.execute(sql, (self.stock_id,))
                row = cursor.fetchone()

            if row:
                self.stock_name = row['stock_name']
                return self.stock_name
            return "未知"

        except Exception as e:
            self.logger.warning(f"获取股票名称失败: {e}")
            return "未知"
        finally:
            self._close_db()

    def fetch_index_metrics_from_db(self) -> bool:
        """
        从数据库获取指数的RSI数据（创业板指399006和上证指数000001）

        Returns:
            bool: 数据获取是否成功
        """
        if not self._connect_db():
            return False

        try:
            # 创业板指 (399006) - stock_id为字符串格式
            sql = """
                SELECT trade_date, rsi_daily, macd, sar, close_price
                FROM stock_index_daily_metrics
                WHERE stock_id = '399006'
                  AND trade_date >= %s
                  AND trade_date <= %s
                ORDER BY trade_date ASC
            """
            with self._connection.cursor() as cursor:
                cursor.execute(sql, (self.start_date, self.end_date))
                rows = cursor.fetchall()

            if rows:
                df = pd.DataFrame(rows)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                self._index_metrics_df = df
                self.logger.info(f"获取创业板指(399006) RSI数据: {len(rows)} 条")
            else:
                self.logger.warning("数据库中未找到创业板指(399006)的RSI数据")

            # 上证指数 (000001) - stock_id为字符串格式
            sql = """
                SELECT trade_date, rsi_daily, macd, sar, close_price
                FROM stock_index_daily_metrics
                WHERE stock_id = '000001'
                  AND trade_date >= %s
                  AND trade_date <= %s
                ORDER BY trade_date ASC
            """
            with self._connection.cursor() as cursor:
                cursor.execute(sql, (self.start_date, self.end_date))
                rows = cursor.fetchall()

            if rows:
                df = pd.DataFrame(rows)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                self._sh_index_metrics_df = df
                self.logger.info(f"获取上证指数(000001) RSI数据: {len(rows)} 条")
            else:
                self.logger.warning("数据库中未找到上证指数(000001)的RSI数据")

            # 科创综指 (000680) - stock_id为字符串格式
            sql = """
                SELECT trade_date, rsi_daily, macd, sar, close_price
                FROM stock_index_daily_metrics
                WHERE stock_id = '000680'
                  AND trade_date >= %s
                  AND trade_date <= %s
                ORDER BY trade_date ASC
            """
            with self._connection.cursor() as cursor:
                cursor.execute(sql, (self.start_date, self.end_date))
                rows = cursor.fetchall()

            if rows:
                df = pd.DataFrame(rows)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date').sort_index()
                self._kc_index_metrics_df = df
                self.logger.info(f"获取科创综指(000680) RSI数据: {len(rows)} 条")
            else:
                self.logger.warning("数据库中未找到科创综指(000680)的RSI数据")

            return True

        except Exception as e:
            self.logger.error(f"获取指数数据失败: {e}")
            return False
        finally:
            self._close_db()

    def prepare_data(self) -> bool:
        """
        准备数据

        Returns:
            bool: 数据准备是否成功
        """
        self.logger.info("=" * 60)
        self.logger.info("准备数据（从MySQL数据库读取）")
        self.logger.info("=" * 60)

        # 获取股票名称
        if not self.stock_name:
            self.stock_name = self.fetch_stock_info_from_db()

        # 获取指标数据
        if not self.fetch_metrics_from_db():
            self.logger.error("数据获取失败")
            return False

        # 获取指数ETF数据（用于大盘参照）
        self.fetch_index_metrics_from_db()

        # 如果配置了 sell_id=4,5,6,10,11,12,13,14,15，初始化指数背离检测器
        if any(sid in [4, 5, 6, 10, 11, 12, 13, 14, 15] for sid in self.judge_sell_ids):
            self._init_index_divergence_detectors()

        return True

    def _init_index_divergence_detectors(self) -> None:
        """
        初始化指数背离检测器（用于 sell_id=4,5,6,10,11,12）

        检测上证指数、创业板指数、科创板指数的顶背离信号。
        """
        # 【关键】设置掘金Token（用于DivergenceDetector获取数据）
        try:
            from gm.api import set_token

            # 尝试多种路径查找settings.json
            possible_paths = [
                Path.cwd() / "config" / "settings.json",
                Path(__file__).resolve().parent.parent.parent / "config" / "settings.json",
                Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "config" / "settings.json"
            ]

            gm_token = None
            for settings_path in possible_paths:
                if settings_path.exists():
                    self.logger.info(f"找到settings.json: {settings_path}")
                    with settings_path.open('r', encoding='utf-8') as f:
                        config = json.load(f)
                        gm_token = config.get('gm_token')
                        if gm_token:
                            break

            if gm_token:
                set_token(gm_token)
                self.logger.info("掘金Token设置成功")
            else:
                self.logger.error("未找到有效的掘金Token，请在config/settings.json中配置gm_token")
                return  # 如果没有Token，直接返回，不初始化指数检测器
        except Exception as e:
            self.logger.error(f"设置掘金Token失败: {e}")
            return

        # 计算预热起始日期（提前365天）
        warmup_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        # 检测上证指数背离（用于 sell_id=4,10）
        if any(sid in [4, 10] for sid in self.judge_sell_ids):
            self.logger.info("初始化上证指数背离检测器...")
            try:
                self._sh_index_divergence_detector = DivergenceDetector()
                df = self._sh_index_divergence_detector.prepare_data(
                    symbol='SHSE.000001',
                    start_date=warmup_start,
                    end_date=self.end_date
                )
                if df is not None and not df.empty:
                    sh_divergences = self._sh_index_divergence_detector.detect_all_divergences()
                    self._sh_index_daily_divergences = sh_divergences.get('daily_top_confirmed', [])
                    self._sh_index_weekly_divergences = sh_divergences.get('weekly_top_confirmed', [])
                    self.logger.info(f"上证指数日线顶背离生效: {len(self._sh_index_daily_divergences)} 个")
                    self.logger.info(f"上证指数周线顶背离生效: {len(self._sh_index_weekly_divergences)} 个")
                else:
                    self.logger.warning("上证指数数据获取失败，跳过背离检测")
                    self._sh_index_daily_divergences = []
                    self._sh_index_weekly_divergences = []
            except Exception as e:
                self.logger.warning(f"上证指数背离检测器初始化失败: {e}")
                self._sh_index_daily_divergences = []
                self._sh_index_weekly_divergences = []

        # 检测创业板指数背离（用于 sell_id=5,11）
        if any(sid in [5, 11] for sid in self.judge_sell_ids):
            self.logger.info("初始化创业板指数背离检测器...")
            try:
                self._cyb_index_divergence_detector = DivergenceDetector()
                df = self._cyb_index_divergence_detector.prepare_data(
                    symbol='SZSE.399006',
                    start_date=warmup_start,
                    end_date=self.end_date
                )
                if df is not None and not df.empty:
                    cyb_divergences = self._cyb_index_divergence_detector.detect_all_divergences()
                    self._cyb_index_daily_divergences = cyb_divergences.get('daily_top_confirmed', [])
                    self._cyb_index_weekly_divergences = cyb_divergences.get('weekly_top_confirmed', [])
                    self.logger.info(f"创业板指数日线顶背离生效: {len(self._cyb_index_daily_divergences)} 个")
                    self.logger.info(f"创业板指数周线顶背离生效: {len(self._cyb_index_weekly_divergences)} 个")
                else:
                    self.logger.warning("创业板指数数据获取失败，跳过背离检测")
                    self._cyb_index_daily_divergences = []
                    self._cyb_index_weekly_divergences = []
            except Exception as e:
                self.logger.warning(f"创业板指数背离检测器初始化失败: {e}")
                self._cyb_index_daily_divergences = []
                self._cyb_index_weekly_divergences = []

        # 检测科创板指数背离（用于 sell_id=6,12）
        if any(sid in [6, 12] for sid in self.judge_sell_ids):
            self.logger.info("初始化科创板指数背离检测器...")
            try:
                self._kc_index_divergence_detector = DivergenceDetector()
                df = self._kc_index_divergence_detector.prepare_data(
                    symbol='SHSE.000680',
                    start_date=warmup_start,
                    end_date=self.end_date
                )
                if df is not None and not df.empty:
                    kc_divergences = self._kc_index_divergence_detector.detect_all_divergences()
                    self._kc_index_daily_divergences = kc_divergences.get('daily_top_confirmed', [])
                    self._kc_index_weekly_divergences = kc_divergences.get('weekly_top_confirmed', [])
                    self.logger.info(f"科创板指数日线顶背离生效: {len(self._kc_index_daily_divergences)} 个")
                    self.logger.info(f"科创板指数周线顶背离生效: {len(self._kc_index_weekly_divergences)} 个")
                else:
                    self.logger.warning("科创板指数数据获取失败，跳过背离检测")
                    self._kc_index_daily_divergences = []
                    self._kc_index_weekly_divergences = []
            except Exception as e:
                self.logger.warning(f"科创板指数背离检测器初始化失败: {e}")
                self._kc_index_daily_divergences = []
                self._kc_index_weekly_divergences = []

        # 检测新指数背离（用于 sell_id=13,14,15）
        # 上证50指数背离检测器
        if any(sid in [13] for sid in self.judge_sell_ids):
            self.logger.info("初始化上证50指数背离检测器...")
            try:
                self._sh50_index_divergence_detector = DivergenceDetector()
                df = self._sh50_index_divergence_detector.prepare_data(
                    symbol='SHSE.000016',
                    start_date=warmup_start,
                    end_date=self.end_date
                )
                if df is not None and not df.empty:
                    sh50_divergences = self._sh50_index_divergence_detector.detect_all_divergences()
                    self._sh50_index_daily_divergences = sh50_divergences.get('daily_top_confirmed', [])
                    self._sh50_index_weekly_divergences = sh50_divergences.get('weekly_top_confirmed', [])
                    self.logger.info(f"上证50指数日线顶背离生效: {len(self._sh50_index_daily_divergences)} 个")
                    self.logger.info(f"上证50指数周线顶背离生效: {len(self._sh50_index_weekly_divergences)} 个")
                else:
                    self.logger.warning("上证50指数数据获取失败，跳过背离检测")
                    self._sh50_index_daily_divergences = []
                    self._sh50_index_weekly_divergences = []
            except Exception as e:
                self.logger.warning(f"上证50指数背离检测器初始化失败: {e}")
                self._sh50_index_daily_divergences = []
                self._sh50_index_weekly_divergences = []

        # 创业板50指数背离检测器
        if any(sid in [14] for sid in self.judge_sell_ids):
            self.logger.info("初始化创业板50指数背离检测器...")
            try:
                self._cyb50_index_divergence_detector = DivergenceDetector()
                df = self._cyb50_index_divergence_detector.prepare_data(
                    symbol='SZSE.399673',
                    start_date=warmup_start,
                    end_date=self.end_date
                )
                if df is not None and not df.empty:
                    cyb50_divergences = self._cyb50_index_divergence_detector.detect_all_divergences()
                    self._cyb50_index_daily_divergences = cyb50_divergences.get('daily_top_confirmed', [])
                    self._cyb50_index_weekly_divergences = cyb50_divergences.get('weekly_top_confirmed', [])
                    self.logger.info(f"创业板50指数日线顶背离生效: {len(self._cyb50_index_daily_divergences)} 个")
                    self.logger.info(f"创业板50指数周线顶背离生效: {len(self._cyb50_index_weekly_divergences)} 个")
                else:
                    self.logger.warning("创业板50指数数据获取失败，跳过背离检测")
                    self._cyb50_index_daily_divergences = []
                    self._cyb50_index_weekly_divergences = []
            except Exception as e:
                self.logger.warning(f"创业板50指数背离检测器初始化失败: {e}")
                self._cyb50_index_daily_divergences = []
                self._cyb50_index_weekly_divergences = []

        # 科创板50指数背离检测器
        if any(sid in [15] for sid in self.judge_sell_ids):
            self.logger.info("初始化科创板50指数背离检测器...")
            try:
                self._kc50_index_divergence_detector = DivergenceDetector()
                df = self._kc50_index_divergence_detector.prepare_data(
                    symbol='SHSE.000688',
                    start_date=warmup_start,
                    end_date=self.end_date
                )
                if df is not None and not df.empty:
                    kc50_divergences = self._kc50_index_divergence_detector.detect_all_divergences()
                    self._kc50_index_daily_divergences = kc50_divergences.get('daily_top_confirmed', [])
                    self._kc50_index_weekly_divergences = kc50_divergences.get('weekly_top_confirmed', [])
                    self.logger.info(f"科创板50指数日线顶背离生效: {len(self._kc50_index_daily_divergences)} 个")
                    self.logger.info(f"科创板50指数周线顶背离生效: {len(self._kc50_index_weekly_divergences)} 个")
                else:
                    self.logger.warning("科创板50指数数据获取失败，跳过背离检测")
                    self._kc50_index_daily_divergences = []
                    self._kc50_index_weekly_divergences = []
            except Exception as e:
                self.logger.warning(f"科创板50指数背离检测器初始化失败: {e}")
                self._kc50_index_daily_divergences = []
                self._kc50_index_weekly_divergences = []

    def _get_trading_dates(self) -> List[pd.Timestamp]:
        """获取回测时间段内的交易日列表"""
        if self._metrics_df is None:
            return []
        return self._metrics_df.index.tolist()

    def _get_metrics_at_date(self, date: pd.Timestamp) -> Dict:
        """
        获取指定日期的所有指标值

        Args:
            date: 日期

        Returns:
            Dict: 指标值字典
        """
        if self._metrics_df is None:
            return {}

        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = self._metrics_df[self._metrics_df.index.strftime('%Y-%m-%d') == date_str]

            if len(matching) > 0:
                row = matching.iloc[0]
                # 使用 float() 确保数值类型，避免 None 导致 np.isnan() 报错
                def safe_float(val):
                    """安全转换为float，None或NaN返回np.nan"""
                    if val is None or pd.isna(val):
                        return np.nan
                    return float(val)

                # 安全获取字段（处理可能不存在的新字段）
                def safe_get_float(row, col_name):
                    """安全获取字段值，不存在则返回np.nan"""
                    try:
                        if col_name in row.index:
                            return safe_float(row[col_name])
                        return np.nan
                    except Exception:
                        return np.nan

                return {
                    'close_price': safe_float(row['close_price']),
                    'low_price': safe_get_float(row, 'low_price'),  # 最低价（用于MACD跌破死叉判断）
                    'daily_rsi': safe_float(row['rsi_daily']),
                    'weekly_rsi': safe_float(row['rsi_weekly']),
                    'macd': safe_float(row['macd']),
                    'dif': safe_get_float(row, 'dif'),  # DIF值（用于MACD死叉判断）
                    'dea': safe_get_float(row, 'dea'),  # DEA值（用于MACD死叉判断）
                    'sar': safe_float(row['sar']),
                    'is_sar_dead_cross': int(row['is_sar_dead_cross']) if pd.notna(row['is_sar_dead_cross']) else 0,
                    'is_daily_top_divergence': int(row['is_daily_top_divergence']) if pd.notna(row['is_daily_top_divergence']) else 0,
                    'is_daily_bottom_divergence': int(row['is_daily_bottom_divergergence']) if pd.notna(row['is_daily_bottom_divergergence']) else 0,
                    'is_weekly_top_divergence': int(row['is_weekly_top_divergence']) if pd.notna(row['is_weekly_top_divergence']) else 0,
                    'is_weekly_bottom_divergence': int(row['is_weekly_bottom_divergergence']) if pd.notna(row['is_weekly_bottom_divergergence']) else 0,
                    'is_local_high': int(row['is_local_high']) if pd.notna(row['is_local_high']) else 0,
                    'is_local_low': int(row['is_local_low']) if pd.notna(row['is_local_low']) else 0,
                    'ma5_ma10_dead_cross': int(row['ma5_ma10_dead_cross']) if pd.notna(row['ma5_ma10_dead_cross']) else 0
                }
            return {}
        except Exception:
            return {}

    def _get_dividend_yield_rate(self, date: pd.Timestamp) -> float:
        """
        从数据库获取个股在指定日期的股息率

        Args:
            date: 查询日期

        Returns:
            float: 股息率(%)，查询失败返回 nan
        """
        if not self._connection:
            if not self._connect_db():
                return np.nan

        try:
            sql = """
                SELECT dividend_yield
                FROM stock_dividend
                WHERE stock_id = %s
                  AND dividend_date <= %s
                ORDER BY dividend_year DESC
                LIMIT 1
            """
            with self._connection.cursor() as cursor:
                cursor.execute(sql, (self.stock_id, date.strftime('%Y-%m-%d')))
                row = cursor.fetchone()

            if row and row.get('dividend_yield') is not None:
                return float(row['dividend_yield'])
        except Exception:
            pass
        return np.nan

    def _get_index_divergence_info(
        self,
        date: pd.Timestamp,
        index_type: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        获取指定日期的指数背离信息（跳过已处理的信号）

        Args:
            date: 当前日期
            index_type: 指数类型 ('sh', 'cyb', 'kc', 'sh50', 'cyb50', 'kc50')
            timeframe: 时间级别 ('daily' 或 'weekly')

        Returns:
            背离信息字典或 None
        """
        # 根据指数类型选择背离信号列表
        if index_type == 'sh':
            divergences = self._sh_index_daily_divergences if timeframe == 'daily' else self._sh_index_weekly_divergences
            processed_set = self._processed_sh_index_divergences
        elif index_type == 'cyb':
            divergences = self._cyb_index_daily_divergences if timeframe == 'daily' else self._cyb_index_weekly_divergences
            processed_set = self._processed_cyb_index_divergences
        elif index_type == 'kc':
            divergences = self._kc_index_daily_divergences if timeframe == 'daily' else self._kc_index_weekly_divergences
            processed_set = self._processed_kc_index_divergences
        elif index_type == 'sh50':
            divergences = self._sh50_index_daily_divergences if timeframe == 'daily' else self._sh50_index_weekly_divergences
            processed_set = self._processed_sh50_index_divergences if hasattr(self, '_processed_sh50_index_divergences') else set()
        elif index_type == 'cyb50':
            divergences = self._cyb50_index_daily_divergences if timeframe == 'daily' else self._cyb50_index_weekly_divergences
            processed_set = self._processed_cyb50_index_divergences if hasattr(self, '_processed_cyb50_index_divergences') else set()
        elif index_type == 'kc50':
            divergences = self._kc50_index_daily_divergences if timeframe == 'daily' else self._kc50_index_weekly_divergences
            processed_set = self._processed_kc50_index_divergences if hasattr(self, '_processed_kc50_index_divergences') else set()
        else:
            return None

        for div in divergences:
            if div.confirmation_date is not None:
                if div.confirmation_date.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'):
                    # 生成唯一ID（使用指数类型+形成日期）
                    div_id = f"{index_type}_{div.date.strftime('%Y-%m-%d')}"

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

    def _get_weekly_rsi_near_divergence(
        self,
        div_date: pd.Timestamp,
        lookback_days: int = 5
    ) -> Tuple[float, pd.Timestamp]:
        """
        获取顶背离形成日期附近（±5个交易日）的最高周线RSI值

        Args:
            div_date: 顶背离形成日期
            lookback_days: 向前/向后查找的天数（默认5天）

        Returns:
            (最高周线RSI值, 对应日期)
        """
        # 检查数据是否已加载
        if self._df is None or len(self._df) == 0:
            return 0.0, div_date

        # 定义查找范围
        start_date = div_date - pd.Timedelta(days=lookback_days)
        end_date = div_date + pd.Timedelta(days=lookback_days)

        # 截取范围内的数据
        range_df = self._df[(self._df.index >= start_date) & (self._df.index <= end_date)]

        if len(range_df) == 0:
            return 0.0, div_date

        # 找到周线RSI最高的日期
        if 'rsi_weekly' not in range_df.columns:
            return 0.0, div_date

        max_rsi_idx = range_df['rsi_weekly'].idxmax()
        max_rsi_value = range_df['rsi_weekly'].max()

        return max_rsi_value, max_rsi_idx

    def _detect_index_macd_cross_down(self, date: pd.Timestamp, index_type: str) -> bool:
        """
        检测指数MACD是否发生死叉（用于 sell_id=7,8,9,10,11,12,13,14,15）

        Args:
            date: 当前日期
            index_type: 指数类型 ('sh', 'cyb', 'kc', 'sh50', 'cyb50', 'kc50')

        Returns:
            bool: 是否发生指数MACD死叉

        【逻辑说明】
        指数MACD死叉：DIF（快线）从上方穿过DEA（慢线）下方
        使用 divergence detector 提供的 MACD 数据进行检测
        """
        # 根据指数类型选择MACD数据
        if index_type == 'sh':
            detector = self._sh_index_divergence_detector
        elif index_type == 'cyb':
            detector = self._cyb_index_divergence_detector
        elif index_type == 'kc':
            detector = self._kc_index_divergence_detector
        elif index_type == 'sh50':
            detector = self._sh50_index_divergence_detector
        elif index_type == 'cyb50':
            detector = self._cyb50_index_divergence_detector
        elif index_type == 'kc50':
            detector = self._kc50_index_divergence_detector
        else:
            return False

        if detector is None:
            return False

        # 使用 divergence detector 的 MACD Calculator
        if detector._macd_calculator is None:
            return False

        dif_series = detector._macd_calculator._dif_series
        dea_series = detector._macd_calculator._dea_series

        if dif_series is None or dea_series is None:
            return False

        # 获取当前日期的DIF/DEA值
        try:
            date_str = date.strftime('%Y-%m-%d')
            # 找到当前日期在数据中的位置
            matching_dates = dif_series.index.strftime('%Y-%m-%d') == date_str
            if matching_dates.sum() == 0:
                return False

            current_idx = matching_dates.argmax()
            if current_idx < 1:
                return False

            current_dif = dif_series.iloc[current_idx]
            current_dea = dea_series.iloc[current_idx]
            prev_dif = dif_series.iloc[current_idx - 1]
            prev_dea = dea_series.iloc[current_idx - 1]

            if np.isnan(current_dif) or np.isnan(current_dea) or np.isnan(prev_dif) or np.isnan(prev_dea):
                return False

            # MACD死叉：DIF从上方穿过DEA下方
            if prev_dif >= prev_dea and current_dif < current_dea:
                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] {index_type}指数MACD死叉: DIF={current_dif:.4f}<DEA={current_dea:.4f}")
                return True

        except Exception as e:
            self.logger.warning(f"检测{index_type}指数MACD死叉失败: {e}")

        return False

    def _is_index_macd_dead_cross_state(self, date: pd.Timestamp, index_type: str) -> bool:
        """
        检查指数MACD是否处于死叉状态（DIF < DEA，即MACD柱 < 0）

        与 _detect_index_macd_cross_down 不同，此方法检查是否已经是死叉状态，
        而不是检测当天是否发生新的死叉。

        Args:
            date: 当前日期
            index_type: 指数类型 ('sh', 'cyb', 'kc', 'sh50', 'cyb50', 'kc50')

        Returns:
            bool: 是否处于MACD死叉状态（MACD柱 < 0）
        """
        # 根据指数类型选择指数数据DataFrame
        if index_type == 'sh':
            index_df = self._sh_index_metrics_df
        elif index_type == 'cyb':
            index_df = self._index_metrics_df  # 创业板指
        elif index_type == 'kc':
            index_df = self._kc_index_metrics_df
        elif index_type == 'sh50':
            index_df = getattr(self, '_sh50_index_metrics_df', None)
        elif index_type == 'cyb50':
            index_df = getattr(self, '_cyb50_index_metrics_df', None)
        elif index_type == 'kc50':
            index_df = getattr(self, '_kc50_index_metrics_df', None)
        else:
            return False

        if index_df is None or index_df.empty:
            return False

        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = index_df.index.strftime('%Y-%m-%d') == date_str
            if matching.sum() == 0:
                return False

            row = index_df[matching].iloc[0]
            macd_value = row.get('macd', None)

            if macd_value is None:
                return False

            # 处理可能的NaN值
            if pd.isna(macd_value):
                return False

            macd_value = float(macd_value)

            # MACD柱 < 0 表示 DIF < DEA，即处于死叉状态
            if macd_value < 0:
                self.logger.info(f"[{date_str}] {index_type}指数MACD已处于死叉状态: MACD柱={macd_value:.4f} < 0")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"检查{index_type}指数MACD状态失败: {e}")
            return False

    def _is_index_sar_dead_cross_state(self, date: pd.Timestamp, index_type: str) -> bool:
        """
        检查指数SAR是否处于死叉状态（SAR在收盘价上方，即绿转红状态）

        Args:
            date: 当前日期
            index_type: 指数类型 ('sh', 'cyb', 'kc', 'sh50', 'cyb50', 'kc50')

        Returns:
            bool: 是否处于SAR死叉状态（SAR > Close）
        """
        # 根据指数类型选择指数数据DataFrame
        if index_type == 'sh':
            index_df = self._sh_index_metrics_df
        elif index_type == 'cyb':
            index_df = self._index_metrics_df  # 创业板指
        elif index_type == 'kc':
            index_df = self._kc_index_metrics_df
        elif index_type == 'sh50':
            index_df = getattr(self, '_sh50_index_metrics_df', None)
        elif index_type == 'cyb50':
            index_df = getattr(self, '_cyb50_index_metrics_df', None)
        elif index_type == 'kc50':
            index_df = getattr(self, '_kc50_index_metrics_df', None)
        else:
            return False

        if index_df is None or index_df.empty:
            return False

        try:
            date_str = date.strftime('%Y-%m-%d')
            matching = index_df.index.strftime('%Y-%m-%d') == date_str
            if matching.sum() == 0:
                return False

            row = index_df[matching].iloc[0]
            close_price = row.get('close_price', None)
            sar_value = row.get('sar', None)

            if close_price is None or sar_value is None:
                return False

            # 处理可能的NaN值
            if pd.isna(close_price) or pd.isna(sar_value):
                return False

            close_price = float(close_price)
            sar_value = float(sar_value)

            # SAR死叉状态：SAR > Close（SAR在价格上方）
            if sar_value > close_price:
                self.logger.info(f"[{date_str}] {index_type}指数SAR已处于死叉状态: SAR={sar_value:.4f} > Close={close_price:.2f}")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"检查{index_type}指数SAR状态失败: {e}")
            return False

    def _execute_sell(self, date: pd.Timestamp, signal: SellSignal) -> None:
        """执行卖出操作"""
        metrics = self._get_metrics_at_date(date)
        price = metrics.get('close_price', np.nan)

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
        self._sold_cash += sell_amount

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
        """执行假卖出操作（无持仓时重置状态）"""
        self.strategy.reset_after_sell(self.state, signal.flag, date, is_fake_sell=True)
        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 假卖出（无持仓）: {signal.reason}")

    def _execute_buy(self, date: pd.Timestamp, signal: BuySignal) -> None:
        """执行买入操作"""
        metrics = self._get_metrics_at_date(date)
        price = metrics.get('close_price', np.nan)

        if np.isnan(price):
            return

        # 计算买入金额
        if signal.is_new_cash:
            buy_amount = self._cash + self._sold_cash
            self._cash = 0.0
            self._sold_cash = 0.0
        else:
            buy_amount = self._sold_cash
            self._sold_cash = 0.0

        if buy_amount <= 0:
            return

        # 计算买入股票数量
        buy_shares = buy_amount / price

        # 更新持仓
        self._shares += buy_shares

        # 获取股息率
        dividend_yield_rate = self._get_dividend_yield_rate(date)

        # 记录交易
        trade = TradeRecord(
            date=date,
            action='buy',
            amount=buy_shares,
            price=price,
            reason=signal.reason,
            daily_rsi=signal.daily_rsi,
            weekly_rsi=signal.weekly_rsi,
            monthly_rsi=signal.monthly_rsi,
            dividend_yield_rate=dividend_yield_rate
        )
        self._trade_records.append(trade)

        # 日志输出
        buy_type = "新资金+买回" if signal.is_new_cash else "做T买回"
        monthly_rsi_str = f"{signal.monthly_rsi:.2f}" if not np.isnan(signal.monthly_rsi) else "N/A"
        div_yield_str = f"{dividend_yield_rate:.2f}%" if not np.isnan(dividend_yield_rate) else "N/A"
        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 买入({buy_type}): {int(buy_shares)}股 @ {price:.3f} | "
                        f"金额:{buy_amount:.2f} | 日RSI:{signal.daily_rsi:.2f} | 周RSI:{signal.weekly_rsi:.2f} | "
                        f"月RSI:{monthly_rsi_str} | 股息率:{div_yield_str} | {signal.reason}")

    def run_backtest(self) -> Dict:
        """
        执行回测

        Returns:
            回测结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("开始回测（数据库数据）")
        self.logger.info("=" * 60)
        self.logger.info(f"股票: {self.stock_id} ({self.stock_name})")
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

        # 如果 fully_invested=1，在第一天全仓买入（基准策略：默认全仓持有）
        if self.fully_invested == 1 and len(trading_dates) > 0:
            first_date = trading_dates[0]
            first_metrics = self._get_metrics_at_date(first_date)
            first_price = first_metrics.get('close_price', np.nan)
            if not np.isnan(first_price) and self._cash > 0:
                # 全仓买入
                buy_shares = self._cash / first_price
                self._shares = buy_shares
                self._cash = 0.0
                self.logger.info(f"[{first_date.strftime('%Y-%m-%d')}] 初始全仓买入: {int(buy_shares)}股 @ {first_price:.3f} | 金额:{self.initial_capital:.2f} | (基准策略)")

        # 逐日遍历
        for date in trading_dates:
            # 获取当日指标值
            metrics = self._get_metrics_at_date(date)

            # 安全获取RSI值，确保是 np.nan 或 float 类型
            daily_rsi = metrics.get('daily_rsi')
            weekly_rsi = metrics.get('weekly_rsi')

            # 如果值为 None，转换为 np.nan（避免 np.isnan(None) 报错）
            if daily_rsi is None:
                daily_rsi = np.nan
            if weekly_rsi is None:
                weekly_rsi = np.nan

            # 从数据库获取的卖出信号标志
            # 使用 SAR 死叉（is_sar_dead_cross）作为 sell_id=1 的触发条件
            sar_cross_down = metrics.get('is_sar_dead_cross', 0) == 1

            # 判断 MACD 死叉（DIF下穿DEA）作为 sell_id=2 的触发条件
            # 不需要额外条件，只要MACD死叉就触发
            macd_cross_down = False
            macd_dead_cross_state = False  # 个股MACD是否处于死叉状态（用于sell_id=10-15）
            current_dif = metrics.get('dif', np.nan)
            current_dea = metrics.get('dea', np.nan)

            if not np.isnan(current_dif) and not np.isnan(current_dea):
                # 检测个股MACD是否处于死叉状态（DIF < DEA）
                if current_dif < current_dea:
                    macd_dead_cross_state = True

                try:
                    date_idx = trading_dates.index(date)
                    if date_idx > 0:
                        prev_date = trading_dates[date_idx - 1]
                        prev_metrics = self._get_metrics_at_date(prev_date)
                        prev_dif = prev_metrics.get('dif', np.nan)
                        prev_dea = prev_metrics.get('dea', np.nan)
                        # 死叉：DIF从上方穿过DEA下方
                        if not np.isnan(prev_dif) and not np.isnan(prev_dea):
                            if prev_dif >= prev_dea and current_dif < current_dea:
                                macd_cross_down = True
                except Exception:
                    pass

            # 根据 judge_sell_ids 选择触发条件
            # sell_id=1: SAR死叉跌破触发
            # sell_id=2: MACD跌破死叉触发
            sell_id = self.judge_sell_ids[0] if self.judge_sell_ids else 1

            # 更新RSI警戒级别
            self.strategy.update_rsi_flag(weekly_rsi, self.state, date)

            # 设置背离状态（从数据库读取）
            daily_top_div = metrics.get('is_daily_top_divergence', 0) == 1
            weekly_top_div = metrics.get('is_weekly_top_divergence', 0) == 1

            # 【周线顶背离形成】设置标志，等待死叉触发生效后卖出
            # 【完整规则】根据用户需求实现三种场景：
            # (1) 若个股处于死叉，则直接卖出
            # (2) 个股未处于死叉，但指数处于死叉，则等待个股或指数再次重新死叉时卖出
            # (3) 若个股未处于死叉，指数也未处于死叉，则等待个股或指数出现死叉时卖出
            if weekly_top_div and self.state.weekly_divergence_flag == 0:
                self.state.weekly_divergence_flag = 1
                divergence_info = {
                    'date': date,
                    'prev_high': metrics.get('close_price', 0.0),
                    'curr_high': metrics.get('close_price', 0.0)
                }
                self.state.weekly_divergence_info = divergence_info
                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 周线顶背离形成，等待死叉触发生效")

                # 【场景(1)】检查个股是否已处于MACD死叉状态 → 直接卖出
                stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                if stock_macd_dead and self._shares > 0:
                    sell_signal = SellSignal(
                        flag=SellFlag.CLEAR_ALL,
                        reason=f"清仓信号 (sell_id={sell_id}-个股周线顶背离): 周线顶背离形成于 {date.strftime('%Y-%m-%d')}, 个股MACD已处于死叉状态，建议清仓",
                        daily_rsi=daily_rsi,
                        weekly_rsi=weekly_rsi
                    )
                    self._execute_sell(date, sell_signal)
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态（DIF<DEA），直接触发卖出")
                    continue  # 卖出后当天不再处理其他信号

                # 【场景(2)和(3)】检查指数死叉状态（仅用于日志，不直接卖出）
                # 确定检查哪个指数
                index_type_for_sell = None
                index_name = ""
                if any(sid in [10, 11, 12] for sid in self.judge_sell_ids):
                    if 10 in self.judge_sell_ids:
                        index_type_for_sell = 'sh'
                        index_name = "上证指数"
                    elif 11 in self.judge_sell_ids:
                        index_type_for_sell = 'cyb'
                        index_name = "创业板指数"
                    elif 12 in self.judge_sell_ids:
                        index_type_for_sell = 'kc'
                        index_name = "科创板指数"
                elif any(sid in [13, 14, 15] for sid in self.judge_sell_ids):
                    if 13 in self.judge_sell_ids:
                        index_type_for_sell = 'sh50'
                        index_name = "上证50指数"
                    elif 14 in self.judge_sell_ids:
                        index_type_for_sell = 'cyb50'
                        index_name = "创业板50指数"
                    elif 15 in self.judge_sell_ids:
                        index_type_for_sell = 'kc50'
                        index_name = "科创板50指数"

                if index_type_for_sell:
                    if self._is_index_macd_dead_cross_state(date, index_type_for_sell):
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] {index_name}MACD已处于死叉状态，但需等待个股死叉或指数再次重新死叉时才卖出")
                    elif self._is_index_sar_dead_cross_state(date, index_type_for_sell):
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] {index_name}SAR已处于死叉状态，但需等待个股死叉或指数再次重新死叉时才卖出")

            # 【日线顶背离形成】设置标志，等待死叉触发生效后卖出
            # 【完整规则】根据用户需求实现三种场景：
            # (1) 若个股处于死叉，则直接卖出
            # (2) 个股未处于死叉，但指数处于死叉，则等待个股或指数再次重新死叉时卖出
            # (3) 若个股未处于死叉，指数也未处于死叉，则等待个股或指数出现死叉时卖出
            if daily_top_div and self.state.daily_divergence_flag == 0:
                self.state.daily_divergence_flag = 1
                divergence_info = {
                    'date': date,
                    'prev_high': metrics.get('close_price', 0.0),
                    'curr_high': metrics.get('close_price', 0.0)
                }
                self.state.daily_divergence_info = divergence_info
                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 日线顶背离形成，等待死叉触发生效")

                # 【场景(1)】检查个股是否已处于MACD死叉状态 → 直接卖出
                stock_macd_dead_daily = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                if stock_macd_dead_daily and self._shares > 0:
                    sell_signal = SellSignal(
                        flag=SellFlag.SELL_ONE_THIRD,  # 日线顶背离卖出1/3
                        reason=f"卖出信号 (sell_id={sell_id}-个股日线顶背离): 日线顶背离形成于 {date.strftime('%Y-%m-%d')}, 个股MACD已处于死叉状态，建议卖出1/3",
                        daily_rsi=daily_rsi,
                        weekly_rsi=weekly_rsi
                    )
                    self._execute_sell(date, sell_signal)
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态（DIF<DEA），直接触发卖出")
                    continue  # 卖出后当天不再处理其他信号

                # 【场景(2)和(3)】检查指数死叉状态（仅用于日志，不直接卖出）
                index_type_for_sell_daily = None
                index_name_daily = ""
                if any(sid in [4, 10] for sid in self.judge_sell_ids):
                    index_type_for_sell_daily = 'sh'
                    index_name_daily = "上证指数"
                elif any(sid in [5, 11] for sid in self.judge_sell_ids):
                    index_type_for_sell_daily = 'cyb'
                    index_name_daily = "创业板指数"
                elif any(sid in [6, 12] for sid in self.judge_sell_ids):
                    index_type_for_sell_daily = 'kc'
                    index_name_daily = "科创板指数"
                elif any(sid in [13, 14, 15] for sid in self.judge_sell_ids):
                    if 13 in self.judge_sell_ids:
                        index_type_for_sell_daily = 'sh50'
                        index_name_daily = "上证50指数"
                    elif 14 in self.judge_sell_ids:
                        index_type_for_sell_daily = 'cyb50'
                        index_name_daily = "创业板50指数"
                    elif 15 in self.judge_sell_ids:
                        index_type_for_sell_daily = 'kc50'
                        index_name_daily = "科创板50指数"

                if index_type_for_sell_daily:
                    if self._is_index_macd_dead_cross_state(date, index_type_for_sell_daily):
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] {index_name_daily}MACD已处于死叉状态，但需等待个股死叉或指数再次重新死叉时才卖出")
                    elif self._is_index_sar_dead_cross_state(date, index_type_for_sell_daily):
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] {index_name_daily}SAR已处于死叉状态，但需等待个股死叉或指数再次重新死叉时才卖出")

            # 【新增】设置指数背离状态并判断是否直接卖出
            # 【重要】根据具体的sell_id只检测对应指数的背离，而不是检测所有指数
            # sell_id=4,10: 上证指数
            # sell_id=5,11: 创业板指数
            # sell_id=6,12: 科创板指数
            # sell_id=13: 上证50指数
            # sell_id=14: 创业板50指数
            # sell_id=15: 科创板50指数

            # 【sell_id=4,10】上证指数背离
            if any(sid in [4, 10] for sid in self.judge_sell_ids):
                sh_daily_div_info = self._get_index_divergence_info(date, 'sh', 'daily')
                sh_weekly_div_info = self._get_index_divergence_info(date, 'sh', 'weekly')
                if sh_daily_div_info is not None or sh_weekly_div_info is not None:
                    self.strategy.set_sh_index_divergence(self.state, sh_daily_div_info, sh_weekly_div_info)

                    index_div_type = "日线顶背离" if sh_daily_div_info else "周线顶背离"
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 上证指数{index_div_type}生效")

                    # 【场景(1)】检查指数是否已处于死叉状态 + 个股周线RSI条件
                    # 根据buy_sell.md：指数顶背离需要个股周线RSI > 80 才能触发卖出
                    # 注意：检查的是顶背离形成日期附近（±5天）的最高周线RSI
                    index_macd_dead = self._is_index_macd_dead_cross_state(date, 'sh')
                    index_sar_dead = self._is_index_sar_dead_cross_state(date, 'sh')
                    if (index_macd_dead or index_sar_dead) and self._shares > 0:
                        # 获取背离形成日期附近（±5天）的最高周线RSI
                        div_date = date  # 背离确认日期
                        max_weekly_rsi, max_rsi_date = self._get_weekly_rsi_near_divergence(div_date, lookback_days=5)

                        # 定义死叉类型（用于日志输出）
                        dead_cross_type = "MACD" if index_macd_dead else "SAR"

                        # 检查个股周线RSI是否满足条件
                        if max_weekly_rsi >= 80:
                            # 根据指数背离类型确定卖出比例
                            if sh_weekly_div_info:
                                # 周线顶背离：RSI > 80 卖出1/2，RSI > 85 清仓
                                if max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.CLEAR_ALL
                                else:
                                    sell_flag = SellFlag.SELL_HALF
                            else:
                                # 日线顶背离：RSI > 80 卖出1/3，RSI > 85 卖出1/2，RSI > 90 清仓
                                if max_weekly_rsi >= 90:
                                    sell_flag = SellFlag.CLEAR_ALL
                                elif max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.SELL_HALF
                                else:
                                    sell_flag = SellFlag.SELL_ONE_THIRD

                            sell_signal = SellSignal(
                                flag=sell_flag,
                                reason=f"{sell_flag.name} (sell_id={sell_id}-上证指数{index_div_type}): 上证指数{index_div_type}形成于 {date.strftime('%Y-%m-%d')}, 上证指数{dead_cross_type}已处于死叉状态, 背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})>80，建议卖出",
                                daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi
                            )
                            self._execute_sell(date, sell_signal)
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 上证指数{index_div_type}+{dead_cross_type}死叉+背离附近周RSI={max_weekly_rsi:.2f}>80，触发卖出")
                            continue
                        else:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 上证指数{index_div_type}+{dead_cross_type}死叉，但背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})<80，不触发卖出")

                    # 【场景(2)和(3)】检查个股死叉状态（仅用于日志，不直接卖出）
                    stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                    if stock_macd_dead:
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态，但需等待上证指数死叉或个股再次重新死叉时才卖出")

            # 【sell_id=5,11】创业板指数背离
            if any(sid in [5, 11] for sid in self.judge_sell_ids):
                cyb_daily_div_info = self._get_index_divergence_info(date, 'cyb', 'daily')
                cyb_weekly_div_info = self._get_index_divergence_info(date, 'cyb', 'weekly')
                if cyb_daily_div_info is not None or cyb_weekly_div_info is not None:
                    self.strategy.set_cyb_index_divergence(self.state, cyb_daily_div_info, cyb_weekly_div_info)

                    index_div_type = "日线顶背离" if cyb_daily_div_info else "周线顶背离"
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板指数{index_div_type}生效")

                    # 【场景(1)】检查指数是否已处于死叉状态 + 个股周线RSI条件
                    # 根据buy_sell.md：指数顶背离需要个股周线RSI > 80 才能触发卖出
                    # 注意：检查的是顶背离形成日期附近（±5天）的最高周线RSI
                    index_macd_dead = self._is_index_macd_dead_cross_state(date, 'cyb')
                    index_sar_dead = self._is_index_sar_dead_cross_state(date, 'cyb')
                    if (index_macd_dead or index_sar_dead) and self._shares > 0:
                        # 获取背离形成日期附近（±5天）的最高周线RSI
                        div_date = date  # 背离确认日期
                        max_weekly_rsi, max_rsi_date = self._get_weekly_rsi_near_divergence(div_date, lookback_days=5)

                        # 定义死叉类型（用于日志输出）
                        dead_cross_type = "MACD" if index_macd_dead else "SAR"

                        # 检查个股周线RSI是否满足条件
                        if max_weekly_rsi >= 80:
                            # 根据指数背离类型确定卖出比例
                            if cyb_weekly_div_info:
                                # 周线顶背离：RSI > 80 卖出1/2，RSI > 85 清仓
                                if max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.CLEAR_ALL
                                else:
                                    sell_flag = SellFlag.SELL_HALF
                            else:
                                # 日线顶背离：RSI > 80 卖出1/3，RSI > 85 卖出1/2，RSI > 90 清仓
                                if max_weekly_rsi >= 90:
                                    sell_flag = SellFlag.CLEAR_ALL
                                elif max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.SELL_HALF
                                else:
                                    sell_flag = SellFlag.SELL_ONE_THIRD

                            sell_signal = SellSignal(
                                flag=sell_flag,
                                reason=f"{sell_flag.name} (sell_id={sell_id}-创业板指数{index_div_type}): 创业板指数{index_div_type}形成于 {date.strftime('%Y-%m-%d')}, 创业板指数{dead_cross_type}已处于死叉状态, 背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})>80，建议卖出",
                                daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi
                            )
                            self._execute_sell(date, sell_signal)
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板指数{index_div_type}+{dead_cross_type}死叉+背离附近周RSI={max_weekly_rsi:.2f}>80，触发卖出")
                            continue
                        else:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板指数{index_div_type}+{dead_cross_type}死叉，但背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})<80，不触发卖出")

                    # 【场景(2)和(3)】检查个股死叉状态（仅用于日志，不直接卖出）
                    stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                    if stock_macd_dead:
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态，但需等待创业板指数死叉或个股再次重新死叉时才卖出")

            # 【sell_id=6,12】科创板指数背离
            if any(sid in [6, 12] for sid in self.judge_sell_ids):
                kc_daily_div_info = self._get_index_divergence_info(date, 'kc', 'daily')
                kc_weekly_div_info = self._get_index_divergence_info(date, 'kc', 'weekly')
                if kc_daily_div_info is not None or kc_weekly_div_info is not None:
                    self.strategy.set_kc_index_divergence(self.state, kc_daily_div_info, kc_weekly_div_info)

                    index_div_type = "日线顶背离" if kc_daily_div_info else "周线顶背离"
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创板指数{index_div_type}生效")

                    # 【场景(1)】检查指数是否已处于死叉状态 + 个股周线RSI条件
                    # 根据buy_sell.md：指数顶背离需要个股周线RSI > 80 才能触发卖出
                    # 注意：检查的是顶背离形成日期附近（±5天）的最高周线RSI
                    index_macd_dead = self._is_index_macd_dead_cross_state(date, 'kc')
                    index_sar_dead = self._is_index_sar_dead_cross_state(date, 'kc')
                    if (index_macd_dead or index_sar_dead) and self._shares > 0:
                        # 获取背离形成日期附近（±5天）的最高周线RSI
                        div_date = date  # 背离确认日期
                        max_weekly_rsi, max_rsi_date = self._get_weekly_rsi_near_divergence(div_date, lookback_days=5)

                        # 定义死叉类型（用于日志输出）
                        dead_cross_type = "MACD" if index_macd_dead else "SAR"

                        # 检查个股周线RSI是否满足条件
                        if max_weekly_rsi >= 80:
                            # 根据指数背离类型确定卖出比例
                            if kc_weekly_div_info:
                                # 周线顶背离：RSI > 80 卖出1/2，RSI > 85 清仓
                                if max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.CLEAR_ALL
                                else:
                                    sell_flag = SellFlag.SELL_HALF
                            else:
                                # 日线顶背离：RSI > 80 卖出1/3，RSI > 85 卖出1/2，RSI > 90 清仓
                                if max_weekly_rsi >= 90:
                                    sell_flag = SellFlag.CLEAR_ALL
                                elif max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.SELL_HALF
                                else:
                                    sell_flag = SellFlag.SELL_ONE_THIRD

                            sell_signal = SellSignal(
                                flag=sell_flag,
                                reason=f"{sell_flag.name} (sell_id={sell_id}-科创板指数{index_div_type}): 科创板指数{index_div_type}形成于 {date.strftime('%Y-%m-%d')}, 科创板指数{dead_cross_type}已处于死叉状态, 背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})>80，建议卖出",
                                daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi
                            )
                            self._execute_sell(date, sell_signal)
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创板指数{index_div_type}+{dead_cross_type}死叉+背离附近周RSI={max_weekly_rsi:.2f}>80，触发卖出")
                            continue
                        else:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创板指数{index_div_type}+{dead_cross_type}死叉，但背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})<80，不触发卖出")

                    # 【场景(2)和(3)】检查个股死叉状态（仅用于日志，不直接卖出）
                    stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                    if stock_macd_dead:
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态，但需等待科创板指数死叉或个股再次重新死叉时才卖出")

            # 【新增】设置新指数背离状态（用于 sell_id=13,14,15）
            # 【重要】根据具体的sell_id只检测对应指数的背离，而不是检测所有指数
            # sell_id=13: 上证50指数
            # sell_id=14: 创业板50指数
            # sell_id=15: 科创板50指数

            # 【sell_id=13】上证50指数背离
            if 13 in self.judge_sell_ids:
                sh50_daily_div_info = self._get_index_divergence_info(date, 'sh50', 'daily')
                sh50_weekly_div_info = self._get_index_divergence_info(date, 'sh50', 'weekly')
                if sh50_daily_div_info is not None or sh50_weekly_div_info is not None:
                    self.strategy.set_sh50_index_divergence(self.state, sh50_daily_div_info, sh50_weekly_div_info)

                    index_div_type = "日线顶背离" if sh50_daily_div_info else "周线顶背离"
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 上证50指数{index_div_type}生效")

                    # 【场景(1)】检查指数是否已处于死叉状态 + 个股周线RSI条件
                    # 根据buy_sell.md：指数顶背离需要个股周线RSI > 80 才能触发卖出
                    # 注意：检查的是顶背离形成日期附近（±5天）的最高周线RSI
                    index_macd_dead = self._is_index_macd_dead_cross_state(date, 'sh50')
                    index_sar_dead = self._is_index_sar_dead_cross_state(date, 'sh50')
                    if (index_macd_dead or index_sar_dead) and self._shares > 0:
                        # 获取背离形成日期附近（±5天）的最高周线RSI
                        div_date = date  # 背离确认日期
                        max_weekly_rsi, max_rsi_date = self._get_weekly_rsi_near_divergence(div_date, lookback_days=5)

                        # 定义死叉类型（用于日志输出）
                        dead_cross_type = "MACD" if index_macd_dead else "SAR"

                        # 检查个股周线RSI是否满足条件
                        if max_weekly_rsi >= 80:
                            # 根据指数背离类型确定卖出比例
                            if sh50_weekly_div_info:
                                # 周线顶背离：RSI > 80 卖出1/2，RSI > 85 清仓
                                if max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.CLEAR_ALL
                                else:
                                    sell_flag = SellFlag.SELL_HALF
                            else:
                                # 日线顶背离：RSI > 80 卖出1/3，RSI > 85 卖出1/2，RSI > 90 清仓
                                if max_weekly_rsi >= 90:
                                    sell_flag = SellFlag.CLEAR_ALL
                                elif max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.SELL_HALF
                                else:
                                    sell_flag = SellFlag.SELL_ONE_THIRD

                            sell_signal = SellSignal(
                                flag=sell_flag,
                                reason=f"{sell_flag.name} (sell_id={sell_id}-上证50指数{index_div_type}): 上证50指数{index_div_type}形成于 {date.strftime('%Y-%m-%d')}, 上证50指数{dead_cross_type}已处于死叉状态, 背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})>80，建议卖出",
                                daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi
                            )
                            self._execute_sell(date, sell_signal)
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 上证50指数{index_div_type}+{dead_cross_type}死叉+背离附近周RSI={max_weekly_rsi:.2f}>80，触发卖出")
                            continue
                        else:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 上证50指数{index_div_type}+{dead_cross_type}死叉，但背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})<80，不触发卖出")

                    # 【场景(2)和(3)】检查个股死叉状态（仅用于日志，不直接卖出）
                    stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                    if stock_macd_dead:
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态，但需等待上证50指数死叉或个股再次重新死叉时才卖出")

            # 【sell_id=14】创业板50指数背离
            if 14 in self.judge_sell_ids:
                cyb50_daily_div_info = self._get_index_divergence_info(date, 'cyb50', 'daily')
                cyb50_weekly_div_info = self._get_index_divergence_info(date, 'cyb50', 'weekly')
                if cyb50_daily_div_info is not None or cyb50_weekly_div_info is not None:
                    self.strategy.set_cyb50_index_divergence(self.state, cyb50_daily_div_info, cyb50_weekly_div_info)

                    index_div_type = "日线顶背离" if cyb50_daily_div_info else "周线顶背离"
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板50指数{index_div_type}生效")

                    # 【场景(1)】检查指数是否已处于死叉状态 + 个股周线RSI条件
                    # 根据buy_sell.md：指数顶背离需要个股周线RSI > 80 才能触发卖出
                    # 注意：检查的是顶背离形成日期附近（±5天）的最高周线RSI
                    index_macd_dead = self._is_index_macd_dead_cross_state(date, 'cyb50')
                    index_sar_dead = self._is_index_sar_dead_cross_state(date, 'cyb50')
                    if (index_macd_dead or index_sar_dead) and self._shares > 0:
                        # 获取背离形成日期附近（±5天）的最高周线RSI
                        div_date = date  # 背离确认日期
                        max_weekly_rsi, max_rsi_date = self._get_weekly_rsi_near_divergence(div_date, lookback_days=5)

                        # 定义死叉类型（用于日志输出）
                        dead_cross_type = "MACD" if index_macd_dead else "SAR"

                        # 检查个股周线RSI是否满足条件
                        if max_weekly_rsi >= 80:
                            # 根据指数背离类型确定卖出比例
                            if cyb50_weekly_div_info:
                                # 周线顶背离：RSI > 80 卖出1/2，RSI > 85 清仓
                                if max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.CLEAR_ALL
                                else:
                                    sell_flag = SellFlag.SELL_HALF
                            else:
                                # 日线顶背离：RSI > 80 卖出1/3，RSI > 85 卖出1/2，RSI > 90 清仓
                                if max_weekly_rsi >= 90:
                                    sell_flag = SellFlag.CLEAR_ALL
                                elif max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.SELL_HALF
                                else:
                                    sell_flag = SellFlag.SELL_ONE_THIRD

                            sell_signal = SellSignal(
                                flag=sell_flag,
                                reason=f"{sell_flag.name} (sell_id={sell_id}-创业板50指数{index_div_type}): 创业板50指数{index_div_type}形成于 {date.strftime('%Y-%m-%d')}, 创业板50指数{dead_cross_type}已处于死叉状态, 背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})>80，建议卖出",
                                daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi
                            )
                            self._execute_sell(date, sell_signal)
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板50指数{index_div_type}+{dead_cross_type}死叉+背离附近周RSI={max_weekly_rsi:.2f}>80，触发卖出")
                            continue
                        else:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板50指数{index_div_type}+{dead_cross_type}死叉，但背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})<80，不触发卖出")

                    # 【场景(2)和(3)】检查个股死叉状态（仅用于日志，不直接卖出）
                    stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                    if stock_macd_dead:
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态，但需等待创业板50指数死叉或个股再次重新死叉时才卖出")

            # 【sell_id=15】科创板50指数背离
            if 15 in self.judge_sell_ids:
                kc50_daily_div_info = self._get_index_divergence_info(date, 'kc50', 'daily')
                kc50_weekly_div_info = self._get_index_divergence_info(date, 'kc50', 'weekly')
                if kc50_daily_div_info is not None or kc50_weekly_div_info is not None:
                    self.strategy.set_kc50_index_divergence(self.state, kc50_daily_div_info, kc50_weekly_div_info)

                    index_div_type = "日线顶背离" if kc50_daily_div_info else "周线顶背离"
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创板50指数{index_div_type}生效")

                    # 【场景(1)】检查指数是否已处于死叉状态 + 个股周线RSI条件
                    # 根据buy_sell.md：指数顶背离需要个股周线RSI > 80 才能触发卖出
                    # 注意：检查的是顶背离形成日期附近（±5天）的最高周线RSI
                    index_macd_dead = self._is_index_macd_dead_cross_state(date, 'kc50')
                    index_sar_dead = self._is_index_sar_dead_cross_state(date, 'kc50')
                    if (index_macd_dead or index_sar_dead) and self._shares > 0:
                        # 获取背离形成日期附近（±5天）的最高周线RSI
                        div_date = date  # 背离确认日期
                        max_weekly_rsi, max_rsi_date = self._get_weekly_rsi_near_divergence(div_date, lookback_days=5)

                        # 定义死叉类型（用于日志输出）
                        dead_cross_type = "MACD" if index_macd_dead else "SAR"

                        # 检查个股周线RSI是否满足条件
                        if max_weekly_rsi >= 80:
                            # 根据指数背离类型确定卖出比例
                            if kc50_weekly_div_info:
                                # 周线顶背离：RSI > 80 卖出1/2，RSI > 85 清仓
                                if max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.CLEAR_ALL
                                else:
                                    sell_flag = SellFlag.SELL_HALF
                            else:
                                # 日线顶背离：RSI > 80 卖出1/3，RSI > 85 卖出1/2，RSI > 90 清仓
                                if max_weekly_rsi >= 90:
                                    sell_flag = SellFlag.CLEAR_ALL
                                elif max_weekly_rsi >= 85:
                                    sell_flag = SellFlag.SELL_HALF
                                else:
                                    sell_flag = SellFlag.SELL_ONE_THIRD

                            sell_signal = SellSignal(
                                flag=sell_flag,
                                reason=f"{sell_flag.name} (sell_id={sell_id}-科创板50指数{index_div_type}): 科创板50指数{index_div_type}形成于 {date.strftime('%Y-%m-%d')}, 科创板50指数{dead_cross_type}已处于死叉状态, 背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})>80，建议卖出",
                                daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi
                            )
                            self._execute_sell(date, sell_signal)
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创板50指数{index_div_type}+{dead_cross_type}死叉+背离附近周RSI={max_weekly_rsi:.2f}>80，触发卖出")
                            continue
                        else:
                            self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创板50指数{index_div_type}+{dead_cross_type}死叉，但背离附近最高周RSI={max_weekly_rsi:.2f}({max_rsi_date.strftime('%Y-%m-%d')})<80，不触发卖出")

                    # 【场景(2)和(3)】检查个股死叉状态（仅用于日志，不直接卖出）
                    stock_macd_dead = metrics.get('macd', 0) < metrics.get('macd_signal', 0) if metrics.get('macd') and metrics.get('macd_signal') else False
                    if stock_macd_dead:
                        self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 个股MACD已处于死叉状态，但需等待科创板50指数死叉或个股再次重新死叉时才卖出")

            # 检测指数MACD死叉（用于 sell_id=7,8,9,10,11,12）
            sh_index_macd_cross_down = self._detect_index_macd_cross_down(date, 'sh') if any(sid in [7, 10] for sid in self.judge_sell_ids) else False
            cyb_index_macd_cross_down = self._detect_index_macd_cross_down(date, 'cyb') if any(sid in [8, 11] for sid in self.judge_sell_ids) else False
            kc_index_macd_cross_down = self._detect_index_macd_cross_down(date, 'kc') if any(sid in [9, 12] for sid in self.judge_sell_ids) else False

            # 检测新指数MACD死叉（用于 sell_id=13,14,15）
            sh50_index_macd_cross_down = self._detect_index_macd_cross_down(date, 'sh50') if any(sid in [13] for sid in self.judge_sell_ids) else False
            cyb50_index_macd_cross_down = self._detect_index_macd_cross_down(date, 'cyb50') if any(sid in [14] for sid in self.judge_sell_ids) else False
            kc50_index_macd_cross_down = self._detect_index_macd_cross_down(date, 'kc50') if any(sid in [15] for sid in self.judge_sell_ids) else False

            # 检测指数是否已处于死叉状态（用于 sell_id=10-15）
            # 当个股顶背离形成时，如果指数已处于死叉状态，需要等待个股死叉才卖出
            sh_index_dead_cross_state = self._is_index_macd_dead_cross_state(date, 'sh') if any(sid in [10] for sid in self.judge_sell_ids) else False
            cyb_index_dead_cross_state = self._is_index_macd_dead_cross_state(date, 'cyb') if any(sid in [11] for sid in self.judge_sell_ids) else False
            kc_index_dead_cross_state = self._is_index_macd_dead_cross_state(date, 'kc') if any(sid in [12] for sid in self.judge_sell_ids) else False
            sh50_index_dead_cross_state = self._is_index_macd_dead_cross_state(date, 'sh50') if any(sid in [13] for sid in self.judge_sell_ids) else False
            cyb50_index_dead_cross_state = self._is_index_macd_dead_cross_state(date, 'cyb50') if any(sid in [14] for sid in self.judge_sell_ids) else False
            kc50_index_dead_cross_state = self._is_index_macd_dead_cross_state(date, 'kc50') if any(sid in [15] for sid in self.judge_sell_ids) else False

            # 调用策略检测卖出信号（检查触发条件：SAR死叉或MACD死叉）
            sell_signal = self.strategy.check_sell_signal(
                date=date,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                sar_cross_down=sar_cross_down,
                macd_cross_down=macd_cross_down,
                state=self.state,
                position=self._shares * metrics.get('close_price', 0) / self.initial_capital if self._shares > 0 else 0,
                sell_id=sell_id,
                sh_index_macd_cross_down=sh_index_macd_cross_down,
                cyb_index_macd_cross_down=cyb_index_macd_cross_down,
                kc_index_macd_cross_down=kc_index_macd_cross_down,
                sh50_index_macd_cross_down=sh50_index_macd_cross_down,
                cyb50_index_macd_cross_down=cyb50_index_macd_cross_down,
                kc50_index_macd_cross_down=kc50_index_macd_cross_down,
                macd_dead_cross_state=macd_dead_cross_state,
                sh_index_dead_cross_state=sh_index_dead_cross_state,
                cyb_index_dead_cross_state=cyb_index_dead_cross_state,
                kc_index_dead_cross_state=kc_index_dead_cross_state,
                sh50_index_dead_cross_state=sh50_index_dead_cross_state,
                cyb50_index_dead_cross_state=cyb50_index_dead_cross_state,
                kc50_index_dead_cross_state=kc50_index_dead_cross_state
            )

            # 处理卖出信号
            if sell_signal is not None:
                if self._shares > 0:
                    self._execute_sell(date, sell_signal)
                    continue  # 实际卖出后，当天不再买入
                else:
                    self._execute_fake_sell(date, sell_signal)
                    # 假卖出后不跳过，继续检测买入信号（假卖出只是重置状态，没有实际交易）

            # 获取指数ETF的RSI数据
            index_daily_rsi = np.nan
            sh_index_daily_rsi = np.nan

            # 获取创业板指数的RSI
            if self._index_metrics_df is not None:
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    matching = self._index_metrics_df[self._index_metrics_df.index.strftime('%Y-%m-%d') == date_str]
                    if len(matching) > 0:
                        rsi_val = matching.iloc[0]['rsi_daily']
                        if pd.notna(rsi_val):
                            index_daily_rsi = float(rsi_val)
                            # 调试：记录创业板RSI值（当buy_id包含创业板相关ID且RSI较低时）
                            if any(bid in [3, 5] for bid in self.judge_buy_ids) and self._sold_cash > 0 and index_daily_rsi < 30:
                                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 创业板指数RSI={index_daily_rsi:.2f}, 待买回资金={self._sold_cash:.2f}")
                except Exception as e:
                    if any(bid in [3, 5] for bid in self.judge_buy_ids):
                        self.logger.warning(f"[{date.strftime('%Y-%m-%d')}] 获取创业板指数RSI失败: {e}")
            else:
                # 调试：记录指数数据缺失（仅在需要创业板数据时）
                if any(bid in [3, 5] for bid in self.judge_buy_ids) and self._sold_cash > 0:
                    self.logger.warning(f"[{date.strftime('%Y-%m-%d')}] 创业板指数数据缺失(_index_metrics_df=None), 无法检查buy_id=5条件")

            # 获取上证指数的RSI
            if self._sh_index_metrics_df is not None:
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    matching = self._sh_index_metrics_df[self._sh_index_metrics_df.index.strftime('%Y-%m-%d') == date_str]
                    if len(matching) > 0:
                        rsi_val = matching.iloc[0]['rsi_daily']
                        if pd.notna(rsi_val):
                            sh_index_daily_rsi = float(rsi_val)
                except Exception:
                    pass

            # 获取科创综指的RSI
            kc_index_daily_rsi = np.nan
            if self._kc_index_metrics_df is not None:
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    matching = self._kc_index_metrics_df[self._kc_index_metrics_df.index.strftime('%Y-%m-%d') == date_str]
                    if len(matching) > 0:
                        rsi_val = matching.iloc[0]['rsi_daily']
                        if pd.notna(rsi_val):
                            kc_index_daily_rsi = float(rsi_val)
                            # 调试：记录科创综指RSI值（当buy_id=8且RSI较低时）
                            if any(bid == 8 for bid in self.judge_buy_ids) and kc_index_daily_rsi < 30:
                                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 科创综指RSI={kc_index_daily_rsi:.2f}, 待买回资金={self._sold_cash:.2f}")
                except Exception as e:
                    self.logger.warning(f"[{date.strftime('%Y-%m-%d')}] 获取科创综指RSI失败: {e}")

            # 调试：根据buy_id记录对应的指数RSI值
            index_rsi_to_check = np.nan
            index_name = "未知指数"
            for buy_id in self.judge_buy_ids:
                if buy_id in [3, 5]:  # 创业板指数
                    index_rsi_to_check = index_daily_rsi
                    index_name = "创业板指数"
                elif buy_id in [6, 7, 10]:  # 上证指数
                    index_rsi_to_check = sh_index_daily_rsi
                    index_name = "上证指数"
                elif buy_id == 8:  # 科创综指
                    index_rsi_to_check = kc_index_daily_rsi
                    index_name = "科创综指"

                if not np.isnan(index_rsi_to_check) and index_rsi_to_check < 25:
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 买入条件检测: {index_name}RSI={index_rsi_to_check:.2f}, 日RSI={daily_rsi:.2f}, 周RSI={weekly_rsi:.2f}, 新资金={self._cash:.2f}, 待买回={self._sold_cash:.2f}")
                    break  # 只打印第一个匹配的指数

            # 获取当前日期的月线RSI（用于buy_id=15-26）
            monthly_rsi = np.nan
            if self._monthly_rsi_series is not None and len(self._monthly_rsi_series) > 0:
                date_str = date.strftime('%Y-%m-%d')
                matching = self._monthly_rsi_series[self._monthly_rsi_series.index.strftime('%Y-%m-%d') == date_str]
                if len(matching) > 0:
                    monthly_rsi = float(matching.iloc[0])
                else:
                    # 如果找不到精确匹配，尝试找最近的日期
                    try:
                        idx = self._monthly_rsi_series.index.get_indexer([date], method='nearest')[0]
                        if idx >= 0:
                            monthly_rsi = float(self._monthly_rsi_series.iloc[idx])
                    except Exception:
                        pass

            # 调用策略检测买入信号
            buy_signal = self.strategy.check_buy_signal(
                date=date,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                state=self.state,
                has_new_cash=self._cash > 0,
                has_sold_cash=self._sold_cash > 0,
                index_daily_rsi=index_daily_rsi,
                sh_index_daily_rsi=sh_index_daily_rsi,
                kc_index_daily_rsi=kc_index_daily_rsi,
                monthly_rsi=monthly_rsi
            )

            # 调试：记录买入信号检测结果
            if self._sold_cash > 0 or self._cash > 0:
                if buy_signal is not None:
                    self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 买入信号触发: {buy_signal.reason}")
                else:
                    # 记录为什么没有触发买入（根据buy_id动态生成）
                    reasons = []
                    if np.isnan(daily_rsi):
                        reasons.append("日RSI无效")
                    if np.isnan(weekly_rsi):
                        reasons.append("周RSI无效")

                    # 检查各buy_id条件（动态根据buy_id类型）
                    for buy_id in self.strategy.buy_ids:
                        if buy_id == 1 and not (daily_rsi < 20 and weekly_rsi < 25):
                            reasons.append(f"buy_id=1不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f})")
                        elif buy_id == 2 and not (daily_rsi < 25 and weekly_rsi < 30):
                            reasons.append(f"buy_id=2不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f})")
                        elif buy_id == 4 and not (daily_rsi < 20 and weekly_rsi < 20):
                            reasons.append(f"buy_id=4不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f})")
                        elif buy_id == 5:
                            if np.isnan(index_daily_rsi):
                                reasons.append(f"buy_id=5不满足(创业板RSI数据缺失)")
                            elif not (index_daily_rsi < 20):
                                reasons.append(f"buy_id=5不满足(创业板RSI={index_daily_rsi:.2f})")
                        elif buy_id == 6:
                            if np.isnan(sh_index_daily_rsi):
                                reasons.append(f"buy_id=6不满足(上证RSI数据缺失)")
                            elif not (sh_index_daily_rsi < 20):
                                reasons.append(f"buy_id=6不满足(上证RSI={sh_index_daily_rsi:.2f})")
                        elif buy_id == 7:
                            if np.isnan(sh_index_daily_rsi):
                                reasons.append(f"buy_id=7不满足(上证RSI数据缺失)")
                            elif not (sh_index_daily_rsi < 25):
                                reasons.append(f"buy_id=7不满足(上证RSI={sh_index_daily_rsi:.2f})")
                        elif buy_id == 8:
                            if np.isnan(kc_index_daily_rsi):
                                reasons.append(f"buy_id=8不满足(科创综指RSI数据缺失)")
                            elif not (kc_index_daily_rsi < 20):
                                reasons.append(f"buy_id=8不满足(科创综指RSI={kc_index_daily_rsi:.2f})")
                        elif buy_id == 15:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=15不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and monthly_rsi < 22):
                                reasons.append(f"buy_id=15不满足(日RSI={daily_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 16:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=16不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and monthly_rsi < 23):
                                reasons.append(f"buy_id=16不满足(日RSI={daily_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 17:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=17不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and monthly_rsi < 20):
                                reasons.append(f"buy_id=17不满足(日RSI={daily_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 18:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=18不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 40):
                                reasons.append(f"buy_id=18不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 19:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=19不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 15):
                                reasons.append(f"buy_id=19不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 20:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=20不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 25):
                                reasons.append(f"buy_id=20不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 21:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=21不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 20):
                                reasons.append(f"buy_id=21不满足(日RSI={daily_rsi:.2f},周RSI={weekly_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 22:
                            if not (daily_rsi < 30):
                                reasons.append(f"buy_id=22不满足(日RSI={daily_rsi:.2f})")
                        elif buy_id == 23:
                            if not (weekly_rsi < 20):
                                reasons.append(f"buy_id=23不满足(周RSI={weekly_rsi:.2f})")
                        elif buy_id == 24:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=24不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and monthly_rsi < 15):
                                reasons.append(f"buy_id=24不满足(日RSI={daily_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 25:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=25不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and monthly_rsi < 40):
                                reasons.append(f"buy_id=25不满足(日RSI={daily_rsi:.2f},月RSI={monthly_rsi:.2f})")
                        elif buy_id == 26:
                            if np.isnan(monthly_rsi):
                                reasons.append(f"buy_id=26不满足(月线RSI数据缺失)")
                            elif not (daily_rsi < 20 and monthly_rsi < 25):
                                reasons.append(f"buy_id=26不满足(日RSI={daily_rsi:.2f},月RSI={monthly_rsi:.2f})")

                    # 只有在有资金且指标接近阈值时才打印调试日志
                    should_log = False
                    for buy_id in self.strategy.buy_ids:
                        if buy_id in [3, 5] and not np.isnan(index_daily_rsi) and index_daily_rsi < 30:
                            should_log = True
                        elif buy_id in [6, 7] and not np.isnan(sh_index_daily_rsi) and sh_index_daily_rsi < 30:
                            should_log = True
                        elif buy_id == 8 and not np.isnan(kc_index_daily_rsi) and kc_index_daily_rsi < 30:
                            should_log = True
                        elif buy_id in [15, 16, 17, 19, 20, 21, 24, 26] and not np.isnan(monthly_rsi) and monthly_rsi < 30:
                            should_log = True
                        elif buy_id in [18, 25] and not np.isnan(monthly_rsi) and monthly_rsi < 50:
                            should_log = True
                        elif buy_id == 22 and not np.isnan(daily_rsi) and daily_rsi < 35:
                            should_log = True
                        elif buy_id == 23 and not np.isnan(weekly_rsi) and weekly_rsi < 30:
                            should_log = True

                    if len(reasons) > 0 and should_log:
                        self.logger.debug(f"[{date.strftime('%Y-%m-%d')}] 买入未触发: {', '.join(reasons)}")

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
        if self._metrics_df is None or len(self._metrics_df) == 0:
            return {
                'total_return': 0.0,
                'benchmark_return': 0.0,
                'excess_return': 0.0,
                'max_drawdown': 0.0,
                'final_shares': 0.0,
                'final_cash': self.initial_capital,
                'final_sold_cash': 0.0,
                'trades': 0,
                'buy_count': 0,
                'sell_count': 0
            }

        # 计算基准收益率（买入持有策略）
        first_price = float(self._metrics_df['close_price'].iloc[0])
        last_price = float(self._metrics_df['close_price'].iloc[-1])
        benchmark_return = (last_price / first_price - 1.0) * 100 if first_price > 0 else 0.0

        if len(self._trade_records) == 0:
            return {
                'total_return': 0.0,
                'benchmark_return': benchmark_return,
                'excess_return': -benchmark_return,
                'max_drawdown': 0.0,
                'final_shares': 0.0,
                'final_cash': self.initial_capital,
                'final_sold_cash': 0.0,
                'trades': 0,
                'buy_count': 0,
                'sell_count': 0
            }

        # 计算最终市值
        final_value = self._shares * last_price + self._cash + self._sold_cash
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

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

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤率（返回负数形式，如-50%表示下跌50%）"""
        if len(self._trade_records) == 0 or self._metrics_df is None:
            return 0.0

        trading_dates = self._get_trading_dates()
        if len(trading_dates) == 0:
            return 0.0

        # 计算每日市值
        daily_values = []
        for date in trading_dates:
            metrics = self._get_metrics_at_date(date)
            price = metrics.get('close_price', np.nan)
            if np.isnan(price) or price <= 0:
                continue

            # 根据交易记录计算当日持仓状态
            shares = 0.0
            cash = self.initial_capital
            sold_cash = 0.0
            for trade in self._trade_records:
                if trade.date <= date:
                    if trade.action == 'buy':
                        # 买入：shares增加，资金减少
                        buy_cost = trade.amount * trade.price
                        shares += trade.amount
                        # 先从sold_cash买回，再从cash买入新资金
                        if buy_cost <= sold_cash:
                            sold_cash -= buy_cost
                        else:
                            remaining_cost = buy_cost - sold_cash
                            sold_cash = 0.0
                            cash -= remaining_cost if remaining_cost <= cash else 0
                    else:
                        # 卖出：shares减少（确保不变成负数），sold_cash增加
                        sell_amount = min(trade.amount, shares) if shares > 0 else 0
                        shares -= sell_amount
                        sold_cash += sell_amount * trade.price

            value = shares * price + cash + sold_cash
            # 确保value是正数才记录
            if value > 0:
                daily_values.append((date, value))

        if len(daily_values) < 2:
            return 0.0

        # 计算最大回撤（使用标准定义：从峰值下跌的百分比，输出负数）
        max_value = daily_values[0][1]
        max_drawdown = 0.0  # 最大回撤幅度（正数，但输出时会变成负数）

        for date, value in daily_values:
            if value > max_value:
                max_value = value
            if max_value > 0:
                drawdown_pct = (max_value - value) / max_value * 100
                if drawdown_pct > max_drawdown:
                    max_drawdown = drawdown_pct

        # 返回负数形式（如-50%表示下跌50%）
        return -max_drawdown

    def generate_report(self) -> Dict:
        """生成回测报告"""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("回测报告（数据库数据）")
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
        last_price = float(self._metrics_df['close_price'].iloc[-1]) if self._metrics_df is not None else 0.0
        final_value = results['final_shares'] * last_price + results['final_cash'] + results['final_sold_cash']

        # 构建报告数据
        report_data = {
            '股票代码': self.stock_id,
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
            '数据来源': 'MySQL数据库',
            '日志文件': self.log_file,
            '图表文件': chart_path or ''
        }

        return report_data

    def plot_backtest_results(self, output_path: Optional[str] = None) -> Optional[str]:
        """绘制回测结果图表"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        if self._metrics_df is None or len(self._trade_records) == 0:
            self.logger.warning("无数据，无法绘制图表")
            return None

        # 获取数据
        dates = self._metrics_df.index
        close_data = self._metrics_df['close_price'].astype(float)
        daily_rsi_data = self._metrics_df['rsi_daily'].astype(float)
        weekly_rsi_data = self._metrics_df['rsi_weekly'].astype(float)

        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        fig.suptitle(f'{self.stock_id} ({self.stock_name}) 回测结果（数据库数据） ({self.start_date} ~ {self.end_date})\n'
                    f'策略: {self.strategy}',
                    fontsize=12, fontweight='bold')

        ax_price = axes[0]
        ax_rsi = axes[1]
        ax_position = axes[2]

        # 绘制价格曲线
        ax_price.plot(dates, close_data.values, 'b-', linewidth=1.5, label='收盘价', alpha=0.8)

        # 标记买入点
        buy_trades = [t for t in self._trade_records if t.action == 'buy']
        for trade in buy_trades:
            ax_price.scatter(trade.date, trade.price, color='green', marker='^', s=150,
                            label='买入' if trade == buy_trades[0] else '', zorder=6)
            ax_price.annotate(f'买入\n{trade.price:.3f}',
                             (trade.date, trade.price),
                             textcoords="offset points", xytext=(5, 15),
                             fontsize=8, color='green', fontweight='bold')

        # 标记卖出点
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

        # 绘制RSI曲线
        ax_rsi.plot(dates, daily_rsi_data.values, 'b-', linewidth=1, label='日线RSI(6)', alpha=0.7)
        ax_rsi.plot(dates, weekly_rsi_data.values, 'r-', linewidth=1.5, label='周线RSI(6)', alpha=0.7)

        # RSI阈值线
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='超卖线(30)')
        ax_rsi.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='超买线(70)')
        ax_rsi.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='警戒线(80)')
        ax_rsi.axhline(y=90, color='purple', linestyle='--', alpha=0.5, label='清仓线(90)')

        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(loc='upper left')
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # 绘制持仓比例曲线
        position_values = []
        current_position = 0.0
        trade_idx = 0

        for date in dates:
            while trade_idx < len(self._trade_records) and self._trade_records[trade_idx].date <= date:
                trade = self._trade_records[trade_idx]
                if trade.action == 'buy':
                    current_position += trade.amount
                else:
                    current_position -= trade.amount
                trade_idx += 1
            position_values.append(current_position)

        ax_position.fill_between(dates, 0, position_values, color='blue', alpha=0.3, label='持仓')
        ax_position.plot(dates, position_values, 'b-', linewidth=1.5, alpha=0.8)

        ax_position.set_ylabel('持仓股数')
        ax_position.legend(loc='upper left')
        ax_position.grid(True, alpha=0.3)
        ax_position.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_position.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        plt.xticks(rotation=45)
        plt.tight_layout()

        # 保存图表
        if output_path is None:
            output_path = str(self.output_dir / f'db_backtest_chart_{self.stock_id}.png')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"回测图表已保存: {output_path}")
        return output_path


# ==================== 批量回测执行器 ====================

class BatchDBBacktestRunner:
    """
    批量数据库回测执行器

    从 stocks.json 加载股票列表，批量执行回测
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000.0) -> None:
        """
        初始化批量回测执行器

        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # 输出目录
        logs_base = project_root / "logs"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = logs_base / f'batch_db_backtest_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 配置日志
        self._setup_logging()

    def _setup_logging(self) -> None:
        """配置日志"""
        log_file = self.output_dir / 'batch_backtest.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(str(log_file), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_stock_list(self) -> List[Dict]:
        """从 stocks_backtest.csv 加载股票列表（只加载 active=1 的股票）"""
        stock_csv_path = project_root / "config" / "stocks_backtest.csv"

        if not stock_csv_path.exists():
            self.logger.warning(f"股票列表文件不存在: {stock_csv_path}")
            return []

        # 使用 dtype 参数确保股票代码作为字符串读取，不丢失前导0
        # comment='#' 支持跳过以 # 开头的注释行
        df = pd.read_csv(str(stock_csv_path), dtype={'symbol': str}, comment='#')

        # 只加载 active=1 的股票
        active_df = df[df['active'] == 1]
        if len(active_df) == 0:
            self.logger.warning("没有活跃的股票")
            return []

        stock_list = []
        for row in active_df.to_dict('records'):
            # 解析策略ID（CSV中用分号分隔）
            def parse_ids(ids_str):
                if pd.isna(ids_str) or ids_str == '':
                    return []
                return [int(float(x)) for x in str(ids_str).split(';') if x.strip()]

            ticker = str(row['symbol'])

            stock_list.append({
                'ticker': ticker,
                'stock_name': str(row.get('name', '')),  # 从CSV获取股票名称
                'start_date': str(row['start_date']),
                'end_date': str(row['end_date']),
                'judge_buy_ids': parse_ids(row.get('judge_buy_ids', '')),
                'judge_t_ids': parse_ids(row.get('judge_t_ids', '')),
                'judge_sell_ids': parse_ids(row.get('judge_sell_ids', '')),
                'initial_capital': float(row.get('initial_capital', 100000)),
                'fully_invested': int(row.get('fully_invested', 1))  # 默认全仓持有
            })

        return stock_list

    def run_batch_backtest(self) -> List[Dict]:
        """
        执行批量回测

        Returns:
            List[Dict]: 所有股票的回测结果
        """
        self.logger.info("=" * 60)
        self.logger.info("批量数据库回测开始")
        self.logger.info("=" * 60)

        stock_list = self.load_stock_list()
        if not stock_list:
            self.logger.error("股票列表为空")
            return []

        self.logger.info(f"加载到 {len(stock_list)} 只股票（active=1）")

        results = []

        for stock in stock_list:
            ticker = stock.get('ticker', '')
            start_date = stock.get('start_date', self.start_date)
            end_date = stock.get('end_date', self.end_date)
            stock_name = stock.get('stock_name', '')  # 从CSV获取股票名称
            judge_buy_ids = stock.get('judge_buy_ids', [1])
            judge_t_ids = stock.get('judge_t_ids', [2])
            judge_sell_ids = stock.get('judge_sell_ids', [1])
            initial_capital = stock.get('initial_capital', self.initial_capital)
            fully_invested = stock.get('fully_invested', 1)

            self.logger.info("")
            self.logger.info(f"处理股票: {ticker} ({stock_name})")
            self.logger.info(f"时间段: {start_date} ~ {end_date}")
            self.logger.info(f"策略: 买入={judge_buy_ids}, 做T={judge_t_ids}, 卖出={judge_sell_ids}, 全仓={fully_invested}")

            try:
                runner = DBBacktestRunner(
                    stock_id=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    stock_name=stock_name,  # 使用CSV中的名称，数据库获取时会补充
                    judge_buy_ids=judge_buy_ids,
                    judge_t_ids=judge_t_ids,
                    judge_sell_ids=judge_sell_ids,
                    initial_capital=initial_capital,
                    output_dir=self.output_dir / ticker,
                    fully_invested=fully_invested
                )

                runner.run_backtest()
                report = runner.generate_report()
                chart_path = runner.plot_backtest_results()
                if chart_path:
                    report['图表文件'] = chart_path
                results.append(report)

            except Exception as e:
                self.logger.error(f"股票 {ticker} 回测失败: {e}")
                results.append({
                    '股票代码': ticker,
                    '股票名称': runner.stock_name if 'runner' in locals() else '',
                    '是否成功': '否',
                    '错误信息': str(e)
                })

        # 输出汇总报告
        self._generate_summary_report(results)

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("批量回测完成")
        self.logger.info("=" * 60)

        return results

    def _generate_summary_report(self, results: List[Dict]) -> None:
        """生成汇总报告"""
        self.logger.info("")
        self.logger.info("【汇总报告】")
        self.logger.info("-" * 80)

        # CSV格式输出
        header = "股票代码,股票名称,策略收益率(%),基准收益率(%),超额收益率(%),最大回撤率(%),交易次数,是否成功,错误信息"
        self.logger.info(header)

        for r in results:
            row = f"{r.get('股票代码', '')},{r.get('股票名称', '')},{r.get('策略收益率(%)', '')},{r.get('基准收益率(%)', '')},{r.get('超额收益率(%)', '')},{r.get('最大回撤率(%)', '')},{r.get('交易次数', '')},{r.get('是否成功', '')},{r.get('错误信息', '')}"
            self.logger.info(row)

        # 保存到CSV文件
        csv_path = self.output_dir / 'batch_backtest_summary.csv'
        df = pd.DataFrame(results)
        df.to_csv(str(csv_path), index=False, encoding='utf-8-sig')
        self.logger.info(f"汇总报告已保存: {csv_path}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    # ===== 默认执行批量回测 =====
    # 从 stocks_backtest.csv 加载股票列表，每只股票使用各自的参数
    START_DATE = '2020-01-01'
    END_DATE = '2026-04-10'
    INITIAL_CAPITAL = 100000

    batch_runner = BatchDBBacktestRunner(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL
    )
    batch_results = batch_runner.run_batch_backtest()

    # ===== 执行单只股票回测（可选）=====
    # STOCK_ID = '159873'              # 医疗设备ETF
    # STOCK_NAME = '医疗设备ETF'
    # START_DATE = '2020-01-01'
    # END_DATE = '2026-04-10'
    # JUDGE_BUY_IDS = [1, 5]
    # JUDGE_T_IDS = [2]
    # JUDGE_SELL_IDS = [1]
    # INITIAL_CAPITAL = 100000
    #
    # runner = DBBacktestRunner(
    #     stock_id=STOCK_ID,
    #     start_date=START_DATE,
    #     end_date=END_DATE,
    #     stock_name=STOCK_NAME,
    #     judge_buy_ids=JUDGE_BUY_IDS,
    #     judge_t_ids=JUDGE_T_IDS,
    #     judge_sell_ids=JUDGE_SELL_IDS,
    #     initial_capital=INITIAL_CAPITAL
    # )
    # results = runner.run_backtest()
    # report = runner.generate_report()
    # runner.generate_chart()