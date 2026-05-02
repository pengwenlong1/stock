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
                 output_dir: Optional[Path] = None) -> None:
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
        """
        self.stock_id = stock_id
        self.stock_name = stock_name
        self.start_date = start_date
        self.end_date = end_date
        self.judge_buy_ids = judge_buy_ids
        self.judge_t_ids = judge_t_ids
        self.judge_sell_ids = judge_sell_ids
        self.initial_capital = initial_capital

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

        # 指数数据（用于大盘参照）
        self._index_metrics_df: Optional[pd.DataFrame] = None  # 创业板指 (399006)
        self._sh_index_metrics_df: Optional[pd.DataFrame] = None  # 上证指数 (000001)

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
        """配置日志"""
        log_file = self.output_dir / f'db_backtest_{self.stock_id}.log'

        # 清除已有handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

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
        self.log_file = str(log_file)

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
                SELECT trade_date, rsi_daily
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
                SELECT trade_date, rsi_daily
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

        return True

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
            current_dif = metrics.get('dif', np.nan)
            current_dea = metrics.get('dea', np.nan)

            if not np.isnan(current_dif) and not np.isnan(current_dea):
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
            if weekly_top_div and self.state.weekly_divergence_flag == 0:
                self.state.weekly_divergence_flag = 1
                divergence_info = {
                    'date': date,
                    'prev_high': metrics.get('close_price', 0.0),
                    'curr_high': metrics.get('close_price', 0.0)
                }
                self.state.weekly_divergence_info = divergence_info
                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 周线顶背离形成，等待死叉触发生效")

            # 【日线顶背离形成】设置标志，等待死叉触发生效后卖出
            if daily_top_div and self.state.daily_divergence_flag == 0:
                self.state.daily_divergence_flag = 1
                divergence_info = {
                    'date': date,
                    'prev_high': metrics.get('close_price', 0.0),
                    'curr_high': metrics.get('close_price', 0.0)
                }
                self.state.daily_divergence_info = divergence_info
                self.logger.info(f"[{date.strftime('%Y-%m-%d')}] 日线顶背离形成，等待死叉触发生效")

            # 调用策略检测卖出信号（检查触发条件：SAR死叉或MACD死叉）
            sell_signal = self.strategy.check_sell_signal(
                date=date,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                sar_cross_down=sar_cross_down,
                macd_cross_down=macd_cross_down,
                state=self.state,
                position=self._shares * metrics.get('close_price', 0) / self.initial_capital if self._shares > 0 else 0,
                sell_id=sell_id
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

            # 获取创业板ETF的RSI
            if self._index_metrics_df is not None:
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    matching = self._index_metrics_df[self._index_metrics_df.index.strftime('%Y-%m-%d') == date_str]
                    if len(matching) > 0:
                        rsi_val = matching.iloc[0]['rsi_daily']
                        if pd.notna(rsi_val):
                            index_daily_rsi = float(rsi_val)
                except Exception:
                    pass

            # 获取上证指数ETF的RSI
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

            # 检查买入冷却期状态（调试）
            if self.state.last_sell_date is not None:
                days_since_sell = (date - self.state.last_sell_date).days
                if days_since_sell < COOLDOWN_DAYS:
                    self.logger.debug(f"[{date.strftime('%Y-%m-%d')}] 买入冷却期: 距上次卖出{days_since_sell}天 < {COOLDOWN_DAYS}天")

            # 调试：记录买入条件检测前的状态
            if not np.isnan(index_daily_rsi) and index_daily_rsi < 20:
                self.logger.debug(f"[{date.strftime('%Y-%m-%d')}] 买入条件可能触发: 创业板RSI={index_daily_rsi:.2f}, last_sell_date={self.state.last_sell_date}")

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
        """计算最大回撤率"""
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
        df = pd.read_csv(str(stock_csv_path), dtype={'symbol': str})

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
                'start_date': str(row['start_date']),
                'end_date': str(row['end_date']),
                'judge_buy_ids': parse_ids(row.get('judge_buy_ids', '')),
                'judge_t_ids': parse_ids(row.get('judge_t_ids', '')),
                'judge_sell_ids': parse_ids(row.get('judge_sell_ids', '')),
                'initial_capital': float(row.get('initial_capital', 100000))
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
            judge_buy_ids = stock.get('judge_buy_ids', [1])
            judge_t_ids = stock.get('judge_t_ids', [2])
            judge_sell_ids = stock.get('judge_sell_ids', [1])
            initial_capital = stock.get('initial_capital', self.initial_capital)

            self.logger.info("")
            self.logger.info(f"处理股票: {ticker}")
            self.logger.info(f"时间段: {start_date} ~ {end_date}")
            self.logger.info(f"策略: 买入={judge_buy_ids}, 做T={judge_t_ids}, 卖出={judge_sell_ids}")

            try:
                runner = DBBacktestRunner(
                    stock_id=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    stock_name='',  # 名称从数据库获取
                    judge_buy_ids=judge_buy_ids,
                    judge_t_ids=judge_t_ids,
                    judge_sell_ids=judge_sell_ids,
                    initial_capital=initial_capital,
                    output_dir=self.output_dir / ticker
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