# -*- coding: utf-8 -*-
"""
数据库回测执行器模块 (backtest_runner_db.py)

【功能说明】
从MySQL数据库读取已计算好的技术指标数据，进行回测判断。
不再现场计算RSI、MACD、均线等指标，直接从stock_daily_metrics表读取。

【数据库配置】
- 库名: stocks
- 用户名: root
- 密码: 123456

【使用方法】
    python src/runner/backtest_runner_db.py

作者：量化交易团队
"""
import csv
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

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

@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    action: str           # 'buy' 或 'sell'
    price: float
    amount: float
    daily_rsi: float
    weekly_rsi: float
    reason: str


@dataclass
class BacktestConfig:
    """回测配置"""
    symbol: str
    start_date: str
    end_date: str
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]
    initial_capital: float


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    config: BacktestConfig
    stock_name: str = ""
    total_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    max_drawdown: float = 0.0
    trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    final_value: float = 0.0
    log_file: str = ""
    chart_file: str = ""
    success: bool = True
    error_message: str = ""


# ==================== 交易所前缀处理 ====================

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

    if code.startswith('6') or code.startswith('5'):
        return f'SHSE.{code}'
    elif code.startswith('0') or code.startswith('3') or code.startswith('1'):
        return f'SZSE.{code}'
    else:
        return f'SZSE.{code}'


# ==================== 数据库连接器 ====================

class MySQLConnector:
    """MySQL数据库连接器"""

    def __init__(self, config: Dict = None) -> None:
        if config is None:
            config = DB_CONFIG

        self.config = config
        self.connection = None

        if pymysql is None:
            raise ImportError("请先安装pymysql: pip install pymysql")

    def connect(self) -> bool:
        try:
            self.connection = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset=self.config['charset']
            )
            logging.info(f"MySQL连接成功: {self.config['database']}")
            return True
        except Exception as e:
            logging.error(f"MySQL连接失败: {e}")
            return False

    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def query(self, sql: str, params: tuple = None) -> List[tuple]:
        if not self.connection:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"查询失败: {e}")
            return []

    def get_stock_id(self, stock_code: str) -> Optional[int]:
        """获取股票ID"""
        result = self.query(
            "SELECT stock_id FROM stock_info WHERE stock_code = %s",
            (stock_code,)
        )
        if result:
            return result[0][0]
        return None

    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        result = self.query(
            "SELECT stock_name FROM stock_info WHERE stock_code = %s",
            (stock_code,)
        )
        if result:
            return result[0][0]
        return "未知"

    def get_daily_metrics(self, stock_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线指标数据

        Args:
            stock_id: 股票ID
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 日线指标数据
        """
        sql = """
        SELECT trade_date, close_price, ma5, ma10, ma5_ma10_dead_cross,
               rsi_daily, rsi_weekly, macd, is_top_divergence, is_bottom_divergence,
               is_local_high, is_local_low, status
        FROM stock_daily_metrics
        WHERE stock_id = %s AND trade_date >= %s AND trade_date <= %s
        ORDER BY trade_date
        """
        results = self.query(sql, (stock_id, start_date, end_date))

        if not results:
            return pd.DataFrame()

        columns = [
            'trade_date', 'close_price', 'ma5', 'ma10', 'ma5_ma10_dead_cross',
            'rsi_daily', 'rsi_weekly', 'macd', 'is_top_divergence', 'is_bottom_divergence',
            'is_local_high', 'is_local_low', 'status'
        ]

        df = pd.DataFrame(results, columns=columns)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        return df


# ==================== 判断逻辑 ====================

class DBJudger:
    """
    数据库判断器

    【功能说明】
    从数据库读取的指标数据，进行买入/卖出判断。
    """

    def __init__(self) -> None:
        """初始化判断器"""
        pass

    def judge_buy(
        self,
        daily_rsi: float,
        weekly_rsi: float,
        buy_id: int,
        is_local_low: int = 0
    ) -> Tuple[bool, str]:
        """
        判断买入条件

        Args:
            daily_rsi: 日线RSI
            weekly_rsi: 周线RSI
            buy_id: 买入策略ID
            is_local_low: 是否局部低点

        Returns:
            Tuple[bool, str]: (是否买入, 原因)
        """
        # 处理None值
        if daily_rsi is None:
            daily_rsi = 50.0
        if weekly_rsi is None:
            weekly_rsi = 50.0

        # buy_id = 1: 保守型买入 - 日线RSI < 20 且 周线RSI < 25
        if buy_id == 1:
            if daily_rsi < 20 and weekly_rsi < 25:
                return True, f"保守型买入: 日RSI={daily_rsi:.2f}<20, 周RSI={weekly_rsi:.2f}<25"
            return False, ""

        # buy_id = 2: 标准型买入 - 日线RSI < 25 且 周线RSI < 30
        elif buy_id == 2:
            if daily_rsi < 25 and weekly_rsi < 30:
                return True, f"标准型买入: 日RSI={daily_rsi:.2f}<25, 周RSI={weekly_rsi:.2f}<30"
            return False, ""

        # buy_id = 3: 指数保护型买入（需要额外数据，这里暂不实现）
        elif buy_id == 3:
            return False, "指数保护型买入需要额外数据"

        # buy_id = 4: 极度保守型买入 - 日线RSI < 20 且 周线RSI < 20
        elif buy_id == 4:
            if daily_rsi < 20 and weekly_rsi < 20:
                return True, f"极度保守型买入: 日RSI={daily_rsi:.2f}<20, 周RSI={weekly_rsi:.2f}<20"
            return False, ""

        # buy_id = 5: 指数保护型买入（需要额外数据，这里暂不实现）
        elif buy_id == 5:
            return False, "指数保护型买入需要额外数据"

        return False, f"未知的买入策略ID: {buy_id}"

    def judge_sell(
        self,
        daily_rsi: float,
        weekly_rsi: float,
        sell_id: int,
        is_top_divergence: int = 0,
        is_local_high: int = 0,
        ma5_ma10_dead_cross: int = 0
    ) -> Tuple[bool, str]:
        """
        判断卖出条件

        Args:
            daily_rsi: 日线RSI
            weekly_rsi: 周线RSI
            sell_id: 卖出策略ID
            is_top_divergence: 是否顶背离
            is_local_high: 是否局部高点
            ma5_ma10_dead_cross: 是否均线死叉

        Returns:
            Tuple[bool, str]: (是否卖出, 原因)
        """
        if daily_rsi is None:
            daily_rsi = 50.0
        if weekly_rsi is None:
            weekly_rsi = 50.0

        # sell_id = 1: 标准卖出策略
        if sell_id == 1:
            # 条件1: 日RSI > 90（清仓）
            if daily_rsi > 90:
                return True, f"清仓信号: 日RSI={daily_rsi:.2f}>90"

            # 条件2: 日RSI > 80 且 周RSI > 70（警戒）
            if daily_rsi > 80 and weekly_rsi > 70:
                return True, f"二级警戒卖出: 日RSI={daily_rsi:.2f}>80, 周RSI={weekly_rsi:.2f}>70"

            # 条件3: 日RSI > 70 且 周RSI > 60（一级警戒）
            if daily_rsi > 70 and weekly_rsi > 60:
                return True, f"一级警戒卖出: 日RSI={daily_rsi:.2f}>70, 周RSI={weekly_rsi:.2f}>60"

            # 条件4: 顶背离
            if is_top_divergence == 1:
                return True, f"顶背离卖出信号"

            # 条件5: 局部高点 + 均线死叉
            if is_local_high == 1 and ma5_ma10_dead_cross == 1:
                return True, f"局部高点+均线死叉卖出"

            return False, ""

        return False, f"未知的卖出策略ID: {sell_id}"


# ==================== 数据库回测执行器 ====================

class DBBacktestRunner:
    """
    数据库回测执行器

    【功能说明】
    从MySQL数据库读取指标数据，执行回测。
    """

    # CSV列顺序
    CSV_COLUMNS = [
        '股票代码', '策略收益率(%)', '基准收益率(%)', '超额收益率(%)', '最大回撤率(%)',
        '买入策略ID', '做T策略ID', '卖出策略ID', '股票名称', '开始日期', '结束日期',
        '初始资金', '交易次数', '买入次数', '卖出次数', '最终市值', '是否成功',
        '错误信息', '日志文件', '图表文件'
    ]

    def __init__(self, config_path: str = None, max_workers: int = 4) -> None:
        """
        初始化回测执行器

        Args:
            config_path: CSV配置文件路径
            max_workers: 最大并行线程数
        """
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'stocks_backtest.csv')

        self.config_path = config_path
        self.max_workers = max_workers

        # 数据库连接
        self.db = MySQLConnector()
        self.db.connect()

        # 判断器
        self.judger = DBJudger()

        # 结果列表
        self.results: List[BacktestResult] = []

        # 创建输出目录
        logs_base = os.path.join(project_root, 'logs')
        if not os.path.exists(logs_base):
            os.makedirs(logs_base)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(logs_base, f'db_backtest_{timestamp}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 汇总CSV路径
        self.summary_csv_path = os.path.join(self.output_dir, 'backtest_summary.csv')

        # 线程锁
        self._csv_lock = threading.Lock()
        self._completed_count = 0
        self._total_count = 0

        # 配置日志
        self._setup_logging()

        # 初始化CSV
        self._init_summary_csv()

    def _setup_logging(self) -> None:
        """配置日志"""
        log_file = os.path.join(self.output_dir, 'db_backtest.log')

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

    def _init_summary_csv(self) -> None:
        """初始化CSV文件"""
        with open(self.summary_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_COLUMNS)
        self.logger.info(f"汇总CSV已创建: {self.summary_csv_path}")

    def _append_result_to_csv(self, result: BacktestResult) -> None:
        """追加结果到CSV"""
        config = result.config
        row_data = [
            result.symbol,
            round(result.total_return, 2),
            round(result.benchmark_return, 2),
            round(result.excess_return, 2),
            round(result.max_drawdown, 2),
            ';'.join(map(str, config.judge_buy_ids)),
            ';'.join(map(str, config.judge_t_ids)),
            ';'.join(map(str, config.judge_sell_ids)),
            result.stock_name,
            config.start_date,
            config.end_date,
            config.initial_capital,
            result.trades,
            result.buy_count,
            result.sell_count,
            round(result.final_value, 2),
            '是' if result.success else '否',
            result.error_message,
            result.log_file,
            result.chart_file
        ]

        with self._csv_lock:
            with open(self.summary_csv_path, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            self._completed_count += 1
            self.logger.info(f"[{self._completed_count}/{self._total_count}] 结果已写入: {result.symbol}")

    def load_configs(self) -> List[BacktestConfig]:
        """从CSV加载配置"""
        configs = []

        if not os.path.exists(self.config_path):
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return configs

        with open(self.config_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    active = row.get('active', '1').strip()
                    if active != '1':
                        continue

                    symbol = _add_exchange_prefix(row['symbol'].strip())
                    start_date = row['start_date'].strip()
                    end_date = row['end_date'].strip()

                    def parse_ids(value: str) -> List[int]:
                        if not value or value.strip() == '':
                            return []
                        return [int(x.strip()) for x in value.split(';') if x.strip()]

                    judge_buy_ids = parse_ids(row.get('judge_buy_ids', ''))
                    judge_t_ids = parse_ids(row.get('judge_t_ids', ''))
                    judge_sell_ids = parse_ids(row.get('judge_sell_ids', '1'))
                    initial_capital = float(row.get('initial_capital', '100000'))

                    config = BacktestConfig(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        judge_buy_ids=judge_buy_ids,
                        judge_t_ids=judge_t_ids,
                        judge_sell_ids=judge_sell_ids,
                        initial_capital=initial_capital
                    )
                    configs.append(config)
                    self.logger.info(f"加载配置: {symbol}")

                except Exception as e:
                    self.logger.error(f"解析配置失败: {row}, 错误: {e}")

        return configs

    def run_single_backtest(self, config: BacktestConfig) -> BacktestResult:
        """执行单个股票回测"""
        self.logger.info(f"开始回测: {config.symbol}")

        try:
            # 获取股票ID和名称
            stock_id = self.db.get_stock_id(config.symbol)
            if stock_id is None:
                return BacktestResult(
                    symbol=config.symbol,
                    config=config,
                    success=False,
                    error_message=f"股票不存在: {config.symbol}"
                )

            stock_name = self.db.get_stock_name(config.symbol)

            # 从数据库获取指标数据
            df = self.db.get_daily_metrics(stock_id, config.start_date, config.end_date)

            if df.empty:
                return BacktestResult(
                    symbol=config.symbol,
                    config=config,
                    stock_name=stock_name,
                    success=False,
                    error_message="无数据"
                )

            self.logger.info(f"获取到 {len(df)} 条数据")

            # 执行回测逻辑
            result = self._execute_backtest(config, df, stock_name)

            # 写入CSV
            self._append_result_to_csv(result)

            return result

        except Exception as e:
            self.logger.error(f"回测失败: {config.symbol}, 错误: {e}")
            result = BacktestResult(
                symbol=config.symbol,
                config=config,
                success=False,
                error_message=str(e)
            )
            self._append_result_to_csv(result)
            return result

    def _execute_backtest(
        self,
        config: BacktestConfig,
        df: pd.DataFrame,
        stock_name: str
    ) -> BacktestResult:
        """
        执行回测逻辑

        Args:
            config: 回测配置
            df: 指标数据DataFrame
            stock_name: 股票名称

        Returns:
            BacktestResult: 回测结果
        """
        # 初始化状态
        cash = config.initial_capital
        shares = 0.0
        sold_cash = 0.0  # 做T卖出后的待买回资金

        trades: List[TradeRecord] = []

        # 遍历每一天
        for i, row in df.iterrows():
            date = row['trade_date'].strftime('%Y-%m-%d')
            close_price = float(row['close_price'])
            daily_rsi = float(row['rsi_daily']) if row['rsi_daily'] is not None else 50.0
            weekly_rsi = float(row['rsi_weekly']) if row['rsi_weekly'] is not None else 50.0
            is_top_div = int(row['is_top_divergence']) if row['is_top_divergence'] is not None else 0
            is_local_high = int(row['is_local_high']) if row['is_local_high'] is not None else 0
            is_local_low = int(row['is_local_low']) if row['is_local_low'] is not None else 0
            ma5_ma10_dead = int(row['ma5_ma10_dead_cross']) if row['ma5_ma10_dead_cross'] is not None else 0

            # ===== 买入判断 =====
            # 1. 新资金买入
            if cash > 0 and len(config.judge_buy_ids) > 0:
                for buy_id in config.judge_buy_ids:
                    should_buy, reason = self.judger.judge_buy(
                        daily_rsi, weekly_rsi, buy_id, is_local_low
                    )
                    if should_buy:
                        buy_shares = cash / close_price
                        shares += buy_shares
                        cash = 0.0
                        trades.append(TradeRecord(
                            date=date, action='buy', price=close_price,
                            amount=buy_shares, daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi, reason=reason
                        ))
                        break

            # 2. 做T买回
            if sold_cash > 0 and len(config.judge_t_ids) > 0:
                for buy_id in config.judge_t_ids:
                    should_buy, reason = self.judger.judge_buy(
                        daily_rsi, weekly_rsi, buy_id, is_local_low
                    )
                    if should_buy:
                        buy_shares = sold_cash / close_price
                        shares += buy_shares
                        sold_cash = 0.0
                        trades.append(TradeRecord(
                            date=date, action='buy', price=close_price,
                            amount=buy_shares, daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi, reason=f"做T买回: {reason}"
                        ))
                        break

            # ===== 卖出判断 =====
            if shares > 0 and len(config.judge_sell_ids) > 0:
                for sell_id in config.judge_sell_ids:
                    should_sell, reason = self.judger.judge_sell(
                        daily_rsi, weekly_rsi, sell_id,
                        is_top_div, is_local_high, ma5_ma10_dead
                    )
                    if should_sell:
                        # 区分做T卖出和清仓卖出
                        # 如果周RSI > 70，认为是清仓；否则是做T
                        if weekly_rsi > 70:
                            # 清仓卖出
                            sell_value = shares * close_price
                            cash += sell_value
                            shares = 0.0
                            trades.append(TradeRecord(
                                date=date, action='sell', price=close_price,
                                amount=shares, daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi, reason=f"清仓: {reason}"
                            ))
                        else:
                            # 做T卖出（保留待买回）
                            sell_value = shares * close_price
                            sold_cash += sell_value
                            shares = 0.0
                            trades.append(TradeRecord(
                                date=date, action='sell', price=close_price,
                                amount=shares, daily_rsi=daily_rsi,
                                weekly_rsi=weekly_rsi, reason=f"做T卖出: {reason}"
                            ))
                        break

        # 计算最终市值
        last_price = float(df.iloc[-1]['close_price'])
        final_value = shares * last_price + cash + sold_cash

        # 计算收益率
        total_return = (final_value - config.initial_capital) / config.initial_capital * 100

        # 计算基准收益率（买入持有）
        first_price = float(df.iloc[0]['close_price'])
        benchmark_shares = config.initial_capital / first_price
        benchmark_value = benchmark_shares * last_price
        benchmark_return = (benchmark_value - config.initial_capital) / config.initial_capital * 100

        excess_return = total_return - benchmark_return

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(df, trades, config.initial_capital)

        # 统计交易次数
        buy_count = sum(1 for t in trades if t.action == 'buy')
        sell_count = sum(1 for t in trades if t.action == 'sell')

        return BacktestResult(
            symbol=config.symbol,
            config=config,
            stock_name=stock_name,
            total_return=total_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            max_drawdown=max_drawdown,
            trades=len(trades),
            buy_count=buy_count,
            sell_count=sell_count,
            final_value=final_value,
            success=True
        )

    def _calculate_max_drawdown(
        self,
        df: pd.DataFrame,
        trades: List[TradeRecord],
        initial_capital: float
    ) -> float:
        """计算最大回撤"""
        # 简化计算：基于收盘价的回撤
        prices = df['close_price'].values
        peak = prices[0]
        max_dd = 0.0

        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def run_batch_backtest(self) -> List[BacktestResult]:
        """批量执行回测"""
        configs = self.load_configs()

        if len(configs) == 0:
            self.logger.error("未加载到配置")
            return []

        self._total_count = len(configs)

        self.logger.info("=" * 60)
        self.logger.info("开始数据库回测")
        self.logger.info("=" * 60)
        self.logger.info(f"配置文件: {self.config_path}")
        self.logger.info(f"股票数量: {len(configs)}")
        self.logger.info(f"并行线程: {self.max_workers}")
        self.logger.info("=" * 60)

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_config = {
                executor.submit(self.run_single_backtest, config): config
                for config in configs
            }

            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"任务异常: {config.symbol}, 错误: {e}")

        self.results = results

        # 输出汇总统计
        self._print_final_summary()

        return results

    def _print_final_summary(self) -> None:
        """输出汇总统计"""
        if not self.results:
            return

        success_count = sum(1 for r in self.results if r.success)
        avg_return = sum(r.total_return for r in self.results if r.success) / success_count if success_count else 0
        avg_excess = sum(r.excess_return for r in self.results if r.success) / success_count if success_count else 0

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("回测完成")
        self.logger.info("=" * 60)
        self.logger.info(f"成功: {success_count}/{len(self.results)}")
        self.logger.info(f"平均收益率: {avg_return:.2f}%")
        self.logger.info(f"平均超额收益: {avg_excess:.2f}%")
        self.logger.info(f"汇总CSV: {self.summary_csv_path}")
        self.logger.info("=" * 60)

    def close(self) -> None:
        """关闭资源"""
        self.db.disconnect()


# ==================== 主函数 ====================

if __name__ == "__main__":
    # ===== 配置参数 =====
    CONFIG_PATH = None           # 使用默认路径
    MAX_WORKERS = 20             # 最大并行线程数

    # ===== 执行回测 =====
    runner = DBBacktestRunner(
        config_path=CONFIG_PATH,
        max_workers=MAX_WORKERS
    )

    results = runner.run_batch_backtest()

    # ===== 关闭 =====
    runner.close()