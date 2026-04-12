# -*- coding: utf-8 -*-
"""
批量回测执行器模块 (batch_backtest_runner.py)

【功能说明】
从 config/stocks_backtest.csv 读取配置，并行执行多个股票的回测。
使用线程池提高效率，生成汇总CSV报告。

【CSV配置格式】
symbol,start_date,end_date,judge_buy_ids,judge_t_ids,judge_sell_ids,initial_capital
SHSE.512480,2024-01-01,2026-04-10,1;5,,1,100000

使用方法：
    python src/runner/batch_backtest_runner.py

作者：量化交易团队
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免线程问题

import csv
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.runner.backtest_runner import BacktestRunner


# ==================== 数据结构 ====================

@dataclass
class BacktestConfig:
    """单个股票的回测配置"""
    symbol: str
    start_date: str
    end_date: str
    judge_buy_ids: List[int]
    judge_t_ids: List[int]
    judge_sell_ids: List[int]
    initial_capital: float

    @classmethod
    def from_csv_row(cls, row: Dict) -> 'BacktestConfig':
        """从CSV行创建配置"""
        def parse_ids(value: str) -> List[int]:
            if not value or value.strip() == '':
                return []
            return [int(x.strip()) for x in value.split(';') if x.strip()]

        # 自动添加交易所前缀
        symbol = row['symbol'].strip()
        if not symbol.startswith('SHSE.') and not symbol.startswith('SZSE.'):
            # 根据股票代码判断交易所
            # 6开头: 上海主板 (SHSE.600xxx, SHSE.601xxx, SHSE.603xxx)
            # 0开头: 深圳主板 (SZSE.000xxx)
            # 3开头: 创业板 (SZSE.300xxx, SZSE.159xxx基金, SZSE.399xxx指数)
            # 5开头: 上海基金/ETF (SHSE.510xxx, SHSE.51xxxx)
            # 1开头: 深圳基金 (SZSE.159xxx)
            if symbol.startswith('6') or symbol.startswith('5'):
                symbol = f'SHSE.{symbol}'
            elif symbol.startswith('0') or symbol.startswith('3') or symbol.startswith('1'):
                symbol = f'SZSE.{symbol}'
            else:
                # 默认深圳
                symbol = f'SZSE.{symbol}'

        return cls(
            symbol=symbol,
            start_date=row['start_date'].strip(),
            end_date=row['end_date'].strip(),
            judge_buy_ids=parse_ids(row.get('judge_buy_ids', '')),
            judge_t_ids=parse_ids(row.get('judge_t_ids', '')),
            judge_sell_ids=parse_ids(row.get('judge_sell_ids', '1')),
            initial_capital=float(row.get('initial_capital', '100000'))
        )


@dataclass
class BacktestResult:
    """单个股票的回测结果"""
    symbol: str
    config: BacktestConfig
    stock_name: str = ""             # 股票名称
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


# ==================== 批量回测执行器 ====================

class BatchBacktestRunner:
    """
    批量回测执行器

    【功能说明】
    从CSV读取配置，使用线程池并行执行多个股票的回测。
    每个股票完成后立即写入汇总CSV，无需等待全部完成。
    """

    # CSV列顺序（按用户指定）
    CSV_COLUMNS = [
        '股票代码', '策略收益率(%)', '基准收益率(%)', '超额收益率(%)', '最大回撤率(%)',
        '买入策略ID', '做T策略ID', '卖出策略ID', '股票名称', '开始日期', '结束日期',
        '初始资金', '交易次数', '买入次数', '卖出次数', '最终市值', '是否成功',
        '错误信息', '日志文件', '图表文件'
    ]

    def __init__(self, config_path: str = None, max_workers: int = 4) -> None:
        """
        初始化批量回测执行器

        Args:
            config_path: CSV配置文件路径
            max_workers: 最大并行线程数
        """
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'stocks_backtest.csv')

        self.config_path = config_path
        self.max_workers = max_workers
        self.results: List[BacktestResult] = []

        # 创建本次回测的输出目录（按时间到秒）
        logs_base = os.path.join(project_root, 'logs')
        if not os.path.exists(logs_base):
            os.makedirs(logs_base)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(logs_base, f'batch_backtest_{timestamp}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 汇总CSV文件路径
        self.summary_csv_path = os.path.join(self.output_dir, 'backtest_summary.csv')

        # 线程锁（保护并发写入CSV）
        self._csv_lock = threading.Lock()

        # 已完成计数
        self._completed_count = 0
        self._total_count = 0

        # 设置日志
        self._setup_logging()

        # 初始化CSV文件（写入表头）
        self._init_summary_csv()

    def _setup_logging(self) -> None:
        """配置主日志"""
        log_file = os.path.join(self.output_dir, 'batch_backtest.log')

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

    def _init_summary_csv(self) -> None:
        """初始化汇总CSV文件（写入表头）"""
        with open(self.summary_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_COLUMNS)
        self.logger.info(f"汇总CSV文件已创建: {self.summary_csv_path}")

    def _append_result_to_csv(self, result: BacktestResult) -> None:
        """
        将单个回测结果追加写入CSV（线程安全）

        Args:
            result: 回测结果
        """
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

        # 使用线程锁保护写入操作
        with self._csv_lock:
            with open(self.summary_csv_path, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

            # 更新完成计数
            self._completed_count += 1
            progress = f"[{self._completed_count}/{self._total_count}]"
            self.logger.info(f"{progress} 结果已写入CSV: {result.symbol} ({result.stock_name})")

    def load_configs(self) -> List[BacktestConfig]:
        """
        从CSV文件加载回测配置

        Returns:
            回测配置列表
        """
        configs = []

        if not os.path.exists(self.config_path):
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return configs

        with open(self.config_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # 检查 active 字段，只有值为1才参与测试
                    active = row.get('active', '1').strip()
                    if active != '1':
                        self.logger.info(f"跳过配置: {row.get('symbol', '')} (active={active})")
                        continue

                    config = BacktestConfig.from_csv_row(row)
                    configs.append(config)
                    self.logger.info(f"加载配置: {config.symbol} | "
                                   f"时间: {config.start_date} ~ {config.end_date} | "
                                   f"buy_ids={config.judge_buy_ids}, t_ids={config.judge_t_ids}, sell_ids={config.judge_sell_ids}")
                except Exception as e:
                    self.logger.error(f"解析配置行失败: {row}, 错误: {e}")

        return configs

    def run_single_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        执行单个股票的回测

        Args:
            config: 回测配置

        Returns:
            回测结果
        """
        self.logger.info(f"开始回测: {config.symbol}")

        try:
            runner = BacktestRunner(
                symbol=config.symbol,
                start_date=config.start_date,
                end_date=config.end_date,
                judge_buy_ids=config.judge_buy_ids,
                judge_t_ids=config.judge_t_ids,
                judge_sell_ids=config.judge_sell_ids,
                initial_capital=config.initial_capital,
                output_dir=self.output_dir  # 使用统一的输出目录
            )

            stock_name = runner.stock_name  # 获取股票名称

            # 运行回测
            results = runner.run_backtest()

            if 'error' in results:
                result = BacktestResult(
                    symbol=config.symbol,
                    stock_name=stock_name,
                    config=config,
                    total_return=0.0,
                    benchmark_return=0.0,
                    excess_return=0.0,
                    max_drawdown=0.0,
                    trades=0,
                    buy_count=0,
                    sell_count=0,
                    final_value=0.0,
                    log_file=runner.log_file,
                    chart_file="",
                    success=False,
                    error_message=results.get('error', '未知错误')
                )
                # 立即写入CSV
                self._append_result_to_csv(result)
                return result

            # 生成报告和图表
            runner.generate_report()
            chart_file = runner.plot_backtest_results()

            # 计算最终市值
            final_value = results.get('final_shares', 0) * runner._close_series.iloc[-1] + \
                          results.get('final_cash', 0) + results.get('final_sold_cash', 0)

            result = BacktestResult(
                symbol=config.symbol,
                stock_name=stock_name,
                config=config,
                total_return=results.get('total_return', 0.0),
                benchmark_return=results.get('benchmark_return', 0.0),
                excess_return=results.get('excess_return', 0.0),
                max_drawdown=results.get('max_drawdown', 0.0),
                trades=results.get('trades', 0),
                buy_count=results.get('buy_count', 0),
                sell_count=results.get('sell_count', 0),
                final_value=final_value,
                log_file=runner.log_file,
                chart_file=chart_file or "",
                success=True
            )

            self.logger.info(f"回测完成: {config.symbol} ({stock_name}) | "
                           f"收益率={result.total_return:.2f}% | "
                           f"超额={result.excess_return:.2f}% | "
                           f"回撤={result.max_drawdown:.2f}%")

            # 立即写入CSV
            self._append_result_to_csv(result)
            return result

        except Exception as e:
            self.logger.error(f"回测失败: {config.symbol}, 错误: {e}")
            result = BacktestResult(
                symbol=config.symbol,
                config=config,
                total_return=0.0,
                benchmark_return=0.0,
                excess_return=0.0,
                max_drawdown=0.0,
                trades=0,
                buy_count=0,
                sell_count=0,
                final_value=0.0,
                log_file="",
                chart_file="",
                success=False,
                error_message=str(e)
            )
            # 立即写入CSV（即使失败）
            self._append_result_to_csv(result)
            return result

    def run_batch_backtest(self) -> List[BacktestResult]:
        """
        使用线程池并行执行批量回测

        Returns:
            回测结果列表
        """
        configs = self.load_configs()

        if len(configs) == 0:
            self.logger.error("没有加载到任何回测配置")
            return []

        # 设置总任务数（用于进度显示）
        self._total_count = len(configs)

        self.logger.info("=" * 60)
        self.logger.info("开始批量回测")
        self.logger.info("=" * 60)
        self.logger.info(f"配置文件: {self.config_path}")
        self.logger.info(f"股票数量: {len(configs)}")
        self.logger.info(f"并行线程: {self.max_workers}")
        self.logger.info(f"汇总CSV: {self.summary_csv_path}")
        self.logger.info("=" * 60)

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_backtest, config): config
                for config in configs
            }

            # 收集结果
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"任务执行异常: {config.symbol}, 错误: {e}")
                    result = BacktestResult(
                        symbol=config.symbol,
                        config=config,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(result)
                    # 立即写入CSV（异常情况）
                    self._append_result_to_csv(result)

        self.results = results

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("批量回测完成")
        self.logger.info("=" * 60)

        # 输出最终汇总统计
        self._print_final_summary()

        return results

    def _print_final_summary(self) -> None:
        """输出最终汇总统计信息"""
        if len(self.results) == 0:
            return

        success_count = sum(1 for r in self.results if r.success)
        avg_return = sum(r.total_return for r in self.results if r.success) / success_count if success_count > 0 else 0
        avg_excess = sum(r.excess_return for r in self.results if r.success) / success_count if success_count > 0 else 0
        avg_drawdown = sum(r.max_drawdown for r in self.results if r.success) / success_count if success_count > 0 else 0

        self.logger.info("")
        self.logger.info("【汇总统计】")
        self.logger.info("-" * 40)
        self.logger.info(f"回测成功: {success_count}/{len(self.results)}")
        self.logger.info(f"平均收益率: {avg_return:.2f}%")
        self.logger.info(f"平均超额收益: {avg_excess:.2f}%")
        self.logger.info(f"平均最大回撤: {avg_drawdown:.2f}%")
        self.logger.info("")
        self.logger.info(f"汇总CSV: {self.summary_csv_path}")
        self.logger.info(f"主日志文件: {self.log_file}")
        self.logger.info("=" * 60)

    # ==================== 主函数 ====================

if __name__ == "__main__":
    # ===== 可配置参数 =====
    CONFIG_PATH = None                   # CSV配置文件路径（None则使用默认路径）
    MAX_WORKERS = 10                      # 最大并行线程数

    # ===== 执行批量回测 =====
    batch_runner = BatchBacktestRunner(
        config_path=CONFIG_PATH,
        max_workers=MAX_WORKERS
    )

    results = batch_runner.run_batch_backtest()