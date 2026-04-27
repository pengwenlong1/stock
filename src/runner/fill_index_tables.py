# -*- coding: utf-8 -*-
"""
指数数据填充脚本 (fill_index_tables.py)

【功能说明】
从 stocks_index_infor.csv 加载指数列表，将指数的技术指标数据写入 stock_index_daily_metrics 表。

【使用方法】
python src/runner/fill_index_tables.py

作者：量化交易团队
创建日期：2026-04-26
"""
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pymysql
import yaml
from pymysql.cursors import DictCursor

# 添加项目路径
project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 导入工具类
from src.tool.rsi_calculator import RSI
from src.tool.sar_strategy import SARStrategy, SARSignal
from src.tool.macd_calculator import MACDCalculator
from src.tool.ma_calculator import MACalculator
from src.tool.peak_detector import PeakDetector
from src.tool.divergence_detector import DivergenceDetector

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
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class DBConfig:
    """数据库配置"""
    host: str
    port: int
    user: str
    password: str
    database: str


# ==================== 配置加载 ====================

def load_db_config() -> DBConfig:
    """从 .env/app_config.yaml 加载数据库配置"""
    app_config_path = project_root / " .env" / "app_config.yaml"

    if not app_config_path.exists():
        logger.error(f"配置文件不存在: {app_config_path}")
        raise FileNotFoundError("缺少app_config.yaml配置文件")

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


def load_gm_token() -> Optional[str]:
    """加载掘金Token"""
    settings_path = project_root / "config" / "settings.json"
    with settings_path.open('r', encoding='utf-8') as f:
        config = json.load(f)
        return config.get('gm_token')


def load_index_list() -> List[Dict]:
    """从 stocks_index_infor.csv 加载指数列表"""
    index_csv_path = project_root / "config" / "stocks_index_infor.csv"

    if not index_csv_path.exists():
        logger.warning(f"指数列表文件不存在: {index_csv_path}")
        return []

    # 使用 dtype 参数确保指数代码作为字符串读取，不丢失前导0
    df = pd.read_csv(str(index_csv_path), dtype={'symbol': str})

    index_list = []
    for row in df.to_dict('records'):
        index_list.append({
            'ticker': str(row['symbol']),
            'start_date': str(row['start_date']),
            'end_date': str(row['end_date'])
        })

    return index_list


# ==================== 指数数据填充类 ====================

class IndexDataFiller:
    """指数数据填充器"""

    def __init__(self):
        self._db_config = load_db_config()
        self._gm_token = load_gm_token()
        self._connection = None

        # 设置掘金Token
        if self._gm_token and set_token:
            set_token(self._gm_token)
            logger.info("掘金Token设置成功")

    def connect_db(self) -> bool:
        """连接数据库"""
        try:
            self._connection = pymysql.connect(
                host=self._db_config.host,
                port=self._db_config.port,
                user=self._db_config.user,
                password=self._db_config.password,
                database=self._db_config.database,
                charset='utf8mb4',
                cursorclass=DictCursor
            )
            logger.info(f"数据库连接成功: {self._db_config.host}:{self._db_config.port}")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False

    def close_db(self):
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            logger.info("数据库连接已关闭")

    def _execute_batch(self, sql: str, params_list: List) -> int:
        """批量执行SQL"""
        if not params_list:
            return 0

        try:
            with self._connection.cursor() as cursor:
                cursor.executemany(sql, params_list)
            self._connection.commit()
            return len(params_list)
        except Exception as e:
            logger.error(f"批量执行SQL失败: {e}")
            self._connection.rollback()
            return 0

    def _add_index_exchange_prefix(self, code: str) -> str:
        """添加指数交易所前缀"""
        code = code.strip()
        if code.startswith('SHSE.') or code.startswith('SZSE.') or code.startswith('BJSE.'):
            return code

        # 北交所指数（北证50等）
        if code.startswith('899'):
            return f'BJSE.{code}'
        # 深证指数（创业板指、深证成指、国证2000等）
        elif code.startswith('399'):
            return f'SZSE.{code}'
        # 科创50指数
        elif code == '000688':
            return f'SHSE.{code}'
        # 上证指数、沪深300等（000xxx开头）
        elif code.startswith('000'):
            return f'SHSE.{code}'
        else:
            return f'SZSE.{code}'

    def _get_index_db_id(self, ticker: str) -> str:
        """获取指数的数据库ID（字符串格式，保留前导零）"""
        clean_ticker = ticker.replace('SHSE.', '').replace('SZSE.', '').replace('BJSE.', '')
        return clean_ticker

    def fill_index_daily_metrics(self, index_list: List[Dict]) -> Dict[str, int]:
        """填充 stock_index_daily_metrics 表"""
        logger.info("=" * 60)
        logger.info("填充stock_index_daily_metrics表（指数每日技术指标数据）")
        logger.info("=" * 60)

        results = {}

        for index in index_list:
            ticker = index.get('ticker', '')
            start_date = index.get('start_date', '2020-01-01')
            end_date = index.get('end_date', '2026-04-10')

            logger.info(f"\n处理指数: {ticker} | 时间段: {start_date} ~ {end_date}")

            try:
                rowcount = self._process_single_index(ticker, start_date, end_date)
                results[ticker] = rowcount
                logger.info(f"指数 {ticker} 处理完成，插入 {rowcount} 条记录")
            except Exception as e:
                logger.error(f"指数 {ticker} 处理失败: {e}")
                results[ticker] = 0

        total = sum(results.values())
        logger.info(f"\n所有指数处理完成，总计插入 {total} 条记录")
        return results

    def _process_single_index(self, ticker: str, start_date: str, end_date: str) -> int:
        """处理单个指数的指标计算和数据插入"""
        if history is None:
            logger.warning("掘金SDK未安装，无法获取历史数据")
            return 0

        # 添加交易所前缀
        full_symbol = self._add_index_exchange_prefix(ticker)

        logger.info(f"获取历史数据: {full_symbol}, {start_date} ~ {end_date}")

        # 获取日线数据
        daily_data = history(
            symbol=full_symbol,
            frequency='1d',
            start_time=start_date + ' 09:00:00',
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
        close_series = pd.Series(daily_data['close'].values, index=daily_data['eob'], name='Close')
        high_series = pd.Series(daily_data['high'].values, index=daily_data['eob'], name='High')
        low_series = pd.Series(daily_data['low'].values, index=daily_data['eob'], name='Low')

        # ========== 计算指标 ==========
        # MA5/MA10
        ma_calculator = MACalculator(fast_period=5, slow_period=10, ma_type='SMA')
        ma5_series, ma10_series = ma_calculator.calculate(close_series)
        ma_dead_cross_series = ma_calculator.detect_all_cross_downs()

        # RSI
        rsi_daily_calculator = RSI(period=6, freq='daily')
        rsi_daily_series = rsi_daily_calculator.calculate(close_series)
        rsi_weekly_calculator = RSI(period=6, freq='weekly')
        rsi_weekly_series = rsi_weekly_calculator.calculate(close_series)

        # MACD
        macd_calculator = MACDCalculator(fast_period=12, slow_period=26, signal_period=9)
        macd_calculator.prepare_data(close_series)
        macd_series = macd_calculator._macd_series

        # SAR
        sar_strategy = SARStrategy(acceleration=0.02, maximum=0.20)
        try:
            sar_strategy.prepare_data(high_series, low_series, close_series)
            sar_series = sar_strategy._sar_series

            # 检测SAR死叉（绿转红）
            sar_signals = sar_strategy.detect_signals(start_date, end_date)
            # 使用日期字符串格式存储，避免时区匹配问题
            sar_dead_cross_dates = set()
            for sig in sar_signals:
                date_str = sig.date.strftime('%Y-%m-%d')
                if sig.signal_type == '绿转红':
                    sar_dead_cross_dates.add(date_str)
        except ImportError:
            logger.warning("TA-Lib未安装，跳过SAR计算")
            sar_series = pd.Series([np.nan] * len(close_series), index=close_series.index)
            sar_dead_cross_dates = set()

        # 局部高低点
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

        # 背离
        divergence_detector = DivergenceDetector()
        try:
            divergence_detector.prepare_data(full_symbol, start_date, end_date)
            divergences = divergence_detector.detect_all_divergences()
            # 使用日期字符串格式存储，避免时区匹配问题
            daily_top_div_dates = set(d.confirmation_date.strftime('%Y-%m-%d') for d in divergences.get('daily_top_confirmed', []) if d.confirmation_date is not None)
            daily_bottom_div_dates = set(d.confirmation_date.strftime('%Y-%m-%d') for d in divergences.get('daily_bottom_confirmed', []) if d.confirmation_date is not None)
            weekly_top_div_dates = set(d.confirmation_date.strftime('%Y-%m-%d') for d in divergences.get('weekly_top_confirmed', []) if d.confirmation_date is not None)
            weekly_bottom_div_dates = set(d.confirmation_date.strftime('%Y-%m-%d') for d in divergences.get('weekly_bottom_confirmed', []) if d.confirmation_date is not None)
        except Exception as e:
            logger.warning(f"背离检测失败: {e}")
            daily_top_div_dates = set()
            daily_bottom_div_dates = set()
            weekly_top_div_dates = set()
            weekly_bottom_div_dates = set()

        # 篩选目标时间段数据
        start_ts = pd.Timestamp(start_date).tz_localize('Asia/Shanghai')
        end_ts = pd.Timestamp(end_date).tz_localize('Asia/Shanghai')
        filtered_dates = close_series.index[(close_series.index >= start_ts) & (close_series.index <= end_ts)]

        if len(filtered_dates) == 0:
            logger.warning(f"筛选后数据为空: {start_date} ~ {end_date}")
            return 0

        # 构建插入数据
        sql = """
            INSERT INTO stock_index_daily_metrics
            (stock_id, trade_date, close_price, ma5, ma10, ma5_ma10_dead_cross,
             rsi_daily, rsi_weekly, macd, sar, sar_line,
             is_daily_top_divergence, is_daily_bottom_divergergence,
             is_weekly_top_divergence, is_weekly_bottom_divergergence,
             is_sar_dead_cross, is_local_high, is_local_low, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                close_price = VALUES(close_price),
                ma5 = VALUES(ma5),
                ma10 = VALUES(ma10),
                ma5_ma10_dead_cross = VALUES(ma5_ma10_dead_cross),
                rsi_daily = VALUES(rsi_daily),
                rsi_weekly = VALUES(rsi_weekly),
                macd = VALUES(macd),
                sar = VALUES(sar),
                sar_line = VALUES(sar_line),
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
            close_price = close_series.loc[date]

            ma5_val = ma5_series.loc[date] if date in ma5_series.index else None
            ma10_val = ma10_series.loc[date] if date in ma10_series.index else None
            ma_dead_cross = 1 if date in ma_dead_cross_series.index and ma_dead_cross_series.loc[date] else 0

            rsi_daily_val = rsi_daily_series.loc[date] if date in rsi_daily_series.index else None
            rsi_weekly_val = rsi_weekly_series.loc[date] if date in rsi_weekly_series.index else None

            macd_val = macd_series.loc[date] if date in macd_series.index else None
            sar_val = sar_series.loc[date] if date in sar_series.index else None

            sar_dead_cross = 1 if date_str in sar_dead_cross_dates else 0
            is_local_high = 1 if date_str in local_high_dates else 0
            is_local_low = 1 if date_str in local_low_dates else 0

            is_daily_top_div = 1 if date_str in daily_top_div_dates else 0
            is_daily_bottom_div = 1 if date_str in daily_bottom_div_dates else 0
            is_weekly_top_div = 1 if date_str in weekly_top_div_dates else 0
            is_weekly_bottom_div = 1 if date_str in weekly_bottom_div_dates else 0

            # 处理NaN值
            ma5_val = None if pd.isna(ma5_val) else round(float(ma5_val), 2)
            ma10_val = None if pd.isna(ma10_val) else round(float(ma10_val), 2)
            rsi_daily_val = None if pd.isna(rsi_daily_val) else round(float(rsi_daily_val), 2)
            rsi_weekly_val = None if pd.isna(rsi_weekly_val) else round(float(rsi_weekly_val), 2)
            macd_val = None if pd.isna(macd_val) else round(float(macd_val), 4)
            sar_val = None if pd.isna(sar_val) else round(float(sar_val), 4)

            params = (
                self._get_index_db_id(ticker),
                date_str,
                round(float(close_price), 2),
                ma5_val,
                ma10_val,
                ma_dead_cross,
                rsi_daily_val,
                rsi_weekly_val,
                macd_val,
                sar_val,
                sar_val,  # sar_line
                is_daily_top_div,
                is_daily_bottom_div,
                is_weekly_top_div,
                is_weekly_bottom_div,
                sar_dead_cross,
                is_local_high,
                is_local_low,
                0
            )
            params_list.append(params)

        rowcount = self._execute_batch(sql, params_list)
        logger.info(f"指数 {ticker} 指标计算完成，准备插入 {len(params_list)} 条记录，实际插入 {rowcount} 条")

        return rowcount


# ==================== 主函数 ====================

def run():
    """主运行函数"""
    # 配置日志文件
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"fill_index_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("指数数据填充脚本启动")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)

    # 加载指数列表
    index_list = load_index_list()

    if not index_list:
        logger.error("指数列表为空，退出程序")
        return

    logger.info(f"加载到 {len(index_list)} 只指数")

    # 创建数据填充器
    filler = IndexDataFiller()

    # 连接数据库
    if not filler.connect_db():
        logger.error("数据库连接失败，退出程序")
        return

    try:
        # 填充指数数据
        results = filler.fill_index_daily_metrics(index_list)

        # 输出统计结果
        logger.info("\n" + "=" * 60)
        logger.info("数据填充统计")
        logger.info("=" * 60)
        for ticker, count in results.items():
            logger.info(f"  {ticker}: {count} 条")

    except Exception as e:
        logger.error(f"数据填充过程出错: {e}")
        raise

    finally:
        filler.close_db()

    logger.info("=" * 60)
    logger.info("脚本执行完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()