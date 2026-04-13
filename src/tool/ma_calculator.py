# -*- coding: utf-8 -*-
"""
均线计算模块 (ma_calculator.py)

【功能说明】
提供移动平均线（Moving Average）的计算功能，包括：
1. SMA（Simple Moving Average，简单移动平均线）
2. EMA（Exponential Moving Average，指数移动平均线）
3. 均线交叉检测（金叉/死叉判断）

【算法说明】
- SMA: N 日收盘价的简单平均值（rolling mean）
- EMA: 指数加权平均，alpha = 2/(N+1)，对近期价格赋予更大权重

【常用周期】
- MA5：周线，短期趋势参考
- MA10：双周线，短期趋势确认
- MA20：月线，中期趋势参考
- MA60：季线，长期趋势参考

【使用方法】
    from src.tool.ma_calculator import MACalculator

    ma = MACalculator(fast_period=5, slow_period=10)
    ma.calculate(close_series)
    is_cross_down = ma.is_cross_down()  # 检测死叉（5日跌破10日）

作者：量化交易团队
创建日期：2024
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict


class MACalculator:
    """
    移动平均线计算器

    【功能说明】
    计算快慢两条移动平均线，并提供交叉检测功能。
    主要用于判断均线金叉/死叉信号。

    【均线交叉含义】
    - 金叉（快线上穿慢线）：短期趋势转强，买入信号
    - 死叉（快线下穿慢线）：短期趋势转弱，卖出信号

    【注意事项】
    - 必须使用 T-1 及之前的数据计算，避免未来函数
    - 前期数据可能存在 NaN（周期不足时）
    - 交叉信号需结合其他指标综合判断

    Attributes:
        fast_period: 快线周期（默认5）
        slow_period: 慢线周期（默认10）
        ma_type: 均线类型（'SMA' 或 'EMA'）
        fast_ma: 快线计算结果
        slow_ma: 慢线计算结果
    """

    def __init__(self,
                 fast_period: int = 5,
                 slow_period: int = 10,
                 ma_type: str = 'SMA') -> None:
        """
        初始化均线计算器

        Args:
            fast_period: 快线周期，默认 5（5日均线）
            slow_period: 慢线周期，默认 10（10日均线）
            ma_type: 均线类型，'SMA'（简单）或 'EMA'（指数），默认 'SMA'

        Raises:
            ValueError: 当周期参数无效时抛出
        """
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError(f"均线周期必须为正整数，当前: fast={fast_period}, slow={slow_period}")

        if fast_period >= slow_period:
            raise ValueError(f"快线周期必须小于慢线周期，当前: fast={fast_period}, slow={slow_period}")

        if ma_type not in ('SMA', 'EMA'):
            raise ValueError(f"均线类型必须为 'SMA' 或 'EMA'，当前: {ma_type}")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type

        # 存储计算结果
        self._fast_ma: Optional[pd.Series] = None
        self._slow_ma: Optional[pd.Series] = None
        self._close: Optional[pd.Series] = None

    def calculate(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算快慢两条均线

        【计算逻辑】
        SMA: N 日收盘价的简单平均值
        EMA: 指数加权平均，对近期价格赋予更大权重

        Args:
            close: 收盘价序列，需为 pd.Series 类型

        Returns:
            Tuple[pd.Series, pd.Series]: (快线, 慢线)

        Raises:
            ValueError: 当输入数据为空时抛出
            TypeError: 当输入非 pd.Series 类型时抛出
        """
        # 参数校验
        if not isinstance(close, pd.Series):
            raise TypeError(f"close 参数必须为 pd.Series，当前类型: {type(close)}")

        if close.empty:
            raise ValueError("收盘价序列不能为空")

        self._close = close

        # 使用 pandas 计算（纯 Python 实现，不依赖外部 API）
        if self.ma_type == 'SMA':
            # SMA: 简单移动平均
            self._fast_ma = close.rolling(window=self.fast_period, min_periods=self.fast_period).mean()
            self._slow_ma = close.rolling(window=self.slow_period, min_periods=self.slow_period).mean()
        else:  # EMA
            # EMA: 指数移动平均，alpha = 2/(N+1) 为标准 EMA 参数
            alpha_fast = 2.0 / (self.fast_period + 1)
            alpha_slow = 2.0 / (self.slow_period + 1)
            self._fast_ma = close.ewm(alpha=alpha_fast, adjust=False).mean()
            self._slow_ma = close.ewm(alpha=alpha_slow, adjust=False).mean()

        # 设置序列名称
        self._fast_ma.name = f'MA{self.fast_period}'
        self._slow_ma.name = f'MA{self.slow_period}'

        return self._fast_ma, self._slow_ma

    def get_fast_ma(self) -> Optional[pd.Series]:
        """
        获取快线数据

        Returns:
            pd.Series: 快线数据，未计算时返回 None
        """
        return self._fast_ma

    def get_slow_ma(self) -> Optional[pd.Series]:
        """
        获取慢线数据

        Returns:
            pd.Series: 慢线数据，未计算时返回 None
        """
        return self._slow_ma

    def is_cross_down(self, lookback: int = 1) -> bool:
        """
        检测均线死叉（快线跌破慢线）

        【判断逻辑】
        死叉定义：前一日快线 >= 慢线，当日快线 < 慢线

        【参数说明】
        lookback: 向前回溯天数，用于检测最近 N 天内是否发生死叉

        Args:
            lookback: 回溯天数，默认 1（仅检测当天）

        Returns:
            bool: 是否发生死叉

        Raises:
            ValueError: 当未执行 calculate() 时抛出
        """
        if self._fast_ma is None or self._slow_ma is None:
            raise ValueError("请先调用 calculate() 方法计算均线")

        if len(self._fast_ma) < 2:
            return False

        # 检测最近 lookback 天内的死叉
        for i in range(1, min(lookback + 1, len(self._fast_ma))):
            prev_fast = self._fast_ma.iloc[-i - 1]
            prev_slow = self._slow_ma.iloc[-i - 1]
            curr_fast = self._fast_ma.iloc[-i]
            curr_slow = self._slow_ma.iloc[-i]

            # 跳过 NaN 值
            if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
                continue

            # 死叉条件：前一日快线 >= 慢线，当日快线 < 慢线
            if prev_fast >= prev_slow and curr_fast < curr_slow:
                return True

        return False

    def is_cross_up(self, lookback: int = 1) -> bool:
        """
        检测均线金叉（快线上穿慢线）

        【判断逻辑】
        金叉定义：前一日快线 <= 慢线，当日快线 > 慢线

        Args:
            lookback: 回溯天数，默认 1（仅检测当天）

        Returns:
            bool: 是否发生金叉

        Raises:
            ValueError: 当未执行 calculate() 时抛出
        """
        if self._fast_ma is None or self._slow_ma is None:
            raise ValueError("请先调用 calculate() 方法计算均线")

        if len(self._fast_ma) < 2:
            return False

        # 检测最近 lookback 天内的金叉
        for i in range(1, min(lookback + 1, len(self._fast_ma))):
            prev_fast = self._fast_ma.iloc[-i - 1]
            prev_slow = self._slow_ma.iloc[-i - 1]
            curr_fast = self._fast_ma.iloc[-i]
            curr_slow = self._slow_ma.iloc[-i]

            # 跳过 NaN 值
            if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
                continue

            # 金叉条件：前一日快线 <= 慢线，当日快线 > 慢线
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                return True

        return False

    def get_cross_type(self) -> str:
        """
        获取当前交叉状态

        Returns:
            str: '金叉'、'死叉' 或 '无交叉'
        """
        if self.is_cross_up():
            return '金叉'
        elif self.is_cross_down():
            return '死叉'
        else:
            return '无交叉'

    def get_position_status(self) -> str:
        """
        获取快线相对于慢线的位置状态

        Returns:
            str: '上方'、'下方' 或 '数据不足'
        """
        if self._fast_ma is None or self._slow_ma is None:
            return '数据不足'

        if len(self._fast_ma) < 1:
            return '数据不足'

        last_fast = self._fast_ma.iloc[-1]
        last_slow = self._slow_ma.iloc[-1]

        if pd.isna(last_fast) or pd.isna(last_slow):
            return '数据不足'

        if last_fast > last_slow:
            return '上方'
        elif last_fast < last_slow:
            return '下方'
        else:
            return '持平'

    def is_below_slow_ma(self) -> bool:
        """
        判断快线是否在慢线下方（5日均线是否在10日均线下方）

        【使用场景】
        用于判断当前短期趋势是否偏弱。

        Returns:
            bool: 快线是否在慢线下方
        """
        return self.get_position_status() == '下方'

    def is_above_slow_ma(self) -> bool:
        """
        判断快线是否在慢线上方

        Returns:
            bool: 快线是否在慢线上方
        """
        return self.get_position_status() == '上方'

    def detect_all_cross_downs(self) -> pd.Series:
        """
        检测所有历史死叉点

        Returns:
            pd.Series: 布尔序列，True 表示当日发生死叉
        """
        if self._fast_ma is None or self._slow_ma is None:
            raise ValueError("请先调用 calculate() 方法计算均线")

        # 前一日快线 >= 慢线，当日快线 < 慢线
        prev_fast = self._fast_ma.shift(1)
        prev_slow = self._slow_ma.shift(1)

        cross_down = (prev_fast >= prev_slow) & (self._fast_ma < self._slow_ma)
        return cross_down.fillna(False)

    def detect_all_cross_ups(self) -> pd.Series:
        """
        检测所有历史金叉点

        Returns:
            pd.Series: 布尔序列，True 表示当日发生金叉
        """
        if self._fast_ma is None or self._slow_ma is None:
            raise ValueError("请先调用 calculate() 方法计算均线")

        # 前一日快线 <= 慢线，当日快线 > 慢线
        prev_fast = self._fast_ma.shift(1)
        prev_slow = self._slow_ma.shift(1)

        cross_up = (prev_fast <= prev_slow) & (self._fast_ma > self._slow_ma)
        return cross_up.fillna(False)

    def get_cross_dates(self) -> Dict[str, list]:
        """
        获取所有交叉日期

        Returns:
            Dict[str, list]: {'金叉': [...], '死叉': [...]}
        """
        if self._fast_ma is None or self._slow_ma is None:
            raise ValueError("请先调用 calculate() 方法计算均线")

        cross_up_dates = self._fast_ma.index[self.detect_all_cross_ups()].tolist()
        cross_down_dates = self._fast_ma.index[self.detect_all_cross_downs()].tolist()

        return {
            '金叉': cross_up_dates,
            '死叉': cross_down_dates
        }

    def __repr__(self) -> str:
        """类实例的字符串表示"""
        return f"MACalculator(fast={self.fast_period}, slow={self.slow_period}, type={self.ma_type})"


# ==================== 测试模块 ====================

if __name__ == "__main__":
    import logging
    import os
    import json
    from datetime import datetime

    # ==================== 测试参数配置 ====================
    # 【手动填写以下三个参数】

    # 测试时间段
    START_DATE = '2026-01-01'
    END_DATE = '2026-04-10'

    # 测试标的（只需输入数字代码，脚本自动添加交易所前缀）
    SYMBOL = '512480'                    # 半导体ETF -> SHSE.512480

    # 自动添加交易所前缀
    def _add_exchange_prefix(code: str) -> str:
        code = code.strip()
        if code.startswith('SHSE.') or code.startswith('SZSE.'):
            return code
        if code.startswith('6') or code.startswith('5'):
            return f'SHSE.{code}'
        elif code.startswith('0') or code.startswith('3') or code.startswith('1'):
            return f'SZSE.{code}'
        else:
            return f'SZSE.{code}'

    SYMBOL_FULL = _add_exchange_prefix(SYMBOL)

    # ==================== 日志配置 ====================

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    LOG_DIR = os.path.join(project_root, 'logs')

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_FILE = os.path.join(LOG_DIR,
                            f'ma_test_{SYMBOL_FULL.replace(".", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # ==================== 测试开始 ====================

    logger.info("=" * 60)
    logger.info("均线计算器测试（MA5 + MA10 + 死叉检测）")
    logger.info("=" * 60)

    logger.info(f"测试参数配置:")
    logger.info(f"  股票代码: {SYMBOL_FULL}")
    logger.info(f"  开始时间: {START_DATE}")
    logger.info(f"  结束时间: {END_DATE}")
    logger.info(f"  日志文件: {LOG_FILE}")
    logger.info("=" * 60)

    # ==================== 使用掘金量化 API 获取真实数据 ====================
    try:
        from gm.api import history, set_token, ADJUST_NONE, get_trading_dates

        # 从配置文件读取 token
        config_path = os.path.join(project_root, 'config', 'settings.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                gm_token = config.get('gm_token')
                if gm_token:
                    set_token(gm_token)
                    logger.info("掘金量化 SDK 初始化成功")
                else:
                    logger.error("配置文件中未找到 gm_token")
                    raise Exception("缺少 gm_token 配置")
        else:
            logger.error(f"配置文件不存在: {config_path}")
            raise Exception("缺少配置文件")

        logger.info("使用掘金量化 API 获取真实历史数据（不复权）...")

        # 获取日线数据
        daily_data = history(
            symbol=SYMBOL_FULL,
            frequency='1d',
            start_time=START_DATE + ' 09:00:00',
            end_time=END_DATE + ' 15:30:00',
            fields='eob,close',
            df=True,
            adjust=ADJUST_NONE
        )

        if daily_data is None or daily_data.empty:
            logger.error(f"获取数据失败，请检查股票代码 {SYMBOL_FULL} 是否正确")
            raise Exception("数据获取失败")

        logger.info(f"获取到 {len(daily_data)} 条日线数据")

        # 构建收盘价序列
        daily_data['eob'] = pd.to_datetime(daily_data['eob'])
        close_series = pd.Series(
            daily_data['close'].values,
            index=daily_data['eob'],
            name='Close'
        )
        dates_index = close_series.index
        prices = close_series.values

    except ImportError:
        logger.error("未安装掘金量化 SDK (gm)，请先安装: pip install gm")
        raise Exception("缺少掘金量化 SDK")
    except Exception as e:
        logger.error(f"获取真实数据失败: {e}")
        raise

    # ---------- 测试 1: MA5/MA10 计算 ----------
    logger.info("")
    logger.info("【1. MA5/MA10 计算测试】")
    logger.info("-" * 40)

    ma = MACalculator(fast_period=5, slow_period=10, ma_type='SMA')
    logger.info(f"创建实例: {ma}")

    fast_ma, slow_ma = ma.calculate(close_series)

    logger.info(f"计算完成: {len(fast_ma)} 个数据点")
    logger.info(f"MA5 有效数据: {fast_ma.notna().sum()} 个（前5个为NaN）")
    logger.info(f"MA10 有效数据: {slow_ma.notna().sum()} 个（前10个为NaN）")

    logger.info("")
    logger.info("MA5/MA10 计算结果（全部）:")
    logger.info("-" * 60)
    logger.info(f"日期              | 收盘价    | MA5       | MA10      | 位置    | 死叉?")
    logger.info("-" * 60)

    # 检测所有死叉点
    all_cross_downs = ma.detect_all_cross_downs()

    for i, (date, close_val, ma5_val, ma10_val) in enumerate(zip(dates_index, prices, fast_ma, slow_ma)):
        date_str = date.strftime('%Y-%m-%d')
        is_cross_down = all_cross_downs.iloc[i] if i < len(all_cross_downs) else False

        if pd.notna(ma5_val) and pd.notna(ma10_val):
            position = '上方' if ma5_val > ma10_val else '下方' if ma5_val < ma10_val else '持平'
            cross_str = '死叉!' if is_cross_down else ''
            logger.info(f"{date_str} | {close_val:8.3f} | {ma5_val:8.3f} | {ma10_val:8.3f} | {position}  | {cross_str}")
        else:
            logger.info(f"{date_str} | {close_val:8.3f} |     NaN  |     NaN  | -       |")

    # ---------- 测试 2: 死叉日期汇总 ----------
    logger.info("")
    logger.info("【2. 死叉日期汇总测试】")
    logger.info("-" * 40)

    cross_dates_dict = ma.get_cross_dates()
    logger.info(f"金叉日期数量: {len(cross_dates_dict['金叉'])}")
    logger.info(f"死叉日期数量: {len(cross_dates_dict['死叉'])}")

    if len(cross_dates_dict['金叉']) > 0:
        logger.info("金叉日期:")
        for d in cross_dates_dict['金叉'][:10]:
            logger.info(f"  {d.strftime('%Y-%m-%d')}")

    if len(cross_dates_dict['死叉']) > 0:
        logger.info("死叉日期:")
        for d in cross_dates_dict['死叉']:
            logger.info(f"  {d.strftime('%Y-%m-%d')}")

    # ---------- 测试 3: 当前状态 ----------
    logger.info("")
    logger.info("【3. 当前状态测试】")
    logger.info("-" * 40)

    position = ma.get_position_status()
    logger.info(f"当前快线(MA5)位置: {position}")

    is_below = ma.is_below_slow_ma()
    is_above = ma.is_above_slow_ma()
    logger.info(f"MA5在MA10下方: {is_below}")
    logger.info(f"MA5在MA10上方: {is_above}")

    last_ma5 = fast_ma.iloc[-1]
    last_ma10 = slow_ma.iloc[-1]
    last_close = prices[-1]
    logger.info(f"最后一天详情:")
    logger.info(f"  收盘价: {last_close:.3f}")
    logger.info(f"  MA5: {last_ma5:.3f}")
    logger.info(f"  MA10: {last_ma10:.3f}")

    # ---------- 测试 4: EMA类型 ----------
    logger.info("")
    logger.info("【4. EMA类型测试】")
    logger.info("-" * 40)

    ma_ema = MACalculator(fast_period=5, slow_period=10, ma_type='EMA')
    logger.info(f"创建实例: {ma_ema}")

    ema_fast, ema_slow = ma_ema.calculate(close_series)

    logger.info("最后5天 SMA vs EMA 对比:")
    logger.info("-" * 60)
    logger.info(f"日期              | SMA5      | EMA5      | SMA10     | EMA10")
    logger.info("-" * 60)
    for i in range(-5, 0):
        date = dates_index[i]
        logger.info(f"{date.strftime('%Y-%m-%d')} | {fast_ma.iloc[i]:8.3f} | {ema_fast.iloc[i]:8.3f} | {slow_ma.iloc[i]:8.3f} | {ema_slow.iloc[i]:8.3f}")

    # ---------- 测试完成 ----------
    logger.info("")
    logger.info("=" * 60)
    logger.info("所有测试完成!")
    logger.info(f"日志已保存至: {LOG_FILE}")
    logger.info("=" * 60)