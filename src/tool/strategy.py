# -*- coding: utf-8 -*-
"""
交易策略模块 (strategy.py)

【功能说明】
定义买卖信号检测逻辑，与回测执行分离。策略类只负责信号检测，不执行实际交易。

【核心逻辑】
卖出条件 (judge_sell_ids):
- sell_id=1: 综合卖出策略（基于SAR死叉触发）
  - 周线顶背离 + SAR死叉 → 清仓 (sell_flag=1)
  - 周线RSI>90 + SAR死叉 → 清仓 (sell_flag=1)
  - 周线RSI>85 + SAR死叉 → 卖出1/2 (sell_flag=2)
  - 日线顶背离 + SAR死叉 → 卖出1/3 (sell_flag=3)
  - 周线RSI>80 + SAR死叉 → 卖出1/3 (sell_flag=3)

- sell_id=2: 标准卖出策略（基于MACD日线死叉触发）
  触发条件：MACD日线跌破死叉
  按优先级判断：
  1. 周线顶背离 + SAR跌破死叉 → 清仓 (sell_flag=1)
  2. 日线顶背离 + SAR跌破死叉 → 卖出1/3 (sell_flag=3)
  3. SAR跌破死叉 + 周线RSI阶梯：
     - 周线RSI > 90 → 清仓 (sell_flag=1)
     - 周线RSI > 85 → 卖出1/2 (sell_flag=2)
     - 周线RSI > 80 → 卖出1/3 (sell_flag=3)

买入条件:
- judge_buy_ids: 新资金买入策略ID（触发时买入新资金+买回卖出资金）
- judge_t_ids: 做T买回策略ID（触发时只买回卖出资金）

buy_id 条件:
- buy_id=1: 日RSI<20 且 周RSI<25 (保守型)
- buy_id=2: 日RSI<25 且 周RSI<30 (标准型)
- buy_id=3: 创业板指数日RSI<25 (指数保护型)
- buy_id=4: 日RSI<20 且 周RSI<20 (极度保守型)
- buy_id=5: 创业板指数日RSI<20 (指数保护型)
- buy_id=6: 上证指数日RSI<20 (上证指数保护型)
- buy_id=7: 上证指数日RSI<25 (上证指数保护型)
- buy_id=8: 科创综指日RSI<20 (科创板买入)
- buy_id=9: 日线底背离形成 (个股底背离买入)
- buy_id=10: 上证指数日RSI<20 且 个股月RSI<30 (红利股买点)
- buy_id=11: 日RSI<20 且 周RSI<20 且 月RSI<30 (底背离加强)
- buy_id=15: 日RSI<20 且 月RSI<22 (极度保守型)
- buy_id=16: 日RSI<20 且 月RSI<23 (极度保守型)
- buy_id=17: 日RSI<20 且 月RSI<20 (极度保守型)
- buy_id=18: 日RSI<20 且 周RSI<20 且 月RSI<40 (月线宽松+日周超卖)
- buy_id=19: 日RSI<20 且 周RSI<20 且 月RSI<15 (三周期共振超卖)
- buy_id=20: 日RSI<20 且 周RSI<20 且 月RSI<25 (日周超卖+月线适中)
- buy_id=21: 日RSI<20 且 周RSI<20 且 月RSI<20 (三周期共振超卖)
- buy_id=22: 日RSI<30 (日线宽松超卖)
- buy_id=23: 周RSI<20 (周线超卖)
- buy_id=24: 日RSI<20 且 月RSI<15 (日线+月线超卖)
- buy_id=25: 日RSI<20 且 月RSI<40 (日线超卖+月线宽松)
- buy_id=26: 日RSI<20 且 月RSI<25 (日线超卖+月线适中)

【买入资金逻辑】
- judge_buy_ids触发: 买入所有可用资金（新资金 + 卖出后待买回资金），无条件限制
- judge_t_ids触发: 只买回卖出后的资金，不买入新资金

作者：量化交易团队
创建日期：2024
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==================== 常量定义 ====================

class SellFlag(Enum):
    """卖出级别枚举"""
    NO_SIGNAL = 0       # 无信号
    CLEAR_ALL = 1       # 清仓
    SELL_HALF = 2       # 卖出1/2
    SELL_ONE_THIRD = 3  # 卖出1/3


# RSI阈值常量
RSI_THRESHOLDS = {
    'daily_buy': {1: 20, 2: 25, 3: 25, 4: 20, 5: 20},
    'weekly_buy': {1: 25, 2: 30, 3: 999, 4: 20, 5: 999},  # 999表示不检查
    'weekly_sell_level1': 80,   # 一级警戒
    'weekly_sell_level2': 85,   # 二级警戒
    'weekly_sell_level3': 90    # 三级警戒(清仓)
}

# 卖出后冷却期天数（已禁用）
COOLDOWN_DAYS = 0

# 指数顶背离与个股RSI峰值最大间隔（工作日）
MAX_DIVERGENCE_RSI_PEAK_DAYS = 5


def count_trading_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """
    计算两个日期之间的工作日数（不含周末）

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        工作日数量（绝对值）
    """
    if start_date is None or end_date is None:
        return 999  # 无法计算时返回大值，视为失效

    # 使用pandas的bdate_range计算工作日
    try:
        # 处理时区差异：统一转换为无时区的日期
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_localize(None)

        # 确保日期顺序正确
        if start_date <= end_date:
            business_days = pd.bdate_range(start=start_date, end=end_date)
        else:
            business_days = pd.bdate_range(start=end_date, end=start_date)
        return len(business_days) - 1  # 减1因为包含边界日期
    except Exception as e:
        # 出错时使用自然日估算（保守）
        logger.debug(f"计算工作日失败: {e}, 使用自然日估算")
        return abs((end_date - start_date).days)


# ==================== 数据结构定义 ====================

@dataclass
class StrategyState:
    """策略状态"""
    rsi_flag: int = 0                               # RSI警戒级别 (0-3)
    rsi_peak_date: Optional[pd.Timestamp] = None    # RSI峰值日期
    rsi_peak_value: float = 0.0                     # RSI峰值
    daily_divergence_flag: int = 0                  # 日线顶背离标志
    weekly_divergence_flag: int = 0                 # 周线顶背离标志
    daily_divergence_info: Optional[Dict] = None    # 日线顶背离详情
    weekly_divergence_info: Optional[Dict] = None   # 周线顶背离详情
    daily_bottom_divergence_flag: int = 0           # 日线底背离标志
    daily_bottom_divergence_info: Optional[Dict] = None  # 日线底背离详情
    last_sell_date: Optional[pd.Timestamp] = None   # 上次卖出日期

    # 指数顶背离状态（用于 sell_id=4,5,6）
    # 上证指数顶背离
    sh_index_daily_divergence_flag: int = 0         # 上证指数日线顶背离标志
    sh_index_weekly_divergence_flag: int = 0        # 上证指数周线顶背离标志
    sh_index_daily_divergence_info: Optional[Dict] = None   # 上证指数日线顶背离详情
    sh_index_weekly_divergence_info: Optional[Dict] = None  # 上证指数周线顶背离详情

    # 创业板指数顶背离
    cyb_index_daily_divergence_flag: int = 0        # 创业板指数日线顶背离标志
    cyb_index_weekly_divergence_flag: int = 0       # 创业板指数周线顶背离标志
    cyb_index_daily_divergence_info: Optional[Dict] = None  # 创业板指数日线顶背离详情
    cyb_index_weekly_divergence_info: Optional[Dict] = None # 创业板指数周线顶背离详情

    # 科创板指数顶背离
    kc_index_daily_divergence_flag: int = 0         # 科创板指数日线顶背离标志
    kc_index_weekly_divergence_flag: int = 0        # 科创板指数周线顶背离标志
    kc_index_daily_divergence_info: Optional[Dict] = None   # 科创板指数日线顶背离详情
    kc_index_weekly_divergence_info: Optional[Dict] = None  # 科创板指数周线顶背离详情

    # 上证50指数顶背离（用于sell_id=13）
    sh50_index_daily_divergence_flag: int = 0       # 上证50指数日线顶背离标志
    sh50_index_weekly_divergence_flag: int = 0      # 上证50指数周线顶背离标志
    sh50_index_daily_divergence_info: Optional[Dict] = None # 上证50指数日线顶背离详情
    sh50_index_weekly_divergence_info: Optional[Dict] = None # 上证50指数周线顶背离详情

    # 创业板50指数顶背离（用于sell_id=14）
    cyb50_index_daily_divergence_flag: int = 0      # 创业板50指数日线顶背离标志
    cyb50_index_weekly_divergence_flag: int = 0     # 创业板50指数周线顶背离标志
    cyb50_index_daily_divergence_info: Optional[Dict] = None # 创业板50指数日线顶背离详情
    cyb50_index_weekly_divergence_info: Optional[Dict] = None # 创业板50指数周线顶背离详情

    # 科创板50指数顶背离（用于sell_id=15）
    kc50_index_daily_divergence_flag: int = 0       # 科创板50指数日线顶背离标志
    kc50_index_weekly_divergence_flag: int = 0      # 科创板50指数周线顶背离标志
    kc50_index_daily_divergence_info: Optional[Dict] = None # 科创板50指数日线顶背离详情
    kc50_index_weekly_divergence_info: Optional[Dict] = None # 科创板50指数周线顶背离详情


@dataclass
class SellSignal:
    """卖出信号"""
    flag: SellFlag                                  # 卖出级别
    reason: str                                     # 原因说明
    daily_rsi: float                                # 当日RSI
    weekly_rsi: float                               # 当日周线RSI
    rsi_peak_date: Optional[pd.Timestamp] = None    # RSI峰值日期
    rsi_peak_value: float = 0.0                     # RSI峰值
    divergence_date: Optional[pd.Timestamp] = None  # 背离形成日期
    divergence_prev_high: float = 0.0               # 背离前高


@dataclass
class BuySignal:
    """买入信号"""
    triggered: bool                                 # 是否触发
    is_new_cash: bool                               # 是否是新资金买入（True=judge_buy_id, False=judge_t_id）
    reason: str                                     # 原因说明
    daily_rsi: float                                # 当日RSI
    weekly_rsi: float                               # 当日周线RSI
    monthly_rsi: float = np.nan                     # 当日月线RSI


# ==================== 交易策略类 ====================

class TradingStrategy:
    """
    交易策略类

    【功能说明】
    根据买卖条件检测交易信号，不执行实际交易。

    【使用方法】
    strategy = TradingStrategy(buy_ids=[1], t_ids=[2], sell_ids=[1])

    # 每日调用
    sell_signal = strategy.check_sell_signal(date, indicators, state)
    buy_signal = strategy.check_buy_signal(date, indicators, state, has_new_cash, has_sold_cash)

    # 更新状态
    strategy.update_state(date, indicators, state, trade_executed)
    """

    def __init__(self,
                 buy_ids: List[int] = [1],
                 t_ids: List[int] = [2],
                 sell_ids: List[int] = [1]) -> None:
        """
        初始化交易策略

        Args:
            buy_ids: 新资金买入策略ID列表（触发时买入新资金+买回卖出资金）
            t_ids: 做T买回策略ID列表（触发时只买回卖出资金）
            sell_ids: 卖出策略ID列表
        """
        self.buy_ids = buy_ids
        self.t_ids = t_ids
        self.sell_ids = sell_ids

        # 验证ID范围
        for buy_id in buy_ids:
            if buy_id not in range(1, 27):
                raise ValueError(f"buy_id 必须为 1-26，当前值: {buy_id}")
        for t_id in t_ids:
            if t_id not in range(1, 27):
                raise ValueError(f"t_id 必须为 1-26，当前值: {t_id}")

        logger.info(f"策略初始化: buy_ids={buy_ids}, t_ids={t_ids}, sell_ids={sell_ids}")

    def check_sell_signal(
        self,
        date: pd.Timestamp,
        daily_rsi: float,
        weekly_rsi: float,
        sar_cross_down: bool,
        macd_cross_down: bool,
        state: StrategyState,
        position: float,
        sell_id: int = 1,
        sh_index_macd_cross_down: bool = False,
        cyb_index_macd_cross_down: bool = False,
        kc_index_macd_cross_down: bool = False,
        sh50_index_macd_cross_down: bool = False,
        cyb50_index_macd_cross_down: bool = False,
        kc50_index_macd_cross_down: bool = False,
        macd_dead_cross_state: bool = False,
        sh_index_dead_cross_state: bool = False,
        cyb_index_dead_cross_state: bool = False,
        kc_index_dead_cross_state: bool = False,
        sh50_index_dead_cross_state: bool = False,
        cyb50_index_dead_cross_state: bool = False,
        kc50_index_dead_cross_state: bool = False
    ) -> Optional[SellSignal]:
        """
        检查卖出信号

        Args:
            date: 当前日期
            daily_rsi: 当日RSI值
            weekly_rsi: 当日周线RSI值
            sar_cross_down: 是否发生SAR死叉跌破（收盘价跌破SAR）
            macd_cross_down: 是否发生MACD日线跌破死叉（MACD死叉 + 最低价跌破SAR）
            state: 当前策略状态
            position: 当前持仓比例
            sell_id: 卖出策略ID
            sh_index_macd_cross_down: 是否发生上证指数MACD日线跌破死叉（用于 sell_id=7和sell_id=10）
            cyb_index_macd_cross_down: 是否发生创业板指数MACD日线跌破死叉（用于 sell_id=8和sell_id=11）
            kc_index_macd_cross_down: 是否发生科创板指数MACD日线跌破死叉（用于 sell_id=9和sell_id=12）
            sh50_index_macd_cross_down: 是否发生上证50指数MACD日线跌破死叉（用于 sell_id=13）
            cyb50_index_macd_cross_down: 是否发生创业板50指数MACD日线跌破死叉（用于 sell_id=14）
            kc50_index_macd_cross_down: 是否发生科创板50指数MACD日线跌破死叉（用于 sell_id=15）
            macd_dead_cross_state: 个股MACD是否已处于死叉状态（DIF < DEA）
            sh_index_dead_cross_state: 上证指数MACD是否已处于死叉状态（DIF < DEA）
            cyb_index_dead_cross_state: 创业板指数MACD是否已处于死叉状态（DIF < DEA）
            kc_index_dead_cross_state: 科创板指数MACD是否已处于死叉状态（DIF < DEA）
            sh50_index_dead_cross_state: 上证50指数MACD是否已处于死叉状态（DIF < DEA）
            cyb50_index_dead_cross_state: 创业板50指数MACD是否已处于死叉状态（DIF < DEA）
            kc50_index_dead_cross_state: 科创板50指数MACD是否已处于死叉状态（DIF < DEA）

        Returns:
            SellSignal 或 None（无信号）

        sell_id 说明:
        - sell_id=1: 触发条件为SAR死叉跌破（收盘价跌破SAR），含RSI阶梯判断
        - sell_id=2: 触发条件为MACD日线跌破死叉，含RSI阶梯判断
        - sell_id=3: 触发条件为MACD日线跌破死叉，仅背离判断（不含RSI阶梯）
        - sell_id=4: 指数顶背离场景：
            (1) 指数顶背离 + 指数处于死叉 → 直接卖出
            (2) 指数顶背离 + 指数未死叉 + 个股处于死叉 → 等待指数死叉卖出
            (3) 指数顶背离 + 指数未死叉 + 个股未死叉 → 等待个股或指数死叉卖出
        - sell_id=5,6: 同sell_id=4，对应创业板/科创板指数
        - sell_id=7,8,9: 指数MACD日线跌破死叉 + 指数顶背离(日线或周线) + 个股周线RSI阶梯
        - sell_id=10: 个股顶背离场景：
            (1) 个股顶背离 + 个股处于死叉 → 直接卖出
            (2) 个股顶背离 + 个股未死叉 + 指数处于死叉 → 等待个股死叉卖出
            (3) 个股顶背离 + 个股未死叉 + 指数未死叉 → 等待个股或指数死叉卖出
        - sell_id=11,12,13,14,15: 同sell_id=10，对应不同指数
        """
        # 检查卖出冷却期（只在有持仓时检查）
        if position > 0 and state.last_sell_date is not None:
            days_since_sell = (date - state.last_sell_date).days
            if days_since_sell < COOLDOWN_DAYS:
                return None

        # 根据sell_id选择触发条件
        if sell_id == 1:
            # sell_id=1: SAR死叉跌破（收盘价跌破SAR）作为触发条件
            trigger_condition = sar_cross_down
            trigger_name = "SAR死叉跌破"
        elif sell_id == 2:
            # sell_id=2: MACD日线跌破死叉（MACD死叉 + 最低价跌破SAR）作为触发条件
            trigger_condition = macd_cross_down
            trigger_name = "MACD跌破死叉"
        elif sell_id == 3:
            # sell_id=3: MACD日线跌破死叉作为触发条件，仅背离判断
            trigger_condition = macd_cross_down
            trigger_name = "MACD跌破死叉"
        elif sell_id == 4:
            # sell_id=4: MACD日线跌破死叉 + 上证指数顶背离 + 个股周线RSI阶梯
            trigger_condition = macd_cross_down
            trigger_name = "MACD跌破死叉"
        elif sell_id == 5:
            # sell_id=5: MACD日线跌破死叉 + 创业板指数顶背离 + 个股周线RSI阶梯
            trigger_condition = macd_cross_down
            trigger_name = "MACD跌破死叉"
        elif sell_id == 6:
            # sell_id=6: MACD日线跌破死叉 + 科创板指数顶背离 + 个股周线RSI阶梯
            trigger_condition = macd_cross_down
            trigger_name = "MACD跌破死叉"
        elif sell_id == 7:
            # sell_id=7: 上证指数MACD日线跌破死叉 + 上证指数顶背离(日线或周线) + 个股周线RSI阶梯
            trigger_condition = sh_index_macd_cross_down
            trigger_name = "上证指数MACD跌破死叉"
        elif sell_id == 8:
            # sell_id=8: 创业板指数MACD日线跌破死叉 + 创业板指数顶背离(日线或周线) + 个股周线RSI阶梯
            trigger_condition = cyb_index_macd_cross_down
            trigger_name = "创业板指数MACD跌破死叉"
        elif sell_id == 9:
            # sell_id=9: 科创板指数MACD日线跌破死叉 + 科创板指数顶背离(日线或周线) + 个股周线RSI阶梯
            trigger_condition = kc_index_macd_cross_down
            trigger_name = "科创板指数MACD跌破死叉"
        elif sell_id == 10:
            # sell_id=10: 个股顶背离(日线或周线) + 上证指数环境
            # 触发条件：个股MACD死叉 或 上证指数MACD死叉
            # 死叉触发后，根据日线/周线顶背离状态决定卖出比例
            trigger_condition = macd_cross_down or sh_index_macd_cross_down
            trigger_name = "个股或上证指数MACD死叉触发"
        elif sell_id == 11:
            # sell_id=11: 个股顶背离(日线或周线) + 创业板指数环境
            # 触发条件：个股MACD死叉 或 创业板指数MACD死叉
            # 死叉触发后，根据日线/周线顶背离状态决定卖出比例
            trigger_condition = macd_cross_down or cyb_index_macd_cross_down
            trigger_name = "个股或创业板指数MACD死叉触发"
        elif sell_id == 12:
            # sell_id=12: 个股顶背离(日线或周线) + 科创板指数环境
            # 触发条件：个股MACD死叉 或 科创板指数MACD死叉
            # 死叉触发后，根据日线/周线顶背离状态决定卖出比例
            trigger_condition = macd_cross_down or kc_index_macd_cross_down
            trigger_name = "个股或科创板指数MACD死叉触发"
        elif sell_id == 13:
            # sell_id=13: 个股顶背离(日线或周线) + 上证50指数环境
            # 触发条件：个股MACD死叉 或 上证50指数MACD死叉
            # 死叉触发后，根据日线/周线顶背离状态决定卖出比例
            trigger_condition = macd_cross_down or sh50_index_macd_cross_down
            trigger_name = "个股或上证50指数MACD死叉触发"
        elif sell_id == 14:
            # sell_id=14: 个股顶背离(日线或周线) + 创业板50指数环境
            # 触发条件：个股MACD死叉 或 创业板50指数MACD死叉
            # 死叉触发后，根据日线/周线顶背离状态决定卖出比例
            trigger_condition = macd_cross_down or cyb50_index_macd_cross_down
            trigger_name = "个股或创业板50指数MACD死叉触发"
        elif sell_id == 15:
            # sell_id=15: 个股顶背离(日线或周线) + 科创板50指数环境
            # 触发条件：个股MACD死叉 或 科创板50指数MACD死叉
            # 死叉触发后，根据日线/周线顶背离状态决定卖出比例
            trigger_condition = macd_cross_down or kc50_index_macd_cross_down
            trigger_name = "个股或科创板50指数MACD死叉触发"
        else:
            # 默认使用SAR死叉跌破
            trigger_condition = sar_cross_down
            trigger_name = "SAR死叉跌破"

        # 必须满足触发条件才继续检查
        if not trigger_condition:
            return None

        # 【关键逻辑】触发条件满足（死叉发生），立即重置flag
        # 保存当前flag值用于判断卖出级别，然后重置
        current_rsi_flag = state.rsi_flag
        current_rsi_peak_date = state.rsi_peak_date
        current_rsi_peak_value = state.rsi_peak_value
        current_weekly_div_flag = state.weekly_divergence_flag
        current_daily_div_flag = state.daily_divergence_flag
        current_weekly_div_info = state.weekly_divergence_info
        current_daily_div_info = state.daily_divergence_info

        # 保存指数顶背离状态（用于 sell_id=4,5,6）
        current_sh_daily_div_flag = state.sh_index_daily_divergence_flag
        current_sh_weekly_div_flag = state.sh_index_weekly_divergence_flag
        current_sh_daily_div_info = state.sh_index_daily_divergence_info
        current_sh_weekly_div_info = state.sh_index_weekly_divergence_info
        current_cyb_daily_div_flag = state.cyb_index_daily_divergence_flag
        current_cyb_weekly_div_flag = state.cyb_index_weekly_divergence_flag
        current_cyb_daily_div_info = state.cyb_index_daily_divergence_info
        current_cyb_weekly_div_info = state.cyb_index_weekly_divergence_info
        current_kc_daily_div_flag = state.kc_index_daily_divergence_flag
        current_kc_weekly_div_flag = state.kc_index_weekly_divergence_flag
        current_kc_daily_div_info = state.kc_index_daily_divergence_info
        current_kc_weekly_div_info = state.kc_index_weekly_divergence_info

        # 保存新指数顶背离状态（用于 sell_id=13,14,15）
        current_sh50_daily_div_flag = state.sh50_index_daily_divergence_flag
        current_sh50_weekly_div_flag = state.sh50_index_weekly_divergence_flag
        current_sh50_daily_div_info = state.sh50_index_daily_divergence_info
        current_sh50_weekly_div_info = state.sh50_index_weekly_divergence_info
        current_cyb50_daily_div_flag = state.cyb50_index_daily_divergence_flag
        current_cyb50_weekly_div_flag = state.cyb50_index_weekly_divergence_flag
        current_cyb50_daily_div_info = state.cyb50_index_daily_divergence_info
        current_cyb50_weekly_div_info = state.cyb50_index_weekly_divergence_info
        current_kc50_daily_div_flag = state.kc50_index_daily_divergence_flag
        current_kc50_weekly_div_flag = state.kc50_index_weekly_divergence_flag
        current_kc50_daily_div_info = state.kc50_index_daily_divergence_info
        current_kc50_weekly_div_info = state.kc50_index_weekly_divergence_info

        # 用于背离判断的触发条件
        divergence_trigger = sar_cross_down if sell_id == 1 else macd_cross_down

        # 立即重置所有flag（死叉触发后清空历史信号）
        state.rsi_flag = 0
        state.rsi_peak_value = 0.0
        state.rsi_peak_date = None
        state.weekly_divergence_flag = 0
        state.weekly_divergence_info = None
        state.daily_divergence_flag = 0
        state.daily_divergence_info = None

        # 重置指数背离状态（用于 sell_id=4,5,6）
        state.sh_index_daily_divergence_flag = 0
        state.sh_index_weekly_divergence_flag = 0
        state.sh_index_daily_divergence_info = None
        state.sh_index_weekly_divergence_info = None
        state.cyb_index_daily_divergence_flag = 0
        state.cyb_index_weekly_divergence_flag = 0
        state.cyb_index_daily_divergence_info = None
        state.cyb_index_weekly_divergence_info = None
        state.kc_index_daily_divergence_flag = 0
        state.kc_index_weekly_divergence_flag = 0
        state.kc_index_daily_divergence_info = None
        state.kc_index_weekly_divergence_info = None

        # 重置新指数背离状态（用于 sell_id=13,14,15）
        state.sh50_index_daily_divergence_flag = 0
        state.sh50_index_weekly_divergence_flag = 0
        state.sh50_index_daily_divergence_info = None
        state.sh50_index_weekly_divergence_info = None
        state.cyb50_index_daily_divergence_flag = 0
        state.cyb50_index_weekly_divergence_flag = 0
        state.cyb50_index_daily_divergence_info = None
        state.cyb50_index_weekly_divergence_info = None
        state.kc50_index_daily_divergence_flag = 0
        state.kc50_index_weekly_divergence_flag = 0
        state.kc50_index_daily_divergence_info = None
        state.kc50_index_weekly_divergence_info = None

        # 【sell_id=3 特殊处理】仅背离判断，不含RSI阶梯
        if sell_id == 3:
            # 优先级1：周线顶背离 → 清仓
            if current_weekly_div_flag == 1:
                div_info = current_weekly_div_info or {}
                divergence_date = div_info.get('date')
                divergence_prev_high_date = div_info.get('prev_high_date')
                divergence_prev_high = div_info.get('prev_high', 0.0)
                prev_high_date_str = divergence_prev_high_date.strftime('%Y-%m-%d') if divergence_prev_high_date else 'N/A'
                reason = f"清仓信号 (sell_id=3-周线顶背离): 周线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, 前高点{prev_high_date_str}, {trigger_name}触发, 前高={divergence_prev_high:.3f}, 建议清仓"
                return SellSignal(
                    flag=SellFlag.CLEAR_ALL,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    divergence_date=divergence_date,
                    divergence_prev_high=divergence_prev_high
                )

            # 优先级2：日线顶背离 → 卖出1/3
            if current_daily_div_flag == 1:
                div_info = current_daily_div_info or {}
                divergence_date = div_info.get('date')
                divergence_prev_high_date = div_info.get('prev_high_date')
                divergence_prev_high = div_info.get('prev_high', 0.0)
                prev_high_date_str = divergence_prev_high_date.strftime('%Y-%m-%d') if divergence_prev_high_date else 'N/A'
                reason = f"卖出信号 (sell_id=3-日线顶背离): 日线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, 前高点{prev_high_date_str}, {trigger_name}触发, 前高={divergence_prev_high:.3f}, 建议卖出1/3"
                return SellSignal(
                    flag=SellFlag.SELL_ONE_THIRD,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    divergence_date=divergence_date,
                    divergence_prev_high=divergence_prev_high
                )

            # sell_id=3 仅背离判断，无其他信号
            return None

        # 【sell_id=4, 5, 6】指数顶背离 + 个股周线RSI阶梯判断
        if sell_id in [4, 5, 6]:
            # 确定检查哪个指数的背离状态
            if sell_id == 4:
                # 上证指数顶背离
                index_name = "上证指数"
                has_index_daily_div = current_sh_daily_div_flag == 1
                has_index_weekly_div = current_sh_weekly_div_flag == 1
                index_daily_div_info = current_sh_daily_div_info
                index_weekly_div_info = current_sh_weekly_div_info
            elif sell_id == 5:
                # 创业板指数顶背离
                index_name = "创业板指数"
                has_index_daily_div = current_cyb_daily_div_flag == 1
                has_index_weekly_div = current_cyb_weekly_div_flag == 1
                index_daily_div_info = current_cyb_daily_div_info
                index_weekly_div_info = current_cyb_weekly_div_info
            elif sell_id == 6:
                # 科创板指数顶背离
                index_name = "科创板指数"
                has_index_daily_div = current_kc_daily_div_flag == 1
                has_index_weekly_div = current_kc_weekly_div_flag == 1
                index_daily_div_info = current_kc_daily_div_info
                index_weekly_div_info = current_kc_weekly_div_info
            else:
                return None

            # 检查指数是否出现顶背离（日线或周线）
            has_index_div = has_index_daily_div or has_index_weekly_div

            # 如果没有指数顶背离，则不触发卖出
            if not has_index_div:
                return None

            # 指数有顶背离，检查个股周线RSI决定卖出比例
            # 优先级：RSI>90 清仓 > RSI>85 卖出1/2 > RSI>80 卖出1/3
            rsi_peak_date = current_rsi_peak_date or date
            rsi_peak_value = current_rsi_peak_value or weekly_rsi

            # 获取指数背离详情
            index_div_type = "日线顶背离" if has_index_daily_div else "周线顶背离"
            index_div_info = index_daily_div_info if has_index_daily_div else index_weekly_div_info
            index_div_date = index_div_info.get('date') if index_div_info else None

            # 优先级1：周线RSI>90 或 rsi_flag=3 → 清仓
            if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level3'] or current_rsi_flag == 3:
                reason = f"清仓信号 (sell_id={sell_id}-{index_name}{index_div_type}): {index_name}{index_div_type}形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>90 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议清仓"
                return SellSignal(
                    flag=SellFlag.CLEAR_ALL,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    rsi_peak_date=rsi_peak_date,
                    rsi_peak_value=rsi_peak_value
                )

            # 优先级2：周线RSI>85 或 rsi_flag=2 → 卖出1/2
            if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2'] or current_rsi_flag == 2:
                reason = f"卖出信号 (sell_id={sell_id}-{index_name}{index_div_type}): {index_name}{index_div_type}形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>85 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/2"
                return SellSignal(
                    flag=SellFlag.SELL_HALF,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    rsi_peak_date=rsi_peak_date,
                    rsi_peak_value=rsi_peak_value
                )

            # 优先级3：周线RSI>80 或 rsi_flag=1 → 卖出1/3
            if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1'] or current_rsi_flag == 1:
                reason = f"卖出信号 (sell_id={sell_id}-{index_name}{index_div_type}): {index_name}{index_div_type}形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>80 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/3"
                return SellSignal(
                    flag=SellFlag.SELL_ONE_THIRD,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    rsi_peak_date=rsi_peak_date,
                    rsi_peak_value=rsi_peak_value
                )

            # 指数有顶背离但个股周线RSI未达到阈值，不触发卖出
            return None

        # 【sell_id=7, 8, 9】指数MACD跌破死叉 + 指数顶背离(日线或周线) + 个股周线RSI阶梯判断
        if sell_id in [7, 8, 9]:
            # 确定检查哪个指数的背离状态
            if sell_id == 7:
                index_name = "上证指数"
                has_index_daily_div = current_sh_daily_div_flag == 1
                has_index_weekly_div = current_sh_weekly_div_flag == 1
                index_daily_div_info = current_sh_daily_div_info
                index_weekly_div_info = current_sh_weekly_div_info
            elif sell_id == 8:
                index_name = "创业板指数"
                has_index_daily_div = current_cyb_daily_div_flag == 1
                has_index_weekly_div = current_cyb_weekly_div_flag == 1
                index_daily_div_info = current_cyb_daily_div_info
                index_weekly_div_info = current_cyb_weekly_div_info
            elif sell_id == 9:
                index_name = "科创板指数"
                has_index_daily_div = current_kc_daily_div_flag == 1
                has_index_weekly_div = current_kc_weekly_div_flag == 1
                index_daily_div_info = current_kc_daily_div_info
                index_weekly_div_info = current_kc_weekly_div_info
            else:
                return None

            # 检查指数是否出现顶背离（日线或周线）
            has_index_div = has_index_daily_div or has_index_weekly_div

            # 如果没有指数顶背离，则不触发卖出
            if not has_index_div:
                return None

            # 指数有顶背离，检查个股周线RSI决定卖出比例
            rsi_peak_date = current_rsi_peak_date or date
            rsi_peak_value = current_rsi_peak_value or weekly_rsi

            # 获取指数背离详情（优先日线，其次周线）
            index_div_type = "日线顶背离" if has_index_daily_div else "周线顶背离"
            index_div_info = index_daily_div_info if has_index_daily_div else index_weekly_div_info
            index_div_date = index_div_info.get('date') if index_div_info else None

            # 优先级1：周线RSI>90 或 rsi_flag=3 → 清仓
            if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level3'] or current_rsi_flag == 3:
                reason = f"清仓信号 (sell_id={sell_id}-{index_name}{index_div_type}): {index_name}{index_div_type}形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>90 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议清仓"
                return SellSignal(
                    flag=SellFlag.CLEAR_ALL,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    rsi_peak_date=rsi_peak_date,
                    rsi_peak_value=rsi_peak_value
                )

            # 优先级2：周线RSI>85 或 rsi_flag=2 → 卖出1/2
            if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2'] or current_rsi_flag == 2:
                reason = f"卖出信号 (sell_id={sell_id}-{index_name}{index_div_type}): {index_name}{index_div_type}形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>85 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/2"
                return SellSignal(
                    flag=SellFlag.SELL_HALF,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    rsi_peak_date=rsi_peak_date,
                    rsi_peak_value=rsi_peak_value
                )

            # 优先级3：周线RSI>80 或 rsi_flag=1 → 卖出1/3
            if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1'] or current_rsi_flag == 1:
                reason = f"卖出信号 (sell_id={sell_id}-{index_name}{index_div_type}): {index_name}{index_div_type}形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>80 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/3"
                return SellSignal(
                    flag=SellFlag.SELL_ONE_THIRD,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    rsi_peak_date=rsi_peak_date,
                    rsi_peak_value=rsi_peak_value
                )

            # 指数有顶背离但个股周线RSI未达到阈值，不触发卖出
            return None

        # 【sell_id=10, 11, 12, 13, 14, 15】指数MACD跌破死叉 + （个股背离或指数背离+个股RSI）
        if sell_id in [10, 11, 12, 13, 14, 15]:
            # 确定检查哪个指数的背离状态
            if sell_id == 10:
                index_name = "上证指数"
                has_index_daily_div = current_sh_daily_div_flag == 1
                has_index_weekly_div = current_sh_weekly_div_flag == 1
                index_daily_div_info = current_sh_daily_div_info
                index_weekly_div_info = current_sh_weekly_div_info
            elif sell_id == 11:
                index_name = "创业板指数"
                has_index_daily_div = current_cyb_daily_div_flag == 1
                has_index_weekly_div = current_cyb_weekly_div_flag == 1
                index_daily_div_info = current_cyb_daily_div_info
                index_weekly_div_info = current_cyb_weekly_div_info
            elif sell_id == 12:
                index_name = "科创板指数"
                has_index_daily_div = current_kc_daily_div_flag == 1
                has_index_weekly_div = current_kc_weekly_div_flag == 1
                index_daily_div_info = current_kc_daily_div_info
                index_weekly_div_info = current_kc_weekly_div_info
            elif sell_id == 13:
                index_name = "上证50指数"
                has_index_daily_div = current_sh50_daily_div_flag == 1
                has_index_weekly_div = current_sh50_weekly_div_flag == 1
                index_daily_div_info = current_sh50_daily_div_info
                index_weekly_div_info = current_sh50_weekly_div_info
            elif sell_id == 14:
                index_name = "创业板50指数"
                has_index_daily_div = current_cyb50_daily_div_flag == 1
                has_index_weekly_div = current_cyb50_weekly_div_flag == 1
                index_daily_div_info = current_cyb50_daily_div_info
                index_weekly_div_info = current_cyb50_weekly_div_info
            elif sell_id == 15:
                index_name = "科创板50指数"
                has_index_daily_div = current_kc50_daily_div_flag == 1
                has_index_weekly_div = current_kc50_weekly_div_flag == 1
                index_daily_div_info = current_kc50_daily_div_info
                index_weekly_div_info = current_kc50_weekly_div_info
            else:
                return None

            rsi_peak_date = current_rsi_peak_date or date
            rsi_peak_value = current_rsi_peak_value or weekly_rsi

            # 【优先级1】个股周线顶背离 → 清仓
            if current_weekly_div_flag == 1:
                div_info = current_weekly_div_info or {}
                divergence_date = div_info.get('date')
                divergence_prev_high = div_info.get('prev_high', 0.0)
                reason = f"清仓信号 (sell_id={sell_id}-个股周线顶背离): 个股周线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, {trigger_name}触发, 前高={divergence_prev_high:.3f}, 建议清仓"
                return SellSignal(
                    flag=SellFlag.CLEAR_ALL,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    divergence_date=divergence_date,
                    divergence_prev_high=divergence_prev_high
                )

            # 【优先级2】个股日线顶背离 → 卖出1/3
            if current_daily_div_flag == 1:
                div_info = current_daily_div_info or {}
                divergence_date = div_info.get('date')
                divergence_prev_high = div_info.get('prev_high', 0.0)
                reason = f"卖出信号 (sell_id={sell_id}-个股日线顶背离): 个股日线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, {trigger_name}触发, 前高={divergence_prev_high:.3f}, 建议卖出1/3"
                return SellSignal(
                    flag=SellFlag.SELL_ONE_THIRD,
                    reason=reason,
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    divergence_date=divergence_date,
                    divergence_prev_high=divergence_prev_high
                )

            # 【优先级3】指数日线顶背离 + 个股周线RSI阶梯
            if has_index_daily_div:
                index_div_info = index_daily_div_info or {}
                index_div_date = index_div_info.get('date') if index_div_info else None

                # 校验：指数顶背离与个股RSI峰值间隔不超过5个工作日，否则顶背离失效
                days_diff = count_trading_days(index_div_date, rsi_peak_date)
                if days_diff > MAX_DIVERGENCE_RSI_PEAK_DAYS:
                    logger.debug(f"{date.strftime('%Y-%m-%d')} | {index_name}日线顶背离({index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'})与个股RSI峰值({rsi_peak_date.strftime('%Y-%m-%d')})间隔{days_diff}个工作日，超过{MAX_DIVERGENCE_RSI_PEAK_DAYS}天，顶背离失效")
                    # 间隔过大，跳过指数日线顶背离检查，继续检查周线顶背离
                else:
                    # RSI>90 或 rsi_flag=3 → 清仓
                    if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level3'] or current_rsi_flag == 3:
                        reason = f"清仓信号 (sell_id={sell_id}-{index_name}日线顶背离): {index_name}日线顶背离形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>90 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议清仓"
                        return SellSignal(
                            flag=SellFlag.CLEAR_ALL,
                            reason=reason,
                            daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi,
                            rsi_peak_date=rsi_peak_date,
                            rsi_peak_value=rsi_peak_value
                        )

                    # RSI>85 或 rsi_flag=2 → 卖出1/2
                    if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2'] or current_rsi_flag == 2:
                        reason = f"卖出信号 (sell_id={sell_id}-{index_name}日线顶背离): {index_name}日线顶背离形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>85 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/2"
                        return SellSignal(
                            flag=SellFlag.SELL_HALF,
                            reason=reason,
                            daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi,
                            rsi_peak_date=rsi_peak_date,
                            rsi_peak_value=rsi_peak_value
                        )

                    # RSI>80 或 rsi_flag=1 → 卖出1/3
                    if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1'] or current_rsi_flag == 1:
                        reason = f"卖出信号 (sell_id={sell_id}-{index_name}日线顶背离): {index_name}日线顶背离形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>80 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/3"
                        return SellSignal(
                            flag=SellFlag.SELL_ONE_THIRD,
                            reason=reason,
                            daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi,
                            rsi_peak_date=rsi_peak_date,
                            rsi_peak_value=rsi_peak_value
                        )

            # 【优先级4】指数周线顶背离 + 个股周线RSI阶梯
            if has_index_weekly_div:
                index_div_info = index_weekly_div_info or {}
                index_div_date = index_div_info.get('date') if index_div_info else None

                # 校验：指数顶背离与个股RSI峰值间隔不超过5个工作日，否则顶背离失效
                days_diff = count_trading_days(index_div_date, rsi_peak_date)
                if days_diff > MAX_DIVERGENCE_RSI_PEAK_DAYS:
                    logger.debug(f"{date.strftime('%Y-%m-%d')} | {index_name}周线顶背离({index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'})与个股RSI峰值({rsi_peak_date.strftime('%Y-%m-%d')})间隔{days_diff}个工作日，超过{MAX_DIVERGENCE_RSI_PEAK_DAYS}天，顶背离失效")
                    # 间隔过大，跳过指数周线顶背离检查
                else:
                    # RSI>85 或 rsi_flag=2/3 → 清仓
                    if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2'] or current_rsi_flag >= 2:
                        reason = f"清仓信号 (sell_id={sell_id}-{index_name}周线顶背离): {index_name}周线顶背离形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>85 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议清仓"
                        return SellSignal(
                            flag=SellFlag.CLEAR_ALL,
                            reason=reason,
                            daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi,
                            rsi_peak_date=rsi_peak_date,
                            rsi_peak_value=rsi_peak_value
                        )

                    # RSI>80 或 rsi_flag=1 → 卖出1/2
                    if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1'] or current_rsi_flag == 1:
                        reason = f"卖出信号 (sell_id={sell_id}-{index_name}周线顶背离): {index_name}周线顶背离形成于 {index_div_date.strftime('%Y-%m-%d') if index_div_date else 'N/A'}, 个股周线RSI={weekly_rsi:.2f}>80 (峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/2"
                        return SellSignal(
                            flag=SellFlag.SELL_HALF,
                            reason=reason,
                            daily_rsi=daily_rsi,
                            weekly_rsi=weekly_rsi,
                            rsi_peak_date=rsi_peak_date,
                            rsi_peak_value=rsi_peak_value
                        )

            # 没有触发任何卖出信号
            return None

        # 【sell_id=1 和 sell_id=2】含RSI阶梯判断的完整逻辑
        # 按优先级检查卖出信号（使用保存的flag值判断）
        sell_flag = SellFlag.NO_SIGNAL
        reason = ""
        rsi_peak_date = None
        rsi_peak_value = 0.0
        divergence_date = None
        divergence_prev_high = 0.0

        # 优先级1：周线顶背离 + 死叉触发生效 → 清仓
        if current_weekly_div_flag == 1 and divergence_trigger:
            sell_flag = SellFlag.CLEAR_ALL
            div_info = current_weekly_div_info or {}
            divergence_date = div_info.get('date')
            divergence_prev_high_date = div_info.get('prev_high_date')
            divergence_prev_high = div_info.get('prev_high', 0.0)
            prev_high_date_str = divergence_prev_high_date.strftime('%Y-%m-%d') if divergence_prev_high_date else 'N/A'
            reason = f"清仓信号 (sell_id={sell_id}-周线顶背离): 周线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, 前高点{prev_high_date_str}, {trigger_name}触发生效, 前高={divergence_prev_high:.3f}, 建议清仓"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                divergence_date=divergence_date,
                divergence_prev_high=divergence_prev_high
            )

        # 优先级2：周线RSI>90 或 rsi_flag=3 → 清仓
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level3'] or current_rsi_flag == 3:
            sell_flag = SellFlag.CLEAR_ALL
            rsi_peak_date = current_rsi_peak_date or date
            rsi_peak_value = current_rsi_peak_value or weekly_rsi
            reason = f"清仓信号 (sell_id={sell_id}-清仓): 周线RSI>90 (RSI峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议清仓"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                rsi_peak_date=rsi_peak_date,
                rsi_peak_value=rsi_peak_value
            )

        # 优先级3：周线RSI>85 或 rsi_flag=2 → 卖出1/2
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2'] or current_rsi_flag == 2:
            sell_flag = SellFlag.SELL_HALF
            rsi_peak_date = current_rsi_peak_date or date
            rsi_peak_value = current_rsi_peak_value or weekly_rsi
            reason = f"卖出信号 (sell_id={sell_id}-阶梯): 周线RSI>85 (RSI峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/2"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                rsi_peak_date=rsi_peak_date,
                rsi_peak_value=rsi_peak_value
            )

        # 优先级4：日线顶背离 + 死叉触发生效 → 卖出1/3
        if current_daily_div_flag == 1 and divergence_trigger:
            sell_flag = SellFlag.SELL_ONE_THIRD
            div_info = current_daily_div_info or {}
            divergence_date = div_info.get('date')
            divergence_prev_high_date = div_info.get('prev_high_date')
            divergence_prev_high = div_info.get('prev_high', 0.0)
            prev_high_date_str = divergence_prev_high_date.strftime('%Y-%m-%d') if divergence_prev_high_date else 'N/A'
            reason = f"卖出信号 (sell_id={sell_id}-日线顶背离): 日线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, 前高点{prev_high_date_str}, {trigger_name}触发生效, 前高={divergence_prev_high:.3f}, 建议卖出1/3"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                divergence_date=divergence_date,
                divergence_prev_high=divergence_prev_high
            )

        # 优先级5：周线RSI>80 或 rsi_flag=1 → 卖出1/3
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1'] or current_rsi_flag == 1:
            sell_flag = SellFlag.SELL_ONE_THIRD
            rsi_peak_date = current_rsi_peak_date or date
            rsi_peak_value = current_rsi_peak_value or weekly_rsi
            reason = f"卖出信号 (sell_id={sell_id}-阶梯): 周线RSI>80 (RSI峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议卖出1/3"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                rsi_peak_date=rsi_peak_date,
                rsi_peak_value=rsi_peak_value
            )

        return None

    def check_buy_signal(
        self,
        date: pd.Timestamp,
        daily_rsi: float,
        weekly_rsi: float,
        state: StrategyState,
        has_new_cash: bool,
        has_sold_cash: bool,
        index_daily_rsi: float = np.nan,
        sh_index_daily_rsi: float = np.nan,
        kc_index_daily_rsi: float = np.nan,
        monthly_rsi: float = np.nan,
        bj_index_daily_rsi: float = np.nan,
        hsTech_index_daily_rsi: float = np.nan,
        hsIndex_index_daily_rsi: float = np.nan
    ) -> Optional[BuySignal]:
        """
        检查买入信号

        Args:
            date: 当前日期
            daily_rsi: 当日RSI值
            weekly_rsi: 当日周线RSI值
            monthly_rsi: 当日月线RSI值（用于buy_id=10,11,15-21,24-26）
            state: 当前策略状态
            has_new_cash: 是否有新资金（未进入市场的资金）
            has_sold_cash: 是否有卖出后待买回的资金
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和buy_id=5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6、7、10）
            kc_index_daily_rsi: 科创综指日线RSI值（用于buy_id=8）
            bj_index_daily_rsi: 北证板块日线RSI值（用于buy_id=12）
            hsTech_index_daily_rsi: 恒生科技ETF日线RSI值（用于buy_id=13）
            hsIndex_index_daily_rsi: 恒生指数ETF日线RSI值（用于buy_id=14）
            monthly_rsi: 当日月线RSI值（用于buy_id=10,11,15-21,24-26）
            state: 当前策略状态
            has_new_cash: 是否有新资金（未进入市场的资金）
            has_sold_cash: 是否有卖出后待买回的资金
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和buy_id=5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6、7、10）
            kc_index_daily_rsi: 科创综指日线RSI值（用于buy_id=8）

        Returns:
            BuySignal 或 None（无信号）

        买入规则：
        - judge_buy_ids触发: 买入所有可用资金（新资金 + 卖出后待买回资金），无条件限制
        - judge_t_ids触发: 只买回卖出后待买回的资金，不买入新资金

        【重要说明】
        judge_buy_ids 是最高优先级买入信号，只要满足条件且有可用资金（新资金或卖出资金）就触发。
        """
        # 检查无效值
        if np.isnan(daily_rsi) or np.isnan(weekly_rsi):
            return None

        # 检查是否有可用资金（新资金或卖出后待买回的资金）
        has_available_cash = has_new_cash or has_sold_cash
        if not has_available_cash:
            return None  # 没有任何可用资金，无法买入

        # 首先检查 judge_buy_ids（最高优先级，买入所有可用资金）
        for buy_id in self.buy_ids:
            condition_met, condition_desc = self._check_single_buy_condition_with_desc(buy_id, daily_rsi, weekly_rsi, state, index_daily_rsi, sh_index_daily_rsi, kc_index_daily_rsi, monthly_rsi, bj_index_daily_rsi, hsTech_index_daily_rsi, hsIndex_index_daily_rsi)
            if condition_met:
                # judge_buy_ids触发：买入所有可用资金（新资金 + 卖出资金）
                return BuySignal(
                    triggered=True,
                    is_new_cash=True,  # 标记为包含新资金，会买入所有可用资金
                    reason=f"买入信号 [{condition_desc}], 买入所有可用资金",
                    daily_rsi=daily_rsi,
                    weekly_rsi=weekly_rsi,
                    monthly_rsi=monthly_rsi
                )

        # 然后检查 judge_t_ids（只买回卖出资金，不买入新资金）
        # 注意：只有在有卖出资金且没有新资金时才检查 t_ids
        # 如果有新资金，应该等待 judge_buy_ids 触发而不是用 t_ids
        if has_sold_cash and not has_new_cash:
            for t_id in self.t_ids:
                condition_met, condition_desc = self._check_single_buy_condition_with_desc(t_id, daily_rsi, weekly_rsi, state, index_daily_rsi, sh_index_daily_rsi, kc_index_daily_rsi, monthly_rsi, bj_index_daily_rsi, hsTech_index_daily_rsi, hsIndex_index_daily_rsi)
                if condition_met:
                    # judge_t_ids触发：只买回卖出资金
                    return BuySignal(
                        triggered=True,
                        is_new_cash=False,  # 不包含新资金，只买回卖出资金
                        reason=f"做T买回信号 [{condition_desc}], 只买回卖出资金",
                        daily_rsi=daily_rsi,
                        weekly_rsi=weekly_rsi,
                        monthly_rsi=monthly_rsi
                    )

        return None

    def _check_single_buy_condition_with_desc(self, buy_id: int, daily_rsi: float, weekly_rsi: float, state: StrategyState, index_daily_rsi: float = np.nan, sh_index_daily_rsi: float = np.nan, kc_index_daily_rsi: float = np.nan, monthly_rsi: float = np.nan, bj_index_daily_rsi: float = np.nan, hsTech_index_daily_rsi: float = np.nan, hsIndex_index_daily_rsi: float = np.nan) -> Tuple[bool, str]:
        """
        检查单个买入ID的条件并返回条件描述

        Args:
            buy_id: 买入策略ID
            daily_rsi: 日线RSI值
            weekly_rsi: 周线RSI值
            monthly_rsi: 月线RSI值（用于buy_id=10,11,15-21,24-26）
            state: 当前策略状态（用于buy_id=9底背离判断）
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和buy_id=5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6、7、10）
            kc_index_daily_rsi: 科创综指日线RSI值（用于buy_id=8）
            bj_index_daily_rsi: 北证板块日线RSI值（用于buy_id=12）
            hsTech_index_daily_rsi: 恒生科技ETF日线RSI值（用于buy_id=13）
            hsIndex_index_daily_rsi: 恒生指数ETF日线RSI值（用于buy_id=14）

        Returns:
            Tuple[bool, str]: (是否满足条件, 条件描述)
        """
        if buy_id == 1:
            # 保守型：日RSI<20 且 周RSI<25
            if daily_rsi < 20 and weekly_rsi < 25:
                return True, f"buy_id=1(保守型): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<25"
            return False, ""

        elif buy_id == 2:
            # 标准型：日RSI<25 且 周RSI<30
            if daily_rsi < 25 and weekly_rsi < 30:
                return True, f"buy_id=2(标准型): 日RSI={daily_rsi:.2f}<25 且 周RSI={weekly_rsi:.2f}<30"
            return False, ""

        elif buy_id == 3:
            # 指数保护型：创业板指数日RSI<25
            if np.isnan(index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少创业板指数RSI数据")
                return False, ""
            if index_daily_rsi < 25:
                return True, f"buy_id=3(指数保护型): 创业板指数日RSI={index_daily_rsi:.2f}<25"
            return False, ""

        elif buy_id == 4:
            # 极度保守型：日RSI<20 且 周RSI<20
            if daily_rsi < 20 and weekly_rsi < 20:
                return True, f"buy_id=4(极度保守型): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<20"
            return False, ""

        elif buy_id == 5:
            # 指数保护型：创业板指数日RSI<20
            if np.isnan(index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少创业板指数RSI数据")
                return False, ""
            if index_daily_rsi < 20:
                return True, f"buy_id=5(指数保护型): 创业板指数日RSI={index_daily_rsi:.2f}<20"
            return False, ""

        elif buy_id == 6:
            # 上证指数保护型：上证指数(000001)日RSI<20
            if np.isnan(sh_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少上证指数RSI数据")
                return False, ""
            if sh_index_daily_rsi < 20:
                return True, f"buy_id=6(上证指数保护型): 上证指数日RSI={sh_index_daily_rsi:.2f}<20"
            return False, ""

        elif buy_id == 7:
            # 上证指数保护型：上证指数(000001)日RSI<25
            if np.isnan(sh_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少上证指数RSI数据")
                return False, ""
            if sh_index_daily_rsi < 25:
                return True, f"buy_id=7(上证指数保护型): 上证指数日RSI={sh_index_daily_rsi:.2f}<25"
            return False, ""

        elif buy_id == 8:
            # 科创板买入：科创综指(000680)日RSI<20
            if np.isnan(kc_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少科创综指RSI数据")
                return False, ""
            if kc_index_daily_rsi < 20:
                return True, f"buy_id=8(科创板买入): 科创综指日RSI={kc_index_daily_rsi:.2f}<20"
            return False, ""

        elif buy_id == 9:
            # 个股底背离买入：日线底背离形成
            if state.daily_bottom_divergence_flag == 1:
                div_info = state.daily_bottom_divergence_info or {}
                div_date = div_info.get('date')
                div_prev_low = div_info.get('prev_low', 0.0)
                return True, f"buy_id=9(底背离买入): 日线底背离形成于 {div_date.strftime('%Y-%m-%d') if div_date else 'N/A'}, 前低={div_prev_low:.3f}"
            return False, ""

        elif buy_id == 10:
            # 红利股买点：上证指数日线RSI<20，且个股月线RSI<30
            if np.isnan(sh_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少上证指数RSI数据")
                return False, ""
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if sh_index_daily_rsi < 20 and monthly_rsi < 30:
                return True, f"buy_id=10(红利股买点): 上证指数日RSI={sh_index_daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<30"
            return False, ""

        elif buy_id == 11:
            # 个股底背离买入加强版：日RSI<20 且 周RSI<20，且月线RSI<30
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 30:
                return True, f"buy_id=11(底背离加强): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<30"
            return False, ""

        elif buy_id == 12:
            # 北证板块买入：北证板块(899050)日RSI<20
            if np.isnan(bj_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少北证板块RSI数据")
                return False, ""
            if bj_index_daily_rsi < 20:
                return True, f"buy_id=12(北证板块买入): 北证板块日RSI={bj_index_daily_rsi:.2f}<20"
            return False, ""

        elif buy_id == 13:
            # 恒生科技板块买入：恒生科技ETF(513180)日RSI<20
            if np.isnan(hsTech_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少恒生科技板块RSI数据")
                return False, ""
            if hsTech_index_daily_rsi < 20:
                return True, f"buy_id=13(恒生科技买入): 恒生科技ETF日RSI={hsTech_index_daily_rsi:.2f}<20"
            return False, ""

        elif buy_id == 14:
            # 恒生指数板块买入：恒生指数ETF(159920)日RSI<20
            if np.isnan(hsIndex_index_daily_rsi):
                logger.warning(f"buy_id={buy_id} 缺少恒生指数板块RSI数据")
                return False, ""
            if hsIndex_index_daily_rsi < 20:
                return True, f"buy_id=14(恒生指数买入): 恒生指数ETF日RSI={hsIndex_index_daily_rsi:.2f}<20"
            return False, ""

        elif buy_id == 15:
            # 极度保守型：日RSI<20 且 月RSI<22
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and monthly_rsi < 22:
                return True, f"buy_id=15(极度保守型): 日RSI={daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<22"
            return False, ""

        elif buy_id == 16:
            # 极度保守型：日RSI<20 且 月RSI<23
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and monthly_rsi < 23:
                return True, f"buy_id=16(极度保守型): 日RSI={daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<23"
            return False, ""

        elif buy_id == 17:
            # 极度保守型：日RSI<20 且 月RSI<20
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and monthly_rsi < 20:
                return True, f"buy_id=17(极度保守型): 日RSI={daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<20"
            return False, ""

        elif buy_id == 18:
            # 月线宽松+日周极度超卖：月RSI<40 且 日RSI<20 且 周RSI<20
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 40:
                return True, f"buy_id=18(月线宽松+日周超卖): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<40"
            return False, ""

        elif buy_id == 19:
            # 三周期共振极度超卖：日RSI<20 且 周RSI<20 且 月RSI<15
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 15:
                return True, f"buy_id=19(三周期共振超卖): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<15"
            return False, ""

        elif buy_id == 20:
            # 日周超卖+月线适中：日RSI<20 且 周RSI<20 且 月RSI<25
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 25:
                return True, f"buy_id=20(日周超卖+月线适中): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<25"
            return False, ""

        elif buy_id == 21:
            # 三周期共振超卖：日RSI<20 且 周RSI<20 且 月RSI<20
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and weekly_rsi < 20 and monthly_rsi < 20:
                return True, f"buy_id=21(三周期共振超卖): 日RSI={daily_rsi:.2f}<20 且 周RSI={weekly_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<20"
            return False, ""

        elif buy_id == 22:
            # 日线宽松超卖：日RSI<30
            if daily_rsi < 30:
                return True, f"buy_id=22(日线宽松超卖): 日RSI={daily_rsi:.2f}<30"
            return False, ""

        elif buy_id == 23:
            # 周线超卖：周RSI<20
            if weekly_rsi < 20:
                return True, f"buy_id=23(周线超卖): 周RSI={weekly_rsi:.2f}<20"
            return False, ""

        elif buy_id == 24:
            # 日线+月线超卖：日RSI<20 且 月RSI<15
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and monthly_rsi < 15:
                return True, f"buy_id=24(日线+月线超卖): 日RSI={daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<15"
            return False, ""

        elif buy_id == 25:
            # 日线超卖+月线宽松：日RSI<20 且 月RSI<40
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and monthly_rsi < 40:
                return True, f"buy_id=25(日线超卖+月线宽松): 日RSI={daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<40"
            return False, ""

        elif buy_id == 26:
            # 日线超卖+月线适中：日RSI<20 且 月RSI<25
            if np.isnan(monthly_rsi):
                logger.warning(f"buy_id={buy_id} 缺少月线RSI数据")
                return False, ""
            if daily_rsi < 20 and monthly_rsi < 25:
                return True, f"buy_id=26(日线超卖+月线适中): 日RSI={daily_rsi:.2f}<20 且 月RSI={monthly_rsi:.2f}<25"
            return False, ""

        return False, ""

    def _check_single_buy_condition(self, buy_id: int, daily_rsi: float, weekly_rsi: float, state: StrategyState, index_daily_rsi: float = np.nan, sh_index_daily_rsi: float = np.nan, kc_index_daily_rsi: float = np.nan, monthly_rsi: float = np.nan, bj_index_daily_rsi: float = np.nan, hsTech_index_daily_rsi: float = np.nan, hsIndex_index_daily_rsi: float = np.nan) -> bool:
        """
        检查单个买入ID的条件

        Args:
            buy_id: 买入策略ID
            daily_rsi: 日线RSI值
            weekly_rsi: 周线RSI值
            monthly_rsi: 月线RSI值（用于buy_id=10,11,15-21,24-26）
            state: 当前策略状态（用于buy_id=9底背离判断）
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6、7、10）
            kc_index_daily_rsi: 科创综指日线RSI值（用于buy_id=8）
            bj_index_daily_rsi: 北证板块日线RSI值（用于buy_id=12）
            hsTech_index_daily_rsi: 恒生科技ETF日线RSI值（用于buy_id=13）
            hsIndex_index_daily_rsi: 恒生指数ETF日线RSI值（用于buy_id=14）

        Returns:
            bool: 是否满足买入条件
        """
        condition_met, _ = self._check_single_buy_condition_with_desc(buy_id, daily_rsi, weekly_rsi, state, index_daily_rsi, sh_index_daily_rsi, kc_index_daily_rsi, monthly_rsi, bj_index_daily_rsi)
        return condition_met

    def update_rsi_flag(
        self,
        weekly_rsi: float,
        state: StrategyState,
        date: pd.Timestamp,
        macd_dif: float = np.nan,
        macd_dea: float = np.nan
    ) -> None:
        """
        更新RSI警戒级别

        Args:
            weekly_rsi: 当日周线RSI值
            state: 当前策略状态
            date: 当前日期
            macd_dif: MACD DIF线（黄线）值
            macd_dea: MACD DEA线（黑线）值

        【关键逻辑】
        当MACD处于下跌趋势（DIF < DEA，即黄线在黑线下方）时，
        RSI峰值和flag不更新，避免在下跌趋势中产生无效的卖出信号。
        """
        if np.isnan(weekly_rsi):
            return

        # 【核心逻辑】MACD下跌趋势判断
        # 如果黄线(DIF)在黑线(DEA)下面，说明处于下跌趋势
        # 此时不更新RSI峰值和flag（峰值保留，等待死叉触发时清除）
        if not np.isnan(macd_dif) and not np.isnan(macd_dea):
            if macd_dif < macd_dea:
                # 下跌趋势中，不更新RSI峰值和flag
                return

        # 更新峰值记录（只在警戒区域内更新峰值）
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1']:
            if weekly_rsi > state.rsi_peak_value:
                state.rsi_peak_date = date
                state.rsi_peak_value = weekly_rsi

        # 更新flag（取最高级别，只在警戒区域内设置）
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level3']:
            state.rsi_flag = 3
        elif weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2']:
            if state.rsi_flag < 2:
                state.rsi_flag = 2
        elif weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1']:
            if state.rsi_flag < 1:
                state.rsi_flag = 1

    def reset_after_sell(
        self,
        state: StrategyState,
        sell_flag: SellFlag,
        date: pd.Timestamp,
        is_fake_sell: bool = False
    ) -> None:
        """
        卖出后重置状态

        Args:
            state: 当前策略状态
            sell_flag: 卖出级别
            date: 卖出日期
            is_fake_sell: 是否为假卖出（无实际交易）

        注意：
        - 假卖出不设置 last_sell_date，因为没有实际交易，不应触发买入冷却期
        - 假卖出也要重置所有背离标志，避免信号残留
        """
        # 重置RSI相关标志
        state.rsi_flag = 0
        state.rsi_peak_value = 0.0
        state.rsi_peak_date = None

        # 重置所有背离标志（不管是什么级别的卖出，都要重置）
        # 避免背离信号长期残留导致后续误触发
        state.weekly_divergence_flag = 0
        state.weekly_divergence_info = None
        state.daily_divergence_flag = 0
        state.daily_divergence_info = None
        state.daily_bottom_divergence_flag = 0
        state.daily_bottom_divergence_info = None

        # 重置指数背离状态（用于 sell_id=4,5,6）
        state.sh_index_daily_divergence_flag = 0
        state.sh_index_weekly_divergence_flag = 0
        state.sh_index_daily_divergence_info = None
        state.sh_index_weekly_divergence_info = None
        state.cyb_index_daily_divergence_flag = 0
        state.cyb_index_weekly_divergence_flag = 0
        state.cyb_index_daily_divergence_info = None
        state.cyb_index_weekly_divergence_info = None
        state.kc_index_daily_divergence_flag = 0
        state.kc_index_weekly_divergence_flag = 0
        state.kc_index_daily_divergence_info = None
        state.kc_index_weekly_divergence_info = None

        # 重置新指数背离状态（用于 sell_id=13,14,15）
        state.sh50_index_daily_divergence_flag = 0
        state.sh50_index_weekly_divergence_flag = 0
        state.sh50_index_daily_divergence_info = None
        state.sh50_index_weekly_divergence_info = None
        state.cyb50_index_daily_divergence_flag = 0
        state.cyb50_index_weekly_divergence_flag = 0
        state.cyb50_index_daily_divergence_info = None
        state.cyb50_index_weekly_divergence_info = None
        state.kc50_index_daily_divergence_flag = 0
        state.kc50_index_weekly_divergence_flag = 0
        state.kc50_index_daily_divergence_info = None
        state.kc50_index_weekly_divergence_info = None

        # 记录卖出日期（用于买入冷却期）
        # 假卖出不设置 last_sell_date，因为没有实际交易
        if not is_fake_sell:
            state.last_sell_date = date

    def reset_after_buy(self, state: StrategyState) -> None:
        """
        买入后重置状态

        【关键逻辑】
        买入后清除底背离标志，避免重复触发买入信号。
        """
        # 清除底背离标志
        state.daily_bottom_divergence_flag = 0
        state.daily_bottom_divergence_info = None

    def set_daily_divergence(
        self,
        state: StrategyState,
        divergence_info: Dict,
        macd_dif: float = np.nan,
        macd_dea: float = np.nan
    ) -> None:
        """
        设置日线顶背离状态

        Args:
            state: 当前策略状态
            divergence_info: 背离信息字典
            macd_dif: MACD DIF线（黄线）值
            macd_dea: MACD DEA线（黑线）值

        【关键逻辑】
        如果MACD处于下跌趋势（DIF < DEA，即黄线在黑线下方），
        则背离信号不生效，不设置背离标志。
        """
        # MACD下跌趋势判断：黄线在黑线下方，背离不生效
        if not np.isnan(macd_dif) and not np.isnan(macd_dea):
            if macd_dif < macd_dea:
                # 下跌趋势中，不设置背离标志
                return

        state.daily_divergence_flag = 1
        state.daily_divergence_info = divergence_info

    def set_weekly_divergence(
        self,
        state: StrategyState,
        divergence_info: Dict,
        macd_dif: float = np.nan,
        macd_dea: float = np.nan
    ) -> None:
        """
        设置周线顶背离状态

        Args:
            state: 当前策略状态
            divergence_info: 背离信息字典
            macd_dif: MACD DIF线（黄线）值
            macd_dea: MACD DEA线（黑线）值

        【关键逻辑】
        如果MACD处于下跌趋势（DIF < DEA，即黄线在黑线下方），
        则背离信号不生效，不设置背离标志。
        """
        # MACD下跌趋势判断：黄线在黑线下方，背离不生效
        if not np.isnan(macd_dif) and not np.isnan(macd_dea):
            if macd_dif < macd_dea:
                # 下跌趋势中，不设置背离标志
                return

        state.weekly_divergence_flag = 1
        state.weekly_divergence_info = divergence_info

    def set_daily_bottom_divergence(
        self,
        state: StrategyState,
        divergence_info: Dict
    ) -> None:
        """
        设置日线底背离状态

        Args:
            state: 当前策略状态
            divergence_info: 背离信息字典

        【关键逻辑】
        底背离用于买入信号判断，当个股形成日线级别底背离时触发买入。
        """
        state.daily_bottom_divergence_flag = 1
        state.daily_bottom_divergence_info = divergence_info

    def clear_daily_bottom_divergence(self, state: StrategyState) -> None:
        """
        清除日线底背离状态

        Args:
            state: 当前策略状态

        【使用场景】
        当买入信号触发后，清除底背离标志，避免重复触发。
        """
        state.daily_bottom_divergence_flag = 0
        state.daily_bottom_divergence_info = None

    # ==================== 指数顶背离设置方法（用于 sell_id=4,5,6）====================

    def set_sh_index_divergence(
        self,
        state: StrategyState,
        daily_info: Optional[Dict] = None,
        weekly_info: Optional[Dict] = None
    ) -> None:
        """
        设置上证指数顶背离状态

        Args:
            state: 当前策略状态
            daily_info: 日线顶背离信息字典（可选）
            weekly_info: 周线顶背离信息字典（可选）

        【使用场景】
        用于 sell_id=4 策略，当上证指数出现顶背离时设置状态，
        后续在 MACD 日线跌破死叉时触发卖出判断。
        """
        if daily_info is not None:
            state.sh_index_daily_divergence_flag = 1
            state.sh_index_daily_divergence_info = daily_info
        if weekly_info is not None:
            state.sh_index_weekly_divergence_flag = 1
            state.sh_index_weekly_divergence_info = weekly_info

    def set_cyb_index_divergence(
        self,
        state: StrategyState,
        daily_info: Optional[Dict] = None,
        weekly_info: Optional[Dict] = None
    ) -> None:
        """
        设置创业板指数顶背离状态

        Args:
            state: 当前策略状态
            daily_info: 日线顶背离信息字典（可选）
            weekly_info: 周线顶背离信息字典（可选）

        【使用场景】
        用于 sell_id=5 策略，当创业板指数出现顶背离时设置状态，
        后续在 MACD 日线跌破死叉时触发卖出判断。
        """
        if daily_info is not None:
            state.cyb_index_daily_divergence_flag = 1
            state.cyb_index_daily_divergence_info = daily_info
        if weekly_info is not None:
            state.cyb_index_weekly_divergence_flag = 1
            state.cyb_index_weekly_divergence_info = weekly_info

    def set_kc_index_divergence(
        self,
        state: StrategyState,
        daily_info: Optional[Dict] = None,
        weekly_info: Optional[Dict] = None
    ) -> None:
        """
        设置科创板指数顶背离状态

        Args:
            state: 当前策略状态
            daily_info: 日线顶背离信息字典（可选）
            weekly_info: 周线顶背离信息字典（可选）

        【使用场景】
        用于 sell_id=6 策略，当科创板指数出现顶背离时设置状态，
        后续在 MACD 日线跌破死叉时触发卖出判断。
        """
        if daily_info is not None:
            state.kc_index_daily_divergence_flag = 1
            state.kc_index_daily_divergence_info = daily_info
        if weekly_info is not None:
            state.kc_index_weekly_divergence_flag = 1
            state.kc_index_weekly_divergence_info = weekly_info

    def clear_sh_index_divergence(self, state: StrategyState) -> None:
        """
        清除上证指数顶背离状态

        Args:
            state: 当前策略状态
        """
        state.sh_index_daily_divergence_flag = 0
        state.sh_index_weekly_divergence_flag = 0
        state.sh_index_daily_divergence_info = None
        state.sh_index_weekly_divergence_info = None

    def clear_cyb_index_divergence(self, state: StrategyState) -> None:
        """
        清除创业板指数顶背离状态

        Args:
            state: 当前策略状态
        """
        state.cyb_index_daily_divergence_flag = 0
        state.cyb_index_weekly_divergence_flag = 0
        state.cyb_index_daily_divergence_info = None
        state.cyb_index_weekly_divergence_info = None

    def clear_kc_index_divergence(self, state: StrategyState) -> None:
        """
        清除科创板指数顶背离状态

        Args:
            state: 当前策略状态
        """
        state.kc_index_daily_divergence_flag = 0
        state.kc_index_weekly_divergence_flag = 0
        state.kc_index_daily_divergence_info = None
        state.kc_index_weekly_divergence_info = None

    # ==================== 新指数顶背离设置方法（用于 sell_id=13,14,15）====================

    def set_sh50_index_divergence(
        self,
        state: StrategyState,
        daily_info: Optional[Dict] = None,
        weekly_info: Optional[Dict] = None
    ) -> None:
        """
        设置上证50指数顶背离状态

        Args:
            state: 当前策略状态
            daily_info: 日线顶背离信息字典（可选）
            weekly_info: 周线顶背离信息字典（可选）

        【使用场景】
        用于 sell_id=13 策略，当上证50指数出现顶背离时设置状态，
        后续在 MACD 日线跌破死叉时触发卖出判断。
        """
        if daily_info is not None:
            state.sh50_index_daily_divergence_flag = 1
            state.sh50_index_daily_divergence_info = daily_info
        if weekly_info is not None:
            state.sh50_index_weekly_divergence_flag = 1
            state.sh50_index_weekly_divergence_info = weekly_info

    def set_cyb50_index_divergence(
        self,
        state: StrategyState,
        daily_info: Optional[Dict] = None,
        weekly_info: Optional[Dict] = None
    ) -> None:
        """
        设置创业板50指数顶背离状态

        Args:
            state: 当前策略状态
            daily_info: 日线顶背离信息字典（可选）
            weekly_info: 周线顶背离信息字典（可选）

        【使用场景】
        用于 sell_id=14 策略，当创业板50指数出现顶背离时设置状态，
        后续在 MACD 日线跌破死叉时触发卖出判断。
        """
        if daily_info is not None:
            state.cyb50_index_daily_divergence_flag = 1
            state.cyb50_index_daily_divergence_info = daily_info
        if weekly_info is not None:
            state.cyb50_index_weekly_divergence_flag = 1
            state.cyb50_index_weekly_divergence_info = weekly_info

    def set_kc50_index_divergence(
        self,
        state: StrategyState,
        daily_info: Optional[Dict] = None,
        weekly_info: Optional[Dict] = None
    ) -> None:
        """
        设置科创板50指数顶背离状态

        Args:
            state: 当前策略状态
            daily_info: 日线顶背离信息字典（可选）
            weekly_info: 周线顶背离信息字典（可选）

        【使用场景】
        用于 sell_id=15 策略，当科创板50指数出现顶背离时设置状态，
        后续在 MACD 日线跌破死叉时触发卖出判断。
        """
        if daily_info is not None:
            state.kc50_index_daily_divergence_flag = 1
            state.kc50_index_daily_divergence_info = daily_info
        if weekly_info is not None:
            state.kc50_index_weekly_divergence_flag = 1
            state.kc50_index_weekly_divergence_info = weekly_info

    def clear_sh50_index_divergence(self, state: StrategyState) -> None:
        """
        清除上证50指数顶背离状态

        Args:
            state: 当前策略状态
        """
        state.sh50_index_daily_divergence_flag = 0
        state.sh50_index_weekly_divergence_flag = 0
        state.sh50_index_daily_divergence_info = None
        state.sh50_index_weekly_divergence_info = None

    def clear_cyb50_index_divergence(self, state: StrategyState) -> None:
        """
        清除创业板50指数顶背离状态

        Args:
            state: 当前策略状态
        """
        state.cyb50_index_daily_divergence_flag = 0
        state.cyb50_index_weekly_divergence_flag = 0
        state.cyb50_index_daily_divergence_info = None
        state.cyb50_index_weekly_divergence_info = None

    def clear_kc50_index_divergence(self, state: StrategyState) -> None:
        """
        清除科创板50指数顶背离状态

        Args:
            state: 当前策略状态
        """
        state.kc50_index_daily_divergence_flag = 0
        state.kc50_index_weekly_divergence_flag = 0
        state.kc50_index_daily_divergence_info = None
        state.kc50_index_weekly_divergence_info = None

    def __repr__(self) -> str:
        """类实例的字符串表示"""
        return f"TradingStrategy(buy_ids={self.buy_ids}, t_ids={self.t_ids}, sell_ids={self.sell_ids})"