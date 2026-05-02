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

# 卖出后冷却期天数
COOLDOWN_DAYS = 5


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
    last_sell_date: Optional[pd.Timestamp] = None   # 上次卖出日期


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
            if buy_id not in [1, 2, 3, 4, 5, 6, 7]:
                raise ValueError(f"buy_id 必须为 1-7，当前值: {buy_id}")
        for t_id in t_ids:
            if t_id not in [1, 2, 3, 4, 5, 6, 7]:
                raise ValueError(f"t_id 必须为 1-7，当前值: {t_id}")

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
        sell_id: int = 1
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
            sell_id: 卖出策略ID（1=SAR死叉触发，2=MACD跌破死叉触发）

        Returns:
            SellSignal 或 None（无信号）

        sell_id 说明:
        - sell_id=1: 触发条件为SAR死叉跌破（收盘价跌破SAR）
        - sell_id=2: 触发条件为MACD日线跌破死叉（MACD死叉 + 最低价跌破SAR）
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
        else:
            # 默认使用SAR死叉跌破
            trigger_condition = sar_cross_down
            trigger_name = "SAR死叉跌破"

        # 必须满足触发条件才继续检查
        if not trigger_condition:
            return None

        # 按优先级检查卖出信号
        sell_flag = SellFlag.NO_SIGNAL
        reason = ""
        rsi_peak_date = None
        rsi_peak_value = 0.0
        divergence_date = None
        divergence_prev_high = 0.0

        # 优先级1：周线顶背离 + 死叉触发生效 → 清仓
        # 根据 sell_id 选择触发条件：sell_id=1 用 SAR死叉，sell_id=2 用 MACD死叉
        divergence_trigger = sar_cross_down if sell_id == 1 else macd_cross_down
        if state.weekly_divergence_flag == 1 and divergence_trigger:
            sell_flag = SellFlag.CLEAR_ALL
            div_info = state.weekly_divergence_info or {}
            divergence_date = div_info.get('date')
            divergence_prev_high = div_info.get('prev_high', 0.0)
            reason = f"清仓信号 (sell_id={sell_id}-周线顶背离): 周线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, {trigger_name}触发生效, 前高={divergence_prev_high:.3f}, 建议清仓"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                divergence_date=divergence_date,
                divergence_prev_high=divergence_prev_high
            )

        # 优先级2：周线RSI>90 → 清仓
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level3'] or state.rsi_flag == 3:
            sell_flag = SellFlag.CLEAR_ALL
            rsi_peak_date = state.rsi_peak_date or date
            rsi_peak_value = state.rsi_peak_value or weekly_rsi
            reason = f"清仓信号 (sell_id={sell_id}-清仓): 周线RSI>90 (RSI峰值:{rsi_peak_date.strftime('%Y-%m-%d')}={rsi_peak_value:.2f}), {trigger_name}触发, 建议清仓"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                rsi_peak_date=rsi_peak_date,
                rsi_peak_value=rsi_peak_value
            )

        # 优先级3：周线RSI>85 → 卖出1/2
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level2'] or state.rsi_flag == 2:
            sell_flag = SellFlag.SELL_HALF
            rsi_peak_date = state.rsi_peak_date or date
            rsi_peak_value = state.rsi_peak_value or weekly_rsi
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
        # 根据 sell_id 选择触发条件：sell_id=1 用 SAR死叉，sell_id=2 用 MACD死叉
        if state.daily_divergence_flag == 1 and divergence_trigger:
            sell_flag = SellFlag.SELL_ONE_THIRD
            div_info = state.daily_divergence_info or {}
            divergence_date = div_info.get('date')
            divergence_prev_high = div_info.get('prev_high', 0.0)
            reason = f"卖出信号 (sell_id={sell_id}-日线顶背离): 日线顶背离形成于 {divergence_date.strftime('%Y-%m-%d') if divergence_date else 'N/A'}, {trigger_name}触发生效, 前高={divergence_prev_high:.3f}, 建议卖出1/3"
            return SellSignal(
                flag=sell_flag,
                reason=reason,
                daily_rsi=daily_rsi,
                weekly_rsi=weekly_rsi,
                divergence_date=divergence_date,
                divergence_prev_high=divergence_prev_high
            )

        # 优先级5：周线RSI>80 → 卖出1/3
        if weekly_rsi >= RSI_THRESHOLDS['weekly_sell_level1'] or state.rsi_flag == 1:
            sell_flag = SellFlag.SELL_ONE_THIRD
            rsi_peak_date = state.rsi_peak_date or date
            rsi_peak_value = state.rsi_peak_value or weekly_rsi
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
        sh_index_daily_rsi: float = np.nan
    ) -> Optional[BuySignal]:
        """
        检查买入信号

        Args:
            date: 当前日期
            daily_rsi: 当日RSI值
            weekly_rsi: 当日周线RSI值
            state: 当前策略状态
            has_new_cash: 是否有新资金（未进入市场的资金）
            has_sold_cash: 是否有卖出后待买回的资金
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和buy_id=5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6和buy_id=7）

        Returns:
            BuySignal 或 None（无信号）

        买入规则：
        - judge_buy_ids触发: 一次性买入新资金 + 买回之前卖出的资金
        - judge_t_ids触发: 只买回之前卖出的资金，不买入新资金
        """
        # 检查卖出冷却期
        if state.last_sell_date is not None:
            days_since_sell = (date - state.last_sell_date).days
            if days_since_sell < COOLDOWN_DAYS:
                return None

        # 检查无效值
        if np.isnan(daily_rsi) or np.isnan(weekly_rsi):
            return None

        # 首先检查 judge_buy_ids（优先级更高，买入新资金+买回卖出资金）
        if has_new_cash or has_sold_cash:
            for buy_id in self.buy_ids:
                condition_met, condition_desc = self._check_single_buy_condition_with_desc(buy_id, daily_rsi, weekly_rsi, index_daily_rsi, sh_index_daily_rsi)
                if condition_met:
                    # judge_buy_ids触发：买入新资金 + 买回卖出资金
                    return BuySignal(
                        triggered=True,
                        is_new_cash=True,  # 包含新资金买入
                        reason=f"买入信号 [{condition_desc}], 买入新资金+买回卖出资金",
                        daily_rsi=daily_rsi,
                        weekly_rsi=weekly_rsi
                    )

        # 然后检查 judge_t_ids（只买回卖出资金）
        if has_sold_cash:
            for t_id in self.t_ids:
                condition_met, condition_desc = self._check_single_buy_condition_with_desc(t_id, daily_rsi, weekly_rsi, index_daily_rsi, sh_index_daily_rsi)
                if condition_met:
                    # judge_t_ids触发：只买回卖出资金
                    return BuySignal(
                        triggered=True,
                        is_new_cash=False,  # 不包含新资金
                        reason=f"做T买回信号 [{condition_desc}], 只买回卖出资金",
                        daily_rsi=daily_rsi,
                        weekly_rsi=weekly_rsi
                    )

        return None

    def _check_single_buy_condition_with_desc(self, buy_id: int, daily_rsi: float, weekly_rsi: float, index_daily_rsi: float = np.nan, sh_index_daily_rsi: float = np.nan) -> Tuple[bool, str]:
        """
        检查单个买入ID的条件并返回条件描述

        Args:
            buy_id: 买入策略ID
            daily_rsi: 日线RSI值
            weekly_rsi: 周线RSI值
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6和7）

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

        return False, ""

    def _check_single_buy_condition(self, buy_id: int, daily_rsi: float, weekly_rsi: float, index_daily_rsi: float = np.nan) -> bool:
        """
        检查单个买入ID的条件

        Args:
            buy_id: 买入策略ID
            daily_rsi: 日线RSI值
            weekly_rsi: 周线RSI值
            index_daily_rsi: 创业板指数日线RSI值（用于buy_id=3和5）
            sh_index_daily_rsi: 上证指数日线RSI值（用于buy_id=6和7）

        Returns:
            bool: 是否满足买入条件
        """
        condition_met, _ = self._check_single_buy_condition_with_desc(buy_id, daily_rsi, weekly_rsi, index_daily_rsi, sh_index_daily_rsi)
        return condition_met

    def update_rsi_flag(
        self,
        weekly_rsi: float,
        state: StrategyState,
        date: pd.Timestamp
    ) -> None:
        """
        更新RSI警戒级别

        Args:
            weekly_rsi: 当日周线RSI值
            state: 当前策略状态
            date: 当前日期
        """
        if np.isnan(weekly_rsi):
            return

        # 更新RSI峰值记录
        if weekly_rsi > state.rsi_peak_value:
            state.rsi_peak_date = date
            state.rsi_peak_value = weekly_rsi

        # 更新flag（取最高级别）
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

        # 记录卖出日期（用于买入冷却期）
        # 假卖出不设置 last_sell_date，因为没有实际交易
        if not is_fake_sell:
            state.last_sell_date = date

    def reset_after_buy(self, state: StrategyState) -> None:
        """
        买入后重置状态（可选，视策略而定）
        """
        # 买入不重置其他状态，保持RSI和背离监控
        pass

    def set_daily_divergence(
        self,
        state: StrategyState,
        divergence_info: Dict
    ) -> None:
        """
        设置日线顶背离状态

        Args:
            state: 当前策略状态
            divergence_info: 背离信息字典
        """
        state.daily_divergence_flag = 1
        state.daily_divergence_info = divergence_info

    def set_weekly_divergence(
        self,
        state: StrategyState,
        divergence_info: Dict
    ) -> None:
        """
        设置周线顶背离状态

        Args:
            state: 当前策略状态
            divergence_info: 背离信息字典
        """
        state.weekly_divergence_flag = 1
        state.weekly_divergence_info = divergence_info

    def __repr__(self) -> str:
        """类实例的字符串表示"""
        return f"TradingStrategy(buy_ids={self.buy_ids}, t_ids={self.t_ids}, sell_ids={self.sell_ids})"