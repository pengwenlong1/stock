---
name: daily-monitoring
description: |
  量化交易每日监控代码规范。当编写、修改或审查监控相关代码时必须使用此skill。

  触发条件：
  - 用户提及"每日监控"、"实时监控"、"monitor"、"告警"、"钉钉通知"
  - 编写或修改 src/runner/monitor*day.py 相关文件
  - 编写或修改信号检测、背离判断、RSI监控相关代码
  - 实现告警机制、消息推送功能
  - 涉及SAR死叉跌破、均线死叉、局部高点检测

  此skill确保监控代码遵循量化开发铁律，杜绝未来函数，保证向量化计算。
---

# 每日监控代码规范

## 一、核心职责分离

监控代码必须遵循职责分离原则：

| 职责 | 处理模块 | 说明 |
|-----|---------|------|
| 策略逻辑 | `strategy.py` | 买入/卖出信号判断 |
| 指标计算 | `rsi_calculator.py`, `macd_calculator.py`, `sar_strategy.py` | 技术指标计算 |
| 背离检测 | `divergence_detector.py` | 顶背离/底背离判断 |
| 监控执行 | `monitor_by_day.py` | 数据获取、信号调用、告警发送 |

**监控脚本职责**：
- 数据准备（获取历史数据、计算指标）
- 调用策略类进行信号检测
- 执行告警（钉钉消息）
- **不包含策略逻辑本身**

### 策略规则参考文件

当需要了解具体的买入/卖出策略条件（buy_id、sell_id 的详细定义）时，请读取：
- **`.claude/common_skills_docu/buy_sell.md`** - 公共文档，包含所有买入卖出策略的完整规则说明

读取时机：
- 实现/修改买入卖出信号判断逻辑时
- 需要确认具体策略条件阈值时
- 日志输出需要包含策略详细信息时

## 二、监控核心逻辑

每日监控的核心流程遵循以下步骤：

### 2.1 监控触发流程

```
监控当天最新交易日的数据
    ↓
检测是否发生死叉（根据judge_sell_ids判断）
    ↓
[发生死叉] → 触发分析流程
    ↓
寻找前一次死叉到当前死叉之间的局部高点
    ↓
判断这些高点是否存在顶背离
    ↓
[存在顶背离] → 生成卖出告警
```

### 2.2 死叉类型判断

根据股票配置中的 `judge_sell_ids` 判断死叉类型：

| sell_id | 死叉类型 | 检测方法 |
|---------|----------|----------|
| sell_id=1 | SAR绿转红 | SAR值从价格下方转到上方 |
| sell_id=2,3 | MACD日线死叉 | DIF下穿DEA |
| sell_id=4,5,6 | MACD日线死叉 + 指数背离 | 个股MACD死叉 + 指数背离 |
| sell_id=7,8,9 | 指数MACD死叉 | 仅判断指数MACD死叉 |
| sell_id=10-15 | 个股顶背离 + 指数环境共振 | 详见下方规则 |

### sell_id=10-15 每日监控触发规则

**核心逻辑**：当个股形成顶背离时，根据个股和指数的MACD状态决定卖出触发时机。

#### 触发场景矩阵

| 场景 | 个股状态 | 指数状态 | 触发条件 | 监控行为 |
|-----|---------|---------|---------|---------|
| **(1)** | 个股顶背离 + 个股已处于死叉 | 不限 | **立即告警卖出** | 当日收盘后直接发送卖出告警 |
| **(2)** | 个股顶背离 + 个股未死叉 | 指数已处于死叉 | **等待个股死叉 或 指数重新死叉** | 设置预警状态，待个股死叉或指数重新死叉确认后告警 |
| **(3)** | 个股顶背离 + 个股未死叉 | 指数未死叉 | **等待任意死叉** | 设置预警状态，待个股或指数死叉后告警 |

#### 每日监控特殊处理

1. **场景(1)直接卖出**：监控当日发现个股顶背离 + 个股已处于死叉状态，立即发送钉钉告警
2. **场景(2)预警等待**：监控当日发现个股顶背离 + 指数已处于死叉（但个股未死叉），发送**提前预警**告警，等待个股死叉 或 指数重新出现死叉确认
3. **场景(3)预警等待**：监控当日发现个股顶背离 + 双方都未死叉，发送**提前预警**告警，等待任意一方死叉确认

#### 告警消息格式示例

```
# 场景(1): 直接卖出告警
2026-06-07 | 日RSI:65.30 | 周RSI:75.20 | 【股票名】卖出信号 (sell_id=10): 个股顶背离+个股已处于死叉状态，建议卖出1/3

# 场景(2): 提前预警告警
2026-06-07 | 日RSI:65.30 | 周RSI:75.20 | 【股票名】提前预警 (sell_id=10): 个股顶背离形成，上证指数已处于死叉状态，等待个股死叉或指数重新死叉确认卖出

# 场景(3): 提前预警告警
2026-06-07 | 日RSI:65.30 | 周RSI:75.20 | 【股票名】提前预警 (sell_id=10): 个股顶背离形成，等待个股或上证指数死叉确认卖出
```

### 2.3 局部高点查找（高点B与高点A判断方式不同）

**核心原则**：每日监控中，高点B和高点A采用不同的判断方式。

| 高点 | 判断方式 | 原因 |
|-----|---------|------|
| **高点B（最新局部高点）** | 直接找金叉买入后最高价点（不使用min_distance） | 实时监控需要快速定位，直接取最高价更直观 |
| **高点A（上一次波峰）** | PeakDetector + min_distance 检测波峰，**必须在金叉之前** | 上一次波峰已确认，使用波峰检测更精确，且间隔合理 |

**【重要】监控模式顶背离检测规则**：
1. **高点B** = 最近一次金叉（买入信号）后到当前日期之间的最高点
2. **高点A** = 金叉之前的峰点列表中，最近一个满足背离条件的峰点（通过PeakDetector检测，受min_distance限制）
3. 只检测这一个顶背离组合，不遍历所有峰点组合

**与回测的区别**：
| 场景 | 高点B判断 | 高点A判断 |
|-----|---------|---------|
| **每日监控** | 金叉后最高价（直接max） | 金叉前的PeakDetector峰点 |
| **回测** | PeakDetector + min_distance | PeakDetector + min_distance |

**注意**：回测的局部高点判断逻辑详见 `backtest-strategy` skill。

#### 2.3.1 局部高点定义（每日监控专用）

**局部高点B（最新）** = 当前金叉买入后到当前时间之间的最高价点（直接取max）

**局部高点A（上一次）** = 金叉之前的峰点列表中，最近一个满足背离条件的峰点（使用PeakDetector + min_distance检测）

**局部高点 = 金叉买入后到当前时间之间的最高价点**

| 概念 | 定义 |
|-----|------|
| 金叉 | SAR绿转红结束（SAR从价格上方转到下方），或MACD金叉（DIF上穿DEA） |
| 局部高点B | 当前金叉买入后至今的最高价点（最高价最大的日期） |
| 局部高点A | 上一次金叉买入后到上一次死叉卖出之间的最高价点 |

#### 2.3.2 局部高点查找流程

```
检测到当前死叉（卖出信号触发）
    ↓
向前回溯找到上一次金叉（买入信号）的日期
    ↓
从上一次金叉日期到当前日期，找出最高价最大的日期作为局部高点B
    ↓
再向前回溯找到更早的金叉和死叉周期
    ↓
从那个金叉到对应的死叉，找出最高价最大的日期作为局部高点A
    ↓
比较高点A和高点B判断顶背离
```

#### 2.3.3 代码实现示例（每日监控专用）

```python
def find_local_high_b_for_monitor(
    df: pd.DataFrame,
    golden_cross_date: pd.Timestamp,
    current_date: pd.Timestamp
) -> Tuple[pd.Timestamp, float, float]:
    """
    找最新的局部高点B（每日监控专用，不使用min_distance）

    高点B判断方式：直接取金叉买入后最高价点

    Args:
        df: 历史数据DataFrame（必须包含high列）
        golden_cross_date: 金叉买入日期
        current_date: 当前日期（或当前死叉日期）

    Returns:
        (高点B日期, 高点B最高价, 高点B MACD柱值)
    """
    # 截取金叉后的数据范围
    range_df = df[(df.index >= golden_cross_date) & (df.index <= current_date)]

    if len(range_df) == 0:
        return None, 0.0, 0.0

    # 直接取最高价最大的日期（不使用min_distance）
    max_high_idx = range_df['high'].idxmax()
    max_high_price = range_df['high'].max()

    # 获取最高点当天的MACD柱值
    macd_at_high = get_macd_value_at_date(df, max_high_idx)

    return max_high_idx, max_high_price, macd_at_high


def find_peak_a_for_monitor(
    peak_detector: PeakDetector,
    df: pd.DataFrame,
    peak_b_date: pd.Timestamp
) -> Tuple[pd.Timestamp, float, float]:
    """
    找上一次波峰A（每日监控专用，使用PeakDetector + min_distance）

    高点A判断方式：通过PeakDetector检测历史波峰

    Args:
        peak_detector: 波峰检测器（已初始化，min_distance=10）
        df: 历史数据DataFrame
        peak_b_date: 高点B日期（用于确定查找范围）

    Returns:
        (高点A日期, 高点A最高价, 高点A MACD柱值)
    """
    # 使用PeakDetector检测历史波峰（min_distance方式）
    peaks = peak_detector.detect_peaks(
        df.index[0].strftime('%Y-%m-%d'),
        peak_b_date.strftime('%Y-%m-%d')  # 只检测高点B之前的波峰
    )

    if len(peaks) == 0:
        return None, 0.0, 0.0

    # 取最近的一个波峰作为高点A
    peak_a = peaks[-1]  # 最近的波峰

    # 获取波峰A的MACD柱值
    macd_at_peak_a = get_macd_value_at_date(df, peak_a.date)

    return peak_a.date, peak_a.price, macd_at_peak_a
```

#### 2.3.4 回测的局部高点判断（backtest专用）

回测场景中，高点A和高点B都使用 `PeakDetector` + `min_distance` 方式检测波峰，详见 `backtest-strategy` skill。

```python
# 回测：高点A和高点B都使用PeakDetector（回测专用）
peak_detector = PeakDetector(
    atr_period=14,
    prominence_factor=1.0,
    min_distance=10  # 回测使用min_distance参数
)
peaks = peak_detector.detect_peaks(start_date, end_date)
# 高点B = peaks[-1]（最新波峰）
# 高点A = peaks[-2]（上一个波峰）
```

#### 2.3.5 重要说明（每日监控）

1. **不使用PeakDetector的min_distance**：
   - PeakDetector使用min_distance参数检测波峰，需要右侧数据确认
   - 监控场景下，直接找金叉后最高价更直观且无未来函数风险

2. **金叉定义**（根据judge_buy_ids）：
   - SAR金叉：SAR从价格上方转到下方（绿转红结束，多头趋势开始）
   - MACD金叉：DIF上穿DEA

3. **时间范围**：
   - 高点B：当前金叉日期 → 当前日期（或当前死叉日期）
   - 高点A：上一次金叉日期 → 上一次死叉日期

4. **向量化实现**：
   ```python
   # 使用向量化操作找最高点，禁止显式for循环
   max_high_idx = range_df['high'].idxmax()  # pandas内置方法
   max_high_price = range_df['high'].max()   # 向量化计算
   ```

#### 2.3.5 高点筛选补充条件

虽然直接取最高价，但仍需满足以下条件才能用于背离判断：

| 条件 | 说明 |
|-----|------|
| MACD柱为正 | 最高点当天MACD柱值 > 0（顶背离需要MACD在零轴上方） |
| 时间间隔合理 | 高点A与高点B间隔 ≥ 5天，≤ 60天 |
| 价格高于MA60 | 两个高点都高于60日均线（确认趋势向上）

### 2.4 顶背离判断

对找到的局部高点进行顶背离判断：

**日线顶背离条件**：
- 高点B最高价 > 高点A最高价（价格创新高）
- 高点B的MACD柱 < 高点A的MACD柱（动能减弱）
- 时间间隔 ≤ 60天
- 回调幅度 ≤ 30%
- 两者均高于MA60（趋势向上）

**周线顶背离条件**：
- 高点B最高价 > 高点A最高价
- 高点B的MACD柱 < 高点A的MACD柱
- 时间间隔 ≤ 20周
- 回调幅度 ≤ 30%

### 2.5 周线RSI判断

除了顶背离判断，还需要检查周线RSI状态：

1. **RSI阈值检查**：
   - RSI > 80：极度超买，强烈卖出信号
   - RSI > 70：超买区域，注意风险
   - RSI < 30：超卖区域

2. **RSI峰值记录**：
   - 记录触发阈值时的日期、RSI值、股价
   - 用于阶梯减仓告警

### 2.6 告警生成

根据检测结果生成两类告警：

**提前预警**（条件接近但未完全触发）：
- 周线RSI > 75（接近80阈值）
- 顶背离形成但未出现死叉确认

**卖出告警**（条件已触发）：
- 死叉 + 顶背离 → "策略1-顶背离均线" 告警
- 死叉 + 周线RSI超买 → "策略1-阶梯" 告警

### 2.7 代码实现示例

```python
def monitor_daily_logic(stock_config: dict, df: pd.DataFrame) -> List[Alert]:
    """
    每日监控核心逻辑

    Args:
        stock_config: 股票配置（含judge_sell_ids）
        df: 历史数据DataFrame

    Returns:
        告警列表
    """
    alerts = []

    # 1. 检测当天是否发生死叉
    death_cross = detect_death_cross(df, stock_config['judge_sell_ids'])

    if death_cross['triggered']:
        # 2. 查找局部高点（从上一次死叉到当前死叉）
        local_highs = find_local_highs(df, death_cross['prev_date'], death_cross['current_date'])

        # 3. 判断顶背离
        divergence = check_top_divergence(df, local_highs)

        # 4. 检查周线RSI
        weekly_rsi = calculate_weekly_rsi(df)

        # 5. 生成告警
        if divergence['detected']:
            alerts.append(generate_divergence_alert(divergence, death_cross))

        if weekly_rsi > 80:
            alerts.append(generate_rsi_alert(weekly_rsi, df))

    return alerts
```

## 三、卖出信号触发条件

### 核心原则

**无论是回测还是每日监控，判断个股卖出是否死叉的条件，都是基于个股的 `judge_sell_ids` 中的选择值来判断。**

### sell_id 与死叉判断对应关系

| sell_id | 死叉判断条件 | 说明 |
|---------|-------------|------|
| sell_id=1 | SAR绿转红（SAR从价格下方转到上方） | 使用SAR跌破信号 |
| sell_id=2 | MACD日线死叉（DIF下穿DEA） | 使用MACD死叉信号 |
| sell_id=3 | MACD日线死叉 | 同sell_id=2 |
| sell_id=4 | MACD日线死叉 + 上证指数背离 | 需结合上证指数判断 |
| sell_id=5 | MACD日线死叉 + 创业板指数背离 | 需结合创业板指数判断 |
| sell_id=6 | MACD日线死叉 + 科创板指数背离 | 需结合科创板指数判断 |
| sell_id=7 | 上证指数MACD日线死叉 | 以指数死叉为准 |
| sell_id=8 | 创业板指数MACD日线死叉 | 以指数死叉为准 |
| sell_id=9 | 科创板指数MACD日线死叉 | 以指数死叉为准 |

### 顶背离形成时指数已处于死叉状态的直接卖出规则

**核心规则**：当顶背离形成时，如果对应的指数已经处于死叉状态（MACD死叉或SAR死叉），则直接触发卖出告警，无需等待新的死叉信号。

#### 触发条件

| 背离类型 | 死叉状态检查 | 建议操作 |
|---------|-------------|---------|
| **周线顶背离** | 指数MACD柱 < 0（DIF < DEA）或 SAR > Close | **建议清仓** |
| **日线顶背离** | 指数MACD柱 < 0（DIF < DEA）或 SAR > Close | **建议卖出1/3** |

#### 适用范围

- **sell_id=10,11,12**：个股周线顶背离 + 指数（上证/创业板/科创板）MACD死叉
- **sell_id=13,14,15**：个股周线顶背离 + 新指数（上证50/创业板50/科创板50）MACD死叉
- **sell_id=4,5,6**：个股日线顶背离 + 指数背离（同样适用直接卖出规则）

#### 监控告警示例

```
2025-03-20 | 日RSI:86.28 | 周RSI:79.87 | 【比亚迪】卖出信号 (sell_id=11): 日线顶背离形成于 2025-03-20, 创业板指数MACD已处于死叉状态，建议卖出1/3

2025-05-23 | 日RSI:81.01 | 周RSI:72.71 | 【比亚迪】清仓信号 (sell_id=11): 日线顶背离形成于 2025-05-23, 创业板指数SAR已处于死叉状态(SAR>Close)，建议清仓
```

#### 为什么需要这个规则

正常流程：顶背离形成 → 设置标志 → 等待指数死叉 → 触发卖出告警

问题场景：顶背离形成时，指数可能已经处于死叉状态（之前已经死叉了），此时如果等待"新的死叉"，会导致：
1. 卖出时机延迟，错过最佳卖点
2. 背离信号可能长期等待，无法及时生效

解决方案：顶背离形成当天，检查指数是否已处于死叉状态，如果是则直接发出卖出告警。

### 顶背离与RSI峰值间隔校验规则（sell_id=10-15专用）

**核心规则**：指数顶背离形成日期与个股周线RSI峰值日期之间的间隔不得超过5个工作日，否则顶背离失效。

#### 规则说明

在sell_id=10-15策略中，当检测到指数顶背离时，需要校验：
- 指数顶背离形成日期（`index_div_date`）
- 个股周线RSI峰值日期（`rsi_peak_date`）
- 两日期间隔工作日数 ≤ 5天

如果间隔超过5个工作日，则：
- 指数顶背离信号失效
- 不触发卖出告警（跳过该背离检查）
- 继续检查其他卖出条件（如周线顶背离）

#### 为什么需要这个规则

问题场景：指数顶背离形成于12月10日，但个股RSI峰值出现在12月26日（相距16天），两者时间关联性弱：
1. 指数背离已过时效，个股RSI峰值是后续行情产生
2. 背离信号与个股状态脱节，卖出信号不准确

解决方案：校验日期间隔，确保背离信号与个股状态时间上匹配。

#### 监控日志示例

```
2025-01-03 | 创业板指数日线顶背离(2024-12-10)与个股RSI峰值(2024-12-26)间隔12个工作日，超过5天，顶背离失效
```

## 四、告警分类规范

### 告警类型定义

每日监控告警分为两类：

| 告警类型 | 说明 | 触发条件 | 用户操作 |
|---------|------|---------|---------|
| **提前预警** | 条件接近触发，提前通知 | 指标接近阈值但尚未触发卖出 | 关注市场，做好卖出准备 |
| **卖出告警** | 条件已触发，建议卖出 | 死叉+背离/RSI触发 | 立即执行卖出操作 |

### 提前预警触发条件

| 预警类型 | 触发条件 | 示例 |
|---------|---------|------|
| RSI高位预警 | 周线RSI > 75 | "周线RSI已达80.5，接近卖出阈值，请关注" |
| 背离形成预警 | 顶背离形成但未生效 | "日线顶背离已形成，待死叉确认" |

### 卖出告警详细信息要求

#### 1. 顶背离卖出告警

**必须包含的信息**：
- **高点A详细信息**：日期、股价、MACD柱值
- **高点B详细信息**：日期、股价、MACD柱值
- 死叉确认日期（基于judge_sell_ids）

**示例格式**：
```
2026-04-03 | 日RSI:65.30 | 周RSI:75.20 | 【股票名】卖出信号 (策略1-顶背离均线): 日线顶背离生效，高点A(2026-03-15, 股价=3.280, MACD=0.0250) vs 高点B(2026-03-28, 股价=3.350, MACD=0.0180)，SAR绿转红确认(2026-04-03)，建议卖出1/3
```

#### 2. RSI阶梯减仓告警

**必须包含的信息**：
- **RSI峰值详细信息**：触发最高RSI的日期、峰值值、当日股价
- 死叉确认日期（基于judge_sell_ids）

**示例格式**：
```
2026-04-03 | 日RSI:75.20 | 周RSI:88.50 | 【股票名】卖出信号 (策略1-阶梯): 触发 [SAR绿转红 (2026-04-03)]，周线RSI flag=2，周线RSI峰值:2026-03-28=88.50(股价=3.350)，建议卖出1/2
```

## 五、杜绝未来函数（Look-ahead Bias）

### 必须遵守的规则

1. **信号生成时机**：
   - 当天收盘后才能确认当天的信号
   - 当天开盘前只能使用前一天的数据

2. **shift偏移规则**：
   ```python
   # 错误：使用当天收盘价判断当天买入
   if df['close'].iloc[-1] > threshold:
       buy_signal = True

   # 正确：使用前一天收盘价判断今天买入
   if df['close'].iloc[-2] > threshold:
       buy_signal = True

   # 或使用shift
   df['prev_close'] = df['close'].shift(1)
   if df['prev_close'].iloc[-1] > threshold:
       buy_signal = True
   ```

3. **局部高点确认**：
   - 局部高点需要右侧数据确认（至少1-2根K线）
   - 监控模式下，最近最高点可能未完全确认，需放宽条件

4. **指标计算时机**：
   - RSI、MACD、SAR等指标当天计算值，在当天收盘后才能使用
   - 使用 `shift(1)` 获取前一天的指标值

### 检查清单

在编写监控代码时，必须检查：
- [ ] 买入信号是否使用前一天数据？
- [ ] 卖出信号是否在SAR跌破/均线死叉当天收盘后判断？
- [ ] 背离判断是否使用已确认的局部高点？
- [ ] RSI阈值判断是否使用正确的shift偏移？

## 六、向量化计算优先

### 禁止显式for循环

核心计算逻辑必须使用pandas/numpy向量化操作：

```python
# 错误：使用for循环计算RSI
rsi_values = []
for i in range(len(df)):
    # ... 计算逻辑

# 正确：使用向量化计算
rsi = RSI(period=14)
rsi_values = rsi.calculate(df['close'])

# 正确：使用rolling/window操作
df['ma5'] = df['close'].rolling(window=5).mean()
```

### 允许例外的情况

以下情况可以使用显式循环：
- 非向量化的复杂事件处理（如背离比较）
- 历史状态遍历（如检查过去7天的信号）
- 告警消息生成（需要逐条格式化）

## 七、数据处理规范

### 时间序列处理

```python
# 必须先将datetime转换为索引并排序
df = df.rename(columns={'bob': 'datetime'})
df = df.set_index('datetime')
df = df.sort_index()

# 获取最新交易日数据
latest_date = df.index[-1].strftime('%Y-%m-%d')
```

### 数据预热要求

```python
# 监控需要足够的历史数据进行指标预热
WARMUP_DAYS = 730  # 至少2年数据（月线RSI计算需要）

# 获取数据时包含预热期
start_date = (datetime.now() - timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
```

### 周线/月线数据聚合

```python
# 周线聚合
weekly_df = df.groupby(df.index.to_period('W')).agg({
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

## 八、告警机制规范

### 钉钉消息格式

告警消息必须包含以下信息：

```
{日期} | 日 RSI:{日线RSI值} | 周 RSI:{周线RSI值} | 【股票名】{信号类型} ({策略ID}): {详细信息}
```

### 各策略日志格式

| 策略类型 | 日志关键字 | 必须包含的信息 |
|---------|-----------|---------------|
| RSI阶梯减仓 | `策略1-阶梯` / `策略2-阶梯` | 均线跌破日期、RSI flag值、RSI峰值日期和值 |
| 日线顶背离 | `策略1-顶背离均线` | 顶背离形成日期、前高价格、当前高价格 |
| 周线顶背离 | `策略1-周线顶背离` | 顶背离形成日期、前高价格 |
| 清仓信号 | `策略1-清仓` | 顶背离时间周期、顶背离形成日期、前高价格 |

### 示例日志

```
# RSI阶梯减仓告警
2026-04-03 | 日 RSI:75.20 | 周 RSI:88.50 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-阶梯): 触发 [日线5日均线下穿10日均线 (2026-04-03)]，周线RSI flag=2 (RSI峰值:2026-03-28=88.50)，建议卖出1/2

# 日线顶背离告警
2026-04-03 | 日 RSI:65.30 | 周 RSI:75.20 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-顶背离均线): 日线顶背离 (2026-03-28) + 5日均线下穿10日均线确认，前高=16.500, 高=16.800, 建议卖出1/3
```

### 告警去重规则

- 卖出后5个交易日内不再发送卖出告警
- 同一信号类型每天只发送一次
- 使用 `Set[str]` 记录已发送告警ID

## 九、监控时间设置

```python
# 监控时间点（交易时间内每30分钟执行）
MORNING_MONITOR_TIMES = ['09:30', '10:00', '10:30', '11:00', '11:30']
AFTERNOON_MONITOR_TIMES = ['13:00', '13:30', '14:00', '14:30', '15:00']

# 监控间隔
MONITOR_INTERVAL = 30 * 60  # 30分钟

# 回溯检查天数（实时监控需要）
RECENT_TRADING_DAYS = 2  # 检查最近2个交易日
RSI_HIGH_LOOKBACK_DAYS = 30  # RSI高点回溯30天
```

## 十、生存者偏差处理

```python
# 配置文件中检查股票状态
# stocks_monitor_by_day.csv 包含 active 字段
# active=1 表示正常监控，active=0 表示已退市或ST

# 代码中需要过滤
active_stocks = df_config[df_config['active'] == 1]
```

## 十一、代码结构模板

```python
# -*- coding: utf-8 -*-
"""
监控脚本模块 (monitor_xxx.py)

【功能说明】
监控XXX指标，使用SAR死叉跌破替代均线死叉。

【监控逻辑】
1. 监控当天最新交易日的数据
2. 当发生SAR死叉跌破时触发分析
3. 根据买入卖出策略判断买卖点

【使用方法】
    python src/runner/monitor_xxx.py

作者：量化交易团队
"""

# ==================== 监控配置 ====================
# 常量定义

# ==================== 数据结构 ====================
# @dataclass 定义

# ==================== 监控执行器 ====================
class MonitorRunner:
    def __init__(self, config):
        """初始化"""
        pass

    def prepare_data(self):
        """数据准备（向量化计算指标）"""
        pass

    def check_signals(self):
        """信号检测（调用strategy.py）"""
        pass

    def send_alert(self, signals):
        """发送告警（钉钉消息）"""
        pass

    def run(self):
        """执行监控"""
        pass

# ==================== 主函数 ====================
def main():
    runner = MonitorRunner(config)
    runner.run()

if __name__ == '__main__':
    main()
```

## 十二、自测检查清单

在提交监控代码前，必须完成以下自测：

1. **未来函数检查**：
   - 运行测试用例，确认没有使用当天数据判断当天信号

2. **边界条件测试**：
   - 数据不足时的处理（如RSI预热期）
   - 空数据/None值的处理
   - 日期越界的处理

3. **告警格式验证**：
   - 钉钉消息格式符合规范
   - 包含所有必要信息

4. **性能测试**：
   - 多股票并行监控的性能
   - 向量化计算是否生效（无显式for循环在核心逻辑）

5. **生存者偏差**：
   - 已退市/ST股票是否被过滤

## 十三、指数RSI与背离日常监控规范

### 13.1 监控指数列表

每日监控必须同时监控以下指数的RSI和背离状态：

| 指数代码 | 指数名称 | 说明 |
|---------|---------|------|
| SZSE.399006 | 创业板指数 | 创业板整体走势 |
| SHSE.000001 | 上证指数 | 上海证券交易所综合指数 |
| SHSE.000680 | 科创综指 | 科创板综合指数 |
| BJSE.899050 | 北证板块 | 北京证券交易所板块指数 |
| SHSE.513180 | 恒生科技ETF | 港股恒生科技指数（ETF替代） |
| SZSE.159920 | 恒生指数ETF | 港股恒生指数（ETF替代） |

### 13.2 指数RSI告警阈值

**日线RSI阈值**：
| RSI范围 | 状态 | 建议操作 |
|--------|------|---------|
| RSI > 80 | 极度超买 | 高风险，密切关注 |
| RSI > 70 | 超买 | 注意回调风险 |
| RSI < 30 | 超卖 | 关注反弹机会 |
| RSI < 20 | 极度超卖 | 强反弹信号 |

**周线RSI阈值**：
| RSI范围 | 状态 | 建议操作 |
|--------|------|---------|
| RSI > 80 | 极度超买 | 强烈卖出信号 |
| RSI > 70 | 超买 | 卖出预警 |
| RSI < 30 | 超卖 | 强买入信号 |
| RSI < 25 | 极度超卖 | 极强买入信号 |

### 13.3 指数日常监控告警格式

**指数RSI状态告警**（每日例行输出）：
```
{日期} | 【指数名称】日线RSI:{日线RSI值} | 周线RSI:{周线RSI值} | 状态:{状态描述}
```

**示例**：
```
2026-04-03 | 【创业板指数】日线RSI:75.20 | 周线RSI:82.50 | 状态:超买预警
2026-04-03 | 【上证指数】日线RSI:45.30 | 周线RSI:38.20 | 状态:正常
2026-04-03 | 【科创综指】日线RSI:28.50 | 周线RSI:25.30 | 状态:超卖关注
```

### 13.4 指数背离监控逻辑

**指数顶背离检测**：
- 使用 `DivergenceDetector` 类检测指数的顶背离
- 检测周期：日线和周线
- 检测逻辑与个股相同（高点B价格>A价格，MACD柱B<A）

**指数底背离检测**：
- 高点B价格 < 高点A价格（价格创新低）
- 高点B的MACD柱 > 高点A的MACD柱（动能增强）

### 13.5 指数背离告警格式

**指数顶背离形成告警**：
```
{日期} | 【指数名称】顶背离形成 ({周期}): 高点A({日期A}, 指数={价格A}, MACD={MACD_A}) vs 高点B({日期B}, 指数={价格B}, MACD={MACD_B})
```

**示例**：
```
2026-04-03 | 【创业板指数】顶背离形成 (日线): 高点A(2026-03-15, 指数=2200.00, MACD=25.00) vs 高点B(2026-03-28, 指数=2250.00, MACD=18.00)
2026-04-03 | 【上证指数】顶背离形成 (周线): 高点A(2026-03-01周, 指数=3100.00, MACD=30.00) vs 高点B(2026-03-15周, 指数=3150.00, MACD=20.00)
```

### 13.6 代码实现示例

```python
def monitor_index_rsi_and_divergence(detector: DivergenceDetector) -> List[str]:
    """
    监控指数的RSI和背离状态

    Args:
        detector: 背离检测器实例

    Returns:
        指数监控告警消息列表
    """
    alerts = []

    # 监控的指数列表
    INDEX_SYMBOLS = {
        'SZSE.399006': '创业板指数',
        'SHSE.000001': '上证指数',
        'SHSE.000680': '科创综指',
        'BJSE.899050': '北证板块',
        'SHSE.513180': '恒生科技ETF',
        'SZSE.159920': '恒生指数ETF'
    }

    for symbol, name in INDEX_SYMBOLS.items():
        # 准备数据
        df = detector.prepare_data(symbol, start_date, end_date)

        if df.empty:
            continue

        # 计算RSI
        daily_rsi = calculate_daily_rsi(df)
        weekly_rsi = calculate_weekly_rsi(df)

        # 判断RSI状态
        status = get_rsi_status(daily_rsi, weekly_rsi)

        # 生成RSI状态告警
        alert = f"{latest_date} | 【{name}】日线RSI:{daily_rsi:.2f} | 周线RSI:{weekly_rsi:.2f} | 状态:{status}"
        alerts.append(alert)

        # 检测背离
        divergences = detector.detect_all_divergences()

        # 生成背离告警
        for div in divergences['daily_top_formed']:
            alert = f"{latest_date} | 【{name}】顶背离形成 (日线): 高点A({div.peak_a_date.strftime('%Y-%m-%d')}, 指数={div.peak_a_price:.2f}, MACD={div.peak_a_macd:.4f}) vs 高点B({div.date.strftime('%Y-%m-%d')}, 指数={div.peak_b_price:.2f}, MACD={div.peak_b_macd:.4f})"
            alerts.append(alert)

        for div in divergences['weekly_top_formed']:
            alert = f"{latest_date} | 【{name}】顶背离形成 (周线): 高点A({div.peak_a_date.strftime('%Y-%m-%d')}, 指数={div.peak_a_price:.2f}, MACD={div.peak_a_macd:.4f}) vs 高点B({div.date.strftime('%Y-%m-%d')}, 指数={div.peak_b_price:.2f}, MACD={div.peak_b_macd:.4f})"
            alerts.append(alert)

    return alerts


def get_rsi_status(daily_rsi: float, weekly_rsi: float) -> str:
    """
    根据日线和周线RSI判断状态

    Args:
        daily_rsi: 日线RSI值
        weekly_rsi: 周线RSI值

    Returns:
        状态描述字符串
    """
    # 优先判断周线（更稳定）
    if weekly_rsi > 80:
        return "极度超买-高风险"
    elif weekly_rsi > 70:
        return "超买预警"
    elif weekly_rsi < 25:
        return "极度超卖-强买入"
    elif weekly_rsi < 30:
        return "超卖关注"

    # 其次判断日线
    if daily_rsi > 80:
        return "日线极度超买"
    elif daily_rsi > 70:
        return "日线超买"
    elif daily_rsi < 20:
        return "日线极度超卖"
    elif daily_rsi < 30:
        return "日线超卖"

    return "正常"
```

### 13.7 指数监控与个股监控的关联

当个股配置中 `judge_sell_ids` 包含以下值时，必须同时监控对应指数：

| sell_id | 需监控的指数 | 说明 |
|---------|-------------|------|
| sell_id=4 | 上证指数 (SHSE.000001) | 个股MACD死叉 + 上证指数背离 |
| sell_id=5 | 创业板指数 (SZSE.399006) | 个股MACD死叉 + 创业板指数背离 |
| sell_id=6 | 科创综指 (SHSE.000680) | 个股MACD死叉 + 科创板指数背离 |
| sell_id=7 | 上证指数 (SHSE.000001) | 上证指数MACD死叉触发卖出 |
| sell_id=8 | 创业板指数 (SZSE.399006) | 创业板指数MACD死叉触发卖出 |
| sell_id=9 | 科创综指 (SHSE.000680) | 科创板指数MACD死叉触发卖出 |

### 13.8 指数监控输出时机

**日常监控输出**：
- 每日监控执行时，先输出所有指数的RSI状态
- 如有背离形成，输出背离详情
- 在个股监控告警前，先展示指数环境信息

**告警消息顺序**：
```
===== 指数环境监控 =====
2026-04-03 | 【创业板指数】日线RSI:75.20 | 周线RSI:82.50 | 状态:超买预警
2026-04-03 | 【上证指数】日线RSI:45.30 | 周线RSI:38.20 | 状态:正常
...

===== 指数背离告警 =====
2026-04-03 | 【创业板指数】顶背离形成 (日线): ...

===== 个股监控告警 =====
2026-04-03 | 日RSI:65.30 | 周RSI:75.20 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-阶梯): ...
```

## 十四、与回测的区别

| 特性 | 实时监控 | 回测 |
|-----|---------|------|
| 数据范围 | 最近2个交易日 | 全历史数据 |
| 信号触发 | 当天收盘后 | 每日遍历 |
| 告警机制 | 钉钉推送 | 日志文件 |
| 去重逻辑 | 5日内不重复发送 | 无去重 |
| 局部高点 | 最近最高点（可能未确认） | 已确认的峰点 |
| 指数监控 | 每日例行输出RSI状态 | 仅在sell_id触发时判断 |