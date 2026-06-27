---
name: backtest-strategy
description: |
  量化交易回测策略代码规范。当编写、修改或审查回测相关代码时必须使用此skill。

  触发条件：
  - 用户提及"回测"、"backtest"、"策略验证"、"绩效评估"
  - 编写或修改 src/runner/backtest*.py 相关文件
  - 编写或修改策略验证、交易模拟、绩效分析相关代码
  - 计算年化收益率、夏普比率、最大回撤、卡玛比率
  - 涉及历史数据回测、交易记录生成

  此skill确保回测代码遵循量化开发铁律，杜绝未来函数，输出标准化绩效指标。
---

# 回测策略代码规范

## 一、核心职责分离

回测代码必须遵循职责分离原则：

| 职责 | 处理模块 | 说明 |
|-----|---------|------|
| 策略逻辑 | `strategy.py` | 买入/卖出信号判断（TradingStrategy类） |
| 指标计算 | `rsi_calculator.py`, `macd_calculator.py`, `sar_strategy.py` | 技术指标计算 |
| 背离检测 | `divergence_detector.py` | 顶背离/底背离判断 |
| 回测执行 | `backtest_runner.py` | 数据准备、指标计算、交易执行、报告生成 |

**回测脚本职责**：
- 数据准备（获取历史数据）
- 调用指标计算模块（预计算所有指标）
- 调用策略类进行信号检测
- 执行模拟交易（买入/卖出记录）
- 生成绩效报告

**回测脚本不包含策略逻辑本身**

### 策略规则参考文件

当需要了解具体的买入/卖出策略条件（buy_id、sell_id 的详细定义）时，请读取：
- **`.claude/common_skills_docu/buy_sell.md`** - 公共文档，包含所有买入卖出策略的完整规则说明

读取时机：
- 实现/修改买入卖出信号判断逻辑时
- 需要确认具体策略条件阈值时
- 回测报告需要包含策略详细信息时

## 二、卖出信号触发条件

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
| sell_id=10-15 | 个股顶背离 + 指数环境共振 | 详见下方规则 |

### sell_id=10-15 个股顶背离 + 指数环境共振规则

**核心逻辑**：当个股形成顶背离时，根据个股和指数的MACD状态决定卖出触发时机。

#### 触发场景矩阵

| 场景 | 个股状态 | 指数状态 | 触发条件 | 说明 |
|-----|---------|---------|---------|------|
| **(1)** | 个股顶背离 + 个股已处于死叉 | 不限 | **立即卖出** | 个股自身确认顶背离，无需等待指数 |
| **(2)** | 个股顶背离 + 个股未死叉 | 指数已处于死叉 | **等待个股死叉 或 指数重新死叉** | 指数环境恶化，若指数再次出现死叉也触发 |
| **(3)** | 个股顶背离 + 个股未死叉 | 指数未死叉 | **等待个股或指数死叉** | 双方都未确认，等待任意确认 |

#### 代码实现逻辑

```python
# sell_id=10: 个股顶背离 + 上证指数环境
stock_top_divergence = state.daily_divergence_flag > 0  # 个股顶背离是否形成

if stock_top_divergence:
    if macd_dead_cross_state:  # 个股已处于死叉状态
        # (1) 个股顶背离 + 个股已处于死叉 → 直接触发
        trigger_condition = True
        trigger_name = "个股顶背离+个股已处于死叉状态"
    elif sh_index_dead_cross_state:  # 指数已处于死叉状态，个股未死叉
        # (2) 个股顶背离 + 指数已处于死叉 → 等待个股死叉 或 指数重新死叉
        trigger_condition = macd_cross_down or sh_index_macd_cross_down
        trigger_name = "个股顶背离+上证指数已处于死叉状态，等待个股死叉或指数重新死叉"
    else:  # 指数未死叉，个股未死叉
        # (3) 个股顶背离 + 双方都未死叉 → 等待任意一方死叉
        trigger_condition = macd_cross_down or sh_index_macd_cross_down
        trigger_name = "个股顶背离，等待个股或上证指数死叉"
else:
    trigger_condition = False  # 无个股顶背离，不触发
```

#### sell_id=10-15 对应指数

| sell_id | 对应指数 | 检测方法 |
|---------|---------|---------|
| sell_id=10 | 上证指数 (SHSE.000001) | `sh_index_dead_cross_state` |
| sell_id=11 | 创业板指数 (SZSE.399006) | `cyb_index_dead_cross_state` |
| sell_id=12 | 科创板指数 (SHSE.000680) | `kc_index_dead_cross_state` |
| sell_id=13 | 上证50指数 (SHSE.000016) | `sh50_index_dead_cross_state` |
| sell_id=14 | 创业板50指数 (SZSE.399673) | `cyb50_index_dead_cross_state` |
| sell_id=15 | 科创板50指数 (SHSE.000688) | `kc50_index_dead_cross_state` |

#### 关键参数说明

| 参数 | 含义 | 检测方式 |
|-----|------|---------|
| `macd_dead_cross_state` | 个股MACD是否已处于死叉状态 | `DIF < DEA` |
| `sh_index_dead_cross_state` | 上证指数是否已处于死叉状态 | `指数MACD柱 < 0` |
| `macd_cross_down` | 当天新发生个股MACD死叉 | `DIF下穿DEA` |
| `sh_index_macd_cross_down` | 当天新发生上证指数MACD死叉 | `指数DIF下穿DEA` |

#### 日志示例

```
# 场景(1): 个股顶背离 + 个股已处于死叉 → 直接卖出
[2025-03-20] 日线顶背离形成
[2025-03-20] 个股MACD已处于死叉状态: DIF=-2.5 < DEA=-1.8
[2025-03-20] 卖出信号触发: 个股顶背离+个股已处于死叉状态

# 场景(2): 个股顶背离 + 指数已处于死叉 → 等待个股死叉
[2025-03-20] 日线顶背离形成
[2025-03-20] 上证指数MACD已处于死叉状态: MACD柱=-6.0 < 0
[2025-03-20] 等待个股死叉触发卖出
[2025-03-25] 个股MACD死叉触发: DIF下穿DEA
[2025-03-25] 卖出信号触发: 个股顶背离+上证指数已处于死叉状态

# 场景(3): 个股顶背离 + 双方未死叉 → 等待任意死叉
[2025-03-20] 日线顶背离形成
[2025-03-20] 个股未处于死叉状态，指数未处于死叉状态
[2025-03-20] 等待个股或上证指数死叉触发卖出
[2025-03-22] 上证指数MACD死叉触发
[2025-03-22] 卖出信号触发: 个股顶背离，上证指数死叉确认
```

### 顶背离形成时指数已处于死叉状态的直接卖出规则

**核心规则**：当顶背离形成时，如果对应的指数已经处于死叉状态（MACD死叉或SAR死叉），则直接触发卖出，无需等待新的死叉信号。

#### 触发条件

| 背离类型 | 死叉状态检查 | 卖出比例 |
|---------|-------------|---------|
| **周线顶背离** | 指数MACD柱 < 0（DIF < DEA）或 SAR > Close | **清仓（卖出全部）** |
| **日线顶背离** | 指数MACD柱 < 0（DIF < DEA）或 SAR > Close | **卖出1/3** |

#### 适用范围

- **sell_id=10,11,12**：个股周线顶背离 + 指数（上证/创业板/科创板）MACD死叉
- **sell_id=13,14,15**：个股周线顶背离 + 新指数（上证50/创业板50/科创板50）MACD死叉
- **sell_id=4,5,6**：个股日线顶背离 + 指数背离（同样适用直接卖出规则）

#### 代码实现逻辑

```python
# 周线顶背离形成时检查指数是否已处于死叉状态
if weekly_top_div and self.state.weekly_divergence_flag == 0:
    # 设置背离标志
    self.state.weekly_divergence_flag = 1
    self.logger.info(f"周线顶背离形成，等待死叉触发生效")

    # 【关键规则】检查指数是否已经是死叉状态
    immediate_sell = False

    # 检查MACD死叉状态（MACD柱 < 0 表示 DIF < DEA）
    if self._is_index_macd_dead_cross_state(date, index_type):
        immediate_sell = True
        self.logger.info(f"指数MACD已处于死叉状态，直接触发卖出")

    # 检查SAR死叉状态（SAR > Close）
    elif self._is_index_sar_dead_cross_state(date, index_type):
        immediate_sell = True
        self.logger.info(f"指数SAR已处于死叉状态，直接触发卖出")

    # 如果需要立即卖出，执行清仓
    if immediate_sell and self._shares > 0:
        sell_signal = SellSignal(
            flag=SellFlag.CLEAR_ALL,  # 周线顶背离清仓
            reason=f"清仓信号: 周线顶背离形成，指数已处于死叉状态"
        )
        self._execute_sell(date, sell_signal)
```

#### 日志示例

```
[2025-03-20] 日线顶背离形成，等待死叉触发生效
[2025-03-20] 创业板指数MACD已处于死叉状态: MACD柱=-6.0169 < 0
[2025-03-20] 创业板指数MACD已处于死叉状态，直接触发卖出
[2025-03-20] 卖出: 260股 @ 131.760 | 卖出信号 (sell_id=11): 日线顶背离形成于 2025-03-20, 创业板指数MACD已处于死叉状态，卖出比例1/3
```

#### 为什么需要这个规则

正常流程：顶背离形成 → 设置标志 → 等待指数死叉 → 触发卖出

问题场景：顶背离形成时，指数可能已经处于死叉状态（之前已经死叉了），此时如果等待"新的死叉"，会导致：
1. 卖出时机延迟，错过最佳卖点
2. 背离信号可能长期等待，无法及时生效

解决方案：顶背离形成当天，检查指数是否已处于死叉状态，如果是则直接卖出。

### 顶背离与RSI峰值间隔校验规则（sell_id=10-15专用）

**核心规则**：指数顶背离形成日期与个股周线RSI峰值日期之间的间隔不得超过5个工作日，否则顶背离失效。

#### 规则说明

在sell_id=10-15策略中，当检测到指数顶背离时，需要校验：
- 指数顶背离形成日期（`index_div_date`）
- 个股周线RSI峰值日期（`rsi_peak_date`）
- 两日期间隔工作日数 ≤ 5天

如果间隔超过5个工作日，则：
- 指数顶背离信号失效
- 不触发卖出（跳过该背离检查）
- 继续检查其他卖出条件（如周线顶背离）

#### 为什么需要这个规则

问题场景：指数顶背离形成于12月10日，但个股RSI峰值出现在12月26日（相距16天），两者时间关联性弱：
1. 指数背离已过时效，个股RSI峰值是后续行情产生
2. 背离信号与个股状态脱节，卖出信号不准确

解决方案：校验日期间隔，确保背离信号与个股状态时间上匹配。

#### 代码实现

```python
# 校验：指数顶背离与个股RSI峰值间隔不超过5个工作日
days_diff = count_trading_days(index_div_date, rsi_peak_date)
if days_diff > MAX_DIVERGENCE_RSI_PEAK_DAYS:
    logger.debug(f"指数顶背离({index_div_date})与个股RSI峰值({rsi_peak_date})间隔{days_diff}个工作日，超过5天，顶背离失效")
    # 跳过该背离检查，不触发卖出
else:
    # 继续检查RSI阈值触发卖出
    if weekly_rsi >= 80:
        return SellSignal(...)
```

#### 常量定义

```python
MAX_DIVERGENCE_RSI_PEAK_DAYS = 5  # 指数顶背离与个股RSI峰值最大间隔（工作日）
```

## 三、杜绝未来函数（Look-ahead Bias）

### 核心原则

回测最关键的规则：**当天收盘后的数据才能决定当天的操作**

### 常见未来函数陷阱

```python
# ❌ 错误：使用当天收盘价判断当天买入
if df['close'].iloc[i] > threshold:
    execute_buy(i)

# ✅ 正确：使用前一天收盘价判断今天买入
if df['close'].iloc[i-1] > threshold:
    execute_buy(i)

# ✅ 正确：使用shift偏移
df['prev_close'] = df['close'].shift(1)
if df['prev_close'].iloc[i] > threshold:
    execute_buy(i)
```

### 指标计算时机

```python
# ❌ 错误：当天RSI值判断当天操作
if rsi_series.iloc[i] < 25:
    buy_signal = True

# ✅ 正确：使用前一天RSI值
if rsi_series.iloc[i-1] < 25:
    buy_signal = True
```

### 局部高点确认

```python
# ❌ 错误：当天检测为高点，当天就判断背离
if is_peak(i) and has_divergence(i):
    sell_signal = True

# ✅ 正确：高点需要右侧数据确认，在确认后才能判断
# 高点确认后，等待均线跌破触发卖出
if peak_confirmed and ma5_cross_down_ma10:
    sell_signal = True
```

### 回测遍历逻辑

```python
# ✅ 正确的回测遍历方式
for i in range(len(df)):
    current_date = df.index[i]

    # 使用截至i-1的数据进行判断
    # 所有信号判断使用 i-1 或更早的数据

    # 当天的操作（如果有信号）
    if buy_signal:
        buy_price = df['close'].iloc[i]  # 当天收盘价买入
        record_trade(i, 'buy', buy_price)
```

### 检查清单

在编写回测代码时，必须检查：
- [ ] 买入信号判断是否使用前一日数据？
- [ ] 卖出信号判断是否在触发条件确认后？
- [ ] 局部高点确认是否有右侧数据验证？
- [ ] RSI/MACD/SAR指标值是否正确偏移？
- [ ] 背离生效是否有均线跌破确认？

## 三、向量化计算优先

### 禁止显式for循环（核心计算）

```python
# ❌ 错误：for循环计算RSI
for i in range(14, len(df)):
    gains = df['close'].iloc[i-14:i].diff().clip(lower=0).mean()
    losses = df['close'].iloc[i-14:i].diff().clip(upper=0).abs().mean()
    rsi = 100 - 100/(1 + gains/losses)

# ✅ 正确：向量化计算
from src.tool.rsi_calculator import RSI
rsi_calc = RSI(period=14)
rsi_series = rsi_calc.calculate(df['close'])
```

```python
# ✅ 正确：rolling/window操作
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
df['ma60'] = df['close'].rolling(window=60).mean()
```

### 允许例外的情况

以下情况可以使用显式循环：
- 交易执行遍历（每日检查信号并执行交易）
- 背离比较逻辑（需要遍历峰点列表）
- 绩效指标计算后的汇总

### 预计算所有指标

```python
# 回测开始前，预计算所有需要的指标
class BacktestRunner:
    def prepare_indicators(self):
        """预计算所有指标"""
        # RSI
        self._daily_rsi = RSI(period=14).calculate(self._df['close'])
        self._weekly_rsi = self._calculate_weekly_rsi()

        # SAR
        self._sar_strategy = SARStrategy(acceleration=0.02, maximum=0.20)
        self._sar_series = self._sar_strategy.calculate(self._df)

        # MACD
        self._macd_calc = MACDCalculator()
        self._macd_series = self._macd_calc.calculate(self._df['close'])

        # 背离信号（预计算）
        self._divergence_detector = DivergenceDetector()
        self._daily_divergences = self._divergence_detector.detect_daily_top_divergence()
        self._weekly_divergences = self._divergence_detector.detect_weekly_top_divergence()
```

## 四、数据处理规范

### 时间序列处理

```python
# 必须先将datetime转换为索引并排序
df = df.rename(columns={'bob': 'datetime'})
df = df.set_index('datetime')
df = df.sort_index()
```

### 数据预热要求

```python
# 回测需要足够的预热数据让指标稳定
# RSI需要至少14天
# MACD需要至少26天
# MA60需要60天
# 月线RSI需要约2年数据

# 获取数据时包含预热期
warmup_days = max(14, 26, 60, 730)  # 取最大需求
start_date_with_warmup = (actual_start - timedelta(days=warmup_days)).strftime('%Y-%m-%d')
```

### 缺失数据处理

```python
# 使用fill_missing参数处理缺失数据
df = history(
    symbol=symbol,
    frequency='1d',
    start_time=start_date,
    end_time=end_date,
    skip_suspended=True,
    fill_missing='Last'  # 用前值填充
)
```

## 五、交易执行规范

### 交易记录结构

```python
@dataclass
class TradeRecord:
    """交易记录"""
    date: pd.Timestamp       # 交易日期
    action: str              # 'buy' 或 'sell'
    amount: float            # 交易比例
    price: float             # 交易价格
    reason: str              # 交易原因（策略ID）
    daily_rsi: float = 0.0   # 当天日线RSI
    weekly_rsi: float = 0.0  # 当天周线RSI
```

### 买入规则

```python
# judge_buy_id触发：新资金买入 + 买回之前卖出的资金
if buy_signal_from_new_capital:
    total_buy_amount = new_capital + previously_sold_capital

# judge_t_id触发：只买回之前卖出的资金
if buy_signal_from_t_strategy:
    total_buy_amount = previously_sold_capital
    # 不买入新资金
```

### 卖出规则

```python
# sell_flag优先级：1 > 2 > 3
# sell_flag=1: 清仓（卖出全部）
# sell_flag=2: 卖出1/2
# sell_flag=3: 卖出1/3

# 多信号同时触发时，执行最大卖出比例
sell_ratio = max(sell_flag_ratios)  # 取优先级最高的
```

### 交易价格约定

```python
# 买入价格：当天收盘价
buy_price = df['close'].iloc[i]

# 卖出价格：当天收盘价
sell_price = df['close'].iloc[i]
```

## 六、绩效评估指标（必须输出）

### 标准化输出格式

任何回测结果必须输出以下核心指标：

```python
# ==================== 绩效指标 ====================
绩效指标 = {
    '年化收益率': annual_return,
    '夏普比率': sharpe_ratio,
    '最大回撤': max_drawdown,
    '卡玛比率': calmar_ratio
}
```

### 计算公式

```python
def calculate_performance_metrics(df_returns: pd.DataFrame) -> Dict:
    """计算绩效指标"""

    # 年化收益率
    total_return = (final_value - initial_capital) / initial_capital
    trading_days = len(df_returns)
    annual_return = total_return * (252 / trading_days)

    # 夏普比率（假设无风险利率=3%）
    excess_returns = df_returns['daily_return'] - 0.03/252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    # 最大回撤
    cumulative_returns = (1 + df_returns['daily_return']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # 卡玛比率 = 年化收益率 / 最大回撤绝对值
    calmar_ratio = annual_return / abs(max_drawdown)

    return {
        '年化收益率': annual_return,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '卡玛比率': calmar_ratio
    }
```

### 输出报告格式

```markdown
# 回测报告

## 基本信息
- 股票代码：SHSE.512480
- 股票名称：半导体ETF
- 回测区间：2024-01-01 ~ 2026-04-30
- 初始资金：100,000 元

## 策略配置
- 买入策略ID：[1]
- 做T策略ID：[2]
- 卖出策略ID：[1]

## 绩效指标
| 指标 | 数值 |
|-----|------|
| 年化收益率 | 15.2% |
| 夏普比率 | 1.35 |
| 最大回撤 | -12.5% |
| 卡玛比率 | 1.22 |

## 交易记录
- 总交易次数：32次
- 买入次数：16次
- 卖出次数：16次
- 盈利交易：20次
- 亏损交易：12次
```

## 七、生存者偏差处理

### 股票池过滤

```python
# 配置文件stocks_backtest.csv需要包含active字段
# active=1: 正常回测
# active=0: 已退市/ST（排除）

def load_stock_pool(config_path: str) -> List[StockConfig]:
    """加载股票池（过滤退市/ST）"""
    df = pd.read_csv(config_path)
    active_stocks = df[df['active'] == 1]

    stock_configs = []
    for row in active_stocks.itertuples():
        config = StockConfig(
            symbol=row.symbol,
            active=row.active,
            # ...
        )
        stock_configs.append(config)

    return stock_configs
```

### 历史数据检查

```python
# 检查股票是否在回测区间内一直存在
def check_stock_survival(symbol: str, start_date: str, end_date: str) -> bool:
    """检查股票是否在区间内存活"""
    df = get_history_data(symbol, start_date, end_date)

    # 检查是否有足够的数据
    if len(df) < min_required_days:
        return False

    # 检查是否有连续缺失（可能退市）
    if has_continuous_missing(df):
        return False

    return True
```

## 八、风险控制

### 单笔交易限制

```python
# 单笔买入不超过总资金的30%
MAX_SINGLE_BUY_RATIO = 0.30

# 单笔卖出不低于总资金的10%
MIN_SINGLE_SELL_RATIO = 0.10
```

### 仓位管理

```python
# 最大持仓不超过总资金的80%
MAX_POSITION_RATIO = 0.80

# 最小持仓不低于总资金的20%（现金缓冲）
MIN_CASH_RATIO = 0.20
```

### 止损/止盈规则

```python
# 可选：硬止损（单笔亏损超过10%强制止损）
HARD_STOP_LOSS = 0.10

# 可选：硬止盈（单笔盈利超过30%强制止盈）
HARD_STOP_PROFIT = 0.30
```

## 九、代码结构模板

```python
# -*- coding: utf-8 -*-
"""
回测执行器模块 (backtest_xxx.py)

【功能说明】
执行历史数据回测验证。调用strategy.py的交易策略类进行信号检测。
不包含策略逻辑，仅负责：数据准备、指标计算、调用策略、执行交易、生成报告。

【使用方法】
修改main函数中的参数后运行：
    python src/runner/backtest_xxx.py

作者：量化交易团队
"""

# ==================== 数据结构 ====================
@dataclass
class TradeRecord:
    """交易记录"""
    date: pd.Timestamp
    action: str
    amount: float
    price: float
    reason: str

# ==================== 回测执行器 ====================
class BacktestRunner:
    def __init__(self, symbol, start_date, end_date, strategy_ids):
        """初始化回测参数"""
        pass

    def prepare_data(self):
        """获取历史数据"""
        pass

    def calculate_indicators(self):
        """预计算所有指标（向量化）"""
        pass

    def run_backtest(self):
        """执行回测（遍历交易日）"""
        pass

    def calculate_performance(self):
        """计算绩效指标"""
        # 必须输出：年化收益率、夏普比率、最大回撤、卡玛比率
        pass

    def generate_report(self):
        """生成回测报告"""
        pass

# ==================== 主函数 ====================
def main():
    runner = BacktestRunner(
        symbol='SHSE.512480',
        start_date='2024-01-01',
        end_date='2026-04-30',
        judge_buy_ids=[1],
        judge_t_ids=[2],
        judge_sell_ids=[1]
    )

    runner.prepare_data()
    runner.calculate_indicators()
    runner.run_backtest()

    # 输出绩效指标
    metrics = runner.calculate_performance()
    print(f"年化收益率: {metrics['年化收益率']:.2%}")
    print(f"夏普比率: {metrics['夏普比率']:.2f}")
    print(f"最大回撤: {metrics['最大回撤']:.2%}")
    print(f"卡玛比率: {metrics['卡玛比率']:.2f}")

    runner.generate_report()

if __name__ == '__main__':
    main()
```

## 十、自测检查清单

在提交回测代码前，必须完成以下自测：

1. **未来函数检查**：
   - 手动检查每个信号判断是否使用正确的时间偏移
   - 运行单元测试验证

2. **绩效指标验证**：
   - 年化收益率、夏普比率、最大回撤、卡玛比率是否全部输出
   - 数值是否合理（检查异常值）

3. **边界条件测试**：
   - 数据不足时的处理
   - 空数据/None值的处理
   - 日期越界的处理

4. **生存者偏差检查**：
   - 已退市/ST股票是否被过滤

5. **性能测试**：
   - 大量数据回测的性能
   - 向量化计算是否生效

## 十一、与实时监控的区别

| 特性 | 回测 | 实时监控 |
|-----|------|---------|
| 数据范围 | 全历史数据 | 最近2个交易日 |
| 信号触发 | 每日遍历 | 当天收盘后 |
| 告警机制 | 日志文件 | 钉钉推送 |
| 去重逻辑 | 无去重 | 5日内不重复 |
| 局部高点 | 已确认峰点 | 最近最高点（可能未确认） |
| 绩效输出 | 必须 | 不需要 |
| 背离生效 | 需均线跌破确认 | 背离形成即告警 |

## 十二、批量回测规范

```python
# 批量回测多个股票
class BatchBacktestRunner:
    def __init__(self, stock_configs: List[StockConfig]):
        self.stock_configs = stock_configs

    def run_all(self):
        """并行回测所有股票"""
        results = []
        for config in self.stock_configs:
            runner = BacktestRunner(config)
            result = runner.run_backtest()
            results.append(result)

        # 生成汇总报告
        self.generate_summary_report(results)

    def generate_summary_report(self, results):
        """汇总报告"""
        # 平均绩效指标
        avg_annual_return = np.mean([r['年化收益率'] for r in results])
        avg_sharpe = np.mean([r['夏普比率'] for r in results])
        avg_max_drawdown = np.mean([r['最大回撤'] for r in results])

        # 输出汇总
        print(f"平均年化收益率: {avg_annual_return:.2%}")
        print(f"平均夏普比率: {avg_sharpe:.2f}")
        print(f"平均最大回撤: {avg_max_drawdown:.2%}")
```