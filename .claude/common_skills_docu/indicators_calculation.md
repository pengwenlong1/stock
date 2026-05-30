# 技术指标计算规则

本文档记录所有技术指标的计算公式、参数配置和使用场景，供监控和回测代码参考。

---

## 一、RSI 相对强弱指标

### 参数配置

| 指标 | 参数 | 说明 |
|-----|------|------|
| RSI(6) | period=6 | 短期RSI，用于日线判断 |
| RSI(14) | period=14 | 标准RSI，用于日线判断 |
| 周线RSI | period=14 | 周线级别RSI |
| 月线RSI | period=14 | 月线级别RSI |

### 计算公式

```python
# RSI计算公式（向量化实现）
def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI相对强弱指标

    Args:
        close_prices: 收盘价序列
        period: RSI周期（默认14）

    Returns:
        RSI值序列（范围0-100）

    计算步骤：
    1. 计算价格变动：delta = close.diff()
    2. 分离上涨和下跌：
       - gain = delta.clip(lower=0)  # 只保留上涨部分
       - loss = delta.clip(upper=0).abs()  # 只保留下跌部分（取绝对值）
    3. 计算平均上涨和下跌（使用EMA）：
       - avg_gain = gain.ewm(span=period, adjust=False).mean()
       - avg_loss = loss.ewm(span=period, adjust=False).mean()
    4. 计算相对强度：RS = avg_gain / avg_loss
    5. 计算RSI：RSI = 100 - 100 / (1 + RS)
    """
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)

    return rsi
```

### 实时周线RSI计算方法

在实际交易分析中，经常需要在周内非周五交易日也能观察周线级别RSI趋势。这需要计算"实时周线RSI"（Real-time Weekly RSI）。

#### 核心思路

将"周线"理解为滚动7天窗口，在周中用"当前未完成周的最新价格"代替该周的收盘价，动态估算当前周线RSI。

#### 实现步骤（以14周周期为例）

**第一步：定义周的边界**
- 每周从周一开盘到周五收盘（A股通常周五为周收盘）
- 对于任意一天，确定它属于哪一周

**第二步：构建有效周收盘价序列**

需要构造一个长度为14的"周收盘价"列表：
- 前13个：已结束的13个完整周的周五收盘价
- 第14个（最新一周）：当天最新价格作为"临时周收盘价"

**举例**：今天是2026-04-09（周三），计算"截至今天的周线RSI"：
1. 取过去13个完整周的周五收盘价（如2026-01-10 到 2026-04-04）
2. 最新一周（2026-04-07 至 2026-04-13）尚未结束，用 2026-04-09 的收盘价作为该周的"临时收盘价"

**第三步：按标准RSI公式计算**

使用13个真实周收盘 + 1个临时周收盘的序列，共14个"周收盘价"，然后：
- 计算相邻周之间的价格变动
- 分离上涨/下跌幅度
- 计算平均上涨和平均下跌（使用Wilder平滑）
- 代入RSI公式

#### Python实现提示

```python
# 使用pandas.resample提取周收盘，再手动替换最后一周为当日价
weekly_closes = df['close'].resample('W-FRI').last()
# 替换最后一周为当日收盘价
weekly_closes.iloc[-1] = df['close'].iloc[-1]
# 计算RSI
weekly_rsi = calculate_rsi(weekly_closes, period=14)
```

#### 注意事项

- 由于最后一周是"未完成"的，这个RSI是动态变化的，每天都会更新，直到周五才固定
- 每日监控中使用此方法可实时跟踪周线RSI变化趋势

### 代码位置

```
src/tool/rsi_calculator.py
```

---

## 二、MACD 指数平滑异同移动平均线

### 参数配置

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| 快线周期 | 12 | DIF的计算周期 |
| 慢线周期 | 26 | DEA的计算周期 |
| 信号线周期 | 9 | DEA的EMA周期 |

### 计算公式

```python
def calculate_macd(close_prices: pd.Series,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标

    Args:
        close_prices: 收盘价序列
        fast_period: 快线周期（默认12）
        slow_period: 慢线周期（默认26）
        signal_period: 信号线周期（默认9）

    Returns:
        (dif, dea, macd柱)

    计算步骤：
    1. 计算快线EMA：EMA_fast = close.ewm(span=fast_period, adjust=False).mean()
    2. 计算慢线EMA：EMA_slow = close.ewm(span=slow_period, adjust=False).mean()
    3. 计算DIF（快慢线差值）：DIF = EMA_fast - EMA_slow
    4. 计算DEA（DIF的EMA）：DEA = DIF.ewm(span=signal_period, adjust=False).mean()
    5. 计算MACD柱：MACD = (DIF - DEA) * 2
    """
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()

    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    macd = (dif - dea) * 2

    return dif, dea, macd
```

### 使用场景

| 场景 | 判断条件 | 说明 |
|-----|---------|------|
| MACD金叉 | DIF上穿DEA | 买入信号参考 |
| MACD死叉 | DIF下穿DEA | 卖出信号触发（sell_id=2/3） |
| 顶背离判断 | 价格新高 + MACD柱降低 | 卖出信号 |

### 顶背离判断逻辑

```
顶背离形成条件：
1. 高点B最高价 > 高点A最高价（价格创新高）
2. 高点B的MACD柱 < 高点A的MACD柱（动能减弱）
3. 时间间隔：日线≤60天，周线≤20周
4. 回调幅度≤30%（同趋势约束）
```

### 代码位置

```
src/tool/macd_calculator.py
```

---

## 三、MA 移动平均线

### 参数配置

| 指标 | 周期 | 说明 |
|-----|------|------|
| MA5 | 5日 | 短期均线 |
| MA10 | 10日 | 短期均线 |
| MA60 | 60日 | 中期均线（趋势确认） |

### 计算公式

```python
def calculate_ma(close_prices: pd.Series, period: int) -> pd.Series:
    """
    计算移动平均线

    Args:
        close_prices: 收盘价序列
        period: 均线周期

    Returns:
        MA值序列

    计算公式：
    MA = close.rolling(window=period).mean()
    """
    return close_prices.rolling(window=period).mean()
```

### 使用场景

| 场景 | 判断条件 | 说明 |
|-----|---------|------|
| 均线金叉 | MA5上穿MA10 | 买入信号参考 |
| 均线死叉 | MA5下穿MA10 | 卖出信号确认 |
| 趋势确认 | 价格 > MA60 | 顶背离需两者高于MA60 |

### 代码位置

```
src/tool/ma_calculator.py
```

---

## 四、SAR 抛物线指标

### 参数配置

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| 加速因子 | 0.02 | SAR(10,2,20)中的2 |
| 最大加速因子 | 0.20 | SAR(10,2,20)中的20 |

### 计算公式

```python
def calculate_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                  acceleration: float = 0.02, maximum: float = 0.20) -> pd.Series:
    """
    计算SAR抛物线指标

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        acceleration: 加速因子（默认0.02）
        maximum: 最大加速因子（默认0.20）

    Returns:
        SAR值序列

    计算逻辑：
    - SAR在价格下方时（多头）：SAR = 前一日SAR + AF * (最高价 - 前一日SAR)
    - SAR在价格上方时（空头）：SAR = 前一日SAR + AF * (最低价 - 前一日SAR)
    - 每次创新高/新低，AF增加0.02，直到maximum
    - 价格突破SAR时，反转趋势，重置AF
    """
    # 详细实现见 src/tool/sar_strategy.py
```

### 使用场景

| 场景 | 判断条件 | 说明 |
|-----|---------|------|
| SAR绿转红 | SAR从价格下方转到上方 | 多头转为空头，卖出信号（sell_id=1） |
| SAR红转绿 | SAR从价格上方转到下方 | 空头转为多头，买入信号参考 |

### 代码位置

```
src/tool/sar_strategy.py
```

---

## 五、指标计算原则

### 1. 向量化计算优先

所有指标计算必须使用 pandas/numpy 向量化操作，禁止在核心计算中使用显式 for 循环。

```python
# ✅ 正确：向量化计算
rsi = close_prices.ewm(span=14, adjust=False).mean()

# ❌ 错误：显式循环
rsi = []
for i in range(len(close_prices)):
    ...
```

### 2. 杜绝未来函数

指标计算当天，只能使用当天及之前的数据：

```python
# ✅ 正确：使用shift偏移
df['prev_rsi'] = df['rsi'].shift(1)
if df['prev_rsi'].iloc[-1] < 25:  # 使用前一天RSI判断今天买入
    buy_signal = True

# ❌ 错误：使用当天RSI判断当天操作
if df['rsi'].iloc[-1] < 25:
    buy_signal = True
```

### 3. 数据预热要求

| 指标 | 最少预热天数 | 说明 |
|-----|-------------|------|
| RSI(14) | 14天 | 需要足够数据让EMA稳定 |
| MACD(12,26,9) | 26天 | 需要慢线完整计算 |
| MA60 | 60天 | 需要完整窗口 |
| 月线RSI | 约730天（2年） | 月线数据需要足够历史 |

---

## 六、指标模块对应关系

| 指标 | 模块文件 | 类名 |
|-----|---------|------|
| RSI | `src/tool/rsi_calculator.py` | `RSI` |
| MACD | `src/tool/macd_calculator.py` | `MACDCalculator` |
| MA | `src/tool/ma_calculator.py` | - (直接使用rolling) |
| SAR | `src/tool/sar_strategy.py` | `SARStrategy` |
| 背离检测 | `src/tool/divergence_detector.py` | `DivergenceDetector` |
| 峰点检测 | `src/tool/peak_detector.py` | `PeakDetector` |

---

## 七、回测与监控的指标使用

### 回测模式

```python
# 预计算所有指标（在回测开始前）
def prepare_indicators(self):
    self._daily_rsi = RSI(period=14).calculate(self._df['close'])
    self._weekly_rsi = self._calculate_weekly_rsi()
    self._sar_series = SARStrategy().calculate(self._df)
    self._dif, self._dea, self._macd = MACDCalculator().calculate(self._df['close'])
```

### 监控模式

```python
# 每次监控重新计算（使用预热数据）
def prepare_indicators(self):
    # 获取包含预热的历史数据
    df = history(symbol, start_date=warmup_start, end_date=end_date)

    # 计算指标
    self._daily_rsi = RSI(period=14).calculate(df['close'])
    self._sar_series = SARStrategy().calculate(df)
```