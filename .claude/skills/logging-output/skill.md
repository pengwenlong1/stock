---
name: logging-output
description: |
  量化交易日志打印输出规范。当编写、修改或审查日志相关代码时必须使用此skill。

  触发条件：
  - 用户提及"日志"、"log"、"打印输出"、"输出格式"
  - 编写或修改日志输出、日志格式化相关代码
  - 编写或修改钉钉告警消息格式
  - 实现卖出/买入信号的日志输出
  - 配置logging模块、设置日志级别

  此skill确保日志输出格式统一规范，便于问题排查和策略复盘。
---

# 日志打印输出规范

## 策略规则参考文件

当需要了解具体的买入/卖出策略条件（buy_id、sell_id 的详细定义）时，请读取：
- **`.claude/common_skills_docu/buy_sell.md`** - 公共文档，包含所有买入卖出策略的完整规则说明

读取时机：
- 编写/修改日志格式时需要了解策略详细信息
- 日志输出需要包含策略条件阈值时
- 告警消息需要显示策略依据时

## 一、日志级别规范

### 级别定义

| 级别 | 用途 | 示例场景 |
|-----|------|---------|
| **INFO** | 正常业务流程 | 交易执行、信号触发、指标计算完成 |
| **WARNING** | 需关注但不影响运行 | 数据缺失、指标异常、配置项缺失 |
| **ERROR** | 错误但可恢复 | 数据获取失败、API调用异常 |
| **DEBUG** | 详细调试信息 | 指标计算过程、背离检测细节 |

### 配置标准

```python
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
    ]
)

# 模块级日志器
logger = logging.getLogger(__name__)

# 设置日志文件（可选）
def setup_log_file(log_path: str):
    """设置日志文件输出"""
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    )
    logger.addHandler(file_handler)
    return file_handler
```

## 二、告警分类规范

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
| 止损预警 | 价格接近止损线 | "当前价格距止损线仅5%，请关注" |

### 卖出告警触发条件

**核心原则**：无论是回测还是每日监控，判断个股卖出是否死叉的条件，都是基于个股的 `judge_sell_ids` 中的选择值来判断。

| sell_id | 死叉判断条件 | 说明 |
|---------|-------------|------|
| sell_id=1 | SAR绿转红（SAR从价格下方转到上方） | 使用SAR跌破信号 |
| sell_id=2 | MACD日线死叉（DIF下穿DEA） | 使用MACD死叉信号 |
| sell_id=3 | MACD日线死叉 | 同sell_id=2 |
| sell_id=4~6 | MACD日线死叉 + 指数背离 | 需结合指数判断 |
| sell_id=7~9 | 指数MACD日线死叉 | 以指数死叉为准 |

## 三、卖出日志格式

### 基本要求

卖出日志必须包含：
1. **卖出时间点**：显示卖出发生的日期
2. **卖出依据策略**：显示依据哪一个策略（策略ID），基于 `judge_sell_ids` 配置
3. **RSI详细信息**：如果依据RSI策略，需要显示：
   - 触发最高RSI的日期
   - RSI峰值值
   - 当日股价
4. **顶背离详细信息**：如果依据顶背离策略，需要显示：
   - **前高点（高点A）日期**：必须显示，格式为"前高点YYYY-MM-DD"
   - 高点A的股价、MACD柱值
   - 高点B（背离形成点）的日期、股价、MACD柱值

**【重要】卖出信号日志必须同时显示背离形成日期和前高点日期**

示例：
```
卖出信号 (sell_id=3-日线顶背离): 日线顶背离形成于 2026-05-27, 前高点2026-05-14, MACD死叉触发, 前高=3.280, 建议卖出1/3
``

### 各策略日志格式

#### 1. RSI阶梯减仓日志

**关键字**：`策略1-阶梯` / `策略2-阶梯`

**必须包含的信息**：
- 均线跌破日期（基于judge_sell_ids配置的死叉类型）
- RSI flag值
- **RSI峰值详细信息**：
  - 最高RSI触发日期
  - RSI峰值值
  - 峰值当日股价

**格式示例**：
```
# 提前预警（钉钉）
2026-04-01 | 日RSI:70.50 | 周RSI:82.30 | 【华泰柏瑞上证红利ETF】提前预警: 周线RSI已达82.30，接近卖出阈值(80)，周线RSI峰值出现在2026-03-25(峰值=85.50, 股价=3.280)，请关注

# 卖出告警（钉钉）
2026-04-03 | 日RSI:75.20 | 周RSI:88.50 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-阶梯): 触发 [SAR绿转红 (2026-04-03)]，周线RSI flag=2，周线RSI峰值:2026-03-28=88.50(股价=3.350)，建议卖出1/2

# 回测日志
[2026-04-03] 卖出：1000股 @ 3.280 | 日RSI:75.20 | 周RSI:88.50 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-阶梯): 触发 [SAR绿转红 (2026-04-03)]，周线RSI flag=2，周线RSI峰值:2026-03-28=88.50(股价=3.350)，建议卖出1/2
```

#### 2. 日线顶背离日志

**关键字**：`策略1-顶背离均线` / `策略2-顶背离均线`

**必须包含的信息**：
- **高点A详细信息**：日期、股价、MACD柱值
- **高点B详细信息**：日期、股价、MACD柱值
- 死叉确认日期

**格式示例**：
```
# 提前预警（钉钉）- 背离形成但未死叉
2026-04-01 | 日RSI:65.30 | 周RSI:75.20 | 【华泰柏瑞上证红利ETF】提前预警: 日线顶背离已形成，高点A(2026-03-15, 股价=3.280, MACD=0.0250) vs 高点B(2026-03-28, 股价=3.350, MACD=0.0180)，待SAR死叉确认

# 卖出告警（钉钉）- 背离生效
2026-04-03 | 日RSI:65.30 | 周RSI:75.20 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-顶背离均线): 日线顶背离生效，高点A(2026-03-15, 股价=3.280, MACD=0.0250) vs 高点B(2026-03-28, 股价=3.350, MACD=0.0180)，SAR绿转红确认(2026-04-03)，建议卖出1/3

# 回测日志
[2026-04-03] 卖出：500股 @ 3.280 | 日RSI:65.30 | 周RSI:75.20 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-顶背离均线): 日线顶背离生效，高点A(2026-03-15, 股价=3.280, MACD=0.0250) vs 高点B(2026-03-28, 股价=3.350, MACD=0.0180)，SAR绿转红确认(2026-04-03)，建议卖出1/3
```

#### 3. 周线顶背离日志

**关键字**：`策略1-周线顶背离`

**必须包含的信息**：
- **高点A详细信息**：日期、股价、周线MACD柱值
- **高点B详细信息**：日期、股价、周线MACD柱值
- 死叉确认日期

**格式示例**：
```
# 提前预警（钉钉）- 周线背离形成但未死叉
2026-04-01 | 日RSI:55.10 | 周RSI:82.30 | 【华泰柏瑞上证红利ETF】提前预警: 周线顶背离已形成，高点A(2026-03-01, 股价=3.200, MACD=0.0300) vs 高点B(2026-03-15, 股价=3.280, MACD=0.0200)，待周线SAR死叉确认

# 卖出告警（钉钉）- 周线背离生效
2026-04-03 | 日RSI:55.10 | 周RSI:82.30 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-周线顶背离): 周线顶背离生效，高点A(2026-03-01, 股价=3.200, MACD=0.0300) vs 高点B(2026-03-15, 股价=3.280, MACD=0.0200)，周线SAR绿转红确认(2026-03-22)，建议清仓

# 回测日志
[2026-04-03] 卖出：2000股 @ 3.150 | 日RSI:55.10 | 周RSI:82.30 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-周线顶背离): 周线顶背离生效，高点A(2026-03-01, 股价=3.200, MACD=0.0300) vs 高点B(2026-03-15, 股价=3.280, MACD=0.0200)，周线SAR绿转红确认(2026-03-22)，建议清仓
```

#### 4. 清仓信号日志

**关键字**：`策略1-清仓` / `策略2-清仓`

**必须包含的信息**：
- 顶背离时间周期（日线/周线）
- **两个高点的详细信息**：日期、股价、MACD柱值
- 死叉确认日期

**格式示例**：
```
# 卖出告警（钉钉）
2026-04-03 | 日RSI:50.20 | 周RSI:78.50 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-清仓): 周线顶背离生效，高点A(2026-03-01, 股价=3.200, MACD=0.0300) vs 高点B(2026-03-15, 股价=3.280, MACD=0.0200)，SAR绿转红确认(2026-04-03)，建议清仓

# 回测日志
[2026-04-03] 卖出：全部持仓 @ 3.150 | 日RSI:50.20 | 周RSI:78.50 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-清仓): 周线顶背离生效，高点A(2026-03-01, 股价=3.200, MACD=0.0300) vs 高点B(2026-03-15, 股价=3.280, MACD=0.0200)，SAR绿转红确认(2026-04-03)，建议清仓
```

#### 5. 指数背离触发日志（sell_id=4~9）

**关键字**：`策略X-指数背离`

**必须包含的信息**：
- 指数名称及背离类型
- **指数两个高点的详细信息**：日期、股价、MACD柱值
- 个股周线RSI值
- 死叉触发条件

**格式示例**：
```
# 卖出告警（钉钉）- sell_id=4 上证指数日线顶背离
2026-04-03 | 日RSI:50.20 | 周RSI:82.50 | 【迈瑞医疗】卖出信号 (策略4-指数背离): 上证指数日线顶背离生效，高点A(2026-03-15, 指数=3100, MACD=25.0) vs 高点B(2026-03-28, 指数=3150, MACD=18.0)，个股周线RSI=82.50>80，建议卖出1/3

# 卖出告警（钉钉）- sell_id=5 创业板指数周线顶背离
2026-04-03 | 日RSI:45.30 | 周RSI:85.20 | 【比亚迪】卖出信号 (策略5-指数背离): 创业板指数周线顶背离生效，高点A(2026-03-01, 指数=2200, MACD=30.0) vs 高点B(2026-03-15, 指数=2250, MACD=20.0)，个股周线RSI=85.20>85，建议卖出1/2
```

### 卖出日志格式汇总表

| 策略类型 | 日志关键字 | 必须包含的信息 | 建议操作 |
|---------|-----------|---------------|---------|
| RSI阶梯减仓 | `策略1-阶梯` | RSI峰值日期、峰值值、股价；死叉类型(基于judge_sell_ids) | 卖出1/2或1/3 |
| 日线顶背离 | `策略1-顶背离均线` | 高点A/B的日期、股价、MACD值；死叉确认日期 | 卖出1/3 |
| 周线顶背离 | `策略1-周线顶背离` | 高点A/B的日期、股价、MACD值；周线死叉确认日期 | 清仓 |
| 清仓信号 | `策略1-清仓` | 高点A/B的详细信息；死叉确认日期 | 清仓 |
| 指数背离 | `策略4~9` | 指数高点A/B详细信息；个股周线RSI；死叉类型 | 按RSI阈值卖出 |

## 三、买入日志格式

### 基本要求

买入日志必须包含：
1. **买入时间点**：显示买入发生的日期
2. **买入依据策略**：显示依据哪一个策略（buy_id）
3. **RSI详细信息**：日线RSI、周线RSI的当前值
4. **买入类型**：新资金买入 vs 做T买回

### 各策略日志格式

#### 1. 新资金买入日志

**关键字**：`新资金买入信号`

**格式示例**：
```
# 回测日志
[2026-04-03] 买入：1000股 @ 15.234 | 日RSI:18.50 | 周RSI:22.30 | 【股票名】新资金买入信号 (buy_id=1): 日线RSI<20 且 周线RSI<25，建议买入

# 钉钉告警日志
2026-04-03 | 日RSI:18.50 | 周RSI:22.30 | 【股票名】新资金买入信号 (buy_id=1): 日线RSI<20 且 周RSI<25，建议买入
```

#### 2. 做T买回日志

**关键字**：`做T买回信号`

**格式示例**：
```
# 回测日志
[2026-04-03] 买入：500股 @ 15.100 | 日RSI:22.30 | 周RSI:28.50 | 【股票名】做T买回信号 (t_id=2): 日线RSI<25 且 周线RSI<30，建议买回之前卖出资金

# 钉钉告警日志
2026-04-03 | 日RSI:22.30 | 周RSI:28.50 | 【股票名】做T买回信号 (t_id=2): 日线RSI<25 且 周线RSI<30，建议买回之前卖出资金
```

#### 3. 指数保护型买入日志

**关键字**：`指数保护型买入`

**格式示例**：
```
# 创业板指数保护
[2026-04-03] 买入：800股 @ 14.800 | 创业板RSI:18.20 | 【股票名】指数保护型买入 (buy_id=3): 创业板指数(SZSE.399006)日线RSI<25，建议买入

# 上证指数保护
[2026-04-03] 买入：600股 @ 14.500 | 上证RSI:19.50 | 【股票名】指数保护型买入 (buy_id=6): 上证指数(SHSE.000001)日线RSI<20，建议买入
```

### 买入日志格式汇总表

| 策略类型 | 日志关键字 | 必须包含的信息 | 买入类型 |
|---------|-----------|---------------|---------|
| 保守型买入 | `buy_id=1` | 日线RSI<20, 周线RSI<25 | 新资金+买回 |
| 标准型买入 | `buy_id=2` | 日线RSI<25, 周线RSI<30 | 常用于做T买回 |
| 指数保护型 | `buy_id=3/5/6/7` | 指数名称、指数RSI值 | 新资金+买回 |
| 极度保守型 | `buy_id=4` | 日线RSI<20, 周线RSI<20 | 新资金+买回 |

## 四、钉钉告警格式

### 消息结构

钉钉告警消息使用统一的格式：

```
{日期} | {日线RSI} | {周线RSI} | 【{股票名}】{信号类型} ({策略ID}): {详细信息}
```

### 字段说明

| 字段 | 格式 | 示例 |
|-----|------|------|
| 日期 | YYYY-MM-DD | 2026-04-03 |
| 日线RSI | 日RSI:XX.XX | 日RSI:75.20 |
| 周线RSI | 周RSI:XX.XX | 周RSI:88.50 |
| 股票名 | 【股票名】 | 【华泰柏瑞上证红利ETF】 |
| 信号类型 | 卖出信号/买入信号 | 卖出信号 |
| 策略ID | 策略X-XXX | 策略1-阶梯 |

### 发送方式

```python
import base64
import hashlib
import hmac
import time
import urllib.parse
import urllib.request

def send_dingtalk_alert(webhook_url: str, secret: str, message: str):
    """
    发送钉钉告警消息（加签安全模式）

    Args:
        webhook_url: 钉钉机器人webhook地址
        secret: 钉钉机器人密钥
        message: 告警消息内容
    """
    # 加签计算
    timestamp = str(round(time.time() * 1000))
    string_to_sign = timestamp + '\n' + secret
    hmac_code = hmac.new(
        secret.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    # 构建完整URL
    url = f"{webhook_url}&timestamp={timestamp}&sign={sign}"

    # 发送消息
    data = {
        "msgtype": "text",
        "text": {
            "content": message
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    response = urllib.request.urlopen(req)
    return response.read()
```

## 五、回测日志格式

### 回测执行日志

```python
# 回测开始
logger.info("=" * 60)
logger.info(f"回测开始: {symbol} ({stock_name})")
logger.info(f"回测区间: {start_date} ~ {end_date}")
logger.info(f"策略配置: buy_ids={buy_ids}, sell_ids={sell_ids}")
logger.info("=" * 60)

# 数据准备
logger.info(f"获取历史数据: {len(df)} 条记录")

# 指标计算
logger.info(f"计算日线RSI完成")
logger.info(f"计算周线RSI完成")
logger.info(f"计算SAR指标完成")
logger.info(f"检测背离信号完成")

# 交易执行
logger.info(f"[{date}] 执行买入: {amount}股 @ {price}")
logger.info(f"[{date}] 执行卖出: {amount}股 @ {price}")

# 回测结束
logger.info("=" * 60)
logger.info("回测完成 - 绩效指标")
logger.info(f"年化收益率: {annual_return:.2%}")
logger.info(f"夏普比率: {sharpe_ratio:.2f}")
logger.info(f"最大回撤: {max_drawdown:.2%}")
logger.info(f"卡玛比率: {calmar_ratio:.2f}")
logger.info("=" * 60)
```

### 背离检测日志

```python
# 日线背离检测
logger.info("[日线] 检测到 X 个峰点")
logger.info(f"  [峰点1] {date} | 最高价={price:.3f} | MACD={macd:.4f}")
logger.info(f"[日线顶背离形成] {date} | 高点A: ... | 高点B: ... | 间隔={days}天")

# 周线背离检测
logger.info("[周线] 检测到 X 个峰点")
logger.info(f"[周线顶背离形成] {date} | ...")

# 背离生效确认
logger.info(f"[{timeframe}顶背离生效] {date} 形成的背离在 {confirm_date} 均线跌破确认生效")
```

## 六、日志文件管理

### 文件命名规范

```python
# 日志文件命名格式
log_filename = f"{module_name}_{timestamp}.log"

# 示例
log_filename = "backtest_20260430_143052.log"
log_filename = "monitor_20260430.log"
log_filename = "divergence_20260430_143052.log"
```

### 目录结构

```
logs/
├── backtest/
│   └── backtest_YYYYMMDD_HHMMSS.log
├── monitor/
│   └── monitor_YYYYMMDD.log
├── divergence/
│   └── divergence_YYYYMMDD_HHMMSS.log
└── error/
    └── error_YYYYMMDD.log
```

### 日志清理策略

```python
# 日志保留策略
LOG_RETENTION_DAYS = 30  # 保留30天

def cleanup_old_logs(log_dir: str, retention_days: int = 30):
    """清理过期日志文件"""
    import os
    from datetime import datetime, timedelta

    cutoff_date = datetime.now() - timedelta(days=retention_days)

    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)
        if os.path.isfile(filepath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_mtime < cutoff_date:
                os.remove(filepath)
                logger.info(f"清理过期日志: {filename}")
```

## 七、异常日志规范

### 异常捕获格式

```python
try:
    # 业务逻辑
    result = process_data(data)
except ValueError as e:
    # 具体异常处理，不暴露底层堆栈
    logger.warning(f"数据处理异常: {e}, 使用默认值")
    result = get_default_value()
except ConnectionError as e:
    # 网络异常，尝试降级方案
    logger.warning(f"API连接失败: {e}, 尝试读取缓存")
    result = read_from_cache()
except Exception as e:
    # 未预期异常，记录详细信息但不暴露给用户
    logger.error(f"未预期异常: {type(e).__name__}: {e}")
    result = safe_default_value()
```

### 错误日志级别

| 异常类型 | 日志级别 | 处理方式 |
|---------|---------|---------|
| 数据格式错误 | WARNING | 使用默认值，继续执行 |
| API连接失败 | WARNING | 读取缓存/重试 |
| 指标计算异常 | ERROR | 跳过当前数据点 |
| 配置缺失 | ERROR | 使用安全默认值 |
| 未预期异常 | ERROR | 记录详情，安全降级 |

## 八、日志输出代码模板

```python
def format_sell_log(
    date: str,
    amount: int,
    price: float,
    daily_rsi: float,
    weekly_rsi: float,
    stock_name: str,
    strategy_id: str,
    detail: str
) -> str:
    """
    格式化卖出日志

    Args:
        date: 卖出日期
        amount: 卖出数量
        price: 卖出价格
        daily_rsi: 日线RSI
        weekly_rsi: 周线RSI
        stock_name: 股票名称
        strategy_id: 策略ID（如 "策略1-阶梯"）
        detail: 详细信息

    Returns:
        格式化的日志字符串
    """
    return f"[{date}] 卖出：{amount}股 @ {price:.3f} | " \
           f"日RSI:{daily_rsi:.2f} | 周RSI:{weekly_rsi:.2f} | " \
           f"【{stock_name}】卖出信号 ({strategy_id}): {detail}"


def format_buy_log(
    date: str,
    amount: int,
    price: float,
    daily_rsi: float,
    weekly_rsi: float,
    stock_name: str,
    buy_type: str,  # "新资金买入" 或 "做T买回"
    strategy_id: str,
    detail: str
) -> str:
    """
    格式化买入日志

    Args:
        date: 买入日期
        amount: 买入数量
        price: 买入价格
        daily_rsi: 日线RSI
        weekly_rsi: 周线RSI
        stock_name: 股票名称
        buy_type: 买入类型
        strategy_id: 策略ID
        detail: 详细信息

    Returns:
        格式化的日志字符串
    """
    return f"[{date}] 买入：{amount}股 @ {price:.3f} | " \
           f"日RSI:{daily_rsi:.2f} | 周RSI:{weekly_rsi:.2f} | " \
           f"【{stock_name}】{buy_type}信号 ({strategy_id}): {detail}"


def format_dingtalk_alert(
    date: str,
    daily_rsi: float,
    weekly_rsi: float,
    stock_name: str,
    signal_type: str,
    strategy_id: str,
    detail: str
) -> str:
    """
    格式化钉钉告警消息

    Args:
        date: 日期
        daily_rsi: 日线RSI
        weekly_rsi: 周线RSI
        stock_name: 股票名称
        signal_type: 信号类型（买入/卖出）
        strategy_id: 策略ID
        detail: 详细信息

    Returns:
        格式化的钉钉消息
    """
    return f"{date} | 日RSI:{daily_rsi:.2f} | 周RSI:{weekly_rsi:.2f} | " \
           f"【{stock_name}】{signal_type} ({strategy_id}): {detail}"
```

## 九、指数RSI与背离日志格式

### 9.1 指数日常监控日志

每日监控必须输出所有指数的RSI状态作为环境信息。

**基本格式**：
```
{日期} | 【{指数名称}】日线RSI:{日线RSI值} | 周线RSI:{周线RSI值} | 状态:{状态描述}
```

**状态描述关键字**：
| 状态 | RSI条件 | 优先级 |
|-----|---------|--------|
| 极度超买-高风险 | 周线RSI > 80 | 最高 |
| 超买预警 | 周线RSI > 70 | 高 |
| 极度超卖-强买入 | 周线RSI < 25 | 高 |
| 超卖关注 | 周线RSI < 30 | 中 |
| 日线极度超买 | 日线RSI > 80（周线正常） | 中 |
| 日线超买 | 日线RSI > 70（周线正常） | 低 |
| 日线极度超卖 | 日线RSI < 20（周线正常） | 中 |
| 日线超卖 | 日线RSI < 30（周线正常） | 低 |
| 正常 | 其他情况 | - |

**日志示例**：
```
# 正常状态
2026-04-03 | 【上证指数】日线RSI:45.30 | 周线RSI:38.20 | 状态:正常

# 超买预警
2026-04-03 | 【创业板指数】日线RSI:75.20 | 周线RSI:82.50 | 状态:超买预警

# 超卖关注
2026-04-03 | 【科创综指】日线RSI:28.50 | 周线RSI:25.30 | 状态:极度超卖-强买入
```

### 9.2 指数背离日志格式

**日线顶背离形成**：
```
{日期} | 【{指数名称}】顶背离形成 (日线): 高点A({日期A}, 指数={价格A}, MACD={MACD_A}) vs 高点B({日期B}, 指数={价格B}, MACD={MACD_B})
```

**周线顶背离形成**：
```
{日期} | 【{指数名称}】顶背离形成 (周线): 高点A({日期A}, 指数={价格A}, MACD={MACD_A}) vs 高点B({日期B}, 指数={价格B}, MACD={MACD_B})
```

**日志示例**：
```
# 日线顶背离
2026-04-03 | 【创业板指数】顶背离形成 (日线): 高点A(2026-03-15, 指数=2200.00, MACD=25.00) vs 高点B(2026-03-28, 指数=2250.00, MACD=18.00)

# 周线顶背离
2026-04-03 | 【上证指数】顶背离形成 (周线): 高点A(2026-03-01, 指数=3100.00, MACD=30.00) vs 高点B(2026-03-15, 指数=3150.00, MACD=20.00)
```

### 9.3 指数监控输出顺序

在每日监控执行时，日志输出顺序应为：

```
===== 指数环境监控 =====
{所有指数RSI状态日志}

===== 指数背离告警 =====
{如有背离形成，输出背离详情}

===== 个股监控告警 =====
{个股买卖信号告警}
```

**完整示例**：
```
===== 指数环境监控 =====
2026-04-03 | 【创业板指数】日线RSI:75.20 | 周线RSI:82.50 | 状态:超买预警
2026-04-03 | 【上证指数】日线RSI:45.30 | 周线RSI:38.20 | 状态:正常
2026-04-03 | 【科创综指】日线RSI:28.50 | 周线RSI:25.30 | 状态:极度超卖-强买入
2026-04-03 | 【北证板块】日线RSI:55.20 | 周线RSI:48.50 | 状态:正常
2026-04-03 | 【恒生科技ETF】日线RSI:42.30 | 周线RSI:35.20 | 状态:正常
2026-04-03 | 【恒生指数ETF】日线RSI:38.50 | 周线RSI:32.10 | 状态:正常

===== 指数背离告警 =====
2026-04-03 | 【创业板指数】顶背离形成 (日线): 高点A(2026-03-15, 指数=2200.00, MACD=25.00) vs 高点B(2026-03-28, 指数=2250.00, MACD=18.00)

===== 个股监控告警 =====
2026-04-03 | 日RSI:65.30 | 周RSI:75.20 | 【华泰柏瑞上证红利ETF】卖出信号 (策略1-阶梯): 触发 [SAR绿转红 (2026-04-03)]，周线RSI flag=2，周线RSI峰值:2026-03-28=88.50(股价=3.350)，建议卖出1/2
```

### 9.4 指数监控代码模板

```python
def format_index_rsi_log(
    date: str,
    index_name: str,
    daily_rsi: float,
    weekly_rsi: float,
    status: str
) -> str:
    """
    格式化指数RSI状态日志

    Args:
        date: 日期
        index_name: 指数名称
        daily_rsi: 日线RSI值
        weekly_rsi: 周线RSI值
        status: 状态描述

    Returns:
        格式化的日志字符串
    """
    return f"{date} | 【{index_name}】日线RSI:{daily_rsi:.2f} | 周线RSI:{weekly_rsi:.2f} | 状态:{status}"


def format_index_divergence_log(
    date: str,
    index_name: str,
    timeframe: str,  # "日线" 或 "周线"
    peak_a_date: str,
    peak_a_price: float,
    peak_a_macd: float,
    peak_b_date: str,
    peak_b_price: float,
    peak_b_macd: float
) -> str:
    """
    格式化指数背离日志

    Args:
        date: 当前日期
        index_name: 指数名称
        timeframe: 时间周期
        peak_a_date: 高点A日期
        peak_a_price: 高点A价格
        peak_a_macd: 高点A MACD值
        peak_b_date: 高点B日期
        peak_b_price: 高点B价格
        peak_b_macd: 高点B MACD值

    Returns:
        格式化的日志字符串
    """
    return f"{date} | 【{index_name}】顶背离形成 ({timeframe}): " \
           f"高点A({peak_a_date}, 指数={peak_a_price:.2f}, MACD={peak_a_macd:.2f}) vs " \
           f"高点B({peak_b_date}, 指数={peak_b_price:.2f}, MACD={peak_b_macd:.2f})"
```

## 十、自测检查清单

在提交日志相关代码前，必须完成以下自测：

1. **格式一致性检查**：
   - 卖出日志是否包含所有必要字段？
   - 买入日志是否区分新资金买入和做T买回？
   - 钉钉消息格式是否符合规范？
   - 指数RSI日志是否包含状态描述？

2. **日志级别检查**：
   - INFO用于正常业务流程？
   - WARNING用于可恢复异常？
   - ERROR用于严重错误？
   - 未暴露底层堆栈给用户？

3. **异常处理检查**：
   - 是否捕获具体异常而非笼统的Exception？
   - 是否提供降级方案？
   - 是否记录足够的调试信息？

4. **文件管理检查**：
   - 日志文件命名是否符合规范？
   - 是否有清理策略防止日志堆积？

5. **指数监控检查**：
   - 是否每日输出所有指数的RSI状态？
   - 指数背离日志格式是否正确？
   - 输出顺序是否符合规范（指数→个股）？