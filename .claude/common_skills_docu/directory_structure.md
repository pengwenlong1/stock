# 项目目录结构规范

本文档记录量化交易项目的目录结构和各模块职责，供代码编写和组织参考。

---

## 一、顶层目录结构

```
stock/                           ← 项目根目录
├── .claude/                      ← Claude配置目录
│   ├── CLAUDE.md                 ← 项目整体规范
│   ├── settings.json             ← Claude设置
│   ├── common_skills_docu/       ← 公共文档目录
│   │   ├── buy_sell.md           ← 买入卖出策略规则
│   │   ├── indicators_calculation.md ← 指标计算规则
│   │   └── directory_structure.md ← 目录结构规范
│   └── skills/                   ← Skills目录
│       ├── backtest-strategy/    ← 回测策略skill
│       ├── daily-monitoring/     ← 每日监控skill
│       └── logging-output/       ← 日志输出skill
│
├── config/                       ← 配置文件目录
│   ├── settings.json             ← 全局设置（GM Token等）
│   ├── stocks_monitor_by_day.csv ← 监控股票池配置
│   ├── stocks_backtest.csv       ← 回测股票池配置
│   └── stocks_process.csv        ← 数据处理配置
│
├── docu/                         ← 文档目录
│   ├── buy_sell.md               ← 买入卖出策略文档
│   ├── 背离规则.md               ← 背离判断规则
│   ├── 脚本代码要求.md           ← 日志格式规范
│   └── 每日监控的逻辑.md         ← 监控逻辑说明
│
├── logs/                         ← 日志存储目录
│   ├── backtest/                 ← 回测日志
│   │   └── backtest_YYYYMMDD_HHMMSS.log
│   ├── monitor/                  ← 监控日志
│   │   └── monitor_YYYYMMDD.log
│   ├── divergence/               ← 背离检测日志
│   │   └── divergence_YYYYMMDD_HHMMSS.log
│   └── error/                    ← 错误日志
│       └── error_YYYYMMDD.log
│
├── src/                          ← 源代码目录
│   ├── runner/                   ← 执行脚本目录
│   │   ├── monitor_by_day.py     ← 每日监控脚本
│   │   ├── backtest_runner.py    ← 回测执行脚本
│   │   ├── db_backtest_runner.py ← 数据库回测脚本
│   │   └── fill_db_tables.py     ← 数据库填充脚本
│   │
│   ├── tool/                     ← 工具模块目录
│   │   ├── strategy.py           ← 交易策略类
│   │   ├── rsi_calculator.py     ← RSI计算器
│   │   ├── macd_calculator.py    ← MACD计算器
│   │   ├── sar_strategy.py       ← SAR策略类
│   │   ├── ma_calculator.py      ← MA计算器
│   │   ├── divergence_detector.py ← 背离检测器
│   │   └── peak_detector.py      ← 峰点检测器
│   │
│   ├── process/                  ← 数据处理目录
│   │   └── data_processor.py     ← 数据处理模块
│   │
│   └── test/                     ← 测试脚本目录
│       ├── test_monitor_by_day.py ← 监控测试
│       ├── test_sar_calculation.py ← SAR计算测试
│       └── test_dingtalk.py      ← 钉钉通知测试
│
├── cankao/                       ← 参考资料目录
│   ├── RSI.md                    ← RSI参考资料
│   └── 背离判断.md               ← 背离判断参考
│
└── requirements.txt              ← Python依赖
```

---

## 二、核心模块职责

### 1. src/runner/ - 执行脚本

| 文件 | 职责 | 说明 |
|-----|------|------|
| `monitor_by_day.py` | 每日监控执行 | 数据获取、信号检测、钉钉告警 |
| `backtest_runner.py` | 回测执行 | 数据准备、交易模拟、绩效报告 |
| `db_backtest_runner.py` | 数据库回测 | 使用数据库数据的回测 |
| `fill_db_tables.py` | 数据填充 | 填充数据库历史数据 |

### 2. src/tool/ - 工具模块

| 文件 | 职责 | 输入 | 输出 |
|-----|------|------|------|
| `strategy.py` | 策略信号判断 | 指标数据 | BuySignal/SellSignal |
| `rsi_calculator.py` | RSI计算 | 收盘价 | RSI序列 |
| `macd_calculator.py` | MACD计算 | 收盘价 | DIF、DEA、MACD柱 |
| `sar_strategy.py` | SAR计算 | high/low/close | SAR序列、转折信号 |
| `divergence_detector.py` | 背离检测 | 价格、MACD | DivergenceSignal |
| `peak_detector.py` | 峰点检测 | high/low/close | PeakInfo列表 |

### 3. src/process/ - 数据处理

| 文件 | 职责 | 说明 |
|-----|------|------|
| `data_processor.py` | 数据预处理 | 时间序列转换、缺失值处理、周线聚合 |

### 4. src/test/ - 测试脚本

| 文件 | 测试内容 | 说明 |
|-----|---------|------|
| `test_monitor_by_day.py` | 监控逻辑 | 测试监控信号生成 |
| `test_sar_calculation.py` | SAR计算 | 测试SAR指标正确性 |
| `test_dingtalk.py` | 钉钉通知 | 测试告警消息发送 |

---

## 三、配置文件格式

### stocks_monitor_by_day.csv

```csv
symbol,judge_buy_ids,judge_t_ids,judge_sell_ids,active,name
510880,1,2;6,1;2,1,华泰柏瑞上证红利ETF
159873,,6,2;5;8,1,天弘中证全指医疗保健设备与服务ETF
```

字段说明：
- `symbol`: 股票代码（掘金格式，如 SHSE.510880）
- `judge_buy_ids`: 新资金买入策略ID（多个用分号分隔）
- `judge_t_ids`: 做T买回策略ID（多个用分号分隔）
- `judge_sell_ids`: 卖出策略ID（多个用分号分隔）
- `active`: 是否监控（1=监控，0=忽略）
- `name`: 股票名称

### settings.json

```json
{
    "gm_token": "your_gm_token_here",
    "dingtalk_webhook": "your_webhook_url",
    "dingtalk_secret": "your_secret_key"
}
```

---

## 四、日志文件规范

### 日志目录结构

```
logs/
├── backtest/                     ← 回测日志
│   └── backtest_SHSE512480_20260430_143052.log
│
├── monitor/                      ← 监控日志（按日期）
│   └── monitor_20260430.log
│
├── divergence/                   ← 背离检测日志
│   └── divergence_SHSE510880_20260430_143052.log
│
└── error/                        ← 错误日志
    └── error_20260430.log
```

### 日志文件命名规范

| 类型 | 命名格式 | 示例 |
|-----|---------|------|
| 回测日志 | `backtest_{symbol}_{timestamp}.log` | `backtest_SHSE512480_20260430_143052.log` |
| 监控日志 | `monitor_{date}.log` | `monitor_20260430.log` |
| 背离日志 | `divergence_{symbol}_{timestamp}.log` | `divergence_SHSE510880_20260430_143052.log` |
| 错误日志 | `error_{date}.log` | `error_20260430.log` |

### 日志保留策略

- 默认保留30天
- 回测日志可长期保留（便于复盘）
- 监控日志定期清理

---

## 五、代码模块化原则

### 职责分离

```
策略逻辑 → strategy.py
指标计算 → rsi_calculator.py, macd_calculator.py, sar_strategy.py
背离检测 → divergence_detector.py
执行脚本 → monitor_by_day.py, backtest_runner.py
```

### 模块调用关系

```
monitor_by_day.py
    ├── 调用 strategy.py（信号判断）
    ├── 调用 rsi_calculator.py（RSI计算）
    ├── 调用 sar_strategy.py（SAR计算）
    ├── 调用 divergence_detector.py（背离检测）
    └── 发送钉钉告警

backtest_runner.py
    ├── 预计算所有指标
    ├── 调用 strategy.py（信号判断）
    ├── 执行模拟交易
    └── 生成绩效报告
```

---

## 六、新增文件放置规则

| 文件类型 | 放置位置 | 说明 |
|---------|---------|------|
| 新策略ID实现 | `src/tool/strategy.py` | 添加到TradingStrategy类 |
| 新指标计算 | `src/tool/xxx_calculator.py` | 创建独立模块 |
| 新监控脚本 | `src/runner/monitor_xxx.py` | 监控变体 |
| 新回测脚本 | `src/runner/backtest_xxx.py` | 回测变体 |
| 测试脚本 | `src/test/test_xxx.py` | 单元测试 |
| 配置文件 | `config/xxx.csv` 或 `config/xxx.json` | 参数配置 |
| 文档 | `docu/xxx.md` | 策略说明文档 |
| 公共参考文档 | `.claude/common_skills_docu/xxx.md` | Skill引用文档 |

---

## 七、文件命名规范

### Python文件

- 模块文件：`xxx_calculator.py`（小写+下划线）
- 执行脚本：`xxx_runner.py` 或 `monitor_xxx.py`
- 测试文件：`test_xxx.py`

### Markdown文件

- 策略文档：`buy_sell.md`、`背离规则.md`
- 规范文档：`xxx_calculation.md`、`xxx_structure.md`

### 配置文件

- CSV配置：`stocks_xxx.csv`
- JSON配置：`xxx_settings.json`

### 日志文件

- 格式：`{module}_{symbol}_{timestamp}.log`
- 时间戳：`YYYYMMDD_HHMMSS`