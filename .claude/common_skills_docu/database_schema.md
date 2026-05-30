# 数据库表结构规范

本文档记录量化交易项目涉及的数据库表结构，供数据读写代码参考。

---

## 数据库信息

| 属性 | 值 |
|-----|------|
| 数据库类型 | MySQL 8.0 |
| 字符集 | utf8mb4 |
| 排序规则 | utf8mb4_unicode_ci |

---

## 一、表结构概览

| 表名 | 用途 | 主要字段 |
|-----|------|---------|
| `stock_info` | 股票基本信息 | stock_id, stock_name, listing_date |
| `stock_daily_metrics` | 个股日线指标 | stock_id, trade_date, 各项指标 |
| `stock_index_daily_metrics` | 指数日线指标 | stock_id, trade_date, 各项指标 |
| `trading_calendar` | 交易日历 | trade_date |

---

## 二、stock_info（股票基本信息表）

### 表结构

| 字段 | 类型 | 说明 | 约束 |
|-----|------|------|------|
| `stock_id` | varchar(10) | 股票代码（如600000） | 主键、唯一索引 |
| `stock_name` | varchar(50) | 股票名称 | NOT NULL |
| `listing_date` | date | 上市日期 | NOT NULL |
| `created_at` | datetime | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | datetime | 更新时间 | ON UPDATE CURRENT_TIMESTAMP |

### 索引

| 索引名 | 字段 | 类型 |
|-------|------|------|
| `stock_id` | stock_id | UNIQUE |
| `uk_stock_id` | stock_id | UNIQUE |
| `idx_stock_code` | stock_id | INDEX |
| `idx_listing_date` | listing_date | INDEX |

### 使用场景

- 获取股票名称（用于日志输出）
- 过滤退市股票（listing_date判断）
- 关联其他表的股票信息

---

## 三、stock_daily_metrics（个股日线指标表）

### 表结构

| 字段 | 类型 | 说明 | 约束 |
|-----|------|------|------|
| `id` | bigint UNSIGNED | 自增主键 | PRIMARY KEY |
| `stock_id` | varchar(10) | 股票代码 | NOT NULL, 关联stock_info |
| `trade_date` | date | 交易日 | NOT NULL |
| `close_price` | decimal(10,2) | 收盘价 | NOT NULL |
| `low_price` | decimal(10,2) | 最低价 | NULL |
| `ma5` | decimal(10,2) | 5日均线 | NULL |
| `ma10` | decimal(10,2) | 10日均线 | NULL |
| `ma5_ma10_dead_cross` | tinyint(1) | MA5/MA10死叉 | DEFAULT 0 (0=否, 1=是) |
| `rsi_daily` | decimal(5,2) | 日线RSI | NULL (范围0~100) |
| `rsi_weekly` | decimal(5,2) | 周线RSI | NULL |
| `macd` | decimal(10,4) | MACD柱值 | NULL |
| `dif` | decimal(10,4) | DIF值（快线） | NULL |
| `dea` | decimal(10,4) | DEA值（慢线） | NULL |
| `sar` | decimal(10,4) | SAR值 | NULL |
| `is_daily_top_divergence` | tinyint(1) | 日线顶背离 | DEFAULT 0 |
| `is_daily_bottom_divergence` | tinyint(1) | 日线底背离 | DEFAULT 0 |
| `is_weekly_top_divergence` | tinyint(1) | 周线顶背离 | DEFAULT 0 |
| `is_weekly_bottom_divergence` | tinyint(1) | 周线底背离 | DEFAULT 0 |
| `is_sar_dead_cross` | tinyint(1) | SAR死叉 | DEFAULT 0 |
| `is_local_high` | tinyint(1) | 局部高点 | DEFAULT 0 |
| `is_local_low` | tinyint(1) | 局部低点 | DEFAULT 0 |
| `status` | tinyint(1) | 计算状态 | DEFAULT 0 (0=完成, 1=有遗漏) |
| `created_at` | datetime | 创建时间 | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | datetime | 更新时间 | ON UPDATE CURRENT_TIMESTAMP |

### 索引

| 索引名 | 字段 | 类型 |
|-------|------|------|
| `PRIMARY` | id | PRIMARY KEY |
| `uk_stock_date` | stock_id, trade_date | UNIQUE |
| `idx_trade_date` | trade_date | INDEX |
| `idx_stock_status` | stock_id, status | INDEX |
| `idx_date_status` | trade_date, status | INDEX |

### 字段分类

**基础数据字段**：
- `stock_id`, `trade_date`, `close_price`, `low_price`

**指标计算字段**：
- `ma5`, `ma10`, `rsi_daily`, `rsi_weekly`
- `macd`, `dif`, `dea`, `sar`

**信号标记字段**：
- `ma5_ma10_dead_cross` - 均线死叉
- `is_sar_dead_cross` - SAR死叉
- `is_daily_top_divergence` / `is_weekly_top_divergence` - 顶背离
- `is_local_high` / `is_local_low` - 局部高低点

**状态字段**：
- `status` - 计算完成状态

---

## 四、stock_index_daily_metrics（指数日线指标表）

### 表结构

与 `stock_daily_metrics` 结构类似，用于记录大盘指数（上证、创业板等）的指标数据。

| 字段 | 类型 | 说明 | 与个股表差异 |
|-----|------|------|-------------|
| `stock_id` | varchar(10) | 指数代码 | 如 SHSE.000001（上证指数） |
| `trade_date` | date | 交易日 | 同个股表 |
| `close_price` | decimal(10,2) | 收盘价 | 同个股表 |
| `ma5` | decimal(10,2) | 5日均线 | 同个股表 |
| `ma10` | decimal(10,2) | 10日均线 | 同个股表 |
| `ma5_ma10_dead_cross` | tinyint(1) | 均线死叉 | 同个股表 |
| `rsi_daily` | decimal(5,2) | 日线RSI | 同个股表 |
| `rsi_weekly` | decimal(5,2) | 周线RSI | 同个股表 |
| `macd` | decimal(10,4) | MACD值 | 同个股表 |
| `sar` | decimal(10,4) | SAR值 | 同个股表 |
| `sar_line` | decimal(10,4) | SAR金叉死叉临界值 | **新增字段** |
| `is_daily_top_divergence` | tinyint(1) | 日线顶背离 | 同个股表 |
| `is_daily_bottom_divergence` | tinyint(1) | 日线底背离 | 同个股表 |
| `is_weekly_top_divergence` | tinyint(1) | 周线顶背离 | 同个股表 |
| `is_weekly_bottom_divergence` | tinyint(1) | 周线底背离 | 同个股表 |
| `is_sar_dead_cross` | tinyint(1) | SAR死叉 | 同个股表 |
| `is_local_high` | tinyint(1) | 局部高点 | 同个股表 |
| `is_local_low` | tinyint(1) | 局部低点 | 同个股表 |
| `status` | tinyint(1) | 计算状态 | 同个股表 |

### 主要指数代码

| 指数名称 | 代码 | 用途 |
|---------|------|------|
| 上证指数 | SHSE.000001 / 000001 | sell_id=4,7 判断 |
| 创业板指数 | SZSE.399006 / 399006 | sell_id=5,8 判断 |
| 科创板指数 | SHSE.000680 / 000680 | sell_id=6,9 判断 |

### 使用场景

- sell_id=4~6：判断指数背离 + 个股RSI阈值
- sell_id=7~9：以指数SAR/MACD死叉为触发条件
- buy_id=3,5：创业板指数RSI保护型买入
- buy_id=6,7：上证指数RSI保护型买入
- buy_id=8：科创板指数RSI保护型买入

---

## 五、trading_calendar（交易日历表）

### 表结构

| 字段 | 类型 | 说明 | 约束 |
|-----|------|------|------|
| `trade_date` | date | 交易日 | PRIMARY KEY |

### 使用场景

- 判断某日期是否为交易日
- 获取最近N个交易日
- 监控脚本判断当天是否需要执行

---

## 六、字段类型说明

### decimal精度

| 字段类型 | 精度 | 适用场景 |
|---------|------|---------|
| decimal(10,2) | 10位总长，2位小数 | 价格字段（收盘价、均线） |
| decimal(5,2) | 5位总长，2位小数 | RSI值（0~100范围） |
| decimal(10,4) | 10位总长，4位小数 | MACD、SAR（需要高精度） |

### tinyint标记

| 值 | 含义 |
|---|------|
| 0 | 否/未触发 |
| 1 | 是/已触发 |

### status状态

| 值 | 含义 |
|---|------|
| 0 | 计算完成 |
| 1 | 有遗漏（需补充计算） |

---

## 七、常用查询示例

### 获取个股最近N天指标

```sql
SELECT stock_id, trade_date, close_price, rsi_daily, rsi_weekly,
       sar, is_sar_dead_cross, is_daily_top_divergence
FROM stock_daily_metrics
WHERE stock_id = '510880'
  AND trade_date >= '2026-04-01'
ORDER BY trade_date DESC
LIMIT 10;
```

### 获取最近SAR死叉日期

```sql
SELECT stock_id, trade_date, close_price, sar
FROM stock_daily_metrics
WHERE stock_id = '510880'
  AND is_sar_dead_cross = 1
ORDER BY trade_date DESC
LIMIT 5;
```

### 获取指数背离状态

```sql
SELECT stock_id, trade_date, is_daily_top_divergence, is_weekly_top_divergence
FROM stock_index_daily_metrics
WHERE stock_id = '000001'  -- 上证指数
  AND trade_date = '2026-04-30';
```

### 获取最近交易日

```sql
SELECT trade_date
FROM trading_calendar
WHERE trade_date <= CURDATE()
ORDER BY trade_date DESC
LIMIT 5;
```

---

## 八、数据写入规范

### 批量插入示例

```python
# 使用pandas批量写入
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql://user:pass@host:3306/stock')

df_metrics.to_sql(
    'stock_daily_metrics',
    engine,
    if_exists='append',
    index=False,
    chunksize=1000
)
```

### 更新状态示例

```python
# 更新计算状态
UPDATE stock_daily_metrics
SET status = 0,
    updated_at = NOW()
WHERE stock_id = '510880'
  AND trade_date BETWEEN '2026-01-01' AND '2026-04-30';
```

---

## 九、表关系图

```
stock_info (股票基本信息)
    │
    ├── stock_id ──────────────────┐
    │                              │
    │                              ▼
    │              stock_daily_metrics (个股指标)
    │              - 同一 stock_id 的日线数据
    │              - UNIQUE(stock_id, trade_date)
    │
    │                              ▼
    │              stock_index_daily_metrics (指数指标)
    │              - 指数代码存储在 stock_id
    │              - 结构与个股表类似
    │
    ▼
trading_calendar (交易日历)
    - 独立表，无外键关联
    - 用于判断交易日
```

---

## 十、数据维护规范

### 每日更新流程

1. 从掘金API获取最新交易数据
2. 计算各项指标（RSI、MACD、SAR、MA）
3. 检测信号（死叉、背离、局部高低点）
4. 写入/更新 `stock_daily_metrics` 表
5. 同步更新 `stock_index_daily_metrics` 表

### 数据校验

- `status` 字段标记计算完整性
- 定期检查遗漏数据（status=1）
- 补充计算后更新 status=0

---

## 十一、Python代码对应

| 表 | 读写代码位置 |
|-----|-------------|
| stock_daily_metrics | `src/runner/db_backtest_runner.py`, `src/runner/fill_db_tables.py` |
| stock_index_daily_metrics | `src/runner/fill_index_tables.py` |
| trading_calendar | 数据获取时自动判断 |

---

## 十二、新增字段规范

如需新增指标字段，遵循以下规则：

1. **命名规范**：使用小写字母+下划线（如 `new_indicator`）
2. **类型选择**：
   - 价格类：decimal(10,2)
   - RSI类：decimal(5,2)
   - MACD/SAR类：decimal(10,4)
   - 标记类：tinyint(1)
3. **默认值**：标记字段 DEFAULT 0，数据字段 NULL
4. **注释**：添加 COMMENT 说明字段用途
5. **索引**：如需频繁查询，添加 INDEX