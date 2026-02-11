# hk-stock-allocation

配置驱动的港股分配与估值信号工具，基于 RQData 港股数据。

## 功能

- 从 `configs/portfolio.yml` 读取股票清单、资金规模和参数
- 按港股一手(`round_lot`)约束计算每只股票可买手数和金额偏差
- 分配定价采用分层回退：`snapshot` -> `1m close` -> `1d close`
- 输出估值分层（统计价格高位，不是基本面估值）和卖出触发时间
- `secondary_fill` 支持现金缓冲、估算费用、超配上限和严格规则约束
- 生成 Excel 报表（工作表名：`分配` / `汇总` / `卖出信号`）

## 环境要求

- Python `>=3.10`
- `uv`
- 可用的 RQData 账号和本机登录配置（`rqdatac.init()`）

## 安装

```bash
uv venv
uv sync
```

开发测试依赖：

```bash
uv sync --group dev
```

## 配置

默认配置文件：`configs/portfolio.yml`

关键字段：

- `portfolio.total_capital`: 组合总资金（HKD）
- `portfolio.allocation.method`: `equal` 或 `custom`
- `portfolio.allocation.secondary_fill`: 逐手补仓参数（见下方策略说明）
- `portfolio.trading.require_stock_connect`: 是否只允许港股通标的
- `portfolio.valuation`: 历史窗口和估值阈值参数
- `tickers[].ticker`: 股票代码（格式 `00941.HK`）

`custom` 分配时，需为每个启用标的提供 `weight`。

`secondary_fill` 常用参数：

- `enabled`: 是否启用补仓
- `avoid_high_valuation`: 尽量避开 `HIGH/EXTREME`（软规则）
- `avoid_high_valuation_strict`: 严格避开 `HIGH/EXTREME`（硬规则）
- `max_steps`: 最大补仓步数（默认 5000，实际还会受现金可买上限约束）
- `cash_buffer_ratio` / `cash_buffer_amount`: 补仓时预留现金缓冲
- `estimated_fee_per_order`: 每次补仓附加的估算交易费用
- `allow_over_alloc`: 是否允许轻微超配（默认否）
- `max_over_alloc_ratio` / `max_over_alloc_amount`: 超配金额上限
- `max_over_alloc_lots_per_ticker`: 单标的允许超配的补仓手数上限

## 运行

```bash
uv run hk-alloc --config configs/portfolio.yml
```

可选参数：

```bash
uv run hk-alloc --config configs/portfolio.yml --as-of 2026-02-10 --output output/manual.xlsx
```

也可以直接用脚本入口：

```bash
./scripts/run.sh --as-of 2026-02-10
```

`scripts` 只是便捷入口，不是必需入口。

## 输出说明

`allocation` 表核心字段：

- `ticker`: 输入代码（兼容历史项目）
- `order_book_id`: RQData 查询代码（`XXXXX.XHKG`）
- `price`, `price_source`, `pricing_date`
- `round_lot`, `lots_base`, `lots_extra`, `lots`, `shares`
- `est_value`, `gap_to_target`, `gap_ratio`
- `lot_cost`: 单手成本
- `valuation`: `LOW/NEUTRAL/HIGH/EXTREME`
- `overpriced_low` / `overpriced_high`: 统计高位阈值区间（未复权历史价格尺度）

导出到 Excel 时会自动转换为中文列名与中文值（如估值分层、价格来源、港股通/可交易状态）。
`allocation` 表默认左侧优先列：`股票代码`、`名称`、`合计手数`、`当前价格`、`估值分层`、`统计高位上沿(未复权)`。

`secondary_fill` 当前策略：

- 只对 `gap_to_target > 0` 的标的尝试补仓
- 买入后必须满足 `abs(new_gap) < abs(old_gap)`，否则不买
- 默认不允许超配；`allow_over_alloc=true` 时，受超配比例/金额/手数上限约束
- 补仓可用资金 = `total_capital - cash_buffer`，且每步会额外计入 `estimated_fee_per_order`
- `avoid_high_valuation_strict=false` 时：先尝试 LOW/NEUTRAL，若无可选才回退 HIGH/EXTREME
- `avoid_high_valuation_strict=true` 时：直接禁止 HIGH/EXTREME

`summary` 表：

- `total_est_value`, `total_gap`, `cash_used_ratio`
- `pricing_source`, `pricing_source_detail`
- `secondary_fill_steps`, `secondary_fill_spent`, `secondary_fill_fee_spent`
- `secondary_fill_cash_buffer`, `secondary_fill_budget_after_buffer`, `cash_remaining_after_fill`
- `cash_remaining_after_fill` 已扣除补仓买入金额与 `secondary_fill_fee_spent`
`summary` 表默认左侧优先列：`组合名称`、`统计日期`、`定价日期`、`价格来源`、`价格来源明细`、`标的数量`。

`sell_signals` 表：

- `sell_trigger`: 偏高阈值
- `extreme_trigger`: 极高阈值
- `last_sell_signal_date`: 最近一次上穿卖出阈值日期
- 卖出信号按“前一交易日已确定阈值”判断：`close_t >= trigger_{t-1}` 且 `close_{t-1} < trigger_{t-1}`
`sell_signals` 表默认左侧优先列：`股票代码`、`名称`、`前复权收盘价`、`估值分层`、`偏高阈值`、`极高阈值`。

## 测试

```bash
uv run pytest
```

或使用脚本：

```bash
./scripts/test.sh
```

运行集成测试（需要真实 RQData）：

```bash
RQDATAC_ENABLED=1 uv run pytest tests/test_integration_rqdata.py
```

## 免责声明

本项目仅用于研究和工程示例，不构成任何投资建议。
