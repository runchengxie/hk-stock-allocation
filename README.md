# hk-stock-allocation

配置驱动的港股分配与估值信号工具，基于 RQData 港股数据。

## 功能

- 从 `configs/portfolio.yml` 读取股票清单、资金规模和参数
- 按港股一手(`round_lot`)约束计算每只股票可买手数和金额偏差
- 分配定价采用分层回退：`snapshot` -> `1m close` -> `1d close`
- 输出估值分层（统计高位）和卖出触发时间
- 生成 Excel 报表（`allocation` / `summary` / `sell_signals`）

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
- `portfolio.allocation.secondary_fill`: 逐手补仓参数
- `portfolio.trading.require_stock_connect`: 是否只允许港股通标的
- `portfolio.valuation`: 历史窗口和估值阈值参数
- `tickers[].ticker`: 股票代码（格式 `00941.HK`）

`custom` 分配时，需为每个启用标的提供 `weight`。

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

## 输出说明

`allocation` 表核心字段：

- `ticker`: 输入代码（兼容历史项目）
- `order_book_id`: RQData 查询代码（`XXXXX.XHKG`）
- `price`, `price_source`, `pricing_date`
- `round_lot`, `lots_base`, `lots_extra`, `lots`, `shares`
- `est_value`, `gap_to_target`, `gap_ratio`
- `lot_cost`: 单手成本
- `valuation`: `LOW/NEUTRAL/HIGH/EXTREME`
- `overpriced_low` / `overpriced_high`: 统计高位阈值区间

导出到 Excel 时会自动转换为中文列名与中文值（如估值分层、价格来源、港股通/可交易状态）。
`allocation` 表默认左侧优先列：`股票代码`、`合计手数`、`估值分层`、`高估上沿`。

启用 `secondary_fill` 后，个别标的 `gap_to_target` 可能为负（代表超配一手）；
但组合层面的 `total_gap` 会尽量压低到“剩余现金 < 任一可买一手成本”。

`summary` 表：

- `total_est_value`, `total_gap`, `cash_used_ratio`
- `pricing_source`, `pricing_source_detail`
- `secondary_fill_steps`, `secondary_fill_spent`, `cash_remaining_after_fill`

`sell_signals` 表：

- `sell_trigger`: 偏高阈值
- `extreme_trigger`: 极高阈值
- `last_sell_signal_date`: 最近一次上穿卖出阈值日期

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
