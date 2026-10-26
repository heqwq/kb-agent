"""确定性生成评估用分析库 data/analytics.db。"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["华东", "华北", "华南", "西南"]
CHANNELS = ["天猫", "京东", "抖音", "线下", "企业直销"]
STATUSES = ["paid", "paid", "paid", "paid", "paid", "paid", "refunded", "pending", "cancelled"]
TIERS = ["普通", "白银", "黄金", "战略"]
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许"


def build(path: Path, seed: int = 20260901) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_name TEXT NOT NULL,
            region TEXT NOT NULL,
            tier TEXT NOT NULL,
            joined_at TEXT NOT NULL
        );
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            region TEXT NOT NULL,
            channel TEXT NOT NULL,
            amount REAL NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id)
        );
        CREATE INDEX idx_orders_date ON orders(created_at);
        CREATE INDEX idx_orders_region ON orders(region);
        """
    )
    customers = []
    for i in range(1, 121):
        region = rng.choices(REGIONS, weights=[34, 24, 24, 18])[0]
        joined = datetime(2024, 1, 1) + timedelta(days=rng.randrange(0, 1000))
        customers.append(
            (
                f"C{i:04d}",
                rng.choice(SURNAMES) + rng.choice("子涵雨晨宇欣嘉怡明轩思远"),
                region,
                rng.choices(TIERS, weights=[50, 30, 15, 5])[0],
                joined.strftime("%Y-%m-%d"),
            )
        )
    connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    start = datetime(2026, 7, 1)
    orders = []
    for i in range(1, 501):
        created = start + timedelta(
            days=rng.randrange(0, 92), hours=rng.randrange(8, 23), minutes=rng.randrange(0, 60)
        )
        region = rng.choices(REGIONS, weights=[34, 24, 24, 18])[0]
        customer = rng.choice(customers)
        orders.append(
            (
                f"SO2026{i:05d}",
                region,
                rng.choices(CHANNELS, weights=[30, 25, 20, 15, 10])[0],
                round(rng.lognormvariate(7.4, 0.55), 2),
                created.strftime("%Y-%m-%d %H:%M:%S"),
                rng.choice(STATUSES),
                customer[0],
            )
        )
    connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)
    connection.commit()
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成确定性分析库")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "analytics.db")
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    build(args.out, args.seed)
    print(f"已生成 {args.out}：120 位客户，500 笔订单，seed={args.seed}")


if __name__ == "__main__":
    main()
