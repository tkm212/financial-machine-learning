#!/usr/bin/env python3
"""
Generate time, tick, volume, and dollar bars from tick data.

Writes cleaned outputs to outputs/ for use in notebooks.
"""

import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from financial_machine_learning.bars import dollar_bars, tick_bars, time_bars, volume_bars
from scripts.convert_to_second_bars import load_tick_data


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    tick_path = root / "inputs" / "btc_bid_ask_data.csv"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_tick_data(tick_path)

    # Time bars (1-second)
    tb = time_bars(df)
    tb = tb[tb["datetime"].dt.year > 2000]  # drop bad rows
    tb.to_csv(out_dir / "btc_bid_ask_data_1s.csv", index=False)
    print(f"Wrote {len(tb)} time bars")

    # Tick bars (100 ticks per bar)
    tick_b = tick_bars(df, threshold=100)
    tick_b.to_csv(out_dir / "btc_bid_ask_data_tick_bars.csv", index=False)
    print(f"Wrote {len(tick_b)} tick bars")

    # Volume bars (~1000 BTC per bar, adjust to ~300 bars for similar granularity)
    vol_per_bar = df["Quantity"].sum() / 300
    vol_b = volume_bars(df, threshold=vol_per_bar)
    vol_b.to_csv(out_dir / "btc_bid_ask_data_volume_bars.csv", index=False)
    print(f"Wrote {len(vol_b)} volume bars (threshold={vol_per_bar:.2f})")

    # Dollar bars (~$2M per bar for similar count)
    dollar_per_bar = (df["Price"] * df["Quantity"]).sum() / 300
    dol_b = dollar_bars(df, threshold=dollar_per_bar)
    dol_b.to_csv(out_dir / "btc_bid_ask_data_dollar_bars.csv", index=False)
    print(f"Wrote {len(dol_b)} dollar bars (threshold=${dollar_per_bar:,.0f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
