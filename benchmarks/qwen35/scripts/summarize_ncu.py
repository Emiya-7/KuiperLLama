#!/usr/bin/env python3
"""Print a compact Markdown summary from qwen35 NCU raw CSV exports."""

import argparse
import csv
from pathlib import Path


def read_single_row(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        headers = next(rows)
        units = dict(zip(headers, next(rows)))
        values = dict(zip(headers, next(rows)))
    return units, values


def number(values, key):
    value = values.get(key, "")
    if not value:
        raise ValueError(f"missing NCU metric {key}")
    return float(value)


def duration_us(units, values):
    key = "gpu__time_duration.sum"
    value = number(values, key)
    unit = units.get(key)
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "ns":
        return value / 1000.0
    raise ValueError(f"unsupported duration unit {unit!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.report_directory.glob("*.raw.csv"))
    if not paths:
        parser.error(f"no *.raw.csv files in {args.report_directory}")

    print("| Case | Duration (us) | DRAM (%) | DRAM (GB/s) | SM (%) | L2 hit (%) | "
          "Occupancy (%) | Registers | Waves/SM | Eligible (%) | Long scoreboard (%) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path in paths:
        units, values = read_single_row(path)
        warp_latency = number(values, "smsp__average_warp_latency_per_inst_issued.ratio")
        long_scoreboard = number(
            values, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
        )
        fields = (
            path.name.removesuffix(".raw.csv"),
            duration_us(units, values),
            number(values, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
            number(values, "dram__bytes.sum.per_second"),
            number(values, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            number(values, "lts__t_sector_hit_rate.pct"),
            number(values, "sm__warps_active.avg.pct_of_peak_sustained_active"),
            number(values, "launch__registers_per_thread"),
            number(values, "launch__waves_per_multiprocessor"),
            number(values, "smsp__issue_active.avg.pct_of_peak_sustained_active"),
            100.0 * long_scoreboard / warp_latency,
        )
        print(
            f"| {fields[0]} | {fields[1]:.3f} | {fields[2]:.2f} | {fields[3]:.2f} | "
            f"{fields[4]:.2f} | {fields[5]:.2f} | {fields[6]:.2f} | {fields[7]:.0f} | "
            f"{fields[8]:.2f} | {fields[9]:.2f} | {fields[10]:.2f} |"
        )


if __name__ == "__main__":
    main()
