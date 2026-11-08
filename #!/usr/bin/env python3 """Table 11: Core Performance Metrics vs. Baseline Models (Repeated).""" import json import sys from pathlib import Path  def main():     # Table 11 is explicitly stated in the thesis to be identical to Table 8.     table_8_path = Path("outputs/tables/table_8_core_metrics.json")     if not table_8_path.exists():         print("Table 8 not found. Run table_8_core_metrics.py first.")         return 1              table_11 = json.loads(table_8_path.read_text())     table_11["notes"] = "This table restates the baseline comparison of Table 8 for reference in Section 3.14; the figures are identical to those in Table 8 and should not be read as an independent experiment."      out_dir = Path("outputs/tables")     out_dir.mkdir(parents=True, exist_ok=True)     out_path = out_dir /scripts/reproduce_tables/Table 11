#!/usr/bin/env python3
"""Table 11: Core Performance Metrics vs. Baseline Models (Repeated)."""
import json
import sys
from pathlib import Path

def main():
    # Table 11 is explicitly stated in the thesis to be identical to Table 8.
    table_8_path = Path("outputs/tables/table_8_core_metrics.json")
    if not table_8_path.exists():
        print("Table 8 not found. Run table_8_core_metrics.py first.")
        return 1
        
    table_11 = json.loads(table_8_path.read_text())
    table_11["notes"] = "This table restates the baseline comparison of Table 8 for reference in Section 3.14; the figures are identical to those in Table 8 and should not be read as an independent experiment."

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_11_core_metrics_repeated.json"
    out_path.write_text(json.dumps(table_11, indent=2))
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
