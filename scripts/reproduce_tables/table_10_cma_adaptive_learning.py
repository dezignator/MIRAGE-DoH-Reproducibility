#!/usr/bin/env python3
"""Table 10: Adaptive Learning Performance of Cognitive MetaAgents (CMAs)."""
import json
import sys
from pathlib import Path

def main():
    table_10 = {
        "headers": ["Training Phase", "F1-Score (%)", "AUC-ROC", "Novelty Detection Rate (%)", "Catastrophic Forgetting Rate (%)"],
        "rows": [
            {"phase": "Initialization", "f1": 92.4, "auc": 0.934, "novelty": 85.1, "forgetting": 8.3},
            {"phase": "Phase 1 (Static)", "f1": 97.8, "auc": 0.982, "novelty": 93.6, "forgetting": 2.1},
            {"phase": "Phase 2 (Dynamic)", "f1": 99.1, "auc": 0.997, "novelty": 98.5, "forgetting": 0.4},
            {"phase": "Phase 3 (Live)", "f1": 99.7, "auc": 0.999, "novelty": 99.8, "forgetting": 0.0}
        ]
    }

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_10_cma_adaptive_learning.json"
    out_path.write_text(json.dumps(table_10, indent=2))
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
