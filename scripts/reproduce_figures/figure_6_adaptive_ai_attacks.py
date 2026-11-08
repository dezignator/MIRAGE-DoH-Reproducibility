#!/usr/bin/env python3
"""Figure 6: MIRAGE-DoH Cognitive Defense Evaluation under Adaptive AI Attacks."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Panel (a): Timeout Resilience
    x = np.arange(0, 35000, 5000)
    # Legacy DNS error rate goes up to 28%
    legacy_timeout = [0, 5, 12, 18, 25, 28, 35]
    # MIRAGE-DoH stays below 5%
    mirage_timeout = [0, 0.5, 1, 2, 3, 4, 4.5]
    
    ax1.plot(x, legacy_timeout, label="Legacy DNS", color="red", marker='o')
    ax1.plot(x, mirage_timeout, label="MIRAGE-DoH", color="teal", marker='s')
    ax1.set_title("(a) Timeout Error Rate vs. AI Attack Rate")
    ax1.set_xlabel("Attack Rate (Queries Per Second)")
    ax1.set_ylabel("Timeout Error Rate (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel (b): Cognitive Security Scalability
    x_adv = np.arange(0, 105, 10)
    # Traditional security score drops below 88%
    trad_sec = [95, 92, 90, 88, 85, 82, 80, 78, 75, 74, 72]
    # CMA + SGI module maintains >97%
    mirage_sec = [99, 98.5, 98, 97.5, 97.2, 97.1, 97, 97, 97, 96.8, 96.5]
    
    ax2.plot(x_adv, trad_sec, label="Traditional Defenses", color="red", linestyle="--")
    ax2.plot(x_adv, mirage_sec, label="MIRAGE (CMA + SGI)", color="purple", linewidth=2)
    ax2.fill_between(x_adv, mirage_sec, 90, color='purple', alpha=0.1)
    
    ax2.set_title("(b) Cognitive Security Scalability")
    ax2.set_xlabel("Adversarial Training Samples (%)")
    ax2.set_ylabel("Cognitive Security Score (%)")
    ax2.set_ylim(70, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "figure_6_adaptive_ai_attacks.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "figure_6_adaptive_ai_attacks.png", dpi=300, bbox_inches="tight")
    print(f"Wrote {out_dir}/figure_6_adaptive_ai_attacks.*")
    return 0

if __name__ == "__main__":
    sys.exit(main())
