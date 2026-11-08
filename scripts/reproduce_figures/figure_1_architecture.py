#!/usr/bin/env python3
"""Figure 1: MIRAGE-DoH Metamorphic Defense Grid Architecture."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  edgecolor="black", facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, fontweight="bold", wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha="center", va="bottom", fontsize=8, color="gray")

def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    
    # Input Layer
    draw_box(ax, 0.5, 6.5, 2, 1, "Input:\nEncrypted DoH\nTraffic", "#E0F7FA")
    
    # Cognitive Defense Stack
    draw_box(ax, 3.5, 6.5, 3, 1, "Cognitive MetaAgents\n(CMA)\nNeuroevolution", "#FFF9C4")
    draw_arrow(ax, 2.5, 7.0, 3.5, 7.0)
    
    # Topological Memory
    draw_box(ax, 3.5, 4.5, 3, 1, "Topological Memory\nNetworks (TMN)\nPersistent Homology", "#E8F5E9")
    draw_arrow(ax, 5.0, 6.5, 5.0, 5.5)
    
    # Crypto Shards
    draw_box(ax, 7.5, 6.5, 2, 1, "Metamorphic\nCryptographic Shards\n(MCS)", "#FFEBEE")
    draw_arrow(ax, 6.5, 7.0, 7.5, 7.0)
    
    # Causal Counterfactual
    draw_box(ax, 3.5, 2.5, 3, 1, "Causal Counterfactual\nEnvironments (CCE)\nProactive Hardening", "#F3E5F5")
    draw_arrow(ax, 5.0, 4.5, 5.0, 3.5)
    
    # Spectral Game Intelligence
    draw_box(ax, 7.5, 4.5, 2, 1, "Spectral Game\nIntelligence (SGI)\nGraph Laplacian", "#E3F2FD")
    draw_arrow(ax, 6.5, 5.0, 7.5, 5.0)
    
    # Orchestration / Output
    draw_box(ax, 3.5, 0.5, 6, 1, "Distributed Orchestration & Autonomous Defense Governance\n(22 ms Adaptation Window)", "#CFD8DC")
    draw_arrow(ax, 5.0, 2.5, 5.0, 1.5)
    draw_arrow(ax, 8.5, 4.5, 8.5, 1.5)
    
    plt.title("Figure 1: MIRAGE-DoH Metamorphic Defense Grid Architecture", fontsize=14, pad=20)
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "figure_1_architecture.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "figure_1_architecture.png", dpi=300, bbox_inches="tight")
    print(f"Wrote {out_dir}/figure_1_architecture.*")
    return 0

if __name__ == "__main__":
    sys.exit(main())
