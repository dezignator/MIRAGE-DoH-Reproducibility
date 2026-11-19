#!/usr/bin/env python3
"""MIRAGE-DoH Chapter 3, system-level attack scenarios and consolidated report (v2).

  S1  scenario catalogue      the five attack types of Tables 6-7 as simulated here
  S2  end-to-end pipeline     detector ensemble -> threat score rho -> shard policy (Eq. 13) ->
                              causal hardening (Eq. 16): time to escalate, false escalations,
                              morph events and the bytes they cost, per attack
  S3  resilience under load   timeout rate of a fixed-capacity resolver vs the same resolver with
                              detection, filtering and capacity reconfiguration (Fig. 6 style);
                              a queueing MODEL whose capacity/delay parameters are stated, not measured
  S4  evasion strength sweep  detection rate as attackers blend timing, size and naming toward
                              benign (Fig. 5a style), ensemble vs Random Forest vs MLP
  S5  hyper-parameter spread  20 random configurations of CMA, Random Forest and MLP (Fig. 5c style)
  S6  arithmetic audit        what the Chapter 3 tables imply when combined (ranges, ratios,
                              instance counts, sample sizes)
  S7  consolidated report     one Markdown + self-contained HTML report, with a claim-by-claim table

The scorers and data come from mirage_detection_experiments_v2.py (imported), the crypto
and causal tables from mirage_crypto_causal_experiments_v2.py (read from its CSV output).

Everything is simulated or measured on the machine running it: traffic is synthetic,
baselines are scikit-learn models, cryptography is CPU-only, and hardware-bound results
(Loihi 2, FPGA, A100) are listed as not reproducible. The tables are not Tables 8-12.

usage:
    python3 experiments/mirage_system_report_v2.py --detect build/detect --crypto build/crypto --out build/system
    python3 experiments/mirage_system_report_v2.py --quick --detect ... --crypto ... --out ...
"""
import argparse
import base64
import html
import importlib.util
import json
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirage_v2 import traffic_v2 as tv, cma_v2 as cma, tmn_v2 as tmn, mcs_v2 as mcs, cce_v2 as cce   # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


det = load("detx", "experiments/mirage_detection_experiments_v2.py")


# ==========================================================================
def s1_catalogue():
    rows = [
        ("AI-powered adversarial flood", "gan_flood", "generated pseudo-words, unique names, 10-50 ms spacing", "Table 6 row 1", "CMA, SGI"),
        ("Quantum-assisted timing probe", "timing_probe", "regular, rapid handshakes with near-constant timing", "Table 6 row 2", "CMA, MCS"),
        ("Protocol downgrade", "downgrade", "TLS 1.2 fallback, ECS manipulation, varying extensions", "Table 6 row 3", "CMA, CCE, MCS"),
        ("Multi-phase tunnelling", "tunnel", "slow, long high-entropy labels, TXT-heavy", "Table 6 row 4", "TMN"),
        ("Structural / spectral DGA", "spectral_dga", "dictionary names, slow cyclic rate modulation", "Table 6 row 5", "SGI"),
    ]
    return pd.DataFrame(rows, columns=["attack (thesis name)", "simulated family", "behaviour", "source", "module expected to carry it"])


# ==========================================================================
# S2 pipeline
# ==========================================================================
class Pipeline:
    def __init__(self, data, cfg, seed, fam):
        self.std, self.cm, self.tm, self.sg = det.fit_all(data, cfg, seed, fam)
        cv = det.comp_scores(data["val"], self.cm, self.tm, self.sg)
        cc = det.comp_scores(data["cal"], self.cm, self.tm, self.sg)
        m = np.isin(data["val"]["k"], ["benign"] + fam)
        Xv = np.column_stack([cv[c][m] for c in ("CMA", "TMN", "SGI")])
        self.clf = tmn.Logistic().fit(Xv, data["val"]["y"][m].astype(float))
        s_cal = self.clf.score(np.column_stack([cc[c] for c in ("CMA", "TMN", "SGI")]))
        self.t99 = float(np.quantile(s_cal, 0.99)); self.med = float(np.median(s_cal))
        self.scale = max((self.t99 - self.med) / 4.0, 1e-6)
        self.s_cal = s_cal

    def ensemble(self, packs):
        F = np.array([p[0] for p in packs]); S = np.array([p[3] for p in packs]); dg = np.empty(len(packs), dtype=object)
        for i, p in enumerate(packs):
            dg[i] = p[2]
        sc = det.comp_scores({"F": F, "dg": dg, "S": S}, self.cm, self.tm, self.sg)
        return self.clf.score(np.column_stack([sc[c] for c in ("CMA", "TMN", "SGI")]))

    def rho(self, s):
        """Threat score in (0,1): logistic of the ensemble score relative to the benign tail
        (benign median -> ~0.02, benign 99th percentile -> 0.5). Mapping is my choice."""
        return 1.0 / (1.0 + np.exp(-(np.asarray(s) - self.t99) / self.scale))


def s2_pipeline(pipe, cfg, rng):
    rows, examples = [], {}
    cce_map = {"gan_flood": "AI mimicry (rule-evasion)", "downgrade": "protocol downgrade (TLS1.3->1.2 via ECS)",
               "timing_probe": "timing probe on key setup"}
    n_pre, n_post = cfg["stream"]["pre"], cfg["stream"]["post"]
    for kind in ["gan_flood", "timing_probe", "downgrade", "tunnel", "spectral_dga"]:
        d1, d2, fp, nmorph, bytes_pre, bytes_nopre, lvl_post = [], [], [], [], [], [], []
        for run in range(cfg["stream"]["runs"]):
            wins = [tv.gen_window("benign", rng) for _ in range(n_pre)] + [tv.gen_window(kind, rng) for _ in range(n_post)]
            packs = [det.window_pack(w) for w in wins]
            s = pipe.ensemble(packs); r = pipe.rho(s)
            th = mcs.AdaptiveThresholds()
            levels = []
            for x in r:
                g1, g2 = th.update(x)
                levels.append(mcs.select_level(x, g1, g2))
            levels = np.array(levels)
            post = levels[n_pre:]
            a1 = np.nonzero(post >= 1)[0]; a2 = np.nonzero(post == 2)[0]
            d1.append(a1[0] if len(a1) else np.nan); d2.append(a2[0] if len(a2) else np.nan)
            fp.append(float(np.mean(levels[:n_pre] >= 1)))
            sw = np.nonzero(levels[1:] != levels[:-1])[0] + 1
            nmorph.append(len(sw))
            def cost(pre):
                return sum(sum(mcs.wire_bytes(int(levels[i]), pre).values()) for i in sw)
            bytes_nopre.append(cost(False)); bytes_pre.append(cost(True))
            lvl_post.append(float(post.mean()))
            if run == 0:
                examples[kind] = (r.copy(), levels.copy())
        row = {"attack": kind, "windows to Pi_1+ after onset": "-" if np.all(np.isnan(d1)) else f"{np.nanmean(d1):.1f}",
               "windows to Pi_2 after onset": "-" if np.all(np.isnan(d2)) else f"{np.nanmean(d2):.1f}",
               "runs reaching Pi_1+": f"{int(np.sum(~np.isnan(d1)))}/{len(d1)}", "runs reaching Pi_2": f"{int(np.sum(~np.isnan(d2)))}/{len(d2)}",
               "false escalations before onset": f"{100*np.mean(fp):.1f}%", "mean level after onset": f"{np.mean(lvl_post):.2f}",
               "morph events/run": f"{np.mean(nmorph):.1f}", "morph bytes/run (McEliece pk sent)": f"{np.mean(bytes_nopre):,.0f}",
               "morph bytes/run (pk pre-provisioned)": f"{np.mean(bytes_pre):,.0f}"}
        if kind in cce_map:
            acts, best, _ = cce.choose_hardening(cce.SCENARIOS[cce_map[kind]])
            base = [a for a in acts if a[0] == "none"][0][1]
            row["causal hardening (CCE)"] = f"{best[0]} (risk {100*base:.1f}% -> {100*best[1]:.1f}%)"
        else:
            row["causal hardening (CCE)"] = "no CCE scenario defined"
        rows.append(row)
    return pd.DataFrame(rows), examples


# ==========================================================================
# S3 resilience model
# ==========================================================================
def simulate_queue(lam_benign, lam_attack, mu, T_end, timeout, rng, attack_start=2.0, filt=None, scale_up=None):
    """Single-server FIFO queue, exponential service (rate mu per capacity unit). Queries whose
    waiting time would exceed `timeout` are dropped and counted as timeouts.
    filt = (delay_s, p_block_attack, p_block_benign): after `delay_s` of attack, attack queries are
    blocked at admission with p_block_attack, benign with p_block_benign.
    scale_up = (delay_s, factor): service rate multiplied by factor after the delay."""
    nb = int(rng.poisson(lam_benign * T_end))
    na = int(rng.poisson(lam_attack * max(T_end - attack_start, 0.0))) if lam_attack > 0 else 0
    t = np.concatenate([rng.uniform(0, T_end, nb), rng.uniform(attack_start, T_end, na)])
    is_att = np.concatenate([np.zeros(nb, bool), np.ones(na, bool)])
    o = np.argsort(t, kind="stable"); t, is_att = t[o], is_att[o]
    n = len(t)
    S = rng.exponential(1.0, n)
    w, last_t, busy_until = 0.0, 0.0, 0.0
    served = timeouts = benign_n = benign_to = benign_blocked = 0
    for i in range(n):
        ti = t[i]
        since = ti - attack_start
        rate = mu
        if scale_up and since >= scale_up[0] and lam_attack > 0:
            rate = mu * scale_up[1]
        if filt and since >= filt[0] and lam_attack > 0:
            if is_att[i] and rng.random() < filt[1]:
                continue
            if (not is_att[i]) and rng.random() < filt[2]:
                benign_n += 1; benign_blocked += 1; continue
        if not is_att[i]:
            benign_n += 1
        wait = max(0.0, busy_until - ti)
        if wait > timeout:
            timeouts += 1
            if not is_att[i]:
                benign_to += 1
            continue
        busy_until = ti + wait + S[i] / rate
        served += 1
    return timeouts, n, benign_to, benign_n, benign_blocked


def s3_resilience(cfg, rng, tpr, fpr, det_delay_s):
    mu, timeout, T_end = cfg["queue"]["mu"], 0.5, cfg["queue"]["T"]
    rows = []
    for lam_a in cfg["queue"]["attack_rates"]:
        res = {}
        for name, kw in [("fixed resolver", {}),
                         ("MIRAGE model: detect+filter", dict(filt=(det_delay_s, tpr, fpr))),
                         ("MIRAGE model: detect+filter+2x capacity", dict(filt=(det_delay_s, tpr, fpr), scale_up=(det_delay_s + 0.5, 2.0)))]:
            vals = []
            for rep in range(cfg["queue"]["reps"]):
                to, n, bto, bn, bb = simulate_queue(cfg["queue"]["benign"], lam_a, mu, T_end, timeout, rng, **kw)
                vals.append((bto + bb) / max(bn, 1))
            res[name] = np.mean(vals)
        rows.append({"attack rate (qps)": lam_a, "benign rate (qps)": cfg["queue"]["benign"],
                     "benign queries lost: fixed resolver": f"{100*res['fixed resolver']:.1f}%",
                     "benign queries lost: detect+filter": f"{100*res['MIRAGE model: detect+filter']:.1f}%",
                     "benign queries lost: detect+filter+2x capacity": f"{100*res['MIRAGE model: detect+filter+2x capacity']:.1f}%"})
    return pd.DataFrame(rows)


# ==========================================================================
# S4 evasion
# ==========================================================================
def evade(win, s, rng, benign_pool):
    """Blend an attack window toward benign: with probability s each inter-arrival, size and name is
    replaced by a draw from a benign window. s=0 leaves the attack untouched."""
    b = benign_pool[int(rng.integers(len(benign_pool)))]
    n = len(win["ts"])
    iat = np.diff(win["ts"], prepend=win["ts"][0] - 1.0)
    biat = np.diff(b["ts"], prepend=b["ts"][0] - 1.0)
    pick = rng.random(n) < s
    iat2 = np.where(pick, biat[rng.integers(0, n, n)], iat)
    out = dict(win)
    out["ts"] = np.cumsum(iat2)
    out["size"] = np.where(rng.random(n) < s, b["size"][rng.integers(0, n, n)], win["size"])
    nm = list(win["names"]); bn = b["names"]
    for i in range(n):
        if rng.random() < s:
            nm[i] = bn[int(rng.integers(n))]
    out["names"] = nm
    for k in ("tls12", "hs", "ecs"):
        out[k] = win[k] * (1 - s) + b[k] * s
    return out


def s4_evasion(pipe, data, cfg, rng, seed, fam):
    rf, mlp = det.baselines(data, pipe.std, fam, seed)
    cal_rf = rf.predict_proba(pipe.std(data["cal"]["F"]))[:, 1]
    cal_ml = mlp.predict_proba(pipe.std(data["cal"]["F"]))[:, 1]
    t_rf, t_ml, t_en = det.thr(cal_rf), det.thr(cal_ml), det.thr(pipe.s_cal)
    pool = [tv.gen_window("benign", rng) for _ in range(40)]
    rows = []
    for s in cfg["evade"]["strengths"]:
        tp = {"Random Forest": [], "MLP": [], "MIRAGE ensemble": []}
        for kind in tv.ATTACKS:
            wins = [evade(tv.gen_window(kind, rng), s, rng, pool) for _ in range(cfg["evade"]["n"])]
            packs = [det.window_pack(w) for w in wins]
            F = np.array([p[0] for p in packs])
            tp["Random Forest"].append((rf.predict_proba(pipe.std(F))[:, 1] > t_rf).mean())
            tp["MLP"].append((mlp.predict_proba(pipe.std(F))[:, 1] > t_ml).mean())
            tp["MIRAGE ensemble"].append((pipe.ensemble(packs) > t_en).mean())
        rows.append({"evasion strength s": s, **{f"{k} TPR": f"{100*np.mean(v):.1f}%" for k, v in tp.items()}})
    return pd.DataFrame(rows)


# ==========================================================================
# S5 hyper-parameter spread
# ==========================================================================
def s5_spread(data, pipe, cfg, rng, fam):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    tr, te = data["train"], data["test"]
    m = np.isin(tr["k"], ["benign"] + fam); mt_ = np.isin(te["k"], ["benign"] + fam)
    X, y = pipe.std(tr["F"][m]), tr["y"][m]; Xt, yt = pipe.std(te["F"][mt_]), te["y"][mt_]
    def f1(pred):
        tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
        return 2 * tp / max(2 * tp + fp + fn, 1)
    out = {"CMA (ES)": [], "Random Forest": [], "MLP": []}
    nov = np.zeros(len(X))
    for _ in range(cfg["spread"]["configs"]):
        th = cma.es_train(cma.init(rng), X, y.astype(float), nov, rng, gens=cfg["spread"]["gens"], pop=int(rng.choice([16, 32, 48, 64])),
                          sigma=float(rng.uniform(0.05, 0.3)), lr=float(rng.uniform(0.01, 0.08)))
        out["CMA (ES)"].append(f1((cma.prob(th, Xt) >= 0.5).astype(int)))
        rf = RandomForestClassifier(int(rng.choice([10, 30, 100, 300])), max_depth=rng.choice([2, 3, 5, 8, None]),
                                    min_samples_leaf=int(rng.choice([1, 5, 20])), random_state=int(rng.integers(1e6))).fit(X, y)
        out["Random Forest"].append(f1(rf.predict(Xt)))
        ml = MLPClassifier((int(rng.choice([4, 16, 64])),), alpha=float(10 ** rng.uniform(-5, -1)), max_iter=200,
                           random_state=int(rng.integers(1e6))).fit(X, y)
        out["MLP"].append(f1(ml.predict(Xt)))
    pd.DataFrame(out).to_csv(Path(cfg.get("raw_dir", ".")) / "s5_raw_f1.csv", index=False)
    rows = [{"model": k, "configs": len(v), "F1 mean": f"{100*np.mean(v):.2f}%", "sd": f"{100*np.std(v, ddof=1):.2f}",
             "min": f"{100*np.min(v):.2f}%", "max": f"{100*np.max(v):.2f}%"} for k, v in out.items()]
    return pd.DataFrame(rows)


# ==========================================================================
# S6 arithmetic audit
# ==========================================================================
def s6_audit():
    R = []
    def add(item, stated, implied, verdict):
        R.append({"item": item, "stated in Chapter 3": stated, "implied / recomputed": implied, "reading": verdict})
    add("Table 8 latency improvement", "62.8 % vs CNN-LSTM", f"{(210-78)/210*100:.1f} %", "consistent")
    add("Table 8 throughput improvement", "75.9 % vs CNN-LSTM", f"{(15.3/8.7-1)*100:.1f} %", "consistent")
    add("Table 8 FPR reduction vs CNN-LSTM", "99.5 %", f"{(1-0.0008/0.15)*100:.2f} %", "consistent")
    add("Table 8 accuracy gain vs CNN-LSTM", "4.48 pp", f"{99.98-95.5:.2f} pp", "consistent")
    add("Table 9 latency speed-up", "6.9x (135-160 us vs 18-28 us)", f"{135/28:.1f}x to {160/18:.1f}x (midpoints {147.5/23:.1f}x)", "6.9x lies inside the range, not derivable from the endpoints")
    add("Table 12 detection-time factor", "5.4x (120-150 ms vs 22-35 ms)", f"{120/35:.1f}x to {150/22:.1f}x (midpoints {135/28.5:.1f}x)", "5.4x lies inside the range")
    add("Table 12 FP-rate factor", "25.7x (1.8 % vs 0.07 %)", f"{1.8/0.07:.1f}x", "consistent")
    add("Table 12 topological recall factor", "1.37x", f"{99.1/72.3:.2f}x", "consistent")
    add("Table 12 spectral accuracy factor", "1.51x", f"{98.9/65.4:.2f}x", "consistent")
    add("Morph time range", "3.8-9.5 ms (Tables 9, 12; Sec 3.15.1), every value < 10 ms",
        "text after Table 12: 'took 4.2-12.1 ms to reconfigure'", "contradiction: 12.1 ms exceeds the 10 ms target and the stated range")
    add("Topological recall of TMN", "99.1 % (Table 12), 99.4 % (Table 9), 99.93 % (Sec 3.15)", "three values for one quantity",
        "state which evaluation each refers to")
    add("False-positive rate", "0.0008 % (Table 8), 0.05 % (Table 9), 0.07 % (Table 12)", "different baselines per table (as the notes say)",
        "fine if kept apart; 0.0008 % needs a very large benign sample")
    from mirage_v2.metrics_v2 import min_negatives_for_fpr
    add("Sample size for FPR 0.0008 %", "CI +-0.0001 pp, no sample counts (Sec 3.16)",
        f"certifying 0 FP at 8e-6 needs >= {min_negatives_for_fpr(8e-6):,} benign windows (95 %)", "report the benign count")
    inst = 15.3e6 * 78e-6
    add("Throughput vs latency (Eq. 31)", "15.3 M qps aggregate, 78 us per query", f"{inst:,.0f} concurrent instances needed (15.3e6 x 78e-6)",
        "check against the < 5 % CPU, 16 GB per instance bound of Sec 3.18")
    add("Eq. 31 dimension", "T_total = (sum_i lambda_i)^-1 * C_parallel", "summing per-class times is not a rate",
        "use C_parallel / mean per-query time")
    add("Detection time vs scoring latency", "22-35 ms end-to-end (Table 12) vs 78 us scoring (Table 8)", "ratio ~ 280-450x", "documented in the Table 12 note")
    add("McEliece public key in a 10 ms morph", "Pi_2 morph 3.8-9.5 ms", f"mceliece6688128 pk = {mcs.MCELIECE_SIZES['pk']:,} B -> {mcs.MCELIECE_SIZES['pk']*8/1e9/0.01:.2f} Gbit/s needed to move it in 10 ms",
        "feasible only if the key is pre-provisioned or the link is >= ~1 Gbit/s with tiny RTT")
    add("Figure 4 plateaus", "all modules start at 0.26 and reach 0.82", "identical start and end values for four curves", "confirm these are separate measurements")
    add("Hyper-parameter sensitivity", "20 configurations, 88-89 % vs 83-89 %", "a 1-point spread over 20 configs", "report sd and the sampled ranges")
    return pd.DataFrame(R)


# ==========================================================================
# plots
# ==========================================================================
def plots(out, examples, s3, s4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 3.2), sharey=True)
    for ax, (k, (r, lv)) in zip(np.atleast_1d(axs), examples.items()):
        ax.plot(r, lw=1, label="threat score rho"); ax.step(np.arange(len(lv)), lv / 2.0, "r", lw=1, label="level/2")
        n_pre = np.argmax(np.arange(len(r)) >= len(r) // 4) if False else None
        ax.set_title(k, fontsize=8); ax.set_xlabel("window")
    np.atleast_1d(axs)[0].legend(fontsize=6); np.atleast_1d(axs)[0].set_ylabel("rho / policy level")
    fig.tight_layout(); fig.savefig(out / "s2_pipeline.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c in [c for c in s3.columns if c.startswith("benign queries lost")]:
        ax.plot(s3["attack rate (qps)"], s3[c].str.rstrip("%").astype(float), "o-", label=c.split(": ")[1])
    ax.set_xlabel("attack rate (qps)"); ax.set_ylabel("benign queries lost (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("S3 resilience model")
    fig.tight_layout(); fig.savefig(out / "s3_resilience.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c in [c for c in s4.columns if c.endswith("TPR")]:
        ax.plot(s4["evasion strength s"], s4[c].str.rstrip("%").astype(float), "o-", label=c.replace(" TPR", ""))
    ax.set_xlabel("evasion strength s"); ax.set_ylabel("TPR at 1% FPR (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("S4 evasion sweep")
    fig.tight_layout(); fig.savefig(out / "s4_evasion.png", dpi=140); plt.close(fig)


# ==========================================================================
# report
# ==========================================================================
CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#1b1f24;line-height:1.45}
h1{font-size:1.7rem}h2{margin-top:2.2rem;border-bottom:1px solid #d0d7de;padding-bottom:.2rem}h3{margin-top:1.4rem}
table{border-collapse:collapse;font-size:.78rem;margin:.6rem 0 1.2rem;width:100%}th,td{border:1px solid #d0d7de;padding:.25rem .45rem;text-align:left;vertical-align:top}
th{background:#f3f5f7}tr:nth-child(even) td{background:#fafbfc}.note{background:#fff8e6;border-left:4px solid #d4a72c;padding:.5rem .8rem;margin:1rem 0}
img{max-width:100%;border:1px solid #d0d7de;margin:.4rem 0}"""


class Report:
    def __init__(self):
        self.md, self.html = [], []

    def h(self, lvl, t):
        self.md.append(f"\n{'#'*lvl} {t}\n"); self.html.append(f"<h{lvl}>{html.escape(t)}</h{lvl}>")

    def p(self, t, note=False):
        self.md.append(t + "\n"); self.html.append(f"<div class='note'>{html.escape(t)}</div>" if note else f"<p>{html.escape(t)}</p>")

    def table(self, df, title=None):
        if title:
            self.h(3, title)
        self.md.append(df.to_markdown(index=False) + "\n"); self.html.append(df.to_html(index=False, escape=True, border=0))

    def image(self, path, alt):
        path = Path(path)
        if path.exists():
            self.md.append(f"![{alt}]({path.name})\n")
            self.html.append(f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'>")

    def write(self, out, stem):
        (out / f"{stem}.md").write_text("\n".join(self.md))
        (out / f"{stem}.html").write_text(f"<!doctype html><meta charset='utf-8'><title>{stem}</title><style>{CSS}</style>\n" + "\n".join(self.html))


def read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None


def claims(dd, cd, sd):
    rows = []
    def add(item, thesis, here, note):
        rows.append({"Chapter 3 item": item, "thesis value": thesis, "this simulation": here, "reading": note})
    try:
        a = read(dd / "d1_seen_protocol.csv"); e = a[a.detector.str.startswith("MIRAGE")].iloc[0]; r = a[a.detector.str.startswith("Random")].iloc[0]
        add("Detection accuracy vs baselines (Table 8)", "99.98 % vs 95.5 % (CNN-LSTM), 92.3 % (RF)",
            f"seen-family protocol: ensemble {e['accuracy']}, RF {r['accuracy']}", "synthetic task is easy for tree models; ordering is not the point")
        add("Zero-day families", "novelty detection 99.8 % (Table 10)", f"unseen: timing_probe {e['*timing_probe']}, spectral_dga {e['*spectral_dga']} (ensemble); RF {r['*timing_probe']} / {r['*spectral_dga']}",
            "components matter on unseen structure; see D5")
        c = read(dd / "d1_throughput.csv")
        add("Throughput 15.3 M qps (Table 8)", "15.3 M qps aggregate", f"{c.iloc[1]['value']} qps per core here -> {c.iloc[2]['value']} instances for 15.3 M qps", "persistence + spectral extraction dominates in Python")
        d5 = read(dd / "d5_ablation.csv")
        full = d5[d5.components == "CMA+TMN+SGI"].iloc[0]
        add("Ablation (not performed in thesis, Sec 7.3.1)", "none reported", "unseen-family TPR by subset: " + "; ".join(f"{r['components']} {r['TPR unseen (zero-day)']}" for _, r in d5.iterrows()), "full factorial in D5")
        d2 = read(dd / "d2_curriculum.csv")
        p3 = d2[d2.phase.str.startswith("Phase 3")]
        add("Catastrophic forgetting (Table 10)", "8.3 % -> 0 %", "Phase 3 forgetting by lambda: " + "; ".join(f"{r['lambda (Fisher)']}: {r['forgetting (%)']} %" for _, r in p3.iterrows()), "no forgetting appears at lambda=0 in this setup, so a benefit of the penalty cannot be shown here")
        d3 = read(dd / "d3_tmn_hops.csv")
        g = d3[(d3["hops L"] == d3["hops L"].max())]
        add("Multi-hop (50+ hops, Table 9/Sec 3.15)", "detected over 50+ hops vs 35 for baselines", "longest loop: " + "; ".join(f"p={r['P(edge observed)']},T={r['memory windows T']}: {100*r['detection']:.0f} %" for _, r in g.iterrows()), "depends on edge observation and memory depth")
        d4 = read(dd / "d4_sgi_dqn.csv")
        t = d4.iloc[0]; f = d4.iloc[1]
        add("Spectral accuracy (Table 12: 98.9 % vs 65.4 %)", "98.9 % vs 65.4 %", f"spectral_dga: spectral state {t['accuracy']} %, flow state {f['accuracy']} %", "not reproduced: the flow metadata here already contains a rate-spectrum feature (lowband_power)")
    except Exception as ex:                                     # noqa: BLE001
        add("detection suite", "-", f"unavailable ({ex})", "run it first")
    try:
        c3 = read(cd / "c3_morph_budget.csv"); n = c3[c3.level.str.startswith("Pi_2") & ~c3.level.str.contains("pre")]
        add("Pi_2 morph 3.8-9.5 ms (Table 9)", "3.8-9.5 ms", "McEliece pk transfer alone: " + "; ".join(f"{r['link']}: {r['transfer+RTT ms']} ms" for _, r in n.iterrows()), "needs pre-provisioned key or >= ~1 Gbit/s")
        c1 = read(cd / "c1_mlkem.csv")
        add("ML-KEM-768/1024 sizes and correctness", "Table 5 uses FIPS 203 sets", "; ".join(f"{r['parameter set']} sizes match: {r['matches FIPS 203 sizes']}, round trips {r.iloc[3]}" for _, r in c1.iterrows()), "reference implementation, not KAT validated")
        c5 = read(cd / "c5_causal.csv")
        add("CCE structural causal inference (Sec 3.8)", "5.4x lower detection time with CMA+SGI; CCE reduces false adaptation", "; ".join(f"{r['scenario'].split(' (')[0]}: naive {r['P(Y|X=1) observational']} vs do {r['P(Y|do(X=1)) Eq.15']}" for _, r in c5.iterrows()), "confounding bias is large in these assumed SCMs")
    except Exception as ex:                                     # noqa: BLE001
        add("crypto suite", "-", f"unavailable ({ex})", "run it first")
    add("Loihi 2 / FPGA latency and energy", "78 us, 8.9 ms morph, sub-2 ms crypto latency", "not reproducible without the hardware", "CPU numbers here are baselines only")
    add("Quantum-adversary simulations (Sec 3.18)", "limited information gain", "not reproducible: no quantum simulator in this artifact", "the thesis itself says no unconditional PQ security is claimed")
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", default="build/detect")
    ap.add_argument("--crypto", default="build/crypto")
    ap.add_argument("--out", default="build/system")
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of s2,s3,s4,s5,final (default: all)")
    a = ap.parse_args()
    dd, cd, out = Path(a.detect), Path(a.crypto), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(stream=dict(pre=60, post=60, runs=6), queue=dict(mu=22000.0, benign=5000.0, T=5.0, attack_rates=[0, 5000, 10000, 20000, 30000], reps=2),
               evade=dict(strengths=[0.0, 0.25, 0.5, 0.75, 1.0], n=60), spread=dict(configs=20, gens=120), gens=400)
    if a.quick:
        cfg = dict(stream=dict(pre=20, post=20, runs=2), queue=dict(mu=22000.0, benign=5000.0, T=1.5, attack_rates=[0, 20000], reps=1),
                   evade=dict(strengths=[0.0, 1.0], n=15), spread=dict(configs=3, gens=40), gens=100)
    pkl = dd / ("data_quick.pkl" if a.quick else "data.pkl")
    if not pkl.exists():
        raise SystemExit(f"{pkl} not found: run mirage_detection_experiments_v2.py first (it caches the dataset)")
    data = pickle.load(open(pkl, "rb"))
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    todo = set(a.only.split(",")) if a.only else {"s2", "s3", "s4", "s5", "final"}
    print("fitting pipeline ...")
    pipe = Pipeline(data, cfg, a.seed, det.SEEN)
    s1 = s1_catalogue(); s1.to_csv(out / "s1_catalogue.csv", index=False)
    if "s2" in todo:
        print("S2 ..."); s2, ex = s2_pipeline(pipe, cfg, np.random.default_rng(a.seed + 2)); s2.to_csv(out / "s2_pipeline.csv", index=False)
        pickle.dump(ex, open(out / "s2_examples.pkl", "wb")); print(s2.to_string(index=False)); print()
    te = data["test"]; ben = te["k"] == "benign"
    sc = det.comp_scores(te, pipe.cm, pipe.tm, pipe.sg)
    s_te = pipe.clf.score(np.column_stack([sc[c] for c in ("CMA", "TMN", "SGI")]))
    t_en = det.thr(pipe.s_cal)
    tpr = float((s_te[~ben] > t_en).mean()); fpr = float((s_te[ben] > t_en).mean())
    if "s3" in todo:
        print(f"ensemble TPR {tpr:.3f} FPR {fpr:.4f} -> queue model"); print("S3 ...")
        s3 = s3_resilience(cfg, np.random.default_rng(a.seed + 3), tpr, fpr, det_delay_s=0.2); s3.to_csv(out / "s3_resilience.csv", index=False); print(s3.to_string(index=False)); print()
    if "s4" in todo:
        print("S4 ..."); s4 = s4_evasion(pipe, data, cfg, np.random.default_rng(a.seed + 4), a.seed, det.SEEN); s4.to_csv(out / "s4_evasion.csv", index=False); print(s4.to_string(index=False)); print()
    if "s5" in todo:
        print("S5 ..."); cfg["raw_dir"] = str(out); s5 = s5_spread(data, pipe, cfg, np.random.default_rng(a.seed + 5), det.SEEN); s5.to_csv(out / "s5_spread.csv", index=False); print(s5.to_string(index=False)); print()
    if "final" not in todo:
        return
    s2, s3, s4, s5 = (pd.read_csv(out / f"{n}.csv") for n in ("s2_pipeline", "s3_resilience", "s4_evasion", "s5_spread"))
    ex = pickle.load(open(out / "s2_examples.pkl", "rb"))
    s6 = s6_audit(); s6.to_csv(out / "s6_arithmetic_audit.csv", index=False)
    plots(out, ex, s3, s4)
    cl = claims(dd, cd, out); cl.to_csv(out / "claims_vs_simulation_v2.csv", index=False)

    R = Report()
    R.h(1, "MIRAGE-DoH Chapter 3: experiments, attack scenarios and performance (v2 artifact)")
    R.p("Every number was produced by the scripts in this repository on the machine described at the end. Traffic is synthetic "
        "(mirage_v2/traffic_v2.py), baselines are scikit-learn models rather than the thesis CNN-LSTM, cryptography runs on a CPU, and "
        "Classic McEliece is represented by a byte-size model only. These are not the values of Tables 8-12; they show how each mechanism "
        "behaves under controlled conditions and where the thesis figures depend on hardware or testbed data not in this repository.", note=True)
    R.h(2, "1. Claim-by-claim comparison"); R.table(cl)
    R.h(2, "2. Detection experiments (D1-D6)")
    for f, t in [("d1_seen_protocol", "D1a Detectors trained on seen families (* = zero-day)"), ("d1_all_protocol", "D1b Detectors trained on all families"),
                 ("d1_cost", "D1c Scoring cost (CPU, one core)"), ("d1_throughput", "D1d Implied instances for 15.3 M qps"),
                 ("d1_ctai", "D1e CTAI (Eq. 29, latency term omitted)"), ("d5_ablation", "D5 Full factorial ablation"),
                 ("d2_curriculum", "D2 CMA curriculum and Fisher penalty"), ("d3_tmn_hops", "D3a TMN relay-loop detection"),
                 ("d3_permutation", "D3b Order dependence"), ("d4_sgi_dqn", "D4 DQN defender, spectral vs flow state"), ("d6_levy_jumps", "D6 Levy jump test")]:
        d = read(dd / f"{f}.csv")
        if d is not None:
            R.table(d, t)
    R.image(dd / "d2_d4_curves.png", "D2/D4 curves"); R.image(dd / "d3_tmn_hops.png", "D3 TMN")
    R.h(2, "3. Cryptographic agility and causal reasoning (C1-C7)")
    for f, t in [("c1_mlkem", "C1 ML-KEM reference"), ("c2_combiner_avalanche", "C2a Combiner avalanche"), ("c2_combiner_broken_component", "C2b One component compromised"),
                 ("c3_morph_compute", "C3a Compute and wire bytes per level"), ("c3_morph_budget", "C3b Morph time vs 10 ms target"), ("c3_mceliece_pk", "C3c McEliece public key"),
                 ("c4_policy", "C4 Threat-level policy"), ("c5_causal", "C5a Causal engine"), ("c5_confounding_sweep", "C5b Confounding sweep"), ("c5_actions", "C5c Hardening actions"),
                 ("c6_se3_checks", "C6a SE(3) flow checks"), ("c6_se3_predictability", "C6b SE(3) predictability"), ("c7_audit", "C7 Scope of Eq. 12")]:
        d = read(cd / f"{f}.csv")
        if d is not None:
            R.table(d, t)
    R.h(2, "4. System-level attack scenarios (S1-S5)")
    R.table(s1, "S1 Scenario catalogue"); R.table(s2, "S2 End-to-end pipeline per attack")
    R.image(out / "s2_pipeline.png", "pipeline timelines")
    R.p("S3 is a queueing model: resolver capacity 22,000 qps, benign load 5,000 qps, 0.5 s timeout, 0.2 s detection delay, filter efficiency taken from the "
        "ensemble's measured TPR/FPR; the capacity and delay values are assumptions.")
    R.table(s3, "S3 Resilience under volumetric load (model)"); R.image(out / "s3_resilience.png", "resilience")
    R.table(s4, "S4 Evasion strength sweep"); R.image(out / "s4_evasion.png", "evasion")
    R.table(s5, "S5 Hyper-parameter spread over 20 random configurations")
    R.h(2, "5. Arithmetic audit of the Chapter 3 tables"); R.table(s6)
    R.h(2, "6. Environment")
    R.p(f"Python {platform.python_version()}, numpy {np.__version__}, {platform.platform()}; system suite wall time {time.time()-t0:.0f}s.")
    R.write(out, "MIRAGE_Chapter3_Experiment_Report_v2")
    print("wrote", out / "MIRAGE_Chapter3_Experiment_Report_v2.html")


if __name__ == "__main__":
    main()
