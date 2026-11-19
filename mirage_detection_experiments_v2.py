#!/usr/bin/env python3
"""MIRAGE-DoH Chapter 3, detection-side experiment suite (v2).

Reproduces, on SYNTHETIC encrypted-DNS metadata, the experiment types behind
Tables 8-12 and Figures 2-6 of Chapter 3:

  D1  detector comparison     MIRAGE ensemble (CMA + TMN + SGI) vs Random Forest, MLP and a
                              rule baseline; one false-positive budget for all; seen vs
                              unseen attack families; latency, throughput, CTAI (Eq. 29)
  D2  CMA curriculum          Table 10 style: initialisation -> static -> dynamic -> live,
                              F1 / AUC / novelty detection / forgetting, with the Fisher
                              penalty of Eq. (3) swept (lambda = 0 is the ablation)
  D3  TMN multi-hop memory    detection of closed relay loops vs hop count, edge-observation
                              probability and memory depth; bounded-receptive-field baseline;
                              permutation (re-ordering) invariance
  D4  SGI                     spectral state vs flow-based state for a DQN defender
                              (256-128-64, replay 1e5, target sync 1000), with a negative control
  D5  full factorial ablation all 8 subsets of {CMA, TMN, SGI}: the ablation that Sec 3.3 and
                              Sec 7.3.1 say was not performed
  D6  Levy-type jump test     Eq. (22): jump vs diffusion separation of a traffic feature stream

All numbers are produced by this script on the machine running it. The traffic
(mirage_v2/traffic_v2.py) is my construction, the baselines are scikit-learn
models rather than the thesis CNN-LSTM, and no Loihi 2 / A100 / FPGA is involved. The
tables show how the mechanisms behave when every method faces identical data;
they are not Tables 8-12.

usage:
    python3 experiments/mirage_detection_experiments_v2.py --out build/detect
    python3 experiments/mirage_detection_experiments_v2.py --quick
"""
import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage_v2 import traffic_v2 as tv, cma_v2 as cma, tmn_v2 as tmn, sgi_v2 as sgi        # noqa: E402
from mirage_v2 import topology_v2 as topo, levy_se3_v2 as levy, metrics_v2 as mt            # noqa: E402

ATTACKS = tv.ATTACKS
FPR_BUDGET = 0.01
SEEN = ["gan_flood", "tunnel", "downgrade"]            # families available to supervised training
UNSEEN = ["timing_probe", "spectral_dga"]              # zero-day families


# ==========================================================================
# data: features, persistence diagrams, spectral states, one pass per window
# ==========================================================================
def window_pack(win):
    F = tv.window_features(win)
    x = np.log(np.diff(win["ts"]) + 1e-3)
    x = (x - x.mean()) / (x.std() + 1e-9)
    dg = topo.vr_persistence(topo.delay_embed(x, 4, 2), max_scale=1.2)[1]
    f = topo.finite(dg)
    life = (f[:, 1] - f[:, 0]) if len(f) else np.array([0.0])
    H = np.array([topo.persistence_entropy(dg), life.max(), len(f)])
    S = sgi.spectral_state(win["ts"])
    return F, H, dg, S


def build_split(counts, rng):
    out = {"F": [], "H": [], "dg": [], "S": [], "y": [], "k": []}
    for kind, n in counts.items():
        for _ in range(n):
            F, H, dg, S = window_pack(tv.gen_window(kind, rng))
            out["F"].append(F); out["H"].append(H); out["dg"].append(dg); out["S"].append(S)
            out["y"].append(0 if kind == "benign" else 1); out["k"].append(kind)
    out = {k: (np.array(v, dtype=object) if k == "dg" else np.array(v)) for k, v in out.items()}
    return out


def build_data(cache, cfg, seed):
    """Builds the four splits one at a time and caches each (resumable: a long build can be
    interrupted and restarted). `cache` is the final combined pickle."""
    cache = Path(cache)
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    n = cfg["n"]
    specs = {"train": {"benign": n["ben_train"], **{a: n["atk_train"] for a in ATTACKS}},
             "val": {"benign": n["ben_val"], **{a: n["atk_val"] for a in ATTACKS}},
             "cal": {"benign": n["ben_cal"]},
             "test": {"benign": n["ben_test"], **{a: n["atk_test"] for a in ATTACKS}}}
    data = {}
    for i, (name, counts) in enumerate(specs.items()):
        part = cache.with_name(cache.stem + f".{name}.part")
        if part.exists():
            data[name] = pickle.load(open(part, "rb")); continue
        t0 = time.time()
        data[name] = build_split(counts, np.random.default_rng(seed + 17 * i))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(data[name], open(part, "wb"))
        print(f"  split {name} built in {time.time()-t0:.0f}s", flush=True)
    pickle.dump(data, open(cache, "wb"))
    return data


class Std:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9

    def __call__(self, X):
        return (X - self.mu) / self.sd


# ==========================================================================
# components as scorers (larger = more anomalous)
# ==========================================================================
class CMAScorer:
    """Evolved policy; trained by ES on the supervised + homological-novelty objective."""

    def __init__(self, data, std, hom, rng, fam, gens=400):
        tr = data["train"]
        m = np.isin(tr["k"], ["benign"] + fam)
        X, y = std(tr["F"][m]), tr["y"][m].astype(float)
        nov = hom.novelty(F=tr["H"][m].astype(float))
        self.std = std
        self.theta = cma.es_train(cma.init(rng), X, y, nov, rng, gens=gens)

    def score(self, F):
        p = cma.prob(self.theta, self.std(F))
        return np.log(p + 1e-9) - np.log(1 - p + 1e-9)


class TMNScorer:
    """Diagram attention + fixed-weight LSTM over 3 consecutive windows, logistic read-out."""

    T = 3

    def __init__(self, data, std, fam, seed):
        self.enc = tmn.TMNEncoder(hidden=24, in_dim=3 + tv.D, seed=seed)
        self.std = std
        tr = data["train"]
        m = np.isin(tr["k"], ["benign"] + fam)
        Z = self._encode(tr["dg"][m], std(tr["F"][m]))
        self.clf = tmn.Logistic().fit(Z, tr["y"][m].astype(float))

    def _encode(self, dgs, X):
        out = []
        for i in range(len(X)):
            lo = max(0, i - self.T + 1)
            out.append(self.enc.encode(dgs[lo:i + 1], X[lo:i + 1]))
        return np.array(out)

    def score(self, dgs, F):
        return self.clf.score(self._encode(dgs, self.std(F)))


class SGIScorer:
    def __init__(self, data, fam):
        tr = data["train"]
        m = np.isin(tr["k"], ["benign"] + fam)
        self.clf = tmn.Logistic().fit(tr["S"][m], tr["y"][m].astype(float))

    def score(self, S):
        return self.clf.score(S)


def rule_scores(F):
    i = {n: k for k, n in enumerate(tv.FEATURE_NAMES)}
    return np.maximum.reduce([F[:, i["ent_mean"]] - 3.6, F[:, i["tls12_frac"]] - 0.5, F[:, i["hs_rate"]] - 0.6,
                              F[:, i["lowband_power"]] - 0.7])


def thr(scores_benign, fpr=FPR_BUDGET):
    return float(np.quantile(scores_benign, 1 - fpr))


def tpr_ci(s, t):
    k = int((s > t).sum()); lo, hi = mt.wilson(k, len(s))
    return k / len(s), lo, hi


def auc(neg, pos):
    s = np.r_[neg, pos]; y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return mt.auc(s, y)


# ==========================================================================
# D1 / D5
# ==========================================================================
def fit_all(data, cfg, seed, fam):
    rng = np.random.default_rng(seed)
    std = Std(data["cal"]["F"])
    hom = cma.Homological(); hom.mu = data["cal"]["H"].astype(float).mean(0)
    Hc = data["cal"]["H"].astype(float)
    hom.Ci = np.linalg.inv(np.cov(Hc.T) + 1e-3 * np.eye(3))
    cm = CMAScorer(data, std, hom, rng, fam, gens=cfg["gens"])
    tm = TMNScorer(data, std, fam, seed)
    sg = SGIScorer(data, fam)
    return std, cm, tm, sg


def comp_scores(split, cm, tm, sg):
    return {"CMA": cm.score(split["F"]), "TMN": tm.score(split["dg"], split["F"]), "SGI": sg.score(split["S"])}


def stack(subset, val, cal, tst, comps_v, comps_c, comps_t, fam):
    """Logistic stacking of the chosen component scores, fitted on the validation split."""
    mv = np.isin(val["k"], ["benign"] + fam)
    Xv = np.column_stack([comps_v[c][mv] for c in subset]); yv = val["y"][mv].astype(float)
    clf = tmn.Logistic().fit(Xv, yv)
    f = lambda comps: clf.score(np.column_stack([comps[c] for c in subset]))
    return f(comps_c), f(comps_t)


def baselines(data, std, fam, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    tr = data["train"]; m = np.isin(tr["k"], ["benign"] + fam)
    X, y = std(tr["F"][m]), tr["y"][m]
    rf = RandomForestClassifier(200, n_jobs=1, random_state=seed).fit(X, y)
    mlp = MLPClassifier((64, 32), max_iter=300, random_state=seed).fit(X, y)
    return rf, mlp


def d1_d5(data, cfg, seed):
    out = {}
    for tag, fam in [("seen", SEEN), ("all", ATTACKS)]:
        std, cm, tm, sg = fit_all(data, cfg, seed, fam)
        cv, cc, ct = (comp_scores(data[s], cm, tm, sg) for s in ("val", "cal", "test"))
        rf, mlp = baselines(data, std, fam, seed)
        te = data["test"]; ben = te["k"] == "benign"
        det = {}
        det["Rule-based"] = (rule_scores(data["cal"]["F"]), rule_scores(te["F"]))
        det["Random Forest (sklearn)"] = (rf.predict_proba(std(data["cal"]["F"]))[:, 1], rf.predict_proba(std(te["F"]))[:, 1])
        det["MLP (sklearn)"] = (mlp.predict_proba(std(data["cal"]["F"]))[:, 1], mlp.predict_proba(std(te["F"]))[:, 1])
        for name, sub in [("CMA alone", ["CMA"]), ("TMN alone", ["TMN"]), ("SGI alone", ["SGI"]),
                          ("MIRAGE ensemble (CMA+TMN+SGI)", ["CMA", "TMN", "SGI"])]:
            det[name] = stack(sub, data["val"], data["cal"], te, cv, cc, ct, fam)
        out[tag] = dict(det=det, std=std, cm=cm, tm=tm, sg=sg, cv=cv, cc=cc, ct=ct, fam=fam, rf=rf, mlp=mlp)
    return out


def table_d1(res, data, tag):
    te = data["test"]; ben = te["k"] == "benign"
    rows = []
    for name, (sc_cal, sc_te) in res[tag]["det"].items():
        t = thr(sc_cal)
        fp = int((sc_te[ben] > t).sum()); lo, hi = mt.wilson(fp, int(ben.sum()))
        row = {"detector": name, "FPR": f"{fp/ben.sum():.2%} [{lo:.2%},{hi:.2%}]"}
        tp_all, tn = [], int((sc_te[ben] <= t).sum())
        for a in ATTACKS:
            s = sc_te[te["k"] == a]
            r, l, h = tpr_ci(s, t)
            tp_all.append(int((s > t).sum()))
            row[("*" if a in UNSEEN and tag == "seen" else "") + a] = f"{r:.1%}"
        n_att = int((~ben).sum())
        row["accuracy"] = f"{(sum(tp_all) + tn) / len(te['k']):.2%}"
        row["mean AUC"] = f"{np.mean([auc(sc_te[ben], sc_te[te['k'] == a]) for a in ATTACKS]):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def table_d5(res, data):
    """Full factorial ablation of {CMA, TMN, SGI}, trained on the seen families, evaluated on all."""
    r = res["seen"]; te = data["test"]; ben = te["k"] == "benign"
    names = ["CMA", "TMN", "SGI"]
    rows = []
    for mask in range(1, 8):
        sub = [n for i, n in enumerate(names) if mask >> i & 1]
        sc_cal, sc_te = stack(sub, data["val"], data["cal"], te, r["cv"], r["cc"], r["ct"], r["fam"])
        t = thr(sc_cal)
        row = {"components": "+".join(sub)}
        seen_tpr = np.mean([(sc_te[te["k"] == a] > t).mean() for a in SEEN])
        unseen_tpr = np.mean([(sc_te[te["k"] == a] > t).mean() for a in UNSEEN])
        row["FPR"] = f"{(sc_te[ben] > t).mean():.2%}"
        row["TPR seen families"] = f"{seen_tpr:.1%}"
        row["TPR unseen (zero-day)"] = f"{unseen_tpr:.1%}"
        row["AUC unseen"] = f"{np.mean([auc(sc_te[ben], sc_te[te['k'] == a]) for a in UNSEEN]):.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def d1_cost(res, data, cfg):
    """Per-window scoring time of every detector; implied instance count for Table 8's throughput."""
    r = res["all"]; te = data["test"]
    idx = np.arange(min(200, len(te["k"])))
    F, dg, S = te["F"][idx], te["dg"][idx], te["S"][idx]
    def tm(f, reps=3):
        t = []
        for _ in range(reps):
            a = time.perf_counter(); f(); t.append(time.perf_counter() - a)
        return min(t) / len(idx) * 1e6
    rows = [("Rule-based", tm(lambda: rule_scores(F))),
            ("Random Forest (200 trees)", tm(lambda: r["rf"].predict_proba(r["std"](F)))),
            ("MLP", tm(lambda: r["mlp"].predict_proba(r["std"](F)))),
            ("CMA policy forward", tm(lambda: r["cm"].score(F))),
            ("TMN (diagram attention + LSTM, 3-window)", tm(lambda: r["tm"].score(dg, F))),
            ("SGI spectral classifier", tm(lambda: r["sg"].score(S)))]
    df = pd.DataFrame(rows, columns=["scorer (given precomputed inputs)", "us/window"])
    # feature extraction is shared by all detectors
    rng = np.random.default_rng(1)
    wins = [tv.gen_window("benign", rng) for _ in range(60)]
    f_t = min(_t(lambda: [tv.window_features(w) for w in wins]) for _ in range(3)) / 60 * 1e6
    h_t = min(_t(lambda: [window_pack(w) for w in wins]) for _ in range(3)) / 60 * 1e6
    df.loc[len(df)] = ("metadata features (20) from a window", f_t)
    df.loc[len(df)] = ("persistence + spectral state from a window", h_t - f_t)
    df["us/window"] = df["us/window"].round(1)
    ens = h_t + sum(v for k, v in rows if k.startswith(("CMA", "TMN", "SGI")))
    q_per_s = (1e6 / ens) * tv.W
    ex = pd.DataFrame([{"quantity": "ensemble cost per 64-query window (us)", "value": f"{ens:.0f}"},
                       {"quantity": "single-core queries/s (this implementation)", "value": f"{q_per_s:,.0f}"},
                       {"quantity": "instances needed for 15.3M qps (Table 8) at this speed", "value": f"{15.3e6 / q_per_s:,.0f}"},
                       {"quantity": "CPU per instance bound in Sec 3.4.2", "value": "< 5 % of resolver capacity"}])
    return df, ex


def _t(f):
    a = time.perf_counter(); f(); return time.perf_counter() - a


def ctai_table(res, data):
    te = data["test"]; ben = te["k"] == "benign"
    rows = []
    r = res["all"]
    for name, (sc_cal, sc_te) in r["det"].items():
        t = thr(sc_cal)
        acc = ((sc_te[~ben] > t).sum() + (sc_te[ben] <= t).sum()) / len(sc_te)
        fp = (sc_te[ben] > t).mean()
        rows.append({"detector": name, "accuracy": f"{acc:.4f}", "FPR": f"{fp:.4f}",
                     "CTAI (alpha=1,1,0 ; latency omitted)": f"{mt.ctai([acc], [fp], [0.0], 1, 1, 0):.4f}"})
    return pd.DataFrame(rows)


# ==========================================================================
# D2 CMA curriculum
# ==========================================================================
def curriculum_phases():
    return [("Initialization", ["gan_flood"], 0.0),
            ("Phase 1 (static morphologies)", ["gan_flood", "timing_probe"], 0.45),
            ("Phase 2 (dynamic threats)", ["downgrade", "tunnel"], 0.0),
            ("Phase 3 (live)", ["spectral_dga", "tunnel", "gan_flood"], 0.0)]


def d2_curriculum(cfg, seed, lam, rng_master):
    """Train sequentially through the four phases. Returns per-phase metrics."""
    rng = np.random.default_rng(seed)
    nb = cfg["cur"]["benign"]
    wins_b_all = tv.make_windows("benign", nb * 5, rng)
    Fb = np.array([tv.window_features(w) for w in wins_b_all])
    Hb = np.array([cma.Homological.feats(w) for w in wins_b_all[:nb * 4]])     # benign test windows need no persistence
    std = Std(Fb[:nb])
    hom = cma.Homological(); hom.mu = Hb[:nb].mean(0); hom.Ci = np.linalg.inv(np.cov(Hb[:nb].T) + 1e-3 * np.eye(3))
    mu_b = Fb.mean(0)
    phases = curriculum_phases()
    n = cfg["cur"]["per_family"]
    pools = {}
    def fam_data(fam, morph):
        wins = tv.make_windows(fam, n * 2, rng)
        F = np.array([tv.window_features(w) for w in wins])
        H = np.array([cma.Homological.feats(w) for w in wins[:n]])        # persistence only where used (training half)
        if morph > 0:
            F = tv.morph(F, morph, mu_b, rng)
        return F[:n], H, F[n:], None
    benign_tr, Hb_tr = Fb[nb:nb * 4], Hb[nb:nb * 4]
    benign_te = Fb[nb * 4:]
    te_cache = {pj: te_sets_for(cfg, rng, phases[pj][1], std, mu_b, phases[pj][2]) for pj in range(len(phases))}
    theta = cma.init(rng); theta_star, Fish = None, None
    rows, history, seen_sets = [], [], []
    f1_peak = {}
    for pi, (name, fams, morph) in enumerate(phases):
        tr_X, tr_y, tr_H, te_sets = [std(benign_tr)], [np.zeros(len(benign_tr))], [hom.novelty(F=Hb_tr)], []
        te_pos = []
        for f in fams:
            Xa, Ha, Xt, Ht = fam_data(f, morph if f in ("gan_flood", "timing_probe") else 0.0)
            tr_X.append(std(Xa)); tr_y.append(np.ones(len(Xa))); tr_H.append(hom.novelty(F=Ha))
            te_pos.append((f, std(Xt)))
        X = np.vstack(tr_X); y = np.concatenate(tr_y); nov = np.concatenate(tr_H)
        Xte_b = std(benign_te)
        # zero-shot novelty detection: before training on this phase, families new to the agent
        new = [f for f in fams if f not in sum(seen_sets, [])]
        zs = np.mean([(cma.prob(theta, Xt) >= 0.5).mean() for f, Xt in te_pos if f in new]) if (new and pi > 0) else np.nan
        theta = cma.es_train(theta, X, y, nov, rng, gens=cfg["cur"]["gens"], lam=lam,
                             theta_star=theta_star, F=Fish)
        # Fisher on the data of THIS phase, anchor = parameters after this phase
        Fish_new = cma.fisher_diag(theta, X, y)
        Fish = Fish_new if Fish is None else 0.5 * Fish + 0.5 * Fish_new
        theta_star = theta.copy()
        seen_sets.append(fams)
        # evaluation on the current phase, plus all previous phases
        ev = {}
        for pj in range(pi + 1):
            fj = phases[pj][1]
            Xp = np.vstack([Xte_b] + [Xt for f, Xt in te_cache[pj]])
            yp = np.r_[np.zeros(len(Xte_b)), np.ones(len(Xp) - len(Xte_b))]
            ev[pj] = cma.f1_auc(theta, Xp, yp)
        f1, au, tpr, fpr = ev[pi]
        for pj in range(pi + 1):
            f1_peak[pj] = max(f1_peak.get(pj, 0.0), ev[pj][0])
        forget = np.mean([max(0.0, (f1_peak[pj] - ev[pj][0]) / max(f1_peak[pj], 1e-9)) for pj in range(pi)]) if pi else 0.0
        adapted = np.mean([(cma.prob(theta, Xt) >= 0.5).mean() for f, Xt in te_pos if f in new]) if new else np.nan
        rows.append(dict(phase=name, f1=f1, auc=au, novelty_zero_shot=zs, novelty_adapted=adapted,
                         forgetting=forget, fpr=fpr))
    return rows


def te_sets_for(cfg, rng, fams, std, mu_b, morph):
    out = []
    for f in fams:
        wins = tv.make_windows(f, cfg["cur"]["per_family"], rng)
        F = np.array([tv.window_features(w) for w in wins])
        if morph > 0 and f in ("gan_flood", "timing_probe"):
            F = tv.morph(F, morph, mu_b, rng)
        out.append((f, std(F)))
    return out


def table_d2(cfg, seeds, out=None, only_lam=None):
    lams = cfg["cur"]["lams"]
    rows, allres = [], {}
    for lam in lams:
        per = []
        for s_ in seeds:
            f = None if out is None else Path(out) / "d2_runs" / f"lam{lam}_seed{s_}.json"
            if f is not None and f.exists():
                per.append(json.load(open(f))); continue
            if only_lam is not None and lam not in only_lam:
                per = None; break
            r = d2_curriculum(cfg, s_, lam, None)
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True); json.dump(r, open(f, "w"))
            per.append(r)
        if per is None:
            continue
        allres[lam] = per
        for pi in range(4):
            g = lambda k: np.array([per[s][pi][k] if per[s][pi][k] is not None else np.nan for s in range(len(seeds))], float)
            rows.append({"lambda (Fisher)": lam, "phase": per[0][pi]["phase"],
                         "F1 (%)": f"{100*g('f1').mean():.1f}", "AUC": f"{g('auc').mean():.3f}",
                         "novelty zero-shot (%)": "-" if np.all(np.isnan(g('novelty_zero_shot'))) else f"{100*np.nanmean(g('novelty_zero_shot')):.1f}",
                         "novelty after adaptation (%)": "-" if np.all(np.isnan(g('novelty_adapted'))) else f"{100*np.nanmean(g('novelty_adapted')):.1f}",
                         "forgetting (%)": f"{100*g('forgetting').mean():.1f} ± {100*1.96*g('forgetting').std(ddof=1)/np.sqrt(len(seeds)):.1f}" if len(seeds) > 1 else f"{100*g('forgetting').mean():.1f}",
                         "FPR (%)": f"{100*g('fpr').mean():.2f}"})
    return pd.DataFrame(rows), allres


# ==========================================================================
# D3 TMN
# ==========================================================================
def d3_hops(cfg, rng):
    n_bg = 120
    trials = cfg["tmn"]["trials"]
    # threshold at FPR 1 % on background-only graphs
    bg = []
    for _ in range(cfg["tmn"]["calib"]):
        t, E = tmn.background_graph(n_bg, rng); bg.append(tmn.chain_score(n_bg, E))
    t_chain = float(np.quantile(bg, 1 - FPR_BUDGET))
    rows = []
    for L in cfg["tmn"]["hops"]:
        for p in cfg["tmn"]["p_obs"]:
            for T in cfg["tmn"]["mem"]:
                det = 0
                for _ in range(trials):
                    t, E = tmn.background_graph(n_bg, rng)
                    E2, chain, N = tmn.add_relay_chain(E, n_bg, L, rng)
                    seen = set()
                    for _w in range(T):                         # memory: union of observed hop edges over T windows
                        seen |= {i for i, e in enumerate(chain) if rng.random() < p}
                    obs = [e for i, e in enumerate(chain) if i in seen]
                    base = [e for e in E2 if e not in chain]
                    det += int(tmn.chain_score(N, base + obs) > t_chain)
                rows.append({"hops L": L, "P(edge observed)": p, "memory windows T": T, "detection": det / trials})
    df = pd.DataFrame(rows)
    # bounded receptive field baseline (modelling assumption): sees a chain only if L <= R
    R = 35
    df["bounded-RF baseline (R=35), full observation"] = [float(L <= R) for L in df["hops L"]]
    return df, t_chain


def d3_permutation(data, cfg, seed):
    """Order-dependence: shuffle packet order inside windows (adversarial permutation of the
    sequence). Persistence features of the *distance-matrix* are unchanged by relabelling of
    points; a sequence-ordered feature (lag autocorrelation) is not."""
    rng = np.random.default_rng(seed)
    rows = []
    wins = tv.make_windows("tunnel", 80, rng) + tv.make_windows("spectral_dga", 80, rng)
    d_topo, d_ac = [], []
    for w in wins:
        x = np.log(np.diff(w["ts"]) + 1e-3); x = (x - x.mean()) / (x.std() + 1e-9)
        E = topo.delay_embed(x, 4, 2)
        perm = rng.permutation(len(E))
        a = topo.vr_persistence(E, 1.2)[1]
        b = topo.vr_persistence(E[perm], 1.2)[1]                  # same point cloud, points relabelled
        d_topo.append(topo.wasserstein(a, b))
        xs = rng.permutation(x)                                    # packet order shuffled before embedding
        ac = lambda v: float(np.mean(v[:-1] * v[1:]))
        d_ac.append(abs(ac(x) - ac(xs)))
    rows.append({"feature": "persistence diagram of the point cloud (relabelled points)", "mean change": f"{np.mean(d_topo):.2e}"})
    rows.append({"feature": "lag-1 autocorrelation (packet order shuffled)", "mean change": f"{np.mean(d_ac):.3f}"})
    return pd.DataFrame(rows)


# ==========================================================================
# D4 SGI / DQN
# ==========================================================================
def d4_dqn(data, cfg, seed, attack="spectral_dga", state="spectral"):
    rng = np.random.default_rng(seed)
    tr = data["train"]; te = data["test"]
    key = "S" if state == "spectral" else "F"
    Xtr_all, Xte_all = tr[key].astype(float), te[key].astype(float)
    std = Std(Xtr_all[tr["k"] == "benign"])
    pool = {k: std(Xtr_all[tr["k"] == k]) for k in ["benign", attack]}
    agent = sgi.DQNAgent(Xtr_all.shape[1], rng, target_every=cfg["dqn"]["target_every"])
    episodes, steps = cfg["dqn"]["episodes"], 12
    curve, eps0 = [], 1.0
    for ep in range(episodes):
        eps = max(0.05, eps0 * (1 - ep / (0.6 * episodes)))
        t0 = int(rng.integers(2, 8)); R = 0.0
        for t in range(steps):
            active = int(t >= t0)
            s = pool["spectral_dga" if active and attack == "spectral_dga" else (attack if active else "benign")]
            s = s[rng.integers(0, len(s))]
            a = agent.act(s, eps)
            r = sgi.REWARD[active, a]
            s2 = pool["benign"][0]                      # one-step (gamma=0) problem: next state unused
            agent.remember(s, a, r, s2); agent.learn(); R += r
        curve.append(R / steps)
    # evaluation
    ben = std(Xte_all[te["k"] == "benign"]); atk = std(Xte_all[te["k"] == attack])
    q = lambda X: np.argmax(agent.q.forward(X), 1)
    a_b, a_a = q(ben), q(atk)
    fpr = float((a_b != 0).mean()); tpr = float((a_a != 0).mean())
    acc = (np.sum(a_b == 0) + np.sum(a_a != 0)) / (len(ben) + len(atk))
    mean_r = (sgi.REWARD[0, a_b].mean() * len(ben) + sgi.REWARD[1, a_a].mean() * len(atk)) / (len(ben) + len(atk))
    return dict(acc=acc, tpr=tpr, fpr=fpr, mean_reward=mean_r, curve=curve)


def table_d4(data, cfg, seeds):
    rows, curves = [], {}
    for attack, role in [("spectral_dga", "target (structural)"), ("gan_flood", "negative control (flow-visible)")]:
        for state in ["spectral", "flow"]:
            res = [d4_dqn(data, cfg, s, attack, state) for s in seeds]
            g = lambda k: np.array([r[k] for r in res])
            rows.append({"scenario": f"{attack} [{role}]", "state": "spectral (k=16 eigenvalues)" if state == "spectral" else "flow metadata (20 features)",
                         "accuracy": f"{100*g('acc').mean():.1f} ± {100*1.96*g('acc').std(ddof=1)/np.sqrt(len(seeds)):.1f}" if len(seeds) > 1 else f"{100*g('acc').mean():.1f}",
                         "TPR": f"{100*g('tpr').mean():.1f}", "FPR": f"{100*g('fpr').mean():.1f}",
                         "mean reward/step": f"{g('mean_reward').mean():.3f}"})
            curves[(attack, state)] = np.mean([r["curve"] for r in res], 0)
    return pd.DataFrame(rows), curves


# ==========================================================================
# D6 Levy
# ==========================================================================
def d6_levy(cfg, rng):
    rows = []
    for jr in [0.0, 0.005, 0.01, 0.02]:
        for js in [3.0, 6.0]:
            if jr == 0 and js == 6.0:
                continue
            stats = []
            for _ in range(cfg["levy"]["trials"]):
                x, _ = levy.levy_path(1000, 1.0, jr, js, rng); stats.append(levy.bns_jump_stat(x))
            rows.append({"jump rate/step": jr, "jump scale (x sigma)": js, "mean 1-BV/RV": np.mean(stats),
                         "sd": np.std(stats, ddof=1)})
    null = []
    for _ in range(cfg["levy"]["trials"] * 2):
        null.append(levy.bns_jump_stat(rng.normal(size=1000)))
    thr_ = float(np.quantile(null, 0.99))
    df = pd.DataFrame(rows)
    pw = []
    for _, r in df.iterrows():
        if r["jump rate/step"] == 0:
            pw.append(float("nan")); continue
        hit = 0
        for _ in range(cfg["levy"]["trials"]):
            x, _ = levy.levy_path(1000, 1.0, r["jump rate/step"], r["jump scale (x sigma)"], rng)
            hit += levy.bns_jump_stat(x) > thr_
        pw.append(hit / cfg["levy"]["trials"])
    df["power @1% FPR"] = pw
    df["mean 1-BV/RV"] = df["mean 1-BV/RV"].round(3); df["sd"] = df["sd"].round(3)
    return df, thr_


# ==========================================================================
def md(df, title):
    return f"### {title}\n\n{df.to_markdown(index=False)}\n"


def plots(out, res_d2, d4curves, d3df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for lam, per in res_d2.items():
        f = np.array([[p["forgetting"] for p in run] for run in per]).mean(0) * 100
        ax[0].plot(range(4), f, "o-", label=f"lambda={lam}")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Init", "P1", "P2", "P3"])
    ax[0].set_ylabel("forgetting (%)"); ax[0].set_title("D2 forgetting vs Fisher weight"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for (a, s), c in d4curves.items():
        k = 25
        sm = np.convolve(c, np.ones(k) / k, mode="valid")
        ax[1].plot(sm, label=f"{a} / {s}")
    ax[1].set_xlabel("episode"); ax[1].set_ylabel("mean reward per step"); ax[1].set_title("D4 DQN training"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "d2_d4_curves.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for (p, T), g in d3df.groupby(["P(edge observed)", "memory windows T"]):
        ax.plot(g["hops L"], g["detection"], "o-", label=f"p={p}, T={T}")
    ax.axvline(35, color="k", ls=":", lw=.8)
    ax.set_xlabel("relay-loop length (hops)"); ax.set_ylabel("detection rate"); ax.set_title("D3 TMN memory"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "d3_tmn_hops.png", dpi=140); plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/detect")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of: data,d1,d2,d3,d4,d6,report (default: all)")
    ap.add_argument("--lam", default="", help="restrict D2 to these lambdas (comma list) in this call")
    a = ap.parse_args()
    cfg = dict(n=dict(ben_train=2400, atk_train=500, ben_val=900, atk_val=200, ben_cal=1500, ben_test=2500, atk_test=500),
               gens=400, seeds=3,
               cur=dict(benign=400, per_family=200, gens=200, lams=[0.0, 10.0, 100.0, 1000.0]),
               tmn=dict(hops=[10, 30, 50, 80], p_obs=[0.95, 0.99], mem=[1, 3], trials=40, calib=200),
               dqn=dict(episodes=400, target_every=1000), levy=dict(trials=60))
    if a.quick:
        cfg = dict(n=dict(ben_train=300, atk_train=80, ben_val=150, atk_val=40, ben_cal=300, ben_test=300, atk_test=60),
                   gens=120, seeds=1, cur=dict(benign=150, per_family=60, gens=60, lams=[0.0, 100.0]),
                   tmn=dict(hops=[10, 50], p_obs=[0.99], mem=[1, 3], trials=8, calib=40),
                   dqn=dict(episodes=60, target_every=100), levy=dict(trials=10))
    todo = set(a.only.split(",")) if a.only else {"data", "d1", "d2", "d3", "d4", "d6", "report"}
    only_lam = [float(x) for x in a.lam.split(",")] if a.lam else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = list(range(cfg["seeds"]))
    t0 = time.time()
    data = build_data(out / ("data_quick.pkl" if a.quick else "data.pkl"), cfg, a.seed)
    if "data" in todo and todo == {"data"}:
        return
    def emit(name, df, title):
        df.to_csv(out / f"{name}.csv", index=False); print(f"== {title}"); print(df.to_string(index=False)); print(flush=True)
    res = None
    if "d1" in todo:
        res = d1_d5(data, cfg, a.seed)
        emit("d1_seen_protocol", table_d1(res, data, "seen"), "D1a seen protocol")
        emit("d1_all_protocol", table_d1(res, data, "all"), "D1b all-family protocol")
        cost, ex = d1_cost(res, data, cfg); emit("d1_cost", cost, "D1c cost"); emit("d1_throughput", ex, "D1d throughput")
        emit("d1_ctai", ctai_table(res, data), "D1e CTAI")
        emit("d5_ablation", table_d5(res, data), "D5 ablation")
    if "d2" in todo:
        d2, _ = table_d2(cfg, seeds, out, only_lam); emit("d2_curriculum", d2, "D2 curriculum (completed lambdas so far)")
    if "d3" in todo:
        d3, tchain = d3_hops(cfg, rng); d3.to_csv(out / "d3_tmn_hops.csv", index=False)
        emit("d3_permutation", d3_permutation(data, cfg, a.seed), "D3b order dependence"); print(d3.to_string(index=False)); print(f"chain threshold {tchain:.3f}")
    if "d4" in todo:
        d4, curves = table_d4(data, cfg, seeds); emit("d4_sgi_dqn", d4, "D4 DQN defender")
        pickle.dump(curves, open(out / "d4_curves.pkl", "wb"))
    if "d6" in todo:
        d6, thr6 = d6_levy(cfg, rng); emit("d6_levy_jumps", d6, f"D6 Levy jump test (null 99th pct {thr6:.3f})")
    if "report" in todo:
        _, res_d2 = table_d2(cfg, seeds, out, only_lam=[])
        d3 = pd.read_csv(out / "d3_tmn_hops.csv"); curves = pickle.load(open(out / "d4_curves.pkl", "rb"))
        plots(out, res_d2, curves, d3)
        rep = ["# MIRAGE-DoH detection experiments (synthetic metadata)\n"]
        for f in ["d1_seen_protocol", "d1_all_protocol", "d1_cost", "d1_throughput", "d1_ctai", "d5_ablation", "d2_curriculum",
                  "d3_tmn_hops", "d3_permutation", "d4_sgi_dqn", "d6_levy_jumps"]:
            if (out / f"{f}.csv").exists():
                rep.append(md(pd.read_csv(out / f"{f}.csv"), f))
        (out / "detection_report_v2.md").write_text("\n".join(rep))
        json.dump(cfg, open(out / "detection_config_v2.json", "w"))
    print(f"stage wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
