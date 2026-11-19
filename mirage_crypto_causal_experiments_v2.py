#!/usr/bin/env python3
"""MIRAGE-DoH Chapter 3, cryptographic-agility and causal-reasoning experiment suite (v2).

  C1  ML-KEM reference            correctness over many trials, implicit rejection, key/ciphertext
                                  sizes against FIPS 203, measured timing (pure Python: upper bound)
  C2  hybrid combiner (Eq. 11)    avalanche, context binding, and the 'one component broken' test
  C3  morph-time budget           Table 5 / Table 9: measured compute + byte accounting + a
                                  network model, against the 10 ms target of Objective 3;
                                  McEliece public-key transfer is the dominant term at Pi_2
  C4  threat-level policy (Eq. 13) Pi_0 floor, escalation delay, switching rate, threshold
                                  evasion and oscillation attacks
  C5  causal engine (Eqs. 14-16)  observational vs interventional vs counterfactual risk,
                                  confounding bias sweep, chosen hardening action
  C6  SE(3) re-randomisation (Eq. 26) group closure, key statistics, avalanche,
                                  predictability vs diffusion schedule, step cost
  C7  Eq. 12 / Table 5 audit      what the security statement claims at each level

Reading guide. Real components: X25519 and ChaCha20-Poly1305 (cryptography package), SHAKE/SHA3
(hashlib). Educational components: ML-KEM (pure Python, FIPS 203 structure, not KAT validated).
Not implemented: Classic McEliece (only a byte-size model is used; its timings are NOT measured).
Network times are computed from stated link parameters, not measured. Nothing here is a
Loihi 2 / FPGA measurement, so Table 9's 3.8-9.5 ms cannot be reproduced; C3 shows what a
CPU-only reference can and cannot say about it.

usage:
    python3 experiments/mirage_crypto_causal_experiments_v2.py --out build/crypto
    python3 experiments/mirage_crypto_causal_experiments_v2.py --quick
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage_v2.mlkem_v2 import MLKEM, PARAMS                                      # noqa: E402
from mirage_v2 import mcs_v2 as mcs, cce_v2 as cce, levy_se3_v2 as se3           # noqa: E402

FIPS203_SIZES = {"ML-KEM-512": (800, 1632, 768), "ML-KEM-768": (1184, 2400, 1088), "ML-KEM-1024": (1568, 3168, 1568)}


def med_ms(xs):
    return float(np.median(xs) * 1e3)


# ==========================================================================
def c1_kem(cfg):
    rows = []
    for name in PARAMS:
        kem = MLKEM(name)
        N = cfg["kem_trials"]
        ok, tk, te, td = 0, [], [], []
        for _ in range(N):
            t = time.perf_counter(); ek, dk = kem.keygen(); tk.append(time.perf_counter() - t)
            t = time.perf_counter(); K, c = kem.encaps(ek); te.append(time.perf_counter() - t)
            t = time.perf_counter(); K2 = kem.decaps(dk, c); td.append(time.perf_counter() - t)
            ok += int(K == K2)
        ek, dk = kem.keygen(); K, c = kem.encaps(ek)
        rej = 0; det = True
        for i in range(cfg["reject_trials"]):
            bad = bytearray(c); bad[int(np.random.default_rng(i).integers(len(c)))] ^= 1 << (i % 8)
            Kb = kem.decaps(dk, bytes(bad)); rej += int(Kb != K)
            det &= kem.decaps(dk, bytes(bad)) == Kb          # deterministic rejection key
        sz = (len(ek), len(dk), len(c))
        rows.append({"parameter set": name, "ek/dk/ct bytes": f"{sz[0]}/{sz[1]}/{sz[2]}",
                     "matches FIPS 203 sizes": sz == FIPS203_SIZES[name],
                     f"round trips ok ({N})": f"{ok}/{N}", f"implicit rejection ({cfg['reject_trials']} corrupted ct)": f"{rej}/{cfg['reject_trials']}",
                     "rejection deterministic": det,
                     "keygen ms": f"{med_ms(tk):.1f}", "encaps ms": f"{med_ms(te):.1f}", "decaps ms": f"{med_ms(td):.1f}"})
    return pd.DataFrame(rows)


# ==========================================================================
def c2_combiner(cfg):
    rng = np.random.default_rng(0)
    N = cfg["combiner_trials"]
    rows = []
    d1, d2, d3 = [], [], []
    for _ in range(N):
        ss1, ss2 = rng.bytes(32), rng.bytes(32); ct1, ct2 = rng.bytes(1088), rng.bytes(32)
        K = mcs.combine(ss1, ss2, ct1, ct2, b"ctx")
        b1 = bytearray(ss1); b1[0] ^= 1
        d1.append(bin(int.from_bytes(K, "big") ^ int.from_bytes(mcs.combine(bytes(b1), ss2, ct1, ct2, b"ctx"), "big")).count("1"))
        b2 = bytearray(ss2); b2[7] ^= 0x10
        d2.append(bin(int.from_bytes(K, "big") ^ int.from_bytes(mcs.combine(ss1, bytes(b2), ct1, ct2, b"ctx"), "big")).count("1"))
        d3.append(bin(int.from_bytes(K, "big") ^ int.from_bytes(mcs.combine(ss1, ss2, ct1, ct2, b"ctx2"), "big")).count("1"))
    for lab, d in [("flip 1 bit of ss1", d1), ("flip 1 bit of ss2", d2), ("change context string", d3)]:
        rows.append({"perturbation": lab, "mean bit flips of 256": f"{np.mean(d):.1f}", "sd": f"{np.std(d, ddof=1):.1f}",
                     "min": int(np.min(d)), "max": int(np.max(d)), "ideal (binomial)": "128.0 +- 8.0"})
    av = pd.DataFrame(rows)
    # one component compromised: adversary knows ss1 exactly, ss2 secret and uniform.
    keys = []
    for _ in range(cfg["bias_keys"]):
        ss1 = b"\x00" * 32                                   # fully known (broken component)
        keys.append(mcs.combine(ss1, rng.bytes(32), b"\x00" * 1088, rng.bytes(32), b"ctx"))
    bits = np.unpackbits(np.frombuffer(b"".join(keys), np.uint8))
    n1 = int(bits.sum()); n = len(bits)
    z = (n1 - n / 2) / np.sqrt(n / 4)
    runs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    pi = n1 / n
    zr = (runs - (2 * n * pi * (1 - pi) + 1)) / (2 * np.sqrt(2 * n) * pi * (1 - pi))
    byte_counts = np.bincount(np.frombuffer(b"".join(keys), np.uint8), minlength=256)
    chi = stats.chisquare(byte_counts)
    broken = pd.DataFrame([{"scenario": "ss1 fully known to the adversary, ss2 uniform",
                            "keys": cfg["bias_keys"], "monobit z": f"{z:+.2f}", "runs z": f"{zr:+.2f}",
                            "byte chi-square p": f"{chi.pvalue:.3f}"}])
    # AEAD tamper check
    key = mcs.combine(b"a" * 32, b"b" * 32, b"c", b"d", b"x")
    blob = mcs.seal(key, b"control-metadata", b"ad")
    bad = bytearray(blob); bad[-1] ^= 1
    try:
        mcs.open_(key, bytes(bad), b"ad"); tam = "ACCEPTED (bug)"
    except Exception:
        tam = "rejected"
    return av, broken, tam


# ==========================================================================
def c3_morph(cfg):
    comp_rows, budget_rows = [], []
    ref = {}
    for L in (0, 1, 2):
        runs = [mcs.handshake(L) for _ in range(cfg["handshakes"])]
        t = {k: np.median([r.t[k] for r in runs]) for k in runs[0].t}
        ref[L] = (runs[0], t)
        meas_cpu = sum(v for k, v in t.items() if not (L == 2 and k.endswith("kem2")))
        names = mcs.LEVELS[L]["kems"]
        comp_rows.append({"level": mcs.LEVELS[L]["name"], "KEMs": " + ".join(names), "NIST category": mcs.LEVELS[L]["nist_category"],
                          "keygen ms": f"{(t['keygen_kem1'] + (0 if L == 2 else t['keygen_kem2']))*1e3:.1f}",
                          "encaps ms": f"{(t['encaps_kem1'] + (0 if L == 2 else t['encaps_kem2']))*1e3:.1f}",
                          "decaps ms": f"{(t['decaps_kem1'] + (0 if L == 2 else t['decaps_kem2']))*1e3:.1f}",
                          "KDF+AEAD ms": f"{(t['combine_kdf'] + t['aead_seal'] + t['aead_open'])*1e3:.3f}",
                          "measured CPU total ms": f"{meas_cpu*1e3:.1f}" + (" (McEliece part not measured)" if L == 2 else ""),
                          "c->s bytes": runs[0].wire_c2s, "s->c bytes": runs[0].wire_s2c})
    nets = [(100, 5), (1000, 1), (1000, 5), (10000, 1)]
    for L in (0, 1, 2):
        for pre in ([False, True] if L == 2 else [False]):
            wb = mcs.wire_bytes(L, pre)
            h, t = ref[L]
            cpu = sum(v for k, v in t.items() if not (L == 2 and k.endswith("kem2")))
            for mbps, rtt in nets:
                tr = mcs.transfer_ms(wb["client_to_server"] + wb["server_to_client"], mbps, rtt, rtts=1)
                tot = cpu * 1e3 + tr
                budget_rows.append({"level": mcs.LEVELS[L]["name"] + (" (pk2 pre-provisioned)" if pre else ""),
                                    "link": f"{mbps} Mbit/s, RTT {rtt} ms", "wire bytes": wb["client_to_server"] + wb["server_to_client"],
                                    "transfer+RTT ms": f"{tr:.2f}", "measured CPU ms (python ML-KEM)": f"{cpu*1e3:.1f}",
                                    "total ms": f"{tot:.1f}", "<= 10 ms": tot <= 10.0,
                                    "left for compute within 10 ms": f"{10.0 - tr:.2f}"})
    # bandwidth needed for the McEliece public key alone to fit in the 10 ms budget
    pk = mcs.MCELIECE_SIZES["pk"]
    need = pk * 8 / (10e-3) / 1e9
    note = pd.DataFrame([{"quantity": "McEliece6688128 public key (bytes)", "value": f"{pk:,}"},
                         {"quantity": "link rate at which this key alone takes 10 ms (Gbit/s)", "value": f"{need:.2f}"},
                         {"quantity": "same key over 100 Mbit/s (ms)", "value": f"{pk*8/100e6*1e3:.0f}"},
                         {"quantity": "ML-KEM-1024 pk + ct (bytes)", "value": f"{1568+1568:,}"}])
    return pd.DataFrame(comp_rows), pd.DataFrame(budget_rows), note


# ==========================================================================
def c4_policy(cfg):
    rng = np.random.default_rng(1)
    n = cfg["policy_steps"]
    rows = []
    def rho_stream(kind):
        if kind == "benign":
            return rng.beta(1, 12, n)
        if kind == "attack burst":
            r = rng.beta(1, 12, n); on = np.zeros(n, bool); on[n // 3: n // 3 + n // 6] = True
            r[on] = rng.beta(8, 2, on.sum()); return r
        if kind == "evasive (rho held just under gamma_1)":
            return np.clip(rng.normal(0.28, 0.02, n), 0, 1)
        if kind == "oscillating around gamma_1":
            return np.clip(0.3 + 0.25 * np.sin(np.arange(n) / 3.0), 0, 1)
        if kind == "saturated (rho = 1)":
            return np.ones(n)
    for kind in ["benign", "attack burst", "evasive (rho held just under gamma_1)", "oscillating around gamma_1", "saturated (rho = 1)"]:
        r = rho_stream(kind)
        for mode in ["fixed thresholds", "adaptive thresholds"]:
            th = mcs.AdaptiveThresholds()
            levels = []
            for x in r:
                g1, g2 = (0.3, 0.7) if mode == "fixed thresholds" else th.update(x)
                levels.append(mcs.select_level(x, g1, g2))
            levels = np.array(levels)
            sw = int(np.sum(levels[1:] != levels[:-1]))
            row = {"threat stream": kind, "thresholds": mode, "min level (floor Pi_0)": int(levels.min()),
                   "time at Pi_0/Pi_1/Pi_2 (%)": "/".join(f"{100*np.mean(levels == l):.0f}" for l in (0, 1, 2)),
                   "level switches": sw}
            if kind == "attack burst":
                on0 = n // 3
                hit = np.nonzero(levels[on0:] == 2)[0]
                row["steps to Pi_2 after onset"] = int(hit[0]) if len(hit) else "never"
            rows.append(row)
    df = pd.DataFrame(rows)
    # cost of each level: handshake bytes
    return df


# ==========================================================================
def c5_causal(cfg):
    rows = []
    for name, spec in cce.SCENARIOS.items():
        scm = cce.attack_scm(spec["p_intent"], spec["p_vec_mal"], spec["p_vec_ben"], spec["p_comp"])
        obs = scm.p_obs(("Y", 1), ("X", 1)); do = scm.p_do(("Y", 1), ("X", 1)); adj = scm.p_do_adjust(("Y", 1), ("X", 1), "intent")
        cf = scm.counterfactual(("Y", 1), ("X", 0), {"X": 1, "Y": 1}, "intent")
        acts, best, _ = cce.choose_hardening(spec)
        base = [a for a in acts if a[0] == "none"][0][1]
        rows.append({"scenario": name, "P(Y|X=1) observational": f"{obs:.3f}", "P(Y|do(X=1)) Eq.15": f"{do:.3f}",
                     "Eq.15 by adjustment": f"{adj:.3f}", "confounding bias": f"{obs-do:+.3f}",
                     "P(Y_x'=0 | X=1,Y=1) counterfactual": f"{cf:.3f}", "baseline risk": f"{base:.3f}",
                     "chosen action P*": best[0], "risk after P*": f"{best[1]:.3f}",
                     "risk reduction": f"{100*(1-best[1]/base):.0f}%"})
    main = pd.DataFrame(rows)
    # confounding sweep: strength = P(X=1|malicious) - P(X=1|benign)
    sweep = []
    spec = cce.SCENARIOS["protocol downgrade (TLS1.3->1.2 via ECS)"]
    for pm in [0.05, 0.25, 0.5, 0.75, 0.95]:
        scm = cce.attack_scm(spec["p_intent"], pm, spec["p_vec_ben"], spec["p_comp"])
        o, d = scm.p_obs(("Y", 1), ("X", 1)), scm.p_do(("Y", 1), ("X", 1))
        sweep.append({"P(X=1|intent=1)": pm, "P(Y|X=1)": f"{o:.3f}", "P(Y|do(X=1))": f"{d:.3f}",
                      "naive overestimate": f"{(o/d-1)*100:+.0f}%"})
    actions = []
    for name, spec in cce.SCENARIOS.items():
        acts, best, _ = cce.choose_hardening(spec)
        for a, risk, reward in acts:
            actions.append({"scenario": name, "action": a, "P(compromise)": f"{risk:.4f}", "expected reward": f"{reward:.2f}",
                            "chosen": a == best[0]})
    return main, pd.DataFrame(sweep), pd.DataFrame(actions)


# ==========================================================================
def c6_se3(cfg):
    N = cfg["se3_steps"]
    r = se3.Rerandomizer(seed=3)
    keys, states = [], []
    t0 = time.perf_counter()
    for t in range(N):
        r.step(0.4, 0.5, 1, 0.3)
        keys.append(r.key()); states.append(r.K.copy())
    step_us = (time.perf_counter() - t0) / N * 1e6
    bits = np.unpackbits(np.frombuffer(b"".join(keys), np.uint8))
    n = len(bits); n1 = int(bits.sum())
    z = (n1 - n / 2) / np.sqrt(n / 4)
    pi = n1 / n; runs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    zr = (runs - (2 * n * pi * (1 - pi) + 1)) / (2 * np.sqrt(2 * n) * pi * (1 - pi))
    chi = stats.chisquare(np.bincount(np.frombuffer(b"".join(keys), np.uint8), minlength=256))
    # successive-key correlation
    arr = np.unpackbits(np.frombuffer(b"".join(keys), np.uint8)).reshape(N, 256).astype(float)
    corr = np.mean([abs(np.corrcoef(arr[:-1, j], arr[1:, j])[0, 1]) for j in range(0, 256, 8)])
    # avalanche: perturb the control input rho by delta and compare keys (state is quantised at 2^-20,
    # so perturbations far below that never reach the key; reported rather than hidden)
    def av_for(delta):
        out_ = []
        for s_ in range(cfg["avalanche"]):
            a, b = se3.Rerandomizer(seed=s_), se3.Rerandomizer(seed=s_)
            for _ in range(5):
                a.rng = np.random.default_rng(100 + s_); b.rng = np.random.default_rng(100 + s_)
                a.step(0.4, 0.5, 1, 0.3); b.step(0.4 + delta, 0.5, 1, 0.3)
            out_.append(bin(int.from_bytes(a.key(), "big") ^ int.from_bytes(b.key(), "big")).count("1"))
        return out_
    av = {d: av_for(d) for d in (1e-9, 1e-6, 1e-3)}
    # state distinctness
    S = np.array([x.ravel() for x in states[: min(N, 1500)]])
    from scipy.spatial.distance import pdist
    dmin = float(pdist(S).min())
    main = pd.DataFrame([
        {"check": f"group closure: ||R^T R - I||_F after {N} steps", "result": f"{r.group_error():.2e}"},
        {"check": f"monobit z over {n:,} key bits", "result": f"{z:+.2f}"},
        {"check": "runs test z", "result": f"{zr:+.2f}"},
        {"check": "byte chi-square p-value", "result": f"{chi.pvalue:.3f}"},
        {"check": "mean |corr| of successive keys (32 bit positions)", "result": f"{corr:.4f}"},
        *[{"check": f"avalanche: bit flips for a {d:g} change in rho ({cfg['avalanche']} trials)", "result": f"{np.mean(v):.1f} of 256 (sd {np.std(v, ddof=1):.1f})"} for d, v in av.items()],
        {"check": "minimum pairwise distance between visited states", "result": f"{dmin:.3e}"},
        {"check": "cost per re-randomisation step (expm 4x4 + KDF), us", "result": f"{step_us:.0f}"},
    ])
    pred = []
    for label, fn in [("sigma = 0.01 (floor)", lambda t: 0.01), ("sigma = 0.05", lambda t: 0.05), ("sigma = 0.2", lambda t: 0.2),
                      ("sigma = 0.6 (cap)", lambda t: 0.6), ("sigma = 0.2/(1+t) (annealed)", lambda t: 0.2 / (1 + t))]:
        pred.append({"diffusion schedule": label, "next-state prediction error (Frobenius, last 100 steps)": f"{se3.predict_next_error(fn, n=400):.4f}"})
    return main, pd.DataFrame(pred)


# ==========================================================================
def c7_audit():
    rows = []
    for L, d in mcs.LEVELS.items():
        rows.append({"level": d["name"], "components": " + ".join(d["kems"]), "NIST category (thesis)": d["nist_category"],
                     "Eq. 12 statement": "lambda_PQ >= max(lambda_1, lambda_2)",
                     "what it covers": "key is secure if at least one component KEM is secure (IND-CCA combiner assumption)",
                     "does not cover": "implementation side channels; PQ strength of X25519 (none); the AEAD of the control channel"})
    return pd.DataFrame(rows)


# ==========================================================================
def md(df, title):
    return f"### {title}\n\n{df.to_markdown(index=False)}\n"


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/crypto")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = dict(kem_trials=200, reject_trials=100, combiner_trials=2000, bias_keys=4000, handshakes=15,
               policy_steps=600, se3_steps=4000, avalanche=100)
    if a.quick:
        cfg = dict(kem_trials=8, reject_trials=10, combiner_trials=100, bias_keys=300, handshakes=3,
                   policy_steps=120, se3_steps=300, avalanche=10)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rep = ["# MIRAGE-DoH cryptographic agility, policy and causal experiments (v2)\n",
           "Real: X25519, ChaCha20-Poly1305, SHAKE/SHA3. Educational: pure-Python ML-KEM (FIPS 203 structure, not KAT validated). "
           "Not implemented: Classic McEliece (size model only). Network times are modelled from link parameters.\n"]
    def emit(name, df, title):
        df.to_csv(out / f"{name}.csv", index=False); rep.append(md(df, title)); print(f"== {title}"); print(df.to_string(index=False)); print()
    emit("c1_mlkem", c1_kem(cfg), "C1 ML-KEM reference implementation")
    av, broken, tam = c2_combiner(cfg)
    emit("c2_combiner_avalanche", av, "C2a Hybrid combiner avalanche (Eq. 11)")
    emit("c2_combiner_broken_component", broken, f"C2b Combiner output when one KEM is fully compromised (AEAD tamper test: {tam})")
    comp, budget, note = c3_morph(cfg)
    emit("c3_morph_compute", comp, "C3a Per-level compute and wire bytes (measured CPU, modelled McEliece sizes)")
    emit("c3_morph_budget", budget, "C3b Morph time against the 10 ms target, by link")
    emit("c3_mceliece_pk", note, "C3c Classic McEliece public-key transfer")
    emit("c4_policy", c4_policy(cfg), "C4 Threat-level policy (Eq. 13) under benign, attack, evasive and oscillating threat scores")
    m, sw, ac = c5_causal(cfg)
    emit("c5_causal", m, "C5a Causal engine: observational vs interventional vs counterfactual (Eqs. 14-16)")
    emit("c5_confounding_sweep", sw, "C5b Bias of the naive estimate vs confounding strength")
    emit("c5_actions", ac, "C5c Hardening actions and expected reward")
    s3, pr = c6_se3(cfg)
    emit("c6_se3_checks", s3, "C6a SE(3) re-randomisation checks (Eq. 26)")
    emit("c6_se3_predictability", pr, "C6b Predictability of the next state vs diffusion schedule")
    emit("c7_audit", c7_audit(), "C7 Scope of the Eq. 12 security statement")
    rep.append(f"\nWall time {time.time()-t0:.0f}s.\n")
    (out / "crypto_report_v2.md").write_text("\n".join(rep))
    json.dump(cfg, open(out / "crypto_config_v2.json", "w"))


if __name__ == "__main__":
    main()
