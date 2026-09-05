"""paper_figures.py -- figures for the rev-3 deadline paper, straight from the result JSONs.

Every number plotted is the same number the registry points at; no recomputation beyond
reading means/CIs. Output: docs/writeups/figs/*.png (+ .pdf).
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
plt.rcParams["axes.prop_cycle"] = cycler(color=OI)
R = "results/rev3"; OUT = "docs/writeups/figs"; os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(os.path.join(R, f)))
EMOS = ["desperate", "afraid", "happy", "calm", "sad", "angry"]


def stab(d, L, est, space="raw"):
    st = d["by_layer"][str(L)]["stability"][est]
    ns = sorted(int(n) for n in st); return ns, [st[str(n)][space]["mean"] if isinstance(st[str(n)].get(space), dict) else st[str(n)]["mean"] for n in ns], \
        [st[str(n)][space]["ci"] if isinstance(st[str(n)].get(space), dict) else st[str(n)]["ci"] for n in ns]


def fig1():
    a27, a8 = J("a2_estimator_qwen36-27b.json"), J("a2_estimator_llama3-abl.json")
    f27, f54, f8 = J("a2_followup_qwen36-27b.json"), J("a2_followup_qwen36-27b-l54.json"), J("a2_followup_llama3-abl.json")
    panels = [("Qwen3.6-27B, layer 43 (focus)", a27, 43, f27), ("Qwen3.6-27B, layer 54", a27, 54, f54), ("Llama-3-8B-abl, layer 21 (focus)", a8, 21, f8)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (title, a, L, f) in zip(axes, panels):
        for src, est, lab, style in [(a, "logreg", "logistic, C = 0.5", "o-"), (f, "logreg_cv", "logistic, tuned C", "s--"),
                                     (f, "logreg_lam", "logistic, fixed per-sample penalty", "^-"), (a, "dom", "difference of means", "D-")]:
            try:
                ns, m, ci = stab(src, L, est)
            except KeyError:
                continue
            lo = [c[0] for c in ci]; hi = [c[1] for c in ci]
            ax.plot(ns, m, style, label=lab, ms=4); ax.fill_between(ns, lo, hi, alpha=.15)
        ax.set_xscale("log"); ax.set_xticks([75, 150, 300, 600, 1200, 2000]); ax.set_xticklabels([75, 150, 300, 600, 1200, 2000], fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("n per half"); ax.grid(alpha=.3)
    axes[0].set_ylabel("split-half cosine (10 disjoint splits)"); axes[0].set_ylim(0.2, 1.0); axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_splithalf_vs_n.png", dpi=180); fig.savefig(f"{OUT}/fig1_splithalf_vs_n.pdf"); plt.close(fig)


def fig2():
    m27, m8 = J("a2_margin_qwen36-27b-l43-v2.json")["by_layer"]["43"], J("a2_margin_llama3-abl-l21-v2.json")["by_layer"]["21"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (title, b) in zip(axes, [("Qwen3.6-27B, layer 43", m27), ("Llama-3-8B-abl, layer 21", m8)]):
        ns = sorted(int(n) for n in b)
        for key, lab, style in [("sh_lam", "split-half: logistic, fixed penalty", "^-"), ("sh_weak2", "split-half: logistic, C = 500", "v-"),
                                ("sh_dom", "split-half: difference of means", "D-"), ("cos_lam_dom", "cos(logistic, difference of means), same half", "o:")]:
            m = [b[str(n)][key]["mean"] for n in ns]; ci = [b[str(n)][key]["ci"] for n in ns]
            ax.plot(ns, m, style, label=lab, ms=4); ax.fill_between(ns, [c[0] for c in ci], [c[1] for c in ci], alpha=.15)
        acc = min(b[str(n)]["lam_train_acc"]["min"] if "min" in b[str(n)]["lam_train_acc"] else b[str(n)]["lam_train_acc"]["mean"] for n in ns)
        ax.text(0.02, 0.95, f"training accuracy = {acc:.3f} at every n (all fits separate their half)", transform=ax.transAxes, fontsize=7.5, va="top")
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns, fontsize=8); ax.set_ylim(0.2, 1.02); ax.set_title(title, fontsize=10); ax.set_xlabel("n per half"); ax.grid(alpha=.3)
    axes[0].set_ylabel("cosine"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{OUT}/fig2_margin_divergence.png", dpi=180); fig.savefig(f"{OUT}/fig2_margin_divergence.pdf"); plt.close(fig)


def fig3():
    d = J("b1c_alllayer_qwen36-27b.json"); S = d["summary"]; doses = [0.0, 0.33, 0.67, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, e in zip(axes.ravel(), EMOS):
        for arm, lab, style in [("none", "none", "k-o"), ("emo13_42", "emotion direction, hs 13–42", "C0-s"), ("emo_all", "emotion direction, hs 13–63", "C5-^"), ("rand_all", "random direction, hs 13–63", "C3--D")]:
            mb = S[e][arm]["mean_by_dose"]; ax.plot(doses, [mb[str(x)] for x in doses], style, ms=4, label=lab)
        sl = S[e]["none"]["present_slope"]; ax.set_title(f"{e}  (none slope {sl['slope']:+.0f} [{sl['ci'][0]:+.0f}, {sl['ci'][1]:+.0f}])", fontsize=9); ax.grid(alpha=.3)
    for ax in axes[1]: ax.set_xlabel("steering dose on A")
    for ax in axes[:, 0]: ax.set_ylabel("B's present-emotion projection")
    axes[0, 0].legend(fontsize=7.5); fig.suptitle("B1c: B's dose-response under all-layer ablation of A's span (29 scenarios × 3 reps)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_b1c_dose_response.png", dpi=180); fig.savefig(f"{OUT}/fig3_b1c_dose_response.pdf"); plt.close(fig)


def fig4(extra=None):
    c, dd = J("b1c_alllayer_qwen36-27b.json")["summary"], J("b1d_subspace_qwen36-27b.json")["summary"]
    rows = [("B1c: emotion dir − random dir (rank 1)", c, "emo_all_vs_rand_all"), ("B1d: affect subspace − permuted subspace (rank 5)", dd, "sub_all_vs_perm_all"),
            ("B1d: permuted subspace − none (rank 5, no emotion info)", dd, "perm_all_vs_none")]
    if extra is not None:
        e = J(extra)["summary"]
        rows += [("B1e: emotion dir − permuted-label dir (rank 1)", e, "emo_all_vs_permdir_all"), ("B1e: permuted-label dir − none (rank 1)", e, "permdir_all_vs_none")]
    fig, ax = plt.subplots(figsize=(10, 3.8)); w = 0.8 / len(rows); x = np.arange(len(EMOS))
    for i, (lab, S, key) in enumerate(rows):
        pts, lo, hi = [], [], []
        for em in EMOS:
            bf = S[em].get("blocked_fraction", {}).get(key)
            if bf is None or bf.get("point") is None:
                con = S[em].get(key); ns = S[em]["none"]["present_slope"]["slope"]
                if con is None: pts.append(np.nan); lo.append(0); hi.append(0); continue
                p = -con["diff"] / ns; r = sorted(-cc / ns for cc in con["ci"]); pts.append(p); lo.append(p - r[0]); hi.append(r[1] - p)
            else:
                p = bf["point"]; r = bf["range"]; pts.append(p); lo.append(p - r[0]); hi.append(r[1] - p)
        ax.bar(x + (i - len(rows) / 2 + 0.5) * w, pts, w, yerr=[lo, hi], capsize=2, label=lab, alpha=.85)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([f"{e}{' (untestable)' if e == 'calm' else ''}" for e in EMOS], fontsize=8)
    ax.set_ylabel("blocked fraction of the none-arm slope"); ax.set_ylim(-0.6, 1.8); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_blocked_fractions.png", dpi=180); fig.savefig(f"{OUT}/fig4_blocked_fractions.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(sys.argv[1] if len(sys.argv) > 1 else None); print("FIGS_DONE")
