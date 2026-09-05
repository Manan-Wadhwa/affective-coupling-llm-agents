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
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    for ax, (title, a, L, f) in zip(axes, panels):
        for src, est, lab, style, col in [(a, "logreg", "logistic, C = 0.5", "o-", OI[0]), (f, "logreg_cv", "logistic, tuned C", "s--", OI[1]),
                                     (f, "logreg_lam", "logistic, fixed per-sample penalty (anchor 600)", "^-", OI[3]), (a, "dom", "difference of means", "D-", OI[2])]:
            try:
                ns, m, ci = stab(src, L, est)
            except KeyError:
                continue
            lo = [c[0] for c in ci]; hi = [c[1] for c in ci]
            ax.plot(ns, m, style, label=lab, ms=4, color=col); ax.fill_between(ns, lo, hi, alpha=.15, color=col)
        ticks = [75, 150, 300, 600, 1200, 2000] if L != 21 else [75, 150, 300, 600, 1200]
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=8); ax.set_xlim(60, 2600 if L != 21 else 1500)
        ax.set_title(title, fontsize=10); ax.set_xlabel("n per half"); ax.grid(alpha=.3)
    axes[0].set_ylabel("split-half cosine (10 disjoint splits)"); axes[0].set_ylim(0.2, 1.0); axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_splithalf_vs_n.png", dpi=180); fig.savefig(f"{OUT}/fig1_splithalf_vs_n.pdf"); plt.close(fig)


def fig2():
    m27, m8 = J("a2_margin_qwen36-27b-l43-v2.json")["by_layer"]["43"], J("a2_margin_llama3-abl-l21-v2.json")["by_layer"]["21"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (title, b) in zip(axes, [("Qwen3.6-27B, layer 43", m27), ("Llama-3-8B-abl, layer 21", m8)]):
        ns = sorted(int(n) for n in b)
        for key, lab, style, col in [("sh_lam", "split-half: logistic, fixed per-sample penalty (anchor 600)", "^-", OI[3]), ("sh_weak2", "split-half: logistic, C = 500", "v-", OI[4]),
                                ("sh_dom", "split-half: difference of means", "D-", OI[2]), ("cos_lam_dom", "cos(logistic, difference of means), same half", "o:", OI[5])]:
            m = [b[str(n)][key]["mean"] for n in ns]; ci = [b[str(n)][key]["ci"] for n in ns]
            ax.plot(ns, m, style, label=lab, ms=4, color=col); ax.fill_between(ns, [c[0] for c in ci], [c[1] for c in ci], alpha=.15, color=col)
        acc = min(b[str(n)]["lam_train_acc"]["min"] if "min" in b[str(n)]["lam_train_acc"] else b[str(n)]["lam_train_acc"]["mean"] for n in ns)
        ax.text(0.02, 0.95, f"training accuracy = {acc:.3f} at every n (all fits separate their half)", transform=ax.transAxes, fontsize=7.5, va="top")
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns, fontsize=8); ax.set_ylim(0.2, 1.02); ax.set_title(title, fontsize=10); ax.set_xlabel("n per half"); ax.grid(alpha=.3)
    axes[0].set_ylabel("cosine"); h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{OUT}/fig2_margin_divergence.png", dpi=180); fig.savefig(f"{OUT}/fig2_margin_divergence.pdf"); plt.close(fig)


def fig3():
    d = J("b1c_alllayer_qwen36-27b.json"); S = d["summary"]; doses = [0.0, 0.33, 0.67, 1.0]
    R = J("paper_revision_stats.json")["b1c_27b"]["per_emotion"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    for ax, e in zip(axes.ravel(), EMOS):
        for arm, lab, style, col in [("none", "none", "-o", "k"), ("emo13_42", "emotion direction, hs 13–42", "-s", OI[0]), ("emo_all", "emotion direction, hs 13–63", "-^", OI[5]), ("rand_all", "random direction, hs 13–63", "--D", OI[3])]:
            bd = R[e]["bands"][arm]; m = [bd[str(x)]["mean"] for x in doses]; lo = [bd[str(x)]["ci"][0] for x in doses]; hi = [bd[str(x)]["ci"][1] for x in doses]
            ax.plot(doses, m, style, ms=4, label=lab, color=col); ax.fill_between(doses, lo, hi, alpha=.12, color=col)
        sl = S[e]["none"]["present_slope"]; sh = R[e]["shape"]
        ax.set_title(f"{e}: slope {sl['slope']:+.0f} [{sl['ci'][0]:+.0f}, {sl['ci'][1]:+.0f}]; steps {sh['step_0_to_033']['diff']:+.0f}, {sh['step_067_to_1']['diff']:+.0f}", fontsize=8.5); ax.grid(alpha=.3)
    for ax in axes[1]: ax.set_xlabel("steering dose on A")
    for ax in axes[:, 0]: ax.set_ylabel("B's present-emotion projection")
    axes[0, 1].legend(fontsize=7.5, loc="lower right"); fig.suptitle("B1c: B's dose-response under ablation of A's span (29 scenarios × 3 reps; bands: scenario-bootstrap 95% CIs; steps: paired 0→0.33 and 0.67→1 changes)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_b1c_dose_response.png", dpi=180); fig.savefig(f"{OUT}/fig3_b1c_dose_response.pdf"); plt.close(fig)


def fig4(extra=None):
    c, dd = J("b1c_alllayer_qwen36-27b.json")["summary"], J("b1d_subspace_qwen36-27b.json")["summary"]
    R = J("paper_revision_stats.json")
    rows = [("B1c: emotion dir − none (rank 1)", "b1c_27b", "emo_all"), ("B1c: random dir − none (rank 1)", "b1c_27b", "rand_all"),
            ("B1d: affect subspace − none (rank 5)", "b1d_27b", "sub_all"), ("B1d: permuted-label subspace − none (rank 5)", "b1d_27b", "perm_all"),
            ("B1d: random 5-frame − none (rank 5)", "b1d_27b", "randsub_all"), ("B1e: permuted-label dir − none (rank 1)", "b1e_27b", "permdir_all"),
            ("B1e: top principal dir − none (rank 1)", "b1e_27b", "pc1_all")]
    fig, ax = plt.subplots(figsize=(11, 3.6)); w = 0.86 / len(rows); x = np.arange(len(EMOS))
    for i, (lab, tag, arm) in enumerate(rows):
        pts, lo, hi, marks = [], [], [], []
        for em in EMOS:
            en = R[tag]["per_emotion"][em]; con = en["arm_vs_none"][arm]; p = con["blocked_fraction"]; r = con["blocked_range"]
            pts.append(p); lo.append(p - r[0]); hi.append(r[1] - p)
            q = R[tag]["bh_q_over_testable"].get(arm, {}).get(em); marks.append("*" if (q is not None and q < 0.05) else ("·" if con["sig"] else ""))
        xs = x + (i - len(rows) / 2 + 0.5) * w
        ax.bar(xs, [min(v, 1.75) for v in pts], w, yerr=[lo, hi], capsize=1.5, label=lab, alpha=.9, error_kw={"lw": .8})
        for xx, v, mk in zip(xs, pts, marks):
            if mk: ax.text(xx, min(v, 1.75) + (0.05 if v >= 0 else -0.12), mk, ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([f"{e}{' (untestable)' if e == 'calm' else ''}" for e in EMOS], fontsize=8)
    ax.set_ylabel("blocked fraction of the none-arm slope"); ax.set_ylim(-0.7, 1.9); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.text(0.99, 0.97, "* BH-FDR q < 0.05 over the five testable emotions\n· unadjusted CI excludes zero only\nbars clipped at 1.75 (calm)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_blocked_fractions.png", dpi=180); fig.savefig(f"{OUT}/fig4_blocked_fractions.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(sys.argv[1] if len(sys.argv) > 1 else None); print("FIGS_DONE")
