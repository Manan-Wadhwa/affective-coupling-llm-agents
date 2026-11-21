"""paper_figures_cr.py -- camera-ready figures (IAB @ NeurIPS 2026), from the same result JSONs.

Same numbers as paper_figures.py; only layout, size and labels change, so that every
figure is legible at the printed width of the NeurIPS template (5.5 in). Figure 2 merges
the old Figures 1 and 2 (split-half reproducibility and the logistic direction's drift);
Figure 4 adds the footprint-matched random 23-frame and states each arm's rank and
footprint in its legend. Output: docs/writeups/camera_ready/figs/*.pdf (+ .png).
Run from the repository root:  python src/rev3/paper_figures_cr.py
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/rev3"; OUT = "docs/writeups/camera_ready/figs"; os.makedirs(OUT, exist_ok=True)
J = lambda f: json.load(open(os.path.join(R, f)))
EMOS = ["desperate", "afraid", "happy", "calm", "sad", "angry"]
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "pink": "#CC79A7",
      "sky": "#56B4E9", "verm": "#D55E00", "grey": "#7f7f7f", "black": "#000000"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "lines.markersize": 4, "lines.linewidth": 1.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def stab(d, L, est):
    st = d["by_layer"][str(L)]["stability"][est]
    ns = sorted(int(n) for n in st)
    get = lambda n, k: st[str(n)]["raw"][k] if isinstance(st[str(n)].get("raw"), dict) else st[str(n)][k]
    return ns, [get(n, "mean") for n in ns], [get(n, "ci") for n in ns]


def fig_estimator():
    a27, a8 = J("a2_estimator_qwen36-27b.json"), J("a2_estimator_llama3-abl.json")
    f27, f8 = J("a2_followup_qwen36-27b.json"), J("a2_followup_llama3-abl.json")
    m27 = J("a2_margin_qwen36-27b-l43-v2.json")["by_layer"]["43"]
    m8 = J("a2_margin_llama3-abl-l21-v2.json")["by_layer"]["21"]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    series = [("a", "logreg", "logistic regression, C = 0.5", "o-", OI["blue"]),
              ("f", "logreg_cv", "logistic regression, C tuned by cross-validation", "s--", OI["sky"]),
              ("f", "logreg_lam", "logistic regression, fixed per-sample penalty", "^:", OI["pink"]),
              ("a", "dom", "difference of means", "D-", OI["green"])]
    for ax, (title, a, f, L, ticks) in zip(axes[:2], [
            ("(a) Qwen3.6-27B, layer 43", a27, f27, 43, [75, 300, 1200]),
            ("(b) Llama-3-8B-abl., layer 21", a8, f8, 21, [75, 300, 1200])]):
        for src, est, lab, sty, col in series:
            try:
                ns, m, ci = stab(a if src == "a" else f, L, est)
            except KeyError:
                continue
            ax.plot(ns, m, sty, label=lab, color=col)
            ax.fill_between(ns, [c[0] for c in ci], [c[1] for c in ci], alpha=.18, color=col)
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.minorticks_off()
        ax.set_ylim(0.2, 1.02); ax.set_title(title, loc="left"); ax.set_xlabel("items per half, n")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("agreement between halves\n(cosine)")
    ax = axes[2]
    for b, lab, col, sty in [(m27, "27B", OI["verm"], "o-"), (m8, "8B", OI["orange"], "s--")]:
        ns = sorted(int(n) for n in b)
        m = [b[str(n)]["cos_lam_dom"]["mean"] for n in ns]; ci = [b[str(n)]["cos_lam_dom"]["ci"] for n in ns]
        ax.plot(ns, m, sty, label=lab, color=col)
        ax.fill_between(ns, [c[0] for c in ci], [c[1] for c in ci], alpha=.18, color=col)
        assert min(b[str(n)]["lam_train_acc"]["min"] for n in ns) == 1.0
    ax.set_xscale("log"); ax.set_xticks([75, 300, 1200]); ax.set_xticklabels([75, 300, 1200]); ax.minorticks_off()
    ax.set_ylim(0.2, 1.02); ax.grid(alpha=.3); ax.set_xlabel("items per half, n")
    ax.set_title("(c) logistic vs. diff. of means", loc="left")
    ax.set_ylabel("cosine, same half")
    ax.text(0.04, 0.04, "training accuracy = 1.000\nin every fit", transform=ax.transAxes, fontsize=8.3, style="italic")
    ax.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.2), fontsize=8.5, handlelength=1.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=9, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    save(fig, "fig_estimator")


def fig_dose():
    d = J("b1c_alllayer_qwen36-27b.json"); S = d["summary"]; doses = [0.0, 0.33, 0.67, 1.0]
    R2 = J("paper_revision_stats.json")["b1c_27b"]["per_emotion"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.45), sharex=True)
    arms = [("none", "no ablation", "-o", OI["black"]),
            ("emo13_42", "emotion direction removed, layers 13–42", "-s", OI["blue"]),
            ("emo_all", "emotion direction removed, layers 13–63", "-^", OI["verm"]),
            ("rand_all", "random direction removed, layers 13–63", "--D", OI["grey"])]
    for ax, e in zip(axes.ravel(), EMOS):
        for arm, lab, sty, col in arms:
            bd = R2[e]["bands"][arm]
            m = [bd[str(x)]["mean"] for x in doses]
            ax.plot(doses, m, sty, label=lab, color=col)
            if arm == "none":   # band for the baseline only; between-arm tests are paired (Section 5)
                ax.fill_between(doses, [bd[str(x)]["ci"][0] for x in doses], [bd[str(x)]["ci"][1] for x in doses], alpha=.13, color=col)
        sl = S[e]["none"]["present_slope"]
        tag = " (untestable)" if e == "calm" else ""
        ax.set_title(f"{e}{tag}: {sl['slope']:+.0f} [{sl['ci'][0]:+.0f}, {sl['ci'][1]:+.0f}]", loc="left", fontsize=9.5)
        ax.grid(alpha=.3); ax.set_xticks(doses); ax.set_xticklabels(["0", ".33", ".67", "1"])
    for ax in axes[1]: ax.set_xlabel("dose on A")
    for ax in axes[:, 0]: ax.set_ylabel("B's projection")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=9, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=0.4, w_pad=0.5)
    save(fig, "fig_dose")


def fig_blocked():
    S = J("paper_revision_stats.json"); K = J("b1d_randk23_summary.json")
    rows = [  # (label, source, arm, colour, hatch)
        ("emotion direction (rank 1; 1×)", "b1c_27b", "emo_all", OI["blue"], ""),
        ("affect subspace (rank 5; 11.6×)", "b1d_27b", "sub_all", OI["sky"], ""),
        ("permuted-label direction (rank 1; 0.85×)", "b1e_27b", "permdir_all", OI["orange"], "//"),
        ("permuted-label subspace (rank 5; 11.2×)", "b1d_27b", "perm_all", OI["verm"], "//"),
        ("random direction (rank 1; 2.0×)", "b1c_27b", "rand_all", "#bdbdbd", ""),
        ("random 5-frame (rank 5; 5.4×)", "b1d_27b", "randsub_all", "#969696", ""),
        ("random 23-frame (rank 23; 11.4×)", "k23", None, "#636363", ""),
        ("top principal direction (rank 1; 5.8×)", "b1e_27b", "pc1_all", OI["green"], ".."),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 2.6)); w = 0.86 / len(rows); x = np.arange(len(EMOS))
    CLIP = 1.0
    for i, (lab, src, arm, col, hat) in enumerate(rows):
        pts, lo, hi, marks = [], [], [], []
        for em in EMOS:
            if src == "k23":
                con = K["per_emotion"][em]["randsub_vs_none"]; q = K["bh_q_over_testable"].get(em)
            else:
                con = S[src]["per_emotion"][em]["arm_vs_none"][arm]; q = S[src]["bh_q_over_testable"].get(arm, {}).get(em)
            p = con["blocked_fraction"]; r = con["blocked_range"]
            pts.append(p); lo.append(max(p - r[0], 0)); hi.append(max(r[1] - p, 0))
            marks.append("*" if (q is not None and q < 0.05 and em != "calm") else ("†" if con["sig"] else ""))
        xs = x + (i - len(rows) / 2 + 0.5) * w
        shown = [min(v, CLIP) for v in pts]
        ax.bar(xs, shown, w, color=col, hatch=hat, edgecolor="white" if not hat else "#333333", lw=0.4, label=lab)
        ax.errorbar(xs, shown, yerr=[[min(l, s + 0.6) for l, s in zip(lo, shown)], [min(h, CLIP + 0.1 - s) for h, s in zip(hi, shown)]],
                    fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.0)
        for xx, v, mk in zip(xs, pts, marks):
            if mk:
                ax.text(xx, min(v, CLIP) + 0.03 if v >= 0 else v - 0.12, mk, ha="center", va="bottom",
                        fontsize=10 if mk == "*" else 8, fontweight="bold")
    ax.axhline(0, color="k", lw=.9)
    ax.axvspan(2.5, 3.5, color="#f0f0f0", zorder=-1)
    ax.set_xticks(x); ax.set_xticklabels([e if e != "calm" else "calm\n(untestable)" for e in EMOS])
    ax.set_ylabel("blocked fraction\nof B's slope"); ax.set_ylim(-0.62, 1.18); ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.3),
              columnspacing=1.5, handlelength=1.5, handleheight=0.9, labelspacing=0.25)
    ax.text(0.005, 0.985, "*  BH-FDR q < 0.05 (five testable emotions)\n†  unadjusted 95% CI excludes zero",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig_blocked")


if __name__ == "__main__":
    fig_estimator(); fig_dose(); fig_blocked(); print("FIGS_DONE")
