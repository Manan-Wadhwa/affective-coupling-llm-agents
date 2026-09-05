"""b1d_frame_geometry.py -- post-hoc geometry of the B1d frames (asked for by report 27's critic).

Run on the box that holds the cached probe features (probefeat_qwen36-27b.npz, B1c/B1d's pool,
n = 1615), with B1d's DIR split (np.random.default_rng(0).permutation(N)[:N//2]) and B1d's
perm_seed rule. For hidden states 13, 30, 43, 63: the rank-5 class-mean frame (true), the
permuted-label frames for (afraid, sad, desperate) x reps 0-2, the top-5 and top-20 principal
components of the centred DIR half; reports mean principal-angle cosines between frames and the
share of DIR-half variance each frame captures (a random 5-frame for scale).
Output: b1d_frame_geometry.json (no provenance stamp; descriptive, post hoc, not pre-registered).
"""
import numpy as np, zlib, json, sys
EMOTIONS = ["happy", "calm", "sad", "angry", "afraid", "desperate"]


def frame(X, y, k=5):
    cls = sorted(set(y.tolist())); M = np.stack([X[y == c].mean(0) for c in cls]); M = M - M.mean(0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False); return Vt[:k].T


def pcs(X, k=5):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); return Vt[:k].T, S


def pcos(A, B):
    return float(np.mean(np.linalg.svd(A.T @ B, compute_uv=False)))


def perm_seed(seed, e, rep):
    return int(zlib.crc32(f"perm|{seed}|{EMOTIONS.index(e)}|{rep}".encode()) & 0x7FFFFFFF)


def main(path="/marimo/results/probefeat_qwen36-27b.npz", out="/marimo/results/b1d_frame_geometry.json"):
    z = np.load(path); yp = np.asarray(z["yp"]).astype(int); N = len(yp)
    DIR = np.random.default_rng(0).permutation(N)[:N // 2]
    res = {}
    for L in (13, 30, 43, 63):
        X = np.asarray(z[f"L{L}"], dtype=np.float64); Xd = X[DIR]; yd = yp[DIR]
        Xc = Xd - Xd.mean(0); tot = Xc.var(0).sum()
        var = lambda Q: float((Xc @ Q).var(0).sum() / tot)
        Qt = frame(Xd, yd); P5, S = pcs(Xd); P20, _ = pcs(Xd, 20)
        rows = []
        for e in ("afraid", "sad", "desperate"):
            for rep in range(3):
                Qp = frame(Xd, np.random.default_rng(perm_seed(0, e, rep)).permutation(yd))
                rows.append((pcos(Qp, P5), pcos(Qp, P20), pcos(Qp, Qt), var(Qp)))
        rows = np.array(rows)
        res[L] = {"true_vs_pc5": pcos(Qt, P5), "true_vs_pc20": pcos(Qt, P20),
                  "perm_vs_pc5": float(rows[:, 0].mean()), "perm_vs_pc20": float(rows[:, 1].mean()),
                  "perm_vs_true": float(rows[:, 2].mean()), "var_true": var(Qt),
                  "var_perm": float(rows[:, 3].mean()), "var_pc5": var(P5),
                  "var_rand5": float(np.mean([var(np.linalg.qr(np.random.default_rng(s).standard_normal((X.shape[1], 5)))[0]) for s in range(3)])),
                  "top5_var_share": float((S[:5] ** 2).sum() / (S ** 2).sum())}
        print(L, res[L], flush=True)
    json.dump(res, open(out, "w"), indent=1); print("GEOM_DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
