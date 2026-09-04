#!/usr/bin/env python3
"""Regenerate A2's cached activations from a saved pool, without regenerating the pool.

The 27B pool (5760 dialogues) survived the 2026-09-04 lease expiry because it was pulled;
the 441 MB feature cache did not. `generate_pool` resumes with every generation cached, so
this is one forward pass over the kept dialogues -- no sampling, no new text -- and the npz
it writes has the layout `pool_features` produces (n, yp, yo, L<hs>...), which
`a2_followup.py --feats` reads.

Usage: python a2_regen_feats.py --model Qwen/Qwen3.6-27B --tag qwen36-27b --k 160 --layers 16,43,54
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _c in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")):
    if os.path.exists(os.path.join(_c, "acl_core.py")):
        sys.path.insert(0, _c); break
import acl_core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=160)
    ap.add_argument("--layers", default="16,43,54")
    ap.add_argument("--workdir", default="/marimo/work")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-bs", type=int, default=16)
    a = ap.parse_args()
    cache = os.path.join(a.workdir, f"pool_{a.tag}.jsonl")
    if not os.path.exists(cache):
        sys.exit(f"pool cache {cache} missing -- upload it first; this script never generates")
    h = C.load(a.model)
    items = C.generate_pool(h, a.k, cache, seed=a.seed)
    hs = [int(x) for x in a.layers.split(",")]
    out = os.path.join(a.workdir, f"feats_{a.tag}.npz")
    feats, yp, yo = C.pool_features(h, items, hs, cache=out, bs=a.pool_bs)
    print(f"[regen] {out}: n={len(items)} layers={sorted(feats)} shapes={[feats[l].shape for l in sorted(feats)]}",
          flush=True)
    print("REGEN_DONE", flush=True)


if __name__ == "__main__":
    main()
