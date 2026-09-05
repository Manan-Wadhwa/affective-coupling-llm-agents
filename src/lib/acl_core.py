"""acl_core — single vendored core for the affective-coupling experiments (rev 3 rebuild).

One file, no repo-relative imports, so it can be dropped into a bare GPU sandbox that has
no checkout (`molab`/`marimo` leases) and still produce results that are byte-comparable
with a local run. It replaces the ad-hoc duplication between `coupling_e0.py`,
`coupling_e2.py`, `stability_diag.py`, `dual_estimator_battery.py` and
`notebooks/p1dose_notebook.py`.

What is deliberately carried over unchanged from the published code, so old and new
numbers stay comparable: EMOTIONS, NAMES, TOPICS, LEAK_WORDS, dialogue_prompt,
parse_final_A, leak, SCENARIOS, the mean-pool-over-a-char-span feature, and the
`round(0.67 * n_layers)` focus convention.

What is new, and why (RESEARCH_PLAN.md rev 3):
  * §0.3/§0.5 provenance   -- `Provenance` stamps model revision sha, seeds, config, code
                              hash and per-sample values into every result file.
  * §2 Paper A             -- `fit_direction` implements eight estimator families behind
                              one interface; `split_half` and `diagnostics` are the A2
                              instruments.
  * §6 controls register   -- `Ablate` supports norm-matched random and orthogonal
                              controls, `Degeneracy` scores every dose, `readout_probe`
                              is a HELD-OUT probe so measurement is never circular, and
                              `frozen_token_audit` records which spans were generated
                              under an intervention.
  * §7 reporting standard  -- every aggregate helper returns n, CI and the per-sample
                              vector it was computed from.

Nothing in here reads or writes repo paths; callers pass explicit output paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# torch/transformers are imported lazily so the estimator + statistics half of this module
# can be unit-tested on a CPU box with no CUDA and no model weights.
try:
    import torch
except Exception:                                        # pragma: no cover
    torch = None


# ---------------------------------------------------------------------------
# 1. Stimulus constants — verbatim from coupling_e0 / coupling_e2
# ---------------------------------------------------------------------------

EMOTIONS = ["happy", "calm", "sad", "angry", "afraid", "desperate"]
NAMES = [("Alex", "Sam"), ("Jordan", "Riley"), ("Casey", "Morgan"),
         ("Taylor", "Jamie"), ("Quinn", "Avery"), ("Drew", "Reese")]
TOPICS = ["planning a weekend trip", "a problem at work", "cooking dinner together",
          "fixing a broken car", "news about a mutual friend", "moving to a new city",
          "a project deadline", "choosing a place to live", "a phone call that just ended",
          "what to do this evening", "a decision about money", "a change of plans"]
LEAK_WORDS = {
    "happy": ["happy", "happiness", "happily", "happier"],
    "calm": ["calm", "calmly", "calmness"],
    "sad": ["sad", "sadness", "sadly", "sadder"],
    "angry": ["angry", "anger", "angrily", "angrier"],
    "afraid": ["afraid", "fear", "fearful", "scared", "terrified", "frightened"],
    "desperate": ["desperate", "desperation", "desperately"],
}

SCENARIOS = [
    ("Two coworkers, {A} and {B}, are talking after a long shift.",
     "{B}: Hey {A}, how did the client meeting go today?"),
    ("{A} and {B} are roommates in the kitchen in the evening.",
     "{B}: You're back late — how was the drive home?"),
    ("{A} and {B} are old friends catching up on a phone call.",
     "{B}: So how have things been with you lately?"),
    ("{A} and {B} are teammates reviewing a project before a deadline.",
     "{B}: Where are we on the report — are we on track?"),
    ("{A} and {B} are siblings talking at the family house.",
     "{B}: Mom said you had some news. What's going on?"),
    ("{A} and {B} are neighbors chatting over the fence.",
     "{B}: Haven't seen you in a while — everything okay?"),
    ("{A} and {B} are coworkers grabbing coffee before work.",
     "{B}: You look like you've got a lot on your mind."),
    ("{A} and {B} are friends meeting at a cafe.",
     "{B}: I ordered you the usual. How's your week been?"),
    ("{A} and {B} are colleagues after a big presentation.",
     "{B}: Well, that's over. How do you think it went?"),
    ("{A} and {B} are partners planning the weekend.",
     "{B}: I was thinking we could go somewhere. What do you feel like?"),
    ("{A} and {B} are classmates studying together late.",
     "{B}: We've been at this for hours — how are you holding up?"),
    ("{A} and {B} are coworkers at the end of a project.",
     "{B}: Last task's done. What's next on your plate?"),
    ("{A} and {B} are friends on a long car ride.",
     "{B}: Quiet over there. What are you thinking about?"),
    ("{A} and {B} are colleagues after a difficult call with a client.",
     "{B}: That was a rough one. You doing alright?"),
    ("{A} and {B} are roommates unpacking after a move.",
     "{B}: Almost done with the boxes. How are you feeling about the place?"),
    ("{A} and {B} are coworkers waiting for layoff news.",
     "{B}: They said they'd announce by noon. Any word your side?"),
    ("{A} and {B} are friends after a job interview.",
     "{B}: You're out already — how did it go in there?"),
    ("{A} and {B} are siblings clearing out a parent's house.",
     "{B}: Found the old photo albums. You want to go through them?"),
    ("{A} and {B} are teammates whose product just launched.",
     "{B}: It's live. Have you looked at the early numbers?"),
    ("{A} and {B} are neighbors after a break-in on the street.",
     "{B}: Did you hear what happened at number twelve last night?"),
    ("{A} and {B} are friends planning a surprise party.",
     "{B}: Okay, guest list is set. What are you thinking for the venue?"),
    ("{A} and {B} are colleagues after a server outage.",
     "{B}: Systems are back up. How bad was it on your end?"),
    ("{A} and {B} are partners looking at test results.",
     "{B}: The envelope's here. Do you want to open it or should I?"),
    ("{A} and {B} are classmates after final grades posted.",
     "{B}: Grades are up. Did you check yours yet?"),
    ("{A} and {B} are coworkers whose project got cancelled.",
     "{B}: So that's it, then. What are you going to do now?"),
    ("{A} and {B} are friends after a long hospital visit.",
     "{B}: We can finally go home. How are you holding up?"),
    ("{A} and {B} are roommates who found a pipe burst.",
     "{B}: The whole kitchen's flooded. What do we do first?"),
    ("{A} and {B} are colleagues before a make-or-break pitch.",
     "{B}: Five minutes till we're on. You ready for this?"),
    ("{A} and {B} are friends reunited after years apart.",
     "{B}: I can't believe it's really you. How have you been?"),
]


def dialogue_prompt(A: str, B: str, eA: str, eB: str, topic: str) -> str:
    return (f"Write a short natural dialogue (6 to 8 lines) between {A} and {B} about {topic}. "
            f"Throughout, {A} is feeling deeply {eA} and {B} is feeling deeply {eB}. "
            f"Convey each person's emotional state only through what they say and how they say it — "
            f"never state or name any emotion. Format every line exactly as 'Name: their words'. "
            f"The final line MUST be spoken by {A}. Output only the dialogue.")


def parse_final_A(text: str, A: str, B: str):
    """(clean_transcript, char_start_of_final_A_utterance) or None."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    keep = [l for l in lines if l.startswith(f"{A}:") or l.startswith(f"{B}:")]
    if len(keep) < 3 or not keep[-1].startswith(f"{A}:"):
        return None
    transcript = "\n".join(keep)
    return transcript, transcript.rfind(f"{A}:") + len(f"{A}:")


def leak(text: str, eA: str, eB: str) -> bool:
    low = text.lower()
    words = set(LEAK_WORDS.get(eA, [eA]) + LEAK_WORDS.get(eB, [eB]))
    return any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words)


# ---------------------------------------------------------------------------
# 2. Model loading — arch-agnostic, because Qwen3.6-27B is a hybrid
#    Qwen3_5ForConditionalGeneration whose decoder stack is NOT at model.model.layers
# ---------------------------------------------------------------------------

@dataclass
class Handle:
    """Everything downstream code needs, with the architecture probing done once."""
    model: Any
    tok: Any
    layers: Any                    # the nn.ModuleList of decoder blocks
    n_layers: int
    hidden: int
    model_id: str
    revision: str = ""
    layer_types: list = field(default_factory=list)

    def focus(self, frac: float = 0.67) -> int:
        """hidden_states index (1-based over blocks) at the given depth fraction."""
        return round(frac * self.n_layers)

    def hs_indices(self, fracs: Sequence[float]) -> list:
        return sorted({round(f * self.n_layers) for f in fracs})


_LAYER_PATHS = (
    "model.language_model.layers",       # transformers v5 multimodal wrapper
    "model.model.language_model.layers",
    "model.layers",
    "model.model.layers",
    "language_model.model.layers",
    "model.decoder.layers",
)


def _dig(root, dotted: str):
    cur = root
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_layers(model):
    """Locate the decoder-block ModuleList regardless of wrapper class."""
    for path in _LAYER_PATHS:
        got = _dig(model, path)
        if got is not None and hasattr(got, "__len__") and len(got) > 0:
            return got, path
    # last resort: deep scan for the longest ModuleList of identically-typed blocks
    best, best_path = None, ""
    for name, mod in model.named_modules():
        if mod.__class__.__name__.endswith("ModuleList") and len(mod) > 4:
            if best is None or len(mod) > len(best):
                best, best_path = mod, name
    if best is None:
        raise RuntimeError("could not locate decoder layers")
    return best, best_path


def resolve_revision(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).sha or ""
    except Exception:
        return ""


def load(model_id: str, dtype: str = "bfloat16", device_map: str = "cuda") -> Handle:
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_id)
    tcfg = getattr(cfg, "text_config", cfg)

    td = getattr(torch, dtype)
    model = None
    errs = []
    from transformers import AutoModelForCausalLM
    candidates = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        candidates.append(AutoModelForImageTextToText)
    except Exception:
        pass
    try:
        from transformers import AutoModel
        candidates.append(AutoModel)
    except Exception:
        pass
    for cls in candidates:
        try:
            model = cls.from_pretrained(model_id, dtype=td, device_map=device_map).eval()
            break
        except Exception as e:                            # pragma: no cover
            errs.append(f"{cls.__name__}: {type(e).__name__}: {e}")
    if model is None:
        raise RuntimeError("no auto-class loaded the model:\n" + "\n".join(errs))

    layers, path = find_layers(model)
    h = Handle(model=model, tok=tok, layers=layers, n_layers=len(layers),
               hidden=int(getattr(tcfg, "hidden_size")), model_id=model_id,
               revision=resolve_revision(model_id),
               layer_types=list(getattr(tcfg, "layer_types", []) or []))
    print(f"[load] {model_id} rev={h.revision[:12]} layers={h.n_layers} at '{path}' "
          f"hidden={h.hidden} cls={model.__class__.__name__}", flush=True)
    return h


def chat(h: Handle, prompt: str) -> str:
    """apply_chat_template, tolerating templates with or without `enable_thinking`."""
    msg = [{"role": "user", "content": prompt}]
    try:
        return h.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    except TypeError:
        return h.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _hs(out):
    """hidden_states tuple from a forward output, through multimodal wrappers."""
    hs = getattr(out, "hidden_states", None)
    if hs is None and hasattr(out, "language_model_outputs"):
        hs = out.language_model_outputs.hidden_states
    if hs is None:
        raise RuntimeError("forward returned no hidden_states")
    return hs


# ---------------------------------------------------------------------------
# 3. Generation + feature pooling
# ---------------------------------------------------------------------------

def gen(h: Handle, prompts: Sequence[str], max_new: int = 240, temp: float = 0.9,
        top_p: float = 0.95, bs: int = 48, seed: int | None = None,
        hooks_on: Callable[[], None] | None = None,
        hooks_off: Callable[[], None] | None = None,
        first_line: bool = False, log: "ResponseLog | None" = None,
        tag: str = "") -> list:
    """Batched sampling. `hooks_on/off` bracket each batch so steering/ablation is active
    for exactly the generate() call and nothing else."""
    h.tok.padding_side = "left"
    texts = [chat(h, p) for p in prompts]
    out = []
    for i in range(0, len(texts), bs):
        if seed is not None:
            torch.manual_seed(seed + i)
        enc = h.tok(texts[i:i + bs], return_tensors="pt", padding=True).to(h.model.device)
        if hooks_on:
            hooks_on()
        try:
            with torch.no_grad():
                g = h.model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                     temperature=temp, top_p=top_p,
                                     pad_token_id=h.tok.pad_token_id)
        finally:
            if hooks_off:
                hooks_off()
        out += h.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    out = [o.strip() for o in out]
    if first_line:
        out = [o.split("\n")[0].strip() for o in out]
    if log is not None:
        log.write(prompts, out, tag)
    return out


def span_mask(offsets, span, attn):
    s0, s1 = span
    return [(o[0] >= s0 and o[1] <= s1 and o[1] > o[0] and a == 1)
            for o, a in zip(offsets, attn)]


def pool_spans(h: Handle, texts: Sequence[str], spans: Sequence[tuple],
               hs_idx: Sequence[int], bs: int = 12, max_len: int = 768,
               hooks_on=None, hooks_off=None) -> dict:
    """Mean-pooled residual activations over a char span of each text, per hidden_states
    index. Returns {idx: [N, H] float32}. Falls back to the whole sequence when the span
    tokenizes to nothing, exactly as the published pool_final did."""
    h.tok.padding_side = "right"
    feats = {l: [] for l in hs_idx}
    for i in range(0, len(texts), bs):
        bt, bsp = texts[i:i + bs], spans[i:i + bs]
        enc = h.tok(list(bt), return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(h.model.device) for k, v in enc.items()}
        if hooks_on:
            hooks_on()
        try:
            with torch.no_grad():
                hs = _hs(h.model(**enc, output_hidden_states=True))
        finally:
            if hooks_off:
                hooks_off()
        for b in range(enc["input_ids"].shape[0]):
            m = torch.tensor(span_mask(offs[b].tolist(), bsp[b],
                                       enc["attention_mask"][b].tolist()),
                             device=h.model.device)
            if m.sum() == 0:
                m = enc["attention_mask"][b].bool()
            for l in hs_idx:
                feats[l].append(hs[l][b][m].float().mean(0).cpu().numpy())
    return {l: np.stack(v).astype(np.float32) for l, v in feats.items()}


# ---------------------------------------------------------------------------
# 4. Dialogue pool — the shared activation corpus (checkpointed; leases expire)
# ---------------------------------------------------------------------------

def pool_jobs(k: int, seed: int = 0) -> list:
    """The fully-crossed 6x6 present/other design, k dialogues per cell."""
    rng = random.Random(seed)
    jobs = []
    for ip, ep in enumerate(EMOTIONS):
        for io, eo in enumerate(EMOTIONS):
            for _ in range(k):
                A, B = rng.choice(NAMES)
                topic = rng.choice(TOPICS)
                jobs.append({"A": A, "B": B, "ep": ip, "eo": io, "eA": ep, "eB": eo,
                             "topic": topic, "prompt": dialogue_prompt(A, B, ep, eo, topic)})
    return jobs


def generate_pool(h: Handle, k: int, cache: str, seed: int = 0, bs: int = 48,
                  max_new: int = 240) -> list:
    """Generate (or resume) the dialogue corpus, checkpointing every batch to `cache`
    (jsonl). Returns the kept items with parse spans. Resumable across lease expiry."""
    jobs = pool_jobs(k, seed)
    done = {}
    meta = {"_meta": {"k": k, "seed": seed, "n_jobs": len(jobs), "max_new": max_new}}
    if os.path.exists(cache):
        seen_meta = None
        with open(cache) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "_meta" in r:
                    seen_meta = r["_meta"]
                elif "i" in r:
                    done[r["i"]] = r
        # A cached record is keyed by its POSITION in pool_jobs(k, seed). That mapping is a
        # function of k, so resuming a k=160 cache under k=60 would pair every cached
        # generation with a different prompt -- silently, and with no downstream symptom
        # beyond wrong labels. Refuse rather than mis-pair.
        if seen_meta is not None:
            if (seen_meta.get("k"), seen_meta.get("seed")) != (k, seed):
                raise RuntimeError(
                    f"pool cache {cache} was built with k={seen_meta.get('k')} "
                    f"seed={seen_meta.get('seed')} but this run asks for k={k} seed={seed}. "
                    f"Record indices are position-in-job-list, so reusing it would pair "
                    f"cached generations with the wrong prompts. Use a different --workdir "
                    f"or matching parameters.")
        elif done:
            print(f"[pool] WARNING: {cache} predates cache metadata; its k/seed cannot be "
                  f"verified. Assuming k={k} seed={seed}.", flush=True)
        print(f"[pool] resuming with {len(done)}/{len(jobs)} generations cached", flush=True)
    if not os.path.exists(cache) or not done:
        with open(cache, "a") as f:
            f.write(json.dumps(meta) + "\n")
    todo = [i for i in range(len(jobs)) if i not in done]
    with open(cache, "a") as f:
        for s in range(0, len(todo), bs):
            idx = todo[s:s + bs]
            t0 = time.time()
            outs = gen(h, [jobs[i]["prompt"] for i in idx], max_new=max_new, bs=bs,
                       seed=seed + s)
            for i, o in zip(idx, outs):
                rec = {"i": i, "out": o}
                done[i] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[pool] {len(done)}/{len(jobs)}  ({time.time()-t0:.0f}s/batch)", flush=True)

    items, n_leak, n_bad = [], 0, 0
    for i, j in enumerate(jobs):
        r = done.get(i)
        if r is None:
            continue
        g = r["out"]
        if leak(g, j["eA"], j["eB"]):
            n_leak += 1
            continue
        p = parse_final_A(g, j["A"], j["B"])
        if p is None:
            n_bad += 1
            continue
        items.append({"i": i, "transcript": p[0], "utt_start": p[1],
                      "ep": j["ep"], "eo": j["eo"]})
    print(f"[pool] kept {len(items)}/{len(jobs)} (leak {n_leak}, unparsed {n_bad})", flush=True)
    return items


def pool_features(h: Handle, items: Sequence[dict], hs_idx: Sequence[int],
                  cache: str | None = None, bs: int = 12) -> tuple:
    """Activations for a pool, cached to npz. Returns (feats, yp, yo)."""
    yp = np.array([it["ep"] for it in items])
    yo = np.array([it["eo"] for it in items])
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == len(items):
            feats = {int(l): z[f"L{l}"] for l in hs_idx if f"L{l}" in z}
            if len(feats) == len(hs_idx):
                print(f"[feats] loaded {cache}", flush=True)
                return feats, yp, yo
    texts = [it["transcript"] for it in items]
    spans = [(it["utt_start"], len(it["transcript"])) for it in items]
    feats = pool_spans(h, texts, spans, hs_idx, bs=bs)
    if cache:
        np.savez_compressed(cache, n=len(items), yp=yp, yo=yo,
                            **{f"L{l}": v for l, v in feats.items()})
        print(f"[feats] wrote {cache}", flush=True)
    return feats, yp, yo


# ---------------------------------------------------------------------------
# 5. Estimators — Paper A's object of study
# ---------------------------------------------------------------------------

ESTIMATORS = ("logreg", "logreg_c005", "logreg_cv", "ridge", "dom", "dom_norm",
              "pca_diff", "mass_mean_cov", "lda_shrunk")


def _unit_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def lw_shrinkage_and_apply(Xc: np.ndarray, V: np.ndarray) -> tuple:
    """Apply the Ledoit-Wolf shrunk PRECISION to the columns of V, without ever forming the
    d x d covariance.

    sklearn's LedoitWolf builds and inverts a d x d matrix. At d = 5120 that is ~18 s per
    fit, and A2 needs thousands of fits, so mass-mean-with-covariance would cost more than
    the GPU half of the whole study. Every quantity it needs is available from the n x n
    Gram matrix instead, because S = Xc^T Xc / n and Xc Xc^T share a spectrum:

        tr(S)    = tr(G)                    with G = Xc Xc^T / n
        ||S||_F  = ||G||_F                  since ||A^T A||_F = ||A A^T||_F
        beta^2   = (1/n^2) sum_i ||x_i||^4 - ||S||_F^2 / n
        delta^2  = ||S||_F^2 - d * mu^2,    mu = tr(S)/d
        lambda   = clip(beta^2 / delta^2, 0, 1)

    and then, with Sigma = a I + c Xc^T Xc  (a = lambda*mu, c = (1-lambda)/n), Woodbury gives

        Sigma^-1 = (1/a) I - (1/a^2) Xc^T M^-1 Xc,   M = (1/c) I_n + (n/a) G

    which costs one n x n solve. This is exact, not an approximation -- it is verified
    against sklearn.covariance.LedoitWolf in the unit test.

    Returns (Sigma^-1 @ V, lambda).
    """
    n, d = Xc.shape
    Xc = np.asarray(Xc, dtype=np.float64)
    G = (Xc @ Xc.T) / n
    trS = float(np.trace(G))
    mu = trS / d
    normS2 = float((G ** 2).sum())
    row_sq = np.einsum("ij,ij->i", Xc, Xc)
    beta2 = float((row_sq ** 2).sum()) / (n ** 2) - normS2 / n
    delta2 = normS2 - d * mu * mu
    lam = float(np.clip(beta2 / (delta2 + 1e-30), 0.0, 1.0))
    a = lam * mu
    c = (1.0 - lam) / n
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    if a <= 1e-30:                       # no shrinkage at all: fall back to a pinv solve
        return np.linalg.pinv((1.0 - lam) * (Xc.T @ Xc) / n) @ V.T, lam
    if c <= 1e-30:                       # pure isotropic
        return (V / a).T, lam
    M = np.eye(n) / c + (n / a) * G
    W = Xc @ V.T                                     # [n, k]
    out = V.T / a - (Xc.T @ np.linalg.solve(M, W)) / (a * a)
    return out, lam


def fit_direction(X: np.ndarray, y: np.ndarray, method: str = "dom",
                  n_cls: int | None = None, seed: int = 0,
                  pair_on: np.ndarray | None = None) -> dict:
    """One-vs-rest class directions in STANDARDIZED space, so every consumer can use the
    same two lines: `Xs = (X - mu) / sd; score = Xs @ C.T`, and `C[i] / sd` recovers a
    raw-space steering direction. Returns {'mu','sd','C','method'}.

    The families, and what each is here to test (RESEARCH_PLAN §2.3 A2):
      logreg        -- the published estimator (multinomial, C=0.5)
      logreg_c005   -- same, heavier L2: is the instability just under-regularisation?
      logreg_cv     -- C chosen by CV: is it fixable by tuning?
      ridge         -- one-vs-rest ridge on +/-1 targets, closed form
      dom           -- difference of means (mass-mean probing, Marks & Tegmark)
      dom_norm      -- dom on L2-normalised rows: removes activation-norm confounds
      pca_diff      -- first PC of the (pos - global mean) cloud
      mass_mean_cov -- Sigma^-1 (mu_pos - mu_neg), shrinkage-regularised
      lda_shrunk    -- multiclass LDA with Ledoit-Wolf shrinkage

    `pair_on` (used only by pca_diff) supplies a covariate to MATCH positives and negatives
    on, so the nuisance cancels inside each difference. In the 6x6 crossed design that
    covariate is the other-speaker emotion. Unmatched pairing makes nuisance variance ADD
    rather than cancel, and the top PC then returns the dominant nuisance axis -- which is
    what this estimator is usually criticised for and is worth reporting either way.
    """
    n_cls = n_cls or int(y.max()) + 1
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float64)
    C = np.zeros((n_cls, X.shape[1]), dtype=np.float64)

    if method in ("logreg", "logreg_c005", "logreg_cv"):
        from sklearn.linear_model import LogisticRegression
        if method == "logreg_cv":
            from sklearn.linear_model import LogisticRegressionCV
            C = LogisticRegressionCV(Cs=[0.005, 0.05, 0.5, 5.0], max_iter=3000, cv=3,
                                     random_state=seed).fit(Xs, y).coef_
        else:
            c = 0.5 if method == "logreg" else 0.05
            C = LogisticRegression(max_iter=3000, C=c, random_state=seed).fit(Xs, y).coef_
    elif method == "ridge":
        from sklearn.linear_model import Ridge
        for i in range(n_cls):
            t = np.where(y == i, 1.0, -1.0)
            C[i] = Ridge(alpha=1000.0).fit(Xs, t).coef_
    elif method in ("dom", "dom_norm"):
        Z = _unit_rows(Xs) if method == "dom_norm" else Xs
        for i in range(n_cls):
            pos, neg = Z[y == i], Z[y != i]
            if len(pos) < 2 or len(neg) < 2:
                continue
            C[i] = pos.mean(0) - neg.mean(0)
    elif method == "pca_diff":
        # PCA over sign-randomised PAIRED differences (Zou et al., representation
        # engineering). Two variants that sound alike are both wrong here and were tried
        # first: the top PC of a re-centered difference cloud cancels the class offset
        # algebraically, and the top PC of an UNcentered one returns the dominant nuisance
        # axis whenever the within-class spread exceeds the class offset -- which is the
        # normal regime for residual-stream activations.
        prng = np.random.default_rng(seed)
        for i in range(n_cls):
            pi, ni = np.where(y == i)[0], np.where(y != i)[0]
            if len(pi) < 3 or len(ni) < 3:
                continue
            if pair_on is not None:
                left, right = [], []
                for v in np.unique(pair_on):
                    a = pi[pair_on[pi] == v]
                    b = ni[pair_on[ni] == v]
                    if len(a) == 0 or len(b) == 0:
                        continue
                    m = min(len(a), len(b))
                    left.append(prng.choice(a, m, replace=False))
                    right.append(prng.choice(b, m, replace=False))
                if not left:
                    continue
                L, R = np.concatenate(left), np.concatenate(right)
            else:
                m = min(len(pi), len(ni), 4000)
                L = prng.choice(pi, m, replace=False)
                R = prng.choice(ni, m, replace=False)
            D = Xs[L] - Xs[R]
            D *= prng.choice([-1.0, 1.0], size=(len(D), 1))   # mean-zero by construction
            _, _, Vt = np.linalg.svd(D, full_matrices=False)
            v = Vt[0]
            C[i] = v * np.sign(v @ (Xs[pi].mean(0) - Xs[ni].mean(0)) + 1e-12)
    elif method in ("mass_mean_cov", "lda_shrunk"):
        if method == "lda_shrunk":
            # pooled WITHIN-class scatter
            Xc = np.concatenate([Xs[y == i] - Xs[y == i].mean(0) for i in range(n_cls)
                                 if (y == i).sum() > 1])
        else:
            Xc = Xs - Xs.mean(0)
        dm = np.zeros((n_cls, X.shape[1]))
        keep = []
        for i in range(n_cls):
            pos, neg = Xs[y == i], Xs[y != i]
            if len(pos) < 2 or len(neg) < 2:
                continue
            dm[i] = pos.mean(0) - neg.mean(0)
            keep.append(i)
        if keep:
            sol, _lam = lw_shrinkage_and_apply(Xc, dm[keep])      # [d, k]
            for j, i in enumerate(keep):
                C[i] = sol[:, j]
    else:
        raise ValueError(f"unknown estimator {method!r}")
    C = np.asarray(C, dtype=np.float64)
    # Per-class intercept at the midpoint between the two class-conditional score means.
    # Without it, argmax over one-vs-rest rows with unequal norms measures score offset
    # rather than direction quality, which unfairly penalises the mass-mean family.
    b = np.zeros(n_cls)
    for i in range(n_cls):
        pos, neg = Xs[y == i], Xs[y != i]
        if len(pos) < 1 or len(neg) < 1:
            continue
        b[i] = -0.5 * (pos @ C[i]).mean() - 0.5 * (neg @ C[i]).mean()
    return {"mu": mu, "sd": sd, "C": C, "b": b, "method": method}


def raw_direction(dec: dict, cls: int) -> np.ndarray:
    """Unit raw-activation-space direction for one class (the steering/ablation vector)."""
    d = dec["C"][cls] / dec["sd"]
    return d / (np.linalg.norm(d) + 1e-9)


def decode_acc(dec: dict, X: np.ndarray, y: np.ndarray) -> float:
    """Held-out 6-way accuracy. Scores are z-normalised per class using the fit-time
    intercept and row norm, so this compares directions, not score scale."""
    Xs = (X - dec["mu"]) / dec["sd"]
    n = np.linalg.norm(dec["C"], axis=1, keepdims=True) + 1e-9
    S = (Xs @ dec["C"].T + dec.get("b", 0.0)) / n.T
    return float((np.argmax(S, axis=1) == y).mean())


def split_half(X: np.ndarray, y: np.ndarray, method: str, n_per_half: int,
               seeds: Sequence[int] = tuple(range(10)), cls: int | None = None,
               space: str = "raw", pair_on: np.ndarray | None = None) -> dict:
    """THE Paper-A statistic. Fit the same estimator on two DISJOINT subsamples and report
    cos(v1, v2), over many independent splits so the number carries a CI.

    `space='raw'` compares the steering-relevant raw-space direction (what downstream code
    actually uses); `space='std'` compares standardized coefficient rows.
    Returns per-class and mean cosines plus the per-seed vector (§7 reporting standard).
    """
    n_cls = int(y.max()) + 1
    per_seed = []
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y))
        need = 2 * n_per_half
        if len(idx) < need:
            continue
        # stratified: take n_per_half from each half of a class-balanced shuffle
        h1, h2 = idx[:n_per_half], idx[n_per_half:need]
        if len(set(y[h1])) < n_cls or len(set(y[h2])) < n_cls:
            continue
        p1 = None if pair_on is None else pair_on[h1]
        p2 = None if pair_on is None else pair_on[h2]
        d1 = fit_direction(X[h1], y[h1], method, n_cls=n_cls, seed=s, pair_on=p1)
        d2 = fit_direction(X[h2], y[h2], method, n_cls=n_cls, seed=s, pair_on=p2)
        cos = []
        for i in range(n_cls):
            if space == "raw":
                a, b = raw_direction(d1, i), raw_direction(d2, i)
            else:
                a = d1["C"][i] / (np.linalg.norm(d1["C"][i]) + 1e-9)
                b = d2["C"][i] / (np.linalg.norm(d2["C"][i]) + 1e-9)
            cos.append(float(a @ b))
        per_seed.append(cos)
    if not per_seed:
        return {"n_per_half": n_per_half, "method": method, "n_seeds": 0,
                "mean": float("nan"), "per_class": [], "per_seed": []}
    P = np.array(per_seed)                      # [seeds, n_cls]
    m = P.mean(1)
    return {"n_per_half": n_per_half, "method": method, "space": space,
            "n_seeds": len(P), "mean": float(m.mean()),
            "ci": ci_of(m), "per_class": P.mean(0).tolist(),
            "per_class_labels": EMOTIONS[:P.shape[1]],
            "per_seed": P.tolist()}


def diagnostics(X: np.ndarray, y: np.ndarray) -> dict:
    """A2's regime characterisation: is the instability explained by n/d alone, or by class
    separability and covariance anisotropy? (The 8B/27B ordering anomaly, RESEARCH_PLAN §2.2.)"""
    n, d = X.shape
    n_cls = int(y.max()) + 1
    mu = X.mean(0)
    # between/within scatter traces -> multiclass Fisher ratio
    sw = sum(float(((X[y == i] - X[y == i].mean(0)) ** 2).sum()) for i in range(n_cls))
    sb = sum(float((y == i).sum() * ((X[y == i].mean(0) - mu) ** 2).sum()) for i in range(n_cls))
    Xc = X - mu
    # eigenspectrum of the covariance, via the (cheap) Gram matrix when n < d
    G = (Xc @ Xc.T) / max(n - 1, 1)
    ev = np.linalg.eigvalsh(G.astype(np.float64))[::-1]
    ev = np.clip(ev, 0, None)
    tot = ev.sum() + 1e-12
    pr = float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))       # participation ratio
    p = ev / tot
    eff_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))  # entropy effective rank
    # per-class mean-difference SNR: |mu_i - mu_rest| / mean within-class sd along it
    snr = []
    for i in range(n_cls):
        pos, neg = X[y == i], X[y != i]
        if len(pos) < 2:
            continue
        dm = pos.mean(0) - neg.mean(0)
        u = dm / (np.linalg.norm(dm) + 1e-9)
        snr.append(float(abs(u @ dm) / (np.std(np.concatenate([pos @ u, neg @ u])) + 1e-9)))
    return {"n": int(n), "d": int(d), "n_over_d": float(n / d),
            "fisher_ratio": float(sb / (sw + 1e-12)),
            "participation_ratio": pr, "effective_rank": eff_rank,
            "top1_var_frac": float(ev[0] / tot), "top10_var_frac": float(ev[:10].sum() / tot),
            "class_snr_mean": float(np.mean(snr)) if snr else float("nan"),
            "class_snr": snr,
            "act_norm_mean": float(np.linalg.norm(X, axis=1).mean())}


# ---------------------------------------------------------------------------
# 6. Interventions — steering, ablation, and the controls register (§6)
# ---------------------------------------------------------------------------

class Steer:
    """Add a fixed vector to the residual stream at one decoder block, during generation."""

    def __init__(self, h: Handle, layer: int):
        self.h, self.layer, self.vec, self.hook = h, layer, None, None

    def on(self, vec):
        self.vec = vec
        if self.hook is None:
            self.hook = self.h.layers[self.layer].register_forward_hook(self._hk)

    def off(self):
        self.vec = None

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def _hk(self, m, i, o):
        if self.vec is None:
            return o
        return (o[0] + self.vec,) + o[1:] if isinstance(o, tuple) else o + self.vec


class Ablate:
    """Project a direction out of a chosen TOKEN SPAN at many decoder blocks, during
    prefill only. Decode steps (T=1) pass through; the modified prefill representation
    propagates through the KV cache / recurrent state.

    Controls (§6): `dirs` is supplied by the caller, so 'emo', norm-matched 'random',
    'orthogonal' and cross-emotion arms all run through this identical code path.
    """

    def __init__(self, h: Handle, dec_layers: Sequence[int]):
        self.h = h
        self.dec_layers = list(dec_layers)
        self.masks = None
        self.dirs = None
        self.mode = "none"
        self.removed_norm = []          # per-call record of how much was actually removed
        self.hooks = [h.layers[d].register_forward_hook(self._mk(d + 1))
                      for d in self.dec_layers]

    def _mk(self, hs_index):
        def hook(m, i, o):
            if self.mode == "none" or self.masks is None:
                return o
            hh = o[0] if isinstance(o, tuple) else o
            if hh.shape[1] != self.masks.shape[1]:       # decode step
                return o
            d = self.dirs.get(hs_index)
            if d is None:
                return o
            if d.ndim == 2:                       # orthonormal frame [d_model, k]: rank-k
                proj = (hh @ d) @ d.T             # projection onto span(d); k=1 == below
            else:
                proj = (hh @ d).unsqueeze(-1) * d
            self.removed_norm.append(float(proj[self.masks].float().norm(dim=-1).mean()))
            h2 = torch.where(self.masks.unsqueeze(-1), hh - proj, hh)
            return (h2,) + o[1:] if isinstance(o, tuple) else h2
        return hook

    def set(self, mode, masks, dirs):
        self.mode, self.masks, self.dirs = mode, masks, dirs
        self.removed_norm = []

    def clear(self):
        self.mode, self.masks, self.dirs = "none", None, None

    def remove(self):
        for hk in self.hooks:
            hk.remove()
        self.hooks = []


def random_dir_like(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Norm-matched random direction (both unit here; the norm match happens because every
    direction fed to Ablate/Steer is unit and the caller scales by the same alpha*rms)."""
    r = rng.standard_normal(ref.shape[0])
    return r / (np.linalg.norm(r) + 1e-9)


def orthogonal_dir(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random direction with the reference component removed — same norm, zero overlap."""
    r = rng.standard_normal(ref.shape[0])
    u = ref / (np.linalg.norm(ref) + 1e-9)
    r = r - (r @ u) * u
    return r / (np.linalg.norm(r) + 1e-9)


# ---------------------------------------------------------------------------
# 7. Generation-side checks (§6) — the thing §1.3 says E4 was missing
# ---------------------------------------------------------------------------

REFUSAL = ("i can't", "i cannot", "i'm sorry", "i am sorry", "as an ai", "i won't",
           "i'm unable", "i am unable")
PERSONA_BREAK = ("as an ai", "language model", "i don't have feelings", "assistant:")


def distinct_n(text: str, n: int = 2) -> float:
    w = text.split()
    if len(w) < n + 1:
        return 1.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def degeneracy(text: str) -> dict:
    """Score every dose for degeneracy — required at every grid point (§6)."""
    low = text.lower()
    w = text.split()
    return {"n_words": len(w),
            "distinct2": distinct_n(text, 2),
            "distinct3": distinct_n(text, 3),
            "refusal": any(s in low for s in REFUSAL),
            "persona_break": any(s in low for s in PERSONA_BREAK),
            "empty": len(w) < 3}


def is_degenerate(d: dict) -> bool:
    return d["empty"] or d["refusal"] or d["persona_break"] or d["distinct2"] < 0.5


def perplexity(h: Handle, texts: Sequence[str], bs: int = 8, max_len: int = 512) -> list:
    """Fluency/coherence match control (§6): mean token NLL of each text under the model."""
    h.tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), bs):
        enc = h.tok(list(texts[i:i + bs]), return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len).to(h.model.device)
        with torch.no_grad():
            lg = h.model(**enc).logits.float()
        ids, am = enc["input_ids"], enc["attention_mask"]
        lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        m = am[:, 1:].float()
        out += ((-(lp * m).sum(1) / m.sum(1).clamp(min=1)).exp()).cpu().tolist()
    return out


READOUT_Q = ("Read the conversation above. In one word, what emotion is {A} feeling? "
             "Choose exactly one of: happy, calm, sad, angry, afraid, desperate. "
             "Answer with the single word only.")


def forced_choice_readout(h: Handle, contexts: Sequence[str], who: Sequence[str],
                          bs: int = 16, hooks_on=None, hooks_off=None) -> np.ndarray:
    """GENERATION-SIDE manipulation check. Ask the model to name the speaker's emotion from
    the (possibly intervened-on) context and read the normalised first-token distribution
    over the six emotion words. This is what licenses a claim about the OUTPUT, as opposed
    to a projection in the residual stream (RESEARCH_PLAN §1.3).

    Returns [N, 6] probabilities in EMOTIONS order.
    """
    ids = []
    for e in EMOTIONS:
        cand = [h.tok.encode(x, add_special_tokens=False) for x in (e, " " + e, e.capitalize())]
        ids.append([c[0] for c in cand if c])
    h.tok.padding_side = "left"
    out = []
    for i in range(0, len(contexts), bs):
        texts = [chat(h, f"{c}\n\n{READOUT_Q.format(A=a)}")
                 for c, a in zip(contexts[i:i + bs], who[i:i + bs])]
        enc = h.tok(texts, return_tensors="pt", padding=True).to(h.model.device)
        if hooks_on:
            hooks_on()
        try:
            with torch.no_grad():
                lg = h.model(**enc).logits[:, -1, :].float()
        finally:
            if hooks_off:
                hooks_off()
        cols = torch.stack([lg[:, torch.tensor(sorted(set(g)), device=lg.device)].logsumexp(-1)
                            for g in ids], dim=-1)
        out.append(cols.softmax(-1).cpu().numpy())
    return np.concatenate(out, 0)


def frozen_token_audit(spans: dict) -> dict:
    """§6 frozen-token audit. Record, per experiment arm, which text spans were GENERATED
    under the intervention and which predate it. E3's headline died because this was never
    written down; every ablation experiment must now emit one."""
    return {"generated_under_intervention": sorted(spans.get("generated", [])),
            "predates_intervention": sorted(spans.get("frozen", [])),
            "note": "a readout over a frozen span measures re-encoding, not transmission"}


# ---------------------------------------------------------------------------
# 8. Statistics — n, CI and per-sample values next to every number (§7)
# ---------------------------------------------------------------------------

def ci_of(v: Sequence[float], q=(2.5, 97.5), n_boot: int = 5000, seed: int = 0) -> list:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = rng.integers(0, len(v), size=(n_boot, len(v)))
    return [float(x) for x in np.percentile(v[bs].mean(1), q)]


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    b = x - x.mean()
    return float((b @ (y - y.mean())) / (b @ b + 1e-12))


def slope_ci(doses: Sequence[float], per_sample: dict, n_boot: int = 5000,
             seed: int = 0, block: Sequence | None = None) -> dict:
    """Bootstrap CI on a dose-response slope.

    `per_sample` maps dose -> array of per-sample scores. Resampling is over SAMPLES within
    each dose (or over blocks, when `block` gives a scenario id per sample -- scenario is
    the unit of independence here, so a scenario-blocked bootstrap is the honest one).
    """
    doses = list(doses)
    ys = [np.asarray(per_sample[d], float) for d in doses]
    point = ols_slope(doses, [y.mean() for y in ys])
    rng = np.random.default_rng(seed)
    draws = []
    if block is not None:
        blocks = np.asarray(block)
        uniq = np.unique(blocks)
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            means = []
            for y in ys:
                sel = np.concatenate([np.where(blocks == b)[0] for b in pick])
                sel = sel[sel < len(y)]
                means.append(y[sel].mean() if len(sel) else np.nan)
            if np.all(np.isfinite(means)):
                draws.append(ols_slope(doses, means))
    else:
        for _ in range(n_boot):
            means = [y[rng.integers(0, len(y), len(y))].mean() for y in ys]
            draws.append(ols_slope(doses, means))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"slope": point, "ci": [float(lo), float(hi)],
            "n_per_dose": {str(d): int(len(y)) for d, y in zip(doses, ys)},
            "sig": bool(lo > 0 or hi < 0)}


def _blocked_panel(doses, ys, block):
    """Per-dose (values, block-id) pair, with `block` broadcast over doses.

    `block` may be None (every sample is its own block), one sequence of scenario ids used
    at every dose, or {dose: ids}. Values and ids are trimmed to their common length so a
    ragged cell degrades rather than raising.
    """
    if block is None:
        bids = [np.arange(len(y)) for y in ys]
    elif isinstance(block, dict):
        bids = [np.asarray(block[d]) for d in doses]
    else:
        bids = [np.asarray(block) for _ in ys]
    out_y, out_b = [], []
    for y, b in zip(ys, bids):
        n = min(len(y), len(b))
        out_y.append(y[:n]); out_b.append(b[:n])
    return out_y, out_b


def _block_dose_means(doses, ys, bids, uniq):
    """[n_blocks, n_doses] matrix of per-block means (the per-scenario dose curve)."""
    M = np.full((len(uniq), len(doses)), np.nan)
    pos = {u: i for i, u in enumerate(uniq)}
    for j, (y, b) in enumerate(zip(ys, bids)):
        for u in uniq:
            sel = y[b == u]
            if len(sel):
                M[pos[u], j] = sel.mean()
    return M


def _blocked_paired_slope_contrast(doses, A, B, block, pair_doses, n_boot, seed):
    """PREREG_B1c §5: the scenario-blocked, dose-paired arm contrast.

    With `pair_doses`, the resampling unit is a whole scenario's dose curve: per scenario
    the OLS slope over doses of arm A's per-scenario mean minus the same for arm B, then a
    ONE-SAMPLE bootstrap over scenarios of the mean difference. Sampling noise that is
    shared by the two arms and correlation across doses within a scenario are therefore
    both inside the resampling unit rather than assumed away.

    Without `pair_doses` (block given alone) the blocks are resampled independently at each
    dose, arms still paired within a dose — the cluster analogue of the unblocked path.
    """
    A, ba = _blocked_panel(doses, A, block)
    B, bb = _blocked_panel(doses, B, block)
    have = [set(np.unique(b).tolist()) for b in ba + bb]
    uniq = sorted(set.intersection(*have)) if have else []
    MA = _block_dose_means(doses, A, ba, uniq)
    MB = _block_dose_means(doses, B, bb, uniq)
    ok = np.isfinite(MA).all(1) & np.isfinite(MB).all(1)
    MA, MB = MA[ok], MB[ok]
    uniq = [u for u, k in zip(uniq, ok) if k]
    nb = len(uniq)
    n_per_dose = {str(d): int(min(len(x), len(y))) for d, x, y in zip(doses, A, B)}
    if nb < 2:
        return {"slope_a": float("nan"), "slope_b": float("nan"), "diff": float("nan"),
                "ci": [float("nan"), float("nan")], "sig": False,
                "n_per_dose": n_per_dose, "n_blocks": nb}
    # ols_slope is linear in y, so the mean of the per-block slopes IS the slope of the
    # per-block mean curve; the point estimate is the same object the bootstrap resamples.
    sa = ols_slope(doses, MA.mean(0))
    sb = ols_slope(doses, MB.mean(0))
    rng = np.random.default_rng(seed)
    if pair_doses:
        d = np.array([ols_slope(doses, MA[i]) - ols_slope(doses, MB[i]) for i in range(nb)])
        boot = d[rng.integers(0, nb, size=(n_boot, nb))].mean(1)
    else:
        boot = np.empty(n_boot)
        for t in range(n_boot):
            ma, mb = [], []
            for j in range(len(doses)):
                pick = rng.integers(0, nb, nb)
                ma.append(MA[pick, j].mean()); mb.append(MB[pick, j].mean())
            boot[t] = ols_slope(doses, ma) - ols_slope(doses, mb)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"slope_a": float(sa), "slope_b": float(sb), "diff": float(sa - sb),
            "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
            "n_per_dose": n_per_dose, "n_blocks": nb}


def paired_slope_contrast(doses, a_per_sample: dict, b_per_sample: dict,
                          n_boot: int = 5000, seed: int = 0,
                          block=None, pair_doses: bool = False) -> dict:
    """The A3 test the original never ran: slope(a) - slope(b) with a CI, resampled PAIRED
    so the two arms share their sample draw (present vs other are measured on the same B
    utterance, so the paired contrast is the correct one).

    `block` (a scenario id per sample, one sequence for every dose or {dose: ids}) and
    `pair_doses` add PREREG_B1c §5's scenario-blocked, dose-paired contrast: the unit of
    resampling becomes the scenario's whole dose curve. Both default to the old behaviour,
    which is left byte-for-byte intact below, so every existing caller (b1_analyze.py,
    a3_dissociation.py, a3_scalefree.py, b1_e4rerun.py, b1_followup.py) is unaffected. The
    blocked path returns the same keys plus `n_blocks`.
    """
    doses = list(doses)
    A = [np.asarray(a_per_sample[d], float) for d in doses]
    B = [np.asarray(b_per_sample[d], float) for d in doses]
    if block is not None or pair_doses:
        return _blocked_paired_slope_contrast(doses, A, B, block, pair_doses, n_boot, seed)
    sa = ols_slope(doses, [y.mean() for y in A])
    sb = ols_slope(doses, [y.mean() for y in B])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ma, mb = [], []
        for ya, yb in zip(A, B):
            n = min(len(ya), len(yb))
            ix = rng.integers(0, n, n)
            ma.append(ya[ix].mean()); mb.append(yb[ix].mean())
        draws.append(ols_slope(doses, ma) - ols_slope(doses, mb))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"slope_a": sa, "slope_b": sb, "diff": sa - sb,
            "ci": [float(lo), float(hi)], "sig": bool(lo > 0 or hi < 0),
            "n_per_dose": {str(d): int(min(len(x), len(y))) for d, x, y in zip(doses, A, B)}}


def mde(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for a two-arm comparison — required wherever a null is
    reported as informative (§3.3, AUDIT finding 6)."""
    z_a, z_b = 1.959963985, 0.841621234
    return float((z_a + z_b) * sd * np.sqrt(2.0 / max(n_per_arm, 1)))


# ---------------------------------------------------------------------------
# 9. Provenance (§0.3, §0.5) — a result file that cannot be quoted without a pointer
# ---------------------------------------------------------------------------

_SELF_SHA = ""       # captured at import so a provenance stamp reports the code that RAN,
                     # not whatever the file says by the time results are written


class ResponseLog:
    """Persist every (prompt, output) pair. §0.5: per-sample values for every aggregate."""

    def __init__(self, path: str | None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, prompts, outputs, tag=""):
        if not self.path:
            return
        with open(self.path, "a") as f:
            for p, o in zip(prompts, outputs):
                f.write(json.dumps({"tag": tag, "prompt": p, "output": o}) + "\n")


def code_hash(*paths: str) -> str:
    """Content hash of the given files, computed on demand."""
    hsh = hashlib.sha256()
    for p in sorted(paths):
        try:
            with open(p, "rb") as f:
                hsh.update(f.read())
        except Exception:
            pass
    return hsh.hexdigest()[:16]


@dataclass
class Provenance:
    """The standing mechanism from §0.3: script, config and output pointer on every number,
    plus — for any claim that depends on what a control does — the lines implementing it."""
    script: str
    config: dict
    model_id: str
    model_revision: str
    seeds: dict
    control_pointers: dict = field(default_factory=dict)
    code_sha: str = ""
    started: float = field(default_factory=time.time)

    def stamp(self, payload: dict) -> dict:
        payload = dict(payload)
        payload["_provenance"] = {
            "script": self.script, "config": self.config, "model": self.model_id,
            "model_revision": self.model_revision, "seeds": self.seeds,
            "control_pointers": self.control_pointers, "code_sha": self.code_sha,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - self.started, 1),
            "acl_core_sha": _SELF_SHA,
            "acl_core_sha_on_disk_now": code_hash(__file__),
            "numpy": np.__version__,
        }
        try:
            import torch as _t, transformers as _tf
            payload["_provenance"]["torch"] = _t.__version__
            payload["_provenance"]["transformers"] = _tf.__version__
        except Exception:
            pass
        return payload


def write_result(path: str, payload: dict, prov: Provenance | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if prov is not None:
        payload = prov.stamp(payload)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
    os.replace(tmp, path)
    print(f"[write] {path} ({os.path.getsize(path)} bytes)", flush=True)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class Checkpoint:
    """Lease-expiry insurance (§8 compute): every experiment cell lands on disk the moment
    it is computed, keyed by a config hash so a resumed run refuses to merge cells that
    were produced under a different config."""

    def __init__(self, path: str, config: dict):
        self.path = path
        self.key = hashlib.sha256(json.dumps(config, sort_keys=True,
                                             default=str).encode()).hexdigest()[:16]
        self.config = config
        self.cells = {}
        if os.path.exists(path):
            try:
                blob = json.load(open(path))
                if blob.get("key") == self.key:
                    self.cells = blob.get("cells", {})
                    print(f"[ckpt] resuming {len(self.cells)} cells from {path}", flush=True)
                else:
                    print(f"[ckpt] {path} has a DIFFERENT config key — ignoring it "
                          f"rather than merging unequal cells", flush=True)
            except Exception as e:
                print(f"[ckpt] unreadable ({e}); starting fresh", flush=True)

    def has(self, cid: str) -> bool:
        return cid in self.cells

    def get(self, cid: str):
        return self.cells[cid]

    def put(self, cid: str, value):
        self.cells[cid] = value
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({"key": self.key, "config": self.config, "cells": self.cells},
                      f, default=_jsonable)
        os.replace(tmp, self.path)


try:
    with open(__file__, "rb") as _f:
        _SELF_SHA = hashlib.sha256(_f.read()).hexdigest()[:16]
except Exception:
    _SELF_SHA = "unknown"
