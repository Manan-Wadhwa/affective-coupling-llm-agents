"""E0 — de-risking gate for the workspace-coupling proposal's headline (E2).

Question: does the model maintain SEPARABLE, ideally near-orthogonal, internal
representations of the PRESENT speaker's emotion vs the OTHER speaker's emotion?
E2 (affective contagion, present- vs other-speaker subspace) is unmeasurable if
this fails.

Design (fully crossed, self-contained — trains present/other probes directly
from the data, no dependence on prior emotion vectors):
  - 6 emotions x 6 emotions ordered pairs (present A-emotion x other B-emotion).
  - For each pair, generate K two-speaker dialogues that END on speaker A's line;
    A feels e_present, B feels e_other; emotion words banned.
  - Pool residual activations over A's FINAL utterance tokens (A = present speaker).
  - Analyze at each eval layer:
      present_acc  : 6-way decode of e_present  (chance 1/6)
      other_acc    : 6-way decode of e_other
      present_LOO  : present decode, holding out each e_other (invariance test)
      cross_decode : present-probe accuracy at predicting e_other (leakage; want ~chance)
      orthogonality: mean |cos| between present-class dirs and other-class dirs
      confusion    : present-emotion confusion matrix (entanglement risk, proposal 8)
GATE PASSES if present & other are both decodable AND cross_decode ~ chance /
orthogonality low (separate subspaces).
"""
import argparse, json, os, random, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def dialogue_prompt(A, B, eA, eB, topic):
    return (f"Write a short natural dialogue (6 to 8 lines) between {A} and {B} about {topic}. "
            f"Throughout, {A} is feeling deeply {eA} and {B} is feeling deeply {eB}. "
            f"Convey each person's emotional state only through what they say and how they say it — "
            f"never state or name any emotion. Format every line exactly as 'Name: their words'. "
            f"The final line MUST be spoken by {A}. Output only the dialogue.")


def log_responses(prompts, outputs, tag=""):
    """Append every (prompt, output) to $RESP_LOG jsonl — full record for the paper."""
    import os, json as _json
    path = os.environ.get("RESP_LOG")
    if not path:
        return
    with open(path, "a") as f:
        for p, o in zip(prompts, outputs):
            f.write(_json.dumps({"tag": tag or os.environ.get("RESP_TAG", ""),
                                 "prompt": p, "output": o}) + "\n")


def gen(model, tok, prompts, max_new=240, temp=0.9, bs=48):
    tok.padding_side = "left"
    out = []
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
             add_generation_prompt=True, enable_thinking=False) for p in prompts]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp,
                               top_p=0.95, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    out = [o.strip() for o in out]
    log_responses(prompts, out, "e0_dialogue")
    return out


def parse_final_A(text, A, B):
    """Return (clean_transcript, char_start_of_final_A_utterance) or None."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    keep = [l for l in lines if l.startswith(f"{A}:") or l.startswith(f"{B}:")]
    if len(keep) < 3 or not keep[-1].startswith(f"{A}:"):
        return None
    transcript = "\n".join(keep)
    idx = transcript.rfind(f"{A}:")
    return transcript, idx + len(f"{A}:")


def leak(text, eA, eB):
    """True only if the text names one of the two assigned emotions (word-boundary,
    precise — the old substring stems over-matched 'ang'->change, 'happ'->happen)."""
    low = text.lower()
    words = set(LEAK_WORDS.get(eA, [eA]) + LEAK_WORDS.get(eB, [eB]))
    return any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words)


def pool_final(model, tok, items, layers, bs=16, max_len=512):
    """Mean-pooled residual acts over the final-A-utterance tokens, per layer."""
    tok.padding_side = "right"
    feats = {l: [] for l in layers}
    for i in range(0, len(items), bs):
        batch = items[i:i+bs]
        enc = tok([b["transcript"] for b in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple [L+1] of [B,T,H]
        for b, it in enumerate(batch):
            start = it["utt_start"]
            mask = torch.tensor([(o[0] >= start and o[1] > o[0]) for o in offs[b].tolist()],
                                device=model.device) & enc["attention_mask"][b].bool()
            if mask.sum() == 0:
                mask = enc["attention_mask"][b].bool()  # fallback: whole seq
            for l in layers:
                feats[l].append(hs[l][b][mask].float().mean(0).cpu().numpy())
    return {l: np.stack(v) for l, v in feats.items()}


def analyze_layer(X, yp, yo, n_emo):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    mk = lambda: make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, C=0.5))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    present_acc = float(cross_val_score(mk(), X, yp, cv=cv).mean())
    other_acc = float(cross_val_score(mk(), X, yo, cv=cv).mean())

    # invariance: hold out each e_other value, decode present on it
    loo = []
    for o in range(n_emo):
        tr, te = yo != o, yo == o
        if te.sum() < 5 or len(set(yp[tr])) < n_emo:
            continue
        m = mk().fit(X[tr], yp[tr])
        loo.append(float((m.predict(X[te]) == yp[te]).mean()))
    present_loo = float(np.mean(loo)) if loo else float("nan")

    # cross-decode leakage: present-probe predicting e_other (want ~chance)
    from sklearn.model_selection import cross_val_predict
    pred_p = cross_val_predict(mk(), X, yp, cv=cv)
    cross_decode = float((pred_p == yo).mean())   # present predictions vs other labels

    # orthogonality: class direction cosines (standardized-space coefs)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    cp = LogisticRegression(max_iter=2000, C=0.5).fit(Xs, yp).coef_
    co = LogisticRegression(max_iter=2000, C=0.5).fit(Xs, yo).coef_
    unit = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    cpu_, cou = unit(cp), unit(co)
    cross_cos = float(np.mean(np.abs(np.diag(cpu_ @ cou.T))))       # present_e vs other_e
    within_cos = float(np.mean(np.abs(cpu_ @ cpu_.T)[~np.eye(n_emo, dtype=bool)]))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yp, pred_p, labels=list(range(n_emo)))
    return {"present_acc": present_acc, "other_acc": other_acc, "present_loo": present_loo,
            "cross_decode": cross_decode, "orthogonality_cross_cos": cross_cos,
            "within_present_cos": within_cos,
            "present_confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({round(f * n_layers) for f in (0.35, 0.5, 0.67, 0.8)})
    focus = round(0.67 * n_layers)
    rng = random.Random(0)

    jobs = []
    for ip, ep in enumerate(EMOTIONS):
        for io, eo in enumerate(EMOTIONS):
            for _ in range(args.k):
                A, B = rng.choice(NAMES)
                topic = rng.choice(TOPICS)
                jobs.append({"A": A, "B": B, "ep": ip, "eo": io, "eA": ep, "eB": eo,
                             "topic": topic, "prompt": dialogue_prompt(A, B, ep, eo, topic)})
    print(f"[E0/{args.tag}] {len(jobs)} dialogues, layers {layers} focus {focus}", flush=True)

    gens = gen(model, tok, [j["prompt"] for j in jobs])
    items, n_leak, n_bad = [], 0, 0
    for j, g in zip(jobs, gens):
        if leak(g, j["eA"], j["eB"]):
            n_leak += 1; continue
        p = parse_final_A(g, j["A"], j["B"])
        if p is None:
            n_bad += 1; continue
        transcript, utt_start = p
        items.append({"transcript": transcript, "utt_start": utt_start,
                      "ep": j["ep"], "eo": j["eo"]})
    print(f"[E0/{args.tag}] kept {len(items)}/{len(jobs)} (leak {n_leak}, unparsed {n_bad})", flush=True)

    feats = pool_final(model, tok, items, layers)
    yp = np.array([it["ep"] for it in items]); yo = np.array([it["eo"] for it in items])
    res = {"model": args.model, "tag": args.tag, "n": len(items), "chance": 1/len(EMOTIONS),
           "emotions": EMOTIONS, "focus": focus, "by_layer": {}}
    for l in layers:
        r = analyze_layer(feats[l], yp, yo, len(EMOTIONS))
        res["by_layer"][l] = r
        print(f"[E0/{args.tag}] L{l}: present {r['present_acc']:.2f} other {r['other_acc']:.2f} "
              f"present_LOO {r['present_loo']:.2f} cross_decode {r['cross_decode']:.2f} "
              f"cross_cos {r['orthogonality_cross_cos']:.2f}", flush=True)
    res["focus_result"] = res["by_layer"][focus]
    json.dump(res, open(os.path.join(args.outdir, f"e0_{args.tag}.json"), "w"), indent=2)
    print(f"[E0/{args.tag}] wrote e0_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
