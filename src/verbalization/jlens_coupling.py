"""J-lens coupling — express the coupling in the VERBALIZABLE J-space basis.

Reuses the logged E5 activation-passing B-replies (tags e5_B_<emo>_{inject,none}).
For each B reply, pool B's residual at a mid layer, TRANSPORT it through the
pre-fitted Jacobian lens into the final basis, unembed to a vocabulary readout,
and score the target emotion's word tokens. Compares:
  inject vs none  : does the passed activation raise B's verbalizable emotion?
  J-lens vs logit : does the Jacobian transport read the emotion where a plain
                    logit-lens (identity transport) cannot?
This gives the coupling a fixed, scenario-independent common basis (the proposal's
premise) — and it is fast: the pre-fitted lens's transport is a matmul, no
per-prompt Jacobian.
"""
import json, os, collections
import numpy as np
import torch, jlens
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
LENS_REPO, LENS_FILE = "agu18dec/qwen3.6-27b-relp-jlens", "relp-n1/lens.pt"
LAYER = 29                                     # mid layer where J-lens verbalizes (from the probe)

EMO_VOCAB = {   # representative verbalizations per emotion
    "afraid": ["fear", "afraid", "scared", "terrified", "dread", "horror", "panic", "nervous", "frightened"],
    "happy": ["happy", "joy", "delight", "thrilled", "glad", "cheerful", "excited", "wonderful"],
    "sad": ["sad", "sorrow", "grief", "loss", "tears", "lonely", "empty", "miserable", "heartbroken"],
    "angry": ["angry", "anger", "furious", "rage", "mad", "frustrated", "resentful", "outrage"],
    "calm": ["calm", "peace", "relaxed", "serene", "gentle", "quiet", "still", "soothing"],
    "desperate": ["desperate", "hopeless", "helpless", "trapped", "frantic", "pleading", "urgent", "panic"],
}


def emo_token_ids(tok):
    ids = {}
    for e, words in EMO_VOCAB.items():
        s = set()
        for w in words:
            for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                t = tok.encode(form, add_special_tokens=False)
                if t:
                    s.add(t[0])
        ids[e] = sorted(s)
    return ids


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    eids = emo_token_ids(tok)
    print("[load] model + lens ready; layer", LAYER, flush=True)

    # collect E5 B-replies by (emotion, condition)
    rows = collections.defaultdict(list)   # (emo, cond) -> [(context, reply)]
    for l in open("results/generations/responses_qwen36.jsonl"):
        r = json.loads(l); t = r["tag"]
        if t.startswith("e5_B_"):
            parts = t.split("_")             # e5_B_<emo>_<cond>
            emo, cond = parts[2], parts[3]
            rows[(emo, cond)].append((r["prompt"], r["output"]))
    print("[data]", {k: len(v) for k, v in rows.items()}, flush=True)

    def jspace_scores(items, emo, use_jacobian, bs=8):
        """Mean logit of emotion-e words in the (J-lens or logit-lens) readout of B's reply."""
        tok.padding_side = "right"; out = []
        ids = torch.tensor(eids[emo])
        for i in range(0, len(items), bs):
            batch = items[i:i+bs]
            trs, sps = [], []
            for ctx, rep in batch:
                rep = rep.strip().split("\n")[0]
                full = f"{ctx}\n{rep}"; trs.append(full)
                sps.append((full.rfind(rep), len(full)))       # B reply span
            enc = tok(trs, return_tensors="pt", padding=True, truncation=True,
                      max_length=768, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(hf.device) for k, v in enc.items()}
            with torch.no_grad():
                hs = hf(**enc, output_hidden_states=True).hidden_states[LAYER]
            for b in range(len(batch)):
                s0, s1 = sps[b]
                m = torch.tensor([(o[0] >= s0 and o[1] > o[0] and a == 1)
                                  for o, a in zip(offs[b].tolist(), enc["attention_mask"][b].tolist())],
                                 device=hf.device)
                if m.sum() == 0: m = enc["attention_mask"][b].bool()
                resid = hs[b][m].float().mean(0)                # pooled B residual [H], float32
                rr = lens.transport(resid, LAYER) if use_jacobian else resid   # transport matrix is float32
                logits = model.unembed(rr.to(torch.bfloat16).unsqueeze(0)).squeeze(0).float()
                logits = logits.log_softmax(-1)                 # normalize
                out.append(float(logits[ids].mean()))
        return out

    results = {}
    for emo in EMO_VOCAB:
        entry = {}
        for cond in ("inject", "none"):
            items = rows.get((emo, cond), [])
            if not items:
                continue
            entry[f"jlens_{cond}"] = float(np.mean(jspace_scores(items, emo, True)))
            entry[f"logit_{cond}"] = float(np.mean(jspace_scores(items, emo, False)))
        if "jlens_inject" in entry and "jlens_none" in entry:
            entry["jlens_coupling"] = entry["jlens_inject"] - entry["jlens_none"]   # verbalizable coupling
            entry["logit_coupling"] = entry.get("logit_inject", 0) - entry.get("logit_none", 0)
        results[emo] = entry
        print(f"[{emo:>9}] J-lens inject {entry.get('jlens_inject',0):+.2f} none {entry.get('jlens_none',0):+.2f} "
              f"-> coupling {entry.get('jlens_coupling',float('nan')):+.3f} | logit coupling "
              f"{entry.get('logit_coupling',float('nan')):+.3f}", flush=True)

    json.dump({"model": MODEL, "layer": LAYER, "results": results},
              open("results/verbalization/jlens_coupling_qwen36-27b.json", "w"), indent=2)
    print("[done] wrote results/jlens_coupling_qwen36-27b.json", flush=True)


if __name__ == "__main__":
    main()
