"""Improved verbalizable coupling — read the STRONG contagion in J-space.

Fixes the weak first attempt: (1) use the strong E2 contagion mechanism (steer A's
message, B catches it) not the weak E5 injection; (2) dose-response alpha=0 vs 1 not
binary; (3) CONTRASTIVE scoring — emotion-e vocab minus the mean over all emotions'
vocab — to isolate specificity. Read B's reply residual at a mid layer, transport
through the Jacobian lens, unembed, score. J-lens vs logit-lens (identity) control.
"""
import json, os
import numpy as np
import torch, jlens
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0
import coupling_e2 as E2

MODEL = "Qwen/Qwen3.6-27B"
LENS_REPO, LENS_FILE = "agu18dec/qwen3.6-27b-relp-jlens", "relp-n1/lens.pt"
LAYER = 29
EMOS = ["desperate", "afraid", "happy", "sad", "angry", "calm"]
ALPHAS = [0.0, 1.0]
EMO_VOCAB = {
    "afraid": ["fear", "afraid", "scared", "terrified", "dread", "horror", "panic", "nervous", "frightened", "trembling"],
    "happy": ["happy", "joy", "delight", "thrilled", "glad", "cheerful", "excited", "wonderful", "smile", "laughing"],
    "sad": ["sad", "sorrow", "grief", "loss", "tears", "lonely", "empty", "miserable", "heartbroken", "crying"],
    "angry": ["angry", "anger", "furious", "rage", "mad", "frustrated", "resentful", "outrage", "seething", "snapped"],
    "calm": ["calm", "peace", "relaxed", "serene", "gentle", "quiet", "still", "soothing", "ease", "steady"],
    "desperate": ["desperate", "hopeless", "helpless", "trapped", "frantic", "pleading", "urgent", "panic", "begging", "cornered"],
}


def emo_ids(tok):
    d = {}
    for e, ws in EMO_VOCAB.items():
        s = set()
        for w in ws:
            for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                t = tok.encode(f, add_special_tokens=False)
                if t: s.add(t[0])
        d[e] = sorted(s)
    return d


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    nL = hf.config.num_hidden_layers
    focus = round(0.67 * nL)
    decs = E2.train_decoders(hf, tok, focus)
    steer = E2.Steer(hf, max(0, focus - 1))
    steer_dir = {e: torch.tensor((decs["Cp"][E0.EMOTIONS.index(e)] / decs["sd"]) /
                 (np.linalg.norm(decs["Cp"][E0.EMOTIONS.index(e)] / decs["sd"]) + 1e-9),
                 dtype=hf.dtype, device=hf.device) for e in EMOS}
    rms = decs["rms"]
    eids = emo_ids(tok)
    all_ids = sorted(set(i for v in eids.values() for i in v))
    S = E2.SCENARIOS[:16]
    print(f"[load] 27B + lens; focus {focus} rms {rms:.0f}; {len(S)} scenarios", flush=True)

    def gen_B(e, al):
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        aprompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            aprompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next reply as one "
                            f"short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
        vec = None if al == 0 else (al * rms * steer_dir[e])
        areps = E2.gen_steered(hf, tok, aprompts, steer, vec)
        bprompts = []
        for c, ar in zip(ctx, areps):
            A = c["A"]; ar = ar if ar.startswith(f"{A}:") else f"{A}: {ar}"
            c["conv"] = f"{c['setting']}\n{c['bopen']}\n{ar}"
            bprompts.append(f"{c['conv']}\nWrite only {c['B']}'s next reply as one short paragraph, "
                            f"starting with '{c['B']}:'.")
        breps = E2.gen_steered(hf, tok, bprompts, steer, None)
        return [(c["conv"], (br if br.startswith(f"{c['B']}:") else f"{c['B']}: {br}"))
                for c, br in zip(ctx, breps)]

    def jscore(items, e, use_jacobian, bs=8):
        """Contrastive J-space score: mean logit of emotion-e vocab minus mean over all emotions' vocab."""
        tok.padding_side = "right"; out = []
        ide = torch.tensor(eids[e]); ida = torch.tensor(all_ids)
        for i in range(0, len(items), bs):
            batch = items[i:i+bs]; trs, sps = [], []
            for ctx, rep in batch:
                full = f"{ctx}\n{rep}"; trs.append(full); sps.append((full.rfind(rep), len(full)))
            enc = tok(trs, return_tensors="pt", padding=True, truncation=True, max_length=768,
                      return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(hf.device) for k, v in enc.items()}
            with torch.no_grad():
                hs = hf(**enc, output_hidden_states=True).hidden_states[LAYER]
            for b in range(len(batch)):
                s0, s1 = sps[b]
                m = torch.tensor([(o[0] >= s0 and o[1] > o[0] and a == 1)
                                  for o, a in zip(offs[b].tolist(), enc["attention_mask"][b].tolist())],
                                 device=hf.device)
                if m.sum() == 0: m = enc["attention_mask"][b].bool()
                resid = hs[b][m].float().mean(0)
                rr = lens.transport(resid, LAYER) if use_jacobian else resid
                lg = model.unembed(rr.to(torch.bfloat16).unsqueeze(0)).squeeze(0).float()
                out.append(float(lg[ide].mean() - lg[ida].mean()))    # contrastive
        return out

    res = {}
    for e in EMOS:
        row = {}
        for al in ALPHAS:
            items = gen_B(e, al)
            row[f"jlens_a{al}"] = float(np.mean(jscore(items, e, True)))
            row[f"logit_a{al}"] = float(np.mean(jscore(items, e, False)))
        row["jlens_contagion"] = row["jlens_a1.0"] - row["jlens_a0.0"]
        row["logit_contagion"] = row["logit_a1.0"] - row["logit_a0.0"]
        res[e] = row
        print(f"[{e:>9}] J-lens contagion {row['jlens_contagion']:+.3f} "
              f"(a0 {row['jlens_a0.0']:+.2f} -> a1 {row['jlens_a1.0']:+.2f}) | "
              f"logit contagion {row['logit_contagion']:+.3f}", flush=True)

    json.dump({"model": MODEL, "layer": LAYER, "results": res},
              open("results/jlens_contagion_qwen36-27b.json", "w"), indent=2)
    print("[done] wrote results/jlens_contagion_qwen36-27b.json", flush=True)


if __name__ == "__main__":
    main()
