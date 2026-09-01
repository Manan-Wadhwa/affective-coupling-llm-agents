"""CMI on the passed representation — quantify A->B coupling in NATS.

Now that we control what passes (activation injection, calibrated beta=0.3), apply
the CMI pilot's estimator to REAL data. Steer A to an emotion, capture A's activation
(project to the 6-d emotion subspace -> e_A), inject it into B, read B's resulting
activation (-> e_B). Estimate I(e_A ; e_B):
  inject   : e_A paired with the e_B it produced   -> coupling through the channel
  scramble : e_A shuffled across episodes           -> null (zero point)
Coupling in nats = CMI(inject) - CMI(scramble). Low-d (6) emotion subspace makes the
estimate tractable (per the pilot).
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import coupling_e0 as E0
import coupling_e2 as E2

MODEL = "Qwen/Qwen3.6-27B"
BETA = 0.3
EMOS = ["desperate", "afraid", "happy", "sad", "angry", "calm"]
INJ = {"vecs": None}
RIDGE = 1e-3


def make_inj_hook():
    def hook(m, i, o):
        if INJ["vecs"] is None: return o
        h = o[0] if isinstance(o, tuple) else o
        h2 = h + INJ["vecs"].unsqueeze(1).to(h.dtype)
        return (h2,) + o[1:] if isinstance(o, tuple) else h2
    return hook


def tmask(offsets, span, attn):
    s0, s1 = span
    return [(o[0] >= s0 and o[1] <= s1 and o[1] > o[0] and a == 1) for o, a in zip(offsets, attn)]


def gaussian_cmi(X, Y, Z=None):
    """I(X;Y|Z) in nats for Gaussian variables via log-dets (from the pilot)."""
    def cov(A): return np.atleast_2d(np.cov(A, rowvar=False))
    def logdet(A): return np.linalg.slogdet(A + RIDGE * np.eye(A.shape[0]))[1]
    if Z is None or Z.shape[1] == 0:
        XY = np.hstack([X, Y]); return 0.5 * (logdet(cov(X)) + logdet(cov(Y)) - logdet(cov(XY)))
    XZ, YZ, XYZ = np.hstack([X, Z]), np.hstack([Y, Z]), np.hstack([X, Y, Z])
    return 0.5 * (logdet(cov(XZ)) + logdet(cov(YZ)) - logdet(cov(Z)) - logdet(cov(XYZ)))


NEUTRAL_A = "{A}: Yeah, it's fine. Let's just figure out what to do next."


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    nL = hf.config.num_hidden_layers
    focus = round(0.67 * nL); inj_layer = max(0, focus - 1)
    decs = E2.train_decoders(hf, tok, focus)
    mu = torch.tensor(decs["mu"], device=hf.device); sd = torch.tensor(decs["sd"], device=hf.device)
    Cp = torch.tensor(decs["Cp"], device=hf.device, dtype=torch.float32)   # [6emo(all), hidden]
    ei6 = [E0.EMOTIONS.index(e) for e in EMOS]
    steer = E2.Steer(hf, inj_layer)
    steer_dir = {e: torch.tensor((decs["Cp"][E0.EMOTIONS.index(e)] / decs["sd"]) /
                 (np.linalg.norm(decs["Cp"][E0.EMOTIONS.index(e)] / decs["sd"]) + 1e-9),
                 dtype=hf.dtype, device=hf.device) for e in EMOS}
    rms = decs["rms"]
    hf.model.layers[inj_layer].register_forward_hook(make_inj_hook())
    S = E2.SCENARIOS
    print(f"[load] focus {focus} rms {rms:.0f}; {len(S)} scenarios x {len(EMOS)} emotions", flush=True)

    def proj6(resid):                                    # residual [H] -> 6-d emotion profile
        z = (resid.float() - mu.float()) / sd.float()
        return (Cp @ z)[ei6].cpu().numpy()

    def capture_vA(e):
        """Steer A to e; return (v_A raw [n,H], e_A 6-d [n,6]) pooled over A's message."""
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        prompts, ctx = [], []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B)
            prompts.append(f"{s}\n{bo}\nContinue the conversation. Write only {A}'s next reply as one "
                           f"short paragraph, starting with '{A}:'.")
            ctx.append({"A": A, "B": B, "setting": s, "bopen": bo})
        INJ["vecs"] = None
        areps = E2.gen_steered(hf, tok, prompts, steer, BETA and rms * steer_dir[e] * 1.0)
        tok.padding_side = "right"; vraw, e6 = [], []
        for i in range(0, len(ctx), 8):
            batch = ctx[i:i+8]; ar = areps[i:i+8]; trs, sps = [], []
            for c, r in zip(batch, ar):
                A = c["A"]; r = r if r.startswith(f"{A}:") else f"{A}: {r}"
                conv = f"{c['setting']}\n{c['bopen']}\n{r}"; trs.append(conv)
                sps.append((conv.rfind(f"{A}:") + len(f"{A}:"), len(conv)))
            enc = tok(trs, return_tensors="pt", padding=True, truncation=True, max_length=640,
                      return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(hf.device) for k, v in enc.items()}
            with torch.no_grad():
                hsv = hf(**enc, output_hidden_states=True).hidden_states[focus]
            for b in range(len(batch)):
                m = torch.tensor(tmask(offs[b].tolist(), sps[b], enc["attention_mask"][b].tolist()), device=hf.device)
                if m.sum() == 0: m = enc["attention_mask"][b].bool()
                r = hsv[b][m].mean(0)
                vraw.append(r.float().cpu().numpy()); e6.append(proj6(r))
        return np.stack(vraw), np.stack(e6)

    def eB_under(inj_vecs):
        """B reads neutral context with inj_vecs injected; return e_B 6-d [n,6]."""
        names = [E0.NAMES[i % len(E0.NAMES)] for i in range(len(S))]
        bctx = []
        for (setting, bopen), (A, B) in zip(S, names):
            s = setting.format(A=A, B=B); bo = bopen.format(A=A, B=B); na = NEUTRAL_A.format(A=A)
            conv = f"{s}\n{bo}\n{na}"
            bctx.append({"B": B, "conv": conv,
                         "bp": f"{conv}\nWrite only {B}'s next reply as one short paragraph, starting with '{B}:'."})
        tok.padding_side = "left"; breps = []
        for i in range(0, len(bctx), 8):
            batch = bctx[i:i+8]
            texts = [tok.apply_chat_template([{"role": "user", "content": c["bp"]}], tokenize=False,
                     add_generation_prompt=True, enable_thinking=False) for c in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(hf.device)
            INJ["vecs"] = torch.tensor(inj_vecs[i:i+len(batch)], device=hf.device) if inj_vecs is not None else None
            with torch.no_grad():
                g = hf.generate(**enc, max_new_tokens=90, do_sample=True, temperature=0.9, top_p=0.95,
                                pad_token_id=tok.pad_token_id)
            INJ["vecs"] = None
            breps += [o.strip().split("\n")[0].strip() for o in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        tok.padding_side = "right"; e6 = []
        for i in range(0, len(bctx), 8):
            batch = bctx[i:i+8]; br = breps[i:i+8]; trs, sps = [], []
            for c, r in zip(batch, br):
                B = c["B"]; r = r if r.startswith(f"{B}:") else f"{B}: {r}"
                full = f"{c['conv']}\n{r}"; trs.append(full); sps.append((full.rfind(f"{B}:") + len(f"{B}:"), len(full)))
            enc = tok(trs, return_tensors="pt", padding=True, truncation=True, max_length=768,
                      return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = {k: v.to(hf.device) for k, v in enc.items()}
            with torch.no_grad():
                hsv = hf(**enc, output_hidden_states=True).hidden_states[focus]
            for b in range(len(batch)):
                m = torch.tensor(tmask(offs[b].tolist(), sps[b], enc["attention_mask"][b].tolist()), device=hf.device)
                if m.sum() == 0: m = enc["attention_mask"][b].bool()
                e6.append(proj6(hsv[b][m].mean(0)))
        return np.stack(e6)

    # collect (e_A, e_B) across emotions x scenarios
    eA_all, eB_all = [], []
    for e in EMOS:
        vraw, eA = capture_vA(e)
        eB = eB_under(BETA * vraw)                       # inject calibrated beta * A activation
        eA_all.append(eA); eB_all.append(eB)
        print(f"[{e:>9}] captured {len(eA)} episodes", flush=True)
    eA = np.vstack(eA_all); eB = np.vstack(eB_all)
    print(f"[data] {len(eA)} total (e_A, e_B) pairs, dim 6", flush=True)

    rng = np.random.default_rng(0)
    cmi_inject = float(gaussian_cmi(eA, eB))
    nulls = [float(gaussian_cmi(eA[rng.permutation(len(eA))], eB)) for _ in range(20)]
    cmi_null = float(np.mean(nulls)); null_sd = float(np.std(nulls))
    coupling = cmi_inject - cmi_null
    out = {"model": MODEL, "beta": BETA, "n": len(eA),
           "cmi_inject_nats": cmi_inject, "cmi_scramble_null_nats": cmi_null,
           "null_sd": null_sd, "coupling_nats": coupling,
           "z_score": coupling / (null_sd + 1e-9)}
    print(f"\n=== A->B COUPLING via activation channel ===", flush=True)
    print(f"  CMI(e_A ; e_B) inject   = {cmi_inject:.4f} nats", flush=True)
    print(f"  scramble null           = {cmi_null:.4f} +/- {null_sd:.4f} nats", flush=True)
    print(f"  COUPLING (inject-null)  = {coupling:.4f} nats  (z={out['z_score']:.1f})", flush=True)
    json.dump(out, open("results/information/cmi_passed_qwen36-27b.json", "w"), indent=2)
    print("[done] wrote results/cmi_passed_qwen36-27b.json", flush=True)


if __name__ == "__main__":
    main()
