"""Probe the Jacobian-lens API on Qwen3.6-27B: load model + pre-fitted lens, run
apply() on emotional vs neutral prompts, inspect output structure, decode the
J-space vocabulary readout (use_jacobian=True) vs the logit-lens control (False)."""
import torch, jlens, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
LENS_REPO = "agu18dec/qwen3.6-27b-relp-jlens"
LENS_FILE = "relp-n1/lens.pt"

PROMPTS = {
    "fear": "She froze in the dark, empty house as the slow footsteps behind her grew louder and closer.",
    "joy": "He tore open the letter, read it twice, and ran through the house laughing and shouting the good news.",
    "anger": "He slammed his fist on the table, jaw clenched, and glared at the man who had lied to his face.",
    "neutral": "The report summarizes quarterly figures for the regional office and lists the scheduled meeting dates.",
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print("[load] model ready; layers", hf.config.num_hidden_layers, flush=True)
    model = jlens.from_hf(hf, tok)
    print("[load] wrapped in HFLensModel", flush=True)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    print("[load] lens loaded", flush=True)

    # ---- inspect apply() output structure once ----
    out = lens.apply(model, PROMPTS["fear"], use_jacobian=True)
    print("[apply] returned", type(out), "len", len(out) if hasattr(out, "__len__") else "?", flush=True)
    ro, x1, x2 = out
    print("[apply] readouts type:", type(ro),
          ("keys " + str(list(ro.keys())) if hasattr(ro, "keys") else "shape " + str(getattr(ro, "shape", "?"))), flush=True)
    for nm, x in (("x1", x1), ("x2", x2)):
        print(f"[apply] {nm}: type {type(x)} shape {getattr(x,'shape','?')} dtype {getattr(x,'dtype','?')}", flush=True)

    layers = list(ro.keys()) if hasattr(ro, "keys") else []
    print("[apply] layers available:", layers, flush=True)

    def top_tokens(vec, k=10):
        """vec is a vocab-logit vector -> top-k decoded tokens."""
        idx = torch.topk(vec.float(), k).indices.tolist()
        return [tok.decode([i]).strip() for i in idx]

    # figure out how to get vocab logits: if readout is already [., vocab] use it,
    # else transport+unembed. Try both, report which is sane.
    V = hf.config.vocab_size
    results = {}
    mid = layers[len(layers) // 2] if layers else None
    for emo, prompt in PROMPTS.items():
        entry = {}
        for uj in (True, False):
            ro, last_resid, _ = lens.apply(model, prompt, use_jacobian=uj)
            # last-position readout at the mid layer
            r = ro[mid] if (hasattr(ro, "keys") and mid is not None) else ro
            r = r[-1] if r.dim() > 1 else r          # last position
            if r.shape[-1] == V:                     # already vocab logits
                toks = top_tokens(r)
            else:                                    # residual -> unembed
                toks = top_tokens(model.unembed(r.unsqueeze(0)).squeeze(0))
            entry["jlens" if uj else "logitlens"] = toks
        results[emo] = entry
        print(f"[{emo:>8}] J-lens: {entry['jlens'][:8]}", flush=True)
        print(f"[{emo:>8}] logit : {entry['logitlens'][:8]}", flush=True)

    json.dump({"mid_layer": mid, "layers": layers, "results": results},
              open("results/verbalization/jlens_probe.json", "w"), indent=2)
    print("[done] wrote results/jlens_probe.json", flush=True)


if __name__ == "__main__":
    main()
