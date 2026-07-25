"""J-lens verbalizable-emotion readout on Qwen3.6-27B — CORRECT version.

The J-lens advantage is at MID layers (at the final layer the transport is identity
-> J-lens==logit-lens). So read from the per-layer `readouts` dict, last content
position, at a sweep of mid-late layers; filter stopwords so emotion content surfaces.
Contrast J-lens (use_jacobian=True) vs logit-lens (False) at the same layer."""
import torch, jlens, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
LENS_REPO, LENS_FILE = "agu18dec/qwen3.6-27b-relp-jlens", "relp-n1/lens.pt"
STOP = set("the a an and or but in on at to for of with as by she he it his her him they "
           "them i you we me my your our their this that these those is was were are be been "
           "being do does did so not no yes if then than out up down over under just very "
           "s t m re ve ll d nt".split())

PROMPTS = {
    "fear": "She froze in the dark, empty house as the slow footsteps behind her grew louder and closer.",
    "joy": "She tore open the acceptance letter, read it twice, and spun around the kitchen laughing with delight.",
    "anger": "He slammed his fist on the table, jaw clenched, furious at the man who had lied straight to his face.",
    "sadness": "He still set two cups out each morning, then quietly put one back, the house silent since she had gone.",
    "desperation": "Third eviction notice, no money left, phone dead — he pounded on every locked door as the storm closed in.",
    "calm": "She let the paddle rest across her knees and drifted on the still lake, watching mist lift off the water.",
    "disgust": "The smell hit him first — rotting meat crawling with maggots — and he gagged, covering his mouth.",
    "neutral": "The report summarizes quarterly figures for the regional office and lists the scheduled meeting dates.",
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    nL = hf.config.num_hidden_layers
    SWEEP = [round(f * nL) for f in (0.45, 0.55, 0.65, 0.75)]
    print(f"[load] 27B({nL}L) + lens ready; sweep {SWEEP}", flush=True)

    def content_tokens(vec, k=25):
        idx = torch.topk(vec.float(), k).indices.tolist(); out = []
        for i in idx:
            t = tok.decode([i]).strip(); tl = t.lower()
            if len(tl) > 1 and tl not in STOP and any(c.isalpha() for c in t):
                out.append(t)
        return out[:8]

    results = {}
    for emo, prompt in PROMPTS.items():
        best = {"jlens": [], "logit": [], "layer": None}
        by_layer = {}
        for L in SWEEP:
            row = {}
            for uj in (True, False):
                ro, _, _ = lens.apply(model, prompt, use_jacobian=uj)
                r = ro[L]                                   # [positions, vocab]
                r = r[-1] if r.dim() > 1 else r             # last content position
                row["jlens" if uj else "logit"] = content_tokens(r)
            by_layer[L] = row
            if len(row["jlens"]) > len(best["jlens"]):      # layer with richest J-lens content
                best = {"jlens": row["jlens"], "logit": row["logit"], "layer": L}
        results[emo] = {"best": best, "by_layer": by_layer}
        print(f"[{emo:>11} L{best['layer']}] J-lens: {best['jlens']}", flush=True)
        print(f"[{emo:>11}    ] logit : {best['logit']}", flush=True)

    json.dump(results, open("results/jlens_readout.json", "w"), indent=2)
    print("[done] wrote results/jlens_readout.json", flush=True)


if __name__ == "__main__":
    main()
