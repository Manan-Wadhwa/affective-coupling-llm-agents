# 08 · Jacobian-lens verbalization (probe, readout, coupling, contagion)

**Files:** `results/verbalization/jlens_probe.json`, `jlens_readout.json`, `jlens_coupling_qwen36-27b.json`, `jlens_contagion_qwen36-27b.json` · **Scripts:** `src/verbalization/jlens_{probe,readout,coupling,contagion}.py` · **Model:** Qwen/Qwen3.6-27B · **Registry:** `jlens.beats_logit_lens` RETRACTED-CLAIM (contagion file); the other three UNREGISTERED · **AUDIT:** 10, 11, 17, 18

## What was run
Probe: 4 hand-written prompts, top-10 J-lens vs logit-lens tokens at the middle layer (31). Readout: 8 prompts, layer sweep at 0.45–0.75 depth, "best" layer chosen by the length of the J-lens token list. Coupling: B's reply residual at layer 29, transported through the J-lens or left as identity, unembedded, mean log-prob of emotion words; its input generations file is not in the repo. Contagion: 16 scenarios × α ∈ {0, 1}, raw contrastive logit (target-emotion words minus all-emotion words) with no normalisation, layer 29 hard-coded.

## What the files show
- Contagion: `jlens_contagion` positive 6/6 (0.05–0.47); `logit_contagion` also positive 6/6 (0.04–0.19). n = 16 per cell, no dispersion.
- Coupling: `jlens_coupling` positive 4/6 (afraid −0.068, happy −0.021); logit control 4/6. n not recorded.
- Readout: every "best" layer is 29 by tie-break; at 29 the J-lens top-8 contains an emotion word for 3/8 prompts; emotion words appear at layers 42/48 that the rule discarded. Logit lens: emotion words in 0/32 cells.

## What can be inferred
- The J-lens produces readable tokens at mid layers where the raw logit lens does not. [supported by file]
- "Beating a logit-lens control" on the contagion deltas is a scale comparison between an unnormalised transported logit and an untransported one; both are positive 6/6. [supported by file]
- The coupling file, whose design matches the 6/6 wording, gives 4/6. [supported by file]
- Nothing here separates reading emotion out of emotional text from reading emotion out of a representation. [needs: pre-reply context readout, matched-emotion control]

## Status
Registry: RETRACTED. Paper C's C0 gate is the rebuild.

### Independent critique (blind: saw only the scripts and the JSONs)

## 1. What each script actually computes

**`jlens_probe.py`** — 4 hand-written prompts, one each for fear/joy/anger + neutral (lines 11–16). Calls `lens.apply(model, prompt, use_jacobian=True/False)` (line 53), takes the readout at `mid = layers[len(layers)//2]` (line 49 → layer 31, `jlens_probe.json:mid_layer`), last position (line 56), prints top-10 raw `topk` tokens (lines 40–43). No score, no filtering, no n, no control beyond `use_jacobian=False`.

**`jlens_readout.py`** — 8 hand-written prompts, one per emotion (lines 17–26). Sweeps layers `[0.45,0.55,0.65,0.75]·nL` = 29/35/42/48 (line 35), stopword-filters top-25 and keeps ≤8 tokens (lines 38–44). "Best" layer is chosen by `len(row["jlens"]) > len(best["jlens"])` (line 58) — a strict `>` over a list capped at 8, so **ties go to the first layer swept**. Every emotion returns 8 J-lens tokens at every layer except joy@48 (verified), so **all 8 "best" entries are layer 29 by tie-break, not by content quality** (`jlens_readout.json:*.best.layer`). Selection is on the J-lens arm only; the logit list at that layer is a by-product.

**`jlens_coupling.py`** — score is a **mean log-probability**: pooled B-reply residual at layer 29 → `lens.transport(...)` or identity (line 87) → `model.unembed` → `log_softmax` → `logits[ids].mean()` (lines 86–90). Layer is **hardcoded** `LAYER = 29` (line 21), not scanned. Vocab ids are the **first subtoken only** of each surface form (line 40). n per condition is **unknowable**: it reads `results/generations/responses_qwen36.jsonl` (line 56), **which does not exist** in the repo (only `responses_qwen36_sample300.jsonl` and `responses_multiturn_qwen36_sample200.jsonl`, neither containing any `e5_` tag). The n printed at line 62 is not written to JSON (lines 110–111). No SD, no CI, no test.

**`jlens_contagion.py`** — score is a **raw contrastive logit**, `lg[ide].mean() - lg[ida].mean()` with **no `log_softmax`** (lines 102–103). Layer again hardcoded 29 (line 18). n = `E2.SCENARIOS[:16]` → **16 generations per (emotion, alpha)** (line 58), averaged at line 111. Two alphas, 0.0/1.0 (line 20). B is generated **unsteered** (line 77, `vec=None`); the score is read off B's own reply text (lines 88, 93, 100). No SD, CI, seed control, or test. It imports `coupling_e0`/`coupling_e2` (line 13) which live in `src/core/`, not on the path from `src/verbalization/`.

**Are the two lenses comparable?** In `jlens_coupling.py` yes in units (both log-softmaxed, line 89) — and on those comparable units the logit lens **wins on level** 6/6 (e.g. `afraid.jlens_inject` −12.139 vs `afraid.logit_inject` −11.251). In `jlens_contagion.py` **no**: raw unembedded logits of a *transported* vector versus an *untransported* one differ by whatever multiplicative norm change `lens.transport` applies. The contrastive subtraction removes an additive offset, not a scale factor. So contagion deltas are not on a common scale.

## 2. Do the JSONs support the claims?

**C2 / coupling** (`jlens_coupling_qwen36-27b.json`, `results.<emo>.jlens_coupling`): afraid **−0.0682**, happy **−0.0212**, sad +0.00277, angry +0.03004, calm +0.11497, desperate +0.06730 → **4/6 positive, not 6/6**. The logit control is also 4/6 positive (afraid −0.00774, sad −0.00970, rest positive). J-lens exceeds logit in *signed* delta only 3/6 (angry: 0.03004 vs 0.03056 — the control wins). n unverifiable (input file missing).

**C1 / contagion** (`jlens_contagion_qwen36-27b.json`, `results.<emo>.jlens_contagion`): desperate +0.1641, afraid +0.3324, happy +0.2065, sad +0.4659, angry +0.2875, calm +0.0549 → **6/6 positive**. But `logit_contagion` is **also 6/6 positive** (+0.1374, +0.0975, +0.1243, +0.1945, +0.1215, +0.0351). n = 16 per cell, no dispersion.

**C3 / readout**: at the reported best layer (29) the J-lens top-8 contains an emotion word for only 3/8 prompts, and for `calm` the hit is "delightful" (wrong emotion); anger, sadness, desperation, disgust and neutral have **zero** at layer 29. `desperation.by_layer.29.jlens` = `["narrative","narr","story","—but","cinematic","/story","storytelling","bite"]` — task/genre tokens, not affect. Emotion words do appear at 35/42/48 (e.g. `fear.by_layer.42.jlens` = terrified/panic/fear; `sadness.by_layer.42.jlens` = loneliness/poignant/mourn/melanch/solitude), i.e. at layers the "best" rule discarded. The logit lens returns emotion words at **0/32** layer×prompt cells. `jlens_probe.json` is weaker still: only `results.fear.jlens` (horror/dread/tense) is emotional; joy, anger and neutral are empty strings and "…".

## 3. Strongest objections

1. **The control passes the same test.** In the contagion file the logit lens is 6/6 positive too. "6/6 emotions, beating a logit-lens control" describes a result the control also achieves; only the magnitude differs, and magnitude is exactly the quantity the missing `log_softmax` (line 103) makes incomparable. A transport that scales the residual norm inflates every logit gap by the same factor — a 1.2×–3.4× ratio (computed from the two `*_contagion` fields) is what a norm change alone would produce.

2. **Where the scales *are* matched, the J-lens loses.** `jlens_coupling.py:89` normalizes both arms, and the logit lens gives a higher emotion-word log-prob in 6/6 emotions and a higher baseline contrastive score in 4/6 contagion cells (`logit_a0.0` > `jlens_a0.0` for desperate, afraid, sad, calm). That is direct counter-evidence to C3 as a quantitative statement.

3. **The 6/6 count is not consistent across the two files.** The coupling file — the one whose design matches C2's wording ("A's injected emotion", `inject` vs `none`, lines 96–103) — gives 4/6. C1 and C2 are presented as versions of one claim but rest on different scripts with different mechanisms (E5 injection vs E2 steering), different scoring (log-prob vs raw contrastive), and opposite verdicts. Reporting 6/6 while a same-named file says 4/6 is selective.

4. **`jlens_coupling.py` is not reproducible.** Its only data source (line 56) is absent, so n, balance, and even whether the JSON was produced by this version of the script cannot be checked.

5. **Unfair control.** The comparison is a *fitted* transport against *identity*. A mid-layer logit lens is known to be near-garbage; the honest control is a tuned lens/affine translator fitted on the same data. Nothing here separates "the Jacobian captures emotion" from "any fitted mid-layer translator produces readable English".

6. **Circularity of the contagion measure.** The residual is pooled over **B's own reply tokens** (lines 88, 93, 100) after B wrote an emotional reply. Reading emotion words out of the representation of emotional text is close to tautological; the 6/6 logit-lens result is the symptom. This is a behavioral contagion result re-described as a verbalizability result.

7. **Numerical floor.** Line 88 casts the transported float32 vector to bf16 before unembedding. bf16's ~0.4% relative precision on logits of order 10–30 is the same order as the reported coupling deltas (0.003–0.115). `sad.jlens_coupling` = +0.00277 is below any credible resolution.

8. **Probe/readout are anecdotes.** 4 and 8 hand-written prompts, one per emotion, no held-out set, no scoring function, no blinding, no neutral-vs-emotional contrast statistic. `neutral.best.jlens` = `["only","team","each","dates","table","teams","next","all"]` shows the lens reads **topic**, not affect; nothing in these files distinguishes the two. They evidence that the lens decodes something contentful at mid layers — not that emotion specifically is verbalizable.

9. **Vocabulary construction.** First-subtoken-only ids (`jlens_coupling.py:40`, `jlens_contagion.py:38`) mean multi-token words contribute shared prefixes; "panic" is in both `afraid` and `desperate` (coupling lines 24, 29), and `all_ids` (contagion line 57) contains the target's own tokens, shrinking the contrast by construction.

10. **Layer 29 is the wrong layer by the authors' own data.** It was inherited from a tie-break artifact (`jlens_readout.py:58`), not from evidence; the probe's own mid layer was 31, and the readout file shows emotion content concentrated at 42/48.

## 4. What would make the claims defensible

- Put both lenses on one scale (apply `log_softmax` in `jlens_contagion.py:103` as `jlens_coupling.py:89` does), or report a scale-invariant statistic (rank of emotion tokens, AUC of emotion-vs-matched-control vocab, calibrated probability).
- Add a fitted control: tuned lens / affine translator trained on the same activations. Identity is a strawman.
- Report per-item distributions: n, SD, paired bootstrap or permutation CI on each delta, corrected across 6 emotions; ≥3 seeds. With n=16 and no dispersion, `calm.jlens_contagion` = +0.0549 is uninterpretable.
- Pre-register the layer, or report the full layer sweep with the selection rule stated; fix the `>` tie-break in `jlens_readout.py:58`.
- Break circularity: read the residual from B's *context* before B emits emotional text, and show the emotion is readable where B's output does not already contain it. Add a mismatched-emotion control (does the "afraid" score also rise under "happy" injection?).
- Restore or ship `results/generations/responses_qwen36.jsonl`, and write n into the coupling JSON.
- Drop the bf16 cast (`jlens_coupling.py:88`) or show the effect survives float32.
- Replace hand-picked prompts with a held-out labelled emotion corpus and a scored metric before claiming "verbalizable".

## 5. Verdicts

- **C1** ("verbalizable for 6/6, beating a logit-lens control") — **NOT SUPPORTED**: the 6/6 sign count holds in `jlens_contagion_qwen36-27b.json`, but the logit-lens control is also 6/6 positive and the two scores are unnormalized raw logits (line 103), so "beating" reduces to an uncontrolled scale difference.
- **C2** ("J-lens readouts on B move with A's injected emotion for 6/6") — **NOT SUPPORTED**: the matching file gives 4/6 (`afraid.jlens_coupling` −0.0682, `happy.jlens_coupling` −0.0212), n is unverifiable, and the input file `results/generations/responses_qwen36.jsonl` (line 56) does not exist.
- **C3** ("J-lens reads emotion words where the logit lens does not") — **SUPPORTED-WITH-CAVEATS**, qualitatively only: the logit lens yields emotion words in 0/32 readout cells versus the J-lens in 15/32, but at the reported layer 29 the J-lens itself is 3/8, the control is unfitted identity rather than a tuned lens, n is 8 hand-written prompts, and on the one scale-matched measurement (`jlens_coupling`) the logit lens scores emotion words *higher* in 6/6 emotions.

## Reconciliation
Agrees with AUDIT 10 (scales), 11 (missing input), 17 (6/6 vs the file). New mechanical findings: the contagion script omits `log_softmax` where the coupling script applies it; the "best layer" rule is a strict `>` on a list capped at 8, so layer 29 is a tie-break artifact; a bf16 cast bounds precision at the size of the reported deltas; first-subtoken vocabularies overlap across emotions. Verdicts: C1 NOT SUPPORTED, C2 NOT SUPPORTED, C3 SUPPORTED-WITH-CAVEATS (qualitative only).
