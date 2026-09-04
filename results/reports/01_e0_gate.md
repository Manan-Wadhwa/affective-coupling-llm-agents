# 01 · E0 — present / other-speaker emotion gate

**Files:** `results/core/e0_qwen36-27b.json` · **Script:** `src/core/coupling_e0.py` · **Model:** Qwen/Qwen3.6-27B · **Status in registry:** QUOTABLE (`e0.present_acc`, `e0.cross_cos`)

## What was run
Crossed two-agent dialogues: A is instructed to feel one of six emotions and B another (6 × 6 cells), the model generates the dialogue, and the final A utterance's residual stream is mean-pooled at four depths (hidden_states 22 / 32 / 43 / 51; focus 43 = round(0.67 × 64)). Two multinomial logistic-regression decoders are fit on the pooled features: one for the *present* speaker's instructed emotion, one for the *other* speaker's. Accuracy is 5-fold stratified CV. n = 395 dialogues kept after leak/parse filtering. Labels are the instructed emotions, not measured ones.

## What the file shows
`focus_result` (layer 43): `present_acc` 0.899 · `other_acc` 0.643 · `present_loo` 0.838 · `cross_decode` 0.192 · `orthogonality_cross_cos` 0.044 · `within_present_cos` 0.193. Chance recorded as 0.167.

| hidden_states | present_acc | other_acc | cross_decode | cross_cos | within_present_cos |
|---|---|---|---|---|---|
| 22 | 0.873 | 0.633 | 0.208 | 0.055 | 0.193 |
| 32 | 0.894 | 0.701 | 0.200 | 0.047 | 0.197 |
| 43 | 0.899 | 0.643 | 0.192 | 0.044 | 0.193 |
| 51 | 0.899 | 0.643 | 0.190 | 0.048 | 0.195 |

No CI, no seed count, no per-fold scores, no retention counts in the file.

## What can be inferred
- The instructed present-speaker emotion is recoverable from the pooled residual at ~0.90 held-out accuracy at every depth tested. [supported by file]
- The *other* speaker's instructed emotion is also recoverable, at 0.63–0.70. [supported by file]
- A decoder trained on present labels does not transfer to other labels (cross-decode 0.19–0.21, near the nominal chance of 0.167). [supported by file]
- Matched-emotion present vs other coefficient directions have |cos| ≈ 0.04–0.06, versus 0.19 between different emotions within the present probe. [supported by file] Whether 0.04 is "near-orthogonal" depends on a reference distribution the file does not contain. [needs: permutation / isotropic null]
- Whether decoding reflects a latent affect representation rather than surface lexis is not testable from this file. [needs: text-only baseline on the same transcripts]
- Class balance after filtering is not recorded; the confusion matrix in the file shows unequal row sums. [needs: retention counts per cell]

## Status
Registry: both numbers QUOTABLE and resolving. AUDIT.md does not flag E0 (lists it as quotable as-is).

### Independent critique (blind: saw only the script and the JSON)

## C1 — "present-speaker emotion decodes at 0.90 (6-way, chance 0.167)"

**1. What the code computes.** Labels are the *instructed* emotions from the prompt, not measured ones: `dialogue_prompt` (L46–51) tells the model "A is feeling deeply eA", and `ep`/`eo` are stored verbatim as ground truth (L192, L205–206). There is no manipulation check anywhere. Generation is sampled (`do_sample=True, temperature=0.9`, L74) with **no torch seed** — `random.Random(0)` (L184) seeds only name/topic choice, `random_state=0` (L131) only the CV split. Filtering: `leak()` drops any dialogue naming either assigned emotion (L98–99); `parse_final_A` requires ≥3 speaker lines with the last by A (L86). Features: mean-pooled residual over tokens whose char offset starts at/after the final `"A:"` (L89–90, L115, L120). **Truncation at `max_length=512` (L108) can cut the final utterance off entirely; L117–118 then silently pools the whole sequence, B's lines included** — this count is never recorded. Layer: `focus = round(0.67*n_layers)` = 43 ⇒ 64 blocks; `hs[43]` indexes a tuple of L+1 whose element 0 is embeddings (L112), so "layer 43" is block 43's output, off-by-one vs 0-indexed naming. Decoder: StandardScaler + multinomial LR (C=0.5), `cross_val_score` with `StratifiedKFold(5, shuffle=True, random_state=0)` (L129–132) — genuinely held-out, but a single CV with a single seed and plain accuracy.

**2. JSON.** `focus_result.present_acc` = `0.8987341772151899` (= `by_layer.43.present_acc`), exactly 355/395 = trace/n of `focus_result.present_confusion`. `n` = 395. No std, no per-fold scores, no CI, no bootstrap, no `k`/`n_generated`/`n_leak`/`n_bad` — those are printed only (L194, L207). **Cutting against:** the design is fully crossed (L187–193), so present classes should be equal, yet row sums are [70, 75, 67, 67, 42, 74] — "afraid" lost ~45% of its cell, because `LEAK_WORDS["afraid"]` has 5 filter words vs 3 for others (L41). Hence **chance is not 0.167**: majority baseline = 75/395 = **0.190**, balanced accuracy = **0.883**, and afraid recall = 28/42 = **0.667**.

**3. Objections.** (a) No text-only baseline. Nothing here distinguishes an internal emotion representation from surface lexis/style; a TF-IDF LR on the same `transcript` field plausibly reaches ~0.9. Falsifiable in one run. (b) No group-aware CV (L131 is `StratifiedKFold`, not `GroupKFold`). Each cell draws from 6×12 = 72 (name, topic) combos with k ≥ 13 (max class 75/6), so identical prompts recur inside a cell and land in different folds. (c) The 512-token fallback (L117–118) can silently make the "present-speaker span" the whole dialogue. (d) Attrition is emotion-dependent, so the surviving "afraid" set is a biased subsample.

**4. To make it defensible.** Report balanced accuracy with a CI over ≥5 generation seeds and repeated CV; group folds by prompt; add a lexical baseline and a manipulation check; persist retention/fallback counts.

**5. Verdict: SUPPORTED-WITH-CAVEATS** — the 0.90 is exactly reproduced from held-out CV, but "chance 0.167" is wrong for this imbalanced sample (0.190; balanced acc 0.883), and surface-text decoding is not excluded.

## C2 — "present and other subspaces near-orthogonal, cos 0.04"

**1. What the code computes.** L152–157: a scaler is fit on **all** X (L152, no held-out split, unlike C1); `cp` = coefficients of a multinomial LR on present labels (L153), `co` the same for other labels (L154); rows are unit-normalized (L155); `cross_cos = mean |diag(cpu_ @ cou.T)|` (L157). That is the mean over only the **6 matched same-emotion pairs** (present-happy vs other-happy, …), not all 36 — the docstring at L19 ("mean |cos| between present-class dirs and other-class dirs") implies all pairs and does not describe the code. These are per-class discriminative coefficient rows, **not subspaces**: no span, no principal angles, no rank analysis. Fit in-sample at fixed C=0.5 on n=395 in a ≥4000-dim space, with no bootstrap, split-half, or permutation null.

**2. JSON.** `focus_result.orthogonality_cross_cos` = `0.04350200295448303` (= `by_layer.43`), which rounds to 0.04. Other layers: `by_layer.22` = 0.0549, `by_layer.32` = 0.0470, `by_layer.51` = 0.0482 — **the quoted 0.04 is the minimum of the four; layer 22 rounds to 0.05.** No CI, one seed, one run. **Cutting against:** `focus_result.within_present_cos` = `0.19343064725399017` — mutually exclusive classes of the *same* probe sit at |cos| ≈ 0.19; and the isotropic null for |cos| between random unit vectors is √(2/πH) ≈ 0.011–0.013 for H = 4096–5120 (hidden size is not recorded in the JSON, so this comparison is conditional on H). By that null, 0.0435 is **3–4× chance**, not "near-orthogonal".

**3. Objections.** (a) "Near-orthogonal" is asserted with no reference distribution. Against isotropic chance the value is 4× too large; against the within-probe figure of 0.19 it is small. Neither is stated, and the two references disagree about the conclusion. Falsifiable by permuting `yo` and recomputing L153–157. (b) `yp` and `yo` are independent **by construction** (fully crossed design, L187–193), so uncorrelated discriminative directions are the design's null expectation, not a discovered property of the model; the finding worth reporting would be a *departure* from orthogonality. (c) No stability check: the directions are estimated from 395 points in ≥4000 dims. If the split-half self-cosine of, e.g., present-"sad" is itself modest, a cross-cos of 0.04 is indistinguishable from estimation noise — two unreliably estimated vectors are near-orthogonal by default. (d) Only 6 of 36 pairs enter, so a strong present-afraid vs other-desperate coupling would be invisible — and the confusion matrix shows exactly that pair is the entangled one (11/42 afraid→desperate). (e) The multinomial/L2 fit constrains rows toward summing to zero, biasing all row-row cosines away from +1 regardless of the data.

**4. To make it defensible.** Report principal angles between the spans of the two 6-row coefficient sets; a permutation/bootstrap null and CI; split-half reliability per direction; all 36 pairwise cosines; directions fit out-of-sample. Nearest defensible wording: *"matched-emotion present/other probe directions have mean |cos| 0.043 at the focus layer (0.047–0.055 across layers), ~4× an isotropic null and well below the 0.19 within-probe reference."*

**5. Verdict: NOT SUPPORTED as worded** — the value exists in the JSON, but it is a mean over 6 matched coefficient rows, not subspaces, and "near-orthogonal" is unjustified without a null (it is 3–4× isotropic chance, and the file's own `within_present_cos` = 0.19 is the only comparison offered).

## Reconciliation
The blind critique goes beyond AUDIT.md on both claims, and neither point is in the registry today:
- **C1.** The leak filter drops more "afraid" dialogues than others (5 filter words vs 3), so the sample is imbalanced (row sums 70/75/67/67/42/74). Majority-class chance is 0.190, not 0.167; balanced accuracy is 0.883. The 0.90 still stands as a held-out number. Recommend: report balanced accuracy and the retention counts alongside it.
- **C2.** `orthogonality_cross_cos` is the mean over the six *matched-emotion* pairs of unit-normalised coefficient rows, not a subspace angle, and the file offers no null. Against an isotropic null (≈0.011–0.013 at d = 5120) the value is 3–4× chance; against the within-probe 0.19 it is small. Recommend: reword `e0.cross_cos` to the critique's "nearest defensible" sentence and add a permutation null before quoting "near-orthogonal". Adjudication is the authors' call; nothing here was edited.
