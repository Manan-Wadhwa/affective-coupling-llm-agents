# Citation validation, round 2 (paper_rev3.tex, references.bib)

Validator: automated check on 2026-09-06. Every entry was fetched at its arXiv/JMLR/publisher page (one fetch per entry) and three sources were read in full text where the paper attributes a specific number or claim (Marks & Tegmark, Herbster et al., Gao et al.). Items marked "from validator's knowledge" were not fetched this session and should be spot-checked before adding to the bib.

## A. Per-entry table

| key | real? | confirmed URL | characterisation correct? | necessary? |
|---|---|---|---|---|
| marks2023geometry | yes. Published at COLM 2024 (bib says arXiv preprint; v-latest 19 Aug 2024) | https://arxiv.org/abs/2310.06824 | yes, at all three sites (l.23, l.38, l.64); quotes in B | yes |
| gao2026raptor | yes (arXiv 2602.00158v2, 4 Feb 2026) | https://arxiv.org/abs/2602.00158 | yes: "directional stability" is named as a requirement, the method is validation-tuned ridge logistic regression, and RAPTOR's mean pairwise absolute cosine under 20% data drops spans exactly 0.87 (STSA/L3.1-8B) to 0.98 (Sarcasm/Q2.5-7B) over six rows, K=20 runs | yes |
| herbster2026steering | yes. Published at COLM 2026 (bib says arXiv preprint; v3 23 Aug 2026) | https://arxiv.org/abs/2604.08169 | numbers correct (0.99 Llama, 0.98 Qwen, compassion); characterisation incomplete, see B | yes |
| braun2025unreliability | yes (arXiv 2505.22637v1, 28 May 2025, preprint). A paper titled "Understanding Unreliability of Steering Vectors in Language Models: Geometric Predictors and the Limits of Linear Approximations" (arXiv 2602.17881) appears to be a longer version; check authorship and cite the later one if so | https://arxiv.org/abs/2505.22637 | yes; verbatim match. Minor: "5-30 draws" should read "5-30 training samples" (source: "few (5-30) randomly sampled training activations") | yes |
| raju2026geometric | yes (arXiv 2601.09173v5, 6 Jul 2026) | https://arxiv.org/abs/2601.09173 | yes: the Shesha metric compares RDMs built from complementary halves of the feature dimensions, "not ... across seeds, data splits, or training runs" | no, decorative (see C) |
| rimsky2023caa | yes; arXiv shows first author as Panickssery, matching the bib author field (key uses the earlier surname, harmless) | https://arxiv.org/abs/2312.06681 | yes (l.21 steering reference; l.62 correctly says CAA uses the plain mean difference: "averaging the difference in residual stream activations between pairs") | yes |
| zou2023repe | yes (v4, 3 Mar 2025; full author list has 21 names, bib truncates with "and others", acceptable) | https://arxiv.org/abs/2310.01405 | yes | yes |
| turner2023actadd | yes; title matches the current v5 (10 Oct 2024) | https://arxiv.org/abs/2308.10248 | yes | yes |
| li2023iti | yes. NeurIPS 2023 spotlight (bib says arXiv preprint) | https://arxiv.org/abs/2306.03341 | yes, as a steering reference; note ITI intervenes on attention heads, not the residual stream | yes (borderline; the other three cover the residual-stream method actually used) |
| soudry2018implicit | yes. JMLR 19(70):1-57, 2018, exactly as in bib | https://jmlr.org/papers/v19/18-188.html | yes at l.38 (attribution); imprecise at l.23 and l.62, see B | yes |
| durupinar2026affect | yes (arXiv 2607.25140, 27 Jul 2026) | https://arxiv.org/abs/2607.25140 | partly wrong at l.21, see B | yes (the closest prior work to the paper's question; deserves a sentence of engagement, not just a list entry) |
| long2025evoemo | yes (arXiv 2509.04310v4, 26 May 2026) | https://arxiv.org/abs/2509.04310 | yes: two LLM agents in multi-turn price negotiation with evolved emotion policies | decorative but acceptable |
| benjamini1995fdr | yes. JRSS-B 57(1):289-300, DOI 10.1111/j.2517-6161.1995.tb02031.x | https://academic.oup.com/jrsssb/article/57/1/289/7035855 | yes | yes |
| efron1993bootstrap | yes. Chapman & Hall, New York, 1993, xvi+436 pp | https://archive.org/details/introductiontobo0000efro | yes (percentile bootstrap); the scenario-blocked (cluster) bootstrap is not specifically covered there, see D10 | yes |

Counts: 14 real, 0 unconfirmed.

Bib metadata to update: marks2023geometry -> COLM 2024; li2023iti -> NeurIPS 2023; herbster2026steering -> COLM 2026; braun2025unreliability -> check 2602.17881. rimsky2023caa was, to the validator's knowledge, published at ACL 2024 (the arXiv page shows no venue note; verify before changing).

## B. Incorrect or incomplete characterisations

### B1. durupinar2026affect at l.21 (partly wrong)

Paper: "Systems of several language-model agents built from one model family are now common, and the agents talk to each other \citep{durupinar2026affect,long2025evoemo}."

Source (abstract): "Each agent perceives its neighbors through visual, auditory, and tactile channels, then appraises these perceptions ... The results show that the system produces emotional contagion dynamics ..." The architecture "contains no hand-authored mechanism for directly transferring affective state between agents; instead, inter-agent influence arises through the perception-appraisal-expression loop."

Problems: (i) Durupinar's agents do not converse; they perceive and express in a crowd simulation. (ii) A single July-2026 preprint does not support "are now common". (iii) "built from one model family" is not something either cited paper is about.

Fix: split the sentence. "Multi-agent systems of language models are common [general multi-agent reference, D6], and affect has begun to be studied as it moves between LLM agents, by expression in crowd simulations \citep{durupinar2026affect} and by emotion policies in two-agent negotiation \citep{long2025evoemo}. Whether anything of one agent's state reaches the other's internal representation is untested." This also turns the two citations from decoration into the paper's motivation.

### B2. herbster2026steering at l.38 and l.62 (numbers right, picture incomplete)

Paper (l.38): "\citet{herbster2026steering} report mean-difference and logistic directions near-identical (0.98--0.99) for compassion; in a quantity close to theirs we find 0.43 and 0.33--0.37." (l.62 repeats the 0.98--0.99.)

Source (Appendix A.3): "For compassion the two methods recover nearly identical directions on both models (mean cosine similarity 0.99 on Llama, 0.98 on Qwen), and Llama honesty agrees almost as well (mean 0.94). Qwen honesty is the clear exception (mean 0.80), consistent with the weaker separability of honesty embeddings for this model (Fig. 1). When the positive and negative distributions overlap more, the logistic regression decision boundary rotates to maximize classification accuracy, yielding a direction that diverges from the simple mean difference."

Problems: (i) Herbster et al. also report a divergent case (0.80) and offer a mechanism (class overlap) that is the opposite of this paper's (perfect separation drives divergence); the paper presents them only as finding near-identity. (ii) Herbster's "Qwen" is Qwen3.6-27B, the same model as this paper (their p.1: "two architectures (Llama-3.3-70B-Instruct and Qwen3.6-27B)"), which makes the disagreement sharper and worth saying. (iii) Herbster's training set is small (50 prompts x 5 responses per class for compassion, Sec. 4); this paper's own Figure 2 shows the logistic-vs-mean-difference cosine starting near 0.97 at small n and falling with n, so the two results may not conflict at all. The sentence should state the n at which 0.43 was measured (\rid{a2f.caa\_raw\_vs\_logreg\_raw}) and note that Herbster's n sits where this paper's curve is still high.

Fix: "\citet{herbster2026steering} report the two directions near-identical for compassion on Llama-3.3-70B and on the same Qwen3.6-27B we use (0.99, 0.98), but 0.80 for Qwen honesty, which they attribute to class overlap. At n = [state it] we find 0.43 and 0.33--0.37; their training sets are of the size at which our cosine is still above 0.9 (Figure 2), so the results are consistent with drift in n rather than in conflict."

### B3. soudry2018implicit at l.23 and l.62 (adjacent theorem, imprecise gloss)

Paper (l.23): "the reason is not the sample size or the penalty but separability, so the fitted row is a tie-break rather than an estimate \citep{soudry2018implicit,marks2023geometry}." (l.62): "the direction is the penalty's tie-break among separating hyperplanes."

Source (Soudry et al., abstract): "We examine gradient descent on unregularized logistic regression problems ... on linearly separable datasets. We show the predictor converges to the direction of the max-margin (hard margin SVM) solution."

Problem: Soudry et al. analyse unregularised gradient descent dynamics. The paper's fits are l2-penalised (C = 0.5 ... 500) solved by lbfgs to convergence, and the paper explicitly declines to claim the max-margin limit. The result that matches the paper's setting, namely that the l2-penalised logistic solution's direction is determined by the penalty and converges to the max-margin direction as the penalty vanishes, is Rosset, Zhu and Hastie (2004) (D2). Soudry is fine as the reference Marks & Tegmark themselves cite, but "tie-break" is the paper's gloss and should cite the regularisation-path result.

Fix: keep Soudry at l.38 (correct attribution of what Marks noted); at l.23 and l.62 cite Rosset et al. 2004 alongside or instead.

### B4. braun2025unreliability at l.38 (wording only)

Paper: "vary widely from 5--30 draws and agree above 0.99 from 200--500". Source: "With few (5-30) randomly sampled training activations ... when drawing many (200-500) training activations, the intra prompt type variance disappears (cosine similarity >0.99)". Replace "draws" with "training samples". Also note the source sentence is from a limitations paragraph, not a headline result; "corroborate" is fair.

### Verified correct (quotes for the record)

- marks2023geometry, l.38 "identified the deficiency and proposed mass-mean probing, noting that logistic regression converges toward the maximum-margin separator on separable data, the implicit bias analysed by Soudry": source "We call the probes p_mm and p_mm^iid mass-mean probes." and "Assuming for simplicity linearly separable data, LR instead converges to the maximum margin separator [Soudry et al. (2018)]."
- marks2023geometry, l.64 "the coefficient row is a fine classifier and a poor direction, the dissociation Marks described": source "Mass-mean probes are about as accurate for classification as LR, while also identifying directions which are more causally implicated in model outputs."
- gao2026raptor, l.38: source "directional stability: the learned direction must remain consistent under minor training perturbations (e.g., resampling, dataset variations, or mild distribution shifts)"; "a simple l2-regularized logistic probe whose validation-tuned ridge strength yields concept vectors from normalized weights"; Table 2 RAPTOR means 0.87, 0.92, 0.88, 0.97, 0.92, 0.98; protocol "we randomly drop 20% of examples, then re-split ... K=20". The paper's remark that "two 80% subsamples share most" is right (any two runs share at least 60% of the pool).
- raju2026geometric, l.38 "measure stability across feature dimensions, a different axis": source defines the metric over "complementary halves of its feature dimensions".

Counts: 1 mischaracterised (durupinar), 2 incomplete/imprecise (herbster, soudry), 1 wording (braun).

## C. Unnecessary citations

- raju2026geometric (l.38): cited only to say it measures something else. The sentence carries no information for the reader; drop it or fold into a footnote. It is the only purely decorative entry.
- long2025evoemo (l.21): supports "agents talk to each other" and nothing more; acceptable only if the sentence is rewritten as in B1 so that it becomes motivation.
- li2023iti (l.21): one of four steering references; harmless, but it is a head-level intervention, and the paper's method is residual-stream addition, so turner/rimsky/zou suffice.

Count: 1 unnecessary (raju), 2 borderline.

## D. Missing citations

Needed (a claim or method in the paper has no citation):

D1. l.64 "Separation is guaranteed for generic data whenever n<d". Cover, T. M. (1965), Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition, IEEE Trans. Electronic Computers EC-14(3):326-334. Overview: https://en.wikipedia.org/wiki/Cover%27s_theorem ; note: https://www.cns.nyu.edu/~eorhan/notes/covers-theorem.pdf . For the sharper high-dimensional statement (separability, hence non-existence of the unpenalised MLE, with probability tending to 1 when d/n > 1/2 for Gaussian features): Candes, E. J. and Sur, P. (2020), The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression, Ann. Statist. 48(1):27-42, https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-1/The-phase-transition-for-the-existence-of-the-maximum-likelihood/10.1214/18-AOS1789.full . The Candes-Sur result also strengthens the paper's point: separation is not only an n<d phenomenon but persists up to n ~ 2d.

D2. l.23 and l.62 "the direction is the penalty's tie-break among separating hyperplanes ... Weakening the penalty ... leaves the direction unchanged". Rosset, S., Zhu, J. and Hastie, T. (2004), Boosting as a regularized path to a maximum margin classifier, JMLR 5:941-973, https://jmlr.org/papers/v5/rosset04a.html (companion: Margin maximizing loss functions, NIPS 2003). From validator's knowledge, not fetched this session. This is the result for l2-penalised logistic regression on separable data, which is the paper's actual estimator.

D3. l.116 (B1c) "Projecting the difference-of-means emotion direction out of A's span at every hidden state 13--63". This is directional ablation with a difference-in-means direction, the pipeline of Arditi, A. et al. (2024), Refusal in Language Models Is Mediated by a Single Direction, NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf (arXiv 2406.11717). It also belongs in "Relation to prior work" as the widely copied DoM-plus-ablation design whose control register the paper is proposing to change.

D4. l.27 and l.126 "run the same estimator on permuted labels at the rank of the intervention". The antecedent is the control task of Hewitt, J. and Liang, P. (2019), Designing and Interpreting Probes with Control Tasks, EMNLP 2019, https://aclanthology.org/D19-1275/ ; survey of probe controls: Belinkov, Y. (2022), Probing Classifiers: Promises, Shortcomings, and Advances, Computational Linguistics 48(1):207-219, https://aclanthology.org/2022.cl-1.7/ . Both from validator's knowledge. The paper's contribution (a permuted-label control for interventions, not just for probes) should be stated relative to Hewitt & Liang.

D5. l.21 "steering vectors to induce a state in the sender" and l.42 (the emotion set desperate/afraid/happy/calm/sad/angry). The paper cites no emotion-steering work. Found this session:
  - Emotion Concepts and their Function in a Large Language Model, arXiv 2604.07729, https://arxiv.org/pdf/2604.07729 . Reports emotion vectors whose steering changes behaviour, including "desperate" and "calm", two of this paper's six emotions; directly relevant to the manipulation check and to the choice of emotions.
  - Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison, arXiv 2604.04064, https://arxiv.org/html/2604.04064 . Compares extraction methods for emotion directions, i.e. the estimator question of Section 3 applied to emotions.
  - Controllable Affective Generation via Latent Vector Steering, arXiv 2608.25569, https://arxiv.org/abs/2608.25569 (OpenReview https://openreview.net/forum?id=9N3ErwB3p4 ). CAA-style emotion vectors with dose control.
  At least the first two are needed; the paper should say how its emotion directions relate to theirs.

D6. l.21 "Systems of several language-model agents ... are now common". Needs a general reference, e.g. Guo, T. et al. (2024), Large Language Model based Multi-Agents: A Survey of Progress and Challenges, arXiv 2402.01680, https://arxiv.org/abs/2402.01680 , or Park, J. S. et al. (2023), Generative Agents, arXiv 2304.03442, https://arxiv.org/abs/2304.03442 (both from validator's knowledge). For affect specifically: Large Language Models have Chain-of-Affective (LLMs-CoA), arXiv 2512.12283, https://arxiv.org/html/2512.12283v1 , which argues that "LLM affect becomes a system-level property shaped by group composition" in multi-agent ecosystems; found this session.

Recommended (closely related work the paper should engage):

D7. l.64 "the coefficient row is a fine classifier and a poor direction". Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models, arXiv 2606.24952, https://arxiv.org/pdf/2606.24952 (found this session; reports that the most-decodable probe direction is far weaker as a causal control than the mean difference). Theory for mean difference over logistic/PCA directions: Belrose, N. (2023), Diff-in-means concept editing is worst-case optimal, https://blog.eleuther.ai/diff-in-means/ ; Im, S. and Li, Y. (2025), A Unified Understanding and Evaluation of Steering Methods, arXiv 2502.02716, https://arxiv.org/abs/2502.02716 ; Wu, Z. et al. (2025), AxBench, arXiv 2501.17148, https://arxiv.org/abs/2501.17148 (these three from validator's knowledge). Together they make the DoM-vs-logistic contrast of Section 3 part of an existing argument, which the "Relation to prior work" paragraph currently presents as involving only Marks and RAPTOR.

D8. l.38 Braun follow-up: Understanding Unreliability of Steering Vectors in Language Models: Geometric Predictors and the Limits of Linear Approximations, arXiv 2602.17881, https://arxiv.org/pdf/2602.17881 . Check whether it is the same group's extended version and cite whichever is current.

D9. l.118 (B1d) rank-5 class-mean subspace removal. Subspace concept erasure has its own literature: Ravfogel, S. et al. (2020), Null It Out (INLP), ACL 2020, https://aclanthology.org/2020.acl-main.647/ ; Belrose, N. et al. (2023), LEACE: Perfect linear concept erasure in closed form, NeurIPS 2023, https://arxiv.org/abs/2306.03819 (both from validator's knowledge). For multi-dimensional concept subspaces in steering: There Is More to Refusal in Large Language Models than a Single Direction, arXiv 2602.02132, https://arxiv.org/html/2602.02132v1 (found this session). LEACE in particular explains why a class-mean subspace shifts the readout at dose 0 (the instrument failure of B1d): erasing the class means also erases the readout's own signal.

D10. l.42 "scenario-blocked percentile bootstrap (29 clusters)". Efron & Tibshirani covers the percentile interval, not the cluster bootstrap. Add Davison, A. C. and Hinkley, D. V. (1997), Bootstrap Methods and their Application, CUP, or Field, C. A. and Welsh, A. H. (2007), Bootstrapping clustered data, JRSS-B 69(3):369-390 (from validator's knowledge). With 29 clusters the percentile interval's coverage is itself a limitation worth one sentence.

D11. l.64 PCA-100 regime (fits stop separating at n >= 600). The implicit bias of gradient descent on nonseparable data is characterised by Ji, Z. and Telgarsky, M. (2019), COLT 2019, https://proceedings.mlr.press/v99/ji19a/ji19a.pdf (found this session); it is the natural companion to Soudry for the regime in which the paper's logistic direction becomes reproducible.

D12. l.23 "Those tools rest on one object, a direction in activation space". Park, K., Choe, Y. J. and Veitch, V. (2024), The Linear Representation Hypothesis and the Geometry of Large Language Models, ICML 2024, https://arxiv.org/abs/2311.03658 (from validator's knowledge). Optional.

Search note: a query for prior work on disjoint-half (split-half) reproducibility of probe directions returned nothing that measures that quantity; the nearest are self-consistency metrics (mean pairwise cosine over resamples), e.g. arXiv 2602.08159. This is consistent with the paper's claim that the disjoint-half quantity is its contribution, but the claim would be safer phrased as "we have not found" rather than left implicit.

Counts: 6 needed (D1-D6), 6 recommended (D7-D12).

## E. Verdict

All fourteen references are real and, with one exception (Durupinar, whose agents do not converse and whose single preprint cannot show multi-agent systems are "common"), the paper attributes to them things they actually say, though the Herbster comparison omits the divergent 0.80 case and the shared Qwen3.6-27B model, and the "tie-break" argument cites an unregularised-GD theorem where the l2-regularisation-path result (Rosset et al. 2004) is the one that fits. The bibliography is thin for what the paper does: the directional-ablation design (Arditi et al.), the permuted-label control's antecedent (Hewitt & Liang), the separability guarantee (Cover; Candes & Sur) and any emotion-steering or general multi-agent reference are absent, and adding the six needed items plus updating four venues would bring it to a defensible state.
