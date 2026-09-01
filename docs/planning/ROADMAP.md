# Roadmap: from workshop draft to main-conference submission

Two-stage target: **NeurIPS workshop** (checkpoint — the narrative must land here) →
**ICLR main conference** (same spine + breadth + rigor). The workshop version de-risks
and timestamps the story; the main-track version scales it.

> Deadlines below are approximate — **verify against the official CfPs before committing.**
> NeurIPS workshop CfPs typically post ~Aug and close ~Sep/Oct. ICLR full-paper deadline
> is typically ~late Sep/early Oct of the preceding year.

## The problem to beat
Current headline is a *null*: coupling is clean in representation/vocabulary but not
detectable in behavior. A well-powered null is a good workshop paper and a borderline
main-track paper. To lift it, the paper must become either (a) surprising-and-true, or
(b) a reusable tool. We pursue both.

## Thrust 1 — flip the null (highest leverage)  [`src/behavioral/behavioral_multiturn.py`]
Single-turn behavior may be the wrong probe: affective contagion is *cumulative*.
- **Multi-turn accumulation:** hold A in a steered state over K turns; at every turn
  measure B's present-emotion (manip) and B's cheat-probability (behavioral). Test
  whether cheat rises with turn under `desperate` but not `calm`/`neutral`
  (accumulation-contrast slope, bootstrap CI, scramble null).
- **Multi-agent (N≥3):** mood propagating around a chain/ring of agents.
- **Agentic stakes:** replace the A/B choice with a real tool-use task.
- Positive → *that* is the paper; the existing E0–E5/CMI/J-lens battery becomes the
  mechanism section explaining why. Flat cheat + rising manip → a strong powered bound.

## Thrust 2 — breadth + scaling  [rigor the main track requires]
Re-run the core battery (E0, E2-CI, CMI, J-lens) across ≥3 model families
(Qwen, Llama, + Mistral/Gemma/OSS-GPT) and sizes. Deliverable: **coupling (nats) vs.
model scale** curve, and a family-robustness table. A monotone curve is a strong figure.

## Thrust 3 — the framework as a contribution  [reusable tool]
Package CMI + activation-passing channel + Jacobian-lens readout as one named framework
for "what crosses between agents beyond text." Validate it on a **second latent**
(e.g. deception or confidence), not just emotion — generality is the pitch.

## Rigor upgrades (gate main-track acceptance regardless)
- **Pre-register** the behavioral test: power analysis + stopping rule, so the null
  becomes "bounded below X at 80% power."
- **Held-out emotions / scenarios** to rule out probe-overfit.
- **Human eval** on a transcript sample (does the caught emotion read as real?).

## Honesty guardrail
The behavioral claim was corrected to "not detectable" because that is what the data
showed. Do **not** soften it for a nicer story. The move is to *find* the positive with
the multi-turn/multi-agent design and report whatever it actually shows.

## Sequence
1. Run `src/behavioral/behavioral_multiturn.py` on 27B (turns=8, reps=4) → decide if the null flips.
2. In parallel, Thrust-2 breadth battery on the smaller models (cheap) for the scaling curve.
3. Thrust-3 second-latent validation.
4. Assemble NeurIPS-workshop writeup around whatever Thrust 1 shows; extend to ICLR.
