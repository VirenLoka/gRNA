# Guide RNA Optimisation Engine

A cross-attention transformer that mutates a low-efficacy sgRNA into a
high-efficacy one, trained by backpropagating through a **frozen DeepCRISPR
on-target CNN**.

```
seed guide (23nt) ──embed──┐
                           ├──> cross-attention transformer ──> logits [23, 4]
target DNA region ──embed──┘                                         │
                                                            PAM logit mask
                                                                     │
                                                    straight-through Gumbel-softmax
                                                                     │
                                                          hard one-hot [23, 4]
                                                                     │
        generator gradient  <────  frozen DeepCRISPR  <───────────────┘
```

Only the generator updates. DeepCRISPR is frozen but remains inside the autograd
graph, so ∂efficacy/∂generator is well defined.

## Quick start

```bash
python -m grna_opt.convert_checkpoint --model ontar_cnn_reg_seq
```

```bash
python run.py
```

Override the target region or the seed without touching the config:

```bash
python run.py --target-file my_region.fa --steps 4000 --device cuda
```

## Where the negatives and targets come from

Default: `DeepCRISPR/examples/eg_reg_off_target.repiotrt`.

There is a naming trap in the DeepCRISPR data. In the **on-target** files the
column the README calls "Target Seq" *is* the 23nt guide+PAM — there is no
separate target DNA region in them. Only the **off-target** files carry two
distinct sequences per row:

| Column | Meaning | Role here |
|---|---|---|
| `Target Seq` | the sgRNA's intended on-target site | **target DNA** |
| `Off-target Seq` | a mismatched genomic site that fails to cut | **negative seed guide** |

So a row gives a genuine (guide, target) pair ~5–6 mismatches apart, which keeps
cross-attention non-degenerate. The default run picks:

```
seed   GAATTGTCTGGATTGTTTTCAGG   sg14 off-target site, predicted efficacy 0.0816
target GACTTGTTTTCATTGTTCTCAGG   sg14 intended site, 5 mismatches away
```

**Labels from these files are ignored.** Every label in the off-target
regression file is `0.0`, and it scores cleavage at the *off-target locus* — not
on-target efficacy, which is what our frozen scorer predicts. Ranking therefore
comes from the scorer itself (`rank_by: predicted`), which sorts all 100 rows by
predicted on-target efficacy (range 0.0816–0.4321) and takes the weakest.

All five example formats load via `grna_opt/data.py`; switch with
`seed_guide.examples.path`. `target.source: config` reverts to pasting your own
region into `config.yaml`, and `seed_guide.mode: dataset` uses the full paper
datasets instead (8101 HeLa guides, worst measured efficacy exactly 0.000000).

## Why the on-target **regression** model

Set by inspecting the checkpoints rather than the paper:

| Checkpoint | `e_1/w` | Head | Needs epigenetics? |
|---|---|---|---|
| `ontar_cnn_reg_seq` | `[1,3,`**`4`**`,32]` | `e_9/w [1,1,1024,1]` | **no** |
| `ontar_pt_cnn_reg` | `[1,3,`**`8`**`,32]` | `[1,1,1024,1]` | yes |
| `ontar_ptaug_cnn` | `[1,3,`**`8`**`,32]` | `[1,1,1024,`**`2`**`]` | yes |

1. **It is the only 4-channel on-target checkpoint.** There is no sequence-only
   classifier. Classification forces the 8-channel model, which needs CTCF,
   DNase, H3K4me3 and RRBS channels tied to a genomic locus — and a *mutated*
   spacer no longer has that locus. See `SPACER_OPTIMIZATION.md`.
2. **Gradient quality.** The classification head is post-softmax
   (`deepcrispr_src.py:160-164`) and saturates exactly when the optimiser is
   doing well. The regression head emits an unsaturated raw value
   (`deepcrispr_src.py:90-91`).
3. **Reward density.** Continuous [0, 1] labels score a 0.05 → 0.30 improvement;
   binary labels may not flip the class at all.

## Two findings that shaped the implementation

**The regression output is not a logit.** Despite the TF variable being named
`logits_l`, the head was trained with MSE directly against the [0, 1] efficacy
labels, so its raw output is already an efficacy estimate. Measured on held-out
paper data (n=300 per cell line):

| Cell line | Spearman | MAE (raw) | MAE (sigmoid) |
|---|---|---|---|
| HeLa | +0.894 | **0.038** | 0.311 |
| HCT116 | +0.889 | **0.035** | 0.327 |
| HL60 | +0.844 | **0.026** | 0.332 |

Applying a sigmoid would crush the range to ~[0.49, 0.73] and flatten the
gradient. Nothing in this codebase sigmoids the score. Those same numbers are the
port's correctness proof — a mis-ported CNN cannot track labels this closely.

**DeepCRISPR had to leave TensorFlow.** The checkpoints are TF 1.3 / Sonnet 1.9,
which needs Python 3.6, but more fundamentally TF1 and PyTorch share no autograd
tape — the gradient in step 3 above cannot cross that boundary. So the CNN is
reimplemented in `deepcrispr_torch.py` and the TF weights are loaded into it.
`tf_checkpoint.py` parses the checkpoint bundle in pure Python (SSTable index +
protobuf entries), so **neither this machine nor the GPU box needs TensorFlow
installed**.

Fidelity details that matter: every BatchNorm in the original runs as
`bn(x, False, test_local_stats=False)` — permanently inference-mode on moving
statistics, so the port is deterministic and has no training-mode BN path; the
encoder BNs carry neither beta nor gamma (a separate `beta_i` is added
post-normalisation); and Sonnet's `SAME` padding on the stride-2 layers is
**asymmetric** at widths 12→6 and 6→3, so PyTorch's `padding=1` would silently
shift the feature map.

## Layout

| File | Role |
|---|---|
| `config.yaml` | every knob; the only file you normally edit |
| `run.py` | end-to-end entry point |
| `grna_opt/tf_checkpoint.py` | pure-Python TF v2 checkpoint reader |
| `grna_opt/convert_checkpoint.py` | one-off TF → PyTorch conversion |
| `grna_opt/deepcrispr_torch.py` | differentiable port of the on-target CNN |
| `grna_opt/scorer.py` | frozen reward model |
| `grna_opt/encoding.py` | nucleotide ↔ one-hot (channel order matches `utils.py`) |
| `grna_opt/data.py` | seed-guide selection, target handling, PAM-site scanning |
| `grna_opt/generator.py` | cross-attention transformer + soft prompt |
| `grna_opt/gumbel.py` | straight-through Gumbel-softmax + annealing |
| `grna_opt/constraints.py` | PAM lock (structural) + optional soft penalties |
| `grna_opt/trainer.py` | training loop, metrics, checkpointing |

Each run writes `runs/<name>/` containing `config.resolved.yaml`, `run.log`,
`metrics.csv` (per-step), `summary.json`, and generator checkpoints.

## Design notes

**PAM lock is structural, not a penalty.** Logits at positions 21–22 are masked
to -inf on every base but `G`, so each sample is a valid SpCas9 site by
construction and no reward is spent learning to keep the PAM.

**`identity_bias` makes it a mutation search.** The output bias is initialised to
favour the seed's own bases, so the untrained model emits *exactly* the seed and
training searches for selective mutations away from it. Set it to `0.0` for an
unbiased random start (verified: `3.0` → 0 mismatches at init, `0.0` → 15).

**Temperature annealing.** `tau` runs 2.0 → 0.5. High early keeps the relaxation
smooth while the generator is random; low late makes the sample sharp so the
surrogate gradient matches what the scorer actually saw.

**`n_samples` cuts variance.** Straight-through is a biased estimator; averaging
the reward over several Gumbel samples per step steadies it.

## ⚠️ Reward hacking is real here

With PAM lock as the only active constraint the optimiser is free to drift
arbitrarily far from the seed, and it does. An 80-step run on the default
sg14 pair produced:

```
seed       GAATTGTCTGGATTGTTTTCAGG   efficacy = 0.082
optimised  GCGGGGGCGGGGGGGGGGTGTGG   efficacy = 0.469   (14 substitutions)
```

That is 18 G's out of 23, with per-position entropy collapsed to 0.046. It
scores well under the frozen CNN and is worthless in practice: poly-G forms
G-quadruplexes, is hard to synthesise, and at 14 mismatches no longer binds the
target at all. This is the expected failure mode of optimising against a fixed
learned reward, not a bug.

Switching the soft constraints on changes the outcome to a plausible edit:

```yaml
constraints: {edit_distance_weight: 0.02, seed_region_weight: 0.05,
              gc_weight: 0.5, homopolymer_weight: 0.5}
```

```
optimised  GAATTGGCTGGATGGTGTTCGGG   efficacy = 0.303   (4 substitutions)
                                     T7G, T14G, T17G, A21G
```

Treat those weights as a starting point, not a tuned setting. In that same run
entropy collapsed to 0.001 within ~25 steps and the reward sat pinned at the
seed's own score — the edit-distance term was dominating, and the 0.303 result
came from an early Gumbel sample rather than from learned improvement. Expect to
trade `edit_distance_weight` down against `training.entropy_weight` up. Both runs
above are 80-step CPU smoke tests and say nothing reliable about 2000-step
behaviour on a GPU.

Two further caveats worth stating plainly:

* The sequence-only model **has no genomic context** (`SPACER_OPTIMIZATION.md`).
  It scores intrinsic sequence-determined cutting propensity, not in-cell
  efficacy. It also never sees the target — the target shapes what the generator
  *conditions on*, not the reward.
* Nothing forces the optimised guide to still bind your target.
* With the paired setup the target is only **23nt**, so cross-attention has a
  short context to work with. `target.source: config` accepts a full region.

`constraints.py` implements the countermeasures; all are weighted `0.0` in the
shipped config per the chosen setup. Raise a weight to switch one on:

```yaml
constraints:
  edit_distance_weight: 0.02   # stay near the seed
  seed_region_weight:   0.05   # protect the PAM-proximal seed region
  gc_weight:            0.5    # keep spacer GC in [0.40, 0.70]
  homopolymer_weight:   0.5    # TTTT terminates Pol III transcription
```

Setting `seed_guide.mode: target_scan` additionally guarantees the starting guide
genuinely binds your target region.

## Requirements

`torch`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `scipy` — no TensorFlow.
