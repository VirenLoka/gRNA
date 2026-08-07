# Spacer Optimization with DeepCRISPR (On-Target)

## What DeepCRISPR predicts
An on-target efficacy score for a 23nt spacer+PAM sequence (SpCas9, NGG only). Two modes:

| Mode | Input | Model dir |
|---|---|---|
| Seq-only | 4-channel one-hot of the 23nt sequence | `ontar_cnn_reg_seq` |
| Full-featured | 4-channel one-hot + 4 binary epigenetic channels (CTCF, DNase, H3K4me3, RRBS) | `ontar_ptaug_cnn` / `ontar_pt_cnn_reg` |

Both are TF/Sonnet CNNs (`build_ontar_model` / `build_ontar_reg_model` in [deepcrispr_src.py](deepcrispr/deepcrispr_src.py)), classification or regression. Input tensor shape: `[batch, channels, 1, 23]`.

## Encoding
Already one-hot, not token IDs — no embedding layer exists in the graph.
- Sequence: `ntmap` in [utils.py](deepcrispr/utils.py) — A/C/G/T → 4-channel one-hot.
- Epigenetics: binary presence/absence per position (1 = signal, 0 = none), sourced from ENCODE at the genomic locus.

Externally-generated one-hot arrays (4 or 8 channel) can be fed directly, bypassing `Sgt`/`Episgt`.

## Key constraint for spacer optimization
The seq-only model's score depends **entirely** on the 23nt sequence — no notion of genomic context. It captures intrinsic sequence-determined cutting propensity (composition, PAM context, position effects), not true in-cell efficacy.

The full-featured model adds locus-specific chromatin state, which is known to affect real editing efficiency independent of sequence.

## Novel / unmapped sequences
Epigenetic channels require a genomic coordinate (cell-type-specific ENCODE tracks) — they cannot be derived from an isolated 23nt sequence. For spacers with no genomic locus (synthetic constructs, unmapped sequence):
- **CTCF**: reasonably predictable via motif scanning (PWM/FIMO).
- **DNase/accessibility**: partially sequence-predictable via sequence-to-epigenome models (Basset, Enformer-class), but need wider genomic context (~1kb+) and a target cell type — this is inference, not lookup.
- **H3K4me3, RRBS**: weakly sequence-determined; not reliably predictable from sequence alone.
- **Practical fallback**: use the seq-only model (`ontar_cnn_reg_seq`) when no genomic context exists rather than fabricating epigenetic channels.

## Open questions to resolve before building a pipeline
- Target cell type(s) for epigenetic tracks, if full-featured mode is used?
- Do candidate spacers always map to a known genome/coordinate, or do we need to support fully synthetic sequences?
- Classification (on/off) or regression (continuous efficacy) output needed?
