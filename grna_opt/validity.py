"""Hard biological-validity gates on generated spacers.

These are **gates, not penalties**.  A penalty is a gradient nudge applied after
the scorer already saw the sequence; a gate makes the violating sequence
unreachable, so it can never be generated and never reaches DeepCRISPR.

Scope: every rule here applies to the **spacer only** (positions 0-19).  The PAM
at positions 20-22 lives in the target DNA and is not part of the sgRNA
molecule, so RNA chemistry does not apply to it.  The PAM is handled separately
by the position-wise lock in :mod:`grna_opt.constraints`.

Why a prefix automaton
----------------------
The PAM lock is a static ``[23, 4]`` additive mask because "position 21 must be
G" is context-free.  Chemical rules are not: whether ``T`` is legal at position
i depends on what was sampled at i-3..i-1.  A precomputed mask cannot express
that, so sampling instead runs left to right and this automaton reports which
bases are legal at each step given the prefix committed so far.

Rule severity, and what is on by default
----------------------------------------
``max_t_run``
    A run of 4+ T's in the spacer becomes 4+ U's in the sgRNA, which terminates
    Pol III (U6/H1) transcription: the guide is truncated and never exists as
    designed.  This is the one true "molecule never gets made" failure, and it
    only applies when transcribing from a Pol III promoter -- irrelevant for
    synthetic/IVT guides.
``max_homopolymer_run``
    Long single-base runs cause synthesis errors and polymerase slippage, and
    poly-G in particular aggregates.

Off by default, one config line away
------------------------------------
``forbid_g_quadruplex``
    Bans the canonical G3N(1-7)G3N(1-7)G3N(1-7)G3 motif, which folds into a
    stable four-stranded structure that competes with the sgRNA fold and blocks
    Cas9 loading.  Note this is a genuinely *different* failure from a long G
    run: one continuous ``GGGGGGGGGG`` is a single tract and does not match the
    motif, while ``GGGAGGGAGGGAGGGA`` matches but has no run longer than 3.
    Catching both needs both rules.
``gc_min`` / ``gc_max``
    The *viability* band (roughly 0.20/0.85), not the 0.40-0.70 comfort band
    already available as a soft penalty.  Below ~20% the RNA:DNA duplex cannot
    hold a stable R-loop; above ~85% it will not unwind.  Enforced exactly by
    feasibility counting, never approximately.
``forbidden_kmers``
    Arbitrary k-mer blacklist.  The intended use is spacer-scaffold
    complementarity: seed it with every k-mer whose reverse complement occurs in
    the sgRNA scaffold, so the spacer cannot base-pair with its own scaffold and
    misfold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from .encoding import IDX_TO_NT, NT_TO_IDX, SPACER_LENGTH, VOCAB, VOCAB_SIZE

_NEG_INF = -1e9
_G = NT_TO_IDX["G"]
_T = NT_TO_IDX["T"]
_GC_IDX = (NT_TO_IDX["G"], NT_TO_IDX["C"])

# Canonical G-quadruplex (Quadparser) motif: four G-tracts of >=3, loops of 1-7.
G4_PATTERN = re.compile(r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}")
G4_MIN_TRACTS = 4
G4_MAX_LOOP = 7
G4_TRACT_LEN = 3


@dataclass
class ValidityRules:
    """Which hard gates are active.  Mirrors ``constraints.validity`` in config."""

    enabled: bool = True
    spacer_length: int = SPACER_LENGTH
    # Ban runs of length >= this. None disables.
    max_homopolymer_run: int | None = 5
    max_t_run: int | None = 4          # Pol III terminator; stricter than the above
    forbid_g_quadruplex: bool = False
    gc_min: float | None = None        # viability floor, e.g. 0.20
    gc_max: float | None = None        # viability ceiling, e.g. 0.85
    forbidden_kmers: tuple[str, ...] = ()

    def __post_init__(self):
        self.forbidden_kmers = tuple(k.strip().upper() for k in self.forbidden_kmers if k.strip())
        for kmer in self.forbidden_kmers:
            if set(kmer) - set(VOCAB):
                raise ValueError(f"forbidden k-mer {kmer!r} contains non-ACGT characters")
        if self.gc_min is not None and self.gc_max is not None and self.gc_min > self.gc_max:
            raise ValueError(f"gc_min ({self.gc_min}) exceeds gc_max ({self.gc_max})")

    @property
    def active(self) -> bool:
        """True when at least one gate would actually constrain sampling."""
        return self.enabled and any((
            self.max_homopolymer_run is not None,
            self.max_t_run is not None,
            self.forbid_g_quadruplex,
            self.gc_min is not None,
            self.gc_max is not None,
            bool(self.forbidden_kmers),
        ))

    def describe(self) -> list[str]:
        """Human-readable summary of the active gates, for the run log."""
        out = []
        if self.max_t_run is not None:
            out.append(f"Pol III terminator: T-runs >= {self.max_t_run} banned")
        if self.max_homopolymer_run is not None:
            out.append(f"homopolymer runs >= {self.max_homopolymer_run} banned")
        if self.forbid_g_quadruplex:
            out.append("canonical G-quadruplex motif banned")
        if self.gc_min is not None or self.gc_max is not None:
            lo = "0" if self.gc_min is None else f"{self.gc_min:.0%}"
            hi = "100" if self.gc_max is None else f"{self.gc_max:.0%}"
            out.append(f"spacer GC constrained to [{lo}, {hi}]")
        if self.forbidden_kmers:
            out.append(f"{len(self.forbidden_kmers)} forbidden k-mer(s)")
        return out


# --------------------------------------------------------------------------- #
# String-level validation (asserts, reporting, checking reference data)
# --------------------------------------------------------------------------- #

def check_sequence(sequence: str, rules: ValidityRules) -> list[str]:
    """Return a list of human-readable rule violations; empty means valid.

    Operates on the spacer portion only.  Accepts either a bare spacer or a full
    23-mer (spacer + PAM).
    """
    spacer = sequence[: rules.spacer_length].upper()
    problems: list[str] = []

    if rules.max_t_run is not None and re.search(f"T{{{rules.max_t_run},}}", spacer):
        run = max(len(m.group(0)) for m in re.finditer(r"T+", spacer))
        problems.append(f"poly-T run of {run} (Pol III terminator, limit {rules.max_t_run - 1})")

    if rules.max_homopolymer_run is not None:
        longest = max((m.group(0) for m in re.finditer(r"(.)\1*", spacer)), key=len, default="")
        if len(longest) >= rules.max_homopolymer_run:
            problems.append(
                f"homopolymer run of {len(longest)} ({longest[0]}), "
                f"limit {rules.max_homopolymer_run - 1}"
            )

    if rules.forbid_g_quadruplex and G4_PATTERN.search(spacer):
        problems.append("canonical G-quadruplex motif")

    if rules.gc_min is not None or rules.gc_max is not None:
        gc = sum(c in "GC" for c in spacer) / max(len(spacer), 1)
        if rules.gc_min is not None and gc < rules.gc_min:
            problems.append(f"GC {gc:.0%} below viability floor {rules.gc_min:.0%}")
        if rules.gc_max is not None and gc > rules.gc_max:
            problems.append(f"GC {gc:.0%} above viability ceiling {rules.gc_max:.0%}")

    for kmer in rules.forbidden_kmers:
        if kmer in spacer:
            problems.append(f"forbidden k-mer {kmer}")

    return problems


def assert_batch_valid(one_hot: torch.Tensor, rules: ValidityRules,
                       context: str = "pre-scorer") -> None:
    """Raise if any sequence in a generated batch violates the active gates.

    This is the tripwire that implements "nothing invalid reaches DeepCRISPR".
    With constrained sampling it should never fire; if it does, the automaton
    and the string checker have diverged, which is a bug worth failing loudly on.

    Deliberately *not* wired into every scorer call: reference sequences from
    ``DeepCRISPR/examples/`` legitimately violate these rules (50 of 610 do,
    including the default seed guide), and ranking real data must not crash.
    """
    if not rules.active:
        return
    indices = one_hot.argmax(dim=-1).detach().cpu()
    for row, idx in enumerate(indices):
        sequence = "".join(IDX_TO_NT[int(i)] for i in idx)
        problems = check_sequence(sequence, rules)
        if problems:
            raise AssertionError(
                f"[{context}] generated sequence {sequence} violates validity gates: "
                f"{'; '.join(problems)} -- this should be unreachable under "
                f"constrained sampling; the automaton and checker have diverged"
            )


def scaffold_forbidden_kmers(scaffold: str, k: int = 7) -> tuple[str, ...]:
    """Every k-mer whose reverse complement occurs in ``scaffold``.

    A spacer containing one of these can base-pair with its own sgRNA scaffold,
    which misfolds the guide and prevents Cas9 loading.  Feed the result into
    ``ValidityRules.forbidden_kmers``.
    """
    from .encoding import reverse_complement

    scaffold = scaffold.strip().upper()
    kmers = {reverse_complement(scaffold[i : i + k]) for i in range(len(scaffold) - k + 1)}
    return tuple(sorted(kmers))


# --------------------------------------------------------------------------- #
# Prefix automaton driving constrained sampling
# --------------------------------------------------------------------------- #

class ValidityAutomaton:
    """Vectorised left-to-right legality oracle over a batch of partial spacers.

    Usage per sampling pass::

        automaton.reset(batch_size, device)
        for i in range(length):
            allowed = automaton.allowed(i)      # [batch, 4] bool
            ...sample position i...
            automaton.advance(i, chosen_indices)

    All state is kept as ``[batch]`` integer tensors so the whole batch advances
    together; there is no Python loop over samples.
    """

    def __init__(self, rules: ValidityRules):
        self.rules = rules
        self.max_kmer = max((len(k) for k in rules.forbidden_kmers), default=0)
        self._kmer_set = set(rules.forbidden_kmers)
        self.relaxation_events = 0
        self._batch = 0

    def reset(self, batch: int, device) -> None:
        self._batch = batch
        self.device = device
        zeros = lambda: torch.zeros(batch, dtype=torch.long, device=device)
        self.last_base = torch.full((batch,), -1, dtype=torch.long, device=device)
        self.run_length = zeros()
        self.gc_count = zeros()
        # G-quadruplex chain state: completed G-tracts, and gap since the last
        # tract closed (only meaningful while a chain is alive).
        self.g4_tracts = zeros()
        self.g4_gap = zeros()
        self.history: list[torch.Tensor] = []

    # -- per-rule legality ------------------------------------------------- #

    def _homopolymer_allowed(self, allowed: torch.Tensor) -> torch.Tensor:
        rules = self.rules
        if rules.max_homopolymer_run is None and rules.max_t_run is None:
            return allowed
        for base in range(VOCAB_SIZE):
            limit = rules.max_homopolymer_run
            if base == _T and rules.max_t_run is not None:
                limit = min(limit, rules.max_t_run) if limit is not None else rules.max_t_run
            if limit is None:
                continue
            # Run length that placing `base` here would produce.
            continues = self.last_base == base
            new_run = torch.where(continues, self.run_length + 1,
                                  torch.ones_like(self.run_length))
            allowed[:, base] &= new_run < limit
        return allowed

    def _gc_allowed(self, position: int, allowed: torch.Tensor) -> torch.Tensor:
        """Exact feasibility: ban a base only if it makes the band unreachable."""
        rules = self.rules
        if rules.gc_min is None and rules.gc_max is None:
            return allowed
        length = rules.spacer_length
        remaining = length - position - 1
        lo = 0 if rules.gc_min is None else int(-(-rules.gc_min * length // 1))  # ceil
        hi = length if rules.gc_max is None else int(rules.gc_max * length)

        for base in range(VOCAB_SIZE):
            gc_after = self.gc_count + (1 if base in _GC_IDX else 0)
            # Overshoot the ceiling, or unable to reach the floor with what's left.
            feasible = (gc_after <= hi) & (gc_after + remaining >= lo)
            allowed[:, base] &= feasible
        return allowed

    def _g4_allowed(self, allowed: torch.Tensor) -> torch.Tensor:
        """Ban the G that would complete a 4th tract in a live chain."""
        if not self.rules.forbid_g_quadruplex:
            return allowed
        continues_g = self.last_base == _G
        new_run = torch.where(continues_g, self.run_length + 1, torch.ones_like(self.run_length))
        completes_tract = new_run == G4_TRACT_LEN
        chain_live = (self.g4_tracts >= G4_MIN_TRACTS - 1) & (self.g4_gap <= G4_MAX_LOOP)
        allowed[:, _G] &= ~(completes_tract & chain_live)
        return allowed

    def _kmer_allowed(self, allowed: torch.Tensor) -> torch.Tensor:
        if not self._kmer_set or not self.history:
            return allowed
        k = self.max_kmer
        if len(self.history) < k - 1:
            return allowed
        suffix = torch.stack(self.history[-(k - 1):], dim=1)  # [batch, k-1]
        suffix_str = [
            "".join(IDX_TO_NT[int(i)] for i in row) for row in suffix.detach().cpu()
        ]
        for row, prefix in enumerate(suffix_str):
            for base in range(VOCAB_SIZE):
                candidate = prefix + IDX_TO_NT[base]
                if any(candidate.endswith(bad) for bad in self._kmer_set):
                    allowed[row, base] = False
        return allowed

    # -- public API -------------------------------------------------------- #

    def allowed(self, position: int) -> torch.Tensor:
        """``[batch, 4]`` boolean mask of legal bases at ``position``."""
        allowed = torch.ones(self._batch, VOCAB_SIZE, dtype=torch.bool, device=self.device)
        if not self.rules.active or position >= self.rules.spacer_length:
            return allowed  # PAM positions carry no RNA chemistry

        allowed = self._homopolymer_allowed(allowed)
        allowed = self._gc_allowed(position, allowed)
        allowed = self._g4_allowed(allowed)
        allowed = self._kmer_allowed(allowed)

        # Defensive: with the default gates a dead end is impossible (at most one
        # base is ever banned).  The optional rules could in principle conflict,
        # so relax rather than emit an all -inf row that would produce NaNs.
        dead = ~allowed.any(dim=-1)
        if bool(dead.any()):
            self.relaxation_events += int(dead.sum())
            allowed[dead] = True
        return allowed

    def advance(self, position: int, chosen: torch.Tensor) -> None:
        """Commit the sampled bases at ``position`` and update state."""
        chosen = chosen.detach().reshape(-1)
        if position >= self.rules.spacer_length:
            return  # PAM positions do not affect spacer chemistry

        continues = self.last_base == chosen
        new_run = torch.where(continues, self.run_length + 1, torch.ones_like(self.run_length))

        if self.rules.forbid_g_quadruplex:
            is_g = chosen == _G
            completes = is_g & (new_run == G4_TRACT_LEN)
            chain_live = self.g4_gap <= G4_MAX_LOOP
            # A completed tract either extends a live chain or starts a new one.
            self.g4_tracts = torch.where(
                completes,
                torch.where(chain_live, self.g4_tracts + 1, torch.ones_like(self.g4_tracts)),
                self.g4_tracts,
            )
            # Inside a tract the gap resets; outside it grows and eventually
            # kills the chain.
            self.g4_gap = torch.where(is_g, torch.zeros_like(self.g4_gap), self.g4_gap + 1)
            self.g4_tracts = torch.where(self.g4_gap > G4_MAX_LOOP,
                                         torch.zeros_like(self.g4_tracts), self.g4_tracts)

        self.run_length = new_run
        self.last_base = chosen
        self.gc_count = self.gc_count + ((chosen == _GC_IDX[0]) | (chosen == _GC_IDX[1])).long()
        self.history.append(chosen)


def build_rules(config, spacer_length: int = SPACER_LENGTH) -> ValidityRules:
    """Construct :class:`ValidityRules` from a ``constraints.validity`` config."""
    return ValidityRules(
        enabled=config.enabled,
        spacer_length=spacer_length,
        max_homopolymer_run=config.max_homopolymer_run,
        max_t_run=config.max_t_run,
        forbid_g_quadruplex=config.forbid_g_quadruplex,
        gc_min=config.gc_min,
        gc_max=config.gc_max,
        forbidden_kmers=tuple(config.forbidden_kmers or ()),
    )
