"""Bradley-Terry ratings with an additive deck term.

Every specialist in this repo only ever plays its own deck.  Fit a plain
per-entry rating to that and you have measured *model strength plus deck
strength* and called it model strength — a1's policy and a1's decklist appear
only ever as a sum, so no number of games separates them.  The fit converges
regardless and prints two confident columns that are, individually, arbitrary.

The model here is

    P(i beats j) = sigma( ln10/400 * (R_i - R_j) ),   R_i = S_model(i) + D_deck(i)

with two constraints pinning the gauge freedoms:

    R_anchor = 1500          (Bradley-Terry is invariant to a global shift)
    sum_d D_d = 0            (S can absorb any constant taken out of D)

What makes S and D separable is that at least one model plays more than one
deck.  In this repo that is the generalist: ``checkpoints_generalist/ckpt-best``
was trained across the whole corpus and is what every specialist warm-starts
from, so it can legitimately pilot any of the six decks.  Entering it once per
deck ties the deck columns together — every difference between those entries is
pure deck strength, because the weights are identical.

The assumption that buys, stated out loud rather than buried: **the deck effect
is additive and model-independent** — a1's deck is worth +X to anybody.  If a1's
specialist exploits its own deck better than the generalist can, that surplus
lands in S, not D.  That is the honest reading and it is the interesting
quantity anyway.

Two failure modes return plausible garbage rather than raising, so both are
checked before the optimiser runs:

* a **disconnected comparison graph** — two islands of entries that never played
  each other have no common scale, and the fit will happily place them on one;
* a **disconnected model-deck incidence graph** — the "no bridge" case.  With
  every model on exactly one deck the design has one gauge direction per deck
  and only two constraints to spend, so the S/D split is decided by the
  optimiser's starting point.

Pure numpy: the log-likelihood's gradient and Hessian are closed-form under this
design, so a damped Newton loop converges in a handful of iterations and hands
back the exact Hessian the confidence intervals need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Bradley-Terry on the Elo scale: a 400-point gap is 10:1 odds.
SCALE = math.log(10.0) / 400.0
ANCHOR_RATING = 1500.0
Z95 = 1.959963984540054


@dataclass(frozen=True)
class EntrySpec:
    """One competitor: a (model, deck) pairing.

    The same *model* appears under several names when it is entered on more than
    one deck — that is the bridge, and it is the whole reason ``S`` and ``D`` are
    separable.  ``name`` is the arena identity; ``model`` and ``deck`` are what
    the rating decomposes into.
    """

    name: str
    model: str
    deck: str


@dataclass(frozen=True)
class Match:
    """Aggregate result of every game played between two entries.

    Draws are scored as half a win to each side, the standard rating-system
    convention, and are kept as their own field so the JSON record says how many
    there were rather than hiding them inside a fractional win count.
    """

    a: str
    b: str
    wins_a: int
    wins_b: int
    draws: int = 0

    @property
    def n(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    @property
    def score_a(self) -> float:
        return self.wins_a + 0.5 * self.draws


@dataclass(frozen=True)
class Estimate:
    """A fitted quantity with a 95% confidence interval."""

    value: float
    stderr: float

    @property
    def lo(self) -> float:
        return self.value - Z95 * self.stderr

    @property
    def hi(self) -> float:
        return self.value + Z95 * self.stderr

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "stderr": self.stderr,
                "lo": self.lo, "hi": self.hi}


@dataclass
class RatingTable:
    """The fit: per-entry ratings plus the S/D decomposition they came from."""

    entries: list[EntrySpec]
    rating: dict[str, Estimate]
    model_strength: dict[str, Estimate]
    deck_effect: dict[str, Estimate]
    anchor: str
    n_matches: int
    n_games: int
    log_likelihood: float
    n_iter: int
    converged: bool
    residual_deviance: float = 0.0
    covariance: np.ndarray | None = field(default=None, repr=False)

    def ranked(self) -> list[tuple[str, Estimate]]:
        return sorted(self.rating.items(), key=lambda kv: -kv[1].value)

    def as_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "scale": "elo (400 points = 10:1 odds)",
            "n_matches": self.n_matches,
            "n_games": self.n_games,
            "log_likelihood": self.log_likelihood,
            "residual_deviance": self.residual_deviance,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "entries": [
                {"name": e.name, "model": e.model, "deck": e.deck}
                for e in self.entries
            ],
            "rating": {k: v.as_dict() for k, v in self.rating.items()},
            "model_strength": {k: v.as_dict() for k, v in self.model_strength.items()},
            "deck_effect": {k: v.as_dict() for k, v in self.deck_effect.items()},
        }


class RatingError(ValueError):
    """The fit refuses to run — the data cannot identify what was asked for."""


# ── Identifiability checks ───────────────────────────────────────────────────


def _components(nodes: Sequence[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Connected components of an undirected graph, as sorted node lists."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def check_comparison_graph(entries: Sequence[EntrySpec],
                           matches: Sequence[Match]) -> None:
    """Every entry must be reachable from every other through played games.

    Two islands share no common scale.  The optimiser still returns a number for
    each — whatever the initial point put there — and the gap between islands is
    a fiction.
    """
    names = [e.name for e in entries]
    edges = [(m.a, m.b) for m in matches if m.n > 0]
    comps = _components(names, edges)
    if len(comps) > 1:
        summary = "; ".join(
            f"{{{', '.join(c[:4])}{', ...' if len(c) > 4 else ''}}} ({len(c)})"
            for c in comps
        )
        raise RatingError(
            f"comparison graph is disconnected into {len(comps)} components: "
            f"{summary}. Entries in different components never played each "
            "other, so no data relates their ratings — any gap the fit reports "
            "between them is the optimiser's starting point, not a measurement."
        )


def check_bridge(entries: Sequence[EntrySpec]) -> None:
    """The model-deck incidence graph must be connected — i.e. a bridge exists.

    Each connected component of (models + decks, one edge per entry) contributes
    its own gauge freedom: add c to every ``S`` in the component and subtract c
    from every ``D`` in it and the likelihood does not move.  Two constraints can
    pin at most two such directions and one of them is already spent on the
    global Bradley-Terry shift, so anything past a single component leaves the
    S/D split undetermined.

    In practice a second component means no model was entered on more than one
    deck: without the generalist bridging them, model strength and deck strength
    are the same column wearing two names.
    """
    models = sorted({e.model for e in entries})
    decks = sorted({e.deck for e in entries})
    nodes = [f"M:{m}" for m in models] + [f"D:{d}" for d in decks]
    edges = [(f"M:{e.model}", f"D:{e.deck}") for e in entries]
    comps = _components(nodes, edges)
    if len(comps) > 1:
        islands = [
            sorted(n[2:] for n in c if n.startswith("D:")) for c in comps
        ]
        raise RatingError(
            f"model/deck design is rank-deficient: {len(comps)} disconnected "
            f"groups, decks {islands}. No model plays decks from more than one "
            "group, so S_model and D_deck only ever appear as a sum within each "
            "group and the split between them is arbitrary. Enter one model "
            "(the generalist) on every deck to bridge them."
        )


def check_separation(entries: Sequence[EntrySpec],
                     matches: Sequence[Match]) -> None:
    """No entry may have won, or lost, every single game it played.

    Its maximum-likelihood rating is then +/-infinity; Newton walks off toward it
    and the standard errors that come back are meaningless.
    """
    score: dict[str, float] = {e.name: 0.0 for e in entries}
    total: dict[str, float] = {e.name: 0.0 for e in entries}
    for m in matches:
        if m.n == 0:
            continue
        score[m.a] += m.score_a
        total[m.a] += m.n
        score[m.b] += m.n - m.score_a
        total[m.b] += m.n

    perfect = [n for n in score if total[n] > 0 and score[n] == total[n]]
    winless = [n for n in score if total[n] > 0 and score[n] == 0.0]
    if perfect or winless:
        raise RatingError(
            "the fit is not identified — "
            + ", ".join(
                filter(None, [
                    f"{perfect} won every game played" if perfect else "",
                    f"{winless} lost every game played" if winless else "",
                ])
            )
            + ". A perfect record puts the maximum-likelihood rating at "
            "infinity; play more games, or drop the entry."
        )


# ── The fit ──────────────────────────────────────────────────────────────────


def _design(entries: Sequence[EntrySpec]) -> tuple[np.ndarray, list[str], list[str]]:
    """``X`` with ``R = X @ [S; D]``, plus the model and deck orderings."""
    models = sorted({e.model for e in entries})
    decks = sorted({e.deck for e in entries})
    mi = {m: k for k, m in enumerate(models)}
    di = {d: k for k, d in enumerate(decks)}
    X = np.zeros((len(entries), len(models) + len(decks)))
    for r, e in enumerate(entries):
        X[r, mi[e.model]] = 1.0
        X[r, len(models) + di[e.deck]] = 1.0
    return X, models, decks


def _constraints(models: Sequence[str], decks: Sequence[str],
                 anchor: EntrySpec) -> tuple[np.ndarray, np.ndarray]:
    """``(C, b)`` for ``C @ theta = b``: sum(D) = 0 and R_anchor = 1500."""
    n_m, n_d = len(models), len(decks)
    C = np.zeros((2, n_m + n_d))
    C[0, n_m:] = 1.0                                   # sum_d D_d = 0
    C[1, list(models).index(anchor.model)] = 1.0       # S_anchor + D_anchor
    C[1, n_m + list(decks).index(anchor.deck)] += 1.0
    return C, np.array([0.0, ANCHOR_RATING])


def _null_space(C: np.ndarray) -> np.ndarray:
    """Orthonormal basis for ``{v : C v = 0}``, as columns."""
    _u, s, vt = np.linalg.svd(C)
    tol = max(C.shape) * (s[0] if s.size else 0.0) * np.finfo(float).eps
    rank = int((s > tol).sum())
    return vt[rank:].T


def fit_bradley_terry(
    matches: Sequence[Match],
    entries: Sequence[EntrySpec],
    anchor: str,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> RatingTable:
    """Fit ``R_i = S_model(i) + D_deck(i)`` by maximum likelihood.

    Parameters
    ----------
    matches
        Aggregated head-to-head results.  Entries not appearing in any match
        are an error, not a zero-information default.
    entries
        The roster.  ``name`` must be unique; ``model``/``deck`` are what the
        rating is decomposed into.
    anchor
        Entry name pinned to 1500.

    Raises
    ------
    RatingError
        If the comparison graph is disconnected, if no model bridges the decks,
        if some entry has a perfect record, or if the Newton solve hits a
        singular Hessian — the four ways this returns confident nonsense.
    """
    if not entries:
        raise RatingError("no entries to rate")
    names = [e.name for e in entries]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise RatingError(f"duplicate entry names: {dupes}")
    index = {n: k for k, n in enumerate(names)}

    played = [m for m in matches if m.n > 0]
    if not played:
        raise RatingError(
            f"no games to fit: {len(matches)} match record(s), none with n > 0"
        )
    for m in played:
        for who in (m.a, m.b):
            if who not in index:
                raise RatingError(f"match references unknown entry {who!r}")
        if m.a == m.b:
            raise RatingError(f"match of {m.a!r} against itself")

    anchor_entry = next((e for e in entries if e.name == anchor), None)
    if anchor_entry is None:
        raise RatingError(f"anchor {anchor!r} is not in the roster")

    check_comparison_graph(entries, played)
    check_bridge(entries)
    check_separation(entries, played)

    X, models, decks = _design(entries)
    C, b = _constraints(models, decks, anchor_entry)
    theta0, *_ = np.linalg.lstsq(C, b, rcond=None)
    N = _null_space(C)
    if N.shape[1] == 0:
        raise RatingError("constraints leave no free parameters")

    ia = np.array([index[m.a] for m in played])
    ib = np.array([index[m.b] for m in played])
    sa = np.array([m.score_a for m in played], dtype=float)
    n = np.array([float(m.n) for m in played])

    # R = X (theta0 + N phi) = r0 + XN phi
    r0 = X @ theta0
    XN = X @ N

    def _ratings(phi: np.ndarray) -> np.ndarray:
        return r0 + XN @ phi

    def _loglik(phi: np.ndarray) -> float:
        d = SCALE * (_ratings(phi)[ia] - _ratings(phi)[ib])
        # log sigma(d) = -log1p(exp(-d)), stable for both signs.
        log_p = -np.logaddexp(0.0, -d)
        log_q = -np.logaddexp(0.0, d)
        return float(sa @ log_p + (n - sa) @ log_q)

    def _grad_hess(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = SCALE * (_ratings(phi)[ia] - _ratings(phi)[ib])
        p = 1.0 / (1.0 + np.exp(-d))
        resid = SCALE * (sa - n * p)             # d loglik / d (R_a - R_b)
        w = (SCALE ** 2) * n * p * (1.0 - p)     # curvature per match

        g_r = np.zeros(len(entries))
        np.add.at(g_r, ia, resid)
        np.add.at(g_r, ib, -resid)

        # H_R = -A^T diag(w) A with A[k] = e_ia - e_ib; build in phi space
        # directly: rows of (XN[ia] - XN[ib]) are A @ XN.
        A = XN[ia] - XN[ib]
        return XN.T @ g_r, -(A.T * w) @ A

    phi = np.zeros(N.shape[1])
    ll = _loglik(phi)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g, H = _grad_hess(phi)
        if np.max(np.abs(g)) < tol:
            converged = True
            break
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError as exc:
            raise RatingError(
                "Newton step hit a singular Hessian — the design does not "
                "identify these parameters even though the graph checks "
                f"passed ({exc})"
            ) from exc
        # Backtracking: the Newton direction is an ascent direction for a
        # concave log-likelihood, but the full step can overshoot early on.
        t, trial, ll_trial = 1.0, phi, ll
        while t > 1e-10:
            trial = phi + t * step
            ll_trial = _loglik(trial)
            if ll_trial >= ll:
                break
            t *= 0.5
        else:
            # No step size improves the objective: we are at the optimum to
            # within floating point, even though the gradient test has not
            # tripped yet.
            converged = True
            break
        phi, ll = trial, ll_trial
    if not converged and np.max(np.abs(_grad_hess(phi)[0])) < 1e-5:
        converged = True

    _g, H = _grad_hess(phi)
    try:
        cov_phi = np.linalg.inv(-H)
    except np.linalg.LinAlgError as exc:
        raise RatingError(f"Hessian is singular at the optimum ({exc})") from exc
    cov_theta = N @ cov_phi @ N.T
    theta = theta0 + N @ phi

    n_m = len(models)
    rating: dict[str, Estimate] = {}
    for k, e in enumerate(entries):
        x = X[k]
        var = float(x @ cov_theta @ x)
        rating[e.name] = Estimate(float(x @ theta), math.sqrt(max(var, 0.0)))

    model_strength = {
        m: Estimate(float(theta[k]), math.sqrt(max(float(cov_theta[k, k]), 0.0)))
        for k, m in enumerate(models)
    }
    deck_effect = {
        d: Estimate(float(theta[n_m + k]),
                    math.sqrt(max(float(cov_theta[n_m + k, n_m + k]), 0.0)))
        for k, d in enumerate(decks)
    }

    # Saturated log-likelihood minus ours, doubled: a rough fit diagnostic.
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = np.clip(sa / n, 1e-12, 1 - 1e-12)
        ll_sat = float(sa @ np.log(phat) + (n - sa) @ np.log1p(-phat))
    return RatingTable(
        entries=list(entries),
        rating=rating,
        model_strength=model_strength,
        deck_effect=deck_effect,
        anchor=anchor,
        n_matches=len(played),
        n_games=int(n.sum()),
        log_likelihood=ll,
        n_iter=it,
        converged=converged,
        residual_deviance=2.0 * (ll_sat - ll),
        covariance=cov_theta,
    )


def expected_score(r_a: float, r_b: float) -> float:
    """P(a beats b) under the fitted model."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def format_table(table: RatingTable, width: int = 46) -> str:
    """The three-column report: entry, rating with CI, and its S/D split."""
    by_name = {e.name: e for e in table.entries}
    lines = [
        "=" * (width + 46),
        f"{'entry':<{width}} {'rating':>16}  {'S_model':>10} {'D_deck':>9}",
        "=" * (width + 46),
    ]
    for rank, (name, est) in enumerate(table.ranked(), 1):
        e = by_name[name]
        s = table.model_strength[e.model].value
        d = table.deck_effect[e.deck].value
        anchor = " *" if name == table.anchor else ""
        lines.append(
            f"{rank:>3}. {name:<{width - 5}} {est.value:>7.0f} "
            f"±{Z95 * est.stderr:>5.0f}  {s:>10.0f} {d:>+9.0f}{anchor}"
        )
    lines.append("-" * (width + 46))
    lines.append("deck effects (sum to zero):")
    for d, est in sorted(table.deck_effect.items(), key=lambda kv: -kv[1].value):
        lines.append(f"     {d:<{width - 5}} {est.value:>+7.0f} ±{Z95 * est.stderr:>5.0f}")
    lines.append("-" * (width + 46))
    lines.append(
        f"{len(table.entries)} entries · {table.n_matches} pairs · "
        f"{table.n_games} games · anchor {table.anchor}=1500 · "
        f"{'converged' if table.converged else 'DID NOT CONVERGE'} "
        f"in {table.n_iter} iters"
    )
    return "\n".join(lines)
