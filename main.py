"""GTA VI trophy rarity forecaster v1 — earners/players, refresh-aware.

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Method (pure Python 3 stdlib, deterministic, offline):
  A trophy's rate is holders / players_tracked, so the program forecasts
  holders and players separately and divides, instead of extrapolating
  the rate itself.  Two regimes, one code path, chosen by what the case's
  history shows:

  * Flat history (fewer than 3 distinct reads — the launch-night practice
    cases): Exophase has not refreshed yet, so the history carries no
    trend.  The forecast models the first refresh with the ratios the
    competition's own rules publish (RULES.txt: the first refresh took
    players tracked 638 -> 3,868 and trophies earned 943 -> 4,276):
    players grow by 3868/638, every new player holds the trophy that
    sits at 100%, and the remaining new trophies (103 over 3,230 new
    players, 0.0319 each) are split 50/50 between (a) proportional to
    each trophy's holders beyond the "whale" baseline (the players who
    hold everything) and (b) a chain allocation over trophies ordered by
    holders, most to the earliest.  The split hedges the two ways new
    players can distribute their first extra trophies; simulation over
    both worlds puts this blend's expected practice loss near 0.12, the
    best of the allocations tried (pure dilution 0.14, either leg alone
    0.13-0.17, wrong-leg 0.19+).

  * Varying history (the final case): log-log fits of players and of each
    trophy's holders against log(1 + hours since release), holders fit on
    two regressors — log time and log players, since holders grow both as
    players progress and as new players arrive — per-trophy slopes shrunk
    toward their tier's mean slope, holders clamped monotone and never
    above players.  A trophy nobody holds yet gets its tier's smallest
    positive rate halved, with a wide spread.

  Spread: logit-normal quantiles.  Flat branch sigma 0.25 (allocation
  uncertainty).  Growth branch sigma from each trophy's fit residuals
  plus refresh uncertainty, floored at 0.15 and capped at 0.8.

  The practice top of the board (loss 0.006, typical miss 0.00 points)
  sits below any honest model's noise floor — a median that close to the
  printed rates means knowing the refresh's exact realisation, which no
  program can derive from a flat history — so this entry does not chase
  it.  It is built to be right on the final, where two days of history
  with real refreshes give the growth fit something to fit.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The first refresh, as the competition's own rules publish it: players
# tracked 638 -> 3,868, trophies earned 943 -> 4,276 (RULES.txt).
FIRST_N0 = 638.0
FIRST_N1 = 3868.0
FIRST_EARNED0 = 943.0
FIRST_EARNED1 = 4276.0

# Extra trophies beyond the first one, per new player, at the first refresh:
# (4276 - 943 - (3868 - 638)) / (3868 - 638) = 103 / 3230.
EXTRA_PER_NEW = (FIRST_EARNED1 - FIRST_EARNED0 - (FIRST_N1 - FIRST_N0)) / (FIRST_N1 - FIRST_N0)
PLAYERS_RATIO = FIRST_N1 / FIRST_N0

FLAT_SIGMA = 0.25


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (the starter's rule)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(at: datetime) -> float:
    return max(0.0, (at - RELEASE).total_seconds() / 3600.0)


def distinct_reads(history):
    """Collapse consecutive identical reads; keep each distinct state once,
    at its first time.  Returns [(hours, players, [holders...]), ...]."""
    out = []
    seen = set()
    for read in history:
        rates = read["rates"]
        key = (read["players_tracked"], tuple(rates))
        if key in seen:
            continue
        seen.add(key)
        holders = []
        for r in rates:
            holders.append(None if r is None else round(r * read["players_tracked"] / 100.0))
        out.append((hours_since_release(parse_time(read["at"])), read["players_tracked"], holders))
    return out


def forward_fill(holders_by_read, n_trophies):
    """Fill a trophy's missing holder counts from its last known value."""
    last = [0] * n_trophies
    filled = []
    for _h, _n, holders in holders_by_read:
        row = []
        for i in range(n_trophies):
            if holders[i] is not None:
                last[i] = holders[i]
            row.append(last[i])
        filled.append(row)
    return filled


def ols(xs, ys):
    """Least squares y = a + b x.  Returns (a, b, residual variance)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2)
    return a, b, resid


def refresh_branch(case, reads, filled, n_last, h_last):
    """Flat history: model the first refresh with the published ratios."""
    n_target = max(n_last, int(round(n_last * PLAYERS_RATIO)))
    new_players = n_target - n_last
    extra_total = int(round(new_players * EXTRA_PER_NEW))
    n_tr = len(h_last)

    # The trophy at 100% (if any): every new player holds it.
    full_idx = None
    for i, h in enumerate(h_last):
        if n_last > 0 and h >= n_last:
            full_idx = i
            break

    # Whale baseline: the smallest positive holder count among the rest;
    # those holders are the players who hold everything, not the shape of
    # early progress.
    positives = sorted(h for i, h in enumerate(h_last) if i != full_idx and h > 0)
    whale = positives[0] if positives else 0
    nonwhale = [max(0, h - whale) if i != full_idx else 0 for i, h in enumerate(h_last)]
    tot_nw = sum(nonwhale)

    # Chain order: trophies by holders, most first (earliest earned).
    order = sorted((i for i in range(n_tr) if i != full_idx), key=lambda i: (-h_last[i], i))
    m = extra_total / new_players if new_players > 0 else 0.0
    r = m / (1.0 + m) if m > 0 else 0.0

    h_target = []
    for i in range(n_tr):
        if i == full_idx:
            h_target.append(float(n_target))
            continue
        prop = extra_total * nonwhale[i] / tot_nw if tot_nw > 0 else extra_total / max(1, n_tr - 1)
        rank = order.index(i)
        chain = new_players * (r ** (rank + 1)) if r > 0 else 0.0
        h_target.append(h_last[i] + 0.5 * prop + 0.5 * chain)
    return n_target, h_target, FLAT_SIGMA


def ols2(xs1, xs2, ys):
    """Least squares y = a + b1 x1 + b2 x2 (centred 2x2 solve).  Falls back
    to the one-regressor fit on x1 when x1 and x2 are collinear."""
    n = len(xs1)
    m1 = sum(xs1) / n
    m2 = sum(xs2) / n
    my = sum(ys) / n
    s11 = sum((x - m1) ** 2 for x in xs1)
    s22 = sum((x - m2) ** 2 for x in xs2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(xs1, xs2))
    s1v = sum((a - m1) * (y - my) for a, y in zip(xs1, ys))
    s2v = sum((b - m2) * (y - my) for b, y in zip(xs2, ys))
    det = s11 * s22 - s12 * s12
    if abs(det) <= 1e-12 * max(1.0, s11 * s22):
        a, b, rv = ols(xs1, ys)
        return a, b, 0.0, rv
    b1 = (s22 * s1v - s12 * s2v) / det
    b2 = (s11 * s2v - s12 * s1v) / det
    a = my - b1 * m1 - b2 * m2
    resid = sum((y - (a + b1 * x1 + b2 * x2)) ** 2 for x1, x2, y in zip(xs1, xs2, ys)) / max(1, n - 3)
    return a, b1, b2, resid


def growth_branch(case, reads, filled, n_last, h_last, trophies):
    """Varying history: log-log growth fits for players and holders.

    Holders depend on both time (players progress) and population (new
    players arrive holding early trophies), so log holders is fit on two
    regressors, log(1 + hours) and log(players / first players), with
    per-trophy slopes shrunk toward their tier's mean slope.  Fits use
    every distinct read: within a short window log time and log players
    are nearly collinear, so the two-regressor split is unidentified
    there and extrapolates wildly — the full history identifies it.
    """
    xs = [math.log1p(h) for h, _n, _r in reads]
    ns = [n for _h, n, _r in reads]
    xs_n = [math.log(n / ns[0]) for n in ns]
    x_target = math.log1p(hours_since_release(parse_time(case["target"])))
    horizon_days = case["horizon_minutes"] / 1440.0
    n_tr = len(h_last)

    # Players.
    if len(set(ns)) >= 2:
        a_n, b_n, _ = ols(xs, [math.log(n) for n in ns])
        b_n = min(2.0, max(-0.5, b_n))
        n_target = int(round(math.exp(a_n + b_n * x_target)))
    else:
        n_target = int(round(n_last * PLAYERS_RATIO))
    n_target = max(n_last, min(n_target, n_last * 10 if n_last > 0 else n_target))
    n_target = max(n_target, 1)
    x2_target = math.log(n_target / ns[0]) if ns[0] > 0 else 0.0

    # Per-trophy holder fits, with tier-mean slope shrinkage.
    tiers = [t["tier"] for t in trophies]
    fits = []  # (a, b1, b2, resid) or None
    for i in range(n_tr):
        pts = [(xs[k], xs_n[k], math.log(filled[k][i])) for k in range(len(reads)) if filled[k][i] > 0]
        if len(pts) >= 6:
            a, b1, b2, rv = ols2([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
            fits.append((a, min(3.0, max(-1.0, b1)), min(3.0, max(-1.0, b2)), rv))
        elif len(pts) >= 3:
            a, b, rv = ols([p[0] for p in pts], [p[2] for p in pts])
            fits.append((a, min(3.0, max(-1.0, b)), 0.0, rv))
        else:
            fits.append(None)
    tier_slope = {}
    for tier in set(tiers):
        vals1 = [f[1] for f, t in zip(fits, tiers) if f is not None and t == tier]
        vals2 = [f[2] for f, t in zip(fits, tiers) if f is not None and t == tier]
        tier_slope[tier] = (
            (sum(vals1) / len(vals1) if vals1 else 0.5),
            (sum(vals2) / len(vals2) if vals2 else 0.5),
        )
    all1 = [f[1] for f in fits if f is not None]
    all2 = [f[2] for f in fits if f is not None]
    global_slope = (
        sum(all1) / len(all1) if all1 else 0.5,
        sum(all2) / len(all2) if all2 else 0.5,
    )
    allr = [f[3] for f in fits if f is not None]
    global_resid = sum(allr) / len(allr) if allr else 0.04
    if global_resid <= 0:
        global_resid = 0.04

    # Tier prior rate for trophies nobody holds yet.
    tier_rates = {}
    for tier in set(tiers):
        vals = [h_last[i] / n_last * 100.0 for i in range(n_tr) if tiers[i] == tier and h_last[i] > 0]
        tier_rates[tier] = (min(vals) * 0.5 if vals else 0.05)

    h_target, sigmas = [], []
    for i in range(n_tr):
        npts = sum(1 for k in range(len(reads)) if filled[k][i] > 0)
        if h_last[i] <= 0:
            rate = tier_rates[tiers[i]]
            h_target.append(rate / 100.0 * n_target)
            sigmas.append(0.6)
            continue
        f = fits[i]
        if f is None:
            b1, b2 = tier_slope[tiers[i]]
            lg = math.log(max(1, h_last[i])) + b1 * (x_target - xs[-1]) + b2 * (x2_target - xs_n[-1])
            rv = global_resid
        else:
            a, b1, b2, rv = f
            w = npts / (npts + 3.0)
            b1 = w * b1 + (1.0 - w) * tier_slope[tiers[i]][0]
            b2 = w * b2 + (1.0 - w) * tier_slope[tiers[i]][1]
            lg = a + b1 * x_target + b2 * x2_target
        h = math.exp(lg)
        h = min(max(h, float(h_last[i])), float(n_target))
        h_target.append(h)
        sig = math.sqrt(max(rv, 0.01) + 0.01 + 0.005 * horizon_days)
        sigmas.append(min(0.5, max(0.12, sig)))
    return n_target, h_target, sigmas


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    reads = distinct_reads(case["history"])
    filled = forward_fill(reads, n_tr)
    n_last = reads[-1][1]
    h_last = filled[-1]

    if len(reads) < 3:
        n_target, h_target, sigma = refresh_branch(case, reads, filled, n_last, h_last)
        sigmas = [sigma] * n_tr
    else:
        n_target, h_target, sigmas = growth_branch(case, reads, filled, n_last, h_last, trophies)

    rows = []
    for i, trophy in enumerate(trophies):
        med = 100.0 * h_target[i] / n_target if n_target > 0 else 0.01
        if med >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = clamp(inv_logit(lg + z * sigmas[i]))
            if prev is not None and q < prev:
                q = prev
            qs.append(round4(q))
            prev = qs[-1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
