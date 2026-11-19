# GTA VI trophy rarity forecaster v1

Forecast each Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate
at the case target as five quantiles (5/25/50/75/95%), for
`gta6-trophy-rarity-forecast-01`. Pure Python 3 stdlib, offline,
deterministic, ~0.1 s per case.

## Method

A trophy's rate is holders / players_tracked, so the program forecasts
holders and players separately and divides, instead of extrapolating the
rate itself. Two regimes, one code path, chosen by what the case's history
shows:

- **Flat history** (fewer than 3 distinct reads — the launch-night practice
  cases): Exophase has not refreshed yet, so the history carries no trend.
  The forecast models the first refresh with the ratios the competition's
  own rules publish (players tracked 638 → 3,868, trophies earned
  943 → 4,276): players grow by 3868/638, every new player holds the
  trophy that sits at 100%, and the remaining new trophies (103 over
  3,230 new players) are split 50/50 between proportional to each
  trophy's holders beyond the "whale" baseline and a chain allocation
  over trophies ordered by holders. The split hedges the two ways new
  players can distribute their first extra trophies.

- **Varying history** (the final case): log-log fits of players and of
  each trophy's holders against log(1 + hours since release), per-trophy
  slopes shrunk toward their tier's mean slope, holders clamped monotone
  and never above players.

Spread is logit-normal quantiles: sigma 0.25 on the flat branch
(allocation uncertainty), fit residuals plus refresh uncertainty on the
growth branch.

## Validated performance (local, simulated)

No local grade exists for this bounty — the scorer and the practice
answers stay on the server. The practice was validated by simulation
instead (`../gta6-trophy-dev/sim_refresh.py`, `sim_alloc.py`,
`validate_v1.py`): the first refresh adds 3,230 new players holding
3,333 trophies, so any forecast that does not know the refresh's exact
realisation pays an irreducible noise floor. This program's simulated
practice loss is 0.109 / 0.084 / 0.161 / 0.141 across the four plausible
refresh processes (proportional / chain-by-holders / chain-by-id /
uniform), expected ~0.12; the best allocation under the most favourable
process scores ~0.024, and the best hedge (the 50/50 blend used here)
is the one shipped. The practice board's top (loss 0.006, typical miss
0.00 points) sits below that floor — a median that close to the printed
rates means knowing the refresh's exact realisation, which no program
can derive from a flat history — so this entry does not chase it. It is
built to be right on the final, where two days of history with real
refreshes give the growth fit something to fit: on two synthetic
final-like worlds the growth branch scores 0.141 (adversarial
never-saturating power-law growth, median rate ratio 0.78) and 0.064
(realistic saturating growth, median ratio 0.92), with 90% coverage
0.93 / 1.00.

Per the entry brief's gate (submit only with a practice loss near or
below ~0.01), this entry was NOT submitted: no honest model reaches
that bar on this practice, and the entry is left ready for the parent
to submit if that gate is overridden for the final.
