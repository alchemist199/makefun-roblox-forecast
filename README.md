# roblox-up-and-coming-forecast-01 — Python forecaster

Forecasts the top 10 of Roblox's Up-and-Coming chart at every hour of the
72 hours after entries close. `main.py` (Python 3.13, standard library only)
reads `/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json` (`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` override
the paths).

## The model

1. **Parse the history.** Every snapshot's `up_and_coming` rows become
   per-game trajectories of rank and log player count, with age (time since
   first seen on the chart) and a shrunk recent rank velocity (momentum).
   All times and the scored window are read from `task.json`, never hardcoded.
2. **Lifecycle drift.** The archive shows a clear lifecycle: young games at
   ranks 50–80 climb ~0.2–0.5 places/hour; games older than ~2 days drift
   down ~0.2–0.4 places/hour; top-10 games are nearly flat. Rank is modelled
   as a random walk with age/rank-dependent drift, a momentum term (recent
   velocity, shrunk by half, fading over the first ~12 h), Gaussian noise
   (sd 1.15 places per sqrt-hour, the measured velocity sd) and occasional
   viral surge/crash jumps (1.5%/hour, ±3–9 places).
3. **Monte Carlo, seeded (fixed seed, deterministic).** 1200 draws × every
   hour of the 72-hour window: at each draw-hour the games are ordered by
   simulated rank and the top 10 recorded. Averaging over draws and hours
   gives each game's probability of holding each place (one forecast is
   scored at every hour, so the mixture over hours is the honest single
   forecast).
4. **Newcomers / unnamed share.** Games not on the chart at the cut cannot
   be named, so a calibrated share of each place is left unnamed. Measured
   on the archive, ~0% of top-10 slots at +24 h but 20–50% at +72 h are
   games not on the cut chart, about half of them brand-new; the share
   grows with place (a new game reaches #7–10 far more often than #1).
   Targets (0.04–0.16 per place) are scaled ×2.0, the value that scored
   best on three independent backtest cuts.
5. **Honest naming rule.** A game is named at a place only when its
   probability beats the per-game unnamed share it would otherwise receive;
   named probabilities are scaled so each place keeps its target unnamed
   share. `top10` is picked greedily by modal place probability.

## Validation (backtest: cut the history, forecast, score vs held-out reads)

Scored exactly as the grader does (mean over reads of the sum of
ln max(P, 0.0001) over the top 10 places), against the persistence
baseline (the kit's starter) on the same reads:

| cut | mine | baseline | delta |
|-----|------|----------|-------|
| 2026-11-02 15:54 | -37.93 | -48.41 | **+10.48** |
| 2026-11-04 09:13 | -35.09 | -50.27 | **+15.18** |
| 2026-11-06 02:27 | -33.64 | -50.78 | **+17.13** |

The house baseline scores -43.51 on the practice forecast; this entry
scores roughly -34 to -38 on comparable backtests. Momentum matters:
turning it off costs ~2.3–2.4 points.

## Checks

- Practice run: ~10 s (limit 300 s), output 14,808 bytes (limit 1 MB),
  exactly the 2 keys (`probabilities`, `top10`); top10 is 10 distinct
  universe ids; every place's named probabilities sum to ≤ 1.
- Deterministic: two runs write byte-identical output.
