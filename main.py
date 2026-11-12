#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads hourly snapshots of Roblox's Up-and-Coming chart from /input
(history.ndjson + task.json) and writes one forecast to /output/forecast.json:
for each of the top 10 places, a probability for each game named being at that
place, and the best guess at the top 10 in order.

Model (standard library only, deterministic, offline):
  * Every game on the chart follows a lifecycle: it enters deep in the chart,
    climbs while young and hot, peaks, then fades as it ages and drops off.
    Rank velocity measured on the archive: young games at ranks 50-80 climb
    ~0.2-0.5 places/hour; games older than ~2 days drift down ~0.2-0.4/hr;
    top-10 games are nearly flat with noise (sd ~1.2 places/hr).
  * Each game's rank is simulated forward as a random walk with
    age/rank-dependent drift, a persistent momentum term (recent velocity,
    shrunk, fading over ~12 h), Gaussian noise and occasional viral
    surge/crash jumps. A fixed seed makes two runs byte-identical.
  * 1200 Monte Carlo draws x every hour of the 72-hour scored window:
    at each draw-hour the games are ordered by simulated rank and the top 10
    recorded. Averaging over draws and hours gives each game's probability
    of holding each place (one forecast is scored at every hour, so the
    mixture over hours is the honest single forecast).
  * Brand-new games (not on the chart at the cut) cannot be named, so a
    calibrated share of each place is left unnamed: measured on the archive,
    ~0% of top-10 slots at +24 h but 20-50% at +72 h are games not on the
    cut chart, about half of them brand-new. The unnamed share grows with
    place (new games reach the bottom of the top 10 more often than #1).
  * A game is named at a place only when its probability beats the
    per-game unnamed share it would otherwise receive; named probabilities
    are scaled so each place keeps its target unnamed share.

Paths come from $MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR, defaulting to
/input and /output. All times are read from task.json; nothing about the
games, the window or the answer hours is hardcoded.
"""
import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
N_DRAWS = int(os.environ.get("MAKEFUN_N_DRAWS", "1200"))
UNNAMED_SCALE = float(os.environ.get("MAKEFUN_UNNAMED_SCALE", "2.0"))
MOMENTUM_ON = os.environ.get("MAKEFUN_MOMENTUM", "1") != "0"
NOISE_SD = 1.15          # places per sqrt(hour), measured sd of rank velocity
JUMP_PROB = 0.015        # per game-hour: viral surge / crash
JUMP_MIN, JUMP_MAX = 3.0, 9.0
CHART_MAX_RANK = 115.0   # beyond this a game has effectively left the chart

# Target unnamed share per place (1..10): probability mass reserved for
# games not on the chart at the cut (brand-new entrants). Grows with place:
# a new game reaches #7-10 far more often than #1. Calibrated on the
# archive (see README): ~0% of top-10 slots at +24h, 20-50% at +72h are
# games not on the cut chart, about half of them brand-new.
UNNAMED_TARGET = (0.04, 0.06, 0.07, 0.09, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16)


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _hours(a, b):
    return (b - a).total_seconds() / 3600.0


def load_history(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except ValueError:
                continue
            if not s.get("ok", True):
                continue
            rows = s.get("up_and_coming")
            if not rows:
                continue
            snaps.append(s)
    snaps.sort(key=lambda s: s["at"])
    return task, snaps


def build_games(snaps):
    """Per-game trajectories from the snapshots.

    Returns dict universe_id -> {
        'obs': [(t_hours, rank, log_playing)], 'name': str,
        'first_t': datetime, 'last_rank': float, 'last_t': datetime,
        'velocity': float (places/hr, recent, shrunk), 'logp': float }
    Times are hours since the first snapshot.
    """
    t0 = _parse_time(snaps[0]["at"])
    games = {}
    for s in snaps:
        t = _parse_time(s["at"])
        th = _hours(t0, t)
        for r in s["up_and_coming"]:
            uid = r["universe_id"]
            g = games.get(uid)
            if g is None:
                g = {"obs": [], "name": r.get("name") or ""}
                games[uid] = g
            playing = r.get("playing") or 0
            g["obs"].append((th, float(r["rank"]),
                             math.log(max(playing, 1))))
    for uid, g in games.items():
        obs = g["obs"]
        g["first_t"] = t0.timestamp() + obs[0][0] * 3600.0
        g["last_rank"] = obs[-1][1]
        g["last_t"] = obs[-1][0]
        g["logp"] = obs[-1][2]
        # Recent rank velocity (places/hr), from the last few observations,
        # shrunk toward 0: a game that climbed fast lately keeps some of it.
        vel = 0.0
        if len(obs) >= 2:
            # use up to the last 4 gaps, weighted toward the most recent
            gaps = []
            for a, b in zip(obs[-5:-1] if len(obs) > 5 else obs[:-1],
                            obs[-4:] if len(obs) > 5 else obs[1:]):
                dt = b[0] - a[0]
                if dt > 0:
                    gaps.append(((b[1] - a[1]) / dt, dt))
            if gaps:
                wsum = 0.0
                vsum = 0.0
                for j, (v, dt) in enumerate(gaps):
                    w = 1.0 + 0.5 * j
                    vsum += w * v
                    wsum += w
                vel = vsum / wsum if wsum else 0.0
        # shrink: only keep half the measured velocity, clamp it
        g["velocity"] = max(-0.6, min(0.6, 0.5 * vel))
    return games, t0


def drift(rank, age_h, momentum):
    """Expected rank change per hour (negative = climbing)."""
    # Age drift: young games hold/climb, old games fade.
    if age_h < 24.0:
        age_d = 0.0
    elif age_h < 48.0:
        age_d = 0.08
    else:
        age_d = 0.25
    # Climb bonus for young games deep in the chart.
    if age_h < 36.0 and rank > 20.0:
        climb = -0.25 * min(1.0, (rank - 20.0) / 40.0)
    else:
        climb = 0.0
    # It is hard to hold the very top: slight downward pressure there.
    if rank <= 3.0:
        top = 0.08
    elif rank <= 10.0:
        top = 0.03
    else:
        top = 0.0
    # Very deep, older games drift off the chart.
    deep = 0.15 if (rank > 85.0 and age_h >= 24.0) else 0.0
    d = age_d + climb + top + deep + momentum
    return max(-0.6, min(0.55, d))


def simulate(games, last_chart, t_last, win_from, win_until, places, n_draws):
    """Monte Carlo over rank trajectories.

    Returns counts[uid][place] = number of (draw, hour) pairs where the game
    held that place, and n_hours = number of evaluated hours.
    """
    rng = random.Random(SEED)
    gauss = rng.gauss
    rnd = rng.random
    uniform = rng.uniform

    # Hours to simulate: from the last snapshot up to the window end.
    total_hours = int(math.ceil(_hours(t_last, win_until))) + 1
    # Evaluation hours: those inside [win_from, win_until).
    eval_hours = []
    for h in range(1, total_hours + 1):
        t = t_last.timestamp() + h * 3600.0
        if win_from.timestamp() <= t < win_until.timestamp():
            eval_hours.append(h)
    if not eval_hours:
        eval_hours = list(range(1, total_hours + 1))
    n_hours = len(eval_hours)
    eval_set = set(eval_hours)

    # Per-game static state.
    uids = list(last_chart)
    n = len(uids)
    init_rank = []
    init_age = []
    init_mom = []
    for uid in uids:
        g = games[uid]
        init_rank.append(g["last_rank"])
        # age at the cut, in hours since first seen
        age_cut = _hours(datetime.fromtimestamp(g["first_t"], tz=timezone.utc),
                         t_last)
        init_age.append(age_cut)
        init_mom.append(g["velocity"] if MOMENTUM_ON else 0.0)

    counts = {uid: [0] * places for uid in uids}
    # Temporary buffers reused each draw.
    ranks = [0.0] * n
    order = list(range(n))

    for _ in range(n_draws):
        for i in range(n):
            ranks[i] = init_rank[i]
        # momentum fades over the first 12 h of the forecast
        for h in range(1, total_hours + 1):
            mom_scale = 1.0 if h <= 12 else max(0.0, 1.0 - (h - 12) / 24.0)
            for i in range(n):
                age = init_age[i] + h
                mu = drift(ranks[i], age, init_mom[i] * mom_scale)
                nr = ranks[i] + mu + gauss(0.0, NOISE_SD)
                if rnd() < JUMP_PROB:
                    nr += uniform(JUMP_MIN, JUMP_MAX) * (1.0 if rnd() < 0.5 else -1.0)
                if nr < 1.0:
                    nr = 1.0 + (1.0 - nr) * 0.2
                elif nr > CHART_MAX_RANK:
                    nr = CHART_MAX_RANK
                ranks[i] = nr
            if h in eval_set:
                order.sort(key=lambda i: ranks[i])
                for place in range(places):
                    counts[uids[order[place]]][place] += 1
    return counts, n_hours


def build_forecast(counts, n_hours, places):
    """Turn Monte Carlo counts into probabilities + top10."""
    n_draws = N_DRAWS
    denom = float(n_draws * n_hours)
    # Raw probabilities per game per place.
    raw = {}
    for uid, c in counts.items():
        raw[uid] = [x / denom for x in c]

    probabilities = []
    for place in range(places):
        k = place  # 0-based
        target_unnamed = (UNNAMED_TARGET[k] if k < len(UNNAMED_TARGET) else 0.15) * UNNAMED_SCALE
        # Per-game unnamed share if this place were left to the ~60 games
        # of a future chart we did not name.
        thresh = max(0.002, target_unnamed / 60.0)
        named = [(uid, raw[uid][k]) for uid in raw if raw[uid][k] >= thresh]
        named.sort(key=lambda x: -x[1])
        named = named[:100]
        s = sum(p for _, p in named)
        keep = 1.0 - target_unnamed
        if s > keep and s > 0:
            scale = keep / s
            named = [(uid, p * scale) for uid, p in named]
            s = keep
        # Round to 6 decimals; drop anything that rounds to 0.
        for uid, p in named:
            pr = round(p, 6)
            if pr > 0:
                probabilities.append({"universe_id": uid, "place": k + 1, "p": pr})

    # top10: greedy by modal place probability, distinct games.
    taken = set()
    top10 = []
    for place in range(places):
        best_uid = None
        best_p = -1.0
        for uid in raw:
            if uid in taken:
                continue
            if raw[uid][place] > best_p:
                best_p = raw[uid][place]
                best_uid = uid
        if best_uid is not None:
            taken.add(best_uid)
            top10.append(best_uid)
    # Fallback: fill with highest total top-10 probability if short.
    if len(top10) < places:
        rest = sorted(raw, key=lambda u: -sum(raw[u]))
        for uid in rest:
            if uid not in taken:
                taken.add(uid)
                top10.append(uid)
            if len(top10) >= places:
                break
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    task, snaps = load_history(INPUT_DIR)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    win = task.get("window", {})
    win_from = _parse_time(win["from"]) if "from" in win else _parse_time(snaps[-1]["at"])
    win_until = _parse_time(win["until"]) if "until" in win else win_from

    games, t0 = build_games(snaps)
    last = snaps[-1]
    t_last = _parse_time(last["at"])
    last_chart = [r["universe_id"] for r in sorted(last["up_and_coming"],
                                                   key=lambda r: r["rank"])]
    # Only simulate games we have trajectories for.
    last_chart = [u for u in last_chart if u in games]
    if not last_chart:
        last_chart = sorted(games, key=lambda u: games[u]["last_rank"])[:100]

    counts, n_hours = simulate(games, last_chart, t_last, win_from,
                               win_until, places, N_DRAWS)
    forecast = build_forecast(counts, n_hours, places)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
