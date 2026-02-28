#!/usr/bin/env python3
"""
NHL Moneyline Model (NHL API only) — FULL CODE REWRITE (LEARNED + CALIBRATED) + FIXES

KEEPING EVERYTHING THE SAME AS YOUR CURRENT SCRIPT,
✅ ONLY ADDING/FINISHING GITHUB PAGES PUBLISH FEATURES
(and the minimum mechanical fixes so publish can work end-to-end)

Usage (daily + write html + publish):
  set GITHUB_TOKEN in env
  python nhl_moneyline_model.py --write_daily --publish_github --github_repo owner/repo --season_start 2025-10-07

Backfill:
  python nhl_moneyline_model.py --backfill_start 2025-10-07 --backfill_end 2026-01-29 --season_start 2025-10-07
"""

from __future__ import annotations
import base64
import json

import argparse
import datetime as dt
import math
import os
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Use script's own directory as working directory (works locally, on Pi, and in CI)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =========================
# CONFIG
# =========================
TZ_DEFAULT = "America/Toronto"
OUT_DIR = "output"
HISTORY_PATH = os.path.join(OUT_DIR, "history_predictions.csv")
COEF_PATH = os.path.join(OUT_DIR, "model_coefficients.csv")
RATES_PATH = os.path.join(OUT_DIR, "prediction_rates.csv")   # Tidbyt-friendly summary
RATES_JSON_PATH = os.path.join(OUT_DIR, "prediction_rates.json")
PICKS_PATH = os.path.join(OUT_DIR, "todays_picks.csv")        # Tidbyt-friendly daily picks
PICKS_JSON_PATH = os.path.join(OUT_DIR, "todays_picks.json")

UA = "Mozilla/5.0"

# Training controls
MIN_TRAIN_GAMES = 200
MAX_TRAIN_GAMES = None  # optionally set e.g. 2500

# Calibration (logit shrink)
CALIBRATION_K = 0.7  # 1.0=no shrink; smaller -> stronger shrink

# =========================
# HOME ICE BIAS
# =========================
# NHL home teams win ~54% historically. The model intercept alone doesn't fully
# capture this — we add a small nudge in probability space AFTER calibration.
# Set to 0.0 to disable. Typical range: 0.02 – 0.04
HOME_ICE_BIAS = 0.03

# =========================
# RECENCY WEIGHTING
# =========================
# Exponential decay: games from N days ago get weight = exp(-RECENCY_HALFLIFE_DAYS * age/halflife)
# Lower halflife = faster decay (recent games count more)
# Set to None to disable (all games equal weight)
RECENCY_HALFLIFE_DAYS = 60   # games ~60 days old get half the weight of today's games

# =========================
# Confidence buckets (custom, merged top bin)
# =========================
CONF_BINS = [0.50, 0.53, 0.58, 0.65, 1.00]
CONF_LABELS = ["50-53", "53-58", "58-65", "65-100"]

# --- Club-stats influence controls ---
CLUB_WEIGHT = 0.15        # 1.0 = full effect, 0.25 = subtle
CLUB_PP_PK_CAP = 0.05      # cap PP/PK diff to ±6 percentage points (since stored as 0-1)
CLUB_SF_SA_CAP = 2.0      # cap shots-for/against diff to ±2.5 shots/game
CLUB_CAP_MODE = "soft"      # "soft" (tanh) or "hard"

# --- Club-stats warm-up ---
CLUB_WARMUP_MIN_LABELED = 50   # number of labeled games before club stats are allowed

# =========================
# GITHUB PAGES PUBLISH (CONFIG)
# =========================
# NOTE: repo MUST be "owner/repo" to work with GitHub Contents API
GITHUB_PUBLISH_DEFAULT = True

GITHUB_REPO = "stat-trick-hockey/API_Model"                 # e.g. "nickwatts/stat-trick-hockey" (prefer CLI or env)
GITHUB_REPO_ENV = "GITHUB_REPO"

GITHUB_BRANCH = "main"
GITHUB_BRANCH_ENV = "GITHUB_BRANCH"

GITHUB_DOCS_DIR = "docs"
GITHUB_DOCS_DIR_ENV = "GITHUB_DOCS_DIR"

GITHUB_LATEST_NAME = "latest.html"     # docs/latest.html
GITHUB_LATEST_NAME_ENV = "GITHUB_LATEST_NAME"

GITHUB_ARCHIVE_DIR = ""                # "" => docs/YYYY-MM-DD.html ; "archive" => docs/archive/YYYY-MM-DD.html
GITHUB_ARCHIVE_DIR_ENV = "GITHUB_ARCHIVE_DIR"

# Token: by default read token from env var named "GITHUB_TOKEN".
# You can override the token env var name by setting env GITHUB_TOKEN_ENV or passing --github_token_env.
GITHUB_TOKEN = ""  # ⚠️ NEVER hard-code tokens here — use env var GITHUB_TOKEN or GitHub Actions secret
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_TOKEN_ENV_ENV = "GITHUB_TOKEN_ENV"

# Commit message templates
GITHUB_COMMIT_MSG_LATEST = "Update latest NHL moneyline report ({date})"
GITHUB_COMMIT_MSG_ARCHIVE = "Add NHL moneyline report archive ({date})"

# GitHub API base (rarely needs changing)
GITHUB_API = "https://api.github.com"
GITHUB_API_ENV = "GITHUB_API"


# =========================
# SMALL HELPERS
# =========================

def club_ramp(n: int, n0: int = 50, n1: int = 200) -> float:
    """
    0.0 until n0; linearly ramps to 1.0 by n1; then stays 1.0
    """
    if n <= n0:
        return 0.0
    if n >= n1:
        return 1.0
    return (n - n0) / float(n1 - n0)


def labeled_games_with_club(history_path: str) -> int:
    if not os.path.exists(history_path):
        return 0
    try:
        h = pd.read_csv(history_path)
    except Exception:
        return 0
    if h.empty:
        return 0

    # must be final AND have at least one club column present
    club_cols = ["pp_pct_diff", "pk_pct_diff", "sf_per_g_diff", "sa_per_g_diff"]
    for c in club_cols:
        if c not in h.columns:
            return 0

    labeled = h["actual_winner"].notna()
    has_club = h[club_cols].notna().any(axis=1)
    return int((labeled & has_club).sum())


def softcap(x: Optional[float], cap: float) -> Optional[float]:
    """Smoothly saturate x to (-cap, cap) using tanh."""
    if x is None or pd.isna(x):
        return None
    x = float(x)
    if cap <= 0:
        return 0.0
    return cap * math.tanh(x / cap)

def hardcap(x: Optional[float], cap: float) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    x = float(x)
    return max(-cap, min(cap, x))


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_json(url: str, timeout: int = 25, tries: int = 3, backoff: float = 0.75) -> dict:
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * attempt)
    raise last_err


def is_num(x) -> bool:
    return x is not None and pd.notna(x)


def as_float(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def shrink_to_50(p: float, k: float = CALIBRATION_K) -> float:
    """
    Pull probabilities toward 0.5 by scaling the logit.
    k in (0,1]: smaller = more shrink; 1.0 = no shrink.
    """
    p = min(1 - 1e-12, max(1e-12, float(p)))
    logit = math.log(p / (1 - p))
    logit *= float(k)
    return sigmoid(logit)


def moneyline_from_prob(p: Optional[float]) -> Optional[int]:
    if not is_num(p):
        return None
    p = float(p)
    if not (0.0 < p < 1.0):
        return None
    if p >= 0.5:
        return int(-round(100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def fmt_pct(p: Optional[float]) -> str:
    return "—" if not is_num(p) else f"{100 * float(p):.1f}%"


def fmt_num(x: Optional[float], nd: int = 2) -> str:
    return "—" if not is_num(x) else f"{float(x):.{nd}f}"


def fmt_signed(x: Optional[float], nd: int = 2) -> str:
    return "—" if not is_num(x) else f"{float(x):+.{nd}f}"


def fmt_ml(ml: Optional[int]) -> str:
    return "—" if ml is None else f"{ml:+d}"


def format_local_time(start_utc: str, tz: ZoneInfo) -> str:
    if not start_utc:
        return ""
    try:
        dt_utc = dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
        local = dt_utc.astimezone(tz)
        return local.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return ""


def daterange_inclusive(start_ymd: str, end_ymd: str) -> List[str]:
    d0 = dt.date.fromisoformat(start_ymd)
    d1 = dt.date.fromisoformat(end_ymd)
    if d1 < d0:
        d0, d1 = d1, d0
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return out


def predicted_winner(away: str, home: str, p_home: Optional[float]) -> Optional[str]:
    if not is_num(p_home):
        return None
    return home if float(p_home) >= 0.5 else away


def actual_winner_from_scores(away: str, home: str, away_score: Optional[float], home_score: Optional[float]) -> Optional[str]:
    if away_score is None or home_score is None:
        return None
    if home_score > away_score:
        return home
    if away_score > home_score:
        return away
    return None


def brier(p_home: float, home_won: int) -> float:
    return (p_home - home_won) ** 2


def logloss(p_home: float, home_won: int) -> float:
    p = min(1 - 1e-12, max(1e-12, float(p_home)))
    return -(home_won * math.log(p) + (1 - home_won) * math.log(1 - p))

def refresh_recent_finals_in_history(history: pd.DataFrame, tz: ZoneInfo, lookback_days: int = 3) -> pd.DataFrame:
    """
    For the last N days (by calendar), re-fetch scoreboard and update any history rows
    that are missing actual_winner (or scores). This solves 'yesterday recap stuck on older date'
    when your cron runs before games go final.
    """
    if history is None or history.empty:
        return history

    h = history.copy()
    if "date_ymd" not in h.columns or "game_id" not in h.columns:
        return history

    h["date_ymd"] = h["date_ymd"].astype(str)
    h["game_id"] = pd.to_numeric(h["game_id"], errors="coerce").astype("Int64")

    # last N calendar dates (based on now in tz)
    today = dt.datetime.now(tz).date()
    dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(1, lookback_days + 1)]

    # Only update rows that exist in those dates AND are missing actual_winner
    needs = h[h["date_ymd"].isin(dates)].copy()
    if needs.empty:
        return history

    needs["actual_winner"] = needs.get("actual_winner")
    missing = needs[needs["actual_winner"].isna()].copy()
    if missing.empty:
        return history

    # Map: date -> {game_id -> (scores, actual_winner)}
    updates = {}

    for d in dates:
        try:
            games = load_scoreboard(d)
        except Exception:
            continue
        m = {}
        for g in games:
            is_final = g.game_state in {"FINAL", "OFF"}
            if not is_final:
                continue
            aw = actual_winner_from_scores(g.away_abbrev, g.home_abbrev, g.away_score, g.home_score)
            m[int(g.game_id)] = {
                "game_state": g.game_state,
                "away_score": g.away_score,
                "home_score": g.home_score,
                "actual_winner": aw,
            }
        updates[d] = m

    # Apply updates
    changed = 0
    for idx, r in h[h["date_ymd"].isin(dates)].iterrows():
        gid = r.get("game_id")
        d = r.get("date_ymd")
        if pd.isna(gid) or d not in updates:
            continue
        u = updates[d].get(int(gid))
        if not u:
            continue

        h.at[idx, "game_state"] = u.get("game_state", h.at[idx, "game_state"])
        h.at[idx, "away_score"] = u.get("away_score")
        h.at[idx, "home_score"] = u.get("home_score")
        h.at[idx, "actual_winner"] = u.get("actual_winner")

        pw = h.at[idx, "predicted_winner"] if "predicted_winner" in h.columns else None
        aw = h.at[idx, "actual_winner"]
        if isinstance(pw, str) and isinstance(aw, str) and pw and aw:
            h.at[idx, "is_correct"] = 1 if pw == aw else 0

        changed += 1

    if changed > 0:
        h.to_csv(HISTORY_PATH, index=False)
        log(f"🔄 Refreshed finals for recent days: updated {changed} row(s) in history.")

    return h

# =========================
# GITHUB HELPERS (Contents API)
# =========================
def gh_get_file_sha(repo: str, path: str, branch: str, token: str) -> Optional[str]:
    api = os.getenv(GITHUB_API_ENV, GITHUB_API)
    url = f"{api}/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)
    if r.status_code == 200:
        j = r.json()
        return j.get("sha")
    if r.status_code == 404:
        return None
    raise RuntimeError(f"GitHub GET failed {r.status_code}: {r.text}")


def gh_put_file(repo: str, path: str, content_bytes: bytes, message: str, branch: str, token: str) -> None:
    api = os.getenv(GITHUB_API_ENV, GITHUB_API)
    url = f"{api}/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    sha = gh_get_file_sha(repo, path, branch, token)

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload, timeout=45)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed {r.status_code}: {r.text}")


def publish_html_to_github_pages(
    html_text: str,
    date_ymd: str,
    repo: str,
    branch: str,
    token_env: str,
    docs_dir: str,
    latest_name: str,
    archive_dir: str,
) -> None:
    # allow overriding the env var name that stores the token
    token_env = os.getenv(GITHUB_TOKEN_ENV_ENV, token_env)
    token = os.getenv(token_env)
    if not token:
        raise RuntimeError(f"Missing GitHub token env var: {token_env}")

    docs_dir = (docs_dir or "").strip("/")
    if not docs_dir:
        raise RuntimeError("GitHub publish: docs_dir cannot be empty (use 'docs').")

    latest_name = (latest_name or "latest.html").strip("/")
    latest_path = f"{docs_dir}/{latest_name}"

    # archive_dir="" => docs/YYYY-MM-DD.html
    # archive_dir="archive" => docs/archive/YYYY-MM-DD.html
    archive_dir = (archive_dir or "").strip("/")
    if archive_dir:
        archive_path = f"{docs_dir}/{archive_dir}/{date_ymd}.html"
    else:
        archive_path = f"{docs_dir}/{date_ymd}.html"

    b = html_text.encode("utf-8")

    gh_put_file(
        repo=repo,
        path=latest_path,
        content_bytes=b,
        message=GITHUB_COMMIT_MSG_LATEST.format(date=date_ymd),
        branch=branch,
        token=token,
    )
    gh_put_file(
        repo=repo,
        path=archive_path,
        content_bytes=b,
        message=GITHUB_COMMIT_MSG_ARCHIVE.format(date=date_ymd),
        branch=branch,
        token=token,
    )

    log(f"🌐 Published GitHub Pages: {latest_path} and {archive_path} on {repo}@{branch}")


# =========================
# SPARKLINE / SUMMARY HTML
# =========================
def _sparkline_svg(series: List[float], width: int = 560, height: int = 56, pad: int = 6) -> str:
    """
    Inline SVG sparkline for values in [0,1].
    Returns SVG markup (no external deps).
    """
    if not series:
        return f"""
        <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
          <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="rgba(255,255,255,.04)" stroke="rgba(255,255,255,.10)"/>
          <text x="{pad}" y="{height//2}" fill="rgba(255,255,255,.55)" font-size="12" font-family="system-ui">No scored games yet</text>
        </svg>
        """

    vals = [max(0.0, min(1.0, float(v))) for v in series]
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        vmin = max(0.0, vmin - 0.05)
        vmax = min(1.0, vmax + 0.05)

    def x(i: int) -> float:
        if len(vals) == 1:
            return pad
        return pad + i * (width - 2 * pad) / (len(vals) - 1)

    def y(v: float) -> float:
        return pad + (1.0 - (v - vmin) / (vmax - vmin)) * (height - 2 * pad)

    pts = " ".join(f"{x(i):.2f},{y(v):.2f}" for i, v in enumerate(vals))
    lx, ly = x(len(vals) - 1), y(vals[-1])

    return f"""
    <svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="sparkFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="rgba(90,170,255,.20)"></stop>
          <stop offset="100%" stop-color="rgba(90,170,255,0)"></stop>
        </linearGradient>
      </defs>

      <rect x="0" y="0" width="{width}" height="{height}" rx="12"
            fill="rgba(255,255,255,.04)" stroke="rgba(255,255,255,.10)"/>

      <line x1="{pad}" y1="{y(0.5):.2f}" x2="{width-pad}" y2="{y(0.5):.2f}"
            stroke="rgba(255,255,255,.10)" stroke-dasharray="4 4"/>

      <polyline fill="none" stroke="rgba(90,170,255,.95)" stroke-width="2" points="{pts}" />
      <circle cx="{lx:.2f}" cy="{ly:.2f}" r="3.5" fill="rgba(90,170,255,1)" />
    </svg>
    """


def build_accuracy_sparkline_block(history: pd.DataFrame, window_games: int = 50, max_points: int = 140) -> str:
    """
    One sparkline: rolling pick accuracy over scored games.
    Uses is_correct in [0,1]. Only scored rows included.
    """
    if history is None or history.empty:
        svg = _sparkline_svg([])
        return f"""
        <div class="sparkCard">
          <div class="sparkHdr">
            <div class="sparkTitle">Prediction Rate</div>
            <div class="sparkMeta muted">No data</div>
          </div>
          {svg}
        </div>
        """

    df = history.copy()
    if "date_ymd" in df.columns:
        df["date_ymd"] = df["date_ymd"].astype(str)

    df["is_correct"] = pd.to_numeric(df.get("is_correct"), errors="coerce")
    df = df[df["is_correct"].isin([0, 1])].copy()
    if df.empty:
        svg = _sparkline_svg([])
        return f"""
        <div class="sparkCard">
          <div class="sparkHdr">
            <div class="sparkTitle">Prediction Rate</div>
            <div class="sparkMeta muted">No scored games</div>
          </div>
          {svg}
        </div>
        """

    if "game_id" in df.columns:
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")
        df = df.sort_values(["date_ymd", "game_id"], ascending=True)
    else:
        df = df.sort_values(["date_ymd"], ascending=True)

    roll = df["is_correct"].rolling(window_games, min_periods=max(5, window_games // 4)).mean()
    roll = roll.dropna().tolist()
    if len(roll) > max_points:
        roll = roll[-max_points:]

    svg = _sparkline_svg(roll)
    all_time = float(df["is_correct"].mean()) if len(df) else None
    last_roll = float(roll[-1]) if roll else None

    return f"""
    <div class="sparkCard">
      <div class="sparkHdr">
        <div class="sparkTitle">Prediction Rate</div>
        <div class="sparkMeta">
          <span class="mono">rolling {window_games}g: <b>{fmt_pct(last_roll)}</b></span>
          <span class="muted" style="margin:0 10px;">•</span>
          <span class="mono muted">all-time: {fmt_pct(all_time)}</span>
          <span class="muted" style="margin:0 10px;">•</span>
          <span class="mono muted">scored: {len(df)}g</span>
        </div>
      </div>
      {svg}
    </div>
    """


def bucket_breakdown_html(df: pd.DataFrame, conf_labels: List[str], conf_bins: List[float]) -> str:
    """
    Buckets based on p_pick (probability of predicted team).
    Equal-width bins via pd.cut using conf_bins/conf_labels.
    """
    def _fmt_acc(x: Optional[float]) -> str:
        return "—" if x is None else f"{100*x:.1f}%"

    def empty_grid() -> str:
        chips = []
        for lab in conf_labels:
            chips.append(f"""
            <div class="bucketChip">
              <div class="bk">{lab}</div>
              <div class="bv">—</div>
              <div class="bn">0g</div>
            </div>
            """)
        return f'<div class="bucketGrid" style="grid-template-columns: repeat({len(conf_labels)}, 1fr);">{"".join(chips)}</div>'

    if df is None or df.empty:
        return empty_grid()

    tmp = df.copy()
    tmp["p_pick"] = pd.to_numeric(tmp.get("p_pick"), errors="coerce")
    tmp["is_correct"] = pd.to_numeric(tmp.get("is_correct"), errors="coerce")
    tmp = tmp[tmp["p_pick"].notna() & tmp["is_correct"].isin([0, 1])].copy()

    if tmp.empty:
        return empty_grid()

    tmp["p_pick"] = tmp["p_pick"].clip(lower=conf_bins[0], upper=conf_bins[-1] - 1e-9)

    tmp["bucket"] = pd.cut(
        tmp["p_pick"],
        bins=conf_bins,
        labels=conf_labels,
        include_lowest=True,
        right=False,
    )

    g = (
        tmp.groupby("bucket", dropna=False)["is_correct"]
        .agg(["count", "mean"])
        .reindex(conf_labels)
    )

    chips = []
    for lab in conf_labels:
        n = int(g.loc[lab, "count"]) if lab in g.index and pd.notna(g.loc[lab, "count"]) else 0
        acc = None if n == 0 else float(g.loc[lab, "mean"])
        chips.append(f"""
        <div class="bucketChip">
          <div class="bk">{lab}</div>
          <div class="bv">{_fmt_acc(acc)}</div>
          <div class="bn">{n}g</div>
        </div>
        """)

    return f'<div class="bucketGrid" style="grid-template-columns: repeat({len(conf_labels)}, 1fr);">{"".join(chips)}</div>'


# =========================
# ARGPARSE (Jupyter-safe)
# =========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (daily mode; default: today in --tz)")
    ap.add_argument("--tz", default=TZ_DEFAULT, help=f"Timezone (default: {TZ_DEFAULT})")
    ap.add_argument("--season_start", default=None, help="YYYY-MM-DD filter for metrics/training")

    # Backfill
    ap.add_argument("--backfill_start", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--backfill_end", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds sleep between backfill days")

    # Outputs
    ap.add_argument("--process_yesterday", action="store_true", 
                    help="Also process yesterday's games (useful for morning runs)")

    # Backfill leakage safety
    ap.add_argument(
        "--backfill_disable_clubstats",
        type=int,
        default=1,
        help="1=disable clubstats in backfill (default). 0=allow (leaky).",
    )

    # Walk-forward efficiency
    ap.add_argument(
        "--retrain_every",
        type=int,
        default=1,
        help="Backfill: retrain model every N days (default 1 = true walk-forward).",
    )

    # Optional: rescore all history with final model
    ap.add_argument("--rescore_history", action="store_true", help="After run, overwrite p_home_raw/p_home for ALL rows using final model.")

    # Evaluation stability
    ap.add_argument("--eval_last_n", type=int, default=None, help="Compute season metrics on last N scored games (stable window).")

    # Debug
    ap.add_argument("--max_teams", type=int, default=None, help="Debug: limit to first N teams")

    # GitHub Pages publish
    ap.add_argument("--publish_github", action="store_true", help="Publish daily HTML to GitHub Pages (docs/)")
    ap.add_argument("--github_repo", default=None, help="owner/repo (overrides config/env)")
    ap.add_argument("--github_branch", default=None, help="branch (overrides config/env)")
    ap.add_argument("--github_docs_dir", default=None, help="docs dir (overrides config/env)")
    ap.add_argument("--github_token_env", default=None, help="token env var name (overrides config/env)")
    ap.add_argument("--github_latest_name", default=None, help="latest filename (overrides config/env)")
    ap.add_argument("--github_archive_dir", default=None, help="archive subdir under docs ('' for none)")
    ap.add_argument("--write_daily", action="store_true", help="Write daily CSV + HTML to output/")
    ap.add_argument(
        "--no_clubstats",
        action="store_true",
        help="Disable club stats (PP/PK/SF/SA) for daily runs",
    )

    args, _unknown = ap.parse_known_args()
    return args


# =========================
# NHL LOADERS
# =========================
@dataclass
class GameRow:
    game_id: int
    start_utc: str
    away_abbrev: str
    home_abbrev: str
    game_state: str
    away_score: Optional[float]
    home_score: Optional[float]


def load_scoreboard(date_ymd: str) -> List[GameRow]:
    data = fetch_json(f"https://api-web.nhle.com/v1/score/{date_ymd}")
    games: List[GameRow] = []
    for g in data.get("games", []):
        state = (g.get("gameState") or "").upper()
        away = g.get("awayTeam", {}) or {}
        home = g.get("homeTeam", {}) or {}
        games.append(
            GameRow(
                game_id=int(g.get("id")),
                start_utc=str(g.get("startTimeUTC", "") or ""),
                away_abbrev=str(away.get("abbrev", "") or ""),
                home_abbrev=str(home.get("abbrev", "") or ""),
                game_state=state,
                away_score=as_float(away.get("score")),
                home_score=as_float(home.get("score")),
            )
        )
    return games


def load_standings(date_ymd: str) -> pd.DataFrame:
    data = fetch_json(f"https://api-web.nhle.com/v1/standings/{date_ymd}")
    rows = []
    for r in data.get("standings", []):
        team_abbrev = r.get("teamAbbrev")
        team = team_abbrev.get("default") if isinstance(team_abbrev, dict) else team_abbrev

        gp = as_float(r.get("gamesPlayed") or r.get("gp"))
        gf = as_float(r.get("goalFor") or r.get("goalsFor") or r.get("gf"))
        ga = as_float(r.get("goalAgainst") or r.get("goalsAgainst") or r.get("ga"))

        points_pct = as_float(
            r.get("pointsPctg")
            or r.get("pointPctg")
            or r.get("pointsPct")
            or r.get("pointsPercentage")
        )

        rows.append({"team": team, "gp": gp, "gf": gf, "ga": ga, "points_pct": points_pct})

    df = pd.DataFrame(rows).dropna(subset=["team"])
    return df


def load_team_rates_summary(team: str, season_id: str) -> dict:
    """
    Fetch team-level rates (PP%, PK%, SF/G, SA/G) from NHL stats API.
    gameTypeId=2 => regular season
    """
    team = str(team).upper()
    url = (
        "https://api.nhle.com/stats/rest/en/team/summary"
        f"?isAggregate=true&isGame=false&limit=1"
        f"&cayenneExp=seasonId={season_id}%20and%20gameTypeId=2%20and%20teamAbbrev=%22{team}%22"
    )
    j = fetch_json(url)
    data = j.get("data") or []
    return data[0] if data else {}


def parse_team_rates_summary(row: dict) -> Dict[str, Optional[float]]:
    if not row:
        return {"pp_pct": None, "pk_pct": None, "sf_per_g": None, "sa_per_g": None}
    return {
        "pp_pct": row.get("powerPlayPct"),
        "pk_pct": row.get("penaltyKillPct"),
        "sf_per_g": row.get("shotsForPerGame"),
        "sa_per_g": row.get("shotsAgainstPerGame"),
    }


def load_team_schedule_now(team: str) -> dict:
    team = str(team).lower()
    return fetch_json(f"https://api-web.nhle.com/v1/club-schedule-season/{team}/now")


# =========================
# SCHEDULE -> LAST10 + FATIGUE
# =========================
def extract_schedule_rows(schedule_json: dict) -> List[dict]:
    raw_games: List[dict] = []
    if isinstance(schedule_json, dict):
        if isinstance(schedule_json.get("games"), list):
            raw_games = schedule_json["games"]
        elif isinstance(schedule_json.get("gameWeek"), list):
            for wk in schedule_json["gameWeek"]:
                if isinstance(wk, dict) and isinstance(wk.get("games"), list):
                    raw_games.extend(wk["games"])

    out: List[dict] = []
    for g in raw_games:
        if not isinstance(g, dict):
            continue
        ymd = g.get("gameDate") or g.get("gameDateUTC") or g.get("date") or ""
        ymd = str(ymd)[:10] if ymd else None
        if not ymd:
            continue

        state = (g.get("gameState") or g.get("gameStatus") or "").upper()
        is_final = state in {"FINAL", "OFF"}

        away = g.get("awayTeam", {}) if isinstance(g.get("awayTeam"), dict) else {}
        home = g.get("homeTeam", {}) if isinstance(g.get("homeTeam"), dict) else {}

        out.append(
            {
                "gameDate": ymd,
                "game_id": g.get("id"),
                "is_final": bool(is_final),
                "away_abbrev": away.get("abbrev"),
                "home_abbrev": home.get("abbrev"),
                "away_goals": as_float(away.get("score")),
                "home_goals": as_float(home.get("score")),
            }
        )
    return out


def last_n_form(team: str, schedule_rows: List[dict], asof_date: str, n: int = 10) -> Dict[str, Optional[float]]:
    finals = [r for r in schedule_rows if r.get("is_final") and r.get("gameDate") and r["gameDate"] < asof_date]
    finals = sorted(finals, key=lambda x: (x["gameDate"], str(x.get("game_id") or "")))
    last = finals[-n:]

    if not last:
        return {"l10_points_pct": None, "l10_gd_per_g": None}

    pts = 0.0
    gf = 0.0
    ga = 0.0
    gp = 0

    for r in last:
        away = r.get("away_abbrev")
        home = r.get("home_abbrev")
        ag = r.get("away_goals")
        hg = r.get("home_goals")
        if away is None or home is None or ag is None or hg is None:
            continue

        if team == away:
            team_g, opp_g = ag, hg
        elif team == home:
            team_g, opp_g = hg, ag
        else:
            continue

        gp += 1
        gf += team_g
        ga += opp_g

        if team_g > opp_g:
            pts += 2.0
        elif team_g == opp_g:
            pts += 1.0

    if gp <= 0:
        return {"l10_points_pct": None, "l10_gd_per_g": None}

    return {"l10_points_pct": pts / (2.0 * gp), "l10_gd_per_g": (gf - ga) / gp}


def schedule_pressure(schedule_rows: List[dict], game_date: str) -> Dict[str, float]:
    past_dates = sorted({r["gameDate"] for r in schedule_rows if r.get("gameDate") and r["gameDate"] < game_date})
    if not past_dates:
        return {"rest_days": 7.0, "b2b": 0.0, "3in4": 0.0, "4in6": 0.0}

    d0 = dt.date.fromisoformat(game_date)
    d_last = dt.date.fromisoformat(past_dates[-1])
    # Cap rest at 5 days — longer breaks (bye weeks, All-Star, Olympics) are all
    # "well rested" and should be treated the same; no need to distinguish 7 vs 21 days.
    rest = min(5, max(0, (d0 - d_last).days - 1))

    def count_in_window(days: int) -> int:
        start = d0 - dt.timedelta(days=days - 1)
        end = d0 - dt.timedelta(days=1)
        c = 0
        for ymd in past_dates:
            dd = dt.date.fromisoformat(ymd)
            if start <= dd <= end:
                c += 1
        return c

    g4 = count_in_window(4)
    g6 = count_in_window(6)

    return {
        "rest_days": float(rest),
        "b2b": 1.0 if rest == 0 else 0.0,
        "3in4": 1.0 if g4 >= 2 else 0.0,
        "4in6": 1.0 if g6 >= 3 else 0.0,
    }


def build_schedule_cache(team_list_date: str, max_teams: Optional[int] = None) -> Dict[str, List[dict]]:
    standings = load_standings(team_list_date)
    teams = sorted(standings["team"].dropna().unique().tolist())
    if max_teams is not None:
        teams = teams[: max_teams]

    cache: Dict[str, List[dict]] = {}
    log(f"📦 Fetching team schedules once: {len(teams)} teams…")
    for i, t in enumerate(teams, 1):
        if i == 1 or i % 8 == 0 or i == len(teams):
            log(f"  …schedule {i}/{len(teams)}: {t}")
        try:
            cache[t] = extract_schedule_rows(load_team_schedule_now(t))
        except Exception:
            cache[t] = []
    return cache


# =========================
# TEAM FEATURES (as-of date)
# =========================
def build_team_feature_table(
    asof_ymd: str,
    schedule_cache: Dict[str, List[dict]],
    use_clubstats: bool,
    max_teams: Optional[int] = None,
) -> pd.DataFrame:
    # --- derive NHL seasonId from as-of date (e.g., 2026-01-30 -> 20252026) ---
    year = int(asof_ymd[:4])
    month = int(asof_ymd[5:7])
    season_start_year = year if month >= 7 else year - 1
    season_id = f"{season_start_year}{season_start_year+1}"

    standings = load_standings(asof_ymd).copy()
    standings["gp"] = pd.to_numeric(standings["gp"], errors="coerce")
    standings = standings[standings["gp"].notna() & (standings["gp"] > 0)].copy()
    standings["gd_per_g"] = (standings["gf"] - standings["ga"]) / standings["gp"]

    teams = sorted(standings["team"].dropna().unique().tolist())
    if max_teams is not None:
        teams = teams[: max_teams]
        standings = standings[standings["team"].isin(teams)].copy()

    club = {t: {"pp_pct": None, "pk_pct": None, "sf_per_g": None, "sa_per_g": None} for t in teams}

    if use_clubstats:
        log(f"    📌 Team summary rates ON — fetching {len(teams)} teams…")
        for i, t in enumerate(teams, 1):
            if i == 1 or i % 10 == 0 or i == len(teams):
                log(f"      …team summary {i}/{len(teams)}: {t}")

            try:
                row = load_team_rates_summary(t, season_id)
                parsed = parse_team_rates_summary(row)
                club[t] = parsed
                if i <= 3:
                    log(f"      ✅ team rates {t}: {parsed}")
            except Exception as e:
                log(f"      ⚠️ team summary failed for {t}: {e}")

        missing_pp = sum(1 for t in teams if club.get(t, {}).get("pp_pct") is None)
        missing_sf = sum(1 for t in teams if club.get(t, {}).get("sf_per_g") is None)
        log(f"    🔎 Team summary coverage: missing PP% {missing_pp}/{len(teams)}, missing SF/G {missing_sf}/{len(teams)}")

    rows = []
    for _, r in standings.iterrows():
        t = r["team"]
        sched = schedule_cache.get(t, [])
        l10 = last_n_form(t, sched, asof_date=asof_ymd, n=10)
        press = schedule_pressure(sched, game_date=asof_ymd)
        cs = club.get(t, {})

        rows.append(
            {
                "team": t,
                "points_pct": r.get("points_pct"),
                "gd_per_g": r.get("gd_per_g"),
                "l10_points_pct": l10.get("l10_points_pct"),
                "l10_gd_per_g": l10.get("l10_gd_per_g"),
                "pp_pct": cs.get("pp_pct"),
                "pk_pct": cs.get("pk_pct"),
                "sf_per_g": cs.get("sf_per_g"),
                "sa_per_g": cs.get("sa_per_g"),
                "rest_days": press.get("rest_days"),
                "b2b": press.get("b2b"),
                "3in4": press.get("3in4"),
                "4in6": press.get("4in6"),
            }
        )

    return pd.DataFrame(rows)


def team_metrics_snapshot(team: str, team_table: pd.DataFrame) -> dict:
    row = team_table.loc[team_table["team"] == team]
    if row.empty:
        return {}
    r = row.iloc[0]

    def f(col: str):
        v = r.get(col)
        return None if (v is None or pd.isna(v)) else float(v)

    return {
        "points_pct": f("points_pct"),
        "gd_per_g": f("gd_per_g"),
        "l10_points_pct": f("l10_points_pct"),
        "l10_gd_per_g": f("l10_gd_per_g"),
        "pp_pct": f("pp_pct"),
        "pk_pct": f("pk_pct"),
        "sf_per_g": f("sf_per_g"),
        "sa_per_g": f("sa_per_g"),
        "rest_days": f("rest_days"),
    }


# =========================
# LEARNED FEATURES (home - away)
# =========================
MODEL_FEATURES = [
    "points_pct_diff",
    "gd_per_g_diff",
    "l10_points_pct_diff",
    "l10_gd_per_g_diff",
    "pp_pct_diff",
    "pk_pct_diff",
    "sf_per_g_diff",
    "sa_per_g_diff",
    "rest_days_diff",
    "b2b_diff",
    "3in4_diff",
    "4in6_diff",
]


def build_game_features(
    away_metrics: dict,
    home_metrics: dict,
    away_fatigue: dict,
    home_fatigue: dict,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}

    def diff(k: str) -> Optional[float]:
        ha = home_metrics.get(k)
        aa = away_metrics.get(k)
        return (ha - aa) if is_num(ha) and is_num(aa) else None

    for k in ["points_pct", "gd_per_g", "l10_points_pct", "l10_gd_per_g", "pp_pct", "pk_pct", "sf_per_g", "sa_per_g", "rest_days"]:
        out[f"{k}_diff"] = diff(k)

    # Cap rest_days_diff to training range (±5) — All-Star/Olympics breaks cause
    # values like ±21 which are wildly OOD and dominate predictions.
    if is_num(out.get("rest_days_diff")):
        out["rest_days_diff"] = max(-5.0, min(5.0, float(out["rest_days_diff"])))

    for k in ["b2b", "3in4", "4in6"]:
        out[f"{k}_diff"] = float(home_fatigue.get(k, 0.0)) - float(away_fatigue.get(k, 0.0))

    # --- Clamp + down-weight club diffs so they only "nudge" probabilities ---
        # --- Warm-up gate + clamp/weight club diffs ---
    club_keys = ["pp_pct_diff", "pk_pct_diff", "sf_per_g_diff", "sa_per_g_diff"]

    # check once per call (cheap)
    club_labeled_n = labeled_games_with_club(HISTORY_PATH)
    allow_club = club_labeled_n >= CLUB_WARMUP_MIN_LABELED

    for k in club_keys:
        v = out.get(k)
        if not is_num(v):
            continue

        # If not warmed up, zero out club influence
        if not allow_club:
            out[k] = None   # will become 0 downstream
            continue

        # otherwise clamp + weight
        cap = CLUB_PP_PK_CAP if k in ("pp_pct_diff", "pk_pct_diff") else CLUB_SF_SA_CAP
        v2 = softcap(v, cap) if CLUB_CAP_MODE == "soft" else hardcap(v, cap)
        ramp = club_ramp(club_labeled_n, n0=50, n1=200)   # start letting it in at 50, full by 200
        out[k] = float(v2) * float(CLUB_WEIGHT) * ramp


    return out



# =========================
# HISTORY IO
# =========================
def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        if "game_id" in df.columns:
            df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
        if "date_ymd" in df.columns:
            df["date_ymd"] = df["date_ymd"].astype(str)
        return df
    return pd.DataFrame()


def upsert_history(rows: pd.DataFrame) -> pd.DataFrame:
    ensure_outdir(OUT_DIR)
    hist = load_history()

    td = rows.copy()
    td["game_id"] = pd.to_numeric(td["game_id"], errors="coerce").astype("Int64")
    td["date_ymd"] = td["date_ymd"].astype(str)

    if hist.empty:
        merged = td
    else:
        hist["game_id"] = pd.to_numeric(hist["game_id"], errors="coerce").astype("Int64")
        merged = pd.concat([hist[~hist["game_id"].isin(td["game_id"])], td], ignore_index=True)
        merged = merged.drop_duplicates(subset=["game_id"], keep="last")

    merged.to_csv(HISTORY_PATH, index=False)
    return merged


def score_final_games_in_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "predicted_winner" not in out.columns:
        return out
    if "actual_winner" not in out.columns:
        return out

    def compute_row(r):
        pw = r.get("predicted_winner")
        aw = r.get("actual_winner")
        if isinstance(pw, str) and isinstance(aw, str) and pw and aw:
            return 1 if pw == aw else 0
        return None

    out["is_correct"] = out.apply(compute_row, axis=1)
    return out


# =========================
# TRAINING
# =========================
def train_model_from_history(history: pd.DataFrame, season_start: Optional[str], train_through_date: Optional[str] = None) -> Optional[Pipeline]:
    df = history.copy()
    if df.empty:
        log("🧠 Train: history empty.")
        return None

    if "date_ymd" in df.columns:
        df["date_ymd"] = df["date_ymd"].astype(str)

    if season_start:
        df = df[df.get("date_ymd", "") >= str(season_start)]

    if train_through_date:
        df = df[df.get("date_ymd", "") < str(train_through_date)]

    df = df[df.get("actual_winner").notna()].copy()
    if df.empty:
        log("🧠 Train: no labeled (final) games yet.")
        return None

    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        log(f"🧠 Train: missing feature columns in history: {missing[:6]}{'...' if len(missing)>6 else ''}")
        return None

    df["home_won"] = (df["actual_winner"] == df["home"]).astype(int)
    X = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["home_won"].astype(int)

    if MAX_TRAIN_GAMES is not None and len(X) > MAX_TRAIN_GAMES:
        X = X.tail(MAX_TRAIN_GAMES)
        y = y.tail(MAX_TRAIN_GAMES)

    if len(X) < MIN_TRAIN_GAMES:
        log(f"🧠 Train: only {len(X)} games (need {MIN_TRAIN_GAMES}). Using fallback 50/50.")
        return None

    # --- Recency weighting ---
    # Exponential decay so recent games count more than early-season games.
    sample_weight = None
    if RECENCY_HALFLIFE_DAYS:
        try:
            dates = df["date_ymd"].astype(str)
            if MAX_TRAIN_GAMES is not None and len(dates) > MAX_TRAIN_GAMES:
                dates = dates.tail(MAX_TRAIN_GAMES)
            max_date = dt.date.fromisoformat(dates.max())
            ages = dates.apply(
                lambda d: max(0, (max_date - dt.date.fromisoformat(d)).days)
            )
            decay = float(math.log(2) / RECENCY_HALFLIFE_DAYS)
            sample_weight = ages.apply(lambda a: math.exp(-decay * a)).values
            log(f"🧠 Recency weighting ON (halflife={RECENCY_HALFLIFE_DAYS}d) — "
                f"oldest game weight: {sample_weight.min():.3f}, newest: {sample_weight.max():.3f}")
        except Exception as e:
            log(f"⚠️ Recency weighting failed, falling back to equal weights: {e}")
            sample_weight = None
    
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                max_iter=600,
                solver="lbfgs",
                C=0.3,  # Stronger regularization to reduce overconfidence from noisy features
                # class_weight omitted — home win rate ~54% is near-balanced, no need
            )),
        ]
    )
    pipe.fit(X, y, clf__sample_weight=sample_weight)

    # Log effective training size accounting for weights
    if sample_weight is not None:
        eff_n = int(sample_weight.sum() ** 2 / (sample_weight ** 2).sum())
        log(f"🧠 Effective sample size (recency-weighted): ~{eff_n} of {len(X)} games")

    coef = pipe.named_steps["clf"].coef_[0]
    intercept = pipe.named_steps["clf"].intercept_[0]
    coef_df = pd.DataFrame({"feature": MODEL_FEATURES, "coef": coef})
    coef_df.loc[len(coef_df)] = {"feature": "intercept", "coef": intercept}
    coef_df = coef_df.sort_values("coef", ascending=False)
    ensure_outdir(OUT_DIR)
    coef_df.to_csv(COEF_PATH, index=False)
    log(f"🧠 Trained logistic regression on {len(X)} games. Saved: {COEF_PATH}")

    return pipe


def predict_prob_home(pipe: Optional[Pipeline], feat_row: Dict[str, Optional[float]]) -> Optional[float]:
    if pipe is None:
        return None
    X = pd.DataFrame([feat_row])[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    p = pipe.predict_proba(X)[0, 1]
    return float(p)


# =========================
# METRICS
# =========================
def compute_season_metrics(history: pd.DataFrame, season_start: Optional[str], eval_last_n: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = history.copy()
    if df.empty:
        summary = pd.DataFrame([{
            "season_start_filter": season_start or "(none)",
            "eval_last_n": eval_last_n,
            "n_games_scored": 0,
            "pick_accuracy": None,
            "brier_mean": None,
            "logloss_mean": None,
        }])
        buckets = pd.DataFrame(columns=["bucket", "count", "accuracy"])
        return summary, buckets

    if "date_ymd" in df.columns:
        df["date_ymd"] = df["date_ymd"].astype(str)

    if season_start:
        df = df[df.get("date_ymd", "") >= str(season_start)]

    df["p_home"] = pd.to_numeric(df.get("p_home"), errors="coerce")
    df = df[df["p_home"].notna() & df.get("actual_winner").notna()].copy()
    if df.empty:
        summary = pd.DataFrame([{
            "season_start_filter": season_start or "(none)",
            "eval_last_n": eval_last_n,
            "n_games_scored": 0,
            "pick_accuracy": None,
            "brier_mean": None,
            "logloss_mean": None,
        }])
        buckets = pd.DataFrame(columns=["bucket", "count", "accuracy"])
        return summary, buckets

    if eval_last_n is not None and eval_last_n > 0 and len(df) > eval_last_n:
        df = df.sort_values(["date_ymd", "game_id"], ascending=True).tail(eval_last_n).copy()

    df = score_final_games_in_history(df)
    scored = df[df["is_correct"].isin([0, 1])].copy()
    pick_acc = float(scored["is_correct"].mean())

    scored["home_won"] = (scored["actual_winner"] == scored["home"]).astype(int)
    scored["brier"] = scored.apply(lambda r: brier(float(r["p_home"]), int(r["home_won"])), axis=1)
    scored["logloss"] = scored.apply(lambda r: logloss(float(r["p_home"]), int(r["home_won"])), axis=1)

    summary = pd.DataFrame([{
        "season_start_filter": season_start or "(none)",
        "eval_last_n": eval_last_n,
        "n_games_scored": int(len(scored)),
        "pick_accuracy": pick_acc,
        "brier_mean": float(scored["brier"].mean()),
        "logloss_mean": float(scored["logloss"].mean()),
    }])

    scored["conf"] = scored["p_home"].apply(lambda p: max(float(p), 1.0 - float(p)))
    n_bins = 6
    scored["bucket"] = pd.qcut(scored["conf"], q=n_bins, duplicates="drop")

    buckets = (
        scored.groupby("bucket", dropna=False)["is_correct"]
        .agg(["count", "mean"])
        .reset_index()
    )
    buckets.rename(columns={"mean": "accuracy"}, inplace=True)
    buckets["accuracy"] = buckets["accuracy"].astype(float)
    buckets["bucket"] = buckets["bucket"].astype(str).str.replace(",", "–")
    return summary, buckets


def build_html_summary_block(h: pd.DataFrame, today_ymd: str) -> str:
    """
    Builds the top "Model Summary" card (All-time / 90 / 60 / 30 / 7 / Yesterday + recap),
    using CONF_BINS/CONF_LABELS buckets on p_pick (prob of the model's pick).
    """
    def fmt_acc(x: Optional[float]) -> str:
        return "—" if x is None or pd.isna(x) else f"{100*float(x):.1f}%"

    def fmt_pct_local(x: Optional[float]) -> str:
        return "—" if x is None or pd.isna(x) else f"{100*float(x):.1f}%"

    def trend_html(cur: Optional[float], prev: Optional[float]) -> str:
        if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
            return '<span class="trend trendFlat">•</span>'
        d = float(cur) - float(prev)
        if abs(d) < 1e-9:
            return '<span class="trend trendFlat">•</span>'
        if d > 0:
            return '<span class="trend trendUp">▲</span>'
        return '<span class="trend trendDown">▼</span>'

    def all_time_stats(df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {"n": 0, "acc": None}
        tmp = df.copy()
        tmp["is_correct"] = pd.to_numeric(tmp.get("is_correct"), errors="coerce")
        tmp = tmp[tmp["is_correct"].isin([0, 1])]
        if tmp.empty:
            return {"n": 0, "acc": None}
        return {"n": int(len(tmp)), "acc": float(tmp["is_correct"].mean())}

    def window_df(df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df is None or df.empty or "date_ymd" not in df.columns:
            return pd.DataFrame()
        maxd = df["date_ymd"].dropna().astype(str).max()
        if not maxd:
            return pd.DataFrame()
        d1 = dt.date.fromisoformat(str(maxd))
        d0 = d1 - dt.timedelta(days=days - 1)
        return df[df["date_ymd"].astype(str) >= d0.isoformat()].copy()

    def compute_p_pick(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        tmp = df.copy()
        tmp["p_home"] = pd.to_numeric(tmp.get("p_home"), errors="coerce")
        tmp["predicted_winner"] = tmp.get("predicted_winner").astype(str)
        tmp["away"] = tmp.get("away").astype(str)
        tmp["home"] = tmp.get("home").astype(str)

        def _pp(r):
            ph = r.get("p_home")
            if ph is None or pd.isna(ph):
                return None
            if r.get("predicted_winner") == r.get("home"):
                return float(ph)
            if r.get("predicted_winner") == r.get("away"):
                return float(1.0 - float(ph))
            return None

        tmp["p_pick"] = tmp.apply(_pp, axis=1)
        return tmp

    def view_block(title: str, df_cur: pd.DataFrame, df_prev: pd.DataFrame, total_label: str, highlight: bool = False) -> str:
        cur = all_time_stats(df_cur)
        prev = all_time_stats(df_prev)
        cls = "summaryStat highlight" if highlight else "summaryStat"
        df_cur2 = compute_p_pick(df_cur)
        return f"""
        <div class="{cls}">
          <div class="k">{title}</div>
          <div class="v">{fmt_acc(cur["acc"])} {trend_html(cur["acc"], prev["acc"])} <span class="muted">({cur["n"]} {total_label})</span></div>
          {bucket_breakdown_html(df_cur2, CONF_LABELS, CONF_BINS)}
        </div>
        """

    if h is None or h.empty:
        latest_scored_date = None
        h_cur = pd.DataFrame()
        h_prev = pd.DataFrame()
    else:
        h_cur = h.copy()
        if "date_ymd" in h_cur.columns:
            h_cur["date_ymd"] = h_cur["date_ymd"].astype(str)

        h_scored = h_cur[h_cur.get("actual_winner").notna()].copy()
        latest_scored_date = str(h_scored["date_ymd"].max()) if not h_scored.empty else None

        if latest_scored_date:
            h_prev = h_cur[h_cur["date_ymd"].astype(str) < latest_scored_date].copy()
        else:
            h_prev = pd.DataFrame()

    all_cur_df = h_cur
    all_prev_df = h_prev

    d90_cur = window_df(h_cur, 90)
    d90_prev = window_df(h_prev, 90)

    d60_cur = window_df(h_cur, 60)
    d60_prev = window_df(h_prev, 60)

    d30_cur = window_df(h_cur, 30)
    d30_prev = window_df(h_prev, 30)

    d7_cur = window_df(h_cur, 7)
    d7_prev = window_df(h_prev, 7)

    if latest_scored_date:
        y = h_cur[h_cur["date_ymd"] == latest_scored_date].copy()
        prev_latest_date = str(h_prev["date_ymd"].max()) if (h_prev is not None and not h_prev.empty) else None
        y_prev = h_prev[h_prev["date_ymd"] == prev_latest_date].copy() if prev_latest_date else pd.DataFrame()
    else:
        y = pd.DataFrame()
        y_prev = pd.DataFrame()

    recap_rows = []
    if not y.empty:
        if "time_local" in y.columns and "game_id" in y.columns:
            y = y.sort_values(["time_local", "game_id"], ascending=[True, True], na_position="last")

        y2 = compute_p_pick(y)

        for _, r in y2.iterrows():
            away = str(r.get("away") or "")
            home = str(r.get("home") or "")
            pw = str(r.get("predicted_winner") or "")
            aw = str(r.get("actual_winner") or "")
            ok = r.get("is_correct")
            ok = int(ok) if ok in [0, 1, 0.0, 1.0] else None
            p_pick = r.get("p_pick")

            badge = "✅" if ok == 1 else ("❌" if ok == 0 else "—")
            recap_rows.append(
                f"""
                <div class="recapRow">
                  <div class="recapLeft">{away} @ {home}</div>
                  <div class="recapMid">Pick: <b>{pw or "—"}</b> <span class="muted">({fmt_pct_local(p_pick)})</span></div>
                  <div class="recapRight">{badge} <span class="muted">Actual: {aw or "—"}</span></div>
                </div>
                """
            )
    else:
        recap_rows.append('<div class="muted" style="padding:8px 0;">No scored games found.</div>')

    recap_html = "\n".join(recap_rows)

    return f"""
    <div class="summaryCard">
      <div class="summaryTitle">Model Summary</div>

      <div class="summaryGrid">
        {view_block("All-time", all_cur_df, all_prev_df, "games", highlight=True)}
        {view_block("Last 90 days", d90_cur, d90_prev, "games")}
        {view_block("Last 60 days", d60_cur, d60_prev, "games")}
        {view_block("Last 30 days", d30_cur, d30_prev, "games")}
        {view_block("Last 7 days", d7_cur, d7_prev, "games")}
        {view_block("Yesterday", y, y_prev, "games", highlight=True)}
      </div>

      <div class="summaryTitle" style="margin-top:14px;">Yesterday Recap <span class="muted">({latest_scored_date or "—"})</span></div>
      <div class="recapBox">
        {recap_html}
      </div>
    </div>
    """


# =========================
# HTML
# =========================
def render_html(df: pd.DataFrame, date_ymd: str, tz_name: str, summary_html: str = "", spark_html: str = "") -> str:
    def chip(text: str, cls: str) -> str:
        return f'<span class="chip {cls}">{text}</span>'

    rows_html = []
    for _, r in df.iterrows():
        away = r.get("away", "")
        home = r.get("home", "")
        p_away = r.get("p_away")
        p_home = r.get("p_home")
        away_ml = r.get("away_ml_fair")
        home_ml = r.get("home_ml_fair")

        pick = r.get("predicted_winner") or "—"
        pick_cls = "pos" if pick in (away, home) else "neutral"

        p_home_raw = r.get("p_home_raw")
        p_home_cal = r.get("p_home")

        if pick == home:
            p_pick = p_home_cal
            p_pick_raw = p_home_raw
        elif pick == away:
            p_pick = (1.0 - p_home_cal) if is_num(p_home_cal) else None
            p_pick_raw = (1.0 - p_home_raw) if is_num(p_home_raw) else None
        else:
            p_pick = None
            p_pick_raw = None

        lean = "—"
        lean_cls = "neutral"
        if is_num(p_home):
            ph = float(p_home)
            if ph >= 0.55:
                lean, lean_cls = f"{home} lean", "pos"
            elif ph <= 0.45:
                lean, lean_cls = f"{away} lean", "pos"
            else:
                lean, lean_cls = "Toss-up", "neutral"

        def g(prefix: str, key: str) -> Optional[float]:
            return r.get(f"{prefix}_{key}")

        a_pick = (pick == away)
        h_pick = (pick == home)

        metrics_block = f"""
        <div class="mBox">
          <div class="mHdr">
            {chip(f"Pick: {pick}", pick_cls)}
            <span class="muted mono" style="margin-left:10px;">P({pick}): {fmt_pct(p_pick)} • raw: {fmt_pct(p_pick_raw)}</span>
          </div>
          <div class="mGrid">
            <div class="mCol {'pickGlow' if a_pick else ''}">
              <div class="mTeam">{away}</div>
              <div class="mLine"><span class="mk">Pts%</span><span class="mv">{fmt_pct(g("away","points_pct"))}</span></div>
              <div class="mLine"><span class="mk">GD/GP</span><span class="mv">{fmt_signed(g("away","gd_per_g"))}</span></div>
              <div class="mLine"><span class="mk">L10 Pts%</span><span class="mv">{fmt_pct(g("away","l10_points_pct"))}</span></div>
              <div class="mLine"><span class="mk">PP / PK</span><span class="mv">{fmt_pct(g("away","pp_pct"))} / {fmt_pct(g("away","pk_pct"))}</span></div>
              <div class="mLine"><span class="mk">SF / SA</span><span class="mv">{fmt_num(g("away","sf_per_g"),1)} / {fmt_num(g("away","sa_per_g"),1)}</span></div>
              <div class="mLine"><span class="mk">Rest</span><span class="mv">{fmt_num(g("away","rest_days"),0)}</span></div>
            </div>

            <div class="mCol {'pickGlow' if h_pick else ''}">
              <div class="mTeam">{home}</div>
              <div class="mLine"><span class="mk">Pts%</span><span class="mv">{fmt_pct(g("home","points_pct"))}</span></div>
              <div class="mLine"><span class="mk">GD/GP</span><span class="mv">{fmt_signed(g("home","gd_per_g"))}</span></div>
              <div class="mLine"><span class="mk">L10 Pts%</span><span class="mv">{fmt_pct(g("home","l10_points_pct"))}</span></div>
              <div class="mLine"><span class="mk">PP / PK</span><span class="mv">{fmt_pct(g("home","pp_pct"))} / {fmt_pct(g("home","pk_pct"))}</span></div>
              <div class="mLine"><span class="mk">SF / SA</span><span class="mv">{fmt_num(g("home","sf_per_g"),1)} / {fmt_num(g("home","sa_per_g"),1)}</span></div>
              <div class="mLine"><span class="mk">Rest</span><span class="mv">{fmt_num(g("home","rest_days"),0)}</span></div>
            </div>
          </div>
        </div>
        """

        rows_html.append(
            f"""
            <tr>
              <td class="muted">{r.get("time_local","")}</td>
              <td><b>{away}</b> @ <b>{home}</b><div class="sub muted">Game {int(r.get("game_id"))}</div></td>
              <td class="center">{fmt_pct(p_away)}</td>
              <td class="center">{fmt_pct(p_home)}</td>
              <td class="center mono">{fmt_ml(away_ml)}</td>
              <td class="center mono">{fmt_ml(home_ml)}</td>
              <td class="center">{chip(lean, lean_cls)}</td>
              <td>{metrics_block}</td>
            </tr>
            """
        )

    rows_html_str = "\n".join(rows_html) if rows_html else "<tr><td colspan='8'>No games found.</td></tr>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NHL Moneyline Model — {date_ymd}</title>
  <style>
    :root {{
      --bg:#0b0e14; --panel:#111624; --text:#e8ecf2; --muted:#aab3c2;
      --line:rgba(255,255,255,.10); --chip:rgba(255,255,255,.10); --pos:rgba(70,200,120,.22);
    }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; }}
    .wrap {{ max-width:1200px; margin:28px auto; padding:0 16px; }}
    .head {{ display:flex; justify-content:space-between; align-items:flex-end; gap:12px; margin-bottom:14px; }}
    .title {{ font-size:22px; font-weight:800; letter-spacing:.2px; }}
    .meta {{ color:var(--muted); font-size:13px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden; }}
    .summaryCard {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px; margin-bottom:14px; }}
    .summaryTitle {{ font-weight:800; margin-bottom:10px; }}
    /* Model Summary tiles: force 2 columns (prevents GH Pages overflow) */
    /* Model Summary tiles: force 2 columns (prevents GH Pages overflow) */
    .summaryGrid{{
      display:grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap:10px;
    }}
    
    /* Mobile: stack */
    @media (max-width: 720px){{
      .summaryGrid{{ grid-template-columns: 1fr; }}
    }}
    
    /* Prevent inner content from forcing overflow */
    .summaryStat{{ min-width:0; }}



    .summaryStat {{ background: rgba(255,255,255,.04); border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .summaryStat .k {{ font-size:12px; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }}
    .summaryStat .v {{ margin-top:6px; font-size:16px; font-weight:800; }}

    .summaryStat.highlight {{
      border-color: rgba(90,170,255,.75);
      box-shadow:
        0 0 0 1px rgba(90,170,255,.45) inset,
        0 0 18px rgba(90,170,255,.12);
      background: linear-gradient(
        180deg,
        rgba(90,170,255,.10) 0%,
        rgba(90,170,255,.03) 100%
      );
    }}
    .summaryStat.highlight .k {{
      color: rgba(150,210,255,1);
      font-weight: 900;
      letter-spacing: .10em;
    }}
    .summaryStat.highlight .v {{
      color: rgba(220,240,255,1);
      font-weight: 900;
    }}

    .recapBox {{ background: rgba(255,255,255,.04); border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .recapRow {{ display:grid; grid-template-columns: 1.2fr 1.1fr 1fr; gap:10px; padding:8px 6px; border-bottom:1px solid rgba(255,255,255,.08); }}
    .recapRow:last-child {{ border-bottom:none; }}
    .recapLeft {{ font-weight:700; }}
    .recapMid {{ font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Courier New",monospace; }}
    .recapRight {{ text-align:right; }}
    .trend {{ display:inline-block; margin:0 6px; font-weight:900; }}
    .trendUp {{ color: rgba(70,200,120,.95); }}
    .trendDown {{ color: rgba(255,90,90,.95); }}
    .trendFlat {{ color: rgba(255,255,255,.55); }}
    .bucketGrid {{ display:grid; gap:6px; margin-top:8px; }}
    @media (max-width: 1100px) {{ .bucketGrid {{ grid-template-columns: repeat(2, 1fr); }} }}

    .bucketChip {{ background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.10); border-radius:10px; padding:8px; }}
    .bucketChip .bk {{ font-size:11px; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }}
    .bucketChip .bv {{ margin-top:4px; font-size:12px; font-weight:800; }}
    .bucketChip .bn {{ margin-top:2px; font-size:11px; color:var(--muted); }}

    .sparkCard {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin-bottom:14px; }}
    .sparkHdr {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; margin-bottom:10px; }}
    .sparkTitle {{ font-weight:800; }}
    .sparkMeta {{ font-size:12px; color:var(--muted); }}
    .spark {{ display:block; width:100%; height:auto; }}
    /* ---------- Mobile table overflow fix ---------- */
    
    /* Wrap the table in a scroll container */
    .tableScroll{{
      width:100%;
      overflow-x:auto;
      -webkit-overflow-scrolling: touch;
    }}
    
    /* Keep table from shrinking weirdly */
    .tableScroll table{{
      min-width: 980px;      /* adjust if you add/remove columns */
      width:100%;
    }}
    
    /* Mobile tightening */
    @media (max-width: 720px){{
      th, td{{
        padding:10px 10px;
      }}
      .wrap{{
        padding:0 10px;
      }}
      .title{{
        font-size:20px;
      }}
      th{{
        font-size:11px;
      }}
    }}


    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ padding:12px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ text-align:left; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
    tr:last-child td {{ border-bottom:none; }}
    .muted {{ color:var(--muted); }}
    .sub {{ margin-top:3px; font-size:12px; }}
    .center {{ text-align:center; }}
    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Courier New",monospace; }}
    .chip {{ display:inline-block; padding:6px 10px; border-radius:999px; background:var(--chip); border:1px solid var(--line); font-size:12px; }}
    .chip.pos {{ background:var(--pos); }}
    .chip.neutral {{ background:rgba(255,255,255,.08); }}
    .mBox {{ background: rgba(255,255,255,.04); border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .mHdr {{ display:flex; align-items:center; margin-bottom:8px; }}
    .mGrid {{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }}
    .mCol {{ border:1px solid rgba(255,255,255,.08); border-radius:10px; padding:8px; background: rgba(0,0,0,.10); }}
    .pickGlow {{ box-shadow: 0 0 0 1px rgba(70,200,120,.25) inset; border-color: rgba(70,200,120,.25); }}
    .mTeam {{ font-weight:800; margin-bottom:6px; }}
    .mLine {{ display:flex; justify-content:space-between; gap:10px; font-size:12px; line-height:1.35; }}
    .mk {{ color: var(--muted); }}
    .mv {{ font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Courier New",monospace; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>
        <div class="title">NHL Moneyline Model</div>
        <div class="meta">{date_ymd} • timezone: {tz_name} • source: api-web.nhle.com • k={CALIBRATION_K} • C=0.3 • home_bias={HOME_ICE_BIAS} • recency_halflife={RECENCY_HALFLIFE_DAYS}d</div>
      </div>
      <div class="meta mono">learned + regularized</div>
    </div>

    {spark_html}
    {summary_html}

    <div class="card">
        <div class="tableScroll">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Matchup</th>
                <th class="center">P(Away)</th>
                <th class="center">P(Home)</th>
                <th class="center">Away ML</th>
                <th class="center">Home ML</th>
                <th class="center">Lean</th>
                <th>Predicted Team Metrics</th>
              </tr>
            </thead>
            <tbody>{rows_html_str}</tbody>
          </table>
        </div>
      </div>
</body>
</html>
"""


# =========================
# PREDICTION RATES CSV (Tidbyt-friendly)
# =========================
def write_todays_picks_csv(df: pd.DataFrame, as_of_date: str) -> None:
    """
    Write today's picks as a 3-row CSV — one column per game, sorted by confidence.
    Row 1 (picks):      team abbreviations  e.g. COL, BOS, PIT ...
    Row 2 (moneylines): fair ML for pick    e.g. -273, -187, -166 ...
    Designed to be polled by Tidbyt.
    """
    ensure_outdir(OUT_DIR)

    if df is None or df.empty:
        log("⚠️  todays_picks.csv: no games today, skipping.")
        return

    tmp = df.copy()
    tmp["p_home"] = pd.to_numeric(tmp["p_home"], errors="coerce")
    tmp["conf"] = tmp["p_home"].apply(lambda p: max(float(p), 1.0 - float(p)) if pd.notna(p) else 0.5)
    tmp = tmp.sort_values("conf", ascending=False)
    tmp = tmp[tmp["predicted_winner"].notna()].copy()

    if tmp.empty:
        log("⚠️  todays_picks.csv: no predicted winners, skipping.")
        return

    picks = tmp["predicted_winner"].tolist()

    # Pull the correct moneyline for each pick (away_ml or home_ml)
    moneylines = []
    for _, row in tmp.iterrows():
        pick = row["predicted_winner"]
        if pick == row.get("home"):
            ml = row.get("home_ml_fair")
        else:
            ml = row.get("away_ml_fair")
        # Format as +120 / -273 string
        try:
            ml_val = int(ml)
            moneylines.append(f"{ml_val:+d}")
        except (TypeError, ValueError):
            moneylines.append("")

    cols = [f"pick_{i+1}" for i in range(len(picks))]
    out = pd.DataFrame([picks, moneylines], columns=cols)
    out.insert(0, "row", ["team", "ml"])
    out.to_csv(PICKS_PATH, index=False)

    # JSON version — cleaner for apps that prefer structured data
    # Format: {"as_of": "2026-02-26", "picks": [{"team": "COL", "ml": "-273"}, ...]}
    picks_json = {
        "as_of": as_of_date,
        "picks": [{"team": t, "ml": ml} for t, ml in zip(picks, moneylines)],
    }
    with open(PICKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(picks_json, f, indent=2)

    log(f"🏒 Saved today's picks ({len(picks)} games): {PICKS_PATH} + {PICKS_JSON_PATH}")


def write_prediction_rates_csv(history: pd.DataFrame, as_of_date: str) -> None:
    """
    Write a compact prediction_rates.csv with one row per time window.
    Designed to be polled by Tidbyt (or any simple HTTP client) via GitHub raw URL.

    Columns: as_of, time_period, prediction_rate, games_scored
    """
    ensure_outdir(OUT_DIR)

    h = history.copy()
    if h.empty:
        log("⚠️  prediction_rates.csv: history empty, skipping.")
        return

    if "date_ymd" not in h.columns:
        return

    h["date_ymd"] = h["date_ymd"].astype(str)
    h["is_correct"] = pd.to_numeric(h.get("is_correct"), errors="coerce")
    scored = h[h["is_correct"].isin([0, 1])].copy()

    if scored.empty:
        log("⚠️  prediction_rates.csv: no scored games yet, skipping.")
        return

    # Latest scored date drives all window calculations
    latest_scored = scored["date_ymd"].max()

    def window_acc(days: Optional[int]) -> dict:
        """Return accuracy + game count for a rolling window (None = all-time)."""
        if days is None:
            sub = scored
        else:
            d1 = dt.date.fromisoformat(latest_scored)
            d0 = d1 - dt.timedelta(days=days - 1)
            sub = scored[scored["date_ymd"] >= d0.isoformat()]

        n = len(sub)
        if n == 0:
            return {"prediction_rate": "", "games_scored": 0}
        acc = float(sub["is_correct"].mean())
        return {
            "prediction_rate": f"{acc * 100:.1f}%",
            "games_scored": n,
        }

    # Yesterday = the most recent date that has scored games
    yesterday_scored = scored[scored["date_ymd"] == latest_scored]
    n_y = len(yesterday_scored)
    acc_y = float(yesterday_scored["is_correct"].mean()) if n_y > 0 else None

    windows = [
        ("All-Time",  window_acc(None)),
        ("Last90d",   window_acc(90)),
        ("Last60d",   window_acc(60)),
        ("Last30d",   window_acc(30)),
        ("Last7d",    window_acc(7)),
        ("Yday",      {
            "prediction_rate": f"{acc_y * 100:.1f}%" if acc_y is not None else "",
            "games_scored": n_y,
        }),
    ]

    rows = [
        {
            "as_of": as_of_date,
            "time_period": label,
            "prediction_rate": data["prediction_rate"],
            "games_scored": data["games_scored"],
        }
        for label, data in windows
    ]

    pd.DataFrame(rows).to_csv(RATES_PATH, index=False)

    # JSON version — cleaner for apps that prefer structured data
    # Format: {"as_of": "2026-02-26", "rates": [{"time_period": "All-Time", "prediction_rate": "60.9%", "games_scored": 2251}, ...]}
    rates_json = {
        "as_of": as_of_date,
        "rates": rows,
    }
    with open(RATES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rates_json, f, indent=2)

    log(f"📊 Saved prediction rates: {RATES_PATH} + {RATES_JSON_PATH}")


# =========================
# RUN ONE DATE
# =========================
def run_for_date(
    date_ymd: str,
    tz: ZoneInfo,
    schedule_cache: Dict[str, List[dict]],
    model_pipe: Optional[Pipeline],
    use_clubstats: bool,
    write_daily: bool,
    max_teams: Optional[int],
    debug_print: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    games = load_scoreboard(date_ymd)
    if not games:
        return None, None

    team_table = build_team_feature_table(
        asof_ymd=date_ymd,
        schedule_cache=schedule_cache,
        use_clubstats=use_clubstats,
        max_teams=max_teams,
    )

    out_rows = []
    for idx, g in enumerate(games):
        away_sched = schedule_cache.get(g.away_abbrev, [])
        home_sched = schedule_cache.get(g.home_abbrev, [])
        away_f = schedule_pressure(away_sched, game_date=date_ymd)
        home_f = schedule_pressure(home_sched, game_date=date_ymd)

        away_m = team_metrics_snapshot(g.away_abbrev, team_table)
        home_m = team_metrics_snapshot(g.home_abbrev, team_table)
        feat = build_game_features(away_m, home_m, away_f, home_f)

        p_home_raw = predict_prob_home(model_pipe, feat)
        if p_home_raw is None:
            p_home_raw = 0.5

        p_home = shrink_to_50(p_home_raw, k=CALIBRATION_K)

        # --- Home ice bias nudge ---
        # Shift probability toward home team by HOME_ICE_BIAS, then renormalize.
        # Applied after calibration so it doesn't interact with the shrink step.
        if HOME_ICE_BIAS and HOME_ICE_BIAS != 0.0:
            p_home = min(0.999, max(0.001, p_home + HOME_ICE_BIAS))

        p_away = 1.0 - p_home

        if debug_print and idx < 6:
            raw_conf = max(p_home_raw, 1 - p_home_raw)
            cal_conf = max(p_home, 1 - p_home)
            log(f"DEBUG {g.away_abbrev}@{g.home_abbrev} raw={p_home_raw:.3f} cal={p_home:.3f} raw_conf={raw_conf:.3f} cal_conf={cal_conf:.3f}")

        away_ml = moneyline_from_prob(p_away)
        home_ml = moneyline_from_prob(p_home)

        pred_win = predicted_winner(g.away_abbrev, g.home_abbrev, p_home)

        is_final = g.game_state in {"FINAL", "OFF"}
        actual_win = actual_winner_from_scores(g.away_abbrev, g.home_abbrev, g.away_score, g.home_score) if is_final else None
        is_correct = None
        if pred_win is not None and actual_win is not None:
            is_correct = 1 if pred_win == actual_win else 0

        out_rows.append(
            {
                "date_ymd": date_ymd,
                "time_local": format_local_time(g.start_utc, tz),
                "game_id": g.game_id,
                "away": g.away_abbrev,
                "home": g.home_abbrev,
                "game_state": g.game_state,
                "away_score": g.away_score,
                "home_score": g.home_score,
                "p_home_raw": float(p_home_raw),
                "p_home": float(p_home),
                "p_away": float(p_away),
                "away_ml_fair": away_ml,
                "home_ml_fair": home_ml,
                "predicted_winner": pred_win,
                "actual_winner": actual_win,
                "is_correct": is_correct,

                "away_points_pct": away_m.get("points_pct"),
                "away_gd_per_g": away_m.get("gd_per_g"),
                "away_l10_points_pct": away_m.get("l10_points_pct"),
                "away_l10_gd_per_g": away_m.get("l10_gd_per_g"),
                "away_pp_pct": away_m.get("pp_pct"),
                "away_pk_pct": away_m.get("pk_pct"),
                "away_sf_per_g": away_m.get("sf_per_g"),
                "away_sa_per_g": away_m.get("sa_per_g"),
                "away_rest_days": away_m.get("rest_days"),

                "home_points_pct": home_m.get("points_pct"),
                "home_gd_per_g": home_m.get("gd_per_g"),
                "home_l10_points_pct": home_m.get("l10_points_pct"),
                "home_l10_gd_per_g": home_m.get("l10_gd_per_g"),
                "home_pp_pct": home_m.get("pp_pct"),
                "home_pk_pct": home_m.get("pk_pct"),
                "home_sf_per_g": home_m.get("sf_per_g"),
                "home_sa_per_g": home_m.get("sa_per_g"),
                "home_rest_days": home_m.get("rest_days"),

                **feat,
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df, None

    df["conf"] = df["p_home"].apply(lambda p: max(float(p), 1.0 - float(p)))
    df = df.sort_values(["conf", "time_local"], ascending=[False, True], na_position="last")

    upsert_history(df)

    if debug_print:
        tmp = df.copy()
        tmp["raw_conf"] = tmp["p_home_raw"].apply(lambda p: max(float(p), 1.0 - float(p)))
        tmp["cal_conf"] = tmp["p_home"].apply(lambda p: max(float(p), 1.0 - float(p)))
        top = tmp.sort_values("raw_conf", ascending=False).head(10)[["away", "home", "p_home_raw", "p_home", "raw_conf", "cal_conf"]]
        log("\nTop-10 most confident (RAW):\n" + top.to_string(index=False))
        log(f"\nShrink sanity: mean |cal-raw| = {(tmp['p_home']-tmp['p_home_raw']).abs().mean():.4f}  max = {(tmp['p_home']-tmp['p_home_raw']).abs().max():.4f}")

    html_text = None
    if write_daily:
        ensure_outdir(OUT_DIR)
        csv_path = os.path.join(OUT_DIR, f"nhl_moneyline_{date_ymd}.csv")
        html_path = os.path.join(OUT_DIR, f"nhl_moneyline_{date_ymd}.html")
        df.to_csv(csv_path, index=False)

        hist_now = load_history()
        hist_now = refresh_recent_finals_in_history(hist_now, tz, lookback_days=3)
        summary_html = build_html_summary_block(hist_now, today_ymd=date_ymd)
        spark_html = build_accuracy_sparkline_block(hist_now, window_games=50, max_points=140)

        html_text = render_html(
            df,
            date_ymd=date_ymd,
            tz_name=str(tz.key),
            summary_html=summary_html,
            spark_html=spark_html,
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)

        # Write Tidbyt-friendly prediction rates CSV
        write_prediction_rates_csv(hist_now, as_of_date=date_ymd)

        # Write Tidbyt-friendly today's picks CSV
        write_todays_picks_csv(df, as_of_date=date_ymd)

        log(f"Saved daily: {csv_path}")
        log(f"Saved daily: {html_path}")

    return df, html_text


# =========================
# OPTIONAL: RESCORE HISTORY WITH FINAL MODEL
# =========================
def rescore_all_history(
    history: pd.DataFrame,
    schedule_cache: Dict[str, List[dict]],
    tz: ZoneInfo,
    season_start: Optional[str],
    max_teams: Optional[int],
) -> pd.DataFrame:
    if history.empty:
        return history

    final_pipe = train_model_from_history(history, season_start=season_start, train_through_date=None)
    if final_pipe is None:
        log("Rescore: final model unavailable (insufficient labeled games). Skipping.")
        return history

    df = history.copy()
    if "date_ymd" not in df.columns:
        log("Rescore: history missing date_ymd. Skipping.")
        return history

    df["date_ymd"] = df["date_ymd"].astype(str)
    all_dates = sorted(df["date_ymd"].dropna().unique().tolist())
    log(f"🔁 Rescore history: {len(all_dates)} dates…")

    new_rows = []
    for i, d in enumerate(all_dates, 1):
        if i == 1 or i % 20 == 0 or i == len(all_dates):
            log(f"  …date {i}/{len(all_dates)}: {d}")

        day = df[df["date_ymd"] == d].copy()
        if day.empty:
            continue

        team_table = build_team_feature_table(
            asof_ymd=d,
            schedule_cache=schedule_cache,
            use_clubstats=False,
            max_teams=max_teams,
        )

        for _, r in day.iterrows():
            away = str(r.get("away") or "")
            home = str(r.get("home") or "")
            if not away or not home:
                new_rows.append(r.to_dict())
                continue

            away_sched = schedule_cache.get(away, [])
            home_sched = schedule_cache.get(home, [])
            away_f = schedule_pressure(away_sched, game_date=d)
            home_f = schedule_pressure(home_sched, game_date=d)

            away_m = team_metrics_snapshot(away, team_table)
            home_m = team_metrics_snapshot(home, team_table)
            feat = build_game_features(away_m, home_m, away_f, home_f)

            p_home_raw = predict_prob_home(final_pipe, feat)
            if p_home_raw is None:
                p_home_raw = 0.5
            p_home = shrink_to_50(p_home_raw, k=CALIBRATION_K)
            p_away = 1.0 - p_home

            row = r.to_dict()
            row["p_home_raw"] = float(p_home_raw)
            row["p_home"] = float(p_home)
            row["p_away"] = float(p_away)
            row["away_ml_fair"] = moneyline_from_prob(p_away)
            row["home_ml_fair"] = moneyline_from_prob(p_home)
            row["predicted_winner"] = predicted_winner(away, home, p_home)

            aw = row.get("actual_winner")
            pw = row.get("predicted_winner")
            row["is_correct"] = (
                1 if (isinstance(aw, str) and isinstance(pw, str) and aw and pw and aw == pw)
                else (0 if (isinstance(aw, str) and isinstance(pw, str) and aw and pw) else None)
            )

            for k, v in feat.items():
                row[k] = v

            new_rows.append(row)

    out = pd.DataFrame(new_rows)
    out = out.drop_duplicates(subset=["game_id"], keep="last")
    out = score_final_games_in_history(out)
    out.to_csv(HISTORY_PATH, index=False)
    log(f"✅ Rescore complete. Overwrote: {HISTORY_PATH}")
    return out


# =========================
# MAIN
# =========================
def main() -> None:
    args = parse_args()
    tz = ZoneInfo(args.tz)
    ensure_outdir(OUT_DIR)

    backfill_mode = bool(args.backfill_start and args.backfill_end)
    anchor_date = args.backfill_end if backfill_mode else (args.date or dt.datetime.now(tz).date().isoformat())

    schedule_cache = build_schedule_cache(anchor_date, max_teams=args.max_teams)
    hist = load_history()

    if backfill_mode:
        dates = daterange_inclusive(args.backfill_start, args.backfill_end)
        use_clubstats = (args.backfill_disable_clubstats == 0)

        log(f"🔁 Backfill (walk-forward): {dates[0]} → {dates[-1]} ({len(dates)} days)")
        log(f"   master only: {HISTORY_PATH}")
        log(f"   club-stats during backfill: {'ON (leaky)' if use_clubstats else 'OFF (recommended)'}")
        log(f"   calibration k={CALIBRATION_K} | retrain_every={args.retrain_every}")

        total_games = 0
        days_with_games = 0
        model_pipe: Optional[Pipeline] = None

        for i, d in enumerate(dates, 1):
            log(f"\n📅 [{i}/{len(dates)}] {d}")

            do_train = (i == 1) or (args.retrain_every and (i - 1) % int(args.retrain_every) == 0)
            if do_train:
                hist_now = load_history()
                model_pipe = train_model_from_history(hist_now, season_start=args.season_start, train_through_date=d)
                if model_pipe is None:
                    log(f"   🧠 Model: None (fallback 50/50) | trained_through < {d}")
                else:
                    log(f"   🧠 Model: OK | trained_through < {d}")

            try:
                df_day, _ = run_for_date(
                    date_ymd=d,
                    tz=tz,
                    schedule_cache=schedule_cache,
                    model_pipe=model_pipe,
                    use_clubstats=use_clubstats,
                    write_daily=False,
                    max_teams=args.max_teams,
                    debug_print=(i == 1),
                )

                if df_day is None or df_day.empty:
                    log("   (no games)")
                else:
                    days_with_games += 1
                    total_games += len(df_day)

                    tmp = df_day.copy()
                    tmp["pick_prob"] = tmp.apply(lambda r: max(float(r["p_home"]), float(r["p_away"])), axis=1)
                    top3 = tmp.sort_values("pick_prob", ascending=False).head(3)

                    parts = []
                    for _, rr in top3.iterrows():
                        parts.append(f'{rr["away"]}@{rr["home"]} {rr["predicted_winner"]} ({100*rr["pick_prob"]:.1f}%)')
                    log("   top picks: " + " | ".join(parts))

            except Exception as e:
                log(f"❌ Error on {d}: {e}")

            if args.sleep and args.sleep > 0:
                time.sleep(args.sleep)

        log(f"\n✅ Backfill complete. Days w/ games: {days_with_games} | Total games: {total_games}")

        if args.rescore_history:
            hist_now = load_history()
            _ = rescore_all_history(
                history=hist_now,
                schedule_cache=schedule_cache,
                tz=tz,
                season_start=args.season_start,
                max_teams=args.max_teams,
            )

    else:
            # ✅ FIXED: Determine which date(s) to process
            now = dt.datetime.now(tz)
            today = now.date()
            
            if args.date:
                # User specified explicit date
                dates_to_process = [args.date]
            elif args.process_yesterday:
                # Process both yesterday and today
                yesterday = today - dt.timedelta(days=1)
                dates_to_process = [yesterday.isoformat(), today.isoformat()]
                log(f"📅 Daily run with --process_yesterday")
            else:
                # Default: just today
                dates_to_process = [today.isoformat()]
            
            log(f"📅 Daily run: {dates_to_process} ({args.tz})")
            log(f"write_daily={'ON' if args.write_daily else 'OFF'} | calibration k={CALIBRATION_K}")
            log("club-stats: ON (daily mode)")
    
            if args.publish_github and not args.write_daily:
                log("⚠️ --publish_github requested without --write_daily. (Publishing requires HTML; enabling write_daily is recommended.)")
    
            # Build schedule cache once (using today as anchor)
            schedule_cache = build_schedule_cache(today.isoformat(), max_teams=args.max_teams)
            
            # ✅ FIXED: Process each date
            all_dfs = []
            html_text = None
            
            for date_ymd in dates_to_process:
                log(f"\n{'='*60}")
                log(f"Processing date: {date_ymd}")
                log(f"{'='*60}")
                
                hist_now = load_history()
                model_pipe = train_model_from_history(hist_now, season_start=args.season_start, train_through_date=date_ymd)
    
                df_day, day_html = run_for_date(
                    date_ymd=date_ymd,
                    tz=tz,
                    schedule_cache=schedule_cache,
                    model_pipe=model_pipe,
                    use_clubstats=not args.no_clubstats,
                    write_daily=bool(args.write_daily),
                    max_teams=args.max_teams,
                    debug_print=True,
                )

    
                if df_day is not None and not df_day.empty:
                    all_dfs.append(df_day)
                    
                    # Keep HTML from the LAST date processed (for publishing)
                    if day_html:
                        html_text = day_html
                    
                    show = df_day.copy()
                    show["P(A)"] = show["p_away"].map(fmt_pct)
                    show["P(H)"] = show["p_home"].map(fmt_pct)
                    show["AwayML"] = show["away_ml_fair"].map(fmt_ml)
                    show["HomeML"] = show["home_ml_fair"].map(fmt_ml)
                    show = show[["time_local", "away", "home", "P(A)", "P(H)", "AwayML", "HomeML", "predicted_winner", "game_id"]]
                    log("\n" + show.to_string(index=False))
                else:
                    log(f"⚠️ No games found for {date_ymd}.")

            # ✅ FIXED: GitHub Publish (only for the latest/last date processed)
            if args.publish_github:
                if not html_text:
                    raise RuntimeError("publish_github: HTML not generated. Run with --write_daily so html_text exists.")
                
                # Use the last date in the list for publishing
                publish_date = dates_to_process[-1]
                
                repo = args.github_repo or os.getenv(GITHUB_REPO_ENV) or GITHUB_REPO
                branch = args.github_branch or os.getenv(GITHUB_BRANCH_ENV) or GITHUB_BRANCH
                docs_dir = args.github_docs_dir or os.getenv(GITHUB_DOCS_DIR_ENV) or GITHUB_DOCS_DIR
                token_env = args.github_token_env or os.getenv(GITHUB_TOKEN_ENV_ENV) or GITHUB_TOKEN_ENV
                latest_name = args.github_latest_name or os.getenv(GITHUB_LATEST_NAME_ENV) or GITHUB_LATEST_NAME
                archive_dir = args.github_archive_dir or os.getenv(GITHUB_ARCHIVE_DIR_ENV) or GITHUB_ARCHIVE_DIR
    
                if not repo:
                    raise RuntimeError("GitHub publish: missing repo. Use --github_repo owner/repo or set env GITHUB_REPO.")
                if "/" not in repo:
                    raise RuntimeError(f"GitHub publish: repo must be 'owner/repo' (got: {repo})")
    
                publish_html_to_github_pages(
                    html_text=html_text,
                    date_ymd=publish_date,
                    repo=repo,
                    branch=branch,
                    token_env=token_env,
                    docs_dir=docs_dir,
                    latest_name=latest_name,
                    archive_dir=archive_dir,
                )

            # retrain after daily update (so tomorrow learns)
            hist2 = load_history()
            _ = train_model_from_history(hist2, season_start=args.season_start, train_through_date=None)

    # Season metrics
    hist_final = load_history()
    summary, buckets = compute_season_metrics(hist_final, season_start=args.season_start, eval_last_n=args.eval_last_n)

    log("\nSeason metrics:")
    log(summary.to_string(index=False))

    if not buckets.empty:
        log("\nAccuracy by confidence bucket:")
        log(buckets.to_string(index=False))

    if "p_home_raw" in hist_final.columns and "p_home" in hist_final.columns:
        tmp = hist_final.copy()
        tmp["p_home_raw"] = pd.to_numeric(tmp["p_home_raw"], errors="coerce")
        tmp["p_home"] = pd.to_numeric(tmp["p_home"], errors="coerce")
        tmp = tmp[tmp["p_home"].notna() & tmp["p_home_raw"].notna()]
        if len(tmp) > 0:
            log("\nSanity check (raw vs calibrated):")
            log(f"raw min/max: {tmp['p_home_raw'].min():.3f} / {tmp['p_home_raw'].max():.3f}")
            log(f"cal min/max: {tmp['p_home'].min():.3f} / {tmp['p_home'].max():.3f}")
            log(f"mean abs diff: {(tmp['p_home'] - tmp['p_home_raw']).abs().mean():.4f}")

    log(f"\nMaster CSV: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
