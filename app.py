import json
from collections import Counter
import random
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from run_matchday_simulation import CALENDAR_WITH_RESULTS_PATH, SIM_CONFIG_PATH


st.set_page_config(page_title="The Betting Scientist", layout="wide")

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

:root {
    --pitch-1: #22c55e;
    --pitch-2: #14532d;
    --slate: #0b1220;
    --card: #111827;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Manrope', 'Segoe UI', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 20%, rgba(34, 197, 94, 0.07), transparent 35%),
                radial-gradient(circle at 70% 0%, rgba(34, 197, 94, 0.05), transparent 30%),
                #0f172a;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--pitch-1), var(--pitch-2));
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.card {
    background: var(--card);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
}

.pill {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

.pill-league {
    background: rgba(34, 197, 94, 0.14);
    color: #34d399;
}

.pill-form {
    margin-right: 0.25rem;
    color: #0b1220;
}

.pill-W { background: #22c55e; }
.pill-D { background: #fbbf24; }
.pill-L { background: #ef4444; }

.headline {
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.4rem;
}

.metric {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.95rem;
}

.metric strong {
    display: block;
    color: #cbd5e1;
}

.score-sample {
    font-size: 2.5rem;
    font-weight: 800;
    color: #f8fafc;
    text-align: center;
    margin: 0.4rem 0;
}

.top-scores span {
    display: inline-block;
    margin-right: 0.35rem;
    padding: 0.25rem 0.5rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.05);
    color: #cbd5e1;
    font-size: 0.9rem;
}

.muted {
    color: var(--muted);
    font-size: 0.9rem;
}

.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

MAJOR_LEAGUES = ["Serie A", "Premier League", "Ligue 1", "Bundesliga 1"]
SIM_ITERATIONS = 20000


@st.cache_data(show_spinner=False)
def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_calendar(calendar_path: Path, cache_bust: float) -> pd.DataFrame:
    # cache_bust ties the cache to the file's mtime so updated calendars refresh automatically
    return pd.read_csv(calendar_path)


def prepare_calendar(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Matchday"] = pd.to_numeric(prepared["Matchday"], errors="coerce").astype("Int64")
    for col in ["FTHG", "FTAG", "HTHG", "HTAG"]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    prepared["FTR"] = prepared["FTR"].replace("", pd.NA)
    prepared["HTR"] = prepared["HTR"].replace("", pd.NA)
    prepared["Country"] = prepared["Country"].fillna("Unknown")
    return prepared


def league_options(calendar_df: pd.DataFrame) -> List[str]:
    leagues = [league for league in calendar_df["League"].dropna().unique()]
    return sorted(leagues)


def country_options(calendar_df: pd.DataFrame) -> List[str]:
    countries = [country for country in calendar_df["Country"].dropna().unique()]
    return sorted(countries)


def matchday_range(calendar_df: pd.DataFrame, league: str) -> Tuple[int, int]:
    matchdays = pd.to_numeric(
        calendar_df.loc[calendar_df["League"] == league, "Matchday"], errors="coerce"
    ).dropna()
    if matchdays.empty:
        return 1, 1
    return int(matchdays.min()), int(matchdays.max())


def next_matchday_to_simulate(calendar_df: pd.DataFrame, league: str) -> Optional[int]:
    """Return the next matchday number: (latest completed matchday) + 1."""
    league_df = calendar_df[calendar_df["League"] == league]
    if league_df.empty:
        return None

    matchdays = pd.to_numeric(league_df["Matchday"], errors="coerce").dropna()
    if matchdays.empty:
        return None

    completed = league_df[league_df["FTR"].notna()]
    last_completed = pd.to_numeric(completed["Matchday"], errors="coerce").dropna()

    if last_completed.empty:
        return int(matchdays.min())

    next_md = int(last_completed.max()) + 1
    md_max = int(matchdays.max())
    return next_md if next_md <= md_max else None


def league_country(calendar_df: pd.DataFrame, league: str) -> Optional[str]:
    countries = calendar_df.loc[calendar_df["League"] == league, "Country"].dropna().unique()
    return countries[0] if len(countries) else None


def init_dashboard_state(calendar_df: pd.DataFrame, allowed_leagues: List[str]) -> None:
    """Initialize session state with a random allowed league and its next matchday."""
    if "active_league" in st.session_state and "active_matchday" in st.session_state:
        return

    candidates = [l for l in MAJOR_LEAGUES if l in allowed_leagues] or allowed_leagues
    random.shuffle(candidates)

    chosen_league = None
    chosen_md = None
    for league in candidates:
        md = next_matchday_to_simulate(calendar_df, league)
        if md is None:
            continue
        if league_matchday_rows(calendar_df, league, md).empty:
            continue
        chosen_league = league
        chosen_md = md
        break

    if chosen_league is None:
        if calendar_df.empty:
            return
        chosen_league = sorted(calendar_df["League"].dropna().unique())[0]
        chosen_md = next_matchday_to_simulate(calendar_df, chosen_league) or matchday_range(calendar_df, chosen_league)[0]

    st.session_state["active_league"] = chosen_league
    st.session_state["active_matchday"] = chosen_md
    st.session_state["active_country"] = league_country(calendar_df, chosen_league) or "Unknown"


@st.cache_data(show_spinner=False)
def compute_strengths_for_columns(
    calendar_df: pd.DataFrame,
    league: str,
    home_goal_col: str,
    away_goal_col: str,
    up_to_matchday: Optional[int] = None,
    fallback_home_avg: float = 1.4,
    fallback_away_avg: float = 1.1,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    league_df = calendar_df[calendar_df["League"] == league].copy()
    if up_to_matchday is not None:
        league_df = league_df[league_df["Matchday"].notna() & (league_df["Matchday"] < up_to_matchday)]

    completed = league_df.dropna(subset=[home_goal_col, away_goal_col])

    league_home_avg = float(completed[home_goal_col].mean()) if not completed.empty else fallback_home_avg
    league_away_avg = float(completed[away_goal_col].mean()) if not completed.empty else fallback_away_avg
    league_home_avg = max(league_home_avg, 0.2)
    league_away_avg = max(league_away_avg, 0.2)

    teams = set(league_df["Hometeam"]).union(set(league_df["Awayteam"]))
    strengths: Dict[str, Dict[str, float]] = {}

    for team in teams:
        home_matches = completed[completed["Hometeam"] == team]
        away_matches = completed[completed["Awayteam"] == team]

        home_scored = float(home_matches[home_goal_col].mean()) if not home_matches.empty else league_home_avg
        home_conceded = float(home_matches[away_goal_col].mean()) if not home_matches.empty else league_away_avg
        away_scored = float(away_matches[away_goal_col].mean()) if not away_matches.empty else league_away_avg
        away_conceded = float(away_matches[home_goal_col].mean()) if not away_matches.empty else league_home_avg

        strengths[team] = {
            "home_attack": home_scored / league_home_avg if league_home_avg else 1.0,
            "home_defense": home_conceded / league_away_avg if league_away_avg else 1.0,
            "away_attack": away_scored / league_away_avg if league_away_avg else 1.0,
            "away_defense": away_conceded / league_home_avg if league_home_avg else 1.0,
        }

    league_avgs = {"home_goal_avg": league_home_avg, "away_goal_avg": league_away_avg}
    return strengths, league_avgs


def compute_league_strengths(
    calendar_df: pd.DataFrame, league: str, up_to_matchday: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    return compute_strengths_for_columns(
        calendar_df,
        league,
        home_goal_col="FTHG",
        away_goal_col="FTAG",
        up_to_matchday=up_to_matchday,
        fallback_home_avg=1.4,
        fallback_away_avg=1.1,
    )


def compute_halftime_strengths(
    calendar_df: pd.DataFrame, league: str, up_to_matchday: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    # First-half goal averages are naturally lower, so halve the full-time fallbacks as a sensible starting point.
    return compute_strengths_for_columns(
        calendar_df,
        league,
        home_goal_col="HTHG",
        away_goal_col="HTAG",
        up_to_matchday=up_to_matchday,
        fallback_home_avg=0.7,
        fallback_away_avg=0.55,
    )


def expected_goals(
    strengths: Dict[str, Dict[str, float]],
    league_avgs: Dict[str, float],
    home_team: str,
    away_team: str,
) -> Tuple[float, float]:
    default_strength = {"home_attack": 1.0, "home_defense": 1.0, "away_attack": 1.0, "away_defense": 1.0}
    home_strength = strengths.get(home_team, default_strength)
    away_strength = strengths.get(away_team, default_strength)

    home_lambda = home_strength["home_attack"] * away_strength["away_defense"] * league_avgs["home_goal_avg"]
    away_lambda = away_strength["away_attack"] * home_strength["home_defense"] * league_avgs["away_goal_avg"]
    return max(home_lambda, 0.01), max(away_lambda, 0.01)


def simulate_match(
    home_lambda: float, away_lambda: float, iterations: int = 2000, goal_line: float = 2.5
) -> Dict[str, object]:
    home_goals = np.random.poisson(home_lambda, iterations)
    away_goals = np.random.poisson(away_lambda, iterations)

    results = {
        "home_win": float(np.mean(home_goals > away_goals)),
        "away_win": float(np.mean(home_goals < away_goals)),
        "draw": float(np.mean(home_goals == away_goals)),
        "over": float(np.mean((home_goals + away_goals) > goal_line)),
        "btts": float(np.mean((home_goals > 0) & (away_goals > 0))),
        "goal_line": goal_line,
    }

    freq = Counter(zip(home_goals, away_goals))
    total = float(iterations)
    top_scores = [(f"{hg}-{ag}", round((count / total) * 100, 2)) for (hg, ag), count in freq.most_common(3)]

    sample_idx = np.random.randint(0, iterations)
    sample_score = (int(home_goals[sample_idx]), int(away_goals[sample_idx]))

    return {
        **results,
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "top_scores": top_scores,
        "sample_score": sample_score,
    }


def cache_match_predictions(
    match_key: str,
    home_lambda: float,
    away_lambda: float,
    home_lambda_ht: float,
    away_lambda_ht: float,
    iterations: int = SIM_ITERATIONS,
) -> Dict[str, Dict[str, object]]:
    """Cache FT/HT simulations so toggles don't trigger fresh randomness."""
    cache = st.session_state.setdefault("match_predictions_cache", {})
    params = (
        round(home_lambda, 6),
        round(away_lambda, 6),
        round(home_lambda_ht, 6),
        round(away_lambda_ht, 6),
        iterations,
    )
    cached = cache.get(match_key)
    if cached and cached.get("params") == params:
        return cached
    cache[match_key] = {
        "params": params,
        "ft": simulate_match(home_lambda, away_lambda, iterations=iterations, goal_line=2.5),
        "ht": simulate_match(home_lambda_ht, away_lambda_ht, iterations=iterations, goal_line=0.5),
    }
    return cache[match_key]


def form_guide(calendar_df: pd.DataFrame, team: str, before_matchday: Optional[int] = None, max_games: int = 5) -> List[str]:
    team_df = calendar_df[(calendar_df["Hometeam"] == team) | (calendar_df["Awayteam"] == team)]
    team_df = team_df.dropna(subset=["FTR", "FTHG", "FTAG"])
    if before_matchday is not None:
        team_df = team_df[team_df["Matchday"].notna() & (team_df["Matchday"] < before_matchday)]
    if team_df.empty:
        return []
    team_df = team_df.sort_values(["Matchday"]).tail(max_games)

    results: List[str] = []
    for _, row in team_df.iterrows():
        if row["Hometeam"] == team:
            if row["FTR"] == "H":
                results.append("W")
            elif row["FTR"] == "D":
                results.append("D")
            else:
                results.append("L")
        else:
            if row["FTR"] == "A":
                results.append("W")
            elif row["FTR"] == "D":
                results.append("D")
            else:
                results.append("L")
    return results[-max_games:]


def render_form_badges(form: List[str]) -> str:
    if not form:
        return "<span class='muted'>no form data</span>"
    badges = "".join(f"<span class='pill pill-form pill-{r}'>{r}</span>" for r in form)
    return badges


def render_top_scores(top_scores: List[Tuple[str, float]]) -> str:
    if not top_scores:
        return "<span class='muted'>no scores</span>"
    spans = "".join(f"<span>{score} ({prob:.1f}%)</span>" for score, prob in top_scores)
    return f"<div class='top-scores'>{spans}</div>"


def league_matchday_rows(calendar_df: pd.DataFrame, league: str, matchday: int) -> pd.DataFrame:
    league_df = calendar_df[calendar_df["League"] == league]
    return league_df[league_df["Matchday"] == matchday]


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_match_card(
    league: str,
    matchday: int,
    home_team: str,
    away_team: str,
    prediction_ft: Dict[str, object],
    prediction_ht: Dict[str, object],
    form_home: List[str],
    form_away: List[str],
    actual_ft_score: Optional[Tuple[float, float]] = None,
    actual_ht_score: Optional[Tuple[float, float]] = None,
    show_sample: bool = False,
    toggle_key: Optional[str] = None,
) -> None:
    container = st.container()
    is_ht_view = False
    toggle_state_key = f"{toggle_key}_state" if toggle_key else None

    if toggle_state_key and toggle_state_key not in st.session_state:
        st.session_state[toggle_state_key] = False

    with container.form(key=f"{toggle_key}_form" if toggle_key else None, border=False):
        is_ht_view = bool(st.session_state.get(toggle_state_key, False))
        badge_text = "Half time view" if is_ht_view else "Full time view"
        button_label = "Switch to full time view" if is_ht_view else "Switch to half time view"

        header_cols = st.columns([6, 2])
        with header_cols[0]:
            st.markdown(
                dedent(
                    f"""
                    <div style="display:flex; flex-wrap:wrap; gap:0.4rem; align-items:center; margin-bottom:0.25rem;">
                        <span class="pill pill-league">{league}</span>
                        <span class="muted">Matchday {matchday}</span>
                        <span class="pill" style="background: rgba(255,255,255,0.08); color: #cbd5e1;">{badge_text}</span>
                    </div>
                    <div class="headline">{home_team} vs {away_team}</div>
                    """
                ),
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            # Place the toggle next to the badge, aligned with the header row.
            submitted = st.form_submit_button(button_label, use_container_width=True)
            if submitted and toggle_state_key:
                st.session_state[toggle_state_key] = not st.session_state[toggle_state_key]
                is_ht_view = bool(st.session_state[toggle_state_key])

        prediction = prediction_ht if is_ht_view else prediction_ft
        actual_score = actual_ht_score if is_ht_view else actual_ft_score
        actual_label = "Actual HT" if is_ht_view else "Actual FT"
        over_label = f"{'HT ' if is_ht_view else ''}Over {prediction['goal_line']:.1f}"
        btts_label = "HT BTTS" if is_ht_view else "BTTS"
        xg_label = f"{'HT ' if is_ht_view else ''}xG"
        top_scores_label = "Top 3 HT likely scores" if is_ht_view else "Top 3 likely scores"

        actual_block = ""
        if actual_score:
            actual_block = f"<div class='muted'>{actual_label}</div><div class='headline'>{int(actual_score[0])}-{int(actual_score[1])}</div>"

        sample_block = ""
        if show_sample:
            sample_block = (
                f"<div><div class='muted'>Sample reality</div>"
                f"<div class='score-sample'>{prediction['sample_score'][0]} - {prediction['sample_score'][1]}</div></div>"
            )

        body_html = f"""
<div class='card'>
  <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; flex-wrap:wrap; gap:0.5rem;">
    <div>{actual_block}</div>
  </div>
  <div class="metric-grid">
    <div class="metric"><strong>{'HT ' if is_ht_view else ''}Home win</strong>{format_pct(prediction["home_win"])}</div>
    <div class="metric"><strong>{'HT ' if is_ht_view else ''}Draw</strong>{format_pct(prediction["draw"])}</div>
    <div class="metric"><strong>{'HT ' if is_ht_view else ''}Away win</strong>{format_pct(prediction["away_win"])}</div>
    <div class="metric"><strong>{over_label}</strong>{format_pct(prediction["over"])}</div>
    <div class="metric"><strong>{btts_label}</strong>{format_pct(prediction["btts"])}</div>
    <div class="metric"><strong>{xg_label}</strong>{prediction["home_lambda"]:.2f} - {prediction["away_lambda"]:.2f}</div>
  </div>
  <div style="display:flex; justify-content: space-between; align-items: center; margin-top:0.6rem; flex-wrap:wrap; gap:0.75rem;">
    <div>
      <div class="muted">Home form</div>
      <div>{render_form_badges(form_home)}</div>
    </div>
    <div>
      <div class="muted">Away form</div>
      <div>{render_form_badges(form_away)}</div>
    </div>
    <div>
      <div class="muted">{top_scores_label}</div>
      {render_top_scores(prediction["top_scores"])}
    </div>
    {sample_block}
  </div>
</div>
"""
        st.markdown(dedent(body_html), unsafe_allow_html=True)


def sample_matches_for_hot_picks(calendar_df: pd.DataFrame, league: str, matchday: int, n: int = 1) -> pd.DataFrame:
    matches = league_matchday_rows(calendar_df, league, matchday)
    if matches.empty:
        return matches
    if len(matches) <= n:
        return matches
    return matches.sample(n=n, random_state=None)


def cached_hot_pick_matches(calendar_df: pd.DataFrame, league: str, matchday: int, n: int = 1) -> pd.DataFrame:
    cache_key = f"{league}_{matchday}"
    cache = st.session_state.setdefault("hot_pick_cache", {})
    if cache_key not in cache:
        sampled = sample_matches_for_hot_picks(calendar_df, league, matchday, n=n)
        cache[cache_key] = [
            (row["Hometeam"], row["Awayteam"]) for _, row in sampled.iterrows()
        ]
    pairs = cache[cache_key]
    league_matches = league_matchday_rows(calendar_df, league, matchday)
    filtered = league_matches[
        league_matches.apply(lambda r: (r["Hometeam"], r["Awayteam"]) in pairs, axis=1)
    ]
    return filtered


def render_matchday_predictions(calendar_df: pd.DataFrame, league: str, matchday: int) -> None:
    matches = league_matchday_rows(calendar_df, league, matchday)
    if matches.empty:
        st.info("No fixtures for this matchday.")
        return

    strengths_dash, league_avgs_dash = compute_league_strengths(
        calendar_df, league, up_to_matchday=matchday
    )
    strengths_dash_ht, league_avgs_dash_ht = compute_halftime_strengths(
        calendar_df, league, up_to_matchday=matchday
    )

    for _, row in matches.iterrows():
        home_lambda, away_lambda = expected_goals(
            strengths_dash, league_avgs_dash, home_team=row["Hometeam"], away_team=row["Awayteam"]
        )
        home_lambda_ht, away_lambda_ht = expected_goals(
            strengths_dash_ht, league_avgs_dash_ht, home_team=row["Hometeam"], away_team=row["Awayteam"]
        )
        match_key = f"{row['League']}|{row['Matchday']}|{row['Hometeam']}|{row['Awayteam']}"
        predictions = cache_match_predictions(
            match_key,
            home_lambda,
            away_lambda,
            home_lambda_ht,
            away_lambda_ht,
            iterations=SIM_ITERATIONS,
        )
        prediction_ft = predictions["ft"]
        prediction_ht = predictions["ht"]
        form_home = form_guide(calendar_df, row["Hometeam"], before_matchday=int(row["Matchday"]))
        form_away = form_guide(calendar_df, row["Awayteam"], before_matchday=int(row["Matchday"]))
        actual_ft = None
        actual_ht = None
        if not pd.isna(row["FTHG"]) and not pd.isna(row["FTAG"]):
            actual_ft = (row["FTHG"], row["FTAG"])
        if not pd.isna(row.get("HTHG")) and not pd.isna(row.get("HTAG")):
            actual_ht = (row["HTHG"], row["HTAG"])
        toggle_key = f"ht_view_{row['League']}_{row['Matchday']}_{row['Hometeam']}_{row['Awayteam']}".replace(" ", "_")
        render_match_card(
            league=row["League"],
            matchday=int(row["Matchday"]),
            home_team=row["Hometeam"],
            away_team=row["Awayteam"],
            prediction_ft=prediction_ft,
            prediction_ht=prediction_ht,
            form_home=form_home,
            form_away=form_away,
            actual_ft_score=actual_ft,
            actual_ht_score=actual_ht,
            show_sample=False,
            toggle_key=toggle_key,
        )


def main() -> None:
    st.title("The Betting Scientist")
    st.caption(
        "Poisson-powered Monte Carlo simulations for European football. "
        "Explore quick hot picks, custom sandbox matchups, and full league dashboards."
    )

    if not SIM_CONFIG_PATH.exists():
        st.error(f"Config file missing at {SIM_CONFIG_PATH}")
        st.stop()
    if not CALENDAR_WITH_RESULTS_PATH.exists():
        st.error(f"Calendar/results file missing at {CALENDAR_WITH_RESULTS_PATH}")
        st.stop()

    _config = load_config(SIM_CONFIG_PATH)
    calendar_df_raw = load_calendar(CALENDAR_WITH_RESULTS_PATH, CALENDAR_WITH_RESULTS_PATH.stat().st_mtime)
    calendar_df = prepare_calendar(calendar_df_raw)

    leagues = league_options(calendar_df)
    countries = country_options(calendar_df)
    allowed_leagues = leagues

    st.sidebar.markdown("### Mode")
    st.sidebar.success("Premium mode enabled. All leagues and the simulator are unlocked. No login required.")

    init_dashboard_state(calendar_df, allowed_leagues)
    active_league = st.session_state.get("active_league")
    active_country = st.session_state.get("active_country")
    active_matchday = st.session_state.get("active_matchday")

    st.markdown(
        f"<div class='muted'>Calendar rows loaded: {len(calendar_df):,} | Leagues: {len(leagues)}</div>",
        unsafe_allow_html=True,
    )

    # Hot Picks
    st.subheader("Hot Picks")
    if not active_league or active_matchday is None:
        st.info("No league/matchday available for hot picks.")
    else:
        st.markdown(
            f"<div class='muted'>League: {active_league} | Matchday {active_matchday}</div>",
            unsafe_allow_html=True,
        )
        hot_matches = cached_hot_pick_matches(calendar_df, active_league, active_matchday, n=1)
        if hot_matches.empty:
            st.info("No upcoming fixtures found for this league/matchday.")
        else:
            strengths_hp, league_avgs_hp = compute_league_strengths(
                calendar_df, league=active_league, up_to_matchday=int(active_matchday)
            )
            strengths_hp_ht, league_avgs_hp_ht = compute_halftime_strengths(
                calendar_df, league=active_league, up_to_matchday=int(active_matchday)
            )
            for _, row in hot_matches.iterrows():
                home_lambda, away_lambda = expected_goals(
                    strengths_hp, league_avgs_hp, home_team=row["Hometeam"], away_team=row["Awayteam"]
                )
                home_lambda_ht, away_lambda_ht = expected_goals(
                    strengths_hp_ht, league_avgs_hp_ht, home_team=row["Hometeam"], away_team=row["Awayteam"]
                )
                match_key = f"{row['League']}|{row['Matchday']}|{row['Hometeam']}|{row['Awayteam']}"
                predictions = cache_match_predictions(
                    match_key,
                    home_lambda,
                    away_lambda,
                    home_lambda_ht,
                    away_lambda_ht,
                    iterations=SIM_ITERATIONS,
                )
                prediction = predictions["ft"]
                prediction_ht = predictions["ht"]
                form_home = form_guide(calendar_df, row["Hometeam"], before_matchday=int(row["Matchday"]))
                form_away = form_guide(calendar_df, row["Awayteam"], before_matchday=int(row["Matchday"]))
                toggle_key = f"ht_view_hot_{row['League']}_{row['Matchday']}_{row['Hometeam']}_{row['Awayteam']}".replace(
                    " ", "_"
                )
                render_match_card(
                    league=row["League"],
                    matchday=int(row["Matchday"]),
                    home_team=row["Hometeam"],
                    away_team=row["Awayteam"],
                    prediction_ft=prediction,
                    prediction_ht=prediction_ht,
                    form_home=form_home,
                    form_away=form_away,
                    actual_ft_score=None,
                    actual_ht_score=None,
                    show_sample=False,
                    toggle_key=toggle_key,
                )

    st.markdown("---")

    # Algorithmic Simulation (custom sandbox)
    st.subheader("Algorithmic Simulation (Sandbox)")
    col_left, col_right = st.columns([1.2, 2])
    with col_left:
        country = st.selectbox("1) Country", countries)
        leagues_in_country = sorted(calendar_df.loc[calendar_df["Country"] == country, "League"].dropna().unique())
        league_choice = st.selectbox("2) League", leagues_in_country)
        teams = sorted(
            set(calendar_df.loc[calendar_df["League"] == league_choice, "Hometeam"]).union(
                set(calendar_df.loc[calendar_df["League"] == league_choice, "Awayteam"])
            )
        )
        home_team = st.selectbox("3) Home Team", teams, index=0 if teams else None)
        away_team = st.selectbox("4) Away Team", [t for t in teams if t != home_team])
        sim_iterations = st.slider("Simulations", min_value=2000, max_value=20000, step=2000, value=10000)
        run_custom = st.button("Run Simulation", type="primary")
    with col_right:
        if run_custom:
            strengths, league_avgs = compute_league_strengths(calendar_df, league_choice)
            strengths_ht, league_avgs_ht = compute_halftime_strengths(calendar_df, league_choice)
            home_lambda, away_lambda = expected_goals(strengths, league_avgs, home_team, away_team)
            home_lambda_ht, away_lambda_ht = expected_goals(strengths_ht, league_avgs_ht, home_team, away_team)
            prediction_ft = simulate_match(home_lambda, away_lambda, iterations=int(sim_iterations), goal_line=2.5)
            prediction_ht = simulate_match(home_lambda_ht, away_lambda_ht, iterations=int(sim_iterations), goal_line=0.5)
            st.session_state["custom_sim_data"] = {
                "home_team": home_team,
                "away_team": away_team,
                "league": league_choice,
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "home_lambda_ht": home_lambda_ht,
                "away_lambda_ht": away_lambda_ht,
                "ft": prediction_ft,
                "ht": prediction_ht,
            }

        if "custom_sim_data" in st.session_state:
            sim_data = st.session_state["custom_sim_data"]
            toggle_key = "custom_ht_view"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            button_cols = st.columns([4, 1])
            with button_cols[1]:
                custom_button_label = (
                    "Switch to full time view" if st.session_state[toggle_key] else "Switch to half time view"
                )
                if st.button(custom_button_label, key=f"{toggle_key}_btn"):
                    st.session_state[toggle_key] = not st.session_state[toggle_key]

            is_ht_view = bool(st.session_state[toggle_key])
            active_prediction = sim_data["ht"] if is_ht_view else sim_data["ft"]
            active_xg = (
                (sim_data["home_lambda_ht"], sim_data["away_lambda_ht"])
                if is_ht_view
                else (sim_data["home_lambda"], sim_data["away_lambda"])
            )
            over_label = f"{'HT ' if is_ht_view else ''}Over {active_prediction['goal_line']:.1f}"
            btts_label = "HT BTTS" if is_ht_view else "BTTS"
            top_scores_label = "Top 3 HT likely scores" if is_ht_view else "Top 3 likely scores"
            view_title = "Half-time scorecard" if is_ht_view else "Full-time scorecard"

            left, right = st.columns([1, 2])
            with left:
                st.markdown(f"#### One simulated reality ({'HT' if is_ht_view else 'FT'})")
                st.markdown(
                    f"<div class='score-sample'>{active_prediction['sample_score'][0]} - {active_prediction['sample_score'][1]}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='muted'>{'HT ' if is_ht_view else ''}xG: {active_xg[0]:.2f} - {active_xg[1]:.2f}</div>",
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(f"#### {view_title}")
                metrics = {
                    f"{'HT ' if is_ht_view else ''}Home win": format_pct(active_prediction["home_win"]),
                    f"{'HT ' if is_ht_view else ''}Draw": format_pct(active_prediction["draw"]),
                    f"{'HT ' if is_ht_view else ''}Away win": format_pct(active_prediction["away_win"]),
                    over_label: format_pct(active_prediction["over"]),
                    btts_label: format_pct(active_prediction["btts"]),
                }
                st.write(pd.DataFrame(metrics, index=["Probability"]).T)
                st.markdown(render_top_scores(active_prediction["top_scores"]), unsafe_allow_html=True)
        else:
            st.info("Pick teams and hit Run Simulation to see a sample reality and the probabilities.")

    st.markdown("---")

    # League Dashboard
    st.subheader("League Dashboard")

    default_country_idx = countries.index(active_country) if active_country in countries else 0
    dash_country = st.selectbox("Country", countries, index=default_country_idx, key="dashboard_country")
    leagues_in_country = sorted(
        calendar_df.loc[calendar_df["Country"] == dash_country, "League"].dropna().unique()
    )
    if not leagues_in_country:
        st.info("No leagues available for this country.")
        return

    default_league_idx = leagues_in_country.index(active_league) if active_league in leagues_in_country else 0
    dash_league = st.selectbox("League", leagues_in_country, index=default_league_idx, key="dashboard_league")

    if dash_country != st.session_state.get("active_country") or dash_league != st.session_state.get("active_league"):
        st.session_state["active_country"] = dash_country
        st.session_state["active_league"] = dash_league
        st.session_state["active_matchday"] = next_matchday_to_simulate(calendar_df, dash_league) or matchday_range(
            calendar_df, dash_league
        )[0]
        active_country = dash_country
        active_league = dash_league
        active_matchday = st.session_state["active_matchday"]

    md_min, md_max = matchday_range(calendar_df, dash_league)
    current_md = st.session_state.get("active_matchday") or md_max
    current_md = min(max(current_md, md_min), md_max)

    dash_matchday = st.slider(
        "Matchday (scroll to browse past or future)",
        min_value=md_min,
        max_value=md_max,
        value=current_md,
        step=1,
        key="dashboard_matchday_slider",
    )
    st.caption("Please scroll the bar to see past or future matchdays.")

    st.session_state["active_matchday"] = int(dash_matchday)
    active_matchday = int(dash_matchday)
    active_league = dash_league
    active_country = dash_country

    render_matchday_predictions(calendar_df, active_league, active_matchday)


if __name__ == "__main__":
    main()
