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
FREE_LEAGUES = {"Premier League", "Serie A", "La Liga Primera"}
DEFAULT_USERS = {
    "test_free@example.com": {"password": "TestFree123!", "role": "free"},
    "test_premium@example.com": {"password": "TestPremium123!", "role": "premium"},
}


@st.cache_data(show_spinner=False)
def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_calendar(calendar_path: Path) -> pd.DataFrame:
    return pd.read_csv(calendar_path)


def prepare_calendar(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Matchday"] = pd.to_numeric(prepared["Matchday"], errors="coerce").astype("Int64")
    for col in ["FTHG", "FTAG"]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    prepared["FTR"] = prepared["FTR"].replace("", pd.NA)
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


def seed_users() -> None:
    if "users" not in st.session_state:
        st.session_state["users"] = DEFAULT_USERS.copy()


def authenticate(email: str, password: str) -> Optional[Dict[str, str]]:
    users = st.session_state.get("users", {})
    user = users.get(email)
    if user and user.get("password") == password:
        return {"email": email, "role": user.get("role", "free")}
    return None


def register_user(email: str, password: str, role: str = "free") -> bool:
    users = st.session_state.get("users", {})
    if email in users:
        return False
    users[email] = {"password": password, "role": role}
    st.session_state["users"] = users
    return True


def role_allows_league(role: str, league: str) -> bool:
    if role in ("premium", "admin"):
        return True
    if role == "free":
        return league in FREE_LEAGUES
    return False


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
def compute_league_strengths(
    calendar_df: pd.DataFrame, league: str, up_to_matchday: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    league_df = calendar_df[calendar_df["League"] == league].copy()
    if up_to_matchday is not None:
        league_df = league_df[league_df["Matchday"].notna() & (league_df["Matchday"] < up_to_matchday)]

    completed = league_df.dropna(subset=["FTHG", "FTAG"])
    fallback_home_avg = 1.4
    fallback_away_avg = 1.1

    league_home_avg = float(completed["FTHG"].mean()) if not completed.empty else fallback_home_avg
    league_away_avg = float(completed["FTAG"].mean()) if not completed.empty else fallback_away_avg
    league_home_avg = max(league_home_avg, 0.2)
    league_away_avg = max(league_away_avg, 0.2)

    teams = set(league_df["Hometeam"]).union(set(league_df["Awayteam"]))
    strengths: Dict[str, Dict[str, float]] = {}

    for team in teams:
        home_matches = completed[completed["Hometeam"] == team]
        away_matches = completed[completed["Awayteam"] == team]

        home_scored = float(home_matches["FTHG"].mean()) if not home_matches.empty else league_home_avg
        home_conceded = float(home_matches["FTAG"].mean()) if not home_matches.empty else league_away_avg
        away_scored = float(away_matches["FTAG"].mean()) if not away_matches.empty else league_away_avg
        away_conceded = float(away_matches["FTHG"].mean()) if not away_matches.empty else league_home_avg

        strengths[team] = {
            "home_attack": home_scored / league_home_avg if league_home_avg else 1.0,
            "home_defense": home_conceded / league_away_avg if league_away_avg else 1.0,
            "away_attack": away_scored / league_away_avg if league_away_avg else 1.0,
            "away_defense": away_conceded / league_home_avg if league_home_avg else 1.0,
        }

    league_avgs = {"home_goal_avg": league_home_avg, "away_goal_avg": league_away_avg}
    return strengths, league_avgs


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


def simulate_match(home_lambda: float, away_lambda: float, iterations: int = 2000) -> Dict[str, object]:
    home_goals = np.random.poisson(home_lambda, iterations)
    away_goals = np.random.poisson(away_lambda, iterations)

    results = {
        "home_win": float(np.mean(home_goals > away_goals)),
        "away_win": float(np.mean(home_goals < away_goals)),
        "draw": float(np.mean(home_goals == away_goals)),
        "over_2_5": float(np.mean((home_goals + away_goals) > 2.5)),
        "btts": float(np.mean((home_goals > 0) & (away_goals > 0))),
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
    prediction: Dict[str, object],
    form_home: List[str],
    form_away: List[str],
    actual_score: Optional[Tuple[float, float]] = None,
    show_sample: bool = False,
) -> None:
    actual_block = ""
    if actual_score:
        actual_block = f"<div class='muted'>Actual FT</div><div class='headline'>{int(actual_score[0])}-{int(actual_score[1])}</div>"

    sample_block = ""
    if show_sample:
        sample_block = (
            f"<div><div class='muted'>Sample reality</div>"
            f"<div class='score-sample'>{prediction['sample_score'][0]} - {prediction['sample_score'][1]}</div></div>"
        )

    html = f"""
<div class='card'>
  <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
    <div>
      <span class="pill pill-league">{league}</span>
      <span class="muted" style="margin-left:0.5rem;">Matchday {matchday}</span>
      <div class="headline">{home_team} vs {away_team}</div>
    </div>
    <div>{actual_block}</div>
  </div>
  <div class="metric-grid">
    <div class="metric"><strong>Home win</strong>{format_pct(prediction["home_win"])}</div>
    <div class="metric"><strong>Draw</strong>{format_pct(prediction["draw"])}</div>
    <div class="metric"><strong>Away win</strong>{format_pct(prediction["away_win"])}</div>
    <div class="metric"><strong>Over 2.5</strong>{format_pct(prediction["over_2_5"])}</div>
    <div class="metric"><strong>BTTS</strong>{format_pct(prediction["btts"])}</div>
    <div class="metric"><strong>xG</strong>{prediction["home_lambda"]:.2f} - {prediction["away_lambda"]:.2f}</div>
  </div>
  <div style="display:flex; justify-content: space-between; align-items: center; margin-top:0.6rem;">
    <div>
      <div class="muted">Home form</div>
      <div>{render_form_badges(form_home)}</div>
    </div>
    <div>
      <div class="muted">Away form</div>
      <div>{render_form_badges(form_away)}</div>
    </div>
    <div>
      <div class="muted">Top 3 likely scores</div>
      {render_top_scores(prediction["top_scores"])}
    </div>
    {sample_block}
  </div>
</div>
"""
    st.markdown(dedent(html), unsafe_allow_html=True)


def sample_matches_for_hot_picks(calendar_df: pd.DataFrame, league: str, matchday: int, n: int = 3) -> pd.DataFrame:
    matches = league_matchday_rows(calendar_df, league, matchday)
    if matches.empty:
        return matches
    if len(matches) <= n:
        return matches
    return matches.sample(n=n, random_state=None)


def render_matchday_predictions(calendar_df: pd.DataFrame, league: str, matchday: int) -> None:
    matches = league_matchday_rows(calendar_df, league, matchday)
    if matches.empty:
        st.info("No fixtures for this matchday.")
        return

    strengths_dash, league_avgs_dash = compute_league_strengths(
        calendar_df, league, up_to_matchday=matchday
    )

    for _, row in matches.iterrows():
        home_lambda, away_lambda = expected_goals(
            strengths_dash, league_avgs_dash, home_team=row["Hometeam"], away_team=row["Awayteam"]
        )
        prediction = simulate_match(home_lambda, away_lambda, iterations=2000)
        form_home = form_guide(calendar_df, row["Hometeam"], before_matchday=int(row["Matchday"]))
        form_away = form_guide(calendar_df, row["Awayteam"], before_matchday=int(row["Matchday"]))
        actual = None
        if not pd.isna(row["FTHG"]) and not pd.isna(row["FTAG"]):
            actual = (row["FTHG"], row["FTAG"])
        render_match_card(
            league=row["League"],
            matchday=int(row["Matchday"]),
            home_team=row["Hometeam"],
            away_team=row["Awayteam"],
            prediction=prediction,
            form_home=form_home,
            form_away=form_away,
            actual_score=actual,
            show_sample=False,
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

    config = load_config(SIM_CONFIG_PATH)
    calendar_df_raw = load_calendar(CALENDAR_WITH_RESULTS_PATH)
    calendar_df = prepare_calendar(calendar_df_raw)

    leagues = league_options(calendar_df)
    countries = country_options(calendar_df)
    seed_users()

    st.sidebar.markdown("### Account")
    if "user" not in st.session_state:
        with st.sidebar.form("login_form", clear_on_submit=False):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Log in")
        with st.sidebar.form("register_form", clear_on_submit=False):
            reg_email = st.text_input("Register email")
            reg_password = st.text_input("Register password", type="password")
            reg_role = st.selectbox("Plan", ["free", "premium"])
            register_clicked = st.form_submit_button("Register")
        if login_clicked:
            user = authenticate(email, password)
            if user:
                st.session_state["user"] = user
                st.success(f"Logged in as {user['email']} ({user['role']})")
            else:
                st.error("Invalid credentials")
        if register_clicked:
            if register_user(reg_email, reg_password, role=reg_role):
                st.success(f"Registered {reg_email} as {reg_role}. Please log in.")
            else:
                st.error("User already exists.")
    else:
        user = st.session_state["user"]
        st.sidebar.success(f"Logged in as {user['email']} ({user['role']})")
        if st.sidebar.button("Log out"):
            st.session_state.pop("user")
            st.experimental_rerun()

    if "user" not in st.session_state:
        st.info("Please log in or register to access dashboards and simulators.")
        st.subheader("Plans")
        st.markdown("- Free: access Premier League, Serie A, La Liga Primera dashboards.\n- Premium: access all leagues and simulators.")
        return

    role = st.session_state["user"]["role"]
    allowed_leagues = leagues if role in ("premium", "admin") else [l for l in leagues if role_allows_league(role, l)]

    init_dashboard_state(calendar_df, allowed_leagues)
    active_league = st.session_state.get("active_league")
    active_country = st.session_state.get("active_country")
    active_matchday = st.session_state.get("active_matchday")

    st.markdown(
        f"<div class='muted'>Calendar rows loaded: {len(calendar_df):,} | Leagues: {len(leagues)}</div>",
        unsafe_allow_html=True,
    )

    # Hot Picks
    st.subheader("🔥 Hot Picks")
    if not active_league or active_matchday is None:
        st.info("No league/matchday available for hot picks.")
    elif not role_allows_league(role, active_league):
        st.warning(
            "This league is locked for your plan. Upgrade to Premium to unlock this league and the simulator."
        )
    else:
        st.markdown(
            f"<div class='muted'>League: {active_league} | Matchday {active_matchday}</div>",
            unsafe_allow_html=True,
        )
        hot_matches = sample_matches_for_hot_picks(calendar_df, active_league, active_matchday, n=3)
        if hot_matches.empty:
            st.info("No upcoming fixtures found for this league/matchday.")
        else:
            strengths_hp, league_avgs_hp = compute_league_strengths(
                calendar_df, league=active_league, up_to_matchday=int(active_matchday)
            )
            for _, row in hot_matches.iterrows():
                home_lambda, away_lambda = expected_goals(
                    strengths_hp, league_avgs_hp, home_team=row["Hometeam"], away_team=row["Awayteam"]
                )
                prediction = simulate_match(home_lambda, away_lambda, iterations=2000)
                form_home = form_guide(calendar_df, row["Hometeam"], before_matchday=int(row["Matchday"]))
                form_away = form_guide(calendar_df, row["Awayteam"], before_matchday=int(row["Matchday"]))
                render_match_card(
                    league=row["League"],
                    matchday=int(row["Matchday"]),
                    home_team=row["Hometeam"],
                    away_team=row["Awayteam"],
                    prediction=prediction,
                    form_home=form_home,
                    form_away=form_away,
                    actual_score=None,
                    show_sample=False,
                )

    st.markdown("---")

    # Algorithmic Simulation (custom sandbox)
    st.subheader("🧪 Algorithmic Simulation (Sandbox)")
    if role not in ("premium", "admin"):
        st.info("Upgrade to Premium to unlock the simulator.")
    else:
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
                home_lambda, away_lambda = expected_goals(strengths, league_avgs, home_team, away_team)
                prediction = simulate_match(home_lambda, away_lambda, iterations=int(sim_iterations))

                left, right = st.columns([1, 2])
                with left:
                    st.markdown("#### One simulated reality")
                    st.markdown(
                        f"<div class='score-sample'>{prediction['sample_score'][0]} - {prediction['sample_score'][1]}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"<div class='muted'>xG: {home_lambda:.2f} - {away_lambda:.2f}</div>", unsafe_allow_html=True)
                with right:
                    st.markdown("#### Scorecard")
                    metrics = {
                        "Home win": format_pct(prediction["home_win"]),
                        "Draw": format_pct(prediction["draw"]),
                        "Away win": format_pct(prediction["away_win"]),
                        "Over 2.5": format_pct(prediction["over_2_5"]),
                        "BTTS": format_pct(prediction["btts"]),
                    }
                    st.write(pd.DataFrame(metrics, index=["Probability"]).T)
                    st.markdown(render_top_scores(prediction["top_scores"]), unsafe_allow_html=True)
            else:
                st.info("Pick teams and hit Run Simulation to see a sample reality and the probabilities.")

    st.markdown("---")

    # League Dashboard
    st.subheader("📊 League Dashboard")

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

    if not role_allows_league(role, active_league):
        st.warning("Upgrade to Premium to unlock this league and the simulator.")
    else:
        render_matchday_predictions(calendar_df, active_league, active_matchday)


if __name__ == "__main__":
    main()
