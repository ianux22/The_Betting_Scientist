import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SIM_CONFIG_PATH = Path("Input/simulate_config.json")
CALENDAR_WITH_RESULTS_PATH = Path("Output/Calendar_with_Results_25_26.csv")


def calculate_team_statistics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    teams = set(df["HomeTeam"]).union(df["AwayTeam"])
    team_stats = {}
    team_goal_averages = {}

    for team in teams:
        home_matches = df[df["HomeTeam"] == team]
        away_matches = df[df["AwayTeam"] == team]
        all_matches = pd.concat([home_matches, away_matches])

        total_games = len(all_matches)
        total_home_games = max(len(home_matches), 1)
        total_away_games = max(len(away_matches), 1)

        total_home_goals_scored = home_matches["FTHG"].sum()
        total_home_goals_conceded = home_matches["FTAG"].sum()
        total_away_goals_scored = away_matches["FTAG"].sum()
        total_away_goals_conceded = away_matches["FTHG"].sum()

        over_2_5 = len(all_matches[(all_matches["FTHG"] + all_matches["FTAG"]) > 2.5])
        under_2_5 = total_games - over_2_5

        both_score = len(all_matches[(all_matches["FTHG"] > 0) & (all_matches["FTAG"] > 0)])
        no_score = total_games - both_score

        home_win = len(home_matches[home_matches["FTR"] == "H"])
        away_win = len(away_matches[away_matches["FTR"] == "A"])

        home_goal = len(home_matches[home_matches["FTHG"] > 0])
        away_goal = len(away_matches[away_matches["FTAG"] > 0])

        home_opp_goal = len(home_matches[home_matches["FTAG"] > 0])
        away_opp_goal = len(away_matches[away_matches["FTHG"] > 0])

        avg_home_goals_scored = total_home_goals_scored / total_home_games
        avg_home_goals_conceded = total_home_goals_conceded / total_home_games
        avg_away_goals_scored = total_away_goals_scored / total_away_games
        avg_away_goals_conceded = total_away_goals_conceded / total_away_games

        team_stats[team] = {
            "Total Games": total_games,
            "Over 2.5 Goals": over_2_5 / total_games if total_games else 0.0,
            "Under 2.5 Goals": under_2_5 / total_games if total_games else 0.0,
            "Both Teams Score": both_score / total_games if total_games else 0.0,
            "At Least One Team Do Not Score": no_score / total_games if total_games else 0.0,
            "Probability of Winning": (home_win + away_win) / total_games if total_games else 0.0,
            "Probability of winning at Home": home_win / total_home_games,
            "Probability of winning Away": away_win / total_away_games,
            "Probability to score at least a goal at Home": home_goal / total_home_games,
            "Probability to score at least a goal when Away": away_goal / total_away_games,
            "Probability to receive at least a goal at Home": home_opp_goal / total_home_games,
            "Probability to receive at least a goal when Away": away_opp_goal / total_away_games,
        }

        team_goal_averages[team] = {
            "Home Goals Scored": avg_home_goals_scored,
            "Home Goals Conceded": avg_home_goals_conceded,
            "Away Goals Scored": avg_away_goals_scored,
            "Away Goals Conceded": avg_away_goals_conceded,
        }

    return pd.DataFrame(team_stats).T.reset_index().rename(columns={"index": "Team"}), team_goal_averages


def simulate_match_probabilities(
    team_goal_averages: Dict[str, Dict[str, float]],
    home_team: str,
    away_team: str,
    num_simulations: int = 10000,
) -> Dict[str, float]:
    if home_team not in team_goal_averages or away_team not in team_goal_averages:
        raise ValueError(f"Missing data for {home_team} or {away_team}")

    home_avg_scored = team_goal_averages[home_team]["Home Goals Scored"]
    home_avg_conceded = team_goal_averages[home_team]["Home Goals Conceded"]
    away_avg_scored = team_goal_averages[away_team]["Away Goals Scored"]
    away_avg_conceded = team_goal_averages[away_team]["Away Goals Conceded"]

    home_lambda = home_avg_scored * away_avg_conceded
    away_lambda = away_avg_scored * home_avg_conceded

    home_goals = np.random.poisson(home_lambda, num_simulations)
    away_goals = np.random.poisson(away_lambda, num_simulations)

    return {
        "Over 2.5 Goals": np.mean((home_goals + away_goals) > 2.5),
        "Under 2.5 Goals": np.mean((home_goals + away_goals) <= 2.5),
        "Both Teams Score": np.mean((home_goals > 0) & (away_goals > 0)),
        "At Least One Team Do Not Score": np.mean((home_goals == 0) | (away_goals == 0)),
        "Home Win": np.mean(home_goals > away_goals),
        "Away Win": np.mean(away_goals > home_goals),
    }


def get_matchday_matches(calendar: pd.DataFrame, matchday: int) -> List[Tuple[str, str]]:
    calendar = calendar.copy()
    calendar["Matchday"] = pd.to_numeric(calendar["Matchday"], errors="coerce").astype("Int64")
    day_matches = calendar[calendar["Matchday"] == matchday]
    return list(zip(day_matches["Hometeam"], day_matches["Awayteam"]))


def get_matchday_probabilities(
    league: str,
    calendar_df: pd.DataFrame,
    matchday: int,
    print_over_perc: float = 0.9,
    num_simulations: int = 10000,
) -> pd.DataFrame:
    league_calendar = calendar_df[calendar_df["League"] == league].copy()
    if league_calendar.empty:
        raise ValueError(f"No calendar rows found for league '{league}'")

    matches = get_matchday_matches(league_calendar, matchday)
    if not matches:
        raise ValueError(f"No matches found for matchday {matchday}")

    results_df = league_calendar.rename(columns={"Hometeam": "HomeTeam", "Awayteam": "AwayTeam"})
    results_df["FTHG"] = pd.to_numeric(results_df["FTHG"], errors="coerce")
    results_df["FTAG"] = pd.to_numeric(results_df["FTAG"], errors="coerce")
    results_df = results_df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    team_probabilities, team_goal_averages = calculate_team_statistics(results_df)

    rows = []
    for home_team, away_team in matches:
        home_lambda = (
            team_goal_averages[home_team]["Home Goals Scored"] * team_goal_averages[away_team]["Away Goals Conceded"]
        )
        away_lambda = (
            team_goal_averages[away_team]["Away Goals Scored"] * team_goal_averages[home_team]["Home Goals Conceded"]
        )
        probs = simulate_match_probabilities(
            team_goal_averages, home_team=home_team, away_team=away_team, num_simulations=num_simulations
        )
        rows.append(
            {
                "Hometeam": home_team,
                "Awayteam": away_team,
                "xG_Home": home_lambda,
                "xG_Away": away_lambda,
                "Matchday": matchday,
                **probs,
            }
        )

    df_match_stats = pd.DataFrame(rows)
    if {"Home Win", "Away Win"}.issubset(df_match_stats.columns):
        df_match_stats["Draw"] = 1.0 - df_match_stats["Home Win"] - df_match_stats["Away Win"]
    df_match_stats["League"] = league

    prob_columns = [
        col for col in df_match_stats.columns if col not in ["Hometeam", "Awayteam", "League", "Matchday", "xG_Home", "xG_Away"]
    ]
    for _, row in df_match_stats.iterrows():
        for col in prob_columns:
            value = float(row[col])
            if value >= print_over_perc:
                print(
                    f"League: {row['League']} | Match: {row['Hometeam']} - {row['Awayteam']} | Outcome: {col} -> {value*100:.2f}%"
                )

    return df_match_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matchday simulations using local calendar/results data.")
    parser.add_argument(
        "--league",
        default=None,
        help="League name as defined in Input/simulate_config.json. If omitted, runs all leagues in 'simulate'.",
    )
    parser.add_argument("--matchday", type=int, default=None, help="Override matchday number to simulate.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold for printing outcomes.")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of Monte Carlo simulations per match.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SIM_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    matchdays: Dict[str, int] = config.get("matchdays", {})
    target_leagues = [args.league] if args.league else config.get("simulate") or list(matchdays.keys())

    if not CALENDAR_WITH_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Calendar/results file not found at {CALENDAR_WITH_RESULTS_PATH}")
    calendar_df = pd.read_csv(CALENDAR_WITH_RESULTS_PATH)

    final_df = pd.DataFrame()
    for league in target_leagues:
        matchday = args.matchday if args.matchday is not None else matchdays.get(league)
        if matchday is None:
            print(f"Skipping {league}: missing matchday (set in simulate_config.json or via --matchday).")
            continue

        print(f"Running {league}: matchday {matchday} using local calendar/results data")
        try:
            df = get_matchday_probabilities(
                league=league,
                calendar_df=calendar_df,
                matchday=matchday,
                print_over_perc=args.threshold,
                num_simulations=args.simulations,
            )
            final_df = pd.concat([final_df, df], ignore_index=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error running simulation for {league}: {exc}")

    if not final_df.empty:
        print("\nSimulation summary:")
        print(final_df)
        final_df.to_csv("Output/matchday_simulation_results.csv", index=False)
    else:
        print("No simulations ran. Check configuration and inputs.")


if __name__ == "__main__":
    main()
