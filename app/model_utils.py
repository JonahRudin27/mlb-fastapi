from scipy.stats import skewnorm
import joblib
import pandas as pd
from datetime import datetime
import os


class Model_Utils:
    def __init__(self):
        [self.cleaned_batting, 
         self.cleaned_pitching, 
         self.norm_params, 
         self.model] = self.load_resources()
        
    def predict(self, away_team, home_team, away_pitcher, home_pitcher):
        x = self.pull_data(away_team, home_team, away_pitcher, home_pitcher)
        y_pred, y_std = self.model.predict(x, return_std=True)
        return y_pred, y_std
    
    def choose_bet(self, run_diff, std, runLine, away_odds, home_odds):
        # Step 1: Create normal distribution of (away - home)
        skew = run_diff/abs(run_diff) * 0.25
        run_dif_dist = skewnorm(a=skew, loc=run_diff, scale=std)
            
        # Step 2: Calculate probability that away team beats the spread
        p_away = 1 - run_dif_dist.cdf(-1 * runLine)
        p_home = 1 - p_away

        # Step 3: Convert betting odds to implied probabilities
        away_prob = (
            -away_odds / (-away_odds + 100) if away_odds < 0 
            else 100 / (away_odds + 100)
        )
        home_prob = (
            -home_odds / (-home_odds + 100) if home_odds < 0 
            else 100 / (home_odds + 100)
        )

        # Step 4: Compute profitability margins
        away_profitability = p_away - away_prob
        home_profitability = p_home - home_prob

        return {
            "p_away": p_away.item(),
            "p_home": p_home.item(),
            "away_profitability": away_profitability.item(),
            "home_profitability": home_profitability.item()
        }

    def pull_data(self, away_team, home_team, away_pitcher, home_pitcher):
        # Step 1: Load normalization parameters
        norm_params = self.norm_params

        # Step 2: Build feature row
        home_data = (
            self.cleaned_batting.query("Tm == @home_team")
            .reset_index(drop=True)
            .add_prefix('home_')
        )
        away_data = (
            self.cleaned_batting.query("Tm == @away_team")
            .reset_index(drop=True)
            .add_prefix('away_')
        )
        home_pitcher_data = (
            self.cleaned_pitching.query("Player == @home_pitcher")
            .reset_index(drop=True)
            .add_prefix('home_')
        )
        away_pitcher_data = (
            self.cleaned_pitching.query("Player == @away_pitcher")
            .reset_index(drop=True)
            .add_prefix('away_')
        )

        data_sets = [
            (home_data, 'home_data'),
            (away_data, 'away_data'),
            (home_pitcher_data, 'home_pitcher_data'),
            (away_pitcher_data, 'away_pitcher_data')
        ]

        for df, label in data_sets:
            if len(df) != 1:
                raise ValueError(
                    f"{label} returned {len(df)} rows, expected exactly 1."
                )

        combined_df = pd.concat(
            [home_data, away_data, home_pitcher_data, away_pitcher_data],
            axis=1
        )
        combined_df['Number_of_Games'] = 0

        # Step 3: Select features
        features = [
            'home_R/G', 'home_PA', 'home_AB', 'home_R', 'home_H',
            'home_2B', 'home_3B', 'home_HR', 'home_RBI', 'home_SB',
            'home_CS', 'home_BB', 'home_SO', 'home_BA', 'home_OBP',
            'home_SLG', 'home_OPS', 'home_OPS+', 'away_R/G', 'away_PA',
            'away_AB', 'away_R', 'away_H', 'away_2B', 'away_3B',
            'away_HR', 'away_RBI', 'away_SB', 'away_CS', 'away_BB',
            'away_SO', 'away_BA', 'away_OBP', 'away_SLG', 'away_OPS',
            'away_OPS+', 'Number_of_Games', 'home_SO/BB', 'home_ERA+',
            'home_WAR', 'home_FIP', 'home_WHIP', 'away_SO/BB',
            'away_ERA+', 'away_WAR', 'away_FIP', 'away_WHIP'
        ]

        x_df = combined_df[features].copy()

        # Step 4: Normalize using saved means and stds
        for col in features:
            mean = norm_params.loc[col, 'mean']
            std = norm_params.loc[col, 'std']
            x_df[col] = (x_df[col] - mean) / std

        # Step 5: Predict
        return x_df.to_numpy()

    def load_resources(self):
        model = joblib.load(os.path.join(os.path.dirname(__file__), "baseball_model.pkl"))

        batting_url = (
            f"https://www.baseball-reference.com/leagues/MLB/2025.shtml"
        )
        tables = pd.read_html(batting_url)
        raw_batting = tables[0].iloc[:30]  # Only real teams
        cleaned_batting = self.clean_batting_table(raw_batting)

        pitching_url = (
            f"https://www.baseball-reference.com/leagues/MLB/2025-standard-pitching.shtml"
        )
        pitcher_stats = pd.read_html(pitching_url)
        raw_pitching = pitcher_stats[1]

        pitcher_columns = {
            "Player",
            "WAR",
            "ERA+",
            "FIP",
            "WHIP",
            "SO/BB"
        }

        cleaned_pitching = raw_pitching[list(pitcher_columns)]
        cleaned_pitching.loc[:, 'Player'] = (
            cleaned_pitching['Player'].str.replace("*", "", regex=False)
        )
        norm_params = pd.read_csv(os.path.join(os.path.dirname(__file__), "normalization_params.csv"))

        return cleaned_batting, cleaned_pitching, norm_params, model

    def clean_batting_table(self, df):
        cols = [0, 3] + list(range(5, 22))  # Tm, R/G, PA–OPS+
        selected = df.iloc[:, cols].copy()
        G = pd.to_numeric(df['G'], errors='coerce')

        # Convert all columns except "Tm" to numeric
        for col in selected.columns[1:]:
            selected[col] = pd.to_numeric(selected[col], errors='coerce')

        # Normalize season-long counting stats
        season_long = list(selected.columns[2:14])
        selected[season_long] = selected[season_long].div(G, axis=0)

        # Rename columns
        selected.columns = [
            "Tm", "R/G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI",
            "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+"
        ]

        team_name_to_acronym = {
            "Arizona Diamondbacks": "ARI",
            "Athletics": "OAK",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Indians": "CLE",     # Note: Became "Guardians" in 2022
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KCR",
            "Los Angeles Angels of Anaheim": "LAA",  # older name
            "Los Angeles Angels": "LAA",
            "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN",
            "New York Mets": "NYM",
            "New York Yankees": "NYY",
            "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SDP",
            "San Francisco Giants": "SFG",
            "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TBR",
            "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSN"
        }

        selected["Tm"] = selected["Tm"].map(team_name_to_acronym)

        return selected