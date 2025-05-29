from scipy.stats import skewnorm
import joblib
import pandas as pd
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

class Model_Utils:
    def __init__(self):
        logger.info("🧠 Initializing Model_Utils...")
        try:
            [self.cleaned_batting, 
             self.cleaned_pitching, 
             self.norm_params, 
             self.model] = self.load_resources()
            logger.info("✅ Resources loaded successfully in __init__")
        except Exception as e:
            logger.error("❌ Failed to load resources in __init__", exc_info=True)
            raise

    def predict(self, away_team, home_team, away_pitcher, home_pitcher):
        logger.info(f"🔮 Running predict() for {away_team} vs {home_team}")
        x = self.pull_data(away_team, home_team, away_pitcher, home_pitcher)
        y_pred, y_std = self.model.predict(x, return_std=True)
        logger.info(f"🎯 Prediction complete: y_pred={y_pred}, y_std={y_std}")
        return y_pred, y_std
    
    def choose_bet(self, run_diff, std, runLine, away_odds, home_odds):
        logger.info("📈 Choosing bet based on model output...")
        skew = run_diff / abs(run_diff) * -0.25
        run_dif_dist = skewnorm(a=skew, loc=run_diff, scale=std)

        p_away = 1 - run_dif_dist.cdf(-1 * runLine)
        p_home = 1 - p_away

        away_prob = (
            -away_odds / (-away_odds + 100) if away_odds < 0 
            else 100 / (away_odds + 100)
        )
        home_prob = (
            -home_odds / (-home_odds + 100) if home_odds < 0 
            else 100 / (home_odds + 100)
        )

        away_profitability = p_away - away_prob
        home_profitability = p_home - home_prob

        logger.info(f"💸 Profit margins - Away: {away_profitability:.3f}, Home: {home_profitability:.3f}")

        return {
            "p_away": p_away.item(),
            "p_home": p_home.item(),
            "away_profitability": away_profitability.item(),
            "home_profitability": home_profitability.item()
        }

    def pull_data(self, away_team, home_team, away_pitcher, home_pitcher):
        logger.info("📊 Pulling and preparing input data...")

        norm_params = self.norm_params

        home_data = self.cleaned_batting.query("Tm == @home_team").reset_index(drop=True).add_prefix('home_')
        away_data = self.cleaned_batting.query("Tm == @away_team").reset_index(drop=True).add_prefix('away_')
        home_pitcher_data = self.cleaned_pitching.query("Player == @home_pitcher").reset_index(drop=True).add_prefix('home_')
        away_pitcher_data = self.cleaned_pitching.query("Player == @away_pitcher").reset_index(drop=True).add_prefix('away_')

        data_sets = [
            (home_data, 'home_data'),
            (away_data, 'away_data'),
            (home_pitcher_data, 'home_pitcher_data'),
            (away_pitcher_data, 'away_pitcher_data')
        ]

        for df, label in data_sets:
            if len(df) != 1:
                logger.error(f"❌ {label} returned {len(df)} rows, expected 1")
                raise ValueError(f"{label} returned {len(df)} rows, expected exactly 1.")

        logger.info("✅ All datasets validated")
        combined_df = pd.concat([home_data, away_data, home_pitcher_data, away_pitcher_data], axis=1)
        combined_df['Number_of_Games'] = 0

        # Select & normalize features
        features = [
            'home_R/G', 'home_PA', 'home_AB', 'home_R', 'home_H', 'home_2B', 'home_3B',
            'home_HR', 'home_RBI', 'home_SB', 'home_CS', 'home_BB', 'home_SO',
            'home_BA', 'home_OBP', 'home_SLG', 'home_OPS', 'home_OPS+',
            'away_R/G', 'away_PA', 'away_AB', 'away_R', 'away_H', 'away_2B',
            'away_3B', 'away_HR', 'away_RBI', 'away_SB', 'away_CS', 'away_BB',
            'away_SO', 'away_BA', 'away_OBP', 'away_SLG', 'away_OPS', 'away_OPS+',
            'Number_of_Games',
            'home_SO/BB', 'home_ERA+', 'home_WAR', 'home_FIP', 'home_WHIP',
            'away_SO/BB', 'away_ERA+', 'away_WAR', 'away_FIP', 'away_WHIP'
        ]
        x_df = combined_df[features].copy()

        logger.info("About to normalize...")

        for col in features:
            if col not in x_df.columns or col not in self.norm_params.index:
                logger.warning(f"Skipping missing column: {col}")
                continue

            mean = self.norm_params.loc[col, 'mean']
            std = self.norm_params.loc[col, 'std']

            if std == 0 or pd.isna(std):
                logger.warning(f"Invalid std for {col} (value: {std}); skipping.")
                continue

            x_df[col] = (x_df[col] - mean) / std

        logger.info("📐 Data normalized and ready for prediction")
        return x_df.to_numpy()

    def load_resources(self):
        logger.info("🔧 Loading resources: model, stats, and normalization parameters...")
        try:
            model = joblib.load(os.path.join(os.path.dirname(__file__), "baseball_model.pkl"))
            logger.info("✅ Model loaded")

            batting_url = "https://www.baseball-reference.com/leagues/MLB/2025.shtml"
            logger.info(f"🌐 Fetching batting data from {batting_url}")
            tables = pd.read_html(batting_url)
            raw_batting = tables[0].iloc[:30]
            cleaned_batting = self.clean_batting_table(raw_batting)
            logger.info("✅ Batting data cleaned")

            pitching_url = "https://www.baseball-reference.com/leagues/MLB/2025-standard-pitching.shtml"
            logger.info(f"🌐 Fetching pitching data from {pitching_url}")
            pitcher_stats = pd.read_html(pitching_url)
            raw_pitching = pitcher_stats[1]

            pitcher_columns = {"Player", "WAR", "ERA+", "FIP", "WHIP", "SO/BB"}
            cleaned_pitching = raw_pitching[list(pitcher_columns)]
            cleaned_pitching.loc[:, "Player"] = cleaned_pitching["Player"].str.replace("*", "", regex=False)
            logger.info("✅ Pitching data cleaned")

            norm_path = os.path.join(os.path.dirname(__file__), "normalization_params.csv")
            norm_params = pd.read_csv(norm_path, index_col=0)            
            logger.info("✅ Normalization parameters loaded")
            return cleaned_batting, cleaned_pitching, norm_params, model

        except Exception as e:
            logger.error("❌ Error during resource loading", exc_info=True)
            raise

    def clean_batting_table(self, df):
        logger.info("🧽 Cleaning batting table...")

        cols = [0, 3] + list(range(5, 22))  # Tm, R/G, PA–OPS+
        selected = df.iloc[:, cols].copy()
        G = pd.to_numeric(df['G'], errors='coerce')
        logger.info("✅ Selected and parsed game counts")

        for col in selected.columns[1:]:
            selected[col] = pd.to_numeric(selected[col], errors='coerce')
        logger.info("✅ Converted stats to numeric")

        season_long = list(selected.columns[2:14])
        selected[season_long] = selected[season_long].div(G, axis=0)
        logger.info("✅ Normalized counting stats by games played")

        selected.columns = [
            "Tm", "R/G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI",
            "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+"
        ]

        logger.info("✅ Renamed columns for consistency")

        team_name_to_acronym = {
            "Arizona Diamondbacks": "ARI",
            "Athletics": "OAK",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Indians": "CLE",
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KCR",
            "Los Angeles Angels of Anaheim": "LAA",
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
        logger.info("✅ Converted team names to acronyms")

        return selected
