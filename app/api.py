import os
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.model_utils import Model_Utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLB Prediction API",
    description="API for predicting MLB game outcomes and betting probabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (if you ever want to serve CSS/JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model utilities on startup
@app.on_event("startup")
async def load_resources():
    try:
        app.state.model_utils = Model_Utils()
        logger.info("✅ Model utilities loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading model utilities: {str(e)}")
        raise

# Serve frontend HTML from file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="frontend.html not found")
    with open(html_path, "r") as f:
        return f.read()

# Health check
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model_utils")
    }

# Prediction endpoint
@app.get("/predict")
async def predict(
    away_team: str,
    home_team: str,
    away_pitcher: str,
    home_pitcher: str,
    runLine: float,
    away_odds: float,
    home_odds: float
) -> Dict[str, Any]:
    try:
        if not all([away_team, home_team, away_pitcher, home_pitcher]):
            raise HTTPException(
                status_code=400,
                detail="All team and pitcher names are required"
            )
        if away_team == home_team:
            raise HTTPException(
                status_code=400,
                detail="Away team and home team must be different"
            )

        model_utils = app.state.model_utils
        y_pred, y_std = model_utils.predict(
            away_team, home_team, away_pitcher, home_pitcher
        )

        recommendations = model_utils.choose_bet(
            float(y_pred), float(y_std), runLine, away_odds, home_odds
        )

        return {
            "y_pred": float(y_pred),
            "y_std": float(y_std),
            "bet_recommendation": recommendations
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
