from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from model_utils import Model_Utils
import logging
from typing import Dict, Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLB Prediction API",
    description=(
        "API for predicting MLB game outcomes "
        "and betting probabilities"
    ),
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


# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")


@app.on_event("startup")
async def load_resources():
    """Load required resources on startup."""
    try:
        app.state.model_utils = Model_Utils()
        logger.info("Model utilities loaded successfully")
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        raise


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML file."""
    try:
        with open("frontend.html") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend file not found")


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if the API is healthy and resources are loaded."""
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model_utils")
    }


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
    """Make a prediction for a baseball game."""
    try:
        # Validate inputs
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

        # Get prediction
        model_utils = app.state.model_utils
        y_pred, y_std = model_utils.predict(
            away_team, home_team, away_pitcher, home_pitcher
        )
        
        # Get betting recommendations
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
        raise HTTPException(status_code=500, detail=str(e))