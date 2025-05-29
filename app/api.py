from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.model_utils import Model_Utils
from app.kelly_optimizer import KellyOptimizer
import logging
from typing import Dict, Any
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🔁 This log is printed at the top level")

app = FastAPI(
    title="MLB Prediction API",
    description=(
        "API for predicting MLB game outcomes "
        "and betting probabilities"
    ),
    version="1.0.0"
)

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:3000",
        "https://*.render.com",  # Allow Render domains
        os.getenv("FRONTEND_URL", ""),  # Allow custom frontend URL from environment
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Mount static files from the current directory
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


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
        frontend_path = os.path.join(BASE_DIR, "frontend.html")
        with open(frontend_path) as f:
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
    
@app.post("/kelly-optimize")
async def kelly_optimize(
    payload: Dict[str, Any] = Body(...)
):
    """
    Optimize bet allocations using the Kelly Criterion.
    Expected JSON format:
    {
        "probabilities": [0.6, 0.55, 0.52],
        "odds": [100, -110, 150],     // American odds
        "kelly_fraction": 0.5
    }
    """
    try:
        probs = payload.get("probabilities")
        odds = payload.get("odds")
        kelly_fraction = payload.get("kelly_fraction", 1.0)

        if not probs or not odds or len(probs) != len(odds):
            raise HTTPException(status_code=400, detail="Invalid input: probs and odds must be equal-length lists")

        optimizer = KellyOptimizer(probs, odds, kelly_fraction)
        allocations = optimizer.kelly_portfolio()

        return {
            "allocations": [float(round(a, 6)) for a in allocations],
            "total_fraction": float(round(sum(allocations), 6))
        }

    except Exception as e:
        logger.error(f"Kelly optimizer error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))