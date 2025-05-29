import cvxpy as cp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class KellyOptimizer:
    def __init__(self, probabilities, american_odds, kelly_fraction=1.0):
        self.probabilities = probabilities
        self.odds = self._convert_to_decimal_odds(american_odds)
        self.kelly_fraction = kelly_fraction
        
        logger.info(f"Initialized KellyOptimizer with:")
        logger.info(f"Probabilities: {probabilities}")
        logger.info(f"American odds: {american_odds}")
        logger.info(f"Decimal odds: {self.odds}")
        logger.info(f"Kelly fraction: {kelly_fraction}")

    def _convert_to_decimal_odds(self, odds):
        # Convert list of American odds to decimal odds
        decimal_odds = [
            (o / 100 + 1) if o > 0 else (-100 / o + 1)
            for o in odds
        ]
        logger.info(f"Converted American odds {odds} to decimal odds {decimal_odds}")
        return decimal_odds

    def kelly_portfolio(self):
        """
        Solve Kelly optimization and return optimal bankroll fractions.
        """
        n = len(self.probabilities)
        f = cp.Variable(n)

        expected_log_return = cp.sum([
            p * cp.log1p(f[i] * (o - 1)) + (1 - p) * cp.log1p(-f[i])
            for i, (p, o) in enumerate(zip(self.probabilities, self.odds))
        ])

        constraints = [cp.sum(f) <= 1, f >= 0]
        problem = cp.Problem(cp.Maximize(expected_log_return), constraints)
        problem.solve()

        allocations = f.value * self.kelly_fraction
        logger.info(f"Kelly optimization result: {allocations}")
        return allocations
