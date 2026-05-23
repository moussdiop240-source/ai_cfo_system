from typing import Any, Dict

from backend.agents.math_engine import FinancialCalculationEngine


class VarianceAnalyzer:
    def __init__(self):
        self._engine = FinancialCalculationEngine()

    def analyze(self, actuals: Dict[str, Any], budget: Dict[str, float]) -> Dict[str, Any]:
        numeric_actuals = {k: float(v) for k, v in actuals.items() if isinstance(v, (int, float))}
        return self._engine.calculate_variance_analysis(numeric_actuals, budget)
