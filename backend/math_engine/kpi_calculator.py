from typing import Any, Dict

from backend.agents.math_engine import FinancialCalculationEngine


class KPICalculator:
    def __init__(self):
        self._engine = FinancialCalculationEngine()

    def calculate_all(self, data: Dict[str, Any]) -> Dict[str, float]:
        return self._engine.calculate_kpis(data)
