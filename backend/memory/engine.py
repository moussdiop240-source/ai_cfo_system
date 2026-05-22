"""
MemoryEngine — persists institutional knowledge across pipeline runs.

Design principles:
- SHA-256 dedup: identical data for the same period is never double-saved.
- Idempotent: calling save_analysis() twice with identical data returns the same snapshot id.
- Transactional: snapshot + KPI rows + insights saved in one DB transaction.
- Lightweight: uses the existing SQLite DB; no new infrastructure required.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import CompanyMemory, InsightMemory, KPITimeSeries, MemoryBase, PeriodSnapshot

# KPIs we track in the time-series table and their display units
_TRACKED_KPIS = {
    "gross_margin_pct":   "%",
    "net_margin_pct":     "%",
    "ebitda_margin_pct":  "%",
    "current_ratio":      "x",
    "quick_ratio":        "x",
    "debt_to_equity":     "x",
    "dso_days":           "days",
    "ccc_days":           "days",
    "basic_eps":          "$",
    "net_debt":           "$",
    "roe_pct":            "%",
    "effective_tax_rate": "%",
    "revenue_growth_pct": "%",
}

_DB_URL_DEFAULT = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', '..', 'ai_cfo.db')}"


class MemoryEngine:
    """
    Public API:
        save_analysis(company, period, results, hitl_state, sector) -> int (snapshot_id)
        update_hitl(company, period, decision, approver, notes)
        get_kpi_trends(company, kpi_names) -> {kpi: [{period, value, unit, ts}]}
        get_peer_benchmarks(exclude_company) -> {kpi: avg}
        synthesize_knowledge(company, adapter) -> str (summary)
        get_company_summary(company) -> dict
        get_all_companies() -> list[str]
        get_period_history(company) -> list[dict]
        get_active_insights(company) -> list[dict]
        get_hitl_log(company) -> list[dict]
    """

    def __init__(self, db_url: Optional[str] = None):
        url = db_url or os.getenv("DATABASE_URL", _DB_URL_DEFAULT)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self._engine = create_engine(url, connect_args=connect_args)
        MemoryBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

    # ── public: save ────────────────────────────────────────────────────────

    def save_analysis(
        self,
        company_name: str,
        period: str,
        results: Dict[str, Any],
        hitl_state: Optional[Dict[str, Any]] = None,
        sector: Optional[str] = None,
    ) -> int:
        """
        Persist a full pipeline run.

        Returns the snapshot id (same id if already saved via SHA-256 dedup).
        """
        data_hash = _sha256(results)

        with self._Session() as db:
            # Dedup check — same (company, period, hash) → no-op
            exact_match = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name, period=period, data_hash=data_hash)
                .first()
            )
            if exact_match:
                return exact_match.id

            # Ensure company master row
            company = db.query(CompanyMemory).filter_by(company_name=company_name).first()
            if not company:
                company = CompanyMemory(company_name=company_name, sector=sector)
                db.add(company)
                db.flush()

            # Previous snapshot for delta calculation (exclude current period)
            prev = (
                db.query(PeriodSnapshot)
                .filter(
                    PeriodSnapshot.company_name == company_name,
                    PeriodSnapshot.period != period,
                )
                .order_by(PeriodSnapshot.saved_at.desc())
                .first()
            )

            # Build snapshot field values
            kpis     = results.get("kpi_metrics") or {}
            gaap     = results.get("gaap_results") or {}
            ifrs     = results.get("ifrs_results") or {}
            hitl     = hitl_state or {}

            deltas = _compute_deltas(kpis, prev)

            snap_fields = dict(
                data_hash         = data_hash,
                revenue           = results.get("revenue"),
                gross_margin_pct  = kpis.get("gross_margin_pct"),
                net_margin_pct    = kpis.get("net_margin_pct"),
                ebitda_margin_pct = kpis.get("ebitda_margin_pct"),
                current_ratio     = kpis.get("current_ratio"),
                net_debt          = kpis.get("net_debt"),
                kpi_deltas_json   = json.dumps(deltas),
                gaap_compliant    = sum(1 for v in gaap.values() if v.get("status") == "COMPLIANT"),
                gaap_issues       = sum(1 for v in gaap.values() if v.get("status") != "COMPLIANT"),
                ifrs_compliant    = sum(1 for v in ifrs.values() if v.get("status") == "COMPLIANT"),
                ifrs_issues       = sum(1 for v in ifrs.values() if v.get("status") != "COMPLIANT"),
                hitl_required     = int(hitl.get("requires_human_approval", False)),
                hitl_decision     = hitl.get("human_decision"),
                hitl_approver     = hitl.get("approved_by"),
                hitl_notes        = hitl.get("human_feedback"),
                hitl_at           = datetime.utcnow() if hitl.get("human_decision") else None,
                full_results_json = json.dumps(results, default=str),
            )

            # Upsert: if a snapshot exists for this period (different hash = revised data), update it
            existing_period = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name, period=period)
                .first()
            )
            if existing_period:
                for k, v in snap_fields.items():
                    setattr(existing_period, k, v)
                existing_period.saved_at = datetime.utcnow()
                snap = existing_period
                db.flush()
                # Remove old KPI time-series for this snapshot (will be re-added below)
                db.query(KPITimeSeries).filter_by(snapshot_id=snap.id).delete()
            else:
                snap = PeriodSnapshot(
                    company_name=company_name,
                    period=period,
                    **snap_fields,
                )
                db.add(snap)
                db.flush()

            # KPI time-series rows
            for kpi_name, unit in _TRACKED_KPIS.items():
                if kpi_name in kpis:
                    db.add(KPITimeSeries(
                        company_name = company_name,
                        period       = period,
                        snapshot_id  = snap.id,
                        kpi_name     = kpi_name,
                        value        = float(kpis[kpi_name]),
                        unit         = unit,
                    ))

            # Insight rows: anomalies
            for flag in (results.get("anomaly_flags") or []):
                db.add(InsightMemory(
                    company_name = company_name,
                    period       = period,
                    insight_type = "anomaly",
                    severity     = "high",
                    source       = "math_engine",
                    description  = flag,
                ))

            # Insight rows: GAAP non-compliant / disclosure required
            for std, r in gaap.items():
                if r.get("status") != "COMPLIANT":
                    db.add(InsightMemory(
                        company_name = company_name,
                        period       = period,
                        insight_type = "gaap",
                        severity     = "critical" if r["status"] == "NON_COMPLIANT" else "high",
                        source       = std.upper(),
                        description  = r.get("finding", r.get("status", "")),
                    ))

            # Insight rows: IFRS non-compliant / disclosure required
            for std, r in ifrs.items():
                if r.get("status") != "COMPLIANT":
                    db.add(InsightMemory(
                        company_name = company_name,
                        period       = period,
                        insight_type = "ifrs",
                        severity     = "critical" if r["status"] == "NON_COMPLIANT" else "high",
                        source       = std.upper(),
                        description  = r.get("finding", r.get("status", "")),
                    ))

            # Update company master
            self._refresh_company_master(db, company_name, sector)

            db.commit()
            return snap.id

    def update_hitl(
        self,
        company_name: str,
        period: str,
        hitl_decision: str,
        hitl_approver: Optional[str] = None,
        hitl_notes: Optional[str] = None,
    ) -> bool:
        """Update the HITL decision on an existing snapshot."""
        with self._Session() as db:
            snap = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name, period=period)
                .order_by(PeriodSnapshot.saved_at.desc())
                .first()
            )
            if not snap:
                return False
            snap.hitl_decision = hitl_decision
            snap.hitl_approver = hitl_approver
            snap.hitl_notes    = hitl_notes
            snap.hitl_at       = datetime.utcnow()
            db.commit()
            return True

    # ── public: read ────────────────────────────────────────────────────────

    def get_kpi_trends(
        self,
        company_name: str,
        kpi_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """Return {kpi_name: [{period, value, unit, ts}]} sorted by period."""
        names = kpi_names or list(_TRACKED_KPIS.keys())
        with self._Session() as db:
            rows = (
                db.query(KPITimeSeries)
                .filter(
                    KPITimeSeries.company_name == company_name,
                    KPITimeSeries.kpi_name.in_(names),
                )
                .order_by(KPITimeSeries.recorded_at)
                .all()
            )
        result: Dict[str, List] = {n: [] for n in names}
        for r in rows:
            result[r.kpi_name].append({
                "period": r.period,
                "value":  r.value,
                "unit":   r.unit,
                "ts":     r.recorded_at.isoformat() if r.recorded_at else None,
            })
        return result

    def get_peer_benchmarks(self, exclude_company: Optional[str] = None) -> Dict[str, float]:
        """Return cross-company average for each tracked KPI."""
        with self._Session() as db:
            q = db.query(KPITimeSeries)
            if exclude_company:
                q = q.filter(KPITimeSeries.company_name != exclude_company)
            rows = q.all()

        totals: Dict[str, List[float]] = {}
        for r in rows:
            if r.value is not None:
                totals.setdefault(r.kpi_name, []).append(r.value)
        return {k: round(sum(v) / len(v), 4) for k, v in totals.items() if v}

    def synthesize_knowledge(self, company_name: str, adapter) -> str:
        """
        Ask the LLM to produce an institutional summary for a company.
        Saves the result to mem_company.institutional_summary.
        """
        trends = self.get_kpi_trends(company_name)
        benchmarks = self.get_peer_benchmarks(exclude_company=company_name)
        insights = self.get_active_insights(company_name)
        history = self.get_period_history(company_name)

        prompt = f"""You are a CFO analyst writing an institutional memory summary for {company_name}.

Based on {len(history)} periods of data:
Periods analysed: {[h['period'] for h in history]}

KPI trends (latest values):
{_format_latest_kpis(trends)}

Peer benchmark comparison (industry averages):
{json.dumps(benchmarks, indent=2)}

Active issues and insights ({len(insights)} total):
{_format_insights(insights)}

Write a concise 3-paragraph institutional summary covering:
1. Financial trajectory and KPI trends
2. Recurring compliance or operational issues
3. Key risks and recommended monitoring priorities
"""
        system = "You are a senior CFO analyst producing an institutional memory summary for internal use."
        try:
            summary = adapter.complete(system, prompt, max_tokens=800)
        except Exception as e:
            summary = f"[Summary generation failed: {e}]"

        with self._Session() as db:
            company = db.query(CompanyMemory).filter_by(company_name=company_name).first()
            if company:
                company.institutional_summary = summary
                company.summary_generated_at  = datetime.utcnow()
                db.commit()

        return summary

    def get_company_summary(self, company_name: str) -> Optional[Dict]:
        with self._Session() as db:
            company = db.query(CompanyMemory).filter_by(company_name=company_name).first()
            if not company:
                return None
            return {
                "company_name":           company.company_name,
                "sector":                 company.sector,
                "first_seen":             company.first_seen.isoformat() if company.first_seen else None,
                "last_updated":           company.last_updated.isoformat() if company.last_updated else None,
                "periods_analysed":       company.periods_analysed,
                "avg_gross_margin":       company.avg_gross_margin,
                "avg_net_margin":         company.avg_net_margin,
                "avg_current_ratio":      company.avg_current_ratio,
                "avg_revenue_growth":     company.avg_revenue_growth,
                "avg_ebitda_margin":      company.avg_ebitda_margin,
                "recurring_issues":       company.recurring_issues,
                "institutional_summary":  company.institutional_summary,
                "summary_generated_at":   company.summary_generated_at.isoformat() if company.summary_generated_at else None,
            }

    def get_all_companies(self) -> List[str]:
        with self._Session() as db:
            rows = db.query(CompanyMemory.company_name).all()
        return [r[0] for r in rows]

    def get_period_history(self, company_name: str) -> List[Dict]:
        with self._Session() as db:
            snaps = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name)
                .order_by(PeriodSnapshot.saved_at)
                .all()
            )
            return [
                {
                    "period":           s.period,
                    "saved_at":         s.saved_at.isoformat() if s.saved_at else None,
                    "gross_margin_pct": s.gross_margin_pct,
                    "net_margin_pct":   s.net_margin_pct,
                    "current_ratio":    s.current_ratio,
                    "gaap_compliant":   s.gaap_compliant,
                    "gaap_issues":      s.gaap_issues,
                    "hitl_decision":    s.hitl_decision,
                    "kpi_deltas":       s.kpi_deltas,
                }
                for s in snaps
            ]

    def get_active_insights(self, company_name: str) -> List[Dict]:
        with self._Session() as db:
            rows = (
                db.query(InsightMemory)
                .filter_by(company_name=company_name, active=1)
                .order_by(InsightMemory.created_at.desc())
                .all()
            )
            return [
                {
                    "period":       r.period,
                    "type":         r.insight_type,
                    "severity":     r.severity,
                    "source":       r.source,
                    "description":  r.description,
                    "created_at":   r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def get_hitl_log(self, company_name: str) -> List[Dict]:
        with self._Session() as db:
            snaps = (
                db.query(PeriodSnapshot)
                .filter(
                    PeriodSnapshot.company_name == company_name,
                    PeriodSnapshot.hitl_required == 1,
                )
                .order_by(PeriodSnapshot.saved_at.desc())
                .all()
            )
            return [
                {
                    "period":       s.period,
                    "decision":     s.hitl_decision,
                    "approver":     s.hitl_approver,
                    "notes":        s.hitl_notes,
                    "decided_at":   s.hitl_at.isoformat() if s.hitl_at else None,
                }
                for s in snaps
            ]

    # ── private helpers ─────────────────────────────────────────────────────

    def _refresh_company_master(self, db: Session, company_name: str, sector: Optional[str]):
        company = db.query(CompanyMemory).filter_by(company_name=company_name).first()
        if not company:
            return

        snaps = (
            db.query(PeriodSnapshot)
            .filter_by(company_name=company_name)
            .all()
        )
        n = len(snaps)
        if n == 0:
            return

        def _avg(attr):
            vals = [getattr(s, attr) for s in snaps if getattr(s, attr) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        company.periods_analysed  = n
        company.avg_gross_margin  = _avg("gross_margin_pct")
        company.avg_net_margin    = _avg("net_margin_pct")
        company.avg_current_ratio = _avg("current_ratio")
        company.avg_ebitda_margin = _avg("ebitda_margin_pct")
        company.last_updated      = datetime.utcnow()
        if sector:
            company.sector = sector

        # Recurring issues: description appearing in ≥2 periods
        all_insights = (
            db.query(InsightMemory)
            .filter_by(company_name=company_name)
            .all()
        )
        desc_counts: Dict[str, int] = {}
        for ins in all_insights:
            desc_counts[ins.description] = desc_counts.get(ins.description, 0) + 1
        recurring = [d for d, c in desc_counts.items() if c >= 2]
        company.recurring_issues = recurring[:10]


# ── module helpers ───────────────────────────────────────────────────────────

def _sha256(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _compute_deltas(kpis: Dict[str, float], prev: Optional[PeriodSnapshot]) -> Dict:
    if not prev:
        return {}
    deltas = {}
    prev_kpis = {}
    if prev.full_results_json:
        prev_kpis = (json.loads(prev.full_results_json) or {}).get("kpi_metrics") or {}
    for k, v in kpis.items():
        if k in prev_kpis and prev_kpis[k] not in (None, 0):
            pv = prev_kpis[k]
            deltas[k] = {
                "value":      v,
                "prev_value": pv,
                "delta_pct":  round((v - pv) / abs(pv) * 100, 2),
            }
    return deltas


def _format_latest_kpis(trends: Dict[str, List]) -> str:
    lines = []
    for kpi, history in trends.items():
        if history:
            last = history[-1]
            lines.append(f"  {kpi}: {last['value']}{last['unit']} (period: {last['period']})")
    return "\n".join(lines) or "  No KPI data available"


def _format_insights(insights: List[Dict]) -> str:
    if not insights:
        return "  No active insights"
    return "\n".join(
        f"  [{i['severity'].upper()}] {i['type']}: {i['description']} ({i['period']})"
        for i in insights[:10]
    )


# ── module singleton ─────────────────────────────────────────────────────────

_singleton: Optional[MemoryEngine] = None


def get_memory_engine(db_url: Optional[str] = None) -> MemoryEngine:
    global _singleton
    if db_url:
        return MemoryEngine(db_url=db_url)
    if _singleton is None:
        _singleton = MemoryEngine()
    return _singleton
