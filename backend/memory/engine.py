"""
Institutional Memory Engine.

Persists and retrieves accumulated company knowledge across sessions.
All reads/writes go through SQLite (same file as the main DB).
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    CompanyMemory,
    InsightMemory,
    KPITimeSeries,
    MemoryBase,
    PeriodSnapshot,
)

# ── KPI units registry ────────────────────────────────────────────────────────
_KPI_UNITS: Dict[str, str] = {
    "gross_margin_pct":   "pct",  "ebitda_margin_pct":  "pct",
    "net_margin_pct":     "pct",  "roe_pct":            "pct",
    "roa_pct":            "pct",  "effective_tax_rate":  "pct",
    "current_ratio":      "ratio","quick_ratio":         "ratio",
    "debt_to_equity":     "ratio","interest_coverage":   "ratio",
    "dso_days":           "days", "dpo_days":            "days",
    "ccc_days":           "days",
    "net_debt":           "usd",  "working_capital":     "usd",
    "basic_eps":          "usd",  "diluted_eps":         "usd",
}

# KPIs written to the time-series table every save
_TRACKED_KPIS = list(_KPI_UNITS.keys())


def _jdump(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _jload(s: str, default: Any = None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class MemoryEngine:
    """
    Public API used by the Streamlit app and pipeline agents.
    Thread-safe for single-process use (Streamlit).
    """

    def __init__(self, db_url: Optional[str] = None):
        url = db_url or os.getenv("DATABASE_URL", "sqlite:///./ai_cfo.db")
        kwargs: Dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self._engine = create_engine(url, **kwargs)
        MemoryBase.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)

    # ── Session helper ────────────────────────────────────────────────────────

    def _session(self) -> Session:
        return self._Session()

    # ═════════════════════════════════════════════════════════════════════════
    #  WRITE
    # ═════════════════════════════════════════════════════════════════════════

    def save_analysis(
        self,
        company_name: str,
        period: str,
        results: Dict[str, Any],
        hitl_state: Optional[Dict[str, Any]] = None,
        sector: str = "",
    ) -> str:
        """
        Persist one complete analysis cycle.  Returns the snapshot ID.
        Safe to call multiple times — deduplicates by data_hash.
        """
        data_hash = _sha256(results.get("data", {}))

        with self._Session() as db:
            # ── Deduplication: skip if same data already stored ──────────────
            existing = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name, period=period, data_hash=data_hash)
                .first()
            )
            if existing:
                # Still update HITL decision if supplied
                if hitl_state and hitl_state.get("hitl_decision") not in (None, "pending"):
                    existing.hitl_decision  = hitl_state.get("hitl_decision", "pending")
                    existing.hitl_approver  = hitl_state.get("hitl_approver", "")
                    existing.hitl_notes     = hitl_state.get("hitl_notes", "")
                    existing.hitl_timestamp = datetime.utcnow()
                    db.commit()
                    self._refresh_company_master(db, company_name, sector)
                return existing.id

            # ── Build period-over-period deltas ──────────────────────────────
            prev = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name)
                .order_by(PeriodSnapshot.analyzed_at.desc())
                .first()
            )
            kpi_deltas: Dict[str, Any] = {}
            curr_kpis: Dict[str, float] = results.get("kpis", {})
            if prev:
                prev_kpis = _jload(prev.kpi_metrics, {})
                for k in _TRACKED_KPIS:
                    pv = prev_kpis.get(k)
                    cv = curr_kpis.get(k)
                    if pv is not None and cv is not None:
                        kpi_deltas[k] = {
                            "prev":  round(pv, 4),
                            "curr":  round(cv, 4),
                            "delta": round(cv - pv, 4),
                            "delta_pct": round((cv - pv) / abs(pv) * 100, 2) if pv else 0.0,
                        }

            # ── Snapshot ──────────────────────────────────────────────────────
            snap_id = str(uuid.uuid4())
            snap = PeriodSnapshot(
                id               = snap_id,
                company_name     = company_name,
                period           = period,
                data_hash        = data_hash,
                kpi_metrics      = _jdump(curr_kpis),
                variance_table   = _jdump(results.get("variance", {})),
                anomaly_flags    = _jdump(results.get("anomalies", [])),
                gaap_results     = _jdump(results.get("gaap", {})),
                ifrs_results     = _jdump(results.get("ifrs", {})),
                approval_triggers = _jdump(
                    results.get("approval_triggers", [])
                ),
                runway_status    = results.get("runway", {}).get("status", "ADEQUATE"),
                ai_narrative     = results.get("ai_narrative", ""),
                ai_confidence    = float(
                    results.get("ai_confidence", 0.0) or 0.0
                ),
                hitl_decision    = (hitl_state or {}).get("hitl_decision", "pending"),
                hitl_approver    = (hitl_state or {}).get("hitl_approver", ""),
                hitl_notes       = (hitl_state or {}).get("hitl_notes", ""),
                hitl_timestamp   = (
                    datetime.utcnow()
                    if (hitl_state or {}).get("hitl_decision") not in (None, "pending")
                    else None
                ),
                kpi_deltas       = _jdump(kpi_deltas),
            )
            db.add(snap)

            # ── KPI time-series rows ──────────────────────────────────────────
            for kpi_name in _TRACKED_KPIS:
                val = curr_kpis.get(kpi_name)
                if val is not None:
                    db.add(KPITimeSeries(
                        company_name = company_name,
                        period       = period,
                        kpi_name     = kpi_name,
                        kpi_value    = float(val),
                        kpi_unit     = _KPI_UNITS.get(kpi_name, ""),
                    ))

            # ── Insights ──────────────────────────────────────────────────────
            for flag in results.get("anomalies", []):
                sev = "critical" if "CRITICAL" in flag else "high"
                db.add(InsightMemory(
                    id           = str(uuid.uuid4()),
                    company_name = company_name,
                    period       = period,
                    insight_type = "anomaly",
                    severity     = sev,
                    content      = flag,
                    source       = "anomaly_detection",
                ))

            for std, res in results.get("gaap", {}).items():
                if res.get("status") in ("NON_COMPLIANT", "DISCLOSURE_REQUIRED"):
                    db.add(InsightMemory(
                        id           = str(uuid.uuid4()),
                        company_name = company_name,
                        period       = period,
                        insight_type = "compliance",
                        severity     = "critical" if res["status"] == "NON_COMPLIANT" else "high",
                        content      = f"GAAP {res.get('standard', std)}: {res.get('finding', '')}",
                        source       = "gaap_engine",
                    ))

            for std, res in results.get("ifrs", {}).items():
                if res.get("status") in ("NON_COMPLIANT", "DISCLOSURE_REQUIRED"):
                    db.add(InsightMemory(
                        id           = str(uuid.uuid4()),
                        company_name = company_name,
                        period       = period,
                        insight_type = "compliance",
                        severity     = "critical" if res["status"] == "NON_COMPLIANT" else "high",
                        content      = f"IFRS {res.get('standard', std)}: {res.get('finding', '')}",
                        source       = "ifrs_engine",
                    ))

            db.commit()
            self._refresh_company_master(db, company_name, sector)
            return snap_id

    def update_hitl(
        self,
        company_name: str,
        period: str,
        hitl_decision: str,
        hitl_approver: str,
        hitl_notes: str,
    ) -> None:
        """Update HITL decision on the most-recent snapshot for this company/period."""
        with self._Session() as db:
            snap = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name, period=period)
                .order_by(PeriodSnapshot.analyzed_at.desc())
                .first()
            )
            if snap:
                snap.hitl_decision  = hitl_decision
                snap.hitl_approver  = hitl_approver
                snap.hitl_notes     = hitl_notes
                snap.hitl_timestamp = datetime.utcnow()
                db.commit()
            self._refresh_company_master(db, company_name)

    # ═════════════════════════════════════════════════════════════════════════
    #  READ
    # ═════════════════════════════════════════════════════════════════════════

    def get_all_companies(self) -> List[Dict[str, Any]]:
        with self._Session() as db:
            rows = db.query(CompanyMemory).order_by(CompanyMemory.last_analyzed.desc()).all()
            return [
                {
                    "company_name":    r.company_name,
                    "sector":          r.sector,
                    "analysis_count":  r.analysis_count,
                    "first_analyzed":  r.first_analyzed,
                    "last_analyzed":   r.last_analyzed,
                    "avg_gross_margin": round(r.avg_gross_margin or 0, 2),
                    "avg_ebitda_margin": round(r.avg_ebitda_margin or 0, 2),
                    "avg_net_margin":   round(r.avg_net_margin or 0, 2),
                    "hitl_approval_rate": round(r.hitl_approval_rate or 0, 2),
                    "institutional_summary": r.institutional_summary or "",
                    "recurring_anomalies":   _jload(r.recurring_anomalies, []),
                    "compliance_issues":     _jload(r.compliance_issues, []),
                }
                for r in rows
            ]

    def get_company_snapshots(self, company_name: str) -> List[Dict[str, Any]]:
        with self._Session() as db:
            rows = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name)
                .order_by(PeriodSnapshot.analyzed_at.asc())
                .all()
            )
            return [
                {
                    "id":             r.id,
                    "period":         r.period,
                    "analyzed_at":    r.analyzed_at,
                    "kpi_metrics":    _jload(r.kpi_metrics, {}),
                    "anomaly_flags":  _jload(r.anomaly_flags, []),
                    "runway_status":  r.runway_status,
                    "hitl_decision":  r.hitl_decision,
                    "hitl_approver":  r.hitl_approver,
                    "hitl_notes":     r.hitl_notes,
                    "hitl_timestamp": r.hitl_timestamp,
                    "kpi_deltas":     _jload(r.kpi_deltas, {}),
                    "approval_triggers": _jload(r.approval_triggers, []),
                    "ai_narrative":   r.ai_narrative,
                    "ai_confidence":  r.ai_confidence,
                }
                for r in rows
            ]

    def get_kpi_trends(
        self,
        company_name: str,
        kpi_names: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return {kpi_name: [{period, value, unit}, ...]} sorted by period."""
        names = kpi_names or _TRACKED_KPIS
        with self._Session() as db:
            rows = (
                db.query(KPITimeSeries)
                .filter(
                    KPITimeSeries.company_name == company_name,
                    KPITimeSeries.kpi_name.in_(names),
                )
                .order_by(KPITimeSeries.analyzed_at.asc())
                .all()
            )
            out: Dict[str, list] = {n: [] for n in names}
            for r in rows:
                out[r.kpi_name].append({
                    "period":  r.period,
                    "value":   r.kpi_value,
                    "unit":    r.kpi_unit,
                    "ts":      r.analyzed_at,
                })
            return {k: v for k, v in out.items() if v}

    def get_insights(
        self,
        company_name: str,
        insight_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        with self._Session() as db:
            q = db.query(InsightMemory).filter_by(company_name=company_name)
            if insight_type:
                q = q.filter_by(insight_type=insight_type)
            if active_only:
                q = q.filter_by(resolved=0)
            rows = q.order_by(InsightMemory.created_at.desc()).all()
            return [
                {
                    "id":           r.id,
                    "period":       r.period,
                    "insight_type": r.insight_type,
                    "severity":     r.severity,
                    "content":      r.content,
                    "source":       r.source,
                    "resolved":     bool(r.resolved),
                    "created_at":   r.created_at,
                }
                for r in rows
            ]

    def get_peer_benchmarks(self, exclude_company: Optional[str] = None) -> Dict[str, Any]:
        """Cross-company averages for benchmarking."""
        with self._Session() as db:
            q = db.query(CompanyMemory)
            if exclude_company:
                q = q.filter(CompanyMemory.company_name != exclude_company)
            rows = q.all()
            if not rows:
                return {}
            n = len(rows)
            return {
                "peer_count":         n,
                "avg_gross_margin":   round(sum(r.avg_gross_margin or 0 for r in rows) / n, 2),
                "avg_ebitda_margin":  round(sum(r.avg_ebitda_margin or 0 for r in rows) / n, 2),
                "avg_net_margin":     round(sum(r.avg_net_margin or 0 for r in rows) / n, 2),
                "avg_current_ratio":  round(sum(r.avg_current_ratio or 0 for r in rows) / n, 2),
                "avg_approval_rate":  round(sum(r.hitl_approval_rate or 0 for r in rows) / n, 2),
            }

    def get_period_delta(self, company_name: str, period: str) -> Dict[str, Any]:
        """Return KPI deltas for the snapshot matching this period."""
        with self._Session() as db:
            snap = (
                db.query(PeriodSnapshot)
                .filter_by(company_name=company_name, period=period)
                .order_by(PeriodSnapshot.analyzed_at.desc())
                .first()
            )
            return _jload(snap.kpi_deltas if snap else "{}", {})

    def synthesize_knowledge(self, company_name: str, adapter: Any) -> str:
        """
        Call the LLM to produce a concise institutional knowledge summary.
        Updates CompanyMemory.institutional_summary in-place and returns it.
        """
        companies = self.get_all_companies()
        co = next((c for c in companies if c["company_name"] == company_name), None)
        if not co:
            return "No history found for this company."

        snapshots = self.get_company_snapshots(company_name)
        insights  = self.get_insights(company_name, active_only=False)
        peer      = self.get_peer_benchmarks(exclude_company=company_name)

        # Build a compact summary for the LLM
        snap_lines = []
        for s in snapshots[-6:]:   # last 6 periods
            kpis = s["kpi_metrics"]
            snap_lines.append(
                f"  {s['period']}: GM={kpis.get('gross_margin_pct', 0):.1f}%  "
                f"EBITDA={kpis.get('ebitda_margin_pct', 0):.1f}%  "
                f"NM={kpis.get('net_margin_pct', 0):.1f}%  "
                f"CR={kpis.get('current_ratio', 0):.2f}x  "
                f"HITL={s['hitl_decision']}"
            )

        issue_lines = [f"  [{i['period']}] {i['insight_type'].upper()}: {i['content'][:120]}"
                       for i in insights[-10:]]

        prompt = f"""You are a Lead Audit CPA maintaining institutional memory for {company_name}.

Historical KPI snapshot (up to 6 most recent periods):
{chr(10).join(snap_lines) if snap_lines else '  No data yet'}

Active insights & recurring issues (up to 10):
{chr(10).join(issue_lines) if issue_lines else '  None recorded'}

Peer benchmarks ({peer.get('peer_count', 0)} companies):
  Avg Gross Margin: {peer.get('avg_gross_margin', 0):.1f}%
  Avg EBITDA Margin: {peer.get('avg_ebitda_margin', 0):.1f}%
  Avg Current Ratio: {peer.get('avg_current_ratio', 0):.2f}x

Produce a concise (≤300 words) Institutional Knowledge Summary covering:
1. Business performance trajectory (improving/declining/stable) with evidence
2. Recurring risks or compliance issues the CFO must watch
3. Where this company stands vs peer benchmarks
4. One CFO-level strategic recommendation based on the pattern

Be specific with numbers.  Do not mention limitations of your knowledge."""

        try:
            summary = adapter.complete(
                "You are a Lead Audit CPA maintaining institutional memory. Be concise and specific.",
                prompt,
                max_tokens=600,
            )
        except Exception as exc:
            summary = f"[AI synthesis unavailable: {exc}]"

        with self._Session() as db:
            cm = db.query(CompanyMemory).filter_by(company_name=company_name).first()
            if cm:
                cm.institutional_summary = summary
                cm.updated_at            = datetime.utcnow()
                db.commit()

        return summary

    # ═════════════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _refresh_company_master(
        self, db: Session, company_name: str, sector: str = ""
    ) -> None:
        """Recompute rolling stats for the CompanyMemory master row."""
        snaps = (
            db.query(PeriodSnapshot)
            .filter_by(company_name=company_name)
            .order_by(PeriodSnapshot.analyzed_at.asc())
            .all()
        )
        if not snaps:
            return

        cm = db.query(CompanyMemory).filter_by(company_name=company_name).first()
        if not cm:
            cm = CompanyMemory(
                id           = str(uuid.uuid4()),
                company_name = company_name,
                sector       = sector,
                first_analyzed = snaps[0].analyzed_at,
            )
            db.add(cm)

        kpi_lists: Dict[str, List[float]] = {k: [] for k in _TRACKED_KPIS}
        revenues: List[float] = []
        approved = rejected = 0

        for s in snaps:
            kpis = _jload(s.kpi_metrics, {})
            for k in _TRACKED_KPIS:
                if k in kpis:
                    kpi_lists[k].append(float(kpis[k]))
            # Revenue for growth calc
            data_raw = kpis  # kpi_metrics only; raw revenue not stored here
            if s.hitl_decision == "approved":
                approved += 1
            elif s.hitl_decision == "rejected":
                rejected += 1

        def _avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        cm.analysis_count    = len(snaps)
        cm.last_analyzed     = snaps[-1].analyzed_at
        cm.avg_gross_margin  = _avg(kpi_lists["gross_margin_pct"])
        cm.avg_ebitda_margin = _avg(kpi_lists["ebitda_margin_pct"])
        cm.avg_net_margin    = _avg(kpi_lists["net_margin_pct"])
        cm.avg_current_ratio = _avg(kpi_lists["current_ratio"])
        if sector:
            cm.sector = sector

        total_hitl = approved + rejected
        cm.hitl_approval_rate = round(approved / total_hitl * 100, 1) if total_hitl else 0.0

        # Recurring anomalies: anomaly text seen in ≥ 2 periods
        all_anomalies: Dict[str, int] = {}
        for s in snaps:
            for a in _jload(s.anomaly_flags, []):
                all_anomalies[a] = all_anomalies.get(a, 0) + 1
        cm.recurring_anomalies = _jdump(
            [a for a, cnt in all_anomalies.items() if cnt >= 2]
        )

        # Recurring compliance issues
        all_issues: Dict[str, int] = {}
        for s in snaps:
            for std, res in _jload(s.gaap_results, {}).items():
                if res.get("status") in ("NON_COMPLIANT", "DISCLOSURE_REQUIRED"):
                    all_issues[std] = all_issues.get(std, 0) + 1
            for std, res in _jload(s.ifrs_results, {}).items():
                if res.get("status") in ("NON_COMPLIANT", "DISCLOSURE_REQUIRED"):
                    all_issues[std] = all_issues.get(std, 0) + 1
        cm.compliance_issues = _jdump(
            [std for std, cnt in all_issues.items() if cnt >= 2]
        )

        cm.updated_at = datetime.utcnow()
        db.commit()


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: Optional[MemoryEngine] = None


def get_memory_engine() -> MemoryEngine:
    global _singleton
    if _singleton is None:
        _singleton = MemoryEngine()
    return _singleton
