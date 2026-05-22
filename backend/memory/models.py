"""
SQLAlchemy models for institutional memory.
Uses a separate MemoryBase so these tables can be created independently
without touching the existing Task/Approval/Document schema.
"""
import json
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class MemoryBase(DeclarativeBase):
    pass


class CompanyMemory(MemoryBase):
    """One master row per company — rolling averages and institutional summary."""
    __tablename__ = "mem_company"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    company_name     = Column(String(255), unique=True, nullable=False, index=True)
    sector           = Column(String(100), nullable=True)
    first_seen       = Column(DateTime, default=datetime.utcnow)
    last_updated     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    periods_analysed = Column(Integer, default=0)

    # Rolling average KPIs (recomputed on every save)
    avg_gross_margin    = Column(Float, nullable=True)
    avg_net_margin      = Column(Float, nullable=True)
    avg_current_ratio   = Column(Float, nullable=True)
    avg_revenue_growth  = Column(Float, nullable=True)
    avg_ebitda_margin   = Column(Float, nullable=True)

    # Recurring issues (JSON list of strings)
    recurring_issues_json = Column(Text, default="[]")

    # AI-generated institutional summary (updated by synthesize_knowledge)
    institutional_summary = Column(Text, nullable=True)
    summary_generated_at  = Column(DateTime, nullable=True)

    snapshots = relationship("PeriodSnapshot", back_populates="company", cascade="all, delete-orphan")
    insights  = relationship("InsightMemory",  back_populates="company", cascade="all, delete-orphan")

    @property
    def recurring_issues(self):
        return json.loads(self.recurring_issues_json or "[]")

    @recurring_issues.setter
    def recurring_issues(self, value):
        self.recurring_issues_json = json.dumps(value)


class PeriodSnapshot(MemoryBase):
    """Full analysis snapshot per company × period."""
    __tablename__ = "mem_period_snapshot"
    __table_args__ = (UniqueConstraint("company_name", "period", name="uq_company_period"),)

    id           = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(255), ForeignKey("mem_company.company_name"), nullable=False, index=True)
    period       = Column(String(50),  nullable=False)
    data_hash    = Column(String(64),  nullable=False)         # SHA-256 of input data
    saved_at     = Column(DateTime,    default=datetime.utcnow)

    # Summary KPIs (denormalised for fast dashboard queries)
    revenue          = Column(Float, nullable=True)
    gross_margin_pct = Column(Float, nullable=True)
    net_margin_pct   = Column(Float, nullable=True)
    ebitda_margin_pct= Column(Float, nullable=True)
    current_ratio    = Column(Float, nullable=True)
    net_debt         = Column(Float, nullable=True)

    # Period-over-period deltas (JSON: {kpi_name: {value, prev_value, delta_pct}})
    kpi_deltas_json  = Column(Text, default="{}")

    # GAAP/IFRS summary counts
    gaap_compliant   = Column(Integer, nullable=True)
    gaap_issues      = Column(Integer, nullable=True)
    ifrs_compliant   = Column(Integer, nullable=True)
    ifrs_issues      = Column(Integer, nullable=True)

    # HITL decision
    hitl_required    = Column(Integer, default=0)   # 0/1 boolean
    hitl_decision    = Column(String(20), nullable=True)  # approved|rejected|pending
    hitl_approver    = Column(String(100), nullable=True)
    hitl_notes       = Column(Text, nullable=True)
    hitl_at          = Column(DateTime, nullable=True)

    # Full results blob (compressed JSON)
    full_results_json = Column(Text, nullable=True)

    company   = relationship("CompanyMemory", back_populates="snapshots")
    kpi_series = relationship("KPITimeSeries", back_populates="snapshot", cascade="all, delete-orphan")

    @property
    def kpi_deltas(self):
        return json.loads(self.kpi_deltas_json or "{}")

    @kpi_deltas.setter
    def kpi_deltas(self, value):
        self.kpi_deltas_json = json.dumps(value)

    @property
    def full_results(self):
        return json.loads(self.full_results_json or "{}")

    @full_results.setter
    def full_results(self, value):
        self.full_results_json = json.dumps(value, default=str)


class KPITimeSeries(MemoryBase):
    """One row per company × period × KPI — used for Altair trend charts."""
    __tablename__ = "mem_kpi_ts"
    __table_args__ = (
        UniqueConstraint("company_name", "period", "kpi_name", name="uq_ts_kpi"),
    )

    id           = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(255), nullable=False, index=True)
    period       = Column(String(50),  nullable=False)
    snapshot_id  = Column(Integer, ForeignKey("mem_period_snapshot.id"), nullable=False)
    kpi_name     = Column(String(100), nullable=False)
    value        = Column(Float,       nullable=True)
    unit         = Column(String(20),  nullable=True)  # "%" | "$" | "x" | "days"
    recorded_at  = Column(DateTime,    default=datetime.utcnow)

    snapshot = relationship("PeriodSnapshot", back_populates="kpi_series")


class InsightMemory(MemoryBase):
    """Anomalies, compliance findings, and AI-flagged items per company."""
    __tablename__ = "mem_insights"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(255), ForeignKey("mem_company.company_name"), nullable=False, index=True)
    period       = Column(String(50),  nullable=False)
    insight_type = Column(String(50),  nullable=False)   # anomaly|gaap|ifrs|hitl|risk
    severity     = Column(String(20),  nullable=False)   # critical|high|medium|low
    source       = Column(String(100), nullable=True)    # which engine/standard
    description  = Column(Text,        nullable=False)
    active       = Column(Integer,     default=1)        # 0/1
    created_at   = Column(DateTime,    default=datetime.utcnow)

    company = relationship("CompanyMemory", back_populates="insights")
