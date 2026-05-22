"""
Institutional Memory — SQLAlchemy models.
Accumulates company KPIs, insights, HITL decisions, and compliance history
across months and years.  Uses the same SQLite engine as the main DB.
"""
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Index, Integer, String, Text,
)
from sqlalchemy.orm import declarative_base

# Use a separate Base so these tables can be migrated independently.
MemoryBase = declarative_base()


class CompanyMemory(MemoryBase):
    """Master record per company — rolling statistics and AI-synthesized profile."""
    __tablename__ = "mem_company"

    id               = Column(String(64),  primary_key=True)
    company_name     = Column(String(256), nullable=False, unique=True, index=True)
    sector           = Column(String(128), default="")
    first_analyzed   = Column(DateTime,    default=datetime.utcnow)
    last_analyzed    = Column(DateTime,    default=datetime.utcnow)
    analysis_count   = Column(Integer,     default=0)

    # Rolling KPI averages (updated on every save)
    avg_gross_margin    = Column(Float, default=0.0)
    avg_ebitda_margin   = Column(Float, default=0.0)
    avg_net_margin      = Column(Float, default=0.0)
    avg_revenue_growth  = Column(Float, default=0.0)
    avg_current_ratio   = Column(Float, default=0.0)

    # Peak / trough tracking
    peak_revenue        = Column(Float, default=0.0)
    trough_gross_margin = Column(Float, default=100.0)

    # Recurring issue tracking (JSON-encoded lists)
    recurring_anomalies  = Column(Text, default="[]")
    compliance_issues    = Column(Text, default="[]")
    hitl_approval_rate   = Column(Float, default=0.0)   # % of periods approved

    # AI-synthesized narrative (refreshed by synthesize_knowledge())
    institutional_summary = Column(Text, default="")
    updated_at            = Column(DateTime, default=datetime.utcnow)


class PeriodSnapshot(MemoryBase):
    """Full analysis snapshot: one row per company × period."""
    __tablename__ = "mem_period_snapshot"

    id           = Column(String(64),  primary_key=True)
    company_name = Column(String(256), nullable=False, index=True)
    period       = Column(String(64),  nullable=False)
    analyzed_at  = Column(DateTime,    default=datetime.utcnow)
    data_hash    = Column(String(64),  default="")   # SHA-256 of raw financials

    # Deterministic outputs (JSON text)
    kpi_metrics      = Column(Text, default="{}")
    variance_table   = Column(Text, default="{}")
    anomaly_flags    = Column(Text, default="[]")
    gaap_results     = Column(Text, default="{}")
    ifrs_results     = Column(Text, default="{}")
    approval_triggers = Column(Text, default="[]")
    runway_status    = Column(String(32), default="ADEQUATE")

    # AI narrative
    ai_narrative  = Column(Text,  default="")
    ai_confidence = Column(Float, default=0.0)

    # HITL decision
    hitl_decision  = Column(String(32),  default="pending")
    hitl_approver  = Column(String(256), default="")
    hitl_notes     = Column(Text,        default="")
    hitl_timestamp = Column(DateTime,    nullable=True)

    # Period-over-period deltas vs previous snapshot (JSON)
    kpi_deltas = Column(Text, default="{}")

    __table_args__ = (
        Index("ix_mem_snapshot_co_period", "company_name", "period"),
    )


class KPITimeSeries(MemoryBase):
    """One row per company × period × KPI — optimised for trend charts."""
    __tablename__ = "mem_kpi_ts"

    id           = Column(Integer,     primary_key=True, autoincrement=True)
    company_name = Column(String(256), nullable=False, index=True)
    period       = Column(String(64),  nullable=False)
    analyzed_at  = Column(DateTime,    default=datetime.utcnow)
    kpi_name     = Column(String(128), nullable=False)
    kpi_value    = Column(Float,       nullable=False)
    kpi_unit     = Column(String(32),  default="")   # pct | ratio | days | usd

    __table_args__ = (
        Index("ix_mem_kpi_co_name", "company_name", "kpi_name"),
    )


class InsightMemory(MemoryBase):
    """Key insights extracted from each analysis cycle."""
    __tablename__ = "mem_insights"

    id            = Column(String(64),  primary_key=True)
    company_name  = Column(String(256), nullable=False, index=True)
    period        = Column(String(64),  nullable=False)
    insight_type  = Column(String(64),  nullable=False)   # risk|opportunity|pattern|anomaly|compliance
    severity      = Column(String(32),  default="medium")  # critical|high|medium|low
    content       = Column(Text,        nullable=False)
    source        = Column(String(64),  default="system")  # ai_analysis|hitl|anomaly|gaap|ifrs
    resolved      = Column(Integer,     default=0)          # 0=active 1=resolved
    created_at    = Column(DateTime,    default=datetime.utcnow)
    resolved_at   = Column(DateTime,    nullable=True)

    __table_args__ = (
        Index("ix_mem_insight_co_type", "company_name", "insight_type"),
    )
