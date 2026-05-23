"""Initial schema — all 7 tables.

Revision ID: 0001
Revises:
Create Date: 2026-05-23
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Core app tables ────────────────────────────────────────────────────────

    op.create_table(
        "tasks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("company_name", sa.String(256)),
        sa.Column("period", sa.String(64)),
        sa.Column("report_format", sa.String(64), server_default="board"),
        sa.Column("submitted_by", sa.String(256)),
        sa.Column("submitted_at", sa.DateTime),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("kpi_metrics", sa.JSON),
        sa.Column("variance_table", sa.JSON),
        sa.Column("gaap_results", sa.JSON),
        sa.Column("ifrs_results", sa.JSON),
        sa.Column("analysis_narrative", sa.Text),
        sa.Column("final_report", sa.Text),
        sa.Column("audit_log", sa.JSON),
        sa.Column("errors", sa.JSON),
        sa.Column("total_tokens_used", sa.Integer, server_default="0"),
        sa.Column("total_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("processing_time_ms", sa.Integer, server_default="0"),
        sa.Column("updated_at", sa.DateTime),
    )

    op.create_table(
        "approvals",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("triggers", sa.JSON),
        sa.Column("decision", sa.String(32)),
        sa.Column("feedback", sa.Text),
        sa.Column("approved_by", sa.String(256)),
        sa.Column("created_at", sa.DateTime),
        sa.Column("decided_at", sa.DateTime),
    )

    op.create_table(
        "rag_documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("title", sa.String(512)),
        sa.Column("content", sa.Text),
        sa.Column("category", sa.String(64)),
        sa.Column("min_role", sa.String(32), server_default="analyst"),
        sa.Column("indexed_at", sa.DateTime),
        sa.Column("metadata", sa.JSON),
    )

    # ── Institutional memory tables ────────────────────────────────────────────

    op.create_table(
        "mem_company",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("company_name", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("sector", sa.String(100)),
        sa.Column("first_seen", sa.DateTime),
        sa.Column("last_updated", sa.DateTime),
        sa.Column("periods_analysed", sa.Integer, server_default="0"),
        sa.Column("avg_gross_margin", sa.Float),
        sa.Column("avg_net_margin", sa.Float),
        sa.Column("avg_current_ratio", sa.Float),
        sa.Column("avg_revenue_growth", sa.Float),
        sa.Column("avg_ebitda_margin", sa.Float),
        sa.Column("recurring_issues_json", sa.Text, server_default="[]"),
        sa.Column("institutional_summary", sa.Text),
        sa.Column("summary_generated_at", sa.DateTime),
    )

    op.create_table(
        "mem_period_snapshot",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("company_name", sa.String(255), sa.ForeignKey("mem_company.company_name"), nullable=False),
        sa.Column("period", sa.String(50), nullable=False),
        sa.Column("data_hash", sa.String(64), nullable=False),
        sa.Column("saved_at", sa.DateTime),
        sa.Column("revenue", sa.Float),
        sa.Column("gross_margin_pct", sa.Float),
        sa.Column("net_margin_pct", sa.Float),
        sa.Column("ebitda_margin_pct", sa.Float),
        sa.Column("current_ratio", sa.Float),
        sa.Column("net_debt", sa.Float),
        sa.Column("kpi_deltas_json", sa.Text, server_default="{}"),
        sa.Column("gaap_compliant", sa.Integer),
        sa.Column("gaap_issues", sa.Integer),
        sa.Column("ifrs_compliant", sa.Integer),
        sa.Column("ifrs_issues", sa.Integer),
        sa.Column("hitl_required", sa.Integer, server_default="0"),
        sa.Column("hitl_decision", sa.String(20)),
        sa.Column("hitl_approver", sa.String(100)),
        sa.Column("hitl_notes", sa.Text),
        sa.Column("hitl_at", sa.DateTime),
        sa.Column("full_results_json", sa.Text),
        sa.UniqueConstraint("company_name", "period", name="uq_company_period"),
    )

    op.create_table(
        "mem_kpi_ts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("company_name", sa.String(255), nullable=False),
        sa.Column("period", sa.String(50), nullable=False),
        sa.Column("snapshot_id", sa.Integer, sa.ForeignKey("mem_period_snapshot.id"), nullable=False),
        sa.Column("kpi_name", sa.String(100), nullable=False),
        sa.Column("value", sa.Float),
        sa.Column("unit", sa.String(20)),
        sa.Column("recorded_at", sa.DateTime),
        sa.UniqueConstraint("company_name", "period", "kpi_name", name="uq_ts_kpi"),
    )

    op.create_table(
        "mem_insights",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("company_name", sa.String(255), sa.ForeignKey("mem_company.company_name"), nullable=False),
        sa.Column("period", sa.String(50), nullable=False),
        sa.Column("insight_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("source", sa.String(100)),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("active", sa.Integer, server_default="1"),
        sa.Column("created_at", sa.DateTime),
    )


def downgrade() -> None:
    op.drop_table("mem_insights")
    op.drop_table("mem_kpi_ts")
    op.drop_table("mem_period_snapshot")
    op.drop_table("mem_company")
    op.drop_table("rag_documents")
    op.drop_table("approvals")
    op.drop_table("tasks")
