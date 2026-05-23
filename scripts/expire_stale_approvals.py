"""
HITL approval timeout script — auto-rejects approvals pending > TIMEOUT_HOURS.

Run via Kubernetes CronJob (k8s/hitl-timeout-cronjob.yaml) or manually:
    python scripts/expire_stale_approvals.py

Exit code 0 on success, 1 on error.
"""
import logging
import os
import sys
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("expire_approvals")

TIMEOUT_HOURS = int(os.getenv("HITL_TIMEOUT_HOURS", "48"))


def expire_stale(dry_run: bool = False) -> int:
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from backend.database.session import get_db
        from backend.database.models import Approval, Task
    except ImportError as exc:
        log.error("Import failed: %s", exc)
        return 1

    cutoff = datetime.utcnow() - timedelta(hours=TIMEOUT_HOURS)
    log.info("Expiring approvals pending since before %s (timeout=%dh)", cutoff.isoformat(), TIMEOUT_HOURS)

    try:
        with get_db() as db:
            stale = (
                db.query(Approval)
                .filter(Approval.status == "pending", Approval.created_at < cutoff)
                .all()
            )
            log.info("Found %d stale pending approval(s)", len(stale))

            if not dry_run:
                for approval in stale:
                    approval.status = "auto_rejected"
                    approval.decision = "rejected"
                    approval.feedback = (
                        f"Auto-rejected: no reviewer response within {TIMEOUT_HOURS}h."
                    )
                    task = db.query(Task).filter(Task.id == approval.task_id).first()
                    if task:
                        task.status = "error"
                        task.errors = ["HITL approval timed out — auto-rejected"]
                    log.info("Auto-rejected approval %s (task %s)", approval.id, approval.task_id)
            else:
                for approval in stale:
                    log.info("[DRY RUN] Would reject approval %s (task %s)", approval.id, approval.task_id)

        return 0
    except Exception as exc:
        log.error("Failed to expire approvals: %s", exc)
        return 1


if __name__ == "__main__":
    dry_run = os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")
    sys.exit(expire_stale(dry_run=dry_run))
