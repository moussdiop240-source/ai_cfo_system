import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ...database.models import Task
from ...database.session import SessionLocal

router = APIRouter(prefix="/tasks", tags=["stream"])


async def _task_event_generator(task_id: str):
    """Server-Sent Events generator — opens a fresh session per poll to avoid
    holding a connection open for the full 5-minute window."""
    max_polls = 300  # 5 minutes max
    payload = {}
    for _ in range(max_polls):
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                yield f"data: {json.dumps({'error': 'task_not_found'})}\n\n"
                return

            payload = {
                "task_id": task_id,
                "status": task.status,
                "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            }

            if task.status in ("complete", "error"):
                payload["final_report"] = task.final_report
                payload["errors"] = task.errors or []
        finally:
            db.close()

        yield f"data: {json.dumps(payload)}\n\n"
        if payload.get("status") in ("complete", "error"):
            return
        await asyncio.sleep(1)

    yield f"data: {json.dumps({'error': 'stream_timeout'})}\n\n"


@router.get("/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE endpoint — streams task progress until completion."""
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    finally:
        db.close()

    return StreamingResponse(
        _task_event_generator(task_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
