"""
ExperimentMind FastAPI Backend
Run with: uvicorn api.main:app --reload --port 8000
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import shutil
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional

from db.schema import (
    init_db, get_session,
    Experiment, Metric, AttentionItem, DigestLog
)
from tools.analysis_tools import (
    query_all_runs, compute_leaderboard,
    get_experiment_summary
)
from tools.logging_tools import query_recent_runs


# ── STARTUP ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs("experiments/queue",   exist_ok=True)
    os.makedirs("experiments/running", exist_ok=True)
    os.makedirs("experiments/done",    exist_ok=True)
    os.makedirs("experiments/failed",  exist_ok=True)
    os.makedirs("digests",             exist_ok=True)
    print("[API] ExperimentMind backend started")
    yield
    print("[API] Shutting down")


app = FastAPI(
    title       = "ExperimentMind API",
    description = "Autonomous ML Experiment Lifecycle Manager",
    version     = "1.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── MODELS ───────────────────────────────────────────────────

class ConfigSubmit(BaseModel):
    yaml_content: str

class ApproveRequest(BaseModel):
    action: str = "approve"  # approve | reject


# ── BACKGROUND PIPELINE ──────────────────────────────────────

active_pipelines = {}

def run_pipeline_background(config_path: str, run_id: str):
    """Run in background thread."""
    try:
        from agents.orchestrator import run_full_pipeline
        active_pipelines[run_id] = "running"
        result = run_full_pipeline(config_path)
        active_pipelines[run_id] = "completed" if result.get("success") else "failed"
    except Exception as e:
        active_pipelines[run_id] = f"error: {str(e)}"


# ── ENDPOINTS ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "ExperimentMind API",
        "version": "1.0.0",
        "status":  "online",
        "docs":    "/docs"
    }


@app.get("/status")
def get_status():
    """System status — queue counts, active pipelines, last digest."""
    queue   = [f for f in os.listdir("experiments/queue")   if f.endswith('.yaml')]
    running = [f for f in os.listdir("experiments/running") if f.endswith('.yaml')]
    done    = [f for f in os.listdir("experiments/done")    if f.endswith('.yaml')]
    failed  = [f for f in os.listdir("experiments/failed")  if f.endswith('.yaml')]

    last_digest = None
    if os.path.exists("digests"):
        digests = sorted(os.listdir("digests"))
        if digests:
            last_digest = digests[-1]

    return {
        "status":           "online",
        "queue_count":      len(queue),
        "running_count":    len(running),
        "completed_count":  len(done),
        "failed_count":     len(failed),
        "active_pipelines": active_pipelines,
        "last_digest":      last_digest,
        "timestamp":        datetime.now().isoformat()
    }


@app.post("/experiments")
async def submit_experiment(
    config: ConfigSubmit,
    background_tasks: BackgroundTasks
):
    """Submit a new experiment config. Triggers full pipeline in background."""
    try:
        parsed = yaml.safe_load(config.yaml_content)
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Invalid YAML — must be a mapping")

        experiment_name = parsed.get("experiment_name", "unnamed")
        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename        = f"{timestamp}_{experiment_name}.yaml"
        config_path     = os.path.join("experiments/queue", filename)

        with open(config_path, 'w') as f:
            f.write(config.yaml_content)

        run_id = f"{experiment_name}_{timestamp}"
        background_tasks.add_task(
            run_pipeline_background, config_path, run_id
        )

        return {
            "success":         True,
            "run_id":          run_id,
            "config_path":     config_path,
            "experiment_name": experiment_name,
            "message":         "Pipeline started in background"
        }
    except yaml.YAMLError as e:
        raise HTTPException(400, f"YAML parse error: {str(e)}")


@app.get("/experiments")
def list_experiments(limit: int = 20):
    """List recent experiment runs from the database."""
    result = query_recent_runs(limit=limit)
    return result


@app.get("/experiments/{run_id}")
def get_experiment(run_id: str):
    """Get a single experiment run with all metrics."""
    session = get_session()
    run = session.query(Experiment).filter(
        Experiment.run_id == run_id
    ).first()

    if not run:
        session.close()
        raise HTTPException(404, f"Run not found: {run_id}")

    metrics = session.query(Metric).filter(
        Metric.run_id == run_id,
        Metric.step == -1
    ).all()

    result = {
        "run_id":          run.run_id,
        "experiment_name": run.experiment_name,
        "status":          run.status,
        "primary_metric":  run.primary_metric,
        "config_path":     run.config_path,
        "created_at":      run.created_at.isoformat() if run.created_at else None,
        "completed_at":    run.completed_at.isoformat() if run.completed_at else None,
        "metrics":         {m.metric_name: m.value for m in metrics}
    }
    session.close()
    return result


@app.get("/leaderboard/{experiment_name}")
def get_leaderboard(experiment_name: str):
    """Get the ranked leaderboard for an experiment."""
    runs_data = query_all_runs(experiment_name)
    if not runs_data.get("success") or not runs_data.get("runs"):
        raise HTTPException(404, f"No runs found for: {experiment_name}")

    runs           = runs_data["runs"]
    primary_metric = runs[0].get("primary_metric", "val_accuracy")
    leaderboard    = compute_leaderboard(runs, primary_metric)
    summary        = get_experiment_summary(experiment_name)

    return {
        "experiment_name": experiment_name,
        "primary_metric":  primary_metric,
        "leaderboard":     leaderboard,
        "summary":         summary
    }


@app.get("/digest/latest")
def get_latest_digest():
    """Get the most recent digest content."""
    session = get_session()
    digest  = session.query(DigestLog)\
                     .order_by(DigestLog.created_at.desc())\
                     .first()
    session.close()

    if not digest:
        # Check file system
        if os.path.exists("digests"):
            files = sorted(
                [f for f in os.listdir("digests") if f.endswith('.md')]
            )
            if files:
                with open(os.path.join("digests", files[-1])) as f:
                    content = f.read()
                return {"found": True, "source": "file",
                        "content": content, "filename": files[-1]}

        raise HTTPException(404, "No digests found yet")

    return {
        "found":      True,
        "source":     "database",
        "date":       digest.digest_date,
        "content":    digest.content_md,
        "created_at": digest.created_at.isoformat() if digest.created_at else None
    }


@app.get("/attention")
def get_attention_items():
    """Get all unresolved attention items."""
    session = get_session()
    items   = session.query(AttentionItem)\
                     .filter(AttentionItem.resolved == False)\
                     .order_by(AttentionItem.created_at.desc())\
                     .all()

    result = [{
        "id":          item.id,
        "type":        item.item_type,
        "run_id":      item.run_id,
        "title":       item.title,
        "description": item.description,
        "created_at":  item.created_at.isoformat() if item.created_at else None
    } for item in items]

    session.close()
    return {"count": len(result), "items": result}


@app.post("/attention/{item_id}/resolve")
def resolve_attention_item(item_id: int, request: ApproveRequest):
    """Resolve an attention item — approve or reject."""
    session = get_session()
    item    = session.query(AttentionItem).filter(
        AttentionItem.id == item_id
    ).first()

    if not item:
        session.close()
        raise HTTPException(404, f"Attention item not found: {item_id}")

    item.resolved = True
    session.commit()
    session.close()

    return {
        "success": True,
        "item_id": item_id,
        "action":  request.action,
        "message": f"Item {item_id} marked as resolved ({request.action})"
    }


@app.post("/digest/generate/{experiment_name}")
def trigger_digest(experiment_name: str, background_tasks: BackgroundTasks):
    """Manually trigger a digest for an experiment."""
    def run_digest():
        from agents.reporter import generate_and_send_digest
        generate_and_send_digest(experiment_name)

    background_tasks.add_task(run_digest)
    return {
        "success": True,
        "message": f"Digest generation started for: {experiment_name}"
    }