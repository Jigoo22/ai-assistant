"""
Лёгкий API-сервер на FastAPI
Дашборд читает задачи отсюда, бот пишет сюда.
Запускать вместе с bot.py: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

TASKS_FILE    = Path("tasks.json")
PROJECTS_FILE = Path("projects.json")

app = FastAPI(title="AI Assistant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # в проде укажите конкретный домен дашборда
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Утилиты ───────────────────────────────────────────────────────
def load(path: Path, default):
    return json.loads(path.read_text("utf-8")) if path.exists() else default

def dump(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

# ── Схемы ─────────────────────────────────────────────────────────
class TaskIn(BaseModel):
    pid:      int
    text:     str
    priority: str = "med"   # high | med | low
    deadline: Optional[str] = None
    source:   Optional[str] = ""
    note:     Optional[str] = ""

class TaskPatch(BaseModel):
    done:     Optional[bool] = None
    priority: Optional[str] = None
    deadline: Optional[str] = None

class CommentIn(BaseModel):
    text:   str
    author: str  = "Я"
    isAi:   bool = False

class ProjectIn(BaseModel):
    name:     str
    color:    str = "var(--accent)"
    deadline: Optional[str] = None

# ── Projects ──────────────────────────────────────────────────────
@app.get("/projects")
def list_projects():
    return load(PROJECTS_FILE, [{"id":1,"name":"Общее","color":"var(--accent)"}])

@app.post("/projects", status_code=201)
def create_project(body: ProjectIn):
    projects = load(PROJECTS_FILE, [])
    new_id   = max((p["id"] for p in projects), default=0) + 1
    project  = {"id": new_id, **body.dict()}
    projects.append(project)
    dump(PROJECTS_FILE, projects)
    return project

@app.delete("/projects/{pid}")
def delete_project(pid: int):
    projects = [p for p in load(PROJECTS_FILE,[]) if p["id"] != pid]
    dump(PROJECTS_FILE, projects)
    return {"ok": True}

# ── Tasks ─────────────────────────────────────────────────────────
@app.get("/tasks")
def list_tasks(pid: Optional[int] = None, done: Optional[bool] = None):
    tasks = load(TASKS_FILE, [])
    if pid  is not None: tasks = [t for t in tasks if t.get("pid") == pid]
    if done is not None: tasks = [t for t in tasks if t.get("done") == done]
    return tasks

@app.post("/tasks", status_code=201)
def create_task(body: TaskIn):
    tasks  = load(TASKS_FILE, [])
    new_id = max((t["id"] for t in tasks), default=0) + 1
    task   = {
        "id":         new_id,
        "done":       False,
        "comments":   [],
        "created_at": datetime.now().isoformat(),
        **body.dict()
    }
    tasks.append(task)
    dump(TASKS_FILE, tasks)
    return task

@app.patch("/tasks/{task_id}")
def patch_task(task_id: int, body: TaskPatch):
    tasks = load(TASKS_FILE, [])
    for t in tasks:
        if t["id"] == task_id:
            patch = {k: v for k, v in body.dict().items() if v is not None}
            t.update(patch)
            dump(TASKS_FILE, tasks)
            return t
    raise HTTPException(404, "Task not found")

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    tasks = [t for t in load(TASKS_FILE,[]) if t["id"] != task_id]
    dump(TASKS_FILE, tasks)
    return {"ok": True}

# ── Comments ──────────────────────────────────────────────────────
@app.get("/tasks/{task_id}/comments")
def list_comments(task_id: int):
    tasks = load(TASKS_FILE, [])
    task  = next((t for t in tasks if t["id"] == task_id), None)
    if not task: raise HTTPException(404, "Task not found")
    return task.get("comments", [])

@app.post("/tasks/{task_id}/comments", status_code=201)
def add_comment(task_id: int, body: CommentIn):
    tasks = load(TASKS_FILE, [])
    task  = next((t for t in tasks if t["id"] == task_id), None)
    if not task: raise HTTPException(404, "Task not found")
    comment = {
        "id":   max((c["id"] for c in task.get("comments",[])), default=0) + 1,
        "time": datetime.now().strftime("%H:%M"),
        **body.dict()
    }
    task.setdefault("comments", []).append(comment)
    dump(TASKS_FILE, tasks)
    return comment

# ── Stats ─────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    tasks = load(TASKS_FILE, [])
    return {
        "total":    len(tasks),
        "active":   sum(1 for t in tasks if not t.get("done")),
        "done":     sum(1 for t in tasks if t.get("done")),
        "high":     sum(1 for t in tasks if t.get("priority")=="high" and not t.get("done")),
        "ai":       sum(1 for t in tasks if t.get("source")=="ai"),
    }
