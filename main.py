"""Lepo Mesto – Občinski AI chatbot."""
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.config import Settings
from app.core.chat_service import get_reply
from app.core.db import init_db, get_sessions, get_messages, get_stats
from app.rag.search import load_knowledge

settings = Settings()
app = FastAPI(title=settings.project_name, version="1.0.0")

init_db()

kb_path = Path(__file__).parent / "knowledge.jsonl"
n = load_knowledge(kb_path)
print(f"[KB] Naloženih {n} zapisov")
print(f"[startup] Lepo Mesto AI ready")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok", "version": "v1", "bot": "lepo-mesto"}


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path("static/widget.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Lepo Mesto AI</h1>")


@app.get("/widget", response_class=HTMLResponse)
def widget():
    html_path = Path("static/widget.html")
    if not html_path.exists():
        return HTMLResponse("<h1>widget.html manjka</h1>", status_code=500)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/admin", response_class=HTMLResponse)
def admin_ui():
    html_path = Path("static/admin.html")
    if not html_path.exists():
        return HTMLResponse("<h1>admin.html manjka</h1>", status_code=500)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = get_reply(req.session_id, req.message)
    return ChatResponse(reply=reply, session_id=req.session_id)


# Admin API (zaščiten z ADMIN_TOKEN)
def _check_token(request: Request) -> bool:
    token = request.headers.get("X-Admin-Token", "") or request.query_params.get("token", "")
    return token == settings.admin_token


@app.get("/api/admin/stats")
def admin_stats(request: Request):
    if not _check_token(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return get_stats()


@app.get("/api/admin/sessions")
def admin_sessions(request: Request):
    if not _check_token(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return get_sessions()


@app.get("/api/admin/sessions/{session_id}")
def admin_session_detail(session_id: str, request: Request):
    if not _check_token(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return get_messages(session_id)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8004))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
