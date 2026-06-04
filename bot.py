"""
AI Personal Assistant Bot v2.1
"""
import os, json, logging, tempfile
from datetime import datetime
from pathlib import Path

import anthropic
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

TELEGRAM_TOKEN    = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ALLOWED_USER_ID   = int(os.environ.get("ALLOWED_USER_ID", 0))
TASKS_FILE        = Path(os.environ.get("TASKS_FILE", "tasks.json"))
PROJECTS_FILE     = Path(os.environ.get("PROJECTS_FILE", "projects.json"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Storage ───────────────────────────────────────────────────────
def load_json(path, default):
    return json.loads(path.read_text("utf-8")) if path.exists() else default

def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

def get_tasks():
    return load_json(TASKS_FILE, [])

def get_projects():
    return load_json(PROJECTS_FILE, [{"id": 0, "name": "Общее", "color": "var(--blue)"}])

def save_task(task: dict):
    tasks = get_tasks()
    new_id = max((t["id"] for t in tasks), default=0) + 1
    task.update({"id": new_id, "done": False, "comments": [], "created_at": datetime.now().isoformat()})
    tasks.append(task)
    save_json(TASKS_FILE, tasks)
    return task

def save_project(name: str, color: str = "var(--accent)"):
    projects = get_projects()
    new_id = max((p["id"] for p in projects), default=0) + 1
    project = {"id": new_id, "name": name, "color": color}
    projects.append(project)
    save_json(PROJECTS_FILE, projects)
    return project

def delete_project(pid: int):
    projects = [p for p in get_projects() if p["id"] != pid]
    save_json(PROJECTS_FILE, projects)

# ── Pending tasks waiting for project selection ───────────────────
pending: dict[int, dict] = {}

def is_allowed(update: Update) -> bool:
    if not ALLOWED_USER_ID:
        return True
    uid = update.effective_user.id if update.effective_user else None
    return uid == ALLOWED_USER_ID

# ── Claude ────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    projects = get_projects()
    proj_list = "\n".join(f'  {p["id"]}: "{p["name"]}"' for p in projects)
    return f"""Ты — личный AI-помощник. Анализируешь сообщения и извлекаешь задачи.

Доступные проекты:
{proj_list}

Отвечай ТОЛЬКО валидным JSON без markdown:
{{
  "tasks": [
    {{
      "text": "краткое название задачи до 80 символов",
      "priority": "high|med|low",
      "deadline": "YYYY-MM-DD или null",
      "source": "msg|call|audio|photo",
      "pid": <число — id проекта, или null если не ясно>,
      "note": "1 предложение контекста"
    }}
  ],
  "summary": "одно предложение о чём сообщение"
}}

Правила:
- priority=high если: срочно, сегодня, завтра, горит, важно
- pid=null если не уверен в проекте
- Только реальные конкретные задачи
- Отвечай на русском"""

async def analyze_text(text: str, source: str = "msg") -> dict:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": f"[Источник: {source}]\n\n{text}"}]
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

async def analyze_audio(file_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        import openai
        oai = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        with open(tmp_path, "rb") as f:
            transcript = oai.audio.transcriptions.create(model="whisper-1", file=f)
        text = transcript.text
        result = await analyze_text(text, "audio")
        result["transcript"] = text
        return result
    except Exception as e:
        log.error(f"Ошибка транскрипции: {e}")
        return {"tasks": [], "summary": f"Ошибка аудио: {e}"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)

async def analyze_image(file_bytes: bytes) -> dict:
    import base64
    b64 = base64.standard_b64encode(file_bytes).decode()
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": "[Источник: photo] Извлеки задачи из изображения"}
        ]}]
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Formatting ────────────────────────────────────────────────────
PRIORITY_EMOJI = {"high": "🔴", "med": "🟡", "low": "🔵"}

def format_task_line(t: dict) -> str:
    p = PRIORITY_EMOJI.get(t.get("priority", "med"), "🟡")
    dl = f" · до {t['deadline']}" if t.get("deadline") else ""
    return f"{p} *{t['text']}*{dl}"

def project_keyboard() -> InlineKeyboardMarkup:
    projects = get_projects()
    buttons = []
    row = []
    for p in projects:
        row.append(InlineKeyboardButton(p["name"], callback_data=f"proj:{p['id']}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("➕ Новый проект", callback_data="proj:new")])
    return InlineKeyboardMarkup(buttons)

async def process_result(result: dict, msg, source: str = "msg"):
    tasks = result.get("tasks", [])
    summary = result.get("summary", "")

    if not tasks:
        text = f"📋 _{summary}_\n\nЗадач не найдено." if summary else "Задач не найдено."
        await msg.reply_text(text, parse_mode="Markdown")
        return

    saved = [t for t in tasks if t.get("pid") is not None]
    unclear = [t for t in tasks if t.get("pid") is None]

    if saved:
        lines = [f"📋 _{summary}_\n"] if summary else []
        for t in saved:
            save_task(t)
            proj = next((p["name"] for p in get_projects() if p["id"] == t.get("pid")), "Общее")
            lines.append(format_task_line(t))
            lines.append(f"   📁 {proj}\n")
        await msg.reply_text("\n".join(lines), parse_mode="Markdown")

    for t in unclear:
        t["source"] = source
        sent = await msg.reply_text(
            f"🤔 В какой проект добавить?\n\n{format_task_line(t)}",
            parse_mode="Markdown",
            reply_markup=project_keyboard()
        )
        pending[sent.message_id] = t

# ── Handlers ──────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    await update.message.reply_text(
        f"👋 *AI-помощник запущен!*\n\n"
        f"Твой User ID: `{update.effective_user.id}`\n\n"
        f"Пересылай сообщения из WhatsApp, голосовые, фото — "
        f"я создам задачи и уточню проект если не пойму.\n\n"
        f"/tasks — активные задачи\n"
        f"/projects — список проектов\n"
        f"/addproject Название — добавить проект\n"
        f"/delproject ID — удалить проект",
        parse_mode="Markdown"
    )

async def cmd_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    tasks = [t for t in get_tasks() if not t.get("done")]
    if not tasks:
        await update.message.reply_text("✅ Нет активных задач!")
        return
    projects = {p["id"]: p["name"] for p in get_projects()}
    lines = [f"📋 *Активные задачи ({len(tasks)}):*\n"]
    cur_pid = -1
    for t in sorted(tasks, key=lambda x: x.get("pid") or 0):
        pid = t.get("pid") or 0
        if pid != cur_pid:
            lines.append(f"\n📁 *{projects.get(pid, 'Общее')}*")
            cur_pid = pid
        p = PRIORITY_EMOJI.get(t.get("priority", "med"), "🟡")
        lines.append(f"  {p} {t['text']}")
    await update.message.reply_text("\n".join(lines[:40]), parse_mode="Markdown")

async def cmd_projects(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    projects = get_projects()
    tasks = get_tasks()
    lines = ["📁 *Проекты:*\n"]
    for p in projects:
        count = sum(1 for t in tasks if t.get("pid") == p["id"] and not t.get("done"))
        lines.append(f"• `{p['id']}` *{p['name']}* — {count} задач")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_addproject(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text("Используйте: `/addproject Название`", parse_mode="Markdown")
        return
    name = " ".join(ctx.args)
    project = save_project(name)
    await update.message.reply_text(f"✅ Проект *{name}* создан (id: `{project['id']}`)", parse_mode="Markdown")

async def cmd_delproject(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    if not ctx.args:
        await update.message.reply_text("Используйте: `/delproject ID`\nID смотрите в /projects", parse_mode="Markdown")
        return
    try:
        pid = int(ctx.args[0])
        name = next((p["name"] for p in get_projects() if p["id"] == pid), None)
        if name is None:
            await update.message.reply_text("❌ Проект не найден")
            return
        delete_project(pid)
        await update.message.reply_text(f"🗑 Проект *{name}* удалён", parse_mode="Markdown")
    except ValueError:
        await update.message.reply_text("❌ ID должен быть числом")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return

    # Если ждём название нового проекта
    if ctx.user_data.get("awaiting_project_for"):
        await handle_new_project_name(update, ctx)
        return

    msg = update.message
    text = msg.text or msg.caption or ""

    # Определяем отправителя пересланного сообщения
    try:
        origin = msg.forward_origin
        if origin:
            name = (getattr(origin, "sender_name", None)
                    or getattr(getattr(origin, "sender_user", None), "full_name", None)
                    or "")
            if name:
                text = f"[Переслано от {name}]\n{text}"
    except AttributeError:
        pass

    if not text.strip():
        return

    thinking = await msg.reply_text("🤔 Анализирую...")
    try:
        result = await analyze_text(text, "msg")
        await thinking.delete()
        await process_result(result, msg, "msg")
    except Exception as e:
        log.error(e)
        await thinking.edit_text(f"❌ Ошибка: {e}")

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    msg = update.message
    voice = msg.voice or msg.audio
    file = await voice.get_file()
    thinking = await msg.reply_text("🎤 Транскрибирую...")
    try:
        file_bytes = await file.download_as_bytearray()
        result = await analyze_audio(bytes(file_bytes))
        if result.get("transcript"):
            await msg.reply_text(f"📝 _{result['transcript']}_", parse_mode="Markdown")
        await thinking.delete()
        await process_result(result, msg, "audio")
    except Exception as e:
        log.error(e)
        await thinking.edit_text(f"❌ Ошибка: {e}")

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    msg = update.message
    file = await msg.photo[-1].get_file()
    thinking = await msg.reply_text("🖼 Анализирую фото...")
    try:
        file_bytes = await file.download_as_bytearray()
        result = await analyze_image(bytes(file_bytes))
        await thinking.delete()
        await process_result(result, msg, "photo")
    except Exception as e:
        log.error(e)
        await thinking.edit_text(f"❌ Ошибка: {e}")

async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    msg = update.message
    doc = msg.document
    mime = doc.mime_type or ""
    name = doc.file_name or ""
    if "audio" in mime or name.endswith((".ogg", ".mp3", ".m4a", ".opus")):
        file = await doc.get_file()
        thinking = await msg.reply_text("🎤 Обрабатываю аудио...")
        try:
            file_bytes = await file.download_as_bytearray()
            result = await analyze_audio(bytes(file_bytes))
            if result.get("transcript"):
                await msg.reply_text(f"📝 _{result['transcript']}_", parse_mode="Markdown")
            await thinking.delete()
            await process_result(result, msg, "audio")
        except Exception as e:
            await thinking.edit_text(f"❌ Ошибка: {e}")
    else:
        await msg.reply_text("📎 Пришли текст, голосовое или фото.")

async def handle_project_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not is_allowed(update): return

    data = query.data
    msg_id = query.message.message_id
    task = pending.get(msg_id)

    if not task:
        await query.edit_message_text("⚠️ Задача устарела, отправьте сообщение снова.")
        return

    if data == "proj:new":
        ctx.user_data["awaiting_project_for"] = msg_id
        await query.edit_message_text(
            f"Введите название нового проекта для задачи:\n\n_{task['text']}_",
            parse_mode="Markdown"
        )
        return

    pid = int(data.split(":")[1])
    task["pid"] = pid
    save_task(task)
    pending.pop(msg_id, None)

    proj_name = next((p["name"] for p in get_projects() if p["id"] == pid), "Общее")
    p_emoji = PRIORITY_EMOJI.get(task.get("priority", "med"), "🟡")
    await query.edit_message_text(
        f"✅ Сохранено!\n\n{p_emoji} *{task['text']}*\n📁 {proj_name}",
        parse_mode="Markdown"
    )

async def handle_new_project_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg_id = ctx.user_data.get("awaiting_project_for")
    name = update.message.text.strip()
    project = save_project(name)
    task = pending.get(msg_id)
    if task:
        task["pid"] = project["id"]
        save_task(task)
        pending.pop(msg_id, None)
    ctx.user_data.pop("awaiting_project_for", None)
    p_emoji = PRIORITY_EMOJI.get(task.get("priority", "med"), "🟡") if task else "🟡"
    task_text = task["text"] if task else ""
    await update.message.reply_text(
        f"✅ Проект *{name}* создан и задача сохранена!\n\n{p_emoji} *{task_text}*",
        parse_mode="Markdown"
    )

# ── Main ──────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",      cmd_start))
    app.add_handler(CommandHandler("tasks",      cmd_tasks))
    app.add_handler(CommandHandler("projects",   cmd_projects))
    app.add_handler(CommandHandler("addproject", cmd_addproject))
    app.add_handler(CommandHandler("delproject", cmd_delproject))
    app.add_handler(CallbackQueryHandler(handle_project_choice, pattern=r"^proj:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO,   handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL,            handle_document))
    log.info("Бот v2.1 запущен.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
