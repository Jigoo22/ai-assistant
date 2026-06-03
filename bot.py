"""
AI Personal Assistant Bot
Получает сообщения из Telegram (переброшенные из WhatsApp через MacroDroid),
анализирует через Claude, создаёт задачи в tasks.json
"""

import os
import json
import logging
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import anthropic
from telegram import Update, Message
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

# ── Конфиг ────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]       # токен от @BotFather
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]   # ключ Anthropic
ALLOWED_USER_ID  = int(os.environ.get("ALLOWED_USER_ID", 0))  # ваш Telegram user_id
TASKS_FILE       = Path(os.environ.get("TASKS_FILE", "tasks.json"))
PROJECTS_FILE    = Path(os.environ.get("PROJECTS_FILE", "projects.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Хранилище задач ───────────────────────────────────────────────
def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def get_tasks():   return load_json(TASKS_FILE,    [])
def get_projects():return load_json(PROJECTS_FILE, [
    {"id": 1, "name": "Общее", "color": "var(--accent)"},
])

def save_task(task: dict):
    tasks = get_tasks()
    task["id"] = max((t["id"] for t in tasks), default=0) + 1
    task["done"] = False
    task["comments"] = []
    task["created_at"] = datetime.now().isoformat()
    tasks.append(task)
    save_json(TASKS_FILE, tasks)
    return task

# ── Проверка доступа ──────────────────────────────────────────────
def is_allowed(update: Update) -> bool:
    if not ALLOWED_USER_ID:
        return True   # если не задан — разрешаем всем (только для теста!)
    return update.effective_user.id == ALLOWED_USER_ID

# ── Claude: анализ текста ─────────────────────────────────────────
SYSTEM_PROMPT = """Ты — личный AI-помощник. Анализируешь сообщения пользователя
(переписку, голосовые, заметки) и извлекаешь задачи, договорённости и напоминания.

Отвечай ТОЛЬКО JSON в формате:
{
  "tasks": [
    {
      "text": "краткое название задачи",
      "priority": "high|med|low",
      "deadline": "YYYY-MM-DD или null",
      "source": "msg|call|audio|photo",
      "pid": 1,
      "note": "краткий контекст откуда задача (1 предложение)"
    }
  ],
  "summary": "одно предложение — о чём было сообщение"
}

Правила:
- priority=high если есть слова: срочно, сегодня, завтра, горит, важно, асап
- Извлекай только реальные задачи — не общие фразы
- Если задач нет — tasks: []
- deadline только если явно указана дата или "до пятницы" / "на следующей неделе" и т.д.
- Всегда отвечай на русском"""

async def analyze_text(text: str, source: str = "msg") -> dict:
    """Отправляет текст в Claude, получает структурированные задачи."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"[Источник: {source}]\n\n{text}"}]
    )
    raw = response.content[0].text.strip()
    # убираем markdown-блоки если Claude завернул
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

async def analyze_audio(file_bytes: bytes, source: str = "audio") -> dict:
    """Транскрибирует аудио через Whisper, затем анализирует текст."""
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        import openai
        oai = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        with open(tmp_path, "rb") as f:
            transcript = oai.audio.transcriptions.create(model="whisper-1", file=f)
        text = transcript.text
        log.info(f"Транскрипция: {text[:100]}...")
        result = await analyze_text(text, source)
        result["transcript"] = text
        return result
    except Exception as e:
        log.error(f"Ошибка транскрипции: {e}")
        return {"tasks": [], "summary": f"Ошибка обработки аудио: {e}", "transcript": ""}
    finally:
        Path(tmp_path).unlink(missing_ok=True)

async def analyze_image(file_bytes: bytes) -> dict:
    """Анализирует изображение через Claude Vision."""
    import base64
    b64 = base64.standard_b64encode(file_bytes).decode()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text",  "text": "[Источник: photo] Извлеки задачи из этого изображения (скриншот переписки, документ, заметка и т.д.)"}
            ]
        }]
    )
    raw = response.content[0].text.strip().replace("```json","").replace("```","").strip()
    return json.loads(raw)

# ── Форматирование ответа ─────────────────────────────────────────
PRIORITY_EMOJI = {"high": "🔴", "med": "🟡", "low": "🔵"}
SOURCE_LABEL   = {"msg": "💬 Сообщение", "call": "📞 Звонок",
                  "audio": "🎤 Голосовое", "photo": "🖼 Фото"}

def format_tasks_reply(result: dict) -> str:
    tasks   = result.get("tasks", [])
    summary = result.get("summary", "")
    lines   = [f"📋 *{summary}*\n"] if summary else []

    if not tasks:
        lines.append("Задач не найдено.")
        return "\n".join(lines)

    lines.append(f"Создано задач: *{len(tasks)}*\n")
    for t in tasks:
        p = PRIORITY_EMOJI.get(t.get("priority","med"), "🟡")
        dl = f" · до {t['deadline']}" if t.get("deadline") else ""
        lines.append(f"{p} *{t['text']}*{dl}")
        if t.get("note"):
            lines.append(f"   _{t['note']}_")
    return "\n".join(lines)

# ── Хэндлеры ─────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    uid = update.effective_user.id
    await update.message.reply_text(
        f"👋 AI-помощник запущен!\n\n"
        f"Твой User ID: `{uid}`\n\n"
        f"Пересылай мне сообщения из WhatsApp, голосовые, фото — "
        f"я буду создавать задачи автоматически.\n\n"
        f"/tasks — список активных задач\n"
        f"/projects — список проектов",
        parse_mode="Markdown"
    )

async def cmd_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    tasks = [t for t in get_tasks() if not t.get("done")]
    if not tasks:
        await update.message.reply_text("✅ Нет активных задач!")
        return
    lines = [f"📋 *Активные задачи ({len(tasks)}):*\n"]
    for t in tasks[:15]:  # максимум 15 в сообщении
        p = PRIORITY_EMOJI.get(t.get("priority","med"),"🟡")
        lines.append(f"{p} {t['text']}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def cmd_projects(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): return
    projects = get_projects()
    tasks    = get_tasks()
    lines    = ["📁 *Проекты:*\n"]
    for p in projects:
        count = sum(1 for t in tasks if t.get("pid")==p["id"] and not t.get("done"))
        lines.append(f"• *{p['name']}* — {count} задач")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Текстовое сообщение или пересланное из WA."""
    if not is_allowed(update): return
    msg   = update.message
    text  = msg.text or msg.caption or ""
    # Для пересланных — добавляем имя отправителя если есть
    if msg.forward_from:
        name = msg.forward_from.full_name
        text = f"[Переслано от {name}]\n{text}"
    elif msg.forward_sender_name:
        text = f"[Переслано от {msg.forward_sender_name}]\n{text}"
    if not text.strip():
        return

    thinking = await msg.reply_text("🤔 Анализирую...")
    try:
        result = await analyze_text(text, "msg")
        for task in result.get("tasks", []):
            save_task(task)
        await thinking.edit_text(format_tasks_reply(result), parse_mode="Markdown")
    except Exception as e:
        log.error(e)
        await thinking.edit_text(f"❌ Ошибка: {e}")

async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Голосовое сообщение (войс из WA или Telegram)."""
    if not is_allowed(update): return
    msg  = update.message
    file = await (msg.voice or msg.audio).get_file()

    thinking = await msg.reply_text("🎤 Транскрибирую аудио...")
    try:
        file_bytes = await file.download_as_bytearray()
        result     = await analyze_audio(bytes(file_bytes), "audio")
        if result.get("transcript"):
            await msg.reply_text(f"📝 Текст: _{result['transcript']}_", parse_mode="Markdown")
        for task in result.get("tasks", []):
            save_task(task)
        await thinking.edit_text(format_tasks_reply(result), parse_mode="Markdown")
    except Exception as e:
        log.error(e)
        await thinking.edit_text(f"❌ Ошибка: {e}")

async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Фото — скриншот переписки, документ, фото доски."""
    if not is_allowed(update): return
    msg  = update.message
    photo = msg.photo[-1]  # берём наибольшее разрешение
    file  = await photo.get_file()

    thinking = await msg.reply_text("🖼 Анализирую изображение...")
    try:
        file_bytes = await file.download_as_bytearray()
        result     = await analyze_image(bytes(file_bytes))
        for task in result.get("tasks", []):
            save_task(task)
        await thinking.edit_text(format_tasks_reply(result), parse_mode="Markdown")
    except Exception as e:
        log.error(e)
        await thinking.edit_text(f"❌ Ошибка: {e}")

async def handle_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Документ (PDF, аудиофайл из WA приходит как document)."""
    if not is_allowed(update): return
    msg  = update.message
    doc  = msg.document
    mime = doc.mime_type or ""

    if "audio" in mime or doc.file_name.endswith((".ogg",".mp3",".m4a",".opus")):
        # Аудиофайл из WhatsApp приходит как document
        file       = await doc.get_file()
        thinking   = await msg.reply_text("🎤 Обрабатываю аудио из WA...")
        file_bytes = await file.download_as_bytearray()
        try:
            result = await analyze_audio(bytes(file_bytes), "audio")
            if result.get("transcript"):
                await msg.reply_text(f"📝 Текст: _{result['transcript']}_", parse_mode="Markdown")
            for task in result.get("tasks", []):
                save_task(task)
            await thinking.edit_text(format_tasks_reply(result), parse_mode="Markdown")
        except Exception as e:
            await thinking.edit_text(f"❌ Ошибка: {e}")
    else:
        await msg.reply_text("📎 Документы пока не поддерживаются. Пришли текст или изображение.")

# ── Запуск ────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("tasks",    cmd_tasks))
    app.add_handler(CommandHandler("projects", cmd_projects))

    # текст и пересылки
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    # голосовые
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO,   handle_voice))
    # фото
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    # документы (аудио из WA)
    app.add_handler(MessageHandler(filters.Document.ALL,            handle_document))

    log.info("Бот запущен. Ожидаю сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
