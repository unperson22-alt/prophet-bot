"""
prophet-bot — Пророк. Агрегатор AI-офиса.
Собирает мнения всех ботов и выдаёт взвешенный прогноз сценариев.
"""
import os
import httpx, asyncio, logging, httpx
from aiohttp import web
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from anthropic import AsyncAnthropic

# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/\\")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "").lower() in ("1", "true", "yes")


class _OllamaResult:
    def __init__(self, text):
        from types import SimpleNamespace
        self.content = [SimpleNamespace(text=text)]


def _try_ollama(messages, system=None, timeout=20.0):
    if not (OLLAMA_ENABLED and OLLAMA_HOST):
        return None
    try:
        ol_messages = []
        if system:
            ol_messages.append({"role": "system", "content": system})
        for m in messages:
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            ol_messages.append({"role": m["role"], "content": content})
        with httpx.Client(timeout=timeout) as cli:
            r = cli.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": ol_messages,
                      "stream": False, "keep_alive": "30m"},
            )
            if r.status_code != 200:
                return None
            text = r.json().get("message", {}).get("content", "")
            return _OllamaResult(text) if text else None
    except Exception as e:
        logger.info(f"Ollama unavailable, fallback to Anthropic: {type(e).__name__}: {e}")
        return None


def _anthropic_call(client, **kwargs):
    """LLM call. Tries Ollama first if enabled, falls back to Anthropic with 529 retry."""
    ol = _try_ollama(kwargs.get("messages", []), kwargs.get("system"))
    if ol is not None:
        return ol
    import time
    last_err = None
    for delay in [0, 2, 4, 8]:
        try:
            if delay:
                time.sleep(delay)
            return client.messages.create(**kwargs)
        except Exception as e:
            if "529" in str(e) or "overloaded" in str(e).lower():
                last_err = e
                continue
            raise
    raise last_err

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def transcribe_voice(file_path: str) -> str | None:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(file_path)
            audio_data = r.content
            r2 = await c.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": ("voice.ogg", audio_data, "audio/ogg")},
                data={"model": "whisper-large-v3-turbo", "language": "ru"}
            )
            return r2.json().get("text", "").strip() or None
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
YOUR_TELEGRAM_ID = int(os.environ["YOUR_TELEGRAM_ID"])
OFFICE_CHAT_ID   = os.environ.get("OFFICE_CHAT_ID", "")
LOG_BOT_URL      = os.environ.get("LOG_BOT_URL", "")
HTTP_PORT        = 8080

# Internal Railway URLs for each advisor
ADVISORS = {
    "Билли":  os.environ.get("BILLY_URL",  "http://billy-bot.railway.internal:8080"),
    "Доктор": os.environ.get("DOCTOR_URL", "http://doctor-bot.railway.internal:8080"),
    "Макс":   os.environ.get("MILLY_URL",  "http://milly-bot.railway.internal:8080"),
    "Тилли":  os.environ.get("TILLY_URL",  "http://tilly-bot.railway.internal:8080"),
}

client = AsyncAnthropic(api_key=ANTHROPIC_KEY)

RANDOM_REPLY_CHANCE_HUMAN = 0.40  # 40% на непрямые сообщения от людей
RANDOM_REPLY_CHANCE_BOT   = 0.30  # 30% на сообщения от ботов


SYSTEM_SHORT = """Ты — Пророк. Говоришь редко, но точно.

Один абзац. Прямой вердикт: да/нет + одна ключевая причина. Без вступлений, без "с одной стороны".
Если вопрос требует глубокого анализа — скажи об этом одной фразой.

Контекст: отвечаешь Владу — курьер DPD в Германии, строит бизнес на автоматизации, иногда трейдит BTC. Системный человек, ценит конкретику.

По-русски. Не используй Markdown-разметку (##, **, таблицы, ---) — пиши простым текстом."""

SYSTEM_FULL = """Ты — Пророк. Не просто суммируешь мнения — ты видишь то, что советники пропускают.

Тебе дают вопрос и ответы советников. Твоя задача — найти противоречия, выявить системный риск и дать вердикт.

Структура ответа:
🔮 Сценарий А: [если сделает X] → конкретный вероятный исход
🔮 Сценарий Б: [если сделает Y] → конкретный вероятный исход
⚖️ Вердикт: одна конкретная рекомендация. Без "зависит от ситуации".

Контекст: Влад — курьер DPD в Германии, строит бизнес на автоматизации (Make, Claude API, Telegram-боты), трейдер в паузе. Системный, амбициозный, идёт к финансовой независимости.

Прямо, без воды. По-русски. Не используй Markdown-разметку (##, **, таблицы, ---) — пиши простым текстом, для структуры используй цифры, тире и символ •."""

async def log(event: str, msg: str):
    if not LOG_BOT_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            await c.post(f"{LOG_BOT_URL}/log",
                json={"agent": "Пророк", "type": event, "message": msg})
    except Exception:
        pass

async def ask_advisor(name: str, url: str, question: str, user_id: int) -> str:
    """Ask one advisor and return their response."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{url}/task",
                json={"message": question, "user_id": user_id})
            data = r.json()
            return data.get("response", "")
    except Exception as e:
        logger.warning(f"Advisor {name} unavailable: {e}")
        return ""

async def prophesy(question: str, user_id: int, short_mode: bool = False) -> str:
    """Gather all advisor opinions and synthesize a prophecy."""
    # Ask all advisors in parallel
    tasks = {
        name: asyncio.create_task(ask_advisor(name, url, question, user_id))
        for name, url in ADVISORS.items()
    }
    responses = {}
    for name, task in tasks.items():
        try:
            answer = await asyncio.wait_for(task, timeout=20)
            if answer:
                responses[name] = answer
        except asyncio.TimeoutError:
            logger.warning(f"{name} timed out")

    if not responses:
        return "❌ Советники недоступны. Попробуй позже."

    # Build context for Prophet
    advisors_block = "\n\n".join(
        f"[{name}]: {text[:400]}" for name, text in responses.items()
    )
    prompt = (
        f"Вопрос пользователя: {question}\n\n"
        f"Мнения советников:\n{advisors_block}\n\n"
        f"Дай прогноз сценариев и вердикт."
    )

    msg = await _anthropic_call(client, 
        model="claude-sonnet-4-6",
        max_tokens=400 if short_mode else 700,
        system=SYSTEM_SHORT if short_mode else SYSTEM_FULL,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


SYSTEM_ORACLE = """Ты — Пророк. В офисной группе говоришь редко — но когда говоришь, это запоминают.

Один абзац, не больше. Реагируй на суть разговора — брось наблюдение, предупреди о риске, или дай неожиданный угол зрения. Можешь быть загадочным, но всегда конкретным. Без воды, без "интересный вопрос".

Офис: Влад (хозяин, строит автобизнес), Лук (друг со школы), и несколько ботов-коллег.

По-русски."""

async def quick_prophesy(text: str) -> str:
    """Быстрый ответ без сбора советников — для случайных ответов в группе."""
    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_ORACLE,
        messages=[{"role": "user", "content": text}]
    )
    return msg.content[0].text

# ── Telegram handler ────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != YOUR_TELEGRAM_ID:
        return
    if update.effective_chat.type in ["group", "supergroup"]:
        return
    question = update.message.text
    await log("MSG_IN", question[:80])
    await update.message.reply_text("🔮 Советуюсь с офисом...")
    answer = await prophesy(question, YOUR_TELEGRAM_ID, short_mode=False)
    await update.message.reply_text(answer, parse_mode=None)
    await log("MSG_OUT", answer[:80])

# ── HTTP endpoint (for Filly routing) ────────────────────────────────────────
async def handle_health(request):
    return web.Response(text="ok")


async def handle_task(request):
    data = await request.json()
    question = data.get("message", "")
    user_id  = data.get("user_id", YOUR_TELEGRAM_ID)
    await log("MSG_IN", f"[HTTP] {question[:80]}")
    response = await prophesy(question, user_id, short_mode=True)
    await log("MSG_OUT", response[:80])
    # Post directly to office group so Vlad sees it inline
    if OFFICE_CHAT_ID:
        try:
            from telegram import Bot as TGBot
            tg = TGBot(token=TELEGRAM_TOKEN)
            await tg.send_message(chat_id=OFFICE_CHAT_ID,
                text=f"Пророк:\n{response}", parse_mode=None)
        except Exception as e:
            logger.warning(f"Prophet group reply failed: {e}")
    return web.json_response({"status": "ok", "response": response})


# ── Group random replies ──────────────────────────────────────────────────────
async def handle_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Случайные ответы в группе: 40% на людей, 17% на ботов."""
    import random

    msg = update.message
    if not msg or not msg.text:
        return

    text     = msg.text.strip()
    is_bot   = msg.from_user.is_bot if msg.from_user else False
    txt_low  = text.lower()

    # Прямое обращение — обрабатывает Филли, не дублируем
    if any(txt_low.startswith(w) for w in ["пророк", "prophet", "@prophet"]):
        return

    chance = RANDOM_REPLY_CHANCE_BOT if is_bot else RANDOM_REPLY_CHANCE_HUMAN
    if random.random() > chance:
        return

    try:
        answer = await quick_prophesy(text)
        await msg.reply_text(f"🔮 {answer}", parse_mode=None)
        await log("RANDOM", f"{'bot' if is_bot else 'human'}: {text[:60]}")
    except Exception as e:
        logger.error(f"handle_group_message failed: {e}")

# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS, handle_group_message))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    # HTTP server
    http_app = web.Application()
    http_app.router.add_post("/task", handle_task)
    http_app.router.add_get("/health", handle_health)
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    logger.info("Prophet online 🔮")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
