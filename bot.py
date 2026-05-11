"""
prophet-bot — Пророк. Агрегатор AI-офиса.
Собирает мнения всех ботов и выдаёт взвешенный прогноз сценариев.
"""
import os, asyncio, logging, httpx
from aiohttp import web
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from anthropic import AsyncAnthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

SYSTEM_SHORT = """Ты — Пророк. Один абзац максимум.
Прямой вердикт: да/нет + одна ключевая причина.
Если нужен полный анализ — пусть пишет напрямую. По-русски."""

SYSTEM_FULL = """Ты — Пророк. Синтезируешь мнения советников и выдаёшь прогноз сценариев.
Тебе дают вопрос и ответы каждого советника.
Структура:
🔮 Сценарий А: [если сделает X] → вероятный исход
🔮 Сценарий Б: [если сделает Y] → вероятный исход
⚖️ Вердикт: конкретная рекомендация одним абзацем.
Прямо, без воды. По-русски."""

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

    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150 if short_mode else 700,
        system=SYSTEM_SHORT if short_mode else SYSTEM_FULL,
        messages=[{"role": "user", "content": prompt}]
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

# ── Main ────────────────────────────────────────────────────────────────────
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    # HTTP server
    http_app = web.Application()
    http_app.router.add_post("/task", handle_task)
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    logger.info("Prophet online 🔮")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
