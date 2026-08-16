"""
prophet-bot — Пророк. Агрегатор AI-офиса.
Собирает мнения всех ботов и выдаёт взвешенный прогноз сценариев.
"""
import os
import httpx, asyncio, logging
import hashlib
from aiohttp import web
from ai_office_shared.shared.auth import office_auth_middleware, office_headers
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
try:
    from telegram.ext import MessageReactionHandler
    HAS_REACTION_HANDLER = True
except ImportError:
    HAS_REACTION_HANDLER = False
from anthropic import AsyncAnthropic, APIError
from ai_office_shared.shared.logging import log_event
from ai_office_shared.shared import banter as _banter
from ai_office_shared.shared import group_history as _ghist
from ai_office_shared.shared.identity import roster_prompt

# ── Routing inline (no shared-lib dependency) ───────────────────────────────
async def forward_to_filly(message: str, user_id: int, reply_bot: str,
                           reply_chat_id: int, group_ctx: str = "") -> bool:
    _filly = os.environ.get("FILLY_URL", "https://filly-bot-production.up.railway.app").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30.0) as _c:
            _r = await _c.post(f"{_filly}/task", json={
                "message": message, "user_id": user_id,
                "reply_bot": reply_bot, "reply_chat_id": reply_chat_id,
                "group_ctx": group_ctx,
            }, headers=office_headers())
            return _r.status_code == 200
    except Exception:
        return False

def is_routed(data: dict) -> bool:
    return data.get("source", "").upper() in ("ФИЛЛИ", "FILLY", "DISPATCHER")
# ─────────────────────────────────────────────────────────────────────────────
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
HTTP_SECRET      = os.environ.get("HTTP_SECRET", "")  # X-Secret-Token для /send

# Feedback loop (реакции 👍/👎 → office:quality:{bot})
BOT_NAME       = "Пророк"
BOT_NAME_LOWER = "пророк"
REACTION_UP    = {"👍", "❤️", "🔥", "🥰", "👏", "🎉", "🤩", "🙏"}
REACTION_DOWN  = {"👎", "💩", "🤬", "🤮", "😢"}

# Redis — best-effort: если модуль/REDIS_URL есть, включаем quality-трекинг,
# иначе handle_reaction тихо выключается (без краша и без жёсткой зависимости).
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None
redis_client = None


# ── Один ответ на одно сообщение ─────────────────────────────────────────────
# У Пророка два независимых входа в группу, и друг о друге они не знают:
#   1. handle_group_message — RANDOM_REPLY_CHANCE_HUMAN = 0.40, он берёт слово
#      сам, не спрашивая Филли;
#   2. HTTP /task, когда роутер выбрал Пророка.
# Тот же расклад, что у Гослинга, где он 16.08 08:41 дал два ответа на одно
# «доброе утро» — короткий и монолог, с разницей в четыре секунды. У Пророка
# это пока не выстрелило только потому, что роутер зовёт его редко.
#
# ВНИМАНИЕ: это вторая копия замка (первая — в gosling-bot). Канонический дом у
# неё в ai_office_shared, но пин Пророка отстаёт от main на 45 коммитов, и
# тащить их в живого бота ради двенадцати строк несоразмерно. Переносить в
# shared — когда пин будут бампать осознанно, вместе с Гослингом.
ANSWER_LOCK_TTL = 180   # с — заведомо больше самого долгого ответа


def _answer_key(text: str) -> str:
    """
    Замок по ТЕКСТУ, а не по message_id: у HTTP-пути message_id нет вовсе,
    Филли передаёт только текст.
    """
    norm = " ".join((text or "").split()).lower()[:300]
    return "office:answered:" + BOT_NAME_LOWER + ":" + \
           hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


async def claim_answer(text: str) -> bool:
    """
    Занять право ответить. False — по этому сообщению уже отвечает другой путь.

    SET NX EX одной операцией: раздельные «проверить» и «занять» оставили бы
    щель ровно того размера, в которую два пути и попадают — секунда-полторы.
    Fail-open: Redis недоступен — лучше два ответа, чем ни одного.
    """
    if redis_client is None or not text:
        return True
    try:
        got = await redis_client.set(_answer_key(text), "1", nx=True,
                                     ex=ANSWER_LOCK_TTL)
        return bool(got)
    except Exception as e:
        logger.warning(f"[dedup] замок недоступен, отвечаю без него: {e}")
        return True


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
                json={"message": question, "user_id": user_id}, headers=office_headers())
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

    # Полный прогноз — ключевой синтез Пророка → Opus 4.8; короткий режим — Sonnet 4.6
    try:
        msg = await _anthropic_call(client,
            model="claude-sonnet-4-6" if short_mode else "claude-opus-4-8",
            max_tokens=400 if short_mode else 700,
            # Ростер офиса фактом: в промте Пророка коллеги были одной строкой
            # «и боты-коллеги» — кто именно, он не знал.
            system=(SYSTEM_SHORT if short_mode else SYSTEM_FULL)
                   + "\n\n" + roster_prompt(BOT_NAME),
            messages=[{"role": "user", "content": prompt}]
        )
    except APIError as e:
        if redis_client is not None:
            await log_event(redis_client, BOT_NAME_LOWER, "api_error", level="error",
                            user_id=user_id, error=str(e)[:200])
        raise
    return msg.content[0].text


SYSTEM_ORACLE = """Ты — Пророк. В офисной группе говоришь редко — но когда говоришь, это запоминают.

Один абзац, не больше. Реагируй на суть разговора — брось наблюдение, предупреди о риске, или дай неожиданный угол зрения. Можешь быть загадочным, но всегда конкретным. Без воды, без "интересный вопрос".

Офис: Влад (хозяин, строит автобизнес), Лук (друг со школы), и боты-коллеги.

По-русски."""

async def quick_prophesy(text: str, short: bool = False) -> str:
    """Быстрый ответ без сбора советников — для случайных ответов в группе."""
    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150 if short else 300,
        system=SYSTEM_ORACLE + "\n\n" + roster_prompt(BOT_NAME),
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
    await log("MSG_IN", question)
    await update.message.reply_text("🔮 Советуюсь с офисом...")
    answer = await prophesy(question, YOUR_TELEGRAM_ID, short_mode=False)
    await update.message.reply_text(answer, parse_mode=None)
    await log("MSG_OUT", answer)

# ── HTTP endpoint (for Filly routing) ────────────────────────────────────────

async def handle_reply(request):
    """POST /reply — принимает ответ от Филли и отправляет пользователю."""
    try:
        data      = await request.json()
        chat_id   = data.get("chat_id")
        text      = data.get("text", "")
        from_agent = data.get("from_agent", "")
        if not chat_id or not text:
            return web.Response(status=400, text="chat_id and text required")
        prefix = f"[{from_agent}] " if from_agent else ""
        await _ptb_bot.send_message(chat_id=int(chat_id), text=prefix + text)
        return web.Response(text="ok")
    except Exception as e:
        logger.error(f"[ПРОРОК] /reply error: {e}")
        return web.Response(status=500, text=str(e))

async def handle_health(request):
    return web.Response(text="ok")


async def handle_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Реакции 👍/👎 → office:quality:{bot} (HASH up/down)."""
    reaction = update.message_reaction
    if not reaction:
        return
    if redis_client is None:
        return  # quality-трекинг выключен (нет Redis) — тихо выходим

    chat_id = reaction.chat.id
    msg_id  = reaction.message_id

    try:
        owner_raw = await redis_client.get(f"office:msg:{chat_id}:{msg_id}")
    except Exception as e:
        logger.warning(f"reaction owner lookup failed: {e}")
        return
    if not owner_raw or owner_raw != BOT_NAME_LOWER:
        return

    old_emojis = {r.emoji for r in (reaction.old_reaction or []) if getattr(r, "emoji", None)}
    new_emojis = {r.emoji for r in (reaction.new_reaction or []) if getattr(r, "emoji", None)}
    added   = new_emojis - old_emojis
    removed = old_emojis - new_emojis

    delta_up   = sum(1 for e in added if e in REACTION_UP)   - sum(1 for e in removed if e in REACTION_UP)
    delta_down = sum(1 for e in added if e in REACTION_DOWN) - sum(1 for e in removed if e in REACTION_DOWN)

    if delta_up == 0 and delta_down == 0:
        return

    try:
        key = f"office:quality:{BOT_NAME_LOWER}"
        if delta_up:
            await redis_client.hincrby(key, "up", delta_up)
        if delta_down:
            await redis_client.hincrby(key, "down", delta_down)
        logger.info(f"REACTION msg={msg_id} added={added} removed={removed} du={delta_up} dd={delta_down}")
    except Exception as e:
        logger.warning(f"quality hincrby failed: {e}")




async def handle_task(request):
    data = await request.json()
    question  = data.get("message", "")
    # Текст ДО обвеса [от X] и [Контекст группового чата] — только он совпадает
    # с тем, что видит телеграм-путь, а значит только по нему сходится замок.
    message_raw = question
    user_id   = data.get("user_id", YOUR_TELEGRAM_ID)
    group_ctx = data.get("group_ctx", "")
    sender    = _banter.sender_of(data)
    # Пророк входит в пул болталки, но его handle_task не читал ни автора,
    # ни контекст группы — он отвечал вслепую на вырванную из нити фразу.
    if sender:
        question = f"[от {sender}] {question}"
    if group_ctx:
        question = f"[Контекст группового чата]\n{group_ctx}\n\n[Запрос]\n{question}"
    await log("MSG_IN", f"[HTTP] {question[:200]}")

    # Болталка — отдельная реплика, а не второй заход на то же сообщение:
    # её замок не касается. Всё остальное проходит через claim_answer, потому
    # что у Пророка два независимых входа в группу (см. комментарий к замку).
    is_banter = _banter.is_banter(data)
    if not is_banter and not await claim_answer(message_raw):
        logger.info("[dedup] на это сообщение уже отвечает телеграм-путь")
        # 200, а не ошибка: Филли доставила, отвечать было не нужно.
        return web.json_response({"status": "ok", "response": "",
                                  "skipped": "duplicate"})

    # Болталка — это одна реплика в чат, а не консилиум. prophesy() опрашивает
    # ВСЕХ советников по HTTP и, если они недоступны, возвращает «❌ Советники
    # недоступны» — постить такое в ответ на трёп нельзя. Для болталки берём
    # quick_prophesy: один вызов, без фан-аута.
    if is_banter:
        response = await quick_prophesy(question, short=True)
    else:
        response = await prophesy(question, user_id, short_mode=True)
    await log("MSG_OUT", response)
    # Post directly to office group so Vlad sees it inline
    if OFFICE_CHAT_ID:
        try:
            from telegram import Bot as TGBot
            tg = TGBot(token=TELEGRAM_TOKEN)
            # Имя НЕ приклеиваем. Telegram и так рисует «Пророк» над каждым
            # сообщением бота, поэтому `f"Пророк:\n{response}"` давал в чате
            # дубль подписи — видно на скриншоте 16.08 12:26. В логи при этом
            # уходил чистый response (строкой выше), так что по логам баг был
            # не виден вовсе: расходились именно чат и лог.
            await tg.send_message(chat_id=OFFICE_CHAT_ID,
                text=response, parse_mode=None)
            await _ghist.push(redis_client, BOT_NAME, response)
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

    # Ботам тут отвечать нельзя и не получится: Telegram не доставляет боту
    # сообщения других ботов, ветка is_bot не срабатывала ни разу. Межботовые
    # реплики идут по HTTP /task (ai_office_shared.shared.banter).
    if is_bot:
        return
    if random.random() > RANDOM_REPLY_CHANCE_HUMAN:
        return

    # Замок занимаем ПОСЛЕ броска: несработавший бросок не должен затыкать
    # HTTP-путь, ради которого Филли и звала.
    if not await claim_answer(text):
        logger.info("[dedup] на это сообщение уже отвечает HTTP-путь — молчу")
        return

    try:
        answer = await quick_prophesy(text)
        await msg.reply_text(f"🔮 {answer}", parse_mode=None)
        await log("RANDOM", f"{'bot' if is_bot else 'human'}: {text[:60]}")
    except Exception as e:
        logger.error(f"handle_group_message failed: {e}")

# ── Main ────────────────────────────────────────────────────────────────────

_ptb_bot = None  # set in main()

async def handle_send(request):
    """POST /send {chat_id, text} — отправить от имени этого бота."""
    secret = request.headers.get("X-Secret-Token", "")
    if HTTP_SECRET and secret != HTTP_SECRET:
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        body = await request.json()
        chat_id = int(body["chat_id"])
        text    = str(body["text"])
    except (KeyError, ValueError) as e:
        return web.json_response({"error": f"bad request: {e}"}, status=400)
    await _ptb_bot.send_message(chat_id=chat_id, text=text)
    return web.json_response({"ok": True})


async def main():
    global _ptb_bot, redis_client
    if aioredis and os.environ.get("REDIS_URL"):
        try:
            redis_client = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)
            await redis_client.ping()
            logger.info("Redis подключён — quality-трекинг активен")
        except Exception as e:
            redis_client = None
            logger.warning(f"Redis недоступен, quality-трекинг выключен: {e}")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS, handle_group_message))
    if HAS_REACTION_HANDLER:
        app.add_handler(MessageReactionHandler(handle_reaction))
    await app.initialize()
    await app.start()
    _ptb_bot = app.bot
    await app.updater.start_polling()
    # HTTP server
    http_app = web.Application(middlewares=[office_auth_middleware])
    http_app.router.add_post("/send",   handle_send)
    http_app.router.add_post("/task",  handle_task)
    http_app.router.add_get("/health",  handle_health)
    http_app.router.add_post("/reply",  handle_reply)
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
    await site.start()
    logger.info("Prophet online 🔮")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
