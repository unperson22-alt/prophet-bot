"""
Что Пророк отправляет в группу и сколько раз.

Два дефекта, оба видны в чате и невидимы в логах:

  1. Дубль подписи. `text=f"Пророк:\\n{response}"` приклеивал имя, хотя Telegram
     и так рисует «Пророк» над сообщением бота — скриншот 16.08 12:26. В лог
     строкой выше уходил чистый response, поэтому по логам баг не читался
     вообще: расходились именно чат и лог.
  2. Два входа в группу. handle_group_message берёт слово сам с шансом 0.40, и
     независимо от него приходит HTTP /task от Филли. У Гослинга тот же расклад
     16.08 08:41 дал два ответа на одно «доброе утро».

Запуск:
    cd prophet-bot && python3 -m unittest discover -s tests -v
"""

import asyncio
import os
import sys
import unittest

os.environ.setdefault("TELEGRAM_TOKEN", "test")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("YOUR_TELEGRAM_ID", "391077101")
os.environ.setdefault("OFFICE_CHAT_ID", "-100500")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot  # noqa: E402


def run(coro):
    try:
        return asyncio.run(coro)
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


class FakeRedis:
    """Только SET NX EX — больше замку ничего не нужно."""

    def __init__(self, broken=False):
        self.keys, self.broken = {}, broken

    async def set(self, key, value, nx=False, ex=None):
        if self.broken:
            raise RuntimeError("redis лёг")
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        return True


class _FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class TestClaimAnswer(unittest.TestCase):
    def setUp(self):
        self._orig = bot.redis_client
        bot.redis_client = FakeRedis()

    def tearDown(self):
        bot.redis_client = self._orig

    def test_first_claim_wins_second_is_refused(self):
        self.assertTrue(run(bot.claim_answer("Ну как там с деньгами?")))
        self.assertFalse(run(bot.claim_answer("Ну как там с деньгами?")))

    def test_whitespace_and_case_do_not_split_the_lock(self):
        self.assertTrue(run(bot.claim_answer("Ну как  там\nс деньгами?")))
        self.assertFalse(run(bot.claim_answer("ну как там с деньгами?")))

    def test_different_messages_do_not_block_each_other(self):
        self.assertTrue(run(bot.claim_answer("первое")))
        self.assertTrue(run(bot.claim_answer("второе")))

    def test_key_is_namespaced_to_this_bot(self):
        self.assertTrue(bot._answer_key("x").startswith("office:answered:пророк:"))

    def test_broken_redis_is_fail_open(self):
        bot.redis_client = FakeRedis(broken=True)
        self.assertTrue(run(bot.claim_answer("текст")))

    def test_no_redis_is_fail_open(self):
        bot.redis_client = None
        self.assertTrue(run(bot.claim_answer("текст")))
        self.assertTrue(run(bot.claim_answer("текст")))


class TestGroupMessageText(unittest.IsolatedAsyncioTestCase):
    """Имя в текст не приклеивается — его рисует Telegram."""

    async def asyncSetUp(self):
        self.sent = []
        self._orig = (bot.prophesy, bot.quick_prophesy, bot.redis_client,
                      bot.log, bot._ghist)
        sent = self.sent

        async def fake_prophesy(question, user_id, **kw):
            return "Ноут чует — сегодня будет нагрузка."

        async def fake_quick(question, **kw):
            return "Это тоже форма стратегии."

        async def fake_log(*a, **k):
            pass

        class _FakeGhist:
            async def push(self, *a, **k):
                pass

        class _FakeTGBot:
            def __init__(self, token=None):
                pass

            async def send_message(self, chat_id=None, text=None, **kw):
                sent.append(text)

        import telegram
        self._orig_tg = telegram.Bot
        telegram.Bot = _FakeTGBot

        bot.prophesy = fake_prophesy
        bot.quick_prophesy = fake_quick
        bot.log = fake_log
        bot._ghist = _FakeGhist()
        bot.redis_client = FakeRedis()

    async def asyncTearDown(self):
        (bot.prophesy, bot.quick_prophesy, bot.redis_client,
         bot.log, bot._ghist) = self._orig
        import telegram
        telegram.Bot = self._orig_tg

    async def test_no_name_prefix_in_the_group_message(self):
        await bot.handle_task(_FakeRequest({"message": "что там по нагрузке"}))
        self.assertEqual(self.sent, ["Ноут чует — сегодня будет нагрузка."])
        self.assertFalse(self.sent[0].startswith("Пророк:"),
                         "имя приклеено к тексту — в чате выйдет дубль подписи")

    async def test_banter_reply_also_has_no_prefix(self):
        await bot.handle_task(_FakeRequest(
            {"message": "[Болталка] что там", "source": "BANTER", "depth": 1}))
        self.assertEqual(self.sent, ["Это тоже форма стратегии."])

    async def test_duplicate_http_call_is_silent(self):
        await bot.claim_answer("что там по нагрузке")      # телеграм-путь успел
        r = await bot.handle_task(_FakeRequest({"message": "что там по нагрузке"}))
        self.assertEqual(self.sent, [], "запостил второй ответ в группу")
        self.assertEqual(r.status, 200, "Филли доставила — это не её ошибка")

    async def test_banter_ping_is_never_deduped(self):
        # Реплика болталки — отдельное высказывание, а не второй заход.
        await bot.claim_answer("[Болталка] что там")
        await bot.handle_task(_FakeRequest(
            {"message": "[Болталка] что там", "source": "BANTER", "depth": 1}))
        self.assertEqual(self.sent, ["Это тоже форма стратегии."])

    async def test_lock_uses_the_raw_message_not_the_decorated_one(self):
        # В question дописываются «[от X]» и «[Контекст группового чата]» —
        # по ним замок с телеграм-путём не сойдётся никогда.
        await bot.claim_answer("что там по нагрузке")
        await bot.handle_task(_FakeRequest({
            "message": "что там по нагрузке",
            "sender": "Влад",
            "group_ctx": "Влад: привет\nБилли: здарова"}))
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
