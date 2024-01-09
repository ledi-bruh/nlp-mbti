from aiogram import Bot, Dispatcher

from .config import settings
from .router import router


__all__ = ['start_bot']


class TgBot:

    __slots__ = (
        '_bot',
        '_dp',
    )

    def __init__(self, token: str):
        self._bot = Bot(token)
        self._dp = Dispatcher()
    
    async def start(self):
        try:
            await self._dp.start_polling(self._bot)
        finally:
            await self._bot.session.close()


async def start_bot():
    bot = TgBot(token=settings.tg_bot_token)
    bot._dp.include_router(router)
    await bot.start()
