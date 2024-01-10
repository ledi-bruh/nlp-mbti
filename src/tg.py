from aiogram import Bot, Dispatcher

from .config import settings
from .router import router


__all__ = ['start_bot']


async def start_bot():
    bot = Bot(token=settings.tg_bot_token)
    dp = Dispatcher()

    dp.include_router(router)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
