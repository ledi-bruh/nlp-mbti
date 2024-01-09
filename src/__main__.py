import asyncio
import logging

from .nlp import load_model
from .tg import start_bot


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] "
               "[%(filename)s %(funcName)s(%(lineno)d)] %(message)s"
    )

    load_model()
    await start_bot()


if __name__ == '__main__':
    asyncio.run(main())
