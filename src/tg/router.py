import logging
from random import shuffle
from collections import defaultdict
from aiogram import Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.enums import ParseMode
from deep_translator import GoogleTranslator

from src.nlp import predict
from .questions import questions
from .functions import actualize_mbti, get_softmax_for_mbti


__all__ = ['router']


router = Router()
_translator = GoogleTranslator('ru', 'en')


class Quiz(StatesGroup):
    start = State()
    stop = State()
    result = State()


async def get_next(state: FSMContext) -> str | None:
    data = await state.get_data()
    order = data['order']
    
    if len(order) > 0:
        cur = order[0]
        await state.update_data(cur=cur)
        return questions[cur]['ru']
    
    return None


async def collect_answer(answer: str, state: FSMContext) -> None:
    data = await state.get_data()
    cur = data['cur']
    mbti = data['mbti']

    prompt = f'{questions[cur]["en"]} -- {_translator.translate(answer)}'
    logging.info(prompt)
    probs = predict(prompt)

    actualize_mbti(mbti, probs)
    
    await state.update_data(
        order=data['order'][1:],
        mbti=mbti,
    )


@router.message(StateFilter(None), Command('start'))
async def start_quiz_handler(message: types.Message, state: FSMContext):
    await message.answer('☢️ Началось тестирование ☢️')
    await message.answer('Старайтесь отвечать не больше предложения.')
    
    order = list(range(len(questions)))
    shuffle(order)
    
    await state.set_state(Quiz.start)
    await state.update_data(
        order=order,
        cur=order[0],
        mbti=defaultdict(float),
    )

    if (question := await get_next(state)):
        await message.answer(question)
    else:
        await message.answer('Вопросов больше нет 🤯')
        await stop_quiz_handler(message, state)


@router.message(Quiz.start, Command('stop'))
async def stop_quiz_handler(message: types.Message, state: FSMContext):
    await message.answer('Спасибо за прохождение тестирования 🐸')
    
    data = await state.get_data()
    mbti_result = get_softmax_for_mbti(data['mbti'])

    
    result_output = '\n'.join(f'{pair[0]}: {pair[1]:.3%}' for pair in mbti_result).replace('.', '\\.')
    await message.answer(f'`{result_output}`', parse_mode=ParseMode.MARKDOWN_V2)
    
    await state.clear()


@router.message(Quiz.start)
async def process_quiz_handler(message: types.Message, state: FSMContext):
    await collect_answer(message.text, state)
    # await message.answer(f'Ответ принят.')

    if (question := await get_next(state)):
        await message.answer(question)
    else:
        await message.answer('Вопросов больше нет 🤯')
        await stop_quiz_handler(message, state)
