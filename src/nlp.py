import torch
from pathlib import Path
from operator import itemgetter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import settings


__all__ = [
    'load_model',
    'get_model',
    'predict',
    'get_softmax_for_mbti',
]


_model = None
_tokenizer = None


def load_model() -> None:
    model_name = settings.nlp_model_name

    global _model
    global _tokenizer

    if not Path(model_path := 'models/model').exists():
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.save_pretrained(model_path)
    else:
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)

    if not Path(tokenizer_path := 'models/tokenizer').exists():
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer.save_pretrained(tokenizer_path)
    else:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def get_model():
    global _model
    global _tokenizer

    return _model, _tokenizer


def predict(prompt: str) -> list[tuple[str, float]]:
    with torch.no_grad():
        model, tokenizer = get_model()

        tokenized = tokenizer(prompt, padding=True, return_tensors='pt')
        pred = model(**tokenized)
        probs: list[float] = pred.logits.softmax(1)[0].tolist()
        id2label = model.config.id2label
        return [(id2label[i], prob) for i, prob in enumerate(probs)]


def get_softmax_for_mbti(mbti: dict[str, float]) -> list[str, float]:
    if not mbti:
        return []

    types, points = zip(*mbti.items())
    probs = torch.tensor(points).softmax(0).tolist()

    return sorted(list(zip(types, probs)), key=itemgetter(1), reverse=True)
