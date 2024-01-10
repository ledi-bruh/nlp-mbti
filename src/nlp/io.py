from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


__all__ = [
    'load_model',
    'get_model',
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
