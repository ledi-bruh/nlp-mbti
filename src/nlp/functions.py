import torch

from .io import get_model


__all__ = ['predict']


def predict(prompt: str) -> list[tuple[str, float]]:
    with torch.no_grad():
        model, tokenizer = get_model()

        tokenized = tokenizer(prompt, padding=True, return_tensors='pt')
        pred = model(**tokenized)
        probs: list[float] = pred.logits.softmax(1)[0].tolist()
        id2label = model.config.id2label
        return [(id2label[i], prob) for i, prob in enumerate(probs)]
