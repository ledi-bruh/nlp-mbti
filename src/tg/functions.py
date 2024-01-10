import torch
from operator import itemgetter
from collections import defaultdict


__all__ = [
    'actualize_mbti',
    'get_softmax_for_mbti',
]


def actualize_mbti(mbti: defaultdict, probs: list[tuple[str, float]]) -> None:
    for mbti_type, prob in probs:
        mbti[mbti_type] += prob


def get_softmax_for_mbti(mbti: dict[str, float]) -> list[str, float]:
    if not mbti:
        return []

    types, points = zip(*mbti.items())
    probs = torch.tensor(points).softmax(0).tolist()

    return sorted(list(zip(types, probs)), key=itemgetter(1), reverse=True)
