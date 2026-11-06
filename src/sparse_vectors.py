import math
import re
from collections import Counter

from qdrant_client import models


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts):
    return {
        token: index
        for index, token in enumerate(
            sorted({token for text in texts for token in tokenize(text)}),
            start=1,
        )
    }


def build_sparse_vector(text, vocabulary, document_frequency, document_count,
                        average_length):
    tokens = tokenize(text)
    counts = Counter(tokens)
    values = {}

    for token, count in counts.items():
        token_id = vocabulary.get(token)
        if token_id is None:
            continue

        idf = math.log(
            1
            + (document_count - document_frequency[token] + 0.5)
            / (document_frequency[token] + 0.5)
        )
        denominator = count + 1.5 * (
            0.25 + 0.75 * len(tokens) / max(average_length, 1)
        )
        values[token_id] = idf * count * 2.5 / denominator

    return models.SparseVector(
        indices=list(values.keys()),
        values=list(values.values()),
    )


def build_sparse_vectors(texts):
    tokenized = [tokenize(text) for text in texts]
    vocabulary = build_vocabulary(texts)
    document_frequency = Counter(
        token for tokens in tokenized for token in set(tokens)
    )
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)

    return [
        build_sparse_vector(
            text,
            vocabulary,
            document_frequency,
            len(tokenized),
            average_length,
        )
        for text in texts
    ], vocabulary, document_frequency, average_length
