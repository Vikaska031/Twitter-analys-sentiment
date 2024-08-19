import numpy as np
from config import SEQ_LEN

def padding(review_int: list, seq_len: int = SEQ_LEN) -> np.array:
    """Pad sequences to the same length."""
    padded_reviews = np.zeros((len(review_int), seq_len))
    for i, review in enumerate(review_int):
        if len(review) < seq_len:
            padded_reviews[i, :len(review)] = review
        else:
            padded_reviews[i] = review[:seq_len]
    return padded_reviews
