"""BLEU score metric for string-level solution comparison."""

import re
from typing import Any

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _tokenize_math(text: str) -> list[str]:
    """Tokenize a math string by splitting on whitespace and operators."""
    # Insert spaces around mathematical operators so they become tokens
    text = re.sub(r"([+\-*/^()=,])", r" \1 ", text)
    return text.split()


def bleu_score(pred_str: str, gt_str: str) -> float:
    """
    Compute BLEU score between predicted and ground-truth solution strings.

    Uses nltk sentence_bleu with smoothing to avoid zero scores on short
    sequences.

    Args:
        pred_str: Predicted solution string.
        gt_str: Ground-truth solution string.

    Returns:
        BLEU score in [0.0, 1.0].
    """
    ref_tokens = _tokenize_math(gt_str)
    hyp_tokens = _tokenize_math(pred_str)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing))
