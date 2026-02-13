"""Discrete points solution evaluator."""

from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_discrete_points(
    pred_points: list[tuple[float, float]],
    gt_points: list[tuple[float, float]],
    x_tolerance: float = 1e-3,
    y_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """
    Compare discrete point predictions with ground truth.

    Matches x-coordinates within tolerance, compares y-values at matched points.

    Args:
        pred_points: List of (x, y) tuples from LLM prediction.
        gt_points: List of (x, y) tuples from ground truth.
        x_tolerance: Tolerance for matching x-coordinates.
        y_tolerance: Tolerance for considering y-values as matching.

    Returns:
        Dictionary with point-wise comparison metrics.
    """
    if not pred_points or not gt_points:
        logger.warning(
            f"Empty discrete_points: pred={len(pred_points)}, gt={len(gt_points)}"
        )
        return {
            "match": False,
            "matched_points": 0,
            "total_points_pred": len(pred_points),
            "total_points_gt": len(gt_points),
            "accuracy": 0.0,
            "max_error": float("inf"),
            "mean_error": float("inf"),
            "mae": float("inf"),
            "rmse": float("inf"),
        }

    matched = 0
    errors = []
    y_differences = []

    for x_pred, y_pred in pred_points:
        # Find closest x-coordinate in ground truth
        if not gt_points:
            break

        closest_gt = min(gt_points, key=lambda p: abs(p[0] - x_pred))
        x_gt, y_gt = closest_gt

        # Check if x-coordinates match within tolerance
        x_diff = abs(x_pred - x_gt)
        if x_diff < x_tolerance:
            # Match found - compare y-values
            y_diff = abs(y_pred - y_gt)
            y_differences.append(y_diff)
            errors.append(y_diff)

            if y_diff < y_tolerance:
                matched += 1

    # Compute metrics
    result = {
        "match": matched > 0 and matched >= len(pred_points) * 0.8,  # 80% threshold
        "matched_points": matched,
        "total_points_pred": len(pred_points),
        "total_points_gt": len(gt_points),
        "accuracy": matched / len(pred_points) if pred_points else 0.0,
        "max_error": float(np.max(errors)) if errors else float("inf"),
        "mean_error": float(np.mean(errors)) if errors else float("inf"),
        "mae": float(np.mean(errors)) if errors else float("inf"),
        "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2)))
        if errors
        else float("inf"),
    }

    return result
