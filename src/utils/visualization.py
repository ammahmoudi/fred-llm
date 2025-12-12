"""
Visualization utilities for Fredholm integral equations.

Provides plotting functions for solutions and comparisons.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)

# Type alias for solution functions
SolutionType = sp.Expr | np.ndarray | Callable[[float], float]
KernelType = sp.Expr | Callable[[float, float], float]


def _evaluate_solution(
    solution: SolutionType,
    x_points: np.ndarray,
) -> np.ndarray:
    """Evaluate a solution at given points."""
    if isinstance(solution, np.ndarray):
        return solution
    elif isinstance(solution, sp.Expr):
        x = sp.Symbol("x")
        func = sp.lambdify(x, solution, modules=["numpy"])
        return np.array([func(float(xi)) for xi in x_points])
    elif callable(solution):
        return np.array([solution(float(xi)) for xi in x_points])
    else:
        raise ValueError(f"Unknown solution type: {type(solution)}")


def plot_solution(
    solution: SolutionType,
    a: float = 0,
    b: float = 1,
    n_points: int = 100,
    title: str = "Solution u(x)",
    xlabel: str = "x",
    ylabel: str = "u(x)",
    save_path: Path | str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Figure:
    """
    Plot a solution function.

    Args:
        solution: Solution to plot (SymPy expr, array, or callable).
        a: Lower domain bound.
        b: Upper domain bound.
        n_points: Number of plot points.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Path to save the figure.
        show: Whether to display the plot.
        **kwargs: Additional matplotlib kwargs.

    Returns:
        Matplotlib figure.
    """
    x_points = np.linspace(a, b, n_points)
    y_points = _evaluate_solution(solution, x_points)

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))

    ax.plot(x_points, y_points, linewidth=2, color=kwargs.get("color", "blue"))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=kwargs.get("dpi", 150), bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_comparison(
    solutions: list[SolutionType],
    labels: list[str],
    a: float = 0,
    b: float = 1,
    n_points: int = 100,
    title: str = "Solution Comparison",
    save_path: Path | str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Figure:
    """
    Plot multiple solutions for comparison.

    Args:
        solutions: List of solutions to plot.
        labels: Labels for each solution.
        a: Lower domain bound.
        b: Upper domain bound.
        n_points: Number of plot points.
        title: Plot title.
        save_path: Path to save the figure.
        show: Whether to display the plot.
        **kwargs: Additional matplotlib kwargs.

    Returns:
        Matplotlib figure.
    """
    x_points = np.linspace(a, b, n_points)

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 6)))

    # Get colormap colors
    cmap = plt.get_cmap("tab10")

    for i, (solution, label) in enumerate(zip(solutions, labels, strict=False)):
        y_points = _evaluate_solution(solution, x_points)
        ax.plot(
            x_points,
            y_points,
            label=label,
            linewidth=2,
            color=cmap(i % 10),
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=kwargs.get("dpi", 150), bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_error(
    predicted: SolutionType,
    ground_truth: SolutionType,
    a: float = 0,
    b: float = 1,
    n_points: int = 100,
    title: str = "Error: |predicted - ground_truth|",
    save_path: Path | str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Figure:
    """
    Plot the error between predicted and ground truth solutions.

    Args:
        predicted: Predicted solution.
        ground_truth: Ground truth solution.
        a: Lower domain bound.
        b: Upper domain bound.
        n_points: Number of plot points.
        title: Plot title.
        save_path: Path to save the figure.
        show: Whether to display the plot.
        **kwargs: Additional matplotlib kwargs.

    Returns:
        Matplotlib figure.
    """
    x_points = np.linspace(a, b, n_points)

    y_pred = _evaluate_solution(predicted, x_points)
    y_true = _evaluate_solution(ground_truth, x_points)
    error = np.abs(y_pred - y_true)

    fig, axes = plt.subplots(1, 2, figsize=kwargs.get("figsize", (14, 5)))

    # Plot solutions
    axes[0].plot(x_points, y_pred, label="Predicted", linewidth=2)
    axes[0].plot(x_points, y_true, label="Ground Truth", linewidth=2, linestyle="--")
    axes[0].set_title("Solutions")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot error
    axes[1].plot(x_points, error, linewidth=2, color="red")
    axes[1].fill_between(x_points, 0, error, alpha=0.3, color="red")
    axes[1].set_title(title)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|error|")
    axes[1].grid(True, alpha=0.3)

    # Add error statistics
    max_err = float(np.max(error))
    mean_err = float(np.mean(error))
    axes[1].text(
        0.02,
        0.98,
        f"Max: {max_err:.2e}\nMean: {mean_err:.2e}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=kwargs.get("dpi", 150), bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_kernel_heatmap(
    kernel: KernelType,
    a: float = 0,
    b: float = 1,
    n_points: int = 50,
    title: str = "Kernel K(x, t)",
    save_path: Path | str | None = None,
    show: bool = True,
    **kwargs: Any,
) -> Figure:
    """
    Plot a heatmap of the kernel function.

    Args:
        kernel: Kernel function K(x, t).
        a: Lower domain bound.
        b: Upper domain bound.
        n_points: Number of points in each dimension.
        title: Plot title.
        save_path: Path to save the figure.
        show: Whether to display the plot.
        **kwargs: Additional matplotlib kwargs.

    Returns:
        Matplotlib figure.
    """
    x_points = np.linspace(a, b, n_points)
    t_points = np.linspace(a, b, n_points)
    X, T = np.meshgrid(x_points, t_points)

    if isinstance(kernel, sp.Expr):
        x, t = sp.symbols("x t")
        func = sp.lambdify((x, t), kernel, modules=["numpy"])
        Z = func(X, T)
    elif callable(kernel):
        Z = kernel(X, T)
    else:
        raise ValueError(f"Unknown kernel type: {type(kernel)}")

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

    im = ax.pcolormesh(X, T, Z, shading="auto", cmap=kwargs.get("cmap", "viridis"))
    plt.colorbar(im, ax=ax, label="K(x, t)")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("t")

    if save_path:
        fig.savefig(save_path, dpi=kwargs.get("dpi", 150), bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig
