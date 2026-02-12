"""Data module for Fred-LLM."""

from src.data.augmentation import augment_dataset
from src.data.dataset_fetcher import FredholmDatasetFetcher, download_fredholm_dataset
from src.data.format_converter import FormatConverter, convert_format
from src.data.fredholm_loader import (
    ExpressionType,
    FredholmDatasetLoader,
    FredholmEquation,
    load_fredholm_dataset,
)
from src.data.loader import DataLoader, load_dataset
from src.data.splitter import get_split_statistics, split_dataset, stratified_sample
from src.data.validator import validate_dataset, validate_equation

__all__ = [
    # Loaders
    "DataLoader",
    "load_dataset",
    # Fredholm-LLM specific
    "FredholmDatasetLoader",
    "FredholmDatasetFetcher",
    "FredholmEquation",
    "ExpressionType",
    "load_fredholm_dataset",
    "download_fredholm_dataset",
    # Utilities
    "convert_format",
    "FormatConverter",
    "augment_dataset",
    "validate_equation",
    "validate_dataset",
    "split_dataset",
    "stratified_sample",
    "get_split_statistics",
]
