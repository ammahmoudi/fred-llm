"""
Fredholm-LLM Dataset Fetcher.

Downloads and manages the Fredholm integral equation dataset from Zenodo.
Source: https://github.com/alirezaafzalaghaei/Fredholm-LLM
"""

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import httpx

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Zenodo dataset information
ZENODO_DOI = "10.5281/zenodo.16784707"
ZENODO_RECORD_ID = "16784707"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Default download directory
DEFAULT_DATA_DIR = Path("data/raw")

# Expected files in the dataset
DATASET_FILES = {
    "full": "Fredholm_Dataset.csv",
    "sample": "Fredholm_Dataset_Sample.csv",  # Created by sampling from full
}

# ZIP archive name on Zenodo
ZENODO_ARCHIVE_NAME = "Fredholm-Dataset-for-LLMs.zip"

# Sample size for creating sample dataset
SAMPLE_SIZE = 5000


class FredholmDatasetFetcher:
    """Fetches and manages the Fredholm-LLM dataset from Zenodo."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        timeout: int = 300,
    ) -> None:
        """
        Initialize the dataset fetcher.

        Args:
            data_dir: Directory to store downloaded data.
            timeout: HTTP request timeout in seconds.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._metadata: dict[str, Any] | None = None

    def get_zenodo_metadata(self) -> dict[str, Any]:
        """
        Fetch metadata from Zenodo API.

        Returns:
            Zenodo record metadata.
        """
        if self._metadata is not None:
            return self._metadata

        logger.info(f"Fetching Zenodo metadata for record {ZENODO_RECORD_ID}")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(ZENODO_API_URL)
            response.raise_for_status()
            self._metadata = response.json()

        # At this point self._metadata is guaranteed to be set
        assert self._metadata is not None
        return self._metadata

    def list_available_files(self) -> list[dict[str, Any]]:
        """
        List all files available in the Zenodo record.

        Returns:
            List of file metadata dictionaries.
        """
        metadata = self.get_zenodo_metadata()
        files = metadata.get("files", [])

        logger.info(f"Found {len(files)} files in Zenodo record")
        for f in files:
            size_mb = f.get("size", 0) / (1024 * 1024)
            logger.debug(f"  - {f['key']}: {size_mb:.2f} MB")

        return files

    def download_file(
        self,
        filename: str,
        force: bool = False,
        verify_checksum: bool = True,
    ) -> Path:
        """
        Download a specific file from the Zenodo record.

        Args:
            filename: Name of the file to download.
            force: Force re-download even if file exists.
            verify_checksum: Verify MD5 checksum after download.

        Returns:
            Path to the downloaded file.
        """
        output_path = self.data_dir / filename

        # Check if already downloaded
        if output_path.exists() and not force:
            logger.info(f"File already exists: {output_path}")
            return output_path

        # Get file metadata
        files = self.list_available_files()
        file_info = next((f for f in files if f["key"] == filename), None)

        if file_info is None:
            available = [f["key"] for f in files]
            raise ValueError(f"File '{filename}' not found. Available: {available}")

        download_url = file_info["links"]["self"]
        expected_checksum = file_info.get("checksum", "").replace("md5:", "")
        file_size = file_info.get("size", 0)

        logger.info(f"Downloading {filename} ({file_size / (1024 * 1024):.2f} MB)...")

        # Download with progress
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            with client.stream("GET", download_url) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 10%
                        if total > 0:
                            progress = (downloaded / total) * 100
                            if downloaded == total or int(progress) % 10 == 0:
                                logger.debug(f"Progress: {progress:.1f}%")

        logger.info(f"Downloaded to {output_path}")

        # Verify checksum
        if verify_checksum and expected_checksum:
            actual_checksum = self._compute_md5(output_path)
            if actual_checksum != expected_checksum:
                output_path.unlink()  # Remove corrupted file
                raise ValueError(
                    f"Checksum mismatch for {filename}. "
                    f"Expected: {expected_checksum}, Got: {actual_checksum}"
                )
            logger.info("Checksum verified successfully")

        return output_path

    def download_dataset(
        self,
        variant: str = "sample",
        force: bool = False,
    ) -> Path:
        """
        Download the Fredholm dataset.

        The dataset is distributed as a ZIP archive on Zenodo.
        This method downloads the archive, extracts it, and returns
        the path to the requested CSV file.

        For the 'sample' variant, a random sample is created from the
        full dataset since the Zenodo archive only contains the full data.

        Args:
            variant: Dataset variant ('full' or 'sample').
            force: Force re-download.

        Returns:
            Path to the downloaded CSV file.
        """
        if variant not in DATASET_FILES:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(DATASET_FILES.keys())}"
            )

        csv_filename = DATASET_FILES[variant]
        csv_path = self.data_dir / csv_filename

        # Check if CSV already exists
        if csv_path.exists() and not force:
            logger.info(f"Dataset already exists: {csv_path}")
            return csv_path

        # First, ensure the full dataset is available
        full_csv_path = self.data_dir / DATASET_FILES["full"]

        if not full_csv_path.exists() or force:
            # Download the ZIP archive
            zip_path = self.data_dir / ZENODO_ARCHIVE_NAME
            if not zip_path.exists() or force:
                self.download_file(ZENODO_ARCHIVE_NAME, force=force)

            # Extract the archive
            logger.info(f"Extracting {ZENODO_ARCHIVE_NAME}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)

        # Verify full dataset exists
        if not full_csv_path.exists():
            raise FileNotFoundError(
                f"Could not find {DATASET_FILES['full']} after extraction. "
                f"Contents: {list(self.data_dir.iterdir())}"
            )

        # If requesting full dataset, we're done
        if variant == "full":
            logger.info(f"Dataset ready: {full_csv_path}")
            return full_csv_path

        # For sample variant, create a random sample from full dataset
        logger.info(f"Creating sample dataset ({SAMPLE_SIZE} rows)...")
        self._create_sample(full_csv_path, csv_path, SAMPLE_SIZE)

        logger.info(f"Sample dataset ready: {csv_path}")
        return csv_path

    def _create_sample(
        self,
        source_path: Path,
        output_path: Path,
        n_samples: int,
    ) -> None:
        """
        Create a random sample from the full dataset.

        Args:
            source_path: Path to full dataset CSV.
            output_path: Path to write sample CSV.
            n_samples: Number of samples to include.
        """
        import pandas as pd

        logger.info(f"Loading full dataset from {source_path}")
        df = pd.read_csv(source_path)

        if len(df) <= n_samples:
            logger.warning(
                f"Full dataset ({len(df)} rows) is smaller than requested "
                f"sample size ({n_samples}). Using entire dataset."
            )
            sample_df = df
        else:
            sample_df = df.sample(n=n_samples, random_state=42)

        sample_df.to_csv(output_path, index=False)
        logger.info(f"Created sample with {len(sample_df)} rows")

    def download_all(self, force: bool = False) -> list[Path]:
        """
        Download all available files from the Zenodo record.

        Args:
            force: Force re-download.

        Returns:
            List of paths to downloaded files.
        """
        files = self.list_available_files()
        downloaded = []

        for file_info in files:
            filename = file_info["key"]
            try:
                path = self.download_file(filename, force=force)
                downloaded.append(path)
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")

        return downloaded

    def extract_if_archive(self, filepath: Path) -> Path:
        """
        Extract file if it's a zip archive.

        Args:
            filepath: Path to the file.

        Returns:
            Path to extracted directory or original file.
        """
        if filepath.suffix.lower() == ".zip":
            extract_dir = filepath.parent / filepath.stem
            extract_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Extracting {filepath} to {extract_dir}")
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(extract_dir)

            return extract_dir

        return filepath

    @staticmethod
    def _compute_md5(filepath: Path) -> str:
        """Compute MD5 checksum of a file."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()


def download_fredholm_dataset(
    variant: str = "sample",
    data_dir: Path | str = DEFAULT_DATA_DIR,
    force: bool = False,
) -> Path:
    """
    Convenience function to download the Fredholm dataset.

    Args:
        variant: Dataset variant ('full' or 'sample').
        data_dir: Directory to store data.
        force: Force re-download.

    Returns:
        Path to the downloaded CSV file.
    """
    fetcher = FredholmDatasetFetcher(data_dir=data_dir)
    return fetcher.download_dataset(variant=variant, force=force)


if __name__ == "__main__":
    # Example usage
    import sys

    from rich.console import Console

    console = Console()

    variant = sys.argv[1] if len(sys.argv) > 1 else "sample"
    path = download_fredholm_dataset(variant=variant)
    console.print(f"[bold green]OK - Dataset downloaded to:[/bold green] {path}")
