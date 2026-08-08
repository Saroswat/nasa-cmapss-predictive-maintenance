from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd

from .config import COLUMNS, DATASET_URL, FD001_SHA256, MIRROR_URL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_fd001(data_dir: Path) -> bool:
    for filename, expected in FD001_SHA256.items():
        path = data_dir / filename
        if not path.exists() or _sha256(path) != expected:
            return False
    return True


def _download_mirror(data_dir: Path) -> None:
    for filename, expected in FD001_SHA256.items():
        destination = data_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".part")
        with (
            urllib.request.urlopen(f"{MIRROR_URL}/{filename}", timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if _sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for mirrored file: {filename}")
        temporary.replace(destination)


def download_dataset(data_dir: Path, *, url: str = DATASET_URL, force: bool = False) -> Path:
    """Download NASA C-MAPSS, with a pinned and checksum-verified FD001 fallback."""
    data_dir = Path(data_dir)
    if _verify_fd001(data_dir) and not force:
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "CMAPSSData.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as bundle:
            root = data_dir.resolve()
            for member in bundle.infolist():
                destination = (data_dir / member.filename).resolve()
                if root not in destination.parents and destination != root:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(data_dir)
    except (OSError, TimeoutError, urllib.error.URLError, zipfile.BadZipFile) as error:
        archive.unlink(missing_ok=True)
        warnings.warn(
            f"NASA archive unavailable ({error}); using the pinned FD001 mirror.",
            RuntimeWarning,
            stacklevel=2,
        )
        _download_mirror(data_dir)

    if not _verify_fd001(data_dir):
        raise ValueError("FD001 files failed checksum verification")
    return data_dir


def _read_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Unexpected column count in {path}: {frame.shape[1]}")
    return frame


def load_fd001(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data_dir = Path(data_dir)
    train = _read_trajectory(data_dir / "train_FD001.txt")
    test = _read_trajectory(data_dir / "test_FD001.txt")
    truth = pd.read_csv(data_dir / "RUL_FD001.txt", sep=r"\s+", header=None).iloc[:, 0]
    truth.name = "rul"
    if test["unit_number"].nunique() != len(truth):
        raise ValueError("Test engine count does not match the RUL truth file")
    return train, test, truth


def add_train_rul(frame: pd.DataFrame, *, cap: int | None = 125) -> pd.DataFrame:
    result = frame.copy()
    max_cycles = result.groupby("unit_number")["time_in_cycles"].transform("max")
    result["rul_raw"] = max_cycles - result["time_in_cycles"]
    result["rul"] = result["rul_raw"].clip(upper=cap) if cap is not None else result["rul_raw"]
    return result


def test_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    indices = frame.groupby("unit_number")["time_in_cycles"].idxmax()
    return frame.loc[indices].sort_values("unit_number").reset_index(drop=True)
