"""Fashion-MNIST dataset loader."""

import gzip
import os
import struct
import tempfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

from dptiny.backend import xp

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny", "fashion-mnist")
_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion"
)

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _download(url: str, path: str) -> None:
    """Download to a temporary file and atomically move it into the cache."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)

    try:
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _read_idx(path: str) -> np.ndarray:
    """Read a gzip-compressed uint8 IDX file and validate its header and size."""
    try:
        with gzip.open(path, "rb") as f:
            raw = f.read()
    except (OSError, EOFError) as exc:
        raise ValueError(f"Invalid gzip/IDX file: {path}") from exc

    if len(raw) < 4:
        raise ValueError(f"IDX file is too small to contain a header: {path}")

    zero1, zero2, dtype_code, ndim = raw[:4]
    if zero1 != 0 or zero2 != 0 or dtype_code != 0x08:
        raise ValueError(f"IDX file is not a uint8 IDX file: {path}")
    if ndim == 0:
        raise ValueError(f"IDX file has zero dimensions: {path}")

    header_size = 4 + 4 * ndim
    if len(raw) < header_size:
        raise ValueError(f"IDX file has an incomplete header: {path}")

    shape = struct.unpack(f">{ndim}I", raw[4:header_size])
    expected_size = header_size + int(np.prod(shape, dtype=np.int64))
    if len(raw) != expected_size:
        raise ValueError(
            f"IDX file size does not match its header: {path} "
            f"(expected {expected_size} bytes, got {len(raw)})"
        )

    return np.frombuffer(raw, dtype=np.uint8, offset=header_size).reshape(shape).copy()


def get_fashion_mnist(
    normalize: bool = True,
    flatten: bool = True,
    data_home: Optional[str] = None,
) -> Tuple["xp.ndarray", "xp.ndarray", "xp.ndarray", "xp.ndarray"]:
    """Load the Fashion-MNIST dataset.

    Args:
        normalize: If True, scale pixel values to [0, 1].
        flatten: If True, return images as (N, 784) vectors.
        data_home: Directory used to cache the downloaded files.

    Returns:
        (X_train, X_test, y_train, y_test) arrays.
    """
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {}
    for key, filename in _FILES.items():
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            _download(f"{_BASE_URL}/{filename}", path)
        paths[key] = path

    X_train = _read_idx(paths["train_images"])
    y_train = _read_idx(paths["train_labels"])
    X_test = _read_idx(paths["test_images"])
    y_test = _read_idx(paths["test_labels"])

    if X_train.shape != (60000, 28, 28):
        raise ValueError(f"Unexpected training image shape: {X_train.shape}")
    if X_test.shape != (10000, 28, 28):
        raise ValueError(f"Unexpected test image shape: {X_test.shape}")
    if y_train.shape != (60000,):
        raise ValueError(f"Unexpected training label shape: {y_train.shape}")
    if y_test.shape != (10000,):
        raise ValueError(f"Unexpected test label shape: {y_test.shape}")

    X_train = X_train.astype(xp.float32)
    X_test = X_test.astype(xp.float32)
    y_train = y_train.astype(xp.int32)
    y_test = y_test.astype(xp.int32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
