"""Data loading and front-end feature extraction for ASVspoof 2019 LA.

The raw corpus is pulled from the Hugging Face Hub::

    from datasets import load_dataset
    ds = load_dataset("Bisher/ASVspoof_2019_LA")

which exposes three splits (``train`` / ``validation`` / ``test``) with the
following columns:

===============  ==========================================================
column           meaning
===============  ==========================================================
speaker_id       speaker label, e.g. ``LA_0079``
audio_file_name  utterance id, e.g. ``LA_T_1138215``
audio            waveform: ``{path, array, sampling_rate}`` with datasets<5,
                 a torchcodec ``AudioDecoder`` with datasets>=5
system_id        spoofing system that produced the file (``-`` = bonafide)
key              ``ClassLabel(["bonafide", "spoof"])``
===============  ==========================================================

:class:`ASVspoofPreprocessor` wraps that call and turns every utterance into a
fixed-size 2-D feature map (log-Mel / MFCC / LFCC), so downstream code only
sees ``(C, T)`` float tensors and integer labels.

Quick start
-----------
>>> from dataPreprocessor import ASVspoofPreprocessor, FeatureConfig
>>> pre = ASVspoofPreprocessor(FeatureConfig(feature="logmel", n_mels=64))
>>> pre.explore()                                  # cheap, no audio decoded
>>> train_ds = pre.prepare("train", limit=512)     # cached in memory
>>> row = train_ds[0]
>>> feats, label = np.asarray(row["features"]), int(row["label"])
>>> feats.shape, label                             # (64, 401) 0
>>> loader = pre.get_dataloader("train", batch_size=32)   # needs torch

Label convention is inherited from the dataset and never flipped:
``0 = bonafide`` (genuine human speech), ``1 = spoof`` (attack).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - import guard only
    from datasets import (
        Array2D,
        Audio,
        ClassLabel,
        Features,
        IterableDataset,
        Sequence,
        Value,
        load_dataset,
        load_from_disk,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dataPreprocessor requires the `datasets` package:\n"
        "    pip install datasets soundfile"
    ) from exc

try:  # torch is optional: only needed for DataLoaders / torch Datasets
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

__all__ = [
    "FeatureConfig",
    "ASVspoofPreprocessor",
    "NumpyFeatureDataset",
    "TorchFeatureDataset",
    "pool_feature_map",
    "mel_filterbank",
    "power_to_db",
    "dct_matrix",
]

LABEL_NAMES: Tuple[str, str] = ("bonafide", "spoof")
FEATURES = ("logmel", "mfcc", "lfcc")
NORMALIZATIONS = ("none", "per_utterance", "global")
POOLINGS = ("mean", "std", "mean_std", "min_max", "flat")


# --------------------------------------------------------------------------- #
# Signal-processing helpers (pure NumPy on purpose: no librosa/torchaudio
# dependency, and `datasets` already hands us decoded float waveforms).
# --------------------------------------------------------------------------- #
def hz_to_mel(freq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """HTK-style hertz -> mel conversion."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """HTK-style mel -> hertz conversion."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    norm: str = "slaney",
) -> np.ndarray:
    """Triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``."""
    if fmax is None:
        fmax = sample_rate / 2.0
    if fmax <= fmin:
        raise ValueError(f"fmax ({fmax}) must be greater than fmin ({fmin})")

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))

    bank = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for m in range(n_mels):
        left, centre, right = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rising = (fft_freqs - left) / max(centre - left, 1e-9)
        falling = (right - fft_freqs) / max(right - centre, 1e-9)
        bank[m] = np.maximum(0.0, np.minimum(rising, falling))

    if norm == "slaney":
        # Equal-area triangles (as in librosa's default for mel spectrograms).
        enorm = 2.0 / np.maximum(edges_hz[2:] - edges_hz[:-2], 1e-9)
        bank *= enorm[:, None]
    elif norm != "none":
        raise ValueError(f"unknown mel norm {norm!r}; use 'slaney' or 'none'")
    return bank


def power_to_db(power: np.ndarray, top_db: float = 80.0, amin: float = 1e-10) -> np.ndarray:
    """Convert a power spectrogram to dB, flooring at ``-top_db``."""
    power = np.maximum(power, amin)
    db = 10.0 * np.log10(power)
    peak = db.max() if db.size else 0.0
    db -= peak
    if top_db is not None and top_db > 0:
        db = np.maximum(db, -float(top_db))
    return db.astype(np.float32)


def dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape ``(n_out, n_in)``."""
    k = np.arange(n_out, dtype=np.float64)[:, None]
    n = np.arange(n_in, dtype=np.float64)[None, :]
    mat = np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * n_in)) * np.sqrt(2.0 / n_in)
    mat[0] *= np.sqrt(0.5)
    return mat.astype(np.float32)


def _stft_power(
    waveform: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> np.ndarray:
    """Power spectrogram ``(n_frames, n_fft // 2 + 1)`` with centred frames."""
    pad = n_fft // 2
    if pad > 0:
        mode = "reflect" if waveform.size > pad else "constant"
        waveform = np.pad(waveform, (pad, pad), mode=mode)

    window = np.zeros(n_fft, dtype=np.float32)
    left = (n_fft - win_length) // 2
    window[left : left + win_length] = np.hanning(win_length).astype(np.float32)

    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    spectrum = np.fft.rfft(frames * window, axis=-1)
    return (spectrum.real**2 + spectrum.imag**2).astype(np.float32)


def pool_feature_map(
    feature: np.ndarray, pooling: str = "mean_std"
) -> np.ndarray:
    """Collapse the time axis of a feature map into one fixed-length vector.

    Classical models (SVM, GMM, logistic regression) need a single vector per
    utterance, while :meth:`ASVspoofPreprocessor.prepare` keeps a ``(C, T)``
    map. This applies the usual "MFCC statistics" pooling:

    ==========  ===========  ==================================================
    pooling     output dims  definition
    ==========  ===========  ==================================================
    ``mean``    ``C``        mean of every coefficient over time
    ``std``     ``C``        standard deviation over time
    ``mean_std`` ``2C``      mean and std concatenated (the SVM default)
    ``min_max`` ``2C``       min and max over time
    ``flat``    ``C*T``      raw frame-level features, no pooling
    ==========  ===========  ==================================================

    Accepts a single ``(C, T)`` map or a batch ``(..., C, T)``; pooling is
    always over the last axis.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    arr = np.asarray(feature, dtype=np.float32)
    if pooling == "flat":
        return arr.reshape(arr.shape[:-2] + (-1,))
    if pooling == "mean":
        return arr.mean(axis=-1)
    if pooling == "std":
        return arr.std(axis=-1)
    if pooling == "mean_std":
        return np.concatenate([arr.mean(axis=-1), arr.std(axis=-1)], axis=-1)
    return np.concatenate([arr.min(axis=-1), arr.max(axis=-1)], axis=-1)  # min_max


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    """Everything that defines the front-end, plus I/O knobs.

    Two preprocessors with an identical config produce identical features, which
    also makes the on-disk cache key stable.
    """

    # ---- dataset -----------------------------------------------------------
    dataset_name: str = "Bisher/ASVspoof_2019_LA"
    config_name: Optional[str] = None
    splits: Tuple[str, ...] = ("train", "validation", "test")
    audio_column: str = "audio"
    label_column: str = "key"

    # ---- waveform ----------------------------------------------------------
    sample_rate: int = 16_000
    duration: float = 4.0          # seconds kept per utterance (ASVspoof LA ~4 s)
    crop_mode: str = "center"      # "center" | "first" | "random"
    pad_mode: str = "constant"     # NumPy pad mode when the clip is too short

    # ---- front-end ---------------------------------------------------------
    feature: str = "logmel"        # "logmel" | "mfcc" | "lfcc"
    n_fft: int = 512
    win_length: int = 400          # 25 ms @ 16 kHz
    hop_length: int = 160          # 10 ms @ 16 kHz
    n_mels: int = 64
    n_mfcc: int = 20
    n_lfcc: int = 20
    fmin: float = 20.0
    fmax: Optional[float] = None   # defaults to sample_rate / 2
    mel_norm: str = "slaney"       # "slaney" | "none"
    top_db: float = 80.0

    # ---- normalisation -----------------------------------------------------
    normalize: str = "per_utterance"   # "none" | "per_utterance" | "global"
    eps: float = 1e-5

    # ---- I/O / speed -------------------------------------------------------
    seed: int = 0
    limit: Optional[int] = None
    streaming: bool = False
    num_proc: int = 1
    cache_dir: Optional[str] = None    # None -> keep features in RAM only
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}, got {self.feature!r}")
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {NORMALIZATIONS}, got {self.normalize!r}"
            )
        if self.crop_mode not in ("center", "first", "random"):
            raise ValueError(f"unknown crop_mode {self.crop_mode!r}")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError("require 0 < win_length <= n_fft")
        if not 0 < self.hop_length <= self.n_fft:
            raise ValueError("require 0 < hop_length <= n_fft")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.dtype not in ("float32", "float16"):
            raise ValueError("dtype must be 'float32' or 'float16'")

    # ---- derived values ----------------------------------------------------
    @property
    def n_samples(self) -> int:
        """Number of audio samples kept per utterance."""
        return int(round(self.duration * self.sample_rate))

    @property
    def n_frames(self) -> int:
        """Number of feature frames produced for a fixed-length clip."""
        return 1 + self.n_samples // self.hop_length

    @property
    def n_coeffs(self) -> int:
        """Feature dimension (rows of the ``(C, T)`` matrix)."""
        return {
            "logmel": self.n_mels,
            "mfcc": self.n_mfcc,
            "lfcc": self.n_lfcc,
        }[self.feature]

    @property
    def feature_shape(self) -> Tuple[int, int]:
        return (self.n_coeffs, self.n_frames)

    def fingerprint(self) -> str:
        """Short stable hash used to key the on-disk feature cache."""
        payload = asdict(self)
        for volatile in ("cache_dir", "limit", "num_proc", "streaming", "seed"):
            payload.pop(volatile, None)
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.md5(blob).hexdigest()[:10]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
class ASVspoofPreprocessor:
    """Load ``Bisher/ASVspoof_2019_LA`` and turn waveforms into feature maps.

    Parameters
    ----------
    config:
        :class:`FeatureConfig`; keyword arguments are forwarded to it, e.g.
        ``ASVspoofPreprocessor(feature="lfcc", n_lfcc=20)``.

    Notes
    -----
    * Splits are loaded lazily, one at a time, so ``prepare("validation")``
      never downloads the training audio.
    * The hub stores one parquet file per split (``train`` ~1.4 GB,
      ``validation`` ~1.4 GB, ``test`` ~4 GB). Non-streaming ``prepare``
      therefore downloads a whole split; use ``streaming=True`` with ``limit``
      for a fast smoke test.
    * Features are cached per split in ``self._prepared`` so repeated calls are
      free. Pass ``cache_dir=...`` to also persist them as Arrow files
      (roughly ``4 * C * T`` bytes per utterance: ~97 kB for a 64x401 log-Mel).
    * ``labels`` keep the dataset's own encoding: ``0=bonafide``, ``1=spoof``.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = FeatureConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either a FeatureConfig or keyword args, not both")
        self.config = config
        self._raw: Dict[str, Any] = {}
        self._prepared: Dict[str, Any] = {}
        self._stats: Optional[Dict[str, np.ndarray]] = None
        self._mel_fb: Optional[np.ndarray] = None
        self._linear_fb: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------ info
    def __repr__(self) -> str:
        c = self.config
        return (
            f"{type(self).__name__}(feature={c.feature!r}, shape={c.feature_shape}, "
            f"normalize={c.normalize!r}, sr={c.sample_rate}, duration={c.duration}s)"
        )

    @property
    def label_names(self) -> Tuple[str, str]:
        return LABEL_NAMES

    @property
    def torch_available(self) -> bool:
        return torch is not None

    # --------------------------------------------------------------- loading
    def _load_raw(self, split: str):
        """``load_dataset(...)`` for one split (memoised, streaming-aware)."""
        if split in self._raw:
            return self._raw[split]

        kwargs: Dict[str, Any] = {
            "split": split,
            "streaming": self.config.streaming,
        }
        if self.config.config_name is not None:
            kwargs["name"] = self.config.config_name

        dataset = load_dataset(self.config.dataset_name, **kwargs)
        dataset = self._ensure_audio_decodable(dataset)
        self._raw[split] = dataset
        return dataset

    def _ensure_audio_decodable(self, dataset):
        """Make sure audio rows can actually be turned into waveforms.

        ``datasets >= 5`` needs ``torchcodec`` for its own decoder. If that is
        unavailable we cast the column to ``decode=False`` and decode the raw
        bytes with ``soundfile`` inside :meth:`_decode_audio` instead, which
        keeps the class usable without pulling in PyTorch.
        """
        if self.config.audio_column not in getattr(dataset, "column_names", []):
            return dataset
        try:
            if isinstance(dataset, IterableDataset):
                next(iter(dataset.take(1)))
            else:
                dataset[0]
        except ImportError:
            try:
                import soundfile  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "This version of `datasets` cannot decode audio without "
                    "`torchcodec`, and `soundfile` is not installed either.\n"
                    "Install one of them:\n"
                    "    pip install soundfile        # lightweight\n"
                    "    pip install torchcodec       # needs torch"
                ) from exc
            dataset = dataset.cast_column(
                self.config.audio_column, Audio(decode=False)
            )
        return dataset

    # ----------------------------------------------------------- front-end
    def _get_mel_fb(self) -> np.ndarray:
        if self._mel_fb is None:
            self._mel_fb = mel_filterbank(
                self.config.sample_rate,
                self.config.n_fft,
                self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                norm=self.config.mel_norm,
            )
        return self._mel_fb

    def _get_linear_fb(self) -> np.ndarray:
        """Equally spaced triangular filters (the classic LFCC filterbank)."""
        if self._linear_fb is None:
            c = self.config
            fmax = c.fmax if c.fmax is not None else c.sample_rate / 2.0
            edges = np.linspace(c.fmin, fmax, c.n_mels + 2)
            freqs = np.fft.rfftfreq(c.n_fft, d=1.0 / c.sample_rate)
            bank = np.zeros((c.n_mels, freqs.size), dtype=np.float32)
            for m in range(c.n_mels):
                left, centre, right = edges[m], edges[m + 1], edges[m + 2]
                rising = (freqs - left) / max(centre - left, 1e-9)
                falling = (right - freqs) / max(right - centre, 1e-9)
                bank[m] = np.maximum(0.0, np.minimum(rising, falling))
            self._linear_fb = bank
        return self._linear_fb

    def _fix_length(self, waveform: np.ndarray) -> np.ndarray:
        """Crop or pad a waveform to exactly ``config.n_samples`` samples."""
        c = self.config
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size < c.n_samples:
            deficit = c.n_samples - waveform.size
            return np.pad(waveform, (0, deficit), mode=c.pad_mode)
        if waveform.size > c.n_samples:
            if c.crop_mode == "first":
                start = 0
            elif c.crop_mode == "random":
                start = int(self._rng.integers(0, waveform.size - c.n_samples + 1))
            else:  # center
                start = (waveform.size - c.n_samples) // 2
            waveform = waveform[start : start + c.n_samples]
        return waveform

    def frontend(self, waveform: np.ndarray) -> np.ndarray:
        """Raw (un-normalised) feature map of shape ``config.feature_shape``."""
        c = self.config
        waveform = self._fix_length(waveform)
        power = _stft_power(waveform, c.n_fft, c.win_length, c.hop_length)

        if c.feature == "logmel":
            spec = power_to_db(self._get_mel_fb() @ power.T, top_db=c.top_db)
        elif c.feature == "mfcc":
            logmel = power_to_db(self._get_mel_fb() @ power.T, top_db=c.top_db)
            spec = dct_matrix(c.n_mels, c.n_mfcc) @ logmel
        else:  # lfcc
            loglin = power_to_db(self._get_linear_fb() @ power.T, top_db=c.top_db)
            spec = dct_matrix(c.n_mels, c.n_lfcc) @ loglin

        # Guard against tiny off-by-one frame differences.
        frames = c.n_frames
        if spec.shape[1] > frames:
            spec = spec[:, :frames]
        elif spec.shape[1] < frames:
            spec = np.pad(spec, ((0, 0), (0, frames - spec.shape[1])), mode="edge")

        return np.ascontiguousarray(spec, dtype=np.float32)

    def _normalize(self, feature: np.ndarray) -> np.ndarray:
        mode = self.config.normalize
        if mode == "none":
            return feature
        if mode == "per_utterance":
            mean = feature.mean(axis=1, keepdims=True)
            std = feature.std(axis=1, keepdims=True)
            return (feature - mean) / (std + self.config.eps)
        # global
        if self._stats is None:
            stats = self._load_stats()
            if stats is None:
                stats = self.compute_normalization_stats()
        mean, std = self._stats["mean"], self._stats["std"]
        return (feature - mean[:, None]) / (std[:, None] + self.config.eps)

    def extract_features(
        self, waveform: np.ndarray, sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Public helper: waveform (float) -> normalised ``(C, T)`` features."""
        if sample_rate is not None and sample_rate != self.config.sample_rate:
            waveform = self._resample(waveform, sample_rate, self.config.sample_rate)
        feature = self.frontend(waveform)
        return self._normalize(feature).astype(np.float32)

    # ------------------------------------------------------------- decoding
    @staticmethod
    def _resample(waveform: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """Cheap linear resampling (only used if the hub changes its sample rate)."""
        waveform = np.asarray(waveform, dtype=np.float32).ravel()
        if sr_in == sr_out or waveform.size == 0:
            return waveform
        n_out = int(round(waveform.size * sr_out / sr_in))
        x_old = np.linspace(0.0, 1.0, waveform.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, waveform).astype(np.float32)

    @staticmethod
    def _read_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
        """Decode an in-memory audio container with ``soundfile``."""
        try:
            import io

            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Audio is stored as bytes and `soundfile` is not installed:\n"
                "    pip install soundfile"
            ) from exc
        array, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return array.mean(axis=1).astype(np.float32), int(sr)

    def _decode_audio(self, audio: Any) -> Tuple[np.ndarray, int]:
        """Accept the HF ``Audio`` dict / torchcodec decoder / array / bytes."""
        if audio is None:
            raise ValueError(
                f"missing '{self.config.audio_column}' column in the dataset"
            )

        if isinstance(audio, dict):
            if audio.get("array") is not None:
                waveform = np.asarray(audio["array"], dtype=np.float32)
                sr = int(audio["sampling_rate"])
            elif audio.get("bytes") is not None:  # decode=False on the hub
                waveform, sr = self._read_audio_bytes(
                    bytes(audio["bytes"])
                )
            else:  # local file path
                path = audio.get("path")
                if path is None:
                    raise ValueError(
                        "audio dict has none of 'array', 'bytes' or 'path': "
                        f"{sorted(audio)}"
                    )
                import soundfile as sf

                data, sr = sf.read(path, dtype="float32", always_2d=True)
                waveform = data.mean(axis=1)
                sr = int(sr)
        elif isinstance(audio, (bytes, bytearray)):
            waveform, sr = self._read_audio_bytes(bytes(audio))
        elif isinstance(audio, (tuple, list)) and len(audio) == 2:
            waveform, sr = np.asarray(audio[0], dtype=np.float32), int(audio[1])
        elif isinstance(audio, np.ndarray):
            waveform, sr = audio.astype(np.float32), self.config.sample_rate
        elif hasattr(audio, "get_all_samples"):
            # datasets>=5 returns a torchcodec AudioDecoder, not a dict
            samples = audio.get_all_samples()
            data = samples.data
            waveform = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            waveform = waveform.astype(np.float32)
            sr = int(getattr(samples, "sample_rate", self.config.sample_rate))
        elif hasattr(audio, "numpy"):  # torch tensor
            waveform, sr = audio.numpy().astype(np.float32), self.config.sample_rate
        else:
            raise TypeError(f"unsupported audio object: {type(audio)!r}")

        if waveform.ndim > 1:  # (channels, time) or (time, channels)
            axis = 0 if waveform.shape[0] < waveform.shape[1] else 1
            waveform = waveform.mean(axis=axis)
        return waveform.astype(np.float32), int(sr)

    # ---------------------------------------------------------------- labels
    def _label_of(self, example: Dict[str, Any]) -> int:
        key = example.get(self.config.label_column)
        if key is None:
            return -1
        if isinstance(key, str):
            return LABEL_NAMES.index(key)
        return int(key)

    # ------------------------------------------------------------- map step
    def _extract_row(
        self, example: Dict[str, Any], pooling: Optional[str] = None
    ) -> Dict[str, Any]:
        waveform, sr = self._decode_audio(example.get(self.config.audio_column))
        feature = self.extract_features(waveform, sr)
        row: Dict[str, Any] = {
            "label": self._label_of(example),
            "speaker_id": example.get("speaker_id"),
            "system_id": example.get("system_id"),
            "audio_file_name": example.get("audio_file_name"),
        }
        if pooling is None:
            row["features"] = feature.astype(self.config.dtype)
        else:
            row["pooled"] = pool_feature_map(feature, pooling)
        return row

    def _output_features(self, raw, pooling: Optional[str] = None) -> Features:
        has_labels = self.config.label_column in getattr(raw, "column_names", [])
        label_type = ClassLabel(names=list(LABEL_NAMES)) if has_labels else Value("int8")
        if pooling is None:
            signal = {
                "features": Array2D(
                    shape=self.config.feature_shape, dtype=self.config.dtype
                )
            }
        else:
            signal = {"pooled": Sequence(Value(self.config.dtype))}
        return Features(
            {
                **signal,
                "label": label_type,
                "speaker_id": Value("string"),
                "system_id": Value("string"),
                "audio_file_name": Value("string"),
            }
        )

    # ------------------------------------------------------------- preparing
    def prepare(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        force: bool = False,
        pooling: Optional[str] = None,
    ):
        """Decode + featurise one split.

        Parameters
        ----------
        split: ``"train"``, ``"validation"`` or ``"test"``.
        limit: keep only the first ``limit`` utterances (debugging).
        filter_fn: predicate on a raw example, e.g. ``lambda ex: ex["key"] == 1``.
        force: recompute even if this split is already cached in memory.
        pooling: when given (see :func:`pool_feature_map`), store one fixed
            length vector per utterance in a ``pooled`` column instead of the
            ``(C, T)`` map. Far cheaper to keep on disk (~4 MB for the whole
            training split with ``mean_std`` MFCC), which is what you want
            before moving features to another machine.

        Returns
        -------
        ``datasets.Dataset`` (or ``IterableDataset`` when ``streaming=True``)
        with columns ``features`` (or ``pooled``), ``label``, ``speaker_id``,
        ``system_id``, ``audio_file_name``.
        """
        if pooling is not None and pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

        cache_key = (split, pooling)
        if cache_key in self._prepared and not force:
            return self._prepared[cache_key]

        if limit is None:
            limit = self.config.limit

        if self.config.normalize == "global" and self._stats is None:
            self._load_stats() or self.compute_normalization_stats()

        use_disk_cache = (
            self.config.cache_dir is not None
            and not self.config.streaming
            and filter_fn is None
        )
        cache_path = self._cache_path(split, limit, pooling) if use_disk_cache else None
        if (
            cache_path is not None
            and not force
            and (cache_path / "dataset_info.json").exists()
        ):
            prepared = load_from_disk(str(cache_path))
            self._prepared[cache_key] = prepared
            return prepared

        raw = self._load_raw(split)

        if limit is not None:
            if self.config.streaming:
                raw = raw.take(limit)
            else:
                limit = min(int(limit), len(raw))
                raw = raw.select(range(limit))

        if filter_fn is not None:
            raw = raw.filter(filter_fn)

        columns = list(getattr(raw, "column_names", []))
        map_kwargs: Dict[str, Any] = {}
        if not self.config.streaming:  # IterableDataset.map has no num_proc/desc
            map_kwargs = {
                "num_proc": self.config.num_proc,
                "desc": f"featurising {split}",
            }
        prepared = raw.map(
            self._extract_row,
            features=self._output_features(raw, pooling),
            remove_columns=columns,
            fn_kwargs={"pooling": pooling},
            **map_kwargs,
        )

        if cache_path is not None:
            if cache_path.exists():  # recomputation after force=True
                shutil.rmtree(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            prepared.save_to_disk(str(cache_path))

        self._prepared[cache_key] = prepared
        return prepared

    def _cache_path(
        self, split: str, limit: Optional[int] = None, pooling: Optional[str] = None
    ) -> Path:
        assert self.config.cache_dir is not None
        suffix = f"-n{limit}" if limit is not None else ""
        pool = f"-{pooling}" if pooling is not None else ""
        return Path(self.config.cache_dir) / (
            f"{split}-{self.config.fingerprint()}{pool}{suffix}"
        )

    # ------------------------------------------------------------- statistics
    def split_statistics(self, split: str) -> Dict[str, Any]:
        """Class balance / speaker / system counts for a single split.

        Only the label and metadata columns are touched, so no audio is decoded
        (though a non-streaming split still has to be materialised once).
        """
        raw = self._load_raw(split)
        if self.config.streaming:
            examples = list(raw.take(self.config.limit or 2000))
            keys = [ex.get(self.config.label_column) for ex in examples]
            speakers = [ex.get("speaker_id") for ex in examples]
            systems = [ex.get("system_id") for ex in examples]
            total = len(examples)
        else:
            columns = list(raw.column_names)
            keys = (
                raw[self.config.label_column]
                if self.config.label_column in columns
                else []
            )
            speakers = raw["speaker_id"] if "speaker_id" in columns else []
            systems = raw["system_id"] if "system_id" in columns else []
            total = len(raw)

        stats: Dict[str, Any] = {"total": int(total)}
        if len(keys):
            labels = np.asarray(
                [LABEL_NAMES.index(k) if isinstance(k, str) else int(k) for k in keys]
            )
            stats["bonafide"] = int(np.sum(labels == 0))
            stats["spoof"] = int(np.sum(labels == 1))
            stats["spoof_ratio"] = stats["spoof"] / max(total, 1)
        if len(speakers):
            stats["n_speakers"] = len({s for s in speakers if s is not None})
        if len(systems):
            unique = {s for s in systems if s is not None}
            stats["n_systems"] = len(unique)
            stats["n_attack_systems"] = len(unique - {"-"})
        return stats

    def label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Statistics for every split in ``config.splits`` (see :meth:`split_statistics`)."""
        return {split: self.split_statistics(split) for split in self.config.splits}

    def class_weights(self, split: str = "train", kind: str = "balanced") -> np.ndarray:
        """Inverse-frequency class weights for ``CrossEntropyLoss(weight=...)``."""
        stats = self.split_statistics(split)
        if "bonafide" not in stats:
            raise ValueError(f"no labels available for split {split!r}")
        counts = np.array([stats["bonafide"], stats["spoof"]], dtype=np.float64)
        if kind == "balanced":
            weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        elif kind == "inverse":
            weights = 1.0 / np.maximum(counts, 1.0)
        else:
            raise ValueError("kind must be 'balanced' or 'inverse'")
        return (weights / weights.mean()).astype(np.float32)

    def explore(self) -> None:
        """Print a compact summary of the corpus (no audio decoding)."""
        print(f"dataset: {self.config.dataset_name}")
        header = (
            f"{'split':<12}{'total':>8}{'bonafide':>10}{'spoof':>9}"
            f"{'spoof%':>8}{'speakers':>10}{'systems':>9}"
        )
        print(header)
        print("-" * len(header))
        for split, stats in self.label_statistics().items():
            print(
                f"{split:<12}{str(stats.get('total')):>8}"
                f"{stats.get('bonafide', '-'):>10}{stats.get('spoof', '-'):>9}"
                f"{stats.get('spoof_ratio', 0.0) * 100:>7.1f}%"
                f"{stats.get('n_speakers', '-'):>10}"
                f"{stats.get('n_systems', '-'):>9}"
            )
        print(
            f"\nfeatures: {self.config.feature} {self.config.feature_shape} "
            f"({self.config.dtype}), normalize={self.config.normalize}"
        )

    # ------------------------------------------------- global normalisation
    def compute_normalization_stats(
        self, split: str = "train", max_examples: Optional[int] = 2000
    ) -> Dict[str, np.ndarray]:
        """Estimate per-coefficient mean/std for ``normalize="global"``."""
        raw = self._load_raw(split)
        if self.config.streaming:
            rows = list(raw.take(max_examples or 1000))
        else:
            n = len(raw)
            if max_examples is not None:
                n = min(n, max_examples)
                idx = np.linspace(0, len(raw) - 1, n).astype(int).tolist() if n else []
            else:
                idx = list(range(n))
            rows = raw.select(idx)

        total = np.zeros(self.config.n_coeffs, dtype=np.float64)
        total_sq = np.zeros(self.config.n_coeffs, dtype=np.float64)
        count = 0
        for example in rows:
            waveform, sr = self._decode_audio(example[self.config.audio_column])
            if sr != self.config.sample_rate:
                waveform = self._resample(waveform, sr, self.config.sample_rate)
            feature = self.frontend(waveform).astype(np.float64)
            total += feature.sum(axis=1)
            total_sq += np.square(feature).sum(axis=1)
            count += feature.shape[1]

        if count == 0:
            raise RuntimeError("no examples were read while computing statistics")
        mean = total / count
        var = np.maximum(total_sq / count - mean**2, 0.0)
        self._stats = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
        }

        if self.config.cache_dir is not None:
            path = Path(self.config.cache_dir) / f"stats-{self.config.fingerprint()}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **self._stats)
        return self._stats

    def _load_stats(self) -> Optional[Dict[str, np.ndarray]]:
        if self._stats is not None:
            return self._stats
        if self.config.cache_dir is None:
            return None
        path = Path(self.config.cache_dir) / f"stats-{self.config.fingerprint()}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            self._stats = {k: data[k] for k in ("mean", "std")}
        return self._stats

    @property
    def normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_stats()

    # ------------------------------------------------------------- consumers
    def get_split(self, split: str = "train"):
        """Alias for :meth:`prepare` (keeps call sites short)."""
        return self.prepare(split)

    def get_numpy_dataset(self, split: str = "train", with_metadata: bool = False):
        """Thin NumPy wrapper (no torch needed)."""
        return NumpyFeatureDataset(self.prepare(split), with_metadata=with_metadata)

    def get_sklearn_dataset(
        self,
        split: str = "train",
        pooling: str = "mean_std",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(X[N, D], y[N])`` utterance vectors for classical ML baselines.

        The time axis is collapsed with :func:`pool_feature_map`, so ``mfcc``
        with ``pooling="mean_std"`` and ``n_mfcc=20`` gives 40 dimensions per
        utterance -- exactly what ``StandardScaler`` + ``SVC`` expect.

        Fit the scaler and the model on ``train`` only, then reuse them for
        ``validation`` / ``test``.

        .. warning::
           With ``normalize="per_utterance"`` every coefficient is already
           zero-mean and unit-variance *inside* each utterance, so mean/std
           pooling yields almost the same vector for every file. For SVM-style
           baselines use ``normalize="none"`` (recommended) or ``"global"``.
        """
        if self.config.normalize == "per_utterance" and pooling != "flat":
            warnings.warn(
                'normalize="per_utterance" + pooling="%s" pools features that '
                "are already standardised per utterance, so the vectors will be "
                'nearly constant. Use normalize="none" for SVM baselines.'
                % pooling,
                RuntimeWarning,
                stacklevel=2,
            )

        # `prepare(..., pooling=...)` pools during extraction, so the split is
        # stored as (N, D) vectors instead of (N, C, T) maps: the full training
        # split stays at a few MB instead of ~800 MB.
        dataset = self.prepare(split, pooling=pooling)

        if self.config.streaming:
            vectors, labels = [], []
            for row in dataset:
                vectors.append(np.asarray(row["pooled"], dtype=np.float32))
                labels.append(int(row["label"]))
            if not vectors:
                return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
            return np.stack(vectors).astype(np.float32), np.asarray(labels, dtype=np.int64)

        features = np.asarray(dataset["pooled"], dtype=np.float32)
        labels = np.asarray(dataset["label"], dtype=np.int64)
        if features.size == 0:
            return np.empty((0, 0), dtype=np.float32), labels
        return features.reshape(len(labels), -1), labels

    def get_torch_dataset(self, split: str = "train", with_metadata: bool = False):
        """``torch.utils.data.Dataset`` yielding ``(float32 tensor[C,T], int64)``."""
        if torch is None:
            raise RuntimeError("PyTorch is not installed: pip install torch")
        return TorchFeatureDataset(
            self.prepare(split),
            label_names=self.label_names,
            with_metadata=with_metadata,
        )

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        **loader_kwargs: Any,
    ):
        """Build a ``DataLoader`` over :meth:`get_torch_dataset`."""
        if torch is None:
            raise RuntimeError("PyTorch is not installed: pip install torch")
        if shuffle is None:
            shuffle = split == "train"
        return torch.utils.data.DataLoader(
            self.get_torch_dataset(split),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **loader_kwargs,
        )

    # ------------------------------------------------------------------ misc
    @staticmethod
    def label_name(label: int) -> str:
        return LABEL_NAMES[label] if 0 <= label < len(LABEL_NAMES) else "unknown"


class NumpyFeatureDataset:
    """Minimal indexable dataset: ``dataset[i] -> (np.ndarray[C,T], int)``."""

    def __init__(self, hf_dataset, with_metadata: bool = False) -> None:
        self.dataset = hf_dataset
        self.with_metadata = with_metadata

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        feature = np.asarray(row["features"], dtype=np.float32)
        label = int(row["label"])
        if not self.with_metadata:
            return feature, label
        return {
            "features": feature,
            "label": label,
            "speaker_id": row["speaker_id"],
            "system_id": row["system_id"],
            "audio_file_name": row["audio_file_name"],
        }

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stack the whole split into ``(X[N,C,T], y[N])`` arrays."""
        if len(self.dataset) == 0:
            raise ValueError("empty dataset")
        features = np.asarray(self.dataset["features"], dtype=np.float32)
        labels = np.asarray(self.dataset["label"], dtype=np.int64)
        return features, labels


class TorchFeatureDataset:
    """``torch.utils.data.Dataset`` wrapper around a prepared split.

    Defined at module level (rather than inside a method) so that it can be
    pickled by ``DataLoader(num_workers>0)`` on spawn-based platforms (macOS).
    """

    def __init__(
        self,
        hf_dataset,
        label_names: Tuple[str, ...] = LABEL_NAMES,
        with_metadata: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed: pip install torch")
        self.dataset = hf_dataset
        self.label_names = label_names
        self.with_metadata = with_metadata

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        feature = np.asarray(row["features"], dtype=np.float32)
        label = int(row["label"])
        if not self.with_metadata:
            return torch.from_numpy(feature), torch.tensor(label, dtype=torch.long)
        name = self.label_names[label] if 0 <= label < len(self.label_names) else "unknown"
        return {
            "features": torch.from_numpy(feature),
            "label": torch.tensor(label, dtype=torch.long),
            "label_name": name,
            "speaker_id": row["speaker_id"],
            "system_id": row["system_id"],
            "audio_file_name": row["audio_file_name"],
        }


# --------------------------------------------------------------------------- #
# CLI: quick sanity check, e.g.
#   python dataPreprocessor.py --limit 8 --feature lfcc
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA preprocessor demo")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--feature", default="logmel", choices=list(FEATURES))
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True,
                        help="stream rows instead of downloading a whole split (default)")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    pre = ASVspoofPreprocessor(
        feature=args.feature,
        n_mels=args.n_mels,
        limit=args.limit,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )
    print(pre)
    pre.explore()

    dataset = pre.prepare(args.split, limit=args.limit)
    row = next(iter(dataset)) if args.streaming else dataset[0]
    features = np.asarray(row["features"])
    print(
        f"\n{args.split}[0]: audio_file_name={row['audio_file_name']}, "
        f"speaker={row['speaker_id']}, system={row['system_id']}, "
        f"label={row['label']} ({pre.label_name(int(row['label']))})"
    )
    print(
        f"features: shape={features.shape}, dtype={features.dtype}, "
        f"min={features.min():.3f}, max={features.max():.3f}, "
        f"mean={features.mean():.3f}"
    )
    print(f"class weights: {pre.class_weights(args.split)}")


if __name__ == "__main__":
    _main()
