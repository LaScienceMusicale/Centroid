"""Signal analysis utilities for the Centroid visualizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SpectralFeatures:
    """Container for spectral descriptors used by the visualizer."""

    centroid: float
    spread: float
    flatness: float
    tonality: float
    flux: float
    rms: float


class SpectralFeatureExtractor:
    """Compute spectral descriptors from audio frames.

    The extractor operates on mono audio buffers sampled at ``sample_rate`` and
    keeps track of the previous magnitude spectrum to compute spectral flux and
    apply simple exponential smoothing to the RMS envelope.
    """

    def __init__(self, sample_rate: int, smoothing: float = 0.6) -> None:
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1")
        self.sample_rate = sample_rate
        self._prev_mag: Optional[np.ndarray] = None
        self._prev_rms: Optional[float] = None
        self._rms_smoothing = smoothing

    def process(self, frame: np.ndarray) -> SpectralFeatures:
        """Return the spectral features for ``frame``.

        Parameters
        ----------
        frame:
            Mono audio buffer. The samples are expected to be in the range
            ``[-1, 1]``.
        """

        if frame.ndim != 1:
            raise ValueError("frame must be mono (1-D array)")

        windowed = frame * np.hanning(len(frame))
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)
        power = mag**2

        freq = np.fft.rfftfreq(len(frame), d=1.0 / self.sample_rate)
        mag_sum = np.sum(mag) + 1e-12

        centroid = float(np.sum(freq * mag) / mag_sum)
        spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * mag) / mag_sum))

        # Spectral flatness using the classic geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(power + 1e-12)))
        arithmetic_mean = np.mean(power + 1e-12)
        flatness = float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))
        tonality = float(np.clip(1.0 - flatness, 0.0, 1.0))

        if self._prev_mag is None:
            flux = 0.0
        else:
            norm_prev = self._prev_mag / (np.linalg.norm(self._prev_mag) + 1e-12)
            norm_curr = mag / (np.linalg.norm(mag) + 1e-12)
            diff = norm_curr - norm_prev
            flux = float(np.sqrt(np.sum(diff**2)))
        self._prev_mag = mag

        rms = float(np.sqrt(np.mean(frame**2)))
        if self._prev_rms is None:
            smoothed_rms = rms
        else:
            alpha = self._rms_smoothing
            smoothed_rms = alpha * rms + (1.0 - alpha) * self._prev_rms
        self._prev_rms = smoothed_rms

        return SpectralFeatures(
            centroid=centroid,
            spread=spread,
            flatness=flatness,
            tonality=tonality,
            flux=flux,
            rms=smoothed_rms,
        )
