"""Realtime 3D visualization driven by spectral descriptors."""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

from queue import Empty, SimpleQueue

import numpy as np
import sounddevice as sd
import soundfile as sf
from vispy import app, scene

from .analysis import SpectralFeatureExtractor, SpectralFeatures


@dataclass
class TimbralPoint:
    """Representation of a single point in the trajectory."""

    position: np.ndarray
    flux: float
    rms: float
    age: float = 0.0
    tonality: float = 0.0
    centroid: float = 0.0
    spread: float = 0.0


class AudioReactiveVisualizer:
    """Visualize live audio in a 3D constellation inspired by Lucio Arese."""

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_size: int = 1024,
        hop_size: int = 1024,
        history_seconds: float = 10.0,
        audio_path: Optional[str] = None,
    ) -> None:
        self._default_sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.history_seconds = history_seconds

        self._queue: SimpleQueue[SpectralFeatures] = SimpleQueue()
        self.history: Deque[TimbralPoint] = deque()

        self._canvas = scene.SceneCanvas(keys="interactive", bgcolor="#050608", size=(1024, 768), show=True)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(fov=60.0, elevation=30.0, azimuth=45.0)
        self._view.camera.distance = 4.0

        grid = scene.visuals.XYZAxis(parent=self._view.scene)
        grid.transform = scene.transforms.STTransform(scale=(1.5, 1.5, 1.5))

        self._markers = scene.visuals.Markers(parent=self._view.scene)
        self._markers.set_gl_state(depth_test=True, blend=True, blend_func=("src_alpha", "one_minus_src_alpha"))

        self._line = scene.visuals.Line(connect="strip", method="gl", parent=self._view.scene)
        self._line.set_gl_state(depth_test=True, blend=True, blend_func=("src_alpha", "one_minus_src_alpha"))

        self._timer = app.Timer(interval=1.0 / 60.0, connect=self._on_timer, start=True)

        self._stream: Optional[sd.InputStream] = None
        self._file_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.audio_path = Path(audio_path).expanduser() if audio_path else None

        native_canvas = getattr(self._canvas, "native", None)
        if native_canvas is not None and hasattr(native_canvas, "setAcceptDrops"):
            native_canvas.setAcceptDrops(True)
        drop_event = getattr(self._canvas.events, "drop", None)
        if drop_event is not None:
            drop_event.connect(self._on_drop)  # type: ignore[attr-defined]

        self._reset_visual_state()
        self._configure_sample_rate(sample_rate)
        self._set_history_length(history_seconds)

        if self.audio_path:
            self.load_audio_file(str(self.audio_path))
        else:
            self._start_microphone_stream()

    # ------------------------------------------------------------------
    # Audio handling
    def _set_history_length(self, history_seconds: float) -> None:
        max_points = int(math.ceil(history_seconds * self.sample_rate / self.hop_size))
        self.history = deque(self.history, maxlen=max_points)

    # ------------------------------------------------------------------
    def _reset_visual_state(self) -> None:
        self.history.clear()
        self._queue = SimpleQueue()

    def _configure_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.extractor = SpectralFeatureExtractor(sample_rate)
        self._buffer = np.zeros(self.frame_size, dtype=np.float32)
        self._buffer_offset = 0
        self._set_history_length(self.history_seconds)

    # ------------------------------------------------------------------
    def _start_microphone_stream(self) -> None:
        self._stop_audio_sources()
        self._reset_visual_state()
        self._configure_sample_rate(self._default_sample_rate)
        self.audio_path = None
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.hop_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _start_file_thread(self, path: Path) -> None:
        self._stop_audio_sources()
        info = sf.info(str(path))
        self._reset_visual_state()
        self._configure_sample_rate(info.samplerate)
        self.audio_path = path
        channels = info.channels

        def worker() -> None:
            hop_duration = self.hop_size / self.sample_rate
            next_time = time.perf_counter()
            try:
                with sf.SoundFile(str(path), "r") as audio_file:
                    while not self._stop_event.is_set():
                        data = audio_file.read(self.hop_size, dtype="float32", always_2d=True)
                        if len(data) == 0:
                            break
                        samples = data.mean(axis=1) if channels > 1 else data[:, 0]
                        self._process_samples(samples)

                        next_time += hop_duration
                        sleep_time = next_time - time.perf_counter()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
            finally:
                self._stop_event.clear()

        self._stop_event.clear()
        self._file_thread = threading.Thread(target=worker, daemon=True)
        self._file_thread.start()

    def _stop_audio_sources(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None
        if self._file_thread is not None and self._file_thread.is_alive():
            self._stop_event.set()
            self._file_thread.join(timeout=1.0)
            self._stop_event.clear()
        self._file_thread = None

    # ------------------------------------------------------------------
    def _process_samples(self, samples: np.ndarray) -> None:
        idx = 0
        while idx < len(samples):
            remaining = self.frame_size - self._buffer_offset
            take = min(remaining, len(samples) - idx)
            end = idx + take
            self._buffer[self._buffer_offset : self._buffer_offset + take] = samples[idx:end]
            self._buffer_offset += take
            idx = end

            if self._buffer_offset == self.frame_size:
                frame = self._buffer.copy()
                self._queue.put(self.extractor.process(frame))

                if self.hop_size < self.frame_size:
                    self._buffer[:-self.hop_size] = self._buffer[self.hop_size :]
                    self._buffer_offset = self.frame_size - self.hop_size
                else:
                    self._buffer_offset = 0

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
        if status:
            print(status)
        samples = indata[:, 0].astype(np.float32)
        self._process_samples(samples)

    # ------------------------------------------------------------------
    def load_audio_file(self, path: str) -> None:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        self.audio_path = file_path
        self._start_file_thread(file_path)

    # ------------------------------------------------------------------
    def _on_drop(self, event) -> None:  # pragma: no cover - GUI callback
        paths = getattr(event, "paths", None)
        if not paths:
            return
        try:
            self.load_audio_file(paths[0])
            print(f"Loaded audio file: {paths[0]}")
        except Exception as exc:  # pragma: no cover - runtime feedback
            print(f"Failed to load audio file: {exc}")

    # ------------------------------------------------------------------
    def _on_timer(self, event) -> None:
        updated = False
        while True:
            try:
                features = self._queue.get_nowait()
            except Empty:
                break
            else:
                self._append_features(features)
                updated = True

        dt = event.dt if event is not None and event.dt is not None else 1.0 / 60.0
        self._age_history(dt)

        if updated or self.history:
            self._update_visuals()

    def _append_features(self, features: SpectralFeatures) -> None:
        nyquist = self.sample_rate / 2.0
        tonality = features.tonality
        centroid_norm = np.clip(features.centroid / nyquist, 0.0, 1.0)
        spread_norm = np.clip(features.spread / nyquist, 0.0, 1.0)

        position = np.array([
            (tonality - 0.5) * 2.0,
            (centroid_norm - 0.5) * 2.0,
            (spread_norm - 0.5) * 2.0,
        ], dtype=np.float32)

        self.history.append(
            TimbralPoint(
                position=position,
                flux=features.flux,
                rms=features.rms,
                tonality=tonality,
                centroid=centroid_norm,
                spread=spread_norm,
            )
        )
    def _age_history(self, dt: float) -> None:
        for point in self.history:
            point.age += dt

    def _update_visuals(self) -> None:
        if not self.history:
            return

        points = list(self.history)
        positions = np.array([p.position for p in points], dtype=np.float32)

        flux_values = np.array([p.flux for p in points], dtype=np.float32)
        rms_values = np.array([p.rms for p in points], dtype=np.float32)

        flux_range = max(np.percentile(flux_values, 95), 1e-3)
        flux_norm = np.clip(flux_values / flux_range, 0.0, 1.0)

        rms_norm = np.clip(rms_values / (np.max(rms_values) + 1e-6), 0.1, 1.0)

        ages = np.array([p.age for p in points], dtype=np.float32)
        max_age = float(np.max(ages)) + 1e-6
        age_norm = np.clip(1.0 - ages / max_age, 0.0, 1.0)

        # Color gradient from deep purple to fiery orange
        base_color = np.array([78, 52, 255], dtype=np.float32) / 255.0
        peak_color = np.array([255, 120, 0], dtype=np.float32) / 255.0
        colors_rgb = base_color + flux_norm[:, None] * (peak_color - base_color)
        alpha = np.clip(age_norm * (0.3 + 0.7 * rms_norm), 0.0, 1.0)
        colors = np.concatenate([colors_rgb, alpha[:, None]], axis=1).astype(np.float32)

        sizes = 6.0 + 18.0 * flux_norm
        self._markers.set_data(
            positions,
            face_color=colors,
            size=sizes,
            edge_width=0.0,
            symbol="square",
        )

        # Build the trajectory line with matching colors
        self._line.set_data(
            pos=positions,
            color=colors,
            width=2.0 + 6.0 * float(flux_norm[-1]),
        )

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the audio stream and run the visualizer event loop."""

        try:
            app.run()
        finally:
            self._stop_audio_sources()
