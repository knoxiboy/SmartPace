"""SmartPace Live — Incremental, Causal Pan-Tompkins R-Peak Detector.

Processes incoming streaming ECG chunks with persistent filter and threshold state.
Does not require reprocessing past windows and guarantees strictly causal operation
(no access to future samples beyond the current incoming chunk).
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as sps


@dataclass
class PeakEvent:
    """Represents a validated, localized R-peak detected by the causal detector."""
    sample_index: int              # Estimated physiological R-peak sample index in recording
    timestamp: float               # Signal time in seconds (sample_index / fs)
    detection_sample_idx: int      # Exact absolute sample index when detection was confirmed
    detection_time_s: float        # Timestamp when peak was emitted (detection_sample_idx / fs)
    amplitude: float               # Raw ECG amplitude at peak
    delay_samples: int             # Causal detection confirmation delay in samples
    delay_ms: float                # Causal detection delay in milliseconds
    confidence: float              # Relative signal-to-noise peak confidence [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalPanTompkins:
    """Stateful online R-peak detector implementing causal Pan-Tompkins signal processing.

    Stages:
    1. Causal IIR Bandpass Filter (5-15 Hz, 2nd order Butterworth SOS).
    2. 5-point causal derivative filter: (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4]) / (8*T).
    3. Squaring function: x[n]^2.
    4. Moving-Window Integration (MWI): causal boxcar window of 150 ms (37 samples at 250 Hz).
    5. Adaptive dual-thresholding with refractory period suppression (200 ms).
    """

    def __init__(self, fs: int = 250):
        self.fs = int(fs)
        self.refractory_samples = int(0.200 * self.fs)  # 200ms refractory period (50 samples @ 250Hz)
        self.window_samples = max(1, int(0.150 * self.fs))  # 150ms MWI window (37 samples @ 250Hz)

        # 1. Bandpass filter design: 5 to 15 Hz
        nyq = self.fs / 2.0
        low = max(0.5, min(5.0, nyq - 1.0))
        high = min(15.0, nyq - 0.5)
        self.sos = sps.butter(2, [low, high], btype="bandpass", fs=self.fs, output="sos")

        self.reset()

    def reset(self, initial_sample_idx: int = 0) -> None:
        """Resets all streaming delay lines, filter conditions, and adaptive thresholds."""
        # Filter state
        self.zi = sps.sosfilt_zi(self.sos) * 0.0

        # Derivative state: past 4 filtered samples
        self.deriv_hist = [0.0, 0.0, 0.0, 0.0]

        # Moving window integration circular buffer
        self.mwi_buf = [0.0] * self.window_samples
        self.mwi_sum = 0.0

        # Raw sample buffer for peak localization and amplitude readout
        self.raw_buf: List[Tuple[int, float]] = []

        # Adaptive thresholds (Pan-Tompkins signal and noise running averages)
        self.spki = 0.25
        self.npki = 0.02
        self.threshold = 0.08

        # In-flight peak tracker
        self.in_peak = False
        self.peak_max_mwi = 0.0
        self.peak_max_sample_idx = 0
        self.last_peak_sample_idx = initial_sample_idx - self.refractory_samples - 100

    def process_chunk(self, chunk: np.ndarray, start_sample_idx: int) -> List[PeakEvent]:
        """Processes a chunk of successive ECG samples incrementally.

        Args:
            chunk: 1D array of raw ECG samples (arbitrary length >= 1).
            start_sample_idx: Absolute sample index corresponding to chunk[0].

        Returns:
            List of PeakEvent objects confirmed during this chunk.
        """
        arr = np.asarray(chunk, dtype=np.float64).flatten()
        if len(arr) == 0:
            return []

        events: List[PeakEvent] = []

        for offset, raw_val in enumerate(arr):
            abs_idx = start_sample_idx + offset

            # Check finiteness
            if not np.isfinite(raw_val):
                raw_val = 0.0

            # 1. Causal SOS IIR Bandpass Filtering
            y_arr, self.zi = sps.sosfilt(self.sos, [raw_val], zi=self.zi)
            y = float(y_arr[0])

            # 2. 5-point causal derivative
            # d[n] = (2*y[n] + y[n-1] - y[n-3] - 2*y[n-4]) * (fs / 8)
            d = (
                2.0 * y
                + self.deriv_hist[0]
                - self.deriv_hist[2]
                - 2.0 * self.deriv_hist[3]
            ) * (self.fs / 8.0)
            self.deriv_hist = [y, self.deriv_hist[0], self.deriv_hist[1], self.deriv_hist[2]]

            # 3. Squaring
            sq = d * d

            # 4. Moving-Window Integration (MWI)
            old_sq = self.mwi_buf.pop(0)
            self.mwi_buf.append(sq)
            self.mwi_sum = max(0.0, self.mwi_sum - old_sq + sq)
            mwi = self.mwi_sum / self.window_samples

            # Store recent raw history for exact R-peak refinement
            self.raw_buf.append((abs_idx, raw_val))
            if len(self.raw_buf) > (self.window_samples * 4):
                self.raw_buf.pop(0)

            # 5. Dual-Threshold Adaptive Peak Search
            if mwi > self.threshold:
                if not self.in_peak:
                    self.in_peak = True
                    self.peak_max_mwi = mwi
                    self.peak_max_sample_idx = abs_idx
                elif mwi > self.peak_max_mwi:
                    self.peak_max_mwi = mwi
                    self.peak_max_sample_idx = abs_idx
            elif self.in_peak:
                # Pulse descended below threshold -> confirm and localize peak
                # Refractory period check: physiologically at least 200 ms between distinct beats
                if (abs_idx - self.last_peak_sample_idx) >= self.refractory_samples:
                    # Search back for physiological R-peak in raw signal within [peak_max - W - 8, peak_max]
                    search_lo = self.peak_max_sample_idx - self.window_samples - 8
                    search_hi = self.peak_max_sample_idx
                    candidates = [item for item in self.raw_buf if search_lo <= item[0] <= search_hi]

                    if candidates:
                        # Find absolute extremum (tallest positive or dominant deflection)
                        best = max(candidates, key=lambda c: abs(c[1]))
                        r_sample_idx = best[0]
                        r_amplitude = best[1]

                        # Calculate detection latency
                        delay_samples = abs_idx - r_sample_idx
                        delay_ms = (delay_samples / self.fs) * 1000.0

                        # Calculate confidence from signal-to-noise ratio
                        confidence = min(1.0, max(0.1, float(self.peak_max_mwi / (self.threshold + 1e-6))))

                        event = PeakEvent(
                            sample_index=r_sample_idx,
                            timestamp=float(r_sample_idx / self.fs),
                            detection_sample_idx=abs_idx,
                            detection_time_s=float(abs_idx / self.fs),
                            amplitude=float(r_amplitude),
                            delay_samples=int(delay_samples),
                            delay_ms=round(delay_ms, 2),
                            confidence=round(confidence, 3),
                        )
                        events.append(event)

                        self.last_peak_sample_idx = r_sample_idx

                        # Update signal peak running average
                        self.spki = 0.125 * self.peak_max_mwi + 0.875 * self.spki
                    else:
                        # Noise pulse without clear candidate in window
                        self.npki = 0.125 * self.peak_max_mwi + 0.875 * self.npki
                else:
                    # Inside refractory period: treat as noise pulse
                    self.npki = 0.125 * self.peak_max_mwi + 0.875 * self.npki

                # Update adaptive threshold: THRI = NPKI + 0.25 * (SPKI - NPKI)
                self.threshold = max(0.01, self.npki + 0.25 * (self.spki - self.npki))
                self.in_peak = False
                self.peak_max_mwi = 0.0

        return events
