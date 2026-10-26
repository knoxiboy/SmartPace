"""Live ECG Streaming Engine and Replay Manager.

Asynchronously streams physical ECG recordings at user-configurable speeds (0.5x, 1x, 2x, 5x),
maintains sliding 10-second windows (2500 samples @ 250 Hz), runs real SmartPace inference,
and broadcasts messages via WebSockets.
"""
import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np

from smartpace import config as C
from .ecg_source import ECGSourceManager
from .event_store import EventStore
from smartpace.adapter import SmartPaceEngine


class StreamManager:
    """Controls the live ECG streaming loop, window buffering, and event dispatch."""

    def __init__(
        self,
        ecg_source: ECGSourceManager,
        engine: SmartPaceEngine,
        event_store: EventStore,
    ):
        self.source = ecg_source
        self.engine = engine
        self.event_store = event_store

        # Playback state
        self.current_record: Optional[Dict[str, Any]] = None
        self.record_id: Optional[str] = None
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.speed: float = 1.0
        self.stride_sec: float = 1.0        # Stride between inferences (1s = 250 samples)
        self.current_sample_idx: int = 0
        self.window_buffer: List[float] = []

        # Background task
        self._stream_task: Optional[asyncio.Task] = None
        self._subscribers: List[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = []
        self._lock = asyncio.Lock()

    def subscribe(self, callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcasts a structured JSON event to all connected WebSocket clients."""
        if not self._subscribers:
            return
        tasks = []
        for cb in list(self._subscribers):
            tasks.append(cb(message))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def start(
        self,
        record_id: str,
        speed: float = 1.0,
        stride_sec: float = 1.0,
        start_time_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Starts real ECG replay for the given record."""
        async with self._lock:
            # Stop existing stream if running
            await self._stop_internal()

            self.current_record = self.source.load_record(record_id)
            self.record_id = record_id
            self.speed = max(0.25, min(10.0, float(speed)))
            self.stride_sec = max(0.5, min(5.0, float(stride_sec)))
            self.current_sample_idx = max(0, min(int(start_time_s * C.FS), self.current_record["total_samples"] - 1))
            self.window_buffer = []
            self.is_playing = True
            self.is_paused = False

            # Spawn background streaming loop
            self._stream_task = asyncio.create_task(self._streaming_loop())

        await self.broadcast({
            "type": "stream_started",
            "record_id": record_id,
            "record_name": self.current_record["name"],
            "lead": self.current_record["lead"],
            "fs": C.FS,
            "duration_s": self.current_record["duration_seconds"],
            "speed": self.speed,
            "start_time_s": float(self.current_sample_idx / C.FS),
        })

        return self.get_status()

    async def pause(self) -> Dict[str, Any]:
        async with self._lock:
            if self.is_playing and not self.is_paused:
                self.is_paused = True
        await self.broadcast({"type": "stream_paused", "current_time_s": float(self.current_sample_idx / C.FS)})
        return self.get_status()

    async def resume(self) -> Dict[str, Any]:
        async with self._lock:
            if self.is_playing and self.is_paused:
                self.is_paused = False
        await self.broadcast({"type": "stream_resumed", "current_time_s": float(self.current_sample_idx / C.FS)})
        return self.get_status()

    async def stop(self) -> Dict[str, Any]:
        async with self._lock:
            await self._stop_internal()
        await self.broadcast({"type": "stream_stopped"})
        return self.get_status()

    async def _stop_internal(self) -> None:
        self.is_playing = False
        self.is_paused = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    async def seek(self, timestamp_s: float) -> Dict[str, Any]:
        async with self._lock:
            if self.current_record:
                total_samples = self.current_record["total_samples"]
                self.current_sample_idx = max(0, min(int(timestamp_s * C.FS), total_samples - 1))
                self.window_buffer = []
        await self.broadcast({"type": "stream_seeked", "current_time_s": float(self.current_sample_idx / C.FS)})
        return self.get_status()

    async def set_speed(self, speed: float) -> Dict[str, Any]:
        async with self._lock:
            self.speed = max(0.25, min(10.0, float(speed)))
        await self.broadcast({"type": "stream_speed_changed", "speed": self.speed})
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current streaming and buffer status."""
        current_time = float(self.current_sample_idx / C.FS) if self.current_record else 0.0
        duration = float(self.current_record["duration_seconds"]) if self.current_record else 1800.0
        return {
            "is_playing": self.is_playing,
            "is_streaming": self.is_playing,
            "is_paused": self.is_paused,
            "record_id": self.record_id or "mitdb_100",
            "record_name": self.current_record["name"] if self.current_record else "MIT-BIH 100",
            "speed": self.speed,
            "playback_speed": self.speed,
            "current_time_s": current_time,
            "elapsed_sec": current_time,
            "duration_s": duration,
            "total_duration_sec": duration,
            "progress_pct": float(current_time / duration * 100.0) if duration > 0 else 0.0,
            "current_sample_idx": self.current_sample_idx,
            "current_sample": self.current_sample_idx,
            "total_samples": self.current_record["total_samples"] if self.current_record else 450000,
            "active_subscribers": len(self._subscribers),
            "active_clients": len(self._subscribers),
            "mode": "replay",
            "window_count": int(current_time / self.stride_sec) if self.stride_sec > 0 else 0,
            "buffer_samples": len(self.window_buffer),
            "target_window_samples": C.WIN,
        }

    async def _streaming_loop(self) -> None:
        """Main async loop delivering sample chunks and triggering window inference."""
        # We deliver chunks of 25 samples every 100ms at 1.0x (250 Hz)
        chunk_size = 25
        base_interval_s = chunk_size / C.FS  # 0.10s = 100ms

        last_inference_sample = self.current_sample_idx
        stride_samples = int(self.stride_sec * C.FS)

        try:
            while self.is_playing:
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue

                all_samples = self.current_record["samples"]
                total = len(all_samples)

                if self.current_sample_idx >= total:
                    # Loop back or finish
                    self.current_sample_idx = 0
                    self.window_buffer = []
                    await self.broadcast({"type": "stream_restarted"})

                end_idx = min(self.current_sample_idx + chunk_size, total)
                chunk = all_samples[self.current_sample_idx:end_idx]
                chunk_start_s = float(self.current_sample_idx / C.FS)
                self.current_sample_idx = end_idx

                # Append to sliding window buffer
                self.window_buffer.extend(chunk.tolist())
                if len(self.window_buffer) > C.WIN * 2:
                    self.window_buffer = self.window_buffer[-C.WIN:]

                # 1. Send ECG Chunk for live waveform rendering
                await self.broadcast({
                    "type": "ecg_chunk",
                    "timestamp": time.time(),
                    "record_time_s": chunk_start_s,
                    "samples": chunk.tolist(),
                    "fs": C.FS,
                })

                # 2. Check if we have accumulated at least a full 10-second window (2500 samples)
                # and stepped past the configured stride
                samples_since_infer = self.current_sample_idx - last_inference_sample
                if len(self.window_buffer) >= C.WIN and (samples_since_infer >= stride_samples or last_inference_sample == 0):
                    last_inference_sample = self.current_sample_idx
                    raw_window = np.array(self.window_buffer[-C.WIN:], dtype=np.float64)

                    # Run inference in a separate thread to prevent blocking the event loop
                    window_start_sec = max(0.0, float((self.current_sample_idx - C.WIN) / C.FS))
                    window_end_sec = float(self.current_sample_idx / C.FS)

                    inference_result = await asyncio.to_thread(
                        self.engine.analyze_window,
                        raw_window=raw_window,
                        use_onnx_edge=True,
                        compute_explanation=False,  # On-demand or for key windows to keep CPU responsive
                    )

                    event_payload = {
                        "type": "inference_result",
                        "event_id": f"ev_{int(time.time()*1000)}",
                        "record_id": self.record_id,
                        "window_start_sec": window_start_sec,
                        "window_end_sec": window_end_sec,
                        "raw_window": raw_window.tolist(),
                        **inference_result,
                    }

                    # Store event in history
                    self.event_store.add_event(event_payload)

                    # Broadcast inference decision to frontend
                    await self.broadcast(event_payload)

                # Sleep according to playback speed
                sleep_duration = base_interval_s / max(0.25, self.speed)
                await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[StreamManager] Streaming error: {e}")
            await self.broadcast({"type": "stream_error", "message": str(e)})
        finally:
            self.is_playing = False
