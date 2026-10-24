"""Event Store Service for SmartPace Live.

Stores real-time inference decisions, reliability metrics, and waveform metadata
in an in-memory ring-buffer with optional local SQLite persistence.
No patient identifiers or telemetry data are transmitted to external services.
"""
from collections import deque
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional


class EventStore:
    """Thread-safe event repository for streaming inference history."""

    def __init__(self, max_memory_events: int = 500, db_path: Optional[str] = "data/event_history.db"):
        self.max_memory_events = max_memory_events
        self.events: deque = deque(maxlen=max_memory_events)
        self.db_path = db_path
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS inference_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp REAL,
                        record_id TEXT,
                        window_start_sec REAL,
                        window_end_sec REAL,
                        predicted_class TEXT,
                        confidence REAL,
                        predictive_entropy REAL,
                        action TEXT,
                        is_accepted INTEGER,
                        rejection_reason TEXT,
                        mahalanobis_distance REAL,
                        sqi_score REAL,
                        heart_rate_bpm REAL,
                        model_latency_ms REAL,
                        total_latency_ms REAL,
                        details_json TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"[EventStore] SQLite init warning: {e}")

    def add_event(self, event: Dict[str, Any]) -> None:
        """Stores a completed inference event."""
        event_id = event.get("event_id", f"ev_{int(time.time()*1000)}_{len(self.events)}")
        event["event_id"] = event_id
        event["stored_at"] = time.time()
        self.events.append(event)

        if self.db_path:
            try:
                rel = event.get("reliability", {})
                lat = event.get("latency", {})
                vitals = event.get("vitals", {})
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO inference_events VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        event_id,
                        event["stored_at"],
                        event.get("record_id", "unknown"),
                        float(event.get("window_start_sec", 0.0)),
                        float(event.get("window_end_sec", 0.0)),
                        event.get("predicted_class", "UNKNOWN"),
                        float(event.get("confidence", 0.0)),
                        float(event.get("predictive_entropy", 0.0)),
                        rel.get("action", "ACCEPT"),
                        1 if rel.get("is_accepted", True) else 0,
                        rel.get("reason", ""),
                        float(rel.get("mahalanobis_distance", 0.0)),
                        float(rel.get("sqi_score", 1.0)),
                        float(vitals.get("heart_rate_bpm") or 0.0),
                        float(lat.get("model_inference_ms", 0.0)),
                        float(lat.get("total_processing_ms", 0.0)),
                        json.dumps({
                            "class_probabilities": event.get("class_probabilities", {}),
                            "explanation_summary": event.get("explanation_summary", {}),
                        }),
                    ))
                    conn.commit()
            except Exception as e:
                print(f"[EventStore] SQLite insert warning: {e}")

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent events in reverse chronological order."""
        res = list(self.events)
        return list(reversed(res[-limit:]))

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Finds an event by ID from memory."""
        for ev in reversed(self.events):
            if ev.get("event_id") == event_id:
                return ev
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates live summary statistics across stored events."""
        events = list(self.events)
        if not events:
            return {
                "total_events": 0,
                "accepted_count": 0,
                "abstained_count": 0,
                "acceptance_rate": 1.0,
                "mean_latency_ms": 0.0,
                "class_counts": {},
            }

        acc = sum(1 for e in events if e.get("reliability", {}).get("is_accepted", True))
        rej = len(events) - acc
        latencies = [e.get("latency", {}).get("total_processing_ms", 0.0) for e in events]
        classes: Dict[str, int] = {}
        for e in events:
            cls = e.get("predicted_class", "UNKNOWN")
            classes[cls] = classes.get(cls, 0) + 1

        return {
            "total_events": len(events),
            "accepted_count": acc,
            "abstained_count": rej,
            "acceptance_rate": float(acc / len(events)),
            "mean_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
            "class_counts": classes,
        }

    def clear(self) -> None:
        self.events.clear()
