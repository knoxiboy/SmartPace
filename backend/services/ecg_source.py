"""Real ECG Data Source Service.

Loads authentic recorded ECG signals from the local dataset repository (e.g. MIT-BIH Arrhythmia
Database in WFDB format or preprocessed caches) and streams real physical sample values.
Never fabricates synthetic ECG data.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import wfdb

from smartpace import config as C
from smartpace.datasets import resample_to_fs, _pick_lead


class ECGSourceManager:
    """Manages real clinical ECG recordings for real-time streaming replay."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.mitdb_dir = os.path.join(data_dir, "mitdb")
        self.cache_dir = os.path.join(data_dir, "cache")
        self.user_upload_dir = os.path.join(data_dir, "uploads")
        os.makedirs(self.user_upload_dir, exist_ok=True)
        self._record_cache: Dict[str, Dict[str, Any]] = {}

    def list_available_records(self) -> List[Dict[str, Any]]:
        """Scans the local filesystem and returns all available real ECG recordings."""
        records = []

        # 1. MIT-BIH WFDB records in data/mitdb/
        if os.path.exists(self.mitdb_dir):
            hea_files = [f for f in os.listdir(self.mitdb_dir) if f.endswith(".hea")]
            for f in sorted(hea_files):
                rec_id = f[:-4]
                dat_path = os.path.join(self.mitdb_dir, f"{rec_id}.dat")
                if os.path.exists(dat_path):
                    # Check comments/description from header
                    desc = f"MIT-BIH Record #{rec_id} (Lead II / MLII)"
                    records.append({
                        "id": f"mitdb_{rec_id}",
                        "name": f"MIT-BIH {rec_id}",
                        "source": "MIT-BIH Arrhythmia Database",
                        "description": desc,
                        "format": "WFDB (physionet)",
                        "record_id": rec_id,
                        "type": "mitdb",
                    })

        # 2. Uploaded records in data/uploads/
        if os.path.exists(self.user_upload_dir):
            for f in sorted(os.listdir(self.user_upload_dir)):
                if f.endswith(".npz") or f.endswith(".csv"):
                    records.append({
                        "id": f"upload_{f}",
                        "name": f,
                        "source": "User Uploaded ECG",
                        "description": "Locally stored user ECG recording",
                        "format": f.split(".")[-1].upper(),
                        "record_id": f,
                        "type": "upload",
                    })

        return records

    def load_record(self, record_id: str) -> Dict[str, Any]:
        """Loads a real recording, resamples to 250 Hz, and returns physical signal samples."""
        clean_id = record_id.strip()
        if clean_id.startswith("MIT-BIH "):
            clean_id = f"mitdb_{clean_id.replace('MIT-BIH ', '').strip()}"
        elif not clean_id.startswith("mitdb_") and not clean_id.startswith("upload_"):
            if os.path.exists(os.path.join(self.mitdb_dir, f"{clean_id}.hea")):
                clean_id = f"mitdb_{clean_id}"

        if clean_id in self._record_cache:
            return self._record_cache[clean_id]

        if clean_id.startswith("mitdb_"):
            raw_id = clean_id.replace("mitdb_", "")
            path = os.path.join(self.mitdb_dir, raw_id)
            if not (os.path.exists(path + ".hea") and os.path.exists(path + ".dat")):
                raise FileNotFoundError(f"MIT-BIH record files not found for '{raw_id}' at {path}")

            rec = wfdb.rdrecord(path)
            lead_idx = _pick_lead(rec)
            lead_name = rec.sig_name[lead_idx] if rec.sig_name else "MLII"
            raw_signal = rec.p_signal[:, lead_idx].astype(np.float64)
            fs_orig = float(rec.fs)

            # Resample to common SmartPace model sampling frequency (250 Hz)
            signal_250hz = resample_to_fs(raw_signal, fs_in=fs_orig, fs_out=C.FS)
            duration_s = len(signal_250hz) / C.FS

            # Read annotation rhythm intervals if present
            annotations_summary = []
            try:
                ann = wfdb.rdann(path, "atr")
                for s, a in zip(ann.sample, ann.aux_note):
                    if a and a.strip("\x00").startswith("("):
                        rhythm = a.strip("\x00").strip()
                        t_sec = float(s) / fs_orig
                        annotations_summary.append({"time_s": t_sec, "rhythm": rhythm})
            except Exception:
                pass

            data = {
                "id": record_id,
                "name": f"MIT-BIH {raw_id}",
                "lead": lead_name,
                "fs": C.FS,
                "fs_orig": fs_orig,
                "samples": signal_250hz,
                "total_samples": len(signal_250hz),
                "duration_seconds": duration_s,
                "annotations": annotations_summary,
                "source": "MIT-BIH Arrhythmia Database",
            }
            self._record_cache[record_id] = data
            return data

        elif record_id.startswith("upload_"):
            filename = record_id.replace("upload_", "")
            path = os.path.join(self.user_upload_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Uploaded record '{filename}' not found at {path}")

            if filename.endswith(".npz"):
                npz = np.load(path)
                sig = npz["x"] if "x" in npz else npz[list(npz.keys())[0]]
                sig = np.asarray(sig, dtype=np.float64).flatten()
            elif filename.endswith(".csv"):
                sig = np.loadtxt(path, delimiter=",").flatten().astype(np.float64)
            else:
                raise ValueError("Unsupported upload format")

            duration_s = len(sig) / C.FS
            data = {
                "id": record_id,
                "name": filename,
                "lead": "Lead II (User Input)",
                "fs": C.FS,
                "fs_orig": C.FS,
                "samples": sig,
                "total_samples": len(sig),
                "duration_seconds": duration_s,
                "annotations": [],
                "source": "Local User Upload",
            }
            self._record_cache[record_id] = data
            return data

        else:
            raise ValueError(f"Unknown record prefix for '{record_id}'")
