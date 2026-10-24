"""Streaming API Routes and WebSocket Handler.

Provides WebSocket /ws/ecg for high-throughput live ECG sample delivery
and REST endpoints to control replay, select records, and adjust speeds.
"""
import asyncio
import json
import os
import shutil
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

router = APIRouter(tags=["Streaming"])


class StreamStartRequest(BaseModel):
    record_id: str = Field(..., description="ID of the record to stream, e.g. 'mitdb_100'")
    speed: float = Field(1.0, ge=0.25, le=10.0, description="Playback speed multiplier")
    stride_sec: float = Field(1.0, ge=0.5, le=5.0, description="Sliding window inference stride in seconds")
    start_time_s: float = Field(0.0, ge=0.0, description="Start time position in seconds")


class StreamSeekRequest(BaseModel):
    timestamp_s: float = Field(..., ge=0.0, description="Target timestamp in seconds")


class StreamSpeedRequest(BaseModel):
    speed: float = Field(..., ge=0.25, le=10.0, description="Playback speed multiplier")


@router.get("/api/records")
async def list_records(request: Request):
    """Returns all available authentic clinical ECG records."""
    source_mgr = request.app.state.ecg_source
    return {"records": source_mgr.list_available_records()}


@router.get("/api/stream/status")
async def stream_status(request: Request):
    """Returns current streaming and buffer status."""
    stream_mgr = request.app.state.stream_manager
    return stream_mgr.get_status()


@router.post("/api/stream/start")
async def start_stream(req: StreamStartRequest, request: Request):
    """Starts live replay of a real ECG recording."""
    stream_mgr = request.app.state.stream_manager
    try:
        status = await stream_mgr.start(
            record_id=req.record_id,
            speed=req.speed,
            stride_sec=req.stride_sec,
            start_time_s=req.start_time_s,
        )
        return {"status": "ok", "stream": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/stream/pause")
async def pause_stream(request: Request):
    """Pauses current stream replay."""
    stream_mgr = request.app.state.stream_manager
    status = await stream_mgr.pause()
    return {"status": "ok", "stream": status}


@router.post("/api/stream/resume")
async def resume_stream(request: Request):
    """Resumes paused stream replay."""
    stream_mgr = request.app.state.stream_manager
    status = await stream_mgr.resume()
    return {"status": "ok", "stream": status}


@router.post("/api/stream/stop")
async def stop_stream(request: Request):
    """Stops stream replay and resets playback position."""
    stream_mgr = request.app.state.stream_manager
    status = await stream_mgr.stop()
    return {"status": "ok", "stream": status}


@router.post("/api/stream/seek")
async def seek_stream(req: StreamSeekRequest, request: Request):
    """Seeks to a specific timestamp in the active recording."""
    stream_mgr = request.app.state.stream_manager
    status = await stream_mgr.seek(req.timestamp_s)
    return {"status": "ok", "stream": status}


@router.post("/api/stream/speed")
async def set_speed(req: StreamSpeedRequest, request: Request):
    """Sets streaming playback speed."""
    stream_mgr = request.app.state.stream_manager
    status = await stream_mgr.set_speed(req.speed)
    return {"status": "ok", "stream": status}


@router.post("/api/upload")
async def upload_ecg_file(request: Request, file: UploadFile = File(...)):
    """Uploads a local CSV or NPZ ECG recording into data/uploads/ for safe local replay."""
    if not (file.filename.endswith(".csv") or file.filename.endswith(".npz")):
        raise HTTPException(status_code=400, detail="Only .csv and .npz ECG files are supported")

    dest_dir = request.app.state.ecg_source.user_upload_dir
    safe_filename = os.path.basename(file.filename)
    dest_path = os.path.join(dest_dir, safe_filename)

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "status": "ok",
        "record_id": f"upload_{safe_filename}",
        "message": f"Successfully uploaded {safe_filename} to local repository",
    }


@router.websocket("/ws/ecg")
async def websocket_ecg_endpoint(websocket: WebSocket):
    """Continuous WebSocket connection delivering live ECG chunks and inference events."""
    await websocket.accept()
    stream_mgr = websocket.app.state.stream_manager

    # Queue for client delivery
    client_queue = asyncio.Queue(maxsize=100)

    async def client_callback(msg: Dict[str, Any]) -> None:
        try:
            if not client_queue.full():
                client_queue.put_nowait(msg)
        except Exception:
            pass

    stream_mgr.subscribe(client_callback)
    sender_task = None

    try:
        # Send initial connection greeting with current stream status
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to SmartPace Live ECG Engine",
            "status": stream_mgr.get_status(),
        })

        # Task to pump queued messages to the WebSocket
        async def sender_loop():
            try:
                while True:
                    msg = await client_queue.get()
                    await websocket.send_json(msg)
                    client_queue.task_done()
            except Exception:
                pass

        sender_task = asyncio.create_task(sender_loop())

        # Listen for client control commands via WebSocket
        while True:
            text = await websocket.receive_text()
            try:
                cmd = json.loads(text)
                action = cmd.get("action")
                if action == "pause":
                    await stream_mgr.pause()
                elif action == "resume":
                    await stream_mgr.resume()
                elif action == "stop":
                    await stream_mgr.stop()
                elif action == "speed":
                    await stream_mgr.set_speed(float(cmd.get("speed", 1.0)))
                elif action == "seek":
                    await stream_mgr.seek(float(cmd.get("timestamp_s", 0.0)))
            except Exception as e:
                try:
                    await websocket.send_json({"type": "command_error", "detail": str(e)})
                except Exception:
                    break

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        stream_mgr.unsubscribe(client_callback)
        if sender_task is not None:
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass
