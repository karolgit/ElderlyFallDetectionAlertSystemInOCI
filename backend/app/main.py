from __future__ import annotations

import os
import logging
import sys
import signal
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from .device import get_torch_device, summarize_device
from .pose import PoseEstimator
from .fall import FallDetector
from .schemas import FrameAnalyzeRequest, FrameAnalyzeResponse, VideoAnalyzeResponse
from .utils import data_url_to_image, pil_to_cv2_rgb
from .draw import draw_skeleton

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: register signal handlers
    try:
        signal.signal(signal.SIGINT, _set_stop_event)
        signal.signal(signal.SIGTERM, _set_stop_event)
        logger.debug("Signal handlers for SIGINT/SIGTERM registered.")
    except Exception as e:
        logger.debug("Signal registration skipped: %s", e)
    # Yield control to the application
    yield
    # Shutdown: set stop event and join workers
    _STOP_EVENT.set()
    logger.debug("Shutdown event set; loops will stop. Joining worker threads...")
    with _WORK_LOCK:
        workers_snapshot = list(_WORKERS)
    for t in workers_snapshot:
        try:
            t.join(timeout=5.0)
        except Exception:
            pass
    logger.debug("Worker join complete.")

app = FastAPI(title="Fall Detection Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global stop event for graceful shutdown (Ctrl+C)
from threading import Event
import signal

_STOP_EVENT = Event()


def _set_stop_event(*_args):
    try:
        logger.debug("Stop signal received; setting stop event.")
        sys.exit(0)
    except Exception:
        pass
    _STOP_EVENT.set()

# Worker tracking for graceful shutdown
from threading import Thread, Lock, current_thread
_WORKERS: List[Thread] = []
_WORK_LOCK = Lock()


def _register_worker(t: Thread) -> None:
    with _WORK_LOCK:
        _WORKERS.append(t)


def _unregister_current_worker() -> None:
    me = current_thread()
    with _WORK_LOCK:
        try:
            _WORKERS.remove(me)  # type: ignore[arg-type]
        except ValueError:
            pass


# (startup/shutdown) handled via lifespan above


_estimators: Dict[str, PoseEstimator] = {}
_fall_detector = FallDetector()


def get_estimator(preferred: str | None) -> PoseEstimator:
    device, device_type = get_torch_device(preferred)
    if device_type not in _estimators:
        logger.debug("Creating PoseEstimator for device_type=%s", device_type)
        _estimators[device_type] = PoseEstimator(preferred_device=device_type)
    return _estimators[device_type]


@app.get("/health")
def health() -> dict:
    device, device_type = get_torch_device(os.getenv("PREFERRED_DEVICE"))
    info = summarize_device(device)
    logger.debug("Health check device info: %s", info)
    return {"status": "ok", "device": info}


@app.post("/analyze_frame", response_model=FrameAnalyzeResponse)
async def analyze_frame(req: FrameAnalyzeRequest) -> FrameAnalyzeResponse:
    logger.debug("/analyze_frame called. preferred_device=%s", req.preferred_device)
    try:
        estimator = get_estimator(req.preferred_device)
        device_info = summarize_device(estimator.device)

        image = data_url_to_image(req.image_base64)
        logger.debug("Decoded image to size=%s", getattr(image, 'size', None))
        people = estimator.estimate(image)
        logger.debug("Pose estimator returned %d people", len(people))
        is_fall, fall_score = _fall_detector.predict(people)
        logger.debug("Fall detection: is_fall=%s score=%.3f", is_fall, fall_score)

        return FrameAnalyzeResponse(
            device=device_info,
            people=people,
            is_fall=is_fall,
            fall_score=fall_score,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /analyze_frame: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_video", response_model=VideoAnalyzeResponse)
async def analyze_video(
    file: UploadFile = File(...),
) -> VideoAnalyzeResponse:
    logger.debug("/analyze_video called. filename=%s content_type=%s", file.filename, file.content_type)
    if _STOP_EVENT.is_set():
        raise HTTPException(status_code=503, detail="Server stopping")
    preferred_device = os.getenv("PREFERRED_DEVICE")
    try:
        estimator = get_estimator(preferred_device)
        device_info = summarize_device(estimator.device)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            logger.debug("Saved temp video to %s, size=%d bytes", tmp.name, len(content))

            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Failed to read video")

            analyzed = 0
            fall_frames: List[int] = []
            scores: List[float] = []
            frame_idx = 0
            while not _STOP_EVENT.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % 3 != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                image = Image.fromarray(rgb)

                people = estimator.estimate(image)
                is_fall, fall_score = _fall_detector.predict(people)
                analyzed += 1
                scores.append(fall_score)
                if is_fall:
                    fall_frames.append(frame_idx)

            cap.release()
            if _STOP_EVENT.is_set():
                logger.debug("Analyze video interrupted by stop event.")
                raise HTTPException(status_code=503, detail="Server stopping")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /analyze_video: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    logger.debug("Video analyzed frames=%d any_fall=%s avg_score=%.3f", analyzed, len(fall_frames) > 0, avg_score)
    return VideoAnalyzeResponse(
        device=device_info,
        analyzed_frames=analyzed,
        any_fall=len(fall_frames) > 0,
        fall_frames=fall_frames,
        average_fall_score=avg_score,
    )


@app.post("/annotate_video")
async def annotate_video(
    file: UploadFile = File(...),
) -> FileResponse:
    logger.debug("/annotate_video called. filename=%s content_type=%s", file.filename, file.content_type)
    if _STOP_EVENT.is_set():
        raise HTTPException(status_code=503, detail="Server stopping")
    preferred_device = os.getenv("PREFERRED_DEVICE")
    import tempfile

    try:
        estimator = get_estimator(preferred_device)
        in_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        content = await file.read()
        in_tmp.write(content)
        in_tmp.flush()
        in_path = in_tmp.name
        in_tmp.close()
        logger.debug("Saved temp input video to %s (size=%d)", in_path, len(content))

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to read video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = out_tmp.name
        out_tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        from PIL import Image
        while not _STOP_EVENT.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            people = estimator.estimate(image)
            draw_skeleton(frame, people)
            writer.write(frame)
            processed += 1
        cap.release()
        writer.release()

        if _STOP_EVENT.is_set():
            logger.debug("Annotate video interrupted by stop event.")
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            raise HTTPException(status_code=503, detail="Server stopping")

        filename = (file.filename or 'annotated').rsplit('.', 1)[0] + '_annotated.mp4'
        logger.debug("Annotated video written to %s frames=%d", out_path, processed)
        return FileResponse(out_path, media_type='video/mp4', filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /annotate_video: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Async annotate with progress
import tempfile
import uuid

_JOBS: Dict[str, Dict[str, object]] = {}
_JOBS_LOCK = Lock()


def _start_job_state(filename: str, total_frames: Optional[int]) -> str:
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "processed": 0,
            "total": total_frames,
            "error": None,
            "out_path": None,
            "filename": filename,
        }
    return job_id


def _update_job(job_id: str, processed: int, total: Optional[int] = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["processed"] = processed
            if total is not None:
                job["total"] = total


def _finish_job(job_id: str, out_path: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["status"] = "done"
            job["out_path"] = out_path


def _error_job(job_id: str, message: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["status"] = "error"
            job["error"] = message


def _annotate_worker(job_id: str, content: bytes, orig_filename: str, preferred_device: Optional[str]) -> None:
    try:
        estimator = get_estimator(preferred_device)
        in_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        in_tmp.write(content)
        in_tmp.flush()
        in_path = in_tmp.name
        in_tmp.close()

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            _error_job(job_id, "Failed to read video")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0 else None
        _update_job(job_id, processed=0, total=total)

        out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = out_tmp.name
        out_tmp.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        from PIL import Image
        processed = 0
        while not _STOP_EVENT.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            people = estimator.estimate(image)
            draw_skeleton(frame, people)
            writer.write(frame)
            processed += 1
            if processed % 5 == 0:
                _update_job(job_id, processed=processed)
        cap.release()
        writer.release()
        if _STOP_EVENT.is_set():
            logger.debug("Annotate worker interrupted by stop event.")
            _error_job(job_id, "Server stopping")
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            return
        _update_job(job_id, processed=processed)
        _finish_job(job_id, out_path)
    except Exception as e:
        logger.exception("Annotate worker error: %s", e)
        _error_job(job_id, str(e))
    finally:
        _unregister_current_worker()


@app.post("/annotate_video_async")
async def annotate_video_async(file: UploadFile = File(...)) -> dict:
    logger.debug("/annotate_video_async called. filename=%s", file.filename)
    if _STOP_EVENT.is_set():
        raise HTTPException(status_code=503, detail="Server stopping")
    preferred_device = os.getenv("PREFERRED_DEVICE")
    content = await file.read()

    # Probe total frames quickly
    try:
        import tempfile as _tf
        tmp = _tf.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(content)
        tmp.flush()
        probe_path = tmp.name
        tmp.close()
        cap = cv2.VideoCapture(probe_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else None
        cap.release()
    except Exception:
        total = None

    job_id = _start_job_state(file.filename or "video.mp4", total)
    t = Thread(target=_annotate_worker, args=(job_id, content, file.filename or "video.mp4", preferred_device), daemon=False)
    _register_worker(t)
    t.start()
    return {"job_id": job_id}


@app.get("/annotate_progress/{job_id}")
async def annotate_progress(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        processed = int(job["processed"])  # type: ignore[index]
        total = job.get("total")
        status = job.get("status")
        error = job.get("error")
    percent = None
    if isinstance(total, int) and total > 0:
        percent = float(min(100.0, max(0.0, (processed / total) * 100.0)))
    return {"status": status, "processed": processed, "total": total, "percent": percent, "error": error}


@app.get("/annotate_result/{job_id}")
async def annotate_result(job_id: str) -> FileResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.get("status") != "done":
            raise HTTPException(status_code=409, detail="job not finished")
        out_path = job.get("out_path")
        filename = job.get("filename")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=410, detail="result expired or missing")
    download_name = (str(filename).rsplit('.', 1)[0] if filename else 'annotated') + '_annotated.mp4'
    return FileResponse(str(out_path), media_type='video/mp4', filename=download_name)
