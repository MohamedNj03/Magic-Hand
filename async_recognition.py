"""Step 5/8 - the slow half of recognition, off the video loop.

submit() -> worker thread -> poll(). The worker never touches the canvas.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import sketch_recognition as sketch


@dataclass(frozen=True)
# --- Jobs and results ----------------------------------------------------
class RecognitionJob:
    object_id: str
    strokes_points: list[list[tuple[int, int]]]
    widths: list[int]


@dataclass(frozen=True)
class RecognitionResult:
    object_id: str
    text: str | None = None
    strokes: list[list[tuple[int, int]]] | None = None


def _default_recognize(job: RecognitionJob) -> RecognitionResult:
    written = sketch.recognize_text(job.strokes_points, job.widths)
    if written is None:
        return RecognitionResult(job.object_id)
    text, strokes = written
    return RecognitionResult(job.object_id, text, strokes)


# --- The service ---------------------------------------------------------
class RecognitionService:
    def __init__(self, recognize=_default_recognize, synchronous: bool = False) -> None:
        self._recognize = recognize
        self._synchronous = synchronous
        self._results: queue.Queue[RecognitionResult] = queue.Queue()
        self._jobs: queue.Queue[RecognitionJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._worker: threading.Thread | None = None

    def submit(self, object_id: str, strokes_points, widths) -> None:
        job = RecognitionJob(
            object_id=object_id,
            strokes_points=[list(pts) for pts in strokes_points],
            widths=list(widths),
        )
        with self._lock:
            self._pending += 1
        if self._synchronous:
            self._run(job)
        else:
            self._ensure_worker()
            self._jobs.put(job)

    def poll(self) -> list[RecognitionResult]:
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = threading.Event()
        step = 0.005
        waited = 0.0
        while self.pending > 0 and waited < timeout:
            deadline.wait(step)
            waited += step
        return self.pending == 0

    def close(self) -> None:
        worker = self._worker
        if worker is not None:
            self._jobs.put(None)
            worker.join(timeout=1.0)
            self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, name="recognition", daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._run(job)

    def _run(self, job: RecognitionJob) -> None:
        try:
            result = self._recognize(job)
        except Exception as exc:
            sketch.log(f"ECRITURE: reconnaissance abandonnee ({exc})")
            result = RecognitionResult(job.object_id)
        self._results.put(result)
        with self._lock:
            self._pending -= 1
