"""
async_recognition.py — La reconnaissance d'écriture, HORS de la boucle vidéo.

LE PROBLÈME, MESURÉ
===================
`sketch.recognize_text()` coûte 330 ms sur cette machine : Tesseract est un
processus séparé, il faut le lancer, lui passer l'image, attendre. Le
chemin hors-ligne (gabarits) coûte 100 ms de plus. Ce travail était fait
DANS `process_frame()`, donc dans la boucle caméra, donc à l'image près où
l'utilisateur finit son trait : dix images perdues d'un coup, exactement au
moment où il regarde le résultat. C'est le à-coup le plus visible de toute
l'application.

LA FORME DE LA CORRECTION
=========================
L'élément est créé TOUT DE SUITE, en dessin propre — la forme géométrique,
elle, se reconnaît en quelques microsecondes et reste synchrone. La lecture
de l'écriture part dans un fil de fond ; quand elle revient, l'élément se
transforme sur place, sans avoir jamais bloqué l'affichage.

Le fil ne touche JAMAIS le canevas : il dépose son résultat dans une file,
et c'est `process_frame` — donc le fil principal, donc un seul propriétaire
des données — qui l'applique. C'est ce qui rend la chose sûre sans un seul
verrou sur l'état de l'application.

Mode `synchronous=True` : le travail est fait immédiatement à la
soumission, le résultat attend dans la même file. Les tests l'utilisent
pour rester déterministes — même code, même chemin, zéro fil.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import sketch_recognition as sketch


@dataclass(frozen=True)
class RecognitionJob:
    object_id: str
    strokes_points: list[list[tuple[int, int]]]
    widths: list[int]


@dataclass(frozen=True)
class RecognitionResult:
    """`text` à None = abstention : le tracé reste le dessin d'origine."""
    object_id: str
    text: str | None = None
    strokes: list[list[tuple[int, int]]] | None = None


def _default_recognize(job: RecognitionJob) -> RecognitionResult:
    written = sketch.recognize_text(job.strokes_points, job.widths)
    if written is None:
        return RecognitionResult(job.object_id)
    text, strokes = written
    return RecognitionResult(job.object_id, text, strokes)


class RecognitionService:
    """Une file d'entrée, un fil, une file de sortie. Rien de plus : un seul
    fil suffit largement (on valide un élément toutes les quelques
    secondes au mieux) et garantit que les résultats sortent dans l'ordre
    où les éléments ont été dessinés."""

    def __init__(self, recognize=_default_recognize, synchronous: bool = False) -> None:
        self._recognize = recognize
        self._synchronous = synchronous
        self._results: queue.Queue[RecognitionResult] = queue.Queue()
        self._jobs: queue.Queue[RecognitionJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._worker: threading.Thread | None = None

    # -- côté application ---------------------------------------------------
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
        """Tous les résultats disponibles, sans jamais attendre. Appelée une
        fois par image ; renvoyer une liste (et pas un seul résultat) évite
        qu'une rafale de validations prenne une image chacune pour revenir."""
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
        """Attend que tout soit reconnu. Réservé aux tests et à la
        fermeture : la boucle vidéo, elle, n'attend jamais."""
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

    # -- côté fil de fond ---------------------------------------------------
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
        """Un échec de reconnaissance ne doit JAMAIS remonter jusqu'à la
        boucle vidéo : on renvoie une abstention, l'élément reste le dessin
        que l'utilisateur a tracé, et l'application continue."""
        try:
            result = self._recognize(job)
        except Exception as exc:  # noqa: BLE001 — un OCR peut échouer de mille façons
            sketch.log(f"ECRITURE: reconnaissance abandonnee ({exc})")
            result = RecognitionResult(job.object_id)
        self._results.put(result)
        with self._lock:
            self._pending -= 1
