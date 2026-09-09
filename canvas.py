"""
canvas.py — Ce qui est dessiné : les traits, les éléments 3D, et le calque
sur lequel tout ça vit.

Séparé de la logique de gestes parce que ce sont deux métiers différents :
ici on ne sait rien des mains, on sait seulement ce qu'est un trait, ce
qu'est un élément, et comment les dessiner. Tout ce fichier se teste sans
caméra et sans MediaPipe.

TROIS RÈGLES QUI GOUVERNENT CE FICHIER
======================================
1. La COULEUR appartient à l'élément. La sélection et le survol passent
   donc par le CADRE autour, jamais par la teinte du dessin — sinon
   recolorer un élément sélectionné ne changerait rien à l'écran.

2. L'ORDRE D'AFFICHAGE SUIT LA PILE. Ce qui est posé sur un autre élément
   est dessiné après lui, donc au-dessus. Et `find_object_at` désigne le
   même élément que celui qu'on voit dessus : sans cette règle, reprendre
   l'objet du haut d'une pile attrapait le support une fois sur deux, au
   hasard de la distance des centres.

3. RIEN N'EST PERDU SANS RETOUR. Effacer tout garde une photo de l'état
   précédent : le geste le plus destructeur de l'application doit être le
   plus facile à annuler.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

import mesh3d
import sketch_recognition as sketch


# Palette du changement de couleur, en BGR. Le vert vif est volontairement
# absent : c'est la couleur du cadre de sélection, un élément qui prendrait
# cette teinte deviendrait ambigu à l'écran.
COLOR_PALETTE = [
    (0, 220, 255),    # ambre (couleur d'origine)
    (255, 170, 0),    # bleu ciel
    (120, 100, 255),  # rouge corail
    (255, 120, 220),  # violet
    (80, 230, 255),   # jaune pâle
    (255, 220, 120),  # cyan clair
    (140, 160, 255),  # rose saumon
    (200, 255, 160),  # menthe
]

KIND_LABELS = {
    "circle": "cercle",
    "triangle": "triangle",
    "square": "carre",
    "rectangle": "rectangle",
    "pentagon": "pentagone",
    "hexagon": "hexagone",
    "text": "lettre",
    "freeform": "dessin",
}

SELECTION_COLOR = (60, 255, 120)
HOVER_COLOR = (180, 180, 180)
LINK_COLOR = (110, 110, 110)

# Ombre portée
SHADOW_STRENGTH = 0.5   # 0 = invisible, 1 = noir total
SHADOW_BLUR = 13        # noyau de flou (sur le masque en demi-résolution)

MAX_STACK_DEPTH = 32    # garde-fou : une chaîne d'empilement plus longue serait une boucle


# --------------------------------------------------------------------------
# Trait
# --------------------------------------------------------------------------
@dataclass
class Stroke:
    id: str
    points: list[tuple[int, int]]
    timestamp: float
    width: int = 4
    type: str = "draw"


def new_stroke(points, timestamp: float, width: int = 4) -> Stroke:
    return Stroke(id=str(uuid.uuid4()), points=list(points), timestamp=timestamp, width=width)


# --------------------------------------------------------------------------
# Élément 3D
# --------------------------------------------------------------------------
def _mesh_depth(bbox_span: float, kind: str) -> float:
    """Épaisseur donnée à l'extrusion 3D. Une caméra RGB ne peut pas savoir
    quelle profondeur l'utilisateur avait en tête : c'est une valeur par
    défaut raisonnable (proportionnelle à la taille), pas une mesure."""
    if kind in sketch.SHAPE_KINDS:
        return max(18.0, bbox_span * 0.35)   # solide plein : bien visible en volume
    return max(8.0, bbox_span * 0.08)        # ruban (dessin libre / lettre) : fin, comme du papier


def build_object_mesh(kind: str, strokes: list[Stroke], centroid: tuple[float, float]) -> mesh3d.Mesh3D:
    cx, cy = centroid
    all_pts = [p for s in strokes for p in s.points]
    xs = [p[0] for p in all_pts] or [cx]
    ys = [p[1] for p in all_pts] or [cy]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    depth = _mesh_depth(span, kind)

    if kind in sketch.SHAPE_KINDS and len(strokes) == 1:
        pts = strokes[0].points
        local_poly = [(x - cx, y - cy) for x, y in pts[:-1]]  # dernier point = doublon de fermeture
        if len(local_poly) >= 3:
            return mesh3d.make_prism_mesh(local_poly, depth)

    parts = []
    for s in strokes:
        local_pts = [(x - cx, y - cy) for x, y in s.points]
        parts.append(mesh3d.make_ribbon_mesh(local_pts, depth))
    return mesh3d.combine_meshes(parts)


@dataclass
class DrawnObject:
    id: str
    strokes: list[Stroke]
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0])
    roll: float = 0.0       # rotation dans le plan de l'image (radians)
    yaw: float = 0.0        # rotation autour de l'axe vertical, VRAIE 3D (radians)
    pitch: float = 0.0      # rotation autour de l'axe horizontal (radians)
    scale: float = 1.0      # redimensionnement (1.0 = taille d'origine)
    selected: bool = False  # posé par process_frame, jamais par le rendu
    kind: str = "freeform"  # "freeform" | "text" | un nom de sketch.SHAPE_KINDS
    mesh: mesh3d.Mesh3D | None = None   # construit une fois, mis en cache (voir ensure_mesh)
    index: int = 0                      # numéro d'affichage, stable (#1, #2, ...)
    text: str | None = None             # texte reconnu, s'il y en a un
    color_idx: int = 0                  # position dans COLOR_PALETTE
    spin: float = 0.0                   # rad/s : élan de rotation conservé après un lâcher
    attached_to: str | None = None      # id de l'élément SOUS celui-ci, quand il y est posé
    created_at: float = 0.0
    awaiting_recognition: bool = False  # l'écriture est en cours de lecture dans le fil de fond
    _bbox_cache: tuple[int, int, int, int] | None = field(default=None, repr=False, compare=False)
    _bbox_key: tuple | None = field(default=None, repr=False, compare=False)

    @property
    def color(self) -> tuple[int, int, int]:
        return COLOR_PALETTE[self.color_idx % len(COLOR_PALETTE)]

    @property
    def label(self) -> str:
        """Étiquette courte et lisible, pour que chaque élément soit
        IDENTIFIÉ à l'écran — sans elle, deux dessins libres se ressemblent
        trop pour qu'on sache lequel on manipule."""
        base = KIND_LABELS.get(self.kind, self.kind)
        if self.awaiting_recognition:
            return f"#{self.index} {base} ..."
        if self.kind == "text" and self.text:
            return f"#{self.index} {base} {self.text}"
        return f"#{self.index} {base}"

    def local_points(self) -> list[tuple[int, int]]:
        pts: list[tuple[int, int]] = []
        for s in self.strokes:
            pts.extend(s.points)
        return pts

    def centroid(self) -> tuple[float, float]:
        pts = self.local_points()
        if not pts:
            return (0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def ensure_mesh(self) -> mesh3d.Mesh3D:
        if self.mesh is None:
            self.mesh = build_object_mesh(self.kind, self.strokes, self.centroid())
        return self.mesh

    def invalidate_mesh(self) -> None:
        """À appeler dès que les TRAITS changent (reconnaissance d'écriture
        arrivée après coup). Le maillage ET la boîte englobante en cache
        décrivent les anciens traits : les garder afficherait la nouvelle
        lettre avec l'ancienne silhouette de collision."""
        self.mesh = None
        self._bbox_cache = None
        self._bbox_key = None

    def replace_strokes(self, strokes: list[Stroke]) -> None:
        """Change les traits SANS déplacer l'élément à l'écran.

        Les nouveaux traits n'ont pas le même barycentre que les anciens, et
        `offset` est relatif à ce barycentre : remplacer naïvement ferait
        sauter la lettre reconnue de quelques pixels au moment exact où elle
        apparaît. On rattrape l'écart dans `offset`."""
        cx, cy = self.center_screen()
        self.strokes = strokes
        self.invalidate_mesh()
        new_cx, new_cy = self.centroid()
        self.offset[0] = cx - new_cx
        self.offset[1] = cy - new_cy

    def center_screen(self) -> tuple[float, float]:
        cx, cy = self.centroid()
        return cx + self.offset[0], cy + self.offset[1]

    def transformed_bbox(self) -> tuple[int, int, int, int]:
        """bbox écran, mise en cache par transformation : le hit-test, le
        rendu, le survol et l'étiquette la demandent tous plusieurs fois par
        frame et par élément, or elle coûte une rotation + une projection de
        tous les sommets."""
        key = (self.roll, self.yaw, self.pitch, self.scale, self.offset[0], self.offset[1], id(self.mesh))
        if self._bbox_key != key or self._bbox_cache is None:
            self._bbox_cache = mesh3d.screen_bounds(
                self.ensure_mesh(), self.roll, self.yaw, self.center_screen(), self.scale, self.pitch
            )
            self._bbox_key = key
        return self._bbox_cache

    def contains_point(self, px: float, py: float, margin: int = 15) -> bool:
        x0, y0, x1, y1 = self.transformed_bbox()
        return (x0 - margin) <= px <= (x1 + margin) and (y0 - margin) <= py <= (y1 + margin)


# --------------------------------------------------------------------------
# Canevas — traits en attente (pas encore un élément) vs éléments validés.
# --------------------------------------------------------------------------
class Canvas:
    def __init__(self) -> None:
        self.pending_strokes: list[Stroke] = []
        self.objects: list[DrawnObject] = []
        self._action_log: list[tuple[str, object]] = []   # ("stroke" | "object" | "clear", donnée)
        self._redo_log: list[tuple[str, object]] = []
        self._active: Stroke | None = None
        self._counter: int = 0                            # numérotation stable des éléments

    # -- trait en cours -----------------------------------------------------
    @property
    def active_stroke(self) -> Stroke | None:
        """Le trait en train d'être tracé, ou None. Exposé — au lieu de
        laisser l'application lire `_active` — parce que trois endroits en
        avaient besoin et allaient le chercher dans les entrailles de la
        classe : le genre de détail qui rend toute refonte impossible."""
        return self._active

    @property
    def is_drawing(self) -> bool:
        return self._active is not None

    def start_stroke(self, t: float, width: int = 4) -> None:
        self._active = new_stroke([], t, width)

    def add_point(self, x: int, y: int) -> None:
        """Ajoute un point au trait en cours. Les doublons consécutifs sont
        ignorés : une main immobile en produit des dizaines par seconde, et
        ils ne font qu'alourdir la simplification puis le maillage."""
        if self._active is None:
            return
        point = (int(x), int(y))
        if self._active.points and self._active.points[-1] == point:
            return
        self._active.points.append(point)

    def end_stroke(self) -> None:
        if self._active is not None and len(self._active.points) >= 2:
            self.pending_strokes.append(self._active)
            self._action_log.append(("stroke", self._active))
            self._redo_log.clear()
        self._active = None

    # -- groupe en attente --------------------------------------------------
    def pending_bbox(self) -> tuple[float, float, float, float] | None:
        pts = [p for s in self.pending_strokes for p in s.points]
        if self._active is not None:
            pts.extend(self._active.points)
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def distance_to_pending(self, x: float, y: float) -> float:
        """Distance d'un point au groupe de traits en attente (0 si dedans).
        Sert à décider si un nouveau trait prolonge le dessin en cours ou en
        commence un autre."""
        box = self.pending_bbox()
        if box is None:
            return math.inf
        x0, y0, x1, y1 = box
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return math.hypot(dx, dy)

    # -- validation ---------------------------------------------------------
    def validate(self, created_at: float = 0.0, color_idx: int = 0, recognizer=None) -> DrawnObject | None:
        """Regroupe les traits en attente en un élément manipulable en 3D.

        Deux vitesses, et c'est tout l'intérêt du découpage :
          * la FORME géométrique se reconnaît en quelques microsecondes,
            donc ici, tout de suite ;
          * l'ÉCRITURE coûte 300 ms, donc elle part dans `recognizer` (un
            fil de fond) et l'élément naît en dessin propre, quitte à
            devenir une lettre une demi-seconde plus tard.

        Sans `recognizer`, tout est fait sur place : c'est le chemin simple,
        celui des tests."""
        if not self.pending_strokes:
            return None

        kept = sketch.clean_strokes(self.pending_strokes)
        if not kept:
            self.pending_strokes = []
            return None  # ne restait que du bruit : rien à valider

        widths = [orig.width for orig, _ in kept]
        points_lists = [pts for _, pts in kept]
        base_ts = kept[0][0].timestamp
        base_width = kept[0][0].width

        text: str | None = None
        awaiting = False
        shape = sketch.classify_shape_only(points_lists)
        if shape is not None:
            kind, display_points = shape
        elif recognizer is not None:
            kind, display_points = "freeform", points_lists
            awaiting = True
        else:
            kind, display_points, text = sketch.classify_ex(points_lists, widths)

        self._counter += 1
        obj = DrawnObject(
            id=str(uuid.uuid4()),
            strokes=[new_stroke(pts, base_ts, base_width) for pts in display_points],
            kind=kind,
            index=self._counter,
            text=text,
            color_idx=color_idx,
            created_at=created_at,
            awaiting_recognition=awaiting,
        )
        self.objects.append(obj)
        self.pending_strokes = []
        while self._action_log and self._action_log[-1][0] == "stroke":
            self._action_log.pop()
        self._action_log.append(("object", obj))
        self._redo_log.clear()

        if awaiting:
            recognizer.submit(obj.id, points_lists, widths)
        return obj

    def apply_recognition(self, result) -> DrawnObject | None:
        """Applique un résultat venu du fil de fond. Appelée UNIQUEMENT
        depuis la boucle principale : le fil de reconnaissance, lui, ne
        touche jamais au canevas."""
        obj = self.get_object(result.object_id)
        if obj is None:
            return None   # l'élément a été supprimé pendant la lecture : rien à faire
        obj.awaiting_recognition = False
        if not result.text or not result.strokes:
            return obj    # abstention : le dessin de l'utilisateur reste tel quel
        base = obj.strokes[0] if obj.strokes else None
        timestamp = base.timestamp if base is not None else obj.created_at
        width = base.width if base is not None else 4
        obj.replace_strokes([new_stroke(pts, timestamp, width) for pts in result.strokes])
        obj.kind = "text"
        obj.text = result.text
        return obj

    # -- annuler / refaire --------------------------------------------------
    def undo(self) -> str | None:
        """Défait la dernière action et dit LAQUELLE — l'appelant peut ainsi
        l'annoncer correctement à l'écran ("TOUT RESTAURE" n'est pas
        "ANNULE"). Renvoie None quand il n'y a plus rien à défaire."""
        while self._action_log:
            kind, item = self._action_log.pop()
            if kind == "object" and item in self.objects:
                self.objects.remove(item)          # type: ignore[arg-type]
                self._redo_log.append((kind, item))
                self.detach_orphans()
                return "object"
            if kind == "stroke" and item in self.pending_strokes:
                self.pending_strokes.remove(item)  # type: ignore[arg-type]
                self._redo_log.append((kind, item))
                return "stroke"
            if kind == "clear":
                objects, pending = item            # type: ignore[misc]
                self.objects = list(objects)
                self.pending_strokes = list(pending)
                self._redo_log.clear()
                return "clear"
            # entrée périmée (l'élément a été supprimé depuis) : on continue
        return None

    def redo(self) -> None:
        if not self._redo_log:
            return
        kind, item = self._redo_log.pop()
        if kind == "object":
            self.objects.append(item)          # type: ignore[arg-type]
        elif kind == "stroke":
            self.pending_strokes.append(item)  # type: ignore[arg-type]
        else:
            return
        self._action_log.append((kind, item))

    # -- suppression --------------------------------------------------------
    def delete_object(self, object_id: str) -> DrawnObject | None:
        for i, obj in enumerate(self.objects):
            if obj.id == object_id:
                self.objects.pop(i)
                self._action_log = [entry for entry in self._action_log if entry[1] is not obj]
                self._redo_log.append(("object", obj))
                self.detach_orphans()
                return obj
        return None

    def delete_last_object(self) -> DrawnObject | None:
        """Supprime le DERNIER élément créé. Différent de `undo`, qui défait
        la dernière ACTION (parfois un trait pas encore validé) : ici on vise
        explicitement un élément."""
        if not self.objects:
            return None
        return self.delete_object(self.objects[-1].id)

    def clear(self) -> None:
        """Tout effacer — en gardant une photo de l'état précédent, pour que
        le geste le plus destructeur de l'application reste le plus facile à
        annuler (voir `undo`)."""
        if self.objects or self.pending_strokes:
            self._action_log.append(("clear", (list(self.objects), list(self.pending_strokes))))
        self._active = None
        self.objects = []
        self.pending_strokes = []
        self._redo_log.clear()

    # -- recherche ----------------------------------------------------------
    def get_object(self, object_id: str | None) -> DrawnObject | None:
        if object_id is None:
            return None
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def stack_depth(self, obj: DrawnObject) -> int:
        """Combien d'éléments il y a SOUS celui-ci. 0 = posé au sol."""
        depth, current = 0, obj
        while current.attached_to is not None and depth < MAX_STACK_DEPTH:
            current = self.get_object(current.attached_to)
            if current is None:
                break
            depth += 1
        return depth

    def find_object_at(self, px: float, py: float, margin: int = 15) -> DrawnObject | None:
        """L'élément visé par un point. En cas de chevauchement on prend
        CELUI QU'ON VOIT DESSUS — c'est-à-dire le plus haut de la pile — et
        seulement à égalité de hauteur, celui dont le centre est le plus
        proche.

        Bug réel corrigé ici : sans la hiérarchie, reprendre l'objet posé
        sur un autre attrapait le support une fois sur deux, puisque les
        deux centres sont presque confondus. La règle est maintenant la
        même que ce que l'écran montre."""
        candidates = [o for o in self.objects if o.contains_point(px, py, margin)]
        if not candidates:
            return None

        def rank(o: DrawnObject) -> tuple[int, float]:
            ox, oy = o.center_screen()
            return (-self.stack_depth(o), (ox - px) ** 2 + (oy - py) ** 2)

        return min(candidates, key=rank)

    def descendants(self, object_id: str) -> list[DrawnObject]:
        """Tous les éléments posés SUR celui-ci, en cascade. Déplacer une
        pile doit emporter ce qui est dessus, et ce qui est dessus de ce qui
        est dessus."""
        found: list[DrawnObject] = []
        frontier = [object_id]
        seen = {object_id}
        while frontier:
            current = frontier.pop()
            for obj in self.objects:
                if obj.attached_to == current and obj.id not in seen:
                    seen.add(obj.id)
                    found.append(obj)
                    frontier.append(obj.id)
        return found

    def detach_orphans(self) -> None:
        """Un élément supprimé ne doit pas laisser ses voisins accrochés à un
        identifiant fantôme : ils se retrouveraient figés, incapables de
        suivre quoi que ce soit."""
        alive = {o.id for o in self.objects}
        for obj in self.objects:
            if obj.attached_to is not None and obj.attached_to not in alive:
                obj.attached_to = None

    def draw_order(self) -> list[DrawnObject]:
        """Les éléments du plus bas au plus haut de leur pile. Ce qui est
        POSÉ SUR quelque chose doit être dessiné APRÈS son support, sinon on
        verrait le support par-dessus ce qu'il porte."""
        return sorted(self.objects, key=self.stack_depth)

    # -- rendu --------------------------------------------------------------
    def shadow_mask(self, shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        """Masque des ombres portées de tous les éléments, en UN seul
        passage, plus la zone qu'il occupe réellement.

        Deux économies, mesurées plutôt que supposées :
          * le masque est calculé en DEMI-RÉSOLUTION puis réagrandi. Une
            ombre est floue de toute façon, personne ne voit la différence,
            et ça divise par quatre le coût du flou.
          * on renvoie la BOÎTE ENGLOBANTE, pour que l'assombrissement ne
            touche que cette zone. Sur trois éléments, le produit sur toute
            l'image coûtait 3,5 ms par frame — la moitié du budget de rendu,
            pour un effet qui ne concerne qu'un coin de l'écran."""
        if not self.objects:
            return None
        h, w = shape
        half = np.zeros((h // 2, w // 2), dtype=np.uint8)
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for obj in self.objects:
            hull = mesh3d.shadow_hull(
                obj.ensure_mesh(), obj.roll, obj.yaw, obj.center_screen(), obj.scale, obj.pitch
            )
            if len(hull) < 3:
                continue
            cv2.fillConvexPoly(half, (hull // 2).astype(np.int32), 255, cv2.LINE_AA)
            x0, y0 = min(x0, hull[:, 0].min()), min(y0, hull[:, 1].min())
            x1, y1 = max(x1, hull[:, 0].max()), max(y1, hull[:, 1].max())
        if x1 < x0:
            return None

        half = cv2.GaussianBlur(half, (SHADOW_BLUR, SHADOW_BLUR), 0)
        mask = cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR)
        pad = SHADOW_BLUR * 2   # le flou déborde du contour : la zone doit en tenir compte
        box = (
            max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(w, int(x1) + pad), min(h, int(y1) + pad),
        )
        return mask, box

    def render(
        self,
        shape: tuple[int, int, int],
        hovered_id: str | None,
        pending_color: tuple[int, int, int] = (0, 220, 255),
        show_labels: bool = True,
    ) -> np.ndarray:
        img = np.zeros(shape, dtype=np.uint8)
        for obj in self.draw_order():
            render_object(img, obj, hovered=(obj.id == hovered_id), show_label=show_labels)

        # Lien d'empilement : un trait fin entre un élément et son support.
        # Sans ce repère, deux éléments liés sont indiscernables de deux
        # éléments simplement voisins, jusqu'à ce qu'on en déplace un.
        for obj in self.objects:
            support = self.get_object(obj.attached_to)
            if support is not None:
                p1 = tuple(int(v) for v in obj.center_screen())
                p2 = tuple(int(v) for v in support.center_screen())
                cv2.line(img, p1, p2, LINK_COLOR, 1, cv2.LINE_AA)
                cv2.circle(img, p2, 3, LINK_COLOR, -1, cv2.LINE_AA)

        pending = self.pending_strokes + ([self._active] if self._active else [])
        for stroke in pending:
            for p1, p2 in zip(stroke.points, stroke.points[1:]):
                cv2.line(img, p1, p2, pending_color, stroke.width, cv2.LINE_AA)
        return img


def render_object(img: np.ndarray, obj: DrawnObject, hovered: bool, show_label: bool = True) -> None:
    """La COULEUR appartient à l'élément : l'état sélectionné/survolé passe
    donc par le CADRE autour, pas par la teinte du dessin — sinon changer la
    couleur d'un élément sélectionné n'aurait rien changé à l'écran."""
    mesh3d.render_mesh(
        img, obj.ensure_mesh(), obj.roll, obj.yaw, obj.center_screen(), obj.color, obj.scale, obj.pitch
    )

    x0, y0, x1, y1 = obj.transformed_bbox()
    if obj.selected:
        cv2.rectangle(img, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), SELECTION_COLOR, 2, cv2.LINE_AA)
        # Petits coins : rendent la sélection lisible même sur un élément
        # très fin, où un simple rectangle se confond avec le dessin.
        for cx, cy in ((x0 - 8, y0 - 8), (x1 + 8, y0 - 8), (x0 - 8, y1 + 8), (x1 + 8, y1 + 8)):
            cv2.circle(img, (cx, cy), 4, SELECTION_COLOR, -1, cv2.LINE_AA)
    elif hovered:
        cv2.rectangle(img, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), HOVER_COLOR, 1, cv2.LINE_AA)

    if show_label:
        label_color = SELECTION_COLOR if obj.selected else (170, 170, 170)
        cv2.putText(img, obj.label, (x0 - 6, max(y0 - 14, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
