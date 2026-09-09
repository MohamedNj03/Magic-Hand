r"""
Main Pinceau — dessiner, reconnaître et manipuler des éléments 3D à la main.

v11 : LES GESTES QUI DEVAIENT MARCHER MARCHENT
==============================================
    Index tendu, tenu .......... dessiner (pause ~0,6 s -> devient un élément)
    Poing fermé ................ DÉPLACER l'élément (ce qui est posé dessus suit)
    Main ouverte DESSUS ........ le TOURNER en 3D, sur place
    ÉLOIGNER la main ........... désélectionner — rien d'autre à faire
    Signe de la loupe .......... zoomer ; avec un élément choisi, MONTER les
                                 trois doigts l'agrandit, les BAISSER le réduit
    Pouce vers le HAUT ......... couleur suivante (claquement de doigts aussi)
    Pouce vers le BAS .......... supprimer le dernier élément dessiné
    Signe V .................... annuler la dernière action
    DEUX MAINS ouvertes, puis on les RETIRE ..... tout effacer

Une seule main pilote TOUT. La deuxième main ne sert qu'à une chose : armer
l'effacement général. C'est la seule action irréversible du lot, et c'est
la seule qui demande les deux mains — impossible de la déclencher par
accident d'une main occupée à dessiner.


CE QUI A CHANGÉ EN v11, ET POURQUOI
===================================

1. TOUT EFFACER = LEVER LES DEUX MAINS, PUIS LES RETIRER
   Le maintien du pouce vers le bas partageait son geste avec la
   suppression simple : la même pose faisait deux choses selon sa durée, ce
   qui obligeait à compter dans sa tête. Maintenant les deux gestes n'ont
   plus rien en commun. Le nouveau se lit tout seul : on montre les deux
   mains (une jauge s'arme), on les retire (une seconde jauge confirme), la
   page est nette. Remettre une main en cours de route annule. Et le geste
   reste RÉVERSIBLE : `z`, ou le signe V, restaure tout.

2. LES POSES NE DÉPENDENT PLUS DE L'ANGLE DE LA MAIN
   Le test « doigt tendu » comparait les ordonnées des points — autrement
   dit il supposait la main parfaitement verticale. En dessinant, on
   incline toujours un peu le poignet : le doigt tendu passait pour replié,
   la pose se perdait, le trait se coupait. On compare maintenant les
   DISTANCES AU POIGNET, invariantes par rotation. C'est la correction qui
   change le plus le ressenti du dessin.

3. L'ÉCRITURE NE BLOQUE PLUS L'IMAGE
   Lire une lettre coûte 330 ms (Tesseract est un processus séparé). Ce
   travail se faisait dans la boucle vidéo : dix images gelées, pile au
   moment où l'utilisateur regarde son trait. Il part maintenant dans un
   fil de fond (`async_recognition.py`) ; l'élément naît immédiatement en
   dessin propre et devient une lettre quand la lecture revient.

4. PLUS DE LETTRES INVENTÉES
   Mesuré : un simple trait diagonal de 20 px revenait en « N » avec 67 %
   de confiance, et le dessin de l'utilisateur était REMPLACÉ par cette
   lettre. Un tracé doit désormais ressembler à de l'écriture avant qu'on
   demande son avis à l'OCR, et le seuil de confiance est passé de 40 à 70.
   Une abstention n'est pas un échec : le dessin reste le dessin.

5. LA COULEUR A UN GESTE FIABLE, ET UNE PALETTE VISIBLE
   Le claquement de doigts reste (c'est le geste agréable), mais il est
   rapide et géométrique : il rate parfois, par construction. Le pouce vers
   le HAUT fait la même chose et ne rate jamais — c'est un geste que le
   classifieur reconnaît nativement. Et les huit couleurs sont affichées :
   on sait enfin combien de gestes séparent du bleu.

6. LE HAUT D'UNE PILE SE REPREND VRAIMENT
   Deux éléments empilés ont presque le même centre : le hit-test choisissait
   l'un ou l'autre au hasard, donc reprendre l'objet du dessus attrapait le
   support une fois sur deux. Il suit maintenant la même règle que l'œil :
   ce qui est dessus est dessiné en dernier, et c'est ce qu'on attrape.


7. L'INDEX DOIT ÊTRE FRANC POUR DESSINER
   Un seul passage de l'index dans la pose suffisait à laisser un trait. Or
   en changeant de geste — un poing qui s'ouvre, par exemple — l'index se
   tend forcément une image ou deux au passage. D'où le sentiment que
   l'application « dessinait toute seule ». Il faut maintenant tenir la pose
   `DRAW_ONSET_FRAMES` images pour COMMENCER un trait. Une fois commencé,
   la tolérance reste maximale : c'est la continuité de l'écriture qu'il ne
   faut pas casser, pas son début.

8. S'ÉLOIGNER SUFFIT À DÉSÉLECTIONNER
   Il fallait une pose exprès (main ouverte ailleurs) pour lâcher un
   élément : on devait demander la désélection alors qu'on l'avait déjà
   exprimée en s'en allant. Maintenant, éloigner la main de
   `DESELECT_DISTANCE` pendant `DESELECT_DELAY` suffit — sauf pendant qu'on
   tient ou qu'on redimensionne l'élément, où être loin fait partie du
   geste.

9. LA TAILLE SE PILOTE EN MONTANT LES TROIS DOIGTS
   C'était l'ouverture du cercle pouce/index : course courte, et un plancher
   dès que les doigts se touchent, donc doubler un élément demandait de
   reprendre le geste plusieurs fois. C'est maintenant la HAUTEUR des trois
   doigts tendus — toute la hauteur de l'image comme course, et le geste que
   l'on fait naturellement pour dire « plus grand ».


10. LES FORMES NE SONT PLUS DES CERCLES PAR DÉFAUT
   Tout partait de l'ENVELOPPE CONVEXE du tracé, ce qui efface exactement
   l'information qui distingue une lettre d'une forme simple : ses creux.
   Un B revenait en cercle, un gribouillage en hexagone. On compare
   maintenant l'aire réelle à celle de l'enveloppe (une forme creuse n'est
   pas une primitive), et le test du cercle vérifie ce qu'un cercle est
   vraiment : tous les points à la même distance du centre. Mesuré sur
   21 tracés : 13 formes reconnues, 8 laissées intactes, 0 erreur.

11. L'ÉCRAN NE RACONTE PLUS LA MACHINERIE
   Il affichait les images par seconde, l'état interne, le nombre de traits
   en attente, le moteur d'OCR et sa version, et une étiquette numérotée
   sur chaque élément. Il ne reste que la palette, un mot quand il se passe
   quelque chose, et les jauges avant un geste destructeur. Tout le reste
   est visuel. Les diagnostics sont derrière la touche `d` : rien n'est
   perdu, on arrête juste de l'imposer. Les textes sont en anglais.

12. LA CAMÉRA S'OUVRE EN GRAND
   1280x720 quand elle sait le faire, et la fenêtre est dimensionnée sur
   l'image REÇUE — OpenCV ne refuse jamais une définition, il donne la plus
   proche sans le dire. Le HUD est à l'échelle de la hauteur, sinon il
   restait minuscule dans un coin.


LIMITE ASSUMÉE, TOUJOURS VRAIE
==============================
Une seule caméra RGB ne donne pas de vraie profondeur. Les signaux yaw et
pitch restent des PROXY relatifs bruités. Ce qui est réel, c'est la 3D de
l'ÉLÉMENT : vraies faces, vraies arêtes, vraie occlusion par profondeur.


FICHIERS DU PROJET
==================
    canvas.py             — traits, éléments 3D, calque, rendu
    filters.py            — lissage One Euro, zone morte, angles
    hand_signals.py       — lecture géométrique de la main (poses, loupe, claquement)
    async_recognition.py  — la lecture d'écriture, hors de la boucle vidéo
    sketch_recognition.py — nettoyage de traits + formes + écriture
    magnifier.py          — la loupe (rendu du disque grossissant)
    mesh3d.py             — moteur 3D (rotation 3 axes, projection, ombre portée)
    hud.py                — l'affichage par-dessus l'image

Installation :
    pip install opencv-python opencv-contrib-python mediapipe numpy
    (facultatif, améliore l'écriture : pip install pytesseract + binaire Tesseract)

Modèle (PowerShell) :
    mkdir models
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" -OutFile "models\gesture_recognizer.task"

Lancement :
    python gesture_canvas_manipulation.py
    q quitter | h aide | o ombres | c couleur | z annuler | s sauvegarder
    d vue technique (images/s, état interne, journal de reconnaissance)
"""

from __future__ import annotations

import collections
import datetime
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import async_recognition
import canvas as canvas_module
import filters
import hand_signals as hs
import hud
import magnifier
import sketch_recognition as sketch

# Ré-exports : ces noms vivaient dans ce fichier avant d'en être extraits.
# On les garde accessibles ici pour que rien de ce qui les importait ne
# casse — et parce que c'est ce fichier qui reste la porte d'entrée.
WRIST = hs.WRIST
THUMB_TIP = hs.THUMB_TIP
INDEX_MCP, INDEX_PIP, INDEX_TIP = hs.INDEX_MCP, hs.INDEX_PIP, hs.INDEX_TIP
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = hs.MIDDLE_MCP, hs.MIDDLE_PIP, hs.MIDDLE_TIP
RING_MCP, RING_PIP, RING_TIP = hs.RING_MCP, hs.RING_PIP, hs.RING_TIP
PINKY_MCP, PINKY_PIP, PINKY_TIP = hs.PINKY_MCP, hs.PINKY_PIP, hs.PINKY_TIP
PALM_LANDMARKS = hs.PALM_LANDMARKS

is_pointing = hs.is_pointing
extended_finger_count = hs.extended_finger_count
hand_pose_signal = hs.hand_pose_signal
get_index_fingertip = hs.get_index_fingertip
get_palm_center = hs.get_palm_center
wrist_roll_angle = hs.wrist_roll_angle
yaw_signal = hs.yaw_signal
pitch_signal = hs.pitch_signal

OneEuroFilter = filters.OneEuroFilter
Point2DFilter = filters.Point2DFilter
clamp = filters.clamp
deadzone = filters.deadzone

Stroke = canvas_module.Stroke
DrawnObject = canvas_module.DrawnObject
Canvas = canvas_module.Canvas
COLOR_PALETTE = canvas_module.COLOR_PALETTE
KIND_LABELS = canvas_module.KIND_LABELS
build_object_mesh = canvas_module.build_object_mesh
_mesh_depth = canvas_module._mesh_depth

apply_shadow = hud.apply_shadow
draw_trail = hud.draw_trail
HELP_LINES = hud.HELP_LINES


# --------------------------------------------------------------------------
# Réglages
# --------------------------------------------------------------------------
YAW_SENSITIVITY = 20.0             # facteur empirique — à ajuster devant ta caméra
PITCH_SENSITIVITY = 16.0
DEPTH_DEADZONE = 0.006             # zone morte des signaux de profondeur (unités z MediaPipe)
MAX_YAW = math.radians(720)        # garde-fou large, pas une vraie limite
MAX_PITCH = math.radians(720)
MIN_OBJECT_SCALE = 0.2             # garde-fous de redimensionnement : ni point invisible,
MAX_OBJECT_SCALE = 5.0             # ni élément absurdement grand

# Création automatique d'élément
AUTO_VALIDATE_DELAY = 0.6          # s de pause après le dernier trait avant que ça devienne un élément
NEW_GROUP_DISTANCE = 260.0         # px : un trait commencé plus loin valide d'abord le groupe en cours

# Écriture rapide. Mesuré : une SEULE frame où la main est floue coupait le
# trait en deux, et trois décrochages sur un mot le découpaient en quatre
# morceaux. Le lissage, lui, n'était pas en cause (0,4 px de retard à
# 750 px/s : le One Euro s'ouvre déjà tout seul à grande vitesse). C'est
# donc la CONTINUITÉ du trait qu'il fallait protéger, pas la trajectoire.
DRAW_ONSET_FRAMES = 4              # frames d'index tendu FRANC avant de COMMENCER un trait.
                                   # En changeant de pose — un poing qui
                                   # s'ouvre, par exemple — l'index se tend
                                   # forcément une image ou deux au passage.
                                   # Sans ce délai, chacun de ces passages
                                   # laissait un trait : c'est le "il dessine
                                   # alors que je ne veux pas dessiner".
DRAW_GRACE_FRAMES = 3              # frames de pose perdue tolérées sans couper le trait
FAST_DRAW_PX = 9.0                 # px/frame : au-delà, la main est clairement encore en train de tracer
MAX_DRAW_JUMP_PX = 120.0           # px/frame : au-delà, ce n'est plus un trait mais un saut de suivi

# Prise / relâchement
GRAB_ONSET_FRAMES = 3              # frames de poing avant de prendre un élément
GRAB_RELEASE_GRACE = 20            # frames de pose ambiguë tolérées avant de lâcher
GRAB_LOST_GRACE = 6                # frames SANS main tolérées avant de lâcher
OPEN_HAND_FRAMES = 3               # frames de main grande ouverte avant qu'elle agisse
GRAB_REACH = 90.0                  # px de marge autour d'un élément pour pouvoir l'attraper
FIST_CONFIDENCE = 0.55             # seuil du classifieur ML (l'un des deux chemins d'onset)

# DÉSÉLECTION PAR ÉLOIGNEMENT. Il fallait une pose exprès (main ouverte
# ailleurs) pour lâcher un élément, ce qui obligeait à demander la
# désélection alors qu'on l'avait déjà exprimée en s'en allant. Maintenant
# il suffit d'éloigner la main : c'est le geste que tout le monde fait déjà.
# La distance est bien plus grande que `GRAB_REACH` pour que travailler
# AUTOUR d'un élément (le tourner, le redimensionner) ne le désélectionne
# jamais par mégarde.
DESELECT_DISTANCE = 200.0          # px au-delà de la boîte de l'élément
DESELECT_DELAY = 0.35              # s à cette distance avant de lâcher — un aller-retour ne compte pas

# Rotation continue : un élément lâché en tournant garde son élan.
SPIN_MIN = 1.5                     # rad/s en-deçà desquels on ne lance rien (un lâcher lent ne doit pas dériver)
SPIN_MAX = 14.0                    # rad/s — garde-fou contre un pic de bruit sur une frame
SPIN_DAMPING = 1.9                 # amortissement par seconde : ~0,4 s pour perdre la moitié de l'élan

# Empilement : poser un élément sur un autre les lie.
STACK_MARGIN = 12.0                # px de tolérance autour de la bbox pour considérer qu'on a posé DESSUS

# Traînée du curseur
TRAIL_POINTS = 14                  # longueur maximale de la comète
TRAIL_MAX_AGE = 0.35               # s : au-delà, un point de traînée est oublié

# Gestes du classifieur ML transformés en UN déclenchement par appui.
# `MIN_PRESS` filtre le clignotement du classifieur (il voit parfois un
# geste sur une frame isolée d'une main en mouvement) ; `GAP_GRACE` évite
# qu'un trou d'une frame compte comme un relâchement, donc qu'un seul appui
# déclenche deux fois.
GESTURE_MIN_PRESS = 0.12           # s
GESTURE_GAP_GRACE = 0.16           # s
UNDO_MIN_PRESS = 0.20              # s — l'annulation demande un appui franc
UNDO_CONFIDENCE = 0.65

# TOUT EFFACER — le seul geste à deux mains, et le seul irréversible... sauf
# qu'il ne l'est plus : il garde une photo, `z` la restaure. Deux temps,
# deux jauges, et un retour en arrière possible à chaque instant.
CLEAR_ARM_SECONDS = 0.6            # s de deux mains ouvertes visibles avant que le geste s'arme
CLEAR_CONFIRM_SECONDS = 0.5        # s sans AUCUNE main pour valider l'effacement
CLEAR_ARM_GRACE = 0.6              # s de tolérance quand MediaPipe perd une des deux mains

# Loupe / redimensionnement
LENS_HOLD_FRAMES = 2               # frames de signe de la loupe avant qu'elle s'active
# La TAILLE se pilote en montant ou baissant les trois doigts tendus : monter
# de `RESIZE_DOUBLING_PX` double l'élément, descendre d'autant le divise par
# deux. Un rapport (et non une addition) parce que doubler puis redoubler
# doit demander le même geste deux fois — c'est ce qui rend la commande
# prévisible aux petites comme aux grandes tailles.
RESIZE_DOUBLING_PX = 180.0

CAPTURE_DIR = "captures"
WINDOW_TITLE = "Main Pinceau"

# Définitions demandées à la caméra, de la plus grande à la plus petite.
# 1280x720 est le bon compromis mesuré : deux fois plus de surface utile
# qu'en 960x540 pour le dessin, et MediaPipe y tient toujours ses 9 ms par
# image sur cette machine. Si la caméra ne sait pas la faire, on redescend.
CAMERA_RESOLUTIONS = ((1088, 612), (1024, 576), (960, 540), (640, 480))
# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
@dataclass
class GrabState:
    """Ancrage d'une manipulation. Tous les champs `origin_*` mémorisent
    l'état de l'élément ET la lecture de la main au moment où le mode a
    commencé : le delta calculé ensuite vaut zéro sur cette frame précise,
    donc l'élément ne saute ni quand on l'attrape, ni quand on passe du
    déplacement à la rotation."""

    object_id: str
    handedness: str | None
    mode: str = "move"  # "move" (main fermée) | "rotate" (main ouverte, sur place)
    origin_signal_angle: float = 0.0
    origin_signal_depth: float = 0.0
    origin_signal_pitch: float = 0.0
    origin_signal_pos: tuple[float, float] = (0.0, 0.0)
    origin_object_roll: float = 0.0
    origin_object_yaw: float = 0.0
    origin_object_pitch: float = 0.0
    origin_object_offset: tuple[float, float] = (0.0, 0.0)
    # Empilement : position de départ de ce qui est posé DESSUS, pour que la
    # pile suive sans dérive cumulée.
    origin_children: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Élan : dernier angle vu et son horodatage, pour mesurer la vitesse de
    # rotation au moment du lâcher.
    prev_angle: float | None = None
    prev_time: float = 0.0
    spin_ema: float = 0.0   # vitesse de rotation lissée (voir _update_manipulation)
    roll_filter: OneEuroFilter | None = None
    yaw_filter: OneEuroFilter | None = None
    pitch_filter: OneEuroFilter | None = None
    pos_filter: Point2DFilter | None = None


def _anchor_grab(grab: GrabState, obj, canvas: Canvas, now: float, angle: float, depth: float,
                 pitch: float, pos: tuple[float, float], mode: str) -> None:
    grab.mode = mode
    grab.origin_signal_angle = angle
    grab.origin_signal_depth = depth
    grab.origin_signal_pitch = pitch
    grab.origin_signal_pos = pos
    grab.origin_object_roll = obj.roll
    grab.origin_object_yaw = obj.yaw
    grab.origin_object_pitch = obj.pitch
    grab.origin_object_offset = (obj.offset[0], obj.offset[1])
    grab.origin_children = {c.id: (c.offset[0], c.offset[1]) for c in canvas.descendants(obj.id)}
    grab.prev_angle = angle
    grab.prev_time = now
    grab.spin_ema = 0.0
    grab.roll_filter = OneEuroFilter(now, angle, min_cutoff=0.8, beta=0.3)
    grab.yaw_filter = OneEuroFilter(now, depth, min_cutoff=0.5, beta=0.2)
    grab.pitch_filter = OneEuroFilter(now, pitch, min_cutoff=0.5, beta=0.2)
    grab.pos_filter = Point2DFilter(now, pos[0], pos[1], min_cutoff=1.4, beta=0.8)


@dataclass
class LensSession:
    """Ancrage de la loupe.

    Deux grandeurs, deux rôles, et c'est la sélection qui tranche :
      * sans sélection, l'OUVERTURE du cercle pouce/index donne le
        grossissement — lecture absolue, comme une vraie lentille ;
      * avec une sélection, c'est la HAUTEUR DES TROIS DOIGTS qui donne la
        taille — lecture RELATIVE au début du geste, donc sans échelle
        absolue à retenir, et avec une course bien plus grande que
        l'ouverture d'un cercle entre deux doigts."""
    origin_aperture: float
    origin_scale: float
    origin_tips_y: float
    center_filter: Point2DFilter
    radius_filter: OneEuroFilter
    aperture_filter: OneEuroFilter
    tips_filter: OneEuroFilter


@dataclass
class GestureLatch:
    """Un geste du classifieur transformé en UN SEUL déclenchement par appui.

    Deux filtres, chacun contre un défaut constaté :
      * `min_press` — le classifieur voit parfois un geste sur une frame
        isolée d'une main en mouvement. Un geste qui ne dure pas n'existe pas.
      * `gap_grace` — il perd aussi le geste une frame au milieu d'un appui.
        Sans cette tolérance, un seul appui déclencherait deux fois."""

    min_press: float = GESTURE_MIN_PRESS
    gap_grace: float = GESTURE_GAP_GRACE
    since: float | None = None
    last_seen: float = -1e9
    fired: bool = False

    def update(self, active: bool, now: float) -> bool:
        """True sur la frame EXACTE où l'appui devient valide."""
        if active:
            if self.since is None or (now - self.last_seen) > self.gap_grace:
                self.since = now
                self.fired = False
            self.last_seen = now
            if not self.fired and (now - self.since) >= self.min_press:
                self.fired = True
                return True
            return False
        if self.since is not None and (now - self.last_seen) > self.gap_grace:
            self.reset()
        return False

    def reset(self) -> None:
        self.since = None
        self.fired = False


@dataclass
class ClearAllGesture:
    """TOUT EFFACER, en deux temps : on montre les deux mains, puis on les
    retire.

    Pourquoi ce geste-là. C'est la seule action qui détruit tout, elle doit
    donc être impossible à faire par accident — or une main occupée à
    dessiner ne peut pas, par définition, en faire un geste à deux mains.
    Et c'est un geste que personne n'a besoin d'apprendre : on balaie la
    table des deux mains, la table est nette.

    Les deux temps ne sont pas une décoration. Le premier (deux mains
    visibles) empêche que baisser les bras efface le travail ; le second
    (les mains hors champ) laisse une demi-seconde pour se raviser — il
    suffit de remontrer une main."""

    two_hands_since: float | None = None
    two_hands_last_seen: float = -1e9
    armed: bool = False
    gone_since: float | None = None
    fired: bool = False   # un seul effacement par armement

    def reset(self) -> None:
        self.two_hands_since = None
        self.armed = False
        self.gone_since = None
        self.fired = False

    def update(self, hand_count: int, both_open: bool, now: float) -> tuple[str, float, bool]:
        """-> (phase, progression 0..1, déclenché)

        phase : "idle" | "arm" (les deux mains montent) | "ready" (armé)
                | "confirm" (les mains sont parties, décompte en cours)"""
        if hand_count >= 2 and both_open:
            self.gone_since = None
            if self.two_hands_since is None:
                self.two_hands_since = now
                self.fired = False
            self.two_hands_last_seen = now
            held = now - self.two_hands_since
            if held >= CLEAR_ARM_SECONDS:
                self.armed = True
                return "ready", 1.0, False
            return "arm", held / CLEAR_ARM_SECONDS, False

        if hand_count == 0:
            if not self.armed:
                self.two_hands_since = None
                return "idle", 0.0, False
            if self.gone_since is None:
                self.gone_since = now
            waited = now - self.gone_since
            if waited < CLEAR_CONFIRM_SECONDS:
                return "confirm", waited / CLEAR_CONFIRM_SECONDS, False
            # Désarmé au moment même où ça part : rester hors champ ne peut
            # donc pas déclencher un second effacement une demi-seconde plus
            # tard, et manger le travail fait entre-temps. Il faut remontrer
            # les deux mains.
            self.reset()
            return "idle", 0.0, True

        # Une main visible (ou deux, mais pas ouvertes) : ce n'est pas
        # "retirer ses deux mains". On désarme — après une grâce, parce que
        # MediaPipe perd régulièrement une des deux mains une frame ou deux.
        self.gone_since = None
        if (now - self.two_hands_last_seen) > CLEAR_ARM_GRACE:
            self.armed = False
            self.two_hands_since = None
            return "idle", 0.0, False
        if self.armed:
            return "ready", 1.0, False
        held = 0.0 if self.two_hands_since is None else now - self.two_hands_since
        return ("arm", clamp(held / CLEAR_ARM_SECONDS, 0.0, 1.0), False) if held > 0 else ("idle", 0.0, False)


@dataclass
class HandObservation:
    landmarks: object
    gesture_name: str | None
    confidence: float
    handedness: str | None

    def is_gesture(self, name: str, min_confidence: float = 0.5) -> bool:
        return self.gesture_name == name and self.confidence >= min_confidence


def top_gesture(gestures_for_hand) -> tuple[str | None, float]:
    if not gestures_for_hand:
        return None, 0.0
    top = gestures_for_hand[0]
    return top.category_name, top.score


def observe_hands(result) -> list[HandObservation]:
    """Normalise un résultat MediaPipe en une observation par main."""
    n = len(result.hand_landmarks) if result.hand_landmarks else 0
    out = []
    for i in range(n):
        gesture_name, confidence = (
            top_gesture(result.gestures[i]) if result.gestures and i < len(result.gestures) else (None, 0.0)
        )
        handedness = None
        hlist = getattr(result, "handedness", None)
        if hlist and i < len(hlist) and hlist[i]:
            handedness = hlist[i][0].category_name
        out.append(HandObservation(result.hand_landmarks[i], gesture_name, confidence, handedness))
    return out


def pick_primary(state, hands: list[HandObservation], w: int, h: int) -> HandObservation | None:
    """LA main qu'on suit. Le détecteur en accepte deux (il en faut deux pour
    tout effacer), mais une seule pilote l'application — et c'est toujours
    la MÊME d'une frame à l'autre : celle qui tient déjà l'élément, sinon la
    plus proche de la position suivie. Se fier à l'ordre de la liste, comme
    autrefois, faisait sauter le suivi d'une main à l'autre sans prévenir."""
    if not hands:
        return None
    if len(hands) == 1:
        return hands[0]
    if state.grab is not None and state.grab.handedness is not None:
        same = next((hd for hd in hands if hd.handedness == state.grab.handedness), None)
        if same is not None:
            return same
    if state.last_palm is not None:
        return min(
            hands,
            key=lambda hd: math.dist(hs.get_palm_center(hd.landmarks, w, h), state.last_palm),
        )
    return hands[0]


# --------------------------------------------------------------------------
# État global
# --------------------------------------------------------------------------
@dataclass
class AppState:
    canvas: Canvas
    recognizer: async_recognition.RecognitionService | None = None
    point_filter: Point2DFilter | None = None
    grab: GrabState | None = None
    lens: LensSession | None = None
    was_drawing: bool = False
    last_status: str = "IDLE"
    fist_run: int = 0          # frames de poing consécutives (onset de prise)
    point_run: int = 0         # frames d'index tendu FRANC (onset de dessin)
    away_since: float | None = None   # depuis quand la main est loin de l'élément sélectionné
    open_run: int = 0          # frames de main grande ouverte (lâcher + désélectionner)
    lens_run: int = 0          # frames de signe de la loupe
    miss_run: int = 0          # frames de pose ambiguë (grâce de relâchement)
    lost_run: int = 0          # frames SANS main détectée (grâce de perte de suivi)
    selected_id: str | None = None      # UN seul élément sélectionné à la fois
    last_touched_id: str | None = None  # dernier élément créé OU manipulé
    pending_since: float | None = None  # date du dernier trait terminé (création automatique)
    last_palm: tuple[float, float] | None = None
    last_index_tip: tuple[float, float] | None = None
    pen_color_idx: int = 0      # couleur COURANTE : celle du trait en cours ET du prochain élément
    draw_miss_run: int = 0      # frames sans pose de dessin alors qu'un trait est ouvert
    last_frame_time: float | None = None
    cursor_trail: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=TRAIL_POINTS)
    )
    status_set_this_frame: bool = False
    snap: hs.SnapDetector = field(default_factory=hs.SnapDetector)
    delete_latch: GestureLatch = field(default_factory=GestureLatch)
    color_latch: GestureLatch = field(default_factory=GestureLatch)
    undo_latch: GestureLatch = field(default_factory=lambda: GestureLatch(min_press=UNDO_MIN_PRESS))
    clear_gesture: ClearAllGesture = field(default_factory=ClearAllGesture)

    def set_status(self, message: str) -> None:
        """Pose le message d'état ET note qu'il a été posé pendant CETTE
        frame. Sans ce drapeau, un statut d'une frame précédente ("PRISE",
        "LOUPE"...) restait affiché après la fin du geste, ce qui donnait
        l'impression que l'application était bloquée dans un mode."""
        self.last_status = message
        self.status_set_this_frame = True

    @property
    def pen_color(self) -> tuple[int, int, int]:
        return COLOR_PALETTE[self.pen_color_idx % len(COLOR_PALETTE)]

    def next_pen_color(self) -> None:
        self.pen_color_idx = (self.pen_color_idx + 1) % len(COLOR_PALETTE)

    def selected_object(self) -> DrawnObject | None:
        return self.canvas.get_object(self.selected_id)

    def prune_selection(self) -> None:
        """Un élément supprimé ne doit pas rester "sélectionné" : sans ce
        ménage, la sélection garderait un identifiant fantôme et le
        redimensionnement viserait un élément qui n'existe plus."""
        self.canvas.detach_orphans()
        alive = {o.id for o in self.canvas.objects}
        if self.selected_id is not None and self.selected_id not in alive:
            self.selected_id = None
        if self.last_touched_id is not None and self.last_touched_id not in alive:
            self.last_touched_id = None

    def deselect(self) -> bool:
        had = self.selected_id is not None
        self.selected_id = None
        return had


@dataclass
class FrameResult:
    triggered: str | None = None      # "Thumb_Down" | "Clear_All"
    hovered_object_id: str | None = None
    drawing: bool = False
    cursor: tuple[float, float] | None = None
    lens: magnifier.MagnifierState | None = None   # loupe à dessiner, s'il y en a une
    resizing: bool = False        # la loupe redimensionne un élément au lieu de zoomer
    snapped: bool = False         # un claquement de doigts a été reconnu cette frame
    snap_armed: bool = False      # main en position de claquer (témoin à l'écran)
    auto_created: bool = False    # un élément vient d'être créé automatiquement
    color_changed: bool = False   # la couleur courante vient d'avancer
    undone: str | None = None     # "object" | "stroke" | "clear" si une annulation a eu lieu
    recognized: list = field(default_factory=list)   # textes reconnus arrivés cette frame
    clear_progress: float = 0.0   # 0..1 — avancement de la phase en cours de "tout effacer"
    clear_phase: str = "idle"     # "idle" | "arm" | "ready" | "confirm"
    trail: list = field(default_factory=list)   # comète derrière le doigt, du plus ancien au plus récent


# --------------------------------------------------------------------------
# Sous-étapes
# --------------------------------------------------------------------------
def _selected_within_reach(state: AppState, palm: tuple[float, float]) -> DrawnObject | None:
    """L'élément sélectionné, s'il est assez près de la main pour qu'ouvrir
    la paume signifie "je le tourne" plutôt que "je le lâche". La même
    portée que pour l'attraper, pour qu'il n'y ait qu'une seule notion de
    "près de" à comprendre."""
    obj = state.selected_object()
    if obj is None:
        return None
    return obj if obj.contains_point(palm[0], palm[1], margin=int(GRAB_REACH)) else None


def _distance_to_object(obj: DrawnObject, px: float, py: float) -> float:
    """Distance d'un point à la boîte d'un élément — 0 s'il est dedans."""
    x0, y0, x1, y1 = obj.transformed_bbox()
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _update_auto_deselect(state: AppState, palm: tuple[float, float] | None, now: float,
                          busy: bool) -> bool:
    """ÉLOIGNER LA MAIN DÉSÉLECTIONNE. Plus besoin d'une pose exprès : quand
    on s'en va, on a déjà dit qu'on avait fini.

    `busy` couvre les cas où la main est loin mais toujours en train de
    travailler sur l'élément — on le tient, ou on le redimensionne à bout de
    bras. Le désélectionner là serait le lâcher en plein geste.

    Le délai de `DESELECT_DELAY` compte : sans lui, un aller-retour rapide
    de la main (ou une frame de suivi qui part au loin) désélectionnerait
    alors qu'on n'a jamais eu l'intention de partir."""
    if state.selected_id is None or busy:
        state.away_since = None
        return False

    obj = state.selected_object()
    if obj is None:
        state.away_since = None
        return False

    # Main hors champ : c'est le "loin" le plus franc qui soit.
    away = palm is None or _distance_to_object(obj, palm[0], palm[1]) > DESELECT_DISTANCE
    if not away:
        state.away_since = None
        return False

    if state.away_since is None:
        state.away_since = now
        return False
    if (now - state.away_since) < DESELECT_DELAY:
        return False

    state.away_since = None
    state.deselect()
    return True


def _update_lens(state: AppState, hand: HandObservation, w: int, h: int, now: float, fr: FrameResult) -> None:
    """Signe de la loupe : le cercle pouce/index, trois doigts tendus. Deux
    rôles selon qu'un élément est sélectionné ou non — la distinction, sans
    zone grise, entre REDIMENSIONNER un élément et ZOOMER la vue :

        élément sélectionné -> MONTER les trois doigts agrandit,
                               les BAISSER réduit
        rien de sélectionné -> l'ouverture du cercle donne le grossissement

    Pourquoi la hauteur des doigts, et pas l'ouverture du cercle, pour la
    taille : le cercle pouce/index a une course courte et un plancher (les
    doigts finissent par se toucher), donc agrandir beaucoup demandait de
    reprendre le geste plusieurs fois. Monter ou baisser la main, elle, a
    toute la hauteur de l'image — et c'est le geste que l'on fait
    naturellement pour dire "plus grand".
    """
    geom = hs.lens_geometry(hand.landmarks, w, h)
    aperture = hs.thumb_index_aperture(hand.landmarks)
    tips = hs.three_finger_tips(hand.landmarks, w, h)
    if geom is None or aperture is None or tips is None:
        return
    center, radius = geom
    tips_y = tips[1]

    if state.lens is None:
        obj = state.selected_object()
        state.lens = LensSession(
            origin_aperture=aperture,
            origin_scale=obj.scale if obj is not None else 1.0,
            origin_tips_y=tips_y,
            center_filter=Point2DFilter(now, center[0], center[1], min_cutoff=1.2, beta=0.6),
            radius_filter=OneEuroFilter(now, radius, min_cutoff=1.0, beta=0.4),
            aperture_filter=OneEuroFilter(now, aperture, min_cutoff=1.0, beta=0.4),
            tips_filter=OneEuroFilter(now, tips_y, min_cutoff=1.0, beta=0.5),
        )

    smooth_cx, smooth_cy = state.lens.center_filter.update(now, center[0], center[1])
    smooth_radius = state.lens.radius_filter(now, radius)
    smooth_aperture = state.lens.aperture_filter(now, aperture)
    smooth_tips_y = state.lens.tips_filter(now, tips_y)

    obj = state.selected_object()
    if obj is not None:
        # `rise` est positif quand les doigts sont MONTÉS depuis le début du
        # geste (les ordonnées écran descendent vers le bas).
        rise = state.lens.origin_tips_y - smooth_tips_y
        obj.scale = clamp(
            state.lens.origin_scale * (2.0 ** (rise / RESIZE_DOUBLING_PX)),
            MIN_OBJECT_SCALE, MAX_OBJECT_SCALE,
        )
        state.last_touched_id = obj.id
        fr.resizing = True
        arrow = "^" if rise > 2 else ("v" if rise < -2 else "-")
        state.set_status(f"TAILLE {arrow} ({obj.label}, x{obj.scale:.2f})")
    else:
        fr.lens = magnifier.MagnifierState(
            center=(smooth_cx, smooth_cy),
            radius=smooth_radius,
            zoom=magnifier.zoom_for_aperture(smooth_aperture),
        )
        state.set_status(f"LOUPE (x{fr.lens.zoom:.1f})")


def _update_manipulation(state: AppState, hand: HandObservation, palm: tuple[float, float],
                         mode: str, now: float, fr: FrameResult) -> None:
    """Les deux façons de manipuler un élément, avec la MÊME main :

        main FERMÉE  -> on le déplace (il suit la paume) et on le fait
                        tourner dans le plan de l'écran avec le poignet ;
        main OUVERTE -> on l'oriente en 3D SUR PLACE (roll, yaw, pitch),
                        sans le déplacer d'un pixel.

    Séparer les deux règle aussi un problème de bruit : les signaux de
    profondeur d'une caméra RGB tremblent en permanence. En les réservant au
    mode "ouvert", déplacer un élément ne peut plus le faire vriller au
    passage — c'est demandé explicitement, donc autant que ce soit net."""
    obj = state.canvas.get_object(state.grab.object_id)
    if obj is None:
        state.grab = None  # élément supprimé pendant qu'on le manipulait
        return

    angle = hs.wrist_roll_angle(hand.landmarks)
    depth = hs.yaw_signal(hand.landmarks)
    pitch = hs.pitch_signal(hand.landmarks)

    if state.grab.mode != mode:
        # Changement de mode en pleine manipulation : on RÉANCRE sur l'état
        # courant, sinon l'élément sauterait au moment où la main s'ouvre.
        _anchor_grab(state.grab, obj, state.canvas, now, angle, depth, pitch, palm, mode)

    smooth_angle = state.grab.roll_filter(now, angle)
    obj.roll = state.grab.origin_object_roll + filters.angle_delta(smooth_angle, state.grab.origin_signal_angle)

    # Vitesse de rotation, mesurée en continu : c'est elle qui donnera son
    # élan à l'élément quand on le lâchera.
    #
    # Moyenne glissante et non vitesse de la dernière frame : sur une seule
    # image, un landmark qui tremble donne une vitesse énorme ou nulle au
    # hasard. Le geste réel, lui, dure plusieurs frames — c'est donc lui
    # qu'il faut mesurer, sinon l'élan dépendrait du moment exact où on
    # relâche plutôt que de la vitesse à laquelle on tournait.
    if state.grab.prev_angle is not None:
        dt = now - state.grab.prev_time
        if dt > 1e-4:
            speed = filters.angle_delta(smooth_angle, state.grab.prev_angle) / dt
            state.grab.spin_ema = 0.6 * state.grab.spin_ema + 0.4 * speed
            obj.spin = (
                clamp(state.grab.spin_ema, -SPIN_MAX, SPIN_MAX)
                if abs(state.grab.spin_ema) >= SPIN_MIN else 0.0
            )
    state.grab.prev_angle = smooth_angle
    state.grab.prev_time = now

    if mode == "move":
        smooth_x, smooth_y = state.grab.pos_filter.update(now, palm[0], palm[1])
        dx = smooth_x - state.grab.origin_signal_pos[0]
        dy = smooth_y - state.grab.origin_signal_pos[1]
        obj.offset[0] = state.grab.origin_object_offset[0] + dx
        obj.offset[1] = state.grab.origin_object_offset[1] + dy
        # Ce qui est posé DESSUS suit le même déplacement. On repart de la
        # position de départ de chaque élément (jamais de la frame
        # précédente) : c'est ce qui évite la dérive qu'un cumul
        # produirait immanquablement sur une pile.
        for child in state.canvas.descendants(obj.id):
            origin = state.grab.origin_children.get(child.id)
            if origin is not None:
                child.offset[0] = origin[0] + dx
                child.offset[1] = origin[1] + dy
        stack = len(state.grab.origin_children)
        state.set_status(f"DEPLACE ({obj.label})" + (f" + {stack} dessus" if stack else ""))
    else:
        smooth_depth = state.grab.yaw_filter(now, depth)
        smooth_pitch = state.grab.pitch_filter(now, pitch)
        obj.yaw = clamp(
            state.grab.origin_object_yaw
            + deadzone(smooth_depth - state.grab.origin_signal_depth, DEPTH_DEADZONE) * YAW_SENSITIVITY,
            -MAX_YAW, MAX_YAW,
        )
        obj.pitch = clamp(
            state.grab.origin_object_pitch
            + deadzone(smooth_pitch - state.grab.origin_signal_pitch, DEPTH_DEADZONE) * PITCH_SENSITIVITY,
            -MAX_PITCH, MAX_PITCH,
        )
        state.set_status(f"ROTATION 3D ({obj.label})")

    state.last_touched_id = obj.id


def _release_grab(state: AppState) -> None:
    """Fin de manipulation. C'est ICI qu'on regarde si l'élément a été POSÉ
    sur un autre : le lâcher est le seul moment où la question a un sens.

    Passer par une fonction unique plutôt que par cinq `state.grab = None`
    dispersés évite qu'un chemin de sortie oublie l'empilement — c'est
    exactement le genre d'oubli qui donne un comportement qui marche "une
    fois sur deux" sans qu'on comprenne pourquoi."""
    if state.grab is None:
        return
    obj = state.canvas.get_object(state.grab.object_id)
    state.grab = None
    if obj is None:
        return

    cx, cy = obj.center_screen()
    descendants = {o.id for o in state.canvas.descendants(obj.id)}
    best, best_area = None, math.inf
    for other in state.canvas.objects:
        if other.id == obj.id or other.id in descendants:
            continue  # ni sur soi-même, ni sur ce qu'on porte : ça ferait une boucle
        x0, y0, x1, y1 = other.transformed_bbox()
        if not (x0 - STACK_MARGIN <= cx <= x1 + STACK_MARGIN and y0 - STACK_MARGIN <= cy <= y1 + STACK_MARGIN):
            continue
        area = max(x1 - x0, 1) * max(y1 - y0, 1)
        if area < best_area:  # le plus PETIT support : le plus précis des candidats empilés
            best, best_area = other, area
    obj.attached_to = best.id if best is not None else None
    if best is not None:
        state.set_status(f"POSE SUR ({best.label})")


def _update_spin(state: AppState, dt: float) -> None:
    """Rotation continue : un élément lâché en tournant garde son élan et
    ralentit doucement.

    Seul le ROLL tourne sur l'élan, pas le yaw ni le pitch : ces deux-là
    viennent de signaux de profondeur bruités, et leur donner de l'inertie
    reviendrait à amplifier le bruit pendant plusieurs secondes après chaque
    lâcher."""
    if dt <= 0:
        return
    held = state.grab.object_id if state.grab is not None else None
    damping = math.exp(-SPIN_DAMPING * dt)
    for obj in state.canvas.objects:
        if obj.spin == 0.0 or obj.id == held:
            continue
        obj.roll += obj.spin * dt
        obj.spin *= damping
        if abs(obj.spin) < 0.15:
            obj.spin = 0.0


def _finish_stroke(state: AppState, now: float) -> None:
    """Ferme proprement le trait en cours. Un seul endroit pour le faire :
    la comète, le filtre de lissage et l'horloge de validation doivent être
    remis à zéro EN MÊME TEMPS que le trait, sinon le trait suivant démarre
    avec la queue de comète du précédent et un filtre ancré ailleurs."""
    state.canvas.end_stroke()
    state.pending_since = now
    state.was_drawing = False
    state.draw_miss_run = 0
    state.point_filter = None
    state.cursor_trail.clear()


def _auto_create_element(state: AppState, now: float, fr: FrameResult) -> None:
    """Après une courte pause, les traits en attente deviennent un élément —
    sans aucun geste de validation."""
    if fr.drawing or state.canvas.is_drawing or not state.canvas.pending_strokes:
        return
    if state.pending_since is None or (now - state.pending_since) < AUTO_VALIDATE_DELAY:
        return
    obj = state.canvas.validate(created_at=now, color_idx=state.pen_color_idx, recognizer=state.recognizer)
    state.pending_since = None
    if obj is not None:
        state.last_touched_id = obj.id
        state.set_status(f"ELEMENT CREE ({obj.label})")
        fr.auto_created = True


def _collect_recognitions(state: AppState, fr: FrameResult) -> None:
    """Récupère ce que le fil de fond a lu. C'est le SEUL endroit où un
    résultat de reconnaissance touche le canevas — donc toujours depuis le
    fil principal, donc sans le moindre verrou à poser ailleurs."""
    if state.recognizer is None:
        return
    for result in state.recognizer.poll():
        obj = state.canvas.apply_recognition(result)
        if obj is not None and result.text:
            fr.recognized.append(result.text)
            state.set_status(f"ECRITURE ({result.text})")


def _update_clear_all(state: AppState, hands: list[HandObservation], now: float, fr: FrameResult) -> None:
    """TOUT EFFACER : les deux mains ouvertes, puis on les retire."""
    both_open = len(hands) >= 2 and all(hs.is_open_palm(hd.landmarks) for hd in hands[:2])
    phase, progress, fired = state.clear_gesture.update(len(hands), both_open, now)
    fr.clear_phase = phase
    fr.clear_progress = progress
    if fired:
        fr.triggered = "Clear_All"


def _update_discrete_gestures(state: AppState, hand: HandObservation | None, now: float,
                              fr: FrameResult) -> None:
    """Les trois gestes "un coup" du classifieur : supprimer, couleur
    suivante, annuler.

    Tous les trois sont DÉSACTIVÉS pendant qu'un élément est tenu. Un geste
    mal classé ne doit jamais pouvoir supprimer, recolorer ou défaire ce
    qu'on a sous la main — et pendant une manipulation, la main est fermée,
    c'est-à-dire exactement la forme que le classifieur confond le plus
    volontiers avec un pouce levé ou baissé."""
    enabled = state.grab is None and hand is not None

    if state.delete_latch.update(enabled and hand.is_gesture("Thumb_Down"), now):
        fr.triggered = "Thumb_Down"
    if state.color_latch.update(enabled and hand.is_gesture("Thumb_Up"), now):
        fr.color_changed = True
    if state.undo_latch.update(enabled and hand.is_gesture("Victory", UNDO_CONFIDENCE), now):
        fr.undone = state.canvas.undo()


def _apply_color_change(state: AppState) -> None:
    """La couleur COURANTE avance — celle du pinceau. Ce qui est actif
    l'adopte aussitôt : l'élément sélectionné s'il y en a un, sinon le trait
    en cours, sinon le dernier élément touché. Un seul concept, une seule
    palette à l'écran, et la couleur se choisit AVANT de dessiner."""
    state.next_pen_color()
    target = state.selected_object()
    if target is None and not state.canvas.pending_strokes and not state.canvas.is_drawing:
        target = state.canvas.get_object(state.last_touched_id)
    if target is not None:
        target.color_idx = state.pen_color_idx
        state.set_status(f"COULEUR ({target.label})")
    else:
        state.set_status("COULEUR DU TRAIT")


def _apply_triggers(state: AppState, fr: FrameResult) -> None:
    if fr.triggered == "Clear_All":
        state.canvas.clear()
        state.grab = None
        state.lens = None
        state.selected_id = None
        state.last_touched_id = None
        state.pending_since = None
        state.was_drawing = False
        state.cursor_trail.clear()
        state.set_status("TOUT EFFACE (z pour restaurer)")
    elif fr.triggered == "Thumb_Down":
        # UNE SEULE RÈGLE : le pouce vers le bas supprime le DERNIER élément
        # dessiné — ou manipulé, si on vient de toucher à un autre. C'est
        # `last_touched_id`, mis à jour à la création comme à la prise.
        #
        # L'ancienne règle visait l'élément sélectionné en priorité. Elle
        # avait un défaut : une sélection oubliée quelque part faisait
        # disparaître un élément qu'on ne regardait même pas. Maintenant que
        # la sélection se relâche d'elle-même quand la main s'éloigne, viser
        # le dernier élément touché est à la fois plus simple à énoncer et
        # plus proche de ce qu'on voit.
        removed = state.canvas.delete_object(state.last_touched_id) if state.last_touched_id else None
        if removed is None:
            removed = state.canvas.delete_last_object()
        state.prune_selection()
        state.set_status(f"SUPPRIME ({removed.label})" if removed is not None else "RIEN A SUPPRIMER")

    if fr.undone is not None:
        state.prune_selection()
        state.set_status({"clear": "TOUT RESTAURE", "object": "ANNULE (element)"}.get(fr.undone, "ANNULE (trait)"))


def _update_drawing(state: AppState, raw_x: float, raw_y: float, tip_travel: float,
                    w: int, h: int, now: float, fr: FrameResult) -> None:
    """Le tracé lui-même.

    Le garde-fou du SAUT DE SUIVI est ici et nulle part ailleurs : quand le
    bout du doigt bondit de plus de `MAX_DRAW_JUMP_PX` en une image, ce
    n'est pas une main qui va vite, c'est le suivi qui a changé d'avis. On
    ferme le trait à l'endroit où il était plutôt que de tirer une droite en
    travers du dessin — et on ne recommence pas un trait depuis cette
    position-là non plus, tant qu'elle n'est pas confirmée à l'image
    suivante."""
    if state.canvas.is_drawing and tip_travel > MAX_DRAW_JUMP_PX:
        _finish_stroke(state, now)
        state.set_status("SAUT DE SUIVI (trait ferme)")
        return

    fr.drawing = True
    if state.point_filter is None or not state.canvas.is_drawing:
        # Un trait qui démarre LOIN du groupe en attente est presque
        # toujours un autre dessin : on ferme le groupe précédent tout de
        # suite plutôt que de les fusionner.
        if state.canvas.pending_strokes and state.canvas.distance_to_pending(raw_x, raw_y) > NEW_GROUP_DISTANCE:
            created = state.canvas.validate(
                created_at=now, color_idx=state.pen_color_idx, recognizer=state.recognizer
            )
            if created is not None:
                state.last_touched_id = created.id
                fr.auto_created = True
            state.pending_since = None
        state.point_filter = Point2DFilter(now, raw_x, raw_y)
        smooth_x, smooth_y = raw_x, raw_y
        state.canvas.start_stroke(now)
    else:
        smooth_x, smooth_y = state.point_filter.update(now, raw_x, raw_y)

    # Le trait reste DANS l'image : un point hors cadre déforme la boîte
    # englobante de l'élément, donc son maillage et sa zone de préhension.
    smooth_x = clamp(smooth_x, 0.0, w - 1.0)
    smooth_y = clamp(smooth_y, 0.0, h - 1.0)
    state.canvas.add_point(int(smooth_x), int(smooth_y))
    state.cursor_trail.append((smooth_x, smooth_y, now))
    state.set_status("DESSIN")


def process_frame(state: AppState, result, w: int, h: int, now: float) -> FrameResult:
    """Met à jour `state` d'après un résultat MediaPipe pour une frame, et
    renvoie ce qu'il faut pour l'affichage. Séparée de la boucle caméra pour
    rester testable sans webcam ni modèle."""
    fr = FrameResult()
    state.status_set_this_frame = False
    state.prune_selection()
    _collect_recognitions(state, fr)

    # Pas de temps borné : sans le plafond, une frame en retard (démarrage,
    # fenêtre déplacée, test qui avance l'horloge d'une seconde) ferait
    # faire un tour complet à un élément lancé sur son élan.
    dt = 0.0 if state.last_frame_time is None else clamp(now - state.last_frame_time, 0.0, 0.1)
    state.last_frame_time = now

    hands = observe_hands(result)
    primary = pick_primary(state, hands, w, h)
    _update_clear_all(state, hands, now, fr)
    palm_now: tuple[float, float] | None = None

    if primary is None:
        # Grâce sur la perte de suivi : perdre la main une frame ou deux est
        # très courant (flou de mouvement, main à moitié hors cadre), et
        # lâcher l'élément à chaque fois rendait toute manipulation pénible.
        state.lost_run += 1
        if state.grab is not None and state.lost_run > GRAB_LOST_GRACE:
            _release_grab(state)
        state.fist_run = state.open_run = state.lens_run = state.point_run = 0
        state.lens = None
        # `last_index_tip` est volontairement CONSERVÉ. En l'effaçant, la
        # main qui réapparaît ailleurs après deux frames perdues rentrait
        # dans le trait en cours sans que le garde-fou du saut de suivi
        # puisse s'en apercevoir : le trait traversait alors tout le dessin
        # pour la rejoindre.
        _update_discrete_gestures(state, None, now, fr)
    else:
        state.lost_run = 0
        lm = primary.landmarks
        raw_x, raw_y = hs.get_index_fingertip(lm, w, h)
        palm = hs.get_palm_center(lm, w, h)
        palm_now = palm
        state.last_palm = palm
        fr.cursor = (raw_x, raw_y)

        # Toutes les poses sont évaluées d'abord, puis départagées par
        # priorité. Les compter séparément évite qu'une pose "gagne" juste
        # parce qu'elle est testée en premier dans le code.
        #
        # Le piège qui a demandé ce découpage : une main "pouce vers le bas"
        # a ses QUATRE doigts repliés, donc elle est géométriquement un
        # poing. Sans exclusion explicite, demander une suppression
        # attrapait l'élément le plus proche au lieu de le supprimer. Même
        # raisonnement pour le signe V, qui est géométriquement la pose
        # "deux doigts tendus" par laquelle passe un trait rapide.
        ml_command = (
            primary.is_gesture("Thumb_Down")
            or primary.is_gesture("Thumb_Up")
            or primary.is_gesture("Victory", UNDO_CONFIDENCE)
        )
        lens_now = hs.is_lens_sign(lm)
        open_now = hs.is_open_palm(lm)
        fist_now = (not ml_command) and (
            hs.is_closed_fist(lm)
            or (primary.gesture_name == "Closed_Fist" and primary.confidence >= FIST_CONFIDENCE)
        )
        point_now = (not ml_command) and hs.is_pointing(lm)
        state.lens_run = state.lens_run + 1 if lens_now else 0
        state.open_run = state.open_run + 1 if open_now else 0
        state.fist_run = state.fist_run + 1 if fist_now else 0
        state.point_run = state.point_run + 1 if point_now else 0
        pose = hs.hand_pose_signal(lm)

        tip_travel = (
            math.dist((raw_x, raw_y), state.last_index_tip)
            if state.last_index_tip is not None else 0.0
        )
        state.last_index_tip = (raw_x, raw_y)

        # DESSINER, en trois niveaux de tolérance croissante — parce que
        # protéger la CONTINUITÉ d'un trait, c'est protéger l'écriture :
        #   * pose stricte (index seul), tenue `DRAW_ONSET_FRAMES` images,
        #     pour COMMENCER un trait — un index qui passe par là en
        #     changeant de pose ne doit pas laisser de trace, et c'est ce
        #     qui faisait « dessiner tout seul » ;
        #   * pose relâchée (index en tête, deux doigts au plus) pour le
        #     CONTINUER — en écrivant vite, le majeur se déplie tout seul ;
        #   * grâce de quelques frames même sans pose du tout, à condition
        #     que le doigt AVANCE encore vite : une main qui file écrit, une
        #     main qui ralentit a fini son trait.
        drawing_pose = state.point_run >= DRAW_ONSET_FRAMES or (
            state.canvas.is_drawing and not ml_command and hs.is_pointing_relaxed(lm)
        )
        moving_fast = FAST_DRAW_PX <= tip_travel <= MAX_DRAW_JUMP_PX
        draw_grace = (
            state.canvas.is_drawing
            and state.draw_miss_run <= DRAW_GRACE_FRAMES
            and moving_fast
            and not open_now
            and not lens_now
            and not ml_command
        )

        _update_discrete_gestures(state, primary, now, fr)

        if state.lens_run >= LENS_HOLD_FRAMES:
            # LOUPE — priorité maximale : on ne peut ni manipuler ni dessiner
            # en faisant ce signe, donc la manipulation se termine proprement.
            _release_grab(state)
            _update_lens(state, primary, w, h, now, fr)

        elif state.open_run >= OPEN_HAND_FRAMES and _selected_within_reach(state, palm) is not None:
            # MAIN OUVERTE SUR L'ÉLÉMENT SÉLECTIONNÉ -> rotation 3D sur place.
            state.lens = None
            target = _selected_within_reach(state, palm)
            if state.grab is None or state.grab.object_id != target.id:
                _release_grab(state)
                state.grab = GrabState(object_id=target.id, handedness=primary.handedness)
                _anchor_grab(state.grab, target, state.canvas, now, hs.wrist_roll_angle(lm),
                             hs.yaw_signal(lm), hs.pitch_signal(lm), palm, "rotate")
                state.miss_run = 0
            _update_manipulation(state, primary, palm, "rotate", now, fr)

        elif state.open_run >= OPEN_HAND_FRAMES:
            # MAIN OUVERTE AILLEURS -> lâcher et désélectionner. Une seule
            # pose, deux effets, départagés par la PROXIMITÉ : on ouvre la
            # main SUR l'élément pour le tourner, ailleurs pour le lâcher.
            state.lens = None
            _release_grab(state)
            if state.deselect():
                state.set_status("DESELECTION")

        elif state.grab is not None and state.grab.mode == "move":
            # DÉPLACEMENT EN COURS. Volontairement placé avant la suppression :
            # pendant qu'on déplace un élément, un "pouce vers le bas" mal
            # classé ne doit jamais pouvoir le supprimer sous la main.
            state.lens = None
            if pose == "closed":
                state.miss_run = 0
                _update_manipulation(state, primary, palm, "move", now, fr)
            else:
                # Pose pas encore franche : on met la manipulation en PAUSE
                # au lieu de lâcher. C'est ce qui permet de traverser une
                # frame mal classée sans tout perdre.
                state.miss_run += 1
                if state.miss_run > GRAB_RELEASE_GRACE:
                    _release_grab(state)
                    state.set_status("RELACHE")

        elif ml_command:
            # Branche volontairement vide : elle existe pour EMPÊCHER la
            # suite (dessin, poing) de s'exécuter. Le déclenchement, lui,
            # est géré par `_update_discrete_gestures`.
            state.lens = None
            _release_grab(state)

        elif drawing_pose or draw_grace:
            state.lens = None
            _update_drawing(state, raw_x, raw_y, tip_travel, w, h, now, fr)

        elif state.fist_run >= GRAB_ONSET_FRAMES:
            state.lens = None
            if state.grab is None:
                # Onset : il faut un élément sous (ou près de) la paume.
                target = state.canvas.find_object_at(palm[0], palm[1], margin=int(GRAB_REACH))
                if target is not None:
                    state.grab = GrabState(object_id=target.id, handedness=primary.handedness)
                    state.selected_id = target.id
                    state.last_touched_id = target.id
                    # Prendre un élément le DÉTACHE de ce qui était dessous :
                    # on est en train de le retirer de la pile.
                    target.attached_to = None
                    target.spin = 0.0
                    _anchor_grab(state.grab, target, state.canvas, now, hs.wrist_roll_angle(lm),
                                 hs.yaw_signal(lm), hs.pitch_signal(lm), palm, "move")
                    state.miss_run = 0
                    _update_manipulation(state, primary, palm, "move", now, fr)
                else:
                    state.set_status("POING (aucun element ici)")

        else:
            state.lens = None
            state.point_filter = None
            hovered = state.canvas.find_object_at(palm[0], palm[1])
            fr.hovered_object_id = hovered.id if hovered else None

        # Claquement de doigts — pas pendant le dessin ni pendant la loupe :
        # dans ces deux cas la main est occupée à autre chose, et écouter un
        # geste rapide y produirait surtout des faux positifs.
        if not fr.drawing and fr.lens is None and not fr.resizing:
            fr.snap_armed = state.snap.is_armed(primary.handedness, now)
            if state.snap.update(lm, primary.handedness, now):
                fr.snapped = True

    # ÉLOIGNER LA MAIN DÉSÉLECTIONNE — sauf pendant qu'on tient ou qu'on
    # redimensionne l'élément, où être loin fait partie du geste.
    if _update_auto_deselect(state, palm_now, now, busy=(state.grab is not None or fr.resizing)):
        state.set_status("DESELECTION (main eloignee)")

    # Fermeture du trait, AVEC la grâce : on ne coupe qu'après plusieurs
    # frames sans pose de dessin, pas dès la première.
    if fr.drawing:
        state.draw_miss_run = 0
        state.was_drawing = True
    elif state.was_drawing:
        state.draw_miss_run += 1
        if state.draw_miss_run > DRAW_GRACE_FRAMES:
            _finish_stroke(state, now)

    _update_spin(state, dt)
    _auto_create_element(state, now, fr)

    # Traînée du curseur : on oublie les points trop vieux, sinon la comète
    # resterait figée en l'air dès que la main s'arrête.
    while state.cursor_trail and (now - state.cursor_trail[0][2]) > TRAIL_MAX_AGE:
        state.cursor_trail.popleft()
    if fr.drawing:
        fr.trail = list(state.cursor_trail)

    if fr.snapped or fr.color_changed:
        fr.color_changed = True
        _apply_color_change(state)

    for obj in state.canvas.objects:
        obj.selected = obj.id == state.selected_id

    _apply_triggers(state, fr)

    # Le statut est recalculé À CHAQUE frame, jamais hérité de la
    # précédente : une branche qui l'a posé plus haut (prise, loupe,
    # claquement...) le garde, sinon on retombe sur l'état général.
    if not state.status_set_this_frame:
        if fr.drawing:
            state.set_status("DESSIN")
        elif fr.clear_phase in ("arm", "ready", "confirm"):
            state.last_status = "TOUT EFFACER ?"
        elif fr.hovered_object_id is not None:
            state.last_status = "SURVOL"
        elif state.selected_id is not None:
            state.last_status = "SELECTIONNE"
        else:
            state.last_status = "IDLE"

    return fr


# --------------------------------------------------------------------------
# Détecteur MediaPipe
# --------------------------------------------------------------------------
def build_recognizer(model_path: str) -> mp_vision.GestureRecognizer:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Télécharge-le (PowerShell) avec :\n"
            '  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/'
            'gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"'
            ' -OutFile "models\\gesture_recognizer.task"'
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.GestureRecognizerOptions(
        base_options=base_options,
        # DEUX mains détectées, UNE seule aux commandes. Il en faut deux
        # pour le geste "tout effacer" ; tout le reste passe par
        # `pick_primary`, qui garde la même main d'une frame à l'autre. La
        # v9 était passée à une seule main parce que l'application se fiait
        # à l'ORDRE de la liste renvoyée par MediaPipe — ordre qui change
        # d'une frame à l'autre. C'est ce défaut-là qui est corrigé, pas le
        # nombre de mains.
        num_hands=2,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.GestureRecognizer.create_from_options(options)


# --------------------------------------------------------------------------
# Boucle principale
# --------------------------------------------------------------------------
def save_snapshot(image: np.ndarray) -> str:
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    name = datetime.datetime.now().strftime("pinceau_%Y%m%d_%H%M%S.png")
    path = os.path.join(CAPTURE_DIR, name)
    cv2.imwrite(path, image)
    return path


def open_camera() -> cv2.VideoCapture:
    """Ouvre la webcam, en demandant la plus grande définition qu'elle
    accepte parmi celles qu'on sait afficher.

    OpenCV ne refuse jamais une définition : il donne la plus proche qu'il
    sait faire, silencieusement. On demande donc du plus grand au plus
    petit et on VÉRIFIE ce qu'on a réellement obtenu, au lieu de supposer.

    Plusieurs index sont essayés : sur un portable avec un dock ou une
    caméra virtuelle, l'index 0 n'est pas toujours la bonne — et un message
    clair vaut mieux qu'une fenêtre noire."""
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        for width, height in CAMERA_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if got_w >= width - 16:
                break
        return cap
    raise RuntimeError(
        "Cannot open the webcam (tried index 0, 1 and 2).\n"
        "Check that no other application is using it, and that Windows\n"
        "allows camera access (Settings > Privacy > Camera)."
    )


def main() -> None:
    engine_status = sketch.diagnose_text_recognition()

    recognizer = build_recognizer("models/gesture_recognizer.task")
    cap = open_camera()

    state = AppState(canvas=Canvas(), recognizer=async_recognition.RecognitionService())

    fps_history: collections.deque = collections.deque(maxlen=30)
    prev_frame_time = time.time()
    start_time = time.time()
    show_shadows = True
    show_debug = False
    window_sized = False
    toast, toast_until = "", 0.0

    # L'aide est affichée au démarrage — c'est là qu'on en a besoin — puis
    # s'efface d'elle-même au premier élément créé : à ce moment-là, elle a
    # servi, et le panneau couvre justement la zone où l'on dessine. `h` la
    # ramène à tout moment.
    show_help = True
    help_auto_hidden = False

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # La fenêtre est dimensionnée sur la PREMIÈRE image reçue, pas
            # sur ce qu'on a demandé à la caméra : les deux ne coïncident
            # pas toujours, et une fenêtre au mauvais format afficherait le
            # flux réduit dans un coin.
            if not window_sized:
                cv2.resizeWindow(WINDOW_TITLE, w, h)
                window_sized = True

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            now = time.time()
            fr = process_frame(state, result, w, h, now)

            if show_shadows:
                shadow = state.canvas.shadow_mask((h, w))
                if shadow is not None:
                    hud.apply_shadow(frame, shadow)

            if fr.cursor is not None and not fr.drawing:
                radius = max(5, int(8 * h / 540))
                cv2.circle(frame, (int(fr.cursor[0]), int(fr.cursor[1])), radius, (150, 150, 150), -1)

            # Les étiquettes numérotées ("#1 cercle") ne s'affichent qu'en
            # vue technique : à l'usage, elles racontent la machinerie
            # par-dessus le dessin. Le cadre vert suffit à dire ce qui est
            # sélectionné.
            canvas_img = state.canvas.render((h, w, 3), fr.hovered_object_id, state.pen_color,
                                             show_labels=show_debug)
            output = cv2.addWeighted(frame, 0.6, canvas_img, 1.0, 0)

            # La comète passe par-dessus tout : c'est le retour immédiat du
            # geste en cours, elle ne doit jamais être masquée par un élément.
            if fr.trail:
                hud.draw_trail(output, fr.trail, state.pen_color)

            # La loupe s'applique à l'image DÉJÀ composée : elle grossit donc
            # à la fois la vidéo et les éléments, et seulement dans son
            # disque. Le reste garde l'échelle 1:1, donc ce qu'on voit sous
            # sa main correspond toujours à ce que le suivi calcule.
            if fr.lens is not None:
                magnifier.draw_magnifier(output, fr.lens)

            fps_history.append(1.0 / max(now - prev_frame_time, 1e-6))
            prev_frame_time = now
            fps = sum(fps_history) / len(fps_history)

            if fr.auto_created and show_help and not help_auto_hidden:
                show_help, help_auto_hidden = False, True
                toast, toast_until = "help hidden - press h", now + 1.8

            hud.draw_hud(output, state, fr, show_help)
            if show_debug:
                hud.draw_debug(output, state, fr, fps, engine_status)
            if now < toast_until:
                hud.draw_toast(output, toast)

            cv2.imshow(WINDOW_TITLE, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_help, help_auto_hidden = not show_help, True
            elif key == ord("o"):
                show_shadows = not show_shadows
            elif key == ord("d"):
                # Vue technique : tout ce qui a été retiré de l'affichage
                # normal, plus le journal de la reconnaissance dans la
                # console. Rien n'a été perdu, c'est juste rangé.
                show_debug = not show_debug
                sketch.VERBOSE = show_debug
                toast, toast_until = f"debug {'on' if show_debug else 'off'}", now + 0.8
            elif key == ord("c"):
                _apply_color_change(state)
                toast, toast_until = "next colour", now + 0.8
            elif key == ord("z"):
                undone = state.canvas.undo()
                state.prune_selection()
                toast = {"clear": "everything restored", "object": "object removed",
                         "stroke": "stroke removed"}.get(undone, "nothing to undo")
                toast_until = now + 1.0
            elif key == ord("s"):
                toast, toast_until = f"saved: {save_snapshot(output)}", now + 1.6
    finally:
        # Fermer explicitement : sans ça, MediaPipe libère son détecteur au
        # ramasse-miettes, après l'interpréteur, et affiche une trace
        # d'erreur à la sortie alors que tout s'est bien passé.
        if state.recognizer is not None:
            state.recognizer.close()
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
