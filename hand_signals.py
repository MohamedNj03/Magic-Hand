"""
hand_signals.py — Tout ce qui se lit GÉOMÉTRIQUEMENT sur une main (les 21
landmarks MediaPipe), sans passer par le classifieur ML de gestes.

v9 : le projet est passé à UNE SEULE MAIN. L'arrivée d'une deuxième main
perturbait le suivi (MediaPipe hésite entre les deux, l'ordre des mains
change d'une frame à l'autre, et les poses à deux mains se déclenchaient
par accident dès qu'une main traînait dans le champ). Toutes les fonctions
"deux mains" ont été retirées et remplacées par des poses à une main qui ne
se recouvrent pas :

    index tendu seul .................. dessiner
    poing fermé (0 doigt tendu) ....... tenir : déplacer + tourner
    main grande ouverte (4 doigts) .... lâcher et désélectionner
    signe de la loupe ................. zoomer / redimensionner
    claquement de doigts .............. couleur suivante

Le partage est fait pour être SANS ZONE GRISE : le nombre de doigts tendus
sépare déjà dessiner (1), tenir (0) et lâcher (4) ; la loupe se distingue de
la main ouverte par l'index REPLIÉ vers le pouce alors que les trois autres
doigts restent tendus.

Portée assumée : une seule caméra RGB ne donne pas de vraie profondeur. Les
signaux qui utilisent `z` (yaw, pitch) sont des PROXY relatifs bruités, pas
des angles mesurés.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Topologie des 21 landmarks MediaPipe
# --------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


def get_lm(hand_landmarks, idx: int):
    """Accès TOLÉRANT à un landmark : renvoie None s'il est absent.

    MediaPipe fournit toujours les 21 points, mais les harnais de test
    fabriquent des mains partielles. Sans ce garde-fou, ajouter une fonction
    qui lit le pouce ferait planter des tests qui ne parlent pas du pouce —
    on préfère que la fonction réponde "je ne sais pas" (False) plutôt que
    de lever une exception."""
    try:
        return hand_landmarks[idx]
    except (KeyError, IndexError, TypeError):
        return None


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# --------------------------------------------------------------------------
# Échelle de la main — indispensable pour que TOUS les seuils ci-dessous
# soient indépendants de la distance à la caméra. Une main près de
# l'objectif occupe 3x plus de pixels qu'une main au fond de la pièce : un
# seuil en valeur absolue marcherait à une distance et pas à l'autre.
# --------------------------------------------------------------------------
def palm_scale(hand_landmarks) -> float:
    """Longueur poignet -> base du majeur, en unités normalisées MediaPipe.
    Sert d'unité de mesure "taille de main"."""
    wrist = get_lm(hand_landmarks, WRIST)
    mid = get_lm(hand_landmarks, MIDDLE_MCP)
    if wrist is None or mid is None:
        return 1.0
    return max(_dist(wrist, mid), 1e-6)


# Marge en unités "taille de main". Elle sert de zone morte entre tendu et
# replié : sans elle, un doigt à mi-chemin bascule d'un état à l'autre à
# chaque frame, et la pose lue change de nom dix fois par seconde.
FINGER_EXTENSION_MARGIN = 0.12


def _finger_extended(hand_landmarks, tip: int, pip: int) -> bool:
    """Doigt tendu ou replié — mesuré SANS SUPPOSER QUE LA MAIN EST DROITE.

    Avant, on comparait les ordonnées (`tip.y < pip.y`), ce qui revient à
    exiger une main parfaitement verticale. Or personne ne dessine comme
    ça : dès qu'on incline le poignet — et on l'incline toujours un peu en
    traçant vers la droite — un doigt tendu passait pour replié. La pose
    "index tendu" se perdait en plein trait, et le trait se coupait. C'est
    la cause n°1 des dessins hachés.

    Ici on compare les DISTANCES AU POIGNET : un doigt tendu met son bout
    plus loin du poignet que son articulation, un doigt replié le ramène
    plus près. Ce rapport ne dépend ni de l'orientation de la main, ni de
    sa distance à la caméra (la marge est normalisée par `palm_scale`)."""
    wrist = get_lm(hand_landmarks, WRIST)
    tip_lm = get_lm(hand_landmarks, tip)
    pip_lm = get_lm(hand_landmarks, pip)
    if wrist is None or tip_lm is None or pip_lm is None:
        return False
    reach = _dist(tip_lm, wrist) - _dist(pip_lm, wrist)
    return reach > FINGER_EXTENSION_MARGIN * palm_scale(hand_landmarks)


def extended_finger_count(hand_landmarks) -> int:
    """Nombre de doigts tendus parmi index/majeur/annulaire/auriculaire (le
    pouce est exclu : sa géométrie TIP/PIP se comporte différemment)."""
    pairs = ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
    return sum(1 for tip, pip in pairs if _finger_extended(hand_landmarks, tip, pip))


def is_pointing(hand_landmarks) -> bool:
    """DESSINER : index tendu, les trois autres doigts repliés."""
    return (
        _finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP)
        and not _finger_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not _finger_extended(hand_landmarks, RING_TIP, RING_PIP)
        and not _finger_extended(hand_landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_pointing_relaxed(hand_landmarks) -> bool:
    """DESSINER, version tolérante — sert UNIQUEMENT à prolonger un trait
    déjà commencé, jamais à en démarrer un.

    Pendant qu'on écrit vite, le majeur se déplie souvent à moitié sans
    qu'on le veuille. Avec la pose stricte, le trait se coupait là ; avec
    celle-ci il continue, tant que l'index mène et qu'on n'est pas en
    train d'ouvrir la main (2 doigts au plus, la main ouverte en a 4)."""
    return _finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP) and extended_finger_count(hand_landmarks) <= 2


def hand_pose_signal(hand_landmarks) -> str:
    """'closed' | 'open' | 'ambiguous'. Utilisé pour CONTINUER une prise
    déjà commencée — volontairement tolérant, 'ambiguous' couvrant la
    transition entre les deux sans lâcher l'élément."""
    n = extended_finger_count(hand_landmarks)
    if n <= 1:
        return "closed"
    if n >= 3:
        return "open"
    return "ambiguous"


def is_closed_fist(hand_landmarks) -> bool:
    """TENIR : poing strict, zéro doigt tendu. Volontairement plus sévère
    que `hand_pose_signal() == 'closed'` (qui accepte 1 doigt tendu, donc
    AUSSI la main qui pointe pour dessiner). C'est la différence qui évite
    qu'un trait tracé près d'un élément l'attrape au lieu de dessiner."""
    return extended_finger_count(hand_landmarks) == 0


def is_open_palm(hand_landmarks) -> bool:
    """LÂCHER ET DÉSÉLECTIONNER : main grande ouverte, les QUATRE doigts
    tendus. Exiger les quatre (et pas trois) est ce qui sépare cette pose
    du signe de la loupe, où l'index est replié vers le pouce pendant que
    les trois autres restent tendus."""
    return extended_finger_count(hand_landmarks) == 4


def get_index_fingertip(hand_landmarks, frame_w: int, frame_h: int) -> tuple[float, float]:
    tip = hand_landmarks[INDEX_TIP]
    return tip.x * frame_w, tip.y * frame_h


def get_palm_center(hand_landmarks, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Centre approximatif de la paume (poignet + 4 bases de doigts). Plus
    stable que le bout de l'index pour saisir : il ne saute pas au moment
    où les doigts se replient en poing."""
    xs = [hand_landmarks[i].x for i in PALM_LANDMARKS]
    ys = [hand_landmarks[i].y for i in PALM_LANDMARKS]
    return (sum(xs) / len(xs)) * frame_w, (sum(ys) / len(ys)) * frame_h


def three_finger_tips(hand_landmarks, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    """Position moyenne des bouts du MAJEUR, de l'ANNULAIRE et de
    l'AURICULAIRE, en pixels — les trois doigts tendus du signe de la loupe.

    C'est la grandeur qui pilote la TAILLE d'un élément : monter les trois
    doigts agrandit, les baisser réduit. Elle a été choisie parce qu'elle
    bouge dans le bon sens pour les deux façons de faire le geste — lever
    toute la main, ou simplement replier les doigts vers le bas — alors
    qu'un signal fondé sur l'angle des doigts ne verrait que la seconde et
    un signal fondé sur la main que la première."""
    tips = [get_lm(hand_landmarks, idx) for idx in (MIDDLE_TIP, RING_TIP, PINKY_TIP)]
    tips = [t for t in tips if t is not None]
    if not tips:
        return None
    return (
        sum(t.x for t in tips) / len(tips) * frame_w,
        sum(t.y for t in tips) / len(tips) * frame_h,
    )


def wrist_roll_angle(hand_landmarks) -> float:
    """Angle (radians) du vecteur poignet -> base du majeur dans le plan
    image. Tourner le poignet comme une clé fait varier cet angle : signal
    ROLL. Fiable, purement 2D."""
    wrist = hand_landmarks[WRIST]
    mid = hand_landmarks[MIDDLE_MCP]
    return math.atan2(mid.y - wrist.y, mid.x - wrist.x)


def yaw_signal(hand_landmarks) -> float:
    """Écart de profondeur entre le bord index et le bord auriculaire :
    quand la paume pivote comme une porte, un bord se rapproche de la
    caméra et l'autre s'éloigne. PROXY bruité, relatif, jamais absolu."""
    return hand_landmarks[INDEX_MCP].z - hand_landmarks[PINKY_MCP].z


def pitch_signal(hand_landmarks) -> float:
    """Écart de profondeur poignet -> base du majeur : basculer la main vers
    l'avant/l'arrière fait varier cet écart. Même nature (et mêmes limites)
    que `yaw_signal`."""
    wrist = get_lm(hand_landmarks, WRIST)
    mid = get_lm(hand_landmarks, MIDDLE_MCP)
    if wrist is None or mid is None:
        return 0.0
    return getattr(mid, "z", 0.0) - getattr(wrist, "z", 0.0)


# --------------------------------------------------------------------------
# Le SIGNE DE LA LOUPE — pouce et index qui forment un cercle, les trois
# autres doigts tendus. C'est le geste du zoom (et du redimensionnement).
#
# Pourquoi celui-là : il fallait un geste à une main qui ne ressemble à
# AUCUN des autres. Le nombre de doigts tendus est déjà pris (0 = tenir,
# 1 = dessiner, 4 = lâcher), donc la loupe occupe le seul créneau qui
# restait — 3 doigts tendus, index replié — et elle apporte en prime une
# grandeur continue gratuite : l'OUVERTURE du cercle.
# --------------------------------------------------------------------------
LENS_MAX_APERTURE = 1.25   # au-delà, pouce et index sont trop écartés pour former un cercle
LENS_MIN_APERTURE = 0.12   # en-deçà, ils sont collés : évite une division par ~0


def thumb_index_aperture(hand_landmarks) -> float | None:
    """Ouverture du cercle pouce/index, NORMALISÉE par la taille de la main
    (donc comparable quelle que soit la distance à la caméra). None si le
    pouce n'est pas disponible."""
    thumb_tip = get_lm(hand_landmarks, THUMB_TIP)
    index_tip = get_lm(hand_landmarks, INDEX_TIP)
    if thumb_tip is None or index_tip is None:
        return None
    return max(_dist(thumb_tip, index_tip) / palm_scale(hand_landmarks), LENS_MIN_APERTURE)


def is_lens_sign(hand_landmarks) -> bool:
    """Index REPLIÉ vers le pouce, majeur + annulaire + auriculaire tendus,
    et cercle pas trop grand ouvert."""
    if _finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP):
        return False
    if not (
        _finger_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and _finger_extended(hand_landmarks, RING_TIP, RING_PIP)
        and _finger_extended(hand_landmarks, PINKY_TIP, PINKY_PIP)
    ):
        return False
    aperture = thumb_index_aperture(hand_landmarks)
    return aperture is not None and aperture <= LENS_MAX_APERTURE


def lens_geometry(hand_landmarks, frame_w: int, frame_h: int) -> tuple[tuple[float, float], float] | None:
    """(centre, rayon en pixels) du cercle formé par le pouce et l'index —
    littéralement le verre de la loupe, tel qu'il sera dessiné à l'écran."""
    thumb_tip = get_lm(hand_landmarks, THUMB_TIP)
    index_tip = get_lm(hand_landmarks, INDEX_TIP)
    if thumb_tip is None or index_tip is None:
        return None
    tx, ty = thumb_tip.x * frame_w, thumb_tip.y * frame_h
    ix, iy = index_tip.x * frame_w, index_tip.y * frame_h
    center = ((tx + ix) / 2, (ty + iy) / 2)
    radius = math.hypot(ix - tx, iy - ty) / 2
    return center, radius


# --------------------------------------------------------------------------
# CLAQUEMENT DE DOIGTS — v2, entièrement réécrit.
#
# MediaPipe n'a aucun geste "claquement" : il faut le reconnaître à la main.
# La v1 était une machine à deux états (contact -> séparation) et ratait
# beaucoup de claquements réels. Trois raisons, trouvées en analysant ce qui
# se passe réellement pendant le geste :
#
#   1. Un claquement dure 50-80 ms. À 30 FPS (33 ms par image), la frame de
#      CONTACT peut tomber entre deux images : la v1 n'était alors jamais
#      "armée" et ne pouvait plus rien reconnaître ensuite.
#   2. Le pouce est flou pendant le mouvement, donc un landmark isolé est
#      parfois aberrant : un seul mauvais échantillon cassait l'état.
#   3. Le seul discriminant était "la main reste fermée". Insuffisant :
#      passer du poing au doigt pointé fait aussi monter la distance.
#
# La v2 garde un HISTORIQUE court (distance + horodatage) et y cherche un
# MOTIF : un contact récent, puis une séparation à grande VITESSE. Regarder
# en arrière sur une fenêtre de 0,3 s rattrape la frame de contact manquée,
# et la vitesse est ce qui sépare vraiment un claquement (très brusque)
# d'une main qui se déplie (lent), quel que soit le nombre de doigts au bout.
# --------------------------------------------------------------------------
SNAP_CONTACT = 0.50        # distance pouce<->majeur (unités main) sous laquelle il y a contact
SNAP_RELEASE = 0.95        # ...et au-dessus de laquelle la séparation est faite
SNAP_MIN_TRAVEL = 0.45     # écart minimal parcouru entre le contact et la séparation
SNAP_MIN_SPEED = 6.0       # unités main / seconde — LE discriminant contre "la main s'ouvre"
SNAP_WINDOW = 0.30         # s — au-delà, le contact est trop vieux pour compter
SNAP_COOLDOWN = 0.45       # s — évite qu'un seul claquement compte double
SNAP_MAX_EXTENDED = 2      # un claquement laisse la main fermée ; une paume ouverte en a 4


@dataclass
class _HandSnapHistory:
    samples: deque = field(default_factory=lambda: deque(maxlen=14))  # (t, distance, doigts tendus)
    last_fire: float = -1e9
    armed_until: float = -1e9   # jusqu'à quand un contact récent reste valable (affichage)


@dataclass
class SnapDetector:
    """Un état par main (clé = handedness). En mode une seule main il n'y en
    a qu'un, mais garder la clé évite de tout mélanger si MediaPipe change
    d'avis sur la main qu'il suit."""

    hands: dict[str, _HandSnapHistory] = field(default_factory=dict)

    def _distance(self, hand_landmarks) -> float | None:
        thumb_tip = get_lm(hand_landmarks, THUMB_TIP)
        middle_tip = get_lm(hand_landmarks, MIDDLE_TIP)
        if thumb_tip is None or middle_tip is None:
            return None
        return _dist(thumb_tip, middle_tip) / palm_scale(hand_landmarks)

    def update(self, hand_landmarks, handedness: str | None, now: float) -> bool:
        """True sur la frame EXACTE où un claquement vient d'être reconnu."""
        key = handedness or "?"
        hist = self.hands.setdefault(key, _HandSnapHistory())

        d = self._distance(hand_landmarks)
        if d is None:
            return False  # pouce indisponible : la fonctionnalité se désactive, sans erreur

        n_extended = extended_finger_count(hand_landmarks)
        hist.samples.append((now, d, n_extended))
        if d < SNAP_CONTACT and n_extended <= SNAP_MAX_EXTENDED:
            hist.armed_until = now + SNAP_WINDOW

        # Frame de séparation ? Sinon il n'y a rien à chercher.
        if d < SNAP_RELEASE or n_extended > SNAP_MAX_EXTENDED:
            return False
        if now - hist.last_fire < SNAP_COOLDOWN:
            return False

        # On remonte l'historique jusqu'au contact le plus RÉCENT encore dans
        # la fenêtre. Le plus récent, et pas le premier trouvé : c'est lui
        # qui donne la vitesse réelle du geste.
        for t_old, d_old, n_old in reversed(hist.samples):
            if now - t_old > SNAP_WINDOW:
                break
            if d_old >= SNAP_CONTACT or n_old > SNAP_MAX_EXTENDED:
                continue
            travel = d - d_old
            elapsed = now - t_old
            if elapsed <= 1e-6:
                continue
            if travel >= SNAP_MIN_TRAVEL and (travel / elapsed) >= SNAP_MIN_SPEED:
                hist.last_fire = now
                hist.samples.clear()   # le même événement ne peut pas se déclencher deux fois
                hist.armed_until = -1e9
                return True
        return False

    def is_armed(self, handedness: str | None, now: float) -> bool:
        """Le contact pouce/majeur a été vu récemment : la main est en
        position de claquer. Sert UNIQUEMENT à l'affichage — voir un témoin
        s'allumer est ce qui rend le geste apprenable, au lieu d'une boîte
        noire qui marche une fois sur trois sans qu'on sache pourquoi."""
        hist = self.hands.get(handedness or "?")
        return hist is not None and now <= hist.armed_until

    def reset(self) -> None:
        self.hands.clear()
