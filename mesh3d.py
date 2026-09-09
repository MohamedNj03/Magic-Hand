"""
mesh3d.py — Petit moteur 3D minimal (numpy + cv2 seulement, RIEN de plus à
installer) : des maillages faits de sommets + faces, une VRAIE rotation 3D
(matrices de rotation, pas un effet visuel), une projection vers l'écran,
et un rendu par tri de profondeur (painter's algorithm).

Portée assumée : ça suffit pour des solides CONVEXES simples (cube, prisme,
cylindre) et des "rubans" extrudés (une ligne à qui on donne une épaisseur)
— pas un moteur 3D général. Pas de z-buffer, pas de culling de faces
cachées : pour un solide convexe vu depuis l'extérieur, trier les faces
par profondeur moyenne et les dessiner de la plus loin à la plus proche
suffit à obtenir un rendu correct, sans la complexité d'un vrai moteur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Mesh3D:
    vertices: np.ndarray  # (N, 3) float, coordonnées LOCALES (centrées sur l'objet)
    faces: list[tuple[int, ...]]  # chaque face = indices dans `vertices` ; 2 = arête, >=3 = polygone
    edge_only_faces: set[int] = field(default_factory=set)  # faces à tracer en contour seulement (pas remplies)


def rotation_matrix(roll: float, yaw: float, pitch: float = 0.0) -> np.ndarray:
    """Combine roll (autour de Z, l'axe caméra — tourner le poignet comme
    une clé), yaw (autour de Y, l'axe vertical — incliner la main comme une
    porte qui s'ouvre) et pitch (autour de X, l'axe horizontal — basculer
    la main vers l'avant/l'arrière). Une VRAIE matrice de rotation 3D :
    valide pour n'importe quel angle, pas une approximation qui se dégrade.

    `pitch` est arrivé après coup et vaut 0 par défaut : tout code qui
    n'appelle qu'avec (roll, yaw) obtient exactement la même matrice
    qu'avant. Avec les trois angles, l'objet est orientable dans n'importe
    quelle direction — avant, deux axes seulement, donc certaines faces
    restaient inatteignables."""
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rot_z = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return rot_z @ rot_y @ rot_x


def project(vertices_3d: np.ndarray, center_2d: tuple[float, float], focal: float = 1400.0) -> np.ndarray:
    """Projection en perspective simple : la caméra regarde le long de +Z ;
    l'objet est placé devant elle. `focal` grand = quasi orthographique
    (peu d'effet de perspective), `focal` petit = perspective marquée."""
    if len(vertices_3d) == 0:
        return np.zeros((0, 2))
    cx, cy = center_2d
    z = vertices_3d[:, 2] + focal
    z = np.where(np.abs(z) < 1e-3, 1e-3, z)
    sx = vertices_3d[:, 0] * (focal / z) + cx
    sy = vertices_3d[:, 1] * (focal / z) + cy
    return np.stack([sx, sy], axis=1)


def _face_shade(rotated_vertices: np.ndarray, face: tuple[int, ...], light_dir=(0.4, -0.6, -0.7)) -> float:
    """Ombrage diffus simple selon l'angle entre la normale de la face et
    une direction de lumière fixe — juste pour donner un indice visuel de
    volume, pas un rendu physique."""
    if len(face) < 3:
        return 1.0
    p0, p1, p2 = (rotated_vertices[i] for i in face[:3])
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return 1.0
    normal = normal / norm
    light = np.array(light_dir, dtype=float)
    light = light / np.linalg.norm(light)
    return min(1.0, 0.45 + 0.55 * max(0.0, float(np.dot(normal, -light))))


def face_draw_order(rotated_vertices: np.ndarray, faces: list[tuple[int, ...]]) -> list[int]:
    """Indices des faces triés de la PLUS LOINTAINE à la PLUS PROCHE de la
    caméra (painter's algorithm : dessiner loin -> proche donne une
    occlusion correcte pour un solide convexe)."""
    depths = []
    for i, face in enumerate(faces):
        if len(face) < 2:
            continue
        avg_z = float(np.mean(rotated_vertices[list(face), 2]))
        depths.append((avg_z, i))
    depths.sort(key=lambda t: t[0], reverse=True)  # z grand (loin) d'abord
    return [i for _, i in depths]


def render_mesh(
    img: np.ndarray,
    mesh: Mesh3D,
    roll: float,
    yaw: float,
    center_2d: tuple[float, float],
    base_color: tuple[int, int, int],
    scale: float = 1.0,
    pitch: float = 0.0,
) -> None:
    if len(mesh.vertices) == 0:
        return
    rot = rotation_matrix(roll, yaw, pitch)
    rotated = (mesh.vertices * scale) @ rot.T
    screen = project(rotated, center_2d)

    for i in face_draw_order(rotated, mesh.faces):
        face = mesh.faces[i]
        pts = screen[list(face)].astype(np.int32)
        if len(face) >= 3 and i not in mesh.edge_only_faces:
            shade = _face_shade(rotated, face)
            fill_color = tuple(int(c * shade) for c in base_color)
            cv2.fillPoly(img, [pts], fill_color, cv2.LINE_AA)
            cv2.polylines(img, [pts], True, tuple(int(c * 0.55) for c in base_color), 1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], len(face) > 2, base_color, 2, cv2.LINE_AA)


def screen_bounds(
    mesh: Mesh3D, roll: float, yaw: float, center_2d: tuple[float, float], scale: float = 1.0,
    pitch: float = 0.0,
) -> tuple[int, int, int, int]:
    """bbox écran (x0,y0,x1,y1) du maillage tourné+projeté — pour le hit-test."""
    if len(mesh.vertices) == 0:
        cx, cy = center_2d
        return (int(cx), int(cy), int(cx), int(cy))
    rot = rotation_matrix(roll, yaw, pitch)
    rotated = (mesh.vertices * scale) @ rot.T
    screen = project(rotated, center_2d)
    return (int(screen[:, 0].min()), int(screen[:, 1].min()), int(screen[:, 0].max()), int(screen[:, 1].max()))


# --------------------------------------------------------------------------
# Constructeurs de maillages
# --------------------------------------------------------------------------
def make_prism_mesh(polygon_local: list[tuple[float, float]], depth: float) -> Mesh3D:
    """Extrude un polygone 2D FERMÉ (coordonnées locales, centrées) en un
    prisme 3D plein : face avant + face arrière + faces latérales."""
    n = len(polygon_local)
    if n < 3:
        return Mesh3D(vertices=np.zeros((0, 3)), faces=[])
    front = [(x, y, depth / 2) for x, y in polygon_local]
    back = [(x, y, -depth / 2) for x, y in polygon_local]
    vertices = np.array(front + back, dtype=float)
    faces: list[tuple[int, ...]] = [tuple(range(n)), tuple(reversed(range(n, 2 * n)))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return Mesh3D(vertices=vertices, faces=faces)


def make_ribbon_mesh(points_local: list[tuple[float, float]], depth: float) -> Mesh3D:
    """Extrude une polyligne (ouverte OU fermée) en un ruban 3D : la ligne
    existe à deux profondeurs reliées par de fines faces latérales — donne
    une épaisseur réelle à un trait de dessin libre, sans supposer une
    forme remplissable."""
    n = len(points_local)
    if n < 2:
        return Mesh3D(vertices=np.zeros((0, 3)), faces=[])
    front = [(x, y, depth / 2) for x, y in points_local]
    back = [(x, y, -depth / 2) for x, y in points_local]
    vertices = np.array(front + back, dtype=float)
    faces: list[tuple[int, ...]] = []
    for i in range(n - 1):
        faces.append((i, i + 1, n + i + 1, n + i))
    front_face_idx = len(faces)
    faces.append(tuple(range(n)))
    back_face_idx = len(faces)
    faces.append(tuple(range(n, 2 * n)))
    return Mesh3D(vertices=vertices, faces=faces, edge_only_faces={front_face_idx, back_face_idx})


def combine_meshes(meshes: list[Mesh3D]) -> Mesh3D:
    all_vertices = []
    all_faces: list[tuple[int, ...]] = []
    all_edge_only: set[int] = set()
    offset = 0
    for m in meshes:
        if len(m.vertices) == 0:
            continue
        all_vertices.append(m.vertices)
        block_start = len(all_faces)
        for face in m.faces:
            all_faces.append(tuple(idx + offset for idx in face))
        for local_idx in m.edge_only_faces:
            all_edge_only.add(block_start + local_idx)
        offset += len(m.vertices)
    if not all_vertices:
        return Mesh3D(vertices=np.zeros((0, 3)), faces=[])
    vertices = np.concatenate(all_vertices, axis=0)
    return Mesh3D(vertices=vertices, faces=all_faces, edge_only_faces=all_edge_only)


# --------------------------------------------------------------------------
# Ombre portée — la silhouette de l'objet, aplatie sur un sol imaginaire.
#
# Pourquoi c'est utile : à l'écran, une forme 3D posée sur une vidéo flotte
# sans repère. Une ombre au contact donne instantanément deux informations
# qu'aucun autre indice ne donne — que l'objet est POSÉ quelque part, et
# comment il est ORIENTÉ, puisque l'ombre suit sa vraie silhouette.
#
# Le calcul repart des mêmes sommets tournés et projetés que le rendu : ce
# n'est donc pas une ellipse décorative posée dessous, mais la vraie
# silhouette écrasée. Tourner l'objet déforme son ombre en conséquence.
# --------------------------------------------------------------------------
SHADOW_SQUASH = 0.20   # écrasement vertical : 1.0 = pas d'ombre au sol, 0 = ligne
SHADOW_SHEAR = 0.42    # décalage horizontal, comme une lumière venant d'en haut à gauche


def shadow_hull(
    mesh: Mesh3D, roll: float, yaw: float, center_2d: tuple[float, float], scale: float = 1.0,
    pitch: float = 0.0, squash: float = SHADOW_SQUASH, shear: float = SHADOW_SHEAR,
) -> np.ndarray:
    """Contour convexe (Nx2, entiers) de l'ombre portée, prêt à remplir.

    L'enveloppe CONVEXE plutôt que la silhouette exacte : une ombre est de
    toute façon diffuse une fois floutée, donc les creux ne se verraient
    pas, et l'enveloppe coûte une fraction du prix d'un vrai calcul de
    silhouette."""
    if len(mesh.vertices) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    rot = rotation_matrix(roll, yaw, pitch)
    rotated = (mesh.vertices * scale) @ rot.T
    pts = project(rotated, center_2d)
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.int32)

    ground_y = float(pts[:, 1].max())
    height_above = pts[:, 1] - ground_y          # négatif : plus c'est haut, plus c'est loin du sol
    flat_x = pts[:, 0] - height_above * shear
    flat_y = ground_y + height_above * squash
    flattened = np.stack([flat_x, flat_y], axis=1).astype(np.float32)

    hull = cv2.convexHull(flattened)
    return hull.reshape(-1, 2).astype(np.int32)
