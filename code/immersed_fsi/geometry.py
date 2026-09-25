"""Reusable geometry utilities for immersed FSI cases.

This module gathers the geometry code that was previously spread across case
drivers and helper scripts. It covers mesh connectivity normalization,
boundary-edge descriptions, area and perimeter measurements, rectangular
obstacle masks on Cartesian grids, MAC-grid geometric indexing, and lightweight
solid mesh generators used by the benchmark cases. It intentionally avoids
fluid or solid system assembly and plotting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


CIRCLE_REGION = 0
TAIL_REGION = 1
REGION_NAMES = {
    CIRCLE_REGION: "circle",
    TAIL_REGION: "tail",
}

SIDE_NAMES = {
    0: "bottom",
    1: "right",
    2: "top",
    3: "left",
}
SIDE_NORMALS = {
    "bottom": (0.0, -1.0),
    "right": (1.0, 0.0),
    "top": (0.0, 1.0),
    "left": (-1.0, 0.0),
}
SIDE_TANGENTS = {
    "bottom": (1.0, 0.0),
    "right": (0.0, 1.0),
    "top": (-1.0, 0.0),
    "left": (0.0, -1.0),
}


@dataclass(frozen=True)
class Material:
    """Linear elastic material data attached to generated solid regions."""

    E: float
    nu: float
    rho: float


@dataclass(frozen=True)
class TurekGeometryConfig:
    """Geometry and mesh controls for the Turek-Hron circle-tail solid."""

    channel_length: float = 2.5
    channel_height: float = 0.41
    circle_center: tuple[float, float] = (0.2, 0.2)
    circle_radius: float = 0.05
    tail_tip_x: float = 0.6
    tail_height: float = 0.02
    tail_x0: float | None = None
    edge_length: float = 0.005
    core_radius: float = 0.025
    boundary_flag: int = -2
    fix_tail_left: bool = False
    plane_stress: bool = False


@dataclass(frozen=True)
class TurekSolidMesh:
    """Triangular Turek solid mesh with region and material arrays."""

    P: np.ndarray
    T: np.ndarray
    bdry: np.ndarray
    region: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    rho: np.ndarray
    dirichlet_nodes: np.ndarray
    config: TurekGeometryConfig


@dataclass(frozen=True)
class SpikeMeshConfig:
    """Default controls for single triangular spike mesh generation."""

    lx_f: float = 1.5
    ly_f: float = 0.7
    fluid_h: float = 0.01
    triangle_center_x: float = 0.51
    triangle_base_width: float = 0.1
    triangle_height: float = 0.46
    solid_edge_length: float = 0.05
    regular_top_edge_factor: float = 0.6
    refine_solid_base: bool = True
    solid_base_layer_eps: float = 5.0e-4


@dataclass(frozen=True)
class SpikeMesh:
    """Reference spike mesh together with its fluid grid metadata."""

    P: np.ndarray
    T: np.ndarray
    B: np.ndarray
    fixed_nodes: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float


@dataclass(frozen=True)
class StarConfig:
    """Controls for the freely immersed star mesh setup."""

    points: int = 12
    outer_radius: float = 0.23
    inner_radius: float = 0.145
    height: float = 0.46
    solid_h: float = 0.0225
    fluid_h: float = 0.005
    lx_f: float = 1.5
    ly_f: float = 0.7


def point_matrix(P, *, copy=False):
    """Return coordinates with shape ``(2, n_points)``."""
    P = np.array(P, dtype=float, copy=copy)
    if P.ndim != 2:
        raise ValueError("P must be a 2D coordinate array.")
    if P.shape[0] != 2 and P.shape[1] == 2:
        P = P.T.copy() if copy else P.T
    if P.shape[0] != 2:
        raise ValueError("P must have shape (2, n_points) or (n_points, 2).")
    return P


def triangle_connectivity(T):
    """Return triangular connectivity with shape ``(n_elem, 3)``."""
    T = np.asarray(T, dtype=int)
    if T.ndim != 2:
        raise ValueError("T must be a 2D array.")
    if T.shape[1] != 3 and T.shape[0] == 3:
        T = T.T
    if T.shape[1] != 3:
        raise ValueError("T must have three nodes per triangle.")
    return T


def quadrilateral_connectivity(T):
    """Return quadrilateral connectivity with shape ``(n_elem, 4)``."""
    T = np.asarray(T, dtype=int)
    if T.ndim != 2:
        raise ValueError("T must be a 2D array.")
    if T.shape[1] != 4 and T.shape[0] == 4:
        T = T.T
    if T.shape[1] != 4:
        raise ValueError("T must have four nodes per quadrilateral.")
    return T


def orient_triangles_counterclockwise(P, T):
    """Orient triangular elements counter-clockwise in physical coordinates."""
    P = point_matrix(P)
    T = triangle_connectivity(T).copy()
    x = P[0, T]
    y = P[1, T]
    signed_double_area = (
        (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    )
    T[signed_double_area < 0.0] = T[signed_double_area < 0.0][:, [0, 2, 1]]
    return T


def triangular_boundary_description(T, flag=-2):
    """Return exterior P1 triangle edges as ``[flag, elem, node0, node1, side]``."""
    T = triangle_connectivity(T)

    edge_map = {}
    local_edges = ((0, 1), (1, 2), (2, 0))
    for elem, nodes in enumerate(T):
        for local_side, (i0, i1) in enumerate(local_edges):
            n0 = int(nodes[i0])
            n1 = int(nodes[i1])
            key = tuple(sorted((n0, n1)))
            edge_map.setdefault(key, []).append((elem, n0, n1, local_side))

    bdry_list = [
        [flag, elem, n0, n1, local_side]
        for entries in edge_map.values()
        if len(entries) == 1
        for elem, n0, n1, local_side in entries
    ]
    if not bdry_list:
        return np.empty((5, 0), dtype=int)
    return np.asarray(bdry_list, dtype=int).T


def boundary_nodes(bdry, flags=None):
    """Return unique nodes touched by boundary edges, optionally filtered by flag."""
    bdry = np.asarray(bdry)
    if bdry.ndim != 2 or bdry.shape[0] < 4:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")
    if bdry.shape[1] == 0:
        return np.array([], dtype=np.int64)
    if flags is None:
        selected = np.ones(bdry.shape[1], dtype=bool)
    else:
        selected = np.isin(bdry[0, :], np.atleast_1d(flags))
    return np.unique(bdry[2:4, selected]).astype(np.int64, copy=False)


def dirichlet_nodes_from_bdry(bdry, flag=-1):
    """Return nodes touching boundary edges marked with the Dirichlet flag."""
    return boundary_nodes(bdry, flags=flag)


def corner_nodes_from_bdry(bdry):
    """Return boundary nodes shared by more than one geometric side label."""
    bdry = np.asarray(bdry)
    if bdry.ndim != 2 or bdry.shape[0] < 5:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")
    edge_nodes = bdry[2:4, :].astype(int)
    edge_sides = bdry[4, :].astype(int)
    sides_by_node = {}
    for side, nodes in zip(edge_sides, edge_nodes.T):
        for node in nodes:
            sides_by_node.setdefault(int(node), set()).add(int(side))
    return np.array(
        sorted(node for node, sides in sides_by_node.items() if len(sides) > 1),
        dtype=np.int64,
    )


def nearest_node(P, point):
    """Return the index of the mesh node nearest to a physical point."""
    P = point_matrix(P)
    point = np.asarray(point, dtype=float).reshape(2)
    dist2 = (P[0] - point[0]) ** 2 + (P[1] - point[1]) ** 2
    return int(np.argmin(dist2))


def nearest_distinct_nodes(P, targets, name="target"):
    """Return nearest node indices for targets and require them to be distinct."""
    P = point_matrix(P)
    nodes = []
    for target in targets:
        nodes.append(nearest_node(P, target))

    nodes = np.asarray(nodes, dtype=np.int64)
    if np.unique(nodes).size != len(nodes):
        raise ValueError(f"Could not identify distinct {name} nodes.")
    return nodes


def mark_wall_dirichlet_edges(P, bdry, wall_y, flag=-1, atol=1.0e-12):
    """Mark boundary edges lying on a horizontal wall as clamped."""
    P = point_matrix(P)
    bdry = np.asarray(bdry, dtype=int).copy()
    wall = np.all(np.isclose(P[1, bdry[2:4]], wall_y, atol=atol), axis=0)
    bdry[0, wall] = flag
    return bdry


def label_polygon_boundary_sides(P, bdry, vertices, label_offset=0):
    """Replace local boundary-side IDs with reference polygon side labels."""
    P = point_matrix(P)
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("vertices must have shape (n_vertices, 2).")

    bdry = np.asarray(bdry, dtype=int).copy()
    side_start = vertices
    side_end = np.roll(vertices, -1, axis=0)
    tangent = side_end - side_start
    length2 = np.sum(tangent**2, axis=1)
    if np.any(length2 <= 0.0):
        raise ValueError("Degenerate polygon side.")

    labels = []
    scale = max(float(np.ptp(P[0])), float(np.ptp(P[1])), 1.0)
    side_tol = 1.0e-9 * scale
    for edge_nodes in bdry[2:4, :].T:
        points = P[:, edge_nodes].T
        best_label = None
        best_error = np.inf
        for label in range(len(vertices)):
            offset = points - side_start[label]
            cross = (
                offset[:, 0] * tangent[label, 1]
                - offset[:, 1] * tangent[label, 0]
            ) / np.sqrt(length2[label])
            fraction = np.sum(offset * tangent[label], axis=1) / length2[label]
            outside = np.maximum(0.0, -fraction) + np.maximum(0.0, fraction - 1.0)
            error = max(float(np.max(np.abs(cross))), float(np.max(outside)))
            if error < best_error:
                best_error = error
                best_label = label
        if best_error > side_tol:
            raise ValueError("Could not assign a boundary edge to a polygon side.")
        labels.append(label_offset + best_label)

    bdry[4, :] = labels
    return bdry


def combine_solid_meshes(meshes):
    """Merge disconnected triangular solid components into one ``P, T, bdry`` tuple."""
    points, triangles, boundaries = [], [], []
    node_offset = 0
    elem_offset = 0
    all_have_boundary = True

    for mesh in meshes:
        if len(mesh) == 2:
            P, T = mesh
            bdry = None
            all_have_boundary = False
        elif len(mesh) == 3:
            P, T, bdry = mesh
        else:
            raise ValueError("Each mesh must be (P, T) or (P, T, bdry).")

        P = point_matrix(P)
        T = triangle_connectivity(T)
        points.append(P)
        triangles.append(T + node_offset)
        if bdry is not None:
            bdry = np.asarray(bdry, dtype=int).copy()
            bdry[1, :] += elem_offset
            bdry[2:4, :] += node_offset
            boundaries.append(bdry)
        node_offset += P.shape[1]
        elem_offset += T.shape[0]

    P = np.hstack(points)
    T = np.vstack(triangles)
    bdry = np.hstack(boundaries) if all_have_boundary else triangular_boundary_description(T)
    return P, T, bdry


def triangle_areas(P, T):
    """Return the positive physical area of each triangular element."""
    P = point_matrix(P)
    T = triangle_connectivity(T)
    x0 = P[0, T[:, 0]]
    y0 = P[1, T[:, 0]]
    x1 = P[0, T[:, 1]]
    y1 = P[1, T[:, 1]]
    x2 = P[0, T[:, 2]]
    y2 = P[1, T[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def solid_area(P, T):
    """Return the total area covered by a triangular solid mesh."""
    return float(np.sum(triangle_areas(P, T)))


def boundary_perimeter(P, bdry):
    """Return the total length of all boundary edges in ``bdry``."""
    P = point_matrix(P)
    bdry = np.asarray(bdry)
    if bdry.ndim != 2 or bdry.shape[0] < 4:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")
    if bdry.shape[1] == 0:
        return 0.0
    edges = bdry[2:4, :].astype(int).T
    dx = P[0, edges[:, 1]] - P[0, edges[:, 0]]
    dy = P[1, edges[:, 1]] - P[1, edges[:, 0]]
    return float(np.sum(np.hypot(dx, dy)))


def triangle_quality(P, T):
    """Return 1 for equilateral triangles and values near 0 for degenerate ones."""
    P = point_matrix(P)
    T = triangle_connectivity(T)
    points = P.T[T]
    a2 = np.sum((points[:, 1] - points[:, 0]) ** 2, axis=1)
    b2 = np.sum((points[:, 2] - points[:, 1]) ** 2, axis=1)
    c2 = np.sum((points[:, 0] - points[:, 2]) ** 2, axis=1)
    double_area = np.abs(
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 2, 0] - points[:, 0, 0]) * (points[:, 1, 1] - points[:, 0, 1])
    )
    return 2.0 * np.sqrt(3.0) * double_area / np.maximum(a2 + b2 + c2, 1.0e-300)


def mesh_quality_summary(P, T):
    """Return compact quality statistics for a triangular mesh."""
    quality = triangle_quality(P, T)
    return {
        "min": float(np.min(quality)),
        "p10": float(np.percentile(quality, 10)),
        "mean": float(np.mean(quality)),
    }


def unique_points(points, decimals=14):
    """Return unique 2D points using rounded coordinates for robustness."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    _, idx = np.unique(np.round(points, decimals=decimals), axis=0, return_index=True)
    return points[np.sort(idx)]


def segment_points(p0, p1, edge_length):
    """Sample a segment with approximately ``edge_length`` spacing."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = np.linalg.norm(p1 - p0)
    if length == 0.0:
        return p0.reshape(1, 2)
    n = max(1, int(np.ceil(length / edge_length)))
    s = np.linspace(0.0, 1.0, n + 1)
    return (1.0 - s[:, None]) * p0 + s[:, None] * p1


def structured_rectangle_mesh(nx, ny, lx, ly, origin=(0.0, 0.0), flags=None):
    """Build a structured Q1 rectangle mesh as ``P, T, bdry``."""
    if flags is None:
        flags = np.array([-2, -2, -2, -2])
    flags = np.asarray(flags, dtype=int).reshape(4)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("lx and ly must be positive.")

    x0, y0 = np.asarray(origin, dtype=float).reshape(2)
    x = np.linspace(x0, x0 + lx, nx + 1)
    y = np.linspace(y0, y0 + ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.vstack((X.ravel(), Y.ravel()))

    ix, jy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    n1 = ix * (ny + 1) + jy
    n2 = n1 + (ny + 1)
    n3 = n2 + 1
    n4 = n1 + 1
    T = np.stack([n1, n2, n3, n4], axis=-1).reshape(-1, 4).T

    bdry_list = []
    bdry_list += [[flags[0], elem, T[0, elem], T[1, elem], 0] for elem in range(0, nx * ny, ny)]
    bdry_list += [
        [flags[1], elem, T[1, elem], T[2, elem], 1]
        for elem in range((nx - 1) * ny, nx * ny)
    ]
    bdry_list += [
        [flags[2], elem, T[2, elem], T[3, elem], 2]
        for elem in range(nx * ny - 1, -1, -ny)
    ]
    bdry_list += [[flags[3], elem, T[3, elem], T[0, elem], 3] for elem in range(ny - 1, -1, -1)]
    bdry = np.asarray(bdry_list, dtype=int).T
    return P, T, bdry


def rectangular_obstacle_mesh(nx, ny, x0, y0, lx, ly, flags=None):
    """Build the historical rectangular obstacle mesh tuple ``pos_x, pos_y, T, bdry``."""
    P, T, bdry = structured_rectangle_mesh(nx, ny, lx, ly, origin=(x0, y0), flags=flags)
    return P[0], P[1], T, bdry


def solid_node_coordinates(nx, ny, lx, ly):
    """Return structured Q1 node coordinates on a rectangle starting at the origin."""
    P, _, _ = structured_rectangle_mesh(nx, ny, lx, ly)
    return P[0], P[1]


def triangular_obstacle_mesh(vertices, edge_length, bottom_flag=-1, boundary_flag=-2):
    """Build a structured barycentric P1 mesh of one triangular obstacle."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must have shape (3, 2).")
    if edge_length <= 0.0:
        raise ValueError("edge_length must be positive.")

    edge_01 = vertices[1] - vertices[0]
    edge_02 = vertices[2] - vertices[0]
    signed_area = 0.5 * (edge_01[0] * edge_02[1] - edge_01[1] * edge_02[0])
    if abs(signed_area) < 1.0e-14:
        raise ValueError("vertices define a degenerate triangle.")
    if signed_area < 0.0:
        vertices = vertices[[0, 2, 1]]

    edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    n = max(1, int(np.ceil(edge_lengths.max() / edge_length)))

    node_index = {}
    points = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            xi = i / n
            eta = j / n
            point = (
                (1.0 - xi - eta) * vertices[0]
                + xi * vertices[1]
                + eta * vertices[2]
            )
            node_index[(i, j)] = len(points)
            points.append(point)

    triangles = []
    for i in range(n):
        for j in range(n - i):
            triangles.append([node_index[(i, j)], node_index[(i + 1, j)], node_index[(i, j + 1)]])
            if j < n - i - 1:
                triangles.append(
                    [
                        node_index[(i + 1, j)],
                        node_index[(i + 1, j + 1)],
                        node_index[(i, j + 1)],
                    ]
                )

    P = np.asarray(points, dtype=float).T
    T = orient_triangles_counterclockwise(P, np.asarray(triangles, dtype=int))
    bdry = triangular_boundary_description(T, flag=boundary_flag)

    bottom_y = float(np.min(vertices[:, 1]))
    bottom_nodes = np.flatnonzero(np.isclose(P[1, :], bottom_y, atol=1.0e-12))
    bottom_node_set = set(int(node) for node in bottom_nodes)
    for edge_id in range(bdry.shape[1]):
        n0 = int(bdry[2, edge_id])
        n1 = int(bdry[3, edge_id])
        if n0 in bottom_node_set and n1 in bottom_node_set:
            bdry[0, edge_id] = bottom_flag

    return P[0, :], P[1, :], T, bdry, bottom_nodes


def insert_solid_base_layer(P, triangles, height):
    """Insert one horizontal layer node per base node below ``height``."""
    P = point_matrix(P)
    triangles = triangle_connectivity(triangles)
    points = [point.copy() for point in P.T]
    intersections = {}
    node_groups = {}
    edges = np.unique(
        np.sort(
            np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
    )
    for i, j in edges:
        if (P[1, i] < height) == (P[1, j] < height):
            continue
        base_node = int(i if P[1, i] < height else j)
        fraction = (height - P[1, i]) / (P[1, j] - P[1, i])
        point = P[:, i] + fraction * (P[:, j] - P[:, i])
        node_groups.setdefault(base_node, []).append(((int(i), int(j)), point))

    for group in node_groups.values():
        point = np.mean([entry[1] for entry in group], axis=0)
        point[1] = height
        for key, _ in group:
            intersections[tuple(sorted(key))] = len(points)
        points.append(point)

    def intersection(i, j):
        return intersections[tuple(sorted((int(i), int(j))))]

    refined = []
    for nodes in triangles:
        below = P[1, nodes] < height
        if not np.any(below) or np.all(P[1, nodes] <= height):
            refined.append(nodes.tolist())
            continue
        if below.sum() == 1:
            a, b, c = np.roll(nodes, -int(np.flatnonzero(below)[0]))
            ab, ac = intersection(a, b), intersection(a, c)
            pieces = ([a, ab, ac], [ab, b, c], [ab, c, ac])
        else:
            a, b, c = np.roll(nodes, 2 - int(np.flatnonzero(~below)[0]))
            bc, ac = intersection(b, c), intersection(a, c)
            pieces = ([a, b, bc], [a, bc, ac], [ac, bc, c])
        refined.extend(piece for piece in pieces if len(set(piece)) == 3)

    P = np.asarray(points).T
    T = orient_triangles_counterclockwise(P, np.asarray(refined, dtype=int))
    bdry = triangular_boundary_description(T)
    bottom = np.all(np.isclose(P[1, bdry[2:4]], 0.0, atol=1.0e-12), axis=0)
    bdry[0, bottom] = -1
    return P, T, bdry


def validate_single_spike_config(config):
    """Validate one-spike settings and return the fluid grid and vertices."""
    positive_fields = [
        "lx_f",
        "ly_f",
        "fluid_h",
        "triangle_base_width",
        "triangle_height",
        "solid_edge_length",
    ]
    for name in positive_fields:
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(config.regular_top_edge_factor) or config.regular_top_edge_factor <= 0:
        raise ValueError("regular_top_edge_factor must be finite and positive.")

    nx = max(2, int(round(config.lx_f / config.fluid_h)))
    ny = max(2, int(round(config.ly_f / config.fluid_h)))
    dx, dy = config.lx_f / nx, config.ly_f / ny
    if config.refine_solid_base and (
        not np.isfinite(config.solid_base_layer_eps)
        or not 0 < config.solid_base_layer_eps < dy
    ):
        raise ValueError("solid_base_layer_eps must lie strictly between 0 and dy_f.")

    half_width = 0.5 * config.triangle_base_width
    vertices = np.array(
        [
            [config.triangle_center_x - half_width, 0.0],
            [config.triangle_center_x + half_width, 0.0],
            [config.triangle_center_x, config.triangle_height],
        ]
    )
    if (
        not np.all(np.isfinite(vertices))
        or np.min(vertices[:, 0]) <= 0
        or np.max(vertices[:, 0]) >= config.lx_f
        or config.triangle_height >= config.ly_f
    ):
        raise ValueError("The triangle must lie inside the channel, with only its base on y=0.")

    return nx, ny, dx, dy, vertices


def build_single_spike_mesh(config=SpikeMeshConfig()):
    """Build the single-spike reference mesh used by spike-style cases."""
    nx, ny, dx, dy, vertices = validate_single_spike_config(config)
    x, y, triangles, bdry, bottom_nodes = triangular_obstacle_mesh(
        vertices,
        config.solid_edge_length,
    )
    P = np.vstack((x, y))
    if config.refine_solid_base:
        layer_height = dy - config.solid_base_layer_eps
        first_row = np.min(y[y > 1.0e-12])
        if layer_height >= first_row:
            raise ValueError("The base layer must lie below the first existing solid node row.")
        P, triangles, bdry = insert_solid_base_layer(P, triangles, layer_height)

    fixed_nodes = dirichlet_nodes_from_bdry(bdry)
    if not np.array_equal(fixed_nodes, bottom_nodes):
        raise RuntimeError("Boundary flags do not identify the entire clamped base.")

    return SpikeMesh(P=P, T=triangles, B=bdry, fixed_nodes=fixed_nodes, nx=nx, ny=ny, dx=dx, dy=dy)


def triangle_width_at_y(config, y):
    """Return triangular spike width at height ``y`` above its base."""
    return config.triangle_base_width * max(0.0, 1.0 - y / config.triangle_height)


def stitch_rows(lower, upper):
    """Triangulate the strip between two rows of node indices."""
    lower = list(lower)
    upper = list(upper)
    if len(lower) == 1 and len(upper) == 1:
        return []

    triangles = []
    i = 0
    j = 0
    while i < len(lower) - 1 or j < len(upper) - 1:
        if j == len(upper) - 1:
            triangles.append([lower[i], lower[i + 1], upper[j]])
            i += 1
        elif i == len(lower) - 1:
            triangles.append([lower[i], upper[j + 1], upper[j]])
            j += 1
        else:
            next_lower = (i + 1) / (len(lower) - 1)
            next_upper = (j + 1) / (len(upper) - 1)
            if next_lower <= next_upper:
                triangles.append([lower[i], lower[i + 1], upper[j]])
                i += 1
            else:
                triangles.append([lower[i], upper[j + 1], upper[j]])
                j += 1
    return triangles


def build_regular_top_spike_mesh(config=SpikeMeshConfig()):
    """Build a row-based spike mesh with a thin deformed base strip."""
    nx, ny, dx, dy, _ = validate_single_spike_config(config)
    target_edge = config.regular_top_edge_factor * config.solid_edge_length
    layer_height = dy - config.solid_base_layer_eps if config.refine_solid_base else 0.0
    if layer_height >= config.triangle_height:
        raise ValueError("The base layer must lie below the spike tip.")

    top_start = layer_height if config.refine_solid_base else 0.0
    vertical_target = 0.5 * np.sqrt(3.0) * target_edge
    top_intervals = max(1, int(np.ceil((config.triangle_height - top_start) / vertical_target)))

    row_heights = [0.0]
    if config.refine_solid_base:
        row_heights.append(layer_height)
    row_heights.extend(
        top_start + (config.triangle_height - top_start) * k / top_intervals
        for k in range(1, top_intervals + 1)
    )

    points = []
    rows = []
    for y in row_heights:
        if np.isclose(y, config.triangle_height, rtol=0.0, atol=1.0e-14):
            xs = np.array([config.triangle_center_x])
        else:
            width = triangle_width_at_y(config, y)
            intervals = max(1, int(round(width / target_edge)))
            xs = np.linspace(
                config.triangle_center_x - 0.5 * width,
                config.triangle_center_x + 0.5 * width,
                intervals + 1,
            )
        row = []
        for x in xs:
            row.append(len(points))
            points.append([x, y])
        rows.append(row)

    triangles = []
    for lower, upper in zip(rows[:-1], rows[1:]):
        triangles.extend(stitch_rows(lower, upper))

    P = np.asarray(points, dtype=float).T
    triangles = orient_triangles_counterclockwise(P, np.asarray(triangles, dtype=int))
    bdry = triangular_boundary_description(triangles)
    bottom = np.all(np.isclose(P[1, bdry[2:4]], 0.0, atol=1.0e-12), axis=0)
    bdry[0, bottom] = -1

    fixed_nodes = dirichlet_nodes_from_bdry(bdry)
    bottom_nodes = np.asarray(rows[0], dtype=int)
    if not np.array_equal(fixed_nodes, bottom_nodes):
        raise RuntimeError("Boundary flags do not identify the entire clamped base.")

    return SpikeMesh(P=P, T=triangles, B=bdry, fixed_nodes=fixed_nodes, nx=nx, ny=ny, dx=dx, dy=dy)


def insert_wall_base_layer(P, triangles, wall_y, points_into_domain, layer_distance):
    """Insert one mesh row at a fixed distance from a clamped horizontal wall."""
    if layer_distance <= 0.0:
        return point_matrix(P), orient_triangles_counterclockwise(P, triangles)

    P = point_matrix(P)
    triangles = triangle_connectivity(triangles)
    wall_distance = points_into_domain * (P[1] - wall_y)
    points = [point.copy() for point in P.T]
    intersections = {}
    node_groups = {}
    edges = np.unique(
        np.sort(
            np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
    )
    for i, j in edges:
        if (wall_distance[i] < layer_distance) == (wall_distance[j] < layer_distance):
            continue
        base_node = int(i if wall_distance[i] < layer_distance else j)
        fraction = (layer_distance - wall_distance[i]) / (wall_distance[j] - wall_distance[i])
        point = P[:, i] + fraction * (P[:, j] - P[:, i])
        point[1] = wall_y + points_into_domain * layer_distance
        node_groups.setdefault(base_node, []).append(((int(i), int(j)), point))

    for group in node_groups.values():
        point = np.mean([entry[1] for entry in group], axis=0)
        point[1] = wall_y + points_into_domain * layer_distance
        for key, _ in group:
            intersections[tuple(sorted(key))] = len(points)
        points.append(point)

    def intersection(i, j):
        return intersections[tuple(sorted((int(i), int(j))))]

    refined = []
    for nodes in triangles:
        distance = wall_distance[nodes]
        below = distance < layer_distance
        if not np.any(below) or np.all(distance <= layer_distance):
            refined.append(nodes.tolist())
            continue
        if below.sum() == 1:
            a, b, c = np.roll(nodes, -int(np.flatnonzero(below)[0]))
            ab, ac = intersection(a, b), intersection(a, c)
            pieces = ([a, ab, ac], [ab, b, c], [ab, c, ac])
        else:
            a, b, c = np.roll(nodes, 2 - int(np.flatnonzero(~below)[0]))
            bc, ac = intersection(b, c), intersection(a, c)
            pieces = ([a, b, bc], [a, bc, ac], [ac, bc, c])
        refined.extend(piece for piece in pieces if len(set(piece)) == 3)

    P = np.asarray(points, dtype=float).T
    return P, orient_triangles_counterclockwise(P, np.asarray(refined, dtype=int))


def spike_width_at_distance(config, distance):
    """Return width of a straight triangular spike at distance from its base."""
    return config.triangle_base_width * max(0.0, 1.0 - distance / config.triangle_height)


def build_wall_spike_mesh(config, center_x, wall_y, points_into_domain, dy):
    """Build one regular-top triangular spike clamped to a horizontal wall."""
    half_width = 0.5 * config.triangle_base_width
    vertices = np.array(
        [
            [center_x - half_width, wall_y],
            [center_x + half_width, wall_y],
            [center_x, wall_y + points_into_domain * config.triangle_height],
        ]
    )
    if (
        not np.all(np.isfinite(vertices))
        or np.min(vertices[:, 0]) <= 0
        or np.max(vertices[:, 0]) >= config.lx_f
        or np.min(vertices[:, 1]) < 0
        or np.max(vertices[:, 1]) > config.ly_f
    ):
        raise ValueError("Each spike must lie inside the channel, with only its base on a wall.")

    target_edge = config.regular_top_edge_factor * config.solid_edge_length
    layer_distance = dy - config.solid_base_layer_eps if config.refine_solid_base else 0.0
    if layer_distance >= config.triangle_height:
        raise ValueError("The base layer must lie below the spike tip.")

    top_start = layer_distance if config.refine_solid_base else 0.0
    vertical_target = 0.5 * np.sqrt(3.0) * target_edge
    top_intervals = max(1, int(np.ceil((config.triangle_height - top_start) / vertical_target)))

    distances = [0.0]
    if config.refine_solid_base:
        distances.append(layer_distance)
    distances.extend(
        top_start + (config.triangle_height - top_start) * k / top_intervals
        for k in range(1, top_intervals + 1)
    )

    points = []
    rows = []
    for distance in distances:
        if np.isclose(distance, config.triangle_height, rtol=0.0, atol=1.0e-14):
            xs = np.array([center_x])
        else:
            width = spike_width_at_distance(config, distance)
            intervals = max(1, int(round(width / target_edge)))
            xs = np.linspace(center_x - 0.5 * width, center_x + 0.5 * width, intervals + 1)
        y = wall_y + points_into_domain * distance
        row = []
        for x in xs:
            row.append(len(points))
            points.append([x, y])
        rows.append(row)

    triangles = []
    for lower, upper in zip(rows[:-1], rows[1:]):
        triangles.extend(stitch_rows(lower, upper))

    P = np.asarray(points, dtype=float).T
    triangles = orient_triangles_counterclockwise(P, np.asarray(triangles, dtype=int))
    bdry = triangular_boundary_description(triangles)
    bdry = label_polygon_boundary_sides(P, bdry, vertices, label_offset=1000)
    bdry = mark_wall_dirichlet_edges(P, bdry, wall_y)
    return P, triangles, bdry


def triangular_circle_mesh(center=(0.0, 0.0), radius=1.0, edge_length=0.1, boundary_flag=-2):
    """Generate a boundary-fitted P1 triangular mesh for a circular domain."""
    center = np.asarray(center, dtype=float).reshape(-1)
    if center.size != 2:
        raise ValueError("center must contain exactly two coordinates.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if edge_length <= 0.0:
        raise ValueError("edge_length must be positive.")

    boundary_segments = max(8, int(np.ceil(2.0 * np.pi * radius / edge_length)))
    theta = np.linspace(0.0, 2.0 * np.pi, boundary_segments, endpoint=False)
    boundary = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    row_spacing = np.sqrt(3.0) * 0.5 * edge_length
    interior_radius = max(radius - 0.35 * edge_length, 0.0)
    interior = []
    row = 0
    y = -radius + row_spacing
    while y <= radius - row_spacing + 1.0e-12:
        x_offset = 0.5 * edge_length if row % 2 else 0.0
        x = -radius + x_offset
        while x <= radius + 1.0e-12:
            if x * x + y * y <= interior_radius**2 + 1.0e-12:
                interior.append([x, y])
            x += edge_length
        row += 1
        y += row_spacing

    if not interior:
        interior.append([0.0, 0.0])

    interior = unique_points(np.asarray(interior, dtype=float))
    points_local = np.vstack([interior, boundary])
    P = (points_local + center).T
    T = orient_triangles_counterclockwise(P, Delaunay(points_local).simplices)
    bdry = triangular_boundary_description(T, flag=boundary_flag)
    return P, T, bdry


def lame_parameters(E, nu, plane_stress=False):
    """Return Lame parameters from Young modulus and Poisson ratio."""
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must be in (-1, 0.5).")

    mu = E / (2.0 * (1.0 + nu))
    if plane_stress:
        lam = E * nu / (1.0 - nu**2)
    else:
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def default_turek_materials(flap_E=1.4e6, flap_nu=0.4, flap_rho=10000.0, circle_stiffness_factor=1.0e3):
    """Return default material map for the Turek circle-tail mesh."""
    tail = Material(E=flap_E, nu=flap_nu, rho=flap_rho)
    circle = Material(E=circle_stiffness_factor * flap_E, nu=flap_nu, rho=flap_rho)
    return {CIRCLE_REGION: circle, TAIL_REGION: tail}


def tail_left_x(config):
    """Return the x coordinate where the Turek tail starts."""
    if config.tail_x0 is not None:
        return float(config.tail_x0)

    cx, _ = config.circle_center
    half_h = 0.5 * config.tail_height
    if half_h >= config.circle_radius:
        return cx + config.circle_radius
    return cx + np.sqrt(config.circle_radius**2 - half_h**2)


def tail_bounds(config):
    """Return ``(x0, x1, y0, y1)`` bounds for the Turek tail."""
    _, cy = config.circle_center
    x0 = tail_left_x(config)
    x1 = float(config.tail_tip_x)
    y0 = cy - 0.5 * config.tail_height
    y1 = cy + 0.5 * config.tail_height
    if x1 <= x0:
        raise ValueError("tail_tip_x must be larger than tail_x0.")
    if y1 <= y0:
        raise ValueError("tail_height must be positive.")
    return x0, x1, y0, y1


def circle_points(center, radius, edge_length):
    """Return boundary and interior points for a circular point cloud."""
    boundary_segments = max(16, int(np.ceil(2.0 * np.pi * radius / edge_length)))
    theta = np.linspace(0.0, 2.0 * np.pi, boundary_segments, endpoint=False)
    boundary = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    row_spacing = np.sqrt(3.0) * 0.5 * edge_length
    interior_radius = max(radius - 0.35 * edge_length, 0.0)
    interior = [[0.0, 0.0]]
    row = 0
    y = -radius + row_spacing
    while y <= radius - row_spacing + 1.0e-12:
        x_offset = 0.5 * edge_length if row % 2 else 0.0
        x = -radius + x_offset
        while x <= radius + 1.0e-12:
            if x * x + y * y <= interior_radius**2 + 1.0e-12:
                interior.append([x, y])
            x += edge_length
        row += 1
        y += row_spacing

    points = np.vstack([interior, boundary])
    return unique_points(points + np.asarray(center, dtype=float))


def tail_points(config):
    """Return boundary and interior points for the Turek tail rectangle."""
    x0, x1, y0, y1 = tail_bounds(config)
    h = config.edge_length

    boundary = np.vstack(
        [
            segment_points((x0, y0), (x1, y0), h),
            segment_points((x1, y0), (x1, y1), h),
            segment_points((x1, y1), (x0, y1), h),
            segment_points((x0, y1), (x0, y0), h),
        ]
    )

    row_spacing = np.sqrt(3.0) * 0.5 * h
    interior = []
    row = 0
    y = y0 + row_spacing
    while y <= y1 - row_spacing + 1.0e-12:
        x_offset = 0.5 * h if row % 2 else 0.0
        x = x0 + h + x_offset
        while x <= x1 - h + 1.0e-12:
            interior.append([x, y])
            x += h
        row += 1
        y += row_spacing

    if interior:
        points = np.vstack([boundary, np.asarray(interior, dtype=float)])
    else:
        points = boundary
    return unique_points(points)


def inside_circle(x, y, center, radius, tol=0.0):
    """Return a mask for points inside a circle."""
    cx, cy = center
    r = radius + tol
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2


def inside_turek_solid(x, y, config, tol=0.0):
    """Return a mask for points inside the Turek circle-tail solid."""
    in_circle = inside_circle(x, y, config.circle_center, config.circle_radius, tol=tol)
    x0, x1, y0, y1 = tail_bounds(config)
    in_tail = (x >= x0 - tol) & (x <= x1 + tol) & (y >= y0 - tol) & (y <= y1 + tol)
    return in_circle | in_tail


def compact_mesh(P, T):
    """Drop unused nodes from a triangular mesh and remap connectivity."""
    P = point_matrix(P)
    T = triangle_connectivity(T)
    used = np.unique(T.ravel())
    remap = -np.ones(P.shape[1], dtype=int)
    remap[used] = np.arange(used.size)
    return P[:, used], remap[T]


def triangle_edge_lengths(P, T):
    """Return all edge lengths from a triangular mesh."""
    P = point_matrix(P)
    T = triangle_connectivity(T)
    pts = P.T
    p0 = pts[T[:, 0]]
    p1 = pts[T[:, 1]]
    p2 = pts[T[:, 2]]
    return np.concatenate(
        [
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p0 - p2, axis=1),
        ]
    )


def classify_turek_elements(P, T, config):
    """Return circle/tail region IDs for Turek mesh elements."""
    P = point_matrix(P)
    T = triangle_connectivity(T)
    centroids = P[:, T].mean(axis=2)
    x0, x1, y0, y1 = tail_bounds(config)
    in_tail = (
        (centroids[0] >= x0 - 1.0e-12)
        & (centroids[0] <= x1 + 1.0e-12)
        & (centroids[1] >= y0 - 1.0e-12)
        & (centroids[1] <= y1 + 1.0e-12)
    )
    region = np.full(T.shape[0], CIRCLE_REGION, dtype=int)
    region[in_tail] = TAIL_REGION
    return region


def material_arrays(region, materials, plane_stress):
    """Expand a Turek region array into elementwise ``lam``, ``mu`` and ``rho``."""
    region = np.asarray(region, dtype=int)
    lam = np.empty(region.size, dtype=float)
    mu = np.empty(region.size, dtype=float)
    rho = np.empty(region.size, dtype=float)

    for region_id in np.unique(region):
        material = materials[int(region_id)]
        lam_i, mu_i = lame_parameters(material.E, material.nu, plane_stress=plane_stress)
        mask = region == region_id
        lam[mask] = lam_i
        mu[mask] = mu_i
        rho[mask] = material.rho

    return lam, mu, rho


def turek_dirichlet_nodes(P, config):
    """Return default fixed nodes for the Turek circle-tail solid mesh."""
    P = point_matrix(P)
    cx, cy = config.circle_center
    r2 = (config.core_radius + 1.0e-12) ** 2
    dist2 = (P[0] - cx) ** 2 + (P[1] - cy) ** 2
    nodes = np.flatnonzero(dist2 <= r2)

    if config.fix_tail_left:
        x0, _, y0, y1 = tail_bounds(config)
        tol = 0.35 * config.edge_length
        tail_left_nodes = np.flatnonzero(
            (np.abs(P[0] - x0) <= tol)
            & (P[1] >= y0 - tol)
            & (P[1] <= y1 + tol)
        )
        nodes = np.concatenate([nodes, tail_left_nodes])

    if nodes.size == 0:
        nodes = np.array([int(np.argmin(dist2))], dtype=int)

    return np.unique(nodes.astype(int))


def turek_solid_mesh(config=None, materials=None):
    """Generate the Turek-Hron circle-tail triangular mesh."""
    if config is None:
        config = TurekGeometryConfig()
    if materials is None:
        materials = default_turek_materials()

    if config.edge_length <= 0.0:
        raise ValueError("edge_length must be positive.")
    if config.circle_radius <= 0.0:
        raise ValueError("circle_radius must be positive.")
    if config.core_radius < 0.0:
        raise ValueError("core_radius must be non-negative.")

    c_points = circle_points(config.circle_center, config.circle_radius, config.edge_length)
    t_points = tail_points(config)
    x0, x1, y0, y1 = tail_bounds(config)
    tail_overlap = (
        (c_points[:, 0] >= x0 - 0.5 * config.edge_length)
        & (c_points[:, 0] <= x1 + 0.5 * config.edge_length)
        & (c_points[:, 1] >= y0 - 0.5 * config.edge_length)
        & (c_points[:, 1] <= y1 + 0.5 * config.edge_length)
    )
    c_points = c_points[~tail_overlap]
    all_points = unique_points(np.vstack([c_points, t_points]))

    tri = Delaunay(all_points)
    P_all = all_points.T
    T_all = orient_triangles_counterclockwise(P_all, tri.simplices)

    pts = P_all.T
    tri_pts = pts[T_all]
    centroids = tri_pts.mean(axis=1)
    mids_01 = 0.5 * (tri_pts[:, 0, :] + tri_pts[:, 1, :])
    mids_12 = 0.5 * (tri_pts[:, 1, :] + tri_pts[:, 2, :])
    mids_20 = 0.5 * (tri_pts[:, 2, :] + tri_pts[:, 0, :])
    tol = 0.25 * config.edge_length

    keep = (
        inside_turek_solid(centroids[:, 0], centroids[:, 1], config, tol=tol)
        & inside_turek_solid(mids_01[:, 0], mids_01[:, 1], config, tol=tol)
        & inside_turek_solid(mids_12[:, 0], mids_12[:, 1], config, tol=tol)
        & inside_turek_solid(mids_20[:, 0], mids_20[:, 1], config, tol=tol)
    )
    T = T_all[keep]
    if T.size == 0:
        raise RuntimeError("Generated mesh is empty; decrease edge_length.")

    P, T = compact_mesh(P_all, T)
    T = orient_triangles_counterclockwise(P, T)
    bdry = triangular_boundary_description(T, flag=config.boundary_flag)
    region = classify_turek_elements(P, T, config)
    lam, mu, rho = material_arrays(region, materials, config.plane_stress)
    dirichlet_nodes = turek_dirichlet_nodes(P, config)

    return TurekSolidMesh(
        P=P,
        T=T,
        bdry=bdry,
        region=region,
        lam=lam,
        mu=mu,
        rho=rho,
        dirichlet_nodes=dirichlet_nodes,
        config=config,
    )


def turek_mesh_summary(mesh):
    """Return compact statistics for a Turek mesh."""
    lengths = triangle_edge_lengths(mesh.P, mesh.T)
    counts = {
        REGION_NAMES[int(region_id)]: int(np.count_nonzero(mesh.region == region_id))
        for region_id in np.unique(mesh.region)
    }
    return {
        "nodes": int(mesh.P.shape[1]),
        "elements": int(mesh.T.shape[0]),
        "boundary_edges": int(mesh.bdry.shape[1]),
        "dirichlet_nodes": int(mesh.dirichlet_nodes.size),
        "region_elements": counts,
        "edge_min": float(lengths.min()),
        "edge_mean": float(lengths.mean()),
        "edge_max": float(lengths.max()),
    }


def tail_tip_corner_nodes(P, config):
    """Return nearest nodes to the two right tail corners."""
    _, x1, y0, y1 = tail_bounds(config)
    return nearest_distinct_nodes(P, ((x1, y0), (x1, y1)), "right tail corner")


def tail_attachment_corner_nodes(P, config):
    """Return nearest nodes to the two tail-attachment corners."""
    x0, _, y0, y1 = tail_bounds(config)
    return nearest_distinct_nodes(P, ((x0, y0), (x0, y1)), "tail attachment corner")


def tail_mix_corner_nodes(P, config):
    """Return four distinct Turek tail corners used by MIX-style transfer."""
    nodes = []
    nodes.extend(tail_attachment_corner_nodes(P, config))
    nodes.extend(tail_tip_corner_nodes(P, config))
    nodes = np.asarray(nodes, dtype=np.int64)
    _, first_idx = np.unique(nodes, return_index=True)
    nodes = nodes[np.sort(first_idx)]
    if nodes.size != 4:
        raise ValueError("Could not identify four distinct MIX tail corner nodes.")
    return nodes


def rightmost_boundary_corner_nodes(pos_x, pos_y, bdry, dx_f, dy_f):
    """Return the lower and upper rightmost boundary nodes within a grid tolerance."""
    pos_x = np.asarray(pos_x, dtype=float).reshape(-1)
    pos_y = np.asarray(pos_y, dtype=float).reshape(-1)
    if pos_x.shape != pos_y.shape:
        raise ValueError("pos_x and pos_y must have the same shape.")

    bdry = np.asarray(bdry)
    if bdry.ndim != 2 or bdry.shape[0] < 4:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")

    nodes = np.unique(bdry[2:4, :]).astype(np.int64)
    if nodes.size < 2:
        return nodes

    x = pos_x[nodes]
    tol = 0.5 * max(float(dx_f), float(dy_f))
    candidates = nodes[x >= np.max(x) - tol]
    if candidates.size < 2:
        candidates = nodes[np.argsort(x)[-2:]]

    y = pos_y[candidates]
    return np.asarray(
        [candidates[int(np.argmin(y))], candidates[int(np.argmax(y))]],
        dtype=np.int64,
    )


def star_polygon(config=StarConfig()):
    """Return a complete CCW star polygon centered in the fluid domain."""
    if not isinstance(config.points, (int, np.integer)) or config.points < 4:
        raise ValueError("points must be an integer >= 4.")
    for name, value in asdict(config).items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if config.inner_radius >= config.outer_radius * np.cos(np.pi / config.points):
        raise ValueError("inner_radius must produce genuinely reentrant valleys.")
    angles = -np.pi / 2 + np.arange(2 * config.points) * np.pi / config.points
    radii = np.where(np.arange(2 * config.points) % 2, config.inner_radius, config.outer_radius)
    polygon = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    polygon[:, 1] *= config.height / np.ptp(polygon[:, 1])
    polygon += np.array([config.lx_f / 2, config.ly_f / 2])
    if (
        polygon[:, 0].min() <= 0
        or polygon[:, 0].max() >= config.lx_f
        or polygon[:, 1].min() <= 0
        or polygon[:, 1].max() >= config.ly_f
    ):
        raise ValueError("The star must fit inside the fluid domain.")
    return polygon


def star_mesh_quality(P, triangles):
    """Return area, minimum angle and maximum edge diagnostics for a star mesh."""
    P = point_matrix(P)
    triangles = triangle_connectivity(triangles)
    xyz = P.T[triangles]
    a = xyz[:, 1] - xyz[:, 0]
    b = xyz[:, 2] - xyz[:, 0]
    areas = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    lengths = np.linalg.norm(xyz - np.roll(xyz, -1, axis=1), axis=2)
    angles = []
    for i in range(3):
        u = xyz[:, (i + 1) % 3] - xyz[:, i]
        v = xyz[:, (i + 2) % 3] - xyz[:, i]
        cosine = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return areas, float(np.min(angles)), float(lengths.max())


def load_star_mesh(filename, lx_f, ly_f):
    """Load, validate and recenter an existing free-star mesh archive."""
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Star mesh not found: {filename}. Generate it first.")
    with np.load(filename, allow_pickle=False) as archive:
        config = StarConfig(**json.loads(str(archive["config_json"].item())))
        P = np.array(archive["P"], dtype=float, copy=True)
        triangles = np.array(archive["T"], dtype=int, copy=True).T
        bdry = np.array(archive["B"], dtype=int, copy=True)
        polygon = np.array(archive["polygon"], dtype=float, copy=True)
        fixed = np.array(archive["dirichlet_nodes"], dtype=int)

    expected_polygon = star_polygon(config)
    if polygon.shape != expected_polygon.shape or not np.allclose(polygon, expected_polygon, rtol=0, atol=1.0e-12):
        raise ValueError("The archive must contain a complete star generated by build_star_mesh.")
    if (
        P.ndim != 2
        or P.shape[0] != 2
        or not np.isfinite(P).all()
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not triangles.size
        or np.any(triangles < 0)
        or np.any(triangles >= P.shape[1])
        or bdry.ndim != 2
        or bdry.shape[0] != 5
    ):
        raise ValueError("Invalid star mesh coordinates, triangles or boundary.")
    if fixed.size or np.any(bdry[0] != -2):
        raise ValueError("The star must have no Dirichlet nodes.")

    expected_bdry = triangular_boundary_description(triangles)
    expected_edges = {tuple(sorted(edge)) for edge in expected_bdry[2:4].T}
    actual_edges = {tuple(sorted(edge)) for edge in bdry[2:4].T}
    if actual_edges != expected_edges or len(actual_edges) != bdry.shape[1]:
        raise ValueError("Star boundary edges do not match the triangular mesh.")

    labels = bdry[4]
    if np.any(labels < 0) or np.any(labels >= len(polygon)):
        raise ValueError("Invalid geometric side labels on the star boundary.")
    start = polygon[labels]
    end = polygon[(labels + 1) % len(polygon)]
    tangent = end - start
    for nodes in bdry[2:4]:
        offset = P.T[nodes] - start
        cross = offset[:, 0] * tangent[:, 1] - offset[:, 1] * tangent[:, 0]
        fraction = np.sum(offset * tangent, axis=1) / np.sum(tangent**2, axis=1)
        if np.any(np.abs(cross) > 1.0e-12) or np.any(fraction < -1.0e-10) or np.any(fraction > 1.0 + 1.0e-10):
            raise ValueError("Star side labels must identify geometric sides.")

    areas, _, _ = star_mesh_quality(P, triangles)
    polygon_area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    if np.any(areas <= 0) or not np.isclose(areas.sum(), polygon_area, rtol=1.0e-10):
        raise ValueError("Star elements must be positively oriented and cover the polygon.")
    P += np.array([(lx_f - config.lx_f) / 2, (ly_f - config.ly_f) / 2])[:, None]
    if np.any(P <= 0) or np.any(P[0] >= lx_f) or np.any(P[1] >= ly_f):
        raise ValueError("The centered star must fit strictly inside the fluid domain.")
    return P, triangles, bdry, fixed


def build_star_mesh(config=StarConfig()):
    """Build a free-star triangular mesh using the optional ``triangle`` package."""
    polygon = star_polygon(config)
    try:
        import triangle
    except ImportError as exc:
        raise ImportError("Install triangle to generate star meshes.") from exc

    points, sides = [], []
    for side, start in enumerate(polygon):
        end = polygon[(side + 1) % len(polygon)]
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / config.solid_h)))
        points.extend(start + np.arange(count)[:, None] / count * (end - start))
        sides.extend([side + 1] * count)
    points = np.asarray(points)
    segments = np.column_stack((np.arange(len(points)), np.roll(np.arange(len(points)), -1)))
    max_area = np.sqrt(3.0) * config.solid_h**2 / 4.0
    mesh = triangle.triangulate(
        dict(vertices=points, segments=segments, segment_markers=np.asarray(sides)[:, None]),
        f"pq28a{max_area:.14f}Q",
    )
    P = np.asarray(mesh["vertices"], dtype=float).T
    triangles = np.asarray(mesh["triangles"], dtype=int)
    bdry = triangular_boundary_description(triangles)
    local_sides = bdry[4].copy()
    labels = {
        tuple(sorted(edge)): int(marker) - 1
        for edge, marker in zip(mesh["segments"], mesh["segment_markers"].ravel())
    }
    boundary_keys = {tuple(sorted(edge)) for edge in bdry[2:4].T}
    if boundary_keys != set(labels):
        raise ValueError("The triangulation boundary differs from the prescribed polygon.")
    bdry[4] = [labels[tuple(sorted(edge))] for edge in bdry[2:4].T]
    fixed = np.unique(bdry[2:4, bdry[0] == -1])
    node_sides = {}
    for edge in bdry.T:
        for node in edge[2:4]:
            node_sides.setdefault(int(node), set()).add(int(edge[4]))
    corners = np.array(sorted(node for node, sides in node_sides.items() if len(sides) > 1))
    free_corners = np.setdiff1d(corners, fixed)

    areas, min_angle, max_edge = star_mesh_quality(P, triangles)
    polygon_area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    edges, counts = np.unique(
        np.sort(
            np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )
    if (
        np.any(areas <= 0)
        or not np.isclose(areas.sum(), polygon_area, rtol=1.0e-10)
        or P.shape[1] - len(edges) + len(triangles) != 1
        or np.any(counts > 2)
        or len(free_corners) != 2 * config.points
        or fixed.size
        or len(np.unique(P.T, axis=0)) != P.shape[1]
    ):
        raise ValueError("Mesh failed area, topology, uniqueness or corner checks.")
    stats = dict(
        nodes=P.shape[1],
        triangles=len(triangles),
        free_corners=len(free_corners),
        sharp_free_tips=config.points,
        fixed_nodes=len(fixed),
        center_x=config.lx_f / 2,
        center_y=config.ly_f / 2,
        area=float(areas.sum()),
        min_angle_degrees=min_angle,
        max_element_edge=max_edge,
        max_element_area=float(areas.max()),
    )
    data = dict(
        P=P,
        T=triangles.T,
        B=bdry,
        boundary_local_side=local_sides,
        polygon=polygon,
        dirichlet_nodes=fixed,
        corner_nodes=free_corners,
    )
    return data, stats


def aligned_index(value, spacing, name):
    """Return the grid index for a coordinate that must align with spacing."""
    index = int(round(value / spacing))
    if not np.isclose(index * spacing, value):
        raise ValueError(f"{name}={value:g} is not aligned with spacing {spacing:g}.")
    return index


def rectangle_cell_block(x0, x1, y0, y1, dx, dy, nx, ny, name="rectangle"):
    """Return Cartesian cell-index bounds for an aligned rectangle."""
    i0 = aligned_index(x0, dx, f"{name}_x0")
    i1 = aligned_index(x1, dx, f"{name}_x1")
    j0 = aligned_index(y0, dy, f"{name}_y0")
    j1 = aligned_index(y1, dy, f"{name}_y1")
    if i0 < 0 or i1 > nx or j0 < 0 or j1 > ny or i0 >= i1 or j0 >= j1:
        raise ValueError(f"{name} is outside the fluid grid or has zero area.")
    return i0, i1, j0, j1


def obstacle_mask_from_cells(nx, ny, obstacle_cells):
    """Build a pressure-cell obstacle mask from ``(i, j)`` cell indices."""
    obstacle_mask = np.zeros((ny, nx), dtype=bool)
    for i, j in obstacle_cells:
        obstacle_mask[j, i] = True
    return obstacle_mask


def obstacle_bounds_from_cells(obstacle_cells, dx, dy):
    """Return physical bounds covered by a set of Cartesian obstacle cells."""
    cells = np.asarray(obstacle_cells, dtype=np.int64)
    if cells.size == 0:
        raise ValueError("obstacle_cells must not be empty.")
    return np.asarray(
        [
            cells[:, 0].min() * dx,
            (cells[:, 0].max() + 1) * dx,
            cells[:, 1].min() * dy,
            (cells[:, 1].max() + 1) * dy,
        ],
        dtype=float,
    )


def obstacle_outline_segments(obstacle_mask, dx, dy):
    """Return line segments outlining a Cartesian obstacle mask."""
    ny, nx = obstacle_mask.shape
    segments = []
    for j in range(ny):
        for i in range(nx):
            if not obstacle_mask[j, i]:
                continue
            x0, x1 = i * dx, (i + 1) * dx
            y0, y1 = j * dy, (j + 1) * dy
            if i == 0 or not obstacle_mask[j, i - 1]:
                segments.append([(x0, y0), (x0, y1)])
            if i == nx - 1 or not obstacle_mask[j, i + 1]:
                segments.append([(x1, y0), (x1, y1)])
            if j == 0 or not obstacle_mask[j - 1, i]:
                segments.append([(x0, y0), (x1, y0)])
            if j == ny - 1 or not obstacle_mask[j + 1, i]:
                segments.append([(x0, y1), (x1, y1)])
    return segments


def build_obstacle(shape, dx, dy, nx, ny):
    """Build the rectangle or left-arrow Cartesian obstacle used in validation cases."""
    if shape == "rectangle":
        x0, x1 = 1.45, 1.85
        y0, y1 = 0.85, 1.35
        i0, i1, j0, j1 = rectangle_cell_block(x0, x1, y0, y1, dx, dy, nx, ny, shape)
        cells = [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
        metadata = {
            "obstacle_shape": shape,
            "obstacle_x0": x0,
            "obstacle_x1": x1,
            "obstacle_y0": y0,
            "obstacle_y1": y1,
            "obstacle_i0": i0,
            "obstacle_i1": i1,
            "obstacle_j0": j0,
            "obstacle_j1": j1,
        }
    elif shape == "left_arrow":
        x0 = 1.45
        center_y = 1.10
        front_square_side = 0.30
        rear_width = 0.20
        rear_height = 0.50

        front_x0 = x0
        front_x1 = x0 + front_square_side
        rear_x0 = front_x1
        rear_x1 = rear_x0 + rear_width
        front_y0 = center_y - 0.5 * front_square_side
        front_y1 = center_y + 0.5 * front_square_side
        rear_y0 = center_y - 0.5 * rear_height
        rear_y1 = center_y + 0.5 * rear_height

        fi0, fi1, fj0, fj1 = rectangle_cell_block(
            front_x0, front_x1, front_y0, front_y1, dx, dy, nx, ny, "front_square"
        )
        ri0, ri1, rj0, rj1 = rectangle_cell_block(
            rear_x0, rear_x1, rear_y0, rear_y1, dx, dy, nx, ny, "rear_rectangle"
        )
        cells = sorted(
            {(i, j) for i in range(fi0, fi1) for j in range(fj0, fj1)}
            | {(i, j) for i in range(ri0, ri1) for j in range(rj0, rj1)}
        )
        metadata = {
            "obstacle_shape": shape,
            "obstacle_x0": min(front_x0, rear_x0),
            "obstacle_x1": max(front_x1, rear_x1),
            "obstacle_y0": min(front_y0, rear_y0),
            "obstacle_y1": max(front_y1, rear_y1),
            "front_square_x0": front_x0,
            "front_square_x1": front_x1,
            "front_square_y0": front_y0,
            "front_square_y1": front_y1,
            "rear_rectangle_x0": rear_x0,
            "rear_rectangle_x1": rear_x1,
            "rear_rectangle_y0": rear_y0,
            "rear_rectangle_y1": rear_y1,
            "front_square_i0": fi0,
            "front_square_i1": fi1,
            "front_square_j0": fj0,
            "front_square_j1": fj1,
            "rear_rectangle_i0": ri0,
            "rear_rectangle_i1": ri1,
            "rear_rectangle_j0": rj0,
            "rear_rectangle_j1": rj1,
        }
    else:
        raise ValueError("obstacle_shape must be 'rectangle' or 'left_arrow'.")

    mask = obstacle_mask_from_cells(nx, ny, cells)
    bounds = obstacle_bounds_from_cells(cells, dx, dy)
    metadata.update(
        obstacle_bounds_x0=float(bounds[0]),
        obstacle_bounds_x1=float(bounds[1]),
        obstacle_bounds_y0=float(bounds[2]),
        obstacle_bounds_y1=float(bounds[3]),
        obstacle_cell_count=len(cells),
    )
    return cells, mask, bounds, metadata


def left_arrow_solid_geometry():
    """Return geometry metadata for the left-arrow validation solid."""
    x0 = 1.45
    center_y = 1.10
    front_square_side = 0.30
    rear_width = 0.20
    rear_height = 0.50

    front_x0 = x0
    front_x1 = x0 + front_square_side
    rear_x0 = front_x1
    rear_x1 = rear_x0 + rear_width
    front_y0 = center_y - 0.5 * front_square_side
    front_y1 = center_y + 0.5 * front_square_side
    rear_y0 = center_y - 0.5 * rear_height
    rear_y1 = center_y + 0.5 * rear_height
    bounds = (front_x0, rear_x1, rear_y0, rear_y1)
    polygon = (
        (front_x0, front_y0),
        (rear_x0, front_y0),
        (rear_x0, rear_y0),
        (rear_x1, rear_y0),
        (rear_x1, rear_y1),
        (rear_x0, rear_y1),
        (rear_x0, front_y1),
        (front_x0, front_y1),
    )

    return {
        "solid_shape": "left_arrow",
        "solid_x0": bounds[0],
        "solid_x1": bounds[1],
        "solid_y0": bounds[2],
        "solid_y1": bounds[3],
        "solid_lx": bounds[1] - bounds[0],
        "solid_ly": bounds[3] - bounds[2],
        "solid_center_x": 0.5 * (bounds[0] + bounds[1]),
        "solid_center_y": center_y,
        "front_square_x0": front_x0,
        "front_square_x1": front_x1,
        "front_square_y0": front_y0,
        "front_square_y1": front_y1,
        "front_square_side": front_square_side,
        "rear_rectangle_x0": rear_x0,
        "rear_rectangle_x1": rear_x1,
        "rear_rectangle_y0": rear_y0,
        "rear_rectangle_y1": rear_y1,
        "rear_rectangle_width": rear_width,
        "rear_rectangle_height": rear_height,
        "solid_area": front_square_side**2 + rear_width * rear_height,
        "solid_polygon": polygon,
    }


def fitted_left_arrow_mesh_size(h_ideal, unit_length=0.10):
    """Snap a left-arrow Q1 mesh size to the arrow unit length."""
    if h_ideal <= 0.0:
        raise ValueError("solid mesh size must be positive.")
    n_unit = max(1, int(np.ceil(unit_length / h_ideal)))
    return unit_length / n_unit, n_unit


def build_left_arrow_square_mesh(geometry, h_ideal, boundary_flag=-2):
    """Build a Q1 left-arrow solid mesh from metadata returned by ``left_arrow_solid_geometry``."""
    h, n_unit = fitted_left_arrow_mesh_size(h_ideal)
    lx = geometry["solid_lx"]
    ly = geometry["solid_ly"]
    nx = int(round(lx / h))
    ny = int(round(ly / h))

    front_i0 = int(round((geometry["front_square_x0"] - geometry["solid_x0"]) / h))
    front_i1 = int(round((geometry["front_square_x1"] - geometry["solid_x0"]) / h))
    front_j0 = int(round((geometry["front_square_y0"] - geometry["solid_y0"]) / h))
    front_j1 = int(round((geometry["front_square_y1"] - geometry["solid_y0"]) / h))
    rear_i0 = int(round((geometry["rear_rectangle_x0"] - geometry["solid_x0"]) / h))
    rear_i1 = int(round((geometry["rear_rectangle_x1"] - geometry["solid_x0"]) / h))
    rear_j0 = int(round((geometry["rear_rectangle_y0"] - geometry["solid_y0"]) / h))
    rear_j1 = int(round((geometry["rear_rectangle_y1"] - geometry["solid_y0"]) / h))

    cells = sorted(
        {(i, j) for i in range(front_i0, front_i1) for j in range(front_j0, front_j1)}
        | {(i, j) for i in range(rear_i0, rear_i1) for j in range(rear_j0, rear_j1)}
    )
    if not cells:
        raise ValueError("left_arrow mesh contains no cells.")

    node_ids = {}
    coords = []

    def node_id(i, j):
        key = (i, j)
        if key not in node_ids:
            node_ids[key] = len(coords)
            coords.append((i * h, j * h))
        return node_ids[key]

    cell_set = set(cells)
    elements = []
    cell_to_element = {}
    for elem, (i, j) in enumerate(cells):
        elements.append([node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1)])
        cell_to_element[(i, j)] = elem

    boundary_edges = []
    neighbors = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
    }
    local_edges = {
        0: (0, 1),
        1: (1, 2),
        2: (2, 3),
        3: (3, 0),
    }
    for i, j in cells:
        elem = cell_to_element[(i, j)]
        element = elements[elem]
        for side, (di, dj) in neighbors.items():
            if (i + di, j + dj) in cell_set:
                continue
            n0, n1 = local_edges[side]
            boundary_edges.append([boundary_flag, elem, element[n0], element[n1], side])

    P = np.asarray(coords, dtype=float).T
    T = np.asarray(elements, dtype=int).T
    bdry = np.asarray(boundary_edges, dtype=int).T
    metadata = {
        "requested_solid_mesh_size": h_ideal,
        "actual_solid_mesh_size": h,
        "solid_mesh_unit_length": 0.10,
        "solid_mesh_unit_subdivisions": n_unit,
        "solid_mesh_nx_bbox": nx,
        "solid_mesh_ny_bbox": ny,
        "solid_mesh_cells": len(cells),
        "solid_mesh_nodes": P.shape[1],
        "solid_mesh_boundary_edges": bdry.shape[1],
    }
    return {"P": P, "T": T, "bdry": bdry, "metadata": metadata}


def centered_mesh_node_block(P, lx_s, ly_s, dx_s, dy_s, width, height):
    """Return nodes inside a centered physical block of a structured mesh."""
    P = point_matrix(P)
    n_anchor_x = int(round(width / dx_s)) + 1
    n_anchor_y = int(round(height / dy_s)) + 1
    x0 = 0.5 * lx_s - 0.5 * (n_anchor_x - 1) * dx_s
    x1 = 0.5 * lx_s + 0.5 * (n_anchor_x - 1) * dx_s
    y0 = 0.5 * ly_s - 0.5 * (n_anchor_y - 1) * dy_s
    y1 = 0.5 * ly_s + 0.5 * (n_anchor_y - 1) * dy_s
    tol = 1.0e-10 * max(lx_s, ly_s, 1.0)

    anchor_mask = (
        (P[0, :] >= x0 - tol)
        & (P[0, :] <= x1 + tol)
        & (P[1, :] >= y0 - tol)
        & (P[1, :] <= y1 + tol)
    )
    anchor_nodes = np.flatnonzero(anchor_mask)
    if anchor_nodes.size == 0:
        raise ValueError("central anchor block contains no solid mesh nodes.")

    i0 = int(round(x0 / dx_s))
    i1 = int(round(x1 / dx_s)) + 1
    j0 = int(round(y0 / dy_s))
    j1 = int(round(y1 / dy_s)) + 1
    return anchor_nodes, anchor_mask, (i0, i1, j0, j1), n_anchor_x, n_anchor_y


def centered_node_block(nx_s, ny_s, n_anchor_x, n_anchor_y):
    """Return node indices and a mask for a centered block in a structured Q1 grid."""
    n_nodes_x = nx_s + 1
    n_nodes_y = ny_s + 1

    if not 1 <= n_anchor_x <= n_nodes_x:
        raise ValueError("n_anchor_x must be between 1 and nx_s + 1.")
    if not 1 <= n_anchor_y <= n_nodes_y:
        raise ValueError("n_anchor_y must be between 1 and ny_s + 1.")

    i0 = (n_nodes_x - n_anchor_x) // 2
    j0 = (n_nodes_y - n_anchor_y) // 2
    i1 = i0 + n_anchor_x
    j1 = j0 + n_anchor_y

    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    anchor_nodes = (ii * n_nodes_y + jj).ravel()

    anchor_mask = np.zeros(n_nodes_x * n_nodes_y, dtype=bool)
    anchor_mask[anchor_nodes] = True

    return anchor_nodes, anchor_mask, (i0, i1, j0, j1)


def centered_node_block_from_size(nx_s, ny_s, dx_s, dy_s, width, height):
    """Return a centered block chosen from physical width and height."""
    n_nodes_x = nx_s + 1
    n_nodes_y = ny_s + 1
    n_anchor_x = int(round(width / dx_s)) + 1
    n_anchor_y = int(round(height / dy_s)) + 1
    n_anchor_x = min(max(n_anchor_x, 1), n_nodes_x)
    n_anchor_y = min(max(n_anchor_y, 1), n_nodes_y)
    return (
        *centered_node_block(nx_s, ny_s, n_anchor_x, n_anchor_y),
        n_anchor_x,
        n_anchor_y,
    )


def nodes_inside_disk(P, center, radius):
    """Return mesh node indices inside a closed disk."""
    P = point_matrix(P)
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    center = np.asarray(center, dtype=float).reshape(2)
    distance = np.linalg.norm(P.T - center, axis=1)
    return np.flatnonzero(distance <= radius)


def rectangle_perimeter(bounds):
    """Return the perimeter of a rectangle ``(x0, x1, y0, y1)``."""
    x0, x1, y0, y1 = bounds
    return 2.0 * ((x1 - x0) + (y1 - y0))


def perimeter_coordinate(side, x, y, bounds):
    """Return clockwise arclength coordinate along a rectangular boundary."""
    x0, x1, y0, y1 = bounds
    width = x1 - x0
    height = y1 - y0
    if side == "bottom":
        return x - x0
    if side == "right":
        return width + y - y0
    if side == "top":
        return width + height + x1 - x
    if side == "left":
        return 2.0 * width + height + y1 - y
    raise ValueError(f"Unknown side {side!r}.")


def corner_arc_distance(s, bounds):
    """Return minimum periodic arclength distance from rectangle corners."""
    x0, x1, y0, y1 = bounds
    width = x1 - x0
    height = y1 - y0
    perimeter = rectangle_perimeter(bounds)
    corner_s = np.array([0.0, width, width + height, 2.0 * width + height])
    s = np.asarray(s, dtype=float)
    raw = np.abs(s[..., np.newaxis] - corner_s[np.newaxis, :])
    periodic = np.minimum(raw, perimeter - raw)
    return np.min(periodic, axis=-1)


def boundary_edge_records(pos_x, pos_y, bdry, bounds):
    """Return sorted midpoint, length and arclength records for rectangular boundary edges."""
    records = []
    nodes = np.asarray(bdry)[2:4, :].astype(int).T
    sides = np.asarray(bdry)[4, :].astype(int)
    pos_x = np.asarray(pos_x, dtype=float).reshape(-1)
    pos_y = np.asarray(pos_y, dtype=float).reshape(-1)

    for (node0, node1), side_code in zip(nodes, sides):
        side = SIDE_NAMES[int(side_code)]
        x0, y0 = pos_x[node0], pos_y[node0]
        x1, y1 = pos_x[node1], pos_y[node1]
        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y0 + y1)
        length = float(np.hypot(x1 - x0, y1 - y0))
        records.append(
            {
                "side": side,
                "node0": int(node0),
                "node1": int(node1),
                "x": x_mid,
                "y": y_mid,
                "length": length,
                "s": perimeter_coordinate(side, x_mid, y_mid, bounds),
            }
        )

    return sorted(records, key=lambda row: row["s"])


def perimeter_probe_points(bounds, spacing):
    """Sample midpoint probes along a rectangular perimeter."""
    x0, x1, y0, y1 = bounds
    width = x1 - x0
    height = y1 - y0
    nx = max(2, int(round(width / spacing)))
    ny = max(2, int(round(height / spacing)))
    ds_x = width / nx
    ds_y = height / ny

    rows = []
    for k in range(nx):
        x = x0 + (k + 0.5) * ds_x
        rows.append(("bottom", x, y0, 0.0, -1.0, 1.0, 0.0, (k + 0.5) * ds_x, ds_x))
    for k in range(ny):
        y = y0 + (k + 0.5) * ds_y
        rows.append(("right", x1, y, 1.0, 0.0, 0.0, 1.0, width + (k + 0.5) * ds_y, ds_y))
    for k in range(nx):
        x = x1 - (k + 0.5) * ds_x
        rows.append(("top", x, y1, 0.0, 1.0, -1.0, 0.0, width + height + (k + 0.5) * ds_x, ds_x))
    for k in range(ny):
        y = y1 - (k + 0.5) * ds_y
        rows.append(
            (
                "left",
                x0,
                y,
                -1.0,
                0.0,
                0.0,
                -1.0,
                2.0 * width + height + (k + 0.5) * ds_y,
                ds_y,
            )
        )

    dtype = [
        ("side", "U8"),
        ("x", "f8"),
        ("y", "f8"),
        ("normal_x", "f8"),
        ("normal_y", "f8"),
        ("tangent_x", "f8"),
        ("tangent_y", "f8"),
        ("s", "f8"),
        ("edge_weight", "f8"),
    ]
    return np.array(rows, dtype=dtype)


def distance_to_rectangle_boundary(x, y, bounds):
    """Return distance to the closest side of a rectangle, inside or outside."""
    x0, x1, y0, y1 = bounds
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    inside_distance = np.minimum.reduce((x - x0, x1 - x, y - y0, y1 - y))

    dx_out = np.maximum.reduce((x0 - x, np.zeros_like(x), x - x1))
    dy_out = np.maximum.reduce((y0 - y, np.zeros_like(y), y - y1))
    outside_distance = np.hypot(dx_out, dy_out)
    return np.where(inside, inside_distance, outside_distance)


def distance_to_corners(x, y, bounds):
    """Return distance to the nearest corner of a rectangle."""
    x0, x1, y0, y1 = bounds
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.min(
        np.hypot(x[..., None] - corners[:, 0], y[..., None] - corners[:, 1]),
        axis=-1,
    )


def inside_rectangle(x, y, bounds):
    """Return a mask for points inside a rectangle including its boundary."""
    x0, x1, y0, y1 = bounds
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def outside_obstacle(x_points, y_points, bounds, buffer=0.0):
    """Return a mask for points outside rectangular obstacle bounds plus buffer."""
    x0, x1, y0, y1 = bounds
    return (
        (x_points < x0 - buffer)
        | (x_points > x1 + buffer)
        | (y_points < y0 - buffer)
        | (y_points > y1 + buffer)
    )


def obstacle_mask_from_bounds(bounds, x_centers, y_centers, buffer=0.0):
    """Return a Cartesian-cell mask from rectangle bounds and center coordinates."""
    x0, x1, y0, y1 = bounds
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    return (
        (xx >= x0 - buffer)
        & (xx <= x1 + buffer)
        & (yy >= y0 - buffer)
        & (yy <= y1 + buffer)
    )


def obstacle_cell_indices(grid, bounds, tol=1.0e-10):
    """Return aligned obstacle-cell index bounds from a fluid grid dict."""
    x0, x1, y0, y1 = bounds
    dx = grid["dx_f"]
    dy = grid["dy_f"]
    nx = grid["nx_f"]
    ny = grid["ny_f"]
    candidates = (
        ("i0", x0, dx),
        ("i1", x1, dx),
        ("j0", y0, dy),
        ("j1", y1, dy),
    )

    indices = {}
    for name, value, step in candidates:
        idx = int(round(value / step))
        if abs(idx * step - value) > tol * max(1.0, abs(value)):
            return None
        indices[name] = idx

    if not (0 < indices["i0"] < indices["i1"] < nx):
        return None
    if not (0 < indices["j0"] < indices["j1"] < ny):
        return None
    return indices["i0"], indices["i1"], indices["j0"], indices["j1"]


def cell_center_coordinates(nx, ny, dx, dy):
    """Return pressure-cell center coordinate arrays."""
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    return np.meshgrid(x, y, indexing="xy")


def face_coordinates(fluid):
    """Return MAC x-face and y-face coordinate arrays from a fluid-grid dict."""
    nx = fluid["nx_f"]
    ny = fluid["ny_f"]
    dx = fluid["dx_f"]
    dy = fluid["dy_f"]

    x_u = np.arange(nx + 1) * dx
    y_u = (np.arange(ny) + 0.5) * dy
    xu, yu = np.meshgrid(x_u, y_u, indexing="xy")

    x_v = (np.arange(nx) + 0.5) * dx
    y_v = np.arange(ny + 1) * dy
    xv, yv = np.meshgrid(x_v, y_v, indexing="xy")
    return xu, yu, xv, yv


def mac_x_index(x, y, fluid):
    """Return the nearest MAC x-face index for a point."""
    ix = int(np.floor((x + 0.5 * fluid["dx_f"]) / fluid["dx_f"]))
    iy = int(np.floor(y / fluid["dy_f"]))
    ix = int(np.clip(ix, 0, fluid["nx_f"]))
    iy = int(np.clip(iy, 0, fluid["ny_f"] - 1))
    return iy, ix


def mac_y_index(x, y, fluid):
    """Return the nearest MAC y-face index for a point."""
    ix = int(np.floor(x / fluid["dx_f"]))
    iy = int(np.floor((y + 0.5 * fluid["dy_f"]) / fluid["dy_f"]))
    ix = int(np.clip(ix, 0, fluid["nx_f"] - 1))
    iy = int(np.clip(iy, 0, fluid["ny_f"]))
    return iy, ix


def solid_cell_mask(positions, triangles, xx, yy):
    """Return a mask of pressure-cell centers covered by a triangular solid mesh."""
    from matplotlib.path import Path as MplPath
    from matplotlib.tri import Triangulation

    positions = point_matrix(positions)
    triangles = triangle_connectivity(triangles)
    try:
        finder = Triangulation(*positions, triangles).get_trifinder()
        return finder(xx, yy) >= 0
    except RuntimeError:
        points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = np.zeros(points.shape[0], dtype=bool)
        for tri in triangles:
            mask |= MplPath(positions[:, tri].T).contains_points(points, radius=1.0e-12)
        return mask.reshape(xx.shape)


__all__ = [
    "CIRCLE_REGION",
    "TAIL_REGION",
    "REGION_NAMES",
    "SIDE_NAMES",
    "SIDE_NORMALS",
    "SIDE_TANGENTS",
    "Material",
    "TurekGeometryConfig",
    "TurekSolidMesh",
    "SpikeMeshConfig",
    "SpikeMesh",
    "StarConfig",
    "point_matrix",
    "triangle_connectivity",
    "quadrilateral_connectivity",
    "orient_triangles_counterclockwise",
    "triangular_boundary_description",
    "boundary_nodes",
    "dirichlet_nodes_from_bdry",
    "corner_nodes_from_bdry",
    "nearest_node",
    "nearest_distinct_nodes",
    "mark_wall_dirichlet_edges",
    "label_polygon_boundary_sides",
    "combine_solid_meshes",
    "triangle_areas",
    "solid_area",
    "boundary_perimeter",
    "triangle_quality",
    "mesh_quality_summary",
    "unique_points",
    "segment_points",
    "structured_rectangle_mesh",
    "rectangular_obstacle_mesh",
    "solid_node_coordinates",
    "triangular_obstacle_mesh",
    "insert_solid_base_layer",
    "validate_single_spike_config",
    "build_single_spike_mesh",
    "triangle_width_at_y",
    "stitch_rows",
    "build_regular_top_spike_mesh",
    "insert_wall_base_layer",
    "spike_width_at_distance",
    "build_wall_spike_mesh",
    "triangular_circle_mesh",
    "lame_parameters",
    "default_turek_materials",
    "tail_left_x",
    "tail_bounds",
    "circle_points",
    "tail_points",
    "inside_circle",
    "inside_turek_solid",
    "compact_mesh",
    "triangle_edge_lengths",
    "classify_turek_elements",
    "material_arrays",
    "turek_dirichlet_nodes",
    "turek_solid_mesh",
    "turek_mesh_summary",
    "tail_tip_corner_nodes",
    "tail_attachment_corner_nodes",
    "tail_mix_corner_nodes",
    "rightmost_boundary_corner_nodes",
    "star_polygon",
    "star_mesh_quality",
    "load_star_mesh",
    "build_star_mesh",
    "aligned_index",
    "rectangle_cell_block",
    "obstacle_mask_from_cells",
    "obstacle_bounds_from_cells",
    "obstacle_outline_segments",
    "build_obstacle",
    "left_arrow_solid_geometry",
    "fitted_left_arrow_mesh_size",
    "build_left_arrow_square_mesh",
    "centered_mesh_node_block",
    "centered_node_block",
    "centered_node_block_from_size",
    "nodes_inside_disk",
    "rectangle_perimeter",
    "perimeter_coordinate",
    "corner_arc_distance",
    "boundary_edge_records",
    "perimeter_probe_points",
    "distance_to_rectangle_boundary",
    "distance_to_corners",
    "inside_rectangle",
    "outside_obstacle",
    "obstacle_mask_from_bounds",
    "obstacle_cell_indices",
    "cell_center_coordinates",
    "face_coordinates",
    "mac_x_index",
    "mac_y_index",
    "solid_cell_mask",
]
