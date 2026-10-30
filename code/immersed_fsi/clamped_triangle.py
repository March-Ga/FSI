"""Mesh and setup preview for an isosceles Neo-Hookean triangle on a wall."""

from pathlib import Path

import numpy as np

from .geometry import triangular_obstacle_mesh, triangle_connectivity, triangle_areas


def clamped_triangle_mesh(base_x=(2.5, 3.0), base_y=0.0, height=1.5, apex_x=None, min_element_area=0.0025):
    """Return a conforming P1 mesh and clamp only the bottom base edge.

    The barycentric mesh preserves the exact vertices and gives n**2 equal-area
    triangles. Choose the largest integer n with total_area/n**2 at least
    min_element_area. This is a reference-mesh constraint; deformation changes
    element areas during a simulation.
    """
    base_x = np.asarray(base_x, dtype=float)
    if base_x.shape != (2,) or not np.all(np.isfinite(base_x)) or base_x[1] <= base_x[0]:
        raise ValueError("solid.base_x must contain two increasing finite x coordinates.")
    if not np.all(np.isfinite([base_y, height, min_element_area])) or height <= 0 or min_element_area <= 0:
        raise ValueError("base_y must be finite; height and min_element_area must be positive and finite.")
    if apex_x is None:
        apex_x = float(base_x.mean())
    if not np.isfinite(apex_x) or not np.isclose(apex_x, base_x.mean(), rtol=0, atol=1e-12):
        raise ValueError("An isosceles triangle requires apex_x at the midpoint of solid.base_x.")
    vertices = np.array([[base_x[0], base_y], [base_x[1], base_y], [apex_x, base_y + height]])
    total_area = float(0.5 * np.diff(base_x)[0] * height)
    tolerance = 64.0 * np.finfo(float).eps
    if min_element_area > total_area * (1.0 + tolerance):
        raise ValueError("The requested minimum element area exceeds the entire triangle area; refine the fluid mesh.")
    n = max(1, int(np.floor(np.sqrt(total_area / min_element_area) + tolerance)))
    while n > 1 and total_area / n**2 < min_element_area * (1.0 - tolerance):
        n -= 1
    longest = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1).max()
    # The existing generator uses ceil(longest/edge_length). Round upward to
    # prevent floating-point division from accidentally requesting n+1 rows.
    edge_length = np.nextafter(longest / n, np.inf)
    x, y, T, bdry, fixed = triangular_obstacle_mesh(
        vertices, edge_length, bottom_flag=-1, boundary_flag=-2,
    )
    P = np.vstack((x, y))
    T = triangle_connectivity(T)
    areas = triangle_areas(P, T)
    if len(T) != n**2 or areas.min() < min_element_area - tolerance * total_area:
        raise ValueError("Generated solid mesh does not satisfy the requested minimum element area.")
    points = P.T[T]
    lengths = np.linalg.norm(np.roll(points, -1, axis=1) - points, axis=2)
    metadata = {
        "solid_geometry": "clamped_isosceles_triangle",
        "solid_mesh_type": "barycentric_P1",
        "solid_base_x0": float(base_x[0]), "solid_base_x1": float(base_x[1]),
        "solid_base_y": float(base_y), "solid_height": float(height),
        "solid_apex_x": float(apex_x), "solid_apex_y": float(base_y + height),
        "solid_subdivisions": n,
        "solid_requested_min_area": float(min_element_area),
        "solid_min_element_area": float(areas.min()),
        "solid_max_element_area": float(areas.max()),
        "solid_min_edge": float(lengths.min()), "solid_max_edge": float(lengths.max()),
        "solid_area": total_area,
        "solid_fixed_nodes": fixed.tolist(),
        "solid_tip_node": int(np.argmax(P[1])),
    }
    return P, T, bdry, metadata


def plot_clamped_triangle_setup(path, grid, solid, metadata):
    """Export fluid grid, exact solid mesh, clamps and boundary conditions."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [1.8, 1]}, layout="constrained")
    x = np.linspace(0, grid["lx"], grid["nx"] + 1)
    y = np.linspace(0, grid["ly"], grid["ny"] + 1)
    vertices = np.array([
        [metadata["solid_base_x0"], metadata["solid_base_y"]],
        [metadata["solid_base_x1"], metadata["solid_base_y"]],
        [metadata["solid_apex_x"], metadata["solid_apex_y"]],
    ])
    for ax in axes:
        ax.vlines(x, 0, grid["ly"], color="0.80", linewidth=0.35, zorder=1)
        ax.hlines(y, 0, grid["lx"], color="0.80", linewidth=0.35, zorder=1)
        ax.add_collection(PolyCollection(solid.P.T[solid.T], facecolors="#edc4b7", edgecolors="#9e3e2c", linewidths=0.45, zorder=2))
        outline = np.vstack((vertices, vertices[0]))
        ax.plot(outline[:, 0], outline[:, 1], color="#7a251e", linewidth=1.4, zorder=3)
        ax.plot(vertices[:2, 0], vertices[:2, 1], color="black", linewidth=4, zorder=4, clip_on=False)
        ax.set(xlabel="x", ylabel="y", aspect="equal")
    full, zoom = axes
    full.set(xlim=(0, grid["lx"]), ylim=(0, grid["ly"]),
             title=f"Fluido: {grid['nx']} x {grid['ny']} celle | area cella = {grid['dx'] * grid['dy']:.6g}")
    for wall in (0, grid["ly"]):
        full.plot([0, grid["lx"]], [wall, wall], color="black", lw=2, clip_on=False)
    for y0 in np.linspace(0.3, grid["ly"] - 0.3, 9):
        full.annotate("", xy=(0.25, y0), xytext=(0.02, y0), arrowprops={"arrowstyle": "->", "color": "#245c8e", "lw": 1.0})
    full.text(0.30, grid["ly"] - 0.22, "u = 1, v = 0", color="#245c8e", fontsize=10,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    full.text(grid["lx"] - 0.1, grid["ly"] - 0.22, "Neumann 0\ntrazione nulla", ha="right", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    full.text(grid["lx"] / 2, grid["ly"] - 0.04, "parete no-slip", ha="center", va="top", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    full.text(1.0, 0.06, "parete no-slip", ha="center", fontsize=9,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    margin = 0.15
    zoom.set(xlim=(vertices[0, 0] - margin, vertices[1, 0] + margin),
             ylim=(-0.07, metadata["solid_apex_y"] + 0.14),
             title=f"Solido P1: {solid.Nb} nodi, {len(solid.T)} triangoli")
    zoom.annotate(f"({vertices[2,0]:g}, {vertices[2,1]:g})", xy=vertices[2],
                  xytext=(vertices[2,0], vertices[2,1] + 0.07), ha="center", fontsize=9)
    zoom.text(vertices[:2, 0].mean(), -0.045, "base incastrata", ha="center", fontsize=9)
    ratio = metadata["solid_min_element_area"] / (grid["dx"] * grid["dy"])
    fig.suptitle(f"Triangolo isoscele | ID + Neo-Hookean | area triangoli / area celle fluide = {ratio:.5f}", fontsize=13)
    fig.supxlabel(
        f"Area iniziale di ogni triangolo = {metadata['solid_min_element_area']:.8f} >= "
        f"{grid['dx'] * grid['dy']:.8f} (cella fluida) | E = {metadata['solid_E']:g}, nu = {metadata['solid_nu']:g}",
        fontsize=9,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
