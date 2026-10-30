"""Geometry and inflow for Black & Park (2025), section 4.3.1 / figure 12.

All quantities use the paper's centimetre-gram-second units. The two leaflets
are disconnected Q1 rectangles, clamped only along their wall attachments.
"""

from pathlib import Path

import numpy as np

from .geometry import structured_rectangle_mesh


def nearest_element_count(length, resolution):
    """Round to nearest, with half cases rounded up rather than to even."""
    if not np.isfinite(length) or not np.isfinite(resolution) or length <= 0 or resolution <= 0:
        raise ValueError("Lengths and shared mesh resolution must be positive and finite.")
    return max(1, int(np.floor(length / resolution + 0.5 + 1e-12)))


def idealized_heart_valve_mesh(
    resolution, *, channel_length=8.0, channel_height=1.61,
    attachment_x=2.0, leaflet_length=0.70, thickness=0.0212,
):
    """Return P, row-wise quad T, boundary data, and mesh/geometry metadata.

    The same nominal resolution sets the fluid grid counts and the solid
    counts. Exact geometry takes precedence over making all edge lengths
    identical. For h=0.02 the fluid is 400x81 and each leaflet is 1x35.
    """
    nx_f = nearest_element_count(channel_length, resolution)
    ny_f = nearest_element_count(channel_height, resolution)
    nx_s = nearest_element_count(thickness, resolution)
    ny_s = nearest_element_count(leaflet_length, resolution)
    if not (0.0 < 2.0 * leaflet_length < channel_height):
        raise ValueError("Leaflets must leave a positive central gap.")
    if not (0.0 < attachment_x < attachment_x + thickness < channel_length):
        raise ValueError("Leaflets must be fully inside the channel in x.")
    meshes = [
        structured_rectangle_mesh(
            nx_s, ny_s, thickness, leaflet_length,
            origin=(attachment_x, 0.0), flags=[-1, -2, -2, -2],
        ),
        structured_rectangle_mesh(
            nx_s, ny_s, thickness, leaflet_length,
            origin=(attachment_x, channel_height - leaflet_length),
            flags=[-2, -2, -1, -2],
        ),
    ]
    points, quads, boundaries = [], [], []
    node_offset = elem_offset = 0
    tip_nodes = []
    for k, (P, T, bdry) in enumerate(meshes):
        tip_y = leaflet_length if k == 0 else channel_height - leaflet_length
        tip_nodes.append((np.flatnonzero(np.isclose(P[1], tip_y)) + node_offset).tolist())
        points.append(P)
        quads.append(T.T + node_offset)
        boundary = bdry.copy()
        boundary[1] += elem_offset
        boundary[2:4] += node_offset
        boundaries.append(boundary)
        node_offset += P.shape[1]
        elem_offset += T.shape[1]
    metadata = {
        "unit_system": "CGS (cm, g, s, dyne)",
        "paper_doi": "10.1016/j.cma.2024.117634",
        "paper_case": "4.3.1, Fig. 12, equations 135-136",
        "mesh_resolution": resolution,
        "fluid_nx": nx_f, "fluid_ny": ny_f,
        "solid_geometry": "idealized_heart_valve",
        "solid_attachment_x": attachment_x,
        "solid_leaflet_length": leaflet_length,
        "solid_thickness": thickness,
        "solid_initial_gap": channel_height - 2.0 * leaflet_length,
        "solid_elements_thickness": nx_s,
        "solid_elements_length": ny_s,
        "solid_dx": thickness / nx_s,
        "solid_dy": leaflet_length / ny_s,
        "lower_tip_nodes": tip_nodes[0], "upper_tip_nodes": tip_nodes[1],
        "fluid_inlet": "5*(sin(2*pi*t)+1.1)*y*(H-y), t>0; 0 at t=0",
        "fluid_outlet": "zero total traction, -p*n + 2*mu*epsilon(v)*n = 0",
        "fluid_outlet_predictor": "time-advanced outlet velocity; extrapolated momentum RHS",
    }
    return np.hstack(points), np.vstack(quads), np.hstack(boundaries), metadata


def heart_valve_inlet(time, *, y, height=1.61):
    """Equation (135), with an initially resting fluid and no added ramp."""
    y = np.asarray(y, dtype=float)
    if time <= 0.0:
        return np.zeros_like(y)
    return 5.0 * (np.sin(2.0 * np.pi * time) + 1.1) * y * (height - y)


def plot_heart_valve_setup(path, fluid_grid, solid, metadata):
    """Export a scientific geometry/mesh and inflow preview without simulation."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lx, ly = fluid_grid["lx"], fluid_grid["ly"]
    x = np.linspace(0, lx, fluid_grid["nx"] + 1)
    y = np.linspace(0, ly, fluid_grid["ny"] + 1)
    fig = plt.figure(figsize=(13, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])
    ax = fig.add_subplot(gs[0, :])
    zoom = fig.add_subplot(gs[1, 0])
    pulse = fig.add_subplot(gs[1, 1])
    polygons = solid.P.T[solid.T]
    for panel in (ax, zoom):
        panel.vlines(x, 0.0, ly, colors="0.80", linewidth=0.25)
        panel.hlines(y, 0.0, lx, colors="0.80", linewidth=0.25)
        panel.add_collection(PolyCollection(polygons, facecolors="#df6956", edgecolors="#792519", linewidths=0.45))
        panel.set(xlabel="x [cm]", ylabel="y [cm]")
    ax.set(xlim=(0, lx), ylim=(0, ly), aspect="equal")
    ax.set_title(
        f"Idealized heart valve | ID + Neo-Hookean | h = {metadata['mesh_resolution']:g} cm\n"
        f"fluid {fluid_grid['nx']} x {fluid_grid['ny']} | each leaflet "
        f"{metadata['solid_elements_thickness']} x {metadata['solid_elements_length']} Q1",
        fontsize=11, pad=12,
    )
    ax.text(0.05, ly / 2, "pulsatile\ninlet", va="center", fontsize=8)
    ax.text(lx - 0.05, ly / 2, "traction = 0", ha="right", va="center", fontsize=8)
    ax.text(lx / 2, ly - 0.04, "no slip; leaflet roots clamped", ha="center", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    attachment_x = metadata["solid_attachment_x"]
    zoom.set(xlim=(attachment_x - 0.025, attachment_x + 0.055), ylim=(0, 0.12), aspect="equal")
    zoom.set_title(f"Lower attachment: exact thickness {metadata['solid_thickness']:g} cm", fontsize=10)
    times = np.linspace(0.0, 3.0, 1501)
    speeds = [heart_valve_inlet(t, y=ly / 2, height=ly) for t in times]
    pulse.plot(times, speeds, color="#285a93", linewidth=1.5)
    pulse.set(xlabel="t [s]", ylabel="inlet centreline u [cm/s]", title="Paper inflow, Eq. (135)", xlim=(0, 3))
    pulse.grid(alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
