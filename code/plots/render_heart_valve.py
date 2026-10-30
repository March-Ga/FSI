"""Render saved heart-valve histories to an MP4 with physical time labels.

Usage from the FSI root:
    python3 code/plots/render_heart_valve.py results.npz --output movie.mp4
    python3 code/plots/render_heart_valve.py results.npz --fluid-only --fps 25
"""

from pathlib import Path
import argparse
import csv
import json
import shutil
import sys

import numpy as np


def project_code():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for root in (base, *base.parents):
            if (root / "immersed_fsi").is_dir():
                return root
            if (root / "code/immersed_fsi").is_dir():
                return root / "code"
    raise RuntimeError("Run from the FSI repository root, or install immersed-fsi.")


sys.path.insert(0, str(project_code()))
from immersed_fsi.fluid import cell_centered_velocity
from immersed_fsi.geometry import dirichlet_nodes_from_bdry, quadrilateral_connectivity
from immersed_fsi.io import load_results


def render_fluid_domain(result_path, output, fps=25, vmin=None, vmax=None, preview_time=None):
    """One fluid-domain panel with Fig. 13-like colors and no constraint markers.

    The fixed color scale represents horizontal fluid velocity in cm/s.
    Neutral leaflet silhouettes mask the fictitious fluid inside the solid.
    No solid element edges, Dirichlet squares, zoom, or diagnostic chart are
    drawn. This layout is separate from the existing diagnostic renderer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import LinearSegmentedColormap

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the MP4.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    keys = ("P", "T", "bdry", "u_f", "u_s", "frame_indices")
    loaded = load_results(result_path, keys=keys)
    arrays, params = loaded.arrays, loaded.params
    if any(key not in arrays for key in keys):
        raise ValueError("Fluid-domain rendering requires saved fluid/solid histories and frame_indices.")
    dt = float(params["dt"])
    steps = np.asarray(arrays["frame_indices"], dtype=np.int64)
    times = steps * dt
    n_frames = len(steps)
    if n_frames < 2 or np.any(np.diff(steps) <= 0):
        raise ValueError("At least two saved states with strictly increasing steps are required.")
    if any(len(arrays[key]) != n_frames for key in ("u_f", "u_s")):
        raise ValueError("Fluid and solid histories must match the saved frame indices.")
    u = arrays.pop("u_f")
    fields = 0.5 * (u[:, 1:-1, 1:-1] + u[:, 1:-1, 2:])
    del u
    if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(arrays["u_s"])):
        raise ValueError("The saved fields contain nonfinite values.")
    minimum, maximum = float(fields.min()), float(fields.max())
    vmin = float(np.floor(minimum)) if vmin is None else float(vmin)
    vmax = float(np.ceil(maximum)) if vmax is None else float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("A finite increasing color range is required.")

    # Approximate the blue/cyan/green/yellow/orange/red palette in Fig. 13.
    # Keep zero cyan and use a single color mapping for the entire movie.
    control_values = np.array([-3.0, 0.0, 2.0, 4.0, 6.0, 12.0])
    control_colors = ["#3030a8", "#00d3da", "#009800", "#e4e800", "#f36d00", "#a21d24"]
    positions = (control_values - control_values[0]) / (control_values[-1] - control_values[0])
    reference = LinearSegmentedColormap.from_list("heart_valve_paper_reference", list(zip(positions, control_colors)), N=1024)
    # Resample the reference ramp if a different numeric range is requested.
    sample_values = np.linspace(vmin, vmax, 1024)
    rgba = reference(np.clip((sample_values + 3.0) / 15.0, 0.0, 1.0))
    cmap = LinearSegmentedColormap.from_list("heart_valve_paper", rgba, N=1024)
    P = np.asarray(arrays["P"])
    T = quadrilateral_connectivity(arrays["T"])
    nb = P.shape[1]
    boundary = np.asarray(arrays["bdry"])
    free_edges = boundary[2:4, boundary[0] != -1].T
    lx, ly = float(params["fluid_lx"]), float(params["fluid_ly"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 3.8), facecolor="white")
    ax = fig.add_axes([0.04, 0.24, 0.92, 0.62])
    color_axis = fig.add_axes([0.27, 0.13, 0.46, 0.035])
    image = ax.imshow(fields[0], origin="lower", extent=[0, lx, 0, ly], cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set(xlim=(0, lx), ylim=(0, ly), aspect="equal")
    ax.set_axis_off()
    body = PolyCollection(P.T[T], facecolors="white", edgecolors="none", antialiaseds=False, zorder=3)
    outline = LineCollection(P.T[free_edges], colors="#444444", linewidths=0.45, zorder=4)
    ax.add_collection(body)
    ax.add_collection(outline)
    extend = "both" if minimum < vmin and maximum > vmax else "min" if minimum < vmin else "max" if maximum > vmax else "neither"
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal", extend=extend)
    ticks = [value for value in (-3, -2.5, 0, 2, 4, 6, 9, 12) if vmin <= value <= vmax]
    if -3 in ticks and -2.5 in ticks:
        ticks.remove(-2.5)
    colorbar.set_ticks(ticks)
    colorbar.set_label(r"$u_x$ [cm/s]", fontsize=10, labelpad=2)
    colorbar.ax.tick_params(labelsize=9)
    title = fig.suptitle("", fontsize=12, y=0.955)

    def update(i):
        positions = P + arrays["u_s"][i].reshape(2, nb)
        image.set_data(fields[i])
        body.set_verts(positions.T[T])
        outline.set_segments(positions.T[free_edges])
        title.set_text(f"Idealized heart valve | t = {times[i]:.3f} s")

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
    ])
    with writer.saving(fig, str(output), dpi=150):
        for i in range(n_frames):
            update(i)
            writer.grab_frame(facecolor="white")
            if i % 50 == 0 or i == n_frames - 1:
                print(f"Encoded fluid frame {i + 1}/{n_frames}, t={times[i]:.3f} s", flush=True)
    preview = n_frames - 1 if preview_time is None else int(np.argmin(np.abs(times - preview_time)))
    update(preview)
    fig.savefig(output.with_suffix(".png"), dpi=150, facecolor="white")
    plt.close(fig)
    summary = {
        "source": str(Path(result_path).resolve()), "video": str(output.resolve()),
        "layout": "fluid_domain_only", "field": "horizontal_velocity", "units": "cm/s",
        "frames": n_frames, "first_step": int(steps[0]), "last_step": int(steps[-1]),
        "physical_start_s": float(times[0]), "physical_end_s": float(times[-1]),
        "fps": fps, "video_duration_s": n_frames / fps,
        "slowdown": round(1.0 / (fps * float(np.median(np.diff(times)))), 6),
        "colormap": "approximate Fig. 13 rainbow", "vmin": vmin, "vmax": vmax,
        "data_min_u_x": minimum, "data_max_u_x": maximum,
        "dirichlet_markers": False, "solid_element_edges": False,
        "leaflets": "neutral white silhouettes with free-boundary outlines",
        "preview_time_s": float(times[preview]),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return output


def render(result_path, output, fps=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import PolyCollection

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the MP4.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    loaded = load_results(result_path)
    arrays, params = loaded.arrays, loaded.params
    for name in ("u_f", "v_f", "u_s", "frame_indices"):
        if name not in arrays:
            raise ValueError(f"Saved history is missing {name}; save a run with fluid and solid histories.")
    dt = float(params["dt"])
    steps = np.asarray(arrays["frame_indices"], dtype=int)
    times = steps * dt
    n_frames = len(times)
    if n_frames < 2 or any(len(arrays[name]) != n_frames for name in ("u_f", "v_f", "u_s")):
        raise ValueError("Matching fluid and solid histories with at least two saved frames are required.")
    P = np.asarray(arrays["P"])
    T = quadrilateral_connectivity(arrays["T"])
    nb = P.shape[1]
    roots = dirichlet_nodes_from_bdry(arrays["bdry"])
    lx, ly = float(params["fluid_lx"]), float(params["fluid_ly"])
    dx, dy = float(params["fluid_dx"]), float(params["fluid_dy"])
    nx, ny = int(params["fluid_nx"]), int(params["fluid_ny"])
    attachment = float(params.get("solid_attachment_x", 2.0))
    zoom_limits = (max(0, attachment - 0.25), min(lx, attachment + 0.95))
    cell_x = (np.arange(nx) + 0.5) * dx
    local_columns = (cell_x >= zoom_limits[0]) & (cell_x <= zoom_limits[1])
    fields, global_speeds, local_speeds, outlet_min, outlet_max, displacements = [], [], [], [], [], []
    for i in range(n_frames):
        uc, vc = cell_centered_velocity(arrays["u_f"][i], arrays["v_f"][i])
        speed = np.hypot(uc, vc)
        fields.append(uc)
        global_speeds.append(float(speed.max()))
        local_speeds.append(float(speed[:, local_columns].max()))
        outlet = arrays["u_f"][i][1:-1, -1]
        outlet_min.append(float(outlet.min()))
        outlet_max.append(float(outlet.max()))
        displacements.append(float(np.linalg.norm(arrays["u_s"][i].reshape(2, nb), axis=0).max()))
    fields = np.asarray(fields)
    full_limit = max(float(np.max(np.abs(fields))), 1.0)
    local_limit = max(float(np.max(np.abs(fields[:, :, local_columns]))), 1.0)
    extent = [0, lx, 0, ly]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    domain = fig.add_axes([0.055, 0.60, 0.86, 0.27])
    cax_domain = fig.add_axes([0.935, 0.61, 0.012, 0.25])
    zoom = fig.add_axes([0.05, 0.105, 0.34, 0.39])
    cax_zoom = fig.add_axes([0.405, 0.105, 0.012, 0.39])
    chart = fig.add_axes([0.54, 0.135, 0.405, 0.315])
    image_full = domain.imshow(fields[0], origin="lower", extent=extent, cmap="RdBu_r",
                               vmin=-full_limit, vmax=full_limit, interpolation="nearest")
    image_zoom = zoom.imshow(fields[0], origin="lower", extent=extent, cmap="RdBu_r",
                             vmin=-local_limit, vmax=local_limit, interpolation="nearest")
    fig.colorbar(image_full, cax=cax_domain, label="u_x [cm/s]")
    fig.colorbar(image_zoom, cax=cax_zoom, label="u_x [cm/s]")
    solids = []
    for ax in (domain, zoom):
        solid = PolyCollection(P.T[T], facecolors="#c57a32", edgecolors="#39200a", linewidths=0.35, zorder=4)
        ax.add_collection(solid)
        ax.scatter(P[0, roots], P[1, roots], marker="s", s=12, color="black", zorder=5, clip_on=False)
        ax.set(xlabel="x [cm]", ylabel="y [cm]", aspect="equal")
        solids.append(solid)
    domain.set(xlim=(0, lx), ylim=(0, ly), title="Canale completo: rosso = flusso verso destra, blu = riflusso")
    zoom.set(xlim=zoom_limits, ylim=(0, ly), title="Lembi: deformazione in scala reale")
    chart.plot(times, global_speeds, color="#a63b32", lw=1.8, label="massimo nel dominio")
    chart.plot(times, local_speeds, color="#255b8c", lw=1.8, label="massimo vicino ai lembi")
    cursor = chart.axvline(0, color="0.3", lw=1)
    point, = chart.plot([0], [global_speeds[0]], "o", ms=5, color="#a63b32")
    chart.set(xlabel="tempo fisico [s]", ylabel="max |v| [cm/s]", xlim=(times[0], times[-1]),
              ylim=(0, max(global_speeds) * 1.08), title="Crescita della velocita nel dominio")
    chart.grid(alpha=0.25)
    chart.legend(loc="upper left", fontsize=8, frameon=False)
    title = fig.suptitle("", fontsize=13, y=0.975)
    frame_dt = float(np.median(np.diff(times)))
    slowdown = 1.0 / (fps * frame_dt)
    fig.text(0.5, 0.025,
             f"ID + Neo-Hookean | h = {float(params.get('mesh_resolution', dx)):g} cm | "
             f"riproduzione {slowdown:g}x rallentata | tempo fisico indicato su ogni fotogramma",
             ha="center", fontsize=9, color="0.3")

    def update(i):
        positions = P + arrays["u_s"][i].reshape(2, nb)
        for collection in solids:
            collection.set_verts(positions.T[T])
        image_full.set_data(fields[i])
        image_zoom.set_data(fields[i])
        cursor.set_xdata([times[i], times[i]])
        point.set_data([times[i]], [global_speeds[i]])
        title.set_text(f"Idealized heart valve | t = {times[i]:.3f} s | passo {steps[i]} | "
                       f"max |v| = {global_speeds[i]:.2f} cm/s")

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart",
    ])
    with writer.saving(fig, str(output), dpi=150):
        for i in range(n_frames):
            update(i)
            writer.grab_frame(facecolor="white")
            if i % 20 == 0 or i == n_frames - 1:
                print(f"Encoded frame {i + 1}/{n_frames}, t={times[i]:.3f} s", flush=True)
    update(n_frames - 1)
    fig.savefig(output.with_suffix(".png"), dpi=150, facecolor="white")
    plt.close(fig)
    diagnostics = output.with_suffix(".csv")
    with diagnostics.open("w", newline="") as handle:
        rows = csv.writer(handle)
        rows.writerow(["step", "time_s", "max_speed_cm_s", "valve_zone_max_speed_cm_s",
                       "outlet_min_u_cm_s", "outlet_max_u_cm_s", "max_solid_displacement_cm"])
        rows.writerows(zip(steps, times, global_speeds, local_speeds, outlet_min, outlet_max, displacements))
    summary = {
        "source": str(Path(result_path).resolve()), "video": str(output.resolve()),
        "frames": n_frames, "fps": fps, "physical_duration_s": float(times[-1]),
        "video_duration_s": n_frames / fps, "slowdown": slowdown,
        "max_speed_cm_s": max(global_speeds), "final_max_speed_cm_s": global_speeds[-1],
        "final_valve_zone_max_speed_cm_s": local_speeds[-1],
        "final_outlet_min_u_cm_s": outlet_min[-1], "final_outlet_max_u_cm_s": outlet_max[-1],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--fluid-only", action="store_true", help="Render one fluid-domain panel, with paper-like colors and no Dirichlet markers.")
    parser.add_argument("--vmin", type=float, help="Fixed lower color limit for --fluid-only.")
    parser.add_argument("--vmax", type=float, help="Fixed upper color limit for --fluid-only.")
    parser.add_argument("--preview-time", type=float, help="Time to use for the PNG preview in --fluid-only mode.")
    args = parser.parse_args()
    output = args.output or args.result.with_suffix(".mp4")
    if args.fluid_only:
        render_fluid_domain(args.result, output, args.fps, args.vmin, args.vmax, args.preview_time)
    else:
        if any(value is not None for value in (args.vmin, args.vmax, args.preview_time)):
            parser.error("--vmin, --vmax and --preview-time require --fluid-only")
        render(args.result, output, args.fps)


if __name__ == "__main__":
    main()
