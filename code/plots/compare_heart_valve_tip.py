"""Compare the saved ID tip histories with the ALE curves in Black & Park, Fig. 14.

From the FSI root:
    python3 code/plots/compare_heart_valve_tip.py

The reference CSV contains published vector-curve coordinates, not the original
ALE solution data. To regenerate it, use --extract-ale with the local paper PDF
(requires Poppler's pdftocairo). Normal plotting only needs numpy/matplotlib.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from xml.etree import ElementTree as ET

import numpy as np


def project_root():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for root in (base, *base.parents):
            if (root / "code/immersed_fsi").is_dir():
                return root
    raise RuntimeError("Run from the FSI repository root.")


ROOT = project_root()
sys.path.insert(0, str(ROOT / "code"))
from immersed_fsi.io import load_results


NUMBER = r"[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?"
REFERENCE_DOI = "10.1016/j.cma.2014.10.040"
PAPER_DOI = "10.1016/j.cma.2024.117634"
CASE = "idealized_heart_valve_id_neo_hookean"


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def extract_ale(pdf_path, csv_path):
    """Extract the two black dashed vector polylines and calibrate their axes.

    This targets Fig. 14, PDF page 24, of the specified Black & Park paper.
    It reads the complete centerlines, including portions obscured by colored
    curves and spaces between dashes. It does not trace pixels or fit a model.
    """
    pdf_path, csv_path = Path(pdf_path), Path(csv_path)
    with tempfile.TemporaryDirectory(prefix="fsi-ale-svg-") as directory:
        svg_path = Path(directory) / "page24.svg"
        subprocess.run(["pdftocairo", "-f", "24", "-l", "24", "-svg",
                        str(pdf_path), str(svg_path)], check=True)
        svg = ET.parse(svg_path).getroot()

    # The relevant paths have only absolute M/L commands and their own affine
    # matrices. Reject a changed SVG structure instead of silently misreading it.
    if any("transform" in e.attrib for e in svg.iter() if e.tag.endswith("}g")):
        raise ValueError("Unexpected group transforms in the paper SVG.")
    paths = []
    for element in svg.iter("{http://www.w3.org/2000/svg}path"):
        attributes = element.attrib
        if "stroke" not in attributes:
            continue
        commands = re.findall(r"[A-DF-Za-df-z]", attributes["d"])
        if any(command not in {"M", "L"} for command in commands):
            continue
        xy = np.array(re.findall(NUMBER, attributes["d"]), dtype=float).reshape(-1, 2)
        transform = attributes.get("transform", "matrix(1 0 0 1 0 0)")
        if not transform.startswith("matrix("):
            raise ValueError("Unexpected path transform in the paper SVG.")
        a, b, c, d, e, f = map(float, re.findall(NUMBER, transform))
        xy = xy @ np.array([[a, b], [c, d]]) + [e, f]
        paths.append((attributes, xy))

    # Four horizontal spines: top/bottom of the left/right axes. The printed
    # tick labels give t = [0, 3] s, ux = [0, .6] cm and uy = [0, .5] cm.
    spines = [xy for _, xy in paths if len(xy) == 2
              and np.ptp(xy[:, 0]) > 100 and np.ptp(xy[:, 1]) < 1e-5]
    curves = [xy for a, xy in paths if len(xy) > 20
              and a.get("stroke-dasharray") and a.get("stroke") == "rgb(0%, 0%, 0%)"]
    if len(spines) != 4 or len(curves) != 2:
        raise ValueError("Expected exactly two axes and two black dashed ALE curves in Fig. 14.")
    spines.sort(key=lambda xy: xy[:, 0].mean())
    curves.sort(key=lambda xy: xy[:, 0].mean())

    rows, panels = [], {}
    for index, (component, ymax) in enumerate((("x", 0.6), ("y", 0.5))):
        borders = spines[2 * index:2 * index + 2]
        x0, x1 = sorted(borders[0][:, 0])
        top, bottom = sorted([border[0, 1] for border in borders])
        if not np.allclose(sorted(borders[1][:, 0]), [x0, x1], atol=1e-3):
            raise ValueError("The plot borders do not align.")
        xy = curves[index]
        times = 3 * (xy[:, 0] - x0) / (x1 - x0)
        values = ymax * (bottom - xy[:, 1]) / (bottom - top)
        if (np.any(np.diff(times) <= 0) or times.min() < 0 or times.max() > 3
                or values.min() < 0 or values.max() > ymax):
            raise ValueError("Extracted ALE coordinates are outside the calibrated axes.")
        rows.extend((component, f"{t:.8f}", f"{v:.8f}") for t, v in zip(times, values))
        panels[component] = {
            "point_count": len(times), "time_range_s": [float(times[0]), float(times[-1])],
            "axis_svg_points": {"left": x0, "right": x1, "top": top, "bottom": bottom},
            "axis_time_range_s": [0, 3], "axis_displacement_range_cm": [0, ymax],
            "maximum_curve_displacement_cm": float(values.max()),
        }
    write_csv(csv_path, ["component", "time_s", "displacement_cm"], rows)
    metadata = {
        "source_pdf": pdf_path.name, "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "source_doi": PAPER_DOI, "figure": 14, "pdf_page": 24,
        "reference_authors": "Kamensky et al. (2015)", "reference_doi": REFERENCE_DOI,
        "reference_model": "ALE; Saint Venant-Kirchhoff; quadratic B-splines",
        "method": "pdftocairo SVG; complete black dashed vector centerlines; linear axis calibration",
        "qualification": "Published graphical data, not original ALE solver output. No fitting, extrapolation, smoothing or phase shift.",
        "panels": panels,
    }
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_ale(csv_path):
    curves = {}
    with Path(csv_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for component in ("x", "y"):
        curve = np.array([(float(row["time_s"]), float(row["displacement_cm"]))
                          for row in rows if row["component"] == component])
        if (curve.ndim != 2 or len(curve) < 2 or curve.shape[1] != 2
                or not np.all(np.isfinite(curve)) or np.any(np.diff(curve[:, 0]) <= 0)):
            raise ValueError(f"Invalid ALE {component}-displacement reference.")
        curves[component] = curve
    return curves


def load_tip(result_path):
    saved = load_results(result_path, keys=("P", "u_s", "frame_indices"))
    arrays, params = saved.arrays, saved.params
    if params.get("solid_geometry") != "idealized_heart_valve":
        raise ValueError("Expected an idealized heart valve result with CGS parameters.")
    if params.get("coupling_method") != "ID" or params.get("solid_material_model") != "neo_hookean":
        raise ValueError("This comparison is labeled for ID + Neo-Hookean results.")
    P, u = np.asarray(arrays["P"]), np.asarray(arrays["u_s"])
    steps = np.asarray(arrays["frame_indices"], dtype=np.int64)
    times = steps * float(params["dt"])
    nodes = np.array([int(s) for s in str(params["upper_tip_nodes"]).split()])
    n = P.shape[1]
    if (len(nodes) < 2 or np.any(nodes < 0) or np.any(nodes >= n)
            or u.shape != (len(times), 2 * n) or len(times) < 2
            or not np.all(np.isfinite(u)) or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid solid history, times or upper tip nodes.")
    nodes = nodes[np.argsort(P[0, nodes])]
    xs = P[0, nodes]
    if np.any(np.diff(xs) <= 0) or np.ptp(P[1, nodes]) > 1e-10:
        raise ValueError("The reference tip must be a horizontal edge.")
    # Evaluate the Q1 free-edge displacement at its material midpoint. With
    # one element across the thickness this is exactly the two-node mean.
    midpoint = 0.5 * (xs[0] + xs[-1])
    right = min(np.searchsorted(xs, midpoint, side="right"), len(xs) - 1)
    left = right - 1
    weight = (midpoint - xs[left]) / (xs[right] - xs[left])
    selected = nodes[[left, right]]
    values = np.column_stack([(1 - weight) * u[:, selected[0] + offset]
                              + weight * u[:, selected[1] + offset] for offset in (0, n)])
    # Include the endpoint histories so that the tip definition is reviewable.
    endpoints = np.column_stack([u[:, node + offset] for offset in (0, n)
                                 for node in (nodes[0], nodes[-1])])
    detail = {"reference_position_cm": [float(midpoint), float(P[1, nodes[0]])],
              "all_upper_tip_nodes": nodes.tolist(), "interpolation_nodes": selected.tolist(),
              "interpolation_weights": [float(1 - weight), float(weight)],
              "definition": "Q1 material midpoint of the upper leaflet free edge; displacement from the undeformed reference"}
    return times, values, endpoints, detail, params


def compare(result_path, reference_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_ale(reference_path)
    times, values, endpoints, tip, params = load_tip(result_path)
    summary = {
        "result": str(Path(result_path).resolve()), "reference": Path(reference_path).name,
        "reference_csv_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        "tip": tip, "saved_frames": len(times), "time_range_s": [float(times[0]), float(times[-1])],
        "dt_s": float(params["dt"]), "fluid_mesh": [int(params["fluid_nx"]), int(params["fluid_ny"])],
        "solid_mesh_per_leaflet": [int(params["solid_elements_thickness"]), int(params["solid_elements_length"])],
        "units": "s, cm", "reference_doi": REFERENCE_DOI, "source_figure_doi": PAPER_DOI,
        "comparison": "Absolute simulation time; reference displacement; no phase alignment or amplitude rescaling",
        "metrics": {},
    }
    write_csv(output_dir / "heart_valve_tip_id.csv",
              ["time_s", "id_tip_ux_cm", "id_tip_uy_cm", "id_left_tip_ux_cm",
               "id_right_tip_ux_cm", "id_left_tip_uy_cm", "id_right_tip_uy_cm"],
              np.column_stack((times, values, endpoints)))
    interpolated = np.full(values.shape, np.nan)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.linewidth": .8,
                         "legend.fontsize": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})

    def draw(ax, index):
        component = "xy"[index]
        curve = reference[component]
        ax.plot(curve[:, 0], curve[:, 1], color="black", linestyle=(0, (5, 4)),
                linewidth=1.8, label="ALE - Kamensky et al. (from Fig. 14)")
        ax.plot(times, values[:, index], color="#0068c8", linewidth=2,
                label="ID + Neo-Hookean")
        ax.set(xlim=(0, 3), ylim=(0, (.6, .5)[index]), xlabel="Time (s)",
               ylabel=f"{component}-displacement (cm)")
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(loc="lower right", framealpha=1, edgecolor=".4", fancybox=False)

    for index, component in enumerate("xy"):
        curve = reference[component]
        valid = (times >= curve[0, 0]) & (times <= curve[-1, 0])
        if not np.any(valid):
            raise ValueError("The result and ALE reference time ranges do not overlap.")
        interpolated[valid, index] = np.interp(times[valid], curve[:, 0], curve[:, 1])
        delta = values[valid, index] - interpolated[valid, index]
        last_id = (times >= 2) & (times <= 3)
        last_ale = (curve[:, 0] >= 2) & (curve[:, 0] <= 3)
        metrics = {"comparison_samples": int(valid.sum()),
                   "rmse_cm": float(np.sqrt(np.mean(delta ** 2))),
                   "maximum_absolute_difference_cm": float(np.max(np.abs(delta)))}
        if np.any(last_id) and np.any(last_ale):
            id_max = float(values[last_id, index].max())
            ale_max = float(curve[last_ale, 1].max())
            metrics.update({"last_cycle_id_max_cm": id_max, "last_cycle_ale_max_cm": ale_max,
                            "last_cycle_peak_difference_percent": 100 * (id_max / ale_max - 1)})
        summary["metrics"][component] = metrics
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        draw(ax, index)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"heart_valve_tip_{component}_comparison.{extension}", dpi=220)
        plt.close(fig)

    # A side-by-side view also mirrors the arrangement of the source figure.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    for index, ax in enumerate(axes):
        draw(ax, index)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"heart_valve_tip_comparison.{extension}", dpi=220)
    plt.close(fig)
    write_csv(output_dir / "heart_valve_tip_comparison.csv",
              ["time_s", "id_ux_cm", "id_uy_cm", "ale_ux_interpolated_cm", "ale_uy_interpolated_cm"],
              np.column_stack((times, values, interpolated)))
    (output_dir / "heart_valve_tip_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path,
                        default=ROOT / f"code/cases/outputs/{CASE}/{CASE}.npz")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "code/cases/reference_data/idealized_heart_valve_ale_fig14.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "code/plots/figures/heart_valve_tip_comparison")
    parser.add_argument("--extract-ale", type=Path, metavar="PAPER_PDF",
                        help="Regenerate the reference CSV/JSON from Fig. 14 before plotting")
    args = parser.parse_args()
    if args.extract_ale:
        metadata = extract_ale(args.extract_ale, args.reference)
        print("Extracted ALE points:", {key: panel["point_count"] for key, panel in metadata["panels"].items()})
    summary = compare(args.result, args.reference, args.output_dir)
    print(json.dumps(summary["metrics"], indent=2))
    print("Figures and comparison CSV:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
