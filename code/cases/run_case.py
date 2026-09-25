"""Run immersed FSI cases from YAML configuration files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without dependency
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from immersed_fsi.fluid import fluid_assembly_ff, fluid_solver_ff  # noqa: E402
from immersed_fsi.geometry import (  # noqa: E402
    boundary_perimeter,
    lame_parameters,
    quadrilateral_connectivity,
    solid_area,
    structured_rectangle_mesh,
    triangle_connectivity,
    triangular_circle_mesh,
)
from immersed_fsi.io import save_results  # noqa: E402
from immersed_fsi.solvers import SolidSystem, solve_ib, solve_id  # noqa: E402
from immersed_fsi.structure import (  # noqa: E402
    assembly_solid_Q1_from_mesh,
    assembly_solid_T_from_mesh,
    vector_mass_matrix,
)


def load_config(path):
    """Load one YAML case file."""
    path = Path(path)
    with path.open() as handle:
        text = handle.read()
    if yaml is not None:
        config = yaml.safe_load(text)
    else:
        config = minimal_yaml_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return config


def minimal_yaml_load(text):
    """Parse the small YAML subset used by case files when PyYAML is absent."""
    root = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def parse_scalar(value):
    """Parse a scalar or inline list from the minimal YAML fallback."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if inner == "":
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def section(config, key):
    """Return a config section as a mapping."""
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be a mapping.")
    return value


def number_pair(value, name):
    """Return a two-entry float array."""
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return values


def grid_from_config(fluid):
    """Return fluid grid metadata from length/height plus resolution or counts."""
    lx = float(fluid.get("lx", fluid.get("length", 1.0)))
    ly = float(fluid.get("ly", fluid.get("height", 1.0)))
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("fluid length/height must be positive.")

    if "resolution" in fluid:
        resolution = float(fluid["resolution"])
        if resolution <= 0.0:
            raise ValueError("fluid.resolution must be positive.")
        nx = int(round(lx / resolution))
        ny = int(round(ly / resolution))
    else:
        nx = int(fluid.get("nx", 20))
        ny = int(fluid.get("ny", 20))
    if nx <= 0 or ny <= 0:
        raise ValueError("fluid nx/ny must be positive.")

    return {
        "lx": lx,
        "ly": ly,
        "nx": nx,
        "ny": ny,
        "dx": lx / nx,
        "dy": ly / ny,
        "rho": float(fluid.get("rho", 1.0)),
        "mu": float(fluid.get("mu", 0.01)),
        "G": float(fluid.get("G", 0.0)),
    }


def material_from_config(solid):
    """Return Lame parameters from explicit values or E/nu."""
    material = section(solid, "material")
    if "lambda" in material and "mu" in material:
        return float(material["lambda"]), float(material["mu"])

    young = float(material.get("E", 1.0e3))
    poisson = float(material.get("nu", 0.3))
    plane_stress = bool(material.get("plane_stress", False))
    return lame_parameters(young, poisson, plane_stress=plane_stress)


def density_scale(name, rho_s, rho_f):
    """Return the density multiplier used for acceleration-style body forces."""
    name = str(name).strip().lower()
    if name in {"buoyant", "difference", "rho_s-rho_f", "rho_s_minus_rho_f"}:
        return rho_s - rho_f
    if name in {"solid", "rho_s"}:
        return rho_s
    if name in {"fluid", "rho_f"}:
        return rho_f
    if name in {"unit", "none"}:
        return 1.0
    return float(name)


def constant_body_force(solid, rho_s, rho_f):
    """Return constant x/y structural body force functions."""
    body = solid.get("body_force", {})
    if body is None:
        values = np.zeros(2)
    elif isinstance(body, dict):
        if "force_density" in body:
            values = number_pair(body["force_density"], "solid.body_force.force_density")
        else:
            acceleration = number_pair(
                body.get("acceleration", [0.0, 0.0]),
                "solid.body_force.acceleration",
            )
            scale = density_scale(body.get("density_scale", "buoyant"), rho_s, rho_f)
            values = scale * acceleration
    else:
        values = number_pair(body, "solid.body_force")

    def force_x(x, y):
        return np.full_like(x, values[0], dtype=float)

    def force_y(x, y):
        return np.full_like(y, values[1], dtype=float)

    return force_x, force_y, values


def build_falling_square(config):
    """Build the centered Q1 rectangular solid used by the falling-square case."""
    fluid_grid = grid_from_config(section(config, "fluid"))
    solid = section(config, "solid")

    size = solid.get("size", None)
    lx_s = float(solid.get("lx", solid.get("width", size if size is not None else 0.2)))
    ly_s = float(solid.get("ly", solid.get("height", size if size is not None else lx_s)))
    if "resolution" in solid:
        resolution = float(solid["resolution"])
        if resolution <= 0.0:
            raise ValueError("solid.resolution must be positive.")
        nx_s = int(round(lx_s / resolution))
        ny_s = int(round(ly_s / resolution))
    else:
        nx_s = int(solid.get("nx", solid.get("elements_x", 4)))
        ny_s = int(solid.get("ny", solid.get("elements_y", nx_s)))
    if lx_s <= 0.0 or ly_s <= 0.0 or nx_s <= 0 or ny_s <= 0:
        raise ValueError("solid size and element counts must be positive.")

    center = number_pair(
        solid.get("center", [0.5 * fluid_grid["lx"], 0.5 * fluid_grid["ly"]]),
        "solid.center",
    )
    origin = center - np.array([0.5 * lx_s, 0.5 * ly_s])
    if np.any(origin < 0.0) or origin[0] + lx_s > fluid_grid["lx"] or origin[1] + ly_s > fluid_grid["ly"]:
        raise ValueError("falling square must be fully inside the fluid domain.")

    boundary_flag = int(solid.get("boundary_flag", -2))
    flags = np.asarray(solid.get("boundary_flags", [boundary_flag] * 4), dtype=int)
    if flags.shape != (4,):
        raise ValueError("solid.boundary_flags must contain four entries.")
    P, T, bdry = structured_rectangle_mesh(nx_s, ny_s, lx_s, ly_s, origin=origin, flags=flags)

    rho_s = float(solid.get("rho", solid.get("density", 1.2)))
    lam_s, mu_s = material_from_config(solid)
    f1, f2, body_force_density = constant_body_force(solid, rho_s, fluid_grid["rho"])
    A_s, b_s, M = assembly_solid_Q1_from_mesh(
        P,
        T,
        lam_s,
        mu_s,
        f1=f1,
        f2=f2,
        dirichlet_nodes=[],
    )
    system = SolidSystem(
        M_el=vector_mass_matrix(M),
        A_s=A_s,
        b_s=b_s,
        P=P,
        T=T,
        M=M,
        bdry=bdry,
    )
    metadata = {
        "solid_lx": lx_s,
        "solid_ly": ly_s,
        "solid_nx": nx_s,
        "solid_ny": ny_s,
        "solid_origin_x": float(origin[0]),
        "solid_origin_y": float(origin[1]),
        "solid_center_x": float(center[0]),
        "solid_center_y": float(center[1]),
        "solid_lambda": lam_s,
        "solid_mu": mu_s,
        "solid_rho": rho_s,
        "solid_body_force_density_x": float(body_force_density[0]),
        "solid_body_force_density_y": float(body_force_density[1]),
    }
    return fluid_grid, system, metadata


def build_falling_cylinder(config):
    """Build the triangular circular solid used by the falling-cylinder case."""
    fluid_grid = grid_from_config(section(config, "fluid"))
    solid = section(config, "solid")

    diameter = solid.get("diameter", solid.get("cylinder_diameter", None))
    radius_default = 0.5 * float(diameter) if diameter is not None else 0.125
    radius = float(solid.get("radius", solid.get("cylinder_radius", radius_default)))
    edge_length = float(
        solid.get(
            "edge_length",
            solid.get("target_edge_length", solid.get("resolution", min(fluid_grid["dx"], fluid_grid["dy"]))),
        )
    )
    if radius <= 0.0 or edge_length <= 0.0:
        raise ValueError("solid radius and edge_length must be positive.")

    center = number_pair(
        solid.get("center", [0.5 * fluid_grid["lx"], 0.5 * fluid_grid["ly"]]),
        "solid.center",
    )
    if (
        center[0] - radius < 0.0
        or center[0] + radius > fluid_grid["lx"]
        or center[1] - radius < 0.0
        or center[1] + radius > fluid_grid["ly"]
    ):
        raise ValueError("falling cylinder must be fully inside the fluid domain.")

    boundary_flag = int(solid.get("boundary_flag", -2))
    P, T, bdry = triangular_circle_mesh(
        center=center,
        radius=radius,
        edge_length=edge_length,
        boundary_flag=boundary_flag,
    )

    rho_s = float(solid.get("rho", solid.get("density", 1200.0)))
    lam_s, mu_s = material_from_config(solid)
    f1, f2, body_force_density = constant_body_force(solid, rho_s, fluid_grid["rho"])
    A_s, b_s, M = assembly_solid_T_from_mesh(
        P,
        T,
        lam_s,
        mu_s,
        f1=f1,
        f2=f2,
        dirichlet_nodes=[],
        bdry=bdry,
    )
    system = SolidSystem(
        M_el=vector_mass_matrix(M),
        A_s=A_s,
        b_s=b_s,
        P=P,
        T=T,
        M=M,
        bdry=bdry,
    )
    metadata = {
        "solid_geometry": "falling_cylinder",
        "solid_mesh_type": "triangular_circle_mesh",
        "solid_radius": radius,
        "solid_diameter": 2.0 * radius,
        "solid_edge_length": edge_length,
        "solid_center_x": float(center[0]),
        "solid_center_y": float(center[1]),
        "solid_bbox_x0": float(center[0] - radius),
        "solid_bbox_x1": float(center[0] + radius),
        "solid_bbox_y0": float(center[1] - radius),
        "solid_bbox_y1": float(center[1] + radius),
        "solid_triangular_nodes": system.Nb,
        "solid_triangular_elements": triangle_connectivity(T).shape[0],
        "solid_triangular_boundary_edges": bdry.shape[1],
        "solid_area": solid_area(P, T),
        "solid_perimeter": boundary_perimeter(P, bdry),
        "solid_lambda": lam_s,
        "solid_mu": mu_s,
        "solid_rho": rho_s,
        "solid_body_force_density_x": float(body_force_density[0]),
        "solid_body_force_density_y": float(body_force_density[1]),
    }
    return fluid_grid, system, metadata


def transfer_kwargs_from_config(coupling):
    """Return transfer kwargs, expanding compact quadrature aliases when needed."""
    transfer_kind = str(coupling.get("transfer_kind", "ID")).strip().lower().replace("-", "_")
    kwargs = dict(coupling.get("transfer_kwargs", {}) or {})
    if transfer_kind in {"mix", "mixed", "mix_noc", "mix_no_c", "mix_nocorner", "mix_nocorners"}:
        quadrature_order = kwargs.pop("quadrature_order", None)
        if quadrature_order is not None:
            kwargs.setdefault("id_quadrature_order", quadrature_order)
            kwargs.setdefault("ib_quadrature_order", quadrature_order)
    return kwargs


def solver_kwargs_from_config(config, fluid_grid, solid_system):
    """Build keyword arguments for solve_id/solve_ib."""
    case = section(config, "case")
    time = section(config, "time")
    fluid = section(config, "fluid")
    coupling = section(config, "coupling")
    output = section(config, "output")

    method = str(case.get("method", "id")).strip().lower()
    if method not in {"id", "ib"}:
        raise ValueError("case.method must be 'id' or 'ib'.")

    vel0 = fluid.get("inlet_velocity", fluid.get("vel0", 0.0))
    vel0_f = np.full(fluid_grid["ny"], float(vel0), dtype=float)

    fluid_mode = str(fluid.get("mode", "free_fall")).strip().lower()
    fluid_assembly_fn = None
    fluid_step_fn = None
    fluid_assembly_kwargs = dict(fluid.get("assembly_kwargs", {}) or {})
    fluid_step_kwargs = dict(fluid.get("step_kwargs", {}) or {})
    if fluid_mode in {"free_fall", "ff", "closed"}:
        fluid_assembly_fn = fluid_assembly_ff
        fluid_step_fn = fluid_solver_ff
        fluid_assembly_kwargs.setdefault("periodic_x", bool(fluid.get("periodic_x", False)))
        fluid_assembly_kwargs.setdefault("hesch_benchmark", bool(fluid.get("hesch_benchmark", False)))
        fluid_step_kwargs.setdefault("periodic_x", bool(fluid.get("periodic_x", False)))
        fluid_step_kwargs.setdefault(
            "zero_mean_periodic_x",
            bool(fluid.get("zero_mean_periodic_x", False)),
        )
        fluid_step_kwargs.setdefault("hesch_benchmark", bool(fluid.get("hesch_benchmark", False)))
        if "body_acceleration" in fluid:
            fluid_step_kwargs.setdefault("fluid_body_acceleration", fluid["body_acceleration"])
    elif fluid_mode not in {"channel", "cf"}:
        raise ValueError("fluid.mode must be 'free_fall' or 'channel'.")

    kwargs = {
        "nx_f": fluid_grid["nx"],
        "ny_f": fluid_grid["ny"],
        "dx_f": fluid_grid["dx"],
        "dy_f": fluid_grid["dy"],
        "Nt": int(time.get("Nt", time.get("steps", 10))),
        "vel0_f": vel0_f,
        "dt": float(time.get("dt", 1.0e-3)),
        "nu_f": fluid_grid["mu"] / fluid_grid["rho"],
        "rho_f": fluid_grid["rho"],
        "G_f": fluid_grid["G"],
        "rho_s": float(section(config, "solid").get("rho", section(config, "solid").get("density", 1.2))),
        "solid_system": solid_system,
        "element_type": coupling.get("element_type", case.get("element_type", "auto")),
        "transfer_kwargs": transfer_kwargs_from_config(coupling),
        "tol": float(coupling.get("tol", coupling.get("tolerance", 1.0e-6))),
        "convergence": coupling.get("convergence", "residual"),
        "relative_tol": bool(coupling.get("relative_tol", True)),
        "residual_reference": coupling.get("residual_reference", "old_lambda"),
        "norm_floor": float(coupling.get("norm_floor", 1.0)),
        "minit": int(coupling.get("min_iterations", 0)),
        "maxit": int(coupling.get("max_iterations", 100)),
        "use_aitken": bool(section(coupling, "aitken").get("enabled", True)),
        "aitken_omega0": float(section(coupling, "aitken").get("omega0", 0.5)),
        "aitken_omega_min": float(section(coupling, "aitken").get("omega_min", 1.0e-4)),
        "aitken_omega_max": float(section(coupling, "aitken").get("omega_max", 1.0)),
        "mass_model": coupling.get("mass_model", "buoyant"),
        "rayleigh_alpha": float(coupling.get("rayleigh_alpha", 0.0)),
        "rayleigh_beta": float(coupling.get("rayleigh_beta", 0.0)),
        "pressure_solver": fluid.get("pressure_solver", "superlu"),
        "fluid_assembly_fn": fluid_assembly_fn,
        "fluid_step_fn": fluid_step_fn,
        "fluid_assembly_kwargs": fluid_assembly_kwargs,
        "fluid_step_kwargs": fluid_step_kwargs,
        "initialize_fluid": bool(fluid.get("initialize", False)),
        "store_fluid_history": bool(output.get("store_fluid_history", True)),
        "store_fluid_pressure": bool(output.get("store_fluid_pressure", True)),
        "store_solid_history": bool(output.get("store_solid_history", True)),
        "history_stride": int(time.get("history_stride", output.get("history_stride", 1))),
        "store_iteration_history": bool(output.get("store_iteration_history", True)),
        "raise_on_nonconvergence": bool(coupling.get("raise_on_nonconvergence", True)),
        "verbose": bool(output.get("verbose", True)),
        "print_every": int(output.get("print_every", 10)),
    }
    if method == "id":
        kwargs["transfer_kind"] = coupling.get("transfer_kind", "ID")
    return method, kwargs


def save_case_result(config_path, config, result, fluid_grid, metadata):
    """Save simulation arrays and parameters according to the YAML output section."""
    case = section(config, "case")
    output = section(config, "output")
    case_name = str(case.get("name", config_path.stem))
    folder = Path(output.get("folder", ROOT / "cases" / "outputs" / case_name))
    if not folder.is_absolute():
        folder = ROOT / folder

    params = {
        "case_name": case_name,
        "config_path": str(config_path),
        "fluid_lx": fluid_grid["lx"],
        "fluid_ly": fluid_grid["ly"],
        "fluid_nx": fluid_grid["nx"],
        "fluid_ny": fluid_grid["ny"],
        "fluid_dx": fluid_grid["dx"],
        "fluid_dy": fluid_grid["dy"],
        "fluid_rho": fluid_grid["rho"],
        "fluid_mu": fluid_grid["mu"],
        **metadata,
    }
    return save_results(
        folder,
        file_base=output.get("file_base", case_name),
        result=result,
        params=params,
        rho_f=fluid_grid["rho"],
        save_mode=output.get("save_mode", "history"),
        compressed=bool(output.get("compressed", False)),
        overwrite=bool(output.get("overwrite", True)),
        unique=bool(output.get("unique", False)),
        include_iteration_history=bool(output.get("store_iteration_history", True)),
    )


def run_case(config_path, dry_run=False):
    """Build and optionally run a configured case."""
    config_path = Path(config_path)
    config = load_config(config_path)
    case = section(config, "case")
    kind = str(case.get("kind", "falling_square")).strip().lower()
    builders = {
        "falling_square": build_falling_square,
        "falling_cylinder": build_falling_cylinder,
    }
    if kind not in builders:
        raise ValueError("case.kind must be 'falling_square' or 'falling_cylinder'.")

    fluid_grid, solid_system, metadata = builders[kind](config)
    method, solver_kwargs = solver_kwargs_from_config(config, fluid_grid, solid_system)
    if dry_run:
        print(f"Case: {case.get('name', config_path.stem)}")
        print(f"Fluid grid: {fluid_grid['nx']} x {fluid_grid['ny']}")
        try:
            n_elements = triangle_connectivity(solid_system.T).shape[0]
        except ValueError:
            n_elements = quadrilateral_connectivity(solid_system.T).shape[0]
        print(f"Solid nodes/elements: {solid_system.Nb} / {n_elements}")
        print(f"Method: {method.upper()}, transfer: {solver_kwargs.get('transfer_kind', 'IB')}")
        print("Dry run complete; simulation not launched.")
        return None

    solver = solve_ib if method == "ib" else solve_id
    result = solver(**solver_kwargs)
    return save_case_result(config_path, config, result, fluid_grid, metadata)


def main(argv=None):
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML case configuration.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the configured case and print a summary without solving.",
    )
    args = parser.parse_args(argv)
    run_case(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
