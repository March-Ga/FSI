"""High-level coupled FSI solvers.

This module coordinates the fluid, structure, and transfer modules without
duplicating their assembly code.  It intentionally exposes two solver families:

* ``solve_ib`` for immersed-boundary coupling, where the fluid velocity is
  imposed only on boundary nodes and the interior structural velocity is solved.
* ``solve_id`` for immersed-domain coupling, where the fluid velocity is
  transferred to all solid nodes.  The same routine can use ``ID``, ``MIX``,
  or ``MIX_noC`` transfer operators.

Both solvers support vector Aitken relaxation, Rayleigh damping, convergence on
the solid position increment, and convergence on the unrelaxed multiplier
residual ``||lambda_raw - lambda_old|| / ||lambda_old||``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import warnings

import numpy as np
from scipy.sparse.linalg import factorized

from .fluid import (
    fluid_assembly_CF,
    fluid_response_inlet_v_step,
    fluid_solver_CF,
    initial_fluid_fields,
    initialize_fluid_from_inlet,
    smooth_inlet_ramp,
)
from .geometry import (
    boundary_nodes,
    dirichlet_nodes_from_bdry,
    point_matrix,
    structured_rectangle_mesh,
)
from .structure import (
    assembly_solid_Q1_from_mesh,
    assembly_solid_T_from_mesh,
    mass_norm,
    rayleigh_damping_matrix,
    solid_assembly,
    solid_assembly_from_mesh,
    vector_mass_matrix,
    zero_body_force,
)
from .transfer import (
    assembly_T_IB_quad,
    assembly_T_IB_tri,
    assembly_T_ID_quad,
    assembly_T_ID_tri,
    assembly_T_MIX_noC_quad,
    assembly_T_MIX_noC_tri,
    assembly_T_MIX_quad,
    assembly_T_MIX_tri,
)


@dataclass(frozen=True)
class AitkenControls:
    """Controls for vector Aitken relaxation in the coupling iteration."""

    min_iterations: int = 0
    max_iterations: int = 100
    omega0: float = 0.5
    omega_min: float = 1.0e-4
    omega_max: float = 1.0
    enabled: bool = True

    def validated(self):
        """Return a validated copy with normalized numeric types."""
        min_iterations = int(self.min_iterations)
        max_iterations = int(self.max_iterations)
        omega0 = float(self.omega0)
        omega_min = float(self.omega_min)
        omega_max = float(self.omega_max)

        if min_iterations < 0:
            raise ValueError("min_iterations must be non-negative.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if min_iterations > max_iterations:
            raise ValueError("min_iterations must not exceed max_iterations.")
        if omega_min <= 0.0:
            raise ValueError("omega_min must be positive.")
        if omega_max < omega_min:
            raise ValueError("omega_max must be at least omega_min.")
        if not omega_min <= omega0 <= omega_max:
            raise ValueError("omega0 must lie within the omega bounds.")

        return AitkenControls(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            omega0=omega0,
            omega_min=omega_min,
            omega_max=omega_max,
            enabled=bool(self.enabled),
        )


@dataclass(frozen=True)
class ConvergenceControls:
    """Convergence settings for one fixed-point coupling solve."""

    tol: float = 1.0e-6
    mode: str = "residual"
    relative: bool = True
    residual_reference: str = "old_lambda"
    norm_floor: float = 1.0


@dataclass
class SolidSystem:
    """Structural data, optionally with ``internal_force(u)`` for nonlinear ID.

    The callback returns the internal force in blocked FSI DOF order; when
    omitted, the solver uses ``A_s @ u`` as before. ``A_s`` is retained as the
    linear reference stiffness. Nonlinear forces are currently supported only
    by the ID family, with zero stiffness-proportional Rayleigh damping.
    """

    M_el: object
    A_s: object
    b_s: np.ndarray
    P: np.ndarray
    T: np.ndarray
    M: object
    bdry: np.ndarray
    internal_force: Callable[[np.ndarray], np.ndarray] | None = None

    @property
    def Nb(self):
        """Return the number of scalar solid nodes."""
        return self.P.shape[1]


@dataclass
class FSIResult:
    """Solution histories and diagnostics produced by ``solve_ib``/``solve_id``."""

    usol_f: list[np.ndarray]
    vsol_f: list[np.ndarray]
    psol_f: list[np.ndarray]
    usol_s: list[np.ndarray]
    vsol_s: list[np.ndarray]
    asol_s: list[np.ndarray]
    P: np.ndarray
    T: np.ndarray
    bdry: np.ndarray
    total_iterations: int
    iteration_history: list[dict] = field(default_factory=list)
    final_lambda: np.ndarray | None = None
    method: str = ""
    transfer_kind: str = ""
    element_type: str = ""

    def as_legacy_tuple(self):
        """Return the historical tuple shape used by the old case scripts."""
        return (
            self.usol_f,
            self.vsol_f,
            self.psol_f,
            self.usol_s,
            self.vsol_s,
            self.asol_s,
            self.P,
            self.T,
            self.bdry,
            self.total_iterations,
        )


class _AitkenRelaxation:
    """Small stateful vector Aitken updater."""

    def __init__(self, controls):
        self.controls = controls.validated()
        self.omega = self.controls.omega0 if self.controls.enabled else 1.0
        self.previous_residual = None

    def update(self, lam_raw, lam_old):
        """Return relaxed lambda, unrelaxed residual, and current omega."""
        residual = lam_raw - lam_old
        if self.controls.enabled and self.previous_residual is not None:
            delta = residual - self.previous_residual
            denom = float(np.dot(delta, delta))
            if denom > 0.0:
                omega = -self.omega * float(
                    np.dot(self.previous_residual, delta) / denom
                )
                if np.isfinite(omega):
                    self.omega = float(
                        np.clip(
                            omega,
                            self.controls.omega_min,
                            self.controls.omega_max,
                        )
                    )

        lam_new = lam_old + self.omega * residual
        self.previous_residual = residual.copy()
        return lam_new, residual, self.omega


def _as_velocity_profile(values, ny_f):
    """Return a one-dimensional inlet profile with one value per fluid row."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(int(ny_f), float(values), dtype=float)
    values = values.reshape(-1)
    if values.size != int(ny_f):
        raise ValueError("vel0_f must be a scalar or have one value per fluid row.")
    return values


def _as_nodes(nodes, name):
    """Return sorted unique node indices."""
    if nodes is None:
        return np.array([], dtype=np.int64)
    nodes = np.asarray(nodes, dtype=int).reshape(-1)
    if nodes.size == 0:
        return np.array([], dtype=np.int64)
    if np.any(nodes < 0):
        raise ValueError(f"{name} contains negative node indices.")
    return np.unique(nodes).astype(np.int64, copy=False)


def _as_offset(values, size, name):
    """Return a scalar or array-like initial offset as a nodal vector."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(size, float(values), dtype=float)
    values = values.reshape(-1)
    if values.size != size:
        raise ValueError(f"{name} must be a scalar or have length {size}.")
    return values


def _as_state_vector(values, size, name):
    """Validate and copy a structural state vector."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != size:
        raise ValueError(f"{name} must have length {size}.")
    return values.copy()


def _validate_grid_and_time(nx_f, ny_f, dx_f, dy_f, Nt, dt, nu_f, rho_f):
    """Validate fluid grid and time-integration scalars."""
    nx_f = int(nx_f)
    ny_f = int(ny_f)
    Nt = int(Nt)
    if nx_f <= 0 or ny_f <= 0:
        raise ValueError("nx_f and ny_f must be positive.")
    if dx_f <= 0.0 or dy_f <= 0.0:
        raise ValueError("dx_f and dy_f must be positive.")
    if Nt < 0:
        raise ValueError("Nt must be non-negative.")
    if dt is None or dt <= 0.0:
        raise ValueError("dt must be positive.")
    if nu_f is None or nu_f < 0.0:
        raise ValueError("nu_f must be non-negative.")
    if rho_f <= 0.0:
        raise ValueError("rho_f must be positive.")
    return nx_f, ny_f, Nt


def _validate_history_stride(history_stride):
    """Return a valid history stride."""
    history_stride = int(history_stride)
    if history_stride < 1:
        raise ValueError("history_stride must be at least 1.")
    return history_stride


def _store_history_frame(step, nt, history_stride):
    """Return true when a time step should be stored."""
    return step % history_stride == 0 or step == nt


def _coerce_solid_mesh(solid_mesh):
    """Return ``P, T, bdry`` from common mesh tuple/dict layouts."""
    if isinstance(solid_mesh, dict):
        if "P" in solid_mesh:
            P = solid_mesh["P"]
        elif "Pos_x" in solid_mesh and "Pos_y" in solid_mesh:
            P = np.vstack((solid_mesh["Pos_x"], solid_mesh["Pos_y"]))
        elif "pos_x" in solid_mesh and "pos_y" in solid_mesh:
            P = np.vstack((solid_mesh["pos_x"], solid_mesh["pos_y"]))
        else:
            raise ValueError("solid_mesh dict must contain P or Pos_x/Pos_y.")
        return point_matrix(P), np.asarray(solid_mesh["T"]), np.asarray(solid_mesh["bdry"])

    if not isinstance(solid_mesh, (tuple, list)):
        raise ValueError("solid_mesh must be a dict, tuple, or list.")
    if len(solid_mesh) == 3:
        P, T, bdry = solid_mesh
        return point_matrix(P), np.asarray(T), np.asarray(bdry)
    if len(solid_mesh) >= 4:
        Pos_x, Pos_y, T, bdry = solid_mesh[:4]
        return point_matrix(np.vstack((Pos_x, Pos_y))), np.asarray(T), np.asarray(bdry)
    raise ValueError("solid_mesh must be (P, T, bdry) or (Pos_x, Pos_y, T, bdry).")


def _coerce_solid_system(solid_system):
    """Return a ``SolidSystem`` from a tuple, dict, or ``SolidSystem`` object."""
    if isinstance(solid_system, SolidSystem):
        system = solid_system
    elif isinstance(solid_system, dict):
        system = SolidSystem(
            M_el=solid_system["M_el"],
            A_s=solid_system["A_s"],
            b_s=solid_system.get("b_s", solid_system.get("b", None)),
            P=solid_system["P"],
            T=solid_system["T"],
            M=solid_system["M"],
            bdry=solid_system["bdry"],
            internal_force=solid_system.get("internal_force"),
        )
    elif isinstance(solid_system, (tuple, list)) and len(solid_system) == 7:
        system = SolidSystem(*solid_system)
    else:
        raise ValueError(
            "solid_system must be SolidSystem, a dict, or "
            "(M_el, A_s, b_s, P, T, M, bdry)."
        )

    P = point_matrix(system.P, copy=True)
    Nb = P.shape[1]
    b_s = np.zeros(2 * Nb) if system.b_s is None else np.asarray(system.b_s, dtype=float).reshape(-1)
    if b_s.size != 2 * Nb:
        raise ValueError("b_s must have length 2*Nb.")
    bdry = np.asarray(system.bdry, dtype=int)
    if bdry.ndim != 2 or bdry.shape[0] < 5:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")

    return SolidSystem(
        M_el=system.M_el,
        A_s=system.A_s,
        b_s=b_s,
        P=P,
        T=np.asarray(system.T, dtype=int),
        M=system.M,
        bdry=bdry,
        internal_force=system.internal_force,
    )


def _infer_element_type(T):
    """Infer triangular or quadrilateral elements from a connectivity array."""
    T = np.asarray(T)
    if T.ndim != 2:
        raise ValueError("T must be a 2D connectivity array.")
    if T.shape[0] == 4 and T.shape[1] != 4:
        return "quad"
    if T.shape[0] == 3 and T.shape[1] != 3:
        return "tri"
    if T.shape[1] == 4:
        return "quad"
    if T.shape[1] == 3:
        return "tri"
    raise ValueError("Could not infer element type from T; pass element_type.")


def _normalize_element_type(element_type, T):
    """Return ``quad`` or ``tri``."""
    if element_type is None or str(element_type).lower() == "auto":
        return _infer_element_type(T)
    element_type = str(element_type).strip().lower()
    aliases = {
        "q1": "quad",
        "quad": "quad",
        "quadrilateral": "quad",
        "quadrilaterals": "quad",
        "square": "quad",
        "p1": "tri",
        "tri": "tri",
        "triangle": "tri",
        "triangular": "tri",
        "triangles": "tri",
    }
    try:
        return aliases[element_type]
    except KeyError as exc:
        raise ValueError("element_type must be 'quad', 'tri', or 'auto'.") from exc


def _resolve_solid_system(
    *,
    method,
    element_type,
    solid_system,
    solid_mesh,
    solid_assembly_fn,
    solid_assembly_kwargs,
    lx_s,
    ly_s,
    nx_s,
    ny_s,
    dx_s,
    dy_s,
    lam_s,
    mu_s,
    solid_flags,
    f1,
    f2,
):
    """Build or normalize the structural system used by the FSI solve."""
    if solid_system is not None:
        system = _coerce_solid_system(solid_system)
        return system, _normalize_element_type(element_type, system.T)

    if solid_assembly_fn is not None:
        kwargs = dict(solid_assembly_kwargs or {})
        system = _coerce_solid_system(solid_assembly_fn(**kwargs))
        return system, _normalize_element_type(element_type, system.T)

    if lam_s is None or mu_s is None:
        raise ValueError("lam_s and mu_s are required unless solid_system is provided.")

    if solid_mesh is not None:
        P, T, bdry = _coerce_solid_mesh(solid_mesh)
        element_type = _normalize_element_type(element_type, T)
        dirichlet = dirichlet_nodes_from_bdry(bdry)
        if element_type == "quad":
            if dirichlet.size or f1 is not None or f2 is not None:
                A_s, b_s, M = assembly_solid_Q1_from_mesh(
                    P,
                    T,
                    lam_s,
                    mu_s,
                    f1=f1,
                    f2=f2,
                    dirichlet_nodes=dirichlet,
                )
                system = SolidSystem(
                    vector_mass_matrix(M), A_s, b_s, P, T, M, bdry
                )
            else:
                system = _coerce_solid_system(
                    solid_assembly_from_mesh(P, T, bdry, lam_s, mu_s)
                )
        else:
            A_s, b_s, M = assembly_solid_T_from_mesh(
                P,
                T,
                lam_s,
                mu_s,
                f1=f1,
                f2=f2,
                dirichlet_nodes=dirichlet,
                bdry=bdry,
            )
            system = SolidSystem(vector_mass_matrix(M), A_s, b_s, P, T, M, bdry)
        return _coerce_solid_system(system), element_type

    element_type = "quad" if element_type is None else _normalize_element_type(element_type, np.empty((4, 0)))
    if element_type != "quad":
        raise ValueError("For triangular solids, pass solid_mesh or solid_system.")

    missing = [
        name
        for name, value in {
            "lx_s": lx_s,
            "ly_s": ly_s,
            "nx_s": nx_s,
            "ny_s": ny_s,
            "dx_s": dx_s,
            "dy_s": dy_s,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing structured solid parameters: " + ", ".join(missing)
        )

    if solid_flags is None:
        solid_flags = np.array([-2, -2, -2, -2], dtype=int)
    else:
        solid_flags = np.asarray(solid_flags, dtype=int).reshape(4)

    if f1 is None and f2 is None:
        system = solid_assembly(
            lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, lam_s, mu_s, solid_flags
        )
    else:
        P, T, bdry = structured_rectangle_mesh(
            nx_s, ny_s, lx_s, ly_s, flags=solid_flags
        )
        A_s, b_s, M = assembly_solid_Q1_from_mesh(
            P,
            T,
            lam_s,
            mu_s,
            f1=zero_body_force if f1 is None else f1,
            f2=zero_body_force if f2 is None else f2,
            dirichlet_nodes=dirichlet_nodes_from_bdry(bdry),
        )
        system = SolidSystem(vector_mass_matrix(M), A_s, b_s, P, T, M, bdry)

    return _coerce_solid_system(system), element_type


def _effective_mass_matrix(M_el, rho_s, rho_f, mass_model, mass_scale):
    """Return the density-scaled vector mass matrix."""
    if mass_scale is not None:
        scale = float(mass_scale)
    else:
        model = str(mass_model).strip().lower()
        if model in {"buoyant", "difference", "rho_s-rho_f", "rho_s_minus_rho_f"}:
            scale = float(rho_s) - float(rho_f)
        elif model in {"solid", "rho_s"}:
            scale = float(rho_s)
        elif model in {"unit", "none", "matrix"}:
            scale = 1.0
        else:
            raise ValueError(
                "mass_model must be 'buoyant', 'solid', 'unit', or pass mass_scale."
            )
    return scale * M_el


def _zero_nodes(values, nodes, Nb):
    """Set both vector components to zero on selected scalar nodes."""
    if nodes.size == 0:
        return values
    values[nodes] = 0.0
    values[nodes + Nb] = 0.0
    return values


def _solid_position_error(M, u_s, u_s_prev, relative, norm_floor):
    """Return the convergence error based on solid position increments."""
    numerator = mass_norm(M, np.asarray(u_s) - np.asarray(u_s_prev))
    if not relative:
        return numerator
    return numerator / max(mass_norm(M, u_s), float(norm_floor))


def _lambda_residual_error(
    residual,
    lam_old,
    lam_new,
    lam_raw,
    *,
    relative,
    reference,
    norm_floor,
):
    """Return the convergence error based on the unrelaxed lambda residual."""
    numerator = float(np.linalg.norm(residual, ord=2))
    if not relative:
        return numerator

    reference = str(reference).strip().lower()
    if reference in {"old_lambda", "old", "previous", "lambda_old"}:
        denominator = np.linalg.norm(lam_old, ord=2)
    elif reference in {"new_lambda", "relaxed", "lambda", "lambda_new"}:
        denominator = np.linalg.norm(lam_new, ord=2)
    elif reference in {"raw_lambda", "raw", "unrelaxed", "lambda_raw"}:
        denominator = np.linalg.norm(lam_raw, ord=2)
    elif reference in {"max", "largest"}:
        denominator = max(
            np.linalg.norm(lam_old, ord=2),
            np.linalg.norm(lam_new, ord=2),
            np.linalg.norm(lam_raw, ord=2),
        )
    else:
        raise ValueError(
            "residual_reference must be 'old_lambda', 'new_lambda', "
            "'raw_lambda', or 'max'."
        )
    return numerator / max(float(denominator), float(norm_floor))


def _select_convergence_error(mode, residual_error, position_error):
    """Combine residual and position errors according to the requested mode."""
    mode = str(mode).strip().lower()
    if mode in {"residual", "lambda", "force"}:
        return residual_error
    if mode in {"position", "solid_position", "solid"}:
        return position_error
    if mode in {"both", "max"}:
        return max(residual_error, position_error)
    if mode in {"either", "any", "min"}:
        return min(residual_error, position_error)
    raise ValueError(
        "convergence mode must be 'residual', 'position', 'both', or 'either'."
    )


def _normalize_transfer_kind(transfer_kind):
    """Return a normalized ID-family transfer kind."""
    transfer_kind = str(transfer_kind).strip().lower()
    aliases = {
        "id": "ID",
        "immersed_domain": "ID",
        "mix": "MIX",
        "mixed": "MIX",
        "mix_noc": "MIX_noC",
        "mix_no_c": "MIX_noC",
        "mix-noc": "MIX_noC",
        "mix-nocorners": "MIX_noC",
        "mix_nocorners": "MIX_noC",
        "mix_nocorner": "MIX_noC",
        "mix_nocorners": "MIX_noC",
        "mix_noc": "MIX_noC",
        "mix_noC".lower(): "MIX_noC",
    }
    try:
        return aliases[transfer_kind]
    except KeyError as exc:
        raise ValueError("transfer_kind must be 'ID', 'MIX', or 'MIX_noC'.") from exc


def _transfer_function(method, transfer_kind, element_type):
    """Return the transfer assembly function for this method."""
    if method == "IB":
        return assembly_T_IB_quad if element_type == "quad" else assembly_T_IB_tri

    transfer_kind = _normalize_transfer_kind(transfer_kind)
    if transfer_kind == "ID":
        return assembly_T_ID_quad if element_type == "quad" else assembly_T_ID_tri
    if transfer_kind == "MIX_noC":
        return (
            assembly_T_MIX_noC_quad
            if element_type == "quad"
            else assembly_T_MIX_noC_tri
        )
    if transfer_kind == "MIX":
        return assembly_T_MIX_quad if element_type == "quad" else assembly_T_MIX_tri
    raise AssertionError("unreachable transfer kind")


def _assemble_transfer(
    *,
    method,
    transfer_kind,
    element_type,
    x_s,
    y_s,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    bdry,
    return_d_inv,
    transfer_kwargs,
):
    """Assemble a transfer pair and optional lumping data."""
    fn = _transfer_function(method, transfer_kind, element_type)
    nx_s = 0 if nx_s is None else nx_s
    ny_s = 0 if ny_s is None else ny_s
    kwargs = dict(transfer_kwargs or {})

    if method == "IB":
        return fn(
            x_s,
            y_s,
            dx_f,
            dy_f,
            nx_s,
            ny_s,
            nx_f,
            ny_f,
            bdry,
            return_d_inv=return_d_inv,
            **kwargs,
        )

    kind = _normalize_transfer_kind(transfer_kind)
    if kind == "ID":
        return fn(
            x_s,
            y_s,
            dx_f,
            dy_f,
            nx_s,
            ny_s,
            nx_f,
            ny_f,
            T,
            return_d_inv=return_d_inv,
            **kwargs,
        )
    return fn(
        x_s,
        y_s,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        bdry,
        return_d_inv=return_d_inv,
        **kwargs,
    )


def _fluid_velocity_on_solid(Tx, Ty, u_f, v_f):
    """Transfer padded MAC velocity fields to solid nodal velocity vectors."""
    v_s_x = Tx @ u_f[1:-1, 1:].ravel()
    v_s_y = Ty @ v_f[1:, 1:-1].ravel()
    return np.concatenate((v_s_x, v_s_y))


def _spread_to_fluid(Tx, Ty, solid_residual, Nb, nx_f, ny_f, dx_f, dy_f, scale_by_cell_area):
    """Spread a solid residual vector back to MAC force arrays."""
    F_x = np.asarray(Tx.T @ solid_residual[:Nb]).reshape((ny_f, nx_f + 1))
    F_y = np.asarray(Ty.T @ solid_residual[Nb:]).reshape((ny_f + 1, nx_f))
    if scale_by_cell_area:
        cell_area = dx_f * dy_f
        F_x = F_x / cell_area
        F_y = F_y / cell_area
    return -F_x, -F_y


def _assemble_fluid_pressure(
    fluid_assembly_fn,
    nx_f,
    ny_f,
    dx_f,
    dy_f,
    G_f,
    pressure_solver,
    fluid_assembly_kwargs,
):
    """Assemble the fluid pressure solver with default channel behavior."""
    if fluid_assembly_fn is None:
        return fluid_assembly_CF(
            nx_f, ny_f, dx_f, dy_f, G_f, pressure_solver=pressure_solver
        )
    kwargs = dict(fluid_assembly_kwargs or {})
    return fluid_assembly_fn(nx_f, ny_f, dx_f, dy_f, G_f, **kwargs)


def _call_fluid_step(
    fluid_step_fn,
    u_old,
    v_old,
    nx_f,
    ny_f,
    dx_f,
    dy_f,
    dt,
    nu_f,
    F_x,
    F_y,
    Af,
    bf,
    vel0_step,
    rho_f,
    inlet_v_step,
    fluid_step_kwargs,
):
    """Advance the fluid one step using the default or user-supplied solver."""
    if fluid_step_fn is None:
        return fluid_solver_CF(
            u_old.copy(),
            v_old.copy(),
            nx_f,
            ny_f,
            dx_f,
            dy_f,
            dt,
            nu_f,
            F_x,
            F_y,
            Af,
            bf,
            vel0_step,
            rho_f,
            inlet_v=inlet_v_step,
        )

    kwargs = dict(fluid_step_kwargs or {})
    return fluid_step_fn(
        u_old.copy(),
        v_old.copy(),
        nx_f,
        ny_f,
        dx_f,
        dy_f,
        dt,
        nu_f,
        F_x,
        F_y,
        Af,
        bf,
        vel0_step,
        rho_f,
        **kwargs,
    )


def _require_converged(method, step, iterations, err, tol, max_iterations, raise_on_nonconvergence):
    """Raise or warn when a fixed-point solve did not satisfy the tolerance."""
    if not np.isfinite(err):
        raise FloatingPointError(
            f"{method} fixed-point produced non-finite error at "
            f"time step {step}, iteration {iterations}."
        )
    if err <= tol:
        return

    message = (
        f"{method} fixed-point did not converge at time step {step}: "
        f"iterations={iterations}/{max_iterations}, error={err:g}, tol={tol:g}."
    )
    if raise_on_nonconvergence:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _solve_fsi(
    *,
    method,
    nx_f,
    ny_f,
    dx_f,
    dy_f,
    Nt,
    vel0_f,
    dt,
    nu_f,
    rho_f=1.0,
    G_f=0.0,
    rho_s=1.0,
    lx_s=None,
    ly_s=None,
    nx_s=None,
    ny_s=None,
    dx_s=None,
    dy_s=None,
    d0_x_s=0.0,
    d0_y_s=0.0,
    mu_s=None,
    lam_s=None,
    Nb=None,
    solid_flags=None,
    solid_mesh=None,
    solid_system=None,
    solid_assembly_fn=None,
    solid_assembly_kwargs=None,
    f1=None,
    f2=None,
    element_type=None,
    transfer_kind="ID",
    transfer_kwargs=None,
    tol=1.0e-6,
    convergence="residual",
    relative_tol=True,
    residual_reference="old_lambda",
    norm_floor=1.0,
    minit=0,
    maxit=100,
    use_aitken=True,
    aitken_omega0=0.5,
    aitken_omega_min=1.0e-4,
    aitken_omega_max=1.0,
    use_true_lambda=False,
    mass_model="buoyant",
    mass_scale=None,
    rayleigh_alpha=0.0,
    rayleigh_beta=0.0,
    newmark_beta=0.25,
    newmark_gamma=0.5,
    anchor_nodes=None,
    fixed_nodes=None,
    pressure_solver="superlu",
    fluid_assembly_fn=None,
    fluid_step_fn=None,
    fluid_assembly_kwargs=None,
    fluid_step_kwargs=None,
    initial_fluid_state=None,
    initial_solid_state=None,
    initial_lambda=None,
    initialize_fluid=False,
    inlet_velocity_fn=None,
    inlet_ramp_time=0.0,
    ramped_inlet=None,
    stop_ramp=0.0,
    inlet_v_shape=None,
    inlet_v_amplitude=0.0,
    inlet_v_frequency=1.0,
    scale_forces_by_cell_area=True,
    store_fluid_history=True,
    store_fluid_pressure=True,
    store_solid_history=True,
    history_stride=1,
    store_iteration_history=True,
    raise_on_nonconvergence=True,
    verbose=True,
    print_every=10,
    return_tuple=False,
):
    """Shared implementation for ``solve_ib`` and ``solve_id``."""
    method = str(method).strip().upper()
    if method not in {"IB", "ID"}:
        raise ValueError("method must be 'IB' or 'ID'.")

    nx_f, ny_f, Nt = _validate_grid_and_time(
        nx_f, ny_f, dx_f, dy_f, Nt, dt, nu_f, rho_f
    )
    vel0_f = _as_velocity_profile(vel0_f, ny_f)
    if inlet_velocity_fn is not None and not callable(inlet_velocity_fn):
        raise ValueError("inlet_velocity_fn must be callable or None.")
    history_stride = _validate_history_stride(history_stride)
    if newmark_beta <= 0.0 or newmark_gamma <= 0.0:
        raise ValueError("newmark_beta and newmark_gamma must be positive.")

    controls = AitkenControls(
        min_iterations=minit,
        max_iterations=maxit,
        omega0=aitken_omega0,
        omega_min=aitken_omega_min,
        omega_max=aitken_omega_max,
        enabled=use_aitken,
    ).validated()
    convergence_controls = ConvergenceControls(
        tol=float(tol),
        mode=convergence,
        relative=bool(relative_tol),
        residual_reference=residual_reference,
        norm_floor=float(norm_floor),
    )
    if convergence_controls.tol < 0.0:
        raise ValueError("tol must be non-negative.")

    system, element_type = _resolve_solid_system(
        method=method,
        element_type=element_type,
        solid_system=solid_system,
        solid_mesh=solid_mesh,
        solid_assembly_fn=solid_assembly_fn,
        solid_assembly_kwargs=solid_assembly_kwargs,
        lx_s=lx_s,
        ly_s=ly_s,
        nx_s=nx_s,
        ny_s=ny_s,
        dx_s=dx_s,
        dy_s=dy_s,
        lam_s=lam_s,
        mu_s=mu_s,
        solid_flags=solid_flags,
        f1=f1,
        f2=f2,
    )

    Nb_s = system.Nb
    if system.internal_force is not None:
        if not callable(system.internal_force):
            raise ValueError("solid_system.internal_force must be callable.")
        if method != "ID":
            raise ValueError("Nonlinear internal forces are currently supported only by solve_id.")
        if rayleigh_beta != 0.0:
            raise ValueError("Nonlinear ID currently requires rayleigh_beta=0.")
    if Nb is not None and int(Nb) != Nb_s:
        raise ValueError(f"Nb={Nb} does not match solid mesh nodes {Nb_s}.")
    Nb = Nb_s

    d0_x_s = _as_offset(d0_x_s, Nb, "d0_x_s")
    d0_y_s = _as_offset(d0_y_s, Nb, "d0_y_s")

    fixed = np.union1d(dirichlet_nodes_from_bdry(system.bdry), _as_nodes(anchor_nodes, "anchor_nodes"))
    fixed = np.union1d(fixed, _as_nodes(fixed_nodes, "fixed_nodes"))
    if fixed.size and np.any(fixed >= Nb):
        raise ValueError("fixed or anchor node index outside valid range.")

    bdry_nodes = boundary_nodes(system.bdry)
    if bdry_nodes.size and np.any(bdry_nodes >= Nb):
        raise ValueError("bdry contains node indices outside valid range.")
    coupling_nodes = np.setdiff1d(bdry_nodes, fixed, assume_unique=False)

    M_eff = _effective_mass_matrix(
        system.M_el,
        rho_s=rho_s,
        rho_f=rho_f,
        mass_model=mass_model,
        mass_scale=mass_scale,
    )
    C_s = rayleigh_damping_matrix(
        M_eff, system.A_s, rayleigh_alpha, rayleigh_beta
    )

    solve_s = None
    if method == "IB":
        prescribed_nodes = np.union1d(bdry_nodes, fixed)
        R = M_eff / (newmark_gamma * dt) + (newmark_beta * dt / newmark_gamma) * system.A_s
        if C_s is not None:
            R = R + C_s
        R = R.tolil()
        for node in prescribed_nodes:
            R[node, :] = 0.0
            R[node, node] = 1.0
            R[node + Nb, :] = 0.0
            R[node + Nb, node + Nb] = 1.0
        solve_s = factorized(R.tocsc())

    Af, bf = _assemble_fluid_pressure(
        fluid_assembly_fn,
        nx_f,
        ny_f,
        dx_f,
        dy_f,
        G_f,
        pressure_solver,
        fluid_assembly_kwargs,
    )

    u_f, v_f, p_f = initial_fluid_fields(initial_fluid_state, nx_f, ny_f)
    if initialize_fluid and initial_fluid_state is None:
        u_f, v_f = initialize_fluid_from_inlet(u_f, v_f, vel0_f, nx_f, ny_f)

    u_s = np.zeros(2 * Nb)
    v_s = np.zeros(2 * Nb)
    a_s = np.zeros(2 * Nb)
    if initial_solid_state is not None:
        if isinstance(initial_solid_state, dict):
            if "u_s" in initial_solid_state:
                u_s = _as_state_vector(initial_solid_state["u_s"], 2 * Nb, "initial_solid_state['u_s']")
            if "v_s" in initial_solid_state:
                v_s = _as_state_vector(initial_solid_state["v_s"], 2 * Nb, "initial_solid_state['v_s']")
            if "a_s" in initial_solid_state:
                a_s = _as_state_vector(initial_solid_state["a_s"], 2 * Nb, "initial_solid_state['a_s']")
        elif isinstance(initial_solid_state, (tuple, list)) and len(initial_solid_state) == 3:
            u_s = _as_state_vector(initial_solid_state[0], 2 * Nb, "initial_solid_state[0]")
            v_s = _as_state_vector(initial_solid_state[1], 2 * Nb, "initial_solid_state[1]")
            a_s = _as_state_vector(initial_solid_state[2], 2 * Nb, "initial_solid_state[2]")
        else:
            raise ValueError("initial_solid_state must be a dict or (u_s, v_s, a_s).")
    _zero_nodes(u_s, fixed, Nb)
    _zero_nodes(v_s, fixed, Nb)
    _zero_nodes(a_s, fixed, Nb)

    if initial_lambda is None:
        lam = np.zeros(2 * Nb)
    else:
        lam = _as_state_vector(initial_lambda, 2 * Nb, "initial_lambda")
        _zero_nodes(lam, fixed, Nb)

    usol_f, vsol_f, psol_f = [], [], []
    usol_s, vsol_s, asol_s = [], [], []
    if store_fluid_history:
        usol_f.append(u_f.copy())
        vsol_f.append(v_f.copy())
        if store_fluid_pressure:
            psol_f.append(p_f.copy())
    if store_solid_history:
        usol_s.append(u_s.copy())
        vsol_s.append(v_s.copy())
        asol_s.append(a_s.copy())

    transfer_kind_normalized = (
        "IB" if method == "IB" else _normalize_transfer_kind(transfer_kind)
    )
    total_iterations = 0
    iteration_history = []

    ramp_time = 0.0
    if ramped_inlet is True:
        ramp_time = float(stop_ramp if stop_ramp else inlet_ramp_time)
        if ramp_time <= 0.0:
            raise ValueError("stop_ramp or inlet_ramp_time must be positive when ramped_inlet is True.")
    elif ramped_inlet is False:
        ramp_time = 0.0
    else:
        ramp_time = float(inlet_ramp_time)

    for step in range(Nt):
        u_f_old = u_f.copy()
        v_f_old = v_f.copy()

        time = (step + 1) * dt
        inlet_scale = smooth_inlet_ramp(time, ramp_time) if ramp_time > 0.0 else 1.0
        vel0_step = inlet_scale * vel0_f
        if inlet_velocity_fn is not None:
            vel0_step = _as_velocity_profile(inlet_velocity_fn(time), ny_f)
            if not np.all(np.isfinite(vel0_step)):
                raise ValueError("inlet_velocity_fn must return a finite inlet profile.")
        inlet_v_step = fluid_response_inlet_v_step(
            inlet_v_shape,
            inlet_v_amplitude,
            inlet_v_frequency,
            time,
            inlet_scale,
        )

        x_s = u_s[:Nb] + system.P[0, :] + d0_x_s
        y_s = u_s[Nb:] + system.P[1, :] + d0_y_s
        transfer_result = _assemble_transfer(
            method=method,
            transfer_kind=transfer_kind_normalized,
            element_type=element_type,
            x_s=x_s,
            y_s=y_s,
            dx_f=dx_f,
            dy_f=dy_f,
            nx_s=nx_s,
            ny_s=ny_s,
            nx_f=nx_f,
            ny_f=ny_f,
            T=system.T,
            bdry=system.bdry,
            return_d_inv=use_true_lambda,
            transfer_kwargs=transfer_kwargs,
        )
        if use_true_lambda:
            Tx, Ty, d_inv = transfer_result
            d_lump = np.zeros_like(d_inv)
            nonzero = d_inv > 0.0
            d_lump[nonzero] = 1.0 / d_inv[nonzero]
            d_inv_vec = np.concatenate((d_inv, d_inv))
            d_lump_vec = np.concatenate((d_lump, d_lump))
        else:
            Tx, Ty = transfer_result
            d_inv_vec = None
            d_lump_vec = None

        un = u_s + dt * v_s + dt**2 * (0.5 - newmark_beta) * a_s
        vn = v_s + dt * (1.0 - newmark_gamma) * a_s
        _zero_nodes(un, fixed, Nb)
        _zero_nodes(vn, fixed, Nb)

        rhs_base = None
        if method == "IB":
            rhs_base = (
                (1.0 / (newmark_gamma * dt)) * (M_eff @ vn)
                - system.A_s @ (un - (newmark_beta * dt / newmark_gamma) * vn)
                + system.b_s
            )

        iteration = 0
        err = np.inf
        residual_error = np.inf
        position_error = np.inf
        relaxation = _AitkenRelaxation(controls)

        while (err > convergence_controls.tol and iteration < controls.max_iterations) or iteration < controls.min_iterations:
            u_s_prev_iter = u_s.copy()

            if method == "IB":
                rhs = rhs_base.copy()
                v_boundary = _fluid_velocity_on_solid(Tx, Ty, u_f, v_f)
                rhs[coupling_nodes] = v_boundary[coupling_nodes]
                rhs[coupling_nodes + Nb] = v_boundary[coupling_nodes + Nb]
                _zero_nodes(rhs, fixed, Nb)
                v_s = solve_s(rhs)
                _zero_nodes(v_s, fixed, Nb)
            else:
                v_s = _fluid_velocity_on_solid(Tx, Ty, u_f, v_f)
                _zero_nodes(v_s, fixed, Nb)

            a_s = (v_s - vn) / (newmark_gamma * dt)
            u_s = un + newmark_beta * dt**2 * a_s
            _zero_nodes(a_s, fixed, Nb)
            _zero_nodes(u_s, fixed, Nb)

            if system.internal_force is None:
                internal_force = system.A_s @ u_s
            else:
                internal_force = np.asarray(system.internal_force(u_s), dtype=float)
                if internal_force.shape != u_s.shape or not np.all(np.isfinite(internal_force)):
                    raise ValueError("internal_force must return a finite vector of length 2*Nb.")
            residual_raw = (M_eff @ a_s) + internal_force - system.b_s
            if C_s is not None:
                residual_raw = residual_raw + C_s @ v_s

            if method == "IB":
                coupling_residual_raw = np.zeros_like(residual_raw)
                coupling_residual_raw[coupling_nodes] = residual_raw[coupling_nodes]
                coupling_residual_raw[coupling_nodes + Nb] = residual_raw[coupling_nodes + Nb]
            else:
                coupling_residual_raw = residual_raw.copy()
                _zero_nodes(coupling_residual_raw, fixed, Nb)

            lam_raw = (
                d_inv_vec * coupling_residual_raw
                if use_true_lambda
                else coupling_residual_raw
            )
            lam_old = lam.copy()
            lam, aitken_residual, omega = relaxation.update(lam_raw, lam_old)
            _zero_nodes(lam, fixed, Nb)

            solid_residual = d_lump_vec * lam if use_true_lambda else lam
            F_x, F_y = _spread_to_fluid(
                Tx,
                Ty,
                solid_residual,
                Nb,
                nx_f,
                ny_f,
                dx_f,
                dy_f,
                scale_forces_by_cell_area,
            )

            u_f, v_f, p_f = _call_fluid_step(
                fluid_step_fn,
                u_f_old,
                v_f_old,
                nx_f,
                ny_f,
                dx_f,
                dy_f,
                dt,
                nu_f,
                F_x,
                F_y,
                Af,
                bf,
                vel0_step,
                rho_f,
                inlet_v_step,
                fluid_step_kwargs,
            )

            residual_error = _lambda_residual_error(
                aitken_residual,
                lam_old,
                lam,
                lam_raw,
                relative=convergence_controls.relative,
                reference=convergence_controls.residual_reference,
                norm_floor=convergence_controls.norm_floor,
            )
            position_error = _solid_position_error(
                system.M,
                u_s,
                u_s_prev_iter,
                convergence_controls.relative,
                convergence_controls.norm_floor,
            )
            err = _select_convergence_error(
                convergence_controls.mode,
                residual_error,
                position_error,
            )

            if store_iteration_history:
                iteration_history.append(
                    {
                        "step": step,
                        "iteration": iteration + 1,
                        "error": float(err),
                        "residual_error": float(residual_error),
                        "position_error": float(position_error),
                        "omega": float(omega),
                    }
                )
            iteration += 1

        _require_converged(
            f"{method}/{transfer_kind_normalized}",
            step,
            iteration,
            err,
            convergence_controls.tol,
            controls.max_iterations,
            raise_on_nonconvergence,
        )

        total_iterations += iteration
        if verbose and print_every and step % int(print_every) == 0:
            print(
                f"time step: {step}/{Nt}, iterations: {iteration}, "
                f"error: {err:g}, residual: {residual_error:g}, "
                f"position: {position_error:g}"
            )

        store_this_frame = _store_history_frame(step + 1, Nt, history_stride)
        if store_fluid_history and store_this_frame:
            usol_f.append(u_f.copy())
            vsol_f.append(v_f.copy())
            if store_fluid_pressure:
                psol_f.append(p_f.copy())
        if store_solid_history and store_this_frame:
            usol_s.append(u_s.copy())
            vsol_s.append(v_s.copy())
            asol_s.append(a_s.copy())

    if not store_fluid_history:
        usol_f = [u_f.copy()]
        vsol_f = [v_f.copy()]
        psol_f = [p_f.copy()] if store_fluid_pressure else []
    if not store_solid_history:
        usol_s = [u_s.copy()]
        vsol_s = [v_s.copy()]
        asol_s = [a_s.copy()]

    if verbose:
        print("total iterations:", total_iterations)

    result = FSIResult(
        usol_f=usol_f,
        vsol_f=vsol_f,
        psol_f=psol_f,
        usol_s=usol_s,
        vsol_s=vsol_s,
        asol_s=asol_s,
        P=system.P,
        T=system.T,
        bdry=system.bdry,
        total_iterations=total_iterations,
        iteration_history=iteration_history,
        final_lambda=lam.copy(),
        method=method,
        transfer_kind=transfer_kind_normalized,
        element_type=element_type,
    )
    return result.as_legacy_tuple() if return_tuple else result


def solve_ib(
    nx_f,
    ny_f,
    dx_f,
    dy_f,
    Nt,
    vel0_f,
    *,
    dt,
    nu_f,
    rho_f=1.0,
    G_f=0.0,
    rho_s=1.0,
    lx_s=None,
    ly_s=None,
    nx_s=None,
    ny_s=None,
    dx_s=None,
    dy_s=None,
    d0_x_s=0.0,
    d0_y_s=0.0,
    mu_s=None,
    lam_s=None,
    Nb=None,
    solid_flags=None,
    solid_mesh=None,
    solid_system=None,
    solid_assembly_fn=None,
    solid_assembly_kwargs=None,
    f1=None,
    f2=None,
    element_type=None,
    transfer_kwargs=None,
    tol=1.0e-6,
    convergence="residual",
    relative_tol=True,
    residual_reference="old_lambda",
    norm_floor=1.0,
    minit=0,
    maxit=100,
    use_aitken=True,
    aitken_omega0=0.5,
    aitken_omega_min=1.0e-4,
    aitken_omega_max=1.0,
    use_true_lambda=False,
    mass_model="buoyant",
    mass_scale=None,
    rayleigh_alpha=0.0,
    rayleigh_beta=0.0,
    newmark_beta=0.25,
    newmark_gamma=0.5,
    anchor_nodes=None,
    fixed_nodes=None,
    pressure_solver="superlu",
    fluid_assembly_fn=None,
    fluid_step_fn=None,
    fluid_assembly_kwargs=None,
    fluid_step_kwargs=None,
    initial_fluid_state=None,
    initial_solid_state=None,
    initial_lambda=None,
    initialize_fluid=False,
    inlet_velocity_fn=None,
    inlet_ramp_time=0.0,
    ramped_inlet=None,
    stop_ramp=0.0,
    inlet_v_shape=None,
    inlet_v_amplitude=0.0,
    inlet_v_frequency=1.0,
    scale_forces_by_cell_area=True,
    store_fluid_history=True,
    store_fluid_pressure=True,
    store_solid_history=True,
    history_stride=1,
    store_iteration_history=True,
    raise_on_nonconvergence=True,
    verbose=True,
    print_every=10,
    return_tuple=False,
):
    """Solve an immersed-boundary FSI problem on the configured MAC grid."""
    return _solve_fsi(
        method="IB",
        nx_f=nx_f,
        ny_f=ny_f,
        dx_f=dx_f,
        dy_f=dy_f,
        Nt=Nt,
        vel0_f=vel0_f,
        dt=dt,
        nu_f=nu_f,
        rho_f=rho_f,
        G_f=G_f,
        rho_s=rho_s,
        lx_s=lx_s,
        ly_s=ly_s,
        nx_s=nx_s,
        ny_s=ny_s,
        dx_s=dx_s,
        dy_s=dy_s,
        d0_x_s=d0_x_s,
        d0_y_s=d0_y_s,
        mu_s=mu_s,
        lam_s=lam_s,
        Nb=Nb,
        solid_flags=solid_flags,
        solid_mesh=solid_mesh,
        solid_system=solid_system,
        solid_assembly_fn=solid_assembly_fn,
        solid_assembly_kwargs=solid_assembly_kwargs,
        f1=f1,
        f2=f2,
        element_type=element_type,
        transfer_kind="IB",
        transfer_kwargs=transfer_kwargs,
        tol=tol,
        convergence=convergence,
        relative_tol=relative_tol,
        residual_reference=residual_reference,
        norm_floor=norm_floor,
        minit=minit,
        maxit=maxit,
        use_aitken=use_aitken,
        aitken_omega0=aitken_omega0,
        aitken_omega_min=aitken_omega_min,
        aitken_omega_max=aitken_omega_max,
        use_true_lambda=use_true_lambda,
        mass_model=mass_model,
        mass_scale=mass_scale,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        newmark_beta=newmark_beta,
        newmark_gamma=newmark_gamma,
        anchor_nodes=anchor_nodes,
        fixed_nodes=fixed_nodes,
        pressure_solver=pressure_solver,
        fluid_assembly_fn=fluid_assembly_fn,
        fluid_step_fn=fluid_step_fn,
        fluid_assembly_kwargs=fluid_assembly_kwargs,
        fluid_step_kwargs=fluid_step_kwargs,
        initial_fluid_state=initial_fluid_state,
        initial_solid_state=initial_solid_state,
        initial_lambda=initial_lambda,
        initialize_fluid=initialize_fluid,
        inlet_velocity_fn=inlet_velocity_fn,
        inlet_ramp_time=inlet_ramp_time,
        ramped_inlet=ramped_inlet,
        stop_ramp=stop_ramp,
        inlet_v_shape=inlet_v_shape,
        inlet_v_amplitude=inlet_v_amplitude,
        inlet_v_frequency=inlet_v_frequency,
        scale_forces_by_cell_area=scale_forces_by_cell_area,
        store_fluid_history=store_fluid_history,
        store_fluid_pressure=store_fluid_pressure,
        store_solid_history=store_solid_history,
        history_stride=history_stride,
        store_iteration_history=store_iteration_history,
        raise_on_nonconvergence=raise_on_nonconvergence,
        verbose=verbose,
        print_every=print_every,
        return_tuple=return_tuple,
    )


def solve_id(
    nx_f,
    ny_f,
    dx_f,
    dy_f,
    Nt,
    vel0_f,
    *,
    dt,
    nu_f,
    rho_f=1.0,
    G_f=0.0,
    rho_s=1.0,
    lx_s=None,
    ly_s=None,
    nx_s=None,
    ny_s=None,
    dx_s=None,
    dy_s=None,
    d0_x_s=0.0,
    d0_y_s=0.0,
    mu_s=None,
    lam_s=None,
    Nb=None,
    solid_flags=None,
    solid_mesh=None,
    solid_system=None,
    solid_assembly_fn=None,
    solid_assembly_kwargs=None,
    f1=None,
    f2=None,
    element_type=None,
    transfer_kind="ID",
    transfer_kwargs=None,
    tol=1.0e-6,
    convergence="residual",
    relative_tol=True,
    residual_reference="old_lambda",
    norm_floor=1.0,
    minit=0,
    maxit=100,
    use_aitken=True,
    aitken_omega0=0.5,
    aitken_omega_min=1.0e-4,
    aitken_omega_max=1.0,
    use_true_lambda=False,
    mass_model="buoyant",
    mass_scale=None,
    rayleigh_alpha=0.0,
    rayleigh_beta=0.0,
    newmark_beta=0.25,
    newmark_gamma=0.5,
    anchor_nodes=None,
    fixed_nodes=None,
    pressure_solver="superlu",
    fluid_assembly_fn=None,
    fluid_step_fn=None,
    fluid_assembly_kwargs=None,
    fluid_step_kwargs=None,
    initial_fluid_state=None,
    initial_solid_state=None,
    initial_lambda=None,
    initialize_fluid=False,
    inlet_velocity_fn=None,
    inlet_ramp_time=0.0,
    ramped_inlet=None,
    stop_ramp=0.0,
    inlet_v_shape=None,
    inlet_v_amplitude=0.0,
    inlet_v_frequency=1.0,
    scale_forces_by_cell_area=True,
    store_fluid_history=True,
    store_fluid_pressure=True,
    store_solid_history=True,
    history_stride=1,
    store_iteration_history=True,
    raise_on_nonconvergence=True,
    verbose=True,
    print_every=10,
    return_tuple=False,
):
    """Solve an ID-family FSI problem, with optional MIX or MIX_noC transfer."""
    return _solve_fsi(
        method="ID",
        nx_f=nx_f,
        ny_f=ny_f,
        dx_f=dx_f,
        dy_f=dy_f,
        Nt=Nt,
        vel0_f=vel0_f,
        dt=dt,
        nu_f=nu_f,
        rho_f=rho_f,
        G_f=G_f,
        rho_s=rho_s,
        lx_s=lx_s,
        ly_s=ly_s,
        nx_s=nx_s,
        ny_s=ny_s,
        dx_s=dx_s,
        dy_s=dy_s,
        d0_x_s=d0_x_s,
        d0_y_s=d0_y_s,
        mu_s=mu_s,
        lam_s=lam_s,
        Nb=Nb,
        solid_flags=solid_flags,
        solid_mesh=solid_mesh,
        solid_system=solid_system,
        solid_assembly_fn=solid_assembly_fn,
        solid_assembly_kwargs=solid_assembly_kwargs,
        f1=f1,
        f2=f2,
        element_type=element_type,
        transfer_kind=transfer_kind,
        transfer_kwargs=transfer_kwargs,
        tol=tol,
        convergence=convergence,
        relative_tol=relative_tol,
        residual_reference=residual_reference,
        norm_floor=norm_floor,
        minit=minit,
        maxit=maxit,
        use_aitken=use_aitken,
        aitken_omega0=aitken_omega0,
        aitken_omega_min=aitken_omega_min,
        aitken_omega_max=aitken_omega_max,
        use_true_lambda=use_true_lambda,
        mass_model=mass_model,
        mass_scale=mass_scale,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
        newmark_beta=newmark_beta,
        newmark_gamma=newmark_gamma,
        anchor_nodes=anchor_nodes,
        fixed_nodes=fixed_nodes,
        pressure_solver=pressure_solver,
        fluid_assembly_fn=fluid_assembly_fn,
        fluid_step_fn=fluid_step_fn,
        fluid_assembly_kwargs=fluid_assembly_kwargs,
        fluid_step_kwargs=fluid_step_kwargs,
        initial_fluid_state=initial_fluid_state,
        initial_solid_state=initial_solid_state,
        initial_lambda=initial_lambda,
        initialize_fluid=initialize_fluid,
        inlet_velocity_fn=inlet_velocity_fn,
        inlet_ramp_time=inlet_ramp_time,
        ramped_inlet=ramped_inlet,
        stop_ramp=stop_ramp,
        inlet_v_shape=inlet_v_shape,
        inlet_v_amplitude=inlet_v_amplitude,
        inlet_v_frequency=inlet_v_frequency,
        scale_forces_by_cell_area=scale_forces_by_cell_area,
        store_fluid_history=store_fluid_history,
        store_fluid_pressure=store_fluid_pressure,
        store_solid_history=store_solid_history,
        history_stride=history_stride,
        store_iteration_history=store_iteration_history,
        raise_on_nonconvergence=raise_on_nonconvergence,
        verbose=verbose,
        print_every=print_every,
        return_tuple=return_tuple,
    )


solve_IB = solve_ib
solve_ID = solve_id
