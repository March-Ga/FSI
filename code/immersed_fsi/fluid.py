"""Fluid-side routines for immersed FSI simulations.

This module collects the Cartesian MAC-grid fluid code that was previously
spread across ``utils.py``, ``components_ff.py``, case drivers, and plotting
helpers. It provides pressure Poisson assembly, SuperLU/DCT/CG pressure
solvers, Chorin projection velocity updates, obstacle-aware fluid steps,
free-fall boundary-condition variants, inlet helpers, and MAC-grid diagnostic
utilities. It intentionally does not contain solid mechanics or transfer
operator assembly.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import dct, idct
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, splu


class FastPoissonDCT:
    """Direct transform solver for the obstacle-free Cartesian pressure matrix."""

    def __init__(self, nx, ny, dx, dy):
        self.nx = int(nx)
        self.ny = int(ny)
        self.shape = (self.nx * self.ny, self.nx * self.ny)

        hx = float(dy) / float(dx)
        hy = float(dx) / float(dy)
        kx = np.arange(self.nx, dtype=float)
        ky = np.arange(self.ny, dtype=float)

        lambda_x = -4.0 * np.sin(np.pi * (kx + 0.5) / (2.0 * self.nx)) ** 2
        lambda_y = -4.0 * np.sin(np.pi * ky / (2.0 * self.ny)) ** 2
        self._denominator = hx * lambda_x[:, None] + hy * lambda_y[None, :]

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != (self.nx * self.ny,):
            rhs = rhs.reshape(self.nx * self.ny)

        coeffs = rhs.reshape((self.nx, self.ny))
        coeffs = dct(coeffs, type=4, norm="ortho", axis=0)
        coeffs = dct(coeffs, type=2, norm="ortho", axis=1)
        coeffs = coeffs / self._denominator
        values = idct(coeffs, type=2, norm="ortho", axis=1)
        values = dct(values, type=4, norm="ortho", axis=0)
        return values.reshape(self.nx * self.ny)


class SparsePoissonCG:
    """CG solve for a negative-definite sparse pressure matrix."""

    def __init__(self, A, rtol=1.0e-10, atol=0.0, maxiter=None, preconditioner=None):
        self.A = A.tocsr()
        self.shape = self.A.shape
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.maxiter = maxiter
        self.last_info = None
        self.last_iterations = 0
        self.total_iterations = 0
        self._x0 = None

        diagonal = -self.A.diagonal()
        if np.any(diagonal <= 0.0):
            raise ValueError("CG pressure matrix must be negative definite.")
        self._operator = -self.A
        if preconditioner is None:
            self._preconditioner = LinearOperator(
                self.shape,
                matvec=lambda x: x / diagonal,
                dtype=float,
            )
        else:
            self._preconditioner = preconditioner

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float).reshape(self.shape[0])
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            self._operator,
            -rhs,
            x0=self._x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self._preconditioner,
            callback=callback,
        )
        self.last_info = info
        self.last_iterations = iterations
        self.total_iterations += iterations
        if info != 0:
            raise RuntimeError(f"CG pressure solve did not converge; info={info}.")
        self._x0 = solution
        return solution


class FluidPressureSolverFF:
    """Wrapper carrying boundary-condition metadata for free-fall pressure solvers."""

    def __init__(self, solver, periodic_x, hesch_benchmark=False, pin_pressure_cell=True):
        self._solver = solver
        self.periodic_x = bool(periodic_x)
        self.hesch_benchmark = bool(hesch_benchmark)
        self.pin_pressure_cell = bool(pin_pressure_cell)
        self.shape = solver.shape

    def solve(self, rhs):
        return self._solver.solve(rhs)


_FluidPressureSolverFF = FluidPressureSolverFF


def build_fluid_grid(ref_params, resolution):
    """Build a standard fluid-grid metadata dictionary from reference parameters."""
    lx_f = ref_params["lx_f"]
    ly_f = ref_params["ly_f"]
    nx_f = int(round(lx_f / resolution))
    ny_f = int(round(ly_f / resolution))
    return {
        "lx_f": lx_f,
        "ly_f": ly_f,
        "nx_f": nx_f,
        "ny_f": ny_f,
        "dx_f": lx_f / nx_f,
        "dy_f": ly_f / ny_f,
        "rho_f": ref_params["rho_f"],
        "mu_f": ref_params["mu_f"],
        "nu_f": ref_params["mu_f"] / ref_params["rho_f"],
        "G_f": float(ref_params.get("G_f_kinematic", 0.0)),
    }


def poiseuille_profile(y, channel_height, max_velocity):
    """Return the parabolic Poiseuille velocity profile."""
    return 4.0 * max_velocity * y * (channel_height - y) / channel_height**2


def turek_inlet_profile(y, channel_height, mean_velocity):
    """Return the Turek-Hron parabolic inlet profile with prescribed mean velocity."""
    return 1.5 * mean_velocity * 4.0 * y * (channel_height - y) / channel_height**2


def turek_inlet_ramp(t, ramp_time=2.0):
    """Return the cosine ramp used by the Turek-Hron inlet."""
    t = np.asarray(t, dtype=float)
    if ramp_time <= 0.0:
        return np.ones_like(t, dtype=float)

    ramp = np.ones_like(t, dtype=float)
    mask = t < ramp_time
    ramp[mask] = 0.5 * (1.0 - np.cos(np.pi * t[mask] / ramp_time))
    return ramp


def inlet_scale(time, ramp_time):
    """Return a scalar cosine inlet ramp."""
    if ramp_time <= 0.0:
        return 1.0
    if time >= ramp_time:
        return 1.0
    return 0.5 * (1.0 - np.cos(np.pi * time / ramp_time))


def smooth_inlet_ramp(time, stop_time):
    """Return a smoothstep inlet ramp from zero to one."""
    if time <= 0.0:
        return 0.0
    if time >= stop_time:
        return 1.0
    s = time / stop_time
    return s**3 * (10.0 - 15.0 * s + 6.0 * s**2)


def fluid_response_inlet_v_step(inlet_v_shape, inlet_v_amplitude, inlet_v_frequency, time, scale):
    """Return an optional sinusoidal vertical inlet velocity profile."""
    if inlet_v_shape is None or inlet_v_amplitude == 0.0:
        return None
    return (
        scale
        * inlet_v_amplitude
        * np.asarray(inlet_v_shape)
        * np.sin(2.0 * np.pi * inlet_v_frequency * time)
    )


def initialize_fluid_from_inlet(u, v, vel0_f, nx_f, ny_f):
    """Initialize padded MAC arrays from an inlet profile on physical u-faces."""
    vel0_f = np.asarray(vel0_f, dtype=float).reshape(-1)
    if vel0_f.size != ny_f:
        raise ValueError("vel0_f must have one value per fluid cell row.")

    u[1:ny_f + 1, 1:nx_f + 2] = vel0_f[:, None]
    v[1:ny_f + 2, 1:nx_f + 1] = 0.0
    u[0, :] = -u[1, :]
    u[-1, :] = -u[-2, :]
    v[1, :] = 0.0
    v[-1, :] = 0.0
    return u, v


def as_state_array(values, shape, name):
    """Return a copied state array after validating its exact shape."""
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return arr.copy()


def initial_fluid_fields(initial_fluid_state, nx_f, ny_f):
    """Return initial padded ``u, v, p`` MAC arrays, optionally from a saved state."""
    shape = (ny_f + 2, nx_f + 2)
    p_f = np.zeros(shape)
    u_f = np.zeros(shape)
    v_f = np.zeros(shape)

    if initial_fluid_state is None:
        return u_f, v_f, p_f

    u_f = as_state_array(initial_fluid_state["u_f"], shape, "initial_fluid_state['u_f']")
    v_f = as_state_array(initial_fluid_state["v_f"], shape, "initial_fluid_state['v_f']")
    if "p_f" in initial_fluid_state:
        p_f = as_state_array(initial_fluid_state["p_f"], shape, "initial_fluid_state['p_f']")
    return u_f, v_f, p_f


def sample_regular_grid(values, x_grid, y_grid, xq, yq):
    """Bilinearly sample values stored on a regular x/y tensor grid."""
    values = np.asarray(values)
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    xq = np.asarray(xq)
    yq = np.asarray(yq)

    xq_clip = np.clip(xq, x_grid[0], x_grid[-1])
    yq_clip = np.clip(yq, y_grid[0], y_grid[-1])

    ix = np.searchsorted(x_grid, xq_clip, side="right") - 1
    iy = np.searchsorted(y_grid, yq_clip, side="right") - 1
    ix = np.clip(ix, 0, x_grid.size - 2)
    iy = np.clip(iy, 0, y_grid.size - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]
    tx = np.divide(xq_clip - x0, x1 - x0, out=np.zeros_like(xq_clip, dtype=float), where=x1 != x0)
    ty = np.divide(yq_clip - y0, y1 - y0, out=np.zeros_like(yq_clip, dtype=float), where=y1 != y0)

    v00 = values[iy, ix]
    v10 = values[iy, ix + 1]
    v01 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def physical_mac_fields(u, v, p, grid):
    """Return physical MAC arrays and coordinate vectors from padded solver arrays."""
    nx = grid["nx_f"]
    ny = grid["ny_f"]
    dx = grid["dx_f"]
    dy = grid["dy_f"]
    rho = grid.get("rho_f", 1.0)

    return {
        "u": u[1:ny + 1, 1:nx + 2],
        "v": v[1:ny + 2, 1:nx + 1],
        "p": rho * p[1:ny + 1, 1:nx + 1],
        "x_u": np.arange(nx + 1) * dx,
        "y_u": (np.arange(ny) + 0.5) * dy,
        "x_v": (np.arange(nx) + 0.5) * dx,
        "y_v": np.arange(ny + 1) * dy,
        "x_p": (np.arange(nx) + 0.5) * dx,
        "y_p": (np.arange(ny) + 0.5) * dy,
    }


def sample_mac_velocity(fields, x, y):
    """Sample MAC velocity components at arbitrary physical points."""
    ux = sample_regular_grid(fields["u"], fields["x_u"], fields["y_u"], x, y)
    vy = sample_regular_grid(fields["v"], fields["x_v"], fields["y_v"], x, y)
    return ux, vy


def resample_reference_velocity(u_ref, v_ref, ref_grid, target_grid):
    """Interpolate a reference MAC velocity field onto another MAC grid."""
    ref_fields = physical_mac_fields(u_ref, v_ref, np.zeros_like(u_ref), ref_grid)

    nx = target_grid["nx_f"]
    ny = target_grid["ny_f"]
    dx = target_grid["dx_f"]
    dy = target_grid["dy_f"]

    u0 = np.zeros((ny + 2, nx + 2))
    v0 = np.zeros_like(u0)

    x_u = np.arange(nx + 1) * dx
    y_u = (np.arange(ny) + 0.5) * dy
    xu, yu = np.meshgrid(x_u, y_u, indexing="xy")
    u0[1:ny + 1, 1:nx + 2] = sample_regular_grid(
        ref_fields["u"], ref_fields["x_u"], ref_fields["y_u"], xu, yu
    )

    x_v = (np.arange(nx) + 0.5) * dx
    y_v = np.arange(ny + 1) * dy
    xv, yv = np.meshgrid(x_v, y_v, indexing="xy")
    v0[1:ny + 2, 1:nx + 1] = sample_regular_grid(
        ref_fields["v"], ref_fields["x_v"], ref_fields["y_v"], xv, yv
    )
    return u0, v0


def pack_mac_fields(u_mac, v_mac, nx, ny):
    """Pack physical MAC arrays into padded solver arrays."""
    u = np.zeros((ny + 2, nx + 2))
    v = np.zeros_like(u)
    u[1:-1, 1:] = u_mac
    v[1:, 1:-1] = v_mac
    return u, v


def pack_mac_fields_with_ghosts(u_mac, v_mac, nx, ny):
    """Pack physical MAC arrays and apply standard channel velocity ghosts."""
    u, v = pack_mac_fields(u_mac, v_mac, nx, ny)
    return apply_fluid_velocity_bc(u, v, vel0=u[1:-1, 1], enforce_outlet_u=True)


def discrete_divergence(u, v, nx, ny, dx, dy):
    """Return the discrete divergence on pressure cells."""
    return (
        (u[1:ny + 1, 2:nx + 2] - u[1:ny + 1, 1:nx + 1]) / dx
        + (v[2:ny + 2, 1:nx + 1] - v[1:ny + 1, 1:nx + 1]) / dy
    )


def periodic_x_divergence_ff(u, v, nx, ny, dx, dy):
    """Return pressure-cell divergence for the free-fall periodic-x layout."""
    U, V = unpack_periodic_x_velocity_ff(u, v, nx, ny)
    return (
        (np.roll(U, -1, axis=1) - U) / dx
        + (V[1:, :] - V[:-1, :]) / dy
    )


def cell_centered_velocity(u, v):
    """Average MAC velocities onto pressure-cell centers."""
    uc = 0.5 * (u[1:-1, 1:-1] + u[1:-1, 2:])
    vc = 0.5 * (v[1:-1, 1:-1] + v[2:, 1:-1])
    return uc, vc


def cell_centered_vorticity(u, v, dx, dy):
    """Average vertex-centered curl onto pressure-cell centers."""
    curl = (v[1:, 1:] - v[1:, :-1]) / dx - (u[1:, 1:] - u[:-1, 1:]) / dy
    return 0.25 * (curl[:-1, :-1] + curl[1:, :-1] + curl[:-1, 1:] + curl[1:, 1:])


def cell_centered_fields(u, v, dx, dy):
    """Average MAC velocities and curl onto pressure-cell centers."""
    uc, vc = cell_centered_velocity(u, v)
    return uc, vc, cell_centered_vorticity(u, v, dx, dy)


def fluid_cell_x(data, dx_f, dy_f, nx_f, ny_f):
    """Return flattened x-velocity MAC control-volume index for a point."""
    ix_x = np.floor((data[0] + dx_f / 2) / dx_f).astype(int)
    iy_x = np.floor(data[1] / dy_f).astype(int)
    ix_x = np.clip(ix_x, 0, nx_f)
    iy_x = np.clip(iy_x, 0, ny_f - 1)
    return int(iy_x * (nx_f + 1) + ix_x)


def fluid_cell_y(data, dx_f, dy_f, nx_f, ny_f):
    """Return flattened y-velocity MAC control-volume index for a point."""
    ix_y = np.floor(data[0] / dx_f).astype(int)
    iy_y = np.floor((data[1] + dy_f / 2) / dy_f).astype(int)
    ix_y = np.clip(ix_y, 0, nx_f - 1)
    iy_y = np.clip(iy_y, 0, ny_f)
    return int(iy_y * nx_f + ix_y)


def fluid_assembly_CF_debug(nx, ny, dx, dy, G):
    """Assemble the channel pressure Poisson operator with explicit branches."""
    nn = nx * ny
    b = np.zeros(nn)
    rows = []
    cols = []
    data = []

    def node_index(i, j):
        return i * ny + j

    def add_entries(row, col, value):
        row = np.asarray(row, dtype=np.int64).ravel()
        col = np.asarray(col, dtype=np.int64).ravel()
        rows.append(row)
        cols.append(col)
        data.append(np.full(row.shape, value, dtype=float))

    hx = dy / dx
    hy = dx / dy

    if nx > 2 and ny > 2:
        i, j = np.meshgrid(np.arange(1, nx - 1), np.arange(1, ny - 1), indexing="ij")
        r = node_index(i, j)
        add_entries(r, r, -2 * hy - 2 * hx)
        add_entries(r, node_index(i + 1, j), hx)
        add_entries(r, node_index(i - 1, j), hx)
        add_entries(r, node_index(i, j + 1), hy)
        add_entries(r, node_index(i, j - 1), hy)

    if nx > 2:
        i = np.arange(1, nx - 1)
        r = node_index(i, 0)
        add_entries(r, r, -hy - 2 * hx)
        add_entries(r, node_index(i + 1, 0), hx)
        add_entries(r, node_index(i - 1, 0), hx)
        add_entries(r, node_index(i, 1), hy)

        r = node_index(i, ny - 1)
        add_entries(r, r, -hy - 2 * hx)
        add_entries(r, node_index(i + 1, ny - 1), hx)
        add_entries(r, node_index(i - 1, ny - 1), hx)
        add_entries(r, node_index(i, ny - 2), hy)

    if ny > 2:
        j = np.arange(1, ny - 1)
        r = node_index(0, j)
        add_entries(r, r, -hx - 2 * hy)
        add_entries(r, node_index(0, j + 1), hy)
        add_entries(r, node_index(0, j - 1), hy)
        add_entries(r, node_index(1, j), hx)
        b[r] = G * dy

        r = node_index(nx - 1, j)
        add_entries(r, r, -3 * hx - 2 * hy)
        add_entries(r, node_index(nx - 1, j + 1), hy)
        add_entries(r, node_index(nx - 1, j - 1), hy)
        add_entries(r, node_index(nx - 2, j), hx)

    r = node_index(0, 0)
    add_entries([r], [r], -hy - hx)
    add_entries([r], [node_index(1, 0)], hx)
    add_entries([r], [node_index(0, 1)], hy)
    b[r] = G * dy

    r = node_index(nx - 1, 0)
    add_entries([r], [r], -hy - 3 * hx)
    add_entries([r], [node_index(nx - 2, 0)], hx)
    add_entries([r], [node_index(nx - 1, 1)], hy)

    r = node_index(0, ny - 1)
    add_entries([r], [r], -hy - hx)
    add_entries([r], [node_index(1, ny - 1)], hx)
    add_entries([r], [node_index(0, ny - 2)], hy)
    b[r] = G * dy

    r = node_index(nx - 1, ny - 1)
    add_entries([r], [r], -hy - 3 * hx)
    add_entries([r], [node_index(nx - 2, ny - 1)], hx)
    add_entries([r], [node_index(nx - 1, ny - 2)], hy)

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nn, nn),
    ).tocsc()
    return splu(A), b


@lru_cache(maxsize=16)
def _fluid_pressure_lu_cached(nx, ny, dx, dy):
    nn = nx * ny
    hx = dy / dx
    hy = dx / dy

    i = np.repeat(np.arange(nx, dtype=np.int64), ny)
    j = np.tile(np.arange(ny, dtype=np.int64), nx)
    r = i * ny + j

    main = np.full(nn, -2.0 * hx - 2.0 * hy)
    main[j == 0] += hy
    main[j == ny - 1] += hy
    main[i == 0] += hx
    main[i == nx - 1] -= hx

    rows = [r]
    cols = [r]
    data = [main]

    east = i < nx - 1
    rows.append(r[east])
    cols.append(r[east] + ny)
    data.append(np.full(np.count_nonzero(east), hx))

    west = i > 0
    rows.append(r[west])
    cols.append(r[west] - ny)
    data.append(np.full(np.count_nonzero(west), hx))

    north = j < ny - 1
    rows.append(r[north])
    cols.append(r[north] + 1)
    data.append(np.full(np.count_nonzero(north), hy))

    south = j > 0
    rows.append(r[south])
    cols.append(r[south] - 1)
    data.append(np.full(np.count_nonzero(south), hy))

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nn, nn),
    ).tocsc()
    return splu(A)


@lru_cache(maxsize=16)
def _fluid_pressure_dct_cached(nx, ny, dx, dy):
    return FastPoissonDCT(nx, ny, dx, dy)


def fluid_assembly_CF(nx, ny, dx, dy, G, pressure_solver="superlu"):
    """Assemble the obstacle-free channel pressure solver for ``fluid_solver_CF``."""
    pressure_solver = str(pressure_solver).strip().lower()
    if pressure_solver in {"superlu", "splu", "lu", "sparse"}:
        solver = _fluid_pressure_lu_cached(nx, ny, float(dx), float(dy))
    elif pressure_solver in {"dct", "fft", "fast"}:
        solver = _fluid_pressure_dct_cached(nx, ny, float(dx), float(dy))
    else:
        raise ValueError(
            "pressure_solver must be 'superlu'/'splu' or 'dct', "
            f"got {pressure_solver!r}"
        )

    b = np.zeros(nx * ny)
    b[:ny] = G * dy
    return solver, b


class TractionFreePressure:
    """Channel projection with -p + 2*nu*du/dx = 0 at the east outlet.

    Pressure is kinematic, as in ``fluid_solver_CF``. The outlet derivative
    uses the last two normal-velocity faces (first-order boundary closure).
    Eliminating the corrected outlet velocity modifies the east pressure row.
    See docs/idealized_heart_valve.md for the discrete boundary equations.
    """

    def __init__(self, nx, ny, dx, dy, dt, nu):
        self.grid = (nx, ny, dx, dy)
        self.dt, self.nu = dt, nu
        self.q = 4.0 * nu * dt / dx**2
        i = np.repeat(np.arange(nx), ny)
        j = np.tile(np.arange(ny), nx)
        r = i * ny + j
        hx, hy = dy / dx, dx / dy
        main = np.full(nx * ny, -2.0 * hx - 2.0 * hy)
        main[j == 0] += hy
        main[j == ny - 1] += hy
        main[i == 0] += hx
        main[i == nx - 1] += (2.0 - 3.0 / (1.0 + self.q)) * hx
        rows, cols, data = [r], [r], [main]
        for mask, offset, weight in (
            (i < nx - 1, ny, hx), (i > 0, -ny, hx),
            (j < ny - 1, 1, hy), (j > 0, -1, hy),
        ):
            values = np.full(np.count_nonzero(mask), weight)
            if offset == -ny:
                values[i[mask] == nx - 1] /= 1.0 + self.q
            rows.append(r[mask])
            cols.append(r[mask] + offset)
            data.append(values)
        A = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nx * ny, nx * ny),
        ).tocsc()
        self.solver = splu(A)

    def solve(self, rhs):
        return self.solver.solve(rhs)


@lru_cache(maxsize=8)
def _traction_free_pressure_cached(nx, ny, dx, dy, dt, nu):
    return TractionFreePressure(nx, ny, dx, dy, dt, nu)


def fluid_assembly_traction_free(nx, ny, dx, dy, G=0.0, *, dt, nu):
    """Assemble a no-slip channel with zero total traction at the outlet."""
    if not np.all(np.isfinite([dx, dy, dt, nu])) or nx < 3 or ny < 2 or dx <= 0.0 or dy <= 0.0 or dt <= 0.0 or nu < 0.0:
        raise ValueError("Traction-free channel requires nx>=3, ny>=2 and valid spacings, dt, nu.")
    if G != 0.0:
        raise ValueError("Traction-free outlet requires G=0.")
    return _traction_free_pressure_cached(nx, ny, dx, dy, dt, nu), np.zeros(nx * ny)


def apply_traction_free_velocity_bc(u, v, vel0, dx, dy, inlet_v=None):
    """Enforce inlet/walls and du/dy + dv/dx = 0 on the open east edge.

    The outlet normal velocity is determined by the projection. Wall/outlet
    corners retain the no-slip wall condition; traction is imposed away from
    those corners.
    """
    u, v = apply_fluid_velocity_bc(u, v, vel0, enforce_outlet_u=False, inlet_v=inlet_v)
    v[2:-1, -1] = v[2:-1, -2] - dx * np.diff(u[1:-1, -1]) / dy
    return u, v


def apply_fluid_velocity_bc(u, v, vel0, enforce_outlet_u=True, inlet_v=None):
    """Apply standard inlet/outlet/no-slip channel velocity boundary conditions."""
    u[1:-1, 1] = vel0
    if inlet_v is None:
        v[:, 0] = -v[:, 1]
    else:
        inlet_v = np.asarray(inlet_v, dtype=v.dtype).reshape(-1)
        ny = v.shape[0] - 2
        if inlet_v.size != ny + 1:
            raise ValueError("inlet_v must have one value per vertical velocity face row.")
        v[1:ny + 2, 0] = 2.0 * inlet_v - v[1:ny + 2, 1]

    if enforce_outlet_u:
        u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]

    u[-1, :] = -u[-2, :]
    v[-1, :] = 0.0
    u[0, :] = -u[1, :]
    v[1, :] = 0.0
    return u, v


_apply_fluid_velocity_bc = apply_fluid_velocity_bc


def fluid_solver_CF_debug(u, v, nx, ny, dx, dy, dt, nu, Fx, Fy, Af, bf, vel0, rho):
    """Debug Chorin projection step on an obstacle-free MAC channel grid."""
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    divut = np.zeros_like(v)

    u, v = apply_fluid_velocity_bc(u, v, vel0, enforce_outlet_u=True)

    ue = 0.5 * (u[1:ny + 1, 3:nx + 2] + u[1:ny + 1, 2:nx + 1])
    uw = 0.5 * (u[1:ny + 1, 2:nx + 1] + u[1:ny + 1, 1:nx])
    un = 0.5 * (u[2:ny + 2, 2:nx + 1] + u[1:ny + 1, 2:nx + 1])
    us = 0.5 * (u[1:ny + 1, 2:nx + 1] + u[0:ny, 2:nx + 1])
    vn = 0.5 * (v[2:ny + 2, 1:nx] + v[2:ny + 2, 2:nx + 1])
    vs = 0.5 * (v[1:ny + 1, 1:nx] + v[1:ny + 1, 2:nx + 1])

    convection = -(ue**2 - uw**2) / dx - (un * vn - us * vs) / dy
    diffusion = nu * (
        (u[1:ny + 1, 1:nx] - 2.0 * u[1:ny + 1, 2:nx + 1] + u[1:ny + 1, 3:nx + 2]) / dx**2
        + (u[0:ny, 2:nx + 1] - 2.0 * u[1:ny + 1, 2:nx + 1] + u[2:ny + 2, 2:nx + 1]) / dy**2
    )
    ut[1:ny + 1, 2:nx + 1] = u[1:ny + 1, 2:nx + 1] + dt * (convection + diffusion + Fx[:, 1:-1] / rho)

    ve = 0.5 * (v[2:ny + 1, 2:nx + 2] + v[2:ny + 1, 1:nx + 1])
    vw = 0.5 * (v[2:ny + 1, 1:nx + 1] + v[2:ny + 1, 0:nx])
    ue = 0.5 * (u[2:ny + 1, 2:nx + 2] + u[1:ny, 2:nx + 2])
    uw = 0.5 * (u[2:ny + 1, 1:nx + 1] + u[1:ny, 1:nx + 1])
    vn = 0.5 * (v[3:ny + 2, 1:nx + 1] + v[2:ny + 1, 1:nx + 1])
    vs = 0.5 * (v[2:ny + 1, 1:nx + 1] + v[1:ny, 1:nx + 1])

    convection = -(ue * ve - uw * vw) / dx - (vn**2 - vs**2) / dy
    diffusion = nu * (
        (v[2:ny + 1, 2:nx + 2] - 2.0 * v[2:ny + 1, 1:nx + 1] + v[2:ny + 1, 0:nx]) / dx**2
        + (v[3:ny + 2, 1:nx + 1] - 2.0 * v[2:ny + 1, 1:-1] + v[1:ny, 1:nx + 1]) / dy**2
    )
    vt[2:ny + 1, 1:nx + 1] = v[2:ny + 1, 1:nx + 1] + dt * (convection + diffusion + Fy[1:-1, :] / rho)

    ut, vt = apply_fluid_velocity_bc(ut, vt, vel0, enforce_outlet_u=True)
    divut[1:-1, 1:-1] = (
        (ut[1:-1, 2:] - ut[1:-1, 1:-1]) / dx
        + (vt[2:, 1:-1] - vt[1:-1, 1:-1]) / dy
    )
    prhs = divut[1:-1, 1:-1].ravel("F") * dx * dy / dt

    sol = Af.solve(prhs + bf)
    p[1:-1, 1:-1] = sol.reshape((ny, nx), order="F")

    u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / dx
    u[1:-1, -1] = ut[1:-1, -1] + 2.0 * dt * p[1:-1, -2] / dx
    v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / dy

    u, v = apply_fluid_velocity_bc(u, v, vel0, enforce_outlet_u=False)
    return u, v, p


def fluid_solver_CF(u, v, nx, ny, dx, dy, dt, nu, Fx, Fy, Af, bf, vel0, rho, inlet_v=None, traction_free=False):
    """Chorin channel step; optional zero total traction at the east outlet.

    ``traction_free=True`` requires ``fluid_assembly_traction_free`` with the
    same grid, dt, and kinematic viscosity. The default keeps the existing
    channel boundary treatment. Returned pressure is kinematic (physical/rho).
    """
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    force_scale = 1.0 / rho

    if traction_free:
        if not isinstance(Af, TractionFreePressure) or Af.grid != (nx, ny, dx, dy) or Af.dt != dt or Af.nu != nu:
            raise ValueError("Traction-free step requires a matching fluid_assembly_traction_free solver.")
        u, v = apply_traction_free_velocity_bc(u, v, vel0, dx, dy, inlet_v=inlet_v)
    else:
        u, v = apply_fluid_velocity_bc(u, v, vel0, enforce_outlet_u=True, inlet_v=inlet_v)

    u_center = u[1:ny + 1, 2:nx + 1]
    u_explicit = (
        -(
            (0.5 * (u[1:ny + 1, 3:nx + 2] + u_center)) ** 2
            - (0.5 * (u_center + u[1:ny + 1, 1:nx])) ** 2
        ) * inv_dx
        - (
            0.5 * (u[2:ny + 2, 2:nx + 1] + u_center)
            * 0.5 * (v[2:ny + 2, 1:nx] + v[2:ny + 2, 2:nx + 1])
            - 0.5 * (u_center + u[0:ny, 2:nx + 1])
            * 0.5 * (v[1:ny + 1, 1:nx] + v[1:ny + 1, 2:nx + 1])
        ) * inv_dy
        + nu * (
            (u[1:ny + 1, 1:nx] - 2.0 * u_center + u[1:ny + 1, 3:nx + 2]) * inv_dx2
            + (u[0:ny, 2:nx + 1] - 2.0 * u_center + u[2:ny + 2, 2:nx + 1]) * inv_dy2
        )
    )
    ut[1:ny + 1, 2:nx + 1] = u_center + dt * (u_explicit + Fx[:, 1:-1] * force_scale)

    v_center = v[2:ny + 1, 1:nx + 1]
    v_explicit = (
        - (
            0.5 * (u[2:ny + 1, 2:nx + 2] + u[1:ny, 2:nx + 2])
            * 0.5 * (v[2:ny + 1, 2:nx + 2] + v_center)
            - 0.5 * (u[2:ny + 1, 1:nx + 1] + u[1:ny, 1:nx + 1])
            * 0.5 * (v_center + v[2:ny + 1, 0:nx])
        ) * inv_dx
        - (
            (0.5 * (v[3:ny + 2, 1:nx + 1] + v_center)) ** 2
            - (0.5 * (v_center + v[1:ny, 1:nx + 1])) ** 2
        ) * inv_dy
        + nu * (
            (v[2:ny + 1, 2:nx + 2] - 2.0 * v_center + v[2:ny + 1, 0:nx]) * inv_dx2
            + (v[3:ny + 2, 1:nx + 1] - 2.0 * v_center + v[1:ny, 1:nx + 1]) * inv_dy2
        )
    )
    vt[2:ny + 1, 1:nx + 1] = v_center + dt * (v_explicit + Fy[1:-1, :] * force_scale)

    if traction_free:
        # Advance the existing outlet velocity in time. Extrapolate only its
        # interior momentum RHS, not the updated velocity itself: replacing
        # u_e by 2*u_i-u_{i-1} discards its history and can cause an O(1) change
        # as dt tends to zero, seeding the observed outlet instability.
        # The known body force is evaluated on the outlet face itself.
        outlet_predictor = u[1:-1, -1] + dt * (
            2.0 * u_explicit[:, -1] - u_explicit[:, -2]
            + Fx[:, -1] * force_scale
        )
        ut[1:-1, -1] = (outlet_predictor + Af.q * ut[1:-1, -2]) / (1.0 + Af.q)
        ut, vt = apply_traction_free_velocity_bc(ut, vt, vel0, dx, dy, inlet_v=inlet_v)
    else:
        ut, vt = apply_fluid_velocity_bc(ut, vt, vel0, enforce_outlet_u=True, inlet_v=inlet_v)
    prhs = (
        ((ut[1:-1, 2:] - ut[1:-1, 1:-1]) * inv_dx + (vt[2:, 1:-1] - vt[1:-1, 1:-1]) * inv_dy)
        .ravel("F")
        * dx
        * dy
        / dt
    )

    sol = Af.solve(prhs + bf)
    p[1:-1, 1:-1] = sol.reshape((ny, nx), order="F")

    u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) * inv_dx
    if traction_free:
        u[1:-1, -1] = (
            outlet_predictor + Af.q * u[1:-1, -2] + 2.0 * dt * p[1:-1, -2] * inv_dx
        ) / (1.0 + Af.q)
        p_boundary = 2.0 * nu * (u[1:-1, -1] - u[1:-1, -2]) * inv_dx
        p[1:-1, -1] = 2.0 * p_boundary - p[1:-1, -2]
        p[1:-1, 0] = p[1:-1, 1]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]
    else:
        u[1:-1, -1] = ut[1:-1, -1] + 2.0 * dt * p[1:-1, -2] * inv_dx
    v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) * inv_dy

    if traction_free:
        u, v = apply_traction_free_velocity_bc(u, v, vel0, dx, dy, inlet_v=inlet_v)
    else:
        u, v = apply_fluid_velocity_bc(u, v, vel0, enforce_outlet_u=False, inlet_v=inlet_v)
    return u, v, p


def obstacle_cell_mask(nx, ny, obstacle_cells):
    """Return a boolean pressure-cell mask with True on obstacle cells."""
    mask = np.zeros((ny, nx), dtype=bool)

    if obstacle_cells is None:
        return mask

    cells = np.asarray(obstacle_cells)
    if cells.size == 0:
        return mask

    if cells.dtype == bool:
        if cells.shape != (ny, nx):
            raise ValueError("Boolean obstacle mask must have shape (ny, nx).")
        return cells.copy()

    cells = np.asarray(obstacle_cells, dtype=np.int64)
    if cells.ndim == 1 and cells.shape[0] == 2:
        cells = cells.reshape(1, 2)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("obstacle_cells must be an array/list of (i, j) cell indices.")

    i = cells[:, 0]
    j = cells[:, 1]
    if np.any(i < 0) or np.any(i >= nx) or np.any(j < 0) or np.any(j >= ny):
        raise ValueError("Obstacle cell indices are outside the fluid grid.")

    mask[j, i] = True
    return mask


_obstacle_cell_mask = obstacle_cell_mask


def fluid_face_masks_from_cells(fluid_cells):
    """Return active physical MAC-face masks from active pressure cells."""
    ny, nx = fluid_cells.shape

    u_faces = np.zeros((ny, nx + 1), dtype=bool)
    u_faces[:, 0] = fluid_cells[:, 0]
    u_faces[:, -1] = fluid_cells[:, -1]
    if nx > 1:
        u_faces[:, 1:-1] = fluid_cells[:, :-1] & fluid_cells[:, 1:]

    v_faces = np.zeros((ny + 1, nx), dtype=bool)
    v_faces[0, :] = fluid_cells[0, :]
    v_faces[-1, :] = fluid_cells[-1, :]
    if ny > 1:
        v_faces[1:-1, :] = fluid_cells[:-1, :] & fluid_cells[1:, :]

    return u_faces, v_faces


_fluid_face_masks_from_cells = fluid_face_masks_from_cells


def apply_obstacle_tangential_no_slip(u_phys, v_phys, fluid_cells):
    """Fill obstacle-side tangential MAC ghosts for stationary no-slip walls."""
    obstacle = ~fluid_cells
    ny, nx = fluid_cells.shape

    if ny > 1 and nx > 1:
        bottom_wall = (
            obstacle[1:, :-1]
            & obstacle[1:, 1:]
            & fluid_cells[:-1, :-1]
            & fluid_cells[:-1, 1:]
        )
        u_phys[1:ny, 1:nx][bottom_wall] = -u_phys[0:ny - 1, 1:nx][bottom_wall]

        top_wall = (
            obstacle[:-1, :-1]
            & obstacle[:-1, 1:]
            & fluid_cells[1:, :-1]
            & fluid_cells[1:, 1:]
        )
        u_phys[0:ny - 1, 1:nx][top_wall] = -u_phys[1:ny, 1:nx][top_wall]

        left_wall = (
            obstacle[:-1, 1:]
            & obstacle[1:, 1:]
            & fluid_cells[:-1, :-1]
            & fluid_cells[1:, :-1]
        )
        v_phys[1:ny, 1:nx][left_wall] = -v_phys[1:ny, 0:nx - 1][left_wall]

        right_wall = (
            obstacle[:-1, :-1]
            & obstacle[1:, :-1]
            & fluid_cells[:-1, 1:]
            & fluid_cells[1:, 1:]
        )
        v_phys[1:ny, 0:nx - 1][right_wall] = -v_phys[1:ny, 1:nx][right_wall]


_apply_obstacle_tangential_no_slip = apply_obstacle_tangential_no_slip


def active_pressure_cells_fortran(fluid_cells):
    """Return active pressure cells flattened in x-major/Fortran order."""
    return np.flatnonzero(fluid_cells.ravel(order="F"))


_active_pressure_cells_fortran = active_pressure_cells_fortran


def fluid_assembly_CF_obstacle(
    nx,
    ny,
    dx,
    dy,
    G,
    obstacle_cells,
    pressure_solver="superlu",
    cg_rtol=1.0e-10,
    cg_atol=0.0,
    cg_maxiter=None,
):
    """Assemble the pressure solver on a Cartesian MAC grid with removed obstacle cells."""
    obstacle = obstacle_cell_mask(nx, ny, obstacle_cells)
    fluid_cells = ~obstacle

    if not np.any(fluid_cells):
        raise ValueError("The obstacle removes every pressure cell.")
    if not np.any(fluid_cells[:, -1]):
        raise ValueError("At least one fluid pressure cell must touch the east outlet.")

    active = active_pressure_cells_fortran(fluid_cells)
    nn = active.size

    global_to_active = np.full(nx * ny, -1, dtype=np.int64)
    global_to_active[active] = np.arange(nn, dtype=np.int64)

    hx = dy / dx
    hy = dx / dy

    rows = []
    cols = []
    data = []
    b = np.zeros(nn)

    def original_index(i, j):
        return i * ny + j

    def active_index(i, j):
        return global_to_active[original_index(i, j)]

    def add(row, col, value):
        rows.append(row)
        cols.append(col)
        data.append(value)

    for i in range(nx):
        for j in range(ny):
            row = active_index(i, j)
            if row < 0:
                continue

            diag = -2.0 * hx - 2.0 * hy

            if i == 0:
                diag += hx
                b[row] = G * dy
            elif fluid_cells[j, i - 1]:
                add(row, active_index(i - 1, j), hx)
            else:
                diag += hx

            if i == nx - 1:
                diag -= hx
            elif fluid_cells[j, i + 1]:
                add(row, active_index(i + 1, j), hx)
            else:
                diag += hx

            if j == 0:
                diag += hy
            elif fluid_cells[j - 1, i]:
                add(row, active_index(i, j - 1), hy)
            else:
                diag += hy

            if j == ny - 1:
                diag += hy
            elif fluid_cells[j + 1, i]:
                add(row, active_index(i, j + 1), hy)
            else:
                diag += hy

            add(row, row, diag)

    A = coo_matrix((data, (rows, cols)), shape=(nn, nn)).tocsc()
    pressure_solver = str(pressure_solver).strip().lower()
    if pressure_solver in {"superlu", "splu", "lu", "sparse"}:
        solver = splu(A)
    elif pressure_solver in {"cg", "iterative"}:
        solver = SparsePoissonCG(A, rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)
    elif pressure_solver in {"pcg_dct", "cg_dct", "dct_pcg"}:
        full_solver = _fluid_pressure_dct_cached(nx, ny, float(dx), float(dy))

        def precondition(active_rhs):
            full_rhs = np.zeros(nx * ny, dtype=float)
            full_rhs[active] = active_rhs
            full_solution = -full_solver.solve(full_rhs)
            return full_solution[active]

        solver = SparsePoissonCG(
            A,
            rtol=cg_rtol,
            atol=cg_atol,
            maxiter=cg_maxiter,
            preconditioner=LinearOperator((nn, nn), matvec=precondition, dtype=float),
        )
    elif pressure_solver in {"dct", "fft", "fast"}:
        raise ValueError("pressure_solver='dct' is not compatible with obstacle cells.")
    else:
        raise ValueError(
            "pressure_solver must be 'superlu', 'cg', or 'pcg_dct' for obstacle flows, "
            f"got {pressure_solver!r}."
        )
    return solver, b


def apply_fluid_velocity_bc_obstacle(
    u,
    v,
    vel0,
    fluid_cells,
    enforce_outlet_u=True,
    lateral_wall_bc="no-slip",
    inlet_v=None,
):
    """Apply channel BCs and stationary no-slip obstacle wall conditions."""
    ny, nx = fluid_cells.shape
    u_faces, v_faces = fluid_face_masks_from_cells(fluid_cells)

    u[1:-1, 1] = vel0 * fluid_cells[:, 0]
    if inlet_v is None:
        v[:, 0] = -v[:, 1]
    else:
        inlet_v = np.asarray(inlet_v, dtype=v.dtype).reshape(-1)
        if inlet_v.size != ny + 1:
            raise ValueError("inlet_v must have one value per vertical velocity face row.")
        v[1:ny + 2, 0] = 2.0 * inlet_v - v[1:ny + 2, 1]

    if enforce_outlet_u:
        u[:, -1] = u[:, -2]
    u[1:-1, -1] *= fluid_cells[:, -1]
    v[:, -1] = v[:, -2]

    if lateral_wall_bc == "no-slip":
        u[-1, :] = -u[-2, :]
        u[0, :] = -u[1, :]
    elif lateral_wall_bc == "free-slip":
        u[-1, :] = u[-2, :]
        u[0, :] = u[1, :]
    else:
        raise ValueError("lateral_wall_bc must be either 'no-slip' or 'free-slip'.")
    v[-1, :] = 0.0
    v[1, :] = 0.0

    u_phys = u[1:ny + 1, 1:nx + 2]
    v_phys = v[1:ny + 2, 1:nx + 1]
    u_phys[~u_faces] = 0.0
    v_phys[~v_faces] = 0.0
    apply_obstacle_tangential_no_slip(u_phys, v_phys, fluid_cells)
    return u, v


_apply_fluid_velocity_bc_obstacle = apply_fluid_velocity_bc_obstacle


def fluid_solver_CF_obstacle(
    u,
    v,
    nx,
    ny,
    dx,
    dy,
    dt,
    nu,
    Fx,
    Fy,
    Af,
    bf,
    vel0,
    rho,
    obstacle_cells,
    lateral_wall_bc="no-slip",
    inlet_v=None,
):
    """Chorin projection step on a MAC grid with removed obstacle pressure cells."""
    obstacle = obstacle_cell_mask(nx, ny, obstacle_cells)
    fluid_cells = ~obstacle
    active = active_pressure_cells_fortran(fluid_cells)
    if active.size != bf.size:
        raise ValueError(
            "Af/bf and obstacle_cells are inconsistent. Reassemble with "
            "fluid_assembly_CF_obstacle using the same obstacle indices."
        )

    u_faces, v_faces = fluid_face_masks_from_cells(fluid_cells)
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    force_scale = 1.0 / rho

    u, v = apply_fluid_velocity_bc_obstacle(
        u,
        v,
        vel0,
        fluid_cells,
        enforce_outlet_u=True,
        lateral_wall_bc=lateral_wall_bc,
        inlet_v=inlet_v,
    )

    u_center = u[1:ny + 1, 2:nx + 1]
    u_explicit = (
        -(
            (0.5 * (u[1:ny + 1, 3:nx + 2] + u_center)) ** 2
            - (0.5 * (u_center + u[1:ny + 1, 1:nx])) ** 2
        ) * inv_dx
        - (
            0.5 * (u[2:ny + 2, 2:nx + 1] + u_center)
            * 0.5 * (v[2:ny + 2, 1:nx] + v[2:ny + 2, 2:nx + 1])
            - 0.5 * (u_center + u[0:ny, 2:nx + 1])
            * 0.5 * (v[1:ny + 1, 1:nx] + v[1:ny + 1, 2:nx + 1])
        ) * inv_dy
        + nu * (
            (u[1:ny + 1, 1:nx] - 2.0 * u_center + u[1:ny + 1, 3:nx + 2]) * inv_dx2
            + (u[0:ny, 2:nx + 1] - 2.0 * u_center + u[2:ny + 2, 2:nx + 1]) * inv_dy2
        )
    )
    ut[1:ny + 1, 2:nx + 1] = u_center + dt * (u_explicit + Fx[:, 1:-1] * force_scale)

    v_center = v[2:ny + 1, 1:nx + 1]
    v_explicit = (
        - (
            0.5 * (u[2:ny + 1, 2:nx + 2] + u[1:ny, 2:nx + 2])
            * 0.5 * (v[2:ny + 1, 2:nx + 2] + v_center)
            - 0.5 * (u[2:ny + 1, 1:nx + 1] + u[1:ny, 1:nx + 1])
            * 0.5 * (v_center + v[2:ny + 1, 0:nx])
        ) * inv_dx
        - (
            (0.5 * (v[3:ny + 2, 1:nx + 1] + v_center)) ** 2
            - (0.5 * (v_center + v[1:ny, 1:nx + 1])) ** 2
        ) * inv_dy
        + nu * (
            (v[2:ny + 1, 2:nx + 2] - 2.0 * v_center + v[2:ny + 1, 0:nx]) * inv_dx2
            + (v[3:ny + 2, 1:nx + 1] - 2.0 * v_center + v[1:ny, 1:nx + 1]) * inv_dy2
        )
    )
    vt[2:ny + 1, 1:nx + 1] = v_center + dt * (v_explicit + Fy[1:-1, :] * force_scale)

    ut, vt = apply_fluid_velocity_bc_obstacle(
        ut,
        vt,
        vel0,
        fluid_cells,
        enforce_outlet_u=True,
        lateral_wall_bc=lateral_wall_bc,
        inlet_v=inlet_v,
    )

    div = (
        (ut[1:-1, 2:] - ut[1:-1, 1:-1]) * inv_dx
        + (vt[2:, 1:-1] - vt[1:-1, 1:-1]) * inv_dy
    )
    prhs_full = div.ravel(order="F") * dx * dy / dt
    sol = Af.solve(prhs_full[active] + bf)

    p_flat = np.zeros(nx * ny)
    p_flat[active] = sol
    p[1:-1, 1:-1] = p_flat.reshape((ny, nx), order="F")

    if nx > 1:
        u_internal_mask = fluid_cells[:, :-1] & fluid_cells[:, 1:]
        u_corrected = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) * inv_dx
        u[1:-1, 2:-1] = np.where(u_internal_mask, u_corrected, 0.0)

    u_outlet = ut[1:-1, -1] + 2.0 * dt * p[1:-1, -2] * inv_dx
    u[1:-1, -1] = np.where(fluid_cells[:, -1], u_outlet, 0.0)

    if ny > 1:
        v_internal_mask = fluid_cells[:-1, :] & fluid_cells[1:, :]
        v_corrected = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) * inv_dy
        v[2:-1, 1:-1] = np.where(v_internal_mask, v_corrected, 0.0)

    u, v = apply_fluid_velocity_bc_obstacle(
        u,
        v,
        vel0,
        fluid_cells,
        enforce_outlet_u=False,
        lateral_wall_bc=lateral_wall_bc,
        inlet_v=inlet_v,
    )
    u[1:ny + 1, 1:nx + 2][~u_faces] = 0.0
    v[1:ny + 2, 1:nx + 1][~v_faces] = 0.0
    return u, v, p


def body_force_vector(body_force, direction):
    """Normalize scalar or vector body-force input to a 2-entry vector."""
    values = np.asarray(body_force, dtype=float)
    if values.ndim == 0:
        direction = direction.lower()
        if direction == "x":
            return np.array([float(values), 0.0])
        if direction == "y":
            return np.array([0.0, float(values)])
        raise ValueError("body_force_direction must be 'x' or 'y' for scalar body_force.")

    values = values.reshape(-1)
    if values.size != 2:
        raise ValueError("body_force must be a scalar or a 2-entry vector.")
    return values.astype(float, copy=False)


_body_force_vector = body_force_vector


@lru_cache(maxsize=16)
def _fluid_pressure_ff_lu_cached(nx, ny, dx, dy):
    """Pressure Poisson operator for the closed free-fall benchmark channel."""
    nn = nx * ny
    hx = dy / dx
    hy = dx / dy

    i = np.repeat(np.arange(nx, dtype=np.int64), ny)
    j = np.tile(np.arange(ny, dtype=np.int64), nx)
    r = i * ny + j

    main = np.full(nn, -2.0 * hx - 2.0 * hy)
    main[j == 0] += hy
    main[j == ny - 1] += hy
    main[i == 0] += hx
    main[i == nx - 1] += hx

    rows = [r]
    cols = [r]
    data = [main]

    east = i < nx - 1
    rows.append(r[east])
    cols.append(r[east] + ny)
    data.append(np.full(np.count_nonzero(east), hx))

    west = i > 0
    rows.append(r[west])
    cols.append(r[west] - ny)
    data.append(np.full(np.count_nonzero(west), hx))

    north = j < ny - 1
    rows.append(r[north])
    cols.append(r[north] + 1)
    data.append(np.full(np.count_nonzero(north), hy))

    south = j > 0
    rows.append(r[south])
    cols.append(r[south] - 1)
    data.append(np.full(np.count_nonzero(south), hy))

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nn, nn),
    ).tolil()
    A[0, :] = 0.0
    A[0, 0] = 1.0
    return splu(A.tocsc())


@lru_cache(maxsize=16)
def _fluid_pressure_ff_periodic_x_lu_cached(nx, ny, dx, dy):
    """Pressure Poisson operator with periodic x and Neumann top/bottom."""
    nn = nx * ny
    hx = dy / dx
    hy = dx / dy

    i = np.repeat(np.arange(nx, dtype=np.int64), ny)
    j = np.tile(np.arange(ny, dtype=np.int64), nx)
    r = i * ny + j

    main = np.full(nn, -2.0 * hx - 2.0 * hy)
    main[j == 0] += hy
    main[j == ny - 1] += hy

    rows = [r]
    cols = [r]
    data = [main]

    rows.append(r)
    cols.append(((i + 1) % nx) * ny + j)
    data.append(np.full(nn, hx))

    rows.append(r)
    cols.append(((i - 1) % nx) * ny + j)
    data.append(np.full(nn, hx))

    north = j < ny - 1
    rows.append(r[north])
    cols.append(r[north] + 1)
    data.append(np.full(np.count_nonzero(north), hy))

    south = j > 0
    rows.append(r[south])
    cols.append(r[south] - 1)
    data.append(np.full(np.count_nonzero(south), hy))

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nn, nn),
    ).tolil()
    A[0, :] = 0.0
    A[0, 0] = 1.0
    return splu(A.tocsc())


@lru_cache(maxsize=16)
def _fluid_pressure_ff_hesch_lu_cached(nx, ny, dx, dy):
    """Pressure Poisson operator for the Hesch falling-cylinder boundary layout."""
    nn = nx * ny
    hx = dy / dx
    hy = dx / dy

    i = np.repeat(np.arange(nx, dtype=np.int64), ny)
    j = np.tile(np.arange(ny, dtype=np.int64), nx)
    r = i * ny + j

    main = np.full(nn, -2.0 * hx - 2.0 * hy)
    main[j == 0] += hy
    main[j == ny - 1] += hy
    main[i == 0] -= hx
    main[i == nx - 1] += hx

    rows = [r]
    cols = [r]
    data = [main]

    east = i < nx - 1
    rows.append(r[east])
    cols.append(r[east] + ny)
    data.append(np.full(np.count_nonzero(east), hx))

    west = i > 0
    rows.append(r[west])
    cols.append(r[west] - ny)
    data.append(np.full(np.count_nonzero(west), hx))

    north = j < ny - 1
    rows.append(r[north])
    cols.append(r[north] + 1)
    data.append(np.full(np.count_nonzero(north), hy))

    south = j > 0
    rows.append(r[south])
    cols.append(r[south] - 1)
    data.append(np.full(np.count_nonzero(south), hy))

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nn, nn),
    )
    return splu(A.tocsc())


def check_fluid_pressure_solver_ff(Af, periodic_x, hesch_benchmark=False):
    """Validate that a free-fall pressure solver matches requested boundary flags."""
    solver_periodic_x = getattr(Af, "periodic_x", None)
    if solver_periodic_x is not None and bool(solver_periodic_x) != bool(periodic_x):
        raise ValueError(
            "Af was assembled with periodic_x="
            f"{solver_periodic_x}, but fluid_solver_ff was called with "
            f"periodic_x={periodic_x}. Reassemble with matching BCs."
        )
    solver_hesch = getattr(Af, "hesch_benchmark", False)
    if bool(solver_hesch) != bool(hesch_benchmark):
        raise ValueError(
            "Af was assembled with hesch_benchmark="
            f"{solver_hesch}, but fluid_solver_ff was called with "
            f"hesch_benchmark={hesch_benchmark}. Reassemble with matching BCs."
        )


_check_fluid_pressure_solver_ff = check_fluid_pressure_solver_ff


def fluid_assembly_ff(nx, ny, dx, dy, G=0.0, periodic_x=False, hesch_benchmark=False):
    """Assemble the pressure solver used by ``fluid_solver_ff``."""
    if hesch_benchmark and periodic_x:
        raise ValueError("hesch_benchmark=True requires periodic_x=False.")
    if abs(G) > 0.0:
        raise ValueError("fluid_assembly_ff represents a closed free-fall benchmark; use G=0.")

    if hesch_benchmark:
        solver = _fluid_pressure_ff_hesch_lu_cached(int(nx), int(ny), float(dx), float(dy))
        pin_pressure_cell = False
    elif periodic_x:
        solver = _fluid_pressure_ff_periodic_x_lu_cached(int(nx), int(ny), float(dx), float(dy))
        pin_pressure_cell = True
    else:
        solver = _fluid_pressure_ff_lu_cached(int(nx), int(ny), float(dx), float(dy))
        pin_pressure_cell = True
    b = np.zeros(nx * ny)
    return FluidPressureSolverFF(
        solver,
        periodic_x,
        hesch_benchmark=hesch_benchmark,
        pin_pressure_cell=pin_pressure_cell,
    ), b


def apply_fluid_velocity_bc_ff(u, v, periodic_x=False, hesch_benchmark=False):
    """Apply velocity boundary conditions for free-fall benchmark variants."""
    if hesch_benchmark and periodic_x:
        raise ValueError("hesch_benchmark=True requires periodic_x=False.")

    if periodic_x:
        u[1:-1, 0] = u[1:-1, -2]
        u[1:-1, -1] = u[1:-1, 1]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]
    elif hesch_benchmark:
        u[1:-1, 1] = u[1:-1, 2]
        u[1:-1, -1] = 0.0
        v[:, 0] = v[:, 1]
        v[:, -1] = -v[:, -2]
    else:
        u[1:-1, 1] = 0.0
        u[1:-1, -1] = 0.0
        v[:, 0] = -v[:, 1]
        v[:, -1] = -v[:, -2]

    if hesch_benchmark:
        u[-1, :] = u[-2, :]
        u[0, :] = u[1, :]
    else:
        u[-1, :] = -u[-2, :]
        u[0, :] = -u[1, :]

    v[1, :] = 0.0
    v[-1, :] = 0.0
    return u, v


_apply_fluid_velocity_bc_ff = apply_fluid_velocity_bc_ff


def pack_periodic_x_velocity_ff(u, v, U, V, nx, ny):
    """Store unique periodic-x MAC velocities into padded solver arrays."""
    u.fill(0.0)
    v.fill(0.0)

    u[1:ny + 1, 1:nx + 1] = U
    u[1:ny + 1, nx + 1] = U[:, 0]
    u[1:ny + 1, 0] = U[:, -1]
    u[0, :] = -u[1, :]
    u[ny + 1, :] = -u[ny, :]

    v[1:ny + 2, 1:nx + 1] = V
    v[:, 0] = v[:, nx]
    v[:, nx + 1] = v[:, 1]
    v[1, :] = 0.0
    v[ny + 1, :] = 0.0
    return u, v


_pack_periodic_x_velocity_ff = pack_periodic_x_velocity_ff


def unpack_periodic_x_velocity_ff(u, v, nx, ny):
    """Return unique periodic-x MAC velocities from padded solver arrays."""
    U = u[1:ny + 1, 1:nx + 1].copy()
    V = v[1:ny + 2, 1:nx + 1].copy()
    V[0, :] = 0.0
    V[-1, :] = 0.0
    return U, V


_unpack_periodic_x_velocity_ff = unpack_periodic_x_velocity_ff


def fluid_solver_ff_periodic_x(
    u,
    v,
    nx,
    ny,
    dx,
    dy,
    dt,
    nu,
    Fx,
    Fy,
    Af,
    bf,
    vel0,
    rho,
    include_convection=True,
    zero_mean_x=False,
    fluid_body_acceleration=None,
):
    """Free-fall Chorin projection step with periodic x boundaries."""
    check_fluid_pressure_solver_ff(Af, periodic_x=True)
    del vel0

    p = np.zeros_like(u)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    force_scale = 1.0 / rho
    if fluid_body_acceleration is None:
        fluid_body_acceleration = np.zeros(2)
    fluid_body_acceleration = body_force_vector(fluid_body_acceleration, "x")

    U, V = unpack_periodic_x_velocity_ff(u, v, nx, ny)
    u, v = pack_periodic_x_velocity_ff(u, v, U, V, nx, ny)

    U_east = np.roll(U, -1, axis=1)
    U_west = np.roll(U, 1, axis=1)
    U_north = np.vstack((U[1:, :], -U[-1:, :]))
    U_south = np.vstack((-U[:1, :], U[:-1, :]))

    U_explicit = nu * (
        (U_west - 2.0 * U + U_east) * inv_dx2
        + (U_south - 2.0 * U + U_north) * inv_dy2
    )

    if include_convection:
        V_west = np.roll(V, 1, axis=1)
        V_north = 0.5 * (V[1:, :] + V_west[1:, :])
        V_south = 0.5 * (V[:-1, :] + V_west[:-1, :])
        U_explicit = U_explicit - (
            (0.5 * (U_east + U)) ** 2
            - (0.5 * (U + U_west)) ** 2
        ) * inv_dx
        U_explicit = U_explicit - (
            0.5 * (U_north + U) * V_north
            - 0.5 * (U + U_south) * V_south
        ) * inv_dy

    Fx_unique = Fx[:, :nx].copy()
    Fx_unique[:, 0] += Fx[:, nx]
    Ut = U + dt * (U_explicit + Fx_unique * force_scale + fluid_body_acceleration[0])

    Q = V[1:ny, :]
    Q_east = np.roll(Q, -1, axis=1)
    Q_west = np.roll(Q, 1, axis=1)
    Q_north = V[2:ny + 1, :]
    Q_south = V[0:ny - 1, :]

    V_explicit = nu * (
        (Q_east - 2.0 * Q + Q_west) * inv_dx2
        + (Q_north - 2.0 * Q + Q_south) * inv_dy2
    )

    if include_convection:
        U_upper = U[1:ny, :]
        U_lower = U[0:ny - 1, :]
        U_east_mid = 0.5 * (np.roll(U_upper, -1, axis=1) + np.roll(U_lower, -1, axis=1))
        U_west_mid = 0.5 * (U_upper + U_lower)
        V_east = 0.5 * (np.roll(Q, -1, axis=1) + Q)
        V_west = 0.5 * (Q + np.roll(Q, 1, axis=1))
        V_north = 0.5 * (Q_north + Q)
        V_south = 0.5 * (Q + Q_south)
        V_explicit = V_explicit - (U_east_mid * V_east - U_west_mid * V_west) * inv_dx
        V_explicit = V_explicit - (V_north**2 - V_south**2) * inv_dy

    Vt = V.copy()
    Vt[1:ny, :] = Q + dt * (V_explicit + Fy[1:ny, :] * force_scale + fluid_body_acceleration[1])
    Vt[0, :] = 0.0
    Vt[-1, :] = 0.0

    prhs = (
        ((np.roll(Ut, -1, axis=1) - Ut) * inv_dx + (Vt[1:, :] - Vt[:-1, :]) * inv_dy)
        .ravel("F")
        * dx
        * dy
        / dt
    )
    prhs[0] = 0.0

    sol = Af.solve(prhs + bf)
    P = sol.reshape((ny, nx), order="F")

    U_new = Ut - dt * (P - np.roll(P, 1, axis=1)) * inv_dx
    if zero_mean_x:
        U_new = U_new - np.mean(U_new)

    V_new = Vt.copy()
    V_new[1:ny, :] = Vt[1:ny, :] - dt * (P[1:, :] - P[:-1, :]) * inv_dy
    V_new[0, :] = 0.0
    V_new[-1, :] = 0.0

    p[1:ny + 1, 1:nx + 1] = P
    p[1:ny + 1, 0] = P[:, -1]
    p[1:ny + 1, nx + 1] = P[:, 0]
    p[0, :] = p[1, :]
    p[ny + 1, :] = p[ny, :]

    u, v = pack_periodic_x_velocity_ff(u, v, U_new, V_new, nx, ny)
    return u, v, p


_fluid_solver_ff_periodic_x = fluid_solver_ff_periodic_x


def fluid_solver_ff(
    u,
    v,
    nx,
    ny,
    dx,
    dy,
    dt,
    nu,
    Fx,
    Fy,
    Af,
    bf,
    vel0,
    rho,
    include_convection=True,
    periodic_x=False,
    zero_mean_periodic_x=False,
    hesch_benchmark=False,
    fluid_body_acceleration=None,
):
    """Fast Chorin projection step with free-fall benchmark boundary conditions."""
    if hesch_benchmark and periodic_x:
        raise ValueError("hesch_benchmark=True requires periodic_x=False.")
    if zero_mean_periodic_x and not periodic_x:
        raise ValueError("zero_mean_periodic_x requires periodic_x=True.")

    check_fluid_pressure_solver_ff(Af, periodic_x, hesch_benchmark=hesch_benchmark)

    if periodic_x:
        return fluid_solver_ff_periodic_x(
            u,
            v,
            nx,
            ny,
            dx,
            dy,
            dt,
            nu,
            Fx,
            Fy,
            Af,
            bf,
            vel0,
            rho,
            include_convection=include_convection,
            zero_mean_x=zero_mean_periodic_x,
            fluid_body_acceleration=fluid_body_acceleration,
        )

    del vel0
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    force_scale = 1.0 / rho
    if fluid_body_acceleration is None:
        fluid_body_acceleration = np.zeros(2)
    fluid_body_acceleration = body_force_vector(fluid_body_acceleration, "x")

    u, v = apply_fluid_velocity_bc_ff(u, v, hesch_benchmark=hesch_benchmark)

    u_center = u[1:ny + 1, 2:nx + 1]
    u_diffusion = nu * (
        (u[1:ny + 1, 1:nx] - 2.0 * u_center + u[1:ny + 1, 3:nx + 2]) * inv_dx2
        + (u[0:ny, 2:nx + 1] - 2.0 * u_center + u[2:ny + 2, 2:nx + 1]) * inv_dy2
    )
    u_explicit = u_diffusion
    if include_convection:
        u_explicit = u_explicit - (
            (0.5 * (u[1:ny + 1, 3:nx + 2] + u_center)) ** 2
            - (0.5 * (u_center + u[1:ny + 1, 1:nx])) ** 2
        ) * inv_dx
        u_explicit = u_explicit - (
            0.5 * (u[2:ny + 2, 2:nx + 1] + u_center)
            * 0.5 * (v[2:ny + 2, 1:nx] + v[2:ny + 2, 2:nx + 1])
            - 0.5 * (u_center + u[0:ny, 2:nx + 1])
            * 0.5 * (v[1:ny + 1, 1:nx] + v[1:ny + 1, 2:nx + 1])
        ) * inv_dy
    ut[1:ny + 1, 2:nx + 1] = (
        u_center
        + dt * (u_explicit + Fx[:, 1:-1] * force_scale + fluid_body_acceleration[0])
    )

    v_center = v[2:ny + 1, 1:nx + 1]
    v_diffusion = nu * (
        (v[2:ny + 1, 2:nx + 2] - 2.0 * v_center + v[2:ny + 1, 0:nx]) * inv_dx2
        + (v[3:ny + 2, 1:nx + 1] - 2.0 * v_center + v[1:ny, 1:nx + 1]) * inv_dy2
    )
    v_explicit = v_diffusion
    if include_convection:
        v_explicit = v_explicit - (
            0.5 * (u[2:ny + 1, 2:nx + 2] + u[1:ny, 2:nx + 2])
            * 0.5 * (v[2:ny + 1, 2:nx + 2] + v_center)
            - 0.5 * (u[2:ny + 1, 1:nx + 1] + u[1:ny, 1:nx + 1])
            * 0.5 * (v_center + v[2:ny + 1, 0:nx])
        ) * inv_dx
        v_explicit = v_explicit - (
            (0.5 * (v[3:ny + 2, 1:nx + 1] + v_center)) ** 2
            - (0.5 * (v_center + v[1:ny, 1:nx + 1])) ** 2
        ) * inv_dy
    vt[2:ny + 1, 1:nx + 1] = (
        v_center
        + dt * (v_explicit + Fy[1:-1, :] * force_scale + fluid_body_acceleration[1])
    )

    ut, vt = apply_fluid_velocity_bc_ff(ut, vt, hesch_benchmark=hesch_benchmark)
    prhs = (
        ((ut[1:-1, 2:] - ut[1:-1, 1:-1]) * inv_dx + (vt[2:, 1:-1] - vt[1:-1, 1:-1]) * inv_dy)
        .ravel("F")
        * dx
        * dy
        / dt
    )
    if getattr(Af, "pin_pressure_cell", True):
        prhs[0] = 0.0

    sol = Af.solve(prhs + bf)
    p[1:-1, 1:-1] = sol.reshape((ny, nx), order="F")

    u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2]) * inv_dx
    v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1]) * inv_dy

    if hesch_benchmark:
        p[1:-1, 0] = -p[1:-1, 1]
        p[1:-1, -1] = p[1:-1, -2]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

    u, v = apply_fluid_velocity_bc_ff(u, v, hesch_benchmark=hesch_benchmark)
    return u, v, p


def fluid_solver_ff_stokes(
    u,
    v,
    nx,
    ny,
    dx,
    dy,
    dt,
    nu,
    Fx,
    Fy,
    Af,
    bf,
    vel0,
    rho,
    periodic_x=False,
    zero_mean_periodic_x=False,
    hesch_benchmark=False,
    fluid_body_acceleration=None,
):
    """Unsteady Stokes projection step for the free-fall benchmark."""
    return fluid_solver_ff(
        u,
        v,
        nx,
        ny,
        dx,
        dy,
        dt,
        nu,
        Fx,
        Fy,
        Af,
        bf,
        vel0,
        rho,
        include_convection=False,
        periodic_x=periodic_x,
        zero_mean_periodic_x=zero_mean_periodic_x,
        hesch_benchmark=hesch_benchmark,
        fluid_body_acceleration=fluid_body_acceleration,
    )


__all__ = [
    "FastPoissonDCT",
    "SparsePoissonCG",
    "FluidPressureSolverFF",
    "_FluidPressureSolverFF",
    "build_fluid_grid",
    "poiseuille_profile",
    "turek_inlet_profile",
    "turek_inlet_ramp",
    "inlet_scale",
    "smooth_inlet_ramp",
    "fluid_response_inlet_v_step",
    "initialize_fluid_from_inlet",
    "as_state_array",
    "initial_fluid_fields",
    "sample_regular_grid",
    "physical_mac_fields",
    "sample_mac_velocity",
    "resample_reference_velocity",
    "pack_mac_fields",
    "pack_mac_fields_with_ghosts",
    "discrete_divergence",
    "periodic_x_divergence_ff",
    "cell_centered_velocity",
    "cell_centered_vorticity",
    "cell_centered_fields",
    "fluid_cell_x",
    "fluid_cell_y",
    "fluid_assembly_CF_debug",
    "fluid_assembly_CF",
    "TractionFreePressure",
    "fluid_assembly_traction_free",
    "apply_traction_free_velocity_bc",
    "apply_fluid_velocity_bc",
    "_apply_fluid_velocity_bc",
    "fluid_solver_CF_debug",
    "fluid_solver_CF",
    "obstacle_cell_mask",
    "_obstacle_cell_mask",
    "fluid_face_masks_from_cells",
    "_fluid_face_masks_from_cells",
    "apply_obstacle_tangential_no_slip",
    "_apply_obstacle_tangential_no_slip",
    "active_pressure_cells_fortran",
    "_active_pressure_cells_fortran",
    "fluid_assembly_CF_obstacle",
    "apply_fluid_velocity_bc_obstacle",
    "_apply_fluid_velocity_bc_obstacle",
    "fluid_solver_CF_obstacle",
    "body_force_vector",
    "_body_force_vector",
    "check_fluid_pressure_solver_ff",
    "_check_fluid_pressure_solver_ff",
    "fluid_assembly_ff",
    "apply_fluid_velocity_bc_ff",
    "_apply_fluid_velocity_bc_ff",
    "pack_periodic_x_velocity_ff",
    "_pack_periodic_x_velocity_ff",
    "unpack_periodic_x_velocity_ff",
    "_unpack_periodic_x_velocity_ff",
    "fluid_solver_ff_periodic_x",
    "_fluid_solver_ff_periodic_x",
    "fluid_solver_ff",
    "fluid_solver_ff_stokes",
]
