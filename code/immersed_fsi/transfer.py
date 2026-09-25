"""Transfer operators for the cleaned immersed FSI codebase.

The module intentionally keeps only the four transfer families currently used
in the cleaned cases:

* ``ID``: area transfer over the whole solid mesh.
* ``IB``: line transfer over the solid boundary.
* ``MIX_noC``: area plus boundary transfer.
* ``MIX``: area plus boundary transfer, enriched with corner nodes.

Each family is available for P1 triangular meshes and Q1 quadrilateral meshes.
The assembled operators are row-normalized so constant MAC velocity fields are
preserved on every solid node that receives quadrature support.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss

from .geometry import (
    corner_nodes_from_bdry,
    dirichlet_nodes_from_bdry,
    point_matrix,
    quadrilateral_connectivity,
    triangle_connectivity,
)

__all__ = [
    "compute_Jacobian",
    "compute_quad",
    "compute_quad_1d",
    "normalize_csr_rows",
    "phi",
    "phi_1d",
    "phi_x",
    "phi_y",
    "assembly_T_ID",
    "assembly_T_ID_quad",
    "assembly_T_ID_tri",
    "assembly_T_IB",
    "assembly_T_IB_quad",
    "assembly_T_IB_tri",
    "assembly_T_MIX_quad",
    "assembly_T_MIX_square",
    "assembly_T_MIX_tri",
    "assembly_T_MIX_noC_quad",
    "assembly_T_MIX_noC_tri",
    "assembly_T_mix",
]


def _as_coordinate_vectors(Pos_x, Pos_y):
    """Return solid coordinates as two flat float arrays with matching shape."""
    Pos_x = np.asarray(Pos_x, dtype=float).reshape(-1)
    Pos_y = np.asarray(Pos_y, dtype=float).reshape(-1)
    if Pos_x.shape != Pos_y.shape:
        raise ValueError("Pos_x and Pos_y must have the same shape.")
    return Pos_x, Pos_y


def _check_fluid_grid(dx_f, dy_f, nx_f, ny_f):
    """Validate MAC grid parameters and return integer grid sizes."""
    if dx_f <= 0.0 or dy_f <= 0.0:
        raise ValueError("dx_f and dy_f must be positive.")
    nx_f = int(nx_f)
    ny_f = int(ny_f)
    if nx_f <= 0 or ny_f <= 0:
        raise ValueError("nx_f and ny_f must be positive.")
    return nx_f, ny_f


def _validate_nodes(nodes, Nb, name):
    """Check that a connectivity array references existing solid nodes."""
    if nodes.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    if np.any(nodes < 0) or np.any(nodes >= Nb):
        raise ValueError(f"{name} contains node indices outside the valid range.")


def _empty_transfer_pair(Nb, nx_f, ny_f):
    """Return empty x/y MAC transfer matrices with the requested shapes."""
    Tx = sp.coo_matrix((Nb, ny_f * (nx_f + 1)), dtype=float)
    Ty = sp.coo_matrix((Nb, (ny_f + 1) * nx_f), dtype=float)
    return Tx, Ty


def _mac_x_columns(x, y, dx_f, dy_f, nx_f, ny_f):
    """Return MAC-column indices for the x-velocity component."""
    ix = np.floor((x + 0.5 * dx_f) / dx_f).astype(np.int64)
    iy = np.floor(y / dy_f).astype(np.int64)
    ix = np.clip(ix, 0, nx_f)
    iy = np.clip(iy, 0, ny_f - 1)
    return iy * (nx_f + 1) + ix


def _mac_y_columns(x, y, dx_f, dy_f, nx_f, ny_f):
    """Return MAC-column indices for the y-velocity component."""
    ix = np.floor(x / dx_f).astype(np.int64)
    iy = np.floor((y + 0.5 * dy_f) / dy_f).astype(np.int64)
    ix = np.clip(ix, 0, nx_f - 1)
    iy = np.clip(iy, 0, ny_f)
    return iy * nx_f + ix


def normalize_csr_rows(matrix, return_d_inv=False, tol=1.0e-14):
    """Return a CSR matrix with every nonzero row divided by its row sum."""
    matrix = matrix.tocsr(copy=True)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    inv_row_sums = np.zeros_like(row_sums)
    nonzero_rows = np.abs(row_sums) > tol
    inv_row_sums[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    matrix.data *= np.repeat(inv_row_sums, np.diff(matrix.indptr))
    if return_d_inv:
        return matrix, inv_row_sums
    return matrix


_normalize_csr_rows = normalize_csr_rows


def _normalize_transfer_pair(Tx, Ty, return_d_inv=False):
    """Normalize an x/y transfer pair and optionally return the averaged inverse."""
    if return_d_inv:
        Tx, d_inv_x = normalize_csr_rows(Tx, return_d_inv=True)
        Ty, d_inv_y = normalize_csr_rows(Ty, return_d_inv=True)
        return Tx, Ty, 0.5 * (d_inv_x + d_inv_y)
    return normalize_csr_rows(Tx), normalize_csr_rows(Ty)


@lru_cache(maxsize=8)
def _gauss_1d(n):
    """Return Gauss-Legendre nodes and weights mapped to [0, 1]."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    nodes, weights = leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def compute_quad(n):
    """Compute tensor-product Gauss quadrature on the unit square."""
    nodes_1d, weights_1d = _gauss_1d(n)
    nodes_2d = [(float(x), float(y)) for x in nodes_1d for y in nodes_1d]
    weights_2d = [float(wx * wy) for wx in weights_1d for wy in weights_1d]
    return nodes_2d, weights_2d


def phi(i, x, y):
    """Return the local Q1 basis function ``i`` on the unit square."""
    if i == 0:
        return (1.0 - x) * (1.0 - y)
    if i == 1:
        return x * (1.0 - y)
    if i == 2:
        return x * y
    if i == 3:
        return y * (1.0 - x)
    raise ValueError("i must be 0, 1, 2, or 3.")


def phi_x(x, y):
    """Return derivatives of the four Q1 basis functions with respect to x."""
    return [y - 1.0, 1.0 - y, y, -y]


def phi_y(x, y):
    """Return derivatives of the four Q1 basis functions with respect to y."""
    return [x - 1.0, -x, x, 1.0 - x]


def compute_Jacobian(e, T, Pos, xq, yq):
    """Return the absolute Jacobian determinant of Q1 element ``e``."""
    Pos = point_matrix(Pos)
    T = quadrilateral_connectivity(T)
    e = int(e)
    if e < 0 or e >= T.shape[0]:
        raise IndexError("e is outside the element range.")

    nodes = T[e]
    phix = phi_x(xq, yq)
    phiy = phi_y(xq, yq)

    x_x = sum(phix[k] * Pos[0, nodes[k]] for k in range(4))
    x_y = sum(phiy[k] * Pos[0, nodes[k]] for k in range(4))
    y_x = sum(phix[k] * Pos[1, nodes[k]] for k in range(4))
    y_y = sum(phiy[k] * Pos[1, nodes[k]] for k in range(4))
    return abs(x_x * y_y - x_y * y_x)


@lru_cache(maxsize=8)
def _id_quadrature_data(n):
    """Return Q1 basis data at tensor-product Gauss points."""
    nodes_1d, weights_1d = _gauss_1d(n)
    xq, yq = np.meshgrid(nodes_1d, nodes_1d, indexing="ij")
    wx, wy = np.meshgrid(weights_1d, weights_1d, indexing="ij")

    xq = xq.ravel()
    yq = yq.ravel()
    quad_phi = np.column_stack(
        (
            (1.0 - xq) * (1.0 - yq),
            xq * (1.0 - yq),
            xq * yq,
            yq * (1.0 - xq),
        )
    )
    quad_phi_x = np.column_stack((yq - 1.0, 1.0 - yq, yq, -yq))
    quad_phi_y = np.column_stack((xq - 1.0, -xq, xq, 1.0 - xq))
    return quad_phi, quad_phi_x, quad_phi_y, (wx * wy).ravel()


@lru_cache(maxsize=8)
def _tri_quadrature_data_uniform(n=24):
    """Composite centroid quadrature on the reference P1 triangle."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")

    xi = []
    eta = []
    for i in range(n):
        for j in range(n - i):
            xi.append((3 * i + 1) / (3 * n))
            eta.append((3 * j + 1) / (3 * n))
    for i in range(n - 1):
        for j in range(n - 1 - i):
            xi.append((3 * i + 2) / (3 * n))
            eta.append((3 * j + 2) / (3 * n))

    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    quad_phi = np.column_stack((1.0 - xi - eta, xi, eta))
    quad_weights = np.full(n * n, 0.5 / (n * n), dtype=float)
    return quad_phi, quad_weights


def phi_1d(k, x):
    """Return the local P1 basis function ``k`` on the unit segment."""
    if k == 0:
        return 1.0 - x
    if k == 1:
        return x
    raise ValueError("k must be 0 or 1.")


def compute_quad_1d(n):
    """Compute Gauss quadrature nodes and weights on the unit segment."""
    nodes, weights = _gauss_1d(n)
    return [float(x) for x in nodes], [float(w) for w in weights]


@lru_cache(maxsize=8)
def _line_quadrature_data(n):
    """Return P1 segment basis data at Gauss points."""
    nodes, weights = _gauss_1d(n)
    quad_phi = np.column_stack((1.0 - nodes, nodes))
    return nodes, weights, quad_phi


def _boundary_segments(bdry, Nb):
    """Return boundary segments as ``(n_edges, 2)`` node indices."""
    bdry = np.asarray(bdry, dtype=int)
    if bdry.ndim != 2 or bdry.shape[0] < 4:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")
    if bdry.shape[1] == 0:
        return np.empty((0, 2), dtype=np.int64)
    nodes = bdry[2:4, :].T.astype(np.int64, copy=False)
    if np.any(nodes < 0) or np.any(nodes >= Nb):
        raise ValueError("bdry contains node indices outside the valid range.")
    return nodes


def _assembly_T_ID_quad_raw(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    quadrature_order=24,
):
    """Assemble unnormalized ID transfer matrices for Q1 quadrilaterals."""
    del nx_s, ny_s
    nx_f, ny_f = _check_fluid_grid(dx_f, dy_f, nx_f, ny_f)
    quad_phi, quad_phi_x, quad_phi_y, quad_weights = _id_quadrature_data(
        quadrature_order
    )

    Pos_x, Pos_y = _as_coordinate_vectors(Pos_x, Pos_y)
    Nb = Pos_x.size
    nodes = quadrilateral_connectivity(T).astype(np.int64, copy=False)
    _validate_nodes(nodes, Nb, "T")

    elem_x = Pos_x[nodes]
    elem_y = Pos_y[nodes]
    x_phys = elem_x @ quad_phi.T
    y_phys = elem_y @ quad_phi.T

    jx = _mac_x_columns(x_phys, y_phys, dx_f, dy_f, nx_f, ny_f)
    jy = _mac_y_columns(x_phys, y_phys, dx_f, dy_f, nx_f, ny_f)

    x_x = elem_x @ quad_phi_x.T
    x_y = elem_x @ quad_phi_y.T
    y_x = elem_y @ quad_phi_x.T
    y_y = elem_y @ quad_phi_y.T
    det_j = np.abs(x_x * y_y - x_y * y_x)
    if np.any(det_j <= 0.0):
        raise ValueError("T contains degenerate quadrilateral elements.")

    values = (
        det_j[:, :, np.newaxis]
        * quad_weights[np.newaxis, :, np.newaxis]
        * quad_phi[np.newaxis, :, :]
    )
    rows = np.broadcast_to(nodes[:, np.newaxis, :], values.shape).ravel()
    cols_x = np.broadcast_to(jx[:, :, np.newaxis], values.shape).ravel()
    cols_y = np.broadcast_to(jy[:, :, np.newaxis], values.shape).ravel()
    data = values.ravel()

    Tx = sp.coo_matrix((data, (rows, cols_x)), shape=(Nb, ny_f * (nx_f + 1)))
    Ty = sp.coo_matrix((data, (rows, cols_y)), shape=(Nb, (ny_f + 1) * nx_f))
    return Tx, Ty


def _assembly_T_ID_tri_raw(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    quadrature_order=24,
    quadrature_data=None,
):
    """Assemble unnormalized ID transfer matrices for P1 triangles."""
    del nx_s, ny_s
    nx_f, ny_f = _check_fluid_grid(dx_f, dy_f, nx_f, ny_f)
    if quadrature_data is None:
        quad_phi, quad_weights = _tri_quadrature_data_uniform(quadrature_order)
    else:
        quad_phi, quad_weights = quadrature_data

    quad_phi = np.asarray(quad_phi, dtype=float)
    quad_weights = np.asarray(quad_weights, dtype=float).reshape(-1)
    if quad_phi.ndim != 2 or quad_phi.shape[1] != 3:
        raise ValueError("quadrature_data[0] must have shape (n_quad, 3).")
    if quad_weights.size != quad_phi.shape[0]:
        raise ValueError("quadrature_data weights must match quadrature points.")

    Pos_x, Pos_y = _as_coordinate_vectors(Pos_x, Pos_y)
    Nb = Pos_x.size
    nodes = triangle_connectivity(T).astype(np.int64, copy=False)
    _validate_nodes(nodes, Nb, "T")

    elem_x = Pos_x[nodes]
    elem_y = Pos_y[nodes]
    x_phys = elem_x @ quad_phi.T
    y_phys = elem_y @ quad_phi.T

    jx = _mac_x_columns(x_phys, y_phys, dx_f, dy_f, nx_f, ny_f)
    jy = _mac_y_columns(x_phys, y_phys, dx_f, dy_f, nx_f, ny_f)

    det_j = np.abs(
        (elem_x[:, 1] - elem_x[:, 0]) * (elem_y[:, 2] - elem_y[:, 0])
        - (elem_x[:, 2] - elem_x[:, 0]) * (elem_y[:, 1] - elem_y[:, 0])
    )
    if np.any(det_j <= 0.0):
        raise ValueError("T contains degenerate triangular elements.")

    values = (
        det_j[:, np.newaxis, np.newaxis]
        * quad_weights[np.newaxis, :, np.newaxis]
        * quad_phi[np.newaxis, :, :]
    )
    rows = np.broadcast_to(nodes[:, np.newaxis, :], values.shape).ravel()
    cols_x = np.broadcast_to(jx[:, :, np.newaxis], values.shape).ravel()
    cols_y = np.broadcast_to(jy[:, :, np.newaxis], values.shape).ravel()
    data = values.ravel()

    Tx = sp.coo_matrix((data, (rows, cols_x)), shape=(Nb, ny_f * (nx_f + 1)))
    Ty = sp.coo_matrix((data, (rows, cols_y)), shape=(Nb, (ny_f + 1) * nx_f))
    return Tx, Ty


def _assembly_T_IB_raw(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    bdry,
    quadrature_order=24,
):
    """Assemble unnormalized IB transfer matrices from boundary segments."""
    del nx_s, ny_s
    nx_f, ny_f = _check_fluid_grid(dx_f, dy_f, nx_f, ny_f)
    _, quad_weights, quad_phi = _line_quadrature_data(quadrature_order)

    Pos_x, Pos_y = _as_coordinate_vectors(Pos_x, Pos_y)
    Nb = Pos_x.size
    nodes = _boundary_segments(bdry, Nb)
    if nodes.size == 0:
        return _empty_transfer_pair(Nb, nx_f, ny_f)

    x0 = Pos_x[nodes[:, 0]]
    y0 = Pos_y[nodes[:, 0]]
    x1 = Pos_x[nodes[:, 1]]
    y1 = Pos_y[nodes[:, 1]]
    length = np.hypot(x1 - x0, y1 - y0)
    if np.any(length <= 0.0):
        raise ValueError("bdry contains degenerate boundary segments.")

    xq = (
        quad_phi[np.newaxis, :, 0] * x0[:, np.newaxis]
        + quad_phi[np.newaxis, :, 1] * x1[:, np.newaxis]
    )
    yq = (
        quad_phi[np.newaxis, :, 0] * y0[:, np.newaxis]
        + quad_phi[np.newaxis, :, 1] * y1[:, np.newaxis]
    )

    jx = _mac_x_columns(xq, yq, dx_f, dy_f, nx_f, ny_f)
    jy = _mac_y_columns(xq, yq, dx_f, dy_f, nx_f, ny_f)

    values = (
        length[:, np.newaxis, np.newaxis]
        * quad_weights[np.newaxis, :, np.newaxis]
        * quad_phi[np.newaxis, :, :]
    )
    rows = np.broadcast_to(nodes[:, np.newaxis, :], values.shape).ravel()
    cols_x = np.broadcast_to(jx[:, :, np.newaxis], values.shape).ravel()
    cols_y = np.broadcast_to(jy[:, :, np.newaxis], values.shape).ravel()
    data = values.ravel()

    Tx = sp.coo_matrix((data, (rows, cols_x)), shape=(Nb, ny_f * (nx_f + 1)))
    Ty = sp.coo_matrix((data, (rows, cols_y)), shape=(Nb, (ny_f + 1) * nx_f))
    return Tx, Ty


def _assembly_T_CORN_raw(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    bdry,
    dirichlet_flag=-1,
):
    """Assemble unnormalized point transfer matrices for geometric corners."""
    del nx_s, ny_s
    nx_f, ny_f = _check_fluid_grid(dx_f, dy_f, nx_f, ny_f)
    Pos_x, Pos_y = _as_coordinate_vectors(Pos_x, Pos_y)
    Nb = Pos_x.size

    bdry = np.asarray(bdry, dtype=int)
    if bdry.ndim != 2 or bdry.shape[0] < 5:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")

    corner_nodes = corner_nodes_from_bdry(bdry)
    if corner_nodes.size and dirichlet_flag is not None:
        dirichlet_nodes = dirichlet_nodes_from_bdry(bdry, flag=dirichlet_flag)
        corner_nodes = np.setdiff1d(
            corner_nodes, dirichlet_nodes, assume_unique=False
        )
    if corner_nodes.size == 0:
        return _empty_transfer_pair(Nb, nx_f, ny_f)
    if np.any(corner_nodes < 0) or np.any(corner_nodes >= Nb):
        raise ValueError("bdry contains corner nodes outside the valid range.")

    x = Pos_x[corner_nodes]
    y = Pos_y[corner_nodes]
    jx = _mac_x_columns(x, y, dx_f, dy_f, nx_f, ny_f)
    jy = _mac_y_columns(x, y, dx_f, dy_f, nx_f, ny_f)

    data = np.ones(corner_nodes.size, dtype=float)
    Tx = sp.coo_matrix((data, (corner_nodes, jx)), shape=(Nb, ny_f * (nx_f + 1)))
    Ty = sp.coo_matrix((data, (corner_nodes, jy)), shape=(Nb, (ny_f + 1) * nx_f))
    return Tx, Ty


def assembly_T_ID_quad(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    return_d_inv=False,
    quadrature_order=24,
):
    """Assemble normalized ID transfer matrices for a Q1 quadrilateral mesh."""
    Tx, Ty = _assembly_T_ID_quad_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        quadrature_order=quadrature_order,
    )
    return _normalize_transfer_pair(Tx, Ty, return_d_inv=return_d_inv)


def assembly_T_ID_tri(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    return_d_inv=False,
    quadrature_order=24,
    quadrature_data=None,
):
    """Assemble normalized ID transfer matrices for a P1 triangular mesh."""
    Tx, Ty = _assembly_T_ID_tri_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        quadrature_order=quadrature_order,
        quadrature_data=quadrature_data,
    )
    return _normalize_transfer_pair(Tx, Ty, return_d_inv=return_d_inv)


def assembly_T_IB(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    bdry,
    return_d_inv=False,
    quadrature_order=24,
):
    """Assemble normalized IB transfer matrices from boundary segments."""
    Tx, Ty = _assembly_T_IB_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        quadrature_order=quadrature_order,
    )
    return _normalize_transfer_pair(Tx, Ty, return_d_inv=return_d_inv)


def assembly_T_IB_quad(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    bdry,
    return_d_inv=False,
    quadrature_order=24,
):
    """Assemble normalized IB transfer matrices for a quadrilateral boundary."""
    return assembly_T_IB(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        return_d_inv=return_d_inv,
        quadrature_order=quadrature_order,
    )


def assembly_T_IB_tri(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    bdry,
    return_d_inv=False,
    quadrature_order=24,
):
    """Assemble normalized IB transfer matrices for a triangular boundary."""
    return assembly_T_IB(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        return_d_inv=return_d_inv,
        quadrature_order=quadrature_order,
    )


def assembly_T_MIX_noC_quad(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    bdry,
    return_d_inv=False,
    id_quadrature_order=24,
    ib_quadrature_order=24,
):
    """Assemble normalized Q1 MIX_noC transfer matrices: ID plus IB."""
    Tx_id, Ty_id = _assembly_T_ID_quad_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        quadrature_order=id_quadrature_order,
    )
    Tx_ib, Ty_ib = _assembly_T_IB_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        quadrature_order=ib_quadrature_order,
    )
    return _normalize_transfer_pair(
        Tx_id + Tx_ib, Ty_id + Ty_ib, return_d_inv=return_d_inv
    )


def assembly_T_MIX_noC_tri(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    bdry,
    return_d_inv=False,
    id_quadrature_order=24,
    ib_quadrature_order=24,
    quadrature_data=None,
):
    """Assemble normalized P1 MIX_noC transfer matrices: ID plus IB."""
    Tx_id, Ty_id = _assembly_T_ID_tri_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        quadrature_order=id_quadrature_order,
        quadrature_data=quadrature_data,
    )
    Tx_ib, Ty_ib = _assembly_T_IB_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        quadrature_order=ib_quadrature_order,
    )
    return _normalize_transfer_pair(
        Tx_id + Tx_ib, Ty_id + Ty_ib, return_d_inv=return_d_inv
    )


def assembly_T_MIX_quad(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    bdry,
    return_d_inv=False,
    id_quadrature_order=24,
    ib_quadrature_order=24,
    dirichlet_flag=-1,
):
    """Assemble normalized Q1 MIX transfer matrices: ID, IB, and corners."""
    Tx_id, Ty_id = _assembly_T_ID_quad_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        quadrature_order=id_quadrature_order,
    )
    Tx_ib, Ty_ib = _assembly_T_IB_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        quadrature_order=ib_quadrature_order,
    )
    Tx_c, Ty_c = _assembly_T_CORN_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        dirichlet_flag=dirichlet_flag,
    )
    return _normalize_transfer_pair(
        Tx_id + Tx_ib + Tx_c,
        Ty_id + Ty_ib + Ty_c,
        return_d_inv=return_d_inv,
    )


def assembly_T_MIX_tri(
    Pos_x,
    Pos_y,
    dx_f,
    dy_f,
    nx_s,
    ny_s,
    nx_f,
    ny_f,
    T,
    bdry,
    return_d_inv=False,
    id_quadrature_order=24,
    ib_quadrature_order=24,
    quadrature_data=None,
    dirichlet_flag=-1,
):
    """Assemble normalized P1 MIX transfer matrices: ID, IB, and corners."""
    Tx_id, Ty_id = _assembly_T_ID_tri_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        T,
        quadrature_order=id_quadrature_order,
        quadrature_data=quadrature_data,
    )
    Tx_ib, Ty_ib = _assembly_T_IB_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        quadrature_order=ib_quadrature_order,
    )
    Tx_c, Ty_c = _assembly_T_CORN_raw(
        Pos_x,
        Pos_y,
        dx_f,
        dy_f,
        nx_s,
        ny_s,
        nx_f,
        ny_f,
        bdry,
        dirichlet_flag=dirichlet_flag,
    )
    return _normalize_transfer_pair(
        Tx_id + Tx_ib + Tx_c,
        Ty_id + Ty_ib + Ty_c,
        return_d_inv=return_d_inv,
    )


# Compatibility aliases for the names already used by older case scripts.
assembly_T_ID = assembly_T_ID_quad
assembly_T_mix = assembly_T_MIX_noC_quad
assembly_T_MIX_square = assembly_T_MIX_quad
