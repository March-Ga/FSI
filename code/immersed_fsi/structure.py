"""Structural mechanics routines for immersed FSI benchmarks.

This module contains finite-element assembly and standalone structural helper
code extracted from the older project. It supports Q1 bilinear elasticity on
structured and general quadrilateral meshes, P1 linear elasticity on triangular
meshes, heterogeneous triangular assemblies such as the Turek-Hron solid, and
small structural utilities such as vector mass matrices, homogeneous Dirichlet
rows, Rayleigh damping, and mass norms. It intentionally does not include
fluid solvers, transfer operators, or FSI coupling loops.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss
from scipy.sparse import bmat, coo_matrix, lil_matrix

from .geometry import (
    TurekSolidMesh,
    dirichlet_nodes_from_bdry,
    point_matrix,
    quadrilateral_connectivity,
    structured_rectangle_mesh,
    triangle_connectivity,
    triangular_boundary_description,
)


def zero_body_force(x, y):
    """Return a zero force field with the same shape as ``x``."""
    return np.zeros_like(x, dtype=float)


def as_nodal_values(values, size, name):
    """Return scalar or array-like data as one value per mesh node."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(size, float(values), dtype=float)
    values = values.reshape(-1)
    if values.size == 1:
        return np.full(size, float(values[0]), dtype=float)
    if values.size != size:
        raise ValueError(f"{name} must return a scalar or {size} nodal values.")
    return values


def as_element_values(values, n_elem, name, positive=False):
    """Return scalar or array-like data as one value per element."""
    if values is None:
        raise ValueError(f"{name} must be provided.")

    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(n_elem, float(values), dtype=float)
    else:
        values = values.reshape(-1)
        if values.size == 1:
            values = np.full(n_elem, float(values[0]), dtype=float)
        elif values.size != n_elem:
            raise ValueError(f"{name} must be a scalar or have one value per element.")

    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must contain positive values.")
    return values


_as_nodal_values = as_nodal_values
_as_element_values = as_element_values


def vector_mass_matrix(M):
    """Return the block mass matrix for two displacement components."""
    return sp.kron(sp.eye(2, format="csr"), M, format="csr")


def apply_homogeneous_dirichlet_rows(A, b, nodes, n_scalar_dofs):
    """Apply homogeneous Dirichlet rows to both vector components."""
    A = A.tolil()
    b = np.asarray(b, dtype=float).copy()
    nodes = np.asarray(nodes, dtype=int).reshape(-1)
    for node in np.unique(nodes):
        if node < 0 or node >= n_scalar_dofs:
            raise ValueError("Dirichlet node index outside valid range.")
        A[node, :] = 0.0
        A[node, node] = 1.0
        b[node] = 0.0

        vector_node = node + n_scalar_dofs
        A[vector_node, :] = 0.0
        A[vector_node, vector_node] = 1.0
        b[vector_node] = 0.0
    return A, b


def rayleigh_damping_matrix(M_eff, A_s, rayleigh_alpha, rayleigh_beta):
    """Return ``alpha*M + beta*K`` Rayleigh damping, or ``None`` if both are zero."""
    if rayleigh_alpha < 0.0 or rayleigh_beta < 0.0:
        raise ValueError("Rayleigh damping coefficients must be non-negative.")

    C_s = None
    if rayleigh_alpha != 0.0:
        C_s = rayleigh_alpha * M_eff
    if rayleigh_beta != 0.0:
        stiffness_damping = rayleigh_beta * A_s
        C_s = stiffness_damping if C_s is None else C_s + stiffness_damping
    return C_s


_rayleigh_damping_matrix = rayleigh_damping_matrix


def mass_norm(M, values):
    """Return the scalar or vector-valued mass norm induced by ``M``."""
    values = np.asarray(values, dtype=float).reshape(-1)
    n = M.shape[0]
    if values.size == n:
        value = values @ (M @ values)
    elif values.size == 2 * n:
        vx = values[:n]
        vy = values[n:]
        value = vx @ (M @ vx) + vy @ (M @ vy)
    else:
        raise ValueError("values must have length n or 2*n for the provided mass matrix.")
    return float(np.sqrt(max(value, 0.0)))


def solid_increment_error(M, u_s, u_s_prev, v_s, v_s_prev):
    """Return a relative mass-norm error between two structural iterates."""
    err_u = mass_norm(M, np.asarray(u_s) - np.asarray(u_s_prev)) / max(mass_norm(M, u_s), 1.0)
    err_v = mass_norm(M, np.asarray(v_s) - np.asarray(v_s_prev)) / max(mass_norm(M, v_s), 1.0)
    return max(err_u, err_v)


_solid_error = solid_increment_error
_solid_error_ff = solid_increment_error


def q1_shape_values(xi, eta):
    """Return Q1 basis values on the unit square."""
    return np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            xi * (1.0 - eta),
            xi * eta,
            (1.0 - xi) * eta,
        ]
    )


def q1_shape_gradients_reference(xi, eta):
    """Return Q1 basis gradients with respect to unit-square coordinates."""
    dphi_dxi = np.array(
        [
            -(1.0 - eta),
            1.0 - eta,
            eta,
            -eta,
        ]
    )
    dphi_deta = np.array(
        [
            -(1.0 - xi),
            -xi,
            xi,
            1.0 - xi,
        ]
    )
    return dphi_dxi, dphi_deta


_q1_shape_values = q1_shape_values
_q1_shape_gradients_reference = q1_shape_gradients_reference


def q1_reference_matrices(dx, dy):
    """Return analytic Q1 scalar derivative and mass matrices on a rectangle."""
    A_xx = (dy / dx) * np.array(
        [
            [1 / 3, -1 / 3, -1 / 6, 1 / 6],
            [-1 / 3, 1 / 3, 1 / 6, -1 / 6],
            [-1 / 6, 1 / 6, 1 / 3, -1 / 3],
            [1 / 6, -1 / 6, -1 / 3, 1 / 3],
        ]
    )
    A_yy = (dx / dy) * np.array(
        [
            [1 / 3, 1 / 6, -1 / 6, -1 / 3],
            [1 / 6, 1 / 3, -1 / 3, -1 / 6],
            [-1 / 6, -1 / 3, 1 / 3, 1 / 6],
            [-1 / 3, -1 / 6, 1 / 6, 1 / 3],
        ]
    )
    A_xy = np.array(
        [
            [1 / 4, -1 / 4, -1 / 4, 1 / 4],
            [1 / 4, -1 / 4, -1 / 4, 1 / 4],
            [-1 / 4, 1 / 4, 1 / 4, -1 / 4],
            [-1 / 4, 1 / 4, 1 / 4, -1 / 4],
        ]
    )
    M = (dx * dy) * np.array(
        [
            [1 / 9, 1 / 18, 1 / 36, 1 / 18],
            [1 / 18, 1 / 9, 1 / 18, 1 / 36],
            [1 / 36, 1 / 18, 1 / 9, 1 / 18],
            [1 / 18, 1 / 36, 1 / 18, 1 / 9],
        ]
    )
    return A_xx, A_xy, A_yy, M


def elasticity_blocks_from_scalar_matrices(A_xx, A_xy, A_yy, lam, mu):
    """Build the 2D linear-elasticity block stiffness matrix."""
    A11 = (lam + 2.0 * mu) * A_xx + mu * A_yy
    A12 = lam * A_xy.T + mu * A_xy
    A21 = lam * A_xy + mu * A_xy.T
    A22 = (lam + 2.0 * mu) * A_yy + mu * A_xx
    return bmat([[A11, A12], [A21, A22]], format="lil")


def solid_assembly(lx, ly, nx, ny, dx, dy, lam, mu, flag):
    """Assemble Q1 linear elasticity on a structured rectangular mesh."""
    flag = np.asarray(flag, dtype=int)
    if flag.shape != (4,):
        raise ValueError("flag must have shape (4,) in [bottom, right, top, left] order.")

    P, T, bdry = structured_rectangle_mesh(nx, ny, lx, ly, flags=flag)
    n_elem = nx * ny
    Nb = P.shape[1]

    A_xx_l, A_xy_l, A_yy_l, M_l = q1_reference_matrices(dx, dy)
    Kg_xx = np.zeros((16, n_elem))
    Kg_xy = np.zeros((16, n_elem))
    Kg_yy = np.zeros((16, n_elem))
    KM = np.zeros((16, n_elem))

    ii = np.array([0, 1, 2, 3] * 4)
    jj = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    Ig = T[ii, :]
    Jg = T[jj, :]

    for elem in range(n_elem):
        Kg_xx[:, elem] = A_xx_l.flatten(order="F")
        Kg_xy[:, elem] = A_xy_l.flatten(order="F")
        Kg_yy[:, elem] = A_yy_l.flatten(order="F")
        KM[:, elem] = M_l.flatten(order="F")

    A_xx = coo_matrix((Kg_xx.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))
    A_xy = coo_matrix((Kg_xy.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))
    A_yy = coo_matrix((Kg_yy.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))
    M = coo_matrix((KM.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb)).tocsc()
    A = elasticity_blocks_from_scalar_matrices(A_xx, A_xy, A_yy, lam, mu)

    b = np.zeros(2 * Nb)
    dirichlet = dirichlet_nodes_from_bdry(bdry)
    A, b = apply_homogeneous_dirichlet_rows(A, b, dirichlet, Nb)
    A = A.tocsc()
    return vector_mass_matrix(M), A, b, P, T, M, bdry


def assembly_solid_Q1_from_mesh(
    P,
    T,
    lam,
    mu,
    f1=None,
    f2=None,
    dirichlet_nodes=None,
    n_quad=4,
):
    """Assemble Q1 linear elasticity on an arbitrary quadrilateral mesh."""
    P = point_matrix(P)
    T = quadrilateral_connectivity(T)
    if T.size == 0:
        raise ValueError("T must contain at least one quadrilateral.")

    Nb = P.shape[1]
    n_elem = T.shape[0]
    if np.any(T < 0) or np.any(T >= Nb):
        raise ValueError("T contains node indices outside the valid range.")

    if dirichlet_nodes is None:
        dirichlet_nodes = np.array([], dtype=int)
    else:
        dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=int).reshape(-1)
        if np.any(dirichlet_nodes < 0) or np.any(dirichlet_nodes >= Nb):
            raise ValueError("dirichlet_nodes contains indices outside the valid range.")
        dirichlet_nodes = np.unique(dirichlet_nodes)

    if n_quad < 1:
        raise ValueError("n_quad must be positive.")
    if f1 is None:
        f1 = zero_body_force
    if f2 is None:
        f2 = zero_body_force

    q_1d, w_1d = leggauss(n_quad)
    q_1d = 0.5 * (q_1d + 1.0)
    w_1d = 0.5 * w_1d

    A_xx = lil_matrix((Nb, Nb))
    A_xy = lil_matrix((Nb, Nb))
    A_yy = lil_matrix((Nb, Nb))
    M = lil_matrix((Nb, Nb))

    for elem, nodes in enumerate(T):
        x = P[0, nodes]
        y = P[1, nodes]
        A_xx_e = np.zeros((4, 4))
        A_xy_e = np.zeros((4, 4))
        A_yy_e = np.zeros((4, 4))
        M_e = np.zeros((4, 4))

        for xi, wx in zip(q_1d, w_1d):
            for eta, wy in zip(q_1d, w_1d):
                phi = q1_shape_values(xi, eta)
                dphi_dxi, dphi_deta = q1_shape_gradients_reference(xi, eta)

                x_xi = dphi_dxi @ x
                x_eta = dphi_deta @ x
                y_xi = dphi_dxi @ y
                y_eta = dphi_deta @ y
                jac = np.array([[x_xi, x_eta], [y_xi, y_eta]])
                det_j = np.linalg.det(jac)
                if det_j <= 1.0e-14:
                    raise ValueError(f"Degenerate or negatively oriented quadrilateral at element {elem}.")

                grads_ref = np.vstack([dphi_dxi, dphi_deta])
                grads_phys = np.linalg.inv(jac).T @ grads_ref
                dphi_dx = grads_phys[0, :]
                dphi_dy = grads_phys[1, :]
                weight = wx * wy * det_j

                A_xx_e += weight * np.outer(dphi_dx, dphi_dx)
                A_xy_e += weight * np.outer(dphi_dy, dphi_dx)
                A_yy_e += weight * np.outer(dphi_dy, dphi_dy)
                M_e += weight * np.outer(phi, phi)

        for i in range(4):
            I = nodes[i]
            for j in range(4):
                J = nodes[j]
                A_xx[I, J] += A_xx_e[i, j]
                A_xy[I, J] += A_xy_e[i, j]
                A_yy[I, J] += A_yy_e[i, j]
                M[I, J] += M_e[i, j]

    A = elasticity_blocks_from_scalar_matrices(A_xx, A_xy, A_yy, lam, mu)
    M = M.tocsc()
    f1_v = as_nodal_values(f1(P[0, :], P[1, :]), Nb, "f1")
    f2_v = as_nodal_values(f2(P[0, :], P[1, :]), Nb, "f2")

    b = np.zeros(2 * Nb)
    b[:Nb] = M @ f1_v
    b[Nb:] = M @ f2_v

    A, b = apply_homogeneous_dirichlet_rows(A, b, dirichlet_nodes, Nb)
    return A.tocsc(), b, M


_assembly_solid_Q1_from_mesh = assembly_solid_Q1_from_mesh


def solid_assembly_from_mesh(P, T, bdry, lam, mu, f1=None, f2=None, n_quad=4):
    """Assemble Q1 linear elasticity on an arbitrary quadrilateral mesh."""
    P = point_matrix(P)
    T_columns = quadrilateral_connectivity(T).T
    bdry = np.asarray(bdry, dtype=int)
    if bdry.ndim != 2 or bdry.shape[0] != 5:
        raise ValueError("bdry must have shape (5, n_boundary_edges).")

    dirichlet_nodes = dirichlet_nodes_from_bdry(bdry)
    A, b, M = assembly_solid_Q1_from_mesh(
        P,
        T_columns,
        lam,
        mu,
        f1=f1,
        f2=f2,
        dirichlet_nodes=dirichlet_nodes,
        n_quad=n_quad,
    )
    return vector_mass_matrix(M), A, b, P, T_columns, M, bdry


def square_to_disk_map(U, V, radius, center):
    """Map reference-square coordinates onto a disk."""
    X = U * np.sqrt(np.maximum(0.0, 1.0 - 0.5 * V * V))
    Y = V * np.sqrt(np.maximum(0.0, 1.0 - 0.5 * U * U))
    return center[0] + radius * X, center[1] + radius * Y


_square_to_disk_map = square_to_disk_map


def assemble_solid_circle(radius, center, nx, ny, lam, mu, flag, f1=None, f2=None, n_quad=4):
    """Assemble Q1 linear elasticity on a mapped quadrilateral disk mesh."""
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")

    center = np.asarray(center, dtype=float).reshape(-1)
    if center.size != 2:
        raise ValueError("center must contain exactly two coordinates.")

    flag = np.asarray(flag, dtype=int)
    if flag.shape != (4,):
        raise ValueError("flag must have shape (4,) in [bottom, right, top, left] order.")

    u = np.linspace(-1.0, 1.0, nx + 1)
    v = np.linspace(-1.0, 1.0, ny + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    X, Y = square_to_disk_map(U, V, radius, center)
    P = np.vstack([X.ravel(), Y.ravel()])

    ix, jy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    n1 = ix * (ny + 1) + jy
    n2 = n1 + (ny + 1)
    n3 = n2 + 1
    n4 = n1 + 1
    T = np.stack([n1, n2, n3, n4], axis=-1).reshape(-1, 4).T

    bdry_list = []
    bdry_list += [[flag[0], elem, T[0, elem], T[1, elem], 0] for elem in range(0, nx * ny, ny)]
    bdry_list += [
        [flag[1], elem, T[1, elem], T[2, elem], 1]
        for elem in range((nx - 1) * ny, nx * ny)
    ]
    bdry_list += [
        [flag[2], elem, T[2, elem], T[3, elem], 2]
        for elem in range(nx * ny - 1, -1, -ny)
    ]
    bdry_list += [[flag[3], elem, T[3, elem], T[0, elem], 3] for elem in range(ny - 1, -1, -1)]
    bdry = np.array(bdry_list, dtype=int).T

    dirichlet = dirichlet_nodes_from_bdry(bdry)
    A, b, M = assembly_solid_Q1_from_mesh(
        P,
        T,
        lam,
        mu,
        f1=f1,
        f2=f2,
        dirichlet_nodes=dirichlet,
        n_quad=n_quad,
    )
    return vector_mass_matrix(M), A, b, P, T, M, bdry


def assembly_solid_T_from_mesh(
    P,
    T,
    lam,
    mu,
    f1=None,
    f2=None,
    dirichlet_nodes=None,
    verbose=False,
    bdry=None,
    return_full=False,
):
    """Assemble P1 linear elasticity on an arbitrary triangular mesh."""
    P = point_matrix(P)
    T = triangle_connectivity(T)
    if T.size == 0:
        raise ValueError("T must contain at least one triangle.")

    Nb = P.shape[1]
    n_elem = T.shape[0]
    if np.any(T < 0) or np.any(T >= Nb):
        raise ValueError("T contains node indices outside the valid range.")

    if dirichlet_nodes is None:
        dirichlet_nodes = np.array([], dtype=int)
    else:
        dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=int).reshape(-1)
        if np.any(dirichlet_nodes < 0) or np.any(dirichlet_nodes >= Nb):
            raise ValueError("dirichlet_nodes contains indices outside the valid range.")
        dirichlet_nodes = np.unique(dirichlet_nodes)

    if bdry is None:
        bdry = triangular_boundary_description(T)
    else:
        bdry = np.asarray(bdry, dtype=int)
        if bdry.ndim != 2 or bdry.shape[0] != 5:
            raise ValueError("bdry must have shape (5, n_boundary_edges).")

    if f1 is None:
        f1 = zero_body_force
    if f2 is None:
        f2 = zero_body_force

    if verbose:
        print(f"Number of nodes:    {Nb}")
        print(f"Number of elements: {n_elem}")
        print(f"Dirichlet nodes:    {len(dirichlet_nodes)}")
        print(f"Boundary edges:     {bdry.shape[1]}")

    A_xx = lil_matrix((Nb, Nb))
    A_xy = lil_matrix((Nb, Nb))
    A_yy = lil_matrix((Nb, Nb))
    M = lil_matrix((Nb, Nb))

    M_ref = np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    ) / 12.0
    grad_ref = np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )

    for elem, nodes in enumerate(T):
        xe = P[0, nodes]
        ye = P[1, nodes]
        B = np.array([[xe[1] - xe[0], xe[2] - xe[0]], [ye[1] - ye[0], ye[2] - ye[0]]])
        detB = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]
        if abs(detB) < 1.0e-14:
            raise ValueError(f"Degenerate triangle detected at element {elem}.")

        area = 0.5 * abs(detB)
        inv_detB = 1.0 / detB
        Binv = np.array([[B[1, 1], -B[0, 1]], [-B[1, 0], B[0, 0]]]) * inv_detB
        grad_phys = Binv.T @ grad_ref
        dNdx = grad_phys[0, :]
        dNdy = grad_phys[1, :]

        A_xx_e = area * np.outer(dNdx, dNdx)
        A_yy_e = area * np.outer(dNdy, dNdy)
        A_xy_e = area * np.outer(dNdy, dNdx)
        M_e = area * M_ref

        for i in range(3):
            I = nodes[i]
            for j in range(3):
                J = nodes[j]
                A_xx[I, J] += A_xx_e[i, j]
                A_yy[I, J] += A_yy_e[i, j]
                A_xy[I, J] += A_xy_e[i, j]
                M[I, J] += M_e[i, j]

    A = elasticity_blocks_from_scalar_matrices(A_xx, A_xy, A_yy, lam, mu)
    M = M.tocsc()
    f1_v = as_nodal_values(f1(P[0, :], P[1, :]), Nb, "f1")
    f2_v = as_nodal_values(f2(P[0, :], P[1, :]), Nb, "f2")

    b = np.zeros(2 * Nb)
    b[:Nb] = M @ f1_v
    b[Nb:] = M @ f2_v

    A, b = apply_homogeneous_dirichlet_rows(A, b, dirichlet_nodes, Nb)
    if verbose:
        print("System assembled.")

    A = A.tocsc()
    if return_full:
        return vector_mass_matrix(M), A, b, P, T.T, M, bdry
    return A, b, M


def assembly_solid_T(nx, ny, lx, ly, lam, mu, f1, f2, dir_nodes, verbose=False):
    """Assemble P1 elasticity on a structured rectangle split into triangles."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("lx and ly must be positive.")

    dir_nodes = np.asarray(dir_nodes, dtype=int)
    if dir_nodes.shape != (4,):
        raise ValueError("dir_nodes must have shape (4,) in [bottom, right, top, left] order.")

    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()]).T
    Nb = P.shape[1]

    T_list = []
    for i in range(nx):
        for j in range(ny):
            n1 = i * (ny + 1) + j
            n2 = n1 + (ny + 1)
            n3 = n2 + 1
            n4 = n1 + 1
            if (i + j) % 2 == 0:
                T_list.append([n1, n2, n3])
                T_list.append([n1, n3, n4])
            else:
                T_list.append([n1, n2, n4])
                T_list.append([n2, n3, n4])
    T = np.array(T_list, dtype=int)

    tol = 1.0e-10
    left_nodes = np.where(np.abs(P[0, :]) < tol)[0]
    right_nodes = np.where(np.abs(P[0, :] - lx) < tol)[0]
    bottom_nodes = np.where(np.abs(P[1, :]) < tol)[0]
    top_nodes = np.where(np.abs(P[1, :] - ly) < tol)[0]
    bdry_nodes = np.unique(np.concatenate((left_nodes, right_nodes, bottom_nodes, top_nodes)))

    dirichlet_list = []
    if dir_nodes[0] == 1:
        dirichlet_list.append(bottom_nodes)
    if dir_nodes[1] == 1:
        dirichlet_list.append(right_nodes)
    if dir_nodes[2] == 1:
        dirichlet_list.append(top_nodes)
    if dir_nodes[3] == 1:
        dirichlet_list.append(left_nodes)
    dirichlet_nodes = np.unique(np.concatenate(dirichlet_list)) if dirichlet_list else np.array([], dtype=int)

    if verbose:
        print(f"Number of nodes:    {Nb}")
        print(f"Number of elements: {T.shape[0]}")
        print(f"Dirichlet nodes:    {len(dirichlet_nodes)}")

    A, b, M = assembly_solid_T_from_mesh(
        P,
        T,
        lam,
        mu,
        f1,
        f2,
        dirichlet_nodes=dirichlet_nodes,
        verbose=False,
    )
    if verbose:
        print("System assembled.")
    return A, b, P, T, M, bdry_nodes, dirichlet_nodes


def assembly_solid_T_from_mesh_heterogeneous(
    P,
    T=None,
    lam_e=None,
    mu_e=None,
    rho_e=None,
    f1=None,
    f2=None,
    dirichlet_nodes=None,
    bdry=None,
    return_full=True,
    return_density_mass=False,
    verbose=False,
):
    """Assemble P1 linear elasticity with elementwise material parameters."""
    if isinstance(P, TurekSolidMesh):
        mesh = P
        P = mesh.P
        if T is None:
            T = mesh.T
        if lam_e is None:
            lam_e = mesh.lam
        if mu_e is None:
            mu_e = mesh.mu
        if rho_e is None:
            rho_e = mesh.rho
        if dirichlet_nodes is None:
            dirichlet_nodes = mesh.dirichlet_nodes
        if bdry is None:
            bdry = mesh.bdry

    P = point_matrix(P)
    if T is None:
        raise ValueError("T must be provided when P is not a TurekSolidMesh.")
    T = triangle_connectivity(T)
    if T.size == 0:
        raise ValueError("T must contain at least one triangle.")

    Nb = P.shape[1]
    n_elem = T.shape[0]
    if np.any(T < 0) or np.any(T >= Nb):
        raise ValueError("T contains node indices outside the valid range.")

    lam_e = as_element_values(lam_e, n_elem, "lam_e")
    mu_e = as_element_values(mu_e, n_elem, "mu_e", positive=True)
    if rho_e is None:
        rho_e = np.ones(n_elem, dtype=float)
    else:
        rho_e = as_element_values(rho_e, n_elem, "rho_e", positive=True)

    if dirichlet_nodes is None:
        dirichlet_nodes = np.array([], dtype=int)
    else:
        dirichlet_nodes = np.asarray(dirichlet_nodes, dtype=int).reshape(-1)
        if np.any(dirichlet_nodes < 0) or np.any(dirichlet_nodes >= Nb):
            raise ValueError("dirichlet_nodes contains indices outside the valid range.")
        dirichlet_nodes = np.unique(dirichlet_nodes)

    if bdry is None:
        bdry = triangular_boundary_description(T)
    else:
        bdry = np.asarray(bdry, dtype=int)
        if bdry.ndim != 2 or bdry.shape[0] != 5:
            raise ValueError("bdry must have shape (5, n_boundary_edges).")

    if f1 is None:
        f1 = zero_body_force
    if f2 is None:
        f2 = zero_body_force

    if verbose:
        print(f"Number of nodes:    {Nb}")
        print(f"Number of elements: {n_elem}")
        print(f"Dirichlet nodes:    {len(dirichlet_nodes)}")
        print(f"Boundary edges:     {bdry.shape[1]}")

    rows_A = []
    cols_A = []
    data_A = []
    rows_M = []
    cols_M = []
    data_M = []
    data_M_rho = []

    M_ref = np.array(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    ) / 12.0
    grad_ref = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    for elem, nodes in enumerate(T):
        xe = P[0, nodes]
        ye = P[1, nodes]
        B = np.array([[xe[1] - xe[0], xe[2] - xe[0]], [ye[1] - ye[0], ye[2] - ye[0]]])
        detB = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]
        if abs(detB) < 1.0e-14:
            raise ValueError(f"Degenerate triangle detected at element {elem}.")

        area = 0.5 * abs(detB)
        inv_detB = 1.0 / detB
        Binv = np.array([[B[1, 1], -B[0, 1]], [-B[1, 0], B[0, 0]]]) * inv_detB
        grad_phys = Binv.T @ grad_ref
        dNdx = grad_phys[0, :]
        dNdy = grad_phys[1, :]

        A_xx_e = area * np.outer(dNdx, dNdx)
        A_yy_e = area * np.outer(dNdy, dNdy)
        A_xy_e = area * np.outer(dNdy, dNdx)

        lam = lam_e[elem]
        mu = mu_e[elem]
        K11 = (lam + 2.0 * mu) * A_xx_e + mu * A_yy_e
        K12 = lam * A_xy_e.T + mu * A_xy_e
        K21 = lam * A_xy_e + mu * A_xy_e.T
        K22 = (lam + 2.0 * mu) * A_yy_e + mu * A_xx_e
        K_e = np.block([[K11, K12], [K21, K22]])

        dofs = np.concatenate([nodes, nodes + Nb])
        rr_A, cc_A = np.meshgrid(dofs, dofs, indexing="ij")
        rows_A.append(rr_A.ravel())
        cols_A.append(cc_A.ravel())
        data_A.append(K_e.ravel())

        M_e = area * M_ref
        rr_M, cc_M = np.meshgrid(nodes, nodes, indexing="ij")
        rows_M.append(rr_M.ravel())
        cols_M.append(cc_M.ravel())
        data_M.append(M_e.ravel())
        data_M_rho.append((rho_e[elem] * M_e).ravel())

    A = sp.coo_matrix(
        (np.concatenate(data_A), (np.concatenate(rows_A), np.concatenate(cols_A))),
        shape=(2 * Nb, 2 * Nb),
    ).tolil()
    M = sp.coo_matrix(
        (np.concatenate(data_M), (np.concatenate(rows_M), np.concatenate(cols_M))),
        shape=(Nb, Nb),
    ).tocsc()
    M_rho = sp.coo_matrix(
        (np.concatenate(data_M_rho), (np.concatenate(rows_M), np.concatenate(cols_M))),
        shape=(Nb, Nb),
    ).tocsc()

    f1_v = as_nodal_values(f1(P[0, :], P[1, :]), Nb, "f1")
    f2_v = as_nodal_values(f2(P[0, :], P[1, :]), Nb, "f2")

    b = np.zeros(2 * Nb)
    b[:Nb] = M @ f1_v
    b[Nb:] = M @ f2_v

    A, b = apply_homogeneous_dirichlet_rows(A, b, dirichlet_nodes, Nb)
    if verbose:
        print("Heterogeneous solid system assembled.")

    A = A.tocsc()
    M_el = vector_mass_matrix(M)
    result = (M_el, A, b, P, T.T, M, bdry) if return_full else (A, b, M)

    if return_density_mass:
        M_rho_el = vector_mass_matrix(M_rho)
        return (*result, M_rho_el, M_rho)
    return result


def assemble_turek_solid_system(mesh, return_density_mass=True):
    """Assemble stiffness, mass and load arrays for a Turek solid mesh."""
    result = assembly_solid_T_from_mesh_heterogeneous(
        mesh,
        return_density_mass=return_density_mass,
    )

    if return_density_mass:
        M_el, A_s, b_s, P, T, M, B, M_rho_el, M_rho = result
    else:
        M_el, A_s, b_s, P, T, M, B = result
        M_rho_el = None
        M_rho = None

    return {
        "M_el": M_el,
        "A_s": A_s,
        "b_s": b_s,
        "P": P,
        "T": T,
        "M": M,
        "B": B,
        "M_rho_el": M_rho_el,
        "M_rho": M_rho,
    }


def solid_system_summary(solid_system):
    """Return compact metadata for an assembled solid system dictionary."""
    A_s = solid_system["A_s"]
    M_el = solid_system["M_el"]
    M = solid_system["M"]
    summary = {
        "solid_assembly": "linear_elasticity",
        "solid_A_shape": " ".join(str(v) for v in A_s.shape),
        "solid_A_nnz": int(A_s.nnz),
        "solid_M_el_shape": " ".join(str(v) for v in M_el.shape),
        "solid_M_el_nnz": int(M_el.nnz),
        "solid_M_shape": " ".join(str(v) for v in M.shape),
        "solid_M_nnz": int(M.nnz),
        "solid_b_norm": float(np.linalg.norm(solid_system["b_s"])),
    }

    if solid_system.get("M_rho_el") is not None:
        summary.update(
            {
                "solid_density_mass_available": True,
                "solid_M_rho_el_shape": " ".join(str(v) for v in solid_system["M_rho_el"].shape),
                "solid_M_rho_el_nnz": int(solid_system["M_rho_el"].nnz),
                "solid_M_rho_shape": " ".join(str(v) for v in solid_system["M_rho"].shape),
                "solid_M_rho_nnz": int(solid_system["M_rho"].nnz),
            }
        )
    else:
        summary["solid_density_mass_available"] = False
    return summary


__all__ = [
    "zero_body_force",
    "as_nodal_values",
    "_as_nodal_values",
    "as_element_values",
    "_as_element_values",
    "vector_mass_matrix",
    "apply_homogeneous_dirichlet_rows",
    "rayleigh_damping_matrix",
    "_rayleigh_damping_matrix",
    "mass_norm",
    "solid_increment_error",
    "_solid_error",
    "_solid_error_ff",
    "q1_shape_values",
    "_q1_shape_values",
    "q1_shape_gradients_reference",
    "_q1_shape_gradients_reference",
    "q1_reference_matrices",
    "elasticity_blocks_from_scalar_matrices",
    "solid_assembly",
    "assembly_solid_Q1_from_mesh",
    "_assembly_solid_Q1_from_mesh",
    "solid_assembly_from_mesh",
    "square_to_disk_map",
    "_square_to_disk_map",
    "assemble_solid_circle",
    "assembly_solid_T_from_mesh",
    "assembly_solid_T",
    "assembly_solid_T_from_mesh_heterogeneous",
    "assemble_turek_solid_system",
    "solid_system_summary",
]
