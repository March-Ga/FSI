"""Local finite-strain materials and reference-domain assembly for immersed ID.

Material responses adapted from structural-mechanics-fem/code/2D at commit
84a5fc5c45b496d2273621a215868f677798f69b (Neo_Hookean.py and
Saint_Venant_Kirchhoff.py). This copy is maintained independently in FSI;
there is no runtime dependency on the structural-mechanics-fem repository.

Plane strain, unit out-of-plane thickness. The assembly uses FSI's blocked
DOFs [all ux, all uy], reference coordinates, and Q1 quads or P1 triangles.
"""

from functools import partial

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix

from .geometry import point_matrix, quadrilateral_connectivity, triangle_connectivity
from .structure import q1_shape_gradients_reference

def neo_hookean_response(F, lambda_, mu, compute_tangent=True, penalty=0.0):
    """Return P and A for the 2D plane-strain compressible Neo-Hookean material.

    Energy: mu/2 * (tr(F.T @ F) - 2) - mu * log(J)
            + lambda_/2 * log(J)**2, where J = det(F) > 0.
    This is the 3D law restricted to F33=1, with zero out-of-plane
    shears; energy and force are per unit out-of-plane thickness.
    A[i, J, j, L] = dP[i, J]/dF[j, L]. F may have leading batch
    dimensions. With compute_tangent=False, return (P, None).
    The penalty argument is reserved for API compatibility and must be zero.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim < 2 or F.shape[-2:] != (2, 2):
        raise ValueError("F must have shape (..., 2, 2)")
    if not np.isfinite(lambda_) or not np.isfinite(mu):
        raise ValueError("lambda_ and mu must be finite")
    if isinstance(penalty, (bool, np.bool_)) or np.ndim(penalty) != 0:
        raise ValueError("penalty must be zero for this Neo-Hookean model")
    try:
        penalty = float(penalty)
    except (TypeError, ValueError) as exc:
        raise ValueError("penalty must be zero for this Neo-Hookean model") from exc
    if not np.isfinite(penalty) or penalty != 0.0:
        raise ValueError("penalty must be zero for this Neo-Hookean model")

    J = np.linalg.det(F)
    if not np.all(np.isfinite(J) & (J > 0.0)):
        raise ValueError("det(F) must be positive and finite")
    Q = np.linalg.inv(F).swapaxes(-1, -2)
    alpha = lambda_ * np.log(J) - mu
    P = mu * F + alpha[..., None, None] * Q
    if not compute_tangent:
        return P, None

    I = np.eye(2)
    A = (
        mu * np.einsum("ij,JL->iJjL", I, I)
        + lambda_ * np.einsum("...iJ,...jL->...iJjL", Q, Q)
        - alpha[..., None, None, None, None]
        * np.einsum("...iL,...jJ->...iJjL", Q, Q)
    )
    return P, A


neo_hookean_response.supports_batch = True


def stvk_response(F, lambda_, mu, compute_tangent=True, penalty=0.0):
    """Return first Piola stress P and its consistent tangent A for 2D plane-strain SVK.

    A[i, J, j, L] = dP[i, J]/dF[j, L]. With compute_tangent=False,
    return (P, None). F may have leading batch dimensions; its last two
    dimensions must be (2, 2). penalty adds penalty/2 * (det(F) - 1)**2
    to the SVK energy; penalty=0 recovers the original compressible model.
    This is the 3D SVK law with F33=1 and all out-of-plane shears zero;
    hence E33=0. Energy and force are per unit out-of-plane thickness.
    Every deformation must have det(F) > 0.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim < 2 or F.shape[-2:] != (2, 2):
        raise ValueError("F must have shape (..., 2, 2)")
    if not np.isfinite(lambda_) or not np.isfinite(mu):
        raise ValueError("lambda_ and mu must be finite")
    if isinstance(penalty, (bool, np.bool_)) or np.ndim(penalty) != 0:
        raise ValueError("penalty must be a finite nonnegative scalar")
    try:
        penalty = float(penalty)
    except (TypeError, ValueError) as exc:
        raise ValueError("penalty must be a finite nonnegative scalar") from exc
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("penalty must be a finite nonnegative scalar")
    J = np.linalg.det(F)
    if not np.all(np.isfinite(J) & (J > 0.0)):
        raise ValueError("det(F) must be positive and finite")

    I = np.eye(2)
    E = 0.5 * (F.swapaxes(-1, -2) @ F - I)
    S = lambda_ * np.trace(E, axis1=-2, axis2=-1)[..., None, None] * I + 2.0 * mu * E
    P = F @ S
    if penalty > 0.0:
        FinvT = np.linalg.inv(F).swapaxes(-1, -2)
        P += (penalty * J * (J - 1.0))[..., None, None] * FinvT
    if not compute_tangent:
        return P, None

    # dP = dF S + F dS, with dS = lambda tr(F^T dF) I + mu (dF^T F + F^T dF).
    FFt = F @ F.swapaxes(-1, -2)
    A = (
        np.einsum("ij,...LJ->...iJjL", I, S)
        + lambda_ * np.einsum("...iJ,...jL->...iJjL", F, F)
        + mu * np.einsum("...iL,...jJ->...iJjL", F, F)
        + mu * np.einsum("JL,...ij->...iJjL", I, FFt)
    )
    if penalty > 0.0:
        A += (
            (penalty * J * (2.0 * J - 1.0))[..., None, None, None, None]
            * np.einsum("...iJ,...jL->...iJjL", FinvT, FinvT)
            - (penalty * J * (J - 1.0))[..., None, None, None, None]
            * np.einsum("...iL,...jJ->...iJjL", FinvT, FinvT)
        )
    return P, A


# The assembly routine can evaluate this material for a block of elements.
stvk_response.supports_batch = True


class HyperelasticForce:
    """Cache reference quadrature and assemble a plane-strain internal force.

    ``P`` and ``T`` use the same conventions as FSI's structural assemblers.
    Quad nodes must be ordered bottom-left, bottom-right, top-right, top-left;
    triangles must be counterclockwise. General Q1 reference quads are supported.
    Calling the object returns a force vector in blocked FSI order. ``assemble``
    can also return the consistent tangent, e.g. for verification or future
    nonlinear structural solvers. The ID coupling only needs the force.
    """

    def __init__(
        self, P, T, lambda_, mu, *, model="neo_hookean", element_type="quad",
        penalty=0.0, quadrature_order=2,
    ):
        model = str(model).strip().lower().replace("-", "_")
        responses = {"svk": stvk_response, "neo_hookean": neo_hookean_response}
        if model not in responses:
            raise ValueError("Nonlinear material model must be 'svk' or 'neo_hookean'.")
        if not np.isfinite(mu) or not np.isfinite(lambda_) or mu <= 0.0 or lambda_ + mu <= 0.0:
            raise ValueError("Plane-strain material requires finite mu > 0 and lambda + mu > 0.")
        if (
            isinstance(quadrature_order, (bool, np.bool_))
            or not isinstance(quadrature_order, (int, np.integer))
            or quadrature_order < 1
        ):
            raise ValueError("quadrature_order must be a positive integer.")
        self.material = partial(responses[model], lambda_=lambda_, mu=mu, penalty=penalty)
        self.material(np.eye(2), compute_tangent=False)  # Validate material parameters.
        self.model = model
        self.P = point_matrix(P, copy=True)
        if not np.all(np.isfinite(self.P)):
            raise ValueError("Reference coordinates must be finite.")
        self.n_nodes = self.P.shape[1]
        if element_type == "quad":
            self.T = quadrilateral_connectivity(T).copy()
            nodes, weights = leggauss(quadrature_order)
            nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
            gradients = np.array([
                np.column_stack(q1_shape_gradients_reference(xi, eta))
                for xi in nodes for eta in nodes
            ])
            qweights = np.outer(weights, weights).ravel()
        elif element_type == "tri":
            self.T = triangle_connectivity(T).copy()
            # P1 deformation gradients are constant within each triangle.
            gradients = np.array([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]])
            qweights = np.array([0.5])
        else:
            raise ValueError("element_type must be 'quad' or 'tri'.")
        if self.T.size == 0 or np.any((self.T < 0) | (self.T >= self.n_nodes)):
            raise ValueError("T must contain elements with valid reference node IDs.")

        coordinates = self.P.T[self.T]
        jac = np.einsum("eai,qaj->eqij", coordinates, gradients)
        det_jac = np.linalg.det(jac)
        if not np.all(np.isfinite(det_jac) & (det_jac > 0.0)):
            raise ValueError("Reference elements must have positive finite Jacobians.")
        self.gradients = np.einsum("qaj,eqji->eqai", gradients, np.linalg.inv(jac))
        self.dV = det_jac * qweights
        self.dofs = np.stack((self.T, self.T + self.n_nodes), axis=-1).reshape(
            len(self.T), -1
        )

    def __call__(self, u):
        """Return f_int(u), integrated over the undeformed reference mesh."""
        return self.assemble(u, compute_tangent=False)[0]

    def assemble(self, u, compute_tangent=False):
        """Return ``(f_int, K_or_None)`` using F = I + grad_X(u)."""
        u = np.asarray(u, dtype=float)
        if u.shape != (2 * self.n_nodes,) or not np.all(np.isfinite(u)):
            raise ValueError("u must be a finite vector of length 2*n_nodes in blocked FSI order.")
        nodal_u = u.reshape(2, self.n_nodes).T
        force = np.zeros_like(u)
        rows, cols, data = [], [], []
        for first in range(0, len(self.T), 128):
            last = min(first + 128, len(self.T))
            G, dV = self.gradients[first:last], self.dV[first:last]
            u_e = nodal_u[self.T[first:last]]
            F = np.eye(2) + np.einsum("eai,eqaJ->eqiJ", u_e, G, optimize=True)
            stress, tangent = self.material(F, compute_tangent=compute_tangent)
            local_force = np.einsum("eqaJ,eqiJ,eq->eai", G, stress, dV, optimize=True)
            dofs = self.dofs[first:last]
            np.add.at(force, dofs.ravel(), local_force.ravel())
            if compute_tangent:
                local_K = np.einsum(
                    "eqaJ,eqiJjL,eqbL,eq->eaibj", G, tangent, G, dV, optimize=True
                ).reshape(len(dofs), dofs.shape[1], dofs.shape[1])
                rows.append(np.broadcast_to(dofs[:, :, None], local_K.shape).ravel())
                cols.append(np.broadcast_to(dofs[:, None, :], local_K.shape).ravel())
                data.append(local_K.ravel())
        K = None
        if compute_tangent:
            K = coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(u.size, u.size),
            ).tocsr()
        return force, K
