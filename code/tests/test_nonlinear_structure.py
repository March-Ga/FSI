"""Numerical invariance, tangent, and coupled ID checks for local materials."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.run_case import (
    build_falling_cylinder,
    build_falling_square,
    material_from_config,
    solver_kwargs_from_config,
)
from immersed_fsi.geometry import structured_rectangle_mesh
from immersed_fsi.nonlinear_structure import HyperelasticForce, neo_hookean_response, stvk_response
from immersed_fsi.solvers import _coerce_solid_system, solve_ib, solve_id
from immersed_fsi.structure import assembly_solid_Q1_from_mesh, assembly_solid_T_from_mesh


MATERIALS = (("svk", 0.0), ("svk", 7.0), ("neo_hookean", 0.0))


def reference_mesh(element_type):
    if element_type == "quad":
        P, T, _ = structured_rectangle_mesh(2, 3, 0.2, 0.3)
        # A nonrectangular reference mesh exercises the geometric mapping.
        P[0] += 0.15 * P[1] + 0.2 * P[0] * P[1]
        return P, T
    return np.array([[0.0, 0.3, 0.25, 0.0], [0.0, 0.0, 0.2, 0.15]]), np.array(
        [[0, 0], [1, 2], [2, 3]]
    )


class MaterialTests(unittest.TestCase):
    def test_material_tangent_matches_stress_derivative(self):
        F = np.array([[[1.15, 0.12], [-0.07, 0.91]], [[0.95, 0.04], [0.13, 1.1]]])
        direction = np.array([[0.2, -0.3], [0.4, 0.1]])
        h = 1e-6
        for model, penalty in MATERIALS:
            with self.subTest(model=model, penalty=penalty):
                response = stvk_response if model == "svk" else neo_hookean_response
                _, A = response(F, 4.0, 3.0, penalty=penalty)
                plus, _ = response(F + h * direction, 4.0, 3.0, penalty=penalty, compute_tangent=False)
                minus, _ = response(F - h * direction, 4.0, 3.0, penalty=penalty, compute_tangent=False)
                assert_allclose(
                    np.einsum("...iJjL,jL->...iJ", A, direction),
                    (plus - minus) / (2 * h), rtol=1e-8, atol=1e-8,
                )

    def test_inverted_deformation_is_rejected(self):
        for response in (stvk_response, neo_hookean_response):
            with self.assertRaisesRegex(ValueError, "det\\(F\\)"):
                response(np.diag([-1.0, 1.0]), 4.0, 3.0)


class AssemblyTests(unittest.TestCase):
    def test_rigid_motion_has_zero_internal_force(self):
        angle = 1.1
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for element_type in ("quad", "tri"):
            P, T = reference_mesh(element_type)
            u = ((rotation - np.eye(2)) @ P + np.array([[0.7], [-0.4]])).ravel()
            for model, penalty in MATERIALS:
                with self.subTest(element_type=element_type, model=model, penalty=penalty):
                    force = HyperelasticForce(P, T, 4.0, 3.0, model=model, penalty=penalty, element_type=element_type)
                    assert_allclose(force(u), 0.0, atol=2e-13)

    def test_reference_tangent_recovers_linear_elasticity(self):
        for element_type in ("quad", "tri"):
            P, T = reference_mesh(element_type)
            assembly = assembly_solid_Q1_from_mesh if element_type == "quad" else assembly_solid_T_from_mesh
            A, _, _ = assembly(P, T, 4.0, 3.0)
            for model in ("svk", "neo_hookean"):
                with self.subTest(element_type=element_type, model=model):
                    force = HyperelasticForce(P, T, 4.0, 3.0, model=model, element_type=element_type, quadrature_order=4)
                    f, K = force.assemble(np.zeros(P.size), compute_tangent=True)
                    assert_allclose(f, 0.0, atol=1e-14)
                    assert_allclose(K.toarray(), A.toarray(), atol=3e-13)

    def test_assembled_tangent_matches_force_derivative(self):
        rng = np.random.default_rng(87)
        for element_type in ("quad", "tri"):
            P, T = reference_mesh(element_type)
            u = (np.array([[0.1, 0.05], [-0.03, -0.04]]) @ P).ravel()
            direction = rng.normal(size=u.size)
            for model, penalty in MATERIALS:
                with self.subTest(element_type=element_type, model=model, penalty=penalty):
                    force = HyperelasticForce(P, T, 4.0, 3.0, model=model, penalty=penalty, element_type=element_type)
                    f, K = force.assemble(u, compute_tangent=True)
                    h = 1e-7
                    derivative = (force(u + h * direction) - force(u - h * direction)) / (2 * h)
                    assert_allclose(K @ direction, derivative, rtol=2e-8, atol=2e-8)
                    assert_allclose(K.toarray(), K.T.toarray(), atol=1e-12)
                    # Total force of an unloaded hyperelastic body is zero.
                    assert_allclose(f.reshape(2, -1).sum(axis=1), 0.0, atol=1e-13)


def small_config(model="neo_hookean", kind="falling_square"):
    return {
        "case": {"method": "id", "kind": kind},
        "time": {"dt": 1e-3, "Nt": 3},
        "fluid": {"mode": "free_fall", "lx": 1.0, "ly": 1.0, "nx": 10, "ny": 10, "rho": 1.0, "mu": 0.01},
        "solid": {
            "size": 0.2, "center": [0.5, 0.5], "nx": 2, "ny": 3,
            "radius": 0.12, "edge_length": 0.1, "rho": 1.2,
            "material": {"model": model, "E": 2.0, "nu": 0.3},
            "body_force": {"acceleration": [0.0, -0.1], "density_scale": "buoyant"},
        },
        "coupling": {"transfer_kind": "ID", "tol": 1e-8, "max_iterations": 100},
        "output": {"verbose": False},
    }


class CouplingTests(unittest.TestCase):
    def test_id_with_deformed_nonlinear_solid(self):
        for kind in ("falling_square", "falling_cylinder"):
            for model in ("svk", "neo_hookean"):
                with self.subTest(kind=kind, model=model):
                    config = small_config(model, kind)
                    builder = build_falling_square if kind == "falling_square" else build_falling_cylinder
                    grid, solid, metadata = builder(config)
                    self.assertEqual(metadata["solid_material_model"], model)
                    _, kwargs = solver_kwargs_from_config(config, grid, solid)
                    initial_u = (np.diag([0.04, -0.01]) @ (solid.P - 0.5)).ravel()
                    kwargs["initial_solid_state"] = {"u_s": initial_u}
                    result = solve_id(**kwargs)
                    self.assertEqual(len(result.usol_s), 4)
                    self.assertTrue(np.all(np.isfinite(result.final_lambda)))
                    self.assertGreater(np.linalg.norm(result.final_lambda), 1e-8)
                    self.assertGreater(np.linalg.norm(result.usol_s[-1] - initial_u), 1e-12)

    def test_linear_default_matches_explicit_linear_force_callback(self):
        config = small_config("linear")
        del config["solid"]["material"]["model"]
        grid, solid, _ = build_falling_square(config)
        self.assertIsNone(solid.internal_force)
        _, kwargs = solver_kwargs_from_config(config, grid, solid)
        initial_u = (np.diag([0.04, -0.01]) @ (solid.P - 0.5)).ravel()
        kwargs["initial_solid_state"] = {"u_s": initial_u}
        original = solve_id(**kwargs)
        solid.internal_force = lambda u: solid.A_s @ u
        callback = solve_id(**kwargs)
        assert_allclose(callback.usol_s, original.usol_s, atol=1e-14)
        assert_allclose(callback.final_lambda, original.final_lambda, atol=1e-14)

    def test_unsupported_combinations_fail_before_solving(self):
        config = small_config()
        grid, solid, _ = build_falling_square(config)
        for method, beta in (("ib", 0.0), ("id", 0.1)):
            invalid = deepcopy(config)
            invalid["case"]["method"] = method
            invalid["coupling"]["rayleigh_beta"] = beta
            with self.assertRaises(ValueError):
                solver_kwargs_from_config(invalid, grid, solid)
        _, kwargs = solver_kwargs_from_config(config, grid, solid)
        kwargs.pop("transfer_kind")
        with self.assertRaisesRegex(ValueError, "only by solve_id"):
            solve_ib(**kwargs)
        kwargs["rayleigh_beta"] = 0.1
        with self.assertRaisesRegex(ValueError, "rayleigh_beta"):
            solve_id(**kwargs)

    def test_material_configuration(self):
        for material in (
            {"model": "neo_hookean", "plane_stress": True},
            {"model": "unknown"},
            {"model": "linear", "penalty": 1.0},
        ):
            with self.subTest(material=material), self.assertRaises(ValueError):
                material_from_config({"material": material})
        grid, solid, _ = build_falling_square(small_config())
        self.assertIs(_coerce_solid_system(vars(solid)).internal_force, solid.internal_force)


if __name__ == "__main__":
    unittest.main()
