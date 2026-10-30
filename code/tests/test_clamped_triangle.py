"""Exact triangle geometry, exported mesh and constant-inlet ID checks."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.run_case import build_clamped_triangle, load_config, run_case, solver_kwargs_from_config
from immersed_fsi.clamped_triangle import clamped_triangle_mesh
from immersed_fsi.fluid import fluid_assembly_traction_free
from immersed_fsi.geometry import dirichlet_nodes_from_bdry, triangle_areas
from immersed_fsi.solvers import solve_id

CONFIG = Path(__file__).resolve().parents[1] / "cases/configs/clamped_triangle_id_neo_hookean.yaml"


class TriangleSetupTests(unittest.TestCase):
    def test_exact_vertices_area_symmetry_and_root_constraints(self):
        P, T, bdry, metadata = clamped_triangle_mesh()
        self.assertEqual(P.shape, (2, 91))
        self.assertEqual(T.shape, (144, 3))
        for vertex in ([2.5, 0], [3, 0], [2.75, 1.5]):
            self.assertTrue(np.any(np.all(np.isclose(P.T, vertex), axis=1)))
        areas = triangle_areas(P, T)
        self.assertTrue(np.all(areas > 0))
        assert_allclose(areas.sum(), 0.375)
        assert_allclose(areas, 0.375 / 144)
        self.assertTrue(np.all(areas >= 0.05**2))
        self.assertEqual(metadata["solid_subdivisions"], 12)
        roots = dirichlet_nodes_from_bdry(bdry)
        self.assertEqual(len(roots), 13)
        assert_allclose(P[1, roots], 0)
        self.assertNotIn(metadata["solid_tip_node"], roots)
        points = {tuple(np.round(p, 12)) for p in P.T}
        mirrored = {tuple(np.round([5.5 - p[0], p[1]], 12)) for p in P.T}
        self.assertEqual(points, mirrored)

    def test_invalid_geometry_is_rejected(self):
        for kwargs in ({"base_x": [3, 2.5]}, {"height": -1}, {"min_element_area": 0},
                       {"min_element_area": 0.4}, {"apex_x": 3.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                clamped_triangle_mesh(**kwargs)
        config = load_config(CONFIG)
        config["solid"]["height"] = 4
        with self.assertRaisesRegex(ValueError, "fit inside"):
            build_clamped_triangle(config)

    def test_export_contains_exact_solid_and_fluid_grid(self):
        with tempfile.TemporaryDirectory(prefix="triangle-mesh-") as folder:
            mesh = Path(folder) / "mesh.npz"
            run_case(CONFIG, dry_run=True, mesh_path=mesh)
            with np.load(mesh) as arrays:
                self.assertEqual(arrays["P"].shape, (2, 91))
                self.assertEqual(arrays["T"].shape, (144, 3))
                assert_allclose(arrays["fluid_x"], np.linspace(0, 5, 101))
                assert_allclose(arrays["fluid_y"], np.linspace(0, 3, 61))
                assert_allclose(arrays["P"][1, arrays["fixed_nodes"]], 0)
                fluid_area = np.diff(arrays["fluid_x"]).max() * np.diff(arrays["fluid_y"]).max()
                self.assertTrue(np.all(triangle_areas(arrays["P"], arrays["T"]) >= fluid_area))

    def test_area_constraint_follows_actual_fluid_grid_and_requested_ratio(self):
        for resolution, ratio in ((0.025, 1.0), (0.07, 1.0), (0.1, 1.0), (0.05, 2.0)):
            with self.subTest(resolution=resolution, ratio=ratio):
                config = load_config(CONFIG)
                config["fluid"]["resolution"] = resolution
                config["solid"]["min_area_ratio"] = ratio
                grid, solid, metadata = build_clamped_triangle(config)
                minimum = grid["dx"] * grid["dy"] * ratio
                areas = triangle_areas(solid.P, solid.T)
                self.assertTrue(np.all(areas >= minimum))
                assert_allclose(areas.sum(), 0.375)
                # One further regular subdivision would violate the bound:
                # the selected mesh is the finest regular admissible mesh.
                n = metadata["solid_subdivisions"]
                self.assertLess(0.375 / (n + 1)**2, minimum)
        # Exact equality is admissible as well as strictly larger elements.
        P, T, _, _ = clamped_triangle_mesh(min_element_area=0.375 / 4)
        self.assertEqual(len(T), 4)
        assert_allclose(triangle_areas(P, T), 0.375 / 4, rtol=0, atol=0)
        for ratio in (0.5, float("nan")):
            config = load_config(CONFIG)
            config["solid"]["min_area_ratio"] = ratio
            with self.assertRaisesRegex(ValueError, "min_area_ratio"):
                build_clamped_triangle(config)

    def test_constant_inlet_neumann_outlet_and_nonlinear_id_startup(self):
        config = load_config(CONFIG)
        config["time"].update(Nt=5, history_stride=1)
        config["output"]["verbose"] = False
        grid, solid, metadata = build_clamped_triangle(config)
        self.assertEqual((grid["nx"], grid["ny"]), (100, 60))
        self.assertEqual(metadata["solid_material_model"], "neo_hookean")
        method, kwargs = solver_kwargs_from_config(config, grid, solid)
        self.assertEqual(method, "id")
        self.assertIs(kwargs["fluid_assembly_fn"], fluid_assembly_traction_free)
        self.assertNotIn("inlet_velocity_fn", kwargs)
        assert_allclose(kwargs["vel0_f"], 1)
        result = solve_id(**kwargs)
        self.assertEqual(len(result.usol_s), 6)
        assert_allclose(result.usol_f[-1][1:-1, 1], 1)
        roots = dirichlet_nodes_from_bdry(solid.bdry)
        assert_allclose(result.usol_s[-1].reshape(2, -1)[:, roots], 0)
        self.assertGreater(np.linalg.norm(result.usol_s[-1]), 0)
        u, v, p = result.usol_f[-1], result.vsol_f[-1], result.psol_f[-1]
        divergence = np.diff(u[1:-1, 1:], axis=1) / grid["dx"] + np.diff(v[1:, 1:-1], axis=0) / grid["dy"]
        assert_allclose(divergence, 0, atol=1e-9)
        normal = -0.5 * (p[1:-1, -1] + p[1:-1, -2]) + 2 * grid["mu"] / grid["rho"] * (u[1:-1, -1] - u[1:-1, -2]) / grid["dx"]
        shear = np.diff(u[1:-1, -1]) / grid["dy"] + (v[2:-1, -1] - v[2:-1, -2]) / grid["dx"]
        assert_allclose(normal, 0, atol=1e-10)
        assert_allclose(shear, 0, atol=1e-10)
        assert_allclose(u[0] + u[1], 0)
        assert_allclose(u[-1] + u[-2], 0)
        assert_allclose(v[[1, -1]], 0)


if __name__ == "__main__":
    unittest.main()
