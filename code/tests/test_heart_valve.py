"""Paper geometry/inflow and discrete traction-free channel verification."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.run_case import build_idealized_heart_valve, load_config, save_case_result, solver_kwargs_from_config
from immersed_fsi.fluid import (
    fluid_assembly_CF, fluid_assembly_traction_free,
    fluid_solver_CF, fluid_solver_CF_debug,
)
from immersed_fsi.geometry import dirichlet_nodes_from_bdry
from immersed_fsi.heart_valve import heart_valve_inlet, idealized_heart_valve_mesh
from immersed_fsi.solvers import solve_id


CONFIG = Path(__file__).resolve().parents[1] / "cases/configs/idealized_heart_valve_id_neo_hookean.yaml"


def divergence(u, v, dx, dy):
    return np.diff(u[1:-1, 1:], axis=1) / dx + np.diff(v[1:, 1:-1], axis=0) / dy


class HeartValveGeometryTests(unittest.TestCase):
    def test_paper_dimensions_shared_resolution_and_clamps(self):
        P, T, bdry, meta = idealized_heart_valve_mesh(0.02)
        self.assertEqual((meta["fluid_nx"], meta["fluid_ny"]), (400, 81))
        self.assertEqual(P.shape, (2, 144))
        self.assertEqual(T.shape, (70, 4))
        self.assertEqual((meta["solid_elements_thickness"], meta["solid_elements_length"]), (1, 35))
        assert_allclose([P[0].min(), P[0].max(), P[1].min(), P[1].max()], [2, 2.0212, 0, 1.61])
        assert_allclose(P[1, meta["lower_tip_nodes"]], 0.70)
        assert_allclose(P[1, meta["upper_tip_nodes"]], 0.91)
        roots = dirichlet_nodes_from_bdry(bdry)
        self.assertEqual(len(roots), 4)
        assert_allclose(np.sort(P[1, roots]), [0, 0, 1.61, 1.61])
        self.assertTrue(set(T[:35].ravel()).isdisjoint(set(T[35:].ravel())))
        for indices in T:
            vertices = P[:, indices]
            area = 0.5 * np.sum(vertices[0] * np.roll(vertices[1], -1) - vertices[1] * np.roll(vertices[0], -1))
            self.assertGreater(area, 0.0)

    def test_invalid_geometry_and_conflicting_resolution(self):
        for kwargs in ({"resolution": 0.0}, {"resolution": 0.02, "leaflet_length": 0.9},
                       {"resolution": 0.02, "attachment_x": -0.1}):
            with self.assertRaises(ValueError):
                idealized_heart_valve_mesh(**kwargs)
        config = load_config(CONFIG)
        config["solid"]["resolution"] = 0.04
        with self.assertRaisesRegex(ValueError, "shared"):
            build_idealized_heart_valve(config)

    def test_equation_135_initial_state_peak_period_and_flow_rate(self):
        H = 1.61
        y = np.linspace(0, H, 101)
        assert_allclose(heart_valve_inlet(0, y=y), 0.0)
        assert_allclose(heart_valve_inlet(0.25, y=y), 10.5 * y * (H - y))
        assert_allclose(heart_valve_inlet(0.25, y=y), heart_valve_inlet(1.25, y=y))
        assert_allclose(heart_valve_inlet(0.75, y=H / 2), 0.5 * H**2 / 4)
        # Gauss integration checks the analytical inflow rate A(t)*H^3/6.
        q, w = np.polynomial.legendre.leggauss(3)
        flow = H / 2 * np.dot(w, heart_valve_inlet(0.25, y=(q + 1) * H / 2))
        assert_allclose(flow, 10.5 * H**3 / 6)


class TractionFreeTests(unittest.TestCase):
    def test_outlet_velocity_increment_vanishes_with_time_step(self):
        # A smooth divergence-free field whose outlet velocity differs from
        # 2*u_i-u_{i-1}. Replacing its value by that extrapolation produces a
        # finite jump as dt -> 0; advancing its momentum must instead give O(dt).
        nx, ny = 12, 8
        dx = dy = 0.1
        X, Y = np.meshgrid(np.arange(nx + 1) * dx, np.arange(ny + 1) * dy)
        psi = 0.01 * (X / (nx * dx))**2 * np.sin(np.pi * Y / (ny * dy))**2
        u0, v0 = np.zeros((2, ny + 2, nx + 2))
        u0[1:-1, 1:] = np.diff(psi, axis=0) / dy
        v0[1:, 1:-1] = -np.diff(psi, axis=1) / dx
        assert_allclose(divergence(u0, v0, dx, dy), 0.0, atol=1e-14)
        self.assertLess(u0[1:-1, -1].min(), 0.0)  # Includes local backflow.
        increments = []
        for dt in (1e-5, 1e-6):
            Af, bf = fluid_assembly_traction_free(nx, ny, dx, dy, dt=dt, nu=0.1)
            u, v, _ = fluid_solver_CF(
                u0.copy(), v0.copy(), nx, ny, dx, dy, dt, 0.1,
                np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx)),
                Af, bf, np.zeros(ny), 100, traction_free=True,
            )
            increment = np.concatenate(((u - u0)[1:-1, 1:].ravel(), (v - v0)[1:, 1:-1].ravel()))
            increments.append(np.linalg.norm(increment))
        self.assertGreater(increments[0], 0.0)
        self.assertLess(increments[1], 0.11 * increments[0])

    def test_projection_incompressibility_traction_and_flux(self):
        nx, ny = 13, 7
        dx, dy, dt, nu = 0.12, 0.1, 0.001, 0.1
        rng = np.random.default_rng(320)
        u, v = rng.normal(scale=0.01, size=(2, ny + 2, nx + 2))
        Fx = rng.normal(size=(ny, nx + 1))
        Fy = rng.normal(size=(ny + 1, nx))
        inlet = heart_valve_inlet(0.13, y=(np.arange(ny) + 0.5) * dy, height=ny * dy)
        Af, bf = fluid_assembly_traction_free(nx, ny, dx, dy, dt=dt, nu=nu)
        for _ in range(3):
            u, v, p = fluid_solver_CF(u, v, nx, ny, dx, dy, dt, nu, Fx, Fy, Af, bf, inlet, 100, traction_free=True)
            assert_allclose(divergence(u, v, dx, dy), 0.0, atol=2e-11)
            p_boundary = 0.5 * (p[1:-1, -1] + p[1:-1, -2])
            normal_traction = -p_boundary + 2 * nu * (u[1:-1, -1] - u[1:-1, -2]) / dx
            shear_rate = np.diff(u[1:-1, -1]) / dy + (v[2:-1, -1] - v[2:-1, -2]) / dx
            assert_allclose(normal_traction, 0.0, atol=2e-11)
            assert_allclose(shear_rate, 0.0, atol=2e-11)
            assert_allclose(u[1:-1, 1], inlet)
            assert_allclose(u[1:-1, -1].sum(), inlet.sum(), atol=2e-11)
            assert_allclose(u[0] + u[1], 0.0)
            assert_allclose(u[-1] + u[-2], 0.0)
            assert_allclose(v[[1, -1]], 0.0)

    def test_existing_channel_matches_debug_reference(self):
        nx, ny = 13, 7
        dx, dy, dt, nu = 0.12, 0.1, 0.001, 0.1
        rng = np.random.default_rng(42)
        u, v = rng.normal(scale=0.01, size=(2, ny + 2, nx + 2))
        Fx, Fy = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
        Af, bf = fluid_assembly_CF(nx, ny, dx, dy, 0)
        args = (nx, ny, dx, dy, dt, nu, Fx, Fy, Af, bf, np.ones(ny), 100)
        expected = fluid_solver_CF_debug(u.copy(), v.copy(), *args)
        actual = fluid_solver_CF(u.copy(), v.copy(), *args)
        for old, new in zip(expected, actual):
            assert_allclose(new, old, atol=1e-12)

    def test_mismatched_pressure_solver_is_rejected(self):
        nx, ny = 13, 7
        u, v = np.zeros((2, ny + 2, nx + 2))
        Af, bf = fluid_assembly_CF(nx, ny, 0.1, 0.1, 0)
        with self.assertRaisesRegex(ValueError, "matching"):
            fluid_solver_CF(u, v, nx, ny, 0.1, 0.1, 0.001, 0.1,
                            np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx)),
                            Af, bf, np.zeros(ny), 100, traction_free=True)


class HeartValveCouplingTests(unittest.TestCase):
    def test_actual_paper_parameters_and_mesh_start_with_pulsatile_id(self):
        config = load_config(CONFIG)
        config["time"].update(Nt=3, history_stride=1)
        config["output"]["verbose"] = False
        grid, solid, meta = build_idealized_heart_valve(config)
        assert_allclose([meta["solid_lambda"], meta["solid_mu"]], [8e7, 2e7])
        assert_allclose([grid["rho"], meta["solid_rho"], grid["mu"]], [100, 100, 10])
        _, kwargs = solver_kwargs_from_config(config, grid, solid)
        times = []
        inlet = kwargs["inlet_velocity_fn"]

        def record(time):
            times.append(time)
            return inlet(time)

        kwargs["inlet_velocity_fn"] = record
        result = solve_id(**kwargs)
        assert_allclose(times, [1e-4, 2e-4, 3e-4])
        assert_allclose(result.usol_f[-1][1:-1, 1], inlet(3e-4))
        assert_allclose(result.usol_f[0], 0.0)
        roots = dirichlet_nodes_from_bdry(solid.bdry)
        assert_allclose(result.usol_s[-1].reshape(2, -1)[:, roots], 0.0)
        self.assertGreater(np.linalg.norm(result.usol_s[-1]), 1e-8)
        self.assertTrue(np.all(np.isfinite(result.final_lambda)))
        assert_allclose(divergence(result.usol_f[-1], result.vsol_f[-1], grid["dx"], grid["dy"]), 0.0, atol=1e-9)
        for step in range(3):
            final = [row for row in result.iteration_history if row["step"] == step][-1]
            self.assertLessEqual(final["residual_error"], config["coupling"]["tol"])
        # A short run ending before the history stride must retain its actual
        # final step/time, rather than inheriting the full YAML duration.
        for name in ("usol_f", "vsol_f", "psol_f", "usol_s", "vsol_s", "asol_s"):
            history = getattr(result, name)
            setattr(result, name, [history[0], history[-1]])
        config["time"]["history_stride"] = 100
        with tempfile.TemporaryDirectory(prefix="heart-valve-results-") as folder:
            config["output"]["folder"] = folder
            saved = save_case_result(CONFIG, config, result, grid, meta)
            self.assertEqual(saved.params["Nt"], 3)
            self.assertEqual(saved.params["dt"], 1e-4)
            with np.load(saved.npz_path) as arrays:
                assert_allclose(arrays["frame_indices"], [0, 3])


if __name__ == "__main__":
    unittest.main()
