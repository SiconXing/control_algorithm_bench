#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "surface_env.h"

namespace py = pybind11;
using namespace surface_env;

PYBIND11_MODULE(_surface_env_cpp, m) {
    m.doc() = "High-dimensional surface optimization environment (C++ backend)";

    // --- SurfaceType enum ---
    py::enum_<SurfaceType>(m, "SurfaceType", "Type of benchmark surface function")
        .value("SPHERE",          SurfaceType::SPHERE)
        .value("RASTRIGIN",       SurfaceType::RASTRIGIN)
        .value("ACKLEY",          SurfaceType::ACKLEY)
        .value("ROSENBROCK",      SurfaceType::ROSENBROCK)
        .value("GRIEWANK",        SurfaceType::GRIEWANK)
        .value("LEVY",            SurfaceType::LEVY)
        .value("MICHALEWICZ",     SurfaceType::MICHALEWICZ)
        .value("CUSTOM_SIN",      SurfaceType::CUSTOM_SIN)
        .value("QUADRATIC_WELL",  SurfaceType::QUADRATIC_WELL)
        .export_values();

    // --- OptMode enum ---
    py::enum_<OptMode>(m, "OptMode", "Optimization direction")
        .value("MINIMIZE", OptMode::MINIMIZE)
        .value("MAXIMIZE", OptMode::MAXIMIZE)
        .export_values();

    // --- SurfaceOptimizationEnv class ---
    py::class_<SurfaceOptimizationEnv>(m, "SurfaceOptimizationEnv",
        "RL environment for navigating high-dimensional surfaces.")
        .def(py::init<int, SurfaceType, OptMode, double, double, double,
                      double, int, unsigned int>(),
             py::arg("dim") = 2,
             py::arg("surface_type") = SurfaceType::SPHERE,
             py::arg("mode") = OptMode::MINIMIZE,
             py::arg("step_size") = 0.1,
             py::arg("bounds_low") = -5.0,
             py::arg("bounds_high") = 5.0,
             py::arg("convergence_tolerance") = 1e-4,
             py::arg("max_steps") = 200,
             py::arg("seed") = 0,
             "Create a surface optimization environment.\n\n"
             "Args:\n"
             "    dim: Number of dimensions\n"
             "    surface_type: Which benchmark function to use\n"
             "    mode: MINIMIZE or MAXIMIZE\n"
             "    step_size: Movement step size per action\n"
             "    bounds_low/bounds_high: Domain bounds for each dimension\n"
             "    convergence_tolerance: Distance to optimal for early stopping\n"
             "    max_steps: Maximum steps per episode\n"
             "    seed: Random seed (0 = random from device)")

        // RL interface
        .def("reset", &SurfaceOptimizationEnv::reset,
             "Reset the environment. Returns initial observation.")
        .def("step", [](SurfaceOptimizationEnv& self, int action) {
             auto r = self.step(action);
             return py::make_tuple(r.observation, r.reward, r.done, r.function_value);
             },
             py::arg("action"),
             "Take a discrete action. Returns (obs, reward, done, value).\n"
             "action: int in [0, 2*dim), even=+step along axis i, odd=-step.")
        .def("step_continuous", [](SurfaceOptimizationEnv& self, const std::vector<double>& action) {
             auto r = self.step_continuous(action);
             return py::make_tuple(r.observation, r.reward, r.done, r.function_value);
             },
             py::arg("action"),
             "Take a continuous action (N-dim vector). Returns (obs, reward, done, value).")

        // Getters
        .def_property_readonly("dim", &SurfaceOptimizationEnv::get_dim)
        .def_property_readonly("action_size", &SurfaceOptimizationEnv::get_action_size)
        .def_property_readonly("optimal_value", &SurfaceOptimizationEnv::get_optimal_value)
        .def_property_readonly("optimal_position", &SurfaceOptimizationEnv::get_optimal_position)
        .def_property_readonly("position", &SurfaceOptimizationEnv::get_position)
        .def_property_readonly("function_name", &SurfaceOptimizationEnv::get_function_name)
        .def_property_readonly("surface_type", &SurfaceOptimizationEnv::get_surface_type)
        .def_property_readonly("mode", &SurfaceOptimizationEnv::get_mode)
        .def_property_readonly("step_size", &SurfaceOptimizationEnv::get_step_size)
        .def_property_readonly("max_steps", &SurfaceOptimizationEnv::get_max_steps)
        .def_property_readonly("bounds", [](const SurfaceOptimizationEnv& e) {
            return py::make_tuple(e.get_bounds_low(), e.get_bounds_high());
        })

        // Configuration
        .def("set_custom_coefficients", &SurfaceOptimizationEnv::set_custom_coefficients,
             py::arg("coeffs"),
             "Set coefficients for CUSTOM_SIN surface type.")
        .def("set_custom_optimal", &SurfaceOptimizationEnv::set_custom_optimal,
             py::arg("position"), py::arg("value"),
             "Manually set the optimal point and value.");
}
