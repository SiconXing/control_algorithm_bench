#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "function_env.h"

namespace py = pybind11;

PYBIND11_MODULE(function_env, m) {
    m.doc() = "Function-fitting RL environment with piecewise reward functions";

    py::enum_<FunctionId>(m, "FunctionId")
        .value("STEP_SINGLE", FunctionId::STEP_SINGLE)
        .value("STEP_MULTI", FunctionId::STEP_MULTI)
        .value("SAWTOOTH", FunctionId::SAWTOOTH)
        .value("STAIRCASE", FunctionId::STAIRCASE)
        .value("GAUSSIAN_BUMPS", FunctionId::GAUSSIAN_BUMPS)
        .value("DISCONTINUOUS_JUMP", FunctionId::DISCONTINUOUS_JUMP)
        .value("POLYNOMIAL_CUBIC", FunctionId::POLYNOMIAL_CUBIC)
        .value("SINUSOID_COMPOSITE", FunctionId::SINUSOID_COMPOSITE)
        .value("V_SHAPED_ABSOLUTE", FunctionId::V_SHAPED_ABSOLUTE)
        .value("COMPLEX_MULTI_SEGMENT", FunctionId::COMPLEX_MULTI_SEGMENT)
        .export_values();

    py::class_<FunctionFittingEnv>(m, "FunctionFittingEnv")
        .def(py::init<FunctionId, int, int>(),
             py::arg("func_id"),
             py::arg("n_actions") = 100,
             py::arg("episode_length") = 200,
             "Create a function-fitting RL environment")
        .def("reset", &FunctionFittingEnv::reset,
             "Reset the environment (sample new random x)")
        .def("get_state", &FunctionFittingEnv::get_state,
             "Return normalized [x] state vector")
        .def("step", &FunctionFittingEnv::step, py::arg("action"),
             "Execute an action (discrete y-bin index); returns reward = -|f(x)-y_pred|")
        .def("is_done", &FunctionFittingEnv::is_done,
             "Whether the episode has ended")
        .def("get_score", &FunctionFittingEnv::get_score,
             "Return cumulative reward for current episode")
        .def("get_action_size", &FunctionFittingEnv::get_action_size,
             "Number of discrete y-bins")
        .def("get_state_size", &FunctionFittingEnv::get_state_size,
             "Dimension of the state vector (1)")
        .def("get_function_name", &FunctionFittingEnv::get_function_name,
             "Return the name of the current piecewise function")
        .def("get_y_min", &FunctionFittingEnv::get_y_min,
             "Return the minimum y value in the action range")
        .def("get_y_max", &FunctionFittingEnv::get_y_max,
             "Return the maximum y value in the action range");
}
