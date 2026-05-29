"""
envs: High-dimensional surface optimization RL environments.

C++/pybind11 backend with a Gymnasium-compatible Python interface.
"""

from .surface_env import (
    SurfaceEnv,
    make_env,
    make_gym_env,
    list_functions,
    SURFACE_TYPES,
    FUNCTION_INFO,
)

__all__ = [
    "SurfaceEnv",
    "make_env",
    "make_gym_env",
    "list_functions",
    "SURFACE_TYPES",
    "FUNCTION_INFO",
]
