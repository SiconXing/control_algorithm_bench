"""
Piecewise function registry and Python ground-truth implementations.

The C++ environment (function_env) evaluates these functions natively.
This module provides Python implementations for plotting and analysis.
"""
import math

import function_env


# ── Python ground-truth implementations (for plotting) ──────────────────

def step_single(x):
    return -1.0 if x < 0.5 else 1.0


def step_multi(x):
    if x < 0.3:
        return -1.0
    elif x < 0.5:
        return 0.0
    elif x < 0.7:
        return 0.5
    else:
        return 1.0


def sawtooth(x):
    period = 0.25
    phase = math.fmod(x, period) / period
    return 2.0 * phase - 1.0


def staircase(x):
    level = min(int(x * 5.0), 4)
    return -1.0 + level * 0.5


def gaussian_bumps(x):
    import math
    g1 = math.exp(-((x - 0.3) / 0.1) ** 2) * 0.8
    g2 = math.exp(-((x - 0.7) / 0.15) ** 2) * 1.0
    g3 = math.exp(-((x - 0.5) / 0.08) ** 2) * 0.6
    return g1 + g2 - g3


def discontinuous_jump(x):
    if x < 0.4:
        return math.sin(4.0 * math.pi * x)
    elif x > 0.6:
        return math.cos(4.0 * math.pi * x) * 0.8
    else:
        return 0.5


def polynomial_cubic(x):
    t = x - 0.5
    return 2.0 * t**3 - t


def sinusoid_composite(x):
    return math.sin(6.0 * math.pi * x) * 0.6 + math.cos(2.0 * math.pi * x) * 0.4


def v_shaped_absolute(x):
    return abs(2.0 * x - 1.0) * 1.5 - 0.5 + 0.15 * math.sin(10.0 * math.pi * x)


def complex_multi_segment(x):
    if x < 0.15:
        return math.sin(8.0 * math.pi * x) * 0.5
    elif x < 0.35:
        return -4.0 * (x - 0.25) ** 2 + 0.6
    elif x < 0.5:
        return 1.5 * (x - 0.35) / 0.15 - 0.5
    elif x < 0.7:
        return math.exp(-(x - 0.5) * 5.0) * 0.8
    elif x < 0.85:
        return math.cos(6.0 * math.pi * (x - 0.7)) * 0.7
    else:
        return 2.0 * (x - 0.85) ** 3 * 8.0 - 0.3


# ── Registry mapping enum → metadata + Python function ─────────────────

PY_FUNCTIONS = {
    function_env.FunctionId.STEP_SINGLE: step_single,
    function_env.FunctionId.STEP_MULTI: step_multi,
    function_env.FunctionId.SAWTOOTH: sawtooth,
    function_env.FunctionId.STAIRCASE: staircase,
    function_env.FunctionId.GAUSSIAN_BUMPS: gaussian_bumps,
    function_env.FunctionId.DISCONTINUOUS_JUMP: discontinuous_jump,
    function_env.FunctionId.POLYNOMIAL_CUBIC: polynomial_cubic,
    function_env.FunctionId.SINUSOID_COMPOSITE: sinusoid_composite,
    function_env.FunctionId.V_SHAPED_ABSOLUTE: v_shaped_absolute,
    function_env.FunctionId.COMPLEX_MULTI_SEGMENT: complex_multi_segment,
}

FUNCTION_LIST = [
    {"id": function_env.FunctionId.STEP_SINGLE,           "name": "step_single",           "difficulty": "easy"},
    {"id": function_env.FunctionId.STEP_MULTI,            "name": "step_multi",            "difficulty": "medium"},
    {"id": function_env.FunctionId.SAWTOOTH,              "name": "sawtooth",              "difficulty": "medium"},
    {"id": function_env.FunctionId.STAIRCASE,             "name": "staircase",             "difficulty": "medium"},
    {"id": function_env.FunctionId.GAUSSIAN_BUMPS,        "name": "gaussian_bumps",        "difficulty": "medium"},
    {"id": function_env.FunctionId.DISCONTINUOUS_JUMP,    "name": "discontinuous_jump",    "difficulty": "hard"},
    {"id": function_env.FunctionId.POLYNOMIAL_CUBIC,      "name": "polynomial_cubic",      "difficulty": "easy"},
    {"id": function_env.FunctionId.SINUSOID_COMPOSITE,    "name": "sinusoid_composite",    "difficulty": "medium"},
    {"id": function_env.FunctionId.V_SHAPED_ABSOLUTE,     "name": "v_shaped_absolute",     "difficulty": "medium"},
    {"id": function_env.FunctionId.COMPLEX_MULTI_SEGMENT, "name": "complex_multi_segment", "difficulty": "hard"},
]

FUNCTION_BY_NAME = {f["name"]: f for f in FUNCTION_LIST}
FUNCTION_BY_ID = {f["id"]: f for f in FUNCTION_LIST}


def get_function_names():
    return [f["name"] for f in FUNCTION_LIST]


def get_function_meta_by_name(name: str):
    return FUNCTION_BY_NAME.get(name)
