"""Test script for the surface optimization RL environment."""

from envs import make_env, list_functions


def test_basic(func: str, dim: int = 3, mode: str = "minimize") -> bool:
    """Quick smoke test for a single function."""
    env = make_env(func, dim=dim, mode=mode, max_steps=50)
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
    return True


def test_continuous():
    """Test continuous action mode."""
    env = make_env("sphere", dim=4, action_type="continuous")
    obs, _ = env.reset()
    import numpy as np
    for _ in range(10):
        action = np.random.uniform(-1, 1, size=4)
        obs, reward, done, _, info = env.step(action)
    return True


def test_max_mode():
    """Test maximization mode."""
    env = make_env("sphere", dim=2, mode="maximize", bounds=(-3, 3))
    assert env.optimal_value == 18.0, f"Expected 18, got {env.optimal_value}"
    return True


def test_custom():
    """Test custom_sin with user coefficients."""
    env = make_env("custom_sin", dim=3)
    env.set_custom_coefficients([1.0, 2.0, 0.0, 0.5, 3.0, 1.0, -0.7, 1.0, 1.5])
    env.set_custom_optimal([0.5, 0.3, -0.1], -0.8)
    assert env.optimal_value == -0.8
    return True


def main():
    all_funcs = list_functions()
    print("=" * 70)
    print("Surface Optimization RL Environment — Test Suite")
    print("=" * 70)

    # 1. Test all functions
    print("\n[1] All 9 benchmark functions (discrete, dim=3, minimize)")
    for name in all_funcs:
        try:
            test_basic(name)
            print(f"    {name:20s}  OK")
        except Exception as e:
            print(f"    {name:20s}  FAIL — {e}")

    # 2. Test varying dimensions
    print("\n[2] Varying dimensions (sphere, 1D → 10D)")
    for d in [1, 3, 5, 10]:
        try:
            test_basic("sphere", dim=d)
            print(f"    dim={d:2d}  OK")
        except Exception as e:
            print(f"    dim={d:2d}  FAIL — {e}")

    # 3. Continuous action mode
    print("\n[3] Continuous action mode")
    try:
        test_continuous()
        print("    OK")
    except Exception as e:
        print(f"    FAIL — {e}")

    # 4. Maximization mode
    print("\n[4] Maximization mode (sphere, bounds [-3,3])")
    try:
        test_max_mode()
        print("    OK — optimal_value=18.0 at corner (-3,-3)")
    except Exception as e:
        print(f"    FAIL — {e}")

    # 5. Custom function
    print("\n[5] Custom sinusoidal function")
    try:
        test_custom()
        print("    OK")
    except Exception as e:
        print(f"    FAIL — {e}")

    # 6. Optimal value benchmarks (all functions, minimize, dim=5)
    print("\n[6] Optimal values (minimize, dim=5)")
    for name in all_funcs:
        try:
            bounds = all_funcs[name].get("typical_bounds", (-5, 5))
            env = make_env(name, dim=5, mode="minimize", bounds=bounds)
            opt = env.optimal_value
            pos = env.optimal_position
            print(f"    {name:20s}  opt={opt:10.6f}  pos=[{pos[0]:.3f}, {pos[1]:.3f}, ...]")
        except Exception as e:
            print(f"    {name:20s}  FAIL — {e}")

    # 7. End-to-end: random walk benchmark
    print("\n[7] Random-walk benchmark (200 steps, minimize, dim=5)")
    for name in ["sphere", "rastrigin", "ackley", "rosenbrock", "griewank"]:
        try:
            bounds = all_funcs[name].get("typical_bounds", (-5, 5))
            env = make_env(name, dim=5, mode="minimize", max_steps=200, bounds=bounds)
            obs, _ = env.reset()
            best = float("inf")
            for _ in range(200):
                a = env.action_space.sample()
                obs, _, done, _, info = env.step(a)
                if info["function_value"] < best:
                    best = info["function_value"]
            gap = best - env.optimal_value
            print(f"    {name:20s}  best={best:10.4f}  opt={env.optimal_value:10.4f}  gap={gap:8.4f}")
        except Exception as e:
            print(f"    {name:20s}  FAIL — {e}")

    print("\n" + "=" * 70)
    print("All tests completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
