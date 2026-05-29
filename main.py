"""Benchmark RL agents on high-dimensional surface optimization."""

from src.evaluator import BenchmarkSuite


def main():
    suite = BenchmarkSuite(episodes=500, verbose=True)

    suite.run(
        functions=["sphere", "rastrigin", "ackley", "griewank"],
        dims=[5],  # Dimension of the optimization problem, it can be increased into a list
        agents=["random", "hill_climb", "dqn", "ppo"],
        modes=["minimize"],
    )

    # Terminal summary
    print("\n" + suite.summary())

    # Generate plots
    print("\nGenerating plots...")
    paths = suite.plot("results")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # DataFrame export (optional)
    df = suite.to_dataframe()
    if df is not None:
        print("\nBest agent per function (by distance to optimal):")
        best = df.groupby(["function", "dim"])["distance"].min().reset_index()
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
