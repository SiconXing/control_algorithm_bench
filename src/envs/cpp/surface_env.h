#pragma once

#include <vector>
#include <string>
#include <random>
#include <stdexcept>

namespace surface_env {

// --- Enums ---
enum class SurfaceType : int {
    SPHERE = 0,
    RASTRIGIN = 1,
    ACKLEY = 2,
    ROSENBROCK = 3,
    GRIEWANK = 4,
    LEVY = 5,
    MICHALEWICZ = 6,
    CUSTOM_SIN = 7,
    QUADRATIC_WELL = 8
};

enum class OptMode : int {
    MINIMIZE = 0,
    MAXIMIZE = 1
};

// --- Return struct for step() ---
struct StepResult {
    std::vector<double> observation;
    double reward;
    bool done;
    double function_value;
};

// --- Main environment class ---
class SurfaceOptimizationEnv {
public:
    SurfaceOptimizationEnv(
        int dim = 2,
        SurfaceType surface_type = SurfaceType::SPHERE,
        OptMode mode = OptMode::MINIMIZE,
        double step_size = 0.1,
        double bounds_low = -5.0,
        double bounds_high = 5.0,
        double convergence_tolerance = 1e-4,
        int max_steps = 200,
        unsigned int seed = 0
    );

    // RL interface
    std::vector<double> reset();
    StepResult step(int action);
    StepResult step_continuous(const std::vector<double>& action);

    // Getters
    int get_dim() const { return dim_; }
    int get_action_size() const { return 2 * dim_; }
    double get_optimal_value() const { return optimal_value_; }
    std::vector<double> get_optimal_position() const { return optimal_position_; }
    std::vector<double> get_position() const { return position_; }
    std::string get_function_name() const;
    SurfaceType get_surface_type() const { return surface_type_; }
    OptMode get_mode() const { return mode_; }
    double get_step_size() const { return step_size_; }
    int get_max_steps() const { return max_steps_; }
    double get_bounds_low() const { return bounds_[0]; }
    double get_bounds_high() const { return bounds_[1]; }

    // Configuration
    void set_custom_coefficients(const std::vector<double>& coeffs);
    void set_custom_optimal(const std::vector<double>& pos, double val);

private:
    double evaluate(const std::vector<double>& x) const;
    void compute_optimal();
    void compute_michalewicz_optimal();
    void clamp(std::vector<double>& x) const;

    // Individual surface functions
    double eval_sphere(const std::vector<double>& x) const;
    double eval_rastrigin(const std::vector<double>& x) const;
    double eval_ackley(const std::vector<double>& x) const;
    double eval_rosenbrock(const std::vector<double>& x) const;
    double eval_griewank(const std::vector<double>& x) const;
    double eval_levy(const std::vector<double>& x) const;
    double eval_michalewicz(const std::vector<double>& x) const;
    double eval_custom_sin(const std::vector<double>& x) const;
    double eval_quadratic_well(const std::vector<double>& x) const;

    // Config
    int dim_;
    SurfaceType surface_type_;
    OptMode mode_;
    double step_size_;
    double bounds_[2];
    double convergence_tolerance_;
    int max_steps_;

    // State
    std::vector<double> position_;
    int steps_taken_;

    // Pre-computed optimal solution
    std::vector<double> optimal_position_;
    double optimal_value_;

    // Custom function data
    std::vector<double> custom_coeffs_;  // for CUSTOM_SIN: a_i, b_i, c_i per dim
    std::vector<double> custom_optimal_pos_;
    double custom_optimal_val_;
    bool has_custom_optimal_ = false;

    // RNG
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
};

} // namespace surface_env
