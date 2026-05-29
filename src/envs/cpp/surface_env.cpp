#include "surface_env.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>

namespace surface_env {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double E  = 2.71828182845904523536;
}

// ============================================================================
// Constructor
// ============================================================================
SurfaceOptimizationEnv::SurfaceOptimizationEnv(
    int dim, SurfaceType surface_type, OptMode mode,
    double step_size, double bounds_low, double bounds_high,
    double convergence_tolerance, int max_steps, unsigned int seed)
    : dim_(dim),
      surface_type_(surface_type),
      mode_(mode),
      step_size_(step_size),
      convergence_tolerance_(convergence_tolerance),
      max_steps_(max_steps),
      position_(dim),
      steps_taken_(0),
      optimal_value_(0.0),
      rng_(seed == 0 ? std::random_device{}() : seed)
{
    if (dim_ < 1)
        throw std::invalid_argument("Dimension must be >= 1");
    if (bounds_low >= bounds_high)
        throw std::invalid_argument("bounds_low must be < bounds_high");

    bounds_[0] = bounds_low;
    bounds_[1] = bounds_high;
    uniform_dist_ = std::uniform_real_distribution<double>(bounds_[0], bounds_[1]);

    compute_optimal();
    reset();
}

// ============================================================================
// RL Interface
// ============================================================================
std::vector<double> SurfaceOptimizationEnv::reset() {
    steps_taken_ = 0;
    for (int i = 0; i < dim_; ++i) {
        position_[i] = uniform_dist_(rng_);
    }
    return position_;
}

StepResult SurfaceOptimizationEnv::step(int action) {
    if (action < 0 || action >= 2 * dim_) {
        throw std::out_of_range("Action must be in [0, " +
            std::to_string(2 * dim_ - 1) + "], got " + std::to_string(action));
    }

    int dim_idx = action / 2;
    double dir = (action % 2 == 0) ? 1.0 : -1.0;

    std::vector<double> new_pos = position_;
    new_pos[dim_idx] += dir * step_size_;
    clamp(new_pos);
    position_ = new_pos;
    steps_taken_++;

    double fval = evaluate(position_);

    double reward = (mode_ == OptMode::MINIMIZE) ? -fval : fval;

    bool done = (steps_taken_ >= max_steps_);
    if (std::abs(fval - optimal_value_) < convergence_tolerance_)
        done = true;

    return {position_, reward, done, fval};
}

StepResult SurfaceOptimizationEnv::step_continuous(const std::vector<double>& action) {
    if (static_cast<int>(action.size()) != dim_) {
        throw std::invalid_argument("Action vector size must match dimension");
    }

    std::vector<double> new_pos = position_;
    for (int i = 0; i < dim_; ++i) {
        new_pos[i] += action[i] * step_size_;
    }
    clamp(new_pos);
    position_ = new_pos;
    steps_taken_++;

    double fval = evaluate(position_);

    double reward = (mode_ == OptMode::MINIMIZE) ? -fval : fval;

    bool done = (steps_taken_ >= max_steps_);
    if (std::abs(fval - optimal_value_) < convergence_tolerance_)
        done = true;

    return {position_, reward, done, fval};
}

// ============================================================================
// Getters
// ============================================================================
std::string SurfaceOptimizationEnv::get_function_name() const {
    switch (surface_type_) {
        case SurfaceType::SPHERE:          return "sphere";
        case SurfaceType::RASTRIGIN:       return "rastrigin";
        case SurfaceType::ACKLEY:          return "ackley";
        case SurfaceType::ROSENBROCK:      return "rosenbrock";
        case SurfaceType::GRIEWANK:        return "griewank";
        case SurfaceType::LEVY:            return "levy";
        case SurfaceType::MICHALEWICZ:     return "michalewicz";
        case SurfaceType::CUSTOM_SIN:      return "custom_sin";
        case SurfaceType::QUADRATIC_WELL:  return "quadratic_well";
    }
    return "unknown";
}

// ============================================================================
// Configuration
// ============================================================================
void SurfaceOptimizationEnv::set_custom_coefficients(const std::vector<double>& coeffs) {
    custom_coeffs_ = coeffs;
    if (surface_type_ == SurfaceType::CUSTOM_SIN && !has_custom_optimal_) {
        compute_optimal();  // re-run with defaults if user hasn't set optimal
    }
}

void SurfaceOptimizationEnv::set_custom_optimal(const std::vector<double>& pos, double val) {
    custom_optimal_pos_ = pos;
    custom_optimal_val_ = val;
    has_custom_optimal_ = true;
    if (surface_type_ == SurfaceType::CUSTOM_SIN) {
        optimal_position_ = pos;
        optimal_value_ = val;
    }
}

// ============================================================================
// Utility
// ============================================================================
void SurfaceOptimizationEnv::clamp(std::vector<double>& x) const {
    for (auto& v : x) {
        if (v < bounds_[0]) v = bounds_[0];
        if (v > bounds_[1]) v = bounds_[1];
    }
}

// ============================================================================
// Master evaluate dispatcher
// ============================================================================
double SurfaceOptimizationEnv::evaluate(const std::vector<double>& x) const {
    switch (surface_type_) {
        case SurfaceType::SPHERE:          return eval_sphere(x);
        case SurfaceType::RASTRIGIN:       return eval_rastrigin(x);
        case SurfaceType::ACKLEY:          return eval_ackley(x);
        case SurfaceType::ROSENBROCK:      return eval_rosenbrock(x);
        case SurfaceType::GRIEWANK:        return eval_griewank(x);
        case SurfaceType::LEVY:            return eval_levy(x);
        case SurfaceType::MICHALEWICZ:     return eval_michalewicz(x);
        case SurfaceType::CUSTOM_SIN:      return eval_custom_sin(x);
        case SurfaceType::QUADRATIC_WELL:  return eval_quadratic_well(x);
    }
    return 0.0;
}

// ============================================================================
// Surface Function Implementations
// ============================================================================

// f(x) = sum(x_i^2)
// Global minimum: f(0,...,0) = 0
double SurfaceOptimizationEnv::eval_sphere(const std::vector<double>& x) const {
    double sum = 0.0;
    for (double xi : x) sum += xi * xi;
    return sum;
}

// f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
// Global minimum: f(0,...,0) = 0
double SurfaceOptimizationEnv::eval_rastrigin(const std::vector<double>& x) const {
    double sum = 10.0 * dim_;
    for (double xi : x) sum += xi * xi - 10.0 * std::cos(2.0 * PI * xi);
    return sum;
}

// f(x) = -20*exp(-0.2*sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + 20 + e
// Global minimum: f(0,...,0) = 0
double SurfaceOptimizationEnv::eval_ackley(const std::vector<double>& x) const {
    double sum_sq = 0.0, sum_cos = 0.0;
    for (double xi : x) {
        sum_sq += xi * xi;
        sum_cos += std::cos(2.0 * PI * xi);
    }
    double n = static_cast<double>(dim_);
    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n));
    double term2 = -std::exp(sum_cos / n);
    return term1 + term2 + 20.0 + E;
}

// f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2]
// Global minimum: f(1,...,1) = 0
double SurfaceOptimizationEnv::eval_rosenbrock(const std::vector<double>& x) const {
    double sum = 0.0;
    for (int i = 0; i < dim_ - 1; ++i) {
        double t1 = x[i + 1] - x[i] * x[i];
        double t2 = 1.0 - x[i];
        sum += 100.0 * t1 * t1 + t2 * t2;
    }
    return sum;
}

// f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i / sqrt(i+1)))
// Global minimum: f(0,...,0) = 0
double SurfaceOptimizationEnv::eval_griewank(const std::vector<double>& x) const {
    double sum = 0.0, prod = 1.0;
    for (int i = 0; i < dim_; ++i) {
        sum += x[i] * x[i];
        prod *= std::cos(x[i] / std::sqrt(static_cast<double>(i + 1)));
    }
    return 1.0 + sum / 4000.0 - prod;
}

// Levy function N.13
// w_i = 1 + (x_i - 1)/4
// f(x) = sin^2(pi*w_1) + sum_{i=1}^{n-1}(w_i-1)^2*[1+10*sin^2(pi*w_i+1)] + (w_n-1)^2*[1+sin^2(2*pi*w_n)]
// Global minimum: f(1,...,1) = 0
double SurfaceOptimizationEnv::eval_levy(const std::vector<double>& x) const {
    std::vector<double> w(dim_);
    for (int i = 0; i < dim_; ++i)
        w[i] = 1.0 + (x[i] - 1.0) / 4.0;

    double sum = 0.0;
    double w1 = w[0];
    sum += std::sin(PI * w1) * std::sin(PI * w1);

    for (int i = 0; i < dim_ - 1; ++i) {
        double wi = w[i];
        double term1 = (wi - 1.0) * (wi - 1.0);
        double term2 = 1.0 + 10.0 * std::sin(PI * wi + 1.0) * std::sin(PI * wi + 1.0);
        sum += term1 * term2;
    }

    double wn = w[dim_ - 1];
    sum += (wn - 1.0) * (wn - 1.0) * (1.0 + std::sin(2.0 * PI * wn) * std::sin(2.0 * PI * wn));

    return sum;
}

// f(x) = -sum(sin(x_i) * sin^{2m}((i+1)*x_i^2 / pi))
// Domain: [0, pi] typically; m=10
// Separable: each dimension optimized independently
double SurfaceOptimizationEnv::eval_michalewicz(const std::vector<double>& x) const {
    constexpr double m = 10.0;
    double sum = 0.0;
    for (int i = 0; i < dim_; ++i) {
        double inner = std::sin(static_cast<double>(i + 1) * x[i] * x[i] / PI);
        sum += std::sin(x[i]) * std::pow(inner, 2.0 * m);
    }
    return -sum;
}

// Custom sinusoidal surface
// If custom_coeffs_ has 3*dim entries: f(x) = sum(a_i * sin(b_i * x_i + c_i))
// If custom_coeffs_ has dim entries: f(x) = sum(coeff_i * sin(x_i))
// If empty: f(x) = sum(sin(x_i))
double SurfaceOptimizationEnv::eval_custom_sin(const std::vector<double>& x) const {
    double sum = 0.0;
    if (custom_coeffs_.size() >= static_cast<size_t>(3 * dim_)) {
        for (int i = 0; i < dim_; ++i) {
            double a = custom_coeffs_[3 * i];
            double b = custom_coeffs_[3 * i + 1];
            double c = custom_coeffs_[3 * i + 2];
            sum += a * std::sin(b * x[i] + c);
        }
    } else if (custom_coeffs_.size() >= static_cast<size_t>(dim_)) {
        for (int i = 0; i < dim_; ++i)
            sum += custom_coeffs_[i] * std::sin(x[i]);
    } else {
        for (double xi : x) sum += std::sin(xi);
    }
    return sum;
}

// f(x) = sum((x_i - center_i)^2)
// Default center = midpoint of bounds
double SurfaceOptimizationEnv::eval_quadratic_well(const std::vector<double>& x) const {
    double center = (bounds_[0] + bounds_[1]) / 2.0;
    double sum = 0.0;
    for (double xi : x) {
        double d = xi - center;
        sum += d * d;
    }
    return sum;
}

// ============================================================================
// Optimal Solution Computation
// ============================================================================
void SurfaceOptimizationEnv::compute_optimal() {
    double center = (bounds_[0] + bounds_[1]) / 2.0;

    if (mode_ == OptMode::MAXIMIZE) {
        // For maximization: find the maximum within bounds
        switch (surface_type_) {
            case SurfaceType::SPHERE:
            case SurfaceType::RASTRIGIN:
            case SurfaceType::ACKLEY:
            case SurfaceType::GRIEWANK:
            case SurfaceType::ROSENBROCK:
            case SurfaceType::LEVY:
            case SurfaceType::QUADRATIC_WELL: {
                // For these functions, max is at a boundary corner.
                // Check all corners if dim is small; otherwise MC sample.
                if (dim_ <= 14) {
                    int n_corners = 1 << dim_;
                    double best_val = -std::numeric_limits<double>::infinity();
                    std::vector<double> best_pos(dim_);
                    for (int mask = 0; mask < n_corners; ++mask) {
                        std::vector<double> corner(dim_);
                        for (int i = 0; i < dim_; ++i)
                            corner[i] = (mask & (1 << i)) ? bounds_[1] : bounds_[0];
                        double v = evaluate(corner);
                        if (v > best_val) { best_val = v; best_pos = corner; }
                    }
                    optimal_position_ = best_pos;
                    optimal_value_ = best_val;
                } else {
                    // Monte Carlo for high dimensions
                    double best_val = -std::numeric_limits<double>::infinity();
                    std::vector<double> best_pos(dim_);
                    std::uniform_real_distribution<double> dist(bounds_[0], bounds_[1]);
                    for (int k = 0; k < 200000; ++k) {
                        std::vector<double> pt(dim_);
                        for (int i = 0; i < dim_; ++i) pt[i] = dist(rng_);
                        double v = evaluate(pt);
                        if (v > best_val) { best_val = v; best_pos = pt; }
                    }
                    optimal_position_ = best_pos;
                    optimal_value_ = best_val;
                }
                break;
            }
            case SurfaceType::MICHALEWICZ: {
                // Michalewicz is separable; grid-search per dim for MAX
                constexpr double m = 10.0;
                constexpr int grid_pts = 10000;
                optimal_position_.resize(dim_);
                double total_opt = 0.0;
                for (int i = 0; i < dim_; ++i) {
                    double best_term = -std::numeric_limits<double>::infinity();
                    double best_x = 0.0;
                    for (int j = 0; j <= grid_pts; ++j) {
                        double x = PI * static_cast<double>(j) / static_cast<double>(grid_pts);
                        double inner = std::sin(static_cast<double>(i + 1) * x * x / PI);
                        double val = -std::sin(x) * std::pow(inner, 2.0 * m);
                        if (val > best_term) { best_term = val; best_x = x; }
                    }
                    optimal_position_[i] = best_x;
                    total_opt += best_term;
                }
                optimal_value_ = total_opt;
                break;
            }
            case SurfaceType::CUSTOM_SIN: {
                if (has_custom_optimal_) {
                    optimal_position_ = custom_optimal_pos_;
                    optimal_value_ = custom_optimal_val_;
                } else {
                    // Sample for max
                    optimal_position_.assign(dim_, bounds_[1]);
                    optimal_value_ = evaluate(optimal_position_);
                    std::uniform_real_distribution<double> dist(bounds_[0], bounds_[1]);
                    for (int k = 0; k < 50000; ++k) {
                        std::vector<double> pt(dim_);
                        for (int i = 0; i < dim_; ++i) pt[i] = dist(rng_);
                        double v = evaluate(pt);
                        if (v > optimal_value_) {
                            optimal_value_ = v;
                            optimal_position_ = pt;
                        }
                    }
                }
                break;
            }
        }
        return;
    }

    // --- MINIMIZE mode (default) ---
    switch (surface_type_) {
        case SurfaceType::SPHERE:
            optimal_position_.assign(dim_, 0.0);
            optimal_value_ = 0.0;
            break;

        case SurfaceType::RASTRIGIN:
            optimal_position_.assign(dim_, 0.0);
            optimal_value_ = 0.0;
            break;

        case SurfaceType::ACKLEY:
            optimal_position_.assign(dim_, 0.0);
            optimal_value_ = 0.0;
            break;

        case SurfaceType::ROSENBROCK:
            optimal_position_.assign(dim_, 1.0);
            optimal_value_ = 0.0;
            break;

        case SurfaceType::GRIEWANK:
            optimal_position_.assign(dim_, 0.0);
            optimal_value_ = 0.0;
            break;

        case SurfaceType::LEVY:
            optimal_position_.assign(dim_, 1.0);
            optimal_value_ = 0.0;
            break;

        case SurfaceType::MICHALEWICZ:
            compute_michalewicz_optimal();
            break;

        case SurfaceType::CUSTOM_SIN:
            if (has_custom_optimal_) {
                optimal_position_ = custom_optimal_pos_;
                optimal_value_ = custom_optimal_val_;
            } else {
                optimal_position_.assign(dim_, 0.0);
                optimal_value_ = 0.0;
            }
            break;

        case SurfaceType::QUADRATIC_WELL:
            optimal_position_.assign(dim_, center);
            optimal_value_ = 0.0;
            break;
    }
}

void SurfaceOptimizationEnv::compute_michalewicz_optimal() {
    constexpr double m = 10.0;
    constexpr int grid_pts = 10000;

    // Michalewicz is separable; optimize each dimension independently
    optimal_position_.resize(dim_);
    double total_opt = 0.0;

    for (int i = 0; i < dim_; ++i) {
        double best_term = -std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        for (int j = 0; j <= grid_pts; ++j) {
            double x = PI * static_cast<double>(j) / static_cast<double>(grid_pts);
            double inner = std::sin(static_cast<double>(i + 1) * x * x / PI);
            double term = std::sin(x) * std::pow(inner, 2.0 * m);
            if (term > best_term) {
                best_term = term;
                best_x = x;
            }
        }
        optimal_position_[i] = best_x;
        total_opt += best_term;
    }

    optimal_value_ = -total_opt;  // function is -sum(...)
}

} // namespace surface_env
