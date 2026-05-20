#include "function_env.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FunctionFittingEnv::FunctionFittingEnv(FunctionId func_id, int n_actions, int episode_length)
    : func_id_(func_id)
    , n_actions_(n_actions)
    , episode_length_(episode_length)
    , rng_(std::random_device{}())
    , x_dist_(0.0f, 1.0f)
{
    if (n_actions_ < 2)
        throw std::invalid_argument("n_actions must be at least 2");

    // Determine y range by sampling the function
    float y_min = 1e10f, y_max = -1e10f;
    for (int i = 0; i <= 1000; ++i) {
        float x = static_cast<float>(i) / 1000.0f;
        float y = evaluate_function(x);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
    }
    // Add 5% padding
    float padding = (y_max - y_min) * 0.05f + 1e-6f;
    y_min_ = y_min - padding;
    y_max_ = y_max + padding;
    bin_width_ = (y_max_ - y_min_) / static_cast<float>(n_actions_ - 1);

    reset();
}

void FunctionFittingEnv::reset() {
    current_x_ = x_dist_(rng_);
    steps_ = 0;
    score_ = 0.0f;
    done_ = false;
}

std::vector<float> FunctionFittingEnv::get_state() const {
    return {current_x_};
}

float FunctionFittingEnv::step(int action) {
    if (done_) return 0.0f;

    ++steps_;

    // Action -> y value
    float y_pred = y_min_ + static_cast<float>(action) * bin_width_;
    float y_true = evaluate_function(current_x_);
    float reward = -std::abs(y_true - y_pred);

    score_ += reward;

    if (steps_ >= episode_length_) {
        done_ = true;
    }

    // Sample new x for next step
    current_x_ = x_dist_(rng_);
    return reward;
}

bool FunctionFittingEnv::is_done() const {
    return done_;
}

std::string FunctionFittingEnv::get_function_name() const {
    switch (func_id_) {
        case FunctionId::STEP_SINGLE:            return "step_single";
        case FunctionId::STEP_MULTI:             return "step_multi";
        case FunctionId::SAWTOOTH:               return "sawtooth";
        case FunctionId::STAIRCASE:              return "staircase";
        case FunctionId::GAUSSIAN_BUMPS:         return "gaussian_bumps";
        case FunctionId::DISCONTINUOUS_JUMP:     return "discontinuous_jump";
        case FunctionId::POLYNOMIAL_CUBIC:        return "polynomial_cubic";
        case FunctionId::SINUSOID_COMPOSITE:      return "sinusoid_composite";
        case FunctionId::V_SHAPED_ABSOLUTE:       return "v_shaped_absolute";
        case FunctionId::COMPLEX_MULTI_SEGMENT:   return "complex_multi_segment";
        default: return "unknown";
    }
}

float FunctionFittingEnv::evaluate_function(float x) const {
    switch (func_id_) {
        // ── 0: Single step ──────────────────────────────────────────
        case FunctionId::STEP_SINGLE:
            return (x < 0.5f) ? -1.0f : 1.0f;

        // ── 1: Multi-step (3 jumps) ─────────────────────────────────
        case FunctionId::STEP_MULTI: {
            if (x < 0.3f)       return -1.0f;
            else if (x < 0.5f)  return 0.0f;
            else if (x < 0.7f)  return 0.5f;
            else                return 1.0f;
        }

        // ── 2: Sawtooth ─────────────────────────────────────────────
        case FunctionId::SAWTOOTH: {
            float period = 0.25f;
            float phase = std::fmod(x, period) / period;
            return 2.0f * phase - 1.0f;
        }

        // ── 3: Staircase (5 levels) ─────────────────────────────────
        case FunctionId::STAIRCASE: {
            int level = static_cast<int>(std::floor(x * 5.0f));
            if (level >= 5) level = 4;
            return -1.0f + static_cast<float>(level) * 0.5f;  // -1, -0.5, 0, 0.5, 1
        }

        // ── 4: Gaussian bumps ───────────────────────────────────────
        case FunctionId::GAUSSIAN_BUMPS: {
            float g1 = std::exp(-std::pow((x - 0.3f) / 0.1f, 2.0f)) * 0.8f;
            float g2 = std::exp(-std::pow((x - 0.7f) / 0.15f, 2.0f)) * 1.0f;
            float g3 = std::exp(-std::pow((x - 0.5f) / 0.08f, 2.0f)) * 0.6f;
            return g1 + g2 - g3;
        }

        // ── 5: Discontinuous jump ───────────────────────────────────
        case FunctionId::DISCONTINUOUS_JUMP: {
            if (x < 0.4f)
                return std::sin(4.0f * M_PI * x);
            else if (x > 0.6f)
                return std::cos(4.0f * M_PI * x) * 0.8f;
            else
                return 0.5f;  // gap region — sudden jump from both sides
        }

        // ── 6: Polynomial cubic ─────────────────────────────────────
        case FunctionId::POLYNOMIAL_CUBIC: {
            float t = x - 0.5f;
            return 2.0f * t * t * t - t;
        }

        // ── 7: Sinusoid composite ───────────────────────────────────
        case FunctionId::SINUSOID_COMPOSITE:
            return std::sin(6.0f * M_PI * x) * 0.6f + std::cos(2.0f * M_PI * x) * 0.4f;

        // ── 8: V-shaped absolute + wiggle ───────────────────────────
        case FunctionId::V_SHAPED_ABSOLUTE:
            return std::abs(2.0f * x - 1.0f) * 1.5f - 0.5f + 0.15f * std::sin(10.0f * M_PI * x);

        // ── 9: Complex multi-segment (6-piece hybrid) ───────────────
        case FunctionId::COMPLEX_MULTI_SEGMENT: {
            if (x < 0.15f)
                // Segment 1: sinusoidal
                return std::sin(8.0f * M_PI * x) * 0.5f;
            else if (x < 0.35f)
                // Segment 2: quadratic dip
                return -4.0f * (x - 0.25f) * (x - 0.25f) + 0.6f;
            else if (x < 0.5f)
                // Segment 3: linear ramp
                return 1.5f * (x - 0.35f) / 0.15f - 0.5f;
            else if (x < 0.7f)
                // Segment 4: exponential decay
                return std::exp(-(x - 0.5f) * 5.0f) * 0.8f;
            else if (x < 0.85f)
                // Segment 5: cosine oscillation
                return std::cos(6.0f * M_PI * (x - 0.7f)) * 0.7f;
            else
                // Segment 6: cubic tail
                return 2.0f * (x - 0.85f) * (x - 0.85f) * (x - 0.85f) * 8.0f - 0.3f;
        }

        default:
            return 0.0f;
    }
}
