#pragma once

#include <vector>
#include <string>
#include <random>

enum class FunctionId : int {
    STEP_SINGLE = 0,
    STEP_MULTI,
    SAWTOOTH,
    STAIRCASE,
    GAUSSIAN_BUMPS,
    DISCONTINUOUS_JUMP,
    POLYNOMIAL_CUBIC,
    SINUSOID_COMPOSITE,
    V_SHAPED_ABSOLUTE,
    COMPLEX_MULTI_SEGMENT,
    FUNCTION_COUNT
};

class FunctionFittingEnv {
public:
    FunctionFittingEnv(FunctionId func_id, int n_actions = 100, int episode_length = 200);

    void reset();
    std::vector<float> get_state() const;
    float step(int action);
    bool is_done() const;

    int get_action_size() const { return n_actions_; }
    int get_state_size() const { return 1; }
    float get_score() const { return score_; }
    std::string get_function_name() const;
    float get_y_min() const { return y_min_; }
    float get_y_max() const { return y_max_; }

private:
    float evaluate_function(float x) const;

    FunctionId func_id_;
    int n_actions_;
    int episode_length_;
    float y_min_;
    float y_max_;
    float bin_width_;

    float current_x_;
    int steps_;
    float score_;
    bool done_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> x_dist_;
};
