#pragma once

#include <vector>
#include <random>

class GridWorld {
public:
    GridWorld(int size = 5);

    void reset();
    std::vector<float> get_state() const;
    float step(int action);
    bool is_done() const;

    int get_action_size() const { return 4; }
    int get_state_size() const { return 2; }
    int get_grid_size() const { return size_; }
    float get_score() const { return score_; }

private:
    int size_;
    int agent_x_, agent_y_;
    int goal_x_, goal_y_;
    float score_;
    int steps_;
    int max_steps_;
    bool done_;
    std::mt19937 rng_;
};
