#include "env.h"
#include <stdexcept>

GridWorld::GridWorld(int size)
    : size_(size)
    , goal_x_(size - 1), goal_y_(size - 1)
    , score_(0.0f)
    , rng_(std::random_device{}())
{
    if (size_ < 3)
        throw std::invalid_argument("Grid size must be at least 3");
    reset();
}

void GridWorld::reset() {
    // Always start at top-left
    agent_x_ = 0;
    agent_y_ = 0;
    score_ = 0.0f;
    steps_ = 0;
    max_steps_ = size_ * size_ * 2;
    done_ = false;
}

std::vector<float> GridWorld::get_state() const {
    // Normalize positions to [0, 1]
    float norm = static_cast<float>(size_ - 1);
    return {
        static_cast<float>(agent_x_) / norm,
        static_cast<float>(agent_y_) / norm
    };
}

float GridWorld::step(int action) {
    if (done_) return 0.0f;

    ++steps_;

    int new_x = agent_x_;
    int new_y = agent_y_;

    switch (action) {
        case 0: --new_y; break;  // up
        case 1: ++new_y; break;  // down
        case 2: --new_x; break;  // left
        case 3: ++new_x; break;  // right
        default:                  // invalid action
            score_ -= 0.1f;
            return -0.1f;
    }

    // Wall collision — stay in place
    if (new_x < 0 || new_x >= size_ || new_y < 0 || new_y >= size_) {
        score_ -= 0.5f;
        return -0.5f;
    }

    agent_x_ = new_x;
    agent_y_ = new_y;

    // Reached goal
    if (agent_x_ == goal_x_ && agent_y_ == goal_y_) {
        done_ = true;
        score_ += 10.0f;
        return 10.0f;
    }

    // Timeout
    if (steps_ >= max_steps_) {
        done_ = true;
    }

    // Small step penalty to encourage shortest path
    score_ -= 0.1f;
    return -0.1f;
}

bool GridWorld::is_done() const {
    return done_;
}
