#include "ale_interface.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Handle {
    ale::ALEInterface ale;
    std::vector<unsigned char> current;
    std::vector<unsigned char> previous;
    std::vector<ale::Action> actions;
};

struct Info {
    float reward;
    int32_t done;
    int32_t truncated;
    int32_t lives;
    int32_t width;
    int32_t height;
};

static void fail(char* out, size_t size, const char* message) {
    if (size) {
        std::strncpy(out, message, size - 1);
        out[size - 1] = 0;
    }
}

static void capture(
    Handle& h,
    uint8_t* rgb,
    size_t capacity,
    uint8_t* ram,
    Info* info,
    bool pool
) {
    info->width = h.ale.getScreen().width();
    info->height = h.ale.getScreen().height();

    if (
        h.current.size() != size_t(info->width) * size_t(info->height) * 3 ||
        capacity < h.current.size()
    ) {
        throw std::runtime_error("Invalid RGB buffer dimensions");
    }

    if (pool && h.previous.size() == h.current.size()) {
        for (size_t i = 0; i < h.current.size(); ++i) {
            rgb[i] = std::max(h.current[i], h.previous[i]);
        }
    } else {
        std::memcpy(rgb, h.current.data(), h.current.size());
    }

    const auto& memory = h.ale.getRAM();

    for (size_t i = 0; i < 128; ++i) {
        ram[i] = memory.get(i);
    }

    info->done = h.ale.game_over();
    info->truncated = h.ale.game_truncated();
    info->lives = h.ale.lives();
}

extern "C" {

void rl_ale_quiet() {
    ale::Logger::setMode(ale::Logger::Error);
}

void* rl_ale_create(
    const char* rom,
    uint32_t seed,
    float sticky,
    int32_t* actions,
    int32_t* count,
    int32_t* width,
    int32_t* height,
    char* error,
    size_t size
) {
    try {
        auto h = std::make_unique<Handle>();

        h->ale.setBool("display_screen", false);
        h->ale.setInt("random_seed", int(seed & 0x7fffffff));
        h->ale.setInt("frame_skip", 1);
        h->ale.setInt("max_num_frames_per_episode", 0);
        h->ale.setFloat("repeat_action_probability", sticky);
        h->ale.loadROM(std::filesystem::u8path(rom));

        h->actions = h->ale.getMinimalActionSet();

        if (h->actions.empty() || h->actions.size() > 18) {
            throw std::runtime_error("Invalid minimal action set");
        }

        *count = int32_t(h->actions.size());

        for (size_t i = 0; i < h->actions.size(); ++i) {
            actions[i] = int32_t(h->actions[i]);
        }

        *width = h->ale.getScreen().width();
        *height = h->ale.getScreen().height();

        return h.release();
    } catch (const std::exception& e) {
        fail(error, size, e.what());
        return nullptr;
    }
}

void rl_ale_destroy(void* handle) {
    delete static_cast<Handle*>(handle);
}

int32_t rl_ale_reset(
    void* handle,
    uint8_t* rgb,
    size_t capacity,
    uint8_t* ram,
    Info* info,
    char* error,
    size_t size
) {
    try {
        auto& h = *static_cast<Handle*>(handle);

        h.ale.reset_game();
        h.ale.getScreenRGB(h.current);
        h.previous.clear();
        info->reward = 0;

        capture(h, rgb, capacity, ram, info, false);
        return 0;
    } catch (const std::exception& e) {
        fail(error, size, e.what());
        return -1;
    }
}

int32_t rl_ale_step(
    void* handle,
    int32_t action,
    uint32_t repeat,
    int32_t pool,
    uint8_t* rgb,
    size_t capacity,
    uint8_t* ram,
    Info* info,
    char* error,
    size_t size
) {
    try {
        auto& h = *static_cast<Handle*>(handle);

        if (
            action < 0 ||
            size_t(action) >= h.actions.size() ||
            repeat < 1 ||
            repeat > 60
        ) {
            throw std::runtime_error("Invalid action/repeat");
        }

        info->reward = 0;
        h.previous.clear();

        for (uint32_t i = 0; i < repeat; ++i) {
            info->reward += float(h.ale.act(h.actions[action]));

            h.previous.swap(h.current);
            h.ale.getScreenRGB(h.current);

            if (h.ale.game_over()) {
                break;
            }
        }

        capture(h, rgb, capacity, ram, info, pool != 0);
        return 0;
    } catch (const std::exception& e) {
        fail(error, size, e.what());
        return -1;
    }
}

}
