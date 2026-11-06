// Profile one warm decode after weights, device caches and the first step exist.
#include "strata/cann/engine.hpp"
#include "strata/cann/device.hpp"
#include <acl/acl_prof.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(aclError error, const char* call) {
    if (error != ACL_SUCCESS) throw std::runtime_error(std::string(call) + ": " + std::to_string(error));
}
class Profile {
public:
    Profile(const std::string& path, std::vector<uint32_t>& devices) {
        std::filesystem::create_directories(path);
        check(aclprofInit(path.c_str(), path.size()), "aclprofInit");
        initialized_ = true;
        config_ = aclprofCreateConfig(devices.data(), uint32_t(devices.size()), ACL_AICORE_PIPE_UTILIZATION,
            nullptr, ACL_PROF_ACL_API | ACL_PROF_RUNTIME_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU);
        if (!config_) { aclprofFinalize(); initialized_ = false; throw std::runtime_error("aclprofCreateConfig failed"); }
        const auto status = aclprofStart(config_);
        if (status != ACL_SUCCESS) { aclprofDestroyConfig(config_); config_ = nullptr; aclprofFinalize(); initialized_ = false; }
        check(status, "aclprofStart");
        started_ = true;
    }
    void stop() {
        check(aclprofStop(config_), "aclprofStop"); started_ = false;
        check(aclprofDestroyConfig(config_), "aclprofDestroyConfig"); config_ = nullptr;
        check(aclprofFinalize(), "aclprofFinalize"); initialized_ = false;
    }
    ~Profile() {
        if (started_) aclprofStop(config_);
        if (config_) aclprofDestroyConfig(config_);
        if (initialized_) aclprofFinalize();
    }
private:
    aclprofConfig* config_ = nullptr;
    bool initialized_ = false, started_ = false;
};
}

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: model_warm_profile model.gguf output-directory\n"); return 2; }
    try {
        std::vector<uint32_t> devices;
        std::stringstream spec(std::getenv("STRATA_CANN_DEVICES") ? std::getenv("STRATA_CANN_DEVICES") : "0");
        std::string part;
        while (std::getline(spec, part, ',')) devices.push_back(uint32_t(std::stoul(part)));
        struct Finish { ~Finish() { strata::cann::shutdown(); } } finish;
        strata::cann::Model model(argv[1], 16);
        auto hidden = model.step(785, 0);
        const auto first = model.top(hidden, 32);
        const auto token = first.best.at(0).second;
        Profile profile(argv[2], devices);
        const auto begin = std::chrono::steady_clock::now();
        hidden = model.step(token, 1);
        const auto second = model.top(hidden, 32);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        profile.stop();
        std::printf("WARM_PROFILE input=785 next=%d result=%d logit=%.6f seconds=%.6f\n", token,
                    second.best.at(0).second, second.best.at(0).first, seconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "warm profile: %s\n", error.what()); return 1;
    }
}
