#pragma once

#include <string>

namespace hybrid {

enum class Role {
    master,
    cpu_worker,
    gpu_worker,
    gpu_master,
    inactive
};

inline const char* role_name(Role role) {
    switch (role) {
    case Role::master: return "master";
    case Role::cpu_worker: return "cpu-worker";
    case Role::gpu_worker: return "gpu-worker";
    case Role::gpu_master: return "gpu-master";
    case Role::inactive: return "inactive";
    }
    return "invalid";
}

inline bool computes(Role role) {
    return role == Role::cpu_worker ||
        role == Role::gpu_worker ||
        role == Role::gpu_master;
}

inline bool uses_gpu(Role role) {
    return role == Role::gpu_worker || role == Role::gpu_master;
}

}  // namespace hybrid
