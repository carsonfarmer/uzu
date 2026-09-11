#include <metal_stdlib>
using namespace metal;
// Host and device share eight uint32 fields. IDs are graph-local, 0...5.
struct Task { uint id, op, a, b, dependencies, unused0, unused1, unused2; };
struct Config { uint jobs, width, workers, phase; };
#define PARAMETERS \
 device const float* input [[buffer(0)]], \
 device float* scratch [[buffer(1)]], \
 device atomic_uint* counts [[buffer(2)]], \
 device uint* completed [[buffer(3)]], \
 device atomic_uint* head [[buffer(4)]], \
 device const Task* tasks [[buffer(5)]], \
 constant Config& cfg [[buffer(6)]], \
 uint group [[threadgroup_position_in_grid]], \
 uint lane [[thread_index_in_threadgroup]]

inline void operation(device const float* input, device float* scratch,
                      device atomic_uint* counts, Task task, uint job,
                      uint lane, uint width) {
    uint base = job * 6 * width;
    float a = task.id == 0 ? input[job * width + lane] : scratch[base + task.a * width + lane];
    float result;
    switch (task.op) {
        case 0: result = a * 2.0f; break;
        case 1: result = a + 1.0f; break;
        case 2: result = a * a; break;
        case 3: result = a + scratch[base + task.b * width + lane]; break;
        case 4: result = a * 0.5f; break;
        default: result = 0.0f; break; // host rejects unsupported opcodes
    }
    scratch[base + task.id * width + lane] = result;
    if (lane == 0) atomic_fetch_add_explicit(counts + job * 6 + task.id, 1u, memory_order_relaxed);
}

kernel void phased(PARAMETERS) {
    Task task = tasks[cfg.phase]; // host supplies topological descriptor order
    operation(input, scratch, counts, task, group, lane, cfg.width);
    if (lane == 0 && cfg.phase == 5) completed[group] = 63;
}

inline void graph(device const float* input, device float* scratch,
                  device atomic_uint* counts, device uint* completed,
                  device const Task* tasks, uint job, uint lane, uint width,
                  bool interpret) {
    uint done = 0;
    // Each worker owns a complete graph. No inter-threadgroup publication or wait.
    if (!interpret) {
        for (uint index = 0; index < 6; ++index) {
            operation(input, scratch, counts, tasks[index], job, lane, width);
            threadgroup_barrier(mem_flags::mem_device);
        }
        done = 63;
    } else {
        // At least one task is ready in every round of a host-validated DAG.
        // Fixed six scans bound the interpreter even for reverse descriptor order.
        for (uint round = 0; round < 6 && done != 63; ++round) {
            for (uint index = 0; index < 6; ++index) {
                Task task = tasks[index];
                uint bit = 1u << task.id;
                if (!(done & bit) && (done & task.dependencies) == task.dependencies) {
                    operation(input, scratch, counts, task, job, lane, width);
                    threadgroup_barrier(mem_flags::mem_device);
                    done |= bit;
                }
            }
        }
    }
    if (lane == 0) completed[job] = done;
    threadgroup_barrier(mem_flags::mem_device);
}

kernel void fixed_fusion(PARAMETERS) {
    graph(input, scratch, counts, completed, tasks, group, lane, cfg.width, false);
}
kernel void static_tasks(PARAMETERS) {
    for (uint job = group; job < cfg.jobs; job += cfg.workers)
        graph(input, scratch, counts, completed, tasks, job, lane, cfg.width, true);
}
kernel void dynamic_tasks(PARAMETERS) {
    threadgroup uint claimed;
    // Each claim removes one independent graph from a finite prepublished queue.
    // The atomic is only an allocator; it never publishes activation data.
    while (true) {
        if (lane == 0) claimed = atomic_fetch_add_explicit(head, 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint job = claimed;
        if (job >= cfg.jobs) break;
        graph(input, scratch, counts, completed, tasks, job, lane, cfg.width, true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
