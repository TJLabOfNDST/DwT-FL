// Native CAS and mutex claim microbenchmark for DwT-FL.
// DwT-FL
//
// This file is intentionally independent from HTTP, Python and OPRF. It
// measures only state-transition throughput under a shared global mutex
// control and the production-style atomic compare-and-swap implementation.



#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define DBT_BENCH_EXPORT extern "C" __declspec(dllexport)
#else
#define DBT_BENCH_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
    constexpr int kModeCas = 0;
    constexpr int kModeMutex = 1;
    constexpr int kScenarioDisjoint = 0;
    constexpr int kScenarioSharedHotset = 1;

    // A cache-line-aligned word prevents artificial false sharing between
    // unrelated tasks.
    struct alignas(64) TaskWord
    {
        std::atomic<std::uint64_t> state{0};
    };

    bool valid_parameters(
        int mode,
        int scenario,
        std::uint32_t workers,
        std::uint32_t operations_per_worker)
    {
        return (mode == kModeCas || mode == kModeMutex)
               && (scenario == kScenarioDisjoint || scenario == kScenarioSharedHotset)
               && workers > 0 && operations_per_worker > 0;
    }

    // Returns monotonic elapsed seconds while providing exact attempt and win
    // counts.
    double run_claims(
        int mode,
        int scenario,
        std::uint32_t workers,
        std::uint32_t operations_per_worker,
        std::uint64_t *attempts_out,
        std::uint64_t *wins_out)
    {
        if (!valid_parameters(mode, scenario, workers, operations_per_worker)
            || attempts_out == nullptr || wins_out == nullptr)
        {
            return -1.0;
        }

        const std::uint64_t task_count = scenario == kScenarioDisjoint
            ? static_cast<std::uint64_t>(workers) * operations_per_worker
            : operations_per_worker;
        if (task_count == 0 || task_count > static_cast<std::uint64_t>(SIZE_MAX))
        {
            return -1.0;
        }
        std::vector<TaskWord> tasks(static_cast<std::size_t>(task_count));
        std::mutex global_mutex;
        std::atomic<std::uint64_t> attempts{0};
        std::atomic<std::uint64_t> wins{0};

        // The barrier completion records the start time before releasing any
        // worker. Its completion happens-before every released participant,
        // so no claim can be omitted from the measured region.
        
        
        auto started = std::chrono::steady_clock::time_point{};
        std::barrier start_gate(
            static_cast<std::ptrdiff_t>(workers) + 1,
            [&started]() noexcept { started = std::chrono::steady_clock::now(); });
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::uint32_t worker = 0; worker < workers; ++worker)
        {
            threads.emplace_back([&, worker]() {
                start_gate.arrive_and_wait();
                std::uint64_t local_attempts = 0;
                std::uint64_t local_wins = 0;
                for (std::uint32_t operation = 0; operation < operations_per_worker; ++operation)
                {
                    const std::uint64_t task_id = scenario == kScenarioDisjoint
                        ? static_cast<std::uint64_t>(worker) * operations_per_worker + operation
                        : operation;
                    auto &word = tasks[static_cast<std::size_t>(task_id)].state;
                    ++local_attempts;
                    bool claimed = false;
                    if (mode == kModeCas)
                    {
                        std::uint64_t expected = 0;
                        claimed = word.compare_exchange_strong(
                            expected, static_cast<std::uint64_t>(worker) + 1,
                            std::memory_order_acq_rel,
                            std::memory_order_acquire);
                    }
                    else
                    {
                        // This is the pessimistic global-lock control.
                        
                        std::lock_guard<std::mutex> guard(global_mutex);
                        if (word.load(std::memory_order_relaxed) == 0)
                        {
                            word.store(static_cast<std::uint64_t>(worker) + 1,
                                       std::memory_order_relaxed);
                            claimed = true;
                        }
                    }
                    if (claimed)
                    {
                        ++local_wins;
                    }
                }
                attempts.fetch_add(local_attempts, std::memory_order_relaxed);
                wins.fetch_add(local_wins, std::memory_order_relaxed);
            });
        }

        start_gate.arrive_and_wait();
        for (auto &thread : threads)
        {
            thread.join();
        }
        const auto finished = std::chrono::steady_clock::now();
        *attempts_out = attempts.load(std::memory_order_relaxed);
        *wins_out = wins.load(std::memory_order_relaxed);
        return std::chrono::duration<double>(finished - started).count();
    }
} // namespace

// Return whether this platform provides lock-free 64-bit atomic state words.

DBT_BENCH_EXPORT int dbt_claim_benchmark_is_lock_free()
{
    std::atomic<std::uint64_t> word{0};
    return word.is_lock_free() ? 1 : 0;
}

// Run one native claim benchmark. mode: 0 CAS, 1 global mutex; scenario:
// 0 disjoint task ranges, 1 one shared hot task set.


DBT_BENCH_EXPORT double dbt_run_claim_benchmark(
    int mode,
    int scenario,
    std::uint32_t workers,
    std::uint32_t operations_per_worker,
    std::uint64_t *attempts_out,
    std::uint64_t *wins_out)
{
    try
    {
        return run_claims(
            mode, scenario, workers, operations_per_worker, attempts_out, wins_out);
    }
    catch (const std::bad_alloc &)
    {
        return -1.0;
    }
}
