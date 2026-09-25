// DwT-FL native in-memory concurrent index.
// 本模块实现预分配、仅追加的数据表和倒排用户表，以及任务状态 CAS。
// This module implements preallocated append-only forward and inverse indexes
// plus task-state CAS. It never opens a database or a file.

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <thread>

// Cross-platform dynamic library export macro
// 跨平台动态库导出宏
#if defined(_WIN32)
#define DBT_EXPORT extern "C" __declspec(dllexport)
#else
#define DBT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
    // Slot lifecycle states / 槽位生命周期状态
    constexpr int kSlotEmpty = 0;   // Unused / 未使用
    constexpr int kSlotWriting = 1; // Locked by a thread for writing / 正被某个线程写入
    constexpr int kSlotReady = 2;   // Fully initialized and read-only / 初始化完成，只读

    // Task state bit-packing constants / 任务状态位打包常量
    constexpr int kEmptyState = 0;
    constexpr int kPendingState = 1;
    constexpr int kCommittedState = 2;
    constexpr int kStateMask = 0x3;                                      // 2-bit state / 2位状态
    constexpr std::uint64_t kTrainerMask = (std::uint64_t{1} << 30) - 1;
    // 30-bit trainer ID / 30 位训练器标识
    constexpr std::uint64_t kVersionMask = (std::uint64_t{1} << 31) - 1;
    // 31-bit version / 31 位版本号
    constexpr int kTrainerShift = 2;
    constexpr int kVersionShift = 32;

    constexpr int kNoLink = -1; // End of linked list / 链表尾部标识

    // RFC 3526 2048-bit group elements use 512 hexadecimal characters plus NUL.
    // RFC 3526 的 2048 位群元素使用 512 个十六进制字符和一个结尾 NUL。
    constexpr std::size_t kLabelBytes = 513;

    // A task slot is written once before lifecycle becomes READY and is never moved.
    // 槽位发布后永不移动或释放，因此并发读者无需内存回收协议。
    struct TaskSlot
    {
        std::atomic<int> lifecycle{kSlotEmpty}; // Three-state lifecycle / 三态生命周期
        std::uint64_t hash{0};                  // FNV-1a hash of the label / 标签的 FNV-1a 哈希值
        char label[kLabelBytes]{};              // Unique task identifier / 唯一任务标识符

        // Packed 64-bit atomic state: [31-bit version][30-bit trainer][2-bit state]
        // 打包的64位原子状态：[31位版本号][30位训练器ID][2位状态]
        std::atomic<std::int64_t> word{0};

        std::atomic<std::uint32_t> previous_trainer{0};

        // Head of the linked list of trainers (clients) owning this task
        // 拥有此任务的训练器（客户端）链表头
        std::atomic<int> owner_head{kNoLink};

        // A per-task admission gate makes owner registration idempotent even
        // when one client retries the same request concurrently. It is not a
        // global mutex; hash-table operations and task-state transitions remain
        // CAS based.
        // 每任务准入门保证同一客户端并发重试时所有者登记仍然幂等。它不是全局互斥锁；
        // 哈希表操作和任务状态迁移仍以 CAS 为基础。
        std::atomic<bool> owner_append_in_progress{false};

        std::atomic<int> recovery_required{0};
        std::uint32_t created_round{0}; // FL round when task was created / 任务创建时的联邦学习轮次
    };

    // One append-only edge belongs simultaneously to a task-owner list and a
    // client-task inverse list. The per-task admission gate prevents duplicate
    // edges for the same task-owner pair.
    // 边同时链接正向所有者表和倒排用户表；每任务准入门阻止同一任务—所有者对出现重复边。
    struct OwnerEdge
    {
        std::atomic<int> next_task{kNoLink}; // Next edge in the task's owner list / 任务所有者链表的下一条边
        std::atomic<int> next_user{kNoLink}; // Next edge in the user's task list / 用户任务链表的下一条边
        std::uint32_t token{0};              // Client/Trainer ID / 客户端/训练器ID
        int task_id{kNoLink};                // Target Task ID / 目标任务ID
    };

    // Main memory arena and index structure
    // 主内存池与索引结构
    struct ConcurrentIndex
    {
        std::uint32_t capacity;    // Max number of tasks / 最大任务数
        std::uint32_t max_clients; // Max number of clients / 最大客户端数
        std::uint32_t max_edges;   // Max number of relations / 最大关联边数

        std::unique_ptr<TaskSlot[]> tasks; // Pre-allocated task array / 预分配任务数组
        // Client-to-task inverse index heads / 客户端到任务的倒排索引表头
        std::unique_ptr<std::atomic<int>[]> inverse_heads;
        std::unique_ptr<OwnerEdge[]> edges; // Pre-allocated edge pool / 预分配边池

        std::atomic<int> next_edge{0};   // Atomic edge allocator / 原子边分配器
        std::atomic<int> ready_count{0}; // Count of tasks in READY state / 处于 READY 状态的任务数

        ConcurrentIndex(std::uint32_t table_capacity, std::uint32_t clients,
                        std::uint32_t edge_capacity)
            : capacity(table_capacity),
              max_clients(clients),
              max_edges(edge_capacity),
              tasks(std::make_unique<TaskSlot[]>(table_capacity)),
               // +1 supports one-based client tokens / +1 适配从 1 开始的客户端标识
               inverse_heads(std::make_unique<std::atomic<int>[]>(clients + 1)),
              edges(std::make_unique<OwnerEdge[]>(edge_capacity))
        {
            // Initialize inverse index heads / 初始化倒排索引表头
            for (std::uint32_t token = 0; token <= max_clients; ++token)
            {
                inverse_heads[token].store(kNoLink, std::memory_order_relaxed);
            }
        }
    };

    // Validates the canonical 512-character lowercase hexadecimal label.
    // 验证规范的 512 位小写十六进制标签。
    bool valid_label(const char *label)
    {
        if (label == nullptr || std::strlen(label) != 512)
        {
            return false;
        }
        for (const unsigned char *cursor = reinterpret_cast<const unsigned char *>(label);
             *cursor != '\0'; ++cursor)
        {
            if (!std::isxdigit(*cursor) || (*cursor >= 'A' && *cursor <= 'F'))
            {
                return false;
            }
        }
        return true;
    }

    // Computes FNV-1a hash for table lookup
    // 计算 FNV-1a 哈希值用于哈希表寻址
    std::uint64_t hash_label(const char *label)
    {
        // FNV-1a is used only to choose a table slot; labels are compared in full.
        // FNV-1a 仅定位槽位，最终仍逐字节比较完整保护标签。
        std::uint64_t value = 1469598103934665603ULL;
        for (const unsigned char *cursor = reinterpret_cast<const unsigned char *>(label);
             *cursor != '\0'; ++cursor)
        {
            value ^= *cursor;
            value *= 1099511628211ULL;
        }
        return value == 0 ? 1 : value;
    }

    // Lock-free hash table insertion using linear probing
    // 基于线性探测的无锁哈希表插入
    int find_or_insert(ConcurrentIndex *index, const char *label,
                       std::uint32_t created_round)
    {
        if (index == nullptr || !valid_label(label))
        {
            return kNoLink;
        }
        const std::uint64_t hash = hash_label(label);
        const std::uint32_t start = static_cast<std::uint32_t>(hash % index->capacity);
        for (std::uint32_t probe = 0; probe < index->capacity; ++probe)
        {
            const std::uint32_t task_id = (start + probe) % index->capacity;
            TaskSlot &slot = index->tasks[task_id];

            // Acquire semantics ensures we see full initialization if state is READY
            // Acquire 语义确保如果状态为 READY，我们能看到完整的初始化数据
            int lifecycle = slot.lifecycle.load(std::memory_order_acquire);
            if (lifecycle == kSlotReady)
            {
                if (slot.hash == hash && std::memcmp(slot.label, label, 512) == 0)
                {
                    return static_cast<int>(task_id); // Found existing / 找到已存在的任务
                }
                continue; // Hash collision, probe next / 哈希冲突，探测下一个
            }

            if (lifecycle == kSlotWriting)
            {
                // Another thread is initializing this slot, spin-yield and retry
                // 另一个线程正在初始化该槽位，让出 CPU 并自旋等待
                do
                {
                    std::this_thread::yield();
                    lifecycle = slot.lifecycle.load(std::memory_order_acquire);
                } while (lifecycle == kSlotWriting);
                --probe; // Retry current slot / 重试当前槽位
                continue;
            }

            // Try to claim the empty slot / 尝试抢占空槽位
            int expected = kSlotEmpty;
            if (!slot.lifecycle.compare_exchange_strong(
                    expected, kSlotWriting, std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                --probe; // Claim failed, retry / 抢占失败，重试
                continue;
            }

            // Slot claimed, perform initialization / 槽位抢占成功，执行初始化
            slot.hash = hash;
            std::memcpy(slot.label, label, 512);
            slot.label[512] = '\0';
            slot.word.store(0, std::memory_order_relaxed);
            slot.previous_trainer.store(0, std::memory_order_relaxed);
            slot.owner_head.store(kNoLink, std::memory_order_relaxed);
            slot.recovery_required.store(0, std::memory_order_relaxed);
            slot.created_round = created_round;

            // Release semantics makes the initialization visible to other threads
            // Release 语义使初始化对其他线程可见
            slot.lifecycle.store(kSlotReady, std::memory_order_release);
            index->ready_count.fetch_add(1, std::memory_order_relaxed);
            return static_cast<int>(task_id);
        }
        return kNoLink; // Table is full / 表已满
    }

    // Lock-free hash table lookup
    // 无锁哈希表查找
    int find_existing(ConcurrentIndex *index, const char *label)
    {
        if (index == nullptr || !valid_label(label))
        {
            return kNoLink;
        }
        const std::uint64_t hash = hash_label(label);
        const std::uint32_t start = static_cast<std::uint32_t>(hash % index->capacity);
        for (std::uint32_t probe = 0; probe < index->capacity; ++probe)
        {
            const std::uint32_t task_id = (start + probe) % index->capacity;
            TaskSlot &slot = index->tasks[task_id];
            int lifecycle = slot.lifecycle.load(std::memory_order_acquire);

            if (lifecycle == kSlotEmpty)
            {
                return kNoLink; // Stop probing on empty slot / 遇到空槽位停止探测
            }
            if (lifecycle == kSlotWriting)
            {
                do
                {
                    std::this_thread::yield();
                    lifecycle = slot.lifecycle.load(std::memory_order_acquire);
                } while (lifecycle == kSlotWriting);
                --probe;
                continue;
            }
            if (slot.hash == hash && std::memcmp(slot.label, label, 512) == 0)
            {
                return static_cast<int>(task_id);
            }
        }
        return kNoLink;
    }

    bool task_id_valid(ConcurrentIndex *index, int task_id)
    {
        return index != nullptr && task_id >= 0 &&
               static_cast<std::uint32_t>(task_id) < index->capacity &&
               index->tasks[task_id].lifecycle.load(std::memory_order_acquire) == kSlotReady;
    }

    // Checks if a task is already associated with a client token
    // 检查任务是否已关联到特定的客户端 ID
    bool task_has_owner(ConcurrentIndex *index, int task_id, std::uint32_t token)
    {
        if (!task_id_valid(index, task_id))
        {
            return false;
        }
        int edge_id = index->tasks[task_id].owner_head.load(std::memory_order_acquire);
        while (edge_id != kNoLink)
        {
            const OwnerEdge &edge = index->edges[edge_id];
            if (edge.token == token)
            {
                return true;
            }
            edge_id = edge.next_task.load(std::memory_order_acquire);
        }
        return false;
    }

    // Holds one task's owner-registration gate without serializing unrelated
    // tasks. The guard uses only atomics and always releases the gate on return.
    // 该守卫只持有一个任务的所有者登记门，不会串行化无关任务；它只使用原子操作，
    // 并保证在返回时释放准入门。
    class OwnerAppendGuard
    {
    public:
        explicit OwnerAppendGuard(std::atomic<bool> &gate) : gate_(gate)
        {
            bool expected = false;
            while (!gate_.compare_exchange_weak(
                expected, true, std::memory_order_acquire, std::memory_order_relaxed))
            {
                expected = false;
                std::this_thread::yield();
            }
        }

        OwnerAppendGuard(const OwnerAppendGuard &) = delete;
        OwnerAppendGuard &operator=(const OwnerAppendGuard &) = delete;

        ~OwnerAppendGuard()
        {
            gate_.store(false, std::memory_order_release);
        }

    private:
        std::atomic<bool> &gate_;
    };

    // Reserves an edge without advancing the allocator after exhaustion.
    // 边分配器耗尽后不再推进计数器，避免失败重试扭曲已分配边的计数。
    int reserve_edge(ConcurrentIndex *index)
    {
        int observed = index->next_edge.load(std::memory_order_acquire);
        while (observed >= 0 && static_cast<std::uint32_t>(observed) < index->max_edges)
        {
            if (index->next_edge.compare_exchange_weak(
                    observed, observed + 1, std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                return observed;
            }
        }
        return kNoLink;
    }

    // Extracts fields from a packed task word. The helpers keep the C API and
    // the Python binding independent from the native bit layout.
    // 从打包任务状态字中提取字段，使 C API 和 Python 绑定无需复制原生位布局。
    int unpack_state(std::int64_t word)
    {
        return static_cast<int>(static_cast<std::uint64_t>(word) & kStateMask);
    }

    std::uint32_t unpack_trainer(std::int64_t word)
    {
        return static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(word) >> kTrainerShift) & kTrainerMask);
    }

    std::uint32_t unpack_version(std::int64_t word)
    {
        return static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(word) >> kVersionShift) & kVersionMask);
    }

    // Packs version, trainer ID, and state into a single 64-bit integer to prevent ABA
    // 将版本号、训练器 ID 和状态打包成单个 64 位整数，用于 CAS 以防止 ABA 问题
    std::int64_t pack_next(std::int64_t current, int state, std::uint32_t trainer)
    {
        const std::uint64_t unsigned_current = static_cast<std::uint64_t>(current);
        const std::uint64_t version =
            ((unsigned_current >> kVersionShift) + 1) & kVersionMask;
        const std::uint64_t packed =
            (version << kVersionShift) |
            ((static_cast<std::uint64_t>(trainer) & kTrainerMask) << kTrainerShift) |
            static_cast<std::uint64_t>(state);
        return static_cast<std::int64_t>(packed);
    }

    // Push an already-reserved edge ID onto one append-only linked list.
    // 将已预留的边 ID 压入仅追加的链表，无需互斥锁。
    void append_edge_id(std::atomic<int> &head, OwnerEdge &edge, int edge_id,
                        bool task_list)
    {
        int observed = head.load(std::memory_order_acquire);
        do
        {
            if (task_list)
            {
                edge.next_task.store(observed, std::memory_order_relaxed);
            }
            else
            {
                edge.next_user.store(observed, std::memory_order_relaxed);
            }
        } while (!head.compare_exchange_weak(
            observed, edge_id, std::memory_order_acq_rel, std::memory_order_acquire));
    }

} // namespace

// -----------------------------------------------------------------------------
// Public C-API for External (Python/FFI) Bindings
// 面向外部（Python/FFI）绑定的公共 C-API
// -----------------------------------------------------------------------------

DBT_EXPORT ConcurrentIndex *dbt_index_create(
    std::uint32_t capacity, std::uint32_t max_clients, std::uint32_t max_edges)
{
    if (capacity == 0 || max_clients == 0 || max_edges == 0)
    {
        return nullptr;
    }
    try
    {
        return new ConcurrentIndex(capacity, max_clients, max_edges);
    }
    catch (const std::bad_alloc &)
    {
        return nullptr;
    }
}

DBT_EXPORT void dbt_index_destroy(ConcurrentIndex *index)
{
    delete index;
}

// Check hardware support for the atomics used by the native index.
// 检查原生索引使用的原子操作是否得到硬件支持。
DBT_EXPORT int dbt_index_is_lock_free(ConcurrentIndex *index)
{
    if (index == nullptr)
    {
        return 0;
    }
    return index->tasks[0].lifecycle.is_lock_free() &&
                   index->tasks[0].word.is_lock_free() &&
                   index->tasks[0].owner_head.is_lock_free() &&
                   index->tasks[0].owner_append_in_progress.is_lock_free() &&
                   index->inverse_heads[0].is_lock_free()
               ? 1
               : 0;
}

// Register a task to a client, creates edges / 将任务注册给客户端，创建关联边
DBT_EXPORT int dbt_index_register_label(
    ConcurrentIndex *index, const char *label, std::uint32_t token,
    std::uint32_t created_round, int *task_id_out)
{
    if (index == nullptr || token == 0 || token > index->max_clients ||
        task_id_out == nullptr)
    {
        return 0;
    }
    const int task_id = find_or_insert(index, label, created_round);
    if (task_id == kNoLink)
    {
        return 0; // Table full / 表满
    }
    // Serialize only this task's owner admission. Without this gate, two
    // concurrent retries by the same client could both observe no owner and
    // consume two append-only edges.
    // 仅串行化当前任务的所有者准入。没有该准入门时，同一客户端的两个并发重试都可能
    // 观察到所有者不存在，并各自消耗一条仅追加边。
    OwnerAppendGuard guard(index->tasks[task_id].owner_append_in_progress);

    // Normal reconnect/retry registration is idempotent and consumes no new edge.
    // 常规重连或重试登记幂等，不额外消耗所有者边容量。
    if (task_has_owner(index, task_id, token))
    {
        *task_id_out = task_id;
        return 1;
    }

    // Allocate a new edge atomically / 原子分配新边
    const int edge_id = reserve_edge(index);
    if (edge_id == kNoLink)
    {
        return 0; // Edge pool exhausted / 边池耗尽
    }
    OwnerEdge &edge = index->edges[edge_id];
    edge.token = token;
    edge.task_id = task_id;

    // Insert into both forward and inverse lists / 插入正向和倒排链表
    append_edge_id(index->tasks[task_id].owner_head, edge, edge_id, true);
    append_edge_id(index->inverse_heads[token], edge, edge_id, false);
    *task_id_out = task_id;
    return 1;
}

DBT_EXPORT int dbt_index_find_label(
    ConcurrentIndex *index, const char *label, int *task_id_out)
{
    if (task_id_out == nullptr)
    {
        return 0;
    }
    const int task_id = find_existing(index, label);
    if (task_id == kNoLink)
    {
        return 0;
    }
    *task_id_out = task_id;
    return 1;
}

DBT_EXPORT int dbt_index_task_has_owner(
    ConcurrentIndex *index, int task_id, std::uint32_t token)
{
    return task_has_owner(index, task_id, token) ? 1 : 0;
}

// Returns the number of successfully reserved owner edges.
// 返回已成功预留的所有者边数量。
DBT_EXPORT int dbt_index_edge_count(ConcurrentIndex *index)
{
    return index == nullptr ? -1 : index->next_edge.load(std::memory_order_acquire);
}

// Export tasks owned by a specific client / 导出特定客户端拥有的所有任务
DBT_EXPORT int dbt_index_copy_client_tasks(
    ConcurrentIndex *index, std::uint32_t token, int *output, std::uint32_t capacity)
{
    if (index == nullptr || token == 0 || token > index->max_clients || output == nullptr)
    {
        return -1;
    }
    std::uint32_t count = 0;
    int edge_id = index->inverse_heads[token].load(std::memory_order_acquire);
    while (edge_id != kNoLink)
    {
        if (count >= capacity)
        {
            return -1; // Buffer too small / 缓冲区过小
        }
        output[count++] = index->edges[edge_id].task_id;
        edge_id = index->edges[edge_id].next_user.load(std::memory_order_acquire);
    }
    return static_cast<int>(count);
}

// Export clients owning a specific task / 导出拥有特定任务的所有客户端
DBT_EXPORT int dbt_index_copy_task_owners(
    ConcurrentIndex *index, int task_id, std::uint32_t *output, std::uint32_t capacity)
{
    if (!task_id_valid(index, task_id) || output == nullptr)
    {
        return -1;
    }
    std::uint32_t count = 0;
    int edge_id = index->tasks[task_id].owner_head.load(std::memory_order_acquire);
    while (edge_id != kNoLink)
    {
        if (count >= capacity)
        {
            return -1; // Buffer too small / 缓冲区过小
        }
        output[count++] = index->edges[edge_id].token;
        edge_id = index->edges[edge_id].next_task.load(std::memory_order_acquire);
    }
    return static_cast<int>(count);
}

DBT_EXPORT int dbt_index_task_label(
    ConcurrentIndex *index, int task_id, char *output, std::uint32_t output_size)
{
    if (!task_id_valid(index, task_id) || output == nullptr || output_size < kLabelBytes)
    {
        return 0;
    }
    std::memcpy(output, index->tasks[task_id].label, kLabelBytes);
    return 1;
}

// State Machine API: Read packed state / 状态机API：读取打包状态
DBT_EXPORT std::int64_t dbt_task_load(
    ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? index->tasks[task_id].word.load(std::memory_order_acquire)
               : -1;
}

// Returns a decoded task state, or -1 when the task identifier is invalid.
// 返回解码后的任务状态；任务标识无效时返回 -1。
DBT_EXPORT int dbt_task_state(ConcurrentIndex *index, int task_id)
{
    if (!task_id_valid(index, task_id))
    {
        return -1;
    }
    return unpack_state(index->tasks[task_id].word.load(std::memory_order_acquire));
}

// Returns the current trainer token, or zero for an invalid task identifier.
// 返回当前训练者令牌；任务标识无效时返回零。
DBT_EXPORT std::uint32_t dbt_task_trainer(ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? unpack_trainer(index->tasks[task_id].word.load(std::memory_order_acquire))
               : 0;
}

// Returns the ABA-protection version, or zero for an invalid task identifier.
// 返回用于防 ABA 的版本号；任务标识无效时返回零。
DBT_EXPORT std::uint32_t dbt_task_version(ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? unpack_version(index->tasks[task_id].word.load(std::memory_order_acquire))
               : 0;
}

// Claims an EMPTY task for one trainer. Exactly one competing caller can win.
// 为一个训练者认领 EMPTY 状态的任务；竞争调用者中恰有一个可以成功。
DBT_EXPORT int dbt_task_try_claim(
    ConcurrentIndex *index, int task_id, std::uint32_t trainer)
{
    if (!task_id_valid(index, task_id) || trainer == 0 || trainer > index->max_clients)
    {
        return 0;
    }
    auto &word = index->tasks[task_id].word;
    std::int64_t observed = word.load(std::memory_order_acquire);
    while (unpack_state(observed) == kEmptyState)
    {
        const std::int64_t desired = pack_next(observed, kPendingState, trainer);
        if (word.compare_exchange_weak(
                observed, desired, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            return 1;
        }
    }
    return 0;
}

// Commits a PENDING task only when the caller still owns its training right.
// 仅当调用者仍持有训练权时，将 PENDING 任务提交为 COMMITTED。
DBT_EXPORT int dbt_task_mark_committed(
    ConcurrentIndex *index, int task_id, std::uint32_t trainer)
{
    if (!task_id_valid(index, task_id) || trainer == 0 || trainer > index->max_clients)
    {
        return 0;
    }
    auto &word = index->tasks[task_id].word;
    std::int64_t observed = word.load(std::memory_order_acquire);
    while (unpack_state(observed) == kPendingState && unpack_trainer(observed) == trainer)
    {
        const std::int64_t desired = pack_next(observed, kCommittedState, trainer);
        if (word.compare_exchange_weak(
                observed, desired, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            return 1;
        }
    }
    return 0;
}

// Revokes a PENDING task only when the identified trainer still owns it.
// 仅当指定训练者仍持有 PENDING 任务时，撤销其训练权。
DBT_EXPORT int dbt_task_release_if_trainer(
    ConcurrentIndex *index, int task_id, std::uint32_t trainer)
{
    if (!task_id_valid(index, task_id) || trainer == 0 || trainer > index->max_clients)
    {
        return 0;
    }
    auto &word = index->tasks[task_id].word;
    std::int64_t observed = word.load(std::memory_order_acquire);
    while (unpack_state(observed) == kPendingState && unpack_trainer(observed) == trainer)
    {
        const std::int64_t desired = pack_next(observed, kEmptyState, 0);
        if (word.compare_exchange_weak(
                observed, desired, std::memory_order_acq_rel, std::memory_order_acquire))
        {
            return 1;
        }
    }
    return 0;
}

// State Machine API: Compare-And-Swap state / 状态机API：CAS更新状态
DBT_EXPORT int dbt_task_compare_exchange(
    ConcurrentIndex *index, int task_id, std::int64_t expected, std::int64_t desired)
{
    if (!task_id_valid(index, task_id))
    {
        return 0;
    }
    return index->tasks[task_id].word.compare_exchange_strong(
               expected, desired, std::memory_order_acq_rel,
               std::memory_order_acquire)
               ? 1
               : 0;
}

// State Machine API: Reset task state (increments version) / 状态机API：重置任务状态（递增版本号防ABA）
DBT_EXPORT int dbt_task_reset(ConcurrentIndex *index, int task_id)
{
    if (!task_id_valid(index, task_id))
    {
        return 0;
    }
    auto &word = index->tasks[task_id].word;
    std::int64_t observed = word.load(std::memory_order_acquire);
    while (!word.compare_exchange_weak(
        observed, pack_next(observed, kEmptyState, 0), std::memory_order_acq_rel,
        std::memory_order_acquire))
    {
    }
    return 1;
}

DBT_EXPORT std::uint32_t dbt_task_previous_get(
    ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? index->tasks[task_id].previous_trainer.load(std::memory_order_acquire)
               : 0;
}

DBT_EXPORT int dbt_task_previous_set(
    ConcurrentIndex *index, int task_id, std::uint32_t token)
{
    if (!task_id_valid(index, task_id))
    {
        return 0;
    }
    index->tasks[task_id].previous_trainer.store(token, std::memory_order_release);
    return 1;
}

DBT_EXPORT int dbt_task_recovery_get(
    ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? index->tasks[task_id].recovery_required.load(std::memory_order_acquire)
               : 0;
}

DBT_EXPORT int dbt_task_recovery_set(
    ConcurrentIndex *index, int task_id, int required)
{
    if (!task_id_valid(index, task_id))
    {
        return 0;
    }
    index->tasks[task_id].recovery_required.store(required ? 1 : 0,
                                                  std::memory_order_release);
    return 1;
}

DBT_EXPORT std::uint32_t dbt_task_created_round(
    ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id) ? index->tasks[task_id].created_round : 0;
}

// Export all valid tasks / 导出所有有效任务
DBT_EXPORT int dbt_index_copy_all_tasks(
    ConcurrentIndex *index, int *output, std::uint32_t capacity)
{
    if (index == nullptr || output == nullptr || capacity < index->ready_count.load())
    {
        return -1;
    }
    std::uint32_t count = 0;
    for (std::uint32_t task_id = 0; task_id < index->capacity; ++task_id)
    {
        if (index->tasks[task_id].lifecycle.load(std::memory_order_acquire) == kSlotReady)
        {
            output[count++] = static_cast<int>(task_id);
        }
    }
    return static_cast<int>(count);
}

// Calculate true memory footprint / 计算真实内存占用
DBT_EXPORT std::uint64_t dbt_index_memory_bytes(
    ConcurrentIndex *index)
{
    if (index == nullptr)
    {
        return 0;
    }
    // Reserved capacity is measured, not only populated entries, because it is
    // the actual in-memory metadata footprint of the no-resize index.
    // 统计预留容量而非已填充条目，因为这才是不可扩容索引的真实元数据占用。
    return sizeof(ConcurrentIndex) +
           static_cast<std::uint64_t>(index->capacity) * sizeof(TaskSlot) +
           static_cast<std::uint64_t>(index->max_clients + 1) * sizeof(std::atomic<int>) +
           static_cast<std::uint64_t>(index->max_edges) * sizeof(OwnerEdge);
}
