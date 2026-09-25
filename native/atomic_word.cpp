// DwT-FL native in-memory concurrent index.

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

#if defined(_WIN32)
#define DBT_EXPORT extern "C" __declspec(dllexport)
#else
#define DBT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
    // Slot lifecycle states
    constexpr int kSlotEmpty = 0;   // Unused
    constexpr int kSlotWriting = 1; // Locked by a thread for writing
    constexpr int kSlotReady = 2;   // Fully initialized and read-only

    // Task state bit-packing constants
    constexpr int kEmptyState = 0;
    constexpr int kPendingState = 1;
    constexpr int kCommittedState = 2;
    constexpr int kStateMask = 0x3;                                      // 2-bit state / 2
    constexpr std::uint64_t kTrainerMask = (std::uint64_t{1} << 30) - 1;
    // 30-bit trainer ID / 30
    constexpr std::uint64_t kVersionMask = (std::uint64_t{1} << 31) - 1;
    // 31-bit version / 31
    constexpr int kTrainerShift = 2;
    constexpr int kVersionShift = 32;

    constexpr int kNoLink = -1; // End of linked list

    // RFC 3526 2048-bit group elements use 512 hexadecimal characters plus NUL.
    // RFC 3526
    constexpr std::size_t kLabelBytes = 513;

    // A task slot is written once before lifecycle becomes READY and is never moved.
    
    struct TaskSlot
    {
        std::atomic<int> lifecycle{kSlotEmpty}; // Three-state lifecycle
        std::uint64_t hash{0};                  // FNV-1a hash of the label
        char label[kLabelBytes]{};              // Unique task identifier

        // Packed 64-bit atomic state: [31-bit version][30-bit trainer][2-bit state]
        
        std::atomic<std::int64_t> word{0};

        std::atomic<std::uint32_t> previous_trainer{0};

        // Head of the linked list of trainers (clients) owning this task
        
        std::atomic<int> owner_head{kNoLink};

        // A per-task admission gate makes owner registration idempotent even
        // when one client retries the same request concurrently. It is not a
        // global mutex; hash-table operations and task-state transitions remain
        // CAS based.
        
        
        std::atomic<bool> owner_append_in_progress{false};

        std::atomic<int> recovery_required{0};
        std::uint32_t created_round{0}; // FL round when task was created
    };

    // One append-only edge belongs simultaneously to a task-owner list and a
    // client-task inverse list. The per-task admission gate prevents duplicate
    // edges for the same task-owner pair.
    
    struct OwnerEdge
    {
        std::atomic<int> next_task{kNoLink}; // Next edge in the task's owner list
        std::atomic<int> next_user{kNoLink}; // Next edge in the user's task list
        std::uint32_t token{0};              // Client/Trainer ID
        int task_id{kNoLink};                // Target Task ID
    };

    // Main memory arena and index structure
    
    struct ConcurrentIndex
    {
        std::uint32_t capacity;    // Max number of tasks
        std::uint32_t max_clients; // Max number of clients
        std::uint32_t max_edges;   // Max number of relations

        std::unique_ptr<TaskSlot[]> tasks; // Pre-allocated task array
        // Client-to-task inverse index heads
        std::unique_ptr<std::atomic<int>[]> inverse_heads;
        std::unique_ptr<OwnerEdge[]> edges; // Pre-allocated edge pool

        std::atomic<int> next_edge{0};   // Atomic edge allocator
        std::atomic<int> ready_count{0}; // Count of tasks in READY state

        ConcurrentIndex(std::uint32_t table_capacity, std::uint32_t clients,
                        std::uint32_t edge_capacity)
            : capacity(table_capacity),
              max_clients(clients),
              max_edges(edge_capacity),
              tasks(std::make_unique<TaskSlot[]>(table_capacity)),
               // +1 supports one-based client tokens / +1
               inverse_heads(std::make_unique<std::atomic<int>[]>(clients + 1)),
              edges(std::make_unique<OwnerEdge[]>(edge_capacity))
        {
            // Initialize inverse index heads
            for (std::uint32_t token = 0; token <= max_clients; ++token)
            {
                inverse_heads[token].store(kNoLink, std::memory_order_relaxed);
            }
        }
    };

    // Validates the canonical 512-character lowercase hexadecimal label.
    
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
    
    std::uint64_t hash_label(const char *label)
    {
        // FNV-1a is used only to choose a table slot; labels are compared in full.
        // FNV-1a
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
            // Acquire
            int lifecycle = slot.lifecycle.load(std::memory_order_acquire);
            if (lifecycle == kSlotReady)
            {
                if (slot.hash == hash && std::memcmp(slot.label, label, 512) == 0)
                {
                    return static_cast<int>(task_id); // Found existing
                }
                continue; // Hash collision, probe next
            }

            if (lifecycle == kSlotWriting)
            {
                // Another thread is initializing this slot, spin-yield and retry
                
                do
                {
                    std::this_thread::yield();
                    lifecycle = slot.lifecycle.load(std::memory_order_acquire);
                } while (lifecycle == kSlotWriting);
                --probe; // Retry current slot
                continue;
            }

            // Try to claim the empty slot
            int expected = kSlotEmpty;
            if (!slot.lifecycle.compare_exchange_strong(
                    expected, kSlotWriting, std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                --probe; // Claim failed, retry
                continue;
            }

            // Slot claimed, perform initialization
            slot.hash = hash;
            std::memcpy(slot.label, label, 512);
            slot.label[512] = '\0';
            slot.word.store(0, std::memory_order_relaxed);
            slot.previous_trainer.store(0, std::memory_order_relaxed);
            slot.owner_head.store(kNoLink, std::memory_order_relaxed);
            slot.recovery_required.store(0, std::memory_order_relaxed);
            slot.created_round = created_round;

            // Release semantics makes the initialization visible to other threads
            // Release
            slot.lifecycle.store(kSlotReady, std::memory_order_release);
            index->ready_count.fetch_add(1, std::memory_order_relaxed);
            return static_cast<int>(task_id);
        }
        return kNoLink; // Table is full
    }

    // Lock-free hash table lookup
    
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
                return kNoLink; // Stop probing on empty slot
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

// Register a task to a client, creates edges
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
        return 0; // Table full
    }
    // Serialize only this task's owner admission. Without this gate, two
    // concurrent retries by the same client could both observe no owner and
    // consume two append-only edges.
    
    
    OwnerAppendGuard guard(index->tasks[task_id].owner_append_in_progress);

    // Normal reconnect/retry registration is idempotent and consumes no new edge.
    
    if (task_has_owner(index, task_id, token))
    {
        *task_id_out = task_id;
        return 1;
    }

    // Allocate a new edge atomically
    const int edge_id = reserve_edge(index);
    if (edge_id == kNoLink)
    {
        return 0; // Edge pool exhausted
    }
    OwnerEdge &edge = index->edges[edge_id];
    edge.token = token;
    edge.task_id = task_id;

    // Insert into both forward and inverse lists
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

DBT_EXPORT int dbt_index_edge_count(ConcurrentIndex *index)
{
    return index == nullptr ? -1 : index->next_edge.load(std::memory_order_acquire);
}

// Export tasks owned by a specific client
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
            return -1; // Buffer too small
        }
        output[count++] = index->edges[edge_id].task_id;
        edge_id = index->edges[edge_id].next_user.load(std::memory_order_acquire);
    }
    return static_cast<int>(count);
}

// Export clients owning a specific task
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
            return -1; // Buffer too small
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

// State Machine API: Read packed state
DBT_EXPORT std::int64_t dbt_task_load(
    ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? index->tasks[task_id].word.load(std::memory_order_acquire)
               : -1;
}

// Returns a decoded task state, or -1 when the task identifier is invalid.

DBT_EXPORT int dbt_task_state(ConcurrentIndex *index, int task_id)
{
    if (!task_id_valid(index, task_id))
    {
        return -1;
    }
    return unpack_state(index->tasks[task_id].word.load(std::memory_order_acquire));
}

// Returns the current trainer token, or zero for an invalid task identifier.

DBT_EXPORT std::uint32_t dbt_task_trainer(ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? unpack_trainer(index->tasks[task_id].word.load(std::memory_order_acquire))
               : 0;
}

// Returns the ABA-protection version, or zero for an invalid task identifier.

DBT_EXPORT std::uint32_t dbt_task_version(ConcurrentIndex *index, int task_id)
{
    return task_id_valid(index, task_id)
               ? unpack_version(index->tasks[task_id].word.load(std::memory_order_acquire))
               : 0;
}

// Claims an EMPTY task for one trainer. Exactly one competing caller can win.

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

// State Machine API: Compare-And-Swap state
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

// State Machine API: Reset task state (increments version)
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

// Export all valid tasks
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

// Calculate true memory footprint
DBT_EXPORT std::uint64_t dbt_index_memory_bytes(
    ConcurrentIndex *index)
{
    if (index == nullptr)
    {
        return 0;
    }
    // Reserved capacity is measured, not only populated entries, because it is
    // the actual in-memory metadata footprint of the no-resize index.
    
    return sizeof(ConcurrentIndex) +
           static_cast<std::uint64_t>(index->capacity) * sizeof(TaskSlot) +
           static_cast<std::uint64_t>(index->max_clients + 1) * sizeof(std::atomic<int>) +
           static_cast<std::uint64_t>(index->max_edges) * sizeof(OwnerEdge);
}
