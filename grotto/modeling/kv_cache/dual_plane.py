from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# ==========================================
# Protocol
# ==========================================


@dataclass(frozen=True)
class CopySlice:
    src_slice: slice
    dst_slice: slice


@dataclass(frozen=True)
class WritePlan:
    ops: List[CopySlice]
    new_valid_len: int
    incoming_len: int


@dataclass(frozen=True)
class ReadPlan:
    start_offset: int
    length: int


# ==========================================
# Control Plane (CPU) - Unchanged
# ==========================================


class KVPlanner:
    __slots__ = ("max_seq_len", "write_pos", "valid_len", "tokens_per_latent", "latent_count")

    def __init__(self, max_seq_len: int, tokens_per_latent: int = 1):
        self.max_seq_len = max_seq_len
        self.tokens_per_latent = tokens_per_latent
        self.write_pos = 0
        self.valid_len = 0
        self.latent_count = 0

    def reset(self):
        self.write_pos = 0
        self.valid_len = 0
        self.latent_count = 0

    def plan_append(self, incoming_len: int) -> WritePlan:
        # Calculate write operations for ring buffer
        ops = []
        start = self.write_pos
        remaining = incoming_len
        src_offset = 0

        while remaining > 0:
            available = self.max_seq_len - start
            chunk = min(remaining, available)
            ops.append(
                CopySlice(
                    src_slice=slice(src_offset, src_offset + chunk),
                    dst_slice=slice(start, start + chunk),
                )
            )
            start = (start + chunk) % self.max_seq_len
            src_offset += chunk
            remaining -= chunk

        # Update valid length before eviction (for latent counting)
        old_valid_len = self.valid_len
        self.write_pos = start

        # Evict old tokens if we would exceed max_seq_len (sliding window)
        if self.valid_len + incoming_len > self.max_seq_len:
            # Oldest tokens will be overwritten by ring buffer wrapping
            # Just update valid_len to maintain sliding window of max_seq_len
            self.valid_len = self.max_seq_len
        else:
            self.valid_len += incoming_len

        actual_added = self.valid_len - old_valid_len
        # print(f"[plan_append] incoming_len={incoming_len}, tokens_per_latent={self.tokens_per_latent}, actual_added={actual_added}, old_latent_count={self.latent_count}")
        if actual_added == self.tokens_per_latent:
            self.latent_count += 1
            # print(f"[plan_append] INCREMENTED latent_count to {self.latent_count}")

        return WritePlan(ops=ops, new_valid_len=self.valid_len, incoming_len=incoming_len)

    def push_latent(self) -> WritePlan:
        return self.plan_append(self.tokens_per_latent)

    def pop_latent(self, count: int = 1) -> int:
        if count <= 0:
            return 0
        # print(f"[pop_latent] Attempting to pop {count} latents, current latent_count={self.latent_count}, valid_len={self.valid_len}, tokens_per_latent={self.tokens_per_latent}")

        actual_pop = min(count, self.latent_count)
        if actual_pop == 0:
            # print(f"[pop_latent] Nothing to pop, latent_count=0")
            return 0

        tokens_to_remove = actual_pop * self.tokens_per_latent
        tokens_to_remove = min(tokens_to_remove, self.valid_len)

        self.valid_len -= tokens_to_remove
        self.write_pos = (self.write_pos - tokens_to_remove) % self.max_seq_len
        self.latent_count -= actual_pop
        # print(f"[pop_latent] Popped {actual_pop} latents, new latent_count={self.latent_count}, new valid_len={self.valid_len}")
        return tokens_to_remove

    def plan_read(self) -> ReadPlan:
        if self.valid_len == 0:
            return ReadPlan(0, 0)
        start_offset = (self.write_pos - self.valid_len) % self.max_seq_len
        return ReadPlan(start_offset=start_offset, length=self.valid_len)

    @property
    def total_tokens(self) -> int:
        return self.valid_len


# ==========================================
# Data Plane (GPU) - Updated for Linear Cache
# ==========================================


class KVStorage:
    def __init__(
        self,
        max_seq_len: int,
        max_incoming_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len

        # 1. Ring Buffer (Physical Storage)
        self.k_cache = torch.zeros((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)

        # 2. Linear Buffer (Contiguous View for Attention)
        #    Pre-allocated to avoid malloc during inference
        self.k_linear = torch.zeros(
            (max_seq_len + max_incoming_len, num_heads, head_dim), dtype=dtype, device=device
        )
        self.v_linear = torch.zeros(
            (max_seq_len + max_incoming_len, num_heads, head_dim), dtype=dtype, device=device
        )

        # 3. Helpers
        self._indices_buf = torch.arange(max_seq_len, device=device, dtype=torch.long)

    def execute_write(self, plan: WritePlan, k: torch.Tensor, v: torch.Tensor) -> None:
        """Writes incoming data into the Ring Buffer."""
        for op in plan.ops:
            self.k_cache[op.dst_slice] = k[op.src_slice]
            self.v_cache[op.dst_slice] = v[op.src_slice]

    def execute_gather(self, plan: ReadPlan) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linearizes the Ring Buffer into self.k_linear / self.v_linear.
        Uses `out=` for zero-allocation.
        """
        if plan.length == 0:
            return self.k_linear[:0], self.v_linear[:0]

        # 1. Calculate read indices
        #    (start_offset + 0..length) % max_len
        read_indices = (plan.start_offset + self._indices_buf[: plan.length]) % self.max_seq_len

        # 2. Gather into Linear Cache (In-place)
        #    We select valid rows from Ring and write them into the start of Linear
        target_k = self.k_linear[: plan.length]
        target_v = self.v_linear[: plan.length]

        torch.index_select(self.k_cache, 0, read_indices, out=target_k)
        torch.index_select(self.v_cache, 0, read_indices, out=target_v)

        # 3. Return Views (Zero-copy)
        return target_k, target_v

    def execute_gather_with_append(
        self, plan: ReadPlan, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimization: Sometimes we want to get the linear view INCLUDING the new tokens
        that we haven't officially 'appended' to the ring yet, or just for convenience.
        This writes history to linear, then appends new stuff to linear.
        """
        # Linearize history first
        k_hist, v_hist = self.execute_gather(plan)

        # Calculate where to put new data in linear cache
        hist_len = plan.length
        new_len = k_new.shape[0]
        total_len = hist_len + new_len

        # Copy new data to linear cache tail
        # Note: Be careful not to overflow max_seq_len logic here if needed
        self.k_linear[hist_len:total_len] = k_new
        self.v_linear[hist_len:total_len] = v_new

        return self.k_linear[:total_len], self.v_linear[:total_len]


class StateError(RuntimeError):
    """Raised when method call violates the Plan-Execute state machine."""

    pass


class DualPlaneKVCache:
    """
    High-level Orchestrator for Separated CPU/GPU KV Cache.

    Enforces a strict transactional lifecycle:
    1. plan_append() -> Updates CPU state, locks transaction.
    2. (Optional) Access metadata (total_tokens, etc.).
    3. execute_append() -> Performs GPU work, unlocks transaction.
    """

    def __init__(
        self,
        max_seq_len: int,
        max_incoming_len: int,
        num_heads: int,
        head_dim: int,
        tokens_per_latent: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        # Inner Components
        self._planner = KVPlanner(max_seq_len, tokens_per_latent)
        self._storage = KVStorage(max_seq_len, max_incoming_len, num_heads, head_dim, dtype, device)

        # Transaction State
        self._pending_write_plan: Optional[WritePlan] = None

    # ==================================================
    # Phase 1: Planning (CPU only)
    # ==================================================

    def plan_append(self, incoming_len: int) -> "DualPlaneKVCache":
        """
        Stage 1: Update logical state and generate instruction set.

        Raises:
            StateError: If a previous plan hasn't been executed yet.
        """
        if self._pending_write_plan is not None:
            raise StateError(
                "Cannot plan a new append while a previous plan is pending execution. "
                "Call execute_append() first to commit data to GPU."
            )

        # 1. Delegate logic to Planner (Fast, Pure CPU)
        # Note: Planner state is updated IMMEDIATELY here.
        plan = self._planner.plan_append(incoming_len)

        # 2. Lock state
        self._pending_write_plan = plan

        # Return self for fluent API: cache.plan_append(10).execute_append(...)
        return self

    # ==================================================
    # Phase 2: Metadata Access (Zero-Cost)
    # ==================================================

    @property
    def total_tokens(self) -> int:
        """Returns the logical token count (includes pending planned tokens)."""
        return self._planner.total_tokens

    @property
    def is_transaction_active(self) -> bool:
        """True if there is a plan waiting to be executed."""
        return self._pending_write_plan is not None

    @property
    def pending_count(self) -> int:
        """How many tokens are waiting to be written to GPU."""
        return self._pending_write_plan.incoming_len if self._pending_write_plan else 0

    # ==================================================
    # Phase 3: Execution (GPU only)
    # ==================================================

    def execute_append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Stage 2: Execute the pending plan on GPU.

        Args:
            k: Key tensor (must match planned length)
            v: Value tensor (must match planned length)
        """
        plan = self._pending_write_plan

        if plan is None:
            raise StateError("No pending plan found. Call plan_append() first.")

        # Safety Check: Input tensor must match the Plan
        if k.shape[0] != plan.incoming_len:
            raise ValueError(
                f"Tensor shape mismatch. Planned for {plan.incoming_len}, "
                f"got {k.shape[0]}. Logic/Data desync detected."
            )

        # 1. Delegate execution to Storage (Pure GPU)
        self._storage.execute_write(plan, k, v)

        # 2. Unlock state
        self._pending_write_plan = None

    # ==================================================
    # Read / Linearize Utilities
    # ==================================================

    def get_linear_view(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a linear (contiguous) view of the cache for Attention.
        Automatically handles planning and gathering.
        """
        # Read operations usually don't need strict transaction locking
        # because they don't mutate state, but we must ensure no write is pending
        # to guarantee we aren't reading stale data (or decide policy).

        # Policy: Reading while a write is pending is allowed,
        # but it will only read what's PHYSICALLY in the ring buffer.
        # However, since Planner updated 'valid_len' during plan_append,
        # reading now might try to read garbage data that hasn't been written yet.

        if self._pending_write_plan is not None:
            raise StateError(
                "Cannot read linear view while a write transaction is pending. "
                "The logical state is ahead of physical storage. "
                "Execute the append first."
            )

        # 1. Plan
        read_plan = self._planner.plan_read()

        # 2. Execute
        return self._storage.execute_gather(read_plan)

    def evict(self, max_allowed_tokens: int) -> int:
        """
        Logically evict oldest tokens, keeping only the most recent max_allowed_tokens.
        In a ring buffer, this only updates valid_len - no data movement.

        Args:
            max_allowed_tokens: Maximum number of tokens to keep

        Returns:
            Number of tokens evicted
        """
        current_len = self._planner.valid_len
        # print(f"[evict] max_allowed_tokens={max_allowed_tokens}, current_len={current_len}, latent_count={self._planner.latent_count}")
        if current_len <= max_allowed_tokens:
            return 0

        tokens_to_remove = current_len - max_allowed_tokens
        self._planner.valid_len = max_allowed_tokens
        # print(f"[evict] Evicted {tokens_to_remove} tokens, new valid_len={self._planner.valid_len}, latent_count unchanged={self._planner.latent_count}")
        return tokens_to_remove

    def get_read_plan(self) -> ReadPlan:
        """
        Get the current read plan without executing it.
        Useful for passing to execute_gather_with_append in read-only mode.
        """
        return self._planner.plan_read()

    def push_latent(self) -> "DualPlaneKVCache":
        if self._pending_write_plan is not None:
            raise StateError("Cannot push latent while a previous plan is pending execution")

        plan = self._planner.push_latent()
        self._pending_write_plan = plan
        return self

    def pop_latent(self, count: int = 1) -> int:
        if self._pending_write_plan is not None:
            raise StateError("Cannot pop latent while a transaction is pending")

        return self._planner.pop_latent(count)

    @property
    def latent_count(self) -> int:
        return self._planner.latent_count

    def reset(self):
        self._planner.reset()
        self._pending_write_plan = None
        # Storage doesn't strictly need reset as it's just memory
