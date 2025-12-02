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
    __slots__ = ("max_seq_len", "write_pos", "valid_len")

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.write_pos = 0
        self.valid_len = 0

    def reset(self):
        self.write_pos = 0
        self.valid_len = 0

    def plan_append(self, incoming_len: int) -> WritePlan:
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

        self.write_pos = start
        self.valid_len = min(self.valid_len + incoming_len, self.max_seq_len)
        return WritePlan(ops=ops, new_valid_len=self.valid_len, incoming_len=incoming_len)

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
        self.k_linear = torch.zeros((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)
        self.v_linear = torch.zeros((max_seq_len, num_heads, head_dim), dtype=dtype, device=device)

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
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        # Inner Components
        self._planner = KVPlanner(max_seq_len)
        self._storage = KVStorage(max_seq_len, num_heads, head_dim, dtype, device)

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
        if current_len <= max_allowed_tokens:
            return 0

        tokens_to_remove = current_len - max_allowed_tokens
        self._planner.valid_len = max_allowed_tokens
        return tokens_to_remove

    def get_read_plan(self) -> ReadPlan:
        """
        Get the current read plan without executing it.
        Useful for passing to execute_gather_with_append in read-only mode.
        """
        return self._planner.plan_read()

    def reset(self):
        self._planner.reset()
        self._pending_write_plan = None
        # Storage doesn't strictly need reset as it's just memory
