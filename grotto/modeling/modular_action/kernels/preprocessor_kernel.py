import torch
import triton
import triton.language as tl


@triton.jit
def mouse_preprocessor_kernel(
    # Pointers
    HIDDEN_STATES_PTR,
    MOUSE_CONDITION_PTR,  # Original unpadded mouse_condition
    FUSED_PTR,
    # Shapes
    B,
    T_q,
    S,
    C_hidden,  # B, T*S, C -> reshaped as (B, T, S, C)
    N_frames,
    C_mouse,  # Original N_frames (no padding)
    # Strides
    stride_hidden_b,
    stride_hidden_ts,
    stride_hidden_c,
    stride_mouse_b,
    stride_mouse_t,
    stride_mouse_c,
    stride_fused_b,
    stride_fused_s,
    stride_fused_t,
    stride_fused_c,
    # Preprocessing Params
    V,  # vae_time_compression_ratio
    W,  # windows_size
    T_OFFSET,
    # Constexprs for block sizes (Triton 性能调优的关键)
    BLOCK_C_HIDDEN: tl.constexpr,
    BLOCK_C_MOUSE: tl.constexpr,
    PAD_T: tl.constexpr,  # pad_t = V * W
):
    """
    Triton kernel to fuse mouse preprocessing.
    Grid: (B, S, T_q)

    Input layout: hidden_states[B, T*S, C], accessed as [B, T, S, C]
    Output layout: fused[B, S, T, C_hidden + PAD_T * C_mouse]

    Simulates PyTorch padding logic by clamping negative indices to 0.
    PyTorch formula: mouse_condition_padded[:, V*(i-W)+pad_t : i*V+pad_t, :]
    where padding is the first frame replicated pad_t times.
    """
    # 1. Get Program IDs (Grid indices)
    pid_b = tl.program_id(axis=0)  # Index for B dimension
    pid_s = tl.program_id(axis=1)  # Index for S dimension
    pid_t_local = tl.program_id(axis=2)  # Index for T_q dimension

    pid_t_global = pid_t_local + T_OFFSET

    # =================================================================
    # Part 1: Copy hidden_states
    # =================================================================

    # Offsets for the C_hidden dimension
    c_hidden_offsets = tl.arange(0, BLOCK_C_HIDDEN)
    c_hidden_mask = c_hidden_offsets < C_hidden

    # Input Pointers for hidden_states[pid_b, pid_t * S + pid_s, :]
    # Original layout: B, T*S, C
    hidden_in_ptr = (
        HIDDEN_STATES_PTR
        + pid_b * stride_hidden_b
        + (pid_t_local * S + pid_s) * stride_hidden_ts
        + c_hidden_offsets
    )

    # Output Pointers for fused[pid_b, pid_s, pid_t, 0:C_hidden]
    fused_out_hidden_ptr = (
        FUSED_PTR
        + pid_b * stride_fused_b
        + pid_s * stride_fused_s
        + pid_t_local * stride_fused_t
        + c_hidden_offsets
    )

    # Load and store
    hidden_vec = tl.load(hidden_in_ptr, mask=c_hidden_mask)
    tl.store(fused_out_hidden_ptr, hidden_vec, mask=c_hidden_mask)

    # =================================================================
    # Part 2: Gather and flatten mouse_condition window
    # =================================================================
    # PyTorch formula: mouse_condition_padded[:, V*(i-W)+pad_t : i*V+pad_t, :]
    # where mouse_condition_padded = cat([pad, mouse_condition], dim=1)
    # and pad is the first frame replicated pad_t times.
    #
    # Padded array layout: [pad_t copies of frame 0][original frames 0 to N_frames-1]
    # To access padded_array[idx]:
    #   - if idx < pad_t: read original_array[0]
    #   - if idx >= pad_t: read original_array[idx - pad_t]

    c_mouse_offsets = tl.arange(0, BLOCK_C_MOUSE)
    c_mouse_mask = c_mouse_offsets < C_mouse

    for k in tl.static_range(PAD_T):
        # Index in the padded coordinate system
        # PyTorch slice: [V*(i-W)+pad_t : i*V+pad_t]
        # For each k, we read: padded_idx = V*(i-W) + pad_t + k
        padded_idx = V * (pid_t_global - W) + PAD_T + k

        # Map padded index to original array index
        # padded_idx < PAD_T → read from frame 0 (padding)
        # padded_idx >= PAD_T → read from frame (padded_idx - PAD_T)
        original_idx = padded_idx - PAD_T
        safe_idx = tl.where(original_idx < 0, 0, original_idx)

        # Also clamp upper bound to prevent out-of-bounds access
        safe_idx = tl.where(safe_idx >= N_frames, N_frames - 1, safe_idx)

        mouse_in_ptr = (
            MOUSE_CONDITION_PTR
            + pid_b * stride_mouse_b
            + safe_idx * stride_mouse_t
            + c_mouse_offsets
        )

        fused_out_mouse_ptr = (
            FUSED_PTR
            + pid_b * stride_fused_b
            + pid_s * stride_fused_s
            + pid_t_local * stride_fused_t
            + (C_hidden + k * C_mouse)
            + c_mouse_offsets
        )

        mouse_vec = tl.load(mouse_in_ptr, mask=c_mouse_mask)
        tl.store(fused_out_mouse_ptr, mouse_vec, mask=c_mouse_mask)


def mouse_preprocessor_triton(
    hidden_states: torch.Tensor,
    mouse_condition: torch.Tensor,
    temporal_shape: int,
    vae_time_compression_ratio: int,
    windows_size: int,
    is_causal: bool = False,
    num_frame_per_block: int = -1,
) -> torch.Tensor:
    """
    Optimized version that accepts B (T*S) C layout directly.

    Args:
        hidden_states: [B, T*S, C_hidden] tensor
        mouse_condition: [B, N_frames, C_mouse] tensor
        temporal_shape: T (temporal dimension), used to split T*S into T and S

    Returns:
        fused: [B, S, T, C_hidden + pad_t * C_mouse] tensor
    """
    # Ensure tensors are contiguous for correct stride calculation in Triton
    if not hidden_states.is_contiguous():
        # print(f"[DEBUG] Making hidden_states contiguous...")
        hidden_states = hidden_states.contiguous()
    if not mouse_condition.is_contiguous():
        print("[DEBUG] Making mouse_condition contiguous...")
        mouse_condition = mouse_condition.contiguous()

    B, T_S, C_hidden = hidden_states.shape
    _, N_frames, C_mouse = mouse_condition.shape

    V = vae_time_compression_ratio
    W = windows_size
    pad_t = V * W

    # Use provided temporal_shape to split T*S
    if is_causal:
        assert num_frame_per_block > 0, "Must provide num_frame_per_block > 0 in causal mode"
        T_q = num_frame_per_block
        # Calculate t_offset from N_frames
        N_feats = (N_frames - 1) // V + 1
        t_offset = N_feats - num_frame_per_block
    else:
        T_q = temporal_shape
        t_offset = 0

    assert T_S % T_q == 0, f"T*S ({T_S}) must be divisible by T ({T_q})"
    S = T_S // T_q

    # Output shape: [B, S, T, C_hidden + pad_t * C_mouse]
    fused_shape = (B, S, T_q, C_hidden + pad_t * C_mouse)
    fused = torch.empty(fused_shape, device=hidden_states.device, dtype=hidden_states.dtype)

    grid = (B, S, T_q)

    BLOCK_C_HIDDEN = triton.next_power_of_2(C_hidden)
    BLOCK_C_MOUSE = triton.next_power_of_2(C_mouse)

    mouse_preprocessor_kernel[grid](
        hidden_states,
        mouse_condition,
        fused,
        B,
        T_q,
        S,
        C_hidden,
        N_frames,
        C_mouse,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        mouse_condition.stride(0),
        mouse_condition.stride(1),
        mouse_condition.stride(2),
        fused.stride(0),
        fused.stride(1),
        fused.stride(2),
        fused.stride(3),
        V,
        W,
        t_offset,
        BLOCK_C_HIDDEN=BLOCK_C_HIDDEN,
        BLOCK_C_MOUSE=BLOCK_C_MOUSE,
        PAD_T=pad_t,
    )
    return fused
