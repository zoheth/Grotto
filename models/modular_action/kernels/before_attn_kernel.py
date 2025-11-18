import torch
import triton
import triton.language as tl
from typing import Dict, Tuple

# ----------------------------------------------------------------------
# 1. 第一个 Kernel：在 GPU 上计算所有索引
# ----------------------------------------------------------------------
@triton.jit
def _compute_indices_kernel(
    # 指针
    Global_End_Idx_ptr,  # [1], int64
    Local_End_Idx_ptr,   # [1], int64
    Params_Buffer_ptr,   # [8], int32. 用于存储所有计算出的索引
    # 参数
    num_new_tokens: tl.int32,
    cache_size: tl.int32,
    max_attention_size: tl.int32,
    sink_tokens: tl.int32
):
    """
    这是一个单线程 Kernel，在 GPU 上计算所有必要的索引和参数，
    并将它们写入 Params_Buffer，以供下一个 Fused Kernel 使用。
    
    Params_Buffer 索引:
    [0]: needs_eviction (0 or 1)
    [1]: num_rolled_tokens
    [2]: src_start
    [3]: dst_start
    [4]: local_start_index
    [5]: local_end_index (新的)
    [6]: window_start
    [7]: current_end (新的 global_end)
    """
    
    # --- 1. 原子性地读取和更新 Global Index ---
    # tl.atomic_add 返回的是 *旧的* 值
    global_end_idx = tl.atomic_add(Global_End_Idx_ptr, num_new_tokens)
    local_end_idx = tl.load(Local_End_Idx_ptr)
    
    current_end = global_end_idx + num_new_tokens

    # --- 2. 在 GPU 上计算所有驱逐（Eviction）参数 ---
    needs_eviction = (num_new_tokens + local_end_idx) > cache_size

    # Compute eviction parameters (always compute, select with where)
    # All computations must result in int32 to match params_buffer type
    num_evicted_tokens_if_evict = (num_new_tokens + local_end_idx - cache_size)
    num_rolled_tokens_if_evict = (local_end_idx - num_evicted_tokens_if_evict - sink_tokens)

    num_evicted_tokens = tl.where(needs_eviction, num_evicted_tokens_if_evict, 0)
    num_rolled_tokens = tl.where(needs_eviction, num_rolled_tokens_if_evict, 0)

    # src_start and dst_start depend on both needs_eviction AND num_rolled_tokens > 0
    has_roll = needs_eviction & (num_rolled_tokens > 0)
    src_start = tl.where(has_roll, sink_tokens + num_evicted_tokens, 0)
    dst_start = tl.where(has_roll, sink_tokens, 0)

    # new_local_end_index
    new_local_end_index = tl.where(needs_eviction,
                                   local_end_idx + num_new_tokens - num_evicted_tokens,
                                   local_end_idx + num_new_tokens)

    local_start_index = new_local_end_index - num_new_tokens

    # --- 3. 计算窗口（Window）参数 ---
    window_start_raw = new_local_end_index - max_attention_size
    window_start = tl.where(window_start_raw < 0, 0, window_start_raw)

    # --- 4. 将所有结果写入 Params_Buffer ---
    # Cast all values to int32 for storage in params_buffer
    tl.store(Params_Buffer_ptr + 0, needs_eviction.to(tl.int32))
    tl.store(Params_Buffer_ptr + 1, num_rolled_tokens.to(tl.int32))
    tl.store(Params_Buffer_ptr + 2, src_start.to(tl.int32))
    tl.store(Params_Buffer_ptr + 3, dst_start.to(tl.int32))
    tl.store(Params_Buffer_ptr + 4, local_start_index.to(tl.int32))
    tl.store(Params_Buffer_ptr + 5, new_local_end_index.to(tl.int32))
    tl.store(Params_Buffer_ptr + 6, window_start.to(tl.int32))
    tl.store(Params_Buffer_ptr + 7, current_end.to(tl.int32))

    # --- 5. 更新 Local Index ---
    # 注意：Global index 已经在开头用 atomic_add 更新过了
    # Store as int64 in the local_end_index tensor
    tl.store(Local_End_Idx_ptr, new_local_end_index.to(tl.int64))


# ----------------------------------------------------------------------
# 2. 第二个 Kernel：Fused [Mean + Eviction + Insertion]
# ----------------------------------------------------------------------
@triton.jit
def _fused_update_mean_kernel(
    # Cache Tensors
    K_cache_ptr, V_cache_ptr,
    # New Tensors (Source for Mean)
    K_new_src_ptr, V_new_src_ptr,
    # Params Buffer
    Params_Buffer_ptr,
    # Shape/Stride info
    S_dim: tl.int32,       # 'S' 维度 (用于 mean)
    num_heads: tl.int32,
    head_dim: tl.int32,
    num_new_tokens: tl.int32,
    
    stride_cache_bs, stride_cache_t, stride_cache_h, stride_cache_d,
    stride_new_s, stride_new_t, stride_new_h, stride_new_d,
    
    # Compile-time constants
    BLOCK_D: tl.constexpr
):
    """
    在一个 Fused Kernel 中执行 Mean, Eviction 和 Insertion。
    Grid: (num_heads, num_new_tokens + num_rolled_tokens)
    """
    
    # --- 1. 获取线程ID和维度 ---
    pid_h = tl.program_id(0)      # Head index
    pid_t_global = tl.program_id(1) # Global Time index (覆盖驱逐+插入)

    # 保护, 以防 grid_t 超出 num_heads
    if pid_h >= num_heads:
        return
        
    # --- 2. 从 Buffer 加载索引 (所有线程都加载) ---
    needs_eviction_int = tl.load(Params_Buffer_ptr + 0)
    num_rolled_tokens = tl.load(Params_Buffer_ptr + 1)
    src_start = tl.load(Params_Buffer_ptr + 2)
    dst_start = tl.load(Params_Buffer_ptr + 3)
    local_start_index = tl.load(Params_Buffer_ptr + 4)

    # Convert needs_eviction from int32 to boolean for conditional
    needs_eviction = needs_eviction_int != 0

    # --- 3. 初始化 Head Dim 偏移量 ---
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    # --- 4. Fused Eviction + Insertion ---

    # 任务分配：
    # pid_t_global < num_rolled_tokens 的线程处理 Eviction
    # pid_t_global >= num_rolled_tokens 的线程处理 Insertion

    if needs_eviction and pid_t_global < num_rolled_tokens:
        # --- 任务 A: 执行 Eviction (Roll) ---
        t = pid_t_global # t in [0, num_rolled_tokens)
        
        # 计算源和目标的指针
        src_k_ptr = (K_cache_ptr + 
                     (src_start + t) * stride_cache_t + 
                     pid_h * stride_cache_h + 
                     d_offsets * stride_cache_d)
        dst_k_ptr = (K_cache_ptr +
                     (dst_start + t) * stride_cache_t +
                     pid_h * stride_cache_h +
                     d_offsets * stride_cache_d)
                     
        src_v_ptr = (V_cache_ptr + 
                     (src_start + t) * stride_cache_t + 
                     pid_h * stride_cache_h + 
                     d_offsets * stride_cache_d)
        dst_v_ptr = (V_cache_ptr +
                     (dst_start + t) * stride_cache_t +
                     pid_h * stride_cache_h +
                     d_offsets * stride_cache_d)
        
        # 执行 DtoD 复制 (Eviction)
        k = tl.load(src_k_ptr, mask=d_mask, other=0.0)
        tl.store(dst_k_ptr, k, mask=d_mask)
        
        v = tl.load(src_v_ptr, mask=d_mask, other=0.0)
        tl.store(dst_v_ptr, v, mask=d_mask)
        
    elif pid_t_global >= num_rolled_tokens:
        # --- 任务 B: 执行 Fused Mean + Insertion ---
        t = pid_t_global - num_rolled_tokens # t in [0, num_new_tokens)
        
        # 仅在 t < num_new_tokens 范围内工作
        if t < num_new_tokens:
            # 初始化累加器 (用于 Mean)
            k_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            v_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            # (融合“之前”) --- Fused Mean 循环 ---
            for s_idx in range(S_dim):
                new_k_ptr = (K_new_src_ptr + 
                             s_idx * stride_new_s +
                             t * stride_new_t +
                             pid_h * stride_new_h +
                             d_offsets * stride_new_d)
                new_v_ptr = (V_new_src_ptr + 
                             s_idx * stride_new_s +
                             t * stride_new_t +
                             pid_h * stride_new_h +
                             d_offsets * stride_new_d)
                             
                k_new = tl.load(new_k_ptr, mask=d_mask, other=0.0)
                v_new = tl.load(new_v_ptr, mask=d_mask, other=0.0)
                
                k_acc = k_acc + k_new
                v_acc = v_acc + v_new
            
            # 计算均值
            k_mean = k_acc / S_dim
            v_mean = v_acc / S_dim

            # (融合“期间”) --- Fused Insertion 写入 ---
            dst_k_ptr = (K_cache_ptr +
                         (local_start_index + t) * stride_cache_t +
                         pid_h * stride_cache_h +
                         d_offsets * stride_cache_d)
            dst_v_ptr = (V_cache_ptr +
                         (local_start_index + t) * stride_cache_t +
                         pid_h * stride_cache_h +
                         d_offsets * stride_cache_d)
                         
            tl.store(dst_k_ptr, k_mean.to(K_cache_ptr.dtype.element_ty), mask=d_mask)
            tl.store(dst_v_ptr, v_mean.to(V_cache_ptr.dtype.element_ty), mask=d_mask)


def initialize_kv_cache_gpu(cache_k: torch.Tensor, cache_v: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    初始化 KV Cache，确保索引在 GPU 上。
    这是使用 update_kv_cache_triton 的 *前提*。
    """
    assert cache_k.device.type == 'cuda'
    # 确保索引是 GPU 上的 1 维张量
    return {
        "k": cache_k,
        "v": cache_v,
        "global_end_index": torch.tensor([0], device='cuda', dtype=torch.int64),
        "local_end_index": torch.tensor([0], device='cuda', dtype=torch.int64),
        # 缓存一个 params_buffer 以避免每次重新分配
        "_params_buffer": torch.zeros(8, device='cuda', dtype=torch.int32),
    }

def update_kv_cache_triton(
    kv_cache: Dict[str, torch.Tensor],
    # 注意：这里我们传入 "Mean" 的源张量
    k_new_source: torch.Tensor,
    v_new_source: torch.Tensor,
    max_attention_size: int,
    sink_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    使用 Triton Fused Kernel 更新 KV Cache。
    
    融合点:
    1. 零 D2H 更新：索引计算和更新完全在 GPU 上。
    2. 融合 "Before"：`mean` 操作在 Fused Kernel 内部完成。
    3. 融合 "During"：`Eviction (roll)` 和 `Insertion` 在同一个 Kernel 中完成。
    
    Args:
        kv_cache: 来自 initialize_kv_cache_gpu() 的字典。
        k_new_source: *用于 Mean 的源 K* [S, num_new_tokens, num_heads, head_dim]
        v_new_source: *用于 Mean 的源 V* [S, num_new_tokens, num_heads, head_dim]
        max_attention_size: 最大注意力窗口。
        sink_tokens: 沉没 Token 数量。

    Returns:
        k_window, v_window, local_start_index, local_end_index (Python ints)
    """
    
    # --- 1. 从张量形状中获取参数 (无 D2H) ---
    cache_k = kv_cache["k"] # [BS, CacheSize, H, D]
    
    # 假设 cache_k 的 BS 维始终为 1
    assert cache_k.size(0) == 1, "Triton Kernel 假设 Cache Batch Size 为 1"
    
    S_dim, num_new_tokens, num_heads, head_dim = k_new_source.shape
    cache_size = cache_k.size(1)
    
    global_end_idx_tensor = kv_cache["global_end_index"]
    local_end_idx_tensor = kv_cache["local_end_index"]

    # Create params_buffer if it doesn't exist (for backward compatibility)
    if "_params_buffer" not in kv_cache:
        kv_cache["_params_buffer"] = torch.zeros(8, device='cuda', dtype=torch.int32)
    params_buffer = kv_cache["_params_buffer"]

    # --- 2. 启动 Kernel 1: 计算索引 (异步) ---
    # 这是一个非常小的 Kernel (1 个线程)，几乎立即完成
    _compute_indices_kernel[(1,)](
        global_end_idx_tensor,
        local_end_idx_tensor,
        params_buffer,
        num_new_tokens=num_new_tokens,
        cache_size=cache_size,
        max_attention_size=max_attention_size,
        sink_tokens=sink_tokens,
    )

    # --- 3. 启动 Kernel 2: Fused [Mean + Evict + Insert] (异步) ---
    
    # BLOCK_D 必须是 head_dim 的下一个 2 次幂
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # 我们需要一个覆盖 (Eviction + Insertion) 所有时间的 Grid
    # 我们从 params_buffer[1] (num_rolled_tokens) 读取，但这是异步的
    # 因此，我们启动一个保守的、足够大的 grid_t
    # max_rolled_tokens 不会超过 cache_size
    grid_t = num_new_tokens + cache_size 
    
    # 更好的策略：读取 num_rolled_tokens，但这需要同步
    # 我们直接使用 params_buffer[1] (num_rolled_tokens)
    # 这 *依赖* Kernel 1 完成。
    # 为了安全，我们在这里同步或使用保守的 grid_t
    
    # 简单起见，我们在这里同步一次以获取 num_rolled_tokens
    # 这是一个 D2H，但只读一个 int，开销远小于 item()
    # params = params_buffer.tolist() # [D2H]
    # num_rolled_tokens = params[1]
    # grid_t = num_new_tokens + num_rolled_tokens
    
    # *无D2H的替代方案*：启动一个保守的 Grid
    # num_rolled_tokens 最多是 (cache_size - sink_tokens)
    max_possible_rolled = cache_size - sink_tokens
    grid_t = num_new_tokens + max_possible_rolled
    
    grid = (num_heads, grid_t)

    cache_v = kv_cache["v"]  # Get V cache

    _fused_update_mean_kernel[grid](
        cache_k, cache_v,  # K_cache, V_cache
        k_new_source, v_new_source,
        params_buffer,
        S_dim=S_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_new_tokens=num_new_tokens,
        
        # Strides (假设都是 contiguous)
        stride_cache_bs=cache_k.stride(0), stride_cache_t=cache_k.stride(1), 
        stride_cache_h=cache_k.stride(2), stride_cache_d=cache_k.stride(3),
        
        stride_new_s=k_new_source.stride(0), stride_new_t=k_new_source.stride(1), 
        stride_new_h=k_new_source.stride(2), stride_new_d=k_new_source.stride(3),
        
        BLOCK_D=BLOCK_D
    )

    # --- 4. 同步并返回 Python 值 ---
    # 这是我们唯一需要 D2H 的地方，以返回 Python int 索引
    # 这是为了匹配您原始 API 的签名
    params_py = params_buffer.tolist() # D2H (读取 8 个 int)
    
    local_start_index = params_py[4]
    local_end_index = params_py[5]
    window_start = params_py[6]

    # 返回 Cache 的 *视图* (View, no copy)
    # [1, CacheSize, H, D] -> [1, window_len, H, D]
    k_window = cache_k[:, window_start:local_end_index]
    v_window = kv_cache["v"][:, window_start:local_end_index]

    return k_window, v_window, local_start_index, local_end_index