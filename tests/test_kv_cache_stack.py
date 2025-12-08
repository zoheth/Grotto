import torch

from grotto.modeling.kv_cache.dual_plane import DualPlaneKVCache


def test_push_pop_single_latent():
    cache = DualPlaneKVCache(
        max_seq_len=100,
        max_incoming_len=20,
        num_heads=4,
        head_dim=64,
        tokens_per_latent=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert cache.latent_count == 0
    assert cache.total_tokens == 0

    k = torch.randn(5, 4, 64)
    v = torch.randn(5, 4, 64)

    cache.push_latent().execute_append(k, v)
    assert cache.latent_count == 1
    assert cache.total_tokens == 5

    removed = cache.pop_latent(1)
    assert removed == 5
    assert cache.latent_count == 0
    assert cache.total_tokens == 0


def test_push_pop_multiple_latents():
    cache = DualPlaneKVCache(
        max_seq_len=100,
        max_incoming_len=20,
        num_heads=4,
        head_dim=64,
        tokens_per_latent=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    k = torch.randn(5, 4, 64)
    v = torch.randn(5, 4, 64)

    for _ in range(5):
        cache.push_latent().execute_append(k, v)

    assert cache.latent_count == 5
    assert cache.total_tokens == 25

    removed = cache.pop_latent(3)
    assert removed == 15
    assert cache.latent_count == 2
    assert cache.total_tokens == 10

    removed = cache.pop_latent(2)
    assert removed == 10
    assert cache.latent_count == 0
    assert cache.total_tokens == 0


def test_pop_more_than_available():
    cache = DualPlaneKVCache(
        max_seq_len=100,
        max_incoming_len=20,
        num_heads=4,
        head_dim=64,
        tokens_per_latent=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    k = torch.randn(5, 4, 64)
    v = torch.randn(5, 4, 64)

    cache.push_latent().execute_append(k, v)
    cache.push_latent().execute_append(k, v)

    assert cache.latent_count == 2

    try:
        cache.pop_latent(3)
    except ValueError as e:
        assert "Cannot pop 3 latents" in str(e)


def test_data_integrity_after_pop():
    cache = DualPlaneKVCache(
        max_seq_len=100,
        max_incoming_len=20,
        num_heads=4,
        head_dim=64,
        tokens_per_latent=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    k1 = torch.ones(5, 4, 64)
    k2 = torch.ones(5, 4, 64) * 2
    k3 = torch.ones(5, 4, 64) * 3
    v = torch.randn(5, 4, 64)

    cache.push_latent().execute_append(k1, v)
    cache.push_latent().execute_append(k2, v)
    cache.push_latent().execute_append(k3, v)

    assert cache.latent_count == 3

    cache.pop_latent(1)
    assert cache.latent_count == 2

    k_read, _ = cache.get_linear_view()
    assert k_read.shape[0] == 10
    assert torch.allclose(k_read[:5], k1)
    assert torch.allclose(k_read[5:], k2)


def test_reset():
    cache = DualPlaneKVCache(
        max_seq_len=100,
        max_incoming_len=20,
        num_heads=4,
        head_dim=64,
        tokens_per_latent=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    k = torch.randn(5, 4, 64)
    v = torch.randn(5, 4, 64)

    cache.push_latent().execute_append(k, v)
    cache.push_latent().execute_append(k, v)

    assert cache.latent_count == 2
    assert cache.total_tokens == 10

    cache.reset()

    assert cache.latent_count == 0
    assert cache.total_tokens == 0


if __name__ == "__main__":
    print("Running test_push_pop_single_latent...")
    test_push_pop_single_latent()
    print("✓ Passed")

    print("Running test_push_pop_multiple_latents...")
    test_push_pop_multiple_latents()
    print("✓ Passed")

    print("Running test_pop_more_than_available...")
    test_pop_more_than_available()
    print("✓ Passed")

    print("Running test_data_integrity_after_pop...")
    test_data_integrity_after_pop()
    print("✓ Passed")

    print("Running test_reset...")
    test_reset()
    print("✓ Passed")

    print("\nAll tests passed!")
