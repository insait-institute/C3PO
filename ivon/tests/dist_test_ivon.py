"""Phase-0 distributed IVON tests: sharded DTensor params on a 2-rank world.

CPU: torchrun --nproc_per_node=2 ivon/tests/dist_test_ivon.py       (gloo)
GPU: torchrun --nproc_per_node=2 ivon/tests/dist_test_ivon.py       (nccl + FSDP2 fully_shard)
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from ivon import IVON


def make_sharded_params(mesh, seed=0):
    torch.manual_seed(seed)  # same full tensors on every rank
    full = [torch.randn(8, 4), torch.randn(16)]
    return [nn.Parameter(distribute_tensor(t, mesh, [Shard(0)])) for t in full]


def set_grads(params, mesh, seed):
    torch.manual_seed(seed)
    for p in params:
        g = torch.randn(p.shape)
        p.grad = distribute_tensor(g, mesh, [Shard(0)])


def test_fully_shard_cuda(rank, world):
    """End-to-end FSDP2 path: fully_shard a real module, IVON over its DTensors."""
    mesh = init_device_mesh("cuda", (world,))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 16)).cuda()
    for layer in model:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    opt = IVON(model.parameters(), lr=1e-3, ess=1e4, hess_init=0.1, noise_seed=7)
    ref = [p.to_local().clone() for p in model.parameters()]

    # noise -> forward -> restore(train) -> step, as in the RL loop
    for _ in range(2):
        opt._sample_params()
        x = torch.randn(4, 64, device="cuda")
        loss = model(x).square().mean()
        loss.backward()
        opt._restore_param_average(train=True)
        opt.step()
        opt.zero_grad()

    assert opt.param_groups[0]["hess"].dtype == torch.float32
    assert any(
        not torch.equal(p.to_local(), r) for p, r in zip(model.parameters(), ref)
    ), "params did not update"

    # offload/load roundtrip actually moves IVON buffers
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

    offload_fsdp_optimizer(opt)
    assert opt.param_groups[0]["hess"].device.type == "cpu"
    assert opt.param_groups[0]["momentum"].device.type == "cpu"
    load_fsdp_optimizer(opt, torch.cuda.current_device())
    assert opt.param_groups[0]["hess"].device.type == "cuda"

    # noise/denoise still works after an offload/load cycle, and the CUDA
    # (Philox) noise stream replays bit-identically from the saved generator
    # state — the invariant the store-free noise design depends on.
    before = [p.to_local().clone() for p in model.parameters()]
    opt._sample_params()
    assert torch.equal(opt._regenerate_noise(), opt._regenerate_noise()), "CUDA noise replay is not deterministic"
    opt._restore_param_average(train=False)
    for p, b in zip(model.parameters(), before):
        assert torch.allclose(p.to_local(), b, atol=1e-6), "roundtrip changed params on CUDA"
    if rank == 0:
        print("FSDP2 fully_shard CUDA test passed")


def main():
    use_cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "run with --nproc_per_node=2"
    if use_cuda:
        torch.cuda.set_device(rank)
        test_fully_shard_cuda(rank, world)
    mesh = init_device_mesh("cpu", (world,)) if not use_cuda else init_device_mesh("cuda", (world,))

    # --- local_numel bookkeeping ---
    params = make_sharded_params(mesh)
    opt = IVON(params, lr=1e-3, ess=1e4, hess_init=0.1, noise_seed=42)
    expected_local = sum(p.to_local().numel() for p in params)
    assert opt._local_numel == expected_local, (opt._local_numel, expected_local)
    assert opt._numel == sum(p.numel() for p in params)

    # --- per-rank noise decorrelation (same base seed, +rank offset) ---
    opt._sample_params()
    noise = opt._regenerate_noise()
    gathered = [torch.zeros_like(noise) for _ in range(world)]
    dist.all_gather(gathered, noise)
    assert not torch.equal(gathered[0], gathered[1]), "per-rank noise is identical -> correlated sample"

    # --- roundtrip restores exact local shards ---
    opt._restore_param_average(train=False)
    params2 = make_sharded_params(mesh)  # regenerate reference
    for p, ref in zip(params, params2):
        assert torch.allclose(p.to_local(), ref.to_local(), atol=1e-6), "roundtrip changed params"

    # --- a full sampled-grad + step cycle runs and keeps fp32 state ---
    opt._sample_params()
    set_grads(params, mesh, seed=rank + 100)
    opt._restore_param_average(train=True)
    opt.step()
    assert opt.param_groups[0]["hess"].dtype == torch.float32
    assert opt.param_groups[0]["momentum"].dtype == torch.float32

    # --- sync guard: sync=True with DTensor params must raise ---
    try:
        IVON(make_sharded_params(mesh), lr=1e-3, ess=1e4, sync=True)
        raise AssertionError("sync=True with DTensor params did not raise")
    except ValueError:
        pass

    # --- HSDP guard: replicated placement on a >1 mesh dim must raise ---
    torch.manual_seed(0)
    replicated = [nn.Parameter(distribute_tensor(torch.randn(8, 4), mesh, [Replicate()]))]
    try:
        IVON(replicated, lr=1e-3, ess=1e4)
        raise AssertionError("Replicate placement did not raise")
    except ValueError:
        pass

    # --- offload helper moves IVON buffers (device no-op on cpu, but must cover them) ---
    from verl.utils.fsdp_utils import _move_optimizer_state

    _move_optimizer_state(opt, "cpu")
    assert opt.param_groups[0]["hess"].device.type == "cpu"

    dist.barrier()
    if rank == 0:
        print("ALL DISTRIBUTED TESTS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
