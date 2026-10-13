import os
import socket
from multiprocessing import get_context

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from downstream_tasks.expression_prediction.distributed_vcc import (
            gather_perturbation_axis,
        )

        # P=5 deliberately exercises unequal shards and padding for world_size=2.
        full = torch.arange(10, dtype=torch.float64).reshape(2, 5)
        width = (full.shape[1] + world_size - 1) // world_size
        start, end = rank * width, min(full.shape[1], (rank + 1) * width)
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        model.weight.data.fill_(1.0)
        model = DistributedDataParallel(model)
        local_input = full[:, start:end].clone()
        local = model(local_input.unsqueeze(-1)).squeeze(-1)
        if local.shape[1] < width:
            local_padded = torch.cat(
                (local, local.new_zeros(local.shape[0], width - local.shape[1])), dim=1
            )
        else:
            local_padded = local

        gathered = gather_perturbation_axis(local_padded, full.shape[1])
        assert torch.equal(gathered.detach(), full)

        # Every rank evaluates the identical full loss. all_gather backward sums
        # those contributions; DDP then averages parameter gradients, recovering
        # the single-process gradient over the complete rectangle exactly.
        gathered.square().sum().backward()
        expected_weight_grad = 2 * full.square().sum()
        exact = torch.allclose(model.module.weight.grad.squeeze(), expected_weight_grad)
        queue.put((rank, exact, tuple(gathered.shape)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_differentiable_gather_reconstructs_rectangle_and_exact_gradient():
    ctx = get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    workers = [ctx.Process(target=_worker, args=(rank, 2, port, queue)) for rank in range(2)]
    for worker in workers:
        worker.start()
    results = [queue.get(timeout=30) for _ in workers]
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    assert sorted(results) == [(0, True, (2, 5)), (1, True, (2, 5))]
