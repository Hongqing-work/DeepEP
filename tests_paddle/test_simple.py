import os
import sys
import time
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import paddle.distributed.communication.deep_ep as deep_ep
from paddle.distributed.communication.group import Group
from paddle.base.core import Config
import numpy as np

# noinspection PyUnresolvedReferences
import utils
from utils import bench, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
#import test_low_latency
from paperf import profile_paddle


def test_main(num_sms: int, local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, buffer: deep_ep.Buffer, group: Group):
    # Settings
    num_tokens, hidden, num_topk_groups, num_topk, num_experts = 4096, 7168, min(num_nodes, 4), 8, (256 // num_ranks) * num_ranks
    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    x = utils.load("x", local_rank)
    x_pure_rand = utils.load("x_pure_rand", local_rank)
    x_e4m3 = utils.load("x_e4m3", local_rank, "tuple")

    topk_idx = utils.load("topk_idx", local_rank)
    topk_weights = utils.load("topk_weights", local_rank)
    topk_weights_pure_rand = utils.load("topk_weights_pure_rand", local_rank)

    num_tokens_per_rank = utils.load('num_tokens_per_rank', local_rank)
    num_tokens_per_rdma_rank = utils.load('num_tokens_per_rdma_rank', local_rank)
    is_token_in_rank = utils.load('is_token_in_rank', local_rank)
    num_tokens_per_expert = utils.load('num_tokens_per_expert', local_rank)
    gbl_num_tokens_per_rank = utils.load('gbl_num_tokens_per_rank', local_rank)
    gbl_num_tokens_per_expert = utils.load('gbl_num_tokens_per_expert', local_rank)

    dispatch_config = Config(24, 20, 512, 32, 128)

    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config}

    #if local_rank == 0:
    #    print_tensor_info(x, "x")
    #    print_tensor_info(num_tokens_per_rank, "num_tokens_per_rank")
    #    print_tensor_info(num_tokens_per_rdma_rank, "num_tokens_per_rdma_rank")
    #    print_tensor_info(is_token_in_rank, "is_token_in_rank")
    #    print_tensor_info(num_tokens_per_expert, "num_tokens_per_expert")
    #    print(f"-- dispatch_args: {dispatch_args}")
    #    print(f"-- dispatch_config: {best_dispatch_results[0]}, {best_dispatch_results[1]}, {nvl_buffer_size}, {best_dispatch_results[2]}, {rdma_buffer_size}")

    for i in range(20):
        profile_paddle.switch_profile(i, 5, 15)
        recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)


def test_loop(num_local_ranks: int):
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()

    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    #local_rank, rank, num_ranks, group = utils.init_dist(num_local_ranks)
    
    test_ll_compatibility = False
    buffer = deep_ep.Buffer(group, int(1e9), int(1e9), low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1))

    num_ranks = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_rank = rank % 8
    print(f'local_rank:{local_rank}, num_local_ranks:{num_local_ranks}, num_ranks:{num_ranks}, rank:{rank}')

    assert num_local_ranks == 8 and num_ranks > 8
    paddle.seed(rank)

    for i in (24, ):
        test_main(i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)
        if local_rank == 0:
            print()


if __name__ == '__main__':
    num_processes = 8
    #torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
    #num_processes = 8
    #paddle.distributed.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)

    world_size = int(os.getenv('WORLD_SIZE', 1))
    mp_degree = world_size * num_processes
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "mp_degree": mp_degree,
        "pp_degree": 1,
        "dp_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    test_loop(num_processes)
