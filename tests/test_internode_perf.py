import os
import sys
import time
import numpy as np

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
import utils
from utils import init_dist, bench, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency
try:
    from paperf import profile_torch
    has_paperf = True
except ImportError:
    has_paperf = False


def init_random_tensors(rank, num_nodes, num_tokens, hidden, num_topk_groups, num_topk, num_experts, dump_input=False):
    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)

    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    if dump_input:
        utils.dump(x, 'x', local_rank)
        utils.dump(x_pure_rand, 'x_pure_rand', local_rank)
        utils.dump(x_e4m3, 'x_e4m3', local_rank)

        utils.dump(topk_idx, 'topk_idx', local_rank)
        utils.dump(topk_weights, 'topk_weights', local_rank)
        utils.dump(topk_weights_pure_rand, 'topk_weights_pure_rand', local_rank)

    return x, x_pure_rand, x_e4m3, topk_idx, topk_weights, topk_weights_pure_rand


def load_dumped_tensors(rank, num_tokens, hidden, num_topk_groups, num_topk, num_experts):
    # x = utils.load("x", local_rank)
    # x_pure_rand = utils.load("x_pure_rand", local_rank)
    # #x_e4m3 = utils.load("x_e4m3", local_rank, "tuple")

    # topk_idx = utils.load("topk_idx", local_rank)
    # topk_weights = utils.load("topk_weights", local_rank)
    # topk_weights_pure_rand = utils.load("topk_weights_pure_rand", local_rank)

    def _load_tensor(rank, name, idx, typehint="tensor"):
        dump_dir = "/root/paddlejob/workspace/env_run/liuyiqun/outputs/ds_8nodes"
        filename = f"{dump_dir}/{idx}_{name}_rank{rank}.npy"
        if typehint == "tensor":
            x_np = np.load(filename)
            if x_np.dtype == np.uint16:
                x = torch.tensor(x_np, device='cuda').view(torch.bfloat16)
            elif x_np.dtype in [np.float32, np.int32, np.int64, np.bool, np.int8]:
                x = torch.tensor(x_np, device='cuda')
            else:
                assert False, f'{name}: {x_np.dtype}'
            return x
        else:
            assert False, f'invalid typehint: {typehint}'

    input_tensors = []
    for i in range(20):
        x = _load_tensor(rank, "dispatch_x", i + 1)
        topk_idx = _load_tensor(rank, "topk_idx", i + 1)
        topk_weights = _load_tensor(rank, "topk_weights", i + 1)
        input_tensors.append({"x": x, "topk_idx": topk_idx, "topk_weights": topk_weights})

    return input_tensors


def test_main(num_sms: int, local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, buffer: deep_ep.Buffer, group: dist.ProcessGroup, use_random_input, dump_input):
    # Settings
    num_tokens = 4096
    hidden = 7168
    num_topk_groups = min(num_nodes, 4)
    num_topk = 8
    num_experts = (256 // num_ranks) * num_ranks

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    if use_random_input:
        x, x_pure_rand, x_e4m3, topk_idx, topk_weights, topk_weights_pure_rand = init_random_tensors(rank, num_nodes, num_tokens, hidden, num_topk_groups, num_topk, num_experts, dump_input)
    else:
        input_tensors = load_dumped_tensors(rank, num_tokens, hidden, num_topk_groups, num_topk, num_experts)

        x = input_tensors[0]["x"]
        topk_idx = input_tensors[0]["topk_idx"]
        topk_weights = input_tensors[0]["topk_weights"]

    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    current_node = rank // num_local_ranks
    mask_rdma_only = (rdma_idx != current_node) & (rdma_idx != -1)
    num_rdma_only_token_sent = mask_rdma_only.sum().item()
    #print(f"-- [local_rank={local_rank}, rank={rank}] num_rdma_token_sent: {num_rdma_token_sent}, num_rdma_token_sent_rdma_only: {num_rdma_only_token_sent}")

    ############################################################################################################
    # get_dispatch_layout
    ############################################################################################################

    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    t = bench(group, lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print()
    #torch.distributed.barrier()
    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)
    handle = None

    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
    dispatch_bf16_rdma_only_send_bytes = num_rdma_only_token_sent * hidden * 2
    dispatch_bf16_nvl_recv_bytes = 0 # unknown
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes
    combine_bf16_rdma_only_recv_bytes = dispatch_bf16_rdma_only_send_bytes

    if local_rank == 0:
        print()

    profile = False
    profile = profile and has_paperf

    if profile:
        profile_torch.switch_profile(0, 0, 1)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    #current_x_list = [x_e4m3, x]
    current_x_list = [x]
    for current_x in current_x_list:
        dtype_str = "FP8" if isinstance(current_x, tuple) else "BF16"
        if profile:
            profile_torch.push_record_event(f"Tune_Dispatch_{dtype_str}")

        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        rdma_only_send_bytes = (dispatch_bf16_rdma_only_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_only_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes

        #nvl_chunk_size_tuning_list = range(4, 33, 4)
        #rdma_chunk_size_tuning_list = range(4, 33, 4)
        nvl_chunk_size_tuning_list = [20]
        rdma_chunk_size_tuning_list = [28]
        for nvl_chunk_size in nvl_chunk_size_tuning_list:
            for rdma_chunk_size in rdma_chunk_size_tuning_list:
                config_str = f"sms={num_sms},nvl={nvl_chunk_size},{nvl_buffer_size},rdma={rdma_chunk_size},{rdma_buffer_size}"
                if profile:
                    profile_torch.push_record_event(f"Dispatch_{dtype_str}_Config({config_str})")

                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                if handle is not None:
                    tune_args = {'x': current_x, 'handle': handle, 'config': config}
                else:
                    tune_args = {
                        'x': current_x,
                        'num_tokens_per_rank': num_tokens_per_rank,
                        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                        'is_token_in_rank': is_token_in_rank,
                        'num_tokens_per_expert': num_tokens_per_expert,
                        'topk_idx': topk_idx,
                        'topk_weights': topk_weights,
                        'config': config
                    }
                result_times = bench(group, lambda: buffer.dispatch(**tune_args))
                t = result_times[0]
                cpu_t = result_times[3]

                if profile:
                    profile_torch.pop_record_event()

                if local_rank == 0:
                    rdma_send_GBs = rdma_send_bytes / 1e9 / t
                    rdma_only_send_GBs = rdma_only_send_bytes / 1e9 / t
                    nvl_recv_GBs = nvl_recv_bytes / 1e9 / t
                    print(f'[tuning] Dispatch ({dtype_str}): SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {rdma_send_GBs:.2f} GB/s (RDMA + NVL), {rdma_only_send_GBs:.2f} GB/s (RDMA), {nvl_recv_GBs:.2f} GB/s (NVL) (time: {t:.5f} s, cpu_time: {cpu_t:.5f} s)')

        if profile:
            profile_torch.pop_record_event()

    group.barrier()

    dispatch_config = deep_ep.Config(num_sms, 20, nvl_buffer_size, 28, rdma_buffer_size)
    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
        'config': dispatch_config
    }

    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)


    if profile:
        profile_torch.push_record_event(f"Tune_Combine_BF16")

    # Tune combine performance
    #nvl_chunk_size_tuning_list = range(1, 5, 1)
    #rdma_chunk_size_tuning_list = range(8, 33, 4)
    nvl_chunk_size_tuning_list = [1]
    rdma_chunk_size_tuning_list = [20]
    for nvl_chunk_size in nvl_chunk_size_tuning_list:
        for rdma_chunk_size in rdma_chunk_size_tuning_list:
            config_str = f"sms={num_sms},nvl={nvl_chunk_size},{nvl_buffer_size},rdma={rdma_chunk_size},{rdma_buffer_size}"
            if profile:
                profile_torch.push_record_event(f"Combine_BF16_Config({config_str})")

            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config}
            result_times = bench(group, lambda: buffer.combine(**tune_args))
            t = result_times[0]
            cpu_t = result_times[3]

            if profile:
                profile_torch.pop_record_event()

            if local_rank == 0:
                combine_bf16_rdma_recv_GBs = combine_bf16_rdma_recv_bytes / 1e9 / t
                combine_bf16_rdma_only_recv_GBs = combine_bf16_rdma_only_recv_bytes / 1e9 / t
                combine_bf16_nvl_send_GBs = combine_bf16_nvl_send_bytes / 1e9 / t
                print(f'[tuning] Combine: SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: {combine_bf16_rdma_recv_GBs:.2f} GB/s (RDMA + NVL), {combine_bf16_rdma_only_recv_GBs:.2f} GB/s (RDMA), {combine_bf16_nvl_send_GBs:.2f} GB/s (NVL) (time: {t:.5f} s, cpu_time: {cpu_t:.5f} s)')

    if profile:
        profile_torch.pop_record_event()

    if profile:
        profile_torch.switch_profile(1, 0, 1)


def test_loop(local_rank: int, num_local_ranks: int):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    buffer = deep_ep.Buffer(group, int(1e9), int(1e9), low_latency_mode=False)
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    use_random_input = False
    dump_input = False
    for i in (20, ):
        test_main(i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group, use_random_input, dump_input)
        if local_rank == 0:
            print()


if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)
