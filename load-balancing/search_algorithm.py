r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch


    
def _global_policy(all_experts_count, all_global_expert_count, num_expert, world_size, d_model, fused, layer_index, frequency=1):
    # if os.environ.get("Planner")=='1':

    # =============Benchmark test=============
    # Bd_AGG = 24 * 1e9 / 8
    # Bd_Trans = 20 * 1e9 / 8
    Bd_AGG = 15 * 1e9 / 8
    Bd_Trans = 15 * 1e9 / 8 
    bw_net = 50 * 1e9 / 8
    Comp_tp = 35.7e12

    # Bd_A2A = [1.5 * 1e9, 3.8 * 1e9, 7.38 * 1e9, 22.5 * 1e9, 34.38 * 1e9, 40.5 * 1e9]   # Tutel original A2A without NVLink on 16GPUs
    Bd_A2A = [6 * 1e9 / 8, 16 * 1e9/8, 42 * 1e9/8, 100 * 1e9/8, 100 * 1e9/8, 100 * 1e9/8]   # Tutel original A2A with/without NVLink
    # Bd_A2A = [100 * 1e9 / 8, 100 * 1e9/8, 100 * 1e9/8, 100 * 1e9/8, 100 * 1e9/8, 100 * 1e9/8] # infiniband bandwidth

    if fused:
        all_experts_count = all_experts_count.sum(dim=-1).view(world_size, world_size, 1)
        all_global_expert_count = all_global_expert_count.sum(dim=-1).view(world_size, world_size, 1)
    
    fwd_expert_counts = all_global_expert_count.sum(1) # [world_size, num_expert] <===> [world_size, 1]

    default_counts = fwd_expert_counts.clone()
    comm_counts = fwd_expert_counts.clone()

    data_size = 4

    AlphaHsquared = alpha * d_model ** 2 * data_size
    Expert_size = AlphaHsquared * num_expert * 2
    comp_time = AlphaHsquared / Comp_tp
    
    A2A_Bytes = comm_counts.max(0)[0] * d_model * data_size
    if A2A_Bytes <= 128 * 1024: 
        Bd_A2A_profile = Bd_A2A[0]
    elif A2A_Bytes <= 256 * 1024:
        Bd_A2A_profile = Bd_A2A[1]
    elif A2A_Bytes <= 1024 * 1024:
        Bd_A2A_profile = Bd_A2A[2] 
    elif A2A_Bytes <= 16 * 1024 * 1024:
        Bd_A2A_profile = Bd_A2A[3]
    elif A2A_Bytes <= 256 * 1024 * 1024:
        Bd_A2A_profile = Bd_A2A[4]
    else :
        Bd_A2A_profile = Bd_A2A[5]        
    lat_comp = 12 * fwd_expert_counts.max(0)[0] * AlphaHsquared / Comp_tp  + 4 * comm_counts.max(0)[0] * d_model * data_size / Bd_A2A_profile

    min_time = lat_comp
    seq, Csmallest = [], []

    # TODO:Adaptive C
    C_max = 1
    C = min(min(torch.sum(all_global_expert_count.view(world_size, world_size * num_expert) == 0, dim=1)).item(), C_max)
    Coe = world_size - C

    i, cntt = 0, 0
    from megatron import get_args
    args = get_args()
    Used = [] 

    # 2bs^2h, 4bsh^2        
    FNEC = (2 * args.global_batch_size / args.world_size * args.seq_length ** 2 * d_model * 2 + 4 * args.global_batch_size / args.world_size * args.seq_length * d_model ** 2) * data_size / Comp_tp
    BNEC = 2 * FNEC

    balanced_condition = 3 * args.top_k * args.global_batch_size * args.seq_length / world_size

    while(fwd_expert_counts.max(0)[0] - fwd_expert_counts.min(0)[0] >= balanced_condition):

        maxx_idx = torch.argmax(fwd_expert_counts, 0).item()
        if maxx_idx in Used:
            break
        Used.append(maxx_idx)

        _, Csmallest_index = torch.topk(all_global_expert_count[maxx_idx], k=C, dim=0, largest=False)

        fwd_expert_counts[maxx_idx] = 0
        fwd_expert_counts += all_global_expert_count[maxx_idx].view(world_size, -1)

        comm_counts -= all_global_expert_count[maxx_idx].view(world_size, -1)
        comm_counts[maxx_idx] = all_global_expert_count[maxx_idx].view(world_size, -1)[maxx_idx]


        A2A_Bytes = comm_counts.max(0)[0] * d_model * data_size
        if A2A_Bytes <= 128 * 1024: 
            Bd_A2A_profile = Bd_A2A[0]
        elif A2A_Bytes <= 256 * 1024:
            Bd_A2A_profile = Bd_A2A[1]
        elif A2A_Bytes <= 1024 * 1024:
            Bd_A2A_profile = Bd_A2A[2]
        elif A2A_Bytes <= 16 * 1024 * 1024:
            Bd_A2A_profile = Bd_A2A[3]
        elif A2A_Bytes <= 256 * 1024 * 1024:
            Bd_A2A_profile = Bd_A2A[4]
        else :
            Bd_A2A_profile = Bd_A2A[5]

        FEC = fwd_expert_counts.max(0)[0] * comp_time * 4
        BEC = 2 * FEC
        comm_Trans = (i+1) * Expert_size * Coe / world_size / Bd_Trans
        comm_AGG = (i+1) * Expert_size * Coe / world_size / Bd_AGG
        # lat_comm = 12 * fwd_expert_counts.max(0)[0] * comp_time + (i+1) * Expert_size / bw_net   # fasterMoE
        # lat_comm = 12 * fwd_expert_counts.max(0)[0] * comp_time + (i+1) * Expert_size / Bd_Trans + (i+1) * Expert_size / Bd_AGG + 4 * comm_counts.max(0)[0] * d_model * data_size / Bd_A2A_profile  # Planner
        lat_comm = 12 * fwd_expert_counts.max(0)[0] * comp_time + 4 * comm_counts.max(0)[0] * d_model * data_size / Bd_A2A_profile + max(0, comm_Trans - FEC - FNEC) + max(0, comm_AGG - BEC - BNEC) # Prophet      

        
        if lat_comm < min_time:
            min_time = lat_comm
            cntt+=1

        seq.append(maxx_idx)
        Csmallest.append(Csmallest_index.tolist() if len(Csmallest_index.tolist())==0 else Csmallest_index.tolist()[0])
    # POE
    return seq[:cntt], Csmallest[:cntt]