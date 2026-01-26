# Pro-Prophet

This repository contains code of Pro-Prophet proposed in arxiv "Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models". 

We implement Pro-Prophet base on Pytorch and FasterMoE. 

## Repository Structure

```python
Pro-Prophet/
├── README.md
├── fastermoe_ppopp22_ae               # Code for FasterMoE. Downloading from https://zenodo.org/records/5728493
├── load-balancing                     # Code for Load-balancing. 
  ├── a2a_and_ec.cuh                   # Code for All-to-All and Expert-Computation operators
  ├── search_algorithm.py              # Code for Search Algorithm
  ├── trans_and_agg.cuh                # Code for Trans and Agg primitives
```

## Prerequisites
Installing PyTorch 1.10.0, CUDA 11.8, cuDNN 8.2.1, NCCL 2.10.3.
Relpacing operators of load-balancing and adapting them to FasterMoE.

## Usage 
srun fastermoe_ppopp22_ae/chaosflow/Megatron-LM/examples/pretrain_MoE-GPT.sh

## Citation

If you find this work useful, please cite:
Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

```
@article{wang2024pro,
  title={Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models},
  author={Wang, Wei and Lai, Zhiquan and Li, Shengwei and Liu, Weijie and Ge, Keshi and Shen, Ao and Su, Huayou and Li, Dongsheng},
  journal={arXiv preprint arXiv:2411.10003},
  year={2024}
}
```

