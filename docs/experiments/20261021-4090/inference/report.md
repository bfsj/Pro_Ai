# 单卡 LLM 推理实验报告

生成时间（UTC）：2026-10-21T13:11:37.620683+00:00。状态：completed。

## 环境与方法

- GPU：NVIDIA GeForce RTX 4090；PyTorch 2.11.0+cu128；Transformers 4.57.6。
- 源码提交：6eb05740822544f35f477d27803b0f4785b229ab；工作区有未提交修改：False。
- 模型：/data/optics/JD/SN/models/Qwen3-0.6B；精度：float32；Attention：sdpa。
- 输出长度：32；预热：2；正式重复：5。
- 等长随机 token 输入、固定输出长度、忽略 EOS；各策略使用相同输入，重复内交错顺序。
- 计时为同步后的本地 wall-clock，包含 Python、采样与缓存初始分配；排除权重加载、分词、输入搬运和预热。
- Prefill 包含首 token；后续 Decode 共 N-1 步。平均每步耗时不是在线 ITL/P95 或服务 TTFT。

## 实测结果

![实测曲线](performance.png)

| Batch | 输入 tokens | Cache | Norm | 吞吐 tokens/s | Prefill+首 token ms | Decode/步 ms | 峰值 allocated MiB | 额外峰值 MiB | CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 128 | dynamic | torch | 48.914 | 24.660 | 20.315 | 2320.925 | 38.549 | 0.001 |
| 1 | 128 | dynamic | triton | 52.114 | 23.119 | 19.062 | 2320.925 | 38.549 | 0.024 |
| 1 | 128 | no_cache | torch | 40.083 | 25.409 | 24.921 | 2290.730 | 8.354 | 0.079 |
| 1 | 128 | no_cache | triton | 42.285 | 24.101 | 23.645 | 2290.730 | 8.354 | 0.019 |
| 1 | 128 | static | torch | 48.304 | 22.963 | 20.632 | 2323.805 | 41.429 | 0.026 |
| 1 | 128 | static | triton | 51.265 | 21.933 | 19.426 | 2323.805 | 41.429 | 0.026 |
| 1 | 512 | dynamic | torch | 46.946 | 25.692 | 21.011 | 2453.896 | 171.517 | 0.046 |
| 1 | 512 | dynamic | triton | 52.447 | 24.217 | 18.903 | 2453.896 | 171.517 | 0.009 |
| 1 | 512 | no_cache | torch | 34.457 | 28.622 | 29.043 | 2313.805 | 31.426 | 0.007 |
| 1 | 512 | no_cache | triton | 36.140 | 27.343 | 27.672 | 2312.958 | 30.578 | 0.001 |
| 1 | 512 | static | torch | 45.746 | 25.381 | 21.689 | 2427.935 | 145.555 | 0.031 |
| 1 | 512 | static | triton | 49.990 | 23.703 | 19.885 | 2427.935 | 145.555 | 0.024 |
| 1 | 1024 | dynamic | torch | 46.136 | 53.451 | 20.668 | 2699.416 | 417.033 | 0.025 |
| 1 | 1024 | dynamic | triton | 48.374 | 51.317 | 19.690 | 2699.416 | 417.033 | 0.024 |
| 1 | 1024 | no_cache | torch | 18.869 | 47.180 | 53.174 | 2342.764 | 60.381 | 0.002 |
| 1 | 1024 | no_cache | triton | 19.538 | 45.308 | 51.375 | 2342.764 | 60.381 | 0.002 |
| 1 | 1024 | static | torch | 46.620 | 44.488 | 20.713 | 2567.211 | 284.828 | 0.023 |
| 1 | 1024 | static | triton | 49.859 | 42.614 | 19.329 | 2567.211 | 284.828 | 0.081 |
| 4 | 128 | dynamic | torch | 191.581 | 24.770 | 20.756 | 2437.629 | 155.250 | 0.005 |
| 4 | 128 | dynamic | triton | 196.813 | 23.482 | 20.182 | 2437.613 | 155.234 | 0.019 |
| 4 | 128 | no_cache | torch | 160.066 | 26.879 | 24.923 | 2315.448 | 33.068 | 0.036 |
| 4 | 128 | no_cache | triton | 169.428 | 25.602 | 23.559 | 2315.448 | 33.068 | 0.005 |
| 4 | 128 | static | torch | 189.641 | 23.181 | 21.022 | 2448.218 | 165.838 | 0.020 |
| 4 | 128 | static | triton | 203.655 | 22.035 | 19.566 | 2451.608 | 169.229 | 0.101 |
| 4 | 512 | dynamic | torch | 170.568 | 84.682 | 21.394 | 2963.944 | 681.553 | 0.025 |
| 4 | 512 | dynamic | triton | 185.419 | 82.294 | 19.599 | 2963.944 | 681.553 | 0.027 |
| 4 | 512 | no_cache | torch | 47.866 | 77.918 | 83.734 | 2398.872 | 116.480 | 0.000 |
| 4 | 512 | no_cache | triton | 49.231 | 75.000 | 81.440 | 2398.872 | 116.480 | 0.001 |
| 4 | 512 | static | torch | 175.474 | 75.566 | 21.052 | 2862.302 | 579.911 | 0.090 |
| 4 | 512 | static | triton | 186.931 | 72.713 | 19.726 | 2862.302 | 579.911 | 0.018 |
| 4 | 1024 | dynamic | torch | 149.316 | 214.229 | 20.728 | 3935.510 | 1653.104 | 0.008 |
| 4 | 1024 | dynamic | triton | 157.189 | 209.909 | 19.479 | 3935.510 | 1653.104 | 0.008 |
| 4 | 1024 | no_cache | torch | 21.873 | 166.799 | 183.389 | 2519.567 | 237.160 | 0.000 |
| 4 | 1024 | no_cache | triton | 22.471 | 161.984 | 178.519 | 2519.020 | 236.613 | 0.000 |
| 4 | 1024 | static | torch | 155.081 | 168.113 | 21.205 | 3416.149 | 1133.743 | 0.022 |
| 4 | 1024 | static | triton | 165.954 | 163.247 | 19.619 | 3415.603 | 1133.196 | 0.016 |

## 配对比较

以下倍数只适用于同一行指定的工作负载，分子/分母均为重复测量的吞吐中位数。

| Batch | 输入 | 比较（分子 / 分母） | 吞吐倍数 |
| --- | --- | --- | --- |
| 1 | 128 | dynamic+torch / no_cache+torch | 1.220x |
| 1 | 128 | static+torch / dynamic+torch | 0.988x |
| 1 | 128 | dynamic+triton / dynamic+torch | 1.065x |
| 1 | 128 | static+triton / static+torch | 1.061x |
| 1 | 512 | dynamic+torch / no_cache+torch | 1.362x |
| 1 | 512 | static+torch / dynamic+torch | 0.974x |
| 1 | 512 | dynamic+triton / dynamic+torch | 1.117x |
| 1 | 512 | static+triton / static+torch | 1.093x |
| 1 | 1024 | dynamic+torch / no_cache+torch | 2.445x |
| 1 | 1024 | static+torch / dynamic+torch | 1.010x |
| 1 | 1024 | dynamic+triton / dynamic+torch | 1.049x |
| 1 | 1024 | static+triton / static+torch | 1.069x |
| 4 | 128 | dynamic+torch / no_cache+torch | 1.197x |
| 4 | 128 | static+torch / dynamic+torch | 0.990x |
| 4 | 128 | dynamic+triton / dynamic+torch | 1.027x |
| 4 | 128 | static+triton / static+torch | 1.074x |
| 4 | 512 | dynamic+torch / no_cache+torch | 3.563x |
| 4 | 512 | static+torch / dynamic+torch | 1.029x |
| 4 | 512 | dynamic+triton / dynamic+torch | 1.087x |
| 4 | 512 | static+triton / static+torch | 1.065x |
| 4 | 1024 | dynamic+torch / no_cache+torch | 6.826x |
| 4 | 1024 | static+torch / dynamic+torch | 1.039x |
| 4 | 1024 | dynamic+triton / dynamic+torch | 1.053x |
| 4 | 1024 | static+triton / static+torch | 1.070x |

## 验证边界与局限

- 本报告是性能结果，不能替代逐位置 logits 校验。请同时保留 suite 的 correctness-fp32.json；低精度失败不能写成通过。
- 静态缓存采用 Transformers StaticCache，动态缓存采用 DynamicCache；原创工作是策略集成、显式位置管理、控制实验与可选 Triton 算子。
- 动态缓存的拼接、静态缓存额外 mask 和容量开销都会影响结果，静态缓存不保证更快。
- 峰值 allocated 包含模型；额外峰值扣除运行前活跃分配。reserved 和整个进程显存是不同口径。
- 数据来自单机单卡和指定模型。没有验证多卡、生产服务、真实到达率或生产质量。
- 失败/OOM 记录：0；吞吐 CV > 10% 的配置：1。原始数据均保留。
- 初始与结束 GPU 快照在 measurements.json 中；共享机器的干扰仍可能影响速度。

## 复现配置

```json
{
  "model": "/data/optics/JD/SN/models/Qwen3-0.6B",
  "model_revision": null,
  "device": "cuda",
  "dtype": "float32",
  "attention": "sdpa",
  "batches": [
    1,
    4
  ],
  "input_lengths": [
    128,
    512,
    1024
  ],
  "new_tokens": 32,
  "cache_modes": [
    "no_cache",
    "dynamic",
    "static"
  ],
  "norm_backends": [
    "torch",
    "triton"
  ],
  "warmup": 2,
  "repeats": 5,
  "seed": 42,
  "workload": "equal-length seeded random token IDs; fixed output length; EOS ignored",
  "timing": "synchronized wall clock, prefill boundary synchronized, no per-token synchronization",
  "cache_lifecycle": "fresh cache on every trial; initial allocation included in prefill"
}
```
