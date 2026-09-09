# Pro_Ai

面向 AI Infra 实习准备的实践项目。第一阶段围绕 **单卡 LLM 推理、KV Cache 正确性与性能分析**，推荐在 RTX 4090 上使用 Qwen3-0.6B 完成实验。

代码提供可运行的参考实现；真正需要自己完成的是阅读、实验、解释瓶颈和提出改进。现有功能不应在简历中表述为个人从零独立实现。

## 当前范围

- 手写无缓存 / 有缓存的逐 token 生成循环。
- 使用固定 token 序列验证完整前向与缓存增量前向的 logits。
- 检查各层 KV shape，比较 GQA 缓存理论大小与实际张量大小。
- 批量测试输入长度、batch size 与缓存开关，保存每次测量和汇总结果。
- 生成吞吐、Decode 耗时与显存曲线。
- 导出单独的 PyTorch Profiler trace。
- 使用随机初始化的微型 Qwen3 运行离线 CPU 测试，配置 GitHub Actions。

**验证边界：** 初始版本已在 Python 3.13.5、PyTorch 2.11.0+cpu、Transformers 4.57.6 上通过 19 项测试，包含本地 checkpoint 的命令行集成检查。CPU 测试检查代码逻辑，不代表已在 4090 上验证性能。仓库没有预填任何 GPU 加速数字。GPU 测量需要在你实际使用的机器上执行。

## 环境准备

推荐 Linux + Python 3.11；Windows 可以使用 WSL2。第一阶段只依赖 PyTorch / Transformers，不需要安装 vLLM、FlashAttention 或编译自定义 CUDA 算子。

```bash
git clone https://github.com/bfsj/Pro_Ai.git
cd Pro_Ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择与你的 NVIDIA 驱动兼容的 CUDA 构建，先执行页面给出的 PyTorch 安装命令，再安装本项目：

```bash
python -m pip install -e ".[dev]"
python -m inference_lab env
```

GPU 实验要求输出中 `cuda_available` 为 `true`，且 GPU 型号与你准备使用的设备一致。`nvidia-smi` 中的 CUDA 版本表示驱动支持上限，不等同于 PyTorch 自带运行时版本。请保存实际环境信息。

第一次加载 `Qwen/Qwen3-0.6B` 会从 Hugging Face 下载权重，也可以把 `--model` 指向已有的本地模型目录。当前脚本仅支持普通 dense Qwen3，统一使用等长、无 padding 的输入。

## 今天先做这三步

```bash
# 1. 检查 logits、每层 K/V shape、权重和缓存大小
python -m inference_lab inspect

# 2. 跑通自己可逐行阅读的生成循环
python -m inference_lab generate --prompt "解释 Prefill 和 Decode 的区别。" --new-tokens 64
python -m inference_lab generate --prompt "解释 Prefill 和 Decode 的区别。" --new-tokens 64 --no-cache

# 3. 先用 FP32 建立更严格的缓存正确性基线
python -m inference_lab check --dtype float32 --input-length 128 --steps 8
```

阅读入口是 [`src/inference_lab/core.py`](src/inference_lab/core.py)。建议先画出一次 Prefill、两次 Decode 的张量变化，再尝试自行重写生成循环并运行测试。

生成命令关闭 Qwen3 thinking，使用 greedy，并固定生成步数、忽略 EOS。这是实验用循环，可能生成重复文字，不代表推荐的对话采样策略。

## 4090 上的性能实验

先跑小规模烟雾测试：

```bash
python -m inference_lab benchmark --batch-sizes 1 --input-lengths 128 --new-tokens 16 --warmup 1 --repeats 2 --output results/smoke.json
```

确认通过后，再运行 18 组配置。默认 BF16，输入长度 128/512/2048，batch 1/4/8，缓存开/关，输出固定 128 tokens；每组预热 2 次、测量 5 次：

```bash
python -m inference_lab check --dtype bfloat16
python -m inference_lab benchmark --output results/4090-baseline.json
python -m inference_lab plot --input results/4090-baseline.json --output results/4090-baseline.png
```

输出包含软件环境、模型 revision（能够获取时）、工作负载、原始逐次结果、中位数和吞吐最小/最大值。输入由固定 seed 生成，cache 开关使用相同 prompt；配置顺序打乱以减少固定运行顺序的影响。OOM 记录为 `oom`，不会当成有效测量。

已有结果不会被覆盖，请为每次实验使用不同文件名。关闭其他占用 GPU 的任务，并记录功耗限制等可能影响结果的运行条件。

### 指标定义

| 字段 | 含义 |
| --- | --- |
| `prefill_first_token_ms` | 完整 prompt 前向、首次 argmax 和写入首 token 的本地耗时 |
| `decode_steps` | `new_tokens - 1`，第一个生成 token 已在 Prefill 阶段产生 |
| `decode_step_ms` | 后续 Decode 总耗时 / Decode 步数；每步处理整个 batch |
| `total_ms` | 上述两阶段的完整本地推理耗时 |
| `output_tokens_per_second` | `batch_size × new_tokens / 完整推理秒数` |
| `peak_allocated_mib` | PyTorch CUDA 峰值 allocated，包含模型和当前实验的活跃张量 |
| `peak_reserved_mib` | PyTorch CUDA 峰值 reserved；包含分配器保留内存 |
| `kv_cache_mib` | 最终持有的 K/V 张量字节数，不是整个进程显存 |

采用同步后的 wall-clock 计时：Prefill 与 Decode 边界同步，不在每个 token 后同步。权重加载、tokenization、输入搬运及预热不计入耗时。Profiler 记录标记在两种模式中保持一致。这里没有网络与请求队列，因此 Prefill 指标不是线上服务的 TTFT。

在生成 N 个 token 后，缓存长度为 `prompt_length + N - 1`：最后生成的 token 尚未再次送入模型。对于普通 dense GQA 缓存，理论字节数为：

```text
2 × 层数 × batch × 缓存长度 × KV 头数 × head_dim × 每元素字节数
```

缓存正确性使用 teacher forcing：两种路径读取完全相同的 token 序列，比较相同位置的 logits。默认 FP32 容差 `atol=rtol=1e-4`，低精度为 `0.05`；报告保留最大绝对误差。低精度容差更宽，不可据此声称逐位相同，也不要为了通过测试随意调宽阈值。

## 性能分析

```bash
python -m inference_lab profile --input-length 512 --new-tokens 32 --output traces/cache-on.json
python -m inference_lab profile --input-length 512 --new-tokens 32 --no-cache --output traces/cache-off.json
```

在 [Perfetto](https://ui.perfetto.dev/) 中打开 trace，查看 `prefill`、`decode` 区域、CPU/GPU 间隙、内存分配和主要算子。分析产生额外开销，trace 中的时间不作为正式 benchmark 数字。

## 本地测试

```bash
python -m pytest -q
```

测试使用随机初始化的微型 Qwen3，不下载模型、不需要 GPU；覆盖 eager/SDPA 的缓存正确性、生成长度和输出一致性、GQA 缓存字节数、结果存储及绘图。CPU 数字只用于检查运行链路，不用于 GPU 性能结论。

如果只有 CPU，可显式用 `--device cpu --dtype float32` 运行各模型命令；请把 benchmark 缩小到单条短输入。完整实验应在 4090 上进行。

## 学习与记录

- [第一阶段 10 天任务](docs/first-10-days.md)
- [实验报告模板](docs/experiment-report-template.md)
- [下一阶段：阅读 nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

`results/`、`traces/` 和模型权重默认不提交。整理出你实际分析过的图表与报告后，可将其放到 `docs/experiments/` 并主动提交。模型遵循其各自许可证。

参考：[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[KV Cache 原理](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)、[PyTorch 测速](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)、[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。
