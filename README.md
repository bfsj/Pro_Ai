# Pro_Ai · Single-GPU LLM Inference Lab

面向 AI Infra 实习准备的单卡推理实验项目。围绕同一个 dense Qwen3 模型，完成 **推理循环 → 缓存正确性 → 受控性能实验 → Triton 算子 → 端到端验证 → 可复现报告**。

## 核心成果

| 模块 | 实现与验收 |
| --- | --- |
| 推理引擎 | 自定义 greedy Prefill/Decode；no-cache、DynamicCache、StaticCache 三种路径；显式 cache position |
| 正确性 | 固定 token 序列逐位置比较 logits；动态/静态缓存、mask、输出长度、批量输入；低精度失败原样保留 |
| 性能实验 | 相同输入、固定生成量、预热、交错重复测量；分阶段耗时、吞吐、额外峰值显存、缓存字节数 |
| Triton | 推理用 RMSNorm，FP32 reduction，保持 Qwen3 舍入顺序；对比 eager 和 torch.compile；可接回模型 |
| 实验交付 | 一条 suite 命令输出正确性 JSON、原始测量、CSV、曲线、Markdown 报告和 Profiler trace |
| 工程验证 | 离线微型 Qwen3 CPU 测试、条件 GPU 测试、GitHub Actions；无须下载模型即可测逻辑 |

**范围：** 单 GPU、等长无 padding、固定输出长度、忽略 EOS。尚未实现生产服务、在线调度、多卡训练或多租户隔离。StaticCache / DynamicCache 来自 Transformers；本项目的工作在于推理执行、策略集成、实验设计、算子和验证。

真实结果已收录到 [实验目录](docs/experiments/README.md)。在 RTX 4090 / Qwen3-0.6B 上：

- FP32、B=4、输入 1024 / 输出 32：动态缓存吞吐 **21.87 → 149.32 tokens/s（6.83×）**，基线为完整前缀重复计算。
- FP32、B=1、输入 512 / 输出 32：接入 Triton RMSNorm 后，独立十次复测吞吐 **49.28 → 53.43 tokens/s（+8.4%）**。
- 保存 180 次正式推理测量、40 次复测、36 组算子结果；服务器 **52 项测试通过**。BF16 指定逐元素校验失败，诊断如实保留。

数字只适用于上述精度、模型与工作负载。算子更快不等于完整推理同比例加速；静态缓存也不保证更快。[查看完整证据与局限](docs/experiments/20261021-4090/README.md)。

## 快速开始

推荐 Linux、Python 3.11、CUDA 版 PyTorch；CPU 也能完成 smoke 和测试。

```bash
git clone https://github.com/bfsj/Pro_Ai.git
cd Pro_Ai
python3 -m venv .venv
source .venv/bin/activate
# 先按 https://pytorch.org/get-started/locally/ 安装适配驱动的 PyTorch
python -m pip install -e ".[dev]"
python -m inference_lab env
python -m pytest -q
```

服务器已有环境可直接使用：

```bash
source /home/optics/miniforge3/etc/profile.d/conda.sh
conda activate JD_Pro
cd /data/optics/JD/SN/Pro_Ai
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export PRO_AI_MODEL=/data/optics/JD/SN/models/Qwen3-0.6B
# 用 nvidia-smi 选空闲卡，将编号替换为实际可用设备
export CUDA_VISIBLE_DEVICES=1
```

模型解析顺序：`--model` > `PRO_AI_MODEL` > 仓库同级 `models/Qwen3-0.6B` > Hugging Face 模型 ID。迁移项目后执行一次 `python -m pip install --no-deps --no-build-isolation -e .` 更新 editable 路径。权重不提交 Git。

## 一键完成实验

先执行小规模实验，默认 FP32：

```bash
python -m inference_lab suite --config configs/smoke.json --output-dir results/smoke-v1
```

正式实验包含两种 batch、三种长度、三种缓存、两种 Norm backend。需要 Linux/CUDA/Triton：

```bash
python -m inference_lab suite --config configs/resume.json --output-dir results/resume-v1
```

suite 首先运行 **FP32 逐位置一致性门禁**，通过才继续。随后执行受控 benchmark，并输出：

```text
results/resume-v1/
├── requested-config.json
├── correctness-fp32.json
├── benchmark.json
├── traces/                  # 诊断 trace 和算子耗时摘要
└── report/
    ├── report.md
    ├── summary.csv
    ├── measurements.json
    └── performance.png
```

更大的扫描配置在 `configs/full.json`，耗时更长。CPU smoke 使用 `--device cpu --dtype float32`；CPU 配置应保持 `norm_backends: ["torch"]`。

`suite --dtype bfloat16` 会额外记录低精度诊断，失败不会被改写成通过。性能通过不代表低精度质量通过。

## 单独运行各模块

```bash
# 模型结构、缓存形状和理论大小
python -m inference_lab inspect --cache-mode static

# 三种生成路径
python -m inference_lab generate --cache-mode no_cache --new-tokens 32
python -m inference_lab generate --cache-mode dynamic --new-tokens 32
python -m inference_lab generate --cache-mode static --new-tokens 32

# 严格 FP32 正确性；失败时输出 JSON 并返回非零退出码
python -m inference_lab check --cache-mode static --input-length 128 --steps 8 --output results/check-static.json

# 收集 BF16 数值差异；diagnostic 保留失败结果但允许命令完成
python -m inference_lab check --dtype bfloat16 --batch-size 4 --input-length 512 --diagnostic --output results/bf16-diagnostic.json

# 隔离 Prefill：记录 SDPA 的 mask/GQA/布局，再进行受控干预
python scripts/attention_layout_probe.py --output results/attention-layout.json

# 只测缓存策略
python -m inference_lab benchmark --batch-sizes 1 4 --input-lengths 128 512 --new-tokens 32 --output results/cache.json

# 单独导出性能分析文件；不把 profiler 时间当正式测速
python -m inference_lab profile --cache-mode dynamic --input-length 512 --output traces/dynamic.json

# 测 RMSNorm，排除编译，比较 eager / compile / Triton
python -m inference_lab kernel-benchmark --rows 1 8 128 512 --widths 128 1024 2048 --output results/norm.json
python -m inference_lab report --kind kernel --input results/norm.json --output-dir results/norm-report

# 从已有原始数据生成报告
python -m inference_lab report --input results/cache.json --output-dir results/cache-report
```

所有输出使用新文件/目录，避免替换已有实验。

## 指标与公平性

- 所有路径使用相同 token IDs、精度和 Attention backend；只将最后一个位置投影到词表，统一预分配输出 buffer。
- 每次运行使用新缓存；静态缓存的初始惰性分配计入 Prefill。静态缓存不保证更快，容量和 mask 本身也有开销。
- 首次 Prefill 产生第 1 个输出，后续 Decode 共 N-1 步。最终缓存长度是 `prompt_length + N - 1`。
- 耗时为同步后的 wall-clock，包含 Python、采样和模型执行；不在每个 token 后同步。加载、分词、输入搬运、预热与编译不计入。
- `output_tokens_per_second` = 实际 batch × 固定输出数量 / 完整推理时间。
- `decode_step_ms` 是本地 Decode 平均每步耗时，不是服务端 P95 ITL；Prefill 指标不等于线上 TTFT。
- `peak_allocated_mib` 包含模型；`extra_peak_allocated_mib` 扣除运行前活跃分配；reserved 与进程显存不是同一指标。
- 普通 GQA 缓存理论字节数：`2 × 层数 × batch × 缓存容量 × KV头数 × head_dim × 每元素字节数`。
- 重复测量保存原始值、中位数和 CV。GPU 快照保留在报告中；共享机器仍可能产生干扰。

## 从哪里读代码

```text
src/inference_lab/
├── kv_cache_exercise.py   # 保留的个人学习练习
├── core.py                # 三种执行路径、位置/缓存管理、正确性诊断
├── benchmark.py           # 相同工作负载、交错重复、统计和环境记录
├── triton_ops.py          # RMSNorm GPU kernel
├── kernels.py             # 模型接入、算子对照实验
├── experiments.py         # 一键实验与 Profiler
├── artifacts.py           # CSV、图表、Markdown 报告
├── report.py              # 性能曲线
└── cli.py                 # 参数和模型加载
```

`scripts/attention_layout_probe.py` 是独立诊断程序，临时截获 SDPA，仅用于单独进程中的 Prefill 对照，不是正式推理实现或低精度修复。

建议按 [阅读路线](docs/reading-guide.md) 一点点阅读，再按 [面试讲解与简历边界](docs/interview-guide.md) 练习复述。

## 设计资料与参考

- [Transformers 缓存策略](https://huggingface.co/docs/transformers/v4.57.1/kv_cache)
- [PyTorch 数值精度](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Triton LayerNorm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：学习参考；本项目不将其已有功能列为自行实现。

项目代码在 AI 辅助下整理。简历应描述你实际理解、验证和维护的工作；未独立完成的部分不要写成独立从零实现。
