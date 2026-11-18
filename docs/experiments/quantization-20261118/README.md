# Qwen3-0.6B · BF16 与 INT4 权重量化

后续在 RTX 4090 和 CPU 上完成了分组大小与裁剪对照。[调参和留出集验证](../quantization-tuning-20261119/README.md)

量化对象为 28 个 Transformer 块内的 Q、K、V、输出投影和 MLP 的 gate、up、down 投影，共 196 个线性层。每 128 个输入权重共享一对 BF16 缩放与偏移，4 bit 编码交给 PyTorch 的 packed matmul。共享词嵌入与输出头、激活和动态 KV Cache 保持 BF16。

每个模式在独立进程中加载同一检查点。每种负载预热一次，固定输入重复三次，各列取中位数。每序列贪心生成 32 token，首 token 时间含 prefill，后续延迟为余下 31 步的平均。共享 CPU 主机固定四线程，两种模式分时执行。每个负载的三次计时重复使用同一输入。模型存储统计参数和缓冲区的字节数。CUDA 峰值单列生成时 PyTorch 已分配的显存，计时排除加载与打包。

## Intel Xeon Platinum 8352V

模型张量存储由 1136.88 降至 520.00 MiB，减少 54.3%。12 段手写中英文短文共计 369 个受评 token，困惑度从 81.016 变为 150.508。每段独立分词，以真实前文预测后续 token，按 token 汇总交叉熵再取指数。该语料用于小规模质量回归。

| Batch | 输入长度 | 模式 | 首 token ms | 后续 ms/token | 整批 tokens/s | 生成峰值显存 MiB |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 128 | bf16 | 1624.77 | 72.73 | 8.24 | — |
| 1 | 128 | int4 | 1000.18 | 45.00 | 13.36 | — |
| 1 | 256 | bf16 | 2672.25 | 75.90 | 6.41 | — |
| 1 | 256 | int4 | 1527.82 | 48.57 | 10.57 | — |
| 1 | 512 | bf16 | 4776.83 | 83.57 | 4.30 | — |
| 1 | 512 | int4 | 2588.17 | 56.21 | 7.27 | — |
| 1 | 1024 | bf16 | 9019.85 | 103.29 | 2.62 | — |
| 1 | 1024 | int4 | 4668.87 | 72.64 | 4.62 | — |
| 4 | 128 | bf16 | 5845.02 | 181.08 | 11.13 | — |
| 4 | 128 | int4 | 3566.83 | 88.27 | 20.27 | — |
| 4 | 256 | bf16 | 9937.07 | 198.87 | 7.94 | — |
| 4 | 256 | int4 | 5412.54 | 102.96 | 14.88 | — |
| 4 | 512 | bf16 | 18847.37 | 242.73 | 4.87 | — |
| 4 | 512 | int4 | 9863.25 | 145.38 | 8.91 | — |
| 4 | 1024 | bf16 | 36279.36 | 312.66 | 2.78 | — |
| 4 | 1024 | int4 | 18800.23 | 216.10 | 5.03 | — |

[BF16 原始记录](bf16-cpu.json) · [INT4 原始记录](int4-cpu.json)

GPU 检查期间三张 RTX 4090 均有其他任务，未取得完整的 BF16 / INT4 配对计时。表中 CPU 数字仅描述本次硬件和线程设置。

## 复现

在仓库根目录安装项目及其依赖，使用 PyTorch 2.11。将 MODEL 指向同一份 Qwen3-0.6B 检查点。切换到空闲 GPU 时将 device 改为 cuda。每个输出使用新文件路径。

```sh
python scripts/quantization_bench.py --model "$MODEL" --mode bf16 --device cpu --output bf16.json
python scripts/quantization_bench.py --model "$MODEL" --mode int4 --device cpu --output int4.json
```

加载 BF16 的 Qwen3 模型后调用 `from inference_lab.int4 import quantize_decoder`、`quantize_decoder(model)` 即可通过已有 generate 或 decode 使用打包权重。权重量化在进程内进行。

[固定评测文本](../../../scripts/quantization_text.txt) · [权重指纹](../20261021-4090/model-sha256.txt) · [PyTorch AO 的 INT4 tile-packed 实现参考](https://github.com/pytorch/ao/blob/main/torchao/quantization/quantize_/workflows/int4/int4_tile_packed_to_4d_tensor.py)
