# 从运行结果读回源码

每次只回答一组问题。先运行，再找代码，再用自己的话复述；不用一次读完全部文件。

## 1. 两个手写循环

阅读 `src/inference_lab/kv_cache_exercise.py`，这是个人学习时保留的实现。定位之前修复的三个问题：缓存没有跨轮传递、首 token 索引偏移、输出缓冲区多一格。

自己画出 prompt 长度 3、生成 3 个 token 时的输入长度、写入位置和缓存长度。说明为什么最终序列长度为 6，而缓存长度为 5。

## 2. 正式推理核心

阅读 `core.decode`。先固定 `cache_mode=dynamic`，逐行找出 Prefill 和后续 N-1 次 Decode。再比较 no_cache 与 static。

需要解释：输出 buffer 为什么预分配；`logits_to_keep=1` 省掉什么；`cache_position` 为什么不能每次从 0 开始；StaticCache 的容量如何决定；静态缓存内尚未写入的槽位由谁屏蔽。

## 3. 正确性不是比较两段文字

阅读 `core.diagnose_cache`。两条路径读取相同的固定序列，比较同一位置预测下一个 token 的 logits。FP32 作为更严格的基线；BF16 的逐元素失败不自动等于缓存写错，也不能靠调大容差抹掉。

看每步最大/平均误差、RMSE、超阈值比例、top-1 一致率。解释为什么参考值接近零会导致很大的相对误差，为什么 top-1 一致不是充分证明。

## 4. 实验设计

阅读 `benchmark.run_benchmark`。定位相同输入的 seed、模式交错顺序、预热和每次新建缓存。自己指出哪些工作被计时、哪些未被计时。

从原始 JSON 手算一条吞吐，区分 Prefill、Decode、整体吞吐、峰值 allocated、额外峰值和 reserved。不能用离线平均耗时冒充线上 P95 或并发服务延迟。

## 5. GPU kernel

阅读 `torch_rms_norm` 与 `triton_ops._rms_norm`，一行一行对齐公式。重点是 FP32 reduction、mask、每行一个 program、归一化后先转换回输入 dtype，再乘权重。

Qwen3 不只有两个 residual-stream RMSNorm，还包含 Q/K Norm。模型接入遍历所有 `Qwen3RMSNorm` 模块，因此需要覆盖小 hidden size 和小/大 token 行数。

## 6. 用证据解释收益

同时阅读 kernel 微基准和端到端报告。回答：相较 eager 是否更快；相较 torch.compile 是否还更快；算子变快为什么可能没有明显改善完整推理；静态缓存为什么可能更慢。

找一份 trace 支撑至少一个结论。没有被数据支持的判断写成假设。

## 7. 独立维护一次

任选一个小改动自行实现并补实验：加入一个新输入长度、检查一个边界、改善图表，或研究一个真实瓶颈。保留改动前后的配置与结果，再将这一部分作为你能独立讲解的贡献。
