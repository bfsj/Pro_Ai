# 设计决策

## 可控执行

输出 tensor 一次预分配；no-cache 每轮输入完整前缀，dynamic/static 后续仅输入一个 token。三条路径采用相同采样、最后位置词表投影和位置 tensor，减少缓存策略之外的差异。固定输出数量便于比较吞吐，暂不实现 EOS、padding 或在线请求调度。

## 两种缓存机制

DynamicCache 随每层 K/V 增长，可能涉及拼接和分配；StaticCache 按上限预分配并使用 cache_position 更新槽位，但可能引入容量、mask 和 Attention kernel 选择开销。因此“静态”是实验变量，不是预先宣布的性能胜利。

缓存初始分配计入本轮 Prefill，每次 trial 新建，尚未研究跨请求缓存池复用，也不将结果称为纯稳态服务性能。

## 数值校验

FP32 固定序列比较构成严格门禁。低精度的误差按步报告，保留失败标记；不为了让示例通过自动扩大容差。top-1 相同只是一项补充指标。

## 算子优化

Triton RMSNorm 每行一个 program，采用 FP32 归约，显式保留 Qwen3 先将归一化值转回输入 dtype 的舍入点。它是前向推理实现，不提供训练 backward。模型接入只替换已有 RMSNorm 的 forward，不改变权重或 Decoder 结构。CPU 路径不导入 Triton。

微基准同时测 eager 与 torch.compile，排除编译开销。端到端再测 torch/triton 开关，以识别算子收益是否被 Python、数据搬运或其他算子限制。

## 统计与复现

输入由独立 CPU generator 和 seed 产生；每轮内交错并打乱模式顺序；保存每次测量、OOM、环境、源码提交及 GPU 快照。JSON 在每轮后落盘，异常运行标记为 interrupted_or_failed，不把部分数据冒充完整结果。
