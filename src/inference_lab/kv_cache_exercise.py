import torch

def generate_no_cache(model, input_ids, max_new_tokens):
    """Generate tokens without using KV cache."""
    model.eval()
    new_tokens = max_new_tokens + 1  # include the first token after the prompt
    all_ids = input_ids
    for _ in range(1, new_tokens):

        outputs = model(input_ids=all_ids, use_cache=False)
        next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids = torch.cat([all_ids, next_id], dim=1)

    return all_ids


def generate_with_cache(model, input_ids, max_new_tokens):
    """Generate tokens using KV cache."""
    model.eval()
    device = input_ids.device
    batch, prompt_length = input_ids.shape
    tokens = torch.empty((batch, prompt_length + max_new_tokens), dtype=torch.long, device=device)
    tokens[:, :prompt_length] = input_ids
    cache = None  # Initialize cache to None for the first step
    for step in range(max_new_tokens):

        position = prompt_length + step
        current = tokens[:, position - 1:position] if step > 0 else tokens[:, :position]
        outputs = model(
            input_ids=current,
            use_cache=True,
            past_key_values=cache,
        )
        cache = outputs.past_key_values

        tokens[:, position] = outputs.logits[:, -1, :].argmax(dim=-1)

    return tokens

import time


@torch.inference_mode()
def measure_once(fn, model, input_ids, max_new_tokens):
    device = input_ids.device

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    output = fn(model, input_ids, max_new_tokens)

    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    return seconds, peak_mib

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/data/optics/JD/SN/models/Qwen3-0.6B"
    device = "cpu"
    torch.set_num_threads(2)

    # 1. 加载本地 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval()

    # 2. 把对话整理成模型需要的格式，关闭 thinking
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释什么是 KV Cache。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # 3. 文本转成形状为 [1, prompt_length] 的 token IDs
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)
    prompt_length = input_ids.shape[1]

    # 4. 调用你自己写的生成函数
    with torch.inference_mode():
        all_ids = generate_no_cache(
            model,
            input_ids,
            max_new_tokens=8,
        )

    with torch.inference_mode():
        all_ids_cache = generate_with_cache(
            model,
            input_ids,
            max_new_tokens=8,
        )
    # 5. 检查长度，只解码新增的 token
    assert all_ids.shape == (1, prompt_length + 8)
    new_ids = all_ids[0, prompt_length:]

    print("输入形状：", tuple(input_ids.shape))
    print("输出形状：", tuple(all_ids.shape))
    print("新增 token IDs：", new_ids.tolist())
    print("生成文本：", tokenizer.decode(
        new_ids.tolist(),
        skip_special_tokens=True,
    ))
    print("Output with cache:")
    print("输入形状：", tuple(input_ids.shape))
    print("输出形状：", tuple(all_ids_cache.shape))
    print("新增 token ID:", all_ids_cache[0, prompt_length:].tolist())
    print("生成文本：", tokenizer.decode(
        all_ids_cache[0, prompt_length:].tolist(),
        skip_special_tokens=True,
    ))
if __name__ == "__main__":
    main()
