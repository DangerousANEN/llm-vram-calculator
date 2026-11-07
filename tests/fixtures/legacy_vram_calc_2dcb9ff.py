#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM VRAM & INFERENCE BENCHMARK TOOL (.llm hubs production kit)
Zero-dependency CLI tool to:
 1. Calculate exact VRAM footprint (Weights, KV-Cache per context/batch, CUDA overhead).
 2. Benchmark real TTFT (Time to First Token) and TPS (Tokens/sec) on any OpenAI-compatible API.

Usage:
  python llm_vram_and_speedtest.py calc --params 7 --quant q4_k_m --ctx 8192 --batch 1
  python llm_vram_and_speedtest.py bench --url http://localhost:8000/v1 --model qwen2.5-7b
"""
import argparse
import json
import math
import sys
import time
import urllib.request


# ==========================================
# 1. VRAM CALCULATION ENGINE
# ==========================================
BYTES_PER_PARAM = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "q8_0": 1.06,
    "q6_k": 0.82,
    "q5_k_m": 0.70,
    "q4_k_m": 0.58,
    "q3_k_m": 0.45,
    "nvfp4": 0.52,
    "q2_k": 0.35,
}

KV_BYTES_PER_ELEMENT = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "q8_0": 1.0,
    "q4_0": 0.5,
}


def calculate_vram(params_b, quant, context_len, batch_size, kv_quant="fp16", mla=False):
    """
    params_b: parameter count in billions (e.g. 7, 8, 14, 32, 70)
    quant: weight quantization type
    context_len: sequence length (e.g. 4096, 8192, 32768)
    batch_size: concurrency batch size
    kv_quant: kv cache precision ('fp16', 'fp8', 'q8_0', 'q4_0')
    mla: Multi-Head Latent Attention (DeepSeek style 5-7x compression)
    """
    bpp = BYTES_PER_PARAM.get(quant.lower(), 2.0)
    weights_gb = (params_b * 1e9 * bpp) / (1024 ** 3)

    # Standard architecture heuristics (GQA / MHA)
    layers = int(math.ceil(14 + 7 * math.log2(max(1, params_b))))
    heads = 8  # KV heads in GQA
    head_dim = 128
    
    kv_bpe = KV_BYTES_PER_ELEMENT.get(kv_quant.lower(), 2.0)
    # KV Cache formula: 2 (Key + Value) * layers * kv_heads * head_dim * seq_len * batch * bytes
    if mla:
        # MLA compresses KV into latent vector dc (~512)
        kv_bytes_per_token = layers * 576 * kv_bpe
    else:
        kv_bytes_per_token = 2 * layers * heads * head_dim * kv_bpe
        
    kv_cache_gb = (kv_bytes_per_token * context_len * batch_size) / (1024 ** 3)
    
    cuda_overhead_gb = 0.65
    activations_gb = 0.08 * batch_size * (context_len / 4096)
    total_gb = weights_gb + kv_cache_gb + cuda_overhead_gb + activations_gb
    safe_vram_gb = total_gb / 0.88  # 12% safety headroom

    return {
        "weights_gb": round(weights_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "cuda_overhead_gb": round(cuda_overhead_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "total_gb": round(total_gb, 2),
        "recommended_gpu_vram": math.ceil(safe_vram_gb),
    }


# ==========================================
# 2. STREAMING SPEEDTEST ENGINE
# ==========================================
def benchmark_endpoint(base_url, model, prompt, max_tokens=150, api_key="sk-local"):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    )

    t_start = time.perf_counter()
    ttft = None
    token_times = []
    text_chunks = []

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                t_now = time.perf_counter()
                decoded = line.decode("utf-8", errors="ignore").strip()
                if not decoded.startswith("data:"):
                    continue
                body = decoded[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = t_now - t_start
                        token_times.append(t_now)
                        text_chunks.append(delta)
                except Exception:
                    continue
    except Exception as exc:
        return {"error": str(exc)}

    t_total = time.perf_counter() - t_start
    total_tokens = len(token_times)
    
    if total_tokens > 1 and ttft is not None:
        tps = (total_tokens - 1) / (token_times[-1] - token_times[0])
    elif total_tokens == 1:
        tps = 1.0 / t_total
    else:
        tps = 0.0

    return {
        "ttft_ms": round((ttft or 0) * 1000, 1),
        "tps": round(tps, 2),
        "total_time_s": round(t_total, 2),
        "tokens_generated": total_tokens,
        "sample_output": "".join(text_chunks)[:100] + "...",
    }


def main():
    parser = argparse.ArgumentParser(description="LLM VRAM & Speedtest Benchmark (.llm hubs)")
    subparsers = parser.add_subparsers(dest="cmd")

    # Subparser calc
    p_calc = subparsers.add_parser("calc", help="Calculate VRAM footprint")
    p_calc.add_argument("--params", type=float, default=7.0, help="Model parameters in billions (e.g. 7, 8, 14, 32)")
    p_calc.add_argument("--quant", type=str, default="q4_k_m", choices=list(BYTES_PER_PARAM.keys()), help="Weight quant")
    p_calc.add_argument("--ctx", type=int, default=8192, help="Context sequence length")
    p_calc.add_argument("--batch", type=int, default=1, help="Concurrency batch size")
    p_calc.add_argument("--kv-quant", type=str, default="fp16", choices=list(KV_BYTES_PER_ELEMENT.keys()), help="KV cache quant")
    p_calc.add_argument("--mla", action="store_true", help="Multi-Head Latent Attention (DeepSeek)")

    # Subparser bench
    p_bench = subparsers.add_parser("bench", help="Benchmark latency and TPS")
    p_bench.add_argument("--url", type=str, default="http://localhost:20128/v1", help="OpenAI-compatible base URL")
    p_bench.add_argument("--model", type=str, default="antigravity/gemini-3.8-flash-high", help="Model name")
    p_bench.add_argument("--prompt", type=str, default="Explain how KV-cache works in 3 bullet points.", help="Prompt")
    p_bench.add_argument("--tokens", type=int, default=120, help="Max tokens")

    args = parser.parse_args()

    if args.cmd == "calc":
        res = calculate_vram(args.params, args.quant, args.ctx, args.batch, args.kv_quant, args.mla)
        print("\n" + "="*50)
        print(f" 🧮 VRAM ALLOCATION REPORT: {args.params}B ({args.quant.upper()})")
        print("="*50)
        print(f" • Веса модели (Weights):       {res['weights_gb']:>6.2f} GB")
        print(f" • KV-Кэш ({args.ctx} токенов, b={args.batch}):   {res['kv_cache_gb']:>6.2f} GB (kv={args.kv_quant})")
        print(f" • CUDA Overhead + Активации:  {res['cuda_overhead_gb'] + res['activations_gb']:>6.2f} GB")
        print("-" * 50)
        print(f" • ИТОГО Потребление:          {res['total_gb']:>6.2f} GB")
        print(f" • Рекомендуемый размер GPU:    {res['recommended_gpu_vram']:>6} GB VRAM")
        print("="*50 + "\n")

    elif args.cmd == "bench":
        print(f"\n🚀 Запуск бенчмарка: {args.model} @ {args.url}...")
        res = benchmark_endpoint(args.url, args.model, args.prompt, args.tokens)
        if "error" in res:
            print(f"❌ Ошибка: {res['error']}")
        else:
            print("\n" + "="*50)
            print(f" ⚡ INFERENCE SPEEDTEST REPORT: {args.model}")
            print("="*50)
            print(f" • TTFT (Time to First Token): {res['ttft_ms']:>8.1f} ms")
            print(f" • Скорость генерации (TPS):   {res['tps']:>8.2f} tok/s")
            print(f" • Всего токенов:              {res['tokens_generated']:>8} tokens")
            print(f" • Общее время:                {res['total_time_s']:>8.2f} s")
            print("-" * 50)
            print(f" Вывод: {res['sample_output']}")
            print("="*50 + "\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
