# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce five KVPress-style cache compression methods on Pythia-70M.

The current kvpress package targets newer Transformers cache APIs and tested model
families such as Llama, Mistral, Phi, Qwen, and Gemma. Pythia-70M uses the older
GPT-NeoX tuple cache in the environment used for this assignment, so this script
implements a small, self-contained reproduction of five algorithms from kvpress:

- RandomPress
- KnormPress
- StreamingLLMPress
- ObservedAttentionPress
- SnapKVPress

It evaluates conditional perplexity and decode speed after compressing a prefix
KV cache. The first continuation token is used as the query token; losses are
computed for subsequent continuation tokens.

The comparison baseline is method ``none``: the same Pythia-70M model, data,
prefix length, and token-by-token evaluation path, but without KV cache
compression. ``RandomCompression`` is kept as an additional random-compression
sanity baseline and is counted among the five reproduced KVPress algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE = ROOT / "experiments" / "kvpress_repro" / "data_cache"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache" / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf_cache" / "datasets"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]
CacheLike = LegacyCache | object


@dataclass
class MethodResult:
    dataset: str
    method: str
    compression_ratio: float
    samples: int
    prefix_len: int
    eval_tokens: int
    loss_tokens: int
    nll: float
    ppl: float
    prefill_s: float
    compress_s: float
    decode_s: float
    total_s: float
    tokens_per_s: float
    cache_before_mib: float
    cache_after_mib: float
    cache_reduction: float


class CacheCompressor:
    name = "base"
    needs_attentions = False

    def __init__(self, compression_ratio: float, seed: int = 0):
        if not 0 <= compression_ratio < 1:
            raise ValueError("compression_ratio must be in [0, 1)")
        self.compression_ratio = compression_ratio
        self.seed = seed

    def score(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compress_layer(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        seq_len = keys.shape[2]
        n_kept = max(1, int(seq_len * (1 - self.compression_ratio)))
        scores = self.score(layer_idx, keys, values, attentions)
        indices = scores.topk(n_kept, dim=-1).indices
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        return keys.gather(2, gather_indices).contiguous(), values.gather(2, gather_indices).contiguous()

    def compress(self, cache: CacheLike, attentions: tuple[torch.Tensor, ...] | None) -> CacheLike:
        if is_dynamic_cache(cache):
            for layer_idx, layer in enumerate(cache.layers):
                layer_attn = None if attentions is None else attentions[layer_idx]
                keys, values = self.compress_layer(layer_idx, layer.keys, layer.values, layer_attn)
                layer.keys = keys
                layer.values = values
            return cache

        compressed_layers = []
        for layer_idx, (keys, values) in enumerate(cache):  # type: ignore[union-attr]
            layer_attn = None if attentions is None else attentions[layer_idx]
            compressed_layers.append(self.compress_layer(layer_idx, keys, values, layer_attn))
        return tuple(compressed_layers)


class NoCompression(CacheCompressor):
    name = "none"

    def score(self, layer_idx, keys, values, attentions):
        return torch.ones_like(keys[..., 0])


class RandomCompression(CacheCompressor):
    name = "random"

    def score(self, layer_idx, keys, values, attentions):
        try:
            generator = torch.Generator(device=keys.device)
        except TypeError:
            generator = torch.Generator()
        generator.manual_seed(self.seed + layer_idx)
        return torch.rand(keys.shape[:-1], generator=generator, device=keys.device, dtype=keys.dtype)


class KNormCompression(CacheCompressor):
    name = "knorm"

    def score(self, layer_idx, keys, values, attentions):
        # kvpress.KnormPress uses the inverse key norm: lower-norm keys receive higher scores.
        return -keys.float().norm(dim=-1)


class StreamingLLMCompression(CacheCompressor):
    name = "streaming_llm"

    def __init__(self, compression_ratio: float, seed: int = 0, n_sink: int = 4):
        super().__init__(compression_ratio=compression_ratio, seed=seed)
        self.n_sink = n_sink

    def score(self, layer_idx, keys, values, attentions):
        seq_len = keys.shape[2]
        n_kept = max(1, int(seq_len * (1 - self.compression_ratio)))
        n_pruned = seq_len - n_kept
        scores = torch.ones_like(keys[..., 0])
        prune_start = min(self.n_sink, seq_len)
        prune_end = min(seq_len, prune_start + n_pruned)
        scores[:, :, prune_start:prune_end] = 0
        return scores


class ObservedAttentionCompression(CacheCompressor):
    name = "observed_attention"
    needs_attentions = True

    def score(self, layer_idx, keys, values, attentions):
        if attentions is None:
            raise ValueError("ObservedAttentionCompression requires output_attentions=True.")
        seq_len = keys.shape[2]
        scores = attentions.float().sum(dim=2)
        denominator = torch.arange(seq_len, 0, -1, device=attentions.device, dtype=scores.dtype)
        return scores / denominator


class SnapKVCompression(CacheCompressor):
    name = "snapkv"
    needs_attentions = True

    def __init__(
        self,
        compression_ratio: float,
        seed: int = 0,
        window_size: int = 32,
        kernel_size: int = 5,
    ):
        super().__init__(compression_ratio=compression_ratio, seed=seed)
        self.window_size = window_size
        self.kernel_size = kernel_size

    def score(self, layer_idx, keys, values, attentions):
        if attentions is None:
            raise ValueError("SnapKVCompression requires output_attentions=True.")
        seq_len = keys.shape[2]
        window = min(self.window_size, max(1, seq_len - 1))
        historical_len = seq_len - window
        if historical_len <= 0:
            return torch.ones_like(keys[..., 0])

        scores = attentions.float()[..., -window:, :historical_len].mean(dim=-2)
        scores = F.avg_pool1d(
            scores,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        scores = scores[..., :historical_len]
        recent_score = scores.max().item() if scores.numel() else 1.0
        return F.pad(scores, (0, window), value=recent_score)


METHODS: dict[str, type[CacheCompressor]] = {
    "none": NoCompression,
    "random": RandomCompression,
    "knorm": KNormCompression,
    "streaming_llm": StreamingLLMCompression,
    "observed_attention": ObservedAttentionCompression,
    "snapkv": SnapKVCompression,
}


def is_dynamic_cache(cache: CacheLike) -> bool:
    return hasattr(cache, "layers")


def iter_cache_tensors(cache: CacheLike):
    if is_dynamic_cache(cache):
        for layer in cache.layers:
            yield layer.keys, layer.values
    else:
        yield from cache  # type: ignore[misc]


def cache_nbytes(cache: CacheLike) -> int:
    return sum(
        keys.numel() * keys.element_size() + values.numel() * values.element_size()
        for keys, values in iter_cache_tensors(cache)
    )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_step(model, step_input: torch.Tensor, current_cache: CacheLike, token_pos: int, device: torch.device):
    position_ids = torch.tensor([[token_pos]], device=device, dtype=torch.long)
    cache_position = torch.tensor([token_pos], device=device, dtype=torch.long)
    kwargs = {
        "input_ids": step_input,
        "past_key_values": current_cache,
        "position_ids": position_ids,
        "use_cache": True,
    }
    if is_dynamic_cache(current_cache):
        kwargs["cache_position"] = cache_position
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def patch_gpt_neox_rotary_cache(model) -> None:
    """Return enough RoPE cache for absolute position_ids after KV pruning.

    GPT-NeoX in Transformers 4.32 builds rotary cos/sin tensors using the current
    cache length. After KV compression, the cache length is shorter than the
    original absolute token position. Returning the full cached RoPE table keeps
    absolute position_ids valid for the compressed cache experiment.
    """

    def forward_with_full_cache(self, x, seq_len=None):
        requested = max(int(seq_len or 0), int(self.max_position_embeddings))
        if requested > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=requested, device=x.device)
        return self.cos_cached[:requested, ...].to(x.device), self.sin_cached[:requested, ...].to(x.device)

    import types

    for layer in model.gpt_neox.layers:
        rotary_emb = getattr(layer.attention, "rotary_emb", None)
        if rotary_emb is not None:
            rotary_emb.forward = types.MethodType(forward_with_full_cache, rotary_emb)


def iter_dataset_texts(dataset: str, split: str) -> Iterable[str]:
    dataset = dataset.lower()
    local_text = DATA_CACHE / f"{dataset}_{split}.txt"
    if local_text.exists():
        text = local_text.read_text(encoding="utf-8")
        if text.strip():
            yield text
            return

    if dataset in {"wikitext", "wiki"}:
        stream = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
    elif dataset in {"pg19", "pg-19"}:
        stream = load_dataset("pg19", split=split, streaming=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for row in stream:
        text = row.get("text", "")
        if text and text.strip():
            yield text


def build_token_samples(
    tokenizer,
    dataset: str,
    split: str,
    prefix_len: int,
    eval_tokens: int,
    max_samples: int,
    max_chars: int,
    refresh_cache: bool,
) -> list[torch.Tensor]:
    needed = prefix_len + eval_tokens + 1
    dataset_key = dataset.lower().replace("-", "_")
    cache_path = DATA_CACHE / f"{dataset_key}_{split}_p{prefix_len}_e{eval_tokens}_n{max_samples}.pt"
    if cache_path.exists() and not refresh_cache:
        cached = torch.load(cache_path, map_location="cpu")
        token_tensor = cached["samples"] if isinstance(cached, dict) else cached
        if token_tensor.ndim != 2 or token_tensor.shape[1] < needed:
            raise RuntimeError(f"Invalid token cache shape in {cache_path}: {tuple(token_tensor.shape)}")
        return [row[:needed].clone() for row in token_tensor[:max_samples]]

    texts = []
    total_chars = 0
    for text in iter_dataset_texts(dataset, split):
        texts.append(text)
        total_chars += len(text)
        if total_chars >= max_chars:
            break

    if not texts:
        raise RuntimeError(f"No text loaded from dataset={dataset!r} split={split!r}.")

    token_ids = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False).input_ids[0]
    if token_ids.numel() < needed:
        raise RuntimeError(
            f"Dataset {dataset} produced only {token_ids.numel()} tokens, but {needed} are required. "
            "Increase --max-chars or reduce --prefix-len/--eval-tokens."
        )

    samples = []
    stride = needed
    for start in range(0, token_ids.numel() - needed + 1, stride):
        samples.append(token_ids[start : start + needed].clone())
        if len(samples) >= max_samples:
            break
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": dataset,
            "split": split,
            "prefix_len": prefix_len,
            "eval_tokens": eval_tokens,
            "max_samples": max_samples,
            "samples": torch.stack(samples),
        },
        cache_path,
    )
    return samples


@torch.no_grad()
def evaluate_method_on_samples(
    model,
    samples: list[torch.Tensor],
    dataset: str,
    method_name: str,
    compression_ratio: float,
    prefix_len: int,
    eval_tokens: int,
    device: torch.device,
    seed: int,
) -> MethodResult:
    compressor = METHODS[method_name](compression_ratio=0.0 if method_name == "none" else compression_ratio, seed=seed)

    total_nll = 0.0
    total_loss_tokens = 0
    prefill_s = 0.0
    compress_s = 0.0
    decode_s = 0.0
    cache_before = 0
    cache_after = 0

    for sample in samples:
        input_ids = sample.to(device).unsqueeze(0)
        prefix_ids = input_ids[:, :prefix_len]

        sync_device(device)
        start = time.perf_counter()
        outputs = model(
            input_ids=prefix_ids,
            use_cache=True,
            output_attentions=compressor.needs_attentions,
        )
        sync_device(device)
        prefill_s += time.perf_counter() - start

        cache = outputs.past_key_values
        attentions = outputs.attentions if compressor.needs_attentions else None
        cache_before += cache_nbytes(cache)

        sync_device(device)
        start = time.perf_counter()
        compressed_cache = compressor.compress(cache, attentions)
        sync_device(device)
        compress_s += time.perf_counter() - start
        cache_after += cache_nbytes(compressed_cache)

        current_cache = compressed_cache
        for offset in range(eval_tokens):
            token_pos = prefix_len + offset
            target_pos = token_pos + 1
            if target_pos >= input_ids.shape[1]:
                break

            step_input = input_ids[:, token_pos : token_pos + 1]

            sync_device(device)
            start = time.perf_counter()
            step_outputs = model_step(model, step_input, current_cache, token_pos, device)
            sync_device(device)
            decode_s += time.perf_counter() - start

            logits = step_outputs.logits[:, -1, :].float()
            target = input_ids[:, target_pos]
            total_nll += F.cross_entropy(logits, target, reduction="sum").item()
            total_loss_tokens += int(target.numel())
            current_cache = step_outputs.past_key_values

    ppl = math.exp(total_nll / max(1, total_loss_tokens))
    total_s = prefill_s + compress_s + decode_s
    cache_before_mib = cache_before / max(1, len(samples)) / 1024**2
    cache_after_mib = cache_after / max(1, len(samples)) / 1024**2
    cache_reduction = 1 - cache_after_mib / cache_before_mib if cache_before_mib else 0.0

    return MethodResult(
        dataset=dataset,
        method=method_name,
        compression_ratio=0.0 if method_name == "none" else compression_ratio,
        samples=len(samples),
        prefix_len=prefix_len,
        eval_tokens=eval_tokens,
        loss_tokens=total_loss_tokens,
        nll=total_nll / max(1, total_loss_tokens),
        ppl=ppl,
        prefill_s=prefill_s,
        compress_s=compress_s,
        decode_s=decode_s,
        total_s=total_s,
        tokens_per_s=total_loss_tokens / decode_s if decode_s > 0 else 0.0,
        cache_before_mib=cache_before_mib,
        cache_after_mib=cache_after_mib,
        cache_reduction=cache_reduction,
    )


def write_results(results: list[MethodResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (output_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models" / "pythia-70m"))
    parser.add_argument("--datasets", default="wikitext,pg19")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--methods",
        default="none,random,knorm,streaming_llm,observed_attention,snapkv",
        help="Comma-separated methods. Include 'none' for the uncompressed baseline.",
    )
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument(
        "--refresh-sample-cache",
        action="store_true",
        help="Rebuild token samples instead of reading experiments/kvpress_repro/data_cache/*.pt.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Attention implementation passed to from_pretrained when supported. 'eager' is required for attention-based methods.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model load dtype. Pythia-70M can produce NaNs in float16 in some environments, so float32 is the default.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "kvpress_repro" / "results"))
    return parser.parse_args()


def load_model(model_path: str, attn_implementation: str, dtype: str):
    kwargs = {"local_files_only": Path(model_path).exists()}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if dtype != "auto":
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(f"Unknown torch dtype: {dtype}")
        kwargs["torch_dtype"] = torch_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=Path(args.model).exists())
    model = load_model(args.model, args.attn_implementation, args.dtype)
    model.to(device)
    model.eval()
    patch_gpt_neox_rotary_cache(model)

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")

    results: list[MethodResult] = []
    for dataset in datasets:
        print(f"Loading {dataset} samples...")
        samples = build_token_samples(
            tokenizer=tokenizer,
            dataset=dataset,
            split=args.split,
            prefix_len=args.prefix_len,
            eval_tokens=args.eval_tokens,
            max_samples=args.max_samples,
            max_chars=args.max_chars,
            refresh_cache=args.refresh_sample_cache,
        )
        print(f"Loaded {len(samples)} samples for {dataset}.")

        for method in methods:
            print(f"Evaluating dataset={dataset} method={method}...")
            result = evaluate_method_on_samples(
                model=model,
                samples=samples,
                dataset=dataset,
                method_name=method,
                compression_ratio=args.compression_ratio,
                prefix_len=args.prefix_len,
                eval_tokens=args.eval_tokens,
                device=device,
                seed=args.seed,
            )
            results.append(result)
            print(
                f"{dataset:8s} {method:18s} ppl={result.ppl:8.3f} "
                f"tok/s={result.tokens_per_s:7.2f} cache_reduction={result.cache_reduction:.1%}"
            )

    write_results(results, Path(args.output_dir))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
