#!/usr/bin/env python3
import argparse
import os
import sys
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import MNN
from transformers import AutoTokenizer

# =============================================================================
# Utilities
# =============================================================================

def now_s() -> str:
    return time.strftime("%H:%M:%S")

def apply_chat_template(
    tokenizer: AutoTokenizer,
    user_text: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": user_text})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"[{now_s()}] warn: chat template failed -> fallback ({e})")
        return "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)

def softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    np.exp(x, out=x)
    s = x.sum(axis=-1, keepdims=True)
    # Avoid div-by-zero
    s = np.where(s <= 0.0, 1.0, s)
    return x / s

def sample_token(
    logits: np.ndarray,
    temp: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    """Temperature, then top-k, then top-p (nucleus). Greedy when temp<=0."""
    if not np.all(np.isfinite(logits)):
        # Fallback to greedy on bad logits
        return int(np.nanargmax(logits))

    if temp <= 0.0:
        return int(np.argmax(logits))

    probs = softmax_stable(logits.astype(np.float64) / float(temp))

    vocab = probs.shape[0]
    if top_k > 0 and top_k < vocab:
        # Keep only top_k highest probabilities
        kth = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[kth] = True
        probs = np.where(mask, probs, 0.0)

    if 0.0 < top_p < 1.0:
        order = np.argsort(-probs)
        sorted_p = probs[order]
        cumsum = np.cumsum(sorted_p)
        # Smallest set with cumprob >= top_p
        cutoff = np.searchsorted(cumsum, top_p, side="left") + 1
        keep_idx = order[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep_idx] = True
        probs = np.where(mask, probs, 0.0)

    s = probs.sum()
    if not np.isfinite(s) or s <= 1e-12:
        return int(np.argmax(logits))

    probs /= s
    return int(rng.choice(probs.shape[0], p=probs))

class Spinner(Thread):
    """Lightweight TTY spinner; auto-clears its line."""
    def __init__(self, label: str = "Working"):
        super().__init__(daemon=True)
        self._stop = Event()
        self._label = label
        self._frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._width = len(label) + 2 + 1

    def run(self):
        i = 0
        while not self._stop.is_set():
            frame = self._frames[i % len(self._frames)]
            msg = f"\r{self._label} {frame}"
            self._width = max(self._width, len(msg))
            print(msg.ljust(self._width), end="", flush=True)
            i += 1
            time.sleep(0.08)

    def stop(self):
        self._stop.set()
        # Clear line
        print("\r" + " " * self._width + "\r", end="", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

# =============================================================================
# MNN LLM Wrapper (KV Cache-enabled)
# =============================================================================

class MNNLLM:
    """KV-cache aware MNN runner with light autodiscovery of IO tensors."""
    _BACKEND_MAP: Dict[str, MNN.ForwardType] = {
        "CPU": MNN.ForwardType.MNN_FORWARD_CPU,
        "OPENCL": MNN.ForwardType.MNN_FORWARD_OPENCL,
        "VULKAN": MNN.ForwardType.MNN_FORWARD_VULKAN,
        "METAL": MNN.ForwardType.MNN_FORWARD_METAL,
    }

    def __init__(self, model_path: str, backend: str = "CPU", threads: int = 4):
        self.interpreter = MNN.Interpreter(model_path)

        sess_cfg = MNN.SessionConfig()
        sess_cfg.numThread = max(0, int(threads))
        sess_cfg.forwardType = self._BACKEND_MAP.get(backend.upper(), MNN.ForwardType.MNN_FORWARD_CPU)

        backend_cfg = MNN.BackendConfig()
        backend_cfg.precision = MNN.BackendConfig.Precision_Normal

        self.session = self.interpreter.createSession(sess_cfg, backend_cfg)

        # Discover IO names
        self.in_names = list(self.interpreter.getSessionInputNames(self.session))
        self.out_names = list(self.interpreter.getSessionOutputNames(self.session))

        # Heuristics for primary io
        # Input: pick first int tensor-like name; common: "input_ids"
        self.input_name = next((n for n in self.in_names if "input" in n.lower()), self.in_names[0])
        # Output: pick logits-like name
        self.output_name = next((n for n in self.out_names if "logits" in n.lower()), self.out_names[0])

        self._input_tensor = self.interpreter.getSessionInput(self.session, self.input_name)
        self._output_tensor = self.interpreter.getSessionOutput(self.session, self.output_name)

        # KV cache in/out discovery (robust to different exporters)
        def match_any(name: str, keys: Iterable[str]) -> bool:
            s = name.lower()
            return any(k in s for k in keys)

        kv_in_keys = ("past", "cache_in", "kvcache_in", "kv_in")
        kv_out_keys = ("present", "cache_out", "kvcache_out", "kv_out")

        self.kv_input_names = [n for n in self.in_names if match_any(n, kv_in_keys)]
        self.kv_output_names = [n for n in self.out_names if match_any(n, kv_out_keys)]

        self.kv_input_tensors = {n: self.interpreter.getSessionInput(self.session, n) for n in self.kv_input_names}
        self.kv_output_tensors = {n: self.interpreter.getSessionOutput(self.session, n) for n in self.kv_output_names}

        self._host_output_buffer: Optional[MNN.Tensor] = None

    @staticmethod
    def _np_to_host_tensor(arr: np.ndarray, dtype) -> MNN.Tensor:
        return MNN.Tensor(arr.shape, dtype, arr, MNN.Tensor_DimensionType_Caffe)

    def _prepare_inputs(
        self,
        input_ids: np.ndarray,
        past_key_values: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        # Input ids must be int32 for MNN
        if input_ids.dtype != np.int32:
            input_ids = input_ids.astype(np.int32, copy=False)

        self.interpreter.resizeTensor(self._input_tensor, input_ids.shape)

        if past_key_values:
            for name, data in past_key_values.items():
                ten = self.kv_input_tensors.get(name)
                if ten is not None:
                    self.interpreter.resizeTensor(ten, data.shape)

        self.interpreter.resizeSession(self.session)

        # Copy primary input
        host_input = self._np_to_host_tensor(input_ids, MNN.Halide_Type_Int)
        self._input_tensor.copyFrom(host_input)

        # Copy KV inputs
        if past_key_values:
            for name, data in past_key_values.items():
                ten = self.kv_input_tensors.get(name)
                if ten is None:
                    continue
                if data.dtype != np.float32:
                    data = data.astype(np.float32, copy=False)
                host_kv = self._np_to_host_tensor(data, MNN.Halide_Type_Float)
                ten.copyFrom(host_kv)

    def _collect_outputs(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # Logits
        out_dims = self._output_tensor.getShape()
        if self._host_output_buffer is None or self._host_output_buffer.getShape() != out_dims:
            self._host_output_buffer = MNN.Tensor(self._output_tensor, MNN.Tensor_DimensionType_Caffe)
        self._output_tensor.copyToHostTensor(self._host_output_buffer)
        logits = np.array(self._host_output_buffer.getData()).reshape(out_dims)

        # KV outputs
        present: Dict[str, np.ndarray] = {}
        for name, ten in self.kv_output_tensors.items():
            host = MNN.Tensor(ten, MNN.Tensor_DimensionType_Caffe)
            ten.copyToHostTensor(host)
            arr = np.array(host.getData()).reshape(ten.getShape())
            present[name] = arr
        return logits, present

    def __call__(
        self,
        input_ids: np.ndarray,
        past_key_values: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self._prepare_inputs(input_ids, past_key_values)
        self.interpreter.runSession(self.session)
        return self._collect_outputs()

# =============================================================================
# Generation
# =============================================================================

@dataclass
class GenParams:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    stop_ids: Optional[List[int]] = None
    seed: Optional[int] = None

def run_generation(
    llm: MNNLLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: List[int],
    params: GenParams,
) -> str:
    rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()

    generated: List[int] = []
    eos_id = tokenizer.eos_token_id
    stop_ids = set(params.stop_ids or ([] if eos_id is None else [eos_id]))

    # Prefill
    prompt_arr = np.asarray([prompt_tokens], dtype=np.int32)
    with Spinner("Prefill"):
        logits, kv = llm(prompt_arr)

    # last-step logits over vocab
    last_logits = logits[0, -1, :]

    next_id = sample_token(last_logits, params.temperature, params.top_p, params.top_k, rng)
    generated.append(next_id)

    if params.stream:
        # Stream detokenized chunk-wise
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.stdout.write(tokenizer.decode([next_id], skip_special_tokens=True))
        sys.stdout.flush()

    # Decode loop
    for _ in range(params.max_new_tokens - 1):
        if next_id in stop_ids:
            break
        inp = np.asarray([[next_id]], dtype=np.int32)
        logits, kv = llm(inp, kv)
        last_logits = logits[0, 0, :]

        next_id = sample_token(last_logits, params.temperature, params.top_p, params.top_k, rng)
        generated.append(next_id)

        if params.stream:
            sys.stdout.write(tokenizer.decode([next_id], skip_special_tokens=True))
            sys.stdout.flush()

    if params.stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return tokenizer.decode(generated, skip_special_tokens=True)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive chat with an MNN LLM (KV-cache).")

    g_model = p.add_argument_group("Model/Backend")
    g_model.add_argument("--model", type=str, required=True, help="Path to .mnn file.")
    g_model.add_argument("--hf_model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Tokenizer model id.")
    g_model.add_argument("--backend", type=str, default="CPU",
                         choices=["CPU", "OPENCL", "VULKAN", "METAL"], help="MNN backend.")
    g_model.add_argument("--threads", type=int, default=1, help="CPU threads (0 for auto).")

    g_gen = p.add_argument_group("Generation")
    g_gen.add_argument("--max_new_tokens", type=int, default=128)
    g_gen.add_argument("--temperature", type=float, default=0.7)
    g_gen.add_argument("--top_p", type=float, default=0.9)
    g_gen.add_argument("--top_k", type=int, default=50)
    g_gen.add_argument("--stream", action="store_true", help="Stream tokens as they are generated.")
    g_gen.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    g_gen.add_argument("--no_chat_template", action="store_true", help="Disable chat template, use raw text.")

    return p.parse_args()

def main() -> None:
    args = parse_args()

    print(f"[{now_s()}] loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)

    print(f"[{now_s()}] loading MNN model…")
    try:
        llm = MNNLLM(args.model, backend=args.backend, threads=args.threads)
        # light warmup
        warm = np.array([[tokenizer.eos_token_id or 0]], dtype=np.int32)
        _ = llm(warm)
    except Exception as e:
        print(f"[{now_s()}] error: failed to load model -> {e}")
        return

    print(f"[{now_s()}] ready. Type 'exit' or 'quit' to end.")
    print("-" * 60)

    history: List[Dict[str, str]] = lizt()
    params = GenParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.stream,
        seed=args.seed,
    )

    try:
        while True:
            user = input("👤 You: ").strip()
            if user.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye.")
                break
            if not user:
                continue

            if args.no_chat_template:
                prompt_text = user
            else:
                prompt_text = apply_chat_template(tokenizer, user, history)

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

            t0 = time.perf_counter()
            reply = run_generation(llm, tokenizer, prompt_ids, params)
            dt = time.perf_counter() - t0

            gen_tokens = tokenizer.encode(reply, add_special_tokens=False)
            tps = (len(gen_tokens) / dt) if dt > 0 else float("inf")

            print(f"\n[{now_s()}] 🤖 Assistant ({dt:.2f}s, {tps:.2f} tok/s):")
            print(reply)
            print("-" * 60)

            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": reply})

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye.")
    except Exception as e:
        print(f"\n[{now_s()}] unexpected error: {e}")

if __name__ == "__main__":
    main()
