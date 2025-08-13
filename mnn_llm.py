import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import time
import numpy as np
from threading import Event, Thread
from transformers import AutoTokenizer
import MNN

# -----------------------------
# Utilities
# -----------------------------
def apply_chat_template(tokenizer, user_text: str) -> str:
    """
    Use Llama 3.2 chat template if available; otherwise return raw text.
    """
    try:
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return user_text

def softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x, dtype=np.float64)
    return (e / np.sum(e)).astype(np.float64)

def sample_from_logits(logits: np.ndarray, temperature: float = 0.0,
                       top_p: float = 0.9, top_k: int = 0) -> int:
    """
    logits: (vocab,)
    """
    if temperature <= 0.0:
        return int(np.argmax(logits))

    probs = softmax_stable(logits / temperature)

    # top-k
    if top_k and top_k > 0:
        top_idx = np.argpartition(-probs, top_k)[:top_k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_idx] = True
        probs = np.where(mask, probs, 0.0)

    # top-p (nucleus)
    if 0.0 < top_p < 1.0:
        sort_idx = np.argsort(-probs)
        sorted_probs = probs[sort_idx]
        cumsum = np.cumsum(sorted_probs)
        keep = cumsum <= top_p
        # ensure at least one token kept
        if not np.any(keep):
            keep[0] = True
        cutoff = np.where(keep)[0][-1]
        mask = np.zeros_like(probs, dtype=bool)
        mask[sort_idx[:cutoff + 1]] = True
        probs = np.where(mask, probs, 0.0)

    # renormalize (avoid zero division)
    s = probs.sum()
    if s <= 0:
        return int(np.argmax(logits))
    probs = probs / s

    return int(np.random.choice(len(probs), p=probs))

class Spinner(Thread):
    def __init__(self, stop_evt: Event, label: str = "Generating"):
        super().__init__(daemon=True)
        self.stop_evt = stop_evt
        self.label = label
        self.frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def run(self):
        i = 0
        while not self.stop_evt.is_set():
            print(f"\r{self.label} {self.frames[i % len(self.frames)]}", end="", flush=True)
            i += 1
            time.sleep(0.08)
        print("\r", end="", flush=True)

# -----------------------------
# MNN LLM wrapper
# -----------------------------
class MNNLLM:
    def __init__(self, mnn_path: str, backend: str = "CPU", threads: int = 0):
        self.interpreter = MNN.Interpreter(mnn_path)

        # Session config
        sess_cfg = MNN.SessionConfig()
        if threads and threads > 0:
            sess_cfg.numThread = int(threads)

        # Backend config (best-effort; safe on CPU if others unavailable)
        backend_cfg = MNN.BackendConfig()
        backend_cfg.precision = MNN.BackendConfig.Precision_Normal

        # Choose forwardType
        fwd_map = {
            "CPU": MNN.ForwardType.MNN_FORWARD_CPU,
            "OPENCL": MNN.ForwardType.MNN_FORWARD_OPENCL,
            "VULKAN": MNN.ForwardType.MNN_FORWARD_VULKAN,
            "METAL": MNN.ForwardType.MNN_FORWARD_METAL,
        }
        sess_cfg.forwardType = fwd_map.get(backend.upper(), MNN.ForwardType.MNN_FORWARD_CPU)

        self.session = self.interpreter.createSession(sess_cfg, backend_cfg)
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.output_tensor = self.interpreter.getSessionOutput(self.session)

        # Cache to avoid repeated resizes when len is the same
        self._last_shape = None

        # Pre-allocate host output wrapper (on-demand once we know shape)
        self._host_output = None

    def _ensure_input(self, arr: np.ndarray):
        shape = arr.shape
        if shape != self._last_shape:
            self.interpreter.resizeTensor(self.input_tensor, shape)
            self.interpreter.resizeSession(self.session)
            self._last_shape = shape

        # MNN host tensor for input (copyFrom pulls from host to device)
        host_in = MNN.Tensor(
            shape,
            MNN.Halide_Type_Int,                    # will cast below if needed
            arr.astype(np.int32, copy=False),       # most exports expect int32
            MNN.Tensor_DimensionType_Caffe
        )
        self.input_tensor.copyFrom(host_in)

    def _host_read(self) -> np.ndarray:
        """
        Device-agnostic safe read of output tensor. Returns np.ndarray.
        """
        # Re-create wrapper if shape changed
        out_dims = self.output_tensor.getShape()
        if (self._host_output is None) or (self._host_output.getShape() != out_dims):
            self._host_output = MNN.Tensor(
                self.output_tensor,
                MNN.Tensor_DimensionType_Caffe
            )
        self.output_tensor.copyToHostTensor(self._host_output)
        data = np.array(self._host_output.getData())
        # If output is flattened, try to reshape using dims
        try:
            if np.prod(out_dims) == data.size:
                data = data.reshape(out_dims)
        except Exception:
            pass
        return data

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        self._ensure_input(token_ids)
        self.interpreter.runSession(self.session)
        return self._host_read()

# -----------------------------
# Chat loop with simple generation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.mnn", help="Path to .mnn file")
    ap.add_argument("--hf_model", default="meta-llama/Llama-3.2-1B", help="Tokenizer model id")
    ap.add_argument("--backend", default="CPU", choices=["CPU","OPENCL","VULKAN","METAL"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0=auto)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    llm = MNNLLM(args.model, backend=args.backend, threads=args.threads)

    # Warm-up (helps first-token latency on some backends)
    _ = llm.forward(np.array([[eos_id]], dtype=np.int32))

    print("Type 'exit' or 'quit' to leave.\n")
    try:
        while True:
            user = input("You: ").strip()
            if user.lower() in {"exit", "quit"}:
                print("종료합니다.")
                break

            prompt = apply_chat_template(tokenizer, user)
            ids = tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
            input_ids = np.array([ids], dtype=np.int32)

            stop_evt = Event()
            spin = Spinner(stop_evt)
            spin.start()

            t0 = time.time()

            # Try one forward to discover output format
            out = llm.forward(input_ids)

            # Case A: the model already returns token IDs (rare but possible)
            decoded_text = None
            if np.issubdtype(out.dtype, np.integer) and out.ndim <= 2:
                gen_ids = out.reshape(-1).tolist()
                decoded_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Case B: logits -> run a simple autoregressive loop
            if decoded_text is None:
                cur = input_ids.copy()
                generated = []
                for _ in range(args.max_new_tokens):
                    logits = out
                    # common shapes: [B, T, V] or [T, V] or [V]
                    if logits.ndim == 3:
                        last = logits[0, -1]
                    elif logits.ndim == 2:
                        last = logits[-1]
                    else:
                        last = logits

                    next_id = sample_from_logits(
                        last, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k
                    )
                    generated.append(next_id)
                    if eos_id is not None and next_id == eos_id:
                        break

                    # Append and run again (naive loop; no KV cache)
                    cur = np.concatenate([cur, np.array([[next_id]], dtype=np.int32)], axis=1)
                    out = llm.forward(cur)

                decoded_text = tokenizer.decode(generated, skip_special_tokens=True)

            dt = time.time() - t0
            stop_evt.set()
            spin.join()

            print(f"Response ({dt:.2f}s): {decoded_text}\n")

    except KeyboardInterrupt:
        print("\n종료합니다.")

if __name__ == "__main__":
    main()
