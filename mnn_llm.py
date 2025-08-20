import argparse
import os
import time
from threading import Event, Thread
from typing import Dict, List, Optional

import MNN
import numpy as np
from transformers import AutoTokenizer

# Suppress tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def apply_chat_template(tokenizer: AutoTokenizer, user_text: str) -> str:
    """Applies the model's chat template if available, otherwise returns raw text."""
    try:
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        print("Warning: Could not apply chat template. Using raw prompt.")
        return user_text

def softmax(x: np.ndarray) -> np.ndarray:
    """Computes a numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sample(logits: np.ndarray, temp: float, top_p: float, top_k: int) -> int:
    """
    Samples a token ID from logits using temperature, top-p (nucleus), and top-k sampling.

    Args:
        logits: A 1D numpy array of raw model logits.
        temp: The temperature for sampling. 0 means greedy.
        top_p: The nucleus sampling probability.
        top_k: The number of top candidates to consider.

    Returns:
        The sampled token ID as an integer.
    """
    if temp == 0.0:
        return int(np.argmax(logits))

    probs = softmax(logits / temp)

    if top_k > 0:
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_k_indices] = True
        probs = np.where(mask, probs, 0.0)

    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Find indices to remove
        indices_to_remove = cumulative_probs > top_p
        # Shift the mask to the right to keep the first element that exceeds top_p
        indices_to_remove[1:] = indices_to_remove[:-1]
        indices_to_remove[0] = False
        
        probs[sorted_indices[indices_to_remove]] = 0.0

    # Renormalize and sample
    norm = np.sum(probs)
    if norm <= 1e-9:  # Fallback to greedy if all probabilities are zero
        return int(np.argmax(logits))
    
    probs /= norm
    return int(np.random.choice(len(probs), p=probs))


class Spinner(Thread):
    """
    A simple spinner to indicate background activity, implemented as a context manager.
    
    Usage:
        with Spinner("Working..."):
            time.sleep(3)
    """
    def __init__(self, label: str = "Generating"):
        super().__init__(daemon=True)
        self._stop_event = Event()
        self._label = label
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def run(self):
        i = 0
        while not self._stop_event.is_set():
            frame = self._frames[i % len(self._frames)]
            print(f"\r{self._label} {frame}", end="", flush=True)
            i += 1
            time.sleep(0.08)
        # Clear the line
        print("\r" + " " * (len(self._label) + 2) + "\r", end="", flush=True)
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self.join()

# --------------------------------------------------------------------------
# MNN LLM Wrapper
# --------------------------------------------------------------------------

class MNNLLM:
    """A wrapper for an MNN Large Language Model."""
    
    _BACKEND_MAP: Dict[str, MNN.ForwardType] = {
        "CPU": MNN.ForwardType.MNN_FORWARD_CPU,
        "OPENCL": MNN.ForwardType.MNN_FORWARD_OPENCL,
        "VULKAN": MNN.ForwardType.MNN_FORWARD_VULKAN,
        "METAL": MNN.ForwardType.MNN_FORWARD_METAL,
    }

    def __init__(self, model_path: str, backend: str = "CPU", threads: int = 4):
        """
        Initializes the MNN interpreter and session.

        Args:
            model_path: Path to the .mnn model file.
            backend: The computation backend to use (e.g., "CPU", "OPENCL").
            threads: Number of threads to use for CPU backend.
        """
        self.interpreter = MNN.Interpreter(model_path)
        
        sess_cfg = MNN.SessionConfig()
        if threads > 0:
            sess_cfg.numThread = threads

        backend_cfg = MNN.BackendConfig()
        backend_cfg.precision = MNN.BackendConfig.Precision_Normal

        backend_type = self._BACKEND_MAP.get(backend.upper(), MNN.ForwardType.MNN_FORWARD_CPU)
        sess_cfg.forwardType = backend_type

        self.session = self.interpreter.createSession(sess_cfg, backend_cfg)
        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.output_tensor = self.interpreter.getSessionOutput(self.session)
        self._last_input_shape: Optional[tuple] = None
        self._host_output_buffer: Optional[MNN.Tensor] = None

    def _prepare_input(self, token_ids: np.ndarray):
        """Resizes session if needed and copies input data to the device."""
        if token_ids.shape != self._last_input_shape:
            self.interpreter.resizeTensor(self.input_tensor, token_ids.shape)
            self.interpreter.resizeSession(self.session)
            self._last_input_shape = token_ids.shape

        host_tensor = MNN.Tensor(
            token_ids.shape,
            MNN.Halide_Type_Int,
            token_ids.astype(np.int32), # Most models expect int32
            MNN.Tensor_DimensionType_Caffe,
        )
        self.input_tensor.copyFrom(host_tensor)

    def _read_output(self) -> np.ndarray:
        """Copies output data from the device and returns it as a NumPy array."""
        out_dims = self.output_tensor.getShape()
        
        if self._host_output_buffer is None or self._host_output_buffer.getShape() != out_dims:
            self._host_output_buffer = MNN.Tensor(
                self.output_tensor, MNN.Tensor_DimensionType_Caffe
            )

        self.output_tensor.copyToHostTensor(self._host_output_buffer)
        data = np.array(self._host_output_buffer.getData())

        # Ensure correct shape, as getData() can return a flattened array
        if np.prod(out_dims) == data.size:
            data = data.reshape(out_dims)
        
        return data

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the model.

        Args:
            token_ids: A NumPy array of input token IDs, typically shape (1, seq_len).

        Returns:
            A NumPy array of the model's output logits.
        """
        self._prepare_input(token_ids)
        self.interpreter.runSession(self.session)
        return self._read_output()

# --------------------------------------------------------------------------
# Generation Logic
# --------------------------------------------------------------------------

def run_generation(
    llm: MNNLLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: List[int],
    args: argparse.Namespace,
):
    """Runs the autoregressive generation loop."""
    
    # NOTE: This is a naive implementation without a KV cache.
    # In each step, the model re-processes all tokens. A KV cache-enabled model
    # would only need to process the newest token after the first pass.
    
    generated_tokens: List[int] = []
    all_tokens = np.array([prompt_tokens], dtype=np.int32)
    eos_token_id = tokenizer.eos_token_id

    for _ in range(args.max_new_tokens):
        logits = llm(all_tokens)

        # Extract logits for the very last token
        # Common shapes: [batch, sequence_len, vocab_size]
        last_token_logits = logits[0, -1, :]

        next_token_id = sample(
            logits=last_token_logits,
            temp=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        generated_tokens.append(next_token_id)
        
        # Append the new token and continue the loop
        all_tokens = np.append(
            all_tokens, [[next_token_id]], axis=1
        ).astype(np.int32)

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

# --------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a chat session with an MNN LLM.")
    
    model_group = parser.add_argument_group("Model and Backend Configuration")
    model_group.add_argument("--model", type=str, required=True, help="Path to the .mnn model file.")
    model_group.add_argument("--hf_model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Hugging Face model ID for the tokenizer.")
    model_group.add_argument("--backend", type=str, default="CPU", choices=["CPU", "OPENCL", "VULKAN", "METAL"], help="MNN backend to use.")
    model_group.add_argument("--threads", type=int, default=4, help="Number of CPU threads for inference (0 for auto).")

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    gen_group.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature. Set to 0 for greedy decoding.")
    gen_group.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling (top-p) probability.")
    gen_group.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff.")
    
    return parser.parse_args()

def main():
    """Main function to run the interactive chat loop."""
    args = parse_args()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    llm = MNNLLM(args.model, backend=args.backend, threads=args.threads)

    # Warm-up run can reduce first-token latency
    warmup_token = tokenizer.eos_token_id or 0
    _ = llm(np.array([[warmup_token]], dtype=np.int32))

    print("\n✅ Model loaded. Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    try:
        while True:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                break
            if not user_input:
                continue

            prompt_text = apply_chat_template(tokenizer, user_input)
            prompt_tokens = tokenizer.encode(prompt_text)

            t_start = time.perf_counter()
            with Spinner("Generating..."):
                response_text = run_generation(llm, tokenizer, prompt_tokens, args)
            t_delta = time.perf_counter() - t_start
            
            num_generated = len(tokenizer.encode(response_text))
            tps = num_generated / t_delta if t_delta > 0 else float('inf')

            print(f"🤖 Assistant ({t_delta:.2f}s, {tps:.2f} t/s):")
            print(response_text)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
