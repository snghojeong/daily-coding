import argparse
import os
import time
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple

import MNN
import numpy as np
from transformers import AutoTokenizer

# Suppress tokenizer parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def apply_chat_template(tokenizer: AutoTokenizer, user_text: str, chat_history: Optional[List[Dict]] = None) -> str:
    """Applies the model's chat template, using a chat history if provided."""
    messages = chat_history or list()
    messages.append({"role": "user", "content": user_text})
    
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        print(f"Warning: Could not apply chat template. Using raw prompt. {e}")
        # Fallback to a simple concatenation if template fails
        full_text = ""
        for msg in messages:
            full_text += f"{msg['role'].title()}: {msg['content']}\n"
        return full_text.strip()

def softmax(x: np.ndarray) -> np.ndarray:
    """Computes a numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sample(logits: np.ndarray, temp: float, top_p: float, top_k: int) -> int:
    """Samples a token ID from logits using temperature, top-p, and top-k."""
    if temp == 0.0:
        return int(np.argmax(logits))

    probs = softmax(logits / temp)

    if top_k > 0:
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        probs = np.where(np.isin(np.arange(len(probs)), top_k_indices), probs, 0.0)

    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        cutoff_index = (cumulative_probs > top_p).argmax()
        probs[sorted_indices[cutoff_index:]] = 0.0
    
    norm = np.sum(probs)
    if norm <= 1e-9:
        return int(np.argmax(logits))
    
    probs /= norm
    return int(np.random.choice(len(probs), p=probs))

class Spinner(Thread):
    """A simple spinner to indicate background activity, implemented as a context manager."""
    def __init__(self, label: str = "Generating"):
        super().__init__(daemon=True)
        self._stop_event = Event()
        self._label = label
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._is_spinning = False

    def run(self):
        self._is_spinning = True
        i = 0
        while not self._stop_event.is_set():
            frame = self._frames[i % len(self._frames)]
            print(f"\r{self._label} {frame}", end="", flush=True)
            i += 1
            time.sleep(0.08)
        # Clear the line
        print("\r" + " " * (len(self._label) + 2) + "\r", end="", flush=True)
        self._is_spinning = False
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._is_spinning:
            self._stop_event.set()
            self.join()

# --------------------------------------------------------------------------
# MNN LLM Wrapper (KV Cache-enabled)
# --------------------------------------------------------------------------

class MNNLLM:
    """A wrapper for a KV-cache-enabled MNN Large Language Model."""
    
    _BACKEND_MAP: Dict[str, MNN.ForwardType] = {
        "CPU": MNN.ForwardType.MNN_FORWARD_CPU,
        "OPENCL": MNN.ForwardType.MNN_FORWARD_OPENCL,
        "VULKAN": MNN.ForwardType.MNN_FORWARD_VULKAN,
        "METAL": MNN.ForwardType.MNN_FORWARD_METAL,
    }

    def __init__(self, model_path: str, backend: str = "CPU", threads: int = 4):
        """Initializes the MNN interpreter and session for a KV-cache-enabled model."""
        self.interpreter = MNN.Interpreter(model_path)
        
        sess_cfg = MNN.SessionConfig()
        sess_cfg.numThread = threads
        backend_cfg = MNN.BackendConfig()
        backend_cfg.precision = MNN.BackendConfig.Precision_Normal
        backend_type = self._BACKEND_MAP.get(backend.upper(), MNN.ForwardType.MNN_FORWARD_CPU)
        sess_cfg.forwardType = backend_type
        
        self.session = self.interpreter.createSession(sess_cfg, backend_cfg)
        
        # Identify the primary input/output tensors and KV cache tensors
        # Assuming the MNN model is structured with KV cache inputs/outputs
        self._input_tensor = self.interpreter.getSessionInput(self.session)
        self._output_tensor = self.interpreter.getSessionOutput(self.session)

        # Assuming KV cache tensors are named as 'past_key_values' and 'present_key_values'
        # The actual names might differ based on the model converter
        # This part requires knowledge of the specific MNN model's structure
        self.kv_input_names = [name for name in self.interpreter.getSessionInputNames(self.session) if 'past_key_values' in name]
        self.kv_output_names = [name for name in self.interpreter.getSessionOutputNames(self.session) if 'present_key_values' in name]
        
        self.kv_input_tensors = {name: self.interpreter.getSessionInput(self.session, name) for name in self.kv_input_names}
        self.kv_output_tensors = {name: self.interpreter.getSessionOutput(self.session, name) for name in self.kv_output_names}
        
        self._host_output_buffer: Optional[MNN.Tensor] = None

    def _prepare_input_and_kv_cache(self, input_ids: np.ndarray, past_key_values: Optional[Dict[str, np.ndarray]] = None):
        """Resizes session if needed and copies input data and KV cache to the device."""
        
        # Resize input tensor
        self.interpreter.resizeTensor(self._input_tensor, input_ids.shape)
        
        # Resize KV cache input tensors based on past_key_values shape
        if past_key_values:
            for name, data in past_key_values.items():
                if name in self.kv_input_tensors:
                    self.interpreter.resizeTensor(self.kv_input_tensors[name], data.shape)
        
        self.interpreter.resizeSession(self.session)

        # Copy input_ids
        host_input = MNN.Tensor(input_ids.shape, MNN.Halide_Type_Int, input_ids.astype(np.int32), MNN.Tensor_DimensionType_Caffe)
        self._input_tensor.copyFrom(host_input)
        
        # Copy past_key_values
        if past_key_values:
            for name, data in past_key_values.items():
                if name in self.kv_input_tensors:
                    host_kv_input = MNN.Tensor(data.shape, MNN.Halide_Type_Float, data.astype(np.float32), MNN.Tensor_DimensionType_Caffe)
                    self.kv_input_tensors[name].copyFrom(host_kv_input)

    def _read_output_and_kv_cache(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Copies output data and KV cache from the device."""
        # Read primary output tensor
        out_dims = self._output_tensor.getShape()
        if self._host_output_buffer is None or self._host_output_buffer.getShape() != out_dims:
            self._host_output_buffer = MNN.Tensor(self._output_tensor, MNN.Tensor_DimensionType_Caffe)
        self._output_tensor.copyToHostTensor(self._host_output_buffer)
        output_data = np.array(self._host_output_buffer.getData()).reshape(out_dims)
        
        # Read KV cache output tensors
        present_key_values = {}
        for name, tensor in self.kv_output_tensors.items():
            out_dims_kv = tensor.getShape()
            host_kv_output = MNN.Tensor(tensor, MNN.Tensor_DimensionType_Caffe)
            tensor.copyToHostTensor(host_kv_output)
            present_key_values[name] = np.array(host_kv_output.getData()).reshape(out_dims_kv)
            
        return output_data, present_key_values

    def __call__(self, input_ids: np.ndarray, past_key_values: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Performs a forward pass.

        Args:
            input_ids: A NumPy array of input token IDs.
            past_key_values: A dictionary of NumPy arrays representing the KV cache from the previous step.

        Returns:
            A tuple of (logits, present_key_values).
        """
        self._prepare_input_and_kv_cache(input_ids, past_key_values)
        self.interpreter.runSession(self.session)
        return self._read_output_and_kv_cache()

# --------------------------------------------------------------------------
# Generation Logic (KV Cache-enabled)
# --------------------------------------------------------------------------

def run_generation(
    llm: MNNLLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: List[int],
    args: argparse.Namespace,
):
    """Runs the autoregressive generation loop with a KV cache."""
    
    generated_tokens: List[int] = []
    eos_token_id = tokenizer.eos_token_id
    
    # 1. Prefill Pass (for prompt)
    prompt_array = np.array([prompt_tokens], dtype=np.int32)
    with Spinner("Prefilling"):
        logits, past_key_values = llm(prompt_array)
    
    # Extract logits for the last token of the prompt
    last_token_logits = logits[0, -1, :]
    
    next_token_id = sample(
        logits=last_token_logits,
        temp=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    generated_tokens.append(next_token_id)

    # 2. Decoding Loop (one token at a time)
    for _ in range(args.max_new_tokens - 1):
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        
        # Input is now just the last generated token
        input_array = np.array([[next_token_id]], dtype=np.int32)
        
        # The model uses the KV cache from the previous step
        logits, past_key_values = llm(input_array, past_key_values)
        
        # Logits shape is now [1, 1, vocab_size]
        last_token_logits = logits[0, 0, :]

        next_token_id = sample(
            logits=last_token_logits,
            temp=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        generated_tokens.append(next_token_id)

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
    model_group.add_argument("--threads", type=int, default=1, help="Number of CPU threads for inference (0 for auto).")

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
    
    try:
        llm = MNNLLM(args.model, backend=args.backend, threads=args.threads)
        # Warm-up run with KV cache logic
        warmup_token = tokenizer.eos_token_id or 0
        _, _ = llm(np.array([[warmup_token]], dtype=np.int32))
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return

    print("\n✅ Model loaded. Type 'exit' or 'quit' to end the session.")
    print("-" * 50)

    chat_history: List[Dict] = []
    
    try:
        while True:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("\n👋 Goodbye!")
                break
            if not user_input:
                continue

            # Update chat history and apply template
            prompt_text = apply_chat_template(tokenizer, user_input, chat_history)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)

            t_start = time.perf_counter()
            response_text = run_generation(llm, tokenizer, prompt_tokens, args)
            t_delta = time.perf_counter() - t_start
            
            num_generated = len(tokenizer.encode(response_text, add_special_tokens=False))
            tps = num_generated / t_delta if t_delta > 0 else float('inf')

            print(f"\n🤖 Assistant ({t_delta:.2f}s, {tps:.2f} t/s):")
            print(response_text)
            print("-" * 50)
            
            # Add user and assistant messages to history for the next turn
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
