#!/usr/bin/env python3
import argparse
import os
import sys
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple

import numpy as np
import MNN
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def get_timestamp() -> str:
    return time.strftime("%H:%M:%S")

def apply_chat_template(tokenizer, user_text: str, history: List[Dict[str, str]]) -> str:
    messages = history + [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"[{get_timestamp()}] Warning: Chat template failed ({e}). Using fallback.")
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

def sample_token(logits: np.ndarray, params: 'GenParams', rng: np.random.Generator) -> int:
    """Sample a token using Temperature, Top-K, and Top-P filtering."""
    if params.temperature <= 0:
        return int(np.argmax(logits))

    # Apply Temperature
    logits = logits.astype(np.float64) / params.temperature
    
    # Softmax
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()

    # Top-K
    if params.top_k > 0:
        indices_to_remove = probs < np.partition(probs, -params.top_k)[-params.top_k]
        probs[indices_to_remove] = 0
        probs /= probs.sum()

    # Top-P (Nucleus)
    if 0.0 < params.top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > params.top_p
        # Shift to keep the first token that exceeds the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        
        probs[sorted_indices[sorted_indices_to_remove]] = 0
        probs /= probs.sum()

    return int(rng.choice(len(probs), p=probs))

class Spinner:
    """A simple context manager for a CLI spinner."""
    def __init__(self, message: str):
        self.message = message
        self.stop_event = Event()
        self.thread = Thread(target=self._spin, daemon=True)

    def _spin(self):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f"\r{self.message} {frames[i % len(frames)]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()

class MNNLLM:
    """MNN Inference Wrapper with KV-Cache support."""
    def __init__(self, model_path: str, backend: str = "CPU", threads: int = 4):
        self.interpreter = MNN.Interpreter(model_path)
        
        config = MNN.SessionConfig()
        config.numThread = threads
        config.forwardType = getattr(MNN.ForwardType, f"MNN_FORWARD_{backend.upper()}", MNN.ForwardType.MNN_FORWARD_CPU)
        
        self.session = self.interpreter.createSession(config)
        
        # IO Mapping
        self.input_tensor = self.interpreter.getSessionInput(self.session, None)
        self.output_tensor = self.interpreter.getSessionOutput(self.session, None)
        
        # KV Cache discovery
        all_inputs = self.interpreter.getSessionInputNames(self.session)
        self.kv_inputs = {n: self.interpreter.getSessionInput(self.session, n) for n in all_inputs if "cache" in n or "past" in n}
        self.kv_outputs = {n: self.interpreter.getSessionOutput(self.session, n) for n in self.interpreter.getSessionOutputNames(self.session) if "cache" in n or "present" in n}

    def _copy_to_tensor(self, mnn_tensor, np_data):
        self.interpreter.resizeTensor(mnn_tensor, np_data.shape)
        # Re-syncing session is required after resizing
        self.interpreter.resizeSession(self.session)
        
        tmp = MNN.Tensor(np_data.shape, MNN.Halide_Type_Float if np_data.dtype == np.float32 else MNN.Halide_Type_Int, 
                         np_data, MNN.Tensor_DimensionType_Caffe)
        mnn_tensor.copyFrom(tmp)

    def run(self, input_ids: np.ndarray, kv_cache: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        self._copy_to_tensor(self.input_tensor, input_ids.astype(np.int32))
        
        if kv_cache:
            for name, data in kv_cache.items():
                if name in self.kv_inputs:
                    self._copy_to_tensor(self.kv_inputs[name], data)

        self.interpreter.runSession(self.session)
        
        # Extract results
        logits = np.array(self.output_tensor.getData()).reshape(self.output_tensor.getShape())
        new_kv = {n: np.array(t.getData()).reshape(t.getShape()) for n, t in self.kv_outputs.items()}
        
        return logits, new_kv

@dataclass
class GenParams:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = True
    seed: Optional[int] = None

def generate(llm: MNNLLM, tokenizer: AutoTokenizer, prompt_ids: List[int], params: GenParams):
    rng = np.random.default_rng(params.seed)
    tokens = []
    kv_cache = None
    curr_input = np.array([prompt_ids])

    for i in range(params.max_new_tokens):
        # Spinner only for the first heavy prefill
        if i == 0:
            with Spinner("Thinking..."):
                logits, kv_cache = llm.run(curr_input)
        else:
            logits, kv_cache = llm.run(curr_input, kv_cache)

        next_token = sample_token(logits[0, -1, :], params, rng)
        tokens.append(next_token)
        
        if next_token == tokenizer.eos_token_id:
            break

        if params.stream:
            print(tokenizer.decode([next_token]), end="", flush=True)

        curr_input = np.array([[next_token]])

    return tokenizer.decode(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--backend", default="CPU")
    args = parser.parse_args()

    print(f"[{get_timestamp()}] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    llm = MNNLLM(args.model, backend=args.backend)
    
    params = GenParams()
    history = []

    print(f"[{get_timestamp()}] System Ready.")
    
    try:
        while True:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ("exit", "quit"): break
            
            prompt = apply_chat_template(tokenizer, user_input, history)
            prompt_ids = tokenizer.encode(prompt)

            print(f"🤖 AI: ", end="")
            response = generate(llm, tokenizer, prompt_ids, params)
            print() # Newline after stream
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
    except KeyboardInterrupt:
        print("\nExit.")

if __name__ == "__main__":
    main()
