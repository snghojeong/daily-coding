import MNN
import numpy as np
from transformers import AutoTokenizer
import time
from threading import Thread
from tqdm import tqdm
import os

# 병렬 경고 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MNN 모델 로딩
interpreter = MNN.Interpreter("llama3.mnn")
session = interpreter.createSession()
input_tensor = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)

def tokenize(text):
    ids = tokenizer(text)["input_ids"]
    return np.array([ids], dtype=np.int32)

def show_progress(stop_flag):
    for _ in tqdm(iter(int, 1), desc="Generating", leave=False):
        if stop_flag[0]:
            break
        time.sleep(0.1)

# 대화 루프
while True:
    prompt = input("You: ")
    if prompt.strip().lower() in {"exit", "quit"}:
        print("종료합니다.")
        break

    input_data = tokenize(prompt)
    shape = input_data.shape

    # 입력 텐서 리사이즈
    interpreter.resizeTensor(input_tensor, shape)
    interpreter.resizeSession(session)

    # 새로운 Tensor 생성 → 값 복사
    tmp_input = MNN.Tensor(shape, MNN.Halide_Type_Int, input_data, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)

    stop_flag = [False]
    progress_thread = Thread(target=show_progress, args=(stop_flag,))
    progress_thread.start()

    interpreter.runSession(session)

    stop_flag[0] = True
    progress_thread.join()

    output_data = output_tensor.getData()

    if isinstance(output_data, (list, np.ndarray)):
        decoded = tokenizer.decode(output_data, skip_special_tokens=True)
    else:
        decoded = str(output_data)

    print("Response:", decoded)
