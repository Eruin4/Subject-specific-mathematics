import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
import pandas as pd
from datasets import Dataset
import json
import time
from vllm import LLM, SamplingParams  # vLLM 추가
import datetime
import sys
# 로그 파일 설정
log_filename = f"pyreft/log/evaluate_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 표준 출력과 파일에 동시에 출력하는 클래스
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 실시간으로 파일에 내용을 쓰기 위해 flush 사용
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 로거 설정
sys.stdout = Logger(log_filename)

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 모델 및 토크나이저 로드
model_id = 'meta-llama/Llama-3.2-3B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Llama 모델은 기본적으로 pad_token이 없으므로 설정해줍니다
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"  # 디코더 전용 모델 권장

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id


# 경로 설정
model_saving_path = "/home/dshs-wallga/pyreft/models"
answer_path = "/home/dshs-wallga/pyreft/answers"
training_path = "/home/dshs-wallga/pyreft/data/train"
test_path = "/home/dshs-wallga/pyreft/data/test"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folders = [
    "Algebra",
    "Geometry",
    "Intermediate",
    "Number_Theory",
    "Prealgebra",
    "Precalculus",
    "Probability"
]

# 현재 학습할 단위 (세션 재시작 시 GPU 메모리를 위해 증분 학습)
unit = folders[0]
# 데이터 로드
filename = os.path.join(training_path, unit + ".csv")
print(f"Loading data from {filename}")

# CSV 파일 로드
df = pd.read_csv(filename)
df['problem'] = df['problem'].fillna('').astype(str)
df['model_solution'] = df['model_solution'].fillna('').astype(str)

# 데이터 형식 변환
converted_data = []
for idx, row in df.iterrows():
    new_item = {
        "instruction": row.get("problem", ""),
        "output": row.get("model_solution", ""),
        "url": ""  # URL 정보가 없으므로 빈 문자열로 채웁니다.
    }
    converted_data.append(new_item)

# Hugging Face 데이터셋으로 변환
train_dataset = Dataset.from_list(converted_data)

# 데이터셋 확인
print(train_dataset)
for i in range(0, min(500, len(train_dataset)), 100):
    print(train_dataset[i])


# 0. 공통 메시지 빌더 ─ system→user 순서를 강제
def build_messages(user_question: str,
                   system_prompt: str = f"Please reason step by step:"):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_question},
    ]

def preprocessing_data(examples):
    input_ids, attention_masks, labels = [], [], []
    max_len = 2400

    for instr, resp in zip(examples["instruction"], examples["output"]):
        # ← 기존 [{'role':'user',...}] 를 build_messages()로 교체
        enc = tokenizer.apply_chat_template(
            build_messages(instr),
            tokenize=True,
            add_generation_prompt=True       # ↔ 평가 때도 반드시 True
        )

        if len(enc) > max_len:
            enc = enc[-max_len:]          # 끝부분 유지(좌측 padding 전략과 호환)

        dec = tokenizer(resp, add_special_tokens=False)["input_ids"]


        pad_len = max_len - len(enc) - len(dec)
        if pad_len < 0:
            dec = dec[:max_len - len(enc)]
            pad_len = 0

        ids  = [tokenizer.pad_token_id]*pad_len + enc + dec
        labs = [-100]*pad_len + [-100]*len(enc) + dec
        mask = [0]*pad_len + [1]*(len(enc)+len(dec))

        input_ids.append(ids); labels.append(labs); attention_masks.append(mask)

    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels}


# 데이터셋 전처리 적용
train_dataset = train_dataset.map(
    preprocessing_data,
    batched=True, num_proc=2,
    remove_columns=['instruction', 'output', 'url']
)
train_dataset.set_format(type="torch",
                         columns=["input_ids", "attention_mask", "labels"])

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=5,
    learning_rate=5e-5,
    remove_unused_columns=False,
    report_to="none",
    logging_strategy='steps',
    label_names=['labels']
)

# Lora Tuning - Llama 3.2 모델용으로 타겟 모듈 설정
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

lora_model = get_peft_model(model, peft_config)

# 모델 훈련
print("Starting model training...")
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# LoRA 모델 저장
lora_save_directory = os.path.join(model_saving_path, f"{unit}_lora")
print(f"Saving LoRA model to {lora_save_directory}")
lora_model.save_pretrained(
    lora_save_directory,
    save_embedding_layers=True
)
tokenizer.save_pretrained(lora_save_directory)

# LoRA 가중치를 기본 모델과 병합하여 완전한 모델 생성
print("Merging LoRA weights with the base model...")
merged_model = PeftModel.from_pretrained(model, lora_save_directory)
merged_model = merged_model.merge_and_unload()

# 병합된 모델 저장
merged_save_directory = os.path.join(model_saving_path, f"{unit}_merged")
print(f"Saving merged model to {merged_save_directory}")
merged_model.save_pretrained(merged_save_directory)
tokenizer.save_pretrained(merged_save_directory)

# GPU 메모리 정리
del lora_model
del model
del merged_model
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# vLLM을 사용한 평가 함수
def evaluate_with_vllm(model_path, test_folders):
    """vLLM을 사용하여 모델의 성능을 빠르게 평가합니다."""
    print("\n=== vLLM을 사용한 평가 시작 ===")
    
    # vLLM 모델 초기화
    print(f"vLLM 모델 초기화 중... (경로: {model_path})")
    vllm_model = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,  # 높은 활용도 설정
        swap_space=0,
        trust_remote_code=True,
    )
    
    # 샘플링 파라미터 설정
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|eot_id|>"] 
    )

    
    for test_unit in test_folders:
        print(f"Testing on {test_unit}")
        result = []
        
        # 테스트 데이터 로드
        testfile = os.path.join(test_path, test_unit + ".csv")
        df = pd.read_csv(testfile)
        
        # 배치 처리를 위한 준비
        all_problems = []
        all_solutions = []
        
        for idx, row in df.iterrows():
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            all_problems.append(problem)
            all_solutions.append(solution)
        
        # 배치 처리를 위한 프롬프트 생성
        batch_texts = []
        for problem in all_problems:
            prompt = tokenizer.apply_chat_template(
                build_messages(problem),           # 동일 함수 사용
                tokenize=False,
                add_generation_prompt=True
            )

            batch_texts.append(prompt)

        
        # 배치 처리 시작
        start_time = time.time()
        print(f"Batch processing {len(batch_texts)} problems...")
        outputs = vllm_model.generate(batch_texts, sampling_params)
        
        # 결과 처리
        for i, output in enumerate(outputs):
            model_answer = output.outputs[0].text
            result.append({
                "problem": all_problems[i],
                "solution": all_solutions[i],
                "model_answer": model_answer
            })
        
        elapsed_time = time.time() - start_time
        print(f"Batch processing completed in {elapsed_time:.2f} seconds, {elapsed_time/len(batch_texts):.2f} seconds/problem")
        
        # 결과 저장
        result_save_path = os.path.join(answer_path, f"{unit}", test_unit + ".json")
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
        with open(result_save_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

# 병합된 모델로 vLLM 평가 실행
print("Starting evaluation with vLLM using the merged model...")
evaluate_with_vllm(
    model_path=merged_save_directory,  # 병합된 모델 경로
    test_folders=folders
)

print("All training and evaluation completed.") 