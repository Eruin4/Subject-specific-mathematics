#!/usr/bin/env python3
#python pyreft/data_maker.py --model_path "Qwen/Qwen2.5-Math-7B-Instruct" --input_file "pyreft/data/train/Algebra.csv" --gpu_id "0"
import os
import time
import torch
import gc
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 인자 파싱 설정
parser = argparse.ArgumentParser(description='Algebra.csv에 model_solution 열 추가 스크립트')
parser.add_argument('--model_path', type=str, required=True,
                    help='사용할 모델 경로 (예: Qwen/Qwen2.5-Math-1.5B-Instruct)')
parser.add_argument('--input_file', type=str, default='pyreft/data/train/Algebra.csv',
                    help='입력 CSV 파일 경로 (기본값: pyreft/data/train/Algebra.csv)')
parser.add_argument('--output_file', type=str, default=None,
                    help='출력 CSV 파일 경로 (기본값: 입력 파일 덮어쓰기)')
parser.add_argument('--gpu_id', type=str, default="0",
                    help='사용할 GPU ID (기본값: 0)')
parser.add_argument('--prompt_template', type=str, 
                    default="""Provide a detailed, step-by-step reasoning that solves the following problem and arrives at the given answer.

Problem: {problem}

Answer: {answer}

Step-by-step solution:""",
                    help='프롬프트 템플릿 (기본값: "Solve this algebra problem step by step:\\n{problem}\\n\\nSolution:")')
args = parser.parse_args()

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# 출력 파일 설정
if args.output_file is None:
    args.output_file = args.input_file

def clear_cuda_memory():
    """GPU 메모리 정리"""
    print("CUDA 메모리 정리 중...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        print(f"메모리 정리 중 오류 발생: {e}")

    gc.collect()
    time.sleep(1)
    print("CUDA 메모리 정리 완료")

def init_model(model_path):
    """vLLM 모델 초기화"""
    clear_cuda_memory()
    print(f"vLLM 모델 초기화: {model_path}")
    
    vllm_model = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    
    return vllm_model

def process_problems(model, problems, answers, prompt_template):
    """문제 전체를 한 번에 처리"""
    print(f"총 {len(problems)}개 문제 처리 중...")
    
    # 프롬프트 준비
    prompts = [prompt_template.format(problem=problems[i], answer=answers[i]) for i in range(len(problems))]
    
    # vLLM 설정
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )
    
    # 모든 문제 처리 시작
    start_time = time.time()
    outputs = model.generate(prompts, sampling_params)
    
    # 결과 추출
    solutions = []
    for output in tqdm(outputs, desc="결과 추출 중"):
        solution = output.outputs[0].text.strip()
        solutions.append(solution)
    
    elapsed_time = time.time() - start_time
    print(f"전체 처리 완료: {elapsed_time:.2f}초, 평균 {elapsed_time/len(problems):.2f}초/문제")
    
    return solutions

def main():
    """메인 함수"""
    # 시작 시간 기록
    start_time = time.time()
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델 경로: {args.model_path}")
    print(f"입력 파일: {args.input_file}")
    print(f"출력 파일: {args.output_file}")
    
    # 모델 초기화
    model = init_model(args.model_path)
    
    try:
        # CSV 파일 로드
        print(f"CSV 파일 로드 중: {args.input_file}")
        df = pd.read_csv(args.input_file)
        print(f"총 {len(df)}개 문제 발견")
        
        # 문제 추출
        problems = df['problem'].tolist()
        answers = df['answer'].tolist()
        
        # 문제 처리 (전체 한 번에)
        solutions = process_problems(model, problems, answers, args.prompt_template)
        
        # 결과 저장
        df['model_solution'] = solutions
        df.to_csv(args.output_file, index=False)
        
        print(f"\n처리 완료: {time.time() - start_time:.2f}초 소요")
        print(f"결과 저장 경로: {args.output_file}")
        
    finally:
        # 모델 정리
        try:
            del model
            clear_cuda_memory()
        except Exception as e:
            print(f"모델 정리 중 오류: {e}")
            pass

if __name__ == "__main__":
    main()