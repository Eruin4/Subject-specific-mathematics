#!/bin/bash
#/home/dshs-wallga/pyreft/data/process_all_data.sh

# 모델 경로와 GPU ID 설정
MODEL_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
GPU_ID="0"
DATA_DIR="pyreft/data/train"

# 시작 시간 기록
START_TIME=$(date +%s)
echo "데이터 처리 시작: $(date)"
echo "모델: $MODEL_PATH"
echo "-----------------------------------"

# 모든 CSV 파일 찾기
for csv_file in "$DATA_DIR"/*.csv; do
    # 파일명 추출
    FILENAME=$(basename "$csv_file")
    
    echo "처리 중: $FILENAME"
    echo "경로: $csv_file"
    
    # data_maker.py 실행
    python pyreft/data/data_maker.py --model_path "$MODEL_PATH" --input_file "$csv_file" --gpu_id "$GPU_ID"
    
    echo "완료: $FILENAME"
    echo "-----------------------------------"
done

# 종료 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo "모든 데이터 처리 완료"
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"