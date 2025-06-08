#!/bin/bash
#/home/dshs-wallga/pyreft/run_all_units.sh

# 경로 설정
PYTHON_PATH="/home/dshs-wallga/dshs/bin/python"
SCRIPT_PATH="/home/dshs-wallga/pyreft/main_for_llama.py"
BASE_DIR="/home/dshs-wallga/pyreft"

# 각 unit 폴더 배열 정의
UNITS=(
    "Algebra"
    "Geometry"
    "Intermediate"
    "Number_Theory"
    "Prealgebra"
    "Precalculus"
    "Probability"
    "Untuned"
)

# 각 unit별로 스크립트 실행
for unit in "${UNITS[@]}"; do
    echo "====================================================="
    echo "시작: $unit 모델 학습 및 평가 ($(date))"
    echo "====================================================="
    
    # 임시 스크립트 생성
    TMP_SCRIPT="${BASE_DIR}/tmp_${unit}.py"
    cp "$SCRIPT_PATH" "$TMP_SCRIPT"
    
    # unit 변수 변경
    sed -i "s/unit = folders\[0\]/unit = \"${unit}\"/g" "$TMP_SCRIPT"
    
    # 실행
    $PYTHON_PATH "$TMP_SCRIPT"
    
    # 종료 상태 확인
    if [ $? -eq 0 ]; then
        echo "성공: $unit 완료 ($(date))"
    else
        echo "실패: $unit 처리 중 오류 발생 ($(date))"
    fi
    
    # 임시 스크립트 삭제
    rm "$TMP_SCRIPT"
    
    echo "---------------------------------------------------"
    echo "GPU 메모리 정리 중..."
    # GPU 메모리 정리를 위한 시간
    sleep 30
    echo "다음 단위로 진행합니다."
    echo ""
done

echo "====================================================="
echo "모든 단위 처리 완료! ($(date))"
echo "=====================================================" 