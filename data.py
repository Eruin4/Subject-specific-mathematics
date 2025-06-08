import os
import json

# 모델(폴더) 목록 – 실제 답안을 가진 폴더 이름입니다.
model_folders = [
    "Algebra",
    "Geometry",
    "Intermediate",
    "Number_Theory",
    "Prealgebra",
    "Precalculus",
    "Probability",
    "Untuned"
]

# 집계할 시험 과목(파일명 앞부분) 목록 – 7개 시험 과목 전체에 대해 집계합니다.
exam_subjects = [
    "Algebra",
    "Geometry",
    "Intermediate",
    "Number_Theory",
    "Prealgebra",
    "Precalculus",
    "Probability"
]

# 답안들이 위치한 기본 경로
base_dir = '/home/dshs-wallga/pyreft/answers'

# 최종 집계 데이터를 저장할 딕셔너리
# key: exam_subject, value: 문제별 집계 리스트
aggregated_data = {}

# 시험 과목별로 처리
for exam in exam_subjects:
    # 각 모델에서 읽은 데이터를 담기 위한 임시 딕셔너리
    model_data = {}
    for model in model_folders:
        file_path = os.path.join(base_dir, model, exam + ".json")
        if os.path.exists(file_path):
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                # 각 JSON 파일은 리스트 형태로 여러 문제를 포함한다고 가정합니다.
                model_data[model] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # 기준 모델 데이터를 첫 번째 모델 폴더의 데이터를 사용 (여기서는 model_folders[0])
    base_model = model_folders[0]
    if base_model not in model_data:
        print(f"기준 모델 '{base_model}'의 데이터가 없으므로 exam_subject '{exam}'은 건너뜁니다.")
        continue

    num_questions = len(model_data[base_model])
    aggregated_questions = []

    for i in range(num_questions):
        aggregated_entry = {}
        # 기준 모델의 해당 문제 항목에서 "문제"와 "정답" 가져오기
        base_entry = model_data[base_model][i]
        aggregated_entry["problem"] = base_entry.get("problem", "")
        aggregated_entry["solution"] = base_entry.get("solution", "")
        # 각 모델별로 같은 인덱스의 "모델답"을 바로 문자열로 저장
        for model in model_folders:
            model_list = model_data.get(model, [])
            if i < len(model_list):
                aggregated_entry[model] = model_list[i].get("model_answer", "")
            else:
                aggregated_entry[model] = ""
        aggregated_questions.append(aggregated_entry)

    aggregated_data[exam] = aggregated_questions

# 집계된 데이터를 시험 과목별로 개별 JSON 파일로 저장
output_dir = '/home/dshs-wallga/pyreft/aggregated_exams'
os.makedirs(output_dir, exist_ok=True)

for idx, exam in enumerate(exam_subjects, start=1):
    output_file = os.path.join(output_dir, f"시험{idx}.json")
    exam_data = aggregated_data.get(exam, [])
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump({exam: exam_data}, outfile, ensure_ascii=False, indent=4)
        print(f"[저장 완료] 시험 과목 '{exam}' 집계 결과가 저장되었습니다: {output_file}")
    except Exception as e:
        print(f"Error saving file {output_file}: {e}")
