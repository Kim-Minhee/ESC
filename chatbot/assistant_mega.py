import os
from dotenv import load_dotenv

import streamlit as st
import google.generativeai as genai
# import google import genai

@st.cache_resource
def load_gemini():
    load_dotenv()
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-1.5-pro')

    return model

def generate_medical_record(llm_model, form_data, prob, diagnosis_result):
    # 환자 기본 정보 추출
    gender = form_data.get('gender', '정보 없음')
    age = form_data.get('age', '정보 없음')
    height = form_data.get('height', '정보 없음')
    weight = form_data.get('weight', '정보 없음')
    
    # 병력 정보 추출
    conditions_self = form_data.get('conditions_self', [])
    conditions_family = form_data.get('conditions_family', [])
    
    # 생활 습관 정보 추출
    smoking_info = form_data.get('smoking', {})
    drinking_info = form_data.get('drinking', {})
    exercise_info = form_data.get('exercise', {})
    
    # 현재병력 구성
    current_condition = "갑상선 종괴 의심으로 내원"
    if conditions_self:
        current_condition += f". 환자는 {', '.join(conditions_self)}의 병력이 있음"
    
    # 흡연 정보 구성
    smoking_status = smoking_info.get('status', '정보 없음')
    smoking_details = ""
    if smoking_status=='함':
        smoking_details = f"({smoking_info.get('details', {}).get('years', '?')}년간 하루 {smoking_info.get('details', {}).get('amount', '?')}개비)"
    
    # 음주 정보 구성
    drinking_status = drinking_info.get('status', '정보 없음')
    drinking_details = ""
    if drinking_status=='함':
        drinking_details = f"(주 {drinking_info.get('details', {}).get('frequency', '?')}회, {drinking_info.get('details', {}).get('type', '?')} {drinking_info.get('details', {}).get('amount', '?')}잔)"
    
    # 운동 정보 구성
    exercise_status = exercise_info.get('status', '정보 없음')
    exercise_details = ""
    if exercise_status=='함':
        exercise_details = f"(주 {exercise_info.get('details', {}).get('frequency', '?')}회, {exercise_info.get('details', {}).get('amount', '?')}분)"

    # Gemini 프롬프트 작성
    prompt = f"""
    당신은 대한민국 최고의 갑상선 전문의입니다.
    환자의 기본 정보와 갑상선 초음파 이미지를 참고하여 초진기록지를 800자 이내로 작성해주세요.

    다음은 환자의 정보와 갑상선 초음파 이미지 진단 결과입니다:
    
    환자 정보:
    - 성별: {gender}
    - 나이: {age}세
    - 키: {height}cm
    - 체중: {weight}kg
    - BMI: {round(weight / ((height/100)**2), 1)}
    
    현재병력: {current_condition}
    
    과거병력: {', '.join(conditions_self) if conditions_self else '특이사항 없음'}
    
    가족력: {', '.join(conditions_family) if conditions_family else '특이사항 없음'}
    
    사회력:
    - 흡연: {smoking_status} {smoking_details}
    - 음주: {drinking_status} {drinking_details}
    - 운동: {exercise_status} {exercise_details}
    
    갑상선 초음파 영상 AI 진단 결과:
    - 진단: {diagnosis_result}
    - 신뢰도: {prob}%
    
    이 정보를 바탕으로 갑상선 진료 초진기록지 초안을 작성해주세요. 다음 내용을 포함해야 합니다:
    1. 주증상
    2. 현재병력
    3. 사회력
    4. 과거병력
    5. 가족력
    6. 환자 정보
    7. 갑상선 초음파 영상 소견
    8. 감별진단
    9. 진단계획
    
    의학 전문용어를 적절히 사용하고, 간결하면서도 전문적인 문체로 작성해주세요.
    """
    
    # Gemini로 초진기록지 생성
    try:
        response = llm_model.generate_content(prompt)
        medical_record = response.text
    except Exception as e:
        medical_record = f"초진기록지 생성 중 오류가 발생했습니다: {str(e)}"
    
    return medical_record