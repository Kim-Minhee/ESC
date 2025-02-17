import io, os
from dotenv import load_dotenv
from PIL import Image

import streamlit as st
import google.generativeai as genai
# import google import genai


### Gemini ###
# 모델 로드
@st.cache_resource
def load_gemini():
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-1.5-pro')
    return model

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []



### Streamlit ###
## 문진표
def create_section_title(title):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write(title)
    with col2:
        st.markdown('<hr style="margin: 12px 0px">', unsafe_allow_html=True)

def create_medical_form():
    st.header('문진표')
    with st.container(border=False, height=450):
        create_section_title('기본')
        # 성별, 나이
        col_gender, col_age = st.columns(2)
        with col_gender:
            gender = st.radio('**성별**', ['남', '여'], horizontal=True)
        with col_age:
            age = st.number_input('**나이 (만)**', value=30, min_value=0, max_value=150, step=1)
        # 신장, 체중
        col_height, col_weight = st.columns(2)
        with col_height:
            height = st.number_input('**신장 (cm)**', value=170.0, min_value=0.0, max_value=300.0, step=0.1, format='%.1f')
        with col_weight:
            weight = st.number_input('**체중 (kg)**', value=70.0, min_value=0.0, max_value=500.0, step=0.1, format='%.1f')
        
        create_section_title('질환력')
        # 본인
        conditions_self = st.multiselect('**본인**',
            ['뇌졸중', '심근경색', '고혈압', '당뇨병', '폐질환', '기타(암 포함)'],
            default=[])
        # 가족
        conditions_family = st.multiselect('**가족** (부모, 형제, 자매)',
            ['뇌졸중', '심근경색', '고혈압', '당뇨병', '기타(암 포함)'],
            default=[])
        
        create_section_title('생활')
        # 흡연
        with st.container(border=True):
            smoking_status = st.radio('**흡연**', ['안 함', '함'], horizontal=True, key='smoking')
            if smoking_status=='함':
                st.markdown('<hr style="margin: -5px 0px">', unsafe_allow_html=True)
                smoking_current = st.radio('현재 흡연 여부', ['현재도 흡연 중', '현재는 끊음'])
                smoking_years = st.number_input('흡연 기간 (년)', value=3, min_value=0, max_value=150, step=1)
                smoking_amount = st.number_input('1일 흡연량 (개비)', value=10, min_value=0, max_value=100, step=1)
        # 음주
        with st.container(border=True):
            drinking_status = st.radio('**음주**', ['안 함', '함'], horizontal=True, key='drinking')
            if drinking_status=='함':
                st.markdown('<hr style="margin: -5px 0px">', unsafe_allow_html=True)
                drinking_freq = st.number_input('한 달 평균 (회)', value=3, min_value=0, max_value=100, step=1)
                drinking_type = st.selectbox('주종을 선택하세요.', ['소주', '맥주', '양주', '막걸리', '와인'])
                drinking_amount = st.number_input('1회 음주량 (잔)', value=3, min_value=0, max_value=100, step=1)
        # 운동
        with st.container(border=True):
            exercise_status = st.radio('**운동**', ['안 함', '함'], horizontal=True, key='exercise')
            if exercise_status=='함':
                exercise_freq = st.number_input('일주일 평균 (회)', value=3, min_value=0, max_value=20, step=1)
                exercise_amount = st.number_input('1일 운동량 (분)', value=30, min_value=0, max_value=1000, step=1)

    return {
        'gender': gender,
        'age': age,
        'height': height,
        'weight': weight,
        'conditions_self': conditions_self,
        'conditions_family': conditions_family,
        'smoking': {
            'status': smoking_status,
            'details': {
                'current': smoking_current if smoking_status=='함' else None,
                'years': smoking_years if smoking_status=='함' else None,
                'amount': smoking_amount if smoking_status=='함' else None
            } if smoking_status=='함' else None
        },
        'drinking': {
            'status': drinking_status,
            'details': {
                'frequency': drinking_freq if drinking_status=='함' else None,
                'type': drinking_type if drinking_status=='함' else None,
                'amount': drinking_amount if drinking_status=='함' else None
            } if drinking_status=='함' else None
        },
        'exercise': {
            'status': exercise_status,
            'details': {
                'frequency': exercise_freq if exercise_status=='함' else None,
                'amount': exercise_amount if exercise_status=='함' else None
            } if exercise_status=='함' else None
        }
    }

## 이미지 업로드
def create_image_upload():
    st.header('파일 업로드')
    uploaded_file = st.file_uploader('JPG 이미지를 업로드하세요.', type=['jpg'])

## 메인
def main():
    st.set_page_config(
        layout = 'wide',
        page_title='MEGA',
        #page_icon='', # 윈도우 창의 탭 아이콘 설정
    )

    # 사이드바
    with st.sidebar:
        # 문진표
        medical_form_data = create_medical_form()

        # 이미지 업로드
        create_image_upload()
    
    # 메인
    st.header('진단 보조 챗봇', divider='gray') # divider 옵션: blue, green, orange, red, violet, gray, grey, rainbow
    st.title('MEGA')
    
    # 소개/설명
    with st.container(border=True):
        st.write('''
        👋 안녕하세요! 진단 보조 챗봇 **메가**입니다.

        왼쪽 사이드바에서 문진표 작성과 이미지 파일을 업로드해주시면 초진기록지 작성을 도와드리겠습니다!

        💡 **사용 방법**
        1. 왼쪽 사이드바에서 간단한 **문진표**를 작성해주세요.
        2. 진단이 필요한 **의료 이미지 파일**(.jpg)을 업로드해주세요.
        3. 업로드가 완료되면 **초진기록지 초안**을 작성해드립니다.
        ''')
    if st.button('입력 데이터 확인'):
        st.write(medical_form_data)
    if uploaded_file:
        st.write(uploaded_file)

    ## 챗봇 ##
    init_session_state()
    gemini = load_gemini()

    # 메시지 기록
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # 채팅
    if prompt:=st.chat_input('메시지를 입력하세요.'):
        st.chat_message('user').markdown(prompt) # 사용자 메시지 출력
        try: # gemini 응답
            response = gemini.generate_content([prompt])
            with st.chat_message('assistant'):
                st.markdown(response.text) # gemini 답변 출력
            st.session_state.messages.append({'role':'user', 'content':prompt}) # 대화 기록 저장
            st.session_state.messages.append({'role':'assistant', 'content':response.text}) # 대화 기록 저장
        except Exception as e:
            st.error(f'Gemini 응답 오류 발생: {str(e)}')


if __name__=='__main__':
    main()