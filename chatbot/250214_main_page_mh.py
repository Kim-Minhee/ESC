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
def main():
    st.set_page_config(
        layout = 'wide',
        page_title='MEGA',
        #page_icon='', # 윈도우 창의 탭 아이콘 설정
    )

    ## 사이드바 ##
    with st.sidebar:
        # 문진표
        st.header('문진표')
        check_list = {
            '항목1': st.checkbox('남자'),
            '항목2': st.checkbox('여자'),
            '항목3': st.checkbox('흡연가'),
            '항목4': st.checkbox('음주가'),
            '항목5': st.checkbox('가족력'),
        }

        # 업로드
        st.header('파일 업로드')
        uploaded_file = st.file_uploader('JPG 이미지를 업로드하세요.', type=['jpg'])
    
    ## 메인 ##
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