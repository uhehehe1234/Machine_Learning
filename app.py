import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
import random as ran


# ================================
# 질문 목록
# ================================
questions_list = [
    "나 오늘 진짜 너무 열 받는 일 있었어ㅠㅠ",
    "학교가기 싫다;; 고3 언제 끝나냐ㅠㅠ",
    "나 OO이랑 친해지고 싶은데 말 걸어볼까?",
    "나 방금 길에서 자빠졌다;;",
    "수능 200일 깨졌다.... 어떡하냐...ㅋㅋ",
    "오늘 쌤한테 ㅈㄴ 깨졌다...ㅋㅋ",
    "오늘 하루 어땠냐?",
    "야야 OO아 (본인을 부르는 상황)"]

questions_extra_list = [
    "너에게 말할게 있어.. 사실 내가 초능력이 있어..",
    "혹시 너.. 은하계에서 온 외계인이야...?",
]

questions = []

# ================================
# 모델 및 리소스 불러오기
# ================================
@st.cache_resource
def load_resources():
    model = load_model("mal_tu_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_resources()

# ================================
# 세션 상태 초기화
# ================================
if "started" not in st.session_state:
    st.session_state.started = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "responses" not in st.session_state:
    st.session_state.responses = []
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "questions_model" not in st.session_state:
    st.session_state.questions_model = []

# ================================
# 타이틀 및 소개
# ================================
st.markdown("<h1 style='text-align: center;'>💬 채팅 말투 분석기</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>아래 질문에 답해주세요!</h4>", unsafe_allow_html=True)
st.markdown("---")

# ================================
# 시작 화면 → 분석 시작 버튼
# ================================
if not st.session_state.started:
    if st.button("🚀 분석 시작"):
        st.session_state.started = True
        numbers = [0,1,2,3,4,5,6,7]
        number_sample = ran.sample(numbers, 3)
        questions.append(questions_list[int(number_sample[0])])
        questions.append(questions_list[int(number_sample[1])])
        questions.append(questions_list[int(number_sample[2])])
        st.session_state.questions_model.append(questions[0])
        st.session_state.questions_model.append(questions[1])
        st.session_state.questions_model.append(questions[2])
        print(questions)
        st.rerun()
    st.stop()

# ================================
# 질문 및 응답 흐름
# ================================
total_steps = len(st.session_state.questions_model)
step = st.session_state.step
if step < total_steps:
    st.markdown(f"### 질문 {step + 1}: {st.session_state.questions_model[step]}")
    user_input = st.text_input("답변을 입력하세요:", key=f"input_{step}")

    if st.button("✅ 응답 완료"):
        if user_input.strip() != "":
            # 응답 저장
            st.session_state.responses.append(user_input)

            # 예측
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=30, padding='post')
            pred = model.predict(padded)
            st.session_state.predictions.append(pred[0])

            # 다음 질문으로 이동
            st.session_state.step += 1
            st.rerun()
        else:
            st.warning("답변을 입력해주세요.")
else:
    # ================================
    # 최종 결과 출력
    # ================================
    st.markdown("## 📊 분석 결과")
    for idx, question in enumerate(st.session_state.predictions):
        response = st.session_state.responses[idx]
        prediction = st.session_state.predictions[idx]
        label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction) * 100

        st.markdown(f"### {idx + 1}. {st.session_state.questions_model[idx]}")
        st.markdown(f"- ✍️ 당신의 답: `{response}`")
        st.markdown(f"- 🔎 예측된 말투 스타일: **{label}** ({confidence:.2f}%)")
        st.progress(int(confidence))
        st.markdown("---")

    # ================================
    # 테스트 재시작
    # ================================
    if st.button("🔄 다시 테스트하기"):
        for key in ["started", "step", "responses", "predictions","questions_model"]:
            del st.session_state[key]
            questions.clear()
        st.rerun()


# questions = ["오늘 기분 어때?",
#              "학교에서 선생님한테 혼났어ㅠㅠ 위로해줘ㅠㅠ",
#              "나 상장 받았다! 축하해줘!"]

# # tab1, tab2, tab3 = st.tabs(["말투 분석 프로그램📈", "소개글", "프로그램 제작 코드"])

# # ==========================
# # 데이터 및 모델 준비
# # ==========================

# @st.cache_resource
# def load_resources():
#     model = load_model("mal_tu_model.h5")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     with open("label_encoder.pkl", "rb") as f:
#         label_encoder = pickle.load(f)
#     return model, tokenizer, label_encoder

# model, tokenizer, label_encoder = load_resources()

# # ==========================
# # Streamlit UI
# # ==========================
# if "test_bool" not in st.session_state:
#     st.session_state.test_bool = False

# if st.session_state.test_bool == False:
#     for i in range(2): #줄바꿈
#         st.title(" ")
#     st.markdown("<h1 style='text-align: center;'>💬 채팅 말투 분석기🔍</h1>", unsafe_allow_html=True)
#     st.markdown("<h1 style='text-align: center; font-size: 24px;'>질문에 대한 답을 입력하세요!</h1>", unsafe_allow_html=True)
#     if st.button("-----------------------------------------------------<테스트 시작>---------------------------------------------------------------"):
#         st.session_state.test_bool = True

# if st.session_state.test_bool == True:

#     user_input = st.text_input("채팅 문장을 입력해보세요:")
#     if user_input:
#         seq = tokenizer.texts_to_sequences([user_input])
#         padded = pad_sequences(seq, maxlen=30, padding='post')
#         pred = model.predict(padded)
#         label = label_encoder.inverse_transform([np.argmax(pred)])
#         st.success(f"예측된 말투 스타일: **{label[0]}**")