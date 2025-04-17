import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

#변수 제작

tab1, tab2, tab3 = st.tabs(["말투 분석 프로그램📈", "소개글", "프로그램 제작 코드"])

# ==========================
# 데이터 및 모델 준비
# ==========================


@st.cache_resource
def load_and_train_model():
    # 데이터 불러오기
    data = pd.read_csv("말투파일.csv")
    texts = data['text']
    labels = data['label']

    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # 🔄 TfidfVectorizer로 교체 + ngram_range 추가
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=5000)
    X = vectorizer.fit_transform(texts)

    # 모델 학습
    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer, label_encoder

# 모델 불러오기
model, vectorizer, label_encoder = load_and_train_model()

# ==========================
# Streamlit UI
# ==========================

with st.sidebar:
    st.title("2025 또진탐 프로젝트")
    st.write("사이트 제작: 이콤시팀")
    with tab1:
        
        st.title("💬 채팅 말투 분석기🔍")
        st.write("문장을 입력하면 해당 말투 스타일을 예측합니다!\n\n")

        user_input = st.text_input("채팅 문장을 입력해보세요:")

        if user_input:
            X_input = vectorizer.transform([user_input])
            pred = model.predict(X_input)
            pred_label = label_encoder.inverse_transform(pred)
            st.success(f"예측된 말투: **{pred_label[0]}**")
            
    with tab2:
        st.write("아무것도 없지롱")
    with tab3:
        st.write("아무것도 없지롱")