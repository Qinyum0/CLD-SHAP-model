#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np

# 页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction",
    layout="wide"
)

# 加载模型
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("cld_model.json")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# 预测函数
def predict(model, data):
    input_df = pd.DataFrame([data])
    dmatrix = xgb.DMatrix(input_df)
    prediction = model.predict(dmatrix)[0]
    return prediction

# 主界面
st.title("Sarcopenia Risk Prediction")

# 输入表单
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 45, 100, 65)
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    residence = st.selectbox("Residence", ["Urban", "Rural"])
    waist = st.slider("Waist Circumference (cm)", 50, 150, 80)

if st.button("Predict Risk"):
    # 准备数据
    patient_data = {
        'age': age,
        'gender': 0 if gender == "Female" else 1,
        'residence': 0 if residence == "Urban" else 1, 
        'waist': waist
    }
    
    # 预测
    risk_score = predict(model, patient_data)
    risk_percentage = risk_score * 100
    
    # 显示结果
    st.subheader("Prediction Result")
    
    if risk_score > 0.5:
        st.error(f"High Risk of Sarcopenia: {risk_percentage:.1f}%")
    else:
        st.success(f"Low Risk of Sarcopenia: {risk_percentage:.1f}%")
    
    st.progress(float(risk_score))
