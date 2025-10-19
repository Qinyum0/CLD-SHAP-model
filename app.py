#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型
try:
    best_xgb_model = xgb.Booster()
    best_xgb_model.load_model("cld_model.json")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        dmatrix = xgb.DMatrix(input_df)
        proba = best_xgb_model.predict(dmatrix)[0]
        prediction = 1 if proba > 0.5 else 0
        return prediction, [1-proba, proba], input_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(model, input_data, feature_names):
    """生成SHAP条形图的函数"""
    try:
        explainer = shap.TreeExplainer(model)
        input_data_numeric = input_data.astype(float)
        shap_values = explainer.shap_values(input_data_numeric)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, input_data_numeric, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        st.error(f"SHAP plot generation error: {str(e)}")
        return None

def main():
    st.title('Sarcopenia Risk Prediction in CLD Patients')
    st.markdown("""
    This tool is used to predict the risk of sarcopenia in patients with chronic lung disease(CLD).
    """)
    
    st.sidebar.header('Patient Parameters')
    age = st.sidebar.slider('Age', 45, 100, 50)
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    residence = st.sidebar.selectbox('Residence', ['Urban', 'Rural'])
    waist = st.sidebar.slider('Waist Circumference (cm)', 15, 150, 60)

    if st.sidebar.button('Predict'):
        patient_data = {
            'age': age,
            'gender': 0 if gender == 'Female' else 1,
            'residence': 0 if residence == 'Urban' else 1,
            'waist': waist
        }
        
        prediction, proba, input_df = predict_prevalence(patient_data)
        
        if prediction is not None:
            st.subheader('Prediction Results')
            
            if prediction == 1:
                st.error(f'High Risk: Sarcopenia probability {proba[1]*100:.2f}%')
            else:
                st.success(f'Low Risk: Sarcopenia probability {proba[0]*100:.2f}%')
            
            st.progress(float(proba[1]))
            st.write(f'Low Risk: {float(proba[0])*100:.2f}% | High Risk: {float(proba[1])*100:.2f}%')
            
            st.subheader('Feature Importance (SHAP)')
            feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
            shap_plot = generate_shap_plot(best_xgb_model, input_df, feature_names)
            
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                SHAP bar plot shows the importance of each feature in the prediction.
                Higher absolute SHAP values indicate greater feature importance.
                """)

if __name__ == '__main__':
    main()
