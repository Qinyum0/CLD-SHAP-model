#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator
import shap
import matplotlib.pyplot as plt
import numpy as np

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型
try:
    best_xgb_model = joblib.load("cld_model.pkl")  
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        proba = best_xgb_model.predict_proba(input_df)[0]
        prediction = best_xgb_model.predict(input_df)[0]
        return prediction, proba, input_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(model, input_data, feature_names):
    """生成SHAP力图的函数 - 使用matplotlib版本"""
    try:
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_data)
        
        # 使用matplotlib创建force plot
        plt.figure(figsize=(10, 3))
        
        # 生成matplotlib版本的force plot
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=input_data.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=0  # 避免文本旋转问题
        )
        
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        st.error(f"SHAP plot generation error: {str(e)}")
        # 如果force plot失败，尝试使用其他SHAP图
        return generate_alternative_shap_plot(model, input_data, feature_names)

def generate_alternative_shap_plot(model, input_data, feature_names):
    """生成替代的SHAP图"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        # 使用waterfall图
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        st.error(f"Alternative SHAP plot also failed: {str(e)}")
        return None

def main():
    st.title('Sarcopenia Risk Prediction in CLD Patients')
    st.markdown("""
    This tool is used to predict the risk of sarcopenia in patients with chronic lung disease(CLD).
    """)
    
    # 侧边栏输入
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
            # 预测结果部分
            st.subheader('Prediction Results')
            
            if prediction == 1:
                st.error(f'High Risk: Sarcopenia probability {proba[1]*100:.2f}%')
            else:
                st.success(f'Low Risk: Sarcopenia probability {proba[0]*100:.2f}%')
            
            st.progress(float(proba[1]))
            st.write(f'Low Risk: {float(proba[0])*100:.2f}% | High Risk: {float(proba[1])*100:.2f}%')
            
            # SHAP解释部分
            st.subheader('Model Interpretation')
            
            # 生成SHAP力图
            feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
            shap_plot = generate_shap_plot(best_xgb_model, input_df, feature_names)
            
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                SHAP force plot shows how each feature contributes to pushing the prediction 
                from the base value (average model output) to the final prediction. 
                Red features increase the risk, while blue features decrease it.
                """)
            else:
                st.warning("SHAP visualization is not available. This might be due to model compatibility issues.")

if __name__ == '__main__':
    main()
