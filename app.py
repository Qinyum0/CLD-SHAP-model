#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    # 确保model.pkl文件存在于同一目录
    best_xgb_model = joblib.load("cld_model.pkl")  
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()  # 如果模型加载失败则停止应用

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        # 确保输入字段与模型训练时完全一致
        proba = best_xgb_model.predict_proba(input_df)[0]
        prediction = best_xgb_model.predict(input_df)[0]
        return prediction, proba, input_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(model, input_data):
    """生成SHAP力图 - 直接使用matplotlib"""
    try:
        # 确保数据是数值类型
        input_data_numeric = input_data.astype(float)
        
        # 使用TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_data_numeric)
        
        # 创建matplotlib图形
        plt.figure(figsize=(10, 6))
        
        # 生成force plot
        shap.force_plot(
            explainer.expected_value,
            shap_values[0], 
            input_data_numeric.iloc[0],
            feature_names=['Age', 'Gender', 'Residence', 'Waist Circumference'],
            matplotlib=True,
            show=False
        )
        
        # 调整布局
        plt.tight_layout()
        
        # 返回图形对象
        return plt.gcf()
        
    except Exception as e:
        st.error(f"SHAP plot generation error: {str(e)}")
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
            st.subheader('Prediction Results')
            if prediction == 1:
                st.error(f'High Risk: Sarcopenia probability {proba[1]*100:.2f}%')
            else:
                st.success(f'Low Risk: Sarcopenia probability {proba[0]*100:.2f}%')
            
            st.progress(float(proba[1]))
            st.write(f'Low Risk: {float(proba[0])*100:.2f}% | High Risk: {float(proba[1])*100:.2f}%')
            
            # 添加SHAP解释
            st.subheader('Model Interpretation - SHAP Analysis')
            st.markdown("""
            SHAP force plot shows how each feature contributes to pushing the prediction 
            from the base value (average model output) to the final prediction. 
            Red features increase the risk, while blue features decrease it.
            """)
            
            # 生成并显示SHAP图
            shap_fig = generate_shap_plot(best_xgb_model, input_df)
            if shap_fig:
                st.pyplot(shap_fig)

if __name__ == '__main__':
    main()
