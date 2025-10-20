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

def generate_shap_plot(model, input_data, feature_names):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        
        # 生成HTML格式的force plot
        force_plot = shap.plots.force(shap_values[0])
        
        # 保存为HTML文件并在Streamlit中显示
        shap.save_html("shap_plot.html", force_plot)
        
        # 在Streamlit中显示HTML
        with open("shap_plot.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=400)
        
        return None  # 因为我们已经直接显示了HTML
    except Exception as e:
        st.error(f"SHAP plot error: {str(e)}")
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
            **SHAP force plot** shows how each feature contributes to pushing the prediction 
            from the base value (average model output) to the final prediction. 
            **Red features** increase the risk, while **blue features** decrease it.
            """)
            
            # 定义特征名称（用于显示）
            feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
            
            # 生成并显示SHAP图
            shap_fig = generate_shap_plot(best_xgb_model, input_df, feature_names)
            if shap_fig:
                st.pyplot(shap_fig)
                
                # 添加详细解释
                st.markdown("""
                ### How to interpret this plot:
                - **Base value**: The average prediction of the model (starting point)
                - **Final prediction**: The model's output for this specific patient
                - **Red bars**: Features that increase the risk of sarcopenia
                - **Blue bars**: Features that decrease the risk of sarcopenia
                - **Length of bars**: Magnitude of the feature's contribution
                """)
            else:
                st.warning("SHAP plot could not be generated, but prediction was successful.")
                
                # 显示特征重要性解释
                st.subheader('Feature Impact Summary')
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Risk Increasing Factors**")
                    st.markdown("""
                    - Higher age
                    - Male gender  
                    - Rural residence
                    - Larger waist circumference
                    """)
                    
                with col2:
                    st.markdown("**Risk Decreasing Factors**")
                    st.markdown("""
                    - Younger age
                    - Female gender
                    - Urban residence  
                    - Smaller waist circumference
                    """)

if __name__ == '__main__':
    main()

