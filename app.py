#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型和SHAP解释器
try:
    # 确保model.pkl文件存在于同一目录
    best_xgb_model = joblib.load("cld_model.pkl")  
    
    # 初始化SHAP解释器
    explainer = shap.Explainer(best_xgb_model)
    shap.initjs()  # 初始化JavaScript可视化
    
except Exception as e:
    st.error(f"Failed to load model or SHAP explainer: {str(e)}")
    st.stop()  # 如果加载失败则停止应用

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

def generate_shap_plot(input_df):
    """生成SHAP解释图"""
    try:
        # 计算SHAP值
        shap_values = explainer(input_df)
        
        # 创建图表
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        
        return fig
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
    waist = st.sidebar.slider('Waist Circumference', 15, 150, 60)
    
    if st.sidebar.button('Predict'):
        patient_data = {
            'age': age,
            'gender': 0 if gender == 'Female' else 1,
            'residence': 0 if residence == 'Urban' else 1,
            'waist': waist
        }
        
        prediction, proba, input_df = predict_prevalence(patient_data)
        
        if prediction is not None:
            # 创建两列布局
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Prediction Results')
                if prediction == 1:
                    st.error(f'High Risk: Sarcopenia probability {proba[1]*100:.2f}%')
                else:
                    st.success(f'Low Risk: Sarcopenia probability {proba[0]*100:.2f}%')
                
                st.progress(float(proba[1]))
                st.write(f'Low Risk: {float(proba[0])*100:.2f}% | High Risk: {float(proba[1])*100:.2f}%')
            
            with col2:
                st.subheader('SHAP Explanation')
                shap_fig = generate_shap_plot(input_df)
                if shap_fig:
                    st.pyplot(shap_fig)
                    st.caption("""
                    SHAP (SHapley Additive exPlanations) shows how each feature contributes to the prediction. 
                    Features pushing the prediction higher (red) increase the risk, while those pushing lower (blue) decrease the risk.
                    """)

if __name__ == '__main__':
    main()


# In[2]:





# In[ ]:




