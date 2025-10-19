#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import warnings
import os
import sys
import logging

# 最彻底的警告和日志抑制
os.environ['XGBOOST_SILENT'] = '1'
warnings.filterwarnings('ignore')

# 抑制所有日志
logging.getLogger('xgboost').setLevel(logging.ERROR)

# 重定向stderr来捕获XGBoost警告
class WarningFilter:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def write(self, message):
        if "WARNING" in message and "xgboost" in message:
            return  # 过滤掉XGBoost警告
        self.original_stderr.write(message)
        
    def flush(self):
        self.original_stderr.flush()

# 应用过滤器
sys.stderr = WarningFilter()

# 自动安装缺失包
try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    
# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)



# 加载预训练模型和SHAP解释器
try:
    # 使用上下文管理器完全抑制所有警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_xgb_model = joblib.load("cld_model.pkl")  
    
    # 使用TreeExplainer
    explainer = shap.TreeExplainer(best_xgb_model)
    
except Exception as e:
    st.error(f"加载失败: {str(e)}")
    st.stop()

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

def generate_shap_force_plot(input_df):
    """生成SHAP力图"""
    try:
        # 计算SHAP值
        shap_values = explainer.shap_values(input_df)
        
        # 创建SHAP力图
        plt.figure()
        shap.plots.force(shap_values[0], matplotlib=True, show=False)
        plt.tight_layout()
        
        return plt.gcf()
    except Exception as e:
        st.error(f"SHAP force plot generation error: {str(e)}")
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
            # 显示预测结果
            st.subheader('Prediction Results')
            if prediction == 1:
                st.error(f'High Risk: Sarcopenia probability {proba[1]*100:.2f}%')
            else:
                st.success(f'Low Risk: Sarcopenia probability {proba[0]*100:.2f}%')
            
            st.progress(float(proba[1]))
            st.write(f'Low Risk: {float(proba[0])*100:.2f}% | High Risk: {float(proba[1])*100:.2f}%')
            
            # 在结果下方显示SHAP力图
            st.subheader('SHAP Force Plot')
            shap_plot = generate_shap_force_plot(input_df)
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                SHAP force plot shows how each feature contributes to pushing the prediction 
                from the base value (average model output) to the final prediction. 
                Red features increase the risk, while blue features decrease it.
                """)

if __name__ == '__main__':
    main()
