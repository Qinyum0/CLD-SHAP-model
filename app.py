#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型
try:
    # 方法1: 使用XGBoost原生方法加载模型
    if os.path.exists("cld_model.json"):
        best_xgb_model = xgb.Booster()
        best_xgb_model.load_model("cld_model.json")
    # 方法2: 如果已经有.pkl文件，尝试用joblib加载但忽略警告
    elif os.path.exists("cld_model.pkl"):
        import joblib
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            best_xgb_model = joblib.load("cld_model.pkl")
    else:
        st.error("Model file not found. Please ensure either 'cld_model.json' or 'cld_model.pkl' exists.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

class XGBoostWrapper:
    """包装XGBoost模型以兼容scikit-learn接口"""
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        # 将DataFrame转换为DMatrix
        dmatrix = xgb.DMatrix(X)
        # 获取预测概率
        proba = self.model.predict(dmatrix)
        # 对于二分类，返回形状为(n_samples, 2)的概率数组
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        
        # 检查模型类型并相应处理
        if hasattr(best_xgb_model, 'predict_proba'):
            # 如果是scikit-learn接口的模型
            proba = best_xgb_model.predict_proba(input_df)[0]
            prediction = best_xgb_model.predict(input_df)[0]
        else:
            # 如果是原生XGBoost模型，使用包装器
            wrapper = XGBoostWrapper(best_xgb_model)
            proba = wrapper.predict_proba(input_df)[0]
            prediction = wrapper.predict(input_df)[0]
            
        return prediction, proba, input_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(model, input_data, feature_names):
    """生成SHAP力图的函数"""
    try:
        # 检查模型类型并创建相应的解释器
        if hasattr(model, 'predict_proba'):
            # scikit-learn接口的模型
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            expected_value = explainer.expected_value
        else:
            # 原生XGBoost模型
            explainer = shap.TreeExplainer(model)
            dmatrix = xgb.DMatrix(input_data)
            shap_values = explainer.shap_values(dmatrix)
            expected_value = explainer.expected_value
        
        # 创建图表
        plt.figure(figsize=(10, 4))
        
        # 生成SHAP力图
        shap.force_plot(
            expected_value, 
            shap_values[0], 
            input_data.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
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
            st.subheader('SHAP Force Plot')
            
            # 生成SHAP力图
            feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
            
            # 检查模型类型并传递正确的模型对象
            if hasattr(best_xgb_model, 'predict_proba'):
                shap_model = best_xgb_model
            else:
                shap_model = XGBoostWrapper(best_xgb_model)
                
            shap_plot = generate_shap_plot(shap_model, input_df, feature_names)
            
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                SHAP force plot shows how each feature contributes to pushing the prediction 
                from the base value (average model output) to the final prediction. 
                Red features increase the risk, while blue features decrease it.
                """)

if __name__ == '__main__':
    main()
