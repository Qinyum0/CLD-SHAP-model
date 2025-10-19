#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型
try:
    # 优先使用XGBoost原生格式
    if os.path.exists("cld_model.json"):
        best_xgb_model = xgb.Booster()
        best_xgb_model.load_model("cld_model.json")
        model_type = "native"
    # 备选方案：使用joblib加载
    elif os.path.exists("cld_model.pkl"):
        import joblib
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            best_xgb_model = joblib.load("cld_model.pkl")
        model_type = "sklearn"
    else:
        st.error("Model file not found. Please ensure either 'cld_model.json' or 'cld_model.pkl' exists.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        
        if model_type == "sklearn":
            # scikit-learn接口的模型
            proba = best_xgb_model.predict_proba(input_df)[0]
            prediction = best_xgb_model.predict(input_df)[0]
        else:
            # 原生XGBoost模型
            dmatrix = xgb.DMatrix(input_df)
            raw_pred = best_xgb_model.predict(dmatrix)[0]
            # 将原始预测转换为概率
            proba = [1 - raw_pred, raw_pred]
            prediction = 1 if raw_pred > 0.5 else 0
            
        return prediction, proba, input_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(input_data, feature_names):
    """生成SHAP力图的函数"""
    try:
        plt.figure(figsize=(10, 4))
        
        if model_type == "sklearn":
            # 对于scikit-learn接口的模型
            explainer = shap.TreeExplainer(best_xgb_model)
            shap_values = explainer.shap_values(input_data)
            expected_value = explainer.expected_value
            
            # 生成SHAP力图
            shap.force_plot(
                expected_value, 
                shap_values[0], 
                input_data.iloc[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        else:
            # 对于原生XGBoost模型
            explainer = shap.TreeExplainer(best_xgb_model)
            dmatrix = xgb.DMatrix(input_data)
            shap_values = explainer.shap_values(dmatrix)
            
            # 对于原生模型，SHAP值可能是一维数组
            if isinstance(shap_values, list):
                shap_val = shap_values[0]
            else:
                shap_val = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # 生成SHAP力图
            shap.force_plot(
                explainer.expected_value, 
                shap_val, 
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
# 在SHAP部分添加备选方案
try:
    # 尝试生成SHAP图
    feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
    shap_plot = generate_shap_plot(input_df, feature_names)
    
    if shap_plot:
        st.pyplot(shap_plot)
    else:
        # 备选：显示特征重要性
        if model_type == "sklearn":
            importances = best_xgb_model.feature_importances_
        else:
            importances = best_xgb_model.get_score(importance_type='weight')
            
        fig, ax = plt.subplots()
        ax.barh(feature_names, importances)
        ax.set_xlabel('Feature Importance')
        st.pyplot(fig)
        st.info("Showing feature importance as SHAP visualization is not available.")
        
except Exception as e:
    st.error(f"Visualization error: {str(e)}")

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
            shap_plot = generate_shap_plot(input_df, feature_names)
            
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                SHAP force plot shows how each feature contributes to pushing the prediction 
                from the base value (average model output) to the final prediction. 
                Red features increase the risk, while blue features decrease it.
                """)
            else:
                st.info("SHAP visualization is not available for the current model type.")

if __name__ == '__main__':
    main()

