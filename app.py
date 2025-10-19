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

# 加载预训练模型 - 使用正确的方式
try:
    # 尝试不同的加载方式
    best_xgb_model = joblib.load("cld_model.pkl")
    
    # 如果是XGBoost模型，尝试使用save_model重新保存
    if hasattr(best_xgb_model, 'save_model'):
        # 这是一个临时解决方案
        best_xgb_model.save_model("temp_model.json")
        best_xgb_model = xgb.Booster()
        best_xgb_model.load_model("temp_model.json")
        
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        
        # 根据模型类型进行预测
        if hasattr(best_xgb_model, 'predict_proba'):
            # scikit-learn接口的模型
            proba = best_xgb_model.predict_proba(input_df)[0]
            prediction = best_xgb_model.predict(input_df)[0]
        else:
            # 原生XGBoost模型
            dmatrix = xgb.DMatrix(input_df)
            proba = best_xgb_model.predict(dmatrix)[0]
            prediction = 1 if proba > 0.5 else 0
            # 将输出转换为二分类概率格式
            proba = [1-proba, proba] if prediction == 1 else [proba, 1-proba]
            
        return prediction, proba, input_df
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(model, input_data, feature_names):
    """简化的SHAP图生成函数"""
    try:
        # 创建解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer(input_data)
        
        # 使用waterfall图作为替代
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        # 如果SHAP仍然不工作，显示特征重要性图
        st.warning(f"SHAP force plot not available: {str(e)}. Showing feature importance instead.")
        
        plt.figure(figsize=(10, 6))
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # 对于原生XGBoost
            importances = model.get_score(importance_type='weight')
            importances = [importances.get(f'f{i}', 0) for i in range(len(feature_names))]
        
        indices = np.argsort(importances)[::-1]
        
        plt.barh(range(len(feature_names)), [importances[i] for i in indices])
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
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
                st.info("SHAP plot is not available for the current model configuration.")

if __name__ == '__main__':
    main()

