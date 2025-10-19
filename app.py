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
        model_type = "native"
    # 方法2: 如果已经有.pkl文件，尝试用joblib加载但忽略警告
    elif os.path.exists("cld_model.pkl"):
        import joblib
        import warnings
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

class XGBoostWrapper:
    """包装XGBoost模型以兼容scikit-learn接口"""
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        # 确保输入数据是数值类型
        X = X.astype(float)
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
        # 确保所有列都是数值类型
        input_df = input_df.astype(float)
        
        # 检查模型类型并相应处理
        if model_type == "sklearn":
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

def generate_shap_bar_plot(model, input_data, feature_names):
    """生成SHAP条形图的函数"""
    try:
        # 确保输入数据是数值类型
        input_data = input_data.astype(float)
        
        # 创建SHAP解释器
        if model_type == "sklearn":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
        else:
            explainer = shap.TreeExplainer(model)
            dmatrix = xgb.DMatrix(input_data)
            shap_values = explainer.shap_values(dmatrix)
        
        # 计算每个特征的平均绝对SHAP值
        if len(shap_values.shape) > 2:
            # 多分类情况
            shap_abs = np.abs(shap_values[0]).mean(0)
        else:
            # 二分类情况
            shap_abs = np.abs(shap_values).mean(0)
        
        # 创建条形图
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        
        # 创建水平条形图
        plt.barh(y_pos, shap_abs)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('平均绝对SHAP值')
        plt.title('特征重要性 (SHAP值)')
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        st.error(f"SHAP条形图生成错误: {str(e)}")
        return generate_feature_importance_plot(model, feature_names)

def generate_feature_importance_plot(model, feature_names):
    """生成特征重要性图作为SHAP的备选方案"""
    try:
        plt.figure(figsize=(10, 6))
        
        if model_type == "sklearn":
            # 对于scikit-learn接口的模型
            importances = model.feature_importances_
        else:
            # 对于原生XGBoost模型
            score_dict = model.get_score(importance_type='weight')
            # 确保所有特征都有重要性值
            importances = np.zeros(len(feature_names))
            for i, feature in enumerate(feature_names):
                # 简化特征名匹配
                short_name = feature.split()[0].lower()  # 取第一个词并小写
                for key in score_dict:
                    if short_name in key.lower():
                        importances[i] = score_dict[key]
                        break
        
        # 创建条形图
        indices = np.argsort(importances)
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title('特征重要性图')
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        st.error(f"特征重要性图错误: {str(e)}")
        # 返回一个简单的错误图
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, '无法生成特征重要性图', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=16)
        plt.tight_layout()
        return plt.gcf()

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
            'age': float(age),
            'gender': 0.0 if gender == 'Female' else 1.0,
            'residence': 0.0 if residence == 'Urban' else 1.0,
            'waist': float(waist)
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
            
            # 模型解释部分
            st.subheader('Model Explanation')
            
            # 生成SHAP条形图
            feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
            
            shap_plot = generate_shap_bar_plot(best_xgb_model, input_df, feature_names)
            
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                SHAP条形图显示了每个特征对模型预测的平均绝对影响。条形越长表示该特征对预测结果的影响越大。
                """)

if __name__ == '__main__':
    main()
