#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np

# 必须在所有Streamlit命令之前设置页面配置
st.set_page_config(
    page_title="Sarcopenia Risk Prediction in CLD Patients",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载预训练模型 - 使用XGBoost原生格式
try:
    # 加载XGBoost模型
    best_xgb_model = xgb.Booster()
    best_xgb_model.load_model("cld_model.json")
    
    # 为了使用predict方法，我们需要创建一个XGBClassifier包装器
    class XGBWrapper:
        def __init__(self, booster):
            self.booster_ = booster
            self.classes_ = np.array([0, 1])  # 假设是二分类
            
        def predict_proba(self, X):
            """预测概率"""
            if isinstance(X, pd.DataFrame):
                X = xgb.DMatrix(X)
            elif not isinstance(X, xgb.DMatrix):
                X = xgb.DMatrix(X)
            
            predictions = self.booster_.predict(X)
            # 如果predictions是二维的，直接返回
            if len(predictions.shape) == 2:
                return predictions
            # 如果是一维的（二元分类），转换为二维概率
            else:
                proba_1 = predictions
                proba_0 = 1 - proba_1
                return np.column_stack([proba_0, proba_1])
            
        def predict(self, X):
            """预测类别"""
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def predict_prevalence(patient_data):
    """使用预训练模型进行预测"""
    try:
        input_df = pd.DataFrame([patient_data])
        # 使用包装器进行预测
        wrapper = XGBWrapper(best_xgb_model)
        proba = wrapper.predict_proba(input_df)[0]
        prediction = wrapper.predict(input_df)[0]
        return prediction, proba, input_df
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def generate_shap_plot(model, input_data, feature_names):
    """生成SHAP力图的函数"""
    try:
        # 创建SHAP解释器 - 使用底层的booster
        explainer = shap.TreeExplainer(model.booster_)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_data)
        
        # 创建图表
        plt.figure(figsize=(10, 4))
        
        # 生成SHAP力图
        shap.force_plot(
            explainer.expected_value, 
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
    This tool is used to predict the risk of sarcopenia in patients with chronic lung disease (CLD).
    Please adjust the parameters in the sidebar and click 'Predict' to see the results.
    """)
    
    # 侧边栏输入
    st.sidebar.header('Patient Parameters')
    age = st.sidebar.slider('Age', 45, 100, 50)
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    residence = st.sidebar.selectbox('Residence', ['Urban', 'Rural'])
    waist = st.sidebar.slider('Waist Circumference (cm)', 15, 150, 60)

    if st.sidebar.button('Predict'):
        # 准备患者数据
        patient_data = {
            'age': age,
            'gender': 0 if gender == 'Female' else 1,
            'residence': 0 if residence == 'Urban' else 1,
            'waist': waist
        }
        
        # 进行预测
        prediction, proba, input_df = predict_prevalence(patient_data)
        
        if prediction is not None:
            # 预测结果部分
            st.subheader('Prediction Results')
            
            # 显示风险等级和概率
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error(f'**High Risk**')
                else:
                    st.success(f'**Low Risk**')
            
            with col2:
                st.metric(
                    label="Sarcopenia Probability", 
                    value=f"{proba[1]*100:.1f}%"
                )
            
            # 进度条显示风险概率
            st.progress(float(proba[1]))
            st.write(f'**Probability Breakdown:** Low Risk: {proba[0]*100:.2f}% | High Risk: {proba[1]*100:.2f}%')
            
            # SHAP解释部分
            st.subheader('Feature Impact Analysis')
            st.write("The SHAP plot below shows how each feature contributes to the prediction:")
            
            feature_names = ['Age', 'Gender', 'Residence', 'Waist Circumference']
            shap_plot = generate_shap_plot(XGBWrapper(best_xgb_model), input_df, feature_names)
            
            if shap_plot:
                st.pyplot(shap_plot)
                st.caption("""
                **Interpretation guide:**
                - **Red features** push the prediction towards higher risk
                - **Blue features** push the prediction towards lower risk  
                - The **base value** is the average model prediction
                - The **output value** is the final prediction for this patient
                """)
            
            # 特征解释文本
            st.subheader('Key Insights')
            st.write("Based on the patient's characteristics:")
            
            insights = []
            if age > 65:
                insights.append(f"• Age ({age} years) increases sarcopenia risk")
            else:
                insights.append(f"• Age ({age} years) decreases sarcopenia risk")
                
            if gender == 'Male':
                insights.append("• Male gender increases sarcopenia risk")
            else:
                insights.append("• Female gender decreases sarcopenia risk")
                
            if waist < 80:
                insights.append(f"• Waist circumference ({waist} cm) may indicate lower muscle mass")
            else:
                insights.append(f"• Waist circumference ({waist} cm) may indicate adequate muscle mass")
            
            for insight in insights:
                st.write(insight)

if __name__ == '__main__':
    main()
