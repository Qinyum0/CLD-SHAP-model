def generate_shap_plot(model, input_data, feature_names):
    """生成SHAP条形图的函数（更稳定）"""
    try:
        # 创建解释器
        explainer = shap.TreeExplainer(model)
        
        # 确保输入数据是数值类型
        input_data_numeric = input_data.astype(float)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_data_numeric)
        
        # 创建条形图显示特征重要性
        plt.figure(figsize=(10, 6))
        
        # 使用SHAP条形图
        shap.summary_plot(shap_values, input_data_numeric, feature_names=feature_names, 
                         plot_type="bar", show=False)
        
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        st.error(f"SHAP plot generation error: {str(e)}")
        return None
