import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径
from sklearn.metrics import accuracy_score

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'stacking_clf.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("Predicting liver metastasis in gastric cancer using Stacking machine learning model")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
st.sidebar.subheader("Picking up parameters")
Age = st.sidebar.slider("Age", min_value=0, max_value=4, value=4, step=1)
Sex = st.sidebar.slider("Sex", min_value=0, max_value=1, value=1, step=1)
Primary_site = st.sidebar.slider("Primary site", min_value=0, max_value=8, value=8, step=1)
Histological_type = st.sidebar.slider("Histological type", min_value=0, max_value=2, value=0, step=1)
T_Stage = st.sidebar.slider("T Stage", min_value=0, max_value=4, value=4, step=1)
N_Stage = st.sidebar.slider("N Stage", min_value=0, max_value=4, value=0, step=1)
Surgery_status = st.sidebar.slider("Surgery status", min_value=0, max_value=2, value=0, step=1)
Radiation_status = st.sidebar.slider("Radiation status", min_value=0, max_value=2, value=2, step=1)
Chemotherapy_status = st.sidebar.slider("Chemotherapy status", min_value=0, max_value=1, value=1, step=1)
Extrahepatic_metastasis = st.sidebar.slider("Extrahepatic metastasis", min_value=0, max_value=0, value=1, step=1)
Tumor_size = st.sidebar.slider("Tumor size", min_value=0, max_value=5, value=1, step=1)

# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    'Age': [Age],
    'Sex': [Sex],
    'Primary site': [Primary_site],
    'Histological type': [Histological_type],
    'T Stage': [T_Stage],
    'N Stage': [N_Stage],
    'Surgery status': [Surgery_status],
    'Radiation status': [Radiation_status],
    'Chemotherapy status': [Chemotherapy_status],
    'Extrahepatic metastasis': [Extrahepatic_metastasis],
    'Tumor size': [Tumor_size]
})

if st.button("Predict"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"Predicted probability of liver metastasis: {final_pred_proba:.2f}%")

