import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'gbdt_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("A precise predictive model for centenarians from the China Healthy Longevity Multicenter Study (CHLMS)")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
st.sidebar.subheader("Picking up parameters")
Gender = st.sidebar.slider("Gender, n (%)", min_value=0, max_value=1, value=0, step=1)
Nation = st.sidebar.slider("Nation, n (%)", min_value=1, max_value=9, value=4, step=1)
Marital_Status = st.sidebar.slider("Marital status, n (%)", min_value=1, max_value=4, value=3, step=1)
Hypertension = st.sidebar.slider("Hypertension, n (%)", min_value=0, max_value=1, value=0, step=1)
Smoking = st.sidebar.slider("Smoking, n (%)", min_value=1, max_value=4, value=3, step=1)
Bedtime = st.sidebar.slider("Bedtime, n (%)", min_value=1, max_value=3, value=1, step=1)
Outdoor_exercise = st.sidebar.slider("Outdoor exercise, n (%)", min_value=0, max_value=1, value=1, step=1)
Play_cards = st.sidebar.slider("Play cards, n (%)", min_value=0, max_value=1, value=0, step=1)
Watching_TV = st.sidebar.slider("Watching TV, n (%)", min_value=0, max_value=1, value=0, step=1)
Feeding = st.sidebar.slider("Feeding, n (%)", min_value=0, max_value=1, value=0, step=1)
Rice = st.sidebar.slider("Rice intake, n (%)", min_value=0, max_value=1, value=1, step=1)
Hearing = st.sidebar.slider("Hearing, n (%)", min_value=1, max_value=3, value=2, step=1)
SBP = st.sidebar.slider("SBP, mmHg", min_value=38, max_value=230, value=148, step=1)
DBP = st.sidebar.slider("DBP, mmHg", min_value=38, max_value=218, value=78, step=1)
TC = st.sidebar.slider("TC, mmol/L", min_value=0.35, max_value=11.81, value=4.34, step=0.01)

# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    'Gender': [Gender],
    'Nation': [Nation],
    'Marital_Status': [Marital_Status],
    'Hypertension': [Hypertension],
    'Smoking': [Smoking],
    'Bedtime': [Bedtime],
    'Outdoor_exercise': [Outdoor_exercise],
    'Play_cards': [Play_cards],
    'Watching_TV': [Watching_TV],
    'Feeding': [Feeding],
    'Rice': [Rice],
    'Hearing': [Hearing],
    'SBP': [SBP],
    'DBP': [DBP],
    'TC': [TC]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"Predicted probability of liver metastasis: {final_pred_proba:.2f}%")
