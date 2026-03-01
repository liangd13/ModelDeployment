import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'trained_model.pkl')

AGE_OPTIONS = ['0-49 years', '50-59 years', '60-69 years', '70-79 years', '80+ years']
PRIMARY_SITE_OPTIONS = [
    'C16.0-Cardia, NOS', 
    'C16.1-Fundus of stomach',
    'C16.2-Body of stomach',
    'C16.3-Gastric antrum',
    'C16.4-Pylorus',
    'C16.5-Lesser curvature of stomach NOS',
    'C16.6-Greater curvature of stomach NOS',
    'C16.8-Overlapping lesion of stomach',
    'C16.9-Stomach, NOS'
]
HISTOLOGICAL_TYPE_OPTIONS = ['Adenocarcinoma', 'Signet ring cell carcinoma', 'spacial type']
T_STAGE_OPTIONS = ['T1', 'T2', 'T3', 'T4', 'TX']
N_STAGE_OPTIONS = ['N0', 'N1', 'N2', 'N3', 'NX']
SURGERY_STATUS_OPTIONS = ['No', 'Surgery performed', 'unknown']
RADIATION_STATUS_OPTIONS = ['Radiation', 'Refused (1988+)', 'Unknown']
CHEMOTHERAPY_STATUS_OPTIONS = ['No/Unknown', 'Yes']
TUMOR_SIZE_OPTIONS = ['>=2cm, <5cm', '>=5cm', 'Unknown', '<2cm']

FEATURE_NAMES = [
    'Age', 'Primary site', 'Histological type', 'T Stage', 'N Stage',
    'Surgery status', 'Radiation status', 'Chemotherapy status', 'Tumor size'
]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    st.set_page_config(
        page_title='胃癌远处转移预测系统',
        page_icon='🏥',
        layout='wide'
    )
    
    st.title('🏥 胃癌远处转移预测系统')
    st.markdown('---')
    
    st.sidebar.header('模型信息')
    st.sidebar.info('Stacking集成模型\n- 6个基学习器\n- RandomForest元学习器')
    
    model = load_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('患者基本信息')
        
        age = st.selectbox('年龄组 (Age)', AGE_OPTIONS, index=2)
        primary_site = st.selectbox('原发部位 (Primary site)', PRIMARY_SITE_OPTIONS)
        histological_type = st.selectbox('组织学类型 (Histological type)', HISTOLOGICAL_TYPE_OPTIONS)
        t_stage = st.selectbox('T分期 (T Stage)', T_STAGE_OPTIONS)
        n_stage = st.selectbox('N分期 (N Stage)', N_STAGE_OPTIONS)
    
    with col2:
        st.subheader('治疗信息')
        
        surgery_status = st.selectbox('手术状态 (Surgery status)', SURGERY_STATUS_OPTIONS)
        radiation_status = st.selectbox('放疗状态 (Radiation status)', RADIATION_STATUS_OPTIONS)
        chemotherapy_status = st.selectbox('化疗状态 (Chemotherapy status)', CHEMOTHERAPY_STATUS_OPTIONS)
        tumor_size = st.selectbox('肿瘤大小 (Tumor size)', TUMOR_SIZE_OPTIONS)
    
    st.markdown('---')
    
    if st.button('🔍 开始预测', type='primary', use_container_width=True):
        feature_values = [
            AGE_OPTIONS.index(age),
            PRIMARY_SITE_OPTIONS.index(primary_site),
            HISTOLOGICAL_TYPE_OPTIONS.index(histological_type),
            T_STAGE_OPTIONS.index(t_stage),
            N_STAGE_OPTIONS.index(n_stage),
            SURGERY_STATUS_OPTIONS.index(surgery_status),
            RADIATION_STATUS_OPTIONS.index(radiation_status),
            CHEMOTHERAPY_STATUS_OPTIONS.index(chemotherapy_status),
            TUMOR_SIZE_OPTIONS.index(tumor_size)
        ]
        
        input_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
        
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        prob_no_metastasis = prediction_proba[0]
        prob_metastasis = prediction_proba[1]
        
        st.subheader('📊 预测结果')
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.error('⚠️ 高风险：预测存在远处转移')
            else:
                st.success('✅ 低风险：预测未发现远处转移')
        
        with result_col2:
            st.metric('转移概率', f'{prob_metastasis*100:.2f}%')
        
        st.subheader('📈 概率分布')
        
        prob_df = pd.DataFrame({
            '类别': ['未转移', '转移'],
            '概率': [prob_no_metastasis, prob_metastasis]
        })
        
        st.bar_chart(prob_df.set_index('类别'))
        
        with st.expander('📋 查看输入特征编码值'):
            encoding_df = pd.DataFrame({
                '特征': FEATURE_NAMES,
                '选项': [age, primary_site, histological_type, t_stage, n_stage,
                        surgery_status, radiation_status, chemotherapy_status, tumor_size],
                '编码值': feature_values
            })
            st.dataframe(encoding_df, use_container_width=True)


if __name__ == '__main__':
    main()
