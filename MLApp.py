import streamlit as st
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import sys

st.sidebar.title('Variables Input')
loaded_model = lgb.Booster(model_file='LGBM_model.txt')

# # 定义预测函数
# def predict(esophageal_cancer_data, model):
#     # 使用模型进行预测
#     predictions = model.predict(esophageal_cancer_data, num_iteration=model.best_iteration)
#     return predictions
#
# # # 创建主函数
# def main():
#     # 加载模型
#     model = loaded_model()

    # 创建标题
st.title('Predicting Liver Metastasis in Esophageal Cancer')

    # 添加输入特征
st.header('Input Features')

    # 通过Streamlit的侧边栏添加输入特征
Sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
Age = st.sidebar.selectbox('Age',['15-19 years','20-24 years','25-29 years','30-34 years','35-39 years','40-44 years','45-49 years','50-54 years','55-59 years','60-64 years','65-69 years','70-74 years','75-79 years','80-84 years','85+ years'])
Race = st.sidebar.selectbox('Race', ['American Indian/Alaska Native','Asian or Pacific Islander','Black','White'])
Primary_site=st.sidebar.selectbox('Primary_site', ['Cervical esophagus','Thoracic esophagus','Abdominal esophagus','Upper third of esophagus','Middle third of esophagus','Lower third of esophagus','Overlapping lesion of esophagus','Esophagus, NOS'])
Grade=st.sidebar.selectbox('Grade', ['Grade I','Grade II','Grade III','Grade IV','Unknown'])
Bone_metastases = st.sidebar.selectbox('Bone metastases', ['No', 'Yes'])
Brain_metastases = st.sidebar.selectbox('Brain metastases', ['No', 'Yes'])
Lung_metastases = st.sidebar.selectbox('Lung metastases', ['No', 'Yes'])
T = st.sidebar.selectbox('T', ['T1','T2','T3','T4'])
N = st.sidebar.selectbox('N', ['N0','N1','N2','N3'])
Marital_status=st.sidebar.selectbox('Marital status', ['Divorced','Married','Separated','Single','Unmarried or Domestic Partner','Widowed'])
Histology = st.sidebar.selectbox('Histology', ['Adenocarcinoma', 'Others','Squamous cell carcinoma'])
Surgery_method=st.sidebar.selectbox('Surgery method',['Esophagectomy','Esophagectomy, NOS WITH laryngectomy and/or gastrectomy, NOS','Local tumor destruction','Local tumor excision','Non-surgery','Partial esophagectomy','Surgery, NOS','Total esophagectomy'])
Radiation_recode=st.sidebar.selectbox('Radiation_recode',['Beam radiation','Combination of beam with implants or isotopes','None/Unknown','Radiation, NOS  method or source not specified','Radioactive implants (includes brachytherapy) (1988+)','Radioisotopes (1988+)','Recommended, unknown if administered','Refused (1988+)'])
Regional_surgery_method = st.sidebar.selectbox('Regional_surgery_method', ['1 to 3 regional lymph nodes removed', '4 or more regional lymph nodes removed','Biopsy or aspiration of regional lymph node, NOS','Non-surgery','Sentinel lymph node biopsy','Sentinel node biopsy and lym nd removed different times','Sentinel node biopsy and lym nd removed same/unstated time'])
Chemotherapy_recode = st.sidebar.selectbox('Chemotherapy recode', ['No/Unknown', 'Yes'])
Sequence_number= st.sidebar.selectbox('Sequence number',['10th of 10 or more primaries','1st of 2 or more primaries','2nd of 2 or more primaries','3rd of 3 or more primaries','4th of 4 or more primaries','5th of 5 or more primaries','6th of 6 or more primaries','7th of 7 or more primaries','8th of 8 or more primaries','One primary only'])

# tumor_size = st.sidebar.number_input('肿瘤大小', min_value=0.0, max_value=20.0, value=5.0)
# 构建特征DataFrame
input_data = pd.DataFrame({
        'Sex': [Sex],
        'Histology':[Histology],
        'Brain metastases':[Brain_metastases],
        'Regional surgery method':[Regional_surgery_method],
        'Chemotherapy recode':[Chemotherapy_recode],
        'Bone metastases':[Bone_metastases],
        'Brain metastases':[Brain_metastases],
        'Lung metastases':[Lung_metastases],
        'Age':[Age],
        'Race':[Race],
        'Primary_site':[Primary_site],
        'Grade':[Grade],
        'T':[T],
        'N':[N],
        'Marital status':[Marital_status],
        'Surgery method':[Surgery_method],
        'Radiation recode':[Radiation_recode],
        'Sequence_number':[Sequence_number]
        # '肿瘤大小': [tumor_size]
    })

    # 对性别进行编码'15-19 years','20-24 years','25-29 years','30-34 years','35-39 years','40-44 years','45-49 years','50-54 years','55-59 years','60-64 years','65-69 years','70-74 years','75-79 years','80-84 years','85+ years'
input_data['Sex'] = input_data['Sex'].map({'Male': 0, 'Female': 1})
input_data['Age'] = input_data['Age'].map({
    '15-19 years': 0,
    '20-24 years': 1,
    '25-29 years': 2,
    '30-34 years': 3,
    '35-39 years': 4,
    '40-44 years': 5,
    '45-49 years': 6,
    '50-54 years': 7,
    '55-59 years': 8,
    '60-64 years': 9,
    '65-69 years': 10,
    '70-74 years': 11,
    '75-79 years': 12,
    '80-84 years': 13,
    '85+ years': 14
})
#input_data['Age'] = input_data['Age'].map({'15-19 years':0,'20-24 years':1,'25-29 years':2,'30-34 years':3,'35-39 years':4,'40-44 years':5,'45-49 years':6,'50-54 years':7,'55-59 years':8,'60-64 years':9,'65-69 years':10,'70-74 years':11,'75-79 years':12,'80-84 years':13,'85+ years':14})input_data['Histology'] = input_data['Histology'].map({'Adenocarcinoma':0, 'Others':1,'Squamous cell carcinoma':2})
input_data['Histology'] = input_data['Histology'].map({'Adenocarcinoma':0, 'Others':1,'Squamous cell carcinoma':2})
input_data['Grade'] = input_data['Grade'].map({'Grade I':0,'Grade II':1,'Grade III':2,'Grade IV':3,'Unknown':4})
input_data['T'] = input_data['T'].map({'T1':0,'T2':1,'T3':2,'T4':3})
input_data['N'] = input_data['N'].map({'N0':0,'N1':1,'N2':2,'N3':3})
#Radiation_recode=st.sidebar.selectbox('Radiation_recode',['Beam radiation','Combination of beam with implants or isotopes','None/Unknown','Radiation, NOS  method or source not specified','Radioactive implants (includes brachytherapy) (1988+)','Radioisotopes (1988+)','Recommended, unknown if administered','Refused (1988+)'])
#Sequence_number= st.sidebar.selectbox('Sequence number',['10th of 10 or more primaries','1st of 2 or more primaries','2nd of 2 or more primaries','3rd of 3 or more primaries','4th of 4 or more primaries','5th of 5 or more primaries','6th of 6 or more primaries','7th of 7 or more primaries','8th of 8 or more primaries','One primary only'])
input_data['Sequence_number'] = input_data['Sequence_number'].map({'10th of 10 or more primaries':0,'1st of 2 or more primaries':1,'2nd of 2 or more primaries':2,'3rd of 3 or more primaries':3,'4th of 4 or more primaries':4,'5th of 5 or more primaries':5,'6th of 6 or more primaries':6,'7th of 7 or more primaries':7,'8th of 8 or more primaries':8,'One primary only':9})
input_data['Radiation recode'] = input_data['Radiation recode'].map({'Beam radiation':0,'Combination of beam with implants or isotopes':1,'None/Unknown':2,'Radiation, NOS  method or source not specified':3,'Radioactive implants (includes brachytherapy) (1988+)':4,'Radioisotopes (1988+)':5,'Recommended, unknown if administered':6,'Refused (1988+)':7})
input_data['Surgery method'] = input_data['Surgery method'].map({'Esophagectomy':0,'Esophagectomy, NOS WITH laryngectomy and/or gastrectomy, NOS':1,'Local tumor destruction':2,'Local tumor excision':3,'Non-surgery':4,'Partial esophagectomy':5,'Surgery, NOS':6,'Total esophagectomy':7})
input_data['Marital status'] = input_data['Marital status'].map({'Divorced':0,'Married':1,'Separated':2,'Single':3,'Unmarried or Domestic Partner':4,'Widowed':5})
input_data['Race'] = input_data['Race'].map({'American Indian/Alaska Native': 0, 'Asian or Pacific Islander': 1,'Black':2,'White':3})
input_data['Brain metastases'] = input_data['Brain metastases'].map({'No': 0, 'Yes': 1})
input_data['Lung metastases'] = input_data['Lung metastases'].map({'No': 0, 'Yes': 1})
input_data['Bone metastases'] = input_data['Bone metastases'].map({'No': 0, 'Yes': 1})
#Primary_site=st.sidebar.selectbox('Primary_site', ['Cervical esophagus','Thoracic esophagus','Abdominal esophagus','Upper third of esophagus','Middle third of esophagus','Lower third of esophagus','Overlapping lesion of esophagus','Esophagus, NOS'])
input_data['Primary_site'] = input_data['Primary_site'].map({'Cervical esophagus':0,'Thoracic esophagus':1,'Abdominal esophagus':2,'Upper third of esophagus':3,'Middle third of esophagus':4,'Lower third of esophagus':5,'Overlapping lesion of esophagus':6,'Esophagus, NOS':7})
input_data['Regional surgery method'] = input_data['Regional surgery method'].map({'1 to 3 regional lymph nodes removed':0, '4 or more regional lymph nodes removed':1,'Biopsy or aspiration of regional lymph node, NOS':2,'Non-surgery':3,'Sentinel lymph node biopsy':4,'Sentinel node biopsy and lym nd removed different times':5,'Sentinel node biopsy and lym nd removed same/unstated time':6})
input_data['Chemotherapy recode'] = input_data['Chemotherapy recode'].map({'No/Unknown':0, 'Yes':1})
    # 展示输入数据
st.write('Input Data：')
st.write(input_data)
st.write("LightGBM AUC：0.88")#是不是固定的值？？
#     # 如果有数据，进行预测
# if st.button('预测'):
#         # 预测
#     # predictions = loaded_model.predict(input_data)
#     lgbm_y_pred = loaded_model.predict(input_data)
#     # lgbm_y_proba = lgbm_model.predict_proba(X_test)[:, 1]
#     # lgbm_cm = confusion_matrix(y_test, lgbm_y_pred)
#     # lgbm_auc = roc_auc_score(y_test, lgbm_y_proba)
#     # lgbm_balanced_accuracy = balanced_accuracy_score(y_test, lgbm_y_pred)
#         # 显示预测结果
#     st.write('预测结果：')
#     st.write(lgbm_y_pred)
#
# # # 运行主函数
# # if __name__ == '__main__':
# #     main()

if st.button('Predict'):
    X = pd.DataFrame(input_data)
    y_pred = loaded_model.predict(X)
    # y_proba = loaded_model.predict_proba(X）
    # auc = roc_auc_score(y_test, y_proba)
    #
    # if y_pred[0]:
    #     st.success('该病人有食管癌肝转移的风险，AUC值为{}'.format(auc))
    # else:
    #     st.success('该病人无食管癌肝转移的风险，AUC值为{}'.format(auc))
    st.write('Probability of liver metastases：')
    st.write(y_pred)