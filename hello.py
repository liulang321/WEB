#这是训练好的模型，等会用cat model带入节省时间（如何将模型与实时输入的数据衔接）
#导入数据
import os
print(os.getcwd())  # 获取当前工作目录
from pyecharts.charts import Map
from pyecharts import options as opts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df1= pd.read_excel('Liver7.xlsx')
import pandas as pd
df1.head()

from sklearn.preprocessing import LabelEncoder
df1.iloc[:,-1] = LabelEncoder().fit_transform(df1.iloc[:,-1])
from sklearn.preprocessing import OrdinalEncoder

#接口categories_对应LabelEncod er的接口classes_，一模一样的功能
df1_ = df1.copy()#对df进行复制，怕弄错
df1_.head()
OrdinalEncoder().fit(df1_.iloc[:,0:-1]).categories_#第0列是性别，到-1不包括-1
df1_.iloc[:,0:-1] = OrdinalEncoder().fit_transform(df1_.iloc[:,0:-1])
df1_.head()
#输出文件为csv
# df_.to_csv('df_.csv')

df2_=df1_

# 假设 df 是你的 DataFrame，'column_name' 是你想填充的列
column_name = 'Race'

# 找到列的众数
column_mode = df2_[column_name].mode()[0]

# 使用众数填充缺失值
df2_[column_name].fillna(column_mode, inplace=True)
#查看填补之后的数据,race被填充了
df2_.info()
df2_.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# Ordinal encode all columns except the last one
df2_.iloc[:, 0:-1] = OrdinalEncoder().fit_transform(df2_.iloc[:, 0:-1])

# Fill missing values in all columns with their respective modes
for column in df1_.columns:
    column_mode = df2_[column].mode()[0]
    df2_[column].fillna(column_mode, inplace=True)

# Check the DataFrame info after filling missing values
df2_.info()

# Display the DataFrame
# df1_.head()

df1_=df2_
df1_=df1_.drop(columns='Patient_ID')
# df1_imputed=df1_.drop(columns='Patient_ID')
# # df1_imputed=df1_
# # print(df_imputed)
#
# # #三个数据，用两个表示即可，00表示C
# # T=pd.get_dummies(df1_imputed['T'],prefix='T',drop_first=True)
# # N=pd.get_dummies(df1_imputed['N'],prefix='N',drop_first=True)
# Sequencenumber=pd.get_dummies(df1_imputed['Sequence#number'],prefix='Sequence#number',drop_first=True)
# # Grade=pd.get_dummies(df1_imputed['Grade'],prefix='Grade',drop_first=True)
# Sex=pd.get_dummies(df1_imputed['Sex'],prefix='Sex',drop_first=True)
# Age=pd.get_dummies(df1_imputed['Age'],prefix='Age',drop_first=True)
# Race=pd.get_dummies(df1_imputed['Race'],prefix='Race',drop_first=True)
# Histology=pd.get_dummies(df1_imputed['Histology'],prefix='Histology',drop_first=True)
# Primary=pd.get_dummies(df1_imputed['Primary#Site'],prefix='Primary#Site',drop_first=True)
# bone=pd.get_dummies(df1_imputed['Bone#metastases'],prefix='Bone#metastases',drop_first=True)
# brain=pd.get_dummies(df1_imputed['Brain#metastases'],prefix='Brain#metastases',drop_first=True)
# lung=pd.get_dummies(df1_imputed['Lung#metastases'],prefix='Lung#metastases',drop_first=True)
# Marital=pd.get_dummies(df1_imputed['Marital#status'],prefix='Marital#status',drop_first=True)
# lymphregiolsurgery=pd.get_dummies(df1_imputed['Regional#surgery#method'],prefix='Regional#surgery#method',drop_first=True)
# Radiation=pd.get_dummies(df1_imputed['Radiation#recode'],prefix='Radiation#recode',drop_first=True)
# Chemotherapy=pd.get_dummies(df1_imputed['Chemotherapy#recode'],prefix='Chemotherapy#recode',drop_first=True)
# surgerymethod=pd.get_dummies(df1_imputed['Surgery#method'],prefix='Surgery#method',drop_first=True)
# # liver=pd.get_dummies(df_imputed['Liver#metastases'],prefix='Liver#metastases',drop_first=False)
# df1_encoded=pd.concat([df1_imputed,Sequencenumber,Sex,Age,Race,Histology,Primary,bone,brain,lung,Marital,lymphregiolsurgery,Radiation,Chemotherapy,surgerymethod],axis=1)
# df1_encoded=df1_encoded.drop(columns=["Sequence#number","Sex", "Age", "Race","Histology","Primary#Site","Bone#metastases","Brain#metastases","Lung#metastases","Regional#surgery#method","Marital#status","Radiation#recode","Chemotherapy#recode","Surgery#method"])
# print(df1_encoded)
# df1_encoded.info()
df1_encoded=df1_
#定义xY
y=df1_encoded['Liver#metastases']
X=df1_encoded.drop(columns=['Liver#metastases'])
y.shape,X.shape

from sklearn.model_selection import train_test_split
xtrain,xtest,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
y_train.shape,xtrain.shape

import pandas as pd

# 将标签列转换为NumPy数组
y_test_array = y_test.to_numpy()

# 创建包含测试样本和标签的 DataFrame
df = pd.DataFrame(np.concatenate((xtest, y_test_array[:, np.newaxis]), axis=1))

# 设置列名
column_names = list(range(df.shape[1] - 1)) + ['Label']
df.columns = column_names

from scipy.stats import ks_2samp
import numpy as np

# Flatten xtrain and xtest arrays
xtrain_flat = np.ravel(xtrain)
xtest_flat = np.ravel(xtest)

# Perform KS test
ks_statistic, p_value = ks_2samp(xtrain_flat, xtest_flat)

# Print the KS statistic and p-value
print("KS Statistic:", ks_statistic)
print("p-value:", p_value)

xtrainRF13=xtrain
xtestRF13=xtest

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

print(y_train.dtype)
print(y_test.dtype)

# Convert columns to numeric types
xtrainRF13['Grade'] = xtrainRF13['Grade'].astype(float)
xtrainRF13['T'] = xtrainRF13['T'].astype(float)
xtrainRF13['N'] = xtrainRF13['N'].astype(float)# Define additional parameters
columns_to_convert = ['Sex', 'Age', 'Histology', 'Primary#Site', 'Radiation#recode', 'Chemotherapy#recode', 'Sequence#number']

# 遍历每个列并尝试将其转换为整数
for col in columns_to_convert:
    try:
        xtrainRF13[col] = xtrainRF13[col].astype(int)
    except ValueError:
        print(f"Unable to convert column '{col}' to int. Check for non-numeric values.")

# # Convert columns to numeric types
# 将列转换为整数类型
xtestRF13['Grade'] = xtestRF13['Grade'].astype(int)
xtestRF13['T'] = xtestRF13['T'].astype(int)
xtestRF13['N'] = xtestRF13['N'].astype(int)
##建立最佳模型（lgbm）
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score

additional_params = {
    'n_estimators': 85,
    'learning_rate': 0.08,
    'max_depth': 10,
    'min_child_samples': 25,
    'num_leaves': 18,
    'reg_alpha': 0.16
}

# Merge additional parameters with existing parameters
all_params = {
    'random_state': 42,
    'boosting_type': 'gbdt',
    'n_jobs': 8,
    'silent': True,
    **additional_params  # Unpack additional parameters
}
# Create the LGBMClassifier model with all parameters
lgbm_model = LGBMClassifier(**all_params)

# Fit the model to the training data
lgbm_model.fit(xtrainRF13, y_train)


##保存这个Lgbm模型
lgbm_model.booster_.save_model('LGBM_model.txt')







# #KNN插补
# #保险起见,复制一下knn的数据集
# df_knn = df_.copy()
# #载入包
# from sklearn.impute import KNNImputer
# import pandas as pd
# # 创建一个包含缺失值的示例 DataFrame
# df_knn_= pd.DataFrame(df_knn)#只考虑列的影响
# # 实例化 KNN 填充器
# imputer = KNNImputer(n_neighbors=5)#选择了5个邻居
# # 使用 fit_transform 进行填充
# df_knn_filled = pd.DataFrame(imputer.fit_transform(df_knn_), columns=df.columns)
# # 打印填充后的 DataFrame
# print(df_knn_filled)
# #查看knn填充后是否填完:全都填完了
# df_knn_filled.info()
# #df_knn_filled 用knn填充之后的结果
#
# df_encoded=df_knn_filled#代换一下，现在不用独热了
# #%%
# #定义xY
# y=df_encoded['Liver#metastases']
# X=df_encoded.drop(columns=['Liver#metastases'])
# y.shape,X.shape
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# y_train.shape,X_train.shape
#
# y_test.value_counts()
# #对训练集过采样
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# X_train_resampled.shape, y_train_resampled.shape
#
# y_train_resampled.value_counts()
#
# #嵌入法
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# base_model=LogisticRegression(solver='liblinear')#增加迭代次数，需要更多的时间才能跑出来
# selector=SelectFromModel(estimator=base_model)
#
# X_train=selector.fit_transform(X_train_resampled,y_train_resampled)
# X_test=selector.transform(X_test)
# feature_names=X.columns[np.where(selector.get_support())]
# X_train=pd.DataFrame(columns=feature_names,data=X_train)
# X_test=pd.DataFrame(columns=feature_names,data=X_test)
# X_train
#
# #建模
# from lightgbm import LGBMClassifier
# from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
# from sklearn.metrics import classification_report, confusion_matrix,roc_curve
# lgbm_model = LGBMClassifier(random_state=42)
# lgbm_model.fit(X_train, y_train_resampled)
# lgbm_model.booster_.save_model('lgbm_model.txt')


# #混淆矩阵
# lgbm_y_pred = lgbm_model.predict(X_test)
# lgbm_y_proba = lgbm_model.predict_proba(X_test)[:, 1]
# lgbm_cm = confusion_matrix(y_test, lgbm_y_pred)
# lgbm_auc = roc_auc_score(y_test, lgbm_y_proba)
# lgbm_balanced_accuracy = balanced_accuracy_score(y_test, lgbm_y_pred)
#
# #输出结果
# print("LightGBM 预测结果：", lgbm_y_pred)
# print("LightGBM 混淆矩阵：")
# print(lgbm_cm)
# print("LightGBM ROC AUC 分数：", lgbm_auc)
# print("LightGBM 平衡准确率：", lgbm_balanced_accuracy)
#
# #分类报告
# print("LightGBM 分类报告:")
# print(classification_report(y_test, lgbm_model.predict(X_test)))



