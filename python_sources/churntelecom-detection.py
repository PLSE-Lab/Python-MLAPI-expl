#!/usr/bin/env python
# coding: utf-8

# # Loading the data using pandas

# In[ ]:


import pandas as pd
import numpy as np

# This is to modify pandas to show more columns.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data=pd.read_csv('../input/ChurnData.csv')


# # Top 5 row

# In[ ]:


data.head(5)


# In[ ]:


data.info()


# # Number of row n column in our data set

# ## 1) Row count

# In[ ]:


data.shape[0]


# ## 2) Column count

# In[ ]:


data.shape[1]


# ### Shape of the Data

# In[ ]:


data.shape


# ### Data types

# In[ ]:


data.dtypes


# # Checking for null values

# In[ ]:


data.isnull().sum()


# # Dropping rows having atleast 100 NAN value

# In[ ]:


data=data.dropna(thresh=100)


# In[ ]:


data.shape


# #  Removing the columns with most of the null values.

# In[ ]:




remove = ['date_of_last_rech_data_6',
          'last_date_of_month_7',
          'last_date_of_month_8',
          'last_date_of_month_9',
          'last_date_of_month_6',
          'date_of_last_rech_6',
          'date_of_last_rech_7',
          'date_of_last_rech_8',
          'date_of_last_rech_9',
          'date_of_last_rech_data_7', 
          'date_of_last_rech_data_8', 
          'date_of_last_rech_data_9', 
          'total_rech_data_6', 
          'total_rech_data_7', 
          'total_rech_data_8', 
          'total_rech_data_9', 
          'max_rech_data_6',
          'max_rech_data_7',
          'max_rech_data_8',
          'max_rech_data_9',
          'count_rech_2g_6',
          'count_rech_2g_7',
          'count_rech_2g_8',
          'count_rech_2g_9',
          'count_rech_3g_6',
          'count_rech_3g_7',
          'count_rech_3g_8',
          'count_rech_3g_9',
          'av_rech_amt_data_6',
          'av_rech_amt_data_7',
          'av_rech_amt_data_8',
          'av_rech_amt_data_9',
          'arpu_3g_6',
          'arpu_3g_7',
          'arpu_3g_8',
          'arpu_3g_9',
          'arpu_2g_6',
          'arpu_2g_7',
          'arpu_2g_8',
          'arpu_2g_9',
          'night_pck_user_6',
          'night_pck_user_7',
          'night_pck_user_8',
          'night_pck_user_9',
          'fb_user_6',
          'fb_user_7',
          'fb_user_8',
          'fb_user_9']
# Total 40
          

data = data.drop(remove, axis=1)


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# # Replacing null values with the mean of the corresponding column

# In[ ]:


''' { dtype - float64} '''


data['onnet_mou_6'].fillna(data['onnet_mou_6'].mean(), inplace=True)
data['onnet_mou_7'].fillna(data['onnet_mou_7'].mean(), inplace=True)
data['onnet_mou_8'].fillna(data['onnet_mou_8'].mean(), inplace=True)
data['onnet_mou_9'].fillna(data['onnet_mou_9'].mean(), inplace=True)


# # offnet_mou_

# In[ ]:


''' { dtype - float64} '''

data['offnet_mou_6'].fillna(data['offnet_mou_6'].mean(), inplace=True)
data['offnet_mou_7'].fillna(data['offnet_mou_7'].mean(), inplace=True)
data['offnet_mou_8'].fillna(data['offnet_mou_8'].mean(), inplace=True)
data['offnet_mou_9'].fillna(data['offnet_mou_9'].mean(), inplace=True)


# In[ ]:


# roam_ic_mou


''' { dtype - float64} '''

data['roam_ic_mou_6'].fillna(data['roam_ic_mou_6'].mean(), inplace=True)
data['roam_ic_mou_7'].fillna(data['roam_ic_mou_7'].mean(), inplace=True)
data['roam_ic_mou_8'].fillna(data['roam_ic_mou_8'].mean(), inplace=True)
data['roam_ic_mou_9'].fillna(data['roam_ic_mou_9'].mean(), inplace=True)
''' { dtype - float64} '''

data['roam_ic_mou_6'].fillna(data['roam_ic_mou_6'].mean(), inplace=True)
data['roam_ic_mou_7'].fillna(data['roam_ic_mou_7'].mean(), inplace=True)
data['roam_ic_mou_8'].fillna(data['roam_ic_mou_8'].mean(), inplace=True)
data['roam_ic_mou_9'].fillna(data['roam_ic_mou_9'].mean(), inplace=True)

#roam_og_mou


''' { dtype - float64} '''

data['roam_og_mou_6'].fillna(data['roam_og_mou_6'].mean(), inplace=True)
data['roam_og_mou_7'].fillna(data['roam_og_mou_7'].mean(), inplace=True)
data['roam_og_mou_8'].fillna(data['roam_og_mou_8'].mean(), inplace=True)
data['roam_og_mou_9'].fillna(data['roam_og_mou_9'].mean(), inplace=True)


#loc_og_t2t_mou


''' { dtype - float64} '''

data['loc_og_t2t_mou_6'].fillna(data['loc_og_t2t_mou_6'].mean(), inplace=True)
data['loc_og_t2t_mou_7'].fillna(data['loc_og_t2t_mou_7'].mean(), inplace=True)
data['loc_og_t2t_mou_8'].fillna(data['loc_og_t2t_mou_8'].mean(), inplace=True)
data['loc_og_t2t_mou_9'].fillna(data['loc_og_t2t_mou_9'].mean(), inplace=True)


# loc_og_t2m_mou


''' { dtype - float64} '''

data['loc_og_t2m_mou_6'].fillna(data['loc_og_t2m_mou_6'].mean(), inplace=True)
data['loc_og_t2m_mou_7'].fillna(data['loc_og_t2m_mou_7'].mean(), inplace=True)
data['loc_og_t2m_mou_8'].fillna(data['loc_og_t2m_mou_8'].mean(), inplace=True)
data['loc_og_t2m_mou_9'].fillna(data['loc_og_t2m_mou_9'].mean(), inplace=True)
# loc_og_t2f_mou


''' { dtype - float64} '''

data['loc_og_t2f_mou_6'].fillna(data['loc_og_t2f_mou_6'].mean(), inplace=True)
data['loc_og_t2f_mou_7'].fillna(data['loc_og_t2f_mou_7'].mean(), inplace=True)
data['loc_og_t2f_mou_8'].fillna(data['loc_og_t2f_mou_8'].mean(), inplace=True)
data['loc_og_t2f_mou_9'].fillna(data['loc_og_t2f_mou_9'].mean(), inplace=True)

#loc_og_t2c_mou


''' { dtype - float64} '''

data['loc_og_t2c_mou_6'].fillna(data['loc_og_t2c_mou_6'].mean(), inplace=True)
data['loc_og_t2c_mou_7'].fillna(data['loc_og_t2c_mou_7'].mean(), inplace=True)
data['loc_og_t2c_mou_8'].fillna(data['loc_og_t2c_mou_8'].mean(), inplace=True)
data['loc_og_t2c_mou_9'].fillna(data['loc_og_t2c_mou_9'].mean(), inplace=True)

#loc_og_mou


''' { dtype - float64} '''

data['loc_og_mou_6'].fillna(data['loc_og_mou_6'].mean(), inplace=True)
data['loc_og_mou_7'].fillna(data['loc_og_mou_7'].mean(), inplace=True)
data['loc_og_mou_8'].fillna(data['loc_og_mou_8'].mean(), inplace=True)
data['loc_og_mou_9'].fillna(data['loc_og_mou_9'].mean(), inplace=True)

#std_og_t2t_mou


''' { dtype - float64} '''

data['std_og_t2t_mou_6'].fillna(data['std_og_t2t_mou_6'].mean(), inplace=True)
data['std_og_t2t_mou_7'].fillna(data['std_og_t2t_mou_7'].mean(), inplace=True)
data['std_og_t2t_mou_8'].fillna(data['std_og_t2t_mou_8'].mean(), inplace=True)
data['std_og_t2t_mou_9'].fillna(data['std_og_t2t_mou_9'].mean(), inplace=True)

#std_og_t2m_mou


''' { dtype - float64} '''

data['std_og_t2m_mou_6'].fillna(data['std_og_t2m_mou_6'].mean(), inplace=True)
data['std_og_t2m_mou_7'].fillna(data['std_og_t2m_mou_7'].mean(), inplace=True)
data['std_og_t2m_mou_8'].fillna(data['std_og_t2m_mou_8'].mean(), inplace=True)
data['std_og_t2m_mou_9'].fillna(data['std_og_t2m_mou_9'].mean(), inplace=True)

# std_og_t2f_mou


''' { dtype - float64} '''

data['std_og_t2f_mou_6'].fillna(data['std_og_t2f_mou_6'].mean(), inplace=True)
data['std_og_t2f_mou_7'].fillna(data['std_og_t2f_mou_7'].mean(), inplace=True)
data['std_og_t2f_mou_8'].fillna(data['std_og_t2f_mou_8'].mean(), inplace=True)
data['std_og_t2f_mou_9'].fillna(data['std_og_t2f_mou_9'].mean(), inplace=True)
# std_og_t2c_mou


''' { dtype - float64} '''

data['std_og_t2c_mou_6'].fillna(data['std_og_t2c_mou_6'].mean(), inplace=True)
data['std_og_t2c_mou_7'].fillna(data['std_og_t2c_mou_7'].mean(), inplace=True)
data['std_og_t2c_mou_8'].fillna(data['std_og_t2c_mou_8'].mean(), inplace=True)
data['std_og_t2c_mou_9'].fillna(data['std_og_t2c_mou_9'].mean(), inplace=True)
# std_og_mou


''' { dtype - float64} '''

data['std_og_mou_6'].fillna(data['std_og_mou_6'].mean(), inplace=True)
data['std_og_mou_7'].fillna(data['std_og_mou_7'].mean(), inplace=True)
data['std_og_mou_8'].fillna(data['std_og_mou_8'].mean(), inplace=True)
data['std_og_mou_9'].fillna(data['std_og_mou_9'].mean(), inplace=True)

# isd_og_mou


''' { dtype - float64} '''

data['isd_og_mou_6'].fillna(data['isd_og_mou_6'].mean(), inplace=True)
data['isd_og_mou_7'].fillna(data['isd_og_mou_7'].mean(), inplace=True)
data['isd_og_mou_8'].fillna(data['isd_og_mou_8'].mean(), inplace=True)
data['isd_og_mou_9'].fillna(data['isd_og_mou_9'].mean(), inplace=True)
# spl_og_mou


''' { dtype - float64} '''

data['spl_og_mou_6'].fillna(data['spl_og_mou_6'].mean(), inplace=True)
data['spl_og_mou_7'].fillna(data['spl_og_mou_7'].mean(), inplace=True)
data['spl_og_mou_8'].fillna(data['spl_og_mou_8'].mean(), inplace=True)
data['spl_og_mou_9'].fillna(data['spl_og_mou_9'].mean(), inplace=True)
# og_others


''' { dtype - float64} '''

data['og_others_6'].fillna(data['og_others_6'].mean(), inplace=True)
data['og_others_7'].fillna(data['og_others_7'].mean(), inplace=True)
data['og_others_8'].fillna(data['og_others_8'].mean(), inplace=True)
data['og_others_9'].fillna(data['og_others_9'].mean(), inplace=True)
# loc_ic_t2t_mou


''' { dtype - float64} '''

data['loc_ic_t2t_mou_6'].fillna(data['loc_ic_t2t_mou_6'].mean(), inplace=True)
data['loc_ic_t2t_mou_7'].fillna(data['loc_ic_t2t_mou_7'].mean(), inplace=True)
data['loc_ic_t2t_mou_8'].fillna(data['loc_ic_t2t_mou_8'].mean(), inplace=True)
data['loc_ic_t2t_mou_9'].fillna(data['loc_ic_t2t_mou_9'].mean(), inplace=True)

# loc_ic_t2m_mou

[ ]
''' { dtype - float64} '''

data['loc_ic_t2m_mou_6'].fillna(data['loc_ic_t2m_mou_6'].mean(), inplace=True)
data['loc_ic_t2m_mou_7'].fillna(data['loc_ic_t2m_mou_7'].mean(), inplace=True)
data['loc_ic_t2m_mou_8'].fillna(data['loc_ic_t2m_mou_8'].mean(), inplace=True)
data['loc_ic_t2m_mou_9'].fillna(data['loc_ic_t2m_mou_9'].mean(), inplace=True)

# loc_ic_t2f_mou


''' { dtype - float64} '''

data['loc_ic_t2f_mou_6'].fillna(data['loc_ic_t2f_mou_6'].mean(), inplace=True)
data['loc_ic_t2f_mou_7'].fillna(data['loc_ic_t2f_mou_7'].mean(), inplace=True)
data['loc_ic_t2f_mou_8'].fillna(data['loc_ic_t2f_mou_8'].mean(), inplace=True)
data['loc_ic_t2f_mou_9'].fillna(data['loc_ic_t2f_mou_9'].mean(), inplace=True)

# loc_ic_mou


''' { dtype - float64} '''

data['loc_ic_mou_6'].fillna(data['loc_ic_mou_6'].mean(), inplace=True)
data['loc_ic_mou_7'].fillna(data['loc_ic_mou_7'].mean(), inplace=True)
data['loc_ic_mou_8'].fillna(data['loc_ic_mou_8'].mean(), inplace=True)
data['loc_ic_mou_9'].fillna(data['loc_ic_mou_9'].mean(), inplace=True)
# std_ic_t2t_mou


''' { dtype - float64} '''

data['std_ic_t2t_mou_6'].fillna(data['std_ic_t2t_mou_6'].mean(), inplace=True)
data['std_ic_t2t_mou_7'].fillna(data['std_ic_t2t_mou_7'].mean(), inplace=True)
data['std_ic_t2t_mou_8'].fillna(data['std_ic_t2t_mou_8'].mean(), inplace=True)
data['std_ic_t2t_mou_9'].fillna(data['std_ic_t2t_mou_9'].mean(), inplace=True)
# std_ic_t2m_mou


''' { dtype - float64} '''

data['std_ic_t2m_mou_6'].fillna(data['std_ic_t2m_mou_6'].mean(), inplace=True)
data['std_ic_t2m_mou_7'].fillna(data['std_ic_t2m_mou_7'].mean(), inplace=True)
data['std_ic_t2m_mou_8'].fillna(data['std_ic_t2m_mou_8'].mean(), inplace=True)
data['std_ic_t2m_mou_9'].fillna(data['std_ic_t2m_mou_9'].mean(), inplace=True)
# std_ic_t2f_mou


''' { dtype - float64} '''

data['std_ic_t2f_mou_6'].fillna(data['std_ic_t2f_mou_6'].mean(), inplace=True)
data['std_ic_t2f_mou_7'].fillna(data['std_ic_t2f_mou_7'].mean(), inplace=True)
data['std_ic_t2f_mou_8'].fillna(data['std_ic_t2f_mou_8'].mean(), inplace=True)
data['std_ic_t2f_mou_9'].fillna(data['std_ic_t2f_mou_9'].mean(), inplace=True)
# std_ic_t2o_mou


''' { dtype - float64} '''

data['std_ic_t2o_mou_6'].fillna(data['std_ic_t2o_mou_6'].mean(), inplace=True)
data['std_ic_t2o_mou_7'].fillna(data['std_ic_t2o_mou_7'].mean(), inplace=True)
data['std_ic_t2o_mou_8'].fillna(data['std_ic_t2o_mou_8'].mean(), inplace=True)
data['std_ic_t2o_mou_9'].fillna(data['std_ic_t2o_mou_9'].mean(), inplace=True)

# std_ic_mou


''' { dtype - float64} '''

data['std_ic_mou_6'].fillna(data['std_ic_mou_6'].mean(), inplace=True)
data['std_ic_mou_7'].fillna(data['std_ic_mou_7'].mean(), inplace=True)
data['std_ic_mou_8'].fillna(data['std_ic_mou_8'].mean(), inplace=True)
data['std_ic_mou_9'].fillna(data['std_ic_mou_9'].mean(), inplace=True)

# spl_ic_mou


''' { dtype - float64} '''

data['spl_ic_mou_6'].fillna(data['spl_ic_mou_6'].mean(), inplace=True)
data['spl_ic_mou_7'].fillna(data['spl_ic_mou_7'].mean(), inplace=True)
data['spl_ic_mou_8'].fillna(data['spl_ic_mou_8'].mean(), inplace=True)
data['spl_ic_mou_9'].fillna(data['spl_ic_mou_9'].mean(), inplace=True)
# isd_ic_mou


''' { dtype - float64} '''

data['isd_ic_mou_6'].fillna(data['isd_ic_mou_6'].mean(), inplace=True)
data['isd_ic_mou_7'].fillna(data['isd_ic_mou_7'].mean(), inplace=True)
data['isd_ic_mou_8'].fillna(data['isd_ic_mou_8'].mean(), inplace=True)
data['isd_ic_mou_9'].fillna(data['isd_ic_mou_9'].mean(), inplace=True)
# ic_others


''' { dtype - float64} '''

data['ic_others_6'].fillna(data['ic_others_6'].mean(), inplace=True)
data['ic_others_7'].fillna(data['ic_others_7'].mean(), inplace=True)
data['ic_others_8'].fillna(data['ic_others_8'].mean(), inplace=True)
data['ic_others_9'].fillna(data['ic_others_9'].mean(), inplace=True)

data['loc_og_t2o_mou'].fillna(data['loc_og_t2o_mou'].mean(), inplace=True)
data['std_og_t2o_mou'].fillna(data['std_og_t2o_mou'].mean(), inplace=True)
data['loc_ic_t2o_mou'].fillna(data['loc_ic_t2o_mou'].mean(), inplace=True)


# # All null valued is removed now

# In[ ]:


data.isnull().sum()


# # Converting Target Categorical variable  in numeric

# In[ ]:


data['Churn1'] = (data['Churn1'] == True ).astype(int)
data['Churn1'] = (data['Churn1'] == False ).astype(int)
data['Churn2'] = (data['Churn2'] == True ).astype(int)
data['Churn2'] = (data['Churn2'] == False ).astype(int)
data['Final churn'] = (data['Final churn'] == True ).astype(int)
data['Final churn'] = (data['Final churn'] == False ).astype(int)


# In[ ]:


data.head(5)


# # Describe : statistics info

# In[ ]:


data.describe()


# # Applying filter to get high value customer & 70%ile  value

# In[ ]:


perc=[.20,.40,.70,.80]
data[['total_rech_amt_6','total_rech_amt_7']].describe(percentiles=perc)


# In[ ]:


data=data[data['total_rech_amt_6']>400 &(data['total_rech_amt_7']>400)]


# # Now we will check noise in data outlier detection

# # Visualization using BOX PLOT

# In[ ]:



import seaborn as sns
import matplotlib.pyplot as plot



col_names = ['aon', 'aug_vbc_3g', 'jul_vbc_3g','jun_vbc_3g',]
fig, ax = plot.subplots(len(col_names), figsize=(10,10))

for i, col_val in enumerate(col_names):

    sns.boxplot(x=data[col_val], ax=ax[i])


# # Calculating Z score

# In[ ]:


from scipy import stats
import numpy as np
z=np.abs(stats.zscore(data))
print(z)


#  # First array showing list of row num , second list of column num of data which are outlier

# In[ ]:


threshold=3  # threshold limit genearlly taken as 3 or -3
print(np.where(z>3))
print(z[0][178])


# > # Visulization using heatmap

# In[ ]:


import seaborn as sns
corre=data.corr()
sns.heatmap(corre,annot=True,cmap='viridis',linewidth=7)


# # standarlization using Min max scaler

# In[ ]:


#standardalization

y=data['Final churn']
X=data.drop(columns=['Final churn'],axis=1)
from sklearn.preprocessing import MinMaxScaler
feature=X.columns.values
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X=pd.DataFrame(scaler.transform(X))
X.columns=feature
X.head()


# # Applying PCA for dimentionality Reduction

# In[ ]:


from sklearn.decomposition import PCA
pca =PCA(n_components=2)

pc=pca.fit_transform(X)
principaldf= pd.DataFrame(data=pc ,columns=['Principal Component 1','Principal Component 2'])
principaldf.head()
finaldf = pd.concat([principaldf, y], axis = 1)
finaldf.head()


# # Train-Test split data

# In[ ]:


from sklearn.model_selection import train_test_split
principaldf_train,principaldf_test,y_train,y_test= train_test_split(principaldf,y,test_size=0.33 ,random_state=42)

print(principaldf_train.shape)
print(principaldf_test.shape)
print(y_train.shape)
print(y_test.shape)


# # checking count of class 0 and class 1

# In[ ]:


print(sum(y_train==0))
print(sum(y_train==1))


# # We can see above  class is  imbalalnce

# # Applying Near for removing class imbalance problem

# In[ ]:


from imblearn.under_sampling  import NearMiss
nr=NearMiss()
principaldf_train,y_train=nr.fit_sample(principaldf_train, y_train)


# # We can see count is balance now

# In[ ]:


print(sum(y_train==0))
print(sum(y_train==1))


# # Applying different classification model

# # Logistics Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


model=LogisticRegression()
output=model.fit(principaldf_train ,y_train)

Predict=model.predict(principaldf_test)
print("Logistc Regression Accuracy")
print(metrics.accuracy_score(y_test,Predict)*100)
print("Area under curve", )
print(roc_auc_score(y_test,Predict))

from sklearn.metrics import confusion_matrix

print("Classification report")
print(classification_report(y_test,Predict))
print(output)
print("Confusion matrix")
print(confusion_matrix(y_test,Predict))


# # Decision tree

# In[ ]:




from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
output=model.fit(principaldf_train,y_train)
prediction=model.predict(principaldf_test)
print("model accuracy")
print(metrics.accuracy_score(y_test,prediction)*100)
print("Area under curve", )
print(roc_auc_score(y_test,prediction))
print("Classification report")
print(classification_report(y_test,prediction))


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

principaldf_train,principaldf_test,y_train,y_test= train_test_split(principaldf,y,test_size=0.33 ,random_state=42)

rf =RandomForestClassifier(n_estimators=70,max_depth=2,random_state=10,criterion='gini')
output1=rf.fit(principaldf_train,y_train)
Predict_out=model.predict(principaldf_test)
print("Random forest Accuracy")
print(metrics.accuracy_score(y_test,Predict_out)*100)
print("Area under curve", )
print(roc_auc_score(y_test,Predict_out))
print ("\n Classification report : \n",classification_report(y_test,Predict_out))


# # Paramete Tuning in Random Forest

# # In parameter tuning we change the value of parameter n_estimators=80,max_depth=2,random_state=10,criterion='gini' to choose one giving best accuracy

# # I have applied for loop for n_estimators range (50,60) to check which one giving highest accuracy

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

principaldf_train,principaldf_test,y_train,y_test= train_test_split(principaldf,y,test_size=0.33 ,random_state=42)

for i in range(50,60):
    rf =RandomForestClassifier(n_estimators=i,max_depth=2,random_state=10,criterion='gini')
    output1=rf.fit(principaldf_train,y_train)
    Predict_out=model.predict(principaldf_test)
    print("Random forest Accuracy")
    print(metrics.accuracy_score(y_test,Predict_out)*100)
    print("Area under curve", )
    print(roc_auc_score(y_test,Predict_out))
    print ("\n Classification report : \n",classification_report(y_test,Predict_out))
    
    

