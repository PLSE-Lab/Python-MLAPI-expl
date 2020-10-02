#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Analysis

# ![Heart Diseases](https://images.indianexpress.com/2017/07/heart-main.jpg?w=759&h=422&imflag=true)

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings 
warnings.filterwarnings("ignore")


# Reading the Dataset

# In[ ]:


df = pd.read_csv("../input/heart.csv")


# In[ ]:


df.head()


# Attribute Information
# 
# **1 age:**
# 
#         age in years 
#         
# **2 sex: **
# 
#         1 : male 
#         0 : female   
#         
# **3 cp: chest pain type**
# 
#         Value 1: typical angina
#         Value 2: atypical angina
#         Value 3: non-anginal pain
#         Value 4: asymptomatic 
#  
# **4 trestbps:** 
# 
#         resting blood pressure (in mm Hg on admission to the hospital)
#         
# **5 chol:** 
# 
#         serum cholestoral in mg/dl 
#         
# **6 fbs: (fasting blood sugar > 120 mg/dl)**
# 
#         1 = true
#         0 = false     
#         
# **7 restecg: resting electrocardiographic results**
# 
#         Value 0: normal
#         Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#         Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
#         
# **8 thalach:  **
# 
#         maximum heart rate achieved 
#         
# **9 exang: **
# 
#         exercise induced angina 
#         1 = yes
#         0 = no    
#         
# **10 oldpeak**
# 
#         ST depression induced by exercise relative to rest 
#         
# **11 slope:  **
# 
#         the slope of the peak exercise ST segment 
#         Value 1: upsloping 
#         Value 2: flat
#         Value 3: downsloping
#         
# **12 ca:**
# 
#         number of major vessels (0-3) colored by flourosopy 
#         
# **13 thal:**
# 
#         3 = normal
#         6 = fixed defect
#         7 = reversable defect 
#         
# **14 num:** 
# 
#         diagnosis of heart disease (angiographic disease status)      
#         Value 0: less than 50% diameter narrowing 
#         Value 1: greater than 50% diameter narrowing

# So the key take away from the dataset we have ordinal variables , but as we understand there is no order of ranking, hence after EDA we might need to apply one hot encoding in model building.

# Also changing the dataframe columns to much understandable name

# In[ ]:


df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# Lets check for some missing Information in the dataset

# In[ ]:


df.isna().sum()


# No Missing Data in the dataset

# ## EDA

# Lets see the data is balanced or unbalanced on the based of labels (Target)

# In[ ]:


sns.countplot(df['target'].value_counts())


# We can clearly see our data is approximatley balanced with more 1 i.e. (greater than 50% diameter narrowing)

# In[ ]:


sns.pairplot(data=df,hue='target')


# Lets Fix the Ordinal Variables:
# We cant rank them as 0,1 or so on as there is no ranking mechanism.
# We need to do some reverse engineering ! Converting numbers to strings and then extracting them back in the form of One hot encoding columns.

# In[ ]:


print("Sex Feature unique Values is\n{0}".format(df['sex'].value_counts()))
print("Chest Pain Feature unique Values is\n{0}".format(df['chest_pain_type'].value_counts()))
print("Fast Blood Sugar Feature unique Values is\n{0}".format(df['fasting_blood_sugar'].value_counts()))
print("rest_ecg Feature unique Values is\n{0}".format(df['rest_ecg'].value_counts()))
print("exercise_induced_angina Feature unique Values is\n{0}".format(df['exercise_induced_angina'].value_counts()))
print("st_slope Feature unique Values is\n{0}".format(df['st_slope'].value_counts()))
print("thalassemia Feature unique Values is\n{0}".format(df['thalassemia'].value_counts()))


# In[ ]:


# Applying Reverse Engineerig on the following features for applying one hot encoding.
# Sex Feature 
df['sex'][df['sex'] == 1] = 'male'
df['sex'][df['sex'] == 0] = 'female'

# Chest Pain Type* Feature
# Value 0: typical angina
# Value 1: atypical angina
# Value 2: non-anginal pain
# Value 3: asymptomatic 
df['chest_pain_type'][df['chest_pain_type'] == 0] = 'typical angia'
df['chest_pain_type'][df['chest_pain_type'] == 1] = 'atypical angia'
df['chest_pain_type'][df['chest_pain_type'] == 2] = 'non-anginal pain'
df['chest_pain_type'][df['chest_pain_type'] == 3] = 'asymptomatic'


# fbs: (fasting blood sugar > 120 mg/dl)** ---> fasting_blood_sugar
#  1 = true
#  0 = false     
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'fasting blood sugar greater 120mg/dl'
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'fasting blood sugar less 120mg/dl'

        
# ** restecg: resting electrocardiographic results** rest_ecg
# Value 0: normal
# Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'
df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'
df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'

# ** exang: ** exercise_induced_angina
# exercise induced angina 
# 1 = yes
# 0 = no    
df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no exercise induced'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes exercise induced'       
        
# ** slope:  ** st_slope
# the slope of the peak exercise ST segment 
# Value 0: upsloping 
# Value 1: flat
# Value 2: downsloping
df['st_slope'][df['st_slope'] == 0] = 'upsloping'
df['st_slope'][df['st_slope'] == 1] = 'flat'  
df['st_slope'][df['st_slope'] == 2] = 'downsloping'  
        
# **thal:** thalassemia
# 0&1 = normal
# 2 = fixed defect
# 3 = reversable defect 
df['thalassemia'][df['thalassemia'] == 0] = 'normal'
df['thalassemia'][df['thalassemia'] == 1] = 'normal'
df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'  
df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'   


# In[ ]:


#Convert it to Object Type
df['sex'] = df['sex'].astype('object')
df['chest_pain_type'] = df['chest_pain_type'].astype('object')
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].astype('object')
df['rest_ecg'] = df['rest_ecg'].astype('object')
df['exercise_induced_angina'] = df['exercise_induced_angina'].astype('object')
df['st_slope'] = df['st_slope'].astype('object')
df['thalassemia'] = df['thalassemia'].astype('object')


# In[ ]:


df.describe()


# In[ ]:


df = pd.get_dummies(df, drop_first=True)


# In[ ]:


df.head()


# ## Model Building

# In[ ]:


cvsrc_all_svc = []
for i in range(1,500):
    clf = make_pipeline(StandardScaler(), SVC(C=i))
    cvsrc = cross_val_score(clf, df.iloc[:,:-1],df.iloc[:,-1], cv=10,scoring="roc_auc")
    cvsrc_all_svc.append(cvsrc.mean())

print("SVC Mean Score is {0}%".format(np.array(cvsrc_all_svc).mean()*100))


cvsrc_all_knn = []
for i in range(1,272):
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=i))
    cvsrc = cross_val_score(clf, df.iloc[:,:-1],df.iloc[:,-1], cv=10,scoring="roc_auc")
    cvsrc_all_knn.append(cvsrc.mean())

print("KNeighbors Mean Score is {0}%".format(np.array(cvsrc_all_knn).mean()*100))


cvsrc_all_rf = []
for i in range(1,100):
    clf = make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=i))
    cvsrc = cross_val_score(clf, df.iloc[:,:-1],df.iloc[:,-1], cv=10,scoring="roc_auc")
    cvsrc_all_rf.append(cvsrc.mean())

print("Random Forest Score is {0}%".format(np.array(cvsrc_all_rf).mean()*100))


# In[ ]:


print("SVC Best Score is {0}% for this {1}th value of C".format(np.array(cvsrc_all_svc).max(),np.array(cvsrc_all_svc).argmax()))


# In[ ]:


print("KNN Best Score is {0}% for this {1}th value of N neighbors".format(np.array(cvsrc_all_knn).max(),np.array(cvsrc_all_knn).argmax()))


# In[ ]:


print("Random Forest Best Score is {0}% for this {1}th value of max_depth".format(np.array(cvsrc_all_rf).max(),np.array(cvsrc_all_rf).argmax()))

