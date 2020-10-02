#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # About the dataset:
# 
# age - age in years
# 
# sex - (1 = male; 0 = female)
# 
# cp - chest pain type
#    
# trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# 
# chol - serum cholestoral in mg/dl
# 
# serum = LDL + HDL + .2 * triglycerides,
# above 200 is cause for concern
# 
# fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false),
# '>126' mg/dL signals diabetes
# 
# restecg - resting electrocardiographic results 
# 
# thalach - maximum heart rate achieved
# 
# exang - exercise(physical activity) induced angina (1 = yes; 0 = no)
# 
# oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
# 
# slope - the slope of the peak exercise ST segment
#    
# ca - number of major vessels (0-3) colored by flourosopy
# colored vessel means the doctor can see the blood passing through
# the more blood movement the better (no clots)
# 
# thal - thalium stress result
# 
# target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# # Import libraries and read the data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()


# # Check for null values

# In[ ]:


features = list(data.columns)
bool_data = data.isnull()

for feat in features:
    print(feat + ' ' + str(bool_data[feat].sum()))


# # Age and Sex vs Target

# In[ ]:


#male = 1 and female = 0.
sns.countplot(data['sex'])
plt.title('sex counts(male=1 and female=0)')


# In[ ]:


#Females are mostly older than males.
sns.boxplot(data['sex'], data['age'])


# In[ ]:


#How sex and age together affect the outcome.
sex_age = pd.DataFrame()

def fill_df_column(cols):
    sex = cols[0]
    target = cols[1]
    
    if sex == 0 and target == 0:
        return 'healthy_female'
    elif sex == 0 and target == 1:
        return 'diseased_female'
    elif sex == 1 and target == 0:
        return 'healthy_male'
    else:
        return 'diseased_male'
    
sex_age['sex_target'] = data[['sex', 'target']].apply(fill_df_column, axis = 1)
sex_age['age'] = data['age']
sex_age.head()


# In[ ]:


#old patients are healthier than the younger ones.
sns.boxplot(sex_age['sex_target'], sex_age['age'])


# In[ ]:


sex_counts = dict(data['sex'].value_counts())
diseased_counts = dict(data[data['target'] == 1]['sex'].value_counts())

print('Percentage of female diseased: ' + str((diseased_counts[0]*100)/(sex_counts[0])))
print('Percentage of male diseased: ' + str(round((diseased_counts[1]*100)/(sex_counts[1]), 2)))


# **Conclusion: If you are a female you are more likely to have the disease male are almost equally likely and the probablity decreases as the age.**

# # Less influencing features on the output

# In[ ]:


#blood pressure vs target
sns.boxplot(data['target'], data['trestbps'])


# In[ ]:


#resting cholestrol vs target
sns.boxplot(data['target'], data['chol'])


# In[ ]:


#blood sugar vs target
sns.countplot(data['target'], hue = data['fbs'])


# **The above three features are not that useful in predicting is a person has heart disease as we can see the plots for target = 0 as well as target = 1 look almost the same. THis means it doesnt matter what the value of this feature is you cannot really be sure if a person has heart disease or not by just looking at these three features.**

# # Chest Pain vs Target

# In[ ]:


#chest pain vs target.
sns.countplot(data['target'], hue = data['cp'])
plt.title('chest pain vs target')


# In[ ]:


#calculate the percentage.
diseased_counts = dict(data[data['target'] == 1]['cp'].value_counts())
cp_counts = dict(data['cp'].value_counts())

print('Percentage of people with heart disease and chest pain level 1: ' + str(round((diseased_counts[0]*100)/cp_counts[0], 2)))
print('Percentage of people with heart disease and chest pain level 2: ' + str(round((diseased_counts[1]*100)/cp_counts[1], 2)))
print('Percentage of people with heart disease and chest pain level 3: ' + str(round((diseased_counts[2]*100)/cp_counts[2], 2)))
print('Percentage of people with heart disease and chest pain level 4: ' + str(round((diseased_counts[3]*100)/cp_counts[3], 2)))


# **We see that people with chest pains of level 1 are very less likely to have heart disease, that is just 27% of patients with level 1 chest pain have a heart disease, than the people with chest pains of higher levels with percentages 82, 79, and 70.**

# # Chest pain experienced while physical work indicates heart disease?
# Its normal that as we do some physical work our heart pumps faster hence it will need more oxygen to do that task of pumping faster. That is why we feel tiredness and breath faster while we are doing some physical work. But lack of oxygen will cause chest pain. Does this indaicte heart disease?

# In[ ]:


sns.countplot(data['target'], hue = data['exang'])


# **We see majority of them donot experience chest pain during physical work. Even if some do experience very less out of them have a heart disease. Les's dive a bit deeper.**

# In[ ]:


cp_pw = data[data['exang'] == 1]
sns.countplot(cp_pw['cp'])
plt.title('chest pain level by physical work counts')


# In[ ]:


cp_counts = dict(cp_pw['cp'].value_counts())
print('Percentage of people having chest pain of level 1 due to physical work: ' + str(round(cp_counts[0]*100/sum(cp_counts.values()), 2)))


# **Why do we see majority of them experiencing chest pain during physical work donot have heart disease? This is because almost 81% of people experiencing chest pain due to physical work is of level 1. As we saw earlier just 27% of total patients having chest pain of level 1 have heart disease. This gives us the answer to the question.**

# In[ ]:


sns.countplot(cp_pw['target'], hue = cp_pw['cp'])


# In[ ]:


print('Percentage of people who experience chest pain during physical work and have heart disease: ' + str(round(cp_pw['target'].sum()*100/cp_pw.shape[0], 2)))


# **Thats reasonably a very less percentage!!**
# 
# **Conclusion is that a person is less likely to have a heart disease if the chest pain experienced by him is during some physical work.**

# In[ ]:


sns.boxplot(data['target'], data['thalach'])


# More the heart rate more is the chances of heart disease. This can answered logically.
# 
# In general, a resting adult heart beats between 60 and 100 times per minute. When an individual has tachycardia, the upper or lower chambers of the heart beat significantly faster.
# 
# When the heart beats too rapidly, it pumps less efficiently and blood flow to the rest of the body, including the heart itself, is reduced.
# 
# Because the heart is beating quicker, the muscles of the heart, or myocardium, need more oxygen. If this persists, oxygen-starved myocardial cells can die, leading to a heart attack.

# In[ ]:


#number of major blood vessels vs target
sns.countplot(data['target'], hue = data['ca'])


# Having more blood vessels is good right. As more major blood vessels will enhance the blood flow. This is why people with zero major blood vessel have very high chances of having heart disease.

# In[ ]:


ca_t1 = dict(data[data['target'] == 1]['ca'].value_counts())
ca_t0 = dict(data[data['target'] == 0]['ca'].value_counts())
print('Percentage of people with ca = 0 and have heart disease: ' + str(round(ca_t1[0]/(ca_t1[0] + ca_t0[0])*100, 2)))
print('Percentage of people with ca = 1 and have heart disease: ' + str(round(ca_t1[1]/(ca_t1[1] + ca_t0[1])*100, 2)))
print('Percentage of people with ca = 2 and have heart disease: ' + str(round(ca_t1[2]/(ca_t1[2] + ca_t0[2])*100, 2)))
print('Percentage of people with ca = 3 and have heart disease: ' + str(round(ca_t1[3]/(ca_t1[3] + ca_t0[3])*100, 2)))
print('Percentage of people with ca = 4 and have heart disease: ' + str(round(ca_t1[4]/(ca_t1[4] + ca_t0[4])*100, 2)))


# Note that the person examples with ca = 4 are outliers. As the percentage should have been decreased but its increased to 80%. This is a very rare case that the model may encounter in the future. Outliers have to be removed from the dataset as they may mislead the model and prevent them from learning accurately and faster. But here we have very less dataset so let's keep it.

# In[ ]:


sns.countplot(data['target'], hue = data['thal'])


# In[ ]:


sns.countplot(data['target'], hue = data['restecg'])


# # Feature Correlation

# In[ ]:


corr = data.corr()

plt.figure(figsize = (10, 10))
sns.heatmap(corr, cmap = 'YlGnBu', annot = True, fmt = '.2f')


# In[ ]:


print(corr['target'])


# **We can see cp, thalach, ca, thal, age, and sex are among the highly related features. Weak features are as already seen trestbps, chol, fbs.**

# # Train the model

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


labels = data['target']
train_data = data.drop('target', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size = 0.20)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=150)
rfc.fit(X_train, y_train)


# # Calculate the accuracy

# In[ ]:


rfc.score(X_test,y_test)


# Accuracy came out to be 83.6% using ensemble model (Random Forest) with 150 decision trees and testset of size 61 examples an train size of size 242 examples.

# # Precision, Recall, F1 Score, and Support

# In[ ]:


y_pred = rfc.predict(X_test)
print(classification_report(y_pred, y_test))


# # Confusion Matrix

# In[ ]:


cf_mat = confusion_matrix(y_pred, y_test)
sns.heatmap(cf_mat, cmap = 'YlGnBu', annot = True)


# In[ ]:




