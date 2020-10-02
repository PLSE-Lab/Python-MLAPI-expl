#!/usr/bin/env python
# coding: utf-8

# # **![](http://)JanataHack: HR Analytics Challenge (1st Machine Learning Code)**
# 
# **Problem Statement: 
# **
# HR analytics is revolutionising the way human resources departments operate, leading to higher efficiency and better results overall. Human resources has been using analytics for years. However, the collection, processing and analysis of data has been largely manual, and given the nature of human resources dynamics and HR KPIs, the approach has been constraining HR. 
# 
# Therefore, it is surprising that HR departments woke up to the utility of machine learning so late in the game. This Janata Hack presents an opportunity to try predictive analytics in HR Domain, so gear up for another fun filled weekend
# 
# 
# **Approach:
# **
# 
# Since I have started learning Machine learning algorithm about a week ago, I had decided to take a plunge to experiment with real time challenge in one of the competition. Based on my limited learning, here is what I did in this competition and ended up at rank 286 out of 326. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#loading datasets
os.getcwd()
Test_data = pd.read_csv('/kaggle/input/test_KaymcHn.csv', sep=',')
Train_data = pd.read_csv('/kaggle/input/train_jqd04QH.csv', sep=',')


# In[ ]:


Test_data.head()


# In[ ]:


Train_data.head()


# In[ ]:


Test_data.info()


# In[ ]:


Test_data.isnull().sum()


# In[ ]:


Train_data.info()


# In[ ]:


Train_data.isnull().sum()


# In[ ]:


sns.countplot(Train_data['gender'])


# In[ ]:


print(Train_data['city'].unique())


# In[ ]:


print(Train_data['enrolled_university'].unique())


# In[ ]:


print(Train_data['city_development_index'].unique())


# In[ ]:


print(Train_data['gender'].unique())


# In[ ]:


print(Train_data['relevent_experience'].unique())


# In[ ]:


print(Train_data['enrolled_university'].unique())


# In[ ]:


print(Train_data['education_level'].unique())


# In[ ]:


print(Train_data['major_discipline'].unique())


# In[ ]:


print(Train_data['experience'].unique())


# In[ ]:


print(Train_data['company_size'].unique())


# In[ ]:


print(Train_data['company_type'].unique())


# In[ ]:


print(Test_data['last_new_job'].unique())


# In[ ]:


print(Train_data['training_hours'].unique())


# In[ ]:


print(Train_data['target'].unique())


# In[ ]:


#Understanding the gender break up - target wise 
sg_mode = Train_data.groupby(['target','gender'])
sg_mode.size()


# In[ ]:


# In both the categories, target 0 or 1: It is male is dominating gender and hence will replace all missing values with 'Male'
Train_data['gender']=np.where((Train_data['gender'].isnull()),"Male",Train_data['gender']) 
Train_data['gender']=np.where((Train_data['gender']=='Other'),"Male",Train_data['gender']) 
Test_data['gender']=np.where((Test_data['gender'].isnull()),"Male",Test_data['gender']) 
Test_data['gender']=np.where((Test_data['gender']=='Other'),"Male",Test_data['gender']) 


# In[ ]:


sg_mode = Train_data.groupby(['target','enrolled_university'])
sg_mode.size()


# In[ ]:


# It is dominated by students without any enrollment in both categories of output - target 0 or 1. 
Train_data['enrolled_university'].fillna('no_enrollment', inplace = True)
Test_data['enrolled_university'].fillna('no_enrollment', inplace = True)


# In[ ]:


sg_mode = Train_data.groupby(['target','education_level'])
sg_mode.size()


# In[ ]:


# It is dominated by students with graduate degree in both categories of output - target 0 or 1. 
Train_data['education_level'].fillna('Graduate', inplace = True)
Test_data['education_level'].fillna('Graduate', inplace = True)


# In[ ]:


sg_mode = Train_data.groupby(['target','major_discipline'])
sg_mode.size()


# In[ ]:


# It is dominated by students with STEM graduation degree in both categories of output - target 0 or 1. 
Train_data['major_discipline'].fillna('STEM', inplace = True)
Test_data['major_discipline'].fillna('STEM', inplace = True)


# In[ ]:


sg_mode = Train_data.groupby(['target','experience'])
sg_mode.size()


# In[ ]:


#around 59 entries has no fields under experience category. dropping these rows for now
Company_size = Train_data['company_size']
Train_data  = Train_data.dropna(subset =['experience','company_size','company_type','last_new_job'])


# In[ ]:


Company_size.fillna('50-99', inplace = True)

# Train_data['company_size'].fillna('50-99', inplace = True)
# Test_data['company_size'].fillna('50-99', inplace = True)


# In[ ]:


Train_data['last_new_job'].describe()


# In[ ]:


sg_mode = Train_data.groupby(['target','education_level'])
sg_mode.size()


# In[ ]:


#Preprocessing city development index into 4 major city categories - Metro, Tier 1, Tier 2, and Tier 3
bins = (0.25,0.5,0.75,1)
Test_data['city'].unique()
group_names = ['Tier 3','Tier 2', 'Tier 1', 'Metro']
Train_CDI = Train_data['city_development_index']
Train_data['city_development_index'] = pd.cut(Train_data['city_development_index'], 4, labels = group_names)
Test_data['city_development_index'] = pd.cut(Test_data['city_development_index'], 4, labels = group_names)
Test_data['city'].unique()


# In[ ]:


Train_data['city'] = Train_data['city_development_index']
Train_data['city_development_index'] = Train_CDI
Test_data['city'] = Test_data['city_development_index']
Test_data['city'].unique()


# In[ ]:


group_names = ['Low','Medium', 'High']
Train_data['training_hours'] = pd.cut(Train_data['training_hours'], 3, labels = group_names)
Test_data['training_hours'] = pd.cut(Test_data['training_hours'], 3, labels = group_names)


# In[ ]:


Train_data['city_development_index'].value_counts()


# In[ ]:


sns.countplot(Test_data['city'])


# In[ ]:


sns.countplot(Train_data['city_development_index'])


# In[ ]:


Train_data.drop(['city_development_index'],axis = 1, inplace = True)
Train_data.drop(['experience'],axis = 1, inplace = True)
Train_data.drop(['last_new_job'],axis = 1, inplace = True)


# In[ ]:


Train_data.info()


# In[ ]:


Train_data.info()


# In[ ]:


# We will now convert the nominal attributes to numbers to use it for dimensonality reduction
label_class= LabelEncoder()
Test_data ['city'] = label_class.fit_transform(Test_data['city'])
Test_data ['gender'] = label_class.fit_transform(Test_data['gender'])
Test_data ['relevent_experience'] = label_class.fit_transform(Test_data['relevent_experience'])
Test_data ['enrolled_university'] = label_class.fit_transform(Test_data['enrolled_university'])

Test_data ['education_level'] = label_class.fit_transform(Test_data['education_level'])

Test_data ['major_discipline'] = label_class.fit_transform(Test_data['major_discipline'])
Test_data.head()


# In[ ]:


# We will now convert the nominal attributes to numbers to use it for dimensonality reduction
label_class= LabelEncoder()
Train_data ['city'] = label_class.fit_transform(Train_data['city'])
Train_data ['gender'] = label_class.fit_transform(Train_data['gender'])
Train_data ['relevent_experience'] = label_class.fit_transform(Train_data['relevent_experience'])
Train_data ['enrolled_university'] = label_class.fit_transform(Train_data['enrolled_university'])

Train_data ['education_level'] = label_class.fit_transform(Train_data['education_level'])

Train_data ['major_discipline'] = label_class.fit_transform(Train_data['major_discipline'])
Train_data.head()


# In[ ]:


Y_train = Train_data['target']
features = ['city','gender','relevent_experience','enrolled_university','education_level','major_discipline','company_size','training_hours']
X_train = pd.get_dummies(Train_data[features])
X_test = pd.get_dummies(Test_data[features])


# In[ ]:


Train_data.info()


# In[ ]:


Test_data.info()


# In[ ]:


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

predict = pd.DataFrame({'enrollee_id': Test_data.enrollee_id, 'target': predictions})
predict.to_csv('Vikash_submission.csv', index=False)

#predict = pd.DataFrame(predictions, columns=['predictions']).to_csv ('predictions1.csv')
print("Your submission was successfully saved!")


# # Result
# 
# *Received score of 0.5011361030 with rank of 286 out of 326.
# *

# # Key Learnings from mistakes
# 
# * Dropped all NA values from train data sets and reduced accuracy in prediction 
# * Only tried RandomForestClassifier algorithm, no optimize done with other algorithm 
# * Should have combined the train data and test data for better prediction and to reduce the mean average error in final prediction 
# 
# 
