#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv("../input/heart.csv")


# In[3]:


data.head()


# In[4]:


sns.pairplot(data,hue='target')


# From the pairplot above, it can be taken noted that 'thalach'(maximum heart rate achieved),
# 'oldpeak'(ST depression induced by exercise relative to rest) and slope(the slope of the peak exercise ST segment)
# are able to segregate the 'target' values quite prominantly.
# To get a closer look, lets slice the data frame and look closer.

# In[5]:


data.columns


# In[6]:


sns.pairplot(data[['thalach','oldpeak','slope','target']],hue='target')


# Without digging down too much we can see that these 3 variables have quite a higher bearing on the output! Lets verify the results too!

# In[7]:


sns.heatmap(data.corr(),cmap='coolwarm')


# The heatmap shows that there is a higher correlation between the target and variables - cp(chest pain type),
# thalach(maximum heart rate achieved) and slope(the slope of the peak exercise ST segment). 

# Lets see if there is some pattern w.r.t gender and age.

# In[8]:


ax = sns.countplot(x='sex',hue='sex',data=data)
ax.set_xticklabels(["Female","Male"])
ax.legend(['Female','Male'])


# In[9]:


count = data.groupby('sex').count()['age']
count_male = data['sex'].value_counts()[1]
count_female = data['sex'].value_counts()[0]
from matplotlib.pyplot import pie
fig1, ax1 = plt.subplots()
ax1.pie([count_female/(count_female+count_male),count_male/(count_female+count_male)], labels=['Female','Male'],autopct='%1.1f%%')


# This shows in a much better way that in our dataset itself the incidence of heart disease is much more prevalent in males than females.

# In[10]:


hrt_prob_data = data[data['target']==1]
not_hrt_prob = data[data['target']==0]
sns.distplot(hrt_prob_data['age'],bins=10)


# The ones who are detected with a heart problem are mostly above 35 years of age, most of them around 50-60 age group.

# In[11]:


sns.distplot(hrt_prob_data['thalach'],hist=False, rug=True, label='Heart Problem')
sns.distplot(not_hrt_prob['thalach'],hist=False, rug=True, label='No Heart Problem')


# Shows that if the maximum heart rate achieved is less than 150bpm then the chances of having a heart problem is relatively less as compared to when it is greater than 150.

# In[12]:


ax1 = sns.distplot(hrt_prob_data['oldpeak'],hist=False, rug=True, label='Heart Problem',bins=10)
ax2 = sns.distplot(not_hrt_prob['oldpeak'],hist=False, rug=True, label='No Heart Problem',bins=10)
ax1.set_xlim(-1,4)
ax2.set_xlim(-1,4)


# In[13]:


print('mean of ST depression when no heart problem:',not_hrt_prob['oldpeak'].mean())
print('mean of ST depression when there is heart problem:',hrt_prob_data['oldpeak'].mean())


# Lets use K Nearest Neighbours to classify a condition as having a heart problem or not based on the given attributes-

# In[14]:


data.info()


# Lets assign the categorical columns a data type 'object' so as to get their dummy variables - 

# In[15]:


data['cp'] = data['cp'].astype('object')
data['fbs'] = data['fbs'].astype('object')
data['restecg'] = data['restecg'].astype('object')
data['exang'] = data['exang'].astype('object')
data['slope'] = data['slope'].astype('object')
data['thal'] = data['thal'].astype('object')


# In[16]:


data_categorical = pd.get_dummies(data, drop_first=True)
data_categorical.head()


# In[17]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_categorical.drop('target',axis=1))


# In[18]:


scaled_features = scaler.transform(data_categorical.drop('target',axis=1))


# In[19]:


feature_data = pd.DataFrame(scaled_features,columns=(data_categorical.drop('target',axis=1)).columns[:])


# In[20]:


feature_data.head()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X = feature_data
y = data_categorical['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# Using Elbow Method to find the most efficient K value

# In[24]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,marker='o')


# K=12 seems to be the most efficient!

# In[25]:


knn_12 = KNeighborsClassifier(n_neighbors=12)
knn_12.fit(X_train,y_train)
pred_12 = knn_12.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred_12))
print('\n')
print(classification_report(y_test,pred_12))


# Using Random Forest - 

# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))


# In[27]:


fig = plt.figure(figsize=(10,10))
important_feat = pd.Series(rfc.feature_importances_,index=X_train.columns)
sns.set_style('whitegrid')
important_feat.sort_values().plot.barh()
plt.title('Important Features')


# Hence, it shows that what we estimated from the pairplot was true for 'thalach' and 'old peak'. These 2 variables do have a bearing on the output. This makes sense too as 'max heart rate achieved' is the immediate, subjective state of the patient at the time of examination.'Oldpeak' i.e. the ST depression relates to the shape and direction of the ST segment. Upward or downward shifts can represent decreased blood flow to the heart from a variety of causes, including heart attack, spasms in one or more coronary arteries (Prinzmetal's angina), infection of the lining of the heart (pericarditis) or the heart muscle itself (myocarditis), an excess of potassium in the bloodstream, a heart rhythm problem, or a blood clot in the lungs (pulmonary embolism)." 
# 
# The above image also shows that 'ca' i.e. the number of colored blood vessels too have a high bearing on the output which has to be true because the more the blood can get to the heart the better the heart is functioning.
