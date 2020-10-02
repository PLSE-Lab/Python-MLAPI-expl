#!/usr/bin/env python
# coding: utf-8

# **Based on Quantitative Checklist for Autism in Toddlers (Q-CHAT) data provided by *ASD Tests* app I shall try to develop a simple prediction model for toddlers to predict probability of showing ASD traits so that their parents/guardians can consider taking steps early.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **We shall also explore ASD data for adults provided by ASD Tests app and gain some insights for that.**

# In[ ]:


df1 = pd.read_csv('../input/autism-screening/Autism_Data.arff',na_values='?')
df2 = pd.read_csv('../input/autism-screening-for-toddlers/Toddler Autism dataset July 2018.csv',na_values='?')


# In[ ]:


df1.head()


# In[ ]:


df1.info()


# In[ ]:


df2.head()


# In[ ]:


df2.info()


# **EDA**

# In[ ]:


sns.set_style('whitegrid')
data1= df1[df1['Class/ASD']=='YES']
data2= df2[df2['Class/ASD Traits ']=='Yes']
print("Adults: ",len(data1)/len(df1) * 100)
print("Toddlers:",len(data2)/len(df2) * 100)


# **Around 1% of the population has ASD, but for this sample we get around 27%  for Adults and 69% for Toddlers of the data with positive ASD. It's so because the test parameters features only qualitative properties of ASDs**

# In[ ]:


#Let's visualize the missing data
fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.heatmap(data1.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
ax[0].set_title('Adult dataset')
sns.heatmap(data2.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
ax[1].set_title('Toddlers dataset')


# In[ ]:


#Let's visualize the jaundice born child based on gender
fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.countplot(x='jundice',data=data1,hue='gender',ax=ax[0])
ax[0].set_title('ASD positive Adults born with jaundice based on gender')
ax[0].set_xlabel('Jaundice while birth')
sns.countplot(x='Jaundice',data=data2,hue='Sex',ax=ax[1])
ax[1].set_title('ASD positive Toddlers born with jaundice based on gender')
ax[1].set_xlabel('Jaundice while birth')


# We can see here almost 6-7 times more (in Adults) and 2-3 times more (in Toddlers) of non-jaundice born ASD positive whereas according to reports that is around 10 times. **Jaundice born child have a weak link with ASD.**
# 
# Also, **according to reports, ASD is more common among boys (around 4-5 times) than among girls.** But here in Adults we see a lower ratio, whereas in Toddlers its nearly 4 times boys than girls, which is quite close to actual ratio. 

# In[ ]:


#Let's see the age distribution of ASD positive
fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.distplot(data1['age'],kde=False,bins=45,color='darkred',ax=ax[0])
ax[0].set_xlabel('Adult age in years')
ax[0].set_title('Age distribution of ASD positive')
sns.distplot(data2['Age_Mons'],kde=False,bins=30,color='darkred',ax=ax[1])
ax[1].set_xlabel('Toddlers age in months')
ax[1].set_title('Age distribution of ASD positive')


# So for adults most of the ASD positive are around 20 or 30 years of age, whereas for toddlers most of them are around 36months. We can see in adults as the age increases the number decreases, whereas in toddlers as the age increases the number increases. It goes well with the research. **For adults, people with autism develop strategies to help them age better. For toddlers, the significant signs of autism reveals around 3 years of age.**

# In[ ]:


# Let's visualize positive ASD positive Adults based on top 15 countries
plt.figure(figsize=(20,6))
sns.countplot(x='contry_of_res',data=data1,order= data1['contry_of_res'].value_counts().index[:15],hue='gender',palette='viridis')
plt.title('Positive ASD Adults country wise distribution')
plt.xlabel('Countries')
plt.tight_layout()


# Even though the reach of the app affects this distribution, it does quite well describing the report. **Developed countries like UK,US, Australia,Canada indeed are the most affected ones.** But we see female population distinguishable compared to males, which is quite contrary.

# In[ ]:


#Lets see the ethnicity value counts
print(data1['ethnicity'].value_counts())
data2['Ethnicity'].value_counts()


# In the sample, **White and European ethnicities data overshadows the rest.** Which is quite close to studies done.

# In[ ]:


#Lets visualize the ASD distribution of Adult White and European ethnicity based on country
#We are considering both country and ethnicity because the reports suggests so.
plt.figure(figsize=(15,6))
sns.countplot(x='contry_of_res',data=data1[data1['ethnicity']=='White-European'],order=data1[data1['ethnicity']=='White-European']['contry_of_res'].value_counts().index[:10],palette='viridis')
plt.title('Positive ASD of White and European Ethnicities country wise distribution')
plt.xlabel('Countries')
plt.tight_layout()


# If we plot for other ethnicities in same country we shall get very low data plots. 
# 
# We can see that above graph compared to countrywise distribution looks very same for top 5 countries **US, UK, Australia, NZ and Canada thus affirming their positions as top contributors of Positive ASD.**

# In[ ]:


#Lets visualize the distribution of autism in family within different ethnicity
fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.countplot(x='austim',data=data1,hue='ethnicity',palette='rainbow',ax=ax[0])
ax[0].set_title('Positive ASD Adult relatives with Autism distribution for different ethnicities')
ax[0].set_xlabel('Adult Relatives with ASD')
sns.countplot(x='Family_mem_with_ASD',data=data2,hue='Ethnicity',palette='rainbow',ax=ax[1])
ax[1].set_title('Positive ASD Toddler relatives with Autism distribution for different ethnicities')
ax[1].set_xlabel('Toddler Relatives with ASD')
plt.tight_layout()


# We can observe that both in Adults and Toddlers, **White and Europeans Ethnicities have very high chance of being ASD positive if they have it in their genes**. Black and Asians follow the next though with smaller ratios. We can not conclude anything firmly, but **we can stay confident that there is a genetic link for ASD positive** as backed by studies.

# *Rest of the parameters are irrelevant for our study on ASD positive.*

# **We shall build three models using Logistic Regression,Random Forrest Classifier and K-NN Classifier on Toddlers Data.**

# **FEATURE ENGINEERING**

# In[ ]:


within24_36= pd.get_dummies(df2['Age_Mons']>24,drop_first=True)
within0_12 = pd.get_dummies(df2['Age_Mons']<13,drop_first=True)
male=pd.get_dummies(df2['Sex'],drop_first=True)
ethnics=pd.get_dummies(df2['Ethnicity'],drop_first=True)
jaundice=pd.get_dummies(df2['Jaundice'],drop_first=True)
ASD_genes=pd.get_dummies(df2['Family_mem_with_ASD'],drop_first=True)
ASD_traits=pd.get_dummies(df2['Class/ASD Traits '],drop_first=True)


# In[ ]:


final_data= pd.concat([within0_12,within24_36,male,ethnics,jaundice,ASD_genes,ASD_traits],axis=1)
final_data.columns=['within0_12','within24_36','male','Latino','Native Indian','Others','Pacifica','White European','asian','black','middle eastern','mixed','south asian','jaundice','ASD_genes','ASD_traits']
final_data.head()


# **MODEL BUILDING:**

# In[ ]:


from sklearn.model_selection import train_test_split
X= final_data.iloc[:,:-1]
y= final_data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)


# **1. Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,y_train)


# For better parameters we will apply GridSearch

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01,0.1,1,10,100,1000]}


# In[ ]:


grid_log = GridSearchCV(LogisticRegression(),param_grid,refit=True)


# In[ ]:


grid_log.fit(X_train,y_train)


# In[ ]:


grid_log.best_estimator_


# In[ ]:


pred_log=grid_log.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred_log))
print(classification_report(y_test,pred_log))


# **2. Random Forrest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc= RandomForestClassifier(n_estimators=500)
rfc.fit(X_train,y_train)


# In[ ]:


pred_rfc= rfc.predict(X_test)
print(confusion_matrix(y_test,pred_rfc))
print(classification_report(y_test,pred_rfc))


# **3. KNN Classifier**

# In[ ]:


#first scale the variables
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()


# In[ ]:


scaler.fit(X)
scaled_features = scaler.transform(X)


# In[ ]:


X_scaled = pd.DataFrame(scaled_features,columns=X.columns)
X_scaled.head()


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=101)


# Let's choose a k-value using elbow method

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
error_rate =[]

for i in range (1,50):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs K-value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# We choose k=13

# In[ ]:


knn= KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train,y_train)


# In[ ]:


pred_knn=knn.predict(X_test)
print(confusion_matrix(y_test,pred_knn))
print(classification_report(y_test,pred_knn))


# **Out of above three models KNN classifer and Random Forrest Classifier performs same overall but much better than Logistic Regression**

# **So based on the KNN classifier model on above dataset, if any parent provides toddler's age,gender,ethnicity,jaundice while birth? and any relative having ASD traits?, the model can predict either the toddler has ASD or not with precision of 71%**

# In[ ]:




