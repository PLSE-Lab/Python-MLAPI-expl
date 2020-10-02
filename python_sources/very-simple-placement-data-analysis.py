#!/usr/bin/env python
# coding: utf-8

#  Hello! All This  is my first kaggle notebook. I would request everyone to please go through this notebook completely
# and if you find any mistake please point it out and also if possible please provide a link to the related article.
# I am sure their is a lot of space for improvement
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score

sns.set()


# In[ ]:


Placement = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv",index_col= 'sl_no')


# In[ ]:


#  Checking for null and missing values 

print(Placement.describe())
print(Placement.isnull().sum())

Placement.info()


# In[ ]:


# Numerical conversion of categorical features

Placement.gender.replace({'M':1,'F':0},inplace=True)
Placement.workex.replace({'No':0 ,'Yes':1},inplace=True)
Placement.status.replace({'Not Placed':0 ,'Placed':1},inplace=True)
Placement.specialisation.replace({'Mkt&HR':0 ,'Mkt&Fin':1},inplace=True)


# In[ ]:


Placement_corr =  Placement.corr()
sns.heatmap(Placement_corr)


# In[ ]:


Placement_corr.status


# In[ ]:


plt.bar(Placement_corr.index,Placement_corr.status)


# In[ ]:


# Distribution of different fields.

fig , axis = plt.subplots(3,2)
fig.set_size_inches(10,10)

Placement_yes= Placement[Placement.status == 1]
Placement_No= Placement[Placement.status == 0]

sns.distplot(Placement_yes.ssc_p , ax= axis[0,0] , color = 'blue')
sns.distplot(Placement_yes.hsc_p , ax= axis[0,1], color = 'blue')
sns.distplot(Placement_yes.degree_p , ax= axis[1,0], color = 'blue')
sns.distplot(Placement_yes.workex ,kde=False, ax= axis[1,1], color = 'blue')
sns.distplot(Placement_yes.etest_p , ax= axis[2,0], color = 'blue')
sns.distplot(Placement_yes.mba_p , ax= axis[2,1], color = 'blue')


sns.distplot(Placement_No.ssc_p , ax= axis[0,0], color = 'red')
sns.distplot(Placement_No.hsc_p , ax= axis[0,1], color = 'red')
sns.distplot(Placement_No.degree_p , ax= axis[1,0],color = 'red')
sns.distplot(Placement_No.workex ,kde=False, ax= axis[1,1], color = 'red')
sns.distplot(Placement_No.etest_p , ax= axis[2,0],color = 'red')
sns.distplot(Placement_No.mba_p , ax= axis[2,1],color = 'red')
plt.show()


# In[ ]:


Placement.describe(include='object')


# In[ ]:


# Comparison of choice 

figure , axes = plt.subplots(2,2)
figure.set_size_inches(10,10)
    
sns.barplot(Placement.hsc_s.value_counts().index , Placement.hsc_s.value_counts(), ax = axes[0,0] )
sns.barplot(Placement.degree_t.value_counts().index , Placement.degree_t.value_counts() , ax=axes[0,1])
sns.barplot(Placement.specialisation.value_counts().index , Placement.specialisation.value_counts() , ax=axes[1,0])
sns.distplot(Placement_yes.etest_p , ax= axes[1,1], color = 'blue')
sns.distplot(Placement_No.etest_p , ax= axes[1,1],color = 'red')


# In[ ]:


Placement.hsc_s.value_counts().index


# In[ ]:


# Students who changed thier field of study after high school

# Commerse to Science
Placement[(Placement.hsc_s == 'Commerce') & (Placement.degree_t == 'Sci&Tech' )]

# Science to Commerce
Placement[(Placement.hsc_s == 'Science') & (Placement.degree_t == 'Comm&Mgmt' )]


# In[ ]:



fig1, axes = plt.subplots(2,2)
fig1.set_size_inches(10,10)
# Pie chart of Placed students

Mkt_Fin_Per = len(Placement_yes[Placement_yes.specialisation == 1])/len(Placement)*100
Mkt_HR_Per = len(Placement_yes[Placement_yes.specialisation == 0])/len(Placement)*100

Not_placed = 100 - (Mkt_Fin_Per + Mkt_HR_Per)

# Placed vs Not- Placed

Status = [len(Placement_yes),len(Placement_No)]
axes[0,0].pie(Status, autopct='%1.1f%%')
axes[0,0].set_title("Placed vs Unplaced")


Pie_Placed = [ Mkt_Fin_Per, Mkt_HR_Per , Not_placed ]
axes[0,1].pie(Pie_Placed, autopct='%1.1f%%')
axes[0,1].set_title("Mkt_Fin vs Mkt_HR vs Unplaced Percentage")

# Placement percentage according to specialisation

Mkt_Fin = len(Placement_yes[Placement_yes.specialisation == 1])/len(Placement[Placement.specialisation == 1])*100
print(Mkt_Fin)

Pie_Mkt_Fin = [ Mkt_Fin , 100-Mkt_Fin  ]
axes[1,0].pie(Pie_Mkt_Fin, autopct='%1.1f%%')
axes[1,0].set_title("Mkt_Fin Placed Percentage")


Mkt_HR = len(Placement_yes[Placement_yes.specialisation == 0])/len(Placement[Placement.specialisation == 0])*100
print(Mkt_HR)

Pie_Mkt_HR = [Mkt_HR , 100-Mkt_HR]
axes[1,1].pie(Pie_Mkt_HR, autopct='%1.1f%%')
axes[1,1].set_title("Mkt_HR Placed Percentage")

plt.show()

axes[0,0].legend()


# In[ ]:


# Getting Dummy variables
Dummy_variables = pd.get_dummies(Placement[['hsc_s','degree_t']])
Dummy_variables


# In[ ]:


# Droping orginal columns and adding dummy variables
Placement.drop(columns=['ssc_b','hsc_b','hsc_s','degree_t'],inplace=True)
Plac_dummy = pd.concat((Placement,Dummy_variables), axis =1)


# In[ ]:


# Separating dependent and Independent variable

Y = Plac_dummy.status
Plac_dummy.drop(columns=['status','salary'],inplace=True)

X = Plac_dummy.iloc[:,0:20]


# In[ ]:


# Train_test_split
x_train , x_test , y_train ,y_test = train_test_split(X,Y,test_size = 0.3 , random_state = 0)
Sc_x = StandardScaler()
x_train = Sc_x.fit_transform(x_train)
x_test= Sc_x.transform(x_test)


# In[ ]:


# importing Random Forest
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators=100,criterion = 'entropy', random_state = 1)
classifier.fit(x_train,y_train)

# Prediction
y_pred = classifier.predict(x_test)
# Confusion_matrix
cm = confusion_matrix(y_test , y_pred)
cm


# In[ ]:


# Accuracy of Model
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:




