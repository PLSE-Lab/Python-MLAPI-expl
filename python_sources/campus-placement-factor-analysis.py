#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw_data=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


raw_data.head()


# In[ ]:


raw_data.info()


# In[ ]:


raw_data.describe()


# First fill NA in salary column to zero. anyways its a unnessaary step :P

# In[ ]:


raw_data['salary']=raw_data['salary'].fillna(0.00)


# * Will check for basic insight of Data. first we need to create dummy variables for important column

# In[ ]:


sns.countplot(x='gender',data=raw_data,hue='status')


# In[ ]:


#placemet ratio gender wise

male_ratio_placed=raw_data[(raw_data['gender']=='M') & (raw_data['status']=='Placed')]['sl_no'].count()/raw_data[raw_data['gender']=='M']['sl_no'].count()
male_ratio_placed


# In[ ]:


#female Ratio placed

female_ratio_placed=raw_data[(raw_data['gender']=='F') & (raw_data['status']=='Placed')]['sl_no'].count()/raw_data[raw_data['gender']=='F']['sl_no'].count()
female_ratio_placed


# In[ ]:


#Gender ratio
gender_ratio=raw_data[raw_data['gender']=='F']['sl_no'].count()/raw_data[raw_data['gender']=='M']['sl_no'].count()
gender_ratio


# In[ ]:


#gender 
print("gender \n",raw_data['gender'].value_counts())


# In[ ]:


#Types of SCC_boards/HSC Boards/hsc subjects/degree techs
print("ssc_b \n",raw_data['ssc_b'].value_counts())
print("\n \n")
print("hsc_b \n",raw_data['hsc_b'].value_counts())
print("\n \n")
print("hsc_s \n",raw_data['hsc_s'].value_counts())
print("\n \n")
print("degree_t \n",raw_data['degree_t'].value_counts())
print("\n \n")
print("specialisation \n",raw_data['specialisation'].value_counts())
print("\n \n")


# In[ ]:


raw_data.head()


# # Exploratory Data Analysis

# In[ ]:


sns.countplot(x='gender',data=raw_data,hue='status')


# In[ ]:


raw_data.columns


# In[ ]:


sns.countplot(x='ssc_b',data=raw_data,hue='status')


# In[ ]:


sns.countplot(x='hsc_b',data=raw_data,hue='status')


# In[ ]:


sns.countplot(x='degree_t',data=raw_data,hue='status')


# In[ ]:


sns.countplot(x='workex',data=raw_data,hue='status')
#This shows that more chances are there if you have a work experience


# In[ ]:



raw_data.columns


# In[ ]:


sns.distplot(raw_data[raw_data['status']=='Placed']['ssc_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['ssc_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Secondary Education Percentage")


# ## This make sense because there is higher chances if you have more marks defeinitely you are good in academics from start

# In[ ]:


sns.distplot(raw_data[raw_data['status']=='Placed']['hsc_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['hsc_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Higher Education Percentage")


# In[ ]:


sns.distplot(raw_data[raw_data['status']=='Placed']['degree_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['degree_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("Degree Education Percentage")


# In[ ]:


sns.distplot(raw_data[raw_data['status']=='Placed']['mba_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['mba_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("MBA Education Percentage")


# In[ ]:


sns.distplot(raw_data[raw_data['status']=='Placed']['etest_p'])
sns.distplot(raw_data[raw_data['status']=='Not Placed']['etest_p'])
plt.legend(["Placed", "Not Placed"])
plt.xlabel("etest_p Percentage")


# In[ ]:


sns.countplot(x='degree_t',data=raw_data,hue='status')


# In[ ]:


sns.countplot(x='specialisation',data=raw_data,hue='status')


# In[ ]:


#mapping the columns statyus to 0/1

raw_data['Placement_status']=raw_data['status'].map({'Placed':1,'Not Placed':0})


# In[ ]:


df=raw_data.copy()


# In[ ]:


df=raw_data.drop(['sl_no','status'],axis=1).copy()


# ### Creating dummies

# In[ ]:


gender=pd.get_dummies(df.gender,drop_first=True)
ssc_b=pd.get_dummies(df.ssc_b,drop_first=True,prefix='ssc_b')
hsc_b=pd.get_dummies(df.hsc_b,drop_first=True,prefix='hsc_b')
hsc_s=pd.get_dummies(df.hsc_s,drop_first=True,prefix='hsc_s')
degree_t=pd.get_dummies(df.degree_t,drop_first=True,prefix='degree_t')
workex=pd.get_dummies(df.workex,drop_first=True,prefix='workex')
spec=pd.get_dummies(df.specialisation,drop_first=True,prefix='mba_spec')


# In[ ]:


df_b4_dummies=df.copy()


# In[ ]:


df_after_dummies=df_b4_dummies.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation'],axis=1)


# In[ ]:


df_after_dummies=pd.concat([df_after_dummies,gender,ssc_b,hsc_b,hsc_s,degree_t,workex,spec],axis=1)


# In[ ]:


df_after_dummies.head() # Data before Dropping salary


# In[ ]:


data=df_after_dummies.drop('salary',axis=1).copy()


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True)


# In[ ]:


data.columns


# # Feature Selection

# We are selecting those having more correlation with the placement status
# 

# In[ ]:


data_affected=data[['ssc_p', 'hsc_p', 'degree_p','mba_p','etest_p','etest_p','workex_Yes', 'mba_spec_Mkt&HR','Placement_status']].copy()


# In[ ]:


data_affected.head()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# ##  Train test split

# In[ ]:


X=data_affected.drop('Placement_status',axis=1)
y=data_affected.Placement_status


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the model through decision Tree Classifier

# In[ ]:


dtree=DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# In[ ]:


pred_dtree=dtree.predict(X_test)


# ### Measuring through metrics

# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[ ]:


print(confusion_matrix(y_test,pred_dtree))
print("\n \n")
print(classification_report(y_test,pred_dtree))
print("\n \n")
print(accuracy_score(y_test,pred_dtree))


# In[ ]:


list(dtree.feature_importances_)


# ## Training the model through Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,predictions))
print("\n \n")
print(classification_report(y_test,predictions))
print("\n \n")
print(accuracy_score(y_test,predictions))


# ## Training the model through Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


RandForest=RandomForestClassifier()


# In[ ]:


RandForest.fit(X_train,y_train)


# In[ ]:


pred_rf=RandForest.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred_rf))
print("\n \n")
print(classification_report(y_test,pred_rf))
print("\n \n")
print(accuracy_score(y_test,pred_rf))


# ## Training the model through KNN

# In[ ]:


#importing a scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X_train=MinMaxScaler().fit_transform(X_train)


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


knn_pred=knn.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,knn_pred))
print("\n \n")
print(classification_report(y_test,knn_pred))
print("\n \n")
print(accuracy_score(y_test,knn_pred))


# ### We have analysis done for placement factor now. Let us discuss for salary anaysis 

# In[ ]:


df_b4_dummies.head()


# In[ ]:


df_salary=df_b4_dummies[df_after_dummies['salary']>0.0]
#df_salary=df_b4_dummies.copy()


# In[ ]:


df_salary.head()


# # Data Analysis for Salary Distribution

# In[ ]:


#Distribution of Salary


# In[ ]:


plt.hist(df_salary.salary)


# It shows us most people package varies from 2-6 lpa and 7 and 9 lpa are outliers

# In[ ]:


sns.kdeplot(df_salary.salary)


# ## Analysis based on gender 

# In[ ]:


#Analysis gender wise 

sns.kdeplot(df_salary[df_salary['gender']=="M"]['salary'])

sns.kdeplot(df_salary[df_salary['gender']=="F"]['salary'])
plt.legend(["Male","female"])
plt.xlabel("Salary")


# In[ ]:


sns.boxplot(x="salary",
    y="gender",
    data=df_salary)


# The above analysis shows that females salaries varies from male i.e. min pacakge is lesser for females and highest package as well

# ## Analysis based on SSC_b

# In[ ]:


df_salary.ssc_b.value_counts()


# In[ ]:


sns.kdeplot(df_salary[df_salary['ssc_b']=="Central"]['salary'])

sns.kdeplot(df_salary[df_salary['ssc_b']=="Others"]['salary'])
plt.legend(["Central","Others"])


# In[ ]:


sns.boxplot(x="salary",
    y="ssc_b",
    data=df_salary)


# In[ ]:


#Analysing hsc_b


# In[ ]:


df_salary.hsc_b.value_counts()


# In[ ]:


sns.kdeplot(df_salary[df_salary['hsc_b']=="Central"]['salary'])

sns.kdeplot(df_salary[df_salary['hsc_b']=="Others"]['salary'])
plt.legend(["Others",'Central'])


# In[ ]:


sns.boxplot(x="salary",
    y="hsc_b",
    data=df_salary,hue='gender')


# There is not too much difference for this factor only the highest package is higher for other boards

# ## Analysing based on hsc_s

# In[ ]:


df_salary.hsc_s.value_counts()


# In[ ]:


sns.kdeplot(df_salary[df_salary['hsc_s']=="Commerce"]['salary'])
sns.kdeplot(df_salary[df_salary['hsc_s']=="Science"]['salary'])
sns.kdeplot(df_salary[df_salary['hsc_s']=="Arts"]['salary'])
plt.legend(["Commerce","Science","Arts"])


# In[ ]:


sns.boxplot(x='hsc_s',y='salary',data=df_salary,hue='gender')


# Commerece students are getting better packages aand Arts student are getting least and Arts student pacakge variation is very less for high school 

# In[ ]:


#degree_t	workex	etest_p	specialisation	


# ## Analysis based on Degree trade

# In[ ]:


df_salary.degree_t.value_counts()


# In[ ]:


sns.kdeplot(df[df.degree_t=='Comm&Mgmt']["salary"])
sns.kdeplot(df[df.degree_t=='Sci&Tech']["salary"])
sns.kdeplot(df[df.degree_t=='Others']["salary"])
plt.legend(['Comm&Mgmt','Sci&Tech','Others'])


# In[ ]:


sns.boxplot(x='degree_t',y='salary',data=df_salary)


# undergraduate in Science &tech getting more salaries 

# ## Analysis based on work experience 

# In[ ]:


sns.kdeplot(df_salary[df_salary['workex']=='Yes']['salary'])
sns.kdeplot(df_salary[df_salary['workex']=='No']['salary'])
plt.legend(["workexp yes","workexp No"])


# In[ ]:


sns.boxplot(x='workex',y='salary',data=df_salary)


# work exp having can add upto higher package

# ## Analysis based on specialisation in MBA

# In[ ]:


df_salary['specialisation'].value_counts()


# In[ ]:


sns.kdeplot(df_salary[df_salary['specialisation']=="Mkt&Fin"]['salary'])
sns.kdeplot(df_salary[df_salary['specialisation']=="Mkt&HR"]['salary'])
plt.legend(["Mkt&Fin","Mkt&HR"])


# In[ ]:


#Average is almost same but Mkt&Fin can get more higer salary 


# In[ ]:


df_salary.head()


# ###  We ll do the quantitative analysis now i.e. we'll be analaysis salary on basis of numarical columns(ssc_p,hsc_p,degree_p,etest_p,mba_p)

# 

# In[ ]:


sns.pairplot(df_salary.drop('Placement_status',axis=1),kind='reg')


# We can see among numerical factors it depend on all 

# Now we ll apply linear regression to this dataset

# ## Feature engineering to apply Linear Regression 

# In[ ]:


df_salary.head()


# In[ ]:


dfsal_b4dummy=df_salary.drop('Placement_status',axis=1).copy()


# In[ ]:


dfsal_b4dummy.head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(dfsal_b4dummy.corr(),annot=True)


# In[ ]:


data=dfsal_b4dummy.copy()
data.head()


# #### Now we'll assign numerical values to catagorical variable to below columns
# **gender,
# ssc_b,
# hsc_b,
# hsc_s,
# degree_t,
# workex,
# specialisation**

# In[ ]:


data["gender"] = data.gender.map({"M":0,"F":1})
data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})
data["ssc_b"]=data.ssc_b.map({"Others":0, "Central":1})
data["hsc_b"]=data.hsc_b.map({"Others":0, "Central":1})
data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})
data["workex"] = data.workex.map({"No":0, "Yes":1})
data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})


# In[ ]:


data.head()


# In[ ]:


df_sal=data.copy()


# # Feature selection

# #### For feature selection 
# ##### 1. We need to remove outliers
# ##### 2. we need to get the most affecting features to the salary
# ##### 3. we also need to find the least significant variable  by removing which will increase our model accuracy.For that we should check the correlation of other features with salary feature  and we can keep improving our result on basis of metrics by keep removing or addding features into linear regression algorithm training. We should also check for outliers so iby removing it we can better optimize the result

# ### Removing outliers

# In[ ]:


plt.hist(df_sal['salary'])


# we can see the significant no of till 4 Lpa but after that we can see outliers.let see it in numerical way as well

# In[ ]:


len(df_sal[df_sal.salary>400000])# this means only 10 kids are outleiars we can remove it.


# In[ ]:


#df_sal=df_sal[df_sal.salary<400000]


# In[ ]:


df_sal.info()


# In[ ]:


df_sal.corr().transpose()['salary']


# we are going to drop every unrelevant column which have less correlartion with salary,later we can see and compare the result.

# In[ ]:


X=df_sal.drop(["salary",'hsc_p','ssc_p','ssc_b','workex','specialisation','degree_p','hsc_b'],axis=1)
#this is ourfinal feature selection on basis of chceking metrics by adding of removing features
y=df_sal['salary']
X.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[ ]:


X_scaled=scaler.fit_transform(X)


# In[ ]:


X_scaled=X_scaled[y <= 400000]
y=y[y <= 400000]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,random_state=41)


# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score,mean_absolute_error,r2_score


# In[ ]:


mean_absolute_error(y_test,reg.predict(X_test))


# In[ ]:


plt.scatter(y_test,reg.predict(X_test))
plt.xlabel('salary')
plt.ylabel('pred_sal')


# ### With Linear regression, this is the best output or regression model we can create 
# ### P.s. : We have started the Regression with more no of columns correlated with the salary finally we can reach to the 5 column which are
#     mba_p--> % in MBA
#     
#     etest_p---> E test score
#     
#     gender-->> Gender
#         
#     degree_t --> degree trade ( science/tech or Commerce etc)
#     
#     hsc_s ---> hogh school stream(science and others)
#        
#    
# 
#     
#     
#     
#     
#     

# **Hi guys,
# 
# I am one of the most new Data scientist aspirants and new to kaaggle >please comment below about what the things you like and area for improvement****

# In[ ]:




