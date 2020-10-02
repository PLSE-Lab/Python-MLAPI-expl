#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing python modules/libraries
import pandas as panda
import numpy as nump
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Importing Dataset .csv file

# In[2]:


dataset=panda.read_csv("../input/who_suicide_statistics.csv")
dataset.head(10)


# In[3]:


#finding total number of null values in attributes of dataset
print(dataset.isnull().sum())


# In[4]:


print(dataset.count())


# In[5]:



#removing null value rows from dataset and again checking number of null rows in dataset
dataset.dropna(inplace=True)
print(dataset.isnull().sum())


# In[6]:


print(dataset.head(10))
print(dataset.count())


# In[7]:


#printing information of dataset
dataset.info()


# In[8]:


#checking Age ranges
print(dataset.age.unique())


# In[9]:


dataset.describe()


# In[10]:


#checking country names
print(dataset['country'].unique())
print("Total Countries: ",len(dataset['country'].unique()))


# In[12]:


#total number of population of world and suicides per year
print(dataset.groupby(['year']).sum())


# In[11]:


#male vs female suicide comparison
print(dataset.groupby('sex')['suicides_no'].sum())
dataset.groupby('sex')['suicides_no'].sum().plot.bar(figsize=(10,5),title = "Male/Female")


# In[13]:


#Age wise Comparison
print(dataset.groupby('age')['suicides_no'].sum())
dataset.groupby('age')['suicides_no'].sum().sort_index().plot.bar(figsize=(15,5),title = "Agewise comparison")


# In[14]:


dataset[['country','suicides_no']].groupby(['country']).sum().plot(kind='bar',figsize=(50,10),label="Total Suicies Per country in following Year")


# In[15]:


#checking top 5 countries highest suicide rate.
c_suicides=dataset[['country','suicides_no']].groupby(['country']).sum()
c_suicides = c_suicides.reset_index().sort_values(by='suicides_no', ascending=False)
top_5 = c_suicides[:5]
top_5.plot(kind='bar',figsize=(15,5),title="Top 5 countries with highest Suicide Rate",x='country',y='suicides_no')


# In[16]:


#Checking total number of suicides per year
dataset[['year','suicides_no']].groupby(['year']).sum().plot(kind='bar',figsize=(15,7),title="Suicide Rate")


# In[18]:


#Preprocessing, Converting AGE Groups into integer values and Gender ~ Female=0,Male=1 for model testing and predictions
mydata=dataset
mydata=mydata.drop(['country'],axis=1)#removed country column from 'mydata'
#replacing age groups with integer values
mydata['age']=mydata['age'].replace('5-14 years',0)
mydata['age']=mydata['age'].replace('15-24 years',1)
mydata['age']=mydata['age'].replace('25-34 years',2)
mydata['age']=mydata['age'].replace('35-54 years',3)
mydata['age']=mydata['age'].replace('55-74 years',4)
mydata['age']=mydata['age'].replace('75+ years',5)
mydata['sex']=mydata['sex'].replace('female',0)
mydata['sex']=mydata['sex'].replace('male',1)

#adding two more columns for model training and testing

mydata['suicides/100k_population']=(mydata.suicides_no/mydata.population)/100000
mydata['fatality_rate']=nump.where(mydata['suicides/100k_population']>mydata['suicides/100k_population'].mean(),0,1)




# In[19]:


#Dividing dataset into training and testing sets
X = nump.array(mydata.drop(['fatality_rate', 'suicides/100k_population'], 1))
y=nump.array(mydata.fatality_rate)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn import utils
from sklearn import preprocessing
from sklearn import metrics
#Encoing labels for continous and multi class output
label_enc=preprocessing.LabelEncoder()
trs=label_enc.fit_transform(y_train)


print("Shape of x_train: ",X_train.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of x_test: ",X_test.shape)
print("Shape of y_test: ",y_test.shape)


# In[20]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
print('Logistic Regression : {:.3f}'.format(accuracy_score(y_test,logreg.predict(X_test))))


# In[21]:


#Checking DecisionTreeClassifier Model and its accuracy
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,trs)
model.predict(X_test)

print('Decision Tree ',accuracy_score(y_test,model.predict(X_test)))


# In[22]:


#checking RandomForestClassifier Model and accuracy
from sklearn.ensemble import RandomForestClassifier
mod=RandomForestClassifier(n_estimators=100,random_state=42)
mod.fit(X_train,trs)
mod.predict(X_test)
print("Random Forest accuracy is: ",accuracy_score(y_test,mod.predict(X_test)))


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(logreg.predict(X_test),y_test))


# In[25]:


#classification report DecisionTreeClassifier

print(classification_report(mod.predict(X_test),y_test))


# In[26]:


#classification report RandomForest

print(classification_report(model.predict(X_test),y_test))

