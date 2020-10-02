#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries :

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv('../input/Iris.csv')


# In[3]:


df.head(5)


# In[385]:


# checking for NaN values :
df.isnull().sum()


# ## Explore the Data :

# In[386]:


plt.figure(figsize=(15,6))
plt.xlabel('SepalLength in Cm' ,fontsize = 12)
plt.ylabel('PetalLength in Cm' ,fontsize = 12)
sns.stripplot(x = 'SepalLengthCm', y = 'PetalLengthCm', data = df,size = 7,jitter = False,palette='cool')


# In[387]:


plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace = .25)
plt.subplot(2,2,1)
sns.boxplot(x="Species", y="PetalLengthCm", data=df,palette='winter')
plt.subplot(2,2,2)
sns.violinplot(x="Species", y="PetalLengthCm", data=df, size=6,palette='spring')
plt.subplot(2,2,3)
sns.boxplot(x="Species", y="PetalWidthCm", data=df,palette='winter')
plt.subplot(2,2,4)
sns.violinplot(x="Species", y="PetalWidthCm", data=df, size=6,palette='spring')


# In[388]:


plt.figure(figsize=(15,6))
plt.xlabel('SepalLength in Cm' ,fontsize = 12)
plt.ylabel('PetalLength in Cm' ,fontsize = 12)
sns.stripplot(x = 'SepalWidthCm', y = 'PetalWidthCm', data = df,size = 7,jitter = False,palette='spring')


# In[212]:


plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace = .25)
plt.subplot(2,2,1)
sns.boxplot(x="Species", y="SepalLengthCm", data=df,palette='winter')
plt.subplot(2,2,2)
sns.violinplot(x="Species", y="SepalLengthCm", data=df, size=6,palette='spring')
plt.subplot(2,2,3)
sns.boxplot(x="Species", y="SepalWidthCm", data=df,palette='winter')
plt.subplot(2,2,4)
sns.violinplot(x="Species", y="SepalWidthCm", data=df, size=6,palette='spring')


# In[213]:


sns.pairplot(df.drop("Id", axis=1), hue="Species", size=3,palette='cool')


# In[214]:


sns.heatmap(cbar=False,annot=True,data=df.corr(),cmap='spring')


# ## Data Preprocessing :

# In[297]:


from sklearn.model_selection import train_test_split
x = df.iloc[:,0:4].values
y = df.iloc[:,5].values


# In[389]:


# Encoding the Categorical Data :
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# In[299]:


# spliting our Data Set :
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state = 0)


# In[ ]:


print('xtrain : ')
print(xtrain)
print('ytrain : ')
print(ytrain)


# In[ ]:


print('xtest : ')
print(xtest)
print('ytest : ')
print(ytest)


# In[300]:


# Manage data at same scale :
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=scx.fit_transform(xtrain)
xtest=scx.transform(xtest)


# # Modeling :

# ### Logistic Regression :

# In[396]:


from sklearn.linear_model import LogisticRegression
logistic_regressor = LogisticRegression(random_state=0)
logistic_regressor.fit(xtrain,ytrain)


# In[397]:


log_predictions = logistic_regressor.predict(xtest)
log_predictions


# In[398]:


logistic_accuracy = logistic_regressor.score(xtest,ytest)
logistic_accuracy


# ### Support Vector Machines : 

# In[399]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(xtrain,ytrain)


# In[401]:


svc_predictions = svc.predict(xtest)
svc_predictions


# In[403]:


svc_accuracy = svc.score(xtest,ytest)
svc_accuracy


# ### Naive Bayes : 

# In[306]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(xtrain, ytrain)


# In[404]:


NB_predictions = NB.predict(xtest)
NB_predictions


# In[405]:


NB_accuracy = NB.score(xtest,ytest)
NB_accuracy


# ### Decision Tree : 

# In[357]:


from sklearn.tree import DecisionTreeClassifier
dec_tree_classifier = DecisionTreeClassifier()
dec_tree_classifier.fit(xtrain, ytrain)


# In[406]:


dec_tree_predictions = dec_tree_classifier.predict(xtest)
dec_tree_predictions


# In[407]:


dec_tree_accuracy = dec_tree_classifier.score(xtest,ytest)
dec_tree_accuracy


# ### Random Forest : 

# In[408]:


from sklearn.ensemble import RandomForestClassifier
ran_forest_classifier = RandomForestClassifier()
ran_forest_classifier.fit(xtrain, ytrain)


# In[409]:


rn_predictions = ran_forest_classifier.predict(xtest)
rn_predictions


# In[410]:


ran_forest_accuracy = ran_forest_classifier.score(xtest,ytest)
ran_forest_accuracy


# In[324]:


Models = ['Logistic Regression','Support Vector Machines','Naive Bayes','Decision Tree', 'Random Forest']


# In[411]:


Accuracy = []

score = [logistic_accuracy,svc_accuracy, NB_accuracy, dec_tree_accuracy, ran_forest_accuracy]

for i in score :
    Accuracy.append(round(i*100))


# In[412]:


Performance_of_Models = pd.DataFrame({'Model' : Models , 'Score' : Accuracy}).sort_values(by='Score', ascending=False)


# In[413]:


Performance_of_Models


# In[414]:


from sklearn.metrics import accuracy_score, confusion_matrix
matrix_1 = confusion_matrix(ytest, log_predictions) 
matrix_2 = confusion_matrix(ytest, svc_predictions) 
matrix_3 = confusion_matrix(ytest, rn_predictions) 


# In[415]:


df_1 = pd.DataFrame(matrix_1,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

df_2 = pd.DataFrame(matrix_2,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

df_3 = pd.DataFrame(matrix_3,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])


# In[444]:


plt.figure(figsize=(20,5))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,3,1)
plt.title('confusion_matrix(logistic regression)')
sns.heatmap(df_1, annot=True,cmap='Blues')
plt.subplot(1,3,2)
plt.title('confusion_matrix(Support vector machines)')
sns.heatmap(df_2, annot=True,cmap='Greens')
plt.subplot(1,3,3)
plt.title('confusion_matrix(Random forest)')
sns.heatmap(df_3, annot=True,cmap='Reds')
plt.show()


# In[ ]:




