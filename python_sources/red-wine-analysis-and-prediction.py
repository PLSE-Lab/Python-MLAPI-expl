#!/usr/bin/env python
# coding: utf-8

# **<font size="5">Red Wine Analysis</font>**<br><br>
# Welcome to the journey of analysis of Wine Quality.<br>
# *Lets get started*

# **Please UPVOTE if it helps you.**

# <font size="4">Importing Libraries (all together)</font>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,accuracy_score


# <font size="4">Reading Data</font>

# In[ ]:


data=pd.read_csv('../input/winequality-red.csv')
# Reading Data 


# In[ ]:


data.head()
# Top 5 rows


# In[ ]:


data.info()
# Information about data types and null values


# In[ ]:


data.describe()
# Statistical Analysis


# In[ ]:


data.isnull().any()
# data[data.isnull()].count()


# <font size="4">Correlation</font>

# In[ ]:


corr=data.corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='magma',linecolor="black")
plt.title('Correlation between features');


# <font size="5">Output Variable</font>

# In[ ]:


data['quality'].value_counts()


# **<font size="5">Data Visualizations</font>**

# <font size="4">Count Plot</font>

# In[ ]:


sns.countplot(x='quality',data=data)
plt.title('Quality Variable Analysis')
plt.xlabel('Quality').set_size(20)
plt.ylabel('Frequency').set_size(20)
plt.show()


# <font size="4">Pair Plot</font>

# In[ ]:


sns.pairplot(data,plot_kws={'alpha':0.3})
# data.hist(bins=50,figsize=(15,15))


# <font size="4">Boxplot</font>

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(data['quality'],data['fixed acidity'])


# <font size="4">Point Plot</font>

# In[ ]:


plt.figure(figsize=(10,8))
sns.pointplot(data['quality'],data['pH'],color='grey')
plt.xlabel('Quality').set_size(20)
plt.ylabel('pH').set_size(20)


# <font size="4">Regression Plot</font>

# In[ ]:


sns.regplot('alcohol','density',data=data)


# In[ ]:


sns.regplot('pH','alcohol',data=data)


# In[ ]:


sns.regplot('fixed acidity','citric acid',data=data)


# In[ ]:


sns.regplot('pH','fixed acidity',data=data)


# <font size="4">Dividing output variable into groups so that it can be easily **Classified**</font>

# In[ ]:


bins=[0,4,7,10]
labels=['bad','acceptable','good']
data['group']=pd.cut(data.quality,bins,3,labels=labels)


# In[ ]:


data.head()


# In[ ]:


data['group'].value_counts()


# In[ ]:


sns.set(palette='colorblind')
sns.countplot(x='group',data=data)
plt.title('Group frequencies')
plt.xlabel('Quality group')
plt.ylabel('Frequency')
plt.show()


# **<font size="5">Prediction</font>**

# In[ ]:


X = data.iloc[:,:-2].values
y = data.iloc[:,-1].values


# <font size="4">LabelEncoder</font>

# In[ ]:


y_le = LabelEncoder()
y = y_le.fit_transform(y)


# <font size="4">Splitting Dataset</font>

# In[ ]:


pca = PCA(n_components=8)
x_new = pca.fit_transform(X)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(x_new,y,test_size=0.2,random_state=0)


# <font size="4">Standardization of features</font>

# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **<font size="5">SVM</font>**

# In[ ]:


classifier = SVC(kernel='linear')
classifier.fit(X_train,y_train)
knn_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,knn_pred))
print(accuracy_score(y_test,knn_pred))


# **<font size="5">k-NN Algorithm</font>**

# In[ ]:


classifier = KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
knn_pred=classifier.predict(X_test)
print(confusion_matrix(y_test,knn_pred))
print(accuracy_score(y_test,knn_pred))


# **<font size="5">Logistic Regression</font>**

# In[ ]:


classifier = LogisticRegression()
classifier.fit(X_train,y_train)
lr_pred = classifier.predict(X_test)
print(confusion_matrix(y_test,lr_pred))
print(accuracy_score(y_test,lr_pred))


# **<font size="5">Thank You</font>**

# **Please UPVOTE if it helps you.**

# In[ ]:





# In[ ]:




