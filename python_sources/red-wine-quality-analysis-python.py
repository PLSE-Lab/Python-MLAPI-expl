#!/usr/bin/env python
# coding: utf-8

# **Importing the libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# **Loading the dataset**

# In[ ]:


data = pd.read_csv("../input/wineQualityReds.csv")
data.head()


# In[ ]:


data.info()


# This gives a concise summary of the dataframe.
# We can see there are no null values in the dataframe, including 13 columns and 1599 entries.The dataset is already clean and tidy.

# In[ ]:


"""Dropping the first column containing the index values
which is of no use for us"""
data = data.iloc[:,1:]
print(data.head())


# Now we are left with 12 columns in our dataframe out of which 1 column contain the quality of wine and other 11 columns contain other properties such as acidity,pH, density etc.

# In[ ]:


data.describe()


# This is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values.

# In[ ]:


data.corr()['quality']


# We are trying to figure out the correlation of every other feature w.r.t the quality of the wine.

# ****Heatmap of correlation matrix****

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,linewidth=0.5,center=0,cmap='coolwarm')
plt.show()


# In[ ]:


plt.hist(data.quality,bins=6,alpha=0.5,histtype='bar',ec='black')
plt.title('Distribution of the Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()


# The above plot shows the distribution of the quality of the wine in the dataset, and represents that most of the wine is of average quality i.e. quality ranging from 5-7 . 

# In[ ]:


plt.figure(figsize=(8,5))
sns.barplot(data['quality'],data['pH'],palette="GnBu_d")
plt.show()


# From the barplot of qulaity vs ph we can visualize that there is a slight decrease in pH with the increase in quality of the wine. 

# In[ ]:


sns.boxplot(x='quality',y='pH',data=data,palette='GnBu_d')
plt.title("Boxplot of Quality and pH")
plt.show()


# In[ ]:


ax = sns.boxplot(x='quality',y='alcohol',data=data,palette='GnBu_d')
plt.title("Boxplot of Quality and Alcohol")
plt.show()


# The above plot shows the increase in the quality of wine with the increase in alcohol. The quality of the wine is directly related to the amount of alcohol in the wine. More the alcohol in the wine better will be the quality.

# In[ ]:


sns.boxplot(x="quality",y="residual.sugar",data=data,palette="GnBu_d")
plt.title("Boxplot of Quality and residual sugar")
plt.show()


# There is not much effect of the residual sugar on the quality of the wine.

# In[ ]:


sns.boxplot(x="quality",y="density",data=data,palette="GnBu_d")
plt.title("Boxplot of Quality and Density")
plt.show()


# Lower the density of wine better will be the quality of the wine.
# From the above boxplot we acn visualize that the quality of wine increases with decrease in density. 

# In[ ]:


sns.boxplot(x="quality",y="sulphates",data=data,palette="GnBu_d")
plt.title("Boxplot of Quality and Sulphates")
plt.show()


# The above plot represents that the quality of alcohol increases with the increase in the amount of sulphates in the wine.

# In[ ]:


sns.boxplot(x="quality",y="chlorides",data=data,palette="GnBu_d")
plt.title("Boxplot of Quality and Chlorides")
plt.show()


# From the above plot we can see that there is not even a slight change in the quality of wine for a particular amount of chlorides mixed in them.

# Effect of acidity on the quality of wine
# * Citric Acid
# * Volatile Acidity
# * Fixed Acidity

# In[ ]:


sns.boxplot(x="quality",y="citric.acid",data=data,palette="coolwarm")
plt.title("Boxplot of Quality and Citric Acid")
plt.show()


# The quality of the wine increses with increase in the amount of citric acid in the wine.

# In[ ]:


sns.boxplot(x="quality",y="volatile.acidity",data=data,palette="coolwarm")
plt.title("Boxplot of Quality and Volatile Acidity")
plt.show()


# From the above boxplot we can see that the quality of wine increases with the decrease in the amount of volatile acids.

# In[ ]:


sns.boxplot(x="quality",y="fixed.acidity",data=data,palette="coolwarm")
plt.title("Boxplot of Quality and Fixed Acidity")
plt.show()


# There is not much effect of fixed acidity on the quality of the wine.

# **Modeling**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X = data.iloc[:,:11].values
Y = data.iloc[:,-1].values

#Splitting the dataset into training and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Training using Logistic Regression
cl = LogisticRegression()
cl.fit(X_train,Y_train)

#Making confusion matrix
cm = confusion_matrix(Y_test,cl.predict(X_test))
print(cm)


# In[ ]:


#Applying SVM
from sklearn.svm import SVC
cl = SVC(kernel="rbf")
cl.fit(X_train,Y_train)

cm = confusion_matrix(Y_test,cl.predict(X_test))
print(cm)


# **Thank You**
