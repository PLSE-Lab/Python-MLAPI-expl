#!/usr/bin/env python
# coding: utf-8

# ![](http://www.excelenciasgourmet.com/sites/default/files/styles/slideshow_large/public/2017-12/vinos_0.jpg?itok=id8HxL6E)

# ## Introduction

# 
# **
# Hello kagglers!  Here is my first project. ** 
# 
# 
# * **In this data, I classified wine qualities into 3 categories as good, mid and bad.  Then, I explored the new data with data visualization libraries.** 
# 
# * **For prediction I used K-Nearest Neighbors, Support Vector Machine and Random Forest models.** 
# 
# * **For conclusion, I matched accuracy scores according to model prediction ratios**
# 

# > **Please leave me a comment and upvote the kernel if you liked at the end.**

# **Basic Imports**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Get The Data**

# In[ ]:


df = pd.read_csv("../input/winequality-red.csv")
df.head(3)


# **Classify The Quality**

# In[ ]:


quality = df["quality"].values
category = []
for num in quality:
    if num<5:
        category.append("Bad")
    elif num>6:
        category.append("Good")
    else:
        category.append("Mid")


# In[ ]:


#Create new data
category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([df,category],axis=1)
data.drop(columns="quality",axis=1,inplace=True)


# In[ ]:


data.head(3)


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# ___
# **Here I counted the number of each class and checked correlation of the columns**

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(data["category"],palette="muted")
data["category"].value_counts()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)


# **According to heatmap, we can focus on alcohol-quality and density-alcohol relations to get meaningful exploration**

# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x=df["quality"],y=df["alcohol"],palette="Reds")


# In[ ]:


plt.figure(figsize=(12,6))
sns.jointplot(y=df["density"],x=df["alcohol"],kind="hex")


# ** Setting features, labels and
# Encoding the categorical data**
# 
# **[](http://)(good=1, med=2, bad=3)**

# In[ ]:


X= data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y= labelencoder_y.fit_transform(y)


# ## Training and Testing Data
# **Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)


# **Scaling the data for optimise predictions**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ## Training the Model and Predicting the Test Data 
# 
# Now its time to train our models on our training data and predict each of them!

# ## Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred_svc =svc.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_svc))


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))


# ## K-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print(classification_report(y_test, pred_knn))


# ## Conclusion
# 
# **Time to match the results!**

# In[ ]:


conclusion = pd.DataFrame({'models': ["SVC","Random Forest","KNN"],
                           'accuracies': [accuracy_score(y_test,pred_svc),accuracy_score(y_test,pred_rfc),accuracy_score(y_test,pred_knn)]})
conclusion


# ## CHEERS!

# ![](http://media-cdn.tripadvisor.com/media/photo-s/10/28/86/6f/wine-cheers.jpg)

# **As a result, we can see Random Forest model has the best accurary ratio for predicting our wine quality!**
# 
# **On the other hand, we can evaluate the model. I will work on it as soon as possible too.**
# 
# **Thank you for your time and attention! Please leave me a comment and upvote the kernel if you liked.**
# 
# **Also I'm sorry for grammar mistakes.**
