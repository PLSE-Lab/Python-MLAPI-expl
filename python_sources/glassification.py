#!/usr/bin/env python
# coding: utf-8

# # Hey
# 
# ## Introduction
# I found this dataset from UCI and decided to improve my skills on it. 
# 
# I will be analyze and visualize the data and then I will apply some ML models on it.   Lastly, I will do comparisons and have conclusion. enjoy!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("../input/glass.csv")


# # EDA

# In[ ]:


data.head(5)     # it seems like columns are showing possible elements and rows are showing the amount of them. And Type column is or label


# In[ ]:


data.tail()     # I see that data is sorted by label.   Data needs shuffling before training and splitting.


# In[ ]:


data.info()          # all of the features are in float format and non-null which is perfect


# In[ ]:


data.describe()             # I see that some columns has generally big values (Si), while some has very small values (Fe).  Data needs normalization before training


# # Visual EDA

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(24,16))

plt.subplot(3,3,1)
sns.violinplot(data.Type,data.RI)

plt.subplot(3,3,2)
sns.violinplot(data.Type,data.Na)

plt.subplot(3,3,3)
sns.violinplot(data.Type,data.Mg)

plt.subplot(3,3,4)
sns.violinplot(data.Type,data.Al)

plt.subplot(3,3,5)
sns.violinplot(data.Type,data.Si)

plt.subplot(3,3,6)
sns.violinplot(data.Type,data.K)

plt.subplot(3,3,7)
sns.violinplot(data.Type,data.Ca)

plt.subplot(3,3,8)
sns.violinplot(data.Type,data.Ba)

plt.subplot(3,3,9)
sns.violinplot(data.Type,data.Fe)

plt.show()


# We see above that some Glass types has data points that has small differences 

# In[ ]:


plt.figure(figsize=(24,20))
sns.heatmap(data.corr(),annot=True,linecolor="white",linewidths=(1,1),cmap="winter")
plt.show()


# There are some highly correlated columns such as Ca and Rl

# In[ ]:


sns.pairplot(data=data,hue="Type",vars=["RI", "Na","Mg","Al","Si","K","Ca","Ba","Fe"])
plt.show()


# # PreProcessing

# Since we understand our dataset more, I would like to shuffle the whole dataset because I dont want to make my ML models to memorize first types mostly

# In[ ]:


data = data.sample(frac=1,random_state=22)


# In[ ]:


data.head()  # as you can see below, it has changed rows randomly (even indexes are same as what they corresponded before)


# I will separate my Features and the Labels

# In[ ]:


y = data.Type.values.reshape(-1,1)
data.drop(["Type"],axis=1,inplace=True)

x = data.values


# In[ ]:





# Since we dont have another csv file for testing, I will use the train_test_split from sklearn

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.3)


# In[ ]:


from sklearn.metrics import confusion_matrix


# # Classification Models

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_res_model = LogisticRegression(solver="newton-cg",max_iter=400,multi_class="multinomial",random_state=42)

log_res_model.fit(x_train,y_train.ravel())
y_pred = log_res_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Logistic Regression: ",log_res_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dec_tree_model = DecisionTreeClassifier(min_samples_split=4,random_state=42)
dec_tree_model.fit(x_train,y_train)
y_pred = dec_tree_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Decision Tree Classifier: ",dec_tree_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=600,random_state=42,max_leaf_nodes=36)
rfc_model.fit(x_train,y_train.ravel())
y_pred = rfc_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Random Forest Classifier: ",rfc_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(x_train,y_train.ravel())
y_pred = nb_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Naive Bayes: ",nb_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1,p=1)
knn_model.fit(x_train,y_train.ravel())
y_pred = knn_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of K Nearest Neighbors: ",knn_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


# In[ ]:


from sklearn.svm import SVC
svc_model = SVC(random_state=42,C=2)
svc_model.fit(x_train,y_train.ravel())
y_pred = svc_model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print("Score of Support Vector Machine: ",svc_model.score(x_test,y_test))

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cbar=False)
plt.show()


# # Conclusion
# 
# As you see above, even if I do tuning I cannot reach high accuracies. 
# I think the reason behind that is dataset itself. Because it has unbalanced + low amount of data    ( I just might be unsuccesful too :D )
# 
# However, Random Forest Classifier and Decision Tree Classifier made a relatively good job with this data compared to other models.
# 
# ** Thanks for visiting my kernel. I am a learner and I would be happy if you have any advice or comments **

# In[ ]:





# In[ ]:





# In[ ]:




