#!/usr/bin/env python
# coding: utf-8

# # <font color=blue|red|green|pink|yellow>GURU</font>
# [aiguru.az](http://aiguru.az)
# 
# ## Analyze Cardio Disease - 70000 Patients
# ### Author : Ramin Hashimzade
# ### Location : Azerbaijan, Baku

# ### Datasset Description
# *   id                number
# *   agein             days
# *   gender            1 - women, 2 - men
# *   height            cm
# *   weight            kg
# *   ap_hiSystolic     blood pressure
# *   ap_loDiastolic    blood pressure
# *   cholesterol       1: normal, 2: above normal, 3: well above normal
# *   gluc              1: normal, 2: above normal, 3: well above normal
# *   smoke             whether patient smokes or not(0 = no, 1 = yes)
# *   alco              Binary feature (0 = no, 1 = yes)
# *   active            Binary feature (0 = passive life, 1 = active life)
# *   cardio            Target variable(0 = no, 1 = yes)

# # Import Library

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[ ]:


df = pd.read_csv("../input/data.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ### Dropping unnecessary columns

# In[ ]:


df.drop("id", axis=1, inplace = True)
df.drop("age_days", axis=1, inplace = True)


# In[ ]:


df.columns


# ### **Correlation between fields. As shown below cardio disease is more relative with AGE, WEIGHT and CHOLESTEROL**
# 
# 

# In[ ]:


plt.figure(figsize = (15,8))
sns.heatmap(df.corr(), linewidths=0.5)


# # **Search for Outliers**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,8))
df_c = df.loc[:,['cholesterol','gluc', 'smoke', 'alco', 'active']]
sns.countplot(x="variable", hue="value",data= pd.melt(df_c), ax=ax);


# In[ ]:


dfmelt = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc', 'smoke', 'alco', 'active'])
sns.catplot(x="variable", hue="value", col="cardio", data=dfmelt, kind="count");


# In[ ]:


plt.scatter('age_year', 'height', data=df, marker='o', color='green')


# In[ ]:


plt.scatter('age_year', 'weight', data=df, marker='o', color='blue')


# In[ ]:


plt.scatter('age_year', 'ap_hi', data=df, marker='o', color='red')


# In[ ]:


plt.scatter('age_year', 'ap_lo', data=df, marker='o', color='orange')


# ### Removing Outliers

# In[ ]:


df = df[df.height <= 200]
df = df[df.height >= 120]


# In[ ]:


df = df[df.weight <= 160]


# In[ ]:


df = df[df.ap_hi.between(0,500)]


# In[ ]:


df = df[df.ap_lo.between(0,2000)]


# # **Prepare data for training**

# #### Split columns to independent variables & target variable

# In[ ]:


X = df.iloc[:, :11]
y = df.iloc[:, 11]


# Split Rows to train & test data

# In[ ]:


# split data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# # **Training 1st Model - KNeighborsClassifier**

# In[ ]:


######################### KNeighborsClassifier = 64%
from sklearn.neighbors import KNeighborsClassifier
# K = 2 because of cardio (target value 0/1)
model = KNeighborsClassifier(n_neighbors=2).fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[ ]:


### Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("KNeighborsClassifier")
print(cm)
print('Accurancy: {:.0f}%'.format(model.score(X_test, y_test)*100))


# # **Training 2nd Model - Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[ ]:


### Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("DecisionTree")
print(cm)
print('Accurancy: {:.0f}%'.format(classifier.score(X_test, y_test)*100))


# # **Training 3rd Model - DecisionTree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 10)
model_tree.fit(X_train, y_train)
y_pred = model_tree.predict(X_test)


# In[ ]:


### Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("DecisionTree")
print(cm)
print('Accurancy: {:.0f}%'.format(model_tree.score(X_test, y_test)*100))


# # **Training 4th Model - RandomForestClassifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test).round(0)


# In[ ]:


### Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("DecisionTree")
print(cm)
print('Accurancy: {:.0f}%'.format(model_rf.score(X_test, y_test)*100))

