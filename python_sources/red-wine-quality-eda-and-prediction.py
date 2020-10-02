#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **Glimpse of the Data** 

# In[ ]:


train_df = pd.read_csv("../input/winequality-red.csv")
train_df.sample(5)


# *Is there null element in the data ?*

# In[ ]:


train_df.isnull().sum()


# *statistical summary, excluding NaN values*

# In[ ]:


train_df.describe()


# **Exploratory Data Analysis**
# 
# Correlations between different features 

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train_df.corr(),color = "k", annot=True)


# *The following features are relatively  correlated:*
# 
# total sulfur dioxide with free sulfur dioxide;
# fixed acidity with density and citric acid;
# alcohol with quality 
# 
# *The following features are inversely  correlated:*
# 
# fixed acidity with pH
# citric acid with pH and volatile acidity
# 
# 
# **Quality of the Wine**(from the Data)
# 

# In[ ]:


sns.countplot(x='quality', data=train_df)


# *Relations between fixed acidity and quality*
# 
# Different Plots

# In[ ]:


plt.figure(figsize=(15,5))
sns.swarmplot(x= "quality", y="fixed acidity" , data = train_df) 
plt.title('fixed acidity and quality')


# In[ ]:


plt.figure(figsize=(15,5))
sns.boxplot(x="quality", y="fixed acidity",   data=train_df )


# In[ ]:


train_df.groupby('quality')['fixed acidity'].mean().plot.line()
plt.ylabel("fixed acidity")


# *Relations between volatile acidity and quality*

# In[ ]:


plt.figure(figsize=(10,4))
sns.barplot(x="quality", y="volatile acidity",   data=train_df )


# In[ ]:


train_df.groupby('quality')['volatile acidity'].mean().plot.line()
plt.ylabel("volatile acidity")


# *Relation between quality and sulphates*

# In[ ]:


plt.figure(figsize=(10,4))
sns.barplot(x="quality", y="sulphates",   data=train_df )


# In[ ]:


train_df.groupby('quality')['sulphates'].mean().plot.line()
plt.ylabel("sulphates")


# In[ ]:


sns.boxplot(x="quality", y="sulphates",   data=train_df )


# *Realtion between quality and pH*

# In[ ]:


sns.boxplot(x="quality", y="pH",   data=train_df )


# In[ ]:


train_df.groupby('quality')['pH'].mean().plot.line()
plt.ylabel("pH")


# *Realtion between fixed acidity and pH*

# In[ ]:


sns.lmplot(x="fixed acidity", y="pH", data=train_df)


# 

# *Realtion between fixed acidity and citric acid*

# In[ ]:


sns.lmplot(y="fixed acidity", x="citric acid", data=train_df)


# **categorising wine quality**

# In[ ]:


reviews = []
for i in train_df['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
train_df['Reviews'] = reviews
trainX = train_df.drop(['quality', 'Reviews'] , axis = 1)
trainy = train_df['Reviews']


# **Different Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# *Standardize the Data*

# In[ ]:


scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size = 0.2, random_state = 42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
score = {}


# *RandomForestClassifier*

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predicted_rfc = rfc.predict(X_test)
print(classification_report(y_test, predicted_rfc))


# In[ ]:


rfc_conf_matrix = confusion_matrix(y_test, predicted_rfc)
rfc_acc_score = accuracy_score(y_test, predicted_rfc)
print(rfc_conf_matrix)
print(rfc_acc_score*100)
score.update({'Random_forest_classifier': rfc_acc_score*100})


# *LogisticRegression*

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
predicted_lr = lr.predict(X_test)
print(classification_report(y_test, predicted_lr))


# In[ ]:


lr_conf_matrix = confusion_matrix(y_test, predicted_lr)
lr_acc_score = accuracy_score(y_test, predicted_lr)
print(lr_conf_matrix)
print(lr_acc_score*100)
score.update({'logistic_regressor': lr_acc_score*100})


# **SVC**

# In[ ]:


svc =  SVC()
svc.fit(X_train, y_train)
predicted_svc = svc.predict(X_test)
print(classification_report(y_test, predicted_svc))


# In[ ]:


svc_conf_matrix = confusion_matrix(y_test, predicted_svc)
svc_acc_score = accuracy_score(y_test, predicted_svc)
print(svc_conf_matrix)
print(svc_acc_score*100)
score.update({'SVC': svc_acc_score*100})


# *Decision Tree Classifier*

# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
predicted_dt = dt.predict(X_test)
print(classification_report(y_test, predicted_dt))


# In[ ]:


dt_conf_matrix = confusion_matrix(y_test, predicted_dt)
dt_acc_score = accuracy_score(y_test, predicted_dt)
print(dt_conf_matrix)
print(dt_acc_score*100)
score.update({'DecisionTreeClassifier': dt_acc_score*100})


# *GaussianNB*

# In[ ]:


gb = GaussianNB()
gb.fit(X_train,y_train)
predicted_gb = gb.predict(X_test)
print(classification_report(y_test, predicted_gb))


# In[ ]:


gb_conf_matrix = confusion_matrix(y_test, predicted_gb)
gb_acc_score = accuracy_score(y_test, predicted_gb)
print(gb_conf_matrix)
print(gb_acc_score*100)
score.update({'GaussianNB': gb_acc_score*100})


# In[ ]:


model_acc = pd.DataFrame()
model_acc['Models'] = score.keys() 
model_acc['Accuracy'] = score.values()
model_acc


# **Comparison Between Different Classifier**

# In[ ]:


from matplotlib.pyplot import xticks
sns.lineplot(x='Models', y='Accuracy',data=model_acc)
xticks(rotation=90)

