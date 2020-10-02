#!/usr/bin/env python
# coding: utf-8

# **Adult Salary Income.**
#  In this kernel we shall explore the data and make predictions if an adult earns more than 50K per year.
# I've used tableau to explore the data using data viz.
# Hope you have fun.

# In[ ]:


#Libraries Required
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data= pd.read_csv("../input/adult.csv")
df= pd.DataFrame(data)


# In[ ]:


data.head()


# We do have some missing values here coded with **'?'**

# In[ ]:


data.info()


# We'll take care of the missing values a little later.

# In[ ]:


#Converting Categorical variables into Quantitative variables
print(set(data['occupation']))
data['occupation'] = data['occupation'].map({'?': 0, 'Farming-fishing': 1, 'Tech-support': 2, 
                                                       'Adm-clerical': 3, 'Handlers-cleaners': 4, 'Prof-specialty': 5,
                                                       'Machine-op-inspct': 6, 'Exec-managerial': 7, 
                                                       'Priv-house-serv': 8, 'Craft-repair': 9, 'Sales': 10, 
                                                       'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13, 
                                                       'Protective-serv': 14}).astype(int)



    


# In[ ]:


data['income'] = data['income'].map({'<=50K': 0, '>50K': 1}).astype(int)


# In[ ]:


data['sex'] = data['sex'].map({'Male': 0, 'Female': 1}).astype(int)


# In[ ]:


data['race'] = data['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 
                                             'Amer-Indian-Eskimo': 4}).astype(int)


# In[ ]:


data['marital.status'] = data['marital.status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 
                                                             'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4, 
                                                             'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)


# All the necessary variables have been converted. Now lets see what we can do with the missing values.

# In[ ]:


df.occupation.replace(0, np.nan, inplace=True)


# In[ ]:


print(df.shape)
df=df.dropna()
print(df.shape)


# In the above code I just dropped about 1800 null entries for the occupation column.

# In[ ]:


df.head(10)


# We're good to go.

# Lets now explore the data with few visualizations.

# In[ ]:


hmap = df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True);


# ![](https://image.ibb.co/ioZvjv/Screen_Shot_2017_08_11_at_10_34_40_PM.png)
# 
# ![](https://image.ibb.co/eokqHF/Screen_Shot_2017_08_11_at_9_01_56_PM.png)
# 
# ![](https://preview.ibb.co/jtg84v/Screen_Shot_2017_08_11_at_8_59_23_PM.png)
# 
# ![](https://image.ibb.co/dAACWa/Screen_Shot_2017_08_11_at_9_02_13_PM.png)
# 
# ![](https://image.ibb.co/ebUvjv/Screen_Shot_2017_08_11_at_9_03_35_PM.png)

# **Inferences:**
# * Married citizens with spouse have higher chances of earning more than those who're unmarried/divorced/widowed/separated.
# * Males on an average make earn more than females.
# * Higher Education can lead to higher income in most cases.
# * Asian-Pacific-Islanders and white are two races that have the highest average income.

# **Setting up the prediction model.** Lets start with **Decission Trees.**

# In[ ]:


#SETTING UP DECISSION TREES
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


X=df[['education.num','age','hours.per.week']].values
y= df[['income']].values

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.3, random_state=21, stratify=y)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predn=clf.predict(X_test)
print('The accuracy of the model is',metrics.accuracy_score(predn,y_test))


# 76.19% in our first attempt. Not bad. Lets keep trying.

# **Setting up a Support Vector Machine**

# In[ ]:


#SVM
from sklearn import svm

svc = svm.SVC(kernel='linear')

svc.fit(X_train, y_train)

y_pred=svc.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print(svc.score(X_test,y_test))


# 77.31%. I think we can do better.

# **Hyper-Parameter tuning with Grid Search CV**

# In[ ]:


#Tuning the model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


param_grid= {'n_neighbors': np.arange(1,80)}
knn = KNeighborsClassifier()
knn_cv=GridSearchCV(knn, param_grid, cv=5)
y = y.reshape(30718,)
knn_cv.fit(X, y)
print(knn_cv.best_params_)
print(knn_cv.best_score_)




# 79.08%. Great. Now lets setup an actual knn to crosscheck.

# In[ ]:


#KNN
model=KNeighborsClassifier(n_neighbors=78) 
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))


# Accuracy score of 79.48%. We're getting there.

# We had included only those attributes that were corelated with the income attribute. Lets now add an another attribute to our final model.

# In[ ]:


X1=df[['education.num','age','hours.per.week', 'capital.gain']].values
y1= df[['income']].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1 ,y1, test_size=0.3, random_state=21, stratify=y)

knn1=KNeighborsClassifier(n_neighbors=78) 
knn1.fit(X1_train,y1_train)
prediction=knn1.predict(X1_test)
print('The accuracy of the KNN1 is',metrics.accuracy_score(prediction,y1_test))


# An accuracy score 81.78% in our Knn model. 

# In[ ]:


from xgboost import XGBClassifier

X2=df[['education.num','age','hours.per.week', 'capital.gain']].values
y2= df[['income']].values

X2_train, X2_test, y2_train, y2_test = train_test_split(X2 ,y2, test_size=0.3, random_state=21, stratify=y)

# fit model no training data
xgbc = XGBClassifier()
xgbc.fit(X2_train, y2_train)
prediction2=xgbc.predict(X2_test)
print('The accuracy of the xGB is',metrics.accuracy_score(prediction2,y2_test))



# Final accuracy of 82.56%
