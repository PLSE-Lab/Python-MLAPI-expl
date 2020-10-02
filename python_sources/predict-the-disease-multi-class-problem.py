#!/usr/bin/env python
# coding: utf-8

# <img src="http://imgur.com/mLsI6kc.jpg" width="500">

# ## Table of Content:
#    * [Introduction](#intro)
#    * [Import Packages](#importpackages)    
#    * [Reading the CSV file](#readcsv)   
#    * [Exploratory Data Analysis](#eda)    
#    * [Data Visualization](#dv)  
#        * [PCA](#pca)   
#            *    [PCA for Data visualization](#pcadv)
#            *    [PCA for dimensionality reduction](#pcadr) 
#    * [Import Metrics](#importmetric)
#    * [Model Building](#model) 
#        * [Random Forest Classifier](#rfc)
#             * [Feature importance according to Random Forest Classifier](#featurerfc)   
#             * [Using Recursive Feature Elimination](#rferfc) 
#        * [Decision Tree Classifier](#dtc) 
#             * [Feature importance according to Decision Tree Classifier](#featuredtc)   
#             * [Using Recursive Feature Elimination](#rfedtc) 
#        * [Logistic Regression](#lr)   
#             * [Using Recursive Feature Elimination](#rfelr) 
#    * [Conclusion](#conclusion)             

# ## Introduction:
# ### The task is to predict the disease with high accuracy rate with the help of list of symptoms provided by patient.
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Import Packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import os


# ## Reading the CSV file

# In[ ]:


df0 = pd.read_csv("/kaggle/input/predict-the-disease/Training.csv")
print("Dataset with rows {} and columns {}".format(df0.shape[0],df0.shape[1]))
df0.head()


# In[ ]:


df1 = pd.read_csv("/kaggle/input/predict-the-disease/Testing.csv")
print("Dataset with rows {} and columns {}".format(df1.shape[0],df1.shape[1]))
df1.head()


# In[ ]:


df0.describe()


# ## Exploratory Data Analysis

# In[ ]:


df0.isnull().sum()


# #### Hence, there are no missing value.

# ### Distribution of TARGET variable : PROGNOSIS ie. total number of disease of each type

# In[ ]:


# Training data
df0.prognosis.value_counts()


# In[ ]:


# Testing data
df1.prognosis.value_counts()


# In[ ]:


df0.prognosis.value_counts(normalize=True)


# ### Thus, the Training dataset is balanced with respect to Target variable: Prognosis.

# In[ ]:


df1.prognosis.value_counts(normalize=True)


# ### Also, the Testing dataset is balanced too with respect to Target variable:Prognosis

# In[ ]:


plt.figure(figsize=(18,5))
sns.countplot('prognosis',data=df0)
plt.show()


# In[ ]:


X= df0.drop(["prognosis"],axis=1)
y = df0['prognosis']


# In[ ]:


X1= df1.drop(["prognosis"],axis=1)
y1 = df1['prognosis']


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(X.corr(),annot=True)


# ## Data Visualization

# ## PCA

# In[ ]:


# initializing the pca
from sklearn import decomposition
pca = decomposition.PCA()


# In[ ]:


# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(X)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)


# ### PCA for Data visualization

# In[ ]:


# attaching the label for each 2-d data point 
pca_data = np.vstack((pca_data.T, y)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "prognosis"))
sns.FacetGrid(pca_df, hue="prognosis", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


# #### From this we can easily classify the disease using PCA.

# ### PCA for dimensionality reduction

# In[ ]:


# PCA for dimensionality redcution (non-visualization)

pca.n_components = 132
pca_data = pca.fit_transform(X)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


# #### Although explained Variance is not a good metric for classification problem but it gives fair-enough idea about dataset.

# In[ ]:


# data prepararion
from wordcloud import WordCloud 
x2011 = df0.prognosis
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=800,
                          height=800
                         ).generate(" ".join(x2011))
plt.title('Diseases',size=30)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ### Import Metrics

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score


# ## Model Building

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Random Forest Classifier

# In[ ]:


rfc_mod = RandomForestClassifier(random_state=42, class_weight='balanced').fit(train_X, train_y)


# In[ ]:


y_pred_rfc = rfc_mod.predict(val_X)
y_pred_rfc


# In[ ]:


y_pred_rfc1 = rfc_mod.predict(X1)
y_pred_rfc1


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_rfc, val_y))
print('cross validation:',cross_val_score(rfc_mod, X, y, cv=3).mean())
print("F1 Score :",f1_score(y_pred_rfc,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_rfc))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_rfc))


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_rfc1, y1))
print('cross validation:',cross_val_score(rfc_mod, X, y, cv=3).mean())
print("F1 Score :",f1_score(y_pred_rfc,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_rfc))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_rfc))


# ### Feature importance according to Random Forest Classifier

# In[ ]:


importances=rfc_mod.feature_importances_
feature_importances=pd.Series(importances, index=train_X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(x=feature_importances[0:20], y=feature_importances.index[0:20])
plt.title('Feature Importance RFC Model',size=20)
plt.ylabel("Features")
plt.show()


# ## Using Recursive Feature Elimination

# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


rfe = RFE(rfc_mod, 20)
rfe.fit(train_X,train_y)


# In[ ]:


train_X.columns[rfe.support_]


# In[ ]:


colm = train_X.columns[rfe.support_]


# In[ ]:


rfc_mod.fit(train_X[colm],train_y)


# In[ ]:


y_pred_rfc2 = rfc_mod.predict(val_X[colm])
y_pred_rfc2


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_rfc2, val_y))
print("F1 Score :",f1_score(y_pred_rfc2,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_rfc2))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_rfc2))


# ### Decision Tree Classifier

# In[ ]:


dectre_mod = DecisionTreeClassifier(random_state=42, class_weight='balanced').fit(train_X,train_y)


# In[ ]:


y_pred_dectre = dectre_mod.predict(val_X)
y_pred_dectre


# In[ ]:


y_pred_dectre1 = dectre_mod.predict(X1)
y_pred_dectre1


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_dectre, val_y))
print("F1 Score :",f1_score(y_pred_dectre,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_dectre))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_dectre))


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_dectre1, y1))
print("F1 Score :",f1_score(y_pred_dectre1,y1,average = "weighted"))
print('Report:\n',classification_report(y1, y_pred_dectre1))
print('Confusion Matrix: \n',confusion_matrix(y1, y_pred_dectre1))


# ## Feature importance according to Decision Tree Classifier

# In[ ]:


importances=dectre_mod.feature_importances_
feature_importances=pd.Series(importances, index=train_X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,7))
sns.barplot(x=feature_importances[0:20], y=feature_importances.index[0:20])
plt.title('Feature Importance Decision Tree Model',size=20)
plt.ylabel("Features")
plt.show()


# ## Using Recursive Feature Elimination

# In[ ]:


rfe = RFE(dectre_mod, 20)
rfe.fit(train_X,train_y)


# In[ ]:


train_X.columns[rfe.support_]


# In[ ]:


col = train_X.columns[rfe.support_]


# In[ ]:


dectre_mod.fit(train_X[col],train_y)


# In[ ]:


y_pred_dectre2 = dectre_mod.predict(val_X[col])
y_pred_dectre2


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_dectre2, val_y))
print("F1 Score :",f1_score(y_pred_dectre2,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_dectre2))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_dectre2))


# ### Logistic Regression

# In[ ]:


logreg_mod = LogisticRegression(random_state=42, solver='lbfgs', class_weight='balanced').fit(train_X,train_y)


# In[ ]:


y_pred_logreg = logreg_mod.predict(val_X)
y_pred_logreg


# In[ ]:


y_pred_logreg1 = logreg_mod.predict(X1)
y_pred_logreg1


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_logreg, val_y))
print("F1 Score :",f1_score(y_pred_logreg,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_logreg))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_logreg))


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_logreg1, y1))
print("F1 Score :",f1_score(y_pred_logreg1,y1,average = "weighted"))
print('Report:\n',classification_report(y1, y_pred_logreg1))
print('Confusion Matrix: \n',confusion_matrix(y1, y_pred_logreg1))


# ## Using Recursive Feature Elimination

# In[ ]:


rfe = RFE(logreg_mod, 20)
rfe.fit(train_X,train_y)


# In[ ]:


train_X.columns[rfe.support_]


# In[ ]:


cols = train_X.columns[rfe.support_]


# In[ ]:


logreg_mod.fit(train_X[cols],train_y)


# In[ ]:


y_pred_logreg2 = logreg_mod.predict(val_X[cols])
y_pred_logreg2


# In[ ]:


for i in y_pred_logreg2:
    print(i)


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_logreg2, val_y))
print("F1 Score :",f1_score(y_pred_logreg2,val_y,average = "weighted"))
print('Report:\n',classification_report(val_y, y_pred_logreg2))
print('Confusion Matrix: \n',confusion_matrix(val_y, y_pred_logreg2))


# In[ ]:


y_pred_logreg3 = logreg_mod.predict(X1[cols])
y_pred_logreg3


# In[ ]:


for i in y_pred_logreg3:
    print(i)


# In[ ]:


print("Accuracy Score:", accuracy_score(y_pred_logreg3, y1))
print("F1 Score :",f1_score(y_pred_logreg3,y1,average = "weighted"))
print('Report:\n',classification_report(y1, y_pred_logreg3))
print('Confusion Matrix: \n',confusion_matrix(y1, y_pred_logreg3))


# ## Conclusion:
# * Logistic regression with one vs rest gives the best accuracy in both dataset ie Validation data and Testing Data.
# * Feature importance obtained from Feature importance according to Random Forest classifier is alomst same as features obtained from logistic regression using RFE.

# In[ ]:




