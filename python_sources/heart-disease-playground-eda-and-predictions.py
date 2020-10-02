#!/usr/bin/env python
# coding: utf-8

# **Heart Disease Study**
# 
# This version of the *Heart Disease UCI* dataset consists of a subset of 14/76 attributes. Goal -> presence from 0 (no presence) to 4. The attributes are:
#         
#         1. Age
#         2. Sex (1 = male; 0 = female)
#         3. Chest pain type (4 values)
#         4. Resting blood pressure
#         5. Serum cholestoral in mg/dl
#         6. Fasting blood sugar > 120 mg/dl
#         7. Resting electrocardiographics results (values 0,1,2)
#         8. Maximum heart rate achieved
#         9. Exercise induced angina
#         10. Oldpeak = ST depression induced by exercise relative to rest
#         11. The slope of the peak exercise ST segment
#         12. Number of major vessels (0-3) colored by flourosopy
#         13. thal: 3 = normal; 6 = fixed defect; 7 = reverable defect

# 
# **Import libraries**

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# **Load the file - 'heart.csv'**

# In[ ]:


path = '../input'
heart_file = 'heart.csv'
file_path = os.listdir(path)
data_heart = pd.read_csv(os.path.join('../input',heart_file))


# **Data exploration**
#     
#         1. Dimensionality of DataFrame
#         2. Explore if there is any missing values
#         3. See the name of columns and separate features from target values
#         4. First five inputs/describe the data

# In[ ]:


data_heart.shape


# > 303 samples and 14 columns

# In[ ]:


data_heart.columns


# In[ ]:


for i in data_heart.index:
    if (data_heart.loc[i].isnull().sum() != 0):
        print('Missing value at ', i)
print('Done!')


# In[ ]:


data_heart_features = data_heart.loc[:,data_heart.columns!='target']
data_heart_target = data_heart.iloc[:,-1]


# In[ ]:


data_heart.head()


# In[ ]:


data_heart.describe()


# **Data visualization**
# 
#             1. Age distribution
#             2. Distribution of the sex
#             3. Distribution of the target
#             4. Representation of chest pain types

# In[ ]:


sns.distplot(data_heart['age'],hist=True,kde=True, 
             color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title('Age Distribution')


# In[ ]:


labels_dict_sex = {0:'Female',1:'Male'}
labels_sex = data_heart['sex'].value_counts().rename(index=labels_dict_sex).index
values_sex = data_heart['sex'].value_counts().values
colors_sex = ['#007ED6','#7CDDDD']
plt.pie(values_sex,explode=None,labels=labels_sex,colors=colors_sex,shadow=False,
        startangle=90,autopct='%.2f%%')
plt.axis('equal')
plt.tight_layout()
plt.title('Sex Distribution')


# In[ ]:


labels_dict_target = {0:'No Disease',1:'Disease'}
labels_target = data_heart['target'].value_counts().rename(index=labels_dict_target).index
values_target = data_heart['target'].value_counts().values
colors_target = ['#FF7300','#FFEC00']
plt.pie(values_target,explode=None,labels=labels_target,colors=colors_target,shadow=False,
        startangle=90,autopct='%.2f%%')
plt.axis('equal')
plt.tight_layout()
plt.title('Target Distribution')


# In[ ]:


labels_dict_cp = {0:'Type 1',1:'Type 2',2:'Type 3',3:'Type 4'}
labels_cp = data_heart['cp'].value_counts().rename(index=labels_dict_cp).index
values_cp = data_heart['cp'].value_counts().values
colors_cp = ['#007ED6','#7CDDDD','#FF7300','#FFEC00']
plt.bar(labels_cp,values_cp,color=colors_cp,align='center',alpha=0.8)
plt.xticks(labels_cp)
plt.ylabel('Amount')
plt.tight_layout()
plt.title('Chest Pain Types Representation')


# **Data exploration: relationship between features-features and features-target**
# 
#             Try too find those features that have the strongest relationship with the output variable, get the feature importance of each feature of the dataset (the higher the score the more relevant is the feature towards the output variable) and how the features are correlated to each other or the target value.
#                     1. Univariate selection
#                     2. Feature importance
#                     3. Correlation matrix with heatmap

# In[ ]:


data_heart_copy = data_heart.copy()


# In[ ]:


data_heart_copy['sex'].replace(1,'Male',inplace=True)
data_heart_copy['sex'].replace(0,'Female',inplace=True)
data_heart_copy['target'].replace(1,'Disease',inplace=True)
data_heart_copy['target'].replace(0,'No Disease',inplace=True)


# In[ ]:


sns.countplot(x=data_heart_copy['sex'],hue=data_heart_copy['target'])


# In[ ]:


data_heart_copy['cp'].replace(0,'Type 1',inplace=True)
data_heart_copy['cp'].replace(1,'Type 2',inplace=True)
data_heart_copy['cp'].replace(2,'Type 3',inplace=True)
data_heart_copy['cp'].replace(3,'Type 4',inplace=True)


# In[ ]:


sns.countplot(x=data_heart_copy['cp'],hue=data_heart_copy['sex'])


# *Univariate selection*

# In[ ]:


best_features = SelectKBest(score_func=chi2,k=10)
fit = best_features.fit(data_heart_features,data_heart_target)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(data_heart_features.columns)
feature_scores = pd.concat([df_columns,df_scores],axis=1)
feature_scores.columns = ['Features','Score']
print(feature_scores.nlargest(13,'Score'))


# *Feature importance*

# In[ ]:


model_trees = ExtraTreesClassifier()
model_trees.fit(data_heart_features,data_heart.target)
print(model_trees.feature_importances_)
feat_importances = pd.Series(model_trees.feature_importances_,index=data_heart_features.columns)
feat_importances.nlargest(13).plot(kind='barh')


# *Correlation Matrix with Heatmap*

# In[ ]:


correlation_matrix = data_heart.corr()
top_correlated_features = correlation_matrix.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data_heart[top_correlated_features].corr(),annot=True,cmap="RdYlGn")


# **Predictive Models**
#     
#             Data standardization: feature scaling to bring features to the same scale (not applicable to RandomForest)
#             Three models: SVM, K-NN, RandomForest
#             Features selected: cp, slope, exang, thal.

# In[ ]:


data_features = data_heart.loc[:,['cp','slope','exang','thal']]
data_target = data_heart.iloc[:,-1]


# In[ ]:


stdsc = StandardScaler()
data_features_std = stdsc.fit_transform(data_features)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(data_features,data_target,test_size=0.20,random_state=42)
print('Shape of X_train = '+str(X_train.shape)+'\n'+'Shape of X_test = '+str(X_test.shape)+'\n'
     +'Shape of y_train = '+str(y_train.shape)+'\n'+'Shape of y_test = '+str(y_test.shape))


# In[ ]:


knn = KNeighborsClassifier()
svclf = SVC()
random_forest = RandomForestClassifier()


# In[ ]:


knn.fit(X_train,y_train)
svclf.fit(X_train,y_train)
random_forest.fit(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)
svclf.score(X_test,y_test)
random_forest.score(X_test,y_test)


# In[ ]:


knn_cross_val = cross_val_score(knn,X_train,y_train,cv=5)
print('K-NN: ',knn_cross_val)
svclf_cross_val = cross_val_score(svclf,X_train,y_train,cv=5)
print('SVC: ',svclf_cross_val)
rf_cross_val = cross_val_score(random_forest,X_train,y_train,cv=5)
print('RandomForest: ',rf_cross_val)

