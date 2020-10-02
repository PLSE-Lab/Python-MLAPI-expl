#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 
# 
# Input dataset
# 
# Check the amount of data and features:

# In[ ]:


train_data = pd.read_csv('../input/Churn_Modelling.csv')
print("amount of data:",train_data.shape[0])
print("amount of feature",train_data.shape[1])


# Check 'class imbalance':
# 
# 1. The graph shows that negative case(y=0) is much more than positive case. So 'Precision-recall curve' might be a more exact way to evaluate the result of algorithme, rather than 'accuracy'.
# 2. For the learning model, we can change our threshold from 0.5 to the real radio of two case or we can use random forest or xgboost algorithme(collective learning)

# In[ ]:


train_data['Exited'].astype(int).plot.hist()


# Delete some features:
# 
# We can drop feature 'Surname', 'RowNumber', 'CustomerId'
# 
# Obviously, these feature can not take any influence on the final result

# In[ ]:


train_data = train_data.drop(['Surname','RowNumber','CustomerId'], axis=1)
train_data


# Check missing data:
# 
# Fortunately, there isn't any missing data.

# In[ ]:


count_miss_val = train_data.isnull().sum()
count_miss_val.head(10)


# Check the non-value feature:
# 
# It turns out there are two label-features, 'geography', 'gender'
# 
# Apparently, all the label-features are nominal

# In[ ]:


train_data.dtypes.value_counts()
train_data.select_dtypes(include = ['object']).apply(pd.Series.nunique, axis = 0)


# For the feature 'Gender', which have 2 optional values, we can use method 'labelEncoder' to fit it as 0-1 value
# 
# For the feature Geography', which have 3 optional values, we convert categorical variable into dummy variables

# In[ ]:


le = LabelEncoder()
le.fit(train_data['Gender'])
train_data['Gender'] = le.transform(train_data['Gender'])
train_data = pd.get_dummies(train_data)
print("amount of training data: %d, amount of features:%d"% (train_data.shape[0],train_data.shape[1]))

Now we can check grossly the distribution of data.
# In[ ]:


train_data['EstimatedSalary'].describe()
train_data['Balance'].describe()
train_data['Age'].describe()


# We can calculate the correlation between 'Exited' and other features, sort the result and figure out the feature has high correlation to 'Exited'
# 
# It turns out feature 'Age', 'Geography_Germany ', 'Balance' have correlation positive with 'Exited' but 'IsActiveMember' have correlation negative.

# In[ ]:


correlations = train_data.corr()['Exited'].sort_values()
correlations


# As a feature that has the highest correlation to feature 'Extied', we can study on 'Age' feature more deeply.
# 
# Firstly, we check the distribution of value 'Age'

# In[ ]:


plt.hist(train_data['Age'], edgecolor = 'k', bins = 25)
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')


# After the kernel density estimation, it turns out that for the old people maybe more likely classified to class 'exited'

# In[ ]:


plt.figure(figsize = (10, 6))
sns.kdeplot(train_data.loc[train_data['Exited'] == 0, 'Age'], label = 'Exited = 0')
sns.kdeplot(train_data.loc[train_data['Exited'] == 1, 'Age'], label = 'Exited = 1')
plt.xlabel('Age')
plt.ylim(0, 0.06)
plt.ylabel('Density')
plt.title('KDE Distribution of Ages')


# Secondly, we regroup our data by age of clients

# In[ ]:


train_data["Age"].describe()
age_data = train_data[['Exited', 'Age']]
age_data['Age'] = pd.cut(age_data['Age'], bins = np.linspace(18, 93,num = 15))
age_groups = age_data.groupby('Age').mean()
age_groups


# Now we can check which group is most likely to be classifed as 'Exited'
# 
# It turns out that clients between 45 and 60 are most likely to exit (possibility more than 50%)

# In[ ]:


plt.figure(figsize = (10,4))
plt.bar(age_groups.index.astype(str), 100 * age_groups['Exited'])
plt.xticks(rotation = 45)
plt.xlabel('Age Group (years)')
plt.ylabel('Exited (%)')
plt.title('Exited by Age Group')


# Now we can discover the relation between different features by 'Correlation Heatmap'
# 
# It shows that feature 'Gemmany' is highly related with feature 'balance'

# In[ ]:


ext_data = train_data[['Exited', 'Age', 'Geography_Germany', 'IsActiveMember', 'Balance']]
ext_data_corrs = ext_data.corr()
plt.figure(figsize = (7,7))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.6, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')


# In addition, we can also create some polynomial features. But created features seem to have not better relations with ours target. So, it's not very necessary to add some polynomial features into ours dataset.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
ptrain_data = train_data[['IsActiveMember','Gender','Geography_France','Balance','Geography_Germany','Age']]
ptarget = train_data['Exited']
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(ptrain_data)
ptrain_data = poly_transformer.transform(ptrain_data)
poly_features_names = poly_transformer.get_feature_names(input_features = ['IsActiveMember','Gender','Geography_France','Balance','Geography_Germany','Age'])
poly_features = pd.DataFrame(ptrain_data, columns = poly_features_names) 
poly_features['TARGET'] = target
poly_corrs = poly_features.corr()['TARGET'].sort_values()
plt.figure(figsize = (10, 50))
poly_corrs.plot(kind='barh')


# We have a tool to adjust dataset automatically, but the result haven't any change relative to our original dataset.

# In[ ]:


import featuretools as ft
auto_train_data = train_data.copy()
es = ft.EntitySet(id = 'train_data') 
es = es.entity_from_dataframe(entity_id = 'train_data', dataframe = auto_train_data, index = 'SK_ID_CURR')
auto_train_data, features = ft.dfs(entityset = es,target_entity='train_data',verbose=True)
auto_train_data


# Input preprocessing models

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Split our data as two separated group, training set and test set
# 
# Turn our training data under standerd scale

# In[ ]:


target = train_data['Exited']
train = train_data.drop(['Exited'], axis = 1)
X_std = StandardScaler().fit_transform(train)
X_std_train,X_std_test,y_train, y_test = train_test_split(X_std, target, test_size = 0.2, random_state = 0)


# Input learning models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier


# I create some functions to evaluate our models

# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc,confusion_matrix
def model_metrics(clf, X_train, X_test, y_train, y_test):    
    y_train_pred = clf.predict(X_train)    
    y_test_pred = clf.predict(X_test)        
    y_train_prob = clf.predict_proba(X_train)[:,1]    
    y_test_prob = clf.predict_proba(X_test)[:,1]
    y_scores = clf.predict_proba(X_test)[:,-1]
    print(classification_report(y_train,y_train_pred, target_names=['non-exited','exited']))
    print('Accurancy:')    
    print('Train set: ','%.4f'%accuracy_score(y_train,y_train_pred), end=' ')    
    print('Test set: ','%.4f'%accuracy_score(y_test,y_test_pred),end=' \n\n')
    acu_curve(y_test,y_scores)
def acu_curve(y_test,y_scores):
    fpr,tpr,threshold = roc_curve(y_test,y_scores) 
    roc_auc = auc(fpr,tpr) 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# Fit five models: logistic regression, support vector machine, decision tree, xgboost, lightgboost, 

# In[ ]:


lr = LogisticRegression()
lr.fit(X_std_train,y_train)
svm = SVC(gamma='scale',probability=True)
svm.fit(X_std_train,y_train)
y_scores = svm.fit(X_std_train,y_train).decision_function(X_std_test)
tree = DecisionTreeClassifier()
tree.fit(X_std_train,y_train)
xgb = XGBClassifier()
xgb.fit(X_std_train,y_train)
lgbm = LGBMClassifier()
lgbm = lgbm.fit(X_std_train,y_train)


# Compare the abilities of different models. We can take surface of curve ROS as most important indice.
# Obviously, the model xgbooster have the best quality.

# In[ ]:


print("Logist regression")
model_metrics(lr,X_std_train,X_std_test,y_train,y_test)
print("Decision tree")
model_metrics(tree,X_std_train,X_std_test,y_train,y_test)
print("Svm")
model_metrics(svm,X_std_train,X_std_test,y_train,y_test)
print("Xgb")
model_metrics(xgb,X_std_train,X_std_test,y_train,y_test)
print("Lgbm")
model_metrics(lgbm,X_std_train,X_std_test,y_train,y_test)


# Then, we can use GridSearchCV to adjust parameters like 'learning_rate', 'max_depth', 'min_child_weight'
# 'gamma', 'subsample', 'colsample_bytree'.

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_test1 = {
 'max_depth':range(3,10,3),
 'min_child_weight':range(1,6,3)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(  learning_rate =0.1, n_estimators=100, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,  scale_pos_weight=1, seed=27), 
 param_grid = param_test1,     scoring='roc_auc', n_jobs=4, iid=False, cv=2)
gsearch1.fit(X_std_train,y_train)


# Finally, after the grid search, we can determine the optimum parameters and fit the best model

# In[ ]:


xgc = XGBClassifier(  learning_rate =0.1, n_estimators=100, max_depth=4, min_child_weight=7, reg_alpha=0.001, reg_lambda = 0,
                    gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4,  scale_pos_weight=1, seed=27)

xgc.fit(X_std_train,y_train)
y_scores = xgc.predict_proba(X_std_test)[:,-1]
fpr,tpr,threshold = roc_curve(y_test,y_scores) 
roc_auc = auc(fpr,tpr) 
print('Train set: ','%.4f'%accuracy_score(y_train, xgc.predict(X_std_train)   ), end=' ')    
print('Test set: ','%.4f'%accuracy_score(y_test, xgc.predict(X_std_test) ),end=' \n\n')
print(roc_auc)


# In[ ]:


import seaborn as sb
confusionMatrix = confusion_matrix(y_test,xgc.predict(X_std_test))
sb.heatmap(confusionMatrix,annot=True,fmt='d')


# In[ ]:




