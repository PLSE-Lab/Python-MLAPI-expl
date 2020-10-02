#!/usr/bin/env python
# coding: utf-8

# # Wine Quality with XGBoost
# 
# This notebook demonstrates the effectivnes of the XGBoost model on the red wine wuality dataset with minimal parameter tuning.
# 
# Comments and criticism welcome.
# 
# __Please upvote if you found this notebook helpful.__

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')


# Taking a look at the data

# In[ ]:


#import data into pandas dataframe
data = pd.read_csv("../input/winequality-red.csv")

#display first 5 lines
data.head()

#print data properties
print('Data Shape: {}'.format(data.shape))

display(data.describe())
display(data.info())
sns.countplot(data['quality'],label="Count")


# We see that the data consitsts of 11 features with no missing or NaN entries. The features vary in magnitude and scale so some feature scaling would be required.
# 
# Note how most wines in the dataset are of mediocre quality (5-6), with a low population of the low quality wines (3-4).
# For this analysis we will rearrange wine quality into low quality wines (<7) and high quality wines (>= 7).
# 
# ## Correlations

# In[ ]:


plt.figure(figsize=(10,10))
corr_mat=sns.heatmap(data.corr(method='spearman'),annot=True,cbar=True,
            cmap='viridis', vmax=1,vmin=-1,
            xticklabels=data.columns,yticklabels=data.columns)
corr_mat.set_xticklabels(corr_mat.get_xticklabels(),rotation=90)


# Note the following features are well correlated:
# -  citric acid and fixed acidity
# -  citric acid and volatile acidity
# -  density and fixed acidity
# -  pH and fixed acidity
# -  pH and citric acidity
# 
# Notice that quality is most correlated with alcohol.
# 
# 
# ## Preparing the Data
# 
# Relabeling wine quality:

# In[ ]:


bins = (1, 6.5, 8.5)
quality_level = ['low', 'high']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = quality_level)
sns.countplot(data['quality'])


# Feature extraction, scaling and splitting into train and test set. Note the use of stratify in the splitting. Since this is a highly unbalanced dataset, it is important to stratitfy the split i.e keep the proportion of classes the same in the splits. 

# In[ ]:


#Extract data and label target
X = data.iloc[:,0:11]
le = LabelEncoder().fit(data['quality'])
y = le.transform(data['quality'])

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)

#Scale data
scaler = MinMaxScaler()
scaler.fit(X_train,y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Pairplot

# In[ ]:


sns.pairplot(data, hue = "quality", diag_kind='kde')


# we can see some the correlations noted previously in the above pairplot. 

# ## XGBoost
# 
# With some parameter selection we get excellent results,

# In[ ]:


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'xgb__n_estimators': [500], 
              'xgb__max_depth': [2,3,4,5], 
              'xgb__alpha': [0.001,0.01,0.1,1],
              'xgb__min_samples_leaf': [1,2,3]}
xgb = XGBClassifier(random_state=0)
pipe = Pipeline([("scaler",MinMaxScaler()), ("xgb",xgb)])
grid_xgb = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.3f}".format(grid_xgb.best_score_))
print("Test set score: {:.3f}".format(grid_xgb.score(X_test,y_test)))
print("Best parameters: {}".format(grid_xgb.best_params_))

conf_mat_xgb = confusion_matrix(y_test, grid_xgb.predict(X_test))
sns.heatmap(conf_mat_xgb, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le.classes_, xticklabels=le.classes_)

# Feature importance
xgb = XGBClassifier(random_state=0, max_depth=grid_xgb.best_params_['xgb__max_depth'],
                                 n_estimators=grid_xgb.best_params_['xgb__n_estimators'],
                                 alpha=grid_xgb.best_params_['xgb__alpha'],
                                 min_samples_leaf = grid_xgb.best_params_['xgb__min_samples_leaf'])
xgb.fit(X_train_scaled,y_train)
plt.figure()
plt.bar(np.arange(X.shape[1]), xgb.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns, rotation=90)
plt.title('Feature Importance')

# Classification Report
print(classification_report(y_test, xgb.predict(X_test_scaled), target_names=le.classes_))


# Although the accuracy of our model is high (91.7%), this does not reflect the face that the dataset is overwhelmed by low quality wine. Looking at the confusion matrix gives us better insight into the prediction of our model. We see that most of the low quality wine was classified correctly, however only about (63%) of high quality wine was correctly classified. This is just a sideeffect of poorly balanced datasets.
# 
# Looking at the feature importance graph we see that the dominant feature when classifying wine quality is the alcohol content. This is expected from our observation of the correlation matrix.

# ## Ensemble Learning with SVC
# 
# In this section we put togethor an ensemble model made up of the previous XGB and an SVC model.

# ## SVC

# In[ ]:


pipe = Pipeline([("scaler",MinMaxScaler()), ("svm",SVC(random_state=0))])
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10],
              'svm__kernel': ['linear', 'rbf']}
grid_svm = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy',n_jobs=-1)
grid_svm.fit(X_train, y_train)
print("Best cross-validation accuracy: {:.3f}".format(grid_svm.best_score_))
print("Test set score: {:.3f}".format(grid_svm.score(X_test,y_test)))
print("Best parameters: {}".format(grid_svm.best_params_))


#SVM Cofusion matrix
conf_mat_svm = confusion_matrix(y_test, grid_svm.predict(X_test_scaled))
sns.heatmap(conf_mat_svm, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le.classes_, xticklabels=le.classes_)

# Classification Report
print(classification_report(y_test, grid_svm.predict(X_test_scaled), target_names=le.classes_))


# Notice how the SVC has classifed all the wines as low quality. Although this is useless on its own, combinig this model with the XGB might help increase the number of correctly classified low quality wines and increase total accuracy.
# 
# ## XGBoost & SVC

# In[ ]:


param = grid_svm.best_params_
svm = SVC(gamma = param["svm__gamma"], C = param["svm__C"], kernel=param["svm__kernel"], probability=True, random_state=99)

xgb = XGBClassifier(random_state=0, max_depth=grid_xgb.best_params_['xgb__max_depth'],
                                 n_estimators=grid_xgb.best_params_['xgb__n_estimators'],
                                 alpha=grid_xgb.best_params_['xgb__alpha'],
                                 min_samples_leaf = grid_xgb.best_params_['xgb__min_samples_leaf'])

ensemble = VotingClassifier(estimators=[('clf1',svm), ('clf2',xgb)], voting='soft', weights=[1,1])
ensemble.fit(X_train_scaled, y_train)
print("Ensemble test score: {:.3f}".format(ensemble.score(X_test_scaled, y_test)))

conf_mat_ens = confusion_matrix(y_test, ensemble.predict(X_test_scaled))
sns.heatmap(conf_mat_ens, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le.classes_, xticklabels=le.classes_)

cv_score=np.mean(cross_val_score(ensemble,X_test,y_test,cv=kfold))
print("Cross Validation score: {:.3f}".format(cv_score))


# As hoped, the combined model performs better on the test set than the XGB alone with an accuracy of 92%. However, this is probably just a lucky shuffle of the train and test sets. The cross validation score gives a more realistic result (88.2%).
# 
# __Please upvote if you found this notebook helpful.__
