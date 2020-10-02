#!/usr/bin/env python
# coding: utf-8

# # Loading package

# In[ ]:


get_ipython().system('pip install seaborn')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# In[ ]:


submission_sample = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')


# # EDA

# In[ ]:


train.sample(5)


# In[ ]:


test.sample(5)


# In[ ]:


print(list(enumerate(train.columns)))


# In[ ]:


train.info()


# In[ ]:


train.nunique()


# In[ ]:


train.describe()


# ### Relationship among features and label

# In[ ]:


import seaborn as sns
plt.figure(figsize=(15,10))
sns.countplot(train['Cover_Type'])
plt.xlabel("Type of Cpver", fontsize=12)
plt.ylabel("Rows Count", fontsize=12)
plt.show()


# In[ ]:


# Bivariate EDA
pd.crosstab(train.Soil_Type31, train.Cover_Type)


# In[ ]:


#Convert dummy features back to categorical
x = train.iloc[:,15:55]
y = train.iloc[:,11:15]
y = pd.DataFrame(y)
x = pd.DataFrame(x)
s2 = pd.Series(x.columns[np.where(x!=0)[1]])
s3 = pd.Series(y.columns[np.where(y!=0)[1]])
train['soil_type'] = s2
train['Wilderness_Area'] = s3
train.head()


# In[ ]:


# Create a new dataset exluding dummies variable for Mutivariate EDA
df_viz = train.iloc[:, 0:15]
df_viz = df_viz.drop(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
                      'Wilderness_Area4'], axis = 1)
df_viz.head()


# In[ ]:


plt.figure(figsize=(15,10))
pd.crosstab(train.Wilderness_Area, train.Cover_Type).plot.barh(figsize=(15,15),stacked = True)


# In[ ]:


plt.figure(figsize=(15,10))
pd.crosstab(train.soil_type, train.Cover_Type).plot.barh(figsize=(15,15),stacked = True)


# In[ ]:


plt.subplots(figsize=(10,10))
corr = df_viz.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# # Feature Engineering

# In[ ]:


def add_feature(data):   
    data['Ele_minus_VDtHyd'] = data.Elevation-data.Vertical_Distance_To_Hydrology
    data['Ele_plus_VDtHyd'] = data.Elevation+data.Vertical_Distance_To_Hydrology
    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways']
    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways']
    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways']
    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways']
    return data


# In[ ]:


train = add_feature(train)
test = add_feature(test)


# In[ ]:


X_train = train.drop(['Id','Cover_Type','soil_type','Wilderness_Area'], axis = 1)
y_train = train.Cover_Type
X_test = test.drop(['Id'], axis = 1)


# # Logistics regression

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlr_pipe = Pipeline(\n    steps = [\n        ('scaler', MinMaxScaler()),\n        ('classifier', LogisticRegression(solver='lbfgs', n_jobs=-1))\n    ]\n)\n\nlr_param_grid = {\n    'classifier__C': [1, 10, 100,1000],\n}\n\n\nnp.random.seed(1)\ngrid_search = GridSearchCV(lr_pipe, lr_param_grid, cv=5, refit='True')\ngrid_search.fit(X_train, y_train)\n\nprint(grid_search.best_score_)\nprint(grid_search.best_params_)")


# # Random Forest

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nrf_pipe = Pipeline(\n    steps = [\n        ('classifier', RandomForestClassifier(n_estimators=500))\n    ]\n)\n\nparam_grid = {\n    'classifier__min_samples_leaf': [2, 3, 4, 8],\n    'classifier__max_depth': [30, 32, 34],\n}\n\nnp.random.seed(1)\nrf_grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, refit='True', n_jobs=-1)\nrf_grid_search.fit(X_train, y_train)\n\nprint(rf_grid_search.best_score_)\nprint(rf_grid_search.best_params_)")


# In[ ]:


rf_model = rf_grid_search.best_estimator_

cv_score = cross_val_score(rf_model, X_train, y_train, cv = 5)
print(cv_score)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))


# In[ ]:


rf = rf_grid_search.best_estimator_.steps[0][1]


# In[ ]:


feat_imp = rf.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature':X_train.columns,
    'feat_imp':feat_imp
})

feat_imp_df.sort_values(by='feat_imp', ascending=False).head(10)


# In[ ]:


sorted_feat_imp_df = feat_imp_df.sort_values(by='feat_imp', ascending=True)
plt.figure(figsize=[6,6])
plt.barh(sorted_feat_imp_df.feature[-20:], sorted_feat_imp_df.feat_imp[-20:])
plt.show()


# # Gradient Boosting

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nxgd_pipe = Pipeline(\n    steps = [\n        ('classifier', XGBClassifier(n_estimators=50, subsample=0.5))\n    ]\n)\n\nparam_grid = {\n    'classifier__learning_rate' : [0.45],\n    'classifier__min_samples_split' : [8, 16, 32],\n    'classifier__min_samples_leaf' : [2],\n    'classifier__max_depth': [15]\n    \n}\n\nnp.random.seed(1)\nxgd_grid_search = GridSearchCV(xgd_pipe, param_grid, cv=5,\n                              refit='True', verbose = 10, n_jobs=-1)\nxgd_grid_search.fit(X_train, y_train)\n\nprint(xgd_grid_search.best_score_)\nprint(xgd_grid_search.best_params_)")


# In[ ]:


xgd_model = xgd_grid_search.best_estimator_

cv_score = cross_val_score(xgd_model, X_train, y_train, cv = 5)
print(cv_score)
print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))


# # Final model

# In[ ]:


final_model = xgd_grid_search.best_estimator_.steps[0][1]


# In[ ]:


final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)


# In[ ]:


print(len(test.Id))


# In[ ]:


print(len(y_pred))


# In[ ]:


from collections import Counter
Counter(y_pred)


# # Submission

# In[ ]:


submission_sample.head()


# In[ ]:


submission = pd.DataFrame({'Id': test.Id,
                           'Cover_Type': y_pred})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

