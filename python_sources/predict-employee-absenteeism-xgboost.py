#!/usr/bin/env python
# coding: utf-8

# # Business Case Study: Employee Absenteeism
# 
# The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.
# ![absemteeism.jpg](attachment:absemteeism.jpg)
# ***
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**
# ***

# # 1. The business Task
# 
# The exercise will address `Absenteeism` at a company during work time.
# 
# **Problem:** 
# The problem is that the work environment of today is more:
# - Competitive
# - Managers set unachievable business goals
# - have an elevated risk of becoming unemployed
# This can be lead to an increase in pressure and stress of the employee. Those factors influence employee health, which is of course indesirable.
# 
# ### What is Absenteeism?
# Absence from work during normal working hours resulting in temporary incapacity to execute a regular working activity.
# 
# - Based on what information should we predict whether an employee is expected to be absent or not?
# - How should we measure absenteeism?
# 
# ### Purpose of the business exercise:
# 
# Explore whether a person presenting certain characteristics is expected to be away from work at some point in time or not.
# 
# We want to know for how many working hours any employee could be away from work based on information like:
# - How far they live from their workplace.
# - How many children and pets they have.
# - Do they have higher education?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# In[ ]:


data = pd.read_excel('/kaggle/input/employee-absenteeism/Absenteeism_at_work_Project.xls')
data.head()


# # 2. Exploratory Data Analysis (EDA)

# In[ ]:


data.info()


# The data doesn't have any missing values.

# In[ ]:


pd.set_option("display.float_format", "{:.2f}".format)
data.describe()


# In[ ]:


for column in data.columns:
    print(f"===============Column: {column}==============")
    print(f"Number of unique values: {data[column].nunique()}")
    print(f"Max: {data[column].max()}")
    print(f"Min: {data[column].min()}")


# In[ ]:


data.columns


# In[ ]:


data.ID.value_counts().hist(bins=data.ID.nunique())


# `ID`: individual identification (in this case we have 34 employees) indicates precisely who has been away during working hours.  Will this information improve our analysis in any way? No, because it's only a label variable (a number that is there to distinguish the individuals from one another, not to carry any numeric information).
# 
# So we are going to drop this column

# In[ ]:


data.drop('ID', axis=1, inplace=True)


# `Reason for Absence`: We have `28` reason of absence from `0` to `28`.

# In[ ]:


# Visulazing the distibution of the data for every feature
data.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));


# In[ ]:


print(f"{data['Absenteeism time in hours'].value_counts()}")
print(f"{data['Absenteeism time in hours'].value_counts()[0] / data['Absenteeism time in hours'].value_counts()[1]}")


# In[ ]:


data["Reason for absence"].value_counts()


# In[ ]:


data["Reason for absence"] = data["Reason for absence"].map({0: "Group_1", 1: "Group_1", 2: "Group_1", 3: "Group_1", 
                                                             4: "Group_1", 5: "Group_1", 6: "Group_1", 7: "Group_1", 
                                                             8: "Group_1", 9: "Group_1", 10: "Group_1", 11: "Group_1", 
                                                             12: "Group_1", 13: "Group_1", 14: "Group_1", 15: "Group_2", 
                                                             16: "Group_2", 17: "Group_2", 17: "Group_2", 18: "Group_3", 
                                                             19: "Group_3", 20: "Group_3", 21: "Group_3", 22: "Group_4", 
                                                             23: "Group_4", 24: "Group_4", 25: "Group_4", 26: "Group_4", 
                                                             27: "Group_4", 28: "Group_4"})
# data["Reason for Absence"] = data["Reason for Absence"].astype("category").cat.codes
data["Reason for absence"].value_counts()


# In[ ]:


data_1 = pd.get_dummies(data, columns=['Reason for absence'])


# In[ ]:


data_1.head()


# In[ ]:


data_1.dtypes


# In[ ]:


data_1.dropna(inplace=True)
data_1.isna().sum()


# In[ ]:


data_1["Education"] = data_1.Education.map({1: 0, 2: 1, 3: 1, 4: 1})


# In[ ]:


data_1.Education.value_counts()


# In[ ]:


data_1.Education.isna().sum()


# In[ ]:


data_2 = pd.get_dummies(data_1, columns=["Education"], drop_first=True)
data_2.columns


# In[ ]:


plt.figure(figsize=(20, 15))
sns.heatmap(data_2.corr(), annot=True)


# In[ ]:


data_2['Absenteeism time in hours'].hist(bins=data_2['Absenteeism time in hours'].nunique())


# In[ ]:


# data_2.drop(["Distance to Work", "Month", "Weekday", "Age", "Daily Work Load Average"], 
#             axis=1, inplace=True)


# # 3. Applying machine learning algorthims

# In[ ]:


X = data_2.drop('Absenteeism time in hours', axis=1)
y = np.where(data_2["Absenteeism time in hours"] > data_2["Absenteeism time in hours"].median(), 1, 0)

print(X.shape)
print(y.shape)


# In[ ]:


y.sum() / y.shape[0]


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

model = LogisticRegression(solver="liblinear")

x_sc = StandardScaler()
X_std = x_sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=20)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def accuracy_report(y_test, y_test_pred, sample="Test"):
    print(f"========={sample} data============ :\n=>Accuracy Score {accuracy_score(y_test, y_test_pred)}")
    print(f"=>Confusion Matrix :\n{confusion_matrix(y_test, y_test_pred)}")


# In[ ]:


accuracy_report(y_train, y_train_pred, "Train")


# In[ ]:


accuracy_report(y_test, y_test_pred, "Test")


# In[ ]:


pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))


# In[ ]:


scores = cross_val_score(model, X_std, y, cv=10)
scores.mean()


# In[ ]:


model.intercept_


# In[ ]:


model


# In[ ]:


feature_name = X.columns
summary_table = pd.DataFrame(columns=["Features_name"], data=feature_name)
summary_table["Coefficients"] = np.transpose(model.coef_)
summary_table


# In[ ]:


summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', model.intercept_[0]]
summary_table.sort_index(inplace=True)


# In[ ]:


summary_table["Odds_ratio"] = np.exp(summary_table.Coefficients)
summary_table.sort_values(by="Odds_ratio", ascending=False)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier(n_estimators=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

param_grid = {'max_depth':[3, None], 'min_samples_split':[2, 3, 10], 
              'min_samples_leaf':[1, 3, 10], 'bootstrap':[True, False], 
              'criterion':["gini", "entropy"]}

random_forest_grid = GridSearchCV(rf_model, param_grid, scoring="accuracy", 
                                  n_jobs=-1, verbose=1, cv=3, iid=True)

random_forest_grid.fit(X_train, y_train)

y_train_pred = random_forest_grid.predict(X_train)
y_test_pred = random_forest_grid.predict(X_test)


# In[ ]:


accuracy_report(y_test, y_test_pred)


# In[ ]:


accuracy_report(y_train, y_train_pred, "Train")


# In[ ]:


pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))


# ## Gradient Boosting Classifier[](http://)

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbm_model = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

gbm_model.fit(X_train, y_train)

y_train_pred = gbm_model.predict(X_train)
y_test_pred = gbm_model.predict(X_test)


# In[ ]:


accuracy_report(y_test, y_test_pred)


# In[ ]:


accuracy_report(y_train, y_train_pred, "Train")


# In[ ]:


pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))


# ## XGBoost

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 0.99]
learning_rate = [0.05, 0.1, 0.15, 0.20]
min_child_weight = [1, 2, 3, 4]

hyperparameter_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,
                       'learning_rate' : learning_rate, 'min_child_weight' : min_child_weight, 
                       'booster' : booster, 'base_score' : base_score
                      }

xgb_model = xgb.XGBClassifier()

xgb_cv = RandomizedSearchCV(estimator=xgb_model, param_distributions=hyperparameter_grid,
                               cv=5, n_iter=50, scoring = 'accuracy',n_jobs =-1, iid=True,
                               verbose = 5, return_train_score = True, random_state=42)


xgb_cv.fit(X_train, y_train)


# In[ ]:


xgb_cv.best_estimator_


# In[ ]:


xgb_best = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bynode=1, colsample_bytree=1, gamma=0,
                         learning_rate=0.1, max_delta_step=0, max_depth=2,
                         min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                         nthread=None, objective='binary:logistic', random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                         silent=None, subsample=1, verbosity=1)


# In[ ]:


xgb_best.fit(X_train, y_train)


# In[ ]:


y_train_pred = xgb_best.predict(X_train)
y_test_pred = xgb_best.predict(X_test)


# In[ ]:


accuracy_report(y_test, y_test_pred)


# In[ ]:


accuracy_report(y_train, y_train_pred, "Train")


# In[ ]:


pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))


# ## Save the model

# In[ ]:


import pickle

with open('model', 'wb') as file:
    pickle.dump(xgb_best, file)

