#!/usr/bin/env python
# coding: utf-8

# # Import python libraries and dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


survey_data = pd.read_csv('../input/bank_customer_survey.csv')


# # Dataset Overview

# In[ ]:


#Sample of dataset
survey_data.head()


# In[ ]:


#Listing column names in the dataset
survey_data.columns


# In[ ]:


#number of rows in the dataset
survey_data.count()[0]


# ## Detailed information about featuresin dataset
# 
# ## Customer information
# 1. **age :** (numeric)
# 2. **job :** type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
# 3. **marital :** marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
# 4. **education :** (categorical: "primary","secondary","tertiary","unknown")
# 5. **default :** has credit in default? (categorical: "no","yes","unknown")
# 6. **balance :** in eur
# 7. **housing :** has housing loan? (categorical: "no","yes","unknown")
# 8. **loan :** has personal loan? (categorical: "no","yes","unknown")
# 
# ## related with the last contact of the current campaign:
# 9. **contact :** contact communication type (categorical: "cellular","telephone") 
# 10. **month :** last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 11. **day_of_week :** last contact day of the week (categorical: "mon","tue","wed","thu","fri")
# 12. **duration :** last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# ## other attributes:
# 13. **campaign:** number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 14. **pdays :** number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted)
# 15. **previous :** number of contacts performed before this campaign and for this client (numeric)
# 16. **poutcome :** outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
# 17. **y :** Whether or not the customer has term deposit 

# In[ ]:


#Check if there are any missing values
survey_data.isnull().sum()


# ## Statistical overview

# In[ ]:


survey_data.describe()


# ### Corelation of all other features with the outcome

# In[ ]:


survey_data[['job', 'y']].groupby("job").mean().reset_index()


# Looks like students have the highest mean of all followed by retired

# In[ ]:


some_data = survey_data[['job', 'y']].groupby("job").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "job", x = 'y',data = some_data)


# In[ ]:


some_data = survey_data[['marital', 'y']].groupby("marital").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "marital", x = 'y',data = some_data)


# In[ ]:


fig, axes = plt.subplots(1,1, figsize = (15,5))
sns.countplot(x = survey_data['education'], hue = survey_data["y"])


# In[ ]:


some_data = survey_data[['education', 'y']].groupby("education").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "education", x = 'y',data = some_data)


# The above graph shows close score for tertiary and unkown, hence its okay to consider the unknown as tertiary.

# In[ ]:


fig, axes = plt.subplots(1,1, figsize = (15,5))
sns.countplot(x = survey_data['default'], hue = survey_data["y"])


# In[ ]:


some_data = survey_data[['default', 'y']].groupby("default").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "default", x = 'y',data = some_data)


# In[ ]:


some_data = survey_data[['housing', 'y']].groupby("housing").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "housing", x = 'y',data = some_data)


# In[ ]:


some_data = survey_data[['loan', 'y']].groupby("loan").mean().reset_index().sort_values("y", ascending=False)
sns.barplot(y = "loan", x = 'y',data = some_data)


# In[ ]:


yes_summary = survey_data.groupby("y")
yes_summary.mean().reset_index()


# Observations:
# * Average balance of subscribers for term deposit is more than non subscribers
# * Average duration of call is also more more those with y = 1
# * Average number of days passed since last contact(pdays) is also high for those with y = 1.

# In[ ]:


pd.DataFrame(abs(survey_data.corr()['y']).reset_index().sort_values('y',ascending = False))


# Duration is the most corelated feature with y

# In[ ]:


# Compute the correlation matrix
corr = survey_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# ## Dealing with categorical data
# There are no missing values in the data.
# Most of the features are categorical in nature ex, type of job, education, marital status.
# Some fields are binary, containing yes or no.
# We need to somehow categorise the "unknown" values to any category. 

# In[ ]:


survey_data['education'].value_counts()


# # Classification

# ## Random forest classifier

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in survey_data.columns:
    if(survey_data[col].dtype == 'object'):
        survey_data.loc[:,col] = le.fit_transform(survey_data.loc[:,col])


# In[ ]:


survey_data.head()


# In[ ]:


X = survey_data.iloc[:,:-1].values
y = survey_data.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Bagging Classifier

# In[ ]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed, n_jobs=-1)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# ## Random forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, n_jobs=-1)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# ## Extra Tree Classifier

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features, n_jobs=-1)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# # Boosting models

# ## adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold, n_jobs=-1)
print(results.mean())


# ## Stochastic Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# ## XGBoost

# In[ ]:


from xgboost import XGBClassifier
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = XGBClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())


# ## Catboost

# In[ ]:


from catboost import CatBoostClassifier
categorical_features_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]
model=CatBoostClassifier(iterations=50, depth=10, learning_rate=0.1, loss_function='Logloss')
model.fit(x_train, y_train,cat_features=categorical_features_indices,eval_set=(x_test, y_test),plot=True)

print(model.get_best_score())


# ## To be continued

# In[ ]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import VotingClassifier
# seed = 7
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# # create the sub models
# estimators = []
# model1 = LogisticRegression()
# estimators.append(('logistic', model1))
# model2 = DecisionTreeClassifier()
# estimators.append(('cart', model2))
# model3 = SVC()
# estimators.append(('svm', model3))
# # create the ensemble model
# ensemble = VotingClassifier(estimators)
# results = model_selection.cross_val_score(ensemble, X, y, cv=kfold, n)
# print(results.mean())

