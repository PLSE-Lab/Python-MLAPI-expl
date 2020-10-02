#!/usr/bin/env python
# coding: utf-8

# **Prediction of employee promotion potential**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train_LZdllcl.csv')
df.columns


# **Distribution of employees across various departments**

# In[ ]:


df['department'].value_counts()


# **Percentage of people who got promoted from each department**

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
ct = pd.crosstab(df.department,df.is_promoted,normalize='index')
ct.plot.bar(stacked=True)
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))


# While Technology department had highest percentage of employees getting promoted, Legal department has the least number. But we don't see major differences in terms of percentages.

# **Percentage of promotions across all the regions**

# In[ ]:


reg = pd.crosstab(df.region,df.is_promoted,normalize='index')
reg.plot.bar(stacked=True)
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))


# **Distribution of promotions among people with different Educational backgrounds**

# In[ ]:


plt.rcParams['figure.figsize'] = [5, 5]
edu = pd.crosstab(df.education,df.is_promoted,normalize='index')
edu.plot.bar(stacked=True)
plt.rcParams['figure.figsize'] = [5, 5]
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))


# As we can see the percentages are pretty much the same aross different educational backgrounds.

# **Variation of promotion percentage with respect to gender**

# In[ ]:


pd.crosstab(df.gender,df.is_promoted,normalize='index')


# There is major differnce in percentages across genders too. 
# Let us check if differnce arises within the departments

# **Variation of promotion percentage with recruitment channel that they have come from**

# In[ ]:


pd.crosstab(df.recruitment_channel,df.is_promoted,normalize='index')


# According to the data, percentage of promotions is higher among the employees who got recruited through referrals.

# **Variation of promotion percentage with respect to KPIs met or not**

# In[ ]:


pd.crosstab(df['KPIs_met >80%'],df.is_promoted,normalize='index')


# Higher percentage of employees got promoted in the group of people whose KPIs_met is greater than 80%.

# Let us check if there is any ratio difference across gender within the departments. For this I chose the top 3 highly populated departments.

# In[ ]:


sales = df[df['department']=='Sales & Marketing']
operations = df[df['department']=='Operations']
technology = df[df['department']=='Technology']
hr = df[df['department']=='HR']
fin = df[df['department']=='Finance']
legal = df[df['department']=='Legal']
RnD = df[df['department']=='R&D']
pd.crosstab(sales.gender,sales.is_promoted,normalize='index')


# In[ ]:


pd.crosstab(operations.gender,operations.is_promoted,normalize='index')


# In[ ]:


pd.crosstab(technology.gender,technology.is_promoted,normalize='index')


# In[ ]:


plt.rcParams['figure.figsize'] = [3, 5]
gender = pd.crosstab(RnD.gender,RnD.is_promoted,normalize='index')
gender.plot.bar(stacked=True)
plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))


# Slight difference in the ratios can be seen within the departments unlike the ratios calculated without any department barriers.

# **Difference in the percentage of promoted employees with respect to previous year ratings**

# In[ ]:


rating = pd.crosstab(df.previous_year_rating,df.is_promoted,normalize='index')
rating.plot.bar(stacked=True)
plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))


# The ratio of promoted employees increases with previous year rating which is quite obvious.
# 
# **Distribution of average training score**

# In[ ]:


bins = [30,40,50,60,70,80,90,100]
labels = ['30-40','40-50','50-60','60-70','70-80','80-90','90-100']
df['score_binned'] = pd.cut(df['avg_training_score'], bins=bins, labels=labels)
df['score_binned'].value_counts()


# While most of the employees have score in the range of 50-60, the least score bin has very faint number,
# 
# **Distribution of promoted employees ratio across different score ranges**

# In[ ]:


plt.rcParams['figure.figsize'] = [10, 5]
score_bin = pd.crosstab(df.score_binned,df.is_promoted,normalize='index')
score_bin.plot.bar(stacked=True)
plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))


# Promotions ratio increases with the score and the ratio is very high in 90-100 range which means getting promoted is highly dependent on the average score.
# 
# **Distribution of promotion ratios with respect to age**

# In[ ]:


plt.rcParams['figure.figsize'] = [5, 5]
age_bins = [20,30,40,50,60]
age_labels = ['20-30','30-40','40-50','50-60']
df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels)
df['age_binned'].value_counts()
age_bin = pd.crosstab(df.age_binned,df.is_promoted,normalize='index')
age_bin.plot.bar(stacked=True)
plt.rcParams['figure.figsize'] = [5, 5]
plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))


# Ratio doesn't vary much with age.

# In[ ]:


plt.rcParams['figure.figsize'] = [14, 5]
sns.scatterplot(x='age',y='avg_training_score',hue='is_promoted',data=df)


# This graph reinforces the fact the promotions are majorly dependent on the score and not on age.
# 
# **Mean score of employees with different educational background**

# In[ ]:


df.groupby(["education"])['avg_training_score'].mean()


# Mean training score doesn't vary with education

# **Filling the missing values**

# In[ ]:


df.isnull().any()


# Fill missing values of  'previous_year_rating' with mean based on 'KPIs_met >80%' and 'education'  with median based on 'department'

# In[ ]:


df['previous_year_rating'] = df.groupby(["KPIs_met >80%"])["previous_year_rating"].apply(lambda x: x.fillna(x.mean()))
df["education"] = df["education"].astype('object')
df['education'] = df.groupby(["department"])["education"].apply(lambda x: x.fillna(x.value_counts().index[0]))


# **Feature engineering**
# Normalize all the numerical features and encode all the categorical features.

# In[ ]:


scaled_features = df.copy()
col_names = ['no_of_trainings', 'age','previous_year_rating','length_of_service','awards_won?','avg_training_score']
label_names = ['department','gender','recruitment_channel','region']
features = scaled_features[col_names]
scaler = preprocessing.StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.get_dummies(scaled_features, columns=label_names, drop_first=True)
scaled_features[col_names] = features
scaled_features.drop(columns=['employee_id','age','education','score_binned','age_binned'],inplace=True)


# The transformed features are fit to a Gradient Boosting Algorithm.
# Grid Search cross validation is used to find the best hyperparameter('n_estimators')

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    scaled_features.loc[:, scaled_features.columns != 'is_promoted'], scaled_features['is_promoted'], test_size=0.33, random_state=42)
#forest = RandomForestClassifier(n_jobs=-1, random_state=0,class_weight='balanced',n_estimators=100,bootstrap=True, max_depth=80)
forest = GradientBoostingClassifier(loss='exponential',max_features='auto')
param_grid = {
    'n_estimators': [200,500,800]
}
grid_search = GridSearchCV(estimator = forest, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_


# According to the results of grid search, 500 is the optimal number of estimators.
# 
# **Finding feature importances**

# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importance = {}
for i in range(len(X_train.columns)):
    feature_importance[X_train.columns[i]] = feature_importances[i]
importance_df = pd.DataFrame(list(feature_importance.items()),columns=['feature','importance'])
importance_df = importance_df.sort_values('importance',ascending=False)
plt.xticks(rotation='vertical')
plt.rcParams['figure.figsize'] = [18, 10]
sns.barplot(x="feature",y="importance",data=importance_df)


# The above graph shows the importance of each feature in building the model. Here also average training score takes first place.

# In[ ]:


pred = grid_search.predict(X_test)
accuracy = metrics.accuracy_score(y_test, pred)
'accuracy - '+str(accuracy)


# In[ ]:


f1 = metrics.f1_score(y_test, pred)
'f1 score - '+str(f1)


# In[ ]:


recall = metrics.recall_score(y_test,pred)
'recall - '+str(recall)


# In[ ]:


precision = metrics.precision_score(y_test,pred)
'precision - '+str(precision)


# The model has a high accuracy but F1 score is less because of lesser Recall value. This means that number of False Negatives are higher which might have arised due to unbalanced classes. Oversampling or undersampling might increase F1 score.
