#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn import metrics
import os


# In[ ]:


#Read in data
data = pd.read_csv("../input/kpmi-mbti-mod-test/kpmi_data.csv", sep=';');


# In[ ]:


data.head()


# In[ ]:


#Check for NAs
display(data.isnull().sum().sort_values(ascending=False))


# In[ ]:


#Percent NAs
display(((data.isnull().sum())/(data.count())).sort_values(ascending=False))


# In[ ]:


#How many unique job titles and job fields
print('Number of job titles:', len(data.jobtitle.unique()), 
      '\nNumber of job fields:', len(data.jobfield.unique()), 
      '\nNumber of types:', len(data.psychotype.unique()))


# In[ ]:


#How many people are there per job field?
figure(figsize=(12, 8))
sns.countplot(y='jobfield', 
              data=data,
              order = data['jobfield'].value_counts().index);


# In[ ]:


#Calculate average job satisfaction
plt_data = data.groupby('jobfield').satisfied.mean().sort_values(ascending=False)

#Create bar plot
figure(figsize=(12, 8))
sns.barplot(y=plt_data.index,x=plt_data.values).set(xlabel='Average Satisfaction', ylabel='');


# In[ ]:


figure(figsize=(12, 8))
sns.countplot(y='psychotype', 
              data=data,
              order = data['psychotype'].value_counts().index);


# In[ ]:


#Calculate average job satisfaction
plt_data = data.groupby('psychotype').satisfied.mean().sort_values(ascending=False)

#Create bar plot
figure(figsize=(12, 8))
sns.barplot(y=plt_data.index,x=plt_data.values).set(xlabel='Average Satisfaction', ylabel='');


# In[ ]:


#Drop job title and field
#because there is a high percentage of missing values
#and many categories
data = data.drop('jobtitle', axis=1)
data = data.drop('jobfield', axis=1)


# In[ ]:


#Lists of columns by column type
num_cols = [cname for cname in data.columns if 
                data[cname].dtype in ['int64', 'float64']]

cat_cols = [cname for cname in data.columns if 
                data[cname].dtype == 'O']


# In[ ]:


#Calculate correlations
cor = data[num_cols].corr()

#Create list to subset correlation matrix
#to reduce number of variables in visualization

l=[]
for i in range(0, len(cor.satisfied)): 
    if abs(cor.satisfied[i]) > 0.15:
        l.append(i)

cor_sub = cor.iloc[l, l]


# In[ ]:


figure(num=None, figsize=(12, 8))

#Plot numerical variables most correlated with satisfaction
sns.heatmap(cor_sub);


# In[ ]:


#Specify dependent variable
y = data.satisfied
data = data.drop('satisfied', axis=1)

#Update column type lists
num_cols = [cname for cname in data.columns if 
                data[cname].dtype in ['int64', 'float64']]

cat_cols = [cname for cname in data.columns if 
                data[cname].dtype == 'O']


# In[ ]:


#Divide data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(data[num_cols], y, random_state=126, train_size = 0.8)


# In[ ]:


#Preprocessing for categorical data
#cat_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='constant', fill_value = 'No Response')),
#    ('encoder', OrdinalEncoder())
#])

#Bundle preprocessing for numerical and categorical data
#preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols)])


# In[ ]:


#Model
#model = RandomForestClassifier(random_state=126)
model = XGBClassifier(random_state=126)

#Random forest
#param_grid = {
#    'model__n_estimators': [100, 120, 150],
#    'model__max_depth': [8, 10, 12]}

#XGBoost
param_grid = {
    'model__n_estimators': [150, 180],
    'model__max_depth': [5, 8],
    'model__learning_rate': [0.08, 0.1, 0.12]}


# In[ ]:


#Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('model', model)])

search = GridSearchCV(my_pipeline, param_grid, n_jobs=-1, verbose=10, cv=5)
search.fit(train_X, train_y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)


# In[ ]:


#Predicting satisfaction 
pred = search.predict(test_X)

score = metrics.roc_auc_score(test_y, pred)
print("AUC:", score)

score = mean_absolute_error(test_y, pred)
print('MAE:', score)

print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(search.score(train_X, train_y), search.score(test_X, test_y)))


# In[ ]:


#model = RandomForestClassifier(max_depth = 12, n_estimators = 150, random_state=126)
model = XGBClassifier(learning_rate = 0.1, n_estimators = 150, max_depth = 5, random_state=126)
model.fit(train_X, train_y);


# In[ ]:


# Extract feature importances
fi = pd.DataFrame({'feature': list(train_X.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)


# In[ ]:


# Display top 30 features
figure(figsize=(10, 8))
sns.barplot(x="importance", y="feature", data=fi.iloc[:30,]);

