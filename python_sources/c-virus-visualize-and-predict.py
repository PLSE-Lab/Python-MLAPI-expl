#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


seed = 42
random.seed(seed)


# In[ ]:


import matplotlib.style as style
style.available


# In[ ]:


style.use('tableau-colorblind10')


# # Importing Files

# In[ ]:


data = pd.read_csv('../input/coronavirusdataset/patient.csv')


# # Exploratory Data Analysis

# In[ ]:


data.drop('id', axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


# missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


fig = plt.figure(figsize=(20, 5))
missing_plt = sns.barplot(x=missing_data.index, y=missing_data.Percent)
missing_plt.set_xticklabels(missing_plt.get_xticklabels(), rotation=45)
print()


# It is clear that our dataset cointains huge amount of null values and we need to deal with it.<br>
# 99 to 93% data is missing in all the major features.

# In[ ]:


print([pd.value_counts(data[cols]) for cols in data.columns])


# <h2> Imputing Missing Values </h2>

# Imputing missing values in a dataset is a laborious task some times it renders data useless. Here we will try to clean data 1 by 1.<br>
# Generally it is recomended to drop rows containig more than 50% empty values but in this case removing those data will lead to shrinkage of dataset to point where it is not useable.
# Firstly we will remove data that are not relevant or not very useful to us.<br>

# <h4> Sex </h4>

# In[ ]:


print('No. of NA values in Sex:', data['sex'].isna().sum())


# Sex have most no. of data besides having 93% missing values, so we will start droping all other data where gender is missing.

# In[ ]:


data_cleaned = data.dropna(axis=0, subset=['sex'])


# <h4>Birth_Year</h4>
# <br>
#     Birth year have some missing values and we will fill it up with mean.

# In[ ]:


data_cleaned['birth_year'].fillna(1973.0, inplace=True)


# <h4>Region</h4>
# Region field has 18 missing field. Region field is not important for us if we want to make predictions but it will be useful for plotting data on map. For now we will remove region column.

# In[ ]:


data_cleaned = data_cleaned.drop('region', axis=1)


# <h4>Disease</h4>
# <br>
# Disease column have null value where it patient does not have any disease, hence we will impute it with 0.0.

# In[ ]:


data_cleaned['disease'].fillna(0.0, inplace=True)


# <h4>Group</h4>
# <br>
# Group would have been useful if had no missing values because it could have been treated as bins, but as of now it serves no purpose to us. Let's remove group.

# In[ ]:


data_cleaned = data_cleaned.drop('group', axis=1)


# <h4>Infection Reason </h4>
# Infection reason seems like an important categorical feature, we will simply impute reason as unknown where it is null in dataset.

# In[ ]:


data_cleaned['infection_reason'].fillna('Unknown', inplace=True)


# <h4>Infection Order, Infected By, Contact Number, Released Date and Deceased Date</h4>
# It contains very low amount of data it's better to get rid of these features.

# In[ ]:


data_cleaned = data_cleaned.drop(['infected_by','infection_order', 'contact_number', 'released_date', 'deceased_date'], axis=1)


# In[ ]:


data_cleaned.head()


# In[ ]:


data_cleaned.info()


# <br>
# By removing all the cluter and irrelevant data we are left with very samll but properly cleand dataset. Now we can analyze data, introduce new relevant features and make infrences about it. most of the data is categorical in nature.<br>
# Let's get familiar with data.

# # Data Visualization

# In[ ]:





# <h3>Gender</h3>
# Gender have only two column male and female we will analyze data on the basis of gender to understand effect of deadly corona virus on both males and females

# In[ ]:


gender_count_plt = sns.countplot(data_cleaned['sex'], )
gender_count_plt.set_title('Gender Count')
plt.show()


# In[ ]:


gender_vs_state_plt = sns.countplot(x='sex', hue='state', data=data_cleaned)
gender_count_plt.set_title('Gender VS State')
plt.show()


# Above plots indidcates that both males and females got equally affected by the virus. With males slightly more than the females. Currently more females are kept isolated than men. Data also shows that although recovery in both males and females are almost identical but more males are deceased due to virus than compared to women.

# <h3> Birth Year </h3>
# <br>
# Birth year will alow us to see which age group is more vulnerable to this deadly virus.

# Let's add an age column to plot data easily.

# In[ ]:


data_cleaned['age'] = 2019 - data_cleaned['birth_year']
data_cleaned['age_bin'] = (data_cleaned['age'] // 10) * 10
data_cleaned = data_cleaned.drop('birth_year', axis=1)


# In[ ]:


sns.distplot(data_cleaned['age_bin'])


# In[ ]:


plt.figure(figsize=(25, 10))
age_sex_plt = sns.countplot(x='age', hue='sex', data=data_cleaned)
age_sex_plt.set_xticklabels(age_sex_plt.get_xticklabels(), rotation=-45)
age_sex_plt.legend(loc='upper right')


# In[ ]:


plt.figure(figsize=(25, 6))
age_sex_plt1 = sns.countplot(x='age_bin', hue='sex', data=data_cleaned)
age_sex_plt1.set_xticklabels(age_sex_plt1.get_xticklabels(), rotation=-45)
age_sex_plt1.legend(loc='upper right')


# In[ ]:


plt.figure(figsize=(25, 6))
age_state_plt = sns.countplot(x='age_bin', hue='state', data=data_cleaned)
age_state_plt.set_xticklabels(age_state_plt.get_xticklabels(), rotation=-45)
age_state_plt.legend(loc='upper right')
age_state_plt.set_title('Age VS State')


# In[ ]:


sns.countplot(data_cleaned['state'])


# Most of the people infected by corona virus are in there 50's with higher females at higher risk than male, followed by 30's and 40's where male are more prone to risk in there 30's and female in there's 40.<br>
# Older males are at higher risk whereas younger females are at higher risk of getting corona virus.<br>
# Fatality in older age is quite higher there is gradual increase in mortatlity rate as age of patient increase. This shows that older people have lower chances of surviving.<br>
# Till now most of the people are isolated it will be harder to predict state of patient as currently, only 55 people were released, 31 were dead and 344 are still isolated. Original data have much higher isolated values.<br>
# Note: As dataset was not clean hence this data is not totally complete this infrences are only made on this dataset, It is assumed that this dataset represents overall trends on actual data.

# <h3>Infection Reason</h3>
# Infection reason seems like an interesting feature to explore.

# In[ ]:


plt.figure(figsize=(20, 5))
infection_reason_plt = sns.countplot(data['infection_reason'])
infection_reason_plt.set_xticklabels(infection_reason_plt.get_xticklabels(), rotation=-45)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 5))
infection_reason_plt1 = sns.countplot(data_cleaned['infection_reason'])
infection_reason_plt1.set_xticklabels(infection_reason_plt1.get_xticklabels(), rotation=-45)
plt.show()


# In[ ]:


plt.figure(figsize=(20, 5))
reason_age_plt = sns.countplot(x='age_bin', hue='infection_reason', data=data_cleaned)
reason_age_plt.set_xticklabels(reason_age_plt.get_xticklabels(), rotation=-45)
reason_age_plt.legend(loc='upper right')
print()


# In[ ]:


plt.figure(figsize=(20, 5))
reason_age_plt = sns.countplot(x=data_cleaned['age_bin'], hue=data['infection_reason'])
reason_age_plt.set_xticklabels(reason_age_plt.get_xticklabels(), rotation=-45)
reason_age_plt.legend(loc='upper right')
print()


# In[ ]:


plt.figure(figsize=(20, 5))
reason_age_plt = sns.countplot(x=data_cleaned['infection_reason'], hue=data_cleaned['state'])
reason_age_plt.set_xticklabels(reason_age_plt.get_xticklabels(), rotation=-45)
reason_age_plt.legend(loc='upper right')
print()


# Due to high unknown reason for a patient to get virus, plots got distorted so we will analyse data with and without unknow reason to better understand it.<br>
# From above plot, Reason how patient got virus is mostly unknow for more than 275 patients.<br>
# Apart from the unknown reason, patinets are mostly affected due to contact with another person. As this virus is highly contagious this seems like obvious reason. Another reasons includes visiting to places where virus is already spreading.
# <br>
# <br>
# Getting infected due to pilgrimage is quite concerning as in Israel virus is not that prevalent yet <b> Pilgrimage have high risk</b> of spreading virus. <b> <br> We should definately avoid mass gathering. </b>

# <h3>Disease</h3>
# Underlying disease can be usefull to know effect of virus on patient with some underlying diseases.

# In[ ]:


disease_vs_state_plt = sns.countplot(x='disease', hue='state', data=data_cleaned)


# In[ ]:


disease_vs_age_plt = sns.countplot(x='age_bin', hue='disease', data=data_cleaned)


# Clearly as age increases underlying disease increases in population although this virus is not only isolated to person with underlying diseases but it definately increases risk for person dignosed with some past medical conditions. <br>
# patients without underlying diseases have chances for recovering and released but patient with underlying conditions have negligible chance of surviving.

# In[ ]:


# Removing redundant confirmed date and age columns
data_cleaned.drop(['confirmed_date', 'age'], axis=1, inplace=True)


# In[ ]:


data_cleaned.head()


# This Sums up our analysis of the Coronavirus data after analyising and cleaning data there is not much useful data left for us.
# I cannot see any scope of feature engineering and make more usefull feature but if you think there is any new useful feature hiding in plain sight just let me know ;)

# # Predicting State

# Now we will try to develop a predictive model to predict whether patinet should be released, isolated or should provided intensive care(can die if predicted deceased and not yet dead).
# <br>
# We don't have much data to play around so we will split 80-20 data in train and test set. as data is not uniformely distributed we will use stratified split

# In[ ]:


target = data_cleaned['state']


# In[ ]:


print(target.head(),'\n',  target.shape)


# In[ ]:


features = data_cleaned.drop('state', axis=1)


# In[ ]:


print(features.head(), '\n',  features.shape)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, stratify=target, random_state=42)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# <h3> Encoding & Scaling </h3>
# Now we will convert categorical data to Onehotencoding along with scaling it.

# In[ ]:


ohe = ce.OneHotEncoder()


# In[ ]:


sc = RobustScaler()


# In[ ]:


pipe = Pipeline(steps=[('ohe', ohe)])


# In[ ]:


x_train = pipe.fit_transform(x_train)


# In[ ]:


x_test = pipe.transform(x_test)


# <h3>Model</h3>
# We will fit a classification model to predict state of the patient.

# <h5>Logistic Regression</h5>
# Let's start with a simple logistic regression.

# In[ ]:


rf = LogisticRegression()


# In[ ]:


grid_param = {'C': [1.0, 1.2, 1.3],
             'fit_intercept': [True, False], 
             'solver': ['newton-cg', 'liblinear', 'lbfgs'],
             'tol': [1e-3, 1e-4],
             'max_iter': [500, 1000,  2000]}


# In[ ]:


grid = GridSearchCV(rf, grid_param, scoring='accuracy', n_jobs=-1, cv=5)


# In[ ]:


grid.fit(x_train, y_train)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


model = grid.best_estimator_


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))


# In[ ]:


report


# In[ ]:


confusion_mat = confusion_matrix(y_test, y_pred)


# In[ ]:


confusion_mat


# In[ ]:


conf_mat = sns.heatmap(confusion_mat, square=True, vmax= 15, vmin=4, annot=True, cmap='YlGnBu')


# Now we can succesfully predict whether a person will be isolated, released or have chances of dying due to virus.<br>
# This model can be useful to predict how much care a patient requires along with chances of whether patient can be released in futire or not.
# <br>
# This is not at all meant for production but rather a simple model to beat hurestic of whether an individual can be released or not.

# In[ ]:




