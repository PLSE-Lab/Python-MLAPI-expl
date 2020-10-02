#!/usr/bin/env python
# coding: utf-8

# In this kernel we will try to analyze the kickstarter data, visualize it and try to make some predictions.

# In[ ]:


import string
import re, math, os, sys
import sklearn
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
pd.options.display.max_rows=100000
pd.options.display.max_columns=100000
pd.options.display.float_format = '{:.2f}'.format
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
plt.gcf().subplots_adjust(top=0.5, bottom=0.4)
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ksData=pd.read_csv("../input/ks-projects-201801.csv", sep=",")
ksData.describe()
ksData.head()


# We will start with some data expelanory analysis.
# 
# First, lets see the count of the projects by category:

# In[ ]:


categoryCount = pd.DataFrame(ksData.groupby('category').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='category', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=25000)
plt.xticks(rotation=75)


# So we see that most of the projects belong to the category of 'Product Design'.
# 
# But if we plot the data based on 'main category' featyure, the results would look a bit different:

# In[ ]:


categoryCount = pd.DataFrame(ksData.groupby('main_category').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='main_category', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=65000)
plt.xticks(rotation=75);


# According to this plot, the most poplular main category is 'Film & Video'.
# 
# Let's check how the projects belong to  'Film & Video' main category are distibuted:

# In[ ]:


FilmsProjects=ksData.loc[ksData['main_category']=='Film & Video',:]
categoryCount = pd.DataFrame(FilmsProjects.groupby('category').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='category', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=19000)
plt.xticks(rotation=75);


# So, among the 'Film & Video' main category, most project belong to the sub0category of 'Documentary'.
# 
# Now let's see how the origins of the projects distibuted:

# In[ ]:


categoryCount = pd.DataFrame(ksData.groupby('country').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='country', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=300000)
plt.xticks(rotation=75);


# Not surprisingly, the vast majority of the projects come from the US. far behind is GB.
# 
# Now, let's go from a different direction. Let's check the possible states that a project can be at, and how the projects distributed among them:

# In[ ]:


ksData['state'].unique()
ksData.groupby('state').size().sort_values(ascending=False).rename('counts').reset_index()
percentage=(ksData.groupby('state').size()/(ksData.shape[0]))
relativePart=percentage.values*100
t=np.char.mod('%.2f', relativePart)
labels=percentage.index+" - "+t+"%"
explode=(0,0,0,0.5,0,0)
matplotlib.pyplot.axis("equal")
patches, texts =plt.pie(percentage, explode=explode, shadow=True, startangle=90, radius=2)
plt.legend(patches, labels, bbox_to_anchor=(-0.1, 1.),
           fontsize=8)


# Okay, so most of the projects are either in state 'successful', 'failed' or 'canceled'. 
# 
# Now, lets look at the goals of the projects (in this analysis we will use the 'usd_goal_real' feature). How do they distributed?

# In[ ]:


categoryCount = pd.DataFrame(ksData.groupby('usd_goal_real').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='usd_goal_real', y="counts", data=categoryCount.head(50))
fig.axis(ymin=0, ymax=35000)
plt.xticks(rotation=75);


# Do the goals distribute differently between the suceessful and the failed projects? let's see:

# In[ ]:


success=ksData.loc[ksData['state']=='successful']
failed=ksData.loc[ksData['state']=='failed']
boxPlotData=[success['usd_goal_real'], failed['usd_goal_real']]
plt.boxplot(boxPlotData, labels=['success', 'failed'])


# Yes, the numbers are significantly more spread smong the 'failed' projects, with many outliers (are the extreme high goals may be the reason they failed? It sounds reasonable and we will get to it later on, in the prediction part).
# 
# Another interesting thing we can check is the distribution of the percentage if the goal that was pledged:

# In[ ]:


ksData['percentage']=(ksData['usd_pledged_real']/ksData['usd_goal_real']);
s=ksData.loc[:,'percentage']
plt.hist(s, bins=100, range=(0,1))


# Most of the projects achieved a very small fraction of the goal they set.
# 
# Next, we will calculate the duration of the projects. in order to do so, we will convert the dates into datetime format.

# In[ ]:


ksData['deadline_new']=pd.to_datetime(ksData['deadline'], dayfirst=True)
ksData['launched_new']=pd.to_datetime(ksData['launched'], dayfirst=True)
ksData['duration']=ksData['deadline_new']-ksData['launched_new']
ksData['duration']=ksData['duration'].dt.days # only days
ksData['duration'].describe()
ksData['duration'].median()


# It seams that the mean and the median are pretty low, but there are outliers. let's plot it:

# In[ ]:


plt.boxplot(ksData['duration'], labels=['duration'])
ksData[ksData['duration'] > 100]


# It seems like in some project, they use some default date in 1970 as the launch date. We will filter them out:

# In[ ]:


ksData.drop(ksData[ksData['duration'] > 100].index, inplace=True)
plt.boxplot(ksData['duration'], labels=['duration'])


# Now the values seem better.
# 
# In this point we would like to try to predict the success of a preject: can we predict id a project will be successful or it would fail? In order to do so, we first drop out the features that are not relevant. We would convert the categorical variables into dummy ones, and filter the data only for usccessfull / failed projects (drop out cancelled/live/suspended/undefined projects).

# In[ ]:


ksDataSF=ksData.loc[(ksData['state']=='successful') | (ksData['state']=='failed'), :]
keep=ksDataSF.columns.drop(['ID', 'name', 'deadline', 'goal', 'launched','pledged','usd pledged','deadline_new', 'launched_new'])
ksDataSF=ksDataSF[keep]
target='state'
ksDataSF=pd.get_dummies(ksDataSF, drop_first=True, columns=['category', 'main_category', 'currency', 'country'])
# ksDataSF.head()
le=sklearn.preprocessing.LabelEncoder()
ksDataSF['state']=le.fit_transform(ksDataSF['state'])
features=ksDataSF.columns.drop('state')
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
le_name_mapping


# We now train a logistic regression model in order to perform the classification.

# In[ ]:


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(ksDataSF[features], ksDataSF[target], train_size=0.7)
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train,y_train)
pred = logistic.predict(X_test)
print("***********")
print("accuracy_score:", sklearn.metrics.accuracy_score(y_test, pred, normalize=True))
print("classification_report:")
print(sklearn.metrics.classification_report(y_test, pred))
print("confusion_matrix:")
print(sklearn.metrics.confusion_matrix(y_test, pred))


# We see that the results are very good - accuracy of 99%! Now let's see what is the contribution of each freature to the prediction:

# In[ ]:


Coef=(logistic.coef_).tolist()[0]
featuresList=features.tolist()
zipped=zip(Coef,featuresList)
list(zipped)


# We see that the features that contribute to the classification the most are 'backers', 'usd_pledged_real', 'usd_goal_real', and 'duration'. In fact, this is kind of cheating, as the features 'backers' and especially 'usd_pledged_real' actually capture the answer to whether the goal was achieved or not.
# 
# We shall try to fit a model without these features:

# In[ ]:


features=features.drop(['backers','usd_pledged_real', 'percentage'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(ksDataSF[features], ksDataSF[target], train_size=0.7)
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train,y_train)
pred = logistic.predict(X_test)
print("***********")
print("accuracy_score:", sklearn.metrics.accuracy_score(y_test, pred, normalize=True))
print("classification_report:")
print(sklearn.metrics.classification_report(y_test, pred))
print("confusion_matrix:")
print(sklearn.metrics.confusion_matrix(y_test, pred))

Coef=(logistic.coef_).tolist()[0]
featuresList=features.tolist()
zipped=zip(Coef,featuresList)
list(zipped)


# The accuracy of the model is now weigh lower. It seems like all the other features - country, currency, category & main_category - don't contribute much to the model, and based on the ones that do - goal & duration - it is very hard to predict the sucess of a project.
