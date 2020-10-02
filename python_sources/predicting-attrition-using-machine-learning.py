#!/usr/bin/env python
# coding: utf-8

# # Predicting attrition with classification models
# ### Dataset provided by IBM
# ##### Matt Warr - Data analyst
# 
# ##### The aim of this notebook is to effectively predict attrition in a business. This will be a step by step process from data collection and cleaning, to model selection and tuning to produce the most accurate score, focusing on effectively representing my skills learnt during the 12 week Data science immersive course at General Assembly (Sydney). 

# ## Import all packages needed.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import xgboost

import os
import itertools

sns.set_style('whitegrid')
sns.set_palette("Set2")

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

sns.palplot(sns.color_palette())


# ## 1. Data Cleaning & Preparation

# The bulk of any data science work is usually the cleaning and preparation of your dataset. This stage can and should take up most of your time throughout a machine learning project. Luckily for me, this dataset was sourced from kaggle and comes in a relatively clean state.
# 
# In the real world however, most of the time, Data will be presented to you in all sorts of forms and conditions. Being able to effectively clean, combine and prepare data will set you up for success whenever you are striving to build a machine learning model.
# 
# There is a reason why it was drilled into us from the start of the course, that 80% of your workload will likely be cleaning, and its best to embrace this rather than fight it. A clean dataset is the holy grail of Data Science.

# In[ ]:


""" Load in the data and examine the head. """

data = '../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(data)
df.head(5)


# ### Target column - Attrition
#     -Binary target currently Yes/no

# ### Examine columns

# In[ ]:


df.columns


# ### Check dataset shape

# In[ ]:


df.shape


# ### view dataframe info
#     - check for missing values
#     - examine the different datatypes

# In[ ]:


df.info()


# ### Use the describe function to examine the dataframe further
#     - Look for outstanding numbers or consistency
#     - This will only show the numeric columns (may need to examine categorical variables further individually)

# In[ ]:


""" Transpose the describe to make it easier to read. """
df.describe().T


# #### insights gained from this:
# - columns such as education that are numerical that actually represent categorical variables may need further investigation.
# - EmployeeCount is a constant column with all values being 1.
# - performance rating whilst being on a scale of 1-4, only has 3s and 4s.
# - Standard Hours is a constant column with all values being 80 hours.

# ## Do any Cleaning needed
# - As this dataset is from Kaggle, dataset is relatively clean, any cleaning done is mainly to improve quality of life (QoL) for coding.

# In[ ]:


""" Drop the two constant columns. """

df.drop(['EmployeeCount','EmployeeNumber'],axis=1,inplace=True)


# In[ ]:


""" Investigate categorical columns that seem irrelevent. """

df.Over18.value_counts()


# In[ ]:


""" Drop Over18 as it is also a constant. """

df.drop('Over18',axis=1,inplace=True)


# In[ ]:


""" Alter the Education column to show categorical variables. This will later be dummied for modelling. """

df.replace({'Education':{1:'Below_college', 2:'College', 3:'Bachelor', 4:'Masters', 5:'Doctor'}}, inplace=True)


# In[ ]:


""" Iterate through the dataframes columns and covert to lowercase for QoL. """
low_col = []
for i in df.columns:
    i = i.lower()
    low_col.append(i)
df.columns = low_col


# In[ ]:


df.columns


# In[ ]:


# """ Save this dataframe in its current state before dummying for EDA purposes. """

#df.to_csv('./Dataset/Clean_nodummy.csv')


# In[ ]:


""" Apply Pandas get_dummies function to the dataframe to transform the categorical variables to numeric.
    This will create additional columns in your dataframe."""

df_dum = pd.get_dummies(df)


# In[ ]:


""" Take a look at the shape before and after dummying to see how many new columns have been created. """

print(df.shape)
print(df_dum.shape)


# In[ ]:


df_dum.columns


# In[ ]:


""" get_dummies has a drop_first parameter, which removes the first column created for each categorical variable
    which acts as a default, however we do not want this to be done for all of our columns, since this is a relatively
    small dataset, i will do this manually."""

df_dum.drop(['gender_Female','overtime_No','attrition_No'],axis=1,inplace=True)


# In[ ]:


""" Due to the presence of capital letters in the variables, once again iterate through and convert all the lowercase """
low_col = []
for i in df_dum.columns:
    i = i.lower()
    low_col.append(i)
df_dum.columns = low_col


# In[ ]:


df_dum.columns


# In[ ]:


#""" Export this dummied dataframe to be used in the modelling phase of this project. """
#df_dum.to_csv('./Dataset/IBM_dummied.csv')


# # 2. EDA

# In this stage of the project, the main objective is to look further into the dataset and try and find insights and potential trends relating to the target variable in order to have a better understanding when it comes to selecting features for the modelling stage of the project.
# 
# Being able to thoroughly explore a dataset and present unique insights can set you apart from other data analysts/scientists. Most semi-capable analysts can find the obvious trends/relationships in data, it takes a keener eye for detail and following your gut instinct to delve deeper and connect the dots to provide those hidden relationships that can potentially make you stand above the rest.
# 
# (This may not always be the case, some datasets are an open book, unfortunately theres not always that secret insight that will jump out at you, but its worth looking, even for a little while just in case.)

# In[ ]:


""" Read in the Undummied CSV. """

df = pd.read_csv('../input/ibm-data/Clean_nodummy.csv', index_col=0)


# In[ ]:


""" Lets have a look at the target variable in this case. """

df.attrition.head()


# In[ ]:


""" At the moment, this column is categorical. For use in visualisations etc. this will need to be numeric,
    so lets convert Yes and No, into binary, or 1's and 0's"""

df.loc[df.attrition == 'Yes', 'attrition'] = 1
df.loc[df.attrition == 'No', 'attrition'] = 0


# In[ ]:


""" Next, take a look at the value_counts to see the state of the target. """
df.attrition.value_counts()


# The first thing we can gather from this is that there is some definite class imbalance to this target variable, With non-attrition outweighing attrition heavily. 
# 
# This tells us that there may potentially be a need for over/under sampling.
# 
# It also means that the baselines for our models will be quite high to begin with as with a binary classification problem, the baseline is determined by the percentage of the dominating class.

# In[ ]:


""" Lets plot the target variable to make it easier to visualise. """

""" Set the figure size. """
plt.figure(figsize=(6,4))

""" We will use a countplot here from the seaborn library. """
fig = sns.countplot(df.attrition)

""" sns.despine allows the customisation of the borders in the plot just for aesthetic purposes. """
sns.despine(left=True)

""" Configure the axes labels. """
fig.set_xlabel('Attrition', fontsize=16)
plt.xticks(fontsize=16)
fig.set_ylabel('Count', fontsize=16, rotation=0)
fig.yaxis.labelpad = 30
plt.yticks(fontsize=16)

""" Add annotations on the plot to show the actual count values on each of the columns."""
for p in fig.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    fig.annotate('{:}'.format(p.get_height()), (x.mean(), y-150), ha='center', va='bottom', fontsize=16, color='white')
plt.show()


# In[ ]:


""" Show the percentage breakdown of the target column. This also shows us the Baseline accuracy for our models. (83.9%) """

print('Percentage breakdown of Attrition')
print('-'*33)
round(df.attrition.value_counts(normalize=True)*100,2)


# In[ ]:


""" Plot the distribution of Age. """

plt.figure(figsize=(12,6))
fig = sns.distplot(df.age,kde=False, bins=10, hist_kws=dict(alpha=1))
sns.despine(left=True)
fig.set_xlabel('Age',fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.yaxis.labelpad = 35
plt.show()


# This graph shows a normal distribution with a slightly positive skew as indicated by the longer tail going off toward the right hand side.
# 
# Lets look into the Age feature in more detail in relation to the target variable.

# In[ ]:


""" Plot the distribution of Age where attrition is true and false. """

plt.figure(figsize=(12,8))

""" Adjusting the bin size can alter the look of your graph, worth testing different sizes to see various plots. """
fig = sns.distplot(df[df['attrition'] == 0]['age'], label='Non Attrition', kde=0, bins=10)
sns.distplot(df[df['attrition'] == 1]['age'], label='Attrition', kde=0, bins=10)

sns.despine(left=1)

""" Removes the vertical gridlines. """
fig.grid(axis='x')

plt.xlabel('Age',fontsize=15)
plt.ylabel('Density',fontsize=15, rotation=0)
fig.yaxis.labelpad = 30
plt.title('Distribution of Age',fontsize=20);
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.yaxis.labelpad = 35

""" Control the size and positioning of the legend. """
plt.legend(fontsize='x-large', bbox_to_anchor=(0.03, 0.95), loc=2, borderaxespad=0., frameon=1)
plt.show()


# From this graph, we can see that attrition is present across all the age ranges.
# 
# However, for some age groups, attrition is much more prevalent. This is especially true in the early 20s age group where it is almost equal.
# 
# Attrition seems to be at its lowest in the mid 40s, and steadily increases as it gets closer the retirement age.

# In[ ]:


plt.figure(figsize=(12,8))
fig = sns.countplot(x='gender', hue='attrition', data=df)
sns.despine(left=True)
fig.set_xlabel('Gender', fontsize=20)
plt.xticks(fontsize=20)
fig.set_ylabel('Count', fontsize=20, rotation=0)
fig.yaxis.labelpad = 30
plt.yticks(fontsize=20)
for p in fig.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    fig.annotate('{:}'.format(p.get_height()), (x.mean(), y-50), ha='center', va='bottom', fontsize=20, color='white')
plt.legend(labels =['Non Attrition','Attrition'],fontsize='x-large', bbox_to_anchor=(0.03, 0.9), loc=2, borderaxespad=0., frameon=0)
plt.show()

print('Female Attrition percentage & count')
print('-'*35)
print(round(df[df.gender == 'Female'].attrition.value_counts(normalize=True)*100,2))
print(df[df.gender == 'Female'].attrition.value_counts())
print('_'*35)
print(''*35)
print('Male Attrition percentage & count')
print('-'*35)
print(round(df[df.gender == 'Male'].attrition.value_counts(normalize=True)*100,2))
print(df[df.gender == 'Male'].attrition.value_counts())
print('_'*35)


# In[ ]:


plt.figure(figsize=(14,8))
fig = sns.distplot(df[df['attrition'] == 0]['monthlyincome'], label='Non Attrition', kde=0, bins=10)
sns.distplot(df[df['attrition'] == 1]['monthlyincome'], label='Attrition', kde=0, bins=10)
sns.despine(left=1)
fig.grid(axis='x')
plt.xlabel('Monthly Income',fontsize=18)
plt.ylabel('Density',fontsize=18, rotation=0)
fig.yaxis.labelpad = 30
plt.title('Distribution of Monthly Income',fontsize=20);
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.yaxis.labelpad = 35
plt.legend(fontsize='x-large', bbox_to_anchor=(0.4, 0.94), loc=2, borderaxespad=0., frameon=0)
plt.show()
print('Average Monthly Income:',df.monthlyincome.mean())
print('Average Monthly Income for Males:',df[df.gender == 'Male']['monthlyincome'].mean())
print('Average Monthly Income for Females:',df[df.gender == 'Female']['monthlyincome'].mean())


# In[ ]:


income = df.groupby(by='jobrole').mean().monthlyincome
inc = pd.DataFrame(income)
inc = inc.sort_values(by='monthlyincome')


# In[ ]:


job_atr = df[df['attrition'] == 1]['jobrole']
job_atr_val = job_atr.value_counts()
job_atr_df = pd.DataFrame(job_atr_val)


# In[ ]:


plt.figure(figsize=(12,4))
fig = sns.barplot(y=inc.index, x='monthlyincome', data=inc,
                  palette=sns.color_palette("Greens", n_colors=len(inc.index)))
fig.set_title('AVG monthly income per Job role',fontsize=18)
fig.set_xlabel('Average monthly income', fontsize=18)
fig.set_ylabel('Job role', fontsize=18, position=(0,1), rotation=0)
fig.yaxis.labelpad= -120
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16)
plt.show()

plt.figure(figsize=(12,4))
fig = sns.barplot(y=job_atr_df.index, x='jobrole', data=job_atr_df, 
                  palette=sns.color_palette("Greens_r", n_colors=len(job_atr_df.index)))
fig.set_title('Attrition count per Job role',fontsize=18)
fig.set_xlabel('Attrition count', fontsize=18)
fig.set_ylabel('Job role', fontsize=18, position=(0,1), rotation=0)
fig.yaxis.labelpad= -120
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16)
plt.show()


# Showing the average monthly income per job role on top of the attrition count per job role unveils an interesting relationship with the highest paid jobs averaging the lowest attrition and vice versa. Jobs such as Sales executive however go against this trend with a higher average income whilst recording the second highest attrition rate out of all the job roles.

# In[ ]:


edu_sal = df.groupby('education').mean().monthlyincome
edu_sal_df=pd.DataFrame(edu_sal)
edu_sal_df = edu_sal_df.sort_values('monthlyincome', ascending=False)


# In[ ]:


plt.figure(figsize=(12,4))
fig = sns.barplot(y=edu_sal_df.index, x='monthlyincome', data=edu_sal_df, 
                  palette=sns.color_palette("Greens_r", n_colors=len(job_atr_df.index)))
fig.set_title('Monthly income per Education level',fontsize=18)
fig.set_xlabel('Average Monthly income', fontsize=18)
fig.set_ylabel('Education level', fontsize=18, position=(0,1), rotation=0)
fig.yaxis.labelpad= -50
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
fig = sns.countplot(x='overtime', hue='attrition', data=df)
sns.despine(left=True)
fig.set_xlabel('Overtime', fontsize=20)
plt.xticks(fontsize=20)
fig.set_ylabel('Count', fontsize=20, rotation=0)
fig.yaxis.labelpad = 30
plt.yticks(fontsize=20)
for p in fig.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    fig.annotate('{:}'.format(p.get_height()), (x.mean(), y-50), ha='center', va='bottom', fontsize=20, color='white')
plt.legend(labels =['Non Attrition','Attrition'],fontsize='x-large', bbox_to_anchor=(0.03, 0.9), loc=2, borderaxespad=0., frameon=0)
plt.show()


# ### EDA Summary
# From this brief EDA of the dataset there are numerous things that stand out in relation to attrition.
# ##### Class imbalance
# Class imbalance in the target variable can be problematic when it comes to machine learning, and in relation to this dataset our focus is the minority class. Class imbalance creates a high baseline accuracy for our models to improve on. Methods to reduce this imbalance may be needed such as under/oversampling.
# ###### Attrition & Age
# Attrition seems to be more prevalent in the early career stages, most notebly between the ages of 20 and 30. Whilst there is records of attrition at almost every age grouping, it would be beneficial to retain these younger employees and develop them within the business.
# ###### Attrition & monthly income
# While it isn't groundbreaking that the employees getting paid more are less likely to quit their jobs, exploring this area did uncover certain job roles where despite their pay level, attrition was abnormally higher than other roles. This relationship between income and retention could also be worth exploring in relation to preventing attrition, by offering incentives in the form of pay increases, how much of an effect, if any, would this have on the employees attrition chance.
# ###### Attrition & overtime
# Employees that worked overtime had a much higher rate of attrition than their colleagues that didnt have to. By increasing the total number of hours worked and making workers stay late/arrive early the chance of them leaving increases. Again this provides another potential area of focus to improve employee retention, by minimising overtime hours, employee satisfaction could increase and the chance of attrition could go down for this specific group of employees.

# # 3. Implementing Machine Learning

# In this stage of the project, the aim is to train a model on a subset of the data (training data) that successfully predicts our target variable of a different subset (test data) more accurately than baseline. By utilizing numerous models, it is possible to assess each model used and then fine tune the best performing to output the best possibly accuracy for this dataset.
# 
# As this is a relatively small dataset, i will be using a wide variety of classification models as the computation time will not be very large. however, this strategy would not suit bigger, more complex datasets as running through multiple models may take an excessive amount of time, and in the real world, cost a business unnecessarily.

# In[ ]:


""" Start by loading in the dummied dataset we created earlier for modelling. """
data = '../input/ibm-data/IBM_dummied.csv'
df = pd.read_csv(data, index_col=0)

""" Convert all to float. """
df = df.astype(float)

df.shape


# In[ ]:


df.rename({'attrition_yes':'attrition'}, axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


""" Seperate dataframe into the target and features. """
X = df.drop('attrition', axis=1)
y = df.attrition


# In[ ]:


""" Split the dataframe into the train and test groups. The split size can be specified, for this i am
    setting aside 20% for the testing data."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


""" Find our baseline accuracy, using value_counts and taking the dominating class since this is a binary target.
    Our baseline accuracy is 83.9%"""
y.value_counts(normalize=True)


# In[ ]:


""" Defining the models i am going to use into a list. """
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB()]
    
""" Logging for visual comparison. """ 

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

""" Iterate through each classification model stated above, fitting the model to the train data and finally
    printing the accuracy and log loss of each model. """

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# In[ ]:


""" From this, we can then sort by accuracy and log loss to effectively visualise our results. """

log1 = log.sort_values(by='Accuracy',ascending=False)
log2 = log.sort_values(by='Log Loss')


# In[ ]:


plt.figure(figsize=(12,4))
fig = sns.barplot(x='Accuracy', y='Classifier', data=log1, palette=sns.color_palette("Blues_r", n_colors=len('classifier')))
plt.xlabel('Accuracy %', fontsize=18)
plt.ylabel('Classifier Model',fontsize=18, position=(0,1),rotation=0)
fig.yaxis.labelpad= -125
fig.set_xticks(ticks=[0,10,20,30,40,50,60,70,80,90])
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.title('Classifier Accuracy', fontsize=18)
plt.axvline(83.8, 0,1, lw=4, color='red')
plt.annotate(s='Baseline:83.8%', xy=(75,-0.5), fontsize=16, color='black')
plt.show()

plt.figure(figsize=(12,4))
fig = sns.barplot(x='Log Loss', y='Classifier', data=log2, palette=sns.color_palette("Blues", n_colors=len('Classifier')))
plt.xlabel('Log Loss', fontsize=18)
plt.ylabel('Classifier Model',fontsize=18, position=(0,1),rotation=0)
fig.yaxis.labelpad= -125
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.title('Classifier Log Loss', fontsize=18)
plt.show()


# ### Initial results
# After training all the different classification models and visualising their accuracy and log loss, multiple models managed to predict more accurately than baseline. However, from here we can now choose the best model from this and further tweak this model through model tuning to see if we can improve the score any more.
# 
# The logistic regression model stands out at the moment as our best model to improve on. This makes sense as it is the most common and often most accurate when presenting with a binary target.

# # 4. Model tuning

# Now that we have a model that sufficiently predicts above baseline, we can then apply further techniques to tune this model through various penalties and parameters. This can also include feature selection depending on the model. For this Logisitic Regression model, I will be using GridsearchCV to find the optimal C parameter and penalty to use to get the best accruacy for this dataset.

# In[ ]:


""" Create the parameter grid that will be supplied and applied to the model on each iteration. """

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty':['l1','l2'] }

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(LogisticRegression(), param_grid)
GridSearchCV(cv=None,
             estimator=LogisticRegression(C=1.0, intercept_scaling=1,   
               dual=False, fit_intercept=True, penalty='l1', tol=0.0001),
             param_grid=param_grid)


# In[ ]:


clf.fit(X_train, y_train)
clf.param_grid


# In[ ]:


""" Create a new DataFrame containing the results from the gridsearch, with the C param and penalty associated. """

results = pd.DataFrame(clf.cv_results_)
final = results[['param_C','param_penalty','mean_test_score']].sort_values('mean_test_score')
final


# In[ ]:


""" Using a catplot, show the average model score for each version of the model. """

sns.catplot(y='mean_test_score', x='param_C', hue='param_penalty', data=final, kind='bar', aspect=15/8.27)
plt.axhline(0.83, 0,1, lw=4, color='red')
plt.title('GridsearchCV scores',fontsize=18)
plt.xlabel('C Parameter', fontsize=16)
plt.ylabel('Average model score', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# In[ ]:


""" Print out the best parameters for the model, along with the final score utilizing these parameters. """
print(clf.best_params_)
print("="*30)
print(clf.best_estimator_)
print("="*30)
print(clf.best_score_)
print("="*30)
y_predict = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_predict)
print('Accuracy of the best classifier after CV is %.2f%%' % (accuracy*100))


# # 5. Conclusion

# ##### final model
# Through the use of many different classification models, and tuning the best performing, my final model was predicting with a 90.14% accuracy.
# 
# While this accuracy sounds high enough to deem it a success, looking back on this project, nothing was done to address the class imbalance, which may have affected our model. 
# 
# The other problem with this project is the fact that it is a fictional dataset, meaning that it is not applicable to real world situations, And even if it were, I dont beleive that it could be implemented into a different business as its scope is set purely to the one business. The same steps could be taken and a model could be generated for a different business, however the results may not be the same due to the differing factors present.
# 
# This project was much more of a exploration of what i have learnt whilst studying at General Assembly, Sydney. A showcase of the broad array of subjects covered in the 12 week Data science immersive course. 
# 
# Any feedback would be appreciated, and if you would like to get in contact with me, i am always happy to discuss all things data science!
# My Linkedin & the Github to this notebook is linked below.
# 
# #### Linkedin : https://www.linkedin.com/in/matt-warr/
# 
# #### github : https://github.com/mjw236/IBM_Attrition
# 
# 
