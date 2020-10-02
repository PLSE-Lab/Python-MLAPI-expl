#!/usr/bin/env python
# coding: utf-8

# # Bank Marketing Prediction of Subscription
# 
# ### Author: Vu Duong
# 
# #### Date: May 28th, 2020

# # Credit
# This work is inspired by multiple great sources done before:
# - https://www.kaggle.com/aleksandradeis/bank-marketing-analysis
# - https://www.kaggle.com/janiobachmann/bank-marketing-campaign-opening-a-term-deposit

# # INTRODUCTION
# 
# For most product or any organizations conducting analysis of marketing data is one the most critical skill so as to contribute a huge impact on the financial budget.
# 
#  
# This is a marketing data which can be used for many business goals:
# 1. Customer segmentation who converted and not converted based on the profile of a customer, thus developing more targeted marketing campaigns.
# 2. Derive the marketing campaign result for each customer from multiple factors. This makes sense in the way how to run campaign more efficiently. 
# 
# 
# Detailed description of dataset content is described in the following link: https://archive.ics.uci.edu/ml/datasets/bank+marketing

# # Approach
# To optimize the marketing campaign with dataset description, we follow these steps:
# 1. Importing essential library for data processing, data visualizing, data modeling, data validation
# 2. Importing dataset
# 3. Exploring data by reading description and information, seeing the correlation between features, looking for row numbers and missing values
# 4. Cleaning the data: remove irrelevant columns, generate new features, deal with missing value based on observation and domain knowledge, turn categorical data into dummy variables.
# 5. Building model with different algorithms like Logistic regression, Support Vector Classifier, XGBoost, ... for comparison. 
# 6. Evaluating model with libraries 
# 7. Interpreting model by ploting a graph to see the most important features

# # LIBRARY

# In[ ]:


# Data Processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline


# Data Visualizing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML

# Math
import math
from scipy.stats import norm, skew
from scipy import stats
from scipy.stats import boxcox
from scipy.special import boxcox1p

# Data Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC, NuSVC
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Data Validation
from sklearn import metrics

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# # FEATURE ENGINEERING

# ### EXLORATORY
# - read_csv
# - head
# - describe
# - info
# - correlation

# In[ ]:


df = pd.read_csv('../input/bank-additional-full.csv',sep=';')


# In[ ]:


df


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.info(verbose=True)


# Convert outcomes from the result of our customers to 0 as no, and 1 as yes so that we can see the inital correlations between features with the target.

# In[ ]:


df['y'] = df['y'].map( {'no':0, 'yes':1})


# In[ ]:


df.corr()['y'].sort_values(ascending = False)


# ### VISUALIZATION
# - dist: numerical
# - count: categorical
# - scatter: numerical + categorical
# - box: numerical
# - bar

# #### Categorical features
# 
# job, marital, education, default, housing, loan, contact, month,poutcome

# In[ ]:


def plot_cat_features(cat_features, nrows, ncols):
    fig = plt.figure(constrained_layout=True, figsize=(12, 5))

    grid = gridspec.GridSpec(nrows=nrows, ncols=ncols,  figure=fig)
    row = 0
    column = 0
    for cat in cat_features:
        ax1 = fig.add_subplot(grid[0, column])
        sns.countplot(df[cat], ax=ax1)
        plt.xticks(rotation=90)

        column += 1
        if column == ncols:
            column = 0
            row += 1


# ##### Observation
# - Job : Admin, Blue-collar, Technician come as the 1st, 2nd, 3rd largest propotion of all, while unknown and housemaid account for smallest parts.
# - Education: University and high school are 2 largest, whereas unknown and illiterate hold tiny segments.
# - Marital:  married is double number of single and fourfold as divorced. Unknown seems to hit 0.
# 
# ==> There may be a relationship between job and education and partially impacted by martial
# - As we may predict, having university degree tend to become admin, technician or services. While education at the level of high school or 9 years may end up being blue-collar.
# - For those with lower education level or older may get divorced, as the majority of the young are single or married.

# In[ ]:


plot_cat_features(['job','education', 'marital'], 1, 3)


# ##### Observation
# - Default: most people do not default
# - Housing: the number of Yes is sightly higher than that of No, implying people have tendency toward house financing 
# 
# ==> Loan: even some people make a loan or housing but they have not defaulted yet. 

# In[ ]:


plot_cat_features(['default', 'housing', 'loan'], 1, 3)


# ##### Observation
# - Contact: cellular is as double number as telephone
# - Month: people made most contact in may, and least in december
# - Poutcome: failure number is higher than success number is

# In[ ]:


plot_cat_features(['contact', 'month','poutcome'], 1, 3)


# #### Numerical features
# 
# balance, day, duration, campaign, pdays, previous

# In[ ]:


def plot_numer_features(numer_features, nrows, ncols):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    row = 0
    column = 0
    for numer in numer_features:
        ax1 = fig.add_subplot(grid[row, column])
        sns.distplot(df[numer], kde=False, ax=ax1)
        plt.xticks(rotation=90)

        column += 1
        if column == ncols:
            column = 0
            row += 1


# ##### Observation
# - Age: the distribtion is right skewed. Some people are up to almost 100 years old, which means they are outlilers.
# - Duration: the distribtion is right skewed. Later, we will explore when people spend more time on last contact they are likely to convert.
# - Campaign: the distribtion is right skewed. We will explore if more contacts will lead to conversion or not.
# - Pdays: most people are not contacted by a previous campaign, 999 explains.
# - Previous: the number of contacts before this campaign is range between 0 and 6. Most people were not contacted previously.

# In[ ]:


plot_numer_features(['age','duration', 'campaign','pdays', 'previous'], 2, 3)


# In[ ]:


def plot_numer_features(numer_features, nrows, ncols):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    grid = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    row = 0
    column = 0
    for numer in numer_features:
        ax1 = fig.add_subplot(grid[row, column])
        sns.distplot(df[numer], kde=False, ax=ax1)
        plt.xticks(rotation=90)

        column += 1
        if column == ncols:
            column = 0
            row += 1


# #### Observation
# - Employment Variation Rate: there are some points on the negative side.
# - Consumer Price Index: over 2 years from 2008 to 2010, the figure fuctuate around 92 and 95
# - Consumer Confidence Index: levels of optimism regarding current economic conditions are seriously hopeless.
# 
# ==> The Great Recession period between 2007 and 2009 marked global declination. This was said as the worst  financial crsis in global history.

# In[ ]:


plot_numer_features(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m'], 2, 2)


# ### Analysis of the Target feature (deposit)

# #### Categorical features affect Target

# In[ ]:


def plot_cat_features_withTarget(cat_features, nrows, ncols):
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))

    grid = gridspec.GridSpec(nrows=nrows, ncols=ncols,  figure=fig)
    row = 0
    column = 0
    for cat in cat_features:
        ax1 = fig.add_subplot(grid[row, column])
        sns.countplot(df[cat],hue=df['y'], ax=ax1)
        plt.xticks(rotation=90)

        column += 1
        if column == ncols:
            column = 0
            row += 1


# In[ ]:


fig = plt.figure()
sns.countplot(df['y'])


# ##### Observation
# - Job: Student are most likely to convert, as oppose to blue-collar people.
# - Education: apart from unknown and illterate groups, people with university and professional level are likely to convert, as oppose to those at 4, 6, and 9 year level.
# - Marital: for those being single are more likely to convert, while people with divorced and married label are similar when it comes to conversion.

# In[ ]:


df.groupby('contact')['y'].mean().sort_values(ascending=False)


# In[ ]:


pd.DataFrame(df.groupby('contact')['y'].mean().sort_values(ascending=False))


# In[ ]:


plot_cat_features_withTarget(['job', 'education', 'marital', 'contact'], 2, 2)


# In[ ]:


# Check our declaration above
def check_observation (features):
    for fea in features:
        display(pd.DataFrame(df.groupby(fea)['y'].mean().sort_values(ascending=False)))

check_observation(['job','education','marital','contact'])


# #### Numerical features affect Target

# ##### Observation
# - Non-converted and converted group share similar age distribution in which age average is around 40 

# In[ ]:


def plot_cat_features_withTarget(features):
    for feat in features:
        target_0 = df.loc[df['y'] == 0, feat]
        target_1 = df.loc[df['y'] == 1, feat]

        sns.distplot(target_0,hist=False, rug=True, label='non-converted')
        sns.distplot(target_1,hist=False, rug=True, label='converted')
        plt.show()
    
        # Source: https://stackoverflow.com/questions/26873127/show-dataframe-as-table-in-ipython-notebook/29665452#29665452
        display(df.groupby('y')[feat].describe())


# In[ ]:


plot_cat_features_withTarget(['age'])


# ### Data Cleaning
# - Preparing dataset before apply machine learning algorithms
# - Group numerical features into bins for the ease of analysis
# - Fill unknown or null values from those features based on the visualization above and our understanding of dataset
# - Convert categorical into dummy variables

# #### Numerical features clearning
# duration, previous, age, campaign, cons.conf.idx, cons.price.idx, emp.var.rate, euribor3m

# ##### Observation
# - Here I group those features into bins. According to my experiment, this does not improve accuracy but more computationally expensive on traning.
# - We can safely remove those new features. However I decided to leave hear.

# In[ ]:


bins = [-1, 250, 500, 750, np.inf]
names = [0, 1, 2, 3]
df['new_duration'] = pd.cut(df['duration'], bins, labels=names).astype('int')

bins = [-1, 1, np.inf]
names = [0, 1]
df['new_previous'] = pd.cut(df['previous'], bins, labels=names).astype('int')

bins = [10, 20, 25, 60, 100]
names = ['Youth','Millennials','WorkingClass','RetirementClass']
df['new_age_Class'] = pd.cut(df['age'], bins, labels=names)

bins = [-1, 3, 6, 60]
names = ['< 3Times','< 6Times','< 100Times']
df['new_campaign'] = pd.cut(df['campaign'], bins, labels=names)

bins = [-65, -50, -40, -30, -20]
names = ['-65,-50','-50,-40','-40,-30','-30,-20']
df['new_cons.conf.idx'] = pd.cut(df['cons.conf.idx'], bins, labels=names)

bins = [91, 93, 94, 96]
names = ['<92-93','93-94','94-95']
df['new_cons.price.idx'] = pd.cut(df['cons.price.idx'], bins, labels=names)

bins = [-3.5, -1, 1.5]
names = ['-3.5to-1','-1to1.4']
df['new_emp.var.rate'] = pd.cut(df['emp.var.rate'], bins, labels=names)

bins = [0, 1, 6]
names = ['0to1','1to6']
df['new_euribor3m'] = pd.cut(df['euribor3m'], bins, labels=names)


# In[ ]:



"""
#arr = np.array([1,np.nan])
#np.unique(df['nr.employed'])
#df['nr.employed'].unique()

Check the null value
    df['cons.conf.idx'].isnull().sum()

"""


# #### Categorical feature cleaning
# job, education, marital, default, housing, loan

# ##### Observation
# - Replace unknown value with nan for an ease of filling missing data
# - Using groupby function as below and Tableau Prep Builder to know how to fill in missing data.
# - The feature engineering does not help accuracy improvement as compared to non-feature engineering.

# In[ ]:


df['job'].replace('unknown', np.nan, inplace=True)
df['education'].replace('unknown', np.nan, inplace=True)
df['marital'].replace('unknown', np.nan, inplace=True)
df['default'].replace('unknown', np.nan, inplace=True)
df['housing'].replace('unknown', np.nan, inplace=True)
df['loan'].replace('unknown', np.nan, inplace=True)


# In[ ]:


df.groupby('education')['age'].agg(['mean','median'])


# In[ ]:


#https://stackoverflow.com/questions/44061607/pandas-lambda-function-with-nan-support/44061892#44061892
def categorical_feature_cleaning(df):
# Job Cleaning
    if ((pd.isnull(df['job'])) & ((df['education']=='basic.4y') | (df['education']=='basic.6y') | (df['education']=='basic.9y'))):
        df['job'] = 'blue-collar'
    elif ((pd.isnull(df['job'])) & (df['education']=='high.school')):
        df['job'] = 'services'
    elif ((pd.isnull(df['job'])) & (df['education']=='professional.course')):
        df['job'] = 'technician'    
    elif ((pd.isnull(df['job'])) & (df['education']=='university.degree') & (df['marital']=='single')):
        df['job'] = 'admin'
    elif ((pd.isnull(df['job'])) & (df['education']=='university.degree')):
        df['job'] = 'management'            
    
# Education Cleaning    
    if ((pd.isnull(df['education'])) & (df['job']=='blue-collar') & (df['age']<=39)):
        df['education'] = 'basic.6y'
    elif ((pd.isnull(df['education'])) & (df['job']=='blue-collar') & (df['age']<=100)):
        df['education'] = 'basic.4y'
    elif ((pd.isnull(df['education'])) & (df['job']=='housemaid')):
        df['education'] = 'basic.4y'
    elif ((pd.isnull(df['education'])) & (df['job']=='retired')):
        df['education'] = 'basic.4y'
    
    elif ((pd.isnull(df['education'])) & (df['job']=='admin.')):
        df['education'] = 'university.degree'
    elif ((pd.isnull(df['education'])) & (df['job']=='entrepreneur')):
        df['education'] = 'university.degree'
    elif ((pd.isnull(df['education'])) & (df['job']=='management')):
        df['education'] = 'university.degree'
    elif ((pd.isnull(df['education'])) & (df['job']=='self-employed')):
        df['education'] = 'university.degree'
    
    elif ((pd.isnull(df['education'])) & (df['job']=='services')):
        df['education'] = 'high.school'
    elif ((pd.isnull(df['education'])) & (df['job']=='student')):
        df['education'] = 'high.school' 
    elif ((pd.isnull(df['education'])) & (df['job']=='unemployed')):
        df['education'] = 'high.school'
        
    elif ((pd.isnull(df['education'])) & (df['job']=='technician')):
        df['education'] = 'professional.course'
        
# both Job and Education have missing values
    if ((pd.isnull(df['job'])) & (pd.isnull(df['education']))):
        if (df['age'] <= 36):
            df['job'] = 'student'
            df['education'] = 'university.degree'
        elif (df['age'] < 55):
            df['job'] = 'blue-collar'
            df['education'] = 'basic.9y'
        else:
            df['job'] = 'retired'
            df['education'] = 'basic.4y'

# # Marital cleaning
    if pd.isnull(df['marital']):
        df['marital'] = 'married'

# Default cleaning
    if pd.isnull(df['default']):
        df['marital'] = 'no'
                 
# Housing cleaning
    if pd.isnull(df['housing']):
        df['marital'] = 'yes'

# Loan cleaning
    if pd.isnull(df['loan']):
        df['loan'] = 'no'
# Number Employed
    if pd.isnull(df['nr.employed']):
        df['nr.employed'] = 5191  
    
    return df


# In[ ]:


df = df.apply(categorical_feature_cleaning, axis=1)


# #### One-hot Encoding

# In[ ]:


df = pd.get_dummies(df, drop_first=True)
df1 = df


# ### Train-Test Split
# - Using Robust Scaler gives better result on prediction as it mitigate the pain of outliers
# - Using stratify as the way to make sure proporation of sample of each label in train and test dataset are the similar.

# In[ ]:


x = RobustScaler().fit_transform(df.drop('y', axis=1))
y = df['y']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, stratify=y, random_state=0)


# # DATA MODELING

# ### Models

# In[ ]:


logreg = LogisticRegression(n_jobs=-1, solver='newton-cg')

knn = KNeighborsClassifier(n_neighbors=13)

gnb = GaussianNB()

linearSVC = LinearSVC()

RbfSVC = SVC()

dt = DecisionTreeClassifier(max_depth=10)

rf = RandomForestClassifier(random_state=0,n_jobs=-1,verbose=0)

adab = AdaBoostClassifier(random_state=0)

gb = GradientBoostingClassifier(random_state=0)

xgb = XGBClassifier(random_state=0)

lgbm = LGBMClassifier(random_state=0)


# ### Model Evaluation
# - Use Best_Parameters_model below as to choose the best parameter yeilding the best prediction. However it is time comsuming a lot.
# - Use model_check to compare among models and choose the best.

# In[ ]:


def Best_Parameters_model(est, para):
    model_table = {}
    
    MLA_name = est.__class__.__name__
    model_table['Model Name'] = MLA_name

    pipe = make_pipeline(GridSearchCV(estimator=est,
                                      param_grid=para,
                                      scoring='accuracy',
                                      cv=3,
                                      n_jobs=-1,
                                      verbose=0, refit=True))
    pipe_result = pipe.fit(x_train, y_train)

    model_table['Best Test Accuracy Mean'] = pipe_result[0].best_score_
    model_table['Best Parameters'] = pipe_result[0].best_params_
    model_table['Test Dataset Score'] = pipe.score(x_test, y_test)
    return model_table

# GBoost = GradientBoostingClassifier(random_state=0)
# GBoost_para = {'learning_rate':[0.1, 0.01],
#                'n_estimators':[100,500],
#                'subsample':[0.9, 0.95],
#                'criterion':['friedman_mse'],
#                'min_samples_split': [4, 5, 6],
#                'max_depth':[3, 4, 5],
#                'max_features':['sqrt']
#               }
# Best_Parameters_model(GBoost,GBoost_para)


# In[ ]:


cv = StratifiedKFold(10, shuffle=True, random_state=0)


def model_check(X, y, estimators, cv):
    model_table = pd.DataFrame()
    
    row_index = 0
    for est in estimators:

        MLA_name = est.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name
        #    model_table.loc[row_index, 'MLA Parameters'] = str(est.get_params())

        est.fit(x_train, y_train)
        
        model_table.loc[row_index, 'Train Accuracy Mean'] = est.score(x_train, y_train)
        model_table.loc[row_index, 'Test Accuracy Mean'] = est.score(x_test, y_test)

        row_index += 1

        model_table.sort_values(by=['Test Accuracy Mean'],
                            ascending=False,
                            inplace=True)

    return model_table


# In[ ]:


estimators = [logreg,knn,gnb,linearSVC,RbfSVC,dt,rf,gb,xgb,lgbm]


# In[ ]:


raw_models = model_check(x, y, estimators, cv)
display(raw_models.style.background_gradient(cmap='summer_r'))


# ### Feature Importance

# ##### Observation
# 
# As we can see from 4 diagram below, the most important features are:
# - Duration
# - nr.employed
# - euribor3m
# - pdays
# - campaign
# - consumer confident index
# - age
# 
# Therefore the outcomes would be:
# - The more time customer spend with the bank, the higher chance they convert.
# - The higher number of employees, as we extracted this information from the internet, the more likely they convert
# - Pday is important as well. The number of days passing by since the last contact with customer.

# In[ ]:


nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("DecisionTree", dt), ("RF", rf), ("GradientBoosting", gb), ("XGB", xgb)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=df.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1


# ### Recommendations

# ##### Observation 1: Duration
# For those spending great amount of time discussing term deposit are more likely to take subscription.

# ##### Observation 2-3: job and age
# As we have seen the diagram of job, age below:
# - We should focus on 3 job groups of student, retired, and unemployed.
# - For student, we focus on those below 26 years old. 
# - For retired, we focus on those above 60 years old. 
# - For unemployed, we focus on those around 40 years old.
# 
# The reasons behind might be that students work as a part-time job and save their money in saving account to earn interest. Retired individuals tend to have term deposits to gain interest payment from the bank until the due date. This group tend not to spend greatly on personal interest rather leading it to the financial institution. 
# 
# Thus, it will be great if the next campaign addressed these 3 categories, increasing the likelihood of more subscriptions.

# In[ ]:


df1 = pd.read_csv('../input/bank-additional-full.csv',sep=';')
df1['y'] = df1['y'].map( {'no':0, 'yes':1})


# In[ ]:


a = df1[(df1['job']=='student') | (df1['job']=='retired') | (df1['job']=='unemployed')]


# In[ ]:


sns.countplot(a['job'], hue=a['y'])


# In[ ]:


print('Student average age: ', df1[df1['job']=='student']['age'].median())
print('Retired average age: ', df1[df1['job']=='retired']['age'].median())
print('Unemployed average age: ', df1[df1['job']=='unemployed']['age'].median())


# ##### Observation 4: campaign
# As we have seen campaign diagram below:
# - The more campaign display to the same group of people does not guarantee to turn ones into our customers

# In[ ]:


df1['campaign_buckets'] = pd.qcut(df1['campaign'], 20, labels=False, duplicates = 'drop')

mean_campaign = df1.groupby(['campaign_buckets'])['y'].mean()
mean_campaign = mean_campaign.reset_index()

sns.lineplot(x=mean_campaign['campaign_buckets'],y=mean_campaign['y'])
plt.ylabel('Percentage of Success')
plt.show()


# ##### Observation 5: pdays
# As we have seen pdays diagram below:
# - If a person had been exposed to one of our marketing campaign before, there would be above 60% of success to make a person to convert.

# In[ ]:


bins = [-1, 50, np.inf]
names = [0, 1]
df1['pdays_bucket'] = pd.cut(df1['pdays'], bins, labels=names).astype('int')

mean_campaign = df1.groupby(['pdays_bucket'])['y'].mean()
mean_campaign = mean_campaign.reset_index()

sns.lineplot(x=mean_campaign['pdays_bucket'],y=mean_campaign['y'])
plt.ylabel('Percentage of Success')
plt.show()

