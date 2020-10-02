#!/usr/bin/env python
# coding: utf-8

# ## Titanic: Machine Learning from Disaster Challenge [Beginner-Intermediate] version 1
# Use Machine Learning to create the model that predicts which passenger survived from Titanic shipwreck. Dataset available at https://www.kaggle.com/c/titanic/data
# 
# ### Steps to create the model for prediction
# 1. Define problem
# 2. Data collection
# 3. Data preprocessing
# 4. Feature engineering
# 5. Exploratory Data Analysis with Statistics (EDA)
# 6. Create model data
# 7. Training the model
# 8. Evaluate the model performance
# 9. Tune model with hyperparameters
# 10. Tune model with feature selections
# 11. Validate and implement
# 12. Optimize and strategy

# ## Load all library or package for Machine Learning
# There are many library/package that used in machine Learning purpose such as scikit-learn, scipy, pandas, numpy, matplotlib, seaborn, etc.

# In[ ]:


# Load libraries or packages for Machine Learning purposes
#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
import lightgbm as lgb

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.impute import SimpleImputer

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# ## Step 1. Define problem
# RMS (Royal Mail Ship) Titanic, British luxury ship that sank on 14-15, 1912 during its mayden voyage en route to New York city from Southampton, England after colliding with iceberg. Killing about 1500 people including passengers and crew. One of the reason that shipwreck led to such loss of life was because there were not enough of lifeboats for passengers and crew altough there are some elements of luck for some groups of people that likely have more chance to survive such as women, children, and upper-class. The challenge is to make model that can predict the people are likely to survive from the infamous tragedy. Source: https://www.britannica.com/topic/Titanic

# ## Step 2. Data collection
# in this case, the data is taken from Kaggle Titanic: Machine Learning from disaster. So we no need to find out or collect the raw data from another source using some methods like web scraping, survey or online quiz, and interviews. Data source: https://www.kaggle.com/c/titanic/data

# ## Step 3. Data preprocessing
# In data preprocessing, the data is removed from missing values, duplicates, aberrant data. Data type conversion in categorical variables and also normalization if needed.

# In[ ]:


# Step 3. Data preprocessing
# load data
train_data_full=pd.read_csv('../input/titanic/train.csv')
test_data_full=pd.read_csv('../input/titanic/test.csv')

# omit missing values on target data (Survived)
train_data_full.dropna(subset=['Survived'], axis=0, inplace=True)

# preview data
# print(train_data_full.head())
# print(test_data_full.head())

# view missing values and summary of data
train_miss_cols=train_data_full.columns[train_data_full.isnull().sum()>0]
test_miss_cols=test_data_full.columns[test_data_full.isnull().sum()>0]
print('-'*10)
print('Missing values on train data:')
print(train_data_full[train_miss_cols].isnull().sum())
print('-'*10)
print('Train data summary:')
print(train_data_full.describe())
print('-'*10)
print('Missing values on test data:')
print(test_data_full[test_miss_cols].isnull().sum())
print('-'*10)
print('Test data summary:')
print(test_data_full.describe())

# ignore/drop Cabin column because there are so many missing values
# drop PassengerId in train data because it's just an ID for testing
# drop Ticket because it's determined by Pclass
train_data_full.drop(['Cabin','PassengerId','Ticket'], axis=1, inplace=True)
test_data_full.drop(['Cabin','Ticket'], axis=1, inplace=True)
train_miss_cols=train_miss_cols.drop(['Cabin'])
test_miss_cols=test_miss_cols.drop(['Cabin'])

# data cleaner for data preprocessing
data_cleaner=[train_data_full, test_data_full]

# show distribution on data that have missing values to decide the type of categorical encoder that fits to data that have missing values
# filter categorical columns that have missing values 
train_miss_cat=train_data_full[[i for i in train_miss_cols if train_data_full[i].dtypes=='object']]
test_miss_cat=test_data_full[[i for i in test_miss_cols if test_data_full[i].dtypes=='object']]
miss_cat_data=[train_miss_cat, test_miss_cat]

# encode with label encoder for categorical columns in missing columns 
def label_encoder(df):
    label_encoder = LabelEncoder()
    for col_name in df.columns:
        series = df[col_name]
        df[col_name] = pd.Series(
            label_encoder.fit_transform(series[series.notnull()]),
            index=series[series.notnull()].index
        )
    return df
encoded_miss_cat_data=[label_encoder(i) for i in miss_cat_data]
# join numeric type with encoded categorical columns in train/test data that have missing values
train_miss=train_data_full[list(set(train_miss_cols)-set(train_miss_cat))].join(encoded_miss_cat_data[0])
test_miss=test_data_full[list(set(test_miss_cols)-set(test_miss_cat))].join(encoded_miss_cat_data[1])

# view data distribution with pairplot seaborn on variables that contain missing values
data_miss=train_data_full[train_miss_cols].join(test_data_full[test_miss_cols].add_suffix('_Test'))
data_miss_plot=train_miss.join(test_miss.add_suffix('_Test'))
skew_val=data_miss_plot.skew(axis=0, skipna=True)
print('Missing values data skewness: \n',skew_val)
g0=sns.pairplot(data_miss_plot, diag_kind='kde')
g0=g0.fig.suptitle('Train and Test data distribution', y=1)

# show boxplot to see skewness and outliers
plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.9)
max_col=len(data_miss_plot.columns)
n=max_col-1
for idx,col in enumerate(data_miss_plot.columns):
    plt.subplot(1,max_col,max_col-n)
    sns.boxplot(data=data_miss_plot[col], showmeans = True, meanline = True)
    plt.title(col)
    n-=1


# #### Define Imputation method based on data distribution on variables that have missing values
# A distribution is simply a collection of data, or scores, on a variable. Usually, these scores are arranged in order from smallest to largest and then they can be presented graphically [Page 6, Statistics in Plain English, Third Edition, 2010].
# A sample of data will form a distribution and by far the most well-known distribution is Gaussian distribution or normal distribution. Skewness is the degree of distortion from the symmetrical bell curve or normal distribution. It measures the lack of symmetry in data distribution. It differentiates extreme values in one versus the other tail. Normal distribution will have a skewness of 0.
# <img src="https://www.dropbox.com/s/gmyeqwieueedmz8/distribution.png?raw=1" width="500px">
# * **Positive Skewness** means when the tail on right side of the distribution is longer or fatter. The mean and median will be greater than the mode.
# * **Negative Skewness** means when the tail on the left side of the distribution is longer or fatter. The mean and median will be less than the mode. 
# We will use data imputation method to handle missing values. There 3 strategies that can be uses in data imputation to fill the missing values such as mean, median, and mode. Because the median is mostly greater than mean in a skewness distribution, so we will use **strategy=median** for skewness and **strategy=mean** for normal distribution in continuous variable. And use **strategy=mode** for categorical variable. But I don't think this matters a lot, you can use mean too.

# In[ ]:


# impute missing values
# from data distribution and box plot above can be known that:
# Age and Age_Test=positive skewness -> strategy=median
# Embarked=categorical variable -> strategy=mode
# Fare=positive skewness -> strategy=median
imp_med=SimpleImputer(strategy='median')
imp_mod=SimpleImputer(strategy='most_frequent')
for col in skew_val.index.drop('Embarked'):
    data_miss[col]=pd.DataFrame(imp_med.fit_transform(data_miss[[col]]))
data_miss['Embarked']=pd.DataFrame(imp_mod.fit_transform(data_miss[['Embarked']]))
# check for missing values
print(data_miss.isnull().sum())

# join imputed missing values to train/test data
train_miss_imputed=data_miss[train_miss.columns]
test_miss_imputed_cols=list(set(data_miss.columns)-set(train_miss.columns))
test_miss_imputed_cols.sort(reverse=True)
test_miss_imputed=data_miss[test_miss_imputed_cols].rename(columns=dict(zip(test_miss_imputed_cols, test_miss.columns)))
train_data_full.drop(train_miss_imputed.columns, axis=1, inplace=True)
test_data_full.drop(test_miss_imputed.columns, axis=1, inplace=True)
train_data_full=train_data_full.join(train_miss_imputed)
test_data_full=test_data_full.join(test_miss_imputed)
# preview encoded data (train, test) without missing values
print('Check missing values on baseline data')
print('-'*10)
print(train_data_full.isnull().sum())
print('-'*10)
print(test_data_full.isnull().sum())
train_data_full.head()


# ## Step 4. Feature Engineering
# * Create Title feature
# Create Title feature from Name such as Mr, Miss, Mrs, etc. It will help uas to categorize passengers to children, young, old, etc.

# In[ ]:


data_df=[train_data_full, test_data_full]
# cleaning name and extracting Title
for df in data_df:
    df['Title']=df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# replacing rare Title with more common ones and also drop column Name
mapping={'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
         'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
for df in data_df:
    df.replace({'Title':mapping}, inplace=True)
# preview train dataset
# data_df[0].sample(5)


# * Adding Family_Size
# Family_Size=Parch+SibSp

# In[ ]:


# add Family_Size in each dataset
for df in data_df:
    df['Family_Size']=df['Parch']+df['SibSp']
# preview train dataset
# data_df[0].sample(5)


# * Making Fare Bins
# Binning (quantization/dicretization) is used to transform continuous numeric features to discrete ones (categories). Each bin represents a specific degree of intensity and hence specific range of continuous numeric values that fall into it. Qcut (quantile cut) is quantile-based discretization. It's ordinal, FareBin=3 is indeed greater than FareBin=1. [ https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html ].

# In[ ]:


# making bins
label=LabelEncoder()
for df in data_df:
    # create bin
    df['FareBin']=pd.qcut(df['Fare'], 5)
    # use label encoder for categorical variables
    df['FareBin_Code']=label.fit_transform(df['FareBin'])
    # drop Fare column
# preview train dataset
# data_df[0].sample(5)


# * Making Age Bins
# Create bins for Age using qcut() function

# In[ ]:


label=LabelEncoder()
for df in data_df:
    df['AgeBin']=pd.qcut(df['Age'], 4)
    df['AgeBin_Code']=label.fit_transform(df['AgeBin'])
# preview train dataset
# data_df[0].sample(5)


# #### Handle Categorical variables and create baseline dataset
# Categorical variables contain a finite number of categories or distinct groups. Categorical data might not have logical order. Example: male or female, young or old, etc. Some Machine Learning algorithm cannot handle categorical data, so we must handle it before feeding to algorithm. There are some appoaches to handle categorical data such as **Drop** categorical variables, **Label encoding**, and **One-hot encoding**.

# In[ ]:


# check categorical variables
cat_cols=[]
for df in data_df:
    cat_cols.append([col for col in df.columns if df[col].dtypes=='object'])
# print(cat_cols)
# print('-'*10)
    
# use label encoder to encode categorical variables
label=LabelEncoder()
train_encoded=data_df[0][cat_cols[0]].apply(label.fit_transform)
test_encoded=data_df[1][cat_cols[1]].apply(label.fit_transform)

# use one-hot encoder
oh_cols=['Title', 'Sex', 'Embarked']
train_encoded_oh=pd.get_dummies(data_df[0][['Title','Sex','Embarked']]).add_suffix('_OH')
test_encoded_oh=pd.get_dummies(data_df[1][['Title','Sex','Embarked']]).add_suffix('_OH')
OH_encoded_cols=train_encoded_oh.columns

# join encoded columns to dataset
train_data_full=data_df[0].join(train_encoded.add_suffix('_Code'))
test_data_full=data_df[1].join(test_encoded.add_suffix('_Code'))
# join one-hot encoded columns to dataset
train_data_full=train_data_full.join(train_encoded_oh)
test_data_full=test_data_full.join(test_encoded_oh)

# select features for baseline and engineered dataset
baseline_feature_cols=['Pclass', 'Sex_Code', 'Embarked_Code', 'Title_Code', 'Family_Size']
engineered_feature_cols=['Pclass', 'Family_Size', 'FareBin_Code', 'AgeBin_Code']+list(OH_encoded_cols)
train_data_full[engineered_feature_cols].head()


# #### Split dataset for training, validation and testing data
# Training data will be split into training, validation and testing data with a 80/10/10 ratio. In this case we will use **train_test_split** from sklearn.

# In[ ]:


# create function for data splits and train model
def get_data_splits(df, valid_fraction=0.1):
    valid_size=int(len(df)*valid_fraction)
    train=df[:-valid_size*2]
    valid=df[-valid_size*2:-valid_size]
    test=df[-valid_size:]
    return train,valid,test

# split training data for train, valid and test
train_,valid_,test_=get_data_splits(train_data_full)


# ## Step 5. Exploratory Data Analysis with Statistics
# Now our data is cleaned and ready to feed to model for training and testing. But before that, we will explore our data with descriptive and graphical statistics to describe and summarize the variables. We will classifying features and determining correlation between features and target variable.

# In[ ]:


# discrete variable correlation on Train data by survival using groupby
for col in train_data_full[['Pclass','Sex','Embarked', 'FareBin', 'AgeBin', 'Title']]:
    print('Survival correlation by: ', col)
    print(train_data_full[[col, 'Survived']].groupby(col, as_index=False).mean())
    print('-'*10)


# In[ ]:


# graph distribution for quantitative data (Age, Fare, Family size) on training data
plt.figure(figsize=(14,12))
# plt.subplots_adjust(wspace=0.9, hspace=0.3)

# Fare boxplot
plt.subplot(231)
sns.boxplot(data=train_data_full['Fare'], showmeans=True, meanline=True)
plt.title('Fare Distribution')
plt.ylabel('Fare ($)')
# Age subplot
plt.subplot(232)
sns.boxplot(data=train_data_full['Age'], showmeans=True, meanline=True)
plt.title('Age Distribution')
plt.ylabel('Age (Years)')
# Family size boxplot
plt.subplot(233)
sns.boxplot(data=train_data_full['Family_Size'], showmeans=True, meanline=True)
plt.title('Family size Distribution')
plt.ylabel('Family size (#))')
# Fare vs survived histogram plot
plt.subplot(234)
plt.hist(x=[train_data_full[train_data_full['Survived']==1]['Fare'], 
           train_data_full[train_data_full['Survived']==0]['Fare']], color=['g','r'], label=['Survived', 'Dead'], bins=20)
plt.title('Fare by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of passengers')
plt.legend()
# Age vs survival
plt.subplot(235)
plt.hist(x=[train_data_full[train_data_full['Survived']==1]['Age'], 
           train_data_full[train_data_full['Survived']==0]['Age']], color=['g','r'], label=['Survived', 'Dead'], bins=30)
plt.title('Age by Survival')
plt.xlabel('Age (years)')
plt.ylabel('# of passengers')
plt.legend()
# Family size vs survival
plt.subplot(236)
plt.hist(x=[train_data_full[train_data_full['Survived']==1]['Family_Size'], 
           train_data_full[train_data_full['Survived']==0]['Family_Size']], color=['g','r'], label=['Survived', 'Dead'], bins=20)
plt.title('Family size by Survival')
plt.xlabel('Family size (#)')
plt.ylabel('# of passengers')
_=plt.legend()


# In[ ]:


# graph for categorical data (Title, Sex, Pclass, Embarked, FareBin, AgeBin)
# graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(14,12))
sns.barplot(x='Title', y='Survived', data=train_data_full, ax=saxis[0,0])
saxis[0,0].set_title('Title vs Survived')
sns.barplot(x='Sex', y='Survived', data=train_data_full, ax=saxis[0,1])
saxis[0,1].set_title('Sex vs Survived')
sns.barplot(x='Pclass', y='Survived', data=train_data_full, ax=saxis[0,2])
saxis[0,2].set_title('Pclass vs Survived')
sns.barplot(x='Embarked', y='Survived', data=train_data_full, ax=saxis[1,0])
saxis[1,0].set_title('Title vs Embarked')
farebinplot=sns.barplot(x='FareBin', y='Survived', data=train_data_full, ax=saxis[1,1])
farebinplot.set_xticklabels(farebinplot.get_xticklabels(), rotation=45, horizontalalignment='right')
saxis[1,1].set_title('FareBin vs Survived')
sns.barplot(x='AgeBin', y='Survived', data=train_data_full, ax=saxis[1,2])
_=saxis[1,2].set_title('AgeBin vs Survived')


# In[ ]:


# graph distribution of qualitative data Pclass compared to other features
# Pclass is mattered for survival
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,7))

sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=train_data_full, ax=axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=train_data_full, ax=axis2)
axis2.set_title('Pclass vs Age Survival')
sns.boxplot(x='Pclass', y='Family_Size', hue='Survived', data=train_data_full, ax=axis3)
_=axis3.set_title('Pclass vs Family Size Survival Comparison')


# In[ ]:


# graph distribution of qualitative data Sex compared to other features
# Sex is mattered for survival
fig, saxis = plt.subplots(1,3,figsize=(14,7))

sns.barplot(x='Sex', y='Survived', hue='Embarked', data=train_data_full, ax=saxis[0])
saxis[0].set_title('Sex vs Embarked Survival Comparison')
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train_data_full, ax=saxis[1])
saxis[1].set_title('Sex vs Pclass Survival')
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train_data_full, ax=saxis[2])
_=saxis[2].set_title('Sex vs Pclass Survival')


# In[ ]:


# pairplot of entire dataset
display_cols=['Survived','Pclass','SibSp','Parch','Fare', 'Age','Family_Size','FareBin_Code', 'AgeBin_Code', 'Title_Code']
pp=sns.pairplot(train_data_full[display_cols], hue='Survived', palette='deep', size=1.2, diag_kind='kde',diag_kws=dict(shade=True), plot_kws=dict(s=10))
for axis in pp.fig.axes:   # get all the axis
    axis.set_xlabel(axis.get_xlabel(), rotation=45)
_=pp.set(xticklabels=[])


# In[ ]:


# heatmap correlation of train dataset
def heatmap_correlation(df):
    _, ax=plt.subplots(figsize=(14,12))
    colormap=sns.diverging_palette(220, 10, as_cmap=True)
    _=sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink':.9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12}
    )
    plt.title('Pearson Correlation of features', y=1.05, size=15)
heatmap_correlation(train_data_full[display_cols])


# #### From data exploratory analysis we can see some fatcs such as:
# * Correlation between Pclass and Survival
# 
# Pclass (Ticket class): A proxy of socio-economy status (SES), types of Pclass:
# 1st (Upper), 2nd (Middle), 3rd (Lower). From the correlation we know that Upper ticket class has more chance to survive than other ticket class with a percentage of about 60%.
# 
# * Correlation between Sex and Survival
# 
# Female has more chance to survive than male with a percentage of about 74%. It can also be seen in Titanic Movie that women and children take precendence to aboard the lifeboats.
# 
# * Correlation between Embarked and Survival
# 
# Embarked: Port of embarkation, C = Cherbourg, Q = Queenstown, S = Southampton. From correlation analysis, people who embarked from Cherbourg have more chance to survive than others with percentage of about 55%.
# 
# * Correlation between Fare and Survival
# 
# Fare: Passenger fare. From the correlation analysis, it seems the higher the fare of passenger the higher the chance to survive than lower fare. It can be seen on correlation between FareBin and Survival. Fare with values between 39 to 512 have more chance to survive with a percentage of about 64%.
# 
# * Correlation between Age and Survival
# 
# People with age between 28 and 35 years old have more chance to survive than other with a percentage of about 43%.
# 
# * Correlation between Title and Survival
# 
# From the analysis we can see that people who have title Miss and Mrs have more chance than others with a percentage of 70% and 79% respectively. And there is no survivors for Reverend (Rev) title.
# 
# * The most correlate with Survived feature
# 
# The most correlate with Survived feature from heatmap plot is Sex_Code with correlation strength is -0.54. This relate with the following probability:
# 
# Sex -> Probability to survive
# * 0 (female) -> 0.742038
# * 1 (male) -> 0.188908

# ## Step 6. Create model data
# 
# **Model**: A machine learning model can be a mathematical representation of a real-world process. The learning algorithm finds patterns in the training data such that the input parameters correspond to the target. The output of the training process is a machine learning model which you can then use to make predictions. Machine Learning can be categorized as **Supervised learning, Unsupervised learning and Reinforced learning**. Supervised learning is where you train the model by presenting it a training dataset that includes the correct answer. Unsupervised learning is where the model is trained by training dataset that not includes the corresct answer. Reinforced learning is a hybrid of the previous two, where the model is not given the correct answer immediately, but later after a sequence of events to reinforce learning. There are many Machine Learning (ML) algorihtms, however they can be reduced to four categories: regression, classification, clustering and dimensionality reduction.
# #### Machine Learning Algorithms
# * Regression (supervised)
#     1. Linear Regression
#     2. Decision Tree Regressor
#     3. k-Nearest Neighbors (k-NN) Regressor
#     4. Random Forest Regressor
#     5. Gradient Boosting Regressor
#     6. XGBoost Regressor
#     7. Light Gradient Boosted Machine (LGBM) Regressor
#     8. CatBoost Regressor
#     9. Naive Bayes (GaussianNB)
#     10. Neural Network
# 
# 
# * Classification (supervised)
#     1. Generalized Linear Model (GLM) (Logistic Regression, Passive Aggresive, Ridge Classifier, SGD Classifier, Perceptron)
#     2. Decision Tree Classifier
#     3. Extra Tree Classifier
#     4. Support Vector Machine (SVM)
#     5. Naive Bayes (BernoulliNB, MultinomialNB)
#     6. k-NN Classifier
#     7. Gaussian Process
#     8. Discriminant Analysis
#     10. Extra Trees
#     11. Random Forest Classifier
#     12. GBM Classifier
#     13. XGBoost Classifier
#     14. AdaBoost Classifier
#     15. Bagging Classifier
#     16. LGBM Classifier
#     17. CatBoost Classifier
#     18. Neural Network Classifier
#     
#     
# * Clustering (unsupervised)
#     1. k-Means
#     2. Apriori
#     
#     
# * Dimensionality Reduction (unsupervised)
#     1. Principal Component Analysis (PCA)
#     2. Principal Component Regression (PCR)
#     3. Partial Least Squares Regression (PLSR)
# 
# 
# Survival prediction on Titanic shipwreck is classification problem, so some classification algorithms will be used such as Ensemble methods (AdaBoost, Bagging, Extra Trees, Gradient Boosting, LGBM, XGBoost, Random Forest), Gaussian proccess, GLM, Naive Bayes, k-NN, SVM, Trees, and Discriminant Analysis.

# ## Step 7. Training the initial model with cross-validation (CV)
# Training is process of learning of a model. The algorithm will find the best model during learning by minimizing the error (difference between prediction and target).

# In[ ]:


# Machine Learning Algorithm (MLA) selection and initialization
MLA=[
    # Ensemble methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    # XGBoost
    XGBClassifier(),
    
    # LightGBM
    lgb.LGBMClassifier(),
    
    # Gaussian process
    gaussian_process.GaussianProcessClassifier(),
    
    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    # Naive bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    # Nearest neighbors
    neighbors.KNeighborsClassifier(),
    
    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    # Discrimant analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
]

# split dataset in cross-validation with the splitter class
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 )

# table to compare MLA metrics
MLA_columns=['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 
            'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare=pd.DataFrame(columns=MLA_columns)

models={}

# feature cols
# cols=[col for col in baseline_feature_cols if col!='Survived']
cols=[col for col in engineered_feature_cols if col!='Survived']

# iterate through MLA and save performance to table
for idx, alg in enumerate(MLA):
    # set name and params
    MLA_name=alg.__class__.__name__
    MLA_compare.loc[idx, 'MLA Name']=MLA_name
    MLA_compare.loc[idx, 'MLA Parameters']=str(alg.get_params())
    
    # score model with cross validation
    cv_results=model_selection.cross_validate(alg, train_data_full[cols], train_data_full['Survived'], cv=cv_split, return_train_score=True)
    MLA_compare.loc[idx, 'MLA Time']=cv_results['fit_time'].mean()
    MLA_compare.loc[idx, 'MLA Train Accuracy Mean']=cv_results['train_score'].mean()
    MLA_compare.loc[idx, 'MLA Test Accuracy Mean']=cv_results['test_score'].mean()
    MLA_compare.loc[idx, 'MLA Test Accuracy 3*STD']=cv_results['test_score'].std()*3
    
    # save MLA predictions score and model
    models[MLA_name]=[cv_results['test_score'].mean(), alg]

# show and sort table
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare
    


# In[ ]:


# barplot for MLA comparison
ax=sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='#52518f')
test_acc=MLA_compare['MLA Test Accuracy Mean'].apply(lambda x: f'{x*100:.2f}')
target_name=MLA_compare['MLA Name']+' ('+ test_acc +'%)'
ax.set_yticklabels(target_name)

# pretify using pyplot
plt.title('Machine Learning Algorithm Test Accuracy Score\n')
plt.xlabel('Accuracy Score (%)')
_=plt.ylabel('Algorithm')


# ## Step 8. Evaluate the model performace
# The top 5 models for this features are **SVC** (Support Vector Classifier) (83.69%), **NuSVC** (83.63%), **RidgeClassifierCV** (83.46%), **LinearSVC** (83.41%), and **LinearDiscrimantAnalysis** (83.35%). The best test accuracy score for those models is SVC (Support Vector Classifier), so we will use it for predict survival on test data. And the difference between train and test accuracy mean is almost the same, so the models are well generalizing for unseen data (not **underfitting** nor **overfitting**).
# 
# <img src="https://www.dropbox.com/s/jsfv6xemog1qeys/Data%20Science%20Role.png?raw=1" width="400">

# ## Step 9. Tune model with hyper-parameters
# Tune hyper-parameters is imprtant to boost model performance. We will tune top 5 models models for predictions (SVC, NuSVC, RidgeClassifierCV, LinearSVC, LinearDiscriminantAnalysis). 

# In[ ]:


# hyper-parameters tune with GridSearchCV
grid_ratio=[.1, .25, .5, .75, 1.0]
best_models={}

estimators=[
    ('svc', svm.SVC()),
    ('nusvc', svm.NuSVC()),
    ('rc', linear_model.RidgeClassifierCV()),
    ('lsvc', svm.LinearSVC()),
    ('lda', discriminant_analysis.LinearDiscriminantAnalysis())
]

grid_param=[
    [{
        # SVC
        'C': [1,2,3,4,5],
        'gamma': grid_ratio,
        'decision_function_shape': ['ovo', 'ovr'],
        'probability': [True],
        'random_state': [0]
    }],
    [{
        # NuSVC
        'nu': [0.5, 0.7],
        'gamma': grid_ratio,
        'decision_function_shape': ['ovo', 'ovr'],
        'probability': [True, False],
        'random_state': [0]
        
    }],
    [{
        # RidgeClassifierCV
        'alphas':[(0.1, 0.5, 7.0), (0.1, 0.7, 10.0), (0.1, 1.0, 10.0)],
        'normalize':[True, False],
        'scoring':[None],
        'class_weight': ['balanced', None]
    }],
    [{
        # LinearSVC
        'penalty':['l2'],
        'loss':['hinge', 'squared_hinge'],
        'C': [1,2,3,4,5]
    }],
    [{
        # LinearDiscriminantAnalysis
        'solver':['svd', 'lsqr'],
        'shrinkage':[None]
    }]
]

start_total=time.perf_counter()
for clf, param in zip(estimators, grid_param):
    start=time.perf_counter()
    best_search=model_selection.GridSearchCV(estimator=clf[1], param_grid=param, cv=cv_split, scoring='roc_auc', n_jobs=-1)
    best_search.fit(train_data_full[cols], train_data_full['Survived'])
    run=time.perf_counter()-start
    
    best_param=best_search.best_params_
    best_score=best_search.best_score_
    best_models[clf[1].__class__.__name__]=[best_score, best_search, run]
    print('Name: ', clf[1].__class__.__name__)
    print('Best score: ', best_score)
    print('best param: ', best_param)
    print('runtime: ', run)
    print('-'*10)
    clf[1].set_params(**best_param)

run_total=time.perf_counter()-start_total
print('Total optimization time: {:.2f} minutes'.format(run_total/60))
print('Finish')


# ## 10. Tune model with feature selection
# Top 5 model will try to predict outcome with different features selection. There are two collection of features, baseline_feature_cols and engineered_feature_cols.

# In[ ]:


# features selection
for idx, features in enumerate([baseline_feature_cols, engineered_feature_cols]):
    print('='*35)
    print('Features type: ', ['Baseline Features', 'Engineered Features'][idx])
    print('='*35)
    for model in best_models.items():
        model[1][1].best_estimator_.fit(train_[features], train_['Survived'])
        y_pred=model[1][1].best_estimator_.predict(valid_[features])
        score=metrics.roc_auc_score(valid_['Survived'], y_pred)
        print('Model: ', model[0])
        print(f'Validation score: {score:.4f}')
        print('-'*25)


# ## 11. Validate and implement
# After selecting the best model and features for dataset and validating it with validation data. It's time to implement the best model to predict outcome (to predict survival in this case) for test data. From the model and features selection, the bset model for this case is SVC (Support Vector Classifier) which can predict the survivor with the ROC_AUC score of about 0.88 in engineered feature.

# In[ ]:


# generate CSV file for submitting survival predictions
model=best_models['SVC'][1]
model.best_estimator_.fit(train_data_full[engineered_feature_cols], train_data_full['Survived'])
y_pred=model.best_estimator_.predict(test_data_full[engineered_feature_cols])

submit=pd.DataFrame({'PassengerId':test_data_full.PassengerId,
                    'Survived':y_pred})
submit.to_csv('submission.csv', index=False)


# ## 12. Optimize and strategy
# When the model predict full test data and submitted to Kaggle, it yields [0.7846] public score. It's far from the validation score and denote there are some reasons, such as inaccurate model or features that are not quite right. And for this problem seems most likely on features selection. For the next strategy, to boost the prediction accuracy we will try to re-engineered features more thoroughly such as generating new features from Name, Title, Fare, and Ticket and then trying fature normalization. And then selecting the best model for generated new features.
# 
# <img src="https://www.dropbox.com/s/gm1687wp1x413ua/kaggle%20submit.png?raw=1" width="650">

# ## Conclusion
# For this practice, some conclusions can be drawn such as: 
# 
# * Feature engineering (like generating new features, categorical encoding, and feature selection) is very important to gain the prediction accuracy.
# * Every model has specific hyper-parameters that need to be tuned to make better prediction.
# * Every case in dataset such as number of features, type of fatures (integer, float or binary(0/1)) has an effect of selecting the best model (must choose ensemble method, k-NN or SVC to fit to dataset better)
# 
# Thanks.
