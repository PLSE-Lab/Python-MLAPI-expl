#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:


train_data.head()


# 1. ID_code is unique Identifier
# 2. Target is binary
# 3. Variables from 0 to 199

# In[ ]:


test_data.head()


# In[ ]:


#plt.figure(figsize=(100, 100))
train_crr=train_data.copy()
train_crr.drop(['ID_code', 'target'],axis=1, inplace=True)
corr = train_crr.apply(lambda x: pd.factorize(x)[0]).corr()
#ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap="YlGnBu")
print(corr)
corr.to_csv("corr.csv")


# In[ ]:


#print("Correlation Matrix")
#print(train_crr.corr())
#print()

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(train_crr, 20))
#all_crr=get_top_abs_correlations(train_crr, 200)
#all_crr.to_csv("all_crr.csv")


# Lets take the raw data and run a random forest model to check feature importance. This will give us an order in which we can explore the variables. We will take a sample of 5000 rows and remove ID column

# In[ ]:


train_data.shape
#(200000, 202)


sample_data=train_data.sample(5000)
sample_data=sample_data.drop(["ID_code"], axis=1)

sample_x = sample_data.loc[:, ~sample_data.columns.isin(['target'])]
sample_y = sample_data.loc[:,'target']


# In[ ]:


sample_x.head()


# In[ ]:


sample_y.head()


# In[ ]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(sample_x, sample_y, test_size=0.3) # 70% training and 30% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model F1 Score
print("F Score:" , metrics.fbeta_score(y_test, y_pred, beta=0.5))


# In[ ]:


feature_imp = pd.Series(clf.feature_importances_,index=sample_x.columns).sort_values(ascending=False)
feature_imp


# In[ ]:


df=train_data.copy()


# As per above results, we will explore variable 147 first

# In[ ]:


def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    ax0 = sns.kdeplot(df[df['target'] == 0][feature].dropna(), color= 'navy', label= 'target: 0')
    ax1 = sns.kdeplot(df[df['target'] == 1][feature].dropna(), color= 'orange', label= 'target: 1')


# In[ ]:


kdeplot('var_147')


# There is not a significant differnece between target values for var_147

# Taking helper function from https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf

# In[ ]:


def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data

    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot

    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    #sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    #plt.show()


# In[ ]:


# Target Variable: Survival
c_palette = ['tab:blue', 'tab:orange']
categorical_summarized(df, y = 'var_147', palette=c_palette)


# * Huge difference between minimum and maximum
# * large proportion of values are negative

# In[ ]:


def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data

    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed

    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    plt.show()


# In[ ]:


# univariate analysis
c_palette = ['tab:blue', 'tab:orange']
quantitative_summarized(dataframe= df, y = 'var_147', palette=c_palette, verbose=False, swarm=False)


# ************************ NEW CODE ****************************************

# In[ ]:


df = train_data.copy()


# In[ ]:


df = df.drop(["ID_code"],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.iloc[:,0:10].hist(figsize=(15, 15), bins=40, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations


# In[ ]:


df_new_1 = df[df.target==1]
#Multiply by some constant except the target variable
df_new_1.iloc[:,0:10].hist(figsize=(15, 15), bins=40, xlabelsize=8, ylabelsize=8);


# In[ ]:


df_new_0 = df[df.target==0]
#Multiply by some constant except the target variable
df_new_0.iloc[:,0:10].hist(figsize=(15, 15), bins=40, xlabelsize=8, ylabelsize=8);


# In[ ]:



df_corr = df.corr()['target'][:-1] # -1 because the latest row is target
golden_features_list = df_corr[abs(df_corr) > 0.1].sort_values(ascending=False)
print("There is {} strongly correlated values with target:\n{}".format(len(golden_features_list), golden_features_list))


# In[ ]:


for i in range(0, 5 , 5):
    sns.pairplot(data=df,
                x_vars=df.columns[i:i+5],
                y_vars=['target'])


# In[ ]:


for i in range(5, 14 , 5):
    sns.pairplot(data=df,
                x_vars=df.columns[i:i+5],
                y_vars=['target'])


# In[ ]:


plt.figure(figsize = (5, 5))
ax = sns.boxplot(y='var_0', x='target', data=df)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)


# In[ ]:


plt.close()
plt.figure(figsize = (5, 5))
ax = sns.boxplot(y='var_10', x='target', data=df)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")


# In[ ]:


# Detect outliers from IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


print((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))


# In[ ]:


print("df.shape:",df.shape)
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("df_out.shape:",df_out.shape)


# In[ ]:


tdata = train_data.copy()

tdata = tdata.loc[tdata.index & df_out.index]
tdata = tdata.loc[np.intersect1d(tdata.index, df_out.index)]
tdata = tdata.loc[tdata.index.intersection(df_out.index)]

print("tdata.shape:",tdata.shape)

tdata.head()

tdata.to_csv("train_data_new.csv")

