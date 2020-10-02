#!/usr/bin/env python
# coding: utf-8

# EDA of Happiness

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tools.plotting import scatter_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#read files, look at columns attributes, generally get a feel for the dataset
happy = pd.read_csv('../input/2015.csv')
happy_16 = pd.read_csv('../input/2016.csv')

happy.head()


# In[ ]:


happy.info()
#No null values, Region is only categorical str column, will use Country col 'identifier'
#Store country name in a different df and use the Happiness Rank as the index
country_key = happy[['Country','Happiness Rank']].copy()
happy.drop(['Country'],axis=1,inplace=True)


# In[ ]:


happy.describe()


# In[ ]:


happy.hist(figsize=(15,10));


# Because the dataset is smaller I will be doing a stratified split.

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

happy['happy_cat'] = np.ceil(happy['Happiness Score'])
happy['happy_cat'].where(happy['Happiness Score'] > 4, 4, inplace=True)
happy['happy_cat'].value_counts()

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(happy,happy['happy_cat']):
    strat_train_set = happy.loc[train_index]
    strat_test_set = happy.loc[test_index]

for set in (strat_train_set,strat_test_set):
    set.drop(['happy_cat'],axis=1,inplace=True)


# In[ ]:


#looking at correlation between variables and Happiness Score
happy = strat_train_set.copy()
corr_matrix = happy.corr()
corr_matrix['Happiness Score'].sort_values(ascending=False)


# In[ ]:


#Scatter matrix of most interesting correlated variables
attributes = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']
scatter_matrix(happy[attributes],figsize=(12,8));


# In[ ]:


#Scatter matrix of some other correlated variables
attributes = ['Happiness Score','Dystopia Residual','Freedom','Trust (Government Corruption)']
scatter_matrix(happy[attributes],figsize=(12,8));


# In[ ]:


#Pretty solid linear relationship with happiness and GDP
happy.plot(kind='scatter',x='Happiness Score',y='Economy (GDP per Capita)',alpha=0.6);


# In[ ]:


happy.groupby('Region').describe()


# In[ ]:


happy = strat_train_set.drop("Happiness Score",axis = 1)
happy_labels = strat_train_set['Happiness Score'].copy()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

happy_num = happy.drop(["Region","Standard Error"], axis=1) 
num_attr = list(happy_num)
cat_attr = ["Region"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attr)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attr)),
        ('label_binarizer',LabelBinarizer()),
    ])

tot_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline),
    ])

happy_live = num_pipeline.fit_transform(happy)
#cat_pipeline.fit_transform(cat_attr)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(happy_live, happy_labels)
happy_predictions = lin_reg.predict(happy_live)
lin_mse = mean_squared_error(happy_labels,happy_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(happy_live,happy_labels)

happy_predictions = tree_reg.predict(happy_live)
tree_mse = mean_squared_error(happy_labels,happy_predictions)
tree_rmse = np.sqrt(tree_mse)

scores = cross_val_score(tree_reg, happy_live, happy_labels,
                        scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)


# In[ ]:


print(rmse_scores)
print(rmse_scores.mean())
print(rmse_scores.std())


# In[ ]:




