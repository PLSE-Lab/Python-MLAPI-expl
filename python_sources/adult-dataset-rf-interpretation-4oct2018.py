#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


adult_df = pd.read_csv('../input/adult.csv')


# In[ ]:


adult_df.head()


# In[ ]:


adult_df.info()


# # Observations:
# 1. No Null values
# 2. quite a few object columns

# In[ ]:


#lets go through each of the columns and decide what action to be taken


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.boxplot(adult_df['age'])
plt.show()


# In[ ]:


plt.hist(adult_df['age'])
plt.show()


# In[ ]:


#i dont see any problems with this col; we can leave it as is


# In[ ]:


obj_cols_to_be_treated = []


# In[ ]:


adult_df['workclass'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('workclass')


# In[ ]:


adult_df.columns


# In[ ]:


adult_df['education'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('education')


# In[ ]:


adult_df['educational-num'].value_counts()


# In[ ]:


#it is clear that the educational-num is an encoded col of education; so we will jus use this and not encode education again
obj_cols_to_be_treated.remove('education')


# In[ ]:


adult_df['marital-status'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('marital-status')


# In[ ]:


adult_df['occupation'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('occupation')


# In[ ]:


#'relationship', 'race', 'gender',


# In[ ]:


adult_df['relationship'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('relationship')


# In[ ]:


adult_df['race'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('race')


# In[ ]:


adult_df['gender'].value_counts()


# In[ ]:


obj_cols_to_be_treated.append('gender')


# In[ ]:


#all but one of the obj cols are reviewed


# In[ ]:


plt.boxplot(adult_df['capital-gain'])
plt.show()


# In[ ]:


plt.hist(adult_df['capital-gain'])
plt.show()


# In[ ]:


#this is a skewed column; but no action for now; 
#lets jus move on to prediction after reviewing the output col


# In[ ]:


adult_df['income'].describe()


# In[ ]:


adult_df['income'].value_counts()


# In[ ]:


adult_df['income_less_than_50K_1_0'] = adult_df['income'].map({'<=50K':1, '>50K':0})


# In[ ]:


adult_df['income_less_than_50K_1_0'].value_counts()


# In[ ]:


#OK, lets numericalize the obj cols


# In[ ]:


obj_cols_to_be_treated


# In[ ]:


for obj_col in obj_cols_to_be_treated:
    adult_df[obj_col + '_cat'] = adult_df[obj_col].astype('category').cat.as_ordered()
    adult_df[obj_col + '_cat_codes'] = adult_df[obj_col + '_cat'].cat.codes


# In[ ]:


encoded_cols = [col for col in adult_df.columns if '_cat_codes' in col]


# In[ ]:


encoded_cols


# In[ ]:


#add the encoded cols and original numeric cols from df to create the input vars list
input_vars = encoded_cols + ['age','educational-num','capital-gain','capital-loss','hours-per-week']


# In[ ]:


output_var = 'income_less_than_50K_1_0'


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#lets start with a single decision tree with max depth of 3
rf1 = RandomForestClassifier(n_estimators=1, max_depth=3)


# In[ ]:


rf1.fit(X=adult_df[input_vars], y=adult_df[output_var])


# In[ ]:


rf1.score(X=adult_df[input_vars], y=adult_df[output_var])


# In[ ]:


from treeinterpreter import treeinterpreter as ti


# In[ ]:


adult_df.head()


# In[ ]:


sample_record = adult_df.loc[0:1]


# In[ ]:


sample_record


# In[ ]:


prediction, bias, contribution = ti.predict(X=sample_record[input_vars], model=rf1)


# In[ ]:


prediction.shape


# In[ ]:


prediction


# In[ ]:


bias.shape


# In[ ]:


bias


# In[ ]:


adult_df['income_less_than_50K_1_0'].value_counts()


# In[ ]:


adult_df['income_less_than_50K_1_0'].value_counts()[0] / adult_df.shape[0]


# In[ ]:


#we can see that the bias value doesnt change across instances; it is the proportion of that output variable


# In[ ]:


contribution.shape


# In[ ]:


contribution[0].shape


# In[ ]:


contribution[0]


# In[ ]:


input_vars


# In[ ]:


#get more trees
rf2 = RandomForestClassifier(n_estimators=10, max_depth=3)


# In[ ]:


rf2.fit(X=adult_df[input_vars], y=adult_df[output_var])
rf2.score(X=adult_df[input_vars], y=adult_df[output_var])


# In[ ]:


rf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
rf3.fit(X=adult_df[input_vars], y=adult_df[output_var])
rf3.score(X=adult_df[input_vars], y=adult_df[output_var])


# In[ ]:


feat_imp_dict ={}
for i in range(len(input_vars)):
    feat_imp_dict[input_vars[i]] = rf3.feature_importances_[i]


# In[ ]:


feat_imp_df = pd.DataFrame.from_dict(feat_imp_dict, orient= 'index')


# In[ ]:


feat_imp_df.reset_index(inplace=True)
feat_imp_df.columns = ['feature', 'feat_imp']


# In[ ]:


feat_imp_df.sort_values(ascending=False, by=['feat_imp'], inplace=True)


# In[ ]:


feat_imp_df.head()


# # Let's use tree interpreter

# In[ ]:


from treeinterpreter import treeinterpreter as ti


# In[ ]:


prediction, bias, contribution = ti.predict(X=sample_record[input_vars], model=rf3)


# In[ ]:


prediction


# In[ ]:


contribution[0]


# In[ ]:


#lets look at a few cases where output var is 0


# In[ ]:


adult_df[adult_df['income_less_than_50K_1_0'] ==0]


# In[ ]:


adult_df[adult_df['income_less_than_50K_1_0'] ==0].loc[2:3]


# In[ ]:


sample_record_0s = adult_df[adult_df['income_less_than_50K_1_0'] ==0].loc[2:3]


# In[ ]:


prediction, bias, contribution = ti.predict(X=sample_record_0s[input_vars], model=rf3)


# In[ ]:


prediction


# In[ ]:


bias


# In[ ]:


contribution[1]


# In[ ]:


#the third item from last is the one which impacts the most
input_vars[-3]


# In[ ]:


sample_record_0s


# In[ ]:


from sklearn.tree import export_graphviz


# In[ ]:


s=export_graphviz(rf1.estimators_[0], out_file=None, feature_names=input_vars, filled=True,
                      special_characters=True, rotate=True, precision=3)


# In[ ]:


from IPython.display import display


# In[ ]:


import IPython


# In[ ]:


from sklearn.tree import export_graphviz


# In[ ]:


import graphviz


# In[ ]:


IPython.display.display(graphviz.Source(s))


# In[ ]:




