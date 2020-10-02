#!/usr/bin/env python
# coding: utf-8

# # Data visualization: Simple, Single unified API for plotting and charting

# During EDA/data preparation stage, I use few fixed chart types to analyse the relation among various features. 
# Few are simple chart like univariate and some are complex 3D or even multiple features>3.
# 
# Over the period it became complex to maintain all relevant codes or repeat codes. 
# Instead I developed a simple, single api to plot various type of relations which will hide all technical/code details from Data Science task and approch.
# 
# Using this approach I just need one api 
# 
# from KUtils.eda import chartil
# 
# - chartil.plot(dataframe, [list of columns]) or
# - chartil.plot(dataframe, [list of columns], {optional_settings})
# 

# ## Chart + Util = Chartil 
# Package available at <a href="https://pypi.org/project/kesh-utils/">Kesh Utils</a>
# 
# Source available at <a href="https://github.com/KeshavShetty/ds/tree/master/KUtils">Github KUtils.eda.chartil</a>
# 
# This custom utility package contains single api/function to plot various charts
# 
#     
#     Importing
#     | import KUtils.chartil as chartil
#     
#     Entry api/function (Usage)
#     | chartilc.plot(dataframe, [list of column names])
#     
#     Other available functions
#     | uni_category_barchart(df, column_name, limit_bars_count_to=10000, sort_by_value=False)
#     | uni_continuous_boxplot(df, column_name)
#     | uni_continuous_distplot(df, column_name)
#     | 
#     | bi_continuous_continuous_scatterplot(df, column_name1, column_name2, chart_type=None)
#     | bi_continuous_category_boxplot(df, continuous1, category2)
#     | bi_continuous_category_distplot(df, continuous1, category2)
#     | bi_category_category_crosstab_percentage(df, category_column1, category_column2)
#     | bi_category_category_stacked_barchart(df, category_column1, category_column2)
#     | bi_category_category_countplot(df, category_column1, category_column2)
#     | bi_continuous_category_violinplot(df, category1, continuous2)
#     | 
#     | multi_continuous_category_category_violinplot(df, continuous1, category_column2, category_column3)
#     | multi_continuous_continuous_category_scatterplot(df, column_name1, column_name2, column_name3)
#     | multi_continuous_category_category_boxplot(df, continuous1, category2, category3)
#     | multi_continuous_continuous_continuous_category_scatterplot(df, continuous1, continuous2, continuous3, category4)
#     | multi_continuous_continuous_continuous_scatterplot(df, continuous1, continuous2, continuous3, maintain_same_color_palette=False)
# 
# 
# 

# # Lets start the demo with one of the popular dataset available UCI Heart Disease 
# You can get this dataset here
# <a href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease/">Heart Disease UCI</a>

# In[ ]:


# Bare minimum required imports
import numpy as np
import pandas as pd


# In[ ]:


get_ipython().system('pip install kesh-utils')


# In[ ]:


# Import the chartil from Kutils 
from KUtils.eda import chartil


# In[ ]:


# Load the dataset
heart_disease_df = pd.read_csv('../input/heart.csv')


# In[ ]:


heart_disease_df.head(10)
heart_disease_df.info()
heart_disease_df.describe()
heart_disease_df.shape


# In[ ]:


# Null checks
heart_disease_df.isnull().sum() # No null found


# In[ ]:


# Number of unique values in each column 
{x: len(heart_disease_df[x].unique()) for x in heart_disease_df.columns}


# In[ ]:


# Quick data preparation, convert few to categorical column and add new age_bin
heart_disease_df['target'].describe()
heart_disease_df['target'] = heart_disease_df['target'].astype('category')
heart_disease_df['age'].describe()
heart_disease_df['age_bin'] = pd.cut(heart_disease_df['age'], [0, 32, 40, 50, 60, 70, 100], 
                labels=['<32', '33-40','41-50','51-60','61-70', '71+'])

heart_disease_df['sex'].describe()
heart_disease_df['sex'] = heart_disease_df['sex'].map({1:'Male', 0:'Female'})

heart_disease_df['cp'].describe()
heart_disease_df['cp'] = heart_disease_df['cp'].astype('category')

heart_disease_df['trestbps'].describe()
heart_disease_df['chol'].describe()

heart_disease_df['fbs'] = heart_disease_df['fbs'].astype('category')
heart_disease_df['restecg'] = heart_disease_df['restecg'].astype('category')

heart_disease_df['thalach'].describe()

heart_disease_df['exang'] = heart_disease_df['exang'].astype('category')
heart_disease_df['oldpeak'].describe()

heart_disease_df['slope'] = heart_disease_df['slope'].astype('category')
heart_disease_df['ca'] = heart_disease_df['ca'].astype('category')
heart_disease_df['thal'] = heart_disease_df['thal'].astype('category')

heart_disease_df.info()


# In[ ]:


import warnings  
warnings.filterwarnings('ignore')


# # Lest start with chartil.plot(..)

# In[ ]:


# Univariate Categorical variable
chartil.plot(heart_disease_df, ['target'])


# In[ ]:


# Univariate Numeric/Continuous variable
chartil.plot(heart_disease_df, ['trestbps'])


# In[ ]:


# Smae as above, but force to use barchart on numeric/Continuous (Automatically creates 10 equal bins)
chartil.plot(heart_disease_df, ['age'], chart_type='barchart')


# In[ ]:


# Age doesn't look normal with auto bin barchart, instead use age_bin column to plot the same
chartil.plot(heart_disease_df, ['age_bin'])


# In[ ]:


chartil.plot(heart_disease_df, ['age_bin'], 
             optional_settings={'sort_by_value':True})


# In[ ]:


chartil.plot(heart_disease_df, ['age_bin'], 
             optional_settings={'sort_by_value':True, 'limit_bars_count_to':5})


# In[ ]:


chartil.plot(heart_disease_df, ['trestbps'], chart_type='distplot')


# In[ ]:


# Bi Category vs Category (+ Univariate Segmented)
chartil.plot(heart_disease_df, ['sex', 'target'])


# In[ ]:


chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='crosstab')


# In[ ]:


chartil.plot(heart_disease_df, ['sex', 'target'], chart_type='stacked_barchart')


# In[ ]:


# Bi Continuous vs Continuous (Scatter plot)
chartil.plot(heart_disease_df, ['chol', 'thalach'])


# In[ ]:


# Bi Continuous vs Category
chartil.plot(heart_disease_df, ['thalach', 'sex'])


# In[ ]:


# Same as above, but use distplot
chartil.plot(heart_disease_df, ['thalach', 'sex'], chart_type='distplot')


# In[ ]:


# Multi variavte - 3D view of 3 Continuous variables coloured by the same contious varibale amplitude in RGB form
chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps'])


# In[ ]:


# Multi 2 Continuous, 1 Category
chartil.plot(heart_disease_df, ['chol', 'thalach', 'target'])


# In[ ]:


# Multi 1 Continuous, 2 Category
chartil.plot(heart_disease_df, ['thalach', 'cp', 'target'])


# In[ ]:


# Same as above, but use violin plot
chartil.plot(heart_disease_df, ['thalach', 'sex', 'target'], chart_type='violinplot')


# In[ ]:


# Multi 3D view of 3 Continuous variable and color it by target/categorical feature
chartil.plot(heart_disease_df, ['chol', 'thalach', 'trestbps', 'target'])


# In[ ]:


# Heatmap (Send list of all columns, it will plot the co-relation matrix of all numerical/continuous variables)
chartil.plot(heart_disease_df, heart_disease_df.columns)


# In[ ]:


# If you want sort the corelation based on one specific columns.
chartil.plot(heart_disease_df, heart_disease_df.columns, optional_settings={'sort_by_column':'thalach'})


# In[ ]:


# Include categorical variables - Internally creates dummies
chartil.plot(heart_disease_df, heart_disease_df.columns, optional_settings={'include_categorical':True} )


# ### <font color=red>Use Co-relation plotting carefully when you have large dataset with lot of features.</font>

# In[ ]:


# If you want sort the corelation based on one specific columns. 
# Below example will sort the feature co-relation by feature 'trestbps'
chartil.plot(heart_disease_df, heart_disease_df.columns, 
             optional_settings={'include_categorical':True, 'sort_by_column':'trestbps'} )


# ## Will add new possible combinations whenever I need one
