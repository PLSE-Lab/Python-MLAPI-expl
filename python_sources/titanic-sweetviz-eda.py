#!/usr/bin/env python
# coding: utf-8

# Here's an example of how to use [sweetviz](https://github.com/fbdesignpro/sweetviz). The result will be output to Output as an html file.
# 
# 
# >Sweetviz is an open source Python library that generates beautiful, high-density visualizations to kickstart EDA (Exploratory Data Analysis) with a single line of code. Output is a fully self-contained HTML application.
# 
# >The system is built around quickly visualizing target values and comparing datasets. Its goal is to help quick analysis of target characteristics, training vs testing data, and other such data characterization tasks.

# In[ ]:


get_ipython().system('pip install sweetviz')


# In[ ]:


import pandas as pd
import sweetviz as sv
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


train.head()


# # Analyzing a single dataframe (and its optional target feature)

# In[ ]:


train_report = sv.analyze(train, target_feat='Survived')
train_report.show_html('train_report.html')


# In[ ]:


feature_config = sv.FeatureConfig(skip="PassengerId", force_text=["Age"])


# In[ ]:


train_report_v2 = sv.analyze(train, target_feat='Survived', feat_cfg=feature_config)
train_report_v2.show_html('train_report_v2.html')


# # Comparing two dataframes (e.g. Test vs Training sets)

# In[ ]:


my_report =     sv.compare([train, "Training Data"],
               [test, "Test Data"],
               target_feat="Survived")


# In[ ]:


my_report.show_html('train_test_compare.html')


# # Comparing two subsets of the same dataframe (e.g. Male vs Female)

# In[ ]:


my_report =     sv.compare_intra(train,
                     train["Sex"] == "male",
                     ["Male", "Female"])


# In[ ]:


my_report.show_html('male_vs_female.html')

