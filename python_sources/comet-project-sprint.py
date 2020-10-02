#!/usr/bin/env python
# coding: utf-8

# McDonald's is the world's leading global food service retailer with over 36,000 locations in over 100 countries. Which is loved by all people. But people get conscious of what they eat nowadays as health is one of the society's top priority now. But we would want to keep track how much nutrition facts we have in each meal we eat.
# 
# 1. Do Higher calories result to higher fat?
# 2. Do McDonalds food have high Protein value?
# 3. Does high sodium increase high cholesterol level?
# 
# The findings will be relevant to people who are consuming McDonalds and to people who are health conscious to what they eat.
# 
# Findings:
# As we can see from the data sets below that the food provided by McDonalds that are high in calories automatically result to higher fats. That means that all foods provided by McDonalds may give you lots of fats which might not be healthy to the body.
# 
# But we would also want to know if they provide high protein value. We can see that the lower part of the protein chart that is ranging 0 - 20 has the highest frequency count while the ones at the higher end are sometimes just 1 to 2. This would suggest that most of the food that McDonalds provide are not healthy food and they are just a means of consumptions but not towards healthy living.
# 
# Finally, the sodium and cholesterol base on the findings does not have a steady flow on the chart. This would suggest that not all the food in McDonalds that have a high sodium rate would literally lead to high cholesterol consumption as well. They are independent from one another and they should not be intervened together when picking out the right food to eat.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_set = pd.read_csv("../input/menu.csv")
data_set


# In[ ]:


data_set.plot.scatter(x = 'Calories', y = 'Total Fat')


# In[ ]:


bs = {'Calories': data_set['Calories'], 'Total Fat': data_set['Total Fat']}
bs


# In[ ]:


Calories_TotalFat_data_frame = pd.DataFrame(bs)


# In[ ]:


Calories_TotalFat_data_frame.corr('pearson')


# In[ ]:


Protein = {'Protein': data_set['Protein']}


# In[ ]:


protein_data_frame = pd.DataFrame(Protein)
protein_data_frame


# In[ ]:


y = protein_data_frame['Protein'].value_counts()
y


# In[ ]:


data_set.plot.scatter(x = 'Sodium', y = 'Cholesterol')


# In[ ]:


sc = {'Sodium': data_set['Sodium'], 'Cholesterol': data_set['Cholesterol']}
sc


# In[ ]:


sodium_cholesterol_data_frame = pd.DataFrame(sc)


# In[ ]:


sodium_cholesterol_data_frame.corr('pearson')

