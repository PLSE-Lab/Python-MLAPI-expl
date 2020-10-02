#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


# In[ ]:


cereal = pd.read_csv('../input/80-cereals/cereal.csv')
cereal.head()


# In[ ]:


cereal['type'].unique()


# knowing the distribution of sodium value with cold / hot type

# In[ ]:


#graph distribution of quantitative data
plt.figure(figsize=[8,6])
plt.hist(x = [cereal[cereal['type']=='C']['sodium'], cereal[cereal['type']=='H']['sodium']], 
         stacked=True, color = ['b','y'],label = ['Cold','Hot'])
plt.title('Sodium Histogram by type of cereal cold/hot')
plt.xlabel('Cold / Hot')
plt.ylabel('sodium')
plt.legend()


# knowing the significance of sodium with hot or cold type

# In[ ]:


hot = cereal[cereal['type']=='H']['sodium']
cold = cereal[cereal['type']=='C']['sodium']
#perform t-test
ttest_ind(hot, cold, equal_var = False)


# because the pvalue has a value less than 0.05 (critical p-value). So, in this case we reject the null hyphothesis, it means that it has weak correlation each other.

# In[ ]:


x = cereal['sodium']
y = cereal['rating']
#perform t-test
ttest_ind(x, y, equal_var = False)


# because the pvalue has a value less than 0.05 (critical p-value). So, in this case we reject the null hyphothesis, it means that it has weak correlation each other.

# In[ ]:


#graph distribution of quantitative data
plt.figure(figsize=[8,6])
plt.hist(x = [cereal[cereal['type']=='H']['rating'], cereal[cereal['type']=='C']['rating']], 
         stacked=True, color = ['r','b'],label = ['Hot','Cold'])
plt.title('Rating Histogram by type of cereal cold/hot')
plt.xlabel('Cold / Hot')
plt.ylabel('Rating')
plt.legend()


# In[ ]:


hot = cereal[cereal['type']=='H']['rating']
cold = cereal[cereal['type']=='C']['rating']
#perform t-test
ttest_ind(hot, cold, equal_var = False)


# because the pvalue has a value more than 0.05 (critical p-value). So, in this case we accept the null hyphothesis, it means that it has strong correlation between rating of the hot cereal and rating of the cold cereal

# In[ ]:


#graph distribution of quantitative data
plt.figure(figsize=[8,6])
plt.hist(x = [cereal[cereal['type']=='H']['carbo'], cereal[cereal['type']=='C']['carbo']], 
         stacked=True, color = ['r','b'],label = ['Hot','Cold'])
plt.title('Carbo Histogram by type of cereal cold/hot')
plt.xlabel('Cold / Hot')
plt.ylabel('Carbo')
plt.legend()


# In[ ]:


hot = cereal[cereal['type']=='H']['carbo']
cold = cereal[cereal['type']=='C']['carbo']
#perform t-test
ttest_ind(hot, cold, equal_var = False)


# because the pvalue has a value more than 0.05 (critical p-value). So, in this case we accept the null hyphothesis, it means that it has strong correlation between the carbo of the hot cereal and the carbo of the cold cereal

# In[ ]:


x = cereal['carbo']
y = cereal['rating']
#perform t-test
ttest_ind(x, y, equal_var = False)


# because the pvalue has a value less than 0.05 (critical p-value). So, in this case we reject the null hyphothesis, it means that it has weak correlation between carbo and rating

# plotting the protein value based on the type of cereal

# In[ ]:


import seaborn as sns 
import matplotlib.pyplot as plt
plt.subplots(figsize=(10,8))
sns.countplot(x="protein",data=cereal,hue = "type").set_title("Protein plot by type of cereal cold/hot")


# In[ ]:


cereal['mfr'].unique()


# plotting the value of MFR

# In[ ]:


import seaborn as sns 
import matplotlib.pyplot as plt
plt.subplots(figsize=(8,6))
sns.countplot(x="mfr",data=cereal).set_title("MFR plot")


# In[ ]:


#graph distribution of quantitative data
plt.figure(figsize=[8,6])
plt.hist(x = [cereal[cereal['mfr']=='N']['rating'], cereal[cereal['mfr']=='Q']['rating'],cereal[cereal['mfr']=='K']['rating'],
             cereal[cereal['mfr']=='R']['rating'],cereal[cereal['mfr']=='G']['rating'],cereal[cereal['mfr']=='P']['rating'],
             cereal[cereal['mfr']=='A']['rating']], stacked=True,label = ['N','Q','K','R','G','P','A'])
plt.title('Rating Histogram by type of MFR')
plt.xlabel('type of MFR')
plt.ylabel('Rating')
plt.legend()


# In[ ]:




