#!/usr/bin/env python
# coding: utf-8

# ### Preparation  

# In[ ]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import train and test datasets
train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')


# In[ ]:


# Check the first five rows of train
train.head()


# In[ ]:


# Dimension of train
print('Train data number of rows and columns: ', train.shape)


# In[ ]:


# Replace Yes and No values with 1 and 0 
train = train.replace(['no', 'yes'], [0, 1])
#test = test.replace(['no', 'yes'], [0, 1])


# ### Distribution of target variable for total sample and household heads

# In[ ]:


# Target labels
dlab = {1:'extreme poverty', 2:'moderate poverty', 3:'vulnerable households', 4:'non vulnerable households'}

# Whole observation targets
target_tot = train.Target.value_counts().to_frame()
idx = pd.Series(target_tot.index)
target_tot.index = idx.apply(lambda x:dlab[x])


# In[ ]:


# Resample for household head
train_hh = train.loc[train.parentesco1 == 1]
target_hh = train_hh.Target.value_counts().to_frame()
idx = pd.Series(target_hh.index)
target_hh.index = idx.apply(lambda x:dlab[x])


# In[ ]:


# Two plots together
explode = (0, .05, .1, .15)

plt.subplot(1, 2, 1)
plt.pie(target_tot, explode = explode, autopct = '%1.1f%%', shadow = True, startangle = 90, center = (-3, 0))
plt.title('Target total')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.pie(target_hh, explode = explode, autopct = '%1.1f%%', shadow = True, startangle = 90, center = (3, 0))
plt.title('Target for housholds')
plt.legend(target_tot.index, loc = 'center left', bbox_to_anchor = (1.1, .5))
plt.axis('equal')

plt.show()


# In[ ]:


# Dimension of household level data
train_hh.shape


# ### Missing data

# In[ ]:


# Percent og missing data
train_hh_na = (train_hh.isnull().sum() / len(train_hh)) * 100
train_hh_na = train_hh_na[train_hh_na != 0].sort_values(ascending = False)
pd.DataFrame({'Percent of missing' :train_hh_na})


# In[ ]:


# Delete columns with over 70% of missing
train_hh = train_hh.drop(['rez_esc', 'v18q1', 'v2a1'], axis = 1)
train_hh.shape


# ### Distribution of continuous variables by target

# In[ ]:


# Function for plot of violin and strip plots together 
def ViolinStrip(xvar, yvar, data, ylab):
    import seaborn as sns
    sns.violinplot(x = xvar, y = yvar, data = data, inner = None, color = 'lightgray')
    sns.stripplot(x = xvar, y = yvar, data = data, size = 2, jitter = True)
    plt.ylabel(ylab)
    return plt.show()


# In[ ]:


# Number of childern younger than 12Y
ViolinStrip(xvar = 'Target', yvar = 'r4t1', data = train_hh, ylab = 'Number of childern younger than 12Y')


# In[ ]:


# Number of childre under 19
ViolinStrip(xvar = 'Target', yvar = 'hogar_nin', data = train_hh, ylab = 'Number of childre under 19')


# In[ ]:


# Total number of household
ViolinStrip(xvar = 'Target', yvar = 'r4t3', data = train_hh, ylab = 'Total number of household')


# ### Bivariate plots

# In[ ]:


# Number of rooms vs target
sns.jointplot(x = 'rooms', y = 'Target', data = train_hh, kind = 'kde')
plt.show()


# In[ ]:


# Average education vs target
sns.jointplot(x = 'meaneduc', y = 'Target', data = train_hh, kind = 'kde')
plt.show()


# In[ ]:


# Age vs target
sns.jointplot(x = 'age', y = 'Target', data = train_hh, kind = 'kde')
plt.show()


# In[ ]:


# Mean education vs age
sns.jointplot(x = 'meaneduc', y = 'age', data = train_hh, kind = 'kde')
plt.show()


# ### Correlalation with target

# In[ ]:


# Correlation with target
cor = pd.DataFrame(train_hh.dropna().corr()['Target'].drop('Target'))
cor['cor_abs'] = cor.abs()
cor.columns = ['CORR', 'CORR_abs']
cor = cor.sort_values('CORR_abs', ascending = False)
cor = cor[cor.CORR_abs >= .2]
cor[:10]


# In[ ]:


# Plot correlation
sns.barplot(x=cor.index, y=cor.CORR)
plt.xticks(rotation = 60)
plt.show()

