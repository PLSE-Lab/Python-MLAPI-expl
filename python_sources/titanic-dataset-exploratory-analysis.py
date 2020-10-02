#!/usr/bin/env python
# coding: utf-8

# # Titanic dataset exploratory analysis
# 
# Purpose of this notebook is to run provide an exploratory data analysis (EDA) of the Titanic dataset provided along with the [Kaggle competion](https://www.kaggle.com/c/titanic).
# 
# This analysis is performed on the training dataset only.
# 
# These are some useful links I came along when diving into theory of EDA:
# 
#  - https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
#  - http://gchang.people.ysu.edu/class/mph/notes_06/Diagram02.pdf

# In[59]:


#load packages
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
from IPython import display
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read train and test data
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


# ## Data types of variables
# 
# Let us have a look at the data to first see which variables are continuous and which are categorical. This knowledge will allow us to decide how to plot each variable.

# **Data types**

# In[6]:


train.dtypes


# **Descriptive statistics**

# In[7]:


train.describe()


# **Number of unique values for each variable**

# In[78]:


# define function to count unique values
def num_unique(series): 
    return len(series.unique())
    
train.apply(num_unique, axis = 0)


# **Head of the dataset**

# In[8]:


train.head()


# **Median of variables**

# In[31]:


train.median()


# Purpose of this document is mainly to plot the data to charts. Therefore we will treat as continuous those variables, which have many unique values and which therefore are suitable for plotting using distribution plot (histograms with estimated densities). On the other hand variables with only few values will be treated as categorical. 
# 
# For other types of analyses this distinction might take a different form. When e.g. studying correlation between number of siblings or parents/childrens and other variables, we might consider these as continous for purposes of regressions or other models.
# 
# For our purposes we therefore distinguish:
# 
# **Categorical:**
#  - Survived
#  - Sex
#  - Embarked
#  - Pclass
#  - SibSp
#  - Parch
#  
# **Continuous:**
#  - Age
#  - Fare

# In[79]:


# declare lists of categorical and numerical variables
CAT_ATTRIBS = ['Survived','Sex','Embarked','Pclass','SibSp','Parch']
CAT_ATTRIBS_WO_SUR = ['Sex','Embarked','Pclass','SibSp','Parch'] # categorical w/o Survive
NUM_ATTRIBS = ['Age','Fare']

n_cat_attribs = len(CAT_ATTRIBS)
n_cat_attribs_wo_sur = len(CAT_ATTRIBS_WO_SUR)
n_num_attribs = len(NUM_ATTRIBS)


# ## Univariate analyses

# ### Categorical variables

# In[81]:


# TBD - rewrite the following code snippet so it is not that hard-coded (for loop)
f,ax = plt.subplots(2,3,figsize=(12,8))
sns.countplot('Sex',data=train,ax=ax[0,0])
sns.countplot('Pclass',data=train,ax=ax[0,1])
sns.countplot('Embarked',data=train,ax=ax[0,2])
sns.countplot('Survived',data=train,ax=ax[1,0])
sns.countplot('SibSp',data=train,ax=ax[1,1])
sns.countplot('Parch',data=train,ax=ax[1,2])
plt.suptitle('Categorical variables distributions', fontsize = 20)
plt.subplots_adjust(top=0.92)
# plt.tight_layout()


# Takeaways from the plot:
# 
# **Sex**
# 
# There were alomost two times more male passengers than female ones.
# 
# **Pclass**
# 
# Most passengers traveled in the 3rd class. The least in the 2nd class.
# 
# **Embarked**
# 
# Majority of passengers embarked in Southampton, whereas the least in Queenstown.
# 
# **Survived**
# 
# There were less survivors than those who died in the accident.
# 
# **SibSp**
# 
# Most people traveled without any siblings or spouse. Though there is a significant number of passengers with one sibling or spouse.
# 
# **Parch**
# 
# Most people traveled without any parents or children. One or two parents/children are also quite frequent category.

# ### Continuous variables

# In[36]:


f,ax = plt.subplots(2,2,figsize=(14,8))

sns.distplot(train['Age'].dropna(),ax=ax[0,0])
sns.distplot(train['Fare'].dropna(),ax=ax[0,1])
sns.boxplot(train['Age'].dropna(),ax=ax[1,0])
sns.boxplot(train['Fare'].dropna(),ax=ax[1,1])

plt.suptitle('Continuous variables distributions', fontsize = 20)
plt.subplots_adjust(top=0.90)


# Takeaways from the plot:
# 
# **Age**
# 
# Average age is less than 30 and median is 28 (see tables above). The distribution is multimodal with estimated density having (another) local maximum at values just above zero. Disregarding this maximum, the distribution is skewed a bit to the right with maximum value 80.
# 
# **Fare**
# 
# Distribution is multimodal and highly skewed to the rigth with several clusters of outliers. The median is as low as 14.5 and the mean is pulled by the outliers to 32 (see table above).  

# ## Multivariate analyses

# ### Categorical vs. categorical variables

# In[42]:


# use categorical variables
data_hist_grid = train.loc[:, CAT_ATTRIBS] #.dropna(axis=0, how='any')

f,ax = plt.subplots(n_cat_attribs,n_cat_attribs,figsize=(20,16))

# iterate over categorical variables (two times)
for i in range(n_cat_attribs):
    for j in range(n_cat_attribs):
        
        # histogram on diagonal
        if i==j:
            sns.countplot(data_hist_grid.columns[i],
                          data=data_hist_grid,
                          ax=ax[i,j])
            
        # histogram with categories off diagonal
        else:
            sns.countplot(data_hist_grid.columns[i],
                          data=data_hist_grid,
                          ax=ax[i,j],
                          hue=data_hist_grid.columns[j])

# The next two lines do not work together with plt.tight_layout()
# plt.suptitle('Continuous variables distributions', fontsize = 20)
# plt.subplots_adjust(top=0.92) #left=0.2, wspace=0.8, )
        
plt.tight_layout()


# Takeaways from the plot:
# 
# **Survived**
#  - females were more likely to survive
#  - people who embarked in Cherbourg were more likely to survive compared to other ports (the highest proportion of the 1st class passengers)
#  - upper class passengers (lower class number) were more likely to survive
#  - having a sibling/spouse increased a chance of surviving (possibly an effect of a females being spouses?)
#  - having a parent/child increased a chance of surviving (possibly an effect of a child being more likely to survive?)
#  
# **Sex**
#  - proportion of men relative to women was the highest for passangers embaring in Southampton
#  - proportion of men relative to women was the higher for lower classes
#  - passanger traveling alone without any sibling/spouse or parent/child were more likely male
#  
# **Embarked**
#  - Passangers that embarked in Queenstown traveled mostly in the 3rd class
#  
# **Pclass**
#  - Passengers travelling without any sibling/spouse were more likely to travel in the 3rd class compare to other classes. Passengers having at least one sibling/spouse did not use any class significantly more than other class.

# ### Continuous vs. continuous variables

# In[56]:


data_for_pairplot = train[['Age','Fare','Survived']].dropna(axis=0, how='any')
sns.pairplot(data_for_pairplot, diag_kind="kde", kind="reg", hue='Survived',
                   vars=('Age','Fare'))
# plt.fig.suptitle('Continuous variables distributions', fontsize = 20)
# plt.fig.subplots_adjust(top=0.90) 


# Takeaways from the plot:
# 
# Not much on this plot. The reggression lines on the bottom left panel can be interpretted in a sense that "older people who survived paid relatively higher fare compared to those who did not survive". Higher fare goes with the fact that mostly higher-class passangers survived. 

# ### Continuous vs. categorical variables

# In[82]:


# cut the outliers in the Fare variable
train_wo_outliers = train[train['Fare'] < 180]


# In[83]:


f,ax = plt.subplots(n_cat_attribs_wo_sur,n_num_attribs,figsize=(20,16))

# iterate over categorical variables
for i in range(n_cat_attribs_wo_sur):
    
    # iterate over numerical varibles
    for j in range(n_num_attribs):
        # print('i' + str(i) + ', j' + str(j))
        
        # create list of unique values
        unique_vals = train_wo_outliers[CAT_ATTRIBS_WO_SUR[i]].unique()
        
        # iterate over each unique value
        for unique_val in unique_vals:
        
            # subset the data
            data_subset = train_wo_outliers.loc[train_wo_outliers[CAT_ATTRIBS_WO_SUR[i]] == unique_val,NUM_ATTRIBS[j]]
            
            # kernel density estimation only works for certain number of observations
            if len(data_subset) > 10:
                sns.kdeplot(data_subset,ax=ax[i,j],label=unique_val)
                
            ax[i,j].set_title(NUM_ATTRIBS[j] + ' vs ' + CAT_ATTRIBS_WO_SUR[i])

plt.tight_layout()


# In[84]:


f,ax = plt.subplots(n_cat_attribs_wo_sur,n_num_attribs,figsize=(20,16))

# iterate over categorical variables
for i in range(n_cat_attribs_wo_sur):
    
    # iterate over numerical varibles
    for j in range(n_num_attribs):
        
        sns.violinplot(x=CAT_ATTRIBS_WO_SUR[i], 
                       y=NUM_ATTRIBS[j],
                       data=train_wo_outliers,
                       palette="muted",
                       split=True,
                       ax=ax[i,j])

plt.tight_layout()


# *Technical note: the Seaborn 'kdeplot' command does not handle datasets with only small number of observations and these cases had to be ommited. Therefore there might be slight differences between the violin and the density plots (missing categories with low number of observations).* 
# 
# *Note: Some of the conclusions below might be confirmed by regression (correlation) analyses.*
# 
# Takeaways from the plot:
# 
# ** Sex **
#  - Age of males and females is roughly similarly distributed.
#  - Modus of fares distribution is much more distinct and lies at lower values for males compared to females. Probably effect of less females traveling in the 3rd class.
#  
# **Embarked**
#  - Age of passengers is similar for all points of embarkement, with Queenstown's distribution being slightly flatter and Southampton's slightly narrower.
#  - Fares distribution is significantly more narrow for Queenstown than for other embrarkment points, which correspond to the fact that mostly 3rd class passengers start their journey in this port. Opposite applies for Cherbourg.
#  
# **Pclass**
#  - Younger the passengers, lower the class.
#  - Higher the class, higher the price.
#  
# **Sibsp**
#  - Higher number of siblings corresponds to lower age. This pattern reverses at value of Sibsp = 4.
#  - Higher number of siblings roughly corresponds to higher fare. 
#  
# ** Parch **
#  - Age decreases with increasing number of parents/children onboard for low values of the latter. Interesting, this trend reverses for higher number of parents/children.
#  - Number of parents/childer is positively correlated with fare
#  
# 

# ### Continuous vs. categorical variables with Survived distinctions

# In[85]:


f,ax = plt.subplots(n_cat_attribs_wo_sur,n_num_attribs,figsize=(20,16))

# iterate over categorical variables
for i in range(n_cat_attribs_wo_sur):
    
    # iterate over numerical varibles
    for j in range(n_num_attribs):
        
        sns.violinplot(x=CAT_ATTRIBS_WO_SUR[i], 
                       y=NUM_ATTRIBS[j],
                       hue="Survived",
                       data=train_wo_outliers,
                       palette="muted",
                       split=True,
                       ax=ax[i,j],
                       inner="stick")

plt.tight_layout()


# Takeaways from the plot:
# 
# **Sex**
# 
#  - Slight multimodality of age distributions for passengers who survived corresponds to the fact children were more likely to survive. This effect is more distinct for males.
# 
# **Embarked**
# 
#  - Age of passengers who embarked in Queenstown and did survived is quite densly disributed around 30 years. Respective distribution for Queenstown passanger who did not survive is much wider.
#  - Passengers from Cherbourg who paid higher fare were more likely to survive compared to passengers paying lower fare. This pattern corresponds to the fact that Cherbourg saw the highest proportion of the 1st class passengers. The pattern does not apply to other embarkment ports as proortion of 3rd (and 2nd) class passengers was much higher in these ports.
#  
# **Pclass**
# 
#  - Most children traveling in the second class survived.
#  - Passengers who traveled in the first class and survived paid rather higher price compared to those who did not survive.
# 
# **SibSp and Parch**
# 
#  - Patterns in these plots are rateher inconclusive due to low number of observations in groups of data.
