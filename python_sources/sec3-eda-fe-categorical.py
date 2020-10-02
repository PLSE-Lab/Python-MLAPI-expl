#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# This section focuses on EDA and feature engineering for categorical columns. 
# 
# For previous work access following link
# 1. Data cleaning : https://www.kaggle.com/lajari/sec1-tedious-data-cleaning
# 2. EDA and Feature engineering for numerical features :https://www.kaggle.com/lajari/sec2-eda-feature-engineering
# 
# We have observed around 43 categorical columns in given datset. Our study highlights, which features have significant information to predict our target column. Basically we are filtering out less significant column and test final data (including numerical and categorical features) on simple regressor.
# 
# Note: The data generated from previous section will be used for analysis. 
# 

# ## Load Data

# In[ ]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train =pd.read_csv('/kaggle/input/sec2-eda-feature-engineering/engineered_train.csv')
test =pd.read_csv('/kaggle/input/sec2-eda-feature-engineering/engineered_test.csv')
traintest = pd.concat([train,test], axis = 0,ignore_index = False)
train.head()


# ## EDA

# In[ ]:


catcols = train.select_dtypes(include=np.object).columns
print('Number of categorical columns:',len(catcols))


# In[ ]:


def generate_plots(r,c,columns):
    """
    Generate pair of boxplot and countplot for each column in columns each row contains two such pairs'
    
    """
    fig ,axs = plt.subplots(r,c,figsize=(20,40))

    axs = axs.flatten()
    i = 0
    for col in columns:
        
        sns.boxplot(x=train[col],y=train['LogPrice'],ax=axs[i])
        sns.countplot(train[col], ax=axs[i+1])
        
        if train[col].nunique()>6:
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
            axs[i+1].set_xticklabels(axs[i+1].get_xticklabels(), rotation=45) 
            
        i=i+2
        plt.tight_layout()
    


# In[ ]:


generate_plots(11,4,catcols[:22])


# In[ ]:


generate_plots(11,4,catcols[22:])


# In above plots we have observed two import things. First, there are some set of features which has most of the data from same group. Second,some set of features whose median of groups are quite same. In next section we will remove such features as they do not provide significant information for prediction.

# ## Analysis of Frequency Ditribution:
# 
# We are aiming to remove the columns with low variability. Here, low variability implies the feature which has more than 95% of total rows contain same value. 

# In[ ]:


low_var_cols = []
for col in catcols:
    freq_db = (traintest[col].value_counts(normalize = True))      # We will analyse for whole dataset (include train and test)
    if freq_db[freq_db>0.95].sum() != 0:
        low_var_cols.append(col)
low_var_cols
    


# ## ANOVA Test:
# 
# ANOVA is called as analysis of Variance. It is used to compare the means of different groups. In this section, we are performing one way ANOVA. It means analysis will contain one feature at a time. and we are trying to analyse the differences in means of various groups (here groups refers to levels in categorical columns). 
# 
# ANOVA Hypothesis:
# 
#     Null hypotheses: Groups means are equal (no variation in means of groups)
#     Alternative hypotheses: At least, one group mean is different from other groups
# 
# If P-value obtained from ANOVA analysis is less than 0.05, then we conclude that there are significant difference among groups. In following code block we will filter out the columns which fails to reject null hypothesis.

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

const_mean_across_grp = []

print('Columns | P-value\n','-'*30)

for col in catcols:
    mod = ols('LogPrice ~ '+col ,data=train).fit()
    anova_table = sm.stats.anova_lm(mod,typ=2)
    
    pr  = anova_table.loc[col,'PR(>F)']
   
    if pr > 0.05:
        print(col,'|',pr)
        const_mean_across_grp.append(col)


# We are going to remove above columns as their P-value >0.05. It means these features group means are not significantly differ.

# ## Dropping less important columns:
# We will drop columns listed in 'low_var_cols' and 'const_mean_across_grp' from train and test dataset and create final dataset for our modelling

# In[ ]:


s1 = set(const_mean_across_grp)
s2 = set(low_var_cols)
dropcols = list(s1.union(s2))

train.drop(dropcols,axis = 1,inplace=True)
test.drop(dropcols,axis = 1,inplace = True)


# ## Test Model:

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# one hot encoding for categoricals
cats = list(train.select_dtypes(object).columns)
all_X = pd.get_dummies(data = train,columns = cats,sparse = True).copy()
all_X.drop(['LogPrice','SalePrice','Id'],axis =1, inplace=True)
all_y = train['LogPrice']

# Modelling and validating simple regressor
scores =cross_val_score(LinearRegression(),all_X,all_y, cv=3,scoring = 'neg_mean_squared_error')

# RMSE score
np.sqrt(-scores.mean())


# We obtained quite good result using simple model. Our next section about predictive modelling is coming soon. In that section, we will go through various traditional and advanced modelling approaches and find out the best model for our given problem.
# 
# ## **Welcome all your comments and feedback. Don't forget to upvote this notebook if you find this study useful.**

# In[ ]:


train.to_csv('eng_filt_train.csv',index = False)
test.to_csv('eng_filt_test.csv',index = False)

