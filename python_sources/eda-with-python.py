#!/usr/bin/env python
# coding: utf-8

# ## Just some basic EDA and a how-to guide. ##

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
# "magic" command to make plots show up in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

filepath = '../input/'
acc_df = pd.read_csv(filepath + 'accepted_2007_to_2016.csv.gz')

# this is a dataset with rejected loans from lendingclub
rej_df = pd.read_csv(filepath + 'rejected_2007_to_2016.csv.gz')


# In[ ]:


# this is how we can see how many entries are in the dataframe (df)
# it also works for numpy arrays
acc_df.shape


# In[ ]:


# this is how many rows pandas will show by default with methods like pd.dataframe.head()
pd.options.display.max_rows


# In[ ]:


# we want to increase it, because in this case there are a lot of column names
pd.options.display.max_rows = 1000


# In[ ]:


# the .T is transposing the matrix.
# We do this so the 111 column dataframe is easier to read (easier to scroll down than sideways) 
# .head() shows the first few rows of the data
acc_df.head().T


# In[ ]:


# .tail() shows the last few rows
acc_df.tail().T


# In[ ]:


# .info() tells us the datatype(int64, `object` is a string)
# and will also tell us the number of non-null (not missing) data points for each column
# because this dataframe is so large, we have to force it to show the datatypes and non-null numbers with the arguments
acc_df.info(verbose=True, null_counts=True)


# # Some common pandas functions
# 
# We find ourselves doing the same types of things often in pandas.  For example, masking.  
# 
# ## Masking dataframes
# 
# Maybe we only want to see one column.  We do this like
# 
# `acc_df['int_rate']`
# 
# We can also subset a dataframe based on some critereon.  Let's look at only the highest interest rates
# 
# `acc_df[acc_df['int_rate'] > 20]`
# 
# We can take the mean, median, get the standard deviation of columns:
# 
# `acc_df['int_rate'].mean()`

# In[ ]:


# shows some common summary statistics
# again, transposing with .T to make it easier to read
acc_df.describe().T


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
# "magic" command to make plots show up in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


acc_df[acc_df['int_rate'] > 20]['int_rate'].mean()
# thats a bit of a complicated statement.  Breaking it down:
# acc_df['int_rate'] > 20 returns a mask: an array of True/False values
# putting this array into acc_df[] returns the dataframe rows where the interest rates are greater than 20

# so this first part: acc_df[acc_df['int_rate'] > 20]
# gives us a dataframe

# we select a column with ['int_rate'] at the end.  Then we get the average value with .mean()


# ## Finding unique values
# 
# Numpy has a function for finding the unique values in an array:
# 
# np.unique(array)
# 
# This is built into pandas:
# 
# acc_df['grade'].unique()
# 
# shows us the unique values in that column.

# In[ ]:


acc_df['grade'].unique()


# In[ ]:


# selecting only grade A loans:
acc_df[acc_df['grade'] == 'A'].describe().T


# In[ ]:


acc_df['loan_status'].unique()


# In[ ]:


# looking at only defaulted loans:
default_categories = ['Default', 'Charged Off', 'Does not meet the credit policy. Status:Charged Off']
# .isin() is a trick for checking if something is in a list
# it's a pandas-specific function
acc_df[acc_df['loan_status'].isin(default_categories)].describe().T
# check out the average interest rate and dti (debt-to-income)


# In[ ]:


# seaborn is a recently-created Python library for easily making nice-looking plots
# you will have to install it with `conda install seaborn` or `pip install seaborn`, etc

# the docs are here for this function: http://seaborn.pydata.org/generated/seaborn.distplot.html
# found by Googling 'seaborn histogram'
f = sns.distplot(acc_df['dti'])


# In[ ]:


# outliers are screwing up the histogram... remove them
# adapted from http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
# we're using interquartile range to determine outliers
def reject_outliers(sr, iq_range=0.5, side='left', return_mask=False):
    """
    Takes an array (or pandas series) and returns an array with outliers excluded, according to the
    interquartile range.
    
    Parameters:
    -----------
    sr: array
        array of numeric values
    iq_range: float
        percent to calculate quartiles by, 0.5 will yield 25% and 75%ile quartiles
    side: string
        if 'left', will return everything below the highest quartile
        if 'right', will return everything above the lowest quartile
        if 'both', will return everything between the high and low quartiles
    """
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    if side=='both':
        mask = (sr - median).abs() <= iqr
    elif side=='left':
        mask = (sr - median) <= iqr
    elif side=='right':
        mask = (sr - median) >= iqr
    else:
        print('options for side are left, right, or both')
    
    if return_mask:
        return mask
    
    return sr[mask]


# In[ ]:


# sweeeeeeeetttt....
dti_no_outliers = reject_outliers(acc_df['dti'], iq_range=0.85) # arrived at 0.85 via trial and error
f = sns.distplot(dti_no_outliers)
# other types of plot examples:
# http://seaborn.pydata.org/examples/


# In[ ]:


# sets the xkcd style if you want to make the plots look funny...may neet to install some fonts
plt.xkcd()


# In[ ]:


f = sns.distplot(dti_no_outliers)


# In[ ]:


# back to default
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# In[ ]:


f = sns.distplot(dti_no_outliers)


# In[ ]:


# other styles:
# http://matplotlib.org/users/style_sheets.html
# this is the R ggplot style, which some people really like
plt.style.use('ggplot')


# In[ ]:


# http://seaborn.pydata.org/generated/seaborn.regplot.html
# takes a long time because there are a lot of points, but works for smaller datasets
# f = sns.regplot(data=acc_df, x='dti', y='int_rate', fit_reg=False)
# instead, lets make a heatmap:
# http://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
# http://seaborn.pydata.org/generated/seaborn.jointplot.html
mask = reject_outliers(acc_df['dti'], iq_range=0.85, return_mask=True)
f = sns.jointplot(data=acc_df.ix[mask, :], x='dti', y='int_rate', kind='hex', joint_kws=dict(gridsize=50))


# In[ ]:




