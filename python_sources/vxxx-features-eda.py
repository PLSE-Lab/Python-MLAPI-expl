#!/usr/bin/env python
# coding: utf-8

# In this kernel I want to make an univariate analysis of Vxxx features. looking ahead, I think that these features are very important for fraud detection, because they give us a lot of information which transaction is fraudent and which is not, and using these features we can, theoretically, reduce our false negatives and false positives rate. Also they give us a lot of insights about feeature selection.
# 
# We do not know, what these features means, only information we have is:
# 
# Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image, display

get_ipython().run_line_magic('matplotlib', 'inline')

# Plots look better and clearer in svg format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Some aesthetic settings
plt.style.use('bmh')
sns.set(style = 'white', font_scale = 0.6, rc={"grid.linewidth": 0.5, "lines.linewidth": 1})


# In[ ]:


# Loading dataset
train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col = 'TransactionID')


# In[ ]:


# Select only Vxx features
v_cols = [f'V{i}' for i in range(1, 340)]
# v_cols.append('TransactionAmt')

train_df = train[v_cols]

train_df['isFraud'] = train['isFraud']

del train

gc.collect()


# In[ ]:


def null_table(dataset):
    
    '''Create table with ammount of null values for dataset'''
    
    return pd.DataFrame({'Null values': dataset.isnull().sum(), 
                         '% of nulls': round((dataset.isnull().sum() / dataset.shape[0]) * 100, 2)}).T


# In[ ]:


train_df.head()


# In[ ]:


# Null values in train dataset
null_table(train_df)


# Vast ammount of features in datasets have a lot of null values, up to 86%.

# In[ ]:


train_df.describe(include = 'all')


# We can see something interesting here - our features have very small numbers in first, second and third quartiles, but maximum value on contrary is very high.
# 
# Also it seems like some features are binary (V1 for example), some looks like ordinal (V2) and some looks like numeric (V126), let's try to divide features by groups.

# In[ ]:


# Logic is simple:
# if feature have only 2 values - it's binary
# if sum of feature values minus integer sum of feature values equal to zero - it's ordinal
# else it's numeric

binary = []
ordinal = []
numeric = []

for col in v_cols:
    if train_df[col].value_counts().shape[0] == 2:
        binary.append(col)
    elif train_df[col].sum() - train_df[col].sum().astype('int') == 0:
        ordinal.append(col)
    else:
        numeric.append(col)
        
print(f'Binary features {len(binary)}: {binary}\n')
print(f'Ordinal features {len(ordinal)}: {ordinal}\n')
print(f'Numeric features {len(numeric)}: {numeric}\n')        


# Now we can start plotting.
# 
# First - binary features (bars in plots ordered by fraud rate), I'll create function to plot categorical features with fraud rate.

# In[ ]:


def plot_cat(col, rot = 0, n = False, fillna = np.nan, annot = False, show_rate = True):
    
    '''       
       col - column name       
       rot - rotation of xticks
       n - plot only n top values
       fillna - fill nulls with specified values
       annot - whether to plot % of transactions
       show_rate - whether to show fraud rate
    '''
    
    # Fraud rate calculation
    rate = (train_df[train_df['isFraud'] == 1][col].fillna(fillna).value_counts() /
            train_df[col].fillna(fillna).value_counts()).sort_values(ascending = False)
    
    # Values ordered by fraud rate
    if n:
        order = rate.iloc[:n].index
    else:
        order = rate.index    
    
    g1 = sns.countplot(train_df[col].fillna(fillna), hue = train_df['isFraud'], order = order)
    g1.set_ylabel('')
    
    # Annotations show
    if annot:
        for p in g1.patches:
            g1.annotate('{:.2f}%'.format((p.get_height() / train_df.shape[0]) * 100, 2), 
                       (p.get_x() + 0.05, p.get_height()+5000))
            
    plt.xticks(rotation = rot)
    
    # Fraud rate show
    if show_rate:
        g2 = g1.twinx()
        g2 = sns.pointplot(x = rate.index.values, y = rate.fillna(0).values, order = order, color = 'black')
        plt.xticks(rotation = rot)


# Looking ahead, I want to plot fraud rates for different values of features.

# In[ ]:


def plot_rate(cols, legend = True, figsize = (13, 4), alpha = 1, df = False, fillna = 'Null'):
    
    '''
        Plot fraud rates for selected features
        cols - list of features
        legend - whether to show legend
        figsize - size of plot
        alpha - alpha
        df - returns only dataframe if True
    '''
    
    cat = []
    val = []
    clmn = []


    for col in cols:
        rate = (train_df[train_df['isFraud'] == 1][col].fillna(fillna).value_counts() /
                    train_df[col].fillna(fillna).value_counts()).sort_values(ascending = False)

        cat += list(rate.index.values)
        val += list(rate.values)
        clmn += [col] * rate.shape[0]

    kur = pd.DataFrame({'cat': cat, 'val': val, 'clmn': clmn})
    
    if df:
        return kur

    fig = plt.figure(figsize = figsize)
    g = sns.pointplot(x = 'cat', y = 'val', hue = 'clmn', data = kur, plot_kws = dict(alpha = alpha))
    plt.setp(g.collections, alpha = alpha) #for the markers
    plt.setp(g.lines, alpha = alpha)       #for the lines
    plt.legend().set_visible(legend)


# In[ ]:


plot_rate(binary, alpha = 0.6)


# We can see that values for our binary features have very similar fraud rates.
# 
# Let's return to plotting

# In[ ]:


fig = plt.figure(figsize = (11, 15))
for i, col in enumerate(binary):
    plt.subplot(f'42{i}')
    plot_cat(col, annot = True, fillna = 'Null')
plt.tight_layout()


# Let's look at counts:

# In[ ]:


for col in binary:
    print(train_df[col].value_counts(), '\n')


# 0 values (2 for V305 feature) of binary features have very small ammount of transactions - less than 0.1% and fraud rate of these transacions is less than 0.1% or equal to 0.

# Next step - Ordinal features.
# 
# We have 257 ordinal features in dataset, i'll plot them by small groups and make some preparations for aesthetic purposes.
# 
# First, I want to divide them by number of values, if feature have more than 20 unique values - it goes to long_ordinal list, else - to short_ordinal list.

# In[ ]:


# Divide features by number of values
short_ordinal = [] # less or equal to 20 values
long_ordinal = [] # more than 20 values

for col in ordinal:
    if train_df[col].value_counts().shape[0] > 20:
        long_ordinal.append(col)
    else:
        short_ordinal.append(col)

print(f'Short: {len(short_ordinal)}', short_ordinal, '\n')
print(f'Long: {len(long_ordinal)}', long_ordinal)


# Let's look at fraud rates.

# In[ ]:


# short_ordinal
plot_rate(short_ordinal, legend = False, figsize = (13, 7), alpha = 0.4)


# 6, 3, 4, 2, 5, 7, 8, 9 values don't give us much information, but we can see that a lot of features have fraud rate close to 0 at Null values, as 1 and 0 values.
# 
# On contrary: 15, 24, 23, 17, 16, 18, 19 values have features with fraud rate equal to 1.
# 
# It's a pity that we can't make such plot for long_ordinal, it's too computational expensive and even if we will have unlimited computational resourses, all we got is mess.
# 
# So, I want to use different approach for these features.

# In[ ]:


# Return dataframe for long_ordinal
long_df = plot_rate(long_ordinal, df = True)
long_df.shape


# In[ ]:


long_df.head()


# In[ ]:


# Group long_df by values and calculate mean and std
long_df = long_df['val'].groupby(long_df['cat']).agg(['mean', 'std'])
long_df = long_df.dropna()
print(long_df.shape)
long_df.head()


# In[ ]:


long_df['cat'] = long_df.index.values
long_df.head()


# In[ ]:


# Points - mean
# Bars - std
fig = plt.figure(figsize = (13, 20))
sns.barplot(y = 'cat', x = 'std', data = long_df)
sns.pointplot(y = 'cat', x = 'mean', data = long_df, color = 'black')
plt.show()


# Now let's return to the routine and make plots for ordinal features.
# 
# When I worked on this part, I faced with a problem, when such number of plots just crashed my kernel, so I decided to save plots in .png fomat and show them here as pictures.
# 
# Also, I'm including code for plots in comments.

# In[ ]:


# Function for plotting in grid
'''
def plot_grid(data, rows, cols, start, end, rot = 90, n = 50, figsize = (11, 8)):    
    fig = plt.figure(figsize = figsize)
    for i, col in enumerate(data[start:end]):
        plt.subplot(f'{rows}{cols}{i}')
        plot_cat(col, annot = False, n = n, fillna = 'Null', rot = rot)
    plt.tight_layout()
'''

# plot_grid(data = short_ordinal, rows = 3, cols = 3, start = 0, end = 9)


# In[ ]:


# short_ordinal
for i in range(1, 16):
    display(Image(f'../input/fraud-detection-plots/{i}.png'))


# We can see something interesting here. Almost every feature in short_ordinal have a fraud peaks on some value for example V17 and V18 have 6 values with 100% fraud rate. Also every feature have values which fraud rate close or equal to zero.

# In[ ]:


# long_ordinal, includes only 50 values sorted by fraud rate
for i in range(16, 32):
    display(Image(f'../input/fraud-detection-plots/{i}.png'))


# Similar situation here, we can easily see which walues give us 100% or 0% fraud rates.
# 
# Next - numeric features. I'll use log1p transformed distribution plots to see fraud and non fraud peaks.

# In[ ]:


print(numeric)


# In[ ]:


def plot_hist(data, start, end, figsize = (11, 7)):
    fig = plt.figure(figsize = figsize)
    for i, col in enumerate(data[start:end]):
        plt.subplot(f'33{i}')
        sns.distplot(train_df[col].apply(np.log1p), hist = False, label = 'Train', color = 'black')
        sns.distplot(train_df[train_df['isFraud'] == 1][col].apply(np.log1p), hist = False, label = 'Fraud', color = 'red')
        sns.distplot(train_df[train_df['isFraud'] == 0][col].apply(np.log1p), hist = False, label = 'NonFraud', color = 'green')
        plt.legend()
    plt.tight_layout()


# In[ ]:


for i in range(9):
    plot_hist(numeric, i * 9,  i * 9 + 9)


# In[ ]:




