#!/usr/bin/env python
# coding: utf-8

# ## Personality Traits Analysis of Celebrity Twitter Accounts

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation Libraries
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import warnings
import re

pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-darkgrid')
pd.set_option('display.max_columns', 50)
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format


# ## Data Loading

# In[ ]:


url = '../input/personality-traits-of-twitter-users-celebrities/analisis.csv'
data = pd.read_csv(url, header='infer')


# ## Data Exploration

# In[ ]:


data.head()


# In[ ]:


#Checking for null/missing values
data.isna().sum()


# In[ ]:


#Renaming the columns
data = data.rename(columns={'usuario':'User','op':'Openness','co':'Conscientiousness','ex':'Extraversion',
                            'ag':'Agreeableness','ne':'Neuroticism','categoria':'Category' })


# Converting the Category column data to data type = category

# In[ ]:


data['Category'] = data['Category'].astype('category')


# Converting the Word Count column to data type = int and round it up

# In[ ]:


data['wordcount'] = data['wordcount'].astype('int')

# Function to round up word count
def roundup_wordcount(count):
    count = round(count)
    return count

# Applying the function to Word Count Column
data['wordcount'] = data['wordcount'].apply(roundup_wordcount)


# In[ ]:


data.head()


# In[ ]:


data.describe().transpose()


# ### Univariate Analysis - Numerical Columns

# In[ ]:


# Function that shows the summary and density distribution of a numerical attribute:
def summary(x):
    x_min = data[x].min()
    x_max = data[x].max()
    Q2 = data[x].quantile(0.50)
    x_mean = data[x].mean()
    print(f'4 Point Summary of {x.capitalize()} Attribute:\n'
          f'{x.capitalize()}(min)   : {x_min}\n'
          f'Q2(Median)              : {Q2}\n'
          f'{x.capitalize()}(max)   : {x_max}\n'
          f'{x.capitalize()}(mean)  : {round(x_mean)}')

    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace = 0.6)
    sns.set_palette('deep')
    
    plt.subplot(221)
    ax1 = sns.distplot(data[x], color = 'r')
    plt.title(f'{x.capitalize()} Density Distribution')
    
    plt.subplot(222)
    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)
    plt.title(f'{x.capitalize()} Violinplot')
    
    plt.subplot(223)
    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)
    plt.title(f'{x.capitalize()} Boxplot')
    
    plt.subplot(224)
    ax3 = sns.kdeplot(data[x], cumulative=True)
    plt.title(f'{x.capitalize()} Cumulative Density Distribution')
    
    plt.show()


# ### Openness Analysis

# In[ ]:


summary('Openness')


# **Analysis: ** 
# The Openness is uniformly distributed between min-30 & max-72

# ### Conscientiousness Analysis

# In[ ]:


summary('Conscientiousness')


# **Analysis:**
# The 'Conscientiousness' is uniformly distributed between min-8 & max-50 

# ### Extraversion Analysis

# In[ ]:


summary('Extraversion')


# **Analysis:**
# 
# The 'Extraversion' is uniformly distributed between min-19 & max-60

# ### Agreeableness Analysis

# In[ ]:


summary('Agreeableness')


# **Analysis:**
# 
# The 'Agreeableness' is uniformly distributed between min-9 & max-41

# ### Neuroticism Analysis

# In[ ]:


summary('Neuroticism')


# **Analysis:**
# 
# The 'Neuroticism' is uniformly distributed between min-1 & max-24

# ### Word Count Analysis

# In[ ]:


summary('wordcount')


# **Analysis:**
# 
# The Word Count is uniformly distributed between min-5 & max-217

# ## Univariate Analysis - Categorical Column

# In[ ]:


# Create a function that returns a Pie chart and a Bar Graph for the categorical variables:
def cat_view(x = 'Education'):
    """
    Function to create a Bar chart and a Pie chart for categorical variables.
    """
   
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
     
    """
    Draw a Pie Chart on first subplot.
    """    
    s = data.groupby(x).size()

    mydata_values = s.values.tolist()
    mydata_index = s.index.tolist()
    
    explode = []
    
    for i in range(len(mydata_index)):
        explode.append(0.1)
    
    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(mydata_values, autopct=lambda pct: func(pct, mydata_values),explode=explode,
                                      textprops=dict(color="w", size=8))

    ax.legend(wedges, mydata_index,
              title=f'{x.capitalize()} Index',
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title(f'{x.capitalize()} Piechart', fontsize=16)
       

    fig.tight_layout()
    plt.show()


# ### Category Analysis

# In[ ]:


cat_view('Category')


# **Analysis:**
# 
# Category-2 has the highest Users, while Category-5 has the least Users

# ## Correlation
# 
# Finding correlation between the 5 personality traits and word-count

# In[ ]:


# Creating a seperate dataset with 5 personality traits & word-count
data_sub = data[['Openness','Conscientiousness','Extraversion', 'Agreeableness', 'Neuroticism', 'wordcount']]

corr = data_sub.corr()
plt.figure(figsize=(8, 8))
g = sns.heatmap(corr, annot=True, cmap = 'PuBuGn_r', square=True, linewidth=1, cbar_kws={'fraction' : 0.02})
g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
g.set_title("Correlation between each of Personality Traits & Word Count", fontsize=14)
plt.show()


# **Analysis:**
# 
# Correlation between the following:
# * Openness & Conscientiousness = Positive
# * Openness & Extraversion = Negative
# * Openness & Agreeableness = Negative
# * Openness & Neuroticism = Negative
# 
# * Openness & Wordcount = Positive
# * Conscientiousness & Wordcount = Positive
# * Extraversion & Wordcount = Negative
# * Agreeableness & Wordcount = Negative
# * Neuroticism & Wordcount = Positive
# 

# ## Finding Users with high & low score in each Personality Traits & Wordcounts

# In[ ]:


# Creating a function to find the above average & below average 
def FindUsers(x):
    x_low = data[x].quantile(0.25)
    x_high = data[x].quantile(0.75)
    
    xx_df = data[['User', x]]
    
    xx_df_high = xx_df[xx_df[x] >= x_high]   # Finding High Value Users
    xx_df_low = xx_df[xx_df[x] <= x_low]     # Finding High Value Users
    
    print(f'Users with High {x.capitalize()}:\n')
    print(' , '.join(xx_df_high['User']))
    print()
    print(f'Users with Low {x.capitalize()}:\n')
    print(' , '.join(xx_df_low['User']))
    
    


# ### Openness
# 
# It indicates how open-minded a person is. 
# 
# A person with a high level of openness to experience in a personality test enjoys trying new things. They are imaginative, curious, and open-minded. 
# 
# Individuals who are low in openness to experience would rather not try new things. They are close-minded, literal and enjoy having a routine.

# In[ ]:


FindUsers('Openness')


# ### Conscientiousness
# 
# A person scoring high in conscientiousness usually has a high level of self-discipline. These individuals prefer to follow a plan, rather than act spontaneously. Their methodic planning and perseverance usually makes them highly successful in their chosen occupation.
# 

# In[ ]:


FindUsers('Conscientiousness')


# ### Extraversion
# 
# It indicates how outgoing and social a person is. 
# 
# A person who scores high in extraversion on a personality test is the life of the party. They enjoy being with people, participating in social gatherings, and are full of energy. 
# 
# A person low in extraversion is less outgoing and is more comfortable working by himself.

# In[ ]:


FindUsers('Extraversion')


# ### Agreeableness
# 
# A person with a high level of agreeableness in a personality test is usually warm, friendly, and tactful. They generally have an optimistic view of human nature and get along well with others. 
# 
# A person who scores low on agreeableness may put their own interests above those of others. They tend to be distant, unfriendly, and uncooperative.

# In[ ]:


FindUsers('Agreeableness')


# ### Neuroticism / Emotional stability
# 
# Emotional stability refers to a person's ability to remain stable and balanced. At the other end of the scale, a person who is high in neuroticism has a tendency to easily experience negative emotions. Neuroticism is similar but not identical to being neurotic in the Freudian sense. Some psychologists prefer to call neuroticism by the term emotional stability to differentiate it from the term neurotic in a career test

# In[ ]:


FindUsers('Neuroticism')


# ### Word Count

# In[ ]:


FindUsers('wordcount')

