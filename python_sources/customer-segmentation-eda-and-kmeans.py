#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Libraries" data-toc-modified-id="Libraries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Libraries</a></span></li><li><span><a href="#Functions" data-toc-modified-id="Functions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Functions</a></span></li><li><span><a href="#EDA" data-toc-modified-id="EDA-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>EDA</a></span></li><li><span><a href="#Training-KMeans" data-toc-modified-id="Training-KMeans-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Training KMeans</a></span><ul class="toc-item"><li><span><a href="#Age-and-Spending-Score" data-toc-modified-id="Age-and-Spending-Score-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Age and Spending Score</a></span></li><li><span><a href="#Annual-Income-and-Spending-Score" data-toc-modified-id="Annual-Income-and-Spending-Score-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Annual Income and Spending Score</a></span></li></ul></li></ul></div>

# Let's analyze data from mall customers to dig intuitions about costumer convergence and apply KMeans algorithm to develop insights about how to deal with different clusters. 

# ## Libraries

# In[25]:


# Importing libs
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ## Functions

# The set of functions in this implementation includes:
#     * format_spines();
#     * count_plot();
#     * compute_square_distances();
#     * plot_elbow_method();
#     * plot_kmeans();
# 
# Each one with its set of parameters.

# In[26]:


def format_spines(ax, right_border=True):
    """docstring for format_spines:
    this function sets up borders from an axis and personalize colors
    input:
        ax: figure axis
        right_border: flag to determine if the right border will be visible or not"""
    
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_color('#FFFFFF')
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')


def count_plot(feature, df, colors='Blues_d', hue=False):
    """docstring for count_plot:
    this function plots data setting up frequency and percentage. This algo sets up borders
    and personalization
    input:
        feature: feature to be plotted
        df: dataframe
        colors = color palette (default=Blues_d)
        hue = second feature analysis (default=False)"""
    
    # Preparing variables
    ncount = len(df)
    fig, ax = plt.subplots()
    if hue != False:
        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue)
    else:
        ax = sns.countplot(x=feature, data=df, palette=colors)

    # Make twin axis
    ax2=ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    # Setting borders
    format_spines(ax)
    format_spines(ax2)

    # Setting percentage
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    if not hue:
        ax.set_title(df[feature].describe().name + ' Analysis', size=13, pad=15)
    else:
        ax.set_title(df[feature].describe().name + ' Analysis by ' + hue, size=13, pad=15)
        
    plt.show()

def compute_square_distances(df, Kmin=1, Kmax=12):
    """docstring for compute_square_distances
    this function computes the square distance of KMeans algorithm through the number of
    clusters in range Kmin and Kmax
    input:
        df: dataframe
        Kmin: min index of K analysis
        Kmax: max index of K analysis"""
    
    square_dist = []
    K = range(Kmin, Kmax)
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(df)
        square_dist.append(km.inertia_)
    return K, square_dist

def plot_elbow_method(df, Kmin=1, Kmax=12):
    """docstring for plot_elbow_method
    this function computes the square distances and plots the elbow method for best cluster
    number analysis
    input:
        df: dataframe
        Kmin: min index of K analysis
        Kmax: max index of K analysis"""
    
    # Computing distances
    K, square_dist = compute_square_distances(df, Kmin, Kmax)
    
    # Plotting elbow method
    fig, ax = plt.subplots()
    ax.plot(K, square_dist, 'bo-')
    format_spines(ax, right_border=False)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Square Dist')
    plt.title(f'Elbow Method - {df.columns[0]} and {df.columns[1]}', size=14)
    plt.show()
    
def plot_kmeans(df, y_kmeans, centers):
    """docstring for plotKMeans
    this function plots the result of a KMeans training
    input:
        df: dataframe
        y_kmeans: kmeans prediction
        centers: cluster centroids"""
    
    # Setting up and plotting
    X = df.values
    sns.set(style='white', palette='muted', color_codes=True)
    fix, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='plasma')
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    ax.set_title('KMeans Applied', size=14)
    ax.set_xlabel(f'{df.columns[0]}', size=12, labelpad=5)
    ax.set_ylabel(f'{df.columns[1]}', size=12, labelpad=5)
    format_spines(ax, right_border=False)
    plt.show()


# ## EDA

# In[27]:


# Reading data
df = pd.read_csv(r'../input/Mall_Customers.csv')
df.head()


# In[28]:


# Dims
df.shape


# In[29]:


# Communication
print(f'This dataset has {df.shape[0]} rows and {df.shape[1]} columns.')


# In[30]:


# Null data
df.isnull().sum()


# In[31]:


# Dataset info
df.info()


# In[32]:


# Some statistics
df.describe()


# In[33]:


# Numerical features distribution
sns.set(style='white', palette='muted', color_codes=True)
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.despine(left=True)
axs[0] = sns.distplot(df['Age'], bins=20, ax=axs[0])
axs[1] = sns.distplot(df['Annual Income (k$)'], bins=20, ax=axs[1], color='g')
axs[2] = sns.distplot(df['Spending Score (1-100)'], bins=20, ax=axs[2], color='r')

fig.suptitle('Numerical Feature Distribution')
plt.setp(axs, yticks=[])
plt.tight_layout()
plt.show()


# In[34]:


# Counting gender
custom_colors = ["#3498db", "#C8391A"]
count_plot(feature='Gender', df=df, colors=custom_colors)


# Let's create a age category for future analysis.

# In[35]:


# Looking at age values
df['Age'].describe()


# In[36]:


# Creating new category
bins = [18, 22, 50, 70]
labels = ['Young', 'Adult', 'Senior']
df['Age Range'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

df.head()


# In[37]:


# Result
count_plot(feature='Age Range', df=df, colors='YlGnBu')


# In[38]:


# Gender by Age Range
count_plot(feature='Gender',df=df, hue='Age Range')


# In[39]:


# Maybe the inverse would be more clear
count_plot(feature='Age Range', df=df, colors=custom_colors, hue='Gender')


# In[40]:


# Spending Score Distribution
fig, ax = plt.subplots(figsize=(10, 4), sharex=True)
female = df.loc[df['Gender'] == 'Female']
male = df.loc[df['Gender'] == 'Male']
ax = sns.distplot(female['Spending Score (1-100)'], bins=20, label='female', 
                  color='r')
ax = sns.distplot(male['Spending Score (1-100)'], bins=20, label='male')
ax.set_title('Spending Score Distribution by Gender', size=14)
format_spines(ax, right_border=False)
plt.legend()
plt.show()


# In[41]:


# Annual Income Distribution
fig, ax = plt.subplots(figsize=(10, 4), sharex=True)
female = df.loc[df['Gender'] == 'Female']
male = df.loc[df['Gender'] == 'Male']
ax = sns.distplot(female['Annual Income (k$)'], bins=20, label='female', 
                  color='r', hist=True)
ax = sns.distplot(male['Annual Income (k$)'], bins=20, label='male')
ax.set_title('Annual Income Distribution by Gender', size=14)
format_spines(ax, right_border=False)
plt.legend()
plt.show()


# **Conclusions:**
# The analysis shows there is low score concentration in male gender (between 0 and 25 score points). In female gender, we have high concentration in ranges between 75 and 100 compared to male gender. In general, women have higher Spending Score than men.
# 
# In other hand, the Annual Income distribution shows that in general, men have higher annunal income than women. These two analysi together could give good insights for mall administrators.

# In[42]:


# Configuration
sns.set(style='white', palette='muted', color_codes=True)
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.despine(left=True)

# Dataframe indexing
young = df.loc[df['Age Range'] == 'Young']
adult = df.loc[df['Age Range'] == 'Adult']
senior = df.loc[df['Age Range'] == 'Senior']
titles = ['Young', 'Adult', 'Senior']
age_range_dataframes = [young, adult, senior]

for idx in range(3):
    age_range = age_range_dataframes[idx]
    axs[idx] = sns.distplot(age_range[age_range['Gender']=='Male']['Spending Score (1-100)'], 
                          bins=20, ax=axs[idx], label='male', color='b', hist=False)
    axs[idx] = sns.distplot(age_range[age_range['Gender']=='Female']['Spending Score (1-100)'], 
                          bins=20, ax=axs[idx], label='female', color='r', hist=False)
    axs[idx].set_title(titles[idx], size=13)

fig.suptitle('Spending Score Distribution by Gender and Age Range')
plt.setp(axs, yticks=[])
plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()


# In[43]:


# Spending Score Distribution by Age Range
fig, ax = plt.subplots(figsize=(10, 4), sharex=True)
young = df.loc[df['Age Range'] == 'Young']
adult = df.loc[df['Age Range'] == 'Adult']
senior = df.loc[df['Age Range'] == 'Senior']
ax = sns.distplot(young['Spending Score (1-100)'], bins=10, label='Young', color='b')
ax = sns.distplot(adult['Spending Score (1-100)'], bins=10, label='Adult', color='g')
ax = sns.distplot(senior['Spending Score (1-100)'], bins=10, label='Senior', color='grey')
ax.set_title('Spending Score Distribution by Gender', size=14)
format_spines(ax, right_border=False)
plt.legend()
plt.show()


# **Conclusions:**
# * Senior Spending Scores concentrates in low and medium values;
# * In high score valuation, adults have the highest levels;
# * In gender comparison, young and senior women have higher Spending Score values than young and senior men.

# ## Training KMeans

# ### Age and Spending Score

# In[44]:


# Indexing dataframe
df_1 = df.loc[:, ['Age', 'Spending Score (1-100)']]

# Searching for optimun K
plot_elbow_method(df_1)


# Let's train our algorithm with 4 clusters.

# In[45]:


# Training KMeans
k_means = KMeans(n_clusters=4)
k_means.fit(df_1)
y_kmeans = k_means.predict(df_1)
centers = k_means.cluster_centers_
plot_kmeans(df_1, y_kmeans, centers)


# **Analysis:**
# Now we have 4 different clusters grouped by `Age` and `Spending Score`.
# 
# * **Group 1: Yellow** - Here we have customers with low score no matter the age. For this group, maybe the mall adminstrators have develop a different strategy including effective marketing action and a different approach to give a new perspective for these costumers. So the score could be raised for higher levels.
# * **Group 2: Purple** - This group identifies young and adult costumers (age < 40) with medium scores (between 35 and 75). To raise the Spending Score of this group, maybe would be good to create new actions for young public like games, customized products and others.
# * **Group 3: Orange** - These costumers have age greater than 40 and medium score. Similar to group 2, for raising the score of group 3 the mall administrators could improve actions for adult and senior public like calm places inside the mall, new restaurants, typical food, and others.
# * **Group 4: Dark Blue** - These are costumers with the highest Spending Score. The actions of mall administrators should be planned to maintain this group with high scores. The challenge is to understand what are the most important mall features for them and improve it.

# ### Annual Income and Spending Score

# In[46]:


# Dataframe indexing
df_2 = df.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']]

# Optimum K
plot_elbow_method(df_2)


# Let's run KMeans with 5 clusters.

# In[47]:


# Training
k_means = KMeans(n_clusters=5)
k_means.fit(df_2)
y_kmeans = k_means.predict(df_2)
centers = k_means.cluster_centers_
plot_kmeans(df_2, y_kmeans, centers)


# Here we have analysis between Annual Income and Spending Score. There is 5 differente group, each one with specific characteristics.
# 
# There are costumers with high, medium and low annual income and with high, medium and low spending score. For each one, the administrators could plan different actions.

# **I hope you enjoy this kernel! Please upvote!**
