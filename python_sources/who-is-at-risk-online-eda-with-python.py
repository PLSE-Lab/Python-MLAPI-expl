#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis with Python
# 
# A processed version of the Kaggle's [Risk of being drawn into online sex work](https://www.kaggle.com/panoskostakos/online-sex-work) dataset, obtained through [this notebook](https://www.kaggle.com/quannguyen135/preliminary-data-cleaning-with-python), is used.
# 
# This notebook explores the differences in a number of features/attributes between the group of low-risk users and that of high-risk users in the dataset through visualization. A graph visualization indicating the network of users who are registered as friends in the online forum is also included.

# ## Loading data

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gc

import warnings
warnings.filterwarnings('ignore')


# In[4]:


# for Kaggle, change to
df = pd.read_csv('../input/cleaned-online-sex-work/cleaned_online_sex_work.csv', index_col=0)
#df = pd.read_csv('../input/cleaned_online_sex_work.csv', index_col=0)
df = df.iloc[: 28831, :]
df = df[~ df.index.duplicated(keep='first')]
df.head()


# In[5]:


for column in df.columns:
    print(column)

df.shape


# In[6]:


train_df = df[df['Risk'].isnull() == False]
train_df['Risk'] = train_df['Risk'].astype(int)
norisk_df = train_df[train_df['Risk'] == 0]
risk_df = train_df[train_df['Risk'] != 0]

print(train_df.shape)


# ## Visualizations

# Here we are comparing low-risk and high-risk users in a number of features/attributes:

# ### Gender, age, location, and whether the user is verified

# In[7]:


f, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.countplot(x='Female', hue='Risk', data=train_df, ax=ax[0][0])

sns.distplot(norisk_df['Age'], kde_kws={'label': 'Low Risk'}, ax=ax[0][1])
sns.distplot(risk_df['Age'], kde_kws={'label': 'High Risk'}, ax=ax[0][1])

sns.countplot(x='Location', hue='Risk', data=train_df, ax=ax[1][0])

sns.countplot(x='Verification', hue='Risk', data=train_df, ax=ax[1][1])

plt.show()


# Some interpretation could be achieved from these visualizations:
# - The ratio of the number of low-risk users to that of high-risk users is high, as is the ratio of the number of males to the number of females. In other words, there are more low-risk than high-risk users, and more male than female users.
# - Most users are in their late 20s to mid 60s, no clear different age distributions between low-risk and high-risk users.
# - There are more low-risk than high-risk users in most registered locations, expect for location M and N.
# - Most users are not registered in the forum.

# ### Orientation

# In[8]:


orientation_df = train_df[['Heterosexual', 'Homosexual', 'bicurious', 'bisexual']].idxmax(1)
orientation_df = pd.concat([orientation_df, train_df['Risk']], axis=1).rename(columns={0: 'Orientation'})
orientation_df.head()


# In[9]:


sns.countplot(x='Orientation', hue='Risk', data=orientation_df)

plt.show()


# We again see that there are more low-risk users in most groups, except for `bicurious`.

# In[10]:


del orientation_df; gc.collect()


# ### Polarity

# In[11]:


polarity_df = train_df[['Dominant', 'Submisive', 'Switch']].idxmax(1)
polarity_df = pd.concat([polarity_df, train_df['Risk']], axis=1).rename(columns={0: 'Polarity'})
polarity_df.head()


# In[12]:


sns.countplot(x='Polarity', hue='Risk', data=polarity_df)

plt.show()


# We again see that there are more low-risk users in most groups, except for `Submissive`.

# In[13]:


del polarity_df; gc.collect()


# ### People the user is looking for

# In[14]:


looking_df = train_df[['Men', 'Men_and_Women', 'Nobody', 'Nobody_but_maybe', 'Women']].idxmax(1)
looking_df = pd.concat([looking_df, train_df['Risk']], axis=1).rename(columns={0: 'Looking_for'})
looking_df.head()


# In[15]:


sns.countplot(x='Looking_for', hue='Risk', data=looking_df)

plt.show()


# We again see that there are more low-risk users in most groups, except for `Nobody`.

# In[16]:


del looking_df; gc.collect()


# ### Points Rank

# In[17]:


f, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(norisk_df['Points_Rank'], kde_kws={'label': 'Low Risk'}, ax=ax[0])
sns.distplot(risk_df['Points_Rank'], kde_kws={'label': 'High Risk'}, ax=ax[1])

plt.show()


# ### Online activity statistics

# In[18]:


f, ax = plt.subplots(2, 2, figsize=(20, 10))

sns.distplot(norisk_df['Number_of_Comments_in_public_forum'], kde_kws={'label': 'Low Risk'}, ax=ax[0][0])
sns.distplot(risk_df['Number_of_Comments_in_public_forum'], kde_kws={'label': 'High Risk'}, ax=ax[0][0])

sns.distplot(norisk_df['Time_spent_chating_H:M'], kde_kws={'label': 'Low Risk'}, ax=ax[0][1])
sns.distplot(risk_df['Time_spent_chating_H:M'], kde_kws={'label': 'High Risk'}, ax=ax[0][1])

sns.distplot(norisk_df['Number_of_advertisments_posted'], kde_kws={'label': 'Low Risk'}, ax=ax[1][0])
sns.distplot(risk_df['Number_of_advertisments_posted'], kde_kws={'label': 'High Risk'}, ax=ax[1][0])

sns.distplot(norisk_df['Number_of_offline_meetings_attended'], kde_kws={'label': 'Low Risk'}, ax=ax[1][1])
sns.distplot(risk_df['Number_of_offline_meetings_attended'], kde_kws={'label': 'High Risk'}, ax=ax[1][1])

plt.show()


# ### Number of friends in the forum

# In[19]:


f, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(norisk_df['Number of Friends'], kde_kws={'label': 'Low Risk'}, ax=ax[0])
sns.distplot(risk_df['Number of Friends'], kde_kws={'label': 'High Risk'}, ax=ax[1])

plt.show()


# ### Network of friends

# In[20]:


import networkx as nx
import matplotlib.patches as mpatches


# In[21]:


network_df = train_df[train_df['Friends_ID_list'].isnull() == False]['Friends_ID_list']
network_df.head()


# In[22]:


graph = nx.Graph()
graph.add_nodes_from(list(network_df.index))


# In[23]:


for train_id in network_df.index:
    friend_ids = list(map(int, network_df.loc[train_id].split(',')))
    for friend_id in friend_ids:
        graph.add_edge(train_id, friend_id)


# In[24]:


f, ax = plt.subplots(1, 1, figsize=(20, 10))

pos = nx.spring_layout(graph)

nodes = nx.draw_networkx_nodes(
    graph,
    pos,
    node_color = 'y',
    node_size = 50
)
nodes.set_edgecolor('black')

norisk_nodelist = list(norisk_df[norisk_df['Friends_ID_list'].isnull() == False].index)
risk_nodelist = list(risk_df[risk_df['Friends_ID_list'].isnull() == False].index)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist = norisk_nodelist,
    node_color = 'b',
    node_size = 50
)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist = risk_nodelist,
    node_color = 'r',
    node_size = 50
)

labels = {}
for node in norisk_nodelist: labels[node] = node
for node in risk_nodelist: labels[node] = node
nx.draw_networkx_labels(graph, pos, labels, font_size=10)

nx.draw_networkx_edges(graph, pos, edge_color='y')

patches = [
    mpatches.Patch(color='y', label='Risk Undetermined'),
    mpatches.Patch(color='b', label='Low Risk'),
    mpatches.Patch(color='r', label='High Risk')
]
plt.legend(handles=patches)

plt.show()


# ## Feature importance
# 
# We will put our data through a machine learning model, Support Vector Machine, and from that we will obtain the coefficient for each column label. Since SVM creates a hyperplan that uses support vectors to maximize the distance between the two classification class.
# 
# So the absolute value of these coefficients can be used to determine feature importance of our column labels.

# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

X_train = train_df.drop(['Friends_ID_list', 'Risk'], axis=1)
y_train = train_df['Risk']


# In[26]:


X_train.head()


# In[27]:


y_train.head()


# ### Mean Encoding for `Location`

# In[28]:


try:
    clf = LinearSVC()
    clf.fit(X_train, y_train)
except Exception as e:
    print('There was a problem: %s' % str(e))


# This error indicates that since the data in the `Location` column is currently string, and cannot be processed by our machine learning model, Support Vector Machine in this case.
# 
# **Mean Encoding** is a way for us to convert these strings to numerical data, in which we replace each label with the average of the rows with that specific label. For example, while considering `Location A`, we take out the rows with `A` in their `Location` column, and compute the average of these rows in the `Risk` column. We then use this number to replace the string `A`.

# In[29]:


location_means = train_df.groupby('Location').mean()['Risk']
location_means


# In[30]:


train_df['Location'] = train_df['Location'].map(location_means)
train_df['Location'].head()


# In[31]:


X_train = train_df.drop(['Friends_ID_list', 'Risk'], axis=1)
try:
    clf = LinearSVC()
    clf.fit(X_train, y_train)
except Exception as e:
    print('There was a problem: %s' % str(e))


# Now we are able to feed our training data into a SVM model, we will extract some information from that model, after we have trained it with out data. Specifically, we are looking at coefficient of each feature.

# In[32]:


nfeatures = 10

coef = clf.coef_.ravel()
top_positive_coefs = np.argsort(coef)[-nfeatures :]
top_negative_coefs = np.argsort(coef)[: nfeatures]
top_coefs = np.hstack([top_negative_coefs, top_positive_coefs])

plt.figure(figsize=(15, 5))
colors = ['red' if c < 0 else 'blue' for c in coef[top_coefs]]
plt.bar(np.arange(2 * nfeatures), coef[top_coefs], color = colors)
feature_names = np.array(X_train.columns)
plt.xticks(np.arange(0, 1 + 2 * nfeatures), feature_names[top_coefs], rotation=60, ha='right')

plt.show()


# Here we see that no particular column label has a large absolute coefficient value, so we say that no feature is significantly important to determine the risk of a specific user. We however see that features like `Number of Friends`, `Member_since_day`, `Number of Comments in public forum`, etc. features representing online activeness all have negative coefficients, so we can guess that online activeness may decrease the probability of a user's risk.
# 
# Other observations might include:
# - The field `Female` has a near-zero coefficient - some can say gender plays no role in determining a user's online risk.
# - `Submissive`'s, though low, is the most positive coefficient - psychologists might find this telling.

# Here we see that no particular column label has a large absolute coefficient value, so we say that no feature is significantly important to determine the risk of a specific user. We however see that features like `Number of Friends`, `Member_since_day`, `Number of Comments in public forum`, etc. features representing online activeness all have negative coefficients, so we can guess that online activeness may decrease the probability of a user's risk.
# 
# Other observations might include:
# - The field `Female` has a near-zero coefficient - some can say gender plays no role in determining a user's online risk.
# - `Submissive`'s, though low, is the most positive coefficient - psychologists might find this telling.

# ## Feature correlation

# In[33]:


corr_matrix = train_df.drop(['Friends_ID_list'], axis=1).corr()

f, ax = plt.subplots(1, 1, figsize=(15, 10))
sns.heatmap(corr_matrix)

plt.show()


# Some observations could be made:
# - Different sexual orientaions have relatively strong negative correlation with one another, which is consistent as sexual orientations are mutually exclusive (an entry can only be one of them). Different polarity (`Dominant`, `Submissive`) also have a relatively strong negative correlation for the same reason.
# - (Gender) `Female` and (looking for) `Women` have a strong negative correlation, which suggests that most female users are heterosexual. This is consistent with what we saw earlier, that the percentage of homosexual, bicurious, and bisexual users are significantly lower than that of heterosexual users. Interestingly, (looking for) `Men` and `Homosexual` have a strong positive correlation, since while there aren't many homosexual users, we can conclude that most of them are male.
# - `Points_Rank` and `Number_of_Comments_in_public_forum` have a relatively strong correlation. This makes sense as some forums award points for each comment a user makes.

# 
