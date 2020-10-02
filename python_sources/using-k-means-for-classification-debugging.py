#!/usr/bin/env python
# coding: utf-8

# # **In this notebook we will compare using an unsupervised learning algorithm (K-Means Clustering) and a simple supervised learning algorithm (Logistic Regression) for binary classification on the mushroom data set. To compare our algorithms, we will use scikit-learn's test_train_split() method.**

# In[ ]:


#Imports
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Read the original data.

# In[ ]:


orig = pd.read_csv('../input/mushrooms.csv')


# # (Don't uncomment this next statement until instructed to below.)

# In[ ]:


#Shuffles the orig DataFrame
#orig = orig.sample(frac=1)


# In[ ]:


orig.head()


# Now we will build our training data set so that the machine can work on it.

# We will use the 'class' attribute as our dependent variable (variable to be predicted) and all the other attributes as our independent variables (variables to be used for prediction).

# In[ ]:


X = orig.drop(['class'], axis=1)
y = orig['class']


# Before we encode each of our categorical variables in X & y with numbers so that algorithm can work with them, we will first take a look at the values contained within each of X's attributes..

# # Exploration

# In[ ]:


for attr in X.columns:
    print('\n*', attr, '*')
    print(X[attr].value_counts())


# Two things to note here: 

# First, the 'veil-type' variable has only one value, 'p'. And since there is only one possible value, it gives us little information - so we can drop this column.

# In[ ]:


X.drop(['veil-type'], axis=1, inplace=True)


# Second, the 'stalk-root' variable has a '?' value for it's missing values. Rather than impute this missing value, I will divide the dataset into two sections: where 'stalk-root'=='?' and where 'stalk-root'!='?', and analyze the distribution of each variable within those two data sets. 

# In[ ]:


for attr in X.columns:
    fig, ax =plt.subplots(1,2)
    sns.countplot(X[X['stalk-root']=='?'][attr], ax=ax[0]).set_title('stalk-root = ?')
    sns.countplot(X[X['stalk-root']!='?'][attr], ax=ax[1]).set_title('stalk-root != ?')
    fig.show()


# Since the distributions are very different among the variables (meaning that the '?' values may not be irrelevant), we will not impute the '?' values for 'stalk-root', rather, we will encode them for our learning algorithms.

# # Encoding:

# We'll use a binary encoding for variables that hold only 2 possible values, and a one-hot-encoding for variables that hold 3 or more possible values.

# In[ ]:


#For columns with only two values
for col in X.columns:
    if len(X[col].value_counts()) == 2:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])


# In[ ]:


X.head()


# And now we 'one-hot-encode' the rest of the variables:

# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


X.head()


# # Clustering: Problem begins here. I'd recommend not commenting anything out until you reach the end of the clustering section so that you can see the inital clustering result. Then after, read the block below for instructions to see how the bug arises. 

# We will cluster the data into two clusters, one will hold most of the 'poison' mushrooms and the other one will hold most of the 'edible' mushrooms.
# 
# Keep in mind that the dataset, 'X' does not specify which mushrooms are 'edible' or 'poisonous', and so the clustering algorithm is ignorant of this too!

# In[ ]:


#New
#train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.0)


# In[ ]:


#New (used to show train_X is indeed the same as X, albeit, shuffled)
#print(len(X))
#print(len(train_X), len(train_y))
#print(len(val_X), len(val_y))


# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=None)

#Old
kmeans.fit(X)

#New
#kmeans.fit(train_X)


# In[ ]:


#Old
clusters = kmeans.predict(X)

#New
#clusters = kmeans.predict(train_X)


# In[ ]:


clusters


# We create a DataFrame to show how each cluster holds different shares of the poisonous mushrooms.

# In[ ]:


cluster_df = pd.DataFrame()
cluster_df['cluster'] = clusters

#Old
cluster_df['class'] = y

#New
#cluster_df['class'] = train_y


# In[ ]:


sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count')


# # End of clustering
# 
# # As we can see from above, one cluster has clustered most of the' edible' mushrooms, while the other cluster has clustered most of the 'poisonous' mushrooms. Beautiful. 
# # Now, to see the bug, run again starting at the Clustering header, but un-comment the '#New' statements and comment-in the '#Old' statements in the code-blocks to see the bug. And after that read the block below this one.

# # And now both clusters hold a majority of edible mushrooms.... Weird, eh?
# 
# # Next, I'd recommend recommenting-out the '#New' statements, uncommenting the old statements, then re-running this kernel, but with the 'orig' data frame shuffled. I left a piece of commented code at the top of the kernel that does this, that you should comment out. 
# # So, to recap, you should re-run the kernel, but with the shuffle statement uncommented, the '#Old' statements uncommented, the '#New' statements commented-in, and observe the resulting factor plot for the clusters. After that, read the block below: 

# # And once again, both clusters have a majority of edible mushrooms...
# # The problem seems to have to do with shuffling the DataFrame, but even that seems strange given how nice the original clustering result was (with most of the edibles in one cluster, and most of the poison edibles in the other).
# # Let me know what you think!

# Disregard the below statements.

# From the above observation, we see that our clustering algorithm could potentially be used to classify mushrooms as either 'poisonous' or 'edible'.
# 
# For example, if we used our clustering model to place a mushroom in cluster 0, we could predict it to be edible, and if it placed the mushroom into cluster 1, we could predict it to be 'poisonous'. 
# 
# This is what we'll do in the next section, and, just for fun, we will compare it's performance to sklearn's LogisticRegression.

# # Classification

# Will get started here after clustering problem is figured out.
