#!/usr/bin/env python
# coding: utf-8

# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRPuCl72q5-misMtB7Lvit9uodiEhjGtfZmm3Sb5y7wC26X2XhP" width="450px" height="450px"/> 

# <img src="https://pbs.twimg.com/profile_images/679147586746322946/RQ78ao4T_400x400.png" width="50px" height="50px"/>

# <img src="https://qph.fs.quoracdn.net/main-qimg-7c9b7670c90b286160a88cb599d1b733" width="450px" height="450px"/>

# # K-Means Clustering vs. Logistic Regression

# ## Contents
# 1. [Introduction:](#1)
# 1. [Imports:](#2)
# 1. [Read the Data:](#3)
# 1. [Exploration:](#4)
# 1. [Encoding:](#5)
# 1. [Cluster Analysis:](#6)
# 1. [Classification:](#7)
# 1. [Model Evaluation:](#8)
# 1. [Verdict](#9)
# 1. [Closing Remarks:](#10)

# <a id="1"> 
# ## 1. Introduction:

# In this notebook, we will be comparing two very different machine learning models on the [Mushroom Classification Dataset](https://www.kaggle.com/uciml/mushroom-classification) for the task of predicting whether a given mushroom is **<font color="purple">poisonous</font>** or **<font color="green">edible</font>**. 
# 
# The first model will be [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) without any parameter tuning, and the second model will be [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering).

# But how exactly does one go about comparing a [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning) Algorithm to an [Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning) Algorithm (especially when it comes to the task of [binary classification](https://en.wikipedia.org/wiki/Binary_classification))? 
# 
# As usual, for the Supervised Learning Algorithm (Logistic Regression), we will simply train the model on 80% of the mushroom data, and then test it's performance on the remaining 20%.
# 
# And as for the Unsupervised Learning Algorithm (K-Means Clustering): I've found that if we cluster the data (with it's labels removed) into two different clusters, then one cluster ends up holding most of the **<font color="purple">poisonous</font>** mushrooms, while the other cluster ends up holding most of the **<font color="green">edible</font>** mushrooms. So, to build our binary classifier, we will cluster that same 80% of mushroom data mentioned above into two clusters, and then classify the remaining 20% of mushrooms as **<font color="purple">poisonous</font>** or **<font color="green">edible</font>** depending on which cluster they belong to.
# 
# And at the end, we will compare the performances of the two algorithms on that test data to see who comes out on top!

# Which model do you think will be the winner?

# <a id="2">
# ## 2. Imports:

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# <a id="3">
# ## 3. Read the data:

# In[ ]:


orig = pd.read_csv('../input/mushrooms.csv')


# In[ ]:


orig.head()


# Divide our data into '*predictors*,' `X` and '*labels*,' `y`:

# In[ ]:


#The 'class' column contains our labels.
#It tells us whether the mushroom is 'poisonous' or 'edible'.
X = orig.drop(['class'], axis=1)
y = orig['class']


# Now before we encode each of our categorical variables in `X` & `y` with numbers so that the our learning algorithms can work with them, we will first do a bit of exploration.

# <a id="4">
# ## 4. Exploration:

# Let's take a look at the values contained within each of `X`'s attributes, so we can get a better picture of the data we're working with:

# In[ ]:


for attr in X.columns:
    print('\n*', attr, '*')
    print(X[attr].value_counts())


# Two things to note here: 
# 
# First, the `veil-type` variable has only one value, '**p**', meaning, every mushroom has the same `veil-type`. And because, every mushrrom has that same `veil-type`: that column doesn't tell us anything useful - so we can drop that column.

# In[ ]:


X.drop(['veil-type'], axis=1, inplace=True)


# Second, the `stalk-root` variable has a '**?**' value for it's missing values. Rather than impute this missing value, I will divide the dataset into two sections: **(1)** *where `X['stalk-root']==?`* and **(2)** *where `X['stalk-root']!=?`*. Then, I will analyze the distribution of each variable within those two data sets to determine if they are similar.
# 
# I'm no mushroom expert, so I would expect that if the distributions vary greatly for each variable, then the fact that the `stalk-root`'s are missing for some of the mushrooms -- *and not missing for the others* --may turn out to be useful/relevant information.

# In[ ]:


for attr in X.columns:
    #Format subplots
    fig, ax = plt.subplots(1,2)
    plt.subplots_adjust(right=2)
    
    #Construct values to count in each column
    a=set(X[X['stalk-root']=='?'][attr])
    b=set(X[X['stalk-root']!='?'][attr])
    c = a.union(b)
    c = np.sort(np.array(list(c)))
    
    #Build each subplot
    sns.countplot(x=X[X['stalk-root']=='?'][attr], order=c, ax=ax[0]).set_title('stalk-root == ?')
    sns.countplot(x=X[X['stalk-root']!='?'][attr], order=c, ax=ax[1]).set_title('stalk-root != ?')
    
    #Plot the plots
    fig.show()


# Since many of the distributions vary greatly, and because

# In[ ]:


print( (len(X[X['stalk-root']=='?']) / len(X))*100, '%', sep='') 


# of the mushrooms have the value **'?'** for their `stalk-root`, we will not impute the **'?'** values, rather, we will encode them just as we would the rest of the values in that column.

# <a id="5">
# ## 5. Encoding:

# We'll use a simple binary encoding for variables that hold only 2 possible values, and a [one-hot-encoding](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding) for variables that hold 3 or more possible values.

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


# <a id="6">
# ## 6. Cluster Analysis:

# Now before we get into building our models and testing them against eachother, I just wanted to show you, visually, the result that arises from clustering the mushroom data set.

# In[ ]:


#Initialize the model
kmeans = KMeans(n_clusters=2)


# In[ ]:


#Fit our model on the X dataset
kmeans.fit(X)


# In[ ]:


#Calculate which mushrooms fall into which clusters
clusters = kmeans.predict(X)


# In[ ]:


#'cluster_df' will be used as a DataFrame
#to assist in the visualization
cluster_df = pd.DataFrame()

cluster_df['cluster'] = clusters
cluster_df['class'] = y


# Now let's visualize the distribution of **<font color="purple">poisonous</font>** vs. **<font color="green">edible</font>** mushrooms in each cluster.

# In[ ]:


sns.factorplot(col='cluster', y=None, x='class', data=cluster_df, kind='count', order=['p','e'], palette=(["#7d069b","#069b15"]))


# Pretty interesting, eh? One cluster mostly contains **<font color="green">edible</font>** mushrooms and the other cluster mostly contains **<font color="purple">poisonous</font>** mushrooms.
# 
# So if we were given a mushroom, and we'd like to predict whether it is **<font color="green">edible</font>** or **<font color="purple">poisonous</font>**, we could first figure out which cluster it belongs to and then make our prediction based off of the percentage of **<font color="purple">poisonous</font>** vs. **<font color="green">edible</font>** mushrooms in that cluster.

# Okay! Now that you have seen how clustering can be used as a classifier for this problem, it's time to get into our testing.

# <a id="7">
# ## 7. Classification:

# But first, we need to encode our `y`-labels numerically so that our model can work with it.
# 
# Since each mushroom is either **<font color="purple">poisonous</font>** or **<font color="green">edible</font>**, we will use another simple binary encoding like before:

# In[ ]:


le = LabelEncoder()
y = le.fit_transform(y)

y


# So, when `y==1`, the mushroom is **<font color="purple">poisonous</font>**, and when `y==0`, the mushroom is **<font color="green">edible</font>**.

# ### Generate 'training' and 'test' sets:

# We will use sklearn's [`train_test_split()`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to generate our 'training' and 'test' sets

# In[ ]:


#Our training set will hold 80% of the data
#and the test set will hold 20% of the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20)


# ### Initialize our models:

# In[ ]:


#K-Means Clustering with two clusters
kmeans = KMeans(n_clusters=2)

#Logistic Regression with no special parameters
logreg = LogisticRegression()


# ### Fit our models:

# In[ ]:


kmeans.fit(train_X)#Note that kmeans is unlabeled...

logreg.fit(train_X, train_y)#... while logreg IS labeled


# ### Make our predictions on the 'test' set:

# In[ ]:


kmeans_pred = kmeans.predict(test_X)

logreg_pred = logreg.predict(test_X)


# ### A little thing about clustering:

# One interesting aspect of K-Means clustering is that it does not always give the same results.
# 
# For example, if we were to run `kmeans.fit(train_X)` multiple times: part of the times, the majority of the **<font color="purple">poisonous</font>** mushrooms will fall into *cluster 0*, and the majority of the **<font color="green">edible</font>** mushrooms will fall into *cluster 1* - and on the other times: vice versa! In fact, if you'd like to observe this phenomenon yourself: fork this kernel and run the code blocks under the [Cluster Analysis](#6) section a few times.
# 
# In order to get around this problem, we will build a second set of predictions from our K-Means model - `kmeans_pred_2`. This second set of predictions will simply be the [bit-wise complement](https://en.wikipedia.org/wiki/Bitwise_operation#NOT) of `kmeans_pred`, and we will use whichever set of predictions gives us a better score as our final prediction set for the K-Means model!

# In[ ]:


kmeans_pred_2 = []
for x in kmeans_pred:
    if x == 1:
        kmeans_pred_2.append(0)
    elif x == 0:
        kmeans_pred_2.append(1)
        
kmeans_pred_2 = np.array(kmeans_pred_2)


# Now, we'll figure out which set of predictions for K-Means we'd like to use.
# 
# We'll use scikit-learn's [`accuracy_score()`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) function to help us decide:

# In[ ]:


if accuracy_score(kmeans_pred, test_y, normalize=False) < accuracy_score(kmeans_pred_2, test_y, normalize=False):
    kmeans_pred = kmeans_pred_2


# <a id="8">
# ## 8. Model Evaluation:

# Now for the payoff: let's see which model performed better on the test-set!

# We will determine which model did better by visualizing the amount of predictions that were made correctly vs.  made incorrectly by each model.
# And to help us visualize the results, we will build a pandas DataFrame, that we can use to help us make the plot with seaborn.

# In[ ]:


#This DataFrame will allow us to visualize our results.
result_df = pd.DataFrame()

#The column containing the correct class for each mushroom in the test set, 'test_y'.
result_df['test_y'] = np.array(test_y) #(don't wanna make that mistake again!)

#The predictions made by K-Means on the test set, 'test_X'.
result_df['kmeans_pred'] = kmeans_pred
#The column below will tell us whether each prediction made by our K-Means model was correct.
result_df['kmeans_correct'] = result_df['kmeans_pred'] == result_df['test_y']

#The predictions made by Logistic Regression on the test set, 'test_X'.
result_df['logreg_pred'] = logreg_pred
#The column below will tell us whether each prediction made by our Logistic Regression model was correct.
result_df['logreg_correct'] = result_df['logreg_pred'] == result_df['test_y']


# In[ ]:


fig, ax = plt.subplots(1,2)
plt.subplots_adjust(right=2)
sns.countplot(x=result_df['kmeans_correct'], order=[True,False], ax=ax[0]).set_title('K-Means Clustering')
sns.countplot(x=result_df['logreg_correct'], order=[True,False], ax=ax[1]).set_title('Logistic Regression')
fig.show()


# <a id="9">
# ## 9. Verdict:

# Judging from the plots above, I'd say that **Logistic Regression** is the clear winner!
# 
# But K-Means Clustering didn't do too bad either; especially when you consider the fact that it's not built for the task of supervised learning, like Logistic Regression is!

# <a id="10">
# ## 10. Closing Remarks:

# I wasn't sure how this exercsie would turn out at first, but I'm glad to have gone through with it, because it was a ton of fun and I learned a bunch in the process!
# 
# If you've got any feedback for me: please leave a comment below, as I'd love to hear what you've got to say.
# And if you found this kernel to be interesting or useful to you, please consider giving it an upvote - I'd appreciate it very much :)
# 
# Also, special thanks to [Konstantin](https://www.kaggle.com/konstantinmasich) for helping me resolve a very difficult problem I had during the development of this kernel.

# Cheers!
# -*Josh*
