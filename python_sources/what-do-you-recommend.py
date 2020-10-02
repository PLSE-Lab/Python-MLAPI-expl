#!/usr/bin/env python
# coding: utf-8

# # Recommendation Engine
# ----
# Often termed as Recommender Systems, they are simple algorithms which aim to provide the most relevant and accurate items to the user by filtering useful stuff from of a huge pool of information base. Recommendation engines discovers data patterns in the data set by learning consumers choices and produces the outcomes that co-relates to their needs and interests.
# 
# ![Recommendation system](https://www.azoft.com/wp-content/uploads/2017/12/operation-principle-of-the-recommendation-system.png)
# 
# Trying to create a recommendaton system based on a minimal dataset.
# 
# The aim is to recommend similar products based on the user's current selection.
# 
# Give it a look and any inputs will be highly appreciated.

# # Loading the libraries

# In[ ]:


# Loading libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data exploration
# ----
# **File name**: ratigs_Beauty.csv
# 
# **Size**: 82.4 MB

# In[ ]:


# Load the data

df = pd.read_csv('../input/ratings_Beauty.csv')

print("Shape: %s" % str(df.shape))
print("Column names: %s" % str(df.columns))

df.head()


# In[ ]:


# Unique Users and Products

print("Unique UserID count: %s" % str(df.UserId.nunique()))
print("Unique ProductID count: %s" % str(df.ProductId.nunique()))


# In[ ]:


# Rating frequency

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set_style('whitegrid')
ax = sns.countplot(x='Rating', data=df, palette=sns.color_palette('Greys'))
ax.set(xlabel='Rating', ylabel='Count')
plt.show()


# # Data wrangling
# ----
# Creating **fields** and **measures** form the existing data
# 
# This helps generate more **data points** and validates the idealogy

# In[ ]:


# Mean rating for each Product

product_rating = df.groupby('ProductId')['Rating'].mean()
product_rating.head()


# In[ ]:


# Mean rating KDE distribution

ax = sns.kdeplot(product_rating, shade=True, color='grey')
plt.show()


# We can notice a large spike in the mean rating at value 5. This is a valuable indicator that points to the skewness of the data. Hence we need to further analyse this issue.

# In[ ]:


# Count of the number of ratings per Product

product_rating_count = df.groupby('ProductId')['Rating'].count()
product_rating_count.head()


# In[ ]:


# Number of ratings per product KDE distribution

ax = sns.kdeplot(product_rating_count, shade=True, color='grey')
plt.show()


# This graphs confirms the expectation that most items have around 50 - 100 ratings. We do have a bunch of outliers that have only a single rating and few Products have over 2000 ratings. 

# In[ ]:


# Un-Reliability factor

unreliability = df.groupby('ProductId')['Rating'].std(ddof = -1)
unreliability.head()


# In[ ]:


# Un-Reliability factor KDE distribution

ax = sns.kdeplot(unreliability, shade=True, color='grey')
plt.show()


# The plot show that a large portion of the products are highly reliable. For this unreliabilit factor we used standard devaiation. But we noticed above that a large porition of the Products have a single review. These items have varying ratings but high reliability. This issue needs to tbe addressed.

# # Data transforming
# ----
# Creating a final collection of all the various measures and features for each product

# In[ ]:


# Data frame with calculated fields and measures

unique_products_list = df.ProductId.unique()
data_model = pd.DataFrame({'Rating': product_rating[unique_products_list],                           'Count': product_rating_count[unique_products_list],                           'Unreliability': unreliability[unique_products_list]})
data_model.head()


# Let's explore the data model

# In[ ]:


print("Data model shape (number of data points): %s" % str(data_model.shape))


# In[ ]:


# Rating versus count

sns.set_style('ticks')
plt.figure(num=None, figsize=(11.7, 8.27), dpi=100, facecolor='w', edgecolor='k')

ax = data_model.plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.1)
plt.show()


# This plot fails to provide much information due to the large number of data points leading to clustered data. So let's break it down into a number of ranges

# In[ ]:


# Less than 100 ratings

ax = data_model[data_model.Count < 101].plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.1)
plt.show()


# In[ ]:


# 100 to 200 ratings

ax = data_model[data_model.Count > 100][data_model.Count<201].plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.4)
plt.show()


# In[ ]:


# 200 to 500 ratings

ax = data_model[data_model.Count > 200][data_model.Count<501].plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.4)
plt.show()


# We notice that the density becomes sparse as the number of ratings (count) increases. Let's have a look if unreliability has any corelation with the count of ratings and mean rating of the Product.

# In[ ]:


# Adding unreliability factor to the above plots 100 to 200 ratings

ax = data_model[data_model.Count > 100][data_model.Count<201].plot(kind='scatter', x='Unreliability', y='Count', c='Rating', cmap='jet', alpha=0.6)
plt.show()


# In[ ]:


# Addding unreliability factor to the above plots 200 to 500 ratings

ax = data_model[data_model.Count > 200][data_model.Count<501].plot(kind='scatter', x='Unreliability', y='Count', c='Rating', cmap='jet', alpha=0.6)
plt.show()


# Wow! Here we see a trend. It looks like the which have a high unreliability score, seem to have a lower rating over a significant count range. Let's see if there is an corelation between these factors.

# In[ ]:


# Coefficient of corelation between Unreliability and Rating

coeff_corelation = np.corrcoef(x=data_model.Unreliability, y=data_model.Rating)
print("Coefficient of corelation: ")
print(coeff_corelation)


# We notice that there is medium-strong negative corelation from the -0.26862181 coefficient. This means that as the unreliability factor increases, there is a medium-strong change that the rating of the product decreases. This is a good indicator as it clarifies any questions regarding unreliability.

# # Data modelling
# ----
# Let's see if we are ready to make prediction. If not we must model the data into an appropriate format.

# In[ ]:


# Summarise Count

print(data_model.Count.describe())


# In[ ]:


# Summarise Rating

print(data_model.Rating.describe())


# In[ ]:


# Summarise Unreliability

print(data_model.Unreliability.describe())


# It's clear that the count ranges form 1 to 7533 ratings, the Mean rating ranges from 1 to 5 and the Unrelaibility factor ranges form 0 to 1.92. These values cannot be use directly as they have a vastly varying range.

# In[ ]:


# Removing outliers and improbable data points

data_model = data_model[data_model.Count > 50][data_model.Count < 1001].copy()
print(data_model.shape)


# In[ ]:


# Normalization function to range 0 - 10

def normalize(values):
    mn = values.min()
    mx = values.max()
    return(10.0/(mx - mn) * (values - mx)+10)
    


# In[ ]:


data_model_norm = normalize(data_model)
data_model_norm.head()


# # Recommendation
# ----
# Once we have modelled the data, we recomending similar items based on Count of ratings, Mean rating and the Unreliability factor

# In[ ]:


# Setting up the model

# Recommend 20 similar items
engine = KNeighborsClassifier(n_neighbors=20)

# Training data points
data_points = data_model_norm[['Count', 'Rating', 'Unreliability']].values

#Training labels
labels = data_model_norm.index.values

print("Data points: ")
print(data_points)
print("Labels: ")
print(labels)

engine.fit(data_points, labels)


# Now that the engine is setup and we have initialized it with the required data points and labels, we can use it to recommend a list of 20 similar items

# In[ ]:


# Enter product ID to get a list of 20 recommended items

# User entered value
product_id = 'B00L5JHZJO'

product_data = [data_model_norm.loc[product_id][['Count', 'Rating', 'Unreliability']].values]

recommended_products = engine.kneighbors(X=product_data, n_neighbors=20, return_distance=False)

# List of product IDs form the indexes

products_list = []

for each in recommended_products:
    products_list.append(data_model_norm.iloc[each].index)

print("Recommended products: ")
print(products_list)

# Showing recommended products

ax = data_model_norm.plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.20)
data_model_norm.iloc[recommended_products[0]].plot(kind='scatter', x='Rating', y='Count',                                                   color='orange', alpha=0.5, ax=ax)

ax2 = data_model_norm.plot(kind='scatter', x='Rating', y='Unreliability', color='grey')
data_model_norm.iloc[recommended_products[0]].plot(kind='scatter', x='Rating', y='Unreliability',                                                   color='orange', alpha=0.5, ax=ax2)
plt.show()


# # Conclusion
# ----
# The engine recommends similar products based on feature such as number of ratings, mean ratings and unreliability factor of the Product. As seen from the above output, we can alter the number of items recommended, and using this we can integrate onine sale trends into retails stores by recommending similar products to the store.
# This also can be used as an added feature as a plus point when discussing item sales and profits with the stores.
