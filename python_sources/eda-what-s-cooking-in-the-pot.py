#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from os import path
import os
from PIL import Image
from collections import defaultdict
import networkx as nx
# import geopandas

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 1. Read and Load ```train.json``` and ```test.json```

# In[ ]:


with open("../input/train.json") as f:
  train = json.load(f)
    
with open("../input/test.json") as f:
  test = json.load(f)


# In[ ]:


pprint(train[0])                
pprint(train[0]["cuisine"])


# ## 2. store values from ```train.json``` and ```test.json``` in DataFrame

# In[ ]:


train_data = pd.DataFrame()
var_list = ["id", "ingredients", "cuisine"]
for v in var_list:
  train_data[v] = [train[i][v] for i in range(len(train))]
    
test_data = pd.DataFrame()
var_list = ["id", "ingredients"]
for v in var_list:
  test_data[v] = [test[i][v] for i in range(len(test))]    


# convert ```ingredients``` (list) to ```ingredients``` (string)

# In[ ]:


train_data["ingredients_list"] = train_data["ingredients"].apply(lambda x: ", ".join(x))
test_data["ingredients_list"] = test_data["ingredients"].apply(lambda x: ", ".join(x))


# ## 3. Feature Engineering
# ### 3.1 Create new variable: number of ingredients used

# In[ ]:


train_data["num_ingred"] = train_data["ingredients"].apply(lambda x: len(x))
test_data["num_ingred"] = test_data["ingredients"].apply(lambda x: len(x))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (15, 10)


# ## 4. Exploratory Data Analysis (Overall)
# ### 4.1 Histogram: distribution of number of ingredients

# In[ ]:


sns.countplot(train_data.num_ingred)


# In[ ]:


sns.countplot(test_data.num_ingred)


# In[ ]:


train_data.cuisine.value_counts()


# ### 4.2 Barplot: number of ingredients for each cuisine
# #### 4.2.1 Max number of ingredients

# In[ ]:


train_data.groupby("cuisine")["num_ingred"].max()


# In[ ]:


sns.barplot(x = "cuisine", y = "num_ingred", data = train_data, estimator = max)


# #### 4.2.2 Average number of ingredients

# In[ ]:


train_data.groupby("cuisine")["num_ingred"].mean()


# In[ ]:


sns.barplot(x = "cuisine", y = "num_ingred", data = train_data)  # default estimator is mean


# ### 4.3 Boxplots: value distribution of number of ingredients by cuisine 

# In[ ]:


sns.boxplot(x = "cuisine", y = "num_ingred", data = train_data)


# ### 4.4 Violinplot - similar to boxplot but show the shape of the distribution 

# In[ ]:


sns.violinplot(x = "num_ingred", y = "cuisine", data = train_data)


# ## 5. EDA for common ingredients (train data)

# ### 5.1 Tokenization 

# In[ ]:


X = train_data.ingredients_list

vect = CountVectorizer(tokenizer = lambda x: x.split(", "), lowercase = False)
X_dtm = vect.fit_transform(X)

# store document term matrix as sparse dataframe
sdf = pd.SparseDataFrame(X_dtm, columns = vect.get_feature_names(), default_fill_value = 0)

# append cuisine to sparse dataframe
sdf["cuisine"] = train_data["cuisine"]


# ### 5.2 sum of each ingredients for each cuisines
# use ```groupby``` to create aggregated values for each ingredients across cuisines, stored as new dataframe ```cuisine_ingred```

# In[ ]:


grouped = sdf.groupby("cuisine")
grouped.groups

cuisine_ingred = grouped.agg(np.sum)


# In[ ]:


cuisine_ingred


# #### 5.2.1 what is the most common ingredient for each cuisine?

# In[ ]:


most_comm_ingred = pd.DataFrame({"most_common_ingred": cuisine_ingred.idxmax(axis=1), "count": cuisine_ingred.max(axis=1)})
print(most_comm_ingred)                                                                                                      


# #### 5.2.2 Top 10 ingredients for each cuisine

# In[ ]:


top_10_ingred = pd.DataFrame(cuisine_ingred.columns[np.argsort(-cuisine_ingred.values, axis = 1)[:,:10]], 
                             index=cuisine_ingred.index)
top_10_ingred


# ### 5.3 Heatmap of ingredients 

# create dataframe of Top 50 common ingredients

# In[ ]:


top_50_ingred = pd.DataFrame(cuisine_ingred.columns[np.argsort(-cuisine_ingred.values, axis = 1)[:,:50]], 
                             index=cuisine_ingred.index)


# find common ingredients across all types of cuisine (ingredients which are available in at least 15 out of 20 cuisines)
# 

# In[ ]:


ding = defaultdict(int)
for i in range(0, len(top_50_ingred)):
  for j in range(0, 50):
    ding[top_50_ingred.iloc[i][j]] += 1

d = dict((k,v) for k, v in ding.items() if v >= 15)


# In[ ]:


# get values for these common items
common_ingred_list = list(d.keys())
# subset cuisine_ingred
common_ingreds = cuisine_ingred[common_ingred_list]
# scale common_ingreds by number of recipes
common_ingreds_scaled = common_ingreds.apply(lambda x: x / train_data.cuisine.value_counts())

# generate heatmap
sns.heatmap(common_ingreds_scaled, cmap="YlGnBu")


# ### 5.4 Wordcloud

# In[ ]:


plt.rcParams["figure.figsize"] = (12, 8)


# In[ ]:


def gen_wordcloud(subset, cols):
  ingred_sum = subset.tolist()              
  sum_dict = {cols[i]: ingred_sum[i] for i in range(0, len(subset))}
    
  wordcloud = WordCloud(background_color = "white", max_words = 100).fit_words(sum_dict)
  plt.imshow(wordcloud, interpolation = "bilinear")
  plt.axis("off")


# generate wordcloud for all types of cuisine combined

# In[ ]:


gen_wordcloud(cuisine_ingred.agg(np.sum), sdf.columns)


# In[ ]:


# generate wordclouds for different types of cuisine
for i, cuisine  in enumerate(cuisine_ingred.index):
  cuisine_subset = cuisine_ingred.iloc[i]
  fig, ax = plt.subplots()
  ax.set_title(cuisine)
  gen_wordcloud(cuisine_subset, sdf.columns)


# ### 5.5 Network Graph: Similarity of cuisines based on common ingredients

# In[ ]:


plt.rcParams["figure.figsize"] = (15, 10)


# In[ ]:


# create common ingredients count dataframe for common ingredients that are used in 5 different cuisines or more
d = dict((k,v) for k, v in ding.items() if v >= 5)
common_ingred_list = list(d.keys())
common_ingreds = cuisine_ingred[common_ingred_list]

common_ingred_T = common_ingreds.transpose()

common_ingred_T = common_ingred_T.rename_axis(None).rename_axis(None, axis=1)

corr = common_ingred_T.corr()

links = corr.stack().reset_index()
links.columns = ["c1", "c2", "corr_value"]

# choose cuisines with correlation > 0.6 and remove corr = 1.0
links_filtered = links.loc[ (links["corr_value"] > 0.5) & (links["c1"] != links["c2"]) ]

G = nx.from_pandas_edgelist(links_filtered, 'c1', 'c2')

nx.draw(G, with_labels = True, node_color = range(20), node_size = 1000, 
        edge_color = "black", width = 1, font_size = 10, cmap = "tab20")


# ### 5.6 Scatterplot: relationship between number of ingredients and number of recipes

# In[ ]:


# number of ingredients vs number of recipes
ratio_df = pd.DataFrame({"num_ingred": np.sum(cuisine_ingred, axis = 1), 
                         "num_recipes": train_data.cuisine.value_counts()})
ratio_df["total_ingreds"] = train_data.groupby("cuisine").num_ingred.sum()

p1 = sns.scatterplot(x = "num_recipes", y = "num_ingred", data = ratio_df, hue = ratio_df.index, alpha = 1, legend = False)
for line in range(0, ratio_df.shape[0]):
  p1.text(ratio_df.num_recipes[line]+0.2, ratio_df.num_ingred[line], ratio_df.index[line], horizontalalignment = "left", size = "small", color = "black") 
plt.xlabel("number of recipes")
plt.ylabel("number of ingredients")


# ## 6. Specialty Ingredients

# In[ ]:


specialty_bool = cuisine_ingred != 0
# sum across cuisines to get the number of cuisines which uses a particular ingredient
specialty_bool_sum = specialty_bool.sum(axis = 0)
# evaluate whether each ingredient is used only for one type of cuisine
specialty_ingred = pd.DataFrame({"special": specialty_bool_sum == 1}, index = None)
specialty_ingred


# There are 2595 ingredients that are used in only one type of cuisine

# In[ ]:


# subset ingredients that are only used in one type of cuisine
specialty = specialty_ingred[specialty_ingred.special == True]
print(len(specialty))


# In[ ]:


specialty_ingred_list = specialty.index
# get the number of specialty ingredients for each cuisine
specialty_bool_subset = specialty_bool[specialty_ingred_list]
specialty_bool_subset.sum(axis = 1)


# In[ ]:


specialty_df = cuisine_ingred[specialty_ingred_list]
cuisines = specialty_df.index
for cuisine in cuisines:
    print(cuisine + " cuisine's top 10 specialty ingredients")
    print(specialty_df.loc[cuisine].sort_values(ascending = False).head(10))
    print("\n")


# ## 7. EDA for ingredients (test data)
# 
# ### 7.1 Tokenization

# In[ ]:


X_test = test_data.ingredients_list

vect = CountVectorizer(tokenizer = lambda x: x.split(", "), lowercase = False)
X_test_dtm = vect.fit_transform(X_test)

# store document term matrix as sparse dataframe
sdf_test = pd.SparseDataFrame(X_test_dtm, columns = vect.get_feature_names(), default_fill_value = 0)


# ### 7.2 calculate the number of times each ingredients is used in test data

# In[ ]:


ingred_test_sum = sdf_test.sum(axis = 0)
ingred_test_sum


# top 10 ingredients (pretty similar to train data)

# In[ ]:


ingred_test_sum.sort_values(ascending = False)[:10]


# ### 7.3 generate wordcloud - top 100 ingredients

# In[ ]:


gen_wordcloud(ingred_test_sum, sdf_test.columns)

