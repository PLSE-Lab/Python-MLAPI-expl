#!/usr/bin/env python
# coding: utf-8

# #  Intro to Machine Learning
# This is a free course offered by [fast.ai](http://www.fast.ai/) (currently [unlisted](http://forums.fast.ai/t/another-treat-early-access-to-intro-to-machine-learning-videos/6826)). There's a github [repository](https://github.com/fastai/fastai/tree/master/courses/ml1).
# 
# ## About this course
# Some machine learning courses can leave you confused by the enormous range of techniques shown and can make it difficult to have a practical understanding of how to apply them.
# 
# The good news is that modern machine learning can be distilled down to a couple of key techniques that are of very wide applicability. Recent studies have shown that the vast majority of datasets can be best modeled with just two methods:
# 
# - **Ensembles of decision trees** (i.e. Random Forests and Gradient Boosting Machines), mainly for **structured data** (such as you might find in a database table at most companies)
# - **Multi-layered neural networks learnt with Stochastic Gradient Descent** (SGD) (i.e. shallow and/or deep learning), mainly for **unstructured data** (such as audio, vision, and natural language)
# 
# ### The lessons
# In this course we'll be learning about:
# - **Random Forests** 
# - **Stochastic Gradient Descent**.
# - **Gradient Boosting** 
# - **Deep Learning**
# 
# ### The dataset
# We will be teaching the course using the [Blue Book for Bulldozers Kaggle Competition](https://www.kaggle.com/c/bluebook-for-bulldozers): 
# - "The goal of the contest is to predict the sale price of a particular piece of heavy equiment at auction based on it's usage, equipment type, and configuration. The data is sourced from auction result postings and includes information on usage and equipment configurations."
# 
# ### Note:
# These are personal notes. For the original code, check the github repository of the course. Also, I will be importing things as I need them.
# 

# # Lecture 3
# It is recommended to [watch the lecture first](https://youtu.be/YSFG_W8JxBo?), then follow the notebook.
# - The lesson material starts at [51:07](https://youtu.be/YSFG_W8JxBo?t=51m7s) because the time before that is spent talking about [another kaggle competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) and [how to deal with really large datasets](https://www.kaggle.com/jagangupta/memory-optimization-and-eda-on-entire-dataset).

# # Random Forest Model interpretation

# ## Load the data
# We'll be loading our feather file from the [last lesson](https://www.kaggle.com/ailobe/fastai-ml1-lesson1-rf/output).
# - feather-format is really **fast**.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# times the whole cell\n\n# import pandas\nimport pandas as pd\n\n# set the path to read df_raw\npath = '../input/df_raw'\n\n# read the data into a pandas DataFrame\ndf_raw = pd.read_feather(path)")


# To know more about the following pieces of code, check the [first lesson](https://www.kaggle.com/ailobe/fastai-ml1-lesson1-rf).
# - We'll use a method from the fastai library , **`proc_df`**,  to get the dataset ready for the random forest.
# - We'll create the function **`split_vals`** to split the dataset into training and validation sets.
# - We'll create the function **`print_scores`** to evaluate our model.

# In[ ]:


# import proc_df and the functions it depends on
from fastai.structured import numericalize, fix_missing, proc_df

X, y, nas_dict = proc_df(df_raw, 'SalePrice')


# In[ ]:


# create a function for splitting X and y into train and test sets of customizable sizes
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

# validation set size: 12000.
validation = 12000

# split point: length of dataset minus validation set size.
split_point = len(X)-validation

# split X
X_train, X_valid = split_vals(X, split_point)

# split y
y_train, y_valid = split_vals(y, split_point)

# dimensions (row, columns) of X_train, y_train and X_valid
X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


# import numpy
import numpy as np

# create a function that takes the RMSE
def rmse(pred,known): return np.sqrt(((pred-known)**2).mean())

# create a function that rounds to 5 decimal places (like kaggle leaderboard)
def rounded(value): return np.round(value, 5)

# create a function that prints a list of 4 scores, rounded:
# [RMSE of X_train, RMSE of X_valid, R Squared of X_train, R Squared of X_valid]
def print_scores(model):
    RMSE_train = rmse(model.predict(X_train), y_train)
    RMSE_valid = rmse(model.predict(X_valid), y_valid)
    R2_train = model.score(X_train, y_train)
    R2_valid = model.score(X_valid, y_valid)
    scores = [rounded(RMSE_train), rounded(RMSE_valid), rounded(R2_train), rounded(R2_valid)]
    if hasattr(m, 'oob_score_'): scores.append(m.oob_score_) # appends OOB score (if any) to the list 
    print(scores)


# In[ ]:


# print 5 first rows
df_raw.head()


# ## Subsampling
# When we are **interpreting our model**, we are not striving for prediction accuracy but for insights into the data.
# - We want a model that indicates the nature of the **relationships betwen features**.
# - We want it to be **reliable enough** so we can trust our interpetratons of its results.
# - There's **no need** to use the **full dataset** on each tree if we just want to get an accurate enough random forest.
# - Using a **subset** will be faster and will allow us to do more iterations.
# - We should use a **big enough subsample** to get similar results each time we run the same analysis.
# 

# In[ ]:


# import set_rf_samples
from fastai.structured import set_rf_samples

# set random subsample size to 50,000 rows
set_rf_samples(50000)


# In[ ]:


# import the class
from sklearn.ensemble import RandomForestRegressor

# instantiate the model with the parameters of the last lesson
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True,
                          random_state=17190)

# fit the model with data and calculate the running time
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')

# [RMSE of X_train, RMSE of X_valid, R Squared of X_train, R Squared of X_valid, OOB score]
print_scores(m)


# ## Confidence based on tree variance
# To give us a sense of **relative confidence** of predictions, we can use the **standard deviation** of the predictions.
# - The standard deviation is a measure that quantifies the **amount of variation** of a set of values ([source](https://en.wikipedia.org/wiki/Standard_deviation)).
#  - A **low** standard deviation indicates that the data points tend to be **close to the mean**.
#  - A **high** standard deviation indicates that the data points are spread out over a **wider range of values**. 
# - We want to be more **cautious** with rows that have a higher standard deviation because they make our model give **inconsistent predictions**.
# 
# We'll create a function to store **the predictions for each individual tree**.

# In[ ]:


# use a list comprehension to loop through the random forest and concatenates the predictions of each individual tree on a new axis
preds = np.stack([tree.predict(X_valid) for tree in m.estimators_])

# dimensions (rows, columns)
preds.shape


# There are **40** sets of predictions (**trees**) with **12,000** values (**predictions**), which corresponds to the size of the validation set.
# 
# Now we can **calculate the mean and the standard deviation of the predictions**, and more interestingly, we can **add that information to the dataset**.
# - We'll use the **df_raw**, because after `proc_df`, the dataset is all numbers and we want to be able to interpret its values.
# - We need to **split it again** because we are only interested in the rows of the **validation set**.
# - We'll **add two new columns** to the dataset: the **predictions** and the **standard deviation** of the predictions.

# In[ ]:


# split df_raw and conserve the part of the validation set
_, raw_valid = split_vals(df_raw, split_point)

# make a copy
validation = raw_valid.copy()

# add new column: calculated standard deviation over row axis
validation['pred_std'] = np.std(preds, axis=0)

# add new column: calculated mean over row axis
validation['pred'] = np.mean(preds, axis=0)


# Now we can use that information, but to **what columns** do we have **to pay attention**?
# - We could try with the features that got selected in our **single small deterministic tree** from the [last lesson](https://www.kaggle.com/ailobe/fastai-ml1-lesson1-rf):
#  - ProductSize, YearMade, fiSecondaryDesc, Hydraulics_Flow and ModelID.
# 
# Considering we are dealing with pieces of **heavy equipment**,  it's probably safe to assume that the **year** of **manufacture** and the **size** of the **machine** might be the most important features out of the selected in deciding the prize.
# - So let's take a look at YearMade and ProductSize.

# In[ ]:


# plots the counts of the unique values of YearMade in the validation set
validation.YearMade.value_counts(dropna=False).plot.bar(figsize=(15,4))


# In[ ]:


# list of selected columns
columns = ['YearMade', 'SalePrice', 'pred', 'pred_std']

# dataframe of selected columns with the rows grouped by the values in YearMade [index 0]
# and with the calculated mean of 'SalePrice', 'pred', 'pred_std'
year = validation[columns].groupby(columns[0]).mean()

# dimensions (rows, columns)
print(year.shape)

# 10 first rows sorted descendingly
year.sort_values(by=['pred_std'],ascending=False).head(10)


# Out of 55 possible values, these are the YearMade values that have the rows with more standard deviation in the prediction (so are less accurate).
# - Except for YearMade 1000, all the other rows have very **little unique value counts**.

# In[ ]:


# plots the counts of the unique values of ProductSize in the validation set
validation.ProductSize.value_counts(dropna=False).plot.barh()


# In[ ]:


# list of selected columns
columns = ['ProductSize', 'SalePrice', 'pred', 'pred_std']

# dataframe of selected columns with the rows grouped by the values in ProductSize [index 0]
# and with the calculated mean of 'SalePrice', 'pred', 'pred_std'
size = validation[columns].groupby(columns[0]).mean()

# calculates the ratio between mean standard deviation and mean prediction
(size.pred_std/size.pred).sort_values(ascending=False)


# The ProductSize values that have the rows with more standard deviation in the prediction are: Large, Compact and Small, and they are also the smallest groups in the validation set.
# - In general, the random forest does a less good job predicting **small groups of data**.
# 
# ## Feature importance
# Feature importance tells us **which columns affect the predictions  the most**.
# -  We'll use a method from the **fastai library** called **`rf_feat_importance`** that uses the `feature_importances_` attribute from the [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) to return a dataframe with the columns and their importance in descending order.
# - It's an **easy way** to know which features are the most important. There are [other methods](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) to do it.

# In[ ]:


# import rf_feat_importance
from fastai.structured import rf_feat_importance

# create a dataframe of the feature importance by passing the model and the training set
fi = rf_feat_importance(m, X_train)

# dimension (rows, columns)
print(fi.shape)

# first 5 rows
fi.head()


# In[ ]:


# plot the feature importance of the column names in 'cols'
fi.plot('cols', 'imp', 'bar', figsize=(15,4))


# As one would expect, not all the features are equally important.
# - There are a **handful** of columns that are really **informative**, while most of them are not.
# 
# We want to **concentrate efforts** on the features that matter the most, so we might want to dispense with the least informative columns and **include** only the **most informative** ones in the **training set**.
# - The number of columns to include/exclude will depend on the **prediction performance**.
# - We want to make **just as good** a model than the model with all the columns, **but** hopefully **simpler** so it generalizes a little better.
# - If the model **gets worse**, we have **exluded too many** columns that weren't redundant after all.
# - We are going to **filter the columns** by importance value.

# In[ ]:


# create a Series with the column names in 'cols' with a greater 'imp' value than 0.005
to_keep = fi[fi.imp>0.005].cols

# dimensions (rows,)
len(to_keep)


# Now we can create a **new dataset** that contain only the **selected columns**.
# - We'll need to **split** the dataframe again into separate training and validation sets.
# 
# Before fitting our model with the new training set we'll plot one more time the feature importance.
# - Chances are that those values will vary when we rerun `rf_feat_importance` on the newly trained model.
# - We' plot only the 10 most important columns so it's easier to see the changes happening there.

# In[ ]:


# plot the feature importance of the first 10 columns
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,7))


# In[ ]:


# create a dataframe with the selected columns
X_keep = X[to_keep].copy()

# split X
X_train, X_valid = split_vals(X_keep, split_point)


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)


# It seems that our model is almost as good as before, although now it's a simpler model.
# - Previous score:
# - Let's **use this dataframe** from here. 
# 
# Now we'll see if the feature importances have changed after dropping redundant columns from the training set.

# In[ ]:


# create a dataframe of the feature importance
fi = rf_feat_importance(m, X_train)

# plot the feature importance of the first 10 columns 
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,7))


# When **redundant columns** are **removed** the **relationships** between those redundant columns and the most **informative columns** are also are removed .
# - Those relationships (**collinearity**) between columns can have an effect on the **feature importance** of some of the columns. 
# 
# The are 4 columns that stand out from the rest in terms of its feature importance. Let's take a closer look at them.

# In[ ]:


# dtypes in the DataFrame
print(df_raw[['YearMade','ProductSize','Coupler_System','fiProductClassDesc']].dtypes)

# generate descriptive statistics of all columns
df_raw[['YearMade','ProductSize','Coupler_System','fiProductClassDesc']].describe(include='all')


# Now that we know that 3 of the 4 most informative columns are categorical variables, we might want to make some modifications so our model can use the information in this categories a little better.
# The way the categories are encoded right now, there are lots of possible values encoded in the same column.
# - That has worked fine until now, but we can try a different approach so it doesn't take the random forest so many decisions to find valuable insights.

# # Lecture 4
# It is recommended to [watch the video-lecture first](https://youtu.be/0v93qHDqq_g), then follow the notebook.

# ## One-hot encoding
# One-hot encoding, like dummy encoding for lineal models, works by recoding all the categorical values in **different columns** with **zeroes** and **ones**. The difference is that one-hot encoding will create n number columns and dummy encoding will create n-1 ([source](https://stats.stackexchange.com/questions/224051/one-hot-vs-dummy-encoding-in-scikit-learn)).
# - If new binary variables are added for each unique categorical value, it takes the random forest only **one step to decide** if it's the best split point.
# - We can use the optional parameter `max_n_cat` in **`proc_df`**, to do **selective one-hot encoding**.
# - It will turn categorical variables into new columns only if their number of categories is **less or equal to chosen value**.
# - Considering the three categorical variables we are interested have 6, 2 and 74 unique values, we might want set **`max_n_cat` to 7** so these two columns are included.
# 
# Now some of these new columns may prove to have more important features than in the earlier situation, where all categories were in one column.
# - We'll use `proc_df` with the new parameter and split the dataframe again.

# In[ ]:


X_one_hot, _, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(X_one_hot, split_point)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)


# In[ ]:


# create a dataframe of the feature importance
fi = rf_feat_importance(m, X_train)

# plot the feature importance of the first 10 columns
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,7))


# More important that a slight decrease or increase in accuracy are the **insights gathered** through this approach.
# - Now we know which **categorical values** are the most important and we'll use this information in further analysis when we look at partial dependence.

# # Removing redundant features

# We can use cluster analysis to better understand the relationships between variables. Clustering essentially means grouping objects by similarity in groups called clusters.
# 
# We'll use **hierarchical clustering** (also called agglomerative clustering) which is a method of cluster analysis that builds **nested clusters** represented as a tree diagram (or dendrogram).
# - We'll use the [scipy's hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html): **`scipy.cluster.hierarchy`**. 
# - All that is used is a **matrix** of distances, that reflects cluster similarity, and a **method** to calculate the distances between clusters which produces a **linkage matrix**.
# - To create a matrix of distances we need a **distance metric** or correlation coefficient.
# - We'll use Spearman correlation coefficient because is a **nonparametric** metric, apropiate for our model (decision trees are nonparametric).
#  - We'll use the [scipy's Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html): **`scipy.stats.spearmanr`**.
# - The [method](https://en.wikipedia.org/wiki/UPGMA) used to perform the clustering it's called [average linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.average.html#scipy.cluster.hierarchy.average): **`scipy.cluster.hierarchy.average`**.
# - Finally, we'll plot the [dendrogram](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html): **`scipy.cluster.hierarchy.dendrogram`**.
# 
# Sources: [1](https://en.wikipedia.org/wiki/Cluster_analysis), [2](https://en.wikipedia.org/wiki/Hierarchical_clustering), [3](https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/).

# In[ ]:


# import stats
from scipy import stats

# create distance matrix
matrix = stats.spearmanr(X_keep).correlation

# show distance matrix as a dataframe for demostration
pd.DataFrame(matrix)


# In[ ]:


# import average linkage
from scipy.cluster.hierarchy import average

# create linkage matrix
linkage = average(matrix)

# show linkage matrix as a dataframe for demostration
pd.DataFrame(linkage)


# In[ ]:


# import dendrogram
from scipy.cluster.hierarchy import dendrogram

# import matplotlib
import matplotlib.pyplot as plt

# set the size of the figure (plot container)
plt.figure(figsize=(15,7))

# plot dendrogram (inside figure)
dendrogram(linkage, labels=X_keep.columns, orientation='left', leaf_font_size=16)

# display figure
plt.show()


# There are **4 groups** of very similar variables (the numbers measure dissimiliraty) compared to the rest, so there's a chance some of them might be redundant:
# - saleYear and saleElapsed
# - ProductGroup and ProductGroupDesc
# - fiBaseModel and fiModelDesc
# - Grouser_Tracks, Coupler_System and Hydraulics_Flow
# 
# Let's try removing some of these related features to see if the model can be simplified without impacting the accuracy.
# - We'll create a function that give us the only the oob_score and we'll remove each variables one at a time to see what impacts does it make.

# In[ ]:


# create a function that takes a dataframe as argument and returns the oob_score of a random forest trained on that fataframe
def get_oob(dataframe):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    X, _ = split_vals(dataframe, split_point)
    m.fit(X, y_train)
    return m.oob_score_


# In[ ]:


# baseline to compare to
get_oob(X_keep)


# In[ ]:


# loop through the selected columns and print the oob_score with that column removed from the dataframe
for column in ('saleYear', 'saleElapsed',
               'ProductGroup' ,'ProductGroupDesc',
               'fiBaseModel','fiModelDesc',
               'Grouser_Tracks', 'Coupler_System', 'Hydraulics_Flow'):
    print(column, get_oob(X_keep.drop(column, axis=1)))


# Removing any of these columns doesn't affect drastically the accuracy of the model, so it looks like we can try to remove one from each group. Let's see what that does.

# In[ ]:


# list of columns names
to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Hydraulics_Flow']

# returns oob_score with the selected columns removed
get_oob(X_keep.drop(to_drop, axis=1))


# It seems that our model is almost as good as before, although now it's a simpler model.
# - Let's **use this dataframe** from here. 

# In[ ]:


# drop inplace the selected columns from X_keep
X_keep.drop(to_drop, axis=1, inplace=True)

# split X_keep
X_train, X_valid = split_vals(X_keep, split_point)


# And let's see how this model looks on the **full dataset**.

# In[ ]:


# import reset_rf_samples
from fastai.structured import reset_rf_samples

# use full bootstrap sample
reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)


# # Partial dependence
# To investigate the interactions between the features and the target, we can use graphical tools like [partial dependence plots](https://www.kaggle.com/dansbecker/partial-dependence-plots).
# - Partial dependence plots show **how** each **variable affects** the model's **predictions**.
# - We won't use the whole dataframe for the partial dependence plots because we don't need that much information.
#  - We'll use a method from the fastai library, **`get_sample`**, to "get a random sample of n rows from df, without replacement".
# - We'll use 1-hot encoding (`max_n_cat` parameter in `proc_df`) and subsampling (`set_rf_samples`).
# 
# We've [added a custom package](https://www.kaggle.com/docs/kernels#modifying-the-default-environment) to the kernel enviroment, **pdpbox**, because it makes tha task easier, but there are [other ways](https://www.kaggle.com/dansbecker/partial-dependence-plots) to create partial dependence plots.

# In[ ]:


set_rf_samples(50000)


# In[ ]:


X_one_hot, _, nas = proc_df(df_raw, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(X_one_hot, split_point)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)


# In[ ]:


fi = rf_feat_importance(m, X_train)
fi[:10].plot('cols', 'imp', 'barh', figsize=(15,4))


# We'll plot partial dependence plots to know more about Enclosure and YearMade.

# In[ ]:


# import get_sample
from fastai.structured import get_sample

# random sample of 1000 rows from X_train
X_sample = get_sample(X_train, 1000)


# In[ ]:


# import partial dependence calculator and plotter
from pdpbox.pdp import pdp_isolate, pdp_plot

# list of features names
features = ['Enclosure_EROPS', 'Enclosure_EROPS AC', 'Enclosure_EROPS w AC', 'Enclosure_NO ROPS', 'Enclosure_None or Unspecified', 'Enclosure_OROPS']

# calculate partial dependence plot (model, dataframe, dataframe.columns, 'feature') 
pdp = pdp_isolate(m, X_sample, X_sample.columns, features)

# plot partial dependent plot (pdp_isolate, 'name')
pdp_plot(pdp, 'Enclosure')


# We can see clearly that 'Enclosure_EROPS w AC' drives the prices up.
# - Now maybe the time to know [what that term means](http://articles.extension.org/pages/66325/rollover-protective-structures).
# - We'll discover that EROPS means that the machine has an **enclosed cabin** and that makes it possible to have heating and **air conditioning**.
# 
# Before we look at YearMade, we need to fix the value YearMade 1000 that we saw earlier so it doesn't mess up with the plot.

# In[ ]:


pd.set_option('display.max_columns', None)
X_train[X_train.YearMade == 1000].head()


# Considering that has rows with Enclosure_EROPS w AC values, it can't be a category for machines so old that don't have any record.
# - It seems more likely to be a marker for missing values.
# - We'll just ignore it in the plot.

# In[ ]:


# pick only rows that have a YearMade value larger than 1000
X_sample = get_sample(X_train[X_train.YearMade>1000], 1000)

# calculate partial dependence plot (model, dataframe, dataframe.columns, 'feature') 
pdp = pdp_isolate(m, X_sample,X_sample.columns,'YearMade')

# plot partial dependent plot (pdp_isolate, 'name')
pdp_plot(pdp, 'YearMade')


# Here the trend is also clear. The newer the machine, the more expensive it is.
# - Obvious if we consider that  'Enclosure_EROPS w AC' category won't comprise the older machines.

# # Lecture 5
# It is recommended to [watch the video-lecture first](https://youtu.be/3jl2h9hSRvc), then follow the notebook.

# # Tree interpreter

# I've been unable to add a second custom package to the kernel enviroment so I'll **skip this section** for now. 
# - [Tree interpreter](https://github.com/andosa/treeinterpreter) is a package for interpreting scikit-learn's decision tree and random forest predictions.
# - Allows decomposing each prediction into bias and feature contribution components as described in this [blog post](http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/).
# - For the original code of this section, check out the [github repo](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb); for the explanations, watch the [video-lecture](https://youtu.be/3jl2h9hSRvc?t=44m26s).

# # Extrapolation

# Random Forest **do not extrapolate**, wich basically means they cannot predict future trends. 
# -  "In a general sense, to extrapolate is to infer something that is not explicitly stated from existing information" ([source](https://whatis.techtarget.com/definition/extrapolation-and-interpolation)).
# - Unlike linear models, random forests will never be able to predict values bigger or smaller than the max and min in the training data ([source](https://www.quantopian.com/posts/random-forest-unable-to-predict-outside-of-training-data)).
# - That means random forests are limited to short-range prediction, assuming the new data won't be much different from the original training data (sources: [1](http://www.breakingbayes.com/2017/05/30/detemporalized-random-forest-time-series-modeling/), [2](http://freerangestats.info/blog/2016/12/10/extrapolation)).
# 
# Luckily, we are making **short-range predictions**, but even there extrapolation can be problematic.
# - One way to improve the ability of our model of predicting future prices is to try to detemporalize the training set.
# - That is to **avoid using time related variables** as predictors in order to force the model to find other variables that have a strong relationship with the oucome and that might work better when trying to predict the future.
# 
# We can use random forest **interpretation** to figure out the best way to **detemporalize** the training set.
# - We'll build a model that is the **opposite** of what we want: it will rely heavily on time related variables to make predictions.
# - Next, we'll use what it has learned as a **guide** of which variables not to use.
# - The way we do it is by performing a little **experiment**:
#  - instead of training the model to predict the **price** (regression problem),
#  - we are going to train it to predict if a given row can be found or not **in the validation set** (classification problem).
# - Since our validation set is new data in the near future, the model will need to learn which variables are more likely to appear in the **future**.
# - Essentially, this will force the model to recognize **temporal patterns** in the data that may be hidden to us.
# - Next, we'll use **feature importance** to know which variables have a stronger temporal component.

# In[ ]:


# make a copy
X = X_keep.copy()

# create a new target column (empty)
X['in_validation_set'] = None

# set rows in the training set to False (up to last 12000 rows)
X.in_validation_set[:split_point] = False

# set rows in the validation set to True (last 12000 rows)
X.in_validation_set[split_point:] = True

# split X, y
X, y, nas = proc_df(X, 'in_validation_set')


# In[ ]:


# import the class
from sklearn.ensemble import RandomForestClassifier

m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X, y)
m.oob_score_ # if the validation set was random (not time dependent), this experiment would not work and the oob error would be bad (far from 1).


# In[ ]:


# create a dataframe of the feature importance
fi = rf_feat_importance(m, X)

# plot the feature importance of first 5 columns
fi.head().plot('cols', 'imp', 'barh')


# Let's take a look at these columns.
# - SalesID: "unique identifier of a particular sale of a machine at auction".
# - saleElapsed:  It's a [unix timestamp](https://en.wikipedia.org/wiki/Unix_time) (time elapsed in seconds since a universal point in the past).
#  - column created by add_datepart method from original column saledate:  "time of sale".
# - MachineID: "identifier for a particular machine;  machines may have multiple sales".

# In[ ]:


X['SalesID'].plot()


# In[ ]:


X['saleElapsed'].plot()


# In[ ]:


X['MachineID'].plot()


# Clearly there is a trend in the validation set that the model has learned which suggest that these variables have an underlying temporal component.
# - Let's try to remove them one at a time and see how the model perfoms without them.

# In[ ]:


# baseline to compare to

X_train, X_valid = split_vals(X_keep, split_point)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)


# In[ ]:


# loop through the selected columns and print the scores of a random forest trained with that column removed from the training set.
for column in ('SalesID', 'saleElapsed', 'MachineID'):
    X_train, X_valid = split_vals(X_keep.drop(column, axis=1), split_point)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(column)
    print_scores(m)


# Let's drop the ones that improved our score and compare it one more time.

# In[ ]:


# drop the columns from the dataframe and split X
X_train, X_valid = split_vals(X_keep.drop(['SalesID', 'MachineID'], axis=1), split_point)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)


# It seems we did a good job forcing the model to use other variables instead of the time related ones.
# - Let's try the model on the **full dataset**.

# In[ ]:


# use full bootstrap sample
reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_scores(m)


# In[ ]:


# plot the feature importance of the column names in 'cols'
rf_feat_importance(m, X_train).plot('cols', 'imp', 'barh', figsize=(12,7))


# # Final model

# Now that we found the best settings for our model, let's train it one last time using lots of trees.
# 
# We cannot submit to the [leaderboard](https://www.kaggle.com/c/bluebook-for-bulldozers/leaderboard) anymore because it's and old competition, but we can compare our RMSLE score of the validation set.

# In[ ]:


m = RandomForestRegressor(n_estimators=160, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_scores(m)

