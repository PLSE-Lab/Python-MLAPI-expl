#!/usr/bin/env python
# coding: utf-8

# # House Prices: Exploratory Data Analysis
# 
# Greetings! In this notebook we will explore the Ames house price data, identifying important characteristics and noting where pre-processing will be required. The purpose of this work is to lay the groundwork for later data cleaning and transformation, with the end goal of applying various machine learning strategies.
# 
# Thank you for reading and please leave a comment below if you have any suggestions. I am always learning and appreciate your feedback.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats

# Prevent Pandas from truncating displayed dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

sns.set(style="white", font_scale=1.2)
plt.rcParams["figure.figsize"] = [10,8]


# **Load the Data**

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")


# **Preview the Data**

# In[ ]:


train.head()


# **Observe the Data Structure & Expected Output**

# Let's look at the sample submission. This shows us what our model will need to output: a sale price for each `Id` in the test set.

# In[ ]:


submission.head()


# To accomplish this task, must train a model using the train dataset and predict values for each observation in the test dataset. What are the sizes of each of these datasets?

# In[ ]:


print("Training Size: {} observations, {} features\nTest Size: {} observations, {} features\n".format(train.shape[0], train.shape[1], test.shape[0], test.shape[1]))


# We can see that there the training and test sets are roughly the same size (around 1460 rows). We have a large number of features (80), some of which might be candidates for deletion.

# In[ ]:


set(train.columns) - set(test.columns)


# As expected, the test set is missing the `SalePrice` column. This is the target variable that we are tasked to predict.
# 
# The `.describe()` method allows us to quickly eyeball various statistics for each feature, including min, median and max.

# In[ ]:


train.describe()


# What numeric features do we have?

# In[ ]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns


# And what categorical features do we have?

# In[ ]:


categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns


# How many different categories are there for each of the categorical variables? Do we have categories that don't appear in both train and test?

# In[ ]:


unique_categories = pd.DataFrame(index=categorical_features.columns, columns=["TrainCount", "TestCount"])
for c in categorical_features.columns:
    unique_categories.loc[c, "TrainCount"] = len(train[c].value_counts())
    unique_categories.loc[c, "TestCount"] = len(test[c].value_counts())
    
unique_categories = unique_categories.sort_values(by="TrainCount", ascending=False)
unique_categories.head()


# A quick plot of the number of unique categories per feature indicates that different categories are indeed present in train vs. test. Our model will need to handle the case when it encounters a new category that it wasn't originally trained on.

# In[ ]:


temp = pd.melt(unique_categories.reset_index(), id_vars="index")
g = sns.catplot(y="index", x="value", hue="variable", data=temp, kind="bar", height=9)
g.set_ylabels("Count")
g.set_xlabels("Categorical Variable")
g.set_xticklabels(rotation=90)
plt.title("Number of Unique Categories by Feature")
plt.show()


# **Check for Nulls**
# 
# Missing values introduce bias into our dataset and can lead to biased conclusions or predictions. There are a variety of methods that can be used to deal with missing values, but first, let's just see which features contain N/As.

# In[ ]:


nulls = train.isnull().sum()[train.isnull().sum() > 0].sort_values(ascending=False).to_frame().rename(columns={0: "MissingVals"})
nulls["MissingValsPct"] = nulls["MissingVals"] / len(train)
nulls


# `PoolQC`, `MiscFeature` and `Alley` are all missing a significant number of values (>90%) - these features are candidates for deletion as there is likely little information contained within the remaining entries.
# 
# Notice how the `Garage_` and `Bsmt_` variables have similar numbers of missing values. One possibility is that these missing values originate from the same set of observations.
# 
# It can be helpful to view the percentage of missing values in a bar chart, to get a sense of the relativities.

# In[ ]:


sns.barplot(y=nulls.index, x=nulls["MissingValsPct"], orient="h")
plt.title("% of Values Missing by Feature")
plt.show()


# We can also observe the co-occurrence of nulls across features. This exhibit confirms our theory that `Bsmt_` and `Garage_` nulls tend to occur within the same observations (see the bands of horizontal white lines).

# In[ ]:


msno.matrix(train, labels=True)
plt.show()


# **Check for Outliers**
# 
# One way to check for outliers is by calculating the Z-score, which represents the number of standard deviations away from the observed mean value. Typically, if $|Z-score| > 3$, the data point is considered an outlier. We may choose to remove these rows prior to training, however we will need to weigh the cost of deleting datapoints for our model.

# In[ ]:


z_threshold = 3
z = pd.DataFrame(np.abs(stats.zscore(train[numeric_features.columns])))
outlier_rows = z[z[z > z_threshold].any(axis=1)] # Rows with outliers
print("# Rows with potential outliers: {}".format(len(outlier_rows)))
outlier_rows.head()


# **Observe the Target Distribution**

# It's important to understand the variable that we are trying to predict. Below, we see that the distribution of `SalePrice` is right skewed, indicating the presence of outliers (unusually high-priced homes); we can also observe these outliers in a box plot. Lastly, we create a Q-Q to confirm that the `SalePrice` distribution does not follow a normal distribution. Since linear models generally work best with normally distributed data, we will need to manipulate `SalePrice`, for example by taking a log-transformation.

# In[ ]:


fig, ax = plt.subplots(1,3, figsize=(15,5))
sns.distplot(train["SalePrice"], ax=ax[0], fit=stats.norm)
sns.boxplot(train["SalePrice"], orient='v', ax=ax[1])
stats.probplot(train["SalePrice"], plot=plt)

ax[0].set_title("SalePrice Distribution vs. Normal Distribution")
ax[1].set_title("Boxplot of SalePrice")
ax[2].set_title("Q-Q Plot of SalePrice")
ax[0].set_ylabel("SalePrice")
ax[1].set_xlabel("All Homes")

for a in ax:
    for label in a.get_xticklabels():
        label.set_rotation(90)
plt.tight_layout()
plt.show()


# After apply a log-transformation, the `SalePrice` distribution is no longer skewed and our outliers are more evenly distributed. Likewise, our Q-Q plot indicates the data is close to normally distributed.

# In[ ]:


log_SalePrice = np.log1p(train["SalePrice"]) # Applies log(1+x) to all elements of column
fig, ax = plt.subplots(1,3, figsize=(15,5))
sns.distplot(log_SalePrice, ax=ax[0], fit=stats.norm)
sns.boxplot(log_SalePrice, orient='v', ax=ax[1])
stats.probplot(log_SalePrice, plot=plt)

ax[0].set_title("Log(SalePrice + 1) vs. Normal Distribution")
ax[1].set_title("Boxplot of Log(SalePrice + 1)")
ax[2].set_title("Q-Q Plot of Log(SalePrice + 1)")
ax[0].set_ylabel("SalePrice")
ax[1].set_xlabel("All Homes")


plt.tight_layout()
plt.show()


# **Observe Numerical Feature Correlation with a Heatmap**
# 
# Diving into the numerical features, we use a heatmap to quickly visualize which variables are correlated (move together).

# In[ ]:


fig = plt.figure(figsize=(15,12))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, square=True)
plt.title("Heatmap of All Numerical Features")
plt.show()


# Notice the significant correlation between `TotalBsmtSF`/`1stFlrSF` and `GarageCars`/`GarageArea`. This indicates that these features provide almost the same information; we may choose to use just one or the other. Let's isolate the features with highest correlation so their interrelationships are easier to see.

# In[ ]:


fig = plt.figure(figsize=(15,12))
sns.heatmap(corr_matrix[(corr_matrix > 0.5) | (corr_matrix < -0.5)], annot=True, annot_kws={"size": 9}, linewidths=0.1, square=True)
plt.title("Heatmap of Highest Correlated Features")
plt.show()


# Does it make sense that some of these variables are highly correlated?
# 
# * `GarageCars` and `GarageArea`: Since the number of cars that can fit in a garage is a byproduct of the garage's area, we expect these features to be correlated.
# * `YearBuilt` and `GarageYrBlt`: This one is more a by-product of the fact that the garage must have been built the same year or sometime after the house was built.
# * `TotalBsmtSF` and `1stFlrSF`: We would expect that the total basement square footage and 1st floor square footage are related.
# * `GrLiveArea` and `TotRmsAbvGrd`: Again, this makes sense. More rooms, more living space.

# We can also "zoom in" to examine variables with the highest correlation; below we see the 10 variables that are most positively correlated with `SalePrice`, including `OverallQual` and `GrLivArea`.
# 
# Finally, on a final feature selection note, `Id` can probably be safely deleted as it is simply a auto-incrementing label for each observation.

# In[ ]:


k = 11 #number of variables for heatmap (including SalePrice)
cols_positive = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols_positive].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_positive.values, xticklabels=cols_positive.values)
plt.show()


# And likewise, the 10 variables which are most negatively correlated with SalePrice.

# In[ ]:


k = 10 #number of variables for heatmap
cols_negative = np.append(['SalePrice'], corr_matrix.nsmallest(k, 'SalePrice')['SalePrice'].index.values)
cm = np.corrcoef(train[cols_negative].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_negative, xticklabels=cols_negative)
plt.show()


# Let's create a pairplot using the positively correlated features that we identified. Take a moment to look at some of the plots to see if you can explain the relationship between various features.

# In[ ]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
sns.pairplot(train[cols_positive], diag_kind='kde')
plt.show()


# **Observe Categorical Features with Box Plots**

# What about the categorical features? Let's use box plots to see which features might be helpful in predicting `SalePrice`.
# 
# First, we'll need to create a new category for missing values.

# In[ ]:


x = train.copy()
for c in categorical_features.columns:
    x[c] = x[c].astype('category')
    if x[c].isnull().any():
        x[c] = x[c].cat.add_categories(['Missing'])
        x[c] = x[c].fillna('Missing')
x["SalePrice"] = train["SalePrice"]
x.head()


# Now we can generate box plots. Note that by creating a category for missing values, we are able to view the distribution of sale price for those observations.

# In[ ]:


def boxplot_custom(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)

df = pd.melt(x, id_vars=["SalePrice"], value_vars=categorical_features)
g = sns.FacetGrid(df, col="variable", col_wrap=3, sharex=False, sharey=False, height=5)
g = g.map(boxplot_custom, "value", "SalePrice")
plt.show()


# Because there are so many categorical variables, we will likely need to identiy and utilize only the categorical variables that have the strongest correlation with the target. Alternatively, we might choose to include all variables and use penalty-based automated variable selection, such as lasso or ridge regression, to prioritize the most salient features. Topics for future notebooks :)

# **Conclusion**

# Thank you very much for reading - I hope you learned a trick or two. 
# 
# Suggestions? Comments? Please leave me a note below.
# 
# Until next time, happy coding :)
