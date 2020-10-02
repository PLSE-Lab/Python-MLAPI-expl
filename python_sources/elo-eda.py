#!/usr/bin/env python
# coding: utf-8

# Elo Merchant Category Recommendation
# =====
# 
# # Objective
# 
# Develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. We have to predict a loyalty score for each ```card_id``` represented in ```test.csv```.
# 
# # Evaluation Metric
# 
# Evaluation metric is RMSE - $\textrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$, 
# where $\hat{y}$ is the predicted loyalty score for each ```card_id```, and $y$ is the actual loyalty score assigned to a ```card_id```.
# 
# # Data
# 
# - **train.csv**: Training set having the information of cards. It also has the loyalty score column.
# - **test.csv**: Testing set having the information on cards
# - **historical_transactions.csv**: Up to 3 months' worth of historical transactions for each card_id
# - **merchants.csv**: Additional information about all merchants / ```merchant_id```s in the dataset.
# - **new_merchant_transactions.csv**: two months' worth of data for each ```card_id``` containing ALL purchases that ```card_id``` made at ```merchant_id```s that were not visited in the historical data.
# 

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')

# display all the outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# utility functions
def get_col_stats(df):
    temp_nulls = df.isnull().sum()
    temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls*100/df.shape[0]})
    
    uniqs = []
    for col in df.columns:
        uniqs.append(df[col].unique().shape[0])
    temp["uniqs"] = uniqs
    return temp

def get_categorical_stats(df, col):
    counts = df[col].value_counts(dropna=False)
    temp = pd.DataFrame({
        "counts": counts,
        "count_per": counts * 100 / df.shape[0],
    })
    return temp

def numeric_to_cat(df, cols):
    for col in cols:
        df[col] = df[col].astype("category")
    return df


# In[ ]:


DATA_ROOT = Path("../input/")

card_f = DATA_ROOT / "train.csv"
merchant_f = DATA_ROOT / "merchants.csv"
hist_f = DATA_ROOT / "historical_transactions.csv"
new_hist_f = DATA_ROOT / "new_merchant_transactions.csv"

test_f = DATA_ROOT / "test.csv"


# # Card Data
# 
# 
# 

# In[ ]:


card_df = pd.read_csv(card_f)
card_df.shape
card_df.head()


# In[ ]:


test_df = pd.read_csv(test_f)
test_df.shape
test_df.head()


# ## Sanity Check
# 
# - Unique ```card_id```'s in training data
# - Unique ```card_id```'s in testing data
# - Overlap of cards in training and testing data
# - Nulls

# In[ ]:


card_df.card_id.unique().shape
card_df.card_id.unique().shape[0] == card_df.shape[0]


# There are no duplicate ```card_id```'s in the train set.

# In[ ]:


test_df.card_id.unique().shape
test_df.card_id.unique().shape[0] == test_df.shape[0]


# There are no duplicate ```card_id```'s in the test set.

# In[ ]:


len(set(test_df.card_id).intersection(set(card_df.card_id)))


# There's no overlapping cards in the training and testing datasets.

# In[ ]:


print("Training data:")
card_df.isna().sum()
print("Testing data:")
test_df.isna().sum()


# In[ ]:


test_df[test_df.first_active_month.isnull()]


# Training dataset has no nulls but testing dataset has one row where the ```first_active_month``` column is Null. Impute testing data specifically?  
# We are done with our sanity check now. Let's explore the table.
# 
# ## Exploration
# 
# ### 1. Target Variable
# 
# Target column is a numerical loyalty score.
# 
# - Negative means the card holder is not loyal?
# - And, positive means the opposite?

# In[ ]:


ax = card_df.target.plot.hist(bins=20, figsize=(10, 5))
_ = ax.set_title("target histogram")
plt.show()

fig, axs = plt.subplots(1,2, figsize=(20, 5))
_ = card_df.target[card_df.target > 10].plot.hist(ax=axs[0])
_ = axs[0].set_title("target histogram for values greater than 10")
_ = card_df.target[card_df.target < -10].plot.hist(ax=axs[1])
_ = axs[1].set_title("target histogram for values less than -10")
plt.show()

card_df.target.describe()


# Observations:
# 
# - Values range from -33.2 to 17.9
# - -33 seems like an outlier as can be seen in the 3rd plot
# - other values less than -10 also seem like outliers due to very less in number
# - All values above 10 are also looking like outliers
# 

# In[ ]:


card_df["target_sign"] = card_df.target.apply(lambda x: 0 if x <= 0 else 1)
card_df.target_sign.value_counts()


# Observations:
# 
# - Negative and positive target values are almost in the same proportion
# 
# ### 2. Anonymised Features
# 
# feature_1, feature_2, feature_3

# In[ ]:


print("feature_1")
pd.DataFrame({"counts": card_df.feature_1.value_counts(), "counts_per": card_df.feature_1.value_counts()*100/card_df.shape[0]})
print("feature_2")
pd.DataFrame({"counts": card_df.feature_2.value_counts(), "counts_per": card_df.feature_2.value_counts()*100/card_df.shape[0]})
print("feature_3")
pd.DataFrame({"counts": card_df.feature_3.value_counts(), "counts_per": card_df.feature_3.value_counts()*100/card_df.shape[0]})


# Observations:
# 
# - feature_1, feature_2, feature_3, all are categorical variables
# - feature_1 has 5 unique values
# - feature_2 has 3 unique values
# - feature_3 is a binary column
# 
# ### 3. first_active_month

# In[ ]:


temp = card_df.first_active_month.value_counts().sort_index()
ax = temp.plot(figsize=(10, 5))
_ = ax.set_xticklabels(range(2010, 2019))
_ = ax.set_title("Distribution across years")


# Observations:
# 
# - Most of the data lies in the years ranging from 2016 to 2018
# 

# In[ ]:


card_df["yr"] = card_df.first_active_month.str.split("-").str[0]
card_df["month"] = card_df.first_active_month.str.split("-").str[1]
card_df.head()


# In[ ]:


temp = get_categorical_stats(card_df, "yr")
temp

ax = temp.counts.sort_index().plot()
_ = ax.set_xticklabels(range(2010, 2019))


# Observations:
# 
# - Years range from 2011 to 2018
# - 64% data is from 2017, followed by 2016 (25%) and 2015 (7%)
# - Very less data from 2017 (may be, test data will have data from 2018?)

# In[ ]:


temp = get_categorical_stats(card_df, "month")
temp

ax = temp.counts.sort_index().plot()
_ = ax.set_xticklabels(range(-1, 13, 2))


# Observations:
# 
# - last 6 months (July to December) has relatively more data than first 6 months (January to June)
# 
# ### 4. card_id

# In[ ]:


card_df["card_id_dec"] = card_df.card_id.str.split("_").str[2].apply(lambda x: int(x, 16))

card_df.card_id.str.split("_").str[0].unique()
card_df.card_id.str.split("_").str[1].unique()
card_df.card_id_dec.describe()


# In[ ]:


card_df[["card_id_dec", "first_active_month"]].sort_values("card_id_dec")


# Observations:
# 
# - tried to see if there was any pattern in the ids.
# - converted hex values to decimal, but when we see it according to the ```first_active_month``` then there is no apparent order in the ids
# 
# 
# ### 5. Anonimised features vs target

# In[ ]:


_ = card_df[["feature_1", "target"]].plot.scatter(x="feature_1", y="target")
_ = card_df[["feature_2", "target"]].plot.scatter(x="feature_2", y="target")
_ = card_df[["feature_3", "target"]].plot.scatter(x="feature_3", y="target")


# Observations:
# 
# - all anonymised features are similar in value ranges for the target column
# - -33 target value is quite distinct across all the variables
# - Maybe, -33 is a default value of loyalty score.
# 
# 
# ### 6. Year vs feature_1
# 

# In[ ]:


card_df.groupby(["yr", "feature_1"])["month"].count()


# Observations:
# 
# - Value 3 (most frequent category in the whole feature_1 column) is at all time high across all the years except 2018
# - Value 2 is another value which is 2nd highest after 2015
# 
# ### 7. Year vs feature_2

# In[ ]:


card_df.groupby(["yr", "feature_2"])["month"].count()


# Observations:
# 
# - 1 and 3 are present for all the years
# - 2 started showing up with good numbers from the year 2015

# # Merchant Data

# In[ ]:


merc_df = pd.read_csv(merchant_f)

minus_1_to_nan_cols = ["city_id", "state_id", "merchant_group_id",
                      "merchant_category_id", "subsector_id"]
for col in minus_1_to_nan_cols:
    merc_df[col] = merc_df[col].replace(-1, pd.np.nan)

num_to_cat_cols = ["category_2", "city_id", "state_id",
                  "merchant_group_id", "merchant_category_id",
                   "subsector_id"]
merc_df = numeric_to_cat(merc_df, num_to_cat_cols)

merc_df.shape
merc_df.head()


# ## Sanity Check
# 
# - Unique ```merchant_id```s
# - Nulls

# In[ ]:


merc_df.merchant_id.unique().shape
merc_df.merchant_id.unique().shape[0] == merc_df.shape[0]


# Few ```merchant_id```s have more than one rows. Lets investigate some of them.

# In[ ]:


temp = merc_df.merchant_id.value_counts()
temp[temp > 1]


# In[ ]:


merc_df[merc_df.merchant_id == "M_ID_d123532c72"]


# Most of the columns have same values, the ```_lag``` columns are different. We can try deduping the rows by taking an average of those values.

# In[ ]:


temp_nulls = merc_df.isnull().sum()
temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls*100/merc_df.shape[0]})
temp


# Great!! No nulls in the id columns.
# 
# ## Exploration
# 
# ### 1. Anonymised measure
# numerical_1, numerical_1

# In[ ]:


merc_df[["numerical_1", "numerical_2"]].describe()


# Observations:
# - No nulls in both the columns
# - ```numerical_1``` values ranges from -0.057 to 183.735
# - ```numerical_2``` values ranges from -0.008 to 182.097
# - 75th quantile is at -0.047 for both ```numerical_1``` and ```numerical_2```
# - max value for both is very close

# In[ ]:


fig, ax = plt.subplots(1, 2)
_ = merc_df[["numerical_1"]].boxplot(ax=ax[0])
_ = merc_df[["numerical_2"]].boxplot(ax=ax[1])

fig, ax = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.numerical_1 < 0.1, ["numerical_1"]].boxplot(ax=ax[0])
_ = merc_df.loc[merc_df.numerical_2 < 0.01, ["numerical_2"]].boxplot(ax=ax[1])
plt.tight_layout()


# Observations:
# - Big tail after IQR (many outliers) for both the columns

# In[ ]:


_ = merc_df[["numerical_1", "numerical_2"]].plot.scatter(x="numerical_1", y="numerical_2")


# Observations:
# - A clean linear relationship b/w numerical_1 and numerical_2
# - will remove one column when being used in the model
# - also handle the outliers (what will be the threshold)

# ### 2. Anon Categories
# category_1, category_2, category_4

# In[ ]:


print("category_1")
get_categorical_stats(merc_df, "category_1")
print("category_2")
get_categorical_stats(merc_df, "category_2")
print("category_4")
get_categorical_stats(merc_df, "category_4")


# Observations:
# - category_1 is a binary column (N/Y)
# - category_4 is a binary column (N/Y)
# - category_1 and category_4 have very high number of N
# - category_2 has 5 categories and also nulls (11,887)
# - category_2 seems like a ratings column with 1 being the most frequent rating followed by 5 and 3.

# In[ ]:


merc_df.groupby(["category_4", "category_2"])["category_1"].count()


# In[ ]:


print("Ratio of category_4 with value N to value Y based on category_2 freqeuncy counts")
temp1 = merc_df.loc[merc_df.category_4 == "N", "category_2"].value_counts(dropna=False)
temp2 = merc_df.loc[merc_df.category_4 == "Y", "category_2"].value_counts(dropna=False)
temp1/temp2


# Observations:
# - Ratio of merchants with N to Y is 3.4 for category_2 as 1
# - Ratio of merchants with N to Y is 4.2 for category_2 as 5
# - All others revolve around 1.5

# In[ ]:


merc_df.groupby(["category_1", "category_2"])["category_4"].count()


# In[ ]:


merc_df.loc[merc_df.category_1 == "Y", "category_2"].value_counts(dropna=False)


# Observations:
# - All the merchants under category_1 as Y have all nulls in category_2
# - Remove category_1 column when using in training as it's not adding much information
# 

# In[ ]:


merc_df.groupby(["category_1", "category_4"])["category_2"].count()


# ### 3. Location ID cols
# 
# city_id, state_id

# In[ ]:


print("city_id")
get_categorical_stats(merc_df, "city_id")


# Observations:
# - 31.4% data in city_id is null
# - There are 216 distinct cities merchants belong to

# In[ ]:


print("state_id")
get_categorical_stats(merc_df, "state_id")


# Observations:
# - There are 24 unique states; not sure if this is US (US has 50 states)
# - 3.5% nulls
# 
# ```state_id``` and ```category_2``` has same number of nulls - 11,887. Let's see if they occur together

# In[ ]:


merc_df[merc_df.state_id.isna()].category_2.value_counts(dropna=False)


# Yup, nulls in both the cols - ```state_id``` and ```category_2``` - occur together. We'll investigate this slice after we are done with the remaining columns in the merchant table.

# ### 4. Merchant ID cols
# 
# merchant_id, merchant_group_id, merchant_category_id, subsector_id

# In[ ]:


get_col_stats(merc_df[["merchant_id", "merchant_group_id", "merchant_category_id", "subsector_id"]])


# In[ ]:


(
    merc_df.merchant_id.astype(str)\
    + merc_df.merchant_group_id.astype(str)
).unique().shape
merc_df.shape


# Since ```merchant_id``` is not uniquely identifying each row in the table, I tried to find some compound key. Unfortunately, no such key is there. Data will need to be deduped on the ```merchant_id``` as we discussed during the sanity check part.

# ### 5. Most recent sales and purchases
# 
# most_recent_sales_range, most_recent_purchases_range
# 
# Thses categorical variables have the following values: A > B > C > D > E
# 
# 
# most_recent_sales_range: Range of revenue (monetary units) in last active month  
# most_recent_purchases_range: Range of quantity of transactions in last active month

# In[ ]:


print("most_recent_sales_range")
get_categorical_stats(merc_df, "most_recent_sales_range")
print("most_recent_purchases_range")
get_categorical_stats(merc_df, "most_recent_purchases_range")


# In[ ]:


merc_df.groupby(["most_recent_sales_range", "most_recent_purchases_range"])    .merchant_id    .count()


# In[ ]:


fig, axs = plt.subplots(2, 3, figsize=(10, 5))

i = 0
j = 0
for val in merc_df.most_recent_purchases_range.unique():
    ax = axs[i, j]
    _ = merc_df        .loc[merc_df.most_recent_purchases_range == val, "most_recent_sales_range"]        .value_counts()        .plot.bar(ax=ax)
    _ = ax.set_title(f"most_recent_purchases_range = {val}")
    if i==0 and j==2:
        i = 1
        j = 0
    else:
        j += 1

plt.tight_layout()


# Observations:
# - Both the columns have 5 unique values
# - No nulls are present
# - most_recent_sales_range give the range of revenue (monetary units) in last active month  
# - most_recent_purchases_range gives the range of quantity of transactions in last active month
# - the above plots show for each transactions bucket, the distribution across the revenue bucket
# - The lowest transactions bucket (E), highest is the lowest revenue bucket (E) showing that most of the merchants are small in size and most of the transactions happening with them are low
# - similarly, A corresponds to A, B to B, C to C and D to D for sales and revenue in the highest values
# - Can I say these columns are correlated?

# ### 6. lag_3 columns
# 
# - ```avg_sales_lag3```: Monthly average of revenue in last 3 months divided by revenue in last active month
# - ```avg_purchases_lag3```: Monthly average of transactions in last 3 months divided by transactions in last active month
# - ```active_months_lag3```: Quantity of active months within last 3 months
# 

# In[ ]:


merc_df[["avg_sales_lag3", "avg_purchases_lag3", "active_months_lag3"]].describe()


# In[ ]:


get_categorical_stats(merc_df, "active_months_lag3")


# Observations:
# - Quantity of active months within last 3 months, is 3 in 99.5% of data meaning, they were active during all the 3 months

# In[ ]:


fig, axs = plt.subplots(1, 2)
_ = merc_df[["avg_sales_lag3"]].boxplot(ax=axs[0])
_ = merc_df[["avg_purchases_lag3"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.avg_sales_lag3 < 10, ["avg_sales_lag3"]].boxplot(ax=axs[0])
_ = merc_df.loc[merc_df.avg_purchases_lag3 < 10, ["avg_purchases_lag3"]].boxplot(ax=axs[1])
plt.tight_layout()


# Observations:
# - both ```avg_purchases_lag3``` and ```avg_sales_lag3``` have a few outliers in the extremes 

# In[ ]:


temp = merc_df.loc[(merc_df.avg_sales_lag3 < 10) & (merc_df.avg_sales_lag3 > -10), ["avg_sales_lag3", "avg_purchases_lag3"]]
temp.plot.scatter(x="avg_sales_lag3", y="avg_purchases_lag3")


# Observations:
# - due to large values, the plot with the whole data was not proper, so i filtered the avg_sales_lag3 between -10 and 10
# - it seems to be between 0 and 10
# - Need to handle the outliers
# - there is not apparent pattern between the columns though

# ### 7. lag_6 columns
# 
# - ```avg_sales_lag6```: Monthly average of revenue in last 6 months divided by revenue in last active month
# - ```avg_purchases_lag6```: Monthly average of transactions in last 6 months divided by transactions in last active month
# - ```active_months_lag6```: Quantity of active months within last 6 months
# 

# In[ ]:


merc_df[["avg_sales_lag6", "avg_purchases_lag6", "active_months_lag6"]].describe()


# In[ ]:


get_categorical_stats(merc_df, "active_months_lag6")


# Observations:
# - Quantity of active months within last 6 months, is 3 in 97.8% of data meaning, they were active during all the 6 months

# In[ ]:


fig, axs = plt.subplots(1, 2)
_ = merc_df[["avg_sales_lag6"]].boxplot(ax=axs[0])
_ = merc_df[["avg_purchases_lag6"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.avg_sales_lag6 < 10, ["avg_sales_lag6"]].boxplot(ax=axs[0])
_ = merc_df.loc[merc_df.avg_purchases_lag6 < 10, ["avg_purchases_lag6"]].boxplot(ax=axs[1])
plt.tight_layout()


# Observations:
# - both ```avg_purchases_lag6``` and ```avg_sales_lag6``` have a few outliers in the extremes 

# In[ ]:


temp = merc_df.loc[(merc_df.avg_sales_lag6 < 10) & (merc_df.avg_sales_lag6 > -10), ["avg_sales_lag6", "avg_purchases_lag6"]]
temp.plot.scatter(x="avg_sales_lag6", y="avg_purchases_lag6")


# Observations:
# - due to large values, the plot with the whole data was not proper, so i filtered the avg_sales_lag3 between -10 and 10
# - it seems to be between 0 and 10
# - Need to handle the outliers
# - there is not apparent pattern between the columns though

# ### 8. lag_12 columns
# 
# - ```avg_sales_lag12```: Monthly average of revenue in last 12 months divided by revenue in last active month
# - ```avg_purchases_lag12```: Monthly average of transactions in last 12 months divided by transactions in last active month
# - ```active_months_lag12```: Quantity of active months within last 12 months
# 
# work in progress

# In[ ]:


merc_df[["avg_sales_lag12", "avg_purchases_lag12", "active_months_lag12"]].describe()


# In[ ]:


get_categorical_stats(merc_df, "active_months_lag12")


# Observations:
# - Quantity of active months within last 12 months, is 3 in 91.1% of data meaning, they were active during all the 12 months

# In[ ]:


fig, axs = plt.subplots(1, 2)
_ = merc_df[["avg_sales_lag12"]].boxplot(ax=axs[0])
_ = merc_df[["avg_purchases_lag12"]].boxplot(ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(1, 2)
_ = merc_df.loc[merc_df.avg_sales_lag12 < 10, ["avg_sales_lag12"]].boxplot(ax=axs[0])
_ = merc_df.loc[merc_df.avg_purchases_lag12 < 10, ["avg_purchases_lag12"]].boxplot(ax=axs[1])
plt.tight_layout()


# Observations:
# - both ```avg_purchases_lag12``` and ```avg_sales_lag12``` have a few outliers in the extremes 

# In[ ]:


temp = merc_df.loc[(merc_df.avg_sales_lag12 < 10) & (merc_df.avg_sales_lag12 > -10), ["avg_sales_lag12", "avg_purchases_lag12"]]
temp.plot.scatter(x="avg_sales_lag12", y="avg_purchases_lag12")


# Observations:
# - due to large values, the plot with the whole data was not proper, so i filtered the avg_sales_lag3 between -10 and 10
# - it seems to be between 0 and 10
# - Need to handle the outliers
# - there is not apparent pattern between the columns though

# ### 9. numerical_1 vs other columns

# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

filt = (merc_df.avg_sales_lag3 < 10) & (merc_df.avg_sales_lag3 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_sales_lag3"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_sales_lag3", ax=axs[0])

filt = (merc_df.avg_sales_lag6 < 10) & (merc_df.avg_sales_lag6 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_sales_lag6"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_sales_lag6", ax=axs[1])

filt = (merc_df.avg_sales_lag12 < 10) & (merc_df.avg_sales_lag12 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_sales_lag12"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_sales_lag12", ax=axs[2])


# Observations:
# - ```avg_sales_lag``` columns show similar patterns according when plotted corresponding to ```numerical_1```.
# 

# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

filt = (merc_df.avg_purchases_lag3 < 10) & (merc_df.avg_purchases_lag3 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_purchases_lag3"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_purchases_lag3", ax=axs[0])

filt = (merc_df.avg_purchases_lag6 < 10) & (merc_df.avg_purchases_lag6 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_purchases_lag6"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_purchases_lag6", ax=axs[1])

filt = (merc_df.avg_purchases_lag12 < 10) & (merc_df.avg_purchases_lag12 > -10)
temp = merc_df.loc[filt, ["numerical_1", "avg_purchases_lag12"]]
_ = temp.plot.scatter(x="numerical_1", y="avg_purchases_lag12", ax=axs[2])


# Observations:
# - ```avg_purchases_lag``` columns show similar patterns according when plotted corresponding to ```numerical_1```.
# 

# In[ ]:


merc_df.head()


# ### 10. Correlation in lag columns
# 
# 

# In[ ]:


lag_cols = [
    "avg_sales_lag3", "avg_purchases_lag3", "active_months_lag3",
    "avg_sales_lag6", "avg_purchases_lag6", "active_months_lag6",
    "avg_sales_lag12", "avg_purchases_lag12", "active_months_lag12"
]
corr = merc_df[lag_cols].corr()
corr.style.background_gradient()


# ### 11. looking at the slice of data where ```state_id``` and ```category_2``` are nulls
# 

# In[ ]:


filt = (merc_df.state_id.isna()) & (merc_df.category_2.isna())
temp = merc_df.loc[filt]
temp.head()


# In[ ]:





# In[ ]:





# In[ ]:





# ### 12. Deduping merchant data based on ```merchant_id```
# 
# work in progress

# In[ ]:





# ### 13. ```category_2``` vs lag columns

# In[ ]:





# In[ ]:





# # Historical Data

# In[ ]:


hist_df = pd.read_csv(hist_f)
hist_df.shape
hist_df.head()


# ## Sanity Check
# 
# - All ```card_id```s in training and testing set
# - All ```merchant_id```s in merchant file
# - Nulls
# 

# In[ ]:


len(set(hist_df.card_id) - set(card_df.card_id) - set(test_df.card_id))


# There are no new ```card_id```s which are not there in training or test set.

# In[ ]:


set(hist_df.merchant_id) - set(merc_df.merchant_id)


# Except null values in ```merchant_id``` column from historical data, every ```merchant_id``` is present in merchant data. Lets explore nulls

# In[ ]:


temp_nulls = hist_df.isnull().sum()
temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls/hist_df.shape[0]})
temp


# Except the id cols, we'll not go into other columns. Among ids, only ```merchant_id``` column has 178,159 nulls, which is 0.0061% of the total data. We'll need to handle these when we are creating the final data

# # New Historical Data

# In[ ]:


new_hist_df = pd.read_csv(new_hist_f)
new_hist_df.shape
new_hist_df.head()


# ## Sanity Check
# 
# - All ```card_id```s in training and testing set
# - All ```merchant_id```s in merchant file
# - Nulls
# 

# In[ ]:


len(set(new_hist_df.card_id) - set(card_df.card_id) - set(test_df.card_id))


# No extra ```card_id``` in the new merchants file.

# In[ ]:


set(new_hist_df.merchant_id) - set(merc_df.merchant_id)


# New merchants file also has nulls in the ```merchant_id```. Other than that every ```merchant_id``` is present in the merchants file.

# In[ ]:


temp_nulls = new_hist_df.isnull().sum()
temp = pd.DataFrame({"nulls": temp_nulls, "null_percent": temp_nulls/new_hist_df.shape[0]})
temp


# Following the same pattern as the historical data, we will only discuss id columns here. Col ```merchant_id``` has 26,216 nulls which is 0.013% of the data.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




