#!/usr/bin/env python
# coding: utf-8

# We're provided very little information about the types of financial instruments included in the dataset. Are they stocks?  Options? Bonds?  A mixture?  Even if the instruments are of similar type it is unknown whether they are all from the same market or from a set of global exchanges.

# If different ids represent instruments of differing class (and possibly from different markets), then one might expect that sets of features will be systematically missing between the different instrument types.  As an extreme example stock features may include underlying company fundamentals, while government bonds would have a dramatically different set of features.  Is there evidence of systematically missing features, and are those reflected in the target variable distribution?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
cmap=plt.cm.get_cmap('Blues')


# In[ ]:


# read in the data
with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# ## Instruments Present Throughout Entire Training Period
# 
# Several actions could complicate the picture with missing values:
# - It's possible that new features have been introduced to the dataset over time.
# - Moving averages could introduce missing values at the start of a collection period.
# - New instruments (especially options) could have been added over time.
# 
# To simplify the initial analysis, only instruments that are present throughout the training set period will be considered.

# In[ ]:


# remove ids that do not have an entry for every timestamp
id_counts = df.id.value_counts()
ids = np.setdiff1d(df.id.unique(), id_counts[id_counts != len(df.timestamp.unique())].index)
enduring = df[df.id.isin(ids)]


# In[ ]:


print('Number of instruments in the full training set:',len(df.id.unique()))
print('Number of instruments in the reduced training set:',len(enduring.id.unique()))


# This cuts the number of instruments by about 2/3.

# ## Missing Features
# 
# Next, the fraction of missing values is calculated for each feature and instrument.  

# In[ ]:


def GetNaFraction(df):
    # form n x m dataframe - one row per id, one column per feature
    # each entry counts the fraction of NAs for the feature for one asset
    nas = pd.DataFrame(index = df.id.unique(), columns=df.columns).T
    for id in df.id.unique():
        nas[id] = df[df.id == id].isnull().mean()
    return nas.T


# In[ ]:


# Calculate the fraction of NAs per instrument per feature
enduring_nas = GetNaFraction(enduring)
enduring_nas.drop(['id', 'timestamp', 'y'], axis=1, inplace=True)
overall_percentages = enduring_nas.mean(axis=0).sort_values(ascending=False)
enduring_nas.head()


# We can see that there are a number of features that are sometimes fully populated and sometimes completely missing eg derived_2, derived_4 for id 26 vs id 31.

# In[ ]:


# Plot an overview
fig, ax = plt.subplots()
idx = np.arange(len(overall_percentages))
ax.barh(idx, overall_percentages, color='cyan', ecolor='black')
ax.set_xlabel("Average fraction of missing values")
ax.set_title("Fraction of missing values in each feature")
plt.margins(0.01)
plt.xlim(xmin=0);


# ### Features with no NAs
# 
# Several features are always present (at least this for this reduced dataset).

# In[ ]:


no_na_ids = overall_percentages[overall_percentages==0].index
print(sorted(no_na_ids))


# ### Features with Consistent NA Levels
# The following have the same fraction of NAs for all the items that are present from start to finish of the dataset.

# In[ ]:


constant_na_ids = enduring_nas.columns[enduring_nas.apply(pd.Series.nunique) == 1]
constant_na_ids = np.setdiff1d(constant_na_ids, no_na_ids)
print(constant_na_ids )


# In[ ]:


constant_nas=enduring[np.append('timestamp', constant_na_ids )].copy()
for col in constant_nas.columns[1:]:
    constant_nas.loc[:,col] = constant_nas[col].isnull().astype(int)

# get the average number of nas per timestamp
constant_nas = constant_nas.groupby(['timestamp']).mean()

# sort the columns shorter->longer moving averages
new_cols = list(constant_nas.apply(sum, axis=0).sort_values(ascending=True).index)

fig = plt.figure()
plt.gca().invert_yaxis()   # flip so that older timestamps are at the top
plt.pcolormesh(constant_nas[new_cols], cmap=cmap)
plt.colorbar(shrink=0.5)
plt.title("Fraction of NAs per Feature per Timestamp")
plt.ylabel("Timestamp")
plt.xlabel("Feature")
plt.axis('tight');


# These appear to be some form of moving average. NAs appear consistently in the early timestamps, and for increasing durations.  (Side note: I believe there are other features that look like moving averages that are zero-padded at the beginning.)

# Let's take a look at how the missing values correlate with each other.  In the diagrams below, each row is an instrument and each column is a single feature.

# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.pcolormesh(enduring_nas,cmap=cmap) 
plt.colorbar(shrink=0.5)
plt.title("Fraction of NAs per Feature per Instrument")
plt.ylabel("Instruments")
plt.xlabel("Features")
plt.axis('tight');


# There's clearly a pattern here -- it looks like the same groupings of features are missing across multiple instruments. Let's change the ordering to try to bring it out further.

# In[ ]:


# resort the matrix to show groupings more strongly
cols = overall_percentages.index.tolist()
cols_to_sort=['fundamental_6', 'fundamental_24']
fig = plt.figure(figsize=(8, 8))
plt.pcolormesh(enduring_nas[cols].sort_values(cols_to_sort),cmap=cmap)
plt.colorbar(shrink=0.5)
plt.title("Fraction of NAs per Feature per Instrument (sorted)")
plt.ylabel("Instruments (sorted)")
plt.xlabel("Features (sorted)")
plt.axis('tight');


# It's possible to get pretty good separation by sorting the instrument 'missingness' with just two features (here fundamental_6' and 'fundamental_24.)'  (The columns(features) are also sorted left to right from most to least missing.)
# 
# There may be some more subtle structures but three main groupings can be clearly seen:
# 
#  - a set of five or six features missing as a group including fundamental_6, fundamental_1, fundamental_61, fundamental_57 and fundamental_26
#  - a set of about thirty two features missing as a group
#  - instruments with very few features missing

# ## Y-Value Distributions
# 
# Do these groupings show any difference with regard to the distribution of the target variable?

# In[ ]:


# plot distribution of the y-values
# Assume that missing values are dominated by the systematic effects
enduring.y[enduring.fundamental_6.isnull()].plot.kde(color='Blue', label='f6 null')
enduring.y[enduring.fundamental_24.isnull()].plot.kde(color='Red', label='f24 null')
enduring.y[enduring.fundamental_6.notnull() & enduring.fundamental_24.notnull()].plot.kde(color='Orange', label='f6,f24 not null')
plt.title("Density of Y-Values for Different Groupings")
plt.xlabel("Y")
plt.legend();


# There do appear to be differences in 'y' distribution, most notably for the first group with 5-6 missing features.   Surprisingly there is less difference between instruments with nearly all features populated vs instruments with the larger block of missing features.

# ### Fundamental_5
# 
# The feature missing most often is 'fundamental_5', however there doesn't seem to be a strong correlation with any other groupings.  Considered alone, 'fundamental_5' shows only a small effect on y-values.

# In[ ]:


enduring.y[enduring.fundamental_5.notnull()].plot.kde(color='Red', label='f5 not null')
enduring.y[enduring.fundamental_5.isnull()].plot.kde(color='Blue', label='f5 null')
plt.title("Density of Y-Values: 'fundamental_5' Missing/Not Missing")
plt.xlabel("Y")
plt.legend();


# ## Conclusions
# 
# It appears that there are features that are missing not at random. Patterns exist where groups of features are missing for many instruments. This would be consistent with the dataset containing financial vehicles of different types, but that remains speculation. There are relatively few such groupings.  The presence of systematic missing features does appear to be associated with variation in the target y-values.
