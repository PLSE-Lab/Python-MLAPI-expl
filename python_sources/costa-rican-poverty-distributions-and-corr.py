#!/usr/bin/env python
# coding: utf-8

# ## Costa Rican Poverty - Distributions and Correlations
# _By Nick Brooks, July 2018_
# 
# Since most of the feature's true identities are concealed, this notebook focuses on automating the visualization of the distributions and top correlations of around 140 features.

# In[ ]:


print("Packages and Load Data..")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing

train = pd.read_csv("../input/train.csv", index_col = "Id")
traindex = train.index
test_df = pd.read_csv("../input/test.csv", index_col = "Id")

test_df["Target"] = np.nan
train["Target"] = train["Target"].map({1: "Extreme Poverty",
                                       2: "Moderate Poverty",
                                       3: "Vulnerable Households",
                                       4: "Non-Vulnerable Households"})
print("Percent Class Distribution:")
print(train["Target"].value_counts(normalize=True)*100)

df = pd.concat([train,test_df],axis=0)

# Label Encode
lbl = preprocessing.LabelEncoder()
print([x for x in train.loc[:,train.dtypes == "object"].columns if x not in "Target"])
for col in [x for x in train.loc[:,train.dtypes == "object"].columns if x not in "Target"]:
    df[col] = lbl.fit_transform(df[col].astype(str))
train = df.loc[train.index,:]
test_df = df.loc[test_df.index,:]

df = pd.concat([train,test_df],axis=0)


# In[ ]:


# Seperating Variables by Number of Unique Values
target_var = "Target"
df_nnunique = df.nunique().reset_index().rename(columns = {"index": "cols",0:"unique_num"})
binary = list(df_nnunique.loc[df_nnunique.unique_num <= 2, "cols"]) + ["Target"]
continuous = list(df_nnunique.loc[df_nnunique.unique_num > 10, "cols"]) + ["Target"]
few_categories = list(df_nnunique.loc[(df_nnunique.unique_num >= 3)
                                      & (df_nnunique.unique_num <= 10) , "cols"])

print("Number of Binary Variables: ", len(binary)-1)
print("Number of Continous Variables: ", len(continuous)-1)
print("Number of Non-Binary, Categorical Variables: ", len(few_categories))


# ## **Continuous Variables** <br>

# In[ ]:


# Melt
melt_df = pd.melt(df.loc[traindex,continuous], id_vars="Target")
grid = sns.FacetGrid(melt_df,col="variable", hue=target_var, col_wrap=3,
                     size=4.0, aspect=1.3, sharex=False, sharey=False)
grid.map(sns.kdeplot, "value")
grid.set_titles(size=25)
grid.add_legend();
plt.show()


# ## **Investigate Top Correlations:** <br>

# In[ ]:


def rank_correlations(df, figsize=(12,20), n_charts = 18, polyorder = 2, asc = False):
    # Rank Correlations
    continuous_rankedcorr = (df
                             .corr()
                             .unstack()
                             .sort_values(ascending=asc)
                             .drop_duplicates().reset_index())
    continuous_rankedcorr.columns = ["f1","f2","Absoluate Correlation Coefficient"]   

    # Plot Top Correlations
    top_corr = [(x,y) for x,y in list(continuous_rankedcorr.iloc[:, 0:2].values) if x != y]
    f, axes = plt.subplots(int(n_charts/3),3, figsize=figsize, sharex=False, sharey=False)
    row = 0
    col = 0
    for (x,y) in top_corr[:n_charts]:
        if col == 3:
            col = 0
            row += 1
        g = sns.regplot(x=x, y=y, data=df, order=polyorder, ax = axes[row,col])
        axes[row,col].set_title('Correlation for\n{} and {}'.format(x, y))
        col += 1
    plt.tight_layout(pad=0)
    plt.show()
print("rank_correlations Plot Function Ready..")


# **Top Positive Correlations:** <Br>

# In[ ]:


rank_correlations(df = df.loc[traindex,continuous])


# **Top Negative Correlations:**

# In[ ]:


rank_correlations(df = df.loc[traindex,continuous], asc=True, polyorder = 2)


# ## Categorical Features
# 
# **Features with 3 to 10 categories:**

# In[ ]:


# Melt
melt_df = pd.melt(df.loc[traindex,few_categories], id_vars="Target")
melt_df = (melt_df.groupby(['Target', "variable"])['value']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index())
melt_df.value = melt_df.value.astype(int)

# Factor Plot
grid = sns.factorplot(x="value", y="percentage", hue= "Target", kind="bar",
                col="variable", data=melt_df, col_wrap=2 , size=4.0, aspect=1.3,
                sharex=False, sharey=False)
grid.set_titles(size=20)
#grid.add_legend();
plt.show()


# **Top Positive Correlations:** <br>

# In[ ]:


rank_correlations(df = df.loc[traindex,few_categories], asc=False, polyorder = 1, figsize=(10,5), n_charts = 6)


# **Top Negative Correlations:** <br>

# In[ ]:


rank_correlations(df = df.loc[traindex,few_categories], asc=True, polyorder = 1, figsize=(10,5), n_charts = 6)


# ## **Binary Variables:** <br>

# In[ ]:


# Melt
melt_df = pd.melt(df.loc[traindex,binary], id_vars="Target")
melt_df = (melt_df.groupby(['Target', "variable"])['value']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index())
melt_df.value = melt_df.value.astype(int)

# Factor Plot
grid = sns.factorplot(x="value", y="percentage", hue= "Target", kind="bar",
                col="variable", data=melt_df, col_wrap=4 , size=4.0, aspect=1.3,
                sharex=False, sharey=False)
grid.set_titles(size=20)
grid.add_legend();
plt.show()


# **Binary Average by Poverty Level:** <br>
# This plot enables us to glance at how the binary values interact over the poverty levels. A value of .9 means that the feature is 90% positive for a certain poverty level. <br>
# This technique is good because it normalizes for the unequal class occurences.

# In[ ]:


cm = sns.light_palette("purple", as_cmap=True)
binary_means = pd.pivot_table(pd.melt(df.loc[traindex,binary], id_vars=target_var), values="value",
               index="variable",columns=["Target"],
               aggfunc = np.mean)
binary_means = binary_means[["Extreme Poverty", "Moderate Poverty", "Vulnerable Households", "Non-Vulnerable Households"]]
binary_means.style.background_gradient(cmap = cm)


# This plot is along the same lines as the last one, but it is not normalized for unequal class occurence.

# In[ ]:


# Melt
melt_df = pd.melt(df.loc[traindex,binary], id_vars=target_var)
binary_data = pd.pivot_table(melt_df, values="value", index="variable",columns=["Target"], aggfunc = np.sum)
binary_data = binary_data[["Extreme Poverty", "Moderate Poverty", "Vulnerable Households", "Non-Vulnerable Households"]]

f, ax = plt.subplots(figsize=[10,10])
sns.heatmap(binary_data, annot=False, fmt=".2f",cbar_kws={'label': 'Occurence'},cmap="YlGnBu",ax=ax)
ax.set_title("Binary Variable Positive Occurence Count by Poverty Level")
plt.show()


# **Correlations:** <Br>

# In[ ]:


def binary_heatmap_rank_correlations(df, figsize=(12,20), n_charts = 18, asc = False):
    # Rank Correlations
    continuous_rankedcorr = (df
                             .corr()
                             .unstack()
                             .sort_values(ascending=asc)
                             .drop_duplicates().reset_index())
    continuous_rankedcorr.columns = ["f1","f2","Absoluate Correlation Coefficient"]   

    # Plot Top Correlations
    top_corr = [(x,y) for x,y in list(continuous_rankedcorr.iloc[:, 0:2].values) if x != y]
    f, axes = plt.subplots(int(n_charts/3),3, figsize=figsize, sharex=False, sharey=False)
    row = 0
    col = 0
    for (x,y) in top_corr[:n_charts]:
        if col == 3:
            col = 0
            row += 1
        axes[row,col] = sns.heatmap(pd.crosstab(df[x], df[y]),
                    annot=False, fmt=".2f",cbar_kws={'label': 'Count'},cmap="plasma",ax=axes[row,col])
        axes[row,col].set_title('Binary Overlap Count for\n{} and {}'.format(x, y))
        col += 1
    plt.tight_layout(pad=0)
    plt.show()
print("binary_heatmap_rank_correlations Plot Function Ready..")


# **Positive Correlations:**

# In[ ]:


binary_heatmap_rank_correlations(df.loc[traindex,binary], figsize=(12,6), n_charts = 6, asc = False)


# **Negative Correlations:** <br>

# In[ ]:


binary_heatmap_rank_correlations(df.loc[traindex,binary], figsize=(12,6), n_charts = 6, asc = True)


# Thanks for reading, I am open to suggestions and improvements.
