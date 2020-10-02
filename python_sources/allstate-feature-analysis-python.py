#!/usr/bin/env python
# coding: utf-8

# **This is my first kernel submission on kaggle. I got the motivation from other EDA notebooks for this competition.**

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:


print("Train data dimensions: ", train_data.shape)
print("Test data dimensions: ", test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


print("Number of missing values",train_data.isnull().sum().sum())


# This is a good news as there are not missing values :)

# **Lets analyze the distribution of continuous features:**

# In[ ]:


train_data.describe()


# In[ ]:


contFeatureslist = []
for colName,x in train_data.iloc[1,:].iteritems():
    #print(x)
    if(not str(x).isalpha()):
        contFeatureslist.append(colName)


# In[ ]:


print(contFeatureslist)


# In[ ]:


contFeatureslist.remove("id")
contFeatureslist.remove("loss")


# ### Box plots for continuous features

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(13,9))
sns.boxplot(train_data[contFeatureslist])


# As we  can see, some of the features like cont13, cont14 etc. are highly skewed. we might need to normalize these features before running any algorithms

# ### Correlation between continuous features

# In[ ]:


# Include  target variable also to find correlation between features and target feature as well
contFeatureslist.append("loss")


# In[ ]:


correlationMatrix = train_data[contFeatureslist].corr().abs()

plt.subplots(figsize=(13, 9))
sns.heatmap(correlationMatrix,annot=True)

# Mask unimportant features
sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)
plt.show()


# ### Analysis of loss feature

# In[ ]:


plt.figure(figsize=(13,9))
sns.distplot(train_data["loss"])
sns.boxplot(train_data["loss"])


# Here, we can see loss is highly right skewed data. This happened because there are many outliers in the data that we ca see from box plot. Lets apply log to see if we can get normal distribution

# In[ ]:


plt.figure(figsize=(13,9))
sns.distplot(np.log1p(train_data["loss"]))


# So we got normal distribution by applying logarithm on loss function

# Bang. Finally we got normal distribution, so we can train model using target feature as log of loss. This way we don't have to remove outliers.

# In[ ]:


catCount = sum(str(x).isalpha() for x in train_data.iloc[1,:])
print("Number of categories: ",catCount)


# There are 116 categories with non alphanumeric values, most of the machine learning algorithms doesn't work with alpha numeric values. So, lets convert it into numeric values

# In[ ]:


catFeatureslist = []
for colName,x in train_data.iloc[1,:].iteritems():
    if(str(x).isalpha()):
        catFeatureslist.append(colName)


# **Unique categorical values per each category**

# In[ ]:


print(train_data[catFeatureslist].apply(pd.Series.nunique))


# ### Convert categorical string values to numeric values

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


for cf1 in catFeatureslist:
    le = LabelEncoder()
    le.fit(train_data[cf1].unique())
    train_data[cf1] = le.transform(train_data[cf1])


# In[ ]:


train_data.head(5)


# In[ ]:


sum(train_data[catFeatureslist].apply(pd.Series.nunique) > 2)


# ### Analysis of categorical features with levels between 5-10

# In[ ]:


filterG5_10 = list((train_data[catFeatureslist].apply(pd.Series.nunique) > 5) & 
                (train_data[catFeatureslist].apply(pd.Series.nunique) < 10))


# In[ ]:


catFeaturesG5_10List = [i for (i, v) in zip(catFeatureslist, filterG5_10) if v]


# In[ ]:


len(catFeaturesG5_10List)


# In[ ]:


ncol = 2
nrow = 4
try:
    for rowIndex in range(nrow):
        f,axList = plt.subplots(nrows=1,ncols=ncol,sharey=True,figsize=(13, 9))
        features = catFeaturesG5_10List[rowIndex*ncol:ncol*(rowIndex+1)]
        
        for axIndex in range(len(axList)):
            sns.boxplot(x=features[axIndex], y="loss", data=train_data, ax=axList[axIndex])
                        
            # With original scale it is hard to visualize because of outliers
            axList[axIndex].set(yscale="log")
            axList[axIndex].set(xlabel=features[axIndex], ylabel='log loss')
except IndexError:
    print("")


# ### Correlation between categorical variables

# In[ ]:


filterG2 = list((train_data[catFeatureslist].apply(pd.Series.nunique) == 2))
catFeaturesG2List = [i for (i, v) in zip(catFeatureslist, filterG2) if v]
catFeaturesG2List.append("loss")


# In[ ]:


corrCatMatrix = train_data[catFeaturesG2List].corr().abs()

s = corrCatMatrix.unstack()
sortedSeries= s.order(kind="quicksort",ascending=False)

print("Top 5 most correlated categorical feature pairs: \n")
print(sortedSeries[sortedSeries != 1.0][0:9])


# Note: We should do chi-square test on categorical features to find independence. I have done pearson correlation which is measure to find association between features

# *Thank you. I hope this will be useful for you. Please share your comments/feedback*

# More EDA analysis for this competition: (Thanks for the motivation)  
# https://www.kaggle.com/nminus1/allstate-claims-severity/allstate-eda-python  
# https://www.kaggle.com/dmi3kno/allstate-claims-severity/allstate-eda  
