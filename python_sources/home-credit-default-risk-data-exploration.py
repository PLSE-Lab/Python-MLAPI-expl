#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
sns.set(color_codes=True)
# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/application_train.csv")


# # Data Exploration

# In[ ]:


#Removing documennt flag columns from training data.
train = train.drop(train.filter(regex="FLAG"), axis=1)


# ## Let's have a look at gender differences

# In[ ]:


ax = sns.countplot(x="CODE_GENDER", hue="TARGET", data=train, palette="Greens_d")
ax.set_title("TARGET BY GENDER")


# In[ ]:


ax = sns.countplot(x="TARGET", data=train, palette="Greens_d")
zeros = train[train.TARGET==0].shape[0]
ones = train[train.TARGET==1].shape[0]
ax.set_title("TARGET COUNT (0={:d}, 1={:d})".format(zeros, ones))


# In[ ]:


gender_x = train.groupby(["CODE_GENDER","TARGET"])["CODE_GENDER"].count() 

F = gender_x["F"][1] / gender_x["F"].sum()
M = gender_x["M"][1] / gender_x["M"].sum()  
X = 0 # There are no XNA coded with target 1
print("F: {:.2f}% \nM: {:.2f}%\nXNA: {:.2f}%".format(F*100, M*100, X*100))


# This data suggests that a lot more females are applying, however it also suggests you're immediately 3% more likely to have payment difficulties if you identify as Male.
# 
# However is the target (1 = payment difficulties) independent of gender?

# In[ ]:


from scipy.stats import chi2_contingency
gender_cross_tab = pd.crosstab(index=train["CODE_GENDER"],columns=train['TARGET']);
display(chi2_contingency(gender_cross_tab))


# A CHI-squlare value of 920.79 and p < 0.005, means we cannot accept the null hypthoesis that gender and target are independent. Therefore there **__may__** be a relationship between them but this is not necessarily causal.
# 

# In[ ]:


k = 10 #number of variables for heatmap
corr = train.corr()
cols = corr.nlargest(k, 'TARGET')['TARGET'].index
cm = np.corrcoef(train[cols].values.T)
mask = np.zeros_like(cm, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1.25)
hm = sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# It doesn't look like there are any particularly __strong__ correlations between basic applicant data and the target.

# In[ ]:


def dfToCorrPlot(df, topK, corrTo):
    k = topK
    corr = df.corr()
    cols = corr.nlargest(k, corrTo)[corrTo].index
    cm = np.corrcoef(df[cols].values.T)
    mask = np.zeros_like(cm, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


# In[ ]:


bureau = pd.read_csv('../input/bureau.csv')
bureau_target = pd.merge(train[['SK_ID_CURR','TARGET']], bureau, on="SK_ID_CURR")


# In[ ]:


dfToCorrPlot(bureau_target, 10, "TARGET")


# Looks like there's a strong correlation between "DAYS_CREDIT" and "DAYS_CREDIT_UPDATE" which is defined as "How many days before current application did client apply for Credit Bureau credit" and "How many days before loan application did last information about the Credit Bureau credit come". This doesn't sound like something applicants have control over but rather maybe an insight into how long it takes from the information being received and being processed.

# 
