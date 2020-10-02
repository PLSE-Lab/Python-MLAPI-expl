#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing all libraries required for Anova test**

# In[ ]:


from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from scipy.stats import shapiro
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from statsmodels.graphics.factorplots import interaction_plot
from pandas.plotting import scatter_matrix


# **Loading Dataset**

# In[ ]:


iris=load_iris()


# In[ ]:


iris.target


# In[ ]:


dataframe_iris=pd.DataFrame(iris.data,columns=['sepalLength','sepalWidth','petalLength','petalWidth'])


# In[ ]:


dataframe_iris.shape


# In[ ]:


dataframe_iris1=pd.DataFrame(iris.target,columns=['target'])


# In[ ]:


dataframe_iris1.shape


# **Iris Data Visualization**

# In[ ]:


scatter_matrix(dataframe_iris[['sepalLength', 'sepalWidth', 'petalLength','petalWidth']],figsize=(15,10))  
plt.show()


# In[ ]:


ID=[]
for i in range(0,150):
    ID.append(i)


# In[ ]:


dataframe=pd.DataFrame(ID,columns=['ID'])


# In[ ]:


dataframe_iris_new=pd.concat([dataframe_iris,dataframe_iris1,dataframe],axis=1)


# In[ ]:


dataframe_iris_new.columns


# In[ ]:


fig = interaction_plot(dataframe_iris_new.sepalWidth,dataframe_iris_new.target,
                       dataframe_iris_new.ID,colors=['red','blue','green'], ms=12)


# In[ ]:


dataframe_iris_new.info()


# In[ ]:


dataframe_iris_new.describe()


# **Anova test:
# Analysis of variance(anova) is a statistical technique that is used to check if means of two or more groups are statistically different from each other.
# Consider a group of three samples means for sepal_width_cm from the iris dataset.
# Our goal is to determine that each group's mean value are statistically different from the other's and to do this we need to evaluate the variability between each of the mean values**

# In[ ]:


##############################################


# **Anova hypothesis**
# 
# **To implement Anova test we have to create null hypothesis and alternate hypothesis**
# 
# **Null hypothesis=sample means are equal**
# 
# **Alternate hypothesis=sample means are not equal**

# In[ ]:


##############################################


# In[ ]:


print(dataframe_iris_new['sepalWidth'].groupby(dataframe_iris_new['target']).mean())


# In[ ]:


dataframe_iris_new.mean()


# **But Anova also analyze variance of differnt groups and evaluate whether we have to reject 
# The null hypothesis and accept alternate hypothesis
# For that Anova calculate f-value and p-value. 
# P-value:-p-value is used to evaluate hypothesis results.P-value is a number between 0 and 1.
# If p-value<0.05 we have to reject null hypothesis
# And p-value>0.05 we have to accept null hypothesis.
# F-value:-f-value is the ratio of variance between groups and variance within groups.
# If f-value is close to 1 then we say that our null hypothesis is true i.e samples have equal mean and
# F-value is greater than 1 then samples have quite different mean values.**

# **Before performing anova test Anova assumes following points. 
# Anova assumptions:
# 1.Normality:-samples are taken from normal distribution.
# To check whether data is normally distributed or not Anova use shapiro-wilks test
# 2.Each sample is independent of other sample.
# 3.Variance:- variance should be same.
# To check whether variance between groups are equal Anova use levene/barlett test.**

# In[ ]:


##############################################


# **Check normal distribution of data(shapiro-wilk test)
# Null hypothesis:- data is drawn from normal distribution
# Alternate hypothesis:- data is not drawn from normal distribution**

# In[ ]:


stats.shapiro(dataframe_iris_new['sepalWidth'][dataframe_iris_new['target']])


# **Interpretation:-As p-value is significant we reject null hypothesis.**

# In[ ]:


##############################################


# **Check equality of variance between groups(levene/bartlett test)**

# In[ ]:


p_value=stats.levene(dataframe_iris_new['sepalWidth'],dataframe_iris_new['target'])


# In[ ]:


p_value


# **Interpretation:- As p-value is significant we reject null hypothesis**

# In[ ]:


##############################################


# **Types of Anova:**
# **One-way Anova:-one way Anova is used to compare means of two or more samples using f-value and p-value.
# Two-way Anova:-in two way Anova, data are classified on the basis of two factors.
# Difference between one-way Anova and two-way Anova.
# One way anova compares three or more than three categorical gropus ,compare their means and to evaluate whether there is difference between them.
# Hypothesis of one way-anova:
# Null hypothesis(h0):-null hypothesis is that all groups of mean are equal,there is no difference between them.
# Alternate hypothesis(h1):-alternate hypothesis states that there is difference between mean.
# Two-way Anova compares means of three or more groups of data, where two independent variables are considered.
# The hypothesis of two way Anova is same as one-way Anova.**

# **Example:-we took iris dataset for Anova testing.Here, we have only one independent variable i.e. Species(iris-setosa,iris-versicolor,iris-virginica) which are in categorical and we took sepal width as a continous variable. For exmaple, if someone wants looked at sepal width in iris-setosa,iris-versicolor and iris virginica,there would be three species analyzed and therfore three groups to the analysis.
# In iris dataset we have only one independent variable i.e. Species so we are doing one-way Anova testing.**

# In[ ]:


##############################################


# In[ ]:


F_value,P_value=stats.f_oneway(dataframe_iris_new['sepalWidth'],dataframe_iris_new['target'])


# In[ ]:


print("F_value=",F_value,",","P_value=",P_value)


# In[ ]:


if F_value>1.0:
    print("******SAMPLES HAVE DIFFERENT MEAN******")
else:
    print("******SAMPLES HAVE EQUAL MEAN******")


# **Looking at f-value, we say that samples have different mean therefore we conclude that samples have different mean**

# In[ ]:


if P_value<0.05:
    print("******REJECT NULL HYPOTHESIS******")
else:
    print("******ACCEPT NULL HYPOTHESIS******")


# **As p-value obtained from one way Anova analysis states that there is significant difference between samples mean.**

# In[ ]:


##############################################


# **There are different post-hoc test to verify results.
# 1.Lsd test.
# 2.Tukey's hsd test
# 3.Scheffe's test
# Here we are going to perform tukey's hsd test to evaluate our hypothesis.
# To know pairs of significant different groups.Tukey hsd perform multiple pairwise comparions between groups.**

# In[ ]:


tukey = pairwise_tukeyhsd(endog=dataframe_iris_new['sepalWidth'], groups=dataframe_iris_new['target'], alpha=0.05)
print(tukey)


# **Interpretation:-Above tukey results show that all pairwise comparison for different groups rejects null hypothesis and indicates statistical significant difference.**

# In[ ]:


##############################################


# **CONCLUSION:-as p-value obtained from Anova test and different test states that we have to reject null hypothesis.
# And accept alternate hypothesis that atleast two groups means are statistically different from each other.**
