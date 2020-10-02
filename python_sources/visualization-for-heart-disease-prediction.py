#!/usr/bin/env python
# coding: utf-8

# # Which features are related with a hearth disease?
# <font color = 'blue'>
# Content:
# 
# 1. [LOAD AND CHECK DATA](#1)
# 1. [VARIABLE DESCRIPTION](#2)
#     * [Categorical Variable](#3)
#     * [Numerical Variable](#4)
# 1. [BASIC DATA ANALYSIS](#5)
# 1. [OUTLIER DETECTION](#6)
# 1. [MISSING VALUE](#7)
#     * [Find Missing Value](#7)
#     * [Fill Missing Value](#7)
# 1. [VISUALIZATION](#8)
#     * [Correlation Between Features vs Hearth Disease](#8)
#     * [thal -- target](#9)
#     * [ca -- target](#10)
#     * [slope -- target](#11)
#     * [exang -- target](#12)
#     * [cp -- target](#13)
#     * [oldpeak -- target](#14)
#     * [thalach -- target](#15)
#     * [slope -- oldpeak -- target](#16)
#     * [slope -- thalach -- target](#17)
#     * [exang -- cp -- target](#18)    
#     * [exang -- thalach -- target](#19)
#     * [cp -- thalach -- target](#20)
#     * [oldpeak -- thalach -- target](#21)
#     * [thalach -- age -- target](#22)
#         
#         

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = "1"></a><br>
# # LOAD AND CHECK DATA

# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')
print(plt.style.available)
plt.style.use('ggplot')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# <a id = "2"></a><br>
# # VARIABLE DESCRIPTION
# 
# 1. age: Age of patient
# 1. sex: Gender of patient (1:Male, 0:Female)
# 1. cp: chest pain type (4 values)
# 1. trestbps: resting blood pressure
# 1. chol: serum cholestoral in mg/dl
# 1. fbs: fasting blood sugar > 120 mg/dl
# 1. restecg: resting electrocardiographic results (values 0,1,2)
# 1. thalach: maximum heart rate achieved
# 1. exang: exercise induced angina (1: yes, 0: no)
# 1. oldpeak: ST depression induced by exercise relative to rest
# 1. slope: the slope of the peak exercise ST segment (values 0,1,2)
# 1. ca: number of major vessels (0-3) colored by flourosopy
# 1. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 1. target: Presence of heart disease (1: yes, 0: No)

# <a id = "3"></a><br>
# ## Categorical Variables
# * sex
# * cp
# * restecg
# * exang
# * slope
# * ca
# * thal
# * target

# In[ ]:


def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count    
    """
    # get feature
    var = data[variable]
    # caount number of categorical variable (value/sample)
    varValue = var.value_counts()
    
    #visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))


# In[ ]:


category = ["sex", "cp", "restecg", "exang", "slope", "ca", "thal", "target"]
for c in category:
    bar_plot(c)


# <a id = "4"></a><br>
# ## Numerical Variables
# * age
# * trestbps
# * chol
# * fbs
# * thalach
# * oldpeak

# In[ ]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(data[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()


# In[ ]:


numericVar = ["age", "trestbps", "chol", "fbs", "thalach", "oldpeak"]
for n in numericVar:
    plot_hist(n)


# <a id = "5"></a><br>
# # BASIC DATA ANALYSIS
# * sex - target
# * cp - target
# * restecg - target
# * exang - target
# * slope - target
# * ca - target
# * thal - target

# In[ ]:


# sex - target
data[["sex", "target"]].groupby(["sex"], as_index = False).mean().sort_values(by = "target", ascending =False)


# In[ ]:


# cp - target
data[["cp", "target"]].groupby(["cp"], as_index = False).mean().sort_values(by = "target", ascending =False)


# As you can see there is a correlation between chest pain and hearth disease.

# In[ ]:


# restecg - target
data[["restecg", "target"]].groupby(["restecg"], as_index = False).mean().sort_values(by = "target", ascending =False)


# In[ ]:


# exang - target
data[["exang", "target"]].groupby(["exang"], as_index = False).mean().sort_values(by = "target", ascending =False)


# In[ ]:


# slope - target
data[["slope", "target"]].groupby(["slope"], as_index = False).mean().sort_values(by = "target", ascending =False)


# In[ ]:


# ca - target
data[["ca", "target"]].groupby(["ca"], as_index = False).mean().sort_values(by = "target", ascending =False)


# In[ ]:


# thal - target
data[["thal", "target"]].groupby(["thal"], as_index = False).mean().sort_values(by = "target", ascending =False)


# <a id = "6"></a><br>
# # OUTLIER DETECTION

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        #Detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


data.loc[detect_outliers(data,["age", "trestbps", "chol", "fbs", "thalach", "oldpeak"])]


# No outliers detected in the data.

# <a id = "7"></a><br>
# # MISSING VALUE
# * Find Missing Value
# * Fill Missing Value

# In[ ]:


data.columns[data.isnull().any()]


# There isn't any missing value so we don't need to fill either.

# <a id = "8"></a><br>
# # VISUALIZATION
# * Correlation Between Features vs Hearth Disease

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(data[["age", "trestbps", "chol", "fbs", "thalach", "oldpeak",
                      "sex", "cp", "restecg", "exang", "slope", "ca", "thal", "target"]].corr(), annot = True)
plt.show()


# It seems that probability of hearth disease (target in this instance) has correlation with:
# * thal (-)
# * ca (-)
# * slope (+)
# * exang (-)
# * cp (+)
# * oldpeak (-)
# * thalac (+)
# 
# It is also seen that:
# * slope has correlation with:
#     * oldpeak (-)
#     * thalac (+)
# * exang has correlation with:
#     * cp (-)
#     * thalac (-)
# * cp has correlation with:
#     * thalac (+)
# * oldpeak has correlation with:
#     * thalac (+)
# * thalac has correlation with:
#     * age (-)
# 
# Now we will visualize these relations

#  <a id = "9"></a>
#  * thal -- target
# 

# In[ ]:


g = sns.factorplot(x = "thal", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()


# * Patiens whose thal = 2 have a very high heart disease probability. 
# * Also thal = 0 patients have a higher risk then thal = 1 or 3

#  <a id = "10"></a><br>
#  * ca -- target
# 

# In[ ]:


g = sns.factorplot(x = "ca", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()


# * ca = 0 or 4 patients have a higher risk then ca = 1, 2 or 3

# <a id = "11"></a><br>
# * slope -- target
# 
# 

# In[ ]:


g = sns.factorplot(x = "slope", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()


# * slope = 2 patients have a higher risk then slope = 0 or 1

# <a id = "12"></a><br> 
# * exang -- target

# In[ ]:


g = sns.factorplot(x = "exang", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()


# * exang = 1 patients have a higher risk then exang = 0

# <a id = "13"></a><br>
# * cp -- target

# In[ ]:


g = sns.factorplot(x = "cp", y = "target", data = data, kind = "bar", size = 6)
g.set_ylabels("Disease Probability")
plt.show()


# Patiens who have chest pain, have a very high probability of a hearth disease

# <a id = "14"></a><br>    
# * oldpeak -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target", size = 6)
g.map(sns.distplot, "oldpeak", bins = 25)
plt.show()


# * For o<oldpeak<2, there is a higher risk of disease

# <a id = "15"></a><br>  
# * thalach -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target")
g.map(sns.distplot, "thalach", bins = 25)
plt.show()


# * As thalach rises over 150, the risk increases

#  <a id = "16"></a><br>  
#  * slope -- oldpeak -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "slope", size = 3)
g.map(plt.hist, "oldpeak", bins = 25)
g.add_legend()
plt.show()


#  <a id = "17"></a><br>  
#  * slope -- thalach -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "slope", size = 3)
g.map(plt.hist, "thalach", bins = 25)
g.add_legend()
plt.show()


# * The risk is higher for slope=2 and thalach>150 patients
# * The risk is lower for slope=1 and thalach>150 patients

# <a id = "18"></a><br>  
# * exang -- cp -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "exang", size = 4)
g.map(plt.hist, "cp", bins = 25)
g.add_legend()
plt.show()


#  <a id = "19"></a><br>  
#  * exang -- thalach -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "exang", size = 4)
g.map(plt.hist, "thalach", bins = 25)
g.add_legend()
plt.show()


#  <a id = "20"></a><br>  
#  * cp -- thalach -- target

# In[ ]:


g = sns.FacetGrid(data, col = "target", row = "cp", size = 2)
g.map(plt.hist, "thalach", bins = 25)
g.add_legend()
plt.show()


#  <a id = "21"></a><br>  
#  * oldpeak -- thalach -- target

# In[ ]:


g = sns.FacetGrid(data, col="target", size = 8)
g.map(plt.scatter, "oldpeak", "thalach", edgecolor="w")
g.add_legend()
plt.show()


# * Heart disease risk increases especially when oldpeak < 2 and thalach > 150

#  <a id = "22"></a><br>  
#  * thalach -- age -- target

# In[ ]:


g = sns.FacetGrid(data, col="target", size = 8)
g.map(sns.kdeplot, "age", "thalach", edgecolor="w")
g.add_legend()
plt.show()


# * The disease risk inceases at age between 40-60 and thalach between 150-185
