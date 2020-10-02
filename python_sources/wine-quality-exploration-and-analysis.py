#!/usr/bin/env python
# coding: utf-8

# # Exploration and Analysis of Wine Quality

# ## **Introduction**
# 
# The following notebook contains the steps enumerated below for analyzing characteristics of red and white variants of the Portuguese "Vinho Verde" wine. Quality is based on sensory scores (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent).<br> 
# Data is available at: https://archive.ics.uci.edu/ml/datasets/Wine+Quality <br>
# <br> 
# 1. [Import Data & Python Packages](#1-bullet) <br><br>
# 2. [Assess Data Quality & Missing Values](#2-bullet)<br><br>
# 3. [Exploratory Data Analysis - ggplot for python](#3-bullet) <br>
#     * [3.1 Red vs. White wines](#3.1-bullet) <br>
#     * [3.2 Facetplot](#3.2-bullet) <br><br>
# 4. [Correlation Heatmaps - seaborn](#4-bullet) <br>
#     * [4.1 Heat Map - Red Wine](#4.1-bullet) <br>
#     * [4.2 Heat Map - White Wine](#4.2-bullet) <br>
#     * [4.3 Biggest Differences between White and Red Correlations](#4.3-bullet) <br><br>
# 5. [Predicting Quality: Linear Regression](#5-bullet) <br>
#     * [5.1 80-20 Split of Training and Hold-Out Data](#5.1-bullet) <br>
#     * [5.2 60-40 Split of Training and Hold-Out Data](#5.2-bullet) <br>
#     * [5.3 Segmented LinReg (White & Red Separate Models)](#5.3-bullet) <br><br>
# 6. [Alternate Approach 1: Support Vector Machine](#6-bullet) <br>

# ### 1. Import Data & Python Packages <a class="anchor" id="1-bullet"></a>

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

from ggplot import *


# In[ ]:


### Load wine quality data into Pandas
df_red = pd.read_csv("../input/winequality_red.csv")

df_white = pd.read_csv("../input/winequality_white.csv")


# In[ ]:


df_red["color"] = "R"


# In[ ]:


df_white["color"] = "W"


# In[ ]:


df_all=pd.concat([df_red,df_white],axis=0)


# In[ ]:


df_all.head()


# In[ ]:


df_white.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)


# In[ ]:


df_red.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)


# In[ ]:


df_all.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)


# In[ ]:


df_all.head()


# In[ ]:


df = pd.get_dummies(df_all, columns=["color"])


# ## 2. Assess Data Quality & Missing Values <a class="anchor" id="2-bullet"></a>

# In[ ]:


df_all.isnull().sum()


# #### There are no missing values in this dateset.  Victory!

# In[ ]:


df_all.describe()


# #### Nothing looks particularly out of place (e.g. no negative values in pH).  Interestingly, no wines scored below a 3 or above a 9 (no perfect 10/10 scores). 

# ## 3. Exploratory Data Analysis - ggplot for python

# ### 3.1 Red vs. White wines

# In[ ]:


print("white mean = ",df_white["quality"].mean())
print("red mean =",df_red["quality"].mean())


# In[ ]:


d = {'color': ['red','white'], 'mean_quality': [5.636023,5.877909]}
df_mean = pd.DataFrame(data=d)
df_mean


# In[ ]:


ggplot(df_mean, aes(x='color', weight='mean_quality')) + geom_bar() +    labs(y = "Average Quality", title = "Average Quality by Wine Color")


# In[ ]:


ggplot(df_all, aes(x='fixed_acidity', y='pH', color='color',size='quality')) + geom_point()


# ## 3.2 Facet Plot 

# Testing out ggplot's facet plot capabilities on a few of our continuous variables (treating quality as categorical for the purposes of wrapping several plots).

# In[ ]:


ggplot(df_all, aes(x='alcohol', y='residual_sugar', color='color')) + geom_point() +    facet_wrap('quality', ncol=2) + scale_color_brewer(type = 'qual', palette = 'Dark2')


# In[ ]:


ggplot(df_all, aes(x='fixed_acidity', y='volatile_acidity', color='color')) + geom_point() + facet_wrap('quality', ncol=2)


# ## 4. Correlation Heat Maps - Seaborn <a class="anchor" id="4-bullet"></a>

# ## 4.1 Red Wine <a class="anchor" id="4.1-bullet"></a>

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Wine Characteristic Correlation Heatmap (Reds)")
corr = df_red.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Reds")
plt.show()


# ## 4.2 White Wine <a class="anchor" id="4.2-bullet"></a>

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Wine Characteristic Correlation Heatmap (Reds)")
corr = df_red.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Purples")


# ## 4.3 Biggest Differences between White and Red Correlations <a class="anchor" id="4.3-bullet"></a>

# ### Red Pearson's Correlation - White Pearson's Correlation <br>
# There are some noteable differences in the way that certain variables interact depending on the variety of wine.  The darker the square, the larger the difference that interaction is between Red and White wines. <br><br>
# For instance, the correlation between alcohol and sugar content is much higher for Red wines than it is for white wines (boozy reds have more sugar than less boozy reds, while boozy whites have *less* sugar than less boozy whites). Closer inspection indicates that the correlation between sugar and alcohol is positive for Red wines (weak positive, 0.042), but it is much more strongly negative for White wines (-0.45). <br><br>

# In[ ]:


df_r_corr=df_red.corr()
df_w_corr=df_white.corr()


# In[ ]:


df_r_corr


# In[ ]:


df_w_corr


# In[ ]:


diff_corr = df_r_corr - df_w_corr


# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Differences between Red and White Wines")
corr = diff_corr
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="coolwarm")


# ## 5. Predicting Quality: Linear Regression <a class="anchor" id="5-bullet"></a>

# ## 5.1 80-20 Split of Training and Hold-Out Data <a class="anchor" id="5.1-bullet"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=4) 
## add a starting random point (4) so results can be reproduced 


# In[ ]:


results1 = smf.ols('quality ~ total_sulfur_dioxide + free_sulfur_dioxide + residual_sugar + fixed_acidity + volatile_acidity + alcohol + sulphates + pH + density + color_R', data=df).fit()
print(results1.summary())


# ### Note the warning: 
# #### "[2] The condition number is large, 2.93e+05. This might indicate that there is strong multicollinearity or other numerical problems." <br> 
# We'll see how our out-of-sample test results perform (if there's a lot of multicollinearity present, we'd expect to see decreased performance)

# In[ ]:


y = train["quality"]
cols = ["total_sulfur_dioxide","free_sulfur_dioxide","residual_sugar","fixed_acidity","volatile_acidity","alcohol","sulphates","pH","density","color_R"]

X=train[cols]


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(X,y)


# In[ ]:


ytrain_pred = regr.predict(X)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y, ytrain_pred))


# In[ ]:


ytest = test["quality"]
cols = ["total_sulfur_dioxide","free_sulfur_dioxide","residual_sugar","fixed_acidity","volatile_acidity","alcohol","sulphates","pH","density","color_R"]

Xtest=test[cols]


# In[ ]:


ypred = regr.predict(Xtest)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytest, ypred))


# #### The out-of-sample MSE isn't too much higher than the train sample, which is a good indication that there isn't too much overfitting in our model.

# ## 5.2 60-40 Split of Training and Hold-Out Data <a class="anchor" id="5.2-bullet"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
train2, test2 = train_test_split(df, test_size=0.4, random_state=4)


# In[ ]:


y2 = train2["quality"]
cols = ["total_sulfur_dioxide","free_sulfur_dioxide","residual_sugar","fixed_acidity","volatile_acidity","alcohol","sulphates","pH","density","color_R"]
X2=train2[cols]
regr.fit(X2,y2)


# In[ ]:


ytrain_pred2 = regr.predict(X2)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y2, ytrain_pred2))


# In[ ]:


ytest2 = test2["quality"]
Xtest2=test2[cols]


# In[ ]:


ypred2 = regr.predict(Xtest2)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytest2, ypred2))


# #### Again, the out-of-sample MSE isn't too much higher than the train sample, which indicates the model isn't overfit on the training data. <br>
# <br>
# *However,* based on our assessment of the correlation heatmap, we can see that there is strong evidence for multicollinearity (total sulur dioxide and free sulfur dioxide are inherently related. The same appears to be true with density, citric acid, and fixed acidity). <br> <br>
# What's more, the adjusted R-squared for this model very low, and indicates that only 29.5% of the variation in a wine's quality is due to variation in these variables. <br> <br>
# One way we can address this is by building separate regressions for Red and White wine (an easy way to segment the data).  From our heatmaps, we already know that there are certain variables that behave differently given the type of wine.  Before we move onto more advanced modeling techniques, let's just try this approach of splitting the data along color.

# ## 5.3 LinReg on Segmented Data (80/20 splits for both segments) <a class="anchor" id="5.3-bullet"></a>

# ### 5.3.1 White Wine Model <a class="anchor" id="5.3.1-bullet"></a>

# In[ ]:


w_train, w_test = train_test_split(df_white, test_size=0.2)


# In[ ]:


results_w = smf.ols('quality ~ free_sulfur_dioxide + residual_sugar + fixed_acidity + volatile_acidity + alcohol + sulphates + pH + density', data=df_white).fit()
print(results_w.summary())


# In[ ]:


y_w = w_train["quality"]
cols_w = ["free_sulfur_dioxide","residual_sugar","fixed_acidity","volatile_acidity","alcohol","sulphates","pH","density"]
X_w=w_train[cols_w]
regr.fit(X_w,y_w)


# In[ ]:


ytrain_predw = regr.predict(X_w)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y_w, ytrain_predw))


# In[ ]:


ytestw = w_test["quality"]
Xtestw = w_test[cols_w]
ypredw = regr.predict(Xtestw)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytestw, ypredw))


# ### 5.3.2 Red Wine Model <a class="anchor" id="5.3.2-bullet"></a>

# In[ ]:


r_train, r_test = train_test_split(df_red, test_size=0.2)


# In[ ]:


results_r = smf.ols('quality ~ free_sulfur_dioxide + residual_sugar + fixed_acidity + volatile_acidity + alcohol + sulphates + pH + density', data=df_white).fit()
print(results_r.summary())


# In[ ]:


y_r = r_train["quality"]
cols_r = ["free_sulfur_dioxide","residual_sugar","fixed_acidity","volatile_acidity","alcohol","sulphates","pH","density"]
X_r=r_train[cols_r]
regr.fit(X_r,y_r)


# In[ ]:


ytrain_predr = regr.predict(X_r)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y_r, ytrain_predr))


# In[ ]:


ytestr = r_test["quality"]
Xtestr = r_test[cols_r]
ypredr = regr.predict(Xtestr)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytestr, ypredr))


# #### combine results for testing

# In[ ]:


y_both = pd.concat([y_w,y_r])


# In[ ]:


ytrain_predW=pd.DataFrame(ytrain_predw)
ytrain_predR=pd.DataFrame(ytrain_predr)

y_train_predboth = pd.concat([ytrain_predW,ytrain_predR])


# In[ ]:


print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y_both, y_train_predboth))


# In[ ]:


ytestboth = pd.concat([ytestw,ytestr])
Xtestboth = pd.concat([Xtestw,Xtestr])
                                          
ypredboth = pd.concat([pd.DataFrame(ypredw),pd.DataFrame(ypredr)])
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytestboth, ypredboth))


# ## 6. Alternate Approach: Support Vector Machine <a class="anchor" id="6-bullet"></a>

# In[ ]:


from sklearn import svm

y = train["quality"]
cols = ["total_sulfur_dioxide","free_sulfur_dioxide","residual_sugar","fixed_acidity","volatile_acidity","alcohol","sulphates","pH","color_R"]
X=train[cols]

clf = svm.SVR(C=1.0, epsilon=0.2)
clf.fit(X, y) 

##http://scikit-learn.org/stable/modules/svm.html


# In[ ]:


ytrain_pred = clf.predict(X)
print("In-sample Mean squared error: %.2f"
      % mean_squared_error(y, ytrain_pred))


# #### The in-sample MSE for the support vector machine is much lower than the regression models.

# In[ ]:


ytest = test["quality"]
Xtest=test[cols]


# In[ ]:


ypred = clf.predict(Xtest)
print("Out-of-sample Mean squared error: %.2f"
      % mean_squared_error(ytest, ypred))


# ### Cross validation results aren't as great. Will need to toy around some more to optimize the hyperparameters for this model.

# In[ ]:




