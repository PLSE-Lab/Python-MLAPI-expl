#!/usr/bin/env python
# coding: utf-8

# Exploration of How Social Media Can Predict Winning Metrics Better Than Salary

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[ ]:


twiter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twiter_df.head()


# In[ ]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# **Data Preprocessing**

# In[ ]:


attendance_valuation_elo_df.drop(["Unnamed: 0","CONF"], axis=1, inplace=True); attendance_valuation_elo_df.head()


# In[ ]:


salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION", "TEAM", "SALARY"], inplace=True, axis=1);salary_df.head()


# In[ ]:


plus_minus_df["NAME"]=plus_minus_df["NAME"].str.split(",").str[0]; plus_minus_df.head()
# plus_minus_df.drop(["TEAM"], axis=1, inplace=True); plus_minus_df.head()


# In[ ]:


twiter_df.rename(columns={"PLAYER":"NAME"}, inplace=True)
twiter_df[["TWITTER_FAVORITE_COUNT", "TWITTER_RETWEET_COUNT"]].astype(float); twiter_df.head()


# In[ ]:


#Merge tables 
merge_tbl_1 = pd.merge(plus_minus_df, salary_df, left_on="NAME", right_on="NAME", how="left"); merge_tbl_1.head()
merge_tbl = pd.merge(merge_tbl_1, twiter_df, left_on="NAME", right_on="NAME", how="left"); merge_tbl.head()


# In[ ]:


ax=sns.pairplot(plus_minus_df)


# In[ ]:



plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017")
corr = merge_tbl.corr()
#insert mask
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): sns.heatmap(corr, mask=mask, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# In[ ]:


corr


# In[ ]:


sns.lmplot(x="WINS", y="TWITTER_FAVORITE_COUNT", data=merge_tbl)


# In[ ]:


sns.lmplot(x="WINS", y="TWITTER_RETWEET_COUNT", data=merge_tbl)


# In[ ]:


sns.lmplot(x="GP", y="WINS", data=merge_tbl)


# In[ ]:


sns.lmplot(x="MPG", y="WINS", data=merge_tbl)


# In[ ]:


sns.lmplot(x="DRPM", y="WINS", data=merge_tbl)


# In[ ]:


merge_tbl.info()


# In[ ]:


#Fill NA
merge_tbl[["TWITTER_FAVORITE_COUNT","TWITTER_RETWEET_COUNT"]]=merge_tbl[["TWITTER_FAVORITE_COUNT","TWITTER_RETWEET_COUNT"]].fillna(0)


# In[ ]:


merge_tbl = merge_tbl.dropna()
len(merge_tbl)


# In[ ]:


# select the least correlated variables for linear regression; avoid autocorrelation
#GP, DRPM, MPG, SALARY_MILLIONS
y_train=merge_tbl[:301]["WINS"]
x_train=merge_tbl[:301][["GP","MPG","DRPM","SALARY_MILLIONS"]]

y_test = merge_tbl[301:]["WINS"]
x_test = merge_tbl[301:][["GP","MPG","DRPM","SALARY_MILLIONS"]]


# In[ ]:


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
results = regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

print(results)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[ ]:


results = smf.ols('WINS ~ GP+MPG+DRPM+SALARY_MILLIONS', data=merge_tbl).fit()
print(results.summary())


# In[ ]:


merge_tbl["TWITTER_RETWEET"] = merge_tbl["TWITTER_RETWEET_COUNT"].apply(lambda x: 1/x if x != 0 else 0)


# In[ ]:


results = smf.ols('WINS ~ GP+MPG+DRPM+SALARY_MILLIONS+TWITTER_RETWEET', data=merge_tbl).fit()
print(results.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




