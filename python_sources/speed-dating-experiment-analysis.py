#!/usr/bin/env python
# coding: utf-8

# ## Speed Dating Experiment Analysis

# ### About the data:
# <ul>
# <li>Data was gathered from 552 participants in experimental speed dating events from 2002-2004.
# <li>During the events, the attendees would have a four minute "first date" with every other participant of the opposite sex.
# <li>At the end of their four minutes, participants were asked if they would like to see their date again. They were also asked to rate their date on six attributes:
#     <ul>
#     <li>Attractiveness
#     <li>Sincerity
#     <li>Intelligence
#     <li>Fun
#     <li>Ambition
#     <li>Shared Interests.
#     </ul>
# <li>The dataset also includes questionnaire data gathered from participants at different points in the process. These fields include:
#     <ul>
#     <li>demographics
#     <li>dating habits
#     <li>self-perception across key attributes
#     <li>beliefs on what others find valuable in a mate
#     <li>lifestyle information
#     </ul>
# </ul>

# #### Lets start Analysis 

# In[ ]:


#importing lib and packages

import pandas as pd
pd.options.display.max_rows = 1000 # for showing truncated result

import matplotlib.pyplot as plt
#to avoid writing plt.show() again and again
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# importing data
data = pd.read_csv('../input/Speed Dating Data.csv', encoding='ISO-8859-1')
# using encoder to read data properly without error
data.head()


# In[ ]:


# Basic EDA and statistical analysis
data.info(verbose = True)


# In[ ]:


#counting null values
data.isnull().sum()


# In[ ]:


#observing the shape of data
data.shape


# ### Data Cleaning and EDA
# 
# From the above information, their are lots of null data, as their are 8378 rows and 195 fileds which have thousands of null values , probably it will hamper our results. So we will just disregards these fields with over 4k null values from the dataset and narrow our analysis to the filed that we can use, but before that , lets take a look before we throw out some fileds and row due to missing values

# In[ ]:


# age distribution of participants
age = data[np.isfinite(data['age'])]['age']
plt.hist(age.values)
plt.xlabel('Age')
plt.ylabel('Frequency')


# so by seeing the graph we can say most of the participants were in their mid 20's to early 30's

# In[ ]:


# lets see how many lucky person found the match
pd.crosstab(index=data['match'], columns='counts')


# In[ ]:


# narrowing the dataset
data_1 = data.iloc[:, 11:28]
data_2 = data.iloc[:,30:35]
data_3 = data.iloc[:, 39:43]
data_4 = data.iloc[:, 45:67]
data_5 = data.iloc[:, 69:74]
data_6 = data.iloc[:, 87:91]
data_7 = data.iloc[:, 97:102]
data_8 = data.iloc[:, 104:107]


date = pd.concat([data.iloc[:, 0],data.iloc[:, 2],data_1,data_2,data_3,data_4,data_5,
                  data_6,data_7,data_8], axis=1)
date.head()


# In[ ]:


# counting null values
date.isnull().sum()


# In[ ]:


# removing null rows
date2 = date.dropna()


# In[ ]:


# creating an object- free data frame
date3 = date2.drop(['field', 'from', 'career'], axis=1)


# In[ ]:


# heat map
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title('Correlation Heatmap')
corr = date3.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)


# while looking at the heatmap above, there are some interesting observations.
# 
# For example, men (gender = 1) seem to have a preference for attractive partners (attr1_1) while women (gender = 0) seem to have a preference for ambitious partners (amb1_1)!
# 
# In other terms, women like shopping and yoga and men like gaming, so at least we know this heatmap is working. Let's look into the interests some more!

# In[ ]:


# looking at intrest overlap for sports and tv-sports

sns.set(style='ticks', color_codes=True)
date_int = pd.concat([date3.iloc[:,1], date3.iloc[:, 30:32]], axis = 1)
g = sns.pairplot(date_int, hue='gender')


# In[ ]:


# looking at intrest overlap for dining, museums and arts
sns.set(style='ticks', color_codes=True)
date_int = pd.concat([date3.iloc[:,1], date3.iloc[:, 33:36]], axis = 1)
g = sns.pairplot(date_int, hue='gender')


# In[ ]:


# looking at intrest overlap for theater, movies and concerts
sns.set(style='ticks', color_codes=True)
date_int = pd.concat([date3.iloc[:,1], date3.iloc[:, 41:44]], axis = 1)
g = sns.pairplot(date_int, hue='gender')


# In[ ]:


# removing intrests
date4 = date3.drop(['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 
                    'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 
                   'shopping', 'yoga'], axis=1)


# Going along investigating further gender differences, I wonder... how many of each gender are there and does that affect the other person's decision? That is, do women receive more positive final decisions from the other person (dec_o) than men do?

# In[ ]:


# looking at dec_o by gender
sns.set(style="ticks", color_codes=True)
g = sns.FacetGrid(date4, col="gender")
g = g.map(plt.hist, "dec_o")
plt.ticklabel_format(useOffset=False, style='plain')


# In[ ]:


# chi-square test
gender_crosstab = pd.crosstab(index=date4.gender, columns=date4.dec_o)
gender_table = sm.stats.Table(gender_crosstab)
gender_result = gender_table.test_nominal_association()
gender_result.pvalue


# It looks like women received about 1750 'no' and about 1600 'yes' for the decision question 
# "Would you like to see him or her again?". 
# Men received about 2050 'no' and about 1300 'yes'. 
# In other words, men are more likely to be rejected by women than women are to be rejected by men. 
# This is a statistically significant difference as confirmed by the above chi-squared test p-value.

# In[ ]:


# unrequited love count
no_love_count = len(date4[(date4['dec_o']==0) & (date4['dec']==1)]) 
+ len(date4[(date4['dec_o']==1) & (date4['dec']==0)])
perc_broken_heart = no_love_count / len(date4.index)
perc_broken_heart*100


# So it seems 26% of participants unfortunately had their heart broken. More than the percentage of people who got a second date!
# 
# On an unrelated note, I wonder if the incidence of unrequited love differs by the attractiveness of the partner.

# In[ ]:


# encoding unrequited love as a new column
date4['url']=np.where(((date4['dec_o']==0) & (date4['dec']==1))|((date4['dec']==0) & (date4['dec_o']==1)),1,0)


# In[ ]:


# looking at url by attractiveness
plt.figure(figsize=(7,9))
sns.boxplot(x='url', y='attr', data=date4, palette='cool')
plt.title('Broken Hearts by Attractiveness of Partner', fontsize=20)
plt.xlabel('Broken Heart', fontsize=16)


# In[ ]:


# chi-square test
bh_crosstab = pd.crosstab(index=date4.attr, columns=date4.url)
bh_table = sm.stats.Table(bh_crosstab)
bh_rslt = bh_table.test_nominal_association()
bh_rslt.pvalue


# Looks like the difference in attractiveness was not statistically significant. So the good news is, the likelihood of getting rejected is not dependent on your attractiveness!

# In[ ]:


date5 = pd.concat([date4['attr3_1'],date4['sinc3_1'],date4['intel3_1'],date4['fun3_1'],date4['attr_o'],
                   date4['sinc_o'],date4['intel_o'],date4['fun_o'],date4['like'],date4['like_o'], 
                   date4['int_corr'],date4['url']],axis=1)
plt.subplots(figsize=(15,10))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = date5.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# It looks like regardless of your attractiveness, or any other trait for that matter, you are just as likely to experience heartbreak!
# 
# It also looks like typically, your own opinion of how attractive you are (attr3_1) is only weakly correlated with how attractive your date finds you (attr_o)! And in fact, there is nearly no correlation between how smart or sincere you think you are versus how smart and sincere your date thinks of you! Perhaps these are tough qualities to get across in a 4 minute date!
# 
# So that brings up the question, between attractiveness, sincerity, intelligence, fun, ambition, and liking, what was the most influential in the final decision (dec)? I'll run a linear regression model.

# In[ ]:


# OLS Model with coefficients
X_ols = date4[['attr','sinc','intel','fun','like','int_corr']]
y_ols = date4.dec
traits = sm.OLS(y_ols, X_ols)
results_traits = traits.fit()
results_traits.summary()


# It turns out that being intelligent or sincere or having similar interests actually slightly hurts your chances at securing that second date! Don't panic though, this is just from 4 minutes of meeting each other! We might take this as advice to focus on breaking the ice and being more fun and likeable in the first date!
# 
# Now let's run an OLS but with both respondents' ratings instead of just one, and this time on match instead of decision (dec).
# 
# 

# In[ ]:


# OLS model with coefficients
X_ols = date4[['dec','dec_o','attr','attr_o','fun','fun_o','like','like_o','int_corr']]
y_ols = date4.match
traits = sm.OLS(y_ols, X_ols)
results_traits = traits.fit()
results_traits.summary()


# #### From the coefficients, it looks like all that really matters is the decision of both participants, and perhaps whether or not they liked one another.
# 
# ## Modeling - Classification 

# In[ ]:


# preparing the data for train and test
X=date4[['like','dec']]
y=date4['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# In[ ]:


# logistic regression classification model
model = LogisticRegression(C=1, random_state=0)
lrc = model.fit(X_train, y_train)
predict_train_lrc = lrc.predict(X_train)
predict_test_lrc = lrc.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_lrc))


# Without knowing what the partner's decision is (dec_o), it turns out that given how much the respondent likes the partner and what the respondent's decision is, we have about an 82.5% accuracy in predicting a match on both the training and the validation using logistic regression. This makes sense given that we know only 26% of people were heartbroken -- if you like someone, odds are they will like you back!
# 
# Let's try some other models to see if we can get closer to predicting a match.

# In[ ]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins = True)


# In[ ]:


# random forest model
model = RandomForestClassifier()
rf_model = model.fit(X_train, y_train)
predict_train_rf = rf_model.predict(X_train)
predict_test_rf = rf_model.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_rf))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_rf))


# Random forest gave us a slightly more accurate model at 82.9% accuracy in train and 82.8% in test.

# In[ ]:


# xgboost model
model = GradientBoostingClassifier()
xgb_model = model.fit(X_train, y_train)
predict_train_xgb = xgb_model.predict(X_train)
predict_test_xgb = xgb_model.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_xgb))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_xgb))


# XGBoost was ever so slightly less accurate than Random Forest in the validation set. Looks like Random Forest is my champion model.

# #### Conclusion 
# Although this was slightly disappointing, it looks like there still is no real answers to understand female. It's not interests or hobbies, it's not attractiveness or intelligence or other traits.
# 
# but we lean a lot in this exploration.

# In[ ]:




