#!/usr/bin/env python
# coding: utf-8

# # 3 questions I have on dating
# Being a romantic myself and after reading a thousand of romance books I've always wondered what practical romance was like (you see I have never dated before nor used dating apps like Tinder myself). This dataset was like a gift from the heavens and it's time for me to do some data analysis !
# 
# **Here are some questions that I will be seeking answers to**
# * What is the ideal male or female like
# * Are shared interests more important than a shared racial background?
# * Can people accurately predict their own perceived value in the dating market?
# ### Table of Contents
#  1. EDA 
#  2. 1st question
#  3. 2nd question
#  4. 3rd question
#  5. Conclusion

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import statsmodels.api as sm
from scipy import stats
from sklearn.pipeline import Pipeline
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1") 
train.head() #yuck lots of objects and floats


# In[ ]:


train.columns #wow that's a lot of features


# In[ ]:


train.dtypes


# In[ ]:


train['match'].describe()


# In[ ]:


# combining columns to make a column that has more rows in it
dating_1 = train.iloc[:, 11:28]
dating_2 = train.iloc[:, 30:35]
dating_3 = train.iloc[:, 39:43]
dating_4 = train.iloc[:, 45:67]
dating_5 = train.iloc[:, 69:74]
dating_6 = train.iloc[:, 87:91]
dating_7 = train.iloc[:, 97:102]
dating_8 = train.iloc[:, 104:107]
#combining everything to make one big column
date = pd.concat([train.iloc[:, 0],train.iloc[:, 2],dating_1,dating_2,dating_3,dating_4,dating_5,
                  dating_6,dating_7,dating_8], axis=1)


# In[ ]:


columns = date.select_dtypes(include=[object])
print(columns)


# In[ ]:


cols_with_missing = [col for col in date.columns 
                                 if date[col].isnull().any()]                                  
candidate_train_predictors = date.drop(['iid'] + cols_with_missing, axis=1)


# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 191 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]


# In[ ]:


# Create a label (category) encoder object
#le = preprocessing.LabelEncoder()


# In[ ]:


# Fit the encoder to the pandas column
#le.fit(date[''])


# In[ ]:


# Standardize the feature matrix
#X = StandardScaler().fit_transform(speed)


# In[ ]:


speed = date.dropna() #dropping missing values


# In[ ]:


#dropping highly correlated features
# Create correlation matrix
corr_matrix = speed.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[ ]:


corrmat = speed.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Age seems to be correlated with going out which kind of makes sense because the younger you are the more appealing it is to go out to places like fancy restaurants and amusement parks. Also there are many observations that we can see from this heatmap but I just found it funny that the ones that club a lot are the ones that work out as well. Getting the body to hit the club ! (Not a bad strategy but a bit overused it appears)

# In[ ]:


# detecting outliers
# Create detector
#speed = pd.get_dummies(train_predictors) you can use this sometimes
#outlier_detector = EllipticEnvelope(contamination=.1)

# Fit detector
#outlier_detector.fit(speed)

# Predict outliers
#outlier_detector.predict(speed)


# In[ ]:


#match correlation matrix
k = 12 #number of variables for heatmap
cols = corrmat.nlargest(k, 'match')['match'].index
cm = np.corrcoef(speed[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Taking a closer look we see that match is highly correlated with final decision (this should've been deleted since this is similar to match) in this dataset and that a person is more likeable if they have nice attributes and are fun.Note that from here on any feature with '_o' means that that's their partner's trait

# # 1st question

# In[ ]:


speed[speed['gender'] == 1].mean() #male


# Men prefer women who are attractive, younger and have shared interests.

# In[ ]:


speed[speed['gender'] == 0].mean() #female prefer men who are ambitious and have shared interests


# # 2nd question

# In[ ]:


speed.pivot_table(['race', 'pf_o_sha'],
               ['match'], aggfunc='sum')


# With pivot table unfortunately we can't normalize so calculating the percentage of matches made for preferred common interests and race are 16.4% and 17.2% so race and shared interests are somewhat similar and have an equal degree of importance

# In[ ]:


_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

sns.countplot(x='race', data=speed, ax=axes[0]);
sns.countplot(x='pf_o_sha', data=speed, ax=axes[1]);


# In[ ]:


sns.lmplot('pf_o_sha', 'race', data=speed, hue='match', fit_reg=False);


# # 3rd question

# In[ ]:


# OLS model with coefficients with decision as our basis of judging other features
X_ols = speed[['attr','sinc','intel','fun','like','int_corr','pf_o_sha','race']]
y_ols = speed.dec
traits = sm.OLS(y_ols, X_ols)
results_traits = traits.fit()
results_traits.summary()


# Our OLS model says a lot of things. Going back to our 2nd question on is race or shared interests more important, it appears that both have an equal weighting. The ideal mate is very likeable and fun but insincere and not that smart.

# In[ ]:


# preparing the data
X=speed[['like','fun','attr']]
y=speed['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=3, stratify=y)


# In[ ]:


# logistic regression classification model
model = LogisticRegression(C=3, random_state=43)
lrc = model.fit(X_train, y_train)
predict_train_lrc = lrc.predict(X_train)
predict_test_lrc = lrc.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_lrc))


# In[ ]:


model = RandomForestClassifier()
rf_model = model.fit(X_train, y_train)
predict_train_rf = rf_model.predict(X_train)
predict_test_rf = rf_model.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_rf))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_rf))


# In[ ]:


# xgboost model
model = GradientBoostingClassifier()
xgb_model = model.fit(X_train, y_train)
predict_train_xgb = xgb_model.predict(X_train)
predict_test_xgb = xgb_model.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_xgb))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_xgb))


# It appears that random forest is the best model to predict your market value on the dating market

# So there you have it, these are the answers to my question and I hope you have enjoyed my journey. Please upvote and feel free to leave any comment (I love feedback), and stay tuned fellas, I've got more in my bag !
