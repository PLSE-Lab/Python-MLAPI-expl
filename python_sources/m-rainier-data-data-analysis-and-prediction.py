#!/usr/bin/env python
# coding: utf-8

# # Mount Rainier data analysis and Success Rate prediction

# Our goal is to find some patterns in the data and then to try different ML models and experiment a little bit with PCA and TSNE reduction

# # Sections:
# 1. Understanding data, data visualisation and cleaning
# 2. Preprocessing data
# 3. ML algorithms comparison
# 4. Final Evaluation and conclution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# I really hate these warnings about changes in future versions of skicit-learn so we will take care of them
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


# Let's import useful packages for start
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Loading datasets
# Weather
df_weather = pd.read_csv("../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv")
df_weather.head()


# In[ ]:


# Climbing
df_climbing = pd.read_csv("../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv")
df_climbing.head()


# In[ ]:


# Let's merge these datasets
df = df_climbing.merge(df_weather, on="Date")
df.head()


# In[ ]:


# Let's look for null values
df.isnull().sum()


# In[ ]:


# Great, no null values
# Let's make feature called Class, which tells us if the overall attempt was a success or not (0 for unsuccessful, 1 for successful)
df["Class"] = 0

for index in df.index:
    if df.iloc[index, df.columns.get_loc("Succeeded")] != 0:
        df["Class"].loc[index] = 1
    else:
        df["Class"].loc[index] = 0
df.head()


# In[ ]:


# Let's take a look on a class plot
f, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.countplot("Class", data=df, ax=ax)
plt.show()


# In[ ]:


# It's clear that we have more successful attempts then unsuccessful
# Let's which routes are more prefered
f, ax = plt.subplots(1, 1, figsize=(10, 6))
df.Route.value_counts().plot(kind="bar", ax=ax)


# In[ ]:


# Now let's see some correlations between weather features
f, ax = plt.subplots(1, 1, figsize=(10, 8))

corr = df_weather.corr()
sns.heatmap(corr, cmap="coolwarm_r", annot=True, ax=ax)
ax.set_ylim(len(df_weather.columns)-1, 0)
plt.show()


# In[ ]:


# We can see that Temperature is high positive correlated to Solare Radiation. That only makes sense.
# Let's now plot correlation of everything to success rate
f, ax = plt.subplots(1, 1, figsize=(10, 8))

corr_succ = df.corr()
sns.heatmap(corr_succ, cmap="coolwarm_r", annot=True, ax=ax)

ax.set_ylim(len(df.columns)-2, 0)
plt.show()


# In[ ]:


# The only positive effect from all features on success percentage has temperature and Solare Radiation
# On the other hand there is only Wind Speed worth noticing (negative correlation).
# We have a lot of dates so let's group them into months

import calendar

df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = 0
df["Month"] = df["Date"].dt.month
df.Month = df.Month.apply(lambda x: calendar.month_abbr[x])

df.head()


# In[ ]:


# Unfortunatlly i didn't succeed in placing months into right order, but it doesn't matter that much

# Let's see what temperature has to do with success rate
f, ax = plt.subplots(1, 1, figsize=(10, 8))

df["Success Percentage"] = df["Success Percentage"] * 100

sns.lineplot(x="Month", y="Temperature AVG", data=df, ax=ax)
sns.lineplot(x=df["Month"], y="Success Percentage", data=df, ax=ax)

ax.set_ylabel("")
f.legend(labels=["Success Rate", "Temperature"])
plt.show()


# In[ ]:


# You can see that almost everytime when temperature is high the success rate is also high and vica versa
# December is usually the coldest so the success rate should be lower there, because the cold is making the mountain harder to climb
# On the other hand the Jul and Jun are the warmest months so you get the idea


# In[ ]:


# Next plot should be success rate vs solare radiation within months
f, ax= plt.subplots(1, 1, figsize=(10, 8))
sns.lineplot(x="Month", y="Solare Radiation AVG", data=df, ax=ax)
sns.lineplot(x="Month", y="Success Percentage", data=df, ax=ax)
f.legend(labels=["Solare Radiation", "Success Rate"])
ax.set_label("")
plt.legend()
plt.show()


# In[ ]:


# You can see that it's almost the same as the other plot. It's because in the weather section solare rad. and temperature are highly positively correlated

#Let's plot humidity as well, because it's the feature with the highest negative correlation
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

sns.lineplot(x="Month", y="Wind Speed Daily AVG", data=df, ax=ax)
sns.lineplot(x="Month", y="Success Percentage", data=df, ax=ax)
ax.set_ylabel("")
fig.legend(labels=["Wind Speed", "Success Rate"])
plt.legend()
plt.show()


# In[ ]:


# It's clear that wind speed curve is going against success rate curve

# Well, we have now better insight into this problem
# We see a lot of other correlations between weather itself, but it's not necessary to plot that as well
# We are only interested in Success Percentage


# In[ ]:


y = df["Success Percentage"] / 100
df.drop(["Class", "Date", "Success Percentage"], axis=1, inplace=True)
X = df


# <font size="4">Data **Scaling**, **Encoding** and **Normalization**</font> 
#   :
#   
# First we have to encode labels in Route and Month features, because ML algorithm is unable to process Categorical features without encoding.
# We will use LabelEncoder first to encode labels into numbers. Then we will use OneHotEncoder to create a binary columns for each category (for each category
# we will need as many columns as we have unique values in these categories). Then scaling etc.
# We will be following this order:
#     1. LabelEncoder
#     2. OneHotEncoder
#     3. RobustScaler (scaling)
#     4. PCA and TSNE reduction (just for model comparison, we wont use it later)
#     5. Model comparison with non-normalized data (we will choose the best one)
#     6. Zscore normalization (There will be an improvement in final accuracy)

# In[ ]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X["Route_encoded"] = label_encoder.fit_transform(X["Route"])
X["Month_encoded"] = label_encoder.fit_transform(X["Month"])
X.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

hot_encoder = OneHotEncoder()

X_route = hot_encoder.fit_transform(X["Route_encoded"].values.reshape(-1, 1)).toarray()
X_month = hot_encoder.fit_transform(X["Month_encoded"].values.reshape(-1, 1)).toarray()


# In[ ]:


# Now we have to create columns for onehot encoded values
df_route = pd.DataFrame(X_route, columns=["Route_"+str(int(i)) for i in range(X_route.shape[1])])
df_month = pd.DataFrame(X_month, columns=["Month_"+str(int(i)) for i in range(X_month.shape[1])])

df_list = [
    df_route,
    df_month,
    df,
]

X = pd.concat(df_list, axis=1)

X.drop(["Route", "Month", "Route_encoded", "Month_encoded"], axis=1, inplace=True)
X.head()


# In[ ]:


# Now we have our encoded data, the next step is scaling
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

for column in X.columns[X.columns.get_loc("Attempted"):]:
    X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))
X.head()


# In[ ]:


# Float to int (categorical values only)
for column in X.columns[:X.columns.get_loc("Attempted")]:
    X[column] = X[column].astype(int)
        
X.head()


# In[ ]:


# PCA and TSNE reduction

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42)

X_reduced_pca = pca.fit_transform(X.values)
X_reduced_tsne = tsne.fit_transform(X.values)
X_reduced_pca.reshape(-1, 1)


# In[ ]:


#Importing models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


# In[ ]:


from sklearn.model_selection import cross_val_score
# Testing classifiers one after another
classifiers = {
    "Linear Regression:": LinearRegression(),
    "Random Forest:": RandomForestRegressor(),
    "Support Vector Regressor:": SVR(),
    "Extra Trees Regressor": ExtraTreesRegressor(),
}


# In[ ]:


# training with X_pca data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_reduced_pca, y, test_size=0.2, random_state=42)

summary = []
result = []
for key, classifier in classifiers.items():
    mse_score= cross_val_score(classifier, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    rmse_score = np.sqrt(-mse_score)
    summary += [classifier.__class__.__name__]
    result += [round(rmse_score.mean(), 4)]


# In[ ]:


# Results represents success RMSE(root mean squared error) of success rate feature (values from 0 to 1)
print("PCA REDUCTION")
for index in range(4):
    print("Classifier: {} has score: {}".format(summary[index], result[index]))


# Notice that our data are kind of complex (they have a lot of features) so I tried PCA and TSNE reduction. We can see that Linear model is the best, but when it comes to data without any
# reduction, LinearRegressor will go crazy

# In[ ]:


#training with X_tsne data
X_train, X_test, y_train, y_test = train_test_split(X_reduced_tsne, y, test_size=0.2, random_state=42)

summary = []
result = []
for key, classifier in classifiers.items():
    mse_score= cross_val_score(classifier, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    rmse_score = np.sqrt(-mse_score)
    summary += [classifier.__class__.__name__]
    result += [round(rmse_score.mean(), 4)]


# In[ ]:


print("TSNE REDUCTION")
for index in range(4):
    print("Classifier: {} has score: {}".format(summary[index], result[index]))


# In[ ]:


#Training with normal data with out reducion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

summary = []
result = []
for key, classifier in classifiers.items():
    mse_score= cross_val_score(classifier, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
    rmse_score = np.sqrt(-mse_score)
    summary += [classifier.__class__.__name__]
    result += [round(rmse_score.mean(), 4)]


# In[ ]:


print("No Reduction:")
for index in range(4):
    print("Classifier: {} has score: {}".format(summary[index], result[index]))


# Yep, this is what I was talking about. LinearRegression is now accurate as hell! Error 6310034155200% No big deal

# In[ ]:


# ZScore scaling 
from scipy.stats import zscore

X_zscore = X.apply(zscore)
X_zscore.head()


# In[ ]:


# Spliting zscore normalized data
X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.2, random_state=42)

# Let's how is our model good in predicting training data

training_score = cross_val_score(ExtraTreesRegressor(), X_train, y_train, scoring="neg_mean_squared_error", cv=5)
rmse = np.sqrt(-training_score)
print("Our RandomForestRegressor has the final RMSE(root mean squared error):  ", round(rmse.mean(), 6)*100, "%")


# In[ ]:


# Wow that's huge error, that means our model is overfitting. It will get better in the future.
# I would like to see learning curves of RandomForestRegressor and ExtraTreesRegressor to see which is better in terms of not overfitting
from sklearn.model_selection import learning_curve, ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


# Random Forest learning curve
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(RandomForestRegressor(), "Random Forest Regressor", X_train, y_train, cv=cv, n_jobs=4)
plt.show()


# In[ ]:


# Extra Trees learning curve
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(ExtraTreesRegressor(), "Extra Trees Regressor", X_train, y_train, cv=cv, n_jobs=4)
plt.show()


# In[ ]:


# We can see that both models are overfitting (huge gap between training and validation score)
# We will prefer RandomForestRegressor, because it overfits less than Extra Trees

# I tried to tune the model, but the best performace i could reach was with model's default hyper parameters


# # Final valuation on test set

# In[ ]:


from sklearn.metrics import mean_squared_error

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

final_pred = forest.predict(X_test)
final_mse = mean_squared_error(y_test, final_pred)
final_rmse = np.sqrt(final_mse)
print("Our RandomForestRegressor has the final RMSE(root mean squared error):  ", round(final_rmse, 6)*100, "%")


# # Conclusion
# 
# Most Success Rate values are between 0 % to 66.6 % so error roughtly 6 % is kind of huge, but we can do nothing about it as we have small dataset. I didn't optimized my RandomForestRegressor, because the best score i can achieve is with forest model's default hyper parameters.
# 
# I will be greatful for every feedback from you guys. Thank you!
