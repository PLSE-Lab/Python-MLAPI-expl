#!/usr/bin/env python
# coding: utf-8

# # Predicting the number of Reviews of a Video Game: A Regression Approach
# We will be using the video games dataset collected from RAWG which is one of the biggest video games databases. A question that will be answered today is if it's possible to create a fairly accurate model that predicts the number of reviews a game would have given a certain amount of independent variables. To answer this question, we will be performing a multiple regression approach first then try out a regularized regression method.

# Let's first load our dependencies.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt


# ## Data Cleaning/Wrangling:
# 
# [](http://)
# Load the dataset.

# In[ ]:


game_df = pd.read_csv('/kaggle/input/rawg-game-dataset/game_info.csv')
game_df


# First, we'd like to train our model with as much data as possible. Let's look at the independent variables with the most amount of missing values (Top 5).

# In[ ]:


percent_missing = game_df.isnull().sum() * 100 / len(game_df)
missing_value_df = pd.DataFrame({'column_name': game_df.columns,
                                 'percent_missing': percent_missing})
missing_value_df.sort_values(by=['percent_missing'], ascending=False).head()


# We'll be removing those columns listed above among other variables that may not improve our model. (tba, platforms, developers, slug, name, updated)

# In[ ]:


game_df = game_df.drop(['website', 'tba', 'publishers', 'esrb_rating', 'metacritic',                        'platforms', 'developers', 'genres', 'slug', 'name', 'updated'], axis=1).dropna()


# Modify years to an integer variable. We'll be retrieving the years and convert the datatype to a numerical one.

# In[ ]:


game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')


# ## Feature Selection:
# Now that we're all set. Let's visualize the correlation heatmap to figure out which variables have moderate to strong relationships.

# In[ ]:


f = plt.figure(figsize=(19, 15))
plt.matshow(game_df.corr(), fignum=f.number)
plt.xticks(range(game_df.shape[1]), game_df.columns, fontsize=14,rotation=45)
plt.yticks(range(game_df.shape[1]), game_df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# "added_status_beaten", "added_status_owned", "added_status_dropped", "added_status_playing", "added_status_toplay" are prime candidates for independent variables for counting the number of reviews. But because of high multicollinearity between some of these variables. "added_status_beaten", "added_status_toplay" and, "added_status_dropped" are the only ones that will be added. "rating_top" and "rating" are both moderately correlated as well with "review count". But again, due to multicollinearity with these variables, we'll only choose to add in "rating". We'll also throw in "suggestions_count" and "game_series_count" despite the weak correlation to test if it improves the predictive power.
# 
# **Note**: We are not including "ratings_count" because giving a rating indicates giving a review as well at RAWG so that shows a clear bias.
# 
# Let's now visualize the plot between each predictor against the number of reviews to check if it follows a linear relationship.

# In[ ]:


def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

matplotlib.rcParams["font.size"] = 16

graph = sns.pairplot(game_df, x_vars=["added_status_beaten", "added_status_dropped",                               "added_status_toplay", "rating", "suggestions_count",                               "game_series_count"], y_vars=["reviews_count"], height=5, aspect=.8, kind="reg");
graph.map(corrfunc)
plt.show()


# It becomes apparent that some of the relationships shown above exhibits a cone-shaped pattern. This could be a problem as the residuals plot may show heteroscedasticity. Some of the relationships as well especially on variables such as "rating" and "suggestions_count" shows that the line is heavily pulled by a lot of data points that have low amount of reviews despite the high rating or high suggestion counts. This makes sense as instances where there's only a single review with a high rating happens often which pulls our line downwards heavily. Let's log transform these variables to see if it fixes some of the problems.

# In[ ]:


game_df['logadded_status_beaten'] = np.log(game_df[['added_status_beaten']] + 1)
game_df['logadded_status_dropped'] = np.log(game_df[['added_status_dropped']] + 1)
game_df['logadded_status_toplay'] = np.log(game_df[['added_status_toplay']] + 1)
game_df['log_rating'] = np.log(game_df[['rating']] + 1)
game_df['log_suggestions_count'] = np.log(game_df[['suggestions_count']] + 1)
game_df['log_game_series_count'] = np.log(game_df[['game_series_count']] + 1)
game_df['logreviews_count'] = np.log(game_df[['reviews_count']] + 1)


# In[ ]:


graph = sns.pairplot(game_df, x_vars=["logadded_status_beaten", "logadded_status_dropped",                               "logadded_status_toplay", "log_rating", "suggestions_count",                               "log_game_series_count"], y_vars=["logreviews_count"], height=5, aspect=.8, kind="reg")
graph.map(corrfunc)
plt.show()


# By performing a log transformation on some of the independent variables along with the dependent variable, we have seen a lot of improvement with the relationships. The only instance where the correlation dropped is when we compare log transformed added_status_beaten and log transformed reviews_count. We see an improvement with every other relationship. The only variable that wasn't log transformed is suggestions_count as the correlation between the log transformed dependent variable and the log-transformed suggestions_count is significantly lower than if we don't do a log transformation on that variable.
# 
# By that, we should be ready to start creating our model.
# 
# ## Modeling
# We'll be using Scikit-Learn to train our multiple regression model. We will be using the closed-form solution way to obtain our model parameters so we won't have to do feature scaling for our variables. We'll also be splitting our dataset into training and testing.

# In[ ]:


X = game_df[["logadded_status_beaten", "logadded_status_dropped",                               "logadded_status_toplay", "log_rating", "suggestions_count",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg = LinearRegression().fit(X_train, y_train)
r2 = reg.score(X_train, y_train)


# Let's have a look at the r-squared and the adjusted r-squared of the trained model.

# In[ ]:


print("R-Squared: " + str(r2))
adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))
print("Adjusted R-Squared: " + str(adj_r2))


# Let's also check if the adjusted r-squared can be improved by dropping any of the predictors.

# In[ ]:


X = game_df[["logadded_status_dropped",                               "logadded_status_toplay", "log_rating", "suggestions_count",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg_test = LinearRegression().fit(X_train, y_train)
r2 = reg_test.score(X_train, y_train)

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-squared Dropping logadded_status_beaten: " + str(adj_r2))

X = game_df[["logadded_status_beaten",                               "logadded_status_toplay", "log_rating", "suggestions_count",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg_test = LinearRegression().fit(X_train, y_train)
r2 = reg_test.score(X_train, y_train)

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-squared Dropping logadded_status_dropped: " + str(adj_r2))

X = game_df[["logadded_status_beaten", "logadded_status_dropped",                               "log_rating", "suggestions_count",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg_test = LinearRegression().fit(X_train, y_train)
r2 = reg_test.score(X_train, y_train)

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-squared Dropping logadded_status_toplay: " + str(adj_r2))

X = game_df[["logadded_status_beaten", "logadded_status_dropped",                               "logadded_status_toplay", "suggestions_count",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg_test = LinearRegression().fit(X_train, y_train)
r2 = reg_test.score(X_train, y_train)

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-squared Dropping log_rating: " + str(adj_r2))

X = game_df[["logadded_status_beaten", "logadded_status_dropped",                               "logadded_status_toplay", "log_rating",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg_test = LinearRegression().fit(X_train, y_train)
r2 = reg_test.score(X_train, y_train)

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-squared Dropping suggestions_count: " + str(adj_r2))

X = game_df[["logadded_status_beaten", "logadded_status_dropped",                               "logadded_status_toplay", "log_rating", "suggestions_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

reg_test = LinearRegression().fit(X_train, y_train)
r2 = reg_test.score(X_train, y_train)

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 
          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-squared Dropping log_game_series_count: " + str(adj_r2))


# As we can see, the adjusted r-squared does not improve when we drop any of the predictors. Hence, we will keep all of the predictors in our current model.
# 
# Let's also create a regularized regression model, Ridge Regression to see if it will perform better than our previous multiple regression approach. We will be using grid search to find the optimal value for the hyperparameter alpha.

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

from sklearn.preprocessing import StandardScaler

X = game_df[["logadded_status_beaten", "logadded_status_dropped",                               "logadded_status_toplay", "log_rating", "suggestions_count",                               "log_game_series_count"]]
y = game_df[["logreviews_count"]]

X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.30, random_state=42)

model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
grid.fit(X_train, y_train)
print("Best Estimator value: " + str(grid.best_estimator_.alpha))


# Create the Ridge Regression model with the best estimator value as the hyperparameter alpha.

# In[ ]:


model = Ridge(alpha=1.0)

rreg = model.fit(X_train, y_train)


# ## Evaluation and Testing:
# Let's then test our models by predicting data that it has not seen before. We'll be trying out the multiple regression model without regularization first.

# In[ ]:


y_pred = reg.predict(X_test)
print("Predictions: " + str(y_pred))
print("Actual: " + str(y_test['logreviews_count'].values))


# Let's have a look at our model's RMSE.

# In[ ]:


rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Model RMSE: " + str(rmse))


# This does not tell us a lot though as the units are dependent on the scale of our dependent variable which has been log transformed. Let's reverse the log transform of the RMSE to check its value on a much understandable scale.

# In[ ]:


print("Model RMSE (Transformed Back): " + str(np.exp(rmse)))


# The model's RMSE (on the scale of the dependent variable) is 1.171 which is a satisfactory result for our model.
# 
# Let's have a look at our Ridge Regression model to see if it has a lower RMSE.

# In[ ]:


y_pred = rreg.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Regularized Model RMSE (Transformed Back): " + str(np.exp(rmse)))


# The results are mostly similar but our Ridge Regression Model has a slightly lower RMSE(very small difference) than our Multiple Regression model without regularization. We will be using the Ridge Regression model as our final model.
# 
# Let's try it out on a sample row to see what our model outputs. We will be choosing the row with the highest amount of reviews to see what our model predicts.

# In[ ]:


y_test = y_test.reset_index()
max_index = list(y_test[y_test['logreviews_count'] == y_test['logreviews_count'].max()].index)[0]
print("Max test value: " + str(y_test['logreviews_count'][max_index]))
print("Max test value (Transformed back): " + str(np.ceil(np.exp(y_test['logreviews_count'][max_index]) - 1)))


# The max amount of reviews for a single game in the test set is 1811 reviews. Let's see what our model predicted.

# In[ ]:


print("Prediction on max test value: " + str(y_pred[max_index][0]))
print("Prediction on max test value (Transformed Back): " + str(np.ceil(np.exp(y_pred[max_index][0]) - 1)))


# The model seemed to do a pretty good job with predicting the amount of reviews as it was off by only 88 reviews.
# 
# ## Conclusion and where to go from here:
# 
# The model is far from perfect but with the given variables, we can show that it is possible to have a fairly accurate model that predicts the amount of reviews a game has. Future works on this problem could try out other feature selection methods and could even go for text mining utilizing the previously dropped categorical variables. Other modeling alternatives could be used as well that could provide more accurate results. 
# 
# 
# Example alternative modeling techniques that could be used in this problem: Deep learning (ReLu as the final activation unit) or other regularized regression techniques.
