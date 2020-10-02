#!/usr/bin/env python
# coding: utf-8

# # Predicting NBA player salaries
# 
# ## Table of Contents:
# * [Scope of the analysis](#1_scope)
# * [Read the data](#2_read)
# * [Preliminary exploratory analysis](#3_exploration)
# * [How are salaries related with the minutes and points per game?](#4_minutes_points)
# * [How are salaries related with offensive/defensive RPM?](#5_defense_offense)
# * [Salaries related with position](#6_position)
# * [How is social impact related with the salary? Twitter](#7_twitter)
# * [How is social impact related with the salary? Wikipedia](#8_wikipedia)
# * [Is salary determined by the age?](#9_age)
# * [Using the previous features combined to predict the salary](#10_all_together)
# 
# ![basket](https://coverfiles.alphacoders.com/749/74928.jpg)
# 
# ## Scope of the analysis <a class="anchor" id="1_scope"></a>
# 
# In this notebook, we want to explore to what extent is possible to predict the salary of the NBA players based on several player attributes. We first select a set of relevant features and we analyze their impact in the player salary separatedly. Then, we build a predictive model with those features that have a larger influence on the player salary.
# 
# Since for each player we have a wide range of statistics at our disposal, we are going to consider for this analysis only some relevant ones, to avoid multicollinearity in our predictors. Thus, we will take into account the minutes per game (MPG), the points per game (POINTS) and the offensive/defensive real plus-minus values (ORPM and DRPM). Furthermore, we are also analyzing other sort of features such as the player position and some social indicators (Twitter and Wikipedia).
# 
# Let's go for it!

# In[ ]:


#Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import math


# ## Read the data <a class="anchor" id="2_read"></a>

# In[ ]:


df_players = pd.read_csv("../input/social-power-nba/nba_2017_players_with_salary_wiki_twitter.csv", index_col=0)
df_players.head()


# We run a Quick analysis in search for null or invalid values.
# There are only few nans in 3 shot % and free throws % and twitter interactions. However, we do not drop these rows because one of them would be Draymond Green, which is a relevant player with one of the highest DRPM. We then opt for substituting these values by 0.

# In[ ]:


df_players.info()


# In[ ]:


df_players['TWITTER_FAVORITE_COUNT'] = df_players['TWITTER_FAVORITE_COUNT'].fillna(0)
df_players['TWITTER_RETWEET_COUNT'] = df_players['TWITTER_RETWEET_COUNT'].fillna(0)


# ## Preliminary exploratory analysis <a class="anchor" id="3_exploration"></a>
# 
# We are curious to know which players have the larger and smaller salaries, which is the quantity that we are trying to predict. We also have a look at the players with the larger RPM metrics out of curiosity :).

# In[ ]:


# Sorted by salary in descending order
df_sorted = df_players.sort_values(by='SALARY_MILLIONS', ascending=False)
df_sorted.head(5)


# In[ ]:


# Sorted by salary in ascending order
df_sorted = df_players.sort_values(by='SALARY_MILLIONS', ascending=True)
df_sorted.head(5)


# The tables above shows how the largest salary corresponds (not so surprisingly) to LeBron James, who earns **30.96M** per year. Lower salary instead is **0.06M** per year and corresponds to Alonzo Gee. The average and standard deviation of the salary are displayed below.

# In[ ]:


print('Mean salary is %.3f' % df_players['SALARY_MILLIONS'].mean())
print('Standard deviation of the salary is %.3f' % df_players['SALARY_MILLIONS'].std())


# In[ ]:


# Sort by the real plus minus
df_sorted = df_players.sort_values(by='RPM', ascending=False)
df_sorted.head(5)


# Let's remove from our dataframe most of the features related to player statistics and let's keep the rest of them. To be fair, we are removing from further analysis some really promising features, like the games played (GP) among others. However, we have observed this variable to be highly correlated with the MPG abd besides, we prefer to have a small amount of predictors to keep the model tractable and the notebook relatively short.
# 
# Then, we are going to plot the pairwise relationships for the variables in the dataset, to see how our predictors are related with each other and with the independent variable. In this figure, we separate the players per position as is the only categorical variable that we have in the selected dataset.

# In[ ]:


df_players.columns


# In[ ]:


df_players = df_players[['PLAYER', 'POSITION', 'AGE', 'MPG', 'POINTS', 'ORPM', 'DRPM', 'RPM', 
                         'SALARY_MILLIONS', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT', 'TWITTER_RETWEET_COUNT']]

# Temporary remove the player name for this plot
df_temp = df_players.drop(columns=['PLAYER'])
sns.set(style="ticks")
sns.pairplot(df_temp, hue="POSITION")
plt.show()


# ## How are salaries related with the minutes and points per game? <a class="anchor" id="4_minutes_points"></a>
# 
# ### Minutes per game <a class="anchor" id="4_1_minutes"></a>
# 
# We first analyze to what extent the player salary can be predicted based on the minutes per game. The same analysis presented below is going to be repeated for each numerical variable throughout this notebook. Therefore, we will explain the reasoning here for the **MPG** and then assume the same reasoning in the subsequent analysis.
# 
# Fistly, we display a scatterplot of the MPG vs the salary which also fits a linear model where the **MPG** acts as the independent variable and the **SALARY_MILLIONS** as the dependent variable. 

# In[ ]:


sns.lmplot(x="MPG", y="SALARY_MILLIONS", data=df_players)
plt.show()


# Secondly, we fit the model using [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) from sklearn library. To test the quality of the fit, we display two different metrics. One is the **coefficient of determination** or **R2** which represents the proportion of the variance in the dependent variable that can be explained by the independent variable. We also illustrate the **root mean squared error** or **rmse**, which is a frequently used measure of dispersion between values, in comparison to the **standard deviation** of the dependent variable.

# In[ ]:


X = df_players[['MPG']]
Y = df_players['SALARY_MILLIONS']

# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X, Y)
sc = lr_model.score(X, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# Finally, we analyze the **Pearson correlation coefficient** between the two variables. Although we have to bear in mind that correlation does not imply causality, it is an interesting indicator of how the degree of statistical association between the two variables.

# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['MPG'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# **Conclusion:** We observe that the salary is, to some extent, influenced by the minutes that a player plays on average per game. After analyzing the rest of the features, we would be able to determine how big is this influence in comparison to other predictors.

# ### Points per game <a class="anchor" id="4_2_points"></a>
# 
# At this point, we perform the same analysis as we did for the minutes per game. However, in this case, the independent variable will the **POINTS** and we use to predict the player salary.

# In[ ]:


sns.lmplot(x="POINTS", y="SALARY_MILLIONS", data=df_players)
plt.show()


# In[ ]:


X = df_players[['POINTS']]
Y = df_players['SALARY_MILLIONS']

# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X, Y)
sc = lr_model.score(X, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['POINTS'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# **Conclusion:** It looks that we have clearly one of the most important indicators so far. We observe that (as expected) the points per game is one key factor to determine the salary of a player. Both, the R2 and the RMSE indicate that it is more strongly related with the salary than the minutes per game. Let's continue analyzing the rest of the variables.

# ## How are salaries related with offensive/defensive RPM? <a class="anchor" id="5_defense_offense"></a>
# 
# ### Defense <a class="anchor" id="5_1_defense"></a>

# In[ ]:


sns.lmplot(x="DRPM", y="SALARY_MILLIONS", data=df_players)
plt.show()


# In[ ]:


X = df_players[['DRPM']]
Y = df_players['SALARY_MILLIONS']

# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X, Y)
sc = lr_model.score(X, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['DRPM'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# ### Offense <a class="anchor" id="5_2_offense"></a>

# In[ ]:


# Plot sepal width as a function of sepal_length across days
sns.lmplot(x="ORPM", y="SALARY_MILLIONS", data=df_players)
plt.show()


# In[ ]:


X = df_players[['ORPM']]
Y = df_players['SALARY_MILLIONS']

# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X, Y)
sc = lr_model.score(X, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['ORPM'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# **Conclusion** There is an interesting tendency to pay somewhat larger salaries to those players that contribute to the team offensively, rather than defensive players. Although this may be seen as obvious, it is always interesting to observe how the data confirms our intuition. However, none of these results suggest these features to be very strong indicators of the player salary.

# ## Salaries related with position <a class="anchor" id="6_position"></a>
# 
# As the player position is the only selected variable that is not numerical, we cannot perform the same type of analysis as before to analyze its influence in the salary. For this matter, we need to perform **One-way Analysis of Variance** or **ANOVA**. This is a statistical technique employed to compare the means values of the dependent variable for different populations or groups, which is the independent variable. In this case, the groups are the player positions and the value to be compared is the mean salary of these groups.
# 
# In order to have a visual insight, we plot the distribution of salaries separated per position in the figure below.

# In[ ]:


# Show each observation with a scatterplot
sns.set(style="whitegrid")
sns.stripplot(x="SALARY_MILLIONS", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, alpha=.50)
sns.pointplot(x="SALARY_MILLIONS", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, palette="dark", markers="d")
plt.show()


# At first glance, it doesn't seem to be important differences between all five distributions. However, we have to confirm this insight statistically using one-way ANOVA. We can do this for two independent groups as ilustrated below. 

# In[ ]:


# One-way ANOVA analysis

# Get each position independently
salary_pg = df_players['SALARY_MILLIONS'][df_players['POSITION']=='PG']
salary_sg = df_players['SALARY_MILLIONS'][df_players['POSITION']=='SG']
salary_sf = df_players['SALARY_MILLIONS'][df_players['POSITION']=='SF']
salary_pf = df_players['SALARY_MILLIONS'][df_players['POSITION']=='PF']
salary_c = df_players['SALARY_MILLIONS'][df_players['POSITION']=='C']

# Example of how it would be done for only two groups
fstat, pval = f_oneway(salary_sg, salary_c)
print('P Value: %.4f' % pval)


# Still, we would be interested in running all possible statistical tests between all five groups. For this, we employ MultiComparison with [Tukey post-hoc test](https://en.wikipedia.org/wiki/Tukey%27s_range_test) to spot the individual differences between groups.

# In[ ]:


# Compare all five positions with post-hoc correction
salary_pos = df_players[['SALARY_MILLIONS', 'POSITION']]

multi_comp = MultiComparison(salary_pos['SALARY_MILLIONS'], salary_pos['POSITION'])
# Print all the possible pairwise comparisons
print(multi_comp.tukeyhsd().summary())


# **Conclusion:** According to the table below, there are no significant differences between the salaries for each position and therefore this variable is unlikely to be an interesting predictor for the player salary.

# ### Same statistical test applied to defensive RPM per position <a class="anchor" id="6_1_defensive_rpm"></a>
# 
# The previous test did not yield any significant result. However, to prove the validity of the previous statistical test, we are going to perform the same analysis but in this case with the DRPM as the dependent variable.

# In[ ]:


sns.set(style="whitegrid")
sns.stripplot(x="DRPM", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, alpha=.50)
sns.pointplot(x="DRPM", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, palette="dark", markers="d")
plt.show()


# In[ ]:


# One-way ANOVA analysis

# Get each position independently
drpm_pg = df_players['DRPM'][df_players['POSITION']=='PG']
drpm_sg = df_players['DRPM'][df_players['POSITION']=='SG']
drpm_sf = df_players['DRPM'][df_players['POSITION']=='SF']
drpm_pf = df_players['DRPM'][df_players['POSITION']=='PF']
drpm_c = df_players['DRPM'][df_players['POSITION']=='C']

# Example of how it would be done for only two groups
fstat, pval = f_oneway(drpm_pg, drpm_sf)
print('P Value: %.4f' % pval)


# In[ ]:


# Compare all five positions with post-hoc correction
drpm_pos = df_players[['DRPM', 'POSITION']]

multi_comp = MultiComparison(drpm_pos['DRPM'], drpm_pos['POSITION'])
# Print all the possible pairwise comparisons
print(multi_comp.tukeyhsd().summary())


# **Conclusion:** We can observe that, indeed, the position is a strong indicator of the defensive real plus-minus of the player, having the frontcourt (SF, PF and C) players a larger impact in the defensive game than the backcourt positions (PG and SG). 

# ## How is social impact related with the salary? Twitter <a class="anchor" id="7_twitter"></a>
# 
# As we see in the plot below, the data has very obvious outliers when it comes to twitter interaction. For this reason, it is very unlikely that a linear model is able to capture the relationship between this and the salary of the player. To explore to what extent the salary can be predicted by the twitter interactions, we build a second order polynomial regression model. 

# In[ ]:


sns.lmplot(x="TWITTER_RETWEET_COUNT", y="SALARY_MILLIONS", order=2, data=df_players)
plt.show()


# We remove the entries above the 95 percentile of the twitter interactions, only to allow a better visualization of the fit.

# In[ ]:


# Remove outliers in the social impact
len_players = len(df_players)
q95 = df_players['TWITTER_RETWEET_COUNT'].quantile(0.95)
data = df_players[df_players['TWITTER_RETWEET_COUNT'] <= q95]
print('Removed %d players' % (len_players-len(data)))


# In[ ]:


sns.lmplot(x="TWITTER_RETWEET_COUNT", y="SALARY_MILLIONS", order=2, data=data)
plt.show()


# In[ ]:


X = df_players[['TWITTER_RETWEET_COUNT']]
Y = df_players['SALARY_MILLIONS']

# Relationship is not linear, we use a polynomial regression
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)


# In[ ]:


# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X_, Y)
sc = lr_model.score(X_, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X_)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['TWITTER_RETWEET_COUNT'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# **Conclusion:** By building a second order polynomial model, we have reduced the **bias** of our predictions, but also increased the **variance** of our model. It is quite likely that our model will tend to overfit to the already-seen data and it won't generalize well. Still, twitter interactions are a poor predictor in comparison to the POINTS or even the ORPM.

# ## How is social impact related with the salary? Wikipedia <a class="anchor" id="8_wikipedia"></a>

# In[ ]:


sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", order=2, data=df_players)
plt.show()


# In[ ]:


sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", order=2, data=data)
plt.show()


# In[ ]:


X = df_players[['PAGEVIEWS']]
Y = df_players['SALARY_MILLIONS']

# Relationship is not linear, we use a polynomial regression
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)


# In[ ]:


# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X_, Y)
sc = lr_model.score(X_, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X_)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['PAGEVIEWS'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# **Conclusion:** The views in the Wikipedia webpage performs slightly better than the twitter interactions to explain the salary of the player. Still, we have built a second order polynomial model that won't probably generalize well to unseen players or new seasons. 

# ## Is salary determined by the age? <a class="anchor" id="9_age"></a>
# 
# We finally analyze the age of the player as a factor to determine his salary.

# In[ ]:


sns.lmplot(x="AGE", y="SALARY_MILLIONS", data=df_players)
plt.show()


# In[ ]:


X = df_players[['AGE']]
Y = df_players['SALARY_MILLIONS']

# Build linear regression model
lr_model = LinearRegression(fit_intercept=True, normalize=False)
lr_model.fit(X, Y)
sc = lr_model.score(X, Y)
print('R2 score: %.3f' % sc)


# In[ ]:


y_pred = lr_model.predict(X)
rmse = math.sqrt(mean_squared_error(Y, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Calculate Pearson correlation coefficient between the two variables
corr, _ = pearsonr(df_players['AGE'], df_players['SALARY_MILLIONS'])
print('Pearson correlation coefficient: %.3f' % corr)


# **Conclusion:** Somehow surprisingly, age is not a key factor to explain the salary. The reason may be the way the salary range is designed in the NBA, where the salaries of the players tend to draw a curve, where the best contracts are signed during the middle period of their career. In this case, it could have made sense to use a second order polynomial model to explain the salary based on the age.

# ## Using the previous features combined to predict the salary <a class="anchor" id="10_all_together"></a>
# 
# We have analyzed the influence of each feature in the player salary independently. Let's say that our team wants to sign a new player and we do not know how much money he deserves and ignore the fact that both our predictors and salary correspond to the same year (biasing therefore our predictions). We then build a predictive model that combines all of the features above to predict the player salary for the next season.
# 
# Most of the models are able to deal with numeric values. However, we have one categorical column, which is the **POSITION**. We are going to transform this information to numerical, but maintaining the order in which this position are disposed in the basket court.

# In[ ]:


mapping = { "PG":1, "SG":2, "SF":3, "PF":4, "C":5}
df_players['POSITION_NUM'] = df_players['POSITION'].map(mapping).copy()
df_players


# We split our data into train and test and build the model on our train set to later check the predictions on the test set. To address the goodness of our predictions, we apply the same metrics as before (R2 and RMSE).

# In[ ]:


X = df_players[['PAGEVIEWS', 'TWITTER_FAVORITE_COUNT', 'TWITTER_RETWEET_COUNT', 'MPG', 
                'POINTS', 'DRPM', 'ORPM', 'POSITION_NUM', 'AGE']]
Y = df_players['SALARY_MILLIONS']


# In[ ]:


# Separate train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[ ]:


lr_model = LinearRegression(fit_intercept=True, normalize=True)
lr_model.fit(X_train, y_train)
sc = lr_model.score(X_train, y_train)
print('R2 score: %.3f' % sc)


# In[ ]:


# Make predictions over train set
y_pred = lr_model.predict(X_train)
rmse = math.sqrt(mean_squared_error(y_train, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Make predictions over test set
y_pred = lr_model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# **Conclusion:** Althout our model still shows significant bias, it performs much better than any of the features considered independently. In fact, our best predictor so far (POINTS per game) only achieved a R2=0.41 and RMSE=5.31, on the train set. Which is significantly worse than the resulting metrics for this model on the test set. Additionally, the model behaves relatively well for unseen data. Still, there is a lot of room for improvement as, on average, **we are off by more than 4.5M per player**, which is an very important amount of money.
# 
# Finally, we are going to examine the regression coefficients as an indicator for the importance of each variable on our model. However, this is only an indicator as we cannot compare these coefficients direclty, since the scale varies largely between the different predictors in our model. In other words, the units in which we measure the twitter interactons have nothing to do with the points per game, which makes it impossible to compare them directly.

# In[ ]:


print(lr_model.coef_)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.bar(X.columns, lr_model.coef_, color=(0.8, 0.2, 0.6, 0.8))
plt.xticks(rotation=90)
plt.title('Standardized regression coefs.')
plt.show()


# ### Using a second order linear model <a class="anchor" id="10_1_second_order"></a>
# 
# Finally, we study the performance of a second order linear model only to observe how the predictions will improve for the train set, but they degrade considerably when using the same model to infer salaries on the test set.

# In[ ]:


# Relationship is not linear, we use a polynomial regression
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)


# In[ ]:


# Separate train and test
X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=0.20, random_state=42)


# In[ ]:


lr_model = LinearRegression(fit_intercept=True, normalize=True)
lr_model.fit(X_train, y_train)
sc = lr_model.score(X_train, y_train)
print('R2 score: %.3f' % sc)


# In[ ]:


# Make predictions over train set
y_pred = lr_model.predict(X_train)
rmse = math.sqrt(mean_squared_error(y_train, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


# Make predictions over test set
y_pred = lr_model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))


# In[ ]:


print(lr_model.coef_)

