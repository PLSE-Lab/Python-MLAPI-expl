#!/usr/bin/env python
# coding: utf-8

# # Overview
# In this notebook, I'll take a look at the 2015-2019 NFL team stats to try to determine what statistics actually contribute to more wins in the regular season. Football's unique regular season structure and large roster sizes/personnel requirements has historically made it difficult to predict how teams will perform over time. It's difficult to use direct comparisons given the small sample size of games played and possible match-up problems resulting in amplication of biased data. However, the statistics that teams are able to put up overall through the season should give some indicators to their actual ability and what kind of results they could put up against the rest of the league.
# 
# An additional inspiration for this notebook was the 2011 film, *Moneyball*, which covered the assembly of a competitive baseball team out of players who were undervalued relative to their actual contributions to winning. We will attempt to do something similar in football based on the following questions:
# 
# Are there certain stats that teams should be chasing more? Do teams actually need to have a offense balanced with passing and rushing in order to do well (in the regular season)? Is it possible to succeed with a weak defense but a strong offense or vice versa?

# # Method
# To explore and evaluate these questions, I'll use multivariable linear regression that is available in python through the scikit-learn library. Linear regression is a model which measures the linear relationship between a target (dependent variable) and one or more features (dependent variables). In our case, I believe that there will be multiple team stats that are tracked by the league which have a linear relationship with the number of wins that the team has over the course of a regular season, so rather than a single linear regression, we'll use a multivariable one.
# 
# Linear regression assumes a few things:
# * That a linear relationship exists between the target and its features (otherwise the linear regression will not fit well)
# * That there is little to no relationship between the features (otherwise their effect on one another may affect the target)
# * That there is no auto-correlation (otherwise values of the target will affect one another)
# 
# **Data**
# Unfortunately, the data on pro-football-reference.com is available in multiple tables, rather than a single table. Also, rather than being formatted for data manipulation and processing, it is primarily for web viewing legibility. Thus, some columns contain similar headings (which prevents us from working with them easily in Python to do the cleaning), and larger aggregate column headers that describe multiple columns. Problems with the loading of this data can be seen below.
# 
# In order to address some of these issues, I will recreate the dataset in a usable format prior to actual analysis:
# * Strip unused aggregate column headers so we only have columns with actionable data
# * Remove the ranking data, since it is an ordinal value with no way numerical way to evalute how much better a team is than another
# * Rename similar titled columns to better explain their use, and avoid duplicate operations in Python
# * Add a Wins columns which our other columns can be correlated against
# * Add Years column so that we can use data from multiple years
# * Combine team defensive stats with offensive stats and match to corresponding teams, since default table is a ranked listing (if columns weren't shared, we might be able to do this in the pandas dataframe, rather than manually)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sea


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


fString = "/kaggle/input/nfl-example/2019_nfl_preclean.csv" # use the built-in Kaggle code printed file names for input data

teams = pd.read_csv(fString) # create a pandas dataframe and assign it to the teams data
teams.head() # preview the data and what it looks like


# In[ ]:


teams.info() # get a summary of our preview dataset


# Our actual column headers have gotten added as the first row within the pandas dataframe, rather than existing as keys which we would use to locate our data. This data only consists of 2019 team offensive stats, is missing the number of wins

# # Analysis

# In[ ]:


fileString = "/kaggle/input/all-nfl/all_nfl.csv" # file location for our complete dataset
data = pd.read_csv(fileString) # use the pandas dataframe to read the csv
data.head() # preview the data to make sure it looks right


# In[ ]:


data.info() # get a summary of our complete datset to ensure that nothing seems out of the place


# Our values look good after checking the info; the only strings in our dataset are the team names, which makes sense.
# Our target variable will be our 3rd column, 'W' (for wins), before we look at multivariable regression, let's see if there is any linearity between our features (every other column beside 'W', 'Year', and 'Tm').
# 
# It's probably generally not a good idea to try and determine linearity between 49 different variables and then trying to plot them, even with a smaller dataset like we're working with, so we'll instead look at the data and plot them according to the groupings that were originally provided from pro-football-reference: Total offensive stats, offensive stats broken out by type, total defensive stats, and defensive stats broken out by type.

# In[ ]:


plt.figure(figsize=(20,10)) # set our figure plot size
sea.pairplot(data=data[data.columns[2:9]], hue='W') # use the offensive total columns after team and year and plot them using seaborn pairwise plotting function


# We see strong correlations between columns with shared stats, like yards and yards/play for obvious reasons. More notably, points for and yardage also seem correlated, with teams with similar records being clustered somewhat together on these graphs. We can somewhat guess that points and yards are correlated since the more distance you are making it down the field, the more chances you will probably have to either score a touchdown, or a field goal. Additionally, the more down the field you are, the further an opponent would need to drive the other way for a chance to score themselves (and the unfortunate reality is that you need to score points to win a football game). It would have been nice to see distributions for say, > 10 wins break away from those with less. There is good clustering in the turnovers lost column at least, where the teams with more wins were distributed much more toward the lower end and had relatively tight distributions. One other correlation would appear to be turnovers lost and points earned, since the give up possessions, the less opportunities you have to score. 
# 
# Seems like turnovers lost, points for, and yardage may be good columns to try some multivariable linear regression on.
# 
# We can take a closer look at the offensive statistics to see if specific types of yards have stronger correlations, or if some other correlations emerge when we slice up the data further.

# In[ ]:


plt.figure(figsize=(30,15)) # set our figure plot size larger since there are more columns to account for
offensive_data = data.iloc[:, np.r_[2,10:27]] # based on our earlier data, grab the index strings using numpy's slice concatenation abizlity
sea.pairplot(data=offensive_data[offensive_data.columns[:]], hue='W') # plot again


# Again, there are the obvious correlations between shared statistics. Interestingly, even though there is a strong linear relationship between rushing yards and rushing first downs, teams with more wins tend to cluster near the center of graph, rather than toward the upper right, where we might expect them to. The same seems to be true for passing yards and passing first downs. It's possible that you don't really need to have the most 1st downs, as long as you're getting them at the right opportunities (i.e., is it better to get a single 80 yard play and be in the end zone, or 7, but then get stopped at that point? In order to score more, likely the former, although there are merits to the latter, as we visually observe at the end of games).
# 
# Nothing particularly stands out in these relationships however, and it doesn't seem to provide us that much additional clarity compared to the total offensive correlation. So let's examine the defensive stats next.

# In[ ]:


plt.figure(figsize=(20,10)) # set our figure plot size small again
total_defensive_data = data.iloc[:, np.r_[2,28:35]] # slice for new defensive total columns
sea.pairplot(data=total_defensive_data[total_defensive_data.columns[:]], hue='W') # plot defensive totals


# Use similar visual methods as earlier, it would appear that teams that limit points and yardage tend to rack up more wins. Surprisingly, yards given up and 1st downs allowed don't seem to track in terms of overall record, despite the fairly linear relationship. This probably suggests that defense is only one side of the story for teams.
# 
# Again, let's try breaking out the individual statistics on defense to see whether there are defensive stats we should weigh more heavily.

# In[ ]:


plt.figure(figsize=(20,15)) # set our figure plot size
ind_defensive_data = data.iloc[:, np.r_[2,36:52]] # slice
sea.pairplot(data=ind_defensive_data[ind_defensive_data.columns[:]], hue='W') # plot individual defensive data


# The most notable feature is that the distribution of wins for rushing attempts against really clustered better teams lower than higher. Quick intuition based on watched games would suggest that in games that teams win, the opponent is trying to catch up in score, and it's certainly faster to do so via passes (in addition to saving clock time for additional comebacks), weighing this one more heavily. We'll flag this as a possible column to watch as well.
# 
# This gives us: **points for, points against, yardage for, turnovers lost, and rushing attempts** against as our coefficients that we'll want to calculate for our linear regression to calculate win totals.

# In[ ]:


data.info() # grab the column titles against based on our conclusions and what we want to evaluate against


# columns 3, 4, 7, 28, and 42 are the ones that we're interested in.

# In[ ]:


features = data.iloc[:, np.r_[3:5, 7, 28,42]] # create a new dataframe with the all rows for the columns we are interested in
features # preview the dataframe


# In[ ]:


target = data['W'] # isolate the wins column
target # preview


# In[ ]:


line_reg = LinearRegression() # create our linear regression model
line = line_reg.fit(features, target) # train it based on our 2015-2019 data
r2 = line.score(features, target) # get the R^2 value for our model
print(r2)


# This isn't a particularly good R^2, but for the sake of trying it, let's see our results.

# In[ ]:


est = line_reg.predict(features) # use our model to calculate the expected wins
est_data = round(data['W']-est) # round our subtraction
est_data.values
est_data.value_counts() # find out what the distribution of our values is


# Okay, that wasn't particularly great, as only 52/160 were results were predicted correctly, which is less than 2 seasons worth of results. However, 74/160 were within 1 game of their actual result. Let's see if this model works a different season of the NFL?

# In[ ]:


file2014 = "/kaggle/input/2014-nfl/2014_nfl.csv" # file location for our 2014 data
data_2014 = pd.read_csv(file2014) # read csv
features_2014 = data_2014.iloc[:, np.r_[3:5, 7, 28,42]]
est_2014 = line_reg.predict(features_2014) # use our model to calculate the expected wins
est14_data = round(data_2014['W']-est_2014) # round our subtraction
est14_data.values
est14_data.value_counts() # find out what the distribution of our values is


# Using our existing model on 2014 data, we get 11/32 ~perfect predictions, 16/32 1 off, and 5/32 2 off. This seems to perform slightly better than our training dataset, but it still isn't a particularly good model, as we already saw with the R^2 score. We could do a statistical significance check here to confirm if there was any difference in the results, but we'll omit it for now.
# 
# The overall results: 
# * ~32.8% (63/192) perfectly modeled
# * ~46.9% (90/192) 1 off
# * ~17.7% (34/192) 2 off
# * ~2.6% (5/192) 3 off
# 
# Our linear regression doesn't seem to be a particularly good estimator of a team's results over the course of a regular season in the NFL.
# 
# For the sake of experiment, let's also see how an overfit model might work, where we didn't pick some of the more linearly correlated columns.

# In[ ]:


features_all = data.iloc[:, np.r_[3:53]] # create features out of everything
features_all

all_reg = LinearRegression() # create our linear regression model
over = all_reg.fit(features_all, target) # train it based on our 2015-2019 data
r22 = over.score(features_all, target) # get the R^2 value for our model
print(r22)


# Although our R^2 improvess, given that we added 48 more variables/features to fit the data, the improvement is pretty low for the additional compution that was required. 
# 
# Let's see how the results pan out using the model.

# In[ ]:


est_all = all_reg.predict(features_all) # use our new overfit model
est_all_data = round(data['W']-est_all) # round the predictions
est_all_data.values
est_all_data.value_counts() # find out what the distribution of our values is


# In[ ]:


features_all_2014 = data_2014.iloc[:, np.r_[3:53]] # try the 2014 data now
over_est_2014 = all_reg.predict(features_all_2014) 
over_est14_data = round(data_2014['W']-over_est_2014) # round
over_est14_data.values
over_est14_data.value_counts()


# Our overall results with the new "overfit" model:
# * ~32.3% (62/1920 perfect
# * ~52.1% (100/192) 1 off 
# * ~13.5% (26/192) 2 off
# * ~2.1% (4/192) 3 off
# 
# This does seem to be a slightly better prediction for our teams, but isn't particularly valuable for estimations.

# # Conclusion/Future Work
# It's unlikely that the linear regression we have produced would be valuable to teams, given it's poor predictive value. If teams are interested in their regular season performance (likely to figure out if they would make the playoffs), unless they are one of the best teams in the league, this prediction could have them missing the playoffs completely, whereas their actual performance has them getting in on a tiebreaker or something similar. Additionally, our findings are not exactly profound at this time. If you put up a lot of points/yards and limit the points your opponent scores, you're likely to do well. Rushing attempts against could also be explained by teams that need to catch up are going to need to do so via passing to preserve clock time. 
# 
# There are numerous shortcomings with the data and work that we've done:
# * Small sample size, only 32 teams over 5 years is not the largest dataset that could ever be assembled.
# * Regular season structure in the NFL makes it hard to trust the data that is available
# * No correction for rule changes that may have been made that has influenced the way that games are played over the time of the dataset
# * Football results are a zero-sum game, but the statistics themselves are not i.e., each game result is independent of your performance in the next one.
# * Our model also assumed that each team is "equivalent" when in fact there are performance differences due to their players and play schedule.
# 
# **Future work**
# Using the dataset that we've already compiled, it would be interesting to explore whether the wins can be better predicted using a polynomial regression. Also, rather than examining end of season totals, if we had each game as an individual row and examined data on a weekly basis, we might be able to eliminate bias against certain teams over the course of a season and also account for the zero-sum nature of win-losses, but non-zero sum nature of stats. An extension of this would include a discrete examination of the NFL might look at how much each player contributes to every win. 
# 
