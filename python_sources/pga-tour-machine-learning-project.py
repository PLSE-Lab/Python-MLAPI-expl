#!/usr/bin/env python
# coding: utf-8

# # PGA Tour Machine Learning Project 

# ## Can We Predict If a PGA Tour Player Won a Tournament in That Year and Their Earnings?

# Having grown up watching golf, I have always been interested in exploring what sets the best golfers (golfers with wins) apart from the rest. Therefore, I decided to explore their statistics. To collect all the data, I scraped the data from the [PGA Tour website](https://www.pgatour.com/stats.html) using python libraries such as beautifulsoup. (The code for the data collection is included in the repository)
# 
# From this data, I performed an exploratory data analysis to explore the distribution of players on numerous aspects of the game, discover outliers, and further explore how the game has changed from 2010 to 2018. I also utilized numerous supervised machine learning models to predict a golfer's earnings and wins. 
# 
# To predict the golfer's win, I used multiple classification methods such as logisitic regression, SVM (Support Vector Machines), and Random Forest Classification. I found that I had the best performance with the Random Forest Classification method. To predict the golfer's earnings, I used linear regression, polynomial features with linear regression, and ridge regression. 

# ## <a id='TOC'>Table of Contents</a>
# <ol>
# <li><a href='#section 1'>Description of the Data</a></li>
# <li><a href='#section_2'>Data Cleaning</a></li>
# <li><a href='#section_3'>Exploratory Data Analysis</a></li>
# <li><a href='#section_4'>Machine Learning Model (Classification)</a></li>
# <li><a href='#section_5'>Machine Learning Model (Regression)</a></li>
# <li><a href='#section_6'>Conclusion</a></li>
# </ol>

# ## 1. <a id='section_1'>Description of the Data</a>
# <a href='#TOC'>Back to table of Contents</a>
#   
# pgaTourData.csv contains 1674 rows and 18 columns. Each row indicates a golfer's performance for that year.
# 
# - Player Name: Name of the golfer
# - Rounds: The number of games that a player played  
# - Fairway Percentage: The percentage of time a tee shot lands on the fairway
# - Year: The year in which the statistic was collected 
# - Avg Distance: The average distance of the tee-shot 
# - gir: (Green in Regulation) is met if any part of the ball is touching the putting surface while the number of strokes taken is at least two fewer than par
# - Average Putts: The average number of strokes taken on the green 
# - Average Scrambling: Scrambling is when a player misses the green in regulation, but still makes par or better on a hole
# - Average Score: Average Score is the average of all the scores a player has played in that year 
# - Points: The number of FedExCup points a player earned in that year. These points can be earned by competing in tournaments.
# - Wins: The number of competition a player has won in that year 
# - Top 10: The number of competitions where a player has placed in the Top 10
# - Average SG Putts: Strokes gained: putting measures how many strokes a player gains (or loses) on the greens.
# - Average SG Total: The Off-the-tee + approach-the-green + around-the-green + putting statistics combined
# - SG:OTT: Strokes gained: off-the-tee measures  player performance off the tee on all par-4s and par-5s. 
# - SG:APR: Strokes gained: approach-the-green measures player performance on approach shots. Approach shots include all shots that are not from the tee on par-4 and par-5 holes and are not included in strokes gained: around-the-green and strokes gained: putting. Approach shots include tee shots on par-3s.
# - SG:ARG: Strokes gained: around-the-green measures player performance on any shot within 30 yards of the edge of the green. This statistic does not include any shots taken on the putting green.
# - Money: The amount of prize money a player has earned from tournaments
# 
# The official explanation for strokes gained is included [here](https://www.pgatour.com/news/2016/05/31/strokes-gained-defined.html).

# ### Importing Packages 

# In[ ]:


# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing and Examining the Data

# In[ ]:


# Importing the data 
df = pd.read_csv('../input/pgaTourData.csv')

# Examining the first 5 data
print(df.head())


# In[ ]:


df.info()


# In[ ]:


df.shape


# We can see that the data has 1674 rows and 18 columns.

# ## 2. <a id='section_2'>Data Cleaning</a>
# <a href='#TOC'>Back to table of Contents</a>
# 
# From a rough look at the initial data, I realized that the data needs to be further cleaned. 
# - For the columns Top 10 and Wins, convert the NaNs to 0s. 
# - Change Top 10 and Wins into an int
# - Drop NaN values for players who do not have the full statistics
# - Change the columns Rounds into int 
# - Change points to int 
# - Remove the dollar sign ($) and commas in the column Money

# In[ ]:


# Replace NaN with 0 in Top 10 
df['Top 10'].fillna(0, inplace=True)
df['Top 10'] = df['Top 10'].astype(int)

# Replace NaN with 0 in # of wins
df['Wins'].fillna(0, inplace=True)
df['Wins'] = df['Wins'].astype(int)

# Drop NaN values 
df.dropna(axis = 0, inplace=True)


# In[ ]:


# Change Rounds to int
df['Rounds'] = df['Rounds'].astype(int)

# Change Points to int 
df['Points'] = df['Points'].apply(lambda x: x.replace(',',''))
df['Points'] = df['Points'].astype(int)

# Remove the $ and commas in money 
df['Money'] = df['Money'].apply(lambda x: x.replace('$',''))
df['Money'] = df['Money'].apply(lambda x: x.replace(',',''))
df['Money'] = df['Money'].astype(float)


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ## 3. <a id='section_3'>Exploratory Data Analysis</a>
# <a href='#TOC'>Back to table of Contents</a>

# ### Distribution of the Data 

# In[ ]:


# Looking at the distribution of data
f, ax = plt.subplots(nrows = 6, ncols = 3, figsize=(20,20))
distribution = df.loc[:,df.columns!='Player Name'].columns
rows = 0
cols = 0
for i, column in enumerate(distribution):
    p = sns.distplot(df[column], ax=ax[rows][cols])
    cols += 1
    if cols == 3:
        cols = 0
        rows += 1


# From the distributions plotted, It appears that most of the graphs are normally distributed. However, we can observe that Money, Points, Wins, and Top 10s tend to are all skewed to the right. This could be explained by the separation of the best players and the average PGA Tour player. The best players have multiple placings in the Top 10 with wins that allows them to earn more from tournaments, while the average player will have no wins and only a few Top 10 placings that prevent them from earning as much. 

# ### Exploring Wins and Top 10 Placings by Year

# In[ ]:


# Looking at the number of players with Wins for each year 
win = df.groupby('Year')['Wins'].value_counts()
win = win.unstack()
win.fillna(0, inplace=True)

# Converting win into ints
win = win.astype(int)

print(win)


# From this table, we can see that most players end the year without a win. In fact it is pretty rare to find a player that has won more than once. 

# In[ ]:


# Looking at the percentage of players without a win in that year 
players = win.apply(lambda x: np.sum(x), axis=1)
percent_no_win = win[0]/players
percent_no_win = percent_no_win*100
print(percent_no_win)


# In[ ]:


# Plotting percentage of players without a win each year 
fig, ax = plt.subplots()
bar_width = 0.8
opacity = 0.7 
index = np.arange(2010, 2019)

plt.bar(index, percent_no_win, bar_width, alpha = opacity)
plt.xticks(index)
plt.xlabel('Year')
plt.ylabel('%')
plt.title('Percentage of Players without a Win')


# From the box plot above, we can observe that the percentages of players without a win are around 80%. There wa also a negligible amount of variation in the percentage of players without a win in the past 8 years. 

# In[ ]:


# Plotting the number of wins on a bar chart 
fig, ax = plt.subplots()
index = np.arange(2010, 2019)
bar_width = 0.2
opacity = 0.7 

def plot_bar(index, win, labels):
    plt.bar(index, win, bar_width, alpha=opacity, label=labels)

# Plotting the bars
rects = plot_bar(index, win[0], labels = '0 Wins')
rects1 = plot_bar(index + bar_width, win[1], labels = '1 Wins')
rects2 = plot_bar(index + bar_width*2, win[2], labels = '2 Wins')
rects3 = plot_bar(index + bar_width*3, win[3], labels = '3 Wins')
rects4 = plot_bar(index + bar_width*4, win[4], labels = '4 Wins')
rects5 = plot_bar(index + bar_width*5, win[5], labels = '5 Wins')

plt.xticks(index + bar_width, index)
plt.xlabel('Year')
plt.ylabel('Number of Wins')
plt.title('Distribution of Wins each Year')
plt.legend()


# By looking at the distribution of Wins each year, we can see that it is rare for most players to even win a tournament in the PGA Tour. Majority of players do not win, and a very few number of players win more than once a year.

# In[ ]:


# Percentage of people who did not place in the top 10 each year
top10 = df.groupby('Year')['Top 10'].value_counts()
top10 = top10.unstack()
top10.fillna(0, inplace=True)
players = top10.apply(lambda x: np.sum(x), axis=1)

no_top10 = top10[0]/players * 100
print(no_top10)


# By looking at the percentage of players that did not place in the top 10 by year, We can observe that only approximately 20% of players did not place in the Top 10. In addition, the range for these player that did not place in the Top 10 is only 9.47%. This tells us that this statistic does not vary much on a yearly basis. 

# ### Exploring the Longest Hitters

# In[ ]:


# Who are some of the longest hitters 
distance = df[['Year','Player Name','Avg Distance']].copy()
distance.sort_values(by='Avg Distance', inplace=True, ascending=False)
print(distance.head())


# We can see that Rory McIlroy is one of the longest hitters in the game, setting the average driver distance to be 319.7 yards in 2018. He was also the longest hitter in 2017 with an average of 316.7 yards. There are other notable players like J.B. Holmes and Dustin Johnson who have an average of over 317 yards.

# ### Exploring the Earnings of Players

# In[ ]:


# Who made the most money
money_ranking = df[['Year','Player Name','Money']].copy()
money_ranking.sort_values(by='Money', inplace=True, ascending=False)
print(money_ranking.head())


# We can see that Jordan Spieth has made the most amount of money in a year. Earning an outstanding total of 12 million dollars

# In[ ]:


# Who made the most money each year
money_rank = money_ranking.groupby('Year')['Money'].max()
money_rank = pd.DataFrame(money_rank)
print(money_rank.iloc[0,0])

indexs = np.arange(2010, 2019)
names = []
for i in range(money_rank.shape[0]):
    temp = df.loc[df['Money'] == money_rank.iloc[i,0],'Player Name']
    names.append(str(temp.values[0]))

money_rank['Player Name'] = names
print(money_rank)


# With this table, we can examine the earnings of each player by year. Some of the most notable were Jordan Speith's earning of 12 million dollars and Justin Thomas earning the most money in both 2017 and 2018. 

# ### Golf Statistics over Time

# In[ ]:


# Looking at the changes in statistics over time 
f, ax = plt.subplots(nrows = 5, ncols = 3, figsize=(35,65))
distribution = df.loc[:,(df.columns!='Player Name') & (df.columns!='Wins')].columns
distribution = distribution[distribution != 'Year']

print(distribution)
rows = 0
cols = 0
for i, column in enumerate(distribution):
    p = sns.boxplot(x = 'Year', y = column, data=df, ax=ax[rows][cols], showfliers=False)
    p.set_ylabel(column,fontsize=20)
    p.set_xlabel('Year',fontsize=20)
    cols += 1
    if cols == 3:
        cols = 0
        rows += 1


# Something that I found interesting by plotting each variable across time was majority of the statistics had little to no change for Professional Golfers in the past 8 years. However, some of the areas where there were changes were in Money, Average Score, and Rounds. 
# 
# This was rather interesting as golf club manufacturers would often advertise about the huge improvements in distance for players when they switched to their latest club. But in fact, there was only an increase in the average distance of 10 yards. 

# ### Comparing the Average and Champions

# In[ ]:


# Defining the players that had a win or more in each year 
champion = df.loc[df['Wins'] >= 1, :]
print(champion.head())


# In[ ]:


f, ax = plt.subplots(nrows = 8, ncols = 2, figsize=(35,65))
distribution = df.loc[:,df.columns!='Player Name'].columns
distribution = distribution[distribution != 'Year']

rows = 0
cols = 0
lower_better = ['Average Putts', 'Average Score']
for i, column in enumerate(distribution):
    avg = df.groupby('Year')[column].mean()
    best = champion.groupby('Year')[column].mean()
    ax[rows,cols].plot(avg, 'o-',)
    ax[rows,cols].plot(best, 'o-',)
    ax[rows,cols].set_title(column, fontsize = 20)
    
    cols += 1
    if cols == 2:
        cols = 0
        rows += 1


# From the Graphs above, we can see the average scores of the best players (players with a win) versus the PGA Tour average. This can give us an indication to which statistics help players win. 
# 
# We can see that the fairway percentage and greens in regulations does not seems to contribute as much to a player's win. However, we can see that all the strokes gained statistics have a large impact on the wins of these players. In addition, we can see that the average score and average putts are lower for players with a win. 

# ### Correlation

# In[ ]:


# Plot the correlation matrix between variables 
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='coolwarm')


# In[ ]:


df.corr()['Wins']


# From the correlation matrix, we can observe that Money is highly correlated to wins along with the FedExCup Points. We can also observe that the fairway percentage, year, and rounds are not correlated to Wins.

# ## 4. <a id='section_4'>Machine Learning Model (Classification)</a>
# <a href='#TOC'>Back to table of Contents</a>
# 
# To predict winners, I used multiple machine learning models to explore which models could accuracy classify if a player is going to win in that year. 
# 
# To measure the models, I used Receiver Operating Characterisitc Area Under the Curve. (ROC AUC) The ROC AUC tells us how capable the model is at distinguishing players with a win. In addition, as the data is skewed with 83% of players having no wins in that year, ROC AUC is a better measure than the accuracy of the model. 

# In[ ]:


# Importing the Machine Learning modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)


# ## Preparing the Data for Classification 
# 

# We know from the calculation above that the data for wins is skewed. Even without machine learning we know that approximately 83% of the players does not lead to a win. Therefore, we will be utilizing ROC AUC as the primary measure of these models

# In[ ]:


# Adding the Winner column to determine if the player won that year or not 
df['Winner'] = df['Wins'].apply(lambda x: 1 if x>0 else 0)

# New DataFrame 
ml_df = df.copy()

# Y value for machine learning is the Winner column
target = df['Winner']

# Removing the columns Player Name, Wins, and Winner from the dataframe
ml_df.drop(['Player Name','Wins','Winner'], axis=1, inplace=True)
print(ml_df.head())


# ## Logistic Regression

# In[ ]:


per_no_win = target.value_counts()[0] / (target.value_counts()[0] + target.value_counts()[1])
per_no_win = per_no_win.round(4)*100
print(str(per_no_win)+str('%'))


# In[ ]:


# Function for the logisitic regression 
def log_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 10)
    clf = LogisticRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
         .format(clf.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
         .format(clf.score(X_test, y_test)))
    cf_mat = confusion_matrix(y_test, y_pred)
    confusion = pd.DataFrame(data = cf_mat)
    print(confusion)
    
    print(classification_report(y_test, y_pred))
    
    # Returning the 5 important features 
    rfe = RFE(clf, 5)
    rfe = rfe.fit(X, y)
    print('Feature Importance')
    print(X.columns[rfe.ranking_ == 1].values)
    
    print('ROC AUC Score: {:.2f}'.format(roc_auc_score(y_test, y_pred)))


# In[ ]:


log_reg(ml_df, target)


# From the logisitic regression, we got an accuracy of 0.9 on the training set and an accuracy of 0.91 on the test set. This was surprisingly accurate for a first run. However, the ROC AUC Score of 0.78 could be improved. Therefore, I decided to add more features as a way of possibly improving the model.

# ### Feature Engineering

# In[ ]:


# Adding Domain Features 
ml_d = ml_df.copy()
# Top 10 / Money might give us a better understanding on how well they placed in the top 10
ml_d['Top10perMoney'] = ml_d['Top 10'] / ml_d['Money']

# Avg Distance / Fairway Percentage to give us a ratio that determines how accurate and far a player hits 
ml_d['DistanceperFairway'] = ml_d['Avg Distance'] / ml_d['Fairway Percentage']

# Money / Rounds to see on average how much money they would make playing a round of golf 
ml_d['MoneyperRound'] = ml_d['Money'] / ml_d['Rounds']


# In[ ]:


log_reg(ml_d, target)


# In[ ]:


# Adding Polynomial Features to the ml_df 
mldf2 = ml_df.copy()
poly = PolynomialFeatures(2)
poly = poly.fit(mldf2)
poly_feature = poly.transform(mldf2)
print(poly_feature.shape)

# Creating a DataFrame with the polynomial features 
poly_feature = pd.DataFrame(poly_feature, columns = poly.get_feature_names(ml_df.columns))
print(poly_feature.head())


# In[ ]:


log_reg(poly_feature, target)


# From feature engineering, there were no improvements in the ROC AUC Score. In fact as I added more features, the accuracy and the ROC AUC Score decreased. This could signal to us that another machine learning algorithm could better predict winners.

# ### SVM (Support Vector Machine)

# In[ ]:


def svc_class(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 10)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    svclassifier = SVC(kernel='rbf', C=10000)  
    svclassifier.fit(X_train_scaled, y_train) 
    y_pred = svclassifier.predict(X_test_scaled) 
    print('Accuracy of SVM on training set: {:.2f}'
         .format(svclassifier.score(X_train_scaled, y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
         .format(svclassifier.score(X_test_scaled, y_test)))

    
    print('ROC AUC Score: {:.2f}'.format(roc_auc_score(y_test, y_pred)))


# In[ ]:


svc_class(ml_df, target)


# In[ ]:


svc_class(ml_d, target)


# In[ ]:


svc_class(poly_feature, target)


# With Support Vector Machines, the ROC AUC Scores were significantly better. The SVM scored a 0.89 on the data with domain features that I included compared to the score of 0.75 on the logisitic regression with polynomial features.

# ### Random Forest Model
# 

# In[ ]:


def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 10)
    clf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
         .format(clf.score(X_train, y_train)))
    print('Accuracy of Random Forest classifier on test set: {:.2f}'
         .format(clf.score(X_test, y_test)))
    
    cf_mat = confusion_matrix(y_test, y_pred)
    confusion = pd.DataFrame(data = cf_mat)
    print(confusion)
    
    print(classification_report(y_test, y_pred))
    
    # Returning the 5 important features 
    rfe = RFE(clf, 5)
    rfe = rfe.fit(X, y)
    print('Feature Importance')
    print(X.columns[rfe.ranking_ == 1].values)
    
    print('ROC AUC Score: {:.2f}'.format(roc_auc_score(y_test, y_pred)))


# In[ ]:


random_forest(ml_df, target)


# In[ ]:


random_forest(ml_d, target)


# In[ ]:


random_forest(poly_feature, target)


# The Random Forest Model was scored highly on ROC AUC Score, obtaining a value of 0.89. With this, we observed that the Random Forest Model and the Support Vector Machine Models could accurately classify players with and without a win. 

# ## 5. <a id='section_5'>Machine Learning Model (Regression)</a>
# <a href='#TOC'>Back to table of Contents</a>
# 
# Can we predict a golfer's earnings by only looking at their statistics. (Not looking at their placings in the year)

# ### Preparing the Data for Regression 

# In[ ]:


# New DataFrame 
earning_df = df.copy()

# Y value for machine learning is the Money column
target = earning_df['Money']

# Removing the columns Player Name, Wins, Winner, Points, Top 10, and Money from the dataframe
earning_df.drop(['Player Name','Wins','Winner','Points','Top 10','Money'], axis=1, inplace=True)

print(earning_df.head())


# In[ ]:


# Importing the Machine Learning modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


def linear_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)
    clf = LinearRegression().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('R-Squared on training set: {:.3f}'
          .format(clf.score(X_train, y_train)))
    print('R-Squared on test set {:.3f}'
          .format(clf.score(X_test, y_test)))
    
    print('linear model coeff (w):\n{}'
         .format(clf.coef_))
    print('linear model intercept (b): {:.3f}'
         .format(clf.intercept_))


# In[ ]:


linear_reg(earning_df, target)


# In[ ]:


# Creating a Polynomial Feature to improve R-Squared
poly = PolynomialFeatures(2)
poly = poly.fit(earning_df)
poly_earning = poly.transform(earning_df)
print(poly_feature.shape)

# Creating a DataFrame with the polynomial features 
poly_earning = pd.DataFrame(poly_feature, columns = poly.get_feature_names(earning_df.columns))


# In[ ]:


linear_reg(poly_earning, target)


# In[ ]:


# Adding a regularization penalty (Ridge)
def linear_reg_ridge(X, y, al):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 10)
    clf = Ridge(alpha = al).fit(X_train, y_train)

    print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
         .format(clf.score(X_train, y_train)))
    print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
         .format(clf.score(X_test, y_test)))
    
    print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
         .format(clf.coef_))
    print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
         .format(clf.intercept_))


# In[ ]:


linear_reg_ridge(poly_earning, target, al = 1)


# In[ ]:


linear_reg_ridge(poly_earning, target, al = 100)


# Out of the 3 models that I implemented, I had the most success with the ridge regression with a polynomial degree of 2 and an alpha of 1. This ridge regression had a R-squared value of 0.770 which was only slightly better than the polynomial regression. 

# ### Cross Validation 

# In[ ]:


from sklearn.model_selection import cross_val_score

def cross_val(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)
    clf = Ridge().fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=5)
    
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)


# In[ ]:


cross_val(poly_earning, target)


# ### Application of the Linear Regression Model 

# In[ ]:


# Using the Linear Regression to predict Tiger Wood's Earnings based on the Model
def find_earning(X,y,name,year):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   random_state = 10)
    clf = Ridge().fit(X_train, y_train)
    y_pred = clf.predict(X)
    y_pred = pd.Series(y_pred)

    pred_data = pd.concat([X, y_pred], axis=1)
    pred_name = pd.concat([pred_data, df['Player Name']], axis=1)

    return pred_name.loc[(pred_name['Player Name']==name) & (pred_name['Year']==year), 0]


# ## 6. <a id='section_6'>Conclusion</a>
# <a href='#TOC'>Back to table of Contents</a>

# ### What I Learned
# 
# From this notebook, I learned about numerous aspects of the game that differentiate the winner and the average PGA Tour player. For example, we can see that the fairway percentage and greens in regulations does not seems to contribute as much to a player's win. However, all the strokes gained statistics contribute pretty highly to wins for these players. It was interesting to see which aspects of the game that the professionals should put their time into. This also gave me the idea of track my personal golf statistics, so that I could compare it to the pros and find areas of my game that need the most improvement. 
# 
# ### Machine Learning Model
# 
# From this PGA Tour EDA and Machine Learning Models, I was able to examine the data of PGA Tour players, classify if a player will win that year or not, and predict their earnings. While, I believe that I can improve my prediction of their earnings, I am satisfied with my classification and regression model. With the random forest classification model, I was able to achieve an ROC AUC of 0.89 and an accuracy of 0.95 on the test set. This was a significant improvement from the ROC AUC of 0.78 and accuracy of 0.91. Because the data is skewed with approximately 80% of players not earning a win, the primary measure of the model was the ROC AUC. I was able to improve my model from ROC AUC score of 0.78 to a score of 0.89 by simply trying 3 different models, adding domain features, and polynomial features.
# 
# ### Moving Forward
# 
# Having done a simple regression model on predicting the earnings of golfers, I would like to come back to this project again with a deeper understanding of other regression models to attempt to better predict the earnings of golfers. As shown below, the model predicted that Tiger Woods would make 1.3 million dollars in the year 2013. But in fact, Tiger Woods made 8 million dollars. Tiger Woods is one of the best players in the world is definately an outlier. However, I would like to come up with a model that can better predict the earnings of even the best players in the world. 
# 
# If you made it to the end of this notebook, I hope you learned something about the PGA Tour statistics!

# In[ ]:


print('Tiger Woods\' Predicted Earning: ' + 
      str(find_earning(X = poly_earning, y = target, name = 'Tiger Woods', year = 2013).values[0]))

# Tiger Wood's actual earnings in 2018 
tw13 = df.loc[(df['Player Name']=='Tiger Woods') & (df['Year']==2013), 'Money']
print('Tiger Woods\' Actual Earning: ' + str(tw13.values[0]))

