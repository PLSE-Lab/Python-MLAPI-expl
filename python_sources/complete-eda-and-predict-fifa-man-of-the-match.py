#!/usr/bin/env python
# coding: utf-8

# ## Loading Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# load data
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


data.head()


# In[ ]:


numerical_features   = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns


# In[ ]:


numerical_features


# In[ ]:


categorical_features


# ## Univariate Analysis

# In[ ]:



data.describe()


# In[ ]:


# pots a histogram reations between numerical data
data.hist(figsize=(30,30))
plt.plot()


# Scatter plot is a great tool to see correlation degree and direction among features. Using seaborn pairplot makes this task easy for us by plotting all possible combinations.
# 

# In[ ]:


var1 = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed']
var1.append('Man of the Match')
sns.pairplot(data[var1], hue = 'Man of the Match', palette="husl")
plt.show()


# In[ ]:


sns.countplot(x='Man of the Match',data = data)


# In[ ]:


# Plotting total goal attempts by teams
attempts=data.groupby('Team')['Attempts'].sum().reset_index().sort_values(by=('Attempts'),ascending=False)

plt.figure(figsize = (15, 12), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Attempts", data=attempts)

plot1.set_xticklabels(attempts['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total attempts')
plot1.set_title('Total goal attempts by teams')


# In[ ]:


# Plotting total goals by teams
goals_by_team=data.groupby('Team')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'),ascending=False)

plt.figure(figsize = (15,12), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Goal Scored", data=goals_by_team)

plot1.set_xticklabels(goals_by_team['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total goals scored')
plot1.set_title('Total goals scored by teams')


# In[ ]:


# Plotting mean ball possession for teams

ball_possession=data.groupby('Team')['Ball Possession %'].mean().reset_index().sort_values(by=('Ball Possession %'),ascending=False)
ball_possession 

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Ball Possession %", data=ball_possession)

plot1.set_xticklabels(ball_possession['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Ball possession')
plot1.set_title('Mean ball possession')


# In[ ]:


# Plotting total Man of the Match awards for teams

# Encoding the values for the column man of the Match
mom_1={'Man of the Match':{'Yes':1,'No':0}}
data.replace(mom_1,inplace=True)

# Converting column datatype to int
data['Man of the Match']=data['Man of the Match'].astype(int)

mom=data.groupby('Team')['Man of the Match'].sum().reset_index().sort_values(by=('Man of the Match'),ascending=False)

plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Man of the Match", data=mom)

plot1.set_xticklabels(mom['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Man of the Matches')
plot1.set_title('Most Man of the Match awards')


# In[ ]:


# Plot of Total On-target and Off-target and blocked attempts by teams

group_attempt = data.groupby('Team')['On-Target','Off-Target','Blocked'].sum().reset_index()

# Changing the dataframe for plotting
group_attempt_sorted = group_attempt.melt('Team', var_name='Target', value_name='Value')

# Plotting the new dataset created above
plt.figure(figsize = (16, 10), facecolor = None)

sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=group_attempt_sorted)

plot1.set_xticklabels(group_attempt_sorted['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Attempts')
plot1.set_title('Total On-Target, Off-Target and Blocked attempts by teams')


# In[ ]:


# Plotting Most saves by teams

saves=data.groupby('Team')['Saves'].sum().reset_index().sort_values(by=('Saves'),ascending=False)

plt.figure(figsize = (15,12), facecolor = None)
sns.set_style("darkgrid")
plot1 = sns.barplot(x="Team", y="Saves", data=saves)

plot1.set_xticklabels(saves['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total Saves')
plot1.set_title('Most Saves')


# In[ ]:


# Plot of total corners, free kicks and offsides for teams

corners_offsides_freekicks = data.groupby('Team')['Corners','Offsides','Free Kicks'].sum().reset_index()
corners_offsides_freekicks

# Changing the dataframe for plotting
corners_offsides_freekicks_sort = corners_offsides_freekicks.melt('Team', var_name='Target', value_name='Value')

# Plotting the new dataset created above
plt.figure(figsize = (16, 10), facecolor = None)

# style
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Value", hue="Target", data=corners_offsides_freekicks_sort)

#labeling
plot1.set_xticklabels(corners_offsides_freekicks_sort['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Totals')
plot1.set_title('Total Corners, free kicks and offsides for teams')


# In[ ]:


# Plot of total goals conceded by teams

# Most goals conceded by teams
goals_conceded = data.groupby('Opponent')['Goal Scored'].sum().reset_index().sort_values(by=('Goal Scored'), ascending=False)

plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Opponent", y="Goal Scored", data=goals_conceded)

plot1.set_xticklabels(goals_conceded['Opponent'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total goals conceded')
plot1.set_title('Total goals conceded')


# In[ ]:


# Plot of Most Yellow Cards conceded by teams

# Most Yellow Cards by teams
yellow_cards = data.groupby('Team')['Yellow Card'].sum().reset_index().sort_values(by=('Yellow Card'), ascending=False)

plt.figure(figsize = (16, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="Team", y="Yellow Card", data=yellow_cards)

plot1.set_xticklabels(yellow_cards['Team'], rotation=90, ha="center")
plot1.set(xlabel='Teams',ylabel='Total yellow cards')
plot1.set_title('Total yellow cards')


# In[ ]:


# Sewness of numerical data
skew_values = skew(data[numerical_features], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)


# For normally distributed data, the skewness should be about 0.
# 
# For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. 
# 
# The function skewtest can be used to determine if the skewness value is close enough to 0, statistically speaking.
# 
# Although data is not normally distribute, there are positive as well have negative skewedness
# 
# 'Yello & Red', 'Red' and 'Goals in PSO' are highly positively skewed.
# ### Missing values

# In[ ]:



missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


# In[ ]:


data.isnull().sum()


# ## Bivariate analysis
# - Understanding how statistics of one feature is impacted in presence of other features
# - Commonly used tools are:
#     - Pearson Correlation Coefficient (or) scatter plots
#     - Pairplots

# ### Correlation Coefficient
# It is a measure of the strength and direction of the linear relationship between two variables that is defined as the covariance of the variables divided by the product of their standard deviations.
# 
# It is of two type: Positive correlation and Negative correlation
# 
# positive correlation if the values of two variables changing with same direction
# 
# negative correlation when the values ofvariables change with opposite direction
# 
# r values always lie between -1 to + 1
# 
# Interpretation:
#  Exactly -1. A perfect downhill (negative) linear relationship
# 
#  Exactly +1. A perfect uphill (positive) linear relationship
# 

# In[ ]:


plt.figure(figsize=(30,30))
sns.heatmap(data[numerical_features].corr(), square=True, annot=True,robust=True, yticklabels=1)


# #### So clearly Goal scored is highest correlated to target

# Correlated columns needs to be removed to avoid multicollinearity. Let's use multicollinearity check
# 
# These features have least or no correlation with 'Man of the Match'
# ['Blocked', 'OffSides', 'Saves','Distance Covered (Kms)', 'Yellow & Red', '1st Goal', 'Goals in PSO']
# 
# These features will not have impact on aur analysis and thus, holding them or retaining them is our choice
# 

# In[ ]:


# Correlation with highally correlated features
var = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
plt.figure(figsize=(15,10))
sns.heatmap((data[var].corr()), annot=True)


# ### Outliers detection and removal

# In[ ]:


dummy_data = data[var1]
plt.figure(figsize=(20,10))
sns.boxplot(data = dummy_data)
plt.show()


# As per boxplot there are :
# -1 outlier in Goal scored
# 
# -2 in On-Target
# 
# -1 in corners
# 
# -2 in Attempts
# 
# -3 in Yellow Card
# 
# -1 in Red
# 
# #### What are Outliers
# In statistics, an outlier is an observation point that is distant from other observations. An outlier may be due to variability in the measurement or it may indicate experimental error; the latter are sometimes excluded from the data set.
#  
# Pragmatic approach: plot scatter visualisation or boxplot and identify abnormally distant points
# 
# The quantity of outliers present in this problem is not too huge and will not have gravity impact if left untreated. They are only few and within range.

# ### Missing value treatment
# 
# features -- ['Own goal Time', 'Own goals', '1st Goal']  have very high percentage of missing data
# 
# so it is better to drop them

# In[ ]:



missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


# In[ ]:


data.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)


# Categorical features encoding
# - As machine laearning models understand only numbers data in different formats including text and dates needs to be mapped into numbers prior to feeding to the model
# - The process of changing non-numerical data into numerical is called 'Encoding'
# - Before encoding let's understand how many categories or levels are present in each categorical features

# In[ ]:


categorical_features


# In[ ]:


# Function for finding no of unique elements in each features
def uniqueCategories(x):
    columns = list(x.columns).copy()
    for col in columns:
        print('Feature {} has {} unique values: {}'.format(col, len(x[col].unique()), x[col].unique()))
        print('\n')
uniqueCategories(data[categorical_features].drop('Date', axis = 1))


# Categorical -['Date', 'Team', 'Opponent','Round', 'PSO']
# 
# Nominal - Team, Opponent
# 
# Ordinal - Round
# 
# Interval - Date, PSO is binary
# 
# 
# I believe 'Round' should also not have any impact on 'Man of the Match' because, a player performance should be consistent over all matches to become man of the match than just in a particular round. Thus, let's give equal weitage to each round.
# PSO is binary
# 
# I am not going to include 'Match date' as it should definately not impact a player formance.
# 
# 

# In[ ]:


data.drop('Date', axis = 1, inplace=True)


# Dropping "Corners', 'Fouls Committed' and 'On-Targets' will remove high correlated elements and remove chances of multi-collinearity. these features are selected based on their low collinearity with 'Man of the Match' and high collinearity with other features.
# 

# In[ ]:


data.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)
print(data.shape)
data.head()


# In[ ]:


cleaned_data  = pd.get_dummies(data)


# In[ ]:


print(cleaned_data.shape)
cleaned_data.head()


# 
# ## Now we can Appy different machine learning algorithms to predict the Man of the Match
# The data has been cleaned and is ready for further steps in data pipeling
#     - Pre-processing
#     - Modeling
#     - Evaluation
#     - Prediction
# 

# ### Base line model

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, classification_report, confusion_matrix, precision_recall_curve


# In[ ]:


df = cleaned_data.copy()
df.describe()


# In[ ]:


df = df.apply(LabelEncoder().fit_transform)
df.head()


# In[ ]:


targetfet = df['Man of the Match']

features = df.drop(['Man of the Match'], axis = 1)
targetfet.shape


# In[ ]:


features.shape


# In[ ]:


####Prediction model########
#Train-Test split
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(features, targetfet, test_size = 0.2, random_state = 42)
label_train.shape


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(data_train, label_train)
logis_score_train = logis.score(data_train, label_train)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(data_test, label_test)
print("Testing score: ",logis_score_test)


# In[ ]:


#decision tree
from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier()
dt.fit(data_train, label_train)
dt_score_train = dt.score(data_train, label_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(data_test, label_test)
print("Testing score: ",dt_score_test)


# In[ ]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(data_train, label_train)
rfc_score_train = rfc.score(data_train, label_train)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(data_test, label_test)
print("Testing score: ",rfc_score_test)


# In[ ]:


#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],
        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],
        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:





# ## Second Modal

# In[ ]:


train = cleaned_data.copy()
train.head()


# In[ ]:


# Specify the label (just in case we want to predict something else)
label_name = 'Man of the Match'

# Categorical features are the non numeric ones
categoricals = train.columns[train.dtypes == 'object'].tolist()

# Label encode them otherwise LightGBM can't use them
for cat_feat in categoricals:
    encoder = LabelEncoder()
    train[cat_feat] = encoder.fit_transform(df[cat_feat])
label = train.pop(label_name)

# Don't specify the label as a categorical
if label_name in categoricals:
    categoricals.remove(label_name)


# Cross validate and predict
# Given the relatively small dataset size, the most robust measure of prediction accuracy will be to use SKLearn's cross_val_predict
# 
# 

# In[ ]:


import lightgbm as lgbm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, classification_report, confusion_matrix, precision_recall_curve


clf = lgbm.LGBMClassifier(
    boosting_type='gbdt',
)
y_prob = cross_val_predict(
    estimator=clf, 
    cv=5, 
    X=train, 
    y=label,
    fit_params={'categorical_feature': categoricals},
    method='predict_proba'
)
y_pred = np.argmax(y_prob, axis=1)


# In[ ]:


y_pred


# ### Check performance
# Using classification_report, we can then quickly see how well the LightGBM classifier has performed:
# 

# In[ ]:


print(classification_report(y_true=label, y_pred=y_pred))


# ## Have any doubts ???
# 
# 

# In[ ]:




