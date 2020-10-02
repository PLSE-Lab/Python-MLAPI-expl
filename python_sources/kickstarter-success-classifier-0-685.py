#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('html', '', '\n<img src="https://cdn-images-1.medium.com/max/2000/1*LwDYkjk7XqPYxb1TUozE-g.png" width="800"/>\nphoto provided by: <a href="https://medium.com/swlh/how-to-succeed-at-kickstarter-6e72d7120cb5">https://medium.com/swlh/how-to-succeed-at-kickstarter-6e72d7120cb5</a>')


# In[ ]:


# import warnings
# warnings.filterwarnings("ignore")


# # Part 1
# ## Exploratory Data Analysis

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../input/ks-projects-201612.csv", encoding='latin1', low_memory=False )

df.head()


# In[ ]:


print(df.shape[0], 'rows and', df.shape[1], 'columns')


# In[ ]:


# get percentage of nulls
df.isnull().sum()/df.shape[0]


# ## Over 99% of the unnamed columns are null, and 1% of usd pledged is null
#     name and category is null for several values only

# In[ ]:


# Strip white space in column names
df.columns = [x.strip() for x in df.columns.tolist()]


# In[ ]:


df[(df['name'].isnull()) | (df['category'].isnull())]


# In[ ]:


# drop all nulls remaining
df = df.dropna(axis=0, subset=['name', 'category'])


# In[ ]:


# drop unnamed columns
df = df.iloc[:,:-4]


# In[ ]:


print(len(df.main_category.unique()), "Main categories")


# In[ ]:


print(len(df.category.unique()), "sub categories")


# In[ ]:


sns.set_style('darkgrid')
mains = df.main_category.value_counts().head(15)

x = mains.values
y = mains.index

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111)
ax = sns.barplot(y=y, x=x, orient='h', palette="cool", alpha=0.8)

plt.title('Kickstarter Top 15 Category Count')
plt.show()


# In[ ]:


cats = df.category.value_counts().head(15)

x = cats.values
y = cats.index

fig = plt.figure(dpi=100)
ax = fig.add_subplot(111)
ax = sns.barplot(y=y, x=x, orient='h', palette="winter", alpha=0.8)

plt.title('Kickstarter Top 15 Sub-Category Count')
plt.show()


# ## Many of the top sub categories overlap with the top main categories such as Technology, Music, Film and Design

# In[ ]:


df.columns = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline',
       'goal', 'launched', 'pledged', 'state', 'backers', 'country',
       'usd_pledged']


# In[ ]:


# Convert string to float
df.loc[:,'usd_pledged'] = pd.to_numeric(df['usd_pledged'], downcast='float', errors='coerce')


# In[ ]:


df.isna().sum()


# In[ ]:


# fill null pledged amounts with 0
df = df.fillna(value=0)


# In[ ]:


df.loc[:,'usd_pledged'].describe()


# ## Look at the distribution of the goal.

# In[ ]:


# convert goal to float
df['goal'] = pd.to_numeric(df.goal, downcast='float', errors='coerce')


# In[ ]:


df.goal.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


# Fill null goals with zero
df = df.fillna(value=0)


# In[ ]:


# Select only projects with goals greater than 0
df = df[df.goal > 0]


# In[ ]:


fig, ax = plt.subplots(1, 1)

g = sns.distplot(np.log10(df.goal), kde=False, bins=30)

plt.xlabel('Log Goal')
plt.title('Distribution of Goal')
plt.show()


# In[ ]:


plt.style.use('seaborn-pastel')

fig, ax = plt.subplots(1, 1, dpi=100)
explode = [0,0,.1,.2, .4]
df.state.value_counts().head(5).plot.pie(autopct='%0.2f%%',
                                        explode=explode)

plt.title('Breakdown of Kickstarter Project Status')
plt.ylabel('')
plt.show()


# # Which countries have the highest number of kickstarters?

# In[ ]:


# Look at null country values
df[~df.country.str.contains('^[A-Z]{2}$', case=False)].country.value_counts()


# ## All null country values contain ```N, "0```

# In[ ]:


# Replace null countries with None
replace = df[~df.country.str.contains('^[A-Z]{2}$', case=False)].country.unique().tolist()
df.loc[:,'country'] = df.country.replace(to_replace=replace, value='None')


# In[ ]:


df.country.value_counts()


# # The US leads with number of kickstarters
#     followed by Great Britain, Canada, and Australia.

# ## Which currencies fund the most Kickstarters?

# In[ ]:


df.currency.value_counts()


# ## Top Currencies are USD, GB(Pounds), Canadanian Dollars, Euros, and Australian Dollars
#     The English speaking world seems to dominate Kickstarter

# In[ ]:


# Convert Backers to integer
df.loc[:,'backers'] = pd.to_numeric(df.backers, errors='raise', downcast='integer')


# In[ ]:


fig, ax = plt.subplots(1, 1)
(df.backers >=1).value_counts().plot.pie(autopct='%0.0f%%', 
                                         explode=[0,.1], 
                                         labels=None, 
                                         shadow=True,
                                         colors=['#a8fffa', '#ffbca8'])

plt.ylabel('')
plt.title('Kickstarter Backer Share')
plt.legend(['backers', 'no backers'], loc=2)

plt.show()


# # Projects with zero backers make up 15% of the population.
#     However, since they may skew the model, only projects with at least 1 backer will be used.

# In[ ]:


# create a dataframe with projects that have 1 or more backers
df = df[(df.backers >= 1)]


# In[ ]:


sns.set_style('darkgrid')
sns.distplot(np.log(df.backers), color='purple', kde=False, bins=10)

plt.title('Backer Distribution')
plt.xlabel('Log backers')
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 1)
(df.usd_pledged > 0).value_counts().plot.pie(autopct='%0.0f%%', 
                                             explode=[0,.6], 
                                             labels=None, 
                                             shadow=True, 
                                             colors=['#b3ff68', '#ff68b4'])

plt.ylabel('')
plt.title('Kickstarter Pledged Share')
plt.legend(['pledges', 'no pledges'], loc=3)

plt.show()


# In[ ]:


# Select only US pledges greater than zero
df = df[df.usd_pledged > 0]


# In[ ]:


sns.distplot(np.log10(df.usd_pledged), color='g', kde=False, bins=8)

plt.title('Distribution of USD Pledged')
plt.xlabel('Log USD Pledged')
plt.show()


# ## Which Days and months had the most launches and deadlines?

# In[ ]:


# Convert launched, and deadline to datetime objects
for col in ['launched', 'deadline']:
    df.loc[:,col] = pd.to_datetime(df[col], errors='coerce')
    
# drop projects with null launch and deadline dates
df = df.dropna()


# In[ ]:


print(
    'first launch:', 
      df.launched.min().strftime('%B %d, %Y'), 
      '\nlast launch:',
      df.launched.max().strftime('%B %d, %Y')
     )


# In[ ]:


# plt.style.use('fivethirtyeight')

cats = df.launched.dt.strftime('%B %d, %Y').value_counts().head(10)

x = cats.values
y = cats.index

fig = plt.figure(figsize=(6,6))
sns.barplot(y=y, x=x, orient='h', palette="summer_r", alpha=0.8)
# df.launched.dt.strftime('%B %d, %Y').value_counts().head(10).plot.barh()

plt.title('Top 10 Kickstarter Launch Dates')
plt.show()


# # Kickstarter's Peak dates were all in the same 2 weeks in mid-july 2014.
#     With several outliers in 2015 and 2016

# In[ ]:


fig = plt.figure(figsize=(15,7), dpi=100)

# plt.style.use('fivethirtyeight')
# fig.suptitle('Normalized Launch Distributions', fontsize=14)

plt.subplot(231)
plt.title('day of month')
df.launched.dt.day.value_counts().hist(density=True, color='g')

plt.subplot(232)
plt.title('month')
df.launched.dt.month.value_counts().hist(density=True)

plt.subplot(233)
plt.title('day of week')
df.launched.dt.dayofweek.value_counts().hist(density=True)

plt.subplot(234)
plt.title('day of year')
df.launched.dt.dayofyear.value_counts().hist(density=True, color='g')

plt.subplot(235)
plt.title('Year')
df.launched.dt.year.value_counts().hist()

plt.subplot(236)
plt.title('Week')
df.launched.dt.week.value_counts().hist(density=True, color='g')

plt.tight_layout()
plt.show()


# In[ ]:


df.launched.dt.year.value_counts()


# ## Distribution of launch day of week, month and day of year appear normal

# In[ ]:


fig = plt.figure(figsize=(12,7), dpi=100)

# plt.style.use('fivethirtyeight')
# fig.suptitle('Normalized Deadline Distributions', fontsize=14)

plt.subplot(231)
plt.title('day of month')
df.deadline.dt.day.value_counts().hist(density=True, color='purple')

plt.subplot(232)
plt.title('month')
df.deadline.dt.month.value_counts().hist(density=True, color='purple')

plt.subplot(233)
plt.title('day of week')
df.deadline.dt.dayofweek.value_counts().hist(density=True, color='purple')

plt.subplot(234)
plt.title('day of year')
df.deadline.dt.dayofyear.value_counts().hist(density=True)

plt.subplot(235)
plt.title('Year')
df.deadline.dt.year.value_counts().hist(density=True, color='purple')

plt.subplot(236)
plt.title('Week')
df.deadline.dt.week.value_counts().hist(density=True)

plt.tight_layout()
plt.show()


# ## Distribution of deadline day of year, and week appears normal

# In[ ]:


(df.deadline - df.launched).dt.days.value_counts().head(10)


# In[ ]:


(df.deadline - df.launched).dt.days.describe()


# In[ ]:


sns.set_style('darkgrid')

sns.distplot(((df.deadline - df.launched).dt.days), kde=False, bins=5)

plt.title('Normalized Duration Distribution')
plt.show()


# In[ ]:


((df.deadline - df.launched).dt.days).quantile(q=.90)


# In[ ]:


(((df.deadline - df.launched).dt.days) <= 59).value_counts()


# In[ ]:


fig, ax = plt.subplots(1, 1)
(((df.deadline - df.launched).dt.days) <= 59).value_counts().plot.pie(autopct='%0.0f%%', 
                                             explode=[0,.6], 
                                             labels=None, 
                                             shadow=True, 
                                             colors=['#f7cc7b', '#a07bf7'])

plt.ylabel('')
plt.title('Kickstarter Duration Share')
plt.legend(['less than 60 days', 'more than 59 days'], loc=3)

plt.show()


# ## 91% of Kickstarters lasted less than 60 days

# In[ ]:


sns.distplot(np.log(df['usd_pledged']/(df['backers'])), kde=False, bins=20)

plt.title('Normalized Distribution of USD Pledged per Backer')
plt.xlabel('Log USD per Backer')
plt.show()


# In[ ]:


sns.distplot(np.log(df['goal']/(df['backers'])), kde=False, bins=15)
plt.title('Normalized Distribution of Goal per Backer')
plt.xlabel('Log Goal per Backer')
plt.show()


# In[ ]:


df['log_usd_per_backer'] = np.log(df['usd_pledged']/df['backers'])
df['log_goal_per_backer'] = np.log(df['goal']/df['backers'])


# In[ ]:


get_ipython().run_cell_magic('html', '', '\n<img src="https://drive.google.com/uc?export=download&id=1Q1u0dJ-SDBwMRgaLSYFPBAGmwcVrOXZg"/>')


# In[ ]:


import matplotlib as mpl
# Reset matplotlib params
mpl.rcParams.update(mpl.rcParamsDefault)

sns.lmplot(x="log_goal_per_backer", y="log_usd_per_backer",
                hue="state",
                palette='Spectral',
                hue_order=['failed', 'canceled', 'suspended', 'live', 'successful'],
                data=df)

plt.title("Log Goal per Backer vs. Log Pledged per Backer")
plt.show()


# In[ ]:


# # Filtering only for successful and failed projects
# df = df[(df['state'] == 'failed') | (df['state'] == 'successful')]


# In[ ]:


# drop live kickstarters as their outcome is undetermined
df = df[(df.state != 'live')]


# ## The scatter plot above shows a linear relationship between log usd per backer and log goal per backer.
#     There is a clear dividing line for the successful vs unsuccessful status.

# # Section 2:
# ## Modeling

# # Feature Engineering
# ```*******************************```

# In[ ]:


features = df.copy()
features['success'] = np.where(features.state == 'successful', 1, 0)


# In[ ]:


features['US'] = np.where(features.country=='US', 1,0)


# In[ ]:


# Replace punctuation and count number of words in name
# features['length_name'] = [len(x) for x in features.name.str.replace('[^\w\s]','').str.split()]


# In[ ]:


features['length_chars'] = features.name.str.len()


# In[ ]:


features['contains_!'] = pd.get_dummies(features.name.str.contains('!'), drop_first=True)


# In[ ]:


features['contains_?'] = pd.get_dummies(features.name.str.contains(r'\?'), drop_first=True)


# In[ ]:


features['contains_title'] = pd.get_dummies(features.name.str.istitle(), drop_first=True)


# In[ ]:


features['log_goal'] = np.log10(features.goal)
features['log_usd_pledged'] = np.log10(features.usd_pledged)


# In[ ]:


features['time_delta'] = (features.deadline.dt.date - features.launched.dt.date).dt.days


# In[ ]:


# Select only log goal greater than or equal to 1
# features = features[(features.log_goal >= 1)]


# In[ ]:


sns.set_style('darkgrid')

sns.distplot(features.log_goal, kde=False, bins=20)

plt.title('Distribution of Goal')
plt.plot()


# In[ ]:


# conditions = [(features.time_delta<15), 
#               (features.time_delta>=15) & (features.time_delta<30),
#               (features.time_delta>=30)&(features.time_delta<45),
#               (features.time_delta>=45)&(features.time_delta<60),
#               (features.time_delta>=60)
#              ]
# choices = ['15_days', '30_days', '45_days', '60_days', '90_days']
# features[('duration')] = np.select(conditions, choices, default='0_days')
# features[('duration')] = pd.get_dummies(features[('duration')])


# In[ ]:


features = pd.concat([features, 
                      pd.get_dummies(features.launched.dt.dayofweek, prefix='day_of_week'),
                     pd.get_dummies(features.launched.dt.week, prefix='week'),
                     pd.get_dummies(features.launched.dt.year, prefix='year'),
                     pd.get_dummies(features.category)],
         axis=1)


# In[ ]:


features = features.iloc[:,15:]


# In[ ]:


sns.heatmap(features.iloc[:, :20].corr(), cmap='Blues')

plt.title('Heatmap of Kickstarter Feature Correlations')
plt.show()


# # There is a strong correlation between log_usd_pledged and success
# - Weak negative correlation betwen contains title and length_chars.
# - Weak negative correlation between log_goal and success.
# - Weak positive correlation log goal, time_delta, and usd_pledged.

# # Try Linear Regression model to predict Log USD Pledged

# In[ ]:


from sklearn import linear_model

# Instantiate and fit our model.
regression = linear_model.LinearRegression()
Y = features[('log_usd_pledged')]
X = features.drop(['log_usd_pledged', 'success'], axis=1)
regression.fit(X, Y)

# Inspect the results.
# print('\nCoefficients: \n', regression.coef_)
print('\nIntercept: \n', regression.intercept_)
print('\nR-squared:')
print(regression.score(X, Y))


# In[ ]:


sns.regplot(x='time_delta', y='log_goal', data=features)
plt.title('Log Goal vs. Duration')
plt.show()


# ## There is a weak linear relationship between duration and log goal
#     This is no surprise, because higher goals should tend to take longer to meet.

# In[ ]:


predicted = regression.predict(X).ravel()
actual = features[('log_usd_pledged')]

# Calculate the error, also called the residual.
residual = actual - predicted

sns.distplot(residual)
plt.title('Residual Counts')
plt.show()


# In[ ]:


sns.regplot(x=predicted, y=residual, fit_reg=False)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()


# ## The Regression Model is suffering from non-homoscedasticity.
#     As the amount predicted increases the residual error increases.

# In[ ]:


# Drop Log USD Pledged
features = features.drop(['log_usd_pledged'], 1)


# # Section 3: Classification

# In[ ]:


X = features.drop(['success'], 1)
y = features.success

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


# Declare a logistic regression classifier.
# Parameter regularization coefficient C described above.
lr = LogisticRegression(penalty='l2', solver='liblinear')

# Fit the model.
fit = lr.fit(X_train, y_train)

# Display.
# print('Coefficients')
# print(fit.coef_)
# print(fit.intercept_)

pred_y_sklearn = lr.predict(X_test)

print('\n Accuracy by success')
print(pd.crosstab(pred_y_sklearn, y_test))

print('\n Percentage accuracy')
print(lr.score(X_test, y_test))

# CV
# scores = cross_val_score(lr, X, y, cv=10)

# print(scores)


# # Baseline Accuracy with Logistic Regression is 67.3%

# In[ ]:


# # Pass logistic regression model to the RFE constructor
# from sklearn.feature_selection import RFE

# selector = RFE(lr)
# selector = selector.fit(X, y)

# print(selector.ranking_)

# # Now turn into a dataframe so you can sort by rank

# rankings = pd.DataFrame({'Features': X.columns, 'Ranking' : selector.ranking_})
# rankings.sort_values('Ranking')


# ## Predicting non-successful Kickstarters is more accurate than predicting successful kickstarters
#     and success is the only feature correlated with any other feature.
#     There is also a slight class imbalance between failed and successful

# In[ ]:


X_pca = features.drop('success', 1)
sklearn_pca = PCA(n_components=5)
Y_sklearn = sklearn_pca.fit_transform(X_pca)


# In[ ]:


print(
    'The percentage of total variance in the dataset explained by each',
    'component from Sklearn PCA.\n',
    sklearn_pca.explained_variance_ratio_)


# # Try random forest

# In[ ]:


# Use 500 Estimators
rfc = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=500)

rfc.fit(X_train, y_train)

print('score:', rfc.score(X_test, y_test))


# In[ ]:


# # CV
# scores = cross_val_score(rfc, X, y, cv=10)

# print(scores)
# print()
# print('Average:', np.mean(scores))


# ## Random Forest performs variably between 64% and 68%

# In[ ]:


# # Pass Random Forest to the RFE constructor
# from sklearn.feature_selection import RFE

# selector = RFE(rfc)
# selector = selector.fit(X, y)


# In[ ]:


# print(selector.ranking_)


# In[ ]:


# # Now turn into a dataframe so you can sort by rank

# rankings = pd.DataFrame({'Features': X.columns, 'Ranking' : selector.ranking_})
# rankings.sort_values('Ranking')


# ## Try Gradient Boosting

# In[ ]:


def gradient_boost(estimators, depth, loss_function, sampling):
    clf = ensemble.GradientBoostingClassifier(n_estimators=estimators, 
                                              max_depth=depth, 
                                              loss=loss_function, 
                                              subsample=sampling
                                              )
    clf.fit(X_train, y_train)
    print('\n Percentage accuracy for Gradient Boosting Classifier')
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)

# Accuracy tables.
    table_train = pd.crosstab(y_train, predict_train, margins=True)
    table_test = pd.crosstab(y_test, predict_test, margins=True)

    train_tI_errors = table_train.loc[0.0,1.0] / table_train.loc['All','All']
    train_tII_errors = table_train.loc[1.0,0.0] / table_train.loc['All','All']

    test_tI_errors = table_test.loc[0.0,1.0]/table_test.loc['All','All']
    test_tII_errors = table_test.loc[1.0,0.0]/table_test.loc['All','All']
    
    train_accuracy = 1 - (train_tI_errors + train_tII_errors)
    test_accuracy = 1 - (test_tI_errors + test_tII_errors)
    
    print((
    'Training set accuracy:\n'
    'Overall Accuracy: {}\n'
    'Percent Type I errors: {}\n'
    'Percent Type II errors: {}\n\n'
    'Test set accuracy:\n'
    'Overall Accuracy: {}\n'
    'Percent Type I errors: {}\n'
    'Percent Type II errors: {}'
    ).format(train_accuracy, train_tI_errors, train_tII_errors, test_accuracy, test_tI_errors, test_tII_errors))


# In[ ]:


#500 estimators, max depth of 2, loss function = 'deviance', subsampling default to 1.0
gradient_boost(500, 2, 'deviance', 1.0)


# # Gradient Boosting gets 68.0% test set accuracy

# In[ ]:


clf = ensemble.GradientBoostingClassifier(n_estimators=500, max_depth=2, loss='deviance', subsample=1.0)
clf.fit(X_train, y_train)

feature_importance = clf.feature_importances_[:40]

# Make importances relative to max importance.
plt.figure(figsize=(5,10))
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:


import lightgbm as lgb
from lightgbm import LGBMClassifier

clf_lgbm = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
    )

clf_lgbm.fit(X_train, 
        y_train,
        eval_set= [(X_train, y_train), (X_test, y_test)], 
        eval_metric='auc', 
        verbose=0, 
        early_stopping_rounds=30
       )

acc_clf_lgbm = round(clf_lgbm.score(X_test, y_test) * 100, 2)
acc_clf_lgbm

# # Run Cross validation
# scores = cross_val_score(clf_lgbm, X, y, cv=5)
# np.mean(scores)


# In[ ]:


# CV
# scores = cross_val_score(clf_lgbm, X, y, cv=10)

# print(scores)
# print()
# print('Average:', np.mean(scores))


# 
# ## Best Run is 68.5% test set accuracy with Lgb Classifier
# ```********************************************************************```

# In[ ]:


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_train, y_train)

acc_bdt = round(bdt.score(X_test, y_test) * 100, 2)
acc_bdt

# Run Cross validation
# scores = cross_val_score(bdt, X, y, cv=5)
# np.mean(scores)


# ## Ada boost gets 65.9% test set accuracy

# In[ ]:


# for min_max scaling
from mlxtend.preprocessing import minmax_scaling
# features['length_name_scaled'] = minmax_scaling(features.length_name, columns = 0)
features['length_chars_scaled'] = minmax_scaling(features.length_chars, columns = 0)
features['time_delta_scaled'] = minmax_scaling(features.time_delta, columns = 0)

# Set X, and y for models and training and test sets for Cross Validation
y = features['success']
X = features.drop(['success', 'length_chars'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
neigh.fit(X_train, y_train)


# ## KNN performs the worst at 61% test set accuracy

# In[ ]:


print(neigh.score(X_test, y_test))

# cross_val_score(neigh, X, y, cv=5)


# # Future work:
# - Feature engineering
# - Hyperparameter Sweeps
# - Create better Classifiers
# 
# ## Run Model on Indiegogo, gofundme, and other crowd source data.

# In[ ]:





# In[ ]:




