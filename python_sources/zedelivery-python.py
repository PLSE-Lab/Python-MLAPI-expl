#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# ---

# ![](http://)In this notebook, I will be using the [Python programming language](https://www.python.org/) to solve the problems provided in this competition. I've also worked in a solution using [R programming language](https://www.r-project.org/), which you can access [through this link](https://www.kaggle.com/cmcoutosilva/zedelivery-r), although it's incomplete.
# 
# This notebook is organized in topics for facilitating understanding. It's worth to note that the "feature engineering" doesn't have a particular topic because I made it across different topics.
# 
# You can easily go right to the topic you are interested in by clicking in the links in summary below:

# # SUMMARY
# 
# * [Settings & libraries](#ch0)
# * 1. [Getting data](#ch1)
# * 2. [Exploring data](#ch2)  
#     \- [Variable's correlation](#ch2_1)  
#     \- [Visualize numerical data](#ch2_2)  
#     \- [Visualize categorical data](#ch2_3)
# * 3. [Dealing with missing data](#ch3)
# * 4. [Prediction](#ch4)
# * 5. [Submission](#ch5)
# 

# <a id="ch0"></a>
# # Settings & libraries
# 
# First, let's load some libraries to analyze the data.

# In[ ]:


# For data manipulation
import pandas as pd
import numpy as np

# For data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
get_ipython().run_line_magic('matplotlib', 'inline')

# For nice displaying output in this notebook
from IPython.display import display, HTML

# For data modeling & prediction
from scipy import stats
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


# And define a function for displaying nicely multiple dataframes.

# In[ ]:


# code snippet from https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
from IPython.core.display import display, HTML

def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


# <a id="ch1"></a>
# # 1. Getting data
# 
# Data is provided as [CSV files](https://en.wikipedia.org/wiki/Comma-separated_values), so we can read them into a dictionary using [Pandas](https://pandas.pydata.org/), parsing the date column as a date type.

# In[ ]:


filepath = '../input/walmart-recruiting-store-sales-forecasting/'
datasets = dict(
    stores = pd.read_csv(f'{filepath}stores.csv'),
    train_data = pd.read_csv(f'{filepath}train.csv.zip', parse_dates=['Date'], compression='zip'),
    test_data = pd.read_csv(f'{filepath}test.csv.zip', parse_dates=['Date'], compression='zip'),
    features = pd.read_csv(f'{filepath}features.csv.zip', parse_dates=['Date'], compression='zip')
)


# We can see they've been successfully loaded by using the head() function by printing the first lines: 

# In[ ]:


display_side_by_side([datasets[k].head() for k in datasets], [k for k in datasets])


# Before going further on exploring the data, let's get them together (train_data, test_data, stores, and features).

# In[ ]:


# Set label for identifying dataset
datasets['train_data']['Set'] = 'Train'
datasets['test_data']['Set'] = 'Test'

# Set missing values for Weekly_sales in test_data (we'll predict these values)
datasets['test_data']['Weekly_Sales'] = np.nan

# Concatenate train & test datasets into a single dataset
full = pd.concat([datasets['train_data'], datasets['test_data']], sort=False)

# Merge train/test dataset with stores & features
full = datasets['stores'].merge(full)
full = full.merge(datasets['features'])

# Order by set
full = (full.sort_values(['Set', 'Date', 'Store', 'Dept'], ascending=[False, True, True, True])
        .reset_index(drop=True))

# Display first lines and dimensions
display(full.head())
print('{dim[0]} rows and {dim[1]} columns.'.format(dim = full.shape))


# Now, let's split the date into new columns, and create another column to label the holidays.  
# It could help us to get better insight from the data.

# In[ ]:


# Split into new columns after the 'Date' column
full.insert(full.columns.tolist().index('Date')+1, 'Week', full.Date.dt.week)
full.insert(full.columns.tolist().index('Date')+2, 'Month', full.Date.dt.month)
full.insert(full.columns.tolist().index('Date')+3, 'Year', full.Date.dt.year)


# In[ ]:


# Copy the holidays specs from the competition data description
holidays = {
    'Super Bowl': pd.to_datetime(['12-Feb-10', '11-Feb-11', '10-Feb-12', '8-Feb-13']),
    'Labor Day': pd.to_datetime(['10-Sep-10', '9-Sep-11', '7-Sep-12', '6-Sep-13']),
    'Thanksgiving': pd.to_datetime(['26-Nov-10', '25-Nov-11', '23-Nov-12', '29-Nov-13']),
    'Christmas': pd.to_datetime(['31-Dec-10', '30-Dec-11', '28-Dec-12', '27-Dec-13'])
}

# Make a new column for specifying the holiday (after the IsHoliday column)
full.insert(full.columns.tolist().index('IsHoliday')+1, 'Holiday', np.nan)
# Label the holidays in this column
for holiday in holidays:
    full.loc[full['Date'].isin(holidays[holiday]), 'Holiday'] = holiday


# Check some basic information of the dataset, like the data type for each column:

# In[ ]:


full.info()


# There are columns classified as integer or object when they could be classified as a category. Let's fix that:

# In[ ]:


category_cols = ['Store', 'Type', 'Dept', 'Week', 'Month', 'Year', 'Holiday', 'Set']
for col in category_cols:
    if col in ['Week', 'Month', 'Year']:
        full[col] = full[col].astype('category').cat.as_ordered()
    else:
        full[col] = full[col].astype('category')


# , then see the changes successfully applied:

# In[ ]:


full.info()


# Now we are ready for starting the exploratory analysis.

# <a id="ch2"></a>
# # 2. Exploring the data
# 
# Let's first take a look at the numeric variables.

# In[ ]:


full.describe()


# We can also look at grouped data by set (train and test dataset):

# In[ ]:


pd.set_option('display.max_rows', None) # set an option to show all rows 
display(full.groupby('Set').describe().T) # show all rows grouping by set
pd.reset_option('display.max_rows') # reset that option


# With this, we can take some important notes:
# - Both Weekly_Sales and Markdown columns have negative values. 
# - There are columns with distinct total counts, which means **missing values**.
# 
# First, let's update these negatives values to zero. We'll deal with the missing values later.

# In[ ]:


# Set negative values to 0
cols = full.filter(regex='^Weekly|^Mark').columns
full[cols] = full[cols].apply(lambda x: np.where(x<0, 0, x))


# Now let's see some statistics for DateTime and categorical variables.

# In[ ]:


full.describe(include=['datetime', 'category'])


# In[ ]:


full.groupby('Set').describe(include=['datetime', 'category']).T


# We can see that the most frequent store is the same for both the train and test dataset, which is good if it provides useful information for modeling. The dates are different for an obvious reason (they correspond to distinct time range).

# <a id="ch2_1"></a>
# ### Variable's correlation
# 
# We can split the 'Type' and 'Holiday' variables so that we can use them for showing for plotting, as well as in the models. By default, Pandas set the new dummy variables as numerical.

# In[ ]:


dummies = pd.get_dummies(full, columns=['Type', 'Holiday']).filter(regex="^[Type|Holiday].+_")
full = pd.concat([full, dummies], axis=1)


# Now we see how the variables relate to each other:

# In[ ]:


# Set categorial variables to numeric in order to appear in the heatmap
full[['Dept', 'Store', 'Week','Month', 'Year']] = full[['Dept', 'Store', 'Week', 'Month', 'Year']].astype('int')

# Plot Heatmap from variable's correlation
plt.figure(figsize=(16,10))
m = full.corr()
np.fill_diagonal(m.values, np.nan)
sns.heatmap(m, cmap='seismic', annot=True, fmt='.2f', annot_kws={"size": 9})


# It seems like the Store and their respective departments, type, and size, are valuable predictor to the weekly sales.

# <a id="ch2_2"></a>
# ### Visualizing numerical data
# 
# Distribution plots can provide a good general idea of how the data is structured.  
# Let's take a look into the data:

# In[ ]:


# Set categorial variables back as a category
full[['Store', 'Dept', 'Week', 'Month', 'Year']] = full[['Store', 'Dept', 'Week', 'Month', 'Year']].astype('category')
# Separate numerical variables for plotting
y_target = 'Weekly_Sales'
x_num_vars = full.select_dtypes(['float', 'int']).columns
x_num_vars = x_num_vars[x_num_vars!=y_target]
markdown_cols = full.filter(regex='Mark*').columns
x_num_vars = x_num_vars[~x_num_vars.isin(markdown_cols)]


# In[ ]:


sns.pairplot(full.query('Set=="Train"'), x_vars=x_num_vars, y_vars=y_target, hue='IsHoliday', height=4, aspect=.8)


# In[ ]:


sns.pairplot(full.query('Set=="Train"'), x_vars=markdown_cols, y_vars=y_target, hue='IsHoliday', height=3.2, aspect=.75 )


# It seems that for Markdown 1, 4, and 5, there are more weekly sales when it's no holiday.  
# We can further examine that with the plot below.

# In[ ]:


mkdn = full[full.Set=='Train'].melt('IsHoliday', [f'MarkDown{i}' for i in [1,4,5]])
f = sns.pointplot(data=mkdn, x='IsHoliday', y='value', hue='variable', estimator=np.mean)
f.set_ylabel('Weekly Sales mean')


# Indeed, at least for the MarkDown 1 and 5, it seems to have a different mean depending if it's holiday or not (which statistical significance could be assessed confirmed by ANOVA).

# <a id="ch2_3"></a>
# ### Visualizing categorical data

# In[ ]:


cat_cols = full.select_dtypes(['category']).columns[:-2]


# In[ ]:


def plt_category(category):
    plt.figure(figsize=(12,6))
    plt_order=full.groupby(category)['Weekly_Sales'].mean().sort_values().index
    fig = sns.pointplot(data=full, x=category, y=y_target, hue='IsHoliday', order=plt_order)
    return fig


# In[ ]:


list(map(plt_category, cat_cols));


# These graphics allow us to draw some preliminary conclusions:
# - Holidays do not influence much the weekly sales according to the store, department or type;
# - The holiday in November (Thanksgiving) has more influence than others, especially the one from December (Christmas), that showed a negative correlation;
# - Still, for each year, there were more weekly sales on holidays.

# It's also interesting to see the relationship of the MarkDown columns with the Store Type and holidays:

# In[ ]:


fig, axes = plt.subplots(1, 5, figsize=(20,5))

for i in range(5):
    mkdn = f'MarkDown{i+1}'
    plt_order=full.groupby('Type')[mkdn].mean().sort_values().index
    sns.pointplot(data=full, x='Type', y=mkdn, hue='IsHoliday', order=plt_order, ax = axes[i])
    axes[i].set_title(mkdn)
    axes[i].set_ylabel('')


# Here we see an evident pattern of correlation of Markdown with Store Type (other than C) and with holidays, which we've already seen in the correlation plot before.

# <a id="ch3"></a>
# ## Dealing with missing data

# Let's see how NAs are distributed on our data:

# In[ ]:


plt.figure(figsize=(12,8))
l = full[full.Set=='Train'].tail(1).index[0]
f = sns.heatmap(full.isnull(), cbar=False, yticklabels='')
f.axhline(l, ls='--', color='r'); # Separate the last occurrence from the train dataset


# The red line delimitates the training and test dataset (train dataset, the biggest one, is above the line). We can see there's no information about Weekly sales for the test dataset (which makes sense, since we're going to predict it). There are a lot of missing values in the Holiday column, which it's also expected due to the manner I made it.
# 
# Interestingly, there's a lot of missing values for the Markdown columns in the training dataset. It's according to the data description: "MarkDown data is only available after Nov 2011, and is not available for all stores all the time". We have to deal with it, but let's first look at how many NAs are for each variable.

# In[ ]:


# Get percentage of NAs for each variable by dataset
nas_perc = full.groupby('Set').apply(lambda x: (x.isna().sum() / full.shape[0])*100).T.sort_values('Train', ascending=False)
nas_perc = nas_perc[nas_perc.apply(lambda x: sum(x)>0, axis=1)]

nas_perc = (
    nas_perc.reset_index()
    .melt(id_vars='index', value_vars=['Test',"Train"])
    .rename({'index':'Variable', 'value': 'NA (%)'}, axis=1)
    .iloc[:, [1,0,2]]
)


# In[ ]:


plt.figure(figsize=(8,6))
f = sns.barplot(data=nas_perc.sort_values(['Set','NA (%)'], ascending=False),
            y='Variable', x='NA (%)', hue='Set')
f.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
f.set_xlabel('NAs');
plt.grid()


# I've decided to deal with missing values in the MarkDown columns by replacing them by the median per group of Type and Holiday since there is a correlation for these variables. For missing values outside these groups, I'll replace them by the overall median of the respective variable.
# 
# I'll take a similar approach for filling missing values to Unemployment and CPI variables, where I'll replace the NAs by the median per Store Type. These modifications will be done on each set separately:

# In[ ]:


# Split the data into training and testing datasets
train = full[full.Set=='Train'].reset_index(drop=True).copy()
test = full[full.Set=='Test'].reset_index(drop=True).copy()


# In[ ]:


## Filling missing values as described above ##
## Train dataset
# For each Markdown column
for col in ["MarkDown" + str(i) for i in range(1,6)]:
    # Get mean by Type and Holiday columns
    grouped = train.groupby(['Type', 'Holiday']).median()
    # For each group
    for name in grouped.index:
        # Update missing values by group median
        idx = (train[col].isna()) & (train.Type==name[0]) & (train.Holiday==name[1])
        train.loc[idx, col] = grouped.loc[name, col]
        # Update remaining missing values by general median
        train.loc[train[col].isna(), col] = train[col].median()
## Test dataset
# For each Markdown column
for col in ["MarkDown" + str(i) for i in range(1,6)]:
    # Get mean by Type and Holiday columns
    grouped = test.groupby(['Type', 'Holiday']).median()
    # For each group
    for name in grouped.index:
        # Update missing values by group median
        idx = (test[col].isna()) & (test.Type==name[0]) & (test.Holiday==name[1])
        test.loc[idx, col] = grouped.loc[name, col]
        # Update remaining missing values by general median
        test.loc[test[col].isna(), col] = test[col].median()
# For Unemployment & CPI columns
for col in ['Unemployment', 'CPI']:
    # Get mean by Store Type
    grouped = test.groupby('Type').median()
    # For each group
    for name in grouped.index:
        # Update missing values by group median
        idx = (test[col].isna()) & (test.Type==name)
        test.loc[idx, col] = grouped.loc[name, col]


# <a id="ch4"></a>
# # Machine Learning

# By looking at the explanatory analysis above, it seems clear there's no linear relationship between the target (weekly sales) and the predictor variables. Therefore, linear regression must not be useful for this data. So I'll try a set of methods that can be used both to classification and regression based on [Decision Trees (DTs)](https://scikit-learn.org/stable/modules/tree.html#tree).

# Let's first select the target columns (target and predictor variables), drop the unwanted ones, and scale the values to get better performance of the algorithms we're running.

# In[ ]:


# Drop columns like object type
obj_like_cols = ['Type', 'Holiday', 'Set']
train = train.drop(obj_like_cols, axis = 1)
test = test.drop(obj_like_cols, axis = 1)
# Transform categorical cols into numeric
cat_cols = test.select_dtypes('category').columns
train[cat_cols] = train[cat_cols].astype('int64')
test[cat_cols] = test[cat_cols].astype('int64')
# Set all integers to int64
train[train.select_dtypes('uint8').columns] = train[train.select_dtypes('uint8').columns].astype('int')
test[test.select_dtypes('uint8').columns] = test[test.select_dtypes('uint8').columns].astype('int')
# Scale variables
cols_to_scale = train.select_dtypes(['int', 'float']).drop(['Weekly_Sales'], axis=1).columns
scaler = preprocessing.StandardScaler().fit(train[cols_to_scale])
train[cols_to_scale] = scaler.transform(train[cols_to_scale])


# Since this competition was evaluated on the weighted mean absolute error (WMAE), I'll define a function for calculating this metric (although I'll analyze the R-square as well):

# In[ ]:


def WMAE(y, y_pred, isholiday):
    W = np.ones(y_hat.shape)
    W[isholiday == 1] = 5
    metric = (1/np.sum(W))*np.sum(W*np.abs(y-y_hat))

    return metric


# Then we run the models and get the error metrics.

# In[ ]:


# Get predictor variables
X = train[['Store', 'Size', 'Dept', 'Week', 'Month', 'Year', 'IsHoliday', 'Temperature', 'Fuel_Price', 
          'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 
          'Type_A', 'Type_B', 'Type_C', 'Holiday_Christmas', 'Holiday_Labor Day',
          'Holiday_Super Bowl', 'Holiday_Thanksgiving']]
# Target variable
y = train[y_target]
# Split trainning data into train and test (validating) data
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


# # Select best parameters and get mean score by cross validation
# gsc = GridSearchCV (
#         estimator=RandomForestRegressor(),
#         param_grid={
#             'max_depth': range(3,7),
#             'n_estimators': (10, 50, 100, 150, 200),
#         },
#         cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# grid_result = gsc.fit(X, y)
# best_params = grid_result.best_params_
# rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"])
# scores = cross_val_score(rfr, X, y, cv=10, scoring='r2')
# np.mean(scores)


# In[ ]:


# Set the models
models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1),
    'extraTreesRegressor': ExtraTreesRegressor(n_estimators=100, max_features='auto', random_state=1),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=1)
}

# Dict for storing model and metrics
res = {}

# Fit the models and performing the prediction
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    wmae = WMAE(X_test.IsHoliday, y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    res[model_name] = (model, wmae, r2)
    print(f'{model_name}:')
    print(f'WMAE = {wmae} and R-square = {r2}')


# Visualizing the estimated importance for each feature from the random forest model:

# In[ ]:


(pd.DataFrame(res['RandomForestRegressor'][0].feature_importances_, index=X.columns, columns=['Importance'])
.sort_values('Importance', ascending=False))


# Let's rerun it, excluding some variables with low importance.

# In[ ]:


# Running again without the columns: Type_C, Year, Holiday_Labor_Day and Holiday_Super_Bowl.
X = train[['Store', 'Size', 'Dept', 'Month', 'IsHoliday', 'Temperature', 'Fuel_Price', 
          'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 
          'Type_A', 'Type_B', 'Holiday_Christmas', 'Holiday_Thanksgiving']]
y = train[y_target]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Set the models
models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1),
    'extraTreesRegressor': ExtraTreesRegressor(n_estimators=100, max_features='auto', random_state=1),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=1)
}

# Dict for storing model and metrics
res = {}

# Fit the models and performing the prediction
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    wmae = WMAE(X_test.IsHoliday, y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    res[model_name] = (model, wmae, r2)
    print(f'{model_name}:')
    print(f'WMAE = {wmae} and R-square = {r2}')


# There's no much difference, so let's keep the first one, which performed a bit better. All three models showed similar results. I'll keep the RandomForest Regressor due to the reasonable tradeoff.
# 
# Now, let's fit the chosen model with the full training data.

# In[ ]:


# Order values by store, department and date
test = test.sort_values(['Store','Dept','Date']).reset_index(drop=True)

# Get Id as requested by the competition
test['Id'] = (test[['Store','Dept','Date']]
              .astype('str')
              .apply('_'.join, axis=1)
)


# In[ ]:


# Select variables
X = train[['Store', 'Size', 'Dept', 'Week', 'Month', 'Year', 'IsHoliday', 'Temperature', 'Fuel_Price', 
          'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 
          'Type_A', 'Type_B', 'Type_C', 'Holiday_Christmas', 'Holiday_Labor Day',
          'Holiday_Super Bowl', 'Holiday_Thanksgiving']]
y = train[y_target]

# Full Model
rf = RandomForestRegressor(n_estimators=100, random_state=1)
rf.fit(X, y)
# Predict testing data
y_hat = rf.predict(test[X.columns])


# <a id="ch5"></a>
# # Submission

# In[ ]:


# Load sample submission
sample_submission = pd.read_csv(f'{filepath}sampleSubmission.csv.zip', compression='zip')

# Create own submission dataframe
submission = pd.DataFrame({'Id': test['Id'], 'Weekly_Sales': y_hat})

# Compare column names and Id row order
print(all(sample_submission.columns == submission.columns)) # column names
print(all(sample_submission.Id == submission.Id)) # Id names


# In[ ]:


# Writing submission file
submission.to_csv('submission.csv', index=False)

