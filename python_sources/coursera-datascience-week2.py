#!/usr/bin/env python
# coding: utf-8

# ### Explatory Data Analysis
# - What and why
# - Things to explore 
# - Exploration and visualization tools
# - Dataset cleanin
# - Kaggle EDA
# 
# #### What is EDA?
# - Understanding and building an intiution about the data
# - Generate hypothesis and find insights
# - Visualization -> Idea , Idea -> Visualization
# - Find magic features
# - Find data leaks: mistakes made by the organizers during data preparation

# ### Building intuition about the data
# - Getting knowledge domain
#     - Understand the column names and the problem
# - Checking if the data is intuitive
#     - Ex, age can't be 350
#     - Create feature named "is_incorrect" and mark incorrect data
# - Understanding how the data was generated
#     - To set up a proper validation

# ### Exploring anonymized data
# - Guess the meaning and the types of the columns
# - Find relation between pairs
# - Find feature groups
# - Try to decode the features
#     - find scaling parameter, backscale and shift back to reach original data
# - Use df.dtypes, df.info(), x.value_counts(), x.isnull()

# In[ ]:


#Label Encoder
#for c in train.columns[train.dtypes == 'object']:
#    X[c] = X[c].factorize()[0]

#plt.plot(rf.feature_importances_)
#plt.xticks(np.arange(X.shape[1]), X.columns.tolist(),rotation=90);


# ### Visualization
# #### Explore Individual Features
# - plt.figure(figsize=(15,5))
# - Histograms
#     - plt.hist(x)
#     - use number of bins
#     - take log and rehistogram to see from different perspective
# - Plots
#     - plt.plot(x,'.')
#     - X axis: row index, Y axis: feature values
# - Scatter Plots
#     - plt.scatter(range(len(x)),x,c=y)
# - Statistics
#     - x.var(), x.mean()
#     - x.describe(), x.isnull(), x.value_counts()
#     
# #### Explore feature relation
# - Scatter Plots
#     - plt.scatter(x1,x2)
# - Correlation Plots
#     - pd.scatter_matrix(df)
#     - df.corr(),plt.matshow(..)
# - Plot (index vs feature statistics)
#     - df.mean().sort_values().plot(style='.')

# ### Dataset Cleaning and things to check
# #### Dataset Cleaning
# - Constant features
#     - remove if all same
#     - train.nunique(axis = 1) == 1
# - Duplicated features
#     - remove one of identical columns
#     - train.T.drop_duplicates()
#     - for categorical features:
# >         for f in categorical_features:
# >             train[f] = train[f].factorize()
# >         train.T.drop_duplicates()
#         
# #### Other things to check
# - Duplicated rows
#     - Understand and why
# - If exact row appears at both train and test sets, label manually
# - Check if data is shuffled
#     - If not shuffled, high chance to find data leakage. check and plot for rolling mean and mean for target.
# - Check null counts:
#     - row: df.isnull().sum(axis=1).head(15)
#     - column : df.isnull().sum(axis=0).head(15)
# - Check unique counts:
#     - df.nunique(dropna=False).sort_values(), drop column if it has only a unique value
# - Check similar columns to create new features
# - Get column types:
#     - num_cols = list(df.select_dtypes(exclude=['object']).columns)
# - New features:
#     - mod, diff, year, month, date
# - Sort correlation matrix

# 

# ### Validation Strategies
# - Holdout: sklearn.model_selection.ShuffleSplit
#     - ngroups = 1
#     - Data -> 0.8*Train + 0.2*Test
# - K-fold: sklearn.model_selection.KFold
#     - ngroups = k
#     - Data = (0.8*Train + 0.2*Test) + (0.6*Train + 0.2*Test + 0.2*Train) + (04*Train + 0.2*Test + 0.4*Train) + (0.2*Train + 0.2*Test + 0.6Train) + (0.2*Test + 0.8*Train):
#     - Final measure is the average
# - Leave-one-out: sklearn.model_selection.LeaveOneOut
#     - ngroups = len(train)
#     - Data = k-1 for train, 1 for test
#     - Use with small amount of data
# - Stratification:
#     - Useful for small, unbalanced datasets, multiclass classification
# ### Data Splitting Strategies
# - Set up validation to mimic train/test split. ***
# - Logic of feature generation depends on the data splitting strategy
# - If time series problem, don't random split.
# - Different splitting strategies depend on:
#     - generated features
#     - way the model will rely on that features
#     - some kind of target leak
# - Split Types:
#     - Random, rowwise
#     - Timewise(moving window validation)
#     - By id
#     - Combined
# ### Validation Problems
# - Validation Stage
#     - Causes:
#         - Too little or diverse and inconsistent data
#     - Solution:
#         - Different Kfolds with differen number of folds and different random seeds.
# - Submission Stage
#     - Think LB as another validation split
#     - Causes:
#         - too little data on LB
#         - train and test are from different distributions   
#         - incorrect train/test split
#     - Solution:
#         - Leaderboard probing
#         - Distribute data in the same way in train and test
#     - train: 80%man-20%woman, test: 20%man-80%woman -> try to mimic test set in validations.
#     - Expect LB shuffle because of randomness, little amount of data and different public/private distributions

# ### Data Leakage
# - Unexpected information in the data that allows us to make unrealistically good predictions.
# - Leaks in time series
#     - Split should be done in time, check it first. If not by time, it mat be a leak. Features like "prices_next_week" will be the most important
#     - Look for the test set and create new features about future.
# - Unexpected Information
#     - Try to find meta information
#     - Information in IDs (may be hash of something)
#     - Row Order
# ### Leaderboard Probing
# - Submit your result multiple times, each time change different part to understand ground truth.
# ### Expedia Kaggle Competition
# - Which hotel group a user is going to book. Search results, clicks, books given.
#     - worked on spherical distance (Haversine Formula)
