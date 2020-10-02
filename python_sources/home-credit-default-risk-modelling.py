#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk Modelling
# 
# 
# ## Introduction
# 
# I am no expert in the field of Risk modelling, but I am aware that this is typical process in many financial services companies. Any organisation that gives Credit to a customer, whether that customer is a typical consumer, business or government organisation, should be able to understand their risk (and potential exposure) should the client default on the loan. This means that Credit Risk modelling is important and is worth learning more about.
# 
# ## Imports

# In[ ]:


# All Package Imports
import pandas as pd
import numpy as np
import os
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import time

# import sklearn packages for modelling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


# First and foremost, we need to understand the data which we have for this model. There are multiple datasets, but we will need to focus on the application train (and application test) datasets first and then use the other data to enrich the train and test with more features to improve the model. 
# 
# So the first objective in this notebook will be to:
# * Exploratory Data Analysis
# * Data Transformation (Encoding)
# * Data Transformation (Scaling/Normalisation)
# * Benchmark Models Cross-validation
# 
# Then I can go on to explore (time permitting):
# * Feature engineering using existing dataset
# * Feature engineering using additional data
# 
# But without further ado, lets start by exploring the data to understand intuitively what is contained within it.

# ## Data Load

# In[ ]:


# List files available
print(os.listdir("../input"))


# In[ ]:


# load and print head of the a_train dataset
a_train = pd.read_csv(r'../input/application_train.csv')
a_train.head()

# keep original version of a_train for exploratory visualisations
original_a_train = a_train.copy(deep = True)


# In[ ]:


# Train matrix shape
a_train.shape


# In[ ]:


# load and print head of the a_train dataset
a_test = pd.read_csv(r'../input/application_test.csv')
a_test.head()


# In[ ]:


# Test matrix shape
a_test.shape


# So there are 121 features in the train and test data, with the train data having an additional target column. We can see that the test set is approximately a sixth of the size of the training set.

# ## Exploratory Data Analysis
# 
# EDA is useful for us to be able to understand what sort of data we are working with and build up some intuition about trends within the data. 
# 

# ### Missing Values
# 
# Many machine learning models are intolerant to missing values, so its important to understand their distribution so that we can come up with an appopriate strategy for handling them.

# In[ ]:


# function to find missing values in datafame so we can reuse if we look at other data sources
# including proper docstring
def missing_values_table(df):
        
        # sum of missing values
        mis_val = df.isnull().sum()
        
        # % of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # concat table
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% Missing Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0
                                                             ].sort_values('% Missing Values', ascending=False).round(1)
        
        # Print some summary information
        print ("The dataframe has {} columns.\n".format(str(df.shape[1])),      
                "There are {} columns that have missing values.\n".format(str(mis_val_table_ren_columns.shape[0])),
               "There are {} columns that have no missing values".format(int(df.shape[1]) - int(mis_val_table_ren_columns.shape[0])) )
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    


# In[ ]:


missing_values_table(a_train)


# There are a lot of missing values in some of the columns (c. 50% or more). We will need to impute these values for many machine learning algorithms to work, we can consider the imputation method later on once we are ready to preprocess the data before loading into the model. Another option would be to drop rows with lots of missing values, but these could be very important to fitting the model. Likewise we can drop the columns which have a high number of missing values, but the information we do have could still be useful to training a good model. For now it makes sense to keep these columns, and work on imputing or removing features once we are trying to optimise fitting a model later.

# ### Columns Summary Statistics
# 
# From the previous section we can immediately see that we have a lot of columns and it is difficult to understand what they contain by using .head() alone. So it makes sense to get some more summary statistics about the columns.

# In[ ]:


# Columns Data Types
train_dtypes = pd.DataFrame(a_train.dtypes.value_counts()).reset_index()
train_dtypes.columns = ['dtypes', 'column count']

train_dtypes


# In[ ]:


# create dict object from columns and datatypes
columns = a_train.columns.to_series().groupby(a_train.dtypes).groups
for key in columns.keys():
    print('\nData Type {} Columns:'.format(key))
    pprint(list(columns[key]))


# We can now see there split of dtypes across the columns. One thing that jumps out is that a lot of the columns are int or float compared to object (or string) columns. 
# 
# However, when we look closer at the column names, we can see that many of the int columns are infact 'FLAG' and take the value 0 or 1. So these are essentially categorical variables which have already been binary encoded. This is ideal for machine learning, as we would need to convert the data set to a matrix of numeric values to ensure that the data set can be used by the different ML algorithms we will want to explore later in the workbook.
# 
# There are also a large number of float columns, which appear to be a mixture of customer information and derived summary statistics (avg, modes etc). These variables will likely need scaling to prevent assigning to much weight to any variables which take high values.
# 
# Lets now take a look at the categorical columns.

# In[ ]:


# desribe the categorical data
a_train.loc[:, a_train.dtypes == np.object].describe()


# One thing that jumps out is that there are also some binary categorical columns here, where there are only 2 unique values.  We will need to encode these variables in order to use them in a machine learning model.

# ## Data Transformation (Encoding)
# ### Encoding of Categorical Variables
# 
# In order to encode the categorical variables, we need to use two methods. 
# 
# * Label Encoder - for categorical variables with 2 classes, converting the value to binary (0 or 1)
# * OneHotEncoding - for categorical variables with multiple classes, one hot encoding transformed each class into a new column which is then a binary value.

# In[ ]:


# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in a_train:
    if a_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(a_train[col].unique())) <= 2:
            print("{} was encoded".format(col))
            # Train on the training data
            le.fit(a_train[col])
            # Transform both training and testing data
            a_train[col] = le.transform(a_train[col])
            a_test[col] = le.transform(a_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# In[ ]:


# one-hot encoding of categorical variables
a_train = pd.get_dummies(a_train)
a_test = pd.get_dummies(a_test)

print('Training Features shape: ', a_train.shape)
print('Testing Features shape: ', a_test.shape)
train_labels = a_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
a_train, a_test = a_train.align(a_test, join = 'inner', axis = 1)

# Add the target back in
a_train['TARGET'] = train_labels
print('Aligned Training Features shape: ', a_train.shape)
print('Aligned Testing Features shape: ', a_test.shape)


# ## Continued Exploratory Data Analysis
# ### Target Variable Distribution
# 
# Let's take a look at the distribution of the target variable, ideally we would have a fairly balanced dataset. Many machine learning models produce a undesired fit on highly imbalanced datasets, as most of them train for accuracy. We can see below that the classes are infact quite imbalanced, with the positive class being under represented. There are multiple ways to deal with an unbalanced classification problem which we could go on to explore in a more detailed notebook.

# In[ ]:


# Plot TARGET distribution
a_train['TARGET'].value_counts()
a_train['TARGET'].value_counts().plot(kind='bar', figsize=(10,5), color = ['grey', 'cornflowerblue'])
plt.xlabel('Target Class')
plt.ylabel('Count') 
plt.show()


# TO DO:
# * Visualisations / kde
# * Find anomalies (programatically if possible)
# * Feature Engineer some features
# * Fit initial models with and without features
# * Test a final model

# ### Visualising affect on Target variable using Proportion Target Positive plots or Kernal Density Estimation (KDE) plots
# 
# In order to help us understand the affect an independent variable is having on our target of default or not default, we should ideally visualise the data.  A histogram will be dependent on the underlying distribution of the independent variable, i.e. if there are lots of young people in the dataset, then the positive class may appear to be higher in this group simply because there are more of both classes. There are a couple of ways around this which can help us understand the data better.
# 
# 1.  For categorical values, we can plot the proportion (%) of each category that has a positive class
# 2. For continuous variables, we can use a KDE plot. This plot estimates the histogram as a probability density function (pdf) which is essentially a smoothed histogram, where the area under the chart sums to a probability of 1. This is useful for us to see how the probability of a negative or positive target class changes with different ranges of the independent variable. 
# 
# Below is an example KDE plotted for Age using a function I have written, which enables you to pick you variable but also reverse the scale when required (say for Age which is in minus days)

# In[ ]:


# Create function for plotting kde with scale reversing
def plot_kde(df, var, reverse_scale = False):
    
    plt.figure(figsize = (12, 6))
    
    if reverse_scale == True:
        r = -1
    else:
        r = 1
    
    # KDE plot of loans that were repaid on time
    sns.kdeplot(df.loc[df['TARGET'] == 0, var] * r, label = 'target: negative class', color = 'grey', shade = True)

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(df.loc[df['TARGET'] == 1, var] * r, label = 'target: positive class', color = 'cornflowerblue', shade = True)

    # Labeling of plot
    plt.xlabel('{}'.format(var)); plt.ylabel('Density'); plt.title('KDE for {}'.format(var));
    plt.show()
    plt.close()

# plot age kde plot
plot_kde(a_train,'DAYS_BIRTH', True)


# ### Age
# 
# We can see straight away that age seems to play a big factor in defaulting on a loan. There is a very clear elevation to default in the younger age groups. This is interesting knowledge to have since in future we could use this kind of knowledge to optimise our model if we were removing features (we would not remove this one) and we could look to engineer further more detailed features using this knowledge.

# ### Iterative Graphing
# 
# Whilst not the prettiest visualisations, we can use python to quickly produce multiple iterations over many variables for this type of graph using the function we defined. 
# 
# We can use this to quickly observe many metrics, pull out immediate trends and spot any issues with our data **quickly**, which is the purpose of this kind of initial EDA and model building. 
# 
# In more detailed studies we could spend time producing more detailed visualisations if there is more to understand.

# In[ ]:


# iterate over float variables and plot KDE
for column in original_a_train.loc[:, (original_a_train.dtypes == np.float64)].columns.values:
    # do not plot target 
    if column != 'TARGET':
        # reverse axis if values are negative
        if (original_a_train[column].median() < 0):
            plot_kde(a_train,column, reverse_scale = True)
        else:
            plot_kde(a_train,column)


# ### Trends from Numeric Visualisations
# 
# #### Feature Outliers
# 
# One thing that is immediately obvious from these kde plots is that some of the variables appear to have huge outliers, and so the kde plot is highly skewed. This means we see a tall, sharp peak on one part of the axis followed by a thin layer on the bottom. It is hard to visualise data that is this skewed, and for these plot we need to investigate further these outliers and (if possible) determine whether they are legitimate or due to error. I will look at this in the next section.
# 
# #### Other intesting trends
# 
# * Days since last phone change shows that those who have changed their phones very recently and those who haven't changed their phone is years are more likely to default on a loan. 
# * The EXT_SOURCE plots all appear to hold information for predicting default, as we see quite a strong separation in the distributions
# * Many of the variables are hard to understand, and appear to be scaled aggregations (avgs, modes, medians) of more detailed metrics.

# ### Outliers
# 
# We established from the charts above that there are outliers in some of the numeric features above. In a more detailed study I would look to observe some of these in more detail, but for the purposes of this notebook and producing an intial baseline model I want to look at an example. In model optimisation, I could look at how these outliers may affect the model and process them in a way to ensure a model that is not skewed due to collection error.
# 
# I will look at total income amount as this is the first plot which displays outlier behaviour above.

# In[ ]:


def analyse_outliers(df, column):
    
    # Print Summary Statistics
    print('Summary Statistics:\n')
    print(df[column].describe())

    # find mean and std
    outlier_df = df[column]
    std = outlier_df.std()
    print('\nStandard Deviation: ', std)
    mean =  outlier_df.mean()
    print('Mean: ', mean)

    # how many std is the max
    max_outlier = int((outlier_df.max() - mean) / std)

    # separate outliers over 2 std from mean
    outliers_l = outlier_df[(outlier_df < mean - 2 * std)]
    outliers_h = outlier_df[(outlier_df > mean + 2 * std)]
    print('\nThere are {} low end outliers in the {} dataset'.format(len(outliers_l), column ))    
    print('There are {} high end outliers in the {} dataset'.format(len(outliers_h), column ))
    print('The max value is {} standard deviations from the mean'.format(max_outlier))
    
    return mean, std

income_mean, income_std = analyse_outliers(a_train, 'AMT_INCOME_TOTAL')   


# One thing that really stands out above is how much of an outlier the maximum value is at 492 standard deviations from the mean. 177M salary could potentially be accurate, and with more information about the data this may be possible to find out. Regardless, it is possible that this value will really skew a model fit to the data, and could potentially be removed, particularly if it is found to be an error.
# 
# This kind of analysis could be repeated for more of the features, and the written above function could be reused to enable this.

# ## Continued Exploratory Data Analysis
# ### Looking at trends in the Categorical Variables
# 
# As mentioned above, we can plot proportions of target being the positiive class against each categorical variable value to see which groups have higher default rates.

# In[ ]:


# define function for plotting categorical bar charts for remaining variables
def categorical_plot(df, variable):
    
    plt.figure(figsize = (11, 5))
    
    df_high = df[df['TARGET'] == 1].groupby(variable)['TARGET'].agg('count')
    df_var = df.groupby(variable)['TARGET'].agg('count')
    categorical = df_high.divide(df_var, fill_value = 0) * 100

    # Convert back to df
    df_categorical = categorical.to_frame().reset_index().sort_values('TARGET', ascending = True)

    # Create plot in Plotly for interactive visualisation (with some Starling colours)
    ax = df_categorical.plot(x = variable, y = 'TARGET', kind = 'barh', figsize=(10,10), color = 'cornflowerblue')
    ax.set_xlabel('Target: Positive %')
    ax.set_ylabel(variable)
    plt.title('% postive plot for {}'.format(variable.lower()));
    plt.show()
    plt.close()


# In[ ]:


# iterate over np.object columns and plot
for column in original_a_train.loc[:, original_a_train.dtypes == np.object].columns.values:
    categorical_plot(original_a_train, column)


# ### Trends from Categorical Visualisations
# 
# * Income type, Education and occupation type appear to have quite a range when it comes to default rate. Which makes sense, as you would imagine that those will lower income jobs are more likely to be unable to repay a loan. 
# * Day of application has some very minor variation, but you would hope that the day of the week someone applies on would be less linked to their likelihood of default. Although until obeserving the trend, we can never simply say that for sure.
# * owning car or realty seems to have little affect, which is interesting as you would perhaps link ownership with more financial stability. But the trend shows that once again we should trust the data over our initial intuition.

# ### Correlations
# 
# To conclude our EDA, we can look at correlation coefficients between features and the target. This gives us a more quantitative measure for the relationships we have just explored visually. Below we will use the encoded variables for the correlations.

# In[ ]:


# Find correlations with the target(takes a while due to many features)
correlations = a_train.corr()['TARGET'].sort_values()


# In[ ]:


# print ordered list of correlations
print('Most Positive Correlations:\n')
print(correlations.sort_values(ascending = False).head(16))
print('\nMost Negative Correlations:\n')
print(correlations.head(15))


# This agrees with some of the insight we were able to pull from the visualisations. Target correlates perfectly with itself as expected (almost acts as a good check for our code. Then we can see that age has a strong positive correlation, which is actually reversed because age is negative.
# 
# We also see that the EXT_SOURCE variables do seem to hold some information that will help us predict the target class. 

# ## Model Fitting
# 
# Before doing any feature engineering, lets run a cross validation against the training set. From this we can determine what the benchmark accuracy metrics are for a variety of models. This can help us narrow down which models are performing well on the data. 
# 
# Our training and test data has already been encoded suring the EDA steps. One final thing to do before passing the train data into a cross validation step would be to scale the numeric quantities, so that features with large values don't get assigned far greater weight in some machine learning algorithms.

# In[ ]:


# create X_train, y_train
X_train = a_train.drop('TARGET', axis = 1)
y_train = a_train['TARGET']
X_test = a_test

X_train = X_train.drop('SK_ID_CURR', axis = 1)
X_test = X_test.drop('SK_ID_CURR', axis = 1)

# Feature names
features = list(X_train.columns)


# In[ ]:


# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(X_train)

# Transform both training and testing data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Repeat with the scaler
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)

print('Training data shape: ', X_train.shape)
print('Testing data shape: ', X_test.shape)


# In[ ]:


# Using Cross Validation to find a good model
num_folds = 5
seed = 1
scoring = 'roc_auc'
models = []

# Typical Classifiers
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))

# iterate over models and print cross val scores
results = []
names = []
print('Please wait while models train..')
for name, model in models:
    
    # start timer
    start = time.time()
    
    # Cross Validation
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    # stop timing
    end = time.time()
    time_run = (end - start)/60
    output = "{}--> auroc: {}   (Training Time: {}mins)".format(name, cv_results.mean(), time_run)
    
    print(output)


# We can see there is not much in it between LR and Random Forest both on the evaluation metric and training time. As we progress our solution we should aim to produce a more optimised model, and as such a more complex model such as Random Forest (which is an ensemble method) will gives us more flexibility when we start training hyperparameters. There are also fair more effective algorithms which I could explore, but they are slightly too complex to cover in this inital notebook, or take a very long time to train on a dataset this large.
# 
# So to create our baseline model I will use Random Forest in this instance. 
# 
# ### Training Model and Test Predictions

# In[ ]:


# Train LR Model
RF = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
RF.fit(X_train,y_train)

# Extract feature importances
feature_importance_values = RF.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': feature_importance_values})

# Make predictions on the test data
predictions = RF.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = a_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline.csv', index = False)


# #### Model Performance
# We now have a baseline dataset for submission. The ROC score for the model upon submission was: **0.653**. 
# 
# 0.5 is not better than random guessing, and 0.80 was a winning score in the competition. For a baseline model this is quite satisfying. However, this is the simple model which we can now aim to build on and improve by:
# 
# * Enriching the dataset with additional data (Adding Features)
# * Create new features from existing data (Feature Engineering)
# * Tuning Hyperparameters to improve the model performance
# * Testing more cutting edge machine learning and deep learning models
# 
# #### Feature Importance
# 
# Below is a function I have created to plot feature importances. Looking at the plot we can immediately see that some of our exploratory data analysis had highlighted features which were important.

# In[ ]:


# Function to plot feature importance
def plot_feature_importance(df):

    # Normalize the feature importances to add up to one
    df['Importance_normalized'] = df['Importance'] / df['Importance'].sum()
    df = df.sort_values('Importance_normalized', ascending = True).tail(20)

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 16))

    ax = df.plot(x = 'Feature' , y = 'Importance_normalized', kind = 'barh', figsize=(10,10), color = 'blue')
    
    # Plot labeling
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()
    
    # return top 20 features
    return(df['Feature'])

top20 = plot_feature_importance(feature_importances)


# ## Feature Engineering
# 
# We can now try some quick feature engineering using the other datasets. 

# In[ ]:


# import bureau data
bureau = pd.read_csv(r'../input/bureau.csv')
bureau.head()


# In[ ]:


# create feature dataframe
bureau_agg = bureau['SK_ID_CURR'].unique()
bureau_agg = pd.DataFrame(bureau_agg, columns = ['SK_ID_CURR'])


# In[ ]:


# previous loan count
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()

bureau_agg = bureau_agg.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')


# In[ ]:


# active loan count
active_loan_counts = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'active_loan_counts'})
active_loan_counts.head()

# join new features
bureau_agg = bureau_agg.merge(active_loan_counts, on = 'SK_ID_CURR', how = 'left')

# fill na
bureau_agg = bureau_agg.fillna(0)


# In[ ]:


# join additional features onto train and test
a_train_features = a_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
a_test_features = a_test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
a_train_features = a_train_features.fillna(0)
a_test_features = a_test_features.fillna(0)


# In[ ]:


# plot kde of new features
plot_kde(a_train_features, 'previous_loan_counts')
plot_kde(a_train_features, 'active_loan_counts')


# In[ ]:


print('Training data shape: ', a_train_features.shape)
print('Testing data shape: ', a_test_features.shape)


# ### Re-run Random Forest Model
# 
# Now that we have an enriched train and test dataset, lets retrain the RF model under the same conditions.

# In[ ]:


# create X_train, y_train
X_train = a_train_features.drop('TARGET', axis = 1)
y_train = a_train_features['TARGET']
X_test = a_test_features

X_train = X_train.drop('SK_ID_CURR', axis = 1)
X_test = X_test.drop('SK_ID_CURR', axis = 1)

# Feature names
features = list(X_train.columns)

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(X_train)

# Transform both training and testing data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Repeat with the scaler
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)

print('Training data shape: ', X_train.shape)
print('Testing data shape: ', X_test.shape)

# Train LR Model
RF = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
RF.fit(X_train,y_train)

# Extract feature importances
feature_importance_values = RF.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': feature_importance_values})

# Make predictions on the test data
predictions = RF.predict_proba(test)[:, 1]

# Make a submission dataframe
submit = a_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_features.csv', index = False)

top20 = plot_feature_importance(feature_importances)


# Despite now seeing that both of the new features are showing in the top list for features importance, this model actually underperforms the original baseline model by approx. 3 AUROC Score. 
# 
# That being said feature enrichment and model optimisation will be the path we need to follow to increase the ROC score. With future work, we should be able to increase that ROC score a fair bit more. 

# ## Using an LGBM Model
# 
# LGBM tends to do really well in competitions so I want to try it out here!

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics


# In[ ]:


submission, fi, metrics = model(a_train_features, a_test_features)
print('Baseline metrics')
print(metrics)


# In[ ]:


def plot_feature_importances(df, num_bars = 15):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:num_bars]))), 
            df['importance_normalized'].head(num_bars), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:num_bars]))))
    ax.set_yticklabels(df['feature'].head(num_bars))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df


# In[ ]:


fi_sorted = plot_feature_importances(fi, 25)


# In[ ]:


submission.to_csv('baseline_lgb_features.csv', index = False)

