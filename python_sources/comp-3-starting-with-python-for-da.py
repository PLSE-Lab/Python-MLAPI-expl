#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# WILL KOEHRSEN'S NOTEBOOK HAS BEEN FOLLOWED TO LEARN MORE ABOUT
# DA USING PYTHON AS A LANGUAGE

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder # for dealing with categorical variables

import os # File system management

import warnings # Suppress warnings
warnings.filterwarnings('ignore')

#Importing libraries for plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# reading the training data
app_train = pd.read_csv('../input/application_train.csv')
print ('Training data shape: ', app_train.shape)
app_train.head(10)


# In[ ]:


# reading the test data
app_test = pd.read_csv('../input/application_test.csv')
print ('Testing data shape: ', app_test.shape)
app_test.head() 


# In[ ]:


# determining the two target values
app_train['TARGET'].value_counts()


# In[ ]:


app_train['TARGET'].plot.hist()


# In[ ]:


# function to calculate missing values by column
def missing_val_table(df):
    # total missing values
    mis_val = df.isnull().sum()
    
    # percentage of missing values
    mis_val_percent = 100*mis_val/len(df)
    
    # make a table with results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
    
    #rename the columns
    mis_val_tab_rename = mis_val_table.rename(columns = {0: 'Missing Values', 1: '% of Total Values'})
    
    #sort the table in descending order of percentage
    mis_val_tab_rename = mis_val_tab_rename[mis_val_tab_rename.iloc[:,1]
    != 0].sort_values('% of Total Values', ascending=False).round(1)
    
    #print some summary info
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are "+str(mis_val_tab_rename.shape[0])+ " columns that have missing values.")
    
    #return the dataframe with the required info
    return mis_val_tab_rename
                                


# In[ ]:


#missing value stats
missing_val = missing_val_table(app_train)
missing_val.head(10)


# In[ ]:


# inspecting data types of columns
app_train.dtypes.value_counts()


# In[ ]:


# looking at number of unique entries in columns
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


#Categorical variable with 2 unique values will be encoded by 
#label encoder whereas, others by one hot-encoding

# creating a label encoder object
le = LabelEncoder()
le_count = 0

#Iterate through columns to find the matching criteria for Label Encoder
for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            #transforming both training ans test data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            le_count += 1
            
print ('%d columns were label encoded.' %le_count)


# In[ ]:


# one hot-encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[ ]:


# aligning the test and train df, so as to make # of variables equal
train_labels = app_train['TARGET']
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# adding the target back in training dataset
app_train['TARGET'] = train_labels

print('Training features shape: ', app_train.shape)
print('Testing features shape: ', app_test.shape)


# In[ ]:


# since in the DAYS_BIRTH column days are calculated realtive to loan
# application data, they are negative

# to see the stats in years
(app_train['DAYS_BIRTH']/-365).describe()


# In[ ]:


# similarly describing days employed to spot outliers, if any

app_train['DAYS_EMPLOYED'].describe()


# In[ ]:


# the maximum value of DAYS_EMPLOYED looks like an outlier
# since it is positive unlike others and also equals 100 years 
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')


# In[ ]:


# comparing anomalous and non-anomalous clients by their default rate

anom = app_train[app_train["DAYS_EMPLOYED"] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print ('The non-anomalies default on %0.2f%% of loans' % (100*non_anom['TARGET'].mean()))
print ('The anomalies default on %0.2f%% of loans' % (100*anom['TARGET'].mean()))
print ('There are %d anomalous days of employment' %len(anom))


# In[ ]:


# for dealing with anamolous values, we will replace the values with NaN
# then create a boolean variable which states whether value was 
# anamolous or non-anamolous

app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == 365243
app_train['DAYS_EMPLOYED'].replace({365243 : np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
print('There are %d anomalies in the test data out of %d entries' % 
      (app_train["DAYS_EMPLOYED_ANOM"].sum(), len(app_train)))


# In[ ]:


# making the same changes to anamolous values to the test data

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % 
      (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

app_test['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')


# In[ ]:


# making a correlation table to determine the realtions between 
# two variables, if any

correlations = app_train.corr()['TARGET'].sort_values()

# displaying the correlations 

print ('Most Positive Correlations:\n', correlations.tail(15))
print ('\nMost Negative Correlations:\n', correlations.head(15))


# In[ ]:


# exploring the realtion between age and loan repayment

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])


# In[ ]:


# above relation implies that as the client gets older, 
# chances of him/her repaying the loan increases
# making an histogram of age distribution

plt.style.use('fivethirtyeight') # plot style

plt.hist(app_train['DAYS_BIRTH']/365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age in years');
plt.ylabel('Count');


# In[ ]:


# doing target variable-wise visualization of age distribution
# kernel density estimation plot

plt.figure(figsize = (10, 8))

# KDE plots of loans repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH']/365,
           label = 'Target: 0')

# KDE plots of loans not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH']/365,
           label = 'Target: 1')

# plot labels
plt.xlabel('Age in years'); plt.ylabel('Density'); plt.title('Age Distribution')


# In[ ]:


# the graph showed an expected slight skewness towards younger
# population which is expected from correlation value

# now we will look at average failure to repay loans
# for that we will be making age categories of 5 years each

# age info in a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH']/365

# making age categories
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], 
                        bins = np.linspace(20,70, num = 11))
age_data.head(10)


# In[ ]:


# grouping by age group and calculating average for each variables
# the mean target value will tell percentage of loans unpaid 

age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups


# In[ ]:


# plotting histogram for better visualization of failure to pay 
# loans by age group

plt.figure(figsize = (8,8)) # dimensions of the plot area

plt.bar(age_groups.index.astype(str), 100*age_groups['TARGET'])
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)');
plt.ylabel('Failure to Repay in %age')
plt.title('Failure to repay by Age group');


# In[ ]:


# the above graph shows a very clear tend of younger people 
# at a greater risk of being a defaulter

# now exploring the 3 variables EXT_SOURCE_1/2/3 as they had the 
# strongest negative correlation
# documentation is not very clear about the meaning these variables
# convey, but we will be using them anyway as they could prove
# important for predicting target variable
# looking at the correlations again

ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
                     'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs


# In[ ]:


# making a correlation heatmap for better visualization

plt.figure(figsize = (8,6))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, 
           annot = True, vmax = 0.6)
plt.title('Correlation heatmap');


# In[ ]:


# all the ext_source show negative corr with target
# implies that as the ext_source increases, client is more likely
# to repay the loan

# distribution of each of these variable color coded according 
# to target variable

plt.figure(figsize = (10,12)) 

# iterating through sources
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # creating a new sub-plot for each variable
    plt.subplot(3, 1, i+1)
    # plotting repaid loans
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], 
               label = 'Target: 0')
     # plotting loans not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], 
               label = 'Target: 1')
    
    # plot labels
    plt.title('Distribution of %s by Target Value' %source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)


# In[ ]:


# making pair plots
# the code may be obscure, but hang on buddy

# copying data for plotting 
plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

# adding the age of client in years
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

# dropping na values and limiting to first 100k fields
plot_data = plot_data.dropna().loc[:100000, :]

# function to calculate correlation coeff. between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
               xy = (.2, .8), xycoords=ax.transAxes,
               size = 20)
    
# creating a pairgrid object
grid = sns.PairGrid(data = plot_data, size=3, diag_sharey=False,
                   hue = 'TARGET', 
                   vars = [x for x in list(plot_data.columns) if 
                          x != 'TARGET'])

# sspecifying position of plots in the grid

grid.map_upper(plt.scatter, alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Features Pairs Plot', 
            size = 32, y = 1.05);


# In[ ]:


# EXT_SOURCE_1 and YEARS_BIRTH seem to show a moderate linear
# relation which could be used in Feature Engineering

#---- FEATURE ENGINEERING ----#

# using POLYNOMIAL FEATURES
# avoiding higher degrees so as to prevent overfitting 
# making a new dataframe for polynomial features

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                          'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                          'DAYS_BIRTH']]

# handling missing values

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])

# imputing missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

# creating the polynomial object with a specific degree
poly_transformer = PolynomialFeatures(degree = 3)


# In[ ]:


# training the polynomial features
poly_transformer.fit(poly_features)

# transforming the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print ('Polynomial Features Shape: ', poly_features.shape)


# In[ ]:


# getting the names of the new features

poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1',
                'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]


# In[ ]:


# finding correlations of new variables with the target variable 
# creating a database for the features

poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names
                            (['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

# adding in the target
poly_features['TARGET'] = poly_target

# finding the correlations with the target 
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# displaying the most negative and the most positive
print (poly_corrs.head(10))
print (poly_corrs.tail(5))


# In[ ]:


# the new variables seem to have a better correlation with the target
# variable which might help in making a better model
# adding the new features to the original databases to try it out

poly_features_test = pd.DataFrame(poly_features_test, 
                    columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))

# merging the dataframes based on primary key
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', 
                                how = 'left')

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', 
                                how = 'left')

# aligning the dataframes
app_train_poly, app_test_poly = app_train_poly.align(app_test_poly,
                                join = 'inner', axis = 1)

# printing the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape: ', app_test_poly.shape)


# In[ ]:


# using financial knowledge to improve the model

app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
    


# In[ ]:


app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


# In[ ]:


# kde plots to visualize these new variables created

plt.figure(figsize = (12,20))

for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 
                             'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    
    # creating a new subplot for better clarity
    plt.subplot(4, 1, i+1)
    
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0,
                                    feature], label = 'Target: 0')
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1,
                                    feature], label = 'Target: 1')
    
    # labelling the plots
    plt.title('Distribution of %s by Target' %feature)
    plt.xlabel('%s' %feature); plt.ylabel('Density')
    
plt.tight_layout(h_pad = 2.5)


# In[ ]:


# the graph don't give much about the usefulness, have to use it 
# in the actual model

#----LOGISTIC REGRESSION----#

# preprocessing by filling in the missing values and 
# normalizing the range of features <feature scaling>

from sklearn.preprocessing import MinMaxScaler, Imputer

# temporarily dropping target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()

# feature names
features = list(train.columns)

# copying the test data
test = app_test.copy()

# replacing missing values with the median
imputer = Imputer(strategy = 'median')

# scale each feature to 0-1 for normalization
scaler = MinMaxScaler(feature_range = (0,1))

# fitting on the training data
imputer.fit(train)

# transforming both the training and the testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

# normalizing both the sets with scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression

# making the model with a specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# training using the training data
log_reg.fit(train, train_labels)


# In[ ]:


# the model has been trained and now could be used to make predictions
# on probabilities of loan repayment
# we will be using predict.proba methos with returns mx2 matrix. 
# we need probability of non-repayment of the loan, so will use second column

log_reg_pred = log_reg.predict_proba(test)[:, 1]


# In[ ]:


# the solution must be in the required format with SK_CURR_ID and TARGET

submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()


# In[ ]:


# saving the submission file to be submitted
submit.to_csv('log_reg_mod.csv', index = False)


# In[ ]:


#---RANDOM FOREST---#
from sklearn.ensemble import RandomForestClassifier

# making the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100,
                random_state = 50, verbose = 1, n_jobs = -1)


# In[ ]:


# training on the TRAIN data
random_forest.fit(train, train_labels)

# extracting importance of features
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# making predictions on the test data
prediction = random_forest.predict_proba(test)[:,1]


# In[ ]:


# making a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = prediction

# saving the file
submit.to_csv('random_forest_mod.csv', index = False)


# In[ ]:


# using the engineered variables to see if they have an impact 
# on the overall score

poly_features_names = list(app_train_poly.columns)

# imputing the polynomial features

imputer = Imputer(strategy = 'median')
poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)

# scaling the features to normalize
scaler = MinMaxScaler(feature_range = (0, 1))

poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators=100, 
                    random_state=50, verbose=1, n_jobs = -1)


# In[ ]:


# training the data
random_forest_poly.fit(poly_features, train_labels)

# making the predictions
predictions = random_forest_poly.predict_proba(poly_features_test)[:,1]


# In[ ]:


# using the domain features which was created based on financial 
# knowledge

app_train_domain = app_train_domain.drop(columns = 'TARGET')

domain_features_names = list(app_train_domain.columns)

# Impute the domainnomial features
imputer = Imputer(strategy = 'median')

domain_features = imputer.fit_transform(app_train_domain)
domain_features_test = imputer.transform(app_test_domain)

# Scale the domainnomial features
scaler = MinMaxScaler(feature_range = (0, 1))

domain_features = scaler.fit_transform(domain_features)
domain_features_test = scaler.transform(domain_features_test)

random_forest_domain = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

# Train on the training data
random_forest_domain.fit(domain_features, train_labels)

# Extract feature importances
feature_importance_values_domain = random_forest_domain.feature_importances_
feature_importances_domain = pd.DataFrame({'feature': domain_features_names, 'importance': feature_importance_values_domain})

# Make predictions on the test data
predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]


# In[ ]:


# submission file
# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_domain_mod.csv', index = False)


# In[ ]:


# looking at importance of features through visualization

def plot_feature_importances(df):
    """
    Plot importances of a feature returned by a model. Can work with with 
    any measure of feature importance, in case higher importance implies 
    better.
    
    Args:
        df (dataframe): must have features in a column called 'features'
        importances in a column called 'importance'
    Returns:
        a plot of 15 most important features 
        
        df (dataframe): feature importances sorted by importance (highest
                to lowest) with a column for normalized importance
    """
    
    # sort features according to importance
    
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # normalizing the feature importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    
    # making a horizontal bar graph for feature importance
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # reversing to plot the most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
           df['importance_normalized'].head(15),
           align = 'center', edgecolor = 'k')
    
    # setting the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # plot labelling
    plt.xlabel('Normalized Importace'); plt.title('Feature Importances')
    plt.show()
    
    return df


# In[ ]:


# displaying the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)


# In[ ]:


# displaying the feature importances inclusive of domain features
feature_importances_domain_sorted = plot_feature_importances(
    feature_importances_domain)


# In[ ]:


# all the domain specific features made it to the most important features
# using the light gradient boosting model

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

# ohe --> one hot encoding

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    """ Train and test a LightGBM using cross validation.
    
    Parameters:
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
    # extracting the IDs
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # extracting the labels for training
    labels = features['TARGET']
    
    # removing the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Aligning the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # integer label encoding
    elif encoding == 'le':
        
        # creating a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # iterating through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # mapping the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # recording the categorical indices
                cat_indices.append(i)
    
    # catching error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # extracting feature names
    feature_names = list(features.columns)
    
    # converting to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # creating the kfold object
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
    
    # Iterating through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # creating the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Training the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # recording the best iteration
        best_iteration = model.best_iteration_
        
        # recording the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # making predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # recording the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # recording the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # cleaning up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # making the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # making the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # adding the overall scores to the metrics
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


submission, fi, metrics = model(app_train, app_test)
print('Baseline Metrics')
print(metrics)


# In[ ]:


submission.to_csv('lgb_mod_1.csv', index = False)


# In[ ]:


# using domain engineered variables
app_train_domain['TARGET'] = train_labels

# testing the domain knowledge features
submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
print('Baseline with domain knowledge features metrics')
print(metrics_domain)


# In[ ]:


submission_domain.to_csv('lgb_mod_domain.csv', index = False)


# In[ ]:




