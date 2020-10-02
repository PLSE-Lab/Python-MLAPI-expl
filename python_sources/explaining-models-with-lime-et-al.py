#!/usr/bin/env python
# coding: utf-8

# # Explaining a Tabular Model with LIME
# 
# [LIME](https://github.com/marcotcr/lime), or Local Interpretable Model-Agnostic Explanations, is one technique that comes up a lot as I search for explaining machine learning models. "local" refers to trying to explain which features contribute the most to explain a prediction/classification of a given datapoint. For example, explaining why a patient was classified with a flu, which features and their values contributed to the classification for this specific patient.
# 
# Here I try the LIME Python implementation on a few datasets to see what kind of results I get. This is on tabular datasets (so CSV files). LIME can also be applied on images and text, maybe another day. Datasets:
# - Titanic: What features contribute to a specific person classified as survivor or not?
# - Heart disease UCI: What features contribute to a specific person being classified at risk of heart disease?
# - Boston housing dataset: What features contribute positively to predicted house price, and what negatively?
# 
# Algorithms applied:
# - Titanic: classifiers from LGBM, CatBoost, XGBoost
# - Heart disease UCI: Keras multi-layer perceptron NN architecture
# - Boston housing dataset: regressor from XGBoost

# In[ ]:


import numpy as np
import pandas as pd

import xgboost as xgb
import lime
import lime.lime_tabular

from sklearn.model_selection import cross_val_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#gender_submission.csv is an example prediction file predicting all female passengers survive, and no others do
#i guess thats politically correct
#!cat /kaggle/input/titanic/gender_submission.csv


# ## Explaining Titanic Survival Classification

# In[ ]:


df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train.head()


# In[ ]:


df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_test.head()


# In[ ]:


df_train["train"] = 1
df_test["train"] = 0
df_all = pd.concat([df_train, df_test], sort=False)
df_all.head()


# Embarked column has some null values. Filling those with some value that is not in the dataset:

# In[ ]:


df_all["Embarked"].unique()


# In[ ]:


df_all["Embarked"] = df_all["Embarked"].fillna("N")


# Following are some extra features I picked up earlier from various Titanic kernels here at Kaggle. Unfortunately that was a while back and I lost all the references to the original ones, but thanks anyway:

# In[ ]:


def parse_cabin_type(x):
    if pd.isnull(x):
        return None
    #print("X:"+x[0])
    #cabin id consists of letter+numbers. letter is the type/deck, numbers are cabin number on deck
    return x[0]

def parse_cabin_number(x):
    if pd.isnull(x):
        return -1
#        return np.nan
    cabs = x.split()
    cab = cabs[0]
    num = cab[1:]
    if len(num) < 2:
        return -1
        #return np.nan
    return num

def parse_cabin_count(x):
    if pd.isnull(x):
        return np.nan
    #a typical passenger has a single cabin but some had multiple. in that case they are space separated
    cabs = x.split()
    return len(cabs)


# In[ ]:


df_all["cabin_type"] = df_all["Cabin"].apply(lambda x: parse_cabin_type(x))
df_all["cabin_num"] = df_all["Cabin"].apply(lambda x: parse_cabin_number(x))
df_all["cabin_count"] = df_all["Cabin"].apply(lambda x: parse_cabin_count(x))
df_all["cabin_num"] = df_all["cabin_num"].astype(int)
df_all.head()


# In[ ]:


df_all["family_size"] = df_all["SibSp"] + df_all["Parch"] + 1
df_all.head()


# In[ ]:


# Cleaning name and extracting Title
for name_string in df_all['Name']:
    df_all['Title'] = df_all['Name'].str.extract('([A-Za-z]+)\.', expand=True)
df_all.head()


# In[ ]:


# Replacing rare titles 
mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 
           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 
           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 
           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}
           
df_all.replace({'Title': mapping}, inplace=True)
#titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']


# In[ ]:


titles = df_all["Title"].unique()
titles


# In[ ]:


titles = list(titles)
# Replacing missing age by median age for title 
for title in titles:
    age_to_impute = df_all.groupby('Title')['Age'].median()[titles.index(title)]
    df_all.loc[(df_all['Age'].isnull()) & (df_all['Title'] == title), 'Age'] = age_to_impute


# In[ ]:


df_all[df_all["Fare"].isnull()]


# In[ ]:


df_all.loc[152]


# In[ ]:


p3_median_fare = df_all[df_all["Pclass"] == 3]["Fare"].median()
p3_median_fare


# In[ ]:


df_all["Fare"].fillna(p3_median_fare, inplace=True)


# In[ ]:


df_all.loc[152]


# In[ ]:


df_all = df_all.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)


# In[ ]:


df_all["cabin_type"].value_counts()


# In[ ]:


df_all["cabin_type"] = df_all["cabin_type"].fillna("Z")


# In[ ]:


df_all["cabin_type"].value_counts()


# In[ ]:


label_encode_cols = ["Sex", "Embarked", "Title", "cabin_type"]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in label_encode_cols:
    le = LabelEncoder()
    label_encoders[col] = le
    df_all[col] = le.fit_transform(df_all[col])
df_all.head()


# In[ ]:


cat_cols = label_encode_cols


# In[ ]:


for col in cat_cols:
    df_all[col] = df_all[col].astype('category')


# In[ ]:


df_all.isnull().sum()


# In[ ]:


df_all["cabin_count"] = df_all["cabin_count"].fillna(1)


# ## One-Hot Encode for XGBoost
# 
# LGBM and CatBoost can use categorical columns as is, but XGBoost needs one-hot encoding. So the "_oh" ending dataframes created here are the same as the ones without the ending, but one-hot encoded.

# In[ ]:


df_all_oh = pd.get_dummies( df_all, columns = cat_cols )
df_all_oh.head()


# In[ ]:


df_all_oh.columns


# In[ ]:


df_train = df_all[df_all["train"] == 1]
df_test = df_all[df_all["train"] == 0]


# In[ ]:


df_train_oh = df_all_oh[df_all_oh["train"] == 1]
df_test_oh = df_all_oh[df_all_oh["train"] == 0]


# In[ ]:


df_train = df_train.drop("train", axis=1)
df_test = df_test.drop("train", axis=1)
df_train.head()


# In[ ]:


df_train_oh = df_train_oh.drop("train", axis=1)
df_test_oh = df_test_oh.drop("train", axis=1)
df_train_oh.head()


# In[ ]:


target = df_train["Survived"]
target.head()


# In[ ]:


df_train = df_train.drop("Survived", axis=1)
df_test = df_test.drop("Survived", axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_train_oh = df_train_oh.drop("Survived", axis=1)
df_test_oh = df_test_oh.drop("Survived", axis=1)


# In[ ]:


df_train_oh.head()


# ## Create Booster Classifiers
# 
# Here I create the boosting classifiers. The hyperparameters are not especially tuned since the point of this notebook is to explore LIME for feature explanations, not to optimize for fractions of accuracy.

# In[ ]:


import lightgbm as lgb

l_clf = lgb.LGBMClassifier(
                        num_leaves=1024,
                        learning_rate=0.01,
                        n_estimators=5000,
                        boosting_type="gbdt",
                        min_child_samples = 100,
                        verbosity = 0)


# In[ ]:


x_clf = xgb.XGBClassifier()


# In[ ]:


import catboost

c_clf = catboost.CatBoostClassifier()


# ## Train-Test Splits

# In[ ]:


from sklearn.model_selection import train_test_split

X = df_train
y = target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

X_oh = df_train_oh
X_train_oh, X_val_oh = train_test_split(X_oh, test_size=0.33, random_state=42)


# In[ ]:


df_train.dtypes


# In[ ]:


df_train_oh.dtypes


# ## Fit all boosters

# In[ ]:


#the if True parts are just to make it simpler to disable some algorithm.

if True:
    l_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        early_stopping_rounds=5,
        verbose=False
    )
    
if True:
    c_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=5,
        cat_features=cat_cols,
        verbose=False
    )

if True:
    x_clf.fit(
        X_train_oh, y_train,
        eval_set=[(X_val_oh, y_val)],
        early_stopping_rounds=5,
        verbose=False
    )


# ## Feature Importance from Classifier
# 
# Plot the overall feature importance given by the algorithms themselves:
# 

# In[ ]:


import matplotlib.pyplot as plt

def plot_feat_importance(clf, train):
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        features = train.columns

        feat_importances = pd.DataFrame()
        feat_importances["weight"] = importances
        feat_importances.index = features
        feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features.csv")
        feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features", color='#86bf91', figsize=(10, 8))
        # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"
        plt.savefig(f'feature-weights.png')
        plt.savefig(f'feature-weights.pdf')
        plt.show()


# Plot permutation importance:

# In[ ]:



def plot_pimp(pimps, train):
    importances = pimps.importances_mean
    features = train.columns

    feat_importances = pd.DataFrame()
    feat_importances["weight"] = importances
    feat_importances.index = features
    feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features.csv")
    feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features", color='#86bf91', figsize=(10, 8))
    # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"
    plt.savefig(f'feature-weights.png')
    plt.savefig(f'feature-weights.pdf')
    plt.show()


# ### LGBM

# In[ ]:


plot_feat_importance(l_clf, X_train)


# In[ ]:


from sklearn.inspection import permutation_importance

l_pimps = permutation_importance(l_clf, X_train, y_train, n_repeats=10, random_state=0)
dir(l_pimps)


# In[ ]:


plot_pimp(l_pimps, X_train)


# ### Catboost

# In[ ]:


plot_feat_importance(c_clf, X_train)


# In[ ]:


c_pimps = permutation_importance(c_clf, X_train, y_train, n_repeats=10, random_state=0)
plot_pimp(c_pimps, X_train)


# In[ ]:


c_pimps.importances_mean


# In[ ]:


X_train.columns[np.argmin(c_pimps.importances_mean)]


# Interesting, permutation importance actually ranks "Parch" as contributing negatively to prediction accuracy.

# ### XGBoost

# In[ ]:


plot_feat_importance(x_clf, X_train_oh)


# In[ ]:


x_pimps = permutation_importance(x_clf, X_train_oh, y_train, n_repeats=10, random_state=0)
plot_pimp(x_pimps, X_train_oh)


# All three seem to rank very similar features highest overall. Title, gender, passenger class, fare, ... I expect fare to be a kind of a proxy for passenger class, which likely defines something about your location on the ship, and so on.
# 
# XGBoost here has much more variables that LGBM or CatBoost. Since XGBoost uses one-hot encoded variables, it has a much larger number of "features", which are just different values of a categorical variable. However, it also shows to rank a specific gender and title higher, so overall its is very similar.
# 
# The less weighted variables seem to have much more variation across the classifiers, but as far as I could tell including them still provides some small gains in accuracy. Just that the specific ways they are combined by the different classifiers has some variation, although results are very similar.
# 
# The difference between the lower ranked features is something that also comes up later with my LIME experiments.
# 

# ## LGBM predictions

# In[ ]:


from sklearn.metrics import accuracy_score, log_loss

val_pred_proba = l_clf.predict_proba(X_val)
#val_pred = np.array(val_pred[:, 1] > 0.5)
val_pred = np.where(val_pred_proba > 0.5, 1, 0)

acc_score = accuracy_score(y_val, val_pred[:,1])
acc_score


# ## CatBoost predictions

# In[ ]:


val_pred_proba = c_clf.predict_proba(X_val)
#val_pred = np.array(val_pred[:, 1] > 0.5)
val_pred = np.where(val_pred_proba > 0.5, 1, 0)

acc_score = accuracy_score(y_val, val_pred[:,1])
acc_score


# ## XGBoost predictions

# In[ ]:


val_pred_proba = x_clf.predict_proba(X_val_oh)
#val_pred = np.array(val_pred[:, 1] > 0.5)
val_pred = np.where(val_pred_proba > 0.5, 1, 0)

acc_score = accuracy_score(y_val, val_pred[:,1])
acc_score


# # Setting up LIME parameters
# 
# Running the LIME explainer requires 
# - a list of features
# - a prediction function
# - a function to translate data from LIME format to prediction algorithm format
# - list of categorical variable names
# - list of categorical variable indices
# - list of value names for categorical variables
# 
# The categorical variable data are used to define how LIME will permutate variables (categorical vs continous), and how it will display the results (variables and values have names vs numbers).

# In[ ]:


feature_names = list(df_train.columns)
feature_names


# In[ ]:


#cat_cols was set up earliner in the notebook to contain list of categorical feature/column names
cat_cols


# In[ ]:


#the corresponding indices of the cat_cols columns in the list of features
cat_indices = [feature_names.index(col) for col in cat_cols]
cat_indices


# In[ ]:


#mapping the category values to their names. for example, {"sex"={0="female", 1="male"}}
cat_names = {}
for label_idx in cat_indices:
    label = feature_names[label_idx]
    print(label)
    le = label_encoders[label]
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    le_value_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print(le_value_mapping)
    cat_names[label_idx] = le_value_mapping


# In[ ]:


cat_names #its actually the feature index mapped to the values and their names


# In[ ]:


explainer = lime.lime_tabular.LimeTabularExplainer(df_train.values, discretize_continuous=True,
                                                   class_names=['not survived', 'survived'], 
                                                   mode="classification",
                                                   feature_names = feature_names,
                                                   categorical_features=cat_indices,
                                                   categorical_names=cat_names, 
                                                   kernel_width=10, verbose=True)


# ## Functions to run Classifiers from LIME
# 
# LIME takes a datapoint, generates N (by default N=5000) synthetic samples around it, and runs the classifier on those synthetic samples to get some estimate on the effects of features. In many classifiers, LIME uses different format for the dataframes than the actual classifier you give it, so have to write a function to convert them to classifier format. The following are functions to do that for LGBM, CatBoost, XGBoost.

# In[ ]:


#def row_to_df(row):
#    rows = []
#    rows.append(X_val.values[0])
#    df = pd.DataFrame(rows, columns=X_val.columns)
#    for col in cat_cols:
#        df[col] = df[col].astype('category')    
#    return df

#when LIME passes synthetic data to your predict function, it gives a list of N synthetic datapoints as a numpy matrix
#this converts that matrix into a dataframe, since some algorithms choke on the pure numpy array (catboost)
def rows_to_df(rows):
    df = pd.DataFrame(rows, columns=X_val.columns)
    #set category columns first to short numeric to save memory etc, then convert to categorical for catboost
    for col in cat_cols:
        df[col] = df[col].astype('int8')
        df[col] = df[col].astype('category')
    #and finally convert all non-categoricals to their original type. since we had to create a fresh dataframe this is needed
    for col in X_val.columns:
        if col not in cat_cols:
            df[col] = df[col].astype(X_val[col].dtype)
    return df

#for one-hot encoding the values from LIME, which uses numbers in a single column to represent categories
#needed for xgboost
def rows_to_df_oh(rows):
    df = pd.DataFrame(rows, columns=X_val_oh.columns)
    for col in cat_cols:
        df[col] = df[col].astype('int8')
    return df

#the function to pass to LIME for running catboost on the synthetic data
def c_run_pred(x):
    p = c_clf.predict_proba(rows_to_df(x))
    return p

#the function to pass to LIME to run LGBM on the synthetic data
def l_run_pred(x):
    p = l_clf.predict_proba(x)
    return p

#the function to pass to LIME to run XGBoost on the synthetic data
def x_run_pred(x):
    df = rows_to_df(x)
    df = pd.get_dummies( df, columns = cat_cols )

    new_df = pd.DataFrame()
    #this look ensure the column order of the dataframe created is same as the original
    for col in X_val_oh.columns:
        if col in df.columns:
            new_df[col] = df[col]
        else:
            #sometimes it seems to happen that a specific value is missing from a category in generation,
            #which leads to missing that column. this zeroes it to ensure it exists
            #print(f"missed col:{col}")
            new_df[col] = 0
    df = new_df

    p = x_clf.predict_proba(df)
    return p

c_predict_fn = lambda x: c_run_pred(x)

l_predict_fn = lambda x: l_run_pred(x)

x_predict_fn = lambda x: x_run_pred(x)


# In[ ]:


l_clf.predict_proba([X_val.values[0]])


# In[ ]:


c_predict_fn(X_val.values)[0]


# In[ ]:


x_predict_fn(X_val.values)[0]


# In[ ]:



x_clf.predict_proba(X_val_oh)[0]


# In[ ]:


#this demonstrates the missing value branch of x_run_pred() with cabin_type=7
df = rows_to_df(X_val.values)
print(df.shape)
print(f"cat cols: {cat_cols}")
df = pd.get_dummies( df, columns = cat_cols )
missing_cols = set( X_val_oh.columns ) - set( df.columns )
print(f"missing: {missing_cols}")
# Add a missing column in test set with default value equal to 0
new_df = pd.DataFrame()
for col in X_val_oh.columns:
    if col in df.columns:
        new_df[col] = df[col]
    else:
        new_df[col] = 0
df = new_df


# In[ ]:


#p = x_clf.predict_proba(df)
#p


# In[ ]:


#x_predict_fn(X_val.values)


# # Explaining a Datapoint with LIME / Booster Classifiers

# In[ ]:


def explain_item(predictor, item):
    exp = explainer.explain_instance(item, predictor, num_features=10, top_labels=1)
    exp.show_in_notebook(show_table=True, show_all=False)


# ## Point 1 explained
# 
# I will run the explainer first once per datapoint. Starting with point 1, or index 0 in the dataset. And then modify one of the higher ranked features to see if it has some effect.

# In[ ]:


#this allows running the experiments N times to see if the random synthetic value generation of LIME has some effect on the results over different runs.

def explain_x_times(x, idx, invert_gender=False):
    row = X_val.values[idx]
    if invert_gender:
        if row[1] > 0:
            row[1] = 0
        else:
            row[1] = 1
    print(f"columns={X_val.columns}")
    
    for i in range(x):
        print(f"Explaining LGBM: index={idx}, row={row}")
        explain_item(l_predict_fn, row)
    for i in range(x):
        print(f"Explaining CatBoost: index={idx}, row={row}")
        explain_item(c_predict_fn, row)
    for i in range(x):
        print(f"Explaining XGBoost: index={idx}, row={row}")
        explain_item(x_predict_fn, row)


# ### Explain Point 1 with all 3 Boosters

# In[ ]:


explain_x_times(2, 0)


# ### Invert Gender for Point 1, Classify and Explain Again

# In[ ]:


explain_x_times(2, 0, invert_gender=True)


# ## Explain Point 2 with All Boosters

# In[ ]:


explain_x_times(2, 1, invert_gender=False)


# ### Invert Gender for Point 2, Classify and Explain Again

# In[ ]:


explain_x_times(2, 1, invert_gender=True)


# In[ ]:





# # Regression
# 
# The above was an experiment on explaining a classification model with LIME. What about tabular data in regression models? In such case we try to predict a continous variable based on a set of features, as opposed to trying to classify someting to a specific category.
# 
# This one uses the Boston housing prices dataset.

# In[ ]:


df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


df_train.head()


# As with the Titanic dataset, I will just use a bunch of derived features copied from some existing Kaggle notebooks. Thanks again :)

# In[ ]:


for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    df_train[col]=df_train[col].fillna('None')
    df_test[col]=df_test[col].fillna('None')

for col in ('Electrical','MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    df_train[col]=df_train[col].fillna(df_train[col].mode()[0])
    df_test[col]=df_test[col].fillna(df_train[col].mode()[0])

for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
            'GarageYrBlt','GarageCars','GarageArea'):
    df_train[col]=df_train[col].fillna(0)
    df_test[col]=df_test[col].fillna(0)

df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_train['LotFrontage'].mean())


# In[ ]:


df_train.dtypes


# In[ ]:


#removing outliers recomended by author
df_train = df_train[df_train['GrLivArea']<4000]


# In[ ]:


len_traindf = df_train.shape[0]
houses = pd.concat([df_train, df_test], sort=False)
houses = houses.fillna(0)


# In[ ]:



# turning some ordered categorical variables into ordered numerical
# maybe this information about order can help on performance
for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
            "FireplaceQu","GarageQual","GarageCond","PoolQC"]:
    houses[col]= houses[col].map({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1})
houses = houses.fillna(0)


# As before, need the list of categorical feature names and indices as input for LIME explainer:

# In[ ]:


import numbers

cdf = df_train.select_dtypes(include=np.number)
cat_names = [key for key in df_train.columns if key not in cdf.columns]
cat_names


# In[ ]:


len_traindf = df_train.shape[0]

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in cat_names:
    le = LabelEncoder()
    label_encoders[col] = le
    houses[col] = le.fit_transform(houses[col])


# Since I am using XGBoostRegressor, I need the one-hot encoded categorical variables again, just like with the Titanic classifier above:

# In[ ]:



df_train = houses[:len_traindf]
df_train = df_train.drop('SalePrice', axis=1)
df_test = houses[len_traindf:]
df_test = df_test.drop('SalePrice', axis=1)

# turning categoric into numeric
houses_oh = pd.get_dummies(houses)

# separating
df_train_oh = houses_oh[:len_traindf]
df_test_oh = houses_oh[len_traindf:]


# In[ ]:


# x/y split
X_train_oh = df_train_oh.drop('SalePrice', axis=1)
y_train = df_train_oh['SalePrice']
X_test_oh = df_test_oh.drop('SalePrice', axis=1)


# A bit of an overkill to optimize the hyperparameters with multiple runs over hyperopt, but the notebook I used as a source does it, so here we go:

# In[ ]:


from hyperopt import hp, tpe, fmin

space = {'n_estimators':hp.quniform('n_estimators', 1000, 4000, 100),
         'gamma':hp.uniform('gamma', 0.01, 0.05),
         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.03),
         'max_depth':hp.quniform('max_depth', 3,7,1),
         'subsample':hp.uniform('subsample', 0.60, 0.95),
         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.95),
         'colsample_bylevel':hp.uniform('colsample_bylevel', 0.60, 0.95),
         'reg_lambda': hp.uniform('reg_lambda', 1, 20)
        }

def objective(params):
    params = {'n_estimators': int(params['n_estimators']),
             'gamma': params['gamma'],
             'learning_rate': params['learning_rate'],
             'max_depth': int(params['max_depth']),
             'subsample': params['subsample'],
             'colsample_bytree': params['colsample_bytree'],
             'colsample_bylevel': params['colsample_bylevel'],
             'reg_lambda': params['reg_lambda']}
    
    xb_a = xgb.XGBRegressor(**params)
    score = cross_val_score(xb_a, X_train_oh, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()
    return -score


# In[ ]:


best = fmin(fn= objective, space= space, max_evals=4, rstate=np.random.RandomState(1), algo=tpe.suggest)
#max_evals=20


# This will be the actual regressor used and explained:

# In[ ]:


X_clf = xgb.XGBRegressor(random_state=0,
                        n_estimators=int(best['n_estimators']), 
                        colsample_bytree= best['colsample_bytree'],
                        gamma= best['gamma'],
                        learning_rate= best['learning_rate'],
                        max_depth= int(best['max_depth']),
                        subsample= best['subsample'],
                        colsample_bylevel= best['colsample_bylevel'],
                        reg_lambda= best['reg_lambda']
                       )

X_clf.fit(X_train_oh, y_train)


# And build a list of categorical column indices for LIME:

# In[ ]:


all_cols = list(df_train.columns)
cat_indices = []
for cat_name in cat_names:
    cat_indices.append(all_cols.index(cat_name))
print(cat_indices)


# In[ ]:





# In[ ]:


houses.head()


# List of all feature names for LIME:

# In[ ]:


feature_names = list(df_train.columns)
print(feature_names)


# Mapping of categorical feature names to their values and their names. Similar to the Titanic dataset above:

# In[ ]:


cat_names = {}
for label_idx in cat_indices:
    label = feature_names[label_idx]
    print(label)
    le = label_encoders[label]
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    le_value_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print(le_value_mapping)
    cat_names[label_idx] = le_value_mapping


# In[ ]:


feature_names_oh = list(df_train_oh.columns)


# In[ ]:


explainer = lime.lime_tabular.LimeTabularExplainer(df_train.values, 
                                                   feature_names=feature_names, 
                                                   class_names=['price'], 
                                                   categorical_features=cat_indices,
                                                   categorical_names=cat_names,
                                                   verbose=True, 
                                                   discretize_continuous=False,
                                                   mode='regression')


# In[ ]:


def explain_xreg(row):
    df = pd.DataFrame(data=row, columns=df_train.columns)
    row = pd.get_dummies(df)
    return X_clf.predict(row)


# In[ ]:


def explain_item(item):
    exp = explainer.explain_instance(item, explain_xreg, num_features=10, top_labels=1)
    exp.show_in_notebook(show_table=True, show_all=False)
    return exp


# In[ ]:


df_test.iloc[0]


# In[ ]:


df_test.iloc[0].values.shape


# In[ ]:


explain_xreg([df_test.iloc[0].values])


# In[ ]:


exp = explain_item(df_test.iloc[0])


# In[ ]:





# In[ ]:


df_test.head()


# In[ ]:


df_test.describe()


# In[ ]:


top_features = exp.as_list()
top_features


# Because the explainer was created above using the top 10 features, the list also shows these 10 features. A brief look at each in this instance vs the overall dataset:

# In[ ]:


for feat, weight in top_features:
    print(feat)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
top_names = [tup[0].split("=")[0] for tup in top_features]
df_test[top_names].hist(figsize=(15,10))


# In[ ]:


df_test[top_names].head(1)


# In[ ]:


y_train.describe()


# In[ ]:


df_test.iloc[0]["KitchenQual"]


# In[ ]:


row = df_test.iloc[0]
row["KitchenQual"] = 2


# In[ ]:


row["KitchenQual"]


# In[ ]:


exp = explain_item(row)


# # Explaining Keras NN Classifier

# Above I tried to explain booster based classifiers and regressors using the scikit interfaces. Neural nets are another type of network often used, so how to fit LIME on a neural net? Here is an example of a classifier based on Keras fully connected neural network.
# 
# This one uses the UCI heart disease dataset. As usual, the base preprocessing and model are based on some existing notebooks. Thanks for your efforts, whoever it was.. My point is simply to use it to try out LIME.

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


# read the csv
cleveland = pd.read_csv('../input/heart-disease-uci/heart.csv')


# In[ ]:


# remove missing data (indicated with a "?")
data = cleveland[~cleveland.isin(['?'])]


# In[ ]:


#drop nans
data = data.dropna(axis=0)


# In[ ]:


data = data.apply(pd.to_numeric)
data.dtypes


# In[ ]:


X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])


# In[ ]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# In[ ]:


# convert the data to categorical labels
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())


# In[ ]:


# fit the model to the training data
#verbose=1 for full output, verbose=2 for list of epochs. 0 for quiet
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=50, batch_size=10, verbose=0)


# In[ ]:


#to see training results, exact accuracy and loss
#history.history


# Visualizing the accuracy and loss over epochs takes much less space..

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# convert into binary classification problem - heart disease or no heart disease
Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# The kernel I am basing on finished with a binary model simpler analysis of output. So here we go.

# In[ ]:


# define a new keras model for binary classification
def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

binary_model = KerasClassifier(build_fn=create_binary_model, epochs=50, batch_size=10, verbose=0)


# In[ ]:


# fit the binary model on the training data
history=binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=50, batch_size=10, verbose=0)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:



def plot_pimp_2(pimps, features):
    importances = pimps.importances_mean

    feat_importances = pd.DataFrame()
    feat_importances["weight"] = importances
    feat_importances.index = features
    feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features.csv")
    feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features", color='#86bf91', figsize=(10, 8))
    # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"
    plt.savefig(f'feature-weights.png')
    plt.savefig(f'feature-weights.pdf')
    plt.show()


# In[ ]:


cleveland.shape


# In[ ]:


df_X = data.drop(['target'], 1)
k_pimps = permutation_importance(binary_model, X_train, y_train, n_repeats=10, random_state=0)
plot_pimp_2(k_pimps, df_X.columns)


# In[ ]:


# generate classification report using predictions for categorical model
from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model.predict(X_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))


# In[ ]:


#model.predict_proba(X_test)


# In[ ]:


feature_names = cleveland.columns


# Now to recall the variables before we try to explain some datapoints from this predictor:

# - age
# - sex
# - chest pain type (4 values)
# - resting blood pressure
# - serum cholestoral in mg/dl
# - fasting blood sugar > 120 mg/dl
# - resting electrocardiographic results (values 0,1,2)
# - maximum heart rate achieved
# - exercise induced angina
# - oldpeak = ST depression induced by exercise relative to rest
# - the slope of the peak exercise ST segment
# - number of major vessels (0-3) colored by flourosopy
# - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[ ]:


data.nunique()


# In[ ]:


#data["thalach"].describe()


# Categorical columns and number of values / categories for each:

# In[ ]:


cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "ca", "thal"]
for col in cat_cols:
    print(f"{col}: {data[col].unique()}")


# Categorical column indices in the list of columns, needed for LIME to print the names on explanations:

# In[ ]:


feature_names = list(feature_names)
cat_indices = [feature_names.index(col) for col in cat_cols]
cat_indices


# In[ ]:


#explainer = lime.lime_tabular.LimeTabularExplainer(df_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=True,
                                                   class_names=['no risk', 'risk of heart'], 
                                                   mode="classification",
                                                   feature_names = feature_names,
                                                   categorical_features=cat_indices,
                                                   categorical_names=[], 
                                                   kernel_width=10, verbose=True)


# In[ ]:


def explain_item(item):

    #    exp = explainer.explain_instance(item, l_predict_fn, num_features=10, top_labels=1)
    exp = explainer.explain_instance(item, model.predict_proba, num_features=10, top_labels=1)
#    exp = explainer.explain_instance(item, l_clf.predict_proba, num_features=10, top_labels=1)
    exp.show_in_notebook(show_table=True, show_all=False)


# In[ ]:


def explain_item_flipped(row, flip=False):
    if flip:
        if row[2] > 0:
            row[2] = 0
        else:
            row[2] = 1
    print(f"columns={X_val.columns}")
    #    exp = explainer.explain_instance(item, l_predict_fn, num_features=10, top_labels=1)
    exp = explainer.explain_instance(row, model.predict_proba, num_features=10, top_labels=1)
#    exp = explainer.explain_instance(item, l_clf.predict_proba, num_features=10, top_labels=1)
    exp.show_in_notebook(show_table=True, show_all=False)


# In[ ]:


cat_cols


# In[ ]:


cat_indices


# In[ ]:


explain_item(X_test[0])


# In[ ]:


explain_item_flipped(X_test[0], False)


# In[ ]:


explain_item_flipped(X_test[1], False)


# In[ ]:


explain_item_flipped(X_test[0], True)


# In[ ]:


explain_item_flipped(X_test[1], True)


# So the above shows an example of application of the explainer, but with very conflicting results. Most variables / features are shown to highly bias the predition towards a risk of heart-disease. Yet the prediction by the classifier is the exact opposite. So there are a few options here:
# 
# - Maybe I am doing something wrong in using LIME. But honestly, whatever I check, I seem to be using it exact as all others. 
# - Maybe LIME sometimes doesn't work so good? But when? How can you rely on it to explain anything if it is that random?
# - Maybe I don't know how to correctly interpret the output? Like should the weights be further weighted by something else? It does not seem to make sense if so, so I guess this is not the case..
# - Sometimes, like here, the explanations vs the predictions seem to be exact opposite of what they should be. But why would it be like that? Can one make such an error? 
# 
# 
# 
# 
# 
# 
# 

# # Conclusions
# 
# Whatever the reason, more critical evaluations on larger scale attempts at using the LIME explainers, and information on why the results sometimes seem so different would be great. 
# 
# That's all for LIME for now. I think I should try SHAP next, since it seems better documented, and more refined in its usage API. But this was an interesting look. Even if the explainers with LIME are not perfect, they can give some good ideas on how to move forward with buildling such explainers, how to present the results to users (I like the weight bars for variables etc).

# In[ ]:




