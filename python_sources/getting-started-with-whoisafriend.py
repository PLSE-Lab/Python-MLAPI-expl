#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = [9, 12]

import warnings
warnings.simplefilter('ignore')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold


# In[ ]:


# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb


# ### Importing Data

# In[ ]:


train = pd.read_csv("/kaggle/input/whoisafriend/train.csv")
test = pd.read_csv("/kaggle/input/whoisafriend/test.csv")
sub = pd.read_csv("/kaggle/input/whoisafriend/sample_submission.csv")

train.shape, test.shape, sub.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:





# ### Understanding the data and Making Hypothesis : 
# 
# - H1. Is there a relation between persons in Train and Test?
# - H2. Does Moon Phase matter / important in being a friend? 
# - H3. Distribution of Years of Knowing and Interaction Duration between friends and non-friends.

# ### H1 :  Is there a relation between persons in Train and Test?

# In[ ]:


# First of all are the persons in Person_A and Person_B from same set or are they from the different set.
train[['Person A', 'Person B']].describe()


# #### It seems there are 100 unique names both in Person A and Person B. 
# #### Now let's check whether they are same or different.
# 
# #### Lets check the count of Person B who are not in Person A list. 
# #### If this is equal to 100 then it says that there is no interaction between both the sets.

# In[ ]:


len(set(train['Person A'].unique()) - set(train['Person B'].unique()))


# In[ ]:


len(set(train['Person B'].unique()) - set(train['Person A'].unique()))


# #### So that clarifies that they both are from different sets. But how about difference between train and test.

# In[ ]:


train_persons_list = train['Person A'].unique().tolist() + train['Person B'].unique().tolist()
test_persons_list = test['Person A'].unique().tolist() + test['Person B'].unique().tolist()


# In[ ]:


len(set(train_persons_list) - set(test_persons_list)), len(set(test_persons_list) - set(train_persons_list))


# #### No Intersection, that means no persons are common in train and test sets.
# 
# So is the feature Person A and Person B not important? Absolutely not. They might be not that important indiviudually but think if we combine both these features so this feature now implies a combination of persons and consider if a model predicts at one record that these two are friends so they musnt be.

# In[ ]:


train['Persons_Name_combined'] = train['Person A'] + train['Person B']
test['Persons_Name_combined'] = test['Person A'] + test['Person B']


# ### H2.  Does Moon Phase matter / important in being a friend? 

# In[ ]:


plt.figure(figsize=(15, 4))
sns.countplot(train['Moon Phase During Interaction'], hue=train['Friends'])
plt.show()


# #### By seeing this plot it doesn't seem Moon Phase seriously affects the Friendship, but let the model verify it.

# ### H3. Distribution of Years of Knowing and Interaction Duration between friends and non-friends.

# In[ ]:


sns.lmplot('Years of Knowing', 'Interaction Duration', data=train, hue='Friends', size=15)


# #### Oh look what we have here. It seems there is a good relationship between "Years_Known" and "Interaction_Duration" for being Friends.
# #### The two lines represent the fitted regression line. Seaborn does is defaultly in ScatterPlots. They also state they have a good relationship and might prove good features.

# In[ ]:


train['mult_years_dur'] = train['Years of Knowing'] + train['Interaction Duration']
test['mult_years_dur'] = test['Years of Knowing'] + test['Interaction Duration']


# ### Data Preprocessing

# #### Missing Values

# In[ ]:


def missing_values(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    return mis_val_table_ren_columns


# In[ ]:


miss_train, miss_test = missing_values(train), missing_values(test)


# #### Label Encoding 

# In[ ]:


cat_cols = ['Person A', 'Person B', 'Interaction Type', 'Moon Phase During Interaction', 'Persons_Name_combined']
target = 'Friends'


# In[ ]:


# Making a dictionary to store all the labelencoders for categroical columns to transform them later.
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col].unique().tolist() + test[col].unique().tolist())
    train[col] = le.transform(train[[col]])
    test[col] = le.transform(test[[col]])
    
    le_dict[col] = le


# ### Baselining Models

# In[ ]:


def baseliner(train, features, target, cv=3, metric='accuracy'):
    """
    Function for baselining Models which return CV Score, Train Score, Valid Score
    """
    print("Baseliner Models\n")
    eval_dict = {}
    models = [lgb.LGBMClassifier(), xgb.XGBClassifier(), GradientBoostingClassifier(), LogisticRegression(), 
              RandomForestClassifier(), DecisionTreeClassifier(), AdaBoostClassifier()
             ]
    print("Model Name \t |   CV") #    | \t TRN   | \t  VAL
    print("--" * 50)

    for index, model in enumerate(models, 0):
        model_name = str(model).split("(")[0]
        eval_dict[model_name] = {}

        results = cross_val_score(model, train[features], train[target], cv=cv, scoring=metric)
        eval_dict[model_name]['cv'] = results.mean()

        print("%s \t | %.4f \t" % (
            model_name[:12], eval_dict[model_name]['cv']))


# In[ ]:


target = 'Friends'
id_col = 'ID'

feat = train.columns.tolist()
feat.remove(target)
feat.remove(id_col)

print("Length of Features : {}".format(len(feat)))


# In[ ]:


baseliner(train, feat, target)


# ### Basic LightGBM Modelling

# In[ ]:


def splitter(train, features, target, ts=False):
    if ts:
        trainX, validX, trainY, validY = train_test_split(train[features],
                                                          train[target], test_size=0.2,
                                                          random_state=13, shuffle=False)
    else:
        trainX, validX, trainY, validY = train_test_split(train[features],
                                                      train[target], test_size=0.2,
                                                      random_state=13)
    return trainX, validX, trainY, validY

def lgb_model(train, test, features, target, ts=False):
    evals_result = {}
    trainX, validX, trainY, validY = splitter(train, features, target, ts=ts)
    print("LGB Model")
    lgb_train_set = lgb.Dataset(trainX, label=trainY)
    lgb_valid_set = lgb.Dataset(validX, label=validY)

    MAX_ROUNDS = 2000
    lgb_params = {
        "boosting": 'gbdt',  # "dart",
        "learning_rate": 0.01,
        "nthread": -1,
        "seed": 13,
        "num_boost_round": MAX_ROUNDS,
        "objective": "binary",
        "metric": "binary_error",
    }

    lgb_model = lgb.train(
        lgb_params,
        train_set=lgb_train_set,
        valid_sets=[lgb_train_set, lgb_valid_set],
        early_stopping_rounds=50,
        verbose_eval=100,
        evals_result=evals_result,
    )

    lgb.plot_importance(lgb_model, figsize=(24, 24))
    lgb.plot_metric(evals_result, metric='binary_error')
    
    preds = lgb_model.predict(test[feat])
    return lgb_model, preds


# In[ ]:


lgbM, lgb_preds = lgb_model(train, test, feat, target, ts=False)


# In[ ]:


threshold = 0.5
lgb_preds[lgb_preds < threshold] = 0
lgb_preds[lgb_preds >= threshold] = 1

sub['lgb'] = lgb_preds
sub['lgb'] = sub['lgb'].astype(np.int)
sub.head()


# ### Distribution of the prediction probabilities

# In[ ]:


plt.figure(figsize=(12, 5))
sns.distplot(lgb_preds)
plt.title("Distritbution of predictions")
plt.show()


# ### Saving the submission

# In[ ]:


sub[['ID', 'lgb']].rename({
    "lgb": "Friends"
}, axis=1).to_csv("lgb_submission.csv", index=False)


# In[ ]:




