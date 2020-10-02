####################################
#  Libraries
####################################

import numpy as np 
import pandas as pd 
# Stats
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
# Data processing, metrics and modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import lightgbm as lgbm

####################################
# Importing data and merging
####################################

# Reading dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Adding a column in each dataset before merging
train['Type'] = 'train'
test['Type'] = 'test'

# Merging train and test
data = train.append(test)

####################################
# Missing values and new features
####################################

# Title
data['Title'] = data['Name']

# Cleaning name and extracting Title
for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
    
# Replacing rare titles 
mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 
           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 
           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 
           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}
           
data.replace({'Title': mapping}, inplace=True)
titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']

# Replacing missing age by median/title 
for title in titles:
    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute
    
# New feature : Family_size
data['Family_Size'] = data['Parch'] + data['SibSp'] + 1
data.loc[:,'FsizeD'] = 'Alone'
data.loc[(data['Family_Size'] > 1),'FsizeD'] = 'Small'
data.loc[(data['Family_Size'] > 4),'FsizeD'] = 'Big'

# Replacing missing Fare by median/Pclass 
fa = data[data["Pclass"] == 3]
data['Fare'].fillna(fa['Fare'].median(), inplace = True)

#  New feature : Child
data.loc[:,'Child'] = 1
data.loc[(data['Age'] >= 18),'Child'] =0

# New feature : Family Survival (https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
DEFAULT_SURVIVAL_VALUE = 0.5

data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE
for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
                               
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                    
####################################
# Encoding and pre-modeling
####################################                  

# dropping useless features
data = data.drop(columns = ['Age','Cabin','Embarked','Name','Last_Name',
                            'Parch', 'SibSp','Ticket', 'Family_Size'])

# Encoding features
target_col = ["Survived"]
id_dataset = ["Type"]
cat_cols   = data.nunique()[data.nunique() < 12].keys().tolist()
cat_cols   = [x for x in cat_cols ]
# numerical columns
num_cols   = [x for x in data.columns if x not in cat_cols + target_col + id_dataset]
# Binary columns with 2 values
bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()
# Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]
# Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])
# Duplicating columns for multi value columns
data = pd.get_dummies(data = data,columns = multi_cols )
# Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(data[num_cols])
scaled = pd.DataFrame(scaled,columns = num_cols)
# dropping original values merging scaled values for numerical columns
df_data_og = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(scaled,left_index = True,right_index = True,how = "left")
data = data.drop(columns = ['PassengerId'],axis = 1)

# Cutting train and test
train = data[data['Type'] == 1].drop(columns = ['Type'])
test = data[data['Type'] == 0].drop(columns = ['Type'])

# X and Y
X = train.drop('Survived', 1)
y = train['Survived']
X_test = test.drop(columns = ['Survived'])

####################################
# Randomized Search & LGBM
####################################     

fit_params = {"early_stopping_rounds" : 100, 
             "eval_metric" : 'auc', 
             "eval_set" : [(X,y)],
             'eval_names': ['valid'],
             'verbose': 0,
             'categorical_feature': 'auto'}

param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
              'num_leaves': sp_randint(6, 50), 
              'min_child_samples': sp_randint(100, 500), 
              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'subsample': sp_uniform(loc = 0.2, scale = 0.8), 
              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
              'colsample_bytree': sp_uniform(loc = 0.4, scale = 0.6),
              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# Number of combinations
n_iter = 500 

# Intializing lgbm and lunching the search
lgbm_clf = lgbm.LGBMClassifier(random_state = 42, silent = True, metric = 'None', n_jobs = 4)
grid_search = RandomizedSearchCV(
    estimator = lgbm_clf, param_distributions = param_test, 
    n_iter = n_iter,
    scoring = 'accuracy',
    cv = 5,
    refit = True,
    random_state = 42,
    verbose = True)

grid_search.fit(X, y, **fit_params)
opt_parameters = grid_search.best_params_

# Predicting y_test
lgbm_clf = lgbm.LGBMClassifier(**opt_parameters)
y_pred = lgbm_clf.fit(X, y).predict(X_test)

####################################
# Submission
####################################  

temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("../working/submission.csv", index = False)