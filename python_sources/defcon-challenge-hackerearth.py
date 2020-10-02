#!/usr/bin/env python
# coding: utf-8

# import the necessary libraries

# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,     classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import itertools
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sn
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, ParameterGrid
import time
from sklearn.metrics import make_scorer, accuracy_score
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,     ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib


# Read data from challenge (train and test dataset)

# In[ ]:


data_train = pd.read_csv('../input/defcon-dataset/train.csv')
data_test = pd.read_csv('../input/defcon-dataset/train.csv')
print (data_train.head(5))


# Check the number of labels on the train set.

# In[ ]:


print(data_train.DEFCON_Level.value_counts())


# Concat train and test. Drop "ID" column and split train data

# In[ ]:


# concat train and test
data = pd.concat([data_train, data_test], axis=0)
# drop 'id' column
data = data.drop(columns=['ID'])
# Retrieve data from the train set
X = data[:10000].drop(columns=['DEFCON_Level'])
# Retrieve label from train set
y = data[:10000].DEFCON_Level
# Retrieve data from test set
X_test = data[10000:].drop(columns=['DEFCON_Level'])
# split train file into train set (0.75) and valid set (0.25). The valid set is used to evaluate the model on the train set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)


# Visualize data to identify important features

# In[ ]:


def feature_importances():
    for c in data.columns[data.dtypes == 'object']:
        X_train[c] = X_train[c].factorize()[0]
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    plt.plot(rf.feature_importances_)
    plt.xticks(np.arange(X_train.shape[1]), X_train.columns.tolist(), rotation=90)
    plt.show()
    
feature_importances()


# Visualize correlation matrix, this stage is very important. A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. A correlation matrix is used to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analyses.

# In[ ]:


corr = X.corr()
corr.style.background_gradient(cmap='coolwarm')


# We will preprocess two important fields in the data: Closest_Threat_Distance, Troops_Mobilized

# In[ ]:


def feature_extract():
    X.Closest_Threat_Distance = X.Closest_Threat_Distance / 1.06
    X.Troops_Mobilized = X.Troops_Mobilized / 100
    values = X.Troops_Mobilized.values.tolist()
    for idx in range(len(values)):
        if ".3" in str(values[idx]):
            values[idx] = round(values[idx] * 3) / 44
        elif ".6" in str(values[idx]):
            values[idx] = round(values[idx] * 3 / 2) / 44
    X.Troops_Mobilized = values
    X.Percent_Of_Forces_Mobilized = X.Percent_Of_Forces_Mobilized / 0.01
    X.Closest_Threat_Distance = X.Closest_Threat_Distance / 1.06
feature_extract()


# Compare tree based models to select good model models for this data set.
# The models I use below are : SVC, DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, MLPClassifier, KNeighborsClassifier, LogisticRegression, LinearDiscriminantAnalysis

# In[ ]:


def compare_tree_based_models():
    random_state = 2
    classifiers = [SVC(random_state=random_state), DecisionTreeClassifier(random_state=random_state),
                   AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state,
                                      learning_rate=0.1), RandomForestClassifier(random_state=random_state),
                   ExtraTreesClassifier(random_state=random_state),
                   GradientBoostingClassifier(random_state=random_state), MLPClassifier(random_state=random_state),
                   KNeighborsClassifier(), LogisticRegression(random_state=random_state), LinearDiscriminantAnalysis()]

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, X_train, y=y_train, scoring="accuracy", cv=4, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame(
        {"CrossValMeans": cv_means, "CrossValerrors": cv_std, "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                                                            "RandomForest", "ExtraTrees",
                                                                            "GradientBoosting",
                                                                            "MultipleLayerPerceptron", "KNeighboors",
                                                                            "LogisticRegression",
                                                                            "LinearDiscriminantAnalysis"]})

    g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()

compare_tree_based_models()


# I will use the decision tree, which model I am most confident in, I will use a combination of Kfold, GridSearchCV and BaggingClassifier to choose the best parameters for this data.

# In[ ]:


scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()
# params for decision tree classifier
dt_param = {"base_estimator__criterion": ["gini", "entropy"],
            "base_estimator__max_depth": list(range(2, 32, 1)),
            'base_estimator__random_state': [21],
            'base_estimator__max_features': [10],
            "base_estimator__min_samples_leaf": list(range(5, 20, 5)),
            'base_estimator__min_samples_split': list(range(5, 20, 5)),
            'base_estimator__min_impurity_decrease': [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01,
                                                      0.05, 0.1]}
max_score = 0
# use Kfold
sss = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)
for train_index, val_index in sss.split(X, y):
    start_time = time.time()
    print("Train:", train_index, "Val:", val_index)
    X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[val_index]
    # standardized data
    X_train = scaler_standard.fit_transform(X_train)
    X_valid = scaler_standard.fit_transform(X_valid)
    # Use BaggingClassifier 
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True, random_state=42)
    # Use GridSearchCV
    model = GridSearchCV(estimator=model, param_grid=dt_param, cv=4, verbose=3)
    model.fit(X_train, y_train)
    print(model.best_score_, model.best_params_)
    # # tree best estimator
    model = model.best_estimator_
    # Save the best params after training
    joblib.dump(model, 'DecisionTreeClassifier.pkl')
    # Evaluate the model with a valid set
    tree_score = cross_val_score(model, X_valid, y_valid, cv=4)
    print('Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')
    print("time execution :", time.time() - start_time)


# After the training, we have found the best params just reload the model and create the sample_submission.csv.

# In[ ]:


def write_submission(md):
    sample_submission = pd.DataFrame()
    sample_submission['ID'] = data_test.ID
    sample_submission['DEFCON_Level'] = md.predict(StandardScaler().fit_transform(X_test))
    sample_submission.to_csv('sample_submission.csv', index=False)
    print('write sample_submission completed')
    
# Load model with best params
model = joblib.load('DecisionTreeClassifier.pkl')
write_submission(model)


# ![LeaderBoard](http://https://imgur.com/a/M9lbxIJ)
# For now, just using the best Decision and params, I'm standing # 5, you can stuff other algorithms, the contest is still going till March 20.
