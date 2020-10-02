#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install impyute


# In[ ]:


import pandas as pd
from impyute.imputation.cs import mice
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
data = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


data


# ****DATA PREPROCESSING****

# **FEATURE ENGINEERING**
# 
# *      Separate titles into category: 
#                                  1 if [Mr]
#                                  2 if [Miss, Ms]
#                                  3 if [Mrs]
#                                  4 if [Master]
#                                  5 if miscellaneous e.g. Rev., Dr., etc.
#     * Convert sex: 1 - female, 0 - male
#     * Group 'Cabin' column into two groups: 0 - if known, 1 - if 'NaN'
#     * Separate'Embarked' column into 2 columns (drop 1 in any of the 3 to avoid collinearity)
#     * Add 'isAlone' feature if passenger is alone: 1 - if alone, 0 - if not
#     * add 'companion' feature containing no. of companions the passenger is aboard with ('Parch'+'SibSp')
#     * impute missing 'Age' data by MICE (Multiple Imputation by Chained Equations) ref: https://towardsdatascience.com/stop-using-mean-to-fill-missing-data-678c0d396e22

# In[ ]:


def process_name (data):
    data['Title'] = data['Name'].apply(lambda name: name.split(',')[1].split(' ')[1])
    data['Title'] = [float(1) if name in ['Mr.'] else float(2) if name in ['Miss.', 'Ms.'] else float(3) if name in ['Mrs.'] else float(4) if name in ['Master.'] else float(5) for name in data['Title']]
    data = data.drop(['Name'], axis = 1)
    return data


def process_sex (data):
    data['Sex'] = [float(0) if sex == 'male' else float(1) for sex in data['Sex']]    
    return data


def process_cabin (data):
    data['Cabin'] = data['Cabin'].isnull().astype(float)
    return data


def process_embarked (data):
#     data['Embarked'] = [float(1) if embarked == 'S' else float(2) if embarked == 'C' else float(3) for embarked in data['Embarked']]
    data['newEmbarkedS'] = [float(1) if embarked == 'S' else float(0) for embarked in data['Embarked']]
    data['newEmbarkedC'] = [float(1) if embarked == 'C' else float(0) for embarked in data['Embarked']]
    data = data.drop(['Embarked'], axis = 1)
    return data


def process_fare (data):
    data['Fare'] = [float(0) if fare <= 7.91 else float(1) if fare > 7.91 and fare <= 14.454 else float(2) if fare > 14.454 and fare <= 31 else float(3) for fare in data['Fare']]
    return data


def add_isAlone (data):
    data['isAlone'] = float(0)
    
    for index, df in data.iterrows():
        if df.SibSp + df.Parch == 0:
            data.at[index, 'isAlone'] = float(1)
    return data


def add_companion (data):
    data['companion'] = float(0)

    for index, df in data.iterrows():
        companion = df.SibSp + df.Parch
        data.at[index, 'companion'] = companion

    return data


def impute (data):
    if 'Survived' in data.columns:
        impute_data = data.drop(['Survived'], axis = 1)
    else:
        impute_data = data
    imputed = mice(impute_data.values)
    imputed_value = imputed[:, data.columns.get_loc('Age')-1]
    imputed_value = [0 if age < 0 else age for age in imputed_value]
    data['Age'] = imputed_value
    return data


def convert_to_float(data):
    data['Pclass'] = data['Pclass'].astype(float)
    data['SibSp'] = data['SibSp'].astype(float)
    data['Parch'] = data['Parch'].astype(float)
    return data


def process_features (data):
    data = data.drop(['Ticket', 'PassengerId'], axis = 1)  # drop 'Ticket' and 'PassengerId' column
    data = process_name(data)
    data = process_sex (data)
    data = process_cabin (data)
    data = process_embarked (data)
#     data = process_fare (data)
    data = add_isAlone(data)
    data = add_companion(data)
    data = convert_to_float(data) # converts int columns to float for MICE imputation
    data = impute(data)
    return data

features = process_features(data)


# In[ ]:


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(features)
label = features['Survived']
features = features.drop(['Survived'], axis=1)


# **Permutation Importance**
# 
# * Check the feature importances

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import eli5
import numpy as np
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(RandomForestClassifier(n_estimators=300), cv=50, random_state=0).fit(features.values, label.values)
eli5.show_weights(perm, feature_names = features.columns.tolist())


# *drop selected parameters*

# In[ ]:


drop_list = ['Parch', 'SibSp', 'Cabin', 'isAlone', 'companion']
features = features.drop(drop_list, axis = 1)


# **MODELS**
# 
# * Random Forest and KNeighbors Clasifier
# * Cross validate through GridSearchCV

# In[ ]:


import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier


def randomForestClassifier(X_train, y_train, n_fold):
	parameter_RandomForest = {
		'n_estimators': (10, 30, 50, 100, 200, 300, 400, 500, 700, 800, 1000),
		'max_features': ('auto', 'sqrt', 'log2', None),
		'criterion': ('gini', 'entropy')
	}
	gs_RandomForest = GridSearchCV(RandomForestClassifier(), parameter_RandomForest, verbose=1, cv=n_fold, n_jobs=-1)
	model = gs_RandomForest.fit(X_train, y_train)
	return model

def kNeighborsClassifier(X_train, y_train, n_fold):
	parameter_KNeighbors = {
		'n_neighbors': (3, 5, 7, 9, 11, 13, 15),
		'weights': ('uniform', 'distance'),
		'algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto')
	}
	gs_KNeighbors = GridSearchCV(KNeighborsClassifier(), parameter_KNeighbors, verbose=1, cv=n_fold, n_jobs=-1)
	model = gs_KNeighbors.fit(X_train, y_train)
	return model


n = 15

model1 = randomForestClassifier(features, label, n)

model2 = kNeighborsClassifier(features, label, n)


print('\nFor Random Forest:\n')
print(model1.best_score_)
print(model1.best_estimator_)
print(model1.best_params_)
print('\n')

print('\nFor KNeighbors:\n')
print(model2.best_score_)
print(model2.best_estimator_)
print(model2.best_params_)


# In[ ]:


# Create submission

def create_submission_csv(model, features, filename):
    pred = model.predict(features)
    passenger_id = np.array(test["PassengerId"]).astype(int)
    my_solution = pd.DataFrame(pred, passenger_id, columns = ["Survived"])
    my_solution.to_csv(filename, index_label = ["PassengerId"])
    
test_x = test
test_x = process_features(test_x)
test_x = test_x.drop(drop_list, axis = 1)
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean())

create_submission_csv(model1, test_x, 'result.csv')

