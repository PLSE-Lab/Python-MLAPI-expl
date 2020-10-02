#!/usr/bin/env python
# coding: utf-8

# # Conoco Phillips Dataset Challenge - Team J.I.I.S
# 
# 
# 
# 
# The initial portion of our code is import statements and our data cleaning function. We experimented with a variety of ways to process the data, mainly due to the issue of large amounts of 'na's being found within. One of the methods implemented was replacing all entries of 'na' with -1 , we also tried dropping all columns above a certain 'na' threshold. In the end we found that a threshold of 79% resulted in the highest score. This threshold was decided based on sensors 41, 42, and 43 being sequential while also having a similar amount of 'na' percentage, around 80%.
# 
# We attempted a few methods of normalization within the data, such as standard and robust normalization. They were implemented using the standard scikit library functions StandardScaler and RobustScaler. However, we found that normalizing the data resulted in lower accuracy.

# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
# from sklearn import neighbors
from sklearn.metrics import f1_score
import seaborn as sns
import collections
import matplotlib.pyplot as plt

training_data = os.path.join(os.getcwd(), "tamu-datathon/equip_failures_training_neg_one.csv")
test_data = os.path.join(os.getcwd(), "tamu-datathon/equip_failures_test_set.csv")
training_data_cleaned = os.path.join(os.getcwd(), 'tamu-datathon/equip_failures_train_clean_drop_79.csv')
test_data_cleaned = os.path.join(os.getcwd(), 'tamu-datathon/equip_failures_test_clean_drop_79.csv')


def clean_data(df_clean):
    df_train = pd.read_csv(settings.training_data)
    df_test = pd.read_csv(settings.test_data)
    d1, d2 = clean_data(df_train, df_test)
    d1.to_csv(training_data_cleaned, index = False)
    d2.to_csv(test_data_cleaned, index = False)

# Clean Data
# clean_data(training_data)


# Another technique we attempted was non-negative matrix factorization. We converted all 'na's to -1, then shifted the entire dataset up by 1, resulting in 'na' being 0. We then converted it to an embedding space using scikit learns MF algorithm. The model was then stored as a pickle object and fed into our Random Forest algorithm. Ultimately we didn't use this method as the results were less accurate than the other data cleaning methods we tried.

# In[ ]:


import pickle
from sklearn.decomposition import NMF

df_training = pd.read_csv(training_data_cleaned)
df_drop = df_training.drop(columns=['id','target'])
df_drop = df_drop + 1

np_drop = df_drop.to_numpy()
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(np_drop)

with open('new_matrix.pickle', 'wb') as handle:
    pickle.dump(W, handle)


# In[ ]:





# We tested machine learning algorithms such as kNN, SVM, Random Forest and AdaBoost. For the majority of experiments, Random Forest gave the best performance regardless of the cleaning or normalization method implemented.
# 
# We did attempt to do some hyperparameter tuning on the algorithms we ran. In some instances (e.g. number of trees comprising Random Forest) we tried manual hyperparameter tuning. We also attempted to run a randomized search of the hyperparameters; however we were unable to get results in time.

# In[ ]:


def gen_output(predictions):
    columns = ['id', 'target']

    df = pd.read_csv(settings.test_data_cleaned)
    X_test = df.iloc[:, 1:]  # First one columns are id
    model = alg.fit(X, Y)
    predictions = model.predict(X_test)
    id = list(range(16002))
    id = id[1::]
    csv = pd.DataFrame()
    csv['id'] = id
    csv['target'] = predictions
    csv.to_csv('sample_drop79.csv', index = False)

print("Executing...")

df = pd.read_csv(training_data_cleaned)
X = df.iloc[:, 2:]  # First two columns are id and target
Y = np.array(df.iloc[:, 1])

alg = RandomForestClassifier(n_estimators=250)

cv = StratifiedKFold(n_splits=10)

fscores = []
for i, (train, test) in enumerate(cv.split(X, Y)):
    model = alg.fit(X.iloc[train], Y[train])
    Y_pred = model.predict(X.iloc[test])
    fscore = f1_score(Y[test], Y_pred, average='weighted', labels=np.unique(Y[test]))
    fscores.append(fscore)
    print('Fold', i, ':', fscore)

print('Average F-measure:', sum(fscores) / len(fscores))


# In[ ]:


#Plot Feature Importance
importances = list(model.feature_importances_)
column_headers = list(df.columns.values)
dicy = dict(zip(importances, column_headers))
dicysort = collections.OrderedDict(sorted(dicy.items()))
sns.set(style='whitegrid')
ax = sns.barplot(x=[dicysort[i] for i in dicysort.keys()], y=dicysort.keys(), data=dict(dicysort))
plt.show()

