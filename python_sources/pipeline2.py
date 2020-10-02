#!/usr/bin/env python
# coding: utf-8

# # Parameter Search & Model Selection & Feature Analysis

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Load Data 

# In[ ]:


X_train = pd.read_csv('../input/algor-s-1st-kernel/train_data.csv').iloc[:,1:]
Y_train = pd.read_csv('../input/algor-s-1st-kernel/train_label.csv').iloc[:,1:]
X_test = pd.read_csv('../input/algor-s-1st-kernel/test_data.csv').iloc[:,1:]
ids = np.asarray(X_test['PassengerId'])
X_test.drop(['PassengerId'],axis=1, inplace=True)
Y_train = Y_train.values.reshape(Y_train.shape[0])


# # Function of parameter search for single model 
# 
#  Currently only for supervised learninig 

# In[ ]:


'''
Input: 
params: parameter set to search on
model: the model to search on
metric: evaluation metric
X_data: input features
Y_data: real output

Return:
best_params: dictionary of best parameters
best_score: best performance
'''
from sklearn.model_selection import GridSearchCV

def Search_para( model, params, metric, X_data, Y_data=None):
    if(Y_data is not None): #supervised learning
        searcher = GridSearchCV(model, params, cv=5, scoring = metric)
        searcher.fit(X_data, Y_data)
        best_params = searcher.best_params_
        best_score = searcher.best_score_

    return (best_params, best_score)


# # Function to search parameter for each model and find the best model  

# In[ ]:


'''
Input:
model_list: list contataining name of models
model_collection: tuples of candidate models
params_collection: tuples of dictionaries of parameters
metric: evaluation metric
X_data: input features
Y_data: real output

Return:
res_dic: Dictionary that stores the best params  and performance as value, model name as key
'''

def Search_Compare(model_list, model_collection, params_collection, metric, X_data, Y_data=None):
    
    res_dic = {}
    #store the scores for model comparison
    scores = []
    
    for model_name, model, params in zip(model_list, model_collection, params_collection):
        res_tup =         Search_para(model, params, metric = metric, X_data=X_data, Y_data = Y_data)
        scores += [res_tup[1]]
        res_dic[model_name]  = res_tup
        print("model: " + model_name)
        print("Best parameter: {}\n Score: {:5f}".format(res_tup[0], res_tup[1]))
    d = {'model': model_list, 'scores': scores}
    
    res_df = pd.DataFrame(data = d)
    print(res_df.sort_values(by=['scores'], ascending=False))
    return res_dic
    
    


# # Set model names list,  model collection, parameter set collection,

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

model_names = ['GBT', 'SVC', 'RF']
models = [SVC(), RandomForestClassifier()]
'''
parameters = [{'learning_rate':[1e-4, 1e-3, 1e-2], 'min_samples_leaf':list(range(80,160,10)),\
              'n_estimators':[3000], 'max_depth':list(range(3,6,1)), \
               'max_features':['sqrt', 'log2', None]},
        {'C':[1e-3, 1e-2, 1e-1, 1, 10], 'kernel' : ['poly', 'rbf', 'sigmoid'],  'degree': [3,4,5,6],\
             'gamma':['auto', 1e-3, 1e-2, 0.1, 1, 10]},
        {'min_samples_leaf':list(range(80,160,10)),'n_estimators':[3000], \
         'max_depth':list(range(3,6,1)), 'max_features':['sqrt', 'log2', None]}]
'''

parameters = [{'C':[1e-3, 1e-2, 1e-1, 1, 10], 'kernel' : ['poly', 'rbf', 'sigmoid'],  'degree': [3,4],             'gamma':['auto', 1e-3, 1e-2, 0.1, 1, 10]},
        {'min_samples_leaf':list(range(80,160,10)),'n_estimators':[3000], \
         'max_depth':list(range(3,6,1)), 'max_features':['sqrt', 'log2', None]}]
            


# # Start searching

# In[ ]:


Search_Compare(model_list=model_names, model_collection=models, params_collection=parameters,               metric='accuracy',X_data=X_train, Y_data=Y_train)


# # Training

# In[ ]:


clf = GradientBoostingClassifier(learning_rate=0.01, max_depth = 4, min_samples_leaf=80,                                  max_features='sqrt',n_estimators=3000)
clf.fit(X_train, Y_train)
print(clf.score(X_train, Y_train))


# # Analysis on misclassified instances

# ## Obtain misclassification index

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
sss = StratifiedKFold(n_splits=5)

model = clf
#store the index of misclassified data
mis_index = []
#store the prediction and index of misclassified data
mis_pred = []
X = X_train.values
y = Y_train
for train_index, test_index in sss.split(X, y):
        
    x_train, y_train = X[train_index], y[train_index]
    x_test, y_test = X[test_index], y[test_index]
    
    model.fit(x_train, y_train)
        
    y_pred = model.predict(x_test)
    
    #Get index of misclassified data
    mis_index_in_test = np.asarray(np.where(y_test != y_pred)).tolist()[0]
    
    temp_index = test_index[mis_index_in_test]
    temp_result = []
    for index, pred in zip(temp_index,y_pred[mis_index_in_test]):
        temp_result.append([index, pred])
    
    mis_index.extend(temp_index)
    mis_pred.extend(temp_result)
    #print(mis_pred)

mis_index= np.unique(mis_index) 
 

#print(mis_pred)
#X_train.iloc[mis_index,:].describe()


# ## Get misclassification data 

# In[ ]:


mis_pred = np.asarray(mis_pred)
mis_pred_df = pd.DataFrame(data=mis_pred, columns=['index', 'mis_pred']).sort_values(by='index').drop_duplicates()
mis_temp_df = X_train.iloc[mis_index,:]
mis_temp_df = mis_temp_df.assign(mis_pred = mis_pred_df['mis_pred'].values)

print(mis_temp_df.describe())


# # Submission

# In[ ]:


Y_pred = clf.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": ids,
        "Survived": Y_pred
    })
submission.to_csv('submit.csv')

