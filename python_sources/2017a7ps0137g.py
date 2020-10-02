#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


t_df =pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
dic = {"new":1,"old":0}
t_df["type"]= t_df["type"].apply(lambda x:dic[x])
for  i in [3,4,5,8,9,10,11]:
    t_df[str("feature"+str(i))] = t_df[str("feature"+str(i))].astype(np.float64)
t_df=t_df.dropna()
t_df.drop(['id'],axis = 1,inplace = True)


# In[ ]:


a=t_df["rating"].values
from collections import Counter
cdict = Counter(a)
ri=[i for i,j in cdict.items()]
rj=[j for i,j in cdict.items()]
sns.barplot(x=ri,y=rj)


# In[ ]:


t_df_1 = t_df
t_df_1.drop(["feature7","feature10"],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x=sc.fit_transform(t_df_1.iloc[:,:-1])
train_y=t_df_1.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(test_size=0.2, random_state=42)
for train_index, test_index in sss.split(train_x,train_y):
     X_train, X_test = train_x[train_index], train_x[test_index]
     Y_train, Y_test = train_y[train_index], train_y[test_index]


# In[ ]:


len(X_train[0])


# In[ ]:


res_0=[]
res_6=[]
res_1=[]
res_5=[]
res_4=[]
for i in range(len(X_train)):
     if Y_train[i] ==0:
        res_0.append(X_train[i])
     if Y_train[i] ==1:
        res_1.append(X_train[i])
     if Y_train[i] ==4:
        res_4.append(X_train[i])
     if Y_train[i] ==5:
        res_5.append(X_train[i])
     if Y_train[i] ==6:
        res_6.append(X_train[i])


# In[ ]:


print(len(res_0),len(res_1),len(res_4),len(res_5),len(res_6))


# In[ ]:


for _ in range(10):
        X_train=np.concatenate((X_train,res_0))
        for j in range(len(res_0)):
            Y_train=np.append(Y_train,0)
for _ in range(20):
    X_train=np.concatenate((X_train,res_6))
    for j in range(len(res_6)):
         Y_train=np.append(Y_train,6)


# In[ ]:


rest=[]
for i in range(len(X_train)):rest.append([X_train[i],Y_train[i]])
import random
random.shuffle(rest)
resi=[i for i,j in rest]
resj=[j for i,j in rest]
resi=np.array(resi)
X_train = resi
Y_train = np.array(resj)


# In[ ]:


X_ctrain=np.concatenate((X_train,X_test))
Y_ctrain=np.concatenate((Y_train,Y_test))
len(X_ctrain),len(Y_ctrain)


# In[ ]:


# best_error = 10000
# best_parameters = []
# def bestparam(parameters,train_df):
#     global best_error,best_parameters
#     t_df_n = train_df[parameters]
#     train_x=t_df_n.iloc[:,:-1].values
#     train_y=t_df_n.iloc[:,-1].values
#     from sklearn.model_selection import StratifiedShuffleSplitA
#     sss = StratifiedShuffleSplit(test_size=0.25, random_state=42)
#     for train_index, test_index in sss.split(train_x,train_y):
#          X_train, X_test = train_x[train_index], train_x[test_index]
#          Y_train, Y_test = train_y[train_index], train_y[test_index]
#     gb = RandomForestClassifier(n_estimators=50)
#     gb.fit(X_train,Y_train)
#     pred = gb.predict(X_test)
#     error=(mean_squared_error(pred,Y_test))**(0.5)
#     if error < best_error:
#         best_parameters = parameters
#         best_error = error
#         print("Yeah baby!*********************************************************************************")
#         print()
#         print()
#     print(error,parameters)
    
    


# In[ ]:


# features = t_df_1.columns[:-1]
# from itertools import combinations

# for i in [10]:
#     prams = combinations(features,i)
#     for prm in prams:
#         fet = list(prm)
#         fet.append("rating")
#         #print(fet)
#         bestparam(fet,t_df_1)
# print("__"*20)
# print()
# print(best_error,best_parameters)


# In[ ]:


# if True:    
#     t_df_n = t_df_1[best_parameters]
#     sc = StandardScaler()
#     train_x=sc.fit_transform(t_df_n.iloc[:,:-1])
#     train_y=t_df_n.iloc[:,-1].values
# #     from sklearn.model_selection import StratifiedShuffleSplit
# #     sss = StratifiedShuffleSplit(test_size=0.2, random_state=42)
# #     for train_index, test_index in sss.split(train_x,train_y):
# #          X_train, X_test = train_x[train_index], train_x[test_index]
# #          Y_train, Y_test = train_y[train_index], train_y[test_index]
# #     n_est = [20,50,100,150,200,250,300,350,400,500,600,700,800,900,950,1000]
# #     res=[]
# #     best_error=1000
# #     best_n=0
# #     for nie in n_est:
# #         gb = RandomForestClassifier(n_estimators=nie)
# #         gb.fit(X_train,Y_train)
# #         pred = gb.predict(X_test)
# #         error=(mean_squared_error(pred,Y_test))**(0.5)
# #         res.append([nie,error])
# #         print(nie,error)
# #         if error < best_error:
# #             best_error = error
# #             best_n=nie
# #             print("YEAH BABY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# #     print(best_error)
# #     xi=[i for i,j in res]
# #     yj=[j for i,j in res]
# #     sns.scatterplot(x=xi,y=yj)
#     gb = RandomForestClassifier(n_estimators=1000)
#     gb.fit(train_x,train_y)    
        


# In[ ]:


# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# """n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num =11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(train_x,train_y)
# print(rf_random.best_params_)"""

# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [30,50,70, 90, 110],
#     'max_features': [2,3,4],
#     'min_samples_leaf': [1,2,3,5],
#     'min_samples_split': [5,7,10,12],
#     'n_estimators': [100,250,500,750,1000]
# }
# # Create a based model
# rt = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rt, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[ ]:


# grid_search.fit(X_train,Y_train)
# grid_search.best_params_


# In[ ]:


gb = RandomForestClassifier(n_estimators= 800,max_depth=80)
gb.fit(X_train,Y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
pred = gb.predict(X_test)
print((mean_squared_error(pred,Y_test))**(0.5))


# In[ ]:


test_df = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
ids= test_df["id"].values
dic = {"new":1,"old":0}
test_df["type"]= test_df["type"].apply(lambda x:dic[x])
test_df = test_df.iloc[:,1:]
for feature in test_df.columns:
    avg_norm_loss = test_df[feature].mean()
    test_df[feature].fillna(value = avg_norm_loss,inplace = True)


# In[ ]:


test_df.drop(["feature7","feature10"],axis=1,inplace=True)


# In[ ]:


sc = StandardScaler()
test_x=sc.fit_transform(test_df)


# In[ ]:


len(test_x[0])


# In[ ]:


pred = gb.predict(test_x)
resu=[[ids[i],pred[i]] for i in range(len(pred))]
result = pd.DataFrame(resu,columns=["id","rating"])
result.head()
result.to_csv("dat2.csv",index=False)


# In[ ]:


create_download_link(result)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


# In[ ]:


clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                           random_state=1,max_iter=700)
clf2 = RandomForestClassifier(n_estimators=1000, random_state=1)
clf3 = SVC(kernel="rbf", C=0.025, probability=True,gamma="scale")
clf4 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3),('gnb',clf4)], voting='soft')


# In[ ]:


for clf, label in zip([clf1, clf2, clf3,clf4, eclf], ['Logistic Regression', 'Random Forest', 'SVM','GaussianNB','Ensemble']):
    scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# In[ ]:


# clf1.fit(train_x,train_y)
 clf2.fit(train_x,train_y)
# clf3.fit(train_x,train_y)
# clf4.fit(train_x,train_y)
# eclf.fit(train_x,train_y)


# In[ ]:


# pred1 = clf1.predict(X_test)
 pred2 = clf2.predict(X_test)
# pred3 = clf3.predict(X_test)
# pred4 = clf4.predict(X_test)
# pred5 = clf5.predict(X_test)


# In[ ]:


# print((mean_squared_error(pred1,Y_test))**(0.5))
print((mean_squared_error(pred2,Y_test))**(0.5))
# print((mean_squared_error(pred3,Y_test))**(0.5))
# print((mean_squared_error(pred4,Y_test))**(0.5))
# print((mean_squared_error(pred5,Y_test))**(0.5))


# In[ ]:


gb = clf2


# In[ ]:


pred = gb.predict(test_x)
resu=[[ids[i],pred[i]] for i in range(len(pred))]
result = pd.DataFrame(resu,columns=["id","rating"])
result.to_csv("data3.csv",index=False)
create_download_link(result)

