#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


POS_path = "../input/POS_CASH_balance.csv"
bureau_balance_path = "../input/bureau_balance.csv"
app_train_path = "../input/application_train.csv"
prev_app_path = "../input/previous_application.csv"
payments_path = "../input/installments_payments.csv"
card_bal_path = "../input/credit_card_balance.csv"
bureau_path = "../input/bureau.csv"
app_test_path = "../input/application_test.csv"


# In[ ]:


#pos_cash_balance = pd.read_csv(POS_path)
bureau_balance = pd.read_csv(bureau_balance_path)
app_train = pd.read_csv(app_train_path)
#payments = pd.read_csv(payments_path)
#prev_app = pd.read_csv(prev_app_path)
#card_bal = pd.read_csv(card_bal_path)
bureau = pd.read_csv(bureau_path)
app_test = pd.read_csv(app_test_path)


# In[ ]:


joinedTrain = app_train.copy()
joinedTest = app_test.copy()


# In[ ]:


threshold = 0.9

#Absolute value correlation matrix
corr_matrix_Train = joinedTrain.corr().abs()
corr_matrix_Bureau = bureau.corr().abs()

#Upper triangle of correlations
upper_Train = corr_matrix_Train.where(np.triu(np.ones(corr_matrix_Train.shape), k=1).astype(np.bool))
upper_Bureau = corr_matrix_Bureau.where(np.triu(np.ones(corr_matrix_Bureau.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
to_drop_Train = [column for column in upper_Train.columns if any(upper_Train[column] > threshold)]
to_drop_Bureau = [column for column in upper_Bureau.columns if any(upper_Bureau[column] > threshold)]

# Remove the columns
joinedTrain = joinedTrain.drop(columns = to_drop_Train)
bureau = bureau.drop(columns = to_drop_Bureau)


# In[ ]:


bureau_balance = pd.get_dummies(bureau_balance)


# In[ ]:


aggFuncs = {'MONTHS_BALANCE':['min'],'STATUS_0':['sum'],'STATUS_1':['sum'],'STATUS_2':['sum'],'STATUS_3':['sum'],'STATUS_4':['sum'],'STATUS_5':['sum'],'STATUS_C':['sum'],'STATUS_X':['sum'],}


# In[ ]:


bureau_balance = bureau_balance.groupby('SK_ID_BUREAU').agg(aggFuncs).reset_index()


# In[ ]:


bureau_balance.columns = bureau_balance.columns.droplevel(1)


# In[ ]:


bureau = pd.get_dummies(bureau)


# In[ ]:


joinedBureaus = bureau.set_index('SK_ID_BUREAU').join(bureau_balance.set_index('SK_ID_BUREAU'),lsuffix='_main', rsuffix='_other',how='left').reset_index()


# In[ ]:


burImp = Imputer()

burImp.fit_transform(joinedBureaus)

joinedBureaus = joinedBureaus.fillna(0)


# In[ ]:


joinedBureaus = joinedBureaus.groupby('SK_ID_CURR').sum().reset_index().drop('SK_ID_BUREAU',axis=1)


# In[ ]:


joinedTrain = joinedTrain.set_index('SK_ID_CURR').join(joinedBureaus.set_index('SK_ID_CURR'),lsuffix='_main', rsuffix='_other',how='left')
joinedTest = joinedTest.set_index('SK_ID_CURR').join(joinedBureaus.set_index('SK_ID_CURR'),lsuffix='_main', rsuffix='_other',how='left')


# In[ ]:


y = joinedTrain['TARGET']
X = joinedTrain.drop('TARGET', axis=1)


# In[ ]:


le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in X:
    if X[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(X[col].unique())) <= 2:
            # Train on the training data
            le.fit(X[col])
            # Transform both training and testing data
            X[col] = le.transform(X[col])
            joinedTest[col] = le.transform(joinedTest[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# In[ ]:


X = pd.get_dummies(X)
joinedTest = pd.get_dummies(joinedTest)


# In[ ]:


m=[]
for i in X.columns:
    if i not in joinedTest.columns:
        X.drop(i,axis=1)
        m.append(i)
print(m)


# In[ ]:


X, joinedTest = X.align(joinedTest, join = 'inner', axis = 1)

print('Training Features shape: ', X.shape)
print('Testing Features shape: ', joinedTest.shape)


# In[ ]:


MyImp = Imputer()

MyImp.fit_transform(X,y)
MyImp.transform(joinedTest)

X = X.fillna(0)
joinedTest = joinedTest.fillna(0)


# In[ ]:


X, joinedTest = X.align(joinedTest, join = 'inner', axis = 1)

print('Training Features shape: ', X.shape)
print('Testing Features shape: ', joinedTest.shape)


# In[ ]:


nBags = 13
bagX, wtX, bagy, wty = train_test_split(X,y,test_size=0.2)
bagFull = bagX.copy()
bagFull['TARGET'] = bagy.copy()
wtPreds = []
preds = []


# In[ ]:


#rfParams = {'n_estimators':[10,100],'class_weight':['balanced','None']}
#gsRF = GridSearchCV(RandomForestClassifier(),rfParams)
#gsRF.fit(bagX,bagy)
#rfParams = gsRF.best_params_


# In[ ]:


#xgbParams = {'learning_rate':[0.1,0.3,0.5],'scale_pos_weight':[0.3,1]}
#gsXGB = GridSearchCV(XGBClassifier(),xgbParams)
#gsXGB.fit(bagX,bagy)
#xgbParams = gsXGB.best_params_


# In[ ]:


#knParams = {'n_neighbors':[25,50,100],'weights':['uniform','distance']}
#gsKN = GridSearchCV(KNeighborsClassifier(),knParams)
#gsKN.fit(bagX,bagy)
#knParams = gsKN.best_params_


# In[ ]:


#lgmParams = {'n_estimators':[25,100,1000],'learning_rate':[0.05,0.1,0.2]}
#gsLGM = GridSearchCV(LGBMClassifier(),lgmParams)
#gsLGM.fit(bagX,bagy)
#lgmParams = gsLGM.best_params_


# In[ ]:


for i in range(nBags):
    bagSamp = bagFull.sample(frac=0.8,replace=True)
    inBagy = bagSamp['TARGET']
    inBagX = bagSamp.drop('TARGET',axis=1)
    #X_train, X_test, y_train, y_test = train_test_split(inBagX,inBagy,test_size = 0.33)
    bagRF = RandomForestClassifier(n_estimators=20,class_weight='balanced')
    bagXGB = XGBClassifier(scale_pos_weight=0.3)
    bagKN = KNeighborsClassifier(n_neighbors=50)
    bagLGM = LGBMClassifier(class_weight='balanced')
    mods = [bagRF,bagXGB,bagKN,bagLGM]
    for mod in mods:
        mod.fit(inBagX,inBagy)
        modProbs = mod.predict_proba(joinedTest)
        wtProbs = mod.predict_proba(wtX)
        preds.append(modProbs[:,1])
        wtPreds.append(wtProbs[:,1])
    


# In[ ]:


subMatrix = np.array(preds).transpose()
wtMatrix = np.array(wtPreds).transpose()


# In[ ]:


weights = np.full((len(wtPreds),1),(1/len(wtPreds)))
wtProba = np.matmul(wtMatrix,weights)
rocTarg = roc_auc_score(wty,wtProba)


# In[ ]:


for 1 in range(20000):
    startWeights = np.multiply(abs(np.random.randn(len(wtPreds))),weights)
    newWeights = startWeights/sum(startWeights)
    newProbs = np.matmul(wtMatrix,newWeights)
    rocScore = roc_auc_score(wty,newProbs)
    if rocScore > rocTarg:
        rocTarg = rocScore
        weights = newWeights
    


# In[ ]:


subProba = np.matmul(subMatrix,weights)


# In[ ]:


subProbsTrue = subProba.copy().flatten()


# In[ ]:


subCol1 = app_test['SK_ID_CURR'].as_matrix()
subCol2 = subProbsTrue


# In[ ]:


subData = {'SK_ID_CURR':subCol1,'TARGET':subCol2}


# In[ ]:


subFrame = pd.DataFrame(data=subData)


# In[ ]:


subFrame.to_csv('6thSub.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




