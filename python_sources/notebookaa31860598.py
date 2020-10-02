#!/usr/bin/env python
# coding: utf-8

# If you don't know where to go, start at the beginning. What are we looking at here? 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sample=pd.read_csv('../input/sample_submission.csv')
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')

#If you check this out in Linux (e.g. head train.csv), you see the extent of the data:
print(train.head(5)) #116 Categorical Values, and 14 continuous ones.
	


# train=pd.read_csv('../input/train.csv')
# nature. What can we do when we know nothing? Like the early taxonomists, we can start sorting! Categorical values appear to be mainly 'A' and 'B', with a few categorical columns containing various other characters.

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb #Run the classifier


print('This article was very helpful: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/')
print('As well as: https://www.kaggle.com/guyko81/allstate-claims-severity/just-an-easy-solution')

test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')
features = [x for x in train.columns if x not in ['id','loss']]
cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

for c in range(len(cat_features)):
	train[cat_features[c]] = train[cat_features[c]].astype('category').cat.codes
    
train['ln_loss'] = np.log(train['loss']) #Take LN of loss column
trainX=pd.DataFrame(train[features])
trainY=pd.DataFrame(train['ln_loss'])
xgdmat = xgb.DMatrix(trainX, trainY) #Put these in DMTATRIX

params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 
             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 
print('Training')
num_rounds = 1000
bst = xgb.train(params, xgdmat, num_boost_round = num_rounds)
print('Done Training')

print('Testing')
y_pred=[]
for c in range(len(cat_features)):
    test[cat_features[c]] = test[cat_features[c]].astype('category').cat.codes	
testDM=xgb.DMatrix(test[features])
y_pred+=list(bst.predict(testDM))
print('Done Testing')

print('Preparing to Submit')
y_pred=np.exp(y_pred)#Convert back to normal values
submission = pd.DataFrame({
        "id": test["id"],
        "loss": y_pred    })
submission.to_csv('XGB_OUT.csv', index=False)	#Including more min_splits and min_leaf helps out a little
print("Submission Saved")


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor as SDG

print('This code works poorly and gives huge swings in loss')
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')
print('Categories from : https://www.kaggle.com/guyko81/allstate-claims-severity/just-an-easy-solution')
features = [x for x in train.columns if x not in ['id','loss']]
cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]		
test,train='',''	

chunksize=20000
print('Training')
train=pd.read_csv('../input/train.csv',iterator=True,chunksize=chunksize)#Load in the file again
for chunk in train:
	conts=[x for x in chunk if 'cont' in x]#Only continuous values
	for c in range(len(cat_features)):
		chunk[cat_features[c]] = chunk[cat_features[c]].astype('category').cat.codes	
	trainX=pd.DataFrame(chunk[features])
	trainY=pd.DataFrame(chunk['loss'])
	alg = SDG(warm_start=True)
	alg.fit(trainX[features],trainY['loss'])
		
print('Done Training')
train,trainX,trainY='','',''

print('Testing')
#Iterate through?
test=pd.read_csv('../input/test.csv',iterator=True,chunksize=chunksize)
y_pred=[]
for chunk in test:
	for c in range(len(cat_features)):
		chunk[cat_features[c]] = chunk[cat_features[c]].astype('category').cat.codes	
	testX=pd.DataFrame(chunk[features]).astype(float)
	y_pred += list(alg.predict(testX))
print('Done Testing')

test=''
test=pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "id": test["id"],
        "loss": y_pred    })
#submission.to_csv('SGD_OUT.csv', index=False)	#Including more min_splits and min_leaf helps out a little
print("Submission Saved")


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor as SDG

print('Maybe transform this to  log_loss as guyko81 does. Also, remove the iterators.')
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')
print('Categories from : https://www.kaggle.com/guyko81/allstate-claims-severity/just-an-easy-solution')
features = [x for x in train.columns if x not in ['id','loss']]
cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]		
test,train='',''	

print('Training')
train=pd.read_csv('../input/train.csv')#Load in the file again


for c in range(len(cat_features)):
	train[cat_features[c]] = train[cat_features[c]].astype('category').cat.codes
train['ln_loss'] = np.log(train['loss']) 
trainX=pd.DataFrame(train[features])
trainY=pd.DataFrame(train['ln_loss'])
alg = SDG()
alg.fit(trainX[features],trainY['ln_loss'])
		
print('Done Training')
train,trainX,trainY='','',''

print('Testing')
#Iterate through?
test=pd.read_csv('../input/test.csv')
y_pred=[]
for c in range(len(cat_features)):
    test[cat_features[c]] = test[cat_features[c]].astype('category').cat.codes	
testX=pd.DataFrame(test[features])
y_pred += list(alg.predict(testX))
print('Done Testing')

print('Preparing to Submit')
y_pred=np.exp(y_pred)#Convert back to normal values
submission = pd.DataFrame({
        "id": test["id"],
        "loss": y_pred    })
submission.to_csv('SGD_OUT2.csv', index=False)	#Including more min_splits and min_leaf helps out a little
print("Submission Saved")

