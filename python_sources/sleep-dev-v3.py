#!/usr/bin/env python
# coding: utf-8

# <h1> Detecting Failure : Team Sleep Deprived </h1>

# members -
# Chen Liang,
# Sunny Guha,
# Prajakta Bedekar,
# Srinivas Subramanian
# 

# <h2>Business Understanding</h2>

# Equipment fail with oil wells can bring various negative impacts to both the company and environment. Thus, detecting failure event is an essential task. Several sensors are used for gathering various information from the equipment in order to detect equipments faliure. With the help of predictive models, equipment failure can be easier to find in a timely manner.

# <h2>Data Understanding</h2>

# A training set is provided for model building. Readings from 107 sensors are recorded, where these sensors can be categorized as two types: discrete and histogram(time-based). The goal is to categorize whether or not the given equipment pattern shows the equipment failure. Thus, the goal of this dataset is to do a classification task on sensor data.

# <h2> Import Packages </h2>

# In[ ]:


from sklearn import preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydot
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import datetime, os

import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from IPython.display import display
get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import normalize
pd.set_option('display.max_columns', None)
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error

import pickle


# <h2>Data Preprocessing</h2>

# In[ ]:


odf=pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv',index_col=0)
odft=pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv',index_col=0)


# In[ ]:


odf.describe()


# In[ ]:


odf.shape


# In[ ]:


odf.dtypes


# It can be found that some of the columns are of type 'object', which indicates that there are potentially other non-numerical data in it.

# <h3>Handle Missing Data</h3>

# In[ ]:


odf.head(5)


# As shown above some of values are 'na' in the type of string. Thus these values needs to be replaced. Since all values are comparatively large (from the df describe()), as for here, we use one traditionally used method to handle missing data, which is to replace them as a very small value. -999999 is used for replacement.

# In[ ]:


df=odf.replace({'na':-999999})
Xt=odft.replace({'na':-999999})
Xt=Xt.astype(float)
df=df.astype(float)
df['target']=df['target'].astype(int)


# <h3>Unbalanced Classes</h3>

# In[ ]:


sns.countplot(df['target'],label="Count")


# The data highly unbalanced, having a ratio of 60:1. So this informs us that for some learning algorithm we use needs to have a very high penalty for the 1s so that it does not get biased to the zeros. 

# <h3>Data Normalization</h3>

# Data is normalized here in case they are needed in future. Original data is also kept here.

# In[ ]:


X=df.iloc[:,1:]
y=df.iloc[:,0]

#L2 Normalize
Xn=normalize(X)
Xtn=normalize(Xt)


# In[ ]:


# Confusion Matrix
def confusion_matrix(target, prediction, score=None):
   cm = metrics.confusion_matrix(target, prediction)
   plt.figure(figsize=(4,4))
   sns.heatmap(cm, annot=True,fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
   plt.ylabel('Act')
   plt.xlabel('Pred')


# <h3> Train and Test Separation  </h3>

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=8)


# <h2> Model Training </h2>

# Various models are trained and compared below. Due to limited time and computing resources, some models are skipped or not fully tuned.

# Since it is a classification task, several classification models are used and compared below. Among these, some of the models are skipped as shown below:
# 1. KNN: Although KNN also does classification, the data has too many features for KNN to work properly. In other words, the dimension is too high and thus it will become sparse in high dimensional space. Thus, KNN is skipped/
# 2. SVM: SVM is skipped due to high training time. Tuning parameters for SVM also takes much time. However, if this works, it will be one of the good choice since it works pretty fast for prediction and can be potentially applied to some embedded systems.
# 
# Below most methods we used are tree-based algorithms since they are pretty good at classifying.

# <h4> Model: Extra Trees Classifier </h4>

# In[ ]:


my_class = ExtraTreesClassifier(random_state=0)
my_class.fit(X_train, y_train)
y_pred= clf.predict(X_test)
print('accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(f'F1: {f1_score(y_test,y_pred)}')
confusion_matrix(y_test,y_pred)


# <h4> Model: AdaBoost Classifier </h4>

# In[ ]:


my_class = AdaBoostClassifier(random_state=0)
my_class.fit(X_train, y_train)
y_pred= my_class.predict(X_test)
print('accuracy: {}'.format(accuracy_score(y_test, y_pred)))
print(f'F1: {f1_score(y_test,y_pred)}')
confusion_matrix(y_test,y_pred)


# <h4> Model: Logstic Regression <h4>

# In[ ]:


lg = LogisticRegression(solver='lbfgs', random_state=18)
lg.fit(X_train, y_train)
logistic_prediction = lg.predict(X_test)
score = metrics.accuracy_score(y_test, logistic_prediction)
print(score)
confusion_matrix(y_test,logistic_prediction)


# Simple regression based models are bad because they are mis-detecting 1s as 0 which is pretty bad considering the dataset is highly unbalanced are there are only a few 1s in the sample. This motivates us to use some forest based techniques.

# <h4> Model: XGBoost <h4>

# In[ ]:


data_dmatrix = xgb.DMatrix(data=Xn,label=y)
xgc = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.2,
                          learning_rate = 0.1, 
                          max_depth = 20, alpha = 10, n_estimators = 700)
xgc.fit(X,y)


# In[ ]:


pred_train=xgc.predict(X)
pred_train.sum()


# In[ ]:


pred_test=xgc.predict(Xt)
pred_test.sum()


# In[ ]:


yt=pd.DataFrame(pred_test)
yt.index=yt.index+1
yt


# In[ ]:





# In[ ]:


test=pd.read_csv('../input/equipfails/equip_failures_test_set.csv',na_values='na')
df= pd.DataFrame()
df['id'] = test['id']
df['target'] = pred_test
df.to_csv('submission2.csv', index=False)


# <h2> Model Save </h2>

# In[ ]:


file_name='submision.csv'
yt.to_csv(file_name,index=True)
# from IPython.display import FileLink
# FileLink(file_name)


# In[ ]:


filename = 'Final_Model.mod'
pickle.dump(xgc, open(filename, 'wb'))

