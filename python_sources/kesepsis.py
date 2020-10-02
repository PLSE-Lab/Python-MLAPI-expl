#!/usr/bin/env python
# coding: utf-8

# # SEPSIS PREDICTION MODEL 

# Import the necessary libraries and packages used.

# In[ ]:


import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer


# Here the platform i used was Kaggle as it provides free GPU along with 16 GB RAM.

# In[ ]:


train = '/kaggle/input/sepsisb/training_setB/'
print('training setA have {} files'.format(len(os.listdir(train))))


# In[ ]:


os.chdir('/kaggle/input/sepsisb/training_setB/')
extension ='psv'
filenames = [i for i in glob.glob('*{}'.format(extension))]


# Extract data from each files in the Training A folder of our dataset. Patience is important parameter here(lol !), as we are extracting and concatinating data from 20,000 files on the go.

# In[ ]:


trainn = pd.concat([pd.read_csv(f , sep='|') for f in filenames])
trainn.to_csv(r'trainn.csv')


# In[ ]:


from IPython.display import FileLink
FileLink('/kaggle/input/trainn.csv/')


# In[ ]:


X = trainn.drop(columns =['SepsisLabel','PaCO2','SBP','MAP','pH','DBP','PTT','FiO2','Potassium','Lactate','Glucose','AST','TroponinI','Hct', 'Hgb','O2Sat','Unit1','Unit2'])


# In[ ]:


X.head()


# In[ ]:


y = trainn['SepsisLabel']


# In[ ]:


y.head()


# In[ ]:


imputer = SimpleImputer(strategy="median")
imputer.fit(X)
X = imputer.transform(X)


# In[ ]:


clf = ExtraTreesClassifier( n_estimators=100, criterion="entropy", max_features="auto", min_samples_leaf=1, min_samples_split=5)

clf.fit(X,y)


# In[ ]:


import pickle


# In[ ]:


filename = '/kaggle/working/model_v1.pkl'
with open( filename, 'wb') as file:
    pickle.dump(clf, file) 


# In[ ]:


with open(filename ,'rb') as f:
    loaded_model = pickle.load(f)
loaded_model


# In[ ]:


X_test = np.array([120,21,16,43,34,25,45,22,45,6,104,0.9,20,8.6,8.9,52,2.2,22,32,28,0,-0.98,258])
X_test = X_test.reshape(1,-1)
y_predict = clf.predict(X_test)


# In[ ]:


y_predict


# In[ ]:


loaded_model.predict(X_test)

