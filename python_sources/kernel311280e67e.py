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


trainA = '/kaggle/input/physionet-challenge-2019early-detection-of-sepsis/training_setA/training'
print('training setA have {} files'.format(len(os.listdir(trainA))))
trainB = '/kaggle/input/physionet-challenge-2019early-detection-of-sepsis/training_setB/training_setB'
print('training setA have {} files'.format(len(os.listdir(trainB))))


# In[ ]:


os.chdir('/kaggle/input/physionet-challenge-2019early-detection-of-sepsis/training_setB/training_setB')
extension ='psv'
filenames = [i for i in glob.glob('*{}'.format(extension))]


# Extract data from each files in the Training A folder of our dataset. Patience is important parameter here(lol !), as we are extracting and concatinating data from 20,000 files on the go.

# In[ ]:


trainn = pd.concat([pd.read_csv(f , sep='|') for f in filenames])
trainn.to_csv('trainn.csv', index=False)


# Save the data extracted in to a new file called 'trainn' and check whether you are doing right! Also, check for missing values!

# In[ ]:


trainn.head()


# In[ ]:


trainn.columns


# In[ ]:


a,b = trainn.shape
x   = trainn.size
if ( a*b!= x):
    print('There is missing values! Clean it!')
else:
    print('There is no missing values! Yet there may be NAN values.')       


# In[ ]:


print('The dimensions of the given Training B dataset is:',trainn.shape)
print('The total number of data in given Training B dataset is',trainn.size)


# In[ ]:


trainn.SepsisLabel.plot.hist( color = "b")
trainn['SepsisLabel'].value_counts()


# Look out for the features that give us best Correlation with Target variable(SepsisLabel)
# Hence, let us use seaborn to check out our best correlation b\w features with target feature.

# In[ ]:


corrmat = trainn.corr()
top = corrmat.index
plt.figure(figsize=(30,30))
g = sns.heatmap(trainn[top].corr(), annot= True, cmap ="RdYlGn")


# Chose your Independent features and Target variable and eliminate unwanted features.

# In[ ]:


features = trainn.drop(columns =['SepsisLabel','PaCO2','SBP','MAP','DBP','SaO2','Potassium','Lactate','Glucose','AST','TroponinI','Hct', 'Hgb'])
features.head()


# In[ ]:


print(features.shape)
print(features.size)


# In[ ]:


training_features, testing_features, training_target, testing_target = train_test_split(features, trainn['SepsisLabel'],train_size=0.80, test_size=0.20)
training_features.shape, testing_features.shape, training_target.shape, testing_target.shape


# This dataset has many NAN values, so let's impute the missing values

# In[ ]:


imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)


# It is unbelievably a large dataset to be handled by this platform. 

# Stricitly make use of GPU's to train your model. I used Kaggle Kernels to train my models.

# In[ ]:


exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesClassifier( n_estimators=100, criterion="entropy", max_features="auto", min_samples_leaf=1, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testing_target,results)
cm


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testing_target,results)
accuracy


# In[ ]:


tn, fp, fn, tp = confusion_matrix(testing_target,results).ravel()
print(tn, fp, fn, tp)  


# In[ ]:


testing_target.value_counts()


# In[ ]:


from sklearn.metrics import precision_score
precision_score(testing_target,results)


# In[ ]:


from sklearn.metrics import recall_score
recall_score(testing_target,results)

