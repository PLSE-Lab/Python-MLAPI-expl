#!/usr/bin/env python
# coding: utf-8

# # Gender Recognition by voice

# <img src="https://d1sr9z1pdl3mb7.cloudfront.net/wp-content/uploads/2018/01/09162655/voice-biometrics-large1-1024x448.jpg"  width="700" height="100" />

# ### Problem Statement

# ##### Gender Recognition by Voice and Speech Analysis
# 
# ##### This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech

# ### Import Data

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
#import category_encoders as ce #encoding
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #dim red
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 



get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the comma separated values file into the dataframe

# In[ ]:


GenReg_ds = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-3/master/Projects/gender_recognition_by_voice.csv')


# In[ ]:


GenReg_ds.head(10)


#    The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of 0hz-280hz

# In[ ]:


GenReg_ds.shape


# In[ ]:


GenReg_ds.info()


# In[ ]:


GenReg_ds.describe()


# #### To find count of Null values

# In[ ]:


GenReg_ds.isnull().sum()


#  There is no null values in this dataset.

# ### To find the category type features

# In[ ]:


GenReg_ds.select_dtypes(include=['object']).head()


# In[ ]:


print("Total number of labels : {} ".format(GenReg_ds.shape[0]))
print("Total number of males : {}".format(GenReg_ds[GenReg_ds.label=='male'].shape[0]))
print("Total number of females : {}".format(GenReg_ds[GenReg_ds.label=='female'].shape[0]))


# ### Checking the correlation between each feature

# In[ ]:


GenReg_ds.corr()


# In[ ]:


sb.heatmap( GenReg_ds.corr());


# Centroid and dfrange both are having more correlated. So, we are going to drop both features.

# In[ ]:


GenReg_ds.drop(['centroid'], axis=1, inplace=True)


# In[ ]:


GenReg_ds.drop(['dfrange'], axis=1, inplace=True)


# In[ ]:


GenReg_ds.columns


# In[ ]:


GenReg_ds['label'].value_counts().plot(kind='bar',figsize = (12,5),fontsize = 14,colormap='Dark2', yticks=np.arange(0, 19, 2))
plt.xlabel('Gender')
plt.ylabel('No. of persons')


# In[ ]:


Male_df = GenReg_ds.loc[GenReg_ds.label == "male"]
Female_df = GenReg_ds.loc[GenReg_ds.label == "female"]
print(Male_df.shape)
print(Female_df.shape)


# In[ ]:


Male_df['meanfreq'].plot(kind='line', figsize=(12,5), color='blue', fontsize=13, linestyle='-.')
plt.ylabel('Meanfreq')
plt.title('Mean Frequency for Male persons')


# In[ ]:


Female_df['meanfreq'].plot(kind='line', figsize=(12,5), color='blue', fontsize=13, linestyle='-.')
plt.ylabel('Meanfreq')
plt.title('Mean Frequency for Female persons')


# ### To findout the features which are standard deviation equals zero

# In[ ]:


stdcol = GenReg_ds['modindx'].std()==0     
stdcol


# ### Defining X and Y values

# In[ ]:


X = GenReg_ds.loc[:,GenReg_ds.columns != 'label']
X.head()


# In[ ]:


drop_cols=[]
for cols in X.columns:
    if X[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols))
print(drop_cols)
X.drop(drop_cols,axis=1, inplace = True)


# We do not have constant columns in this dataset.

# In[ ]:


Y = GenReg_ds.loc[:,GenReg_ds.columns == 'label']
Y.head()


# ### Label Encoding

# In[ ]:


gender_encoder = LabelEncoder()
Y = gender_encoder.fit_transform(Y)
Y


# ### Data Standardisation
# 
# Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model. Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data.

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# ### Splitting dataset into training set and testing set for better generalisation

# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 1)


# In[ ]:


X_train


# ### Running SVM with default hyperparameter

# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test,Y_pred))


# ### Default Linear kernel

# In[ ]:


svc = SVC(kernel='linear')
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test,Y_pred))


# ### Default RBF kernel

# In[ ]:


svc=SVC(kernel='rbf')
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(Y_test,Y_pred))


# We can see from above accuracy score that svm default parameter for kernel is "rbf" 

# ### Default Polynomial kernel

# In[ ]:


svc=SVC(kernel='poly')
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test,Y_pred))


# In[ ]:


Y


# Polynomial kernel is performing poorly. The reasonbbehind this maybe it is overfitting the training dataset. 

# ### Cross validation with different kernels

# #### CV for Linear kernel

# In[ ]:


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear')
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)


# In[ ]:


print(scores.mean())


# #### CV for RBF kernel

# In[ ]:


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='rbf')
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)


# In[ ]:


print(scores.mean())


# #### CV for polynomial kernel

# In[ ]:


from sklearn.model_selection import cross_val_score
svc=SVC(kernel='poly')
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)


# In[ ]:


print(scores.mean())


# When K-fold cross validation is done we can see different score in each iteration.This happens because when we use train_test_split method,the dataset get split in random manner into testing and training dataset.Thus it depends on how the dataset got split and which samples are training set and which samples are in testing set.
# 
# With K-fold cross validation we can see that the dataset got split into 10 equal parts thus covering all the data into training as well into testing set.This is the reason we got 10 different accuracy score.

# In[ ]:




