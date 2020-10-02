#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('../input/data.csv')


# In[ ]:


data.info()


# In[ ]:


# There is no null entries
data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


data.cov()


# In[ ]:


data.std()


# In[ ]:


#let us try to find Categorical variables
data['job'].value_counts()


# In[ ]:


# Find number of Cat..variables for job
len(data['job'].value_counts())


# In[ ]:


data['education'].value_counts()


# In[ ]:


#There are 8 diff cat..variables for education
len(data['education'].value_counts())


# In[ ]:


# We need to get rid of categories by transforming to Numeric Data using Labelencoder in seaborn
data.columns


# In[ ]:


len(data.columns)


# In[ ]:


data.shape


# In[ ]:


labelencoder=LabelEncoder()


# In[ ]:


for col in data.columns:
    data[col]=labelencoder.fit_transform(data[col])


# In[ ]:


#check the data
data.head()


# In[ ]:


#Now we will try to find principal classifiers in data


# In[ ]:


data.shape


# In[ ]:


X=data.iloc[:,0:20]


# In[ ]:


print(X)


# In[ ]:


# y is label in data
y=data.iloc[:,20]


# In[ ]:


print(y)


# In[ ]:


#now Normalize data columns  using standardscaler in seaborn
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


print(scaler)


# In[ ]:


type(scaler)


# In[ ]:


X=scaler.fit_transform(X)


# In[ ]:


print(X)


# In[ ]:


#Principal Component analysis(PCA)
from sklearn.decomposition import PCA
pca = PCA()


# In[ ]:


pca.fit_transform(X)


# In[ ]:


covariance=pca.get_covariance()


# In[ ]:


print(covariance)


# In[ ]:


#it shows covariance of each of 20 variables/features
covariance.shape


# In[ ]:


explained_variance=pca.explained_variance_
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    
    plt.bar(range(20), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


N=data.values    #np array of values of dataframe named "data"
pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y'
                  }


# In[ ]:


label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.show()


# In[ ]:


pca_modified=PCA(n_components=18)
pca_modified.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# In[ ]:


model_LR= LogisticRegression()
model_LR.fit(X_train,y_train)
y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_LR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

