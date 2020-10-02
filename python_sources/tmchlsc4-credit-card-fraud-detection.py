#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing all necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as m
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize']=14,8
RANDOM_SEED=42
Labels=['Normal','Fraud']


# In[ ]:


#reading the file
r=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
r.head(10)


# In[ ]:


#analyzing the datatypes of features
r.info()


# In[ ]:


#checking for null values
r.isnull().values.any()


# In[ ]:


#determining inbalanced dataset with data visualization
sns.countplot(r['Class'])
plt.xticks(range(2),Labels)
plt.title('TRANSACTION DISTRIBUTION')
plt.xlabel("Transaction type")
plt.ylabel("Frequency")


# In[ ]:


#keeping class values equal to 1 under fraud and 0 under normal
fraud=r[r['Class']==1]
normal=r[r['Class']==0]
print(fraud.shape,normal.shape)


# In[ ]:


#analyzing fraud and normal dataset
fraud.Amount.describe()


# In[ ]:


normal.Amount.describe()


# In[ ]:


#visualizing the behaviour of amount of both features with respect to transaction 
f,(ax1,ax2)=plt.subplots(2,1,sharex=True)
f.suptitle('Amount per transaction by class')
ax1.set_title('Fraud')
ax1.hist(fraud.Amount,bins=50)
ax2.set_title('Normal')
ax2.hist(normal.Amount,bins=50)
plt.xlabel('Amount(Rs.)')
plt.ylabel('Number of transactions')
plt.yscale('log')
plt.xlim((0,20000))
plt.show()


# In[ ]:


#creating the sample
w=r.sample(frac=0.1,random_state=1)
w.shape


# In[ ]:


r.shape


# In[ ]:


#keeping class values equal to 1 under fraud and 0 under valid in the sample dataset
fraud=w[w['Class']==1]
valid=w[w['Class']==0]
of=len(fraud)/float(len(valid))#calculation of outliers
print(of)
print('Fraud values:{}'.format(len(fraud)))
print('Valid values:{}'.format(len(valid)))


# In[ ]:


#finding correlation
w_=w.corr()
w_features=w_.index
plt.figure(figsize=(30,20))
#plotting the heatmap
sns.heatmap(w[w_features].corr(),annot=True,cmap='viridis')


# In[ ]:


#dividing the sample into dependent and independent features
x=w.iloc[:,:-1]
y=w.iloc[:,-1]
state=np.random.RandomState(42)
xo=state.uniform(0,1,size=(x.shape[0],x.shape[1]))
print(x.shape,y.shape)


# In[ ]:


#making use of IFA,LOF and SVM for determining the outliers and prediction
classifiers={
    "IFA":IsolationForest(n_estimators=100, max_samples=len(x), contamination=of,random_state=state, verbose=0),
    "LOF":LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski',p=2,metric_params=None,contamination=of),
    "SVM":OneClassSVM(kernel='rbf',degree=3, gamma=0.1,nu=0.05,max_iter=-1)
}


# In[ ]:


#printing the classifiaction metrics and prediction for each of the classifiers
for i ,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="LOF":
        y_pred=clf.fit_predict(x)
        sp=clf.negative_outlier_factor_
    elif clf_name=="SVM":
        clf.fit(x)
        y_pred=clf.predict(x)
    else:
        y_pred=clf.fit_predict(x)
        sp=clf.decision_function(x)
    #re-shaping of prediction values,0 for valid and 1 for fraud transactions    
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors=(y_pred!=y).sum()
    print('{}:{}'.format(clf_name,n_errors))
    print("Accuracy is",m.accuracy_score(y,y_pred))
    print("Classification Report:")
    print(m.classification_report(y,y_pred))


# Since IFA has highest accuracy and detection of 77 outliers so it should be used instead of LOF and SVM for prediction and outlier detection.
# 
# 
