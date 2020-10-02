#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


data=pd.read_csv('../input/insurance.csv')


# In[ ]:


data.head()


# In[ ]:


#The data consists of 30 rows and 13 columns
data.shape


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


#Factorizing the dependent variable
ins_factors=pd.factorize(data['Insurance Type'])

data['Insurance Type']=ins_factors[0]

ins_definations=ins_factors[1]

#creating one hot encoders
data=pd.get_dummies(data,drop_first=True)

data.head()


# In[ ]:


ins_factors[1]


# In[ ]:


data.dtypes,data.shape


# In[ ]:


#dividing the data into train test split
from sklearn.model_selection import train_test_split


# In[ ]:


train,test=train_test_split(data,test_size=0.2,random_state=10)


# In[ ]:


train.shape,test.shape


# In[ ]:


X_train=train.drop('Insurance Type',1)
y_train=train['Insurance Type']


# In[ ]:



X_test=test.drop('Insurance Type',1)
y_test=test['Insurance Type']


# In[ ]:


scaler=StandardScaler()


# In[ ]:


#scaling the independent varaibles so that the model could learn better
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:



# Making RF classifier object using entropy and random state as 10 in RF classifier
classifier=RandomForestClassifier(criterion='entropy',random_state=10)


# In[ ]:


classifier.fit(X_train,y_train)


# In[ ]:



#predicting the test data over out trained model
y_pred=classifier.predict(X_test)


# In[ ]:



#reverse factoring our dependent variable so that the resulst are readable.
ins_reversefactor=dict(zip(range(3),ins_definations))
ins_reversefactor


# In[ ]:


y_test = np.vectorize(ins_reversefactor.get)(y_test)
y_pred = np.vectorize(ins_reversefactor.get)(y_pred)


# In[ ]:


print("This is y_test",y_test)
print("This is y_pred",y_pred)


# In[ ]:


#Making a pandas cross table for visualizing the results
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))


# In[ ]:


prediction=pd.DataFrame(columns=['Actual','Predicted'])


# In[ ]:


prediction['Actual']=y_test
prediction['Predicted']=y_pred
prediction


# In[ ]:


#checking the accuracy of model
print(confusion_matrix(y_test,y_pred))


# In[ ]:


print(accuracy_score(y_test,y_pred)*100)


# In[ ]:


print(classification_report(y_test, y_pred,labels=None, sample_weight=None))


# In[ ]:




