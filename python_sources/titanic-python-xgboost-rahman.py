#!/usr/bin/env python
# coding: utf-8

# ### Let's Start

# We will solve this problem following some simple steps using **xgboost**.

# At first we import **libraries** we need. (We need more which we will import **later**).

# In[ ]:


import numpy as np
import pandas as pd


# ##### Then It's tiem to import the** data**.
# ####### And we slice the data into independent and dependent variables.

# In[ ]:


dataset = pd.read_csv('../input/titanic/train.csv')
X = dataset.iloc[:,[2,4,5,9,11]].values
y = dataset.iloc[:, 1].values


# Now we take care of missing data.

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:, 2:3])

imputer_2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_2.fit(X[:, [4]])
X[:, [4]] = imputer_2.transform(X[:, [4]])


# now we encode texts

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

X[:, 4] = le.fit_transform(X[:,4])


# now feature scaling (xgBoost needs feature scaling)

# In[ ]:


# Fearure Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)


# let's see how our data table looks

# In[ ]:


X


# now let's split our data into train and test sets.
# (I prefered to know the accuracy before submitting. but we will use only 3% for test)

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state = 0 )


# Then we fit our **classifier**

# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# Now time to test our classifier

# In[ ]:


y_pred = classifier.predict(X_test)


# Let's see the result in confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, classifier.predict(X_train))
print(cm)


# looks good. let's measure the accuracy.

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# Not bad!

# ## Now the fun part

# In[ ]:


dataset1 = pd.read_csv('../input/titanic/test.csv')
X_test = dataset1.iloc[:,[1,3,4,8,10]].values

# No y 


# now let's follow the same steps like before

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test[:,2:3])
X_test[:,2:3] = imputer.transform(X_test[:, 2:3])

imputer_2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_2.fit(X[:, [4]])
X[:, [4]] = imputer_2.transform(X[:, [4]])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_test[:,1] = le.fit_transform(X_test[:,1])
X_test[:, 4] = le.fit_transform(X_test[:,4])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test=sc.fit_transform(X_test)


# let's predict

# In[ ]:


y_pred = classifier.predict(X_test)


# we have our prediction ready. Now let's make a new table with the PassengerID and prediction

# In[ ]:


final = dataset1.iloc[:,0:2].values

for i in range(len(y_pred)):
  final[i,1] = y_pred[i]


# let's export the table to a csv file

# In[ ]:


np.savetxt("titanic_prediction_Rahman.csv", final, delimiter = ',')


# In[ ]:


final


# In[ ]:




