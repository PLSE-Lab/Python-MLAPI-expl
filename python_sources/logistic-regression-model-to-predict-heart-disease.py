#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# **Importing dataset**

# In[ ]:


dataset = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# **Taking care of Missing Values**

# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X)
X = imputer.transform(X)


# **Splitting Dataset**

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# **Feature Scaling for Improving model Performance**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# **Training the logistic regression model**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train,y_train)


# **Predicting test set results**

# In[ ]:


y_pred = classifier.predict(x_test)


# **Evaluating Confusion Matrix**

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# **Accuracy of Model**

# In[ ]:


from sklearn.metrics import accuracy_score
print('Accuracy of my model on testing set :' ,accuracy_score(y_test,y_pred))

