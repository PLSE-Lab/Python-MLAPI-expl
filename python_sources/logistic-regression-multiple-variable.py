#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Logistic Regression: Multiclass Classification'''


# In[ ]:


from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[ ]:


plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i])


# In[ ]:


dir(digits)


# In[ ]:


digits.data[0]


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.predict(digits.data[0:5])


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


y_predicted = model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[ ]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




