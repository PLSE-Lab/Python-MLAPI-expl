#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Confusion matrix example.


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


actual = [1, 0, 0, 1, 1, 0, 1, 1, 1, 0] 


# In[ ]:


predicted = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] 


# In[ ]:


results = confusion_matrix(actual, predicted)


# In[ ]:


print ('Confusion Matrix :')
results


# In[ ]:


print ('Accuracy Score :',accuracy_score(actual, predicted) )

