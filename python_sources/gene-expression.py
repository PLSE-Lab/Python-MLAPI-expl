#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train.head()


# In[ ]:


pc_test = patient_cancer[patient_cancer.patient > 38].reset_index(drop = True)


# In[ ]:


test = test.reset_index(drop = True)


# In[ ]:


test = pd.concat([pc_test,test],axis = 1)


# In[ ]:


train.head()


# In[ ]:


sample = train.iloc[:,2:].sample(n = 100, axis = 1) ##sample is random sample in dataset
sample["cancer"] = train.cancer 


# In[ ]:


sample


# In[ ]:


sample.describe().round()


# In[ ]:


from sklearn import preprocessing
sample = sample.drop("cancer",axis = 1)
sample.plot(kind = "hist",legend = None, bins = 20,color = 'k')
sample.plot(kind = "kde",legend = None)


# In[ ]:




