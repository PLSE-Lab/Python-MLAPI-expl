#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nimport matplotlib.pyplot as plt #for plotting\nimport seaborn as sns\n%matplotlib inline')


# #### Loading Test  and training dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "train=pd.read_csv('../input/train.csv')\ntest=pd.read_csv('../input/test.csv')")


# In[ ]:


train.head(5)


# In[ ]:


train.shape


# In[ ]:


train.columns


# #### Training data target label

# In[ ]:


sns.countplot(train['label'])


# In[ ]:


x_train=(train.ix[:,1:].values).astype('float32')
y_train=(train.ix[:,0].values).astype('int32')


# #### Visualization of training set inputs images

# In[ ]:



# preview the images first
plt.figure(figsize=(12,10))
x,y=10,3
for i in range(30):  
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')
plt.show()


# ### Test data

# In[ ]:


test=(test.values).astype('float32')


# In[ ]:


test.shape


# #### Visualization of test set inputs images

# In[ ]:



# preview the images first
plt.figure(figsize=(12,10))
x,y=10,3
for i in range(30):  
    plt.subplot(y, x, i+1)
    plt.imshow(test[i].reshape((28,28)),interpolation='nearest')
plt.show()


# ### Neural_network  model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.neural_network import MLPClassifier')


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = MLPClassifier()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf.fit(x_train, y_train)')


# #### Prediction

# In[ ]:


output=clf.predict(test)


# In[ ]:


d={"ImageId": list(range(1,len(output)+1)),
                         "Label": output}


# #### Submission

# In[ ]:


sub=pd.DataFrame(d)
sub.to_csv('Submission_digit.csv',index=False)


# In[ ]:




