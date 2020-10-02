#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


# In[2]:


train_labels = []
train_samples = []


# total 2100 samples of participants aged between 13 to 100 for drug test.
# 
# 95% of patients above 65 have side effects
# 
# 95% of patients below 65 do not have side effects

# In[3]:


for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)
    
for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)


# In[4]:


train_samples[:10]


# In[5]:


train_labels[:10]


# In[6]:


train_samples = np.array(train_samples)
train_labels = np.array(train_labels)


# In[7]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))


# In[8]:


scaled_train_samples[:10]


# In[9]:


test_labels = []
test_samples = []


# adding 220 test samples, (similar distribution as of train samples)
# 
# in which again 95% of patients above 65 have side effects
# 
# and 95% of patients of below 65 age has no side effects

# In[10]:


for i in range(10):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)
    
for i in range(200):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


# In[11]:


test_labels = np.array(test_labels)
test_samples = np.array(test_samples)


# In[12]:


scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))


# In[13]:


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# In[14]:


model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])


# In[15]:


model.summary()


# In[16]:


model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[17]:


#model.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)


# In[18]:


model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=20, verbose=2, shuffle=True)


# In[21]:


prediction = model.predict(scaled_test_samples, batch_size=10, verbose=0)


# In[23]:


prediction[:10]


# In[24]:


rounded_prediction = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)


# In[25]:


rounded_prediction[:10]


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[28]:


cm = confusion_matrix(test_labels, rounded_prediction)


# In[29]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")
        
    print(cm)
    
    thres = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j] > thres else "black")
        
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')


# In[30]:


cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm, cm_plot_labels, title='confusion matrix')


# In[ ]:




