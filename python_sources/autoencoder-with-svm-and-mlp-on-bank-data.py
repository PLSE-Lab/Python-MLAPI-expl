#!/usr/bin/env python
# coding: utf-8

# # Categorical Variable Embeddings

# In this activity we will build two different models
# 
#     For first model, categorical attributes are convered in to dummy numeric variables
#     For second model, categorical attributes not convered in to numberic. Each level is given unique number starting with zero and categorical embedding is used 

# In[ ]:


import os
print(os.listdir("../input"))


# #### Load the requied libraries

# In[ ]:


import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, concatenate, Flatten, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# #### Read the data

# In[ ]:


df = pd.read_csv("../input/bank-full-1540270465813.csv",na_values=["NA"," ","","?"],delimiter = ';' )


# #### Understand the data

# In[ ]:


df.shape


# Look at first few records

# In[ ]:


df.head()


# Summary statistics 

# In[ ]:


df.poutcome.value_counts()


# In[ ]:


df.y[df.poutcome == 'success'].value_counts()


# In[ ]:


df.y[df.duration > 319].value_counts().value_counts()


# In[ ]:


df.pdays[df.pdays== -1].sum()


# In[ ]:


df.describe(include='all')


# ## Type of all the attributes

# In[ ]:


df.dtypes


# ## Checking for NULL or NA values

# In[ ]:


df.isna().sum()


# In[ ]:


cat_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome","y"]


# In[ ]:


num_attr = ['age', 'balance', 'day', 'duration', 'campaign','pdays','previous']


# ##### Following two cells are for explination purpose

# Convert the attributes to appropriate type

# In[ ]:


for i in cat_cols:
    df[i] = df[i].astype("category")


# In[ ]:


df.dtypes


# Summary Statistics

# In[ ]:


df.describe(include = "all")


# #### Missing value imputation

# In[ ]:


df.isnull().sum()


# In[ ]:


names = df.columns


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# ### Transforming all the catagorical attributes into numerical using LabelEncoder 

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


LE = LabelEncoder()
for i in cat_cols:
    df[i] = LE.fit_transform(df[i])  


# In[ ]:


for i in num_attr:
    df[i] = df[i].astype("int64")


# #### Select only numeric independent attributes

# In[ ]:


df.dtypes


# In[ ]:


num_ind_attr = df[num_attr]


# In[ ]:


num_ind_attr.head()


# In[ ]:


scaled_num_ind_attr = num_ind_attr.values


# In[ ]:


scaled_num_ind_attr


# #### Select categorical attributes

# In[ ]:


cat_attr_names = cat_cols


# In[ ]:


cat_ind_attr_names = cat_attr_names

print(cat_ind_attr_names)

cat_tar_attr_names = ['y']


# In[ ]:


df.y.shape


# #### Dummification 
#     
#     For first model, convert independetn categorical attributes to numeric using dummificcation
#     
#     Target categorical attribut has more than two level. For both the models, target categorical attribute is convert to numeric use dummification.

# Convert independent categorical attribute to numeric using dummification 

# In[ ]:


cat_ind_attr = pd.get_dummies(df[cat_ind_attr_names]).values


# In[ ]:


cat_ind_attr


# In[ ]:


cat_ind_attr.shape


# Convert targeet categorical attribute to numeric using dummification

# In[ ]:


df[cat_tar_attr_names] = df[cat_tar_attr_names].astype('category')


# In[ ]:


cat_tar_attr = pd.get_dummies(df[cat_tar_attr_names]).values


# In[ ]:


cat_tar_attr


# In[ ]:


job_attr = df.job.values
marital_attr = df.marital.values
education_attr = df.education.values
default_attr = df.default.values 
housing_attr = df.housing.values 
loan_attr = df.loan.values 
contact_attr = df.contact.values 
month_attr = df.month.values 
poutcome_attr = df.poutcome.values 


# In[ ]:


job_levels = np.size(np.unique(job_attr, return_counts=True)[0])
marital_levels = np.size(np.unique(marital_attr, return_counts=True)[0])
education_levels = np.size(np.unique(education_attr, return_counts=True)[0])
default_levels = np.size(np.unique(default_attr, return_counts=True)[0])
housing_levels = np.size(np.unique(housing_attr, return_counts=True)[0])
loan_levels = np.size(np.unique(loan_attr, return_counts=True)[0])
contact_levels = np.size(np.unique(contact_attr, return_counts=True)[0])
month_levels = np.size(np.unique(month_attr, return_counts=True)[0])
poutcome_levels = np.size(np.unique(poutcome_attr, return_counts=True)[0])


# In[ ]:


scaled_num_ind_attr_train, scaled_num_ind_attr_test, cat_ind_attr_train, cat_ind_attr_test, Y_train, Y_test, Y_trainS, Y_testS              = train_test_split(scaled_num_ind_attr,
                                                         cat_ind_attr, 
                                                         cat_tar_attr,
                                                         df.y,
                                                         test_size=0.3, random_state=123) 


# ### Build First Model

# In[ ]:


Y_train.shape


# In[ ]:


X_train = np.hstack((scaled_num_ind_attr_train, cat_ind_attr_train))
X_test = np.hstack((scaled_num_ind_attr_test, cat_ind_attr_test))


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clf = SVC()
clf.fit(X_train,Y_trainS)


# In[ ]:


clf.score(X_train,Y_trainS)


# In[ ]:


clf.score(X_test,Y_testS)


# ## Building a Perceptron with normal data

# In[ ]:


model = Sequential()
##model.add(Dense(250, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(125, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, Y_train, epochs=20, verbose=1,batch_size=80)


# In[ ]:


model.evaluate(X_test, Y_test, )


# In[ ]:


model.metrics_names


# In[ ]:


p = model.predict(X_test)
p[:50]


# ## Building a Autoencoder

# In[ ]:


encoding_dim = 9 
actual_dim = X_train.shape[1]


# In[ ]:


input_img = Input(shape = (actual_dim,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim,activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(actual_dim,activation='sigmoid')(encoded)


# In[ ]:


autoencoder = Model(input_img,decoded)
print(autoencoder.summary())


# In[ ]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


autoencoder.fit(X_train, X_train, epochs=100, batch_size=80)


# In[ ]:


encoder = Model(input_img,encoded)
print(encoder.summary())


# ## Extracting encoded features

# In[ ]:


X_train_nonLinear_features = encoder.predict(X_train)
X_test_nonLinear_features = encoder.predict(X_test)


# In[ ]:


X_train1=np.concatenate((X_train, X_train_nonLinear_features), axis=1)
X_test1=np.concatenate((X_test, X_test_nonLinear_features), axis=1)


# ## Building a SVM on Encoded features

# In[ ]:


clf = SVC()
clf.fit(X_train1,Y_trainS)


# In[ ]:


clf.score(X_train1,Y_trainS)


# In[ ]:


clf.score(X_test1,Y_testS)


# ## Building a MLP over Encoded features

# In[ ]:


model = Sequential()
##model.add(Dense(250, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, input_dim=X_train1.shape[1], activation='relu'))
model.add(Dense(4, input_dim=X_train1.shape[1], activation='relu'))
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train1, Y_train, epochs=20, verbose=1,batch_size=50)


# In[85]:


model.evaluate(X_test1, Y_test, )


# ## Now we can see the power of embaddings in Neural Network.

# In[ ]:


import matplotlib.pyplot as plt

history = model.fit(X_train1, Y_train, validation_split=0.25, epochs=30, batch_size=80, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

