#!/usr/bin/env python
# coding: utf-8

# ### <center> In this Notebook, the task is a binary classification of fraudulent behaviour by analysing text </center>
# 
# #### Here is what ive used:
# #### 1. Doc2Vec
# #### 2. textclean
# #### 3. Keras

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Import the data and fill NAs
# 
# - Notice that ived replaced "benefits" with "Adequete benefits".
# - This is because doc2vec will intepret this as a vector. 
# - This vector will contain information on its general context

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df1=pd.read_csv("/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv",index_col="job_id")
df1["location"]=df1["location"].fillna("LOC")
df1["department"]=df1["department"].fillna("DEPART")
df1["salary_range"]=df1["salary_range"].fillna("0-0")
df1["company_profile"]=df1["company_profile"].fillna("No Description")
df1["description"]=df1["description"].fillna("No Description")
df1["requirements"]=df1["requirements"].fillna("No Description")
df1["benefits"]=df1["benefits"].fillna("Adequete benefits")
df1["employment_type"]=df1["employment_type"].fillna("Other")
df1["required_experience"]=df1["required_experience"].fillna("Not Applicable")
df1["required_education"]=df1["required_education"].fillna("Bachelor's Degree")
df1["industry"]=df1["industry"].fillna("None")
df1["function"]=df1["function"].fillna("None")
df1.head()
# df1.industry.value_counts()


# In[ ]:


df1.info()


# In[ ]:


from gensim.models import Doc2Vec
model=Doc2Vec.load("/kaggle/input/doc2vec-english-binary-file/doc2vec.bin")


# ### Ive converted the relavant columns to categorical here

# In[ ]:


#=========[CLEAN DATA]==============#
# !pip install textcleaner==0.4.26
import string

#=========[CLEAN PANDAS]==============#
# employment_type	required_experience	required_education	industry	function
from sklearn.preprocessing import LabelEncoder
df1["location"]=LabelEncoder().fit_transform(df1["location"])
df1["department"]=LabelEncoder().fit_transform(df1["department"])
df1["salary_range"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["employment_type"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["required_experience"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["required_education"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["industry"]=LabelEncoder().fit_transform(df1["salary_range"])
df1["function"]=LabelEncoder().fit_transform(df1["salary_range"])
df1.head()


# ### We text the text cleaner here
# - ive used this text cleaner because it does not remove stop words
# - all words are required in order for doc2vec to capture the relavant context as implied by the name
# - you can try removing stop words and see how it affects accuraccy (from my tests, it reduces accuracy if stopwords are taken out)

# In[ ]:


get_ipython().system('pip install clean-text')
from cleantext import clean

print("#==============[BEFORE]======================#")
print(df1["company_profile"].iloc[0])
print("#==============[AFTER]======================#")
text=clean(df1["company_profile"].iloc[0],no_punct=True)
print(text)


# ### This cell converts the text to the word2doc embeddings
# - ived saved the dataframe in a .npy file as this cell will take awhile to run
# - you can uncomment the lines below to try for yourself

# In[ ]:


def convert_to_embeddings(text):
    try:
        text=clean(text,no_punct=True)
    except:
        text=" "
    return model.infer_vector(text.split())



#==========[IVED SAVED THIS PORTION IN .NPY FILE]=======================#
# df1["title"]=df1["title"].apply(convert_to_embeddings)
# df1["company_profile"]=df1["company_profile"].apply(convert_to_embeddings)
# df1["description"]=df1["description"].apply(convert_to_embeddings)
# df1["requirements"]=df1["requirements"].apply(convert_to_embeddings)
# df1["benefits"]=df1["benefits"].apply(convert_to_embeddings)


# ### We then normalize the data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

swag=np.load("/kaggle/input/df1tonpy1/data.npy",allow_pickle=True)

training_data_text=np.hstack([np.vstack(swag[:,0]),np.vstack(swag[:,4]),np.vstack(swag[:,5]),np.vstack(swag[:,6]),np.vstack(swag[:,7])])
training_data_text.shape

training_data=np.hstack([training_data_text,swag[:,1:3],swag[:,8:]])


training_data=scaler.fit_transform(training_data)


# ### Split the data into test and train set
# - 0.1 split was used

# In[ ]:


X=training_data[:,:-1]
Y=training_data[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


# ### Here we define the model
# - Noticed that ived add in dropout for regularization
# - the starting few codes are meant to reduce EOM errors on my gpu
# - BatchNorm is the most important layer here
#     - Batch Norm will cause the accuracy to go up significantly, 
#     - You can test this by yourself
#     - This is because it reduces the covariate shift problem (Think of a classifier trained on black cats but the test set is on ginger cats)

# In[ ]:


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential

model2=Sequential()
model2.add(Input(shape=(X.shape[1])))
model2.add(BatchNormalization())
model2.add(Dense(128,activation=tf.nn.selu))
model2.add(Dropout(0.5))
model2.add(Dense(64,activation=tf.nn.selu))
model2.add(Dropout(0.2))
model2.add(Dense(32,activation=tf.nn.selu))
model2.add(Dense(1,activation=tf.nn.sigmoid))


model2.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

model2.summary()


# ### The training is slightly overfitted here, but as you can see, it achieves impressive results
# ### -0.99 Accuraccy
# ### -0.98 Val Accuaraccy

# In[ ]:


history=model2.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=80)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

pred=model2.predict(X_test)
pred=np.array([1 if row>=0.5 else 0 for row in pred])
print(classification_report(y_test,pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)
plt.show()

pred=model2.predict(X_train)
pred=np.array([1 if row>=0.5 else 0 for row in pred])
print(classification_report(y_train,pred))
sns.heatmap(confusion_matrix(y_train,pred),annot=True)


# ### Loss function confirms overfitting

# In[ ]:


plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])


# In[ ]:





# In[ ]:




