#!/usr/bin/env python
# coding: utf-8

# ### Classifikation of a cardiovascular desease with a Neuronal Network
# This is my first kernel here on kaggle. 
# In this kernel I only drop some data with medically impossible value. There is few data analysis or visualisation, because there are some great examples in other kernels. Furthermore I am not experienced in data engeneering  (maybe you can give me some hints ;) and some things i tried beforehand had no positive impact towards the accuracy of my network.

# In[ ]:


# Imports
import numpy as np 
import pandas as pd 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from keras.optimizers import *
from keras.initializers import *
from keras.models import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Get data
df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', delimiter=';')
df.drop('id', axis=1, inplace=True)
df.head()


# In[ ]:


df.describe()


# As you can see the min and max values of 'ap_hi' and 'ap_lo' are medically not possible.

# In[ ]:


df[df['ap_lo'] >= df['ap_hi']]


# Higher diastolic than systolic blood pressure is impossible, too.
# So let's remove these.

# In[ ]:


df.drop(df[df["ap_lo"] > df["ap_hi"]].index, inplace=True)
df.drop(df[df["ap_lo"] <= 30].index, inplace=True)
df.drop(df[df["ap_hi"] <= 40].index, inplace=True)
df.drop(df[df["ap_lo"] >= 200].index, inplace=True)
df.drop(df[df["ap_hi"] >= 250].index, inplace=True)
df[['ap_lo', 'ap_hi']].describe()


# Now it is time to get our X and Y, to split the data in train and test sets and to scale it.

# In[ ]:


X = df.drop('cardio', axis=1)
Y = df['cardio']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.transform(x_test)

# Split train set in train and validation set:
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=0)


# Defining the NN is the next step.

# In[ ]:


# Silence warnings
import warnings as w
w.simplefilter('ignore')


def create_model():
    # Hyperparameter:
    init_w = glorot_uniform(seed=0)
    loss = "binary_crossentropy"
    optimizer = Adadelta()
    
    # Defining the model:
    model = Sequential()

    model.add(Dense(50, kernel_initializer=init_w, input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.1))

    model.add(Dense(25, kernel_initializer=init_w))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(rate=0.1))

    model.add(Dense(12, kernel_initializer=init_w))
    model.add(LeakyReLU())

    model.add(Dense(1, kernel_initializer=init_w))
    model.add(Activation("sigmoid"))
    
    model.summary()
    
    # Training
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"])

    return model


# In[ ]:


nn = create_model()
nn.fit(
    x=x_train,
    y=y_train,
    verbose=2,
    epochs=50,
    batch_size=256,
    validation_data=[x_valid, y_valid])


# In[ ]:


# Testing
test_score = nn.evaluate(x_test, y_test)
print("Testing Acc:", test_score[1])


# We get an accuricy on the test set between: **73.3% - 73.7%**

# In[ ]:


y_pred = nn.predict(x_test)
cm = confusion_matrix(y_test, y_pred.round())
print("Confusion Matrix:", "\n", cm)


# In[ ]:


tpr, fpr, threshold = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
print("AUC-score:", auc_score)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(tpr, fpr)
plt.show()


# To me this looks pretty ok.
# Hints and advice would be welcome. (I evaluated the hyperparameter that worked best for me beforehand with some runs of a GridSearch. This is not in the kernel, because of the required runtime. If you want me to share my code for the GridSearch or KFold, feel free to tell me. I am going to put it online then.)

# Thank you for visiting my very first Kernel :)
