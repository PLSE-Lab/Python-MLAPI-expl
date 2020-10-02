#!/usr/bin/env python
# coding: utf-8

# # <font color='tomato'>Digit Recognizer Uisng CNN</font>

# ## Importing Data and Libraries

# In[ ]:


# !pip install tensorflow-gpu==2.1-rc2


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
sns.set(style='whitegrid')
# DL libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #### <font color = 'tomato'>Version - 1</font>: [Convolutional Neural Network(CNN)](https://www.kaggle.com/syamkakarla/digit-recognizer-using-cnn-99-55?scriptVersionId=32857972)
# #### <font color = 'tomato'>Version - 2</font>:[ Residual Network(ResNet)](https://www.kaggle.com/syamkakarla/digit-recognizer-using-cnn-99-55?scriptVersionId=32918260)

# Important links to learn about CNN and it's implementation:
# 
# 1. [Keras Documentation](https://keras.io/layers/core/)
# 

# ## Read Data

# In[ ]:


df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


# Let's see train data
df.head()


# In[ ]:


df.info()


# ### Countplot

# In[ ]:


plt.figure(figsize=(14, 6))
ax = sns.countplot(data = df, x= 'label', palette = 'Set2')
for p in ax.patches:
    x = 0
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:}'.format(height),
            ha="center") 
    x+=1
plt.show()


# ## Visualizing data w.r.t each class

# In[ ]:


def plot_samples(df, cls = 1):
    fig = plt.figure(figsize = (10,10))
    fig.suptitle('Samples of Class - {}'.format(cls), fontsize=16)
    data = df[df.label == cls]
    for i in range(1, 10):
        fig.add_subplot(3,3,i)
        plt.imshow(data.iloc[i, 1:].values.reshape(28, 28))
        plt.axis('off')
    plt.show()


# In[ ]:


for i in range(1, 10):
    plot_samples(df, cls = i)


# In[ ]:


X = df.iloc[:, 1:].values
y = df.iloc[:, 0]


# In[ ]:


X.shape, y.shape


# ## Normalizating the Data
# 
# why should we do [normalization](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/).

# In[ ]:


# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(X)
# apply transform
normalized = scaler.transform(X)


# In[ ]:


# Reshaping data into 28X28
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes = 10)


# ## Split data into train and validation

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state = 8)

print('Train Shape: {}\nValid Shape: {}'.format(X_train.shape, X_valid.shape))


# ## Data Agumentation
# 
# Best [bolg](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/) to learn about Data Agumentaion.

# In[ ]:


data_gen = ImageDataGenerator(rescale=1./255,
                                rotation_range=10,
                                width_shift_range=.10,
                                height_shift_range=.10,
                                horizontal_flip=False,
                                zoom_range = 0.10)


# ## Build CNN Model 

# In[ ]:


# model = Sequential() 

# model.add(Conv2D(128, (5, 5), input_shape=(28, 28, 1), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(128, (5,5),activation ='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

# model.add(Conv2D(64, (3,3),activation ='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3,3),activation ='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D((2,2), (2,2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(32, (3,3), activation ='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# model.add(Flatten())
# model.add(Dense(256, activation = "relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Dense(10, activation = "softmax"))
# model.summary()


# In[ ]:


from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import elu
# tf.compat.v1.enable_eager_execution(
#     config=None, device_policy=None, execution_mode=None
# )
def Elu_BN(x):
    x = elu(x, alpha=0.5)
    x = BatchNormalization()(x)
    return x

def residual_block(r, filters, kernel_size, name):
    
    y = Conv2D(kernel_size=kernel_size,filters=filters, padding="same", name = name)(r)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = Conv2D(kernel_size=kernel_size, filters=filters, padding="same")(y)
        
    out = Add()([r, y])
    out = ReLU()(out)
    out = BatchNormalization()(out)
    return out


inputs = Input((28, 28, 1))
num_filters = 64

r = BatchNormalization()(inputs)
r = Conv2D(kernel_size=5, strides=1, filters=num_filters, padding="same")(r)
r = Elu_BN(r)
r = residual_block(r, filters = 64, kernel_size = 5, name = 'Residual_Block_1')
r = residual_block(r, filters = 64, kernel_size = 5, name = 'Residual_Block_2')
r = AveragePooling2D(2)(r)
r = residual_block(r, filters = 64, kernel_size = 3,name = 'Residual_Block_3')
r = residual_block(r, filters = 64, kernel_size = 3,name = 'Residual_Block_4')
r = AveragePooling2D(2)(r)
r = Flatten()(r)
r = Dense(100, activation = 'relu')(r)
r = Dropout(0.2)(r)
r = Dense(100, activation = 'relu')(r)
outputs = Dense(10, activation='softmax')(r)

model = Model(name = 'ResidualNet', inputs = inputs, outputs =outputs)

model.summary()


# In[ ]:


# Plot Model
plot_model(model, to_file='model.png', show_shapes=True)


# In[ ]:


model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


## Callbacks
model_check = ModelCheckpoint('best_model.h5', monitor='accuracy', verbose=0, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=0, mode='max', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.000001)


# In[ ]:


history = model.fit_generator(data_gen.flow(X_train, y_train, batch_size=64, seed=1),
                              steps_per_epoch=(len(X_train)*0.8)//64, epochs=100, 
                              validation_data=data_gen.flow(X_valid, y_valid, batch_size=64, seed=2), 
                              validation_steps=(len(X_valid)*0.2)//64,
                             callbacks = [model_check, early, reduce_lr])


# ## Accuracy and Loss Graphs

# In[ ]:


hist_df = pd.DataFrame(data = history.history)
fig = go.Figure()
ind = np.arange(1, len(history.history['accuracy'])+1)
fig.add_trace(go.Scatter(x=ind, mode='lines+markers', y=hist_df['accuracy'], marker=dict(color="dodgerblue"), name="Train_Accyracy"))
    
fig.add_trace(go.Scatter(x=ind, mode='lines+markers', y=hist_df['val_accuracy'], marker=dict(color="darkorange"),name="Validation_Accuracy"))
    
fig.update_layout(title_text='Accuracy', yaxis_title='Accuracy', xaxis_title="Epochs", template="plotly_white")

fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=ind, mode='lines+markers', y=hist_df['loss'], marker=dict(color="dodgerblue"), name="Train_Loss"))
    
fig.add_trace(go.Scatter(x=ind, mode='lines+markers', y=hist_df['val_loss'], marker=dict(color="darkorange"),name="Validation_Loss"))
    
fig.update_layout(title_text='Loss', yaxis_title='Loss', xaxis_title="Epochs", template="plotly_white")

fig.show()


# ## Model Evaluation

# In[ ]:


loss, acc = model.evaluate(data_gen.flow(X_valid, y_valid, batch_size=64, seed=2))
print("Loss: {}\nAccuracy: {}".format(loss, acc))


# In[ ]:


pred = np.argmax(model.predict(X_valid, verbose=1), axis=1)


# ## Confusion Matrix

# In[ ]:


plt.figure(figsize = (15, 15))
sns.heatmap(confusion_matrix(np.argmax(y_valid, axis=1), pred), annot=True, annot_kws={"size": 16}, fmt = 'd') 
plt.show()


# ## Classification Report

# In[ ]:


print(classification_report(np.argmax(y_valid, axis=1), pred))


# ## Sample predicted Images

# In[ ]:


fig = plt.figure(figsize = (10, 10))
for i in range(1, 10):
    fig.add_subplot(3,3,i)
    q = np.random.randint(len(X_valid))
    plt.imshow(X_valid[q].reshape(28, 28))
    plt.title("True: {} --- Pred: {}".format(np.argmax(y_valid[q]), np.argmax(model.predict(X_valid[q].reshape(-1,28,28,1)))))
    plt.axis('off')
plt.show()


# ## Prediction

# In[ ]:


X_test = test.iloc[:, :].values

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)

X_test = X_test.reshape(-1, 28, 28, 1)
print('Test Data Shape: ', X_test.shape)


# In[ ]:


predictions = np.argmax(model.predict(X_test), axis=1)


# In[ ]:


sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sub['Label'] = predictions

sub.head()


# In[ ]:


sub.to_csv('MNIST_Submission_ResNet_5.csv', index=False)


# ---
