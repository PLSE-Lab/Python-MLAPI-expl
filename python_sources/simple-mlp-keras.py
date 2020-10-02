#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# ### Load data

# In[27]:


def load_train_data():
    return pd.read_csv('../input/quickdraw_train.csv', index_col='id')

def load_test_data():
    return pd.read_csv('../input/quickdraw_test_x.csv', index_col='id')


# In[34]:


df_train = load_train_data()
df_train.shape


# In[35]:


df_test = load_test_data()
df_test.shape


# #### Insight into data
# As you can see, there are 2 categories. These categories are not used as labels not are they in the test set. Is there a way to use them? Who knows, feel free to find out for yourself!

# In[17]:


df_train['category'].unique()


# There are 42 different classes for the images, these are the labels that we aim to predict. In the training set there are 1000 images per class. The test set contains 500 images per class, 50% will be used for the public leaderboard, 50% will be used for the final score in the private leaderboard. Be aware that your score can thus change after the submission deadline.

# In[16]:


df_train['subcategory'].unique()


# In[13]:


df_train['subcategory'].value_counts().head()


# In[70]:


def show_sample(df):
    fig = plt.figure(figsize=(16, 16)) 
    unique_samples = df.sample(df.shape[0]).drop_duplicates('subcategory').sort_values('subcategory').reset_index(drop=True)
    images_per_row = 6
    rows = int(np.ceil(unique_samples.shape[0] / images_per_row))
    for i, row in unique_samples.iterrows():
        plt.subplot(rows, images_per_row, i+1)
        im = row[:-2].astype(float).values 
        plt.title(f"{row[-1]} ({row[-2]})")
        plt.imshow(im.reshape([28,28]), cmap='gray', vmin=0, vmax=255, )
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        
def show_drawing(data):
    """
    Show a drawing from either a dataframe or a numpy array
    """
    if type(data) == type(pd.Series()):
        im = data[[f"pix{x}" for x in range(28*28)]].values.reshape((28,28)).astype(float)
        title = ''
        if 'subcategory' in data.index:
            title = f"{data['subcategory']} ({data['category']})"
    elif type(data) == type(np.zeros(1)): 
        im = data.reshape((28,28)).astype(float)
        title = ''
    else:
        print('ERROR: data not suitable: dataframe or numpy array supported')
        return

    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    


# In[74]:


show_sample(df_train)


# ### to show that the images in the test set are similar

# In[75]:


show_drawing(df_test.iloc[0])


# ## Example code for benchmark submission
# ## Simple Keras MLP from MNIST example 
# (https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)

# In[76]:


df_train.head()


# In[122]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

batch_size = 128
num_classes = 42
epochs = 5 #20


# ### Preprocessing data

# In[123]:


df_x = df_train.drop(['category', 'subcategory'], axis=1)
df_y = df_train[['subcategory']]
# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values, test_size=0.1)

# floats!
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#don't forget to normalize your data!
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('x_train shape:', x_train.shape)
print('x_test shape: ', x_test.shape)

# convert class vectors to binary class matrices
le = LabelEncoder()
y_train = le.fit_transform(y_train.flatten())
y_test  = le.transform(y_test.flatten())
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape: ', y_test.shape)


# ### Specify the model architecture
# It contains two dense (feed-forward) neural network layers of each 512 hidden nodes, followed by a relu activation.

# In[124]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Val loss:', score[0])
print('Val accuracy:', score[1])


# ### Now predict on the test set
# 1. Create a prediction
# 2. Save the prediction as submission file using the `create_submissiong()` method.
# 3. Upload on kaggle
# 4. Check your score and win!

# In[143]:


predictions = np.array(model.predict(df_test), dtype='int')
predictions = le.inverse_transform(np.argmax(predictions, axis=1))


# In[151]:


df_submission = df_test.copy()
df_submission['subcategory'] = predictions
df_submission = df_submission[['subcategory']]


# In[153]:


def make_submission_file(df, filename='submission.csv'):
    assert 'subcategory' in df.columns, 'subcategory columns is missing'
    assert df.shape[0] == 21000, 'you should have 21000 rows in your submission file'
    df.to_csv(filename)


# In[154]:


make_submission_file(df_submission, 'mlp_kernel_submission.csv')


# In[ ]:




