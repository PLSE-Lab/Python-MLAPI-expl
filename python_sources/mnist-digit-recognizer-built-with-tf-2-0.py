#!/usr/bin/env python
# coding: utf-8

# Let's start with importing some libraries and functions:
# - Numpy: linear algebra and math operations, here used mostly for reshaping data
# - Matplotlib: data visualization
# - Pandas: Data manipulation, reading/saving data
# - train_test_split from sklearn: splitting dataset training and test subsets

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


training_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)
training_dataset.head()


# # *EXPLORATORY DATA ANALYSIS*

# Let's see proportions of each class in 'label' column:

# In[ ]:


bincount = np.bincount(training_dataset.label)
print(dict(zip(np.nonzero(bincount)[0], bincount)))


# Now, I will split original training dataset to train and test subsets - file 'test.csv' does not contain a 'label' column.
# Parameter 'stratify' will be used in orde to make sure that our target variable has the same proportions in both train and test sets.

# In[ ]:


y = training_dataset['label'].copy()
X = training_dataset.drop(['label'], axis='columns')
del training_dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# In[ ]:


print('X train shape: {}'.format(X_train.shape))
print('y train shape: {}'.format(y_train.shape))
print('X test shape: {}'.format(X_test.shape))
print('y test shape: {}'.format(y_test.shape))


# Let's see if class proportions in both training and test sets are more or less equal:

# In[ ]:


train_class_counter = dict(zip(np.nonzero(np.bincount(y_train))[0], np.bincount(y_train)))
print(train_class_counter)


# In[ ]:


test_class_counter = dict(zip(np.nonzero(np.bincount(y_test))[0], np.bincount(y_test)))
print(test_class_counter)


# Seems like all sets have correct shape and size. Now, let's turn our data into NumPy arrays, in order to plot some digits along with labels. 

# In[ ]:


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

print('X train shape: {}'.format(X_train.shape))
print('y train shape: {}'.format(y_train.shape))
print('X test shape: {}'.format(X_test.shape))
print('y test shape: {}'.format(y_test.shape))


# Let's plot some digits along with labels from the training set.

# In[ ]:


def show_digits(X=X_train, y=y_train, n=10, figsize=(20,3)):
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    for i in range(n):
        ax = axes[i]
        ax.imshow(X[i], cmap='gray_r')
        ax.set_title(str(y[i]), fontsize=12, color='white')
    plt.tight_layout()
    plt.show()


# In[ ]:


show_digits(n=13, figsize=(26, 2))


# Next thing that we should do is normalization of our X-es, both train and test.
# This operation should make our neural network learn much faster.

# In[ ]:


X_train = X_train / 255.
X_test = X_test / 255.


# Let's plot our train digits again to see if inputs were normalized correctly.

# In[ ]:


show_digits(n=13, figsize=(26, 2))


# # *BUILDING A NN MODEL IN TENSORFLOW 2.0 AND KERAS*

# Let's import some essential functions from tensorflow.keras:

# In[ ]:


from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[ ]:


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation='relu'),
    Dropout(0.2),
    Dense(units=64, activation='relu'),
    Dropout(0.15),
    Dense(units=10, activation='softmax')
])

optim = Adam(learning_rate=0.0001, epsilon=1e-8)

model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X_train, y_train, epochs=200, batch_size=1024)


# In[ ]:


model.evaluate(X_test, y_test, verbose=2)


# In[ ]:


metrics = pd.DataFrame(history.history)
metrics.head()


# In[ ]:


epochs = range(1, 201)
f, axes = plt.subplots(1, 2, figsize=(16, 8))
sns.set()
sns.lineplot(x=epochs, y=metrics.loss, ax=axes[0]).set(title='Training loss', xlabel='Epochs', ylabel='Loss value')
sns.lineplot(x=epochs, y=metrics.accuracy, ax=axes[1], color='orange').set(title='Training accuracy', xlabel='Epochs', ylabel='Accuracy value')
plt.show()


# In[ ]:


submission_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submission_data.head()


# In[ ]:


submission_data = np.array(submission_data)
submission_data = submission_data.reshape(-1, 28, 28)
print(submission_data[0])


# In[ ]:


submission_data = submission_data / 255.
submission_classes = model.predict_classes(submission_data)
print(submission_classes[0])


# In[ ]:


submission_classes.shape


# In[ ]:


sub = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sub.head()


# In[ ]:


sample_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
sample_sub.head()


# In[ ]:


ids = [i for i in range(1, sub.shape[0] + 1)]
submission = pd.DataFrame({'ImageId': ids, 'Label': submission_classes})
submission.head()


# In[ ]:


filename = 'Mnist_submission.csv'
submission.to_csv(filename, index=False)

