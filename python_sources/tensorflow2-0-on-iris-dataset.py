#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# # 1. Quick introduction:
# - Tensorflow 2.0 with the use of Keras high level API is a good approach of on many machine learning and deep learning problem
# - This notebook shows how to use tf.GradientTape() to optimize model explicitly

# # 2. Load dataset:
# - Iris dataset is to be used
# - We need to use 4 features (SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm) to predict the target variable Species 

# In[ ]:


# Load data
Iris = pd.read_csv("../input/iris/Iris.csv")


# In[ ]:


Iris.head()


# In[ ]:


# split data and target
data = Iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
target = Iris['Species']


# In[ ]:


print('Number of records: ', len(data))


# # 2. Basic EDA

# In[ ]:


# check null value: there is no missing value
data.isnull().sum()


# In[ ]:


target.isnull().sum()


# In[ ]:


# The 4 features are likely to be at the same scale
data.describe()


# In[ ]:


# distribution of each feature
sns.boxplot(data=data)


# In[ ]:


# distribution of each feature within each class of target
plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.violinplot(x=data['SepalLengthCm'], y=target)
plt.subplot(2,2,2)
sns.violinplot(x=data['SepalWidthCm'], y=target)
plt.subplot(2,2,3)
sns.violinplot(x=data['PetalLengthCm'], y=target)
plt.subplot(2,2,4)
sns.violinplot(x=data['PetalWidthCm'], y=target)


# - It looks like PetalLengthCm and PetalWidthCm are the most useful features because their distributions divide into 3 seperated parts corresponding to 3 classes of target. Let's use correlation heatmap to check if our assumption is right

# In[ ]:


# label encode target to numerical
le = LabelEncoder()
target_label_encoded = le.fit_transform(target)
# create a temporary variable to concat data and encoded target
temp = data.copy()
temp['Species'] = target_label_encoded
temp.corr()['Species'].sort_values(ascending=False)


# - Our assumption when looking at the plot seems right

# In[ ]:


sns.heatmap(temp.corr())


# In[ ]:


# scatter plot on 2 features
sns.scatterplot(x=data['SepalLengthCm'], y=data['PetalLengthCm'], hue=target)


# In[ ]:


sns.scatterplot(x=data['SepalWidthCm'], y=data['PetalWidthCm'], hue=target)


# # 3. Training and Evaluating

# In[ ]:


# split data set into train and test set
X_train, X_test, y_train, y_test = train_test_split(data, target_label_encoded, test_size=0.2)


# In[ ]:


# create tf dataset for training and validation
# because tf dataset needs input as type numpy ndarray so we need convert pandas df into numpy array by using .value
train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_test)).batch(32)


# In[ ]:


# define model architecture
def build_model():
    # Our model will have 1 input layer of 4 nodes, 1 Dense layer with 32 nodes 
    # and 1 output layer 3 nodes corresponding to 3 species classes
    x = tf.keras.layers.Input(shape=(4,))
    dense1 = tf.keras.layers.Dense(32, activation='relu')(x)
    y = tf.keras.layers.Dense(3, activation='softmax')(dense1)
    model = tf.keras.models.Model(inputs=x, outputs=y)

    return model

model = build_model()
model.summary()


# These code is referenced to https://www.tensorflow.org/tutorials/quickstart/advanced

# In[ ]:


# define loss functions
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# optimizer adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# tensor that holds metric values such as loss value and accuracy value
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# In[ ]:


@tf.function
def train_step(features, targets):
    with tf.GradientTape() as tape:
        # compute the predictions
        predictions = model(features)
        # compute loss between predictions and targets
        loss = loss_object(targets, predictions)
    # use gradient tape to track the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # use gradients computed above to update trainable weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(targets, predictions)
    
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


# In[ ]:


EPOCHS = 15
list_train_losses = []
list_train_accs = []
list_test_losses = []
list_test_accs = []

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for features, targets in train_ds:
        train_step(features, targets)

    for features, targets in test_ds:
        test_step(features, targets)
    
    list_train_losses.append(train_loss.result())
    list_test_losses.append(test_loss.result())
    list_train_accs.append(train_accuracy.result())
    list_test_accs.append(test_accuracy.result())
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))


# In[ ]:


# plot training, testing loss curves
plt.plot(np.arange(EPOCHS), list_train_losses)
plt.plot(np.arange(EPOCHS), list_test_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')


# In[ ]:


# plot training, testing accuracy curves
plt.plot(np.arange(EPOCHS), list_train_accs, 'r--')
plt.plot(np.arange(EPOCHS), list_test_accs, 'b--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')


# In[ ]:





# In[ ]:




