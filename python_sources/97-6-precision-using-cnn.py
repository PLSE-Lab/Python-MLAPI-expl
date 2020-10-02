#!/usr/bin/env python
# coding: utf-8

# 
# # Malaria Detector
# 
# ### By Ali Shannon
# 
# In this module, I am creating a Malaria detector that uses 27,558 cell images. The dataset is povided by [NIH](https://ceb.nlm.nih.gov/repositories/malaria-datasets/) and made available on [Kaggle](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).
# 
# ---
# 
# I will be using Convolutional Neural Networks so I will need a minimal amount of preprocessing. 
# 
# First, I will resize the images to 92x92 pixels then store them into numpy arrays of 92x92x3 (for RGB)
# 
# Second, I will pass them into a custom network design that has the following configuration:
# 
# 1. 3x3x3 and 5x5x3 kernels convolutional layers with pooling layers, batch normalizations, and dropouts in between.
# 
# 2. Two fully connected layers with L2 regularizations.
# 
# 3. All activations are set to ELU except for the output which uses sigmoid for probability

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import glob, os

np.random.seed(42) # uniform output across runs

tf.logging.set_verbosity(tf.logging.INFO) #turn off annoying error messages


# In[13]:


classes = np.array(os.listdir('../input/cell_images/cell_images'), dtype = 'O')


# In[14]:


images = []; filenames = []

for cl in classes:
    for file in glob.glob('../input/cell_images/cell_images/{}/*.png'.format(cl)): # import all png images from folder 
        im = Image.open(file) 
        filenames.append(im.filename) # label images
        im = im.resize((92,92)) # resize all images
        images.append(im) # store images


# In[15]:


X = np.array([np.array(image) for image in images])

labels = []

for file in filenames:
    for class_name in classes:
        if class_name in file:
            labels.append(class_name)


# In[16]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder().fit(labels)

y = encoder.transform(labels)


# In[17]:


unique_vals, counts = np.unique(labels, return_counts = True)

plt.figure(figsize = (7,4))
plt.bar(unique_vals, counts/counts.sum(), color = 'LightBlue');
vals = np.arange(0, 0.52, 0.1)
plt.yticks(vals, ["{:,.0%}".format(x) for x in vals])
plt.title('Distribution of images')
plt.show()


# Dataset seems pretty balanced, no sampling needed

# In[18]:


plt.figure(figsize=(9,9))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j = np.random.randint(len(X))
    plt.imshow(X[j], cmap=plt.cm.binary)
    plt.xlabel(labels[j] + ' : ' + str(y[j]))
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[20]:


shapes = (92, 92, 3) # input shapes of all images

l2_regularizer = keras.regularizers.l2(0.001) # penalty

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), activation='elu', input_shape=shapes),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.25), 
    keras.layers.Conv2D(32, kernel_size=(5, 5), activation='elu'),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.25),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='elu'),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.15),
    keras.layers.Flatten(),    
    keras.layers.Dense(128, activation = 'elu', kernel_regularizer = l2_regularizer),
    keras.layers.Dense(32, activation = 'elu', kernel_regularizer = l2_regularizer),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dense(1, activation= 'sigmoid')
])
model.compile(optimizer= 'adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])


# In[21]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 15, batch_size= 100)


# In[22]:


loss, accuracy = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)


# In[23]:


from sklearn.metrics import confusion_matrix

conf = confusion_matrix(y_test, [np.round(v) for v in y_pred])

fig, ax = plt.subplots(figsize = (7,7))
ax.matshow(conf, cmap='Pastel1')

ax.set_ylabel('True Values')
ax.set_xlabel('Predicted Values', labelpad = 10)
ax.xaxis.set_label_position('top') 



ax.set_yticks(range(len(classes)))
ax.set_xticks(range(len(classes)))
ax.set_yticklabels(encoder.classes_)
ax.set_xticklabels(encoder.classes_)


for (i, j), z in np.ndenumerate(conf):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.show()


# In[24]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)


# In[25]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print('Precision =', conf[0][0]/(conf[0][0]+conf[1][0]))
print('Recall =', conf[0][0]/(conf[0][0]+conf[0][1]))


# This model optimizes precision, which means it produces less false positives. However, for Malaria detection, we might benefit more from recall which is essentially the measure of sensitivity of our model.
