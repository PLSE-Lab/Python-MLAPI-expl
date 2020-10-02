#!/usr/bin/env python
# coding: utf-8

# # Waste Classification using ResNet50
# 
# This implementation is inspired from [Transfer Learning tutorial](https://www.kaggle.com/dansbecker/transfer-learning) by Dans Becker and another notebook by [Gianluca Pagliara](https://www.kaggle.com/gianpgl/garbage-classification-vgg16/output)

# In[ ]:


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# ## Build the model
# 
# First of all, we build the model by importing Resnet architecture available. In this one, we will use Resnet50 available from `tensorflow.keras.applications`. 

# In[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# We then add the architecture with additional `Dense` layer consisting of `num_class` nodes, according to number of classes in the dataset

# In[ ]:


num_class = 6

model = Sequential([ResNet50(include_top=False,
                             weights='imagenet',
                             pooling='avg'),
                    Dense(num_class, activation='softmax')])

model.layers[0].trainable = False


# After creating the architecture, we compiled the model by providing another required items, such as 
# * the optimizer method to be used
# * loss function used to calculate losses
# * additional metric that we would like to measure

# In[ ]:


from tensorflow.keras.optimizers import SGD


# In[ ]:


lr = 0.01
momentum = 0.001
opt = SGD(learning_rate=lr, momentum=momentum)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ## Load the dataset
# 
# Later then, we could start importing the data to be used in training. In this case, we use `ImageDataGenerator` to gather the data whilst applying preprocessing steps on the data before entering the model.
# 
# `ImageDataGenerator` also provides easier method of reading data by using `flow_from_directory`, which will read data from certain directory. If a subsequent subdirs found inside the directory, `ImageDataGenerator` will consider it as the class label of the data inside said subdir.

# In[ ]:


data_path = "../input/split-garbage-dataset/split-garbage-dataset"

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

img_shape = 224
train_gen = data_generator.flow_from_directory(data_path + '/train',
                                               target_size=(img_shape, img_shape),
                                               batch_size=64,
                                               class_mode='categorical')

val_gen = data_generator.flow_from_directory(data_path + '/valid',
                                             target_size=(img_shape, img_shape),
                                             batch_size=1,
                                             class_mode='categorical',
                                             shuffle=False)


# ## Train the model
# 
# After the model and the data needed are ready, we can start training our model. 
# 
# Keras also provides us with callbacks, a set of functions which will be executed during certain time in training (before an epoch ends, before an iteration finished, etc). These callbacks provide features that could be beneficial in monitoring the performance of model during training phase.
# 
# In this one, we will use `ModelCheckpoint`, a built-in callback that provide automated checkpointing of trained model.

# In[ ]:


from tensorflow.keras.callbacks import ModelCheckpoint


# Inside `ModelCheckpoint`, we could configure the callback according to the behaviour we needed, such as
# * filename of the saved model
# * what metric will the callback monitor to execute the process, this will helps in optimizing the process of saving model
# * do we need to `save_best_only`, which will overwrite saved model with better one (if the monitored metric improves)

# In[ ]:


n_epoch = 50

model_name = 'resnet50_batch64_sgd01m001'
checkpoint = ModelCheckpoint('./' +  model_name + '.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             verbose=1)

history = model.fit_generator(train_gen,
                              steps_per_epoch=train_gen.samples/train_gen.batch_size,
                              validation_data=val_gen,
                              validation_steps=val_gen.samples/val_gen.batch_size,
                              epochs=n_epoch,
                              callbacks=[checkpoint])


# After training the model, `fit_generator` dumps all the recorded metrics inside a `History` object. We could access these metrics by loading the object with index name of the metric we would like to see, such as `accuracy` or `loss`. You can also add `val_` prefix in front of each if you want to access the validation metrics (only if you provided validation data for your training process)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
loss = history.history['loss']


# 1. For better visualization, we can use `matplotlib` to plot the values into graphs

# In[ ]:


plt.plot(range(n_epoch), acc, 'b*-', label = 'Training accuracy')
plt.plot(range(n_epoch), val_acc, 'r', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()


# In[ ]:


plt.plot(range(n_epoch), loss, 'b*-', label = 'Training loss')
plt.plot(range(n_epoch), val_loss, 'r', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()


# ## Test model
# 
# Testing the model are step to measure model capability when facing data not from what it have learned. This step provides us with information on how good the model has learned the data
# 
# In this case, we could also provide the data in the format of `ImageDataGenerator` iterator, similar to training phase

# In[ ]:


data_path = "../input/split-garbage-dataset/split-garbage-dataset"

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = data_generator.flow_from_directory(data_path + '/test',
                                              target_size=(img_shape, img_shape),
                                              batch_size=1,
                                              class_mode='categorical',
                                              shuffle=False)


# If you have the saved model somewhere in your PC, you could load the model back using `load_model` function provided

# In[ ]:


from tensorflow.keras.models import load_model
import numpy as np


# In[ ]:


eval_model = load_model('./' + model_name + '.h5')
eval_model.summary()


# After loaded the model, we can start feeding the model with test data and get the prediction result.

# In[ ]:


y_pred = eval_model.predict_generator(test_gen)
y_pred = np.argmax(y_pred, axis=1)


# To evaluate the prediction result, we can use `classification_report` to get statistic of it and `confusion_matrix` for further analysis of the prediction

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(test_gen.classes, y_pred))


# As we can see from above, the general performance of the model can be seen on the 'accuracy' row. We can also see how good the model performs wrt. each classes in the data, and also which class the model might be bias to.

# In[ ]:


cf_matrix = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(8,5))
heatmap = sns.heatmap(cf_matrix, annot=True, fmt='d', color='blue')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion matrix of model')


# Using the confusion matrix we got from `scikit_learn`, we could plot it using `seaborn`'s heatmap visualization for better insight one the model.
# 
# We can see which part in each classes the model predicted. Using this matrix, we can also analyze which class the model still confuses and which class does the model performs better. With this analysis, we could develop a better training strategies to get a better-performed model in the future
