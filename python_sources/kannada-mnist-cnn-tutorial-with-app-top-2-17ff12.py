#!/usr/bin/env python
# coding: utf-8

# 

# # ** 3. An Image Classification Application with CNN** <a id="200"></a>
# <mark>[Return Contents](#0)
# <hr>

# ## **Import Modules** <a id="210"></a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D,LeakyReLU
from tensorflow.keras.optimizers import RMSprop,Nadam,Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# * With this code below you can check if the kernel use GPU or not.

# In[ ]:


tf.test.gpu_device_name()


# ## **Understanding the Data** <a id="220"></a>
# <mark>[Return Contents](#0)

# * We have training and test set CSV files,
# * In order to evaluate the generalization skill of the model we will split our training set into training and validation sets.
# * After training the data, Kaggle will evaluate the final performance of our data with test set predictions.

# * Let's read csv files

# In[ ]:


raw_train = pd.read_csv('../input/Kannada-MNIST/train.csv')
raw_test = pd.read_csv('../input/Kannada-MNIST/test.csv')


# * We have $28$x$28$ dimension handwritten pics.
# * Dataset has been already flattened and has 784-pixel values for each pic.
# * Totaly we have $60000$ pics in training set.

# In[ ]:


raw_train.iloc[[0,-1],[1,-1]] # First and last values of dataset


# * It is important to know the distribution of data according to the labels they have.
# * This data set is __homogeneously__ distributed as you see below.

# In[ ]:


num = raw_train.label.value_counts()
sns.barplot(num.index,num)
numbers = num.index.values


# * __If the data wasn't homogeneously__ distributed what would we do?
#     1. Then we could use data augmentation techniques to generate new data for low quantity labels,
#     2. Or if we have enough data we can discard some high quantity labels

# #### **Image of Handwritten Character** <a id="2"></a>
# <mark>[Return Contents](#0)
# <hr>
#     
#  * An overview of a picture
#  * You can change the $num$ variable to see other numbers.

# In[ ]:


num=6
number = raw_train.iloc[num,1:].values.reshape(28,28)
print("Picture of "+ str(num) + "in Kannada style")
plt.imshow(number, cmap=plt.get_cmap('gray'))
plt.show()


# ## **Data Preprocessing** <a id="230"></a>
# ### **Normalizing Data** <a id="231"></a>
# <mark>[Return Contents](#0)
# <hr>
# * What is normalizing? Normalization means that adjusting values measured on different scales to a notionally common scale.
# * Why should you normalize the data?  With a normalized data weight values reach optimum value faster.
# * On image processing applications generally we normalize data to 0-1 scale with dividing data to 255.
# * Because each pixel in every sample of training set has integer values from 0 to 255.
# * In order to normalize training set data, we need to convert x to float type.

# In[ ]:


x = raw_train.iloc[:, 1:].values.astype('float32') / 255
y = raw_train.iloc[:, 0] # labels


# ### **Cross Validation - Training- Validation Set Split** <a id="232"></a>
# <mark>[Return Contents](#0)
#     
# * In order to measure the generalization ability of the model, we train the data with the training set and make the model arrangement according to the error value in the validation set. In addition, we determine the final performance of the model with the test set. 
# * The reason for using the test set on the final evaluation is the model would have a bias on the validation set because we developed the model according to the validation set performance. So a kind of overfitting on the validation set is formed.
# * This Kernel is prepared on a Kaggle competition dataset. They give us a training set for training the model and a test set without labels for prediction. As we don't have the labels we don't know the final performance of the test set until we submit our predictions.
# * To evaluate the model we need to split our training set into training and validation set.
#     
# * For I prefer to split
#     * Training set - $\%80$
#     * Validation set - $\%20$
# 

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42) 


# ### **Reshape Data to Fit Model** <a id="233"></a>
# <mark>[Return Contents](#0)

# > * In order to feed the CNN model we need to reshape our $54000$x$784$ flatten image data to $54000$x$28$x$28$x$1$ dimensions.;

# In[ ]:


x_train.shape


# In[ ]:


x_train = x_train.reshape(-1, 28, 28,1)
x_val = x_val.reshape(-1, 28, 28,1)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# * You can leave one unknown dimension as -1.
# * Whenever numpy has -1 value on reshape method it will calculate the dimension which denoted with -1 automatically.

# ## **Building and Training a CNN Model** <a id="250"></a>
# <mark>[Return Contents](#0)
# <hr>
# * On Keras Sequential Networks there is three-stage for training building, compiling and fitting the model.
#     
# #### **Model - Build** <a id="251"></a>
# 
#     
# * On building stage you specify the architecture of the model mainly.
# * You can decide the [Filter](#110) size and [Padding](#150) type you will use on [Convolution](#110) operations and add [Pooling](#150), Batch Normalization, Dropout, activation function layers with build section.

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(128, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(128, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),    
    
    tf.keras.layers.Conv2D(256, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(256, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),##
    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])


# ### **Model - Compile** <a id="252"></a>
# <mark>[Return Contents](#0)
#     
# * On Compile Section we specify the loss function, optimizer algorithm and metric to use for evaluating the model.

# In[ ]:


optimizer = RMSprop(learning_rate=0.002,###########
    rho=0.9,
    momentum=0.1,
    epsilon=1e-07,
    centered=True,
    name='RMSprop')
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# #### **Compile ->Loss Function ->Categorical Crossentropy** <a id="253"></a>
# <mark>[Return Contents](#0)
#     
# * The Categorical Crossentropy Loss Function computes between network predictions and target values for multiclass classification problems.
# 
# * The loss is calculated using the following formula;
# \begin{equation}
# Loss = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} Y^i_klog(\hat{Y^i_k})(1-Y^i_k)log(1-\hat{Y^i_k}) \\
# \end{equation}
# 
# where $k$ demonstrates class, $i$ demonstrates sample number, $\hat{Y_c}$ is the predicted value, $Y_c$ is the ground truth value, $m$ is the sample number in a batch and $K$ is the total number of classes.
# 
# * Why do we use log? Because cross-entropy function penalize bigger difference more and smaller difference less as mentioned below.
# 
# [![cross-entropy.png](https://i.postimg.cc/qvwNHJpL/cross-entropy.png)](https://postimg.cc/687WdN22)

# #### **Compile -> Optimizer** <a id="255"></a>
# <mark>[Return Contents](#0)
# * To optimize the weight values of the network we should choose an optimizer algorithm. 
# * In the first lessons of artificial learning, the gradient descent algorithm is taught as the optimization algorithm used in backpropagation. 
# * Over time, the gradient descent algorithm was developed and the algorithms that achieved faster and more accurate results were obtained. 
# * Some of them are ADAGRAD, ADAM, ADAMAX, NADAM and RMSPROP. 
# * These algorithms use techniques such as adaptive learning rate and momentum to achieve the global minimum.
# * If you want to take a more detailed look at Gradient Descent algorithms, [here](https://ruder.io/optimizing-gradient-descent/) is a very nice overview article written by [Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/).
# 
# * __As you see below__ while Stochastic Gradient Descent (SGD) which is a basic gradient descent algorithm cannot escape the saddle point, more advanced algorithms escape the saddle point __at different speeds.__
# 

# ![CNN.jpg](https://ruder.io/content/images/2016/09/saddle_point_evaluation_optimizers.gif)
# ![CNN.jpg](https://ruder.io/content/images/2016/09/contours_evaluation_optimizers.gif)

# In[ ]:


batch_size = 1024
num_classes = 10
epochs = 58


# #### **On-the-Fly Data Augmentation** <a id="257"></a>
# <mark>[Return Contents](#0)
# * On classification tasks on image datasets data augmentation is a common way to increase the generalization of the model. 
# 
# [![Plot-of-Images-Generated-with-a-Rotation-Augmentation.png](https://i.postimg.cc/m2JBtVsK/Plot-of-Images-Generated-with-a-Rotation-Augmentation.png)](https://postimg.cc/3dXPqXWZ)
# 
# * With the ImageDataGenerator on Keras, we can handle this objective easily.
# * [Here](https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/) is a comprehensive and inspiring article about data augmentation and ImageDataGenerator written by [Adrian Rosebrock](https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/) who is the author of PyImageSearch a very instructive web-site about computer vision.
# * __By changing__ some of the properties of the image from the code below, __you can observe what changes are happening in the dataset__. 
# * With this observation, you can roughly specify the range you should choose.

# In[ ]:


# An observation code for our dataset
datagen_try = ImageDataGenerator(rotation_range=15,
                             width_shift_range = 0.25,
                             height_shift_range = 0.25,
                             shear_range = 25,
                             zoom_range = 0.4,)
# fit parameters from data
datagen_try.fit(x_train)
# configure batch size and retrieve one batch of images
for x_batch, y_batch in datagen_try.flow(x_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		plt.subplot(330 + 1 + i)
		plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
	# show the plot
	plt.show()
	break


# In[ ]:


datagen_train = ImageDataGenerator(rotation_range = 10,
                                   width_shift_range = 0.25,
                                   height_shift_range = 0.25,
                                   shear_range = 10,
                                   zoom_range = 0.1,
                                   horizontal_flip = False)

datagen_val = ImageDataGenerator() 


step_train = x_train.shape[0] // batch_size
step_val = x_val.shape[0] // batch_size

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau( 
    monitor='loss',    # Quantity to be monitored.
    factor=0.25,       # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=2,        # The number of epochs with no improvement after which learning rate will be reduced.
    verbose=1,         # 0: quiet - 1: update messages.
    mode="auto",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 
                       # in the max mode it will be reduced when the quantity monitored has stopped increasing; 
                       # in auto mode, the direction is automatically inferred from the name of the monitored quantity.
    min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.
    cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.
    min_lr=0.00001     # lower bound on the learning rate.
    )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)


# ### **Model - Fit** <a id="259"></a>
# <mark>[Return Contents](#0)

# In[ ]:


history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=len(x_train)//batch_size,
                              epochs=epochs,
                              validation_data=(x_val, y_val),
                              validation_steps=50,
                              callbacks=[learning_rate_reduction, es],
                              verbose=2)


# ## **Evaluation of the Model** <a id="270"></a>
# ### **Accuracy and Loss Curves** <a id="271"></a>
# <mark>[Return Contents](#0)

# * On classification, accuracy metric is calculated as below;
# 
# \begin{equation}
# classification~accuracy = \frac{correct~predictions}{total~predictions} * 100 \\
# \end{equation}

# In[ ]:


tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# ### **Test Set Accuracy Score** <a id="272"></a>
# <mark>[Return Contents](#0)

# In[ ]:


model.evaluate(x_val, y_val, verbose=2);


# ### **Confusion Matrix** <a id="273"></a>
# <mark>[Return Contents](#0)

# In[ ]:


y_predicted = model.predict(x_val)
y_grand_truth = y_val
y_predicted = np.argmax(y_predicted,axis=1)
y_grand_truth = np.argmax(y_grand_truth,axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_grand_truth, y_predicted)


# * If 90% of the data set is cat image and 10% is dog image, your accuracy will be 90% even if you estimate the entire test set as a cat.
# * But in another aspect, the model's success in predicting dogs is 0%.
# * In this context, accuracy may not always give us realistic information about the actual performance of the model.
# * The confusion matrix shows how confused your classification model is for which classes by detailing the relationship between actual class and predicted class.
# * If there is an anomaly something like above mentioned, you can specify the problem with confusion matrix and improve accuracy by various methods like adding more data for a specific class, etc.

# In[ ]:


f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.1, linecolor="purple", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Grand Truth")
plt.show()


# ### **F1 Score Calculation** <a id="274"></a>
# <mark>[Return Contents](#0)

# * As mentioned above, [accuracy](#271) gives a general idea about the performance of the model but does not provide any information about the model's trends.
# * In classification algorithms, it is important to analyze the false predictions of the model.
# * There are 2 kinds of false predictions; 
#   * __Predicted as "1" but Ground Truth is "0" (False Positive)__
#   * __Predicted as "0" but Ground Truth is "1" (False Negative)__
# 
# [![f1.png](https://i.postimg.cc/1X1h0vM1/f1.png)](https://postimg.cc/Hczhd4f6)
# * The F1 score creates a success performance metric, taking into account both of these incorrect prediction performances as well as the true positive predictions
# * Since our problem has more than 2 classes, we calculated the F1 score for each class on a one-to-all basis.
# * [Here](https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1) is a nice article on how to calculate the F1 score in multiple classes, supported by examples.

# In[ ]:


scores = np.zeros((10,3))
def calc_F1(num):
  TP = cm[num,num]
  FN = np.sum(cm[num,:])-cm[num,num]
  FP = np.sum(cm[:,num])-cm[num,num]
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  F1_score = 2*(recall * precision) / (recall + precision)
  return precision, recall, F1_score
for i in range(10):
   precision, recall, F1_score = calc_F1(i)
   scores[i,:] = precision, recall, F1_score
scores_frame = pd.DataFrame(scores,columns=["Precision", "Recall", "F1 Score"], index=[list(range(0, 10))])


# In[ ]:


f, ax = plt.subplots(figsize = (4,6))
ax.set_title('Number Scores')
sns.heatmap(scores_frame, annot=True, fmt=".3f", linewidths=0.5, cmap="PuBu", cbar=True, ax=ax)
bottom, top = ax.get_ylim()
plt.ylabel("")
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# ### **Evaluate with Another Dataset** <a id="275"></a>
# <mark>[Return Contents](#0)

# In[ ]:


raw_dig = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
raw_dig.head()
x_dig = raw_dig.iloc[:, 1:].values.astype('float32') / 255
y_dig = raw_dig.iloc[:, 0].values

x_dig = x_dig.reshape(-1,28,28,1)
y_dig = to_categorical(y_dig)
model.evaluate(x_dig, y_dig, verbose=2)


# ### **Submit for Competition** <a id="276"></a>
# <mark>[Return Contents](#0)

# In[ ]:


sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
raw_test_id=raw_test.id
raw_test=raw_test.drop("id",axis="columns")
raw_test=raw_test / 255
test=raw_test.values.reshape(-1,28,28,1)
test.shape


# In[ ]:


sub=model.predict(test)     ##making prediction
sub=np.argmax(sub,axis=1) ##changing the prediction intro labels

sample_sub['label']=sub
sample_sub.to_csv('submission.csv',index=False)


# # **Conclusion** <a id="276"></a>
# <mark>[Return Contents](#0)
# <font color='green'>
# * Please do not hesitate to comment and ask questions.
# * If you found it useful, I would appreciate it if you upvote    
#     
#     [![smile.jpg](https://i.postimg.cc/0jVh4z64/smile.jpg)](https://postimg.cc/fS0HtTf7)
# 
