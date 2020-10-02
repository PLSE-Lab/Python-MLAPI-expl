#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense,BatchNormalization, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy import stats
import csv
import gc


# We will assess the working of neural nets on the MNIST digits dataset using three approaches :
# 1. **Using the standard neural network architecture.**
# 2. **Using the convolutional neural network architecture.**
# 3. **Adding Data Augmentation and then using the convolutional neural network architecture.**

# First we need to load and process the data, as the data provided in not in form of images but rather in form of a dataframe. Lets check what data is provided in the dataframe.

# In[ ]:


# defining data directory paths
train_dir = "../input/train.csv"
test_dir = "../input/test.csv"

df = pd.read_csv(train_dir)
df.info()


# * The dataframe contains image labels defining which class the particular image belongs to, also the columns ranging pixel0 - pixel738 contains a value for that particular pixel.
# * The image is of the size 28x28

# **Defining Labels :** extracting labels from dataframe -> converting to numpy array -> converting to one-hot format.
# 

# In[ ]:


labels = df["label"].values.tolist() # extracting labels from the database and converting it into a list
labels = np.array(labels)

n_classes = len(set(labels)) # defining number of classes

labels = keras.utils.to_categorical(labels) # converting the labels to one-hot format


# **Training data :** Extracting from dataframe -> converting to list -> numpy array -> scaling into range 0-1.

# In[ ]:


df_train = df.drop(["label"], axis = 1) # extracting the image data
data = df_train.values.tolist() # converting image data to list
data = np.array(data)
data = data.astype('float32')/255.0 # converting data into range 0-1


# Plotting an image from each class to get insight on image data.
# Plotting 5 images from each class

# In[ ]:


dataframes_i = []
for i in range(10):
    tempdf = None
    tempdf = df[df["label"]==i].drop(["label"], axis = 1)
    temp = tempdf.values.tolist()
    dataframes_i.append(temp[0:5])
    
fig = plt.figure(figsize = (8,20)) #defining figure
def plot_images(image, index):
    fig.add_subplot(10,5, index)
    plt.axis("on")
    plt.tick_params(left = False, bottom=False, labelbottom=False, labelleft = False,)
    plt.imshow(image, cmap = 'Greys')
    return

index = 1
for i in dataframes_i:
    for j in i:
        x = np.array(j)
        x = x.reshape(28,28)
        plot_images(x, index)
        index += 1
plt.show()


# Checking shape of training data and labels

# In[ ]:


print("Training data shape = " + str(data.shape))
print("Training labels shape = " + str(labels.shape))


# **Defining Standard Neural Network** : 
# Now that we have the training data and labels we will define a simple neural network.

# In[ ]:


gen_model = Sequential()
gen_model.add(Dense(784, activation = 'relu', input_shape = (784,)))
gen_model.add(Dense(512, activation = 'relu'))
gen_model.add(Dense(264, activation = 'relu'))
gen_model.add(Dense(10, activation = 'softmax'))
print("STANDARD NEURAL NETWORK MODEL :-")
gen_model.summary()


# In[ ]:


gen_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])


# In[ ]:


gen_model_hist = gen_model.fit(data, labels, batch_size = 32, epochs = 5, validation_split = 0.1)


# Now plotting the training and validation accuracy plots :

# In[ ]:


plt.plot(gen_model_hist.history["acc"])
plt.plot(gen_model_hist.history["val_acc"])
plt.title("Training vs Validation Accuracy")
plt.legend(["Training","Validation"], loc = 'lower right')
plt.show()


# In[ ]:


del gen_model, gen_model_hist
gc.collect()


# **CONVOLUTIONAL NEURAL NETWORK**

# Now we will use the convolutional neural network architecture to train the model, for this we need to modify our data as :
# * reshaping the training data into (n, 28, 28, 1) as there is only one channel and image is of size 28x28.

# In[ ]:


X_train_cnn = data.reshape(len(data), 28, 28, 1)


# **Defining the CNN model :**

# In[ ]:


cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size = [3,3], activation = 'relu', input_shape = (28,28,1)))
cnn_model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPool2D(pool_size = [2,2], strides = 2))
cnn_model.add(Conv2D(128, kernel_size = [3,3], activation = 'relu'))
cnn_model.add(MaxPool2D(pool_size = [2,2], strides = 2))
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation = 'relu'))
cnn_model.add(Dense(10, activation = 'softmax'))
print("CONVOLUTIONAL NEURAL NETWORK MODEL :-")
cnn_model.summary()


# Compiling and fitting the data in the model.

# In[ ]:


cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
cnn_model_hist = cnn_model.fit(X_train_cnn, labels, batch_size = 32, epochs = 6, validation_split = 0.1)


# Plotting the model metrics:

# In[ ]:


plt.plot(cnn_model_hist.history["acc"])
plt.plot(cnn_model_hist.history["val_acc"])
plt.title("Training vs Validation Accuracy (CNN Model)")
plt.legend(["Training","Validation"], loc = 'lower right')
plt.show()


# In[ ]:


del cnn_model, cnn_model_hist
gc.collect()


# **ADDING DATA AUGMENTATION AND ENSEMBLING MODELS:**

# Now we will train 7 different classifiers on the cnn architecture and then use them for preditions.
# 1. We will build 7 different models.
# 2. Train those models on different splits of data.
# 3. Use those models for predictions as : we take the mode from the predictions.

# As we are adding the data augmentation and training models several times, this process will take some time.

# In[ ]:


data_aug = ImageDataGenerator(featurewise_center = False,
                             samplewise_center = False,
                             featurewise_std_normalization = False,
                             samplewise_std_normalization = False,
                             zca_whitening = False,
                             rotation_range = 10,
                             zoom_range = 0.1,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             horizontal_flip = False,
                             vertical_flip = False)


# In[ ]:


# defining several models
models_ensemble = []
for i in range(7):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = [3,3], activation = 'relu', input_shape = (28,28,1)))
    model.add(Conv2D(64, kernel_size = [3,3], activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size = [2,2], strides = 2))
    model.add(Conv2D(128, kernel_size = [3,3], activation = 'relu'))
    model.add(MaxPool2D(pool_size = [2,2], strides = 2))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    models_ensemble.append(model)


# In[ ]:


# defining training routine
model_histories = []
i = 1
for model in models_ensemble:
    xtrain, xtest, ytrain, ytest = train_test_split(X_train_cnn, labels, test_size = 0.07)
    print("Model " +str(i)+ " : ",end="")
    model_history = model.fit_generator(data_aug.flow(xtrain, ytrain, batch_size = 64), epochs = 1, verbose = 1, validation_data = (xtest, ytest), steps_per_epoch = xtrain.shape[0])
    model_histories.append(model_history)
    i += 1


# **Now we will use all our models to make predictions**
# * Make predictions from one model -> add to list -> repeat for all models -> then we will find the mode of predictions -> use it as final prediction.

# In[ ]:


# import and preprocess test data
testdata = pd.read_csv(test_dir)
testdata = testdata.values.tolist()
testdata = np.array(testdata)
testdata_reshaped = testdata.reshape(testdata.shape[0], 28, 28, 1)
testdata_reshaped = testdata_reshaped.astype('float')/255.0

def make_predictions_final_model(curr_model):
    prediction_array = curr_model.predict_on_batch(testdata_reshaped)
    predictions = [np.argmax(i) for i in prediction_array]
    return predictions


# * Firstly we will make predictions on each model and then save it into lists, this will create 5 different prediction lists.
# * Then we will use these lists to make new list exclusively for each image which will contain predictions from each model.
# * Finally we will find the mode from each list and then append it to a new list which will be the final predictions for our model.

# In[ ]:


predictions_ensemble = [] 

# Make predictions using seperate models
for model in models_ensemble:
    curr_predictions = make_predictions_final_model(model)
    predictions_ensemble.append(curr_predictions)

prediction_per_image = []
# Make a list of predictions for a particular image 
for i in range(len(predictions_ensemble[0])):
    temppred = [predictions_ensemble[0][i], predictions_ensemble[1][i], predictions_ensemble[2][i], predictions_ensemble[3][i], predictions_ensemble[4][i], predictions_ensemble[5][i], predictions_ensemble[6][i]]
    prediction_per_image.append(temppred)
    
# Find the maximum occuring element in the array (list)
prediction_per_image = np.array(prediction_per_image)
modes = stats.mode(prediction_per_image, axis = 1)

# append the modes to the final prediction list
final_predictions = []      
for i in modes[0]:
    final_predictions.append(i[0])


# Creating the output csv file.

# In[ ]:


final_csv = []
csv_title = ['ImageId', 'Label']
final_csv.append(csv_title)
for i in range(len(final_predictions)):
    image_id = i + 1
    label = final_predictions[i]
    temp = [image_id, label]
    final_csv.append(temp)

print(len(final_csv))

with open('submission_csv_aug.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(final_csv)
file.close()

