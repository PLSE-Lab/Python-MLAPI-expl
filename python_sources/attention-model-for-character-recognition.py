#!/usr/bin/env python
# coding: utf-8

# This is a model I built to show myself how to put together an attention model in Keras. I was pleasantly surprised to see it outperform all of the CNN models I had trained previously on this data. This is still a bit primitive, so feel free to experiment with different models using this script as a jumping off point. Have fun.
# 
# Since this model takes a very long time to train, I have never actually run this script as a Kaggle kernel (they will freeze it after 20 minutes). Rather, I have run it locally on my machine, on which it seems to work fine. Please tell me if you find any issues here.

# In[ ]:


import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Convolution2D
from keras.layers import advanced_activations
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape

import matplotlib as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# The first step will be to read in our data and format it correctly as 28x28x1 images. Then the data will be split into a training and validation set so we can get a sense of how the model improves as it trains.

# In[ ]:


trainRaw = pd.read_csv('../input/train.csv');
testRaw = pd.read_csv('../input/test.csv');


# In[ ]:


trainxraw = trainRaw.iloc[:,1:].astype(np.float32);
trx = trainxraw.as_matrix()
trx2 = []
for i in range(trx.shape[0]):
    trx2.append(np.reshape(trx[i],(28,28,1)))
trx3 = np.asarray(trx2)
trx3.shape


# In[ ]:


trainyraw = pd.get_dummies(trainRaw.iloc[:,0]).astype(np.float32);
finalx = testRaw;
trainxraw.shape


# In[ ]:


trainx = trx3[:40000,:]
trainy = trainyraw.iloc[:40000,:]
testx = trx3[40000:,:]
testy = trainyraw.iloc[40000:,:]
trainx = np.asarray(trainx)
trainy = np.asarray(trainy)
testx = np.asarray(testx)
testy = np.asarray(testy)


# Now that we have that set up, let's build the model. I have a 16 3x3 convolutional filter pass over the image to detect edges and simple features. The resulting tensor is then flattened into a matrix of features to be fed into the LSTM layer. Finally, the LSTM layer output in put into a single fully connected layer and then into the output layer. There is definitely room for experimentation here and I'm not entirely satisfied with how this works right now, but it seems to do OK.

# In[ ]:


model = Sequential()

model.add(BatchNormalization(input_shape = trainx.shape[1:]))

print(model.output_shape)
model.add(Convolution2D(16, 3, 3))
model.add(advanced_activations.ELU())
print(model.output_shape)

print(model.output_shape)
model.add(Reshape((model.output_shape[1],-1)))
print(model.output_shape)

model.add(LSTM(16, return_sequences=False))
model.add(advanced_activations.LeakyReLU());

model.add(Dense(32))
model.add(advanced_activations.LeakyReLU());
model.add(BatchNormalization())

model.add(Dense(trainy.shape[1]))
model.add(Activation('softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


batch_size = 5000
nb_epoch = 300
verb=1
data_augmentation = True


# Now for training. As with many image problems, I strongly recommend using some form of data augmentation to introduce noise to the training data. Thankfully, Keras has a class that handles that for us really easily. I will not be flipping the images, since a flipped 6 would be more similar to a 9 than to the intended 6 and so forth, but I did decide to use the zoom, shift, and rotation features. This process does wonders to prevent over-fitting.

# In[ ]:


X_train = trainx;
Y_train = trainy;
X_test = testx;
Y_test = testy;

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              verbose=verb,
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        zoom_range = 0.1,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(trainx)
    dataflow = datagen.flow(trainx, Y_train,
                                     batch_size=batch_size)
    model.fit_generator(dataflow,
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        verbose=verb,
                        validation_data=(X_test, Y_test))


# Unfortunately, the LSTM model takes a long time to train. I get pretty good results after 200-300 epochs of training, but running it that long exceeds the default kernel time limit on Kaggle. Feel free to run this code locally on a machine for a couple hours to replicate my results. I'd be interested to hear any observations people make on this.

# In[ ]:


pred = model.predict_classes(np.asarray(finalx).reshape(-1,28,28,1), batch_size=1000);


# As a quick test to make sure I'm not doing something wrong with my model, I like to check a few of it's predictions by hand to make sure it generally knows what it's doing. Here I checked the first 5 images from the test data. The model correctly predicted all 5 of these images.

# In[ ]:


print(pred[0:5])
#should be 2, 0, 9, 0, 3
for i in range(5):
    img = np.asarray(finalx).reshape(-1,28,28)[i].reshape(28,28);
    plt.pyplot.imshow(img, cmap='Greys_r')
    plt.pyplot.show()


# In[ ]:


finaly = pd.DataFrame([]);
finaly['ImageId'] = np.arange(1,pred.shape[0] + 1);
finaly['Label'] = pred[:];
finaly.to_csv('output.csv', columns = ['ImageId','Label'], index = False);


# The best score I could get with this model was just over 98.9% accuracy on the test data, which as of my uploading this kernel was my best submission yet. That was after running this for ~7 hours on my crumby laptop. I hope to revisit the idea of an attention model on this competition later and see if I can improve on that.

# In[ ]:




