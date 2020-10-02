#!/usr/bin/env python
# coding: utf-8

# # **Use keras to count Sea Lions**
# 
# This kernel is a lite version of my approach.
# 
# [for more information...][1]
# 
# 
#   [1]: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35408

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.feature
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


# **Scale and patch**

# In[ ]:


r = 0.4     #scale down
width = 100 #patch size 


# **Get dot coordinates and cut image to patches :** (thanks to Radu Stoicescu)

# In[ ]:


def GetData(filename):
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    img1 = cv2.GaussianBlur(image_1,(5,5),0)

    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4,axis=2)

    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h,w,d = image_2.shape

    res=np.zeros((int((w*r)//width)+1,int((h*r)//width)+1,5), dtype='int16')

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25: # RED
            res[x1,y1,0]+=1
        elif R > 225 and b > 225 and g < 25: # MAGENTA
            res[x1,y1,1]+=1
        elif R < 75 and b < 50 and 150 < g < 200: # GREEN
            res[x1,y1,4]+=1
        elif R < 75 and  150 < b < 200 and g < 75: # BLUE
            res[x1,y1,3]+=1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1,y1,2]+=1

    ma = cv2.cvtColor((1*(np.sum(image_1, axis=2)>20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(image_2 * ma, (int(w*r),int(h*r)))
    h1,w1,d = img.shape

    trainX = []
    trainY = []

    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainY.append(res[i,j,:])
            trainX.append(img[j*width:j*width+width,i*width:i*width+width,:])

    return np.array(trainX), np.array(trainY)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# **Use only 1 image, split to train/test.**
# 
# In my real approach:
# 
#  - r = 1 to 0.6561 (0.9^0, 0.9^1 ... 0.9^4)
#    
#  - patch size = 300x300
#    
#  - cut whole training set to patches, number of positive(all) vs
#    background(random) = 1 : 3
#    
#  - 95% for training, 5% for validation
# 
#  - data augmentation by flip, rotate, change saturation, brightness, contrast

# In[ ]:


trainX, trainY = GetData("0.jpg")

np.random.seed(1004)
randomize = np.arange(len(trainX))
np.random.shuffle(randomize)
trainX = trainX[randomize]
trainY = trainY[randomize]

n_train = int(len(trainX) * 0.7)
testX = trainX[n_train:]
testY = trainY[n_train:]
trainX = trainX[:n_train]
trainY = trainY[:n_train]

print(trainY.shape, trainY[0])
print(testY.shape, testY[0])


# **Patches looks like :**

# In[ ]:


fig = plt.figure(figsize=(12,12))
for i in range(4):
    ax = fig.add_subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(trainX[i], cv2.COLOR_BGR2RGB))
print(trainY[:4])


# **Keras CNN model, for example**

# In[ ]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(width,width,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='linear'))

#model.summary()


# full version model:
# 
#     initial_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(300,300,3))
#     last = initial_model.output
#     x = Flatten()(last)
#     x = Dense(1024)(x)
#     x = LeakyReLU(alpha=.1)(x)
#     preds = Dense(5, activation='linear')(x)
#     model = Model(initial_model.input, preds)

# **Start training slowly :**

# In[ ]:


optim = keras.optimizers.SGD(lr=1e-5, momentum=0.2)
model.compile(loss='mean_squared_error', optimizer=optim)
model.fit(trainX, trainY, epochs=8, verbose=2)


# **Then speed up :**

# In[ ]:


optim = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=optim)
model.fit(trainX, trainY, epochs=30, verbose=2)


# In[ ]:


# The kernel was killed for running longer than 1200 seconds ...
model.fit(trainX, trainY, epochs=20, verbose=2)


# **Test :**

# In[ ]:


result = model.predict(trainX)
print('Training set --')
print('    ground truth: ', np.sum(trainY, axis=0))
print('  evaluate count: ', np.sum(result*(result>0.3), axis=0).astype('int'))

result = model.predict(testX)
print('Testing set --')
print('    ground truth: ', np.sum(testY, axis=0))
print('   predict count: ', np.sum(result*(result>0.3), axis=0).astype('int'))


# ## Experience ##
# 
# The challenge is scale problem. They distinguish sea lion by size. In different images, one juveniles is larger than adult_females in another.
# 
# I can't handle it well, so I decided to fit LB score:
# 
#  - scale down testing image get better score
#  - more juveniles (less adult_females) get better score
# 
# The final submission is made by:
# 
#  - testing image scale: 0.48
#  - add 50% juveniles, and subtract adult_females with the same amount
#  - add 20% pups
# 
# **Post processing details:**
# 
# These lucky variables are according to patch level regression.
# 
# The relationship between adult_females and juveniles in patches is:
# 
# ![juveniles regression][1]
# 
#  - value in table = average of juveniles# / (adult_females# + juveniles#) @ juveniles number range in patches
# 
#  - r#.# means image scale
# 
#  - *#.# means juveniles increase ratio
# 
#   [1]: http://i.imgur.com/IkucSf6.gif
