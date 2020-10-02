#!/usr/bin/env python
# coding: utf-8

# **Some exploration of CNN feature extraction from MNIST**
# 
# Whilst I understand that in terms or prediction accuracy CNN are doing an amazing job on image data what really interests me is their ability to extract compact feature sets which replace the original images. The earlier works on NN (e.g. Bishop) represent this compactness in terms of a regularisation method which makes a lot of sense as this will lead to a lower generalisation error on unseen data which is, afterall, the key point in predictive models.
# 
# I have done a bit of playing around with MNIST features, nothing special, but I thought some people might find it interesting.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
#from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.models import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os


# Load in the data - do a bit of checking etc

# In[ ]:


Xtrain = np.loadtxt("../input/train_images_mnist.csv", delimiter=",");
Ytrain = np.loadtxt("../input/train_labels_mnist.csv", dtype = int, delimiter=",");
Xtest = np.loadtxt("../input/test_images_mnist.csv", delimiter=",");
Ytest = np.loadtxt("../input/test_labels_mnist.csv", dtype = int, delimiter=",");

print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)


print("NOW Already normalised look at first image " + str(Xtrain[0,:].max()))
print("Check left upper pixel is always 0  " + str(Xtrain[:,0].max()))

# also now convert to 1 hot encode for NN
Ytrain = to_categorical(Ytrain)
Ytest = to_categorical(Ytest)
print(Ytrain.shape)
print(Ytest.shape)


# In[ ]:


# make a version for cnn
input_shape = (28,28,1)
# note these reshaped inputs will be used in CNN 
trainX = np.array([i for i in Xtrain]).reshape(-1,28,28,1,order='F')
testX = np.array([i for i in Xtest]).reshape(-1,28,28,1,order='F')
trainY = Ytrain
testY = Ytest

for i in range(10):
     plt.imshow(trainX[i,:,:,0],cmap = 'gray')
     str_label = 'Label is :' + str(trainY[i,:].argmax())
     plt.title(str_label)
     plt.show()




# **Start the modelling now**
# 
# Before getting on to CNN I just start by fitting a couple of fully connected NN to the data by treating as 784 input. Naturally these models still do a decent job even though they do not utilise any of the local features derived by convolutions/cross correlation filters. The network can still define boundaries in 784 dim space that the images are embedded in

# In[ ]:


model = Sequential()
model.add(Dense(12, input_dim=784, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(Xtrain, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(Xtest, Ytest))

train_performance = model.evaluate(Xtrain, Ytrain,batch_size=10, verbose=0)
test_performance =  model.evaluate(Xtest, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', train_performance[0])
print('Train accuracy fully connected:', train_performance[1])

print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])


# Pretty good result.  As shown below even a fully connected NN without a hidden layer, essentially now a gradient-based learning logistic regression, still does a pretty good job indicating that the images a largely linearly seperable in high dimensional raw space

# In[ ]:


model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(Xtrain, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(Xtest, Ytest))

glm_train_performance = model.evaluate(Xtrain, Ytrain,batch_size=10, verbose=0)
glm_test_performance =  model.evaluate(Xtest, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', glm_train_performance[0])
print('Train accuracy fully connected:', glm_train_performance[1])

print('Test loss fully connected:', glm_test_performance[0])
print('Test accuracy fully connected:', glm_test_performance[1])


# **CNN**
# 
# We all understand that a CNN will beat these models easily however I am interested in building some CNN's that produce feature representations of the original images that I can easily look at. To do this I build a cnn model that will severly reduce the dimension of the original images via conv and pooling so that each final feature map 
# is 1 pixel only - in this case 32 of them so I can view it as a 32 element vector representign the original image. I connect it to a 1 hidden layer fully connected NN to learn the feature map to output

# In[ ]:



model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
# this final layer will compromise a 1x1 feature map only - 32 of them
model1.add(MaxPooling2D(pool_size=(2, 2)))
# need to connect it to the flatten and fully connected network so it is 
# motivated to learn the right features 
model1.add(Flatten())
# this part is really just a 1 hidden layer MLP from here
model1.add(Dense(12, activation='relu'))
model1.add(Dense(10, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model1.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(testX, testY))
train_performance_cnn = model1.evaluate(trainX, trainY,batch_size=10, verbose=1)
test_performance_cnn =  model1.evaluate(testX, testY,batch_size=10, verbose=1)

#compare to the fully connected model
print('Test loss cnn:', test_performance_cnn[0])
print('Test accuracy cnn:', test_performance_cnn[1])
print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])


# Despite the constraints put on the final feature maps the above network did quite well.
# Now a cnn model - same as above but don't give the fully connected a hidden layer
# meaning it can only learn really well if the feature maps now linearly sperate the 
# data - interestingly very similar accuracy indicating these CNN derived feature maps are doing a great job!!

# In[ ]:


model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model1.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(testX, testY))
train_performance_cnn_no_hid = model1.evaluate(trainX, trainY,batch_size=10, verbose=1)
test_performance_cnn_no_hid =  model1.evaluate(testX, testY,batch_size=10, verbose=1)


print('Test loss cnn no hidden layer in mlp fc part:', test_performance_cnn_no_hid[0])
print('Test accuracy cnn no hidden layer in mlp fc part::', test_performance_cnn_no_hid[1])
print('Test loss cnn:', test_performance_cnn[0])
print('Test accuracy cnn:', test_performance_cnn[1])


# Now look at these feature maps and the images they have now replace. It is interesting to look for similarities between the same digits and difference between different digits

# Now I look at these feature maps in more detail by developing a NN without the flatten and fully connected parts and outputting the feature maps 

# In[ ]:


# layer_name = 'max_pooling2d_16'
# model.get_layer(index=0) specify the index
# Indices are based on order of horizontal graph traversal (bottom-up).
# change get_layer(layer_name) to get_layer(index=5) as it is the 6th layer
# just before flatten

intermediate_layer_model = Model(inputs=model1.input,
                                 outputs=model1.get_layer(index=5).output)
intermediate_output = intermediate_layer_model.predict(trainX)
# as the size of the output is 60000*1*1*32 (32 feature maps of 1*1 for each image)
# we can put each image as a row in a matrix - essentially flattenning it ourselves
# this is the part learnt by the fully connected part
intermediate_output_reshape = intermediate_output.reshape(60000,32)


# In[ ]:


for i in range(20):
    plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,32),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
    
for i in range(20):
    #plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    #str_label = 'Label is :' + str(trainY[i,:].argmax())
    #plt.title(str_label)
    #plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,32),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()


# In[ ]:


# may want to use the imtermediate layer to generate features that are scored by any ML model
intermediate_output_test = intermediate_layer_model.predict(testX)
print(intermediate_output_test.shape)
intermediate_output_test_reshape = intermediate_output_test.reshape(10000,32)
print(intermediate_output_test_reshape.shape)


# Now can train some machine learning models on the features only
# If I now fit fully connected networks to the feature map outputs it stands to reason they should give very similar results to above for the full CNN

# In[ ]:


# glm equivalent
model = Sequential()
model.add(Dense(10, input_dim=32, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(intermediate_output_reshape, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(intermediate_output_test_reshape, Ytest))

glm_train_performance = model.evaluate(intermediate_output_reshape, Ytrain,batch_size=10, verbose=0)
glm_test_performance =  model.evaluate(intermediate_output_test_reshape, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', glm_train_performance[0])
print('Train accuracy fully connected:', glm_train_performance[1])

print('Test loss fully connected:', glm_test_performance[0])
print('Test accuracy fully connected:', glm_test_performance[1])


# In[ ]:


# some hidden layers
model = Sequential()
model.add(Dense(12, input_dim=32, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(intermediate_output_reshape, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(intermediate_output_test_reshape, Ytest))

train_performance = model.evaluate(intermediate_output_reshape, Ytrain,batch_size=10, verbose=0)
test_performance =  model.evaluate(intermediate_output_test_reshape, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', train_performance[0])
print('Train accuracy fully connected:', train_performance[1])

print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])


# The similarity of the above results indicate that the feature set derived by CNN largely linearly seperates the data. To test this try other ML model such as rf and logistic regression

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# becareful for random forest not to use the image data but the flattened data
rfc = RandomForestClassifier(40)
rfc.fit(intermediate_output_reshape, trainY.argmax(axis=1))
print('\n RF train_performance ' + str(rfc.score(intermediate_output_reshape, trainY.argmax(axis=1))))
print('\n RF test_performance ' + str(rfc.score(intermediate_output_test_reshape, testY.argmax(axis=1))))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs',)
logreg.fit(intermediate_output_reshape, trainY.argmax(axis=1))
print('\n Logistic Regression train_performance ' + str(logreg.score(intermediate_output_reshape, trainY.argmax(axis=1))))
print('\n Logistic Regression test_performance ' + str(logreg.score(intermediate_output_test_reshape, testY.argmax(axis=1))))


# Now change the output to an even smaller dimension so the final output is 3d only so that I can view the final features represenations in 3D plots to check for seperability. The fully connected part has no hiddne layers. Naturally the accuracy will go down even further with these restrictions but it makes interpretable features.

# In[ ]:


model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
# add some more if you want
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

# at this point only make a set of 3 1d feature maps
model1.add(Conv2D(3, kernel_size=(3, 3),
                 activation='relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model1.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(testX, testY))
train_performance_cnn = model1.evaluate(trainX, trainY,batch_size=10, verbose=1)
test_performance_cnn =  model1.evaluate(testX, testY,batch_size=10, verbose=1)


print('Train loss cnn:', train_performance_cnn[0])
print('Train accuracy cnn:', train_performance_cnn[1])
print('Test loss cnn:', test_performance_cnn[0])
print('Test accuracy cnn:', test_performance_cnn[1])
print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])


# The accuracy drops but looks quite good considering the restrictions

# In[ ]:


intermediate_layer_model = Model(inputs=model1.input,
                                 outputs=model1.get_layer(index=5).output)
intermediate_output = intermediate_layer_model.predict(trainX)
# as the size of the output is 60000*1*1*3 (3 feature maps of 1*1 for each image)
# we can put each image as a row in a matrix - essentially flattenning it ourselves
# this is the part learnt by the fully connected part
intermediate_output_reshape = intermediate_output.reshape(60000,3)


# Once again can view by eye to look for similarities and differences

# In[ ]:


for i in range(20):
    plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,3),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
 
for i in range(20):
    #plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    #str_label = 'Label is :' + str(trainY[i,:].argmax())
    #plt.title(str_label)
    #plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,3),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
   


# Since each image has now been reduced to a 3 element vector I can look at a 3D scatterplot of the 3D feature maps
# Shows pretty clear structure

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(intermediate_output_reshape[:,0], intermediate_output_reshape[:,1], intermediate_output_reshape[:,2],c=trainY.argmax(axis=1))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


# Now I do some plotly 3D scatter plots here between certain numbers to look again for seperability. The results look nice showing the effectiveness of these 3D feature maps.
# I restricted the plots to the differences between 2 numbers at a time to make the interpretation easier plus I did them in plotly (great plotting software) so the 3D graphs can be rotated to examine the seperability in 3D space
# 

# In[ ]:


import plotly
import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


plotly.offline.init_notebook_mode(connected=True)
plotly.__version__


# In[ ]:


type(intermediate_output_reshape)


# In[ ]:


intermediate_output_reshape.shape


# In[ ]:


target = trainY.argmax(axis=1)


# In[ ]:


print(target.shape)


# In[ ]:


fours = intermediate_output_reshape[target == 4,:]
zeros = intermediate_output_reshape[target == 0,:]
fives = intermediate_output_reshape[target == 5,:]
sixs = intermediate_output_reshape[target == 6,:]


# In[ ]:


print(fours.shape)
print(zeros.shape)


# First the difference between 5's and 6's

# In[ ]:


trace1 = go.Scatter3d(
    x=fives[:,0],
    y=fives[:,1],
    z=fives[:,2],
    mode='markers',
    marker=dict(
        size=5,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=sixs[:,0],
    y=sixs[:,1],
    z=sixs[:,2],
    mode='markers',
    marker=dict(
        size=5,
        symbol= "x",
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='mnist-3d-scatter')


# Now 4's and 0's'

# In[ ]:


trace1 = go.Scatter3d(
    x=fours[:,0],
    y=fours[:,1],
    z=fours[:,2],
    mode='markers',
    marker=dict(
        size=5,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=zeros[:,0],
    y=zeros[:,1],
    z=zeros[:,2],
    mode='markers',
    marker=dict(
        size=5,
        symbol= "x",
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='mnist-3d-scatter1')


# All done. Hope it was interesting, I love the way these CNN's work
