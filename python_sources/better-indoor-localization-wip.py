#!/usr/bin/env python
# coding: utf-8

# <h1>Overview</h1>
# 
# The goal of this kernel is to improve upon previous work done to use bluetooth iBeacon's in order to accurately locate a person in the building. This has various useful applications such as giving a user their location in a mall via phone app, making sure people aren't going into restricted areas, monitoring foot traffic and flow patterns and many other creative applications
# 
# <h2>Motivation</h2>
# I thought this was an interesting dataset and semi-related to my work on smart buildings, but originally glossed over it. I viewed the starter kernel that was available and read the research paper that backed it and saw various low hanging fruit I thought could be plucked so decided to throw my hat into the ring
# 
# <h2>Major Improvements/Goals</h2>
# 
# <h3>1.</h3>The first major thing I wanted to address was the original author created two separate models for predicting the X and Y coordinates. To me it seemed to make much more sense to do both in one single model. 
# 
# <h3>2.</h3>The second thing I wanted to address was the validation strategy. This is time series data, the way it was captured was moving the bluetooth device around the building while pausing at each coordinate for 3 seconds and gathering average values for each period. Doing a random shuffle and split means that you very likely trained on two points that were before and after the point you are trying to predict. In lieu of this method I wanted to implement a valid split where I predicted on the last 20% of the data so it is truly unscene. 
# 
# <h3>3.</h3>The third major change I wanted to make was redoing the problem from using a MLP to a CNN. The way the original work handles the inputs is it takes all 13 sensors and then creates a different input neuron for each possible value of that sensor. So this means we are dealing with 13 * number of possible readings input neurons and these values don't hold much transfer value. If you were bring this setup to a new building zero knowledge would transfer. The CNN solves this problem by turning the inputs into an image rather than a really long vector. I converted the inputs to basically be an image by using each x, y represent a 10x10m box like shown in the image provided in the original project. I also manually entered the positions of the sensors. When a sensors value went up that corresponding grid value would go up. In this way we deal with each 10x10 box like a pixel and can apply a CNN to the problem. This captures the spatial understanding and means that in theory we could move a sensor, remove a sensor, add a sensor or apply a pretrained model to an entirely different building given that the scale is the same and sensor behave similarly. This system is much more generalizable. 
# 
# <h3>4.</h3>The fourth major change I implemented was unsupervised pretraining. In this dataset we have 4 times as much unlabeled data as we do labeled data. It would be great if there was some way to use this unlabeled data in order to create some understanding of the building and then finetune a model on top of that given our other labeled data. This is very important because it would potentially eliminate the need to physically visit a building and measure out x, y coordinates of a bluetooth device while also measuring signal strength. This will help the solution be cheaper to provide because a technician would not need to come in and take measurements and calibrate the system. This is still highly experimental, but the method I was using was autoencoder. With this architecture I was trying to start with the original input, compress it down to a much smaller representation, ideally down to our final output goal of just an x, y coordinate and then try to recreate the original sensor readings after the information has been compressed. Initially results are promsing but also kind of finnicky and fragile. I'm sure with a more highly tuned system performance could be better. 
# 
# 

# In[ ]:


# %matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ## euclidean distnce between two points:

# In[ ]:


def l2_dist(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dx = dx ** 2
    dy = dy ** 2
    dists = dx + dy
    dists = np.sqrt(dists)
    return np.mean(dists), dists


# ## Load labeled dataset:

# In[ ]:


path='../input/iBeacon_RSSI_Labeled.csv'
x = read_csv(path, index_col=None)


# In[ ]:


x.head(5)


# The first thing I did was view the histograms of the sensor values. It seems like it is common for the sensors to either have some or near max signal or no signal at all. In this case higher numbers means better signal and -200 is minimum signal, meaning there is basically no connection. I will just print a few of them but there are 13 sensors in total in this dataset

# In[ ]:


for col in x.columns[10:]:
    x.hist(column = col)


# Inputs originally came in this annoying format with the X axis represented by letters rather than numbers so I had to go in and correct this.

# In[ ]:


def fix_pos(x_cord):
    x = 87 - ord(x_cord.upper())
    return x


# In[ ]:


path='../input/iBeacon_RSSI_Labeled.csv'
x = read_csv(path, index_col=None)
x['x'] = x['location'].str[0]
x['y'] = x['location'].str[1:]
x.drop(["location"], axis = 1, inplace = True)
x["x"] = x["x"].apply(fix_pos)
x["y"] = x["y"].astype(int)


# Doing a train test split with no shuffling in order to preserve time series information

# In[ ]:


y = x.iloc[:, -2:]
x = x.iloc[:, 1:-2]
train_x, val_x, train_y, val_y = train_test_split(x,y, test_size = .2, shuffle = False)


# Creating a simple MLP model like in the original work, but this time with X, Y both predicted by a single model. 

# In[ ]:


from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
def create_deep(inp_dim):
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(50, input_dim=inp_dim, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer=Adam(.001), metrics=['mse'])
    return model


# In[ ]:


es = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto', restore_best_weights=True)
model = create_deep(train_x.shape[1])
hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=1000, batch_size=1000,  verbose=0, callbacks = [es])


# Making predictions and measuring how far off the predictions are

# In[ ]:


preds = model.predict(val_x)
l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
print(l2dists_mean)


# Using the same CDF code from the orignal work we can visualize the error. We can see that this model is already significantly better than the original. Specifically it gets to perfect accuracy within 6 meters instead of 10

# In[ ]:


sortedl2_deep = np.sort(l2dists)
prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
fig, ax = plt.subplots()
lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
plt.title('CDF of Euclidean distance error')
plt.xlabel('Distance (m)')
plt.ylabel('Probability')
plt.grid(True)
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.savefig('Figure_CDF_error.png', dpi=300)
plt.show()
plt.close()


# I put together a plotly animation with the room image in the background in order to visualize the actual object position and the predicted position. The image is not perfectly to scale and lined up in the background but it is reasonably close. I could not find a perfect method in order to get those lined up correctly. It is interesting that the predicted dot seems to have some stickyness to the sensors

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
import numpy as np
from PIL import Image

image = Image.open("../input/iBeacon_Layout.jpg")
init_notebook_mode(connected=True)

xm=np.min(val_y["x"])-1.5
xM=np.max(val_y["x"])+1.5
ym=np.min(val_y["y"])-1.5
yM=np.max(val_y["y"])+1.5

data=[dict(x=[0], y=[0], 
           mode="markers", name = "Predictions",
           line=dict(width=2, color='green')
          ),
      dict(x=[0], y=[0], 
           mode="markers", name = "Actual",
           line=dict(width=2, color='blue')
          )
      
    ]

layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),
            title='Moving Dots', hovermode='closest',
            images= [dict(
                  source= image,
                  xref= "x",
                  yref= "y",
                  x= -3.5,
                  y= 22,
                  sizex= 36,
                  sizey=25,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")]
            )

frames=[dict(data=[dict(x=[preds[k, 0]], 
                        y=[preds[k, 1]], 
                        mode='markers',
                        
                        marker=dict(color='red', size=10)
                        ),
                   dict(x=[val_y["x"].iloc[k]], 
                        y=[val_y["y"].iloc[k]], 
                        mode='markers',
                        
                        marker=dict(color='blue', size=10)
                        )
                  ]) for k in range(int(len(preds))) 
       ]    
          
figure1=dict(data=data, layout=layout, frames=frames)          
iplot(figure1)


# My abysmal attempts at making the data lineup

# In[ ]:


# data=[dict(x=[1, 1], y=[16, 16], 
#            mode="markers", name = "Predictions",
#            line=dict(width=2, color='green')
#           ),
#       dict(x=[2, 2], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[3, 3], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[4, 4], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[5, 5], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[6, 6], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[7, 7], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[8, 8], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[9, 9], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[10, 10], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[11, 11], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[12, 12], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[13, 13], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[14, 14], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[15, 15], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[16, 16], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[17, 17], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[18, 18], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[19, 19], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[20, 20], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[20, 20], y=[16, 16], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[20, 20], y=[17, 17], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[20, 20], y=[18, 18], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           ),
#       dict(x=[20, 20], y=[19, 19], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           )
#       ,
#       dict(x=[20, 20], y=[0, 0], 
#            mode="markers", name = "Actual",
#            line=dict(width=2, color='blue')
#           )
      
#     ]
# layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),
#             yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),
#             title='Moving Dots', hovermode='closest',
#             images= [dict(
#                   source= image,
#                   xref= "x",
#                   yref= "y",
#                   x= -3.5,
#                   y= 22,
#                   sizex= 36,
#                   sizey=25,
#                   sizing= "stretch",
#                   opacity= 0.5,
#                   layer= "below")]
#             )
# figure1=dict(data=data, layout=layout)          
# iplot(figure1)


# Loading in the data again for round two. 

# In[ ]:


path='../input/iBeacon_RSSI_Labeled.csv'
x = read_csv(path, index_col=None)
x['x'] = x['location'].str[0]
x['y'] = x['location'].str[1:]
x.drop(["location"], axis = 1, inplace = True)
x["x"] = x["x"].apply(fix_pos)
x["y"] = x["y"].astype(int)
y = x.iloc[:, -2:]
x = x.iloc[:, 1:-2]


# I logged the placement of each iBeacon and then created an array of size 25x25 full of minimum signal. From there I plugged in the sensors values in the arrays specific x,y coordinates

# In[ ]:


x["b3001"].values.shape


# In[ ]:


img_x = np.zeros(shape = (x.shape[0], 25, 25, 1, ))
beacon_coords = {"b3001": (5, 9), 
                 "b3002": (9, 14), 
                 "b3003": (13, 14), 
                 "b3004": (18, 14), 
                 "b3005": (9, 11), 
                 "b3006": (13, 11), 
                 "b3007": (18, 11), 
                 "b3008": (9, 8), 
                 "b3009": (2, 3), 
                 "b3010": (9, 3), 
                 "b3011": (13, 3), 
                 "b3012": (18, 3), 
                 "b3013": (22, 3),}
for key, value in beacon_coords.items():
    img_x[:, value[0], value[1], 0] -= x[key].values/200
    print(key, value)
# img_x = (img_x) / 200
train_x, val_x, train_y, val_y = train_test_split(img_x, y, test_size = .2, shuffle = False)


# In[ ]:


img_x[103, :, :, 0].mean()


# In[ ]:


#what one sample looks like 
img_x[1, :, :, 0]


# What the array looks like as an image. You can see that there really is only signal strength coming from the one iBeacon. Scaled up so some antialiasing. Should just be one invididual pixel being lit up. 

# In[ ]:


img = Image.fromarray(img_x[19, :, :, 0] * 255, "L")
img.resize((250, 250))


# In[ ]:


from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Conv2DTranspose


# I am choosing to use RMSE so the far errors are penalized more heavily. 

# In[ ]:


from keras import backend as K
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# I use a semi-standard CNN model in order to predict the x, y coordinates given the signal strength. 

# In[ ]:


inputs = Input(shape=(train_x.shape[1], train_x.shape[2], 1))

# a layer instance is callable on a tensor, and returns a tensor
x = Conv2D(3, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(inputs)
x = MaxPooling2D(2)(x)
x = Conv2D(6, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)
x = MaxPooling2D(2)(x)
x = Conv2D(12, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)
predictions = Dense(2, activation='relu')(Flatten()(x))

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=Adam(.001),
              loss=rmse,
              metrics=['accuracy'])
model.summary()
hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=1000, batch_size=200,  verbose=0, callbacks = [es])


# Slightly worse performance, but I'm sure with a more tuned system it could at least equal the previous method and I would still argue this is a better setup because it is possible to generalize. 

# In[ ]:


preds = model.predict(val_x)
l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
print(l2dists_mean)


# Something is obviously not quite right with the model/input normalization because the loss curve is fragile and spiky. Partially to be expected from such a small training set, but could probably be fixed

# In[ ]:



plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


sortedl2_deep = np.sort(l2dists)
prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
fig, ax = plt.subplots()
lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
plt.title('CDF of Euclidean distance error')
plt.xlabel('Distance (m)')
plt.ylabel('Probability')
plt.grid(True)
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.savefig('Figure_CDF_error.png', dpi=300)
plt.show()
plt.close()


# In[ ]:


from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
import numpy as np
from PIL import Image

image = Image.open("../input/iBeacon_Layout.jpg")
init_notebook_mode(connected=True)

xm=np.min(val_y["x"])-1.5
xM=np.max(val_y["x"])+1.5
ym=np.min(val_y["y"])-1.5
yM=np.max(val_y["y"])+1.5

data=[dict(x=[0], y=[0], 
           mode="markers", name = "Predictions",
           line=dict(width=2, color='green')
          ),
      dict(x=[0], y=[0], 
           mode="markers", name = "Actual",
           line=dict(width=2, color='blue')
          )
      
    ]

layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),
            title='Moving Dots', hovermode='closest',
            images= [dict(
                  source= image,
                  xref= "x",
                  yref= "y",
                  x= -3.5,
                  y= 22,
                  sizex= 36,
                  sizey=25,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")]
            )

frames=[dict(data=[dict(x=[preds[k, 0]], 
                        y=[preds[k, 1]], 
                        mode='markers',
                        
                        marker=dict(color='red', size=10)
                        ),
                   dict(x=[val_y["x"].iloc[k]], 
                        y=[val_y["y"].iloc[k]], 
                        mode='markers',
                        
                        marker=dict(color='blue', size=10)
                        )
                  ]) for k in range(int(len(preds))) 
       ]    
          
figure1=dict(data=data, layout=layout, frames=frames)          
iplot(figure1)


# Loading in the unlabeled data for pretraining

# In[ ]:


path='../input/iBeacon_RSSI_Unlabeled.csv'
x_un = read_csv(path, index_col=None)
# x['x'] = x['location'].str[0]
# x['y'] = x['location'].str[1:]
x_un.drop(["location", "date"], axis = 1, inplace = True)
# x["x"] = x["x"].apply(fix_pos)
# x["y"] = x["y"].astype(int)
# y = x.iloc[:, -2:]
# x = x.iloc[:, 1:-2]


# In[ ]:


img_x = np.zeros(shape = (x_un.shape[0], 25, 25, 1, ))
beacon_coords = {"b3001": (5, 9), 
                 "b3002": (9, 14), 
                 "b3003": (13, 14), 
                 "b3004": (18, 14), 
                 "b3005": (9, 11), 
                 "b3006": (13, 11), 
                 "b3007": (18, 11), 
                 "b3008": (9, 8), 
                 "b3009": (2, 3), 
                 "b3010": (9, 3), 
                 "b3011": (13, 3), 
                 "b3012": (18, 3), 
                 "b3013": (22, 3),}
for key, value in beacon_coords.items():
    img_x[:, value[0], value[1], 0]  -= x_un[key].values/200
    print(key, value)
train_x_un, val_x_un = train_test_split(img_x, test_size = .2, shuffle = False)


# Now I will do something wacky like setup an autoencoder to try to compress and then recreate our original signal readings

# In[ ]:


from keras.layers import GaussianNoise, AveragePooling2D, SpatialDropout2D
inputs = Input(shape=(train_x.shape[1], train_x.shape[2], 1))

# a layer instance is callable on a tensor, and returns a tensor
x = Conv2D(24, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(inputs)
x = MaxPooling2D(2)(x)
x = Conv2D(24, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)
x = MaxPooling2D(2)(x)
x = Conv2D(24, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)
x = Conv2DTranspose(24, kernel_size=(3,3),strides = (2,2), activation='relu', padding = "valid", data_format="channels_last")(x)
x = Conv2DTranspose(16, kernel_size=(3,3),strides = (2,2), activation='relu', padding = "valid", data_format="channels_last")(x)
x = Conv2DTranspose(8, kernel_size=(3,3),strides = (2,2), activation='relu', padding = "valid", data_format="channels_last")(x)
x = Conv2DTranspose(1, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)

# This creates a model that includes
# the Input layer and three Dense layers
model2 = Model(inputs=inputs, outputs=x)
model2.compile(optimizer=Adam(.0001, clipnorm = .5, clipvalue = .5),
              loss='mse',
              metrics=['accuracy'])
model2.summary()
hist = model2.fit(x = train_x_un, y = train_x_un, validation_data = (val_x_un,val_x_un),epochs=15, batch_size=10, verbose=2, callbacks = [es])


# Now I will chop off the decoder part of the autoencoder and then add on a new dense layer to make predictions. 

# In[ ]:


predictions = Dense(8, activation='relu')(Flatten()(model2.layers[5].output))
predictions = Dense(2, activation = 'relu')(predictions)


# In[ ]:


model3 = Model(inputs=model2.input, outputs=predictions)


# In[ ]:


model3.summary()


# I am only going to train the last layer to start. This will help prevent the auto encoders being immediately being washed out because the last layer is uninitialized

# In[ ]:


for layer in model3.layers[:-2]:
    layer.trainable = False
model3.compile(optimizer=Adam(.001, clipnorm = .5, clipvalue = .5),
              loss=rmse,
              metrics=['accuracy'])


# In[ ]:


hist = model3.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=500, batch_size=50,  verbose=0, callbacks = [es])


# This gives us pretty abysmal results right off the bat, but I will try training the rest of the layers now.

# In[ ]:


preds = model3.predict(val_x)
l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
print(l2dists_mean)


# In[ ]:



plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Unfreeze the rest of the layers and voila. Even better performance than without the pretraining on unlabeled data

# In[ ]:


for layer in model3.layers[:-2]:
    layer.trainable = True
model3.compile(optimizer=Adam(.001, clipnorm = .5, clipvalue = .5),
              loss=rmse,
              metrics=['accuracy'])
hist = model3.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=500, batch_size=50,  verbose=0, callbacks = [es])
preds = model3.predict(val_x)
l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))
print(l2dists_mean)


# In[ ]:



plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


sortedl2_deep = np.sort(l2dists)
prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
fig, ax = plt.subplots()
lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
plt.title('CDF of Euclidean distance error')
plt.xlabel('Distance (m)')
plt.ylabel('Probability')
plt.grid(True)
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.savefig('Figure_CDF_error.png', dpi=300)
plt.show()
plt.close()


# In[ ]:


from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML
import numpy as np
from PIL import Image

image = Image.open("../input/iBeacon_Layout.jpg")
init_notebook_mode(connected=True)

xm=np.min(val_y["x"])-1.5
xM=np.max(val_y["x"])+1.5
ym=np.min(val_y["y"])-1.5
yM=np.max(val_y["y"])+1.5

data=[dict(x=[0], y=[0], 
           mode="markers", name = "Predictions",
           line=dict(width=2, color='green')
          ),
      dict(x=[0], y=[0], 
           mode="markers", name = "Actual",
           line=dict(width=2, color='blue')
          )
      
    ]

layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),
            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),
            title='Moving Dots', hovermode='closest',
            images= [dict(
                  source= image,
                  xref= "x",
                  yref= "y",
                  x= -3.5,
                  y= 22,
                  sizex= 36,
                  sizey=25,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")]
            )

frames=[dict(data=[dict(x=[preds[k, 0]], 
                        y=[preds[k, 1]], 
                        mode='markers',
                        
                        marker=dict(color='red', size=10)
                        ),
                   dict(x=[val_y["x"].iloc[k]], 
                        y=[val_y["y"].iloc[k]], 
                        mode='markers',
                        
                        marker=dict(color='blue', size=10)
                        )
                  ]) for k in range(int(len(preds))) 
       ]    
          
figure1=dict(data=data, layout=layout, frames=frames)          
iplot(figure1)


# In[ ]:





# In[ ]:




