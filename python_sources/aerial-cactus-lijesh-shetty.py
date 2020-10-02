#!/usr/bin/env python
# coding: utf-8

# **Our model will be able to detect Cactus from aerial images provided.**
# 
# I will take three approaches for building the model.
# 
# **First, ** I will take VGG16 model, which is a pre-trained model. I will exclude its top layer. 
# I will then pass my data into it, and extract features. 
# These features are basically pixel feature data coming out from the trained model. 
# I will feed this data into my dense, fully connected model to output whether the image is cactus or not.
# 
# **My Second approach is**, I will take inceptionV3 Model, again a pretrained model, and add to that my densely connected model as a top layer.
# I will feed my data into it, and determine whether the image is of cactus or not.
# 
# **Finally,** my third model will be my model. I will build a CNN Model. 
# In CNN Model, it focusses on a small section of the image, and takes stride through the entire image.MaxPooling selects the max feature to carry forward to next layer and so on. 
# CNN Model, MaxPooling & Activation, and you can repeat it several times... 
# I also add drop features to regularize the learning, and the model to not overfit. 
# Then I flatten, and connect it to densely, fully connected layers, and finally output whether the image is cactus or not.
# 
# **As we go through this, we will find out which one gives one more accuracy, and I will submit that to the competition...
# **
# 
# **SO ENJOY!!!!**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
#train_df.head()
#train_df.has_cactus.value_counts()

test_df = pd.read_csv('../input/sample_submission.csv')
test_df.head()


# lets read the image, and see how it looks.

# In[ ]:


#import matplotlib.pyplot as plt
#pix = plt.imread(os.path.join("../input/train/train",df.iloc[2,0]))
#plt.imshow(pix)

from IPython.display import Image
Image(os.path.join("../input/train/train",train_df.iloc[2,0]),width=350,height=350)


# I will create a validation set and training set from our data.
# This I will use to verify our model accuracy.

# In[ ]:


train_X = train_df['id'].iloc[:15000]
train_y= train_df['has_cactus'].iloc[:15000]

val_X = train_df['id'].iloc[15000:]
val_y= train_df['has_cactus'].iloc[15000:]

val_X.shape


# In[ ]:


# for image data generator these labels have to be of type string.
test_df['has_cactus'] = train_df.has_cactus
train_df.has_cactus = train_df.has_cactus.astype('str')
test_df.has_cactus = test_df.has_cactus.astype('str')


# Following is the image data generator, which will load image using dataframe id column for both training and testing data.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,rescale = 1./255)

train_data_iterator = datagen.flow_from_dataframe(dataframe=train_df[:15000],directory='../input/train/train',x_col='id',
                                            y_col='has_cactus',class_mode='binary',batch_size=200,
                                            target_size=(250,250))

validation_data_iterator = datagen.flow_from_dataframe(dataframe=train_df[15000:],directory='../input/train/train',x_col='id',
                                            y_col='has_cactus',class_mode='binary',batch_size=200,
                                            target_size=(250,250))

test_data_iterator = datagen.flow_from_dataframe(dataframe=test_df,directory='../input/test/test',x_col='id',
                                            y_col='has_cactus',class_mode='binary',batch_size=200,
                                            target_size=(250,250))


# **Approach 1 : VGG16 MODEL **
# 
# I will import VGG16 model,exclude its top layer, provide it with our training image features, extract the output features from the model, provide that as an input to my densely connected model, and output from my model whether the fed image is of cactus or not.

# In[ ]:


from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

# lets print the layers of VGG16 to get more intution of the # of layers and what they do.
base_model = VGG16(include_top=False,weights='imagenet')
for i, layer in enumerate(base_model.layers):
    print(i,layer.name,layer.output_shape)


# Here I have written a simple get feature method. 
# It provides the image features, and uses VGG16 Model to output features.
# These features and labels are returned from this method.

# In[ ]:


def getFeatures(record_count,iterator_name):
    count=0
    labels = np.zeros(shape = record_count)
    features=np.zeros(shape=(record_count,7,7,512))

    for batch_features, batch_labels in iterator_name:
        features[count*200:(count+1)*200] = base_model.predict(batch_features)
        labels[count*200:(count+1)*200] = batch_labels
        count += 1
        #print('value of feature is',features[(count+1)*200])
        print('count is',count)
        if(count*200 >= record_count):
            break
    return (features,labels)


# I call the method and get the VGG16 features output for my training data.

# In[ ]:


train_features,train_labels = getFeatures(17500,train_data_iterator)
train_features[200]


# Below I separate training and validation feature data output of VGG16 model.

# In[ ]:


train_features_train = train_features[:15000]
train_label_train = train_labels[:15000]

train_features_validation = train_features[15000:]
train_label_validation = train_labels[15000:]
train_features_validation.shape


# I call the method and get the VGG16 features output for my testing data.

# In[ ]:


test_features,test_labels = getFeatures(4000,test_data_iterator)
test_features[35].shape


# I am reshaping the data below to make it ready to my model of densely connected layer.

# In[ ]:


train_features_train.reshape(15000,7*7*512)
train_features_validation.reshape(2500,7*7*512)
test_features.reshape(4000,7*7*512)
test_features.shape


# In[ ]:


# lets build the final layer
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers

model=Sequential()
model.add(Flatten())
#model.add(Dense(256,activation='relu',input_dim=(7*7*512)))
model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l1_l2(.001),input_dim=(7*7*512)))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[ ]:


history=model.fit(train_features_train,train_label_train,epochs=10,batch_size=15,validation_data=(train_features_validation,train_label_validation)
            )
print('history keys are ', history.history.keys())


# **For Training set, loss is: 1.1549  & accuracy is : 0.9029 (90.29%)
# For Validation set, loss is : 1.0813  and accuracy is: 0.8944 (89.44%)
# **
# We will Save the model, and use it later.

# In[ ]:


model_json = model.to_json()
open('my_vgg16_model_arch.json','w').write(model_json)
# save the weights learned as well
model.save_weights('my_vgg16_model_weights.h5',overwrite=True)


# In[ ]:


## Lines to free up the memory...
import gc
del model
del train_features


gc.collect()


# I am loading from my saved model. Incase, if I loose my connectivity....:)

# In[ ]:


from keras.models import model_from_json

model_arch = 'my_vgg16_model_arch.json'
model_wts = 'my_vgg16_model_weights.h5'
loaded_model = model_from_json(open(model_arch).read())
loaded_model.load_weights(model_wts)


# Below I am plotting accuracy of the model for my training and validation set..

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','validation'],loc='upper left')
plt.show()


# In[ ]:


proba_label = loaded_model.predict_proba(test_features)


# In[ ]:


df_test=pd.read_csv('../input/sample_submission.csv')
output_df=pd.DataFrame({'id':df_test['id'] })
output_df['has_cactus']=proba_label
output_df.head()
#output_df.to_csv("submission.csv",index=False)


# In[ ]:


## Lines to free up the memory...
import gc

del loaded_model
del test_features
del history
del output_df

gc.collect()


# **Approach 2 : INCEPTION V3 MODEL **
# 
# In this approach we will use InceptionV3 model,and replace its top layer with our fully connected dense layer.
# I will pass the features of the image, and get accuracy for its prediction on test set.
# 
# I create a inception 3 model, exclude its top layer, and set the layers to non trainable.
# Then I add my deeply connected dense model to it as a top layer. 
# I then train only my newly added layers with the features.
# 
# I then go back, and train some top layers of inception3 layer along with my top layers.
# 
# Once I am done through above steps my model is ready to predict on test set...
# **Enjoy!!!**
# 

# In[ ]:


from keras.models import Model, Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D
from keras import backend as K


base_model = InceptionV3(include_top=False,weights='imagenet')
for i, layer in enumerate(base_model.layers):
    print(i,layer.name,layer.output_shape)
    layer.trainable = False


# Add our fully connected dense model to the inception_v3 model

# In[ ]:


# Add our fully connected dense model to the inception_v3 model
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(256, activation='relu')(x)
x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# lets fit the model with the layers of InceptionV3 frozen. 
# I am training only the layers which I have added on top of it below.

# In[ ]:


# train only the top model since all the layers of inception_v3 is set to non train.
history = model.fit_generator(train_data_iterator,steps_per_epoch=len(train_df)/200,epochs=50, verbose=1, 
                              validation_data=validation_data_iterator,validation_steps=50)


# Once we have trained the model. 
# I am going to train top layers, above 200th layer, of InceptionV3 and my added layers to the model.
# I will then train it with the training data.

# In[ ]:


for layer in model.layers[:200]:
    layer.trainable = False
for layer in model.layers[200:]:
    layer.trainable = True    


# In[ ]:


from keras.optimizers import SGD
model.compile(loss='binary_crossentropy',optimizer= SGD(lr=0.0001,momentum=0.9),metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_data_iterator,steps_per_epoch=len(train_df)/200,epochs=50, verbose=1, 
                              validation_data=validation_data_iterator,validation_steps=50)


# **Awesome!!! Inception V3 model accuracy for training data is 96.71%**

# In[ ]:


model_json = model.to_json()
open('my_incpn_model_arch.json','w').write(model_json)
# save the weights learned as well
model.save_weights('my_incpn_model_weights.h5',overwrite=True)


# In[ ]:


## Lines to free up the memory...
import gc

del model
del layer
gc.collect()


# In[ ]:


from keras.models import model_from_json

model_arch = 'my_incpn_model_arch.json'
model_wts = 'my_incpn_model_weights.h5'
loaded_model = model_from_json(open(model_arch).read())
loaded_model.load_weights(model_wts)


# Now I am going to get features of the image for the testing data.
# I will use the test generator, and then feed this data into the model to get the prediction.

# In[ ]:


features=np.zeros(shape=(4000,250,250,3))
i=0
for test_features_batch, test_labels_batch in test_data_iterator:
        
        features[i*200:(i+1)*200] = test_features_batch
        print('count',i)
        i +=1
        if(i*200 >= 4000):
            break
        
features[2500]
    


# In[ ]:


proba_label = loaded_model.predict(features)


# **Since InceptionV3 outcome is really good, I am going to submit for competition.**

# In[ ]:


df_test=pd.read_csv('../input/sample_submission.csv')
output_df=pd.DataFrame({'id':df_test['id'] })
output_df['has_cactus']=proba_label
output_df.head()
output_df.to_csv("submission.csv",index=False)


# In[ ]:


## Lines to free up the memory...
import gc

del loaded_model
del features
del output_df

gc.collect()


# **Approach 3 : Build my Own CNN Model**
# 
# Now lets build our own model, and see what its accuracy is...

# In[ ]:


from keras.models import Model,Sequential
from keras.preprocessing import utils
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D,Conv2D,MaxPooling2D
from keras import backend as K

# We will build the model...

model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(250,250,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))



model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_data_iterator,steps_per_epoch=len(train_df)/200,epochs=5, verbose=1, 
                              validation_data=validation_data_iterator,validation_steps=50)


# **Accuracy with the self built model is 94.%**
# 
# As in the previous cases, lets save the model.

# In[ ]:


model_json = model.to_json()
open('my_cnv_model_arch.json','w').write(model_json)
# save the weights learned as well
model.save_weights('my_cnv_model_weights.h5',overwrite=True)


# In[ ]:


from keras.models import model_from_json

model_arch = 'my_cnv_model_arch.json'
model_wts = 'my_cnv_model_weights.h5'
my_loaded_model = model_from_json(open(model_arch).read())
my_loaded_model.load_weights(model_wts)


# Finally predict the outcome for the test data with my model...

# In[ ]:


# I could have used the loaded model, but instead of using model directly.
cnv_proba_label = model.predict(features)


# **See Ya!!!! And Continue Building....
# KNOW ML.**
