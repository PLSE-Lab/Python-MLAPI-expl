#!/usr/bin/env python
# coding: utf-8

# # Finding the best learning rate
# 
# Deep neural network model are capable of approximating arbitrary functions [0](https://en.wikipedia.org/wiki/Universal_approximation_theorem), and their ability to transform input data in a wide variety of ways is an important part of what makes them so powerful. Being able to fit some arbitrary data relationship is not the same thing as being able to learn it, however, and one of the most visible hyperparameters available for tuning is the learning rate. Optimal learning rates and learning rate schedules (even [silly ones](https://blog.evjang.com/2018/04/aesthetic-lr.html)) are worth thinking about. 
# 
# Setting your learning rate too low is generally considered to be harmless so long as you are comfortable waiting forever, but setting your learning rate too high can push a model into corners of parameter space far removed from an optimal solution, eventually dropping into ```nan``` territory. 
# 
# Setting a learning rate to Karpathy's constant is one way to choose, but another more effective yet still simple way to find a good starting learning rate is described in a [2015 article by Leslie Smith](https://arxiv.org/abs/1506.01186). Starting with a learning rate much smaller than what you expect to be effective and incremented it gradually over several learning batches until learning degrades, many learning situations will generate a trough shape that can be used to set the initial learning rate. By choosing a learning rate where the decrease in loss was greatest (not, as many may assume, as large as possible before learning blows up), your model may move to a local optima much more quickly than with a naive learning rate choice. 
# 
# 

# In[ ]:


from IPython.display import display, Image
display(Image(filename='../input/digitslrimages/karpathy_optimal.png'))


# In[ ]:


display(Image(filename='../input/digitslrimages/leet.png'))


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import tensorflow as tf
import keras
from keras.layers import Activation, Dropout, Flatten, Dense, SpatialDropout2D, Conv2D, MaxPooling2D, AveragePooling1D, Reshape
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import EarlyStopping, LearningRateScheduler

import matplotlib.pyplot as plt
print(os.listdir('../input/'))


# # Set up the training data
# 
# First we just need to parse in our training data to get something we can feed into our model. 

# In[ ]:


x_train = pd.read_csv('../input/digit-recognizer/train.csv')
x_train.head()


# In[ ]:


# set up training data and labels
dim_x = 28
dim_y = 28
batch_size=32

# read in data/labels
x_train.shape
y_train = np.array(x_train['label'])
x_train.drop('label', axis = 1, inplace = True)
x_train = np.array(x_train.values)

print("data shapes", x_train.shape, y_train.shape, "classes: ",len(np.unique(y_train)))

classes = len(np.unique(y_train))
x_train = x_train.reshape((-1, dim_x,dim_y,1))
# convert labels to one-hot
print(np.unique(y_train))
y = np.zeros((np.shape(y_train)[0],len(np.unique(y_train))))

# convert index labels to one-hot
for ii in range(len(y_train)):
    #print(y_train[ii])
    y[ii,y_train[ii]] = 1
y_train = y


# In[ ]:


# split into training/validation
no_validation = int(0.1 * int(x_train.shape[0]))

x_val = x_train[0:no_validation,...]
y_val = y_train[0:no_validation,...]

x_train = x_train[no_validation:,...]
y_train = y_train[no_validation:,...]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

# define image generators with mild augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,                                   rotation_range=15,                                   width_shift_range=0.125,                                   height_shift_range=0.125,                                   shear_range=0.25,                                   zoom_range=0.075)

train_generator = train_datagen.flow(x=x_train,                                     y=y_train,                                     batch_size=batch_size,                                     shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)

val_generator = test_datagen.flow(x=x_val,                                    y=y_val,                                    batch_size=batch_size,                                    shuffle=True)


# # Building the model 
# 
# Put together a model or load a pre-trained model to train. This one is a relatively small conv-net that eschews dense layers on the top for average pooling the output neurons with ```AveragePooling1D```.

# In[ ]:


# define model (topless conv-net)
model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=1024, kernel_size=(3,3), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu))
model.add(SpatialDropout2D(rate=0.5))
model.add(Conv2D(filters=512, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu))
model.add(Conv2D(filters=250, kernel_size=(2,2), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu))
model.add(Reshape((250,1)))
model.add(AveragePooling1D(pool_size=25,strides=25))
model.add(Reshape(([10])))
model.add(Activation(tf.nn.softmax))

model.summary()


# # People of straw: learning rates that are too low and too high
# 
# You've probably heard the conventional wisdom: learning rates that are too low are relatively safe but may take forever to learn. Learning rates that are too high are unstable and will push your model to a part of parameter space that is far, far removed from the optimal parameters, eventually setting everything to ```nans```. Below we'll set learning rates that fit these criteria before making some more sensible heuristic assumptions later on. Note that in all these example, I decimate ```steps_per_epoch``` so we can get more informative plots of the learning process. 

# In[ ]:


steps_per_epoch = int(len(y_train)/batch_size)
max_epochs = 10
lo_lr = 1e-8

lo_lr_model = keras.models.clone_model(model) #, input_tensors)model
lo_lr_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lo_lr), metrics=['accuracy'])

lo_lr_history = lo_lr_model.fit_generator(generator=train_generator,                                steps_per_epoch=steps_per_epoch/10,                                validation_data=val_generator,                                validation_steps=50,                                epochs=max_epochs*10,                                verbose=0)

hi_lr = 9e-1

hi_lr_model = keras.models.clone_model(model) #, input_tensors)model
hi_lr_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=hi_lr), metrics=['accuracy'])

hi_lr_history = hi_lr_model.fit_generator(generator=train_generator,                                steps_per_epoch=steps_per_epoch/10,                                validation_data=val_generator,                                validation_steps=50,                                epochs=max_epochs*10,                                verbose=0)


# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(211)
plt.plot(lo_lr_history.history['acc'],':',lw=3)
plt.plot(lo_lr_history.history['val_acc'],':',lw=3)

plt.plot(hi_lr_history.history['acc'],'-.',lw=3)
plt.plot(hi_lr_history.history['val_acc'],'-.',lw=3)
plt.title("Accuracy and Loss, bad learning rates",fontsize=28)
plt.ylabel('accuracy',fontsize=24)
plt.legend(['Lo-Train','Lo-Val', 'Hi-Train', 'Hi-Val'],fontsize=18)

plt.subplot(212)
plt.plot(lo_lr_history.history['loss'],':',lw=3)
plt.plot(lo_lr_history.history['val_loss'],':',lw=3)

plt.plot(hi_lr_history.history['loss'],'-.',lw=3)
plt.plot(hi_lr_history.history['val_loss'],'-.',lw=3)
plt.xlabel('epoch',fontsize=24)
plt.ylabel('loss',fontsize=24)
plt.legend(['Lo-Train','Lo-Val', 'Hi-Train', 'Hi-Val'],fontsize=18)
plt.show()


# # Learning by rule-of-thumb
# 
# Despite the trap of setting silly learning rates, it's possible to get good results using an empirical heuristic, _i.e._ something that seems to work pretty well most of the time. This works especially well with an adaptive gradient descent method like Adam. Andrej Karpathy is well known as an evangelist for using an [Adam optimizer with a learning rate of 3e-4](https://karpathy.github.io/2019/04/25/recipe/), so much so that some people call this the ['Karpathy constant'](https://twitter.com/karpathy/status/801621764144971776). In any case it works pretty well for simple problems.

# In[ ]:


steps_per_epoch = int(len(y_train)/batch_size)
max_epochs = 10
naive_lr = 3e-4

naive_lr_model = keras.models.clone_model(model) #, input_tensors)model
naive_lr_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=naive_lr), metrics=['accuracy'])

naive_lr_history = naive_lr_model.fit_generator(generator=train_generator,                                steps_per_epoch=steps_per_epoch/10,                                validation_data=val_generator,                                validation_steps=50,                                epochs=max_epochs*10,                                verbose=0)


# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(211)
plt.plot(naive_lr_history.history['acc'],'--',lw=3)
plt.plot(naive_lr_history.history['val_acc'],'--',lw=3)

plt.title("Accuracy and Loss, 3e-4 Adam",fontsize=28)
plt.ylabel('accuracy',fontsize=24)
plt.legend(['Lo-Train','Lo-Val', 'Hi-Train', 'Hi-Val'],fontsize=18)

plt.subplot(212)
plt.plot(naive_lr_history.history['loss'],'--',lw=3)
plt.plot(naive_lr_history.history['val_loss'],'--',lw=3)

plt.xlabel('epoch',fontsize=24)
plt.ylabel('loss',fontsize=24)
plt.legend(['Lo-Train','Lo-Val', 'Hi-Train', 'Hi-Val'],fontsize=18)
plt.show()


# # Learning rate finder
# 
# We'll use the following ```lr_finder``` class to try to find a good initial learning rate. This class generates a learning rate schedule sweeping logarithmically from low to high. The lower and upper bound is set as the steepest descent and saddle point where the model begins to "blow up," that is the point at which increasing the learning rate further leeds to higher losses. You can use the plot to manually choose a good learning, or, as below, take the geometric mean of the upper and lower bound to get something reasonable. Note that in it's current state there are no sanity checks on bounds recommended by ```lr_finder```, so if the recommendation looks weird it probably is.

# In[ ]:


class lr_finder():
    
    def __init__(self,model,begin_lr=1e-8, end_lr=1e-1, num_epochs=10, period=5):
        # lr_finder generates learning schedules for finding upper and lower bounds on the best learning rate, as well as 
        # a cyclical learning rate schedule based on those bounds
        self.period = period
        # make a copy of the model to train through a sweep of learning rates
        self.model = keras.models.clone_model(model)
        
        # define bounds to sweep through
        self.begin_lr = np.log(begin_lr)/np.log(10)
        self.end_lr = np.log(end_lr)/np.log(10)
        self.num_epochs = num_epochs
        self.lower_bound = begin_lr
        self.upper_bound = 1e-2 #end_lr
        # define learning rates to use in schedules
        self.lr = np.logspace(self.begin_lr,self.end_lr,self.num_epochs)
        self.clr = np.logspace(np.log(self.lower_bound)/np.log(10), np.log(self.upper_bound)/np.log(10), self.period)
        
        
    def reset_model(self, model):
        # reset the model to find new lr bounds 
        self.begin_lr = -10 
        self.end_lr = 0 
        self.lr = np.logspace(self.begin_lr,self.end_lr,self.num_epochs)
        self.model = keras.models.clone_model(model)
        
    def lr_schedule(self,epoch):
        # return lr according to a sweeping schedule
        if epoch < self.num_epochs:
            return self.lr[epoch]
        else:
            return self.lr[0]
        
    def clr_schedule(self,epoch,period=5):
        # return lr according to cyclical learning rate schedule
        my_epoch = int(epoch % self.period)
        return self.clr[my_epoch]
    
    def lr_vector(self,epochs):
        # return the vector of learning rates used in a schedule
        lrv = []
        for ck in range(epochs):
            lrv.append(self.lr_schedule(ck))
        return lrv
    
    def lr_plot(self,history_loss,please_plot=True):
        # plot the lr sweep results and set upper and lower bounds on learning rate
        x_axis = self.lr_vector(self.num_epochs)
        y_axis = history_loss
                   
        d_loss = []
        for cc in range(1,len(y_axis)):
            if cc == 1:
                d_loss.append(y_axis[cc] - y_axis[cc-1])
            else:
                d_loss.append(0.8*(y_axis[cc] - y_axis[cc-1])+0.2*(y_axis[cc-1] - y_axis[cc-2]))
        d_loss = np.array(d_loss)
        
        self.lower_bound = x_axis[d_loss.argmin()]
        self.upper_bound = x_axis[np.array(y_axis).argmin()]
        self.clr = np.logspace(np.log(self.lower_bound)/np.log(10), np.log(self.upper_bound)/np.log(10), self.period)
        
        print("recommended learning rate: more than %.2e, less than %.2e "%(self.lower_bound, self.upper_bound))
        if(please_plot):
            plt.figure(figsize=(10,5))
            plt.loglog(x_axis,y_axis)
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.title('Loss / learning rate progression')
            plt.show()
            
    def get_lr(self,epoch):
        # return the geometric mean of the upper and lower bound learning rates
        return (self.lower_bound *self.upper_bound)**(1/2)
    
    


# In[ ]:


# initialize learning rate finder callback

lrf = lr_finder(model,begin_lr=1e-8, end_lr=1e0, num_epochs=20)
lr_rate = LearningRateScheduler(lrf.lr_schedule)


# In[ ]:


steps_per_epoch = int(len(y_train)/batch_size)
max_epochs = 20

lrf.model.compile(loss='categorical_crossentropy',optimizer=Adam(), metrics=['accuracy'])


lr_history = lrf.model.fit_generator(generator=train_generator,                                steps_per_epoch=steps_per_epoch/20,                                validation_data=val_generator,                                validation_steps=50,                                epochs=max_epochs,                                callbacks=[lr_rate],                                verbose=0)


# In[ ]:


lrf.lr_plot(lr_history.history['loss'])


# In[ ]:


steps_per_epoch = int(len(y_train)/batch_size)
max_epochs = 10

best_lr_model = keras.models.clone_model(model) #, input_tensors)model
# take the geometric mean of learning rates
learning_rate = (lrf.lower_bound*lrf.upper_bound)**(1/2)
best_lr_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

print("Learning with rate %.2e:"%learning_rate)
best_lr_history = best_lr_model.fit_generator(generator=train_generator,                                steps_per_epoch=steps_per_epoch/10,                                validation_data=val_generator,                                validation_steps=50,                                epochs=max_epochs*10,                                verbose=0)


# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(211)
plt.plot(naive_lr_history.history['acc'],'--',lw=3)
plt.plot(naive_lr_history.history['val_acc'],'--',lw=3)

plt.plot(best_lr_history.history['acc'],marker='o',linestyle='--',lw=3)
plt.plot(best_lr_history.history['val_acc'],marker='o',linestyle='--',lw=3)


plt.title("Optimal learning rate vs thumbwise 3e-4 ",fontsize=28)
plt.ylabel('accuracy',fontsize=24)
plt.legend(['3e-4 Train','3e-4 Val', 'Optimal Train', 'Optimal Val'],fontsize=18)

plt.subplot(212)
plt.plot(naive_lr_history.history['loss'],'--',lw=3)
plt.plot(naive_lr_history.history['val_loss'],'--',lw=3)


plt.plot(best_lr_history.history['loss'],marker='o',linestyle='--',lw=3)
plt.plot(best_lr_history.history['val_loss'],marker='o',linestyle='--',lw=3)

plt.xlabel('epoch',fontsize=24)
plt.ylabel('loss',fontsize=24)
plt.legend(['3e-4 Train','3e-4 Val', 'Optimal Train', 'Optimal Val'],fontsize=18)
plt.show()


# # Training with a cyclical learning rate
# 
# Sweeping through learning rates is one small part of the 2015 paper [Cyclical Learning Rates for Training Neural Networks
# ](https://arxiv.org/abs/1506.01186) by Leslie Smith. In fact, the paper suggests we can improve our results by using a cyclical learning rate schedule, sweeping between a high and low bound. We can generate such a schedule with the function ```clr_rate``` in our ```lr_finder``` class.

# In[ ]:


clr_rate = LearningRateScheduler(lrf.clr_schedule)

max_epochs = 10

best_clr_model = keras.models.clone_model(model) #, input_tensors)model
best_clr_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

best_clr_history = best_clr_model.fit_generator(generator=train_generator,                                    steps_per_epoch=steps_per_epoch/10,                                    validation_data=val_generator,                                    validation_steps=50,                                    callbacks = [clr_rate],                                    epochs=max_epochs*10,                                    verbose=0)
#lrf.reset_model(best_clr_model)


# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(211)

plt.plot(naive_lr_history.history['acc'],'--',lw=3)
plt.plot(naive_lr_history.history['val_acc'],'--',lw=3)

plt.plot(best_lr_history.history['acc'],marker='o',linestyle='--',lw=3)
plt.plot(best_lr_history.history['val_acc'],marker='o',linestyle='--',lw=3)

plt.plot(best_clr_history.history['acc'],marker='+',linestyle='--',lw=3)
plt.plot(best_clr_history.history['val_acc'],marker='+',linestyle='--',lw=3)

plt.title("Optimal learning rate vs thumbwise 3e-4 ",fontsize=28)
plt.ylabel('accuracy',fontsize=24)
plt.legend(['3e-4 Train','3e-4 Val', 'Optimal Train', 'Optimal Val',            'Cyclical Train', 'Cyclical Val'],fontsize=18)
plt.axis([5,100,0.85,1.01])

plt.subplot(212)
plt.plot(naive_lr_history.history['loss'],'--',lw=3)
plt.plot(naive_lr_history.history['val_loss'],'--',lw=3)

plt.plot(best_lr_history.history['loss'],marker='o',linestyle='--',lw=3)
plt.plot(best_lr_history.history['val_loss'],marker='o',linestyle='--',lw=3)

plt.plot(best_clr_history.history['loss'],marker='+',linestyle='--',lw=3)
plt.plot(best_clr_history.history['val_loss'],marker='+',linestyle='--',lw=3)

plt.xlabel('epoch',fontsize=24)
plt.ylabel('loss',fontsize=24)
plt.legend(['3e-4 Train','3e-4 Val', 'Optimal Train', 'Optimal Val',            'Cyclical Train', 'Cyclical Val'],fontsize=18)
plt.axis([5,100,0,0.4])
plt.show()


# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(211)

plt.plot(naive_lr_history.history['acc'],'--',lw=3)
plt.plot(naive_lr_history.history['val_acc'],'--',lw=3)

plt.plot(best_lr_history.history['acc'],marker='o',linestyle='--',lw=3)
plt.plot(best_lr_history.history['val_acc'],marker='o',linestyle='--',lw=3)


plt.plot(best_clr_history.history['acc'],marker='+',linestyle='--',lw=3)
plt.plot(best_clr_history.history['val_acc'],marker='+',linestyle='--',lw=3)

plt.plot(lo_lr_history.history['acc'],':',lw=3)
plt.plot(lo_lr_history.history['val_acc'],':',lw=3)

plt.plot(hi_lr_history.history['acc'],'-.',lw=3)
plt.plot(hi_lr_history.history['val_acc'],'-.',lw=3)

plt.title("Heuristic vs. optimal vs. cyclical learning rate",fontsize=28)
plt.ylabel('accuracy',fontsize=24)
plt.legend(['3e-4 Train','3e-4 Val', 'Optimal Train', 'Optimal Val',            'Cyclical Train', 'Cyclical Val', 'Lo Train', 'Lo Val','Hi Train', 'Hi Val'],fontsize=18)

plt.subplot(212)
plt.plot(naive_lr_history.history['loss'],'--',lw=3)
plt.plot(naive_lr_history.history['val_loss'],'--',lw=3)

plt.plot(best_lr_history.history['loss'],marker='o',linestyle='--',lw=3)
plt.plot(best_lr_history.history['val_loss'],marker='o',linestyle='--',lw=3)


plt.plot(best_clr_history.history['loss'],marker='+',linestyle='--',lw=3)
plt.plot(best_clr_history.history['val_loss'],marker='+',linestyle='--',lw=3)



plt.plot(lo_lr_history.history['loss'],':',lw=3)
plt.plot(lo_lr_history.history['val_loss'],':',lw=3)

# loss blows up, doesn't fit nicely with other runs
#plt.plot(hi_lr_history.history['loss'],'-.',lw=3)
#plt.plot(hi_lr_history.history['val_loss'],'-.',lw=3)

plt.xlabel('epoch',fontsize=24)
plt.ylabel('loss',fontsize=24)
plt.legend(['3e-4 Train','3e-4 Val', 'Optimal Train', 'Optimal Val',            'Cyclical Train', 'Cyclical Val', 'Lo Train', 'Lo Val'],fontsize=18)
plt.show()


# # Predict classes for test data
# 
# Now that we've trained our model a few different ways, we can choose the one that seems to be the best and make predictions on the test set.  Hopefully all our effort to find an optimal learning rate and design a good learning rate schedule have paid off, and we'll use the ```best_clr_model``` variant to make test predictions. However, in this case with training limited to 10 epochs, the optimal learning rate model seems to fare best on the validation data, so we'll use that. Extending the training or training on more complicated datasets may have different results. The model using 3e-4 or the optimal learning rate without a schedule also probably did pretty well and should make reasonable predictions, and are worth trying.

# In[ ]:


x_test = pd.read_csv('../input/digit-recognizer/test.csv')
x_test.head()

x_test = np.array(x_test.values)
x_test = x_test / 255.

print("data shape", x_test.shape)

x_test = x_test.reshape((-1, dim_x,dim_y,1))


# In[ ]:


# predict!
y_pred = best_lr_model.predict(x_test)


# In[ ]:


# visualize success (?!?) :/

def imshow_w_labels(img,  pred,count):
    plt.subplot(1,4,count+1)
    plt.imshow(img, cmap="gray")
    plt.title("Prediction: %i, "%(pred), fontsize=24)
    
    
count = 0
mask = [1,3,3,7]
plt.figure(figsize=(24,6))
for kk in range(50,600):
    
    if y_pred[kk,:].argmax() == mask[count]:
        imshow_w_labels(x_test[kk,:,:,0],y_pred[kk,...].argmax(), count)
        count += 1
    if count >= 4: break
plt.show()


# In[ ]:


# convert one-hot predictions to indices
results = np.argmax(y_pred,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

