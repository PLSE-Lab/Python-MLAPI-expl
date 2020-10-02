#!/usr/bin/env python
# coding: utf-8

# ![](https://www.cenozon.com/wp-content/uploads/2018/07/Computer_visioning.jpg)

# # Goal of this kernel is to introduce the reader to NN aspects of computer vision
# 
# 
# ## I wanted to teach myself more about the subject hence this kernel & paper
# 
# 
# 

# # 1.Theory(math) behind image processing and cNN can be found in my subject sumarization [Theory of image processing](https://drive.google.com/file/d/1AJHe5nV1qAI7zE6TGYhzxPrsKedTEcRr/view?usp=sharing)
# 
# ### I heavily relied on [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/)

# Now that we got the basic idead what happens behind the scenes we can tackle some problems. First and most basic one will be the MINST data set. in the nutshell we have gray-scale images of 10 possible digits (0-9) and our job is to correctly classify them. 
# [Data-set description](https://www.kaggle.com/c/digit-recognizer/data). Now this data-set is some what idealistic in a sense that we already have quantified images, i.e. images are already represented in a pixaleted form via matrix (as explained in the link). So only thing that we need to think about is some pre-processing and building the model. Other more realistic data-set that we are going to analyse is the [Humpback-whale](https://www.kaggle.com/c/humpback-whale-identification) data set. Goal is pretty straight-foreward: Given an image of a tail of a whale, predict its unique id. Now this is a bit specific competition since we have only a few images for every id. But more on that in a different kernel (to avoid over-stuffing this kernel)

# ## a) Digit recognizer

# **Necessary modules**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical #one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Now let us load the data and extract the independent variable Y (standard with unsupervised learning...)

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 

sns.countplot(Y_train)


# As we can see the distribution of the classes is pretty much balanced. That is good since we wont need to use stratified partitioning (i.e. making sure that every subset represents the whole set) when subsetting the train and test set.

# In[ ]:


X_train.isnull().values.any()


# No NAN values (corrupted images) this is just a standard Data Science pipeline of pre-processing...

# In[ ]:


test.isnull().values.any()


# We should also normalize the data, right now we have values in the range of 0 to 255. Reason being is that we only have black-white images. When we translate it to numerical values this coresponds to one gray-level where pixel values can take values between 0 and 255 (depending on the position). We can normalize these values to speed up the algorithm.

# In[ ]:


X_train = X_train / 255.0
test = test / 255.0


# Now if read the teory we can notice that one thing thats beneficial to cNN (among others regarding image processing) ist that input values (images) are forwarded in the exactly the same fashion as they are pixelated. What I mean by that is that images are usually represented as 3 dimensional objects (array containing numerical values). Now in this data set one row is one picture, but is actually a 784 (28*28*1) dimensional vector. We need to transform it

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# NExt thing we have to think about (again standard toguht process in ML pipeline) is categorical variables. We ought to label on one-hot encode these variables. WE already have target variable as label encoded one, but we do not want that (in short because 9 is not more "valuable" than 1 and with label encodign we are going to achieve that) hence we are going to one-hot encode target-y variable.

# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)


# Now splitting our into our training and testing set (validation is already given and with bad notaiton also called test). Note that because of even distribution of classes we do not need to use stratified partitioning and we can use simple train_test_split function from skicit learn with random_state parameter to ensure reproducability.

# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=123)


# But can we acutally see a picture that is encoded numerically?  Sure, plotly library that we already imported helps us with that

# In[ ]:


plt.imshow(X_train[5][:,:,0])


# Cool, now we can start thinking about building a model (cNN!) now we already discussed theoretically what woud it look like.  Pooling and convolution layers are novice things, but also what would it look like as a whole (assuming one has no experiance with NN in python-keras)?

# In[ ]:


model = Sequential()


model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# WE already imported
# 
# 
# from keras.models import Sequential
# 
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
# 
# necessary modules from keras, they should already be familiar from the paper. What we basically built is a pipeline cNN. What each and every part does is already discussed in the paper as well as the input parameters into each individual part. For example 
# 
# Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
#                  activation ='relu', input_shape = (28,28,1))
#                  
#    
#    
#   filters=32 specifies number of filters, kernel_size size of the convolution matrix, padding= do we add 0 in order to control convolution, activation="Relu" specifies with what function do we activate the output (remember it is still a NN), and inpute_shape tells us the what shape should we expect as input (remember we transformed it!)
#   
#   Helpful is also [keras documunetation cNN](https://keras.io/layers/convolutional/)
#   
#   
#   # Important
#   last layer is softmax function (with 10 output values). Why? Well we feed our cNN wit hone picture of a number than the last layer (with softmax) will compress the values to values between 0 and 1. Which coresponds to probability. Than in the end we are going to say that this number is the number with the highest probability (just like with binary classification-we get probabilities that for which we set a tresh hold level that determines wether it is 0 or 1)

# **But why do we choose exactly THIS cNN architecture?** IT looks rather arbitrary. Well that is an issue and there are not a lot of quantitative methods around that tackel these issues (Bayesian optimization,Grid search, random search ...) But there are good starting points with following google paper
# [Recommended cNN architecture](https://arxiv.org/abs/1512.00567)

# **Optimizer** First I started with adam, but I did not achieve desired speed (sgb would be even slower) hence I opted for RMSProp. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.

# In[ ]:


rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# Let us proceede, what else do we need? Backpropagation is still the main algo of NN and we need loss function. Since we perform multiclassification we will need categorical (instead binary) crossentropy. Optimizer (a standard choice) is adam, and standard accuracy measure.

# In[ ]:


model.compile(optimizer = rmsprop , loss = "categorical_crossentropy", metrics=["accuracy"])


# Now we want to accelrate our algo even further, hence we ough to modify the lr (learning rate) Idea is quite simple. In the begininng we should learn rather quickly but as we more towards the optimum (of the loss function) we should decrease the learning rate. (to avoid oscilation)
# To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).(Waiting 2 epochs to see if it changes)
# 
# With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 2 epochs, setting the minimal possible lr to 0.0001
# 

# In[ ]:


lr_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)


# **Data augmentation**
# 
# What else do we need, well if we think about it and have a look at the couple of images we can see that they are a too much idealistic. In reality our model could be tested on a set that has rotated, re-shaped, increased etc... nubmers (images in general) hence we ought make sure that also this modifications of images are to be found in our data-set. It benefits us in couple of ways. When we are done we are going to increase our training sample, we wont over-fit and we will be able to generalize.

# In[ ]:



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# Let us apply this to our model now

# In[ ]:


epochs = 1
batch_size = 86


# In[ ]:



history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[lr_reduction])


# Still with only one foreward and backward pass (1 epoch) we get relatively big run time. To achieve better results (with even bigger run time) one should increase the number of epochs (not to much we still want the model to generalize)

# **Confusion Matrix**
# 
# As with every classification task confusion matrix is very good way to see where did the model make mistakes. Overall we can see that results are good but in some cases model mistakes certain numbers for others

# In[ ]:


# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# Now we see that algo often mistakens 9 and 4. We want to further analyze it. But how? Note that predictions are not classes right away, with argmax function we decide which class do we want to predict (based on highes probability). So we can argue that those that misssed the most had the highest probabilty for the actual class. Those cases we want to observe...

# In[ ]:


# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# I do not know about you but some of the would fool even a human....

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




