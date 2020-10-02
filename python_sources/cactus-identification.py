#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import os as os
os.getcwd()


# ### Loading Data
# First the labels...

# In[ ]:


train = pd.read_csv('../input/train.csv')

train.head()


# ...then the images. 
# (This bit I had a horrible time with: evidence that it would do me good to spend some more time practicing for loops and syntactic understanding.)

# In[ ]:


train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('../input/train/train/'+ train['id'][i], target_size=(32, 32, 1), 
                         grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


# In[ ]:


X.shape


# Below, creating an object with the outcome variable (1 = image has a cactus in it, 2 = image does not) in it.

# In[ ]:


y=train['has_cactus']
y = to_categorical(y)


# ### Separating the data into a training and testing set.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                   test_size=.33)


# In[ ]:


#verifying shape of each object
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Defining the model
# This is a 2-dimensional convolutional neural network created using the Keras library.

# In[ ]:


nn1 = Sequential()
nn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
               input_shape=(32,32,3)))
nn1.add(Conv2D(64, (3,3), activation = 'relu'))
nn1.add(MaxPooling2D(pool_size=(2,2)))
nn1.add(Dropout(0.25))
nn1.add(Flatten())
nn1.add(Dense(128, activation='relu'))
nn1.add(Dropout(0.5))
nn1.add(Dense(2, activation='softmax'))


# In[ ]:


nn1.compile(loss='categorical_crossentropy',
            optimizer = 'Adam', metrics = ['accuracy'])


# #### Training Wheels
# Note that, for the purpose of the Kaggle competition I have separated the provided training data into a training and test set. The test set is really a validation set. 
# 
# The true test set is the data for which I have no labels and will be submitting my predictions for the competition.

# In[ ]:


nn1.fit(X_train, y_train, epochs=10, 
        validation_data=(X_test, y_test))


# #### Evaluation on Validation set (Test set from Training Data)

# In[ ]:


nn1_score = nn1.evaluate(X_test, y_test, batch_size=128)


# In[ ]:


print('Loss ----------------- Accuracy')
print(nn1_score)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils

nn1_pred = nn1.predict(X_test)

nn1_pred_as_class = nn1_pred.argmax(axis=-1)
y_test_as_class = y_test.argmax(axis=-1)

print(classification_report(y_test_as_class, nn1_pred_as_class))


# This seems like a really good first result. On the 1 class, the class we're most concerned with, we identified 99% of all cactuses in the data and were correct 100% of the time. In our train_test_split we did a 67% training size and a 33% test size. 
# 
# Before I ran this model, I accidentally flipped the train/test split. The model ran better on a larger training size. The precision on the accidental 33% training size was .99 instead of 1. This greater amount of precision was also reflected in the 0-class precision and recall as well (as follows logically).

# #### Predicting on the Test Data
# In order to make predictions, we have to import the test images in the same way we did the training images.

# In[ ]:


import glob
from PIL import Image
folder = glob.glob('../input/test/test/*.jpg')


# In[ ]:


Z = np.array([np.array(Image.open(img)) for img in folder])
Z.shape


# In[ ]:


sub = nn1.predict_proba(Z) #This command gives us probability for each class. 


# In[ ]:


sub_df = pd.DataFrame(sub, columns = ['no_cactus','has_cactus'])


# In[ ]:


sub_df.head()


# Now that we've made our prediciton, we need to adjoin the file name for each image.
# 
# This Kaggle competition requires that submissions be made via a CSV with the filename in column 1 and the probability that the image has a cactus in it in column 2. 

# In[ ]:


img_names = os.listdir('../input/test/test/')

sub_df['id'] = img_names


# In[ ]:


del sub_df['no_cactus'] #We only need the probability that the image does in fact have a cactus


# In[ ]:


sub_df = sub_df[['id', 'has_cactus']]
sub_df.head()


# We now have a dataframe set up in accordance with the competition rules. 

# In[ ]:


sub_df.to_csv('sub_2.csv', index=False) #Creating the CSV for submission


# Honestly it's a good feeling to just get a submission in. But let's run another model just for shits and giggles. 
# 
# ### Second Model
# #### Base MLP 
# I want to run a really basic multi-layer perceptron. I don't anticipate that this will perform better, but I'd like to compare as an exploration. 
# 
# Since MLP takes a 2D list as it's feature list, we'll need to flatten our 32, 32, 2 array. 

# In[ ]:


X_train_flat = []
for sublist in X_train:
    for item in sublist:
        X_train_flat.append(item)
        
X_test_flat = []
for sublist in X_test:
    for item in sublist:
        X_test_flat.append(item)
        
#X_train_flat


# In[ ]:


X_train_pixels = X_train.flatten().reshape(11725, 3072)
#these shape sizes are calculated as such: 
    #first number is number of records in array
    #second number is multiplcation of the subsequent numbers in array, 
            #in this case: 32 * 32 * 3

X_test_pixels = X_test.flatten().reshape(5775, 3072)


# In[ ]:


#Here I'm transforming the target list to a dataframe so I can then delete the list of values for
#the 0 (no cactus) class. 
y_train_df = pd.DataFrame(y_train, index=y_train[:,0])
del y_train_df[0]
y_train_array = y_train_df.values

y_test_df = pd.DataFrame(y_test, index=y_test[:,0])
del y_test_df[0]
y_test_array = y_test_df.values


# #### First go at MLPclassifier

# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

MLP1 = MLPClassifier(activation='relu', hidden_layer_sizes=(18,9,5), learning_rate='constant',
       random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, warm_start=False)
MLP1.fit(X_train_pixels, y_train_array)
MLP1_preds = MLP1.predict(X_test_pixels)
print("Accuracy", accuracy_score(y_test_array, MLP1_preds))
target_names = ["No Cactus", "Cactus"]
print(classification_report(y_test_array, MLP1_preds, target_names=target_names))


# As expected, this didn't work nearly as well. We'll go with the original more complicated convolutional neural net.
# 
# What if we grid search for better parameters?

# I tried to run the below grid search for the best number of layers/nodes. I went a little bonkers on number of nodes, however. After 5 hours, the model was still running. For this reason, I abandoned ship.
# 
# 

# In[ ]:


'''
from sklearn.model_selection import GridSearchCV
import time
start_time = time.clock()
parameters = {'hidden_layer_sizes':[(1500, 1500, 1500), (1500, 800, 400), (500, 500, 500)]}
MLP2 = MLPClassifier(activation='relu', learning_rate='constant', random_state=1, solver='adam')
grid_MLP2 = GridSearchCV(MLP2, parameters, n_jobs=-1, cv=5)
grid_MLP2.fit(X_train_pixels, y_train_array)
print("BEST PARAM", MLP2.best_params_)
print("Time to run", time.clock() - start_time, "seconds")
'''


# Something a little more conservative....

# In[ ]:


'''
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
start_time = time.clock()
parameters = {
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}
MLP3 = MLPClassifier(hidden_layer_sizes = (30, 50, 30), random_state=1)
grid_MLP3 = GridSearchCV(MLP3, parameters, n_jobs=-1, cv=5)
grid_MLP3.fit(X_train_pixels, y_train_array)
print("BEST PARAM", MLP3.best_params_)
print("Time to run", time.clock() - start_time, "seconds")
'''


# The above was still taking a very long time. Let's forget the grid search and just try switching some parametrs. 

# #### MLPClassifier 2

# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

MLP2 = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 30, 50), learning_rate='constant',
       random_state=1, shuffle=True, solver='adam', tol=0.0001, warm_start=False)
MLP2.fit(X_train_pixels, y_train_array)
MLP2_preds = MLP2.predict(X_test_pixels)
print("Accuracy", accuracy_score(y_test_array, MLP2_preds))
target_names = ["No Cactus", "Cactus"]
print(classification_report(y_test_array, MLP2_preds, target_names=target_names))


# Overall, this new model actually improved inferior to our first first instance of MLP. I was somewhat unscientific, however, because I changed more than one parameter. I'm going to run the model again, reverting to the parametrs of the first model except for the node and layers. 
# 

# #### MLPClassifier 3

# In[ ]:


MLP3 = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 30, 30), learning_rate='constant',
       random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, warm_start=False)
MLP3.fit(X_train_pixels, y_train_array)
MLP3_preds = MLP3.predict(X_test_pixels)
print("Accuracy", accuracy_score(y_test_array, MLP3_preds))
target_names = ["No Cactus", "Cactus"]
print(classification_report(y_test_array, MLP3_preds, target_names=target_names))


# This model performs marginally better than our first MLP classifier. It is still much inferior to the convolutional neural network. 

# #### MLPClassifier 4

# In[ ]:


''''''
MLP4 = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 30, 30, 30), learning_rate='constant',
       random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, warm_start=False)
MLP4.fit(X_train_pixels, y_train_array)
MLP4_preds = MLP3.predict(X_test_pixels)
print("Accuracy", accuracy_score(y_test_array, MLP4_preds))
target_names = ["No Cactus", "Cactus"]
print(classification_report(y_test_array, MLP4_preds, target_names=target_names))
''''''


# In this run I added a fourth hidden layer with 30 nodes. This had exactly no difference in performance from the previous iteration. To cut down on processing time, I've commented out the code. 
# 
# ### Conclusion
# The internet can provide some great things! I found the code for the convolutional neural network on from a page called Analytics Vidhya. The model was tweaked to fit the data. It outperformed my from-scratch MLP by a long shot. In the future, I'd like to learn more about CNN so that I can improve my skills at tuning. For the purpose of this submit and this final project, however, I'm out. 
