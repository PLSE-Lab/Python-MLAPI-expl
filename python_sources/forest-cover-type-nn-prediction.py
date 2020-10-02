#!/usr/bin/env python
# coding: utf-8

# ## Predicting the cover type of forest for our data set is a very interesting problem. Although we have more than 580 thousand observations, our neural network model is suffering from the problem of overfitting. Training accuracy is around 80% whereas test accuracy is hovering around 62%. After doing batch normalization on data, we are getting little improvement for test accuracy . 

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.data import Dataset

import os
print(os.listdir("../input"))


# In[ ]:


#loading the data
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

covtype = pd.read_csv("../input/covtype.csv", sep=",")
covtype = covtype.reindex(
    np.random.permutation(covtype.index)) # randomize the position of records


# In[ ]:


covtype.head() #checking the data index was radomized


# In[ ]:


covtype.shape #number of records


# In[ ]:


#since data is ready with all dummy variables, we don't need to transform it
covtype['Cover_Type'].unique()


# In[ ]:


covtype['Cover_Type']=covtype['Cover_Type'].astype('category') # convert to categorical datatype for multiple output prediction


# In[ ]:


#target is categorical data, we convert it into dummies
dummies= pd.get_dummies(covtype, columns=['Cover_Type'])


# In[ ]:


dummies.shape


# In[ ]:


#dummies['new']=covtype['Cover_Type']


# In[ ]:


one_hot_code=dummies # assign dataset to one_hot_code

#one_hot_code=covtype
one_hot_code.head()


# In[ ]:


one_hot_code.columns #easier to copy column names for next code


# In[ ]:





# In[ ]:


one_hot_code


# In[ ]:


mean_tr=1
std_tr=1


# In[ ]:



def preprocess_features(one_hot_code, batch_norm_required):
    selected_features = one_hot_code[ # features that will be used for training data
    [ 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40']]
    batch_norm = selected_features[['Aspect','Elevation',  'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']] 
    if batch_norm_required:
        global mean_tr, std_tr
        mean_tr=batch_norm.mean()
        std_tr=batch_norm.std()
     
    batch_norm=(batch_norm-mean_tr)/std_tr
    selected_features.update(batch_norm)

        
    # Add noise
    processed_features = selected_features.copy()  
    
    return selected_features

def preprocess_targets(one_hot_code): # targets that are used for adjusting the model
    output_targets = one_hot_code[
        [ 
        'Cover_Type_1' , 'Cover_Type_2',
       'Cover_Type_3', 'Cover_Type_4', 'Cover_Type_5', 'Cover_Type_6',
       'Cover_Type_7'
        ]]
    return output_targets


# In[ ]:


#(one_hot_code.head(464808)["Horizontal_Distance_To_Hydrology"]-one_hot_code.head(464808)["Horizontal_Distance_To_Hydrology"].mean())/(one_hot_code.head(464808)["Horizontal_Distance_To_Hydrology"]).std()


# In[ ]:





# In[ ]:


# Train and test sets
# Choose examples for training.
training_examples = preprocess_features(one_hot_code.head(464808), 1)
training_targets = preprocess_targets(one_hot_code.head(464808))

# Choose examples for validation.
validation_examples = preprocess_features(one_hot_code.tail(116203), 0)
validation_targets = preprocess_targets(one_hot_code.tail(116203))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


# In[ ]:


training_examples.shape # neurons should be two times the amount of features, usually


# In[ ]:


# model construction: forming the network layers using keras

model = keras.Sequential([
 keras.layers.Dense(1024, activation=tf.nn.relu, # 108 neurons with relu activation, first layer with input
 input_shape=(training_examples.shape[1],)), 
 #keras.layers.Dropout(0.5), # dropout for reducing the overfitting problem
 keras.layers.Dense(512, activation=tf.nn.softplus), # second layer
 #keras.layers.Dropout(0.5),
 keras.layers.Dense(256, activation=tf.nn.relu),

 keras.layers.Dense(7, activation=tf.nn.softmax) #  output layer with 7 categories
 ])

optimizer = tf.train.AdamOptimizer()

model.compile(loss='categorical_crossentropy', #this loss method is useful for multiple categories, otherwise our model does not work
 optimizer=optimizer,
 metrics=['accuracy'])
#model.summary()


# In[ ]:


class PrintDot(keras.callbacks.Callback):
 def on_epoch_end(self, epoch, logs):
  if epoch % 100 == 0: print('')
  print('.', end='')
EPOCHS = 5  
# Store training stats
b_history = model.fit(training_examples, training_targets, epochs=EPOCHS,batch_size=1024, #divisible by training size
                    validation_data= (validation_examples, validation_targets), verbose=0,
                    callbacks=[PrintDot()])
#l2_history = l2_model.fit(training_examples, training_targets, epochs=EPOCHS,batch_size=2568,
#                    validation_data= (validation_examples, validation_targets), verbose=0,
#                    callbacks=[PrintDot()])


# In[ ]:


#b_history.history


# In[ ]:


import matplotlib.pyplot as plt
def plot_history(b_history):
 plt.figure()
 plt.xlabel('Epoch')
 plt.ylabel('accuracy')
 plt.plot(b_history.epoch, np.array(b_history.history['acc']),
 label='Train acc')
 plt.plot(b_history.epoch, np.array(b_history.history['val_acc']),
 label = 'Val acc')
 plt.legend()
 plt.ylim([0, 1])
plot_history(b_history)


# In[ ]:


#plot_history(l2_history)
print('training acc.:',b_history.history['acc'][-1],'\n','test acc.:', (b_history.history['val_acc'])[-1])


# In[ ]:


#tr_loss, tr_acc = model.evaluate(training_examples, training_targets)
#test_loss, test_acc = model.evaluate(validation_examples, validation_targets)
#print('Training accuracy:', tr_acc, '\n Test accuracy:', test_acc) # baseline model accuracy


# In[ ]:


#tr_loss, tr_acc = l2_model.evaluate(training_examples, training_targets)
#test_loss, test_acc = l2_model.evaluate(validation_examples, validation_targets)
#print('Training accuracy:', tr_acc, '\n Test accuracy:', test_acc) # l2 regularized model


# # Baseline model with dropouts and L2 regularized models are both showing the problem of overfitting. More tuning of the models is required. One way to reduce overfitting is dimension reduction where we remove some features.
