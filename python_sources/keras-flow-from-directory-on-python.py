#!/usr/bin/env python
# coding: utf-8

# From the beginning, I decided to use Keras for the task. As for the start I googled "keras images classification" and found great blog post by Francois Chollet.
# 
# (blog https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# 
# github code https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975)
# 
# 
# The code in the blog has simple and understandable structure. Also, there are few posts in the competition Kernels where the similar approach was applied. Thus some addition insight is available.
# 
# 
# (Specifically, these two kernels helped me a lot:
# 
# https://www.kaggle.com/ogurtsov/0-99-with-r-and-keras-inception-v3-fine-tune
# 
# https://www.kaggle.com/fujisan/use-keras-pre-trained-vgg16-acc-98)

# This script was designed to work on local machine, thus its sections are surrounded by ''' marks to avoid running it in Jupiter Notebook. I use Python 2.7 and Keras with Theano backend. The code is tested and runs well.

# As the first step pictures were uploaded to my local hard drive and placed into the project folder. Following script was applied to get txt files with list of training and validation data for each class.

# In[ ]:


import pandas as pd
import numpy as np

'''
labels_df = pd.read_csv('train_labels.csv')
print (labels_df['invasive'].value_counts())
classes = [0,1]

for i in classes:
    df_class = labels_df.loc[labels_df['invasive'] == i]
    df_class = df_class['name'].astype(str)+'.jpg'
    n = lambda x: 100 if i == 1 else 37
    validation_pack = np.random.choice(df_class.values, n(i), replace = False)
    np.savetxt('class_{0}_val.txt'.format(i), validation_pack, fmt = '%s')

    training_pack = np.setdiff1d(df_class.values, validation_pack)
    np.savetxt('class_{0}_tr.txt'.format(i), training_pack, fmt = '%s')
'''


# Next step was to create folder "data" and move into it folders with train and test data. Inside of train subfolder it was necessary to make train and validation class for each folder. 
# 
# At this step my structure of the project folder was following:
#     
#     [project folder] "invasive_species"/..
#         [subfolder] data/..
#             [subfolder] train/..
#                 [subfolder] class_0_tr/..
#                 [subfolder] class_0_val/..
#                 [subfolder] class_1_tr/..
#                 [subfolder] class_1_val/..
#             [subfolder] test/..
#  
#  Text files "class_0_tr", "class_0_val", "class_1_tr", "class_1_val" were placed into "train" subfolder. When all these were done I could use following code in command line (I'm using Windows).
#  
#      cd ../../invasive_species/data/train (here you should put your path)
#      for /f "delims=" %i in (class_0_tr.txt) do copy "%i" class_0_tr
#      for /f "delims=" %i in (class_0_val.txt) do copy "%i" class_0_val
#      for /f "delims=" %i in (class_1_tr.txt) do copy "%i" class_1_tr
#      for /f "delims=" %i in (class_1_val.txt) do copy "%i" class_1_val
# 
# This allocated all pictures by respective folders. Then I changed the structure of the project folder to the following:
#     
#     [project folder] "invasive_species"/..
#         [subfolder] data/..
#             [subfolder] train_tr/..
#                 [subfolder] class_0_tr/..
#                 [subfolder] class_1_tr/..
#              [subfolder] train_val/..
#                 [subfolder] class_0_val/..
#                 [subfolder] class_1_val/..
#             [subfolder] test/test/..
# 
# Please notice that folder "test" was putted inside of another folder "test". Without this step function "flow_from_directory" retrieved no test images.
# 
# Now it was time to implement approach described in Keras blog.

# In[ ]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score

# place to put some parameters
batch_size = 20
num_val_samples = 137
steps = 2295*1.4/batch_size


# Data generator for train set included augmentation and processing (shear, zoom, horizontal_flip, rescale). For the test set only rescale should be made (we don't want to mess with new data - just predict its class).

# In[ ]:


'''
# data generator for training set
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, # random application of shearing
    zoom_range = 0.2,
    horizontal_flip = True) # randomly flipping half of the images horizontally

# data generator for test set
test_datagen = ImageDataGenerator(rescale = 1./255)
'''


# Data generators take data from specified folders. Also it is time to put some additional parameters, like class_mode, target_size and color mode. 
# 
# Batch_size for train and validation data was judgmentally selected as 20. For test data batch size is 1 as we apply trained model to all test pictures without separating them by batches.
# 
# It is important to put "shuffle" equal to False in test generator. Otherwise order of test pictures will be distorted by the function and predictions won't fit order expected by Kaggle checking system.

# In[ ]:


'''
# generator for reading train data from folder
train_generator = train_datagen.flow_from_directory(
    'data/train_tr',
    target_size = (256, 256),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'binary')

# generator for reading validation data from folder
validation_generator = test_datagen.flow_from_directory(
    'data/train_val',
    target_size = (256, 256),
    color_mode = 'rgb',
    batch_size = batch_size,
    class_mode = 'binary')

# generator for reading test data from folder
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size = (256, 256),
    color_mode = 'rgb',
    batch_size = 1,
    class_mode = 'binary',
    shuffle = False)
'''


# The model architecture was taken from the blog. Rmsprop optimizer showed the best LB result with 10 epochs (more epoch provided inferior results). I expect that SGD with small learning rate and more epochs should make it better.

# In[ ]:


'''
# neural network model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch = steps,
                    epochs = 10,
                    validation_data = validation_generator,
                    validation_steps = num_val_samples/batch_size)

#model.save_weights('nn_weights.h5')
#model.load_weights('nn_weights.h5')
'''


# Check against validation set  provided some degree of confidence that the code was made in a right way. But after few tries it was clear that validation results are higher than LB score, and there is not always positive correlation between them. In case of powerful machine cross-validation may be applied.

# In[ ]:


'''
# AUC for prediction on validation sample
X_val_sample, val_labels = next(validation_generator)
val_pred = model.predict_proba(X_val_sample)
val_pred = np.reshape(val_pred, val_labels.shape)
val_score_auc = roc_auc_score(val_labels, val_pred)
print ("AUC validation score")
print (val_score_auc)
print ('\n')
'''


# After draft code had been done I had a good validation score (0.975) and depressingly low LB score (around 0.5). Eventually it became clear that the problem is related to how Keras tool flow_from_directory handle download from folder with test images.
# 
# "flow_from_directory" sort names in the target folder. Thus while Kaggle checking system expects predictions for labels  "1, 2, 3, ..., 1531" the function make predictions for labels "1, 10, 100, 1000, 1001, ..., 2, 20, 200, ..., 1531". 
# 
# The code below "sort back" results of the predictions.

# In[ ]:


'''
# test predictions with generator
test_files_names = test_generator.filenames
predictions = model.predict_generator(test_generator,
                                      steps = 1531)
predictions_df = pd.DataFrame(predictions, columns = ['invasive'])
predictions_df.insert(0, "name", test_files_names)
predictions_df['name'] = predictions_df['name'].map(lambda x: x.lstrip('test\\').rstrip('.jpg'))
predictions_df['name'] = pd.to_numeric(predictions_df['name'], errors = 'coerce')
predictions_df.sort_values('name', inplace = True)
predictions_df.to_csv('predictions_df.csv', index = False)
'''


# The script gets 0.96749 of LB score and definitely has lots of possible improvements like use of pre-trained model and parameters tuning. Also it is reasonable to select the best model/parameters using validation and then train on full set.
