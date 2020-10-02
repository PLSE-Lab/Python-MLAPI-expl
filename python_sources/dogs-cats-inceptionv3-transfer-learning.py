#!/usr/bin/env python
# coding: utf-8

# # Import Library

# * number of training samples: 8000  (4000 cat - 4000 dog)
# * number of validation samples: 1600 (800 cat - 800 dog)

# In[ ]:


import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras import applications
from pathlib import Path
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import random
import os


# # Preparing dataset

# In[ ]:


import zipfile
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as zip_ref:
    zip_ref.extractall("train")

with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as zip_ref:
    zip_ref.extractall("test1")


# In[ ]:


train_directory = "train/train/"
test_directory  = "test1/test1/"
# See sample image
filenames = os.listdir(train_directory)
sample = random.choice(filenames)
print(sample)
image = load_img(train_directory + sample)
plt.imshow(image)


# In[ ]:


# 8000 train samples
# 1600 validation samples
import shutil
source_dir = 'train/'
def copy_files(prefix_str, range_start, range_end, target_dir):
    image_paths = []
    for i in range(range_start, range_end):
        image_path = os.path.join(source_dir,'train', prefix_str + '.'+ str(i)+ '.jpg')
        image_paths.append(image_path)
    dest_dir = os.path.join( 'data', target_dir, prefix_str)
    os.makedirs(dest_dir)

    for image_path in image_paths:
        shutil.copy(image_path,  dest_dir)

copy_files('dog', 0, 4000, 'train')
copy_files('cat', 0, 4000, 'train')
copy_files('dog', 4000, 4800,'validation')
copy_files('cat', 4000, 4800, 'validation')


# In[ ]:


# All data, 12500 cat, 12500 dog
source_dir = 'train/'
def copy_files(prefix_str, range_start, range_end, target_dir):
    image_paths = []
    for i in range(range_start, range_end):
        image_path = os.path.join(source_dir,'train', prefix_str + '.'+ str(i)+ '.jpg')
        image_paths.append(image_path)
    dest_dir = os.path.join( 'Alldata', target_dir, prefix_str)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for image_path in image_paths:
        shutil.copy(image_path,  dest_dir)

copy_files('dog', 0, 12500, 'train')
copy_files('cat', 0, 12500, 'train')


# In[ ]:


#remove train folder
if  os.path.exists('train'):
    #os.removedirs("train")
    shutil.rmtree("train") 


# # Preparing Library

# In[ ]:


#Learning curves
def Polt_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    print("Accuracy = %0.3f" % (acc[epochs-1]*100),  ", val_acc = %0.3f" % (val_acc[epochs-1]*100))
    print("loss     = %0.3f" % loss[epochs-1], ", val_loss= %0.3f" % val_loss[epochs-1])
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# In[ ]:


# Model predict
def Model_predict(Model,Test_dir):  
    test_filenames = []
    for file in os.listdir(Test_dir):   
        test_filenames.append(os.path.join(Test_dir,file))  

    test_df = pd.DataFrame({
        'filename': test_filenames
    })

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator=test_datagen.flow_from_dataframe(
                dataframe=test_df,
                x_col="filename",
                y_col=None,
                batch_size=50,
                seed=42,
                shuffle=False,
                class_mode=None,
                target_size=(img_height,img_width))
    
    nb_test_samples = len(test_df)
    test_steps=nb_test_samples // 50
    pred=Model.predict_generator(test_generator,
                    steps=test_steps,
                    verbose=1)
    
    pred = [1 if p[0] > 0.5 else 0 for p in pred]
    print (pred[:12])
    #predicted_class_indices=np.argmax(pred,axis=1)
    predicted_class_indices=np.argmax(pred)

    #len(predicted_class_indices)
    #print(predicted_class_indices[:12])
    return pred,test_df
    #return predicted_class_indices,test_df


# In[ ]:


#testing known data in train folder: on 25000 image 
def Test_Model_known_Data(Model):
    print("Testing cats....")
    model_pred_cat,test_df  = Model_predict(Model,"Alldata/train/cat") #0
    print("Testing dogs....")
    model_pred_dog,test_df  = Model_predict(Model,"Alldata/train/dog") #1

    #print result
    model_true_cat  = len(test_df) - sum (model_pred_cat)
    model_true_dog  = sum (model_pred_dog)
    model_true      = model_true_cat + model_true_dog
    # model result
    print("  model result")
    print("cat accuracy  = %2.3f" % (model_true_cat /len(test_df) *100))
    print("dog accuracy  = %2.3f" % (model_true_dog /len(test_df) *100))
    print("Total accuracy= %2.3f" % (model_true /(2*len(test_df)) *100))


# In[ ]:


# Plot predict image output
get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def Plot_predict(predicted_class_indices,Test_dir,test_df):
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 12
    ncols = 4
    pic_index = 0 # Index for iterating over images
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    for i, img_path in enumerate(test_df.filename[:48]):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        #img = mpimg.imread(img_path, target_size=(256, 256))Test_dir
        img = load_img( img_path, target_size=(150,150))
        plt.imshow(img) 
        result = predicted_class_indices[i]
        if (result == 1 ):
            name = 'Dog'
        else :
            name = 'Cat'
        plt.title( name )


# In[ ]:


# Save Submission to csv file
def Save_Submission(predict,model,mod,test_df):
    if not os.path.exists(mod):
        os.makedirs(mod)
        
    test_df['category'] = predict
    submission_df = test_df.copy()
    #submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['id'] = submission_df['filename'].str.split('.').str[0].str.split('/').str[1]
    submission_df['label'] = submission_df['category']
    submission_df.drop(['filename', 'category'], axis=1, inplace=True)
    submission_df.index += 1 
    submission_df.to_csv( mod + '/submission_AM_'+ mod +'.csv', index=True)

    #plt.figure(figsize=(10,5))
    submission_df['label'].value_counts().plot.bar()
    plt.title("(Test data , "+mod + " )")


# # Preparing data

# In[ ]:


# dimensions of our images.
img_width, img_height = 150, 150      #299,299
IMG_SHAPE = (img_width, img_height, 3)

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 8000
nb_validation_samples = 1600
epochs = 5
batch_size = 32


# In[ ]:


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                seed=42,
                class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                seed=42,
                class_mode='binary')#binary  categorical


# # Model 1

# In[ ]:


# build the InceptionV3 network
base_model = applications.InceptionV3(input_shape=IMG_SHAPE,
                                      weights='imagenet',
                                      include_top=False) #, pooling='average'
print("base_model.layers", len(base_model.layers)) #311

#Freeze the convolutional base
#for layer in base_model.layers[:100]:
#    layer.trainable = False
for layer in base_model.layers:
    layer.trainable = True

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(base_model)
model.add(top_model)

#base_model.summary()
#top_model.summary()
model.summary()


# In[ ]:


if not os.path.exists('model'):
    os.makedirs("model")

def model_train(learningRate, epochs, itration):
# compile the model with a RMSprop optimizer and a very slow learning rate.
    model.compile(loss='binary_crossentropy',  #categorical_crossentropy
                  #optimizer=optimizers.SGD(lr=learningRate, momentum=0.9),
                  optimizer=optimizers.RMSprop(lr=learningRate),
                  metrics=['accuracy'])
    save = 'model/model.weights.best_InceptionV3_' + itration + '.hdf5'
    checkpointer = ModelCheckpoint(filepath= save, 
                                   verbose=1, save_best_only=True)
    # fit the model
    hist = model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            callbacks=[checkpointer] )
    return hist


# In[ ]:


initial_epochs = 5
learningRate = 1e-4
hist = model_train(learningRate, initial_epochs, '1')


# In[ ]:


# Save neural network structure and weights
model_structure = model.to_json()
f = Path("model/model_structure_InceptionV3.json")
f.write_text(model_structure)
model.save_weights("model/model_weights_InceptionV3_1.h5")


# ### Model output: Learning curves

# In[ ]:


Polt_history(hist)
plt.savefig('model/hist.png')


# # Continue train: fine tune model 

# In[ ]:


# compile the model with a SGD/momentum optimizer and a very slow learning rate.
epochs = 5
learningRate=1e-5
hist_2 = model_train(learningRate, epochs, '2')


# In[ ]:


initial_epochs = 5


# In[ ]:


# Save neural network weights
model.save_weights("model/model_weights_InceptionV3_2.h5")


# In[ ]:


#Learning curves
Polt_history(hist_2)
plt.savefig('model/hist_2.png')


# In[ ]:


def Polt_history_fineTune(hist1,hist2):
    acc      = hist1.history['accuracy']     + hist2.history['accuracy']  
    val_acc  = hist1.history['val_accuracy'] + hist2.history['val_accuracy']

    loss     = hist1.history['loss']         + hist2.history['loss']
    val_loss = hist1.history['val_loss']     + hist2.history['val_loss']
    #initial_epochs = initial_epochs
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    #plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    #plt.ylim([0,1.0])
    #plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
Polt_history_fineTune(hist, hist_2)
plt.savefig('model/hist_3.png')


# # Testing model 1

# ### * Testing model 1: on 25000 image 

# In[ ]:


#testing known data in train folder
Test_Model_known_Data(model)


# ### * Testing model 1: on 12500 image (test data)

# In[ ]:


#testing unknown data in test folder
predict,test_df =Model_predict(model,test_directory)
Save_Submission(predict,model,"model",test_df)


# # Plot sample of predicted result

# In[ ]:


Plot_predict(predict,test_directory,test_df)
plt.savefig('model/predicted.png')


# #  Model 2 

# In[ ]:


if not os.path.exists('model2'):
    os.makedirs("model2")   
# build the InceptionV3 network
base_model2 = applications.InceptionV3(input_shape=IMG_SHAPE,
                                       weights='imagenet', 
                                       include_top=False) #, pooling='average
print("base_model.layers= ", len(base_model2.layers)) #155

#Feature extraction
#Freeze the convolutional base
#for layer in base_model2.layers[:100]:
#    layer.trainable = False
for layer in base_model2.layers:
    layer.trainable = True    
# build a classifier model to put on top of the convolutional model
top_model2 = Sequential()
top_model2.add(GlobalAveragePooling2D())
top_model2.add(Dense(1, activation='sigmoid'))

model2 = Sequential()
model2.add(base_model2)
model2.add(top_model2)

model2.summary()


# In[ ]:


def model2_train(learningRate, epochs, itration):
    model2.compile(loss='binary_crossentropy',  #categorical_crossentropy
                  optimizer=optimizers.RMSprop(lr=learningRate),
                  #optimizer=optimizers.SGD(lr=learningRate, momentum=0.9),
                  metrics=['accuracy'])
    
    save = 'model2/model2.weights.best_InceptionV3_' + itration + '.hdf5'
    checkpointer = ModelCheckpoint(filepath= save , 
                                   verbose=1, save_best_only=True)

    hist2 = model2.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            callbacks=[checkpointer])
    return hist2


# In[ ]:


epochs = 6
learningRate=1e-4
hist2 = model2_train(learningRate, epochs, '1')


# In[ ]:


# Save neural network structure and weights
model2_structure = model2.to_json()
f = Path("model2/model2_structure_InceptionV3.json")
f.write_text(model2_structure)
model2.save_weights("model2/model2_weights_InceptionV3_1.h5")


# In[ ]:


Polt_history(hist2)
plt.savefig('model2/hist2.png')


# ## Fine tune model 2

# In[ ]:


# Load neural network structure and weights
#model2.load_weights("model2/model2_weights_ResNet50.h5")
epochs = 5
learningRate=1e-5
hist2_2 = model2_train(learningRate, epochs, '2')


# In[ ]:


model2.save_weights("model2/model2_weights_InceptionV3_2.h5")
Polt_history(hist2_2)
plt.savefig('model2/hist2_2.png')


# # Testing model 2

# * Testing model 2: on 25000 image 

# In[ ]:


#testing known data in train folder
Test_Model_known_Data(model2)


# * Testing model 2: on 12500 image (test data)

# In[ ]:


#testing unknown data in test folder
predict2,test_df =Model_predict(model2,test_directory)
Save_Submission(predict2,model2,"model2",test_df)


# # Plot sample of predicted result

# In[ ]:


Plot_predict(predict2,test_directory,test_df)
plt.savefig('model2/predicted2.png')


# In[ ]:


#remove test folder
if  os.path.exists('test1'):
    shutil.rmtree("test1") 
if  os.path.exists('data'):
    shutil.rmtree("data")
if  os.path.exists('Alldata'):
    shutil.rmtree("Alldata") 
    
file1 = "model/model.weights.best_InceptionV3_1.hdf5"
file2 = "model/model.weights.best_InceptionV3_2.hdf5"
file3 = "model/model_weights_InceptionV3_1.h5"
file4 = "model2/model2.weights.best_InceptionV3_1.hdf5"
file5 = "model2/model2.weights.best_InceptionV3_2.hdf5"
file6 = "model2/model2_weights_InceptionV3_1.h5"

if  os.path.isfile(file1):
    os.remove(file1)    
if  os.path.isfile(file2):
    os.remove(file2) 
if  os.path.isfile(file3):
    os.remove(file3) 
if  os.path.isfile(file4):
    os.remove(file4) 
if  os.path.isfile(file5):
    os.remove(file5) 
if  os.path.isfile(file6):
    os.remove(file6) 

