#!/usr/bin/env python
# coding: utf-8

# #### This is another well known dataset to do hands on *Image Classification * using Convolutional Neural Network (CNN).
# #### CNN as a subset of *Deep Learning* uses *Convolution* instead of linear matrix operation. *Convolution* is a mathematical operation between 2 functions f(x) and g(x) expressing how the shape of one is modified by the other as f *o* g(x).
# 
# #### Import the libraries.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os


# #### See the files in the input directory.

# In[ ]:


print(os.listdir("../input/dogs-vs-cats/"))


# In[ ]:


os.listdir("../input/dogs-vs-cats/")[0]


# #### Unzip the 'train' zip file into a 'Temp' folder in /kaggle/working directory.

# In[ ]:


from zipfile import ZipFile
zf = ZipFile('../input/dogs-vs-cats/train.zip', 'r')
zf.extractall('../kaggle/working/Temp')
zf.close()


# #### Check whether the Unzip has worked.

# In[ ]:


#Commented to reduce display...
#print(os.listdir("../kaggle/working/Temp/train"))


# #### The filename of a cat's image will start with 'cat' and a dog's image will start with 'dog'. So using this feature create a dataframe with file names and their categories.
# #### I took help for the following code block from https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification.

# In[ ]:


filenames = os.listdir("../kaggle/working/Temp/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append('dog')
    else:
        categories.append('cat')

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# #### First few rows of the dataframe.

# In[ ]:


df.head()


# #### Check the distibution of categories.

# In[ ]:


df['category'].value_counts()


# In[ ]:


sns.countplot(x='category', data=df)


# #### We can see that there are same number of images of each category.
# #### We will now see a sample image from the 'train' set. Since each image has different dimension we will use a standard dimension of 128x128 which is a reduced version of the images' actual dimension.

# In[ ]:


filenames[0]


# In[ ]:


from tensorflow.keras.preprocessing import image
img = image.load_img("../kaggle/working/Temp/train/"+filenames[0])
plt.imshow(img)


# In[ ]:


test_image = image.load_img("../kaggle/working/Temp/train/"+filenames[0], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(test_image[:, :, 2])


# #### Create a *validation set* with 20% images from the 'train' set.

# In[ ]:


from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(df, test_size=0.20, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data   = val_data.reset_index(drop=True)


# #### Check first few lines from train dataset.

# In[ ]:


train_data.head()


# #### Check first few lines from validation dataset.

# In[ ]:


val_data.head()


# #### Check the distribution of category in train dataset.

# In[ ]:


train_data['category'].value_counts()


# In[ ]:


sns.countplot(x='category', data=train_data)


# #### Check the distribution of category in validation dataset.

# In[ ]:


val_data['category'].value_counts()


# In[ ]:


sns.countplot(x='category', data=val_data)


# #### A CNN will have multiple layers of Convolution layers and then it will be fed into a fully connected network.
# 
# In our case we will have 2 layers of **Convolution Layers** and each will have below features -
# 
# 1. **Filters**: The number of output filters in the convolution
# 2. **Kernel Size**: The height and width of the convolution window
# 3. **Strides**: The stride of the convolution
# 4. **Input Shape**: The first Convolution layer will have input shape of 128x128x3 (128x128 is the image size and 3 specifies the channel as 'RGB')
# 
# Then we have **Batch Normalization** and **Dropout** as measure to prevent over-fitting and increase balance.
# 
# **Max Pooling** reduces the dimension of the cluster from one layer to the next by using the maximum value.
# 
# **Flatten** is used to change the dimension so that the output of Convolutional layer can be fed into a fully connected layer.
# 
# We are going to use **ImageDataGenerator** to preprocess the images.

# In[ ]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


# #### Build the model.

# In[ ]:


classifier = Sequential([Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(128,128,3),
                            padding='valid', activation='relu'),
                         BatchNormalization(),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.2),
                         Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            padding='valid', activation='relu'),
                         BatchNormalization(),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.2),
                         Flatten(),
                         Dense(512, activation='relu'),
                         BatchNormalization(),
                         Dropout(0.25),
                         Dense(2, activation='softmax')])

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.summary()


# #### Build the train_generator.

# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(
        train_data,
        "../kaggle/working/Temp/train/",
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')


# #### Build the validation_generator.

# In[ ]:


val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(
        val_data,
        "../kaggle/working/Temp/train/",
        x_col='filename',
        y_col='category',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')


# #### Early Stopping and Checkpoint

# In[ ]:


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


# #### Fit the data into model.

# In[ ]:


history=classifier.fit_generator(train_generator,
                                steps_per_epoch=625,
                                epochs=50,
                                validation_data=val_generator,
                                validation_steps=200,
                                callbacks=[es, mc])


# #### Plotting model accuracy.

# In[ ]:


history.history


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'], '')
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.title('Change of Accuracy over Epochs')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()


# #### Plotting model loss.

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], '')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.title('Change of Loss over Epochs')
plt.legend(['loss', 'val_loss'])
plt.show()


# #### We have specified y_col='category' and class_mode='categorical'. So the 'ImageDataGenerator' will convert the 'category' column into 2D one-hot-encoded matrix and we can see the value assigned to each class of the 'category' column through 'class_indices'.

# In[ ]:


train_generator.class_indices


# #### Unzip the 'test' set.

# In[ ]:


from zipfile import ZipFile
zf = ZipFile('../input/dogs-vs-cats/test1.zip', 'r')
zf.extractall('../kaggle/working/Temp')
zf.close()


# #### Check whether the Unzip has worked.

# In[ ]:


#Commented to reduce display...
#print(os.listdir("../kaggle/working/Temp/test1"))


# #### Create a dataframe for the test data.

# In[ ]:


filenames = os.listdir("../kaggle/working/Temp/test1")

test_data = pd.DataFrame({
    'filename': filenames
})


# #### Load the best model.

# In[ ]:


from keras.models import load_model

saved_model = load_model('best_model.h5')


# #### Use one sample image from test set and predict its class.

# In[ ]:


img = image.load_img("../kaggle/working/Temp/test1/"+filenames[29])
                            
test_image = image.load_img("../kaggle/working/Temp/test1/"+filenames[29], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(img)
test_image = np.expand_dims(test_image, axis=0)
result = saved_model.predict(test_image)
print(np.argmax(result, axis=1))


# #### We can see the class of the image is correctly predicted as '1' which means 'dog'.

# In[ ]:


img = image.load_img("../kaggle/working/Temp/test1/"+filenames[39])
                            
test_image = image.load_img("../kaggle/working/Temp/test1/"+filenames[39], 
                            target_size=(128, 128))
test_image = image.img_to_array(test_image)
plt.imshow(img)
test_image = np.expand_dims(test_image, axis=0)
result = saved_model.predict(test_image)
print(np.argmax(result, axis=1))


# #### We can see the class of the image is correctly predicted as '0' which means 'cat'.

# #### Preprocess the images from test set.

# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
        test_data,
        "../kaggle/working/Temp/test1/",
        x_col='filename',
        y_col=None,
        target_size=(128, 128),
        batch_size=32,
        class_mode=None)


# #### Predict the classes of all the images from test set.

# In[ ]:


predict = saved_model.predict_generator(test_generator)
final_prediction = np.argmax(predict, axis=1)


# #### Create the submission file.

# In[ ]:


predict_df = pd.DataFrame(final_prediction, columns=['label'])
submission_df = test_data.copy()
submission_df['id'] = (submission_df['filename'].str.split('.').str[0]).astype(int)
submission_df = pd.concat([submission_df, predict_df], axis=1)
submission_df = submission_df.drop(['filename'], axis=1)
submission_df = submission_df.sort_values(by=['id'])
submission_df = submission_df.reset_index(drop=True)
submission_df.to_csv('submission.csv', index=False)

