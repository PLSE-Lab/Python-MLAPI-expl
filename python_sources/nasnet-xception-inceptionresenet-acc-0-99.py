#!/usr/bin/env python
# coding: utf-8

# # [<center>Dogs Vs Cats</center>](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

# * Public and private score : 0.04758
# * I've already been tried with resnet, vgg16 and inception. I'm gonna try something different with nasnet, inception resnet and exception. 
# * This time my accuracy is almost 99%. Essential tips are included at the top of the code.

# * calculate the total time spent at the end of this notebook

# In[ ]:


import time
start = time.time()


# # Import Packages

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from keras.layers import Dense, Flatten, Dropout, Lambda, Input, Concatenate, concatenate
from keras.models import Model
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers


# # See which directories have you got

# In[ ]:


for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Unzip the zip files

# In[ ]:


get_ipython().system('unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train')
get_ipython().system('unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test')


# # Check out where my output files reside

# In[ ]:


print(os.listdir('../'))


# # Manually labelling 2 different datasets and save them into labels
# 
# file.split('.')[0] means - 
# * the particular filename is a String separated by dots.
# * line.split(".")[0] returns the 1st item of the array. (the actual file looks like "cat.100.jpg")

# I'm gonna label the data, so that in train test split I won't have to use stratification.

# In[ ]:


filenames = os.listdir("../working/train/train")

labels = []
for file in filenames:
    category = file.split('.')[0]
    if category == 'cat':
        labels.append('cat')
    else:
        labels.append('dog')


# * In case of train_test_split, allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes. I'm using pandas dataframes. 
# * Our dependent variable will be label and filename will act as an independent variable.

# In[ ]:


df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})


# In[ ]:


df.head(10)


# # Make sure that the labels are proportionately equal before and after the train_test_split

# In[ ]:


def get_class_counts(df):
    grp = df.groupby(['label']).nunique()
    return {key: grp[key] for key in list(grp.keys())}

def get_class_proportions(df):
    class_counts = get_class_counts(df)
    return {val[0]: round(val[1]/df.shape[0],4) for val in class_counts.items()}


# In[ ]:


print("Dataset class counts", get_class_counts(df))


# * We can see the classes are divided equally into 0.5.

# In[ ]:


print("Dataset class proportions", get_class_proportions(df))


# # Train Test Splitting

# * Divide the labels according to "label" stratification.

# In[ ]:


train_df, validation_df = train_test_split(df, 
                                           test_size=0.1, 
                                           stratify=df['label'],
                                           random_state = 42)


# * Confirm that the labels of both training and validation sets are equally divided.

# In[ ]:


print("Train data class proportions : ", get_class_proportions(train_df))
print("Validation data class proportions : ", get_class_proportions(validation_df))


# In[ ]:


train_df.head(10)


# * We can see, there is no need of indexing, so I'm dropping it.

# In[ ]:


train_df = train_df.reset_index(drop=True)

validation_df = validation_df.reset_index(drop=True)


# In[ ]:


train_df.head(10)


# In[ ]:


batch_size = 64
train_num = len(train_df)
validation_num = len(validation_df)

print("The number of training set is {}".format(train_num))
print("The number of validation set is {}".format(validation_num))


# # Preprocessing
# 
# 1. Augmentation
# 2. Flow_from_dataframe

# # Specify DataFrame or Directory 

# * Since we're gonna use two different types of multi input model with flow from directory, I'm using two generator.
# * In case of generator, "yield" is used instead of "return". Cz, we need to generate image on the fly, which iterates over the loop once.

# In[ ]:


def two_image_generator(generator, df, directory, batch_size,
                        x_col = 'filename', y_col = None, model = None, shuffle = False,
                        img_size1 = (224, 224), img_size2 = (299,299)):
    gen1 = generator.flow_from_dataframe(
        df,
        directory,
        x_col = x_col,
        y_col = y_col,
        target_size = img_size1,
        class_mode = model,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = 1)
    gen2 = generator.flow_from_dataframe(
        df,
        directory,
        x_col = x_col,
        y_col = y_col,
        target_size = img_size2,
        class_mode = model,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = 1)
    
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        
        if y_col:
            yield [X1i[0], X2i[0]], X1i[1]  #X1i[1] is the label
        else:
            yield [X1i, X2i]
        


# * Test if the generator generates same images with two different sizes(225 & 300)

# In[ ]:


ex_df = pd.DataFrame()
ex_df['filename'] = filenames[:5]
ex_df['label'] = labels[:5]
ex_df.head()

train_aug_datagen = ImageDataGenerator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
e1 = two_image_generator(train_aug_datagen, ex_df, '../working/train/train/',
                                      batch_size = 2, y_col = 'label',
                                      model = 'binary', shuffle = True)

fig = plt.figure(figsize = (10,10))
batches = 0
rows = 5
cols = 5
i = 0
j = 0
indices_a = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
indices_b = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
for [x_batch, x_batch2], y_batch in e1:
    for image in x_batch:
        fig.add_subplot(rows, cols, indices_a[i])
        i += 1
        plt.imshow(image.astype('uint8'))
        
    for image in x_batch2:
        fig.add_subplot(rows, cols, indices_b[j])
        j += 1
        plt.imshow(image.astype('uint8'))
    
    batches += 1
    if batches >= 6:
        break
plt.show()


# # Add data augmentation

# In[ ]:


train_aug_datagen = ImageDataGenerator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
train_generator = two_image_generator(train_aug_datagen, train_df, '../working/train/train/',
                                      batch_size = batch_size, y_col = 'label',
                                      model = 'binary', shuffle = True)


# * Adding augmented data will not improve the accuracy of the validation. Which is why, augmetation on validation dataset is kind of superfluous

# In[ ]:


validation_datagen = ImageDataGenerator()

validation_generator = two_image_generator(validation_datagen, validation_df,
                                           '../working/train/train/', batch_size = batch_size,
                                           y_col = 'label',model = 'binary', shuffle = True)


# # Modeling
# 
# 1. Create model architecture
# 2. Compile
# 3. callbacks
# 4. fit_generator

# In[ ]:


def create_base_model(MODEL, img_size, lambda_fun = None):
    inp = Input(shape = (img_size[0], img_size[1], 3))
    x = inp
    if lambda_fun:
        x = Lambda(lambda_fun)(x)
    
    base_model = MODEL(input_tensor = x, weights = 'imagenet', include_top = False, pooling = 'avg')
        
    model = Model(inp, base_model.output)
    return model


# > You can get the idea of different models from here : https://keras.io/api/applications/

# # Model generation

# In[ ]:


#define vgg + resnet50 + densenet
model1 = create_base_model(nasnet.NASNetLarge, (224, 224), nasnet.preprocess_input)
model2 = create_base_model(inception_resnet_v2.InceptionResNetV2, (299, 299), inception_resnet_v2.preprocess_input)
model3 = create_base_model(xception.Xception, (299, 299), xception.preprocess_input)
model1.trainable = False
model2.trainable = False
model3.trainable = False

inpA = Input(shape = (224, 224, 3))
inpB = Input(shape = (299, 299, 3))
out1 = model1(inpA)
out2 = model2(inpA)
out3 = model3(inpB)

x = Concatenate()([out1, out2, out3])                
x = Dropout(0.6)(x)
x = Dense(1, activation='sigmoid')(x)
multiple_pretained_model = Model([inpA, inpB], x)

multiple_pretained_model.compile(loss = 'binary_crossentropy',
                          optimizer = 'rmsprop',
                          metrics = ['accuracy'])

multiple_pretained_model.summary()


# * best weight will be stored in dogcat.weights.best.hdf5

# # Callback

# In[ ]:


checkpointer = ModelCheckpoint(filepath='dogcat.weights.best.hdf5', verbose=1, 
                               save_best_only=True, save_weights_only=True)


# # Fit model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'multiple_pretained_model.fit_generator(\n    train_generator,\n    epochs = 3,\n    steps_per_epoch = train_num // batch_size,\n    validation_data = validation_generator,\n    validation_steps = validation_num // batch_size,\n    verbose = 1,\n    callbacks = [checkpointer]\n)')


# In[ ]:


multiple_pretained_model.load_weights('dogcat.weights.best.hdf5')


# # Preprocessing test data

# In[ ]:


test_filenames = os.listdir("../working/test/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})

num_test = len(test_df)

test_datagen = ImageDataGenerator()

test_generator = two_image_generator(test_datagen, test_df, '../working/test/test/', batch_size = batch_size)


# # Prediction

# In[ ]:


prediction = multiple_pretained_model.predict_generator(test_generator, 
                                         steps=np.ceil(num_test/batch_size))


# * values smaller than 0.005 become 0.005, and values larger than 0.995 become 0.995.

# # Clipping predicted result

# In[ ]:


prediction = prediction.clip(min = 0.005, max = 0.995)


# # Submission

# In[ ]:


submission_df = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

for i, fname in enumerate(test_filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    submission_df.at[index-1, 'label'] = prediction[i]
submission_df.to_csv('Cats&DogsSubmission.csv', index=False)


# In[ ]:


submission_df.head()


# In[ ]:


## print run time
end = time.time()
print(round((end-start),2), "seconds")

