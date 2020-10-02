#!/usr/bin/env python
# coding: utf-8

# # Case 2. Diabetic Retinopathy Analysis
# ## 1. Objective
# Learn to use convolutional neural networks to classify medical images
# ## 2. References
# - [Kevin Mader's InceptionV3 for Retinopathy (GPU-HR) notebook](https://www.kaggle.com/kmader/inceptionv3-for-retinopathy-gpu-hr)
# - [Sakari Lukkarinen's Demo 11. Xception and Dense 2 notebook](https://www.kaggle.com/sakarilukkarinen/demo-11-xception-and-dense-2)
# 

# ## 3. Required libraries

# In[ ]:


import os
import cv2
import time
import shutil
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from random import randint
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from math import ceil
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 32
EPOCHS = 200
TRAIN_SAMPLE = 2500
TEST_SAMPLE = 800
IMG_SIZE = 440
SEED = randint(0,100)


# ## 4. Data desciption and preprocessing
# The dataset is part of [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) challenge. The data has been preprocessed for easier training. The color palettes have been normailzed to match each other as mutch as possible.<br>
# In the original dataset the is healthy persons and the diseased are devided by sevierty. Zero means healthy and numbers form 1-4 indicate the sevierty of the disease. I have grouped the diseased to one group becasue there is so few images of diseased.<br>
# In addition images are cropped to be square so the images are not streched by the network.

# ### Setup input data

# In[ ]:


train_df = pd.read_csv('../input/preprocessed-diabetic-retinopathy-trainset/newTrainLabels.csv')
test_df = pd.read_csv('../input/preprocessed-diabetic-retinopathy-trainset/retinopathy_solution.csv')

train_input_dir = '../input/preprocessed-diabetic-retinopathy-trainset/300_train/300_train/'
test_input_dir = '../input/preprocessed-diabetic-retinopathy-trainset/300_test/300_test/'

print(train_df.info())
#binary
train_df['level'] = 1*(train_df['level'] > 0)
test_df['level'] = 1*(test_df['level'] > 0)

#Test out images can be found
image_path = train_input_dir + train_df['image'][0] +'.jpeg'
if os.path.isfile(image_path) == False:
    raise Exception('Uable to find train image file listed on dataframe')
else:
    print('Train data frame and file path ready')
    
image_path = test_input_dir + test_df['image'][0] +'.jpeg'
if os.path.isfile(image_path) == False:
    raise Exception('Uable to find test image file listed on dataframe')
else:
    print('Test data frame and file path ready')


# Take subset of dataset and observe distribution

# In[ ]:


train_sample_df = train_df.sample(n=TRAIN_SAMPLE)
test_sample_df = test_df.sample(n=TEST_SAMPLE)
train_sample_df[['level']].hist()


# Crop images for porcessing

# In[ ]:


def cropImages(dataFrame, inputDir, destDir):
    try:
        os.mkdir(destDir)
        print('Created a directory:', destDir)
    except:
        print(destDir, 'already exists!')

    # Crop and resize all images. Store them to train_dir
    print('Cropping and rescaling the images:')
    start = time.time()
    for i, file in enumerate(dataFrame['image']):
        try:
            fname = inputDir + file + '.jpeg'
            img = cv2.imread(fname)

            # Crop the image to the height
            h, w, c = img.shape
            if w > h:
                wc = int(w/2)
                w0 = wc - int(h/2)
                w1 = w0 + h
                img = img[:, w0:w1, :]
            # Rescale to N x N
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Save
            new_fname = destDir + file + '.png'
            cv2.imwrite(new_fname, img)
        except Exception as e:
            # Display the image name having troubles
            print(fname, str(e))
            break

        # Print the progress for every N images
        if (i % 500 == 0) & (i > 0):
            print('{:} images resized in {:.2f} seconds.'.format(i, time.time()-start))
    print('Images cropped and saved to: ', destDir)


# In[ ]:


train_dir = '/tmp/train/'
test_dir = '/tmp/test/'
cropImages(train_sample_df, train_input_dir, train_dir)
cropImages(test_sample_df, test_input_dir, test_dir)


# Create sample generator combinig dataframe infromation and sample images

# In[ ]:


generator = ImageDataGenerator(rescale=1./255, validation_split=0.22, rotation_range=160)

#add file extentions so generator can find them
train_sample_df['image'] = train_sample_df['image'].apply(lambda x: str(x) + '.png')
test_sample_df['image'] = test_sample_df['image'].apply(lambda x: str(x) + '.png')

#integer labels to strings so generator can read them
train_sample_df['level'] = train_sample_df['level'].apply(str)
test_sample_df['level'] = test_sample_df['level'].apply(str)

train_flow = generator.flow_from_dataframe(dataframe=train_sample_df,
                                         directory=train_dir,
                                         x_col='image',
                                         y_col='level',
                                         target_size=(IMG_SIZE, IMG_SIZE),
                                         class_mode='binary',
                                         batch_size=BATCH_SIZE, #default 32
                                         seed=SEED,
                                         subset='training')

val_flow = generator.flow_from_dataframe(dataframe=train_sample_df,
                                         directory=train_dir,
                                         x_col='image',
                                         y_col='level',
                                         target_size=(IMG_SIZE, IMG_SIZE),
                                         class_mode='binary',
                                         batch_size=BATCH_SIZE, #default 32
                                         seed=SEED,
                                         subset='validation')

test_flow = generator.flow_from_dataframe(dataframe=test_sample_df,
                                         directory=test_dir,
                                         x_col='image',
                                         y_col='level',
                                         target_size=(IMG_SIZE, IMG_SIZE),
                                         class_mode='binary',
                                         batch_size=BATCH_SIZE, #default 32
                                         seed=SEED)


# See if image generator outputs images correctly

# In[ ]:


test_batch = train_flow.next()

print(test_batch[0][0].shape)

rows = 2
cols = 2
f, ax = plt.subplots(rows, cols, figsize=(rows*3, cols*3), squeeze=True)
for i in range(rows*cols):
    a = ax.flat[i]
    a.imshow(test_batch[0][i])
plt.show()

#reset the batch index of generator
train_flow.reset()


# ## 5. Model creation and training
# Model uses Google's Inception v3 model with pretrained weights. I have frosen the first 150 layers of the network. On top of the Inception I have added my own dropout to combat overfitting and dense layers. In the end is binary layer with sigmoids activation.

# In[ ]:


ins_weights = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
#pre trained model
base_model = InceptionV3(weights=ins_weights,
                         include_top=False,
                         input_shape=(IMG_SIZE,IMG_SIZE,3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4, seed=SEED)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2, seed=SEED)(x)
pred = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=base_model.input, outputs=pred)

for layer in model.layers[:99]:
    layer.trainable = False

model.summary()


# In[ ]:


rmsprop = optimizers.RMSprop(lr=0.0005)

model.compile(loss='binary_crossentropy',
             optimizer=rmsprop,
             metrics=['acc'])


# In[ ]:


trainSteps = ceil(train_flow.n/BATCH_SIZE)
valSteps = ceil(val_flow.n/BATCH_SIZE)

results = model.fit_generator(
    train_flow,
    steps_per_epoch=trainSteps,
    verbose=1,
    epochs=EPOCHS,
    validation_data= val_flow,
    validation_steps=valSteps
)
model.save('ret_model.h5')
with open('trainHistory1', 'wb') as file_pi:
        pickle.dump(results.history, file_pi)


# ##  6. Results

# In[ ]:


with open('trainHistory1', 'rb') as file:
    history = pickle.load(file)
#history = results.history

acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('results1_acc.png')
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('results1_loss.png')
plt.show()


# ### Test validation

# In[ ]:


test_evals = model.evaluate_generator(test_flow,
                                   steps=test_flow.n/BATCH_SIZE,
                                   verbose=1)
print(model.metrics_names[0], test_evals[0], model.metrics_names[1], test_evals[1])


# In[ ]:


test_pred = model.predict_generator(test_flow,
                                   steps=test_flow.n/BATCH_SIZE,
                                   verbose=1)


# In[ ]:


#delete processed images
shutil.rmtree('/tmp/test')
shutil.rmtree('/tmp/train')


# In[ ]:


predictions = np.argmax(test_pred, axis=1)
cm = confusion_matrix(test_flow.classes, predictions, )#labels=[0,1,2,3,4])

cm_df = pd.DataFrame(cm)
plt.figure(figsize=(7,5))

hmap = sb.heatmap(cm_df, annot=True, fmt='d')
hmap.set(xlabel='pred',ylabel='actual')
plt.savefig('confusion_matrix.png', dpi=600)
plt.show()


# In[ ]:


FP = (cm_df.sum(axis=0) - np.diag(cm_df)).values
FN = (cm_df.sum(axis=1) - np.diag(cm_df)).values
TP = np.diag(cm_df)
TN = cm_df.values.sum() - (FP + FN + TP)

print('Sensitivity: ', sum(TP/(TP+FN))/2)
print('Specificity: ', sum(TN/(TN+FP))/2)
print('Accuracy: ', (TP+TN)/(TP+FP+FN+TN))


# ## 7. Previous steps
# - Created my own simple conv2D network.
# - Changed to InceptionV3 network with added dense layers.
#  - Split training into two parts. First only the last dense layers then in the second round released more of Inception model's weights to be trained.
# - Add dropout to the end to combat overfitting and decreased the dense network size.
# - Removed first training step and released more weights to be trained in the second step.
# - Add rotation to images in data generator.
# ## 8. Conclutions
# Too late in doing this case I realized that I used too little dataset to see any meaningful results. The model is able to learn the dataset after in this final version but overfits almost instantly. More steps should be added to combat it.
# 

# In[ ]:




