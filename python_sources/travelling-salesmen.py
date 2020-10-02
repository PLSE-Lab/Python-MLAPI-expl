#!/usr/bin/env python
# coding: utf-8

# Our final model comprises an ensemble of four finetuned models of Keras' InceptionV3, InceptionResNetV2, and Xception architectures (pretrained on ImageNet). This notebook shall describe the approach we took to train these models and ensemble them, using Xception as an example.

# ### 1. Data Preprocessing
# We did not perform much preprocessing, simply removing images that cannot be opened by `PIL` before splitting 10% of the training data into a validation set.

# In[ ]:


# Setup
import numpy as np
import keras
from keras.applications import inception_v3, inception_resnet_v2, xception
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.layers import Dense, Dropout
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
PREDICT_DIR = "TestImages"
NUM_TRAIN, NUM_TEST, NUM_PREDICT = 34398, 3813, 16111
IMG_WIDTH, IMG_HEIGHT = 256, 256  # On hindsight, we realised these weren't the default image sizes to the keras pre-trained models
NB_CLASSES = 18


# ### 2. Data Augmentation
# We used Keras' `ImageDataGenerator` to augment the images in real-time for us

# In[ ]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    fill_mode = 'nearest',
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    rotation_range = 30)
test_datagen = ImageDataGenerator(
    rescale = 1./255)
predict_datagen = ImageDataGenerator(
    rescale = 1./255)


# ### 3. Transfer Learning
# All of our final models used a simple FC -> Dropout -> Softmax top layer classifier. We froze all other layers and trained just the new classifier. This got us to about 60-65% validation accuracy.

# In[ ]:


# Hyperparams
NUM_EPOCHS = 25
BATCH_SIZE = 54
FC_SIZE = 1024

xception_model = xception.Xception(include_top=False, weights='imagenet', pooling='avg', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
# Freeze the old layers
for layer in xception_model.layers:
    layer.trainable = False
# Attach our top layer classifier
x = xception_model.output
x = Dense(FC_SIZE, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(NB_CLASSES, activation='softmax')(x)
xception_final_model = Model(inputs=xception_model.input, outputs=pred)
xception_final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks and generators
checkpoint = ModelCheckpoint("xception.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = "categorical")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    class_mode = "categorical")

# Start training
xception_final_model.fit_generator(
    train_generator,
    steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
    epochs = NUM_EPOCHS,
    validation_data = test_generator,
    callbacks = [checkpoint, early])


# ### 4. Fine Tuning
# We unfroze only the top few blocks for one of the inceptionV3 models, which got us to around 70-75%. We wanted to do the same for inception_resnet_v2, but couldn't quite figure out the layers that corresponded to each block in the Keras implementation, so we just unfroze everything. Surprisingly, this got us to >80%, so we did the same for all of our other models.
# 

# In[ ]:


# We had to reduce batch size due to GPU memory constraints
BATCH_SIZE = 18

for layer in xception_final_model.layers:
    layer.trainable = True
# We switch to SGD with low LR for transfer learning, so as to not wreck the previously learned weights
xception_final_model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# We also added a reduce LR callback, which may or may not have helped
reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.4, patience=4)
# Start fine tuning
xception_final_model.fit_generator(
    train_generator,
    steps_per_epoch = NUM_TRAIN/BATCH_SIZE,
    epochs = NUM_EPOCHS,
    validation_data = test_generator,
    callbacks = [checkpoint, early, reduceLR])


# ### 5. Ensembling
# At this point we had several seperately trained models with individual (validation) accuracy of around 83-84%. We ensembled the models by averaging their outputs, and one of the four-model ensembles (2x InceptionV3, 1x InceptionResNetV2, 1x Xception) got us our best (public) score of 0.85350, which we used as our final submission

# In[ ]:


predict_generator = predict_datagen.flow_from_directory(
    PREDICT_DIR,
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    class_mode = "categorical",
    shuffle = False)
model1 = load_model("inception_v3_83.h5")
model2 = load_model("inception_v3_84.h5")
model3 = load_model("inception_v4_84.h5")
model4 = load_model("xception_84.h5")

results1 = model1.predict_generator(predict_generator, verbose=True)
results2 = model2.predict_generator(predict_generator, verbose=True)
results3 = model3.predict_generator(predict_generator, verbose=True)
results4 = model4.predict_generator(predict_generator, verbose=True)

ensemble = np.argmax(results1 + results2 + results3 + results4, axis = 1)
image_ids = [int(f[10:-4]) for f in predict_generator.filenames]

# Create the CSV
sub = pd.DataFrame({'id':image_ids, 'category':ensemble})
sub = sub.sort_values(by=['id'])
# Place id column first
sub = sub[['id', 'category']]
sub.to_csv("submission.csv", index=False)


# ### 6. ???
# Upon further analysis we realised our models are performing (relatively) badly on women's tops, frequently misclassifying, for example, a _**women striped top**_ as a _**women long sleeved top**_.
# 
# Personally, looking at the dataset, we don't reckon we'd have fared much better, but naturally we hold our models to a higher standard, and hence we tried our best to teach them the nuances of women's fashion. Unfortunately we did not observe any significant improvement. Here are some ideas we tried:
# - Feeding the model only women's tops data
# - Training a new model to classify only women's tops, then passing all predicted women's tops into this model for re-classification
# - Switching to hinge loss to punish misclassification (this may or may not have helped.. one of our final models in the ensemble was trained with hinge loss in the final stages, but there was no significant change in validation accuracy)
# - Scraping for more data on women's tops (this was near the end of the competition and we did not manage to set up a full scale pipeline for scraping images)
