#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import glob


# # 1. Data Exploration
# ## 1.1 Load the data frame
# Load the data frame with paths (*id*) and labels (*has_cactus*)

# In[ ]:


df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
df.has_cactus = df.has_cactus.astype(str) # Convert the column to string to be used by keras generator later on
df.head()


# ## 1.2 Visualize the images

# In[ ]:


all_paths = glob.glob("../input/aerial-cactus-identification/train/train/*.jpg")
plt.figure(figsize=(18, 4))

for i in range(1, 15):
    ax = plt.subplot(1, 15, i)
    ax.imshow(plt.imread(all_paths[i]))
    ax.grid(False)
    ax.axis('off')

plt.show()


# # 2. Transfer learning VGG-19
# ## 2.1 Base model
# Let's take the base vgg-19 model from pre trained keras model without the last layer. Then set all the vgg19 layer to be non-trainable.

# In[ ]:


vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

for layer in vgg.layers:
    layer.trainable = False


# ## 2.2 Finishing the model
# Let's add the three additional dense layer to the model to be trained, a dropout layer and finally a dense layer with only one neuron with the sigmoid as activation function.
# The output of the last layer will be a number between 0 and 1.

# In[ ]:


x = vgg.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=vgg.inputs, outputs=outputs)
model.summary()


# ## 2.3 Build the data pipeline
# We use the keras image data generator to load the images into the model, using the pandas dataframe for the file paths and labels.
# The dataset is composed of 175000 images. We use the first 15k images as train set and the last 2.5k as valid set.

# In[ ]:


datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(df[:15000], 
                                              x_col="id", y_col="has_cactus", 
                                              directory="../input/aerial-cactus-identification/train/train/", 
                                              class_mode="binary", 
                                              batch_size=128, target_size=(150, 150))

valid_generator = datagen.flow_from_dataframe(df[15000:], 
                                              x_col="id", y_col="has_cactus", 
                                              directory="../input/aerial-cactus-identification/train/train/", 
                                              class_mode="binary", 
                                              batch_size=128, target_size=(150, 150))


# ## 2.4 Compile and train the model

# In[ ]:


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:


history = model.fit_generator(train_generator, validation_data=valid_generator, epochs=10)


# # 3. Submission
# ## 3.1 Make predictions

# In[ ]:


test_generator = datagen.flow_from_directory("../input/aerial-cactus-identification/test", classes=None, target_size=(150, 150))


# In[ ]:


def load_and_scale_img(path):
    img = keras.preprocessing.image.load_img(path, target_size=(150, 150))
    img = keras.preprocessing.image.img_to_array(img)
    img /= 255.
    return img


# In[ ]:


test_paths = glob.glob("../input/aerial-cactus-identification/test/test/*.jpg")
print("Found %d images." % len(test_paths))


# In[ ]:


test_set = [load_and_scale_img(path) for path in test_paths]


# In[ ]:


predictions = model.predict([test_set])


# In[ ]:


submission_df = pd.DataFrame(data={
    "id": [path.split("/")[-1] for path in test_paths], 
    "has_cactus": predictions[:, 0].astype(int)
})
submission_df.head()


# In[ ]:


submission_df.to_csv('sample_submission.csv', index=False)

