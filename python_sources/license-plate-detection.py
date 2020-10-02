#!/usr/bin/env python
# coding: utf-8

# # Introduction
# I was trying to find a dataset with labeled license plates to train a simple model. A Google search bought me to this Indian License Plates dataset. I wrote a notebook with a semi-working deep CNN model. However, it took quite a lot of time to understand how to get data and how to use it. Thus, I decided to create a more readible version of the notebook that I previously create in Kaggle.

# In[ ]:


import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import time
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Methods
# There are three steps in this notebook. The first one is getting the data from the given JSON file. The second one is creating a usable CSV from it. And the third one is creating and training a deep CNN for license plate detection. I used Keras to make the CNN part simpler.
# 
# ## The Dataset
# The dataset isn't similar to those that I saw before. It comes with a JSON format with multiline records in it. The two main elemnts are *content* and *annotation*. *content* contains links to images and *annotation* contains some information about the respected image.

# In[ ]:


df = pd.read_json("/kaggle/input/vehicle-number-plate-detection/Indian_Number_plates.json", lines=True)
df.head()


# In[ ]:


os.mkdir("Indian Number Plates")


# I wrote a simple script to download and save all images to a directory while recording their respected annotation information to a dictionary. The informations that I recorded were `image_width`, `image_height`, x and y coordinates of top left corner and x and y coordinates of bottom right corner of the bounding box (`[top_x, top_y, bottom_x, bottom_y]`).
# 
# At first, I thought all images are JPEG. However, a quick inspection of downloaded images showed that this assumption was wrong. Some of the images are GIF. So, before saving images, I converted them to JPEG images with three (RGB) channels by using `PIL.Image` module.

# In[ ]:


dataset = dict()
dataset["image_name"] = list()
dataset["image_width"] = list()
dataset["image_height"] = list()
dataset["top_x"] = list()
dataset["top_y"] = list()
dataset["bottom_x"] = list()
dataset["bottom_y"] = list()

counter = 0
for index, row in df.iterrows():
    img = urllib.request.urlopen(row["content"])
    img = Image.open(img)
    img = img.convert('RGB')
    img.save("Indian Number Plates/licensed_car{}.jpeg".format(counter), "JPEG")
    
    dataset["image_name"].append("licensed_car{}".format(counter))
    
    data = row["annotation"]
    
    dataset["image_width"].append(data[0]["imageWidth"])
    dataset["image_height"].append(data[0]["imageHeight"])
    dataset["top_x"].append(data[0]["points"][0]["x"])
    dataset["top_y"].append(data[0]["points"][0]["y"])
    dataset["bottom_x"].append(data[0]["points"][1]["x"])
    dataset["bottom_y"].append(data[0]["points"][1]["y"])
    
    counter += 1
print("Downloaded {} car images.".format(counter))


# After that, I created a Dataframe object from the dictionary that I've mentioned before.

# In[ ]:


df = pd.DataFrame(dataset)
df.head()


# Then, I saved it for later more simpler use. You can also download this CSV from the output section of the notebook.

# In[ ]:


df.to_csv("indian_license_plates.csv", index=False)


# Next, I read the previously recorded CSV file and cropped some information from it. Since I fixed the image width and height to 128px by 128px, those columns were not necessary.

# In[ ]:


df = pd.read_csv("indian_license_plates.csv")
df["image_name"] = df["image_name"] + ".jpeg"
df.drop(["image_width", "image_height"], axis=1, inplace=True)
df.head()


# I picked five random records from the dataframe for a later visiual inspection of predictions. I dropped these records from the original dataframe by aiming to prevent the model to be trained on them.

# In[ ]:


lucky_test_samples = np.random.randint(0, len(df), 5)
reduced_df = df.drop(lucky_test_samples, axis=0)


# In[ ]:


WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def show_img(index):
    image = cv2.imread("Indian Number Plates/" + df["image_name"].iloc[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(WIDTH, HEIGHT))

    tx = int(df["top_x"].iloc[index] * WIDTH)
    ty = int(df["top_y"].iloc[index] * HEIGHT)
    bx = int(df["bottom_x"].iloc[index] * WIDTH)
    by = int(df["bottom_y"].iloc[index] * HEIGHT)

    image = cv2.rectangle(image, (tx, ty), (bx, by), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()


# Here, you can see a sample image from the dataset with a bounding box over the license plate.

# In[ ]:


show_img(5)


# I created an `ImageDataGenerator` object from Keras to load batches of images to memory. This process is necessary because we do not have infinite memory in both RAM and GPU RAM. I splitted the data into two with a batch size of 32 images. One for training (80% of the data) and one for validation (20% of the data) during training. Validation is important to see if the model overfit to the training data.
# 
# ## The Model

# In[ ]:


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    reduced_df,
    directory="Indian Number Plates/",
    x_col="image_name",
    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],
    target_size=(WIDTH, HEIGHT),
    batch_size=32, 
    class_mode="other",
    subset="training")

validation_generator = datagen.flow_from_dataframe(
    reduced_df,
    directory="Indian Number Plates/",
    x_col="image_name",
    y_col=["top_x", "top_y", "bottom_x", "bottom_y"],
    target_size=(WIDTH, HEIGHT),
    batch_size=32, 
    class_mode="other",
    subset="validation")


# I created a relatively "not so deep" convolutional neural network. It have 8 convolutinal layers with 4 max pool layers and a fully connected network with 2 hidden layers.

# In[ ]:


model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, CHANNEL)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

model.summary()


# To find the minimum amount of step count to cover all the batches, the following equations are necessary. Mathematically;
# 
# $$
# \text{Step size} = \lceil \frac{\text{Number of elements}}{\text{Batch Size}} \rceil
# $$

# In[ ]:


STEP_SIZE_TRAIN = int(np.ceil(train_generator.n / train_generator.batch_size))
STEP_SIZE_VAL = int(np.ceil(validation_generator.n / validation_generator.batch_size))

print("Train step size:", STEP_SIZE_TRAIN)
print("Validation step size:", STEP_SIZE_VAL)

train_generator.reset()
validation_generator.reset()


# ## Training

# I used Adam to optimize the weights and mean squared error as my loss function.

# In[ ]:


adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss="mse")


# In[ ]:


history = model.fit_generator(train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=30)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# The model's success over the validation data is 80%. However, you can see that from the above figure, the training is stopped after 30th epoch. This may because of the low number of training samples or my model is not capable of learning such data. If you have an idea, please comment below.
# 
# ## Tests

# In[ ]:


model.evaluate_generator(validation_generator, steps=STEP_SIZE_VAL)


# Remember that, we had picked five lucky test samples for visiual inspection. Here they are.

# In[ ]:


for idx, row in df.iloc[lucky_test_samples].iterrows():    
    img = cv2.resize(cv2.imread("Indian Number Plates/" + row[0]) / 255.0, dsize=(WIDTH, HEIGHT))
    y_hat = model.predict(img.reshape(1, WIDTH, HEIGHT, 3)).reshape(-1) * WIDTH
    
    xt, yt = y_hat[0], y_hat[1]
    xb, yb = y_hat[2], y_hat[3]
    
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(img, (xt, yt), (xb, yb), (0, 0, 255), 1)
    plt.imshow(image)
    plt.show()


# # Conclusion
# The model defined above is a very simple toy model. The training is straightforward and tests may not be reliable since the dataset is so small. However, it proves a concept that you can find plates from images with a simple CNN. This network may be used as a part of other more complicated networks.
# 
# Thank you :)
