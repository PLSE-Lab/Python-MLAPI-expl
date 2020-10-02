#!/usr/bin/env python
# coding: utf-8

# **Hello friends,**
# 
# **This is my first informative post here on Kaggle.**
# 
# **I have been learning Neural Network for about 1 year now, and I have learnt so much from the fello Kagglers, so I thought to give it back to the community.**
# 
# **I hope this will help.**

# **First Things First - What is our input**
# 
# 1. We have **train.csv** which contains id of each image and label for respective image.
# 2. We have **train.zip** this contains four images of each sample, they are **red, green, blue, yellow**.
# 3. We have **test.zip** contains images for testing and submitting your work, ids for test images are given in **sample_submission.csv**
# 4. Images are of two different size and type,
#     512x512 PNG files
#     2048x2048 and 3072x3072 TIFF files
#     we will work with 512x512 images.
#     
# 5. We have everything above mentioned in **../input/** directory.    

# In[ ]:


# Some basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# PIL library to read images
# PIL library is the fastest as per my knowledge
from PIL import Image


# In[ ]:


# Read the train.csv fiel
train_csv = pd.read_csv("../input/train.csv")

# Training images path
TRAIN_PATH = "../input/train/"
# Testing images path
TEST_PATH = "../input/test/"

# Four colours for images
colours = ["red", "green", "blue", "yellow"]

# Training image ids
ids = train_csv["Id"]
# Training image labels
targets = train_csv["Target"]


# In[ ]:


train_csv.head()


# **Let's see the images**

# In[ ]:


ids[0]


# In[ ]:


# The whole set of images for one sample is as follows
print(TRAIN_PATH+ids[0]+"_"+colours[0]+".png")
print(TRAIN_PATH+ids[0]+"_"+colours[1]+".png")
print(TRAIN_PATH+ids[0]+"_"+colours[2]+".png")
print(TRAIN_PATH+ids[0]+"_"+colours[3]+".png")


# As mentioned in the data of the competition ,in this tutorial we will work with **Green** image.

# In[ ]:


green = np.asarray(Image.open(TRAIN_PATH+ids[0]+"_"+colours[1]+".png"))

plt.imshow(green)
plt.show()


# In[ ]:


target = targets[1].split(" ")
print(target)


# **Now the labels**
# 
# We have different 28 labels for one sample, we convert them into
# [0, 0, 0, ... ,0, 0, 0]
# 
# for example sample 0 [16, 0] **Protein** present in them
# we will convert them into
# 
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 
# This is a **Multi Lable Classification Problem**

# In[ ]:


# First create empty array and then fill 1 where needed
label = np.zeros((1, 28))
print(label)


# In[ ]:


for value in target:
    label[0, int(value)] = 1

print(label)


# So now we understand how to read the image and convert label into required form let's create a CNN.

# **Create you own model here.**

# In[ ]:


# Create your own batches here
batches = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
           1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
           1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
           1071]
numb_labels = 28


# In[ ]:


# Model fitting parameters
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

batch_id = 1
index = 0

for batch in batches:
    print("Processing batch number " + str(batch_id))
    # Create empty images and labels for batch
    images = np.zeros((batch, 512, 512, 1), dtype=np.float)
    labels = np.zeros((batch, numb_labels), dtype=np.float)
    
    for i in range(batch):
        
        # Get the image
        green = np.asarray(Image.open(TRAIN_PATH+ids[index]+"_"+colours[1]+".png"))
        index += 1
        # Add to images
        images[i] = green.reshape(512, 512, 1)/255
        
        # Same for labels
        target = targets[i].split(" ")
        
        for value in target:
            labels[i, int(value)] = 1
        
    print("Fitting the data to the model.")
    # Train the model
    # --> Youer model here
    batch_id += 1
    index += 1


# Now let's test our model.

# In[ ]:


test_csv = pd.read_csv("../input/sample_submission.csv")
test_csv.head()


# In[ ]:


ids_test = test_csv["Id"]
ids_test[0]


# In[ ]:


y_pred = np.zeros((len(ids_test), numb_labels), dtype=np.float)
images = np.zeros((1, 512, 512, 3), dtype=np.float)

for i in range(len(ids_test)):
    red = np.asarray(Image.open(TEST_PATH+ids_test[i]+"_"+colours[0]+".png"))
    green = np.asarray(Image.open(TEST_PATH+ids_test[i]+"_"+colours[1]+".png"))
    blue = np.asarray(Image.open(TEST_PATH+ids_test[i]+"_"+colours[2]+".png"))
    
    img_rgb = np.stack((red, green, blue), axis=-1)
    img_rgb = img_rgb/255
    
    images[0] = img_rgb
    
    # Your model
    # y_pred[i] = model.predict(images, verbose=1)


# In[ ]:


y_pred = (y_pred > 0.4).astype(int)


# In[ ]:


# Convert 1 and 0 into 0 to 27 digits for our labels
y_sub = []
for label_set in y_pred:
    index = 0
    l = ""
    for label in label_set:
        if label == 1:
            l += str(index)
            l += " "
            index += 1
        else:
            index += 1
    y_sub.append(l[0:-1])
        


# In[ ]:


# Prepare submission file
submission = pd.DataFrame({"Predicted":y_sub}, index=ids_test)
submission.to_csv("submission_one.csv", index=False)


# If you like my work consider up voting it.
# Thank you.
# 
# 
# 
# **Enjoy The Life, Feel The Music**

# In[ ]:




