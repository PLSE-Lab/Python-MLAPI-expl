#!/usr/bin/env python
# coding: utf-8

# # Welcome to my kernel !!
# > Please give your feedback & **upvote** this kernel if you like it.

#  # Install mtcnn for Face Detection

# In[ ]:


get_ipython().system('pip install mtcnn                               ')


# In[ ]:


from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import * 
import pandas as pd 


# # Load the model & summarize Input and Output shape.

# In[ ]:


model = load_model('../input/facenet-keras/facenet_keras.h5')      

print(model.inputs)                                                
print(model.outputs)


# # **get_embedding()** function will return the 128 bit embeddings. 

# In[ ]:


def get_embedding(model, face_pixels):               
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# #  **extract_face()** function will return the detected cropped face. 

# In[ ]:


def extract_face(filename, required_size=(160, 160)): 
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# # Exract the faces from Original & test Images.

# In[ ]:


originalface = extract_face('../input/original-image/billgates2.jpg')
testface     = extract_face('../input/sample-image/billgates1.jpeg')


# In[ ]:


plt.imshow(originalface)   #cropped face of original image


# In[ ]:


plt.imshow(testface)  #cropped face of test image


# # Forward propagation to predict embeddings.  

# In[ ]:


originalembedding = get_embedding(model,originalface)    
testembedding = get_embedding(model,testface)


# # Calculate the Euclidean Distance 

# In[ ]:


dist = linalg.norm(testembedding-originalembedding)    


# In[ ]:


print(dist)


# Euclidean Distance between the two embeddings of same person's is relatively small & Big for differnt person's.
# 
# By this idea we can verify that the given images are of same person or different.
# 
# this is called the one shot learning, which means we don't have to train model for every one.   

# In[ ]:




