#%matplotlib inline

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.cm as cm


#image number to output
IMAGE_TO_DISPLAY = 10

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

#image
images = train.iloc[:, 1:].values
images = images.astype(np.float)
#covert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0/255.0)

image_size = images.shape[1]
#all image are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

#display image
def display(img):
    #(784)=>(28,28)
    one_image = img.reshape(image_width, image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
#output image
display(images[IMAGE_TO_DISPLAY])





#print('images({0}, {0})'.format(image_width, image_height))