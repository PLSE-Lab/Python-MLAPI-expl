# %% [code] {"id":"YQnKpMOyBZra","outputId":"3cd2b297-263c-41f2-e30d-77a589d99339"}
from google.colab import drive
drive.mount('/content/drive')

# %% [code] {"id":"pbh7mTJjBkvp"}
# Code to read csv file into colaboratory:
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# %% [code] {"id":"_WDSdm-VB3bF"}
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# %% [code] {"id":"xLm-NDAmCHUA"}
downloaded = drive.CreateFile({'id':'1ANZKYJGS-HdXvHNK3QDhmyypB1_-DVBP'}) # replace the id with id of file you want to access
downloaded.GetContentFile('opencv-tutorial.zip') 

# %% [code] {"id":"Wit2GzhxCIld","outputId":"c5f4cf5f-5812-4cb5-967e-db01cfde103a"}
!unzip opencv-tutorial.zip

# %% [code] {"id":"REnDakhXCYMy"}
# import the necessary packages
import imutils
import cv2

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# %% [code] {"id":"Mi5qO5WeDghp"}
# import the necessary packages
import argparse
import imutils
import cv2


# %% [code] {"id":"hmCJGOwwDw5H"}
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# %% [code] {"id":"2sMgREhsFseO","outputId":"3c81e65b-55ae-4733-d452-21ea43839176"}
import sys
sys.argv[1:]

sys.argv[1:] = '-i image'.split()
args = ap.parse_args()
args

# %% [code] {"id":"b2coAeQRHOjh"}
from google.colab.patches import cv2_imshow

# %% [code] {"id":"zGsAjoUmD--B","outputId":"f0a85870-5b2f-43a2-ada8-e9ccdbde0752"}
# load the input image (whose path was supplied via command line
# argument) and display the image to our screen

#Import image
image = cv2.imread("/content/opencv-tutorial/tetris_blocks.png", cv2.IMREAD_UNCHANGED)
cv2_imshow(image)

# %% [code] {"id":"_3bSNpWJJL7V","outputId":"f41b2e87-ee1a-4ec3-8b4a-03249566e244"}
# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# %% [code] {"id":"pcWKYOLCEDIn"}


# %% [code] {"id":"1eUY5GCREDZy","outputId":"8812018d-5de1-4fad-c36d-1111fda3d93d"}
# applying edge detection we can find the outlines of objects in
# images
edged = cv2.Canny(gray, 30, 150)
cv2_imshow(edged)

# %% [code] {"id":"W8EQEAbwEL4k","outputId":"0d8afe75-17db-4fdb-a6c4-34f3df179843"}
# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2_imshow(thresh)

# %% [code] {"id":"LPu9tdP7EPyh"}
# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

# %% [code] {"id":"zbq9SBVqEUFh","outputId":"f705ad00-adcd-423b-9a49-c42634e85b2a"}
# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv2_imshow(output)

# %% [code] {"id":"0o4DdtzvEYmi","outputId":"9dbbe7d2-191f-48d8-9611-37ef77ce27fd"}
# draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2_imshow(output)

# %% [code] {"id":"Bjxn__h_EdCB","outputId":"478e1f66-4d6d-4e00-8f49-7285439b3796"}

# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2_imshow(mask)

# %% [code] {"id":"pvD-wrd7EDwf","outputId":"f5a5d394-c563-489a-88ec-5379bf3391f3"}
# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2_imshow(mask)


# %% [code] {"id":"BSiWpYpWEECJ","outputId":"8c949843-692c-422f-9cce-6d5434fffa07"}
# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2_imshow(output)

# %% [code] {"id":"8vu6Di86KzoP"}
