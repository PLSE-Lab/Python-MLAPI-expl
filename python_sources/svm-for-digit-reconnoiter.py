import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

images = train.iloc[0:5000,1:]
labels = train.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(
    images,labels, train_size=0.9, random_state =0
)

#view an image
i = 1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])