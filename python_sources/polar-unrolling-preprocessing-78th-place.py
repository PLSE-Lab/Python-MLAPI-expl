#!/usr/bin/env python
# coding: utf-8

# # Polar unrolling

# Polar unrolling allows to better utilize pixel space, remove "rotation" from a list of augmentations and obtain uniformly scaled eye images (for the cases of no/partial cropping of the fundus image with preservation of radius).

# Initially we have images with noticeable black areas.
# In order to remove them we at the beginning apply an autocrop.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


# In[ ]:


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10, 9))

for i in range(3):
    image_path = df_train.loc[i,'id_code']
    image_id = df_train.loc[i,'diagnosis']
    img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = preprocess(img)

    fig.add_subplot(3,1, i+1)
    plt.title(image_id)
    plt.imshow(img)

plt.tight_layout()


# In[ ]:


def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

fig=plt.figure(figsize=(10, 9))

for i in range(3):
    image_path = df_train.loc[i,'id_code']
    image_id = df_train.loc[i,'diagnosis']
    img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = autocrop(img,10)

    fig.add_subplot(3,1, i+1)
    plt.title(image_id)
    plt.imshow(img)

plt.tight_layout()


# After autocrop, we have radius of a circle (represented by widest side of an image). We can extract circle using polar unrolling.
# 
# The method is applied as follows:
# - Radius of a circle determited by maximum width of an image.
# - Perform the unrolling (using cv2.linearPolar).
# - Remove very close and very far sides of unrolled image (because areas close to the center appear to be stretched and areas far from center appear highlighted).
# - Transpose an image (initially, height of unrolled image is much larger than width)
# - Resize an image. In fact, by unrolling we should get images with sizes R (radius) and C (C = 2 * pi * R). But initial size of C is too wide and can increase memory consumption. Thus, image resized not to ratio 1:3.14, but 1:2.56 (subjectively selected value, which maintains ballance between memory consumtion and perception of features)

# In[ ]:


def preprocess(img):
    img = autocrop(img,10)
    value = np.max([img.shape[1]/2.0, img.shape[0]/2.0])
    value = value - value/20.0
    polar_image = cv2.linearPolar(img,(img.shape[1]/2, img.shape[0]/2), value, cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    polar_image = cv2.resize(polar_image,(316,768),interpolation=cv2.INTER_CUBIC)
    res = cv2.transpose(polar_image[:,16:,:])
    res = cv2.addWeighted (res,4, cv2.GaussianBlur(res, (0,0) ,10.0), -4, 128)
    return res

fig=plt.figure(figsize=(10, 9))

for i in range(3):
    image_path = df_train.loc[i,'id_code']
    image_id = df_train.loc[i,'diagnosis']
    img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img)

    fig.add_subplot(3,1, i+1)
    plt.title(image_id)
    plt.imshow(img)

plt.tight_layout()


# # Advantages:
# - No need for rotation augmentation. By unrolling we changed coordinate space. Now rotation becomes just plain shift by x axis (what doesn't matter for convolutional neural networks). It is more than absence of this type of augmentation. Actually now we have all the possible rotations considered in the model (except some borders, but this can easily be solved by the single 50% shift by x axis).
# - Uniform scale (because we are extracting an actual circle based on image sizes). There are cropped images in the 2019 dataset, but we still have the radius. By having radius we have the same scale for features, because we assume very low variation in sizes of eyes.
# - No black pixels from after the radius. And consequently - better image space utilization (if circle is not cropped then we have an image completely filled with an image of an eye).
# - Much higher resultion. Obtained 300 x 768 is related to an image with size 600x600 (radius=300). 600 x 600 = 360,000 pixels (and ~20% are just black pixels if it is a complete circle). 300 x 768 = 230,400 pixels.
# 
# # Disadvantages:
# - High memory consumption (an actual use of such preprocessing technique produces wide (2 x pi xR) images).
# - Correct unrolling is not always possible. Some circles just highly cropped. In this case, by using widest side, this technique can't produce correct unrolling. Because some parts of an image are out of radius defined by widest side (in this case, unrolling should be done using diagonal, not side).

# # About Augmentations
# - Unrolling makes it hard to use zoom with coefficients less than 1.0, because of cropped areas (represented as elliptical areas). One can use zoom 1.0 .. 1.*.
# - Horizontal shift by x axis can by done only once (and probably it is unnecessary at all).
# - Horizontal/vertical flip is ok.
# - No need for rotation.
# 

# Polar unrolling perfectly fits 2015 dataset (it contains mostly complete circles). But in 2019 dataset there are images, which are highly cropped, which reduces applicability of such method.
# 
# EfficientNetB3 model pretrained on images preprocessed by this technique on 2015 dataset and then fine-tuned on 2019 dataset allowed to get score 0.794 for a single model.
