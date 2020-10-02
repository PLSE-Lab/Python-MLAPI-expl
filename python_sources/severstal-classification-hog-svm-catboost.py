#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


data = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")
data.info()


# #Here we prepare the dataset for a training 

# # Here we get set of rows with notnull masks in the EncodedPixels column an set to 1

# In[ ]:


defects = data[pd.notna(data.EncodedPixels)]
defects.EncodedPixels = 1
defects.info()
print(defects)


# Here we get rows without data in the EncodedPixels column and set class to 0

# In[ ]:


print((data.EncodedPixels).isnull())
NoDefects = data[(data.EncodedPixels).isnull()]
NoDefects.EncodedPixels = 0
NoDefects.info()
print(NoDefects)


# Here we combine same number of images from both classes, and shuffle them

# In[ ]:


dataset= NoDefects.sample(defects.shape[0])
dataset = dataset.append(defects,ignore_index=True)
dataset = dataset.sample(frac=1, replace=True, random_state=1)
dataset


# show image example

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
filename = str(dataset.sample(1).ImageId_ClassId.values)[2:]
filename = filename[:-4]
filename = "../input/severstal-steel-defect-detection/train_images/"+filename
print(filename)

img=mpimg.imread(filename)
imgplot = plt.imshow(img)
plt.show()

split train, val, test
# In[ ]:


val = dataset[0:1000]
test = dataset[1000:2000]
train = dataset[2000:]
train.info


# convert images to HOG vector

# In[ ]:


from skimage.feature import hog
import cv2

def my_extractHOG(filename):
    filename = str(filename)
    filename = filename[:-2]
    filename = "../input/severstal-steel-defect-detection/train_images/" + filename
    img = mpimg.imread(filename)
    img = cv2.resize(img, dsize=(600, 70), interpolation=cv2.INTER_CUBIC)
    print(str(i)+"/"+str(train.ImageId_ClassId.shape[0]))
    img = img / 256
    fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
    return fd,hog_image

ppc = 16
hog_images = []
hog_features = []

for i, filename in enumerate(train.ImageId_ClassId):
    fd,hog_image = my_extractHOG(filename)
    if i<6 : hog_images.append(hog_image) # save some of images for example purpose only
    hog_features.append(fd)


# In[ ]:


plt.imshow(hog_images[3])
print(hog_features[3].shape)


# svm

# In[ ]:


from sklearn.svm import SVC
clf = SVC(gamma='auto')
print(train.EncodedPixels.values.shape)
y = train.EncodedPixels.values
X = np.array(hog_features)
print(X.shape)
clf.fit(X,y)


# test SVM predictions

# In[ ]:


from sklearn.metrics import roc_auc_score
y_scores = [] # init array
hog_features2 = []
for i, filename in enumerate(test.ImageId_ClassId):
    fd,hog_image = my_extractHOG(filename)
    out = clf.predict([np.array(fd)])
    y_scores.append(out)
    print(len(y_scores))
    hog_features2.append(fd)
y_true = test.EncodedPixels.values
y_scores = np.array(y_scores)
roc_auc_score(y_true, y_scores)


# catboost

# In[ ]:


from catboost import CatBoostClassifier, Pool
cat_features = [0]
X = 10000 * X
X = X.astype(int)
print(X)
y.astype(int)
print(y)
Xval = 10000*np.array(hog_features2)
print(Xval)

train
# In[ ]:


train_dataset = Pool(data=X,
                     label=y,
                     cat_features=cat_features)

eval_dataset = Pool(data=Xval.astype(int),
                    label=y_true,
                    cat_features=cat_features)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=300,
                           learning_rate=1,
                           depth=2,
                           custom_metric='AUC')
# Fit model
model.fit(train_dataset, eval_set=eval_dataset, use_best_model=True)
# Get predicted classes
preds_class = model.predict(eval_dataset)
print(preds_class)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_dataset)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_dataset,
                          prediction_type='RawFormulaVal')
print(model.get_best_score())
model.save_model('1layer_catboost')

