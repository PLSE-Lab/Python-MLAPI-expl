#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import json
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization
# from tensorflow.keras.layers import Activation, MaxPooling2D
# from tensorflow.keras.layers import Conv2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model, Model
from PIL import Image
from mtcnn.mtcnn import MTCNN


# In[ ]:


images=os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images")
# print(len(os.listdir(images)))
annotations = os.path.join("/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations")


# In[ ]:


a=os.listdir(images)
b=os.listdir(annotations)
a.sort()
b.sort()
# print(a[1698:1708])
# print(b[:10])


# In[ ]:


test_images=a[:1698]
train_images=a[1698:]
train_ann=b
len(train_images)==len(train_ann)


# In[ ]:


ann_path = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations/"
jdata = json.load(open(ann_path+train_ann[1860]))
anns = jdata["Annotations"]
# bb = anns[0]['BoundingBox']
bb = get_boxes('1861.jpg')
imgpath = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/1861.jpg"
im = cv2.imread(imgpath)
fig,ax = plt.subplots(1)
ax.imshow(im)
print(bb)
for box in bb:
    print(box)
    rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
img=plt.imread(os.path.join(images,train_images[0]))
plt.imshow(img)
plt.show()


# In[ ]:


train_csv=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/train.csv"))
submission=pd.read_csv(os.path.join("/kaggle/input/face-mask-detection-dataset/submission.csv"))


# In[ ]:


df = train_csv
bbox=[]
for i in range(len(train_csv)):
    arr=[]
    for j in df.iloc[i][["x1",'x2','y1','y2']]:
        arr.append(j)
    bbox.append(arr)


# In[ ]:


df["bbox"]=bbox
# df.head()


# In[ ]:


def get_boxes(id):
    boxes=[]
    for i in df[df["name"]==str(id)]["bbox"]:
        boxes.append(i)
    return boxes
# print(get_boxes('1810.jpg'))


# In[ ]:


image=train_images[11]

img=plt.imread(os.path.join(images,image))

fig,ax = plt.subplots(1)
ax.imshow(img)
boxes=get_boxes(image)
for box in boxes:
    rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()


# In[ ]:


df[df["name"]==train_images[11]]


# # PreProcessing for training process

# In[ ]:


path = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/"
train_features = []
train_labels = []
img_size = 128

for image_name in range(3550):
    img = cv2.imread(path + train_images[image_name])
    boxes = get_boxes(train_images[image_name])
    for idx, bb in enumerate(boxes):
        x,y,w,h = bb
        label = list(df[df["name"]==train_images[image_name]]["classname"])
        #if label[idx] == "face_no_mask" or label[idx] == "face_with_mask":
        roi = img[y:h, x:w]
        try:
            roi = cv2.resize(roi, (img_size, img_size), cv2.INTER_AREA)
            train_features.append(roi)
            train_labels.append(label[idx])
        except Exception as e:
            print("[ERROR]")


# In[ ]:


X = np.array(train_features, dtype="float32")
X /= 255.0
y = np.array(train_labels)


# In[ ]:


X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3)


# In[ ]:


classes = ["hijab_niqab", "mask_colorful", "mask_surgical", "face_no_mask",
          "face_with_mask_incorrect", "face_with_mask", "face_other_covering",
          "scarf_bandana", "balaclava_ski_mask", "face_shield", "gas_mask",
          "turban", "helmet", "sunglasses", "eyeglasses", "hair_net", "hat",
          "goggles", "hood", "other"]


# In[ ]:


le = LabelEncoder()
le.fit(y)
y = le.transform(y)
y = to_categorical(y, num_classes=len(classes))


# In[ ]:


# X, y = shuffle(X, y, random_state=2)
# import pickle
# with open("X.pickle","wb") as f1:
#     pickle.dump(X, f1)
# with open("y.pickle","wb") as f2:
#     pickle.dump(y, f2)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)


# In[ ]:


# MY CUSTOM littleVGG
# img_size = 128

# model = Sequential()
# #1st layer
# model.add(Conv2D(64,kernel_size=(3,3),padding="same",input_shape=(img_size,img_size,)))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# #2nd layer
# model.add(Conv2D(64,kernel_size=(3,3)))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# #3rd layer
# model.add(Conv2D(128,kernel_size=(3,3),padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# #4th layer
# model.add(Conv2D(128,kernel_size=(3,3)))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))
# #5th layer
# model.add(Conv2D(256,kernel_size=(3,3),padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# #6th layer
# model.add(Conv2D(256,kernel_size=(3,3)))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.2))

# model.add(Flatten())
# #7th layer
# model.add(Dense(256))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# #8th layer
# model.add(Dense(256))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# #9th layer
# model.add(Dense(len(classes)))
# model.add(Activation("softmax"))


# In[ ]:


img_size = 128

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
for layer in vgg.layers:
    layer.trainable = False
top = vgg.output
top = GlobalAveragePooling2D()(top)
top = Dense(units=256, activation="relu")(top)
top = Dense(units=128, activation="relu")(top)
top = Dense(units=len(classes), activation="softmax")(top)

model = Model(inputs=vgg.input, outputs=top)
print(model.summary())


# In[ ]:


optimizer = Adam(lr=0.001)


# In[ ]:


checkpoint = ModelCheckpoint('face_mask.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)
reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [checkpoint, reduceLR]


# In[ ]:


model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


hist = model.fit(x_train, y_train, batch_size=64, epochs=45, 
                 validation_data=(x_test, y_test), verbose=1,
                callbacks=callbacks)


# In[ ]:


# model.save('face_mask_last.h5')
# import pickle
# pickle_in1 = open('X.pickle', "rb")
# X = pickle.load(pickle_in1)
# pickle_in2 = open('y.pickle', "rb")
# y = pickle.load(pickle_in2)


# # Testing 

# In[ ]:


model = load_model('face_mask.h5')


# In[ ]:


# score = model.evaluate(x_test, y_test)
# score
# accuracy = 80.93% on 10 epochs


# In[ ]:


sub = "/kaggle/input/face-mask-detection-dataset/submission.csv"
subdf = pd.read_csv(sub)
submission_images = list(subdf["name"])
# FILE NAMES IS INCORRECT,SHOULD BE JPEG BUT ITS JPE


# In[ ]:


path = "/kaggle/input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images/"

predicted_classes = []
coordinates = []
image_names = []

detector = MTCNN()

for img_name in submission_images:
    first = img_name.split(".")[0]
    last = img_name.split(".")[1]
    if last == "jpe":
        img_name = first+"."+"jpeg"
    im = cv2.imread(path+img_name)
    color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.asarray(color)
    faces = detector.detect_faces(im)
    for i in range(len(faces)):
        x,y,w,h = faces[i]['box']
        x, y = abs(x), abs(y)
        roi = color[y:y+h, x:x+w]
        roi = cv2.resize(roi, (128,128), cv2.INTER_AREA)
        roi = np.array(roi).astype('float32')
        roi = roi.reshape(1, 128, 128, 3)
        preds = model.predict(roi)
        pred = np.argmax(preds, axis=1)
        predicted_classes.append(classes[int(pred)])
        coordinates.append([x,y,w,h])
        image_names.append(img_name)


# In[ ]:


print("Total size:", len(predicted_classes))
print("Image name:", image_names[6])
print("coordinates:", coordinates[6])
print("predicted class:", predicted_classes[6])


# In[ ]:


df_names = pd.DataFrame(image_names, columns=["name"])
df_coord = pd.DataFrame(coordinates, columns=['x1','x2','y1','y2'], dtype=float)
df_class = pd.DataFrame(predicted_classes, columns=["classname"])


# In[ ]:


dataframes = [df_names, df_coord, df_class]
result = pd.concat(dataframes, axis=1)


# In[ ]:


result.to_csv(r'final_submission.csv')


# In[ ]:


from IPython.display import FileLink
FileLink(r'final_submission.csv')


# In[ ]:


result.head(n=20)


# In[ ]:


subdf.head(n=20)

