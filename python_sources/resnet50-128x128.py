#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install keras==2.2.4')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import shutil

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Preprocessing input

# ### Importing train.csv

# In[ ]:


from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

batch_size = 256
input_shape = (128, 128)
input_shape_2 = (128, 128, 3)


# In[ ]:


train_df = pd.read_csv("../input/humpback-whale-identification/train.csv")
train_df.head()


# In[ ]:


bbox = pd.read_csv('../input/generating-whale-bounding-boxes/bounding_boxes.csv')
bbox.head()


# In[ ]:


from PIL import Image as pimg
from scipy.misc import imresize

image_name = '72c3ce75c.jpg'
img = image.load_img("../input/humpback-whale-identification/train/"+image_name)
x = image.img_to_array(img)
x = np.uint8(x)
details = bbox[bbox['Image']==image_name]
new_x = x[int(details.x0):int(details.x1), int(details.y0):int(details.y1)]

plt.figure(1)
plt.imshow(x)
plt.figure(2)
plt.imshow(new_x)

newnew = imresize(new_x, size=input_shape_2)
plt.figure(3)
plt.imshow(newnew)


# In[ ]:


def prepareImages_bbox(data, bbox, m, dataset, preprocess=False):
    print("Preparing images")
    X_train = np.zeros((m, 128, 128, 3))
    count = 0
    
    for fig in data['Image']:
        # Load images into images of size 100x100x3
#         if not preprocess:
#             img = image.load_img("../input/humpback-whale-identification/"+dataset+"/"+fig)
#             x = image.img_to_array(img)
#             x0 = int(bbox[bbox['Image']==fig].x0)
#             x1 = int(bbox[bbox['Image']==fig].x1)
#             y0 = int(bbox[bbox['Image']==fig].y0)
#             y1 = int(bbox[bbox['Image']==fig].y1)
#             if not (x0 >= x1 or y0 >= y1):
#                 x = x[y0:y1, x0:x1,:]
#             else :
#                 x = x[x0:x1, y0:y1, :]
#             try:
#                 x = imresize(x, size=(128, 128, 3))
#             except:
#                 print(x.shape)
#                 break
#         else:
        img = image.load_img("../input/humpback-whale-identification/"+dataset+"/"+fig, target_size=(128,128,3))
        x = image.img_to_array(img)
        if preprocess == True:
            x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

# dtypes = {
#     'Image' : 'str',
#    'Id' : 'str',
# }

# train_dir = "../input/humpback-whale-identification/train"
# test_dir = "../input/humpback-whale-identification/test"

# df = pd.read_csv('../input/humpback-whale-identification/train.csv', dtype = dtypes)
# df = df[df['Id'] != 'new_whale']

# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# train_generator = datagen.flow_from_dataframe(
#     dataframe = df,
#     directory = train_dir,
#     x_col = "Image",
#     y_col = "Id",
#     has_ext = True,
#     classes = np.unique(df.Id.values).tolist(),
#     class_mode = "categorical",
#     target_size = input_shape,
#     batch_size = batch_size
# )


# In[ ]:


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder


# In[ ]:


'''
Let's remove the new_whale class and see the results now
'''
new_whale_excluded = train_df[train_df['Id'] != 'new_whale']

new_whale_excluded = (new_whale_excluded.reset_index()).drop(columns='index')


# In[ ]:


X_val = prepareImages_bbox((new_whale_excluded[int(len(new_whale_excluded)*0.9):]).reset_index(), bbox, len((new_whale_excluded[int(len(new_whale_excluded)*0.9):])), "train")
X = prepareImages_bbox((new_whale_excluded[:int(len(new_whale_excluded)*0.9)]).reset_index(), bbox, len((new_whale_excluded[:int(len(new_whale_excluded)*0.9)])), "train")


# In[ ]:


y, label_encoder = prepare_labels(new_whale_excluded['Id'])
y_val = y[int(len(new_whale_excluded)*0.9):]
y = y[:int(len(new_whale_excluded)*0.9)]


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

'''
Additional parameters if you want to use 
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
'''

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=60,
    width_shift_range=0.2,
    shear_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


datagen.fit(X)


# ## Building the network (ResNet50)

# In[ ]:


from keras.applications.resnet50 import ResNet50

conv_base = ResNet50(include_top = False,
                    weights = 'imagenet',
                    input_shape = (input_shape_2))

set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'res5b_branch2a':
        set_trainable = True
        print('Got here')
    if set_trainable:
        layer.trainable = True
    else :
        layer.trainable = False
        
conv_base.summary()


# ## Add the Fully connected layers on top of the conv_base

# 
# ### Define the model

# In[ ]:


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, categorical_crossentropy, top_k_categorical_accuracy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Lambda


'''
Build_model() function to generate a generic model 
'''

def build_model(conv_base):
    model = Sequential()
    model.add(Lambda(preprocess_input, name='preprocessing', input_shape=(128, 128, 3)))
    model.add(conv_base)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(5004, activation='softmax')) # 5005 classes
    
    model.compile(optimizer=Adam(lr=0.001, decay=1e-6),
                 loss='categorical_crossentropy',
                 metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
    
    model.summary()
    
    return model


# 
# ### Fitting the model

# In[ ]:


model = build_model(conv_base)

reduce_lr = ReduceLROnPlateau(monitor='val_top_5_accuracy', factor=0.2,
                              patience=5, min_lr=0.0005)

early = EarlyStopping(monitor='val_top_5_accuracy', mode='max', patience=5)

history = model.fit_generator(datagen.flow(X, y, batch_size=batch_size), validation_data=(X_val, y_val), epochs=50, verbose=1, callbacks=[reduce_lr, early])


# In[ ]:


# model = build_model(conv_base)

# train_samples = train_generator.filenames
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
#                               patience=1, min_lr=0.0005)

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=len(train_samples)/batch_size,
#     epochs=50,
#     callbacks=[reduce_lr]
# )


# In[ ]:


'''
Plotting loss and accuracy
'''

import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
top5_acc = history.history['top_5_accuracy']
val_acc = history.history['val_categorical_accuracy']
val_top5_acc = history.history['val_top_5_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(2)

plt.plot(epochs, top5_acc, 'bo', label='Training top_5_accuracy')
plt.plot(epochs, val_top5_acc, 'b', label='Validation top_5_accuracy')
plt.title('Training and validation top5_accuracy')
plt.legend()

plt.figure(3)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# ### Save model

# In[ ]:


# model.save('resnet-v3.h5')


# ### Create test directory

# In[ ]:


sample_sub = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")
test = list(sample_sub.Image)
print(len(test))

col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

test_dir = "../input/humpback-whale-identification/test"

test_generator = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = test_dir,
    x_col = "Image",
    y_col = "Id",
    has_ext = True,
    class_mode = None,
    target_size = input_shape,
    batch_size = batch_size
)


# ### Convert predictions back to their original name

# In[ ]:


# X = prepareImages_bbox(test_df, bbox, test_df.shape[0], "test", preprocess=True)
unique_labels = np.unique(new_whale_excluded['Id'].values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])


# In[ ]:


test_samples = test_generator.filenames
print(len(test_samples))

test_generator.reset()
predictions = model.predict_generator(
    test_generator,
    steps=len(test_samples)/batch_size,
    verbose=1
)


# In[ ]:


print(len(labels_list))


# In[ ]:


best_th = 0.38

preds_t = np.concatenate([np.zeros((predictions.shape[0],1))+best_th, predictions],axis=1)
np.save("preds.npy",preds_t)


# In[ ]:


sample_df = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")
sample_list = list(sample_df.Image)
labels_list = ["new_whale"]+labels_list
pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(test_samples,pred_list))
pred_list_cor = [' '.join(pred_dic[id]) for id in sample_list]
df = pd.DataFrame({'Image':sample_list,'Id': pred_list_cor})
df.to_csv('submission.csv', header=True, index=False)
df.head()

