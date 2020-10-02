#!/usr/bin/env python
# coding: utf-8

# ## 5.Required Packages

# In[ ]:


from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from scipy.stats import logistic
from os.path import join
from tqdm import tqdm
from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Input, Dense, Dropout, Lambda, Convolution2D, MaxPooling2D, Flatten, Conv2D, BatchNormalization
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D,     Lambda, MaxPooling2D, Reshape
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
for i in [DeprecationWarning,FutureWarning,UserWarning]:
    warnings.filterwarnings("ignore", category = i)

print(os.listdir("../input/"))


# ## 6.Define Parameter

# In[ ]:


bbox = pd.read_csv("../input/generating-whale-bounding-boxes/bounding_boxes.csv")
print(bbox.head())


# In[ ]:


batch_size = 24
embedding_dim = 128
image_size = 192
path_base = '../input/humpback-whale-identification/'
path_train = join(path_base,'train')
path_test = join(path_base,'test')
path_model = join(path_base,'MyModel.hdf5')
path_csv = '../input/humpback-whale-identification/train.csv'


# ## 7.Helping Function

# In[ ]:


class sample_gen(object):
    def __init__(self, file_class_mapping):
        self.file_class_mapping= file_class_mapping
        self.class_to_list_files = defaultdict(list)
        self.list_other_class = []
        self.list_all_files = list(file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_class_mapping.items():
            self.class_to_list_files[class_].append(file)
        
        self.list_classes = list(set(self.class_to_list_files.keys()))
        self.range_list_classes= range(len(self.list_classes))
        
        print("class_to_list_files, list_classes", len(self.class_to_list_files), len(self.list_classes))
        
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = self.class_weight/np.sum(self.class_weight)

    def get_sample(self):
        class_idx = np.random.choice(self.range_list_classes, 1)[0]
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)
        positive_example_1 = self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]]
        positive_example_2 = self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]
        pos_binary = np.random.randint(2, size=1)
        
        if pos_binary[0] == 1:
            return positive_example_1, positive_example_2
        
        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] ==                 self.file_class_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        return positive_example_1, negative_example
    
def read_and_resize(filepath):
    #print(filepath[-13:])
    _, _, x0, y0, x1, y1 = list(bbox[bbox['Image'] == filepath[-13:]].reset_index().loc[0])
    #print(x0, y0, x1, y1)
    if x0 >= x1 or y0 >= y1:
        x0 = 0
        y0 = 0
        x1 = image_size
        y1 = image_size
    im = Image.open(filepath).convert('RGB')
    im = np.array(im, dtype="float32")
    #print(im.shape, 'y')
    imout = im[y0:y1, x0:x1, :]
    #print(imout.shape, 'x')
    imout = cv2.resize(imout, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)
    #print(imout.shape)
    return imout


def augment(im_array, fixed=False):
    if fixed == True or np.random.uniform(0, 1) > 0.8:
        im_array = np.fliplr(im_array)
    return im_array

def gen(couplet_gen):
    while True:
        list_examples_1 = []
        list_examples_2 = []
        labels = []
        #print("ONE CALL")

        for i in range(batch_size):
            example_1, example_2 = couplet_gen.get_sample()
            path_pos1 = join(path_train, example_1)
            path_pos2 = join(path_train, example_2)
            
            example_1_img = read_and_resize(path_pos1)
            example_2_img = read_and_resize(path_pos2)
            
            example_1_img = augment(example_1_img)
            if example_1 == example_2:
                example_2_img = augment(example_2_img, fixed=True)
                label = 1
            else:
                example_2_img = augment(example_2_img)
                label = 0
            
            list_examples_1.append(example_1_img)
            list_examples_2.append(example_2_img)
            labels.append(label)
        
        list_examples_1 = preprocess_input(np.array(list_examples_1))
        list_examples_2 = preprocess_input(np.array(list_examples_2))
        labels = np.array(labels)
        
        yield ([list_examples_1, list_examples_2], labels)


# ## 8.Introduction to Triplet Loss 

# In[ ]:


def triplet_loss(inputs, dist='euclidean', margin='softplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = 0.6 + positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

def triplet_loss_np(inputs, dist='euclidean', margin='softplus'):
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = .6 + positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)

def check_loss():
    batch_size = 10
    shape = (batch_size, 4096)

    p1 = normalize(np.random.random(shape))
    n = normalize(np.random.random(shape))
    p2 = normalize(np.random.random(shape))
    
    input_tensor = [K.variable(p1), K.variable(n), K.variable(p2)]
    out1 = K.eval(triplet_loss(input_tensor))
    input_np = [p1, n, p2]
    out2 = triplet_loss_np(input_np)

    assert out1.shape == out2.shape
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


# In[ ]:


check_loss()


# ## 9.Model Design

# In[ ]:


def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def build_model(lr, l2, activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=(image_size, image_size, 3))  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, **kwargs)

    x = GlobalMaxPooling2D()(x)  # 512
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])
    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=(image_size, image_size, 3))
    img_b = Input(shape=(image_size, image_size, 3))
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model


model, branch_model, head_model = build_model(64e-5, 0)


# In[ ]:


data = pd.read_csv(path_csv)
print(data.shape)
data = data[data['Id'] != 'new_whale']
print(data.shape)
train, test = train_test_split(data, train_size=0.7, random_state=1337)
file_id_mapping_train = {k: v for k, v in zip(train.Image.values, train.Id.values)}
file_id_mapping_test = {k: v for k, v in zip(test.Image.values, test.Id.values)}
gen_tr = gen(sample_gen(file_id_mapping_train))
gen_te = gen(sample_gen(file_id_mapping_test))

checkpoint = ModelCheckpoint(path_model, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min',factor=0.5, patience=5, min_lr=0.00000001, verbose=1)
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
model_checkpoint = ModelCheckpoint('siamese.model',monitor='val_loss', 
                                   mode = 'min', save_best_only=True, verbose=1)
callbacks_list = [model_checkpoint, reduce_lr]  # early


# In[ ]:


def ShowImg(img):
    plt.figure(figsize=(15,8))
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()

batch = next(gen_tr)

img = batch[0][0][0]
print(img.shape)
mean = [103.939, 116.779, 123.68]
img[..., 0] += mean[0]
img[..., 1] += mean[1]
img[..., 2] += mean[2]
img = img[..., ::-1]
ShowImg(img)


# # Installation of Resnet 50 Weight to keras

# In[ ]:


history = model.fit_generator(gen_tr,
                              validation_data = gen_te,
                              epochs=20, 
                              verbose=1, 
                              workers=4,
                              steps_per_epoch=200,
                              validation_steps = 80,
                              use_multiprocessing=True)


# In[ ]:


import pickle
with open('siamese.pickle', 'wb') as handle:
    pickle.dump(model.get_weights(), handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


def data_generator(fpaths, batch=16):
    i = 0
    imgs = []
    fnames = []
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        img = read_and_resize(path)
        imgs.append(img)
        fnames.append(os.path.basename(path))
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            yield fnames, imgs
            
    if i != 0:
        imgs = np.array(imgs)
        yield fnames, imgs
        
    raise StopIteration()


# In[ ]:


data = pd.read_csv(path_csv)
file_id_mapping = {k: v for k, v in zip(data.Image.values, data.Id.values)}
import glob

train_files = glob.glob(join(path_train, '*.jpg'))
test_files = glob.glob(join(path_test, '*.jpg'))
print(len(train_files), len(test_files))


# In[ ]:


fdict = {}
counts = {}
#print(path_train)
for file in train_files:
    #print(str(file[15:]))
    img = read_and_resize(file)
    img = img.reshape((1, image_size, image_size, 3))
    class_ = file_id_mapping[str(file[-13:])]              
    features = embedding_model.predict(img)
    if class_ not in fdict:
        fdict[class_] = features
        counts[class_] = 1
    else:
        fdict[class_] += features
        counts[class_] += 1


# In[ ]:


for class_ in fdict:
    fdict[class_] /= counts[class_]

for file in train_files[:500]:
    #print(str(file[15:]))
    trueclass = file_id_mapping[str(file[-13:])]
    if trueclass == 'new_whale':
        continue
    img = read_and_resize(file)
    img = img.reshape((1, image_size, image_size, 3))
    dists = []
    cl = []
    """
    for c in fdict:
        dist = features - fdict[c]
        dists.append(dist)
        cl.append(c)
    """
    for _file in train_files:
        _class = file_id_mapping[str(_file[-13:])]
        _img = read_and_resize(_file)
        _img = _img.reshape((1, image_size, image_size, 3))
        dist = model.predict(img, _img)
        dists.append(dist)
        cl.append(_class)
    res = [x for _,x in sorted(zip(dists, cl))]
    d = [x for x,_ in sorted(zip(dists, cl))]
    trueclass = file_id_mapping[str(file[-13:])]
    print(res.index(trueclass), trueclass, counts[trueclass], d[:5])
            

