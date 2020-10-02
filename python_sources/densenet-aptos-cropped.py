#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">APTOS Diabetic Retinopathy</font></center></h1>
# 
# 
# 
# 
# <img src="https://www.eye7.in/wp-content/uploads/illustration-showing-diabetic-retinopathy.jpg" width="800"></img>
# 
# 
# 
# <br>

# This kernel is forked from **xhlulu's** kernel : https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter
# 
# Things I have added :
# *  Ben's processing on images 
# *  Cropping images 
# * Added EarlyStopping Callback
# * Increased Dropout ...both of these to reduce overfitting.
# 
# Both the ideas are taken from **Neuron Engineer's** kernel : https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
# 
# I thank both of them for their amazing kernels.
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Loading & Exploration</a>  : A quick overview of the dataset
# - <a href='#2'>Ben's,Cropping and Loading images</a> : Here we shall apply Ben's preprocessing, crop the images, resize and load them
# - <a href='#3'>Mixup and Generator</a> :  We show how to create a data generator that will perform random transformation to our datasets (flip vertically/horizontally, rotation, zooming). This will help our model generalize better to the data, since it is fairly small (only ~3000 images). 
# - <a href='#4'> Quadratic Weight Kappa</a>    : A thorough overview of the metric used for this competition, with an intuitive example. Check it out!
# - <a href='#5'>DenseNet121</a>   : We will use a DenseNet-121 pre-trained on ImageNet. We will finetune it using Adam for 15 epochs, and evaluate it on an unseen validation set.   
# - <a href='#6'>Training & Evaluation</a>
# 
# 
# ### Citations & Resources
# 
# * I had the idea of using mixup from [KeepLearning's ResNet50 baseline](https://www.kaggle.com/mathormad/aptos-resnet50-baseline). Since the implementation was in PyTorch, I instead used an [open-sourced keras implementation](https://github.com/yu4u/mixup-generator).
# * The transfer learning procedure is mostly inspired from my [previous kernel for iWildCam](https://www.kaggle.com/xhlulu/densenet-transfer-learning-iwildcam-2019). The workflow was however heavily modified since then.
# * Used similar [method as Abhishek](https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa) to find the optimal threshold.
# * [Lex's kernel](https://www.kaggle.com/lextoumbourou/blindness-detection-resnet34-ordinal-targets) prompted me to try using Multilabel instead of multiclass classification, which slightly improved the kappa score.

# In[ ]:


# To have reproducible results
seed = 5 
import numpy as np 
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)


# # <a id='1'>Loading and Exploration</a> 

# In[ ]:


import json
import math
import os

import cv2
from PIL import Image

from keras import layers
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
print(os.listdir('../input/aptos2019-blindness-detection'))
get_ipython().run_line_magic('matplotlib', 'inline')

IMG_SIZE=256
BATCH_SIZE = 32


# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
# pretrain_df=pd.read_csv('../input/diabetic-retinopathy-resized/trainLabels.csv')
# print(train_df.shape)
# print(test_df.shape)
# print(pretrain_df.shape)
# pretrain_df.head()


# In[ ]:


train_df.head()


# In[ ]:


# train_df['id_code']=train_df['id_code'].apply(lambda x: x+'.png')


# In[ ]:


train_df['diagnosis'].value_counts()


# In[ ]:


# pretrain_df['level'].value_counts()


# In[ ]:


# df0=pretrain_df[pretrain_df['level']==0]
# df1=pretrain_df[pretrain_df['level']==1]
# df2=pretrain_df[pretrain_df['level']==2]
# df3=pretrain_df[pretrain_df['level']==3]
# df4=pretrain_df[pretrain_df['level']==4]

# df0=df0.sample(n=1000,random_state=seed)
# df1=df1.sample(n=1000,random_state=seed)
# df2=df2.sample(n=1000,random_state=seed)
# # df3=df3.sample(n=1500,replace=True,random_state=seed)
# # df4=df4.sample(n=1500,replace=True,random_state=seed)


# In[ ]:


# pre_df=pd.concat([df0,df1,df2,df3,df4],axis=0,ignore_index=True)


# In[ ]:


# pre_df=pre_df.sample(frac=1,random_state=seed).reset_index(drop=True)


# In[ ]:


# pre_df['level'].value_counts().plot.bar()


# In[ ]:


train_df['diagnosis'].hist()
train_df['diagnosis'].value_counts()


# In[ ]:


# pre_df.head()


# ### Displaying some Sample Images

# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.axis('off')
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# In[ ]:


# def display_samples2(df, columns=4, rows=3):
#     fig=plt.figure(figsize=(5*columns, 4*rows))

#     for i in range(columns*rows):
#         image_path = df.loc[i,'image']
#         image_id = df.loc[i,'level']
#         img = cv2.imread(f'../input/diabetic-retinopathy-resized/resized_train/resized_train/{image_path}.jpeg')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         fig.add_subplot(rows, columns, i+1)
#         plt.title(image_id)
#         plt.axis('off')
#         plt.imshow(img)
    
#     plt.tight_layout()

# display_samples2(pre_df)


# # <a id='2'>Ben's preprocessing and Cropping images</a> 
# 
# We will resize the images to 224x224, then create a single numpy array to hold the data.

# In[ ]:


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def preprocess_image(path, sigmaX=40):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

# def preprocess_image(image_path):
#     im = Image.open(image_path)
#     im = im.resize((IMG_SIZE, )*2, resample=Image.LANCZOS)
    
#     return im


# In[ ]:


# fig,ax=plt.subplots(5,5,figsize=(10,10))
# for i in range(25):
#     data=train_df[train_df['diagnosis']==i//5].reset_index()
#     a=np.random.randint(len(data))
#     img_code=data['id_code'][a]
#     img=preprocess_image_with_ben(f'../input/aptos2019-blindness-detection/train_images/{img_code}.png')
#     ax[i//5,i%5].imshow(img)
#     ax[i//5,i%5].axis('off')
#     ax[i//5,i%5].set_title(data['diagnosis'][a])
        
# plt.tight_layout()
# plt.show()


# In[ ]:


# def display_samples3(df, columns=4, rows=3):
#     fig=plt.figure(figsize=(5*columns, 4*rows))

#     for i in range(columns*rows):
#         image_path = df.loc[i,'image']
#         image_id = df.loc[i,'level']
#         img = cv2.imread(f'../input/diabetic-retinopathy-resized/resized_train/resized_train/{image_path}.jpeg')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
#         fig.add_subplot(rows, columns, i+1)
#         plt.title(image_id)
#         plt.axis('off')
#         plt.imshow(img)
    
#     plt.tight_layout()

# display_samples3(pre_df)


# In[ ]:


N = train_df.shape[0]
x_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    )


# In[ ]:


# N = test_df.shape[0]
# x_test = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# for i, image_id in enumerate(tqdm(test_df['id_code'])):
#     x_test[i, :, :, :] = preprocess_image_with_ben(
#         f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
#     )


# In[ ]:


y_train = pd.get_dummies(train_df['diagnosis']).values


print(y_train.shape)


# ## Creating multilabels
# 
# Instead of predicting a single label, we will change our target to be a multilabel problem; i.e., if the target is a certain class, then it encompasses all the classes before it. E.g. encoding a class 4 retinopathy would usually be `[0, 0, 0, 1]`, but in our case we will predict `[1, 1, 1, 1]`. For more details, please check out [Lex's kernel](https://www.kaggle.com/lextoumbourou/blindness-detection-resnet34-ordinal-targets).

# In[ ]:


y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# Now we can split it into a training and validation set.

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=0.15, 
    random_state=2019
)


# In[ ]:


x_val=x_val/255


# # <a id='3'>Mixup Generator</a> 

# In[ ]:


class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


# In[ ]:


# pre_df.head()


# In[ ]:


# pre_df['image']=pre_df['image'].apply(lambda x: x + '.jpeg')


# In[ ]:


# pre_df['level']=pre_df['level'].astype(str)


# In[ ]:


train_df['diagnosis']=train_df['diagnosis'].astype(str)


# In[ ]:




def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,# randomly flip images
        rotation_range=360,
        rescale=1./255
    )

# train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.15,horizontal_flip=True,vertical_flip=True,validation_split=0.2,
#                                 preprocessing_function=preprocess_input)
# val_datagen=ImageDataGenerator(rescale=1./255)
# Using original generator
# train_gen = train_datagen.flow_from_dataframe(pre_df, directory='../input/diabetic-retinopathy-resized/resized_train/resized_train',
#                                                       batch_size=BATCH_SIZE,x_col='image',y_col='level',target_size=(IMG_SIZE,IMG_SIZE),
#                                              subset='training')

# val_generator=train_datagen.flow_from_dataframe(pre_df,directory='../input/diabetic-retinopathy-resized/resized_train/resized_train',
#                                              batch_size=BATCH_SIZE,x_col='image',y_col='level',target_size=(IMG_SIZE,IMG_SIZE),
#                                                subset='validation')
# Using Mixup
# mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()


# In[ ]:


data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)


# In[ ]:


# val_labels=pd.get_dummies(train_df['diagnosis']).values
# val_labels


# In[ ]:


# y_val=[np.argmax(val_generator[i][1],axis=1) for i in range(np.ceil(val_generator.samples/BATCH_SIZE).astype(int))]


# In[ ]:


# y_val=np.concatenate(y_val,axis=0)


# In[ ]:


# train_df.head()


# # <a id='4'>Quadratic Weighted Kappa</a> 
# 
# Quadratic Weighted Kappa (QWK, the greek letter $\kappa$), also known as Cohen's Kappa, is the official evaluation metric. For our kernel, we will use a custom callback to monitor the score, and plot it at the end.
# 
# ### What is Cohen Kappa?
# 
# According to the [wikipedia article](https://en.wikipedia.org/wiki/Cohen%27s_kappa), we have
# > The definition of $\kappa$ is:
# > $$\kappa \equiv \frac{p_o - p_e}{1 - p_e}$$
# > where $p_o$ is the relative observed agreement among raters (identical to accuracy), and $p_e$ is the hypothetical probability of chance agreement, using the observed data to calculate the probabilities of each observer randomly seeing each category.
# 
# ### How is it computed?
# 
# Let's take the example of a binary classification problem. Say we have:

# In[ ]:


true_labels = np.array([1, 0, 1, 1, 0, 1])
pred_labels = np.array([1, 0, 0, 0, 0, 1])


# We can construct the following table:
# 
# | true | pred | agreement      |
# |------|------|----------------|
# | 1    | 1    | true positive  |
# | 0    | 0    | true negative  |
# | 1    | 0    | false negative |
# | 1    | 0    | false negative |
# | 0    | 0    | true negative  |
# | 1    | 1    | true positive  |
# 
# 
# Then the "observed proportionate agreement" is calculated exactly the same way as accuracy:
# 
# $$
# p_o = acc = \frac{tp + tn}{all} = {2 + 2}{6} = 0.66
# $$
# 
# This can be confirmed using scikit-learn:

# In[ ]:


accuracy_score(true_labels, pred_labels)


# Additionally, we also need to compute `p_e`:
# 
# $$p_{yes} = \frac{tp + fp}{all} \frac{tp + fn}{all} = \frac{2}{6} \frac{4}{6} = 0.222$$
# 
# $$p_{no} = \frac{fn + tn}{all} \frac{fp + tn}{all} = \frac{4}{6} \frac{2}{6} = 0.222$$
# 
# $$p_{e} = p_{yes} + p_{no} = 0.222 + 0.222 = 0.444$$
# 
# Finally,
# 
# $$
# \kappa = \frac{p_o - p_e}{1-p_e} = \frac{0.666 - 0.444}{1 - 0.444} = 0.4
# $$
# 
# Let's verify with scikit-learn:

# In[ ]:


cohen_kappa_score(true_labels, pred_labels)


# ### What is the weighted kappa?
# 
# The wikipedia page offer a very concise explanation: 
# > The weighted kappa allows disagreements to be weighted differently and is especially useful when **codes are ordered**. Three matrices are involved, the matrix of observed scores, the matrix of expected scores based on chance agreement, and the weight matrix. Weight matrix cells located on the diagonal (upper-left to bottom-right) represent agreement and thus contain zeros. Off-diagonal cells contain weights indicating the seriousness of that disagreement.
# 
# Simply put, if two scores disagree, then the penalty will depend on how far they are apart. That means that our score will be higher if (a) the real value is 4 but the model predicts a 3, and the score will be lower if (b) the model instead predicts a 0. This metric makes sense for this competition, since the labels 0-4 indicates how severe the illness is. Intuitively, a model that predicts a severe retinopathy (3) when it is in reality a proliferative retinopathy (4) is probably better than a model that predicts a mild retinopathy (1).

# ### Creating keras callback for QWK

# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
# #         X_val, y_val =self.val_generator[0],self.val_generator[1]
# #         y_val = y_val.sum(axis=1) - 1
#         y_pred=self.model.predict_generator(val_generator,steps=np.ceil(val_generator.samples/BATCH_SIZE).astype(int))
# #         y_pred = self.model.predict(X_val) > 0.5
# #         y_pred = y_pred.astype(int).sum(axis=1) - 1
#         y_pred=np.argmax(y_pred,axis=1)

    
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1
        
        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            model.save_weights('model.h5')
            model_json = model.to_json()
            with open('model.json', "w") as json_file:
                json_file.write(model_json)
            json_file.close()

        return


# # <a id='5'>Model : DenseNet121</a> 

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(IMG_SIZE,IMG_SIZE,3)
)


# In[ ]:


# resnet=ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                include_top=False,input_shape=(IMG_SIZE,IMG_SIZE,3))


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
#     for layer in model.layers:
#         layer.trainable=True
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# # <a id='6'>Training and Evaluation</a> 

# In[ ]:


kappa_metrics = Metrics()
# filename='1500images_lr003.h5'
# checkpoint=ModelCheckpoint(filename,monitor='val_loss',save_best_only='True')
est=EarlyStopping(monitor='val_loss',patience=5, min_delta=0.005)
call_backs=[est,kappa_metrics]

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    validation_data=(x_val,y_val),
     epochs=20,
    callbacks=call_backs)


# In[ ]:


# with open('history.json', 'w') as f:
#     json.dump(history.history, f)

# history_df = pd.DataFrame(history.history)
# history_df[['loss', 'val_loss']].plot()
# history_df[['acc', 'val_acc']].plot()


# 

# In[ ]:


# plt.plot(kappa_metrics.val_kappas)


# ## Find best threshold

# In[ ]:


# model.load_weights('model.h5')
# # y_val_pred = model.predict(x_val)

# # def compute_score_inv(threshold):
# #     y1 = y_val_pred > threshold
# #     y1 = y1.astype(int).sum(axis=1) - 1
# #     y2 = y_val.sum(axis=1) - 1
# #     score = cohen_kappa_score(y1, y2, weights='quadratic')
    
# #     return 1 - score

# # simplex = scipy.optimize.minimize(
# #     compute_score_inv, 0.5, method='nelder-mead'
# # )

# # best_threshold = simplex['x'][0]

# # print('Best threshold is : ',best_threshold)

# # print('Validation QWK score: ',(1-compute_score_inv(best_threshold)))


# ## Submit

# In[ ]:


# y_test = model.predict(x_test) > best_threshold
# y_test = y_test.astype(int).sum(axis=1) - 1

# test_df['diagnosis'] = y_test
# test_df.to_csv('submission.csv',index=False)

