#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import scipy as sp
from functools import partial
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


# In[ ]:


IMG_SIZE = 224


# In[ ]:


def load_img(path):
    image = cv2.imread(path)
    return image


# In[ ]:


def remove_unwanted_space(image, threshold=7):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray_image > threshold
    cropped_image = image[np.ix_(mask.any(1), mask.any(0))]
    if cropped_image.shape[0] == 0:
        return image
    return cropped_image


# In[ ]:


def preprocess_img(path):
    image = load_img(path)
    image = remove_unwanted_space(image, 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image,4, cv2.GaussianBlur(image, (0,0), 35), -4, 128)
    image = image/255.0
    return image


# In[ ]:


plt.imshow(preprocess_img('../input/aptos2019-blindness-detection/train_images/'+train_df.loc[1]['id_code']+'.png'))


# In[ ]:


N = train_df.shape[0]
train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
for i, image_id in enumerate(tqdm(train_df['id_code'])):
    train[i,:,:,:] = preprocess_img('../input/aptos2019-blindness-detection/train_images/'+image_id+'.png')


# Let us not take y as a one hot vecor describing output classes. Instead let us convert y as follows:
# 
# 0 : [1, 0, 0, 0, 0]
# 
# 1 : [1, 1, 0, 0, 0]
# 
# 2 : [1, 1, 1, 0, 0]
# 
# 3 : [1, 1, 1, 1, 0]
# 
# 4 : [1, 1, 1, 1, 1]

# The intuition is that the output classes are not independent. A particular stage of the disease will also have symptoms of the previous stage.

# In[ ]:


y = np.zeros((train_df.shape[0],5), dtype=np.int8)


# In[ ]:


for i, diagnosis in enumerate(train_df['diagnosis']):
    for k in range(diagnosis+1):
        y[i,k] = 1


# In[ ]:


y[:5,:]


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.15, random_state=42)


# In[ ]:


BATCH_SIZE = 32


# Let us create generators for training, validation and test,

# In[ ]:


class TestSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size, aug=True):
        self.x = x_set
        self.batch_size = batch_size
        self.aug = aug
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
       
        images = np.empty((batch_x.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for i, image_id in enumerate(batch_x):
            preprocessed_img = preprocess_img('../input/aptos2019-blindness-detection/test_images/'+image_id+'.png')
            if self.aug:
                h_flip = bool(random.getrandbits(1))
                v_flip = bool(random.getrandbits(1))
                rotate = random.randint(0,20)
                transform_parameters={'theta':rotate, 'flip_horizontal': h_flip, 'flip_vertical': v_flip}
                #transform_parameters={'flip_horizontal': h_flip, 'flip_vertical': v_flip}
                preprocessed_img = tf.keras.preprocessing.image.ImageDataGenerator().apply_transform(preprocessed_img, transform_parameters)
            images[i,:,:,:] = preprocessed_img
        return  images


# In[ ]:


train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        rotation_range=15,
    )


# In[ ]:


val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()


# In[ ]:


train_gen = train_data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_gen = val_data_generator.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)


# Add callbacks for calculating quadratic weighted kappa score after each epoch.

# In[ ]:


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data, batch_size = 20):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = metrics.cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f" val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# In[ ]:


kappa_cb = Metrics(((X_val, y_val)))


# In[ ]:


densenet = tf.keras.applications.DenseNet121(include_top=False, weights='../input/densenet121/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(IMG_SIZE, IMG_SIZE, 3))


# In[ ]:


model = tf.keras.Sequential([
    densenet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='sigmoid')
])


# In[ ]:


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.6, patience=6, verbose=1)
class_weights = class_weight.compute_class_weight('balanced', np.unique(np.sum(y_train, axis=1)), np.sum(y_train, axis=1))


# In[ ]:


optimizer = tf.keras.optimizers.Adam(lr=5e-5)


# In[ ]:


steps_per_epoch = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
val_steps_per_epoch = int(np.ceil(X_val.shape[0]/BATCH_SIZE))


# In[ ]:


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_gen, validation_data=(X_val, y_val), steps_per_epoch=steps_per_epoch, 
                              callbacks=[kappa_cb], epochs=25, class_weight=class_weights)


# In[ ]:


model.load_weights('model.h5')


# Let us just take a look at the targets and our model outputs.

# In[ ]:


val_preds = np.squeeze(model.predict_generator(val_gen, steps=val_steps_per_epoch))
list(zip(val_preds, y_val))[:10]


# In[ ]:


pred = val_preds > 0.5
pred = np.sum(pred.astype(np.int8), axis=1)-1
kappa, acc = metrics.accuracy_score(pred, y_val.sum(axis=1)-1), metrics.cohen_kappa_score(y_val.sum(axis=1)-1, pred, weights='quadratic')
print('kappa score: {:.4f}, accuracy: {:.4f}'.format(kappa, acc))


# In[ ]:


sns.heatmap(metrics.confusion_matrix(y_val.sum(axis=1)-1, pred), annot=True, cmap='Blues', fmt='g')


# Now we have to optmize threshold values which determine to which classes our prediction belongs to from our model outputs. Although we have assumed the threshold as  [0.5, 1.5, 2.5, 3.5], we can further improve our quadratic weighted kappa score by optimizing the thresholds.

# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# In[ ]:


optR = OptimizedRounder()
optR.fit(np.sum(val_preds,axis=1), np.sum(y_val, axis=1))
coefficients = optR.coefficients()


# In[ ]:


coefficients


# In[ ]:


sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")


# In[ ]:


test_gen = TestSequence(sample['id_code'], BATCH_SIZE, True)


# It is time for test time augmentation! Since our test data generator randomly augments the images, we make multiple predictions on the image and agree upon the most voted class.

# In[ ]:


prediction = np.empty((5,len(sample)))
for i in tqdm(range(5)):
    pred = model.predict_generator(test_gen) > 0.5
    prediction[i,:] = np.sum(pred.astype(np.int8), axis=1)-1
prediction = prediction.astype(np.int8)


# In[ ]:


def vote(x):
    return np.argmax(np.bincount(x))


# In[ ]:


preds = np.apply_along_axis(vote, 0, prediction)


# In[ ]:


plt.hist(preds)


# In[ ]:


sample.diagnosis = preds
sample.to_csv("submission.csv", index=False)

