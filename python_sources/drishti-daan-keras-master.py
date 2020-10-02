#!/usr/bin/env python
# coding: utf-8

# 
# ![](https://www.eye7.in/wp-content/uploads/illustration-showing-diabetic-retinopathy.jpg)
# 
# Increased urbanization, the consumption of less-nutritious foods, more sedentary lifestyles and resulting obesity have all contributed to the dramatic rise in the global prevalence of diabetes, particularly in resource-poor countries.
# 
# ![](http://atlas.iapb.org/wp-content/uploads/VA-DR-Map-new3.gif)
# 
# Currently, South East Asia and the Western Pacific account for more than half of adults with diabetes world-wide. Efforts to reduce its prevalence and more effectively manage its health consequences are further undermined by the fact that roughly half of all people with diabetes are currently undiagnosed.
# 
# Every person living with diabetes is at risk of developing Diabetic Retinopathy (DR). Poorly controlled blood sugars, high blood pressure and high cholesterol increase the likelihood of vision loss due to DR, along with the risk of associated vision disorders such as Cataract or Glaucoma.
# 
# DR is the leading cause of vision loss in working-age adults (20 to 65 years) so it has profound economic consequences from lost productivity and the cost of ongoing care. Approximately one in three people living with diabetes have some degree of DR and one in 10 will develop a vision-threatening form of the disease . Over time, poor glycemic control can result in improper growth or blockage of blood vessels that nourish the retina and lead to leakage of blood, fluids and the formation of lipid deposits in the eye. In more advanced forms of DR, new abnormal vessel growth occurs due to reduced oxygen flow caused by damaged or blocked vessels. The resulting retinal scarring, retinal detachment, along with fluid build-up and swelling in the central part of the retina (the macula), lead to impaired vision. Damage to the retina is often irreversible in the later stages of the disease and results in blindness.
# 
# **What are the stages of Diabetic Retinopathy?**
# 
# Diabetic Retinopathy has four stages:
# 
# *     Mild Nonproliferative Retinopathy
# *     Moderate Nonproliferative Retinopathy
# *     Severe Nonproliferative Retinopathy
# *     Proliferative Retinopathy
# 
# 
# 
# **How does Diabetic Retinopathy cause vision loss?**
# 
# Blood vessels damaged from Diabetic Retinopathy can cause vision loss in two ways:
# 
# *     Fragile, abnormal blood vessels can develop and leak blood into the center of the eye, blurring vision. This is proliferative retinopathy and is the fourth and most advanced stage of the disease.
# *     Fluid can leak into the center of the macula, the part of the eye where sharp, straight-ahead vision occurs. The fluid makes the macula swell, blurring vision. This condition is called Macular Edema. It can happen at any stage of Diabetic Retinopathy, although it is more likely to occur as the disease progresses. About half of the people with proliferative retinopathy also have Macular Edema.
# 

# **Global prevalence of people with diabetes and Diabetic Retinopathy**
# 
# The Vision Loss Expert Group estimated that in 2015 , some 1.07% of blindness world-wide could be attributed to Diabetic Retinopathy; this is predicted to increase as the global prevalence of diabetes continues to rise. Although early identification and treatment can prevent almost all blindness from DR, people living with diabetes are often unaware that they should have their vision examined annually, are asymptomatic during the early stages of DR, and fail to access timely care. In most resource-poor countries, the clinicians, medical technology and systems of care needed to identify and effectively treat DR are often lacking, so significant investment will be required in order to forestall the inevitable rise in vision loss to DR.
# 
# ![](http://atlas.iapb.org/wp-content/uploads/VA-DR-infographic-resized.gif)
# 
# 
# 
# Diabetes prevention and treatment can play a vital role in reducing vision loss from DR. Although more than 75% of people with diabetes will develop some retinopathy after 15 years, recent studies have determined that intensive glucose therapy achieved through diet and/or medication can reduce the onset of Diabetic Retinopathy by 76% and the progression of the disease by 54%.
# 
# The most effective Diabetic Retinopathy programmes will take a holistic approach, focusing on patient education, behaviour change, and effective disease management strategies in addition to the provision of annual vision exams and high-quality, affordable treatment. Increased co-operation between the diabetes care and ophthalmic communities is essential to prevent the impending epidemic of vision loss due to Diabetic Retinopathy.
# 
# **How is Diabetic Retinopathy treated?**
# 
# During the first three stages of Diabetic Retinopathy, no treatment is needed, unless you have Macular Edema. To prevent progression of Diabetic Retinopathy, people with diabetes should control their levels of blood sugar, blood pressure, and blood cholesterol.
# 
# Proliferative Retinopathy is treated with laser surgery. This procedure is called Panretinal Photocoagulation which helps to shrink the abnormal blood vessels. Your doctor places 1,000 to 2,000 laser burns in the areas of the retina away from the macula, causing the abnormal blood vessels to shrink. Because a high number of laser burns are necessary, two or more sessions usually are required to complete treatment. Although you may notice some loss of your side vision, Laser Photocoagulation can save the rest of your sight.
# 
# Laser Photocoagulation works better before the fragile, new blood vessels have started to bleed. That is why it is essential to have regular, comprehensive dilated eye exams. Even if bleeding has started, laser treatment may still be possible, depending on the amount of bleeding.
# 
# If the bleeding is severe, you may need a surgical procedure called a vitrectomy. During a vitrectomy, blood is removed from the center of your eye.
# 

# In[ ]:


import json
import math
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.random.seed(2019)
tf.set_random_seed(2019)


# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
train_df['diagnosis'].hist()


# In[ ]:


test_df.head()


# **A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:**
# 
# 
# 
#     0 - No DR
# 
#     1 - Mild
# 
#     2 - Moderate
# 
#     3 - Severe
# 
#     4 - Proliferative DR
# 
# 

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
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


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



def circle_crop(img, sigmaX=45):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.resize(img, (224, 224))
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 


# In[ ]:


N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = circle_crop(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png')


# In[ ]:


N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = circle_crop(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png')


# In[ ]:


y_train = pd.get_dummies(train_df['diagnosis']).values

y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=0.15, 
    random_state=2019
)


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


BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)
# Using Mixup
mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()


# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
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
            self.model.save('model.h5')

        return


# In[ ]:


from keras.applications import DenseNet169
densenet = DenseNet169(
    weights='../input/densenet-keras/DenseNet-BC-169-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.0005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


kappa_metrics = Metrics()


# In[ ]:




history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


model.load_weights('model.h5')
y_val_pred = model.predict(x_val)

def compute_score_inv(threshold):
    y1 = y_val_pred > threshold
    y1 = y1.astype(int).sum(axis=1) - 1
    y2 = y_val.sum(axis=1) - 1
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return 1 - score

simplex = scipy.optimize.minimize(
    compute_score_inv, 0.5, method='nelder-mead'
)

best_threshold = simplex['x'][0]


# In[ ]:


y_test = model.predict(x_test) > 0.5
y_test = y_test.astype(int).sum(axis=1) - 1

test_df['diagnosis'] = y_test


# In[ ]:


def xdisplay_samples(df, columns=4, rows=12):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        #img = cv2.imread(f'../input/aptos2019-blindness-detection/test_images/{image_path}.png')
        image = circle_crop(f'../input/aptos2019-blindness-detection/test_images/{image_path}.png')
        
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(image)
    
    plt.tight_layout()

xdisplay_samples(test_df)


# **This Images are the preprocessed one with the predicted classes **

# In[ ]:


model.save("modelx.h5")


# In[ ]:


test_df.to_csv('submission.csv',index=False)


# In[ ]:




