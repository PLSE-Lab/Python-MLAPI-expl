#!/usr/bin/env python
# coding: utf-8

# ### About
# Since I am new to learning from image segmentation and kaggle in general I want to share my noteook.
# I saw it is similar to others as it uses the U-net approach. I want to share it anyway because:
# 
# - As said, the field is new to me so I am open to suggestions.
# - It visualizes some of the steps, e.g. scaling, to learn if the methods do what I expect which might be useful to others (I call them sanity checks).
# - Added stratification by the amount of salt contained in the image.
# - Added augmentation by flipping the images along the y axes (thanks to the forum for clarification).
# - Added dropout to the model which seems to improve performance.

# In[ ]:


import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize
from keras import optimizers
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.losses import binary_crossentropy
from keras import backend as K
from tqdm import tqdm_notebook
from imgaug import augmenters as iaa


# # Params and helpers

# In[ ]:


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    #res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    #res[:img_size_ori, :img_size_ori] = img
    #return res
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    #return img[:img_size_ori, :img_size_ori]


# # Loading of training/testing ids and depths
# Reading the training data and the depths, store them in a DataFrame. Also create a test DataFrame with entries from depth not in train.

# In[ ]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


# In[ ]:


this_split=5
debug=False
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
if debug:
    train_df=train_df.iloc[:1000,:]
    depths_df=depths_df.iloc[:1000,:]
    test_df=test_df.iloc[:2000,:]


# In[ ]:





# # Read images and masks
# Load the images and masks into the DataFrame and divide the pixel values by 255.

# In[ ]:


train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]


# In[ ]:


train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]


# # Calculating the salt coverage and salt coverage classes
# Counting the number of salt pixels in the masks and dividing them by the image size. Also create 11 coverage classes, -0.1 having no salt at all to 1.0 being salt only.
# Plotting the distribution of coverages and coverage classes, and the class against the raw coverage.

# In[ ]:


train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)


# In[ ]:


def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(train_df.coverage, kde=False, ax=axs[0])
sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")


# In[ ]:


plt.scatter(train_df.coverage, train_df.coverage_class)
plt.xlabel("Coverage")
plt.ylabel("Coverage class")


# # Plotting the depth distributions
# Separatelty plotting the depth distributions for the training and the testing data.

# In[ ]:


sns.distplot(train_df.z, label="Train")
sns.distplot(test_df.z, label="Test")
plt.legend()
plt.title("Depth distribution")


# # Show some example images

# In[ ]:


# max_images = 60
# grid_width = 15
# grid_height = int(max_images / grid_width)
# fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
# for i, idx in enumerate(train_df.index[:max_images]):
#     img = train_df.loc[idx].images
#     mask = train_df.loc[idx].masks
#     ax = axs[int(i / grid_width), i % grid_width]
#     ax.imshow(img, cmap="Greys")
#     ax.imshow(mask, alpha=0.3, cmap="Greens")
#     ax.text(1, img_size_ori-1, train_df.loc[idx].z, color="black")
#     ax.text(img_size_ori - 1, 1, round(train_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
#     ax.text(1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top")
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
# plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")


# # Build model

# In[ ]:


def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, padding='same')(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='sigmoid')(o)
	return Model(inputs=i, outputs=o)
def step_decay(epoch):
    if epoch<=60:
        return 0.01
    elif epoch<=90:
        return 0.001
    else:
        return 0.0001

def build():
#     sgd=optimizers.SGD(lr=0.01,decay=1e-4,momentum=0.9)
    model = UNet((img_size_target,img_size_target,1),start_ch=16,depth=5,batchnorm=True)
    model.compile(loss=bce_dice_loss, optimizer='adam', metrics=["accuracy"])
    return model


# In[ ]:


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


# In[ ]:



# for i,(trdex,valdex) in enumerate(skf.split(X=train_df.index.values,y=train_df.coverage_class.values)):
#     if i!=this_split:
#         continue
def valid_on_best_iou(x_valid,y_valid,time):
    threshes=[]
    model = load_model(str(this_split)+"_"+str(time)+'_keras.model',{'bce_dice_loss': bce_dice_loss})
    preds_valid = model.predict(x_valid).reshape(-1, img_size_target, img_size_target)
    preds_valid = np.array([downsample(x) for x in preds_valid])
    y_valid = np.array([downsample(x) for x in y_valid])
    thresholds = np.linspace(0, 1, 50)
    ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    threshes.append(threshold_best)
#     print(thresholds,ious)
#     print(threshold_best,iou_best)
    return threshold_best,iou_best,thresholds,ious
    


# In[ ]:


seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip
    iaa.OneOf([
        iaa.Noop(),
        iaa.Noop(),
        iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
    ])
])
def gen_flow_for_two_inputs(X, y):
    genX1 = gen.flow(X,y,  batch_size=batch_size)
    while True:
        X=genX1.next()
        img_mask=np.concatenate([X[0],X[1]],axis=3)
        img_mask_aug=seq.augment_images(img_mask)
        yield np.expand_dims(img_mask_aug[:,:,:,0],axis=3),np.expand_dims(img_mask_aug[:,:,:,1],axis=3)
#         imgs=[]
#         masks=[]
#         for index in range(len(X[0])):
#             img=X[0][index]
#             mask=X[0][index]
#             img,mask,_=train_augment(img,mask,0)
#             imgs.append(img)
#             masks.append(mask)
#         imgs=np.array(imgs)
#         masks=np.array(masks)
#         yield np.expand_dims(imgs,axis=3),np.expand_dims(masks,axis=3)
#         yield X[0], X[1]


# In[ ]:


print(type(depths_df.z.values[1]))


# # Training

# In[ ]:


from sklearn.model_selection import StratifiedKFold


n_split=10

models=[]
historys=[]
ious_list=[]
epochs = 90
batch_size=32
val_loss_limit=0.29
val_iou_limit=0.796
if debug:
    epochs=2
    n_split=4
    val_loss_limit=1
    val_acc_limit=0.1
#print(train_df.index.values)
skf=StratifiedKFold(n_splits=n_split)
for i,(trdex,valdex) in enumerate(skf.split(X=train_df.index.values,y=train_df.coverage_class.values)):
    f=open('train_3600_'+str(i),'w')
    for ids in train_df.index.values[trdex]:
        f.write('train/'+ids+'\n')
    f.close()
    f=open('valid_400_'+str(i),'w')
    for ids in train_df.index.values[valdex]:
        f.write('train/'+ids+'\n')
    f.close()
#     print(trdex)
#     f1=open('train_3200_sallow_'+str(i),'w')
#     f2=open('train_3200_mid_'+str(i),'w')
#     f3=open('train_3200_deep_'+str(i),'w')
#     for ids in train_df.index.values[trdex]:
#         if depths_df.loc[ids,'z']<=400:
#             f1.write('train/'+ids+'\n')
#         elif depths_df.loc[ids,'z']<=700:
#             f2.write('train/'+ids+'\n')
#         else:
#             f3.write('train/'+ids+'\n')
#     f1.close()
#     f2.close()
#     f3.close()
#     f1=open('valid_800_sallow_'+str(i),'w')
#     f2=open('valid_800_mid_'+str(i),'w')
#     f3=open('valid_800_deep_'+str(i),'w')
#     for ids in train_df.index.values[valdex]:
#         if depths_df.loc[ids,'z']<=400:
#             f1.write('train/'+ids+'\n')
#         elif depths_df.loc[ids,'z']<=700:
#             f2.write('train/'+ids+'\n')
#         else:
#             f3.write('train/'+ids+'\n')
#     f1.close()
#     f2.close()
#     f3.close()
#     if i!=this_split:
#         print(i,this_split)
#         continue
#     ids_train=train_df.index.values[trdex]
#     x_train=np.array(train_df.loc[ids_train].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
#     y_train=np.array(train_df.loc[ids_train].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
#     #Augmengtation
# #     x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
# #     y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
#     #Valid Set
#     ids_valid=train_df.index.values[valdex]
#     y_valid=np.array(train_df.loc[ids_valid].masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
#     x_valid=np.array(train_df.loc[ids_valid].images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1) 
#     #Visualization
# #     fig, axs = plt.subplots(2, 10, figsize=(15,3))
# #     for j in range(10):
# #         axs[0][j].imshow(x_train[j].squeeze(), cmap="Greys")
# #         axs[0][j].imshow(y_train[j].squeeze(), cmap="Greens", alpha=0.3)
# #         axs[1][j].imshow(x_train[int(len(x_train)/2 + j)].squeeze(), cmap="Greys")
# #         axs[1][j].imshow(y_train[int(len(y_train)/2 + j)].squeeze(), cmap="Greens", alpha=0.3)
# #     fig.suptitle("Top row: original images, bottom row: augmented images")
#     #Train
#     early_stopping = EarlyStopping(patience=10, verbose=1)
#     reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
#     #reduce_lr=LearningRateScheduler(step_decay)
#     best_model=None
#     best_history=None
#     best_iou_max=0
#     best_thres_max=0
#     val_loss_min=99
#     for time in range(6):
#         model_checkpoint = ModelCheckpoint("./"+str(i)+"_"+str(time)+"_keras.model", save_best_only=True, verbose=1)
#         model=build()
#         gen = ImageDataGenerator()
#         gen_flow = gen_flow_for_two_inputs(x_train, y_train)
#         history = model.fit_generator(gen_flow,
#                     validation_data=[x_valid, y_valid], 
#                     epochs=epochs,
#                     steps_per_epoch=len(x_train) / batch_size,
#                     callbacks=[early_stopping, model_checkpoint, reduce_lr])
# #         history = model.fit(x_train, y_train,
# #                         validation_data=[x_valid, y_valid], 
# #                         epochs=epochs,
# #                         batch_size=batch_size,
# #                         callbacks=[early_stopping, model_checkpoint, reduce_lr],shuffle=True,verbose=0)
#         t_b,i_b,ts,bs=valid_on_best_iou(x_valid,y_valid,time)
#         ious_list.append([min(history.history['val_loss']),i_b])
#         print('Splits: '+str(i+1)+
#               ', \nTime: '+ str(time)+
#               ', \nbest epoch valid loss: '+
#               str(min(history.history['val_loss']))+
#               ', \nvalid accuracy: '+
#               str(max(history.history['val_acc']))+
#               ', \nvalid iou: '+
#               str(i_b))
#         if min(history.history['val_loss'])<val_loss_min and i_b>best_iou_max:
#             best_model=load_model("./"+str(i)+"_"+str(time)+"_keras.model",{'bce_dice_loss': bce_dice_loss})
#             best_history=history
#             val_loss_min=min(history.history['val_loss'])
#             best_iou_max=i_b
#             best_thres_max=t_b
#         if min(history.history['val_loss'])<val_loss_limit and i_b>val_iou_limit:
#             best_model=load_model("./"+str(i)+"_"+str(time)+"_keras.model",{'bce_dice_loss': bce_dice_loss})
#             best_history=history
#             val_loss_min=min(history.history['val_loss'])
#             best_iou_max=i_b
#             best_thres_max=t_b
#             break
#     print(min(best_history.history['val_loss']))
#     print(best_iou_max,i_b)
#     print(best_thres_max,t_b)
#     best_model.save("./"+str(i)+"_keras_"+str(best_iou_max)+".model")
#     print(ious_list)


# In[ ]:


ls


# In[ ]:


# avg=0
# for i,h in enumerate(historys):
#     print(min(h.history['val_loss']))
#     avg+=min(h.history['val_loss'])
# avg=avg/len(historys)
# print('avg val loss: '+str(avg))
# avg_acc=0
# for i,h in enumerate(historys):
#     print(max(h.history['val_acc']))
#     avg_acc+=max(h.history['val_acc'])
# avg_acc=avg_acc/len(historys)
# print('avg val acc: '+str(avg_acc))


# # Validation

# # Submission
# Load, predict and submit the test image predictions.

# In[ ]:


# # Source https://www.kaggle.com/bguberfain/unet-with-depth
# def RLenc(img, order='F', format=True):
#     """
#     img is binary mask image, shape (r,c)
#     order is down-then-right, i.e. Fortran
#     format determines if the order needs to be preformatted (according to submission rules) or not

#     returns run length as an array or string (if format is True)
#     """
#     bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
#     runs = []  ## list of run lengths
#     r = 0  ## the current run length
#     pos = 1  ## count starts from 1 per WK
#     for c in bytes:
#         if (c == 0):
#             if r != 0:
#                 runs.append((pos, r))
#                 pos += r
#                 r = 0
#             pos += 1
#         else:
#             r += 1

#     # if last run is unsaved (i.e. data ends with 1)
#     if r != 0:
#         runs.append((pos, r))
#         pos += r
#         r = 0

#     if format:
#         z = ''

#         for rr in runs:
#             z += '{} {} '.format(rr[0], rr[1])
#         return z[:-1]
#     else:
#         return runs


# In[ ]:


# x_test = np.array([upsample(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale=True))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)


# In[ ]:


# preds_test=np.zeros([len(x_test),img_size_target, img_size_target],dtype=np.float32)
# avg_thres=0
# test_batch_length=1000
# for i in range(n_split):
#     model = load_model("./"+str(i)+"_keras.model")
#     avg_thres+=threshes[i]*(min(historys[i].history['val_loss']))
#     for b in range(int(len(x_test)/test_batch_length)):
#         print(str(i)+' split: '+str(b)+' batch')
#         x_test_batch=x_test[b*test_batch_length:(b+1)*test_batch_length]
#         preds_test_batch = model.predict(x_test_batch)
#         preds_test[b*test_batch_length:(b+1)*test_batch_length]+=(preds_test_batch).astype(np.float32).squeeze()*(min(historys[i].history['val_loss']))
# preds_test/=(avg*n_split)
# avg_thres/=(avg*n_split)


# In[ ]:


# avg_thres=0
# for i in range(n_split):
#     model = load_model("./"+str(i)+"_keras.model")
#     avg_thres+=threshes[i]
#     if i==0:
#         preds_test = model.predict(x_test)
#     else:
#         preds_test+=model.predict(x_test)
# preds_test/=n_split
# avg_thres/=n_split


# In[ ]:


# pred_dict = {idx: RLenc(np.round(downsample(preds_test[i]) > avg_thres)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}


# In[ ]:


# sub = pd.DataFrame.from_dict(pred_dict,orient='index')
# sub.index.names = ['id']
# sub.columns = ['rle_mask']
# sub.to_csv('submission.csv')

