#!/usr/bin/env python
# coding: utf-8

# # Why bother?
# As one of the main purpose of any competition is making a better score so why shold we even bother about this in the first Place? I think model building, training and evaluating is like attending an exam. For exam, we prepare , we attend exam and then we get results. After result we get to see our mistakes so that we can correct them. So, after training and evaluating our model shouldn't we need to find our model's mistakes??? Otherwise how we can correct them?

# # What we expecting from this notebook:
# * Grad-CAM visualization
# * Saliency Map
# * Some useful tool that might come handy

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')


# In[ ]:


train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'


# In[ ]:


train_df['image_path'] = train_dir + train_df.image_name +'.jpg'
test_df['image_path'] = test_dir + test_df.image_name +'.jpg'


# In[ ]:


train_df.head()


# In[ ]:


train_df['y'] = train_df.target.astype(str)


# In[ ]:


train_df.y.value_counts()


# In[ ]:


train_df_0 = train_df[train_df.target ==0].iloc[:584,:]
train_df_1 = train_df[train_df.target==1]
train_df_short = pd.concat((train_df_0 , train_df_1),axis = 0)


# In[ ]:


train_df_short.shape


# In[ ]:


X, y = train_df_short[train_df_short.columns[:-1]], train_df_short.y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val , y_train, y_val = train_test_split(X, y, random_state = 10, test_size = .2, stratify = y)


# In[ ]:


image_size = (256, 256)


# # Converting images to array for better speed

# ## Train

# In[ ]:


import cv2
import gc

train_images = np.zeros(shape = (X_train.shape[0], *image_size, 3))
train_labels = np.zeros(X_train.shape[0])

for idx in tqdm(range(X_train.shape[0])):
    img = cv2.imread(X_train.image_path.iloc[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize = image_size)
    
    
    train_images[idx,:,:] = img
    train_labels[idx] = int(y_train.iloc[idx])
    
    del img
    gc.collect()
    
# train_images = np.array(train_images)
# train_labels = np.array(train_labels)
train_images.shape, train_labels.shape


# ## Validation

# In[ ]:


import cv2
val_images = np.zeros(shape = (X_val.shape[0], *image_size, 3))
val_labels = np.zeros(X_val.shape[0])

for idx in tqdm(range(X_val.shape[0])):
    
    img = cv2.imread(X_val.image_path.iloc[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize = image_size)
    
    
    val_images[idx,:,:] = img
    val_labels[idx] = int(y_val.iloc[idx])
    
    del img
    gc.collect()

val_images.shape, val_labels.shape


# In[ ]:


plt.imshow(val_images[2].astype('uint8'))


# In[ ]:


train_df_short = pd.concat((X_train, y_train), axis = 1)
val_df = pd.concat((X_val, y_val), axis = 1)


# # ImageDataGenerator

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16


# In[ ]:


train_gen = ImageDataGenerator(
                                rotation_range=30,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
#                                 brightness_range=[0.7, 1.3],
                                shear_range=0.0,
                                zoom_range=0.2,
                                fill_mode="nearest",
                                horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1/255.0)

test_gen = ImageDataGenerator(rescale=1/255.0)


# In[ ]:


train_generator = train_gen.flow(
    x = train_images,
    y= train_labels,
    batch_size=32,
    shuffle=True,
    seed=1001
)

val_generator = test_gen.flow(
    x = val_images,
    y= val_labels,
    batch_size=32,
    shuffle=False,
    seed=1001
)

# Loading all test 10K images as array will consume all the memory so it's wise to use flow_from_dataframe()
test_generator = test_gen.flow_from_dataframe(
                                                test_df,
                                                x_col="image_path",
                                                target_size=image_size,
                                                class_mode=None,
                                                batch_size=32,
                                                shuffle=False,
                                                seed=10
                                            )


# 

# # Why just AUC when can get so much more?
# Here we'll be using a callback which will give almost every classification report possible with AUC and Confusion Matrix Plot

# ## Plot Confusion Matrix

# In[ ]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas.util.testing as tm
from sklearn import metrics
import seaborn as sns
sns.set()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(b=False)
    


# ## Function to get all the score

# In[ ]:


def evaluate_model(model, test_generator, y_test, class_labels, cm_normalize=True,                  print_cm=True):
    
    from datetime import datetime

    results = dict()

    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred_original = model.predict(test_generator,verbose=1)
    # y_pred = (y_pred_original>0.5).astype('int')

    y_pred = (y_pred_original>0.5).astype('int')
    #y_test = np.argmax(testy, axis=-1)
    
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
    
    
    # balanced_accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    print('---------------------')
    print('| Balanced Accuracy  |')
    print('---------------------')
    print('\n    {}\n\n'.format(balanced_accuracy))

    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    

    # calculate Cohen Kappa Score of the model
    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    print('---------------------')
    print('|      Cohen Kappa Score      |')
    print('---------------------')
    print('\n    {}\n\n'.format(kappa))

    
    print('---------------------')
    print('|      ROC Plot      |')
    print('---------------------')

    fpr, tpr, thr = metrics.roc_curve(y_test, y_pred_original)
    auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(4,4))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    
    # calculate auc score of the model
    auc = metrics.auc(fpr, tpr)
    # store accuracy in results
    print('---------------------')
    print('|      ROC AUC Score      |')
    print('---------------------')
    print('\n    {}\n\n'.format(auc))
    

    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(6,4))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return


# In[ ]:


from keras.callbacks import Callback
class Scores(Callback):
  
  def __init__(self, test_generator, y_test, class_labels):
    super(Scores, self).__init__()
    self.test_generator = test_generator
    self.y_test = y_test
    self.class_labels = class_labels
        
  def on_epoch_end(self, epoch, logs=None):

    evaluate_model(self.model, self.test_generator, self.y_test, self.class_labels)


# # EpochPlot Callback
# `This callback is for those who are really impatient like me to wait for the whole time until the model training is done...`
# 
# This callback will create EpochPlot after certain epoch. In that way we'll have better understanding if our model is overfitting or not..

# In[ ]:


import seaborn as sns
sns.set()

from keras.callbacks import Callback

class EpochPlot(Callback):
    
    def __init__(self, freq): # Frequency of EpochPlot
        super(EpochPlot, self).__init__()
        self.freq = freq

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        if (self.i % self.freq)==0:

          f, (ax1, ax2) = plt.subplots(1, 2, figsize =  (24,6), sharex=True, constrained_layout = True)
          
          # ax1.set_yscale('log')
          ax2.plot(self.x, self.losses, label="loss", marker = 'o')
          ax2.plot(self.x, self.val_losses, label="val_loss", marker = 'o')
          ax2.legend()
          
          ax1.plot(self.x, self.acc, label="acc", marker = 'o')
          ax1.plot(self.x, self.val_acc, label="val_acc", marker = 'o')
          ax1.legend()
          
          plt.show();


# # As our main focus is on what our model is learning, I won't be bulding any fancy model

# In[ ]:


from keras.layers import *
from keras.models import *
from keras.utils import *


# In[ ]:


from keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape = (*image_size, 3), weights = 'imagenet', include_top = False)
base_model.summary()
# for layer in base_model.layers[:-4]:
#     layer.trainable = False


# In[ ]:


from keras.optimizers import Adam
inputs = base_model.input
outputs = base_model.output

outputs = GlobalAveragePooling2D()(outputs)
outputs = Dense(256, activation = 'relu')(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(128, activation = 'relu')(outputs)
outputs = Dropout(0.3)(outputs)
outputs = Dense(64, activation = 'relu')(outputs)
outputs = Dropout(0.1)(outputs)
outputs = Dense(1, activation = 'sigmoid')(outputs)

model = Model(inputs = inputs, outputs = outputs)
model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['acc'])
model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint

filepath = 'best_model.h5'
callback1 = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True,
    save_weights_only=False, mode='max'
)

callback2 = Scores(val_generator, y_val.values.astype(int), ['benign', 'malignant'])
callback3 = EpochPlot(5)
callbacks = [callback1, callback2, callback3]


# In[ ]:


model.fit(train_generator, 
       validation_data = val_generator,
       steps_per_epoch = train_generator.n//train_generator.batch_size,
       epochs = 50,
#        class_weight = class_weight,
       callbacks = callbacks,
       verbose = 1)


# # Loading our best Model

# In[ ]:


model = load_model('/kaggle/working/best_model.h5')


# # Before visualization let's get our model prediction so that we can plot class wise TP, FP , TN or FN images

# In[ ]:


def TP(row):
    if (row.y_pred ==1)& (row.y_true==1):
        return True
    else:
        return False
def TN(row):
    if (row.y_pred ==0)& (row.y_true==0):
        return True
    else:
        return False
def FP(row):
    if (row.y_pred ==1)& (row.y_true==0):
        return True
    else:
        return False
def FN(row):
    if (row.y_pred ==0)& (row.y_true==1):
        return True
    else:
        return False
    

preds = model.predict(val_generator, verbose = 1)    
y_df = pd.DataFrame({'image_path':X_val.image_path.values,
                     'y_true':y_val.values.astype(int), 
                     'y_pred': preds.squeeze().round().astype(int)})

y_df['TP'] = y_df.apply(TP, axis = 1).values
y_df['TN'] = y_df.apply(TN, axis = 1).values
y_df['FP'] = y_df.apply(FP, axis = 1).values
y_df['FN'] = y_df.apply(FN, axis = 1).values


# In[ ]:


y_df.head()


# # Let's just clear some memory before Plot

# In[ ]:


del train_images
del val_images
del train_labels
del val_labels

import gc
gc.collect()


# # Grad-CAM
# ![Grad-CAM](https://media.arxiv-vanity.com/render-output/2950516/figures/gcam_ablation_gap_gmp.png)
# ![Grad-CAM](http://gradcam.cloudcv.org/static/images/network.png)
# Check this paper [here](https://arxiv.org/abs/1610.02391)

# # Saliency Map
# ![](https://miro.medium.com/max/1400/1*vmOVSK7KplO77BSWhzzJkA.gif)
# **There is a already a package for this kind of visualization: [keras-vis](https://github.com/raghakot/keras-vis)**

# # This Function generates both Grad-CAM and Saliency Map

# In[ ]:


def visualize(img_path , model , last_conv_layer_name,  image_size = (256, 256), alpha = 0.4):

  import os
  import numpy as np
  import pandas as pd
  import cv2
  #   from google.colab.patches import cv2_imshow
  from PIL import Image
  from matplotlib import pyplot as plt

  from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
  from keras.preprocessing.image import load_img, img_to_array
  from keras.models import Model, load_model
  from keras import backend as K



  # ==================================
  #   1. Test images prediction
  # ==================================
  img = load_img(img_path, target_size= (int(model.input.shape[1]), int(model.input.shape[2])))
  # msk = load_img(mask_path, target_size=image_size, color_mode= 'grayscale')
  img = img_to_array(img)
  img = img/255.0
  pred_img = np.expand_dims(img, axis=0)
  # pred_img = preprocess_input(img)
  pred = model.predict(pred_img)

  # ==============================
  #   2. Heatmap visualization 
  # ==============================
  # Item of prediction vector
  pred_output = model.output[:, np.argmax(pred)]

  # Feature map of 'conv_7b_ac' layer, which is the last convolution layer
  last_conv_layer = model.get_layer(last_conv_layer_name)

  # Gradient of class for feature map output of 'conv_7b_ac'
  grads = K.gradients(pred_output, last_conv_layer.output)[0]

  # Feature map vector with gradient average value per channel
  pooled_grads = K.mean(grads, axis=(0, 1, 2))

  # Given a test image, get the feature map output of the previously defined 'pooled_grads' and 'conv_7b_ac'
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  # Put a test image and get two numpy arrays
  pooled_grads_value, conv_layer_output_value = iterate([pred_img])

  # print(pooled_grads.shape[0])
  # Multiply the importance of a channel for a class by the channels in a feature map array
  for i in range(int(pooled_grads.shape[0])):
      conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  # The averaged value along the channel axis in the created feature map is the heatmap of the class activation
  heatmap = np.mean(conv_layer_output_value, axis=-1)

  # Normalize the heatmap between 0 and 1 for visualization
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap_img = heatmap


  # =======================
  #   3. Apply Grad-CAM
  # =======================
  ori_img = load_img(img_path, target_size=image_size)

  heatmap = cv2.resize(heatmap, image_size)
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

  superimposed_img = heatmap * alpha + ori_img
  cv2.imwrite('grad_cam_result.png', superimposed_img) 
  grad_img = cv2.imread('grad_cam_result.png')
#   grad_img = cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB)

  # =======================
  #   4. Saliency Map
  # =======================

  import tensorflow as tf
  pred_output = model.output[:, np.argmax(pred)]
  sal = K.gradients(tf.reduce_sum(pred_output), model.input)[0]

  # Keras function returning the saliency map given an image input
  sal_fn = K.function([model.input], [sal])
    
  # Generating the saliency map and normalizing it
  img_sal = sal_fn([np.resize(pred_img, (1, int(model.input.shape[1]), int(model.input.shape[2]), 3))])[0]
  img_sal = np.abs(img_sal)
  img_sal /= img_sal.max()
  img_sal = cv2.resize(img_sal[0,:,:,0], dsize = image_size)


  return ori_img, pred, heatmap_img, grad_img, img_sal


# # Let's see how our model performed..

# ## True Positive(TP)

# In[ ]:


n_img = 10
last_conv_layer_name = 'block5_conv3'
df = y_df[(y_df.TP==True) & (y_df.y_true == 1)]


if df.shape[0]<n_img:
    print('Number of Image exceeds')

else:
    fig, ax = plt.subplots(n_img, 3, figsize = (12,n_img*4), constrained_layout = True)

    for idx in tqdm(range(n_img)):

        img, pred, map_img, grad_img, sal = visualize(df.image_path.iloc[idx],
                                                  model ,
                                                  last_conv_layer_name,
                                                  image_size = (256, 256),
                                                  alpha = 0.6)

        ax[idx][0].imshow(img)
        ax[idx][0].grid(b = False)
        ax[idx][0].set_xticks([])
        ax[idx][0].set_yticks([])
        ax[idx][0].set_title('Original Image:', fontsize = 15)

        ax[idx][1].imshow(grad_img)
        ax[idx][1].grid(b = False)
        ax[idx][1].set_xticks([])
        ax[idx][1].set_yticks([])
        ax[idx][1].set_title(f'Grad-CAM: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)

        ax[idx][2].imshow(sal)
        ax[idx][2].grid(b = False)
        ax[idx][2].set_xticks([])
        ax[idx][2].set_yticks([])
        ax[idx][2].set_title(f'Saliency-MAP: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)


# ## True Negative(TN)

# In[ ]:


n_img = 10
last_conv_layer_name = 'block5_conv3'
df = y_df[y_df.TN==True]

if df.shape[0]<n_img:
    print('Number of Image exceeds !!!')

else:
    fig, ax = plt.subplots(n_img, 3, figsize = (12,n_img*4), constrained_layout = True)

    for idx in tqdm(range(n_img)):

        img, pred, map_img, grad_img, sal = visualize(df.image_path.iloc[idx],
                                                  model ,
                                                  last_conv_layer_name,
                                                  image_size = (256, 256),
                                                  alpha = 0.6)

        ax[idx][0].imshow(img)
        ax[idx][0].grid(b = False)
        ax[idx][0].set_xticks([])
        ax[idx][0].set_yticks([])
        ax[idx][0].set_title('Original Image:', fontsize = 15)

        ax[idx][1].imshow(grad_img)
        ax[idx][1].grid(b = False)
        ax[idx][1].set_xticks([])
        ax[idx][1].set_yticks([])
        ax[idx][1].set_title(f'Grad-CAM: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)

        ax[idx][2].imshow(sal)
        ax[idx][2].grid(b = False)
        ax[idx][2].set_xticks([])
        ax[idx][2].set_yticks([])
        ax[idx][2].set_title(f'Saliency-MAP: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)


# ## False Positive(FP)

# In[ ]:


n_img = 10
last_conv_layer_name = 'block5_conv3'
df = y_df[y_df.FP==True]

if df.shape[0]<n_img:
    print('Number of Image exceeds !!!')

else:
    fig, ax = plt.subplots(n_img, 3, figsize = (12,n_img*4), constrained_layout = True)

    for idx in tqdm(range(n_img)):

        img, pred, map_img, grad_img, sal = visualize(df.image_path.iloc[idx],
                                                  model ,
                                                  last_conv_layer_name,
                                                  image_size = (256, 256),
                                                  alpha = 0.6)

        ax[idx][0].imshow(img)
        ax[idx][0].grid(b = False)
        ax[idx][0].set_xticks([])
        ax[idx][0].set_yticks([])
        ax[idx][0].set_title('Original Image:', fontsize = 15)

        ax[idx][1].imshow(grad_img)
        ax[idx][1].grid(b = False)
        ax[idx][1].set_xticks([])
        ax[idx][1].set_yticks([])
        ax[idx][1].set_title(f'Grad-CAM: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)

        ax[idx][2].imshow(sal)
        ax[idx][2].grid(b = False)
        ax[idx][2].set_xticks([])
        ax[idx][2].set_yticks([])
        ax[idx][2].set_title(f'Saliency-MAP: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)


# ## False Negative(FN)

# In[ ]:


n_img = 10
last_conv_layer_name = 'block5_conv3'
df = y_df[y_df.FN==True]

if df.shape[0]<n_img:
    print('Number of Image exceeds !!!')

else:
    fig, ax = plt.subplots(n_img, 3, figsize = (12,n_img*4), constrained_layout = True)

    for idx in tqdm(range(n_img)):

        img, pred, map_img, grad_img, sal = visualize(df.image_path.iloc[idx],
                                                  model ,
                                                  last_conv_layer_name,
                                                  image_size = (256, 256),
                                                  alpha = 0.6)

        ax[idx][0].imshow(img)
        ax[idx][0].grid(b = False)
        ax[idx][0].set_xticks([])
        ax[idx][0].set_yticks([])
        ax[idx][0].set_title('Original Image:', fontsize = 15)

        ax[idx][1].imshow(grad_img)
        ax[idx][1].grid(b = False)
        ax[idx][1].set_xticks([])
        ax[idx][1].set_yticks([])
        ax[idx][1].set_title(f'Grad-CAM: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)

        ax[idx][2].imshow(sal)
        ax[idx][2].grid(b = False)
        ax[idx][2].set_xticks([])
        ax[idx][2].set_yticks([])
        ax[idx][2].set_title(f'Saliency-MAP: {df.y_true.iloc[idx]} | {pred[0][0]:.2f}', fontsize = 15)
        
    plt.show()


# # Thanks for reading this kernel
# Please let me know if anything else could have been added...
