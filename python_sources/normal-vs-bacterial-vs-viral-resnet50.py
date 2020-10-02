#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
from tqdm import tqdm
from glob import glob
from keras.layers import *
from keras.models import *
from keras.utils import *
import numpy as np
import os


# # **Loading Training Files**

# I have converted all images to numpy array to boost speed

# In[ ]:


normal = np.load('../input/pneumonia-chest-xray-npy/Array128/Array128/train_Normal_128.npy')
viral = np.load('../input/pneumonia-chest-xray-npy/Array128/Array128/train_Virus_128.npy')
bacterial = np.load('../input/pneumonia-chest-xray-npy/Array128/Array128/train_bacteria_128.npy')


# In[ ]:


normal.shape, viral.shape, bacterial.shape


# In[ ]:


label_normal = np.zeros(len(normal))
label_bacterial = np.ones(len(bacterial))
label_viral = np.full(len(viral),2, dtype = int)


# In[ ]:


train_data = np.concatenate((normal,bacterial,viral),axis=0)
train_label = np.concatenate((label_normal,label_bacterial,label_viral),axis=0)


# In[ ]:


train_label.shape, train_data.shape


# ## Visualization

# In[ ]:


import matplotlib.pyplot as plt


# ### Normal

# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(normal[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# # Viral

# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(viral[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# ## Bacterial

# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(bacterial[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# # Loading Test Data

# In[ ]:


test_normal = np.load('../input/pneumonia-chest-xray-npy/Array128/Array128/test_Normal_128.npy')
test_viral = np.load('../input/pneumonia-chest-xray-npy/Array128/Array128/test_Virus_128.npy')
test_bacterial = np.load('../input/pneumonia-chest-xray-npy/Array128/Array128/test_bacteria_128.npy')


# In[ ]:


test_normal.shape, test_viral.shape , test_bacterial.shape


# In[ ]:


label_test_normal = np.zeros(len(test_normal))
label_test_bacterial = np.ones(len(test_bacterial))
label_test_viral = np.full(len(test_viral),2, dtype = int)


# In[ ]:


test_data = np.concatenate((test_normal, test_bacterial, test_viral),axis=0)
test_label = np.concatenate((label_test_normal,label_test_bacterial,label_test_viral),axis=0)


# In[ ]:


test_data.shape


# ## Visualization

# ## Normal

# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(test_normal[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# ## Viral

# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(test_viral[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# ## Bacterial

# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(test_bacterial[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# # Dealing with Class Imbalance

# In[ ]:


from sklearn.utils import class_weight
 
 
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_label),
                                                 train_label)
class_weights


# # Custom Callback

# In[ ]:


# plot confusion matrix

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas.util.testing as tm
from sklearn import metrics
import seaborn as sns
sns.set()

plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save = False):
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
    if save == True:
      plt.savefig('Confusion Matrix.png', dpi = 900)
    
def plot_roc_curve(y_true, y_pred, classes):

    from sklearn.metrics import roc_curve, auc

    # create plot
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    for (i, label) in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_true[:,i].astype(int), y_pred[:,i])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

    # Set labels for plot
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    c_ax.set_title('Roc AUC Curve')


# In[ ]:


pred = model.predict_generator(val_generator, steps= test_label.shape[0]/batch_size)


# In[ ]:


pred.shape


# In[ ]:


test_label.shape[0]/32


# In[ ]:


# test model performance
from datetime import datetime
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def test_model(model, test_generator, y_test, class_labels, cm_normalize=True,                  print_cm=True):
    
    # BS = 16
    results = dict()
    
    # n = len(testy)// BS

    # testX = testX[:BS*n]
    # testy = testy[:BS*n]
    steps = y_test.shape[0]/batch_size
#     y_test = y_test[:steps*batch_size]

    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred_original = model.predict_generator(test_generator, verbose=1, steps = steps)
    # y_pred = (y_pred_original>0.5).astype('int')

    y_pred = np.argmax(y_pred_original, axis = 1)
    # y_test = np.argmax(testy, axis= 1)
    #y_test = np.argmax(testy, axis=-1)
    
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
    y_test = y_test.astype(int) # sparse form not categorical
    

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
    

    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)


    
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
    
    #roc plot
#     plot_roc_curve(to_categorical(y_test), y_pred_original, class_labels)
    
    from sklearn.metrics import roc_curve, auc

    # create plot
    fig, c_ax = plt.subplots(1,1, figsize = (7, 7))
    for (i, label) in enumerate(class_labels):
        fpr, tpr, thresholds = roc_curve(to_categorical(y_test)[:,i].astype(int), y_pred_original[:,i])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))

    # Set labels for plot
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    c_ax.set_title('Roc AUC Curve')
    plt.show()


    
    # add the trained  model to the results
    results['model'] = model
    
    return


from keras.callbacks import Callback
class MyLogger(Callback):
  
  def __init__(self, test_generator, y_test, class_labels):
    super(MyLogger, self).__init__()
    self.test_generator = test_generator
    self.y_test = y_test
    self.class_labels = class_labels
    
  def on_epoch_end(self, epoch, logs=None):
    test_model(self.model, self.test_generator, self.y_test, self.class_labels)


# # One Hot Encoding the labels

# In[ ]:


from keras.utils import to_categorical
train_label = to_categorical(train_label, num_classes= 3)
test_label  = to_categorical(test_label, num_classes = 3)


# # ImageDataGenerator

# In[ ]:


batch_size = 32


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import *


train_datagen = ImageDataGenerator(rescale = 1/255,
                                  width_shift_range = 0.1,
                                  height_shift_range = 0.1,
                                  fill_mode = 'constant',
                                  zoom_range = 0.2,
                                  rotation_range = 30)

val_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow(train_data,
                                     train_label, 
                                     batch_size = batch_size, 
                                     shuffle = True)

val_generator = val_datagen.flow(test_data,
                                 test_label,
                                 batch_size = batch_size,
                                 shuffle = False)


# # Vizualization After Augmentation

# In[ ]:


images, labels = train_generator.next()


# In[ ]:


images.shape


# In[ ]:


n_row = 3
n_col = 5

fig, ax = plt.subplots(n_row, n_col, figsize = (n_col*3, n_row*3), constrained_layout = True)

for row in tqdm(range(n_row)):
    
    for col in range(n_col):
        
        ax[row][col].imshow(images[row*n_col + col,:,:,0], cmap = 'bone')
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])


# # Fixing the channels

# In[ ]:


def fix_generator(data_generator):
    
    for img, label in data_generator:
        
        img = np.stack((img.squeeze(),)*3, axis=-1)
        
        yield img, label

train_generator = fix_generator(train_generator)
val_generator = fix_generator(val_generator)


# # Callback

# In[ ]:


os.mkdir('Model')
os.mkdir('History')


# In[ ]:



def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.000001, lr_rampup_epochs=10, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * 4

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


# In[ ]:


from keras.callbacks import *

def get_callbacks():
    
    filepath = 'Model/best_model_multiclass_128.h5'
    callback1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callback2 = MyLogger(val_generator, 
                         y_test = np.argmax(test_label, axis = 1),
                         class_labels = ['Normal', 'Viral', 'Bacterial'])
    
    callback3 = CSVLogger('History/Multiclass_Log_128.csv')
    lr_schedule = LearningRateScheduler(build_lrfn(), verbose=1)

    return [callback1 ,callback2, callback3, lr_schedule]


# # Building Model

# In[ ]:


from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers  import *

base_model = ResNet50(input_shape = (128, 128, 3), weights = 'imagenet', include_top = False)

inputs = base_model.input
outputs = base_model.output
outputs = GlobalAveragePooling2D()(outputs)
outputs = Dense(64, activation = 'relu')(outputs)
outputs = Dense(3, activation = 'softmax')(outputs)

model = Model(inputs, outputs)
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-6), metrics = ['acc'])
model.summary()


# In[ ]:


test_model(model, val_generator, y_test = np.argmax(test_label, axis = 1),
                         class_labels = ['Normal', 'Viral', 'Bacterial'])


# # Training

# In[ ]:


history = model.fit_generator(train_generator,
                              steps_per_epoch = len(train_data)/ batch_size,
                              validation_data=val_generator,
                              validation_steps= len(test_data)/batch_size,
                              class_weight =class_weights,
                              epochs = 300,
                              callbacks = get_callbacks(),
                              verbose = 1
                              )


# # Loading Best Model

# In[ ]:


from keras.models import load_model
best_model = load_model('/kaggle/working/Model/best_model_multiclass_128.h5')


# # Best Model Performance

# In[ ]:


test_model(best_model, 
           val_generator,
           y_test = np.argmax(test_label, axis = 1),
           class_labels = ['Normal', 'Viral', 'Bacterial'])


# # Plotting EpochPlot

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
acc = model.history.history['acc']
val_acc = model.history.history['val_acc']
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

epochs = range(0,len(acc))
fig = plt.gcf()
fig.set_size_inches(16, 8)

plt.plot(epochs, acc, 'r', label='Training accuracy',marker = "o")
plt.plot(epochs, val_acc, 'b', label='Validation accuracy',marker = "o")
plt.title('Training and validation accuracy')
plt.xticks(np.arange(0, len(acc), 10))
plt.legend(loc=0)
plt.figure()

fig = plt.gcf()
fig.set_size_inches(16, 8)
plt.plot(epochs, loss, 'r', label='Training Loss',marker = "o")
plt.plot(epochs, val_loss, 'b', label='Validation Loss',marker = "o")
plt.title('Training and validation Loss')
plt.xticks(np.arange(0, len(acc), 10))
plt.legend(loc=0)
#plt.savefig('Multiclass Model .png')
plt.figure()
plt.show()


# # Grad-CAM and Saliency Map 
# coming soon....
