#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import cv2
import datetime as dt
import glob
import itertools
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from keras import models, layers, optimizers, Model
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from keras import backend as K
import tensorflow as tf


# In[ ]:


import tensorflow as tf
print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[ ]:


print(os.listdir("../input"))


# print(os.listdir("../input/monkey-classifier-cnn-xception-v3"))

# In[ ]:


train_dir = Path('../input/databreastcancer/breastcancer_dataset/Training/')
test_dir = Path('../input/databreastcancer/breastcancer_dataset/Testing')


# In[ ]:


print(train_dir)


# In[ ]:


#label info
cols = ['Label', 'Name','Train Images', 'Validation Images']
csv = pd.read_csv("../input/databreastcancer/breastcancer_dataset/Breastcancer_labels.txt", names=cols, skiprows=1)
csv


# In[ ]:


common_names = csv['Name']
common_names


# In[ ]:


labels = csv['Label'].str.strip()
labels


# In[ ]:


img = cv2.imread('../input/databreastcancer/breastcancer_dataset/Training/1/8863_idx5_x1151_y1151_class1.png')
print(img.shape)
plt.imshow(img);


# In[ ]:


train_dir


# In[ ]:


# from imblearn.datasets import make_imbalance
# class_dict = dict()


# In[ ]:


# class_dict[0] = 1; class_dict[1] = 1; 
# print(class_dict)


# In[ ]:


#restnet50 input shape=224
height=50
width=50
channels=3
batch_size=32
seed=1337

# Training generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(height,width),
                                                    batch_size=batch_size,
                                                    seed=seed,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
#example from boneage
#val_datagen = ImageDataGenerator(width_shift_range=0.25,
#                                 height_shift_range=0.25,
#                                 horizontal_flip = True)
test_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(height,width), 
                                                  batch_size=batch_size,
                                                  seed=seed,
                                                  class_mode='categorical')


# In[ ]:


# X, y = make_imbalance(train_generator,test_generator, sampling_strategy={0: 10, 1: 10},)


# In[ ]:


train_generator.class_indices


# In[ ]:


train_generator.classes


# In[ ]:


type(train_generator.classes)


# In[ ]:


height


# In[ ]:


from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16 #(None, 224, 224, 3) 
from keras.applications.vgg19 import VGG19 #(None, 224, 224, 3) 
from keras.applications.xception import Xception #299x299.

weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #224x224.
height, width, channels =  50, 50, 3
#weights='../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
#weights='../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = ResNet50(weights=weights,
                      include_top=False,
                      input_shape=(height, width, channels))
base_model.summary()


# base_model.layers[:-1] #get all except the last one

# In[ ]:


base_model.layers[-1].output.shape


# In[ ]:


base_model.layers[-1].name


# In[ ]:


D1 = base_model.layers[-1].output.shape[1]
D2 = base_model.layers[-1].output.shape[2]
D3 = base_model.layers[-1].output.shape[3]


# In[ ]:


D1, D2, D3


# In[ ]:


def extract_features(sample_count, datagen):
    start = dt.datetime.now()
    features =  np.zeros(shape=(sample_count, D1, D2, D3))
    labels = np.zeros(shape=(sample_count,2))
    generator = datagen
    i = 0
    for inputs_batch,labels_batch in generator:
        stop = dt.datetime.now()
        time = (stop - start).seconds
        print('\r',
              'Extracting features from batch', str(i+1), '/', len(datagen),
              '-- run time:', time,'seconds',
              end='')
        
        features_batch = base_model.predict(inputs_batch)
        print('\nfeatures_batch.shape:', features_batch.shape)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        
        if i * batch_size >= sample_count:
            break
            
    print("\n")
    return features,labels


# In[ ]:


t1 = time.time()
with tf.device('/device:GPU:0'):
    test_features, test_labels = extract_features(6000, test_generator)
t2 = (time.time()-t1)/60
print('Extracting time: %s min' % ( t2 ))


# In[ ]:


test_features.shape


# In[ ]:


10*10*2048*272


# In[ ]:


flat = test_features.flatten()
flat.shape


# In[ ]:


test_labels.shape


# In[ ]:


train_generator.n


# In[ ]:


train_generator.n//batch_size


# In[ ]:


len(train_generator)


# In[ ]:


t3 = time.time()
with tf.device('/device:GPU:0'):
    train_features, train_labels = extract_features(20000, train_generator)
t4 = (time.time()-t3)/60
print('Extracting time: %s min' % ( t4 ))
print('Total extracting time: %s min' % (t2+t4))


# In[ ]:


print('before, train_features.shape:', train_features.shape)
print('before, test_features.shape:', test_features.shape)


# In[ ]:


flat_dim = int(D1) * int(D2) * int(D3)
train_features = np.reshape(train_features, (train_features.shape[0], flat_dim))
test_features = np.reshape(test_features, (test_features.shape[0], flat_dim))


# In[ ]:


print('train_features.shape:', train_features.shape)
print('test_features.shape:', test_features.shape)


# In[ ]:


flat_dim


# In[ ]:


from keras.layers.pooling import MaxPooling1D

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_dim=flat_dim))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()


# # Reset model weights
# K.get_session().close()
# K.set_session(tf.Session())
# K.get_session().run(tf.global_variables_initializer())

# # initial weights
# model.save_weights('model.h5')

# In[ ]:


model.layers[-1]


# In[ ]:


# reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
#                                          factor=0.1,
#                                          patience=2,
#                                          cooldown=2,
#                                          min_lr=0.00001,
#                                          verbose=1)

# callbacks_list = [reduce_learning_rate]


# In[ ]:


# EPOCHS = train_features.shape[0]//batch_size + 1
# EPOCHS


# In[ ]:


t = time.time()
with tf.device('/device:GPU:0'):
    history = model.fit(train_features, 
                    train_labels, 
                    #epochs=EPOCHS,
                    epochs=20,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.3,
                    #callbacks=callbacks_list
                       )

print('Training time: %s min' % ( (time.time()-t)/60 ))


# ### https://stackoverflow.com/questions/40496069/reset-weights-in-keras-layer
# 
# 
# Save the initial weights right after compiling the model but before training it:
# 
# model.save_weights('model.h5')
# and then after training, "reset" the model by reloading the initial weights:
# 
# model.load_weights('model.h5')
# This gives you an apples to apples model to compare different data sets and should be quicker than recompiling the entire model.

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()


# In[ ]:





# json_file = open('../input/monkey-classifier-cnn-xception-v3/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(weight_path)
# print("Loaded {} model from disk".format(weight_path))

# In[ ]:


train_labels.shape


# In[ ]:


#preds = np.zeros(shape=(272,10))
t = time.time()
with tf.device('/device:GPU:0'):
    preds = model.predict(test_features)
print('Prediction time: %s min' % ( (time.time()-t)/60 ))


# In[ ]:


preds.shape


# In[ ]:


preds[0]


# In[ ]:


# Change labels from one-hot encoded
predictions = list()
y_true = list()
predictions = [i.argmax() for i in preds]
y_true = [i.argmax() for i in test_labels]


# In[ ]:


predictions[0:10]


# In[ ]:


y_true[0:10]


# In[ ]:


def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=14)
        plt.yticks(tick_marks, target_names, fontsize=14)

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm,2)
        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass),
              fontsize=14)
    plt.show()


# In[ ]:


set(y_true) - set(predictions)


# In[ ]:


from sklearn import metrics
metrics.f1_score(y_true, predictions, average='weighted', labels=np.unique(predictions))


# In[ ]:


metrics.f1_score(y_true, predictions, average='macro', labels=np.unique(predictions))


# In[ ]:


metrics.f1_score(y_true, predictions, average='micro', labels=np.unique(predictions))


# In[ ]:


metrics.accuracy_score(y_true, predictions)


# In[ ]:


metrics.cohen_kappa_score(y_true, predictions)


# In[ ]:


type(predictions), type(y_true)


# In[ ]:


cm = confusion_matrix(y_pred=predictions, y_true=y_true)
type(cm), id(cm)


# In[ ]:


plot_confusion_matrix(cm, normalize=True, target_names=labels)


# In[ ]:


plot_confusion_matrix(cm,target_names=labels)


# In[ ]:





# import seaborn as sns
# plt.figure(figsize=(10, 8), dpi=300)
# ax = sns.heatmap(cm, annot=True,
#                  cbar=True,
#                  fmt='d', 
#                  linewidths=.05
#                 )

# In[ ]:





# import seaborn as sns
# plt.figure(figsize=(10, 8), dpi=300)
# ax = sns.heatmap(cm, annot=True,
#                  cbar=True,
#                  fmt='d', 
#                  linewidths=.05,
#                  xticklabels=labels,
#                  yticklabels=labels
#                 )

# In[ ]:


import seaborn as sns
target_names = labels
plt.figure(figsize=(10, 8), dpi=300)
accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, fontsize=14)
plt.yticks(tick_marks, target_names, fontsize=14)

#sns.set(font_scale=1.4)
ax = sns.heatmap(cm, annot=True,
                 cbar=True,
                 fmt='d', 
                 linewidths=.05,
                 xticklabels=labels,
                 yticklabels=labels
                )

plt.tight_layout()
plt.ylabel('True label', fontsize=16)
plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass),
          fontsize=16)


# In[ ]:





# In[ ]:


import seaborn as sns
target_names = labels
plt.figure(figsize=(10, 8), dpi=300)
accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, fontsize=14)
plt.yticks(tick_marks, target_names, fontsize=14)

#sns.set(font_scale=1.4)
cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)
ax = sns.heatmap(cm, annot=True,
                 cbar=True,
                 fmt='d', 
                 linewidths=.05,
                 xticklabels=labels,
                 yticklabels=labels,
                 cmap=cmap
                )

plt.tight_layout()
plt.ylabel('True label', fontsize=16)
plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass),
          fontsize=16)


# In[ ]:





# In[ ]:


import seaborn as sns
target_names = labels
plt.figure(figsize=(10, 8), dpi=300)
accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, fontsize=14)
plt.yticks(tick_marks, target_names, fontsize=14)

#sns.set(font_scale=1.4)
cmap = sns.cubehelix_palette(dark=1, light=0, as_cmap=True)
ax = sns.heatmap(cm, annot=True,
                 cbar=True,
                 fmt='d', 
                 linewidths=.05,
                 xticklabels=labels,
                 yticklabels=labels,
                 cmap=cmap
                )

plt.tight_layout()
plt.ylabel('True label', fontsize=16)
plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass),
          fontsize=16)


# In[ ]:


from sklearn.metrics import explained_variance_score
explained_variance_score(y_true, predictions) 


# In[ ]:





# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_pred=predictions, y_true=y_true,
                            target_names=common_names))


# In[ ]:





# def test_f(a, b):
#     import numpy as np
#     from scipy import interp
#     import matplotlib.pyplot as plt
#     from itertools import cycle
#     from sklearn.metrics import roc_curve, auc
#     from matplotlib.pyplot import figure
#     print(a, b)
#     
# test_f(1,2)    

# In[ ]:





# In[ ]:


def plot_roc_auc(y_score, y_test, num_classes):
    #y_score = model.predict(test_features)
    #y_test = test_labels in one hot array
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from matplotlib.pyplot import figure

    # Plot linewidth.
    lw = 2
    n_classes = num_classes
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    #plt.figure
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                   'red', 'darkgoldenrod', 'darkgreen',
                   'darkmagenta', 'darkorange', 'firebrick', 'rosybrown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate', fontsize=14) # 1 - Specificity 
    #plt.ylabel('True Positive Rate', fontsize=14) # Sensitivity
    plt.xlabel('1 - Specificity ', fontsize=14) # 1 - Specificity 
    plt.ylabel('Sensitivity', fontsize=14) # Sensitivity
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Receiver operating characteristic to multi-class', fontsize=14)
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


#preds = model.predict(test_features)    

num_classes = 10
#y_score = model.predict(test_features)
y_score = preds
y_test = test_labels

plot_roc_auc(y_score, y_test, num_classes)


# In[ ]:





# ### Plot precision_recall_curve

# In[ ]:





# In[ ]:


test_labels


# In[ ]:





# In[ ]:


# For each class
precision = dict()
recall = dict()
average_precision = dict()
n_classes = 10


# In[ ]:


def plot_precision_recall_curve(y_score, Y_test, n_classes):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    
    #Y_test = test_labels #in one hot array
    #n_classes = 10
    #y_score from model.predict

    # For each class
    #precision = dict()
    #recall = dict()
    #average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    #plt.figure()
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='g',
                     #**step_kwargs
                    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]), fontsize=16)


# In[ ]:





# In[ ]:


plot_precision_recall_curve(y_score=y_score, Y_test=y_test, n_classes=10)


# In[ ]:





# ### Plot Precision-Recall curve for each class and iso-f1 curves

# In[ ]:


def plot_precision_recall_each_class():
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    #plt.figure(figsize=(7, 8))
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    #plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title('Extension of Precision-Recall curve to multi-class', fontsize=16)
    plt.legend(lines, labels, 
               loc=(0, -1.1), 
               prop=dict(size=12), 
               #loc="lower right"
              )


    plt.show()


# In[ ]:


plot_precision_recall_each_class()


# ### Commit to store test_features and train_features

# In[ ]:


cm


# In[ ]:


FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[ ]:


print('ACC:', ACC, '{:2.4}'.format(ACC.mean()), '{:2.4}'.format(ACC.std()))


# In[ ]:


p_measures = dict()
p_measures = {
    'TP': TP,
    'TN': TN,
    'FP': FP,
    'FN': FN,
    'TPR': TPR,
    'TNR': TNR,
    'PPV': PPV,
    'NPV': NPV,
    'FPR': FPR,
    'FNR': FNR,
    'FDR': FDR,
    'ACC': ACC
}


# In[ ]:


where_are_NaNs = np.isnan(FDR)
FDR[where_are_NaNs] = 0


# In[ ]:


where_are_NaNs = np.isnan(PPV)
PPV[where_are_NaNs] = 0


# In[ ]:


for key, value in p_measures.items():
    print(key, value, '\n', '{:2.4}'.format(value.mean()), 
          '\t', '{:2.4}'.format(value.std()), '\n')


# print(
#     'TP:', TP, '\n',
#     'FN:', FN, '\n',
#     'TP:', TP, '\n',
#     'TN:', TN, '\n',
#      )

# # Sensitivity, hit rate, recall, or true positive rate
# print('TPR:', TPR)
# # Specificity or true negative rate
# print('TNR:', TNR)
# # Precision or positive predictive value
# print('PPV:', PPV)
# # Negative predictive value
# print('NPV:', NPV)
# # Fall out or false positive rate
# print('FPR:', FPR)
# # False negative rate
# print('FNR:', FNR)
# # False discovery rate
# print('FDR:', FDR)
# 
# # Overall accuracy
# ACC = (TP+TN)/(TP+FP+FN+TN)
# print('ACC:', ACC)

# tpr_mean = TPR.mean()
# tpr_std = TPR.std()
# print('TPR.mean: {:2.4f}'.format(tpr_mean))
# print('TPR.std: {:2.4f}'.format(tpr_std))

# In[ ]:


labels


# In[ ]:


cf = classification_report(y_pred=predictions,
                            y_true=y_true,
                            target_names=labels,
                            output_dict = True
                          )
type(cf)


# In[ ]:


cf.keys()


# In[ ]:


cf['weighted avg']


# In[ ]:


mean_f1_score = cf['weighted avg']['f1-score']
mean_f1_score


# In[ ]:


cf['n0']['f1-score']


# In[ ]:


len(labels)


# In[ ]:


type(labels)


# In[ ]:


labelList = list(labels)
type(labelList)


# In[ ]:


labelList


# for key, value in cf.items():
#     if key in labelList:
#         print(key, value)
#     

# In[ ]:


# https://www.mathsisfun.com/data/standard-deviation-formulas.html
# Standard Deviation of F1-score
mean_f1score = cf['weighted avg']['f1-score']
sum_msd = 0
for key, value in cf.items():
    if key in labelList:
        #print(type(value), value)
        mean_square_dif = (value['f1-score'] - mean_f1score)**2
        sum_msd += mean_square_dif
        print(key, 'f1-score: {:2.4f}'.format(value['f1-score']),
              'mean_square_dif: {:2.5f}'.format(mean_square_dif )
             )
        
std_f1score = (sum_msd*(1/len(labelList)))**(1/2)
print('mean_f1score: {:2.4f}'.format(mean_f1score), 
      'sd_f1score: {:2.4f}'.format(std_f1score))

