#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


get_ipython().system('pip install imutils')
get_ipython().system('pip install image-classifiers==1.0.0b1')


# In[ ]:


# import the necessary packages
import tensorflow as tf
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, DenseNet169
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Activation, BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN, LearningRateScheduler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
from classification_models.tfkeras import Classifiers
from datetime import datetime
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


dataset_path = './dataset'
log_path = './logs'


# ## Build Dataset

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'rm -rf dataset\nmkdir -p dataset/covid\nmkdir -p dataset/normal\nmkdir -p dataset/pneumonia\nmkdir -p logs')


# In[ ]:


len(os.listdir('../input/covid-chest-xray/images/'))


# ### Covid xray dataset

# In[ ]:


samples = 140


# In[ ]:


covid_dataset_path = '../input/covid-chest-xray'


# In[ ]:


# construct the path to the metadata CSV file and load it
csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
df = pd.read_csv(csvPath)

# loop over the rows of the COVID-19 data frame
for (i, row) in df.iterrows():
    # if (1) the current case is not COVID-19 or (2) this is not
    # a 'PA' view, then ignore the row
    if row["finding"] != "COVID-19" or row["view"] != "PA":
        continue

    # build the path to the input image file
    imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

    # if the input image file does not exist (there are some errors in
    # the COVID-19 metadeta file), ignore the row
    if not os.path.exists(imagePath):
        continue

    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = row["filename"].split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# ### Build normal xray dataset

# In[ ]:


pneumonia_dataset_path ='../input/chest-xray-pneumonia/chest_xray'


# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:samples]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# In[ ]:


samples = 130


# In[ ]:


basePath = os.path.sep.join([pneumonia_dataset_path, "train", "PNEUMONIA"])
imagePaths = list(paths.list_images(basePath))

# randomly sample the image paths
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:samples]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the filename from the image path and then construct the
    # path to the copied image file
    filename = imagePath.split(os.path.sep)[-1]
    outputPath = os.path.sep.join([f"{dataset_path}/pneumonia", filename])

    # copy the image
    shutil.copy2(imagePath, outputPath)


# ## Plot x-rays

# In[ ]:


len(os.listdir('../working/dataset/normal'))


# In[ ]:


len(os.listdir('../working/dataset/pneumonia/'))


# Helper function to plot the images in a grid

# In[ ]:


def ceildiv(a, b):
    return -(-a // b)

def plots_from_files(imspaths, figsize=(10,5), rows=1, titles=None, maintitle=None):
    """Plot the images in a grid"""
    f = plt.figure(figsize=figsize)
    if maintitle is not None: plt.suptitle(maintitle, fontsize=10)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img, cmap = 'gray')


# In[ ]:


normal_images = list(paths.list_images(f"{dataset_path}/normal"))
covid_images = list(paths.list_images(f"{dataset_path}/covid"))


# In[ ]:


plots_from_files(normal_images, rows=5, maintitle="Normal X-ray images")


# In[ ]:


plots_from_files(covid_images, rows=5, maintitle="Covid-19 X-ray images")


# ## Data preprocessing

# In[ ]:


class_to_label_map = {'pneumonia' : 2, 'covid' : 1, 'normal' : 0}


# In[ ]:


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(class_to_label_map[label])
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
data = np.array(data) / 255.0
labels = np.array(labels)


# In[ ]:


# perform one-hot encoding on the labels
# perform one-hot encoding on the labels
#labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15, stratify=labels, random_state=42)
# initialize the training data augmentation object
train_datagen = ImageDataGenerator(
                                   rotation_range=60,
                                   horizontal_flip = True ,
                                   vertical_flip = True ,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator()


# In[ ]:


trainYSparse = trainY
trainY = to_categorical(trainY)


# ## Model

# In[ ]:


def get_baseline_model(img_size):
  
    weight_decay = 1e-4

    visible = Input(shape=(img_size, img_size, 3), dtype=tf.float32)

    conv = Conv2D(32, (5, 5))(visible)
    conv_act = Activation('relu')(conv)
    conv_act_batch = BatchNormalization()(conv_act)
    conv_maxpool = MaxPooling2D()(conv_act_batch)
    conv_dropout = Dropout(0.1)(conv_maxpool)

    conv = Conv2D(64, (3, 3))(conv_dropout)
    conv_act = Activation('relu')(conv)
    conv_act_batch = BatchNormalization()(conv_act)
    conv_maxpool = MaxPooling2D()(conv_act_batch)
    conv_dropout = Dropout(0.2)(conv_maxpool)

    conv = Conv2D(128, (3, 3))(conv_dropout)
    conv_act = Activation('relu')(conv)
    conv_act_batch = BatchNormalization()(conv_act)
    conv_maxpool = MaxPooling2D()(conv_act_batch)
    conv_dropout = Dropout(0.3)(conv_maxpool)

    conv = Conv2D(256, (3, 3))(conv_dropout)
    conv_act = Activation('relu')(conv)
    conv_act_batch = BatchNormalization()(conv_act)
    conv_maxpool = MaxPooling2D()(conv_act_batch)
    conv_dropout = Dropout(0.4)(conv_maxpool)

    conv = Conv2D(512, (3, 3))(conv_dropout)
    conv_act = Activation('relu')(conv)
    conv_act_batch = BatchNormalization()(conv_act)
    conv_maxpool = MaxPooling2D()(conv_act_batch)
    conv_dropout = Dropout(0.5)(conv_maxpool)
    
    gap2d = GlobalAveragePooling2D()(conv_dropout)
    act = Activation('relu')(gap2d)
    batch = BatchNormalization()(act)
    dropout = Dropout(0.3)(batch)

    fc1 = Dense(256)(dropout)
    act = Activation('relu')(fc1)
    batch = BatchNormalization()(act)
    dropout = Dropout(0.4)(batch)

    # and a logistic layer
    predictions = Dense(3, activation='softmax')(dropout)
    
    # Create model.
    model = tf.keras.Model(visible, predictions, name='baseline')

    return model
model = get_baseline_model(224)


# In[ ]:


from math import floor
N_FOLDS = 8
EPOCHS = 8
INIT_LR = 3e-4
T_BS = 16
V_BS = 16
decay_rate = 0.95
decay_step = 1

skf = StratifiedKFold(n_splits=N_FOLDS, random_state=1234,)
log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [ModelCheckpoint(filepath='best_vgg16_model.h5', monitor='val_loss',mode='min',verbose=1,save_best_only=True,save_weights_only=True),
             LearningRateScheduler(lambda epoch : INIT_LR * pow(decay_rate, floor(epoch / decay_step))), tensorboard_callback]


# In[ ]:


resnet34, _ = Classifiers.get('densenet121')
conv_base = resnet34(weights='imagenet',
              include_top=False,
              input_shape=(224, 224, 3))

conv_base.trainable = False


gap2d = GlobalAveragePooling2D()(conv_base.output)
act = Activation('relu')(gap2d)
batch = BatchNormalization()(act)
dropout = Dropout(0.3)(batch)

fc1 = Dense(256)(dropout)
act = Activation('relu')(fc1)
batch = BatchNormalization()(act)
dropout = Dropout(0.4)(batch)
predictions = Dense(3, activation='softmax')(dropout)

model = Model(conv_base.input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=INIT_LR),
              metrics=['acc', tf.keras.metrics.AUC()])


# ### Training

# In[ ]:


submission_predictions = []
for epoch, skf_splits in zip(range(0,N_FOLDS),skf.split(trainX,trainYSparse)):

    train_idx = skf_splits[0]
    val_idx = skf_splits[1]
    
    # Create Model
    resnet34, _ = Classifiers.get('densenet121')
    conv_base = resnet34(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

    conv_base.trainable = False


    gap2d = GlobalAveragePooling2D()(conv_base.output)
    act = Activation('relu')(gap2d)
    batch = BatchNormalization()(act)
    dropout = Dropout(0.3)(batch)

    fc1 = Dense(256)(dropout)
    act = Activation('relu')(fc1)
    batch = BatchNormalization()(act)
    dropout = Dropout(0.4)(batch)
    predictions = Dense(3, activation='softmax')(dropout)

    model = Model(conv_base.input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=INIT_LR),
                  metrics=['acc', tf.keras.metrics.AUC()])
    
    if epoch != 0:
        # Load Model Weights
        model.load_weights('best_vgg16_model.h5') 
    
    history = model.fit(
                train_datagen.flow(trainX[train_idx], trainY[train_idx], batch_size=T_BS),
                steps_per_epoch=len(train_idx) // T_BS,
                epochs=EPOCHS,
                validation_data = val_datagen.flow(trainX[val_idx], trainY[val_idx], batch_size=V_BS),
                validation_steps = len(val_idx) // V_BS,
                callbacks=callbacks)
    
    if epoch >= 1:
        preds = model.predict(testX, batch_size=V_BS)
        submission_predictions.append(preds)
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("number of epochs")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('loss_performance'+'_'+str(epoch)+'.png')
    plt.clf()
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='valid')
    plt.title("model acc")
    plt.ylabel("accuracy")
    plt.xlabel("number of epochs")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('acc_performance'+'_'+str(epoch)+'.png')
    
    del history
    del model
    gc.collect()
    


# ## Grad-CAM

# In[ ]:


class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName

		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name

		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output, 
                     
				self.model.output])

		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]

		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)

		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]

		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))

		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")

		# return the resulting heatmap to the calling function
		return heatmap

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_JET):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)


# In[ ]:


model.load_weights('../working/best_vgg16_model.h5')


# In[ ]:


np.where(testY == 0)


# In[ ]:


covid1 = testX[50]
covid2 = testX[4]
covid3 = testX[35]
covid_list = [covid1, covid2, covid3]
pneumonia1 = testX[2]
pneumonia2 = testX[43]
pneumonia3 = testX[52]
pneumonia_list = [pneumonia1, pneumonia2, pneumonia3]


# In[ ]:


image = testX[7]
preds = model.predict(image[np.newaxis,...])
preds


# In[ ]:



preds = model.predict(image[np.newaxis,...])
i = np.argmax(preds[0])

# decode the ImageNet predictions to obtain the human-readable label

# # initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image[np.newaxis,...])

img_copy = np.copy(image)
img_copy -= img_copy.min((0,1))
img_copy = (255*img_copy).astype(np.uint8)
# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, img_copy, alpha=0.5)


# In[ ]:


plt.imshow(output, cmap = 'gray')


# In[ ]:


def plot_map(img_list):
    fig, axes = plt.subplots(len(img_list), 2, figsize=(15, 15))
    fig.suptitle('Pneumonia Grad-CAM\n',fontsize=20)
    
    for i, img in enumerate(img_list):
        preds = model.predict(img[np.newaxis,...])
        axes[i,0].imshow(img, cmap = 'bone')
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
      #  axes[i,0].set_title(f'{class_label[np.argmax(preds[:, 1:]) + 1]} / {class_label[np.argmax(label[:, 1:]) + 1]} / {np.max(preds[:, 1:]):.4f}')
        heatmap = cam.compute_heatmap(img[np.newaxis,...])
        img_copy = np.copy(img)
        img_copy -= img_copy.min((0,1))
        img_copy = (255*img_copy).astype(np.uint8)
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (img_copy.shape[1], img_copy.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, img_copy, alpha=0.5)
        axes[i,1].imshow(output)
        axes[i,1].set_xticks([])
        axes[i,1].set_yticks([])
        #axes[i,1].set_title("heatmap showing hemorrhage location")
    plt.subplots_adjust(wspace=1, hspace=0.2)
    plt.savefig('CovidGradCAM.png')


# In[ ]:


plot_map(covid_list)


# In[ ]:


plot_map(pneumonia_list)


# ### Evaluation

# In[ ]:


predY = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])


# In[ ]:


roc_auc_score(testY, predY, multi_class='ovo')


# In[ ]:


roc_auc_score(testY, predY, multi_class='ovr')


# In[ ]:


class_to_label_map = {2 : 'pneumonia', 1 : 'covid', 0 : 'normal'}


# In[ ]:


import seaborn as sns
def plot_multiclass_roc(y_test, y_score, n_classes, figsize=(17, 6)):

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], class_to_label_map[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(testY, predY, n_classes=3, figsize=(16, 10))


# In[ ]:


cm_mat = confusion_matrix(testY, np.argmax(predY, axis = -1))


# In[ ]:


import numpy as np

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 'xx-large')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
plot_confusion_matrix(cm_mat, 
                      normalize = False,
                      target_names = ['normal', 'covid', 'pneumonia'],
                      title        = "Confusion Matrix")


# In[ ]:


print(classification_report(testY, np.argmax(predY, axis = -1), target_names = ['normal', 'covid', 'pneumonia']))


# #### Confusion matrix

# In[ ]:


# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY, np.argmax(predY, axis = -1))
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))


# In[ ]:


get_ipython().system('rm -rf dataset')
get_ipython().system('rm -rf logs')

