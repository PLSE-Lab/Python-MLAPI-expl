#!/usr/bin/env python
# coding: utf-8

# **Image Classification: Dog Breeds Classification using Convolutional Neural Networks and Transfer Learning**
# <br><br>We will use pre-trained state-of-the-art benchmark CNN models to classify various dog breeds based on input images of dogs. We will use the Keras deep learning library and supported functionalities for our deep learning model.
# <br>-10,000 training examples for training and validation sets, 10,000 test examples
# <br>-120 classes of dog breeds
# <br>-(X,Y) supervised learning dataset where X is the input image and Y is the ground-true class label
# <br><br><u>Methodology:</u>
# <br>-Use a pre-trained CNN to extract an input image's feature-learned-representation-vector during forward-pass
# <br>-Take said representation from the CNN architecture's "bottleneck" layer
# <br>-Run the feature-representation-vectors through a linear Logistic Regression classifier for prediction purposes
# <br><br><u>Notes:</u>
# <br>-Instead of extracting the bottleneck feature-vector then running it through a [separate] LR classifier, we could alternatively have appended another fully-connected layer
# to the end of the CNNs. This would have likely produced better results, but would be more time-consuming and resource-intensive to train and fine-tune. Classifying the feature-representations via a simple logistic regression classifier provides a good starting baseline
# <br>-CNN architectures include VGG16, Xception, Inception, and [Xception+Inception] ensemble via extracted-feature-vector concatenation
# <br><br><u>Sources:</u>
# <br>Kaggle Challenge Competition: https://www.kaggle.com/c/dog-breed-identification
# <br>Stanford Dogs Dataset: http://vision.stanford.edu/aditya86/ImageNetDogs/
# <br>Keras Pre-trained Models: https://keras.io/applications/#models-for-image-classification-with-weights-trained-on-imagenet

# **Import Dependencies**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm # Fancy progress bars

from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3


# **Loading up Keras Pretrained Models into Kaggle Kernels**

# In[ ]:


# This project was done in Kaggle Kernels, where we'd need to copy the Keras pretrained models
# into the cache directory (~/.keras/models) where keras is looking for them

# Display the pretrained models that we have prepared in our file directory
get_ipython().system('ls ../input/keras-pretrained-models/')


# In[ ]:


# Create keras cache directories in Kaggle Kernels to load the pretrained models into
cache_dir = expanduser(join('~', '.keras')) # Cache directory
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models') # Models directory
if not exists(models_dir):
    makedirs(models_dir)
    
# Copy a selection of our pretrained models files onto the keras cache directory so Keras can access them
# Selection includes all the models labeled notop; and both resnet50 models
get_ipython().system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/')

# Display our pretrained models that are located in the keras cache directory
get_ipython().system('ls ~/.keras/models')


# In[ ]:


get_ipython().system('ls ../input/dog-breed-identification')


# **Use a Subset of the Total Dataset for Faster Prototyping**
# <br>-Using all of the images would take a significant amount of computation time. For faster prototyping, let's initially look at the top 25 classes by frequency
# <br>-Can expand to include full 120 classes with more compute

# In[ ]:


INPUT_SIZE = 224
NUM_CLASSES = 25
SEED = 7

# Read the y-true-labels file as well as the prediction file
data_dir = '../input/dog-breed-identification'
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))

print("Num training examples files: " + str(len(listdir(join(data_dir, 'train')))) + " | Num training examples labels: "
      + str(len(labels)))
print("Num test examples files: " + str(len(listdir(join(data_dir, 'test')))) + " | Num test examples predictions: " 
     + str(len(sample_submission)))


# In[ ]:


selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
group = labels.groupby(by='breed', as_index=False).agg({'id': pd.Series.nunique})
group = group.sort_values('id',ascending=False)
print(group)
labels['rank'] = group['breed']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)

np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))

# Train / validation split into 80%/20%
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]


# In[ ]:


def read_img(img_id, train_or_test, size):
    """
    Read and resize image.
    # Args:
        img_id: image filepath string
        train_or_test: string "train" or "test"
        size: resize the original image
    # Returns:
        Image as a numpy array
    """
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size = size)
    img = image.img_to_array(img)
    return img


# **Extracting Image Feature-Representations: VGG16 Network**

# In[ ]:


INPUT_SIZE = 224
POOLING = 'avg'
x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype = 'float32')
for i, img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
print('Training images shape: {} size: {:,}'.format(x_train.shape, x_train.size))


# In[ ]:


# Train / validation split via index
Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print("X_train.shape: " + str(Xtr.shape))
print("y_train.shape: " + str(ytr.shape))
print("X_val.shape: " + str(Xv.shape))
print("y_val.shape: " + str(yv.shape))

# Extracting image representation bottleneck features ("bf")
vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)
train_vgg_bf = vgg_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size=32, verbose=1)
print('VGG training set bottleneck features shape: {} size: {:,}'.format(train_vgg_bf.shape, train_vgg_bf.size))
print('VGG validation set bottleneck features shape: {} size: {:,}'.format(valid_vgg_bf.shape, valid_vgg_bf.size))
print("VGG bottleneck features should be a 512-dimensional vector for each image example / prediction")


# **Logistic Regression on Extracted Bottleneck Features: VGG16**
# <br>Note: We could have also attached a fully-connected layer onto the end of the pre-trained network.
# <br>This would have worked fine, although requiring more compute and fine-tuning vs simply taking the extracted feature-representation from a pre-trained network
# <br> and doing a logistic regression on the feature-vector.

# In[ ]:


# Optimizer: Limited-memory BFGS
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_vgg_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_vgg_bf)
valid_preds = logreg.predict(valid_vgg_bf)

print('Validation VGG LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation VGG Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))


# **Extracting Image Feature-Representations: Xception Network**

# In[ ]:


INPUT_SIZE = 299
POOLING = 'avg'
x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
for i, img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
print("Training images shape: {} size: {:,}".format(x_train.shape, x_train.size))


# In[ ]:


Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print("X_train.shape: " + str(Xtr.shape))
print("y_train.shape: " + str(ytr.shape))
print("X_val.shape: " + str(Xv.shape))
print("y_val.shape: " + str(yv.shape))

xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
print('Xception training bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
print('Xception validation bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))


# **Logistic Regression on Extracted Bottleneck Features: Xception**

# In[ ]:


logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_x_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)
print("Validation Xception LogLoss {}".format(log_loss(yv, valid_probs)))
print("Validation Xception Accuracy {}".format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))


# **Extracting Image Feature-Representations: Inception Network**

# In[ ]:


Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print("X_train.shape: " + str(Xtr.shape))
print("y_train.shape: " + str(ytr.shape))
print("X_val.shape: " + str(Xv.shape))
print("y_val.shape: " + str(yv.shape))

inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
train_i_bf = inception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
valid_i_bf = inception_bottleneck.predict(Xv, batch_size=32, verbose=1)
print('InceptionV3 training bottleneck features shape: {} size: {:,}'.format(train_i_bf.shape, train_i_bf.size))
print('InceptionV3 validation bottleneck features shape: {} size: {:,}'.format(valid_i_bf.shape, valid_i_bf.size))


# **Logistic Regression on Extracted Bottleneck Features: Inception**

# In[ ]:


logreg = LogisticRegression(multi_class='multinomial', solver = 'lbfgs', random_state=SEED)
logreg.fit(train_i_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_i_bf)
valid_preds = logreg.predict(valid_i_bf)

print('Validation Inception LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation Inception Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))


# **Logistic Regression on Combination of Extracted Features: [Xception + Inception]**
# <br>-Leveraging the feature-extracting capabilities of multiple pre-trained models
# <br>-Benefit of using a separate independent classifier as opposed to attaching FC-layers to each of these networks: easier ensembling

# In[ ]:


X = np.hstack([train_x_bf, train_i_bf]) # This is a array-concat function that stacks horizontally instead of vertically
V = np.hstack([valid_x_bf, valid_i_bf])
print("Full training bottleneck features shape: {} size: {:,}".format(X.shape, X.size))
print("Full validation bottleneck features shape: {} size: {:,}".format(V.shape, V.size))

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(X, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(V)
valid_preds = logreg.predict(V)
print("Validation Xception+Inception LogLoss {}".format(log_loss(yv, valid_probs)))
print("Validation Xception+Inception Accuracy {}".format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), 
                                                                        valid_preds)))


# **Summary and Error Checking**
# <br><br>-We observe that the Xception+Inception ensemble performs very well on the given classification problem, yielding LogLoss of <0.12 and 96.5%+ accuracy.
# <br>-Below are some instances of the model's misclassifications

# In[ ]:


valid_breeds = (yv * range(NUM_CLASSES)).sum(axis=1)
error_idx = (valid_breeds != valid_preds)
for img_id, breed, pred in zip(labels.loc[valid_idx, 'id'].values[error_idx],
                               [selected_breed_list[int(b)] for b in valid_preds[error_idx]],
                               [selected_breed_list[int(b)] for b in valid_breeds[error_idx]]):
    fix, ax = plt.subplots(figsize=(5,5,))
    img = read_img(img_id, 'train', (299,299))
    ax.imshow(img/255)
    ax.text(10, 250, 'Prediction: %s' % pred, color='w', backgroundcolor='r', alpha=0.8)
    ax.text(10, 270, 'LABEL: %s' % breed, color='k', backgroundcolor='g', alpha=0.8)
    ax.axis('off')
    plt.show()


# **References:**
# <br>-The above project takes significant inspiration and guidance from beluga's posted public kernel on Kaggle: https://www.kaggle.com/gaborfodor
# <br>-https://www.kaggle.com/gaborfodor/dog-breed-pretrained-keras-models-lb-0-3

# In[ ]:




