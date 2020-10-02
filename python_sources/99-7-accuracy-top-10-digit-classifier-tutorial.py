#!/usr/bin/env python
# coding: utf-8

# Contents
# ==
# 
# [Introduction to The Kernel](#intro)<br/>
# [1. Preprocessing Data](#pre) <br/>
# &emsp;&emsp; [1.1 Reshaping Data](#reshape) <br/>
# &emsp;&emsp; [1.2 Normalizing Data](#norm) <br/>
# &emsp;&emsp; [1.3 Train-Test Split](#split) <br/>
# &emsp;&emsp; [1.4 Augmenting Data](#augment)<br/>
# [2. Building The Model](#model) <br/>
# &emsp;&emsp; [2.1 Model Architecture](#arch)<br/>
# &emsp;&emsp; [2.2 Hyperparameters Tuning](#params)<br/>
# [3. Model Evaluation](#eval) <br/>
# &emsp;&emsp; [3.1 Confusion Matrix](#metrics)<br/>
# [4. Error Analysis](#error)<br/>
# [5. General Notes](#notes)<br/>
# [6. Submission](#submit)

# <a id="intro"></a>
# # Introduction to The Kernel
# 

# Instead of using a MLP, I'll preprocesss the data from MNIST to attain the original images from the unravelled pixels and use a CNN as my model.
# CNNs are the best choice when it comes to image classification or any computer vision task, as well as providing relatively small number of weight parameters compared to MLP
# 

# ## Steps of The Kernel
# 1. Preprocessing the data, which includes reshaping, normalization and images augmentation
# 2. Building the model, trying different architectures and hyperparameters tuning
# 3. Model Evaluation and calculating the metrics of the model's performance
# 4. Error analysis and see how to improve furture performace
# 5. General notes on the model and MNIST dataset

# ## Importing Python Modules
# * **Data Handling:** numpy, pandas
# * **DL Model Building:** keras 
# * **Preprocessing:** keras, sklearn
# * **Evaluation Metrics:** sklearn
# * **Visualization:** matplotlib, seaborn, IPython.display

# In[ ]:


# import NumPy and Panas
import numpy as np 
import pandas as pd

# import Keras and its layers
import keras
from keras import Model
from keras.layers import Dense, Conv2D, pooling, Activation, BatchNormalization, Input, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# import preprocessing functions and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# import matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

# import time for checkpoint saving
import time


# Firstly, let's use pandas to load the data and see how is it structured

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


pd.set_option('display.max_columns', 6)
print(train_data.head(10), '\n')
print(test_data.head(10), '\n')

print(f'train data shape: {train_data.shape}')
print(f'test data shape: {test_data.shape}')


# The test dataset here is unlabelled and can only be used for the kernel submission, so we can only judge the model's performance on it manually.

# <a id="pre"></a>
# # 1. Preprocessing Data
# 

# <a id="reshape"></a>
# ## 1.1 Reshaping Data
# 
# The first step to do is to reshape the data to obtain the original images
# 

# In[ ]:


# storing the labeled dataset in numpy arrays
dataset_x = np.array(train_data.iloc[:, 1:]).reshape(42000, 28, 28, 1)
dataset_y = np.array(train_data.iloc[:, 0]).ravel()

# one-hot encoded labels instead of digits to use in categorical/multi-class classification
dataset_one_hot = keras.utils.to_categorical(dataset_y, num_classes=10)

# storing the unlabelled dataset in a numpy array for manual model testing
unlabelled_test_x = np.array(test_data.iloc[:, :]).reshape(28000, 28, 28, 1)

# make sure that the number of examples equals the number of labels after reshaping
assert(dataset_x.shape[0] == dataset_y.size)


# To make sure that our dataset has almost equal representation (equal number of examples) for all classes so that the model doesn't train on one class over another, we'll plot a countplot of the labels in the dataset.

# In[ ]:


# To get more control over visualization we'll define our figure instead of simply using plt.
fig = plt.figure(figsize=(8, 8))  # create figure
ax = fig.add_subplot(111)  # add subplot

sns.countplot(dataset_y);
ax.set_title(f'Labelled Digits Count ({dataset_y.size} total)', size=20);
ax.set_xlabel('Digits');

# writes the counts on each bar
for patch in ax.patches:
        ax.annotate('{:}'.format(patch.get_height()), (patch.get_x()+0.1, patch.get_height()-150), color='w')


# <a id="norm"></a>
# ## 1.2 Normalizing Data
# 
# Next step is to normalize the data, so that they get a value between 0 and 1.
# 
# This will make the model train more efficiently as it will reduce the variance in the dataset distribution, so the activations of different examples won't have big differences.
# 
# Think about it as if the image lies in 784-dim space (Features Space) and by normalizing we are making all the images bound by a hypersphere of radius 1

# In[ ]:


dataset_x = dataset_x / 255.0
unlabelled_test_x = unlabelled_test_x / 255.0


# Let's see how the images look after reshaping.

# In[ ]:


fig, axs = plt.subplots(1, 5, figsize=(25, 5))
indexes = np.random.choice(range(dataset_x.shape[2]), size=5)  # returns random 5 indexes

fig.suptitle('Original Images of MNIST', size=32)
for idx, ax in zip(indexes, axs):
    ax.imshow(dataset_x[idx,:, :, 0], cmap='gray');
    ax.set_title(f'Label: {dataset_y[idx]}', size= 20);


# <a id="split"></a>
# ## 1.3 Train-Test Split
# 
# Since the test dataset is unlabelled we'll be taking a portion of the labelled dataset to use as a test set and to evaluate the model's metrics on.

# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_one_hot, test_size=0.1)


# Since the dataset size is small we'll use an augmented images generator to give images from the same distribution as the examples in the train dataset but not necessairly the original examples.

# <a id="augment"></a>
# ## 1.4 Augmenting Data

# In[ ]:


# Augmentation Ranges
# Large values might lead to insignificant outliers or noisy examples
transform_params = {
    'featurewise_center': False,
    'featurewise_std_normalization': False,
    'samplewise_center': False,
    'samplewise_std_normalization': False,
    'rotation_range': 10, 
    'width_shift_range': 0.1, 
    'height_shift_range': 0.1,
#     'shear_range': 0.15, 
    'zoom_range': 0.1,
    'validation_split': 0.15
}
img_gen = ImageDataGenerator(**transform_params) 


# The augmentation generator will only use the train portion we got from the split.
# The model must never learn from the test images or even their augmentations, to have a meaningful performance and error analysis.

# In[ ]:


# used only to visualize the augmentations
visualizaion_flow = img_gen.flow(train_x, train_y, batch_size=1, shuffle=False)

# used to feed the model augmented training data
train_gen = img_gen.flow(x=train_x, y=train_y, subset='training', batch_size=96)

# used to feed the model augmented validation data
valid_gen = img_gen.flow(x=train_x, y=train_y, subset='validation', batch_size=96)


# The valid_gen used for validation will also use the training dataset from the split, as we said before that the model must never learn from the test data.
# 
# So, we are performing another split of the labelled dataset.

# Now, let's visualize how the generator augments the original images

# In[ ]:


fig, axs = plt.subplots(2, 4, figsize=(20,10))  # let's see 4 augmentation examples
fig.suptitle('Augmentation Results', size=32)

for axs_col in range(axs.shape[1]):
    idx = np.random.randint(0, train_x.shape[0])  # get a random index
    img = train_x[idx,:, :, 0]  # the original image
    aug_img, aug_label = visualizaion_flow[idx]  # the same image after augmentation
    
    axs[0, axs_col].imshow(img, cmap='gray');
    axs[0, axs_col].set_title(f'example #{axs_col} - Label: {np.argmax(train_y[idx])}', size=20)
    
    axs[1, axs_col].imshow(aug_img[0, :, :, 0], cmap='gray');
    axs[1, axs_col].set_title(f'Augmented example #{axs_col}', size=20)


# As we can see the new images are very similar but not quite the original ones.
# The model augments data randomly each time the training requires a new batch of images, so the model isn't training on the same examples exactly at each epoch which is the case if we have simply fitted the model using the original images. 

# <a id="model"></a>
# # 2. Building The Model

# Let's first define a Convolution Block to avoid repition while building our model.
# 
# It consists of
# 1. 2D Convolution - extracts features
# 2. BatchNormalization - enhances training efficiency and reduces Vanishing/Exploding Gradients effect
# 3. ReLU Activation Function
# 4. 2D Max Pool - optional, reduces overfitting and reduces the shape of the tensor 

# <a id="arch"></a>
# ## 2.1 Model Architecture

# In[ ]:


def conv_block(x, filters, kernel_size, strides, layer_no, use_pool=False, padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, name=f'conv{layer_no}',
               padding=padding)(x)
    
    x = BatchNormalization(name=f'bn{layer_no}')(x)
    x = Activation('relu', name=f'activation{layer_no}')(x)
    if use_pool:
        x = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], name=f'pool{layer_no}', padding='same')(x)
    return x


# After trying multiple architectures, the final one is
# 
# 
# <li> Input Layer </li>
# |
# <li> Series of ConvBloccks (conv1 to conv6)  </li>
# | 
# <li> Flatten Layer </li>
# <li> Dropout  </li>
# |
# <li> Fully-Connected Layer  </li>
# <li> BatchNormalization  </li>
# <li> Dropout  </li>
# <li> ReLU Activation Function  </li>
# | 
# <li> Fully-Connected Layer  </li>
# <li> BatchNormalization  </li>
# <li> Softmax Activation Function </li>

# In[ ]:


def conv_model(X):
    h, w, c = X.shape[1:]  # get shape of input (height, width, channels)
    X = Input(shape=(h, w, c))
    conv1 = conv_block(X, filters=8, kernel_size=[3, 3], strides=[1, 1], layer_no=1)
    conv2 = conv_block(conv1, filters=16, kernel_size=[2, 2], strides=[1, 1], layer_no=2)
    conv3 = conv_block(conv2, filters=32, kernel_size=[2, 2], strides=[1, 1], layer_no=3, use_pool=True)
    
    conv4 = conv_block(conv3, filters=64, kernel_size=[3, 3], strides=[2, 2], layer_no=4)
    conv5 = conv_block(conv4, filters=128, kernel_size=[2, 2], strides=[1, 1], layer_no=5)
    conv6 = conv_block(conv5, filters=256, kernel_size=[2, 2], strides=[1, 1], layer_no=6, use_pool=True)
    
    flat1 = Flatten(name='flatten1')(conv6)
    drop1 = Dropout(0.35, name='Dopout1')(flat1)
    
    dens1 = Dense(128, name='dense1')(drop1)
    bn7 = BatchNormalization(name='bn7')(dens1)
    drop2 = Dropout(0.35, name='Dopout2')(bn7)
    relu1 = Activation('relu', name='activation7')(drop2)
    
    dens1 = Dense(256, name='dense01')(relu1)
    bn7 = BatchNormalization(name='bn07')(dens1)
    drop2 = Dropout(0.5, name='Dopout02')(bn7)
    relu1 = Activation('relu', name='activation07')(drop2)
    
    dens2 = Dense(10, name='dense2')(relu1)
    bn8 = BatchNormalization(name='bn8')(dens2)
    output_layer = Activation('softmax', name='softmax')(bn8)
    
    model = Model(inputs=X, outputs=output_layer)
    return model


# In[ ]:


# create the model
model = conv_model(train_x)


# <a id="params"></a>
# ## 2.2 Hyperparameters Tuning

# One of the hyperparameters to tune is the learning rate and the decay.
# Funnily enough, I found a decay rate equal to half of the learning rate that I used works pretty well.
# If you want, I suggest taking a look at keras annealers to raise the efficiency of training even more.

# In[ ]:


def build_model(learning_rate=0.00065):
    optimizer = keras.optimizers.RMSprop(lr=learning_rate, decay=learning_rate/2)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# not used - add to callbacks if wanted
plateau_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                              patience=6, min_lr=0.0000001)


# In[ ]:


# build the model computational graph and print out its summary
build_model()
model.summary()


# To have an updated visualization of the training process to see how it's doing over ep-ochs instead of just looking at values prointed out each epoch, we'll build a Plotter class to serve as a keras callback which keras executes it's methods on different milestones in the training.

# In[ ]:


class Plotter(keras.callbacks.Callback):
    def plot(self):  # Updates the graph
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Training Curves', size=32)
        
        # plot the losses
        ax1.plot(self.epochs, self.losses, label='train_loss')
        ax1.plot(self.epochs, self.val_losses, label='val_loss')
        
        # plot the accuracies
        ax2.plot(self.epochs, self.acc, label='train_acc')
        ax2.plot(self.epochs, self.val_acc, label='val_acc')
    
        ax1.set_title(f'Loss vs Epochs')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        
        ax2.set_title(f'Accuracy vs Epochs')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        
        ax1.legend()
        ax2.legend()
        plt.show()
        
        # print out the accuracies at each epoch
        print(f'Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]:.5f}, val_acc={self.val_acc[-1]:.5f}')
    
    def on_train_begin(self, logs={}):
        # initialize lists to store values from training
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.batch_no = []
        self.acc = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs={}):
        # append values from the last epoch
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.epochs.append(epoch)
        self.plot()  # update the graph
        
    def on_train_end(self, logs={}):
        self.plot()
               
plotter = Plotter()


# In[ ]:


use_generator = True  # toggles between using the augmentation generator or the original images


# The learning rate and the batch size are already tuned, the only one that remains is the number of epochs.

# In[ ]:


callbacks = [plotter]


# In[ ]:


if use_generator:
    model.fit_generator(train_gen, validation_data=valid_gen, epochs=120, 
                        steps_per_epoch=train_x.shape[0]*0.85//96, 
                        validation_steps=train_x.shape[0]*0.15//96, callbacks=callbacks)
else:
    model.fit(x=train_x, y=train_y, epochs=80, batch_size=32, callbacks=callbacks,
              validation_split=0.15)


# In[ ]:


# save model
keras.models.save_model(model, f'model_{time.time()}.h5')


# Since we're using Dropout, it's not surprising to see the training accuracy less than the validation accuracy.
# That's because Dropout is used only in training to prevent overfitting, then disabled in both validation and testing.
# 
# Dropout shuts down some neurons randonly each training step, so it's like using multiple weaker models during training.

# <a id="eval"></a>
# # 3. Model Evaluation

# The calculate_performace function will calculate the accuracy and gives back data about the falses of the model's predictions to visualize and analyze them

# In[ ]:


def calculate_performance(labels, pred, dataset):
    pred_cat = np.argmax(pred, axis=1)  # categorical predictions 0-9
    labels_cat = np.argmax(labels, axis=1)  # categorical labels 0-9
    
    # a boolean vector of element-wise comparison between prediction and label
    corrects = (pred_cat == labels_cat)
    
    # get the falses data
    falses = dataset[~corrects]  # the falses images
    falses_labels = labels_cat[~corrects]  # true labels of the falsely classified images - categorical
    falses_preds = pred[~corrects]  # the false predictions of the images - 10-dim prediction
     
    examples_num = labels.shape[0]  # total numbers of examples
    accuracy = np.count_nonzero(corrects) / examples_num

    return accuracy, [falses, falses_labels, falses_preds], [labels_cat, pred_cat]


# Even though we used augmented versions of the training dataset while training we can't use it to get intuition of the model's performance or metrics because it in a way learnt from these images.
# 
# Note that augmentation would give an augmented image, in the Features Space, very close to the original image so the hyperplane which is our model would very likely classify the original image correctly.

# We'll only test our model on the original training examples to make sure it's performace is very high as expected and to work as a debug point if we notrice that there's something off about it's performance.

# <a id="metrics"></a>
# ## 3.1 Confusion Matrix

# In[ ]:


train_pred = model.predict(train_x)  # predict the whole training dataset

# calculate the accuracy over the whole dataset and get information about falses
train_accuracy, train_falses_data, (true_labels, pred_labels) = calculate_performance(train_y, train_pred ,train_x)

print(f'Don\'t use as a metric - Original Training Dataset Accuracy: {np.round(train_accuracy*100, 3)}%')

plt.figure(figsize=(10, 10))

# Calculate the confusion matrix and visualize it
train_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)
sns.heatmap(data=train_matrix, annot=True, cmap='Blues', fmt=f'.0f')

plt.title('Confusion Matrix - Training Dataset', size=24)
plt.xlabel('Predictions', size=20);
plt.ylabel('Labels', size=20);


# We'll gain insight from the model evaluation on the test dataset.
# For exapmle, we can find out if a false classifications to a certain digit is biased towards another specific digit.

# In[ ]:


test_pred = model.predict(test_x)  # predict the whole test dataset

# calculate the accuracy over the whole dataset and get information about falses
test_accuracy, test_falses_data, (true_labels, pred_labels) = calculate_performance(test_y, test_pred, test_x)

print(f'Test Dataset Accuracy: {np.round(test_accuracy*100, 3)}%')

plt.figure(figsize=(10, 10))

# Calculate the confusion matrix and visualize it
test_matrix = confusion_matrix(y_pred=pred_labels, y_true=true_labels)
sns.heatmap(data=test_matrix, annot=True, cmap='Blues', fmt=f'.0f')

plt.title('Confusion Matrix - Test Dataset', size=24)
plt.xlabel('Predictions', size=20);
plt.ylabel('Labels', size=20);


# **You should be getting a model's accurace around 99.5% on the test dataset.** <br/>
# ** Note that the accuracy will be different for each run of the kernel, but all of them would almost be equal.**
# 
# 
# Our model has better acuracy on both Training and Test Datasets than the generated images meaning that the generated images were harder for the model, but they are all very close as they all fall into the same distribution.

# Let's visualize random examples predictions from the test dataset.

# In[ ]:


random_idx = np.random.choice(range(test_y.shape[0]), size=10)  # get random 10 indexes

examples = test_x[random_idx]  # the images
preds = model.predict(examples)  # the predictions - 10-dim probabilities
labels = np.argmax(test_y[random_idx], axis=1)  # the labels - categorical
preds_cat = np.argmax(preds, axis=1)  # the predictions - categorical


# In[ ]:


fig, axs = plt.subplots(2, 5, figsize=(28, 12));
fig.suptitle('Model Predictions - Test Dataset', size=32)

for ax, pred, pred_prob, label, example in zip(axs.ravel(), preds_cat, preds, labels, examples):
    ax.imshow(example[:, :, 0] ,cmap='gray');
    ax.set_title(f'Label: {label}\nPrediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.',
                size = 15)
    ax.axis('off')


# Very Good!
# We can see that the model classifies the imgaes not only correctly but with a great confidence.
# It's acceptable if there is some of examples that the model clssifies correctly but with a confidence/probability less than 0.5 on hard examples 
# To judge the model fairly if it's doing well or not on a certain image, we should ask ourselves 2 questions:
# 
# **1. Is a human being able to classify the image corerectly with great confidence?**
#     Maybe the image is a very hard example (blurry, unclear...etc)
# 
# 
# **2. How does the prediction's probability fare when compared to random chance?** Each class has a 10% probability if chosen randomly, so if the model's probability of a correctly classified image is above 10% then it has some knowledge about classifying the class correctly, not a lot but some.

# Let's visualize random examples from the unlablled dataset and see how the model performs

# In[ ]:


random_idx = np.random.choice(range(test_y.shape[0]), size=10)  # get random 10 indexes

examples = unlabelled_test_x[random_idx]  # the images
preds = model.predict(examples)  # the predictions - 10-dim probabilities
preds_cat = np.argmax(preds, axis=1)  # the predictions - categorical


# In[ ]:


fig, axs = plt.subplots(2, 5, figsize=(30, 12));
fig.suptitle('Predictions of Unlabelled Images', size=32)

for ax, pred, pred_prob, example in zip(axs.ravel(), preds_cat, preds, examples):
    ax.imshow(example[:, :, 0] ,cmap='gray');
    ax.set_title(f'Prediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.', size = 15)
    ax.axis('off')


# <a id="error"></a>
# # 4. Error Analysis

# Analyzing Falses and Errors to find out where the model fails and hopefully gain an intuition on how to enhance it.
# 

# Checking the error of the test dataset as the model hasn't seen its examples or augmented versions of them before, so they would serve as a fair test to help us find out where our model fails.

# In[ ]:


# checking the falses of the test dataset
falses_data = test_falses_data # set the dataset to check
falses_examples, falses_labels, falses_preds = falses_data
falses_idx = np.argmax(falses_preds, axis=1)


# By plotting a countbar of the labels of the images that were misclassified, we can see if there's any Digit that the model needs to train on more.
# 
# 
# The confusion matrix would help us find if the misclassifications of one digit are heavily biased towards another digit.
# This can happen to similar digits like 4 and 9

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('False Classifications Plots', size=32)

sns.countplot(falses_labels, ax=ax1);
ax1.set_title(f'Falses ({falses_labels.size} total)',size=20);
ax1.set_xlabel('Digits', size=15);
ax1.set_ylabel('Count', size=15);

for patch in ax1.patches:
    bar_height = patch.get_height()
    ax1.annotate(f'{bar_height}', (patch.get_x()+0.25, bar_height-0.2), color='w', size=15);
    

falses_matrix = confusion_matrix(y_pred=falses_idx, y_true=falses_labels)
sns.heatmap(data=falses_matrix, annot=True, cmap='Blues', fmt=f'.0f')

plt.xlabel('Predictions', size=20);
plt.ylabel('Labels', size=20);


# As we can see, the number of the misclassified images is rather small compared to the dataset size.
# The misclassifications aren't exclusive to one digit in particular even if some digits have more number of falses than other ones.
# 
# 
# The confusion matrix displays an almost equal spread which means there's no falses that are biaseds towards a certain digit.

# Let's visualize some of the flase predictions, to try to get more of an understanding of the model's misclassifications.

# In[ ]:


random_falses = np.random.choice(range(falses_labels.size), size=np.min((10, falses_labels.size)), replace=False)

examples = falses_examples[random_falses]
preds_probs = falses_preds[random_falses]
labels = falses_labels[random_falses]
preds_binary = np.argmax(preds_probs, axis=1)


# In[ ]:


fig, axs = plt.subplots(2, 5, figsize=(30, 12));
fig.suptitle('Misclassified Images', size=32)

for ax, pred, pred_prob, label, example in zip(axs.ravel(), preds_binary, preds_probs, labels, examples):
    ax.imshow(example[:, :, 0] ,cmap='gray');
    ax.set_title(f'Label: {label}\nPrediction: {pred} with {np.round(np.max(pred_prob)*100, 3)}% Confidence.',
                size = 15)
    ax.axis('off')


# As seen above, the model struggles with hard examples that humans will struggle with some of them as well.
# There are mislabelled images in the MNIST that serve as noise or an outlier for the model.

# <a id="notes"></a>
# # 5. General Notes

# * This dataset has a relatively small size, that's why we used augmentation to reduce the chance of overfitting while remaining in the overall distributioin of the dataset.
# 
# 
# * MNIST is a pre-cleaned dataset and used as a models' performaces benchmarks. A more uncurated dataset would need more cleaning and preprocessing like checking if there are any labels missing or if the dataset require feature engineering like Dimensionality Reduction.
# 
# 
# * This model suffers from a fatal flaw that is it will give a prediction of a digit to any image it sees, so if we give it a picture of a human, dog, cat...etc with the correct shape it will classify it as a number. This is discussed in detail in this informative post. I highly recommend reading it. [https://emiliendupont.github.io/2018/03/14/mnist-chicken/](http://)
# 
# 
# * We can add another class to the dataset called "Not-a-Digit" and contains random images of non digits, but the dataset needs to be larger so that the dataset isn't inproportionate to one of its classes or add an appropriate number of examples for this new class, but because of the random nature of the class, its examples won't have consistient features nor distribution thus the model won't do well in case of unseen images of this new model.
# 
# 
# * We can only use this model to classify hand-written images (same distribution) if we want it to have the same accuracy with the new data, and the images must be of the same shape and be preprocessed the same way the training data were before being given to the model.
# 
# 
# * We can use this model as an initial model for recognizing digits outside this distribution but with training on the new digits distribution, its weights will be fine-tuned to be able to classify them, or we can use transfer learning to freeze some of the model's layers weights and change the architecture a bit then retrain it for a whole new problem that doesn't have anything to do with digits.
# 
# 
# * Finally, I hope this kernel explained a lot of tools used in preprocessing, model training, evaluation and error analyzing. You're welcome to fork it and change it as you like to try your own concepts. Feel free to use the visualiztion in this kernel to save some time while building your own kernels. 

# <a id="submit"></a>
# # 6. Submission

# In[ ]:


# cateforical predictions of the dataset
pred = model.predict(unlabelled_test_x)
pred_cat = np.argmax(pred,axis = 1)

pred_series = pd.Series(pred_cat,name='Label')  # pandas series of predictions
id_series = pd.Series(range(1, unlabelled_test_x.shape[0]+1), name='ImageId') # pandas series of IDs

submission = pd.concat((id_series, pred_series), axis=1)

print(submission.head(5))


# In[ ]:


# save to csv
submission.to_csv('submission.csv', index=False)

