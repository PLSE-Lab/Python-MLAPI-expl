#!/usr/bin/env python
# coding: utf-8

# # MNIST digit classification with advanced CNN
# 
# The purpose of this kernel is to explore how CNN behave with MNIST?  
# Short answer : very well with the right layer setup
# 
# The best model achieve a 0.9962 accuracy on test set with 30 epochs / 5 folds and 0.997 with 200 epochs / 10 folds using this layer setup:
# -  [C32-C32-BN-MP-Dr0.25]-[C64-C64-BN-MP-Dr0.25]-[C128-C128-BN-MP-Dr0.25]-D300-D10
# 
# Where the layers are abbreviated using *Chris Deotte* convention with  
# - C32 = Conv2D(32)
# - BN = Batch Normalization
# - MP = MaxPoolin
# - Dr0.25  = Dropout(0.25)
# - D300 = Dense(300)
# 
# Thanks to the many public kernels and online ressources where inspiration has been taken from.
# 
# Aditionally a very good description of accuracy reachable for various models can be found on [How to score 97%, 98%, 99%, and 100%](https://www.kaggle.com/c/digit-recognizer/discussion/61480)
# 
# # Dataset load

# In[ ]:



import math
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.shape


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.shape


# In[ ]:


img_width  = 28
img_height = 28
nclasses   = 10
classes    = np.arange(10)
labels     = [ str(c) for c in classes ]
train_X = train.iloc[:,1:].values.reshape(train.shape[0], img_width, img_height, 1)
train_y = train.iloc[:,0].values


# In[ ]:


test_X = test.iloc[:,0:].values.reshape(test.shape[0], img_width, img_height, 1)


# # Digit distribution among examples

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(10,4))
sns.countplot(train_y, ax=ax)
txt = ax.set_title("Train Digits distribution, std(count)={0:0.2f}".format(np.std(np.histogram(train_y, np.arange(11))[0])))


# Values are well distributed although with some drift between digits, notably the 5 is underrepresented

# # Input visualization
# 
# - green number is the digit train label

# In[ ]:


def plot_digits(X, y, sel, predict=None, suptitle=""):
    nrow = math.ceil(len(sel)/20)
    fig, axes = plt.subplots(nrow, 20, figsize=(14, nrow*1.2))
    for i,ax in enumerate(axes.flatten()):
        ax.imshow(X[sel[i],:,:].squeeze(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if y is not None:
            ax.text(0, -2, str(y[sel[i]]), color="white", bbox=dict(facecolor='green', alpha=0.5))
        if predict is not None:
            ax.text(img_width/2, -2, str(predict[sel[i]]), color="white", bbox=dict(facecolor='purple', alpha=0.5))
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle(suptitle)

def get_shuffle(n, seed=12):
    rstate = np.random.RandomState(seed)
    sel = np.arange(n)
    rstate.shuffle(sel)
    return sel
    
    
# TRAIN
train_sel = get_shuffle(len(train_X))
plot_digits(train_X, train_y, train_sel[0:60], suptitle="Train digits")

# TEST
test_sel = get_shuffle(len(test_X))
plot_digits(test_X, None, test_sel[0:60], suptitle="Test digits")


# # Normalization step1: check intensity distribution
# 
# First let check pixels intensity distribution along  and Y axis

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
for digit in range(10):
    ax1.plot( (train_X[train_y==digit]**2).mean(axis=(0,1,3)), label=str(digit) )
    ax2.plot( (train_X[train_y==digit]**2).mean(axis=(0,2,3)), label=str(digit) )
ax1.set_title("TRAIN: mean digit Power distribution over X axis")
ax1.set_xlabel("X")
ax2.set_title("TRAIN: mean digit Power distribution over Y axis")
ax2.set_xlabel("Y")
leg = ax1.legend()
leg = ax2.legend()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))
ax1.plot( (test_X**2).mean(axis=(0,1,3)), label="TEST" )
ax1.plot( (train_X**2).mean(axis=(0,1,3)), label="TRAIN" )
ax2.plot( (test_X**2).mean(axis=(0,2,3)), label="TEST" )
ax2.plot( (train_X**2).mean(axis=(0,2,3)), label="TRAIN" )
ax1.set_title("TEST: mean Power distribution over X axis")
text = ax1.set_xlabel("X")
ax2.set_title("TEST: mean Power distribution over Y axis")
text = ax2.set_xlabel("Y")
leg = ax1.legend()
leg = ax2.legend()


# The distribution looks very different among digits of the training set.
# This is very encouraging for classification. It also explains why strategies like linear regressions and random forrest are very successfull.
# 
# Both sets have a surprisingly similar envelope distribution: this is also a good hint for generalization!
# 
# 
# Let have a look at pixel values distribution:

# In[ ]:


fig = plt.figure(figsize=(14,3))
train_count, train_bins = np.histogram(train_X, range(256))
plt.plot(train_bins[1:-1], train_count[1:], label="train")

test_count, test_bins = np.histogram(test_X, range(256))
plt.plot(test_bins[1:-1], test_count[1:], label="test")
plt.title("pixel values distribution")

leg = plt.legend()


# The zero value has been omitted as most of the pixels are black. 

# # Max normalization
# 
# The data has a clean distribution in pixel space and a spiked distribution in pixel values, but in the first try I noticed a very low accuracy rate of 0.1 which was really surprising.  
# After checking, the dtype input was `int64` and setting it to `float64` by using max normalization got accuracy over 0.97 on the first epoch with the Model 1 (see bellow) simple CNN :-)

# In[ ]:


train_X = train_X / train_X.max()
test_X = test_X / test_X.max()


# # Keras CNN Model

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow import set_random_seed


# In[ ]:


class MNIST_CNN:
    """All models have a final Dense(300) layer with Dropout
    Model 1: [Conv2d(16/32/64) + Dropout(0.5)] x 3 
        => 0.991 : wow simple is beautiful!
    Model2: data aug(rot/shift/zoom) +  [Conv2d(16/32/64) + Dropout(0.5)] x 3  
        => 0.994
    Model3: data aug(rot/shift/zoom) +  [Conv2d(16/32/64) + BatchNorm + MaxPool + Dropout(0.25)] x 3
        => 0.995
    Model3: data aug(rot/shift/zoom) +  [Conv2d(32/64/128) + BatchNorm + MaxPool + Dropout(0.25)] x 3
        => 0.99657
    """
    def build(self, img_rows, img_cols, nclasses=10):
        input_shape=(img_rows, img_cols, 1)
        model = Sequential()
        
        # Suite of 3x Conv2D + BatchNorm + Dropout
        model.add(Conv2D(32, (3,3), activation="relu", input_shape=(img_rows, img_cols, 1),
                        kernel_initializer="glorot_normal", padding="same"))
        model.add(Conv2D(32, (3,3), activation="relu",
                        kernel_initializer="glorot_normal", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), activation="relu",
                        kernel_initializer="glorot_normal", padding="same"))
        model.add(Conv2D(64, (3,3), activation="relu",
                        kernel_initializer="glorot_normal", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, (3,3), activation="relu",
                        kernel_initializer="glorot_normal", padding="same"))
        model.add(Conv2D(128, (3,3), activation="relu",
                        kernel_initializer="glorot_normal", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
      
        # Final layers
        model.add(Flatten())
        model.add(Dense(300, activation="relu"))
        model.add(Dropout(0.25))

        model.add(Dense(nclasses, activation="softmax"))
        self.model = model
    
    def compile(self, **kw):
        kw1 = dict(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
        kw1.update(kw)
        self.model.compile(**kw1)
        
    def fit(self, *arg, **kw):
        return self.model.fit(*arg, **kw)
        
mdl = MNIST_CNN()
mdl.build(img_width, img_height)
mdl.compile()
mdl.model.summary()
        


# # Use data augmentation
# 
# Implementing data augmentation with ImageDataGenerator adds a continuous flow of slightly modified images from the original inputs. 
# 
# When enabling data augmentation it has the following effect that 
# - the cross validation accuracy is actually better than the trainig set    
# 
# This may be explained by the fact that the network does never see twice the same training input, however generalization becomes better while averaging the multiple inputs.
# On the contrary the cross validation set is never augmented, hence it is easier to classify.
# 
# One open question is how much similar should be both metrics in order to be otimal?

# In[ ]:


datagen = ImageDataGenerator(
    rotation_range=30,
    #shear_range=30,
    zoom_range=[0.9, 1.1],
    width_shift_range = 2,
    height_shift_range = 2)
datagen.fit(train_X)

aug_X, aug_y = datagen.flow(train_X, train_y, batch_size=60).next()
plot_digits(aug_X, aug_y, np.arange(60), suptitle="Data Augmented digits")


# # Use a stratified K-Fold validation strategy
# 
# K-Fold has the advantage that it provides a reliable estimator of model generalization on test data.

# In[ ]:


DEBUG = 0
K = 3 if DEBUG else 10
set_random_seed(777)
skf = StratifiedKFold(n_splits=K, shuffle=True)
train_y_cat = to_categorical(train_y)
histories = []
test_pred  = np.zeros((len(test_X), nclasses), dtype=np.float64)
train_pred = np.zeros((len(train_X), nclasses), dtype=np.float64)
val_pred   = np.zeros((len(train_X), nclasses), dtype=np.float64)
train_accs = []
val_accs   = []

for i, (train_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
    t0 = time.time()
    batch_size=256 if DEBUG else 64
    mdl = MNIST_CNN()
    mdl.build(img_width, img_height)
    mdl.compile()
    histories.append(
        mdl.model.fit_generator(
            datagen.flow(train_X[train_idx], train_y_cat[train_idx], batch_size=batch_size),
            epochs=5 if DEBUG else 200,
            verbose=1 if DEBUG else 0,
            steps_per_epoch=len(train_idx) / batch_size,
            validation_data=(train_X[val_idx], train_y_cat[val_idx]), 
        ),
    )
    test_pred  += mdl.model.predict(test_X)
    train_pred += mdl.model.predict(train_X)
    val_pred[val_idx] += mdl.model.predict(train_X[val_idx])
    train_accs.append( accuracy_score(train_y, mdl.model.predict(train_X).argmax(axis=1)) )
    val_accs.append( accuracy_score(train_y[val_idx], val_pred[val_idx].argmax(axis=1)) )
    print("******* [dt={0}] accuracy_scores at fold {1}: train={2} val={3}".format(
        time.time() - t0, i + 1, train_accs[-1], val_accs[-1] ) )
    
test_y = np.argmax(test_pred, axis=1)
val_y  = np.argmax(val_pred, axis=1)


# In[ ]:


print("global fit accuracy_scores: train={0} val={1}".format(
    accuracy_score(train_y, train_pred.argmax(axis=1)),
    accuracy_score(train_y, val_pred.argmax(axis=1)) ))

print(f"{K}-Fold train accuracy = {np.mean(train_accs):0.4f} +- {np.std(train_accs):0.4f}")
print(f"{K}-Fold cross validation accuracy = {np.mean(val_accs):0.4f} +- {np.std(val_accs):0.4f}")


# # Learning metrics
# 
# Bellow are shown the metrics for train and validation. The variance of each metric is estimated with the k-fold runs. The enveloppe shows the $25$ and $75$ percentiles.

# In[ ]:


def plot_metrics(histories, metrics, ax, 
                 agg=lambda x: np.median(x, axis=0), 
                 env0=lambda x: np.percentile(x, 25, axis=0), 
                 env1=lambda x: np.percentile(x, 75, axis=0), 
                 process=lambda x: x, colors=None):
    x = histories[0].epoch
    for i, metric in enumerate(metrics):
        # convert metric histories for each k-fold into an (nfold, epochs) array 
        data = np.asarray([h.history[metric] for h in histories])
        data = process(data)
        ax.fill_between(x, env0(data), env1(data), 
                        color=colors[i] if colors else None, alpha=0.5)
        ax.plot(x, agg(data), color=colors[i] if colors else None, label=metric)
        
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8))

plot_metrics(histories, ['loss', 'val_loss'], ax1, colors=["green", "orange" ])
plot_metrics(histories, ['categorical_accuracy', 'val_categorical_accuracy'], 
             ax2, colors=["green", "orange" ], process=lambda x: 1-x)

ax1.set_yscale('log'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.set_ylim(1e-3, 1);
ax2.set_yscale('log'); ax2.set_ylabel("Error = 1-acc");  ax2.legend(); ax2.set_ylim(1e-3, 1e-1);
ax2.set_xlabel('Epoch')
fig.subplots_adjust(hspace=0.02)


# A noticeable fact is that the error levels are higher than the value obtained above corresponding to the *global* accuracy which is obtained by averaging the predictions over all folds.
# 
# When epochs is 30 the plateau is clearly not reached, meaning that an increase of about 0.001 in accuracy can be expected for values of epoch over 100. 
# 
# For recall, in this setup, an epoch correspond to a full training set. Hence  to reach the full potential of data augmentation we need many epochs.

# # Confusion matrix

# In[ ]:


def plot_confusion(ytrue, ypred, label=""):
    cnf_matrix = confusion_matrix(ytrue, ypred)
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cnf_matrix_norm, index=classes, columns=classes)

    plt.figure(figsize=(14, 9))
    ax = plt.axes()
    sns.heatmap(df_cm, annot=True, fmt='.4f', cmap="flag_r", ax=ax)
    ax.set_title(f"Confusion matrix for {label}")
    ax.set_ylabel("True digit")
    ax.set_xlabel("Predicted digit")
    plt.show()

plot_confusion(train_y, train_pred.argmax(axis=1), label="TRAIN set (averaged over k-folds)")
plot_confusion(train_y, val_y, label="VALIDATION set (built over k-folds)")


# Most digit are very well predicted on the **validation** set.  
# Some expected confusions are visible:
# - `0` => `6`  
# - `1` => `7`
# - `2` => `7`
# - `4` => `9`
# - `6` => `0`
# - `7` => `9`, `1`
# - `8` => `2`, `4`, `6`
# - `9` => `4`, `5`, `7`
# 
# The `9` and the `1` have the highest error rate, which is rather expected, 
# but the `1` digit is only confused with the `7`. 
# Hence the `9` is problematic because of the proximity to several other classes.
# 
# 

# # Examples of predictions

# In[ ]:


plot_digits(test_X, None, test_sel[0:60], predict=test_y, suptitle="Test digits")


# Nice!
# 
# The predictions all match the visual digit for 60 random cases.

# # Misclassified digits in the validation set
# Bellow are the first 60 missed predictions
# - green = true value
# - purple = predicted

# In[ ]:


miss = np.where(train_y != val_y)[0]
plot_digits(train_X[miss], train_y[miss], np.arange(60), predict=val_y[miss], 
            suptitle="Misclassified validation digits")
print(f"There are {len(miss)} missclassified digits")


# Curiously many of the misclassified digits shown here can be correctly classified by a human. 
# While the global accuracy is good, this means that the classifier has still room for improvement. It is a bit surprising as the accurcy is already so close to the expected maximum reachable accuracy (0.998) by deep networks.

# # Check predicted digits distribution

# In[ ]:


def countplot_ratio(df, x, y, hue, ax=None):
    # cook book : https://github.com/mwaskom/seaborn/issues/1027
    df = df[x].groupby(df[hue])        .value_counts(normalize=True)        .rename(y)        .reset_index()
    sns.barplot()
    sns.barplot(x=x, y=y, hue=hue, data=df, ax=ax)
    
        
df_count= pd.DataFrame({'digit': np.concatenate([train_y, test_y]), 
                        'data set': np.concatenate([ ["train"] * len(train_X), ["test"] * len(test_X)])})
fig, ax = plt.subplots(1, 1, figsize=(10,4))
countplot_ratio(df_count, 'digit', 'ratio', 'data set' )


# Distribution of digits is very close between test and train data sets.  
# This is a very clean result, indicating that the test split has been built so the digit ratios are the same in both sets.
# 

# # Submit predictions

# In[ ]:


df_test = pd.DataFrame({'ImageId': 1 + np.arange(len(test_X)), 
                        'Label':   test_y })
df_test.to_csv('submit.csv', index=False)


# # Conclusions
# 
# The kaggle MNIST dataset is really easy to go along with due to its nice preprocessing as seen in the exploratory section.  
# On overall this is a fun dataset to play with :-)
# 
# It behaves very well with convolutional neural networks:
# - 0.991 is reaches easily on test set with no particular adjustments other than using 5-folds of 30 epochs
# - 0.994 is attained with data augmentation by including rotation, shear, zoom and shift
# - 0.995 Next milestone is with a sandwich of Conv2D + BatchNorm + MaxPool  using 16/32/64 convs
#     - I noticed that BatchNorm behaved correctly if coupled with a MaxPool Layer, otherwise accuracy was as low as 0.1
# - 0.9965 A significant improvement comes by doubling the size of the conv layers to 32/64/128
# - 0.9970 is finally reached by pushing epochs to 200 meaning that the nb of epochs is not very significant above 30
# 
