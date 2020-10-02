#!/usr/bin/env python
# coding: utf-8

# # Detecting Credit Card Fraud with Neural Networks

# Let's see if we can predict whether or not a given credit card transacation is fraudulent using a neural network.

# ## Loading and Observing the Data

# Let's load the necessary packages and view the data file's location.

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# Let's now load the data into a pandas dataframe and take a peek at its contents.

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# Looking at the above data description, we can see that the 'Time' feature has an extremely large standard deviation, and viewing some of the dataset above, it seems to be different for almost every data instance. Since training a machine learning model on this feature will likely lead to overfitting it,  we shall drop 'Time' from our dataset.

# In[ ]:


data.drop("Time", axis=1, inplace=True)


# There is apparently a huge class imbalance in this dataset. We can confirm this by plotting a seaborn countplot of the different class labels.

# In[ ]:


sns.countplot(data["Class"])


# As we can see from the above plot, fraudulent activities make up a very small fraction of this dataset. Let's further check to see if there are any null values in the data.

# In[ ]:


data.isnull().any().describe()


# Fortunately, the dataset is not missing any values.

# ## Training a Model

# Let's partition our dataset into a train, validation, and test set, where 90% of the data will be a part of the training set and 5% will be allocated for each of the validation and test set.

# In[ ]:


limit = int(0.9*len(data))
train = data.loc[:limit]
val_test = data.loc[limit:]
val_test.reset_index(drop=True, inplace=True)
val_test_limit = int(0.5*len(val_test))
val = val_test.loc[:val_test_limit]
test = val_test.loc[val_test_limit:]


# Let's check to see that the validation and test set include a fair amount of fraudulent activites before going any further.

# In[ ]:


print("Number of fraudulent transactions in the validation set: {}"      .format(val["Class"].value_counts()[1]))
print("Number of fraudulent transactions in the test set: {}"      .format(test["Class"].value_counts()[1]))


# Now we can focus on developing a model to accurately detect fraudulent activity. Due to the huge class imbalance in our dataset, a model that simply identifies all transactions as not being fraudulent would score high accuracy. There also would not be many fraudulent samples for the model to learn from to be able to accurately identify what a fraulent transaction is. Therefore, we should find a way to balance out the number of positive and negatives instances in our training set. This can be done by either oversampling the positive instances, or undersampling the negative instances. Undersampling the negative instances would involve reducing the number of non-fraudulent transactions until the ratio between positive and negative instances was approximately 1-to-1. Since we don't have that many data samples, I fear doing so would severely limit our model's performance, since it would have much less data to train on. We shall therefore oversample the positive instances in our training data.
# 
# To oversample the positive instances in our training set, we will add copies of them to it, but with their feature values slightly tweaked. This slight manipulation of the data is done so as to have the copies being different from their original counterparts, but not too different as to end up teaching our model false information. This tweaking will be done by multiplying each positive sample copy's feature values by a number between the uniform distribution of 0.9 and 1.1.

# In[ ]:


train_positive = train[train["Class"] == 1]
train_positive = pd.concat([train_positive] * int(len(train) / len(train_positive)), ignore_index=True)
noise = np.random.uniform(0.9, 1.1, train_positive.shape)
train_positive = train_positive.multiply(noise)
train_positive["Class"] = 1
train_extended = train.append(train_positive, ignore_index=True)
train_shuffled = train_extended.sample(frac=1, random_state=0).reset_index(drop=True)


# The ratio of positive to negative instances in our training set should now be much more balanced.

# In[ ]:


sns.countplot(train_shuffled["Class"])


# Let's now separate our train, validation, and test data into their predictors and labels.

# In[ ]:


X_train = train_shuffled.drop(labels=["Class"], axis=1)
Y_train = train_shuffled["Class"]
X_val = val.drop(labels=["Class"], axis=1)
Y_val = val["Class"]
X_test = test.drop(labels=["Class"], axis=1)
Y_test = test["Class"]


# Let's also standardize the values in our dataset so that when building a machine learning model we don't unintentionally lend more weight to some features over others. We will fit a standardizer to the training data, and transform the training, validation, and test data based on this scaler.

# In[ ]:


scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_val[X_val.columns] = scaler.transform(X_val)
X_test[X_test.columns] = scaler.transform(X_test)


# Let's now build and train a feedforward neural network to detect fraudulent activity. The neural network will contain several layers of hidden units with ReLU activation functions, and a sigmoid output unit to output the probability of a given transaction being fraudulent. The Adam optimizer will be used with an initial learning rate of 1e-4 and a binary cross-entropy loss function. The model will be trained for a maximum of 50 epochs, though the learning rate of the Adam optimizer will be reduced by a factor of 0.1 if the validation loss does not improve after 3 epochs of training. This will continue until a minimum learning rate of 1e-6 is reached. If the validation loss does not improve after 5 epochs of training, we will halt training of the neural network.

# In[ ]:


model = Sequential()
model.add(Dense(64, activation="relu", input_dim=(X_train.shape[1])))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(X_train, 
                    Y_train, 
                    epochs=50, 
                    validation_data=(X_val, Y_val), 
                    callbacks=[ReduceLROnPlateau(patience=3, verbose=1, min_lr=1e-6), 
                               EarlyStopping(patience=5, verbose=1)])


# ## Analyzing our Model

# With training ceased for our neural network, let's observe how the loss and accuracy evolved during training for both the training and validation set.

# In[ ]:


num_epochs = len(history.history["loss"])
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
axarr[0].set_xlabel("Number of Epochs")
axarr[0].set_ylabel("Loss")
sns.lineplot(x=range(1, num_epochs+1), y=history.history["loss"], label="Train", ax=axarr[0])
sns.lineplot(x=range(1, num_epochs+1), y=history.history["val_loss"], label="Validation", ax=axarr[0])
axarr[1].set_xlabel("Number of Epochs")
axarr[1].set_ylabel("Accuracy")
axarr[1].set_ylim(0, 1)
sns.lineplot(x=range(1, num_epochs+1), y=history.history["acc"], label="Train", ax=axarr[1])
sns.lineplot(x=range(1, num_epochs+1), y=history.history["val_acc"], label="Validation", ax=axarr[1])


# Viewing the loss and accuracy graphs, we can see that after some initial improvements, we seemed to reach convergence after only a few epochs. Let's view the accuracy achieved by this neural network on the test set.

# In[ ]:


test_results = model.evaluate(X_test, Y_test)
print("The model test accuracy is {}.".format(test_results[1]))


# Achieving almost 100% accuracy on the test set is of course thrilling, but one must not forget about the huge class imbalance still present in the test set. A model that simply outputted that there were no fraudulent transactions would achieve high accuracy as well. The dataset info recommends using the AUPRC as an evaluation metric instead, to gather a better understanding of how well the model truly performed. We will use the average precision score instead, which is an evaluation metric available in scikit-learn that is sometimes used as an alternative to AUPRC. The average precision score will have a value between 0 and 1, with a better model having a greater score. Let's view the test average precision score for our model.

# In[ ]:


predictions = model.predict_classes(X_test)
ap_score = average_precision_score(Y_test, predictions)
print("The model test average precision score is {}.".format(ap_score))


# As we can see, the model didn't perform quite as well as we initially thought. Let's view a confusion matrix of the predictions made by the model on the test set, to gain a better idea of what type of errors it is making.

# In[ ]:


confusion = pd.DataFrame(confusion_matrix(Y_test, predictions))
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
sns.heatmap(confusion, annot=True)
plt.yticks(rotation=0)


# Viewing the confusion matrix, we can see that the neural network detected around half of the fraudulent transactions, but misidentifed a few non-fraudulent transactions as fraudulent.

# ## Final Remarks

# Training a feedforward neural network, we were able to build a machine learning model that detected credit card fraud in around half of the guilty transactions in the test set without overly missclassifying non-fraudulent transaction as fraudulent. Spending more time fine-tuning the neural network architecture, as well as manipulating the features in the dataset, could perhaps further improve upon these results. Regardless, I hope this kernel serves as a satisfactory example for others on how to approach training a neural network for a binary classification problem with a very imbalanced dataset.
