#!/usr/bin/env python
# coding: utf-8

# # MNIST - EDA & Baseline Linear Regression Model
# 
# Greetings! In this notebook we will perform basic exploratory data analysis on the MNIST handwritten digit dataset.
# 
# Then, we will examine the performance of a baseline logistic regression model, which will serve as the the basis of comparison for future model development.
# 
# Please leave any suggestions for improvement in the comments section below.
# 
# Let's get into it!

# ### Load the Good Stuff

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import keras

params = {'legend.fontsize': 'large',
          'figure.figsize': (10, 8),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Prevent Pandas from truncating displayed dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

sns.set(style="white")

SEED = 42


# ### Load & Preview Data

# In[ ]:


# Load master copies of data - these remain pristine
train_ = pd.read_csv("../input/digit-recognizer/train.csv")
test_ = pd.read_csv("../input/digit-recognizer/test.csv")
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

# Take copies of the master dataframes
train = train_.copy()
test = test_.copy()


# In[ ]:


train.shape, test.shape


# Previewing the training data as a dataframe, we can see that there are 785 features: 784 pixel values and 1 label. The 784 pixel values correspond to 28x28 px images that have been "flattened" from 2D into 1D. The label is our target value, i.e. the correct digit value represented by the flattened vector.

# In[ ]:


train.head()


# In[ ]:


# Separate the target variable from the digits
y = train.pop("label")


# Obviously, viewing images as vectors of numbers is not very helpful. Instead, we can reverse engineer our recgonizable MNIST digits.

# In[ ]:


n_preview = 10
fig, ax = plt.subplots()

for i in range(n_preview):
    plt.subplot(2, 5, i+1)
    image = train.iloc[i].values.reshape((28,28))
    plt.imshow(image, cmap="Greys")
    plt.axis("off")

plt.suptitle("The First 10 MNIST Handwritten Digits", y=0.9)
plt.show()


# Likewise, we can look at the corresponding true labels for the first 10 digits.

# In[ ]:


y[0:10].values


# ### Explore the Target Distribution

# The target variable is reasonably well-distributed between the 10 digits. This is good news - we won't have to deal with an imbalanced dataset.

# In[ ]:


digit_frequency = y.value_counts(normalize=True).to_frame()
unique_digits= np.sort(y.unique())
unique_digits_str = [str(d) for d in unique_digits]

plt.bar(digit_frequency.index, digit_frequency["label"].values)
plt.title("Training Digit Frequency")
plt.xticks(unique_digits, unique_digits_str)
plt.show()


# ### Preprocessing
# 
# We need to scale pixel values to range from 0 to 1. This prevents some data points from having a disproportionate impact on our model.

# In[ ]:


train = train / np.max(np.max(train))
test = test / np.max(np.max(test))


# ### Baseline Logistic Regression Model
# 
# Let's compare two methods of establishing a baseline model: `train_test_split` and `cross_validate`.
# 
# First, we'll split the data into training and validation sets. Then a Logistic Regression model is fit on the training set and scored on the validation set.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2, random_state=SEED)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[ ]:


lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)


# In[ ]:


yhat = lr.predict(X_valid)
score = lr.score(X_valid, y_valid)
print("Baseline score: {:.1%}".format(score))


# So our basic Logistic Regression model scores in the low 90s! Where is it going wrong?
# 
# When we create a confusion matrix of actual vs. predicted labels, we see that the model's errors are reasonable. For example, `3` is often mis-classified as `8`. Likewise, there is some confusion between `7` and `9`.

# In[ ]:


c_matrix = confusion_matrix(y_valid, yhat, normalize="true")

plt.figure()
sns.heatmap(c_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = "Spectral_r")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.title("Accuracy Score: {:.1%}".format(score), size = 15)
plt.show()


# **But**, before we get too comfortable with our 91.9% accuracy, let's check to see how the target distributions vary between the training and validation sets.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,6))

sns.countplot(y_train, ax=ax[0])
sns.countplot(y_valid, ax=ax[1])
ax[0].set_title("Training Labels")
ax[1].set_title("Validation Labels")
plt.show()


# The validation set appears to have disproportionately fewer `5`s. Interesting.
# 
# Let's see if performing cross validation gives us better results on the validation set.

# In[ ]:


lr = LogisticRegression(max_iter=1000)

cv_results = cross_validate(lr, train, y, cv=5, return_train_score=True)
cv_results.keys()


# In[ ]:


print("Train: {}, Validation: {}".format(cv_results["train_score"].mean(), cv_results["test_score"].mean()))


# Appears that the model does indeed score in the low 90s, regardless of the particular split that we use for the data.

# ### Previewing Misclassified Digits
# 
# I think we can cut the model some slack! Some of these digits would be challenging to identify, even for a human.

# In[ ]:


misclassified = []
for i, (pred, actual) in enumerate(zip(yhat, y_valid)):
    if pred != actual:
        misclassified.append(i)


# In[ ]:


n_preview = 15
samples = X_valid.iloc[misclassified]
fig, ax = plt.subplots(figsize=(18,10))

for i in range(n_preview):
    plt.subplot(3, 5, i+1)
    image = samples.iloc[i].values.reshape((28,28))
    plt.imshow(image, cmap="Greys")
    plt.title("Predicted: {} | Actual: {}".format(yhat[misclassified[i]], y_valid.values[misclassified[i]]))
    plt.axis("off")

plt.suptitle("10 Misclassified Digits: Predicted & Actual Labels", y=0.99)
plt.show()


# ### Generate Submission
# 
# Just in case you choose to submit baseline results to see how they stack up relative to other entrants on the leaderboard (answer: not very well!).

# In[ ]:


lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

preds = lr.predict(test)


# In[ ]:


sample_submission["Label"] = preds
# sample_submission.to_csv("baseline-logistic-regression.csv", index=False)
sample_submission.head()


# How is the predicted target variable distributed?

# In[ ]:


sns.countplot(preds)
plt.title("Count of Predicted Digits")
plt.show()


# ### Conclusion
# 
# Thanks very much for reading. I hope this notebook is a helpful starting point that enables you to perform your own analyses on the MNIST dataset.
# 
# Stay tuned - in a follow up notebook I will experiment with various neural network configurations to see how they compare with the baseline logistic regression model.
# 
# Until next time, happy coding :)
