#!/usr/bin/env python
# coding: utf-8

# **Fraud detection (classification) using neural networks**
# 
# In this tutorial we are going to build a simple fraud classification model for credit card data. For data loading, preparation and visualization we are going to use the following modules.
# * numpy
# * pandas
# * matplotlib
# * seaborn
# 
# Let us import those modules.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns


# The first step is to load the data and gain a basic insight. Therefore we are going to use the describe method of the pandas dataframe. We are also going to print the first 5 lines using the head method.

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")

data.describe()


# In[ ]:


print(data.head())


# We are provided with Time information, 24 columns of anonymous data, the amount of the fraud and the class (1=fraud). By looking at the mean of the class we can see that we are dealing with a heavily unbalanced dataset. In order to avoid the network predicting no fraud all the time we have to take some measures. This is going to be discussed later on.

# Let us now look at correlations to find out more about our data. For that we are using the heatmap plot of searborn.

# In[ ]:


corr = data.corr()

plot.figure(figsize=(30,30))
sns.heatmap(corr, annot=True)


# This is a lot of data. We can see that mostly there are very low correlations (0.1 or less). Let us filter out only the large ones.

# In[ ]:


corr1 = corr[corr>0.1]
plot.figure(figsize=(20,20))
sns.heatmap(corr1, annot=True)


# We can now see that most of the variables are not really correlated with the class. It might be useful to perform a principal component analysis here (PCA). Therefore we are importing sklearn. We want roughly 5 components to remain.
# 
# We are dropping the solution from the data before fitting the PCA.

# In[ ]:


from sklearn.decomposition import PCA

#dropping the solution
pca_data = data.drop("Class", 1)

pca = PCA(n_components=5)
pca.fit(pca_data)

pca_data = pd.DataFrame(pca.transform(pca_data))
print(pca_data.shape)


# Next we are going to normalize our data, which helps our neural network to perform better.

# In[ ]:


means = []
stds = []

for col in range(pca_data.shape[1]):
        mn = np.mean(pca_data.iloc[:,col])
        st = np.mean(pca_data.iloc[:,col].std())
        
        #storing statistical data for later
        means.append(mn)
        stds.append(st)
        
        pca_data.iloc[:,col] = (pca_data.iloc[:,col]-mn)/st
        pca_data.iloc[:,col] = np.nan_to_num(pca_data.iloc[:,col])
        
pca_data.describe()


# We can see that the means are 0 and the standard deviations are 1 now. Perfect!
# 
# Now we have reduced the dimensionality of our data significantly. We are now splitting off data to evaluate our model later on.

# In[ ]:


test_ratio=0.2

#combining with solutions to keep order
new_data = pd.concat([pca_data, data["Class"]],1)

test_data = new_data.sample(frac=test_ratio)
train_data = new_data.drop(test_data.index)

test_sols = test_data["Class"]
test_data = test_data.drop("Class", 1)

train_sols = train_data["Class"]
train_data = train_data.drop("Class", 1)



#get dummies for better classification
train_sols = pd.get_dummies(train_sols, prefix="Class")
test_sols = pd.get_dummies(test_sols, prefix="Class")

print(new_data.shape)
print(train_data.shape)
print(test_data.shape)

print(train_sols.head())


# Perfect! Let us build the model now. We are using keras as the frontend and tensorflow as our backend. The network will be a simple dense neural network. For introducing non-linearity and improving convergence we are using the Leaky-Rectified-Linear (LeakyReLU) activation function for intermediate layers. For the last layer we are going to use softmax.

# In[ ]:


from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, LeakyReLU
from keras.callbacks import ModelCheckpoint

inp = Input(shape=(5,))
x = Dense(64)(inp)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(32)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(8)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In order to minimize the influence of the heavily unbalanced dataset we are going to calculate class weights, that will penalize wrong detection of the minority class more than of the majority class. Therefore we are calculating the inverse number of elements in each category.

# In[ ]:


weight0 = 1.0/train_sols["Class_0"].sum()
weight1 = 1.0/train_sols["Class_1"].sum()

_sum = weight0+weight1

weight0 /= _sum
weight1 /= _sum

print(weight0)
print(weight1)


# Let us now define a model checkpoint, which saves the model for the best accuracy.

# In[ ]:


callback = [ModelCheckpoint("check.h5", save_best_only=True, monitor="val_acc", verbose=0)]


# We can now finally fit our model.

# In[ ]:


model.fit(train_data, train_sols, batch_size=500, epochs=100, verbose=0, callbacks=callback, validation_split=0.2, shuffle=True, 
         class_weight={0:weight0, 1: weight1})


# Let us now load the best model of our training epochs.

# In[ ]:


best_model = load_model("check.h5")


# To evaluate our model we can use the evaluate model provided by keras.

# In[ ]:


score = model.evaluate(test_data, test_sols)
print(score[1])


# The accuracy seems to be high right? Careful! Keep in mind that we have a highly unbalanced dataset with many zeros and only a few ones! Let us check how the model performs for ones only.

# In[ ]:


test_data2 = pd.concat([test_data, test_sols],1)
test_data2 = test_data2.loc[test_data2["Class_1"]==1]

test_sols2 = test_data2[["Class_0", "Class_1"]]
test_data2 = test_data2.drop(["Class_0", "Class_1"],1)

score2 = model.evaluate(test_data2, test_sols2)
print(score2[1])


# Ha! The model obviously performs worse for these cases but it is still quite good. Neat!
# 
# Finally I would like to compare our neural network to a gradient boosting classifier. Therefore we are using the gradient boosting classifier of sklearn.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

xgmodel = GradientBoostingClassifier(n_estimators=200)

#weighting the samples
xgweight0 = train_sols["Class_0"].values*weight0
xgweight1 = train_sols["Class_1"].values*weight1

xgweights = xgweight0+xgweight1
print(xgweights.shape)
print(train_sols["Class_1"].shape)
#fit
xgmodel.fit(train_data, train_sols["Class_1"], xgweights)

xgscore = xgmodel.score(test_data, test_sols["Class_1"])
xgscore2 = xgmodel.score(test_data2, test_sols2["Class_1"])


# In[ ]:


print("Overall score:\t\t"+str(xgscore))
print("Rare case score:\t"+str(xgscore2))


# At the first glance it looks like the gradient boosting classifier outperforms the neural network, but when having a closer look at the rare cases we see that the neural network gives much a litte more accurate predictions.
# 
# A few interesting ideas to play with:
# * How does the number of principal components affect the accuracy?
# * How does the complexity of the neural network afferct the accuracy and training speed?
# * Does binary_crossentropy and only one column for the class perform also that well?
# * Can some kind of feature engineering improve the performance?
# * How does the number of estimators influence the performance of the gradient boosting classifier?
# * ...
# 
# You see that there are still many open questions one could try to answer in order to optimize the model. I am for now happy with the current model.
