#!/usr/bin/env python
# coding: utf-8

# # Call Me By Your Number: An Exercise in Machine Learning with Python
# 
# ## Contents
# 1. [Overview](#overview)
# 2. [A Look at the Data](#look)
# 3. [Visualizing the Data](#heatmap)
# 4. [Dimensionality Reduction](#dim)
#     - [LDA](#lda)
#     - [PCA](#pca)
#     - [PCA with K-Means](#kmeans)
# 5. [Classification with a Neural Network](#keras)
# 6. [Modeling](#model)
# 7. [Results](#results)
# 8. [Summary](#summary)
# 
# # Overview <a name = 'overview'></a>
# 
# The primary purpose of this notebook is to explore dimensionality reduction techniques and neural networks for classifying images of handwritten letters from the MNIST dataset. Some helpful notebooks I found were:
# 
# * [Interactive Intro to Dimensionality Reduction](https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction) by **Anisotropic** (code sometimes doesn't run, but still a good resource)
# * [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6#2.-Data-preparation) by **Yassine Ghouzam, PhD**
# 
# Note: the scatter plots in this notebook are interactive, so feel free to move things around!
# 
# # A Look at the Data <a name = 'look'></a>
# MNIST is a collection of 28x28 pixel images of handwritten numbers. Numerically, they are represented by 28x28 matrices with each entry each falling on a scale from 0-255&mdash;rating their intensity&mdash;with 0 being nothing (or total black), and 255 being the maximum (or pure white).

# In[ ]:


import pandas as pd
import numpy as np

# for reproducibility
import os
os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(5)

# options
pd.set_option('display.max_columns', 28)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape


# We see the sets are fairly large at 42,000 and 28,000 in training and test, respectively.

# In[ ]:


train.isnull().sum().sum(), test.isnull().sum().sum()


# In[ ]:


train.values.min(), train.values.max(), test.values.min(), test.values.max()


# There is nothing missing, which is very nice. We also see there are no entries outside of the established 0-255 range, so we don't need to fix anything at all.

# # Visualizing the Data <a name = 'heatmap'></a>
# Seeing (hah!) as these rows represent images, we should be able to look at them. Let's plot 10 random entries from the training set.

# In[ ]:


#heatmap
to_heatmap = []
for i in train.sample(n = 10, random_state = 5).index:
    new_map = pd.DataFrame(np.array(train.drop(columns = 'label').iloc[i])
              .reshape(28, 28)[::-1])
    to_heatmap.append(new_map)


# In[ ]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=5)

for i in range(len(to_heatmap)):
    if i < 5:
        fig.add_trace(go.Heatmap(z = to_heatmap[i],
                                 colorscale = 'Greys',
                                 showscale = False),
                             row = 1,
                             col = i+1)
    else:
        fig.add_trace(go.Heatmap(z = to_heatmap[i],
                                 colorscale = 'Greys',
                                 showscale = False),
                             row = 2,
                             col = i-4)
fig.update_layout(title_text = 'Figure 1: Random Training Numbers')
fig.show()


# Yep, those are numbers.

# In[ ]:


import plotly.express as px

plot_df = pd.concat([train['label'],train.drop(columns = 'label')],
                     axis = 1)
plot_df = plot_df.sort_values('label', axis = 0)
plot_df['label'] = plot_df['label'].transform(lambda x: x.astype(object))
fig = px.scatter_3d(plot_df, 
                 x = 'pixel387', 
                 y = 'pixel397',
                 z = 'pixel402',
                 color = 'label',
                 title = 'Figure 2: Plot of Class Groupings by Pixels')
fig.show()


# We see that the labels are fairly mixed up at least as far as one pixel relates to another, but there appears to be a semblance of clustering in the 0s and 4s in this particular visualization. Of course, we can't get a full view of the 784 dimensional space.

# # Dimensionality Reduction <a name = 'dim'></a>

# ### Linear Discriminant Analysis <a name = 'lda'></a>
# 
# Linear Discriminant Analysis is a supervised learning method, meaning it is provided with class labels in order to execute its covariance based algorithms. This enables it to perform quite well in dimensionality reduction. Essentially, LDA projects the data onto vectors (linear discriminants) which maximize the separability between the resulting scalars for each class ([here](http://courses.cs.tamu.edu/rgutier/cs790_w02/l6.pdf) are some excellent slides on the derivation of the formula). The result is a matrix of size (*n* rows, *p-1* columns), where *p* is the number of classes. LDA is also a method of classification.

# In[ ]:


# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

response = train['label']

train_predictors, test_predictors, train_response, test_response = train_test_split(train.drop(columns = 'label'),
                                                                response,
                                                                train_size = 0.7,
                                                                shuffle = True)


# In[ ]:


# reduce
lda = LinearDiscriminantAnalysis()
train_lda = lda.fit_transform(train_predictors, train_response)
test_lda = lda.transform(test_predictors)


# In[ ]:


np.cumsum(pd.Series(lda.explained_variance_ratio_))[np.cumsum(lda.explained_variance_ratio_) > 0.95]


# We see that at the 7th index (8th discriminant) more than 97% of the variance is explained, and a fully 100% at the 9th and final discrimiant. This means we will only be working with ***nine*** features. That is a huge improvement in computational time from the original 784.

# In[ ]:


lda_df = pd.concat([train_response.reset_index(), 
                    pd.DataFrame(train_lda)], 
                    axis = 1).drop(columns = 'index')
lda_df = lda_df.sort_values('label', axis = 0)
labels = ['label']
for i in range(len(lda_df.columns)-1):
    labels.append(f'Linear Discriminant {i+1}')
lda_df.columns = labels
lda_df['label'] = lda_df['label'].transform(lambda x: x.astype(object))
lda_df.describe()


# We see that LDA also drastically reduces the range of each column, so further normalization will probably not be necessarily as we aren't concerned with linearity.

# In[ ]:


fig = px.scatter_3d(lda_df, 
                 x = 'Linear Discriminant 1', 
                 y = 'Linear Discriminant 2', 
                 z = 'Linear Discriminant 3',
                 color = 'label',
                 title = 'Figure 3: Training Data Projected Onto LD1, LD2, and LD3')
fig.show()


# There is pretty obvious clustering of each of the values, so this looks like a very good option.

# ### Principal Component Analysis <a name = 'pca'></a>
# Principal Component Analysis works very differently. It projects the data onto vectors that are orthogonal to each other and thus linearly independent. [Principal Component Analysis](http://www.ccs.neu.edu/home/vip/teach/MLcourse/5_features_dimensions/lecture_notes/PCA/PCA.pdf) (Li, Wang) gives excellent detail on the methodology and theory. A important difference between PCA and LDA is that PCA is unsupervised, meaning it does not take given classes into account when deriving the new set of vectors, which can significantly impact the groupings of the data.
# 
# According to Li and Wang, the data must be normalized first. We have some totally empty columns, so we have to define a new function to normalize our data to avoid NaNs.

# In[ ]:


def normalize(data):
    if np.std(data) == 0:
        return data
    else:
        return (data - np.mean(data))/np.std(data)

train_predictors = train_predictors.transform(lambda x: normalize(x))


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(random_state = 5)
train_pca = pca.fit_transform(train_predictors)


# In[ ]:


# number of entries for > 0.999% variance is explained
np.cumsum(pd.Series(pca.explained_variance_ratio_))[np.cumsum(pd.Series(pca.explained_variance_ratio_)) > 0.95][:1]


# In[ ]:


# number of entries for > 0.999% variance is explained
np.cumsum(pd.Series(pca.explained_variance_ratio_))[np.cumsum(pd.Series(pca.explained_variance_ratio_)) > 0.99][:1]


# In contrast to LDA, PCA requires 314 features to explain at least 95% of the variance, and 525 to get above 99%. It is a step down from 784, but it's not too big of one in comparison.

# In[ ]:


pca = PCA(n_components = 525,
          random_state = 5)
train_pca = pca.fit_transform(train_predictors)


# In[ ]:


pca_df = pd.concat([train_response.reset_index(), 
                    pd.DataFrame(train_pca)], 
                    axis = 1).drop(columns = 'index')
pca_df = pca_df.sort_values('label', axis = 0)


# In[ ]:


labels = ['label']
for i in range(len(pca_df.columns)-1):
    labels.append(f'Principal Component {i+1}')
pca_df.columns = labels
pca_df['label'] = pca_df['label'].transform(lambda x: x.astype(object))

fig = px.scatter_3d(pca_df, 
                 x = 'Principal Component 1', 
                 y = 'Principal Component 2', 
                 z = 'Principal Component 3',
                 color = 'label',
                 title = 'Figure 4: Training Data Projected Onto PC1, PC2, and PC3')
fig.show()


# The clustering does look ok, but perhaps not quite as tight as with LDA.

# ### PCA with K-Means Clustering <a name = 'kmeans'></a>
# K-Means Clustering generates cluster centers based on the grouping of the data and then iteratively assigns each data point as belonging to one of these clusters based on distance. As with PCA, it is unsupervised, so the cluster centers are necessarily unlabeled according to the true classes. This means there is a degree of inaccuracy in the clustering that must be taken into account. There are also some other [drawbacks of k-means clustering](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means), which are worth looking in to.

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10,
                random_state = 5)
train_kmeans = kmeans.fit_predict(train_pca)


# In[ ]:


kmeans_df = pd.concat([pd.Series(train_kmeans).rename('cluster'),
                       pd.DataFrame(train_pca)], 
                       axis = 1)
kmeans_df = kmeans_df.sort_values('cluster', axis = 0)


# In[ ]:


labels = ['cluster']
for i in range(len(kmeans_df.columns)-1):
    labels.append(f'Principal Component {i+1}')
kmeans_df.columns = labels
kmeans_df['cluster'] = kmeans_df['cluster'].transform(lambda x: x.astype(object))

fig = px.scatter_3d(kmeans_df, 
                 x = 'Principal Component 1', 
                 y = 'Principal Component 2',
                 z = 'Principal Component 3',
                 color = 'cluster',
                 title = 'Figure 5: Training Data Projected Onto PC1, PC2, and PC3 with K-Means Clustering')
fig.show()


# We see a fairly decent grouping here. As we actually have the class labels for our data, we will be sticking with LDA for reducing the dimensions.

# # Classification with a Neural Network <a name = 'keras'></a>
# 
# I decided to use a multilayer perceptron-style model as I want to employ dimensionality reduction, save some time, and, if possible, present a method which is perhaps more scalable at the expense of accuracy. A convolutional network generally performs better for image data, but requires the original, unreduced dataset. Ghouzam mentions in his "Introduction to CNN Keras" that his high-performance convolutional model took 2.5 hours to run over 2 epochs (probably on a system similar to mine). The Keras documentation includes a [model which performed at 99.25% test accuracy](https://keras.io/examples/mnist_cnn/) that took just over 3 minutes to run on a modern GPU, which I don't have.
# 
# One of the common quandaries in designing a neural net is how many layers and nodes to choose. There seem to be few real answers. Of the only definite ones, the use of linear activation functions in an MLP model is [equivalent to a 2-layer, input-output model](https://en.wikipedia.org/wiki/Multilayer_perceptron#Activation_function), so we want to use non-linear activators for a multilayer model to be sensible. As far as the number of neurons in each layer, there seems to be a lot of tinkering involved. Here are a few resources that discuss this:
# 
# * [How to choose the number of hidden layers and nodes in a feedforward neural network? (StackExchange)](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw)
# 
# * [How many hidden units should I use?](ftp://ftp.sas.com/pub/neural/FAQ3.html#A_hu)
# 
# * [Review on Methods to Fix Number of Hidden Neurons in Neural Networks (Hindawi)](https://www.hindawi.com/journals/mpe/2013/425740/)
# 
# For this model, I found it was very nearly a situation of the more, the merrier. As I'm not trying to absolutely maximize my results, but instead learn, I decided at some point that "that's pretty good" and went with it. That said, I tried many, many different configurations before stopping. [Dropout:  A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) (Srivastava et al., 2014) talks at length about some optimization techniques, of which I applied dropout and max-norm which are good methods to prevent overfitting of the data. There have been good results in using pretrained models, as well, if you can find them for your application.

# # Modeling <a name = 'model'></a>
# I chose to set aside 10% from the training data for validation in each epoch of the keras model. Normally, I would run a 10-fold cross-validation loop, but I'm exploring the capabilities of the model. With the data being shuffled before each epoch, this should give a good indicator of performance.

# In[ ]:


train_lda = lda.fit_transform(train, response)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import RMSprop
import time


# In[ ]:


callback = [EarlyStopping(monitor = 'loss',
                          min_delta = 0.001,
                          patience = 5)]
optim = RMSprop(lr = 0.05)

model = Sequential()
model.add(Dense(2048, input_dim = 9, kernel_constraint = MaxNorm(4)))
model.add(Activation('softmax'))
model.add(Dropout(0.5))
model.add(Dense(2048))
model.add(Activation('softmax'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))



model.compile(optimizer = optim,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


# In[ ]:


start_time = time.time()

history = model.fit(train_lda, 
                    response, 
                    batch_size = 200, 
                    epochs = 20,
                    verbose = 0,
                    callbacks = callback,
                    shuffle = True,
                    validation_split = 0.1)

print(f'Runtime {time.time() - start_time} seconds')


# # Results <a name = 'results'></a>

# In[ ]:


history_plot = history.history
history_plot.update({'epoch' : history.epoch})


# In[ ]:


fig = make_subplots(rows = 1,
                    cols = 1,
                    x_title = 'Epoch')

trace1 = go.Scatter(x = history_plot['epoch'], 
                    y = history_plot['acc'],
                    name = 'Training Accuracy')

trace2 = go.Scatter(x = history_plot['epoch'], 
                    y = history_plot['loss'],
                    name = 'Training Loss')

trace3 = go.Scatter(x = history_plot['epoch'], 
                    y = history_plot['val_acc'],
                    name = 'Validation Accuracy')

trace4 = go.Scatter(x = history_plot['epoch'], 
                    y = history_plot['val_loss'],
                    name = 'Validation Loss')

fig.add_traces([trace1, trace2, trace3, trace4])
fig.update_layout(title = 'Figure 6: Model Accuracy and Loss per Epoch')
fig.show()


# # Summary <a name = 'summary'></a>

# Our neural network seems to be fairly well generalized and accurate with both an in- and out-of-sample accuracy of ~91-92% (note: this varies from run to run, even without changing any parameters and setting random seeds). This won't win any competitions, but it's fast and it performs well. For images, convolutional networks appear to be the way to go. In a pinch, this one can do. I think it's important to note that while experimenting with the execution of this report, I found that this model performed about as well as a k-nearest neighbors classifier with n = 7 neighbors, but that KNN was significantly faster both in setup and execution. KNN is also a lot easier to understand and explain.
# 
# The big takeaway for me is that there are few hard and fast rules about neural networks and how to make them perform better. Mostly, they require a lot of tinkering. After adjusting the batch size, number of epochs, number of layers, number of neurons in each layer, layer activation algorithms, and whatever else I could think to based on a variety of resources, there were few discernible patterns in the results. Accuracy across these tests ranged from 9-96%. A configuration that seemed quite sensible would perform horrible, and another that was a shot in the dark would be fantastic. Even running the exact same model didn't usually garner the same results. Reproducibility is a common problem with neural networks, so saving past models is crucial.
# 
# Ultimately, neural networks are a useful predictive tool, but a great deal of care must go into their utilization.
