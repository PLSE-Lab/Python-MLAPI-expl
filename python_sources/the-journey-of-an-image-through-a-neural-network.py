#!/usr/bin/env python
# coding: utf-8

# # The Journey of an Image Through a Neural Network
# 
# ***
# **Update 21/7/18:** Added visualisation function using UMAP instead of TSNE, which is much faster. Documentation for UMAP can be found here: https://github.com/lmcinnes/umap
# ***
# 
# This is my first time playing with convolutional neural networks, tensorflow and keras, and the beauty of these packages is how quickly you can produce a network that gives very good results, especially on something like the MNIST dataset. However, they hide almost all the details of what they're doing in the background, and I wanted to get an idea of what's really happening behind the scenes.
# 
# I spent quite some time trying different approaches to look in to what the networks I created were doing, and the few I thought were most useful are shown in this notebook. The main three are plotting the confusion matrix, visualising the network's ability to separate image classes after each layer using TSNE, and showing how the representation of the image changes after each network layer. The notebook starts with the [class I have written](#classcode) with various functions to produce these network visualisations. But it has several hundred lines of code and to begin with I'd suggest you jump straight to [where I begin putting the class in action](#simplenetwork). For me the aim of this notebook is to understand the networks, rather than the details of the code or the accuracy they achieve.
# 
# If you do want to look at the class code I have added a lot of documentation/comments to it, so hopefully it's understandable. I've also tried to make the class flexible enough to work with different networks, or even different datasets (though I haven't tested that - in particular it's likely non-numeric image classes would give errors). It would be great if people fork the notebook and use the class themselves. Even better if you publicly post what you've done with it and let me know so I can have a look! Also, if you have suggestions for things I could add I'd be interested to hear about those too.
# 
# ***
# **NOTE:** Descriptions in the text below may not exactly match the images you see, as I have not fixed random seeds so the output can be slightly different each time.
# ***
# 
# **Sections of this notebook:**
# * [Modules I have used](#modules)
# * [NetworkVisualiser Class](#classcode)
# * [Visualising a simple network with one dense layer](#simplenetwork)
# * [Visualising a network with one convolutional layer](#oneconvol)
# * [Visualising a network with multiple convolutional layers](#multiconvol)
# 
# **Links that may be useful:**
# * [Kaggle Deep Learning Track](https://www.kaggle.com/learn/deep-learning): The neural networks I define generally use the same packages and have the same type of infrastructure from these courses, and these work very well on the MNIST dataset.
# * [A Guide to TF Layers: Building a Convolutional Neural Network](https://www.tensorflow.org/tutorials/layers): A guide in the TensorFlow documentation using the MNIST data. It doesn't use keras as I do here, but it gave me some ideas about combinations of layers I could try, including MaxPool2D layers.
# * [Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) by [Yassine Ghouzam](https://www.kaggle.com/yassineghouzam): An excellent and popular kernel that shows how you can get very high accuracies using similar networks to the ones I define here, with a few extra tips and tricks to improve the results.
# * [sklearn documentation on manifold learning](http://scikit-learn.org/stable/modules/manifold.html), which includes the TSNE technique I use here.
# * [sklearn documentation on component analysis](http://scikit-learn.org/stable/modules/decomposition.html).

# <a id='modules'></a>
# ## Module Imports

# In[ ]:


# general data and misc libraries
import pandas as pd
import numpy as np
from math import ceil

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# tensorflow/keras for cnn training
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

# sklearn component analysis and utilities
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# umap dimensionality reduction
#https://github.com/lmcinnes/umap
import umap


# <a id='classcode'></a>
# ## NetworkVisualiser class
# 
# Click the button on the right to see the class code --------->

# In[ ]:


class NetworkVisualiser:
    '''Utility to visualise the progression of an image through the layers of a convolutional neural network. 
    Networks are created with tensorflow.python.keras. 
    
    layers: A list of tensorflow.python.keras.layers from which the neural network is created.
    The layers are added to a Sequential model, and the last (output) layer should be a Dense layer
    with num_classes nodes. It is highly recommended to set the name parameter of each layer
    to a short meaningful value to aid the interpretation of the visualisations created.
    
    data_file: Path to file containing image data. Each image must be represented by a flattened
    row of pixel values in the file.
        
    label_col: Column in data_file to use for image class labels.
    
    num_classes: Number of different image classes contained in the data.
    
    img_rows: Height of each image in pixels.
    
    img_cols: Width of each image in pixels.'''
    
    def __init__(self, layers, data_file='../input/train.csv',label_col='label',
                 num_classes=10,img_rows=28,img_cols=28):
        
        # definition of network infrastructure
        self.layers = layers
        # no. layers in defined network
        self.n_layers = len(layers)

        # no. of classes to identify in images
        self.num_classes = num_classes
        # height of images in pixels 
        self.img_rows = img_rows
        # width of images in pixels
        self.img_cols = img_cols
        
        # load data file
        self.load_data(data_file, label_col)
                
    def load_data(self, data_file, label_col):
        '''Load data from file, separate images and class labels,
        reshape the images, and create class vectors. Called automatically
        during class initialisation.
        
        data_file: Path to file containing image data. Each image must be
        represented by a flattened row of pixel values in the file.
        
        label_col: Data column to use for image class labels.'''
        
        print('Loading data...')
        df = pd.read_csv(data_file)
        print('Shape of data file:',df.shape)

        # get data excluding label column
        X = df.drop(label_col,axis=1)
       
        # reconstruct images from flattened rows
        X = X.values.reshape(len(X),self.img_rows,self.img_cols,1)
        
        # normalise X to lie between 0 and 1
        X = X/X.max()
       
        self.X = X
        print('Shape of network input:',X.shape)

        # extract true label of each image
        self.labels = df[label_col].values
        
        # convert labels in to dummy vectors
        y = keras.utils.to_categorical(self.labels, self.num_classes)
        self.y = y
        print('Shape of label vectors:',y.shape)
        print('First label vector:',y[0])
        
    def show_images(self, images_per_class=10):
        '''Display example images from each class.
        
        images_per_class: defines how many images will be displayed
        for each class.'''
        plt.figure(figsize=(images_per_class,self.num_classes))

        for i in range(self.num_classes):
            # select images in class i
            tmp = self.X[self.y[:,i]==1]

            # display 1st 10 images in class i
            for j in range(10):
                plt.subplot(self.num_classes,10,(10*i)+(j+1))
                plt.imshow(tmp[j][:,:,0])

                # use the same colour range for each image
                plt.clim(0,1)
                # don't show axes
                plt.axis('off')
                
    def fit(self,
            loss=keras.losses.categorical_crossentropy,
            optimizer='adam', metrics=['accuracy'],
            epochs=3, batch_size=100, validation_split=0.2):
        '''Creates a network using the layers defined in self.layers,
        fits the network, and then calls self.set_layer_outputs.
        
        Arguments are passed to tensorflow.python.keras.models.Sequential.compile
        and tensorflow.python.keras.models.Sequential.fit.'''
        
        # Buld model
        self.model = Sequential()

        # add each defined layer to the model
        for layer in self.layers:
            self.model.add(layer)
            
        # get layer names
        self.layer_names = [self.model.layers[i].name for i in range(self.n_layers)]

        # set model optimisation parameters
        self.model.compile(loss=loss,optimizer=optimizer,
                           metrics=metrics)

        # Fit model
        self.model.fit(self.X, self.y,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split = validation_split)
        
        # calculate output at each network layer
        self.set_layer_outputs()

    def set_layer_outputs(self): 
        '''Calculates the representation of each image in each layer of the network.'''
        
        print('Calculating output at each layer...')
        # maximum no. filters in any convolutional layer, used to define figure sizes later
        # initialise at 9 (min fig size), overwrite in loop if a layer has more than 9
        self.max_filters = 9

        self.layer_out = []
        prev_layer_out = None

        for key in self.layer_names:
            # create a model consisting of only the current layer
            layer_model = Sequential()
            layer_model.add(self.model.get_layer(key))

            # if this is the first layer, predict using the input images
            if prev_layer_out is None:
                curr_layer_out = layer_model.predict(self.X)
                #self.layer_out.append(layer_model.predict(self.X))
            # otherwise, predict using output of previous layer
            else:
                curr_layer_out = layer_model.predict(prev_layer_out)

            self.layer_out.append(curr_layer_out)
            
            # check if this is a convolution layer with more filters
            # than the current maximum.
            if curr_layer_out.ndim==4:
                n_filters = curr_layer_out.shape[-1]
                if n_filters>self.max_filters:
                    self.max_filters = n_filters

            prev_layer_out = curr_layer_out

            print(key, 'layer has shape',curr_layer_out.shape)
            
        # probability network assigns to correct label for each image
        # (self.layer_out[-1] is output layer)
        self.true_prob = pd.Series([self.layer_out[-1][i,self.labels[i]] for i in range(len(self.labels))])

        
    def show_conf_matrix(self):
        '''Calculate and display the network confusion matrix.'''
        # calculate confusion matrix
        cm=confusion_matrix(self.labels,
                            self.layer_out[-1].argmax(axis=1))

        # convert to percentages
        cm=100*cm/cm.sum(axis=1)

        # display
        plt.figure(figsize=(10,8))
        sns.heatmap(cm,annot=True,square=True,cmap=plt.rcParams['image.cmap'],fmt='.1f',
                    vmax=2*max([np.triu(cm,k=1).max(),np.tril(cm,k=-1).max()]))
        # vmax: used to limit range of colour bar to highlight incorrect predictions,
        # rather than letting diagonal dominate. Current limit is double the maximum
        # incorrect prediction prediction for any number (2*max in upper or lower triangle).
        plt.ylabel('TRUE')
        plt.xlabel('PREDICTED')
        
    def umap_layers(self,img_ids):
        '''Use umap to produce a 2D representation of the segmentation
        of image classes in each layer of the network. 
        
        img_ids: Indices of images to use to calculate and display results.
        Choosing ~1000 images typically
        gives good results.'''
        
        # umap to see distinction between digits at different layers
        print('Calculating UMAP representations...')

        # make figure
        rows = ceil(np.sqrt(self.n_layers+1))
        cols = ceil((self.n_layers+1)/rows)
        plt.figure(figsize=(5*cols,4*rows))

        # visualise structure of input 
        i=1
        print('Input...',end='')
        # flatten data
        pipe_in = self.X[img_ids].reshape([len(img_ids),-1])
        # transform data
        pipe_out = umap.UMAP().fit_transform(pipe_in)
        # plot data, with points coloured by class
        plt.subplot(rows,cols,i)
        plt.scatter(pipe_out[:,0],pipe_out[:,1],c=self.labels[img_ids],cmap='tab10',s=2)
        plt.xticks([]), plt.yticks([])
        plt.colorbar()
        plt.title('Input')
        i+=1

        # visualise structure of network layers
        for id_layer in range(self.n_layers):
            key = self.layer_names[id_layer]
            out = self.layer_out[id_layer]
            print(key+'...',end='')
            
            pipe_in = out[img_ids].reshape([len(img_ids),-1])
            pipe_out = umap.UMAP().fit_transform(pipe_in)

            plt.subplot(rows,cols,i)
            plt.scatter(pipe_out[:,0],pipe_out[:,1],c=self.labels[img_ids],cmap='tab10',s=2)
            plt.xticks([]), plt.yticks([])
            plt.colorbar()
            plt.title(key)
            i+=1
        print('Done!')

    def tsne_layers(self,img_ids):
        '''Use sklearn.manifold.TSNE to produce a 2D representation of the segmentation
        of image classes in each layer of the network. 
        
        As TSNE is computationally expensive and impractical to use for internal layers
        that may have thousands of features, the top self.num_classes components in the
        output of each layer are first extracted using sklearn.decomposition import PCA.
        TSNE is then run on the PCA components.
        
        img_ids: Indices of images to use to calculate and display the TSNE results.
        Strongly recommended not to use all images, which will likely fail to compute
        and in any case would produce messy plots. Choosing ~1000 images typically
        gives good results.'''
        
        # t-sne to see distinction between digits at different layers
        print('Calculating TSNE representations...')
        # TSNE too computationally expensive to run on data with many features.
        # First use PCA to extract the first num_classes components, then follow with TSNE.
        pipe = Pipeline([('pca',PCA(n_components=self.num_classes)),('tsne',TSNE())])

        # make figure
        rows = ceil(np.sqrt(self.n_layers+1))
        cols = ceil((self.n_layers+1)/rows)
        plt.figure(figsize=(5*cols,4*rows))

        # visualise structure of input 
        i=1
        print('Input...',end='')
        # flatten data
        pipe_in = self.X[img_ids].reshape([len(img_ids),-1])
        # transform data
        pipe_out = pipe.fit_transform(pipe_in)
        # plot data, with points coloured by class
        plt.subplot(rows,cols,i)
        plt.scatter(pipe_out[:,0],pipe_out[:,1],c=self.labels[img_ids],cmap='tab10',s=2)
        plt.xticks([]), plt.yticks([])
        plt.colorbar()
        plt.title('Input')
        i+=1

        # visualise structure of network layers
        for id_layer in range(self.n_layers):
            key = self.layer_names[id_layer]
            out = self.layer_out[id_layer]
            print(key+'...',end='')
            
            pipe_in = out[img_ids].reshape([len(img_ids),-1])
            pipe_out = pipe.fit_transform(pipe_in)

            plt.subplot(rows,cols,i)
            plt.scatter(pipe_out[:,0],pipe_out[:,1],c=self.labels[img_ids],cmap='tab10',s=2)
            plt.xticks([]), plt.yticks([])
            plt.colorbar()
            plt.title(key)
            i+=1
        print('Done!')
    
        
    def visualise_network(self,img_id):
        '''Visualise one image in all layers of the neural network.
        
        img_id: The index of the image to be displayed.'''
        
        # no. rows in figure (including extra row for input layer)
        nrows = self.n_layers+1
        # no. additional columns for text labels
        txtwidth = max([int(self.max_filters/12),2])
        ncols = self.max_filters+txtwidth
        # text box/font style
        props = dict(boxstyle='round', facecolor='w')
        fontsize = 1.3*ncols
        # create figure
        plt.figure(figsize=(2*ncols,2*nrows))

        ####################
        # input
        ####################

        # title
        row = 1
        ax = plt.subplot2grid((nrows, ncols), (0, 0), colspan=txtwidth)
        plt.plot([0,0],[1,1])
        ax.text(0, 1, 'Input', verticalalignment='center',fontsize=fontsize,bbox=props)
        plt.axis('off')

        # show input image
        plt.subplot2grid((nrows, ncols), (0, int((ncols-txtwidth)/2 + 0.5*txtwidth)), colspan=txtwidth)
        plt.imshow(self.X[img_id][:,:,0])
        plt.clim(0,1)
        plt.axis('off')

        # true label of input image
        ax = plt.subplot2grid((nrows, ncols), (0, txtwidth), colspan=txtwidth)
        plt.plot([0,0],[1,1])
        ax.text(0,1,
                'True label: '+str(self.y[img_id,:].argmax()),
                verticalalignment='center',horizontalalignment='center',fontsize=fontsize,bbox=props)
        plt.axis('off')

        # predicted label for input image
        ax = plt.subplot2grid((nrows, ncols), (0, ncols-txtwidth-1), colspan=txtwidth)
        plt.plot([0,0],[1,1])
        ax.text(0,1,
                'Predicted: '+
                str(self.layer_out[-1][img_id,:].argmax())+
                ' ({:.1f}% probability)'.format(100*self.layer_out[-1][img_id,:].max()),
                verticalalignment='center',horizontalalignment='center',
                fontsize=fontsize,bbox=props)
        plt.axis('off')

        #######################
        # remaining layers
        #######################

        for id_layer in range(self.n_layers):
            key = self.layer_names[id_layer]
            out = self.layer_out[id_layer]

            row += 1

            # layer title
            ax = plt.subplot2grid((nrows, ncols), (row-1, 0), colspan=txtwidth)
            plt.plot([0,0],[1,1])
            ax.text(0, 1, key, verticalalignment='center',fontsize=fontsize,bbox=props)
            plt.axis('off')

            # annotated heatmap for output layer
            if id_layer is self.n_layers-1:
                plt.subplot2grid((nrows, ncols), (row-1, txtwidth), colspan=ncols-txtwidth)
                sns.heatmap(out[img_id,:].reshape([1,-1])*100,
                            annot=True,annot_kws={"size": fontsize},fmt='.1f',
                            cmap=plt.rcParams['image.cmap'],cbar=False)
                plt.yticks([])
                plt.xticks(fontsize=fontsize)

            # plot image representation for convolutional layers (4 dimensions: image, row, column, filter)
            elif out.ndim==4:
                # no. filters in layer
                n_filters = out.shape[-1]
                # no. subplots each filter spans in this layer
                nsub_per_filt = self.max_filters/n_filters
                # plot each filter
                for i in range(n_filters):
                    plt.subplot2grid((nrows, ncols), (row-1, txtwidth+int(i*nsub_per_filt)), colspan=int(nsub_per_filt))
                    plt.imshow(out[img_id,:,:,i])
                    plt.axis('off')

            # 1d image plot for flattened layers (e.g. dense)
            else:
                n_filters = out.shape[-1]

                plt.subplot2grid((nrows, ncols), (row-1, txtwidth), colspan=ncols-txtwidth)
                plt.imshow(out[img_id,:].reshape(1,n_filters),aspect=max([n_filters/20,1]))
                plt.axis('off')

    def visualise_classbest(self):
        '''Visualise the image in each class that the network predicts most accurately,
        i.e. the image assigned the highest probability matching the true image label in each class.'''
        
        for i in range(self.num_classes):
            self.visualise_network(self.true_prob[self.labels==i].sort_values(ascending=False).index[0])   

            
    def visualise_classworst(self):
        '''Visualise the image in each class that the network predicts least accurately,
        i.e. the image assigned the lowest probability matching the true image label in each class.'''
        
        for i in range(self.num_classes):
            self.visualise_network(self.true_prob[self.labels==i].sort_values().index[0])   

    def visualise_best(self, n_plots=5):
        '''Visualise the images that are most accurately predicted by the model
        (irrespective of class), i.e. the images with the largest probability
        assigned to the true image label.
        
        n_plots: Number of images to visualise.'''
        wrong_pred = self.true_prob.sort_values(ascending=False).index[:n_plots]

        for img_id in wrong_pred:
            self.visualise_network(img_id)

            
    def visualise_worst(self,n_plots=5):
        '''Visualise the images that are least accurately predicted by the model
        (irrespective of class), i.e. the images with the lowest probability
        assigned to the true image label.
        
        n_plots: Number of images to visualise.'''
        
        wrong_pred = self.true_prob.sort_values().index[:n_plots]

        for img_id in wrong_pred:
            self.visualise_network(img_id)

    def visualise_unsure(self,n_plots=5):
        '''Visualise the images that the network is least confident about which
        class the image should be assigned to, i.e. the images which have the
        smallest probabilities assigned to the predicted labels.
        
        n_plots: Number of images to visualise.'''

        # visualise predictions network is least sure about - smallest max probability
        worst_pred = self.layer_out[-1].max(axis=1).argsort()[:n_plots]

        for img_id in worst_pred:
            self.visualise_network(img_id)


# <a id='simplenetwork'></a>
# ## Starting Simple: A one layer network
# 
# ### Building the Network
# 
# In the MNIST dataset we have 28 x 28 pixel (785 pixels per image total) greyscale images of digits. Our goal is to predict which single digit is shown in each image using a neural network and I will start with the simplest one possible, a network with no hidden layers. 
# 
# In this network, the image is flattened in to a vector (with shape 1 x 785) and this vector is passed directly to the network's output layer, which is a dense layer with ten nodes (one for each digit). A "dense" layer is one in which all the outputs from the previous layer contribute to the input of each node in the current layer. In our simple network this means the output node for each digit calculates a weighted sum of all 785 pixel values, plus a further (image independent) bias term. The network therefore has 786 x 10 = 7860 parameters in total. Our predicted digit is the one whose output node has the largest value, and I use the "softmax" function so the sum of all node values is equal to 1.
# 
# This network looks something like this:
# 
# ![Image](https://raw.githubusercontent.com/jackroberts89/kaggle-digit-recognizer/master/simplenetwork.png)
# 
# This basic infrastructure in practice will work more or less like logistic regression, and doesn't take advantage of the strengths of neural networks (let alone convoluted neural networks). But it's a good place to start and gives a baseline to compare with more complex networks.
# 
# My `NetworkVisualiser` class requires the network infrastructure to be given as a list of keras layers, in this case one `Flatten` layer and the `Dense` output layer, defined as below. Note that I make use of the `name` argument for each layer - these strings will be used as plot titles by the `NetworkVisualiser` class.

# In[ ]:


num_classes = 10

layers = [
          Flatten(name='Flatten'),
          Dense(num_classes, activation='softmax',
                          name='Output')
        ]


# As I've setup the `NetworkVisualiser` class with default values appropriate for the MNIST dataset, all I need to specify when initialising the `NetworkVisualiser` object is the list of layers. In the background the class loads the `train.csv` datafile, and it then prints some basic information about the data as well as converting the image labels (true digit values) in to 1 x 10 vectors as required for the tensorflow/keras model.

# In[ ]:


nvis = NetworkVisualiser(layers)


# With the `nvis` object created, let's make use of the `NetworkVisualiser.show_images()` function to see what some of the MNIST images look like:

# In[ ]:


# set default colour map to use
plt.rcParams['image.cmap'] = 'Reds'

nvis.show_images()


# The function (by default) displays the first ten images for each digit. It's already possible to see areas where our networks, particularly this first simple one, might struggle, for example:
# * Some digits are written in multiple styles, such as 2, 4 and 7.
# * Some people's handwriting is really bad! E.g. in the row of twos one looks more like a fly than a two, and another looks more like a snake. One of the eights appears to have had its head chopped off, and one of the sixes seems to be asleep.

# ### Fitting the Network
# 
# Ok let's get to work and fit the network by calling `nvis.fit()`. Again sensible default values to use for MNIST are defined in the class so I call it without arguments (but I could, for example, specify `epochs=10` if I wanted to train the network for longer than the default of 3). As well as optimising all the model parameters (using the `tf.keras` library) the `NetworkVisualiser.fit()` performs an additional step - it calculates and saves the representation of each image in each layer of the network. These values are then used when creating visualisations later.

# In[ ]:


nvis.fit()


# Somewhat surprisingly, this network already achieves better than 91% accuracy on the validation set!

# ### Confusion Matrix
# 
# To look in to the performance of the network in a bit more detail let's look at its confusion matrix using `nvis.show_conf_matrix()`, which uses `sklearn` to calculate the matrix and `seaborn` to make the visualisation. The confusion matrix shows how often each digit was predicted correctly by the network, and which digitis they were mistaken for when a digit was incorrectly predicted. In the plot below the annotated values are percentages.

# In[ ]:


nvis.show_conf_matrix()


# The network is most accurate at predicting 1 (98% accuracy) and least accurate at predicting 5 (86% accuracy). In general the network seems to do a better job of predicting digits with the simplest shapes, and struggle with more complex shapes. 5 is often mistaken with 8 or 3 which have similar features (the 'S' shape within 8, or the curved bottom and flat top of 3), for example.

# ### TSNE or UMAP Visualisations
# 
# TSNE is a dimensionality reduction technique, the output of which can be used to visualise structure within multi-dimensional data. Our input data has 785 dimensions (pixels), and the more complex networks will create even more, but we can use TSNE to reduce that to 2 dimensions and produce some interesting plots.
# 
# This can be done using the `NetworkVisualiser.tsne_layers` function. As TSNE is computationally intensive (and in any case a plot with 40000+ points gets messy) the function is run on a defined subset of the images - here I run it on the first 1000 images by passing in `range(1000)`. The function also uses principal component analysis (PCA) prior to TSNE to further reduce computation time.
# 
# Alternatively UMAP, a newer dimensionality reduction technique, can be used instead with the function `NetworkVisualiser.umap_layers`. The `umap_layers` function doesn't use the PCA pre-processing step, but it still much quicker than `tmap_layers`.
# 
# The output of the function is a plot representing the network's ability to separate the images in to different classes after each layer:

# In[ ]:


nvis.umap_layers(range(1000))


# Even before the images have been passed through the network (the "Input" plot above) the different digits tend to be clustered together in groups, although there is quite some overlap between them (particularly between 4, 7 and 9, for example). The "flatten" layer doesn't change the ability of the network to distinguish digits, it only reshapes the data. The plot looks different as UMAP (or TSNE) is not guaranteed to give the same output each time, but if you look closely you will see the "flatten" plot is more or less a rotation of the "input" plot.
# 
# In the output layer the clustering is much improved - the digits are grouped more tightly and most of the groups are well separated from each other. But some points are assigned to the wrong group and there are still groups that overlap, for example 4 with 9, and 3 with 5 (these were also highlighted as issues in the confusion matrix above).

# ### The Journey of an Image Through the Network
# 
# The confusion matrix and UMAP plots give an idea of how the network is performing globally, but what is actually happening to each image as it steps through the network? The `NetworkVisualiser` class has functions to visualise how an image changes in each layer. 
# 
# Let's start with `NetworkVisualiser.visualise_classbest()`, which displays each digit in the training set that the network most confidently predicts correctly. Each row of each figure corresponds to a layer of the network, as follows:
# 
# * **Input row:** Displays the original label, the correct label for that image and the predicted label (along with the probability with which the network predicted that digit).
# 
# * **Flatten row:** Shows a representation of the image after it has been reshaped from 28 x 28 in to a 1 x 785 vector. Darker shades of red correspond to larger pixel values, as in the original image.
# 
# * **Output row:** The probability that the image belongs to each digit class, according to the network.

# In[ ]:


nvis.visualise_classbest()


# With this simple network the visualisations are maybe not so interesting, but they get better later! The best predicted images are all clear examples of the most common handwritten form of each digit. I found it interesting to see how certain digits look in flattened form - for example the vertical line of a "1" produces many thin, evenly spaced bands in the flattened vector, whereas horizontal lines like the top of "7" produce fewer but thicker bands of non-zero values.

# ### Images the Network gets really wrong
# 
# The `NetworkVisualiser.visualise_classworst()` function displays the worst network predictions for each digit - the images where the network assigned close to zero probability to the correct digit.

# In[ ]:


nvis.visualise_classworst()


# In some cases it's clear why the network struggles. The example of 4 above would probably most likely be assigned as a 9 by a human, the network sees a 0, and the 4 is only really visible once you know the true label. Although the 1 appears to be well drawn, the network is probably used to the simpler style of one and mistakes it for a 2. The same probably applies for the 7. For others, such as the 8, it's probably a case of our simple network just not being up to the job.

# ### Images that confuse the network
# 
# Finally, the `NetworkVisualiser.visualise_unsure()` function displays the five (by default) images that the network was least sure about which digit to assign them to. Some of these are going to be difficult for any network (or person), like the 7 below which appears to be an upside down mirror image. Others, like the 5s, appear to be quite clear but the simple network struggles with them.

# In[ ]:


nvis.visualise_unsure()


# <a id='oneconvol'></a>
# ## Adding a Convolutional Layer to the Network
# 
# Let's convert our little network into a convolutional neural network by adding a `Conv2D` layer. Our new network looks something like this:
# 
# ![Image](https://raw.githubusercontent.com/jackroberts89/kaggle-digit-recognizer/master/oneconvolutional.png)
# 
# I've added a layer of 12 `Conv2D` filters, each with a 5 x 5 kernel. The 25 values in each kernel grid are parameters the model needs to fit, so this layer adds 25 x 12 = 300 parameters plus 12 bias terms (1 per filter). The kernels are multiplied element-wise with each 5 x 5 pixel region in the input image and then summed ([here](https://www.saama.com/blog/different-kinds-convolutional-filters/) is an explanation of this concept). The output of the convolutional layer is 12 new 'convoluted' images, each of which may highlight different features, such as horizontal or vertical lines, depending on the values in its corresponding kernel.
# 
# The convoluted images are slightly smaller than the input image, 24 x 24 instead of 28 x 28, and flattening the 12 convoluted images gives a 1 x 6912 vector (24 x 24 x 12 = 6912). Each of the final output nodes therefore has 6912 weights and one bias term, giving 69130 parameters. This network has almost 10 times more parameters than the first example.

# ### Fitting the network
# 
# Let's go ahead and create the network in the `NetworkVisualiser` class, adding the `Conv2D` layer with 12 filters to our `layers` list. The functions are the same as before, so I'll jump straight in to calling `nvis.fit()` and `nvis.show_conf_matrix()` to see some first results:

# In[ ]:


img_rows=28
img_cols=28

layers = [
          Conv2D(12, kernel_size=(5, 5), 
                      activation='relu',name='Conv',
                      input_shape=(img_rows, img_cols, 1)),
          Flatten(name='Flatten'),
          Dense(num_classes, activation='softmax',
                          name='Output')
        ]

nvis = NetworkVisualiser(layers)

nvis.fit()
nvis.show_conf_matrix()


# The accuracy on the validation set (`val_acc` after Epoch 3 above) is now better than 97%, a big improvement over the 91% we got with the simple one layer network. The ability of the network to recognise a 5 (as seen in the confusion matrix) was 86% but now has risen all the way to 99%. The digit the network struggles with most is now 9, which is often confused for a 7.

# ### UMAP Visualisation
# 
# Running `nvis.umap_layers()` with the same arguments as before, we see that in the output layer all the digits are well separated into there own clusters, whereas in the simpler network some of the clusters were overlapping. Now there are just a few images assigned to the wrong image cluster. It's more difficult to see any improvement immediately after the convolutional layer, but if you look closely it does look like the clusters are starting to separate away from each other.

# In[ ]:


nvis.umap_layers(range(1000))


# ### Visualising the Layers of the Network
# 
# Now we get to the main purpose of this exercise for me - what are the convolutional layers really doing? The output of running `nvis.visualise_classbest()` now has an extra row, `Conv`, which shows what the image looks like after being processed through each of the twelve 5x5 filters.
# 
# Each filter picks out different shapes in the input image. For example, in the visualisation of a 3 below some of the filters clearly pick out the three horizontal arms of the digit, and others pick out the near vertical line of the right-hand side of the 3 or more complex shapes.

# In[ ]:


nvis.visualise_classbest()


# ### Images the network predicts most incorrectly
# 
# It's just as interesting, and sometimes more insightful, to look at the images the network struggles with, so let's use `nvis.visualise_classworst()` to look at the examples where the network completely fails to identify the correct digit. In some cases the output of the convolutional layer makes it clear why the network makes mistakes. For example, the 'squarely' drawn 6 below is predicted to be a 5 or an 8. Many of the convoluted images indeed look more like a 5 or an 8 than they do a 6. Alternatively, the example of 8 below is badly drawn and has a 'filled in', flat top, which looks like a 5 in many of the convoluted layers and tricks the network.

# In[ ]:


nvis.visualise_classworst()


# ### Images that confuse the network
# 
# Finally, let's look at the five images for which the network is most unsure about which digit to predict (using `nvis.visualise_unsure()`. The majority of these images share something in common - the digits are rotated/drawn with a slant. This is something that could be improved by using `keras.preprocessing.image.ImageDataGenerator`, which can scale and rotate the input images whilst training the network to make it more robust to these sorts of issues. But I haven't done that here.

# In[ ]:


nvis.visualise_unsure()


# <a id='multiconvol'></a>
# ## Getting Serious: Multiple Convolutional and Dense Layers!
# 
# The last example I'll show is a network with multiple convolutional and dense layers, the kind of network that should be able to achieve good scores on the leaderboard if trained carefully. I'll also introduce a new type of layer, `MaxPool2D`, which is used to downsample an image (see [here](https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks)). 
# 
# The network layers are:
# * **Conv1:** 10 convolutional filters with a 5x5 kernel.
# * **Conv2:** 15 convolutional filters with a 5x5x10 kernel.
# * **Conv3:** 20 convolutional filters with a 3x3x15 kernel.
# * **Conv4:** 25 convolutional filters with a 3x3x20 kernel.
# * **Pool1:** MaxPool2D with size 2x2 and stride 2, which halves the width and height of the image.
# * **Flatten:** Flatten the downsampled, convoluted images into a row vector.
# * **Dense1:** Dense layer with 50 nodes.
# * **Output:** Dense output layer with 10 nodes.
# 
# Note that, with multiple convolutional layers, the output images from all filters in the previous layer are passed to each filter in the next layer. So if the first convolutional layer has 10 filters, the kernel in the next layer has a 'depth' of 10 (the kernels are 3-dimensional, with the length of the 3rd dimension being the no. of filters in the previous layer).
# 
# Testing my diagram drawing skills to the limit, this network looks something like this:
# 
# ![Image](https://raw.githubusercontent.com/jackroberts89/kaggle-digit-recognizer/master/multiconvolutional.png)
# 
# The image size in each layer, and the number of parameters that need to be fitted, are:
# 
# * **Conv1:** 250 weights + 10 bias. Image size: 24x24
# * **Conv2:** 3750 weights + 15 bias. Image size: 20x20
# * **Conv3:** 2700 weights + 20 bias. Image size: 18x18
# * **Conv4:** 4500 weights + 25 bias. Image size: 16x16
# * **Pool1:** 0 parameters. Image size: 8x8.
# * **Flatten:** 0 parameters. Image size: 1x1600.
# * **Dense1:** 80000 weights + 50 bias. Image size: 1x50.
# * **Output:** 500 weights + 10 bias. Image size: 1x10.
# 
# Overall, the number of parameters to fit increases from around 70000 to 90000 compared to the previous (one convolutional layer) model. Including the `Pool2D` layer in particular stops the number of parameters from increasing too much.

# ### Fitting the network
# 
# Let's define the new list of layers, create the `NetworkVisualiser` object, fit the network, and display the confusion matrix as before. This network takes a lot longer to train, about 2:30 minutes per epoch on my (underpowered) laptop.

# In[ ]:


# Define layers to include in the network
# recommend giving each layer a meaningful name
# output should always be dense with num_classes nodes
layers = [
          Conv2D(10, kernel_size=(5, 5), 
                          activation='relu',name='Conv1',
                          input_shape=(img_rows, img_cols, 1)),
          Conv2D(15, kernel_size=(5, 5), 
                          activation='relu',name='Conv2'),
          Conv2D(20, kernel_size=(3, 3), 
                          activation='relu',name='Conv3'),  
          Conv2D(25, kernel_size=(3, 3), 
                          activation='relu',name='Conv4'),    
          MaxPool2D(pool_size=(2, 2), strides=2,
                          name='Pool1'),
          Flatten(name='Flatten'),
          Dense(50, activation='relu',name='Dense1'),   
          Dense(num_classes, activation='softmax',
                          name='Output')
        ]

nvis = NetworkVisualiser(layers)

nvis.fit()
nvis.show_conf_matrix()


# Our new complex network improves the accuracy from 97% to above 98%. By training for more epochs, adding dropout layers, and performing transformations on the input layers it could achieve much better accuracies. The network struggles most with 9, actually getting a worse score than the previous network, but overall does better and correctly identifies six 99.9% of the time!

# ### UMAP Visualisations
# 
# With the additional convolutional layers it becomes clear that they improve the separation of the digits into clusters, whereas it wasn't so apparent in these plots before with only one layer. By the output of the 4th convolutional layer clearly defined groups are visible for each digit. The two dense layers then finish the job and tighten and separate the clusters even further.

# In[ ]:


nvis.umap_layers(range(1000))


# ### Visualising the Network
# 
# With all the extra layers we've added there's now much more going on in the plots from `nvis.visualise_classbest()`. After each convolutional layer the filters progressively focus on smaller, more specific regions of the image. After the `MaxPool2D` layer the image is barely recognisable to a human anymore in most cases, but as far as the network is concerned they are clearer than ever before.

# In[ ]:


nvis.visualise_classbest()


# ### Images the Network Gets Wrong
# 
# The images the network incorrectly identifies now all have clear issues, which in some cases would also confuse a human:
# * **0:** incomplete loop tricks the network into seeing a 6.
# * **1:** curved style confuses the network and it sees a 2.
# * **2:** poorly drawn/squashed vertically.
# * **3:** poorly drawn
# * **4:** appears to be a 7 incorrectly labelled as a 4
# * **5:** thick brush strokes make it difficult to identify true digit.
# * **6:** rotated and poorly drawn.
# * **7:** very difficult to distinguish from a 1.
# * **8:** poorly drawn
# * **9:** short tail makes it difficult to distinguish from a 0.

# In[ ]:


nvis.visualise_classworst()


# ### Images That Confuse the Network
# 
# Again, most of these difficulties are also understandable. The first image, of an 8, I'd also be more likely to recognise as a 9. The 5 appears to be clearer, but it is drawn strangely and the networks seems to pick up the clear 7 shape in the lower right corner of the image.

# In[ ]:


nvis.visualise_unsure()
plt.savefig('unsure9.png')


# ## Summary
# 
# That's all from me for now. Thanks for reading and I hope some of you find it useful! Feel free to ask any questions and like I said at the beginning I'd love to see other people playing with the `NetworkVisualiser` class.
