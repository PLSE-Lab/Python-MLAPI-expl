#!/usr/bin/env python
# coding: utf-8

# <h1> First Step:  Analyze Dataset ! </h1>
# 
# The first step to any machine learning problem is to *know thy dataset*. To this end, there are many tools available in one's aresenal to analzye as well as visualize the distribution of one's data. I will try to cover two of two of these here : 
# - **Principal Component Analysis (PCA)**
# -  ** t-Distributed Stochastic Neighbour Embedding (t-SNE) **
# 
# Both of these are ways of reducing the *dimensionality* of your data. For this challenge you have been given image data which tecnhically has 2 dimensions. (3 if you have color images). However that is not the dimensionality of the information within that image; in order to represent the content of those images (i.e the digits), so that a computer can distinguish 1 from 2 will require *features which are of a higher dimensionality* . These features could be a range of things, say the presence or absence of a closed loop as in the case of 1 and 0 or the curvature of the bottom half of the digit, as in the case of 1 and 4 or even more abstract things which wouldn't make sense to us humans. The more complex the subject in the image is, the more dimensions you will need to represent it so that a computer can *learn* what it is.
# 
# The problem nowadays is that most datasets have a large number of variables (orientation, size, color, quality of handwriting etc). In other words, they have a high number of dimensions along which the data is distributed. Visually exploring the data to identify potential problems or solutions can thus be very difficult.  But such visual exploration is incredibly important in any data-related problem. Therefore it is key to understand how to visualise/represent high-dimensional datasets. This is where data dimensionality reduction plays a big role.  PCA and t-SNE can be applied to a wide range of data - audio, image, video, numerical etc.
# 
# I will describe each technique as we go along. Let's dive in ! 
# 
# <h2> Importing the Libraries </h2>
# 
# - File I/O and Image processing will be done exclusively using Glob and OpenCV.  
# - Plotly, for those who don't know, is an excellent libary to create amazing visualizations. 
# - PCA and tSNE implementations can be found within the sk-learn library 

# In[151]:


# basic libraries required for i/o and processing
import glob
import cv2
import numpy as np

# importing plotly and setting up offline notebook
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

# importing libraries from sklearn to apply PCA and TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# <h2> Loading Data </h2>
# 
# 
# The function below is used to load the information of one or more datasets. It offers a neat and clean way to handle all the data. See the docstring and code below for usage info. 

# In[152]:


def loadLabels(databases, dataFolder = '../input'):
    
    '''
     Helper function to facilitate easy loading of data labels
     given the name of the training set. You can append data labels
     from several csv files easily by passing all of them as a list
     
     e.g : data = loadLabels(['training-a','training-b'])
    '''
    
    # using a dictionary to store filepaths and labels
    data = {}
    
    for db in databases:
        
        # data directory followed by labels file name
        labelsPath = dataFolder + '/%s.csv' % db
        labels = np.genfromtxt(labelsPath, delimiter=',', dtype=str)

        # field names in first row (row 0)
        fields = labels[0].tolist()

        # getting all rows from row 1 onwards for filename and digit column 
        fileNames = labels[1:, fields.index('filename')]
        digits = labels[1:, fields.index('digit')]

        # creating a dictionary - will come in handy later to pick and choose digits !        
        for (fname, dgt) in zip(fileNames, digits):
            data[fname] = {}
            data[fname]['path'] = dataFolder + '/%s/%s' % (db,fname)
            data[fname]['label'] = int(dgt)
    
    return data


# Now we can load as many or as few of the databases we want. Just add the names to the list **db** to load from other databases. For now, we will be looking at Training-A dataset
# 
# Since most these datasets contain 10,000+ images, we take about **n = 6000 ** random samples  from the dataset. The original images along with the labels are loaded into the lists **images** and **labels** respectively

# In[153]:


# data sets to analyze
db = ['training-a']

# loading data from csv file
print("Loading Labels for %s" % " ".join(db))
dataLabels = loadLabels(db)

# randomly selecting n samples from dataset; WARNING : large no. of samples may take a while to process
n = 6000
samples =  np.random.choice( list(dataLabels.keys()), n if n<len(dataLabels) else len(dataLabels), replace=False)

# loading selected images (as grayscale) along with labels
print("Loading Images")
images = [cv2.imread(dataLabels[fname]['path'],0) for fname in samples]
labels = [dataLabels[fname]['label'] for fname in samples]

# annotations are filenames; used to label points on scatter plot below
annots = [fname for fname in samples]


# <h2> Processing Images </h2>
# 
# Now on preliminary visual inspection, we know there are images which have differently colored backrounds. There are badly written digits, digits written at different angles sizes.  Besides finding out the original distribution of the data, we also would like to know how is the data affected by applying some processes on it (i.e whether they become more distinguishable or not). To this  end, the function defined below performs some basic operations on images given to it, namely : 
# - Blurring to remove speckle noise
# - Thresholding to make image binary (black (0) and white (255) only)
# - Converting Images to be White Text on Black Backgrounds
# - Cropping and Centering Digits
# Further processes may be added as the need aries. 

# In[154]:


def process(img, crop=True, m = 5):
    '''
    This function takes care of all pre-processing, from denoising to 
    asserting black backgrounds. If the crop flag is passed, the 
    image is cropped to its approximate bounding box (bbox). 
    The m parameter (pixels) adds a some extra witdh/height to the bbox 
    to ensure full digit is contained within it.
    '''
    # blurring - removes high freq noise
    img2 = cv2.medianBlur(img,3)

    # threshold image
    thresh, img3 = cv2.threshold(img2, 128, 255, cv2.THRESH_OTSU)
    
    # ensure backroung is black
    # if no. of white pixels is majority => bg = white, need to invert
    if len(img3[img3==255]) > len(img3[img3==0]):
        img3 = cv2.bitwise_not(img3)

    if crop:      
        # getting contours on the image  
        _,contours,_ = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # get largest contour by area
        cMax = max(contours, key=cv2.contourArea) 

        # get box that encloses the largest contour -> bounding box
        x,y,w,h = cv2.boundingRect(cMax)

        # adding margins
        x1 = (x-m) if (x-m)>0 else x
        y1 = (y-m) if (y-m)>0 else y
        
        x2 = (x+w+m) if (x+w+m)<img.shape[1] else (x+w)
        y2 = (y+h+m) if (y+h+m)<img.shape[0] else (y+h)

        # cropping
        img3 = img3[ y1:y2, x1:x2 ]

    return img3


# On top of applying these processes, we also resize the images to a fixed size of 32x32 pixels. Now this is nothing written in stone. Larger sizes may help with recognization. But for analyzing the data I found this size to be good and fast enough.
# Both processed images and original images are put through all subsequent processes in order to analyze the effect of processing the images.
# 
# In order to perform PCA, we need 1D arrays. To do this numpy provides the *flatten* method (images read by OpenCV are numpy arrays), which simply just rerranges the rows of the images into one long array . After flattening, we also perform a normalizing operation using the *StandardScaler* function from sk-learn so that they are within a fixed range of values and zero centered

# In[155]:


# processing images - cleaning, thresholding, cropping
print("Processing Images")
processed = [process(img) for img in images]

# resizing down to model input size : (32,32) for conveinice, images processed quickly 
print("Resizing")
imagesR = [ cv2.resize(img, (32,32)) for img in images ]
processedR = [ cv2.resize(img, (32,32)) for img in processed ]

# flattening all images (2D) to 1D array; i.e simply taking each
# row of each image and stacking next to each other
print("Flattening")
imagesF = [ img.flatten() for img in imagesR ]
processedF = [ img.flatten() for img in processedR ]

# normalizing pixel values
print("Normalizing")
imagesN = StandardScaler().fit_transform(imagesF)
processedN = StandardScaler().fit_transform(processedF)


# <h2> Principal Component Analysis (PCA) </h2>
# Now we can finally apply PCA on the processed and original images. But *what is PCA* you ask ? As the name states, PCA is a rigourous statistical procedure opreation which tries to find the *Principal Components* (PCs) of the dataset. 
# 
# These components can be thought of as the directions /axes/dimensions along which the* data varies the most*. In essence, the different values of principal components are responsible for the diversity in the dataset. Just as the components of say Displacement along the three axis x,y,z give meaningful description of location, the PCs provide some abstract representation of the data.  
# 
# If you want a more technical description, the components are defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal/independent/uncorrelated/ to the preceding components.
# 
# Now we can perform PCA for as many components as we want but our goal is ***dimensionality reduction***. We wan to be able to slowly decompose the data into lower dimensions so that we can capture its rawest form and visualize it. For now, we will be performing PCA for 3 components so that we can plot them in 3D space. 
# 
# Below, both othe original images and processed images are put under PCA. As before, the ouptut components are normalized so that they are within a fixed range of values and zero centered

# In[156]:


# Running PCA on scaled orginal images and processed; generating 3 components;
print("Performing PCA for 3 components")
pca = PCA(n_components=3)
pca0 = pca.fit_transform(imagesN)
pca0 = StandardScaler().fit_transform(pca0)

pca = PCA(n_components=3)
pca1 = pca.fit_transform(processedN)
pca1 = StandardScaler().fit_transform(pca1)


# <h3>Plotting the output in 3D Space </h3>
# 
# The function below uses the plotly library to create interactive 3D plots of any data we pass to it. Let's see it in action !

# In[157]:


def plotly3D(data, labels, annotations= None, title = 't-SNE Plot'):
    '''
    This function takes in 3 dimensional data points 
    and plots them in a 3D scatter plot, color coded
    by their labels
    '''
    # getting unique classes from labels
    # in this case: 0-9
    nClasses = len(np.unique(labels))
    
    # we will plot points for each digit seperately 
    # and color coded; they be stored here
    points = []
    
    # going over each digit
    for label in np.unique(labels):
        
        # getting data points for that digit; coods must be column vectors
        x = data[np.where(labels == label), 0].reshape(-1,1)
        y = data[np.where(labels == label), 1].reshape(-1,1)
        z = data[np.where(labels == label), 2].reshape(-1,1)
        
        # adding file name to each point
        if annotations is not None:
            annotations = np.array(annotations)
            ptLabels = annotations[np.where(labels == label)]
        else:
            ptLabels = None
            
        # creating points in 3d space
        trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', text = ptLabels,
                             marker=dict(size=8, color=label, colorscale='Viridis', opacity=0.6))
        
        # adding plot to list of plots
        points.append(trace)
    
    # plotting all of the datapoints
    layout = dict(title = title,showlegend=True) 
    fig = go.Figure(data=points, layout=layout)
    py.iplot(fig)


# The colors representing the 10 digits are given in the legend. Points of each digit can be isolated by double clicking on its entry in the legend. A single click will hide it. Some browsers may not be able to display the interactive plots on kaggle; in that case you can refer to to plots hosted on my Plotly account below:
# - Before : https://plot.ly/~shahruk10/9.embed
# - After  :  https://plot.ly/~shahruk10/7.embed

# In[158]:


# plotting t-SNE before and after processing
plotly3D(pca0, labels, annotations = annots, title = 'PCA before Processing')
plotly3D(pca1, labels, annotations = annots, title = 'PCA after Processing')


# 
# <h3> More PCs ! </h3>
# 
# We can definitely see better clustering of the digits and more seperation between those clusters after processing. This means the data is more *distinguishable* than before. Any machine learning model should be to learn these boundaries pretty well and hence accurately classify digits.  What happens if we up the no. of PCs? Let's do PCA for 20 components. But how will we visualize 20 components? Ah ! TSNE !

# In[159]:


# Running PCA on scaled orginal images and processed; generating 20 components;
print("Performing PCA for 20 components")
pca = PCA(n_components=20)
pca0 = pca.fit_transform(imagesN)
pca0 = StandardScaler().fit_transform(pca0)

pca = PCA(n_components=20)
pca1 = pca.fit_transform(processedN)
pca1 = StandardScaler().fit_transform(pca1)


# <h1>t-Distributed Stochastic Neighbour Embedding (t-SNE) </h1>
# 
# Like PCA, t-SNE is another dimensionality reduction technique, developed in part by the godfather of machine learning himself, Geoffrey Hinton. Quoting Wikipedia : It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability. 
# 
# Traditionally t-SNE is applied on data which has already been reduced dimensionally by other means such as PCA. This is what we will do ! We will apply 3-component t-SNE on the outputs of the 20-component PCA. This will allow us to plot the t-SNE representation in 3D space 

# In[160]:


# Running t-SNE on PCA outputs; generating 3 dimensional data points
print("Calculating TSNE")
tsne = TSNE(n_components=3, perplexity=40, verbose=2, n_iter=500,early_exaggeration=1)
tsne0 = tsne.fit_transform(pca0)
tsne0 = StandardScaler().fit_transform(tsne0)

tsne = TSNE(n_components=3, perplexity=40, verbose=2, n_iter=500,early_exaggeration=1)
tsne1 = tsne.fit_transform(pca1)
tsne1 = StandardScaler().fit_transform(tsne1)


# <h3>Plotting the output in 3D Space  </h3>
# 
# The plots show that the clustering is much better after applying preprocessing. Each digit almost occupies it's own chunk of spage with little entanglement with other digits. You can analyze the distribution better by zooming into the center of cloud in the interactive plot and look from the inside out. Each direction you look, there is largely a cloud of points of a single color, i.e single type of digit. 
# 
# - Before : https://plot.ly/~shahruk10/11.embed
# - After  :  https://plot.ly/~shahruk10/13.embed

# In[162]:


# plotting t-SNE before and after processing
plotly3D(tsne0, labels, annotations=annots, title = 't-SNE before Processing')
plotly3D(tsne1, labels, annotations=annots, title = 't-SNE after Processing')


# <h3>Snapshots from the 3D Plot</h3>
# - Most blobs occupy their own subspace
# ![](https://lh4.googleusercontent.com/2LzIgVoFv0sXztukqNMgDiFD0CSQRV0W8L5okO41snd1HKSwcVKXpGJKF938_Xmco6Qiptu_3xd-nuo00vYS=w1854-h982)
# 

# 
# - The blue (9)  and orange (1) blobs are close together as they are similiar feature wise. Still there is a visible seperation between major portions of the blobs
# ![](https://lh4.googleusercontent.com/DWbzE1A3YuILWqIvP-FT72MRFWo0S1a6niKBPB4q8rQgjRWGYIt1KN7IYNLsGxNXyBqVnfV7fEkzW9HABb7D=w1855-h982)

# <h2>Results from Test Datasets</h2>
# We can perform similar analysis on the other datasets, including those for which labels are not given (test sets) and get a feel for it's distribution to.
# 
# TODO

# 
