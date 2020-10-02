#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The main purpose of Principal Component Analysis is to identify the key patterns (components) which contribute to our data and in turn, reduce the dimensions of the data with a minimal loss of information. 
# 
# We want to take our $n$-dimensional data and project it on a smaller sub-space that represents our data 'well'. PCA provides the advantage reducing computational costs and errors during paramter estimation by reducing the number of dimensions in the data to a subspace which 'best' describes the data. Primarly applications of PCA is for pattern recognition, classification, dimensionality reduction. 

# ## Intuition
# 
# Imagine an _ellipsoid_; which looks like a sphere which has been scaled on any of its axis. Think of it as a basketball&mdash;with concentric cirlces going from top to bottom like in the picture&mdash;which you can stretch or squeeze both vertically and/or horizontally but cannot twist or deform it in a way that the lines on its surface no longer remain parallel. It follows the constraints of a linear transformation in that respect. 
# 
# If our underlying pattern, or function is found on the tip of the ellipsoid, then imagine all the features spread across the different axeses of the ellipsoid. If an axis is small, then variance of the feature spread on that axis is also small. By omitting that axis and the principal component (feature) associated with that axis, we reduce our feature space and only lose a commensurately small amount of information. 
# 
# Reference: https://en.wikipedia.org/wiki/Principal_component_analysis
# 
# **Note**: PCA does **not** improve accuracy. It is a method to reduce dimensions in order to improve computation, reduce noise, and it can also be used as a regularizing method to prevent overfitting. However, it usually does not make any difference when it comes to accuracy. 
# 
# PCA can also be thought of as an unsupervised algorithm since it ignores the class labels and the goal is to find the principal components (or directions) that maximize the variance in the dataset.
# 
# ![Ellipsoide](https://i.imgur.com/ehwizHt.png)
# ### Linear Algebra
# **Eigen-stuff**: Whenever a linear space goes through a scalar transformation, the vector space only gets scaled or squeezed, there is no rotation or direction change in the vector space which offsets the vectors of their spans. In matrix transformations though the vectors do get knocked off their span because the vector space gets rotated and squished and stretched as well. However, in certain cases of vector transformations, there are some vectors in the vector space which&mdash;despite undergoing a matrix transformation&mdash;do not get knocked off their span and retain it. These vectors are known as Eigenvectors (also known as proper vectors). 
# 
# They merely get squeezed or stretched like during a scalar transformation. The magnitude of difference between their length before the transformation and their length after the transformation are known as Eigenvalues.
# 
# Reference: https://www.youtube.com/watch?v=PFDu9oVAE-g
# 
# We want to use this same principle to find the eigenvectors for the covariance matrix of our dataset. The covariance matrix describes how one feature covaries with respect to the other feature. Finding the eigenvectors of this feature space will help us identify the magnitude of the span of each feature vector (also known as eigenvalues). If the span (axis) is small, then the variance of the feature spread on that axis is also small and we can drop it.

# ## Steps
# 
# 1. Create a $k$-dimensional vector comprising of the means of all the features of our dataset.
# 2. Subtract the mean vector from the feature set to center the data around the origin.
# 3. Compute the covariance matrix of the data. 
# 4. Calculate the eigenvectors and eigenvalues of the covariance matrix.
# 5. Each of the mutually orthogonal, unit eigenvectors can be interpreted as an axis of the ellipsoid fitted to the data.
# 6. This choice of basis will transform our covariance matrix into a diagonalised form with the diagonal elements representing the variance of each axis.
# 7. The proportion of the variance that each eigenvector represents can be calculated by dividing the eigenvalue corresponding to that eigenvector by the sum of all eigenvalues.

# ## Classifying Stars, Galaxies, and Quasars
# 
# ### Sloan Digital Sky Survey
# The Sloan Digital Sky Survey is a project which offers public data of space observations. It has created the most detailed three-dimensional maps of the Universe ever made, with deep multi-color images of one third of the sky, and spectra for more than three million astronomical objects. Observations have been made since 1998 and have been made accessible to everyone who is interested.
# 
# For this purpose a special 2.5 m diameter telescope was built at the Apache Point Observatory in New Mexico, USA. The telescope uses a camera of 30 CCD-Chips with 2048x2048 image points each. The chips are ordered in 5 rows with 6 chips in each row. Each row observes the space through different optical filters (u, g, r, i, z) at wavelengths of approximately 354, 476, 628, 769, 925 nm.
# 
# The telescope covers around one quarter of the earth's sky&mdash;therefore focuses on the northern part of the sky.
# 
# 

# In[ ]:


#Importing required libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.color_palette("Blues")
import pandas as pd
import numpy as np
sns.set_style('whitegrid')


# In[ ]:


#importing in the data
sky = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")

#taking a peek at the data
sky.head()


# ### Feature Description
# 
# **Camera Features**
# 
# - objid = Object Identifier
# - ra = J2000 Right Ascension (r-band)
# - dec = J2000 Declination (r-band)
# 
# **Right ascension (abbreviated ra)** is the angular distance measured eastward along the celestial equator from the Sun at the March equinox to the hour circle of the point above the earth in question. When paired with **declination (abbreviated dec)**, these astronomical coordinates specify the direction of a point on the celestial sphere (traditionally called in English the skies or the sky) in the equatorial coordinate system.
# 
# Source: https://en.wikipedia.org/wiki/Right_ascension
# 
# - u = better of DeV/Exp magnitude fit
# - g = better of DeV/Exp magnitude fit
# - r = better of DeV/Exp magnitude fit
# - i = better of DeV/Exp magnitude fit
# - z = better of DeV/Exp magnitude fit
# 
# **The Thuan-Gunn astronomic magnitude system. u, g, r, i, z represent the response of the 5 bands of the telescope.**
# 
# Further reading: https://www.astro.umd.edu/~ssm/ASTR620/mags.html
# 
# - run = Run Number
# - rereun = Rerun Number
# - camcol = Camera column
# - field = Field number
# 
# **Run, rerun, camcol and field are features which describe a field within an image taken by the SDSS. A field is basically a part of the entire image corresponding to 2048 by 1489 pixels.** A field can be identified by:
# 
# - run number, which identifies the specific scan,
# - the camera column, or "camcol," a number from 1 to 6, identifying the scanline within the run, and
# - the field number. The field number typically starts at 11 (after an initial rampup time), and can be as large as 800 for particularly long runs.
# - An additional number, rerun, specifies how the image was processed.
# 
# **Physics Features**
# 
# - specobjid = Object Identifier
# - class = object class (galaxy, star or quasar object)
# 
# **The class identifies an object to be either a galaxy, star or quasar. This will be the response variable which we will be trying to predict.**
# 
# - redshift = Final Redshift
# - plate = plate number
# - mjd = MJD of observation
# - fiberid = fiber ID
# 
# In physics, **redshift** happens when light or other electromagnetic radiation from an object is increased in wavelength, or shifted to the red end of the visual spectrum.
# 
# Each spectroscopic exposure employs a large, thin, circular metal **plate** that positions optical fibers via holes drilled at the locations of the images in the telescope focal plane. These fibers then feed into the spectrographs. Each plate has a unique serial number, which is called plate in views such as SpecObj in the CAS.
# 
# **Modified Julian Date (mjd)**, used to indicate the date that a given piece of SDSS data (image or spectrum) was taken.
# 
# The SDSS spectrograph uses optical fibers to direct the light at the focal plane from individual objects to the slithead. Each object is assigned a corresponding **fiberID**.
# 
# Further information on SDSS images and their attributes:
# 
# http://www.sdss3.org/dr9/imaging/imaging_basics.php
# 
# http://www.sdss3.org/dr8/glossary.php
# 

# ### Exploratory Analysis

# In[ ]:


#Looking at some summary statistics
sky.describe()


# **Observations**
# 
# - 10,000 examples
# - No categorical features
# - The **5 bands of the telescope (except u)** seem to follow a _normal distribution_. Based on the distribution of u in its four quarters, it seems to be _skewed to the right_.
# - Based on the quarter distribution, **redshift** is skewed to the left with a few outlier to the right.

# In[ ]:


#Let's check for any NULL values
sky.isnull().sum()


# Great! No NULL values! At this point, we're ready to drop a few columns which have no significance on our classifications. 
# 
# - We'll drop objid and specobjid because they are random object identifiers.
# - We will also drop field, run, rerun, and camcol because they are features representing camera positions and particulars which dont have any significance with astronomical objects we are observing.

# In[ ]:


drop_columns = ['objid','specobjid','camcol','rerun','run','field']
sky.drop(drop_columns, axis=1, inplace=True)

sky.head()


# Let's quickly check the distribution of our three categories (STAR, GALAXY, and QUASARS).

# In[ ]:


sky['class'].value_counts()


# In[ ]:


values = sky['class'].value_counts().values
proportion = values/np.sum(values)
classes = ['Galaxies','Stars','Quasars']
for i in range(3):
    print(f"Proportion of {classes[i]}: {round(proportion[i]*100,2)}%")


# From a physics perspective, this is what we expect to observe, an almost even distribution of galaxies and stars but quasars are rare so less number of quasar observations in our dataset. From an analysis perspective, our dataset looks to be heavily biased against identifying quasars.
# 
# Let's look at a scatter matrix for the dataset.

# In[ ]:


sns.pairplot(sky)
plt.show()


# - There looks to be a strong linear relationship between the 5 bands of the telescope. Because of potential multicollinearity these 5 are good candidates for dimensionality reduction.
# -  mjd and plate also look to be highly correlated and are also good candidates for dimensionality reduction.
# 
# Let's have a look at the correlation matrix for the dataset.

# In[ ]:


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(sky)


# - We see that the 5 bands of the telescope are highly correlated.
# - As are mpj and plate.
# 
# Let's explore the distributions of the three classes separately.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(sky[sky['class']=='STAR'][['u', 'g', 'r', 'i', 'z']].corr(), linewidths=0.1, ax = axes[0],                  cmap='coolwarm', linecolor='white', annot=True)
ax.set_title('Star')
ax = sns.heatmap(sky[sky['class']=='GALAXY'][['u', 'g', 'r', 'i', 'z']].corr(), linewidths=0.1, ax = axes[1],                  cmap='coolwarm', linecolor = 'white', annot=True)
ax.set_title('Galaxy')
ax = sns.heatmap(sky[sky['class']=='QSO'][['u', 'g', 'r', 'i', 'z']].corr(), linewidths=0.1, ax = axes[2],                  cmap='coolwarm', linecolor = 'white', annot=True)
ax = ax.set_title('QSO')


# The correlation between the 5 bands of the telescope is evenly distributed amongst the three classes of objects.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(sky[sky['class']=='STAR'][['mjd','plate']].corr(), linewidths=0.1, ax = axes[0],                  cmap='coolwarm', linecolor='white', annot=True)
ax.set_title('Star')
ax = sns.heatmap(sky[sky['class']=='GALAXY'][['mjd','plate']].corr(), linewidths=0.1, ax = axes[1],                  cmap='coolwarm', linecolor = 'white', annot=True)
ax.set_title('Galaxy')
ax = sns.heatmap(sky[sky['class']=='QSO'][['mjd','plate']].corr(), linewidths=0.1, ax = axes[2],                  cmap='coolwarm', linecolor = 'white', annot=True)
ax = ax.set_title('QSO')


# The correlation between mjd and plate is also evenly distributed.

# In[ ]:


# plt.figure(figsize=(15,7))
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,7))
ax1.boxplot(sky[sky['class']=='QSO']['redshift'])
ax1.set_title("Quasars")
ax2.boxplot(sky[sky['class']=='STAR']['redshift'])
ax2.set_title("Stars")
ax3.boxplot(sky[sky['class']=='GALAXY']['redshift'])
ax3.set_title("Galaxy")
plt.show()


# - The distribution of Quasars looks to be uniform between 0-3 and then it has a few positively skewed outliers.
# - Stars look like they follow a truncated normal distribution centered at 0. Most of the data seems to lie between -0.0001 and 0.0001.
# - Galaxies also look to be normally distributed centered around 0.75 but with a positive skew.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(sky[sky['class']=='STAR']['redshift'], bins = 30, ax = axes[0], kde = True)
ax.set_title('Star')
ax = sns.distplot(sky[sky['class']=='GALAXY']['redshift'], bins = 30, ax = axes[1], kde = True)
ax.set_title('Galaxy')
ax = sns.distplot(sky[sky['class']=='QSO']['redshift'], bins = 30, ax = axes[2], kde = True)
ax = ax.set_title('Quasar')
plt.show()


# Redshift is a good estimation of how far away the celestial body being observed is from the Earth. 
# - In this case it looks like the observed Quasars, by virtue of being so violent, are the furthest from Earth.
# - Other galaxies&mdash;Earth being in the milky way&mdash; will be farther from the Earth than the Starts we observe in the night sky (from our own or other galaxies).
# - Also, since galaxies and quasars are stronger than stars, their light can be observed from further distances which is also why the redshift of these two objects is much higher than Stars. 

# ## Feature Engineering
# ### Principal Component Analysis
# 
# **Preprocessing**:  
# Separate out the features undergoing PCA for the three different classes.

# In[ ]:


#Separating out the dataframe
stars = sky[sky['class'] == "STAR"]
galaxies = sky[sky['class'] == "GALAXY"]
quasars = sky[sky['class'] == "QSO"]

#Extracting the features
all_samples = sky[['u','g','r','i','z']].values

print(f'Shape of Stars: {all_samples.shape}')


# **Step 1**:  
# Extract the means of the features and create a mean vector.

# In[ ]:


means = np.zeros((5,1))
for i in range(5):
    means[i,0] = np.mean(all_samples[:,i], axis=0)

print(f"Shape of means: {means.shape}")


# **Step 2**:  
# Subtract the mean vector from the array of all samples.

# In[ ]:


all_samples -= means.T


# **Step 3**:  
# Create the covariance matrix.

# In[ ]:


cov_mat = np.cov([all_samples[:,i] for i in range(5)])
print('Covariance Matrix:\n', cov_mat)


# **Step 4**:  
# Extract the eigenvectors and eigenvalues.

# In[ ]:


eigen_values, eigen_vectors = np.linalg.eig(cov_mat)


# In[ ]:


eigen_values


# Let's look at the proportions of eigen value distribution amongst the features.

# In[ ]:


eigen_values_p = eigen_values/np.sum(eigen_values)
print(eigen_values_p)
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]


# **Step 5**:  
# Sort the eigen vectors based on their eigen values.

# In[ ]:


eigen_pairs.sort(key = lambda x: x[0], reverse=True)
eigen_pairs


# Reshape our eigen vectors into (n,1) arrays.

# In[ ]:


eigen_vectors_final = np.hstack((eigen_pairs[0][1].reshape(-1,1), eigen_pairs[1][1].reshape(-1,1),                                  eigen_pairs[2][1].reshape(-1,1)))
eigen_vectors_final


# **Step 6**:  
# Find the dot product with the sample data and the eigen vectors.

# In[ ]:


ugriz = eigen_vectors_final.T.dot(all_samples.T)
ugriz.T


# Perfect! Now, let's quickly check our implementation using sci-kit learn's PCA library. 
# 
# **Check**

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=3)
ugriz = pca.fit_transform(sky[['u', 'g', 'r', 'i', 'z']])
ugriz


# Things look good! Our eigen vectors are of the same magnitude. You may notice that the first eigen vector has opposite signs as to our sign but that's part of sci-kit learn's implementation and you can read more about it [here](https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative). The signs do not have any effect on the variance of the vector because the span remains the same. 
# 
# Now, let's do the same with mjd and plate.

# In[ ]:


pca = PCA(n_components=1)
mjdplate = pca.fit_transform(sky[['mjd','plate']])
mjdplate


# Let's update our data with the PCA columns.

# In[ ]:


# update dataframe 
sky_pca = pd.concat((sky, pd.DataFrame(ugriz)), axis=1)
sky_pca.rename({0: 'bands_1', 1: 'bands_2', 2: 'bands_3'}, axis=1, inplace = True)
sky_pca = pd.concat((sky_pca, pd.DataFrame(mjdplate)), axis =1)
sky_pca.rename({0: 'mjdplate_1'}, axis=1, inplace = True)

sky_pca.drop(['u','g','r','i','z'], axis = 1, inplace=True)

#Encoding the class variables to quantitative variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(sky_pca['class'])
sky_pca['class'] = y_encoded


# In[ ]:


sky_pca.head()


# Let's extract the feature vectors and class vector separately for classification.

# In[ ]:


X = sky_pca[['bands_1','bands_2','bands_3','redshift','ra','dec','fiberid','mjdplate_1']].values
y = sky_pca[['class']].values


# ### Standardization/Normalization

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scaled = ss.fit_transform(X)


# ### Train-Test Split

# In[ ]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 420)


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

LG = LogisticRegression(penalty="l2")
LG.fit(X_train, y_train)


# ### Assessment

# In[ ]:


y_preds = LG.predict(X_test)
accuracy = LG.score(X_test,y_test)

print(f"Accuracy of Logistic Model: {accuracy*100}%")


# In[ ]:


#Plotting the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
ss, sg, sq, gs, gg, gq, qs, qg, qq = confusion_matrix(y_test,y_preds).ravel()
cm = confusion_matrix(y_test, y_preds)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt="g", cmap="Blues_r"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Stars', 'Galaxies','Quasars']); 
ax.yaxis.set_ticklabels(['Stars', 'Galaxies','Quasars']);


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test,y_preds, average="micro")
recall = recall_score(y_test, y_preds, average="micro")
f1 = f1_score(y_test, y_preds, average="micro")

print(f"Precision Score: {precision}\nRecall Score: {recall}\nF1 Score: {f1}")


# ### Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(LG, X_train, y_train.squeeze(), cv=10, scoring = "accuracy")

print(f"Mean Cross Validation Score: {scores.mean()}\nCross Validation Score Standard Deviation: {scores.std()}")


# ## Disclaimer  
# 1. This notebook was just meant to give an intuitive tutorial on how to go about understanding and implementing PCA. I in no way claim that this was the best place to use PCA, or the classification model which was run post the PCA was optimized or has provided better results than it would have pre-PCA. 
# 2. I am **not** an expert. I'm someone who is learning himself, so I welcome any and everyone to find any problem or flaws in this notebook and let me know about it. I'd love to improve my own mathematical intuition in this case. 

# ## Key References
# 1. **Sebastian Raschka PCA tutorial**: http://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/principal_component_analysis.ipynb
# 2. **Lennart Grosser (Data Analysis and Classification using XGBoost)**: https://www.kaggle.com/lucidlenn/data-analysis-and-classification-using-xgboost
# 3. **Is PCA unstable under multicollinearity?**: https://stats.stackexchange.com/questions/9542/is-pca-unstable-under-multicollinearity

# In[ ]:




