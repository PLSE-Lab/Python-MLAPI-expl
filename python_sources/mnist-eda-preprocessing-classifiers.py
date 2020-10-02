#!/usr/bin/env python
# coding: utf-8

# # Machine Learning on MNIST dataset (Handwritten digits recognition)
# *Upload the notebook to the Kaggle competition: https://www.kaggle.com/c/digit-recognizer/*
# 
# **Datasets**
# 
# - train.csv: training set containing the label and the 784 pixel values (28x28 images), between 0-255
# - test.csv : the set to make predictions over
# - sample_submission.csv : example of submission csv format for the competition
# 
# **Metric**: accuracy of the classification (% of right predictions)

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


# ## EDA

# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.head())
print(train.info())
print("\n SHape of the dataset:", train.shape)


# In[ ]:


#NaN values in the dataset ?
nan = train.isnull().sum()
print(nan[nan != 0])


# *--> No missing values*

# In[7]:


#Displays 4 handwritten digit images
def display_digits(N):
    """Picks-up randomly N images within the 
    train dataset between 0 and 41999 and displays the images
    with 4 images/row"""
    
    train = pd.read_csv('../input/digit-recognizer/train.csv')
    images = np.random.randint(low=0, high=42001, size=N).tolist()
    
    subset_images = train.iloc[images,:]
    subset_images.index = range(1, N+1)
    print("Handwritten picked-up digits: ", subset_images['label'].values)
    subset_images.drop(columns=['label'], inplace=True)

    for i, row in subset_images.iterrows():
        plt.subplot((N//8)+1, 8, i)
        pixels = row.values.reshape((28,28))
        plt.imshow(pixels, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.title('Randomly picked-up images from the training dataset')
    plt.show()

    return ""


# In[8]:


display_digits(40)


# In[ ]:


#Analyse the pixels intensity values
subset_pixels = train.iloc[:, 1:]
subset_pixels.describe()


# *--> Some pixels have always an intensity of 0 (max=0) or 255 (min=255)*

# In[9]:


#Distribution of the digits in the dataset
_ = train['label'].value_counts().plot(kind='bar')
plt.show()


# ---

# ## Preprocessing

# ### Pre-traitment
# #### Remove pixels with constant intensity
# The removal of constant pixels (smae intensity in the entire training set) allow to reduce the number of features and compute faster

# In[ ]:


def remove_constant_pixels(pixels_df):
    """Removes from the images the pixels that have a constant intensity value,
    either always black (0) or white (255)
    Returns the cleared dataset & the list of the removed pixels (columns)"""

    #Remove the pixels that are always black to compute faster
    changing_pixels_df = pixels_df.loc[:]
    dropped_pixels_b = []

    #Pixels with max value =0 are pixels that never change
    for col in pixels_df:
        if changing_pixels_df[col].max() == 0:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixels_b.append(col)
    print("Constantly black pixels that have been dropped: {}".format(dropped_pixels_b))


    #Same with pixels with min=255 (white pixels)
    dropped_pixels_w = []
    for col in changing_pixels_df:
        if changing_pixels_df[col].min() == 255:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixel_w.append(col)
    print("\n Constantly white pixels that have been dropped: {}".format(dropped_pixels_b))

    print(changing_pixels_df.head())
    print("Remaining pixels: {}".format(len(changing_pixels_df.columns)))
    print("Pixels removed: {}".format(784-len(changing_pixels_df.columns)))
    
    return changing_pixels_df, dropped_pixels_b + dropped_pixels_w


# In[ ]:


train_pixels_df = pd.read_csv('../input/digit-recognizer/train.csv').drop(columns=['label'])
train_changing_pixels_df, dropped_pixels = remove_constant_pixels(train_pixels_df)


# In[2]:


#To save time and not have to run the entire function
DROPPED_PIX = ['pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 
               'pixel10', 'pixel11', 'pixel16', 'pixel17', 'pixel18', 'pixel19', 'pixel20', 'pixel21', 'pixel22', 
               'pixel23', 'pixel24', 'pixel25', 'pixel26', 'pixel27', 'pixel28', 'pixel29', 'pixel30', 'pixel31', 
               'pixel52', 'pixel53', 'pixel54', 'pixel55', 'pixel56', 'pixel57', 'pixel82', 'pixel83', 'pixel84', 
               'pixel85', 'pixel111', 'pixel112', 'pixel139', 'pixel140', 'pixel141', 'pixel168', 'pixel196', 
               'pixel392', 'pixel420', 'pixel421', 'pixel448', 'pixel476', 'pixel532', 'pixel560', 'pixel644', 
               'pixel645', 'pixel671', 'pixel672', 'pixel673', 'pixel699', 'pixel700', 'pixel701', 'pixel727', 
               'pixel728', 'pixel729', 'pixel730', 'pixel731', 'pixel754', 'pixel755', 'pixel756', 'pixel757', 
               'pixel758', 'pixel759', 'pixel760', 'pixel780', 'pixel781', 'pixel782', 'pixel783']
train_changing_pixels_df = pd.read_csv('../input/changing-pixels/train_changing_pixels_DB.csv', index_col=0)
print(train_changing_pixels_df.head())
train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.head())


# #### Pictures intensity rescaling
# - Accentuation of contrasts by resclaling
# - changing pixels values to only 0 or 255 (blakc or white)

# In[116]:


#Pick-up one random image from original training set
i = np.random.randint(low=0, high=42001, size=1).tolist()[0]
pixels = train.iloc[i, 1:]
image = train.iloc[i, 1:].values.reshape((28,28))

#Pixel intensity hstogram
plt.hist(pixels, bins=256, range=(0,256), normed=True)
plt.title('original image - pixel intensity distribution')
plt.show()

#Rescaling the intensity
pmin, pmax = image.min(), image.max()
rescaled_image = 255*(image-pmin) / (pmax - pmin)
rescaled_pixels = rescaled_image.flatten()

#Only black or white pixels
bw_pixels = pixels.apply(lambda x: 0 if x<128 else 255)
bw_image = bw_pixels.values.reshape((28,28))


#Visual comparison of images
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('original image')
plt.subplot(1, 3, 2)
plt.imshow(rescaled_image, cmap='gray')
plt.title('rescaled image')
plt.subplot(1, 3, 3)
plt.imshow(bw_image, cmap='gray')
plt.title('black&wite only image')
plt.show()


#Visual Histogram comparison
plt.subplot(1, 2, 1)
plt.hist(pixels, bins=256, range=(0,256), normed=True)
plt.title('original image')
plt.subplot(1, 2, 2)
plt.hist(rescaled_pixels, bins=256, range=(0,256), normed=True)
plt.title('rescaled image')
plt.show()


# *Image rescaling is not applicable here as the intensity distribution is not srpead enough*

# #### Dummy variables and target ?

# In[ ]:


#Dummy target ?


# ### Dimension Reduction
# #### Principal Components Analysis (PCA)

# In[ ]:


#Preparing samples and labels arrays
#s = np.random.randint(low=0, high=42001, size=1050).tolist()
samples = train_changing_pixels_df.values  #.iloc[s, :]
digits = train['label'].tolist()  #.iloc[s, :]
print(samples.shape)


# In[ ]:


#PCA model
pca = PCA()
pca.fit(samples)

#PCA features variance visualization
pca_features = range(pca.n_components_)
_ = plt.figure(figsize=(30,20))
_ = plt.bar(pca_features, pca.explained_variance_)
_ = plt.xticks(pca_features)
_ = plt.title('Principal Components Analysis for Dimension Reduction')
_ = plt.xlabel('PCA features')
_ = plt.ylabel('Variance of the PCA feature')
_ = plt.savefig('visualizations/PCA features variance.png')
plt.show()

#PCA features variance visualization - ZOOM in
l= 100
x = range(l)
_ = plt.figure(figsize=(30,20))
_ = plt.bar(x, pca.explained_variance_[:l])
_ = plt.xticks(x)
_ = plt.title('Principal Components Analysis for Dimension Reduction - Zoom In {} first features'.format(l))
_ = plt.xlabel('PCA features')
_ = plt.ylabel('Variance of the PCA feature')
_ = plt.savefig('visualizations/PCA features variance_zoom.png')
plt.show()


# *The best number of PCA features to keep (reduced dimension) is <80. The 1st PCA feature represents a lot of the variance, as the 5 following ones.*

# In[ ]:


#Visualization of the variance of the data carried by the number of PCA features
n_components = np.array([1,2,3,4,5,6, 10, 30, 60, 80, 100, 200, 400, 700])
cumul_variance = np.empty(len(n_components))
for i, n in enumerate(n_components):
    pca = PCA(n_components=n)
    pca.fit(samples)
    cumul_variance[i] = np.sum(pca.explained_variance_ratio_)

print(cumul_variance)

_ = plt.figure(figsize=(30,20))
_ = plt.grid(which='both')
_ = plt.plot(n_components, cumul_variance, color='red')
_ = plt.xscale('log')
_ = plt.xlabel('Number of PCA features', size=20)
_ = plt.ylabel('Cumulated variance of data (%)', size=20)
_ = plt.title('Data variance cumulated vs number of PCA features', size=20)
plt.savefig('visualizations/cumulated variance_pca features.png')
plt.show()


# *50% of the data variance is carried by 10 PCA features, and 90% of the data variance is carried by 80 PCA features (vs 708 features initially). The reduction is huge !*
# 
# *However, the PCE features have no interpretable meaning.*
# 
# *--> The use of PCA() in the pipleine along with StandardScaler() and SVC() increased the accuracy up to 0.97 trained on half of the samples.*

# #### Non-Negative MAtrix Factorization (NMF)

# In[ ]:


#Preparing samples and labels arrays
s = np.random.randint(low=0, high=42001, size=8200).tolist()
samples = train.drop(columns='label').values  #.iloc[s, :]
digitsa = train['label'].tolist()  #.iloc[s, :]
print(samples.shape)


# In[ ]:


#Creating the NMF model and the features & components
nmf = NMF(n_components=16)
nmf_features = nmf.fit_transform(samples)
nmf_components = nmf.components_
print("Shape of NMF features: {}, shape of NMF components: {}".format(nmf_features.shape, nmf_components.shape))

#Visualization of the features
for i, component in enumerate(nmf_components):
    N = nmf_components.shape[0]
    ax = plt.subplot((N//3)+1, 3, i+1)
    bitmap = component.reshape((28,28))
    plt.imshow(bitmap, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.title('NMF Components from the original images')
plt.show()


# *The samples have been split between NMF features and components, in a way that Features x Components = Samples (with matrices product)*
# 
# *The images have been decomposed into combinations of the NMF components displaed above.*

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#Sample randomly the dataset using discrete_uniform pick-up to reduce the amount of data\n#sample = np.random.randint(low=0, high=42001, size=8400).tolist()\nX = train_changing_pixels_df.values  #.iloc[sample, :]\nX = X / 255.0\ny = train[\'label\'].values  #.iloc[sample, :]\nprint("Shape of X and Y arrays: {}".format((X.shape, y.shape)))\n\n#Split the training set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)\n\n#Yielding the scores according to number of NMF components\ncomponents = np.arange(1, 100)\nscores = np.empty(len(components))\nfor n in components:\n    pipeline = make_pipeline(NMF(n_components=n), SVC(kernel=\'rbf\', cache_size=1000))\n    pipeline.fit(X_train, y_train)\n    scores[n-1] = pipeline.score(X_test, y_test)\n\n#Plotting of the scores evlution\n_ = plt.figure(figsize=(30,20))\n_ = plt.grid(which=\'both\')\n_ = plt.plot(components, scores)\n_ = plt.xlabel(\'Number of NMF components\', size=20)\n_ = plt.ylabel(\'Score obtained\', size=20)\n_ = plt.title(\'Evolution of SVC classification score (samples={})\'.format(len(y)), \n              size=30)\nplt.savefig(\'visualizations/Score vs components NMF.png\')\nplt.show()\n\nprint("Best score {} obtained for {} components".format(scores.max(), scores.argmax()+1))')


# *Highest score of 0.83 obtained for 16 components*

# #### Sparse Matrix
# As the pixel inteisties present a lot of 0 values, we can turn the train 2D array into a sparse matrix to reduce the memory size
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
# - TruncatedSVD for the equivalent of PCA()
# - NMF() directly with the csr_matrix

# In[ ]:


#Sparsity visuaization & sparse matrix creation
samples = train_changing_pixels_df.values
_ = plt.figure(figsize=(10,100))
_ = plt.spy(samples)
plt.show()

sparse_samples = csr_matrix(samples)

#Memory Size comparison
dense_size = samples.nbytes/1e6
sparse_size = (sparse_samples.data.nbytes + 
               sparse_samples.indptr.nbytes + sparse_samples.indices.nbytes)/1e6
print("From {} to {} Mo in memory usage with the sparse matrix".format(dense_size, sparse_size))

#Dimension reduction using PCA equivalent for sparse matrix
model = TruncatedSVD(n_components=10)
model.fit(sparse_samples)
reduced_sparse_samples = model.transform(sparse_samples)
print(reduced_sparse_samples.shape)


# *The sparsity visualization confirms the relevance of the use of a csr matrix for the samples*

# ## Simple Classifiers
# - training datasets
#     - changing pixels only dataset
#     - randomly sampled
# - Classifier to try: 
#     - k-nn, 
#     - LDA/QDA,
#     - SVC & Linear SVC, 
#     - Decision tree,
#     - LVQ

# In[2]:


from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
from sklearn.decomposition import PCA, NMF, TruncatedSVD

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV


# #### K-NN

# In[14]:


#Sample randomly the dataset using discrete_uniform pick-up
sample = np.random.randint(low=0, high=42001, size=2100).tolist()

#Prepare the X (features) and y (label) arrays for the sampled images
X = train_changing_pixels_df.iloc[sample, :].values
y = train.loc[sample, 'label'].values#.reshape(-1,1)
print("Shape of X and Y arrays: {}".format((X.shape, y.shape)))

#Split the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)

#Fine tune the k value
param_grid = {'n_neighbors': np.arange(1,10)}
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_cv.fit(X_train, y_train)

#Best k parameter
best_k = knn_cv.best_params_
best_accuracy = knn_cv.best_score_
print("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))


# *Obtained score on Kaggle: 0.94300 trianed on entire original dataset*

# #### Linear Discriminant Analysis

# In[9]:


get_ipython().run_cell_magic('time', '', '\n#Sample randomly the dataset using discrete_uniform pick-up\n#sample = np.random.randint(low=0, high=42001, size=4200).tolist()\n\n#Prepare the X (features) and y (label) arrays for 4000 images\nX = train_simplified_pixels_df.values #.iloc[sample, 1:]\ny = train.loc[:, \'label\'].values#.reshape(-1,1)\nprint("Shape of X and Y arrays: {}".format((X.shape, y.shape)))\n\n#Split the training set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)\n\n#Fit the model (no hyperparameter tuning for this model)\nlda = LinearDiscriminantAnalysis(solver=\'lsqr\', shrinkage=\'auto\') #best results with these arguments\n#lda = QuadraticDiscriminantAnalysis()#reg_parameter=?\nlda.fit(X_train, y_train)\nscore = lda.score(X_test, y_test)\nprint("Accuracy on test set: {}".format(score))\n\n#Best cv scores\nlda_cv_scores = cross_val_score(lda, X_train, y_train, cv=5)\nbest_accuracy = lda_cv_scores.max()\nprint("Best accuracy during CV is {}".format(best_accuracy))')


# *--> Linear Discriminant Analysis returns a score around 87%, better than the Quadratic Discriminant Analysis, within less than 2min execution on the whole training set (which is pretty fast !)*
# 
# *--> Kaggle submission score:  0.86971*

# #### Support Vector Machines
# ##### SVC()
# - test different kernels of SVC (linear, polynomial, rbf, sigmoid)
# - RandomizedSearchCV to fine tune the hyperparameters (depending on the kernel)

# In[4]:


get_ipython().run_cell_magic('time', '', '\n#Sample randomly the dataset using discrete_uniform pick-up\nsample = np.random.randint(low=0, high=42001, size=2100).tolist()\n\n#Prepare the X (features) and y (label) arrays for the sampled images\nX = train_simplified_pixels_df.iloc[sample, :].values\ny = train.loc[sample, \'label\'].values#.reshape(-1,1)\nprint("Shape of X and Y arrays: {}".format((X.shape, y.shape)))\n\n#Split the training set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)\n\n#Fine tune the hyperparameters using RandomizeSearchCV rather than GridSearchCV (too expensive with more than 1 hyperparameter)\nparam_grid = {\'C\': np.logspace(0, 3, 20),\n              \'gamma\':np.logspace(0, -4, 20)#, not for linear kernel\n              #\'degree\': [2,3,4,5]  #only for poly kernel\n              #\'coef0\': []  #only for poly & sigmoid kernels\n             }\nsvm_cv = RandomizedSearchCV(SVC(kernel=\'rbf\', cache_size=3000), \n                            param_grid, cv=5)\nsvm_cv.fit(X_train, y_train)\n\n#Best k parameter\nbest_k = svm_cv.best_params_\nbest_accuracy = svm_cv.best_score_\nprint("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))')


# **Scores of submission, SVC() fitted on simplified dataset: --> scaled data required! **
# - rbf, C=4.4984, gamma=0.01842: 0.9606 on test set, 0.95642 on submission
# - poly, C=39.069, gamma = 0.01098, degre=3  : 0.937 on test, 0.93 on submission
# - sigmoid, C=754.3, gamma=0.00095  : 0.899 on test*

# ##### LinearSVC()

# In[5]:


get_ipython().run_cell_magic('time', '', '\n#Sample randomly the dataset using discrete_uniform pick-up\nsample = np.random.randint(low=0, high=42001, size=4100).tolist()\n\n#Prepare the X (features) and y (label) arrays for the sampled images\nX = train_simplified_pixels_df.iloc[sample, :].values\ny = train.loc[sample, \'label\'].values#.reshape(-1,1)\nprint("Shape of X and Y arrays: {}".format((X.shape, y.shape)))\n\n#Split the training set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)\n\n#Fine tune the hyperparameters using RandomizeSearchCV rather than GridSearchCV (too expensive with more than 1 hyperparameter)\nparam_grid = {\'multi_class\': [\'ovr\', \'crammer_singer\'],\n              \'penalty\': [\'l1\', \'l2\'],\n              \'C\': np.logspace(0, 4, 50)}\n\nlinsvc_cv = GridSearchCV(LinearSVC(dual=False), param_grid, cv=5)\n#linsvc_cv = RandomizedSearchCV(LinearSVC(dual=False), param_grid, cv=5)\nlinsvc_cv.fit(X_train, y_train)\n\n#Best k parameter\nbest_k = linsvc_cv.best_params_\nbest_accuracy = linsvc_cv.best_score_\nprint("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))')


# *--> C=1, multi-class, crammer_singer, penatly=l1, result on test = 0.845*

# #### Decision Tree classifier

# In[9]:


get_ipython().run_cell_magic('time', '', '\n#Sample randomly the dataset using discrete_uniform pick-up\n#sample = np.random.randint(low=0, high=42001, size=4200).tolist()\n\n#Prepare the X (features) and y (label) arrays for 4000 images\nX = train_changing_pixels_df.values  #.iloc[sample, :]\ny = train.loc[:, \'label\'].values#.reshape(-1,1)\nprint("Shape of X and Y arrays: {}".format((X.shape, y.shape)))\n\n#Split the training set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)\n\n#Fine tune the hyperparameters\nparam_grid = {\'max_depth\': np.arange(3, 50),\n              \'min_samples_leaf\': np.arange(5, 50, 1),\n              \'min_samples_split\': np.arange(2,50, 1)\n             }\ntree_cv = RandomizedSearchCV(DecisionTreeClassifier(),\n                       param_grid, cv=5)\ntree_cv.fit(X_train, y_train)\n\n#Best k parameter\nbest_k = tree_cv.best_params_\nbest_accuracy = tree_cv.best_score_\nprint("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))')


# *--> 0.85 on test*

# ## Pipeline
# - Using the changing pixels dataset
# - Rescaling intensity between 0 and 1 (StandardScaler() (pixels with different variances) or the custom function)
# - Normalize the samples ?
# - no missing values --> no need for an Imputer
# - DImension Reduction (PCA / NMF/ sparce matrix + TruncatedSVD)
# - Hyperparameters Fine tune (GriSearchCV/RandomisedSearchCV)
# - simple classifiers performances: SVC(rbf kernel) > k-nn > LDA > LinearSVC > DecisionTree,...
# - Bagging, Boost

# In[ ]:


#To save time and not have to run the entire function
DROPPED_PIX = ['pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 
               'pixel10', 'pixel11', 'pixel16', 'pixel17', 'pixel18', 'pixel19', 'pixel20', 'pixel21', 'pixel22', 
               'pixel23', 'pixel24', 'pixel25', 'pixel26', 'pixel27', 'pixel28', 'pixel29', 'pixel30', 'pixel31', 
               'pixel52', 'pixel53', 'pixel54', 'pixel55', 'pixel56', 'pixel57', 'pixel82', 'pixel83', 'pixel84', 
               'pixel85', 'pixel111', 'pixel112', 'pixel139', 'pixel140', 'pixel141', 'pixel168', 'pixel196', 
               'pixel392', 'pixel420', 'pixel421', 'pixel448', 'pixel476', 'pixel532', 'pixel560', 'pixel644', 
               'pixel645', 'pixel671', 'pixel672', 'pixel673', 'pixel699', 'pixel700', 'pixel701', 'pixel727', 
               'pixel728', 'pixel729', 'pixel730', 'pixel731', 'pixel754', 'pixel755', 'pixel756', 'pixel757', 
               'pixel758', 'pixel759', 'pixel760', 'pixel780', 'pixel781', 'pixel782', 'pixel783']
train_changing_pixels_df = pd.read_csv('../input/changing-pixels/train_changing_pixels_DB.csv', index_col=0)
print(train_changing_pixels_df.head())
train = pd.read_csv('../input/digit-recognizer/train.csv')
print(train.head())


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n#Sample randomly the dataset using discrete_uniform pick-up to reduce the amount of data\n#sample = np.random.randint(low=0, high=42001, size=21000).tolist()\n\n#Prepare the X (features) and y (label) arrays for the images\nX = csr_matrix(train_changing_pixels_df.values)  #use .iloc[sample, :] for reduced sample\n#X = X / 255.0  #intensities recaled between 0 and 1, NMF don\'t take negative values\ny = train[\'label\'].values#.reshape(-1,1)   #idem if using sample\nprint("Shape of X and Y arrays: {}".format((X.shape, y.shape)))\n\n#Split the training set\nX_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=4, stratify=y)\n\n#Pipeline with fine tune\npipeline = Pipeline([#(\'scaler\', StandardScaler()), \n                     (\'pca\', TruncatedSVD()),\n                     (\'svm\', SVC(kernel=\'rbf\', cache_size=3000))\n                    ])\nparam_grid = {\'pca__n_components\': np.arange(5, 80),\n              \'svm__C\': np.logspace(0, 4, 50),\n              \'svm__gamma\':np.logspace(0, -4, 50)}\npipeline_cv = RandomizedSearchCV(pipeline, param_grid, cv=5)\n\n#fitting\npipeline_cv.fit(X_train, y_train)\n\n#Best k parameter\nbest_k = pipeline_cv.best_params_\nbest_accuracy = pipeline_cv.best_score_\nprint("Best accuracy on test set during training is {} obtained for {}".format(best_accuracy, best_k))')


# In[7]:


get_ipython().run_cell_magic('time', '', '\n#Predict on the test dataset (holdout) that MUST contain as many columns (ie pixels) than in the training set\nholdout = pd.read_csv(\'test.csv\').drop(columns=DROPPED_PIX)\nX_holdout = holdout.values\nprint(X_holdout.shape)\n\npredictions = pipeline_cv.predict(X_holdout)\nsubmission_df = pd.DataFrame({\'ImageId\': range(1,28001), \'Label\': predictions})\nprint("Overview of the obtained predictions :\\n", submission_df.head())\n\n#Save as submission file for competition\nsubmission_df.to_csv(\'submission_pca_svc_DB.csv\', index=False)')


# - PCA, SVM: (no scaler, no normalizer) *0.9747 obtained for {gamma: 0.1, C: 345.5107, n_components: 40}*

# In[ ]:




