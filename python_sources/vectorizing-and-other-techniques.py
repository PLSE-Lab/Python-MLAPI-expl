#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
base_skin_dir = os.path.join('..', 'input')


# # read prepare further data
# * fill in lesions, imagepath,celltype and cancer type
# * read mnist28 file since this one has prepared a compressed format of all images, lets have a look if this files does the job
# * check if the minst file label is the same as the tile cell type idx
# * encode the tile_df such that the basic fields become all numeric

# In[ ]:


imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'melanoma ',   #this is an error in the other scripts
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df.sample(3)
tile_df.describe(exclude=[np.number])


# In[ ]:


# load in all of the images
from skimage.io import imread
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)

tile_df['image'] = tile_df['path'].map(imread)


# In[ ]:





# In[ ]:


import cv2
from sklearn.cluster import KMeans

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        img = cv2.imread(self.IMAGE)
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)


# dominant colours in picture

# In[ ]:


img=tile_df.loc[1,'path']

clusters = 5
dc = DominantColors(img, clusters) 
colors = dc.dominantColors()
print(colors)


# In[ ]:


tile_df.loc[1,'image']


# Splitting the image into seperate colour components

# In[ ]:


import numpy as np
import matplotlib.pylab as plt

get_ipython().run_line_magic('matplotlib', 'inline')

imgnr=1440
im=tile_df.loc[imgnr,'image']
im.shape
def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')

plti(im)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

for c, ax in zip(range(3), axs):
    tmp_im = np.zeros(im.shape, dtype="uint8")
    tmp_im[:,:,c] = im[:,:,c]
    ax.imshow(tmp_im)
    ax.set_axis_off()


# Colour Transformations

# In[ ]:


def do_normalise(im):
    return -np.log(1/((1 + im)/257) - 1)
 
def undo_normalise(im):
    return (1 + 1/(np.exp(-im) + 1) * 257).astype("uint8")

def rotation_matrix(theta):
    """
    3D rotation matrix around the X-axis by angle theta
    """
    return np.c_[
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)]
    ]

im_normed = do_normalise(im)
im_rotated = np.einsum("ijk,lk->ijl", im_normed, rotation_matrix(np.pi))
im2 = undo_normalise(im_rotated)

plti(im2)


# greyscale

# In[ ]:


def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
    """
    Transforms a colour image to a greyscale image by
    taking the mean of the RGB values, weighted
    by the matrix weights
    """
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2)

img = to_grayscale(im)

plti(img, cmap='Greys')


# blurring uniform window, the Gaussian window and the median filter

# In[ ]:


from scipy.ndimage.interpolation import zoom
im_small = zoom(im, (0.2,0.2,1))

from scipy.signal import convolve2d

def convolve_all_colours(im, window):
    """
    Convolves im with window, over all three colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = convolve2d(im[:,:,d], window, mode="same", boundary="symm")
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    
    
    
    return im_conv

n=50
window = np.ones((n,n))
window /= np.sum(window)
plti(convolve_all_colours(im_small, window))


from scipy.ndimage import median_filter

def make_gaussian_window(n, sigma=1):
    """
    Creates a square window of size n by n of weights from a gaussian
    of width sigma.
    """
    nn = int((n-1)/2)
    a = np.asarray([[x**2 + y**2 for x in range(-nn,nn+1)] for y in range(-nn,nn+1)])
    return np.exp(-a/(2*sigma**2))

def median_filter_all_colours(im_small, window_size):
    """
    Applies a median filer to all colour channels
    """
    ims = []
    for d in range(3):
        im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
        ims.append(im_conv_d)

    im_conv = np.stack(ims, axis=2).astype("uint8")
    
    return im_conv

window_sizes = [9,17,33,65]
fig, axs = plt.subplots(nrows=3, ncols=len(window_sizes), figsize=(15,15));

# mean filter
for w, ax in zip(window_sizes, axs[0]):
    window = np.ones((w,w))
    window /= np.sum(window)
    ax.imshow(convolve_all_colours(im_small, window));
    ax.set_title("Mean Filter: window size: {}".format(w));
    ax.set_axis_off();
    
# gaussian filter
for w, ax in zip(window_sizes, axs[1]):
    window = make_gaussian_window(w,sigma=w)
    window /= np.sum(window)
    ax.imshow(convolve_all_colours(im_small, window));
    ax.set_title("Gaussian Filter: window size: {}".format(w));
    ax.set_axis_off();
    
# median filter
for w, ax in zip(window_sizes, axs[2]):
    ax.imshow(median_filter_all_colours(im_small, w));
    ax.set_title("Median Filter: window size: {}".format(w));
    ax.set_axis_off();
    


#  gradients in an image for each colour

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nn=100\nsobel_x = np.c_[\n    [-1,0,1],\n    [-2,0,2],\n    [-1,0,1]\n]\n\nsobel_y = np.c_[\n    [1,2,1],\n    [0,0,0],\n    [-1,-2,-1]\n]\n\nims = []\nfor d in range(3):\n    sx = convolve2d(im_small[:,:,d], sobel_x, mode="same", boundary="symm")\n    sy = convolve2d(im_small[:,:,d], sobel_y, mode="same", boundary="symm")\n    ims.append(np.sqrt(sx*sx + sy*sy))\n\nim_conv = np.stack(ims, axis=2).astype("uint8")\n\nplti(im_conv)')


# In[ ]:


im_smoothed = median_filter_all_colours(im_small, 71)

sobel_x = np.c_[
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

sobel_y = np.c_[
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
]

ims = []
for d in range(3):
    sx = convolve2d(im_smoothed[:,:,d], sobel_x, mode="same", boundary="symm")
    sy = convolve2d(im_smoothed[:,:,d], sobel_y, mode="same", boundary="symm")
    ims.append(np.sqrt(sx*sx + sy*sy))

im_conv = np.stack(ims, axis=2).astype("uint8")

plti(im_conv)


# blur only one colour channel at a tim

# In[ ]:


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15,5))

ax= axs[0]
ax.imshow(im_small)
ax.set_title("Original", fontsize=20)
ax.set_axis_off()

w=75
window = np.ones((w,w))
window /= np.sum(window)

ax= axs[1]
ims = []
for d in range(3):
    if d == 0:
        im_conv_d = convolve2d(im_small[:,:,d],window, mode="same", boundary="symm")
    else:
        im_conv_d = im_small[:,:,d]
    ims.append(im_conv_d)
ax.imshow(np.stack(ims, axis=2).astype("uint8"))
ax.set_title("Red Blur", fontsize=20)
ax.set_axis_off()

ax= axs[2]
ims = []
for d in range(3):
    if d == 1:
        im_conv_d = convolve2d(im_small[:,:,d], window, mode="same", boundary="symm")
    else:
        im_conv_d = im_small[:,:,d]
    ims.append(im_conv_d)
ax.imshow(np.stack(ims, axis=2).astype("uint8"))
ax.set_title("Blue Blur", fontsize=20)
ax.set_axis_off()

ax= axs[3]
ims = []
for d in range(3):
    if d == 2:
        im_conv_d = convolve2d(im_small[:,:,d], window, mode="same", boundary="symm")
    else:
        im_conv_d = im_small[:,:,d]
    ims.append(im_conv_d)
ax.imshow(np.stack(ims, axis=2).astype("uint8"))
ax.set_title("Green Blur", fontsize=20)
ax.set_axis_off()


# convert the image to greyscale, and find a threshold

# In[ ]:


def simple_threshold(im, threshold=128):
    return ((im > threshold) * 255).astype("uint8")

thresholds = [100,120,128,138,150]

fig, axs = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(20,5));
gray_im = to_grayscale(im)
                        
for t, ax in zip(thresholds, axs):
    ax.imshow(simple_threshold(gray_im, t), cmap='Greys');
    ax.set_title("Threshold: {}".format(t), fontsize=20);
    ax.set_axis_off();


#  threshold which minimises the inter pixel variance in the foreground and background

# In[ ]:


def otsu_threshold(im):

    pixel_counts = [np.sum(im == i) for i in range(256)]

    s_max = (0,-10)
    ss = []
    for threshold in range(256):

        # update
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])

        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0       
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # calculate 
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        ss.append(s)

        if s > s_max[1]:
            s_max = (threshold, s)
            
    return s_max[0]

t = otsu_threshold(gray_im)
plti(simple_threshold(gray_im, t), cmap='Greys')


# treshold each colour channel

# In[ ]:


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

c_ims = []
for c, ax in zip(range(3), axs):
    tmp_im = im[:,:,c]
    t = otsu_threshold(tmp_im)
    tmp_im = simple_threshold(tmp_im, t)
    ax.imshow(tmp_im, cmap='Greys')
    c_ims.append(tmp_im)
    ax.set_axis_off()


#  combine each channel into one image is to take the intersection of each thresholded colour channel

# In[ ]:


plti(c_ims[0] & c_ims[1] & c_ims[2], cmap='Greys')


# clusters in colour space

# In[ ]:


from sklearn.cluster import KMeans

h,w = im_small.shape[:2]
im_small_long = im_small.reshape((h * w, 3))
im_small_wide = im_small_long.reshape((h,w,3))

km = KMeans(n_clusters=3)

km.fit(im_small_long)

cc = km.cluster_centers_.astype(np.uint8)
out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))

plti(out)


# random popcolours

# In[ ]:


rng = range(4)

fig, axs = plt.subplots(nrows=1, ncols=len(rng), figsize=(15,5))
gray_im = to_grayscale(im)
                        
for t, ax in zip(rng, axs):
    rnd_cc = np.random.randint(0,256, size = (3,3))
    out = np.asarray([rnd_cc[i] for i in km.labels_]).reshape((h,w,3))
    ax.imshow(out)
    ax.set_axis_off()


# vectorize
#  grabbing the segment of the tumour

# In[ ]:


dog = np.asarray([cc[i] if i >0.5 else [0,0,0]
                  for i in km.labels_]).reshape((h,w,3))

plti(dog)


# In[ ]:


from skimage import measure

seg = np.asarray([(1 if i >0.5 else 0)
                  for i in km.labels_]).reshape((h,w))

contours = measure.find_contours(seg, 0.5, fully_connected="high")

simplified_contours = [measure.approximate_polygon(c, tolerance=1) for c in contours]

plt.figure(figsize=(5,10))

for n, contour in enumerate(simplified_contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    
plt.ylim(h,0)
plt.axes().set_aspect('equal')


# In[ ]:


from matplotlib.patches import PathPatch
from matplotlib.path import Path

# flip the path coordinates so it is consistent with matplotlib
paths = [[[v[1], v[0]] for v in vs] for vs in simplified_contours]

def get_path(vs):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vs)-2) + [Path.CLOSEPOLY]
    return PathPatch(Path(vs, codes))

plt.figure(figsize=(5,10))

ax = plt.gca()

for n, path in enumerate(paths):
    ax.add_patch(get_path(path))


    
plt.ylim(h,0)
plt.xlim(0,w)

plt.axes().set_aspect('equal')


# In[ ]:


class Polygon:
    def __init__(self, outer, inners):
        self.outer = outer
        self.inners = inners
        
    def clone(self):
        return Polygon(self.outer[:], [c.clone() for c in self.inners])

def crosses_line(point, line):
    """
    Checks if the line project in the positive x direction from point
    crosses the line between the two points in line
    """
    p1,p2 = line
    
    x,y = point
    x1,y1 = p1
    x2,y2 = p2
    
    if x <= min(x1,x2) or x >= max(x1,x2):
        return False
    
    dx = x1 - x2
    dy = y1 - y2
    
    if dx == 0:
        return True
    
    m = dy / dx

    if y1 + m * (x - x1) <= y:
        return False
    
    return True

def point_within_polygon(point, polygon):
    """
    Returns true if point within polygon
    """
    crossings = 0
    
    for i in range(len(polygon.outer)-1):
        line = (polygon.outer[i], polygon.outer[i+1])
        crossings += crosses_line(point, line)
        
    if crossings % 2 == 0:
        return False
    
    return True

def polygon_inside_polygon(polygon1, polygon2):
    """
    Returns true if the first point of polygon1 is inside 
    polygon 2
    """
    return point_within_polygon(polygon1.outer[0], polygon2)


    
def subsume(original_list_of_polygons):
    """
    Takes a list of polygons and returns polygons with insides.
    
    This function makes the unreasonable assumption that polygons can
    only be complete contained within other polygons (e.g. not overlaps).
    
    In the case where polygons are nested more then three levels, 
    """
    list_of_polygons = [p.clone() for p in original_list_of_polygons]
    polygons_outside = [p for p in list_of_polygons
                        if all(not polygon_inside_polygon(p, p2) 
                               for p2 in list_of_polygons 
                               if p2 != p)]
    
    polygons_inside = [p for p in list_of_polygons
                        if any(polygon_inside_polygon(p, p2) 
                               for p2 in list_of_polygons
                               if p2 != p)]
    
    for outer_polygon in polygons_outside:
        for inner_polygon in polygons_inside:
            if polygon_inside_polygon(inner_polygon, outer_polygon):
                outer_polygon.inners.append(inner_polygon)
                
    return polygons_outside
                
    for p in polygons_outside:
        p.inners = subsume(p.inners)
        
    return polygons_outside
polygons = [Polygon(p,[]) for p in paths]
subsumed_polygons = subsume(polygons)

def build_path_patch(polygon, **kwargs):
    """
    Builds a matplotlb patch to plot a complex polygon described
    by the Polygon class.
    """
    verts = polygon.outer[:]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts)-2) + [Path.CLOSEPOLY]

    for inside_path in polygon.inners:
        verts_tmp = inside_path.outer
        codes_tmp = [Path.MOVETO] + [Path.LINETO] * (len(verts_tmp)-2) + [Path.CLOSEPOLY]

        verts += verts_tmp
        codes += codes_tmp

    drawn_path = Path(verts, codes)

    return PathPatch(drawn_path, **kwargs)

def rnd_color():
    """Returns a randon (R,G,B) tuple"""
    return tuple(np.random.random(size=3))

plt.figure(figsize=(5,10))
ax = plt.gca()  
c = rnd_color()

for n, polygon in enumerate(subsumed_polygons):
    ax.add_patch(build_path_patch(polygon, lw=0, facecolor=c))
    
plt.ylim(h,0)
plt.xlim(0,w)
#ax.set_axis_bgcolor('k')
plt.axes().set_aspect('equal')
plt.show()


# In[ ]:


n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
        
fig.savefig('category_samples.png', dpi=300)


# # define three functions
# * list of classifiers
# * tree function
# * classification function

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.svm import LinearSVR,SVC
from sklearn.utils import check_array


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed
    
Classifiers = [
               #Perceptron(n_jobs=-1),
               #SVR(kernel='rbf',C=1.0, epsilon=0.2),
               #CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=4, method='sigmoid'),    
               #OneVsRestClassifier( SVC(    C=50,kernel='rbf',gamma=1.4, coef0=1,cache_size=3000,)),
               KNeighborsClassifier(10),
               DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=200),
               ExtraTreesClassifier(n_estimators=250,random_state=0), 
               OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10)) , 
              # MLPClassifier(alpha=0.510,activation='logistic'),
               LinearDiscriminantAnalysis(),
               #OneVsRestClassifier(GaussianNB()),
               #AdaBoostClassifier(),
               #GaussianNB(),
               #QuadraticDiscriminantAnalysis(),
               SGDClassifier(average=True,max_iter=100),
               XGBClassifier(max_depth=5, base_score=0.005),
               #LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1),
               LabelPropagation(n_jobs=-1),
               LinearSVC(),
               #MultinomialNB(alpha=.01),    
                   make_pipeline(
                    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
                    AdaBoostClassifier()
                ),

              ]


# In[ ]:


def treeprint(estimator):
    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()
    return


# In[ ]:


def klasseer(e_,mtrain,mtest,veld,idvld,thres,probtrigger):
    # e_ total matrix without veld, 
    # veld the training field
    #thres  threshold to select features
    label = mtrain[veld]
    # select features find most relevant ifo threshold
    #e_=e_[1:]
    clf = ExtraTreesClassifier(n_estimators=100)
    ncomp=int(e_.shape[1]/5)
    model = SelectFromModel(clf, prefit=True,threshold =(thres)/1000)
       # SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=ncomp, n_iter=7, random_state=42)
    #svd transformation is trying to throw away the noise
    e_=svd.fit_transform(e_)


       #tsne not used
    from sklearn.manifold import TSNE
    #e_=TSNE(n_components=3).fit_transform(e_)
    #from sklearn.metrics.pairwise import cosine_similarity
    
    #robustSVD not used
    #A_,e1_,e_,s_=robustSVD(e_,140)
    clf = clf.fit( e_[:len(mtrain)], label)
    New_features = model.transform( e_[:len(mtrain)])
    Test_features= model.transform(e_[-len(mtest):])
    #New_features =  e_[:len(mtrain)]
    #Test_features= e_[-len(mtest):] 
    pd.DataFrame(New_features).plot(x=0,y=1,c=mtrain[veld]+1,kind='scatter',title='classesplot',colormap ='jet')
    pd.DataFrame(np.concatenate((New_features,Test_features))).plot.scatter(x=0,y=1,c=['r' for x in range(len(mtrain))]+['g' for x in range(len(mtest))])    

    print('Model with threshold',thres/1000,New_features.shape,Test_features.shape,e_.shape)
    print('____________________________________________________')
    
    Model = []
    Accuracy = []
    for clf in Classifiers:
        #train
        fit=clf.fit(New_features,label)
        #if clf.__class__.__name__=='DecisionTreeClassifier':
            #treeprint(clf)
        pred=fit.predict(New_features)
        Model.append(clf.__class__.__name__)
        Accuracy.append(accuracy_score(mtrain[veld],pred))
        #predict
        sub = pd.DataFrame({idvld: mtest[idvld],veld: fit.predict(Test_features)})
        sub.plot(x=idvld,kind='kde',title=clf.__class__.__name__ +str(( mtrain[veld]==pred).mean()) +'prcnt') 
        sub2=pd.DataFrame(pred,columns=[veld])
        #estimate sample if  accuracy
        if veld in mtest.columns:
            print( clf.__class__.__name__ +str(round( accuracy_score(mtrain[veld],pred),2)*100 )+'prcnt accuracy versus unknown',(sub[veld]==mtest[veld]).mean() )
        #write results
        klassnaam=clf.__class__.__name__+".csv"
        sub.to_csv(klassnaam, index=False)
        if probtrigger:
            pred_prob=fit.predict_proba(Test_features)
            sub=pd.DataFrame(pred_prob)
    return sub


# # Split in train and test
# * make a numeric file without the index field since this fields has some 'forecasting value in the tree classifiers methods'
# * dx must be dropped too

# images=images.reset_index()
# images=(images.T.append(imagesL.drop('label',axis=1).T)).T
# images=(images.T.append(images8.drop('label',axis=1).T)).T
# images=(images.T.append(images8L.drop('label',axis=1).T)).T
# 
# images=(images.T.append(tile_df[['dx_type','age','sex','localization']].T)).T
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(images.drop(['label'],axis=1),images['label'], test_size=0.2, random_state=42)

# # tam tam, here are the results
# * suprisingly i can get a 100% correctforecast 
# * index leaks  information...
# * remove index
# 

# totaal=(X_train.append(X_test)).fillna(0)
# totaal=totaal.drop('index',axis=1).values
# 
# subx=klasseer(totaal,(X_train.T.append(y_train.T)).T,(X_test.T.append(y_test.T)).T,'label','index',0.01,False)
