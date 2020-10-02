#!/usr/bin/env python
# coding: utf-8

# # Feature Generation and Analysis with Facial Landmarks and Pretrained Model Embeddings
# 
# <div id="intro"><hr></div>
# 
# ## Introduction
# 
# In this kernel I generate a large number of features using Dlib's facial landmark detector and the pretrained Facenet and VGGFace models in order to identify a smaller group of highly influential features. After using a series of simple transformations to create new features, I use a logistic regression model with a l1 regularization parameter to indentify the most useful features. Feel free to skip to the conclusion where I discuss the results of the kernel and to use any of the new features in your own models. The kernel is organized as follows.
# 
# * [Introduction](#intro)
# * [Feature Generation](#featgen)
# * [Feature Selection](#featsel)
# * [Submission](#sub)
# * [Conclusion](#conc)
# 
# <div id="featgen"><hr></div>
# 
# ## Feature Generation
# 
# ### Facial Landmarks
# 
# The first set of features will stem from facial landmark coordinates calculated using Dlib's face detector and facial landmark predictor. I initialized the predictor with weights pretrained on the iBUG 300-W dataset, available for research purposes only, which can be downloaded [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). In order to speed up the execution and elevate the readability of this kernel I have calculated the coordinates for every image in the training and testing data in a seperate kernel. For more information on DLib and the example code that I used, visit the [facial landmark tutorial from pyimagesearch](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/). Beyond the raw coordinates, I will create features based on the distance to the average point, distances from points on a person's face, and distances from points across faces.
# 
# #### Raw Coordinates
# 
# DLib's facial landmark predictor generates 68 points across a person's face. There are 9 specific landmarks: the jawline, left eyebrow, right eyebrow, nasal ridge, nasal tip, left eye, right eye, outer lips, and inner lips, with a variable number of points for each. I have named each point in a landmark-point-axis format (Ex: jaw0X, lipOut11Y), with point numbers in the order shown below (image courtesy of [ibug](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)):
# <img src="https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg" width="50%"/>
# <br>
# I have already calculated the landmark coordinates for every image so here I load the required packages and read in the csv file. With 68 points we have 136 features per face and 272 per pair.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from itertools import combinations
import cv2
import os
import random
from time import time
from math import sqrt
from tqdm import tqdm
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
from scipy.spatial import distance
import matplotlib.pyplot as plt
from seaborn import kdeplot


# In[ ]:


prerun_data = pd.read_csv("../input/calculating-dlib-facial-landmarks-for-rfiw-images/facial_landmark_coordinates.csv")
prerun_data.head()


# #### Distance to Average Point
# 
# Here I find the average X and Y values for each of the 68 points that have been calculated by DLib's facial landmark predictor. Then for every face I find the distance of their points to the corresponding average. An average for each point leads to 68 new features per face and 136 per pair.

# In[ ]:


point_names = ["jaw"+str(i) for i in range(17)]+["browL"+str(i) for i in range(5)]+            ["browR"+str(i) for i in range(5)]+["nRidge"+str(i) for i in range(4)]+            ["nTip"+str(i) for i in range(5)]+["eyeL"+str(i) for i in range(6)]+            ["eyeR"+str(i) for i in range(6)]+["lipOut"+str(i) for i in range(12)]+            ["lipIn"+str(i) for i in range(8)]

for i in point_names:
    x_mean = prerun_data[i +'X'].mean()
    y_mean = prerun_data[i +'Y'].mean()
    col_name = "d2Avg(" + i + ")"
    prerun_data[col_name] = np.sqrt(((prerun_data[i+'X']-x_mean)**2)+((prerun_data[i+'Y']-y_mean)**2))
    
prerun_data.iloc[0:5,137:205]


# #### Point to Point Distance
# Ideally I would calculate the distance between all possible pairs of the 68 points, but that would result in 2278 features, too many for the scope of this kernel. I also believe that many of those features would be trivial. For that reason I will select a number of points that I think will provide the most valuable information. Those points are the tips and bottom of the jawline, ends of both eyebrows, corners of the eyes, the top of the nasal bridge, the ends and middle of the nasal tip, and the top, bottom, and corners of the outer lips, for a total of 19 points. 19 points creates 171 combinations of two, meaning 171 features per face and 342 per pair.

# In[ ]:


features = ['jaw0','jaw8','jaw16','browL0','browL4','browR0',
            'browR4','nRidge3','nTip0','nTip2','nTip4',
            'eyeL0','eyeL3','eyeR0','eyeR3','lipOut0','lipOut3',
            'lipOut6','lipOut9']
features = list(combinations(features, 2))

for i in range(0, len(features)):
    name1 = features[i][0]
    name2 = features[i][1]
    col_name = "dist(" + name1 + "," + name2 + ")"
    
    x1 = prerun_data[name1+'X']
    y1 = prerun_data[name1+'Y']
    x2 = prerun_data[name2+'X']
    y2 = prerun_data[name2+'Y']
    
    prerun_data[col_name] = np.sqrt(((x1-x2)**2) + ((y1-y2)**2))
    
prerun_data.iloc[0:5,205:376]


# ### Pretrained Models
# 
# #### Facenet Embeddings
# 
# Facenet is a powerful facial recognition and clustering model that has been trained on the Labeled Faces in the Wild dataset and has proven to be effective in this competition. Based on the popular kernel from Khoi Nguyen, I use the pretrained facenet model to calculate embeddings for each image. The computed embeddings are 128-dimensional vectors, meaning 128 features for each face and 256 for each pair. 

# In[ ]:


model_path = '../input/facenet-keras/facenet_keras.h5'
facenet_model = load_model(model_path)


# In[ ]:


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_and_align_images(filepaths, image_size = 160):
    
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)
            
    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=512):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size]))
        pd.append(facenet_model.predict_on_batch(aligned_images))
    embs = l2_normalize(np.concatenate(pd))

    return embs

print("Helper functions compiled succesfully.")


# In[ ]:


facenet_embs = calc_embs(prerun_data["path"])
facenet_embs = pd.DataFrame(facenet_embs, columns=["facenet_"+str(i) for i in range(128)])
prerun_data = pd.concat([prerun_data, facenet_embs], axis=1)
prerun_data.iloc[0:5,376:504]


# ### Facenet Embedding Distance
# Many high-scoring submissions have been as simple as finding the distance between faces based on embeddings. Here I do the same thing, but calculating the distance between two images requires the embeddings of both images, meaning that we need to create our training pairs first. I create the pairs in the 'Feature Selection' section of the kernel, so I will just write a function that can be used for both model training and generating the submissions. This will only create 1 new feature per pair.

# In[ ]:


def calc_distance(embs_img1, embs_img2):
    dists = distance.euclidean(embs_img1, embs_img2)
    return dists


# ### VGGFace Embeddings and Distance
# 
# The VGGFace model has been another popular model in this competition and seen great success, such as [this kernel](https://www.kaggle.com/ateplyuk/vggface-baseline-in-keras/data) from Alexander Teplyuk which was my inspiration for much of this section. Just like the Facenet model, I calculate the embeddings for each face as well as the distance between image pairs. I have put the pretrained model into a private Kaggle dataset, but you can easily download it from [rcmalli's github repository](https://github.com/rcmalli/keras-vggface/releases), as I did. The distance function created in the Facenet section can be used for the VGGFace model as well. The embeddings are 512-dimensional vectors, meaning 512 features per face and 1024 per pair, as well as another single feature per pair from the embedding distance.

# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')


# In[ ]:


from keras_vggface.vggface import VGGFace
vggface_model = VGGFace(include_top=False, input_shape=(160, 160, 3), pooling='avg')


# In[ ]:


def calc_embs_vggface(filepaths, margin=10, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = prewhiten(load_and_align_images(filepaths[start:start+batch_size]))
        pd.append(vggface_model.predict_on_batch(aligned_images))
    embs = l2_normalize(np.concatenate(pd))

    return embs

print("Helper functions successfully compiled.")


# In[ ]:


vggface_embs = calc_embs_vggface(prerun_data["path"])
vggface_embs = pd.DataFrame(vggface_embs, columns=["vggface_"+str(i) for i in range(512)])
prerun_data = pd.concat([prerun_data, vggface_embs], axis=1)
prerun_data.iloc[0:5,505:1017]


# <div id="featsel"><hr></div>
# 
# ## Feature Selection
# 
# We have a lot of features and lot of observations. Cutting down the computational time and power needed to train a model may be essential for industry applications or to beat Kaggle's 6 hour runtime limit. Using fewer observations is one way to accomplish this, but doing so may cause models to overfit. Trimming features, however, can be done in such a way that model performance is largely unaffected, with much lower computational requirements. A smaller, concentrated group of powerful features may be just as effective as a larger, noisier group. I will use Lasso regularization within a logistic regression model to identify meaningful features. According to the [Stanford Statistics Department](http://statweb.stanford.edu/~tibs/lasso.html):
# > Lasso is a shrinkage and selection method for linear regression. It minimizes the usual sum of squared errors, with a bound on the sum of the absolute values of the coefficients. 
# 
# I have chosen Lasso because it tends to prefer solutions with fewer non-zero features compared to other regularization techniques like Ridge or Elastic-Net. Before we can train our model we need to complete just a few data-cleaning tasks.
# 
# ### Scaling
# 
# Logistic regression models benefit greatly from normalized data, so it is neccessary that I scale the features I have created. Most of the features are already fairly normal, as you can see below, so I will use a basic z-score scaler in scikit's StandardScaler. 

# In[ ]:


f = plt.figure(figsize=(13,3))
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)

ax1.hist(prerun_data['browL2X'].dropna(), 25, alpha=0.75)
ax1.set_title('Left Brow, Point 2, X Coord')
ax2.hist(prerun_data['dist(jaw0,jaw8)'].dropna(),25, alpha=0.75)
ax2.set_title('Dist. from Jaw0 to Jaw8')
ax3.hist(prerun_data['facenet_12'].dropna(), 25, alpha=0.75)
ax3.set_title('Facenet Embedding 12')
plt.show()


# In[ ]:


scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(prerun_data.iloc[:,1:])
scaled_data = pd.DataFrame(scaled_data, columns = prerun_data.columns[1:])
scaled_data.index = prerun_data.index
scaled_data.insert(loc=0, column="path", value=prerun_data["path"])


# ### Image Pairing
# 
# The competition provided a list of individuals who have a kinship relation, but obviously more work needs to be done before our data are ready to be modeled. First, I format the given relationships and then generate an equal number of non-kinship pairs by randomly selecting individuals from different families. This method has a few flaws, including the fact that it will never create a pair of individuals from within a family who do not have a kinship relation. If I were attempting to compete for first place I would devise a more thorough data-cleaning pipeline, but my aim is just to provide a high level look at what features seem to be powerful so I will stick with this more simplistic method for now.

# In[ ]:


random.seed(6242)

train_folder = '../input/recognizing-faces-in-the-wild/train/'
family_names = sorted(os.listdir(train_folder))
kin_pairs = pd.read_csv('../input/recognizing-faces-in-the-wild/train_relationships.csv')

kin_files = []
for i in range(0, len(kin_pairs.index)):
    if os.path.exists(train_folder+kin_pairs.iloc[i,0]) and os.path.exists(train_folder+kin_pairs.iloc[i,1]):
        pair = [kin_pairs.iloc[i,0], kin_pairs.iloc[i,1], 1]
        kin_files.append(pair)
for i in range(0, len(kin_files), 2):
    x = kin_files[i][0]
    kin_files[i][0] = kin_files[i][1]
    kin_files[i][1] = x
        
random_files = []
for i in range(len(kin_files)):
    x, y = 0, 0
    while x == y:
        x = random.randint(0,469)
        y = random.randint(0,469)

    fam1 = family_names[x]
    fam2 = family_names[y]

    fam1_folder = train_folder + fam1 + '/'
    fam1_members = sorted(os.listdir(fam1_folder))
    m = random.randint(0,len(fam1_members)-1)
    pers1 = fam1_members[m]

    fam2_folder = train_folder + fam2 + '/'
    fam2_members = sorted(os.listdir(fam2_folder))
    n = random.randint(0,len(fam2_members)-1)
    pers2 = fam2_members[n]

    pair = [fam1+'/' +pers1, fam2+'/'+pers2, 0]
    random_files.append(pair)

image_files = kin_files + random_files
print("There are ", len(image_files), " pairs of individuals in the training data")


# Now that I have the pairs of individuals, I collect the features for each image into a single observation, effectively doubling the number of features. Individuals may have more than one picture, so the training data has many more observations than the number of pairs I created above. Within the code below I also calculate the Facenet and VGGFace embedding distance for each image pair and scale the data using the StandardScaler. Another issue with the data is that Dlib's facial recognition model occasionaly cannot detect a face in the images, meaning some pairs will have to be discarded as the logistic regression model cannot handle missing values.

# In[ ]:


print("There are", pd.isnull(scaled_data).sum()[1], "incorrectly detected images.")  


# In[ ]:


col_names = []
images_per_person = 3
landmark_data = pd.DataFrame(
    columns = [name + "_A" for name in scaled_data.columns[1:]] + \
        [name + "_B" for name in scaled_data.columns[1:]] + \
        ["dist(Facenet)", "dist(VGGFace)", "Kin"])
emb_cols = ["facenet_" + str(i) for i in range(128)]
emb_cols_vggface = ["vggface_" + str(i) for i in range(512)]

for i in range(len(image_files)):
    rowEntry = []
    pers1 = image_files[i][0]
    pers2 = image_files[i][1]
    images1 = sorted(os.listdir(train_folder+pers1+'/'))
    images2 = sorted(os.listdir(train_folder+pers2+'/'))
    num_imgs1 = len(images1)
    num_imgs2 = len(images2)
    
    if num_imgs1 == 0:
        continue
    elif num_imgs1 > images_per_person:
        num_imgs1 = images_per_person
        
    if num_imgs2 == 0:
        continue
    elif num_imgs2 > images_per_person:
        num_imgs2 = images_per_person
    
    for j in range(0,num_imgs1):
        for k in range(j, num_imgs2):
            img1 = train_folder+pers1+'/'+images1[j]
            img2 = train_folder+pers2+'/'+images2[k]
            points1 = scaled_data.loc[scaled_data['path'] == img1]
            points2 = scaled_data.loc[scaled_data['path'] == img2]
            points1 = points1.values.tolist()[0][1:]
            points2 = points2.values.tolist()[0][1:]
            
            embs_img1 = prerun_data.loc[prerun_data["path"]==img1,emb_cols]
            embs_img2 = prerun_data.loc[prerun_data["path"]==img2,emb_cols]
            emb_dist = calc_distance(embs_img1,embs_img2)
            
            embs_img1_vgg = prerun_data.loc[prerun_data["path"] == img1, emb_cols_vggface]
            embs_img2_vgg = prerun_data.loc[prerun_data["path"] == img2, emb_cols_vggface]
            emb_dist_vggface = calc_distance(embs_img1_vgg, embs_img2_vgg)
            
            rowEntry = points1 + points2 + [emb_dist, emb_dist_vggface] + [image_files[i][2]]
            if len(rowEntry) == len(landmark_data.columns):
                landmark_data.loc[len(landmark_data)] = rowEntry

dist_scaler = preprocessing.StandardScaler()
distance_col = np.asarray(landmark_data["dist(Facenet)"].tolist()).reshape(-1,1)
scaled_dists = dist_scaler.fit_transform(distance_col)
landmark_data["dist(Facenet)"] = scaled_dists

vggface_dist_scaler = preprocessing.StandardScaler()
distance_col = np.asarray(landmark_data["dist(VGGFace)"].tolist()).reshape(-1,1)
scaled_dists = vggface_dist_scaler.fit_transform(distance_col)
landmark_data["dist(VGGFace)"] = scaled_dists

print("We now have", len(landmark_data.index), "pairs of faces.")


# Due the errors with the Dlib's facial recognition, a number of the training pairs have missing values. Let's remove those now.

# In[ ]:


lm_size = len(landmark_data.index)
landmark_data = landmark_data.dropna()
print("There are now", len(landmark_data.index), "pairs left, a loss of", lm_size-len(landmark_data.index))


# ### Modeling
# 
# Our data is finally wrangled and it is time to train the logistic regression model. Training sklearn's logistic regression model will provide coefficients for each feature that will estimate how important each individual feature is. To demonstrate the benefits of using a concentrated group of features, I will train another model with the top-250 most influential coefficients and post the results in the comments at the bottom of the kernel.

# In[ ]:


start = time()
logReg = LogisticRegression(penalty="l1", solver="liblinear", random_state=8119)
logReg.fit(landmark_data.drop("Kin", axis=1), landmark_data["Kin"].astype("int"))
end = time()

all_coord_time = end - start


# In[ ]:


print("That took about", int(all_coord_time/60), "minutes.", sum(logReg.coef_.reshape(len(landmark_data.columns)-1,) == 0), "coefficients were reduced to zero.")
print("Lets train it again with only 250 features and see if we can reduce that time.")


# Now that we have an indication of what features are useful, lets train the model again with only the top 250 features. Hopefully we will only have a limited drop in score for a large gain in speed!

# In[ ]:


coefficients = logReg.coef_.reshape(len(landmark_data.columns)-1,)
k = len(coefficients) - 250
drop_cols = np.argpartition(coefficients, k)
drop_cols = drop_cols[:k]

top250_data = landmark_data.drop(landmark_data.columns[(drop_cols)], axis=1)

start = time()
logReg250 = LogisticRegression(penalty="l1", solver="liblinear", random_state=8119)
logReg250.fit(top250_data.drop("Kin", axis=1), top250_data["Kin"].astype("int"))
end = time()

coord_time_250 = end - start


# In[ ]:


print("That took about", int(coord_time_250/60), "minutes, much faster. Let's see how it scores.")


# <div id="sub"><hr></div>
# 
# ## Submission
# 
# Now that the models have been trained a competition submission can be created. This code is very similar to that in the feature selection section except that the pairs come from the provided sample submission file.

# In[ ]:


test_pairs = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')
test_pairs_t250 = test_pairs.copy()

test_folder = '../input/recognizing-faces-in-the-wild/test/'
detection_errors = 0

for i in range(len(test_pairs)):
    
    rowEntry = []
    missing_vals = 0
    
    pair = test_pairs.iloc[i,0]
    
    pers1 = pair.split('-')[0]
    pers2 = pair.split('-')[1]
    
    pers1 = test_folder+pers1
    points1 = scaled_data.loc[scaled_data['path'] == pers1]
    points1 = points1.drop('path', axis=1)
    points1.reset_index(drop=True, inplace=True)
    if points1.isnull().values.any():
        missing_vals = 1
    
    pers2 = test_folder+pers2
    points2 = scaled_data.loc[scaled_data['path'] == pers2]
    points2 = points2.drop('path', axis=1)
    points2.reset_index(drop=True, inplace=True)
    if points2.isnull().values.any():
        missing_vals = 1
    
    embs_img1 = prerun_data.loc[prerun_data["path"] == pers1, emb_cols]
    embs_img2 = prerun_data.loc[prerun_data["path"] == pers2, emb_cols]
    emb_dist = calc_distance(embs_img1, embs_img2)
    emb_dist = (emb_dist - dist_scaler.mean_[0])/dist_scaler.scale_[0]
    emb_dist = pd.DataFrame([emb_dist], columns=["dist(Facenet)"])
    
    embs_img1_vgg = prerun_data.loc[prerun_data["path"] == img1, emb_cols_vggface]
    embs_img2_vgg = prerun_data.loc[prerun_data["path"] == img2, emb_cols_vggface]
    emb_dist_vggface = calc_distance(embs_img1_vgg, embs_img2_vgg)
    emb_dist_vggface = (emb_dist_vggface - vggface_dist_scaler.mean_[0])/vggface_dist_scaler.scale_[0]
    emb_dist_vggface = pd.DataFrame([emb_dist_vggface], columns=["dist(VGGFace)"])
    
    rowEntry = pd.concat([points1, points2, emb_dist, emb_dist_vggface], axis=1, ignore_index = True)
    rowEntry.columns = [name + "_A" for name in scaled_data.columns[1:]] +                         [name + "_B" for name in scaled_data.columns[1:]] +                         ["dist(Facenet)", "dist(VGGFace)"]
    
    rowEntry_t250 = rowEntry.drop(rowEntry.columns[(drop_cols)], axis=1)
        
    if missing_vals == 0:
        test_pairs.iloc[i,1] = logReg.predict_proba(rowEntry)[0][0]
        test_pairs_t250.iloc[i,1] = logReg250.predict_proba(rowEntry_t250)[0][0]
    else:
        detection_errors += 1
        test_pairs.iloc[i,1] = 0.5
        test_pairs_t250.iloc[i,1] = 0.5
    
if(logReg.classes_[0] == 0): # Probabilities depend on the order of the classes in the model
    test_pairs["is_related"] = [1-x for x in test_pairs["is_related"]]
    test_pairs_t250["is_related"] = [1-x for x in test_pairs_t250["is_related"]]
    
print(detection_errors, "pairs have been lost due to errors with dLib's facial detection.")
print("All missing cases (", round(detection_errors/len(test_pairs.index),3),"%) have been set to 0.5.")


# In[ ]:


test_pairs.to_csv('submission_file.csv', index=False)
test_pairs_t250.to_csv('submission_file_t250.csv', index=False)


# <div id="conc"><hr></div>
# 
# ## Conclusion
# 
# ### Results
# 
# There are a number of simple improvements that could be made to this kernel to improve its standings in the competition. From the beginning, errors with Dlib's pretrained models have forced me to throw away valuable data and blindly guess on around 1% of the test cases. By implementing a more accurate facial recognition model, that problem could be solved. I noted earlier that my data collection process was flawed; a more robust method could be devised. I declined to focus on these issues and others because they effect more the competition score rather than my high level analysis of feature generation. With this kernel I seek not to win the competition but to learn and create.
# 
# Regardless of the overall competition score, the question is: which features were strong and worth investigating further and which features are of little use. Here I will show and discuss general trends and only a few of the top features, but I will create a .csv file with all of the features and their respective coefficients for further examination.
# 
# #### Coefficient Spread
# 
# I examine the features by dividing them into three groups: those that were reduced to zero by Lasso, those in the top-250 that were selected to train on the secondary model, and those in the middle that contribute only weakly. I define the 'influence' or 'strength' of a feature by the absolute value of its coefficient because the coefficient determines the feature's effect on the model's prediction. Below I create those three divisions as well as write a .csv file with all the features and coefficients for your further reference.

# In[ ]:


feature_coeffs = pd.DataFrame()
feature_coeffs["feature"] = landmark_data.columns[:-1]
feature_coeffs["coefficient"] = logReg.coef_[0]
feature_coeffs.to_csv('feature_coefficients.csv', index=False)

sorted_coeffs = feature_coeffs.reindex(feature_coeffs.coefficient.abs().sort_values(ascending=False).index)

zero_coeffs = feature_coeffs[feature_coeffs["coefficient"]==0]
best_coeffs = sorted_coeffs.iloc[0:250,:].reset_index(drop=True)
middle_coeffs = sorted_coeffs.iloc[250:len(feature_coeffs.index)-len(zero_coeffs.index),:]


# The plots below show the distribution of coefficients across the three catagories for each type of feature. The middle-ground group has 1384 coefficients while the top has 250 and the bottom has 398, so we should expect each graph to have a large middle bar. That is clearly not the case, however. While the 'Distance to the Avergae Point' features and VGGFace embeddings have largely mediocre strengths, the raw coordinates and point to point distances have a lot of bad features and a lot of strong features. The Facenet embeddings mostly fall in the bad and middle sections.

# In[ ]:


objects = ("Zero", "Middle", "Top-250")

f = plt.figure(figsize=(13,8))
ax = f.add_subplot(231)
ax2 = f.add_subplot(232)
ax3 = f.add_subplot(233)
ax4 = f.add_subplot(234)
ax5 = f.add_subplot(235)

data = [sum(["X_" in name or "Y_" in name for name in zero_coeffs["feature"]]),
        sum(["X_" in name or "Y_" in name for name in middle_coeffs["feature"]]),
        sum(["X_" in name or "Y_" in name for name in best_coeffs["feature"]])]
ax.bar(objects, data, align='center', alpha=0.75)
ax.set_title('Raw Coordinates')

data = [sum(["d2Avg(" in name for name in zero_coeffs["feature"]]),
        sum(["d2Avg(" in name for name in middle_coeffs["feature"]]),
        sum(["d2Avg(" in name for name in best_coeffs["feature"]])]
ax2.bar(objects, data, align='center', alpha=0.75)
ax2.set_title('Dist. to Average Point')

data = [sum(["dist(" in name for name in zero_coeffs["feature"]]),
        sum(["dist(" in name for name in middle_coeffs["feature"]]),
        sum(["dist(" in name for name in best_coeffs["feature"]])]
ax3.bar(objects, data, align='center', alpha=0.75)
ax3.set_title('Point to Point Dist.')

data = [sum(["facenet_" in name for name in zero_coeffs["feature"]]),
        sum(["facenet_" in name for name in middle_coeffs["feature"]]),
        sum(["facenet_" in name for name in best_coeffs["feature"]])]
ax4.bar(objects, data, align='center', alpha=0.75)
ax4.set_title('Facenet Embeddings')

data = [sum(["vggface_" in name for name in zero_coeffs["feature"]]),
        sum(["vggface_" in name for name in middle_coeffs["feature"]]),
        sum(["vggface_" in name for name in best_coeffs["feature"]])]
ax5.bar(objects, data, align='center', alpha=0.75)
ax5.set_title('VGGFace Embeddings')


# Given the success of pretrained-models in the competition, it is odd that the calculated embeddings perform so poorly. That said, collecting all the embeddings into a single face-to-face distance feature proved to be more successful. The Facenet embedding distance is one of the top features, and significantly more influential than the VGGFace distance feature. This discrepancy is especially interesting when you consider how poorly the Facenet embeddings performed on their own.

# In[ ]:


feature_coeffs.iloc[2030:2032,:]


# #### The Best of The Best
# 
# Here are the features with a coefficient greater than 1:

# In[ ]:


best_coeffs[abs(best_coeffs["coefficient"]) > 1]


# Over half of the best features are point-to-point distances and most of the rest are raw coordinates. The odd feature out is the Facenet embedding distance. The presence of point-to-point features does not suprise me, as I chose points that had a high likelihood of providing valuable information, but having such a large presence of raw coordinates is intriguing, as I intended for them to be the building blocks of more sophisticated features rather than intergral parts of the model. This odd collection of features demonstrates how differently machine learning models think from humans.
# 
# If you read through my whole kernel or even skipped to the end, I appreciate and thank you for your interest. Feel free to use any of the features I created or try to create some of your own! This was my first major kernel on Kaggle and would love your feedback in the comments or with an upvote.
# 
# ### Sources
# 
# * [Facial landmark tutorial from pyimagesearch](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
# * [ibug](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
# * [Original Facenet Article](https://arxiv.org/abs/1503.03832)
# * [Facenet Keras Dataset from Khoi Nguyen](https://www.kaggle.com/suicaokhoailang/facenet-keras)
# * [Facenet Baseline in Keras from Khoi Nguyen](https://www.kaggle.com/suicaokhoailang/facenet-baseline-in-keras-0-749-lb)
# * [Alexander Teplyuk's VGGFace Kernel](https://www.kaggle.com/ateplyuk/vggface-baseline-in-keras/data)
# * [rcmalli's github repository](https://github.com/rcmalli/keras-vggface/releases)
# * [Feature Scaling from Ben Alex Keen](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
