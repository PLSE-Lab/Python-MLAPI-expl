#!/usr/bin/env python
# coding: utf-8

# ### Hu Moments for Image Recognition: Intro
# 
# Moments are very common features extracted from images to be used in pattern recogntion tasks, such as face recognition and shape retrieval. In this short paper, let's take a look at Hu Moments, undoubtedly the most important work of this domain. The main reference is the work **Visual Pattern Recognition by Moment Invariants** wrote by MK Hu in [1962](https://ieeexplore.ieee.org/document/1057692). Both the math and Python code are provided to better understanding of the subject, alongside more detailed explanations. Please feel free to help, share and improve this document.

# ### Raw Moments
# 
# We define _Image Moments_ as **the weighted average of all pixel intensities of a image**. Consider a binary image described by the function $I(x, y)$ with dimenisons $NxN$ pixels (any size is possible, not just square ones), we can calculate the raw moments using:
# 
# $$M_{pq} = \sum_{x=0}^{N-1}\sum_{y=0}^{N-1}x^py^qI(x, y)$$
# 
# The above expression shows a summation of all pixel intensities pondered by its location $(x, y)$ over the powers $p$ and $q$. In other words, image moments are values that carry both spatial and intensity information, e.g, **shape**. $p$ and $q$ are the weights of the horizontal and vertical dimensions, respectivelly. The sum $p+q$ is the _moment order_.

# ### Centroid Localization
# 
# We can use the raw moments to extract important information of an image. By doing $M_{00}$, we are accumulating the non-zero intensities. It's like describing the spatial information of the pixel "blob". Similary, doing for the $X$ and $Y$ dimensions ($M_{10}$ and $M_{01}$), one can pinpoint the centroid coordinates $(\bar{x}, \bar{y})$ of the blob by doing:
# 
# $$\bar{x} = \frac{M_{10}}{M_{00}}$$
# $$\bar{y} = \frac{M_{01}}{M_{00}}$$

# ### Translation Invariance
# 
# The centroid can be used to rewrite the raw moment equation to achieve the **translation invariant** momento $\mu_{pq}$:
# 
# $$\mu_{pq} = \sum_{x=0}^{N-1}\sum_{y=0}^{N-1}(x - \bar{x})^p(y - \bar{y})^qI(x, y)$$
# 
# Now, the relative spatial information of the centroid is being take in consideration, so no matter where the blob is localized the moments will be (roughly) the same.

# ### Scale Invariance
# 
# Scaling (change of size) is another very common transformation performed in images. This scaling can be uniform (the same in both dimensions) or non-uniform. Hu showed that you can relate the zero order translate invariant moment to get scale invariants $\eta_{pq}$:
# 
# $$\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{1 + \frac{p+q}{2}}}$$

# ### Hu Moments
# 
# We call **Hu Moments** the set of 7 values propesed by Hu in his 1962 work _Visual Pattern Recognition by Moment Invariants_:
# 
# $h_1 = \eta_{20} + \eta_{02}$
# 
# $h_2 = (\eta_{20} - \eta_{02})^2 + 4(\eta_{11})^2$
# 
# $h_3 = (\eta_{30} - 3\eta_{12})^2 + 3(\eta_{03} - 3\eta_{21})^2$
# 
# $h_4 = (\eta_{30} + \eta_{12})^2 + (\eta_{03} + \eta_{21})^2$
# 
# $h_5 = (\eta_{30} - 3\eta_{12})(\eta_{30} + \eta_{12})[(\eta_{30} + \eta_{12})^2 - 3(\eta_{03} + \eta_{21})^2] + (3\eta_{21} - \eta_{03})(\eta_{03} + \eta_{21})[3(\eta_{30} + \eta_{12})^2 - (\eta_{03} + \eta_{21})^2]$
# 
# $h_6 = (\eta_{20} - \eta_{02})[(\eta_{30} + \eta_{12})^2 - 7(\eta_{03} + \eta_{21})^2] + 4\eta_{11}(\eta_{30} + \eta_{12})(\eta_{03} + \eta_{21})$
# 
# $h_7 = (3\eta_{21} - \eta_{03})(\eta_{30} + \eta_{12})[(\eta_{30} + \eta_{12})^2 - 3(\eta_{03} + \eta_{21})^2] + (\eta_{30} - 3\eta_{12})(\eta_{03} + \eta_{21})[3(\eta_{30} + \eta_{12})^2 - (\eta_{03} + \eta_{21})^2]$

# ### Example: The Four Shapes Dataset
# 
# To illustrate the use of Hu Moments in pattern recognition, we will use the **Four Shapes Dataset**. This dataset contains 14970 samples of four classes: circles, squares, stars and triangles. You can get a free copy of the data in [Kaggle](https://www.kaggle.com/smeschke/four-shapes). We created the bellow animations using the first 400 samples of each class to showcase the diversity of examples:
# 
# |                Circles               |                Squares               |                 Stars                |               Triangles              |
# |:------------------------------------:|:------------------------------------:|:------------------------------------:|:------------------------------------:|
# | ![](https://i.imgur.com/gt2I0N1.gif) | ![](https://i.imgur.com/7HyJn4x.gif) | ![](https://i.imgur.com/VQAmDUK.gif) | ![](https://i.imgur.com/nzYtSmL.gif) |

# ### The Code
# 
# The code is written in Python 3.x and makes use of the following packages: ```opencv-python```, ```scikit-learn```, ```numpy```, ```pandas``` and ```matplotlib```. First, let's import the all the necessary packages:

# In[ ]:


import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC as SVM
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import NearestCentroid as NC
from sklearn.model_selection import train_test_split as data_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


# Next, we define two dictionaries. The first represents the relation between the class name literal and a numerical label. The second, the set of classifiers chosen for this example:

# In[ ]:


classes = {
	"circle"   : 1,
	"square"   : 2,
	"star"     : 3,
	"triangle" : 4
}

classifiers = {
	"NC"         : NC(),
	"LDA"        : LDA(),
	"QDA"        : QDA(),
	"SVM_linear" : SVM(kernel="linear"),
	"SVM_radial" : SVM(kernel="rbf")
}


# The first function we gonna create is responsible to extract the Moments of all samples, storing the class label as a eight column and save everything in a ```CSV``` file. Also, a simple logarithmic transformation is performed to equalize the orders of the moments, wich is a good pre-processing step when handling this type of feature.

# In[ ]:


def feature_extraction(data_file, dataset="../input/shapes/"):
	dump = []
	
	print("Extracting Hu moments...")
	
	for c, idx in classes.items():
		class_folder = dataset + "{}/".format(c)
		
		for f in os.listdir(class_folder):
			fpath = class_folder + f
			sample = int(f.replace(".png", ""))
			
			img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
			img = cv2.bitwise_not(img)
			hu = cv2.HuMoments(cv2.moments(img))
			
			for i in range(0, 7):
				hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))
			
			hu = hu.reshape((1, 7)).tolist()[0] + [sample, idx]
			dump.append(hu)
		
		print(c, "ok!")

	cols = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "sample", "class"]
	
	df = pd.DataFrame(dump, columns=cols)
	df.to_csv(data_file, index=None)
	
	print("Extraction done!")


# By executing this function and asking to store the results in ```hu_moments.csv```, we have:

# In[ ]:


data_file = "hu_moments.csv"

feature_extraction(data_file)


# The next function performs the classification. The feature vectors were split in training and test sets with 70/30 proportion, respectively, with a default number of test iterations set to 100. This function returns a dataframe with the results of each round.

# In[ ]:


def classification(data_file, rounds=100, remove_disperse=[]):
	df = pd.read_csv(data_file)
	df = df.drop(["sample"], axis=1)
	
	if remove_disperse:
		df = df.drop(remove_disperse, axis=1)
	
	X = df.drop(["class"], axis=1)
	y = df["class"]
	
	ans = {key: {"score" : [], "sens" : [], "spec" : []}
	       for key, value in classifiers.items()}
	
	print("Classifying...")
	
	for i in range(rounds):
		X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.3)
		
		for name, classifier in classifiers.items():
			scaler = StandardScaler()
			scaler.fit(X_train)
			X_train = scaler.transform(X_train)
			X_test = scaler.transform(X_test)
			
			classifier.fit(X_train, y_train)
			score = classifier.score(X_test, y_test)
			
			ans[name]["score"].append(score)
		
	print("Classification done!")
	
	return ans


# The next line executes it and stores the result in ```ans```:

# In[ ]:


ans = classification(data_file)


# Let's visualize the classification performance in a more frendly way using the ```summary()``` function:

# In[ ]:


def sumary(ans, title="Summary"):
	size = 70
	separator = "-"
	
	print(separator*size)
	print("SUMARY: {}".format(title))
	print(separator*size)
	print("CLASSIF\t\tMEAN\tMEDIAN\tMINV\tMAXV\tSTD")
	print(separator*size)
	
	for n in ans:
		m = round(np.mean(ans[n]["score"])*100, 2)
		med = round(np.median(ans[n]["score"])*100, 2)
		minv = round(np.min(ans[n]["score"])*100, 2)
		maxv = round(np.max(ans[n]["score"])*100, 2)
		std = round(np.std(ans[n]["score"])*100, 2)
		
		print("{:<16}{}\t{}\t{}\t{}\t{}".format(n, m, med, minv, maxv, std))
	
	print(separator*size)
	print()


# In[ ]:


sumary(ans)


# ### Conclusion
# 
# As we can see, all classifiers performed almost perfectly in this task, with the only exception being the NC. The SVM classifier configured with Linear Kernel had the highest mean accuracy and lowest standard deviation, making it the best choice in this simple application. With simple math involved and a rather fast extraction/classification time, the Hu Moments are a good benchmark for visual pattern recognition, as well as a good entry example for those who are starting with computer vision.
