#!/usr/bin/env python
# coding: utf-8

# Nama : Fachry Muhammad  |
# NRM   : 1313617019  |
# Prodi   : Ilmu Komputer  | Lecture3 Digit Recoqnition Experiment
# 
# 
# 
# # **DISCLAIMER**
# The Dataset is not mine, and mostly copied from this [tutorial](http://https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification) for my reference. I intended to do modification later to the tutorial, so please permit me for using it. The purpose of this note book is to learn and ML experiment of Digit Recognition on Kaggle.com. 
# 
# 
# **Welcome to my first Kernel on Kaggle.com**
# 
# This kernel is an experiment to learn how to run ML experiment of Digit Recognition on Kaggle.com. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **STEP 1 : Lybrary Import**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
#%matplotlib inline
print('Import complete.')


# **STEP 2: Loading the data**
# 
# inputting some known test data.
# 
# **For the sake of time, we're only using 5000 images. You should increase or decrease this number to see how it affects model training.**

# In[ ]:


# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:] #return a DataFrame consisted of rows indexed in 0:5000, and columns indexed from index 1.
labels = labeled_images.iloc[0:5000,:1] #return a DataFrame consisted of rows indexed in 0:5000, and columns indexed from beginning to index 1
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)


# In[ ]:


# Data check
print(labeled_images.describe())
print(images.describe())
print(labels.describe())


# # **Question 1**
# 
# Notice in the above we used _images.iloc?, can you confirm on the documentation? what is the role?
# 
# **Q1 Answer**
# 
# According to the [Documentation](http://https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html) . 
# Basically .iloc is an index based location selector, it's used to select parts of dataframe. .iloc will raise IndexError if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing (this conforms with python/numpy slice semantics). .iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
# 
# Where in this case;
# 1. > images = labeled_images.iloc[0:5000,1:]
#  
#  image will be a DataFrame consisted of rows indexed in 0:5000, and columns indexed from index 1.
#   
# 2. > labels = labeled_images.iloc[0:5000,:1] 
#  
# And label a DataFrame consisted of rows indexed in 0:5000, and columns indexed from beginning to index 1
# 

# **STEP 3: Viewing an Image**
# Since the image is currently one-dimension, we load it into a numpy array and reshape it so that it is two-dimensional (28x28 pixels)
# Then, we plot the image and label with matplotlib
# **You can change the value of variable i to check out other images and labels.**

# In[ ]:


i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


# # **Question 2**
# 
# Now plot an image for each image class
# 
# **Q2 Answer**
# 
# to plot an image for each of class in every digit number (0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 and 9).

# In[ ]:


number = np.unique(train_labels) # np.unique is for returning the unique element of given array
print (number)


index = [0,0,0,0,0,0,0,0,0,0]    # declare temporary index for searching the result in train_labels, 
                                 # this index will eventualy be replace with the number location in every class from 0-9

# In order to find said Index, we use loops in every row in train_labels 

for i in range (len(train_labels)):
    label = train_labels.iloc[i].label # label variable contain number class, in order to find the index of i
    index[label] = i                   # (iloc[i] serach the row of i, and .label search the coloumn of label)
    
for i in index:     # Now we print the image for every column of number
    plt.figure()
    img=train_images.iloc[i].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])


# Now plot the histogram within img
# 
# we used plt.hist() for generate the histogram for the amount (frequency) of number occurence on the index of i 
# 
# 
# **STEP 4: Examining the Pixel Values**
# 
# Note that these images aren't actually black and white (0,1). They are gray-scale (0-255).
# A histogram of this image's pixel values shows the range.

# In[ ]:


#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])


# # **Question 3**
# 
# Can you check in what class does this histogram represent?. How many class are there in total for this digit data?. How about the histogram for other classes
# 
# **Q3 Answer **
# 
# TL;DR  basically it represent the amount of number/coloured pixel( represented in value of 0-255) in the array. with the total class of 10 number(0-9)
# 
# The histogram contain information about the classes of each pixel values. The "X" direction represent the frequency, and "Y" represent for  each class from one of the sample number 6 class as the 2nd data in train data. With that information and the information that the pixel value is grayscale betwwen black (0) and white (255), we can determine tahat there are not only a lot of black blank spaces but also small portion of white/gray-ish pixels  ( with 0< value < 255) which is important for learning the stroke of handwritten digit numbers in order to determine the result of our data.

# In[ ]:


labelcount = train_labels["label"]
print(labelcount.value_counts())


# In[ ]:


# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
#data1 = train_images.iloc[1]
#data2 = train_images.iloc[3]
#data1 = np.array(data1)
#data2 = np.array(data2)
#data3 = np.append(data1,data2)
#print(len(data3))
#plt.hist(data3)

label = [[],[],[],[],[],[],[],[],[],[]]
for j in range(10):
    for i in range(len(train_images)):
        if (train_labels.iloc[i].label == j):
            data = train_images.iloc[i]
            data = np.array(data)
            label[j] = np.append(label[j],data)
            
    plt.figure(j)
    plt.hist(label[j])
    plt.title(j)


# **STEP 5: Training our model**
# 
# First, we use the sklearn.svm module to create a vector classifier.
# Next, we pass our training images and labels to the classifier's fit method, which trains our model.
# Finally, the test images and labels are passed to the score method to see how well we trained our model. Fit will return a float between 0-1 indicating our accuracy on the test data set
# Try playing with the parameters of svm.SVC to see how the results change.

# In[ ]:


clf = svm.SVC() # Define model
clf.fit(train_images, train_labels.values.ravel()) # Fit: Capture patterns from provided data.
clf.score(test_images,test_labels) # Determine how accurate the model's


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=0)
tree.fit(train_images, train_labels)
test_predict = tree.predict(test_images)
print(mean_absolute_error(test_labels, test_predict))


# # **Question 4**
# 
# In above, did you see score() function?, open SVM.score() documentation at SKLearn, what does it's role?. Does it the same as MAE discussed in class previously?. Ascertain it through running the MAE. Now does score() and mae() produce the same results?.
# 
# **Q4 Answer**
# 
# 1. .score() function will determine how accurate the model's given from the test images, with the result of 0.887, (score with value closer to 1 is better)
# 2. .mean_absolute_error() function will determine the average error from the test labels and prediction from test images, with the result of 0,782. (MAE with value closer to 0 is better)

# In[ ]:


print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number

for y in index:
    plt.figure(y)
    img=train_images.iloc[y].values
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[y,0])


# **STEP 6: How did our model do?**
# 
# You should have gotten around 0.10, or 10% accuracy. This is terrible. 10% accuracy is what get if you randomly guess a number. There are many ways to improve this, including not using a vector classifier, but here's a simple one to start. Let's just simplify our images by making them true black and white.
# To make this easy, any pixel with a value simply becomes 1 and everything else remains 0.
# We'll plot the same image again to see how it looks now that it's black and white. Look at the histogram now.

# **Improving Performance**
# 
# Did you noticed, that the performance is so miniscule in range of ~0.1. Before doing any improvement, we need to analyze what are causes of the problem?. But allow me to reveal one such factor. It was due to pixel length in [0, 255]. Let's see if we capped it into [0,1] how the performance are going to improved.

# In[ ]:


test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])


# In[ ]:


plt.hist(train_images.iloc[i])


# **Retraining our model**
# 
# We follow the same procedure as before, but now our training and test sets are black and white instead of gray-scale. 
# Our score still isn't great, but it's a huge improvement

# In[ ]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# 

# # **Question 5**
# 
# Based on this finding, Can you explain why if the value is capped into [0,1] it improve significantly?. Perharps you need to do several self designed test to see why.
# 
# **Q5 Answer**
# Determine the capped value to [0,1], will change the performance because it reduce the probability from 255 into 2. As we seen before we process the data, it contain variety of shade and colour (white,gray,black). And thus the capped value of [0,1] will resulting the picture only have black or white colour.

# **Labelling the test data**
# 
# Now for those making competition submissions, we can load and predict the unlabeled data from test.csv. Again, for time we're just using the first 5000 images. We then output this data to a results.csv for competition submission.

# In[ ]:


# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])


# In[ ]:


# separate code section to view the results
print(results)
print(len(results))


# In[ ]:


df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)


# In[ ]:


#check if the file created successfully
print(os.listdir("."))


# # **Data Download**
# 
# We have the file, can listed it but how we are take it from sever. Thus we also need to code the download link.

# In[ ]:


# from https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)


# # **Question 6**
# 
# Alhamdulillah, we have completed our experiment. Here's things to do for your next task:
# 
# 1. What are the overfitting factor of SVM algorithm?. Previously on decision tree regression, the factor was max_leaf nodes. Do similar expriment using SVM!
# 2. Apply Decision Tree Classifier on this dataset, seek the best overfitting factor, then compare it with results of SVM.
# 3. Apply Decision Tree Regressor on this dataset, seek the best overfitting factor, then compare it with results of SVM & Decision Tree Classifier. Provides the results in table/chart. I suspect they are basically the same thing.
# 4. Apply Decision Tree Classifier on the same dataset, use the best overfitting factor & value. But do not use unnormalized dataset, before the value normalized to [0,1]
# 
# **Q6 answers**
# 
# Overfitting factor of SVM algorithm is the kernel, gamma and C parameter.
# When modifying certain data parameter will resulted in different MAE value.
# 
# In order to get the optimal value of gamma and C parameter, we can use [grid search](http://). Grid search will allow us to seatch all the value needed for to determine the parameter needed for the estimator
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Set the parameters by cross-validation
parameters = {'gamma': [0.01, 0.001, 0.0001],'C': [1, 10, 100,1000]}

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid = parameters)
clf.fit(train_images,train_labels.values.ravel())


# In[ ]:


print('Best C:',clf.best_estimator_.C) 
print('Best Gamma:',clf.best_estimator_.gamma)


# In[ ]:


#final svm
best_c = 10
best_gamma = 0.01
clf_final = svm.SVC(C=best_c,gamma=best_gamma)
clf_final.fit(train_images, train_labels.values.ravel())
finalsvm = clf_final.score(test_images,test_labels)
print(clf_final.score(test_images,test_labels))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
def get_mae_train_classifie(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

def get_mae_test_classifie(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


maee_train=[]
maee_test= []
leaf_nodes=[5,25,50,70,100,300,500,1000,3000,5000,7000]
for max_leaf_nodes in leaf_nodes:
    my_maetrain = get_mae_train_classifie(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    my_maetest = get_mae_test_classifie(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    maee_train.append(my_maetrain)
    maee_test.append(my_maetest)

plt.figure()
plt.plot(leaf_nodes,maee_test,color="red",label='Validation')
plt.plot(leaf_nodes,maee_train,color="blue",label='Training')
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Classifier")
plt.legend()
plt.show()

print (maee_train)
print (maee_test)


# In[ ]:


#final model dtclassifier

best_tree_size=1000
treeclassifie = DecisionTreeClassifier(max_leaf_nodes=best_tree_size, random_state=0)
treeclassifie.fit(train_images,train_labels)
scoredtc = treeclassifie.score(test_images,test_labels)
print ("DTC = ",scoredtc)
print ("SVM = ",finalsvm)


# In[ ]:


# Decision Tree Regressor
def get_mae_train_regress(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_train = model.predict(train_X)
    mae = mean_absolute_error(train_y, preds_train)
    return(mae)

def get_mae_test_regress(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

maee_train=[]
maee_test= []
leaf_nodes=[5,25,50,70,100,300,500,1000,3000,5000,7000]
for max_leaf_nodes in leaf_nodes:
    my_maetrain = get_mae_train_regress(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    my_maetest = get_mae_test_regress(max_leaf_nodes, train_images, test_images, train_labels, test_labels)
    maee_train.append(my_maetrain)
    maee_test.append(my_maetest)

plt.figure()
plt.plot(leaf_nodes,maee_test,color="red",label='Validation')
plt.plot(leaf_nodes,maee_train,color="blue",label='Training')
plt.xlabel("Tree Depth")
plt.ylabel("MAE")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

print (maee_train)
print (maee_test)


# In[ ]:


#final model dtregressor
best_tree_size=1000
treeregres = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
treeregres.fit(train_images,train_labels)
scoredtr = treeregres.score(test_images,test_labels)
print ("DTR = ",scoredtr)
print ("DTC = ",scoredtc)
print ("SVM = ",finalsvm)


# In[ ]:


# Decision Tree Classifier
# labeled_images.iloc[0:5000,1:], yang terseleksi kedalam variabel images baris ke-0 sampai 4999 dan kolom ke-1 sampai kolom terakhir
images = labeled_images.iloc[0:5000,1:]
# labeled_images.iloc[0:5000,:1], yang terseleksi kedalam variabel labels baris ke-0 sampai 4999 dan kolom ke-0
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
best_tree_size=1000
tree = DecisionTreeClassifier(max_leaf_nodes=best_tree_size,random_state=0)
tree.fit(train_images, train_labels)
test_predict = tree.predict(test_images)
#print(mean_absolute_error(test_labels, test_predict))
scoredtr_ver2 = tree.score(test_images,test_labels)
print ("DTC = ",scoredtr_ver2)


# 

# 

# 

# 
