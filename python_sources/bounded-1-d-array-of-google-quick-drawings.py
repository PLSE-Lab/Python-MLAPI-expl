#!/usr/bin/env python
# coding: utf-8

# Hello, and thank you for reading my Kernal submission for the Google Quick Draw image recogniztion challenge. 
# 
# This kernel is dedicated to taking a non-standard approach to 2-D  image recogniztion for an exercise in the merits and drawbacks of potential innovations.
# The approach I will take is as follows: 
# 1.  Scale the drawings down into smaller resolutions
# 2. Convert the drawings from 2-D arrys to 1-D arrays of pixel intensity, 1 and 0 (present or not), where the index corresponds to a pixel location in a NxN grid
# 3. Convert each 1-D array into a row, where each column is a pixel location, and thus a factor for training. *
# 4. Train various mdoels on these factors, and the word is a target, then predict future targets. 
# 
# This approach is not typical from what I've seen other talented Kagglers take - which is why I wanted to do it. I can immeidately identify possible reasons this will go horribly, horribly wrong. I will discuss these reasons at the end, but for now items with asterisks that could be potentially problematic. 
# 
# For now, let's get going. First - let's import and inspect the simplified data:

# In[ ]:


# This Python 3 environment 
import pandas as pd
import numpy as np  
import os
import json
import copy
from sklearn.externals import joblib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))


# Any results you write to the current directory are saved as output.


# One by one, I'll select a few rows from a few categories of words and construct a primary dataframe, df. 

# In[ ]:


df = pd.DataFrame()
files_directory = sorted([f for f in os.listdir("../input/train_simplified")]) # Sort the directory first
training_categories = len(files_directory) #340
category_samples = 60
test_sample_cats = 112199 # Number of rows to load from the test set
downsize = 25 # Resolution of images, max is 255 but please don't do this unless you've got time to kill / compuational power
zero_lst = np.zeros(downsize**2, dtype=int)

df_from_each_file = [pd.read_csv("../input/train_simplified/"+f, nrows=category_samples) for f in files_directory]
df = pd.concat(df_from_each_file, ignore_index=True)

print("From {a} to {z}..".format(a=files_directory[0], z=files_directory[training_categories-1]))
print("-------")


# Great, do we've got df which contains all the information about the drawings, with a small sample from the number of categories defined by "training_categories".
# 
# Now, I only want to include correctly identified drawings. Furthermore, I want to make sure that the lists are being evaluated as python lists, not as strings which is what the default behavior of pandas is. For this excersize, I also only include data on the drawings and nothing else. We lose information, but we save some memory which is a very real objective on my current machine, but not so much for others. 

# In[ ]:


df = df[df["recognized"]==True]
df['drawing'] = df['drawing'].apply(json.loads)
df = df[["drawing","word"]].reset_index(drop=True)
print(df.info())


# This is where things start to deviate from a typical approach. I will now define a series of functions. The first function will create a list of point coordiate tuples. The next will transform the tuples into a lower resolution, scaled down to "downsize" number of pixels in width and height. The last function will convert the "drawing", now a collection of scaled points, to a 1-D array of pixel intensity. 
# 

# In[ ]:


## Gather all the points
def pointList(draw_list):
    point_list = []
    for n in range(0,len(draw_list)):
        for x,y in list(zip(draw_list[n][0],draw_list[n][1])):
            point_list.append((x,y))
    return point_list


# In[ ]:


## Bound the points, and scale
def transform(pt_list):
    xlist = []
    ylist = []
    scl_list = []
    for x,y in pt_list:
        xlist.append(x)
        ylist.append(y)
    xmx = max(xlist)
    xmn = min(xlist)
    ymx = max(ylist)
    ymn = min(ylist)
    for x,y in pt_list:
        try:
            x_scl = round((x-xmn)*downsize/(xmx - xmn))
            y_scl = round((y-ymn)*downsize/(ymx - ymn))
        except ZeroDivisionError:
            x_scl = round((x-xmn)*downsize/(xmx - xmn + 0.0001))
            y_scl = round((y-ymn)*downsize/(ymx - ymn + 0.0001))
        scl_list.append((x_scl, y_scl))
    return scl_list


# The "drawing" column of the simplified data is a list. The Nth element of the list is the Nth stroke of the user. A stroke is defined by another list. For each stoke, a list with two elements is created. The first element is the x value of the points along that stroke, and the second element is the y values along that stroke. For example, a "drawing" data point may be a list with three elements.  The first element is a list with two elements, the x- and y-points created during that stroke. The next stoke has a different set of x- and y-values, etc. The raw dataset also has a timestamp of each point, but I don't need that for this exercize. 
# 
# The following function will take the drawing data, and convert it to a single list of either 0 or 1. A zero means a pixel is not present, and a 1 means a pixel is present. The length of the list is downsize^2 (30^2)elements. The positional index corresponds uniquely to a point in 2-D space. 0 is the origin, and 900 is the point(30,30).
# 
# The 1-D vector array is mostly zeroes. So I only replace it with a 1 where a pixel exists. 

# In[ ]:


def ConvertTo1D(scl_pt_list):
    indx_lst = copy.copy(zero_lst)
    for x,y in scl_pt_list:
            i = x + downsize*(y-1)
            indx_lst[i-1] = 1
    return indx_lst


# Now I simply apply these function to every drawing in the df. The power of pandas is palpable. We also convert each vector element to a new column in a new training df. 
# 

# In[ ]:


print("Preparing df_vec..")
df_vec = df["drawing"].apply(pointList
                     ).apply(transform
                     ).apply(ConvertTo1D
                     ).apply(pd.Series)


# Now we are ready to train a series of models. I will choose a few here, then use the best one to predict drawings on our test set. I like to do at least two-fold cross valdiations, but more folds should yield a more accuracte picture at additional computational cost. 

# In[ ]:


print("Learning..")
X = df_vec
y = df["word"]
print("{} accuracy would be better than guessing".format(round(1/training_categories, 4)))


# In[ ]:


# Let's get learning!
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

classifiers = {#"K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=2),
               "Random Forest" : RandomForestClassifier(n_estimators=10, random_state=0),
               #"Support Vector Clf" : LinearSVC(penalty="l2", random_state=0),
               "Logistic Regression" : LogisticRegression(penalty="l2", random_state=0),
               #"Perceptron" : Perceptron(penalty="l2", random_state=0),
               #"Naive Bayes" : GaussianNB(),
               #"Decision Tree" : DecisionTreeClassifier(random_state=0)
               }


# Please note that this process below takes a bit of computational time. From running this kernel a few times, I already knmow which model will perform best. If you would like to try this kernel yourself with different parameters, make sure you un-comment these codes after you make a fork. 

# In[ ]:


# Find best Test_Train test accuracy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

splits=2
rounding_prec = '.4f'
clfy_acc = []
for name,clfy in classifiers.items():
    res = format(cross_val_score(clfy, X, y, cv=splits).mean(), rounding_prec)
    print("{s}-fold split: {n} test accuracy = {r}".format(s=splits, n=name, r=res))
    clfy_acc.append(res)
  
# Save the best trained model
print("..Saving..")
clf_acc_finder = list(zip(classifiers.keys(), clfy_acc))
for cl,ac in clf_acc_finder:
    if ac == max(clfy_acc):
        fitted_clf = classifiers[cl].fit(X, y)
#        joblib.dump(classifiers[cl], 
#                            "{c}_Trained_{cat}_by_{sam}.joblib".format(c=str(cl),
#                                                                       cat=training_categories,
#                                                                       sam=category_samples)) 
        print("Model Selected! {c} with {a} testing accuracy:".format(c=cl, a=ac))


# Some of the models have better accuracy than just randomly guessing. Now is the time to import our test set, convert the drawings, and throw it into the correct model. Due to  computational limitations, I practice with a quantity called "test_sample_cats" which limits the test set to work with. For submission, we have to use the full set. 
# 
# For the real submission, I import my pre-trained model.

# In[ ]:


print("..Loading & transforming test data..")
test_df = pd.read_csv("../input/test_simplified.csv", nrows=test_sample_cats)
X_test = test_df["drawing"].apply(json.loads
                          ).apply(pointList
                          ).apply(transform
                          ).apply(ConvertTo1D
                          ).apply(pd.Series)

X_test_keys = test_df["key_id"]


# In[ ]:


## Load in the model

# model_loaded = os.listdir("../input/training-vector-representations-of-google-q-d/")[-2] # Picks the model out
# clf_loaded = joblib.load("../input/training-vector-representations-of-google-q-d/"+str(model_loaded))
# print("..{} model selected!".format(str(model_loaded)[0:-24]))
print("..Predicting..")

pred = fitted_clf.predict(X_test)
pred = [p.replace(" ", "_") for p in pred]


# Finally, let's prepare the predictions for submission to the challenge by setting up a predictions dataframe. At this time, I'm not sure how to predict multiple targets, so I just predict a single label. *

# In[ ]:


df_pred = pd.DataFrame.from_dict({"key_id":X_test_keys, 
                                  "word":["{pr}".format(pr=p) for p in pred]}, orient="columns")

print("Submission shape: {s}".format(s=df_pred.shape))
print("Number of possible classes: {n}".format(n=training_categories))

df_pred.to_csv("SubmissionMG.csv", index=False)
print("Sucessfully created file") 


# Thank for you reading my Kernel, I hope I have inspired you to try something new even if the results don't get you into the top leaderboard. An accuracy of 1/(# of categories) is the same as randomly guessing, so my opinion is that if it's better, it's better.
# 
# As I mentioned early on, there are a multitude of ways this algorithm can be improved. The largest problem is that since each image is reduced to a string of pixels, rotations, horizontal,  and vertical translations will result in different combinations of columns being "activated". When taken all togther, it'll just be a superimposition of all the images of that word, which will not be very effective. One solution I propose is to some how make sure the images are alinged first. Second, I can modify the algorithm to predict multiple values instead of just one. This should also improve accuracy drastically. 
# 
# It may also be more efficient to export a trained model in pne phase, and use it on a pre-processed test set in a second phase. I am currently investigating this approach. 
# 
# Please feel free to fork and modify this algorithm by changing the number of categories imported, number of sample in each category, the number of testing samples, or whatever you want - let me know where you take this idea. I welcome constructive feedback and comments. 
# 
# **Thank you again**
# 
# Michael Greene
