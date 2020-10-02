#!/usr/bin/env python
# coding: utf-8

# # Find and examine MNIST images failing in test submission.
# *CLM:  A forked-fork from Chris Deotte's "MNIST Perfect 100% using kNN". 
# 
# I want to use Chris Deotte's method to find the ground truth labels for the Kaggle test set (Chris found that they were taken, unaltered, from the original MNIST data) to identify and examine the particular images that are impairing my score from the CNNs that I've attempted (my best is about 0.9967).
# 
# (from Chris Deotte's "MNIST Perfect 100% using kNN")
# ""... verifies that Kaggle's unknown "test.csv" images are entirely contained unaltered within MNIST's original dataset with known labels. Therefore we CANNOT train with MNIST's original data, we must train our models with Kaggle's "train.csv" 42,000 images, data augmentation,  and/or non-MNIST image datasets.""
# 
# My suspicion is that there are a few images in the data having unusual features which are confusing for the classifiers (or human readers for that matter).  For example, Chris Deotte showed a "9" image that had a small dash in the lower right-hand corner.  If there are many such images, then we need to make sure that our augmentation method generates an appropriate percentage of similar features.  If there are images that would be confusing for a human reader, perhaps they are also not adequately represented in the training data, and we might design some special augmentation for these (for example, clipping or blotting).
# 
# In addition to augmentation, perhaps we can see a spatial relation between some unusual feature(s) that might necessitate the use of some particular kernel size or pooling stride.
# 
# So, I have trained my network up with the usual kaggle training set, created my submission set as usual, then  I've compared my own submission set using the labels Chris' method identified, then create a confusion matrix for my kaggle failures.
# 
# Then, finally, I display all the images (about 125) that my classifier failed for, and argue that 85 of these are so ambiguous or corrupted as to be considered illegible both to humans and probably to machine. 85/28000 implies a maximum Kaggle competition score of about 0.9970, which agrees with Chris Deotte's literature survey.
# 
# 
# # Load MNIST's 70,000 original images

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from   sklearn.metrics import  confusion_matrix


# In[2]:


# Taken verbatim from Chris:
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('../input/mnist-numpy/mnist.npz')


# In[3]:


x_train = x_train.reshape(x_train.shape[0],784)
y_train = y_train.reshape(y_train.shape[0],1)
x_test = x_test.reshape(x_test.shape[0],784)
y_test = y_test.reshape(y_test.shape[0],1)
MNIST_image = np.vstack( (x_train,x_test) )
MNIST_label = np.vstack( (y_train,y_test) )


# In[4]:


MNIST_image.shape


# # Load Kaggle's 28,000 competition test images

# In[5]:


Kaggle_test_image = pd.read_csv("../input/digit-recognizer/test.csv")
Kaggle_test_image = Kaggle_test_image.values.astype("uint8")
Kaggle_test_label = np.empty( (28000,1), dtype="uint8" )


# # Match images from Kaggle's test set (unlabeled) with MNIST originals
# 
# (Chris Deotte:)
# "For each of Kaggle's 28000 test images, we search within MNIST's original 70000 image dataset for the closest image. If the closest image has distance zero that means the images are exactly the same and the label of the unknown Kaggle test image is precisely the label of the known MNIST image."

# In[7]:


#  Chris' search method takes a while on kaggle.  I'm doing several things to try to speed it up.
#    1) Check first a few pixels in the middle of images before checking all 28x28  (28X speedup ?)
#    2) Short-circuit the j-loop when a match is found.   (2X speedup ?)
#    3) Keep track also of which MNIST images have been matched with kaggles, avoid further comparisons
#        to those.   ((28000/2)/70000 speedup ?)

# Since it's convenient to test for < 0, initialize these lists to value "-1"
MNIST_match_list  = np.full(       (MNIST_image.shape[0]), -1, dtype="int32" )
kaggle_match_list = np.full( (Kaggle_test_image.shape[0]), -1, dtype="int32" )

print("Starting search to compare Kaggle test images to MNIST images. \n  This can take 30-60 minutes...")
#  Is it reasonable to ASSUME that there are no repetitions of kaggle images in the test set?
for i in range(0,Kaggle_test_image.shape[0]): # go through the Kaggle test images
    found_kaggle_match = False  # We'll use this flag to stop the search through the MNIST set when a match is found
    for j in range(0,MNIST_image.shape[0]): # search through the larger MNIST image set
        if found_kaggle_match:  # as soon as a match is found, then break from the j-loop
            break
        if MNIST_match_list[j] < 0:  # then this MNIST image is not yet paired with a kaggle test image
            #  another way to speed this up would be to check one or a few pixels in the middle of the image first
            #    and if those match, then check the entire image:
            if Kaggle_test_image[i,407] == MNIST_image[j,407]:  
                if np.absolute(Kaggle_test_image[i,395:417] - MNIST_image[j,395:417]).sum()==0:
                    if np.absolute(Kaggle_test_image[i,] - MNIST_image[j,]).sum()==0:
                        # Then this j-th MNIST image exactly matched the i-th Kaggle test image
                        found_kaggle_match   = True
                        Kaggle_test_label[i] = MNIST_label[j]
                        kaggle_match_list[i] = j  # Well, yes it would be more pythonic to use a dictionary...
                        MNIST_match_list[j]  = i
    # if found_kaggle_match == False:
    #    print("index ",i": after search thru MNIST, no exact match is found:")
    if i%1000==0:
          print("  Progress:  %d kaggle test images searched" % (i))

# CLM: well, it did speedup but maybe not by 50X (why?).  Takes about an hour on kaggle's CPU.


# In[61]:


#  See if there are any kaggle images that don't have a match
#    The result array here should be empty
np.where( kaggle_match_list < 0 )


# In[63]:


#  Now, Chris Deotte was interested in how many of Kaggles test came from below or above MNIST 60000
print("%d images are in MNIST-train's 60k and %d are in MNIST-test's 10k"       % ( np.sum( kaggle_match_list < 60000 ), np.sum( kaggle_match_list > 60000 ) )  )


# In[65]:


# illustrate, just a reality check, of a sample of the match lists:
print( kaggle_match_list[0:10] )
#  if we map from a sequence to kaggle to MNIST and back we should get the sequence:
print( MNIST_match_list[kaggle_match_list[0:10]]  )


# In[66]:


#  Just another visual of how the kaggle matches mapped from the MNIST set:
kaggle_matches_histogram = plt.hist( kaggle_match_list, bins = 100 )
plt.title("histogram of kaggle_match_list; indices into MNIST_images")
plt.grid(True)


# In[14]:


# CLM:  Chris Deotte's version, which gave the number of images taken from above and below the MNIST 60000 index.
#        I still need to implement the equivalent...
#     c1=0; c2=0;
#     print("Classifying Kaggle's 'test.csv' using kNN k=1 and MNIST 70k images")
#     for i in range(0,28000): # loop over Kaggle test
#         for j in range(0,70000): # loop over MNIST images
#             if np.absolute(Kaggle_test_image[i,] - MNIST_image[j,]).sum()==0:
#                 Kaggle_test_label[i] = MNIST_label[j]
#                 if i%1000==0:
#                     print("  %d images classified perfectly" % (i))
#                 if j<60000:
#                     c1 += 1
#                 else:
#                     c2 += 1
#                 break
#     if c1+c2==28000:
#         print("  28000 images classified perfectly")
#         print("Kaggle's 28000 test images are fully contained within MNIST's 70000 dataset")
#         print("%d images are in MNIST-train's 60k and %d are in MNIST-test's 10k" % (c1,c2))


# # Result 100% kaggle_test image identification from MNIST set
# (CLM: now, we also know this since every element of my kaggle_match_list has a non-negative value).
# 
# (Chris:)  Every Kaggle "test.csv" image was found unaltered within MNIST's 70,000 image dataset. Therefore we CANNOT use the original 70,000 MNIST image dataset to train models for Kaggle's MNIST competition.
# 
# *(CLM:  "Cannot" in the sense that if we do try to get augmentation by using the original MNIST, we're "cheating".)*
# 
# (Chris:)  Since MNIST's full dataset contains labels, we would know precisely what each unknown Kaggle test image's label is. We must train our models with Kaggle's "train.csv" 42,000 images, data augmentation,  and/or non-MNIST image datasets. The following csv file would score 100% on Kaggle's public and private leaderboard if submitted.

# In[15]:


results = pd.Series(Kaggle_test_label.reshape(28000,),name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("kaggle_test_labels_recovered.csv",index=False)


# In[67]:


#  check that the file was written:
os.listdir(".")


# # Examples, for reality check...

# In[68]:


# plot some examples, more or less at random, to visually check that the matched images look identical 

plt.figure(figsize=(15,5))
for i in range(6):
    #  CLM: we can look up the match for ANY of kaggle test images now:
    mnist_match_indx = kaggle_match_list[i]  #  CLM: I have a list of ALL the matches!  :)
    plt.subplot(2, 6, 2*i+1)
    plt.imshow(Kaggle_test_image[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("Kaggle test #%d\nLabel unknown" % i,y=0.9)
    plt.axis('off')
    plt.subplot(2, 6, 2*i+2)
    # plt.imshow(MNIST_image[index[i]].reshape((28,28)),cmap=plt.cm.binary)
    plt.imshow(MNIST_image[mnist_match_indx].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("MNIST #%d\nLabel = %d" % (mnist_match_indx,MNIST_label[mnist_match_indx]),y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


# (From Chris:)
# Using kNN k=1, we know with 100% accuracy that Kaggle's first six test images are digits 2, 0, 9, 0, 3, 7 respectively. Similarly we know the next 27994 test images perfectly too.

# # What accuracy is actually possible?
# *(CLM: Here, verbatim from Chris Deotte's kernel:)*
# 
# Here are the best published MNIST classifiers found on the internet:
# * 99.79% [Regularization of Neural Networks using DropConnect, 2013][1]
# * 99.77% [Multi-column Deep Neural Networks for Image Classification, 2012][2]
# * 99.77% [APAC: Augmented PAttern Classification with Neural Networks, 2015][3]
# * **99.76% [Kaggle published kernel, 2018][12]**
# * 99.76% [Batch-normalized Maxout Network in Network, 2015][4]
# * 99.71% [Generalized Pooling Functions in Convolutional Neural Networks, 2016][5]
# * More examples: [here][7], [here][8], and [here][9]  
#   
# On Kaggle's website, there is only one published kernel more accurate than 99.70%. The few other high scoring published kernels train with the full MNIST 70000 image dataset and therefore have actual accuracy under 99.70%.
# [1]:https://cs.nyu.edu/~wanli/dropc/
# [2]:http://people.idsia.ch/~ciresan/data/cvpr2012.pdf
# [3]:https://arxiv.org/abs/1505.03229
# [4]:https://arxiv.org/abs/1511.02583
# [5]:https://arxiv.org/abs/1509.08985
# [7]:http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
# [8]:http://yann.lecun.com/exdb/mnist/
# [9]:https://en.wikipedia.org/wiki/MNIST_database
# [10]:https://www.kaggle.com/cdeotte/mnist-perfect-100-using-knn/
# [12]:https://www.kaggle.com/cdeotte/35-million-images-0-99757-mnist

# # Now, examine the Kaggle test images that  my DNN failed to match correctly
# 

# # Add or upload dataset
# ### places selected files in the input folder of Workspace for this kernel
# ###  you can Add directly from Kaggle, or you could download from another kernel, then upload into this one. 
# ### -> File -> Add or upload dataset -> Kernel Output Files -> Your Work 
# ###  -> (point to your kernel which has the submission Output file.)
# ##  OR
# ###  Upload
# When you've successfully added your Mnist classification submission file to the input for your fork of this kernel,
# it will appear in your input folder, in a folder/directory with the name of the kernel that created the submission.
# OR if you used the Upload feature, it is placed in a folder/directory with the name you gave in the dialog.
# ## -->  Next, update the path and file you wish to examine (next code cell below)
# ### Note that Kaggle's dialog seems to always change ''_'' to ''-'' in the actual filenames, but not in the Workspace summary.

# In[18]:


path_to_your_added_submission_file    = '../input/sample-submission-file'
submission_filename               = 'mnist_tf_dnn_submit_v5.csv'

os.listdir( path_to_your_added_submission_file )
# os.listdir("../input" )


# In[19]:


kaggle_labels    = pd.read_csv('kaggle_test_labels_recovered.csv', delimiter=',')
mnist_submission = pd.read_csv(path_to_your_added_submission_file+"/"+submission_filename, delimiter=',')


# ## simple interactive check that the files read as expected

# In[20]:


kaggle_labels.info()


# In[21]:


kaggle_labels.head()


# In[22]:


kaggle_labels.iloc[:,1].values


# ## Now, at last we can see where our submission file performed well or poorly.

# In[23]:


kag_test_submit_confusion = confusion_matrix( kaggle_labels.iloc[:,1].values, mnist_submission.iloc[:,1].values   )
kag_test_submit_confusion


# In[24]:


#  Sum up just the errors in each row
kag_test_submit_confusion_errno =     [kag_test_submit_confusion[i,:].sum()-kag_test_submit_confusion[i,i] for i in range(kag_test_submit_confusion.shape[0])]
kag_test_submit_confusion_errno


# In[25]:


# The total over all rows is
np.sum( kag_test_submit_confusion_errno )


# In[26]:


max_confusion_of_one_digit = np.max(kag_test_submit_confusion_errno)


# # Now, examine each of these digit categories for clues as to where and why my DNN fails

# In[27]:


def find_and_plot_wrong_digits(digit):
    plt.figure(figsize=(16,16))
    k = 0
    image_error_list = []
    for i in range(kaggle_labels.shape[0]):
        if kaggle_labels.iloc[i,1] == digit:
            if kaggle_labels.iloc[i,1] != mnist_submission.iloc[i,1]:
                image_error_list.append( [i,kaggle_labels.iloc[i,1],mnist_submission.iloc[i,1]])
                plt.subplot(4, 6, k+1)
                plt.imshow(Kaggle_test_image[i].reshape((28,28)),cmap=plt.cm.binary) # cmap=plt.cm.binary  cmap='gray'
                plt.title("Kgl-test %d \n GT:%d Pred:%d " % (i,kaggle_labels.iloc[i,1], mnist_submission.iloc[i,1]), y=1.0)
                # plt.axis('off')
                k = k+1
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    return image_error_list


# In[28]:


digit_0_error_list = find_and_plot_wrong_digits(0)


# In[29]:


digit_1_error_list = find_and_plot_wrong_digits(1)


# In[30]:


digit_2_error_list = find_and_plot_wrong_digits(2)


# In[31]:


digit_3_error_list = find_and_plot_wrong_digits(3)


# In[32]:


digit_4_error_list = find_and_plot_wrong_digits(4)


# In[33]:


digit_5_error_list = find_and_plot_wrong_digits(5)


# In[34]:


digit_6_error_list = find_and_plot_wrong_digits(6)


# In[35]:


digit_7_error_list = find_and_plot_wrong_digits(7)


# In[36]:


digit_8_error_list = find_and_plot_wrong_digits(8)


# In[37]:


digit_9_error_list = find_and_plot_wrong_digits(9)


# # Can we agree that many of these digits are illegible to a human reader?
# ###  If so, what is the real-life value of classifying them to a category of a normal digit?  In real life, do we want a bank to process a check where a "9" is classified as a "4", or "7" is classified as a "1" ?
# ##  As a human reader, how many of these images would I consider really ambiguous?
# ### (Note: as a Navy avionics technician in the 1970's, we were expected to follow certain rules: "zero" and "seven" have a slash, "one" has lower serif (a "base"), but no upper serif, "four" is open at top. The point being that NavAir recognized that handwritten numerals are often ambiguous without imposing extra rules.  And, us oldtimers can remember when people wrote checks by hand, and we wrote the amounts in words as well as numerals.)
# 
# ## I'm counting about 40 of the above that I think a human reader could correctly guess, due to some sublety of feature, and presumably we might to train a classifier to correctly identify.  The remaining 85 are so ambiguous or corrupted that I would expect many human readers to mis-read them.

# In[38]:


(28000-85)/28000


# ## If we expect human reader to misclassify more than 85 of the 28000,  then the best accuracy we should expect is 0.9970, and that is in agreement with what Chris Deotte found from the literature.
# 
