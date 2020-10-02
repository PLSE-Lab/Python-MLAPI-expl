#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Hi there! I'm kind of new to Kaggle but would like to use this competition as an opportunity to demonstrate the Automunge tool for automated data-wrangling. This entry is possibly in the realm of Automated ML Tool territory, but to be fair Automunge isn't really a turnkey machine learning replacement, it's intended use is primarily for the preparation of tabular data for machine learning (basically performing the data preparation pipelines in the steps immediately preceding training a machine learning model or the subsequent consistent processing to prepare data to generate predictions), and the application of predictive algorithms are intended to be conducted seperately. That being said, the tool does make use of some predictive algorithms along the way, including the optional use of machine learning to predict infill to missing or improperly formatted data, what we call ML infill (more on that later).
# 
# In short, Automunge prepares tabular data intended for training a machine learning model, and enables consistent processing of subsequently available data for generating predictions from that same model. Through preparation, numerical data is normalized, categorical data is encoded, and time-series data is also encoded. A user may defer to automated methods where the tool infers properties of each column to assign a processing method, or alternately assign custom processign methods to distinct columns from our library of feature engineering transformations. 
# 
# A user may also consider automunge a platform for data wrangling, and may pass their own processing functions incorproating simple data structures such that through the incorproation of their trasnforms into the tool they can make use of extremely useful methods such as machine learning derived infill to missing or improperly formatted data (ML infill), feature importance evaluation, automated dimensionality reduction via feature importance results or Principle Components Analysis (PCA), and perthaps most importantly the simplest means for consistent processing of subsequently available data with just the simplest of function calls. In short, we make machine learning easy.

# # Prerequisites

# Before proceeding with the demonstration we'll conduct a few data preparations. Note that Automunge needs following prerequisites to operate:
# 
# - tabular data in Pandas dataframe or Numpy array format
# - "tidy data" (meaning one feature per column and one observation per row)
# - if available label column included in the set with column name passed to function as string
# - a "train" data set intended to train a machine learning model and if available a "test" set intended to generate predictions from the same model
# - the train and test data must have consistently formatted data and consistent column headers
# 
# Ok well introductions complete let's go ahead and manually munge to meet these requirements.

# # Data imports and preliminary munging

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#First we'll grab the file paths
train_identify_filepath = "../input/ieee-fraud-detection/train_identity.csv"
train_transaction_filepath = "../input/ieee-fraud-detection/train_transaction.csv"
test_identity_filepath = "../input/ieee-fraud-detection/test_identity.csv"
test_transaction_filepath = "../input/ieee-fraud-detection/test_transaction.csv"
sample_submission_filepath = "../input/ieee-fraud-detection/sample_submission.csv"


# In[ ]:


#Now let's import them as dataframes. Note both the identify and transaction sets include 
#a single common column, TransactionID, so we'll use that as an index column to merge

train_identify = pd.read_csv(train_identify_filepath, error_bad_lines=False, index_col="TransactionID")
train_transaction = pd.read_csv(train_transaction_filepath, error_bad_lines=False, index_col="TransactionID")
test_identity = pd.read_csv(test_identity_filepath, error_bad_lines=False, index_col="TransactionID")
test_transaction = pd.read_csv(test_transaction_filepath, error_bad_lines=False, index_col="TransactionID")
sample_submission = pd.read_csv(sample_submission_filepath, error_bad_lines=False)


# In[ ]:


#Note that by inspection we end up with the identify columns on about a quarter 
#of the rows of the transaction data
print("   train_identify.shape = ", train_identify.shape)
print("train_transaction.shape = ", train_transaction.shape)


# In[ ]:


#Here we'll concatinate the two sets based on the common index points
master_train = pd.concat([train_transaction, train_identify], axis=1, sort=False)
master_test = pd.concat([test_transaction, test_identity], axis=1, sort=False)

print("master_train.shape = ", master_train.shape)
print("master_test.shape = ", master_test.shape)


# In[ ]:


#Because I'm going to be doing a whole bunch of demonstrations 
#in this notebook, I'm going to carve out a much smaller set 
#to speed up the writing.

columns_subset = list(master_train)[:15]
#(test columns won't have the label column)
test_columns_subset = list(master_test)[:14]

small_train = master_train[columns_subset]
small_test = master_test[test_columns_subset]

from sklearn.model_selection import train_test_split
big_train, tiny_train = train_test_split(small_train, test_size=0.002, random_state=42)
big_test, tiny_test = train_test_split(small_test, test_size=0.002, random_state=42)


print(list(tiny_train))
print("")
print("tiny_train.shape = ", tiny_train.shape)
print("big_train.shape = ", big_train.shape)
print("")
print("tiny_test.shape = ", tiny_test.shape)
print("big_test.shape = ", big_test.shape)


# # Automunge install and initialize

# In[ ]:


#Ok here's where we import our tool with pip install. Note that this step requires  
#access to the internet. (Note this import procedure changed with version 2.58.)

get_ipython().system(' pip install Automunge')

# #or to upgrade (we currently roll out upgrades pretty frequently)
# ! pip install Automunge --upgrade


# In[ ]:


#And then we initialize the class.

from Automunge import Automunger
am = Automunger.AutoMunge()


# # Ok let's give it a shot

# Well at the risk of overwhelming the reader I'm just going to throw out a full application. Basically, we pass the train set and if available a consistently formatted test set to the function and it returns normalized and numerically encoded sets suitable for the direct application of machine learning. The function returns a series of sets (which based on the options selected may be empty), I find it helps to just copy and paste the full range of arguments and returned sets from the documentation for each application. 

# In[ ]:


#So first let's just try a generic application with our tiny_train set. Note tiny_train here
#represents our train set. If a labels column is available we should include and designate, 
#and any columns we want to exclude from processing we can designate as "ID columns" which
#will be carved out and consitnelty shuffled and partitioned.

#Note here we're only demonstrating on the set with the reduced number of features to save time.


train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', trainID_column = False,             testID_column = False, valpercent1=0.0, valpercent2 = 0.0,             shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False,             binstransform = False, MLinfill = False, infilliterate=1, randomseed = 42,             numbercategoryheuristic = 15, pandasoutput = True, NArw_marker = False,             featureselection = False, featurepct = 1.0, featuremetric = .02,             featuremethod = 'pct', PCAn_components = None, PCAexcl = [],             ML_cmnd = {'MLinfill_type':'default',                        'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}},                        'PCA_type':'default',                        'PCA_cmnd':{}},             assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[],                          'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[],                          'bins':[], 'bint':[],                          'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[],                          'log0':[], 'log1':[], 'pwrs':[],                          'bnry':[], 'text':[], 'ordl':[], 'ord2':[],                          'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[],                          'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[]},             assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[],                             'adjinfill':[], 'meaninfill':[], 'medianinfill':[]},             transformdict = {}, processdict = {},             printstatus = True)


# So what's going on here is we're calling the function am.automunge and pass the returned sets to a series of objects:
# ```
# train, trainID, labels, \
# validation1, validationID1, validationlabels1, \
# validation2, validationID2, validationlabels2, \
# test, testID, testlabels, \
# labelsencoding_dict, finalcolumns_train, finalcolumns_test, \
# featureimportance, postprocess_dict = \
# ```
# 
# Again we don't have to include all of the parameters when calling the function, but I find it helpful just to copy and paste them all. For example if we just wanted to defer to defaults we could just call:
# ```
# train, trainID, labels, \
# validation1, validationID1, validationlabels1, \
# validation2, validationID2, validationlabels2, \
# test, testID, testlabels, \
# labelsencoding_dict, finalcolumns_train, finalcolumns_test, \
# featureimportance, postprocess_dict = \
# am.automunge(tiny_train)
# ```
# 
# Those sets returned from the function call are as follows:
# 
# - __train, trainID, labels__ : these are the sets intended to train a machine learning model. (The ID set is simply any columns we wanted to exclude from transformations comparably partitioned and shuffled)
# - __validation1, validationID1, validationlabels1__ : these are sets carved out from the train set intended for hyperparameter tuning validation based on the designated validation1 ratio (defaults to 0.0)
# - __validation2, validationID2, validationlabels2__ : these are sets carved out from the train set intended for final model validation based on the designated validation2 ratio (defaults to 0.0)
# - __test, testID, testlabels__ : these are the sets derived from any passed test set intended to generate predictions from the machine learning model trained form the train set, consistently processed as the train set
# - __labelsencoding_dict__ : this is a dictionary which may prove useful for reverse encoding predictions generated from the machine learning model to be trained from the train set
# - __finalcolumns_train, finalcolumns_test__ : a list of the columns returned from the transformation, may prove useful in case one wants to ensure consistent column labeling which is required for subsequent processing of any future test data
# - __featureimportance__ : this stores the results of the feature importance evaluation if user elects to conduct
# - __postprocess_dict__ : this dictionary should be saved as it may be used as an input to the postmunge funciton to consistently process any subsequently available test data

# Let's take a look at a few items of interest from the returned sets.
# 
# Notice that the returned sets now include a suffix appended to column name. These suffixes identify what type of transformation were performed. Here we see a few different types of suffixes:

# In[ ]:


#sffixes identifying steps of transformation
list(train)


# In[ ]:


#And here's what the returned data looks like.
train.head()


# Upon inspection:
# - addr2, card4, and ProductCD both have a series of suffixes which represent the different categories derived from a one-hot-encoding of a categorical set
# - each of TransactionDT, TransactionAmt, card1, card2, card3, card5, addr1, addr2, dist1 have the suffix 'nmbr' which represents a z-score normalization
# - card6 has the suffix 'bnry' which represents a binary (0/1) encoding
# - P_emaildomain has the suffix 'ordl' which represents an ordinal (integer) encoding

# Automunge uses suffix appenders to track the steps of transformations. For example, one could assign transformations to a column which resulted in multiple suffix appenders, such as say:
# column1_bxcx_nmbr
# Which would represent a column with original header 'column1' upon which was performed two steps of transformation, a box-cox power law transform followed by a z-score normalization.

# # Labels
# 
# When we conducted the transfomation we also desiganted a label column which was included in the set, so let's take a peek at the returned labels.

# In[ ]:


list(labels)


# In[ ]:


#as you can see the returned values on the labels column are consistently encoded
#as were passed
labels['isFraud_bnry'].unique()


# In[ ]:


#Note that if or original labels weren't yet binary encoded, we could inspect the 
#returned labelsencoding_dict object to determine the basis of encoding.

#Here we just see that the 1 value originated from values 1, and the 0 value
#originated from values 0 - a trivial example, but this could be helpful if
#we had passed a column containing values ['cat', 'dog'] for instance.

labelsencoding_dict


# # Subsequent consistent processing with postmunge(.)
# 
# Another important object returned form the automunge application is what we call the "postprocess_dict". In fact, good practice is that we should always save externally any postprocess_dict returned from the application of automunge whose output was used to train a machine learning model. Why? Well using this postprocess_dict object, we can then pass any subsequently available "test" data that we want to use to generate predictions from that machine learning model giving fully consistent processing and encoding. Let's demonstrate. 
# 
# When we performed a train_test_split above to derive the "tiny_train" set, we also ended up with a bigger set called "tiny_train_bigger". Let's try applying the postmunge function to consistently process.
# 
# Note a few pre-requisites for the appplication of postmunge:
# 
# - requires passing a postprocess_dict that was dervied from the application of automunge
# - consistently formatted data as the train set used in the application of automunge from which the postprocess_dict was derived
# - consistent column labeling as the train set used in the application of automunge from which the postprocess_dict was derived
# 
# And there we have it, let's demonstrate the postmunge function on the set "tiny_test" we prepared above.

# In[ ]:


test, testID, testlabels, labelsencoding_dict, finalcolumns_test = am.postmunge(postprocess_dict, tiny_test, testID_column = False,              labelscolumn = False, pandasoutput=True, printstatus=True)


# In[ ]:


#And if we're doing our job right then this set should be formatted exaclty like that returned
#from automunge, let's take a look.

test.head()


# In[ ]:


#Looks good! 

#So if we wanted to generate predictions from a machine learning model trained 
#on a train set processed with automunge, we now have a way to consistently 
#prepare data with postmunge.


# # Let's explore a (few) of the automunge parameters

# Ok let's take a look at a few of the optional methods available here. First here again is what a full automunge call looks like:
# 
# ```
# train, trainID, labels, \
# validation1, validationID1, validationlabels1, \
# validation2, validationID2, validationlabels2, \
# test, testID, testlabels, \
# labelsencoding_dict, finalcolumns_train, finalcolumns_test, \
# featureimportance, postprocess_dict = \
# am.automunge(df_train, df_test = False, labels_column = False, trainID_column = False, \
#             testID_column = False, valpercent1=0.0, valpercent2 = 0.0, \
#             shuffletrain = False, TrainLabelFreqLevel = False, powertransform = False, \
#             binstransform = False, MLinfill = False, infilliterate=1, randomseed = 42, \
#             numbercategoryheuristic = 15, pandasoutput = True, NArw_marker = True, \
#             featureselection = False, featurepct = 1.0, featuremetric = .02, \
#             featuremethod = 'pct', PCAn_components = None, PCAexcl = [], \
#             ML_cmnd = {'MLinfill_type':'default', \
#                        'MLinfill_cmnd':{'RandomForestClassifier':{}, 'RandomForestRegressor':{}}, \
#                        'PCA_type':'default', \
#                        'PCA_cmnd':{}}, \
#             assigncat = {'mnmx':[], 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[], \
#                          'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[], 'MAD3':[], \
#                          'bins':[], 'bint':[], \
#                          'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[], \
#                          'log0':[], 'log1':[], 'pwrs':[], \
#                          'bnry':[], 'text':[], 'ordl':[], 'ord2':[], \
#                          'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[], \
#                          'excl':[], 'exc2':[], 'exc3':[], 'null':[], 'eval':[]}
#             assigninfill = {'stdrdinfill':[], 'MLinfill':[], 'zeroinfill':[], 'oneinfill':[], \
#                             'adjinfill':[], 'meaninfill':[], 'medianinfill':[]}, \
#             transformdict = {}, processdict = {}, \
#             printstatus = True)
# ```

# So let's just go through these one by one. (This section is kind of diving into the weeds, not required reading)
# 
# __df_train__ and __df_test__ First note that if we want we can pass two different pandas dataframe sets to automunge, such as might be beneficial if we have one set with labels (a "train" set) and one set without (a "test" set). Note that the normalization parameters are all derived just from the train set, and applied for consistent processing of any test set if included. Again a prerequisite is that any train and test set must have consistently labeled columns and consistent formated data, with the exception of any designated "ID" columns or "label" columns which will be carved out and consistenlty shuffled and partitioned. Note too that we can pass these sets with non-integer-range index or even multi column indexes, such that such index columns will be carved out and returned as part of the ID sets, consistently shuffled and partitioned. If we only want to process a train set we can pass the test set as "False".
# 
# __labels_column__ is intended for passing string identifiers of a column that will be treated as labels. Note that some of the methods require the inclusion of labels, such as feature importance evaluation or label frequency levelizer (for oversampling rows with lower frequency labels).
# 
# __trainID_column__ and testID_column are intended for passing strings or lists of strings identifying columns that will be carved out before processing and consistently shuffled and partitioned. 
# 
# __valpercent1__ and __valpercent2__ parameters are intended as floats between 0-1 that indicate the ratio of the sets that will be carved out for the two validation sets. If shuffle train is activated then the sets will be carved out randomly, else they will be taken from the bottom sequnetial rows of the train set and randomly partitioned between the two validaiton sets. Note that these values default to 0.
# 
# __shuffletrain__ parameter indicates whether the train set will be (can you guess?) yep you were right the answer is shuffled.
# 
# __TrainLabelFreqLevel__ parameter indicates whether the train set will have the oversampling method applied where rows with lower frequency labels are copied for more equal distribution of labels, such as might be beneficial for oversampling in the training operation.
# 
# __powertransform__ parameter indicates whether default numerical coloumn evluation will include an inference of distribution properties to assign between z-score normalization, min-max scaling, or box-cox power law trasnformation. Note this one is still somewhat rough around the edges and we will continue to refine mewthods going forward.
# 
# __binstransform__ indicates whether defauilt z-score normalizaiton applicaiton will include the develoipment of bins sets identifying a point's placement with respect to number of standard deviations from the mean.
# 
# __MLinfill__ indicates whether default infill methods will predict infill for missing points using machine learning models trained on the rest of the set in generalizaed and automated fashion. Note that this method benefits from increased scale of data in teh train set, and mmodels derbvied from the train set are used for consistent prediction methods for the test set.
# 
# __infilliterate__ indicates whether the predictive methods for MLinfill will be iterated by this integer such as may be beneficial for particularily messy data.
# 
# __randomseed__ seed for randomness for all of the random seeded methods such as predcitive algorithms for ML infill, feature importance, PCA, shuffling, etc
# 
# __numbercategoryheuristic__ an integer indicating for categorical sets the threshold between processing with one-hot-encoding vs ordinal methods
# 
# __pandasoutput__ quite simply True means returned sets are pandas dataframes, False means Numpy arrays (defaults to Numpy arrays)
# 
# __NArw_marker__ indicates whether returned columns will include a derived column indicating rows that were subject to infill (can be identified with the suffix "NArw")
# 
# __featureselection__ indicated whether a feature importance evlauation will be performed (using then shuffle permeation method), note this requires the inclusion of a designated loabels column in the train set. Results are presented in the returned object "featureimportance"
# 
# __featurepct__ if feature selection performed and featuremethod == 'pct', indicates what percent of columns will be retained from the feature importance dimensionality reduction (columns are ranked by importance and the low percent are trimmed). Note that a value of 1.0 means no trimming will be done.
# 
# __featuremetric__ if feature selection performed and featuremethod == 'metric', indicates what threshold of importance metric will be required for retained columns from the feature importance dimensionality reduction (columns feature importance metrics are derived and those below this threshold are trimmed). Note that a value of 0.0 means no trimming will be done.
# 
# __feteaturemethod__ accepts values of 'pct' or 'metric' indicates method used for any feature importance dimensionality reduction
# 
# __PCAn_components__ Triggers PCA dimensionality reduction when != None. Can be a float indicating percent of columns to retain in PCA or an integer indicated number of columns to retain. The tool evaluates whether set is suitable for kernel PCA, sparse PCA, or PCA. Alternatively, a user can assign a desired PCA method in the ML-cmnd['PCA_type']. Note that a value of None means no PCA dimensionality reduction will be performed unless the scale of data is below a heuristic based on the number of features. (A user can also just turn off default PCA with ML-cmnd['PCA_type'])
# 
# __PCAexcl__ a list of any columns to be excluded from PCA trasnformations
# 
# __ML_cmnd__ allows a user to pass parameters to the predictive algorithms used in ML infill, feature importance, and PCA (I won't go into full detail here, although note one handy feature is we can tell the algorithm to exlcude boolean columns form PCA which is useful)
# 
# __assigncat__ allows a user to assign distinct columns to different processing methods, for those columns that they don't want to defer to default automated processing. For example a user could designate columns for min-max scaling instead of z-score, or box-cox power law trasnform, or you know we've got a whole library of methods that we're continueing to build out. These are defined in our READ ME. Simply pass the column header string identifier to the list associated with any of these root categories.
# 
# __assigninfill__ allows a user to assign disinct columns to different infill methods for missing or improperly formatted data, for those columns that they don't want to defer to default automated infill whi ch could be either standard infill (mean to numerical sets, most common to binary, and boolean identifier to categorical), or ML infill if it was selected. 
# 
# __transformdict__ and __processdict__ allows a user to design custom trees or trasnformations or even custom processing functions such as documented in our essays that no one reads. Once defined a column can be assigned to these methods in the assigncat.
# 
# __printstatus__ You know, like, prints the status during operation. Self-explanatory!
# 
# 
# Now we'll demonstrate a few.

# # trainID_column, shuffletrain, valpercent1

# In[ ]:


#great well let's try a few of these out. How about the ID columns, let's see what happens when we pass one.
#Let's just pick an arbitrary one, TransactionDT

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', trainID_column = 'TransactionDT',              valpercent1=0.20, shuffletrain = True, pandasoutput=True, printstatus = False)


# In[ ]:


#Now we'll find that the TransactionDT column is missing from the train set, left 
#unaltered instead in the ID set, paired with the Transaction ID which was put
#in the ID set because it was a non-integer range index column (thus if we wanted
#to reassign the original index column we could simply copy the TransactionID column
#from the ID set back to the processed train set)

trainID.head()


# In[ ]:


#note that since our automunge call included a validation ratio, we'll find 
#a portion of the sets partitioned in the validation sets, here for instance
#is the validaiton ID sets 

#(we'll also find returned sets in the validation1, and validationlabels1)

#note that since we activated the shuffletrain option these are randomly
#selected from the train set

validationID1.head()


# # TrainLabelFreqLevel

# In[ ]:


#Let's take a look at TrainLabelFreqLevel, which serves to copy rows such as to
#(approximately) levelize the frequency of labels found in the set.

#First let's look at the shape of a train set returtned from an automunge
#applicaiton without this option selected

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', TrainLabelFreqLevel=False,              pandasoutput=True, printstatus=False)

print("train.shape = ", train.shape)


# In[ ]:


#OK now let's try again with the option selected. If there was a material discrepency in label frequency
#we should see more rows included in the returned set

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', TrainLabelFreqLevel=True,              pandasoutput=True, printstatus=False)

print("train.shape = ", train.shape)


# # binstransform

# In[ ]:


#binstransform just means that default numerical sets will include an additional set of bins identifying
#number of standard deviations from the mean. We have to be careful with this one if we don't have a lot
#of data as it adds a fair bit of dimensionality

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', binstransform=True,              pandasoutput=True, printstatus=False)

print("list(train):")
list(train)


# In[ ]:


#so the interpretation should be for columns with suffix including "bint" that indicates 
#bins for number fo standard deviations from the mean. For example, nmbr_bint_t+01
#would indicated values between mean to +1 standard deviation.


# # MLinfill

# In[ ]:


#So MLinfill changes the default infill method from standardinfill (which means mean for 
#numerical sets, most common for binary, and boolean marker for categorical), to a predictive
#method in which a machine learning model is trained for each column to predict infill based
#on properties of the rest of the set. This one's pretty neat, but caution that it performs 
#better with more data as you would expect.

#Let's demonstrate, first here's an applicaiton without MLinfill, we'll turn on the NArws option
#to output an identifier of rows subject to infill

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', MLinfill=False,              NArw_marker=True, pandasoutput=True, printstatus=False)

print("train.head()")
train.head()


# In[ ]:


#So upon inspection it looks like we had a few infill points on
#columns originating from dist1 (as identified by the NArw columns)
#so let's focus on that

#As you can see the plug value here is just the mean which for a 
#z-score normalized set is 0

columns = ['dist1_nmbr', 'dist1_NArw']
train[columns].head()


# In[ ]:


#Now let's try with MLinfill

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', MLinfill=True,              NArw_marker=True, pandasoutput=True, printstatus=False)

print("train[columns].head()")
train[columns].head()


# In[ ]:


#As you can see the method predicted a unique infill value to each row subject to infill
#(as identified by the NArw column). We didn't include a lot of data with this small demonstration
#set, so I expect the accuracy of this method would improve with a bigger set


# # numbercategoryheuristic

# In[ ]:


# numbercategoryheuristic just changes the threshold for number of unique values in a categorical set
#between processing a categorical set via one-hot encoding or ordinal processing (sequential integer encoding)

#for example consiter the returned column for the email domain set in the data, if we look above we see the
#set was processed as ordinal, let's see why

print("number of unique values in P_emaildomain column pre-processing")
print(len(train['P_emaildomain_ordl']))


# In[ ]:


#So yeah looks like that entry has a unique entry per row, so really not really a good candidate for inclusion at
#all, this might be better served carved out into the ID set until such time as we can extract some info from it
#prior to processing. But the poitn is if we had set numbercategoryheuristic to 1478 instead of 15 we would have 
#derived 1477 one-hot-encoded columns from this set which obviosuly would be an issue for this scale of data.


# # pandasoutput

# In[ ]:


#pandasoutput just tells whether to return pandas dataframe or numpy arrays (defaults to numpy which
#is a more universal elligible input to the different machine learning frameworks)

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',               pandasoutput=False, NArw_marker = False, printstatus=False)

print("type(train)")
print(type(train))


# In[ ]:


#note that if we return numpy arrays and want to view the column headers 
#(which remember track the steps of transofmations in their suffix appenders)
#good news that's available in the returned finalcolumns_train
print("finalcolumns_train")
finalcolumns_train


# In[ ]:


#or with pandasoutput = True

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',               pandasoutput=True, NArw_marker = True, printstatus=False)

print("type(train)")
print(type(train))


# # NArw_marker

# In[ ]:


#The NArw marker helpfully outputs from each column a marker indicating what rows were
#subject to infill. Let's quickly demonstrate. First here again are the returned columns
#without this feature activated.

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=False, pandasoutput=True, printstatus=False)

print("list(train)")
list(train)


# In[ ]:


#Now with NArw_marker turned on.

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=True, pandasoutput=True, printstatus=False)

print("list(train)")
list(train)


# In[ ]:


#If we inspect one of these we'll see a marker for what rows were subject to infill
#(actually already did this a few cells ago but just to be complete)

columns = ['dist1_nmbr', 'dist1_NArw']
train[columns].head()


# # featureselection

# In[ ]:


#featureselection performs a feature importance evaluation with the permutaion method. 
#(basically trains a machine learning model, and then measures impact to accuaracy 
#after randomly shuffling each feature)

#Let's try it out. Note that this method requires the inclusion of a labels column.

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              featureselection=True, pandasoutput=True, printstatus=False)


# In[ ]:


#Now we can view the results like so.
#(a future iteration of tool will improve the reporting method, for now this works)
for keys,values in featureimportance.items():
    print(keys)
    print('shuffleaccuracy = ', values['shuffleaccuracy'])
    print('baseaccuracy = ', values['baseaccuracy'])
    print('metric = ', values['metric'])
    print('metric2 = ', values['metric2'])
    print()


# In[ ]:


#I suspect the small size of this demonstration set impacted these results.

#Note that for interpretting these the "metric" represents the impact
#after shuffling the entire set originating from same feature and larger
#metric implies more importance
#and metric2 is derived after shuffling all but the current column originating from same
#feature and smaller metric2 implies greater relative importance in that set of
#derived features. In case you were wondering.


# # PCAn_components, PCAexcl

# In[ ]:


#Now if we want to apply some kind of dimensionality reduction, we can conduct 
#via Principle Component Analysis (PCA), a type of unsupervised learning.

#a few defaults here is PCA is automatically performed if number of features > 50% number of rows
#(can be turned off via ML_cmnd)
#also the PCA type defaults to kernel PCA for all non-negative sets, sparse PCA otherwise, or regular
#PCA if PCAn_components pass as a percent. (All via scikit PCA methods)

#If there are any columns we want to exclude from PCA, we can specify in PCAexcl

#We can also pass parameters to the PCA call via the ML_cmnd

#Let's demosntrate, here we'll reduce to four PCA derived sets, arbitrarily excluding 
#from the transofrmation columns derived from dist1


train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              PCAn_components=4, PCAexcl=['dist1'],              pandasoutput=True, printstatus=False)

print("derived columns")
list(train)


# In[ ]:


#Noting that any subsequently available data can easily be consistently prepared as follows
#with postmunge (by simply passing the postprocess_dict object returned from automunge, which
#you did remember to save, right? If not no worries it's also possible to consistnelty process
#by passing the test set with the exact saem original train set to automunge)

test, testID, testlabels, labelsencoding_dict, finalcolumns_test = am.postmunge(postprocess_dict, tiny_test, testID_column = False,              labelscolumn = False, pandasoutput=True, printstatus=False)

list(test)


# In[ ]:


#Another useful method might be to exclude any boolean columns from the PCA
#dimensionality reduction. We can do that with ML_cmnd by passing following:

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              PCAn_components=4, PCAexcl=['dist1'],              pandasoutput=True, printstatus=False,              ML_cmnd = {'MLinfill_type':'default',                         'MLinfill_cmnd':{'RandomForestClassifier':{},                                          'RandomForestRegressor':{}},                         'PCA_type':'default',                         'PCA_cmnd':{'bool_PCA_excl':True}})

print("derived columns")
list(train)


# # assigncat

# In[ ]:


#A really important part is that we don't have to defer to the automated evaluation of
#column properties to determine processing methods, we can also assign distinct processing
#methods to specific columns.

#Now let's try assigning a few different methods to the numerical sets:

#remember we're assigninbg based on the original column names before the appended suffixes

#How about let's arbitrily select min-max scaling to these columns 
minmax_list = ['card1', 'card2', 'card3']

#And since we previously saw that Transaction_Amt might have some skewness based on our
#prior powertrasnform evaluation, let's set that to 'pwrs' which puts it into bins
#based on powers of 10
pwrs_list = ['TransactionAmt']

#Let's say we don't feel the P_emaildomain is very useful, we can just delete it with null
null_list = ['P_emaildomain']

#and if there's a column we want to exclude from processiong, we can exclude with excl
#note that any column we exclude from processing needs to be already numerically encoded
#if we want to use any of our predictive methods like MLinfill, feature improtance, PCA
#on other columns. (excl just passes data untouched, exc2 performs a modeinfill just in 
#case some missing points are found.)
exc2_list = ['card5']

#and we'll leave the rest to default methods

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud', NArw_marker=False,              pandasoutput=True, printstatus=False,              assigncat = {'mnmx':minmax_list, 'mnm2':[], 'mnm3':[], 'mnm4':[], 'mnm5':[], 'mnm6':[],                          'nmbr':[], 'nbr2':[], 'nbr3':[], 'MADn':[], 'MAD2':[],                          'bins':[], 'bint':[],                          'bxcx':[], 'bxc2':[], 'bxc3':[], 'bxc4':[],                          'log0':[], 'log1':[], 'pwrs':pwrs_list,                          'bnry':[], 'text':[], 'ordl':[], 'ord2':[],                          'date':[], 'dat2':[], 'wkdy':[], 'bshr':[], 'hldy':[],                          'excl':[], 'exc2':exc2_list, 'exc3':[], 'null':null_list, 'eval':[]})

print("derived columns")
list(train)


# In[ ]:


#Here's what the resulting derivations look like
train.head()


# # assigninfill

# In[ ]:


#We can also assign distinct infill methods to each column. Let's demonstrate. 
#I remember when we were looking at MLinfill that one of our columns had a few NArw
#(rows subject to infill), let's try a different infill method on those 

#how about we try adjinfill which carries the value from an adjacent row

#remember we're assigning columns based on their title prior to the suffix appendings

train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=True, pandasoutput=True, printstatus=False,              assigninfill = {'adjinfill':['dist1']})

columns = ['dist1_nmbr', 'dist1_NArw']
train[columns].head()


# # transformdict and processdict

# In[ ]:


#trasnformdict and processdict are for more advanced users. They allow the user to design
#custom compositions of transformations, or even incorporate their own custom defined
#trasnformation functions into use on the platform. I won't go into full detail on these methods
#here, I documented these a bunch in the essays which I'll link to below, but here's a taste.

#Say that we have a numerical set that we want to use to apply multiple trasnformations. Let's just
#make a few up, say that we have a set with fat tail characteristics, and we want to do multiple
#trasnformions including a bocx-cox trasnformation, a z-score trasnformation on that output, as
#well as a set of bins for powers of 10. Well our 'TransactionAmt' column might be a good candiate
#for that. Let's show how.

#Here we define our cusotm trasnform dict using our "family tree primitives"
#Note that we always need to uyse at least one replacement primitive, if a column is intended to be left
#intact we can include a excl trasnfo0rm as a replacement primitive.

#here are the primitive definitions
# 'parents' :           upstream / first generation / replaces column / with offspring
# 'siblings':           upstream / first generation / supplements column / with offspring
# 'auntsuncles' :       upstream / first generation / replaces column / no offspring
# 'cousins' :           upstream / first generation / supplements column / no offspring
# 'children' :          downstream parents / offspring generations / replaces column / with offspring
# 'niecesnephews' :     downstream siblings / offspring generations / supplements column / with offspring
# 'coworkers' :         downstream auntsuncles / offspring generations / replaces column / no offspring
# 'friends' :           downstream cousins / offspring generations / supplements column / no offspring

#So let's define our custom trasnformdict for a new root category we'll call 'cstm'
transformdict = {'cstm' : {'parents' : ['bxcx'],                            'siblings': [],                            'auntsuncles' : [],                            'cousins' : ['pwrs'],                            'children' : [],                            'niecesnephews' : [],                            'coworkers' : [],                            'friends' : []}}

#Note that since bxcx is a parent category, it will look for offspring in the primitives associated
#with bxcx root cateogry in the library, and find there a downstream nmbr category

#Note that since we are defining a new root category, we also have to define a few parameters for it
#demonstrate here. Further detail on thsi step available in documentation. If you're not sure you might
#want to try just copying an entry in the READ ME.

#Note that since cstm is only a root cateogry and not included in the family tree primitives we don't have to
#define a processing funciton (for the dualprocess/singleprocess/postprocess entries), we can just enter None

processdict = {'cstm' : {'dualprocess' : None,                          'singleprocess' : None,                          'postprocess' : None,                          'NArowtype' : 'numeric',                          'MLinfilltype' : 'numeric',                          'labelctgy' : 'nmbr'}}

#We can then pass this trasnformdict to the automunge call and assign the intended column in assigncat
train, trainID, labels, validation1, validationID1, validationlabels1, validation2, validationID2, validationlabels2, test, testID, testlabels, labelsencoding_dict, finalcolumns_train, finalcolumns_test, featureimportance, postprocess_dict = am.automunge(tiny_train, df_test = False, labels_column = 'isFraud',              NArw_marker=True, pandasoutput=True, printstatus=False,              assigncat = {'cstm':['TransactionAmt']},              transformdict = transformdict, processdict = processdict)

print("list(train)")
list(train)


# In[ ]:


#and then of course use also has the ability to define their own trasnformation functions to
#incorproate into the platform, I'll defer to the essays for that bit in the interest of brevity


# # postmunge

# In[ ]:


#And the final bit which I'll just reiterate here is that automunge facilitates the simplest means
#for consistent processing of subsequently available data with just a single function call
#all you need is the postprocess_dict object returned form the original automunge call

#This even works when we passed custom trasnformdict entries as was case with last postprocess_dict
#derived in last example, however if you're defining custom trasfnormation functions for now you
#need to save those custom function definitions are redefine in the new notewbook when applying postmunge

#Here again is a demosntration of postmunge. Since the last postprocess_dict we returned
#was with our custom transfomrations in preceding excample, the 'TransactionAmt' column will
#be processed consistently

test, testID, testlabels, labelsencoding_dict, finalcolumns_test = am.postmunge(postprocess_dict, tiny_test, testID_column = False,              labelscolumn = False, pandasoutput=True, printstatus=True)


# In[ ]:


list(test)


# # Closing thoughts

# Great well certainly appreciate your attention and opportunity to share. I suppose next step for me is to try and hone in on my entry and perhaps get on the leaderboard. That'd be cool. 
# 
# Oh before I go if you'd like to see more I recently published my first collection of essays titled "From the Diaries of John Henry", which a big chunk included the documentation through the development of Automunge. Check it out it's all online.
# 
# [turingsquared.com](http://turingsquared.com)
# 
# Or for more on Automunge our website and contact info is available at 
# 
# [automunge.com](https://www.automunge.com)

# In[ ]:




