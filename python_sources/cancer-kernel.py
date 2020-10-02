#!/usr/bin/env python
# coding: utf-8

# ## Import libraries and input files

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Training Text Data

# In[ ]:


#read the training file
training_text_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text",
            sep='\|\|', 
            names=["ID","Text"],
            skiprows=1,
            header=None
            )


# In[ ]:


# Display the obervations and features of training data
training_text_data.shape


# ## Check for Unique Text in Traning Text

# In[ ]:


print("{} unique Variation in traning text dataset".format(len(training_text_data["Text"].unique())))


# In[ ]:


training_text_data.head(7)


# ## Training Variants data

# In[ ]:


#read the training_variants file
training_variants_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_variants")


# In[ ]:


# Display the obervations and features of training  variants data
training_variants_data.shape


# ## Check for unquie Gene and Variations

# In[ ]:


print("{} unique Gene in traning variants dataset".format(len(training_variants_data["Gene"].unique())))
print("{} unique Variation in traning variants dataset".format(len(training_variants_data["Variation"].unique())))


# In[ ]:


training_variants_data.head(7)


# In[ ]:


# As both traning text and traning varaints have 3321 records we can merge the dataframe using ID column

#Merge traning text and traning variants data

full_df = pd.merge(left=training_text_data,
                   right=training_variants_data,
                   how='inner',
                   on='ID')


# In[ ]:


full_df.head(7)


# ## Check for Missing values

# In[ ]:


#check for any nulls in the dataframe
full_df.isnull().sum()


#  ## Imputing Null values

# In[ ]:


#filtering only null rows of Text column
print(full_df.loc[full_df['Text'].isnull(),'Text'])

#imputing the null rows of Text data with Gene row
full_df.loc[full_df['Text'].isnull(),'Text'] = full_df['Gene']


# In[ ]:


#check for any nulls in the dataframe after Imputing
full_df.isnull().sum()


# In[ ]:


print("{} unique Gene in full_df set".format(len(full_df["Gene"].unique())))
print("{} unique Variation in full_df set".format(len(full_df["Variation"].unique())))


# ## Visualize Class distribution

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", color_codes=True)

plt.figure(figsize=(12,8))
ax = sns.countplot(x="Class", data=full_df,palette="GnBu_d")
plt.ylabel('Frequency')
plt.xlabel('Class')
plt.title('Class Distribution')
plt.show()


# In[ ]:


full_df.info()


# In[ ]:


#Fixing the Class column as target variable
y=full_df['Class']


# In[ ]:


y.head(7)


# In[ ]:


#Take Text and Variation columns as Predictive variables to train the data
X=full_df[["Text","Variation"]]


# In[ ]:


X.head(7)


# ## Splitting full_df into Train and Test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=5)


# In[ ]:


X_train.head(7)


# In[ ]:


X_test.head(7)


# In[ ]:


y_test.head(7)


# In[ ]:


y_train.head(7)


# ## Vectorization of text and variation data

# In[ ]:


# vectorizing the text and remove the stop words

from sklearn.feature_extraction.text import CountVectorizer

# vectorzing object for Text column

vector_text = CountVectorizer(stop_words='english')

# vectorzing object for Variation column

vector_variation = CountVectorizer(stop_words='english')


# In[ ]:


# Get the count of repeated words in X_train and X_test by vectorzing Text column

vector_text.fit(X_train['Text'])
vector_text.fit(X_test['Text'])


# In[ ]:


# Get the count of repeated words in X_train and X_test by vectorzing Variation column

vector_variation.fit(X_train['Variation'])
vector_variation.fit(X_test['Variation'])


# In[ ]:


# Display the voucabulary count in vectorized Text column

vector_text.vocabulary_


# In[ ]:


# After fitting, now transform the count of Text words into matrix

transform_text_train= vector_text.transform(X_train['Text'])
transform_text_test= vector_text.transform(X_test['Text'])


# In[ ]:


# Display the voucabulary count in vectorized Variation column

vector_variation.vocabulary_


# In[ ]:


# After fitting, now transform the count of Variation words into matrix

transform_variation_train= vector_variation.transform(X_train['Variation'])
transform_variation_test= vector_variation.transform(X_test['Variation'])


# In[ ]:


# Merge train data horizontally from the transformed Matrix of Text and Variation 
import scipy.sparse as sp
x_train_mod=sp.hstack((transform_variation_train,transform_text_train))


# In[ ]:


x_train_mod.shape


# In[ ]:


# Merge test data horizontally from the transformed Matrix of Text and Variation 
x_test_mod=sp.hstack((transform_variation_test,transform_text_test))


# In[ ]:


x_test_mod.shape


# ## Run Logistic regression model 

# In[ ]:


from sklearn.linear_model import LogisticRegression

#importing metrics to calculate various scores

from sklearn import metrics

#create object from Logistic Regression classs
logr = LogisticRegression()

#fitting the logistic regression model to data

logr.fit(x_train_mod,y_train)


# In[ ]:


#predicting target varible using test data

y_pred_class = logr.predict(x_test_mod)


# ## Generating classification report for Logistic regression model

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_class))


# ## Run Linear SVM model

# In[ ]:


from sklearn import svm

#create object from SVM classs
lsvm = svm.LinearSVC()

#fitting the SVM model to data

lsvm.fit(x_train_mod,y_train)


# In[ ]:


#predicting target varible using test data

y_pred_class = lsvm.predict(x_test_mod)


# ## Generating classification report for SVM model

# In[ ]:


print(classification_report(y_test,y_pred_class))


# ## Test the model using stage 2 files

# In[ ]:


#reading stage 2 test data
stage2_test_text_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/stage2_test_text.csv",
                                  sep='\|\|', 
                                  names=["ID","Text"],
                                  header=None,
                                  skiprows=1)
stage2_test_variant_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/stage2_test_variants.csv")


# ## Check for Missing Values in  stage2 test and variant data

# In[ ]:


#check for any nulls in the test dataframe
stage2_test_text_data.isnull().sum()


# In[ ]:


#check for any nulls in the varaint dataframe
stage2_test_variant_data.isnull().sum()


# In[ ]:


print(stage2_test_text_data.head(7))
print(stage2_test_variant_data.head(7))


# In[ ]:


# # Display the obervations and features of stage2 text and variant
print(stage2_test_text_data.shape)
print(stage2_test_variant_data.shape)


# In[ ]:


print("{} unique Variation in stage2 traning variants dataset".format(len(stage2_test_variant_data["Variation"].unique())))


# In[ ]:


# As both traning text and traning varaints have 986 records we can merge the dataframe using ID column

#Merge traning text and traning variants data

test_full_df = pd.merge(left=stage2_test_text_data,
                   right=stage2_test_variant_data,
                   how='inner',
                   on='ID')


# In[ ]:


#display the data records of merged test dataframe
test_full_df.head(7)


# ## Vectorization of Stage2 Text and Variation

# In[ ]:


# Generate matrix with vectorizored object used for training so we get same no. of columns for test data as like Training data

stage2_transform_vector_text=vector_text.transform(test_full_df['Text'])
stage2_transform_vector_variation=vector_variation.transform(test_full_df['Variation'])

#Merge transformed columns horizontally
x_train_stage2_sub= sp.hstack((stage2_transform_vector_variation,stage2_transform_vector_text))


# In[ ]:


# Display shape of stage 2 test and train data

print("Shape of train dataset", x_train_mod.shape)

print("Shape of stage2 test dataset", x_train_stage2_sub.shape)


# ## SVM accuracy score is 0.55 where as accuracy score of Logistic regression 0.62. So let us build the modle for stage2 data using Logistic regression
# 
# 

# In[ ]:


# Run the Logistic regression model for predicting stage 2 data

logr_s2= LogisticRegression()
logr_s2.fit(x_train_mod,y_train)


# ## Calculate probabilities

# In[ ]:


#predict stage 2 test data
y_pred_test=logr_s2.predict_proba(x_train_stage2_sub)
print(y_pred_test)


# In[ ]:


# Compare the actual vs predicted for first 9 classes
print(list(zip(y_test[1:10],y_pred_test[1:10].max().index.values)))


# ## Calcualte Log loss

# In[ ]:


from sklearn.metrics.classification import log_loss
print("The log loss is:",log_loss(y_test,y_pred_test[:831], eps=1e-15))


# In[ ]:


# `As we have 9 classes of probabilites in the array , let us create a dataframe holding 9 classes

cols=['class1','class2','class3','class4','class5','class6','class7','class8','class9']
y_pred_test=pd.DataFrame(y_pred_test,columns=cols)

y_pred_test.head(10)


# In[ ]:


test_full_df.head(7)


# ## Prepare Submission File

# In[ ]:



#Concatinating ID column of test_full_df and all the columns of y_pred_test
submission_file=pd.concat([test_full_df.ID,y_pred_test],axis=1)
submission_file.head(7)


# In[ ]:


submission_file.info()


# In[ ]:


#Converting the submission file into CSV

submission_file.to_csv('Submisson_File',
                       sep=',',
                       header=True,
                       index=None
                      )
submission_file.to_csv(r'Submission_File.csv',index=False)


# ## Export Submission File

# In[ ]:


#Export the submission file
from IPython.display import FileLink
FileLink(r'Submission_File.csv')

