#!/usr/bin/env python
# coding: utf-8

# - The notebook has been created as a solution to a handson training session, introduction to machine learning
# - This notebook is converted to fill-in-the-blanks that trainees can use during the session. The main notebook with no blanks is Base
# - Template_1 has less blanks than this notebook

# # Business Problem and Solution Framework

# 1. **Business Problem** To predict which credit card accounts are expected to default in their payments next month
# 2. **Available Data**
#     1. Geography: Taiwan 
#     2. Duration: April 2005 - September 2005
#     3. Source: UCI Machine Learning Repostiory https://archive.ics.uci.edu/ml/
#     4. Contents of Data: Information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients
# 4. **Success Criteria** 
#     1. Accuracy of prediction

# # Setup

# ### Kaggle Data Location

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# ### Library Imports

# Python has a suite of libraries that are aimed at data science. Few commonly used libraries of python are:
# * pandas : meant for tabular data manipulation and analysis library
# * numpy : meant for linear algebra operations
# * scikit-learn : meant for machine learning and statistial models
# * matplotlib and seaborn : meant for visualizations and graphs
# * nltk : meant for natural language processing
# 
# **Import the necessary libraries for our notebook**

# In[ ]:


# import the tabular data manipulation and analysis library: pandas as pd
# TODO

# import thel linear algebra library: numpy as np
#TODO

# import machine learning model library: sklearn as sk
#TODO

# import visualization libraries: matplotlib, seaborn
from matplotlib import pyplot as plt 
import seaborn as sns


# # Data Import and Understanding

# - Before we begin the model development exercise we need to understand the data available, its quantity and its quality
# - Load the data and try answering the following questions that form the first impression of the data
#     1. Understand how big the data is, number of rows, number of columns, size in MBs/GBs/TBs
#     1. What is the label variable ?
#     1. What attributes are available in the data to predict the label ?

# ### Load the data

# - We will use pandas and it's functions to load and maintain data in the notebook
# - Read the data into a pandas dataframe using read_csv() function of pandas
#     - pd.read_csv(*file_to_read*)
#     - location of file: */kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv*
#     - Returns: Pandas dataframe holding the data
# - You can view top n-rows of the data in a DataFrame using the function head(n)

# In[ ]:


# Read the variable into a pandas DataFrame
fulldata= # TODO put in the read_csv() function here


# In[ ]:


# Take a peek at top-10 rows of data using the head function of dataframe
# TODO use the head command to view data


# - pandas has a limit on number of columns and rows it shows without truncation. We can alter the limit by setting the variables:
#     - pd.options.display.max_columns
#     - pd.options.display.max_rows

# In[ ]:


pd.options.display.max_columns=50 # increase the number of columns displayed without truncation
pd.options.display.max_rows=999 # increase the number of rows displayed without truncation


# In[ ]:


# Take a peek at top-10 rows of data using the head function of the dataframe
# TODO check the head function


# ### Data Understanding

# - Q. What variable in the data holds the label ?
#     - A. TODO
# - Q. What are the attribute variables in the data ?
#     - A. TODO
# - Q. How can one describe a row in the above dataset ?
#     - A. TODO

# **P1. What is the size of the data ?**
# - Input: Raw Data
# - Output:
#     - Number of rows
#     - Number of columns
#     - *Number of files*
#     - *Size of files*
# 
# shape attribute of DataFrame holds the number of rows and columns in a dataframe

# In[ ]:


# Use shape attribute of DataFrame object to obtain size
# TODO put in the shape command here


# **P2. What information is available in the data ?**
# - Input: Raw Data
# - Output: Detailed understanding of features and rows of the data

# **Model development requires careful selection of attributes**
# 1. Adding irrelevant attributes act as noise and requires an effort from the model to learn to ignore. They also cost computation.
# 2. Dropping relevant attributes make it difficult for model to learn to make the correct prediction
# 3. Further, incorrectly chosen attributes, sometimes (in case of *Target Leakage*), can give false impression of excellent model performance
# 
# Though automated techniques exist for feature selection they are not robust against all the issues above. Hence, the first step is always gaining an understanding of attributes available and the values they take.This requires close collaboration with domain experts.
# 
# Since we are using a public dataset we have a well defined data dictionary for the dataset

# **Attribute Descripition**
# - **ID**
#     - ID: ID of each client
# - **Numeric Variables**
#     - LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
#     - BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
#     - BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
#     - BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
#     - BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
#     - BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
#     - BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
#     - PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
#     - PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
#     - PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
#     - PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
#     - PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
#     - PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# - **Ordinal Variables**
#     - PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
#     - PAY_2: Repayment status in August, 2005 (scale same as above)
#     - PAY_3: Repayment status in July, 2005 (scale same as above)
#     - PAY_4: Repayment status in June, 2005 (scale same as above)
#     - PAY_5: Repayment status in May, 2005 (scale same as above)
#     - PAY_6: Repayment status in April, 2005 (scale same as above)
# - **Categorical Variables**
#     - SEX: Gender (1=male, 2=female)
#     - EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#     - MARRIAGE: Marital status (1=married, 2=single, 3=others)
#     - AGE: Age in years
# - **Target / Label**
#     - default.payment.next.month: Default payment (1=yes, 0=no)

# **Todo**
# - Education has 2 values that are unknown => merge the two values
# - Categorical attributes have been defined using numeric values => ensure pandas treats categorical attributes as categorical
# - ID column is not a useful attribute and should be used as the index for rows

# In[ ]:


# Replace all rows having education==6 with education = 5
# .loc command allows us to replace values in columns and rows of a dataFrame
# Format of loc command DataFrame.loc[rowCriteria,colCriteria]=VALUE
# TODO .loc command to replace value 6 with value 5


# **P3. Ensure correctness the data type for each attribute**
# 
# Given some attributes look like numbers but are categories we need to ensure that pandas treats them as categorical attributes and not numbers

# In[ ]:


# Check the current dtype
fulldata.dtypes


# In[ ]:


# Correct the dtypes for categorical variables
fulldata['ID']=fulldata['ID'].astype(object)
fulldata['SEX']=fulldata['SEX'].astype(object)
fulldata['EDUCATION']=fulldata['EDUCATION'].astype(object)
fulldata['MARRIAGE']=fulldata['MARRIAGE'].astype(object)


# In[ ]:


# Re-check the current dtype
fulldata.dtypes


# In[ ]:


# convert ID as the index for the dataframe
fulldata=fulldata.set_index('ID')
fulldata.head()


# ### Detailed Attribute Analysis

# Detailed attribute analysis allows one to get a detailed picture of dataset. It helps discover:
# 1. Extent of missing values in each columns
# 1. Range of categorical values each attribute takes along with frequency of each
# 1. Average values of numeric attributes
# 1. Anomalous/extreme/outlier values present in data
# 1. Deviations, if any, from the data dictionary
# 1. Irrelevant attributes(e.g. attributes that take only one value)
# 1. Distribution of target labels

# **P4. Study attribute value distribution**
# 
# ***Numerical Attributes***

# describe() function of pandas DataFrame gets a summary of all numeric attributes. It can be pushed to get summary for non-numeric attributes too but that part is not comprehensive so we don't use it that often.
# 
# Use describe() function below to get profile summary of numeric attributes

# In[ ]:


# Use describe() function of pandas DataFrame to get a summary of all numeric attributes. Use a .T at the end of the function call to make the output more readable
# TODO fill in the describe command here


# **Observations**
# * No. of columns having one or more missing value is 0
# * % of cases having default is 22%
# * median age in the dataset is 34 while minimum is 21

# In[ ]:


# Use KDE PLOT to get detailed distribution for each attribute
for aCol in fulldata.columns:
    if fulldata[aCol].dtype==object:
        continue
    print('Column:',aCol)
    sns.kdeplot(fulldata[aCol],shade=True)
    plt.show()


# ***Categorical Attribute***

# value_counts() function of pandas DataFrame column(Pandas Series) allows to get frequency distribution for categorical variables. Check it out in action below

# In[ ]:


# Use _value_counts() to plot values using histogram
for aCol in fulldata.columns:
    if fulldata[aCol].dtype==object:
        if aCol=='ID':
            continue
        print(aCol)
        print('----------------------------')
#         plt.figure(figsize=(15,5))
        sns.barplot(fulldata[aCol].value_counts().index,fulldata[aCol].value_counts())
        plt.show()
        print(fulldata[aCol].value_counts())


# **Observation**
# - What is odd about attribute values for marriage and education, compared to data dictionaries
#     - TODO

# # Data Split: Training data and Test data

# **How do we measure model performance**
# - During training we want the model to learn just the right level of rules for classification
# - If model learns more detailed rules than necessary, it will make correct predictions on existing data but incorrect predictions on new data
# - If model learns less detailed rules than necessary, it will make incorrect predicitons on existing data as well as new data
# 
# **Held Out Data Sample Setup**
# - To evaluate if the model has learnt the right level of rules:
#     - we hide part of labeled data, called test data, from training process
#     - train the model on remaining available data called training data
#     - evaluate model performance on training data
#     - evaluate model performance on test data
#     - ensure that performance is as high as possible on test data while being similar to performance on training data
# 
# 
# Typical Ratio of training data and test data size: 8:2

# **Split the read data into training and test data points in the ratio 8:2**
# - Function train_test_split in sklearn.model_selection package provides a functionality to split training and test data sets
# - Typically different runs of training and test data split give different results due to randomization
# - To ensure that we get same results with each randomization, we provide a specific random_state value to the train_test_split function

# In[ ]:


rseed=11 # ensures reproducibility of results, detailed later


# In[ ]:


from sklearn.model_selection import train_test_split # helps split the data into multiple components


# In[ ]:


# split into X and y
fullX=fulldata.iloc[:,:-1]
fully=fulldata.iloc[:,-1]


# In[ ]:


# use train validation test split using command from sklearn to split into Train, Validation and Test
# TODO use train_test_split() to split the fullX and fullY


# # Feature Creation

# - Since algorithms expect everything to be numeric, we need to modify the categorical variables to numeric columns before feeding them to the model
# - One-hot encoding is a way to convert categorical variables to numeric columns
# 
# **Converting all categorical variables to numeric variables using an object of OneHotEncoder**

# In[ ]:


# convert categorical to one hot
catCols=[]
i=-1
for aCol in trainX.columns:
    i+=1
    if trainX[aCol].dtype != object:
        continue
    catCols.append(i)
    print(aCol)
print('Categorical Features:',catCols)
ohe=sk.preprocessing.OneHotEncoder(categorical_features=catCols)
ohe=ohe.fit(trainX)
trainX2=pd.DataFrame(ohe.transform(trainX).toarray())
testX2=pd.DataFrame(ohe.transform(testX).toarray())


# In[ ]:


# checking what trainX2 looks like
# TODO use the head command here


# In[ ]:


# values identified
ohe.categories_


# In[ ]:


# comparing with initial dataframe
trainX.head()


# With all attribute columns being numeric we can now proceed towards training a classification model.

# # Train Model

# - There are many models that can be used for classification, like, Logistic Regression, Decision Trees and Neural Networks
# - Various models differ in what types of rules they can create to do classification and how they discovered those rules from the data
# - We will try Decision Trees for classifying our data in this exercise
# - Decison trees create a rules in form of a tree like flowchart
# - Decision trees are available in sklearn.tree library in form of class DecisionTreeClassifier
# 
# **Create a decision tree classifier and train it using training data prepared above**

# In[ ]:


# import decision tree module from sklearn
from sklearn.tree import DecisionTreeClassifier

# create a DecisiopnTreeClassifier() object
model=... # TODO createa a decision tree classifier


# - Once we have the classifier object, sklearn allows us to train the classifier by using the .fit() function
# - .fit() function takes in training data features, X, and training data labels, y as parameters
# 
# **Use .fit() function of the model to train classifier**

# In[ ]:


# use the fit function on the model to train the model using training data features and training data labels as parameters
# syntax model.fit(training data features, training data labels)
# TODO put the fit() command here


# - once trained .predict() function of the classifier can be used to make predictions
# 
# **Use .predict() function to get predictions on training and test data sets**

# In[ ]:


# use the predict function on training and test data to come up with training data predictions
# syntax model.predict(features)
trainp=... # call the predict function on train features
testp=... # call the predict function on test features


# # Model Performance Evaluation

# - Once predictions are available, model performance can be evaluated using functions like accuracy_score() from sklearn.metrics module
# 
# ### Metric: Accuracy
# 
# **Estimate the accuracy score for training and test datasets**

# In[ ]:


print('training dataset accuracy:',sk.metrics.accuracy_score(trainy,trainp))


# In[ ]:


print('test dataset accuracy:',...) # TODO fill in the ... with accuracy_score function for test data


# - Although accuracy looks great, note that we can obtain an accuracy of 78% without any learning algorithm
# - Because of above accuracy is not a widely used metric for classification problems
# - Instead of accuracy people use other metrics like Precision, Recall and F1 to capture model performance
# - Most of the metrics are based on confusion matrix which is what we would like to work on next

# ### Evaluation: Confusion Matrix
# 
# **Plot the confusion matrix for training and test datasets**

# In[ ]:


print('TRAINING DATA')
plt.figure(figsize=(4,4))
sns.heatmap(sk.metrics.confusion_matrix(trainy,trainp),annot=True,fmt='d',linewidths=0.5,annot_kws={'size':20})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


print('TESTING DATA')
plt.figure(figsize=(4,4))
sns.heatmap(sk.metrics.confusion_matrix(testy,testp),annot=True,fmt='d',linewidths=0.5,annot_kws={'size':20})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

