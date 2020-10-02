#!/usr/bin/env python
# coding: utf-8

# This notebook is an attempt to try out the Lasso Regression Algorithm
# 
# First we will do some data exploration to find the missing values and replace them with appropriate values before proceeding to the models.
# 
# We will find the correlation and skew. Remove the unwanted features from the train and test data.
# 
# Convert the categorical features to numeric features using onehot encoder.
# 
# Then check the model on the test data.
# 
# Let's get started!

# In[ ]:


#Import the imported libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Read the data
train = pd.read_csv("../input/train.csv")

#Check the data

print(train.head())
print(train.shape)


# In[ ]:


notnullcount = train.count()
# List the columns with more than 30 % missing values
nullmorethan30 = [n for n in notnullcount if n < 0.3 * train.shape[0]]
removablecolumns =[]
for v in nullmorethan30:
    colr = notnullcount[notnullcount == v].index[0]
    removablecolumns.append(colr)


# In[ ]:


train = train.drop(removablecolumns,1)    


# Now fill the missing numeric values with mean and the non numeric values with the most frequent values#

# In[ ]:


import numpy as np

trainnew = train
columns = trainnew.columns
for col in columns:
    if(trainnew[col].dtype == np.dtype('O')):
        trainnew[col] = trainnew[col].fillna(trainnew[col].value_counts().index[0])
        #print(trainnew[col].value_counts().index[0])
    else:
        trainnew[col] = trainnew[col].fillna(trainnew[col].mean())
        


# In[ ]:


#Check if any value is null
print(trainnew.isnull().any().value_counts())


# So we removed the 4 columns and fill the missing cells.

# # Shape of Data
# 
# The data has 77 columns. 
# 
# As it is clearly visible Id column and saleprice columns will not be used to model the data.
# 
# So we have a total of 75 features. Understanding each feature and it's impact on the sale price is not possible just by looking at the feature and generalizing the meaning, we need to do some plotting.
# 
# But before that we will divide the dataframes into two types - numeric df and non numeric df so that we can deal with both types of values separately.
# 

# Let's remove the column Id data so that we can concentrate on our features and target variable only

# In[ ]:


dataset = trainnew.drop(['Id'], axis = 1)


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

dataset_numeric = dataset.select_dtypes(include=numerics)


# In[ ]:


dataset_numeric.shape


# In[ ]:


nonnumeric = ['object']
dataset_nonnumeric = trainnew.select_dtypes(include=nonnumeric)


# In[ ]:


dataset_nonnumeric.shape


# So we have 36 numeric features, 1 target variable (numeric) and 39 non numeric features

# In[ ]:


# Skewness in the data

dataset_numeric.skew()


# #Skew 
# Some of the columns contain highly skewed data. Let's plot the histogram to better understand the skew
# 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
cols = dataset_numeric.columns
for c in cols:
    sns.violinplot(dataset_numeric[c])
    plt.xlabel(c)
    plt.show()
   


# In[ ]:


# Skew Correction
#log1p function applies log(1+x) to all elements of the column
skew = dataset_numeric.skew()

skewedfeatures = [s for s in skew if(s > 5.0)]
skewedfeatures
for skf in skewedfeatures:
    sk = skew[skew == skf].index[0]
    dataset_numeric[sk] = np.log1p(dataset_numeric[sk])


# #The skew in data now decreased to some extent.
# 
# #Understanding Correlation
# 
# We can see from the above that features like FullBath, GrLivArea, 1stFlrSF are positively correlated with the sale price. It seems price are higher for houses with such first floor sqaure feet, Full bathrooms above grade,Above grade (ground) living area square feet. 
# 
# There are few features which are negatively correlated but not too much.
# 
# Since there are large number of features - and some of the features are not too correlated with the sales price, we can remove them. Lasso regression should be a good choice as it remove some of the unwanted features.

# In[ ]:


correlation= dataset_numeric.corr()


# Le's find the pair of features which are highly correlated so that we can remove any
# ny one such feature

# In[ ]:


# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore categorical data

# Calculates pearson co-efficient for all combinations
data_corr = dataset_numeric.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

size = 36

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# We can see that garagecars and garageare are highly correlated so we can remove one of them. Let's remove the feature garagecars.

# In[ ]:


dataset_numeric = dataset_numeric.drop('GarageCars', axis = 1)
dataset_numeric.shape


# # Now Let's come to the categorical features. We need to convert them to numeric features.
# 
# We will use on hot encoding to do this or we can use some vectorizer to do this. Let's first find the unqiue values of each categorical variables and their counts.

# We will find out the no of unqiue labels

# In[ ]:


cols = dataset_nonnumeric.columns
split = 39
labels = []
for i in range(0,split):
    train = dataset_nonnumeric[cols[i]].unique()
    labels.append(list(set(train)))


# In[ ]:


#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_nonnumeric.iloc[:,i])
    feature = feature.reshape(dataset_nonnumeric.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)


# In[ ]:


# Make a 2D array from a list of 1D arrays
import numpy
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)


# In[ ]:


dataset_encoded = numpy.concatenate((encoded_cats,dataset_numeric.values),axis=1)


# In[ ]:


dataset_encoded.shape


# # There will be 239 features more will be added along with 35 numeric features. Total no of features will be 274 and 1 target variable

# # Make Predictions

# In[ ]:


#Read test dataset
dataset_test = pd.read_csv("../input/test.csv")
#Drop unnecessary columns
ID = dataset_test['Id']
dataset_test.drop('Id',axis=1,inplace=True)
dataset_test.shape


# As while exploring the train data we deleted few columns and also skewed some data. Let's do the same process for test data

# In[ ]:


dataset_test = dataset_test.drop(removablecolumns,1)

import numpy as np

columns = dataset_test.columns
for col in columns:
    if(dataset_test[col].dtype == np.dtype('O')):
        print(5)
        print(dataset_test[col].dtype)
        print(dataset_test[col].value_counts().index[0])
        dataset_test[col] = dataset_test[col].fillna(dataset_test[col].value_counts().index[0])
        #print(trainnew[col].value_counts().index[0])
    else:
        dataset_test[col] = dataset_test[col].fillna(dataset_test[col].mean())
        #print(4)
        print(dataset_test[col].dtype)
datasettest_numeric = dataset_test.select_dtypes(include=numerics)
datasettest_nonnumeric = dataset_test.select_dtypes(include=nonnumeric)

for skf in skewedfeatures:
    sk = skew[skew == skf].index[0]
    datasettest_numeric[sk] = np.log1p(datasettest_numeric[sk])
datasettest_numeric = datasettest_numeric.drop('GarageCars', axis = 1)
datasettest_numeric.shape


# In[ ]:


#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(datasettest_nonnumeric.iloc[:,i])
    feature = feature.reshape(datasettest_nonnumeric.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)


# In[ ]:


# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)


# In[ ]:


encoded_cats.shape


# In[ ]:


#Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats,datasettest_numeric.values),axis=1)


# In[ ]:


X_test.shape


# In[ ]:


#get the number of rows and columns
r, c = dataset_encoded.shape

y_train = dataset_encoded[:,c-1]
X_train = dataset_encoded[:,0:c-1]


# In[ ]:


from sklearn.linear_model import Lasso
ls = Lasso(alpha = 1.0, max_iter = 100)
ls.fit(X_train, y_train)
predictions = ls.predict(X_test)


# In[ ]:



results_dataframe = pd.DataFrame({
    "Id" : ID,
    "SalePrice": predictions
})


# In[ ]:


results_dataframe.to_csv("first_submission.csv", index = False)


# In[ ]:




