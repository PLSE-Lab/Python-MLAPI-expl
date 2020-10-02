#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


#Print all rows and columns. Dont hide any
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train.head(3)


# In[ ]:


print(train.shape, test.shape)


# This data set contains 116 categorical columns plus 14 constant columns and one predictor (loss) column

# In[ ]:


train.describe()


# __Data Cleaning__

# In[ ]:


#Checking missing values
print('Is the training data contains any missing values? ' + str(train.isnull().any().any()) + '\n'
     + 'Is the testing data contains any missing values? ' + str(test.isnull().any().any()))


# In[ ]:


train_list_name = list(train.columns.values)
train_list_name.pop() #pop out the loss column
test_list_name = list(test.columns.values)
print('Are the columns identical to each other for both train & test dataset? ' + str(train_list_name == test_list_name))


# In[ ]:


#Check column values for each categorical columns
def showunique(df):
    list_name = list(train.columns.values)
    for i, col_name in enumerate(list_name):
        if col_name[:3] == 'cat':
            print(df.groupby('cat' + str(i))['id'].nunique())


# In[ ]:


showunique(train)


# - The takeaways from this column unique value check shows that from cat1 to cat73, we only have 2 selection choices A and B.<br />
# 
# For these columns some of them have uneven distribution of As and Bs, which makes the values not important, __for example__: <br />
# 
# In cat 70 there are 188295 As (99.98%) and 23Bs (0.02%). These columns can be removed to reduce the dimensionality of our model<br /> 
# 
# - As for cat73 to cat76 we have A, B and C. <br />
# 
# - From cat77 to cat87 we have A, B, C and D.<br />
# 
# - Cat88 has value A, B, D, E. <br />
# 
# - Starting from cat89 we have more than 4 unique values for each categorical features. <br />
# 

# ## __First of all, Let us take care of the constant part of the dataset__

# In[ ]:


#Separate the dataset, starting from column index 117
train_cont = train.iloc[:, 117:]
test_cont = test.iloc[:, 117:]


# In[ ]:


train_cont.head(3)


# In[ ]:


test_cont.head(3)


# In[ ]:


#Checking the skewness of the remaining dataset, the ones close to 0 are less skewed data
print(train_cont.skew())


# In[ ]:


#Heatmap to check the correlation
cor = train_cont.corr()
f, ax = plt.subplots(figsize = (12, 8))
sns.heatmap(cor, vmax = 0.9, annot = True, square = True, fmt = '.2f')


# __Takeaways from the heatmap__ <br />
# 
# - Most of the continous variables are somewhat correlated with the loss (lowest cont13)
# - Some of the variables are strongly correlated with each other (eg: cont1 and cont9 correlation is 0.93)
# - PCA or SVD could be applied to extract the most important features from these variables

# In[ ]:


#Let us apply PCA
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

loss = train_cont.loc[:, train_cont.columns == 'loss']
train_cont_phase = train_cont.loc[:, train_cont.columns != 'loss']

scaled_train_cont = StandardScaler().fit_transform(train_cont_phase)
scaled_test_cont = StandardScaler().fit_transform(test_cont)


# In[ ]:


#Do the PCA
pca = PCA()
pca.fit(scaled_train_cont)
pca.data = pca.transform(scaled_train_cont)


# In[ ]:


#Percentage variance of each pca component stands for
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
#Create labels for the scree plot
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]


# In[ ]:


#Plot the data
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label = labels)
plt.ylabel('percentage of Explained Variance')
plt.xlabel('Principle Component')
plt.title('Scree plot')
plt.show()


# In[ ]:


variance = 0
count = 0
for i in pca.explained_variance_ratio_:
    if variance <= 95:
        variance += i * 100
        count+=1
print(str(np.round(variance, 2)) + '% of the variance is explained by ' + str(count) + ' of Principle Components')


# __Thus, we are going to use 9 Principle components to preserve 96.8% of the entire 14 constant part of variance__

# In[ ]:


#Extract the PC1 through PC9 information
train_append = pd.DataFrame(data=pca.data[:,:9], columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'
                                                           , 'PC8', 'PC9'])


# In[ ]:


train_append.head(3)


# In[ ]:


#Glue the PC data back to the training dataset
new_train = pd.concat((train.iloc[:, :117], train_append), axis = 1)
new_train.head(3)


# In[ ]:


#Now performing the same action for the testing dataset
pca.fit(scaled_test_cont)
pca.data = pca.transform(scaled_test_cont)
test_append = pd.DataFrame(data=pca.data[:,:9], columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'
                                                           , 'PC8', 'PC9'])
new_test = pd.concat((test.iloc[:, :117], test_append), axis = 1)
new_test.head(3)


# __Looking good, now we try to handle the 116 categorical variables__
# 
# __Strategy:__
#  - Remove the noisy columns that has very unevenly distributed columns
#  - One hot encoding the columns (Since the columns are more like the nominal variables)
#  - We are not considering using label encoding since the variables does not look like ordinal

# In[ ]:


#Check the distributions of the catevariables:
# Count of each label in each category

#names of all the columns
cols = new_train.columns

#Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
    for j in range(n_cols):
        sns.countplot(x=cols[i*n_cols+j+1], data=new_train, ax=ax[j])


# According to the graph we plotted: Cat14, 15, 17, 18, 19, 20, 21, 22, 29, 30, 32, 33, 34, 35, 42, 43, 45, 46, 47, 48, 51, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 74, 77, 78, 85 can be categorized as noisy columns
# 
# But we can not just rely on our eyes, we are going to remove them using calculations

# In[ ]:


#Show dominanting percentage less than 2%, and drop them during the process
def show_and_drop_percentage(df, df2):
    for i in range(1, 117):
        A = df['cat' + str(i)].value_counts()
        per = sum(A[1:]) / sum(A) * 100
        if per < 2:
            print('cat' + str(i) + ': ' + 'Dominating percentage is: ' + str(np.round(per, 2)) + '%')
            df = df.drop(['cat' + str(i)], axis = 1)
            df2 = df2.drop(['cat' + str(i)], axis = 1)
    print('-' * 80 + '\n')
    print('Cleaning complete for columns cat1 to cat 116, The above categories had been dropped\n')
    return df, df2


# In[ ]:


#Operate on the training set
removed_train, removed_test = show_and_drop_percentage(new_train, new_test)


# In[ ]:


removed_train.head(3)


# In[ ]:


removed_test.head(3)


# In[ ]:


#Check if the same procedure was done on the train & test columns
any(removed_train.columns == removed_test.columns)


# ### __Now we dummy coding the remaining categorical variables__
# 
# Thanks to: https://www.kaggle.com/sharmasanthosh/exploratory-study-on-ml-algorithms
# provide the method to achieve one-hot encoding considering value difference in values in train & test dataset
# (pandas get dummies would result in unevenly columns due to column value differences)

# In[ ]:


#Remove the id columns
removed_train = removed_train.iloc[:, 1:]
removed_test = removed_test.iloc[:, 1:]


# In[ ]:


removed_train.head(3)


# In[ ]:


#range of features considered
split = 78

#Grab out the categorical variables
cat_train = removed_train.iloc[:, :split]
cat_test = removed_test.iloc[:, :split]

#List the column names
cols = cat_train.columns

#Variable to hold the list of variables for an attribute in the train and test data
labels = []

for i in range(0,split):
    train = cat_train[cols[i]].unique()
    test = cat_test[cols[i]].unique()
    labels.append(list(set(train) | set(test))) 


# __For Training Data Set__

# In[ ]:


#One hot encode all categorical attributes 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(cat_train.iloc[:,i])
    feature = feature.reshape(cat_train.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,categories= [range(len(labels[i]))])
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)
    
# Make a 2D array from a list of 1D arrays
encoded_cats = np.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
train_encoded = np.concatenate((encoded_cats,removed_train.iloc[:,split:].values),axis=1)

#Transfer it back into pandas dataframe
train_encoded = pd.DataFrame(data=train_encoded)
train_encoded.head(3)


# __For Testing Data Set__

# In[ ]:


#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(cat_test.iloc[:,i])
    feature = feature.reshape(cat_test.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,categories= [range(len(labels[i]))])
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)
    
# Make a 2D array from a list of 1D arrays
encoded_cats = np.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
test_encoded = np.concatenate((encoded_cats,removed_test.iloc[:,split:].values),axis=1)

test_encoded = pd.DataFrame(data=test_encoded)
test_encoded.head(3)


# ## __Now we can prepare our data for modeling__
# 
# Models to consider:
#    - linear Regression
#    - Lasso
#    - Ridge
#    - Elastic Net
#    - Stochastic Gradient Descent (SGD)
#    - RandomForest
#    - Xgboost
#    - LightGBM

# In[ ]:


#First of all, we do train test split of our dataset
from sklearn.model_selection import train_test_split

#Set our random seed to ensure productive result
seed = 2019

X_train, X_test, y_train, y_test = train_test_split(
     train_encoded, loss, test_size=0.25, random_state=seed)


# ### __Linear Regression__

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) 
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

#Calculating the MAE
lin_pred = lin_reg.predict(X_test_scaled)
lin_result = mean_absolute_error(y_test, lin_pred)
lin_result


# __the MAE of linear regession probably told us why we should NEVER use simple linear regression for our model__
# 
# __We should go ahead and try some more complex model for our prediction__

# ### __Elastic Net__

# In[ ]:


from sklearn.linear_model import ElasticNet

ela = ElasticNet(random_state=seed)
ela.fit(X_train_scaled, y_train)

ela_pred = ela.predict(X_test_scaled)
ela_result = mean_absolute_error(y_test, ela_pred)
ela_result


# Elastic Net is doing much better, but still with quite large error rate

# In[ ]:


#Make predictions using the model
#Write it to the file
Test_scaled = scaler.transform(test_encoded)

predictions = ela.predict(Test_scaled)

pd.DataFrame(predictions, columns = ['loss']).to_csv('submission.csv')


# ### __Stochastic Gradient Descent (SGD)__

# In[ ]:


from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor(max_iter = 1500, eta0=1e-14,
                  learning_rate = 'adaptive',
                  penalty = 'elasticnet')
sgd.fit(X_train_scaled, y_train)

sgd_pred = sgd.predict(X_test_scaled)
sgd_result = mean_absolute_error(y_test, sgd_pred)
sgd_result


# In[ ]:


sgd_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




