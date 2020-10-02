#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier 
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# **Command to hide warnings**

# In[2]:


warnings.simplefilter('ignore')


# **Preprocessing**
# Do not run the following cells since the dataset used is already processed.

# In[ ]:


#Dropping the necessary columns
#taxi_df =taxi_df.drop(['tpep_dropoff_datetime', 'RatecodeID', 'store_and_fwd_flag', 'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge'], axis=1)

#Categorizing time into 1's and 0's
'''
taxi_df["tpep_pickup_datetime"] =pd.to_datetime(taxi_df["tpep_pickup_dat etime"])
taxi_df["tpep_pickup_datetime"] =taxi_df["tpep_pickup_datetime"].dt.hour 
for index, row in taxi_df.iterrows():
    print(index)
    if(taxi_df["tpep_pickup_datetime"][index] > 6 and taxi_df["tpep_pick up_datetime"][index] < 18):
        taxi_df["tpep_pickup_datetime"][index] =1         #morning
    else:
        taxi_df["tpep_pickup_datetime"][index] =0         #night
'''


# **Calculation of Tip_rate_20**

# In[ ]:


'''
taxi_df['tip_rate_20'] = (taxi_df.tip_amount/taxi_df.fare_amount) for index, row in taxi_df.iterrows():
print(index) if(taxi_df["tip_rate_20"][index] >=0.2):
taxi_df["tip_rate_20"][index] =1 else:
        taxi_df["tip_rate_20"][index] =0
taxi_df.head(10)
'''


# **Working on the final processed dataset**

# In[7]:


#sampled the data for 10,000 rows using the library subsample
taxi_df =pd.read_csv('../input/nycyellowtaxi2017/y_2017_01.csv')
taxi_df =taxi_df.fillna('NA')
#Assigning negative values to 0 for tip_amount taxi_df[taxi_df['tip_amount']<0] =0
mean =taxi_df["fare_amount"].mean() 
taxi_df[taxi_df['fare_amount']<0].fillna(mean, inplace =True) #Getting a sample view of the data with 5 rows. 
taxi_df.head(10)


# **Dropping the necessary columns**

# In[8]:


n, bins, patches = plt.hist(x=taxi_df["fare_amount"], bins='auto', color
='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')
plt.xlim([0,100])
plt.title('Distribution of Fare Amount');


# ### Trip Fare Prediction

# **Linear Regression to predict Fare Amount**

# Dividing the dataset into dependent and independent variables

# In[9]:


X =taxi_df[['tpep_pickup_datetime', 'passenger_count','trip_distance','PULocationID', 'DOLocationID', 'payment_type']].values
y =taxi_df["fare_amount"].values
y =np.array([y])
y =y.transpose()


# Applying Linear Regression

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=15)
lm =LinearRegression()
model = lm.fit(X_train,y_train)
y_pred =lm.predict(X_test)


# Applying K-Cross Fold Validation

# In[11]:


#k =5
accuracies =cross_val_score(estimator =lm, X =X_train, y =y_train, cv =5)


# In[12]:


#Accuracy with 20% test size gives around 79%
print("The accuracy is: ", np.mean(accuracies)*100)


# Calculating Mean Squared Errors

# In[13]:


#Calculating MSE off of expected and predicted values
mse =mean_squared_error(y_test, y_pred)
print("Mean Squared Error is: ", mse)
print("Root Mean Squared Error is: ", sqrt(mse))


# Calculating Standard Deviation

# In[14]:


print(np.std(y_test))
print(np.std(y_pred))


# ## Using KNN to predict Fare Amount

# Scaling the training and testing data

# In[15]:


scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)
X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# Calculating the RMSE values to find the optimum value of K

# In[16]:


rmse_val = [] #to store rmse values for different k 
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    
    model.fit(X_train, y_train) #fit the model 
    pred=model.predict(X_test) #make prediction on test set 
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse 
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# ### Plotting Elbow Curve

# In[17]:


curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot();


# ### Using KNN regression to predict and find accuracy of model

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=100)
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
encoded1 = lab_enc.fit_transform(y_test)
for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(2, weights=weights)
    predKNN = knn.fit(X_train, encoded).predict(X_test)
    acc =metrics.r2_score(predKNN, encoded1)
print(acc)


# In[19]:


print("The accuracy for KNN Regression is: ",acc*100)


# ### Results
# We can have used two models i.e Linear Regression and K- Nearest Neighbors (KNN). From the results above, we have seen that KNN Regression has been the better model of the two. For the Linear Regression model, I have taken the feature set and trained it. After this, I have applied a K - Cross fold validation where K=5. This allows us to shuffle the training set and apply a cross fold for every iteration so that there is a less chance of bias. On calculating the accuracy, it comes out to be around 80.3% with a split size of training and testing of 70/30.
# For the KNN model, I had first taken a classification approach, which didn't seem right since I had to predict the value of a particular attribute rather than classify it. Upon training and predicting the values I was getting an accuracy of around 10%. I then went for the KNN Regression approach. After splitting the data into the training and testing sets (70/30 split size), I went about calculating the optimal value of K. For this I calculated the Root Mean Squared Error (RMSE) value for every value of K from 1 - 10. This gave me an elbow plot. The point at which the elbow plot seems to bend is usually taken as the value of K. From the plot above, we can see that for K=2, the plot seems to bend. Taking this value of K, I then ran the model and predicted the values for the fare amount. This gave me an accuracy of around 80.6%
# We can tune the hyper-parameters and take a bigger sample set to probably obtain a higher value of accuracy for the models.

# ## Trip Rate Classification

# **Sampling 1000 rows**

# In[21]:


#used the subsampling utility to randomly sample 1000 rows.
trip_class =pd.read_csv('../input/sampled-nyc-data/y_2017_01_1K.csv')
trip_class.shape


# ## KNN model to predict the tip_rate_20

# In[22]:


X =taxi_df[['tpep_pickup_datetime', 'passenger_count','trip_distance','payment_type']].values
y =taxi_df["tip_rate_20"].values
y =np.array([y])
y =y.transpose()


# Splitting the data into training and testing

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=100)


# Running KNN with K=5

# In[24]:


lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
encoded1 = lab_enc.fit_transform(y_test)
for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsClassifier(n_neighbors =5)
    predKNN = knn.fit(X_train, encoded).predict(X_test)
    acc =metrics.r2_score(predKNN, encoded1)


# In[25]:


accuracies =cross_val_score(estimator =knn, X= X_train, y= y_train, cv=5)


# In[26]:


print("The accuracy is: ", np.mean(accuracies)*100)


# Printing the Precision, Recall and F-Score

# In[29]:


(precision, recall, fscore, support) =precision_recall_fscore_support(y_test, predKNN, average='macro')
print('Precision: {}\tRecall: {}\tF-Score: {}'.format(precision, recall,fscore, support))


# ### Decision Tree to predict tip_rate_20

# Splitting data into training and testing

# In[30]:


X =taxi_df[['tpep_pickup_datetime', 'passenger_count','trip_distance','payment_type']].values
y =taxi_df["tip_rate_20"].values
y =np.array([y])
y =y.transpose()


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=15)
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y_train)
encoded1 = lab_enc.fit_transform(y_test)


# Creating the model for Decision Tree using the Entropy method

# In[32]:


dt =DecisionTreeClassifier(criterion = "entropy", random_state = 15,max_depth=5, min_samples_leaf=10)
dt.fit(X_train, encoded)


# In[33]:


y_pred_en = dt.predict(X_test)
y_pred_en


# Printing the accuracy

# In[34]:


print("Accuracy is ", accuracy_score(encoded1,y_pred_en)*100)


# Applying K-Cross Fold Validation

# In[35]:


# k=5
accuracies =cross_val_score(estimator =dt, X =X_train, y =y_train, cv =5)
print("The accuracy after cross-validation is: ",np.mean(accuracies*100))


# Printing the Precision, Recall and F-Score

# In[38]:


(precision, recall, fscore, support) =precision_recall_fscore_support(encoded1, y_pred_en, average='macro')
print('Precision: {}\tRecall: {}\tF-Score: {}'.format(precision, recall,fscore, support))


# ## Results
# I have applied KNN and Decision Tree algorithms for predicting the tip fare amount (tip_amount_20). For the KNN algorithm, I have chosen K=5 and performed the same procedure as done in Problem 2. In this case I have used the KNN classifier instead of regression. I have also applied K-Fold Cross validation with K=5. The accuracy came out to be around 80%.
# For the Decision Tree classifier, I have used the entropy index. This gave a better performance then when compared to the gini index. The accuracy came out to be around 83% and after cross validation the accuracy came out to be around 82.75%. For cross-validation, K was equal to 5.
# We can also see that the Precision, Recall, and F-score are much better in the Decision Tree classifier than the KNN classifier. This supports our claim in saying that the Decision Tree model was much better than that of the KNN model.
# Thus we can say that Decision Tree algorithm performed better than KNN and we can probably tune the parameters to obtain a higher accuracy.
