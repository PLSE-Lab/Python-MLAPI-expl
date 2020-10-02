#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing the dataset

# In[ ]:


pip install pyxlsb


# In[ ]:


data = pd.read_excel('/kaggle/input/datml.xlsb', engine='pyxlsb')


# # Basic EDA.

# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.nunique()


# In[ ]:


data.count


# In[ ]:


data.isna().sum()


# It clearly says that we have null values

# # Summary of the dataset

# In[ ]:


data.describe().transpose()


# In[ ]:


data.isnull().sum()


# Null Values are there in our dataset.

# Lets check which class problem it is

# In[ ]:


data['converted_in_7days'].nunique()


# It is a 4 class problem.

# #From my basic understanding of the dataset I came to a stage where I would like to take some points into consideration and below are mentioned
# #Removing out some features which i think they are not of any help to my analysis.
# feature selection is done by taking out 
# #1)client id----which doesnot affect the buying pattern.
# #2)country
# #3)date--user last visited date---which doesnt affect buying pattern.
# #4)region
# #5)sourcemedium
# #6)device--mobile/desktop cannot say that it affects the buying pattern.
# 
# #So, will go ahead and drop these columns from our dataset and then treat the variables with missing values 
# 
# 

# Dropping these variables which are not giving anything meaningful to our data.

# In[ ]:


new_data=data.drop(['client_id', 'country', 'date', 'device', 'region', 'sourceMedium'], axis = 1) 


# In[ ]:


new_data.head(10)


# Now we drop our output or predicted variable.

# In[ ]:


new_data2=new_data.drop(['converted_in_7days'],axis=1)


# # Seperating the predictor variables and the predicted variable from the dataset.

# Independent variables

# In[ ]:


new_data_x=new_data2.iloc[:,:].values
new_data_x


# Dependent variable

# In[ ]:


new_data_y=new_data['converted_in_7days']


# In[ ]:


new_data_y.head(2)


# Now we have removed some features which we think that they are of no help to us in the analysis part.
# #As mentioned earlier now that we have the data which is good to go with for our further analysis, we will start doing the operations to the data to make it fit for our model building.
# #So, i will list out all the variables which have missing values and the approach i chose to fill those missing values as per my understanding of the data.
# #1bounces_hist---missing values are present and the strategy to replace this is median
# #2help_me_buy_evt_count_hist---median
# #3pageviews_hist---median
# #4paid_hist---median
# #5phone_clicks_evt_count_hist---median
# #6sessionDuration_hist---median
# #7sessions_hist---median
# #8visited_air_purifier_page_hist--median
# #9visited_checkout_page_hist---median
# #10visited_contactus_hist--median
# #11visited_customer_service_amc_login_hist---median
# #12visited_customer_service_request_login_hist---median
# #13visited_demo_page_hist---median
# #14visited_offer_page_hist---median
# #15visited_security_solutions_page_hist---median
# #16visited_storelocator_hist---median
# #17visited_vacuum_cleaner_page_hist---median
# #18visited_water_purifier_page_hist---median
# 
# lets do the imputation for these variables and then we can go ahead for further analysis

# # Imputing the missing values of the above mentioned variables with the median.

# In[ ]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="median")
imputer.fit(new_data_x[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]])
new_data_x[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]]=imputer.transform(new_data_x[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]])


# In[ ]:


new_data_x


# #Now i got my values imputed with median values and the data is good to go with, i will move a head and do the model building and relevant stuff.

# # Splitting into test and train 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(new_data_x,new_data_y,test_size=0.2)


# *As it is a classification problem and the data size is round off 7 lakh. i rule out the below mentioned algorithms for the reasons mentioned 
# 1) logistic regression---as it is good for 2 class problem and here we have 4 class problem.
# 2) KNN---dataset is too large for knn to handle 
# 3) Svm---dataset is too large to handle.
# 
# And now i will be going with tree and ensemble method.
# *

# # Lets build the tree and ensemble model.

# In[ ]:


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier_tree.fit(x_train, y_train)

# Predicting the Test set results
y_predict = classifier_tree.predict(x_test)


# In[ ]:


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))


# In[ ]:


#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c_tree=confusion_matrix(y_test,y_predict)
print(c_tree)
Accuracy_tree=sum(np.diag(c_tree))/(np.sum(c_tree))
Accuracy_tree


# A 99 % accurate model, but the recall, precision and f1- score are very less which is like a metric trap.

# # Random Forest With Class Weighting

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_ensemble = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0,class_weight='balanced')
classifier_ensemble.fit(x_train, y_train)

# Predicting the Test set results
y_predict1 = classifier_ensemble.predict(x_test)


# In[ ]:


#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c_ensemble=confusion_matrix(y_test,y_predict1)
print(c_ensemble)
Accuracy_ensemble=sum(np.diag(c_ensemble))/(np.sum(c_ensemble))
Accuracy_ensemble


# In[ ]:


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict1))


# Even this is not helping us to treat with an imbalanced dataset. We are falling for this metric trap again.

# # if we carefully observe we are getting into a metric trap which is known as accuracy paradox.
# The accuracy is very high, but the precision recall and f1 score are very low.
# so what we gonna do here is make the imbalanced dataset a balanced one.

# #Here we have two options
# #1 upsampling---good to go with 
# #2 downsampling---problem is we miss some serious critical information if we go with this.
# 
# #Now we will be taking both the samples and check for the best one.
# #1) SMOTE
# #2) Random upsampling

# In[ ]:


data.head(5)


# In[ ]:


data.shape


# Lets see the count of classes in target variable

# In[ ]:


data.converted_in_7days.value_counts()


# It is a 4 class problem and as it is a huge margin and the class 2 and class 3 are very less in number I can remove those two classes from my dataset .

# Dropping classes 2 and 3

# In[ ]:


data.drop(data[data['converted_in_7days'] == 2].index, inplace = True)


# In[ ]:


data.shape


# In[ ]:


data.drop(data[data['converted_in_7days'] == 3].index, inplace = True)


# In[ ]:


data.shape


# Now we have removed class 2 and class 3 from our dataset.

# In[ ]:


data.nunique()


# In[ ]:


data.describe().transpose()


# In[ ]:


data['converted_in_7days'].nunique()


# 2 classes are there now.

# In[ ]:


data.shape


# Dropping the unwanted columns from the dataset

# In[ ]:


new_data1=data.drop(['client_id', 'country', 'date', 'device', 'region', 'sourceMedium','converted_in_7days'], axis = 1) 


# Now we have to see the independentvariables

# In[ ]:


new_data_X=new_data1.iloc[:,:].values
new_data_X.shape


# Dependent variable

# In[ ]:


new_data_Y=data['converted_in_7days']
new_data_Y.shape


# Imputing the missing values 

# In[ ]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="median")
imputer.fit(new_data_X[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]])
new_data_X[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]]=imputer.transform(new_data_X[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]])


# Split into train and test.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(new_data_X,new_data_Y,test_size=0.2)


# random forest classification

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_ensemble1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_ensemble1.fit(X_train, Y_train)

# Predicting the Test set results
y_predict_ensemble1 = classifier_ensemble1.predict(X_test)


# In[ ]:


#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c_ensemble1=confusion_matrix(Y_test,y_predict_ensemble1)
print(c_ensemble1)
Accuracy_ensemble1=sum(np.diag(c_ensemble1))/(np.sum(c_ensemble1))
Accuracy_ensemble1


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_predict_ensemble1))


# * We have removed those two classes and we have got something better now with the random forest classifier
# * Now my model got better when compared to the early one. the metrics are okay when compared to the previous scenario****

# # lets try with smote upsampling

# In[ ]:


pip install imblearn


# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


smt = SMOTE()
X_train1, Y_train1 = smt.fit_sample(X_train, Y_train)


# In[ ]:


np.bincount(Y_train1)


# * random forest now over this SMOTE

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_ensemble3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_ensemble3.fit(X_train1, Y_train1)

# Predicting the Test set results
y_predict_ensemble3 = classifier_ensemble3.predict(X_test)


# In[ ]:


#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c_ensemble3=confusion_matrix(Y_test,y_predict_ensemble3)
print(c_ensemble3)
Accuracy_ensemble3=sum(np.diag(c_ensemble3))/(np.sum(c_ensemble3))
Accuracy_ensemble3


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_predict_ensemble3))


# it is not giving any further improvement in metrics

# # lets try with random upsampling.

# Target variable count

# In[ ]:


target_count = data.converted_in_7days.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])


# Lets see the proportion of the imbalance of the classes.

# In[ ]:


print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');


# In[ ]:


# Class count
count_class_0, count_class_1 = data.converted_in_7days.value_counts()

# Divide by class
df_class_0 = data[data['converted_in_7days'] == 0]
df_class_1 = data[data['converted_in_7days'] == 1]


# In[ ]:


df_class_1_over = df_class_1.sample(count_class_0, replace=True)
data_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(data_over.converted_in_7days.value_counts())

data_over.converted_in_7days.value_counts().plot(kind='bar', title='Count (target)')


# We have done random over sampling and this is the result of it. Both the  classes are having same records now.

# removing the unwanted variables from the dataset

# In[ ]:


new_data4=data_over.drop(['client_id', 'country', 'date', 'device', 'region', 'sourceMedium','converted_in_7days'], axis = 1) 


# Independent variables

# In[ ]:


new_data1_X=new_data4.iloc[:,:].values
new_data1_X.shape


# Dependent variables

# In[ ]:


new_data1_Y=data_over['converted_in_7days']
new_data1_Y.shape


# lets impute the missing values with median

# In[ ]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="median")
imputer.fit(new_data1_X[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]])
new_data1_X[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]]=imputer.transform(new_data1_X[:,[3,15,19,21,23,26,28,33,35,37,39,41,43,45,47,49,52,54]])


# Lets split the data into test and train

# In[ ]:


from sklearn.model_selection import train_test_split
X1_train,X1_test,Y1_train,Y1_test=train_test_split(new_data1_X,new_data1_Y,test_size=0.2)


# # Random Forest

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_ensemble4 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_ensemble4.fit(X1_train, Y1_train)

# Predicting the Test set results
y_predict_ensemble4 = classifier_ensemble4.predict(X1_test)


# In[ ]:


#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c_ensemble4=confusion_matrix(Y1_test,y_predict_ensemble4)
print(c_ensemble4)
Accuracy_ensemble4=sum(np.diag(c_ensemble4))/(np.sum(c_ensemble4))
Accuracy_ensemble4


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(Y1_test,y_predict_ensemble4))


# #We got the best metrics here out of all the models built in various scenarios.
# #I have reached here by taking the random over sampling into the picture.

# # Neural Network 

# Basically what i will be doing is using the MLPClassifier from scikit learn.

# model-1 on data with 4 classes

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)
nn.fit(x_train,y_train)
y_predict_nn1=nn.predict(x_test)


# In[ ]:


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict_nn1))


# Same problem exists here as well.

# Model no-2 on data with 2 classes

# In[ ]:


nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)
nn.fit(X_train,Y_train)
y_predict_nn2=nn.predict(X_test)


# In[ ]:


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(Y_test,y_predict_nn2))


# Slightly better when compared with the previous one

# Model no 3 on data of random over sampling

# In[ ]:


nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)
nn.fit(X1_train,Y1_train)
y_predict_nn3=nn.predict(X1_test)


# In[ ]:


#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(Y1_test,y_predict_nn3))


# Slightly better metrics over the previois models build using MLP classifier.

# # Now let us build using keras package

# Lets try to implement on imbalanced dataset directly

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


# In[ ]:


from keras.utils import to_categorical
y_binary = to_categorical(new_data_Y)


# In[ ]:


from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test=train_test_split(new_data_X,y_binary,test_size=0.2)


# In[ ]:


model = Sequential()

model.add(Dense(1500, activation='relu', input_dim=56))
model.add(Dropout(0.5))
model.add(Dense(750, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

sgd= SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


history=model.fit(x2_train,y2_train,epochs=1)
score=model.evaluate(x2_test,y2_test)


# In[ ]:



model.metrics_names


# In[ ]:


score

