#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score,auc, classification_report, confusion_matrix, f1_score, roc_curve

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[ ]:


#reading the dataset

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head(10)


# In[ ]:


#checking for null values
print("Missing Values:- \n", df.isnull().sum())
print("_________________________________________________________")
print()
print("Target Value Count\n", df['Class'].value_counts())
print()


# In[ ]:


# Checking summary of the data

df.describe()


# In[ ]:


# Visualizing different features using histogram

df.hist(figsize=(10, 10))
plt.tight_layout()
plt.show()


# In[ ]:


# Arranging the columns 

cols = ['Time', 'Amount' ,'V1','V2','V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class']
df = df[cols]


# In[ ]:


# Visualizing the features based on the class 0(No Fraud) or 1(Fraud)

plt.figure(figsize = (10, 80))
count1 = 1
for i in (df.columns):
    
    ax = plt.subplot(16, 2, count1)
    
    sns.distplot(df[i][df["Class"] == 1], bins = 100, ax = ax, color='blue' )
    sns.distplot(df[i][df["Class"] == 0], bins = 100 , ax = ax, color = 'orange')
    
    ax.set_xlabel('')
    ax.set_title('Histogram for feature: ' + str(i))
    count1 = count1+1

plt.show()


# In[ ]:


# Checking the summary of data based on the class 0 (No Fraud)

df[df["Class"] == 0].describe()


# In[ ]:


# Checking the summary of data based on the class 1 (Fraud)

df[df['Class'] == 1].describe()


# In[ ]:


# From the above plots I saw that lot of features are skewed  
#So, I tried to convert those features into boolean features to see if I can improve my accuracy. 

# Created a new dataframe to check the accuracy with new boolean features
new_df = df.copy()

#Converting features to boolean features by using the 75 percentile as the threshold value for 0 and 1
new_df['V1'] = new_df['V1'].map(lambda x: 1 if (x<-0.41) else 0)
new_df['V2'] = new_df['V2'].map(lambda x: 1 if (x<4.97) else 0)
new_df['V3'] = new_df['V3'].map(lambda x: 1 if (x<-2.27) else 0)
new_df['V4'] = new_df['V4'].map(lambda x: 1 if (x<6.34) else 0)
new_df['V5'] = new_df['V5'].map(lambda x: 1 if (x<0.21) else 0)
new_df['V6'] = new_df['V6'].map(lambda x: 1 if (x<-0.41) else 0)
new_df['V7'] = new_df['V7'].map(lambda x: 1 if (x<-0.945) else 0)
new_df['V8'] = new_df['V8'].map(lambda x: 1 if (x<1.76) else 0)
new_df['V9'] = new_df['V9'].map(lambda x: 1 if (x<-0.787) else 0)
new_df['V10'] = new_df['V10'].map(lambda x: 1 if (x<-2.61) else 0)
new_df['V11'] = new_df['V11'].map(lambda x: 1 if (x<5.30) else 0)
new_df['V12'] = new_df['V12'].map(lambda x: 1 if (x<-2.97) else 0)
new_df['V13'] = new_df['V13'].map(lambda x: 1 if (x<0.67) else 0)
new_df['V14'] = new_df['V14'].map(lambda x: 1 if (x<-4.28) else 0)
new_df['V15'] = new_df['V15'].map(lambda x: 1 if (x<0.609) else 0)
new_df['V16'] = new_df['V16'].map(lambda x: 1 if (x<-1.22) else 0)
new_df['V17'] = new_df['V17'].map(lambda x: 1 if (x<-1.34) else 0)
new_df['V18'] = new_df['V18'].map(lambda x: 1 if (x<0.092) else 0)
new_df['V19'] = new_df['V19'].map(lambda x: 1 if (x<1.649) else 0)
new_df['V20'] = new_df['V20'].map(lambda x: 1 if (x<0.822) else 0)
new_df['V21'] = new_df['V21'].map(lambda x: 1 if (x<1.244) else 0)
new_df['V22'] = new_df['V22'].map(lambda x: 1 if (x<0.617) else 0)
new_df['V23'] = new_df['V23'].map(lambda x: 1 if (x<0.308) else 0)
new_df['V24'] = new_df['V24'].map(lambda x: 1 if (x<0.285) else 0)
new_df['V25'] = new_df['V25'].map(lambda x: 1 if (x<0.456) else 0)
new_df['V26'] = new_df['V26'].map(lambda x: 1 if (x<0.396) else 0)
new_df['V27'] = new_df['V27'].map(lambda x: 1 if (x<0.827) else 0)
new_df['V28'] = new_df['V28'].map(lambda x: 1 if (x<0.381) else 0)


# ##### Time is the seconds elapsed between each transaction and the first transaction in the dataset and is represented as numbers. There is a lot of variation in the first value and the last value so I applied standard scaler in order to standardize the results. 
# ##### Amount feature also needs to be scaled because there is also a huge difference in the values of amount. 
# 
# ##### Also, the mean of both these columns varies a lot as compared to other features and Time and Amount will dominate other features. Therefore, I scaled both of these features

# In[ ]:


# Using standard Scaler to Standardize the Time and Amount Feature
sc = StandardScaler()

df["Time"] = sc.fit_transform(df['Time'].values.reshape(-1,1))
df['Amount'] = sc.fit_transform(df['Amount'].values.reshape(-1,1))

df.head()


# In[ ]:


#Visualizing the boxplot to see if there are any outliers

plt.figure(figsize = (10,10))
sns.set(style= 'whitegrid')
sns.boxplot(data = df, palette= 'Set3')
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


df.shape


# In[ ]:


# Checking the interquartile. This will help in removeing the outliers

from scipy import stats

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


# Removing the outliers and creating a new dataframe. 

df_out = df[~((df < (Q1 - 3 * IQR)) |(df > (Q3 + 3 * IQR))).any(axis=1)]
df_out.shape


# In[ ]:


# Checking the counts of two classes after removing the outliers. But it seems that all are class 1 value have been removed 
# and therefore I will not remove outliers from my data. 
df_out['Class'].value_counts()


# In[ ]:


# Setting the width and height of the figure
plt.figure(figsize=(8,4))

# Adding title
plt.title("Fraud vs Non - Fraud Cases")

# count plot with fraud and non fraud cases
sns.countplot('Class', data=df)

# Add label for vertical axis
plt.ylabel("Count")


# ##### As I can see, this is a highly imbalanced data with very few non fraud cases. I will try using techniques like oversampling, under sampling and random sample to determine the better results.
# 
# ##### But first, I will  split my data into training and testing dataset so that I can keep some data aside for final testing.

# In[ ]:


#Creating X data frame without the target variable

X = df.drop("Class", axis = 1)

# Creating X_processed dataframe with V1, V2...V28 as boolean features
X_processed = new_df.drop("Class", axis = 1)

# Creating y which has target values
y = df["Class"]

#Created y_processed which has target Values
y_processed = new_df["Class"]


# In[ ]:


#Spliting the data set into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 0)


X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_processed, y_processed, test_size = 0.3, stratify = y_processed, random_state = 0)


# ##### First I will create simple Logistic Regression, Random Forest Classifier and XGBoost without setting the class weight.
# 
# #### And to easily access the models, I have created different functions: 
# ####  1. modelling() - This function will run different models and will return the accuracy scores, recall scores, confusion matrix data to plot confusion matrix, roc curve data (fpr, tpr and auc) to plot roc curve and precision recall data to plot the precision recall curve.
# ####  2. conf_matrix() - This function will plot the confusion matrix from the data that I have got from the 1st function.
# ####  3. roc_curve_plot() - This function will plot the ROC curve.
# ####  4. precision_recall_curve_plot() - This function will plot the Precision Recall Curve.

# In[ ]:


# Creating a modelling() function to run different models

from sklearn.metrics import recall_score

# Function definition
def modelling(model_list, X1, X2, y1, y2):
    
    
    X_train = X1
    X_test  = X2 
    y_train = y1 
    y_test  = y2
    
    matrix_data = {}    # dictionary to store confusion matrix data
    auc_data = {}       # dictionary to store the ROC curve data
    precision_recall_data = {}  # dictionary to store the Precision Recall Curve Data
     
    # For loop to access different models that are stored in dictionary

    for name, models in model_list.items():
        model = models
        model.fit(X_train, y_train)     #training a model
        
        y_pred = model.predict(X_test)  # predicting the results 
        
        y_probs = model.predict_proba(X_test)    # predicting the probabilities of both the classes
        y_probs = y_probs[:,1]                   # selecting the probability of class 1
        
        matrix_data[name] = confusion_matrix(y_test, y_pred, normalize = None)   # storing the confusion matrix data 
        
        fpr, tpr, threshold = roc_curve(y_test, y_probs)    # calculating the fpr and tpr values
        
        auc_score = roc_auc_score(y_test, y_probs)    # calculating the auc score 
        
        auc_data[name] = [fpr, tpr, auc_score]  # storing the fpr, tpr and auc_score in auc_data 
        
        model_precision, model_recall, _ = precision_recall_curve(y_test, y_probs)   #calculating the precision and recall

        model_f1 = f1_score(y_test, y_pred)             #calculating the f1 score of a model

        model_auc = auc(model_recall, model_precision)   # calculating the auc of precision recall curve
        
        # storing precision, recall, f1 and auc values in precision_recall_data dictionary
        precision_recall_data[name] = [model_precision, model_recall, model_f1, model_auc]  

        # Printing the different scores of the models. Summarizing the result of each model
        print("********************************************************************************")
        print('Model: {} \t F1 Score = {:.3f} auc(precision_recall_curve) = {:.3f}\n'.format(name, model_f1, model_auc))
        print('Model: {} \t Recall Score = {:.3f} \n'.format(name, recall_score(y_test, y_pred)))
        print("Model: {} \t Accuracy of Tranining Data is: {}\n".format(name, accuracy_score(y_train, model.predict(X_train))))
        print("Model: {} \t Accuracy of Test Data is: {}\n".format(name, accuracy_score(y_test, y_pred)))
        print("Model: {} \t Classification report is:\n".format(name))
        print((classification_report(y_test, y_pred)))
        print("*********************************************************************************")
    
    return matrix_data, auc_data, precision_recall_data


# In[ ]:


# Plotting the confusion matrix 

# function definition 
def conf_matrix(data, Labels = ['No-Fraud', 'Fraud']):
    
    count = 1
    # using for loop to access the data (its a dictionary)
    for name, data in matrix_data.items():
        
        # setting the figure size
        plt.figure(figsize=(8,16), facecolor='white')
        
        # assigning the number of subplots and with each count increment the axis
        ax = plt.subplot(3, 1,count)
        
        # using seaborn heatmap to plot confusion matrix
        sns.heatmap(data, xticklabels= Labels, yticklabels= Labels, annot = True, ax = ax, fmt= 'g')
        
        # setting the title name
        ax.set_title("Confusion Matrix of {}".format(name))
        
        plt.xlabel("Predicted Class") #setting the x label
        plt.ylabel("True Class")   # setting the y label
        plt.show()
        count = count +1   # incrementor to increase the value of axis with each loop 


# In[ ]:


from sklearn.metrics import roc_curve

# function definition for roc curve plot 
def roc_curve_plot(data):
    
    
    # Setting the figure size
    plt.figure(figsize=(8,6))
    
    # using the for loop to access the data dictionary to plot roc curve
    for name, data in data.items():
        plt.plot(data[0], data[1],label="{}, AUC={:.3f}".format(name, data[2]))
    
    # plotting a linear line
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    # setting the x ticks and x label
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)
    
    # setting the y ticks and y label
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    
    # setting the title of the plots and plotting the legent
    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
        


# In[ ]:


# function definition to plot the precision recall curve

def precision_recall_curve_plot(data):
    # setting the figure size 
    plt.figure(figsize=(8,6))
    
    # using for loop to access the precision recall data dictionary 
    for name, data in data.items():
        # plotting the precision recall curve
        plt.plot(data[1], data[0], marker='.', label='{}, AUC = {:.3f}'.format(name, data[3]))
    
    # plotting a flat line
    single_line = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [single_line,single_line], linestyle='--')


    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    


# In[ ]:


df["Class"].value_counts()


# In[ ]:


492/284315


# In[ ]:


# Creating a baseline model without weight adjustments
model_list1 = {"Logistic": LogisticRegression(max_iter = 2000), "RandomForest": RandomForestClassifier(), "XGBoost": XGBClassifier()}


# Creating a dictionary of different models with weight adjustments
model_list__with_class_weights = {"Logistic": LogisticRegression(max_iter = 2000, class_weight= {0:0.01, 1:1}), "RandomForest": RandomForestClassifier(class_weight={0:0.01, 1:1}), "XGBoost": XGBClassifier(class_weight= {0:0.01, 1:1})}


# In[ ]:



# running the models (without weight adjustments) using modeeling function 
matrix_data, auc_data, precision_recall_data = modelling(model_list1, X_train, X_test, y_train , y_test)


# In[ ]:


# Building the confusion matrix of the three models

conf_matrix(matrix_data)


# In[ ]:


# plotting the ROC curve of the three models (without weight adjustments)

roc_curve_plot(auc_data)


# In[ ]:


# plotting the precision recall curve with wright adjustments

precision_recall_curve_plot(precision_recall_data)


# In[ ]:


y_test.value_counts()


# ### Trying the same three machine learning models by adjusting the class_weights

# In[ ]:


# Running the different models with weight adjustments using the modelling() function 

matrix_data, auc_data, precision_recall_data = modelling(model_list__with_class_weights, X_train, X_test,y_train, y_test)


# In[ ]:


# plotting the confusion matrix 

conf_matrix(matrix_data)


# In[ ]:


# Plotting the ROC curve 

roc_curve_plot(auc_data)


# In[ ]:


# Plotting the precision recall curve (with weight adjusted models)

precision_recall_curve_plot(precision_recall_data)


# ### Trying Under Sampling (Near Miss) with all the above algorithms

# In[ ]:


#let us try undersampling the negative class (i.e., low rating)
from imblearn.under_sampling import NearMiss 
nm = NearMiss() 

X_resampled, y_resampled = nm.fit_sample(X_train, y_train) 

matrix_data, auc_data, precision_recall_data = modelling(model_list1, X_resampled, X_test, y_resampled, y_test)


# In[ ]:


# Plotting the the confusion matrix for undersampled data
conf_matrix(matrix_data)


# In[ ]:


# Plotting the roc curve for undersampled data 

roc_curve_plot(auc_data)


# In[ ]:


# plotting the precision recall curve for undersampled data 

precision_recall_curve_plot(precision_recall_data)


# In[ ]:


#let us try undersampling the negative class (i.e., low rating)

# Using RandomUnderSampler this time to see if we can improve an accuracy a little bit
from imblearn.under_sampling import RandomUnderSampler 

rus = RandomUnderSampler() 

X_resampled, y_resampled = rus.fit_sample(X_train, y_train) 

matrix_data, auc_data, precision_recall_data = modelling(model_list1, X_resampled, X_test, y_resampled, y_test)


# In[ ]:


# plotting the confusion matrix 

conf_matrix(matrix_data)


# In[ ]:


# plotting the Roc Curve for undersample data

roc_curve_plot(auc_data)


# In[ ]:


# plottiing the precision recall curve for under sampled data 

precision_recall_curve_plot(precision_recall_data)


# ### Oversampling of  data using SMOTE and ADASYN

# In[ ]:


# Using SMOTE to oversample the data 

from imblearn.over_sampling import SMOTE

smt = SMOTE()

# creating oversampled data from training dataset 
X_oversampled, y_oversampled = smt.fit_sample(X_train, y_train)

# running the oversampled data using modelling function
matrix_data, auc_data, precision_recall_data = modelling(model_list1, X_oversampled, X_test, y_oversampled, y_test)


# In[ ]:


# plotting confusion matrix of oversampled data 

conf_matrix(matrix_data)


# In[ ]:


# plotting ROC curve for undersampled data 

roc_curve_plot(auc_data)


# In[ ]:


# plotting the precision recall curve for oversampled data 

precision_recall_curve_plot(precision_recall_data)


# In[ ]:


# Using another over sampling algorithm ADASYN

from imblearn.over_sampling import ADASYN 
ada = ADASYN() 

# creating the oversampled data from training data set 
X_resampled, y_resampled = ada.fit_sample(X_train, y_train) 

# running the models using over sampled data 
matrix_data, auc_data, precision_recall_data = modelling(model_list1, X_resampled, X_test, y_resampled, y_test)


# In[ ]:


# plotting a confusion matrix of oversampled datga  

conf_matrix(matrix_data)


# In[ ]:


# plotting the ROC curve for oversampled data 

roc_curve_plot(auc_data)


# In[ ]:


# plotting the precision recall curve for oversampled data 

precision_recall_curve_plot(precision_recall_data)


# #### Trying Deep Learning using Keras and Tensorflow to see if recall accuracy is improved

# In[ ]:


# importin the keras libraries to create a deep learning model

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

# creating a deep learning model 

deep_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])


# In[ ]:


# compiling a model
deep_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# fitting my training data to the deep learning model
deep_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)


# In[ ]:


# predicting the values
deep_model_pred = deep_model.predict_classes(X_test, batch_size=200, verbose=0)


# In[ ]:


# creating a confusion matrix 

deep_model_cm = confusion_matrix(y_test, deep_model_pred)
labels = ['No Fraud', 'Fraud']


# In[ ]:


#function to display confusion matrix
def draw_matrix(conf_matrix, LABELS = ["Low Rating", "High Rating"]):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, xticklabels=LABELS,
                yticklabels=LABELS, annot=True, fmt="g");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    
draw_matrix(deep_model_cm)


# In[ ]:


print("Accuracy of a deep learning model is: {}".format(accuracy_score(y_test, deep_model_pred)))
print("Recall score of a deep learning model is: {}".format(recall_score(y_test, deep_model_pred)))
print("F1 Score is: {}".format(f1_score(y_test, deep_model_pred)))
print("\n Classification report is: ")
print(classification_report(y_test, deep_model_pred))


# In[ ]:


y_probs =   deep_model.predict(X_test, batch_size=200, verbose=0)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_probs)    # calculating the fpr and tpr values
        
auc_score = roc_auc_score(y_test, y_probs)    # calculating the auc score 
        
#calculating the precision and recall        
deep_precision, deep_recall, _ = precision_recall_curve(y_test, y_probs)  

deep_f1 = f1_score(y_test, deep_model_pred)             #calculating the f1 score of a model

deep_auc = auc(deep_recall, deep_precision)   # calculating the auc of precision recall curve
        


# In[ ]:


# Setting the figure size
plt.figure(figsize=(8,6))
    
# using the for loop to access the data dictionary to plot roc curve
plt.plot(fpr, tpr,label="Deep Learning Model, AUC={:.3f}".format(auc_score))
    
# plotting a linear line
plt.plot([0,1], [0,1], color='orange', linestyle='--')

# setting the x ticks and x label
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)
    
# setting the y ticks and y label
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
# setting the title of the plots and plotting the legent
plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
        
    
   
    


# In[ ]:


# setting the figure size 
plt.figure(figsize=(8,6))

# plotting the precision recall curve
plt.plot(deep_recall, deep_precision, marker='.', label='Deep Learning Model, AUC = {:.3f}'.format(deep_auc))
    
# plotting a flat line
single_line = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [single_line,single_line], linestyle='--')


# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# ### Summary

# Implement SMOTE() for oversampling our imbalanced dataset actually helped in improving the recall score of the model along with maintaining the precision score of the model. Out of the other models I have used the best performance was by the XGBoost followed by the Random Forest Classifier. The highest recall accuracy I got was 96.6 from the Random Forest Classifier when I used the under sampling technique called Near Miss. But with undersampling, there was a trade off with the precision score and recall score. I got the highest recall score but my precision for class 1 was very low which means that I was misclassifying the No-Fraud as Fraud classes. 
# I also tried deep learning model with Keras and I recieved the recall score of 74.3. 
# 
# The best model which I have got is the Random Forest Classifier and XGBoost when I have used SMOTE and ADASYN for oversampling with recall, f1 and AUC scores more than 80. These are the best models since they are correctly identifying the fraud and no fraud.
# 
# 

# <table border=0 cellpadding=0 cellspacing=0 width=898 style='border-collapse:
#  collapse;table-layout:fixed;width:674pt'>
#  <col width=290 style='mso-width-source:userset;mso-width-alt:10325;width:218pt'>
#  <col width=154 style='mso-width-source:userset;mso-width-alt:5489;width:116pt'>
#  <col width=169 style='mso-width-source:userset;mso-width-alt:6001;width:127pt'>
#  <col width=118 style='mso-width-source:userset;mso-width-alt:4181;width:88pt'>
#  <col width=86 style='mso-width-source:userset;mso-width-alt:3043;width:64pt'>
#  <col width=81 style='mso-width-source:userset;mso-width-alt:2872;width:61pt'>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 width=290 style='height:14.4pt;width:218pt'>Model
#   Name</td>
#   <td class=xl65 width=154 style='border-left:none;width:116pt'>Accuracy Score
#   (Training)</td>
#   <td class=xl65 width=169 style='border-left:none;width:127pt'>Accuracy Score
#   (Testing)</td>
#   <td class=xl65 width=118 style='border-left:none;width:88pt'>F1 Score</td>
#   <td class=xl65 width=86 style='border-left:none;width:64pt'>Recall Score</td>
#   <td class=xl65 width=81 style='border-left:none;width:61pt'>AUC Score</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Logistic
#   Regression<span style='mso-spacerun:yes'>  </span>(No weights adjusted)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.926</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.92</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>73.2</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>62.8</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>74</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Logistic
#   Regression (Weights Adjusted)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.57</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.5</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>42.1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>85.1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>72</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Logistic
#   Regression (Near Miss)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>97.09</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>53.5</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>0.7</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>93.2</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>3</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Logistic
#   Regression (Random UnderSampler)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>94.9</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>97.8</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>12.8</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>89.9</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>70</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Logistic
#   Regression (SMOTE)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>95</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>97.6</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>11.8</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>91.2</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>74</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Logistic
#   Regression (ADASYN)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>88.95</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>91.5</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>3.6</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>92.6</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>76</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Random Forest
#   Classifier (No weights adjusted)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.94</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>82.2</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>71.6</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>83</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Random Forest
#   Classifier (Weights Adjusted)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.94</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>82.3</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>72.3</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>84.7</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Random Forest
#   Classifier (Near Miss)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>3.95</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>0.3</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>96.6</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>61</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Random Forest
#   Classifier (Random UnderSampler<span style='display:none'>)<span
#   style='mso-spacerun:yes'> </span></span></td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>98.32</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>15.4</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>87.8</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>75</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl66 style='height:14.4pt;border-top:none'>Random Forest
#   Classifier (SMOTE)</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>99.95</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>84.9</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>79.7</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>84</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Random Forest
#   Classifier (ADASYN)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.94</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>83.1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>76.4</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>83</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl66 style='height:14.4pt;border-top:none'>XGBoost
#   Classifier (No weights adjusted)</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>99.95</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>84.1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>75</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>84.6</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl66 style='height:14.4pt;border-top:none'>XGBoost
#   Classifier (Weights Adjusted)</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>99.95</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>84.1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>75</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>84.6</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>XGBoost
#   Classifier (Near Miss)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>10.46</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>0.4</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>95.9</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>13.2</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>XGBoost
#   Classifier (Random Under Sampler)</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>97.74</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>12</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>89.2</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>71.1</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl66 style='height:14.4pt;border-top:none'>XGBoost
#   Classifier (SMOTE)</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>99.93</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>81.4</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>81.1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>83.5</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl66 style='height:14.4pt;border-top:none'>XGBoost
#   Classifier (ADASYN)</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>1</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>99.92</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>78.6</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>81.8</td>
#   <td class=xl66 align=right style='border-top:none;border-left:none'>82.8</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#  </tr>
#  <tr height=19 style='height:14.4pt'>
#   <td height=19 class=xl65 style='height:14.4pt;border-top:none'>Deep Learning
#   Model (Keras )</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.96</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>99.94</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>81.7</td>
#   <td class=xl65 align=right style='border-top:none;border-left:none'>74.3</td>
#   <td class=xl65 style='border-top:none;border-left:none'>&nbsp;</td>
#  </tr>
#  <tr height=0 style='display:none'>
#   <td width=290 style='width:218pt'></td>
#   <td width=154 style='width:116pt'></td>
#   <td width=169 style='width:127pt'></td>
#   <td width=118 style='width:88pt'></td>
#   <td width=86 style='width:64pt'></td>
#   <td width=81 style='width:61pt'></td>
#  </tr>
# </table>
# 

# In[ ]:




