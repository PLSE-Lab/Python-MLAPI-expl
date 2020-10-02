#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# The following notebook deals with the Credit Card Fraud Detection problem. A cursory glance on the dataset reveals that the data is highly imbalanced and the accuracy score would not suffice as a performance metric. Here's a ouline of what has been done in this notebook.

# ![fraud1.jpg](attachment:fraud1.jpg)

# # Outline of the notebook
# 
# 1. Import the required libraries 
# 2. Import the data into a dataframe and preview the target column
# 3. Remove the time column as time reveals the time after the first transaction and logically should not impact the target column
# 4. Create an empty dataframe for storing the performance metrics for multiple algorithms
# 5. Split the data into training set and testing set
# 6. Deal with problem of class imbalance with the following methods:
#    a) By doing nothing and training the algorithm with the data as it is
#    b) By oversampling the minority class in the training dataset
#    c) By undersampling the majority class in the training dataset
#    d) By applying SMOTE(Synthetic Minority Oversampling TEchnique) on the training dataset
# 7. The accuracy, precision, recall and f1 score were compared for all the above methods for 2 algorithms:
#    a) Logistic Regression
#    b) Decision Tree

# *1. Import the required libraries*

# In[ ]:


import numpy as np
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix, f1_score


# *2. Import the data into a dataframe and preview the target column*

# In[ ]:


df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
sns.countplot(df["Class"]);


# In[ ]:


#Check the count of the 2 classes
df["Class"].value_counts()


# # **Noteworthy points**
# 
# 1. The data has original columns hidden due to privacy issues and columns available with us are the **28 principal components** of the original columns. As we know that PCA requires data to be scaled before usage hence we can safely *assume* that the 28 columns are already scaled.
# 2. Amount column is not scaled and hence we will try to scale it.
# 3. Train test split must be done **before** applying any sort of resampling techniques. The reason is as follows:
# 
# Let's suppose that you perform oversampling of the minority class before the train test split. What oversampling will do is, it will replicate the minority class records and in our case the 492 records will be replicated to ensure that their count matches that of the majority class i.e. 284315.
# 
# > You see the problem here?
# 
# After oversampling when you perform the train test split, the model will have close to 90% (if not 100%) overlapping of minority class records in the training dataset and testing dataset. Thus, the model already has learnt how to handle the  testing dataset records (from the training dataset) and will pass all the evaluation metrics with flying colours. 
# 
# > How do we handle this situation?
# 
# Perform the train test split first and then apply any resampling techniques. This will ensure that that the minority class records in the training dataset are replicated to form a balanced training dataset. The model will learn to differentiate between the fraud and non-fraud cases using the balanced training dataset and its performance can be efficiently and unbiasedly be evaluated using the unseen testing dataset.

# *3. Remove the time column as time reveals the time after the first transaction and logically should not impact the target column*

# In[ ]:


#scaling the amount column using a standard scaler
scaler=StandardScaler()
df["Amount"]=scaler.fit_transform(np.array(df["Amount"]).reshape(-1,1))

#Dropping the time column
df.drop(columns = ["Time"], inplace = True)


# In[ ]:


#Checking the Amount column after scaling
df["Amount"].head()


# *4. Create an empty dataframe for storing the performance metrics for multiple algorithms*

# Here the empty data frame contains:
#     1.  8 rows corresponding to the class imbalance handling techniques
#     2.  6 columns corresponding to algorithm name, method and the 4 performance metrics - Accuracy, Precision, Recall and F1_Score.

# In[ ]:


#Create an empty dataframe for storing the performance metrics for multiple algorithms
dfEval = pd.DataFrame({"Algorithm": ["Logistic Regression","Logistic Regression","Logistic Regression","Logistic Regression",
                                    "Decision Tree","Decision Tree","Decision Tree","Decision Tree"],
                       "Method" : [ "Unbalanced", "Oversample", "Undersample", "SMOTE",
                                    "Unbalanced", "Oversample", "Undersample", "SMOTE"],
                       "Accuracy" : [0,0,0,0,0,0,0,0],"Precision" : [0,0,0,0,0,0,0,0],"Recall" : [0,0,0,0,0,0,0,0],
                       "F1_Score" : [0,0,0,0,0,0,0,0]})


# In[ ]:


#Checking the evaluation metric dataframe before inserting records in it
dfEval


# *5. Split the data into training set and testing set*
# 
# *6. Deal with problem of class imbalance with the following methods: *

#     1. By doing nothing and training the algorithm with the data as it is 
#     2. By oversampling the minority class in the training dataset 
#     3. By undersampling the majority class in the training dataset 
#     4. By applying SMOTE(Synthetic Minority Oversampling TEchnique) on the training dataset

# In[ ]:


#Creating a loop that runs 4 times. In each run of the loop, logistic regression as well as decision tree is applied.
#Run 1 : Original sample without any resampling
#Run 2 : Minority sample is oversampled
#Run 3 : Majority sample is undersampled
#Run 4 : SMOTE(Synthetic Minority Oversampling Technique) is applied. SMOTE uses KNN to generate similar minority samples

#Loop variables for filling the performance metric dataframe
cnt = 0
#Logistic Regression Count variable
LR_cnt = 0 
#Decision Tree Count variable
DT_cnt = 4

for i in range(4):
    
    #Creating X and y for storing the regressor and the target variable
    y = df["Class"].copy()
    X = df.drop(columns=["Class"]).copy()
    
    #Splitting the data into training set and testing set before applying any imbalance handling measures
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42)
    
    if i == 0:
        #DO not perform any imbalance handling measures
        pass
    
    elif (i==1) or (i==2):
        
        #Concatenate the regressors and the target variable of the training set and get the fraud and not_fraud records
        dfTemp = pd.concat([X_train , y_train], axis = 1)
        fraud = dfTemp[dfTemp["Class"] == 1]
        not_fraud = dfTemp[dfTemp["Class"] == 0]
        
        if i==1:
            #Oversample minority class to make the number of fraud samples = no of non fraud samples
            oversampled_fraud = resample(fraud, replace = True, n_samples = len(not_fraud), random_state = 42)
            data1 = not_fraud
            data2 = oversampled_fraud
            
        else:
            #Undersample majority class to make the number of non fraud samples = no of fraud samples
            undersampled_not_fraud = resample(not_fraud, replace = False, n_samples = len(fraud), random_state = 42)
            data1 = fraud
            data2 = undersampled_not_fraud
            
        #Concatenate the balanced data of fraud and non-fraud cases to get a new dataframe
        dfNew = pd.concat([data1, data2], axis =0)
        
        #Split the balanced data into X_train and y_train datasets
        y_train = dfNew["Class"].copy()
        X_train = dfNew.drop(columns = ["Class"]).copy()
        
        
    else:
        #SMOTE method
        smote = SMOTE(random_state=42, sampling_strategy=1.0, n_jobs=-1)
        
        #Create synthetic samples from the training dataset to balance the fraud and non-fraud records by oversampling minority class
        X_train, y_train = smote.fit_sample(X_train, y_train)
        
    
    #Apply logistic regression and Decision Tree classification for the above 4 cases
    for j in range(2):
        if j == 0:
            
            #Model for logistic regression
            model = LogisticRegression()
            
            #Incrementing the loop counters
            cnt = LR_cnt
            LR_cnt = LR_cnt + 1
        else:
            #Model for Decision Tree classification
            model = DecisionTreeClassifier()
            
            #Incrementing the loop counters
            cnt = DT_cnt
            DT_cnt = DT_cnt + 1
        
        
        #Fitting the model on the training data
        model.fit(X_train,y_train)
        
        #Getting the predictions for the testing data
        y_pred = model.predict(X_test)
        
        #Filling the performance metrics dataframe with the model values for accuracy, precision, recall and F1 score
        dfEval.iloc[cnt,2] = round( 100 * model.score(X_test,y_test) , 2)
        dfEval.iloc[cnt,3] = round( 100 * precision_score(y_test, y_pred) , 2)
        dfEval.iloc[cnt,4] = round( 100 * recall_score(y_test, y_pred) , 2)
        dfEval.iloc[cnt,5] = round( 100 * f1_score(y_test, y_pred) , 2)
        
        


# *7. The accuracy, precision, recall and f1 score were compared for all the above methods for 2 algorithms: a) Logistic Regression b) Decision Tree*

# In[ ]:


#Checking the evaluation metric dataframe after the models have been trained and tested
dfEval


# In[ ]:


fig, axes = plt.subplots(nrows=1,ncols=2,figsize = (15,5))
dfTemp = dfEval[["Method", "Accuracy","Precision","Recall","F1_Score"]]
dfTemp[:4].plot(x="Method", kind = "line",legend = True, grid = True, ax = axes[0],
               title="Logistic Regression Scores")

dfTemp[4:].plot(x="Method", kind = "line",legend = True, grid = True, ax = axes[1],
               title="Decision Tree Scores");


# # Concluding Remarks
# 
# Checking the above plots, we come to know that for this particular problem, Undersampling seems to underperform as compared to other methods. Although, undersampling is giving quite good recall scores, it is doing so at the cost of the precision scores.
# Even, SMOTE gives average performance for the given data and I would place my bets on Oversampling technique in combination with a decision tree model to achieve decent values of precision, recall, accuracy and even F1 score.
# 
# Last but not the least, the above techniques are just the ones which I have implemented and there definitely would be room for improvement. 
# 
# Thanks for going through this kernel. Do let me know in the comments if you have any suggestions or doubts.

# In[ ]:




