#!/usr/bin/env python
# coding: utf-8

# Task: Create a Machine Learning model that approves or decline a loan application based on the information provided.
# 
# Problem Type: Binary Classification.
# 
# Published by Adedayo Okubanjo.

# **IMPORT LIBRARIES AND LOAD DATA**

# In[ ]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
#import data
data = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
#preview top five rows in the dataset
data.head()


# From previewing the data, we can already see that we need to convert some of the independent variables to integers: Gender, Married, Education, Self_Employed and Property_Area to integers.

# In[ ]:


#check the number of row and columns in the imported data
data.shape   #614 rows and 13 columns


# In[ ]:


#check the description of the data such as mean,count, min, max for each numeric cell
data.describe()


# We can already see that Credit_History is between 0 and 1 from the min and max rows and with 75% of the records having value as 1.
# 
# Credit_History column also has a mean of 0.842199.

# In[ ]:


data["Credit_History"].median() #median value of 1


# In[ ]:


#check if Credit_History is in % or 0 and 1 values
data.groupby(["Credit_History"])["Loan_ID"].count() 


# This shows that Credit_History is a categorical data with 0 and 1 values (Good or Bad)

# Looking deeper into the credit history column shows

# In[ ]:


#check the number of null records per column
data.isnull().sum()


# In[ ]:


#remove the Loan_ID column as it's not needed as an independent variable for the model. 
#it's just an ID to uniquely identify each row and does not describe the data in anyway.
data = data.drop(columns = ("Loan_ID") )
data.head()


# **HANDLING NULL VALUES**

# In[ ]:


#Drop null values in all the categorical columns
data = data.dropna(subset = ["Gender","Married","Dependents","Education","Self_Employed","Credit_History","Property_Area"])
data.isnull().sum()


# In[ ]:


#Fill null values in columns with continous data with the mean of each column
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].mean())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mean())
data.isnull().sum()


# **EXPLORATORY ANALYSIS**

# In[ ]:


#Loan Status by Marital Status
sb.countplot(x=data["Married"], hue = data["Loan_Status"], data = data)


# In[ ]:


#LoanAmount distribution
data["LoanAmount"].plot.hist(data["LoanAmount"])


# There are outlier in the loan amount but most of the loans are around the 100 to < 200 range.

# In[ ]:


fig = data.groupby("Dependents")["Loan_Status"].count().plot.bar(color = "red")
fig.set_ylabel('Count')
fig.set_title("Count of loans requested by No of Dependents")



# In[ ]:


#Check for correlation between all numeric columns
plt.title('Correlation Matrix')
sb.heatmap(data.corr(),annot=True)


# It can be deduced from the chart above that there's a correlation between the Loan Amount and the income of both the Applicant and Coapplicant.

# In[ ]:


#this scatter plot also shows the correlation above.
data.plot.scatter("ApplicantIncome", "LoanAmount", color = "blue")


# In[ ]:


data.groupby(["Gender"])["Loan_Status"].count() #Female and Male


# In[ ]:


data.groupby(["Married"])["Loan_Status"].count() #Yes and No


# In[ ]:


data.groupby(["Dependents"])["Loan_Status"].count() #0, 1, 2, 3+ and Male


# In[ ]:


data.groupby(["Education"])["Loan_Status"].count() #Graduate and Non Graduate


# In[ ]:


data.groupby(["Self_Employed"])["Loan_Status"].count() #Yes or No


# In[ ]:


data.groupby(["Loan_Status"])["Loan_Status"].count() #Y or N


# In[ ]:


data.groupby(["Property_Area"])["Loan_Status"].count() #Rural, Semiurban or Urban


# Replace all non numeric category data to integers e.g. Yes and No to 1 and 0 for better model performance

# In[ ]:


data["Gender"] = data["Gender"].replace(["Female","Male"], [0, 1])
data["Married"] = data["Married"].replace(["No","Yes"], [0, 1])
data["Dependents"] = data["Dependents"].replace(["0","1","2","3+"], [0, 1,2,3])
data["Education"] = data["Education"].replace(["Not Graduate","Graduate"], [0, 1])
data["Self_Employed"] = data["Self_Employed"].replace(["No","Yes"], [0, 1])
data["Loan_Status"] = data["Loan_Status"].replace(["N","Y"], [0, 1])
data["Property_Area"] = data["Property_Area"].replace(["Rural","Semiurban", "Urban"], [0, 1, 2])
data.head()


# In[ ]:


#check data types of all the columns to make sure they are all numeric (float or integer)
data.dtypes


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))  
sb.heatmap(data.corr(),annot = True, ax=ax)


# #From the correlation matrix, we can see that Credit_History has the highest correlation with the Loan_Status which makes sense as your credit history should determine to a large extent if you can access a loan or not.

# **LOGISTIC REGRESSION**

# In[ ]:


#Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

#Define x and y (independent variable(s) and dependent variable respectively)
y = pd.DataFrame(data.iloc[:,11:])
x = pd.DataFrame(data.iloc[:,0:11])

#Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30, random_state = 1)

#Initialize and fit Logistic Regression Model
classifier = LogisticRegression(max_iter = 10000)
classifier.fit(x_train,y_train.values.ravel())


# In[ ]:


#Pass the test data (x_test) to the model to predict y
y_pred = classifier.predict(x_test)
y_pred


# In[ ]:


#score your model
classifier.score(x_train,y_train)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


#check the share of total predictions that were accurate (possitive and negative)
accuracy_score(y_test,y_pred)


# In[ ]:


#check the share of total positive prediction that were accurate
precision_score(y_test,y_pred)


# In[ ]:


#checks the share of all the positives in the test data that the model was able to accurately predict
recall_score(y_test,y_pred)


# In[ ]:


#weighted average of the precision score and recall score
f1_score(y_test,y_pred)


# Logistics Regression performed well (Accuracy of 81%)
# 
# * Accuracy Score = 81%
# * Recall Score = 99%
# * Precision Score = 77%
# * F1_Score = 87%

# **DECISION TREE**

# In[ ]:


#Initialize and fit Decision Tree Model
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train,y_train)


# In[ ]:


#Pass the test data (x_test) to the model to predict y
dt_y_pred = dt_classifier.predict(x_test)
dt_y_pred


# In[ ]:


#score your model
dt_classifier.score(x_train,y_train)


# In[ ]:


confusion_matrix(y_test,dt_y_pred)


# In[ ]:


#check the share of total predictions that were accurate (both positive and negative)
accuracy_score(y_test,dt_y_pred)


# In[ ]:


#check the share of total positive prediction that were accurate
precision_score(y_test,dt_y_pred)


# In[ ]:


#checks the share of all the positives in the test data that the model was able to accurately predict
recall_score(y_test,dt_y_pred)


# In[ ]:


#weighted average of the precision score and recall score
f1_score(y_test,dt_y_pred)


# Logistics Regression
# 
# * Accuracy Score = 81%
# * Recall Score = 99%
# * Precision Score = 77%
# * F1_Score = 87%
# 
# Decision Tree
# * Accuracy Score = 73%
# * Recall Score = 80%
# * Precision Score = 78%
# * F1_Score = 79%

# **RANDOM FOREST**

# In[ ]:


#Initialize and fit Random Forest Model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train,y_train.values.ravel())


# In[ ]:


#Pass the test data (x_test) to the model to predict y
rf_y_pred = rf_classifier.predict(x_test)
rf_y_pred


# In[ ]:


#score your model
rf_classifier.score(x_train,y_train)


# In[ ]:


confusion_matrix(y_test,rf_y_pred)


# In[ ]:


#check the share of total predictions that were accurate (both positive and negative)
accuracy_score(y_test,rf_y_pred)


# In[ ]:


#check the share of total positive prediction that were accurate
precision_score(y_test,rf_y_pred)


# In[ ]:


#checks the share of all the positives in the test data that the model was able to accurately predict
recall_score(y_test,rf_y_pred)


# In[ ]:


#weighted average of the precision score and recall score
f1_score(y_test,rf_y_pred)


# **Logistic Regression has the best accuracy score followed by Random Forest then Decision Tree.**
# 
# 1) Logistics Regression
# * Accuracy Score = 81%
# * Recall Score = 99%
# * Precision Score = 77%
# * F1_Score = 87%
# 
# 2) Decision Tree
# * Accuracy Score = 73%
# * Recall Score = 80%
# * Precision Score = 78%
# * F1_Score = 79%
# 
# 3) Random Forest
# * Accuracy Score = 78%
# * Recall Score = 94%
# * Precision Score = 77%
# * F1_Score = 85%
