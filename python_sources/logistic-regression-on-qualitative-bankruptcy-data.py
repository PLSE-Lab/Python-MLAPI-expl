#!/usr/bin/env python
# coding: utf-8

# **Qualitative_Bankruptcy Data Set**
# 
# Abstract: Predict the Bankruptcy from Qualitative parameters from experts. The data is taken from UCI Machine Learning Repository.
# 
# Attribute Information: (P=Positive,A-Average,N-negative,B-Bankruptcy,NB-Non-Bankruptcy)
# 
# 1.Industrial Risk: {P,A,N}
# 
# 2.Management Risk: {P,A,N}
# 
# 3.Financial Flexibility: {P,A,N}
# 
# 4.Credibility: {P,A,N}
# 
# 5.Competitiveness: {P,A,N}
# 
# 6.Operating Risk: {P,A,N}
# 
# 7.Class: {B,NB}
# 

# **Importing Libraries**

# In[ ]:


from sklearn import linear_model as lm
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as sts
import pandas as pd
import scipy as scp
import numpy as np
import sklearn.preprocessing as preproc
from sklearn.model_selection import train_test_split  ### for train and test split package
from sklearn import metrics  ## For calculation of MSE & RMSE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score


# **Reading the csv data through Pandas DataFrame**

# In[ ]:


bank = pd.read_csv("../input/QualitativeBankruptcy.csv")


# **Exploratory data analysis (EDA)**

# In[ ]:


bank.head()


# In[ ]:


bank.shape   ## It display the no of rows and column


# **The Dataset has 250 rows and 7 columns**

# In[ ]:


## It gives the unique label of each column

for col in bank:
    print (col)
    print (bank[col].unique())


# In[ ]:


bank.describe()  ##It gives the summary of the DataFrame


# **The columns are in categorical text form. For our modelling purpose we have to convert it into numerical form, so we have to apply Label Encoding. For that we have to assign all the columns name to a variable to perform Label Encoding in loop.**

# In[ ]:


var = ['indRisk','mgtRisk','finFlexibility','Credibility','Competitiveness', 'OperatingRisk', 'bclass']
var


# In[ ]:


def func_labelEncoder(var,features):
    encode= LabelEncoder()
    features[var] = encode.fit_transform(features[var].astype(str))
    
for i in var:
    func_labelEncoder(i,bank)


# In[ ]:


bank.head()


# **Through Label Encoder we have transformed all the cloumns into Numerical value so as it will best suit for our modelling purpose.**

# **Doing EDA on transformed data**

# In[ ]:


bank.describe()


# **We got all the details of the columns value. Their mean,standard deviation, count,min , max etc.**

# **We have to check if any null value exist for any columns or not.**

# In[ ]:


bank.isnull().any()


# In[ ]:


sns.pairplot(bank)


# **Calculating the Correlation for each columns**

# In[ ]:


bank.corr()


# **In the above we can see the correlation between different variables are between -0.5 to 0.5, so we can take all the variables for developing prediction model.**

# **Storing the dependent variables to xVal and independent variable to yVal**

# In[ ]:


xVal = bank.drop(['bclass'], axis=1)


# In[ ]:


xVal.head()


# In[ ]:


yVal = bank.bclass.values.reshape(-1,1)


# In[ ]:


yVal.shape


# **We will split the whole dataset into test and train row.The train data will be 80% of the total dataset and test data will be 20% of the dataset.**

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(xVal,yVal, test_size=0.2, random_state=42)


# In[ ]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", Y_test.shape)


# **Applying Logistic Regression Model**

# In[ ]:


### 1. Logistic Regression Model

lmod = lm.LogisticRegression(penalty='l2',fit_intercept=True,max_iter=500,solver='lbfgs',tol=0.0001,multi_class='ovr')


# **Training the Logistic Model**

# In[ ]:


lrmod = lmod.fit(X_train,Y_train.ravel())


# In[ ]:


lrmod.intercept_  ### Intercapt (B0)


# In[ ]:


lrmod.coef_   ### Coefficients (B1, B2...)


# **Predicting the test data**

# In[ ]:


predicted_data = lrmod.predict(X_test)  ### Predicting the  model for independent test data


# In[ ]:


predicted_data


# **Confusion Matrix**

# In[ ]:


confusion_matrix(Y_test, predicted_data)


# In[ ]:


from sklearn import metrics as accuracyMatrics


# In[ ]:


accuracyMatrics.accuracy_score(Y_test, predicted_data)  ## Predicting accuracy score


# **Here we are getting accuracy of 62%. As we have less number of rows in this dataset thats why we are getting less accuracy.**

# **Precision**

# In[ ]:


prec = accuracyMatrics.precision_score(Y_test, predicted_data)  ## Precision score
prec


# **Recall**

# In[ ]:


recall = accuracyMatrics.recall_score(Y_test, predicted_data)  ## Recall score
recall


# In[ ]:


probPred = lrmod.predict_proba(X_test)
predictProbAdmit = probPred[:,1]


# **Calculating values for ROC Curve**

# In[ ]:


### ROC curve calculation

fpr, tpr, threshold = accuracyMatrics.roc_curve(Y_test,predictProbAdmit)


# **AUC Value**

# In[ ]:


auc_val = accuracyMatrics.auc(fpr,tpr)
auc_val   ### AUC Value


# In[ ]:


threshold


# **ROC Curve**

# In[ ]:


## Plotting ROC Curve

plt.plot(fpr,tpr,linewidth=2, color='g',label='ROC curve (area = %0.2f)' % auc_val)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')


# **F1 Score**

# In[ ]:


## F1 score 

F1_score = f1_score(Y_test, predicted_data)
F1_score


# **This is a simple representation of basic Logistic Regression with all the evaluation metrics. When we have complex dataset then we get more oppurtunity to clean and treat missing values and outliers.**
# 
# **Hope you like this Material**

# In[ ]:





# In[ ]:




