#!/usr/bin/env python
# coding: utf-8

# ## Packages used in the Project

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For Plotting and Visualization
import seaborn as sns # For Plotting and Visualization
from sklearn.model_selection import train_test_split # For Splitting the Dataset into Training and Test Dataset
import statsmodels.api as sm # For Building Linear Model
from statsmodels.stats.outliers_influence import variance_inflation_factor # To Calculate Variance Inflation Factor Between Indepenent Variables
import os
print(os.listdir("../input"))
    


# ## Classes and Functions

#  Created a Class which contains reusable functions used for retrieving statistical information from the Linear Regession Model

# In[ ]:


class PredictGreScore():
    
    def __init__(self,x,y,model):
        self.predictordata=x
        self.actualresponsevalue=y
        self.model=model
        # Calculate Degree of Freedom of Predictor Variable Variance
        self.df_pred=x.shape[0]-1
        # Calculate Degree of Freedom of Population Error Variance
        self.df_error=x.shape[0]-x.shape[1]-1
    
    # Calculate Total Sum of Squares
    
    def TSS(self):
        avg_y=np.mean(self.actualresponsevalue)
        squared_errors=((self.actualresponsevalue)-(avg_y))**2
        return np.sum(squared_errors)
    
    # Calculate Residual Sum of Squares
    
    def RSS(self):
        ActualValue=self.actualresponsevalue
        PredictedValue=self.model.predict(self.predictordata)
        ResidualError = (ActualValue-PredictedValue)**2
        return np.sum(ResidualError)
    
    # Calculate R-Squared Value
    
    def r_squared(self):
        return 1 - self.RSS()/self.TSS()
    
    # Calculate Adjusted R-Squared Value
    
    def adj_rsquared(self):
        return 1-(self.RSS()/self.df_error)/(self.TSS()/self.df_pred)
    
    # Plot Residual Analysis of Error Data
    
    def plot_residualanalysis(self):
        fig=plt.figure()
        sns.distplot(self.actualresponsevalue-self.model.predict(self.predictordata))
        plt.xlabel('Errors',fontsize=18)
        
    


# **Function to Print the Statistical Results**

# In[ ]:


def print_statsresults(stats_obj):
    items=(('Residual Sum of Squares:',stats_obj.RSS()),('Total Sum of Squares:',stats_obj.TSS()),('R-Squared:',stats_obj.r_squared()),('Adjusted R-Squared:',stats_obj.adj_rsquared()))
    for item in items:
        print('{0:8}{1:.4f}'.format(item[0],item[1]))


# In[ ]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read the input dataset
input=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
input.head()


# In[ ]:


# Rows and Columns in the dataset
input.shape


# In[ ]:


# Datatypes of each Column in the Dataset
input.info()


# In[ ]:


# Understand the Correlation between each Column in the dataset

sns.set(style='ticks',color_codes=True)
sns.pairplot(input)
plt.show()


# `GRE Score` is said to be highly correlated with `TOEFL Score`, `CGPA` and `Chance of Admit`

# In[ ]:


# check for missing values in the dataset

input.isnull().sum()


# - There is no missing values in any of the Columns in the Dataset.
# - we have necessary data to build the Linear Regression Model
# 

# ## Divide the Dataset into Training and Test Data

# In[ ]:


# Dataset is divided into Training and Testing Data using train_test_split imported from sklearn.model_selection

np.random.seed(0)

# split the dataframe

input_train,input_test=train_test_split(input,train_size=0.6,test_size=0.4,random_state=None)


# In[ ]:


# Perform Correlation on the Training Data to identify the Predictor Variables Highly Correlated with the Response Variable

plt.figure(figsize = (25, 10))
sns.heatmap(input_train.corr(),annot=True,cmap='YlGnBu')
plt.show()


# `GRE Score` is said to be highly correlated with `TOEFL Score` ,`CGPA`and `Chance of Admit`

# -  We have successfully split the dataset into training and testing datasets. 
# -  Identified the highly Correlated Predictor Variables with Response Variable
# - Next Step is to split the training dataset into Predictor(X) and Response (Y) models

# ## Dividing the Training Dataset into X and Y Models

# In[ ]:


y_train=input_train.pop('GRE Score')
x_train=input_train


# ## Building the Linear Model

# Let us fit a regression line for training data using statsmodels. when statsmodels library is used for building the model
# we need to explicitly define the constant.

# - To build the best fit **Multiple LinearRegression Model** ,**Feature Selection** is one of the most crucial aspects .
# -  I will be using  **Manual Feature Elimination**  Procedure as the number of Predictor Variables is very less.
# -  As part of **Manual Feature Elimination** ,we build multiple models and choose the best fit model whose Residual Sum of Squares       is minimal and high Adjusted R-Squared Value.

# ## Multiple Linear Regression - Model 1

# **Considered Predictor Variables which are highly correlated**

# In[ ]:


# Adding the Constant

x_train_lm=sm.add_constant(x_train[['TOEFL Score','CGPA','Chance of Admit ']])


# In[ ]:


# Create Linear Regression Model

lr=sm.OLS(y_train,x_train_lm).fit()


# In[ ]:


x_train_lm.head()


# In[ ]:


print(lr.summary())


# - From the warnings in the Model, it has been observed there is multicollinearity existing between the predictor variables.
# - Let us calculate the Variance Inflation Factor to identify how much strong each predictor variables are correlated against each other

# **Variance Inflation Factor**

# In[ ]:


# Creating a dataframe which contains the list of Predictor Variables and their VIF's

vif=pd.DataFrame()
vif['Features']=x_train_lm.columns
vif['vif']=[variance_inflation_factor(x_train_lm.values,i) for i in range(x_train_lm.shape[1])]
vif['vif']=round(vif['vif'],2)
vif=vif.sort_values(by="vif",ascending=False)
vif


# 1. VIF values obtained for the predicted variables (CGPA, Chance of Admit ,TOEFL Score) is within the level of acceptance.
# 2. VIF Under Ideal Conditions should be less than 3 but it is still acceptable if it under 10
# 3. So this concludes i have selected decent predictor variables for the Model

# **Residual Analysis**

# In[ ]:


ResidualAnalysis=PredictGreScore(x_train_lm,y_train,lr)
ResidualAnalysis.plot_residualanalysis()


# Residual Analysis Plot clearly shows the Error Terms are Normally Dsitributed from which we can confirm we built a model which produced the best estimates

# ** Display the Statistical Results of Model**

# In[ ]:


stats=PredictGreScore(x_train_lm,y_train,lr)
print_statsresults(stats)


# - We have built a model with `Adjusted R-Squared Value` of `77%` and `Residual Sum of Squares value` said to be `8756.0620`
# - Let us Compare this Model with other Linear Regression Models and then decide which Model fits the best fil regression line.

# ## Multiple Linear Regression - Model 2

# - Let say we consider the VIF Value to be less than or almost equal to 3 is ideal
# - Let us remove CGPA which has High VIF when compared to other Predictor Variables and observe how the model behaves

# In[ ]:


# Building a new model by Considering TOEFL Score and Chance of Admit as Predictor Variables and removed CGPA which has high VIF 
x_train_model2=sm.add_constant(x_train[['TOEFL Score','Chance of Admit ']])


# In[ ]:


# Run the Model
lr_model2=sm.OLS(y_train,x_train_model2).fit()


# In[ ]:


print(lr_model2.summary())


# In[ ]:


# Plotting Residual Analysis to see if error terms are normally dsitributed
ResidualAnalysis=PredictGreScore(x_train_model2,y_train,lr_model2)
ResidualAnalysis.plot_residualanalysis()


# ** Display the Statistical Results of the Model**

# In[ ]:


stats=PredictGreScore(x_train_model2,y_train,lr_model2)
print_statsresults(stats)


# - When Compared to the results of `Model1` ,`Model2` has high Residual Sum of Squares and Adjsuted R - Squared Value is deceased when Compared to Model1
# - After Comaprison, We can say Model 1 is better when Compared to Model 2 as for a Model to be significant and to best fit the Regression Line its Residual Sum of Squares should be minimal and Adjusted R-Square should be moderate.

# * From the Correlation Plot, we can notice University Rating has good Correlation with GRE Score.Let us build a Model which contains University Rating also as one of the Predictor Varibles.

# ## Multiple Linear Regression - Model 3

# In[ ]:


x_train_model3=sm.add_constant(x_train[['TOEFL Score','CGPA','University Rating','Chance of Admit ']])


# In[ ]:


lr_model3=sm.OLS(y_train,x_train_model3).fit()


# In[ ]:


print(lr_model3.summary())


# In[ ]:


# Plotting Residual Analysis to see if error terms are normally dsitributed
ResidualAnalysis=PredictGreScore(x_train_model3,y_train,lr_model3)
ResidualAnalysis.plot_residualanalysis()


# **Display the Statistical Results of the Model**

# In[ ]:


stats=PredictGreScore(x_train_model3,y_train,lr_model3)
print_statsresults(stats)


# - `Model 3` has Residual Sum of Squres equal to `Model 1` and `Adjsuted R-Squared` is more or less equal to `Model 1`
# - But the `University Rating` Predictor has `high p-value(>0.05)` which proves there is not enough evidence to reject the null hypothesis and also to conclde that non-zero correlation exists.
# - So `Univesrity Rating` Predictor is said to be statistically `insignificant` and thus we can avoid Model 3

# **Model Selection**

# After Comparison of all the models, we choose `Model 1` as the model which is able to produce the best estimates and will be making prediction of the test data using `Model 1`

# ## Model Prediction

# In[ ]:


# Let us split the test data into x_test and y_test
y_test=input_test.pop('GRE Score')
x_test=input_test


# In[ ]:


# Now lets use Model 1 to make Predictions and select the Predictor Variables used in Model 1

# Drop the Constant Variable Column from the train columns used in the Model 1
x_train_new=x_train_lm.drop(['const'],axis=1)

x_test_new=x_test[x_train_new.columns]

# Adding a Constant Variable
x_test_new=sm.add_constant(x_test_new)

#Making the Predictions

y_pred=lr.predict(x_test_new)


# **Display the Statistical Results of Predicted Values**

# In[ ]:


stats=PredictGreScore(x_test_new,y_test,lr)
print_statsresults(stats)


# Acheived  Residual Sum of Squares Value of `6257.6383` and  `72%` Adjusted R-Squared

# ## Model Evalulation

# In[ ]:


# Plotting the y_test and y_pred to Understand the Spread

fig=plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred',fontsize=20)
plt.xlabel('y_test',fontsize=18)
plt.ylabel('y_pred',fontsize=16)


# From the plot it is evident that there is a Linear Relationship established between the Actual Response Variable and Predicted Response Variable

# ## Equation for Best Fit Regression Line

#  GRE Score = 175.4594 + 0.712 * `TOEFL Score` + 5.9518*`CGPA` + 18.4741*`Chance of Admit`
# 
# 
