#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv("../input/smarket.csv")
df = df.drop([df.columns[0]],axis=1)
df.head()


# **1. Do Quantile Analysis for each variable**

# **1.1 Analysis of Year**

# In[3]:


df.boxplot(column=["Year"])


# In[4]:


df["Year"].describe()


# In[ ]:


df["Year"].median()


# 
# 
# 
# 

# 1.1.1 Q: Qualify the spread. Is it evenly spread? 
# 

# Ans: Yes, because the mean and median are the same so this data set is symmetric.

# 

# 

# **1.2 Analysis of Lag1**

# In[5]:


df.boxplot(column=["Lag1"])


# In[6]:


df["Lag1"].describe()


# In[10]:


skew = (df["Lag1"].quantile(q=0.75) - df["Lag1"].quantile(q=0.5)) - (df["Lag1"].quantile(q=0.5) - df["Lag1"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.2.1 Q: Qualify the spread. Is it evenly spread? 
# 

# Ans: No, as the mean and medien are different

# 1.2.2 Q: Skewed to lower or upper end?
# 

# Ans: Skew to lower, because Q3-Q2 < Q2-Q1 

# 1.2.3 Q: Is the skew moderate or significant? 

# Ans: Moderate, because Q3-Q2 is not significantly different from Q2-Q1

# 1.2.4 Is it useful to do Quantile analysis on this variable?

# Ans: Yes, it helps to know the distribution of sampling data.

# **1.3 Analysis of Lag2**

# In[11]:


df.boxplot(column=["Lag2"])


# In[12]:


df["Lag2"].describe()


# In[14]:


skew = (df["Lag2"].quantile(q=0.75) - df["Lag2"].quantile(q=0.5)) - (df["Lag2"].quantile(q=0.5) - df["Lag2"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.3.1 Q: Qualify the spread. Is it evenly spread? 

# Ans: No, as the mean and medien are different

# 1.3.2 Q: Skewed to lower or upper end?
# 

# Ans: Skew to lower, because Q3-Q2 < Q2-Q1 

# 1.3.3 Q: Is the skew moderate or significant? 

# Ans: Moderate, because Q3-Q2 is not significantly different from Q2-Q1

# 1.3.4 Is it useful to do Quantile analysis on this variable?

# Ans: Yes, it helps to know the distribution of sampling data.

# **1.4 Analysis of Lag3**

# In[15]:


df.boxplot(column=["Lag3"])


# In[16]:


df["Lag3"].describe()


# In[18]:


skew = (df["Lag3"].quantile(q=0.75) - df["Lag3"].quantile(q=0.5)) - (df["Lag3"].quantile(q=0.5) - df["Lag3"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.4.1 Q: Qualify the spread. Is it evenly spread?
# 
# Ans: No, as the mean and medien are different
# 
# 1.4.2 Q: Skewed to lower or upper end?
# 
# Ans: Skew to lower, because Q3-Q2 < Q2-Q1
# 
# 1.4.3 Q: Is the skew moderate or significant?
# 
# Ans: Moderate to low, because Q3-Q2 is not significantly different from Q2-Q1
# 
# 1.4.4 Is it useful to do Quantile analysis on this variable?
# 
# Ans: Yes, it helps to know the distribution of sampling data.

# **1.5 Analysis of Lag4**

# In[ ]:


df.boxplot(column=["Lag4"])


# In[25]:


df.describe()


# In[26]:


skew = (df["Lag4"].quantile(q=0.75) - df["Lag4"].quantile(q=0.5)) - (df["Lag4"].quantile(q=0.5) - df["Lag4"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.5.1 Q: Qualify the spread. Is it evenly spread?
# 
# Ans: No, as the mean and medien are different
# 
# 1.5.2 Q: Skewed to lower or upper end?
# 
# Ans: Skew to lower, because Q3-Q2 < Q2-Q1
# 
# 1.5.3 Q: Is the skew moderate or significant?
# 
# Ans: Moderate to low, because Q3-Q2 is not significantly different from Q2-Q1
# 
# 1.5.4 Is it useful to do Quantile analysis on this variable?
# 
# Ans: Yes, it helps to know the distribution of sampling data.

# **1.6 Analysis of Lag5**

# In[ ]:


df.boxplot(column=["Lag5"])


# In[27]:


skew = (df["Lag5"].quantile(q=0.75) - df["Lag5"].quantile(q=0.5)) - (df["Lag5"].quantile(q=0.5) - df["Lag5"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.6.1 Q: Qualify the spread. Is it evenly spread?
# 
# Ans: No, as the mean and medien are different
# 
# 1.6.2 Q: Skewed to lower or upper end?
# 
# Ans: Skew to lower, because Q3-Q2 < Q2-Q1
# 
# 1.6.3 Q: Is the skew moderate or significant?
# 
# Ans: Moderate to low, because Q3-Q2 is not significantly different from Q2-Q1
# 
# 1.6.4 Is it useful to do Quantile analysis on this variable?
# 
# Ans: Yes, it helps to know the distribution of sampling data.

# **1.7 Analysis of Volume**

# In[ ]:


df.boxplot(column=["Volume"])


# In[28]:


skew = (df["Volume"].quantile(q=0.75) - df["Volume"].quantile(q=0.5)) - (df["Volume"].quantile(q=0.5) - df["Volume"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.7.1 Q: Qualify the spread. Is it evenly spread?
# 
# Ans: No, as the mean and medien are different
# 
# 1.7.2 Q: Skewed to lower or upper end?
# 
# Ans: Skew to upper, because Q3-Q2 > Q2-Q1
# 
# 1.7.3 Q: Is the skew moderate or significant?
# 
# Ans: Moderate to low, because Q3-Q2 is not significantly different from Q2-Q1
# 
# 1.7.4 Is it useful to do Quantile analysis on this variable?
# 
# Ans: Yes, it helps to know the distribution of sampling data.

# **1.8 Analysis of Today**

# In[ ]:


df.boxplot(column=["Today"])


# In[29]:


skew = (df["Volume"].quantile(q=0.75) - df["Volume"].quantile(q=0.5)) - (df["Volume"].quantile(q=0.5) - df["Volume"].quantile(q=0.25))
print("skew: "+ str(skew))


# 1.8.1 Q: Qualify the spread. Is it evenly spread?
# 
# Ans: No, as the mean and medien are different
# 
# 1.8.2 Q: Skewed to lower or upper end?
# 
# Ans: Skew to upper, because Q3-Q2 > Q2-Q1
# 
# 1.8.3 Q: Is the skew moderate or significant?
# 
# Ans: Moderate to low, because Q3-Q2 is not significantly different from Q2-Q1
# 
# 1.8.4 Is it useful to do Quantile analysis on this variable?
# 
# Ans: Yes, it helps to know the distribution of sampling data.

# **1.9 Analysis of Direction**

# 1.9.1 Q: Is it useful to do Quantileanalysis on this variable?
# 
# Ans: No, this data set is categorial

# **2. Do Pairwise Correlations for each pair of variables**

# In[ ]:


sns.pairplot(df)


# From the graph above, There might be some correlations between volumes and each Lag variables including today. This is interparated from huge group lines of which solpe are negative. However, the strengths of those correlations are moderate to low as those are not strong narrow lines.  

# For other pair variables like Lag1 and Today, there are no correlation or are very weak correlations between them as the graphs show circle groups at the center.

# **3. Pairwise Correlation Coefficients**

# In[ ]:


df.corr()


# 3.1 Q:Is there a correlation?
# 
# Ans: According to the correlation coefficient, there are some correlations existing in this dataset as those values are not zero. Those are listed below.
#     1. Volume and Year
#     2. Volume and each Lag variables
#     3. Volume and Today
#     4. Between each Lag variables
#     5. Each Lag variables and today
#     
# 3.2 Q: Is it positive or negative?
#     1. Volume and Year: Positive
#     2. Volume and each Lag variables: Volume and Lag1 is positive. Others are negative.
#     3. Volume and Today: Positive
#     4. Between each Lag variables: Neative
#     5. Each Lag variables and today: Negative
#     
# 3.3 Q: Is it low, moderate or high?
#     1. Volume and Year: Moderate
#     2. Volume and each Lag variables: Low
#     3. Volume and Today: Low
#     4. Between each Lag variables Low
#     5. Each Lag variables and today: Low
#     
# 3.4 Q: Is this correlation useful?
# 
# Ans: Yes, it helps to recognize which variables should be picked to form a model as they have high effects on the predicted value.

# 3.2 What interesting correlation do you see between Volume and the lags and also between Today and the lags? (Highlighted in yellow)

# Ans: For the Volume and Lags, there are both positive and negative correlations. The correlation of Volume and Lags1 is positive but the correlations of Volume and other Lags are negative. This may refer to the fact that although the volume is high, it doesn't guarantee that you will get high returns always.

# 3.3 Do you think Lags can reasonably predict the Volume or Today?

# Ans: Although the correlations between them are not zero, but the correlations are so weak. They might can be used to predict the value but not only one Lag. There should be combination of many Lags used to generate outcome. 

# **4. Logistic Regression for Entire Dataset**

# 4.1 Build a regression model for the entire data

# In[31]:


#Turn Up/Down to 0,1
df['DirectionNumber'] = df['Direction'].map({'Up': 1, 'Down': 0})
df.head()


# In[32]:


#Create Input
inputDf = df[['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]
outputDf = df[['DirectionNumber']]


# In[33]:


#Create Model
logisticRegr = LogisticRegression()
logisticRegr.fit(inputDf, outputDf)
print(logisticRegr.intercept_, logisticRegr.coef_)


# 4.2 Predict and Compare with actual Up/Down values in the sample data set

# In[34]:


#Prediction
y_pred = logisticRegr.predict(inputDf)

dresult = pd.DataFrame()
dresult["y_pred"] = y_pred;
dresult["y_original"] = df["DirectionNumber"];
print(dresult)


# 4.3 Count when Up was predicted correctly and incorrectly. Determine what percent of outcomes were predicted correctly.

# In[35]:


#counting result
correctValue =  len(dresult[(dresult['y_original'] == 1) & (dresult['y_pred'] == 1)])
incorrectValue = len(dresult[(dresult['y_original'] == 1) & (dresult['y_pred'] == 0)])
print("correct:" + str(correctValue))
print("incorrect:" + str(incorrectValue))
print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))


# 4.4 Count when Down was predicted correctly and incorrectly

# In[36]:


#counting result
correctValue =  len(dresult[(dresult['y_original'] == 0) & (dresult['y_pred'] == 0)])
incorrectValue = len(dresult[(dresult['y_original'] == 0) & (dresult['y_pred'] == 1)])
print("correct:" + str(correctValue))
print("incorrect:" + str(incorrectValue))
print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))


# 4.5 Which is more correct Up or Down? Is there a difference?

# Ans: Up is more correct as its percentage is 100%. Meanwhile, the percentage of incorrectness of down is only 1.2%.

# **5. Logistic Regression with Training/Test Separation**

# In[37]:


df.describe()


# 5.1 Use data for 2005 as the test set and ALL other data as training set

# In[38]:


#Divide the data into Training and Test data sets
#Use data for 2005 as the test set and ALL other data as training set
dfWithOut2005 = df[(df['Year'] != 2005)]
inputDfWithOut2005 = dfWithOut2005[['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]
outputDfWithOut2005 = dfWithOut2005[['DirectionNumber']]
inputDfWithOut2005.describe()


# In[39]:


#Create Model
logisticRegrTestTrain = LogisticRegression()
logisticRegrTestTrain.fit(inputDfWithOut2005, outputDfWithOut2005)
print(logisticRegrTestTrain.intercept_, logisticRegrTestTrain.coef_)


# In[40]:


#Create Testing Data
dfOnly2005 = df[(df['Year'] == 2005)]
inputDfOnly2005 = dfOnly2005[['Year', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today']]
outputDfOnly2005 = dfOnly2005['DirectionNumber']
outputDfOnly2005 = outputDfOnly2005.reset_index(drop=True)
inputDfOnly2005.describe()


# 5.2 Predict and Compare with actual Up/Down values in the sample data set

# In[41]:


#Prediction
y_pred_Only2005 = logisticRegrTestTrain.predict(inputDfOnly2005)

dresultTestTrain = pd.DataFrame()
dresultTestTrain["y_pred_Only2005"] = y_pred_Only2005;
dresultTestTrain["y_original_Only2005"] = outputDfOnly2005;
print(dresultTestTrain)


# 5.3 Count when Up was predicted correctly and incorrectl

# In[42]:


correctValue =  len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 1) & (dresultTestTrain['y_pred_Only2005'] == 1)])
incorrectValue = len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 1) & (dresultTestTrain['y_pred_Only2005'] != 1)])
print("correct:" + str(correctValue))
print("incorrect:" + str(incorrectValue))
print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))


# 5.4 Count when Down was predicted correctly and incorrectly

# In[43]:


correctValue =  len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 0) & (dresultTestTrain['y_pred_Only2005'] == 0)])
incorrectValue = len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 0) & (dresultTestTrain['y_pred_Only2005'] != 0)])
print("correct:" + str(correctValue))
print("incorrect:" + str(incorrectValue))
print("percentage:" + str(correctValue*100/(correctValue+incorrectValue)))


# 5.5 How does this result compare with the when you used the entire data
# set? Which one do you prefer?

# In[54]:


pFirstMethodCorrect = len(dresult[(dresult['y_original'] == 0) & (dresult['y_pred'] == 0)]) + len(dresult[(dresult['y_original'] == 1) & (dresult['y_pred'] == 1)])
pFirstMethodTotal = dresult['y_original'].count()
print('percentage Corrected of First Method:' + str(pFirstMethodCorrect*100/pFirstMethodTotal))


# In[56]:


pSecondMethodCorrect = len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 0) & (dresultTestTrain['y_pred_Only2005'] == 0)]) + len(dresultTestTrain[(dresultTestTrain['y_original_Only2005'] == 1) & (dresultTestTrain['y_pred_Only2005'] == 1)])
pSecondMethodTotal = dresultTestTrain['y_original_Only2005'].count()
print('percentage Corrected of Second Method:' + str(pSecondMethodCorrect*100/pSecondMethodTotal))


# From the above calculation, the first method can generate the better result. This is because in the second training dataset, there is no data of 2005. This results in the model lacking foundation data of year 2005. Thus, when we feed the input of year 2005, the model generate worse result.
# 
# However, although the first dataset can generate the great result, it might be to fit to the testing data as the testing data is a part of traning data.

# 
