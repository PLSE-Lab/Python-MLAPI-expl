#!/usr/bin/env python
# coding: utf-8

# # Graduate Admission Prediction.

# This NoteBook is a practice for Linear Regression and Logistic Regression, to find out chances of Admission Depends on which factor and Whether a candidate can get Admission or not.

# Importing all the required libraries for data analysis and data visualizations.

# In[ ]:


import numpy as numpyInstance
import pandas as pandasInstance
import seaborn as seabornInstance
import matplotlib.pyplot as matplotLibInstance
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf


# In[ ]:


init_notebook_mode(connected=True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# Now importing the data regarding different tests and other factors of the candidates.

# In[ ]:


candidates_Data = pandasInstance.read_csv('../input/Admission_Predict_Ver1.1.csv')


# Now checking the Header of the Data.

# In[ ]:


candidates_Data.head()


# Now let's get the information about columns.

# In[ ]:


candidates_Data.info()


# Now let's get some more statistical information about the data.

# In[ ]:


candidates_Data.describe()


# ### Part 1 Some Data Exploration.

# Let's Categories and See the Number of Universites According to the University Rating.

# In[ ]:


matplotLibInstance.figure(figsize=(12,10))
seabornInstance.countplot(x='University Rating',data=candidates_Data,palette='winter')


# #### From Above Count Plot we can conclude that there mostly Universities with Rating 3 in data.

# Let's see the Average of the GRE Test.

# In[ ]:


matplotLibInstance.figure(figsize=(12,10))
seabornInstance.distplot(candidates_Data['GRE Score'],color='seagreen')


# #### From Above Histogram we can see that the Average of GRE Score Lies Between 310 to 330.

# Let's See the Average CGPA of the Students.

# In[ ]:


matplotLibInstance.figure(figsize=(12,10))
seabornInstance.distplot(candidates_Data['CGPA'],color='red')


# #### From Above we can conclude that the Average CGPA Lies Between 8 and 9 or 9.5

# Let's Compare the GRE and TOEFL Score Togther and see the comparison. 

# In[ ]:


matplotLibInstance.figure(figsize=(12,10))
seabornInstance.jointplot(x='GRE Score',y='TOEFL Score',data=candidates_Data,kind='hex')


# #### From Above we can conclude that the candidates which have the GRE Score Betwenn 325 to 330 has TOEFL Score of 113 to 115.

# ### Part 2 Determining the Factors on which the Chances of Admissions are Depending here.

# Now Training and Testing the Model.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


candidates_Data.columns


# In[ ]:


X = candidates_Data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]


# In[ ]:


Y = candidates_Data['Chance of Admit ']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linearRegressionInstance = LinearRegression()


# In[ ]:


linearRegressionInstance.fit(X_train,y_train)


# Let's Check the Intercept and the Co-Efficients.

# In[ ]:


linearRegressionInstance.intercept_


# In[ ]:


linearRegressionInstance.coef_


# Let's Describe the Co-Efficient in more Understandable Form.

# In[ ]:


coefficientsDescription = pandasInstance.DataFrame(data=linearRegressionInstance.coef_,index=X_train.columns,columns=['Values'])


# In[ ]:


coefficientsDescription


# Now let's Make Predictions of the Chances of Admission.

# In[ ]:


predictionsOfAdmission = linearRegressionInstance.predict(X_test)


# ##### Now let's check that whether the Linear Regression Model was a good choice or not.

# In[ ]:


matplotLibInstance.figure(figsize=(12,10))
seabornInstance.scatterplot(x=y_test,y=predictionsOfAdmission,color='crimson')


# Let's Do another Test to Verify.

# In[ ]:


matplotLibInstance.figure(figsize=(12,10))
seabornInstance.distplot(y_test - predictionsOfAdmission,color='purple')


# ##### As We Can See that there is a Normal Distributation so it means that it was a good choice to Select The Linear Regression Model.

# Now let's Look the Coefficients what they say.

# In[ ]:


coefficientsDescription


# #### We can see taht If GRE Score Increase by one unit then the Chances of Admission will be Increased by 0.001452

# #### We can see taht If TOEFL Score Increase by one unit then the Chances of Admission will be Increased by 0.003024

# #### We can see taht If University Rating Increase by one unit then the Chances of Admission will be Increased by 0.003024

# #### We can see taht If SOP Increase by one unit then the Chances of Admission will be Increased by 0.006722

# #### We can see taht If LOR Increase by one unit then the Chances of Admission will be Increased by 0.013184

# #### We can see taht If CGPA Increase by one unit then the Chances of Admission will be Increased by 0.120029

# #### We can see taht If Research increased by one unit then the Chances of Admission will be Increased by 0.024772

# ### Part 3 Predicting Whether a candidate can get admission or not.

# Let's Define a Function which will return 1 if Chances of Getting Admission is Higher then 0.60 otherwise it will return 0.

# In[ ]:


def calGetAdmssionOrNot(cal):
    if cal >=0.60:
        return 1
    else:
        return 0
    


# In[ ]:


gotAdmissionOrNot = candidates_Data['Chance of Admit '].apply(calGetAdmssionOrNot)


# In[ ]:


candidates_Data['Get Admission'] = gotAdmissionOrNot


# In[ ]:


candidates_Data


# In[ ]:


candidates_Data.head()


# Now let's seaperate the Factors on which predictions will be made.

# In[ ]:


candidates_Data.columns


# In[ ]:


X = candidates_Data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]


# In[ ]:


Y_Predict = candidates_Data['Get Admission']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y_Predict, test_size=0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logisticRegressionInstance = LogisticRegression()


# In[ ]:


logisticRegressionInstance.fit(X_train,y_train)


# In[ ]:


admissionPredictions = logisticRegressionInstance.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,admissionPredictions))


# #### From Here We can Say that With the Precision of 90% 163 Students have a chance to get Admission.
