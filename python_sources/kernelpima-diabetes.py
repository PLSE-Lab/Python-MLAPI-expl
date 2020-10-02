#!/usr/bin/env python
# coding: utf-8

# # Pima Diabetes Data Analytics - Neil
# Here are the set the analytics that has been run on this data set
# - Data Cleaning to remove zeros
# - Data Exploration for Y and Xs
# - Descriptive Statistics - Numerical Summary and Graphical (Histograms) for all variables
# - Screening of variables by segmenting  them by Outcome
# - Check for normality of dataset
# - Study bivariate relationship between variables using pair plots, correlation and heat map
# - Statistical screening using Logistic Regression
# - Validation of the model its precision and ploting of confusion matrix

# # Importing necessary packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes =True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the Diabetes CSV data file
# - Import the data and test if all the columns are loaded
# - The Data frame has been assigned a name of 'diab'

# In[ ]:


diab=pd.read_csv("../input/pima-diabetes/diabetes.csv")
diab.head()


# ## About data set
# In this data set, Outcome is the Dependent Variable and Remaining 8 variables are independent variables. 

# ## Finding if there are any null and Zero values in the data set
# 

# In[ ]:


diab.isnull().values.any()
## To check if data contains null values


# ## Inference:
# - Data frame doesn't have any NAN values
# - As a next step, we will do preliminary screening of descriptive stats for the dataset

# In[ ]:


diab.describe()
## To run numerical descriptive stats for the data set


# ## Inference at this point
# - Minimum values for many variables are 0. 
# - As biological parameters like Glucose, BP, Skin thickness,Insulin & BMI cannot have zero values, looks like null values have been coded as zeros
# - As a next step, find out how many Zero values are included in each variable

# In[ ]:


(diab.Pregnancies == 0).sum(),(diab.Glucose==0).sum(),(diab.BloodPressure==0).sum(),(diab.SkinThickness==0).sum(),(diab.Insulin==0).sum(),(diab.BMI==0).sum(),(diab.DiabetesPedigreeFunction==0).sum(),(diab.Age==0).sum()
## Counting cells with 0 Values for each variable and publishing the counts below


# ## Inference: 
# - As Zero Counts of some the variables are as high as 374 and 227, in a 768 data set, it is better to remove the Zeros uniformly for 5 variables (excl Pregnancies & Outcome)
# - As a next step, we'll drop 0 values and create a our new dataset which can be used for further analysis

# In[ ]:


## Creating a dataset called 'dia' from original dataset 'diab' with excludes all rows with have zeros only for Glucose, BP, Skinthickness, Insulin and BMI, as other columns can contain Zero values.
drop_Glu=diab.index[diab.Glucose == 0].tolist()
drop_BP=diab.index[diab.BloodPressure == 0].tolist()
drop_Skin = diab.index[diab.SkinThickness==0].tolist()
drop_Ins = diab.index[diab.Insulin==0].tolist()
drop_BMI = diab.index[diab.BMI==0].tolist()
c=drop_Glu+drop_BP+drop_Skin+drop_Ins+drop_BMI
dia=diab.drop(diab.index[c])


# In[ ]:


dia.info()


# ## Inference
# - As in above, created a cleaned up list titled "dia" which has 392 rows of data instead of 768 from original list
# - Looks like we lost nearly 50% of data but our data set is now cleaner than before
# - In fact the removed values can be used for Testing during modeling. So actually we haven't really lost them completly.

# ## Performing Preliminary Descriptive Stats on the Data set
#  - Performing 5 number summary
#  - Usually, the first thing to do in a data set is to get a hang of vital parameters of all variables and thus understand a little bit about the data set such as central tendency and dispersion

# In[ ]:


dia.describe()


# ## Split the data frame into two sub sets for convenience of analysis
#  - As we wish to study the influence of each variable on Outcome (Diabetic or not), we can subset the data by Outcome
#  - dia1 Subset : All samples with 1 values of Outcome
#  - dia0 Subset: All samples with 0 values of Outcome
#  

# In[ ]:


dia1 = dia[dia.Outcome==1]
dia0 = dia[dia.Outcome==0]


# In[ ]:


dia1


# In[ ]:


dia0


# ## Graphical screening for variables
# - Now we will start graphical analysis of outcome. At the data is nominal(binary), we will run count plot and compute %ages of samples who are diabetic and non-diabetic

# In[ ]:


## creating count plot with title using seaborn
sns.countplot(x=dia.Outcome)
plt.title("Count Plot for Outcome")


# In[ ]:


# Computing the %age of diabetic and non-diabetic in the sample
Out0=len(dia[dia.Outcome==1])
Out1=len(dia[dia.Outcome==0])
Total=Out0+Out1
PC_of_1 = Out1*100/Total
PC_of_0 = Out0*100/Total
PC_of_1, PC_of_0


# ## Inference on screening Outcome variable
# - There are 66.8% 1's (diabetic) and 33.1% 0's (nondiabetic) in the data
# - As a next step, we will start screening variables

# ## Graphical Screening for Variables
# - We will take each variable, one at a time and screen them in the following manner
# - Study the data distribution (histogram) of each variable - Central tendency, Spread, Distortion(Skewness & Kurtosis)
# - To visually screen the association between 'Outcome' and each variable by plotting histograms & Boxplots by Outcome value  

# ## Screening Variable - Pregnancies

# In[ ]:


## Creating 3 subplots - 1st for histogram, 2nd for histogram segmented by Outcome and 3rd for representing same segmentation using boxplot
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.set_style("dark")
plt.title("Histogram for Pregnancies")
sns.distplot(dia.Pregnancies,kde=False)
plt.subplot(1,3,2)
sns.distplot(dia0.Pregnancies,kde=False,color="Blue", label="Preg for Outome=0")
sns.distplot(dia1.Pregnancies,kde=False,color = "Gold", label = "Preg for Outcome=1")
plt.title("Histograms for Preg by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Pregnancies)
plt.title("Boxplot for Preg by Outcome")


# ## Inference on Pregnancies
# - Visually, data is right skewed. For data of count of pregenancies. A large proportion of the participants are zero count on pregnancy. As the data set includes women > 21 yrs, its likely that many are unmarried
# - When looking at the segemented histograms, a hypothesis is the as pregnancies includes, women are more likely to be diabetic
# - In the boxplots, we find few outliers in both subsets. Esp some non-diabetic women have had many pregenancies. I wouldn't be worried. 
# - To validate this hypothesis, need to statistically test. 

# ## Screening Variable - Glucose

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
plt.title("Histogram for Glucose")
sns.distplot(dia.Glucose, kde=False)
plt.subplot(1,3,2)
sns.distplot(dia0.Glucose,kde=False,color="Gold", label="Gluc for Outcome=0")
sns.distplot(dia1.Glucose, kde=False, color="Blue", label = "Gloc for Outcome=1")
plt.title("Histograms for Glucose by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Glucose)
plt.title("Boxplot for Glucose by Outcome")


# ## Inference on Glucose
#  - 1st graph - Histogram of Glucose data is slightly skewed to right. Understandably, the data set contains over 60% who are diabetic and its likely that their Glucose levels were higher. But the grand mean of Glucose is at 122.\
#  - 2nd graph - Clearly diabetic group has higher glucose than non-diabetic. 
#  - 3rd graph - In the boxplot, visually skewness seems acceptable (<2) and its also likely that confidence intervels of the means are not overlapping. So a hypothesis that Glucose is measure of outcome, is likely to be true. But needs to be statistically tested.

# ## Screening Variable - Blood Pressure

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.BloodPressure, kde=False)
plt.title("Histogram for Blood Pressure")
plt.subplot(1,3,2)
sns.distplot(dia0.BloodPressure,kde=False,color="Gold",label="BP for Outcome=0")
sns.distplot(dia1.BloodPressure,kde=False, color="Blue", label="BP for Outcome=1")
plt.legend()
plt.title("Histogram of Blood Pressure by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.BloodPressure)
plt.title("Boxplot of BP by Outcome")


# ## Inference on Blood Pressure
# - 1st graph - Distribution looks normal. Mean value is 69, well within normal values for diastolic of 80. One should expect this data to be normal, but as we don't know if the particpants are only hypertensive medication, we can't comment much.
# - 2nd graph - Most non diabetic women seem to have nominal value of 69 and diabetic women seems to have high BP. 
# - 3rd graph - Few outliers in the data. Its likely that some people have low and some have high BP. So the association between diabetic (Outcome) and BP is an suspect and needs to be statistically validated.

# ## Screening Variable - Skin Thickness

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.SkinThickness, kde=False)
plt.title("Histogram for Skin Thickness")
plt.subplot(1,3,2)
sns.distplot(dia0.SkinThickness, kde=False, color="Gold", label="SkinThick for Outcome=0")
sns.distplot(dia1.SkinThickness, kde=False, color="Blue", label="SkinThick for Outcome=1")
plt.legend()
plt.title("Histogram for SkinThickness by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.SkinThickness)
plt.title("Boxplot of SkinThickness by Outcome")


# ## Inferences for Skinthickness
# - 1st graph - Skin thickness seems be be skewed a bit.
# - 2nd graph - Like BP, people who are not diabetic have lower skin thickness. This is a hypothesis that has to be validated. As data of non-diabetic is skewed but diabetic samples seems to be normally distributed.

# ## Screening Variable - Insulin

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.Insulin,kde=False)
plt.title("Histogram of Insulin")
plt.subplot(1,3,2)
sns.distplot(dia0.Insulin,kde=False, color="Gold", label="Insulin for Outcome=0")
sns.distplot(dia1.Insulin,kde=False, color="Blue", label="Insuline for Outcome=1")
plt.title("Histogram for Insulin by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.Insulin)
plt.title("Boxplot for Insulin by Outcome")


# ## Inference for Insulin
# - 2hour serum insulin is expected to be between 16 to 166. Clearly there are Outliers in the data. These Outliers are concern for us and most of them with higher insulin values ar also diabetic. So this is a suspect.

# ## Screening Variable - BMI

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.BMI, kde=False)
plt.title("Histogram for BMI")
plt.subplot(1,3,2)
sns.distplot(dia0.BMI, kde=False,color="Gold", label="BMI for Outcome=0")
sns.distplot(dia1.BMI, kde=False, color="Blue", label="BMI for Outcome=1")
plt.legend()
plt.title("Histogram for BMI by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.BMI)
plt.title("Boxplot for BMI by Outcome")


# ## Inference for BMI
# - 1st graph - There are few outliers. Few are obese in the dataset. Expected range is between 18 to 25. In general, people are obese
# - 2nd graph - Diabetic people seems to be only higher side of BMI. Also the contribute more for outliers
# - 3rd graph - Same inference as 2nd graph

# ## Screening Variable - Diabetes Pedigree Function

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.DiabetesPedigreeFunction,kde=False)
plt.title("Histogram for Diabetes Pedigree Function")
plt.subplot(1,3,2)
sns.distplot(dia0.DiabetesPedigreeFunction, kde=False, color="Gold", label="PedFunction for Outcome=0")
sns.distplot(dia1.DiabetesPedigreeFunction, kde=False, color="Blue", label="PedFunction for Outcome=1")
plt.legend()
plt.title("Histogram for DiabetesPedigreeFunction by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.DiabetesPedigreeFunction)
plt.title("Boxplot for DiabetesPedigreeFunction by Outcome")


# ## Inference of Diabetes Pedigree Function
# - I dont know what this variable is. But it doesn't seem to contribute to diabetes
# - Data is skewed. I don't know if his parameter is expected to be a normal distribution. Not all natural parameters are normal
# - As DPF increases, there seems to be a likelihood of being diabetic, but needs statistical validation

# ## Screening Variable - Age

# In[ ]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.Age,kde=False)
plt.title("Histogram for Age")
plt.subplot(1,3,2)
sns.distplot(dia0.Age,kde=False,color="Gold", label="Age for Outcome=0")
sns.distplot(dia1.Age,kde=False, color="Blue", label="Age for Outcome=1")
plt.legend()
plt.title("Histogram for Age by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Age)
plt.title("Boxplot for Age by Outcome")


# ## Inference for Age
# - Age is skewed. Yes, as this is life data, it is likely to fall into a weibull distribution and not normal
# - There is a tendency that as people age, they are likely to become diabetic. This needs statistical validation
# - But diabetes, itself doesn't seem to have an influence of longetivity. May be it impacts quality of life which is not measured in this data set.
# 

# ## Normality Test
# 
# Inference: None of the variables are normal. (P>0.05) May be subsets are normal

# In[ ]:


## importing stats module from scipy
from scipy import stats
## retrieving p value from normality test function
PregnanciesPVAL=stats.normaltest(dia.Pregnancies).pvalue
GlucosePVAL=stats.normaltest(dia.Glucose).pvalue
BloodPressurePVAL=stats.normaltest(dia.BloodPressure).pvalue
SkinThicknessPVAL=stats.normaltest(dia.SkinThickness).pvalue
InsulinPVAL=stats.normaltest(dia.Insulin).pvalue
BMIPVAL=stats.normaltest(dia.BMI).pvalue
DiaPeFuPVAL=stats.normaltest(dia.DiabetesPedigreeFunction).pvalue
AgePVAL=stats.normaltest(dia.Age).pvalue
## Printing the values
print("Pregnancies P Value is " + str(PregnanciesPVAL))
print("Glucose P Value is " + str(GlucosePVAL))
print("BloodPressure P Value is " + str(BloodPressurePVAL))
print("Skin Thickness P Value is " + str(SkinThicknessPVAL))
print("Insulin P Value is " + str(InsulinPVAL))
print("BMI P Value is " + str(BMIPVAL))
print("Diabetes Pedigree Function P Value is " + str(DiaPeFuPVAL))
print("Age P Value is " + str(AgePVAL))


# ## Screening of Association between Variables to study Bivariate relationship
# - We will use pairplot to study the association between variables - from individual scatter plots
# - Then we will compute pearson correlation coefficient
# - Then we will summarize the same as heatmap

# In[ ]:


sns.pairplot(dia, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
plt.title("Pairplot of Variables by Outcome")


# ## Inference from Pair Plots
# 
# - From scatter plots, to me only BMI & SkinThickness and Pregnancies & Age seem to have positive linear relationships. Another likely suspect is Glucose and Insulin.
# - There are no non-linear relationships
# - Lets check it out with Pearson Correlation and plot heat maps

# In[ ]:


cor = dia.corr(method ='pearson')
cor


# In[ ]:


sns.heatmap(cor)


# ## Inference from 'r' values and heat map
# - No 2 factors have strong linear relationships
# - Age & Pregnancies and BMI & SkinThickness have moderate positive linear relationship
# - Glucose & Insulin technically has low correlation but 0.58 is close to 0.6 so can be assumed as moderate correlation

# # Final Inference before model building
# 
# - Data set contains many zero values and they have been removed and remaining data has been used for screening and model building
# - Nearly 66% of participants are diabetic in the sample data
# - Visual screening (boxplots and segmented histograms) shows that few factors seem to influence the outcome 
# - Moderate correlation exists between few factors and so while building model, this has to be borne in mind. If co-correlated factors are included, it might lead to Inflation of Variance.
# ----------------------------------------------
# - As a next step, a binary logistic regression model has been built

# #  Logistic Regression 
# - A logistic regression is used from the dependent variable is binary, ordinal or nominal and the independent variables are either continuous or discrete
# - In this scenario, a Logit Model has been used to fit the data
# - In this case an event is defined as occurance of '1' in outcome
# - Basically logistic regression uses the odds ratio to build the model

# In[ ]:


cols=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"]
X=dia[cols]
y=dia.Outcome


# In[ ]:


## Importing stats models for running logistic regression
import statsmodels.api as sm
## Defining the model and assigning Y (Dependent) and X (Independent Variables)
logit_model=sm.Logit(y,X)
## Fitting the model and publishing the results
result=logit_model.fit()
print(result.summary())


# ## Inference from the Logistic Regression
# - The R sq value of the model is 56%.. that is this model can explain 56% of the variation in dependent variable
# - To identify which variables influence the outcome, we will look at the p-value of each variable. We expect the p-value to be less than 0.05(alpha risk)
# - When p-value<0.05, we can say the variable influences the outcome
# - Hence we will eliminate Diabetes Pedigree Function, Age, Insulin and re run the model

# ## 2nd itertion of the Logistic Regression with fewer variables

# In[ ]:


cols2=["Pregnancies", "Glucose","BloodPressure","SkinThickness","BMI"]
X=dia[cols2]


# In[ ]:


logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# ## Inference from 2nd Iteration
# - We will now eliminate BMI and re run the model

# ## 3rd iteration of Logistic Regression

# In[ ]:


cols3=["Pregnancies", "Glucose","BloodPressure","SkinThickness"]
X=dia[cols3]
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


# ## Inference from 3rd Iteration
# - Now the P value of skinthickness is greater than 0.05, hence we will eliminate it and re run the model

# ## 4th Iteration of Logistic Regression

# In[ ]:


cols4=["Pregnancies", "Glucose","BloodPressure"]
X=dia[cols4]
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


# ## Inference from 4th Run
# - Now the model is clear. We have 3 variables that influence the Outcome and then are Pregnancies, Glucose and BloodPressure
# - Luckly, none of these 3 variables are co-correlated. Hence we can safetly assume tha the model is not inflated

# In[ ]:


## Importing LogisticRegression from Sk.Learn linear model as stats model function cannot give us classification report and confusion matrix
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
cols4=["Pregnancies", "Glucose","BloodPressure"]
X=dia[cols4]
y=dia.Outcome
logreg.fit(X,y)
## Defining the y_pred variable for the predicting values. I have taken 392 dia dataset. We can also take a test dataset
y_pred=logreg.predict(X)
## Calculating the precision of the model
from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


# ## Precision of the model is 77%

# In[ ]:


from sklearn.metrics import confusion_matrix
## Confusion matrix gives the number of cases where the model is able to accurately predict the outcomes.. both 1 and 0 and how many cases it gives false positive and false negatives
confusion_matrix = confusion_matrix(y, y_pred)
print(confusion_matrix)


# ## The result is telling us that we have 234+69 are correct predictions and 61+28 are incorrect predictions.
