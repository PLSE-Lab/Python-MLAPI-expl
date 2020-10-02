#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction - multiple regression
# In this tutorial I explain how to perform multiple regression using python. I am not ver familiar with python, so for me this is also a good excercise to learn python.
# 
# In regression analysis you try to fit a predictive model to your data and use that model to predict an outcome variable from one or more independent predictor variables. With simple regression you try to predict an outcome variable from a single predictor variable and with multiple regression you try to predict an outcome variable from multiple predictor variables. 
# 
# This predictive model uses a straight line to summarize the data and the method of least squares is used to get the linear line that gives the description (best fit) of the data. 
# 
# Lets start with importing some packages and reading in the files

# In[ ]:


## import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


## import data
insurance = pd.read_csv('../input/insurance.csv')


# In[ ]:


## have a peak at the data
print(insurance.head()) # print first 4 rows of data
print(insurance.info()) # get some info on variables


# It is always a good idea to have a peak at the data. Here we see that the dataset contains 1338 observations of 7 variables. The variable charges is the one we have to predict. From the output we see that this is a float variable. This means that this is a numerical variable containing a decimal. The variable charge can be predicted to make use of the following predictors:
# age, sex, bmi, children, smoker and region. The variable age and bmi are continous variables, the variables sex, smoker and region are categorical variables. In the summary statistics we also see that there are no missings in the dataset. 
# 
# Lets make some plot to have a better look at the data.

# In[ ]:


# scatter plot charges ~ age
insurance.plot.scatter(x='age', y='charges') 

# scatter plot charges ~ bmi
insurance.plot.scatter(x='bmi', y='charges')


# In the first plot we see that there is a trend that with older age the charges increase. There are also three groups/lines visible. In the second plot we see some sort of trend that with increasing bmi the charges increase, however this is not very clear. Here there might also be two different groups.

# In[ ]:


print(insurance.boxplot(column = 'charges', by = 'smoker'))

print(insurance.boxplot(column = 'charges', by = 'sex'))

print(insurance.boxplot(column = 'charges', by = 'children'))

print(insurance.boxplot(column = 'charges', by = 'region'))


# The first boxplot (left upper corner) shows us that females and males pay on avarage the same charges. When looking at the second boxplot (right upper corner) we see that smokers pay higher charges compared to non smokers. Also people with more childres pay more charges and it seems that the region has not an influence on the charges. In all instances the charges have a skewed distribution.
# 
# # 2. Simple regression
# 
# Lets start predicting the outcome variable charges using only the predictor variable age. This is called simple regression. With simple linear regression you use the method of least squares to find a line that best fits the data. With other words finding a line that goes through as many points as possible (or tries to be very close to those points). However, since the line is linear it will never cross al datapoints. This means that there is a difference between the model (the linear line) and the reality (all the points). You can calculate the difference between the model and the reality by taking the difference between the line and the datapoints. These differences are called residuals. Here is the formula: 
# 
#    $deviation =  \sum(observed - model)^{2}$
#    
# This formula simply means take the difference between each observed value and the value according to the model, sum these differences and take the square.
# 
# ## 2.1 Goodness-of-fit, sums of squares, R and R-squared
# 
# Imaging if we have only information on charges, than if you need to make a prediction, the best thing you can do is taking the mean charges. This is the most simple model available. This model is represented by the vertical black line in the plot below. Using the mean we can calculate the difference between the mean and all the observed values. This is called the total sum of squares (SSt). Next you obtain information on age and use this information to make a linear model to predict charges. This model is represented by the blue regression line. Here you can calculate the differences between all the observed values and the regression line. This is called the residual sum of squares (SSr). Now it is possible to calculate the differences between the mean value of charges and the regression line. This is called the model sum of squares (SSm). 
# Now you can use the SSm and the SSt to calculate the percentage of variation in the outcome (charges) explained by the model using the following formula:
# 
# $R^{2} = \frac{SSm}{SSt}$
# 
# This is what you use to assess goodness-of-fit.
# 
# In our case only one variable (age) is used to predict the outcome (charge), so if you take the square root of R-squared you get the Pearson correlation coefficient. Note that this is not working when you have more than one predictors. 

# In[ ]:


## make scatter plot with age on x and charges on y, 
## also show regression line 
insurance.plot.scatter(x='age', y='charges') 


# In[ ]:


# simple linear regression using age as a predictor
X = insurance["age"] ## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model1 = sm.OLS(y, X).fit()
predictions = model1.predict(X) # make the predictions by the model

# Print out the statistics
model1.summary()


# The above output the is the summary of our model. We see that our dependent variable is charges and that we used the method of least squares to make the regression line. The R-square is 0.089, meaning that age explains 8.9% of the variation of charges. We also see that the p-value of the F-stastic is < 0.001 meaning that our model is statistically significant improve than just taking the mean charges. Note that when you take the square root of square-R, you also get the correlation coefficient between age and charges.
# 
# The coefficient for age is 257.7, this means that with every increase in age (in years) the charges increase with 257.7. The intercept (see const) is at 3165.9. Thus our model for predicting charges using age becomes: 
# 
# $charges = 3165.9 + (257.7 * age) + error$
# 
# # 3. Multiple regression
# 
# The formula for multiple regression looks as follows: 
# 
# $Y_i = (b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n) + e_i$
# 
# $Y$ is the outcome variable, $b_1$ is the coefficient for the first predictor ($X_1$), $b_2$ is the coefficient for the second predictor ($X_2$) and $b_n$ is the coefficient for the nth predictor ($X_n$). $b_o$ is the intercept, the point were the regression line crosses the y-axis. Ei is the difference between the predicted and the observed value of $Y$ for the ith participant. Lets put this into practice by adding a second variable to our model, for example body mass index (BMI). Here is how the model looks like:
# 
# $Charges = b_0 + b_1*age_i + b_2*bmi_i + e_i$
# 
# Lets apply this in Python:

# In[ ]:


# multiple linear regression using age and bmi as a predictor
X = insurance[["age", "bmi"]]## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model2 = sm.OLS(y, X).fit()
predictions = model2.predict(X) # make the predictions by the model

# Print out the statistics
model2.summary()


# In the output we see that we have a multiple R-squared of 0.1172, this means that our model explains 11.72% of the variation in the outcome variable. Multiple R-squared can range from 1 to 0, were 1 means that the model perfectly fits the observed data and explains 100% of the variation in the outcome variable. In our model with only the predictor age, the R-squared was 0.08941. Thus adding BMI to the model improves the fit. But how do we know if this improvement in R-square is significant? Well we can compare both models by calculating the F-ratio. Here is how to do is in python:

# In[ ]:


# calculate f-ratio 
anovaResults = anova_lm(model1, model2)
print(anovaResults)


# The output shows that the F-value is 42.00, and with a p<0.001 we know that the change in explained variance is significant (I have no idea why I get these errors). Note that you can compare only hierarchical models. So the second model should contain all variables from the first model plus some new and the third model should contain all variables from the second model plus some new and so on. 
# 
# ## 3.1 Methods of regression
# 
# When you are building a model with several predictors, how do you know which predictors to use? The best practice is that the selection of predictors you want to add to your model are based on previous research. At least you should not add hundreds of random predictors to you model. Another problem is to decide in which order you should enter the variables in the model. If all the predictors are not correlated, than the order is not a problem, however this usually not the case. 
# 
# There are several ways to decide on the order about putting variables in the model
# 
# * Hierarchical or blockwise entry: the predictors are based on past work and the researcher decides in which order the variables are entered in the model. This order should be based on the importance of the variables. The most important variable is entered first and so on. Than the new predictors can be entered.
# 
# * forced entry: all the predictors are put in the model at once
# 
# * stepwise methods: the order in which the predictors are entered in the model are based on mathematical criteria. You can use a forward and a backward method. In a forward method the first variable that in entered in the model, is the one that explains most of the variation of the outcome variable, the next variable entered in the model explains the largest part of the remaining variation and so on. In a backward method all the variables are entered in the model and one by one the variables are removed that explain the smallest part of the variation. To avoid overfitting it is important to cross-validate the model.
# 
# * all sub-sets method: need to explain this further
# 
# In this tutorial I assume I did background research (which I didn't) and a found that smoking was the most important predictor for charges, next came age and than bmi. I want to know if sex, region and the number of children will improve the model. First I make a model that includes the known predictors. Since I don't which of the new predictors (sex, region or number of children is important, I will add then add once in the new model.
# 
# For a following tutorial it would be interesting to compare the different approaches.
# 
# Lets start with the first model containing the predictors smoker, age and bmi

# In[ ]:


# make dummy variable for categorical variables
insurance = pd.get_dummies(insurance, columns=['smoker'])
print(insurance.head()) # print first 4 rows of data


# In[ ]:


# multiple linear regression using smoker, age and bmi as a predictor
X = insurance[["smoker_yes", "age", "bmi"]] ## the input variables,
                                            ## only include smoker_yes
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order, first y than X
model3 = sm.OLS(y, X).fit()
predictions = model3.predict(X) # make the predictions by the model

# Print out the statistics
model3.summary()


# When looking at the R-squared I see that 74.7 percent of the variation in the outcome variable charge is explained by the three predictor variables smoker, age and bmi. The adjusted R-square is almost equal to the multiple R-square, this meaning that the model has good cross-validity. From the F-statistic (P<0.001) we can conclude that the model is better than just taking the mean charges. We also notice that the predictor variables are all significant. From the coefficients we can make our model:
# 
# $charges = -11676.83 + 23823.68*smoker + 259.55*age + 322.62*bmi + error$
# 
# Next we make the following model were we also include the predictor variables sex, children and region.

# In[ ]:


# make dummy variables
insurance = pd.get_dummies(insurance, columns=['sex', 'region'])
print(insurance.info()) # print first 4 rows of data

# multiple linear regression using smoker, age and bmi as a predictor
X = insurance[["smoker_yes", "age", "bmi", "sex_male", 
               "children", "region_northwest",'region_southeast', 
               'region_southwest']] ## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

X.head()
# Note the difference in argument order, first y than X
model4 = sm.OLS(y, X).fit()
#predictions = model2.predict(X) # make the predictions by the model

# Print out the statistics
model4.summary()


# In the new model the multiple R_square is 0.7509, meaning that the new model explains 75.1 of the variation in charges. When looking at the p-values we see that sex is not significant. Lets remove sex from the model. Lets compare both models

# In[ ]:


# multiple linear regression using smoker, age and bmi as a predictor
X = insurance[["smoker_yes", "age", "bmi", 
               "children", "region_northwest",'region_southeast', 
               'region_southwest']] ## the input variables
y = insurance["charges"] ## the output variables, the one you want to predict
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

X.head()
# Note the difference in argument order, first y than X
model5 = sm.OLS(y, X).fit()
#predictions = model2.predict(X) # make the predictions by the model

# Print out the statistics
model5.summary()

