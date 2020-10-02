#!/usr/bin/env python
# coding: utf-8

# ## Exploring Graduate Admissions data and using Linear Regression to fit a predictive model
# 
# The data is about the GRE and TOEFL scores, University Rating, SOP and LOR, CGPA, Research Experience and the chances of securing an admit in an US university - of students in India.
# 
# This notebook does 
# - exploratory data analysis
# - A couple of hypothesis tests
# - A detailed walk through on fitting a Linear Regression model on this data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


graduate_data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")


# In[ ]:


graduate_data.shape


# In[ ]:


## Dropping the Serial No column
graduate_data.drop(columns = 'Serial No.', inplace = True)


# In[ ]:


## check for missing values
## turns our there aren't any
graduate_data.isnull().sum()


# ### Distribution of the variables
# 
# Some observations from the histograms of all the variables:
# - The GRE scores are concentrate around the 320 mark and the lowest is around 290.
# - Most of the students have TOEFL scores in the 100-110 range.
# - Most of the students have average to good SOPs and LORs.
# - CGPA is concentrated on the higher side with the mean around 8.5.
# - Almost half the students do not have research experience.
# - More than 40% feel they have a greater than 75% chance of getting an admit.

# In[ ]:


fig, ax = plt.subplots(2,4)
fig.set_size_inches(11,11)
fig.suptitle("Distribution of the different variables", fontsize = 16)
fig.text(0.06, 0.5, 'Count of Occurence', ha='center', va='center', rotation='vertical', fontsize=16)
k = 0
for i in range(0,2):
    for j in range(0,4):
        ax[i,j].hist(graduate_data.iloc[0:,k], bins = 10)
        ax[i,j].set_title(graduate_data.columns[k])
        k += 1
        #print(k)
       
plt.show()


# ### Bi-variate relationship between the independent variables and the target
# Pretty much conforms to established notions of the influence of various independent factors on the chances of admit.
# - Higher the GRE and TOEFl scores - higher the chances of admit. 
# - Better undergraduate University rating, better the chances of admit as is the case with SOP and LOR.
# - Undergraduate CGPA clearly has a strong impact on the chances of admit - but the same cannot be said about the research experience.

# In[ ]:


fig, ax = plt.subplots(2,4, sharey = 'row')
fig.set_size_inches(11,11)
fig.suptitle("Relation between Independent variables and Dependent variable", fontsize = 16)
k = 0
for i in range(0,2):
    for j in range(0,4):
        ax[i,j].scatter(graduate_data.iloc[0:,k],graduate_data.iloc[:,7])
        ax[i,j].set_title(graduate_data.columns[k])
        k += 1
        #print(k)
fig.text(0.06, 0.5, 'Chances of Admit', ha='center', va='center', rotation='vertical', fontsize=16)
fig.delaxes(ax[1,3])        
plt.show()


# ### Checking for correlation between variables
# 
# Some quick observations from the correlation plot:
# 
# - There is no negative correlation between any of the variables
# - GRE scores and TOEFL scores are highly correlated
# - GRE score has just a weak correlation with LOR and Research experience
# - I was expecting a high correlation between LOR and CGPA - but not really the case
# - More than GRE/ TOEFL, chances of admit is highly correlated with CGPA. Least correlated with Research experience

# In[ ]:


fig = plt.figure(figsize = (9,9))

corr_graduate = graduate_data.corr()
# Generate a mask for the upper right triangle of the square - one half is enough to convey the correlation 
## between the predictors
mask = np.zeros_like(corr_graduate, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate the correlation plot 
sns.heatmap(corr_graduate, mask = mask, center=0, annot = True, square=True, linewidths=.5).set_title("Correlation Plot", fontsize = 16)

plt.show()


# ### To check if there is a clear distinction in admit chances for different brackets of GRE/ TOEFL scores
# 
# To do this, the Admit chances have been binned into four quartiles. 
# 
# Some observations:
# - The green lump at the top right is quite homogenous - i.e mostly composed of the top quartile of the admit chances.
# - There are a few students in the second highest admit quartile but with relatively lesser TOEFL/ GRE scores. It would be interesting to find out the factor(s) which differentiates or gives these students a higher admit chance than students who have scored slightly better than them in GRE/ TOEFL.
# - Anything less than 305/ 102 in GRE/ TOEFL, surely decreases the chances of admit.

# In[ ]:


##Binning the admit chances into four quartiles
#graduate_data['Admit_quartiles'] = pd.qcut(graduate_data['Chance of Admit '], 4, labels = ['low','medium','good','Almost there'])
graduate_data['Admit_quartiles'] = pd.qcut(graduate_data['Chance of Admit '], 4, labels = False)


# In[ ]:


## Scatter Plot with GRE/ TOEFL scores on axes and the the admit chance quartiles as different quartiles
## The highest admit chance quartile is 3 which is mapped to Green and the lowest is 0 which is mapped to Red. 

color_dict = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'green'}

fig, ax = plt.subplots()
fig.set_size_inches(9,9)
for quartile in graduate_data['Admit_quartiles'].unique():
    index = graduate_data.index[graduate_data['Admit_quartiles']==quartile]
    ax.scatter(graduate_data.loc[index,'GRE Score'], graduate_data.loc[index,'TOEFL Score'], 
               c = color_dict[quartile], label = quartile, s = 50)
ax.legend()
ax.set_xlabel("GRE Score")
ax.set_ylabel("TOEFL Score")
ax.set_title("Relation between Admit chance and GRE/ TOEFL score", fontsize = 16)
plt.show()


# ### To check if there is a clear distinction in admit chances for different brackets of University Rating/ CGPA
# - Again a similar trend with high University Rating and high CGPA clearly increasing the chances of admit
# - There are a few students though who despite their lower University Rating figure in the top quartile of Admits - possibly because of their high CGPA and may be high GRE/ TOEFL scores.

# In[ ]:


## Scatter Plot with University Rating/ CGPA on axes and the the admit chance quartiles as different quartiles
## The highest admit chance quartile is 3 which is mapped to Green and the lowest is 0 which is mapped to Red. 

color_dict = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'green'}

fig, ax = plt.subplots()
fig.set_size_inches(9,9)
for quartile in graduate_data['Admit_quartiles'].unique():
    index = graduate_data.index[graduate_data['Admit_quartiles']==quartile]
    ax.scatter(graduate_data.loc[index,'University Rating'], graduate_data.loc[index,'CGPA'], 
               c = color_dict[quartile], label = quartile, s = 50)
ax.legend()
ax.set_xlabel("University Rating")
ax.set_ylabel("CGPA")
ax.set_title("Relation between Admit chance and University Rating/ CGPA score", fontsize = 16)
plt.show()


# ### If GRE/ TOEFL scores are low, can other factors increase your chances of admit?
# 
# A box plot of the Admit quartiles vis-a-vis the different variables is created - for students whose GRE/ TOEFL scores are less than 320 and 110 respectively but more than the minimum scores of the second best Admit quartile.
# 
# Observations:
# - Considering only the second top admit quartile (as the top admit quartile has only two students in this low GRE/ TOEFL score group), we find that barring Research experience (or the lack of it), all the other factors are better for the students with higher admit chances compared to the the lower admit chance students.
# - (The university rating box plot for students in Admit quartile 2 is just a line without a box as the entire interquartile range is composed of the median value 3 - encountering such a box plot for the first time)

# In[ ]:


min_GREscore_Admit2 = graduate_data[graduate_data['Admit_quartiles']==2]['GRE Score'].min()
min_TOEFLscore_Admit2 = graduate_data[graduate_data['Admit_quartiles']==2]['TOEFL Score'].min()

lower_score_analysis = graduate_data.loc[(graduate_data['GRE Score']<320)&(graduate_data['TOEFL Score']<110)&(graduate_data['GRE Score']>min_GREscore_Admit2)&(graduate_data['TOEFL Score']>min_TOEFLscore_Admit2)]


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize = (15,15))
axes = axes.flatten()

#2,3,4,5,6 indicates the column indexes of the columns which we are interested in analysing
for i in [2,3,4,5,6]:
    sns.boxplot(x="Admit_quartiles", y=lower_score_analysis.iloc[:,i], data=lower_score_analysis, orient='v', ax=axes[i-2])
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()


# ### Let us test some assumptions aka hypothesis
# 
# <b>Hypothesis 1: </b> The means of the GRE scores of the student group in Admit quartile 3 (top) is statistically different from the means of the GRE scores of the student group in Admit quartile 2.
# 
# The two sample t-test compares the means of the two sample groups. The null hypothesis here is that there is no statistical difference in the means of the two score groups and the alternate hypothesis is that there is a statistically significant difference between the scores of the two groups.
# 
# With a significance level of 1% (there is a less than 1% chance thst the difference between the two means is due to random chance), we find that the null hypothesis is rejected.

# In[ ]:


from scipy import stats
#from statsmodels.stats import weightstats as stests


# In[ ]:


def t_test_fn(group1, group2, **kwargs):
    ttest, pval = stats.ttest_ind(group1,group2, **kwargs)
    
    if pval<0.01:
        result = "reject null hypothesis"
    else:
        result = "accept null hypothesis"
    #print(result)
    return(ttest, pval, result)


# In[ ]:


GRE_scores_group1 = graduate_data.loc[graduate_data['Admit_quartiles']==3,'GRE Score'].tolist()
GRE_scores_group2 = graduate_data.loc[graduate_data['Admit_quartiles']==2,'GRE Score'].tolist()

t_stat, p_value, result = t_test_fn(GRE_scores_group1, GRE_scores_group2, equal_var = False)
print("T statistic is ",t_stat)
print("Probability of getting this T statistic due to random chance ",p_value)
print(result)


# <b>Hypthesis 2: </b> Since, Research is the least correlated with the chances of admit, we would want to check if there is a statistical difference in the means of Research experience between the groups in the top 2 admit quartiles. 
# 
# But it turns out there is a statistical difference in Research experience between the two groups and the Null Hypothesis is rejected.
# 
# Extending the same test to the bottom two quartiles of Admit chances, the p-value is no more significant and hence we fail to reject the Null Hypothesis - showing that in the bottom two Admit quartiles, there is no statistical difference in the Research experience of the two groups.

# In[ ]:


Research_group1 = graduate_data.loc[graduate_data['Admit_quartiles']==3,'Research'].tolist()
Research_group2 = graduate_data.loc[graduate_data['Admit_quartiles']==2,'Research'].tolist()

t_stat, p_value, result = t_test_fn(Research_group1, Research_group2, equal_var = False)
print("T statistic is ",t_stat)
print("Probability of getting this T statistic due to random chance ",p_value)
print(result)


# In[ ]:


Research_group1 = graduate_data.loc[graduate_data['Admit_quartiles']==1,'Research'].tolist()
Research_group2 = graduate_data.loc[graduate_data['Admit_quartiles']==0,'Research'].tolist()

t_stat, p_value, result = t_test_fn(Research_group1, Research_group2, equal_var = False)
print("T statistic is ",t_stat)
print("Probability of getting this T statistic due to random chance ", p_value)
print(result)


# ### Linear Regression
# 
# Before we attempt Linear Regression, a quick summary of what Linear Regression is all about.
# 
# Assuming we have some data points which has some independent variables and a target variable - can we come up with an equation (of the form y = ax + b) to explain the relationship between the various predictors and the target variable - that in a nutshell is Linear regression.
# 
# Lets tinker with the equation a little more using our Graduate data example albeit only the GRE score and Chances of Admit.
# 
# The population for this data is effectively all the students in the world who would attempt for a Masters in the US by writing GRE and assuming GRE scores does have an impact on the admit, we can say,
# 
# <center><b>$ Y $ (Chances of Admit) $= A * X $(GRE Scores) $+ B + Error $ </b></center>
# 
# The reason for having the error term is because GRE scores are not the sole criteria for the admit chances and there might be other factors which our simple model has not accounted for or just plain random error which cannot be explained by any factor - and hence the Error term.
# 
# The other terms - A and B are the co-efficient of the Score variable and the intercept constant respectively. These are unknown to us. 
# 
# Now, in this specific case, we just have a handful (sample) of the data of students who are attempting for a US college Masters. And to model the data as a straight line relation between the GRE scores of our sample and the chances of Admit, we need to make good estimates of the population co-efficients (A and B).
# 
# <center><b>$ y(est) = A(est) * x + B(est) $</b></center>
# 
# These estimates of the co-efficients are arrived at by minimizing the sum of the square of the errors where the error is the difference between our prediction of Y and the actual Y. 
# 
# The same concept can be extended to multiple Linear regression where we attempt to model a relationship between more than one independent variable and the target variable. Instead of fitting a line in a two dimensional space, a hyperplane is fit on the n+1 dimensional space where n is the number of predictors.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler


# In[ ]:


graduate_data.columns


# In[ ]:


X = graduate_data.drop(['Chance of Admit ','Admit_quartiles'], axis = 1)
y = graduate_data['Chance of Admit ']


# ### Splitting the data into training and testing with a 3:1 split

# In[ ]:


## The random state value helps in selecting the same samples for the train test split - so that we can validate 
## results over multiple runs with the same train/ test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 111)


# ### Fitting the Linear Regression model

# In[ ]:


grad_model = LinearRegression()

##Model training using the training data
grad_model.fit(X_train, y_train)

## using the fitted model to predict on the test data
grad_prediction = grad_model.predict(X_test)

## Getting the co-efficients of the independent variables and the intercept
print("The model coefficients are: ",grad_model.coef_)
print("The model intercept is: %.2f"%grad_model.intercept_)

## Finding how much of the variation in the Target variable is explained by our model
print('Target variation explained by model: %.2f' % r2_score(y_test, grad_prediction))

## Finding the error in the predictions
print('Mean squared error: %.3f' % mean_squared_error(y_test, grad_prediction))


# ### Interpreting the Regression co-efficients
# 
# The co-efficient of an independent variable signifies the average change (increase or decrease depending on the sign of the co-efficient) in the dependent variable for an unit change in the independent variable keeping all other independent variables constant. <br> For example, in our model equation, the co-efficient of CGPA is 0.11 - for every unit increase in the CGPA, there is an average increase of 0.11 in the chances of admit. 

# ### Plotting the Errors
# 
# We interpreted the Regression co-efficients. But before we trust the regression co-efficients, we need to make a couple of simple checks. One is creating residual plots based on the prediction errors made by the model (also known as residuals) and the other is checking the p-values of the regression co-efficients. <br> <br>
# The residual plot is created with the predicted or fitted values in the x-axis and the errors (actual values of the target variable and the predictions made) in the y-axis. This plot lets us know if the errors are random over the entire range of the fitted values or if there is any pattern to it. Ok. But why is a pattern not a good sign of a good model? Let us go back to the equation that we created: <br> <br>
# 
# <center><b>$ Y $ (Chances of Admit) $=$ Co-efficients $*$ Independent Variables $+ Error $ </b></center>
# 
# The idea is to create a model which can account for almost all the variation in the dependent variable - except the random component. If there is a pattern in our residual plot, then it means some of the predictable variation in the target variable is yet to be accounted for by our model. For example, let us assume that we did not consider CGPA in our model - and let us also assume that for two students with the same high GRE scores, the balance is tilted in favour of the student with the higher CGPA - then our residual plot will have a funnel shape with the big mouth towards the right of the x-axis (towards the higher values of the fitted values).
# 
# In our case, we do have a funnel in the residual plot - but with the big mouth towards the origin. At this moment, I'm not too sure if
# - this is due to a missing variable or any other causes
# - how to fix this issue

# In[ ]:


## finding the residuals - the difference between the actual target variable values and the predictions
residuals = y_test - grad_prediction

plt.figure(figsize = (7,7))
plt.scatter(grad_prediction, residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual Plot", fontsize = 16)
plt.show()


# ### Checking if the predictors are significant
# 
# statsmodels provides a more detailed analysis of the regression output. We will decipher some of the important terms in this result and also check the significance of the predictor variables. <br>
# 
# - R-squared: Simply, put R_squared quantifies the amount of variation in the target variable that is explained by the model. A very intuitive and graphical understanding of R_squared can be found in the book Regression basics by Leo Kahane. Our model is able to explain around 82% of the variation in the Chances of Admit variable.
# 
# - F-statistic: This is the result of the hypothesis testing to see if it is meaningful to build a model with all the explanatory variables to explain the variation in the target variable or if it is only as good (or bad) as an intercept only model. 
# 
# - Prob(F-statistic): This gives the probability of getting the F-statistic due to random chance. If it is very low (below a set significance level), then we can conclude that there is a statistically significant difference between our model and the intercept only model.
# 
# Understanding the observations in the table:
# - the first column is the coefficient of that variable, 
# - the second column is the standard error - basically how many standard deviations away is the mean of the co-efficient as compared to the true co-efficient value.
# - t-statistic: Similar to the F-test - just that here the test is done one variable at a time with the intercept only model. 
# - Prob(t): The probability of getting the t-statistic value due to random chance. If it is low, then we can conclude that our variable in question has a significant relationship :-) with the dependent variable. In our results, except SOP and University rating, all the other variables are significant.
# - The last two columns are the confidence interval - the interval within which is the true value of the co-efficient. If the co-efficient estimate is outside this interval, then most likely that variable is not a significant predictor of the target variable.

# In[ ]:


import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train,X_train).fit()
model.summary()


# ### Redefining the model
# 
# We can try dropping the variables which are not significant and run the regression again.
# 
# <b>Results:</b> We observe that the R-squared is almost the same despite the reduced number of variables. With similar predictive power, it is always better to opt for the simpler model with the fewer predictors.

# In[ ]:


X = graduate_data.drop(['Chance of Admit ','Admit_quartiles', 'SOP', 'University Rating'], axis = 1)
y = graduate_data['Chance of Admit ']


# In[ ]:


## The random state value helps in selecting the same samples for the train test split - so that we can validate 
## results over multiple runs with the same train/ test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 111)

grad_model = LinearRegression()

##Model training using the training data
grad_model.fit(X_train, y_train)

## using the fitted model to predict on the test data
grad_prediction = grad_model.predict(X_test)

## Getting the co-efficients of the independent variables and the intercept
print("The model coefficients are: ",grad_model.coef_)
print("The model intercept is: %.2f"%grad_model.intercept_)

## Finding how much of the variation in the Target variable is explained by our model
print('Target variation explained by model: %.2f' % r2_score(y_test, grad_prediction))

## Finding the error in the predictions
print('Mean squared error: %.3f' % mean_squared_error(y_test, grad_prediction))


# In[ ]:


import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train,X_train).fit()
model.summary()


# ### Summary
# 
# A quick recap of the observations/ insights from this exercise:
# - The data indicates a pretty evident linear relationship between GRE/ TOEFL/ CGPA and the chances of Admit.
# - Few students with lesser GRE/ TOEFL score but good in the other areas have managed to increase their chances of admit. But anything less than 305/ 102 in GRE/ TOEFL will have a very bleak chance of an Admit.
# - There is a statistically significant difference between the means of the GRE scores within the top two Admit quartiles. This extends to the Research variable too despite it being the least correlated with the target variable.
# - After fitting the Linear Regression, SOP and University Rating turned out to be insignificant variables in predicting the variation in the target variable.

# #### References:
# - https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/linear_regression.pdf
# - statisticsbyjim.com/
# - http://scipy-lectures.org/packages/statistics/

# 
