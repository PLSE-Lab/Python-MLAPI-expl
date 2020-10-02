#!/usr/bin/env python
# coding: utf-8

# # Prediction of back pain using Logistic Regression, Python
# 
# The data comprises of 13 columns and 310 observations. 12 columns are numerical attributes of the spine/ back. The last column is the Status of the patient - Abnormal indicates presence of Back pain and Normal indicates no back pain. The intent is to predict the Status based on the 12 variables. 
# 
# In this kernel, I explore the different variables and use Logistic Regression to predict and understand the causal effects of the different predictors.

# In[ ]:


#Loading the necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# In[ ]:


back_data = pd.read_csv("../input/Dataset_spine.csv")


# ## Unnamed column names
# 
#  - There is an additional column with notes about the names of each of the individual attributes. This is removed.
#  - The columns are not named in the dataset. Adding the column names to each of the corresponding columns. 

# In[ ]:


del back_data['Unnamed: 13']
back_data.columns = ['pelvic_incidence','pelvic tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','Status']


# ## Initial Data Exploration
#  - Checking if there are any missing values in the dataset. Turns out there aren't any.
#  - Checking the frequency of the different values of the Status column. Abnormal is 210 of the 310 observations. This means that it is not really an unbalanced dataset.

# In[ ]:


## Understanding the structure of the data variables
back_data.info()

##Checking for missing values. There are no missing values
print(back_data.isnull().sum())

## split of the Status column between the two levels Abnormal and Normal
print(back_data.Status.describe())


# ## Multi Collinearity check
#  - Checking if the individual columns are correlated with each other. In which case, they might end up having the same predictive power or explaining the same variation in the dependent variable. 
#  - The correlation matrix/ plot is a good way to establish multi collinearity between the dependent variables. Anything closer to +- 1 indicates high correlation between those two predictor variables.
#  - We can observe from the plot that pelvic_incidence is highly correlated with pelvic tilt, sacral slope, degree spondylolisthesis and lumbar lordosis angle.

# In[ ]:


corr_back = back_data.corr()
# Generate a mask for the upper right triangle of the square - one half is enough to convey the correlation 
## between the predictors
mask = np.zeros_like(corr_back, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate the correlation plot 
sns.heatmap(corr_back, mask=mask, center=0, square=True, linewidths=.5)

plt.show()


# In[ ]:


# Seeing the correlation values
corr_back


# ## Bi-variate analysis - Relation of each predictor with Target variable
# 
# The status column has two values - 'Abnormal' and 'Normal'. We would like to explore how each of the 12 predictor variables vary with respect to the Status value. We would be more interested in those predictor variables which have a noticeable difference in their values corresponding to 'Normal' and 'Abnormal'. 
# 
# We calculate the mean/ median of values corresponding to Normal and Abnormal for each of the predictor variables. Some observations:
# 
#  - The first six variables have noticeable difference in the means corresponding to Normal and Abnormal.
#  - The mean corresponding to Abnormal for degree_spondylolisthesis is quite higher than the corresponding median. Could be due to outliers.
#  - pelvic_radius has lower values for Abnormal as compared to Normal.

# In[ ]:


back_data.groupby('Status').mean()


# In[ ]:


back_data.groupby('Status').median()


# ### Box Plots
# To visualise the above data exploration, we make use of Box Plots as we are comparing Categorical and continuous variables.
# 
# A couple of observations which jump out:
#  - For the variable degree_spondilolisthesis, 'Normal' status clearly has a much lower range of values as compared to 'Abnormal'. Also shows the presence of a distant outlier. Not removing the outlier - as without domain knowledge, it would be hard to interpret whether it is an incorrect or a rare value.
#  - For the variable 'Pelvic Radius' while 'Abnormal' has a much higher range of values, the median value of 'Abnormal' is lower than the median values of 'Normal'.

# In[ ]:


## Generating 3*4 matrix of box plots
fig, axes = plt.subplots(3, 4, figsize = (15,15))
axes = axes.flatten()

for i in range(0,len(back_data.columns)-1):
    sns.boxplot(x="Status", y=back_data.iloc[:,i], data=back_data, orient='v', ax=axes[i])

plt.tight_layout()
plt.show()


# ## Final processing
# 
#  - For modelling purpose, we map all the predictor variables to a array X and the target variable to an array Y. 
#  - The class labels 'Abnormal' and 'Normal' are numerically encoded to 1 and 0. While this is not necessary as the sklearn module can handle it internally, it is convenient for graphing the Receiver Operating curve (if required).
#  - The variables are subjected to Standardization (mean zero and unit variance) before being fed to the model.

# In[ ]:


back_data.loc[back_data.Status=='Abnormal','Status'] = 1
back_data.loc[back_data.Status=='Normal','Status'] = 0


# In[ ]:


X = back_data.loc[:, back_data.columns != "Status"]
y = back_data.loc[:, back_data.columns == "Status"]


# ## Modelling and Feature Interpretation
# 
#  - Implementing a logistic regression classifier with a train test split in a 70:30 ratio. 
#  - The fitted model when applied on the test data returns an accuracy of 81.7%. 

# In[ ]:


def data_preprocess(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.3, random_state=0)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

    scaler.fit(X_train)

    # Now apply the transformations to the data:
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    return(train_scaled, test_scaled, y_train, y_test)


# In[ ]:


def logistic_regression(x,y):
    logreg = LogisticRegression().fit(x, y)
    return(logreg)


# In[ ]:


X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X,y)

logreg_result = logistic_regression(X_train_scaled, y_train)

print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))


# ## To explain or to predict
# 
#  - While the previous model implemented was good in predicting the Target variable on a test set, we did not interpret anything about the individual features - which variable(s) influence the Target variable more. 
#  - Since this is a medical dataset, there could be a need for explaining the effect of individual variables on the response variable. Hence it would be a good idea to explore the model co-efficients of the predictor variables to see how well each of them influence the response. 
#  - Let us attempt this in the next section.

# In[ ]:


logit_model = sm.Logit(y_train, X_train_scaled)
result = logit_model.fit()
print(result.summary2())


# ## Understanding the model result summary
# 
#  - The above model did not converge because some variables were highly correlated with each other and this would have led to the correlation/ covariance matrix to be singular. 
#  - A matrix can become singular if any rows(columns) can be expressed as a linear combination of any other rows (columns). 
#  - In fact, it was very intersting to note that in our data, the Pelvic Incidence column values are an exact sum of Pelvic Tilt and Sacral Slope. So that explains.
#  - Also in our statistical test results, the Standard error values are very high and p-value is 1 for these three variables. Hence we will remove them and re run the model. 

# In[ ]:


#Removing the highly correlated variables which also had high standard error
cols_to_include = [cols for cols in X.columns if cols not in ['pelvic_incidence', 'pelvic tilt','sacral_slope']]
X = back_data[cols_to_include]


# In[ ]:


X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X,y)

logreg_result = logistic_regression(X_train_scaled, y_train)

print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))


# The test results indicate that the predictive power has gone down after removing the highly correlated variables. Let us look at the statistical summary below.

# In[ ]:


# to get the statistical summary of the new model
logit_model=sm.Logit(y_train,X_train_scaled)
result=logit_model.fit()
print(result.summary2())


# ## Model convergence
# - The model has now converged after removing the highly correlated variables.
# - There are a few predictors with p-values less than 0.05 (assuming a 95% confidence level). Let us consider only those predictors and re run the model

# In[ ]:


# considering only the variables which have p-value less than 0.05
X_trim_1 = X.loc[:,['lumbar_lordosis_angle','pelvic_radius','degree_spondylolisthesis']]


# ## Final Model Selection
# 
# - The test result below indicate that there is a marginal increase in predictive power - the accuracy on the test set has increased from 74% to 77%. 
# - This can be our final chosen model (even though Variable 1 - lumbar_lordosis_angle has a p-value marginally greater than 0.05). 
# 

# In[ ]:


X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X_trim_1,y)

logreg_result = logistic_regression(X_train_scaled, y_train)

print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))


# In[ ]:


logit_model=sm.Logit(y_train,X_train_scaled)
result=logit_model.fit()
print(result.summary2())


# ## Interpreting the causative power of the predictors
# 
# - Now that we have decided on our final model, let us interpret the co-efficient estimates of the three predictor variables from the above statistical summary. 
#     - Both lumbar_lordosis_angle and pelvic_radius have a negative co-efficient indicating that for every unit increase in the values of these variables, the log of odds of the Status being Abnormal decreases. 
#     - degree_spondylolisthesis has a positive relationship with Abnormal status i.e for every unit increase in the value of degree_spondylolisthesis the log of odds of the Status being Abnormal increases.

# In[ ]:


# assigning the model predicted values to y_pred
y_pred = logreg_result.predict(X_test_scaled)

# assigning the string Normal and Abnormal to the 0 and 1 values respectively. This is useful in plotting 
# the confusion matrix
y_pred_string = y_pred.astype(str)
y_pred_string[np.where(y_pred_string == '0')] = 'Normal'
y_pred_string[np.where(y_pred_string == '1')] = 'Abnormal'

y_test_string = y_test.astype(str)
y_test_string[np.where(y_test_string == '0')] = 'Normal'
y_test_string[np.where(y_test_string == '1')] = 'Abnormal'


# ## Sensitivity and Specificity
# 
# From the confusion matrix, we can calculate the Sensitivity True Positive/ (True positive + False Negative) and Specificity - True Negative / (True Negative + False Positive). 
# Sensitivity = 74.6% and Specificity = 76.6%
# 
# I am not quite sure in this specific problem context, which one should be given more importance (or should have a higher value). 

# In[ ]:


from sklearn.metrics import confusion_matrix
ax= plt.subplot()
labels = ['Abnormal','Normal']
cm = confusion_matrix(y_test_string, y_pred_string, labels)
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Abnormal', 'Normal']); ax.yaxis.set_ticklabels(['Abnormal', 'Normal']);
plt.show()


# ### Thanks for coming along all the way. Would love to hear inputs for improvement. Thanks.
