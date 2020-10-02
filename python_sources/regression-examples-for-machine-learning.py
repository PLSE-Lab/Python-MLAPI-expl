#!/usr/bin/env python
# coding: utf-8

# This document includes below examples;
# 1. [**Linear Regression (LR)**](#1)
#    1. [Prediction](#11)
#    1. [R Square (LR)](#12)
# 1. [**Multiple Linear Regression**](#2)
#    1. [Prediction](#21)
#    1. [R Square (LR)](#22)
# 1. [**Polynomial Regression (PR)**](#3)
#    1. [Prediction](#31)
#    1. [R Square (LR)](#32)
# 1. [**Decision Tree Regression (DTR)**](#4)
#    1. [Prediction](#41)
#    1. [R Square (LR)](#42)
# 1. [**Random Forest Regression (RFR)**](#5)
#    1. [Prediction](#51)
#    1. [R Square (LR)](#52)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read the data
dframe = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")


# In[ ]:


dframe.head()


# In[ ]:


# Check whether there are empty rows or not.
dframe.info()


# In[ ]:


dframe.describe()


# In[ ]:


# Correlation 
# We excluded "Serial No" with data.iloc[:,1:]) )

dframe.iloc[:,1:].corr()


# In[ ]:


# Correlation map
f, axx = plt.subplots(figsize=(10,10))
sns.heatmap(dframe.iloc[:,1:].corr(), linewidths=0.5, cmap="Blues", annot=True,fmt=".1f", ax=axx)
plt.show()


# CGPA, GRE Score and TOEFL Scores are 3 most correlated features for the "Chance of Admit".
# 
# Let's drop all the duplicated values from the data frame.

# In[ ]:


# Drop the duplicated values of the Chance of Admit.
df= dframe.drop_duplicates(subset=["Chance of Admit "])
df.info()


# In[ ]:


df= df.drop_duplicates(subset="CGPA")
df= df.drop_duplicates(subset="GRE Score")
df= df.drop_duplicates(subset="TOEFL Score")
df.info()


# In[ ]:


df.describe()


# In[ ]:


# Correlation 
# We excluded "Serial No" with data.iloc[:,1:]) )

df.iloc[:,1:].corr()


# In[ ]:


# Correlation map
f, axx = plt.subplots(figsize=(10,10))
sns.heatmap(df.iloc[:,1:].corr(), linewidths=0.5, cmap="Blues", annot=True,fmt=".2f", ax=axx)
plt.show()


# According to the new data frame (with non-duplicated values);
# 
# TOEFL Scores, CGPA and GRE Score are 3 most correlated features for the "Chance of Admit".

# In[ ]:


df.columns


# In[ ]:


# Mean value of "Chance of Admit " is 0.677368.
# Output is on above; df.describe()

# Create a new column for High and Low.

df["Admit Level"] = ["Low" if each < 0.677368 else "High" for each in df["Chance of Admit "]]
df.head()


# In[ ]:


df.info()


# In[ ]:


# Vizualization
# CGPA, GRE Score and TOEFL Scores / Chance of Admit

import plotly.graph_objs as go

trace1 = go.Scatter(
                        x = df["Chance of Admit "],
                        y = df.CGPA,
                        mode = "markers",
                        name = "CGPA",
                        marker = dict(color="rgba(255, 100, 128, 0.8)"),
                        text = df["Admit Level"]
                        )
trace2 = go.Scatter(
                        x = df["Chance of Admit "],
                        y = df["GRE Score"],
                        mode = "markers",
                        name = "GRE Score",
                        marker = dict(color="rgba(80, 80, 80, 0.8)"),
                        text = df["Admit Level"]
                        )
trace3 = go.Scatter(
                        x = df["Chance of Admit "],
                        y = df["TOEFL Score"],
                        mode = "markers",
                        name = "TOEFL Score",
                        marker = dict(color="rgba(0, 128, 255, 0.8)"),
                        text = df["Admit Level"]
                        )
data = [trace1, trace2, trace3]
layout = dict(title="CGPA, GRE Score and TOEFL Scores v Chance of Admit",
             xaxis=dict(title="Chance of Admit", ticklen=5, zeroline=False),
             yaxis=dict(title="Values", ticklen=5, zeroline=False)
             )
fig = dict(data=data, layout=layout)
iplot(fig)


# <a id="1"></a> <br>
# 1. **Linear Regression (LR)**
# 
# y = b0 + b1*x

# In[ ]:


# Sklearn library
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()


# For the fit operation, we need to use numpy arrays on the x and y axixes, so we will use "df.ColumnName.values". But, its type will be "(500,)" so we will reshape it. The reason we write -1 is because we may don't know the size, we only need to set the second value to 1.
# 
# The most correlated feature with "Chance of Admit" is "CGPA".

# In[ ]:


print(df.CGPA.values.shape)
print(df["Chance of Admit "].values.shape)

# Reshape
x = df.CGPA.values.reshape(-1,1)
y = df["Chance of Admit "].values.reshape(-1,1)
print("After resphape:\nX:", x.shape)
print("Y:", y.shape)


# Now, we can use above x&y axises on the fit operation of the linear regression model.

# In[ ]:


linear_reg.fit(x,y)


# In[ ]:


# Formula
# y = b0 + b1*x

b0 = linear_reg.intercept_
print("b0:", b0) # the spot where the linear line cuts the y-axis

b1 = linear_reg.coef_
print("b1:", b1) # slope

print("Linear Regression Formula:", "y = {0} + {1}*x".format(b0,b1))


# <a id="11"></a> <br>
# **1.1. Prediction**
# 
# We will predict the values according to linear_reg model.

# In[ ]:


x[0:5]


# In[ ]:


# CGPA-9.65 = Chance of Admit -0.92
df[df.CGPA == 9.65].loc[:,"Chance of Admit "]


# In[ ]:


linear_reg.predict([[9.8]])


# In[ ]:


print(min(x), max(x))


# In[ ]:


# CGPA values that will be predicted.

# Chance of Admit (predicted values)
y_head = linear_reg.predict(x)

plt.figure(figsize=(10,10))
plt.scatter(x,y, alpha=0.7)  # Real values (blue)
plt.plot(x,y_head, color="red") # Predicted values for numpay array (arr).
plt.show()


# <a id="12"></a> <br>
# **1.2. R Square (LR)**
# 
# We can evaluate the linear regression model performance with R Square.
# * y: Chance of Admit values
# * y_head: predicted Chance of Admit value
# 
# First, we must be sure that y and y_head values are using the same number of samples. If not, we will get an error like this:
# 
# ValueError: Found input variables with inconsistent numbers of samples: [500, 312]

# In[ ]:


# Same shapes
print(y.shape, y_head.shape)


# In[ ]:


# R Square Library
from sklearn.metrics import r2_score
# y: Chance of Admit values
# y_head: predicted Chance of Admit values with LR
print("r_square score: ", r2_score(y, y_head))


# Success ratio is around **% 70** for the LR prediction.

# <a id="2"></a> <br>
# 2. **Multiple Linear Regression**
# 
# y = b0 + b1x1 + b2x2 + ... bnxn

# In[ ]:


# Sklearn library
# we already imported -- > from sklearn.linear_model import LinearRegression


# In[ ]:


# Define and reshape the variables

x1 = df.loc[:, ["CGPA", "GRE Score", "TOEFL Score"]]
y1 = df["Chance of Admit "].values.reshape(-1,1)


# In[ ]:


# Creat the model and fit the x&y values.
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x1,y1)


# In[ ]:


# Formula
# y = b0 + b1*x1 + b2*x2 + ... bn*xn
b0 = multiple_linear_regression.intercept_
b1,b2,b3 = zip(*multiple_linear_regression.coef_) 
print("b1:", b1, "b2:", b2, "b3:", b3)
print("b0:", multiple_linear_regression.intercept_)
print("b1, b2:", multiple_linear_regression.coef_)
print("Multiple Linear Regression Formula:", "y = {0} + {1}*x1 + {2}*x2 + {3}*x3".format(b0,b1,b2,b3))


# <a id="21"></a> <br>
# **2.1. Prediction**

# In[ ]:


print("CGPA:", min(x1["CGPA"]),"-", max(x1["CGPA"]))
print("GRE Score:", min(x1["GRE Score"]),"-", max(x1["GRE Score"]))
print("TOEFL Score:", min(x1["TOEFL Score"]), "-", max(x1["TOEFL Score"]))
plt.figure(figsize=(10,5))
plt.scatter(df["Chance of Admit "], df.CGPA, color="blue", label="CGPA")
plt.scatter(df["Chance of Admit "], df["GRE Score"], color="green", label="GRE Score")
plt.scatter(df["Chance of Admit "], df["TOEFL Score"], color="orange", label="TOEFL Score")
plt.legend()
plt.show()


# In[ ]:


# 1st: CGPA: 6.8 - 9.92
# 2nd: GRE Score: 290 - 340
# 3rd: TOEFL Score: 92 - 120
# Prediction: Chance of Admit

print("Values= np.array( [[6,280,90]])) Prediction =",
      multiple_linear_regression.predict(np.array( [[6,280,90]])))

print("Values= np.array( [[8,300,100]])) Prediction =",
      multiple_linear_regression.predict(np.array( [[8,300,100]])))

print("Values= np.array( [[10,350,130]])) Prediction =",
      multiple_linear_regression.predict(np.array( [[10,350,130]])))


# In[ ]:


x1.head()


# y1_head keeps the prediction values of x1 which has CGPA, GRE Score and	TOEFL Score values.  

# In[ ]:


y1_head = multiple_linear_regression.predict(x1)
y1_head[:5]


# In[ ]:


plt.figure(figsize=(10,20))

plt.scatter(y, x1.iloc[:,0], color="blue", alpha=0.7) # CGPA
plt.scatter(y1_head, x1.iloc[:,0], color="black", alpha=0.7)

plt.scatter(y, x1.iloc[:,1], color="green", alpha=0.7) # GRE Score
plt.scatter(y1_head, x1.iloc[:,1], color="black", alpha=0.7)

plt.scatter(y, x1.iloc[:,2],color="orange", alpha=0.7) # TOEFL  Score
plt.scatter(y1_head, x1.iloc[:,2], color="black", alpha=0.7)
plt.show()


# Black values shows the predicted values, other colors are the real values. As you can see, the predicted values are converging to the real values.

# <a id="22"></a> <br>
# **2.2. R Square**

# In[ ]:


# R Square Library

# Imported on previous sections
# from sklearn.metrics import r2_score

# y: Chance of Admit values
# y1_head: predicted Chance of Admit values with MLR
print("r_square score: ", r2_score(y,y1_head))


# Success ratio is **% 75** for the  MLR prediction.

# <a id="3"></a> <br>
# 3. **Polynomial Regression (PR)**

# y = b0 + b1x+ b2x^2 + b3x^3 + ... + bnx^n
# 

# In[ ]:


# Sklearn library 
from sklearn.preprocessing import PolynomialFeatures

# We have chose the second degree equation with (degree=2)
polynomial_regression = PolynomialFeatures(degree=2)
# y = b0 + b1*x + b2*x^2
x = df["TOEFL Score"].values.reshape(-1,1)
# y = df["Chance of Admit "].values.reshape(-1,1)
x_ploynominal = polynomial_regression.fit_transform(x)

linear_regression_poly = LinearRegression()
linear_regression_poly.fit(x_ploynominal, y)


# <a id="31"></a> <br>
# **3.1. Prediction**

# In[ ]:


# Linear Regression (LR) section: x = df.CGPA.values.reshape(-1,1)
# Linear Regression (LR) section: y = df["Chance of Admit "].values.reshape(-1,1)
print("x:\n", x[:5], "\ny:\n",y[:5])


# In[ ]:


# Predicted values
y_head_poly = linear_regression_poly.predict(x_ploynominal)
y_head_poly[:5]


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(x, y, color="blue", alpha=0.7) # CGPA
plt.scatter(x, y_head_poly, label="poly (degree=2)", color="black") # predicted Chance of Admit
plt.xlabel("TOEFL Score")
plt.ylabel("chance")
plt.legend()
plt.show()


# 2nd degre equationd didn't give a proper model. But, we can modify the degree of the equation to converge to real values.

# In[ ]:


# y = b0 + b1*x + b2*x^2 + ..... b10*x^10
polynomial_regression7 = PolynomialFeatures(degree=7)

# x = df.CGPA.values.reshape(-1,1)
x_ploynominal_7 = polynomial_regression7.fit_transform(x)

linear_regression_poly_7 = LinearRegression()
linear_regression_poly_7.fit(x_ploynominal_7, y)

# Predicted values
y_head_poly_7 = linear_regression_poly_7.predict(x_ploynominal_7)


# In[ ]:


# y = b0 + b1*x + b2*x^2 + ..... b30*x^30
polynomial_regression30 = PolynomialFeatures(degree=30)

# x = df.CGPA.values.reshape(-1,1)
x_ploynominal_30 = polynomial_regression30.fit_transform(x)

linear_regression_poly_30 = LinearRegression()
linear_regression_poly_30.fit(x_ploynominal_30, y)

# Predicted values
y_head_poly_30 = linear_regression_poly_30.predict(x_ploynominal_30)


# Compare the predicted values of different equations.

# In[ ]:


plt.figure(figsize=(12,12))
plt.scatter(x, y, color="blue", alpha=0.7) # TOEFL Score
plt.scatter(x, y_head_poly, label="poly (degree=2)", color="black", alpha="0.7") # predicted Chance of Admit
plt.scatter(x, y_head_poly_7, label="poly (degree=7)", color="red", alpha="0.7") # predicted Chance of Admit
plt.scatter(x, y_head_poly_30, label="poly (degree=30)", color="green", alpha="0.7") # predicted Chance of Admit
plt.xlabel("TOEFL Score")
plt.ylabel("chance")
plt.legend()
plt.show()


# We can see that red predicted values (degree=7) are more convergent on the bottom-left of the graph and they are similar with the green predicted values (degre=30) in the middle and upper-right of the graph.
# 
# The most proper degree may differ between different datas. We don't always have to increase it to get more accurate prediction.

# <a id="32"></a> <br>
# **3.2. R Square**

# In[ ]:


# R Square Library

# Imported on previous sections
# from sklearn.metrics import r2_score

print("r_square score for degree=2: ", r2_score(y, y_head_poly))
print("r_square score for degree=7: ", r2_score(y, y_head_poly_7))
print("r_square score for degree=30: ", r2_score(y, y_head_poly_30))


# Success ratio is **%76** for degree=2 and it is **%78.84** for degree=7 for PR.

# <a id="4"></a> <br>
# 4. **Decision Tree Regression (DTR)**

# "Decision Tree Regression" method divides the areas between values based on the conditions, assing the average value of the values for each area which is called "leaf".
# 
# For example let's take below tree as an example. We can divide the conditions as [0,30), (30,40), (40,50) and (50,100].
# 
# If x = 51 or 100, it means y is 100. It shows that y values for all "x>50" means 100 for the model. 
# If x = 31 or 39, it means y is 35. (Y values are chosen arbitrary, it doesnt have to be linear propotional.)
# 
# 
# x=[0, 100]
# 
#                           x1 > 50
#                 yes                no
#                y=100              x>30
#                               yes       no
#                               x<40      y=25
#                             yes   no
#                             y=35  y=45

# In[ ]:


df.head()


# In[ ]:


# Decision Tree Library
from sklearn.tree import DecisionTreeRegressor

x = df["TOEFL Score"].values.reshape(-1,1)
y = df["Chance of Admit "].values.reshape(-1,1)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)


# In[ ]:


plt.scatter(df["TOEFL Score"] , df["Chance of Admit "],alpha=0.8)
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.show()


# <a id="41"></a> <br>
# **4.1. Prediction**

# In[ ]:


y_head_dtr = tree_reg.predict(x)


# In[ ]:


plt.scatter(x, y, color="blue", alpha = 0.7)
plt.scatter(x, y_head_dtr, color="black", alpha = 0.4)
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.show()


# Black dots are predicted values and they overlap the real values. Because our prediction values (x) are same with the real values. So, we need some other range to predict according to real x values.

# In[ ]:


# Let's make a new array in the range of TOEFL Score values increased by 0.01
x001= np.arange(min(x), max(x), 0.01).reshape(-1,1) # (start, end, increase value)
y_head001dtr = tree_reg.predict(x001)


# In[ ]:


len(np.unique(y_head001dtr))

# 19 unique values for all values


# In[ ]:


plt.figure(figsize=(20,10))
plt.scatter(x,y, color="blue", s=100, label="real TOEFL Score") # real y (Chance of Admit) values
plt.scatter(x001,y_head001dtr, color="red", alpha = 0.7, label="predicted TOEFL Score") # to see the predicted values one by one
plt.plot(x001,y_head001dtr, color="black", alpha = 0.7)  # to see the average values for each leaf.
plt.legend()
plt.show()


# Black plot shows the predicted values, red points show them one by one. As you can see, TOEFL Score values (x) divided between leaves (ranges) and each leaf have an average value as the predicted value. So, we see a constant line for each leaf.

# <a id="42"></a> <br>
# **4.2. R Square**

# In[ ]:


# Same shapes, y and y_head_dtr
print(y.shape, y_head_dtr.shape, y_head001dtr.shape)


# In[ ]:


from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_head_dtr))


# Success ratio is **%100** for DTR.

# **sklearn.tree.DecisionTreeRegressor:**
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# 
# **score(self, X, y, sample_weight=None)**
# 
# Returns the coefficient of determination R^2 of the prediction.
# 
# The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
# 
# Parameters:	
# X : array-like, shape = (n_samples, n_features)
# Test samples. For some estimators this may be a precomputed kernel matrix instead, shape = (n_samples, n_samples_fitted], where n_samples_fitted is the number of samples used in the fitting for the estimator.
# 
# y : array-like, shape = (n_samples) or (n_samples, n_outputs)
# True values for X.
# 
# sample_weight : array-like, shape = [n_samples], optional
# Sample weights.
# 
# Returns:	
# score : float
# R^2 of self.predict(X) wrt. y.
# 
# Notes
# 
# The R2 score used when calling score on a regressor will use multioutput='uniform_average' from version 0.23 to keep consistent with metrics.r2_score. This will influence the score method of all the multioutput regressors (except for multioutput.MultiOutputRegressor). To specify the default value manually and avoid the warning, please either call metrics.r2_score directly or make a custom scorer with metrics.make_scorer (the built-in scorer 'r2' uses multioutput='uniform_average').

# In[ ]:


from sklearn.model_selection import cross_val_score
#cross_val_score(tree_reg, boston.data, boston.target, cv=10)
print(tree_reg.score(x001, y_head001dtr))
print(tree_reg.score(x, y))


# In[ ]:


from sklearn.metrics import r2_score
print("r_score: ", r2_score(y,y_head_dtr))

from sklearn.model_selection import cross_val_score
print(tree_reg.score(x001, y_head001dtr))
print(tree_reg.score(x, y))


# <a id="5"></a> <br>
# 5. **Random Forest Regression (RFR)**

# Random forest regression combined by  multiple regression. 
# 
# It chooses n examples, divides the data to sub datas and uses multiple trees.
# 
#                      data
#                        |
#                        |
#                     n sample
#                        |
#                        |
#                     sub_data
#          tree1   tree2  tree3 .... tree n
#          ________________________________
#         |           average               |
#          ________________________________
#                      result
#                      
#             
#                  
#                  
# 
# RandomForestRegressor(**n_estimators** = 100, **random_state** = 42)
# 
# This means we will use 100 tree (DTR) and 42 sample. The algorithm chooses the n samples randomly. We gave a constant number for the random state, therefore the algorithm will select the same 42 examples on the next time.

# In[ ]:


plt.scatter(x, y, color="blue", alpha = 0.7)
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.show()


# <a id="51"></a> <br>
# **5.1. Prediction**

# In[ ]:


x = df["TOEFL Score"].values.reshape(-1,1)
y = df["Chance of Admit "].values.reshape(-1,1)

print(min(x), max(x))
print(min(y), max(y))


# In[ ]:


# Random Forest Regression Library

from sklearn.ensemble import RandomForestRegressor
 
random_forest_reg = RandomForestRegressor(n_estimators = 100, random_state = 42)
# n_estimators = 100 --> Tree number
# random_state = 42  --> Sample number
random_forest_reg.fit(x,y)

print(random_forest_reg.predict([[98]]))


# In[ ]:


# New prediction examples with (Start, End, Increase)
x001 = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head001rf = random_forest_reg.predict(x001)

print(min(x001), max(x001))
print(min(y_head001rf), max(y_head001rf))


# In[ ]:


len(np.unique(y_head001rf))


# 46 unique values for all values


# In[ ]:


plt.figure(figsize=(20,10))
plt.scatter(x,y, color="blue", label="real TOEFL Score")
plt.scatter(x001,y_head001rf, color="red", label="predicted TOEFL Score")
plt.plot(x001,y_head001rf, color="black")
plt.legend()
plt.xlabel("TOEFL Score")
plt.ylabel("Chance of Admit")
plt.show()


# Random forest regression (RFR) uses more trees (100), it gave more accurate predicted values than Decision Tree Regression (DTR).

# <a id="52"></a> <br>
# **5.2. R Square**

# In[ ]:


from sklearn.model_selection import cross_val_score

print(tree_reg.score(x001, y_head001rf))
print(tree_reg.score(x, y))


# In[ ]:


from sklearn.metrics import r2_score

y_headrf = random_forest_reg.predict(x)
print("r_score: ", r2_score(y,y_headrf))

from sklearn.model_selection import cross_val_score
print(tree_reg.score(x001, y_head001rf))
print(tree_reg.score(x, y))

