#!/usr/bin/env python
# coding: utf-8

# <h1><center> Fish Weight Prediction </center></h1>

# <h2> Table of contents </h2>
# Fist of all we need to understand the dataset a bit. </br>
# </br>
# 
# *   Species: species name of fish
# *   Weight: weight of fish in Gram g
# *   Length1: vertical length in cm
# *   Length2: diagonal length in cm
# *   Length3: cross length in cm
# *   Height: height in cm
# *   Width: diagonal width in cm
# 
# </br>
# 
# We will use the following plan to make sure the data are ready to be used for a machine learning implementation:
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# <ul>
#     <li><a href="#data_wrangling">Data Wrangling</a></li>
#     <li><a href="#identify_handle_missing_values">Identify and handle missing values</a>
#         <ul>
#             <li><a href="#missing_values">Identify missing values</a></li>
#         </ul>
#     </li>
#     <li><a href="#group_pivot">Grouping and Pivoting</a></li>
#     <li><a href="#correlation_causation">Correlation and Causation - Pearson and Pvalue</a></li>
#     <li><a href="#summary">Summary of the Data Exploration</a></li>
#     <li><a href="#model_development">Model development</a>
#         <ul>
#             <li><a href="#slr">Linear Regression and Multiple Linear Regression</a></li>
#             <li><a href="#poly">Pipeline and Polynomial Regression</a></li>
#         </ul>
#     </li>
#     <li><a href="#conclusions">Conclusions</a></li></ul>
# </br>
# <hr>

# In[ ]:


# Import the main libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2 id="data_wrangling"> Data Wrangling </h2>

# In[ ]:


# Import the dataset and print the first 5 rows
path = '../input/fish-market/Fish.csv'
df = pd.read_csv(path)
df.head()


# <hr>
# <h3 id="missing_values"> Identify Missing Values </h3>

# In[ ]:


# Let's count the null values in the dataset
df_null = df.isnull()
df_null.head()


# In[ ]:


# Let's print the missing values for each column name
for column in df_null.columns.to_list():
  print(column)
  print(df_null[column].value_counts())
  print('')


# Apparently we do not have Null values in the dataframe but some of the missing value could have a different character such as a question mark. Let's print the data types of each column.

# In[ ]:


df.shape


# In[ ]:


df.dtypes


# As we can see all the columns are correctly formatted. Let's print some statistics about the dataset.

# In[ ]:


df.describe()


# As we can see, the minimum value of the column weight is equal to 0. This means we have missing values into the dependent variable. Let's print how many rows have the weight equal to 0.

# In[ ]:


# Identify the rows where the Weight is missing
df.loc[df['Weight']==0]


# In[ ]:


# Drop the row where weight=0
df = df[df['Weight'] != 0]
print(df.shape)
df.head()


# Let's see how many different species we have in the dataset.

# In[ ]:


df.describe(include='object')


# As we can see there are 7 different species of fish in the dataset and the most frequent is the 'Perch'. </br>
# I want to count how many records I have for each species.

# In[ ]:


df['Species'].value_counts()


# I want to plot a bar chart to graphically show it.

# In[ ]:


# Define the x_labels
species = df['Species'].unique()

# Define the bar chart
plt.figure(figsize=(8,6))
plt.bar(species, df['Species'].value_counts(), color='G')

# Graphics
plt.xlabel('Species', fontsize=12, color='W')
plt.ylabel('Number of records', fontsize=12, color ='W')
plt.title('Barplot of number of species in the dataset', fontsize=16, color ='W')


# <h2 id="group_pivot"> Grouping and Pivoting </h2>

# In[ ]:


# I want to identify the average weight, length1, length2, lenght3, height and width grouped by species
df_s_group = df.groupby('Species').mean()
df_s_group


# In[ ]:


sp_list = df['Species'].unique()
for sp in sp_list:
  print(sp)
  print(df[df['Species'] == sp].describe())
  print('')


# As highlighted in the tables above, each species is very different from another.

# In[ ]:


sns.pairplot(df, kind='scatter', hue='Species')


# In[ ]:


sp_list = df['Species'].unique()
for sp in sp_list:
  print(sp)
  print(df[df['Species'] == sp].corr())
  print('')


# In[ ]:


df.corr()


# As we can see, is better to differentiate the model based on the "Species" in order to maximize the correlation between the independent variables and the dependent variable. It's important to notice that the relationship between the independent variable Height and the dependent variable Weight are highly correlated if you consider the species once per time. This doesn't happend when we consider the entire dataset.

# In[ ]:


df_Perch = df[df['Species'] == 'Perch']
sns.pairplot(df_Perch, kind='scatter')


# <h3> Note: </h3>
# It is probabily a good idea to build 2 different models. The first one using all the features we have, the second one performing a dimensionality reduction using the principal component analysis and saving probably just one of the column of 'Length1', 'Length2' and 'Length3'.
# 
# 
# ---
# 
# For semplicity, in this case we will fit models taking in consideration the entire dataset as generalization.

# <h2 id="correlation_causation">Correlation and Causation - Pearson and Pvalue</h2>

# <p><b>Correlation</b>: a measure of the extent of interdependence between variables.</p>
# 
# <p><b>Causation</b>: the relationship between cause and effect between two variables.</p>
# 
# <p>It is important to know the difference between these two and that correlation does not imply causation. Determining correlation is much simpler  the determining causation as causation may require independent experimentation.</p>

# <p3>Pearson Correlation</p>
# <p>The Pearson Correlation measures the linear dependence between two variables X and Y.</p>
# <p>The resulting coefficient is a value between -1 and 1 inclusive, where:</p>
# <ul>
#     <li><b>1</b>: Total positive linear correlation.</li>
#     <li><b>0</b>: No linear correlation, the two variables most likely do not affect each other.</li>
#     <li><b>-1</b>: Total negative linear correlation.</li>
# </ul>

# <b>P-value</b>: 
# <p>What is this P-value? The P-value is the probability value that the correlation between these two variables is statistically significant. Normally, we choose a significance level of 0.05, which means that we are 95% confident that the correlation between the variables is significant.</p>
# 
# By convention, when the
# <ul>
#     <li>p-value is $<$ 0.001: we say there is strong evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.05: there is moderate evidence that the correlation is significant.</li>
#     <li>the p-value is $<$ 0.1: there is weak evidence that the correlation is significant.</li>
#     <li>the p-value is $>$ 0.1: there is no evidence that the correlation is significant.</li>
# </ul>

# In[ ]:


# Import from scipy library the stats module
from scipy import stats


# We want to identify when the correlation between the dependent variable "Weight" and the indipendent variables is statistically significant.
# We can do it using a loop.

# In[ ]:


col_list = df.columns.to_list()[2:]
Y = df['Weight']
for x_pearson in col_list:
  pearson_coef, p_value = stats.pearsonr(df[x_pearson], Y)
  print(x_pearson)
  print('The Pearson Correlation Coefficient is ', pearson_coef, ' and the P-value is ', p_value)
  print('')


# <h2 id="summary"> Summary of the data exploration</h2>
# All the columns are well related with the dependent variable. We can use all of them or perform a PCA to reduce the number of independent variables.
# 

# <h1 id="model_development">Model Development</h1>

# <p>In this section, we will develop several models that will predict the weight of the fish using the variables or features. This is just an estimate but should give us an objective idea of how much the fish should weight.</p>

# <p>A Model will help us understand the exact relationship between different variables and how these variables are used to predict the result.</p>

# <h2 id="slr">Linear Regression and Multiple Linear Regression</h2>

# <h3>Linear Regression</h3>

# 
# <p>One example of a Data  Model that we will be using is</p>
# <b>Simple Linear Regression</b>.
# 
# <br>
# <p>Simple Linear Regression is a method to help us understand the relationship between two variables:</p>
# <ul>
#     <li>The predictor/independent variable (X)</li>
#     <li>The response/dependent variable (that we want to predict)(Y)</li>
# </ul>
# 
# <p>The result of Linear Regression is a <b>linear function</b> that predicts the response (dependent) variable as a function of the predictor (independent) variable.</p>
# 
# 

# $$
#  Y: Response \ Variable\\
#  X: Predictor \ Variables
# $$
# 

#  <b>Linear function:</b>
# $$
# Yhat = a + b  X
# $$

# <ul>
#     <li>a refers to the <b>intercept</b> of the regression line0, in other words: the value of Y when X is 0</li>
#     <li>b refers to the <b>slope</b> of the regression line, in other words: the value with which Y changes when X increases by 1 unit</li>
# </ul>

# In[ ]:


# Import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression


# In[ ]:


# Let's define a new LinearRegression model
lm = LinearRegression()
lm


# In[ ]:


# Let's identify the X variable and the Y variable. Considering the Pearson test, the best variable to use to  develop a Simple Linear Regression is 'Length3'
X = df[['Length3']]
Y = df[['Weight']]

lm.fit(X,Y)


# In[ ]:


# I want to print the intercept and the coefficient of the Linear Regression
print('The coefficient is: ', lm.coef_)
print('The intercept is: ', lm.intercept_)


# In[ ]:


# Use seaborn to plot the Linear Regression model
sns.regplot(X,Y)


# <h3> Is this a good approssimation? Is this a good model? </h3>
# In order to evaluate the model and to compare it with others, is important to have some metrics that summarize it.
# These metrics can be the R^2 and the Mean Squared Error.
# </br>
# The R^2 tell us how well the liner regression approssimate the data

# In[ ]:


# Import the metrics I need
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:


# Let's plot the residual
sns.residplot(X,Y)


# We can notice that the residual are not randomply distributed around the mean. This means that even if a Simple Linear Regression is a good approssimation of the distribution, there are other models that better rappresent the data.

# In[ ]:


Yhat_lm = lm.predict(X)
Yhat_lm[0:4]


# In[ ]:


mse_lm = mean_squared_error(Y,Yhat_lm)
r_score_lm = r2_score(Y, Yhat_lm)
print('The Mean Squared Error is ', mse_lm, ' and the R^2 score is ', r_score_lm)


# The higher the R^2 score and the lower the Mean Squared Error, the better.

# <h3> Automate the process for all the columns </h3>

# In[ ]:


# Use the col_list to define the independent variable.

for x in col_list:
  print(x)
  loop_lm = LinearRegression().fit(df[[x]], Y)
  print('The intercept is ', loop_lm.intercept_, ' and the coefficient is ', loop_lm.coef_)
  Yhat_loop_lm = loop_lm.predict(df[[x]])
  print('The Mean Squared Error is ', mean_squared_error(Y, Yhat_loop_lm), ' and the R^2 score is ', r2_score(Y,Yhat_loop_lm))
  print('')


# If we have to use a Simple Linear Regression as model, the best rappresentation of the data and the clostest prediction would be achieved by using the 'Length3' column.

# <h2 id="mlr"> Multiple Linear Regression </h2>
# 

# <p>What if we want to predict fish weigth using more than one variable?</p>
# 
# <p>If we want to use more variables in our model to predict fish weight, we can use <b>Multiple Linear Regression</b>.
# Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between one continuous response (dependent) variable and <b>two or more</b> predictor (independent) variables.
# Most of the real-world regression models involve multiple predictors. We will illustrate the structure by using four predictor variables, but these results can generalize to any integer:</p>

# $$
# Y: Response \ Variable\\
# X_1 :Predictor\ Variable \ 1\\
# X_2: Predictor\ Variable \ 2\\
# X_3: Predictor\ Variable \ 3\\
# X_4: Predictor\ Variable \ 4\\
# $$

# $$
# a: intercept\\
# b_1 :coefficients \ of\ Variable \ 1\\
# b_2: coefficients \ of\ Variable \ 2\\
# b_3: coefficients \ of\ Variable \ 3\\
# b_4: coefficients \ of\ Variable \ 4\\
# $$

# The equation is given by

# $$
# Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$

# From the previous anasysis we highlighted that all the dependent variables we have are good to predict the fish weight. We are going to use all of them to train a model.

# In[ ]:


# Define the Multiple Linear Regression Model and the independent variables. Train it on dependent and independent variables.
X = df[col_list]
mlrm = LinearRegression().fit(X,Y)
mlrm


# In[ ]:


print('The intercept is ', mlrm.intercept_, ' and the coefficients are ', mlrm.coef_)


# In[ ]:


# Let's plot the distribution of the Y and the predicted Y
Yhat_mlrm = mlrm.predict(X)

plt.figure(figsize=(8,6))

ax1 =  sns.distplot(Y, hist=False, color='R', label='Actual weight')
sns.distplot(Yhat_mlrm, hist=False, color='B', ax=ax1, label='Predicted weight')


# As we can see the distribution of the predicted values is close to the distribution of the actual values but there are rooms for improvement.

# In[ ]:


# Calculate the Mean Squared Error and the R^2 score value
print('The Mean Squared Error is ', mean_squared_error(Y, Yhat_mlrm), ' and the R^2 Score is ', r2_score(Y, Yhat_mlrm))


# The best Simple Linear Regression model we have trained so far hat the following results:
# 
# Length3
# The Mean Squared Error is  18804.23911410419  and the R^2 score is  0.852095743638909
# </br>
# Considering that the Mean Squared Error of the Multiple Linear Regression is lower than the Mean Squared Error from the Simple Linear Regrassion model trained on the 'Length3' column, the MLR predict the fish weigth better than the SLR.</br>
# What about the R^2 score? </br>
# The R^2 score of the Multiple Linear Regression model is higher then the SLR model. This means that in general the MLRM better rappresent the dataset.

# <h2 id="poly"> Pipelines and Polynomial Regressions </h2>

# <p><b>Polynomial regression</b> is a particular case of the general linear regression model or multiple linear regression models.</p> 
# <p>We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.</p>
# 
# <p>There are different orders of polynomial regression:</p>

# <center><b>Quadratic - 2nd order</b></center>
# $$
# Yhat = a + b_1 X^2 +b_2 X^2 
# $$
# 
# 
# <center><b>Cubic - 3rd order</b></center>
# $$
# Yhat = a + b_1 X^2 +b_2 X^2 +b_3 X^3\\
# $$
# 
# 
# <center><b>Higher order</b>:</center>
# $$
# Y = a + b_1 X^2 +b_2 X^2 +b_3 X^3 ....\\
# $$

# <p>We saw earlier that a linear model did not provide the best fit while using highway-mpg as the predictor variable. Let's see if we can try fitting a polynomial model to the data instead.</p>

# <p>We will use the following function to plot the data:</p>

# In[ ]:


def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(min(independent_variable)*0.98, max(independent_variable)*1.01, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Weight')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Weight of fish')

    plt.show()
    #plt.close()


# In[ ]:


# Train a 5 degrees polynome
pol = np.polyfit(df['Width'], df['Weight'], 5)
func = np.poly1d(pol)
print(func)


# In[ ]:


# Plot the function
PlotPolly(func, df['Width'], df['Weight'], 'Width')


# The model seams to better rappresent the distribution.

# In[ ]:


for x in col_list:
  pol_loop = np.polyfit(df[x], df['Weight'], 5)
  func_loop = np.poly1d(pol_loop)
  print(func_loop)
  plt.figure()
  PlotPolly(func_loop, df[x], Y, Name=x)
  plt.show()


# <p>The analytical expression for Multivariate Polynomial function gets complicated. For example, the expression for a second-order (degree=2)polynomial with two variables is given by:</p>

# $$
# Yhat = a + b_1 X_1 +b_2 X_2 +b_3 X_1 X_2+b_4 X_1^2+b_5 X_2^2
# $$

# We can perform a polynomial transform on multiple features. First, we import the module:

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Define the features we need to take in consideration for this model
X_polF = df[col_list]


# In[ ]:


# Define the Input for the pipeline
Input = [('standardscaler', StandardScaler()), ('polynomial', PolynomialFeatures(degree=2, include_bias=False)), ('model', LinearRegression()) ]


# In[ ]:


pipe = Pipeline(Input)
pipe


# In[ ]:


pipe.fit(X_polF, Y)


# In[ ]:


Yhat_pipe = pipe.predict(X_polF)
Yhat_pipe[0:4]


# In[ ]:


# We can visualise the distribution of Yhat_pipe and the actual Y values in order to understand if the Polynomial Feature Regression model is a better model.
plt.figure(figsize=(12,10))
ax2 = sns.distplot(Y, hist=False, color='R', label='Actual values')
sns.distplot(Yhat_pipe, hist=False, color='G', label='Predicted values')
plt.show()


# As we can see from the distribution, the Polynomial Features Regression Model rappresent the data in a better way. Now we need to do some evaluation of the model. </br>
# </br>
# The previous best model was the following Simple Linear Regression: </br>
# <br>
# _Length3 The Mean Squared Error is 18804.23911410419 and the R^2 score is 0.852095743638909_</br>
# </br>
# Let's calculate the Mean Squared Error and the R^2 score of the new model.

# In[ ]:


print("The Mean Squared Error of the Polynomial Multiple Linear Regression is ",  mean_squared_error(Y, Yhat_pipe), ' and the R^2 score is ', r2_score(Y, Yhat_pipe))


# In[ ]:


Y_pred = pd.DataFrame(data=Yhat_pipe, columns=['Estimate Weight'])
prediction_df = pd.concat([Y_pred, Y], axis=1)
prediction_df


# <h2 id="conclusions">Conclusions </h2>
# </br>
# As we can see, the Mean Squared Error of the new model is ways better than the privious models and the R^2 score is very high. This means that the model is able to predict almost perfectly the weight of the fish based on his dimensions.

# In[ ]:




