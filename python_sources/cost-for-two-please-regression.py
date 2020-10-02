#!/usr/bin/env python
# coding: utf-8

# In this kernel, I am going to build a regression model that can predict the ***Average Cost for two***.
# Let's get started ! :) 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#load the csv file data into dataframe
df = pd.read_csv("../input/zomato.csv",encoding="ISO-8859-1")
country = pd.read_excel('../input/Country-Code.xlsx') # load the country values from excel file
df = pd.merge(df, country, on='Country Code')
df.head()


# In[ ]:


# how many data points for India?
df = df[(df['Country']=='India')]
df.shape # (row,column)


# In[ ]:


#Let's convert the boolean columns into integers 
df['Has Table booking'].replace({'Yes':1,'No':0},inplace=True)
df['Has Online delivery'].replace({'Yes':1,'No':0},inplace=True)
df['Switch to order menu'].replace({'Yes':1,'No':0},inplace=True)
df.head()


# In[ ]:


# The average cost for two can be dependent on the cuisine of the restaurant
# Continental or Italian will be more costly than NorthIndian :) 
# Let's find out the the number of unique cuisines.
cuisines = list(set(df['Cuisines'].str.cat(sep=',').replace(" ","").split(',')))
cuisines.sort()
cuisines


# In[ ]:


# Add one column for each cuisine and set value 1 for the column if restaurant 
# serves that cuisine else 0 
# Ultimately ,we are doing conversion of categorical columns into numerical columns.
for cuisine in cuisines:
    df[cuisine] = df['Cuisines'].str.contains(cuisine)
    df[cuisine].replace({True:1,False:0},inplace=True)
df.head()
# All cuisines added as column!


# In[ ]:


# Let's find out how each column is linearly related with Cost for two
corr = df.corr()[['Average Cost for two']].sort_values('Average Cost for two', ascending=False)
corr[corr['Average Cost for two']>0.2] # min 20% correlation !


# In[ ]:


# We are going to pick the top 5 features for building our regression model. 
df = df[['Price range','Has Table booking','Aggregate rating','Continental','Votes','Average Cost for two']]
df.head()
#Our final dataset ! :) 


# In[ ]:


# scaling data - all have different units for measurement. 
# We want all values to be across the same scale because cost for two
# is in rupees whereas aggregate rating is simple numbers from 0 to 5.
# Let's scale using min max scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),columns=df.columns)
df.head()


# In[ ]:


# Let's visualize how each columns is related to all other columns of the dataframe.
pd.scatter_matrix(df,figsize=(16,9),diagonal='kde',alpha=0.2)


# In[ ]:


# split data into training and testing set
from sklearn.model_selection import train_test_split

train,test = train_test_split(df,random_state=50)
X_train = train.iloc[:,df.columns!='Average Cost for two']
X_test = test.iloc[:,df.columns!='Average Cost for two']
y_train = train['Average Cost for two']
y_test = test['Average Cost for two']
print('Training set size - ' , X_train.shape)
print('Testing set size - ' , X_test.shape)


# In[ ]:


# We are going to start with the simplest type of regression - LinearRegression :) 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_test,y_pred)
score
# Not that impressive :( 


# As we can see that LinearRegression gives us r2_score of 0.7 but we should try to find out if any other model can give us better results. Let's try to build a model using **Polynomial Regression.** To help us find the best degree of fit for polynomial regression, we will be using **validation curve.** 

# In[ ]:


# Function to plot validation curve
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def plot_validation_curve(model,param_name,x_label,param_range=np.arange(1,7)):
    train_scores,validation_scores = validation_curve(model,X_train, y_train,
                                                     param_name = param_name,param_range=param_range,
                                                     scoring='r2',cv=3)
    validation_scores[validation_scores < 0] = 0 # we are not going to plot any negative numbers!
    print('Training scores  ',train_scores.mean(axis=1))
    print('Validation scores  ',validation_scores.mean(axis=1))
    
    plt.figure(figsize=(6, 4))
    plt.plot(param_range,validation_scores.mean(axis=1),lw=2, label='validation')
    plt.plot(param_range,train_scores.mean(axis=1),lw=2, label='training')
    plt.xlabel(x_label)
    plt.ylabel('Score')
    plt.title('Validation curve')
    plt.legend(loc='best')
    plt.show()


# In[ ]:


model = make_pipeline(PolynomialFeatures(),LinearRegression())
plot_validation_curve(model,'polynomialfeatures__degree',x_label='Degree of polynomial')


# Both the training and cross validation scores increase till 3rd degree of polynomial after which the cross validation score starts to decrease while the training score keeps increasing i.e our model starts to overfit for higher degree of polynomial. So, our best pick for polynomial **degree is 3**. Let's build the model and test it on our test data!

# In[ ]:


polynomial_features= PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)
reg = LinearRegression()
reg.fit(X_train_poly, y_train)
y_pred = reg.predict(X_test_poly)
score = r2_score(y_test,y_pred)
score


# Our r2_score increased from 0.7 to 0.82 ! Not bad ;) 
# Do you think **Decision Tree** can give better results? Let's find out ! :) 
# We are going to use validation curve to help us find the best number for max_depth.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
plot_validation_curve(DecisionTreeRegressor(random_state=42),'max_depth','Max Depth')


# Following the validation curve, our best pick for **max_depth should be 3.  **

# In[ ]:


reg = DecisionTreeRegressor(max_depth=3,random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_test,y_pred)
score


# The score above is not better than polynomial regression but definitely better than Linear Regression.
# Well, there's another algorithm - **RandomForestRegressor** which I am going to try next!
# It fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
plot_validation_curve(RandomForestRegressor(random_state=50,n_estimators=10),'max_depth','Max Depth',np.arange(1,10))


# In[ ]:


# From the graph, we can see that 4 is the best value for max_depth after which 
# the cross validation error starts to increase. 
# How about we confirm that using GridSearch Cross validation ;) 
from sklearn.model_selection import GridSearchCV
rfr_cv = GridSearchCV(RandomForestRegressor(random_state=50, n_estimators=100),
                     param_grid={'max_depth': np.arange(1,10)},
                     scoring='r2', cv=3)
rfr_cv.fit(X_train, y_train)
rfr_cv.best_params_


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

reg = RandomForestRegressor(max_depth=4,n_estimators=10,random_state=50)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_test,y_pred)
score


# This score is very close to polynomial regression! 
# I think Polynomial Regression is the winner wrt r2_score :) 

# **Learning Curve**

# Learning Curves can give a clear indication of under-fitting/over-fitting.
# Let's visualize it wrt RandomForestRegressor :) 

# In[ ]:


from sklearn.model_selection import learning_curve
underfitting_max_depth = 3
best_max_depth = 4
overfitting_max_depth = 20

def plot_learning_curve(max_depth,title):
    train_sizes, train_scores, validation_scores = learning_curve(RandomForestRegressor
                                                                  (max_depth=max_depth,n_estimators=18,
                                                                   random_state=50),
                                                                  X_train, y_train, cv = 5,
                                                                  train_sizes = np.linspace(.1, 1.0, 5))
    plt.plot(train_sizes, validation_scores.mean(axis=1), 'o-', color="r", label="Cross-validation score")
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="g", label="Training score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc='best') 
    plt.title(title)


# In[ ]:


plot_learning_curve(1,'Under-fitting')
# Meet at a lower point. Low training and testing score as number of samples increases.
# Performs bad on both training and cross-validation data.


# In[ ]:


plot_learning_curve(4,'Best fit')
# Meet at some higher point as both training and testing scores are high.


# In[ ]:


plot_learning_curve(50,'Over-fitting')
# The lines are far apart. High training score(memorizes the data) but low cross-validation score. 
# So, the lines never meet!


# Thank you for going through this kernel. Please upvote if you found this uselful :) 
