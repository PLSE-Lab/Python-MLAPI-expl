#!/usr/bin/env python
# coding: utf-8

# # Melbourne House Price Prediction
# 
# In my former Kaggle Kernels, I did some training models or data visualization issues. It seems I did some job unclear to express my thoughts. Then I realized that it is not enough if you do not explain your codings' intentions and graphs' meanings. So I learned Dan Becker's tutorial: **Intro to Machine Learning**. He detailed explained the process of how to train and analyze the model. So this time, based on his introduction, I am going to recap his work with my understanding. It will divede into five parts:
# #### Part1: EDA(Explotary Data Analysis)
# #### Part2: Training Decision Tree Model
# #### Part3: Model Valudation
# #### Part4: Underfitting and Overfitting Analysis

# Before we explotray this dataset, we have a brief background introduction. Melbourne is the capital and the most populous city of the Australia. It's real estate is booming. If we can predict housing prices accurately, it would be very useful for estate clients. This dataset is a snapshot of a dataset created bu Tony Pino. It was scraped from publicly available results posted every week from Domain.com.au. He cleaned it well, and now we can make data analysis. 
# #### The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.

# ### Part1: EDA(Explotary Data Analysis)
# 

# In[ ]:


import pandas as pd
data = pd.read_csv('/kaggle/input/melbourne-housing-snapshot/melb_data.csv')
data.head()


# In[ ]:


data.columns


# In[ ]:


data.describe()


# ##### From the information above, we can get the features of this dataset.
# From data.describe(), there are 8 columns for each column in this dataset. The first line means there are totally 13580 rows. The second value is mean, which is the average. The third calue is standard deviation, which measures how numberically spread out the values are.

# In order to better explotary the data, we can with the help of charts to show data's value. So we can import seaborn and matplotlib to make charts. 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(data['Price'])


# In[ ]:


sns.heatmap(data.corr())


# When we have a data set with many columns, a good way to quickly check correlations among columns is by **visualizing the correlation matrix as a heatmap**. White means positive, black means negative. The stronger the color, the smaller the correlation magnitude. So, what do you eues jump first when you look at the chart? We can find that the Rooms and Bedroom2 have a stronger relation.

# In[ ]:


sns.pairplot(data)


# Once we got a dataset, the next step is Exploratory Data Analysis. EDA is the process of figuring out what the data can tell us and we use EDA to find patterns, relationships, or anomalies to inform our subsequent analysis. While there are almost overwhelming number of methods to use in EDA, one of the most effective starting tools is the pairs plot. The pair plot allows us to see both the distribution of single variable and relationships between two variables. Pair plots are a great method to identify trends for follow-up analysis and, fortunately, are easily implemented in Python. There are so many features and we will choose some of them to analyze.

# ### Part2: Training Model
# 

# Before we start to train models, let's have a look of how machine learning models work.![avarar](http://i.imgur.com/7tsb5b1.png)

# This model called Decision tree and we will make prediction based on this model. It divides houses into only two categories. The predicted price for any house under consideration is the historical average price of houses in the same category. We use data to decide how to break the houses into two groups, and then again to determine the predicted price in each group. This step of capturing patterns from data is called fitting or **training the model**. The data used to fit the model is called the **training data**.

# There are too many variables to wrap your head and you can not distinguish which is the most relevant one. So at first, we start by picking a few varibles using our intuition. Next, we will choose prediction target and relevant features.
# * Choose Prediction Target: We'll use the dot notation to select the column we want to predict, which is called the prediction target. By convention, the prediction target is called y. 
# * Choose "Features": The columns that are inputted into our model (and later used to make predictions) are called "features." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as features. Other times you'll be better off with fewer features. For now, we'll build a model with only a few features. 

# In[ ]:


y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]


# In[ ]:


X.describe()


# In[ ]:


X.head()


# The most important part is coming. Use the scikit-learn library to create our models.The steps to building and using a model are:
# * Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# * Fit: Capture patterns from provided data. This is the heart of modeling.
# * Predict: Just what it sounds like
# * Evaluate: Determine how accurate the model's predictions are.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 10)


# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


model.fit(X_train, y_train)


# Many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice. You use any number, and model quality won't depend meaningfully on exactly what value you choose. We now have a fitted model that we can use to make predictions.

# In[ ]:


predictions = model.predict(X_test)


# ### Part3: Model Validation

# We want to use model validation to measure the quality of model. Measuring model quality is the key to iteratively improving models. We'll want to evaluate almost every model you ever build. In most (though not all) applications, the relevant measure of model quality is predictive accuracy. In other words, will the model's predictions be close to what actually happens. There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE). Let's break down this metric starting with the last word, error.
# * error = actual - predictions

# In[ ]:


plt.scatter(y_test, predictions)


# In[ ]:


sns.distplot((y_test-predictions), bins=50)


# Then, here is how we calculate the mean absolute error:

# In[ ]:


from sklearn.metrics import mean_absolute_error
predict_train = model.predict(X_train)
mean_absolute_error(y_train, predict_train)


# ### Part4: Underfitting and Overfitting Analysis
# 

# If we want to measure model accuracy, one way is to train it with another model and see which gives the better prediction. There are also other ways to decision tree models to measure model accuracy. The most important options is termine the tree's depth.

# In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses) and a leaf. As the tree gets deeper, the dataset gets sliced up into leaves with fewer houses. If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses. Splitting each of those again would create 8 groups. If we keep doubling the number of groups by adding more splits at each level, we'll have a large number of groups of houses by the time we get to the 10th level. That's 1024 leaves.
# 
# When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
# 
# This is a phenomenon called **overfitting**, where a model matches the training data almost perfectly, but does poorly in validation and other new data. On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.
# 
# At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called **underfitting**.

# Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting. Visually, we want the **low point** of the (red) validation curve.

# ![2q85n9s.png](attachment:2q85n9s.png)

# There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes. But the max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
# 
# We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[ ]:


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

