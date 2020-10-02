#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# # Project: Yelp Rating Regression Predictor
# 
# The restaurant industry is tougher than ever, with restaurant reviews blazing across the Internet from day one of a restaurant's opening. But as a lover of food, you and your friend decide to break into the industry and open up your own restaurant, Danielle's Delicious Delicacies. Since a restaurant's success is highly correlated with its reputation, you want to make sure Danielle's Delicious Delicacies has the best reviews on the most queried restaurant review site: Yelp! While you know your food will be delicious, you think there are other factors that play into a Yelp rating and will ultimately determine your business's success. With a dataset of different restaurant features and their Yelp ratings, you decide to use a Multiple Linear Regression model to investigate what factors most affect a restaurant's Yelp rating and predict the Yelp rating for your restaurant!
# 
# In this project we'll be working with a real dataset provided by Yelp. We have provided six files, listed below with a brief description:
# * `yelp_business.json`: establishment data regarding location and attributes for all businesses in the dataset
# * `yelp_review.json`: Yelp review metadata by business
# * `yelp_user.json`: user profile metadata by business
# * `yelp_checkin.json`: online checkin metadata by business
# * `yelp_tip.json`: tip metadata by business
# * `yelp_photo.json`: photo metadata by business
# 
# For a more detailed explanation of the features in each `.json` file, see the accompanying [explanatory feature document](https://docs.google.com/document/d/1V6FjJpKspVBOOBs4E7fBfp_yzHn0--XJkC2uUtWuRgM/edit).
# 
# Let's get started by exploring the data in each of these files to see what we are working with.

# ## Load the Data and Take a Peek
# 
# To get a better understanding of the dataset we can use Pandas to explore the data in DataFrame form. In the code block below we have imported Pandas for you. The `read_json()` method reads data from a json file into a DataFrame, as shown below:
# ```python
# df = pd.read_json('file_name.json', lines=True)
# ```
# Load the data from each of the json files with the following naming conventions:
# * `yelp_business.json` into a DataFrame named `businesses`
# * `yelp_review.json` into a DataFrame named `reviews`
# * `yelp_user.json` into a DataFrame named `users`
# * `yelp_checkin.json` into a DataFrame named `checkins`
# * `yelp_tip.json` into a DataFrame named `tips`
# * `yelp_photo.json` into a DataFrame named `photos`
# 
# Importing that data could take 10 to 20 seconds to run depending on your computer, but don't worry, once it's loaded in you're ready to go!

# In[ ]:


import pandas as pd

businesses = pd.read_json('../input/yelp_business.json', lines=True)
reviews = pd.read_json('../input/yelp_review.json', lines=True)
users = pd.read_json('../input/yelp_user.json', lines=True)
checkins = pd.read_json('../input/yelp_checkin.json', lines=True)
tips = pd.read_json('../input/yelp_tip.json', lines=True)
photos = pd.read_json('../input/yelp_photo.json', lines=True)


# In order to more clearly see the information in our DataFrame, we can adjust the number of columns shown (`max_columns`) and the number of characters shown in a column (`max_colwidth`) with the below code:
# 
# ```python
# pd.options.display.max_columns = number_of_columns_to_display
# pd.options.display.max_colwidth = number_of_characters_to_display
# ```
# 
# Set `max_columns` to `60` and `max_colwidth` to `500`. We are working with some BIG data here!

# In[ ]:


max_columns = 60
max_colwidth = 500


# Inspect the first five rows of each DataFrame using the `.head()` method to get an overview of the data (make sure to check each DataFrame in a separate cell in order to view it properly).

# In[ ]:


businesses.head()


# In[ ]:


reviews.head()


# In[ ]:


users.head()


# In[ ]:


checkins.head()


# In[ ]:


tips.head()


# In[ ]:


photos.head()


# How many different businesses are in the dataset? What are the different features in the review DataFrame?

# In[ ]:


print(businesses.business_id.nunique())

print(list(reviews.columns))


# What is the range of values for the features in the user DataFrame?

# In[ ]:


print(users.describe())
#Another option
users.describe()


# What is the Yelp rating, or `stars`, of the establishment with `business_id` = `5EvUIR4IzCWUOm0PsUZXjA`. Use Pandas boolean indexing to find the Yelp rating, using the syntax below:
# 
# ```python
# df[df['column_we_know'] == 'value_we_know']['column_we_want']
# ```

# In[ ]:


businesses[businesses['business_id'] == '5EvUIR4IzCWUOm0PsUZXjA']['stars']


# In[ ]:


# What feature, or column, do the DataFrames have in common?


# ## Merge the Data
# 
# Since we are working with data from several files, we need to combine the data into a single DataFrame that allows us to analyze the different features with respect to our target variable, the Yelp rating. We can do this by merging the multiple DataFrames we have together, joining them on the columns they have in common. In our case, this unique identifying column is the `business_id`. We can merge two DataFrames together with the following syntax:
# 
# ```python
# pd.merge(left, right, how='inner/outer/left/right', on='column(s)_to_merge_on')
# ```
# * `left` is the DataFrame on the left side of our merge
# * `right` is the DataFrame on the right side of our merge
# * `how` describes the style of merge we want to complete (similar to inner/outer/left/right joins in SQL)
# * `on` is the column or columns to perform the merge on (the column connecting the two tables)
# 
# Given our six DataFrames, we will need to perform 5 merges to combine all the data into one DataFrame. In the cell below we merged the business table and the review table into a new DataFrame, `df`, for you. After the merge we've added all the rows from `businesses` and `reviews` together, but kept the same total number of rows! Run the cell to perform the merge and confirm the number of rows in `df`. 

# In[ ]:


df = pd.merge(businesses, reviews, how='left', on='business_id')
print(len(df))


# Merge each of the other 4 DataFrames into our new DataFrame `df` to combine all the data together. Make sure that `df` is the left DataFrame in each merge and `how=left` since not every DataFrame includes every business in the dataset (this way we won't lose any data during the merges). Once combined, print out the columns of `df`. What features are in this new DataFrame?

# In[ ]:


df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')

print(df.columns)


# ## Clean the Data
# 
# We are getting really close to the fun analysis part! We just have to clean our data a bit so we can focus on the features that might have predictive power for determining an establishment's Yelp rating.
# 
# In a Linear Regression model, our features will ideally be continuous variables that have an affect on our dependent variable, the Yelp rating. For this project with will also be working with some features that are binary, on the scale [0,1]. With this information, we can remove any columns in the dataset that are not continuous or binary, and that we do not want to make predictions on. The cell below contains a list of these unnecessary features. Drop them from `df` with Pandas' drop syntax, provided below:
# 
# ```python
# df.drop(list_of_features_to_remove, axis=1, inplace=True)
# ```
# 
# * `list_of_features_to_remove` is, you guessed it, the list of features we want to remove!
# * `axis=1` lets Pandas know we want to drop columns, not rows, from our DataFrame (axis=0 is used for computations along rows!) 
# * `inplace=True` lets us drop the columns right here in our DataFrame, instead of returning a new DataFrame that we could store in a new variable

# In[ ]:


features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']

df.drop(labels=features_to_remove, axis=1, inplace=True)


# Now we just have to check our data to make sure we don't have any missing values, or `NaN`s, which will prevent the Linear Regression model from running correctly. To do this we can use the statement `df.isna().any()`. This will check all of our columns and return `True` if there are any missing values or `NaN`s, or `False` if there are no missing values. Check if `df` is missing any values.

# In[ ]:


df.isna().any()


# As you can see, there are a few columns with missing values. Since our dataset has no information recorded for some businesses in these columns, we will assume the Yelp pages did not display these features. For example, if there is a `NaN` value for `number_pics`, it means that the associated business did not have any pictures posted on its Yelp page. Thus we can replace all of our `NaN`s with `0`s. To do this we can use the `.fillna()` method, which takes a dictionary as shown below:
# 
# ```python
# df.fillna({'column_1':val_to_replace_na,
#            'column_2':val_to_replace_na,
#            'column_3':val_to_replace_na},
#           inplace=True)
# ```
# 
# * `column_1`, `column_2`, and `column_3` are the columns with missing values that we want to fill. We can include as many columns as we like in the dictionary that is passed to `.fill_na()`
# * `val_to_replace_na` is the value that will replace the missing values, or `NaN`s
# * `inplace=True` since we want to perform our changes in place and not return a new DataFrame
# 
# Fill the missing values in `df` with `0`. Afterwards, confirm the missing values have been filled with `df.isna().any()`.

# In[ ]:


df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips':0,
           'average_caption_length':0,
           'number_pics':0},
          inplace=True)

df.isna().any()


# ## Exploratory Analysis
# 
# Now that our data is all together, let's investigate some of the different features to see what might correlate most with our dependent variable, the Yelp rating (called `stars` in our DataFrame). The features with the best correlations could prove to be the most helpful for our Linear Regression model! Pandas DataFrames have a really helpful method, `.corr()`, that allows us to see the correlation coefficients for each pair of our different features. Remember, a correlation of `0` indicates that two features have no linear relationship, a correlation coefficient of `1` indicates two features have a perfect positive linear relationship, and a correlation coefficient of `-1` indicates two features have a perfect negative linear relationship. Call `.corr()` on `df`. You'll see that `number_funny_votes` has a correlation coefficient of `0.001320` with respect to `stars`, our Yelp rating. This is a very weak correlation. What features best correlate, both positively and negatively, with Yelp rating?

# In[ ]:


df.corr()   


# To further visualize these relationships, we can plot certain features against our dependent variable, the Yelp rating. In the cell below we have provided the code to import Matplotlib. We can use Matplotlib's `.scatter()` method with the below syntax to plot what these correlations look like:
# 
# ```python
# plt.scatter(x_values_to_plot, y_values_to_plot, alpha=blending_val)
# ```
# 
# * `x_values_to_plot` are the values to be plotted along the x-axis
# * `y_values_to_plot` are the values to be plotted along the y-axis
# * `alpha=blending_val` is the blending value, or how transparent (0) or opaque (1) a plotted point is. This will help us distinguish areas of the plot with high point densities and low point densities
# 
# Plot the three features that correlate most with Yelp rating (`average_review_sentiment`, `average_review_length`, `average_review_age`) against `stars`, our Yelp rating. Then plot a lowly correlating feature, such as `number_funny_votes`, against `stars`.
# 
# >What is `average_review_sentiment`, you ask? `average_review_sentiment` is the average sentiment score for all reviews on a business' Yelp page. The sentiment score for a review was calculated using the sentiment analysis tool [VADER](https://github.com/cjhutto/vaderSentiment). VADER uses a labeled set of positive and negative words, along with codified rules of grammar, to estimate how positive or negative a statement is. Scores range from `-1`, most negative, to `+1`, most positive, with a score of `0` indicating a neutral statement. While not perfect, VADER does a good job at guessing the sentiment of text data!
# 
# What kind of relationships do you see from the plots? Do you think these variables are good or bad features for our Yelp rating prediction model?

# In[ ]:


from matplotlib import pyplot as plt

# plot average_review_sentiment against stars here
plt.scatter(df.average_review_sentiment, df.stars)
plt.xlabel('Average Review Sentiment')
plt.ylabel('Ratings')
plt.title('Correlation between review sentiment and Yelp rating')
plt.show()


# In[ ]:


# plot average_review_length against stars here
plt.scatter(df.average_review_length, df.stars)
plt.xlabel('Average Review Length')
plt.ylabel('Ratings')
plt.title('Correlation between review length and Yelp rating')
plt.show()


# In[ ]:


# plot average_review_age against stars here
plt.scatter(df.average_review_age, df.stars)
plt.xlabel('Average Review Age')
plt.ylabel('Ratings')
plt.title('Correlation between review age and Yelp rating')
plt.show()


# In[ ]:


# plot number_funny_votes against stars here
plt.scatter(df.number_funny_votes, df.stars)
plt.xlabel('Number of Funny Votes')
plt.ylabel('Ratings')
plt.title('Correlation between funny votes and Yelp rating')
plt.show()


# In[ ]:


#Why do you think `average_review_sentiment` correlates so well with Yelp rating?


# ## Data Selection
# 
# In order to put our data into a Linear Regression model, we need to separate out our features to model on and the Yelp ratings. From our correlation analysis we saw that the three features with the strongest correlations to Yelp rating are `average_review_sentiment`, `average_review_length`, and `average_review_age`. Since we want to dig a little deeper than `average_review_sentiment`, which understandably has a very high correlation with Yelp rating, let's choose to create our first model with `average_review_length` and `average_review_age` as features.
# 
# Pandas lets us select one column of a DataFrame with the following syntax:
# 
# ```python
# subset_of_data = df['feature_to_select']
# ```
# Pandas also lets us select multiple columns from a DataFrame with this syntax:
# 
# ```python
# subset_of_data = df[list_of_features_to_select]
# ```
# Create a new DataFrame `features` that contains the columns we want to model on: `average_review_length` and `average_review_age`. Then create another DataFrame `ratings` that stores the value we want to predict, Yelp rating, or `stars` in `df`.

# In[ ]:


features = df[['average_review_length', 'average_review_age']]
ratings = df[['stars']]


# ## Split the Data into Training and Testing Sets
# 
# We are just about ready to model! But first, we need to break our data into a training set and a test set so we can evaluate how well our model performs. We'll use scikit-learn's `train_test_split` function to do this split, which is provided in the cell below. This function takes two required parameters: the data, or our features, followed by our dependent variable, in our case the Yelp rating. Set the optional parameter `test_size` to be `0.2`. Finally, set the optional parameter `random_state` to `1`. This will make it so your data is split in the same way as the data in our solution code. 
# 
# Remember, this function returns 4 items in this order:
# 1. The training data (features), which we can assign to `X_train`
# 2. The testing data (features), which we can assign to `X_test`
# 3. The training dependent variable (Yelp rating), which we can assign to `y_train`
# 4. The testing dependent variable (Yelp rating), which we can assign to `y_test`

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)


# ## Create and Train the Model
# 
# Now that our data is split into training and testing sets, we can finally model! In the cell below we have provided the code to import `LinearRegression` from scikit-learn's `linear_model` module. Create a new `LinearRegression` object named model. The `.fit()` method will fit our Linear Regression model to our training data and calculate the coefficients for our features. Call the `.fit()` method on `model` with `X_train` and `y_train` as parameters. Just like that our model has now been trained on our training data!

# In[ ]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# ## Evaluate and Understand the Model
# 
# Now we can evaluate our model in a variety of ways. The first way will be by using the `.score()` method, which provides the R^2 value for our model. Remember, R^2 is the coefficient of determination, or a measure of how much of the variance in our dependent variable, the predicted Yelp rating, is explained by our independent variables, our feature data. R^2 values range from `0` to `1`, with `0` indicating that the created model does not fit our data at all, and with `1` indicating the model perfectly fits our feature data. Call `.score()` on our model with `X_train` and `y_train` as parameters to calculate our training R^2 score. Then call `.score()` again on model with `X_test` and `y_test` as parameters to calculate R^2 for our testing data. What do these R^2 values say about our model? Do you think these features alone are able to effectively predict Yelp ratings?

# In[ ]:


model.score(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


print('Only 8.08% of the variance in Yelp rating is explained by chosen features.')


# After all that hard work, we can finally take a look at the coefficients on our different features! The model has an attribute `.coef_` which is an array of the feature coefficients determined by fitting our model to the training data. To make it easier for you to see which feature corresponds to which coefficient, we have provided some code in the cell that `zip`s together a list of our features with the coefficients and sorts them in descending order from most predictive to least predictive.

# In[ ]:


sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)


# Lastly we can calculate the predicted Yelp ratings for our testing data and compare them to their actual Yelp ratings! Our model has a `.predict()` method which uses the model's coefficients to calculate the predicted Yelp rating. Call `.predict()` on `X_test` and assign the values to `y_predicted`. Use Matplotlib to plot `y_test` vs `y_predicted`. For a perfect linear regression model we would expect to see the data plotted along the line `y = x`, indicating homoscedasticity. Is this the case? If not, why not? Would you call this model heteroscedastic or homoscedastic?

# In[ ]:


y_predicted = model.predict(X_test)
plt.scatter(y_test, y_predicted)
plt.xlabel('Yelp Rating')
plt.ylabel('Predicted Yelp Rating')
plt.show()


# ## Define Different Subsets of Data
# 
# After evaluating the first model, you can see that `average_review_length` and `average_review_age` alone are not the best predictors for Yelp rating. Let's go do some more modeling with different subsets of features and see if we can achieve a more accurate model! In the cells below we have provided different lists of subsets of features that we will model with and evaluate. What other subsets of features would you want to test? Why do you think those feature sets are more predictive of Yelp rating than others? Create at least one more subset of features that you want to predict Yelp ratings from.

# In[ ]:


# subset of only average review sentiment
sentiment = ['average_review_sentiment']


# In[ ]:


# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']


# In[ ]:


# subset of all features that vary on a greater range than [0,1]
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']


# In[ ]:


# all features
all_features = binary_features + numeric_features

features = df[all_features]

X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


print('All features explain 67.82% of the variance in Yelp rating.')


# In[ ]:


# add your own feature subset here

# I will choose from the features indicating larger correlation (> 0.1) with stars in the df.corr() table
feature_subset =['average_review_age', 'average_review_length', 'average_review_sentiment']
features = df[feature_subset]

X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)


# In[ ]:


print('My chosen features explain 64.96% of the variance in Yelp rating. A huge improvement from previous results!')


# ## Further Modeling

# Now that we have lists of different feature subsets, we can create new models from them. In order to more easily compare the performance of these new models, we have created a function for you below called `model_these_features()`. This function replicates the model building process you just completed with our first model! Take some time to review how the function works, analyzing it line by line. Fill in the empty comments with an explanation of the task the code beneath it is performing.

# In[ ]:


import numpy as np

# take a list of features to model as a parameter
def model_these_features(feature_list):
    
    # #select the dataframe with stars and features 
    ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]
    
    # #split train and test data
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    # don't worry too much about these lines, just know that they allow the model to work when
    # we model on just one feature instead of multiple features. Trust us on this one :)
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    # #train
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    # #get score 
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # print the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    # #get prediced data
    y_predicted = model.predict(X_test)
    
    # #compare predicted data with actual data
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()


# Once you feel comfortable with the steps of the function, run models on the following subsets of data using `model_these_features()`:
# * `sentiment`: only `average_review_sentiment`
# * `binary_features`: all features that have a response range [0,1]
# * `numeric_features`: all features that vary on a greater range than [0,1]
# * `all_features`: all features
# * `feature_subset`: your own feature subset
# 
# How does changing the feature sets affect the model's R^2 value? Which features are most important to predicting Yelp rating in the different models? Which models appear more or less homoscedastic?

# In[ ]:


# create a model on sentiment here
model_these_features(sentiment)


# In[ ]:


# create a model on all binary features here
model_these_features(binary_features)


# In[ ]:


# create a model on all numeric features here
model_these_features(numeric_features)


# In[ ]:


# create a model on all features here
model_these_features(all_features)


# In[ ]:


# create a model on your feature subset here
model_these_features(feature_subset)


# ## Danielle's Delicious Delicacies' Debut
# 
# You've loaded the data, cleaned it, modeled it, and evaluated it. You're tired, but glowing with pride after all the hard work. You close your eyes and can clearly see opening day of Danielle's Delicious Delicacies with a line out the door. But what will your Yelp rating be? Let's use our model to make a prediction.
# 
# Our best model was the model using all features, so we'll work with this model again. In the cell below print `all_features` to get a reminder of what features we are working with.

# In[ ]:


print(all_features)


# Run the cell below to grab all the features and retrain our model on them.

# In[ ]:


features = df.loc[:,all_features]
ratings = df.loc[:,'stars']
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
model = LinearRegression()
model.fit(X_train,y_train)


# To give you some perspective on the restaurants already out there, we have provided the mean, minimum, and maximum values for each feature below. Will Danielle's Delicious Delicacies be just another average restaurant, or will it be a 5 star behemoth amongst the masses?

# In[ ]:


pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])


# Based on your plans for the restaurant, how you expect your customers to post on your Yelp page, and the values above, fill in the blanks in the NumPy array below with your desired values. The first blank corresponds with the feature at `index=0` in the DataFrame above, `alcohol?`, and the last blank corresponds to the feature at ``index=24``, `weekend_checkins`. Make sure to enter either `0` or `1` for all binary features, and if you aren't sure of what value to put for a feature, select the mean from the DataFrame above. After you enter the values, run the prediction cell below to receive your Yelp rating! How is Danielle's Delicious Delicacies debut going to be?

# In[ ]:


danielles_delicious_delicacies = np.array([0,1,1,1,1,1,10,2,3,10,10,1200,0.9,3,6,5,50,3,50,1800,12,123,0.5,0,0]).reshape(1,-1)


# In[ ]:


model.predict(danielles_delicious_delicacies)


# ## Next Steps
# 
# You have successfully built a linear regression model that predicts a restaurant's Yelp rating! As you have seen, it can be pretty hard to predict a rating like this even when we have a plethora of data. What other questions come to your mind when you see the data we have? What insights do you think could come from a different kind of analysis? Here are some ideas to ponder:
# 
# * Can we predict the cuisine of a restaurant based on the users that review it?
# * What restaurants are similar to each other in ways besides cuisine?
# * Are there different restaurant vibes, and what kind of restaurants fit these conceptions?
# * How does social media status affect a restaurant's credibility and visibility?
# 
# As you progress further into the field of data science, you will be able to create models that address these questions and many more! But in the meantime, get back to working on that burgeoning restaurant business plan.
