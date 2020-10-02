#!/usr/bin/env python
# coding: utf-8

# ## Spotify Top 50 Songs - Data Analysis
# 
# 
# ### Purpose: 
# 
# The purpose of this notebook is to analyze the dataset in place for top 50 Songs in Spotify. Per the initial data analysis, we may think of using appropriate machine learning algorithms to see patterns in the data if any. 
# 
# Overall looking at the description of the data, it seems to be both a supervised as well as unsupervised learning problem, but we'll starting finding useful information first out of our data, post which we will focus on model building and its improvement. 
# 
# The idea is to overall explore the data and find as much information as possible. 
# 

#   

# ### 1. Data Loading & Basic EDA

# In[ ]:


# Importing all required libraries. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Disabling in-line warnings in the Notebook. 
warnings.filterwarnings('ignore')


# Let's read the data and store it into a dataframe for further analysis. 

# In[ ]:


# Loading the Raw data. 
spotify = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv', encoding='cp1252')


# In[ ]:


# Looking at the sample rows of the Raw data. 
spotify.head(10)


# We can rename the unnamed column as the serial number, or even drop the column, as it is not making any use/sense. 

# In[ ]:


# Drop the unnamed column. 
spotify.drop('Unnamed: 0', axis=1, inplace=True)


# Let's check the overall summary statistics of the numeric fields in the Dataframe.

# In[ ]:


# Let's check the overall summary statistics of the numeric fields in the Dataframe.
spotify.describe()


# It seems the data does not contain much outliers. Also, there seems to be no missing values, but let's confirm. 

# In[ ]:


# Checking sum of NA values. 
spotify.isna().sum()


# Surely there are no issues with missing data. Let's now try to see some aspects of the dataset to find more about the Top 50 Songs. 

# What's the most energetic song? 

# In[ ]:


# Checking the song with Max energy. 
spotify[spotify.Energy == np.max(spotify.Energy)]


# Which song has most danceability factor? 

# In[ ]:


# Checking the song with Max Danceability. 
spotify[spotify.Danceability == np.max(spotify.Danceability)]


# Which song is more loud, at least as per the Loudness Decibels? 

# In[ ]:


# Checking the song with Max Loudness. 
spotify[spotify['Loudness..dB..'] == np.max(spotify['Loudness..dB..'])]


# Which song is the most Lively? 

# In[ ]:


# Checking the song with Max Danceability. 
spotify[spotify['Liveness'] == np.max(spotify['Liveness'])]


# Which is the most lengthy song? 

# In[ ]:


# Checking the song with Max Danceability. 
spotify[spotify['Length.'] == np.max(spotify['Length.'])]


# Finally, which is the most popular song? 

# In[ ]:


# Checking the song with Max Danceability. 
spotify[spotify['Popularity'] == np.max(spotify['Popularity'])]


# Let's check the structure of our data and try to find visual infomration, firstly out of the categorical information, followed by the numerical fields (wherever applicable). Later' we may also do combinations of these features/columns

# In[ ]:


# Checking structure of the dataframe. 
spotify.info()


# Let's check if there are any patterns in the Artist name. 

# In[ ]:


# Checking Histogram of artist name. 
plt.figure(figsize=(20,10))
sns.countplot(spotify['Artist.Name'])
plt.show()


# As we see no obvious patters in the data, let's check the Genre column of these songs. 

# In[ ]:


# Checking Histogram of artist name. 
plt.figure(figsize=(20,10))
sns.countplot(spotify['Genre'])
plt.show()


# Let's look at the Genre column once closely to see if we can find some meaningful information. From the earlier output of sample rows, we could see that keywords like "pop" which is a very casual nature of music is available in many Genres of many songs but is represented in different ways. We can perform some data cleaning there to get the actual Genre of the song, wherever applicable/possible. 

# In[ ]:


# Checking all rows of Genre column. 
spotify['Genre']


# In[ ]:


# Selecting rows where Genre contains the word "pop"
spotify[spotify['Genre'].str.contains('pop')]


# In[ ]:


# Checking the count of Songs with Pop Genre
spotify[spotify['Genre'].str.contains('pop')].count()


# Clearly out of 50 songs,23 belongs to Pop. Let's see what other Genres are available. 

# In[ ]:


# Checking rows with Genre other than Pop
spotify[~spotify['Genre'].str.contains('pop')]


# It seems we have some songs which are either Latin or Rap. let's check them one after another. 

# In[ ]:


# Checking the Latin songs. 
spotify[spotify['Genre'].str.contains('latin')].count()


# In[ ]:


# Checking the Rap songs. 
spotify[spotify['Genre'].str.contains('rap')].count()


# We have 5 Latin and 5 Rap songs. Let's check for hip hop category as well. 

# In[ ]:


# Checking the Hip Hop songs. 
spotify[spotify['Genre'].str.contains('hip')].count()


# Totally we have 4 songs only which are Hip Hop in Genre. Let's replace the Genre values in these columns as part of Data Cleaning. 

# In[ ]:


# Imputing values for Pop Genre
spotify.loc[spotify['Genre'].str.contains('pop', case=False), 'Genre'] = 'Pop'


# In[ ]:


# Imputing values for Latin Genre
spotify.loc[spotify['Genre'].str.contains('latin', case=False), 'Genre'] = 'Latin'


# In[ ]:


# Imputing values for Rap Genre
spotify.loc[spotify['Genre'].str.contains('rap', case=False), 'Genre'] = 'Rap'


# In[ ]:


# Imputing values for Hip Hop Genre
spotify.loc[spotify['Genre'].str.contains('hip', case=False), 'Genre'] = 'Hip-Hop'


# In[ ]:


# Checking the final status of Genre column. 
spotify.Genre


# Now let's take a histogram and see the overall distribution. 

# In[ ]:


# Checking distribution of the Genre column. 
plt.figure(figsize=(20,10))
sns.countplot(spotify['Genre'])
plt.show()


# Overall now we have seen the patterns in Genre of our Data. It is clear that Pop Genre is the most liked songs in the Top 50 songs list. 
# 
# Now let's try to see the distributions of the numerical variables/features, and then, further, combine the numerical and categorical variables . 

# In[ ]:


# Checking the list of columns in our dataset. 
spotify.columns


# In[ ]:


# Checking pairplots of all variables first. 
plt.figure(figsize=(20,10))
sns.pairplot(spotify, hue='Genre')
plt.show()


# In[ ]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (20, 10))
sns.heatmap(spotify.corr(), annot = True, cmap="YlGnBu")
plt.show()


# We see no linear relationships amongst any of the numerical variables. Also, other than popularity and some other columns, we see that for almost all of the cases the Pop Genre has the higher distribution than other Genres, which is also obvious. We can try to look at the data again to see if they have any regression properties.   

# Overall it seems that other than the Speechiness of any song, there is no linear relationship with any other features/columns with the Popularity of the song. We may later try to fit a Multiple Linear regression problem to find a model for predicting the Popularity of a song and the variables effecting or resulting into Popularity of a song. Also, we can shorten the dataset with only the top 4 Genres of songs (Pop, Latin, Hip-Hop and Rap) and try to fit a Clustering algorithm to see the clusters of data, but since we have 23 Pop songs in the top 50 songs, that'd not be a good model. 
# 
# In lieu of this, let's try and build a Multiple Linear Regression Model and see how it works with this data. We will prepare a different dataframe with the Numeric fields, while converting the text fields into Numerical fields (excluding the obvious fields like Track Title and Artist Name), followed by dividing the data into Training and Testing sets, and then we will fit our model. Finally, we will try to find our model's performance, and then try other forms of Regression (Lasso &/or Ridge) to find better results, wherever applicable. Definitely, we will have to check the distribution of the variables/features and scale them wherever appropriate. 

# ### 2. Data Preparation For Modelling

# Let's prepare the data for fitting a multiple linear regression. Although we have very less data, there is no harm in giving this a try. For this we have to drop a few columns from the dataset and also, we will have to encode the Genre column. 

# In[ ]:


# Checking the list of columns in our data. 
spotify.columns


# In[ ]:


# Dropping the Track Name column. 
spotify.drop('Track.Name', axis=1, inplace=True)


# In[ ]:


# Checking the columns again.
spotify.columns


# In[ ]:


# Dropping the Artist Name column. 
spotify.drop('Artist.Name', axis=1, inplace=True)


# In[ ]:


# importing library for label encoding the Genre Data. 
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# creating object for label encoding. 
le = LabelEncoder()


# In[ ]:


# Encoding the Genre column. 
spotify.Genre = le.fit_transform(spotify.Genre)


# In[ ]:


# Checking the dataset information. 
spotify.info()


# In[ ]:


# Checking the sample data. 
spotify.head(10)


# In[ ]:


# Check the distribution of target variable. 
plt.figure(figsize=(20,10))
sns.distplot(spotify.Popularity)
plt.show()


# In[ ]:


# Creating the Features and Targets datasets. 
X = spotify[['Genre', 'Beats.Per.Minute', 'Energy', 'Danceability',
       'Loudness..dB..', 'Liveness', 'Valence.', 'Length.', 'Acousticness..',
       'Speechiness.']]

y = spotify.Popularity


# In[ ]:


# Importing library for Train Test split. 
from sklearn.model_selection import train_test_split


# In[ ]:


# Creating the splits. 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[ ]:


# Checking the Training data. 
X_train


# In[ ]:


# Checking the dimensions of the training and testing sets. 
print("Training Feature data : ", X_train.shape)
print("Training Feature data : ", X_test.shape)
print("Training Feature data : ", y_train.shape)
print("Testing Target data : ", y_test.shape)


# In[ ]:


# Importing library for standard scaling
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Creating the scaler object
scaler = StandardScaler()


# In[ ]:


# Scaling the Training and Testing Data. 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Checking the dimensions of the training and testing sets. 
print("Training Feature data : ", X_train.shape)
print("Training Feature data : ", X_test.shape)
print("Training Feature data : ", y_train.shape)
print("Testing Target data : ", y_test.shape)


# In[ ]:


# Checking the training data. 
X_train


# We have now prepared our data for performing the multiple linear regression modelling. Let's see what our initial model yields, basis which we will take further steps. 

# ### 3. Initial Model

# In[ ]:


# Import the libraries .
from sklearn.linear_model import LinearRegression


# In[ ]:


# Creating the object
regressor = LinearRegression()


# In[ ]:


# Fit the model. 
regressor.fit(X_train, y_train)


# In[ ]:


# Predicting the test results. 
y_pred = regressor.predict(X_test)


# In[ ]:


# Checking the predictions. 
y_pred


# In[ ]:


# Checking the actuals
y_test


# In[ ]:


# Checking the model coefficients. 
regressor.coef_


# In[ ]:


# spotify dataset columns. 
X.columns


# In[ ]:


# Creating dataframe of features and coefficients. 
output = {'Features': X.columns, 'Coefficient': regressor.coef_}
output_df = pd.DataFrame(output)
output_df


# In[ ]:


# Checking RMSE

# Import libraries. 
from sklearn.metrics import mean_squared_error

# Checking the RMSE
mean_squared_error(y_pred, y_test)


# In[ ]:


# Checking the intercept
regressor.intercept_


# We have now created the Linear Regression Model with all variables. Note that the features Accousticness and Speechiness of a song contributes most to the Popularity of the song. The model has done a good job, but not great, in predicting the outcomes as we can see from comparing the Predictions and Actuals. However, we have a big intercept value of 88, I.e,, without the effect of any variable, our model suggest a value of 88 for the popularity score of a song on Spotify, which does not look good. It seems we may have to further do some preparations/cleaning of our data to get rid of unwanted features, by means of VIF (Variance Inflation Factor) analysis &/or PCA, so that we can get a better model, post which we can try out other modelling options like Lasso and Ridge. 

# ### 4. Model Improvement 

# Let's perform RFE method for eliminating the non required features from our dataset and try the modelling once more. 

# In[ ]:


# Importing the RFE Library. 
from sklearn.feature_selection import RFE


# In[ ]:


# Running RFE with the output number of the variable equal to 5
# We select 5 as we have total 10 variables, hence 5 looks to be a good number,
# Considering we do not loose much information from the functional perspective as well .. !! 
rfe = RFE(regressor, 5) # running RFE
rfe = rfe.fit(X_train, y_train) # Fitting the training data


# In[ ]:


# Getting the columns with RFE
list(zip(X.columns,rfe.support_,rfe.ranking_))


# From the Factor analysis using RFE method, we see that there are 5 variables which are on the top list, which is Energy of the song, Valence of the song, Length, Speechiness, and finally the accousticness of the song. Let's use these features to build our next model and see how it works.

# In[ ]:


# Getting total list of columns. 
X.columns


# In[ ]:


# Creating new set of features. 
X_new = X[['Energy', 'Valence.', 'Length.', 'Acousticness..', 'Speechiness.']]


# In[ ]:


# Creating new test and train sets. 
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)


# In[ ]:


# Scaling the Training and Testing Data. 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Creating the object
regressor_rfe = LinearRegression()


# In[ ]:


# Fitting the model. 
regressor_rfe.fit(X_train, y_train)


# In[ ]:


# Getting the predictions. 
y_pred_rfe = regressor_rfe.predict(X_test)


# In[ ]:


# Checking the RMSE
mean_squared_error(y_pred_rfe, y_test)


# In[ ]:


# Checking the intercept
regressor_rfe.intercept_


# In[ ]:


# Creating dataframe of features and coefficients. 
output_rfe = {'Features': X_new.columns, 'Coefficient': regressor_rfe.coef_}
output_df_rfe = pd.DataFrame(output_rfe)
output_df_rfe


# In[ ]:


# Creating dataframe of actuals and predictions for side by side comparisons. 
# We will compare the differences as well, if possible. 
prediction_diff = y_pred_rfe - y_test
check_predictions = {'Predictions': y_pred_rfe, 'Actuals': y_test, 'Difference': prediction_diff}
check_predictions_df = pd.DataFrame(check_predictions)
check_predictions_df


# In[ ]:


# Checking shape of the output predictions comparisons' dataframe. 
check_predictions_df.shape


# We have quite a good model now. Only in 3 cases we have considerable differences in the predictions out of 15 cases, while the rest of the cases are almost a near to accurate match to the actual target variable. Hence, we are able to quite accurately predict the popularity of the song at least from the top 50 songs list extract out of Spotify. Needless to say, if we input more data and improve our model further, we may have very good predictions as well, as compared to the current model. 
# 
# Looking at the coefficients of our Model, again, it is evident that Accousticness and Speechiness and Energy of the song effect the Popularity of our songs very much positively. Valence and Length of the song effects the popularity of the song negetively. 
# 
# Further, let's perform Variance Inflation Factor analysis, to see if we can find a better model yet again by removing such features, which are inter-correlated with each other, so that we can explain the maximum possible variance in our data. 

# In[ ]:


# Import the required library. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Calculate the VIFs for all the variables/features in our dataset. 

vif_all = pd.DataFrame()

vif_all['Features'] = X.columns

vif_all['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_all['VIF'] = round(vif_all['VIF'], 2)

vif_all = vif_all.sort_values(by = "VIF", ascending = False)

vif_all


# In[ ]:


# Calculate the VIFs for the new model

vif = pd.DataFrame()

vif['Features'] = X_new.columns

vif['VIF'] = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif


# In[ ]:


# Dropping the Energy column from our new dataset and checking the VIF again. 

X_new.drop('Energy', axis=1, inplace=True)

vif_2 = pd.DataFrame()

vif_2['Features'] = X_new.columns

vif_2['VIF'] = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]

vif_2['VIF'] = round(vif_2['VIF'], 2)

vif_2 = vif_2.sort_values(by = "VIF", ascending = False)

vif_2


# ### 5. Final Modelling, Explaination & Evaluation

# In[ ]:


# Creating new test and train sets. 
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)


# In[ ]:


# Scaling the Training and Testing Data. 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Creating the object
regressor_vif = LinearRegression()


# In[ ]:


# Fitting the model. 
regressor_vif.fit(X_train, y_train)


# In[ ]:


# Making predictions. 
y_pred_vif = regressor_vif.predict(X_test)


# In[ ]:


# Creating dataframe of actuals and predictions for side by side comparisons. 
# We will compare the differences as well, if possible. 
prediction_diff = y_pred_vif - y_test
check_predictions = {'Predictions': y_pred_vif, 'Actuals': y_test, 'Difference': prediction_diff}
check_predictions_df = pd.DataFrame(check_predictions)
check_predictions_df


# In[ ]:


# Checking the RMSE
mean_squared_error(y_pred_vif, y_test)


# In[ ]:


# Checking the intercept
regressor_vif.intercept_


# In[ ]:


# Creating dataframe of features and coefficients. 
output_vif = {'Features': X_new.columns, 'Coefficient': regressor_vif.coef_}
output_df_vif = pd.DataFrame(output_vif)
output_df_vif


# Finally, we have much more stable model without any inflation factors or inter-correlations between the features themselves. 
# 
# Part of the model's output, we find that Accousticness and Speechiness is highly positively correlated with the Popularity of the song, while the Valence and Length of the song reduces the Popularity of the song. 
# 
# Finally, let's check our linear predictions by plotting the Actuals and Predictions on Test set and also if our error terms have normal distribution or not. 

# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure(figsize=(20,10))
plt.scatter(y_test,y_pred_vif)
fig.suptitle('Actuals v/s Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Actuals', fontsize=18)                          # X-label
plt.ylabel('Predicted', fontsize=16)                          # Y-label


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_test - y_pred_vif), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# The model output has somewhat uptrending linear relationship with some scatter. With more data, we can further optimize this model. 
# 
# The error terms are not entirely normally distributed, but they are not bad as well. 

# ## End of Analysis. 
