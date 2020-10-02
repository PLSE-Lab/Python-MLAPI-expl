#!/usr/bin/env python
# coding: utf-8

# <h1>Using a Single Layer Keras Model vs Sklearn Linear Regression</h1>
# 
# Recently, I completed the course "Zero to Deep Learning" on Udemy and wanted to find an unclean data set to work with the kearas package. I have utilized neural networks and regression before but not with Keras or Tensorflow. I decided to take this course to learn how to utilized the package and thought it would be a good idea to try my new skills on a Kaggle dataset. 
# 
# As I utilized the dataset, I realized what a great dataset it is for improving data analysis skills both with statistical modeling and data analysis/processing. Also, I wanted to try my hand at building a good kernal. Later in the analysis I noticed that I was getting weird results from the Keras model and learned about how Keras can give strange results for easy to learn data sets or data sets where it is not finding correlation. I believe that the simplicity of the dataset was the issue in this occurance. 
# 
# Normally, when I model data, I define my packages and functions upfront but for this project I will include these packages as I code so that it is more relatable to the sections of code where the package is applicable. 
# 
# <ol>
#     <h5>Sections</h5>
#         <li>Explore and Clean Data</li>
#         <li>Process Data for Modeling</li>
#         <li>Model Data</li>
#         <li>Evaluate Model</li>
# </ol>

# In[ ]:


import pandas as pd


# In[ ]:


#Load the data set
hdf = pd.read_csv('../input/housing.csv')


# <h3> Explore and Clean Data</h3>
# 
# To decide how we should proceed with modeling the data we first need to do some exploratory analysis. To do this we need to look for missing data (clean this data), turn categorical data into numeric ,analyze basic statistical metrics and plot the data to find insights. 

# In[ ]:


import numpy  as np


# <h4>Data Types and Columns</h4>
# First, we need to look at the data to determine if any transformations need to be made to the original data. Not in terms of necessary for linear fit but to ensure that the variables are on the same level. 
# 
# <ol><h5>Sections</h5>
#     <li>Numeric Transformations</li>
#     <li>One Hot Encoding</li>
#     <li>Missing Data</li>
#     <li>Explratory Visualizations</li>
#     <li>Outlier Treatment</li>
# </ol>

# In[ ]:


hdf.columns


# In[ ]:


# Find the datatypes of the data
[i + ' '+ str(hdf[i].dtype) for i in hdf.columns]


# In[ ]:


#Let's have a look at the numeric variables. 
hdf.describe()


# <h4>Numeric Transformation</h4>
# From a quick look at the data we can see that the relationship between the in puts and the target are not on the same scale. Therfore, we need to do some simple math to correct these issues. 

# In[ ]:


#Transform the data inside the dataframe
hdf['avg_rooms'] = hdf.total_rooms/hdf.households
hdf['avg_bedrooms'] = hdf.total_bedrooms/hdf.households
hdf['avg_non_bedrooms'] = hdf.avg_rooms - hdf.avg_bedrooms
hdf['household_size'] = hdf.population/hdf.households
hdf['median_house_value_100000'] = hdf.median_house_value/100000


# In[ ]:


#Now let's check the values
hdf.describe()


# <ul><li><em>It looks like there are some outliers and unreal data points; in the dataset. We will take care of these later in the notebook.</em></li></ul>

# <h5>One Hot Encoding</h5> 
# We see that there is one object type variable. Let's explore this column more and encode the inputs that can be used in our model. 

# In[ ]:


#First let's have a look what categories are stored in this column. 
hdf.ocean_proximity.unique()


# <ul><li><em>This looks good. There are not too many categories for us to deal with in our model.</em></li></ul>

# In[ ]:


#Now we need to append our new dummy variables to the main data frame. 
dummy_cols = pd.get_dummies(hdf.ocean_proximity)
hdf = pd.concat([hdf, dummy_cols ],axis=1)


# <h4>Missing Data</h4>
# Look at the data set to see if there are missing data points. This will help fix some of the issues with data points in being unreasonable in the dataset.

# In[ ]:


#Save records with missing data in its own dataframe. 
miss_df = hdf[hdf.isnull().T.any().T]
hdf = hdf[~hdf.isnull().T.any().T]
print('There are {} missing records.'.format(str(len(miss_df))))


# In[ ]:


hdf.info()


# <h4>Exploratory Visualizations</h4>
# The best way to learn about a data set and find outliers / bad records is with plotting. This is esspecially true with spatial data where we know that all records should be in a specific geographic region. After we identify the outliers / bad points we can either correct, drop or account for the points in our model. 
# 
# <ol>
#     <h6>Here are the plots I will be builing below:</h6>
#     <li>Box Plots: Numeric Features</li>
#     <li>Bar Chart: Categorical Features</li>
#     <li>Correlation Matrix</li>
#     <li>Feature Ranking</li><ol>

# In[ ]:


import matplotlib.pyplot as plt


# <h5>Box Plots</h5>
# Use histograms to visualize the distribution of the data and the box and whisker plots visualize how many data points are extremes.

# In[ ]:


hdf.iloc[:,:3].boxplot(figsize=(10,5))
plt.title('Box and Whisker Plots for Housing Data')
plt.show()
hdf[['median_income', 'median_house_value_100000']].boxplot(figsize=(10,5))
plt.show()
hdf.iloc[:,-10:-6].boxplot(figsize=(10,5))
plt.show()


# In[ ]:


hdf.head()


# <h5>Bar Charts</h5>
# Plot histograms and barcharts to look at the distribution of the data. 

# In[ ]:


bar_df = hdf[['ocean_proximity', 'median_house_value']]
bar_df.columns = ['ocean_proximity', 'count']
bar_df.groupby('ocean_proximity').count().plot(kind='bar')
plt.title('Count of Geographic Categories')


#  Let's clean up our dataframe to only have variables that will be used in our modeling. 

# In[ ]:


hdf=hdf[['longitude', 'latitude', 'housing_median_age', 'population', 
         'median_income', 'avg_rooms','avg_bedrooms', 'avg_non_bedrooms', 'household_size', '<1H OCEAN',
        'INLAND', 'ISLAND','NEAR BAY', 'NEAR OCEAN', 'median_house_value']]


# Now we will build a histogram to observe the distribtuions in our data. 

# In[ ]:


hdf.hist(figsize=(20,15))
plt.tight_layout()


# <h5>Correlation Matrix</h5>
# Observe the correlations and plot

# In[ ]:


import seaborn as sea    


# In[ ]:


corr_matrix = hdf.corr()
fig, ax = plt.subplots(figsize=(15,15))
sea.heatmap(corr_matrix, annot= True)
plt.title('Correlation Matrix: California Housing Data')


# <h5>Feature Ranking</h5>
# From the correlation analysis, we can look to see which features are most correlated to the target. This will give us more insight into the data and the relationships that are being modeled. Remember causation does not equal correlation but we can still learn from the correlation. 

# In[ ]:


ft_list = abs(corr_matrix.iloc[:-1,-1]).sort_values()
ft_list


# In[ ]:


ft_list.plot(kind='barh', figsize=(30,15))
plt.title('Feature Importance Chart: California Housing Data', fontsize = 20)
plt.xlabel('Correlation', fontsize=18)
plt.ylabel('Variable', fontsize=18)


# <h6>Conclusions from Correlation Analysis</h6>
# From our analysis we can see that the most important variable is median_income of the neighborhood followed by if it is inland or near the ocean. Using some jugement here is the list of variables I will include in my models due to my analysis.
# <ol><li>median_income</li>
#     <li>inland</li>
#     <li>1h ocean</li>
#     <li>avg_rooms_non_bedroom</li>
#     <li>near bay</li>
#     <li>near ocean</li>
#     <li>housing_median_age</li>
# </ol>
# 
# You may ask why I did not include some of the other variables that were more highly correlated than some of my last choices. Take a look at the correlation matrix and you will see that those variables are highly correlated to variables I have already chosen. Linear regression has issues with multicolinearity but it is know that neural networks can as well. This is why I excluded these variables. 
# 
# Finally, let's have one look at the correlations to our target to see if the which direction the relationships should go (positive or negative). 

# In[ ]:


corr_matrix.iloc[:-1,-1]


# <h3>Data Processing for Modeling</h3>
# It's best to standardize or normalize our data when using neural networks and other machine learning techniques. By standardizing the data we are utilizing the z-score for each of the variable columns. This puts all of the data on the same scale and helps with the fitting/weighting of the neural network. We will also split our data into training and test in this section. 

# <h4>Outlier Treatment</h4>
# Now that we have visualized our data we have a couple of choices for dealing with the outliters. We can choose what an outlier is with two simple methods. One we can pull the points that looked out of place by the grahpics we completed or we can set up a general rule to identify the outliers. In this excercise, I will remove all points that outside of 3 standard devations from our data set on which we will be predicting. Here are some strategies for dealing with outliers. 
# <ol>
#     <li>Decide there are no outliers and use our current dataset.</li>
#     <li>Account for the outlier records with a dummy variable that is representative of the outling point.
#         <ul><em> Note: When we know there are outliers and do not want to lose the information they provide this is the best strategy</em></ul></li>
#     <li>Remove records that are outliers from the dataset and train the model. We do this because we view the records as potentially bad data or events that are not likely to happen again.</li>
# </ol>

# In[ ]:


#Let's look and see if there are still values that look unreal in the dataset. 
hdf.describe()


# The a record with an average number of non-bedrooms being 116.27 seems unreasonable let's look to see if there are a few records that do not look right. 

# In[ ]:


hdf.sort_values(by='avg_non_bedrooms', ascending=False).head(10)


# From the other data points, it does not seem that we have enough information to determine if this point is an outlier or not. Now lets look to see if any of the target points are outside 3 stds. Since we are unsure about the avg_non_bedrooms, I will run the check on this variable as well.

# In[ ]:


#Look at the average non-bedrooms variable for potential outliers. 
rom_out= hdf[~(np.abs(hdf.avg_non_bedrooms-hdf.avg_non_bedrooms.mean())
               <= (3*hdf.avg_non_bedrooms.std()))]
hdf_no_ouliers= hdf[(np.abs(hdf.avg_non_bedrooms-hdf.avg_non_bedrooms.mean())
                               <= (3*hdf.avg_non_bedrooms.std()))]
print('''There are possible {} outling points in the avg_non_bedrooms variable. Lets make an attempt at modeling the data before we take further action.'''.format(str(len(rom_out))))


# In[ ]:


rom_out= hdf_no_ouliers[~(np.abs(hdf_no_ouliers.household_size-hdf_no_ouliers.household_size.mean())
                          <= (3*hdf_no_ouliers.household_size.std()))]
hdf_no_ouliers = hdf_no_ouliers[(np.abs(hdf_no_ouliers.household_size-hdf_no_ouliers.household_size.mean())
                               <= (3*hdf_no_ouliers.household_size.std()))]
print('''There are possible {} outling points in the household_size variable. Lets make an attempt at modeling the data before
we take further action.'''.format(str(len(rom_out))))


# <h5>Stardardize the Data</h5>
# Here we are going to standardize the variables to use in our models. I realize that sklearn has a function to do this but I enjoy using my own. I like to see under the hood for this opereation. 

# In[ ]:


def stdize_data(df):
    mean = df.mean()
    std = df.std()
    std_df = (mean - df)/std
    return mean, std, std_df


# In[ ]:


def unstdize_data(npar, mean, std):
    new_values = ((npar*std)-mean)*-1
    return new_values


# In[ ]:


hdf_dummies = hdf[['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN']]


# In[ ]:


m, s, new_df = stdize_data(hdf[['median_income', 
                                'avg_non_bedrooms', 'housing_median_age', 
                                'avg_bedrooms', 'household_size', 'latitude', 'median_house_value' ]])


# In[ ]:


#Let's inspect the new values. 
new_df = pd.concat([hdf_dummies, new_df], axis=1)
new_df.head()


# <h5>Test Train Split</h5>
# Now let's split our data into train and test to train our models.

# In[ ]:


from sklearn.model_selection import train_test_split


# Frist, we need to create our arrays for our x and y variables

# In[ ]:


x= new_df.iloc[:,:-1].values
x.shape


# In[ ]:


y= new_df.iloc[:,-1].values
y.shape


# In[ ]:


y


# Now we create our training and test sets.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1234, test_size=.3)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# <h3>Modeling the Data</h3>
# Many libraries and functions can be utilized to build linear regressions or neural networks. In this excercise, we will be using keras for a model and sklearn for a model and evaluation metrics. 
# 
# <ol><h5>Model Types </h5>
#     <li>Single Layer Linear Model Keras</li>
#     <li>Sklearn Linear Regression</li>
#     </ol>

# In[ ]:


#Setting up the packages to build our models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD


# <h4>Single Layer Keras</h4>

# In[ ]:


def keras_model_lreg(var_cnt=2):
    model= Sequential()
    model.add(Dense(1, input_shape= (var_cnt,)))
    model.compile(SGD(lr=.0001), 'mean_squared_error')
    return model


# In[ ]:


reg= keras_model_lreg(var_cnt=10)
reg.fit(x_train, y_train, epochs = 30)


# In[ ]:


y_pred = reg.predict_classes(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


#Unstandardize the Data
y_test = unstdize_data(y_test, m[-1], s[-1])
y_pred = unstdize_data(y_pred, m[-1], s[-1])


# Let's check the diagnositcs of our model

# In[ ]:


rmse = np.sqrt(mean_squared_error(y_pred, y_test))
mpe  = np.mean((y_test- y_pred)/y_test*100)
mape = np.mean(abs(y_test- y_pred)/y_test*100)
me = np.mean((y_test- y_pred))


# In[ ]:


print('RMSE = {}'.format(rmse))
print('MPE = {}'.format(mpe))
print('MAPE = {}'.format(mape))
print('ME = {}'.format(me))


# Let's look at what the model predicted. 

# In[ ]:


pd.DataFrame(y_pred).drop_duplicates()


# We can see from this model that it does not work well. From further exploration I learned that this is because our model is probably too complex for what we are trying to predict. Let's try some sklearn models and see if they do better. I would love for you to comment if you see that I did something wrong. 

# <h4>Sklearn Linear Regression</h4>
# Now let's test our hypothesis that the Keras model we used was too complex for the dataset by attempting to build a linear regression that can do better than the keras model in sklearn. We will use only the variables we found in our feature ranking excercise. Finally, I will not do any tuning to my linear regression (further transformations of the variables). This will show us if a linear regression can fit the data better than our keras model. 
# 
# <em>Note: I tried only using these variables in the keras model and the results were worse than the final results.</em>

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


hdf_no_ouliers.head()


# In[ ]:


x = hdf_no_ouliers[['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN','median_income', 
                    'avg_non_bedrooms', 'housing_median_age', 'household_size' ]].values


# In[ ]:


y= hdf_no_ouliers.iloc[:,-1].values


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 1234, test_size=.3)


# In[ ]:


l_reg = LinearRegression()


# In[ ]:


l_reg.fit(x_train, y_train)


# In[ ]:


y_pred = l_reg.predict(x_test)


# In[ ]:


rmse = np.sqrt(mean_squared_error(y_pred, y_test))
mpe  = np.mean((y_test- y_pred)/y_test*100)
mape = np.mean(abs(y_test- y_pred)/y_test*100)
me = np.mean((y_test- y_pred))


# In[ ]:


print('RMSE = {}'.format(rmse))
print('MPE = {}'.format(mpe))
print('MAPE = {}'.format(mape))
print('ME = {}'.format(me))


# In[ ]:


y_pred = l_reg.predict(x_train)


# In[ ]:


rmse = np.sqrt(mean_squared_error(y_pred, y_train))
mpe  = np.mean((y_train- y_pred)/y_train*100)
mape = np.mean(abs(y_train- y_pred)/y_train*100)
me = np.mean((y_train- y_pred))


# In[ ]:


print('RMSE = {}'.format(rmse))
print('MPE = {}'.format(mpe))
print('MAPE = {}'.format(mape))
print('ME = {}'.format(me))


# We see that the Linear Regression Model beat the Keras model. There is still plenty of room for improvement in our model and maybe I will work on that in another kernal. I am going to experiment with Keras on a few other datasets but this was a fun start. 
# 
# If someone sees something that I could have done better with the Keras model I would apperciate the feedback. It was greatly frustrating that I could only get two outcomes. 

# <ol><h3> Sites and Courses Useful in Constructing This Notebook</h3>
#     <li>"Zero to Deep Learning" by Jose Portilla and Francesco Mosconi
#         <ul> Link: https://www.udemy.com/zero-to-deep-learning</ul></li>
# </ol>
#     

# In[ ]:




