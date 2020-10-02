#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, 
import seaborn as sns #for plotting 
import os


# **Question**
# 
# I just came back from a trip to Disneyland and the lines were horrendous even with fastpass and I left the park feeling a bit bitter. So, I wondered if hospitals may also have the same problem with patients having to wait for long periods of time just to see a doctor or a nurse. 
# 
# Wtih that said, I would be curious to see what are the qualities of a hospital that would result in poor timeliness of care?
# 
# But, first...
# Let's load in the dataset and print out the first 5 rows with the head() function to take a quick skim of the data to see if any bizarre data sticks out,

# In[2]:


df = pd.read_csv("../input/HospInfo.csv")
df.head()


# **Observations**
# Several things that stood out that are wortth checking/cleaning:
# * 1: Overall, the dataset is quite clean. Each hospital seems to be uniquely identifed by the ProviderID. It would be a good idea to double check if that's the case
# * 2: Right away, we see there are potential missing values, labeled 'Not Available', that should be considered NA. However, they are technically not read as such. This creates a problem when we want to pass in these values into a predictive model because it would consider these 'Not Available' as real data and not missing values that should be ignored. (Ex. Take a look at Hale County Hospital's hospital rating)
# * 3: Further note on the categorical data, each variable is written in two variables - one that contains the actual descriptive data and the other that contains additional explanation if the original value was not given or 'Not Available'' in this sense. It seems that the only explanation (so far) for a value that does not have any available data will state "Results are not available for this reporting...". For now, I would toss out any variables that have the words "footnote" to make our dataset smaller and easier to read because containing these variables results in redundancy. 
# 
# *For exploratory data analysis* There weren't any numerical variables (you can't add phone numbers together). There are PLENTY of categorical, specifically ordinal, variables that are present in the latter half of the dataset. We could make same bar charts to see if there is any association with timeliness of care.

# *Observation #1* Check if first observation is true - is Provider ID truly an unique identifier? The output below confirms this assumption, so it is safe to use this identifier moving forward. (we could also use hospital name as an identifier, but it may be possible that some hospitals have the same names)

# In[3]:


nrows = df.shape[0]
num_unique_hosp = df['Provider ID'].nunique()
print("Is Provider ID a unique identifier?: {0}".format(nrows == num_unique_hosp))


# *Observation #2*  Incorporate NAs. Note that NaN is now written as Hale County Hospital's rating. (NaN is equivalent to NA in R)
# 
# *Observation #3* Remove redundant columns for ease of reading.

# In[4]:


#Replace text with NaN that can be read as true missing value in Python
df = df.replace('Not Available', np.nan )

#Drop all columns whose name contains 'footnote'
cols_to_drop = list(df.filter(regex='footnote'))
df_clean = df[df.columns.drop(cols_to_drop)]

#print to see results
df_clean.head()


# **Exploratory Data Analysis**
# 
# Let's tackle the hospitals with long waiting times. Approximately, how many of these hospitals are in the dataset?

# In[5]:


#normalize = True gives the percentages of each value instead of frequencies
df_clean['Timeliness of care national comparison'].value_counts(normalize=True)


#  From the output, we see a little over a quarter of hospitals have below national average. This may be a small representation, but keep in mind that this is a quarter of ALL hospitals in the US in the CMS system. How many does this mean? Let's find out.

# In[6]:


print("Out of {0} total hospitals, how many have below average wait times? {1} hospitals".format(nrows, round(nrows * 0.255)))


# Roughly 1200+ hospitals have longer wait times than the national average. This is a problem since longer waiting times are correlated with patient dissatisifaction. Having patients wait longer could also induce more stress and anxiety (https://www.bmj.com/content/342/bmj.d2983) 
# 
# Let's explore other variables in the dataset to see which features are most correlated with these hospitals. In the end, we could use unsupervised learning and predictive models to confirm these attributes. Once we have discovered these features, we could make recomendations for these hospitals to allocate their resources more effectively that will, hopefully, improving timeliness of care. 

# *NOTE*
# 
# Sorry, guys! I'm limited in time and couldn't figure out how to visualize what I needed in python (in R I could because that's the language I'm much more familiar with). So, I've published my visualizations through tableau dashboard that you can look at here. https://public.tableau.com/profile/chelsea.lee#!/vizhome/KaggleHospitalGeneralInformation/LocatingHospitals
# 
# Each tab illustrates the counts of hospitals per category in each of the variables. I've taken out hospitals that have "same as national average" so it is easier to compare below vs above average hospitals. The barplots are stacked so that the green portion corresponds to hospitals with above average timely care and red corresponds to the contrary. 
# 
# From these charts, we can see that some variables show no apparent association that separates fast hospitals from slow hospitals. These include:
# * Hospital type
# * Emergency Services
# * Meets criteria for meaningful use of EHRs
# * Effectiveness of care national comparison
# 
# But, there are variables that do highlight differences among these hospitals in terms of timeliness of care. These include:
# * Hospital Rating - Hospitals with ratings of 3 and 4 are correlated with "faster" hospitals.
# * Hospital Ownership - Government - Hospital District or Authority and Proprietary hospitals are correlated with  "faster" hospitals.
# * Safety Care - Below average safety is correlated with "slower" hospitals. 
# * Readmissions - (Interesting) Below average readmissions is correlated with "slower" hospitals.
#      Fewer people being readmitted is related to slower hospitals..
# * Patient Experience - Below average patient experience is correlated with "slower" hospitals. 
#      This makes sense. You're more likely to be disastisfied with care of service if you had to wait too long.
# * Efficient use of Medical Imaging - (interesting) Above average use of medical imaging is correlated with "slower" average hospitals. 
#      Why would some "slower" hospitals have better use of medical imaging? Could it be that these hospitals are just more careful and sophisticated with their medical imaging tools?

# **Data Preprocessing**
# Now, before we go into modeling, we should do clean our data by dealing with:
# * missing values in both our data and target variable ('Timeliness of care national comparison')
# * text data that must be converted into a format the XGBoost can read, which are categorical variables

# *Missing Values*
# 
# Based on our visualizations, there are variables that have quite a lot of missing data. From the output below, we see that two variables have a little less than half of their values missing: Safety of care (~45%) and efficient use of medical imaging (~42%). Since they're less than half, their values can still be useful for modeling, so I'll keep them here. XGboost is also known to be robust against having missing values. 

# In[25]:


#Calculate percentage of missing data in each column
df_clean.isnull().mean().sort_values(ascending=False)


# There were missing values in our target variable: 'Timeliness of care national comparison' so let's remove all rows that contain those. 
# 
# From this process, we've dropped approximately 26% of the rows. We haven't lost much information, so this is good news
# 

# In[27]:


#store the number of rows before (b) dropping
num_rows_b = df_clean.shape[0]

df_clean = df_clean.loc[df_clean["Timeliness of care national comparison"].notnull(), :]

#check if there is no missnig data in target variable
print("% of missing data in target variable after cleaning: {:.0%}"      .format(df_clean["Timeliness of care national comparison"].isnull().mean()))

#store the number of rows after (a) dropping
num_rows_a = df_clean.shape[0]

#Show the change in number of rows
print("# of rows before dropping NAs: {0}\n# of rows after dropping NAs: {1}"      .format(num_rows_b, num_rows_a))


# Dropping more unnecessary variables in order to be more efficient and to not confuse the model.

# In[73]:


#Remove Hospital Name, Address, City, State, Zip Code, County Name, Phone Number, and Location 
#Keep Provider ID for key later on so that we could pull in other information if we want to.
df_clean = df_clean.drop([
    "Hospital Name", "Address", "City", 
    "State", "ZIP Code", "Phone Number",
    "County Name", "Location"
], axis =1)

#See if values that are categorical are truly categorical, bools as truly bool and int as ints
df_clean.dtypes


# In[15]:


#Categorical variables are correctly casted as object type
#Emergency Services is bool but Meets criteria for meaningful use of EHR is not. Let's convert this to bool
df_clean['Meets criteria for meaningful use of EHRs'] = df_clean['Meets criteria for meaningful use of EHRs'].astype(bool)

#hospital overall rating should be numerical type (int doesn't accept missing values, so conver to float type)
df_clean['Hospital overall rating'] = df_clean['Hospital overall rating'].astype(float)

df_clean.dtypes


# *Creation of Categorical and Dummy Variables*
# 
# Since all of our data has text data, let's convert these values to ordinal values (categorical values with an order to them. For example, "low", "medium" and "high"  can be represented numerically by 1, 2, 3. where "high" (3)  is greater than "low" (1).)
# 
# We could use the factorize() function from Pandas library or use LabelEncoder from the Scikit library which maps categorical values into integers. You've probably heard or seen other Kaggle members use one hot encoding, which takes a step further from LabelEncoder by taking columns of integers and encode them as dummy variables. In this case, there isn't a need to do this process, because most of the categorical data that I will be feeding into the model is ordinal in nature.  I have read that one hot encoding is necessary for true categorical data without an order to them, because otherwise this will create problems for your ML algorithm by making it learn the data incorrectly. You can read more about this here under "Why Use a One Hot Encoding?" : https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/ 
# 
# In summary, these variables will be converted into the following format as follows:
# * Dummy Variables for nominal data: Hospital Type and Hospital Ownership
# * Categorical variables(factorization) for ordinal data: Emergency Services, Meets criteria for meaningful use of EHRs, the remaining variables with "national comparison" 
# * Categorical variables for boolean data: Emergency Services and Meets criteria for meaningful use of EHRs
# 
# *Note: Hospital overall rating is already in factorized form, so we can leave this alone and convert back to integers later on*. 
#         
# (Earlier, I did LabelEncoder mistakenly on the entire dataset. You can explore this code at the very bottom of this kernel if you like to check it out)

# In[75]:


#Create dummy variables for Hospital Type and Hospital Ownership and save into dv 
dv = pd.get_dummies(df_clean[['Hospital Type', 'Hospital Ownership']] )
dv.head()

#drop old columns and concatenate new dummy variables
df_clean = df_clean.drop(['Hospital Type', 'Hospital Ownership'], axis=1)
df_clean = pd.concat([df_clean, dv], axis=1)

#print head to check results (they're appended to the end)
df_clean.head()
#Remember that Hospital Type and Hospital Ownership did NOT have missing data from the original data.


# In[76]:


#create list of columns to convert to ordinal
    # only modify variables that have "national compmarison" in naming
ordinal_col = list(df_clean.filter(regex="national comparison"))

#Create customized mapper to factorize variables that are ordinal nature
mapper = {
    'Below the national average' : 0,
    'Same as the national average' : 1, 
    'Above the national average' : 2
}
for col in ordinal_col:
    df_clean.loc[:, df_clean.columns == col]= df_clean.loc[:, df_clean.columns == col].replace(mapper)

#print results. 
df_clean.head() 


# In[77]:


#Factorize Emergency and Meets criteria for meaningful use of EHRs into booleans
    #true = 1 and False = 0
bool_cols = ['Emergency Services', 'Meets criteria for meaningful use of EHRs']

df_clean[bool_cols] = (df_clean[bool_cols] == True).astype(int)

#print head to see results
df_clean.head()


# **Machine Learning**
# Now, let's try to predict which hospitals have below average wait times using supervised learning (I don't really have much hope for this since the features don't provide enough information...but let's give it a go!). So, our target variable is the "Timeliness of care national comparison". For personal preference, I will experiment with XGBoost as it is a widely popular supervised learning model. We have more samples than features, so using this method is appropriate for this scenario. As with any supervised learning, we need our data matrix, X, and vector of target values, y. 
# 
# Our y will be the "Timeliness of care national comparison" 
# 
# Moving on, let's split the data into train/test splits where we will train the model using the training data and evaluate the performance of our model on the testing data (It's importnt to note that we will not touch this until the very last step of testing. This is crucial to understanding how well the model will perform on unseen data. It would make no sense to train on both training and testing data, AND THEN predict on the testing data...it would yield 100% accuracy since we're predicting on what we already know.) 

# In[17]:


y = df_clean.pop("Timeliness of care national comparison")
X = df_clean

#randomly split into training and testing data. Let's aside 20% of our data for testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Compare dimensions:
#Remember that after we dropped rows from earlier, there were 3,546 rows
print("Original X: {0}, Original y: {1}".format(X.shape, y.shape))
print("X Train: {0}, y train: {1}".format(X_train.shape, y_train.shape))
print("X Train: {0}, y test: {1}".format(X_test.shape, y_test.shape))


# In[79]:


#Now remove provider ID after we have split into train/test
X_train_id = X_train.pop("Provider ID")
X_test_id = X_test.pop("Provider ID")


# In[18]:


import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from xgboost import plot_importance

#Instantiate XGB classifier model
xgb = XGBClassifier(seed = 123)

# fit model no training data
xgb.fit(X_train, y_train)

#Predict the lables of the test test
preds = xgb.predict(X_test)

# Compute the accuracy: accuracy

accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: {:.2f}%".format(accuracy * 100))


# In[83]:


#plot feature importance graph to see which features contribute to predicting the outcomes
plot_importance(xgb)
plt.show()


# In[84]:


#subset features there are "fairly importnat" relate to other features
subset = [
    "Hospital overall rating", "Safety of care national comparison",
    "Efficient use of medical imaging national comparison", 
    "Patient experience national comparison", "Mortality national comparison", 
    "Effectiveness of care national comparison", "Readmission national comparison",
    "Hospital Ownership_Proprietary"
]
X_train = X_train[subset]
X_test = X_test[subset]

xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)

accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: {:.2f}%".format(accuracy * 100))


# With an accuracy of ~52%, is it only slightly better than random chance. Let's tune a few of the important hyperparameters to see if we can increase our accuracy. We'll use RandomSearchCV to pick the optimal configuration. In a nutshell, it randomly picks a value for each hyperparameter from the ranges given, then performs k crossfold validation using these parameters. It then averages the testing scores. Whichever set of hyperparameters gives the best 'scoring' is the "best" model. (You can check out GridSearch if you like, but I prefer to save some time here) 
# 
# Why not just split the train set into dev/train? Well, we don't have significant amounts of data, so it will not take much compile time to do CV. 
# 
# However, it is best to not simply train only on the entire training set. From learning theory, a validation accuracy rate is a much more robust measure of testing accuracy rather than training accuracy~ 

# In[85]:


from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 
xgb_param_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'n_estimators': np.arange(100, 400, 10),
    'max_depth': np.arange(2, 11)
}

# Perform random search with scoring metric negative MSE. 
#Perform 5 fold CV (this is arbitrary)
randomized_mse = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_grid, 
                                    scoring = "accuracy",n_iter=10, cv=5, verbose=1)


# Fit randomized_mse to the data
randomized_mse.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

#Predict the lables of the test test
preds = randomized_mse.predict(X_test)

# Compute the accuracy: accuracy

accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: {:.2f}%".format(accuracy * 100))


# ![](http://)-facepalm- Accuracy still very low..barely better than random chance

# That is it! Predictions are considerably bad...This could be because most of the variables provided don't distinguish slow hospitals from the fast ones, so the model cannot truly learn from the data (since there weren't much patterns to learn from!). More variables would definitely help to improve its predicting power.
# 
# As a side note, you can tell from the feature importance graph that the most important features that helped predict hospital's quality of timely care matches what we've seen in our exploratory data analysis process.
# 
# If I were to explore this dataset further, I would answer these questions:
# 1. Which city should you live in that has better hospitals? By better, I mean 
#  * Lower readmissions 
#  * Shorter waiting periods
#  * Lower mortality rates
# 2. Given a certain disease, which hospital is best for a patient based on shorter waiting period time better mortality rates, and proximity to hospital based on patient's location.
# 3. I assume that cities have the best hospitals because cites tend to attract better talent. Is this true?
# 4. If assumption 2 isn't correct, then what types of cities do the best hospitals occur in? Large, mid-sized, or small? Do they differ by geographical regions? (west ocast, east coast, mid-west)
# 
# Let me know what you have found down in the comments below. I'll love to know them :) And, thanks for checking out my kernel! (Please upvote this kernel if you've learned a trick or two for support. :) )
# 
# --------------------
# Below contains my work using LabelEncoder if you wanted to explore this.

# In[ ]:


"""
#Import necessary functions
from sklearn.preprocessing import LabelEncoder

#Create a vector of booleans for categorical columns
categorical_mask = (df_clean.dtypes == object)

#Subset list of categorical columns
categorical_columns = df_clean.columns[categorical_mask].tolist()

#Print the head of the categorical columns (should be the same right after we dropped the variables from the first line of code of this chunk)
df_clean[categorical_columns].head()
#Instantiate LabelEncoder Object
le = LabelEncoder()

#Create empty list to store all mappings. This essentially serves as our "data dictionaries"
cat_column_mappings = []

#df_clean = df_clean.fillna("99")

#Loop through each of the categorical columns to convert to discrete numerical values. 
    #At the same time, create the dictionary and append back to cat_column_mappings
for col in categorical_columns:
    df_clean[col] = le.fit_transform(df_clean[col])
    
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    cat_column_mappings.append(pd.DataFrame(sorted(le_name_mapping.items()), columns = [col, "Encoded Labels"]))

#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#le_name_mapping
#cat_column_mappings

#print to see results from LabelEncoder
df_clean[categorical_columns].head()

#Seems the labels of overall ratings have shifted backwards by one. Let's refer to the fourth index of the data dictionary vector to double check
cat_column_mappings[4] # IT is! Quite confusing to read to be honest. Perhaps another mapping would do?
"""

