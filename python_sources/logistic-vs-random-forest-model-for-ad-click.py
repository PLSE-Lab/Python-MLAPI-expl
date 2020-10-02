#!/usr/bin/env python
# coding: utf-8

#  <h1><center>Ad Click Prediction</center></h1>

# # Introduction
# 
# The goal of the project is to __Predict who is likely going to click on the Ad__ on a website based on the features of a user. Following are the features involved in this dataset which is obtained from Kaggle.
# 
# |           Feature               |                  Description                           |
# |---------------------------------|--------------------------------------------------------|
# |1. __Daily Time Spent on a Site__   | Time spent by the user on a site in minutes.        |
# |2. __Age__                          | Customer's age in terms of years.                   |
# |3. __Area Income__                  | Average income of geographical area of consumer.    |
# |4. __Daily Internet Usage__         | Avgerage minutes in a day consumer is on the internet.|
# |5. __Ad Topic Line__                | Headline of the advertisement.                      | 
# |6. __City__                         | City of the consumer.                               |
# |7. __Male__                         | Whether or not a consumer was male.                 |
# |8. __Country__                      | Country of the consumer.                            |
# |9. __Timestamp__                    | Time at which user clicked on an Ad or the closed window.|
# |10. __Clicked on Ad__               | 0 or 1 is indicated clicking on an Ad.              |
# 
# 
# 
# This notebook will contain exploratory data analysis along with classification models related to this project. 
# 
# Steps involved in this Notebook
# 
# - [Getting to know about the Data](#Examine-the-data)
# - [Extract New features](#Extracting-Datetime-Variables)
# - [Check distribution of target variable](#Visualize-Target-Variable )
# - [Understand Relationship between variables](#Distribution-and-Relationship-Between-Variables)
# - [Identifying Potential Outliers](#Identifying-Potential-Outliers-using-IQR)
# - [Building a basic model](#Basic-model-building-based-on-the-actual-data)
# - [feature engineering](#Feature-Engineering)
# - [Building Logistic Regression Model](#Building-Logistic-Regression-Model )
# - [Random Forest Model](#Random-Forest-Model)
# - [Models Performances on Test Data](#Test-Models-Performance)
# - [Feature Importances](#Random Forest Feature Importances)
# 

# # Load Libraries

# In[ ]:


import numpy as np                    # Linear Algebra
import pandas as pd                   # Data processing 
import matplotlib.pyplot as plt       # Visualizations
import seaborn as sns                 # Visualizations
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix 
import warnings                       # Hide warning messages
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data 

# In[ ]:


# Reading the file 
df = pd.read_csv("../input/advertising.csv") 


# # Examine the data

# In[ ]:


df.head(10) # Checking the 1st 10 rows of the data


# # Data type and length of the variables

# In[ ]:


df.info() # gives the information about the data


# # Duplicates Checkup

# In[ ]:


df.duplicated().sum() # displays duplicate records


# # Numerical and Categorical Variables Identification

# In[ ]:


df.columns # displays column names


# In[ ]:


df.select_dtypes(include = ['object']).columns # Displays categorical variables which are detected by python 


# In[ ]:


# Assigning columns as numerical variables
numeric_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage' ]


# In[ ]:


# Assigning columns as categorical variables
Categorical_cols = [ 'Ad Topic Line', 'City', 'Male', 'Country', 'Clicked on Ad' ]


# # Summarizing Numerical Variables

# In[ ]:


df[numeric_cols].describe()
# Decribe method is used to give statistical information on the numerical columns


# As the __mean__ and __median__(50% percentile) are very similar to each other which indicates that our data is not skewed and we do not require any data transformations.We shall confirm this by visualizing as well.

# # Summarizing Categorical Variables

# In[ ]:


df[Categorical_cols].describe(include = ['O'])
# Decribe method is used to give statistical information on the categorical columns


# As we have many different cities (__Unique__) and also not many people belonging to a same city(__freq__). So, it probably means that this feature is having no or very less predictive power. However we have less diversity with country feature so we have to further investigate it.

# ### Investing Country Variable

# In[ ]:


pd.crosstab(df['Country'], df['Clicked on Ad']).sort_values(1,0, ascending = False).head(10)


# In[ ]:


pd.crosstab(index=df['Country'],columns='count').sort_values(['count'], ascending=False).head(10)


# It seems that users are from all over the world with maximum from france and czech republic with a count of 9 each.

# # Check for Missing Values

# In[ ]:


df.isnull().sum() # Number of missing values in each column


# # Extracting Datetime Variables
# 
# Utilizing timestamp feature to better understand the pattern when a user is clicking on a ad.

# In[ ]:


# Extract datetime variables using timestamp column
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
# Converting timestamp column into datatime object in order to extract new features
df['Month'] = df['Timestamp'].dt.month 
# Creates a new column called Month
df['Day'] = df['Timestamp'].dt.day     
# Creates a new column called Day
df['Hour'] = df['Timestamp'].dt.hour   
# Creates a new column called Hour
df["Weekday"] = df['Timestamp'].dt.dayofweek 
# Creates a new column called Weekday with sunday as 6 and monday as 0
# Other way to create a weekday column
#df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday()) # Monday 0 .. sunday 6
# Dropping timestamp column to avoid redundancy
df = df.drop(['Timestamp'], axis=1) # deleting timestamp


# In[ ]:


df.head() # verifying if the variables are added to our main data frame


# # Visualize Target Variable 

# In[ ]:


# Visualizing target variable Clicked on Ad
plt.figure(figsize = (14, 6)) 
plt.subplot(1,2,1)            
sns.countplot(x = 'Clicked on Ad', data = df)
plt.subplot(1,2,2)
sns.distplot(df["Clicked on Ad"], bins = 20)
plt.show()


# So from the plot we can see that the number of users who click on a ad and who do not are equal in numbers i.e 500,  that makes it very interesting.

# In[ ]:


# Jointplot of daily time spent on site and age
sns.jointplot(x = "Age", y= "Daily Time Spent on Site", data = df) 


# We can see that more people aged between 30 to 40 are spending more time on site daily.

# # Distribution and Relationship Between Variables 

# In[ ]:


# Creating a pairplot with hue defined by Clicked on Ad column
sns.pairplot(df, hue = 'Clicked on Ad', vars = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'], palette = 'husl')


# Pairplot represents the relationship between our target feature/variable and explanatory variables. It provides the possible direction of the relationship between the variables. We can see that people who spend less time on site and have less income and are aged more relatively are tend to click on ad. 

# In[ ]:


plots = ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']
for i in plots:
    plt.figure(figsize = (14, 6))
    plt.subplot(1,2,1)
    sns.boxplot(df[i])
    plt.subplot(1,2,2)
    sns.distplot(df[i],bins= 20)    
    plt.title(i)    
    plt.show()


# We can clearly see that daily interent usage and daily time spent on a site has 2 peaks (Bi-model in statistical terms). It indicates that there are two different groups present in our data. We dont expect the users to be normally distributed as there are people who spend more time on internet/website and people who spend less time. Some regularly use the website and some less often so they are perfectly alright.  

# In[ ]:


print('oldest person was of:', df['Age'].max(), 'Years')
print('Youngest person was of:', df['Age'].min(), 'Years')
print('Average age was of:', df['Age'].mean(), 'Years')


# In[ ]:


f,ax=plt.subplots(2,2, figsize=(20,10))
sns.violinplot("Male","Age", hue= "Clicked on Ad", data=df,ax=ax[0,0],palette="spring")
ax[0,0].set_title('Gender and Age vs Clicked on Ad or not')
ax[0,0].set_yticks(range(0,80,10))
sns.violinplot("Weekday","Age", hue="Clicked on Ad", data=df,ax=ax[0,1],palette="summer")
ax[0,1].set_title('Weekday and Age vs Clicked on Ad or not')
ax[0,1].set_yticks(range(0,90,10))
sns.violinplot("Male","Daily Time Spent on Site", hue="Clicked on Ad", data=df,ax=ax[1,0],palette="autumn")
ax[1,0].set_title('Gender and Daily time spent vs (Clicked on ad or not)')
#ax[1,0].set_yticks(range(0,120,10))
sns.violinplot("Weekday","Daily Time Spent on Site", hue="Clicked on Ad", data=df,ax=ax[1,1],palette="winter")
ax[1,1].set_title('Weekday and Daily time spent vs (Clicked on ad or not)')
#ax[1,1].set_yticks(range(0,120,10))
plt.show()


# # Correlation Between Variables

# In[ ]:


fig = plt.figure(figsize = (12,10))
sns.heatmap(df.corr(), cmap='Blues', annot = True) # Degree of relationship i.e correlation using heatmap


# Heatmap gives us better understanding of relationship between each feature. Correlation is measured between __-1__ and __1__. Higher the absolute value, higher is the degree of correlation between the variables. We expect daily internet usage and daily time spent on site to be more correlated with our target variable. Also, none of our explantory variables seems to correlate with each other which indicates there is no collinearity in our data. 

# # Extracted Features Visualizations

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
df['Month'][df['Clicked on Ad']==1].value_counts().sort_index().plot(ax=ax[0])
ax[0].set_title('Months Vs Clicks')
ax[0].set_ylabel('Count of Clicks')
pd.crosstab(df["Clicked on Ad"], df["Month"]).T.plot(kind = 'Bar',ax=ax[1])
#df.groupby(['Month'])['Clicked on Ad'].sum() # alternative code
plt.tight_layout()
plt.show()


# Line chart showing the count of clicks for each month. Grouped bar chart shows distribution of target variable across 7 months. 2nd Month seems to be the best for clicking on a Ad.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab(df["Clicked on Ad"], df["Hour"]).T.plot(style = [], ax = ax[0])
pd.pivot_table(df, index = ['Weekday'], values = ['Clicked on Ad'],aggfunc= np.sum).plot(kind = 'Bar', ax=ax[1]) # 0 - Monday
plt.tight_layout()
plt.show()


# Line chart here indicates that user tends to click on a Ad later in a day or probably early in the morning. It is expected based on the age feature that most people are working so it seems appropriate as they either find time early or late in the day. Also sunday seems to be effective for clicking on a ad from the bar chart.

# ## Clicked Vs Not Clicked

# In[ ]:


df.groupby('Clicked on Ad')['Clicked on Ad', 'Daily Time Spent on Site', 'Age', 'Area Income', 
                            'Daily Internet Usage'].mean()


# Average profile of a user who will click on a ad or not.

# In[ ]:


df.groupby(['Male','Clicked on Ad'])['Clicked on Ad'].count().unstack()


# Distribution of clicks by gender. It seems that more number of females have clicked on ad.

# In[ ]:


hdf = pd.pivot_table(df, index = ['Hour'], columns = ['Male'], values = ['Clicked on Ad'], 
                     aggfunc= np.sum).rename(columns = {'Clicked on Ad':'Clicked'})

cm = sns.light_palette("green", as_cmap=True)
hdf.style.background_gradient(cmap=cm)  # Sums all 1's i.e clicked for each hour


# Distribution by each hour and by gender. Overall females tend to click on a Ad more often than males.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
sns.set_style('whitegrid')
sns.countplot(x='Male',hue='Clicked on Ad',data=df,palette='bwr', ax = ax[0]) # Overall distribution of Males and females count
table = pd.crosstab(df['Weekday'],df['Clicked on Ad'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, ax=ax[1], grid = False) # 0 - Monday
ax[1].set_title('Stacked Bar Chart of Weekday vs Clicked')
ax[1].set_ylabel('Proportion by Day')
ax[1].set_xlabel('Weekday')
plt.tight_layout()
plt.show()


# From the stacked bar chart it seems that there more chances of user clicking on a ad if its a thursday!

# In[ ]:


sns.factorplot(x="Weekday", y="Age", col="Clicked on Ad", data=df, kind="box",size=5, aspect=2.0) 


# Comparison of users who have clicked on ad or not in terms of age and weekday. It is clear that people with higher age tend to click on a ad.

# In[ ]:


sns.factorplot('Month', 'Clicked on Ad', hue='Male', data = df)
plt.show()


# # Identifying Potential Outliers using IQR

# In[ ]:


for i in numeric_cols:
    stat = df[i].describe()
    print(stat)
    IQR = stat['75%'] - stat['25%']
    upper = stat['75%'] + 1.5 * IQR
    lower = stat['25%'] - 1.5 * IQR
    print('The upper and lower bounds for suspected outliers are {} and {}.'.format(upper, lower))


# # Basic model building based on the actual data

# In[ ]:


# Importing train_test_split from sklearn.model_selection family
from sklearn.model_selection import train_test_split


# In[ ]:


# Assigning Numerical columns to X & y only as model can only take numbers
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = df['Clicked on Ad']


# In[ ]:


# Splitting the data into train & test sets 
# test_size is % of data that we want to allocate & random_state ensures a specific set of random splits on our data because 
#this train test split is going to occur randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
# We dont have to use stratify method in train_tst_split to handle class distribution as its not imbalanced and does contain equal number of classes i.e 1's and 0's
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # Building a Basic Model

# In[ ]:


# Import LogisticRegression from sklearn.linear_model family
from sklearn.linear_model import LogisticRegression


# In[ ]:


# Instantiate an instance of the linear regression model (Creating a linear regression object)
logreg = LogisticRegression()
# Fit the model on training data using a fit method
model = logreg.fit(X_train,y_train)
model


# # Predictions

# In[ ]:


# The predict method just takes X_test as a parameter, which means it just takes the features to draw predictions
predictions = logreg.predict(X_test)
# Below are the results of predicted click on Ads
predictions[0:20]


# # Performance Metrics

# Now we need to see how far our predictions met the actual test data (y_test) by performing evaluations using classification report & confusion matrix on the target variable and the predictions.
# 
# **confusion matrix** is used to evaluate the model behavior from a matrix. Below is how a confusion matrix looks like:
# 
#                     Predicted No   Predicted Yes
#     Actual No          TN                 FP 
# 
#     Actual Yes         FN                 TP   
# 
# TP- True Positive 	TN- True Negative 
# FP- False Positive 	FN- False Negative 
# 
# True Positive is the proportion of positives that are correctly identified. Similarly, True Negative is the proportion of negatives that are correctly identified. False Positive is the condition where we predict a result that is actually doesn't fulfill. Similarly, False Negative is the condition where the prediction failed, when it was actually successful.
# 
# If we want to calculate any specific value, we can do it from confusion matrix directly.
# 
# **classification_report** will basically tell us the precision, recall value's accuracy, f1 score & support. This way we don't have to read it ourself from a confusion matrix.
# 
# **precision** is the fraction of retrieved values that are relevant to the data. The precision is the ratio of tp / (tp + fp).
# 
# **recall** is the fraction of successfully retrieved values that are relevant to the data. The recall is the ratio of          tp / (tp + fn). 
# 
# **f1-score** is the harmonic mean of precision and recall. Where an fscore reaches its best value at 1 and worst score at 0.
# 
# **support** is the number of occurrences of each class in y_test.

# In[ ]:


# Importing classification_report from sklearn.metrics family
from sklearn.metrics import classification_report

# Printing classification_report to see the results
print(classification_report(y_test, predictions))


# In[ ]:


# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))


# ## Results for Basic Model

# The results from evaluation are as follows:
# 
# **Confusion Matrix:** 
# 
# The users that are predicted to click on commercials and the actually clicked users were 144, the people who were predicted not to click on the commercials and actually did not click on them were 156.
# 
# The people who were predicted to click on commercial and actually did not click on them are 6, and the users who were not predicted to click on the commercials and actually clicked on them are 24.
# 
# We have only a few mislabelled points which is not bad from the given size of the dataset.
# 
# **Classification Report:**
# 
# From the report obtained, the precision & recall are 0.91 which depicts the predicted values are 91% accurate. Hence the probability that the user can click on the commercial is 0.91 which is a good precision value to get a good model.  

# # Feature Engineering

# In[ ]:


new_df = df.copy() # just to keep the original dataframe unchanged


# In[ ]:


# Creating pairplot to check effect of datetime variables on target variable (variables which were created)
pp = sns.pairplot(new_df, hue= 'Clicked on Ad', vars = ['Month', 'Day', 'Hour', 'Weekday'], palette= 'husl')


# There dont seems to be any effect of month, day, weekday and hour on the target variable. 

# In[ ]:


# Dummy encoding on Month column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Month'], prefix='Month')], axis=1) 
# Dummy encoding on weekday column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Weekday'], prefix='Weekday')], axis=1)


# In[ ]:


# Creating buckets for hour columns based on EDA part
new_df['Hour_bins'] = pd.cut(new_df['Hour'], bins = [0, 5, 11, 17, 23], 
                        labels = ['Hour_0-5', 'Hour_6-11', 'Hour_12-17', 'Hour_18-23'], include_lowest= True)


# In[ ]:


# Dummy encoding on Hour_bins column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Hour_bins'], prefix='Hour')], axis=1)


# In[ ]:


# Feature engineering on Age column
plt.figure(figsize=(25,10))
sns.barplot(new_df['Age'],df['Clicked on Ad'], ci=None)
plt.xticks(rotation=90)


# In[ ]:


# checking bins
limit_1 = 18
limit_2 = 35

x_limit_1 = np.size(df[df['Age'] < limit_1]['Age'].unique())
x_limit_2 = np.size(df[df['Age'] < limit_2]['Age'].unique())

plt.figure(figsize=(15,10))
#sns.barplot(df['age'],df['survival_7_years'], ci=None)
sns.countplot('Age',hue='Clicked on Ad',data=df)
plt.axvspan(-1, x_limit_1, alpha=0.25, color='green')
plt.axvspan(x_limit_1, x_limit_2, alpha=0.25, color='red')
plt.axvspan(x_limit_2, 50, alpha=0.25, color='yellow')

plt.xticks(rotation=90)


# In[ ]:


# Creating Bins on Age column based on above plots
new_df['Age_bins'] = pd.cut(new_df['Age'], bins=[0, 18, 30, 45, 70], labels=['Young','Adult','Mid', 'Elder'])


# In[ ]:


sns.countplot('Age_bins',hue='Clicked on Ad',data= new_df) # Verifying the bins by checking the count


# In[ ]:


# Dummy encoding on Age column
new_df = pd.concat([new_df, pd.get_dummies(new_df['Age_bins'], prefix='Age')], axis=1) 


# In[ ]:


# Dummy encoding on Column column based on EDA
new_df = pd.concat([new_df, pd.get_dummies(new_df['Country'], prefix='Country')], axis=1)


# In[ ]:


# Remove redundant and no predictive power features
new_df.drop(['Country', 'Ad Topic Line', 'City', 'Day', 'Month', 'Weekday', 
             'Hour', 'Hour_bins', 'Age', 'Age_bins'], axis = 1, inplace = True)
new_df.head() # Checking the final dataframe


# # Building Logistic Regression Model

# In[ ]:


X = new_df.drop(['Clicked on Ad'],1)
y = new_df['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


# Standarizing the features
from  sklearn.preprocessing  import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


import  statsmodels.api  as sm
from scipy import stats

X2   = sm.add_constant(X_train_std)
est  = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


# We can see that the feature __Male(Gender)__ does not contribute to the model (i.e., see x4) so we can actually remove that variable from our model. After removing the variable if the __Adjusted R-squared has not changed__ from the previous model. Then we could conclude that the feature indeed was not contributing to the model. Looks like the contributing features for the model are:
# 
# - Daily Time Spent on site
# - Daily Internet Usage
# - Age
# - Country
# - Area income

# In[ ]:


# Applying logistic regression model to training data
lr = LogisticRegression(penalty="l2", C= 0.1, random_state=42)
lr.fit(X_train_std, y_train)
# Predict using model
lr_training_pred = lr.predict(X_train_std)
lr_training_prediction = accuracy_score(y_train, lr_training_pred)

print( "Accuracy of Logistic regression training set:",   round(lr_training_prediction,3))


# In[ ]:


#Creating K fold Cross-validation 
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(lr, # model
                         X_train_std, # Feature matrix
                         y_train, # Target vector
                         cv=kf, # Cross-validation technique
                         scoring="accuracy", # Loss function
                         n_jobs=-1) # Use all CPU scores
print('10 fold CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[ ]:


from sklearn.model_selection import cross_val_predict
print('The cross validated score for Logistic Regression Classifier is:',round(scores.mean()*100,2))
y_pred = cross_val_predict(lr,X_train_std,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="winter")
plt.title('Confusion_matrix', y=1.05, size=15)


# # Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', n_estimators=400,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=42,n_jobs=-1)
rf.fit(X_train_std,y_train)
# Predict using model
rf_training_pred = rf.predict(X_train_std)
rf_training_prediction = accuracy_score(y_train, rf_training_pred)

print("Accuracy of Random Forest training set:",   round(rf_training_prediction,3))


# In[ ]:


kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, # model
                         X_train_std, # Feature matrix
                         y_train, # Target vector
                         cv=kf, # Cross-validation technique
                         scoring="accuracy", # Loss function
                         n_jobs=-1) # Use all CPU scores
print('10 fold CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# In[ ]:


from sklearn.model_selection import cross_val_predict
print('The cross validated score for Random Forest Classifier is:',round(scores.mean()*100,2))
y_pred = cross_val_predict(rf,X_train_std,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="winter")
plt.title('Confusion_matrix', y=1.05, size=15)


# # Test Models Performance

# In[ ]:


print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, lr.predict(X_test_std))

print ("Logistic Regression AUC = %2.2f" % lr_auc)
print(classification_report(y_test, lr.predict(X_test_std)))

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test_std))

print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test_std)))


# We can observe that random forest has higher accuracy compared to logistic regression model in both test and train data sets.

# # ROC Graph

# In[ ]:


# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test_std)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test_std)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)


# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# # Random Forest Feature Importances

# In[ ]:


columns = X.columns
train = pd.DataFrame(np.atleast_2d(X_train_std), columns=columns) # Converting numpy array list into dataframes


# In[ ]:


# Get Feature Importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances.head(10)


# In[ ]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the Feature Importance
sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances[0:10],
            label="Total", color="b")


# # Recommendations

# Above are the dominant features our model is predicting so our target population are the people:
# 
# - Who Spends less time on the internet
# - Who spends less time on the website
# - Who has lower income
# - Who are older than our average sample (mean around 40 years old)
