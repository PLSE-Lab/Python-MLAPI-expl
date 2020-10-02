#!/usr/bin/env python
# coding: utf-8

# # Rain in Australia

# #### This kernel predicts whether it will rain tomorrow.

# #### I'm a newbie in ML,if you like it, please upvote :)

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

#Showing full path of datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

#Disable warnings
import warnings
warnings.filterwarnings('ignore')


# # Import Dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")


# ## Head

# In[ ]:


df.head()


# ## Shape

# In[ ]:


#Number of rows and columns in our dataset
df.shape


# ## Columns

# In[ ]:


#The 24 columns 
df.columns


# #### Drop Risk_MM column

# In[ ]:


#As mentioned in the dataset description , 
#we should exclude the variable Risk-MM when training a binary classification model.
#Not excluding it will leak the answers to your model and reduce its predictability.

df.drop(['RISK_MM'],axis=1,inplace=True)


# ## Info

# In[ ]:


#Basic Information of dataset

df.info()


# In[ ]:


#Before looking at the description of the data
#We can see that there are few columns with very less data
#Evaporation,Sunshine,Cloud9am,Cloud3pm
#It is better to remove these four columns as it will affect our prediction even if we
#fill the na values...

#Date and Location is also not required
#As we are predicting rain in australia and not when and where in australia


drop_cols = ['Evaporation','Sunshine','Cloud9am','Cloud3pm','Date','Location']

df.drop(columns=drop_cols,inplace=True,axis=1)


# #### Data after dropping columns

# In[ ]:


df.info()


# In[ ]:


#Basic description of our data
#Numerical features first
df.describe()


# * The count is different for all the features
# * We can see difference in mean and max is huge in many features.

# In[ ]:


#Including Categorical features with include object
df.describe(include='object')


# * Here too there are null values in our categorical data

# In[ ]:


#Now including all the features
df.describe(include='all')


# ## Null Values

# In[ ]:


#Our dataset consists of 142193 rows and the count for many features is less than 142193.
#This shows presence of Null values.
#Let's look at the null values..

df.isna().sum()


# * Except Date,Location and our target feature,
# * All the other features have null values
# * We'll deal with these later in the notebook..

# ## Skewness

# In[ ]:


df.skew()


# * The features with skewness values near zero may follow gaussian distribution.
# * Rainfall is strongly right skewed (9.88)
# * We'll have a look at the distributions for further clarity.

# In[ ]:


#Filling missing values

#We can see that there are outliers in our data
#So the best way to fill the na values in our numerical features is with median
#Because median deals the best with outliers

#Let's separate numerical and categorical
#data type of numerical features is equal to float64
#With the help of following list comprehension we separate the numerical features...

num = [col for col in df.columns if df[col].dtype=="float64"]

for col in num:
    df[col].fillna(df[col].median(),inplace=True)
    
cat = [col for col in df.columns if df[col].dtype=="O"]
for col in cat:
    df[col].fillna(df[col].mode()[0],inplace=True)


# In[ ]:


#Check missing values
df.isna().sum()


# * There are no missing values present now and we can start our analysis.

# # Correlation

# In[ ]:


df.corr().style.background_gradient(cmap="Reds")


# # Heatmap

# In[ ]:


#With the use of heatmap
corr = df.corr()

fig = plt.figure(figsize=(12,12))
sns.heatmap(corr,annot=True,fmt=".1f",linewidths="0.1")


# **Correlated features**
# 
# * MinTemp -- MaxTemp (0.7)
# * MinTemp -- Temp3pm (0.7)
# * MaxTemp -- Temp9am (0.9)
# * WindGustSpeed -- WindSpeed9am (0.6)
# * WindGustSpeed -- WindSpeed3pm (0.7)
# * Humidity9am -- Humidity3pm (0.7)
# * Humidity3pm -- Temp3pm (-0.6)
# * Temp9am -- MinTemp (0.9)
# * Temp9am -- Temp3pm

# **Features that are less correlated with other features**
# 
# * Rainfall
# * Pressure9am
# * Pressure3pm

# * Later we'll have a look at the scatterplots of these correlated features.....

# ## Study the numerical features

# In[ ]:


print("Numerical features :: {}\n".format(num))
print("No of Numerical features :: {}".format(len(num)))


# # Distributions of each numerical feature

# In[ ]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=0.5)

i=1
colors = ['Red','Blue','Green','Cyan',
         'Red','Blue','Green','Cyan',
         'Red','Blue','Green','Cyan']
j=0
for col in num:
    plt.subplot(3,4,i)
    a1 = sns.distplot(df[col],color=colors[j])
    i+=1
    j+=1


# * Rainfall and Evaporation as seen with skewness value (9.88) and (3.74) are right skewed as seen above.
# * Cloud9am and Cloud3pm behave as categorical features.

# # Boxplot

# In[ ]:


plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

i=1
for col in num:
    plt.subplot(6,2,i)
    a1 = sns.boxplot(data=df,x="RainTomorrow",y=col)
    i+=1


# **With the help of skewness values and above box plots,Rainfall,WindGustSpeed,WindSpeed9am and WindSpeed3pm**
# **may contain outliers**

# # Find the outliers

# In[ ]:


#Create a loop that finds the outliers in train and test  and removes it
features_to_examine = ['Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm']

for col in features_to_examine:
    IQR = df[col].quantile(0.75) - df[col].quantile(0.25) 
    Lower_Bound = df[col].quantile(0.25) - (IQR*3)
    Upper_Bound = df[col].quantile(0.75) + (IQR*3)
    
    print("The outliers in {} feature are values <<< {} and >>> {}".format(col,Lower_Bound,Upper_Bound))
    
    minimum = df[col].min()
    maximum = df[col].max()
    print("The minimum value in {} is {} and maximum value is {}".format(col,minimum,maximum))
    
    if maximum>Upper_Bound:
          print("The outliers for {} are value greater than {}\n".format(col,Upper_Bound))
    elif minimum<Lower_Bound:
          print("The outliers for {} are value smaller than {}\n".format(col,Lower_Bound))


# # Barplots

# In[ ]:


plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

i=1
for col in num:
    plt.subplot(6,2,i)
    a1 = sns.barplot(data=df,x="RainTomorrow",y=col)
    i+=1


# **Now let's have a look at the correlated variables from the above histogram**

# # Scatterplots

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplots_adjust(hspace=0.5)

i=1
features_list = ["MaxTemp","Temp9am","Temp3pm"]
for feature in features_list:
    plt.subplot(1,3,i)
    sns.scatterplot(data=df,x="MinTemp",y=feature,hue="RainTomorrow")
    i+=1


# In[ ]:


plt.figure(figsize=(15,8))
plt.subplots_adjust(hspace=0.5)

plt.subplot(3,2,1)
sns.scatterplot(data=df,x="WindSpeed9am",y="WindGustSpeed",hue="RainTomorrow")

plt.subplot(3,2,2)
sns.scatterplot(data=df,x="WindSpeed3pm",y="WindGustSpeed",hue="RainTomorrow")

plt.subplot(3,2,3)
sns.scatterplot(data=df,x="Humidity9am",y="Humidity3pm",hue="RainTomorrow")

plt.subplot(3,2,4)
sns.scatterplot(data=df,x="Temp9am",y="Temp3pm",hue="RainTomorrow")

plt.subplot(3,2,5)
sns.scatterplot(data=df,x="MaxTemp",y="Temp9am",hue="RainTomorrow")

plt.subplot(3,2,6)
sns.scatterplot(data=df,x="Humidity3pm",y="Temp3pm",hue="RainTomorrow")


# # Categorical Features

# In[ ]:


cat


# ### WindGustDir

# In[ ]:


df['WindGustDir'].value_counts()


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(data=df,x="WindGustDir",hue="RainTomorrow");


# ### WindDir9am

# In[ ]:


df['WindDir9am'].value_counts()


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(data=df,x="WindDir9am",hue="RainTomorrow");


# ### WindDir3pm

# In[ ]:


df['WindDir3pm'].value_counts()


# In[ ]:


fig = plt.figure(figsize=(15,5))
sns.countplot(data=df,x="WindDir3pm",hue="RainTomorrow");


# ## Target Feature

# In[ ]:


df['RainTomorrow'].value_counts()


# In[ ]:


sns.countplot(data=df,x="RainTomorrow")


# #### Splitting the data into **training** and **testing** sets

# In[ ]:


from sklearn.model_selection import train_test_split as tts
y=df[['RainTomorrow']]
X=df.drop(['RainTomorrow'],axis=1)

X_train,X_test,y_train,y_test = tts(X,y,test_size=0.3,random_state=0)


# In[ ]:


X_train


# In[ ]:


X_test


# # Feature Engineering

# #### First we will remove any outliers present in our data

# #### We have found the outliers in Rainfall,WindGustSpeed,WindSpeed9am and WindSpeed3pm.
# **We Will cap these outliers now**

# 
# **Let's have a look at their histograms.**

# In[ ]:


#We'll plot these four as subplots 

plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

features_to_examine = ['Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
i=1
for col in features_to_examine:
    plt.subplot(6,2,i)
    fig = df[col].hist(bins=10)
    fig.set_xlabel(col)
    fig.set_ylabel('RainTomorrow')
    i+=1


# **We can clearly see right skewed histograms in all the four**

# ***We'll try to cap these outliers that will help us in predictions later.***

# In[ ]:


def remove_outliers(df,col,Lower_Bound,Upper_Bound):    
    minimum = df[col].min()
    maximum = df[col].max()
    
    if maximum>Upper_Bound:
        return np.where(df[col]>Upper_Bound,Upper_Bound,df[col])
          
    elif minimum<Lower_Bound:
        return np.where(df[col]<Lower_Bound,Lower_Bound,df[col])


# In[ ]:


for df1 in [X_train,X_test]:
    df1['Rainfall'] = remove_outliers(df1,'Rainfall',-1.799,2.4)
    df1['WindGustSpeed'] = remove_outliers(df1,'WindGustSpeed',-14.0,91.0)
    df1['WindSpeed9am'] = remove_outliers(df1,'WindSpeed9am',-29.0,55.0)
    df1['WindSpeed3pm'] = remove_outliers(df1,'WindSpeed3pm',-20.0,57.0)


# In[ ]:


#If we look at their boxplots we can see that the outliers are now capped...
plt.figure(figsize=(15,30))
plt.subplots_adjust(hspace=0.5)

features_to_examine = ['Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
i=1
for col in features_to_examine:
    plt.subplot(6,2,i)
    fig = sns.boxplot(data=X_train,y=col)
    fig.set_xlabel(col)
    fig.set_ylabel('RainTomorrow')
    i+=1


# In[ ]:


#Describe helps us understand more about the mean and max values

X_train[features_to_examine].describe()


# In[ ]:


X_test[features_to_examine].describe()


# # Encode categorical variables

# In[ ]:


#Our next step is to encode all the categorical variables.
#first we will convert our target variable

for df2 in [y_train,y_test]:
    df2['RainTomorrow'] = df2['RainTomorrow'].replace({"Yes":1,
                                                    "No":0})


# Encode **RainToday** variable

# In[ ]:


import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[ ]:


#Now we will make our training dataset

X_train = pd.concat([X_train[num],X_train[['RainToday_0','RainToday_1']],
                    pd.get_dummies(X_train['WindGustDir']),
                    pd.get_dummies(X_train['WindDir9am']),
                    pd.get_dummies(X_train['WindDir3pm'])],axis=1)


# In[ ]:


X_train.head()


# In[ ]:


#Same for testing set

X_test = pd.concat([X_test[num],X_test[['RainToday_0','RainToday_1']],
                    pd.get_dummies(X_test['WindGustDir']),
                    pd.get_dummies(X_test['WindDir9am']),
                    pd.get_dummies(X_test['WindDir3pm'])],axis=1)


# In[ ]:


X_test.head()


# # Feature Scaling

# In[ ]:


#our training and testing set is ready for our model
#But ,before that we need to bring all the features to same scale with feature scaling
#For this we will use MinMaxScaler
#As there our negative values in our dataset and MinMaxScaler scales our data in range -1 to 1.

cols = X_train.columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


X_train = pd.DataFrame(X_train,columns=cols)
X_test = pd.DataFrame(X_test,columns=cols)


# **Finally ,after removing outliers,encoding the categorical variables and scaling**
# 
# **Our training and testing sets are ready**

# # Model Training

# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(X_train, y_train)


# In[ ]:


#Prediction on Xtest

y_pred_test = logreg.predict(X_test)

y_pred_test


# In[ ]:


#using predict_proba gives the probability value for the target feature

logreg.predict_proba(X_test)


# In[ ]:


#probability of getting no rain (0)

logreg.predict_proba(X_test)[:,0]


# In[ ]:


#probability of getting rain (1)

logreg.predict_proba(X_test)[:,1]


# In[ ]:


#Check accuracy with accuracy_score

from sklearn.metrics import accuracy_score

predict_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of model on test set :: {}".format(predict_test))


# In[ ]:


#Creating confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)


# ### The result is telling us that we have 31308+4554 correct predictions and 5068+1728 incorrect predictions.

# **35,862 correct predictions.**
# 
# **6796 Incorrect predictions.**

# In[ ]:


#Classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))


# In[ ]:


#Comparing train and test accuracy

y_pred_train = logreg.predict(X_train)
y_pred_train


# In[ ]:


#Check accuracy of our model with train set

predict_train = accuracy_score(y_train,y_pred_train)
print("Accuracy of our model on train set :: {}".format(predict_train))


# ### We can see somewhat same score for both out training and testing datasets using this model

# In[ ]:


#Overall Accuracy

print("Accuracy of our model :: {}".format(logreg.score(X_test,y_test)))


# ### 84% accuracy is good but we can still improve it

# In[ ]:


#Let's try to improve the accuracy of our model

#Let's try different C values

#Now what is C


# #### C is inverse of regularization strength.

# #### Higher values of C correspond to less regularization

# #### By default , C is equal to 1

# * Now let's reduce the regularization strength

# In[ ]:


#C=100

# instantiate the model
logreg100 = LogisticRegression(solver='liblinear',C=100, random_state=0)


# fit the model
logreg100.fit(X_train, y_train)

#Prediction on Xtest

y_pred_test = logreg100.predict(X_test)

y_pred_test


# In[ ]:


predict_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of model on test set :: {}".format(predict_test))


# In[ ]:


#Overall Accuracy

print("Accuracy of our model :: {}".format(logreg100.score(X_test,y_test)))


# In[ ]:


#Confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)


# **35,886 Correct predictions**
# 
# **6,792 Incorrect predictions**

# #### We can see increase in correct predictions and decrease in incorrect predictions

# In[ ]:


#Classification report
print(classification_report(y_test, y_pred_test))


# #### We can see a slight increase in our model with C=100

# In[ ]:


#Let's increase the regularization strength

#C=0.01

# instantiate the model
logreg001 = LogisticRegression(solver='liblinear',C=0.01, random_state=0)


# fit the model
logreg001.fit(X_train, y_train)

#Prediction on Xtest

y_pred_test = logreg001.predict(X_test)

y_pred_test


# In[ ]:


predict_test = accuracy_score(y_test,y_pred_test)

print("Accuracy of model on test set :: {}".format(predict_test))


# In[ ]:


#Overall Accuracy

print("Accuracy of our model :: {}".format(logreg001.score(X_test,y_test)))


# #### We can see a decrease in our model accuracy with C=0.01

# ## Our aim is to predict whether it will rain or not tomorrow in australia.

# #### Let's see the probability of raining with the help of histogram

# In[ ]:


# store the predicted probabilities for class 1 - Probability of rain

y_pred1 = logreg100.predict_proba(X_test)[:, 1]
y_pred0 = logreg100.predict_proba(X_test)[:, 0]


# In[ ]:


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)
plt.hist(y_pred0, bins = 10)

# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities')


# set the x-axis limit
plt.xlim(0,1)

#Set legend
plt.legend('upper left' , labels = ['Rain','No Rain'])

# set the title
plt.xlabel('Predicted probabilities')
plt.ylabel('Frequency')


# * The above histogram is highly right skewed for rain
# * Highly left skewed for no rain 
# * There is less chance that it will rain tomorrow as most of the predicted probabilities are near to zero.
# * Higher chance as probabilities are close to 1.
# 

# Conclusion :-
#     
#     * Our model predicts that there's a higher chance of not raining tomorrow in australia as seen in the above histogram.
#     
#     * Accuracy of our model is 84%.

# ### PLEASE GIVE A UPVOTE IF YOU LIKE THIS KERNEL :)
