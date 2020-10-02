#!/usr/bin/env python
# coding: utf-8

# # Black Friday Analysis and stepwise Prediction

# ## Import Libraries

# In[131]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import category_encoders as ce


# ### Import the Dataset and Evaluate it for Data Cleaning

# In[132]:


df = pd.read_csv('../input/BlackFriday.csv')


# * Visual inspection of the Dataset

# In[133]:


df.head()


# In[134]:


df.tail()


# * From the Above Data sections it is easy to observe that Dataset has mostly Categorical Variables as features and most of them are untidy and hence cannot be used directly in the model.

# In[135]:


df.info()


# * Here, we can observe that the Categories like *** Occupation , Product_Category_1 , Product_Category_2 and Product_Category_3 *** are of the type integer and float, We will need to consider them as Objects and handle them in the same manner as other Categorical data.

# # Data Visualization

# In[136]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot( x = 'Gender' , data = df)


# * The countPlot of the Dataset clearly demonstrates that **Men** are purchasing products much more than the **Women**.

# In[137]:


labels = ['City_A' , 'City_B', 'City_C']
sizes = [ df['City_Category'].value_counts()[2], df['City_Category'].value_counts()[0], df['City_Category'].value_counts()[1]]
explode = (0, 0.1, 0) 

fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(sizes , labels = labels , explode = explode, autopct = '%1.1f%%' , shadow = True)
ax1.axis('equal')
plt.show()


# * Number of people of **City B** are buying products more than **City A or City C**, Suggesting that although the margin is not much high but maintaining the stores and its supplies in **City B** can be more profitable.

# In[138]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot( x = 'Age' , data = df)


# * The CountPlot of *Age* group shows that people between the age of **26-35** are the potential customers of the store and as such products consumed by them must be kept in supply and stocked.

# In[139]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot( x = 'Occupation' , data = df)


# * The CountPlot of the *Occupation Category* demonstrates that people with **Job Type 0 ,4 , 7** are purchasing more products but since the plot is spread over many values , nothing can be deduced decisively about purchase according to Occupation of a person.

# In[140]:


labels = [ 1 ,2 ,3 , '4+' , 0]
stay_count = df['Stay_In_Current_City_Years'].value_counts()
sizes = [ stay_count[0] , stay_count[1] , stay_count[2] , stay_count[3] , stay_count[4] ]
explode = (0.1 , 0 , 0 , 0 , 0.1)

fig1 , ax1 = plt.subplots(figsize = (12 ,7))
ax1.pie( sizes , labels = labels , explode = explode , autopct = '%1.1f%%' , shadow = True)
ax1.axis('equal')
plt.show()


# * Above chart demonstrates that percentage of people living in the current city buying products , it can be observed that people living in the current city for 2 years are buying more products but not with much higher margin than the others.

# In[141]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot( x = 'Marital_Status' , data = df)


# * The CountPlot of *Marital Status* shows that unmarried people are buying products more but here also the margin between the categories is not much high in order to determine clear relation with purchase.

# In[142]:


fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(df['City_Category'],hue=df['Age'])


# * Above CountPlot verifies our hunch that people of age group **26-35** and living in **City B** are the potential Customers, But here we can also see that **City A and C** also has considerable number of buyers between **26-35 Age**.

# In[143]:


fig1 , ax1 = plt.subplots(figsize = (12,7))
sns.boxplot('Age' , 'Purchase' , data = df)
plt.show()


# * The BoxPlot between *Purchase amount* and *Age* demonstrates that there are some outliers in each category of Age which will affect the model calculations. 

# In[144]:


fig1 , ax1 = plt.subplots(figsize = (12,7))
plt.hist( 'Purchase' , data = df)
plt.show()


# * Above Histogram proves that high amount of *Purchase* is in low numbers that is outliers won't affect our model much. 

# # Data Cleaning

# In[145]:


pattern = re.compile('\d*\+')

def stay_in_city(row , pattern):
    stay = row['Stay_In_Current_City_Years']
    
    if bool(pattern.match(stay)):
        stay = stay.replace("+","")
        return stay
    else:
        return stay
    
df['Stay_In_Current_City_Years'] = df.apply( stay_in_city , axis = 1 , pattern = pattern )


# * Above code Removes the **"+"** from the feature *Stay_In_Current_City_Years* and makes the feature ready to be used.

# In[146]:


df[['Product_Category_1' , 'Product_Category_2' , 'Product_Category_3']] = df[['Product_Category_1' , 'Product_Category_2' , 'Product_Category_3']].fillna(0)


# * *Product_Category_1 , Product_Category_2 and Product_Category_3* all have missing data in them which we found out by observing and also from the info method,  hence the above code assign *0* to missing values.
# 

# ### **Our Dataset After Cleaning and Substituting Missing values in it is given Below : -**

# In[147]:


df.head()


# * We can observe that the *Age* also needs to be cleaned but since it is a category and can also be compared with eachother hence, we will manage it in our Preprocessing step.

# In[148]:


df.tail()


# In[149]:


df.info()


# * **Our Data is now ready to for the Preprocessing steps.**

# # Data PreProcessing Step

# * Creating two Copies of Our Dataset , Not a necessary step but one can use it to keep track of changes. 

# In[150]:


#Creating the dataset copies
dataset = df
#dataset = dataset.drop( columns = 'Unnamed: 0')
df = dataset.copy()


# In[151]:


df_dummy = df.iloc[:, 2:].values
pd.DataFrame(df_dummy).head()


# * Here we can observe all the categorical data that we need to take care of by using various **Encoding methods**.

# In[152]:


#Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
df_dummy[:, 0] = labelencoder_X_1.fit_transform(df_dummy[:, 0])
df_dummy[:, 1] = labelencoder_X_1.fit_transform(df_dummy[:, 1])
df_dummy[:, 3] = labelencoder_X_1.fit_transform(df_dummy[:, 3])
#Creating Dummy Variables for City Categories
onehotencoder = OneHotEncoder(categorical_features = [3])
df_dummy = onehotencoder.fit_transform(df_dummy).toarray()

#Removing Dummy variable trap
df_dummy = df_dummy[:, 1:]


# * Above code Encodes the categorical features ('Gender' , 'Age'  and 'City') and also creates dummy variables for city categories.
# * **Now ,Our Data Looks Like This :- **

# In[153]:


pd.DataFrame(df_dummy).head()


# In[154]:


#Binary Encoding
encoder = ce.BinaryEncoder(cols = [4 , 7 , 8 ,9]) 
df_dummy = encoder.fit_transform(df_dummy)


# * Above code Encodes Categorical features *( Ocuupation ,Product_Category_1 , Product_Category_2 and Product_Category_3)*  in binary format because of their large set of range values.

# * Our Dataset now looks like this and ready to use :-

# In[155]:


df_dummy.head()


# In[156]:


#Seprating Independent and Dependent variables
y = df_dummy.iloc[: , 29].values
X = df_dummy.iloc[:, 0:29].values


# In[157]:


#Seprating Dataset into Test set and Training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# # Predicting Purchase Using RandomForest Regression

# In[101]:


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)


# In[102]:


# Predicting a new result
y_pred = regressor.predict(X_test)


# * **Visual Comparison of the Predicted values and True values.**

# In[103]:


data = { 'Actual_Purchase' : y_test , 'Predicted_Purchase' : y_pred }
pd.DataFrame(data).head(10)


# In[104]:


data = { 'Actual_Purchase' : y_test , 'Predicted_Purchase' : y_pred }
pd.DataFrame(data).tail(10)


# ### K-fold Cross-Validation 

# In[118]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)


# In[119]:


print ( "Accuracies :"+str(accuracies)+"\nMean_accuracies :"+str(accuracies.mean())+"\nStandard_deviation :"+str(accuracies.std()))


# In[130]:


from sklearn.metrics import mean_squared_error , r2_score
print ("RMSE value :"+str(np.sqrt(mean_squared_error(y_test, y_pred))))
print ("R2 Score :"+str(r2_score(y_test , y_pred)))


# * *Model predicted values with 61.75% accuracy rate which not bad given that all the features were Categorical and no clear Relation can established between Purchase amount and them.*

# #### Do upvote if you liked reading it.
