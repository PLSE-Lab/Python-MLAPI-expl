#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("../input/wine8kfinal.csv")
df.head()


# In[3]:


df.info(verbose=True)


# In[4]:


df.describe()


# In[5]:


df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9])


# In[6]:


sns.pairplot(df)


# In[7]:


def create_label_encoder_dict(df):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    for column in df.columns:
        # Only create encoder for categorical data types
        if not np.issubdtype(df[column].dtype, np.number) and column != 'age':
            label_encoder_dict[column]= LabelEncoder().fit(df[column])
    return label_encoder_dict


# In[8]:


label_encoders = create_label_encoder_dict(df)
print("Encoded Values for each Label")
print("="*32)
for column in label_encoders:
    print("="*32)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)


# In[9]:


# Apply each encoder to the data set to obtain transformed values
df3 = df.copy() # create copy of initial data set
for column in df3.columns:
    if column in label_encoders:
        df3[column] = label_encoders[column].transform(df3[column])

print("Transformed data set")
print("="*32)
df3.head(10)


# In[10]:


sns.pairplot(df3)


# In[11]:


df3['price'].plot.hist(bins=25,figsize=(6,4))


# In[12]:


df3['price'].plot.density()


# In[13]:



#Correlation matrix and heatmap
df3.corr()


# In[14]:


plt.figure(figsize=(10,7))
sns.heatmap(df3.corr(),annot=True,linewidths=2)


# In[15]:


df4=df3.copy()
del df4['price']
df4.head(5)


# In[16]:


#Make a list of data frame column names
l_column = list(df3.columns) # Making a list out of column names
len_feature = len(l_column) # Length of column vector list
l_column


# In[17]:


#Put all the numerical  in X and Price in y, ignore Address which is string for linear regression
X = df3.loc[:, df3.columns != 'price']
y = df3[l_column[len_feature-8]]
print("Feature set size:",X.shape)
print("Variable set size:",y.shape)


# In[47]:


X.head(10)


# In[19]:


y.head()


# In[20]:


from sklearn.cross_validation import train_test_split


# In[21]:


#Create X and y train and test splits in one command using a split ratio and a random seed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)


# In[22]:


#Check the size and shape of train/test splits 

print("Training feature set size:",X_train.shape)
print("Test feature set size:",X_test.shape)
print("Training variable set size:",y_train.shape)
print("Test variable set size:",y_test.shape)


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[24]:


# Creating a Linear Regression object 'lm'
lm = LinearRegression() 

# Fit the linear model on to the 'lm' object itself
lm.fit(X_train,y_train) 


# In[25]:


print("The intercept term of the linear model:", lm.intercept_ )
print("The coefficients of the linear model:", lm.coef_)


# In[26]:


cdf = pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=["Coefficients"])
cdf


# In[75]:


#Scatter plot of predicted price and y_train set to see if the data fall on a 45 degree straight line
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)


plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted wine prices",fontsize=25)
plt.xlabel("Actual test set wine prices",fontsize=18)
plt.ylabel("Predicted wine prices", fontsize=18)
plt.scatter(x=y_train,y=train_pred)
#plt.plot(xfit, yfit,  c='red', linewidth=2)


# In[27]:


train_pred=lm.predict(X_train)
print("R-squared value of this fit:",round(metrics.r2_score(y_train,train_pred),3))


# In[28]:


#Prediction, error estimate, and regression evaluation matrices
#Prediction using the lm model
predictions = lm.predict(X_test)
print ("Type of the predicted object:", type(predictions))
print ("Size of the predicted object:", predictions.shape)


# In[49]:


X_new = pd.DataFrame({'price': [df3.price.min(), df3.price.max()]})
X_new.head()


# In[ ]:





# In[72]:


#Scatter plot of predicted price and y_test set to see if the data fall on a 45 degree straight line
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
x=y_test
y=predictions
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])


plt.figure(figsize=(10,7))
plt.title("Actual vs. predicted wine prices",fontsize=25)
plt.xlabel("Actual test set wine prices",fontsize=18)
plt.ylabel("Predicted wine prices", fontsize=18)
plt.scatter(x=y_test,y=predictions)
plt.plot(xfit, yfit,  c='red', linewidth=2)


# In[ ]:





# In[30]:


#Plotting histogram of the residuals i.e. predicted errors 

plt.figure(figsize=(10,7))
plt.title("Histogram of residuals to check for normality",fontsize=25)
plt.xlabel("Residuals",fontsize=18)
plt.ylabel("Kernel density", fontsize=18)
sns.distplot([y_test-predictions])


# In[31]:


#Scatter plot of residuals and predicted values (Homoscedasticity)

plt.figure(figsize=(10,7))
plt.title("Residuals vs. predicted values plot (Homoscedasticity)\n",fontsize=25)
plt.xlabel("Predicted wine prices",fontsize=18)
plt.ylabel("Residuals", fontsize=18)
plt.scatter(x=predictions,y=y_test-predictions)


# In[32]:


#Regression evaluation metrices

print("Mean absolute error (MAE):", metrics.mean_absolute_error(y_test,predictions))
print("Mean square error (MSE):", metrics.mean_squared_error(y_test,predictions))
print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[ ]:


#R-square value

print("R-squared value of predictions:",round(metrics.r2_score(y_test,predictions),3))


# In[ ]:




