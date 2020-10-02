#!/usr/bin/env python
# coding: utf-8

# # MULTIPLE LINEAR REGRESSION MODEL FOR WEIGHT ESTIMATION FROM MEASUREMENTS OF THE FISH AND LOGISTICAL MODEL FOR SPECIE'S  PREDICTION

# ### The purpose of study of this data is to estimate the weight of individual fish with the help of a multiple linear regression model which considers features such as height ,width,length etc of the fish, as well as prediction of a any particular specie with the help of a logistical regression model 
# 
# 

# 
#     Positive outcomes: 
#                        - Improvment in studies of fishes in any water body .
#                        - Estimation of individual fish speices population in the selected water mass.
#                        - Obtaining more info on fishes without any physical contact that might harm
#                          them.
#                   
#          Environment:  
#                        -python 3 
#                   

# This data set contains the following features:
# 
# * 'Species': Name of the species 
# * 'Weight': Weight of individual fish in grams
# * 'Length 1': Vertical length in cm
# * 'Length 2': Diagonal length in cm
# * 'Length 3': Cross length in cm
# * 'Height': Height in cm
# * 'Width': Width in cm
# 
# Our dependent variable is 'Weight'. Independent variables are 'species', different lengths, 'height' and 'width'.
# 
# I will use independent variables (measurements of the fish) to estimate dependent variable (weight of the fish).
# 
# 

# ### Let's start with importing all our required libraries 

# In[ ]:


#Data manipulation libraries

import pandas as pd # reading and writing CSV etc 
import numpy as np # handling mathematical functions

#Data Viz libraries
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

#Scientific Learning libraries
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# ### Now let's read the data from our project directory 

# In[ ]:


df=pd.read_csv('../input/fish-market/Fish.csv')


# In[ ]:


print(str('Is there any NaN value in the dataset: '), df.isnull().values.any()) ## checking for null values


# ### Surveying data with the help of .info(),.head(),.describe() 

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Now let's move to our Exploratory Data Analysis
# ### We'll be using seaborn for plotting the graphs 

# In[ ]:


sns.countplot(x=df['Species']) 


# In[ ]:


sns.barplot(y=df['Height'],x=df['Species']) #Bream has the greatest height


# In[ ]:


sns.barplot(y=df['Width'],x=df['Species']) #whitefish has maximum width along side bream


# In[ ]:


sns.barplot(y=df['Weight'],x=df['Species']) #Pike has maximum weight


# In[ ]:


sns.heatmap(df.corr(),annot=True) ## To check all the correlations present 


# In[ ]:


sns.pairplot(df) 


# ### The dataset seems to be fairly consistent ,but still we'll quickly rush through some basic techniques to detect any sorts of outliers 

# In[ ]:


plt.figure(figsize=(5,8))
sns.boxplot(y=df['Weight']) 


# ### Well it seems like i was wrong after all about the outliers, we'll try and reduce them with the help of Z-score and IQR 

# ### Z-score analysis 

# In[ ]:


z = np.abs(stats.zscore(df.drop('Species',axis=1)))
threshold=3
print(np.where(z>3))
print('\n')
print(np.where(z<-3)) ## caught some outliers, here i have their indexes  


# ### IQR analysis

# In[ ]:


df_= df.drop('Species',axis=1)
Q1 =df_.quantile(0.25)
Q3 = df_.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# ### Since there are very less number of outliers ill be removing them

# In[ ]:


df_ = df_[(z < 3).all(axis=1)]


# In[ ]:


df_out = df_[~((df_ < (Q1 - 1.5 * IQR)) |(df_ > (Q3 + 1.5 * IQR))).any(axis=1)]
df_out.shape


# ### Now let's see if we have done enough to get rid of them

# In[ ]:


plt.figure(figsize=(5,8))
sns.boxplot(y=df_out['Weight']) ## fair enough 


# ### Now that i have my data clean for estimating the weight ,ill start with same thing again for the predicting of species

# In[ ]:


df_s=df
df['Species'].unique()


# #### I'll seperate out each species so that later on i can see the outliers if individually if any are there 

# In[ ]:


df_bream=df[df['Species']=='Bream']
df_roach=df[df['Species']=='Roach']
df_whitefish=df[df['Species']=='Whitefish']
df_parkki=df[df['Species']=='Parkki']
df_perch=df[df['Species']=='Perch']
df_pike=df[df['Species']=='Pike']
df_smelt=df[df['Species']=='Smelt']


# In[ ]:


sns.pairplot(df_s,hue='Species')


# ### Let's start with catching outliers 

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(y=df['Weight'],x=df['Species'])


# ### Seems like only Roach and Smelt have outliers, so we'll remove them on the basis Z-score and IQR score

# In[ ]:


z = np.abs(stats.zscore(df_roach.drop('Species',axis=1)))
df_q= df_roach.drop('Species',axis=1)
Q1 =df_q.quantile(0.25)
Q3 = df_q.quantile(0.75)
IQR = Q3 - Q1
df_roach = df_roach[(z < 3).all(axis=1)]
df_roach = df_roach[~((df_roach < (Q1 - 1.5 * IQR)) |(df_roach > (Q3 + 1.5 * IQR))).any(axis=1)]


z = np.abs(stats.zscore(df_smelt.drop('Species',axis=1)))
df_q= df_smelt.drop('Species',axis=1)
Q1 =df_q.quantile(0.25)
Q3 = df_q.quantile(0.75)
IQR = Q3 - Q1
df_smelt = df_smelt[(z < 3).all(axis=1)]
df_smelt = df_smelt[~((df_smelt < (Q1 - 1.5 * IQR)) |(df_smelt > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:


d=[df_bream,df_roach,df_whitefish,df_parkki, df_perch,df_pike,df_smelt]
result=pd.concat(d)


# ### Seems like we have taken them out and are ready with our final dataset for predicting the species. But first let's make sure that they are really gone.

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(y=result['Weight'],x=result['Species'])


# # Machine Learning Model Prep Start's here: 
# ## Here i've selected linear Regression model for estimation of weight(dependent variable) with the help of width , length, height etc(independent variable). 

# ### Let's seperate out dependent and independent values first 

# In[ ]:


X=df_out.drop(['Weight'],axis=1)
y=df_out['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


lr=LinearRegression() #creating a LinearRegression model instance


# In[ ]:


lr.fit(X_train,y_train) #now we'll directly fit the model and predict the outcome


# In[ ]:


pre=lr.predict(X_test)


# In[ ]:


print('R2_score:')
print(metrics.explained_variance_score(y_test,pre)) 


# ### Not a bad score for such a lazy model xD

# In[ ]:


sns.distplot((y_test-pre))


# In[ ]:


plt.scatter(y_test,pre)


# ## Now let's perform some preprocessing with 2nd degree polynomial featuring 

# In[ ]:


pf=PolynomialFeatures() #Creating a PolynomialFeatures instance


# In[ ]:


quad=pf.fit_transform(X) #Creating a transformed data 


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(quad, y, test_size=0.20, random_state=42)


# In[ ]:


lr.fit(X_train,y_train)  #Now go with the basic Linear Regression Steps fit and predict


# In[ ]:


pred=lr.predict(X_test)


# In[ ]:


print('R2_score:',metrics.explained_variance_score(y_test,pred))
print('MAE:',metrics.mean_absolute_error(y_test,pred))
print('MSE:',metrics.mean_squared_error(y_test,pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# ### This score seems to be pretty good, but if we change random_state our train and test samples will be different and our model's score will be different. In order to eliminate this change I will use cross validation:
# 

# In[ ]:


cross_val_score_train = cross_val_score(lr, X_train, y_train, cv=10, scoring='r2')
print(cross_val_score_train)


# ### The mean of all the above values would give me a almost perfect conclusion about my R2 score

# In[ ]:


print(np.mean(cross_val_score_train))


# ## Now we are talking results 
# ### So this was my best trained model for estimating the weight of individual fish based on given data set

# In[ ]:


# bit of graphical plotting just to make sure that i have not overfitted the data in any case
sns.distplot((y_test-pred))


# In[ ]:


plt.scatter(y_test,pred,marker='o')


# In[ ]:


# Model Seems to be pretty good


# ## Now let's start with predicting species(Dependent feature), based upon rest of the independent features.
# ### Here i've selected Logistical Regression algorithm because of the small size of data and catagorical target feature

# ### Let's seperate out the dependent and independent variable from the later created final dataset 'result'

# In[ ]:


X=result.drop(['Species'],axis=1)
y=result['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


logr=LogisticRegression()  #creating an LogisticRegression instance


# In[ ]:


logr.fit(X_train,y_train)  #now will fit and predict


# In[ ]:


pre=logr.predict(X_test)


# In[ ]:


logr.score(X_test,y_test)


# ## Well looking at the size of data this seems like a fine result 

# ### Let's checkout the results by comparing actual and predicted values

# In[ ]:


data={'Actual':y_test.array,'Predicted':pre}


# In[ ]:


res=pd.DataFrame(data)


# In[ ]:


res


# # So this was all from my side, feel free to post any query or issues related to it.
