#!/usr/bin/env python
# coding: utf-8

# ## The probability estimator for student getting desired university

# ### step 1-
# #### import libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:



data = pd.read_csv("../input/Admission_Predict.csv")


# In[ ]:


data.head(3)


# here comes Data Preparation
# 

# In[ ]:


## remove un necessary columns - as Serial No doenst have any impact of results
data.drop(['Serial No.'],inplace=True,axis=1)


# In[ ]:


data.head(2)


# In[ ]:


## check the correlation between data types


# In[ ]:


data.corr()


# In[ ]:


## here the Target is Chance of Admit 
## so now he have to understand the percentage of correlation between available features to target
## lets visualize the relation using heat map
sns.heatmap(data.corr(),annot=True)


# In[ ]:


## the CGPA stand first followed by GRE Score and TOEFL score
## lets understand each data in detail
## 1.CGPA
data['CGPA'].hist()


# In[ ]:


## the distribution is symetric here for CGPA
## 2.GRE Score
x=sns.barplot(y='Chance of Admit ',x='GRE Score',data=data)
plt.gcf().set_size_inches(15,10)


# In[ ]:


## from above we can observe in most of cases with the raise in GRE score chance of admit is increasing
## same way lets check for TOEFL


# In[ ]:


sns.kdeplot(data['TOEFL Score'],data['Chance of Admit '],shade=True)


# In[ ]:


data.columns


# In[ ]:


## you dont have any set of rules in choosing the plot type for data visualization
## the plot that best describes the relation and infromation best suits for visualization
## you can try with any no of plots you like
## lets also see the regression plot to validate the features
sns.regplot(x='GRE Score',y='Chance of Admit ',data=data)


# In[ ]:


## the above plot clearly says the relationship,as the data points are almost close to the regression line in most of the cases


# ### its time to look into the outliers 
# ### outliers plays very important role in Data Analysis 

# In[ ]:


## box plot is the best plot to find out the outliers
sns.boxplot(y='CGPA',data=data,showmeans=True,meanline=True)
plt.gcf().set_size_inches(5,5)


# In[ ]:


## the above plot says only one outlier avaliable and inner quartile range gives 50% of data -most of the cgpa in inner quartile range varies in b/w 8.2 to 9.1


# In[ ]:


## Now bivariate analysis 
## here we are going to comare two features


# In[ ]:


sns.boxplot(x='University Rating',y='GRE Score',data=data)


# In[ ]:


## it shows many box plots 
## now lets understand the above figure
## here we can see outliers in second,fourth and fifth box plots
## second box says - student with score in 330 - 340 range is expecting university of rating 2 - but why :) ?
## in 4 and 5,students with range of 290 - 310 are expecting top rated universities 


# In[ ]:


## can try the same for CGPA
sns.boxplot(y='CGPA',x='University Rating',data=data)


# In[ ]:


## now its time to compare multiple feature values
## note the word - lmplot, you many come across this many times


# In[ ]:


data.columns


# In[ ]:


sns.lmplot(x='Chance of Admit ',y='GRE Score',col='University Rating',data=data)


# In[ ]:


## its (facetgrid - means a combination of (axes - means a single plot))


# In[ ]:


## okay now ,we understood our data using the heat map ,correlation and other data visualization
## let me draw a heat map for reference
sns.heatmap(data.corr(),annot=True)


# ## Building ML Model 

# In[ ]:


## from above grap i can see all the features having good correlation- all features with more than 65%
## so we can try with different combination of features 
## fisrtly im going to select GRE,TOEFL,CGPA as features which are top 3 and Chance of Admit is our Target

features=data[['GRE Score','TOEFL Score','CGPA']]
target=data[['Chance of Admit ']]


# In[ ]:


features.head()


# In[ ]:


target.head()


# In[ ]:


## lets import ML libraries
## sklearn library holds all the algorithms that is used to predict the values
## though we are just reusing the in built libraries,it is very important to learn algorithms and the logics behind it
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_val,y_train,y_val=train_test_split(features,target,random_state=1,test_size=0.2)


# In[ ]:


## what did i do-
## I've imported a method called train_test_split,as name says its going to split the data into train and test part
## lets see the data one by one


# In[ ]:


X_train.head()


# In[ ]:


X_val.head()


# In[ ]:


y_train.head()


# In[ ]:


y_val.head()


# In[ ]:


## now with train data we are going to make our model learn our data 
## like i mentioned before ,many algorithms are available in sklearn library to build our model
## lets use linear regression mmodel
from sklearn.linear_model import LinearRegression


# In[ ]:


## here model is studing our train data
student_model=LinearRegression()
student_model.fit(X_train,y_train)


# In[ ]:


## to check the score of our current train data
student_model.score(X_train,y_train)


# In[ ]:


## its 80% which is pretty good to build the model


# In[ ]:


## here we have predited the score by giving the X_val(sample output - that is chances of getting admission as input)
y_pred=student_model.predict(X_val)


# In[ ]:


## now we are going to compare how close the above predicted value is with the y_val
## for this we have set of inbuilt functions that can be imported
from sklearn.metrics import r2_score


# In[ ]:


accuracy_result_LR=round(r2_score(y_pred,y_val)*100,2)


# In[ ]:


## its 69 which is a good model,but we can try for better accuracy if possible
accuracy_result_LR


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


DTR_model=DecisionTreeRegressor()


# In[ ]:


DTR_model.fit(X_train,y_train)


# In[ ]:


y_pred=DTR_model.predict(X_val)
accuracy_result_DTR=round(r2_score(y_pred,y_val)*100,2)
accuracy_result_DTR


# In[ ]:


## the decision tree model is giving very low prediction results, so lets try with new feature combination 


# In[ ]:


## here ive included all features
## for this data we havent encountered any feature with less correlation, but you ll going to deal with lot of data that gives very bad correlation
features2=data.drop(['Chance of Admit '],axis=1)
target2=data['Chance of Admit ']


# In[ ]:


X_train2,X_val2,y_train2,y_val2=train_test_split(features2,target2,random_state=1,test_size=0.2)


# In[ ]:


lin_model=LinearRegression()


# In[ ]:


lin_model.fit(X_train2,y_train2)


# In[ ]:


y_pred=lin_model.predict(X_val2)
accuracy_per=round(r2_score(y_pred,y_val2)*100,2)
accuracy_per


# In[ ]:


lin_model.score(X_train2,y_train2)


# In[ ]:


## this is a good considerable accuracy score
print(f'the accuracy percentage for our model is {accuracy_per}')


# ### pickle the model

# In[ ]:


pd.to_pickle(lin_model,'chance_prediction.pickle')


# In[ ]:


## why do we pickle?
## when we deal with large data frames ,its going to take long time to build and train a model
## so on pickling a model ,we dont need to build and train a model 


# ### consume the created model to check user inputs

# In[ ]:


## this is the way to extrat the pickle file and reuse it
## my_model=pd.read_pickle('chance_prediction.pickle')


# In[ ]:


##we have built our model sucessfully - when any user inputs are to be given , uncomment the below code and pass the input
## GRE=int(input('enter GRE score ,360 is max score - '))
## TOEFL=int(input('enter TOEFL score ,120 is max score -'))
## required_university_rating= int(input('enter university rating in 1-5 range -'))
## SOP=float(input('enter sop score in 1-5 range-'))
## LOR=float(input('enter LOR score in 1-5 range-'))
## CGPA=float(input('enter CGPA in 1-10 range-'))
## Research=int(input('enter Research score 0-if no and 1-if yes-'))


# In[ ]:


## inputs=[GRE,TOEFL,required_university_rating,SOP,LOR,CGPA,Research]


# In[ ]:


## this will be the result predition
## result=my_model.predict([inputs])
## print(f'so,the probability of you getting desired University is {round(result[0]*100,2)}')

