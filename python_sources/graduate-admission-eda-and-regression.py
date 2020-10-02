#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
   Author : Kenil Shah
   Github: Data-Science-Analytics/Datasets/UCLA Graduate Admission Prediction/Graduate_Admission_Prediction.ipynb
   
"""   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')


# # **UCLA Admission Predictor**

# ## **Exploratory Data Analysis**

# In[ ]:


dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
dataset.head() 


# In[ ]:


dataset.info()


# This tells us that there are no missing values in the dataset which is a plus point for us as we do not need to worry about missing values and what to do with it.

# The Serial Number just looks like an index to me and we can remove that column as it wont have any affect on our predictive model.

# In[ ]:


dataset.drop('Serial No.',axis = 1,inplace = True)
dataset.columns = ['GRE', 'TOEFL', 'University Rating','SOP','LOR','CGPA','Research','Chance of Admit']


# Now lets look at what each column indicates in the Dataset.
# GRE Score : Tells us about the GRE Score of each person.
# TOEFL Score : Tells us about the TOEFL Score of each person.
# University Rating : The Rating which the University has given to the Student
# SOP : The rating which is given to the Statement of Purpose for the Student
# LOR : The rating which is given to the Letter of Recommendation of the Student
# CGPA : Tells us about the CGPA of the student in the previous college
# Research : Tells us whether the studetn has done research or not
# Chance of Admit : Is the probability the student will get into the University considering all the parameters above

# Now lets look at the distribution plot of each of this variables and the average score.

# ### ***GRE Scores***

# In[ ]:


plt.rcParams['figure.figsize'] = 15,10
dataset['GRE'].plot(kind = 'kde')
dataset['GRE'].plot(kind = 'hist',density = True,color = 'g',alpha = 0.25)
plt.xlabel('GRE Score',fontsize = 20)
plt.ylabel('Frequency',fontsize = 20)
plt.title('Histogram and Distribution Plot of GRE Score',fontsize = 20)
plt.text(308, 0.013, 'Average Score %d' %(int(np.mean(dataset['GRE']))), fontsize=15)
plt.show()


# Now lets look at the distribution of GRE Score for all the students who had changes of admit  > 0.75. This will give us an idea about the GRE Score you need for higher chances

# In[ ]:


dataset[dataset['Chance of Admit'] > 0.75]['GRE'].plot(kind = 'hist',x = 'GRE',color = 'g',alpha = 0.25)
plt.text(320, 20, 'Average Score %d' %(int(np.mean(dataset[dataset['Chance of Admit'] > 0.75]['GRE']))), fontsize=15)
plt.xlabel('GRE Score',fontsize = 20)
plt.ylabel('Frequency',fontsize = 20)
plt.title('Histogram and Distribution Plot of GRE Score with Chances higher than 75%',fontsize = 20)
plt.show()


# Now we can see that to have a chance of admit just based on GRE Scores we need a score aorund 325+ to have a chance greter than 75%

# Now let us see if doing research has some effect on the chances and the average GRE Score than.

# In[ ]:


# add SNS plot here with hue = 'Research'
plt.rcParams['figure.figsize'] = 10,20
sns.lmplot(x = 'GRE' , y = 'Chance of Admit',hue = 'Research',data = dataset,fit_reg = False)
plt.axvline(x = 300,ymin = 0,ymax = 1)
plt.axvline(x = 320,ymin = 0,ymax = 1,color = 'orange')
plt.axhline(y = 0.6,color = 'green')
plt.title('Impact of Research')
plt.show()


# This tells us that if you have score around 300~320 and you do research you can have a chance higher than 60% to get into the university. You can also see for the higher scores in that range and having research they have more chances of an admit. Also if you do research and you have a more than decent score you have given yourself a good chance.

# ### ***TOEFL Scores***

# In[ ]:


plt.rcParams['figure.figsize'] = 15,10
dataset['TOEFL'].plot(kind = 'kde')
dataset['TOEFL'].plot(kind = 'hist',density = True,color = 'r',alpha = 0.25)
plt.text(100, 0.02, 'Average Score %d' %(int(np.mean(dataset['TOEFL']))), fontsize=15)
plt.xlabel('TOEFL Score',fontsize = 20)
plt.ylabel('Frequency',fontsize = 20)
plt.title('Histogram and Distribution Plot of TOEFL Score',fontsize = 20)
plt.show()


# So on an average student has a TOEFL score of 107 who applies to this university.

# Now lets look at the distribution of TOEFL Score for all the students who had changes of admit  > 0.75. This will give us an idea about the TOEFL Score you need for higher chances

# In[ ]:


dataset[dataset['Chance of Admit'] > 0.75]['TOEFL'].plot(kind = 'hist',x = 'TOEFL',color = 'r',alpha = 0.25)
plt.text(110, 15, 'Average Score %d' %(int(np.mean(dataset[dataset['Chance of Admit'] > 0.75]['TOEFL']))), fontsize=15)
plt.xlabel('TOEFL Score',fontsize = 20)
plt.ylabel('Frequency',fontsize = 20)
plt.title('Histogram and Distribution Plot of TOEFL Score with chances higher than 75%',fontsize = 20)
plt.show()


# And as you can see we need a score of 112+ to have a admission chance of around 75%

# Now let us see if doing research has some effect on the chances and the average TOEFL Score than.

# In[ ]:


# add SNS plot here with hue = 'Research'
plt.rcParams['figure.figsize'] = 10,20
sns.lmplot(x = 'TOEFL' , y = 'Chance of Admit',hue = 'Research',data = dataset,fit_reg = False)
plt.axvline(x = 100,ymin = 0,ymax = 1)
plt.axvline(x = 110,ymin = 0,ymax = 1,color = 'orange')
plt.axhline(y = 0.6,color = 'green')
plt.title('Impact of Research')
plt.show()


# This shows similiar results to GRE score and student will lower TOEFL scores can also increase their chances if they have some research goin on or have done some research.

# > ### ***University Rating***

# In[ ]:


dataset['University Rating'].unique() # We have 5 different Ratings by the University


# We will now look at the distribution of Chance of Admit for each value of University Rating

# In[ ]:


plt.rcParams['figure.figsize'] = 15,5
sns.swarmplot(x = 'University Rating', y = 'Chance of Admit', hue = 'Research',data = dataset)
plt.title('Impact of Research')


# This is kind of self explanatory that someone who has done research will have higher chance of admit and a better university rating. It also shows how important it is to do research to get a university rating of 4 or 5. And even in university rating 3 we see that if you have done research you have a higher chance. 

#  The presence of rare cases like research is there but still not a good rating is becuase there is some other factor on their profile which is not that good and hence their chances are reduced.

# In[ ]:


dataset[(dataset['University Rating'] >= 4) & (dataset['Research'] == 1)].sort_values(by = ['Chance of Admit']).head(5)


# As expected we can see that they have a low score on any other column of their profile.

# > ### ***CGPA*** 

# Well CGPA is said to be the most important factor in the decision making of a university and lets see how much impact does it have here.

# In[ ]:


plt.rcParams['figure.figsize'] = 10,5
dataset['CGPA'].plot(kind = 'kde')
dataset['CGPA'].plot(kind = 'hist',density = True,color = 'y',alpha = 0.25)
plt.text(7.88, 0.2, 'Average CGPA %d' %(int(np.mean(dataset['CGPA']))), fontsize=15)
plt.xlabel('CGPA',fontsize = 20)
plt.ylabel('Frequency',fontsize = 20)
plt.title('Histogram and Distribution Plot of CGPA',fontsize = 20)
plt.show()


# WE see that the students who applied at the university had on an average 8 CGPA.

# Now lets look at the distribution of CGPA for all the students who had changes of admit  > 0.75. This will give us an idea about the CGPA you need for higher chances

# In[ ]:


dataset[dataset['Chance of Admit'] > 0.75]['CGPA'].plot(kind = 'hist',x = 'CGPA',color = 'y',alpha = 0.25)
plt.text(8.75, 20, 'Average Score %d' %(int(np.mean(dataset[dataset['Chance of Admit'] > 0.75]['CGPA']))), fontsize=15)
plt.xlabel('CGPA',fontsize = 20)
plt.ylabel('Frequency',fontsize = 20)
plt.xlim(8,10)
plt.title('Histogram and Distribution Plot of CGPA with chances higher than 75%',fontsize = 20)
plt.show()


# So to have a chance of 75% or more we need to have a CGPA of 9+. We can say that the University is very CGPA oriented and this could be a big factor in determining the chance of admission in an university.

# Now let us see if doing research has some effect on the chances and the average CGPA Score than.

# In[ ]:


# add SNS plot here with hue = 'Research'
plt.rcParams['figure.figsize'] = 10,20
sns.lmplot(x = 'CGPA' , y = 'Chance of Admit',hue = 'Research',data = dataset,fit_reg = False)
plt.axvline(x = 7,ymin = 0,ymax = 1)
plt.axvline(x = 9,ymin = 0,ymax = 1,color = 'orange')
plt.axhline(y = 0.6,color = 'green')
plt.title('Impact of Research')
plt.show()


# Again showing the same thing, if you have done research and have a lower CGPA you still have a higher chance. Though one thing as mentioned above, the university being CGPA centric, higer CGPA continues to have more affect than Research. Another strange thing is that in the higher regions of GRE,TOEFL and CGPA there were many student who has done research and many few who has not. This shows people would better grades tend to do research as well. But Causation does not mean Correlation.

# ### ***SOP and LOR***

# In[ ]:


print('Average SOP :', int(np.mean(dataset['SOP'])))
print('Average LOR :', int(np.mean(dataset['LOR'])))
plt.rcParams['figure.figsize'] = 10,5

dataset['SOP'].plot(kind = 'kde')
dataset['LOR'].plot(kind = 'kde')
plt.legend(['SOP','LOR'])
#dataset['CGPA'].plot(kind = 'hist',density = True,color = 'y',alpha = 0.25)


# SO we see that on an average they have and LOR and SOP value of 3. There is not much to look into this. But we will look at how research value impact the chances for SOP and LOR.

# In[ ]:


# Swarmplot for SOP and LOR values with hue Reasearch and y Chance of Admit
plt.rcParams['figure.figsize'] = 15,5
sns.swarmplot(x = 'SOP', y = 'Chance of Admit', hue = 'Research',data = dataset)
plt.title('Impact of Research')


# A possible explanation for this is that someone who has done research will have lots to write in the SOP and hence will have a better rating for that SOP as they would be able to explain alot of things about their research interests work done by them etc.

# In[ ]:


plt.rcParams['figure.figsize'] = 15,5
sns.swarmplot(x = 'LOR', y = 'Chance of Admit', hue = 'Research',data = dataset)
plt.title('Impact of Research')


# A similar explantation to the one given above for SOP, someone who has done research will know many professors who would write a good LOR for him/her and so will have a higher rating on LOR.Infact it is not strange to see that for Ratings of SOP and LOR above 4 we have more research students and only a few non-research students. Those exceptional ones would have something really interesting in thei SOPs to get such high scores.

# ### **Correlation of Variables**

# In[ ]:


sns.pairplot(dataset,vars = ['GRE','TOEFL','University Rating','SOP','LOR','CGPA','Chance of Admit'],
             kind = 'reg',diag_kind = 'kde',palette="husl")


# This shows that almost all the variables has good correlation with each other and a positive relationship to be precise. The all seem to have a positive relation with the Chance of Admit as well. Well enough of exploration now. Lets move to the Prediction part of the dataset

# ## **Regression** 

# Lets first divide the dataset into train and test models.

# In[ ]:


#Dividing it into Independent and Dependent Variables

X = dataset.iloc[:,:-1].values # Independent Variables
Y = dataset.iloc[:,7].values # Dependent Variables


# ### **Multiple Linear Regression**

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Training the Model*

# In[ ]:


model_simple = LinearRegression()
model_simple.fit(train_X,train_Y)


# #### *Predicting the outcome*

# In[ ]:


pred = model_simple.predict(test_X)


# #### *Checking for Accuracy*

# In[ ]:


print('Mean Square Error is: ', mean_squared_error(test_Y,pred))
print('Model Accuracy Score : ',r2_score(test_Y,pred))


# #### *Residual vs Fitted Curve*

# In[ ]:


plt.scatter(model_simple.predict(train_X),model_simple.predict(train_X) - train_Y, c = 'b')
plt.hlines(y = 0,xmin = min(model_simple.predict(train_X)),xmax = max(model_simple.predict(train_X)))


# We can see that the points are randomly distributed around the line y = 0 and hence we cna say that the assumption of a linear fit on the data is actually true and this model can be used.

# ### **Polynomial Regression**

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Fitting Polynomial to the Dataset*

# In[ ]:


polynomial = PolynomialFeatures(degree = 2)   # Degree 2
polynomial_x = polynomial.fit_transform(train_X)
polynomial.fit(polynomial_x, train_Y)

polynomial_3 = PolynomialFeatures(degree = 3)   # Degree 3
polynomial_x_3 = polynomial_3.fit_transform(train_X)
polynomial_3.fit(polynomial_x_3, train_Y)


# #### *Training the Model*

# In[ ]:


model_poly = LinearRegression()  # Degree 2
model_poly.fit(polynomial_x,train_Y)

model_poly_3 = LinearRegression() # Degree 3
model_poly_3.fit(polynomial_x_3,train_Y)


# #### *Predicting the outcome*

# In[ ]:


pred_2 = model_poly.predict(polynomial.fit_transform(test_X)) # Degree 2
pred_3 = model_poly_3.predict(polynomial_3.fit_transform(test_X)) # Degree 3


# #### *Checking for Accuracy*

# In[ ]:


print('Mean Square Error for Polynomial degree 2 is: ', mean_squared_error(test_Y,pred_2))
print('Model Accuracy Score for Polynomial degree 2 is : ',r2_score(test_Y,pred_2))
print('Mean Square Error for Polynomial degree 3 is: ', mean_squared_error(test_Y,pred_3))
print('Model Accuracy Score for Polynomial degree 3 is : ',r2_score(test_Y,pred_3))


# #### *Actual vs Predicted*

# In[ ]:


plt.figure(figsize=(15,5))
x = np.arange(1,50)
plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')
plt.plot(x, pred[:49], ':o', label='Predicted',color = 'red',linewidth = 1)
plt.plot(x, pred_3[:49], ':x', label='Predicted',color = 'blue',linewidth = 1)
plt.legend();


# Now we can see that as we increase the degree of the polynomial our mean square error increases and R-squared value decreases. This is because we may be overfitting the data and hence we might be predicting correct values for the training set but not the test set. Since we saw in the Residual vs Fitted value graph for Multiple Linear Regression that Linear assumption for the model is indeed a correct assumption , we would stick to that and assume the model to be linear and hence would perform better with degree 1.

# ### **Support Vector Regression**

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Training the model*

# In[ ]:


model_svr_rbf = SVR(kernel = 'rbf',C = 10,gamma = 0.01) # Gaussian Kernel
model_svr_rbf.fit(train_X,train_Y)

model_svr_linear = SVR(kernel = 'linear', C = 1) #Linear kernel
model_svr_linear.fit(train_X,train_Y)


# #### *Predicting Outcome*

# In[ ]:


pred_svr_rbf = model_svr_rbf.predict(test_X)
pred_svr_linear = model_svr_linear.predict(test_X)


# #### *Checking for Accuracy*

# In[ ]:


print('Mean Square Error for Gaussian(Radial) kernel is: ', mean_squared_error(test_Y,pred_svr_rbf))
print('Model Accuracy Score for Gaussian(Radial) kernel is : ',r2_score(test_Y,pred_svr_rbf))
print('Mean Square Error for Linear kernel is: ', mean_squared_error(test_Y,pred_svr_linear))
print('Model Accuracy Score for Linear kernel is : ',r2_score(test_Y,pred_svr_linear))


# #### *Actual and Predicted*

# In[ ]:


plt.figure(figsize=(15,5))
x = np.arange(1,50)
plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')
plt.plot(x, pred_svr_rbf[:49], ':o', label='Predicted',color = 'red',linewidth = 1)
plt.plot(x, pred_svr_linear[:49], ':x', label='Predicted',color = 'blue',linewidth = 1)
plt.legend();


# As we know this is a linear dataset we expect the linear kernel to give a better result then the gaussia kernel. But by tweaking the parameters of the guassian kernel we are able to get a model that gives a similar result like the linear kernel. That being said we still have the Multiple Linear Regression as the best model till now.

# ### **Decision Tree Regression**

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Training the Dataset*

# In[ ]:


model_dtree = DecisionTreeRegressor(random_state = 0,max_depth = 5,max_features = 5,min_samples_split = 10)
model_dtree.fit(train_X,train_Y)


# #### *Predicting Outcome*

# In[ ]:


pred_dtree = model_dtree.predict(test_X)


# #### *Checking for Accuracy*

# In[ ]:


print('Mean Square Error : ', mean_squared_error(test_Y,pred_dtree))
print('Model Accuracy Score : ',r2_score(test_Y,pred_dtree))


# #### *Actual and Predicted*

# In[ ]:


plt.figure(figsize=(15,5))
x = np.arange(1,50)
plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')
plt.plot(x, pred_dtree[:49], ':o', label='Predicted',color = 'red',linewidth = 2)
plt.legend();


# ### **Random Forest Regression**

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Training the Dataset*

# In[ ]:


model_rforest = RandomForestRegressor(n_estimators = 500,random_state = 0,max_depth = 7
                                      ,max_features = 5,min_samples_split = 10)
model_rforest.fit(train_X,train_Y)


# #### *Predicting the Outcome*

# In[ ]:


pred_rforest = model_rforest.predict(test_X)


# #### *Checking for Accuracy*

# In[ ]:


print('Mean Square Error : ', mean_squared_error(test_Y,pred_rforest))
print('Model Accuracy Score : ',r2_score(test_Y,pred_rforest))


# #### *Actual and Predicted*

# In[ ]:


plt.figure(figsize=(15,5))
x = np.arange(1,50)
plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')
plt.plot(x, pred_rforest[:49], ':o', label='Predicted',color = 'red',linewidth = 2)
plt.legend();


# So we finally got a model that predicts better than the Linear Regression used above. And it RandomForest has been the best algorithm till now.

# ### **Principal Component Regression**

# We will try to fit the model with less number of independent variables using Principal Components Analysis by setting up a threshold of the variance we want and then using that number ot train our model.

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Finding the Principal Components*

# In[ ]:


pca = PCA(n_components = None)
train_X_pca = pca.fit_transform(train_X)
test_X_pca = pca.fit(test_X)
explained_variance = pca.explained_variance_ratio_


# In[ ]:


for x in explained_variance:
    print(round(x,2))


# As we can the variance explained by the first two components is enough and the other components can be ignored. Now using this components to create the new linear regression model

# #### *Training the Model*

# In[ ]:


pca = PCA(n_components = 4)
train_X_pca = pca.fit_transform(train_X)
test_X_pca = pca.transform(test_X)
model_simple_pca = LinearRegression()
model_simple_pca.fit(train_X_pca,train_Y)


# #### *Predicting the Outcome*

# In[ ]:


pred_pca = model_simple_pca.predict(test_X_pca)


# ####* Checking for Accuracy*

# In[ ]:


print('Mean Square Error : ', mean_squared_error(test_Y,pred_pca))
print('Model Accuracy Score : ',r2_score(test_Y,pred_pca))


# So this does not perform as good as the Linear Regression model.

# #### *Predicted and Actual*

# In[ ]:


plt.figure(figsize=(15,5))
x = np.arange(1,50)
plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')
plt.plot(x, pred_pca[:49], ':o', label='Predicted',color = 'red',linewidth = 2)
plt.legend();


# ### **K Nearest Neighbor Regressor**

# #### *Splitting the Dataset*

# In[ ]:


#Splitting it into train and test dataset
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 0) 
# Splitting it into 400 train and 100 test data set


# #### *Training the Model*

# In[ ]:


model_knn = KNeighborsRegressor(n_neighbors =10, metric = 'minkowski' , p = 2)
model_knn.fit(train_X,train_Y)


# #### *Predicting the Outcome*

# In[ ]:


pred_knn = model_knn.predict(test_X)


# #### *Checking for Accuracy*

# In[ ]:


print('Mean Square Error : ', mean_squared_error(test_Y,pred_knn))
print('Model Accuracy Score : ',r2_score(test_Y,pred_knn))


# #### *Actual vs Predicted*

# In[ ]:


plt.figure(figsize=(15,5))
x = np.arange(1,50)
plt.plot(x,test_Y[:49], '-o', label='Actual',color ='green')
plt.plot(x, pred_knn[:49], ':o', label='Predicted',color = 'red',linewidth = 2)
plt.legend();


# So that is the end of the Regression model and let us now look at how each model has performed.

# In[ ]:





# Since we now got the best model as Random Forest lets try Cross validation and see if we can improve the model further

# ### **Cross Validation on Random Forest**

# In[ ]:


cv = cross_val_score(estimator = model_rforest,X = X,y = Y,cv = 10)  # 10 parts


# #### *Checking for different Accuracy*

# In[ ]:


for accuracy in cv:
    print(accuracy)    


# In[ ]:


print('Accuracy mean:',cv.mean())
print('Accuracy Standard Deviation:',cv.std())


# We can see that we are further able to increase the accuracy of the model by using the cross validation approach and taking the value of k as 10 meaning dividing the dataset into 10 parts using 9 to train the model and 1 to test the model.

# The next thing we can do to further improve the model is Normalization of all its columns . We will take the top 2 algorithms i.e Random Forest and Linear Regression for this.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
train_X = minmax.fit_transform(train_X)
test_X = minmax.transform(test_X)


# In[ ]:


model_simple_normalized = LinearRegression()
model_simple_normalized.fit(train_X,train_Y)
pred_normalized = model_simple_normalized.predict(test_X)
print('Mean Square Error is: ', mean_squared_error(test_Y,pred_normalized))
print('Model Accuracy Score : ',r2_score(test_Y,pred_normalized))


# In[ ]:


model_rforest_normalized = RandomForestRegressor(n_estimators = 500,random_state = 0,max_depth = 7
                                      ,max_features = 5,min_samples_split = 10)
model_rforest_normalized.fit(train_X,train_Y)
pred_rforest_normalized = model_rforest_normalized.predict(test_X)
print('Mean Square Error is: ', mean_squared_error(test_Y,pred_rforest_normalized))
print('Model Accuracy Score : ',r2_score(test_Y,pred_rforest_normalized))


# In[ ]:


index = ['Linear','Polynomial_2','Polynomial_3','SupportVectorGuassin','SupportVectorLinear',
         'DecisionTree','RandomForest','PCR','KNearest','RandomForestNormalized','LinearNormalized']
mse = [[mean_squared_error(test_Y,pred),r2_score(test_Y,pred)],
       [mean_squared_error(test_Y,pred_2),r2_score(test_Y,pred_2)],
       [mean_squared_error(test_Y,pred_3),r2_score(test_Y,pred_3)],
       [mean_squared_error(test_Y,pred_svr_rbf),r2_score(test_Y,pred_svr_rbf)],
       [mean_squared_error(test_Y,pred_svr_linear),r2_score(test_Y,pred_svr_linear)],
       [mean_squared_error(test_Y,pred_dtree),r2_score(test_Y,pred_dtree)],
       [mean_squared_error(test_Y,pred_rforest),r2_score(test_Y,pred_rforest)],
       [mean_squared_error(test_Y,pred_pca),r2_score(test_Y,pred_pca)],
       [mean_squared_error(test_Y,pred_knn),r2_score(test_Y,pred_knn)],
       [mean_squared_error(test_Y,pred_rforest_normalized),r2_score(test_Y,pred_rforest_normalized)],
       [mean_squared_error(test_Y,pred_normalized),r2_score(test_Y,pred_normalized)]]
data = pd.DataFrame(data = mse,index = index)
data.columns = ['MSE','Accuracy']
data.sort_values(by = 'Accuracy',ascending = False)


# In[ ]:




