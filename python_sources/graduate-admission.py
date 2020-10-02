#!/usr/bin/env python
# coding: utf-8

# # Graduate Admission
# 
# ### Hi I have been working in Machine Learning from past one year. This is my first project working with Text Data. I have been working with Image Data more
# 
# ### Idea behind this project is to create a model and deploy on Heroku, which will help you get idea how machine learning product can be used End-To-End.
# 
# ### Dataset Link and details: https://www.kaggle.com/mohansacharya/graduate-admissions
# 
# ### In this notebook, you will get basic idea of what are the steps for developing Machine Learning Model
# 
# ### Feel free to reach out to me: 
# - Email- satish.fulwani63@gmail.com
# - Github - https://github.com/satishf889
# 
# ### Below are the useful links that will help you understand Heatmap used in this notebook
#   - https://medium.com/@rokaandy/python-data-visualization-heatmaps-79fa7506c410

# ### Import All Dependencies

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# ### Load the dataset

# In[ ]:


graduation_data=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
graduation_data.head()


# ### Understanding and visualising data

# In[ ]:


#Getting all the information of our dataset
graduation_data.info()


# In[ ]:


#We will find out mean,max,count of all the columns in our dataframe
graduation_data.describe()


# #### As we can see Serial number is independent column so we will remove Serial Number

# In[ ]:


graduation_data.drop(['Serial No.'],axis=1,inplace=True)

#Now we will be declare features and output
feature_X=graduation_data.drop(['Chance of Admit '],axis=1)
feature_Y=graduation_data['Chance of Admit ']


# In[ ]:


feature_X.head()


# #### As we can see that values of feature are very high so it is good practice to normalize the data

# In[ ]:


#Initialize sklearn MinMaxScalar
scaler =MinMaxScaler()
feature_to_normalize=feature_X.values
normalized_feature=scaler.fit_transform(feature_to_normalize)
# Create dataframe of normalized feature
df_normalized = pd.DataFrame(normalized_feature)
df_normalized.columns=feature_X.columns


# ### Data after normalization

# In[ ]:


df_normalized.head()


# ### Visualize data after normalization

# In[ ]:


f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(df_normalized.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# #### As from the above figure we can see that correlation coeffient of GRE Score, TOEFL Score and CGPA is same so we can keep one feature and remove other two

# In[ ]:


df_normalized.drop(['TOEFL Score','CGPA'],axis=1,inplace=True)


# ### Now our data is ready to be processed 

# In[ ]:


x=df_normalized
y=feature_Y


# ### Spliting the data in Train and Test Set

# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,shuffle='false')


# ### Creating Models
# #### We will be creating various models and will be using 'Mean Square Error' as measure of Accuracy

# ### Using Linear Regressor

# In[ ]:


model_LR=LinearRegression()
model_LR.fit(X_train,Y_train)

prediction=model_LR.predict(X_test)
print(f"Mean Square Error using Linear Regressor is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")


# ### Using DecisionTree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

model_DT=DecisionTreeRegressor()
model_DT.fit(X_train,Y_train)

prediction=model_DT.predict(X_test)
print(f"Mean Square Error using Decison Tree is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")


# ### Using RandomForestRegressor

# In[ ]:


model_RF=RandomForestRegressor()
model_RF.fit(X_train,Y_train)

prediction=model_RF.predict(X_test)
print(f"Mean Square Error using RandomForestRegressor is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")


# ### Using Kneighbors

# In[ ]:


model_KN=KNeighborsRegressor()
model_KN.fit(X_train,Y_train)

prediction=model_KN.predict(X_test)
print(f"Mean Square Error using Kneighbors is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")


# ### Using SVR

# In[ ]:


model_SVR=SVR()
model_SVR.fit(X_train,Y_train)
prediction=model_SVR.predict(X_test)
print(f"Mean Square Error using SVR is {(np.sqrt(mean_squared_error(Y_test, prediction)))}")


# ### Step 6:Conclusion
# #### Model with minimum 'Mean Square Error' is best fit for our data. So we would be using Linear Regresser with 0.7 Error
# 
# ### We have to use this model for prediction, so we would store our model using pickle.
# 
# #### Pickle Details: https://docs.python.org/3/library/pickle.html

# In[ ]:


import pickle
# print(os.listdir())
filename='admission_model.pkl'
# pickle.dump(model_LR, open("./Model/"+filename, 'wb'))

