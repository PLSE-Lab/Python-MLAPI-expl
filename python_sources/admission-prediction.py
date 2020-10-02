#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# 
# #### Predicting the chances of admition mainly through logistic regression
# #### Admit class was classified into two categories  0 and 1
# #### Steps taken in preprocessing includes Data cleaning, Standardizationetc
# #### All our variables in this dataset are numerical
# #### Other models where used to compare accuracy
# 
# ### SIDE NOTE
# #### You can leave your question about any unclear part in the comment section
# #### Any correction will be highly welcomed

# ### LOADING THE DATAFRAME

# In[ ]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head(3)


# ### DEALING WITH MISSING VALUES

# In[ ]:


df.info()


# #### This dataset is clean it does not have any missing value

# ### DUMMY INDICATOR
# #### Converting our target variable into a dummy indicator where a value greater than 0.5 chance of admit represents 1 else 0

# In[ ]:


df['admit'] =  np.where(df['Chance of Admit '] > 0.5,1,0)


# In[ ]:


df.head()


# In[ ]:


#Dropping useless variables
df.drop(['Chance of Admit ', 'Serial No.'], axis = 1, inplace = True)


# In[ ]:


df.head(3)


# In[ ]:


df.describe()


# ### CHECKING OLS ASSUMPTIONS
# 
# #### Let's check that our dataset are not violating any of this assumptions which includes:
# #### 1. No Endogeneity
# #### 2. Normality and Homoscedasticity
# #### 3.No Autocorrelation
# #### 4.NO multicollinearity: making sure our independents variables are not strongly related(correlated) with each other
# 
# ####  We are not violating  assumptions 1 through 3 but for NO multicollinearity we need to check

# In[ ]:


df.columns.values


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# the target column (in this case 'admit') should not be included in variables
#Categorical variables already turned into dummy indicator may or maynot be added if any
variables = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ',
       'CGPA',]]
X = add_constant(variables)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]
vif['features'] = X.columns
vif

#Using 10 as the minimum vif values i.e any independent variable 10 and above will have to be dropped
#From the results all independent variable are below 10


# ### Standardization
# 
# #### Standardizing helps to give our independent varibles a more standard and relatable numeric scale, it also helps in improving model accuracy

# In[ ]:


#Declaring our target variable as y
#Declaring our independent variables as x
y = df['admit']
x = df.drop(['admit'], axis = 1)


# In[ ]:


scaler = StandardScaler() #Selecting the standardscaler

scaler.fit(x)#fitting our independent variables


# In[ ]:


scaled_x = scaler.transform(x)#scaling


# ### LOGISTIC REGRESSION

# In[ ]:


#Splitting our data into train and test dataframe
x_train, x_test, y_train, y_test = train_test_split(scaled_x,y , test_size = 0.2, random_state = 49)


# In[ ]:


reg = LogisticRegression()#Selecting our model
reg.fit(x_train,y_train)


# In[ ]:


y_new = reg.predict(x_test) #Predicting with our already trained model using x_test


# In[ ]:


#Getting the accuracy of our model
acc = metrics.accuracy_score(y_new,y_test)
acc


# In[ ]:


#The intercept for our regression
reg.intercept_


# In[ ]:


#Coefficient for all our variables
reg.coef_


# ### CONFUSION MATRIX

# In[ ]:


cm = confusion_matrix(y_new, y_test)
cm


# In[ ]:


# Format for easier understanding
cm_df = pd.DataFrame(cm)
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# #### Our model predicted '0' correctly once while NEVER predicting '0' incorrectly
# #### Also it predicted '1' correctly 93 times while predicting '1' incorrectly 6 times
# 

# ### OTHER MODELS

# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

dnew = dt.predict(x_test)

acc2 = metrics.accuracy_score(dnew,y_test)
acc2


# In[ ]:


sv = svm.SVC() #select the algorithm
sv.fit(x_train,y_train) # we train the algorithm with the training data and the training output
y_pred = sv.predict(x_test) #now we pass the testing data to the trained algorithm
acc_svm = metrics.accuracy_score(y_pred,y_test)
print('The accuracy of the SVM is:', acc_svm)


# In[ ]:


knc = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
knc.fit(x_train,y_train)
y_pred = knc.predict(x_test)
acc_knn = metrics.accuracy_score(y_pred,y_test)
print('The accuracy of the KNN is', acc_knn)


# #### After comparison with some other model we see that Logistic regression gave us the highest accuracy ~94%

# ###  CONCLUSION
# #### Let's try to make a table and interpret what weight(BIAS) and odds means

# In[ ]:


df1 = pd.DataFrame(data = x.columns.values, columns = ['Features'])

df1['weight'] = np.transpose(reg.coef_)
df1['odds'] = np.exp(np.transpose(reg.coef_))
df1


# #### Remember we standardized all independents variables so the odds values have no direct interpretation
# #### Nevertheless using LOR as an example we can say for one standard deviation increase in LOR it is amost twice likely to cause a change in our target variable

# 
# 
# 
# 
# #### If you find this notebook useful don't forget to upvote. #Happycoding
# 

# In[ ]:




