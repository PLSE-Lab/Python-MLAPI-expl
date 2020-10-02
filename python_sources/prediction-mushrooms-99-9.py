#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# 
# #### Predicting the mushrooms mainly through logistic regression
# #### Mushroom class was classified into two categories  poisonous(0) and edible(1)
# #### Steps taken in preprocessing includes Data cleaning,etc
# #### All our variables in this dataset are categorical
# 
# ### SIDE NOTE
# #### You can leave your question about any unclear part in the comment section
# #### Any correction will be highly welcomed

# In[ ]:


#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# ### LOADING THE DATASET

# In[ ]:


path = '/kaggle/input/mushroom-classification/mushrooms.csv'

df = pd.read_csv(path)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ### DEALING WITH MISSING VALUES

# In[ ]:


df.info()


# #### This dataset is clean it does not have any missing value

# ### DUMMY VARIABLES

# #### Our target column 'class' contains two unique values 'e' and 'p' which we will map to '1' and '0' respectively
# #### note 'e' stands for edible while 'p' for  poisonous

# In[ ]:


#Replacing e with 1
#Replacing  p with 0
df['class'] = df['class'].map({'e':1, 'p':0})


# In[ ]:


#Note that all our independent varibles are categorical variables
#Using attribute get_dummies to convert conver our category variables into dummy indicator
df_dummies = pd.get_dummies(df, drop_first = True)


# In[ ]:


df_dummies.head()


# ### LOGISTIC REGRESSION
# #### Here we will create, fit and train our model in addition to that we will try to predict using the already trained model

# In[ ]:


#Declaring our target variable as y
#Declaring our independent variables as x
x = df_dummies.drop('class', axis = 1)
y = df_dummies['class']


# In[ ]:


#Selecting the model
reg = LogisticRegression()


# In[ ]:


#Splitting our dataset into train and test datasets 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 24)


# In[ ]:


#We train the model with x_train and y_train
reg.fit(x_train, y_train)


# In[ ]:


#Predicting with our already trained model using x_test
y_hat = reg.predict(x_test)


# In[ ]:


#Mesuring the accuracy of our model
acc = metrics.accuracy_score(y_hat, y_test)
acc


# In[ ]:


#The intercept for our regression
reg.intercept_


# In[ ]:


#Coefficient for all our variables
reg.coef_


# ### CONFUSION MATRIX

# In[ ]:


cm = confusion_matrix(y_hat, y_test)
cm


# In[ ]:


# Format for easier understanding
cm_df = pd.DataFrame(cm)
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# #### Our model predicted '0' correctly 755 times while NEVER predicting '0' incorrectly 
# #### Also it predicted  '1'  correctly 869 times while predicting '1' incorrectly  ONCE

# ### OTHER MODELS

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # for K nearest neighbours
from sklearn import svm #for Support Vector Machine (SVM) 


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y1 = dt.predict(x_test)
acc1 = metrics.accuracy_score(y1, y_test)
acc1


# In[ ]:


kk = KNeighborsClassifier()
kk.fit(x_train,y_train)
y2 = kk.predict(x_test)
acc2 = metrics.accuracy_score(y2, y_test)
acc2


# In[ ]:


sv = svm.SVC()
sv.fit(x_train,y_train)
y3 = sv.predict(x_test)
acc3 = metrics.accuracy_score(y3, y_test)
acc3


# #### After comparison with some other model we see that Decision tree and KNeigbors both gave us 100% accuracy but our model was close enough with ~99.9% accuracy 

# ###  CONCLUSION
# #### Let's try to make a table and interpret what weight(BIAS) and odds means

# In[ ]:


pd.options.display.max_rows = 999
result = pd.DataFrame(data = x.columns, columns = ['Features'])
result['BIAS'] = np.transpose(reg.coef_)
result['odds'] = np.exp(np.transpose(reg.coef_))
result


# In[ ]:


#T be able to identify our refernce model
df['cap-shape'].unique()


# #### Using feature cap-shape as an example cap-shape_b is our baseline model i.e all cap-shape feature will be referenced to it and note that it is the only one that doesn't appear in the table above

# #### From the dataset description which can be found on kaggle 'b' represent bell while 's' represent sunken
# #### To interpret cap-shape_s in terms of odds we can say that  cap-shape_s  is almost twice more likely to cause a change in our target varable than cap-shape_b(our baseline model)

# #### If you find this notebook useful don't forget to upvote. #Happycoding

# In[ ]:




