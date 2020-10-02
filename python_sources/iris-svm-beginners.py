#!/usr/bin/env python
# coding: utf-8

# #Classify iris plants into three species in this classic dataset.
# #This notebook is for beginners who has a little required knowledge of data science.
# #Here i have tried to put things into simple way...And i hope this will help everyone.
# #DONT FORGET TO UPVOTE AND DO COMMENT IN CASE OF ANY DOUBTS AND PROVIDE FEEDBACK AND HELP ME IMPROVE.
# 

# In[ ]:


#Lets import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Now lets read the data
df=pd.read_csv(r'../input/iris/Iris.csv')
df.drop('Id',inplace=True,axis=1)  #drop Id column 
df.head()


# In[ ]:


#lets explore data
df.info()
df.describe()


# Now after exploring 'df.info()' we get to know that we have 6 data columns with no missing values as you can see their are 150 entries in each column.

# In[ ]:


#Now lets create a plot to know more about data
sns.set_style('darkgrid')
sns.pairplot(df,hue='Species',palette='Dark2')


# By the above plot we can interpret that Iris-setosa is the most separable species.

# In[ ]:


#Train Test Split
#now lets split train and test
from sklearn.model_selection import train_test_split

X=df.drop('Species',axis=1)
y=df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# In[ ]:


#Now lets train the model and predict
#Now its time to train a Support Vector Machine Classifier.
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)


# In[ ]:


#Now get predictions and model evaluation
pred=svc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))


# So we are done here.
# You should have noticed that your model was good but we also have to keep in mind the data set is quite small, if the data set is big and we dont get the good accuracy we can move forward using GridSearch.
# 
# "THAT'S IT GUYS...KEEP PRACTICING"
