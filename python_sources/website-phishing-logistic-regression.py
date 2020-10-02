#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Website phishing dataset problem.
# Fitting logistic regression and creating confusion matrix of predicted values and real values I was able to get 87.91% accuracy.
# 

# # STEP #0: Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # STEP #1: IMPORT DATASET

# In[ ]:


train=pd.read_csv('../input/Website Phishing.csv')
train.head()


# In[ ]:


a=len(train[train.Result==0])
b=len(train[train.Result==-1])
c=len(train[train.Result==1])
print(a,"times 0 repeated in Result")
print(b,"times -1 repeated in Result")
print(c,"times 1 repeated in Result")
sns.countplot(train['Result'])


# In[ ]:


sns.heatmap(train.corr(),annot=True)


# # STEP #2: Explore /Visualze Data set

# In[ ]:


sns.pairplot(train)
train.describe()


# # STEP #3: Prepare the Data for Training / Data Cleaning

# In[ ]:


train.info()


# In[ ]:


sns.heatmap(train.isnull(),cmap='Blues')


# # STEP #4: Model Training

# In[ ]:


X = train.drop('Result',axis=1).values 
y = train['Result'].values


# In[ ]:


y


# In[ ]:


X


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40,random_state=10)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# In[ ]:


from sklearn.linear_model import LogisticRegression

#create logistic regression object
Classifier=LogisticRegression(random_state= 0, multi_class='multinomial' , solver='newton-cg')
 
#Train the model using training data 
Classifier.fit(X_train,y_train)


# # STEP #5: Model Testing

# In[ ]:


#import Evaluation metrics 
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
 
#Test the model using testing data
predictions = Classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,predictions)
sns.heatmap(cm,annot=True)


# In[ ]:


print("f1 score is ",f1_score(y_test,predictions,average='weighted'))
print("matthews correlation coefficient is ",matthews_corrcoef(y_test,predictions))

#secondary metric,we should not consider accuracy score because the classes are imbalanced.

print('****************************************************************************************')
print("The accuracy of your Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,predictions))
print('****************************************************************************************')


# In[ ]:


for i in (0,len(predictions)-1):
    if(y_test[i] == predictions[i]):
        print('index = {} , y test = {} ,y predict ={}\n'.format(i,y_test[i],predictions[i]))


# In[ ]:





# In[ ]:




