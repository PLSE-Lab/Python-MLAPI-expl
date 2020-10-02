#!/usr/bin/env python
# coding: utf-8

# First Lets import the libraries and read the data.

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
train=pd.read_csv('../input/exoTrain.csv')
test=pd.read_csv('../input/exoTest.csv')


# Lets check with data.

# In[ ]:


train.head()


# In[ ]:


test.head()


# Here all the data obtained from scientific instrument. Besides some instrumental error we hardly expect any other error. Woww then We dont have to do data cleaning.
# 
# Thats cool. We can do EDA but we had 3197 features and finding the right one is really gonna tough task.
# 
# > But as far as flux is considered, Every flux should have equal correlation with prediction. So one thing is clear that We gonna have all flux as features and we gonna train the model.
#  
#  But Wait which model we going to use to train?
#  
#  Now We got problem having binary distribution so We can use naive bayes algorithm with is good at that. 
#  
#  Then Lets do it check with accuaracy.

# In[ ]:


x_train=train.drop('LABEL',axis=1)
y_train=train[['LABEL']]
x_test=test.drop('LABEL',axis=1)
y_test=test[['LABEL']]


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
model=BernoulliNB(alpha=0.1)
model.fit(x_train,y_train)


# In[ ]:


prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,prediction)


# By using the naive bayes theorem we got 87% accuracy on testing set. Which is pretty good.

# In[ ]:


confusion_matrix(y_test,prediction)


#  But is there any other algorithm which will work best on this problem.
# 
# Lets think.
# 
# We got flux and We have to dicide the class to wich the particular star belong. Whether is it exoplanet or not? Now Can We somehow plot this in high dimention and we can get a result. 
# 
# > The closer star will have higher flux and the star which is far will have lower flux. Then We can easily do this. 
# > 
# Wait I know one algorithm Which works in same way. **The SVM.**
# 
# Yes in such scenario this is the best algorithm. 
# 
# Lets apply it and check with accuracy. 

# In[ ]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)


# In[ ]:


prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,prediction)


# Woww finally we got 99% accuarcy and If we had more training data We might have minimised that one. 
# 
# Lets check how many stars got misclassified. 

# In[ ]:


confusion_matrix(y_test,prediction)


# Confusion matrix tells us that only 5 stars got misclassified. This may be instrument error or outlier.
# 
# We will work on that in next kernel.
# 
# **Thank You for reading!!! Dont forget to upvote if you like.**
