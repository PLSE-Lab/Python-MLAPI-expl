#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


dataset=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
dataset.head(3)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.dtypes.sample(12)


# In[ ]:


dataset["quality"].unique()
y=dataset["quality"]
x=dataset.drop("quality",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)


# In[ ]:


reg=linear_model.LogisticRegression()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_predict,y_test)


# In[ ]:


# Now i have seen few kernals where they have used Stochastic gradient descent classifier
# to increase the accuracy for the model

reg2=linear_model.SGDClassifier(penalty=None)
reg2.fit(x_train,y_train)
y_pred=reg2.predict(x_test)
accuracy_score(y_pred,y_test)


# In the cells above we have used SGDC and LR to classify the output.
# Now we will be splitting the output in two parts,either good or bad
# 
# 

# In[ ]:


bins=(2,5,8)
group_name=["bad","good"]
dataset["quality"]=pd.cut(dataset["quality"],bins=bins,labels=group_name)


# In[ ]:


dataset["quality"].head()
from sklearn.preprocessing import LabelEncoder
label_quality=LabelEncoder()
dataset["quality"]=label_quality.fit_transform(dataset["quality"])


# In[ ]:


dataset["quality"].head()

# Splitting the dataset again
y=dataset["quality"]
x=dataset.drop("quality",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)
accuracy_score(y_predict,y_test)


# In[ ]:


confusion_matrix(y_predict,y_test)


# In[ ]:


sns.pairplot(dataset)


# In[ ]:


sns.pairplot(dataset,hue="quality")


# In[ ]:


# Trying to see it accuracy greater than the previous one can be achieved with SGDC
reg2.fit(x_train,y_train)
y_pred=reg2.predict(x_test)
print(y_pred)
accuracy_score(y_pred,y_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# We can see that for logisti regression there was a huge jump in the accuracy after we have made ranges and provided labels
# 
