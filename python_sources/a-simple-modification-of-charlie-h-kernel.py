#!/usr/bin/env python
# coding: utf-8

# $\text{A review of Charlie H. intro for image classification in python from his/her kaggle kernal.}$
# 
# $\text{https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification}$
# 
# $\text{Rewritten/ modfied  by EG Timerise (aka Dr.E for those on kaggle )}$
# 
# $\text{From the data we are given a list of pixals and labels.}$

# In[ ]:


import pandas as pd
train=pd.read_csv('../input/train.csv')


# In[ ]:


test=pd.read_csv('../input/test.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


subtrain=train.iloc[:400,1:]
subresponse=train.iloc[:400,:1]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_vaild,y_train,y_vaild=train_test_split(subtrain,subresponse,test_size=.2,stratify=subresponse,random_state=42)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpi
def imageplots(k):
    viewed_image=x_train.iloc[k,:].as_matrix()
    #28*28= 784 which is the number of pixels used per image
    # So like the previous kernal it really is pushing 1D to 2D
    # Although im not sure if thats usually how its done
    im=viewed_image.reshape((28,28))
    plt.title(y_train.iloc[k])
    
    return plt.imshow(im,cmap='binary')


# In[ ]:


imageplots(0)


# In[ ]:


imageplots(1)


# In[ ]:


imageplots(5)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler()
dtr=scal.fit_transform(x_train)
x_train_s=pd.DataFrame(dtr,columns=x_train.columns)
dvr=scal.transform(x_vaild)
x_vaild_s=pd.DataFrame(dvr,columns=x_vaild.columns)
dtt=scal.transform(test)
test_s=pd.DataFrame(dtt,columns=test.columns)


# In[ ]:


from sklearn import svm

from sklearn.model_selection import GridSearchCV

parmer_grid={'C':range(1,10),'gamma':[0.01,0.0001,1/785,0.00001]}

sv=svm.SVC()

vt=GridSearchCV(sv,parmer_grid,cv=10)

vt.fit(x_train_s,y_train.values.ravel())

vt.score(x_vaild_s,y_vaild)


# $\text{I read some of the comments about his kernal and thought it would be quick and easy to apply some of the suggestions.}$
# 
# $\text{I know his some of the messages are old, but thats life.}$
# 
# $\text{ Its a simple example of the use of grid search for finding better model parameters.}$
# 
# $\text{The model is still pretty bad , but its a really good start.}$
