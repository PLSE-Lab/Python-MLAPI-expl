#!/usr/bin/env python
# coding: utf-8

# # Kannada MNIST Using Random Forest Regressor

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fastai import *
from fastai.vision import *
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/Kannada-MNIST/train.csv')


# In[ ]:


df.head(10)


# In[ ]:


test=pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


k0=df.iloc[0].values
k1=df.iloc[1].values
k2=df.iloc[2].values
k3=df.iloc[3].values
k4=df.iloc[4].values
k5=df.iloc[5].values
k6=df.iloc[6].values
k7=df.iloc[7].values
k8=df.iloc[8].values
k9=df.iloc[9].values


# In[ ]:


zero=k0[1:].reshape((28,28))
one=k1[1:].reshape((28,28))
two=k2[1:].reshape((28,28))
three=k3[1:].reshape((28,28))
four=k4[1:].reshape((28,28))
five=k5[1:].reshape((28,28))
six=k6[1:].reshape((28,28))
seven=k7[1:].reshape((28,28))
eight=k8[1:].reshape((28,28))
nine=k9[1:].reshape((28,28))


# In[ ]:


number=['zero','one','two','three','four','five','six','seven','eight','nine']


# In[ ]:


plt.figure(1,figsize=(20,20))
#for i in range(431,440):
plt.subplot(431)
plt.imshow(zero);
plt.title('zero')
plt.colorbar()
plt.subplot(432)
plt.imshow(one);
plt.title('one')
plt.colorbar()
plt.subplot(433)
plt.imshow(two);
plt.title('two')
plt.colorbar()
plt.subplot(434)
plt.imshow(three);
plt.title('three')
plt.colorbar()
plt.subplot(435)
plt.imshow(four);
plt.title('four')
plt.colorbar()
plt.subplot(436)
plt.imshow(five);
plt.title('five')
plt.colorbar()
plt.subplot(437)
plt.imshow(six);
plt.title('six')
plt.colorbar()
plt.subplot(438)
plt.imshow(seven);
plt.title('seven')
plt.colorbar()
plt.subplot(439)
plt.imshow(eight);
plt.title('eight')
plt.colorbar()
plt.show()


# In[ ]:


plt.imshow(nine);
plt.colorbar()
plt.title('Nine')
plt.show()


# # Loading the classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


y=df['label'].copy()


# In[ ]:


y.head()


# # Data is Pixel form of 28x28

# In[ ]:


data=df.copy()
data=data.drop(['label'],axis=1)
data.head(15)


# # Splitting the DataSet into Train and Test

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(data,y,shuffle=True,test_size=0.3)


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


model=RandomForestClassifier(n_estimators=200,n_jobs=-1)
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)


# # Accuracy Score of 97.994 % without any parameter change in simple classifier

# In[ ]:


score=accuracy_score(y_test,y_pred)*100
score=round(score,4)
print('Accuracy Score is : '+str(score)+" %")


# In[ ]:


t1=X_test.iloc[0].values
t1=t1.reshape((28,28))
actual=y_test.values[0]
predicted=y_pred[0]
plt.imshow(t1);
plt.colorbar()
plt.title('actual : '+str(actual)+" / "+'predicted: '+str(predicted))
plt.show()


# In[ ]:


from random import randint


# In[ ]:


randint(20,100)


# # Plotting the predicted Numbers

# In[ ]:


for i in range(10):
    sha=randint(523,5232)
    t1=X_test.iloc[sha].values
    t1=t1.reshape((28,28))
    actual=y_test.values[sha]
    predicted=y_pred[sha]
    plt.imshow(t1);
    plt.colorbar()
    plt.title('actual : '+str(actual)+" / "+'predicted: '+str(predicted))
    plt.show()


# In[ ]:


test.head(10)


# In[ ]:


idcode=test.id.values


# In[ ]:


test=test.drop(['id'],axis=1)


# # Predicting test data

# In[ ]:


y_prediction=model.predict(test)


# In[ ]:


output=pd.DataFrame({'id':idcode,'label':y_prediction})


# In[ ]:


output.sample(n=20)


# In[ ]:


output.to_csv('submission.csv',index=False)


# # Thanks

# In[ ]:




