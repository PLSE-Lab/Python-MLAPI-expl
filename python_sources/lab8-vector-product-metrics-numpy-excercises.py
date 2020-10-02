#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.cross([1,0,0],[0,1,0])


# Returns vector perpendicular to both vectors and has length equal to the area spanned by the two vectors. Only defined for 3 dimensions but can be extended to any dimension. By giving $n-1$ vectors, you can have it return a vector perpendicular to all and with length equal to the volume spanned by the $n-1$ vectors. 

# In[2]:


v1 = np.array([3, 5, 4])
v2 = np.array([2, 1, 6])
np.cross(v1,v2)


# In[3]:


np.cross(v2,v1)


# In[4]:


np.cross(v2,v1+v2)


# In[5]:


np.cross(v2,v1+v2*5)


# Predicting a label. In a stupid way, just guessing. 

# In[6]:


true=np.random.randint(0,2,100)
prediction=np.random.randint(0,2,100)
from sklearn.metrics import confusion_matrix
confusion_matrix(true,prediction)


# The first column predicted 0, the second column predicted 1. The first row has truth 0, the second row has truth 1. 

# In[7]:


true


# In[8]:


prediction


# In[9]:


true==0


# In[10]:


(true==0)*(prediction==0)


# In[11]:


tn=sum((true==0)*(prediction==0)) # true negative
tn


# In[12]:


tp=sum((true==1)*(prediction==1)) # true positive
tp


# In[13]:


fp=sum((true==0)*(prediction==1)) # false positive
fp


# In[14]:


fn=sum((true==1)*(prediction==0)) # false negative
fn


# In[15]:


# accuracy
# proportion of correctly predicted
(tp+tn)/(tp+tn+fn+fp)


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(true, prediction)


# In[17]:


accuracy_score(1-true, 1-prediction)


# In[18]:


accuracy_score(true, prediction,normalize=False)


# In[19]:


tp+tn


# In[20]:


from sklearn.metrics import precision_score
precision_score(true,prediction)


# In[21]:


precision_score(1-true,1-prediction)


# In[22]:


# precision
# proportion actually true out of ones that were predicted true
precision=tp/(tp+fp)
precision


# In[23]:


14/(41+14),tn/(tn+fn)


# In[24]:


# recall
# otherwise known as sensitivity
# true positive rate
# of the positives, what proportion was correctly predicted to be positive
recall=tp/(tp+fn)
recall


# In[25]:


from sklearn.metrics import recall_score
recall_score(true,prediction)


# In[26]:


recall_score(1-true,1-prediction)


# In[28]:


14/(14+27)


# In[29]:


# specificity
# true negative rate
# proportion of negative correctly identified as such
tn/(fp+tn)


# In[30]:


# false positive rate
fp/(fp+tn)


# $F$ measure, harmonic mean of recall and precision. Inverse of the average of the inverse. If we have $\frac{1}{A}$ and $\frac{1}{B}$, the inverse is $A$ and $B$. The average of which is $\frac{A+B}{2}$. The inverse of which is $\frac{2}{A+B}$. 

# In[31]:


2/(1/recall+1/precision) # the larger the better


# In[32]:


from sklearn.metrics import f1_score
f1_score(true,prediction)


# In[33]:


f1_score(1-true,1-prediction)


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(true,prediction))


# In[35]:


print(classification_report(1-true,1-prediction))


# If your response and prediction are both real numbers...

# In[36]:


truth=np.random.normal(10, 3, 1000)
truth


# We use for prediction the same thing.

# In[37]:


pred=np.random.normal(10, 3, 1000)
pred


# In[39]:


from sklearn import metrics


# In[42]:


metrics.mean_absolute_error(truth, pred)


# In[43]:


abs(truth-pred).mean()


# In[40]:


metrics.mean_squared_error(truth, pred)


# In[41]:


((truth-pred)**2).mean()


# In[44]:


metrics.r2_score(truth, pred)


# In[47]:


metrics.r2_score(truth, np.repeat(10,1000))


# In[48]:


SST=sum((truth-truth.mean())**2)
SST


# In[49]:


SSR=sum((truth-pred)**2)
SSR


# In[50]:


1-SSR/SST


# In[52]:


SSR=sum((truth-np.repeat(10,1000))**2)
SSR


# In[53]:


1-SSR/SST


# The reference for below is [here](http://www.procrasist.com/entry/2-100-numpy-exercise) and [here](http://www.procrasist.com/entry/3-numpy-100-exercise).

# In[55]:


print(np.array([1,2,3,4,5]))


# In[57]:


print(np.zeros((3,3)))


# In[58]:


print(np.ones((3,3)))


# In[59]:


np.random.random(5)


# In[60]:


np.random.randint(0,5,10)


# In[61]:


np.random.uniform(0,1,5)


# In[62]:


np.random.normal(0,1,(3,3))


# In[63]:


np.arange(10)


# In[64]:


np.arange(1,10,2)


# In[65]:


np.linspace(0,10,10)


# In[66]:


np.linspace(0,10,10,endpoint=False)


# In[67]:


np.linspace(0,9,10)


# In[69]:


A = np.zeros((5,5), [('x',float),('y',float)])
A['x'], A['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(A)


# In[68]:


np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))


# In[70]:


def generate():
    for x in range(10):
        yield x
a = np.fromiter(generate(),dtype=float)
print(a)


# In[71]:


np.tile(np.array([[0,1],[1,0]]),(4,4))


# In[72]:


1-np.tile(np.array([[0,1],[1,0]]),(4,4))


# In[73]:


a = np.arange(10)
print(a)


# In[74]:


a.reshape(2,5)


# In[75]:


a.reshape(5,2).T


# In[76]:


a = np.arange(4)
b = np.arange(4)[::-1]
a,b


# In[77]:


print(a,b)


# In[78]:


print(np.hstack((a,b)))


# In[79]:


print(np.vstack((a,b)))


# In[80]:


a = np.arange(4)
print(a)


# In[81]:


print(a+1)


# In[82]:


A = np.arange(0,12).reshape(4,3)
B = np.arange(0,6).reshape(3,2)
print(A)
print("\n")
print(B)
print("\n")
print(A@B)


# In[83]:


a = np.arange(15)
print(a)
a[(a>3)&(a<=8)] *= -1
print("\n")
print(a)


# In[84]:


Z = np.random.uniform(0,10,10)
print("The original:")
print(Z)
print("Remove remainder:")
print(Z - Z%1)
print("Largest integer smaller than the number:")
print(np.floor(Z))
print("Smallest integer larger than the number:")
print(np.ceil(Z))
print("Change to int:")
print(Z.astype(int))
print("Only keep integer part:")
print(np.trunc(Z))
print("Return the integer part of a division:")
print(Z//1)      


# In[ ]:


5//2


# In[ ]:


# How to add fast
a = np.arange(100)
print(a)
import time
start1 = time.time()
print(np.add.reduce(a))
end1=time.time()-start1
start2 = time.time()
print(sum(a))
end2=time.time()-start2
print(end1,end2)


# In[ ]:


stone = np.zeros(10)
stone.flags.writeable = False
# stone[0]=1 will give error


# In[ ]:


x = np.zeros((32,32))
print(x)


# In[ ]:


np.set_printoptions(threshold=np.nan)
print(x)


# In[ ]:


A = np.arange(9).reshape(3,3)
print(A)


# In[ ]:


# ndenumerate returns coordinates and values
# nd is for n dimensional
start=time.time()
for index, value in np.ndenumerate(A):
    print(index, value)
print(time.time()-start)


# In[ ]:


#ndindex returns index
start=time.time()
for index in np.ndindex(A.shape):
    print(index, A[index])
print(time.time()-start)


# In[ ]:


A = np.arange(27).reshape(3,3,3)
print(A)


# In[ ]:


start=time.time()
for index in np.ndindex(A.shape):
    print(index, A[index])
print(time.time()-start)

