#!/usr/bin/env python
# coding: utf-8

# # Hello and welcome to my notebook
# ## you will find:
#         * EDA
#         * Visual EDA
#         * K-Nearest-Neighbors approach to the data
#         * Comparison and Conclusion

# In[ ]:


# importing common libraries...

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# ## EDA

# In[ ]:


data_train = pd.read_csv("../input/train.csv")  # loading the data


# In[ ]:


data_train.info()  # checking for data types


# In[ ]:


print(list(data_train.any().isnull()))   # there is no null value in our columns, which is great


# In[ ]:


data_train.describe()   # I see that in most of the cases values are distributed between -1.00 and 1.00  ...


# In[ ]:


data_train.head(20)


# 
# # In Overview of this dataset, I realize that column "rn" is pretty much an id number of activity. But this is ruining the dataset !!! because labels grouped in order of "rn" column and this would effect our models very dramatically. (Which is basically cheating)
# 
# To be fair, I would rather making a model for real life scenarios without correlated ID numbers.  So, I will drop that "rn" column.

# In[ ]:


data_train.drop(["rn"],axis=1,inplace=True)    ## removing the rn column.


# In[ ]:


data_train.head(10)  ## As seen, problem solved!! :)


# ## Visual EDA

# In[ ]:


# importing data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(30,20))
sns.heatmap(data=(data_train.corr()*data_train.corr()),cmap="BuPu",vmin=0.4)
plt.show()

# what I do here is: only showing correlation x on => ((x^2)>0.4).  By this we only see highly correlated columns. and seems like there are many of them.


# As we see above, our data has too many columns and they are sensor results. So they are not practically the best data to viusalize.

# ### PreProcessing the data

# In[ ]:


# Let's see our labels.
labels = list(data_train.activity.unique())
print(labels) 


# In[ ]:


# they are strings, we should make them numerical values.  for this i will use scikit-learn

# importing and setting
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# fitting each possible label
le.fit(labels)
# updating our data with LabelEncoder
data_train.activity = le.transform(data_train.activity)


# In[ ]:


y_train = data_train.activity.values.reshape(-1,1)  # scikit doesn't likes when it is like (n,). it rathers (n,m).. that's why I used reshape


# In[ ]:


x_train = data_train.values 


# ## K Nearest Neighbors approach

# ### Before training my completed model, I want to check it's accuracy
# for this I will use train_test_split on my training data and then I will be able to compare

# In[ ]:


from sklearn.model_selection import train_test_split 
x_tr,x_tst,y_tr,y_tst = train_test_split(x_train,y_train,test_size=0.2,random_state=42)


# now I can create my initial model to estimate my final accuracy.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1,algorithm="auto")

knn_model.fit(x_tr,y_tr.ravel())

y_head = knn_model.predict(x_tst)


# In[ ]:


knn_model.score(x_tst,y_tst)


# ## As you can see above, model reached 97.6% of success.
# which is acceptable for me but we should tune and see the best amount of n_neighbors for our data

# # Comparison
# fo this my range will be 1 to 10 neighbors.

# In[ ]:


n = range(1,30)
results = []
for i in n:
    #print(i)
    knn_tester = KNeighborsClassifier(n_neighbors=i)
    knn_tester.fit(x_tr,y_tr.ravel())
    results.append(knn_tester.score(x_tst,y_tst))


# In[ ]:


plt.clf()

plt.suptitle("SCORES",fontsize=18)

plt.figure(figsize=(20,10))
plt.plot(n,results,c="red",linewidth=4)
plt.xlabel("n neighbors")
plt.ylabel("score")
plt.show()


# ## Conclusion
# 
# To sum up, I see that in some datasets, it is not very good to increase the amount of n_neighbors

# ### Here is the Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


conf = confusion_matrix(y_pred=y_head,y_true=y_tst)
conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(conf,annot=True,cmap="summer")
plt.show()


# In[ ]:





# In[ ]:


## I am currently improving my skills on data science. If you have any advice or comment, make sure you show it.  Best Regards.


# In[ ]:





# In[ ]:




