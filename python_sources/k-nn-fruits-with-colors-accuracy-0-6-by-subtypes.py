#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
import seaborn as sn
from sklearn.model_selection import train_test_split


# In[ ]:


fruits = pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')
fruits.head()


# In[ ]:


# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
lookup_fruit_name


# In[ ]:


fruits.shape


# In[ ]:


for i in fruits.fruit_label.unique():
    print(i,":",len(fruits[fruits['fruit_label']==i]),"(",lookup_fruit_name[i],")")


# In[ ]:


fruits2=fruits[:]


# In[ ]:


fruits2.fruit_subtype.unique()


# In[ ]:


fruits2.fruit_subtype.unique()[0]


# In[ ]:


lookup_fruit_name2 = dict()
lookup_fruit_name2


# In[ ]:


c = fruits2.fruit_subtype.unique()
cc = len(c)
cc


# In[ ]:


for i in range(cc):
    lookup_fruit_name2[fruits2.fruit_subtype.unique()[i]] = i
lookup_fruit_name2


# In[ ]:


o = fruits2.fruit_subtype
oo = len(o)
fruit_label2 = np.zeros(oo)

for i in range(oo):
    p = o[i]
    fruit_label2[i] = lookup_fruit_name2[o[i]]

fruit_label2 = np.array(fruit_label2, dtype=int)
fruit_label2


# In[ ]:


fruits4 = fruits2.assign(fruit_label2 = fruit_label2)
fruits4


# In[ ]:


fruits4.shape


# In[ ]:


fruits4.head()


# In[ ]:


def reverse_dict(x):
    q = list(lookup_fruit_name2.keys())[list(lookup_fruit_name2.values()).index(x)]
    return q


# In[ ]:


reverse_dict(8)


# In[ ]:


for i in fruits4.fruit_label2.unique():
    print(i,":",len(fruits4[fruits4['fruit_label2']==i]),"(",reverse_dict(i),")")


# In[ ]:


list(lookup_fruit_name2.keys())[list(lookup_fruit_name2.values()).index(8)]


# In[ ]:


len(fruits4[fruits4['fruit_label2']==0])


# In[ ]:


# For this example, we use the mass, width, and height features of each fruit instance
X = fruits4[['mass', 'width', 'height']]
y = fruits4['fruit_label2']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm
fruit_prediction = knn.predict([[100,6.3,8.5]])
# lookup_fruit_name2[fruit_prediction[0]]
fruit_prediction[0]


# In[ ]:


reverse_dict(fruit_prediction[0])


# In[ ]:


# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
fruit_prediction = knn.predict([[20,4.3,5.5]])
reverse_dict(fruit_prediction[0])


# In[ ]:


fruit_prediction = knn.predict([[180,6.5,5]])
reverse_dict(fruit_prediction[0])


# In[ ]:


fr_pred = knn.predict(X_test)
fr_pred


# In[ ]:


fr_pred.shape


# In[ ]:


y_test


# In[ ]:


(fr_pred,np.array(y_test))


# In[ ]:


# its not the end, see below
knn.score(X_test,y_test)


# In[ ]:


fruits4


# In[ ]:


fruits4.fruit_label2.unique()


# In[ ]:


lookup_lname = dict()
lookup_lname


# In[ ]:


c = fruits4.fruit_label2
cc = len(c)
cc


# In[ ]:


for i in range(cc):
    lookup_lname[fruits4.fruit_label2[i]] = fruits4.fruit_label[i]
lookup_lname


# In[ ]:


fr_pred


# In[ ]:


y_te = np.array(y_test)
y_te


# In[ ]:


fr_pred = [lookup_lname[i] for i in fr_pred]
fr_pred


# In[ ]:


y_te = [lookup_lname[i] for i in y_te]
y_te


# In[ ]:


(fr_pred,y_te)


# In[ ]:


z = [a - b for a, b in zip(fr_pred,y_te)]
z


# In[ ]:


len(z)


# In[ ]:


z.count(0)


# In[ ]:


#accuracy
accur = (z.count(0) / len(z))
accur

