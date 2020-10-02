#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ## in this notebook, I will implement CatBoost to predict if a passenger of Titanic survived or not. 

# Importing the required libraries.

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier


# Loading the datasets

# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# ## **EDA** 
# on both datasets

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


print("******** TRAIN ********")
print(train_data.info())
print("******** TEST ********")
print(test_data.info())


# When looking above, I see that most of the rows of the column "Cabin" is null. Also, "PassengerID" and "Name" is not going to be super useful.
# 
# So, I decided to drop that columns on both datasets completely.
# 
# ** I will only save the PassengerId of test_data for my future submission. **

# In[ ]:


train_data.drop(axis=1,columns=(["PassengerId","Cabin","Name"]),inplace=True)

submission_passengers = test_data.PassengerId

test_data.drop(axis=1,columns=(["PassengerId","Cabin","Name"]),inplace=True)


# And I will continue by finding the best way for filling small amounts of null values.
# 
# On train.csv there are only 2 null rows on "Embarked" column.
# To fill this one, I will use most commonly used option from 3 different string objects. [ "S" , "C" , "Q" ]
# 
# Rest of null rows are on numerical Columns.
# To fill them, I will use some random but unused value. (e.g. -10)

# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Embarked CountPlot")
sns.countplot(data=train_data, x="Embarked")
plt.show()


# As we see above, "S" is most common value. I will use it to fill my null rows.

# In[ ]:


train_data.Embarked.fillna(value="S",inplace=True)


# Now I can fill everything left with a random negative numerical value.

# In[ ]:


train_data.fillna(value=-10,inplace=True)
test_data.fillna(value=-10,inplace=True)


# Our Dataset has no more null value now.

# In[ ]:


print("******** TRAIN ********")
print(train_data.info())
print("******** TEST ********")
print(test_data.info())


# ## Splitting
# 
# Now I can split features and labels for my future model.

# In[ ]:


train_y = train_data.Survived

train_data.drop(axis=1,columns=(["Survived"]),inplace=True)

train_x = train_data
test_x = test_data


# ## The CatBoost Model
# 
# cat_features should be equal to index values of columns that aren't float type.
# 
# learning_rate was optional so I made it 0.3 for fun :)

# In[ ]:


cat_model = CatBoostClassifier(cat_features=[0,1,3,4,5,7])


# Then I continue by training my model

# In[ ]:


cat_model.fit(train_x,train_y)


# Then predicting

# In[ ]:


predictions = cat_model.predict(test_x)


# Lastly, I need to save my predictions in format that is acceptable for the competition

# In[ ]:


my_submission = pd.DataFrame({'PassengerId': submission_passengers ,'Survived': (predictions.astype("Int64")) })


# To finish, I do export my submission file. :)

# In[ ]:


my_submission.to_csv("my_first_sub.csv",index=False)


# # Conclusion
# 
# CatBoost made my life so much easier with the "cat_features" Because I didn't have to deal with any Labelling, which is cool :)
# 
# 
# ## Thanks for reading and Best Regards, Efe.

# In[ ]:




