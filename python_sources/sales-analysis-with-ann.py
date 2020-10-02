#!/usr/bin/env python
# coding: utf-8

# <center><h1>BLACK FRIDAY SALE ANALYSIS</h1></center>

# ![](https://www.thesierraleonetelegraph.com/wp-content/uploads/2018/11/black-friday-sale.jpg)

# The dataset here is a sample of the transactions made in a retail store. The store wants to know better the customer purchase behaviour against different products. Specifically, here the problem is a regression problem where we are trying to predict the dependent variable (the amount of purchase) with the help of the information contained in the other variables.

# <h2>So let's get started...</h2>
# <p>We start with importing our favourite libraries</p>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
import os
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))


# In[ ]:


original_data = pd.read_csv('../input/BlackFriday.csv')


# Now let's have a look on how the data looks like and explore its datatypes.

# In[ ]:


original_data.head()


# In[ ]:


original_data.columns


# In[ ]:


original_data.describe()


# In[ ]:


original_data.info()


# <p>From above it seems like we have some null values residing out there.</p>
# <p>Lets find out in more detail :</p>

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(16,9))
sns.heatmap(original_data.isnull(),cmap="viridis",cbar=False,yticklabels=False)


# From above its clear that a large chunk of data is not available for 'Product_Category_2' and 'Product_Category_3'.
# <p>We will handle this part later in this kernel.</p>

# <h2>PART 1:- Data Analysis and Visualisation</h2>

# In[ ]:


plt.figure(figsize=(16,9))
sns.distplot(original_data['Purchase'],bins=80,kde=False)


# **Takeaway** :- Most of the purchase amount of the shopping lies between 5000-10000 (Hmm..interesting).

# In[ ]:


sns.countplot(x='Gender',data=original_data,hue='Marital_Status')


# **Takeaway** :- Most of the customers who came for shopping are males and unmarried. 
# <h3>Well this is getting even more interesting...</h3>

# In[ ]:


plt.figure(figsize=(16,9))
sns.countplot(x='Age',data=original_data,hue='Gender')


# **Takeaway** :- Most of the customers belong to the age group of 26-35 which is quite obvious as most of the customers who came for shopping were unmarried.

# In[ ]:


plt.figure(figsize=(16,9))
sns.countplot(x='Occupation',data=original_data,hue='Gender')


# **Takeaway** :- For this we cannot say much as we are unware of the occupation here.However its clear that people are in Occupation 4 and 0 outnumbered the others which some what means that these are well paid jobs or they have enough time for shopping.
# Also, we can witness that male workers are outnumbering the female workers.

# In[ ]:


plt.figure(figsize=(16,9))
sns.boxplot(x='Age',y='Purchase',data=original_data)


# **Takeaway** :- The purchase amount in each age group is almost similar which means most of the products which were on sale were general items that could be used by all age groups.
# <br/>
# **But wait a moment...**
# We already know that mostly the sale was attended by the age group of 26-35 and the age group which had the least number of population is 0-17, however the twist here is that there purchase amount is almost similar to the others.

# In[ ]:


plt.figure(figsize=(16,9))
sns.violinplot(x='Occupation',y='Purchase',data=original_data)


# **Takeaway** :-  The purchase amount is also well distributed among the occupation.However we know the fact that most of the people who came for shopping were from occupation 4 and 0 and still the purchase range for these groups is similar to others.
# <br/>
# And the group belonging to the Occupation 13 despite being least in number have purchased amount similar to others.Now that means that **Occupation 13** is the most highly paid job here.

# In[ ]:


sns.countplot(x='City_Category',data=original_data)


# **Takeaway** :- Most of the people who came for shopping hail from City B, from which we can infer that the sale was in City B or people of city B are richer as compared to other cities.

# In[ ]:


sns.violinplot(x='City_Category',y='Purchase',data=original_data)


# **Takeaway** :- The purchasing amount is well distributed among all cities but as we know from above that the least number of people who came for shopping were from city A.
# Despite this fact there purchasing amount is similar to others which means people from city A spent more which means they are richer than others.

# In[ ]:


plt.figure(figsize=(16,9))
sns.countplot(x='Stay_In_Current_City_Years',data=original_data)


# **Takeaway** :- Most of the people who came for shopping had recently shifted to the city which is quite obvious as lots of things are required to set up the home after shifting.

# In[ ]:


plt.figure(figsize=(16,9))
sns.boxplot(x='Stay_In_Current_City_Years',y='Purchase',data=original_data)


# <h3>Top ten most sold items</h3>

# In[ ]:


plt.figure(figsize=(16,9))
data = original_data['Product_ID'].value_counts().sort_values(ascending=False)[:10]
sns.barplot(x=data.index,y=data.values)


# <h3>Top ten most valuable customers</h3>

# In[ ]:


plt.figure(figsize=(16,9))
data = original_data['User_ID'].value_counts().sort_values(ascending=False)[:10]
sns.barplot(x=data.index,y=data.values)


# <p>The data exploration was very interesting, is'nt it ?</p>
# <h3>Now lets move on and try to build a prediction model based upon this</h3>

# Lets recollect our data first...

# In[ ]:


original_data.columns


# <h2>PART 2:- Data Preprocessing</h2>

# We now start with processing our data.
# <p>At first we will fill the missing data of the Product_Category_2 and Product_Category_3 with 0 because the data these columns refer to the quantity of products belonging to those category and so no data available for them means the purchase did not include any products from that category,therefore we can replace NaN with 0 here.</p>

# In[ ]:


original_data['Product_Category_2'].fillna(0, inplace=True)
original_data['Product_Category_3'].fillna(0, inplace=True)


# We also need to change there datatypes from float to int.

# In[ ]:


original_data['Product_Category_2'] = original_data['Product_Category_2'].astype(int)
original_data['Product_Category_3'] = original_data['Product_Category_3'].astype(int)


# Now we replace the string '4+' with integer 4 and changing the datatype to int.

# In[ ]:


original_data.Stay_In_Current_City_Years = original_data.Stay_In_Current_City_Years.replace('4+',4)
original_data['Stay_In_Current_City_Years'] = original_data['Stay_In_Current_City_Years'].astype(int)


# Lets have a look what we have done to our data so far...

# In[ ]:


original_data.head()


# We now remove the first two columns of data as they are not required here.

# In[ ]:


X = original_data.iloc[:,2:11].values
y = original_data.iloc[:,11].values


# We start with **Label Encoding** our data to make this data machine readable.

# In[ ]:


lb_x_1 = LabelEncoder()
X[:,0] = lb_x_1.fit_transform(X[:,0])
lb_x_2 = LabelEncoder()
X[:,1] = lb_x_2.fit_transform(X[:,1])
lb_x_4 = LabelEncoder()
X[:,3] = lb_x_2.fit_transform(X[:,3])
lb_x_3 = LabelEncoder()
X[:,2] = lb_x_3.fit_transform(X[:,2])


# We also apply **OneHotEncoding** to some of our columns. Please note we have not included Gender column here as it is not required beacuse it is already in binary form of 0 and 1.

# In[ ]:


onh = OneHotEncoder(categorical_features=[1,2,3])
X = onh.fit_transform(X).toarray()


# Now a major part is done in data preprocessing and now we move to splitting our data in **training** and **test** sets.

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# With that done we have completed the part 2 of this kernel.
# <br/>
# Now we will move to the next part where we will develop our model using **Artificial Neural Networks**.
# <br/><br/>
# <h2>PART 3 :- Building model  in ANN</h2>

# In[ ]:


model = Sequential()

# The Input Layer :
model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()


# We will also add Checkpoint callback to our model

# In[ ]:


checkpoint_name = '{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[ ]:


def get_best_weight(num_ext, ext):
    tmp_file_name = os.listdir('.')
    test = []
    num_element = -num_ext
    all_weights_file = [k for k in tmp_file_name if '.hdf5' in k]
    for x in range(0, len(all_weights_file)):
        test.append(all_weights_file[x][:num_element])
        float(test[x])

    lowest = min(test)
    return str(lowest) + ext


# <h3>Lets rotate the wheel now ! </h3>
# We start with the training now...

# In[ ]:


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# From the above result we get the best Weights file.

# In[ ]:


weights_file = get_best_weight(5, ".hdf5")
model.load_weights(weights_file)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# <h3>And the magic begins ! </h3>
# <p>We will now make predictions on the Test set</p>

# In[ ]:


predictions = model.predict(X_test)


# **Well enough** ! Lets visualise what we have done

# In[ ]:


plt.figure(figsize=(16,9))
plt.scatter(y_test[:500],predictions[:500]) #Taken only 500 points here for better visualisation
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# <h3>And finally lets print out some of the evaluation metrics for our model.</h3>

# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# <center>For any clarifications and suggestions please comment below. <br/>**  Also if you like this kernel please vote for it as this will keep me motivated  **.</center>

# <center><h2>------------------------ Thank You ------------------------</h2></center>
