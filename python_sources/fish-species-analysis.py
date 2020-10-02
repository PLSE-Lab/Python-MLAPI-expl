#!/usr/bin/env python
# coding: utf-8

# So here we are going to do the analysis on the type of fish we have based on the given features and remmember whenever you need to identify the cat,dog,type of fish or any other related classification then go for Logistic regression.
# So hope that you will like my way of doing!!!!:)

# In[ ]:


import pandas as pd
df = pd.read_csv("/kaggle/input/fish-market/Fish.csv")
df


# So since here the data is very small we can go for the heatmaps to see the correlation for the differnt coloumns which will be helping us very much to find which ones to choose for the prediction..!!!

# In[ ]:


import seaborn as sns
sns.heatmap(df.corr(),annot=True)


# So you can see above all the factors are very important here in order to determine the type of fish all have a correlation >0.8 and also we can apply feature engineering which i will be doing in upcomming vidoes which is also a very important concept.
# Well actually for beginners i think this is the best way:):):)

# In[ ]:


import plotly.express as px
fig = px.histogram(df, x="Height", y="Weight", color="Species")
fig.show()


# Getting the X and y for the training of the logistic regression in x exculding the species coloumn from df and y includes the species.

# In[ ]:


X= df.drop('Species',axis=1)
y = (df['Species'])


# Importing the logistic Regression and then using train test split to divide the data(80:20) and then fitting on our model the x_train and y_train....:):)

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# So you can see that we have got a great accuracy and yes we are all done !!
# Just relax and try guys seriously its a great way to learn:)
# Hope you enjoy my notebook !!!!!

# here is what our model predicted the actual name and the one predicted by model.....:)
# So we can now idntify the fish if we have above features Yipeee!!!!!!!!!!!

# In[ ]:


df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_new


# In[ ]:




