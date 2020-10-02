#!/usr/bin/env python
# coding: utf-8

# Import all the necessary libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing the CSV file

# In[ ]:


df=pd.read_csv('../input/glass/glass.csv')


# Let us check the head of the file to understand the data 

# In[ ]:


df.head()


# In[ ]:


df.info()


# We can see that there are no  null values.

# In[ ]:


df.describe()


# Let us draw a correlation heat map to see the relation between different parameters

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# Let us visualize the content of different elements in the  various types of glass

# In[ ]:


sns.stripplot(x='Type',y='RI',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Na',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Mg',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Al',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Si',data=df)


# In[ ]:


sns.stripplot(x='Type',y='K',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Ca',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Ba',data=df)


# In[ ]:


sns.stripplot(x='Type',y='Fe',data=df)


# After visualization we cann see that the contents vary for different types of glass

# Let us import StandardScaler to normalize the data because  the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


scaler.fit(df.drop('Type',axis=1))


# In[ ]:


scaled_features=scaler.transform(df.drop('Type',axis=1))
df_head=pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_head,df['Type'], test_size=0.3, random_state=40)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


pred=knn.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print('Classification Report ',classification_report(y_test,pred))


# In[ ]:


print('Confusion Matrix',confusion_matrix(y_test,pred))


# Let us use the elbow method to choose the best value for K.

# In[ ]:


error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    error_rate.append(np.mean(pred_i!=y_test))
    


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('error_rate vs. K Value')
plt.xlabel('K')
plt.ylabel('error_rate')


# We can see that at K=1 we have low error rate.
# Hence we will perform the test with K=1

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:


pred=knn.predict(x_test)


# In[ ]:


print('Classification Report ',classification_report(y_test,pred))


# In[ ]:


print('Confusion Matrix',confusion_matrix(y_test,pred))


# We can see that at K=1 precision has increases 
# Hence selecting K=1 is justified 

# In[ ]:





# In[ ]:




