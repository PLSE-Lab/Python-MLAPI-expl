#!/usr/bin/env python
# coding: utf-8

# # Objectives of this Kernel:
# - Objective 1 :  Checking out popular applications in each genre in Apple Store
# - Objective 2 : Checking the trend of an App's **User Experience** with respect to it's **cost**, **User Rating count** and ** Number of devices and Languages**  it supports
# - Objective 3 :  Judging a game's popularity by it's APK size and make a **Random Forest Classifier** to classify by popularity
# 
# Hope that you would enjoy your EDA and ML journey with me!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


myDf = pd.read_csv('../input/AppleStore.csv', index_col = 'id')
myDf.drop(columns = ['Unnamed: 0'], inplace=True)
myDf.head()


# # EDA Objective 1
# 
# Checking out the most popular applications in Apple Store of each genre and checking how Number of User ratings is a better factor for judgement rather than solely user ratings!

# In[ ]:


# A utility to convert bytes to megabyte
def byteToMB(b):
    MB = b/(1024.0*1024)
    return MB

    


# In[ ]:


myDf['size_in_mb'] = myDf['size_bytes'].apply(byteToMB) 
myDf.drop(columns=['size_bytes'], inplace=True)


# In[ ]:


# Updated Dataframe
myDf.head()


# In[ ]:


myDf['prime_genre'].unique()


# Now, let's look out for the best 5 User Rated App in each genre using Solely **User Ratings**
# 

# In[ ]:


for i in myDf['prime_genre'].unique():
    newVar = myDf[myDf['prime_genre'] == i]
    newVar.sort_values(by = ['user_rating'], inplace = True)
    print("Top 5 for {} genre are".format(i))
    print (newVar['track_name'][::-1][:6])
    print("\n")
    


# But now, let's try and include the popylarity of the application using **rating_count** of at least 1 million

# In[ ]:


for i in myDf['prime_genre'].unique():
    refinedDf = myDf[(myDf['rating_count_tot'] > 50000) & (myDf['prime_genre'] == i)]
    refinedDf.sort_values(['user_rating','rating_count_tot'], inplace = True)
    print("Top 5 for {} genre are".format(i))
    print (refinedDf['track_name'][::-1][:6])
    print ("\n")
    


# ### See the Difference?
# In the above two cells, one can see how important Popularity and User Ratings are in terms of defining the true success of the App. 
# But now let's explore some Free versions of the same, because right now, it's including both free and paid versions. Also, let's try and lower the popularity bar a little

# In[ ]:


for i in myDf['prime_genre'].unique():
    refinedDf = myDf[(myDf['rating_count_tot'] > 20000) & (myDf['prime_genre'] == i) & (myDf['price'] == 0.00)]
    refinedDf.sort_values(['user_rating','rating_count_tot'], inplace = True)
    print("Top 5 for {} genre are".format(i))
    print (refinedDf['track_name'][::-1][:6])
    print ("\n")
    


# # EDA Objective 2
# Checking the trend of an App's **User Experience** with respect to it's **cost**, **User Rating count** and ** Number of devices and Languages**  it supports

# In[ ]:


eda2df = myDf[myDf['price'] == 0.00]
eda2df.sort_values(by = ['sup_devices.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num']][::-1].head(10)


# The above Dataframe gives us the top 5 best device versatile apps, but are they any popular? I doubt it. Now let's try out one with most language support

# In[ ]:


eda2df = myDf[myDf['price'] == 0.00]
eda2df.sort_values(by = ['lang.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num']][::-1].head(10)


# In this department of Free and most language sopportive apps, Google clearly stands out in the competition! But, what about the paid apps? Let's have a look at them too

# In[ ]:


eda2df = myDf[myDf['price'] != 0.00]
eda2df.sort_values(by = ['sup_devices.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num', 'price']][::-1].head(10)


# Do the apps sound familiar? They should. Most of them are gaming applications as expected.

# In[ ]:


eda2df = myDf[myDf['price'] != 0.00]
eda2df.sort_values(by = ['lang.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num', 'price']][::-1].head(10)


# Interesting, In paid apps, Tinybop has been pretty busy I guess.!

# ## Looking for Linear Patterns
# As we have seen so far, there is no such famous pattern as yet. How about trying out to find a pattern between the numerical columns? It's worth a try!

# In[ ]:


numCol = myDf[['rating_count_tot', 'user_rating', 'sup_devices.num', 'price', 'lang.num', 'prime_genre']]
sns.pairplot(data = numCol, dropna=True, hue='prime_genre',palette='Set1')


# Okay, You can't pull off any linear assesment in this dataset as you might have noticed already by looking at the pair plots, But Some interesting patterns worth noticing is the pattern between language support and price of the application
# 
# You may like to appreciate the following fact that most of the apps are free (have a look at the graph given below)

# In[ ]:


sns.set_style("darkgrid")

plt.hist(myDf['price'], bins = 100)


# # EDA Objective 3
# Now, Judging an App's popularity by it's APK size. 
# 
# **My judgement criteria would be as follows : **
# * If an app has Number of User ratings < 10k, it will be rated as poor or 0
# * If an app has Number of User ratings >= 10k but < 100k , it will be rated as average or 1
# * If an app has Number of User ratings >= 100k, it will be rated as popular or 2
# 
# So now let's prepare the dataset

# In[ ]:


# A utility function to create categories according to views
def df_categorizer(rating):
    if rating >= 100000:
        return 2
    elif rating < 10000:
        return 0
    else:
        return 1


# Here, I am just going to provide a simple category  mean values for Application Size in MBs and observe the mean values for the same.
# As you might have noticed, the **average**  mean values in terms of sizze doesn't really differ much. But still, let's see how does our Classifier fare!

# In[ ]:


myDf['pop_categories'] = myDf['rating_count_tot'].apply(df_categorizer)

finalDf = myDf[['size_in_mb', 'rating_count_tot', 'pop_categories']]
finalDf.groupby(['pop_categories']).mean()


# In[ ]:


# Importing the tasty stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# In[ ]:


X = finalDf['size_in_mb']
y = finalDf['pop_categories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)


# Reshaping the array as per the needs of single feature processing in **sklearn** random forest classifier.

# In[ ]:


npX_train = np.array(X_train)
npX_train = npX_train.reshape(-1,1)

npX_test = np.array(X_test)
npX_test = npX_test.reshape(-1,1)


# # Scaling the input, of course, for quicker training

# In[ ]:


scaler = StandardScaler()

npX_train = scaler.fit_transform(npX_train)
npX_test = scaler.transform(npX_test)


# # Where The classifier begins
# 
# Making a  random forest classifier with **'entropy'** criteria and total estimators amounting to 10

# In[ ]:


classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=42)
classifier.fit(npX_train, y_train)


# # Confusion Matrix and Accuracy
# Once the classifier is trained, we predict our test inputs and checkout the confusion matrix!

# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(npX_test)

#Reverse factorize (converting y_pred from 0s,1s and 2s to poor, average and popular
reversefactor = dict(zip(range(3),['poor', 'average', 'popular']))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix
cnf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species'])


# In[ ]:


cnf_matrix


# # Conclusion
# Looking at the matrix, we can easily figure out that out of 1800 inputs, it has correctly classified 1379 inputs which amounts it to an accuracy of **76.61 %**. 
# 
# Although, if we focus closely into the case here, you may have already noticed that this classifier well mostly in the case of poor popularity of application : hitting an accuracy of **87.84%** but is really bad at classifying popular apps (an accuracy of only **2.04%** )
# 
# So, one may safely conclude that looking solely at the aspect of application size as predicting feature can backfire really bad for any classification purpose. Although, if you are still interested in making this work, I'd suggest you to consider the **User Rating** as an input feature along with the application size.
# 
# It would be great to experiment with the popularity dependence and learning capability of our model when looking at **language support**,  **device support** and **development version** of the app as possible input features!
# 
# Any constructive suggestion is welcome!
# 
# 

# In[ ]:




