#!/usr/bin/env python
# coding: utf-8

# ## Avery Piraino - Predicting Successful Movies

# This notebook is divided into five sections: \
#     1. Data Cleansing \
#     2. Dependent Variable Creation\
#     3. Feature Engineering\
#     4. Insights to a Successful Movie\
#     5. Modeling 
# 
# I have interpretted the term "successful" to mean a movie that has a positive ROI and a movie score greater than 7. I have calculated ROI by dividing the "gross" columns from the "budget" column. I chose this definition due to the fact that all movie studios have the end goal of increasing profit. Therefore, a movie that can generate a positive ROI is likely a success. However, I think it is important to not consider poorly rated movies a success because it could deter from future profit. After collaborating with the studio owner, I might discover that he/she wants a high ROI and does not care about the movie score. This problem can be adjusted to meet the needs of the business owner.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
pd.set_option('display.max_columns', 50)


# In[ ]:


raw_data = pd.read_csv('the_zebra_movie_data.csv')


# In[ ]:


raw_data.head()


# # 1: Data Cleansing
# ## Now we will start doing some basic data quality checks. I will drop duplicate records and check for erroneous data.

# In[ ]:


raw_data = raw_data.drop_duplicates()


# After dropping duplicates it still seems that we have repeated movie titles. Upon looking into those differences, many of them had varrying numbers of facebook likes. We will take the mean of all numerical values to sort this issue out. In the case of categorical variables not matching, we are going to use the "first" function to arbitrarily select one.

# In[ ]:


numericalCols = list(raw_data.select_dtypes(include=['float64']).columns) + list(raw_data.select_dtypes(include=['int64']).columns)
for each in numericalCols:
    raw_data[each] = raw_data.groupby('movie_title')[each].transform('mean')


# In[ ]:


raw_data = raw_data.groupby('movie_title', as_index=False).first()


# In[ ]:


raw_data['movie_title'].value_counts().head()


# Here we have some distributions for our numerical columns. I am not going to remove outliers but I do want to remove values that should be impossible. An example of an impossible value in this case would be a negative number of facebook likes or a title year that is greater than the current year.

# In[ ]:


#If you would like to see all plots you can uncomment the line below. I am showing three features as to
#not overcrowd the notebook.

for each in numericalCols[:3]:
#for each in numericalCols:
    plt.hist(raw_data[each])
    plt.xlabel(each)
    plt.ylabel('Count')
    plt.show()


# There are some numbers that look strange, such as a minimum duration of 7 and a maximum of 511. However, nothing seems strictly impossible so I will move to the next step.

# In[ ]:


raw_data.describe() 


# # 2: Dependent Variable Creation
# ### Now that we have cleaner data, we will create the columns needed to describe success. We begin by calculating the ROI. We then add a column marking whether or not the ROI and movie rating thresholds have been met.

# In[ ]:


raw_data["roi"]  = (raw_data['gross']/raw_data['budget'])*100
#raw_data["net_profit"]  = raw_data['gross']-raw_data['budget'] -This could be another option to predict on instead of ROI


# It appears that we do not have a large amount of null data. However, since we are going to use the ROI and the movie score columns to determine a "successful movie", we need to remove rows that are null in these columns.

# In[ ]:


cols_nan = pd.DataFrame({'Total Null': raw_data.isnull().sum().sort_values(ascending=False).values,                        'Total NonNull': raw_data.count().sort_values().values,                        'Percent Null': (raw_data.isnull().sum()/raw_data.shape[0]).sort_values(ascending =False).values},                       index = raw_data.count().sort_values().index)
cols_nan.head()


# In[ ]:


raw_data = raw_data.dropna(axis=0, subset=['roi', 'movie_score'])


# In[ ]:


raw_data['success'] = np.where((raw_data['roi']>=0) &(raw_data['movie_score']>=7), 1, 0)


# Below is a graph demonstrating what percentage of movies are considered successful.

# In[ ]:


labels = "Successful", "Non-Successful"
explode = (.03, 0.0)
sizes = [raw_data[raw_data['success']==1].shape[0] ,raw_data[raw_data['success']==0].shape[0]]
plt.pie(sizes, explode=explode, labels =labels, autopct='%1.f%%')
plt.title("Distribution of Successful Movies")
plt.show()


# # 3: Feature Engineering
# ### We will now begin feature engineering work. First, we will deal with categorical features and then we will create some domain specific features.

# In[ ]:


categoryCols = list(raw_data.select_dtypes(include=['object']).columns)
categoryCols


# I will one hot encode any column that has five or fewer values. If a column has over five values, I would like to look at it manually. In this case, color is the only column that meets this criteria.

# In[ ]:


for columns in categoryCols:
    if len(raw_data[columns].value_counts()) <= 5:
        dummies = pd.get_dummies(raw_data[columns], prefix = columns)
        raw_data = pd.concat([raw_data, dummies], axis=1)
        raw_data = raw_data.drop([columns], axis = 1)


# Next we see columns such as "director_name" and "actor_name". I will be dropping them at this point. There are too many values to easily one hot encode. With more time, I would try target encoding or mapping to a "popularity" tier based on number of times they appear.

# In[ ]:


raw_data = raw_data.drop(['director_name','actor_1_name','actor_2_name','actor_3_name','plot_keywords'], axis=1)


# Next on our list is the "genre" column. This uses a pipe delimiter. We will remove the pipe and create a list, then one hot encode that list.

# In[ ]:


raw_data['genres'] = raw_data['genres'].str.split('|')
raw_data = raw_data.drop('genres', 1).join(raw_data.genres.str.join('|').str.get_dummies())


# I will assume that this movie studio is in the USA. Since language and country have many variables to one hot encode, I will categorize them as either being "foreign" or not.

# In[ ]:


raw_data['language'] = np.where(raw_data['language'].str.lower()=='english', 1, 0)
raw_data['country']= np.where(raw_data['country'].str.lower()=='usa', 1, 0)


# Using domain knowledge, we know that if a movie title ends in a number then it is a sequel. This means that previous movies were a success and could be a possible indicator of a succesful movie. This method does have some issues. For example, the movie "Star Wars: Episode VII - The Force Awakens" does not get marked as a sequel using this method. However, for this quick analysis this method should be sufficient.

# In[ ]:


raw_data['sequel'] = np.where(raw_data['movie_title'].str[-2:-1].str.isnumeric(), 1, 0)
raw_data = raw_data.drop('movie_title', axis = 1)                     


# When we look at the data in our content rating, the need for mapping stands out. The rating "X" is an older version of the rating "NC-17". I will rate a movie as "0 - Child Friendly", "1 - Teenager Friendly", or "2 - Adult Required". This mapping will be created using my interpretation of movie rankings and could be benefitted further with the guidence of a SME.

# In[ ]:


raw_data['content_rating'].value_counts()


# In[ ]:


content_map = {'R': 2, 'PG-13': 1, 'PG': 0, 'Not Rated' : 2, 'G': 0, 'Unrated' : 2, 'Approved' : 0,               'X': 2, 'Passed': 1,'NC-17':2, 'GP': 0 ,'M': 2}


# In[ ]:


raw_data['content_rating'] = raw_data['content_rating'].map(content_map)


# # 4: Insights to a Successful Movie
# ### Now that we have cleaned the data, created our dependent variable, and developed new features, we can look at some of the differences between these two groups.

# In[ ]:


#If you would like to see all plots you can uncomment the line below. I am showing three features as to
#not overcrown the notebook.

#for each in raw_data.columns:
for each in ['director_facebook_likes','Short','Thriller']:
    sns.factorplot('success', each, data=raw_data, kind='bar',ci= None)
    plt.show()


# # 5: Modeling
# ### Now we will continue with some basic modeling without hyper parameter tuning

# In[ ]:


#I will fill in null data with the respective column's average. This is something to look further into.
#In the case of facebook likes, a null might represent the fact that the actor does not have a facebook so they actually have 0 likes.
raw_data = raw_data.fillna(raw_data.mean())


# In[ ]:


from sklearn.model_selection import train_test_split
#We need to remove the dependent variable and the features that were used to calcuate it's value
X = raw_data.drop(['success','movie_score','roi', 'gross','budget'], axis=1)
y = raw_data['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# I am using sklearn's Gradient Boosted Classifier with the default parameters

# In[ ]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)


# In[ ]:


predictions = clf.predict(X_test)
predictions_train = clf.predict(X_train)

print("Confusion Matrix: ")
print(confusion_matrix(y_test, predictions))

print("[[True Non-Success, False Non-Success]")
print("[False Success, True Success]]")

print("Classification Report")
print(classification_report(y_test, predictions))


# Since our training auc is higher than the test auc, we can see that the model is slightly overfitting. From this point, we should begin hyperparameter tuning and introduce regularization as well as smaller trees. 

# In[ ]:


print(roc_auc_score(y_test, predictions))
print(roc_auc_score(y_train, predictions_train))


# ## In conclusion, we can return to the movie studio owner and offer him this model to predict whether a movie will be a success. He can also use the analysis by class that we created in section 4 to make more general assumptions. We could also offer to keep improving the model through continued feature selection and model tuning.

# In[ ]:




