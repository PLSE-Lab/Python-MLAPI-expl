#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>G</font><font color='red'>o</font><font color='F2EA0D'>o</font><font color='blue'>g</font><font color='green'>l</font><font color='red'>e</font> Play Store Applications Analysis
# 
# ***
# # UPDATE - Fix These: 
# ## Issues found with the model/process used in this Jupyter Notebook:
# * When preprocessing the model, OneHotEncoding should have been used rather than the LabelEncoder. This is because a bias may exist between numbers in 0-33 rather than a matrix of 0 and 1 for category, genre, and content rating. 
# * A K-fold Cross Validation set should have been used to find tune the model before using the test set. 
# * Other models could have been used during the cross validation process to determine the best binary classifier model to use. 
# * ROC AUC or precision/recall curves should have been created to determine the best decision threshold for the model.
# 
# ***
# ## Contents
# 1. <font color='DE1212'> Research Questions & Motivation </font>
# 2. <font color='DE1212'> The Data </font>
# 3. <font color='DE1212'> The Exploration </font>
# 4. <font color='DE1212'> The Model - (Decision Tree Classifier) </font>
# 5. <font color='DE1212'> Conclusion </font>
# 
# ### Research Questions: 
# * <font color='DE1212'> Can you predict an app's popularity on the Google Play Store using a Decision Tree? </font>
# * <font color='DE1212'> If a developer were to create a new app, what qualities should this app have in order to generate the most ad revenue? </font>
# 
#     
# ### Motivation: 
# * <font color='DE1212'> Gain edge over the industry competition for app success. </font>
# * <font color='DE1212'> Provide insight for advertisement companies on which apps would generate the most revenue if ads were added. </font>
# * <font color='DE1212'> Assist Android developers to develop state-of-the-art apps that the public deserve. </font>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns #plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)


# ## The Data

# <font color="navy">Load the datasets into pandas dataframes</font>

# In[ ]:


df_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")
df_reviews.head()


# In[ ]:


df_apps = pd.read_csv("../input/googleplaystore.csv")
df_apps.head()


# <font color="navy">We will only be using the df_apps dataframe.</font>

# <font color="navy">Next, find the number of unique app categories.</font>

# In[ ]:


categories = list(df_apps["Category"].unique())
print("There are {0:.0f} categories! (Excluding/Removing Category 1.9)".format(len(categories)-1))
print(categories)
#Remove Category 1.9
categories.remove('1.9')


# <font color='navy'>Drop rows with Category "1.9" from dataframe. As seen below, this incorrectly labeled app category only affected one app. We can remove this row from the dataframe.

# In[ ]:


a = df_apps.loc[df_apps["Category"] == "1.9"]
print(a.head())
print("This mislabeled app category affects {} app at index {}.".format(len(a),int(a.index.values)))
df_apps = df_apps.drop(int(a.index.values),axis=0)


# In[ ]:


df_apps['Rating'].isnull().sum()


# <font color="navy">Delete rows that don't have any ratings.</font>

# In[ ]:


df_apps = df_apps.drop(df_apps[df_apps['Rating'].isnull()].index, axis=0)


# ## The Exploration

# In[ ]:


df_apps.info()


# <font color='navy'> As seen above, there are not any null values.</font>

# In[ ]:


df_apps["Rating"].describe()


# In[ ]:


layout = go.Layout(
    xaxis=dict(title='Ratings'),yaxis=dict(title='Number of Apps'))
data = [go.Histogram(x=df_apps["Rating"])]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic histogram')


# In[ ]:


#Show top 35 app genres
plt.figure(figsize=(16, 9.5))
genres = df_apps["Genres"].value_counts()[:35]
ax = sns.barplot(x=genres.values, y=genres.index, palette="PuBuGn_d")


# ### <font color="navy">Which categories have the best overall rating? Also, which category had the most installs? Let's find out!</font>

# In[ ]:


sns.set(rc={'figure.figsize':(20,10)}, font_scale=1.5, style='whitegrid')
ax = sns.boxplot(x="Category",y="Rating",data=df_apps)
labels = ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')


# <font color='navy'>All of the categories have close rating averages. In order to further define which categories are the highest rated, we will only look at the data for each category that has more than or equal to 4.0 in rating.</font>

# In[ ]:


#Cut away rows which have < 4.0 ratings
highRating = df_apps.copy()
highRating = highRating.loc[highRating["Rating"] >= 4.0]
highRateNum = highRating.groupby('Category')['Rating'].nunique()
highRateNum


# <font color='navy'> There are many categories of apps that are equal in terms of being the highest rated. This is great, however, the interest should lie within the app categories which have the lowest number of high ratings. These poorly rated apps deserve more attention because if a new sleek new app in that category were to be put on the app store, then the developers could satisfy the demand for innovation in this area. In this case,</font><font color='black'> **The Categories of Importance are "AUTO_AND_VEHICLES"and "ENTERTAINMENT."**

# ### <font color="navy">Now to analyze the apps which would produce the most ad revenue
# One parameter that would affect ad revenue the most is the number of installs an app has. More installs means more people are opening the app and viewing the embedded ads, hence, there is more money being made. A free application may lead to more installs, however, other parameters may alter how many installs an app will have. **Let's see if there is a correlation between installs and other parameters!**

# In[ ]:


df_apps.dtypes
df_apps["Type"] = (df_apps["Type"] == "Paid").astype(int)
corr = df_apps.apply(lambda x: x.factorize()[0]).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=True)


# Above, we can see that Installs and Reviews has the strongest inverse correlation. This is reasonable because more reviews are conducted on apps that are the most popular. Since Installs was not correlated to Type, this disproves our intuition that free apps lead to more installs. Since the Installs parameter is independent and not correlated to any other parameters, we must only use Installs to show the popularity of an app. Apps with larger amounts of installs would generate the most revenue.
# Let's take a look at the **Top 40 Apps that businesses should consider signing advertising deals with! **

# In[ ]:


#Extract App, Installs, & Content Rating from df_apps
popApps = df_apps.copy()
popApps = popApps.drop_duplicates()
#Remove characters preventing values from being floats and integers
popApps["Installs"] = popApps["Installs"].str.replace("+","") 
popApps["Installs"] = popApps["Installs"].str.replace(",","")
popApps["Installs"] = popApps["Installs"].astype("int64")
popApps["Price"] = popApps["Price"].str.replace("$","")
popApps["Price"] = popApps["Price"].astype("float64")
popApps["Size"] = popApps["Size"].str.replace("Varies with device","0")
popApps["Size"] = (popApps["Size"].replace(r'[kM]+$', '', regex=True).astype(float) *        popApps["Size"].str.extract(r'[\d\.]+([kM]+)', expand=False).fillna(1).replace(['k','M'], [10**3, 10**6]).astype(int))
popApps["Reviews"] = popApps["Reviews"].astype("int64")

popApps = popApps.sort_values(by="Installs",ascending=False)
popApps.reset_index(inplace=True)
popApps.drop(["index"],axis=1,inplace=True)
popApps.loc[:40,['App','Installs','Content Rating']]


# ## The Model - (Decision Tree Classifier)

# <font color="navy">In order to predict if an app will be successful, we must first determine what shows success. In this case a popular app has a high install value. The way in which we will go about preprocessing the data is by binarizing the Installs column. Anything above 100,000 will be considered equal to 1, and everything below that threshold will be equal to 0. This data split is not symmetric and will cause the model to be biased when predicting popularity of an app. We will pop off the enough values of each group to make a 50-50 training set, and the rest will be used for our test set. Also, we will encode the object labels of desired features.</font>

# In[ ]:


popAppsCopy = popApps.copy()
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'Category'. 
popAppsCopy['Category']= label_encoder.fit_transform(popAppsCopy['Category']) 
popAppsCopy['Content Rating']= label_encoder.fit_transform(popAppsCopy['Content Rating']) 
popAppsCopy['Genres']= label_encoder.fit_transform(popAppsCopy['Genres']) 
popAppsCopy.dtypes


# <font color="navy"> Since the important data is already preprocessed into floats and integers, we can drop the object features and build an 80/20 training/test split.</font>

# In[ ]:


popAppsCopy = popAppsCopy.drop(["App","Last Updated","Current Ver","Android Ver"],axis=1)
print("There are {} total rows.".format(popAppsCopy.shape[0]))
countPop = popAppsCopy[popAppsCopy["Installs"] > 100000].count()
print("{} Apps are Popular!".format(countPop[0]))
print("{} Apps are Unpopular!\n".format((popAppsCopy.shape[0]-countPop)[0]))
print("For an 80-20 training/test split, we need about {} apps for testing\n".format(popAppsCopy.shape[0]*.20))
popAppsCopy["Installs"] = (popAppsCopy["Installs"] > 100000)*1 #Installs Binarized
print("Cut {} apps off Popular df for a total of 3558 Popular training apps.".format(int(4568*.22132)))
print("Cut {} apps off Unpopular df for a total of 3558 Unpopular training apps.\n".format(int(4324*.17738)))

testPop1 = popAppsCopy[popAppsCopy["Installs"] == 1].sample(1010,random_state=0)
popAppsCopy = popAppsCopy.drop(testPop1.index)
print("Values were not dropped from training dataframe.",testPop1.index[0] in popAppsCopy.index)

testPop0 = popAppsCopy[popAppsCopy["Installs"] == 0].sample(766,random_state=0)
popAppsCopy = popAppsCopy.drop(testPop0.index)
print("Values were not dropped from training dataframe.",testPop0.index[0] in popAppsCopy.index)

testDf = testPop1.append(testPop0)
trainDf = popAppsCopy

#Shuffle rows in test & training data set
testDf = testDf.sample(frac=1,random_state=0).reset_index(drop=True)
trainDf = trainDf.sample(frac=1,random_state=0).reset_index(drop=True)

#Form training and test data split
y_train = trainDf.pop("Installs")
X_train = trainDf.copy()
y_test = testDf.pop("Installs")
X_test = testDf.copy()

X_train = X_train.drop(['Reviews', 'Rating'], axis=1) #REMOVE ROW TO INCLUDE REVIEWS & RATINGS IN ML MODEL ~93% accurate
X_test = X_test.drop(['Reviews', 'Rating'], axis=1)   #REMOVE ROW TO INCLUDE REVIEWS & RATINGS IN ML MODEL ~93% accurate


# In[ ]:


print("{} Apps are used for Training.".format(y_train.count()))
print("{} Apps are used for Testing.".format(y_test.count()))
X_test.head(3)


# ### <font color="navy"> Fit on Train Set </font>

# In[ ]:


popularity_classifier = DecisionTreeClassifier(max_leaf_nodes=29, random_state=0)
popularity_classifier.fit(X_train, y_train)


# ### <font color="navy"> Predict on Test Set </font>

# In[ ]:


predictions = popularity_classifier.predict(X_test)
print("Predicted: ",predictions[:30])
print("Actual:    ",np.array(y_test[:30]))


# ### <font color="navy"> Measure Accuracy of Classifier </font>

# In[ ]:


accuracy_score(y_true = y_test, y_pred = predictions)


# ### Find out what caused higher popularity

# <font color="navy">If different apps with the same app sizes are compared, we can see that the Category and the Genres columns are the only parameters that differ when determining popularity. Shown below, the 1 in the "Popular?" column may be an outlier, so as a whole, given all columns below, we can predict with ~72% accuracy the success of an app.

# In[ ]:


X_testCopy = X_test.copy()
X_testCopy["Popular?"] = y_test
X_testCopy[X_test["Size"] == 3600000].head(10)


# <font color="navy"> When running the kernel, the Accuracy of this Decision Tree Classifier will be about 95% (IF INCLUDING REVIEWS & RATINGS). When not including the rating and reviews features, the Classifier has around 72% Accuracy. This shows that **given the Size, Type, Price, Content Rating, and Genre of an app, we can predict within 72% certainty if an app will have more than 100,000 installs and be a hit on the Google Play Store.** </font>

# ## Conclusion

# * **<font color="DE1212"> For Innovation </font>** - **Developers** should focus in on apps with a category  of **Auto and Vehicles** and **Entertainment**, as there are not many highly rated apps in these categories.
# ***
# * **<font color="DE1212"> For Revenue </font>** - **Marketers** should advertise on the top 40 most installed apps list above, in order to reach the maximum viewing of their advertisements.
# ***
# * **<font color="DE1212"> For Popularity </font>** - **Everyone** building apps should consider that the Category and Genre of an app may strongly dictate if an app will be popular or not. However, the Size, Type, Price, Content Rating, and Genre features should all be used to most accurately determine if an app will gain maximum installs. 

# ## Model Limitations
# * Better performance for predicting app success would come by using alternative ML models. Random Forests, Logistic Regression, and K Nearest-Neighbors would be great models to test against Decision Trees. More than likely, we could achieve up to 80% accuracy using one of these models rather than the current ML model. 
