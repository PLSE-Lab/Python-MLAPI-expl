#!/usr/bin/env python
# coding: utf-8

# # Analysis of UK Traffic Accidents and Predicting the Accident Severity
# ###### Ravinder Singh          
# 

# ### Introduction
# 
# This project is analysis on U.K accidents data from year 2005 to 2015. There are some questions that can be answered using this data such as -
# - What are the regions or areas with most frequent accidents?
# - What kind of street or highways are more liekly to have accidents?
# - What are the age group are most likely to be involved in accidents?
# - What are the areas with higher accident severity or lower accident severity?
#    
# There are endless questions that can be answered with this dataset. We will be answering few of the questions as I mentioned above. We will also figure out some way to implement the machine learning on this dataset and see what we can come up with.
# 
# 
# The data comes from governemnt website www.data.gov.uk. UK police forces collect the accidents data using the form called Stats19. The data consists of all kind of vehicle collisons from 2005 to 2015. Every column of the dataset is in numerical format. A supporting document to understand each numerical category in accidents datset is provided on the www.data.gov.uk website as well. There is another dataset available to get general average traffic report for all the regions. 

# ## Importing Data and cleaning
# - We import three files to perform analysis on this data. This data is consist of three files that are accidents, casualities and vehicles. However, we have one more file which is general information about the traffic count for year 2000 to 2015. We can use general traffic information data for machine learning part.

# In[ ]:


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import TimeSeriesSplit
plt.style.use('ggplot')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


accidents = pd.read_csv('../input/dft-accident-data/Accidents0515.csv',index_col='Accident_Index')
casualties=pd.read_csv('../input/dft-accident-data/Casualties0515.csv' , error_bad_lines=False,index_col='Accident_Index',warn_bad_lines=False)
vehicles=pd.read_csv('../input/dft-accident-data/Vehicles0515.csv', error_bad_lines=False,index_col='Accident_Index',warn_bad_lines=False)
#general_info = pd.read_csv('ukTrafficAADF.csv')


# In[ ]:


accidents.head()


# In[ ]:


accidents = accidents.join(vehicles, how='outer')


# ## Identifying Missing Values
# 
# In this particular dataset, there are two types of missing values '-1' and 'Nan'. We will invesitigate each column with total missing values.
# We will not be imputing any mean or median value since the dataset is big enough to perform analysis.

# In[ ]:


accidents.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR','LSOA_of_Accident_Location',
                'Junction_Control' ,'2nd_Road_Class'], axis=1, inplace=True)
accidents['Date_time'] =  accidents['Date'] +' '+ accidents['Time']

for col in accidents.columns:
    accidents = (accidents[accidents[col]!=-1])
    #print(col ,' ' , x)
for col in casualties.columns:
    casualties = (casualties[casualties[col]!=-1])

    accidents['Date_time'] = pd.to_datetime(accidents.Date_time)
accidents.drop(['Date','Time'],axis =1 , inplace=True)
accidents.dropna(inplace=True)


# Our dataset is clean to do some analysis. We would be using very few columns to do analysis since the dataset is fairly large.

# # Data Visualization

# #### The first thing we can do is to find out about accidents time to get intution and some driver's age who are involved in the accident.
# - We can find out the number of accidents on the days of a week.
# - We can find out about the accidents number using hours of the day.
# - Finding out about the age of driver can tell us more about the accidents.

# In[ ]:


plt.figure(figsize=(12,6))
accidents.Date_time.dt.dayofweek.hist(bins=7,rwidth=0.55,alpha=0.5, color= 'orange')
plt.title('Accidents on the day of a week' , fontsize= 30)
plt.grid(False)
plt.ylabel('Accident count' , fontsize = 20)
plt.xlabel('0 - Sunday ,  1 - Monday  ,2 - Tuesday , 3 - Wednesday , 4 - Thursday , 5 - Friday , 6 - Saturday' , fontsize = 13)


# As we can see that thursday has the highest amount of accidents in this dataset from 2005 to 2015. We have to keep in mind that accidents numbers could be depending on traffic amount on particular day.

# In[ ]:


plt.figure(figsize=(12,6))
accidents.Date_time.dt.hour.hist(rwidth=0.75,alpha =0.50, color= 'orange')
plt.title('Time of the day/night',fontsize= 30)
plt.grid(False)
plt.xlabel('Time 0-23 hours' , fontsize = 20)
plt.ylabel('Accident count' , fontsize = 15)


# We found out that the most of accidents happened around after noon. We can assume that this time of the day has the most traffic moving such as people leaving from work.
# 
# 
# #### Age band of casualities
# 
# In this dataset, age band is grouped in 11 different codes. We will create the labels and pass it to the plot as xticks so we can have idea about the bins representation.

# In[ ]:


objects = ['0','0-5','6-10','11-15','16-20','21-25','26-35',
          '36-45', '46-55','56-65','66-75','75+']

plt.figure(figsize=(12,6))
casualties.Age_Band_of_Casualty.hist(bins = 11,alpha=0.5,rwidth=0.90, color= 'red',)
plt.title('Age of people involved in the accidents', fontsize = 25)
plt.grid(False)
y_pos = np.arange(len(objects))
plt.xticks(y_pos , objects)
plt.ylabel('Accident count' , fontsize = 15)
plt.xlabel('Age of Drivers', fontsize = 15,)


# This is very interesting fact about this dataset. Most of the drivers age is around 225 to 35 who are involved in the accident. However, we do not know the number of drivers with age 25 to 35 on the road compare to other ages. Intutively, I would assume that the driver with age 25 to 35 are more in the number of drivers with different age.

# In[ ]:


speed_zone_accidents = accidents.loc[accidents['Speed_limit'].isin(['20' ,'30' ,'40' ,'50' ,'60' ,'70'])]
speed  = speed_zone_accidents.Speed_limit.value_counts()

explode = (0.0, 0.0, 0.0 , 0.0 ,0.0,0.0) 
plt.figure(figsize=(10,8))
plt.pie(speed.values,  labels=None, 
        autopct='%.1f',pctdistance=0.8, labeldistance=1.9 ,explode = explode, shadow=False, startangle=160,textprops={'fontsize': 15})
 
plt.axis('equal')
plt.legend(speed.index, bbox_to_anchor=(1,0.7), loc="center right", fontsize=15, 
           bbox_transform=plt.gcf().transFigure)
plt.figtext(.5,.9,'Accidents percentage in Speed Zone', fontsize=25, ha='center')
plt.show()


# Most of the accidents occured on the road where the speed limit is 30. I was expecting more accidents on highway or major roadways. Some of the accidents could be cause of stop sign, changing lanes or turning into parking lot etc.

# ## Co-relation between variables
# 
# Since our dataset is in numeric values. We can findout correlation between columns.

# In[ ]:


corr =  accidents.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr)


# As we see that there is not so much strong correlations between any variables. I was expecting weather condition to be strong correlation with any of the variable. 
# - There is only one postiive strong correlation between speed limit and Urban or Rural Area. 

# In[ ]:


accidents_2014 = accidents[accidents.Date_time.dt.year ==2014]
accidents_2014_01 = accidents_2014[accidents_2014.Accident_Severity == 1]
accidents_2014_02 = accidents_2014[accidents_2014.Accident_Severity == 2]
accidents_2014_03 = accidents_2014[accidents_2014.Accident_Severity == 3]


# ## Plotting accidents Location on Google Maps
# 
# Now we will be using google maps to plot the accidents. Using longitude and latitude information, we can see what area has the most accidents. However, it actually depends on how much traffic the area has. We can also get the idea of busiest area even if we do not want to look at just accidents. The accident plots acan give us really good idea about traffic in any area of the UK.
# 
# Also, I have taken the screenshot of output plots so it can be seen when saved in html or pdf format.

# In[ ]:


#import gmaps
#from ipywidgets.embed import embed_minimal_html
#gmaps.configure(api_key='AIzaSyDFOjxJ23DfYRLTqEuNsgnqwP0E79Aybpk')

#fig = gmaps.figure(center=(53.0, 1.0), zoom_level=6)
#3heatmap_layer = gmaps.heatmap_layer(accidents_2014_01[["Latitude", "Longitude"]],
                                    #max_intensity=30,point_radius=5)
#heatmap_layer = gmaps.heatmap_layer(accidents_2014_02[["Latitude", "Longitude"]],
                                    #max_intensity=5,point_radius=3)
#heatmap_layer = gmaps.heatmap_layer(accidents_2014_03[["Latitude", "Longitude"]],
                                    #max_intensity=1,point_radius=1)
#fig.add_layer(heatmap_layer)
#fig
#embed_minimal_html('export1.html', views=[fig])


# In[ ]:


import matplotlib.image as mpimg
plt.figure(figsize=(18,8))
img=mpimg.imread('../input/photos/map1.png')
imgplot = plt.imshow(img)
plt.grid(False)
plt.show()


# In[ ]:


#import gmaps
#gmaps.configure(api_key="AIzaSyDFOjxJ23DfYRLTqEuNsgnqwP0E79Aybpk") 

#maps_df = accidents_2014_01[['Latitude', 'Longitude']]
#maps_layer = gmaps.symbol_layer(
   #maps_df, fill_color="green", stroke_color="red", scale=1
#)
#fig = gmaps.figure()
#fig.add_layer(maps_layer)
#fig


# In[ ]:


import matplotlib.image as mpimg
plt.figure(figsize=(18,8))
img=mpimg.imread('../input/photos/map2.png')
imgplot = plt.imshow(img)
plt.grid(False)
plt.show()


# As we can see that most of fatal accidents happened locally within cities instead on highways. It could be the reason of the traffic is more congested locally than on highways.

# ## Machine Learning
# 
# We will be looking at different columns to figure out predicting about the accidents severity. After we can predict the accident severity, we can make some recommendation to law enforcement for looking into this and be prepared for the future. We can also have more emergency medical services available for those situations.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import log_loss


# ## Normalize the Data
# There are few columns that we will standarize, so it would not effect negatively on our machine learning algorithms. Age of driver is from 18 to 88 in the dataset and we can normalize it. Also, the age of vehicle is also from 0 to 100 and it can skew the performance of your machine learning algorithm and we will normalize this predictor too.

# In[ ]:


sns.distplot(accidents['Age_of_Driver']);
fig = plt.figure()
sns.distplot(accidents['Age_of_Vehicle']);
fig = plt.figure()


# In[ ]:


accidents['Age_of_Driver'] = np.log(accidents['Age_of_Driver'])
accidents['Age_of_Vehicle'] = np.log(accidents['Age_of_Vehicle'])
sns.distplot(accidents['Age_of_Driver']);
fig = plt.figure()
sns.distplot(accidents['Age_of_Vehicle']);
fig = plt.figure()


# ## Spliting the data into training data and test data
# We will also consider few features as predictors for machine learning algorithm.

# In[ ]:


accident_ml = accidents.drop('Accident_Severity' ,axis=1)
accident_ml = accident_ml[['Did_Police_Officer_Attend_Scene_of_Accident' , 'Age_of_Driver' ,'Vehicle_Type', 'Age_of_Vehicle','Engine_Capacity_(CC)','Day_of_Week' , 'Weather_Conditions' , 'Road_Surface_Conditions'
                          , 'Light_Conditions', 'Sex_of_Driver' ,'Speed_limit']]

# Split the data into a training and test set.
X_train, X_test, y_train, y_test = train_test_split(accident_ml.values, 
                                              accidents['Accident_Severity'].values,test_size=0.20, random_state=99)


# ## Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train,y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_test, y_test)
acc_random_forest1 = round(random_forest.score(X_test, y_test) * 100, 2)

sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)
print("Accuracy" , acc_random_forest1)
print(sk_report)
pd.crosstab(y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# ## Logistic Regression 

# In[ ]:


lr = LogisticRegression()
# Fit the model on the trainng data.
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=y_pred)
print("Accuracy", round(accuracy_score(y_pred, y_test)*100,2))
print(sk_report)
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# ## Decision Tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree1 = round(decision_tree.score(X_test, y_test) * 100, 2)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)
print("Accuracy", acc_decision_tree1)
print(sk_report)
### Confusion Matrix 
pd.crosstab(y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# As we can see that Logistic regression did pretty well in terms of number. If we look carefully at the confusion matrix. We can definitely tell that Decision tree algorithm did much better. It predicted more fatal and serious injuries as true positive. The accuracy score is lower compare to another algorithm because other algorithm predicted majority of slightly accidents and those numbers are really high overall in the dataset. Confusion matrix helps us to understand what algorithm actually worked better in terms of looking at all different prediction of each class.

# # Hyperparameters tuning for the models
# 

# ### Logistic Regression with Hyperparameter tuning
# 

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV(cv=3, random_state=0,multi_class='multinomial')
# Fit the model on the trainng data.
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=y_pred)
print("Accuracy", round(accuracy_score(y_pred, y_test)*100,2))
print(sk_report)
pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# As we can see that Logistic regression still didn't predict two classes of accident severity out of 3. Even though it is showing the 86.2% accuracy. 

# ### Decision Tree hyperparameters tuning
# 
# All we are going to do is find the best values for mininum sample leaf and maximum features to get the best score.

# In[ ]:


decision_tree = DecisionTreeClassifier(min_samples_leaf=12, max_features=4)
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree1 = round(decision_tree.score(X_test, y_test) * 100, 2)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)
print("Accuracy", acc_decision_tree1)
print(sk_report)
### Confusion Matrix 
pd.crosstab(y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# We really didn't see much difference in Accident severity 1 and 2. However we did improve the accuracy of Accident severity 3. It jumped the accuracy from 75.1% to 85.8%.

# ###  Random Forest Hyperparameter tuning
# First, we will see the default parameters of the random forest model before we tune the parameters.

# In[ ]:


random_forest.get_params()


# We will implement the grid search using sklearn library. 

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [4, 5],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300]
}
# Create a based model
random_f = RandomForestClassifier()
# Instantiate the grid search model
grid_search = RandomizedSearchCV(estimator = random_f, param_distributions = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,y_train)


# ## Feature importance
# We can use Sklearn's random forest library to find out the most important features. We will be plotting in  ascending order so we know what features are most important to predict the accident severity.

# In[ ]:


plt.figure(figsize=(12,6))
feat_importances = pd.Series(random_forest.feature_importances_, index=accident_ml.columns)
feat_importances.nlargest(5).plot(kind='barh')


# In[ ]:


Y_pred = grid_search.predict(X_test)
acc_random_forest1 = round(grid_search.score(X_test, y_test) * 100, 2)

sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=Y_pred)
print("Accuracy" , acc_random_forest1)
print(sk_report)
pd.crosstab(y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# Random forest took lots of time to tune the hyperparameter. Most of the algorithm works well only with default values except decision tree.

# ## Conclusion
# As we have implemented the Logistic Regression, Decision Tree and Random Forest algorithms to predict the accident severity. There are two things that we can conclude from this learning.
# 
# #### Machine Learning Conclusion
# As we have tried three different algorithms to predict the accident severity. It was clear that Decision tree and Random Forest performed much better in terms of predicting all the classes of accident severity. Logistic regression has better accuracy but it does not mean it did better than other algorithm. We even tried multi-nomial to predict all the classes in hyperparameter tuning section. It still predicted only one of the higher occuring class.
# 
# #### Recommendation for Public or Law Enforcement
# There are two things that were clear from this project. First, the most of the accidents occured locally. Secondly, we used feature importance and vehicle engine and age of the driver are the biggest factor in accident severity. We already know that the car insurances are expensive for the young age people and this is one of the reasons.

# In[ ]:





# In[ ]:




