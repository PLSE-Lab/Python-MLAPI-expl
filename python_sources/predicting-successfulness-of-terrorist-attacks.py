#!/usr/bin/env python
# coding: utf-8

# #  Predicting Successfulness of Terrorism in Europe 
# ## using Decision Trees and Random Forests

# The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the world from 1970 through 2016. It includes systematic data on terrorist incidents that have occurred during this time period and now includes more than 170,000 cases = rows of data. 
# 
# This analysis focuses on terrorist attacks in Europe during this 46-year period, which includes over 21.000 incidents = rows of data. Each incident in the GTD is described with almost 50 different variables. The data includes both numerical and categorical data. <br><br>
# One of the binomial variables is whether an attack has been succedful or not. The GTD Codebook  https://www.start.umd.edu/gtd/downloads/Codebook.pdf  defines this variable as follows:<br><br>
# *Success of a terrorist strike is defined according to the tangible effects of the attack. Success is not judged in terms of the larger goals of the perpetrators. For example, a bomb that exploded in a building would be counted as a success even if it did not succeed in bringing the building down or inducing government repression. The definition of a successful attack depends on the type of attack. Essentially, the key question is whether or not the attack type took place. If a case has multiple
# attack types, it is successful if any of the attack types are successful, with the exception of assassinations, which are only successful if the intended target is killed.<br>
# 1 = "Yes" The incident was successful.<br>
# 0 = "No" The incident was not successful.*<br><br>
# 
# This analysis aims to find a model, using the available variables, for predicting the succefulness of terrorist attacks in Europe. The methods used are Decision Trees and Random Forests.<br><br>
# The steps are the following
# - Downloading and shaping the data and viewing it from various angles
# - Creating a Train and Test set
# - Presenting the Performance Measures for the models
# - Creating two Decision Trees models; a simple and a more complex one
# - Creating a Random Forest model
# - Checking for Feature Importance 
# - Implementing the model
# - Conclusions

# In[ ]:


import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import seaborn as sns #remove?
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
#import pydotplus
from sklearn.ensemble import RandomForestClassifier


# ### Downloading, cleaning and shaping
# I start by downloading the data and choosing the variables. There are many reasons why only about half of the available variables are chosen. For example, <br>
# - The variables may have been strings, like name of the city or the perpetrator group name (70+ entries) making them difficult to categorize.<br>
# - There were too many missing values for a variable and replacing them artificially might have distorted the results, so they were left out. For example, *'number of perpetrators'*; too many were unknown.
# 

# In[ ]:


input_file = "../input/globalterrorismdb_shorter.csv"

df = pd.read_csv(input_file, header = 0,usecols=['iyear', 'imonth', 'iday', 'extended', 'country', 'country_txt', 'region', 'latitude', 'longitude','success', 'suicide','attacktype1','attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1','natlty1_txt','weaptype1', 'weaptype1_txt' ,'nkill','multiple', 'individual', 'claimed','nkill','nkillter', 'nwound', 'nwoundte'])
df.head(5)
df.info()


# #### Some of the variables explained:
# - **extended**: The duration of an incident extended more than 24 hours (1) - Less than 24 hours (0).
# - **multiple**: The attack is part of a multiple incident (1) - Not part of a multiple incident (0).
# - **suicide**: The incident was a suicide attack (1) - There is no indication of a suicide attack (0).
# - **attacktype**: Assassination(1), Hijacking(2), Kidnapping(3), Barricade Incident(4), Bombing/Explosion(5), Armed Assault(6), Unarmed Assault(7), Facility/Infrastructure Attack(8), Unknown(9)
# - **targtype**: 22 categories ranging from Business(1), Government(general)(2), Police(3),...Utilities(21), Violent Political Parties(22)
# - **natlty**: Nationality of target/victim
# - **individual**: Whether the attack was carried out by an individual or several individuals known to be affiliated with a group or organization(1) or not affiliated with a group or organization(0)
# - **claimed**: A group or person claimed responsibility for the attack (1) - No claim of responsibility was made(0).
# - **weaptype**: 13 categories ranging from Biological(1), Chemical(2), Radiological(3),...Other(12), Unknown(13)
# - **nkill**: Total number of fatalities including  all victims and attackers who died as a direct result of the incident.
# - **nkillter**:  Limited to only perpetrator fatalities
# - **nwound**:  Total number of injured victims and attackers
# - **nwoundte**: Total number of injured perpetrators <br><br>
#  Next, I narrow the data down to incidents occured only in Europe and check how the data looks like:<br>

# In[ ]:


df_WEur= df[df.region == 8] # A dataframe with region Western Europe
df_EEur= df[df.region ==9] # A dataframe with region Eastern Europe

euro_frames = [df_WEur, df_EEur]
df_Euro = pd.concat(euro_frames) # # A dataframe with whole Europe, both Western & Eastern  
df_Euro.info()


# I get rid of some of the variables: <br>
# - **claimed, nkillter, nwound, nwoundte** - too many of the values are missing and replacing them artificially would be misleading.<br><br>
# Also, we don't need the variable **region** anymore since all are within Europe.
# 

# In[ ]:


df_Europe = df_Euro.drop([ 'region', 'claimed', 'nkillter', 'nwound','nwoundte'], axis=1)  
df_Europe.head()


# I'm now down to mostly integers and floats.

# In[ ]:


df_Europe.info()


# Next, let's look at a summary of the numerical variables.

# In[ ]:


df_Europe.describe() 


# ### The anatomy of an attack - a statistical average
# Finally, having the European data in one data frame we can take a closer look at it. For example, what have, then,  been the characteristics of an *average attack* in Europe between 1970-2016?
# 

# 
# ![](https://github.com/LJANGN/Predicting-terrorism-in-Europe-through-Decision-Trees-and-Random-Forests/blob/master/AverageLocation.JPG?raw=true)

# > It's apparent that historical averages do not give a very realistic prediction, especially for the location of an attack!

# From the rough geographical sketch plotting all the incidents, we can see that Western Europe as the scene dominates clearly.

# In[ ]:


df_Europe.plot(kind= 'scatter', x='longitude', y='latitude', alpha=0.4, figsize=(16,7))
plt.show()


# As for the number of casualties over the years the Balkans and Ukraine have been the most violent areas, closely followed by Northern Ireland.

# In[ ]:


df_Europe.plot(kind= 'scatter', x='longitude', y='latitude', alpha=1.0,  figsize=(18,6),  
               s=df_Europe['nkill']*3, label= 'Nr of casualties', fontsize=1, c='nkill', cmap=plt.get_cmap("jet"), colorbar=True)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()


# There were still NaN's in some of the variables. I use the means of the available values to replace them :<br>
# - **nkill** = 	0.686445<br>
# - **latitude** = 47.004651
# - **longitude** = 10.921231
# - **natlty1** = 167.954530

# In[ ]:


df_Europe['nkill'].fillna(0.686445, inplace=True)
df_Europe['latitude'].fillna(47.004651, inplace=True)
df_Europe['longitude'].fillna(10.921231, inplace=True)
df_Europe['natlty1'].fillna(167.954530, inplace=True)

df_Europe.info()


# I now have a complete set of data. There are still 25 entries missing from nationalities (**natlty1_text**) but since it's a string variable and will not be used in calculations I decided to ignore this. <br><br>
# Next, we can start visualizing the data.  <br><br>
# From the summary of the numerical attributes it can be seen that the mean for **success** is 0.856; i.e. the terror attacks in Europe have on average - unfortunately - been succesful in c. 17 cases out of 20. This high success rate is also visible in the charts.

# In[ ]:


terror_peryear = np.asarray(df_Europe.groupby('iyear').iyear.count())
successes_peryear = np.asarray(df_Europe.groupby('iyear').success.sum())

terror_years = np.arange(1970, 2016)

trace1 = go.Bar(x = terror_years, y = terror_peryear, name = 'Nr of terrorist attacks',
         width = dict(color = 'rgb(118,238,198)', width = 3))

trace2 = go.Scatter(x = terror_years, y = successes_peryear, name = 'Nr of succesful terrorist attacks',
         line = dict(color = ('rgb(205, 12, 24)'), width = 5,))

layout = go.Layout(title = 'Terrorist Attacks by Year in Europe (1970-2016)', legend=dict(orientation="h"),
         barmode = 'group')

figure = dict(data = [trace1,trace2], layout = layout)
iplot(figure)
#Image("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/Global Terrorism Database/Charts/AttacksByYear.png")


# The most active terrorist groups in the peak years have been
# - **1979:** **France:** Corsican National Liberation Front (FLNC), **Spain:** Basque Fatherland and Freedom (ETA), **U.K.:** Irish Republican Army (IRA)
# - **1992:** The three above plus **Germany:** Neo-Nazi extremists
# - **2013-14:** Mainly unknown perpetrators in Russia and eastern Ukraine <br><br>
# The following bar charts show the number of attacks per country in total and the share of those that were succesful.<br> **Please note!** The Top 10 is plotted in a separate graph due to their overweigth in the overall statistics. This way the "below Top 10" chart becomes more readable.

# In[ ]:


attacks_per_country = (df_Europe.groupby('country_txt').country_txt.count()) 
successes_per_country = (df_Europe.groupby('country_txt').success.sum()) 

trace1 = go.Bar(y=['Ireland','West Germany (FRG)','Germany','Greece','Italy','Ukraine','Russia','France','Spain','United Kingdom'],
    x=[290,541,703,1231,1556,1650,2158,2642,3245,5098],
    name='Nr of terrorist attacks per country', orientation = 'h', 
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(color = 'rgba(0, 255,0,0, 0.)', width = 0.5)))

trace2 = go.Bar(y=['Ireland','West Germany (FRG)','Germany','Greece','Italy','Ukraine','Russia','France','Spain','United Kingdom'],
    x=[135,465,633,1092,1384,1479,1780,2441,2814,4107],
    name='Nr of succesful terrorist attacks per country', orientation = 'h',
    marker = dict(color = 'rgba(128,128,0, 0.4)', line = dict(color = 'rgba(246, 78, 139, 0.2)',width = 0.5)))

data = [trace2,trace1]
layout = go.Layout(
    legend=dict(x=0.5, y=0.5), # placing legend in the middle
    title = 'Terrorist attacks in Europe 1970-2016 <br>by Country  - TOP 10', barmode='group',
    bargap=0.1, 
    autosize=False,
    width=1000,
    height=1000)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Image("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/Global Terrorism Database/Charts/Top10.png")


# In[ ]:


trace1 = go.Bar(
    y=['Andorra','Vatican City','Iceland','Montenegro','Romania','Slovenia','Lithuania','Czechoslovakia',
       'Serbia','Serbia-Montenegro','Belarus','Estonia','Latvia','Luxembourg','Finland','Norway','Slovak Republic',
       'Malta','Moldova','Czech Republic','Poland','East Germany (GDR)','Denmark','Hungary','Bulgaria','Croatia',
       'Soviet Union','Albania','Switzerland','Austria','Macedonia','Sweden','Netherlands','Cyprus','Portugal',
       'Belgium','Bosnia-Herzegovina','Kosovo','Yugoslavia'],
    x=[1,1,4,5,6,6,8,10,11,11,13,16,16,16,18,18,18,20,21,29,36,38,41,46,52,57,78,79,108,109,117,118,128,132,139,148,159,188,203],
    name='Nr of terrorist attacks per country',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255,0,0, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 0.2)',
            width = 1)))

trace2 = go.Bar(
    y=['Andorra','Vatican City','Iceland','Montenegro','Romania','Slovenia','Lithuania','Czechoslovakia',
       'Serbia','Serbia-Montenegro','Belarus','Estonia','Latvia','Luxembourg','Finland','Norway','Slovak Republic',
       'Malta','Moldova','Czech Republic','Poland','East Germany (GDR)','Denmark','Hungary','Bulgaria','Croatia',
       'Soviet Union','Albania','Switzerland','Austria','Macedonia','Sweden','Netherlands','Cyprus','Portugal',
       'Belgium','Bosnia-Herzegovina','Kosovo','Yugoslavia'],
    x=[1,0,4,5,4,6,7,7,10,10,13,16,12,14,17,16,15,19,18,20,31,35,35,40,46,55,67,63,90,87,106,104,106,112,129,123,151,165,179],
    name='Nr of succesful terrorist attacks per country',
    orientation = 'h',
    marker = dict(
        color = 'rgba(128,128,0, 0.4)',
        line = dict(
            color = 'rgba(246, 78, 139, 0.2)',
            width = 0.5)))

data = [trace2, trace1]
layout = go.Layout(
    legend=dict(x=0.5, y=0.5), # placing legend in the middle
    title = 'Terrorist attacks in Europe 1970-2016 <br>by Country outside the Top 10',
    barmode='group',
    bargap=0.1,
    #bargroupgap=0.1,
    autosize=False,
    width=900,
    height=1500,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Image("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/Global Terrorism Database/Charts/BelowTop10.png")


# ### Succesful attacks arranged by attack type
# Next we look at the attacks by type seeing that Bombing/Explosion has by far been the dominating type. 

# In[ ]:


attacks_per_type = (df_Europe.groupby('attacktype1_txt').attacktype1_txt.count())
successes_per_type = (df_Europe.groupby('attacktype1_txt').success.sum())

trace2 = go.Bar(
    y=['Hijacking','Hostage Taking (Barricade Incident)','Unarmed Assault','Unknown','Hostage Taking (Kidnapping)','Facility/Infrastructure Attack','Armed Assault','Assassination','Bombing/Explosion'],
    x=[89,106,183,373,485,2752,2911,3295,11144],
    name='Nr of terrorist attacks',
    orientation = 'h',
    marker = dict(
        color = 'rgba(128,0,0, 0.8)',
        line = dict(
            color = 'rgba(246, 78, 139, 0.2)',
            width = 1)))

trace1 = go.Bar(
    y=['Hijacking','Hostage Taking (Barricade Incident)','Unarmed Assault','Unknown','Hostage Taking (Kidnapping)','Facility/Infrastructure Attack','Armed Assault','Assassination','Bombing/Explosion'],
    x=[76,105,149,333,457,2530,2637,2588,9393],
    name='Nr of succesful terrorist attacks',
    orientation = 'h',
    marker = dict(
        color = 'rgba(128,128,0, 0.4)',
        line = dict(
            color = 'rgba(246, 78, 139, 0.2)',
            width = 0.5)))

data = [trace1, trace2]
layout = go.Layout(
    legend=dict(x=0.5, y=0.5), # placing legend in the middle
    title = 'Terrorist attacks in Europe 1970-2016 <br>by Type',
    barmode='group',
    bargap=0.1,
    bargroupgap=0,
    autosize=False,
    width=1000,
    height=500,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Image("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/Global Terrorism Database/Charts/AttacksByType.png")


# Next a look at the variables' correlations, numerically and graphically.

# In[ ]:


df_Europe.corr()


# In[ ]:


corrmat = df_Europe.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1, square=True);
plt.show()


# There are some obvious higher correlations, like between **nationality** & **country** and **attacktype** & **weapontype**. Otherwise variables are fairly non-correlated, which looks good for the model; we don't have closely related variables basically just capturing the same thing.<br><br>
# Going forward, I will further drop string objects from the data with the extension '_txt'. These variables will still be found as *numerical* values in the data, though. 

# ### Train Test split 
# Now I need to split our data to a train set (80%) and test set (20%).  The variable we're trying to predict is **success**; what determines whether a terrorist attack will be succesful or not. <br>
# The **random_state** -variable of the split is set to a fixed number, here the somewhat arbitrary '42', thereby keeping the random number generator constant. This way I will always be getting the same split and avoid the risk of introducing sampling bias.

# In[ ]:


#from sklearn.model_selection import train_test_split
X = df_Europe.drop(['iyear', 'success','country', 'country_txt', 'attacktype1_txt','targtype1_txt','natlty1', 'natlty1_txt', 'weaptype1_txt'], axis=1)
y = df_Europe['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Next, I need to separate the *features* from the *target* that I'm trying to build a Decision Tree for (and later a Random Forest).<br>
# I skip variable **'iyear'**, since past years cannot reoccur and have therefore no impact on the prediction. Also, I skip variable **'country'**, since it must fit together with the variables **'longitude'** and **'latitude'** anyway and does not bring additional value on its own.
# 

# In[ ]:


features = [ 'imonth', 'iday', 'extended',  'latitude', 'longitude', 'multiple','suicide','attacktype1',
            'targtype1', 'individual', 'weaptype1', 'nkill']


# ### Performance Measures for Prediction and Evaluation
# I use a Confusion Matrix first on a Decision Tree then on a Random Forest to evaluate the accuracy of each method.<br>
# 
# |       |                 |
# |-----------------------|:--------------------:|
# | True Negatives (TN)  |  False Positives (FP) |
# | False Negatives (FN)  |  True Positives (TP) |
# <br>
# In a perfect model both **False Positives** and **False Negatives** in the matrix would be zero!<br><br>
# In our model for predicting the succesfulness of terrorist attacks the interpretation is as follows:<br><br>
# - TN = The model predicts correctly a non-succesful attack
# - TP = The model predicts correctly a succesful attack
# - FN = The model predicts a succesful attack wrongfully to be non-succesful
# - FP = The model predicts a non-succesful attack wrongfully to be succesful<br><br>
# 
# The **Precision**(=accuracy of the positive predictions), **Recall**(=ratio of positive instances correctly detected by the classifier) and **f1-score** may be more concise metrics, however.
# 
# - **Precision** for 'success' = TP/(TP+FP) <br>
# - **Precision** for *not* 'success' = TN/(TN+FN) <br><br>
# - **Recall** for 'success' = TP/(TP+FN)  <br>
# - **Recall** for *not* 'success' = TN/(TN+FP)  <br><br>
# - The **f1-score** is the harmonic mean of Precision and Recall.
# 

# ### Decision Tree - with max node depth of 3

# Now I will construct the actual Decision Tree. Just to make the first try more visual I stick to a max depth of 3 for the decision nodes.

# In[ ]:


y = df_Europe['success'] #this is what we're trying to predict!
X = df_Europe[features]
dtc = tree.DecisionTreeClassifier(max_depth=3) 
dtc = dtc.fit(X_train,y_train)
#Two lines of code to create the classifier!!


# dot_data = StringIO()  
# tree.export_graphviz(dtc, out_file=dot_data,  
#                          feature_names=features)  
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())  

# ### Please note! *Pydotplus* is not working in Kaggle so the tree is presented here as an image from the original Jupyter notebook

# ![](https://github.com/LJANGN/Predicting-terrorism-in-Europe-through-Decision-Trees-and-Random-Forests/blob/master/PydotplusDecisionTree.JPG?raw=true)

# And a look at the results:

# In[ ]:


dtc_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,dtc_pred))


# In[ ]:


print(confusion_matrix(y_test,dtc_pred))


# ###  Decision Tree - no max node depth

# In[ ]:


y = df_Europe['success'] 
X = df_Europe[features]
dtc = tree.DecisionTreeClassifier() 
dtc = dtc.fit(X_train,y_train)


# The graphical Decision Tree would be too complex to be presented visually so I go with numerical results only.

# In[ ]:


dtc_pred = dtc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,dtc_pred))


# In[ ]:


print(confusion_matrix(y_test,dtc_pred))


# **True Negatives** have more than doubled, **False Positives** have decreased by a third and **False Negatives** are much higher.<br>
# Since Decision Trees can suffer from overfitting, I will continue with Random Forests to see if I can improve the model.

# ### Random Forest
# After trying with various *n_estimators*, i.e. "number of trees in the forest", with 10 being the default, the optimal **confusion_matrix** was received with (n_estimators=400). Increasing the number to up to 1000 did not improve the outcome, but slightly to the contrary. 

# In[ ]:


rfc = RandomForestClassifier(n_estimators=400) 
rfc = rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)


# In[ ]:


#rfc_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


print(confusion_matrix(y_test,rfc_pred))


# Overall, comparing the Confusion Matrix of the latter Decision Tree with the Random Forest, the difference is not very striking, except for the clear reduction in False Negatives. The Classification Report does show a clearer improvement, though. It's clear we go with the Random Forest model!

# ### Checking for feature importance
# I can also check for the relative importance of each attribute for making accurate predictions. With this information, I could drop some of the less useful features, should I decide to fine-tune the model further.

# In[ ]:


for name, score in zip(X_train[features], rfc.feature_importances_):
    print(name, score)


# I sort the results and take a look at them graphically:

# In[ ]:


data = go.Bar(
    y=['suicide', 'individual', 'extended', 'multiple', 'weaptype1', 'nkill', 'attacktype1','targtype1', 
       'imonth',  'iday', 'latitude',  'longitude'],
    x=[0.001182,0.002047,0.002392,0.011422,0.041777,0.085776,0.107362,0.109867,0.113829,0.167432,0.173583,0.18333],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Relative Importance of the Features in the Random Forest',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Image("C:/Users/jangn/CODE/Sundog_DataScience/DataScience/DataScience-Python3/data_sets/Global Terrorism Database/Charts/FeatureImportance.png")


# Looks like dropping the features **extended**, **individual** and **suicide** might be considered. 

# ### Checking the outcome with inserted data
# Now we finally have a model using which we can actually predict whether an attack is expected to succeed or fail.<br>
# By typing in the twelve variables - here rather randomly chosen -  the model gives the predicted outcome.

# In[ ]:


succeed_or_fail = RandomForestClassifier(n_estimators=400) 
succeed_or_fail = rfc.fit(X, y) #clf

month = 12           # in which month would the attack take place
day = 23             # on which day of the month would the attack take place
extended = 0         # 1=yes, 0=no
latitude = 48.8566
longitude = 2.3522
multiple = 0         # attack is part of a multiple incident (1), or not (0)
suicide = 0          # suicide attack (1) or not (0)
attackType = 3       # 9 categories
targetType = 7       # 22 categories
individual = 0       # known group/organization (1) or not (0)
weaponType = 6       # 13 categories
nkill = 0            # number of total casualties from the attack

outcome = (succeed_or_fail.predict([[(month),(day),(extended),(latitude),(longitude),(multiple),(suicide),(attackType),(targetType),(individual),(weaponType),(nkill)]])) 
if outcome == 1:
    print(outcome)
    print("The attack based on these features would be succesful.")
elif outcome == 0:
    print(outcome)
    print("The attack based on these features would NOT be succesful.")


# ### Conclusions

# I decided to narrow the data down to cover only attacks on European soil, i.e. 21.000+ incidents during 1970-2016. Maybe a more accurate model would have been achieved by using the global data, 170.000+ incidents. However, by using global data I might loose some of the model's predicting power for Europe due to the large variance in the data, especially for longitude and latitude.<br><br>
# Perhaps a more interesting analysis would be whether an attack is expected to lead to civilian casualties. The data for this variable, *total number of fatalities including all victims and attackers*, had unfortunately quite many gaps in it due to the uncertain number of terrrorists killed.<br><br>
# The model could be fine-tuned - and simplified - further by skipping the 4 least important features; **multiple, extended, individual, suicide**. However, for now, I conclude the analysis here.
