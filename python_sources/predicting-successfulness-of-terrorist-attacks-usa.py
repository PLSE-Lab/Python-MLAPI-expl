#!/usr/bin/env python
# coding: utf-8

# **Predicting whether a terrorist attack in the USA would be successful**
# 
# Recently, I stumbled upon a kernel by Jan Nordin that aimed to predict the successfulness of terrorist attacks in Europe. I found the kernel fascinating and so aim to replicate the procedure with a different region (North America) and a different y value. I will also try to extend it, for example by predicting the kill count of terrorist attacks as well as whether they are successul. As this is the first time I have ever used Pandas and ML it will largely be a copy of his work so I encourage you to look at his original version - it also has a very good version of some of the variables. 
# 
# The Global Terrorism Database (GTD) is an open-source database including information on terrorist attacks around the world from 1970 through 2016. It includes systematic data on terrorist incidents that have occurred during this time period and now includes more than 170,000 cases (records).
# 
# This analysis focuses on terrorist attacks in the USA during this 46-year period, which includes 2758 incidents (records). Each incident in the GTD is described with almost 58 variables. 

# In[ ]:


import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
#import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.tree import DecisionTreeRegressor as dtr


# In[ ]:


input_file = "../input/globalterrorismdb_shorter.csv"
df = pd.read_csv(input_file, header = 0,usecols=['iyear', 'imonth', 'iday', 'extended', 'country', 'country_txt', 'region', 'latitude', 'longitude','success', 'suicide','attacktype1','attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1','natlty1_txt','weaptype1', 'weaptype1_txt' ,'nkill','multiple', 'individual', 'claimed','nkill','nkillter', 'nwound', 'nwoundte'])
df.head(10)


# We can see from the head of the data frame that the nkill, nkillter, nwound, nwoundte variables have many NaN values which will have to be removed or replaced.

# In[ ]:


df.info()


# Next, we narrow down the data frame to only hold the terrorist attacks in the USA. 

# In[ ]:


df_USA = df[df.country == 217]


# In[ ]:


df_USA.head()


# In[ ]:


df_USA.info()


# We can see from the df_USA.info() command that there are too many missing values in nkillter, nwoundte and claimed so they should be removed. In addition, there is no need for the region, country, or country_txt fields anymore as all the terrorist attacks were in the USA. Though there are null values for many of the fields as well, particularly nkill and nwound, these are not as significant.

# In[ ]:


df_USA = df_USA.drop([ 'region', 'country', 'country_txt','claimed', 'nkillter', 'nwoundte'],axis=1)


# In[ ]:


df_USA.tail()


# In[ ]:


df_USA.describe()


# **Statistical average of Attacks in USA**
# 1. The average time for a terrorist attack was mid June 
# 2. The average terrorist attack was a successful, isolated hijacking against aircraft/airports carried out by someone not affiliated with a terrorist group. 
# 3. The average terrorist attack was less than 24 hours, involved fake weapons but still managed to kill 1.35 people and wound on average just less than 7. 
# 4. The average terrorist attack in the USA took place in West Plains, Missouri. 
# 5. The average intended victim was from Tuvalu.
# 
# Naturally, please take these averages with a pinch of salt. They seem to get progressively more outlandish from top to bottom.
# 
# **MAPPING**

# In[ ]:


df_USA.plot(kind= 'scatter', x='longitude', y='latitude', alpha=0.4, figsize=(16,7))
plt.show()


# **REMOVING ERRORS**
# 
# Quite how it is possible that this is displaying terrorist attacks in Mongolia I do not know. I will now investigate this by sorting the dataset by latitude. Also to investigate other outliers that appear to be in areas like Hawaii and Alaska and Puerto Rico.

# In[ ]:


df_USA.nsmallest(6,"longitude")


# The first four sets of coordinates are all in Hawaii. The fifth is Alaska and the sixth is already mainland USA. This has confirmed my suspiscions and the records do not need to be removed. I will now check the outliers further East.

# In[ ]:


df_USA.nlargest(6,"longitude")


# So the first three are records are of attacks that have taken place in Inner Mongolia, China and the fourth is from Tibet. 
# Puerto Rico is the lcoation for the next two which will remain in the data frame.
# I do not know why those four attacks had an incorrect nation label. 

# In[ ]:


df_USA = df_USA[df_USA.longitude < 0]


# In[ ]:


df_USA.nlargest(6,"longitude")


# For some reason the code "df_USA = df_USA.drop(["2179","15275","33294","69529"], axis = 0)" would not work so I used the code above to remove those errors in a different way: making the data frame equal to the all the records in the data frame as long as the longitude field in that record has a value less than 0. The value of 0 was completely arbitrary but I used it as the table above shows us that a value between -65 and 90 had to be used. 

# In[ ]:


df_USA.plot(kind= 'scatter', x='longitude', y='latitude', alpha=0.4, figsize=(16,9))
plt.show()


# In[ ]:


df_USA.plot(kind= 'scatter', x='longitude', y='latitude', alpha=1.0,  figsize=(18,6),  
               s=df_USA['nkill']*3, label= 'Nr of casualties', fontsize=1, c='nkill', cmap=plt.get_cmap("jet"), colorbar=True)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()


# In[ ]:


terror_peryear = np.asarray(df_USA.groupby('iyear').iyear.count())
successes_peryear = np.asarray(df_USA.groupby('iyear').success.sum())

terror_years = np.arange(1970, 2016)

trace1 = go.Bar(x = terror_years, y = terror_peryear, name = 'Nr of terrorist attacks')

trace2 = go.Scatter(x = terror_years, y = successes_peryear, name = 'Nr of succesful terrorist attacks', line = dict(color = ('rgb(205, 12, 24)'),width=5))

layout = go.Layout(title = 'Terrorist Attacks by Year in USA (1970-2016)', legend=dict(orientation="h"),
         barmode = 'group')

figure = dict(data = [trace1,trace2], layout = layout)
iplot(figure)


# In[ ]:


attacks_per_type = (df_USA.groupby('attacktype1_txt').attacktype1_txt.count())
successes_per_type = (df_USA.groupby('attacktype1_txt').success.sum())
print(attacks_per_type, successes_per_type)


# In[ ]:


trace2 = go.Bar(
    y=['Unknown','Hijacking','Hostage Taking (Kidnapping)','Unarmed Assault','Hostage Taking (Barricade Incident)','Assassination','Armed Assault','Facility/Infrastructure Attack','Bombing/Explosion'],
    x=[11,17,20,56,59,128,249,836,1377],
    name='Nr of terrorist attacks',
    orientation = 'h',
    marker = dict(color = 'rgb(255,140,0)'))

trace1 = go.Bar(
    y=['Unknown','Hijacking','Hostage Taking (Kidnapping)','Unarmed Assault','Hostage Taking (Barricade Incident)','Assassination','Armed Assault','Facility/Infrastructure Attack','Bombing/Explosion'],
    x=[8,15,20,31,56,80,233,748,1080],
    name='Nr of successful terrorist attacks',
    orientation = 'h',
    marker = dict(color = 'rgb(0,200,200)'))
data = [trace1, trace2]
layout = go.Layout(
    legend=dict(x=0.5, y=0.5), # placing legend in the middle
    title = 'Terrorist attacks in USA 1970-2016 <br>by Type',
    barmode='group',
    bargap=0.1,
    bargroupgap=0,
    autosize=False,
    width=1000,
    height=500,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


terror_peryear = np.asarray(df_USA.groupby('iyear').iyear.count())
affiliated_attacks_peryear = np.asarray(df_USA.groupby('iyear').individual.sum())
percentage = affiliated_attacks_peryear / terror_peryear * 100

terror_years = np.arange(1970, 2016)

trace1 = go.Bar(x = terror_years, y = terror_peryear,name = 'Terrorist attacks')

trace2 = go.Scatter(x = terror_years, y = affiliated_attacks_peryear,name = 'Terrorist attacks by people affiliated with terrorist organisations',yaxis = "y2")

trace3 = go.Scatter(x = terror_years, y = percentage, name = "Percentage of terrorist attacks carried out by people affiliated with terrorist organisations", yaxis  = "y3")

data = [trace1,trace2,trace3]

layout = go.Layout(
    title='Rise of Terrorist groups in the USA',
    yaxis1=dict(title='Terrorist attacks',showline = False,showgrid=False),
    yaxis2=dict(
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        showgrid=False,
        zeroline= False,
        showline=False,
        ticks="",
        showticklabels=False,
        overlaying='y',
        autorange = True,
        side='right'),
    yaxis3 = dict(
        titlefont = dict(
            color = "rgb(124,252,0)"
        ),
        showgrid=False,
        zeroline= False,
        showline=False,
        ticks = "",
        showticklabels=False,
        overlaying = "y",
        autorange = True,
        side = "right"
    ),
    legend=dict(orientation="h")
)

figure = go.Figure(data = data, layout = layout)
iplot(figure)


# **Does anyone know why this displays the values of 0 for the percentage value at well above the x axis?**

# In[ ]:


df_USA.info()


# There are still NaN values for latitude and longitude, natlty1, natlty1_txt, nkill and nwound. 
# For lat, long, nkill, nwound I will fill in the averages as I have no clue what the actual values are. 
# For natlty1 and natlty1_txt , I will fill in the corresponding values of USA as the vast majority of victims were american.

# In[ ]:


df_USA.describe()


# In[ ]:


df_USA['nkill'].fillna(1.361194, inplace=True)
df_USA['nwound'].fillna(6.802632, inplace=True)
df_USA['latitude'].fillna(36.683652, inplace=True)
df_USA['longitude'].fillna(-92.125972, inplace=True)
df_USA["natlty1"].fillna(217, inplace=True)
df_USA["natlty1_txt"].fillna("United States", inplace=True)
df_USA.info()


# Now we have absolutely no missing pieces of data. 
# 
# **Correlations**

# In[ ]:


df_USA.corr()


# In[ ]:


corrmat = df_USA.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1, square=True);
plt.show()


# There are some obviuous correlations such as the attacktype1 variable correlating well with the weapon type variable. Another common sense correlation is that between the nkill and nwound variables. 
# 
# Most other correlations are minor. Interestingly, there is a corrrelation between whether the individual(s) carrying out the attack were affiliated with a group. That could be due to the rise of terrorist organisations from further afield such as Al-Qaeda and ISIS, or the decline of internal groups such as the KKK or black power. I am not sure which is which. However, it is interesting to note that being affiliated to a terror organisation does not correlate with killing more people. A thing that does correlate well with killing more people on the other hand is whether or not the attack was a suicide attack. 
# 
# **Train Test split**
# 
# Now I need to split our data to a train set (80%) and test set (20%). The variable we're trying to predict is nkill.
# 
# The random_state variable will be set the value of 1 (arbitrary but must be kept constant). 
# 
# The fields ending in 'txt' will be dropped as there are numerical equivalents in the data frame.

# In[ ]:


df_USA = df_USA.drop(["iyear","attacktype1_txt","targtype1_txt", "weaptype1_txt", "natlty1_txt"], axis=1)


# In[ ]:


y = df_USA["success"]
features_success = ['imonth', 'iday', 'extended', 'latitude', 'longitude', 'multiple', 'suicide', 'attacktype1', 'targtype1', 'natlty1','individual', 'weaptype1', 'nkill', 'nwound']
X = df_USA[features_success]


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)


# **Creating the classifier to predict successfulness**
# 
# As this is my introduction to using classifiers, I wanted to try out different possibilities. Therefore, I will create four different classifiers and compare the precision of each classifier by looking at values for FPs, FNs, TPs, TNs.
# 
# > TN = The model predicts correctly a non-succesful attack
# TP = The model predicts correctly a succesful attack
# FN = The model predicts a succesful attack wrongfully to be non-succesful
# FP = The model predicts a non-succesful attack wrongfully to be succesful
# 
# > The Precision(=accuracy of the positive predictions), Recall(=ratio of positive instances correctly detected by the classifier) and f1-score may be more concise metrics, however.
# 
# > Precision for 'success' = TP/(TP+FP) 
# Precision for not 'success' = TN/(TN+FN) 
# 
# > Recall for 'success' = TP/(TP+FN) 
# Recall for not 'success' = TN/(TN+FP) 
# 
# > The f1-score is the harmonic mean of Precision and Recall.
# 
# **1. Using both max_depth and max_leaf_nodes**

# In[ ]:


terrorism_success_model_depth_leaves = tree.DecisionTreeClassifier(random_state = 1, max_depth = 3, max_leaf_nodes = 10)
terrorism_success_model_depth_leaves.fit(X_train, y_train)


# In[ ]:


success_pred_depth_leaves = terrorism_success_model_depth_leaves.predict(X_val)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_val,success_pred_depth_leaves))


# In[ ]:


print(confusion_matrix(y_val,success_pred_depth_leaves))


# **2. Using just max_depth**

# In[ ]:


terrorism_success_model_depth = tree.DecisionTreeClassifier(random_state = 1, max_depth = 3)
terrorism_success_model_depth.fit(X_train, y_train)
success_pred_depth = terrorism_success_model_depth.predict(X_val)
print(classification_report(y_val,success_pred_depth))


# In[ ]:


print(confusion_matrix(y_val,success_pred_depth))


# **3. Using just max_leaf_nodes**

# In[ ]:


terrorism_success_model_leaves = tree.DecisionTreeClassifier(random_state = 1, max_leaf_nodes = 10)
terrorism_success_model_leaves.fit(X_train, y_train)
success_pred_leaves = terrorism_success_model_leaves.predict(X_val)
print(classification_report(y_val,success_pred_leaves))


# In[ ]:


print(confusion_matrix(y_val,success_pred_leaves))


# **4. WIthout max_leaf_nodes and max_depth**

# In[ ]:


terrorism_success_model_bare = tree.DecisionTreeClassifier(random_state = 1)
terrorism_success_model_bare.fit(X_train, y_train)
success_pred_bare = terrorism_success_model_bare.predict(X_val)
print(classification_report(y_val,success_pred_bare))


# In[ ]:


print(confusion_matrix(y_val,success_pred_bare))


# **5. Using Random Forests instead**

# In[ ]:


rf = RandomForestClassifier(n_estimators=100) 
rf = rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
print(classification_report(y_val,rf_pred))


# In[ ]:


print(confusion_matrix(y_val,rf_pred))


# Clearly the random forest classifier is more precise than a decision tree classifier. This should be expected. Of the four decision tree classifier models I used, the max_leaf_nodes classifiers was the most precise, though the values for max_depth and max_leaf_nodes were selected rather arbitrarily out of intuition to balance underfitting and overfitting. 

# **Feature Importance**

# In[ ]:


for name, score in zip(X_train, rf.feature_importances_):
    print(name, score)


# In[ ]:


data = go.Bar(
    y=['extended','suicide', 'individual',  'multiple', 'nkill',"nwound","natlty1",'weaptype1',  'attacktype1','targtype1', 
       'imonth', 'longitude', 'iday', 'latitude'],
    x=[0.0008434176541557457, 0.0014349551050431954,0.01660395793541506,0.021650941814853716,
       0.026608046715114467,0.032573556495003576,0.042784340039457504,0.046320397204197755,
       0.0754779145247244,0.10563968568238052,0.1245681748749377,0.1588732207915285,
       0.16474122932796273,0.18188016183522507],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Relative Importance of the Features in the Random Forest',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **Predicting the number of kills**

# In[ ]:


features_nkill = ['imonth', 'iday', 'extended', 'latitude', 'longitude', 'multiple','success', 'suicide', 'attacktype1', 'targtype1', 'natlty1','individual', 'weaptype1', 'nwound']
X_nkill = df_USA[features_nkill]
y_nkill = df_USA["nkill"]
X_nkill_train, X_nkill_val, y_nkill_train, y_nkill_val = train_test_split(X_nkill, y_nkill, test_size=0.20, random_state=1)


# **1. No max depth or max leaf nodes**

# In[ ]:


nkill_model = dtr(random_state = 1)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))


# **2. Not max depth but max leaf nodes**

# In[ ]:


nkill_model = dtr(random_state = 1, max_leaf_nodes = 27)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))


# **3. Max depth but not max leaf nodes** 

# In[ ]:


nkill_model = dtr(random_state = 1, max_depth = 15)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))


# **4. Max depth and max leaf nodes**

# In[ ]:


nkill_model = dtr(random_state = 1, max_depth = 15, max_leaf_nodes = 27)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))


# **5. Random forest regressor**

# In[ ]:


nkill_model = rfr(random_state = 1, n_estimators = 10)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))


# Clearly the random forest regressor is the most precise as on average it has the lowest absolute error. 
# Now I will examine the feature importance as I had done with the successfulness classifier. 

# In[ ]:


for name, score in zip(X_nkill_train, nkill_model.feature_importances_):
    print(name, score)


# In[ ]:


features_nkill = ['imonth', 'iday', 'extended', 'latitude', 'longitude', 'multiple','success', 'suicide', 'attacktype1', 'targtype1', 'natlty1','individual', 'weaptype1']
X_nkill = df_USA[features_nkill]
y_nkill = df_USA["nkill"]
X_nkill_train, X_nkill_val, y_nkill_train, y_nkill_val = train_test_split(X_nkill, y_nkill, test_size=0.20, random_state=1)
nkill_model = rfr(random_state = 1, n_estimators = 100)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))


# Without the nwound variable the mae value is considerably higher, around 33% more. 

# In[ ]:


sorted_importance_dict = sorted(zip(X_nkill_train, nkill_model.feature_importances_), key=lambda x: x[1])
print(sorted_importance_dict)


# In[ ]:


data = go.Bar(
    y=['extended','multiple','success','weaptype1','natlty1','iday','imonth','attacktype1',
       'individual','targtype1','latitude','suicide','longitude'],
    x=[6.229373475964622e-06, 0.00018061437686766628,0.00022199675803425399,0.0007881470118179646,
       0.0017783660577080336,0.00888759926303102,0.012838555852620923,0.018069844219501384,
       0.06589093075573221,0.09836071796324783,0.10878349610841871,0.271867786077219,0.41232571618232533],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Relative Importance of the Features in the Random Forest',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **Predicting for a new attack**

# In[ ]:


nkill_model = rfr(random_state = 1, n_estimators = 10)
nkill_model.fit(X_nkill,y_nkill)

month = 9
day = 11
extended = 0
latitude = 40.711675 
longitude = -70.013285
multiple = 1
success = 1
suicide = 1
attackType = 8
targetType = 11
natlty1 = 217
individual = 1
weaponType = 12

kill_count = nkill_model.predict([[month,day,extended,latitude,longitude,multiple,success,suicide,
                                  attackType,targetType,individual,weaponType, natlty1]])
print("Unfortunately, this attack will kill "+str(int(kill_count[0]))+" people...")


# Intuitively, I would say that this model is pretty awful. You would not expect an attack to kill 1124 people. Hoever, you never know, an UNARMED assault may be able to kill 1065 people, especially if the unarmed assaulter is jackie chan. 
# 
# **Final Questions**
# 
# 1. How can I improve these models (clearly they could do with some improvement)?
# 2. Is there a way to visualise these models?
# 3. Are there any models that work better than the random forests and decision trees?
# 4. Is there a way to find the ideal value for max_depth, max_leaf_nodes and n_estimators?
# 5. As someone new to this field, what advice would you have on learning the ins and outs of these and other models?
# 6. What should my next steps be? Are there any models I should look at or should I learn some special maths before progressing? 
# 
# As I say, this is the first time I have worked with data and models so any advice, feedback or suggestions would be greatly appreaciated. 
# 
# AG
