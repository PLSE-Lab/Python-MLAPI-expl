#!/usr/bin/env python
# coding: utf-8

# # Electrical Failure Analysis

# by [Prashant Brahmbhatt](https://www.github.com/hashbanger)

# ![Electrical](http://www.belyeapower.com/assets/templates/belyea/images/animation3.jpg)

# _______

# #### Importing libraries

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


sns.set_style('darkgrid')


# Importing the dataset

# In[ ]:


data = pd.read_csv('../input/electric_faults_data.csv')
data.head()


# In[ ]:


print("The shape of the data is :",data.shape)


# In[ ]:


data.describe()


# Our Target Class is the **type_of_fault** and **nature** of the fault

# ## Handling Missing Values

# In[ ]:


plt.figure(figsize=(12,7))
f = sns.heatmap(data.isnull(), cbar = False, cmap = 'viridis')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.title("Heatmap of Missing Values", fontsize = 15)
plt.show()


# Observations:  
# the columns **other_line_status** and **observation** are almost all missing so we can drop those.  
# We will observe the significant missing values in column **repairs** before dropping.

# There is a slight missing terms in the **tripping_reason** column.  
# So we can fill it with the most frequent value

# In[ ]:


data['tripping_reason'].value_counts()


# filling the missing value as ***transient_fault***

# In[ ]:


data['tripping_reason'].fillna(value = 'transient fault', inplace = True)


# Another column having missing values is **other_circuit**

# In[ ]:


data['other_circuit'].value_counts()


# Since we observe that the column contain only single and almost definitely occuring type of value so it is not suggestable to include this column having very less entropy, in the model

# In[ ]:


data.drop('other_line_status', inplace = True, axis  =1)
data.drop('observation', inplace = True, axis  =1)
data.drop('other_circuit', inplace = True, axis  =1)


# Observing the missing values in **repairs_carried** column

# In[ ]:


data['repair_carried'].value_counts()


# filling it with most frequent value

# In[ ]:


data['repair_carried'].fillna(value = 'nil', inplace= True)


# ## EDA

# Separating the **years**, **months** and **hours** from the date and time

# In[ ]:


#Separating year
data['trip_year'] = pd.to_datetime(data['date_of_trip'], dayfirst= True ).dt.year
data['restore_year'] = pd.to_datetime(data['date of restoration'], dayfirst= True ).dt.year

#Separating month
data['trip_month'] = pd.to_datetime(data['date_of_trip'], dayfirst= True ).dt.month
data['restore_month'] = pd.to_datetime(data['date of restoration'], dayfirst= True ).dt.month


#separating hours
data['trip_hour'] = pd.to_datetime(data['time_of_trip']).dt.hour
data['restore_hour'] = pd.to_datetime(data['time_of_restoration']).dt.hour


# Mapping the integer **month** values to names

# In[ ]:


data.head(10)


# In[ ]:


data['trip_month'] = data['trip_month'].map({1:'January', 2:'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',
                               8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})

data['restore_month'] = data['restore_month'].map({1:'January', 2:'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',
                               8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'})


# Getting the weekdays from the date

# In[ ]:


data['weekday'] = pd.to_datetime(data['date_of_trip']).dt.weekday


# Mapping the days number to names, where 0 maps to monday and so on

# In[ ]:


data['weekday'] = data['weekday'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})


# In[ ]:


temp = data['weekday'].value_counts().reset_index()

plt.figure(figsize= (12,7))
plt.title('Trips on Weekdays',fontsize = 15)
f = sns.barplot(x = temp['index'], y = temp['weekday'], palette = 'hls')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.xlabel('Weekdays', fontsize = 16)
plt.yticks(list(range(max(temp['weekday']))))
plt.show()


# In[ ]:


temp = data['trip_year'].value_counts()

plt.figure(figsize= (12,7))
plt.title('Trips in Years',fontsize = 15)
f = sns.barplot(x = temp.index, y = temp.values, palette = 'Set2')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.xlabel('Years', fontsize = 16)
plt.yticks(list(range(max(temp))))
plt.show()


# Observation: (**Barplot**)    
# We can see that year **2016** has most faults above all while 2015 has the least.

# In[ ]:


temp = data['trip_month'].value_counts()

plt.figure(figsize= (12,7))
plt.title('Trips in Months',fontsize = 15)
f = sns.barplot(x = temp.index[::-1], y = temp.values[::-1], palette = 'rainbow')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.xlabel('Months', fontsize = 16)
plt.yticks(list(range(max(temp))))
plt.show()


# Observation: (**Barplot**)  
# From the above plot we can observe that most number of faults are during Summer season and tens to happen less during the winter months.

# In[ ]:


temp = data['weather'].value_counts()

plt.figure(figsize= (12,7))
plt.title('Trips in Weather',fontsize = 15)
f = sns.barplot(x = temp.index, y = temp.values, palette = 'inferno')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.xlabel('Weather', fontsize = 16)
plt.yticks(list(range(0,max(temp)+2)))
plt.show()


# Observation: (**Barplot**)  
# The faults are mostly during Clear skies.

# In[ ]:


temp = data['line_trip'].value_counts().reset_index()

plt.figure(figsize=(9,9))
f = plt.pie(x = temp['line_trip'],labels = ['Yes','No'], colors=('lightblue','orange'), autopct= "%1.1f%%")
plt.title('Line Trips at other End', fontsize  = 15)
plt.show()


# Observation: (**Barplot**)  
# Trips in the faults has been more than thrice than no trips.

# In[ ]:


temp = data['tripping_reason'].value_counts()

plt.figure(figsize= (12,7))
plt.title('Trips Reasons',fontsize = 15)
f = sns.barplot(x = temp.index, y = temp.values, palette = 'autumn')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13}, rotation = 30)
plt.xlabel('Reasons', fontsize = 16)
plt.yticks(range(max(temp)+2))
plt.show()


# Observation: (**Barplot**)  
# Among the faults reasons that most account of faults is due to **transient_fault** or **bad_weather**

# In[ ]:


plt.figure(figsize= (15,10))
#plt.suptitle("Distributions of Different Features", fontsize = 20)
#Histograms
plt.subplot(3,3,1)
sns.distplot(data['voltage'], rug = True, kde = False)
plt.xlabel('Voltage in KiloVolts', fontsize = 12)
plt.title('Distribution of Voltage',fontsize = 15)

plt.subplot(3,3,2)
sns.distplot(data['load_of_line'], color= 'green',rug = True, kde = False)
plt.title('Distribution of Load of Line',fontsize = 15)
plt.xlabel('Load on line in Amperes', fontsize = 12)

plt.subplot(3,3,3)
sns.distplot(data['frequency'], rug= True, color= 'orange', kde = False)
plt.xlabel('Voltage in KiloVolts', fontsize = 12)
plt.title('Distribution of Frequency',fontsize = 15)


#Kde Plots
plt.subplot(3,3,4)
sns.kdeplot(data['voltage'], shade = True)
plt.xlabel('Voltage in KiloVolts', fontsize = 12)
plt.title('Distribution of Voltage',fontsize = 15)

plt.subplot(3,3,5)
sns.kdeplot(data['load_of_line'], shade = True, color = 'g')
plt.title('Distribution of Load of Line',fontsize = 15)
plt.xlabel('Load on line in Amperes', fontsize = 12)

plt.subplot(3,3,6)
sns.kdeplot(data['frequency'],shade= True, color = 'Orange')
plt.title('Distribution of Frequency',fontsize = 15)

#Box Plots
plt.subplot(3,3,7)
sns.boxplot(x = data['voltage'], orient = 'v',color= 'b', boxprops=dict(alpha=.5))
plt.subplot(3,3,8)
sns.boxplot(x = data['load_of_line'], orient = 'v', color= 'g', boxprops=dict(alpha=.5))
plt.subplot(3,3,9)
sns.boxplot(x = data['frequency'], orient = 'v', color= 'Orange', boxprops=dict(alpha=.5))

plt.tight_layout()
plt.show()


# Observation: (**Histogram, Kernel Plot and Box Plot**)  
#  
# We can see that **Voltage** and **Current** have considerable spread however there is very little spread in the **frequency** parameter.  
# Since it is not entirely fixed rather than dropping it we can scale it for the model.  
# There is not a sufficient amount of data to have some outliers either.

# ### Versus Plots

# #### Voltage vs load_of_line

# In[ ]:


sns.jointplot(x = data['load_of_line'], y = data['voltage'], kind = 'reg', color= 'g')

plt.show()


# #### load_of_line vs frequency

# In[ ]:


sns.jointplot(x = data['load_of_line'], y = data['frequency'], kind = 'reg', color= 'darkorange')
plt.show()


# #### voltage vs frequency

# In[ ]:


sns.jointplot(x = data['voltage'], y = data['frequency'], kind = 'reg', color = 'blue')
plt.show()


# Observation: (**Joint Plots**)  
# 1. **Voltage vs load_of_line** shows a sufficiently strong negative correlation of -0.76.  
#     So we might consider dropping the voltage parameter since the dataset is not very large and we want to reduce any multicollinearlity also  
#     as per Occam's Razor principle we will drop it if the accuracy is not drastically effected.  
# 2. **load_of_line vs frequency** does not have any significant relationship.  
# 3. same goes for **voltage vs frequency**.  

# In[ ]:


temp = data['trip_hour'].value_counts()
plt.figure(figsize= (10,10))

plt.subplot(2,1,1)
sns.pointplot(x = temp.index, y = temp.values ,palette= 'Reds')
sns.pointplot(x = temp.index, y = temp.values ,join= True, color = 'r',markers = '')
plt.title('Trips on Hours',fontsize = 15)
plt.xlabel('Trip Hours of Day', fontsize = 12)
plt.ylabel('Number of Hours', fontsize = 12)
plt.yticks([0,1,2,3,4])

temp = data['restore_hour'].value_counts()
plt.subplot(2,1,2)
sns.pointplot(x = temp.index, y = temp.values ,palette= 'Greens')
sns.pointplot(x = temp.index, y = temp.values ,join= True, color='g', markers = '')
plt.title('Restoration on Hours',fontsize = 15)
plt.xlabel('Restore Hours of Day', fontsize = 12)
plt.ylabel('Number of Hours', fontsize = 12)
plt.yticks([0,1,2,3,4])
plt.show()


# Observation: (**Line Plot**)  
# We can observe that the most fault peaks (plot 1 in red), has highest peaks during the very early morning and during dusk.  
# There is not sufficient regular pattern so it may or may not be a considerable parameter.

# In[ ]:


for i in range(0,len(data['repair_carried'])):
    if data['repair_carried'][i] == 'nil':
        data['repair_carried'][i] = 'None'
temp = data['repair_carried'].value_counts()
plt.figure(figsize= (12,7))
plt.title('Repairs Carried',fontsize = 15)
f = sns.barplot(x = temp.index, y = temp.values, palette = 'Set1')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.xlabel('Repair Types', fontsize = 15)
plt.yticks(range(max(temp)+2))
plt.show()


# Mostly there were no repairs were done or maybe not recorded and if there are no faults then the repairs would be **None** for them too. So we can rule this out as a prarmeter

# #### Visualizing the nature column

# The nature column is also a target class

# In[ ]:


plt.figure(figsize= (9,7))
plt.title('Nature of the fault',fontsize = 15)
f = sns.countplot(data['nature'], palette= 'hls')
f.set_xticklabels(labels = f.get_xticklabels(), fontdict={'fontsize':13})
plt.xlabel("Types", fontsize= 14)
plt.show()


# There is not sufficient data to classify this column.

# ### Mapping the **line_trip** feature

# Converting the **line_trip** feature from categorical to numeric

# In[ ]:


data['line_trip'] = data['line_trip'].map({'no':0, 'yes':1})


# ### Mapping the Target Class

# Mapping the **type_of_fault** to integer values signified by:  
# **-1** - Low Fault  
# **0** - Medium Fault  
# **1** - High Fault

# In[ ]:


data['type_of_fault'] = data['type_of_fault'].map({'low':-1, 'medium':0, 'high': 1})


# Finally the data look as below

# In[ ]:


data.head()


# Due to small size of data choosing lot of features will increase the chances of overfitting and since lack of sufficient evidence in many fatures we will limit our feature list to **line_trip**, **load_of_line** and **frequency**

# ## Predictive Modelling

# ### Scaling the features

# As we have previously observed **frequency** needs to be scaled and so do the **load_of_line** parameter

# Using **Standardization Equation** for scaling  
# ![sc](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2015/09/24071007/Z-score-form.png)

# #### Preparing the Train Fetures and Train Labels

# In[ ]:


X_full = data.iloc[:, [3,4,6]].values
y_full = data['type_of_fault'].values


# #### Scaling the data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_full)
X_full = sc.transform(X_full)


# #### Splitting the data into 75% Train set and 25% Test Set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.25, random_state = 1)


# ### Using KNeighborsClassifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier.predict(X_test)


# #### Accuracy on Train Set

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_train, classifier.predict(X_train)))


# So we got approximately 94% accuracy on the train set.

# #### Accuracy on Test Set

# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_test_pred)
print(cr)


# So we got approximately 91% accuracy on the train set.

# #### To check for single observation

# In[ ]:


t = input("Enter Trip\t")
l = float(input("Enter load of line\t"))
f = float(input("Enter Frequency\t"))

if t =='yes':
    t = 1
elif t == 'no':
    t = 0
    
samp = np.array([[int(t), int(l), float(f)]])
samp = sc.transform(samp)
res = classifier.predict(samp)
print("\n------Output-----\n")
if res == -1:
    print("Low Fault")
elif res == 0:
    print("Medium Fault")
else:
    print("High Fault")


# ### Using Multinomial Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver= 'newton-cg',multi_class= 'multinomial')
classifier.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier.predict(X_test)


# #### Accuracy on Train Test

# In[ ]:


print(classification_report(y_train, classifier.predict(X_train)))


# So we got approximately 94% accuracy on the train set.

# #### Accuracy on Test Set

# In[ ]:


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_test_pred)
print(cr)


# So we got approximately 93% accuracy on the train set.

# In[ ]:


t = input("Enter Trip\t")
l = float(input("Enter load of line\t"))
f = float(input("Enter Frequency\t"))

if t =='yes':
    t = 1
elif t == 'no':
    t = 0
    
samp = np.array([[int(t), int(l), float(f)]])
samp = sc.transform(samp)
res = classifier.predict(samp)
print("\n------Output-----\n")
if res == -1:
    print("Low Fault")
elif res == 0:
    print("Medium Fault")
else:
    print("High Fault")


# ### de nada!

# Any suggestions or correction are welcome with heart!
