#!/usr/bin/env python
# coding: utf-8

# **Thank you for opening this script!**
# 
# **I have made all efforts to walkthrough this dataset and have tried making it simple and steady so that this notebook acts as a good starting point for new Kagglers and new machine learning enthusiasts.**
# 
# **Please upvote this kernel if you find it useful for your understanding at this data. Your comments on how we can improve this kernel is always  welcome. Thanks.**
# 
# **My other exploratory studies can be accessed here : https://www.kaggle.com/roshanchoudhary/kernels** (Very few as of now)

# **Data_Dictionary**
# 
# 1. Elevation = Elevation in meters.
# 2. Aspect = Aspect in degrees azimuth.
# 3. Slope = Slope in degrees.
# 4. Horizontal_Distance_To_Hydrology = Horizontal distance to nearest surface water features.
# 5. Vertical_Distance_To_Hydrology = Vertical distance to nearest surface water features.
# 6. Horizontal_Distance_To_Roadways = Horizontal distance to nearest roadway.
# 7. Hillshade_9am = Hill shade index at 9am, summer solstice. Value out of 255.
# 8. Hillshade_Noon = Hill shade index at noon, summer solstice. Value out of 255.
# 9. Hillshade_3pm = Hill shade index at 3pm, summer solstice. Value out of 255.
# 10. Horizontal_Distance_To_Fire_Point = sHorizontal distance to nearest wildfire ignition points.
# 11. Wilderness_Area1 = Rawah Wilderness Area
# 12. Wilderness_Area2 = Neota Wilderness Area
# 13. Wilderness_Area3 = Comanche Peak Wilderness Area
# 14. Wilderness_Area4 = Cache la Poudre Wilderness Area
# 
# **Soil_Type1 to Soil_Type40 [Total 40 Types]**
# 
# **Cover_TypeForest Cover Type designation. Integer value between 1 and 7, with the following key:**
# 1. Spruce/Fir
# 2.  Lodgepole Pine
# 3.  Ponderosa Pine
# 4.  Cottonwood/Willow
# 5.  Aspen
# 6.  Douglas-fir
# 7.  Krummholz

# In[ ]:


#Let's import necessary dependencies 
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[ ]:


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)

#Read data for analysis
data=pd.read_csv('../input/covtype.csv')


#  **Explore Data Dimension and count of values without any sneak peek in Data**

# In[ ]:


print('Data Dimension:')
print('Number of Records:', data.shape[0])
print('Number of Features:', data.shape[1])


# In[ ]:


#Names of columns
print('Feature Names')
print(data.columns)


# **Looks like we got many binary independent features. Good!**
# **Now let us understand the data type of each features**

# In[ ]:


#A huge list!
print(data.info())


# In[ ]:



plt.figure(figsize=(6,4))
sns.countplot(y=data.dtypes ,data=data)
plt.xlabel("Data Type Count")
plt.ylabel("Data types")


# 1. **So we have complete Numeric Data, Even Better!!**
# 2. **Also there doesn't seem to be any missing value. Good work at Data Collection**

# In[ ]:


#Let's check for missing values once again
data.isnull().sum()


# **We forgot to check the Data distribution for each feature. Spend some good time here. Lot's of inferences I believe**

# In[ ]:


data.describe()


# **#Inferences:**
# 1. Few of the features looks skewed, we'll see those later.
# 2. No missing Values (We say this for the third time :p)
# 3. Wilderness Area and Soil Type are one hot coded.
# 4. Scales are different over the whole data, hence might need to scale for some required algorithms.

# **Skewness**
# >The skewness for a normal distribution is zero, and any symmetric data should have a skewness near zero. 
# >Negative values for the skewness indicate data that are skewed left and positive values for the skewness indicate data that are skewed right. 
# >By skewed left, it means that the left tail is long relative to the right tail. Similarly, skewed right means that the right tail is long relative to the left tail.

# In[ ]:


print('Skewness of the below features:')
print(data.skew())


# In[ ]:


skew=data.skew()
skew_df=pd.DataFrame(skew,index=None,columns=['Skewness'])
plt.figure(figsize=(15,7))
sns.barplot(x=skew_df.index,y='Skewness',data=skew_df)
plt.xticks(rotation=90)


# **#Inferences:**
# > Some of the Variables are heavily skewed hence need to be corrected or transformed  on a later stage. 

# **How about the class balance? We'll see**

# In[ ]:


class_dist=data.groupby('Cover_Type').size()
class_label=pd.DataFrame(class_dist,columns=['Size'])
plt.figure(figsize=(8,6))
sns.barplot(x=class_label.index,y='Size',data=class_label)


# > But I'm interested in percentwise distribution of each class. Let's check

# In[ ]:


for i,number in enumerate(class_dist):
    percent=(number/class_dist.sum())*100
    print('Cover_Type',class_dist.index[i])
    print('%.2f'% percent,'%')


# **#Inferences:**
# 1.  Cover_Type 1 and 2 i.e **Spruce/Fir** and **Lodgepole Pine** seems to dominate the area. 
# 2.  Also the Cover_Type 4 i.e **Cottonwood/Willow** is minimal compare to the rest

# **Oh common let us check the data atleast, enough with size and dimension**

# In[ ]:


data.head()


# >Nice! Now, Let's convert the whole data into few Mini datasets. I'll make use of it in plots
# * cont_data - Data without binary features i.e continuous features
# * binary_Data - Data having all binary features [Wilderness Areas + Soil Types]
# * wilderness_Data - Binary Wilderness Areas
# * Soil_Data - Binary Soil Types

# In[ ]:


cont_data=data.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points']

binary_data=data.loc[:,'Wilderness_Area1':'Soil_Type40']

Wilderness_data=data.loc[:,'Wilderness_Area1': 'Wilderness_Area4']

Soil_data=data.loc[:,'Soil_Type1':'Soil_Type40']


# 
# **I want to see the number of  values counts within each features, mainly for the Binary types**

# In[ ]:


#Iterate via columns of data having only binary features
for col in binary_data:
    count=binary_data[col].value_counts()
    print(col,count)


# **#Inferences:**
# > **This tells me lots of valuable insights. Mostly regarding the soil types. Wanna know? Ok, let me not hide it from you**.
# * It's Just that, there are some of the Soil types which consists of very few counts.  
# * Statistically speaking, for half a million records, balance number per soil type (total 40 in number) is 581012/40 = 14.5k
# * Whereas, here we see a different figure. I know that data need not be balanced all the times. But may be we can get rid of really small size features. Isn't it?
# * Let me list down those along with there size. I'm displaying the Soil type having less than 1000 occurence size

# In[ ]:


print('Soil Type',' Occurence_count')
for col in binary_data:
    count=binary_data[col].value_counts()[1] #considering all one's among 1 and 0's in each soil type
    if count < 1000:
        print(col,count)


# * I know this will make more sense in a visual such as bar graph right? I'm excited to see it too. But let's infer more from the numbers as of now. 
# * We'll do plottings once we start with Bivariate and Multivariate analysis. 
# * We'll see if we need to really drop the above soil types. 
# * We can only confirm on it if it is not aligned (give any relation) to our target variable i.e Cover_Type. So, please wait, do not conclude. Climax is yet to come :D

# **Let's get started with plots based EDA (Exploratory Data Analysis) **
# *  Fun begins here, am I right?
# * Data Distribution of features via Histograms. Although I love box plots more than histograms, we'll use boxplot to check distribution with respect to categorical variable. In our case that is Cover_Type, having 7 different category of classes.

# In[ ]:


# data_num = data.select_dtypes([np.int, np.float]) #If you need to select only numeric features. 
#Here we already have all numeric Data.

for i, col in enumerate(cont_data.columns):
    plt.figure(i)
    sns.distplot(cont_data[col])


# * > The above plots more or less tells us about the skewness that we saw earlier. Let's dig down into Bivariate and Multivariate Analysis
# * > Let's check for distribution with respect to our target. This is where magic happens!

# * > Here, First i want to check the shape of continous features with respect to the target class. Hence I'll use the continuous_data (cont_data) and plot a boxplot against target. 
# * > You can also look at violinplot here, It's visually appealing. 

# In[ ]:


# %%time
data['Cover_Type']=data['Cover_Type'].astype('category') #To convert target class into category

for i, col in enumerate(cont_data.columns):
    plt.figure(i,figsize=(8,4))
    sns.boxplot(x=data['Cover_Type'], y=col, data=data, palette="coolwarm")


# *  Plots looks cool right? What's Even more cool you know?
# *  The insights. Let's figure out very general insights
# *  There are couple of features which shows not much of variance with respect to classes
# *  And features such as 'Elevation', 'slope' and 'horizontal distance to road_ways does a good job

# > Let's do something similar for our binary features. This time we'll use countplot.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for i, col in enumerate(binary_data.columns):\n    plt.figure(i,figsize=(6,4))\n    sns.countplot(x=col, hue=data[\'Cover_Type\'] ,data=data, palette="rainbow")')


# * > So the plot does justice to the distribution which each class but I want to have a single feature having Soil_Type corresponding to each row. 
# * > Let's see if I can do it.  This will help me to visualize it better, instead of counting 0's and 1's in each one hot coded Soil types.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#If someone can help me with function to reverse one hot coding, please let me know in comment. I know this is not the robust way.\ndef rev_code(row):\n    for c in Soil_data.columns:\n        if row[c]==1:\n            return c  \n\ndata['Soil_Type']=Soil_data.apply(rev_code, axis=1) #Time consuming")


# > I'll do the same for Wilderness Area

# In[ ]:


get_ipython().run_cell_magic('time', '', "def rev_code(row):\n    for c in Wilderness_data.columns:\n        if row[c]==1:\n            return c  \n\ndata['Wilderness_Type']=Wilderness_data.apply(rev_code, axis=1) #Time consuming")


# > Yup! It's done. Looks like we have a desired single Soil_Type and Wilderness_Type feature. Let's now use count plot against our Target Cover_Type

# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(16,8))\nsns.countplot(x=\'Wilderness_Type\', hue=\'Cover_Type\',data=data, palette="rainbow")\nplt.xticks(rotation=90)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(16,8))\nsns.countplot(x=\'Soil_Type\', hue=\'Cover_Type\',data=data, palette="rainbow")\nplt.xticks(rotation=90)')


# * >Above two plots tells  us the count of trees in each class considering Wilderness and Soil Type.
# * >Soil_Type plot is not very clear since it's  too vast. So let's go by the number. We'll see how many and what type of Cover_Type we have under each soil Type

# In[ ]:


soil_counts = []
for num in range(1,41):
    col = ('Soil_Type' + str(num))
    this_soil = data[col].groupby(data['Cover_Type'])
    totals = []
    for value in this_soil.sum():
        totals.append(value)
    total_sum = sum(totals)
    soil_counts.append(total_sum)
    print("Total Trees in Soil Type {0}: {1}".format(num, total_sum))
    percentages = [ (total*100 / total_sum) for total in totals]
    print("{0}\n".format(percentages))
print("Number of trees in each soil type:\n{0}".format(soil_counts))


# **Did we check the co-relation??**
#  * > No we didn't. This is something that I usually check first. No, problem. it's never too late.
#  * > Let's better vizualise it via heatmap. All in one!

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(cont_data.corr(),cmap='magma',linecolor='white',linewidths=1,annot=True)


# * >Couple of features are have a good amount of co-relation. Guess which one? I'll tell you.
# * >  Hillshade_9am ~ Hillshade_3pm and Aspect ~ Hillshade_3pm

# In[ ]:


g = sns.PairGrid(cont_data)
g.map(plt.scatter)


# * > This gives us the relation and its shape with respect to other features. Various inferences can be drwan out.
# * > Pairgrid plot is just awesome. And it's even more awesome when it's combined with KDE clusters. 
# * > But for considerably heavy data, its time consuming. Be aware before running the below plot.

# In[ ]:


# %%time
# g = sns.PairGrid(cont_data)
# g.map_diag(plt.hist)
# g.map_upper(sns.kdeplot)
# g.map_lower(sns.kdeplot)


# * > There's lot of scope for Data Viz. as far this dataset is concerned. My objective was a surface walkthrough the dataset. I would roll out new versions on this part by part. 
# * >Let's now wind it up by Data Modelling. Another Excitement, right?

# **Data Modelling**

# * X = Input or independent variables
# * y=  Target variable ('Cover_Type')

# In[ ]:


X=data.loc[:,'Elevation':'Soil_Type40']
y=data['Cover_Type']


# >Let us take a step to remove the features with low Std deviation as demonstrated earlier. 
# >Also I'll remove one of the co-related variable

# In[ ]:


#Features to be removed before the model
rem=['Hillshade_3pm','Soil_Type7','Soil_Type8','Soil_Type14','Soil_Type15',
     'Soil_Type21','Soil_Type25','Soil_Type28','Soil_Type36','Soil_Type37']


# In[ ]:


#Remove the unwanted features
X.drop(rem, axis=1, inplace=True)


# In[ ]:


#Splitting the data into  train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# * ** I have tried various Classification algorithms out of which KNN served the best.**
# * ** Algorithms such as RandomForest and DecisionTree are doing a decent job here. So please explore.**

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Setup arrays to store training and test accuracies\nneighbors = np.arange(1,7)\ntrain_accuracy =np.empty(len(neighbors))\ntest_accuracy = np.empty(len(neighbors))\n\nfor i,k in enumerate(neighbors):\n    #Setup a knn classifier with k neighbors\n    knn = KNeighborsClassifier(n_neighbors=k)\n    \n    #Fit the model\n    knn.fit(X_train, y_train)\n    \n    #Compute accuracy on the training set\n    train_accuracy[i] = knn.score(X_train, y_train)\n    \n    #Compute accuracy on the test set\n    test_accuracy[i] = knn.score(X_test, y_test) ')


# > Let's visualize the change in accuracies with respect to train and test data at different neighbors 

# In[ ]:


#Generate plot
plt.figure(figsize=(10,6))
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
#plt.show()


# > Neighbor value = 5 yeilds the best result. Let's go by that for now. 

# In[ ]:


#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=5) #Using Eucledian distance


# In[ ]:


#Fit the model
knn.fit(X_train,y_train)


# In[ ]:


#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
Accuracy=knn.score(X_test,y_test)
print('KNN Accuracy:',Accuracy)


# **Not bad. KNN works great here. Lazy learner is doing a good work at differentiating a CoverType. **

# **I'll put the accuracies obtained by various other classification techniques. Try to enhance it more via Cross Validation may be.**
# **Let me know in comment if you manage to raise your accuracies. **
# > **I'm gonna do it too. Those are my next steps, such as CV, more insights, Feature engineering etc. I'll roll out updates part by part**

# In[ ]:


import scipy.stats as ss
from statsmodels.formula.api import ols
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[ ]:


get_ipython().run_cell_magic('time', '', 'MLA = []\nZ = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , GaussianNB() ,RandomForestClassifier() , \n     GradientBoostingClassifier()]\nX = ["LinearSVC" , "DecisionTreeClassifier" , "LogisticRegression" , "GaussianNB" ,"RandomForestClassifier" , \n     "GradientBoostingClassifier"]\n\nfor i in range(0,len(Z)):\n    model = Z[i]\n    model.fit( X_train , y_train )\n    pred = model.predict(X_test)\n    MLA.append(accuracy_score(pred , y_test))')


# In[ ]:


d = { "Algorithm" : X, "Accuracy" : MLA }

dfm = pd.DataFrame(d)
dfm


# *  **Try to surpass these accuracies. **
# *  **My objective was to 'Get to know' the Forest Cover Type Dataset for which I tried to articulate it step by step.**
# *  **If you liked it, please let me know with a upvote, It serves a Motivation. Good Luck and Thank you for spending your time here! :)**
