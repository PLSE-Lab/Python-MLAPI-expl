#!/usr/bin/env python
# coding: utf-8
### Context
Call Test Measurements for Mobile Network Monitoring and Optimization.
### Content
The measurements were performed with smartphones and collected on proprietary databases.
https://www.kaggle.com/valeriol93/predict-qoe
# In[ ]:


#import Libraries
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")    


# Importing Dataset

# In[ ]:


df = pd.read_excel("../input/dataset.xlsx")


# In[ ]:


df.head() 


# # mos research & domain knowledge
# https://en.wikipedia.org/wiki/Timing_advance  - range of GSM 35km <br>
# https://en.wikipedia.org/wiki/Cell_site<br>
# https://en.wikipedia.org/wiki/Mean_opinion_score  - information about MOS range <br>
# https://www.voipmechanic.com/mos-mean-opinion-score.htm <br>
# https://www.voip-info.org/call-quality-metrics/ <br>
# https://www.twilio.com/docs/glossary/what-is-mean-opinion-score-mos <br>
# 

# In[ ]:


df.info()


# In[ ]:





# In[ ]:


df.columns


# #### Renaming feature columns

# In[ ]:


df.columns = ['DateOfTest','Signal','Speed','DistanceFromSite','CallTestDuration','CallTestResult','CallTestTechnology','CallTestSetupTime','MOS']


# In[ ]:


df.head()


# In[ ]:


df.MOS.nunique()


# ## Looking at the output class it will be hard to classify in 35 categories,from the domain research we can classify this values into a particular Range

# ## I tried to solve this problem using regression as well which gave me RMSE score of 0.86 using RFR , but here requirement is to use classification algorithm

# In[ ]:


df.MOS.unique()


# # Minimum opinion score 
# 
# Very satisfied	         4.3-5.0
# 
# Satisfied	             4.0-4.3
# 
# Some users satisfied	 3.6-4.0
# 
# Many users dissatisfied	 3.1-3.6
# 
# all users dissatisfied   2.6-3.1
# 
# Not recommended	         1.0-2.6

# In[ ]:


df.shape


# In[ ]:


df.MOS.value_counts()


# In[ ]:


df.MOS.value_counts(bins=5)


# In[ ]:


df.MOS.value_counts(bins=4)


# In[ ]:


df.MOS.value_counts(bins=3)


# In[ ]:


sns.countplot(x=df.MOS)


# #Satisfied	4.0-4.4
# #Some users satisfied	3.6-4.0
# #Many users dissatisfied	3.1-3.6
# #Nearly all users dissatisfied	2.6-3.1
# #Not recommended	1.0-2.6

# In[ ]:


df.loc[(df['MOS']>=1 )&(df['MOS']<=2.6),'MOS'].shape


# In[ ]:


df.loc[(df['MOS']>2.6 )&(df['MOS']<=3.6),'MOS'].shape


# In[ ]:


df.loc[(df['MOS']>3.6 )&(df['MOS']<=4.4),'MOS'].shape


# We can categorize this 35 output categories (Continious values)  into 3 categories which will be <br>
# 1) Poor MOS (1 to 2.6) <br>
# 2) Average MOS Range 2.6 - 3.6 <br>
# 3) Good MOS Range 3.6 - 4.4 <br>

# In[ ]:


### transforming 35 values into 3 classess
df.loc[(df['MOS']>=1 )&(df['MOS']<=2.6),'MOS'] = 2
df.loc[(df['MOS']>2.6 )&(df['MOS']<=3.6),'MOS'] = 3
df.loc[(df['MOS']>3.6 )&(df['MOS']<=4.4),'MOS'] = 4        


# In[ ]:


df['MOS'].value_counts().plot(kind= 'barh', color = 'orange', title = 'MOS')
plt.show()


# In[ ]:


##descibe the data for better understanding`
df.describe()


# ### From the description we can see DistanceFromSite feature has very high varience in the records , there may be outliers since the max distance is 700km which is not possible as the maximum coverage of antenna can be of 35km 

# In[ ]:


df.describe(include= 'all')


# In[ ]:


#Non Numeric features
object_columns_df = df.select_dtypes(include=["object"])
print (object_columns_df.iloc[0])


# In[ ]:


##transforming categorical features


# In[ ]:


df.CallTestResult.value_counts()


# In[ ]:


df.groupby('CallTestResult')['MOS'].value_counts()


# # Pandas-Profiling
# ##lets look the data in more detail using pandas profiling
# #using pandas profiling we can look more into the data in shorter time

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling
profile = pandas_profiling.ProfileReport(df)
#generate html
#profile.to_file(outputfile="output.html")


# In[ ]:


profile


# using Pandas profile output we can analyse the data,like missing values,feature's description , min max mean IQR Q1, Q3 values etc

# In[ ]:


result_replace = {
    "CallTestResult" : {
        "SUCCESS": 1,
        "FAILURE - SETUP FAIL": 0,
        "FAILURE - DROP CALL" : 0,
    }
}
df = df.replace(result_replace)


# In[ ]:


df.groupby('CallTestResult')['MOS'].value_counts()


# In[ ]:


df['CallTestTechnology'].value_counts()


# In[ ]:


df.groupby('CallTestTechnology')['MOS'].value_counts()


# In[ ]:


df.groupby('CallTestTechnology')['CallTestResult'].value_counts()


# In[ ]:


df.groupby('CallTestResult')['MOS'].value_counts()


# In[ ]:


df.info()


# In[ ]:


#OneHotEncoder
dummy_df = pd.get_dummies(df['CallTestTechnology'])
df = pd.concat([df, dummy_df], axis=1)
df = df.drop(['CallTestTechnology'], axis=1)


# In[ ]:


df.head()


# In[ ]:


#DateTime
df["DateOfTest"]=pd.to_datetime(df['DateOfTest']) #convert an object to a datetime type


# In[ ]:


df.head()


# #Converting DateTime Feature because this MOS is depend on Datetime aswell

# In[ ]:


import calendar
df['minute']=df['DateOfTest'].apply(lambda x:x.minute)
df['second']=df['DateOfTest'].apply(lambda x:x.second)
df['day']=df['DateOfTest'].apply(lambda x:x.day)
df['hour']=df['DateOfTest'].apply(lambda x:x.hour)
df['weekday']=df['DateOfTest'].apply(lambda x:calendar.day_name[x.weekday()])
df['month']=df['DateOfTest'].apply(lambda x:x.month)
df['year']=df['DateOfTest'].apply(lambda x:x.year)


# In[ ]:


#df['minute'].nunique() # 60
#df['second'].nunique() # 60
#df['hour'].nunique() # 24 
#df['day'].nunique() # 31
#df['weekday'].nunique() #7
df['month'].nunique() #4
#df['year'].nunique() #1


# this data is of year 2017 and records are for 4 months

# In[ ]:


##dropping month and year columns and dateoftest
df = df.drop(['year','month','DateOfTest'],axis=1)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df['weekday'].unique()


# In[ ]:


##transforming weekday column
#df.weekday = df.weekday.map({'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6})


# Using OneHotEncoder instead of LabelEncoder will increase the Accuracy by 1% 
# I tried it on the other notenook

# In[ ]:


#OneHotEncoder
dummy_df = pd.get_dummies(df['weekday'])
df = pd.concat([df, dummy_df], axis=1)
df = df.drop(['weekday'], axis=1)


# In[ ]:


df.info()


# In[ ]:


plt.hist(df['MOS'])
plt.show()


# In[ ]:


#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [5])
#df = onehotencoder.fit_transform(df).toarray()


# In[ ]:


df.describe()


# In[ ]:


object_columns_df = df.select_dtypes(include=["object"])
print (object_columns_df.iloc[0])


# In[ ]:


##missing value
df.isna().sum()


# Signal and DistanceFromSite features has missing value, 
# We can replace missing value from Signal with mean 
# but since DistanceFromSite feature has outliers which are responsible for increasing the mean
# this data is 10 % off the total records so it will impact the prediction

# In[ ]:


##correation between the features
correlations = df.corr()
sns.heatmap(data= correlations,square =True , cmap = "bwr")
plt.yticks(rotation= 0)
plt.xticks(rotation= 90)


# In[ ]:


df[["UMTS","Signal"]].corr()


# In[ ]:


df.describe()


# #distancefromSite has variation in data  
# min and max value has bi difference
# #Domain knowledge
# maximum coverage of network can be of 35 km

# # Outlier Detection

# In[ ]:


Q1 = df['DistanceFromSite'].quantile(0.25)
Q3 = df['DistanceFromSite'].quantile(0.75)
print('Q1 :' + str(Q1))
print('Q3 :' + str(Q3))
IQR = Q3 - Q1
print('IQR :' + str(IQR))
print(Q1 - IQR)
print(Q3 + IQR)


# In[ ]:


print('max value : ' + str(df['DistanceFromSite'].max()))
print('min value :' + str(df['DistanceFromSite'].min()))
print('mean value: '+ str(df['DistanceFromSite'].mean()))
print('mode value:'+ str(df['DistanceFromSite'].mode()))


# In[ ]:


print('No. of records with distance > 100km : ' + str(df[((df['DistanceFromSite'] > 100000))].shape))
print('No. of records with distance > 1343 meter : ' + str(df[((df['DistanceFromSite'] > 1343))].shape))
print('No. of records with distance > 35km : ' + str(df[((df['DistanceFromSite'] > 35000))].shape))
#df[((df['DistanceFromSite'] > 100000) & (df['MOS'] > 3))].shape
#df[((df['DistanceFromSite'] > 1343))].shape  # IQR


# In[ ]:


df.DistanceFromSite.isna().sum()


# 

# In[ ]:


import seaborn as sns
sns.boxplot(df.DistanceFromSite)
###can not consider outlier


# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['DistanceFromSite'], df['MOS'])
ax.set_xlabel('DistanceFromSite')
ax.set_ylabel('MOS')
plt.show()


# In[ ]:


import seaborn as sns
sns.boxplot(x = 'MOS' , y = 'DistanceFromSite' , data = df)


# In[ ]:


df["Signal"].fillna(df['Signal'].mean(), inplace = True)


# In[ ]:


##missing value
df.isna().sum()


# In[ ]:


##decision to replace the missing records from distancefromsite feature
df[["DistanceFromSite","LTE"]].corr()


# In[ ]:


df['DistanceFromSite'].isna().sum()


# In[ ]:


df[((df['DistanceFromSite'] > 400000) & (df['MOS'] > 3))].shape #& (df['MOS'] > 3)


# In[ ]:


df[((df['DistanceFromSite'] > 1300))].shape


# In[ ]:


sns.distplot(df.Signal)


# In[ ]:


df['DistanceFromSite'].shape


# In[ ]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['DistanceFromSite'], df['MOS'])
ax.set_xlabel('DistanceFromSite')
ax.set_ylabel('MOS')
plt.show()


# In[ ]:


#df[(df['DistanceFromSite'] > 20000)].shape  #(3143, 15)
#df[(df['DistanceFromSite'] < 1343)].shape  #(82685, 15)
#df[(df['DistanceFromSite'] > 1343)].shape  #(12784, 15)
df[(df['DistanceFromSite'] > 20000) & (df['MOS'] == 4)].shape  #(1365, 15)
#df[(df['DistanceFromSite'] < 35000)].shape  #(12784, 15)


# #for DistanceFromSite
# 
# data missing : 10359
# 
# mean using all data : 7797 meter distance
# 
# data less than outlier 1343 : 12784

# In[ ]:


df[(df['DistanceFromSite'] < 150000) & (df['DistanceFromSite'] > 35000)].shape 


# #########in the next training will go with 35km distance instead of 1343

# In[ ]:


df[(df['DistanceFromSite'] > 1343) & (df['DistanceFromSite'] < 35000)].shape 


# #dataloss will be 10% if we choose 1343
# 
# #we will exclude the records which are more than 35km

# In[ ]:


import seaborn as sns
sns.boxplot(df.CallTestDuration)


# In[ ]:


Q1 = df['CallTestDuration'].quantile(0.25)
Q3 = df['CallTestDuration'].quantile(0.75)
print('Q1 :' + str(Q1))
print('Q3 :' + str(Q3))
IQR = Q3 - Q1
print('IQR :' + str(IQR))
print(Q1 - IQR)
print(Q3 + IQR)


# In[ ]:


df['CallTestDuration'].min()


# In[ ]:


df[(df['CallTestDuration'] > 120 )].shape


# In[ ]:


df[(df['CallTestDuration'] < 30 )].shape


# In[ ]:


df.describe()


# In[ ]:


##removing outliers from DistanceFromSite
df = df[(df['DistanceFromSite'] < 35000) | df['DistanceFromSite'].isna()]


# In[ ]:


print(df.shape)
df['DistanceFromSite'].mean()


# In[ ]:


df["DistanceFromSite"].fillna(df['DistanceFromSite'].mean(), inplace = True) 


# In[ ]:


#find null values

for _ in df.columns:
    print("The number of null values in:{} == {}".format(_, df[_].isnull().sum()))


# In[ ]:


df.MOS.value_counts()


# # Model Training

# In[ ]:


x=df.drop("MOS",axis=1)   #  feature 
y=df["MOS"]    # Target


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)


# In[ ]:


#regressiom algorithm give RMSE value 0.8688302922677146 for RandomForestRegressor
#from sklearn.ensemble import RandomForestRegressor
#rfrmodel = RandomForestRegressor(n_estimators=10000, random_state=101)
#rfrmodel.fit(x_train,y_train)
#rfrmodel_pred= rfrmodel.predict(x_test)
#rfrmodel_rmse=np.sqrt(mean_squared_error(rfrmodel_pred, y_test))
#print("RMSE value for Random forest regression is ",rfrmodel_rmse)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import multilabel_confusion_matrix
model = LogisticRegression()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))
print(classification_report(y_test,predictedvalues))
#print('multilabel CM')  #new version of sklearn - scikit-learn
#print(multilabel_confusion_matrix(y_true, y_pred))


# ## from the above metrics LogisticRegression Classifier gives accuracy which is 0.6699 % , lets try fine tuning and other algorithms

# In[ ]:


d = [0.01, 0.05, 0.25, 0.5, 1]
val_acc = []
for c in d:
    
    lr = LogisticRegression(C=c)
    #pipe = Pipeline([('cnt',X),('LR',lr)])
    lr.fit(x_train, y_train)
    acc= accuracy_score(y_test, lr.predict(x_test))
    print ("Accuracy for C=%s: %s" 
           % (c, acc))
    val_acc.append(acc)


# #accuracy is higher for c = 1

# for 2 classes RFClassifier gave 87% accuracy in other notebook

# In[ ]:


#Lets try random forest classifier.
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))
print(classification_report(y_test,predictedvalues))


# In[ ]:


model = AdaBoostClassifier()
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))
print(classification_report(y_test,predictedvalues))


# In[ ]:


gbclassfier = GradientBoostingClassifier(n_estimators=100)
gbclassfier.fit(x_train, y_train)
predictedvalues=gbclassfier.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))
print(classification_report(y_test,predictedvalues))


# In[ ]:


from xgboost import XGBClassifier 
clf = XGBClassifier() 
clf.fit(x_train, y_train) 
y_pred2 = clf.predict(x_test) 
print(accuracy_score(y_test,y_pred2))


# In[ ]:


from xgboost import XGBClassifier 
clf = XGBClassifier(learning_rate=0.01,n_estimators=899,max_depth=15,min_child_weight=1,gamma=0.3,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',nthread=4,scale_pos_weight=1,seed=3217) 
clf.fit(x_train, y_train) 
y_pred = clf.predict(x_test) 
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:


#Lets try random forest classifier.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import ZeroCount
model = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.2, min_samples_leaf=5, min_samples_split=20, n_estimators=100)
model.fit(x_train, y_train)
predictedvalues=model.predict(x_test)
print(accuracy_score(y_test,predictedvalues))
print(confusion_matrix(y_test, predictedvalues))
print(classification_report(y_test,predictedvalues))


# # RandomForestClassifier with the above hyperparameter is giving 72.26 %  accuracy as compare to other model RFC is best classifier

# ### Lets serialize this classifier to deploy on heroku 

# In[ ]:


import pickle
filename = 'MosClassifier'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:


#feature scaling did not improve the accuracy
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(x_train)
#X_test = sc.transform(x_test)


# In[ ]:


print(df[10:11:])


# ## lets check our serialized model for prediction

# In[ ]:


rfclassifier = pickle.load(open(filename, 'rb'))


# In[ ]:


predictedvalues=rfclassifier.predict([[-105.0,0.82,256.07,60.0,1,3.26,0,0,1,23,37,1,0,0,0,1,0,0,0,0]])
predictedvalues[0]


# ##Our model has predicted correctly

# In[ ]:





# ##output of AUTOML from another notebook
# Generation 1 - Current best internal CV score: 0.7184866084873001
# Generation 2 - Current best internal CV score: 0.7198832011843657
# Generation 3 - Current best internal CV score: 0.7198832011843657
# Generation 4 - Current best internal CV score: 0.7198832011843657
# Generation 5 - Current best internal CV score: 0.7205685337488389
# 
# Best pipeline: RandomForestClassifier(ZeroCount(MinMaxScaler(SelectPercentile(input_matrix, percentile=97))), bootstrap=True, criterion=entropy, max_features=0.2, min_samples_leaf=5, min_samples_split=20, n_estimators=100)
# 0.7240282411358523

# ### Model deployed on : https://mosclassifier.herokuapp.com/

# In[ ]:




