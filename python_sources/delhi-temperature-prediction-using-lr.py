# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#Lib Import 

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Data set reading process
# current delhiweather dataset is have morthen 10 years data.so we need to extract only 2017 data to Delhi.csv.
# Then load it uing pandas
data=pd.read_csv('../input/delhi-weather-dataset-2017/Delhi.csv')

data.columns

data['datetime_utc']=pd.to_datetime(data['datetime_utc'])
#data.set_index('datetime_utc',inplace=True)

data.drop('datetime_utc', axis = 1, inplace=True)
columns_name=list(data.columns)
#columns_name[0].replace("_","").strip()
new_columns=list(i.replace("_","").strip() for i in columns_name)

data.columns=new_columns


# Analyse the dataset
                

data.head()

data.info()


#sns.catplot(x='_conds',y='_tempm',data=data)

sns.heatmap(data.corr())

data.isnull().sum()


# Remove Null values from dataset.
              
              

# here heatindexm and precipm and wgustm and windchillm clomns is fully having maxinum null point 
# so we drop this columns fron dataset.

data.drop(['heatindexm','precipm','wgustm','windchillm'],axis=1,inplace=True)


data.describe()
data.describe().plot()

# here tornado,snow, hail columns mean value is zero so we drop that columns.

data.drop(['tornado','snow', 'hail'],axis=1,inplace=True)

sns.distplot(data['fog'])

data.isnull().sum()

# so here wdird and wdire is having more null points so we drop both columns

# data = data[np.isfinite(data['wdird'])]

# OR

data.dropna(subset=['wdird', 'wdire'],inplace=True)

data.isnull().sum()  # now most of the null rows are removed 

# so conds coluns is fill uing before day conds data

data['conds'].fillna(method='ffill',inplace=True)  # we fill null with front fill method

data.isnull().sum()  # Now fill the all null values.


# Feature Selection 


data.dtypes # here find our dataset data type for all columns

# conds and wdire only is categorical data otherwise is numerical data.

# The wind direction and thunder not need to find our temperature prediction so drop that.

data.drop(['wdire','thunder'],axis=1,inplace=True) 


#   Data visualization

sns.pairplot(data)

sns.distplot(data['tempm'])
sns.distplot(data['pressurem'])

sns.jointplot(x='tempm',y='hum',data=data)

# this plot is represent the low humidity  is get high temperature .
# it is linearly increase .This is the main factor of our project

sns.boxplot(x='tempm',y='hum',data=data,palette='rainbow')

# the box plot show exactly

# Data pre processing 
     
# In our dataset is having only one categorical data otherwise is numerical.
# so categorical data in convert to numerical data 
     
data['conds'].unique() # got more unique result 

data['conds'].nunique() # number of unique # totally 23 unique data 

data['conds'].value_counts()


# categorical value convert into numeric is having 3 methods.

# method 1 is manual chaning processing.

# method 2 is lable encoding 

## without sklearn
data["conds"] = data["conds"].astype('category')
#
## Then you can assign the encoded variable to a new column using the cat.codes
#
data["conds"]= data["conds"].cat.codes

# split features and lables  


data.describe()

X=data.drop(['tempm'],axis=1)

Y=data['tempm']

# split training and testing data


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# Build LinearRegression model

li_model=LinearRegression()


# Fit to model using training data


li_model.fit(x_train,y_train)

# predict the model using test data

prediction=li_model.predict(x_test)

# Validate the model using testing data


accuary=li_model.score(x_test,y_test)

# Plot the results


plt.scatter(x=x_train['hum'],y=y_train[:],color='r')
#plt.plot(x=x_test['hum'],y=prediction[:],color='g')
plt.scatter(x=x_test['hum'],y=prediction[:],color='g')
plt.show()

sns.distplot((y_test-prediction),bins=50)

# Find the Error value 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))