#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns',None)

from sklearn.model_selection import train_test_split,cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as r
from scipy import stats


# In[ ]:


#loading the data set 
df=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
df.head()


# # Initial Intuition on features
# 
# Symboling:
#     As symboling is the process which is assosciated with the price(which mean symboling will be done once the Price is fixed).
#     In any model building, it is not a good practice to include the features which are derived after the predicted 
#     feature(which here is price). 
#         So we can directly drop this feature.
# 
# CarName:Car price mostly depends on the brand of the car. We ca extract the maker of the car and divide it into low,medium
# and high bidget cars .
# 
# Car_length: In US the width of the car plays an imp role as regulations are set by Govt on it. Where as in India its on car length.
# 
# WheelBase : Wheelbase depends on car length . Price of the car will be decided based on car length(considering wheel base vs length)
#             So, we can drop this feature.
#              
# Engine size(Engine capacity): Its the feature which is calculated from bore , stroke, number of cylinders. So, we can drop 
#     those three features.
#                 
#                 Engine size= (3.1416*bore^2*stroke*no. of cylinders)/4  cubic inches

# In[ ]:


#before exploring the further, lets check if there are any null values in the data.

df.isnull().sum()[df.columns[df.isnull().sum()>0]]


# In[ ]:


#visualizing nulls
sns.heatmap(df.isnull(),cbar=False,yticklabels=False)
plt.show()
#no nulls in the dataset


# In[ ]:


#removing car id and symboling, Car name
df.drop(['car_ID','symboling'],1,inplace=True)


# In[ ]:


#We will proceed with analysing numerical and categorical columns saperately

df_numeric=df.select_dtypes(include=['int64','float64'])
df_catg=df.select_dtypes(include=object)


# # Exploratory Data Analysis and Feature Engineering

# # Numerical Featurs Analysis

# In[ ]:


df_numeric.head()


# In[ ]:


#1:Deriving Avg mpg based on city and highway mpg 
def Avg_mpg(x):
    city=x[0]
    highway=x[1]
    return ( (city+highway)/2)


# In[ ]:


Avgmpg=df_numeric[['citympg','highwaympg']].apply(Avg_mpg,axis=1)
df_numeric.insert(len(df_numeric.columns)-1,'Avgmpg',Avgmpg)


# In[ ]:


#dropping the features
df_numeric.drop(['citympg','highwaympg'],1,inplace=True)


# In[ ]:


df_numeric.describe().T


# # Observations:
# -> We can see that horse power, curb weight,price,engne size are the features which are not normally distributed indicating that these have outliers.
# 
# ->All the features are in different units.Have to perform scaling
# 

# In[ ]:


#viewing the pair plot `
sns.pairplot(df_numeric,diag_kind='kde')
plt.show()


# In[ ]:


#heatmap
plt.figure(figsize=(20,10))
sns.heatmap(df_numeric.corr(),annot=True)
plt.show()

Inference
->Wheelbase , Car Height, stroke, compression ratio, peak rpm show least-to-no relation with price. Hence we can omit it.

->Wheelbase is strongly correlated with length as already said . So , we can drop it.    
->We can also obeseve that the other features are not linearly dependent on price.We may have perform transformation.

->Car length and car width are positively correlated , which is obvious, to maintain the aerodynamics of car.
(Not considering Height as there are cars which are of good length and width but with low height(eg: Ferrari,Lamborghini)).

-> Curb Weight shows positive correlation with both length and width.This is obvious that as length and width increases
the quantity of materials used will be increased resulting in increase in weight.

->Engine size and Curb Weight shows a strong positive correlation . We can consider having engine size is the model building     as, weight is of a car depends on engine size(the  capacity of the engine) , type of materials used etc

->Horse power and engine size are positively correlated.Which is as expected as the number of cylinders increase engines size also increase resulting in high horse power.Both individually are also equally correlated with price.

->Mileage per gallon shows a string negative correlation with price.This is expected as higher the price, low the mileage
# In[ ]:


#removing the above said columns .
cols_to_remove=['wheelbase','carheight','stroke','compressionratio','peakrpm']
df_numeric.drop(cols_to_remove,1,inplace=True)


# # Dealing with Categorical Features

# In[ ]:


df_catg=pd.concat((df_catg,df.iloc[:,-1]),1)
df_catg.head()


# In[ ]:


#Extracting Car Maker
Car_Maker=df_catg['CarName'].apply(lambda x : x.split()[0])
df_catg.insert(1,'Car_Maker',Car_Maker)
df_catg.drop('CarName',1,inplace=True)


# In[ ]:


df_catg.Car_Maker.unique()


# In[ ]:


#We have the same Makers names repeated, lets combine them

df_catg['Car_Maker']=df_catg['Car_Maker'].replace('maxda','mazda')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('Nissan','nissan')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('Nissan','nissan')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('porcshce','porsche')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('toyouta','toyota')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('vokswagen','volkswagen')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('vw','volkswagen')

df_catg['Car_Maker']=df_catg['Car_Maker'].apply(lambda x : x.capitalize())


# In[ ]:


plt.figure(figsize=(10,5))
df_catg['Car_Maker'].value_counts().plot(kind='bar')
plt.xlabel('Car Brands')
plt.ylabel('Number of Units in Market')
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
sns.barplot(df_catg['Car_Maker'],df_catg['price'],ci=None,estimator=np.min)
plt.xlabel('Car Brands')
plt.ylabel('Avg Price')
plt.show()

Observation
-> Toyota , Mercury are the most and the least preferred car brands by the people of America.
->The Avg Pricing of Jaguar and Buick are high compared to others.
->As already said, dividing it into bins as low , medium , high budget cars based on min price.
   (As the segment of the car brand can be decided based on the entry level car price,here we are diriving it by the min price)
# In[ ]:


def Car_Class(x):
    price=x
    if price>0 and price<10000:
        return 'Low Budget'
    elif price>=10000 and price<20000:
        return 'Medium Budget'
    else:
        return 'High Bidget'
    


# In[ ]:


Car_class=df_catg['price'].apply(Car_Class)
df_catg.insert(len(df_catg.columns)-1,'Car_Class',Car_class)


# In[ ]:


df_catg.drop('Car_Maker',1,inplace=True)


# In[ ]:


d={'Low Budget':0,'Medium Budget':1,'High Bidget':2}

df_catg['Car_Class']=df_catg['Car_Class'].map(d)


# In[ ]:



plt.figure(figsize=(20,20))
plt.subplot(521)
df_catg['fueltype'].value_counts().plot('bar')
plt.xlabel('Type of Fuel')
plt.ylabel('Number of Units')

plt.subplot(522)
sns.boxplot(df_catg['fueltype'],df_catg['price'])

#####################################################

plt.figure(figsize=(20,20))
plt.subplot(525)
df_catg['aspiration'].value_counts().plot('bar')
plt.xlabel('Type of Aspiration')
plt.ylabel('Number of Units')

plt.subplot(526)
sns.boxplot(df_catg['aspiration'],df_catg['price'])

######################################################

plt.figure(figsize=(20,20))
plt.subplot(529)
df_catg['doornumber'].value_counts().plot('bar')
plt.xlabel('Number of Doors')
plt.ylabel('Number of Units')

plt.subplot(5,2,10)
sns.boxplot(df_catg['doornumber'],df_catg['price'])
plt.show()

Observations:
-> Most of the Americans prefer vehicles which run on Gas as the gas price is low in US due to less excise tax rate.
->Avg Price of Diesel Cars are more than Gas Cars,as it requires few extra engine kits(parts) to run the car efficiently.
-> The median prices for gas and diesel are around 10000 and 13500(which are approximately near). This feature may not be significant in determining price. (Significance will be determined later)


-> Most of the vehicles which Americans buy have naturally aspirated engine(std aspiration)
->The price of the cars with naturally aspirated engines are low compared to turbocharged engines as naturally aspirated engines are those which brings the air into it naturally(atmospheric pressure). Where as for turbocharged engine we have to provide additional kit for the air intake which naturally increases the cost.
-> More over , turbo charged engines are the ones with high horse powers. Lets check it out :)

->As most of the family members will be leaving the country they will prefer a four door car.
->The median values of two types of cars with respect to price are equal, which mean door number may not be significant
in predicting price.
# In[ ]:


plt.figure(figsize=(20,20))
plt.subplot(321)
sns.boxplot(df_catg['fueltype'],df_numeric['horsepower'])

plt.subplot(322)
sns.boxplot(df_catg['fueltype'],df['compressionratio'])

##########################################################

plt.figure(figsize=(20,20))
plt.subplot(325)
sns.boxplot(x=df_catg['aspiration'],y=df_numeric['horsepower'])

plt.subplot(326)
sns.boxplot(df_catg['aspiration'],df_numeric['Avgmpg'])
plt.show()

Observations:
->Horse power of diesel cars are low compared to gasoline cars.
->Diesel cars have high compression ratio. 
    This is because diesel cars do not have spark plug and the cylinder has to travel long distance to ignite the mixture of       air and fuel.
->As turbo charged engines mainly aimedd for racing cars, its fuel efficency naturally will be low.
# In[ ]:


plt.figure(figsize=(20,20))
plt.subplot(521)
df_catg['carbody'].value_counts().plot('bar')
plt.xlabel('Type of Car')
plt.ylabel('Number of Units')

plt.subplot(522)
sns.boxplot(df_catg['carbody'],df_catg['price'])
##################################################

plt.figure(figsize=(20,20))
plt.subplot(525)
df_catg['drivewheel'].value_counts().plot('bar')
plt.xlabel('Type of Drive')
plt.ylabel('Number of Units')

plt.subplot(526)
sns.boxplot(df_catg['drivewheel'],df_catg['price'],hue=df_catg['enginelocation'])
###################################################

plt.figure(figsize=(20,20))
plt.subplot(529)
df_catg['enginelocation'].value_counts().plot('bar')
plt.xlabel('Location of Engine')
plt.ylabel('Number of Units')

plt.subplot(5,2,10)
sns.boxplot(hue=df_catg['drivewheel'],y=df_numeric['Avgmpg'],x=df_catg['enginelocation'])
plt.show()

Observation:
-> It will be benificial to the company if it enters into US by first launching sedans and hatchbacks into market as they are 
most preferred by people.
->The median values of convertible and hardtop cars are high when compared to others, this is obvious as these type of 
cars have the roof of folding type which includes extra mechanism resulting in increase of price.
->Convertible and hardtop cars are those which will have high horse power than others.Lets check :).

-> Most of the cars which people buy are of front wheel drive with fron engine location as the cost of these cars are low compared to others.This is because fron engine and fwd do not require any additional differential to transmit the power.

->Its obvious the the cars with engine at rear are costly as they are mainly meant for racing.
->And naturally, fuel economy of front engine with fwd is will be more when compared to other combinations as the distance
of power availabilty to the wheels is less. Vizualization is below.

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(131)
sns.boxplot(hue=df_catg['drivewheel'],y=df_numeric['enginesize'],x=df_catg['enginelocation'])

plt.subplot(133)
sns.boxplot(df_catg['aspiration'],df_numeric['enginesize'])
plt.show()
##################################################################

plt.figure(figsize=(14,7))
sns.boxplot(df_catg['cylindernumber'],df_numeric['enginesize'],order=['two','three','four','five','six','eight','twelve'])
plt.show()

Observations:
-> As the number of cylinders increase, engine becomes more powerful(which means it is likely to be a sports car).And for these 
types of cars, engine will be at rear.
# In[ ]:


plt.figure(figsize=(20,20))
plt.subplot(321)
sns.boxplot(df_catg['carbody'],df['horsepower'])

plt.subplot(322)
sns.boxplot(df_catg['carbody'],df_numeric['Avgmpg'])
#########################################################

plt.figure(figsize=(20,20))
plt.subplot(325)
sns.boxplot(df_catg['cylindernumber'],df_numeric['Avgmpg'])

plt.subplot(326)
sns.boxplot(df_catg['drivewheel'],df_numeric['Avgmpg'])

plt.show()

Observatios:
->As the number of cyliinders increases , the fuel economy if cars decreases.
->Rear wheel drive cars give less fuel economy.

Thus we can say that convertible and hardtop are the type of cars with more number of cylinders and are either 4wd or rwd.
# In[ ]:


plt.figure(figsize=(20,25))
plt.subplot(521)
df_catg['enginetype'].value_counts().plot('bar')
plt.xlabel('Type of Engine')
plt.ylabel('Number of Units')

plt.subplot(522)
sns.boxplot(df_catg['enginetype'],df_catg['price'],hue=df_catg['enginelocation'])
##################################################################################

plt.figure(figsize=(20,25))
plt.subplot(525)
df_catg['fuelsystem'].value_counts().plot('bar')
plt.xlabel('Type of fuel injection')
plt.ylabel('Number of Units')

plt.subplot(526)
sns.boxplot(df_catg['fuelsystem'],df_numeric['price'])
plt.show()

observation:
-> Most of the cars availabe in US market are of ohc engine type and people also buy it as it has low cost
->Looks like ohcf is the engine type used in rear wheel drive cars(aka expensive cars)

MPFI:Multi point fuel injection
->mpfi is the most preferred fuel injection system by the makers of the car. This obviously increase the cost of the car 
as it requires multiple injection plugs to enter into the combustion chamber which increase the cost eventually.
# # Model Building : Lineear Regression 
# 

# In[ ]:


df_numeric.head()


# In[ ]:


#removig as already said above
df_numeric.drop('boreratio',1,inplace=True)
df_numeric.head()


# In[ ]:


#removing price column from catg data frame as already present in numeric one
df_catg=df_catg[df_catg.columns[:-1]]


# In[ ]:


#removig as already said above
df_catg=df_catg.drop(['cylindernumber','doornumber'],1)


# In[ ]:


cat_cols=df_catg.columns[:-1]


# In[ ]:


#renaming the data points in categorical feature with count <20 to other 
for col in cat_cols:
    a=df_catg[col].value_counts()
    for  j,i in enumerate(a):
        if i<20:
            name=a.index[j]
            df_catg=df_catg.replace(name,'other')


# In[ ]:


for i in df_catg.columns:
    print(df_catg[i].value_counts())


# In[ ]:


#removing the other data points whose counts is <20

for col in cat_cols:
    count = df_catg[col].value_counts()
    k = count.index[count>20][:-1]
    
    for cat in k:
        name = col + ' ' + cat
        df_catg[name] = (df_catg[col] == cat).astype(int)
    del df_catg[col]


# In[ ]:


df=pd.concat((df_numeric,df_catg),1)
df.head()


# In[ ]:


df=df.transform(lambda x : np.log1p(x))


# In[ ]:


#target feature is approx normal 
sns.distplot(df['price'])


# In[ ]:


X=df.drop('price',1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=45)


# In[ ]:


print(X.shape)


# In[ ]:


#as there are 14 columns , lets select around 5 featres 

lr=LinearRegression()
rfe=RFE(lr,5)
rfe=rfe.fit(X_train,y_train)


# In[ ]:


X_train_new=X_train[X_train.columns[rfe.support_]]
X_test_new=X_test[X_train_new.columns]
X_train_new.head()


# In[ ]:


model_lr=lr.fit(X_train_new,y_train)

print(f'The R square value for the training data is {round(model_lr.score(X_train_new,y_train)*100,2)}')
print(f'The R square value for the testing data is {round(model_lr.score(X_test_new,y_test)*100,2)}')


# In[ ]:


#as the data points are only 200, lets perforn 3 fold CV
X1=X[X_train_new.columns]

print(f'The mean R square value we get by this model is {round(cross_val_score(lr,X1,y,cv=3).mean()*100,2)}')


# In[ ]:


pd.DataFrame(zip(X_train_new.columns,model_lr.coef_),columns=['Col_name','Coeff_value'])

We can see that carwidth is the feature which has max impact on pricing in USA.
# In[ ]:


X_const=sm.add_constant(X_train_new)
model_ols=sm.OLS(y_train,X_const).fit()
model_ols.summary()

We can see that none of the p value is greater than 0.05 which means these features are helpful in determining price.
Also, DW score is 2.27 which mean there is no auto correlation within the target variable
P value of JB test is greater than 0.05 which mean the residuals are normally distributed.
# In[ ]:


#cheking for multicollinearity

v=[VIF(X_const.values,i) for i in range(X_const.shape[1])]

pd.DataFrame(zip(X_const,v),columns=['col_name','vif_value'])

As all the values are <5 : there is no multicollinearity withing the selected features.
# In[ ]:


sns.distplot(model_ols.resid)
plt.show()

The residuals are approx normally distributed
# # Random Forest Regression

# In[ ]:


rf=RandomForestRegressor()
params={'max_depth':r(1,25), 'min_samples_split':r(2,20),
       'min_samples_leaf':r(2,15),'max_samples':r(50,75),'max_features':r(5,7),'n_estimators':r(1,50)}

rsearch=RandomizedSearchCV(rf,param_distributions=params,n_jobs=-1,cv=3,return_train_score=True,random_state=45)


# In[ ]:


rsearch.fit(X_train_new,y_train)


# In[ ]:


#the best estimator
print(rsearch.best_estimator_)


# In[ ]:


rsearch.best_params_


# In[ ]:


rfr=RandomForestRegressor(**rsearch.best_params_,random_state=45)


# In[ ]:


model_rfr=rfr.fit(X_train_new,y_train)


# In[ ]:


print(f'The R square value for the training data is {round(model_rfr.score(X_train_new,y_train)*100,2)}')
print(f'The R square value for the testing data is {round(model_rfr.score(X_test_new,y_test)*100,2)}')


# In[ ]:


X2=X[X_train_new.columns]
print(f'The mean R square value we get by this model is {round(cross_val_score(rfr,X2,y,cv=3).mean()*100,2)}')


# In[ ]:


#checking the normality of residuals
resid=y_train-model_rfr.predict(X_train_new)

print(f'P value for normality check is {stats.jarque_bera(resid)[1]}')
#Residuals are nomrmally disributed


# # Gradient Boost Regression 

# In[ ]:


gbr=GradientBoostingRegressor()


# In[ ]:


params_gb={'n_estimators':r(10,50),'min_samples_split':r(2,50),'min_samples_leaf':r(2,50),'max_depth':r(3,50)}


# In[ ]:


rsearch_gb=RandomizedSearchCV(gbr,param_distributions=params_gb,random_state=45,n_jobs=-1)


# In[ ]:


rsearch_gb.fit(X_train_new,y_train)


# In[ ]:


rsearch_gb.best_params_


# In[ ]:


g=GradientBoostingRegressor(**rsearch_gb.best_params_,random_state=45)


# In[ ]:


model_gb=g.fit(X_train_new,y_train)


# In[ ]:


print(f'The R square value for the training data is {round(model_gb.score(X_train_new,y_train)*100,2)}')
print(f'The R square value for the testing data is {round(model_gb.score(X_test_new,y_test)*100,2)}')
print(f'The mean R square value we get by this model is {round(cross_val_score(g,X2,y,cv=3).mean()*100,2)}')


# In[ ]:


resid=y_train-model_gb.predict(X_train_new)

print(f'P value for normality check is {stats.jarque_bera(resid)[1]}')
#Residuals are nomrmally disributed


# # Making Pipelines

# In[ ]:


from sklearn.pipeline import Pipeline


# In[ ]:


pipeline_lr=Pipeline([('lr_regression',LinearRegression())])
pipeline_rfr=Pipeline([('random_regression',RandomForestRegressor(**rsearch.best_params_,random_state=45))])
pipeline_gbr=Pipeline([('gradient_regression',GradientBoostingRegressor(**rsearch_gb.best_params_,random_state=45))])


# In[ ]:


pipelines = [pipeline_lr, pipeline_rfr, pipeline_gbr]
pipe_dict = {0: 'Linear Regression', 1: 'Random Forest Regression ', 2: 'Gradient Boost Regression'}


# In[ ]:


for model in pipelines:
    model.fit(X_train_new,y_train)


# In[ ]:


for i,model in enumerate(pipelines):
    print("{} Mean Accuracy: {}".format(pipe_dict[i],round((cross_val_score(model,X2,y,cv=5).mean()) *100),2))


# # Thus We can go with Random Forest Regression model 
