#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import all libraries
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt
from matplotlib import style
#sta matplotlib to inline and displays graphs below the corresponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from sklearn.datasets import *
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv("../input/Automobile.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


#data cleaning
df.isnull().sum()


# In[ ]:


df.describe().T


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


cols=['symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
       'number_of_doors', 'body_style', 'drive_wheels', 'engine_location',
       'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type',
       'number_of_cylinders', 'engine_size', 'fuel_system', 'bore', 'stroke',
       'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',
       'highway_mpg', 'price']
for i in cols:
    def check(data):
        t=data[i].loc[data[i]=='?']
        return t
    

    g=check(df)
    print(g)


# In[ ]:


#thus the data shows there is no null values and also no special characters in the place of values


# In[ ]:


#what are the columns which are object
obj=list(df.select_dtypes(include=['object']))
obj


# In[ ]:


#what are the columns which are float and int
flint=list(df.select_dtypes(include=['int64','float64']))
flint


# In[ ]:


#checking for outliers
plt.figure(figsize=(15,8))
sns.boxplot(data=df)


# In[ ]:


df[['engine_size','peak_rpm','curb_weight','horsepower','price']].hist(figsize=(10,8),bins=6,color='Y')
plt.tight_layout()
plt.show()

Inferences:
    .Most of the car has a Curb Weight is in range 1900 to 3100
    .The Engine Size is inrange 60 to 190
    .Most vehicle has horsepower 50 to 125
    .Most Vehicle are in price range 5000 to 18000
    .peak rpm is mostly distributed between 4600 to 5700
# In[ ]:


print('the minimum price of car: %0.2d, the maximum price of the car: %0.2d'%(df['price'].min(),df['price'].max()))


# In[ ]:


df['make'][df['price']>=30000].count()


# In[ ]:


#there are 14 cars like that which are highly priced due to its features


# In[ ]:


d=df['make'][df['price']>=30000].value_counts().count()
print(d)
df['make'][df['price']>=30000].value_counts()


# In[ ]:


df.head()


# In[ ]:


df.aspiration.value_counts()


# In[ ]:


fig,a=plt.subplots(1,2,figsize=(10,5))
df.groupby('aspiration')['price'].agg(['mean','median','max']).plot.bar(rot=0,ax=a[0])
df.aspiration.value_counts().plot.bar(rot=0,ax=a[1])


# #There are some cars which are std aspiration are higher in price rather than Turbo but people are purcahsing more
# std aspiration cars only due to the price, as we can see that the average price is less than 10000 dollars. So many
# of the people pretend to buy the std cars.

# In[ ]:


plt.figure(1)
plt.subplot(221)
df['engine_type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')
plt.title("Number of Engine Type frequency diagram")
plt.ylabel('Number of Engine Type')
plt.xlabel('engine-type');

plt.subplot(222)
df['body_style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange')
plt.title("Number of Body Style frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('body-style');

plt.subplot(223)
df['number_of_doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')
plt.title("Number of Door frequency diagram")
plt.ylabel('Number of Doors')
plt.xlabel('num-of-doors');

plt.subplot(224)
df['fuel_type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='purple')
plt.title("Number of Fuel Type frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('fuel-type');


plt.tight_layout()
plt.subplots_adjust(wspace=0.3,hspace=0.5)
plt.show()

.About 70% of cars are manufactured with Over head Camshaft which can be assumed as the mordern cars are using OHC
.About 40% of sales are going on Sedan cars following by Hatchback,Wagon
.The cars  got sold were high in number which are having four doors rather than two doors.
.About 80% of cars have Gas fuel system .
# In[ ]:


fig,a=plt.subplots(1,2,figsize=(10,2))
df.body_style.value_counts().plot.pie(explode=(0.03,0,0,0,0),autopct='%0.2f%%',figsize=(10,5),ax=a[0])
a[0].set_title('No. of cars sold')


df.groupby('body_style')['price'].agg(['mean','median','max']).sort_values(by='median',ascending=False).plot.bar(ax=a[1])
a[1].set_title('Price of each body_style')
plt.tight_layout()
plt.show()

Inference:
    . 46.7% of the cars were sold out are sedan model whose price are between 10000 to 15000 dollars following 33.8% of
        cars were hatchback.
    . The least no.of cars were sold out is the Convertible body type as its average price is somewhat higher as well
       as its costlier
# In[ ]:


sns.catplot(data=df, y="normalized_losses", x="symboling"  ,kind="point")

Inference:
    . Here +3 means risky vehicle and -2 means safe vehicle
    . Increased in risk rating linearly increases in normalised losses in vehicle
# In[ ]:


df.head()


# In[ ]:


sns.lmplot('engine_size','highway_mpg',hue='make',data=df,fit_reg=False)
plt.title('Engine_size Vs highway_mpg')
plt.show()

inference:
    1. As the engine size increases the highway_mpg decreases, most of the volvo company cars has higher highway_mpg
       because of its lesser engine size
    2. The cars from jaguar are having less highway_mpg as its having heavy weight of engine
# In[ ]:


sns.lmplot('engine_size','city_mpg',hue='make',data=df,fit_reg=False)
plt.title('Engine_size Vs city_mpg')

inference:
    1. As the engine size increases the highway_mpg decreases, most of the volvo company cars has higher city_mpg
       because of its lesser engine size
    2. The cars from jaguar are having less highway_mpg as its having heavy weight of engine
# In[ ]:


df[['make','fuel_type','aspiration','number_of_doors','body_style','drive_wheels','engine_location']][df['engine_size']>=300]


# In[ ]:


# df.drive_wheels.value_counts().plot.bar()
fig,ax=plt.subplots(2,1,figsize=(15,5))
sns.countplot(x='drive_wheels',data=df,ax=ax[0])
df.groupby(['drive_wheels','make'])['price'].mean().plot.bar(ax=ax[1])
plt.grid()
plt.show()

Inference:
    1.Most of the fwd vehicles sales are high followed by rwd and 4wd,the cars which are having forward weel drive
    having a price less than the rear-Wheel drive.
# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(15,5))
sns.countplot(x='drive_wheels',data=df,ax=ax[0])
df.groupby(['drive_wheels','body_style'])['price'].mean().plot.bar(ax=ax[1])
ax[1].set_ylabel('Price')
plt.grid()
plt.show()


# As per before inferences, the people who purchased 46.77% sedan bodystyles cars which are having forward wheel drive(fwd)
# whose cost range between 10000 to 11000 dollars followed by 33.83% of hatchback and 12.44% of wagon body styles are purchased
# 

# In[ ]:


dff=pd.pivot_table(df,index=['body_style'],columns=['drive_wheels'],values=['engine_size'],
                   aggfunc=['mean'],fill_value=0)

dff.plot.bar(figsize=(15,5),rot=45)
plt.show()
dff


# #from the pivot table and plot we can observe that the bodystyle of hardtop's engine size with 'rwd' is very much
# greater also observed that [Convertible,Hardtop] body style's doesnot contain any '4wd'.

# In[ ]:


sns.kdeplot(df['price'],shade=True)


# In[ ]:


df.price.max()


# In[ ]:


df.groupby('drive_wheels')[['city_mpg','highway_mpg']].agg('sum').plot.bar()
df.groupby('drive_wheels')[['city_mpg','highway_mpg']].agg('sum')


# The cars which are having forward wheel drive(fwd) having high milage followed by Rear wheel drive(rwd),4wd.

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)

From above heat map we can observe the positive and negative correaltions of each and every variable
.Especially the PRICE of a car is based on the [horsepower,engine_size,curb_weight,width,length]
.The milage is based of a car is based on the [horsepower,engine_size,curb_weight,width,length,]
.The [bore,stroke,compression ratio] have no that much correlated to other variables
# In[ ]:


sns.pairplot(df,aspect=1.5)


# From the above pairplot we can observe that which Variables are postively, negatively, and No correlations at all.

# In[ ]:


sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg','horsepower','engine_size','curb_weight'],y_vars=['price'],kind='reg',size=2.8)

Inference:
    . the city_mpg,highway_mpg are negatively correlated to price
    . the horse_power,engine_size,curb_weight are positively correlated to price
    . It means that ciy
# In[ ]:


sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg'],y_vars=['horsepower'],hue='price',size=4)
plt.savefig('pairplot.png')


# In[ ]:


sns.pairplot(data=df,x_vars=['city_mpg','highway_mpg'],y_vars=['horsepower'],kind='reg',size=4)

Inference:
    . the city_mpg and highway_mpg are negatively correlated to horsepower and price also
    . if the horse power increases the milage will decrease vice-versa.
# ### BUILDING A MODEL

# Since it can be clearly shown from pair plot that the label(PRICE) is linearly proportional to some of the explanatory
# variables. For such instances it good to go with LinearRegression model 

# #### Feature Engineering

# In[ ]:


df=pd.read_csv("../input/Automobile.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


cols=list(df.select_dtypes(include=['object']))
cols


# In[ ]:


for i in cols:
    print(i)
#     print()
    print(df[i].value_counts())
    print()


# In[ ]:


#  We dont have any further use of ["make","body_style"] to build our model, so drop ['make','body_style'] columns

df.drop(['make','body_style'],1,inplace=True)


# In[ ]:


df.columns


# In[ ]:


df.describe(include='all').T


# In[ ]:





# In[ ]:


#For those columns which are having 2 samples lets do label encoding and those columns which are having more than 
#2 samples lets do get_dummies for that
labcols=['engine_location','number_of_doors','fuel_type','aspiration']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in labcols:
#     print(i)
    df[i]=le.fit_transform(df[i])
# df.head()


# In[ ]:


#get dummies
dumcols=['drive_wheels','engine_type','fuel_system','number_of_cylinders']
df=pd.get_dummies(df,columns=dumcols,drop_first=True)


# In[ ]:


df.head()


# In[ ]:


y=df['price']
X=df.drop('price',1)


# In[ ]:


X.shape[1]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(X_train,y_train)


# In[ ]:


y_pred=lr.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error
print('r2_score:',r2_score(y_test,y_pred))
# print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[ ]:


y_train_pred=lr.predict(X_train)
print('train_score:',r2_score(y_train,y_train_pred))
print('test_score:',r2_score(y_test,y_pred))


# #we can observe that there is slight difference between train and test scores. As the train_score is high , the the model is overfit,it consists of Variance Error rather than Bias Error

# In[ ]:


import statsmodels.api as sm
Xc=sm.add_constant(X_train)
model=sm.OLS(y_train,Xc).fit()
result=model.summary()
result


# In[ ]:


r=df.columns
r.shape


# #there are 40 columns but as per Stats the columns whose PValue is >0.05 will makes the null Hypothesis to be True.
# The Null Hypothesis for this is the mean for all columns are equal. Practically it can't be happen. So, Lets drop the cols whose PValue is greater than significant level

# In[ ]:


# Backward Elimination
cols=list(X_train.columns)
while(len(cols)>0):
    p=[]
    X=X_train[cols]
    Xc=sm.add_constant(X)
    model=sm.OLS(y_train,Xc).fit()
    p=pd.Series(model.pvalues.values[1:],index=cols)
    pmax=max(p)
    feature_with_p_max=p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE=cols
print(selected_features_BE) 
print()
print('count_selected_colums: ',len(cols))
    
    
    


# In[ ]:


### if the columns are more then there will be chance of increment in VARIACNE ERROR,so thats why we sorted columns based on PValues


# In[ ]:


#from 40 colums it reduced to 17 columns,continue with the model building
X_final=df[selected_features_BE]
y=df['price']


# In[ ]:


print(X_final.shape,y.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(X_final,y)


# In[ ]:


lr.coef_


# In[ ]:


pd.DataFrame(lr.coef_,index=X_final.columns,columns=['coefficents'])


# In[ ]:


print('beta0:',lr.intercept_)


# In[ ]:


from sklearn.model_selection import KFold,cross_val_score
kfold=KFold(n_splits=3, shuffle=True, random_state=1)
cv_score=cross_val_score(lr,X_final,y,cv=kfold)
score=np.mean(cv_score)
# cv=cross_val_score(lr,X_final,y,scoring='neg_mean_squared_error',cv=kfold)
# rm=(-1)*(cv)
print('the r2_score of the model:',score)
# rmse=(np.sqrt(rm))
# print("rmse:%0.02f,variance:%0.02f" % (np.mean(rmse),np.var(rmse,ddof=1)))


# In[ ]:


print('rsquared value:',model.rsquared)


# ### THE END

# In[ ]:





# In[ ]:


import pandas as pd
Automobile = pd.read_csv("../input/Automobile.csv")

