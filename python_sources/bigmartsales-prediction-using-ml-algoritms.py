#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[ ]:


Train_data=pd.read_csv("../input/Train_UWu5bXk.csv")
Test_data=pd.read_csv("../input/Test_u94Q5KV.csv")


# In[ ]:


Train_data.head()


# In[ ]:


Test_data.head()


# In[ ]:


print(Train_data.shape)
print(Test_data.shape)


# In[ ]:


Train_data.info()


# In[ ]:


['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type']


# In[ ]:


Train_data.isnull().sum()


# In[ ]:


Data=Train_data.append(Test_data,sort=False)


# In[ ]:


Data.isnull().sum()


# In[ ]:





# In[ ]:


Train_data.describe()


# In[ ]:


Train_data.describe().columns


# In[ ]:


for i in Train_data.describe().columns:
    sns.distplot(Data[i].dropna())
    plt.show()


# In[ ]:


for i in Train_data.describe().columns:
    sns.boxplot(Data[i].dropna())
    plt.show()


# ## Imputing Item Visibltiy values via mean

# In[ ]:


sns.boxplot(Data['Item_Visibility'])


# In[ ]:


Data['Item_Visibility'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


sns.boxplot(y=Data['Item_Weight'],x=Data['Outlet_Identifier'])
plt.xticks(rotation='vertical')


# In[ ]:


Data['Item_Fat_Content'].value_counts()


# In[ ]:


Data['Item_Fat_Content'] = Data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})


# In[ ]:


Data.groupby('Item_Identifier')['Item_Weight'].mean().head(5)


# In[ ]:


for i in Data.groupby('Item_Identifier')['Item_Weight'].mean().index:
    Data.loc[Data.loc[:,'Item_Identifier']==i,'Item_Weight']=Data.groupby('Item_Identifier')['Item_Weight'].mean()[i]


# In[ ]:


Data['Outlet_Type'].value_counts()


# In[ ]:


Data.Outlet_Size[Data['Outlet_Type']=='Grocery Store'].value_counts()


# In[ ]:


Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type1'].value_counts()


# In[ ]:


Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type2'].value_counts()


# In[ ]:


Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type3'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:


#Data['Outlet_Size'].fillna(Data['Outlet_Size'].mode()[0],inplace=True)
Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Grocery Store'].mode()[0],inplace=True)


# In[ ]:


Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type1'].mode()[0],inplace=True)


# In[ ]:


Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type2'].mode()[0],inplace=True)


# In[ ]:


Data.Outlet_Size.fillna(Data.Outlet_Size[Data['Outlet_Type']=='Supermarket Type3'].mode()[0],inplace=True)


# # imputing Item visiblity

# In[ ]:


for i in Data.groupby('Item_Identifier')['Item_Visibility'].mean().index:
    Data.loc[Data.loc[:,'Item_Identifier']==i,'Item_Visibility']=Data.groupby('Item_Identifier')['Item_Visibility'].mean()[i]


# # Dealing with outlet establishment year

# In[ ]:


Data['Outlet_Establishment_Year']=2013-Data['Outlet_Establishment_Year']


# In[ ]:


Data


# In[ ]:





# In[ ]:


Data.isnull().sum()


# In[ ]:


Train_data=Data.dropna()


# In[ ]:


Test_Data=Data[Data['Item_Outlet_Sales'].isnull()]
Test_Data.drop('Item_Outlet_Sales',axis=1,inplace=True)


# In[ ]:


sns.boxplot(Train_data['Item_Visibility'])

# Remove outliers from Item visiblity


# In[ ]:


Train_data['Item_Visibility'].describe()


# In[ ]:





# In[ ]:





# In[ ]:


print(Test_Data.shape)
print(Train_data.shape)


# In[ ]:


len(Train_data)


# In[ ]:


len(Test_data)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


categorical_list=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year']


# In[ ]:


'''le = LabelEncoder()
for i in categorical_list:
    Data[i]=le.fit_transform(Data[i])
    Data[i]=Data[i].astype('category')'''


# In[ ]:


le = LabelEncoder()
for i in categorical_list:
    Train_data[i]=le.fit_transform(Train_data[i])
    Train_data[i]=Train_data[i].astype('category')
    Test_Data[i]=le.fit_transform(Test_Data[i])
    Test_Data[i]=Test_Data[i].astype('category')


# In[ ]:





# In[ ]:


Data


# In[ ]:


Test_Data.head()


# In[ ]:


Train_data.head()


# In[ ]:


Train_data.corr()


# In[ ]:


#from sklearn.model_selection import train_test_split


# In[ ]:


#X_train, x_test, Y_train, y_test = train_test_split(Data.drop(['Item_Outlet_Sales','Item_Identifier'],axis=1), Data['Item_Outlet_Sales'], test_size = 0.3)


# In[ ]:


#Data.Item_Visibility[Data['Item_Visibility']==0].value_counts()


# ## Model 1

# In[ ]:


from sklearn.linear_model import LinearRegression as LR


# In[ ]:


Lm=LR(normalize=True)


# In[ ]:


Lm.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])


# In[ ]:


#Lm.score(x_test,y_test)


# In[ ]:


#pred_train=Lm.predict(X_train)


# In[ ]:


#pred_test=Lm.predict(x_test)


# In[ ]:


#from sklearn import metrics


# In[ ]:


#metrics.mean_squared_error(Y_train,pred_train)


# In[ ]:


#metrics.mean_squared_error(y_test,pred_test)


# In[ ]:


#metrics.mean_squared_error(Y_train,pred_train)-metrics.mean_squared_error(y_test,pred_test)


# ## Prediction Using Regression

# In[ ]:


Train_data


# In[ ]:


Y_train=Train_data['Item_Outlet_Sales']
X_train=Train_data.drop('Item_Outlet_Sales',axis=1)


# In[ ]:





# In[ ]:


train=Train_data.drop(['Item_Outlet_Sales'],axis=1)
predictions=Train_data['Item_Outlet_Sales']
out=[]
LM_model=LR(normalize=True)
for i in range(len(Test_Data)):
    LM_fit=LM_model.fit(train.drop(['Outlet_Identifier','Item_Identifier'],axis=1),predictions)
    Output=LM_fit.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1)[Test_Data.index==i])
    out.append(Output)
    train.append(Test_Data[Test_Data.index==i])
    predictions.append(pd.Series(Output))
    
    


# In[ ]:


len(out)


# In[ ]:


len(Test_Data)


# In[ ]:


outp=np.vstack(out)


# In[ ]:


ansp=pd.Series(data=outp[:,0],index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


Outp_df=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],ansp]).T


# In[ ]:


Outp_df.to_csv('UploadLMP.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


mod1_train_pred=Lm.predict(Train_data.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis=1))


# In[ ]:


from sklearn import metrics
from math import sqrt


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],mod1_train_pred))/np.mean(Train_data['Item_Outlet_Sales'])


# In[ ]:


# analytics vidhya score
#1273.7483459686


# In[ ]:





# In[ ]:





# In[ ]:


ans=Lm.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


answer=pd.Series(data=ans,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


#len(ans)


# In[ ]:


#pd.DataFrame(np.array([[Test_Data['Item_Identifier']],[Test_Data['Outlet_Identifier']],[answer]]),index=Test_Data.index,columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])


# In[ ]:


#dict(ItemTest_Data['Item_Identifier'],Test_Data['Outlet_Identifier'],ans)


# In[ ]:





# In[ ]:


#(pd.DataFrame(Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],answer).T).to_csv("upload.csv",encoding='utf-8', index=False)


# In[ ]:


#Out_df=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],answer]).T


# In[ ]:


#Out_df


# In[ ]:


#Out_df.to_csv('Upload2.csv',index=False)


# ---
# ---

# # Model 2
# 

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


#predictors = [x for x in train.columns if x not in [target]+IDcol]


# In[ ]:


rr=Ridge(alpha=0.5,fit_intercept=True,normalize=True)


# In[ ]:


rr.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])


# In[ ]:


rr_pred=rr.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


rr_pred_train=rr.predict(Train_data.drop(['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'],axis=1))


# In[ ]:


rr_ans=pd.Series(data=rr_pred,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


len(ans)


# In[ ]:


rr_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],rr_ans]).T


# In[ ]:


#rr_out.to_csv('Uploadrr.csv',index=False)


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],rr_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])


# In[ ]:





# ### Result
#  + Ridge regression with alpha=0.1, socre is 1284
#  + Ridge regression with alpha=0.5 score is 1338

# ---
# ---

# # Model 3

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


lasso=Lasso(alpha=0.5,fit_intercept=True,normalize=True)


# In[ ]:


lasso.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])


# In[ ]:


lasso_pred=lasso.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


lasso_pred_train=lasso.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))


# In[ ]:


lasso_ans=pd.Series(data=lasso_pred,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


len(lasso_pred)


# In[ ]:


lasso_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],rr_ans]).T


# In[ ]:


#lasso_out.to_csv('Uploadlasso.csv',index=False)


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],lasso_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])


# # Lasso results
# + with alpha=0.1, is 1338
# + with alpha=0.5, is 1338

# ---
# ---

# # Model 4 (SVR)

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


svr=SVR(kernel='linear',gamma='auto',C=5,epsilon=1.2)


# In[ ]:


svr.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])


# In[ ]:


svr_pred=svr.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


svr_pred_train=svr.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))


# In[ ]:


svr_ans=pd.Series(data=svr_pred,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


svr_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],svr_ans]).T


# In[ ]:


#svr_out.to_csv('Uploadsvr.csv',index=False)


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],svr_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],svr_pred_train))


# In[ ]:


svr_pred_train


# In[ ]:


#Train_data['Item_Outlet_Sales']


# ---
# ---

# ---
# ---

# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


DTR=DecisionTreeRegressor()


# In[ ]:


DTR.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])


# In[ ]:


dtr_pred_train=DTR.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],dtr_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],dtr_pred_train))


# In[ ]:


dtr_pred_train


# In[ ]:


dtr_pred=DTR.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


dtr_ans=pd.Series(data=dtr_pred,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


dtr_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],dtr_ans]).T


# In[ ]:


#dtr_out.to_csv('Uploaddtr.csv',index=False)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(Train_data['Item_Outlet_Sales'],dtr_pred_train)


# In[ ]:





# ---
# ---

# # MLP

# In[ ]:


from sklearn.neural_network import MLPRegressor


# In[ ]:


ann=MLPRegressor(activation='relu',alpha=2.0,learning_rate='adaptive',warm_start=True,hidden_layer_sizes=(2500,),max_iter=1000)


# In[ ]:


ann.fit(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),Train_data['Item_Outlet_Sales'])


# In[ ]:


ann_train_pred=ann.predict(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))


# In[ ]:


ann_train_pred


# In[ ]:


r2_score(Train_data['Item_Outlet_Sales'],ann_train_pred)


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],ann_train_pred))/np.mean(Train_data['Item_Outlet_Sales'])


# In[ ]:


ann_pred=ann.predict(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


ann_ans=pd.Series(data=ann_pred,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


ann_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],ann_ans]).T


# In[ ]:


ann_out.to_csv('Uploadann.csv',index=False)


# ---
# ---

# # ANN using keras

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


X_train=np.array(Train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))
y_train=np.array(Train_data['Item_Outlet_Sales'])


# In[ ]:


#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0], 1,))


# In[ ]:


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(91, input_dim=9, kernel_initializer='Orthogonal', activation='elu'))
    model.add(Dense(78,activation='relu',kernel_initializer='Orthogonal'))
    model.add(Dense(65,activation='relu',kernel_initializer='Orthogonal'))
    model.add(Dense(52,activation='relu',kernel_initializer='Orthogonal'))
    model.add(Dense(39,activation='relu',kernel_initializer='Orthogonal'))    
    model.add(Dense(26,activation='relu',kernel_initializer='Orthogonal'))
    model.add(Dense(1,activation='relu',kernel_initializer='Orthogonal'))   
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[ ]:


seed = 7
np.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=20, verbose=0)


# In[ ]:


estimator.fit(X_train,y_train, batch_size = 50, epochs = 200)


# In[ ]:


ann1_pred_train=estimator.predict(X_train)


# In[ ]:


r2_score(Train_data['Item_Outlet_Sales'],ann1_pred_train)


# In[ ]:


sqrt(metrics.mean_squared_error(Train_data['Item_Outlet_Sales'],ann1_pred_train))/np.mean(Train_data['Item_Outlet_Sales'])


# In[ ]:


#X_test=np.array(Test_Data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


#X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[ ]:


X_train


# In[ ]:


Test_Data.head()


# In[ ]:


Test_Data.shape


# In[ ]:


Train_data.shape


# In[ ]:


ann1_pred=estimator.predict(np.array(Test_Data.drop(['Outlet_Identifier','Item_Identifier'],axis=1)))


# In[ ]:


ann1_ans=pd.Series(data=ann1_pred,index=Test_Data.index,name='Item_Outlet_Sales')


# In[ ]:


ann1_out=pd.DataFrame([Test_data['Item_Identifier'],Test_data['Outlet_Identifier'],ann1_ans]).T


# In[ ]:


ann1_out.to_csv('Uploadann1.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ---
# ---
