#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

title_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }
curve_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }


# In[ ]:


df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head()


# In[ ]:


df.columns


# In[ ]:


df['is_canceled'].value_counts() #target variable


# In[ ]:


sns.countplot(df['is_canceled'])


# In[ ]:


df.shape


# # Missing Values

# In[ ]:


df.isnull().sum()*100/df.shape[0]


# In[ ]:


df.drop('company',axis=1, inplace=True)


# In[ ]:


df['children'].unique()


# In[ ]:


df['country'].unique()


# In[ ]:


df['agent'].unique()


# In[ ]:


df[df['children']==10]


# In[ ]:


df.drop([328],axis=0,inplace=True) # outliers values delete. 


# In[ ]:


df['children'].unique()


# # Fillna Values

# In[ ]:


df['country'].replace(np.nan,"Undefined",inplace=True)
df['agent'].replace(np.nan , 0 , inplace=True)
df['children'].replace(np.nan , 0 , inplace=True)


# In[ ]:


df.isnull().sum()*100/df.shape[0]


# In[ ]:


df=df.to_csv('Clear_Hotel_Booking.csv',encoding='utf8')


# # Virtualization with Clear Data

# In[ ]:


df= pd.read_csv('Clear_Hotel_Booking.csv')
df.drop('Unnamed: 0',axis=1 , inplace=True)
df.head()


# In[ ]:


df.describe().T


# In[ ]:


sns.set(style = "darkgrid")
plt.title("Canceled ", fontdict = {'fontsize': 20})
ax = sns.countplot(x = "is_canceled", data = df)


# In[ ]:


plt.figure(figsize =(13,10))
sns.set(style="darkgrid")
plt.title("Total Customers - Monthly ", fontdict={'fontsize': 20})
ax = sns.countplot(x = "arrival_date_month", hue = 'is_canceled', data = df)


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x = "market_segment", y = "stays_in_weekend_nights", data = df, hue = "is_canceled", palette = 'Set1')


# In[ ]:


plt.figure(figsize = (13,10))
sns.set(style = "darkgrid")
plt.title("Countplot Distrubiton of Segment by Deposit Type", fontdict = {'fontsize':20})
ax = sns.countplot(x = "market_segment", hue = 'deposit_type', data = df)


# In[ ]:


plt.figure(figsize = (13,10))
sns.set(style = "darkgrid")
plt.title("Countplot Distributon of Segments by Cancellation", fontdict = {'fontsize':20})
ax = sns.countplot(x = "market_segment", hue = 'is_canceled', data = df)


# # Encoding Process

# In[ ]:


df['hotel'] = pd.get_dummies(df['hotel'],drop_first=True)
df['meal'] = pd.get_dummies(df['meal'],drop_first=True)
df['market_segment'] = pd.get_dummies(df['market_segment'],drop_first=True)
df['distribution_channel'] = pd.get_dummies(df['distribution_channel'],drop_first=True)
df['deposit_type'] = pd.get_dummies(df['deposit_type'],drop_first=True)
df['customer_type'] = pd.get_dummies(df['customer_type'],drop_first=True)
df['assigned_room_type'] = pd.get_dummies(df['assigned_room_type'],drop_first=True)
df['country'] = pd.get_dummies(df['country'],drop_first=True)


# In[ ]:


df.head()


# In[ ]:


df_corr=df.corr()
df_corr


# In[ ]:


plt.figure(figsize=(28,20))
sns.heatmap(df_corr, square=True, annot=True, linewidths=.5, vmin=0, vmax=1, cmap='viridis')
plt.title("Correlation Matrix", fontdict=title_font)
plt.show()


# # Model

# In[ ]:


import statsmodels.formula.api as smf
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[ ]:


df.head()


# # Stats Model

# ### We create model from statsmodels.api because we apply backward elimination on dependent variables.

# In[ ]:


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

X = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type","babies"],axis=1)
y = df['is_canceled']

sc = StandardScaler()
X_scl = sc.fit_transform(X)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=111, stratify = y)


loj = sm.Logit(y_train,X_train)
loj_model = loj.fit()
loj_model.summary()


# In[ ]:


def create_model(X,y,model,tip):
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=111, stratify = y)
    model.fit(X_train, y_train)
    
    prediction_train=model.predict(X_train)
    prediction_test=model.predict(X_test)
    
    prediction_test_prob = model.predict_proba(X_test)[:,1]
    
    cv = cross_validate(estimator=model,X=X,y=y,cv=10,return_train_score=True)
    
    d = pd.Series({'Accuracy_Train':accuracy_score(y_train,prediction_train),
                   'Precision_Train':precision_score(y_train,prediction_train),
                   'Recall_Train':recall_score(y_train,prediction_train),
                   'F1 Score_Train':f1_score(y_train,prediction_train),
                   'Accuracy_Test':accuracy_score(y_test,prediction_test),
                   'Precision_Test':precision_score(y_test,prediction_test),
                   'Recall_Test':recall_score(y_test,prediction_test),
                   'F1 Score_Test':f1_score(y_test,prediction_test),
                   'AUC Score':roc_auc_score(y_test, prediction_test_prob),
                   "Cross_val_train":cv['train_score'].mean(),
                   "Cross_val_test":cv['test_score'].mean() },name=tip)
    return d


# In[ ]:


X = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type","babies"],axis=1)

y = df['is_canceled']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X)

logistic=LogisticRegression()

metrics=pd.DataFrame()
metrics=metrics.append(create_model(X_scl,y,logistic,tip='Logistic_Regr.'))
metrics


# # Model Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV

logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X, y)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)


# In[ ]:


logistic=LogisticRegression(C=1000,penalty='l2',solver='liblinear' )
metrics=metrics.append(create_model(X_scl,y,logistic,tip='Logistic_Regr_tuning'))
metrics


# # Resampling

# In[ ]:


from sklearn.utils import resample
canceled_customer=df[df.is_canceled==1]
not_canceled_customer=df[df.is_canceled==0]

canceled_customer_resample= resample(canceled_customer,
                                     replace = True,
                                     n_samples = len(not_canceled_customer),
                                     random_state = 111)

resample_df = pd.concat([not_canceled_customer, canceled_customer_resample])
resample_df.is_canceled.value_counts()


# In[ ]:


X_r = resample_df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)
y_r = resample_df['is_canceled']

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_r)

logistic=LogisticRegression()

metrics=metrics.append(create_model(X_scl,y_r,logistic,'Resampled_Logistic'))
metrics


# # Model Tuning

# In[ ]:


logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X_scl, y_r)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)


# In[ ]:


logistic=LogisticRegression(C= 0.001, penalty= 'l1', solver= 'liblinear')

metrics=metrics.append(create_model(X_scl,y_r,logistic,'Resampled_Logistic_tuning'))
metrics


# # Logistic Regression with SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE

y_s = df['is_canceled']
X_s = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)

sm = SMOTE(random_state=111)
X_smote, y_smote = sm.fit_sample(X_s, y_s)

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_smote)

logistic=LogisticRegression()

metrics=metrics.append(create_model(X_scl,y_smote,logistic,'SMOTE_Logistic'))
metrics


# # Model Tuning

# In[ ]:


logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X_scl, y_smote)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)


# In[ ]:


logistic=LogisticRegression(C= 0.001, penalty= 'l1', solver= 'liblinear')

metrics=metrics.append(create_model(X_scl,y_r,logistic,'SMOTE_Logistic_tuning'))
metrics


# # Logistic Regression with ADASYN

# In[ ]:


from imblearn.over_sampling import ADASYN
y_a = df['is_canceled']
X_a = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)

ad = ADASYN(random_state=111)
X_adasyn, y_adasyn = ad.fit_sample(X_a, y_a)

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_adasyn)

logistic=LogisticRegression()

metrics=metrics.append(create_model(X_scl,y_adasyn,logistic,'ADASYN_Logistic'))
metrics


# # Model Tuning

# In[ ]:


logistic=LogisticRegression()
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2'],'solver': ('linear', 'lbfgs', 'liblinear')}

grid_cv = GridSearchCV(estimator=logistic,
                       param_grid = parameters,
                       cv = 10)
grid_cv.fit(X_scl, y_adasyn)
print("The best parameters : ", grid_cv.best_params_)
print("The best score         : ", grid_cv.best_score_)


# In[ ]:


logistic=LogisticRegression(C= 0.001, penalty= 'l1', solver= 'liblinear')

metrics=metrics.append(create_model(X_scl,y_adasyn,logistic,'ADASYN_Logistic_tuning'))
metrics


# # Conclusion
# 
# Auc Score gives us information about true positive rate and false positive rate. We want AUC Score to be high, having a high rate gives us information about the accuracy of our model's predictions. Here, the highest AUC Score value belongs to Smote.
# 
# The Acuracy Score shows us how much we can explain with the coefficients (coeff) and the fixed or constant number (bias), where StandarScale and linear classifier tuning are higher than others. This shows us how much of the data we have allocated for testing (Accuracy Test) also shows that we can explain 75% of the data in the same way. Care must be taken when making the decision.
# 
# The F1 Score shows us that the data is not more balanced, but is more successfully predicted. It offers convenience while working, it is the average of precision and recall harmonics. Here, Smote explains to us that the applied data is more successful.
# 
# Precision gives us the percentage of what is assumed to be 1 in reality. For example, we have 100 data and all of them are 1. When we model it, let's estimate all 5 of them as 0, so our precision value becomes 0.95. If we apply to the situation here. Almost all of the data with a value of 1 are correctly predicted in Logistic_reg and Resample_reg, so we actually guessed 0.99 of those who are actually 1 and our precision value is 1%. We have 1% error. If we go back to the example, when we set up and estimate our 100-item data, there is only one data that we estimate as 0.
# 
# Recall gives us the information of how much of what is actually estimated to be 1 is actually estimated. When you look at the table, don't let the precision value mislead you. Although the precision value is 0.99, the recall value is 0.32. That is, only 0.32 of the rate given as 0.99 is actually correct. The rate of really knowing correctly has a success of 32%. Instead, Smote showed a more consistent rate. The precision value is 0.65 and the recall value is 0.62. that is, the accurately estimated rate of a 65% slice is 62%. This is a more reliable and high rate.

# # The Best Model

# In[ ]:


metrics.iloc[[4], :]


# In[ ]:


y_s = df['is_canceled']
X_s = df.drop(['reservation_status_date',"is_canceled","arrival_date_year","arrival_date_month","reservation_status"
             ,"required_car_parking_spaces"
             ,"reserved_room_type"],axis=1)

sm = SMOTE(random_state=111)
X_smote, y_smote = sm.fit_sample(X_s, y_s)

scaler=StandardScaler()
X_scl=scaler.fit_transform(X_smote)

logistic=LogisticRegression()
logistic_model=logistic.fit(X_scl,y_smote)


# In[ ]:


logistic_model.coef_


# In[ ]:


logistic_model.intercept_


# # Formulation

# In[ ]:


print("0.65971861","+ hotel * "+str(-0.14196832),"+ lead_time * "+str(0.34654226),"+ total_of_special_requests * "+str(-0.07286722),"+ reservation_status_date * "+str(-0.00639928),
    
"+ arrival_date_week_number * "+str(-0.05187588),"+ arrival_date_day_of_month * "+str(0.07373981),"+ stays_in_weekend_nights *"+str(0.0303397),
      "+ stays_in_week_nights *"+str(0.05642681), "+ adults *"+str(-0.00788824), "+ children *"+str(0.02600913),"+ babies *"+str(0.07122837), 
     "+ meal *"+str(0.05067837), "+ market_segment *"+str(-0.33731968), "+ distribution_channel *"+str(-0.19108182),"+ is_repeated_guest *"+str(2.31945534),
     "+ previous_cancellations *"+str(-0.55344701), "+ previous_bookings_not_canceled *"+str(-0.06139697), "+ assigned_room_type *"+str(-0.36896794),"+ booking_changes *"+str( 1.90414296),
     "+ deposit_type *"+str(0.08676973), "+ agent *"+str( -0.09250369), "+ days_in_waiting_list *"+str(-0.07363203),"+ customer_type *"+str( 0.34808834),"+ adr *"+str(-0.50650057),sep='\n')
print("= is_canceled")

