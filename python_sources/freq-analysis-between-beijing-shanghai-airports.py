#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model, metrics, model_selection, preprocessing, multiclass, svm
from scipy import interp
import os, datetime , time, sys
from itertools import cycle

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# The analysis below is performed for only one dataset i.e. sha_pek.csv.
# 
# The ML models used are: Linear and Logistic Regression (OVR and Multinomial)
# 
# Evaluation Metrics: AUC-ROC, F1 Score, Mean Absolute Square, MSE, RMSE, R^2 Score, Confusion Matrix
# 
# Any additional comments or suggestions are welcome.

# In[ ]:


sha_pek = pd.read_csv('/kaggle/input/air-tickets-between-shanghai-and-beijing/sha-pek.csv')


# # EXPLORATORY DATA ANALYSIS

# In[ ]:


sha_pek.head(5)


# # Check for Valid and Invalid Data
# 
# There were 175 rows of invalid data observed in this dataset.
# 
# Determination: EDA was performed and there were flights with booking date(CreateDate) after the departure date (departureDate) which is not possible unless someone booked a flight after it's departure. Another reason that can be possible, is the delay in entering the data into the system by the airlines.
# 
# We consider the column dateDifference to perform our analysis. The rows with dateDifference less than 0 were marked as invalid. Apart from that, the data where the dateDifference was 0 and the dates of createDate and departureDate were the same but there was a time difference, were also considered invalid. Ex:
# 
# departureDate: 2015-08-01 16:00:00
# createDate: 2015-08-01 22:30:00

# In[ ]:


sha_pek['createDate'] = pd.to_datetime(sha_pek['createDate'])
sha_pek['departureDate'] = pd.to_datetime(sha_pek['departureDate'])
sha_pek_invalid =  sha_pek[((sha_pek['departureDate']-sha_pek['createDate'])/np.timedelta64(1,'s'))<0]
sha_pek_invalid.head(5)


# The reason for invalid dateDifference could be:
# 
# 1. Tickets bought over the counter and recorded late into the system
# 2. Agent purchase which was recorded late in the system
# 3. Tickets purchased after the flight departed (highly unlikely but possible)

# In[ ]:


sha_pek_valid = sha_pek[((sha_pek['departureDate']-sha_pek['createDate'])/np.timedelta64(1,'s'))>0]
min((sha_pek_valid['departureDate']-sha_pek_valid['createDate'])/np.timedelta64(1,'s'))


# Minimum Time Difference between Create Date and Departure Date is 155 seconds, realistically impossible as 155 seconds is an impossible time difference because of airport protocols. We can consider this as invalid data, but there is an exception here, through research, it was found that many third party online booking websites show the booking of a flight even before the exact time of it's departure. For the analysis below this was the reason considered for these close bookings.

# In[ ]:


#EDA of whether the price column is actually the discounted price.

singleFlight_Filter = sha_pek_valid.loc[sha_pek_valid.flightNumber.isin(['MU5389']) & 
                  sha_pek_valid.cabinClass.isin(['C']) & 
                  sha_pek_valid.departureDate.isin(['2019-07-21 07:20:00'])]

singleFlight_Filter


# It was observed that the column rate lies between 0 and 1, the lower the rate, the higher the discounted price.
# 
# For rate = 1.0, no discount was observed, and that price was treated as original price of the flight. In the example above flightNumber: 'MU5389' had two prices 1,210 and 5,660 at rates 0.22 and 1.0 respectively for cabin class 'C'. We believe the price 5,660 was the original price of the flight. Similar assumptions were made for other flights as well.

# In[ ]:


#CALCULATING DEPARTURE TIME FROM THE DEPARTURE DATE COLUMN
sha_pek_valid['departureTime'] =pd.to_datetime(sha_pek_valid['departureDate']).dt.strftime('%H:%M')

#SELECTING NECESSARY COLUMNS FROM THE VALID DATASET
tabular_subset = sha_pek_valid[['flightNumber','departureTime','cabinClass','price','rate']]

#DETERMINING ORIGINAL PRICE OF EACH FLIGHT WHERE RATE = 1.0 which means no discount
tabular_subset = sha_pek_valid.loc[sha_pek_valid['rate']==1.0]

#FINAL OUTPUT
tabular_subset[['flightNumber','departureTime','cabinClass','price']]


# Above grouping is done to display the original price of every flightNumber grouped by their cabinClass and departureTime. 
# 
# Note: departureTime was sliced from departureDate

# # LINEAR REGRESSION

# Below we will first analyse some relationships between a few columns. We will start with departureTime and price.
# 
# To determine the relation between departureTime and price according to the flight names, our strategy will be to first check the peak hours and frequency of flights in a day during all of their time intervals.

# In[ ]:


##MAKING TIME EXECUTABLE FOR REGRESSION

tabular_subset['departureTime']=tabular_subset['departureTime'].str.replace(':','.')

#DETERMINING THE FLIGHT NAME BY TAKING THE FIRST TWO LETTERS ACCORDING TO THEIR NAMING CONVENTION
tabular_subset['flightType']=tabular_subset['flightNumber'].str.slice(0,2)

#GENERATING GRID PLOTS ACCORDING TO DIFFERENT FLIGHT NAMES COMPUTED FROM ABOVE

airlines = tabular_subset.flightType.to_list()
airlines = set(airlines)

fig = plt.figure(figsize=(100,100))

for al,num in zip(airlines,range(1,7)):
    al_flights = tabular_subset.loc[tabular_subset['flightType']==al].sort_values(by=['departureTime'])
    ax=fig.add_subplot(20,20,num)
    ax.plot(al_flights['departureTime'], al_flights['price'])
    ax.set_title(al)
    
plt.show()


# According to the dataset, above observations have been recorded on the departureTime vs price relation. The Graphs are classified into 6 representing each flight in the 'valid' data.
# 
# 1. Flight CA is spread throughout the day with peak prices throughout morning to evening.
# 2. Flight MU is also spread throughout the day with peak price in late morning and early evening.
# 3. Flight CZ runs only on two times both ranging from 11:45 to 12:30, hitting its peak price at 12.30.
# 4. Flight FM runs one flight in the afternoon while three in the nights, with the peak in the afternoon.
# 5. Flight HO runs only twice in the night at times 21:45 and 22:00 with peak prices.
# 6. Flight HU runs through different intervals thoughout the day, with peak prices throughout those intervals.

# In[ ]:


#Linear Regression for Q5 (DEPARTURE TIME vs PRICE)

x=tabular_subset[['departureTime']].values
y=tabular_subset['price'].values
rm = linear_model.LinearRegression()
rm.fit(x,y)
sst = np.sum((y-np.mean(y))**2)
ssr = np.sum((rm.predict(x)-np.mean(y))**2)
sse = np.sum((rm.predict(x)-y)**2)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Total Sum of Squares is: ', sst)
print('The Residual Sum of Squares is: ',sse)
print('The Explained Sum of Squares is: ', ssr)
print('The R^2 from regressor: ', rm.score(x,y))
print('The R^2 from ssr/sst: ', ssr/sst)


# From the above stats, we can see that there isn't a significant R^2 for the relation between departureTime and price. However, when taking a deeper dive into different airlines (as shown in the grid graphs above), it shows different stories for different flights.
# 

# Next we analyse the relation between dateDifference and rate
# 
# To analyse the relation between dateDifference and Rate, we need to first plot the datedifference vs rate plot to observe their relation, and then perform linear regression to calculate the R^2 evaluation metric to determine a conclusion

# In[ ]:


#Plotting datedifference vs rate

cabins = sha_pek_valid.cabinClass.to_list()
cabins = set(cabins)

fig, ax = plt.subplots()

sha_pek_valid.plot(x='dateDifference',y='rate',ax=ax)
plt.show()


# The graph shows a **'not-so'** significant relation between the two attributes

# In[ ]:


#Linear Regression Date Difference vs Rate

x=sha_pek_valid[['dateDifference']].values
y=sha_pek_valid['rate'].values
rm = linear_model.LinearRegression()
rm.fit(x,y)
sst = np.sum((y-np.mean(y))**2)
ssr = np.sum((rm.predict(x)-np.mean(y))**2)
sse = np.sum((rm.predict(x)-y)**2)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Total Sum of Squares is: ', sst)
print('The Residual Sum of Squares is: ',sse)
print('The Explained Sum of Squares is: ', ssr)
print('The R^2 from regressor: ', rm.score(x,y))
print('The R^2 from ssr/sst: ', ssr/sst)


# The Relation between dateDifference and rate isn't significant as shown from the graph and the R^2 proves it with a smaller value. We can see from the data that, the lowest and the highest rates are available at both the maximum and minimum dateDifference.

# Next we perfomed the impact of different attributes on the columns: price and rate.
# 
# For Linear Regression Model, we will use flightNumber and cabinClass as two attributes to determine the model on price and rate
# respectively.
# 
# We will change the flightNumber and cabinClass into categorical dummy variables (OneHotEncoding) in order to regress against the price, since both flightNumber and cabinClass are strings
# 
# PRICE

# In[ ]:


#LINEAR REGRESSION OF THE ATTRIBUTES flightNumber, cabinClass, departureTime on Target: Price

lrAttributes = sha_pek_valid[['flightNumber','price','cabinClass','departureTime']]

#MAKING TIME EXECUTABLE FOR REGRESSION
lrAttributes['departureTime'] = lrAttributes['departureTime'].str.replace(':','.')

#CATEGORIZING OHC 
cabinClassEnc = pd.get_dummies(lrAttributes['cabinClass'])
flightNumberEnc = pd.get_dummies(lrAttributes['flightNumber'])

#CONCATENATING THE OHC to DATASET
lrAttributes = pd.concat([lrAttributes,cabinClassEnc,flightNumberEnc],axis=1)

#DELETING UNNECESSARY COLUMNS
lrAttributes = lrAttributes.drop(['flightNumber','cabinClass'],axis=1)

#SEPARATING PRICE vs THE REST
yAttributes = ['price']
xAttributes = list(set(list(lrAttributes.columns))-set(yAttributes))
xPrice = lrAttributes[xAttributes].values
yPrice = lrAttributes[yAttributes].values
xTrainPrice, xTestPrice, yTrainPrice, yTestPrice = model_selection.train_test_split(xPrice, 
                                                                                    yPrice, 
                                                                                    test_size=0.2, 
                                                                                    random_state = 2020)

#LINEAR REGRESSION
rm = linear_model.LinearRegression()
rm.fit(xTrainPrice,yTrainPrice)
trainPredPrice = rm.predict(xTrainPrice)
testPredPrice = rm.predict(xTestPrice)

#EVALUATION METRICS
print('R^2 for Training Data: ', rm.score(xTrainPrice,yTrainPrice))
print('R^2 for Test Data: ', rm.score(xTestPrice,yTestPrice))
print('Explained Metrics Score Test Data: ', metrics.explained_variance_score(yTestPrice,testPredPrice))
print('Mean Absolute Error Test Data: ', metrics.mean_absolute_error(yTestPrice,testPredPrice))
print('Mean Squared Error Test Data: ', metrics.mean_squared_error(yTestPrice,testPredPrice))
print('Root Mean Squared Error Test Data: ', np.sqrt(metrics.mean_squared_error(yTestPrice,testPredPrice)))


# The R^2 for the Training Data is 70.83% while the R^2 for the Test Data is 70.65% which shows a good performance.
# 
# RATE

# In[ ]:


#LINEAR REGRESSION OF THE ATTRIBUTES flightNumber, cabinClass, departureTime on Target: Rate

lrAttributes = sha_pek_valid[['flightNumber','rate','cabinClass','departureTime']]

#MAKING TIME EXECUTABLE FOR REGRESSION
lrAttributes['departureTime'] = lrAttributes['departureTime'].str.replace(':','.')

#CATEGORIZING OHC
cabinClassEnc = pd.get_dummies(lrAttributes['cabinClass'])
flightNumberEnc = pd.get_dummies(lrAttributes['flightNumber'])

#CONCATENATING THE OHC to DATASET
lrAttributes = pd.concat([lrAttributes,cabinClassEnc,flightNumberEnc],axis=1)

#DELETING UNNECESSARY COLUMNS
lrAttributes = lrAttributes.drop(['flightNumber','cabinClass'],axis=1)

#SEPARATING PRICE vs THE REST
yAttributes = ['rate']
xAttributes = list(set(list(lrAttributes.columns))-set(yAttributes))
xRate = lrAttributes[xAttributes].values
yRate = lrAttributes[yAttributes].values
xTrainRate, xTestRate, yTrainRate, yTestRate = model_selection.train_test_split(xRate, 
                                                                                    yRate, 
                                                                                    test_size=0.2, 
                                                                                    random_state = 2020)

#LINEAR REGRESSION
rm = linear_model.LinearRegression()
rm.fit(xTrainRate,yTrainRate)
trainPredRate = rm.predict(xTrainRate)
testPredRate = rm.predict(xTestRate)

#EVALUATION METRICS
print('R^2 for Training Data: ', rm.score(xTrainRate,yTrainRate))
print('R^2 for Test Data: ', rm.score(xTestRate,yTestRate))
print('Explained Metrics Score Test Data: ', metrics.explained_variance_score(yTestRate,testPredRate))
print('Mean Absolute Error Test Data: ', metrics.mean_absolute_error(yTestRate,testPredRate))
print('Mean Squared Error Test Data: ', metrics.mean_squared_error(yTestRate,testPredRate))
print('Root Mean Squared Error Test Data: ', np.sqrt(metrics.mean_squared_error(yTestRate,testPredRate)))


# The R^2 for the Training Data is 7.33% and the Test Data is 7.28% which shows bad performance.
# 
# <b>The Performance of the two models are significantly different.</b>

# # LOGISTIC REGRESSION
# 
# Next we analyse the impact of attributes: flightNumber, cabinClass, traAirport and priceClass over the column of rate. We use the categorization of the column rate as:
# 
# - rate = 1.0, no discount (new value = 0)
# - rate < 1.0, discount (new value = 1)

# In[ ]:


#TAKING RELEVANT COLUMNS INTO CONSIDERATION

binAttributes = sha_pek_valid[['flightNumber','traAirport','cabinClass','priceClass','rate']]
binAttributes['rate'][binAttributes['rate']!=1]=0

#CATEGORIZING ONE HOT ENCODER VALUES FOR CLASSIFICATION
fNumberEnc = pd.get_dummies(binAttributes['flightNumber'])
traAirportEnc = pd.get_dummies(binAttributes['traAirport'])
cClassEnc = pd.get_dummies(binAttributes['cabinClass'])
priceClassEnc = pd.get_dummies(binAttributes['priceClass'])

#COMBINING OHC VALUES TO THE ORIGINAL DATASET
binAttributes = pd.concat([binAttributes,fNumberEnc,traAirportEnc,cClassEnc,priceClassEnc], axis=1)
#DELETING UNNECESSARY COLUMNS
binAttributes = binAttributes.drop(['flightNumber','traAirport','cabinClass','priceClass'], axis=1)

#SEPARATING RATE vs THE REST
yAttribute = ['rate']
xAttribute = list(set(list(binAttributes.columns))-set(yAttribute))

#LOGISTIC REGRESSION

xR = binAttributes[xAttribute].values
yR = binAttributes[yAttribute].values.astype(int)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xR,yR,test_size=0.25,random_state=2021)

logR = linear_model.LogisticRegression(solver='lbfgs')
logR.fit(xTrain,yTrain)

testPred = logR.predict(xTest)
print('The Accuracy Score is: ', metrics.accuracy_score(yTest,testPred)) #0.97216


# <b> The accuracy score out of the regression was observed to be <i>0.97216</i> approximately.</b>

# The analyses below are done using OneVsRestRegression (OVR) and Multinomial models. Again we are categorizing rate into different classes. However, this time, we are not considering the no discount rates i.e. rate = 1.0. For the OVR analysis we will be categorizing data into three classes:
# 
# - if 0.75 < rate < 1.0, then class 1 (new value = 1.0)
# - if 0.5 < rate <= 0.75, the class 2 (new value = 2.0)
# - if rate <= 0.5, then class 3 (new value = 3.0)

# In[ ]:


#TAKING NECESSARY COLUMNS INTO CONSIDERATION

rocAttributes = sha_pek_valid[['flightNumber','traAirport','cabinClass','priceClass','rate']]
rocAttributes = rocAttributes.loc[rocAttributes['rate'] != 1]

#CATEGORIZING DATA ACCORDING TO THE QUESTION

for i in range(0,len(rocAttributes['rate'].values)):
    if rocAttributes['rate'].values[i] < 1 and rocAttributes['rate'].values[i] > 0.75:
        rocAttributes['rate'].values[i] = 1
    elif rocAttributes['rate'].values[i] <= 0.75 and rocAttributes['rate'].values[i] > 0.5:
        rocAttributes['rate'].values[i] = 2
    elif rocAttributes['rate'].values[i] <=0.5:
        rocAttributes['rate'].values[i] = 3

#CREATING ONE HOT ENCODER VALUES FOR CLASSIFICATION
fNumberEnc_roc = pd.get_dummies(rocAttributes['flightNumber'])
traAirportEnc_roc = pd.get_dummies(rocAttributes['traAirport'])
cClassEnc_roc = pd.get_dummies(rocAttributes['cabinClass'])
priceClassEnc_roc = pd.get_dummies(rocAttributes['priceClass'])

#DELETING REDUNDANT DATA
rocAttributes = pd.concat([rocAttributes,fNumberEnc_roc,traAirportEnc_roc,cClassEnc_roc,priceClassEnc_roc], axis=1)
rocAttributes = rocAttributes.drop(['flightNumber','traAirport','cabinClass','priceClass'], axis=1)

#SEPARATING RATE VS THE REST OF THE ARGUMENTS
yAttr = ['rate']
xAttr = list(set(list(rocAttributes.columns))-set(yAttribute))

#ADDING NOISY FEATURES
xR1 = rocAttributes[xAttr].values
yR1 = rocAttributes[yAttr].values.astype(int)
yR1 = preprocessing.label_binarize(yR1,classes=[1,2,3])

nClasses = yR1.shape[1]
randomState = np.random.RandomState(0)

#SPLITTING TRAIN AND TEST BY 70:30 RATIO
xTrainR, xTestR, yTrainR, yTestR = model_selection.train_test_split(xR1, yR1, test_size=0.3, random_state=0)


# In[ ]:


#OVR LOGISTIC REGRESSION AND TEST PREDICTION

cfier = multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=0))
yScore = cfier.fit(xTrainR,yTrainR).decision_function(xTestR)


# In[ ]:


#COMPUTING ROC CURVE AND ROC AREA FOR EACH CLASS

fpr=dict()
tpr=dict()
roc_auc=dict()


for i in range(nClasses):
    fpr[i], tpr[i], _ = metrics.roc_curve(yTestR[:,i],yScore[:,i])
    roc_auc[i] = metrics.auc(fpr[i],tpr[i])

    
#AGGREGATE ALL FALSE POSITIVE RATES    
allFpr = np.unique(np.concatenate([fpr[i] for i in range(nClasses)]))

#INTERPOLATE ALL ROC CURVES AT THESE POINTS
meanTpr = np.zeros_like(allFpr)
for i in range(nClasses):
    meanTpr += interp(allFpr, fpr[i], tpr[i])

#AVERAGE IT AND CALCULATE AUC
meanTpr /= nClasses

fpr["macro"] = allFpr
tpr["macro"] = meanTpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(yTestR.ravel(), yScore.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


#ROC Curves
plt.figure()
lw=2
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nClasses), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi_Class ROC')
plt.legend(loc="lower right")
plt.show()


# We can see from the plot that:
# 
# 1. Micro-Average ROC Curve (AREA): 0.95
# 2. Macro-Average ROC Curve (AREA) = 0.94
# 3. ROC Curve Class 0 (AREA) = 0.98
# 4. ROC Curve Class 1 (AREA) = 0.91
# 5. ROC Curve Class 2 (AREA) = 0.94

# Multinomial Analysis (Stacking the binary and Multi Class classifiers from the above analyses.The updated classes:
# 
# - if rate = 1.0, class 0 (new value = 0.0)
# - if 0.75 < rate < 1.0, class 1 (new value = 1.0)
# - if 0.5 < rate <= 0.75, class 2 (new value = 2.0)
# - if rate <= 0.5, class 3 (new value = 3.0)

# In[ ]:


#TAKING NECESSARY COLUMNS INTO CONSIDERATION
stackAttributes = sha_pek_valid[['flightNumber','traAirport','cabinClass','priceClass','rate']]

#CATEGORIZING DATA ACCORDING TO THE QUESTION

for i in range(0,len(stackAttributes['rate'].values)):
    if stackAttributes['rate'].values[i] == 1:
        stackAttributes['rate'].values[i] = 0
    elif stackAttributes['rate'].values[i] < 1 and stackAttributes['rate'].values[i] > 0.75:
        stackAttributes['rate'].values[i] = 1
    elif stackAttributes['rate'].values[i] <= 0.75 and stackAttributes['rate'].values[i] > 0.5:
        stackAttributes['rate'].values[i] = 2
    elif stackAttributes['rate'].values[i] <=0.5:
        stackAttributes['rate'].values[i] = 3

#CREATING ONE HOT ENCODER VALUES FOR CLASSIFICATION

fNumberEnc_stack = pd.get_dummies(stackAttributes['flightNumber'])
traAirportEnc_stack = pd.get_dummies(stackAttributes['traAirport'])
cClassEnc_stack = pd.get_dummies(stackAttributes['cabinClass'])
priceClassEnc_stack = pd.get_dummies(stackAttributes['priceClass'])

#DELETING REDUNDANT DATA TO AVOID PERFORMANCE ISSUES
stackAttributes = pd.concat([stackAttributes,fNumberEnc_stack,traAirportEnc_stack,cClassEnc_stack,priceClassEnc_stack], axis=1)
stackAttributes = stackAttributes.drop(['flightNumber','traAirport','cabinClass','priceClass'], axis=1)

#SEPARATING RATE VS THE REST OF THE ARGUMENTS
yAttr2 = ['rate']
xAttr2 = list(set(list(stackAttributes.columns))-set(yAttribute))

#LOGISTIC REGRESSION
xR2 = stackAttributes[xAttr].values
yR2 = stackAttributes[yAttr].values.astype(int)

xTrainS, xTestS, yTrainS, yTestS = model_selection.train_test_split(xR2, yR2, test_size=0.3, random_state=0)

clsfier = linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial')
clsfier.fit(xTrainS,yTrainS)

testPredS = clsfier.predict(xTestS)


# Confusion Matrix

# In[ ]:


confMat = metrics.confusion_matrix(yTestS,testPredS,[0,1,2,3])
confMat


# In[ ]:


fig = plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(confMat)
fig.colorbar(cax)
plt.title('Confusion Matrix')
ax.set_xticklabels(['']+[0,1,2,3])
ax.set_yticklabels(['']+[0,1,2,3])
plt.xlabel('Predictions')
plt.ylabel('Actuals')


# Precision, Recall and F1 Score

# In[ ]:


print(metrics.classification_report(yTestS, testPredS))


# Performance Measurement between OVR and Multinomial

# In[ ]:


print('The Training Score of OVR is: ',cfier.score(xTrainR,yTrainR))
print('The Training Score of Multinomial is: ', clsfier.score(xTrainS,yTrainS))


# ### Therefore we can see that Multinomial Logistic Regression has performed better compared to the OVR Logistic Regression.
