# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot as plt 
%matplotlib inline


# !pip install pytrends
from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


kw_list = ["Corona", "Coronavirus", "Covid", "Coronavirus symptoms"]
pytrends.build_payload(kw_list, cat=0, timeframe='2020-01-21 2020-05-01', geo='US-NY', gprop='')
data_out = pytrends.interest_over_time()
data_array = data_out.to_numpy()
print(data_out)
#print(data_array)

print(data_out.shape)

#rawcases = pd.read_excel("/kaggle/input/statecorrona/StateData.xlsx", index = None)
#print(rawcases)
state = pd.read_excel("/kaggle/input/state-data/StateData.xlsx",1, index = None)

statewo = state[0:100]
print(statewo)
#deaths = pd.read_excel("/kaggle/input/statecorrona/StateData.xlsx",2, index = None)
#print(deaths)
#casesLandmark = pd.read_excel("/kaggle/input/statecorrona/StateData.xlsx",3, index = None)
#print(casesLandmark)
#deathLandmarl = pd.read_excel("/kaggle/input/statecorrona/StateData.xlsx",4, index = None)
#print(deathsLandmark)




X = data_out.reset_index()
X = X.iloc[:,1:5]
X1 = X['Corona'].values.reshape(-1,1)
X2 = X['Coronavirus'].values.reshape(-1,1)
X3 = X['Covid'].values.reshape(-1,1)
X4 = X['Coronavirus symptoms'].values.reshape(-1,1)
y = statewo['New York New']
y1 = statewo['New York New'].values.reshape(-1,1)
# y_df = y.to_frame()


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y1, test_size=0.2, random_state=0)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y1, test_size=0.2, random_state=0)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y1, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# print("x test size:", X_test.size)
# print("y test size:", y_test.size)

# 'Corona'
regressor1 = LinearRegression()  
regressor1.fit(X_train1, y_train1) #training the algorithm
score1 = regressor1.score(X_train1, y_train1)
y_pred1 = regressor1.predict(X_test1)
print("\'Corona\' regression")
print("score", score1)
print("MAE", metrics.mean_absolute_error(y_test, y_pred1))
print("MSE", metrics.mean_squared_error(y_test, y_pred1))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

# 'Coronavirus'
regressor2 = LinearRegression()  
regressor2.fit(X_train2, y_train2)
score2 = regressor2.score(X_train2, y_train2)
y_pred2 = regressor2.predict(X_test1)
print("\'Coronavirus\' regression")
print("score", score2)
print("MAE", metrics.mean_absolute_error(y_test, y_pred2))
print("MSE", metrics.mean_squared_error(y_test, y_pred2))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))

# 'Covid'
regressor3 = LinearRegression()  
regressor3.fit(X_train3, y_train3)
score3 = regressor3.score(X_train3, y_train3)
y_pred3 = regressor3.predict(X_test3)
print("\'Covid\' regression")
print("score", score3)
print("MAE", metrics.mean_absolute_error(y_test, y_pred3))
print("MSE", metrics.mean_squared_error(y_test, y_pred3))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))

# 'Coronavirus symptoms'
regressor4 = LinearRegression()  
regressor4.fit(X_train4, y_train4)
score4 = regressor4.score(X_train4, y_train4)
y_pred4 = regressor4.predict(X_test4)
print("\'Coronavirus symptoms\' regression")
print("score", score4)
print("MAE", metrics.mean_absolute_error(y_test, y_pred4))
print("MSE", metrics.mean_squared_error(y_test, y_pred4))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))

# multiple regression
regressor_all = LinearRegression()
regressor_all.fit(X_train, y_train)
score_all = regressor_all.score(X_train, y_train)
y_pred = regressor_all.predict(X_test)
print("Multiple regression with all 4 search terms")
print("score", score_all)
print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Print regression equations
print("Corona:\t\t\ty = ", regressor1.coef_[0][0], "x + ", regressor1.intercept_[0], "\t\tR^2 = ", score1, sep='')
print("Coronavirus:\t\ty = ", regressor2.coef_[0][0], "x + ", regressor2.intercept_[0], "\t\tR^2 = ", score2, sep='')
print("Covid:\t\t\ty = ", regressor3.coef_[0][0], "x + ", regressor3.intercept_[0], "\t\tR^2 = ", score3, sep='')
print("Coronavirus symptoms:\ty = ", regressor4.coef_[0][0], "x + ", regressor4.intercept_[0], "\t\tR^2 = ", score4, sep='')
print("All:\ty = ", regressor_all.coef_[0], "(x1) + ", regressor_all.coef_[1], "(x2) + ", regressor_all.coef_[2], "(x3) + ", regressor_all.coef_[3], "(x4) + ", regressor_all.intercept_, sep='')
print("\tR^2 = ", score_all, sep='')


# sorting
sorting = pd.DataFrame({'Ypred': y_pred, 'Ytest': y_test})
sort = sorting.append(X_test)
sort.sort_values(by=['Ytest'])
df = pd.DataFrame({'Actual': sort['Ytest'], 'Predicted': sort['Ypred']})




# df.plot(kind='bar',figsize=(16,10))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()


# plt.scatter(sort["Coronavirus"], sort['Ytest'],  color='gray')
# plt.plot(sort["Coronavirus"], sort["Ypred"], color='red', linewidth=2)
# plt.show()



# predicted vs actual bar chart
fig, ax = plt.subplots()
ind = np.arange(len(y_test))  # the x locations for the groups
width = 0.35
ax.bar(ind - width/2, y_pred, width, label='Predicted')
ax.bar(ind + width/2, y_test, width, label='Actual')
ax.set_ylabel('Cases')
ax.set_title('Predicted vs Actual Cases')
ax.set_xticks(ind)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'))
ax.legend()
fig.tight_layout()
plt.show()


# compare predicted and actual scatterplots
fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
C = X_test4.ravel()
ax1.scatter(C, y_pred)
ax1.set_xlabel("\'Coronavirus symptoms\' Interest")
ax1.set_ylabel("Cases")
ax1.set_title("Predicted Cases")
fig.tight_layout()

ax2.scatter(C, y_test)
ax2.set_xlabel("\'Coronavirus symptoms\' Interest")
ax2.set_ylabel("Cases")
ax2.set_title("Actual Cases")
fig.tight_layout()
plt.show()


# graph diff idk
def diff(pred, actual):
    dff = []
    for i in range(0,len(pred)):
        dff.append(abs(pred[i] - actual[i]))
    return dff

percent_error = diff(y_pred, y_test.to_numpy())
fig, ax = plt.subplots()
ind = np.arange(len(y_test))
width = 0.35
ax.bar(ind, percent_error, width)
ax.set_ylabel('Difference')
ax.set_title('Difference Between Predicted and Actual Values')
ax.set_xticks(ind)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'))
ax.axhline(0,color='black') # x = 0
fig.tight_layout()
plt.show()



# x = np.arange(0,len(data_array)) 
# #print(x)
# y1 =[i[0] for i in data_array]
# y2 =[i[1] for i in data_array]
# y3 =[i[2] for i in data_array]
# y4 =[i[3] for i in data_array]
# #print(y)

# fig, axs = plt.subplots(2, 2)
# fig.tight_layout()
# fig.set_size_inches(18.5, 10.5)
# axs[0,0].plot(x,y1)
# axs[0,0].set_title("\"Corona\"") 
# #axs[0,0].xlabel("Days Since March 1st") 
# #axs[0,0].ylabel("Search Term Interest") 

# axs[0,1].plot(x,y2)
# axs[0,1].set_title(" \"Coronavirus\" ") 
# #axs[0,1].xlabel("Days Since March 1st") 
# #axs[0,1].ylabel("Search Term Interest") 

# axs[1,0].plot(x,y3)
# axs[1,0].set_title("\"COVID\"") 
# #axs[1,0].xlabel("Days Since March 1st") 
# #axs[1,0].ylabel("Search Term Interest") 

# axs[1,1].plot(x,y4)
# axs[1,1].set_title("\"coronavirus symptoms\"") 
# #axs[1,0].xlabel("Days Since March 1st") 
# #axs[1,0].ylabel("Search Term Interest")
