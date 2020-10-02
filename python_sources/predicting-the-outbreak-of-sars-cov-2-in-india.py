ekimport pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file1 = r'../input/novel-corona-virus-2019-dataset/covid_19_data.csv'


df = pd.read_csv(file1)
df1 = df.loc[df['Country/Region'] == 'India', ['ObservationDate', 'Confirmed', 'Deaths']]
print ("\nFirst 5 rows:\n",df1.head())
print ("\nLast 5 rows:\n",df1.tail())
print("\ndescribe:\n", df1.describe())
print('\n', df1.dtypes)
new_confirmed = df1["Confirmed"].astype(float)
new_deaths = df1["Deaths"].astype(float)

#date-parsed

df1['Date-parsed'] = pd.to_datetime(df1['ObservationDate'],format = "%d%m%y",
   infer_datetime_format=True)
#print(df1['Date-parsed'].iloc[5:])
print('\nDate_parsed:\n',df1['Date-parsed'].head())

#----------------------------------------------------------------------------

week_of_the_year = df1['Date-parsed'].dt.week
print("\nWeek after binning:\n",week_of_the_year.head())

new_week =week_of_the_year.astype(float)

bins = np.linspace(min(week_of_the_year), max(week_of_the_year), 4)
group = ["<week 4","week 5-8","week 8>"]

df1['week-binned'] = pd.cut(week_of_the_year, bins,
                           labels=group, include_lowest = True)
print('\nweek_binned data:\n',df1['week-binned'].head())
print('\nweek_describe\n', week_of_the_year.describe())
v_counts = df1["week-binned"].value_counts()
v_counts.index.name = "Outbreak Range"
print("\n", v_counts)

bins = np.linspace(min(new_confirmed), max(new_confirmed), 4)
group = ["Mild","Severe","Critical"]

df1['Confirmed-binned'] = pd.cut(new_confirmed, bins,
                           labels=group, include_lowest = True)
print('\nconfirmed_binned data:\n',df1['Confirmed-binned'].head())
v_counts = df1["Confirmed-binned"].value_counts()
v_counts.index.name = "Outbreak Range"
print("\n", v_counts)

#----------------------------------------------------------------------------
#twinx method

fig,ax = plt.subplots()
ax.plot(df1["Confirmed"],week_of_the_year , color="red")
ax.set_xlabel("Confirmed cases",fontsize=14)
ax.set_ylabel("Week 22nd Jan_24th Mar",color="red",fontsize=14)
#plt.xlim(0,12)
#plt.ylim(0,15)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
ax2.plot(df1["Confirmed"],df1["Deaths"] ,color="blue",marker="o")
ax2.set_ylabel("Deaths",color="blue",fontsize=14)
#plt.xlim(0,12)
#plt.ylim(0,15)
plt.title("SARS CoV-2 outbreak in India")
plot1 = plt.savefig('Outbreak analysis.png')
plt.show(plot1)

#-----------------------------------------------------------------------------

X = df1[["Confirmed"]]
Y = week_of_the_year
Z = df1[['Confirmed','Deaths']]
lm = LinearRegression()

lm.fit(X,Y)
Yhat = lm.predict(X)
print("\nIntercept:\n",lm.intercept_)
print("\nCoefficient:\n",lm.coef_)
print("\nThis shows the rise in Confirmed cases has very less impact on average change in weeks.\n")

#print("\nRegression Plot:")
sns.regplot(X, Y, data=df1, color="blue", label="A")
sns.regplot(X, Yhat, data=df1, color="red", label="B")
plt.legend(labels=['Actual values', 'Fitted values'])
plt.xlabel('Confirmed cases')
plt.ylabel('Weeks of 2020')
plt.title("Regression Plot\nCOVID-19 outbreak in India")
plot2 = plt.savefig('Regression Analysis.png')
plt.show(plot2)

"""
#print("\nResidual Plot:")
sns.residplot(X, Y)
ax1 = sns.distplot(X, hist=False, color = 'r', label = "Actual value")
sns.distplot(Yhat, hist=False, color='b', label="Fitted value", ax=ax1)
plt.ylim(0, 0.02)
plt.title('Residual plot:')
"""

#-----------------------------------------------------------------------------
#mse and r_sq
print("\n")
from sklearn.metrics import mean_squared_error as mse
MSE = mse(X, Yhat)
print('\nMSE:\n',MSE)
R_sq = lm.score(X, Yhat)
print('\nR_squared:\n',R_sq)
#-------------------------------------------------------------------------
#train_test
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(X, Y, test_size=0.4,
                                       random_state=0)
#Cross-validation
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import cross_val_predict as cvp
cv_s = cvs(lm, X, Yhat, cv=3)
Mean_score1 = np.mean(cv_s)
print("\nMean score for Cross validation score:\n",Mean_score1)
cv_p = cvp(lm, X, Yhat, cv=3)
Mean_score2 = np.mean(cv_p)
print("\nMean score for Cross validation predict:\n",Mean_score2)

#predictingTheDecision
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(df1[['Confirmed']], df1[['Date-parsed']]) 
prediction = clf.predict(df1[['Confirmed']])
print(prediction)


df2 = pd.DataFrame({
        'Confirmed':prediction,
        'Weeks of the year':df1[['Date-parsed']]
})
df2.to_csv('csv_to_submit.csv', index = False)