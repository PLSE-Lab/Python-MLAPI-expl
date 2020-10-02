import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

fname='/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv'
dtset=pd.read_csv(fname)

msk=np.random.rand(len(dtset))<0.8
trn=dtset[msk]
tst=dtset[~msk]

regr=linear_model.LinearRegression()
x=np.asanyarray(trn[['YearsExperience']])
y=np.asanyarray(trn[['Salary']])
regr.fit(x,y)

plt.scatter(trn.YearsExperience, trn.Salary,  color='blue')
plt.plot(x, regr.coef_[0][0]*x + regr.intercept_[0], '-r')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

y_ht=regr.predict(tst[['YearsExperience']])
x_h=np.asanyarray(tst[['YearsExperience']])
y_h=np.asanyarray(tst[['Salary']])

print("R2-score: %.2f" % regr.score(x_h ,y_h) )

