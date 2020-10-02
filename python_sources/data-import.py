# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # import the plot/graph package, !but can not display in kaggle
from sklearn import linear_model 
import statsmodels.api as sm



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))



def SelectPanCol(Col_Name,DF):
# this function will return the columns of a pandas data frame
    return DF[Col_Name]
    


#import train data set to dataframe
train_df=pd.read_csv('../input/train.csv')
#import test data set to data frame
test_df=pd.read_csv('../input/test.csv')


"""below part is for data visilization"""


age_col=train_df[np.isfinite(train_df['Age'])][['Survived','Age']]
#print(age_col)
plt.hist(age_col['Age'],bins=np.arange(0,100,10))
plt.show()

"""Below is the linear regression part"""
regressor=np.vstack([age_col['Age'],np.ones(len(age_col['Age']))]).T

#print(regressor)

result=np.linalg.lstsq(regressor,age_col['Survived'])
print("the coeffient for X and 1 are \n",result[0])


"""now using sklinear"""
regr=linear_model.LinearRegression()
sk_regressor=SelectPanCol(['Age','Survived'],train_df)
sk_regressor=sk_regressor.dropna()

#print(sk_regressor)
X=sk_regressor['Age'].values.reshape(-1,1)
Y=sk_regressor['Survived']
regr.fit(X,Y)
print("Coefficients are:\n",regr.coef_)
print('Variance score: %.4f' % regr.score(X, Y))


"""using stats model package"""
X_with_Constant=sm.add_constant(X)
est=sm.OLS(Y,X_with_Constant)
est_fit=est.fit()
#print(est_fit.summary())


train_df_clean=train_df.dropna()
X1=train_df_clean[['Age','Fare']]
Y1=train_df_clean['Survived']


X1_with_Constant=sm.add_constant(X1)
est1=sm.OLS(Y1,X1_with_Constant)
est_fit1=est1.fit()
print(est_fit1.summary())

#print(X1)


#Y=train_df_




#print(X_with_Constant)

if __name__=="__main__":
    #print(SelectPanCol(['Age','Survived'],train_df)) #test for function
    pass

#print(linear_result)


#print(train_df["Age"])







