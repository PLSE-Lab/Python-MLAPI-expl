# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#importing the essential libraries 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

#importing dataset
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col=0) 
#independent or explanatory variables 
feature_cols=['GRE Score','TOEFL Score','LOR ','CGPA','Research']
X=np.array(df[feature_cols],dtype=np.float64)
#dependent or response variable
y=np.array(df['Chance of Admit '],dtype=np.float64)
#cross validation; splitting dataset into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
#Linear Regression model
lr=LinearRegression(n_jobs=-1)
#fitting the model
lr.fit(X_train,y_train,sample_weight=None)
y_pred=lr.predict(X_test) #predicting using the test dataset
accuracy=lr.score(X_test,y_test) #R squared shows how the data fit the model 
#pred_x=np.array([337,118,4,4.5,4.5,9.65,1]) //array numbers for prediction on test dataset
#pred_x=pred_x.reshape(1,-1)
#pred_y=lr.predict(pred_x)
#print(pred_y)
print("R^2 or Accuracy : " , round(accuracy,2))
#print(X_test[:5])
#print(y_pred[:5])

#inputting of independent variables to show whether you can be admitted or not
v1 = input('Please enter your GRE score : ')
v2 = input('Please enter your TOEFL score : ')
v3 = input('Please enter your LOR : ')
v4 = input('Please enter your CGPA : ')
v5 = input('Please enter "1" if you did a research or "0" if you did not do a research : ')
pred_n=np.array([v1,v2,v3,v4,v5],dtype=np.float64)
pred_n=pred_n.reshape(1,-1)
pred_ad=lr.predict(pred_n)
print('============================================================================== \n ===Records entered===')
print(' GRE Score : ', v1, '\n TOEFL Score : ', v2, '\n LOR : ', v3, '\n CGPA : ', v4)
if(v5=="1"):
    print (' Research : Yes')
elif(v5=="0"):
    print(' Research : No')
elif(v5>"1"):
    print(' Research : Invalid input')
else:
    print(' Research : Invalid input')

pre_ad_round=np.round(np.array(pred_ad*100),2)#round-off the result
predict_admissions = str(pre_ad_round)[1:-1]#remove the square bracket
print(' Your chance of admission is', predict_admissions,'%')
if(pre_ad_round > 55.5):
    print (" Congratulations!!! You have the chance of admission")
else :
    print (" Whooops!!! You did not reach the pass mark, sorry more room for improvement!")
print(" ==========\n Thank you\n ==========")
