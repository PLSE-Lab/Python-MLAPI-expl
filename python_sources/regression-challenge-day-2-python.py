# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Note that the Kaggle data seems to be in latin-1 encoding
#kaggle = read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding='iso-8859-1')
stackoverflow = pd.read_csv("../input/survey_results_public.csv", encoding='utf8')
#print(stackoverflow.head())
# lets check for correlation between continous data
# correlation between numerical variables is something like this
# if we increase one variable, there is a siginficant almost increase/decrease
# in the other variable. it varies from -1 to 1
'''
continous_train = stackoverflow.dtypes[stackoverflow.dtypes != "object"].index
correlation_train = stackoverflow[continous_train].corr()


print(correlation_train['JobSatisfaction'].sort_values(ascending=False))
'''

JobSatisfaction=stackoverflow[['JobSatisfaction']].copy()
#Salary_Exp=stackoverflow[['Salary','ExpectedSalary']].copy()
Satisfaction=stackoverflow[['JobSatisfaction','CareerSatisfaction']].copy()
CareerSatisfaction=stackoverflow[['CareerSatisfaction']].copy()
print(CareerSatisfaction.describe())
print(CareerSatisfaction.head())
print(JobSatisfaction.describe())
print(JobSatisfaction.head())

#Salary=Salary.replace(to_replace='0',value=np.nan)
#Salary_Exp=Salary_Exp.dropna(how='any')
Satisfaction=Satisfaction.dropna(axis=0)
print(Satisfaction.head())
#Salary_Exp=Salary_Exp[Salary_Exp['Salary']<1]
print(Satisfaction.head())
#Salary['Salary'].fillna(Salary['Salary'].mean(),inplace=True)
#Salary_Exp['YearsProgram'].fillna(Salary_Exp['YearsProgram'].mean(),inplace=True)
#ExpectedSalary = stackoverflow['ExpectedSalary']
#ExpectedSalary.fillna(0, inplace=True)

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
x = Satisfaction[['JobSatisfaction']].copy()
y = Satisfaction[['CareerSatisfaction']].copy()
reg = LinearRegression()
reg.fit(x,y)
y_pred = reg.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='blue', linewidth=3)
plt.savefig('resule.png')
