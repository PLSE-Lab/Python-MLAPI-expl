import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv("../input/harvard-course-enrollment-fall-2015.csv")

print( df.shape )
#print data_csv['CLASSNBR'].value_counts().shape

# cleaning columns
df['TOTALENROLLMENT'] = df['TOTALENROLLMENT'].apply(lambda row: 0 if pd.isnull(row) else row)
df['HCOL'] = df['HCOL'].apply(lambda row: 0 if pd.isnull(row) else row)
df['GSAS'] = df['GSAS'].apply(lambda row: 0 if pd.isnull(row) else row)
df['NONDGR'] = df['NONDGR'].apply(lambda row: 0 if pd.isnull(row) else row)
df['VUS'] = df['VUS'].apply(lambda row: 0 if pd.isnull(row) else row)
df['XREG'] = df['XREG'].apply(lambda row: 0 if pd.isnull(row) else row)

#remove unused columns
df = df.drop(['COURSE', 'DEPARTMENT', 'COURSEID'], axis=1)

print ( df.head() )
print ( df.isnull().any() ) # is there some null?
#print df['COURSEID'].value_counts().shape
# x
features = ['GSAS','HCOL','NONDGR','VUS','XREG']
predictors = df[features]

# y
targets = df['TOTALENROLLMENT']

X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size=.3)

# LinearRegression 
model = LinearRegression()
model.fit(X_train, y_train)

print ("Score LinearRegression :", model.score(X_test, y_test) )
print ("Predict ", model.predict(X_test.iloc[:5]) )
print ("Result for Prediction ", y_test.iloc[:5] )


# Ridge 
model1 = Ridge()
model1.fit(X_train, y_train)

print ("Score Ridge :", model1.score(X_test, y_test) )
print ("Predict ", model1.predict(X_test.iloc[:5]) )
print ("Result for Prediction ", y_test.iloc[:5] )


