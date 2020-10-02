import pandas as pd
test = pd.read_csv('../input/test.csv', index_col='PassengerId')
test['Survived']=0
test.loc[((test['Sex']=='female') & (test['Age']<64)) | (test['Age']<8) | test['Name'].str.contains('Master.'), 'Survived'] = 1
test[['Survived']].to_csv('simple.csv')