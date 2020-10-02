import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.info())

print(train['PassengerId'][:1])

number_passengers = np.size(train['PassengerId'][:1].astype(np.float))

number_survived = np.sum(train['PassengerId'][:1].astype(np.float))
proportion_survivors = number_survived / number_passengers 

is_women = train['Sex'] == "female"
is_men = train['Sex'] != "female"

output=test.ix[:,['PassengerId']]
print(output.info())
output['Survived']= (test['Sex'].map(lambda x: 1 if x=='female' else 0))
print('output-------')
print(output.head())

print("\n\nSummary statistics of training data")
print(output.describe())

#Any files you save will be available in the output tab below
output.to_csv('output.csv', index=False)


women_onboard = train[is_women]['Sex'].value_counts()
men_onboard = train[is_men]['Sex'].value_counts()

print(women_onboard)

# and derive some statistics about them
proportion_women_survived = women_onboard.values[0]/ len(train)
proportion_men_survived = men_onboard / len(train)

print ('Proportion of women who survived is %s' % proportion_women_survived)
print ('Proportion of men who survived is %s' % proportion_men_survived)

# Now that I have my indicator that women were much more likely to survive,