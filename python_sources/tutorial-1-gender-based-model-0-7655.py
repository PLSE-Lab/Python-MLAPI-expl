

# Tutorial 1: ADOPTED FROM Gender Based Model (0.76555) - by Myles O'Neill
#https://www.kaggle.com/mylesoneill/titanic/tutorial-part-1-naive-gender-prediction
import numpy as np
import pandas as pd
import pylab as plt

# (1) Import the Data into the Script
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# (2) Create the submission file with passengerIDs from the test file
submission_naive = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})


# (3) Assume that everyone died. 
submission_naive.Survived = 0


# (4) Create a new submission based on gender. 
submission = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pd.Series(dtype='int32')})

# (5) Fill the Data for the survived column, all females live (1) all males die (0)
submission.Survived = [1 if x == 'female' else 0 for x in test['Sex']]


# (6) Create final submission file
submission_naive.to_csv("submission_naive.csv", index=False)
submission.to_csv("submission_gender.csv", index=False)

# (7) Confusion Matrix (TBD)



# (8) A Nice Visual showing Survival of Men and Women to justify this approach


female_stats = train[train['Sex'] == 'female']
females_alive = np.sum(female_stats['Survived'] == 1)
females_dead = len(female_stats.index) - females_alive
female_survival = pd.Series((females_alive, females_dead), name='')

female_chart = female_survival.plot(kind='pie', title='Female Survival', labels=['Survived','Died'], colors=['#21BFFF','#90DFFF'], figsize=(4,4), autopct='%1.1f%%')
female_chart.set_aspect('equal')
fig_female = female_chart.get_figure()
fig_female.savefig('x1_female_survival.png')

plt.clf()

male_stats = train[train['Sex'] == 'male']
males_alive = np.sum(male_stats['Survived'] == 1)
males_dead = len(male_stats.index) - males_alive
male_survival = pd.Series((males_alive, males_dead), name='')

male_chart = male_survival.plot(kind='pie', title='Male Survival', labels=['Survived','Died'], colors=['#21BFFF','#90DFFF'], figsize=(4,4), autopct='%1.1f%%')
male_chart.set_aspect('equal')
fig_male = male_chart.get_figure()

fig_male.savefig('x2_male_survival.png')

#1P. Calculate and print a value for the accracy for both the Naive and Gender models.  
#2P 



