# IMPORTING THE DATA SET #
import pandas as pd

dataset = pd.read_csv('../input/wine-dataset/wine_dataset.csv')

dataset['style'] = dataset['style'].replace('red',0)
dataset['style'] = dataset['style'].replace('white',1)

# SEPARATING THE VARIABLES IN PREDICTORS AND TARGET VARIABLES #

t = dataset['style']
p = dataset.drop('style', axis = 1)

# CREATING THE TRAINING AN TEST DATA SET #

from sklearn.model_selection import train_test_split

t_train, t_test, p_train, p_test = train_test_split(t, p, test_size = 0.3)

# MODEL CREATION #

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(p_train, t_train) 

# PRINTING THE RESULTS #

result = (model.score(p_test, t_test))*100
print(f'Performance:  {result}' '%')


