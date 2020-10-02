import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())



#train_small = train.loc[:,['Name','SibSp','Parch']]
train_small = (train.loc[:,['Name','SibSp','Age','Parch']]).query("Parch > 0 or SibSp > 0")
train_small_sort = train_small.sort(['Name'])
#surname_list = []
#for text in train_small['Name']:
#    surname_list.append(text)#.split(',', 2)[0])
#surname_list.sort()    
#print(surname_list)
    
print (train_small_sort.head(200))
#print ((train.loc[:,['Name','SibSp','Parch']])[train.Parch > 0].head(30))
print(train.info())

#Any files you save will be available in the output tab below
train_small_sort.to_csv('copy_of_the_training_data.csv', index=False)