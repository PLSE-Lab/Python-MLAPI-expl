import numpy as np
import pandas as pd




# Data cleanup
# TRAIN DATA
train_df = pd.read_csv("../input/train.csv")
#train_df.Sex similar to saying train_df['Sex']
# Convert geneder to 0 and 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
# All missing Embarked -> just make them embark from most common place

bb= pd.read_csv("../input/train.csv")



ne=np.where(bb['Embarked'] != train_df['Embarked'])
print("hey")
print (ne[0])


#if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
#    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values


#bb.Embarked.fillna(bb.Embarked.mode().iloc[0])
#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
print("finish")