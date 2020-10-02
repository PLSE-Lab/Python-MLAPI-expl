import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
#print(train)

train = train[train["Survived"] < 1]

#df = df[df['Correlation'] >= 0]


train = train["Age"].fillna(train["Age"].mean())

ones = 0
tens = 0
twenties = 0
thirties = 0
forties = 0
fifties = 0
old = 0

for value in train:
    if value < 11:
        ones+=1
    elif value < 21:
        tens+=1
    elif value < 31:
        twenties+=1
    elif value < 41:
        thirties+=1
    elif value < 51:
        forties+=1
    elif value < 61:
        fifties+=1
    else:
        old+=1
        
print("1-10: " + str(ones))
print("11-20: " + str(tens))
print("21-30: " + str(twenties))
print("31-40: " + str(thirties))
print("41-50: " + str(forties))
print("51-60: " + str(fifties))
print("61+: " + str(old))